# Deployment

Three containers: `web` (nginx serving the React build and proxying), `accounts` (auth and the
authenticating proxy), `api` (the job API). The GPU pipeline is **not** here — it runs as a
Vertex AI Custom Job from an image built separately, so none of these containers needs a GPU.

```
browser ──► web (nginx, the ONLY published port)
              ├─ /auth/*    ─► accounts
              └─ /api/peptide/*  ─► accounts ─► api ──► Vertex AI + Cloud Storage
```

**The job API has no authentication and gains none.** It is safe only because it publishes no
host port and the accounts service is the only route to it. *Protected by the frontend is not
protection.* `services/accounts/tests/test_deploy.py` fails if a port is added to `api` or
`accounts`, if the signing key gets a default, if a container runs as root, or if nginx appends
rather than overwrites `X-Forwarded-For`.

---

## Run it locally

```bash
cp deploy/.env.example deploy/.env
```

Fill it in, then:

```bash
docker compose -f deploy/compose.yaml --env-file deploy/.env up -d --build
```

The app is on `http://127.0.0.1:8088` by default.

**Check the port is free first.** `WEB_PORT` defaults to 8088 and the job API's own default is
8080; on at least one developer machine here both were already taken (by another application and
by Open WebUI respectively). A collision on 8080 is the worse one: the accounts proxy forwards to
whatever answers, so pointing `PEPTIDE_UPSTREAM` at the wrong service makes the app fail in
confusing ways rather than obviously.

```bash
ss -ltn | grep -E ':(8080|8088)'
```

Stop it with:

```bash
docker compose -f deploy/compose.yaml --env-file deploy/.env down
```

---

## Credentials

The API uses Application Default Credentials.

- **On a GCP VM** created with `--scopes=cloud-platform`, leave `GOOGLE_CREDENTIALS_FILE` and
  `GOOGLE_APPLICATION_CREDENTIALS` empty. The metadata server supplies them and nothing is mounted.
- **Off GCP**, point `GOOGLE_CREDENTIALS_FILE` at a service-account key on the host. Compose mounts
  it read-only at `/secrets/credentials.json`, and `GOOGLE_APPLICATION_CREDENTIALS` must name that
  path.

  **The file must be readable by uid 10001**, which is the non-root user the API container runs
  as. A bind mount keeps the host's ownership and permissions, so a key written by `gcloud` —
  mode `0600`, owned by your own uid — is *not* readable inside the container, and the API answers
  **500** on every call that touches Google. Nothing says why; this was hit while testing.

  Give it to that uid explicitly rather than making a credentials file world-readable:

  ```bash
  sudo install -o 10001 -g 10001 -m 0400 /path/to/key.json /etc/peptide/credentials.json
  ```

  then set `GOOGLE_CREDENTIALS_FILE=/etc/peptide/credentials.json`. On a GCP VM none of this
  applies — leave both variables empty and the metadata server handles it.

The service account needs to create and cancel Vertex Custom Jobs, and to read and write the
`VERTEX_ARTIFACTS_DIR` and `VERTEX_MODEL_STORE` prefixes.

---

## One instance, and why

The accounts database is SQLite. Several uvicorn workers inside one container are fine — every
piece of shared state (users, captcha nonces, ownership, history, rate-limit counters) is in that
database, with WAL and a busy timeout. **Several replicas are not.** `docker compose up --scale
accounts=2`, a second VM, or a load balancer across two hosts will split or corrupt the data.

---

## Deploying to a GCP VM

Not executed here — this is the runbook to follow.

```bash
gcloud config set project <GCP_PROJECT>
```

```bash
gcloud compute instances create <NAME> --zone=<ZONE> --machine-type=e2-standard-2 --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=50GB --tags=web --scopes=cloud-platform
```

```bash
gcloud compute firewall-rules create <NAME>-web --allow=tcp:80,tcp:443 --target-tags=web
```

`e2-standard-2` is sized for a control plane, not for inference: these containers submit jobs and
read artifacts, and the GPU work happens on Vertex.

### The signing key belongs in Secret Manager

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(48))' | gcloud secrets create accounts-secret-key --data-file=-
```

Grant the VM's service account `roles/secretmanager.secretAccessor`, then on the VM:

```bash
echo "ACCOUNTS_SECRET_KEY=$(gcloud secrets versions access latest --secret=accounts-secret-key)" >> deploy/.env
```

Changing this key signs every user out.

### On the VM

Install Docker Engine and the compose plugin, clone the repository with a **read-only deploy key**,
write `deploy/.env` (including `WEB_BIND=127.0.0.1` and `WEB_PORT=8088`), then:

```bash
docker compose -f deploy/compose.yaml --env-file deploy/.env up -d --build
```

### TLS

Put Caddy on the VM in front of `web` (`reverse_proxy 127.0.0.1:8088`, automatic certificates),
and point the domain's A record at a static IP from `gcloud compute addresses create`.

**Then fix the client address.** Once Caddy is in front, nginx sees Caddy's address for every
request, so the accounts service's per-address rate limits collapse into one shared allowance.
Configure nginx's real-IP module for wherever Caddy runs before relying on them:

```
set_real_ip_from 127.0.0.1;
real_ip_header X-Forwarded-For;
```

A Google HTTPS load balancer works too, with the same rule applied to its ranges.

### Backups

The named volume lives on the boot disk. Snapshot it:

```bash
gcloud compute resource-policies create snapshot-schedule daily-backup --region=<REGION> --daily-schedule --start-time=03:00 --max-retention-days=14
```

```bash
gcloud compute disks add-resource-policies <NAME> --zone=<ZONE> --resource-policies=daily-backup
```

### Updates

`git pull`, then `up -d --build`. The Models & health page confirms what is serving.

---

## Before this is reachable from the internet

**Set a hard billing cap on the Vertex project.** Registration is open and there is no per-user
quota (decisions 0.2 and 0.3 in `docs/DECISIONS.md`): anyone who can reach the app and register
can start L4 GPU jobs that bill to the project. This was accepted deliberately, not overlooked,
and a budget cap is the control that makes it survivable.

Also worth knowing before you expose it:

- **The captcha is text in an SVG and a parser solves it.** It stops drive-by bots, nothing more.
- **Signing out revokes nothing server-side.** Sessions are stateless signed tokens. Changing a
  password bumps a version counter embedded in every token, which is the only real revocation.
- **A run reports `status: "pending"` forever if its worker dies** without writing `results.json`.
  The UI reads `vertex_state` to tell the truth about this; anything else consuming the API
  directly needs to do the same.
