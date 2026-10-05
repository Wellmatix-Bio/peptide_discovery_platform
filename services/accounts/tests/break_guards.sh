#!/usr/bin/env bash
# §12: break each guard deliberately, confirm a test fails, restore.
# Exits non-zero if any sabotage goes UNDETECTED.
set -u
cd /home/ajit/Documents/peptide_discovery_platform
A=services/accounts/accounts
T=services/accounts/tests
PY=.venv/bin/python
BAK=$(mktemp -d)
cp -r $A "$BAK/accounts"
cp -r $T "$BAK/tests"
restore() { rm -rf $A $T; cp -r "$BAK/accounts" $A; cp -r "$BAK/tests" $T; find services/accounts -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; }
FAILED_TO_CATCH=0

check() { # name, expected-failing-test-selector
  local name="$1"; shift
  local sel="$1"; shift
  out=$($PY -m pytest -q "$sel" 2>&1 | tail -1)
  if echo "$out" | grep -qEi "[0-9]+ failed|error"; then
    printf '  CAUGHT   %-52s %s\n' "$name" "$out"
  else
    printf '  MISSED!! %-52s %s\n' "$name" "$out"
    FAILED_TO_CATCH=1
  fi
  restore
}

echo "== breaking guards in the accounts service =="

# 1. Ownership check removed entirely.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/proxy.py"); s = p.read_text()
s = s.replace("""    for job_id in sorted(_job_ids(path, query)):
        if not accounts.db.owns(user.id, job_id):
            raise HTTPException(404, NOT_YOURS.format(job_id=job_id))""", "    pass")
p.write_text(s)
EOF
check "ownership check removed" "$T/test_proxy_peptide.py"

# 2. Ownership checks path but not query values.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/proxy.py"); s = p.read_text()
s = s.replace("""    found = set(JOB_ID.findall(path))
    for _, value in parse_qsl(query, keep_blank_values=True):
        found |= set(JOB_ID.findall(value))
    return found""", "    return set(JOB_ID.findall(path))")
p.write_text(s)
EOF
check "query-string job ids unchecked" "$T/test_proxy_peptide.py::test_a_job_id_in_a_query_value_is_checked_too"

# 3. request_id passed through as the user typed it.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/proxy.py"); s = p.read_text()
s = s.replace("    generated = new_request_id(user_id)", "    generated = typed")
p.write_text(s)
EOF
check "request_id not replaced (traversal reaches API)" "$T/test_proxy_peptide.py"

# 4. request_id generated without the per-user namespace.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/proxy.py"); s = p.read_text()
s = s.replace('    return f"{namespace(user_id)}{uuid.uuid4().hex}"', '    return uuid.uuid4().hex')
p.write_text(s)
EOF
check "namespace prefix dropped from request_id" "$T/test_proxy_peptide.py::test_the_request_id_sent_upstream_is_generated_not_the_users"

# 5. Response un-namespacing turned off.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/proxy.py"); s = p.read_text()
s = s.replace("    content = _unnamespaced(content, response_type, user.id)", "    pass")
p.write_text(s)
EOF
check "namespace not stripped from responses" "$T/test_proxy_peptide.py::test_the_users_own_namespace_prefix_is_stripped_from_the_response"

# 6. Recorder failures propagate to the caller.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/history.py"); s = p.read_text()
s = s.replace("""    except Exception:  # noqa: BLE001 - recording must never reach the response
        log.exception("history: failed to record an observed response; the response is unaffected")""",
"""    except Exception:
        raise""")
p.write_text(s)
EOF
check "recorder failure reaches the response" "$T/test_proxy_peptide.py::test_a_recorder_failure_cannot_change_the_response"

# 7. A status poll overwrites a stored results body instead of coalescing.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/db.py"); s = p.read_text()
s = s.replace(" envelope = COALESCE(?, envelope),", " envelope = ?,")
p.write_text(s)
EOF
check "status poll wipes the stored results body" "$T/test_proxy_peptide.py::test_a_later_status_poll_does_not_wipe_a_stored_results_body"

# 8. The dead-run warning removed from the summary.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/history.py"); s = p.read_text()
s = s.replace('''    if status == "pending" and vertex_state in TERMINAL_VERTEX_STATES:
        summary += " -- reported pending, but Vertex has finished; the worker wrote no result"''', "    pass")
p.write_text(s)
EOF
check "dead run summarised as merely pending" "$T/test_proxy_peptide.py::test_a_run_vertex_has_failed_is_not_summarised_as_merely_pending"

# 9. Hiding a run also releases its ownership.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/db.py"); s = p.read_text()
s = s.replace('''                "DELETE FROM history WHERE id = ? AND user_id = ?", (entry_id, user_id)
            )''', '''                "DELETE FROM history WHERE id = ? AND user_id = ?", (entry_id, user_id)
            )
            db.execute("DELETE FROM owned WHERE user_id = ?", (user_id,))''')
p.write_text(s)
EOF
check "hiding a run releases ownership" "$T/test_proxy_peptide.py::test_hiding_a_run_does_not_release_its_ownership"

# 10. A route registered on the ungated public router.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/proxy.py"); s = p.read_text()
s = s.replace("router = APIRouter(dependencies=[Depends(current_user)], tags=[\"signed in\"])",
              "router = APIRouter(tags=[\"signed in\"])")
p.write_text(s)
EOF
check "proxy router registered without the gate" "$T/test_proxy_peptide.py::test_a_new_route_on_a_gated_router_is_refused_without_asking_for_it"

# 10b. The auth router's own gate removed.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/auth.py"); s = p.read_text()
before = s
s = s.replace('gated = APIRouter(dependencies=[Depends(current_user)], tags=["signed in"])', 'gated = APIRouter(tags=["signed in"])')
assert s != before, "sabotage 10b did not apply -- the gated router declaration moved"
p.write_text(s)
EOF
check "auth gated router registered without the gate" "$T/test_proxy_peptide.py::test_a_new_route_on_a_gated_router_is_refused_without_asking_for_it"

# 11. The route walker stops at the _IncludedRouter nodes (the FastAPI 0.142 trap).
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/tests/conftest.py"); s = p.read_text()
s = s.replace("""            inner = getattr(route, "original_router", None)
            if inner is not None:
                walk(inner)
            elif hasattr(route, "path"):""", """            if hasattr(route, "path"):""")
p.write_text(s)
EOF
check "route walker not descending into included routers" "$T/test_proxy_peptide.py::test_every_route_is_either_public_or_gated"

# 12. Authorization forwarded upstream.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/proxy.py"); s = p.read_text()
s = s.replace('REQUEST_HEADERS = frozenset({"content-type", "accept", "accept-language"})',
              'REQUEST_HEADERS = frozenset({"content-type", "accept", "accept-language", "authorization", "cookie"})')
p.write_text(s)
EOF
check "Authorization and Cookie forwarded upstream" "$T/test_passthrough.py::test_authorization_cookie_and_host_are_never_forwarded"

# 13. Upstream docs forwarded.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/proxy.py"); s = p.read_text()
s = s.replace('    "peptide": ("", "docs", "redoc", "openapi.json"),', '    "peptide": (),')
p.write_text(s)
EOF
check "upstream docs and root forwarded" "$T/test_passthrough.py"

# 14. Equal-work on an unknown email removed (login timing oracle).
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/auth.py"); s = p.read_text()
before = s
s = s.replace("        crypto.spend_equal_work(body.password)\n", "")
assert s != before, "sabotage 14 did not apply -- the equal-work call moved"
p.write_text(s)
EOF
check "equal-work on unknown email removed" "$T/test_auth.py"

# 15. password_version no longer compared (revocation broken).
$PY - <<'EOF'
import pathlib, re
p = pathlib.Path("services/accounts/accounts/auth.py"); s = p.read_text()
s = re.sub(r'([ \t]*)if .*password_version.*:\n(?:\1[ \t]+.*\n)+', r'\1pass\n', s, count=1)
p.write_text(s)
EOF
check "password_version not compared (no revocation)" "$T/test_auth.py"

# 16. Captcha marked spent only on a correct answer.
$PY - <<'EOF'
import pathlib
p = pathlib.Path("services/accounts/accounts/db.py"); s = p.read_text()
s = s.replace('"UPDATE captchas SET spent = 1 WHERE nonce = ? AND spent = 0 AND expires_at >= ?"',
              '"UPDATE captchas SET spent = 0 WHERE nonce = ? AND spent = 0 AND expires_at >= ?"')
p.write_text(s)
EOF
check "captcha not spent on a wrong answer" "$T/test_captcha.py"

restore
echo
if [ $FAILED_TO_CATCH -eq 0 ]; then
  echo "RESULT: every sabotage was caught."
else
  echo "RESULT: at least one sabotage went UNDETECTED -- see MISSED above."
fi
rm -rf "$BAK"
exit $FAILED_TO_CATCH
