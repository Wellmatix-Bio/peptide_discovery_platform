"""Accounts, sessions and the authenticating proxy in front of the unauthenticated job API.

DEMO-SCOPED. This identity layer is to be deleted, not grown, when a platform identity system
replaces it. Before adding anything to it, ask whether it is worth building twice.

It imports none of the pipeline's code -- not `src/`, not `model_store/`. It reaches the job API
over HTTP only, so nothing here can drift with the pipeline's internals, and the backend gains no
dependency on this service.

THIS SERVICE IS WHAT MAKES "THE JOB API HAS NO AUTH" SAFE, not the web app in front of it. The
API publishes no host port and this is the only route to it. Anything reachable without a valid
session here is reachable by anyone.

Design and decision record: docs/DECISIONS.md. Measured backend baseline, including the API
behaviours this service has to work around: docs/BASELINE.md.
"""

__version__ = "0.1.0"
