"""The Vertex AI Custom Job worker, shared by every pipeline.

One implementation. `backend.worker_e2e.worker`, `backend.worker_metabolic.worker` and
`backend.worker_cosmetic.worker` are each a dozen lines that name their pipeline and call into
here, which is also what their Docker images still run -- so the image entry points, the
cloudbuild files and the tests all keep their existing module paths.
"""
