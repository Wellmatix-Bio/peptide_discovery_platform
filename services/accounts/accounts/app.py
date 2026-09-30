"""The application factory. `uvicorn accounts.app:build --factory` from services/accounts.

Everything is built from a Settings object so a test serves the same code a deployment does.
`transports` lets a test point the passthrough at an in-process upstream app.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from . import __version__, proxy
from .auth import Context, gated, public
from .config import Settings, from_environment
from .db import Database

logging.basicConfig(level=logging.INFO)


def create_app(
    settings: Settings, transports: dict[str, httpx.AsyncBaseTransport] | None = None
) -> FastAPI:
    transports = transports or {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        clients = {
            name: httpx.AsyncClient(
                base_url=settings.upstreams[name],
                transport=transports.get(name),
                timeout=settings.upstream_timeout_seconds,
                # Upstream redirects are returned to the caller as they are, not followed here.
                follow_redirects=False,
            )
            for name in proxy.SERVICES
        }
        app.state.upstream_clients = clients
        try:
            yield
        finally:
            for client in clients.values():
                await client.aclose()

    app = FastAPI(
        title="accounts",
        version=__version__,
        description=(
            "Email/password accounts in front of the unauthenticated peptide job API."
            " Demo-scoped: to be deleted, not grown, when a platform identity system"
            " replaces it."
        ),
        # The proxy's own interactive docs would list every route including the gated ones;
        # nothing needs them in production.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.accounts = Context(
        db=Database(settings.database),
        key=settings.secret_key,
        token_ttl=settings.token_ttl_seconds,
        captcha_ttl=settings.captcha_ttl_seconds,
        client_ip_header=settings.client_ip_header,
        rate=settings.rate_policy,
    )
    app.include_router(public)
    app.include_router(gated)
    app.include_router(proxy.router)
    return app


def build() -> FastAPI:
    return create_app(from_environment())
