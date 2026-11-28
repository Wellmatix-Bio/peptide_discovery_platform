#!/bin/sh
# Runs when the container starts (nginx image's /docker-entrypoint.d): the page reads its API address from config.js beside it.
set -eu
printf 'window.WMX_CONFIG = { apiBase: "%s" };\n' "${WMX_API_BASE_URL:-http://localhost:8000}" > /usr/share/nginx/html/config.js
