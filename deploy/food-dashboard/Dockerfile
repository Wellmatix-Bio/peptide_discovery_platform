# The food-to-product dashboard: one static page served by nginx. The API address is not baked in: the container writes config.js from WMX_API_BASE_URL when it starts.
FROM nginx:1.31-alpine
COPY index.html /usr/share/nginx/html/index.html
COPY --chmod=755 write-config.sh /docker-entrypoint.d/40-write-config.sh
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
