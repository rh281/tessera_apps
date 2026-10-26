# The hosted demo page (demo/app.py) as a container image. Build it from the repository root, so the
# lock and the package source are in the build context next to the page:
#
#   docker build -f demo/Dockerfile -t tessera-apps-demo .
#   docker run --rm -p 8501:8501 tessera-apps-demo
#
# then open http://localhost:8501. The page reads the anonymous store, so no AWS credentials are
# needed to build or run the image, and the image holds none. demo/README.md covers deploying it.
#
# Needs BuildKit with Dockerfile frontend 1.6 or later (Docker Engine 25 or later): the legacy
# builder cannot build it because of RUN --mount, and older BuildKit rejects
# HEALTHCHECK --start-interval.
#
# Behind a TLS-intercepting proxy (most machines are not), give the proxy and its CA bundle to the
# build only: the predefined proxy build arguments are not recorded in the image history, and the CA
# goes in as the BuildKit secret proxy_ca, which is mounted for the install steps and never written to
# a layer:
#
#   docker build --network host --build-arg HTTPS_PROXY --build-arg NO_PROXY \
#     --secret id=proxy_ca,src=/path/to/ca-bundle.crt -f demo/Dockerfile -t tessera-apps-demo .
#
# The apt-get step also needs a Debian mirror; Debian's sources are plain HTTP, so a proxy for them
# goes in --build-arg http_proxy.

# python:3.12-slim (Debian), pinned by the digest of its multi-platform index, so every build starts
# from the same base until the digest is changed here on purpose.
ARG BASE=python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

# ---- runtime: the base image and the one system library the wheels do not bring ----
FROM ${BASE} AS runtime

# rasterio's wheel bundles GDAL but links it to the system's libexpat.so.1, which the slim image does
# not have: without it, importing rasterio fails with "libexpat.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# ---- build: install everything into a virtual environment, which the final stage copies whole ----
FROM ${BASE} AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN python -m venv /opt/venv

WORKDIR /src

# 1. The package's tested dependencies, exactly as pinned in the lock (it also pins the test and
#    notebook tools, which the page does not use). Wheels only: the slim image has no compiler, and a
#    package without a wheel for this platform should stop the build here, not compile.
COPY requirements-lock.txt ./
RUN --mount=type=secret,id=proxy_ca,required=false \
    if [ -s /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    /opt/venv/bin/pip install --only-binary=:all: -r requirements-lock.txt

# 2. tessera_apps itself (0.2.0, not yet on PyPI) from the source, resolving nothing again.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=type=secret,id=proxy_ca,required=false \
    if [ -s /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    /opt/venv/bin/pip install --no-deps .

# 3. The page's own requirements (Streamlit and streamlit-folium, pinned); its tessera_apps==0.2.0
#    line is already satisfied by step 2. It keeps its path, demo/requirements.txt, so its
#    "-c ../requirements-lock.txt" line finds the lock copied in step 1.
COPY demo/requirements.txt ./demo/requirements.txt
RUN --mount=type=secret,id=proxy_ca,required=false \
    if [ -s /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    /opt/venv/bin/pip install --only-binary=:all: -r demo/requirements.txt \
    && /opt/venv/bin/pip check

# ---- final: the virtual environment and the page, run by a user without root ----
FROM runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PORT=8501

# A numeric uid, so a platform that checks for a non-root user (Kubernetes runAsNonRoot) can tell.
# The home directory is there in case a library keeps per-user files; the page itself writes nothing.
RUN useradd --create-home --uid 10001 --user-group --shell /usr/sbin/nologin app

COPY --from=build /opt/venv /opt/venv

# The page, its click logic and the precomputed demos, owned by root and read-only to the app user.
WORKDIR /app
COPY demo/app.py demo/trajectory.py demo/
COPY demo/precomputed demo/precomputed

USER 10001:10001

# Stop the build, rather than the first visit, if anything the page imports cannot load.
RUN python -c "import rasterio, pyogrio, streamlit, streamlit_folium, tessera_apps.apps.wildfire, demo.trajectory"

EXPOSE 8501

# Streamlit's own health endpoint answers "ok" once the server is up. Python, because the slim image
# has no curl; the opener ignores any proxy settings, since the request never leaves the container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:' + (os.environ.get('PORT') or '8501') + '/_stcore/health', timeout=4).read()"]

# A shell, only to expand ${PORT} (some hosts set it; 8501 otherwise); exec then
# makes Streamlit the container's main process, so it receives the stop signal itself. The file
# watcher is off because the page's files never change inside the image.
CMD ["sh", "-c", "exec streamlit run demo/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false"]
