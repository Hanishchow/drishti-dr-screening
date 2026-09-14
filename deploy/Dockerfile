# Drishti API image.
#
# One image serves both roles. The district runs it against Postgres with a GPU
# runtime; a PHC edge node runs the same tag against SQLite on CPU. Keeping one
# image means the edge cannot silently drift to a different pipeline version
# than the district it syncs into.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# opencv needs these even in headless builds; curl is for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so application edits do not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core ./core
COPY dr ./dr
COPY sim ./sim
COPY server ./server
COPY web ./web
COPY README.md .

# Run unprivileged. A container that processes uploaded images should not be
# able to write its own code.
RUN useradd --create-home --uid 10001 drishti \
    && mkdir -p /app/storage /app/artifacts \
    && chown -R drishti:drishti /app/storage /app/artifacts
USER drishti

ENV DRISHTI_STORAGE_DIR=/app/storage \
    DRISHTI_MODEL_DIR=/app/artifacts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
