# Build from the repository root:  docker build -f api/Dockerfile -t alexandria-api .
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: OpenMP runtime required by LightGBM (the second-stage ranker).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY core ./core
COPY api/pyproject.toml ./api/pyproject.toml
COPY api/app ./api/app
RUN pip install ./core ./api

RUN useradd --create-home --uid 10001 alexandria
USER alexandria
WORKDIR /home/alexandria

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# $PORT is injected by hosts like Render/Fly; defaults to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
