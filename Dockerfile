# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS base-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV SFUMATO_HOME=/opt/sfumato
ENV SFUMATO_CONFIG=/data/config.toml
ENV SFUMATO_DATA_DIR=/data
ENV RIJKSMUSEUM_API_KEY=

WORKDIR ${SFUMATO_HOME}

FROM base-runtime AS builder

RUN python -m pip install --upgrade pip build

COPY pyproject.toml README.md ./
COPY src ./src
COPY templates ./templates

RUN python -m build --wheel

FROM base-runtime AS runtime

RUN python -m pip install --upgrade pip

COPY --from=builder /opt/sfumato/dist/*.whl /tmp/dist/
RUN python -m pip install /tmp/dist/*.whl && \
    python -m playwright install --with-deps chromium && \
    rm -rf /tmp/dist

COPY src ./src
COPY templates ./templates
COPY docker ./docker

ENV PYTHONPATH=/opt/sfumato/src

RUN mkdir -p /data/paintings /data/state /data/output && \
    chmod +x /opt/sfumato/docker/entrypoint.sh

VOLUME ["/data"]

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD ["python", "/opt/sfumato/docker/healthcheck.py"]

ENTRYPOINT ["/opt/sfumato/docker/entrypoint.sh"]
CMD ["sfumato", "watch", "--config", "/data/config.toml"]
