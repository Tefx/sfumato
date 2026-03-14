# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS base-runtime
LABEL org.sfumato.contract="true"
LABEL org.sfumato.contract.stage="base-runtime"

FROM base-runtime AS builder
LABEL org.sfumato.contract="true"
LABEL org.sfumato.contract.stage="builder"

FROM base-runtime AS runtime
LABEL org.sfumato.contract="true"
LABEL org.sfumato.contract.stage="runtime"

ENV SFUMATO_CONFIG=/data/config.toml
ENV SFUMATO_DATA_DIR=/data
ENV RIJKSMUSEUM_API_KEY=

VOLUME ["/data"]

ENTRYPOINT ["python3", "-c", "raise SystemExit('Dockerfile contract stub only; see DEPLOYMENT_CONTRACT.md')"]
