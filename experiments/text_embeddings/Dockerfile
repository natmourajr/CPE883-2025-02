ARG BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04

FROM $BASE_IMAGE

ARG EXPERIMENT

# Set timezone
ENV TZ=Etc/UTC

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

COPY . /app

WORKDIR /app/experiments/${EXPERIMENT}

# Setup UV and Python environment
RUN uv venv /venv
ENV UV_PROJECT_ENVIRONMENT=/venv
ENV UV_PYTHON=/venv/bin/python
ENV PATH=/venv/bin:$PATH

RUN if [ -f uv.lock ]; then uv sync --locked; else uv pip install -r requirements.txt; fi
