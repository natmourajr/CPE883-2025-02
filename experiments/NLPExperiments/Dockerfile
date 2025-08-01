ARG BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04

FROM $BASE_IMAGE

ARG EXPERIMENT

# Set timezone
ENV TZ=Etc/UTC

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

#COPY . /app
#RUN pip install -r /workspaces
#WORKDIR /app/experiments/${EXPERIMENT}
# Install build dependencies for packages that need compilation
RUN apt-get update && apt-get install -y build-essential gcc g++ 

# Setup UV and Python environment
#RUN apt install uv 
RUN uv python install 3.10
RUN uv venv -p 3.10 /venv
#RUN uv venv /venv
ENV UV_PROJECT_ENVIRONMENT=/venv
ENV UV_PYTHON=/venv/bin/python
ENV PATH=/venv/bin:$PATH
#RUN uv pip install -r /workspaces/CPE883-2025-02/models/katbert/requirements.txt

# Not sure if needed
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy requirements file into the image during build
COPY models/NLPModels/ /app/
COPY dataloaders/ /app/
RUN uv pip install -r /app/requirements.txt
RUN uv pip install --no-build-isolation -e /app/rational_kat_cu/
RUN uv pip install -e /app/NLPLoader/
RUN uv pip install -e /app/nanos/
RUN uv pip install -e /app/efficient-kan/

# RUN if [ -f uv.lock ]; then uv sync --locked; else uv pip install -r requirements.txt; fi
