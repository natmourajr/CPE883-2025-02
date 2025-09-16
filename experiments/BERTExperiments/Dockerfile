FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies including CUDA toolkit
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    curl \
    gnupg2 \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA CUDA toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-4 && \
    rm cuda-keyring_1.0-1_all.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set environment variables to accept TOS automatically
ENV CONDA_ALWAYS_YES=true

# Update conda
RUN conda update -n base conda

# Copy the environment file
COPY models/ModernBERT/ /app/

# Create the conda environment using conda
RUN conda env create -f /app/environment.yaml

# Set up conda activation
RUN echo "conda activate bert24" >> ~/.bashrc

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "bert24", "/bin/bash", "-c"]

RUN pip install -e /app/efficient-kan/
RUN pip install -e /app/src/
# Install flash attention (requires CUDA)
RUN pip install "flash_attn==2.6.3" --no-build-isolation

# Set environment variables for the conda environment
ENV CONDA_DEFAULT_ENV=bert24
ENV PATH=/opt/conda/envs/bert24/bin:$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/bert24/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV NCCL_SHM_DISABLE=1
ENV NCCL_P2P_DISABLE=1

# Create entrypoint script
RUN echo '#!/bin/bash\nsource /opt/conda/etc/profile.d/conda.sh\nconda activate bert24\nexec "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]