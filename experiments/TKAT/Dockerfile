# Use official PyTorch image with CUDA (for GPU) or CPU base if needed
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Clone KAT repository
RUN git clone https://github.com/Adamdad/kat.git

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install numpy timm matplotlib

# Copy your forecasting script into the container
COPY forecast_200_steps.py /app/

# Set entrypoint
CMD ["python", "TKAT.py"]