FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    unixodbc-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
RUN conda create -n chatkbqa python=3.8 -y
SHELL ["conda", "run", "-n", "chatkbqa", "/bin/bash", "-c"]

# Install PyTorch (CPU version for portability; replace cu117 for GPU)
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code (models and large data should be mounted via volumes)
COPY . .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
CMD ["conda", "run", "-n", "chatkbqa", "uvicorn", "src.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
