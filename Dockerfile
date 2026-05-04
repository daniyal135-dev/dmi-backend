FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (including OpenCV libraries)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements — install order matters for image size:
# grad-cam / transformers pull torch from PyPI (CUDA ~2GB+) if torch is not already installed.
# Install CPU-only PyTorch FIRST, then the rest (excluding torch/torchvision lines).
COPY requirements.txt .

RUN pip install --default-timeout=1000 --retries 10 \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --default-timeout=1000 --retries 10 \
    $(grep -v "^torch" requirements.txt | grep -v "^torchvision" | grep -v "^#" | sed '/^[[:space:]]*$/d' | tr '\n' ' ')

COPY . .

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

