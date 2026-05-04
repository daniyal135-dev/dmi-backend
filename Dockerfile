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

# Copy requirements and install packages with retries
# Note: Install CPU-only PyTorch for Docker (GPU requires NVIDIA Container Toolkit)
COPY requirements.txt .
# First install all packages except torch/torchvision
RUN pip install --default-timeout=1000 --retries 10 $(grep -v "^torch" requirements.txt | grep -v "^torchvision" | grep -v "^#" | tr '\n' ' ') || true
# Then install CPU-only PyTorch (much faster, no CUDA libraries)
RUN pip install --default-timeout=1000 --retries 10 torch torchvision --index-url https://download.pytorch.org/whl/cpu || \
    (echo "First attempt failed, retrying..." && \
     pip install --default-timeout=1000 --retries 10 torch torchvision --index-url https://download.pytorch.org/whl/cpu)

COPY . .

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

