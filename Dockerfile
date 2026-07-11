FROM python:3.11-slim

WORKDIR /app

# System dependencies: libsndfile1 (for librosa) + ffmpeg (audio fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (no CUDA libs — saves ~2GB image size and ~200MB RAM)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements first (Docker layer caching)
COPY web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, model, and templates
COPY web/app.py .
COPY web/model.pth .
COPY web/templates/ ./templates/

# Render sets PORT env var (default 10000)
ENV PORT=10000
EXPOSE $PORT

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
