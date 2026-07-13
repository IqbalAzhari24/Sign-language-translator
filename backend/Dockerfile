FROM python:3.11-slim

# mediapipe needs system OpenGL libs just to import, even for CPU-only inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgles2 libegl1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY train/ train/
COPY models/ models/
COPY checkpoints/ checkpoints/

EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
