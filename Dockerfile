FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUN_MODE=api \
    PORT=8000

WORKDIR /app

# Install minimal system dependencies needed by ML/vector packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Preload the default embedding model at build time so Render startup is faster
# and health checks are less likely to timeout on free-tier instances.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

CMD gunicorn "main:create_app()" --bind 0.0.0.0:${PORT} --workers 1 --timeout 180