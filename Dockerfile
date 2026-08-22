# Multi-stage production Dockerfile for CineFlow AI Recommendation Engine
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH="/app/src"

# Copy dependency requirements
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install CPU-optimized dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY configs/ configs/
COPY src/ src/
COPY main.py ./
COPY .streamlit/ .streamlit/

# Create data and artifact mount points
RUN mkdir -p data/processed artifacts/models artifacts/embeddings artifacts/benchmarks

# Default to Streamlit UI port
EXPOSE 8501 8000

# Default command launches Streamlit Cinema UI
CMD ["python", "main.py", "ui", "--port", "8501"]
