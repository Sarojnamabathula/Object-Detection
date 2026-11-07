# =============================================================================
# Object Detection System — Multi-Stage Dockerfile
# =============================================================================
# Build a production-ready container for the Object Detection System.
#
# Ports:
#   8000  — FastAPI REST API
#   8501  — Streamlit dashboard
#
# Usage:
#   docker build -t object-detection-system .
#   docker run -p 8000:8000 -p 8501:8501 object-detection-system
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base image with system dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Metadata labels
LABEL maintainer="ODS Team"
LABEL version="1.0.0"
LABEL description="Real-time Object Detection System with YOLOv11, FastAPI, and Streamlit"
LABEL org.opencontainers.image.source="https://github.com/ods/object-detection-system"

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by OpenCV and general operation
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 2: Python dependency installation
# ---------------------------------------------------------------------------
FROM base AS dependencies

WORKDIR /tmp

# Copy and install Python requirements separately for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 3: Final production image
# ---------------------------------------------------------------------------
FROM dependencies AS production

WORKDIR /app

# Create a non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Create application directories with correct ownership
RUN mkdir -p /app/uploads /app/outputs /app/weights \
    && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose ports for the API and Streamlit dashboard
EXPOSE 8000 8501

# Healthcheck — ping the FastAPI /health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: start the FastAPI API server
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
