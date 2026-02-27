# Stage 1: Build frontend
FROM node:20-slim AS frontend

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend + serve built frontend
FROM python:3.12-slim

# System dependencies for pyvips, GDAL, rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvips-dev \
    gdal-bin \
    libgdal-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
COPY mapgen/ ./mapgen/
RUN pip install --no-cache-dir -e ".[api]"

# Copy built frontend
COPY --from=frontend /app/frontend/dist /app/static

# Create data directory for volumes
RUN mkdir -p /app/data/cache /app/data/output

ENV MAPGEN_CACHE_DIR=/app/data/cache
ENV MAPGEN_OUTPUT_DIR=/app/data/output

EXPOSE 8000

CMD uvicorn mapgen.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
