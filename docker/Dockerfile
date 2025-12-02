# Lightweight Python base
FROM python:3.10-slim

# Prevent Python from buffering stdout
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install OS-level deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Entrypoint script for clients
RUN chmod +x docker/entrypoint.sh

ENTRYPOINT ["/bin/bash", "docker/entrypoint.sh"]
