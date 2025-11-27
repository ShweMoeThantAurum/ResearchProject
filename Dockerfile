FROM python:3.11-slim

# -------------------------------------------------------------
# System dependencies required for PyTorch + build tasks
# -------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------
# Create non-root user for safer execution
# -------------------------------------------------------------
RUN useradd -m aefluser
USER aefluser
WORKDIR /app

# Add project to PYTHONPATH
ENV PYTHONPATH=/app

# -------------------------------------------------------------
# Install Python dependencies
# -------------------------------------------------------------
COPY --chown=aefluser:aefluser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure NumPy < 2 for PyTorch ABI compatibility
RUN pip install --no-cache-dir "numpy<2" --force-reinstall

# Install CPU-only PyTorch wheels
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# -------------------------------------------------------------
# Copy project code
# -------------------------------------------------------------
COPY --chown=aefluser:aefluser . .

# -------------------------------------------------------------
# Default command (overridden by docker-compose)
# -------------------------------------------------------------
CMD ["python"]
