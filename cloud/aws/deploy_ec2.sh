#!/bin/bash
# ============================================================
# AEFL-Traffic Cloud Setup Script for AWS EC2
# Repository: https://github.com/ShweMoeThantAurum/ResearchProject
# ============================================================

set -e  # stop on first error

# === 1. Update and install dependencies ===
echo "[Setup] Updating and installing base packages..."
sudo apt update -y
sudo apt install -y git python3 python3-venv python3-pip docker.io

# === 2. Ensure correct working directory ===
cd ~
if [ -d "ResearchProject/.git" ]; then
  echo "[Info] Existing repository detected. Pulling latest changes..."
  cd ResearchProject
  git pull origin main
else
  echo "[Setup] Cloning AEFL Research Project repository..."
  rm -rf ResearchProject  # cleanup any old folders
  git clone https://github.com/ShweMoeThantAurum/ResearchProject.git
  cd ResearchProject
fi

# === 3. Create Python virtual environment ===
echo "[Setup] Creating Python virtual environment..."
python3 -m venv .venv

# === 4. Activate environment and install dependencies ===
echo "[Setup] Activating virtual environment..."
source .venv/bin/activate

echo "[Setup] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# === 5. Verify installation ===
echo "[Verify] Python version:"
python3 --version

echo "[Verify] PyTorch check:"
python -c "import torch; print('PyTorch available:', torch.cuda.is_available())"

# === 6. Verify Docker setup ===
echo "[Verify] Docker version:"
sudo systemctl enable docker
sudo systemctl start docker
sudo docker --version

# === 7. Summary ===
echo "============================================================"
echo "[SUCCESS] AEFL environment is ready on this EC2 instance."
echo "Repository path: ~/ResearchProject"
echo "Activate venv with: source ~/ResearchProject/.venv/bin/activate"
echo "Run example: python src/experiments/run_aefl.py --config configs/aefl_sz.yaml"
echo "============================================================"
