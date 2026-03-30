#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  OpenLens 2.0 — Linux Server Setup
#
#  Installs everything needed for production / mass-usage deployment on Linux
#  with NVIDIA GPU, including Flash Attention 2 for maximum inference speed.
#
#  Usage:
#      chmod +x setup_server.sh
#      ./setup_server.sh
# ──────────────────────────────────────────────────────────────────────────────
set -e

echo ""
echo "============================================================"
echo "  OpenLens 2.0 — Server Setup (Linux + CUDA)"
echo "============================================================"
echo ""

# ── Check Python ────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Install it first:"
    echo "  sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PYVER=$(python3 --version 2>&1)
echo "[OK] $PYVER found."

# ── Create virtual environment ──────────────────────────────────────────────
echo ""
echo "[1/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "       venv already exists, skipping."
else
    python3 -m venv venv
    echo "       Done."
fi

source venv/bin/activate

# ── Upgrade pip ─────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Upgrading pip..."
pip install --upgrade pip --quiet

# ── Install PyTorch with CUDA ──────────────────────────────────────────────
echo ""
echo "[3/6] Installing PyTorch with CUDA support..."
if nvidia-smi &>/dev/null; then
    echo "       NVIDIA GPU detected."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    echo "       [WARNING] No NVIDIA GPU detected. Installing CPU-only PyTorch."
    echo "       Flash Attention will NOT be available."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# ── Install Flash Attention 2 ──────────────────────────────────────────────
echo ""
echo "[4/6] Installing Flash Attention 2..."
if nvidia-smi &>/dev/null; then
    # flash-attn requires: Linux, CUDA toolkit, a C++ compiler
    if pip install flash-attn --no-build-isolation --quiet 2>/dev/null; then
        echo "       Flash Attention 2 installed successfully!"
        echo "       The model will use flash_attention_2 backend (fastest)."
    else
        echo "       [WARNING] Flash Attention build failed."
        echo "       Make sure you have: CUDA toolkit, gcc/g++, and ninja installed."
        echo "         sudo apt install build-essential ninja-build"
        echo "       Falling back to SDPA (still fast, ~30% slower than flash-attn)."
    fi
else
    echo "       Skipped (no GPU)."
fi

# ── Install project dependencies ───────────────────────────────────────────
echo ""
echo "[5/6] Installing Python dependencies..."
pip install -r requirements.txt --quiet
echo "       Done."

# ── Download model ──────────────────────────────────────────────────────────
echo ""
echo "[6/6] Downloading dots.mocr model weights..."
python3 -c "
from transformers import AutoTokenizer, AutoModel
import os
model_id = 'rednote-hilab/dots.mocr'
cache_dir = os.path.join('models', 'dots_mocr')
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
print('  Downloading model weights...')
AutoModel.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
print('  Model ready.')
"

# ── Ensure pipeline __init__ ────────────────────────────────────────────────
mkdir -p pipeline
touch pipeline/__init__.py

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  To start the app:"
echo "    source venv/bin/activate"
echo "    python3 app.py"
echo ""
echo "  For production (with Gunicorn or similar):"
echo "    source venv/bin/activate"
echo "    python3 app.py"
echo "    # Or use: gunicorn -w 1 -b 0.0.0.0:7860 'app:demo.app'"
echo ""
