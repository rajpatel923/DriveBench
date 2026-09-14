#!/usr/bin/env bash
# =============================================================================
# 00_setup_rife.sh  —  One-time setup for RIFE temporal interpolation.
#
# What this does:
#   1. Clones hzwer/ECCV2022-RIFE into recovery/rife/  (if not present)
#   2. Checks that PyTorch is installed (installs if missing)
#   3. Verifies that HDv3 weights exist in recovery/rife/train_log/
#      and prints download instructions if they are missing.
#
# Usage:
#   bash script/00_setup_rife.sh
#
# After success, run:
#   bash script/01_build_recovery.sh --with-rife
# =============================================================================
set -euo pipefail

RIFE_DIR="recovery/rife"
WEIGHTS_DIR="$RIFE_DIR/train_log"

echo "========================================"
echo " RIFE setup"
echo "========================================"

# ---------- 1. Clone repo ----------------------------------------------------
if [[ -d "$RIFE_DIR/.git" ]]; then
    echo "[OK] RIFE repo already present at $RIFE_DIR"
else
    echo "Cloning ECCV2022-RIFE into $RIFE_DIR ..."
    git clone https://github.com/hzwer/ECCV2022-RIFE "$RIFE_DIR"
    echo "[OK] Cloned."
fi

# ---------- 2. PyTorch -------------------------------------------------------
if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    echo "[OK] PyTorch $TORCH_VER is already installed."
else
    echo "PyTorch not found — installing (CPU/MPS build)..."
    pip install torch torchvision
fi

# ---------- 3. RIFE weights --------------------------------------------------
if [[ -d "$WEIGHTS_DIR" ]] && ls "$WEIGHTS_DIR"/*.pkl 1>/dev/null 2>&1; then
    echo "[OK] RIFE weights found in $WEIGHTS_DIR"
    echo ""
    echo "========================================"
    echo " RIFE is ready."
    echo " Run:  bash script/01_build_recovery.sh --with-rife"
    echo "========================================"
    exit 0
fi

# Weights missing — print instructions and exit non-zero
cat <<'EOF'

[!!] RIFE weights not found in recovery/rife/train_log/

Download the HDv3 weights from the RIFE model list:
  https://github.com/hzwer/ECCV2022-RIFE#model-list
  → "RIFE v4.6  (or HDv3)"

Manual steps:
  1. Download the zip / tar from the Google Drive link on that page.
  2. Extract so that the following file exists:
       recovery/rife/train_log/flownet.pkl
     (other .pkl files like contextnet.pkl, unet.pkl may also be present)

Linux shortcut (gdown):
  pip install gdown
  # replace FOLDER_ID with the Google Drive folder ID from the model-list page
  gdown --folder "FOLDER_ID" -O recovery/rife/train_log/

After placing the weights, re-run this script to confirm setup, then:
  bash script/01_build_recovery.sh --with-rife
EOF
exit 1
