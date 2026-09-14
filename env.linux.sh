#!/usr/bin/env bash
# Linux GPU (bare-metal, non-WSL2) environment setup.
# Source this before running vLLM-based inference on a CUDA machine.
#
# Platform guide:
#   macOS (Apple Silicon) : source env.sh
#   Linux GPU (bare-metal): source env.linux.sh   <-- this file
#   Linux GPU (WSL2)      : source env.wsl5090.sh

. "$(dirname "$0")/env.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
