#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-7862}"
GPU="${GPU:-6}"
OUT_DIR="${OUT_DIR:-/nas/xiaoyingwang/datasets/sam3_gradio_masks}"

cd /data/xiaoyingwang/projects/sam3_infer
CUDA_VISIBLE_DEVICES="${GPU}" \
  python sam3_click_gradio.py \
  --out_dir "${OUT_DIR}" \
  --port "${PORT}"
