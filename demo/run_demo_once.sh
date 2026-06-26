#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-/data/xiaoyingwang/projects/sam3_infer/real_more/1.jpg}"
MASK="${2:-/data/xiaoyingwang/projects/sam3_infer/real_more_mask/1}"
OUT_DIR="${3:-${REPO_ROOT}/demo_outputs/sample_real_more_1}"
STEM="${4:-demo}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
DEPTH_SOURCE="${DEPTH_SOURCE:-da3}"

cd "$REPO_ROOT"

python -m demo.run_once \
  --image "$IMAGE" \
  --mask "$MASK" \
  --mask-source upload \
  --depth-source "$DEPTH_SOURCE" \
  --work-dir "$OUT_DIR" \
  --stem "$STEM" \
  --seed 42 \
  --unipc-steps 10 \
  --commands-txt "${OUT_DIR}/commands.txt"
