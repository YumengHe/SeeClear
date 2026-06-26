#!/bin/bash

DATA_DIR="dataset/my_data"
PYTHON_BIN="python"

BASE_CKPT_DIR="/nas/xiaoyingwang/seeclear/train_runs/mask_refiner"

BATCH_SIZE=16
EPOCHS=500
LR=1e-4

STRATEGY=""
while getopts "s:" opt; do
  case $opt in
    s)
      STRATEGY=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "Usage: $0 -s<strategy_number>"
      echo "Example: $0 -s9  (for BBox mode 1)"
      exit 1
      ;;
  esac
done

if [ -z "$STRATEGY" ]; then
    echo "Error: Strategy not specified!"
    echo "Usage: $0 -s<strategy_number>"
    echo "  S1-4:  Point strategies"
    echo "  S5-8:  Mask augmentation schemes"
    echo "  S9-12: BBox modes 1-4"
    echo "Example: $0 -s9  (BBox mode 1: GT BBox)"
    exit 1
fi

if ! [[ "$STRATEGY" =~ ^([1-9]|1[0-2])$ ]]; then
    echo "Error: Strategy must be between 1 and 12"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

SAVE_DIR="${BASE_CKPT_DIR}/s${STRATEGY}_${TIMESTAMP}"

mkdir -p $BASE_CKPT_DIR

echo "=========================================="
echo "Training Mask Refiner - Strategy $STRATEGY"
echo "=========================================="
echo "Data Dir:  $DATA_DIR"
echo "Save Dir:  $SAVE_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs:    $EPOCHS"
echo "Learning Rate: $LR"
echo "=========================================="

"$PYTHON_BIN" mask_refiner/train.py \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --strategy $STRATEGY \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --resume

echo "Training completed! Models saved to: $SAVE_DIR"
