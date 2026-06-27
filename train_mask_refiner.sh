#!/bin/bash
set -euo pipefail

DATA_DIR="dataset/my_data"
PYTHON_BIN="python"
EXTRA_ARGS=()

BASE_CKPT_DIR="outputs/mask_refiner"

BATCH_SIZE=16
EPOCHS=500
LR=1e-4

STRATEGY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s)
            STRATEGY="$2"
            shift 2
            ;;
        -s*)
            STRATEGY="${1#-s}"
            shift
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
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

if [ ! -d "$DATA_DIR/opaque" ] || [ ! -d "$DATA_DIR/transparent" ] || [ ! -d "$DATA_DIR/mask" ]; then
    echo "Error: Mask refiner dataset is incomplete under: $DATA_DIR"
    echo "Expected directories:"
    echo "  $DATA_DIR/opaque"
    echo "  $DATA_DIR/transparent"
    echo "  $DATA_DIR/mask"
    exit 1
fi

for split_file in train_list.txt val_list.txt test_list.txt; do
    if [ ! -f "$DATA_DIR/$split_file" ]; then
        echo "Error: Missing dataset split file: $DATA_DIR/$split_file"
        echo "Run: python scripts/split_dataset.py --data_dir $DATA_DIR"
        exit 1
    fi
done

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
    --resume \
    "${EXTRA_ARGS[@]}"

echo "Training completed! Models saved to: $SAVE_DIR"
