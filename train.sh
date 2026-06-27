#!/bin/bash
set -euo pipefail

# 
#   bash train.sh
#   bash train.sh --from-pbe
#
#   bash train.sh -r outputs/opacification/<run_name>
#   bash train.sh -r outputs/opacification/<run_name>/checkpoints/last.ckpt

RESUME_PATH=""
PYTHON_BIN="python"
EXTRA_ARGS=""
PRETRAINED_MODEL="pretrained_models/seeclear_pretrained.ckpt"
INIT_MODE="SeeClear fine-tuning"
DATA_DIR="dataset/my_data"

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        --from-pbe)
            PRETRAINED_MODEL="pretrained_models/pbe_pretrained.ckpt"
            INIT_MODE="Paint-by-Example initialization"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

if [ -n "$RESUME_PATH" ]; then
    echo "========================================"
    echo "Resume Training Mode"
    echo "Resume from: $RESUME_PATH"
    echo "========================================"
    echo ""
    
    if [ ! -e "$RESUME_PATH" ]; then
        echo "Error: Resume path does not exist: $RESUME_PATH"
        exit 1
    fi
    
    echo "Note: Config files will be automatically loaded from training directory"
    echo "      (train_opacification.py will auto-load configs/*.yaml from the resume directory)"
    echo ""
    
    "$PYTHON_BIN" -u train_opacification.py \
        --logdir outputs/opacification \
        -r "$RESUME_PATH" \
        --base \
        --scale_lr False \
        $EXTRA_ARGS
    
else
    echo "========================================"
    echo "New Training Mode"
    echo "Using config: configs/opacification_train.yaml"
    echo "Initialization: $INIT_MODE"
    echo "Checkpoint: $PRETRAINED_MODEL"
    echo "========================================"
    echo ""

    if [ ! -f "$PRETRAINED_MODEL" ]; then
        echo "Error: Pretrained checkpoint does not exist: $PRETRAINED_MODEL"
        exit 1
    fi

    for split_file in train_list.txt val_list.txt test_list.txt; do
        if [ ! -f "$DATA_DIR/$split_file" ]; then
            echo "Error: Missing dataset split file: $DATA_DIR/$split_file"
            echo "Run: python scripts/split_dataset.py --data_dir $DATA_DIR"
            exit 1
        fi
    done
    
    "$PYTHON_BIN" -u train_opacification.py \
        --logdir outputs/opacification \
        --pretrained_model "$PRETRAINED_MODEL" \
        --base configs/opacification_train.yaml \
        --scale_lr False \
        --no-test True \
        $EXTRA_ARGS
fi

echo ""
echo "========================================"
echo "Training completed"
echo "========================================"
