#!/bin/bash

# 
#   bash train.sh
#
#   bash train.sh -r /nas/xiaoyingwang/seeclear/train_runs/opacification/<run_name>
#   bash train.sh -r /nas/xiaoyingwang/seeclear/train_runs/opacification/<run_name>/checkpoints/last.ckpt

RESUME_PATH=""
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--resume)
            RESUME_PATH="$2"
            shift 2
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
        --logdir /nas/xiaoyingwang/seeclear/train_runs/opacification \
        -r "$RESUME_PATH" \
        --base \
        --scale_lr False \
        $EXTRA_ARGS
    
else
    echo "========================================"
    echo "New Training Mode"
    echo "Using config: configs/opacification_train.yaml"
    echo "========================================"
    echo ""
    
    "$PYTHON_BIN" -u train_opacification.py \
        --logdir /nas/xiaoyingwang/seeclear/train_runs/opacification \
        --pretrained_model pretrained_models/seeclear_pretrained.ckpt \
        --base configs/opacification_train.yaml \
        --scale_lr False \
        $EXTRA_ARGS
fi

echo ""
echo "========================================"
echo "Training completed"
echo "========================================"
