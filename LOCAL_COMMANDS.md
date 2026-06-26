# Local Commands

## Train Opacification

```bash
cd /data/xiaoyingwang/projects/seeclear
source /data/xiaoyingwang/tools/miniconda3/etc/profile.d/conda.sh
conda activate /data/xiaoyingwang/projects/seeclear/.conda/seeclear
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/data/xiaoyingwang/projects/seeclear/.conda/matplotlib
export XDG_CACHE_HOME=/data/xiaoyingwang/projects/seeclear/.conda/cache
export HF_HOME=/nas/xiaoyingwang/seeclear/hf_home
export TRANSFORMERS_CACHE=/nas/xiaoyingwang/seeclear/hf_home/transformers
export TMPDIR=/nas/xiaoyingwang/seeclear/gradio_runs
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash train.sh
```

## Train Mask Refiner

```bash
cd /data/xiaoyingwang/projects/seeclear
source /data/xiaoyingwang/tools/miniconda3/etc/profile.d/conda.sh
conda activate /data/xiaoyingwang/projects/seeclear/.conda/seeclear
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/data/xiaoyingwang/projects/seeclear/.conda/matplotlib
export XDG_CACHE_HOME=/data/xiaoyingwang/projects/seeclear/.conda/cache
export HF_HOME=/nas/xiaoyingwang/seeclear/hf_home
export TRANSFORMERS_CACHE=/nas/xiaoyingwang/seeclear/hf_home/transformers
export TMPDIR=/nas/xiaoyingwang/seeclear/gradio_runs
export CUDA_VISIBLE_DEVICES=0
bash train_mask_refiner.sh -s5
```

## Infer: Image to Depth

```bash
cd /data/xiaoyingwang/projects/seeclear
source /data/xiaoyingwang/tools/miniconda3/etc/profile.d/conda.sh
conda activate /data/xiaoyingwang/projects/seeclear/.conda/seeclear
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/data/xiaoyingwang/projects/seeclear/.conda/matplotlib
export XDG_CACHE_HOME=/data/xiaoyingwang/projects/seeclear/.conda/cache
export HF_HOME=/nas/xiaoyingwang/seeclear/hf_home
export TRANSFORMERS_CACHE=/nas/xiaoyingwang/seeclear/hf_home/transformers
export TMPDIR=/nas/xiaoyingwang/seeclear/gradio_runs
export CUDA_VISIBLE_DEVICES=0
python -m demo.run_once \
  --image examples/demo/1.jpg \
  --mask-source trans4trans \
  --depth-source da3 \
  --work-dir /nas/xiaoyingwang/seeclear/demo_runs/image_to_depth \
  --stem real15 \
  --seed 42 \
  --unipc-steps 10
```

## Infer: Image and Mask to Depth

```bash
cd /data/xiaoyingwang/projects/seeclear
source /data/xiaoyingwang/tools/miniconda3/etc/profile.d/conda.sh
conda activate /data/xiaoyingwang/projects/seeclear/.conda/seeclear
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/data/xiaoyingwang/projects/seeclear/.conda/matplotlib
export XDG_CACHE_HOME=/data/xiaoyingwang/projects/seeclear/.conda/cache
export HF_HOME=/nas/xiaoyingwang/seeclear/hf_home
export TRANSFORMERS_CACHE=/nas/xiaoyingwang/seeclear/hf_home/transformers
export TMPDIR=/nas/xiaoyingwang/seeclear/gradio_runs
export CUDA_VISIBLE_DEVICES=0
python -m demo.run_once \
  --image examples/demo/1.jpg \
  --mask examples/demo/masks/mask_1_transparent_objects.png \
  --mask-source upload \
  --depth-source da3 \
  --work-dir /nas/xiaoyingwang/seeclear/demo_runs/mask_to_depth \
  --stem real15 \
  --seed 42 \
  --unipc-steps 10
```

## Infer: Image and Mask to Opaque

```bash
cd /data/xiaoyingwang/projects/seeclear
source /data/xiaoyingwang/tools/miniconda3/etc/profile.d/conda.sh
conda activate /data/xiaoyingwang/projects/seeclear/.conda/seeclear
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/data/xiaoyingwang/projects/seeclear/.conda/matplotlib
export XDG_CACHE_HOME=/data/xiaoyingwang/projects/seeclear/.conda/cache
export HF_HOME=/nas/xiaoyingwang/seeclear/hf_home
export TRANSFORMERS_CACHE=/nas/xiaoyingwang/seeclear/hf_home/transformers
export TMPDIR=/nas/xiaoyingwang/seeclear/gradio_runs
export CUDA_VISIBLE_DEVICES=0
python scripts/infer_opacification.py \
  --image examples/demo/1.jpg \
  --mask examples/demo/masks/mask_1_transparent_objects.png \
  --work_dir /nas/xiaoyingwang/seeclear/demo_runs/mask_to_opaque \
  --stem real15 \
  --opacification_ckpt pretrained_models/seeclear_opacification.ckpt \
  --config configs/opacification_inference.yaml \
  --mask_refiner_path pretrained_models/mask_refiner.pth \
  --unipc_steps 10 \
  --seeds 42 \
  --batch_size 8 \
  --prep_mode fast
```

## Demo

```bash
cd /data/xiaoyingwang/projects/seeclear
source /data/xiaoyingwang/tools/miniconda3/etc/profile.d/conda.sh
conda activate /data/xiaoyingwang/projects/seeclear/.conda/seeclear
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/data/xiaoyingwang/projects/seeclear/.conda/matplotlib
export XDG_CACHE_HOME=/data/xiaoyingwang/projects/seeclear/.conda/cache
export HF_HOME=/nas/xiaoyingwang/seeclear/hf_home
export TRANSFORMERS_CACHE=/nas/xiaoyingwang/seeclear/hf_home/transformers
export TMPDIR=/nas/xiaoyingwang/seeclear/gradio_runs
export CUDA_VISIBLE_DEVICES=0
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_SERVER_PORT=7861
export GRADIO_TEMP_DIR=/nas/xiaoyingwang/seeclear/gradio_runs
mkdir -p "$GRADIO_TEMP_DIR"
python -m demo.app
```

## Connect to Demo From Local Machine

```bash
ssh -N -L 7861:127.0.0.1:7861 -J delluluwxy@leap.math.ucla.edu xiaoyingwang@fanfuai.math.ucla.edu
```

```text
http://127.0.0.1:7861
```

```bash
ssh -N -L 7862:127.0.0.1:7861 -J delluluwxy@leap.math.ucla.edu xiaoyingwang@fanfuai.math.ucla.edu
```

```text
http://127.0.0.1:7862
```
