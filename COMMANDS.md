# SeeClear Commands

This file lists the common commands for running the SeeClear demo, training the
two models, and running inference.

Run commands from the repository root unless noted otherwise:

```bash
cd /data/xiaoyingwang/projects/seeclear
conda activate seeclear
```

Set the GPU before running GPU jobs:

```bash
export CUDA_VISIBLE_DEVICES=0
```

The core environment is defined by `environment.yaml`. Optional SAM3, GSAM2,
and Trans4Trans demo dependencies are listed in `environment.optional.yaml` and
should be installed into the same environment when those modes are needed.

## Required Files

The code expects the following local paths by default:

```text
pretrained_models/seeclear_pretrained.ckpt
pretrained_models/clip-vit-large-patch14/
pretrained_models/seeclear_opacification.ckpt
pretrained_models/mask_refiner.pth
pretrained_models/sam3.pt
configs/opacification_inference.yaml
```

In the code-only release, model weights are not included. Put weights in those
paths, or create symlinks from those paths to external weight storage.

The full image-to-depth pipeline can also call external repositories:

```text
../Trans4Trans_clean
../Depth-Anything-3
../moge
```

## Demo

### Gradio Demo

```bash
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_SERVER_PORT=7861
export GRADIO_TEMP_DIR=/nas/xiaoyingwang/seeclear/gradio_runs
mkdir -p "$GRADIO_TEMP_DIR"

python -m demo.app
```

The demo caches the opacification model, mask refiner, Depth Anything V3,
MoGe, and SAM3 interactive model in the server process after first use.

Open:

```text
http://127.0.0.1:7861
```

From a local machine, create an SSH tunnel:

```bash
ssh -N -L 7861:127.0.0.1:7861 -J delluluwxy@leap.math.ucla.edu xiaoyingwang@fanfuai.math.ucla.edu
```

If local port 7861 is occupied:

```bash
ssh -N -L 7862:127.0.0.1:7861 -J delluluwxy@leap.math.ucla.edu xiaoyingwang@fanfuai.math.ucla.edu
```

Then open:

```text
http://127.0.0.1:7862
```

### CLI Demo With an Existing Mask

```bash
python -m demo.run_once \
  --image examples/demo/1.jpg \
  --mask examples/demo/masks \
  --mask-source upload \
  --depth-source da3 \
  --work-dir /nas/xiaoyingwang/seeclear/demo_runs/demo_mask_to_depth \
  --stem demo \
  --seed 42 \
  --unipc-steps 10
```

Main outputs:

```text
/nas/xiaoyingwang/seeclear/demo_runs/demo_mask_to_depth/opaque_0/blend/demo_result.png
/nas/xiaoyingwang/seeclear/demo_runs/demo_mask_to_depth/opaque_0/pred_mask/demo/union.png
/nas/xiaoyingwang/seeclear/demo_runs/demo_mask_to_depth/depth/demo.npy
/nas/xiaoyingwang/seeclear/demo_runs/demo_mask_to_depth/depth/demo_color.png
```

## Training

### Train the Diffusion Opacification Model

This trains the transparent-to-opaque diffusion model. The default config
matches the final model settings:

```bash
bash train.sh
```

Resume from a training directory:

```bash
bash train.sh -r outputs/opacification/<run_name>
```

Resume from a checkpoint:

```bash
bash train.sh -r outputs/opacification/<run_name>/checkpoints/last.ckpt
```

Default training config:

```text
configs/opacification_train.yaml
```

Default output directory:

```text
outputs/opacification/
```

### Train the Mask Refiner

This trains the lightweight mask refinement model used after diffusion.

```bash
bash train_mask_refiner.sh -s5
```

Strategy options:

```text
S1-S4   point strategies
S5-S8   mask augmentation schemes
S9-S12  bbox modes
```

Default output directory:

```text
outputs/mask_refiner/
```

The default release checkpoint path used by inference is:

```text
pretrained_models/mask_refiner.pth
```

## Inference

### 1. Image to Depth

This runs automatic mask prediction with Trans4Trans, diffusion opacification,
mask refinement, compositing, and depth prediction.

```bash
python -m demo.run_once \
  --image examples/demo/1.jpg \
  --mask-source trans4trans \
  --depth-source da3 \
  --work-dir /nas/xiaoyingwang/seeclear/demo_runs/image_to_depth \
  --stem demo \
  --seed 42 \
  --unipc-steps 10
```

Use MoGe instead of Depth Anything 3:

```bash
python -m demo.run_once \
  --image examples/demo/1.jpg \
  --mask-source trans4trans \
  --depth-source moge \
  --work-dir /nas/xiaoyingwang/seeclear/demo_runs/image_to_depth_moge \
  --stem demo \
  --seed 42 \
  --unipc-steps 10
```

### 2. Image and Mask to Depth

This skips automatic segmentation and uses the provided mask.

```bash
python -m demo.run_once \
  --image examples/demo/1.jpg \
  --mask examples/demo/masks \
  --mask-source upload \
  --depth-source da3 \
  --work-dir /nas/xiaoyingwang/seeclear/demo_runs/mask_to_depth \
  --stem demo \
  --seed 42 \
  --unipc-steps 10
```

### 3. Image and Mask to Opaque Image Only

This runs only diffusion opacification plus mask refinement and compositing. It
does not run the depth model.

```bash
python scripts/infer_opacification.py \
  --image examples/demo/1.jpg \
  --mask examples/demo/masks \
  --work_dir /nas/xiaoyingwang/seeclear/demo_runs/mask_to_opaque \
  --stem demo \
  --opacification_ckpt pretrained_models/seeclear_opacification.ckpt \
  --config configs/opacification_inference.yaml \
  --mask_refiner_path pretrained_models/mask_refiner.pth \
  --unipc_steps 10 \
  --seeds 42 \
  --batch_size 8 \
  --prep_mode fast
```

Main opaque outputs:

```text
/nas/xiaoyingwang/seeclear/demo_runs/mask_to_opaque/opaque_0/blend/demo_result.png
/nas/xiaoyingwang/seeclear/demo_runs/mask_to_opaque/opaque_0/mask_blend/demo_result.png
/nas/xiaoyingwang/seeclear/demo_runs/mask_to_opaque/opaque_0/pred_mask/demo/union.png
```
