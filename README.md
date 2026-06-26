# SeeClear: Reliable Transparent Object Depth Estimation via Generative Opacification

<h4 align="center">

Xiaoying Wang<sup>*</sup>, Yumeng He<sup>*</sup>, Jingkai Shi<sup>*</sup>, Jiayin Lu, Yin Yang, Ying Jiang, Chenfanfu Jiang

[![arXiv](https://img.shields.io/badge/arXiv-2603.19547-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2603.19547)
[![Project Page](https://img.shields.io/badge/Project%20Page-SeeClear-blue.svg)](https://heyumeng.com/SeeClear-web/)
[![Code](https://img.shields.io/badge/Code-SeeClear-green.svg)](#quick-start)
[![Model](https://img.shields.io/badge/Model-Coming%20Soon-lightgrey.svg)](#model-files)
[![Dataset](https://img.shields.io/badge/Dataset-Coming%20Soon-lightgrey.svg)](#dataset)
[![Demo](https://img.shields.io/badge/Demo-Coming%20Soon-lightgrey.svg)](#quick-start)

<p align="center">
  <img src="assets/teaser.png" alt="SeeClear teaser" width="100%">
</p>

</h4>

This repository contains the official implementation of **SeeClear**, a
plug-and-play framework for transparent-object depth estimation. SeeClear first
converts transparent regions into geometry-consistent opaque appearances, then
feeds the optimized image to an off-the-shelf monocular depth estimator. This
keeps the depth model unchanged while improving depth stability on transparent
objects.

The released code includes the final inference pipeline, diffusion
opacification training code, mask refinement training code, and demo wrappers
for image-to-depth and mask-to-depth workflows.

## News

- **Coming soon**: Pretrained checkpoints and dataset release links will be
  added.
- **2026-03-20**: SeeClear is available on arXiv.

## Method Overview

Given an RGB image containing transparent objects, SeeClear runs the following
pipeline:

1. Localize transparent regions using automatic segmentation or an uploaded
   mask.
2. Generate an opaque version of the transparent region with a conditional
   latent diffusion model.
3. Refine the blending mask with a lightweight mask head.
4. Composite the generated opaque region into the original image.
5. Predict depth with a foundation monocular depth estimator.

The default automatic segmentation path uses Trans4Trans. The demo code also
keeps auxiliary SAM3 and GSAM2 mask wrappers used for interactive or
prompt-based mask preparation.

## Installation

Create the core SeeClear environment for diffusion opacification, mask
refinement, training, inference, and the base Gradio demo:

```bash
conda env create -f environment.yaml
conda activate seeclear
```

Optional demo backends can be installed into the same environment:

```bash
conda env update -n seeclear -f environment.optional.yaml
```

The optional file updates the same environment; it does not create a second
environment.

The optional automatic image-to-depth and interactive mask modes expect these
external repositories next to this repository:

```text
../Trans4Trans_clean
../Depth-Anything-3
../moge
```

## Model Files

This code release does not include model weights or datasets. Put the required
weights at the default paths below, or create symlinks from these paths to your
external storage:

```text
pretrained_models/seeclear_pretrained.ckpt
pretrained_models/clip-vit-large-patch14/
pretrained_models/seeclear_opacification.ckpt
pretrained_models/mask_refiner.pth
pretrained_models/sam3.pt
```

The final inference config is included at:

```text
configs/opacification_inference.yaml
```

## Quick Start

Run the Gradio demo:

```bash
source /data/xiaoyingwang/tools/miniconda3/etc/profile.d/conda.sh
conda activate seeclear
export CUDA_VISIBLE_DEVICES=0
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_SERVER_PORT=7861
export GRADIO_TEMP_DIR=/nas/xiaoyingwang/seeclear/gradio_runs
mkdir -p "$GRADIO_TEMP_DIR"
python -m demo.app
```

The Gradio demo keeps the opacification model, mask refiner, Depth Anything V3,
MoGe, and SAM3 models cached in the server process after first use.

Connect from a local machine:

```bash
ssh -N -L 7861:127.0.0.1:7861 -J delluluwxy@leap.math.ucla.edu xiaoyingwang@fanfuai.math.ucla.edu
```

Open:

```text
http://127.0.0.1:7861
```

Run single-image inference with an existing mask:

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

Run automatic image-to-depth inference with Trans4Trans masks:

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

For the complete command list, including opaque-only inference, see
[COMMANDS.md](COMMANDS.md).

## Training

Train the diffusion opacification model with the final default config:

```bash
bash train.sh
```

Resume from a training directory or checkpoint:

```bash
bash train.sh -r outputs/opacification/<run_name>
bash train.sh -r outputs/opacification/<run_name>/checkpoints/last.ckpt
```

Train the mask refinement head:

```bash
bash train_mask_refiner.sh -s5
```

The default diffusion training config is:

```text
configs/opacification_train.yaml
```

## Dataset

The paper introduces **SeeClear-396k**, a synthetic paired dataset of
transparent and opaque renderings with aligned masks, depth, and normals for
training the generative opacification model. The dataset is not included in this
code-only release.

## Repository Layout

```text
configs/                opacification training and inference configs
demo/                   Gradio demo and single-image inference wrappers
ldm/                    latent diffusion model implementation
mask_refiner/           mask refinement model and training code
pretrained_models/      default checkpoint location; weights are excluded
scripts/                command-line inference entry points
COMMANDS.md             detailed runnable commands
train.sh                diffusion opacification training entry point
train_mask_refiner.sh   mask refinement training entry point
```

## Citation

If you find SeeClear useful, please cite:

```bibtex
@article{wang2026seeclear,
  title={SeeClear: Reliable Transparent Object Depth Estimation via Generative Opacification},
  author={Wang, Xiaoying and He, Yumeng and Shi, Jingkai and Lu, Jiayin and Yang, Yin and Jiang, Ying and Jiang, Chenfanfu},
  journal={arXiv preprint arXiv:2603.19547},
  year={2026}
}
```

## Acknowledgements

SeeClear builds on latent diffusion models, transparent-object segmentation
models, and modern monocular depth estimators. We thank the authors and
maintainers of these open-source projects for making their work available.

## Contact

For questions, please open an issue or contact the authors listed on the
project page.
