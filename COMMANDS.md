# SeeClear Commands

Run commands from the repository root after activating the SeeClear environment:

```bash
conda activate seeclear
```

## Demo

Download the SeeClear demo checkpoints:

```bash
python scripts/download_checkpoints.py
```

```bash
python -m demo.app
```

Gradio prints the local URL in the terminal after the server starts.

## Dataset Split

Training uses three filename lists:

```text
dataset/my_data/train_list.txt
dataset/my_data/val_list.txt
dataset/my_data/test_list.txt
```

Create the lists from `dataset/my_data/opaque/`:

```bash
python scripts/split_dataset.py \
  --data_dir dataset/my_data \
  --train_size <num_train> \
  --val_size <num_val> \
  --test_size <num_test> \
  --seed 42
```

The training scripts use `train_list.txt` for optimization and `val_list.txt`
for validation. `test_list.txt` is reserved for final evaluation and is not used
while training.

## Training

Fine-tune the diffusion opacification model from the released SeeClear
checkpoint:

```bash
bash train.sh
```

Train the diffusion opacification model from the original Paint-by-Example
initialization:

```bash
bash train.sh --from-pbe
```

Resume diffusion training:

```bash
bash train.sh -r outputs/opacification/<run_name>
```

Train the mask refinement head:

```bash
bash train_mask_refiner.sh -s5
```

Mask refiner strategy options:

```text
S1-S4   point strategies
S5-S8   mask augmentation schemes
S9-S12  box modes
```

## Inference

Image to depth:

```bash
python -m demo.run_once \
  --image examples/demo/1.jpg \
  --mask-source trans4trans \
  --depth-source da3 \
  --work-dir outputs/demo/image_to_depth \
  --stem demo \
  --seed 42 \
  --unipc-steps 10
```

Image and mask to depth:

```bash
python -m demo.run_once \
  --image examples/demo/1.jpg \
  --mask examples/demo/masks \
  --mask-source upload \
  --depth-source da3 \
  --work-dir outputs/demo/mask_to_depth \
  --stem demo \
  --seed 42 \
  --unipc-steps 10
```

Image and mask to opaque image:

```bash
python scripts/infer_opacification.py \
  --image examples/demo/1.jpg \
  --mask examples/demo/masks \
  --work_dir outputs/demo/mask_to_opaque \
  --stem demo \
  --opacification_ckpt pretrained_models/seeclear_opacification.ckpt \
  --config configs/opacification_inference.yaml \
  --mask_refiner_path pretrained_models/mask_refiner.pth \
  --unipc_steps 10 \
  --seeds 42 \
  --batch_size 8 \
  --prep_mode fast
```
