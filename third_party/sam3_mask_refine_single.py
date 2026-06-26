#!/usr/bin/env python3
import os
import glob
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_tracker
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor


# ==========================================

# ==========================================
def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return torch.load(path, map_location="cpu")


def load_sam3_predictor(device="cuda"):
    tracker = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
    ckpt_path = os.environ.get("SEECLEAR_SAM3_CKPT")
    if not ckpt_path:
        ckpt_path = Path(__file__).resolve().parents[1] / "pretrained_models" / "sam3.pt"
    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {ckpt_path}")
    ckpt = _torch_load(ckpt_path)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    sd = {}
    for k, v in ckpt.items():
        if k.startswith("tracker."):
            sd[k[len("tracker."):]] = v
        elif k.startswith("detector.backbone."):
            new_key = k.replace("detector.backbone.", "backbone.")
            sd[new_key] = v
            if "vision_backbone.convs." in new_key:
                sd[new_key.replace("convs.", "sam2_convs.")] = v

    tracker.load_state_dict(sd, strict=False)
    tracker.eval().to(device)
    return SAM3InteractiveImagePredictor(tracker)


# ==========================================

# ==========================================
def filter_small_components(mask, min_area=10):
    mask_bin = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    output_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output_mask[labels == i] = 255
    return output_mask


def post_process_union_smooth(sam_mask, orig_mask, kernel_size=5):
    sam_bin  = (sam_mask  > 0).astype(np.uint8) * 255
    orig_bin = (orig_mask > 0).astype(np.uint8) * 255
    union = cv2.bitwise_or(sam_bin, orig_bin)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    refined = cv2.morphologyEx(union,    cv2.MORPH_OPEN,  k, iterations=2)
    refined = cv2.morphologyEx(refined,  cv2.MORPH_CLOSE, k, iterations=2)
    return filter_small_components(refined, min_area=10)


# ==========================================

# ==========================================
def process_mask_refine(predictor, image_np, mask_path, low_res_size):
    h, w = image_np.shape[:2]
    orig_mask = np.array(Image.open(mask_path).convert("L").resize((w, h), Image.NEAREST))
    orig_mask = (orig_mask > 127).astype(np.uint8) * 255

    if not np.any(orig_mask > 127):
        return None

    low_res = cv2.resize(orig_mask, (low_res_size, low_res_size), interpolation=cv2.INTER_NEAREST)
    mask_input = (low_res.astype(np.float32) / 255.0 - 0.5) * 20.0

    masks, iou_preds, _ = predictor.predict(
        mask_input=mask_input[None, :, :],
        multimask_output=True,
        normalize_coords=True,
    )
    sam_raw = (masks[np.argmax(iou_preds)] > 0).astype(np.uint8) * 255
    final = post_process_union_smooth(sam_raw, orig_mask, kernel_size=5)
    return final


# ==========================================

# ==========================================
def main():
    parser = argparse.ArgumentParser(description="SAM3 mask refinement")
    parser.add_argument("--img_root",  required=True,
                        help="Input image folder. Filenames should match mask folder stems.")
    parser.add_argument("--mask_root", required=True,
                        help="Mask folder with layout <stem>/mask_0.png ...")
    parser.add_argument("--out_root",  required=True,
                        help="Output folder with the same layout as mask_root.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    predictor = load_sam3_predictor(device)
    low_res_size = (predictor.model.image_size // predictor.model.backbone_stride) * 4
    print(f"low_res_size = {low_res_size}")

    stem_dirs = sorted([
        d for d in glob.glob(os.path.join(args.mask_root, "*"))
        if os.path.isdir(d)
    ])

    if not stem_dirs:
        print(f"[ERROR] No subdirectories found in {args.mask_root}")
        return

    print(f"Found {len(stem_dirs)} image(s) to process.")

    for stem_dir in tqdm(stem_dirs, desc="refining"):
        stem = os.path.basename(stem_dir)


        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            c = os.path.join(args.img_root, stem + ext)
            if os.path.exists(c):
                img_path = c
                break

        if img_path is None:
            print(f"  [WARN] image not found for stem '{stem}', skipping.")
            continue

        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        predictor.set_image(img_rgb)

        out_stem_dir = os.path.join(args.out_root, stem)
        os.makedirs(out_stem_dir, exist_ok=True)

        mask_files = sorted(glob.glob(os.path.join(stem_dir, "*.png")))
        if not mask_files:
            print(f"  [WARN] no masks in {stem_dir}, skipping.")
            continue

        for mask_file in mask_files:
            final = process_mask_refine(predictor, img_rgb, mask_file, low_res_size)
            if final is None:
                print(f"  [WARN] empty mask: {mask_file}")
                continue
            save_path = os.path.join(out_stem_dir, os.path.basename(mask_file))
            Image.fromarray(final).save(save_path)

    print(f"\n[DONE] refined masks saved to: {args.out_root}")


if __name__ == "__main__":
    main()
