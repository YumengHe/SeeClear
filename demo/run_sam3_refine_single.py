from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _collect_masks(mask_input: Path) -> list[Path]:
    if mask_input.is_dir():
        return sorted(p for p in mask_input.glob("*.png") if p.is_file())
    return [mask_input] if mask_input.is_file() else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine one image's masks with sam3_mask_refine_single.py")
    default_sam3_root = Path(__file__).resolve().parents[1] / "third_party"
    parser.add_argument("--sam3-root", default=str(default_sam3_root))
    parser.add_argument("--checkpoint", default=str(Path(__file__).resolve().parents[1] / "pretrained_models" / "sam3.pt"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True, help="One png mask or a directory of png masks.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default="demo")
    args = parser.parse_args()

    sam3_root = Path(args.sam3_root).resolve()
    sam3_package_root = sam3_root / "sam3"
    checkpoint = Path(args.checkpoint).resolve()
    image_path = Path(args.image).resolve()
    mask_input = Path(args.mask).resolve()
    out_dir = Path(args.out_dir).resolve()
    masks = _collect_masks(mask_input)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not masks:
        raise FileNotFoundError(f"No png masks found from: {mask_input}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

    stage = out_dir / "_sam3_refine_stage"
    img_root = stage / "img_root"
    mask_root = stage / "mask_root"
    refined_root = out_dir / "refined"
    img_root.mkdir(parents=True, exist_ok=True)
    stem_mask_dir = mask_root / args.stem
    stem_mask_dir.mkdir(parents=True, exist_ok=True)
    refined_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, img_root / f"{args.stem}{image_path.suffix.lower() or '.png'}")
    for idx, mask_path in enumerate(masks):
        shutil.copy2(mask_path, stem_mask_dir / f"mask_{idx}.png")

    cmd = [
        sys.executable,
        str(sam3_root / "sam3_mask_refine_single.py"),
        "--img_root",
        str(img_root),
        "--mask_root",
        str(mask_root),
        "--out_root",
        str(refined_root),
    ]
    print("[RUN]", " ".join(cmd), flush=True)
    env = os.environ.copy()
    pythonpath = [str(sam3_package_root), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([p for p in pythonpath if p])
    env["SEECLEAR_SAM3_CKPT"] = str(checkpoint)
    subprocess.run(cmd, cwd=str(sam3_root), check=True, env=env)
    print("[DONE]")
    print("  refined:", refined_root / args.stem)


if __name__ == "__main__":
    main()
