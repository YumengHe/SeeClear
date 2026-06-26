import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from demo.visualization import save_depth_visualizations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moge-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default="demo")
    parser.add_argument("--pretrained", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--resolution-level", type=int, default=7)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    moge_root = Path(args.moge_root).resolve()
    if str(moge_root) not in sys.path:
        sys.path.insert(0, str(moge_root))

    from moge.model.v2 import MoGeModel

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained(args.pretrained).to(device).eval()
    if args.fp16:
        model = model.half()

    image_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        output = model.infer(
            image_tensor,
            resolution_level=args.resolution_level,
            use_fp16=args.fp16,
        )

    depth = output["depth"].detach().float().cpu().numpy().astype(np.float32)
    npy_path = out_dir / f"{args.stem}.npy"
    np.save(npy_path, depth)
    save_depth_visualizations(
        npy_path,
        out_dir / f"{args.stem}_gray_near_black.png",
        out_dir / f"{args.stem}_color.png",
    )
    print("[DONE]")
    print("  npy:", npy_path)
    print("  gray:", out_dir / f"{args.stem}_gray_near_black.png")
    print("  color:", out_dir / f"{args.stem}_color.png")


if __name__ == "__main__":
    main()
