from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from demo.visualization import save_depth_visualizations


def _da3_stem(image_path: Path) -> str:
    stem = image_path.stem
    for suffix in ("_input", "_result"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


class DA3Runtime:
    def __init__(self, repo_root: Path, model_name: str):
        package_root = Path(repo_root) / "src"
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from depth_anything_3.api import DepthAnything3

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnything3.from_pretrained(model_name).to(self.device)

    @torch.no_grad()
    def run(self, image_path: Path, out_dir: Path) -> Path:
        image_path = Path(image_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _da3_stem(image_path)
        intrinsics = np.array(
            [[256.0, 0.0, 256.0], [0.0, 256.0, 256.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        pred = self.model.inference([str(image_path)], intrinsics=intrinsics[None, ...])
        depth = np.asarray(pred.depth[0], dtype=np.float32)
        out_npy = out_dir / f"{stem}.npy"
        np.save(out_npy, depth)
        return out_npy


class MoGeRuntime:
    def __init__(self, repo_root: Path, pretrained: str = "Ruicheng/moge-2-vitl-normal", fp16: bool = True):
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from moge.model.v2 import MoGeModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16
        self.model = MoGeModel.from_pretrained(pretrained).to(self.device).eval()
        if fp16:
            self.model = self.model.half()

    @torch.no_grad()
    def run(self, image_path: Path, out_dir: Path, stem: str = "demo", resolution_level: int = 7) -> Path:
        image_path = Path(image_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        output = self.model.infer(
            image_tensor,
            resolution_level=resolution_level,
            use_fp16=self.fp16,
        )
        depth = output["depth"].detach().float().cpu().numpy().astype(np.float32)
        out_npy = out_dir / f"{stem}.npy"
        np.save(out_npy, depth)
        save_depth_visualizations(
            out_npy,
            out_dir / f"{stem}_gray_near_black.png",
            out_dir / f"{stem}_color.png",
        )
        return out_npy


class DepthRuntimeCache:
    def __init__(self):
        self.da3 = None
        self.da3_key = None
        self.moge = None
        self.moge_key = None

    def run_da3(self, repo_root: Path, model_name: str, image_path: Path, out_dir: Path) -> Path:
        key = (str(Path(repo_root).resolve()), model_name)
        if self.da3 is None or self.da3_key != key:
            self.da3 = DA3Runtime(Path(repo_root), model_name)
            self.da3_key = key
        return self.da3.run(Path(image_path), Path(out_dir))

    def run_moge(self, repo_root: Path, image_path: Path, out_dir: Path, stem: str) -> Path:
        key = str(Path(repo_root).resolve())
        if self.moge is None or self.moge_key != key:
            self.moge = MoGeRuntime(Path(repo_root))
            self.moge_key = key
        return self.moge.run(Path(image_path), Path(out_dir), stem=stem)
