import shutil
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DA3_ROOT = REPO_ROOT / "third_party" / "depth_anything_3"


@dataclass(frozen=True)
class DemoPaths:
    repo_root: Path
    da3_root: Path
    trans4trans_root: Path
    gsam2_root: Path
    moge_root: Path
    python: Path
    pretrained_root: Path
    opacification_ckpt: Path
    config: Path
    mask_refiner_path: Path
    depth_model: str


@dataclass(frozen=True)
class PreparedLayout:
    rgb_dir: Path
    mask_dir: Path
    output_base: Path
    depth_dir: Path
    stem: str


def default_paths() -> DemoPaths:
    current_python = Path(sys.executable)
    return DemoPaths(
        repo_root=REPO_ROOT,
        da3_root=DA3_ROOT,
        trans4trans_root=REPO_ROOT / "third_party" / "trans4trans",
        gsam2_root=REPO_ROOT / "third_party" / "grounded_sam2",
        moge_root=REPO_ROOT / "third_party" / "moge",
        python=current_python,
        pretrained_root=REPO_ROOT / "pretrained_models",
        opacification_ckpt=REPO_ROOT / "pretrained_models" / "seeclear_opacification.ckpt",
        config=REPO_ROOT / "configs" / "opacification_inference.yaml",
        mask_refiner_path=REPO_ROOT / "pretrained_models" / "mask_refiner.pth",
        depth_model="depth-anything/DA3-GIANT-1.1",
    )


def prepare_single_image_layout(
    image_path: Union[Path, str],
    mask_paths: Iterable[Union[Path, str]],
    work_dir: Union[Path, str],
    stem: str = "demo",
) -> PreparedLayout:
    image_path = Path(image_path)
    work_dir = Path(work_dir)
    rgb_dir = work_dir / "rgb"
    mask_dir = work_dir / "mask"
    output_base = work_dir / "opaque"
    depth_dir = work_dir / "depth"
    image_ext = image_path.suffix.lower() or ".png"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    stem_mask_dir = mask_dir / stem
    stem_mask_dir.mkdir(parents=True, exist_ok=True)
    output_base.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, rgb_dir / f"{stem}{image_ext}")
    for idx, mask_path in enumerate(mask_paths):
        shutil.copy2(Path(mask_path), stem_mask_dir / f"mask_{idx}.png")

    return PreparedLayout(
        rgb_dir=rgb_dir,
        mask_dir=mask_dir,
        output_base=output_base,
        depth_dir=depth_dir,
        stem=stem,
    )


def collect_mask_paths(mask_input: Union[Path, str]) -> list:
    path = Path(mask_input)
    if path.is_dir():
        masks = sorted(path.glob("*.png"))
    else:
        masks = [path]
    return [p for p in masks if p.is_file()]


def result_image_path(layout: PreparedLayout) -> Path:
    return layout.output_base.with_name(f"{layout.output_base.name}_0") / "blend" / (
        f"{layout.stem}_result.png"
    )


def materialize_opaque_image(layout: PreparedLayout) -> Path:
    png_path = result_image_path(layout)
    npy_path = png_path.with_suffix(".npy")
    if not png_path.exists() and npy_path.exists():
        Image.fromarray(np.load(npy_path)).save(png_path)
    return png_path


def build_evaluate_command(
    paths: DemoPaths,
    layout: PreparedLayout,
    seeds: Sequence[int] = (42,),
    unipc_steps: int = 10,
    batch_size: int = 8,
) -> list:
    seed_args = [str(seed) for seed in seeds]
    return [
        str(paths.python),
        str(paths.repo_root / "scripts" / "infer_opacification.py"),
        "--opacification_ckpt",
        str(paths.opacification_ckpt),
        "--config",
        str(paths.config),
        "--rgb_dir",
        str(layout.rgb_dir),
        "--mask_dir",
        str(layout.mask_dir),
        "--output_base",
        str(layout.output_base),
        "--mask_refiner_path",
        str(paths.mask_refiner_path),
        "--unipc_steps",
        str(unipc_steps),
        "--seeds",
        *seed_args,
        "--batch_size",
        str(batch_size),
        "--prep_mode",
        "fast",
    ]


def build_da3_command(paths: DemoPaths, layout: PreparedLayout) -> list:
    return [
        str(paths.python),
        str(paths.da3_root / "infer_single.py"),
        "--img",
        str(result_image_path(layout)),
        "--out_dir",
        str(layout.depth_dir),
        "--model",
        paths.depth_model,
    ]


def build_commands(
    paths: DemoPaths,
    layout: PreparedLayout,
    seeds: Sequence[int] = (42,),
    unipc_steps: int = 10,
) -> list:
    evaluate_cmd = build_evaluate_command(paths, layout, seeds=seeds, unipc_steps=unipc_steps)
    depth_cmd = build_da3_command(paths, layout)
    return [evaluate_cmd, depth_cmd]


def build_trans4trans_command(paths: DemoPaths, image_path: Path, out_dir: Path) -> list:
    return [
        str(paths.python),
        str(paths.repo_root / "demo" / "run_trans4trans_single.py"),
        "--repo-root",
        str(paths.trans4trans_root),
        "--image",
        str(image_path),
        "--out-dir",
        str(out_dir),
        "--config",
        "configs/trans10kv2/pvt_medium_FPT.yaml",
        "--checkpoint",
        str(paths.pretrained_root / "trans4trans_medium.pth"),
    ]


def build_sam3_refine_command(
    paths: DemoPaths,
    image_path: Path,
    mask_input: Path,
    out_dir: Path,
    stem: str,
) -> list:
    return [
        str(paths.python),
        str(paths.repo_root / "demo" / "run_sam3_refine_single.py"),
        "--image",
        str(image_path),
        "--mask",
        str(mask_input),
        "--out-dir",
        str(out_dir),
        "--stem",
        stem,
    ]


def build_gsam2_command(
    paths: DemoPaths,
    image_path: Path,
    prompt: str,
    out_dir: Path,
    stem: str,
    synonyms_per_object: int = 1,
    box_threshold: float = 0.35,
) -> list:
    return [
        str(paths.python),
        str(paths.repo_root / "demo" / "run_gsam2_single.py"),
        "--repo-root",
        str(paths.gsam2_root),
        "--script",
        str(paths.gsam2_root / "grounded_sam2_ttop.py"),
        "--image",
        str(image_path),
        "--prompt",
        prompt,
        "--out-dir",
        str(out_dir),
        "--stem",
        stem,
        "--synonyms-per-object",
        str(synonyms_per_object),
        "--box-threshold",
        str(box_threshold),
        "--sam2-checkpoint",
        str(paths.pretrained_root / "sam2.1_hiera_large.pt"),
    ]


def build_moge_command(
    paths: DemoPaths,
    layout: PreparedLayout,
    pretrained: str = "Ruicheng/moge-2-vitl-normal",
    resolution_level: int = 7,
    fp16: bool = True,
) -> list:
    cmd = [
        str(paths.python),
        str(paths.repo_root / "demo" / "run_moge_single.py"),
        "--moge-root",
        str(paths.moge_root),
        "--image",
        str(result_image_path(layout)),
        "--out-dir",
        str(layout.depth_dir),
        "--stem",
        layout.stem,
        "--pretrained",
        pretrained,
        "--resolution-level",
        str(resolution_level),
    ]
    if fp16:
        cmd.append("--fp16")
    return cmd


def build_baseline_depth_command(
    paths: DemoPaths,
    image_path: Path,
    out_dir: Path,
    stem: str,
    source: str,
) -> list:
    """Run depth model on the original RGB image (baseline, no opaque processing)."""
    if source == "da3":
        return [
            str(paths.python),
            str(paths.da3_root / "infer_single.py"),
            "--img", str(image_path),
            "--out_dir", str(out_dir),
            "--model", paths.depth_model,
        ]
    if source == "moge":
        cmd = [
            str(paths.python),
            str(paths.repo_root / "demo" / "run_moge_single.py"),
            "--moge-root", str(paths.moge_root),
            "--image", str(image_path),
            "--out-dir", str(out_dir),
            "--stem", stem,
            "--pretrained", "Ruicheng/moge-2-vitl-normal",
            "--resolution-level", "7",
            "--fp16",
        ]
        return cmd
    raise ValueError(f"Unsupported depth source: {source}")


def build_depth_command(paths: DemoPaths, layout: PreparedLayout, source: str) -> list:
    if source == "da3":
        return build_da3_command(paths, layout)
    if source == "moge":
        return build_moge_command(paths, layout)
    raise ValueError(f"Unsupported depth source: {source}")


def expected_outputs(layout: PreparedLayout) -> dict:
    out0 = layout.output_base.with_name(f"{layout.output_base.name}_0")
    return {
        "direct_opaque": out0 / "mask_blend" / f"{layout.stem}_result.png",
        "optimized_opaque": out0 / "blend" / f"{layout.stem}_result.png",
        "pred_mask": out0 / "pred_mask" / layout.stem / "union.png",
        "depth_vis": layout.depth_dir / f"{layout.stem}_vis.png",
        "depth_gray": layout.depth_dir / f"{layout.stem}_gray_near_black.png",
        "depth_color": layout.depth_dir / f"{layout.stem}_color.png",
        "depth_npy": layout.depth_dir / f"{layout.stem}.npy",
    }
