import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def _copy_binary_masks(raw_stem_dir: Path, out_dir: Path) -> None:
    mask_paths = sorted(raw_stem_dir.glob("mask_*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"Trans4Trans produced no mask_*.png files in {raw_stem_dir}")
    for idx, mask_path in enumerate(mask_paths):
        arr = np.array(Image.open(mask_path).convert("L"))
        binary = (arr > 0).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(out_dir / f"mask_{idx}.png")


def _write_runtime_config(config_path: Path, checkpoint: Path, out_dir: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Trans4Trans config is invalid: {config_path}")
    config.setdefault("TEST", {})["TEST_MODEL_PATH"] = str(checkpoint)
    runtime_config = out_dir / "trans4trans_runtime_config.yaml"
    with runtime_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/trans10kv2/pvt_medium_FPT.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    image_path = Path(args.image).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = repo_root / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"Trans4Trans config not found: {config_path}")

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else repo_root / "workdirs/trans10kv2/trans4trans_medium.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Trans4Trans checkpoint is missing. Expected "
            f"{checkpoint}. Put the medium Trans4Trans checkpoint there, "
            "or pass --checkpoint."
        )
    runtime_config = _write_runtime_config(config_path, checkpoint, out_dir)

    cmd = [
        sys.executable,
        str(repo_root / "demo.py"),
        "--config-file",
        str(runtime_config),
        "--input-img",
        str(image_path),
        "--output-dir",
        str(out_dir / "raw"),
    ]
    print("[RUN]", " ".join(cmd), flush=True)
    env = os.environ.copy()
    pythonpath = [
        str(Path(__file__).resolve().parent / "mmcv_stub"),
        str(repo_root),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join([p for p in pythonpath if p])
    subprocess.run(cmd, cwd=str(repo_root), check=True, env=env)

    raw_stem_dir = out_dir / "raw" / image_path.stem
    raw_copy_dir = out_dir / "trans4trans_raw"
    if raw_copy_dir.exists():
        shutil.rmtree(raw_copy_dir)
    shutil.copytree(raw_stem_dir, raw_copy_dir)
    _copy_binary_masks(raw_stem_dir, out_dir)
    print("[DONE]")
    print("  raw:", raw_copy_dir)
    print("  masks:", out_dir)


if __name__ == "__main__":
    main()
