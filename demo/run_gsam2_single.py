from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


def _collect_output_masks(folder: Path) -> list[Path]:
    masks = sorted(folder.glob("mask_*.png"))
    if not masks:
        mask = folder / "mask.png"
        if mask.is_file():
            masks = [mask]
    return masks


def _normalize_object_prompt(prompt: str) -> str:
    terms = [term.strip().lower().rstrip(".") for term in re.split(r"[\n,;]+", prompt or "") if term.strip().strip(".")]
    if not terms:
        raise ValueError("Grounded-SAM2 prompt is empty.")
    return ". ".join(terms) + "."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Grounded-SAM2 mask generation case.")
    default_repo_root = Path(__file__).resolve().parents[1] / "third_party" / "grounded_sam2"
    parser.add_argument("--repo-root", default=str(default_repo_root))
    parser.add_argument("--script", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default="demo")
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--synonyms-per-object", type=int, default=1)
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--has-desk", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    script = (Path(args.script) if args.script else repo_root / "grounded_sam2_ttop.py").resolve()
    image_path = Path(args.image).resolve()
    out_dir = Path(args.out_dir).resolve()
    input_dir = out_dir / "_input"
    raw_out = out_dir / "raw"

    if not repo_root.is_dir():
        raise FileNotFoundError(f"Grounded-SAM2 repo not found: {repo_root}")
    if not script.is_file():
        raise FileNotFoundError(f"Grounded-SAM2 script not found: {script}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    input_dir.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)
    staged_image = input_dir / f"1{image_path.suffix.lower() or '.png'}"
    shutil.copy2(image_path, staged_image)
    prompts_file = out_dir / "prompts.txt"

    ckpt = Path(args.sam2_checkpoint).resolve() if args.sam2_checkpoint else repo_root / "checkpoints/sam2.1_hiera_large.pt"
    cfg = repo_root / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    if not ckpt.is_file():
        raise FileNotFoundError(f"SAM2.1 checkpoint not found: {ckpt}")
    if not cfg.is_file():
        raise FileNotFoundError(f"SAM2.1 config not found: {cfg}")

    prompt = _normalize_object_prompt(args.prompt)
    prompts_file.write_text(prompt + "\n", encoding="utf-8")

    cmd = [
        os.sys.executable,
        str(script),
        "--input-dir",
        str(input_dir),
        "--prompts-file",
        str(prompts_file),
        "--sam2-checkpoint",
        str(ckpt),
        "--sam2-model-config",
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "--output-dir",
        str(raw_out),
        "--box-threshold",
        str(args.box_threshold),
        "--synonyms-per-object",
        str(args.synonyms_per_object),
    ]
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.has_desk:
        cmd.append("--has-desk")

    env = os.environ.copy()
    if "SEECLEAR_CACHE_DIR" in env:
        cache_root = Path(env["SEECLEAR_CACHE_DIR"])
    elif "GRADIO_TEMP_DIR" in env:
        cache_root = Path(env["GRADIO_TEMP_DIR"]).parent / "cache"
    else:
        cache_root = out_dir.parents[1] / "cache" if len(out_dir.parents) > 1 else out_dir.parent / "cache"
    env.setdefault("HF_HOME", str(cache_root / "hf_home"))
    env.setdefault("TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions"))
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), str(script.parent), env.get("PYTHONPATH", "")]
    )
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_root), check=True, env=env)

    raw_stem_dir = raw_out / "1"
    masks = _collect_output_masks(raw_stem_dir)
    if not masks:
        raise FileNotFoundError(f"Grounded-SAM2 produced no masks in {raw_stem_dir}")

    instances_dir = out_dir / "instances"
    if instances_dir.exists():
        shutil.rmtree(instances_dir)
    instances_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask_path in enumerate(masks):
        shutil.copy2(mask_path, instances_dir / f"mask_{idx}.png")
        shutil.copy2(mask_path, out_dir / f"mask_{idx}.png")

    annotated = raw_stem_dir / "annotated.jpg"
    if annotated.is_file():
        shutil.copy2(annotated, out_dir / "annotated.jpg")
    results = raw_stem_dir / "results.json"
    if results.is_file():
        shutil.copy2(results, out_dir / "results.json")

    print("[DONE]")
    print("  masks:", instances_dir)
    print("  raw:", raw_stem_dir)


if __name__ == "__main__":
    main()
