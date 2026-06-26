from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from demo.pipeline import (
    build_depth_command,
    build_evaluate_command,
    build_sam3_refine_command,
    build_trans4trans_command,
    collect_mask_paths,
    default_paths,
    expected_outputs,
    materialize_opaque_image,
    prepare_single_image_layout,
)
from demo.visualization import save_depth_visualizations


def write_commands(commands: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(cmd) for cmd in commands]
    out_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def run_command(cmd: list, cwd: Path) -> None:
    print("\n[RUN]", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _resolve_masks(args: argparse.Namespace, paths, image_path: Path, work_dir: Path) -> list[Path]:
    if args.mask_source in ("trans4trans", "trans4trans_sam3"):
        trans4trans_dir = work_dir / "trans4trans_mask"
        cmd = build_trans4trans_command(paths, image_path, trans4trans_dir)
        if args.trans4trans_checkpoint:
            cmd.extend(["--checkpoint", str(Path(args.trans4trans_checkpoint).resolve())])
        run_command(cmd, cwd=paths.repo_root)
        mask_paths = collect_mask_paths(trans4trans_dir)
        if args.mask_source == "trans4trans":
            return mask_paths
        return _refine_masks(paths, image_path, trans4trans_dir, work_dir, args.stem)

    if args.mask_source == "sam3":
        if not args.mask:
            raise ValueError(
                "--mask-source sam3 expects --mask pointing to the mask saved by the SAM3 click UI. "
                "Start it with: bash demo/run_sam3_mask_ui.sh"
            )
        return collect_mask_paths(Path(args.mask).resolve())

    if not args.mask:
        raise ValueError("--mask is required when --mask-source upload.")
    mask_paths = collect_mask_paths(Path(args.mask).resolve())
    if args.sam3_refine:
        return _refine_masks(paths, image_path, Path(args.mask).resolve(), work_dir, args.stem)
    return mask_paths


def _refine_masks(paths, image_path: Path, mask_input: Path, work_dir: Path, stem: str) -> list[Path]:
    refine_dir = work_dir / "sam3_refined_mask"
    cmd = build_sam3_refine_command(paths, image_path, mask_input, refine_dir, stem)
    run_command(cmd, cwd=paths.repo_root)
    return collect_mask_paths(refine_dir / "refined" / stem)


def _write_extra_depth_visuals(outputs: dict) -> None:
    if outputs["depth_npy"].is_file():
        save_depth_visualizations(outputs["depth_npy"], outputs["depth_gray"], outputs["depth_color"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument(
        "--mask",
        default=None,
        help="Mask png path or a directory containing png masks. Required for upload and SAM3 modes.",
    )
    parser.add_argument(
        "--mask-source",
        choices=["upload", "trans4trans", "trans4trans_sam3", "sam3"],
        default="upload",
        help="How to obtain the transparent-object mask.",
    )
    parser.add_argument(
        "--depth-source",
        choices=["da3", "moge"],
        default="da3",
        help="Depth backend for the optimized opaque image.",
    )
    parser.add_argument("--trans4trans-checkpoint", default=None)
    parser.add_argument(
        "--sam3-refine",
        action="store_true",
        help="Refine uploaded masks with SAM3 mask-input refinement before opaque generation.",
    )
    parser.add_argument("--work-dir", default="/nas/xiaoyingwang/seeclear/demo_runs/manual", help="Output workspace.")
    parser.add_argument("--stem", default="demo", help="Output sample id.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unipc-steps", type=int, default=10)
    parser.add_argument("--commands-txt", default=None)
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    work_dir = Path(args.work_dir).resolve()

    paths = default_paths()
    mask_paths = _resolve_masks(args, paths, image_path, work_dir)
    if not mask_paths:
        raise FileNotFoundError(f"No png mask found for mask source: {args.mask_source}")

    layout = prepare_single_image_layout(
        image_path=image_path,
        mask_paths=mask_paths,
        work_dir=work_dir,
        stem=args.stem,
    )
    evaluate_cmd = build_evaluate_command(paths, layout, seeds=[args.seed], unipc_steps=args.unipc_steps)
    depth_cmd = build_depth_command(paths, layout, args.depth_source)
    commands = [evaluate_cmd, depth_cmd]
    commands_txt = Path(args.commands_txt).resolve() if args.commands_txt else layout.output_base.parent / "commands.txt"
    write_commands(commands, commands_txt)

    print("[INFO] Commands written to:", commands_txt)
    run_command(evaluate_cmd, cwd=paths.repo_root)
    materialize_opaque_image(layout)
    run_command(depth_cmd, cwd=paths.repo_root)
    _write_extra_depth_visuals(expected_outputs(layout))

    print("\n[OUTPUTS]")
    for name, path in expected_outputs(layout).items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
