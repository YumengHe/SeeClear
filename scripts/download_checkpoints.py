import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "2bidoubi/SeeClear-weights"

DEMO_CHECKPOINTS = {
    "seeclear_opacification.ckpt": 5_300_115_082,
    "mask_refiner.pth": 113_782,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download SeeClear demo checkpoints from Hugging Face."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face model repository ID.")
    parser.add_argument(
        "--output-dir",
        default="pretrained_models",
        help="Directory for downloaded checkpoints.",
    )
    parser.add_argument("--revision", default="main", help="Repository revision to download from.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    return parser.parse_args()


def should_skip(path: Path, expected_size: int, force: bool) -> bool:
    if force or not path.exists():
        return False
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        print(
            f"[WARN] {path} exists but has size {actual_size}; "
            f"expected {expected_size}. Use --force to replace it."
        )
    else:
        print(f"[SKIP] {path} already exists.")
    return True


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected_size in DEMO_CHECKPOINTS.items():
        target = output_dir / filename
        if should_skip(target, expected_size, args.force):
            continue
        print(f"[DOWNLOAD] {args.repo_id}/{filename} -> {target}")
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            repo_type="model",
            revision=args.revision,
            local_dir=output_dir,
            force_download=args.force,
        )
        downloaded = Path(path)
        actual_size = downloaded.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Downloaded {downloaded} has size {actual_size}; expected {expected_size}."
            )
        print(f"[OK] {downloaded}")


if __name__ == "__main__":
    main()
