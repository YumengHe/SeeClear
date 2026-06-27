import argparse
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/val/test filename lists for a SeeClear dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("dataset/my_data"),
        help="Dataset root containing the opaque image directory.",
    )
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=30)
    parser.add_argument("--test_size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source_dir",
        type=str,
        default="opaque",
        help="Subdirectory used to enumerate sample filenames.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional filename substring filter.",
    )
    return parser.parse_args()


def write_list(path, filenames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for filename in filenames:
            f.write(f"{filename}\n")


def main():
    args = parse_args()
    source_dir = args.data_dir / args.source_dir
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    filenames = sorted(
        path.name
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix in IMAGE_EXTENSIONS
    )
    if args.filter:
        filenames = [name for name in filenames if args.filter in name]

    requested = args.train_size + args.val_size + args.test_size
    if requested > len(filenames):
        raise ValueError(
            f"Requested {requested} samples, but only found {len(filenames)} in {source_dir}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(filenames)

    train_end = args.train_size
    val_end = train_end + args.val_size
    splits = {
        "train": filenames[:train_end],
        "val": filenames[train_end:val_end],
        "test": filenames[val_end:val_end + args.test_size],
    }

    for split_name, split_files in splits.items():
        output_path = args.data_dir / f"{split_name}_list.txt"
        write_list(output_path, split_files)
        print(f"{split_name}: {len(split_files)} -> {output_path}")

    print(f"seed: {args.seed}")
    print(f"source: {source_dir}")


if __name__ == "__main__":
    main()
