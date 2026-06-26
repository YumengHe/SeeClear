from pathlib import Path

from colorize_depth import load_depth_image, normalize_depth, apply_colormap
import cv2


# Root folder to search recursively
OUTPUTS_DIR = Path("outputs")


def process_png(png_path: Path) -> None:
    output_path = png_path.with_name(f"{png_path.stem}_color.png")

    depth = load_depth_image(str(png_path))
    norm = normalize_depth(depth)
    color = apply_colormap(norm)

    success = cv2.imwrite(str(output_path), color)
    if not success:
        raise IOError(f"Failed to save output image: {output_path}")

    print(f"Saved: {output_path}")


def main() -> None:
    if not OUTPUTS_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {OUTPUTS_DIR}")

    png_files = [
        p for p in OUTPUTS_DIR.rglob("*.png")
        if not p.name.endswith("_color.png")
    ]

    if not png_files:
        print(f"No PNG files found under: {OUTPUTS_DIR}")
        return

    for png_path in png_files:
        try:
            process_png(png_path)
        except Exception as e:
            print(f"Failed on {png_path}: {e}")


if __name__ == "__main__":
    main()
