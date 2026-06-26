import os
import glob
import argparse
import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images(img_dir: str):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))

    paths = sorted(set(paths))
    return paths

def read_union_mask(mask_dir: str, target_hw):
    """
    mask_dir: ./xxx_out/<basename>/
    Unions all mask_*.png files.
    Returns a uint8 union mask with values 0/255 and shape (H, W), or None if no masks exist.
    """
    H, W = target_hw
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "mask_*.png")))
    if not mask_paths:
        return None

    union = np.zeros((H, W), dtype=np.uint8)
    ok = False

    for mp in mask_paths:
        m = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue


        if m.ndim == 3:
            m = m[:, :, 0]


        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        union = np.maximum(union, (m > 0).astype(np.uint8) * 255)
        ok = True

    return union if ok else None

def paint_solid_color(img_bgr: np.ndarray, mask_bool: np.ndarray, color_bgr):
    out = img_bgr.copy()
    out[mask_bool] = color_bgr
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="dataset folder name, e.g. cleargrasp / clearpose / xxx")
    parser.add_argument("--root", default=None,
                        help="project root. default: directory of this script")
    parser.add_argument("--out_prefix", default="mask_vis_rgb",
                        help="output folder prefix (default: mask_vis_rgb)")
    parser.add_argument("--keep_if_no_mask", action="store_true",
                        help="if set, images without masks will still be copied to output folders")
    args = parser.parse_args()

    root = args.root if args.root else os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(root, args.name)
    mask_root = os.path.join(root, f"{args.name}_out")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"Mask root not found: {mask_root} (expected '{args.name}_out')")

    out_root = os.path.join(root, f"{args.out_prefix}_{args.name}")
    out_red = os.path.join(out_root, "red")
    out_green = os.path.join(out_root, "green")
    out_blue = os.path.join(out_root, "blue")
    ensure_dir(out_red); ensure_dir(out_green); ensure_dir(out_blue)

    img_paths = list_images(img_dir)
    if not img_paths:
        raise FileNotFoundError(f"No jpg/jpeg/png images found in: {img_dir}")

    # OpenCV BGR
    COLOR_RED   = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE  = (255, 0, 0)

    miss_mask = 0
    processed = 0

    for ip in tqdm(img_paths, desc=f"Processing {args.name}"):
        base = os.path.splitext(os.path.basename(ip))[0]
        mask_dir = os.path.join(mask_root, base)

        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] failed to read image: {ip}")
            continue

        H, W = img.shape[:2]
        union_mask = read_union_mask(mask_dir, (H, W))

        if union_mask is None:
            miss_mask += 1
            if not args.keep_if_no_mask:
                continue

            out_name = f"{base}.png"
            cv2.imwrite(os.path.join(out_red, out_name), img)
            cv2.imwrite(os.path.join(out_green, out_name), img)
            cv2.imwrite(os.path.join(out_blue, out_name), img)
            processed += 1
            continue

        m = (union_mask > 0)

        red_img = paint_solid_color(img, m, COLOR_RED)
        green_img = paint_solid_color(img, m, COLOR_GREEN)
        blue_img = paint_solid_color(img, m, COLOR_BLUE)

        out_name = f"{base}.png"
        cv2.imwrite(os.path.join(out_red, out_name), red_img)
        cv2.imwrite(os.path.join(out_green, out_name), green_img)
        cv2.imwrite(os.path.join(out_blue, out_name), blue_img)
        processed += 1

    print(f"Done: dataset={args.name}")
    print(f"  images_found: {len(img_paths)}")
    print(f"  processed:    {processed}")
    print(f"  missing_mask: {miss_mask} (mask dir missing or no mask_*.png)")
    print(f"  output:       {out_root}")

if __name__ == "__main__":
    main()
