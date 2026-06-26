import numpy as np
import cv2
import re
import sys


def decode_3_channels(raw, max_depth=1000):
    raw = raw.astype(np.float32)
    out = raw[:, :, 2] + raw[:, :, 1] * 256 + raw[:, :, 0] * 256 * 256
    out = out / (256 * 256 * 256 - 1) * max_depth
    return out


def read_pfm(path):
    with open(path, "rb") as file:
        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        endian = "<" if scale < 0 else ">"
        scale = -scale if scale < 0 else scale

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale


def read_d(path, scale_factor=256.):
    if path.endswith("pfm"):
        d, _ = read_pfm(path)
        if d.ndim == 3:
            d = d[:, :, 0]
        return d
    if path.endswith("npy"):
        return np.load(path)
    if path.endswith("exr"):
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return d[:, :, 0] if d.ndim == 3 else d
    if path.endswith("png"):
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(path)
        if d.ndim == 3:
            d = decode_3_channels(d)
        elif d.dtype == np.uint16:
            d = d.astype(np.float32) / scale_factor
        return d
    # fallback grayscale
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    if d.ndim == 3:
        d = d[:, :, 0]
    return d


def compute_scale_and_shift(prediction, target, mask):
    # prediction/target/mask: (B,H,W)
    a_00 = np.sum(mask * prediction * prediction, axis=(1, 2))
    a_01 = np.sum(mask * prediction, axis=(1, 2))
    a_11 = np.sum(mask, axis=(1, 2))

    b_0 = np.sum(mask * prediction * target, axis=(1, 2))
    b_1 = np.sum(mask * target, axis=(1, 2))

    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    return x_0, x_1
