import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as T
import numpy as np
import cv2
import random




def is_thin_mask(mask, args):
    """
    thin mask ()
    """
    thin_ratio_threshold = args.get('thin_ratio_threshold', 0.50)
    
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return False
    
    mask_area = len(xs)
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
    
    if bbox_area == 0:
        return False
    
    area_ratio = mask_area / float(bbox_area)
    if area_ratio < thin_ratio_threshold:
        return True

    check_ksize = 5 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (check_ksize, check_ksize))
    
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    opened_area = np.sum(opened_mask > 127)
    lost_area = mask_area - opened_area
    
    loss_ratio = lost_area / float(mask_area)
    
    if loss_ratio > 0.015: 
        return True
        
    return False


def get_mask_bbox(mask):
    """Helper function."""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w-1, h-1
    return xs.min(), ys.min(), xs.max(), ys.max()


def protect_thin_mask(mask, args):
    """
    thin mask
    """
    kernel_size = args.get('thin_protect_ksize', 5)
    iterations = args.get('thin_protect_iter', 4)
    
    if iterations > 2:
        iterations = 2 
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    protected = cv2.dilate(mask, kernel, iterations=iterations)
    return protected


def apply_smooth_displacement(mask, args):
    """
    /
    """
    strength = args.get('disp_strength', 5.0)
    grid_res = args.get('disp_grid_res', 30)
    
    h, w = mask.shape
    
    dx = np.random.randn(h // grid_res, w // grid_res).astype(np.float32) * strength
    dy = np.random.randn(h // grid_res, w // grid_res).astype(np.float32) * strength
    
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC)
    
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    safe_radius = 6.0 
    
    weight_map = dist_transform / safe_radius
    weight_map = np.clip(weight_map, 0.1, 1.0)
    
    dx = dx * weight_map
    dy = dy * weight_map
    # =================================================
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    displaced = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    _, displaced = cv2.threshold(displaced, 127, 255, cv2.THRESH_BINARY)
    
    return displaced


def get_boundary_band(mask, args):
    """
    mask
    """
    inner_width = args.get('geo_band_inner', 5)
    outer_width = args.get('geo_band_outer', 20)
    
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_width, outer_width))
    outer_band = cv2.dilate(mask, kernel_outer, iterations=1)
    
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_width, inner_width))
    inner_band = cv2.erode(mask, kernel_inner, iterations=1)
    
    boundary_band = cv2.subtract(outer_band, inner_band)
    
    return boundary_band


def add_geometric_shapes_on_boundary(mask, args):
    """
    
    """
    num_shapes = args.get('geo_num_shapes', 4)
    size_ratio_range = args.get('geo_size_ratio_range', (0.06, 0.15))
    
    h, w = mask.shape
    result = mask.copy()
    
    x_min, y_min, x_max, y_max = get_mask_bbox(mask)
    bbox_long_side = max(x_max - x_min, y_max - y_min)
    
    if bbox_long_side < 20:
        return result
    
    band_outer = random.randint(15, 25) 
    boundary_band = get_boundary_band(mask, {'geo_band_inner': 5, 'geo_band_outer': band_outer})
    band_ys, band_xs = np.where(boundary_band > 127)
    
    if len(band_xs) < 100:
        return result
    
    for _ in range(num_shapes):
        idx = random.randint(0, len(band_xs) - 1)
        cx, cy = int(band_xs[idx]), int(band_ys[idx])
        
        size_ratio = random.uniform(*size_ratio_range)
        radius = int(bbox_long_side * size_ratio)
        
        shape_mask = np.zeros((h, w), dtype=np.uint8)
        shape_type = random.choice(['circle', 'rectangle'])
        
        if shape_type == 'circle':
            cv2.circle(shape_mask, (cx, cy), radius, 255, -1)
        else:
            angle = random.uniform(-45, 45)
            rect_w = int(radius * random.uniform(1.5, 2.0))
            rect_h = int(radius * random.uniform(0.8, 1.2))
            rect = ((float(cx), float(cy)), (float(rect_w), float(rect_h)), float(angle))
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.fillPoly(shape_mask, [box], 255)
        
        if random.random() < 0.5:
            result = cv2.bitwise_or(result, shape_mask)
        else:
            result = cv2.bitwise_and(result, cv2.bitwise_not(shape_mask))
    
    return result


def keep_largest_component(mask):
    """
    
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        return mask
    
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result = ((labels == largest_label) * 255).astype(np.uint8)

    return result


def validate_augmented_mask(aug_mask, gt_mask, is_thin=False, args=None):
    """
    mask
    """
    iou_threshold = args.get('val_iou_threshold', 0.40)
    
    aug_mask = keep_largest_component(aug_mask)
    
    aug_area = np.sum(aug_mask > 127)
    if aug_area == 0:
        return False, None
    
    intersection = np.sum(cv2.bitwise_and(aug_mask, gt_mask) > 127)
    union = np.sum(cv2.bitwise_or(aug_mask, gt_mask) > 127)
    
    if union == 0:
        return False, None
    
    iou = intersection / float(union)
    
    if iou < iou_threshold:
        return False, None
    
    return True, aug_mask


def scheme1_pure_morphology(mask, is_thin=False, args=None):
    """
    Scheme 1: 
    """
    result = mask.copy()
    
    ksize_range = args.get('s1_ksize_range', (7, 11))
    sigma_range = args.get('s1_blur_sigma_range', (1.5, 3.0))
    open_ksize = args.get('s1_open_ksize', 5)
    
    if is_thin:
        ksize_range = (max(3, ksize_range[0] // 2), max(3, ksize_range[1] // 2))
        sigma_range = (sigma_range[0] / 2.0, sigma_range[1] / 2.0)
        open_ksize = max(3, open_ksize // 2)
    
    ksize = random.choice([k for k in range(ksize_range[0], ksize_range[1]+1, 2) if k > 0])
    iterations = random.randint(1, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    
    if random.random() < 0.5:
        result = cv2.erode(result, kernel, iterations=iterations)
    else:
        result = cv2.dilate(result, kernel, iterations=iterations)
    
    sigma = random.uniform(*sigma_range)
    ksize_blur = int(sigma * 3) * 2 + 1
    blurred = cv2.GaussianBlur(result.astype(np.float32), (ksize_blur, ksize_blur), sigma)
    _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    result = result.astype(np.uint8)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_open)
    
    return result


def scheme2_pure_displacement(mask, is_thin=False, args=None):
    """
    Scheme 2:  ( Thin Mask  /2.0)
    """
    result = mask.copy()
    
    strength_range = args.get('s2_disp_strength_range', (5.0, 10.0))
    close_ksize = args.get('s2_close_ksize', 7)
    
    if is_thin:
        strength_range = (strength_range[0] / 2.0, strength_range[1] / 2.0)
        close_ksize = max(3, close_ksize // 2)
    # ===================================
    
    strength = random.uniform(*strength_range)
    result = apply_smooth_displacement(result, {'disp_strength': strength, 'disp_grid_res': 30})
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    blurred = cv2.GaussianBlur(result.astype(np.float32), (5, 5), 1.5)
    _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    result = result.astype(np.uint8)
    
    return result


def scheme3_pure_geometry(mask, is_thin=False, args=None):
    """
    Scheme 3: 
    """
    result = mask.copy()
    
    num_shapes_range = args.get('s3_num_shapes_range', (4, 6))
    size_range = args.get('s3_size_ratio_range', (0.06, 0.15))
    open_ksize = args.get('s3_open_ksize', 3)
    
    if is_thin:
        num_shapes_range = (max(1, num_shapes_range[0] // 2), max(1, num_shapes_range[1] // 2))
        size_range = (size_range[0] / 2.0, size_range[1] / 2.0)
        open_ksize = max(3, open_ksize // 2)

    num_shapes = random.randint(*num_shapes_range)
    
    result = add_geometric_shapes_on_boundary(result, {'geo_num_shapes': num_shapes, 'geo_size_ratio_range': size_range})
    
    result = keep_largest_component(result)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    blurred = cv2.GaussianBlur(result.astype(np.float32), (3, 3), 1.0)
    _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    result = result.astype(np.uint8)
    
    return result


def scheme4_low_res_upsample(mask, is_thin=False, args=None):
    """
    Scheme 4: Downsample/Upsample
    """
    h, w = mask.shape
    
    downsample_ratio_range = args.get('s4_downsample_ratio_range', (16, 32))
    
    if is_thin:
        downsample_ratio_range = (downsample_ratio_range[0] // 2, downsample_ratio_range[1] // 2)
        downsample_ratio_range = (max(8, downsample_ratio_range[0]), max(16, downsample_ratio_range[1])) 

    ratio = random.randint(*downsample_ratio_range)
    
    new_h, new_w = h // ratio, w // ratio
    if new_h < 8 or new_w < 8:
        new_h, new_w = h // 8, w // 8
    
    low_res_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    result = cv2.resize(low_res_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
    
    return result


def augment_mask_scheme(mask, scheme=1, args=None):
    """
    mask,().
    """
    MAX_ATTEMPTS = 50
    
    is_thin = is_thin_mask(mask, args)
    
    original_mask = mask.copy()
    
    if is_thin:
        mask_base = protect_thin_mask(mask, args)
    else:
        mask_base = mask.copy()
    
    if scheme == 4:
        augmented = scheme4_low_res_upsample(mask_base, is_thin, args)
        return keep_largest_component(augmented) 

    attempt = 0
    augmented = mask_base.copy()
    is_valid = False
    
    while attempt < MAX_ATTEMPTS:
        if scheme == 1:
            augmented = scheme1_pure_morphology(mask_base, is_thin, args)
        elif scheme == 2:
            augmented = scheme2_pure_displacement(mask_base, is_thin, args)
        elif scheme == 3:
            augmented = scheme3_pure_geometry(mask_base, is_thin, args)
        
        is_valid, validated = validate_augmented_mask(augmented, original_mask, is_thin, args)
        
        if is_valid:
            return validated
        
        attempt += 1
    
    if augmented is not None:
        final_result = keep_largest_component(augmented)
        if np.sum(final_result > 127) == 0:
            return original_mask
        return final_result
    
    return original_mask

# ============= Occluder Pool & Helpers (Added from Reference) =============
class OccluderPool:
    """Helper function."""
    def __init__(self, data_dir, image_files, max_pool_size=500):
        self.data_dir = data_dir
        pool_files = random.sample(image_files, min(max_pool_size, len(image_files)))
        self.occluders = []
        
        for img_name in pool_files:
            try:
                occ_img_path = os.path.join(data_dir, 'opaque', img_name)
                occ_mask_path = os.path.join(data_dir, 'mask', img_name)
                if os.path.exists(occ_img_path) and os.path.exists(occ_mask_path):
                    occ_img = np.array(Image.open(occ_img_path).convert('RGB'))
                    occ_mask = np.array(Image.open(occ_mask_path).convert('L'))
                    _, occ_mask = cv2.threshold(occ_mask, 127, 255, cv2.THRESH_BINARY)
                    if np.sum(occ_mask > 127) > 1000:
                        self.occluders.append((occ_img, occ_mask, img_name))
            except: pass
    
    def get_random_occluder(self):
        if len(self.occluders) == 0: return None, None, None
        idx = random.randint(0, len(self.occluders) - 1)
        return self.occluders[idx]
    
def create_occluder_patch(occ_img, occ_mask, area_ratio=(0.2, 0.6)):
    ys, xs = np.where(occ_mask > 127)
    if len(xs) == 0: return None, None
    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    padding = 5
    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
    x_max = min(occ_img.shape[1] - 1, x_max + padding)
    y_max = min(occ_img.shape[0] - 1, y_max + padding)
    crop_img = occ_img[y_min:y_max+1, x_min:x_max+1].copy()
    crop_mask = occ_mask[y_min:y_max+1, x_min:x_max+1].copy()
    h_crop, w_crop = crop_mask.shape
    target_area = h_crop * w_crop * random.uniform(*area_ratio)
    aspect_ratio = random.uniform(0.5, 2.0)
    patch_h = min(int(np.sqrt(target_area / aspect_ratio)), h_crop)
    patch_w = min(int(patch_h * aspect_ratio), w_crop)
    if patch_h >= h_crop or patch_w >= w_crop:
        patch_img, patch_mask = crop_img, crop_mask
    else:
        y_start = random.randint(0, h_crop - patch_h)
        x_start = random.randint(0, w_crop - patch_w)
        rect_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
        rect_mask[y_start:y_start+patch_h, x_start:x_start+patch_w] = 255
        patch_mask = cv2.bitwise_and(crop_mask, rect_mask)
        patch_img = crop_img.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(patch_mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        patch_mask = (labels == largest_label).astype(np.uint8) * 255
    if np.sum(patch_mask > 127) < 100: return None, None
    return patch_img, patch_mask

def transform_occluder(occ_img, occ_mask, 
                       scale_range=(0.6, 1.4), 
                       rotation_range=(-15, 15), 
                       hue_shift_range=(0, 180),
                       saturation_range=(0.7, 1.5),
                       brightness_range=(0.7, 1.3)):
    h, w = occ_mask.shape
    scale = random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)
    scaled_img = cv2.resize(occ_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(occ_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    angle = random.uniform(*rotation_range)
    center = (new_w // 2, new_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w_rot = int((new_h * sin) + (new_w * cos))
    new_h_rot = int((new_h * cos) + (new_w * sin))
    M[0, 2] += (new_w_rot / 2) - center[0]
    M[1, 2] += (new_h_rot / 2) - center[1]
    rotated_img = cv2.warpAffine(scaled_img, M, (new_w_rot, new_h_rot), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rotated_mask = cv2.warpAffine(scaled_mask, M, (new_w_rot, new_h_rot), flags=cv2.INTER_NEAREST, borderValue=0)
    hsv = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    hue_shift = random.uniform(*hue_shift_range)
    h = (h + hue_shift) % 180
    saturation_multiplier = random.uniform(*saturation_range)
    s = np.clip(s * saturation_multiplier, 0, 255)
    brightness_multiplier = random.uniform(*brightness_range)
    v = np.clip(v * brightness_multiplier, 0, 255)
    final_hsv = cv2.merge([h, s, v]).astype(np.uint8)
    color_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return color_img, rotated_mask

def place_occluder(img_h, img_w, mask_obj, occ_mask, overlap_range=(0.1, 0.8), max_tries=10):
    occ_h, occ_w = occ_mask.shape
    mask_area = np.sum(mask_obj > 127)
    if mask_area == 0: return None, None
    ys, xs = np.where(mask_obj > 127)
    for _ in range(max_tries):
        idx = random.randint(0, len(xs) - 1)
        center_x, center_y = int(xs[idx]), int(ys[idx])
        x_start, y_start = center_x - occ_w // 2, center_y - occ_h // 2
        x_end, y_end = x_start + occ_w, y_start + occ_h
        if x_start < 0 or y_start < 0 or x_end > img_w or y_end > img_h: continue
        placed_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        placed_mask[y_start:y_end, x_start:x_end] = occ_mask
        overlap = np.sum(cv2.bitwise_and(placed_mask, mask_obj) > 127)
        overlap_ratio = overlap / float(mask_area)
        if overlap_range[0] <= overlap_ratio <= overlap_range[1]:
            return center_x, center_y
    return None, None

def blend_occluder(img, occ_img, occ_mask, center_x, center_y, feather_sigma=1.5):
    img_h, img_w = img.shape[:2]
    occ_h, occ_w = occ_mask.shape
    x_start, y_start = center_x - occ_w // 2, center_y - occ_h // 2
    x_end, y_end = x_start + occ_w, y_start + occ_h
    x_start, y_start = max(0, x_start), max(0, y_start)
    x_end, y_end = min(img_w, x_end), min(img_h, y_end)
    actual_w, actual_h = x_end - x_start, y_end - y_start
    if actual_w <= 0 or actual_h <= 0: return img, np.zeros((img_h, img_w), dtype=np.uint8)
    occ_x_start = 0 if x_start >= 0 else -x_start
    occ_y_start = 0 if y_start >= 0 else -y_start
    occ_img_crop = occ_img[occ_y_start:occ_y_start+actual_h, occ_x_start:occ_x_start+actual_w]
    occ_mask_crop = occ_mask[occ_y_start:occ_y_start+actual_h, occ_x_start:occ_x_start+actual_w]
    alpha = occ_mask_crop.astype(np.float32) / 255.0
    ksize = int(feather_sigma * 4) | 1
    alpha = cv2.GaussianBlur(alpha, (ksize, ksize), feather_sigma)
    alpha = np.clip(alpha, 0, 1)[:, :, np.newaxis]
    blended_img = img.copy()
    roi = blended_img[y_start:y_end, x_start:x_end]
    blended_roi = (1 - alpha) * roi.astype(np.float32) + alpha * occ_img_crop.astype(np.float32)
    blended_img[y_start:y_end, x_start:x_end] = blended_roi.astype(np.uint8)
    placed_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    placed_mask[y_start:y_end, x_start:x_end] = occ_mask_crop
    return blended_img, placed_mask

def apply_realistic_occlusion(img_list, mask_gt, occluder_pool, args):
    occ_img, occ_mask, _ = occluder_pool.get_random_occluder()
    if occ_img is None: return None, mask_gt
    ys, xs = np.where(occ_mask > 127)
    if len(xs) == 0: return None, mask_gt
    padding = 5
    y_min = max(0, ys.min() - padding)
    y_max = min(occ_img.shape[0] - 1, ys.max() + padding)
    x_min = max(0, xs.min() - padding)
    x_max = min(occ_img.shape[1] - 1, xs.max() + padding)
    patch_img  = occ_img[y_min:y_max+1, x_min:x_max+1].copy()
    patch_mask = occ_mask[y_min:y_max+1, x_min:x_max+1].copy()
    if np.sum(patch_mask > 127) < 100: return None, mask_gt
    transformed_img, transformed_mask = transform_occluder(
        patch_img, patch_mask, 
        scale_range=args.get('occ_scale_range', (0.6, 1.4)), 
        rotation_range=args.get('occ_rotation_range', (-15, 15)), 
        hue_shift_range=args.get('occ_hue_shift_range', (0, 180)),
        saturation_range=args.get('occ_saturation_range', (0.7, 1.5)), 
        brightness_range=args.get('occ_brightness_range', (0.7, 1.3))
    )
    h, w = mask_gt.shape
    center_x, center_y = place_occluder(h, w, mask_gt, transformed_mask, 
                                        overlap_range=args.get('occ_overlap_range', (0.05, 0.30)), 
                                        max_tries=args.get('occ_max_tries', 10))
    if center_x is None: return None, mask_gt
    occluded_imgs = []
    placed_mask = None
    for img in img_list:
        occluded_img, current_placed_mask = blend_occluder(
            img, transformed_img, transformed_mask, center_x, center_y, args.get('occ_feather_sigma', 1.5)
        )
        occluded_imgs.append(occluded_img)
        if placed_mask is None:
            placed_mask = current_placed_mask
    occluded_mask = cv2.bitwise_and(mask_gt, cv2.bitwise_not(placed_mask))
    return occluded_imgs, occluded_mask

# Tensor conversion functions (unchanged)
def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [T.ToTensor()]
    if normalize:
        transform_list += [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return T.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [T.ToTensor()]
    if normalize:
        transform_list += [T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                       (0.26862954, 0.26130258, 0.27577711))]
    return T.Compose(transform_list)

def generate_heatmap(shape, center_point, sigma=10):
    """Helper function."""
    h, w = shape
    center_x, center_y = center_point
    y_grid, x_grid = np.ogrid[:h, :w]
    distances = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    heatmap = np.exp(-distances**2 / (2 * sigma**2))
    return heatmap.astype(np.float32)

class MyTransparentDataset(data.Dataset):
    def __init__(self, state, **args):
        self.args = args 
        self.data_dir = args['dataset_dir']
        self.state = state
        self.output_size = args.get('img_size', 512)
        self.heatmap_sigma = args.get('heatmap_sigma', 10)
        
        self.mask_strategy = 5
        self.mask_aug_scheme = args.get('mask_aug_scheme', None)
        
        self.condition_type = args.get('condition_type', 'mask_aug')
        
        list_file_name = f"{state}_list.txt"
        list_file = os.path.join(self.data_dir, list_file_name)
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                self.image_files = [line.strip() for line in f if line.strip()]
        else:
            opaque_dir = os.path.join(self.data_dir, 'opaque')
            self.image_files = sorted([f for f in os.listdir(opaque_dir) 
                                      if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        self.length = len(self.image_files)
        print(f"[{state}] condition_type={self.condition_type}, mask_aug_scheme={self.mask_aug_scheme}, heatmap_sigma={self.heatmap_sigma}")

        # === Occluder Pool Initialization (Added) ===
        if state == 'train' and args.get('use_realistic_occlusion', False):
            self.occluder_pool = OccluderPool(self.data_dir, self.image_files, 
                                              max_pool_size=args.get('occluder_pool_size', 500))
        else:
            self.occluder_pool = None
        # ============================================

    def __len__(self):
        return self.length

    def create_white_bg_direct(self, rgba_opaque, rgba_ref, mask_gt):
        target_size = (self.output_size, self.output_size)
        
        obj_opaque = cv2.resize(rgba_opaque, target_size, interpolation=cv2.INTER_LINEAR)
        obj_ref = cv2.resize(rgba_ref, target_size, interpolation=cv2.INTER_LINEAR)
        
        alpha_channel = obj_opaque[..., 3]
        obj_mask = (alpha_channel > 127).astype(np.uint8) * 255

        def composite_with_own_alpha(img_rgba):
            rgb = img_rgba[..., :3].astype(np.float32)
            alpha = img_rgba[..., 3].astype(np.float32) / 255.0
            alpha = alpha[..., np.newaxis]
            
            white_bg = np.ones_like(rgb) * 255.0
            
            blended = rgb * alpha + white_bg * (1.0 - alpha)
            return blended.astype(np.uint8)

        final_opaque = composite_with_own_alpha(obj_opaque)
        final_ref = composite_with_own_alpha(obj_ref)
        
        return final_opaque, final_ref, obj_mask

    def get_strategy_point(self, mask):
        h, w = mask.shape
        return w // 2, h // 2

    def __getitem__(self, index):
        resize_dim = (self.output_size, self.output_size)
        info = ""

        img_name = self.image_files[index]
        opaque_path = os.path.join(self.data_dir, 'opaque', img_name)
        ref_path = os.path.join(self.data_dir, 'transparent', img_name)
        mask_path = os.path.join(self.data_dir, 'mask', img_name)

        if not os.path.exists(opaque_path) or not os.path.exists(ref_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Failed to load image: {img_name}")

        x_opaque_np = cv2.imread(opaque_path)
        x_in_np = cv2.imread(ref_path)
        M_gt_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        x_opaque_np = cv2.cvtColor(x_opaque_np, cv2.COLOR_BGR2RGB)
        x_in_np = cv2.cvtColor(x_in_np, cv2.COLOR_BGR2RGB)
        x_opaque_np = cv2.resize(x_opaque_np, resize_dim, interpolation=cv2.INTER_LINEAR)
        x_in_np = cv2.resize(x_in_np, resize_dim, interpolation=cv2.INTER_LINEAR)
        M_gt_np = cv2.resize(M_gt_np, resize_dim, interpolation=cv2.INTER_NEAREST)
        _, M_gt_binary_np = cv2.threshold(M_gt_np, 127, 255, cv2.THRESH_BINARY)
        info = f"dataset_{self.mask_strategy}"

        # === Realistic Occlusion Logic (Added) ===
        if self.occluder_pool is not None and random.random() < self.args.get('realistic_occlusion_prob', 0.1):
            images_to_occlude = [x_opaque_np, x_in_np]
            occluded_results, occluded_mask = apply_realistic_occlusion(
                images_to_occlude, M_gt_binary_np, self.occluder_pool, self.args
            )
            if occluded_results is not None:
                x_opaque_np = occluded_results[0]
                x_in_np = occluded_results[1]
                M_gt_binary_np = occluded_mask
                info += "_occluded"
        # =========================================
        
        # This part remains as you provided
        M_gt_mask_float_raw = M_gt_binary_np.astype(np.float32) / 255.0
        M_gt_mask_float_blurred = cv2.GaussianBlur(M_gt_mask_float_raw, (3, 3), 1.0)
        x_target_np = x_opaque_np.copy()

        inpaint_image_np = x_in_np.copy()
        
        center_point_pixels = (self.output_size // 2, self.output_size // 2)
        heatmap_np = M_gt_mask_float_raw

        # ================================================================
        # ================================================================
        if self.condition_type == 'mask':
            heatmap_np = M_gt_mask_float_raw
            info += "_cond_mask"

        elif self.condition_type == 'bbox':
            ys_b, xs_b = np.where(M_gt_binary_np > 127)
            if len(xs_b) == 0:
                x_min_b, y_min_b = 0, 0
                x_max_b, y_max_b = self.output_size - 1, self.output_size - 1
            else:
                x_min_b, x_max_b = int(xs_b.min()), int(xs_b.max())
                y_min_b, y_max_b = int(ys_b.min()), int(ys_b.max())
            heatmap_np = np.zeros((self.output_size, self.output_size), dtype=np.float32)
            heatmap_np[y_min_b:y_max_b + 1, x_min_b:x_max_b + 1] = 1.0
            center_point_pixels = ((x_min_b + x_max_b) // 2, (y_min_b + y_max_b) // 2)
            info += "_cond_bbox"

        elif self.condition_type == 'point':
            ys_p, xs_p = np.where(M_gt_binary_np > 127)
            if len(xs_p) > 0:
                x_min_p, y_min_p, x_max_p, y_max_p = xs_p.min(), ys_p.min(), xs_p.max(), ys_p.max()
                bbox_cx = (x_min_p + x_max_p) / 2
                bbox_cy = (y_min_p + y_max_p) / 2
                dists = (xs_p - bbox_cx) ** 2 + (ys_p - bbox_cy) ** 2
                closest_idx = np.argmin(dists)
                cx, cy = int(xs_p[closest_idx]), int(ys_p[closest_idx])
            else:
                cx, cy = self.output_size // 2, self.output_size // 2
            center_point_pixels = (cx, cy)
            heatmap_np = generate_heatmap(
                (self.output_size, self.output_size),
                (cx, cy),
                sigma=self.heatmap_sigma
            )
            info += "_cond_point"

        elif self.condition_type == 'mask_aug':
            if self.state == 'train':
                if self.mask_aug_scheme is None:
                    scheme_probs = self.args.get('mask_aug_scheme_probs', [0.50, 0.15, 0.25, 0.10])
                    chosen_scheme = random.choices([1, 2, 3, 4], weights=scheme_probs)[0]
                else:
                    chosen_scheme = self.mask_aug_scheme
            else:
                chosen_scheme = 3
            augmented_mask = augment_mask_scheme(M_gt_binary_np, scheme=chosen_scheme, args=self.args)
            heatmap_np = augmented_mask.astype(np.float32) / 255.0
            info += f"_cond_mask_aug_scheme{chosen_scheme}"

        else:
            raise ValueError(f"Unknown condition_type: '{self.condition_type}'. "
                             f"Choose from: 'mask', 'bbox', 'point', 'mask_aug'.")
        
        ref_img_224_np = cv2.resize(x_in_np, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        def numpy_to_tensor(img_np, normalize=True):
            img_tensor = torch.from_numpy(img_np.copy()).permute(2, 0, 1).float() / 255.0
            if normalize:
                img_tensor = (img_tensor - 0.5) / 0.5
            return img_tensor
        
        def numpy_to_tensor_clip(img_np):
            img_tensor = torch.from_numpy(img_np.copy()).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            return (img_tensor - mean) / std
        
        def numpy_to_tensor_mask(mask_np):
            return torch.from_numpy(mask_np.copy()).unsqueeze(0).float() / 255.0
        
        GT_tensor = numpy_to_tensor(x_target_np, normalize=True)
        inpaint_image_tensor = numpy_to_tensor(inpaint_image_np, normalize=True)
        ref_tensor_224 = numpy_to_tensor_clip(ref_img_224_np)
        M_gt_tensor = numpy_to_tensor_mask(M_gt_binary_np)
        heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0).float()
        
        return {
            "GT": GT_tensor,
            "inpaint_image": inpaint_image_tensor,
            "inpaint_mask": heatmap_tensor,
            "ref_imgs": ref_tensor_224,
            "M_gt": M_gt_tensor,
            "x_in": inpaint_image_tensor,
            "heatmap": heatmap_tensor,
            "center_point_pixels": torch.tensor(center_point_pixels, dtype=torch.int32), 
            "augmentation_info": info
        }