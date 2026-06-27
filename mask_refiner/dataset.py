import os
from PIL import Image
import torch
import torch.utils.data as data
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
    
    h, w = result.shape
    filled = result.copy()
    mask_for_fill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, mask_for_fill, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    result = cv2.bitwise_or(result, filled_inv)
    
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


def generate_heatmap(shape, center_point, sigma=10):
    """Helper function."""
    h, w = shape
    center_x, center_y = center_point
    y_grid, x_grid = np.ogrid[:h, :w]
    distances = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    heatmap = np.exp(-distances**2 / (2 * sigma**2))
    return heatmap.astype(np.float32)

class MyTransparentMaskHeadDataset(data.Dataset):
    def __init__(self, state, **args):
        self.args = args
        self.data_dir = args['dataset_dir']
        self.state = state
        self.output_size = args.get('img_size', 512)
        self.heatmap_sigma = args.get('heatmap_sigma', 10)
        
        self.mask_strategy = args.get('mask_strategy', 1) 
        
        self.bbox_wh_range = args.get('bbox_wh_range', 0.3)
        self.bbox_center_radius = args.get('bbox_center_radius', 0.1)
        
        self.white_bg_prob = args.get('white_bg_prob', 0.1)
        self.rgba_data_dir = os.path.join(os.path.dirname(self.data_dir), 'rgba')
        self.rgba_obj_files = []
        
        if state == 'train' and os.path.exists(self.rgba_data_dir):
            rgba_opaque_dir = os.path.join(self.rgba_data_dir, 'opaque')
            if os.path.isdir(rgba_opaque_dir):
                self.rgba_obj_files = sorted([f for f in os.listdir(rgba_opaque_dir) 
                                             if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"[{state}] Initialized. Strategy: {self.mask_strategy}, WhiteBG Prob: {self.white_bg_prob}, RGBA files: {len(self.rgba_obj_files)}")

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
        
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        self.sam_mask_dir = os.path.join(project_root, "dataset/sam3_test_masks")

    def __len__(self):
        return self.length

    def create_white_bg_direct(self, rgba_opaque, rgba_ref, mask_gt):
        """Helper function."""
        target_size = (self.output_size, self.output_size)
        
        obj_opaque = cv2.resize(rgba_opaque, target_size, interpolation=cv2.INTER_LINEAR)
        obj_ref = cv2.resize(rgba_ref, target_size, interpolation=cv2.INTER_LINEAR)
        obj_mask = cv2.resize(mask_gt, target_size, interpolation=cv2.INTER_NEAREST)

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

    def get_bbox(self, mask):
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            h, w = mask.shape
            return w // 2, h // 2, w // 2, h // 2
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return x_min, y_min, x_max, y_max

    def get_augmented_bbox(self, mask, bbox_mode):
        """Helper function."""
        x_min, y_min, x_max, y_max = self.get_bbox(mask)
        
        if bbox_mode == 1:
            return x_min, y_min, x_max, y_max
        
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        long_side = max(w, h)
        
        if bbox_mode == 2:
            w_scale = 1.0 + random.uniform(-self.bbox_wh_range, self.bbox_wh_range)
            h_scale = 1.0 + random.uniform(-self.bbox_wh_range, self.bbox_wh_range)
            new_w = w * w_scale
            new_h = h * h_scale
            new_cx, new_cy = cx, cy
        
        elif bbox_mode == 3:
            radius = long_side * self.bbox_center_radius
            angle = random.uniform(0, 2 * np.pi)
            new_cx = cx + radius * np.cos(angle)
            new_cy = cy + radius * np.sin(angle)
            new_w, new_h = w, h
        
        elif bbox_mode == 4:
            wh_range_half = self.bbox_wh_range / 2.0
            w_scale = 1.0 + random.uniform(-wh_range_half, wh_range_half)
            h_scale = 1.0 + random.uniform(-wh_range_half, wh_range_half)
            new_w = w * w_scale
            new_h = h * h_scale
            radius = long_side * self.bbox_center_radius / 2.0
            angle = random.uniform(0, 2 * np.pi)
            new_cx = cx + radius * np.cos(angle)
            new_cy = cy + radius * np.sin(angle)
        
        else:
            return x_min, y_min, x_max, y_max
        
        new_x_min = new_cx - new_w / 2.0
        new_x_max = new_cx + new_w / 2.0
        new_y_min = new_cy - new_h / 2.0
        new_y_max = new_cy + new_h / 2.0
        
        h_img, w_img = mask.shape
        new_x_min = np.clip(new_x_min, 0, w_img - 1)
        new_x_max = np.clip(new_x_max, 0, w_img - 1)
        new_y_min = np.clip(new_y_min, 0, h_img - 1)
        new_y_max = np.clip(new_y_max, 0, h_img - 1)
        
        return int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)

    def get_strategy_point(self, mask):
        """Helper function."""
        h, w = mask.shape
        ys, xs = np.where(mask > 127)
        
        if len(xs) == 0:
            return w // 2, h // 2

        if self.mask_strategy == 1:
            x_min, y_min, x_max, y_max = self.get_bbox(mask)
            bbox_cx = (x_min + x_max) / 2
            bbox_cy = (y_min + y_max) / 2
            
            dists = (xs - bbox_cx)**2 + (ys - bbox_cy)**2
            closest_idx = np.argmin(dists)
            return xs[closest_idx], ys[closest_idx]

        elif self.mask_strategy == 2:
            x_min, y_min, x_max, y_max = self.get_bbox(mask)
            bbox_cx = (x_min + x_max) / 2
            bbox_cy = (y_min + y_max) / 2
            
            bbox_long_side = max(x_max - x_min, y_max - y_min)
            radius = bbox_long_side / 4.0
            
            dists_to_bbox_center = np.sqrt((xs - bbox_cx)**2 + (ys - bbox_cy)**2)
            valid_indices = np.where(dists_to_bbox_center <= radius)[0]
            
            if len(valid_indices) > 0:
                rand_idx = random.choice(valid_indices)
                return xs[rand_idx], ys[rand_idx]
            else:
                dists = (xs - bbox_cx)**2 + (ys - bbox_cy)**2
                closest_idx = np.argmin(dists)
                return xs[closest_idx], ys[closest_idx]

        elif self.mask_strategy == 3:
            mask_512_binary = (mask > 127).astype(np.uint8) * 255
            dist_transform = cv2.distanceTransform(mask_512_binary, cv2.DIST_L2, 5)
            max_dist = dist_transform.max()
            
            if max_dist < 2.0:
                idx = random.randint(0, len(xs) - 1)
                return xs[idx], ys[idx]
            
            distance_threshold = max_dist / 4.0
            
            eroded_area_mask = (dist_transform >= distance_threshold).astype(np.uint8) * 255
            e_ys, e_xs = np.where(eroded_area_mask > 127)
            
            if len(e_xs) > 0:
                idx = random.randint(0, len(e_xs) - 1)
                return e_xs[idx], e_ys[idx]
            else:
                distance_threshold = max_dist / 8.0
                eroded_area_mask = (dist_transform >= distance_threshold).astype(np.uint8) * 255
                e_ys, e_xs = np.where(eroded_area_mask > 127)
                if len(e_xs) > 0:
                    idx = random.randint(0, len(e_xs) - 1)
                    return e_xs[idx], e_ys[idx]
                else:
                    idx = random.randint(0, len(xs) - 1)
                    return xs[idx], ys[idx]

        elif self.mask_strategy == 4:
            idx = random.randint(0, len(xs) - 1)
            return xs[idx], ys[idx]
        
        idx = random.randint(0, len(xs) - 1)
        return xs[idx], ys[idx]

    def __getitem__(self, index):
        use_white_bg = (self.state == 'train' and 
                       len(self.rgba_obj_files) > 0 and 
                       random.random() < self.white_bg_prob)
        
        resize_dim = (self.output_size, self.output_size)
        info = ""

        if use_white_bg:
            rgba_obj_name = random.choice(self.rgba_obj_files)
            
            rgba_opaque = np.array(Image.open(os.path.join(self.rgba_data_dir, 'opaque', rgba_obj_name)).convert('RGBA'))
            rgba_ref = np.array(Image.open(os.path.join(self.rgba_data_dir, 'reference', rgba_obj_name)).convert('RGBA'))
            rgba_mask = np.array(Image.open(os.path.join(self.rgba_data_dir, 'masks', rgba_obj_name)).convert('L'))
            _, rgba_mask = cv2.threshold(rgba_mask, 127, 255, cv2.THRESH_BINARY)
            
            x_opaque_np, x_in_np, M_gt_np = self.create_white_bg_direct(rgba_opaque, rgba_ref, rgba_mask)
            _, M_gt_binary_np = cv2.threshold(M_gt_np, 127, 255, cv2.THRESH_BINARY)
            
            info = f"white_bg_{self.mask_strategy}"
            
        else:
            img_name = self.image_files[index]
            opaque_path = os.path.join(self.data_dir, 'opaque', img_name)
            ref_path = os.path.join(self.data_dir, 'reference', img_name)
            mask_path = os.path.join(self.data_dir, 'masks', img_name)
            
            x_opaque_np = cv2.imread(opaque_path)
            x_in_np = cv2.imread(ref_path)
            M_gt_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if x_opaque_np is None or x_in_np is None or M_gt_np is None:
                raise FileNotFoundError(f"Failed to load image: {img_name}. Check paths:\n"
                                      f"  opaque: {opaque_path}\n"
                                      f"  reference: {ref_path}\n"
                                      f"  mask: {mask_path}")
            
            x_opaque_np = cv2.cvtColor(x_opaque_np, cv2.COLOR_BGR2RGB)
            x_in_np = cv2.cvtColor(x_in_np, cv2.COLOR_BGR2RGB)
            
            x_opaque_np = cv2.resize(x_opaque_np, resize_dim, interpolation=cv2.INTER_LINEAR)
            x_in_np = cv2.resize(x_in_np, resize_dim, interpolation=cv2.INTER_LINEAR)
            M_gt_np = cv2.resize(M_gt_np, resize_dim, interpolation=cv2.INTER_NEAREST)
            
            _, M_gt_binary_np = cv2.threshold(M_gt_np, 127, 255, cv2.THRESH_BINARY)
            info = f"dataset_{self.mask_strategy}"

        M_gt_mask_float_raw = M_gt_binary_np.astype(np.float32) / 255.0
        
        M_gt_mask_float_blurred = cv2.GaussianBlur(M_gt_mask_float_raw, (3, 3), 1.0)
        M_gt_3ch_blurred = M_gt_mask_float_blurred[:, :, np.newaxis]
        
        # GT = Mask_Blurred * Opaque + (1-Mask_Blurred) * Reference
        x_target_np = (M_gt_3ch_blurred * x_opaque_np.astype(np.float32) + 
                       (1.0 - M_gt_3ch_blurred) * x_in_np.astype(np.float32))
        x_target_np = np.clip(x_target_np, 0, 255).astype(np.uint8)

        heatmap_np = M_gt_mask_float_raw
        
        if self.mask_strategy in [1, 2, 3, 4]:
            if self.state == 'train':
                center_point = self.get_strategy_point(M_gt_binary_np)
            else:
                old_strategy = self.mask_strategy
                self.mask_strategy = 4
                center_point = self.get_strategy_point(M_gt_binary_np)
                self.mask_strategy = old_strategy
            
            heatmap_np = generate_heatmap(resize_dim, center_point, sigma=self.heatmap_sigma)
        
        elif self.mask_strategy in [5, 6, 7, 8]:
            scheme = self.mask_strategy - 4  # 5->1, 6->2, 7->3, 8->4
            
            if self.state in ['train', 'val']:
                condition_binary_np = augment_mask_scheme(M_gt_binary_np, scheme=scheme, args=self.args)
            else:
                if use_white_bg:
                    raise ValueError(f"Strategy {self.mask_strategy} (S5-8) does not support white background mode during inference.")
                
                sam_path = os.path.join(self.sam_mask_dir, f"{index:02d}.png")
                
                if not os.path.exists(sam_path):
                    raise FileNotFoundError(f"SAM mask not found at {sam_path} (index={index}). "
                                          f"Strategy {self.mask_strategy} requires SAM masks during inference.")
                
                sam_mask = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)
                sam_mask = cv2.resize(sam_mask, resize_dim, interpolation=cv2.INTER_NEAREST)
                _, condition_binary_np = cv2.threshold(sam_mask, 127, 255, cv2.THRESH_BINARY)
            
            heatmap_np = condition_binary_np.astype(np.float32) / 255.0
        
        elif self.mask_strategy in [9, 10, 11, 12]:
            bbox_mode = self.mask_strategy - 8  # 9->1, 10->2, 11->3, 12->4
            
            if self.state == 'train':
                bbox_augmented = self.get_augmented_bbox(M_gt_binary_np, bbox_mode)
            else:
                bbox_augmented = self.get_augmented_bbox(M_gt_binary_np, 1)
            
            x_min, y_min, x_max, y_max = bbox_augmented
            heatmap_np = np.zeros((self.output_size, self.output_size), dtype=np.float32)
            heatmap_np[y_min:y_max+1, x_min:x_max+1] = 1.0
        
        def numpy_to_tensor(img_np):
            """Helper function."""
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            return (img_tensor - 0.5) / 0.5
        
        def numpy_to_tensor_mask(mask_np):
            """Helper function."""
            return torch.from_numpy(mask_np).unsqueeze(0).float() / 255.0
        
        return {
            "x_gen_pixel": numpy_to_tensor(x_target_np),
            "x_rgb": numpy_to_tensor(x_in_np),
            "m_cond": torch.from_numpy(heatmap_np).unsqueeze(0).float(),
            "M_gt": numpy_to_tensor_mask(M_gt_binary_np)
        }
