import argparse
import os
import cv2
import json
import logging
import warnings
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
import re
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoConfig, AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers.utils import logging as transformers_logging
def normalize_terms_from_prompt(prompt: str):
    # "a. b. c." -> ["a","b","c"]
    terms = [t.strip().lower().rstrip('.') for t in prompt.split('.') if t.strip().strip('.') != ""]
    return terms

def union_masks(masks_bool_list):
    if len(masks_bool_list) == 0:
        return None
    out = masks_bool_list[0].copy()
    for m in masks_bool_list[1:]:
        out |= m
    return out
parser = argparse.ArgumentParser()
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
parser.add_argument("--input-dir", default="input", help="Directory containing input images")
parser.add_argument("--prompts-file", default="prompts.txt", help="Text file containing prompts, one per line")
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--output-dir", default="outputs/grounded_sam2_batch")
parser.add_argument("--no-dump-json", action="store_true")
parser.add_argument("--force-cpu", action="store_true")
parser.add_argument("--box-threshold", default=0.35, type=float, help="Box confidence threshold")
parser.add_argument("--synonyms-per-object", default=3, type=int, help="Number of synonyms per object group")
parser.add_argument("--has-desk", action="store_true",  # [MOD] new flag
                    help="If set, detect desk/table first and filter objects that overlap with the desk")
parser.add_argument("--single-prompt", type=str, default=None,
                    help="Use the same prompt for ALL images (ignore prompts.txt)")



parser.add_argument("--single-object", action="store_true",
                    help="Treat ALL terms in --single-prompt as synonyms of ONE object; output ONE mask named mask.png")
parser.add_argument("--top-k", type=int, default=None,
                    help="Find top K objects by confidence score (ignores group matching). Example: --top-k 9")

args = parser.parse_args()

GROUNDING_MODEL = args.grounding_model
INPUT_DIR = Path(args.input_dir)
PROMPTS_FILE = args.prompts_file
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)
DUMP_JSON_RESULTS = not args.no_dump_json
BOX_THRESHOLD = args.box_threshold
SYNONYMS_PER_OBJECT = args.synonyms_per_object
HAS_DESK = args.has_desk  # [MOD]
SINGLE_PROMPT = args.single_prompt
USE_SINGLE_PROMPT = SINGLE_PROMPT is not None and SINGLE_PROMPT.strip() != ""
TOP_K = args.top_k  # Number of top objects to find by confidence

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

model_id = GROUNDING_MODEL
warnings.filterwarnings("ignore", message="TORCH_CUDA_ARCH_LIST is not set.*")
transformers_logging.get_logger("transformers.models.grounding_dino.modeling_grounding_dino").setLevel(logging.ERROR)
processor = AutoProcessor.from_pretrained(model_id)
grounding_config = AutoConfig.from_pretrained(model_id)
grounding_config.disable_custom_kernels = True
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id,
    config=grounding_config,
).to(DEVICE)

if USE_SINGLE_PROMPT:
    text_prompt_global = SINGLE_PROMPT.strip()
    if not text_prompt_global.endswith('.'):
        text_prompt_global += '.'
    prompts = None
    print(f"[MODE] Using single prompt for all images: {text_prompt_global}")
    # [ADD] If single-object mode: allow "--single-prompt 'cup bottle'" or "cup,bottle"
    if args.single_object:
        # split by comma or whitespace
        terms = [t.strip().lower().rstrip('.') for t in re.split(r"[,\s]+", text_prompt_global) if t.strip().strip('.') != ""]

        # build "cup. bottle." format so GroundingDINO works the same way
        text_prompt_global = ". ".join(terms) + "."
        print(f"[MODE] Single-object synonyms: {terms}")

else:
    print(f"Reading prompts from {PROMPTS_FILE}")
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    prompts = [p if p.endswith('.') else p + '.' for p in prompts]
    prompts = [p.lower() if not p.endswith('.') else p.lower() for p in prompts]
    prompts = [p if p.endswith('.') else p + '.' for p in prompts]

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
image_files = []
for ext in image_extensions:
    image_files.extend(list(INPUT_DIR.glob(f'*{ext}')))
    image_files.extend(list(INPUT_DIR.glob(f'*{ext.upper()}')))

image_files = list(set([f.resolve() for f in image_files]))


def get_image_number(filepath):
    stem = filepath.stem
    num_str = ''
    for char in stem:
        if char.isdigit():
            num_str += char
        else:
            break
    return int(num_str) if num_str else float('inf')


image_files.sort(key=get_image_number)

print(f"Found {len(image_files)} images in {INPUT_DIR}")
if not USE_SINGLE_PROMPT:
    print(f"Found {len(prompts)} prompts")
else:
    print("[INFO] Single-prompt mode: prompts.txt is ignored")


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def parse_object_groups(text_prompt, synonyms_per_object):
    terms = [t.strip().rstrip('.') for t in re.split(r"[\n,.;]+", text_prompt) if t.strip().strip('.') != ""]
    groups = []
    for i in range(0, len(terms), synonyms_per_object):
        group = terms[i:i + synonyms_per_object]
        if group:
            groups.append(group)
    return groups


def largest_connected_component(mask, min_area=500):
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    best_label = None
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_label = label
            best_area = area
    if best_label is None or best_area < min_area:
        return None
    return labels == best_label


for img_path in image_files:
    if USE_SINGLE_PROMPT:
        text_prompt = text_prompt_global
        img_number = None
    else:
        img_number = int(''.join(c for c in img_path.stem if c.isdigit()))
        if img_number < 1 or img_number > len(prompts):
            print(f"\n{'=' * 80}")
            print(f"[SKIP] {img_path.name} - No matching prompt (line {img_number})")
            print(f"{'=' * 80}")
            continue
        text_prompt = prompts[img_number - 1]

        # [MOD] do NOT map prompt by image number; use the same prompt for all images
        # text_prompt = prompts[0]
        # img_number = None

    object_groups = parse_object_groups(text_prompt, SYNONYMS_PER_OBJECT)
    # [ADD] In single-object mode, force ONE group containing all terms (synonyms)
    if USE_SINGLE_PROMPT and args.single_object:
        terms = [t.strip().lower().rstrip('.') for t in text_prompt.split('.') if t.strip().strip('.') != ""]
        object_groups = [terms]

    print(f"\n{'=' * 80}")
    if USE_SINGLE_PROMPT:
        print(f"Processing: {img_path.name} -> Single prompt: {text_prompt}")
    else:
        print(f"Processing: {img_path.name} -> Prompt line {img_number}")
    print(f"Detected {len(object_groups)} object groups ({SYNONYMS_PER_OBJECT} synonyms each)")
    print(f"{'=' * 80}")


    img_output_dir = OUTPUT_DIR / img_path.stem
    img_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_rgb = np.array(image)
        sam2_predictor.set_image(image_rgb)

        # -----------------------------
        # [STEP 1] Desk detection (optional)
        # -----------------------------
        desk_mask = None  # [MOD] initialize here
        if HAS_DESK:  # [MOD] only run when user indicates a desk exists
            desk_prompt = "table."
            print(f"\n[STEP 1] Detecting desk/table...")

            desk_inputs = processor(images=image, text=desk_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                desk_outputs = grounding_model(**desk_inputs)

            desk_results = processor.post_process_grounded_object_detection(
                desk_outputs,
                desk_inputs.input_ids,
                target_sizes=[image_rgb.shape[:2]]
            )

            if len(desk_results[0]["boxes"]) > 0 and desk_results[0]["scores"].max() > 0.20:
                best_desk_idx = desk_results[0]["scores"].argmax()
                desk_box = desk_results[0]["boxes"][best_desk_idx:best_desk_idx + 1].cpu().numpy()

                desk_masks, _, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=desk_box,
                    multimask_output=False,
                )

                if desk_masks.ndim == 4:
                    desk_masks = desk_masks.squeeze(1)

                desk_mask = desk_masks[0]

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                desk_mask_uint8 = (desk_mask * 255).astype(np.uint8)
                desk_mask_uint8 = cv2.erode(desk_mask_uint8, kernel, iterations=2)
                desk_mask = (desk_mask_uint8 > 127).astype(bool)

                print(f"[STEP 1] Desk detected with confidence {desk_results[0]['scores'][best_desk_idx]:.3f}")
            else:
                print(f"[STEP 1] No desk detected")
        else:
            print("\n[STEP 1] Skipped desk/table detection (flag --has-desk not set)")  # [MOD]

        # Step 2 & 3: Multi-round detection with grouping
        matched_masks = []
        matched_boxes = []
        matched_confidences = []
        matched_names = []
        matched_group_ids = set()

        DESK_OVERLAP_THRESHOLD = 0.5
        MAX_ROUNDS = 5
        MIN_THRESHOLD = 0.1
        MAX_MASK_OVERLAP = 0.5


        if TOP_K is not None and USE_SINGLE_PROMPT:
            current_threshold = BOX_THRESHOLD
            retry_prompt = text_prompt
            print(f"\n[TOP-K MODE] Finding top {TOP_K} objects")
            print(f"  Threshold: {current_threshold:.3f}")
            print(f"  Prompt: {retry_prompt}")

            inputs = processor(images=image, text=retry_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[np.array(image.convert("RGB")).shape[:2]]
            )

            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]
            keep = scores > current_threshold

            input_boxes = boxes[keep].cpu().numpy()
            confidences_round = scores[keep].cpu().numpy().tolist()
            class_names_round = [labels[i] for i in range(len(labels)) if keep[i]]

            print(f"  Detected {len(input_boxes)} objects above threshold")

            if len(input_boxes) > 0:
                # SAM2 segmentation
                masks_round, _, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

                if masks_round.ndim == 4:
                    masks_round = masks_round.squeeze(1)


                target_terms = [t.strip().lower().rstrip('.') for t in text_prompt.split('.') if t.strip().strip('.') != ""]
                candidates = []  # (conf, i, detected_name, mask_bool, box)

                for i, (detected_name, conf) in enumerate(zip(class_names_round, confidences_round)):
                    name_lower = str(detected_name).lower()


                    if not any((term in name_lower) or (name_lower in term) for term in target_terms):
                        continue

                    mask = masks_round[i]
                    if mask.dtype != bool:
                        mask = mask.astype(bool)

                    mask_area = int(np.sum(mask))
                    if mask_area < 500:
                        continue


                    if desk_mask is not None:
                        overlap_area = int(np.sum(mask & desk_mask))
                        overlap_ratio = overlap_area / (mask_area + 1e-6)
                        if overlap_ratio > DESK_OVERLAP_THRESHOLD:
                            continue

                    candidates.append((float(conf), i, str(detected_name), mask, input_boxes[i]))

                print(f"  Collected {len(candidates)} candidate objects")


                candidates.sort(key=lambda x: x[0], reverse=True)


                selected_count = 0
                for conf, idx, name, mask, box in candidates:
                    if selected_count >= TOP_K:
                        break


                    mask_area = int(np.sum(mask))
                    max_overlap = 0.0
                    for existing_mask in matched_masks:
                        overlap_area = int(np.sum(mask & existing_mask))
                        overlap_ratio = overlap_area / (mask_area + 1e-6)
                        max_overlap = max(max_overlap, overlap_ratio)


                    if max_overlap > MAX_MASK_OVERLAP:
                        print(f"  [SKIP] {name} (conf {conf:.3f}) - overlaps {max_overlap:.2f} with existing mask")
                        continue

                    matched_masks.append(mask)
                    matched_boxes.append(box)
                    matched_confidences.append(conf)
                    matched_names.append(name)
                    selected_count += 1

                    print(f"  [OK] #{selected_count}: {name} (conf {conf:.3f}, area {mask_area})")

            print(f"\n[SUMMARY] Total matched (top-k): {len(matched_masks)}/{TOP_K} object(s)")


        elif USE_SINGLE_PROMPT:

            current_threshold = BOX_THRESHOLD
            retry_prompt = text_prompt
            print(f"\n[ROUND 1 - single prompt] Threshold: {current_threshold:.3f}")
            print(f"  Single prompt text: {retry_prompt}")

            inputs = processor(images=image, text=retry_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[np.array(image.convert("RGB")).shape[:2]]
            )

            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]
            keep = scores > current_threshold

            input_boxes = boxes[keep].cpu().numpy()
            confidences_round = scores[keep].cpu().numpy().tolist()
            class_names_round = [labels[i] for i in range(len(labels)) if keep[i]]

            print(f"  Detected {len(input_boxes)} objects above threshold")

            if len(input_boxes) > 0:
                masks_round, _, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

                if masks_round.ndim == 4:
                    masks_round = masks_round.squeeze(1)



                # ====== UNION MODE: collect all relevant masks then OR-union ======
                target_terms = [t.strip().lower().rstrip('.') for t in text_prompt.split('.') if t.strip().strip('.') != ""]
                collected = []  # (mask_area, conf, i, detected_name, mask_bool)

                for i, (detected_name, conf) in enumerate(zip(class_names_round, confidences_round)):
                    name_lower = str(detected_name).lower()


                    if not any((term in name_lower) or (name_lower in term) for term in target_terms):
                        continue

                    mask = masks_round[i]
                    if mask.dtype != bool:
                        mask = mask.astype(bool)

                    mask_area = int(np.sum(mask))
                    if mask_area < 500:
                        continue


                    if desk_mask is not None:
                        overlap_area = int(np.sum(mask & desk_mask))
                        overlap_ratio = overlap_area / (mask_area + 1e-6)
                        if overlap_ratio > DESK_OVERLAP_THRESHOLD:
                            continue

                    collected.append((mask_area, float(conf), i, str(detected_name), mask))

                print(f"  Collected {len(collected)} candidate masks for union")

                if len(collected) > 0:

                    collected.sort(key=lambda x: x[0], reverse=True)
                    max_area = collected[0][0]
                    AREA_RATIO_LIMIT = 2.5

                    selected = []
                    for area, conf, idx, name, mask in collected:
                        if area <= max_area * AREA_RATIO_LIMIT:
                            selected.append((area, conf, idx, name, mask))

                    print(f"  Union using {len(selected)}/{len(collected)} masks (AREA_RATIO_LIMIT={AREA_RATIO_LIMIT})")

                    # union masks
                    union_mask = selected[0][4].copy()
                    for _, _, _, _, m in selected[1:]:
                        union_mask |= m

                    if int(np.sum(union_mask)) >= 500:
                        matched_masks.append(union_mask)


                        ys, xs = np.where(union_mask)
                        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                        matched_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))


                        matched_confidences.append(collected[0][1])
                        matched_names.append("union")

                        print(f"  [OK] Union mask area={int(np.sum(union_mask))}")
                    else:
                        print("  [WARN] Union result too small, skipped")



            print(f"\n[SUMMARY] Total matched (single prompt): {len(matched_masks)} object(s)")


        else:
            matched_group_ids = set()
            for round_num in range(1, MAX_ROUNDS + 1):
                if len(matched_group_ids) >= len(object_groups):
                    break

                if round_num == 1:
                    current_threshold = BOX_THRESHOLD
                    retry_prompt = text_prompt
                else:
                    current_threshold = max(MIN_THRESHOLD, BOX_THRESHOLD * (1 - 0.2 * (round_num - 1)))

                    unmatched_terms = []
                    for gid in range(len(object_groups)):
                        if gid not in matched_group_ids:
                            unmatched_terms.extend(object_groups[gid])

                    if len(unmatched_terms) == 0:
                        break

                    retry_prompt = ' . '.join(unmatched_terms) + '.'

                print(f"\n[ROUND {round_num}] Threshold: {current_threshold:.3f}")
                if round_num > 1:
                    print(f"  Unmatched groups: {len(object_groups) - len(matched_group_ids)}")
                    print(f"  Retry prompt: {retry_prompt}")

                inputs = processor(images=image, text=retry_prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    outputs = grounding_model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    target_sizes=[np.array(image.convert("RGB")).shape[:2]]
                )

                boxes = results[0]["boxes"]
                scores = results[0]["scores"]
                labels = results[0]["labels"]
                keep = scores > current_threshold

                input_boxes = boxes[keep].cpu().numpy()
                confidences_round = scores[keep].cpu().numpy().tolist()
                class_names_round = [labels[i] for i in range(len(labels)) if keep[i]]

                print(f"  Detected {len(input_boxes)} objects above threshold")

                if len(input_boxes) == 0:
                    continue

                # SAM2 segmentation
                masks_round, _, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

                if masks_round.ndim == 4:
                    masks_round = masks_round.squeeze(1)

                used_indices = set()
                newly_matched = []

                for gid, group_synonyms in enumerate(object_groups):
                    if gid in matched_group_ids:
                        continue

                    candidates = []
                    for i, (detected_name, conf) in enumerate(zip(class_names_round, confidences_round)):
                        if i in used_indices:
                            continue

                        if any((syn in detected_name or detected_name == syn) for syn in group_synonyms):
                            mask = masks_round[i]
                            if mask.dtype != bool:
                                mask = mask.astype(bool)

                            mask_area = np.sum(mask)
                            if mask_area < 500:
                                continue

                            # Only apply desk filtering when a desk mask exists (i.e., --has-desk true and desk found)
                            if desk_mask is not None:  # [MOD] guard remains; no change to logic
                                overlap_area = np.sum(mask & desk_mask)
                                overlap_ratio = overlap_area / mask_area if mask_area > 0 else 0
                                if overlap_ratio > DESK_OVERLAP_THRESHOLD:
                                    continue

                            max_overlap_with_existing = 0.0
                            for existing_mask in matched_masks:
                                overlap_with_existing = np.sum(mask & existing_mask)
                                overlap_ratio_with_existing = overlap_with_existing / mask_area if mask_area > 0 else 0
                                max_overlap_with_existing = max(max_overlap_with_existing, overlap_ratio_with_existing)

                            if max_overlap_with_existing > MAX_MASK_OVERLAP:
                                continue

                            candidates.append((i, conf, detected_name))

                    if candidates:
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        best_idx, best_conf, best_name = candidates[0]

                        matched_masks.append(masks_round[best_idx].astype(bool))
                        matched_boxes.append(input_boxes[best_idx])
                        matched_confidences.append(best_conf)
                        matched_names.append(best_name)
                        matched_group_ids.add(gid)
                        newly_matched.append(gid)
                        used_indices.add(best_idx)

                        print(f"  [OK] Group {gid + 1} ({group_synonyms[0]}) -> {best_name} (conf {best_conf:.3f})")

                print(
                    f"  Round {round_num} result: {len(newly_matched)} newly matched, {len(object_groups) - len(matched_group_ids)} still missing")

            print(f"\n[SUMMARY] Total matched: {len(matched_masks)}/{len(object_groups)} groups")

        if len(matched_masks) == 0:
            print(f"[ERROR] No masks matched")
            continue

        masks = np.array(matched_masks)
        input_boxes = np.array(matched_boxes)
        confidences = matched_confidences
        class_names = matched_names
        class_ids = np.array(list(range(len(class_names))))

        print(f"\n[STEP 4] Cleaning masks...")
        MIN_MASK_AREA = 500
        MORPH_KERNEL_SIZE = 7

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
        cleaned_masks = []
        valid_indices = []

        for i, mask in enumerate(masks):
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

            h, w = mask_opened.shape
            mask_floodfill = mask_opened.copy()
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(mask_floodfill, flood_mask, (0, 0), 255)
            mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
            mask_filled = mask_opened | mask_floodfill_inv

            cleaned_mask = (mask_filled > 127).astype(bool)
            cleaned_mask = largest_connected_component(cleaned_mask, min_area=MIN_MASK_AREA)
            if cleaned_mask is None:
                continue

            cleaned_area = np.sum(cleaned_mask)
            if cleaned_area < MIN_MASK_AREA:
                continue

            cleaned_masks.append(cleaned_mask)
            valid_indices.append(i)

        if len(cleaned_masks) == 0:
            print(f"[ERROR] No valid masks after cleaning")
            continue

        masks = np.array(cleaned_masks)
        input_boxes = input_boxes[valid_indices]
        confidences = [confidences[i] for i in valid_indices]
        class_names = [class_names[i] for i in valid_indices]
        class_ids = np.array(list(range(len(class_names))))

        print(f"\n[STEP 5] Saving outputs...")
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        if desk_mask is not None:
            desk_mask_save = np.zeros((h, w), dtype=np.uint8)
            desk_mask_save[desk_mask] = 255
            cv2.imwrite(str(img_output_dir / "mask_0_desk.png"), desk_mask_save)

        # [ADD] single-object mode: save only one mask named "mask.png"
        if USE_SINGLE_PROMPT and args.single_object:
            individual_mask = np.zeros((h, w), dtype=np.uint8)
            individual_mask[masks[0]] = 255
            cv2.imwrite(str(img_output_dir / "mask.png"), individual_mask)
        else:
            for i, mask in enumerate(masks):
                individual_mask = np.zeros((h, w), dtype=np.uint8)
                individual_mask[mask] = 255
                mask_filename = f"mask_{i + 1}_{class_names[i].replace(' ', '_')}.png"
                mask_path = img_output_dir / mask_filename
                cv2.imwrite(str(mask_path), individual_mask)

        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)

        output_img_path = img_output_dir / "annotated.jpg"
        cv2.imwrite(str(output_img_path), annotated_frame)

        if DUMP_JSON_RESULTS:
            mask_rles = [single_mask_to_rle(mask) for mask in masks]
            input_boxes_list = input_boxes.tolist()
            scores_list = [float(confidences[i]) for i in range(len(confidences))]

            json_results = {
                "image_path": str(img_path),
                "object_groups": object_groups,
                "annotations": [
                    {
                        "class_name": class_name,
                        "bbox": box,
                        "segmentation": mask_rle,
                        "score": score,
                    }
                    for class_name, box, mask_rle, score in zip(class_names, input_boxes_list, mask_rles, scores_list)
                ],
                "box_format": "xyxy",
                "img_width": image.width,
                "img_height": image.height,
            }

            output_json_path = img_output_dir / "results.json"
            with open(output_json_path, "w") as f:
                json.dump(json_results, f, indent=4)

        print(f"\n[SUCCESS] Completed {img_path.name}")

    except Exception as e:
        print(f"\n[ERROR] Failed processing {img_path.name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'=' * 80}")
print(f"Batch processing complete! Results saved to {OUTPUT_DIR}")
print(f"{'=' * 80}")
