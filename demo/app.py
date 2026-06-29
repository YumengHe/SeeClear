from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import os
import base64
import hashlib
import re
import time
from pathlib import Path

APP_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GRADIO_TEMP_DIR", str(APP_REPO_ROOT / "outputs" / "gradio"))
os.environ.setdefault("TMPDIR", os.environ["GRADIO_TEMP_DIR"])
SEECLEAR_CACHE_DIR = Path(os.environ.get("SEECLEAR_CACHE_DIR", str(APP_REPO_ROOT / "outputs" / "cache")))
os.environ.setdefault("HF_HOME", str(SEECLEAR_CACHE_DIR / "hf_home"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(SEECLEAR_CACHE_DIR / "torch_extensions"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

import gradio as gr
import cv2
import numpy as np
import torch
from PIL import Image

from demo.depth_runtime import DepthRuntimeCache
from demo.pipeline import (
    REPO_ROOT,
    PreparedLayout,
    build_baseline_depth_command,
    build_depth_command,
    build_gsam2_command,
    build_sam3_refine_command,
    build_trans4trans_command,
    collect_mask_paths,
    default_paths,
    expected_outputs,
    materialize_opaque_image,
    prepare_single_image_layout,
    result_image_path,
)
from demo.run_once import write_commands
from demo.visualization import save_depth_color, save_depth_visualizations, save_depth_per_instance_vis
from scripts.infer_opacification import OpacificationRuntime, build_arg_parser, run_opacification


SAM3_ROOT = REPO_ROOT / "third_party" / "sam3"
DEFAULT_EXAMPLE_IMAGE = REPO_ROOT / "examples" / "demo" / "1.jpg"
DEFAULT_EXAMPLE_MASK_DIR = REPO_ROOT / "examples" / "demo" / "masks"
DEFAULT_EXAMPLE_MASK_UNION = Path(os.environ["GRADIO_TEMP_DIR"]) / "seeclear_default_example_mask.png"


class Sam3State:
    def __init__(self):
        self.model = None
        self.processor = None
        self.inference_state = None
        self.current_image_rgb = None
        self.current_mask = None
        self.current_iou = None
        self.points = []
        self.labels = []
        self.saved_masks = []

    def reset_prompts(self):
        self.current_mask = None
        self.current_iou = None
        self.points = []
        self.labels = []

    def reset_image(self):
        self.reset_prompts()
        self.saved_masks = []


SAM3_STATE = Sam3State()
OPACIFICATION_RUNTIME = None
OPACIFICATION_RUNTIME_KEY = None
DEPTH_RUNTIME_CACHE = DepthRuntimeCache()


def _select_server_port(server_name: str) -> int:
    if "GRADIO_SERVER_PORT" in os.environ:
        return int(os.environ["GRADIO_SERVER_PORT"])
    bind_host = "127.0.0.1" if server_name in {"0.0.0.0", "::"} else server_name
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return sock.getsockname()[1]


class MaskState:
    def __init__(self):
        self.mask_paths = []
        self.image_signature = None
        self.source = None

    def set(self, mask_paths, image_signature, source):
        self.mask_paths = list(mask_paths)
        self.image_signature = image_signature
        self.source = source

    def clear(self):
        self.mask_paths = []
        self.image_signature = None
        self.source = None


MASK_STATE = MaskState()


class OpaqueState:
    def __init__(self):
        self.layout = None
        self.image_signature = None
        self.opaque_path = None
        self.is_uploaded = False

    def set(self, layout, image_signature, opaque_path, is_uploaded=False):
        self.layout = layout
        self.image_signature = image_signature
        self.opaque_path = opaque_path
        self.is_uploaded = is_uploaded

    def clear(self):
        self.layout = None
        self.image_signature = None
        self.opaque_path = None
        self.is_uploaded = False


OPAQUE_STATE = OpaqueState()


class GptPromptState:
    def __init__(self):
        self.full_prompt = None
        self.display_prompt = None
        self.image_signature = None

    def set(self, full_prompt, display_prompt, image_signature):
        self.full_prompt = full_prompt
        self.display_prompt = display_prompt
        self.image_signature = image_signature

    def clear(self):
        self.full_prompt = None
        self.display_prompt = None
        self.image_signature = None

    def matches(self, image_signature, display_prompt):
        return (
            self.full_prompt is not None
            and self.display_prompt is not None
            and self.image_signature == image_signature
            and (display_prompt or "").strip() == self.display_prompt.strip()
        )


GPT_PROMPT_STATE = GptPromptState()


class Gsam2State:
    def __init__(self):
        self.saved_masks = []
        self.current_masks = []
        self.current_image_rgb = None

    def reset_saved(self):
        self.saved_masks = []
        self.current_masks = []

    def reset_all(self):
        self.saved_masks = []
        self.current_masks = []
        self.current_image_rgb = None


GSAM2_STATE = Gsam2State()


class BbxState:
    def __init__(self):
        self.points = []
        self.current_mask = None  # bool ndarray for current unsaved polygon
        self.saved_masks = []     # list of bool ndarray
        self.current_image_rgb = None

    def reset_current(self):
        self.points = []
        self.current_mask = None

    def reset_saved(self):
        self.saved_masks = []

    def reset_all(self):
        self.points = []
        self.current_mask = None
        self.saved_masks = []
        self.current_image_rgb = None


BBX_STATE = BbxState()


def _image_signature(image) -> str:
    image_rgb = _normalize_image(image)
    if image_rgb is None:
        raise gr.Error("Upload an RGB image.")
    return _image_array_signature(image_rgb)


def _image_array_signature(image_rgb: np.ndarray) -> str:
    digest = hashlib.md5(image_rgb.tobytes()).hexdigest()
    return f"{image_rgb.shape}:{digest}"


def _save_upload_image(image, path: Path) -> Path:
    if image is None:
        raise gr.Error("Upload an RGB image.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.convert("RGB").save(path)
    else:
        Image.fromarray(image).convert("RGB").save(path)
    return path


def _save_upload_mask(mask, path: Path) -> Path:
    if mask is None:
        raise gr.Error("Upload a binary or grayscale transparent-object mask.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(mask, Image.Image):
        mask.convert("L").save(path)
    else:
        Image.fromarray(mask).convert("L").save(path)
    return path


def _upload_mask_to_array(mask) -> np.ndarray:
    if mask is None:
        raise gr.Error("Upload a binary or grayscale transparent-object mask.")
    if isinstance(mask, Image.Image):
        arr = np.array(mask.convert("L"))
    else:
        arr = np.array(mask)
        if arr.ndim == 3:
            arr = Image.fromarray(arr).convert("L")
            arr = np.array(arr)
    return arr


def _component_min_area(height: int, width: int) -> int:
    return 10


def _save_upload_mask_instances(mask, out_dir: Path) -> list[Path]:
    arr = _upload_mask_to_array(mask)
    binary = arr > 0
    if not np.any(binary):
        raise gr.Error("Uploaded mask is empty.")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    min_area = _component_min_area(binary.shape[0], binary.shape[1])
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((label, area))
    if not components:
        raise gr.Error(f"Uploaded mask has no connected component >= {min_area} pixels.")

    components.sort(key=lambda item: item[1], reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, (label, _) in enumerate(components):
        mask_path = out_dir / f"mask_{idx}.png"
        inst = (labels == label).astype(np.uint8) * 255
        Image.fromarray(inst, mode="L").save(mask_path)
        paths.append(mask_path)
    return paths


def _manual_gsam_prompt(prompt: str) -> tuple[str, int]:
    terms = [t.strip().lower().rstrip(".") for t in re.split(r"[\n,;]+", prompt or "") if t.strip().strip(".")]
    if not terms:
        raise gr.Error("Enter at least one Grounded SAM 2 text prompt, for example: bottle, glass.")
    return ". ".join(terms) + ".", 1


def _gpt_transparent_object_prompt(image_path: Path) -> tuple[str, int, str]:
    raise gr.Error("GPT prompt generation is disabled for this release.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise gr.Error("OPENAI_API_KEY is not set on the server.")
    from openai import OpenAI

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    client = OpenAI()
    response = client.responses.create(
        model=os.environ.get("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Find all transparent or translucent physical objects in this image. "
                            "Return ONLY lines in this exact format, one object per line: "
                            "object_name: synonym1. synonym2. synonym3. "
                            "Use concise English object names and include three grounding terms per object. "
                            "Do not include non-transparent background objects."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                    },
                ],
            }
        ],
    )
    all_terms = []
    display_terms = []
    for line in response.output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1]
        group_terms = [t.strip().lower().rstrip(".") for t in line.split(".") if t.strip().strip(".")]
        if not group_terms:
            continue
        all_terms.extend(group_terms[:3])
        display_terms.append(group_terms[0])
    if not all_terms:
        raise gr.Error("GPT returned no transparent-object prompt terms.")
    return ". ".join(all_terms) + ".", 3, ". ".join(display_terms) + "."


def _normalize_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    elif not isinstance(image, np.ndarray):
        image = np.array(image)
    if image.dtype in (np.float32, np.float64):
        image = (image * 255).clip(0, 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def _overlay_red(image_rgb: np.ndarray, mask_any: np.ndarray, alpha=0.5):
    mask_any = np.asarray(mask_any)
    if mask_any.ndim == 3:
        if mask_any.shape[0] == 1:
            mask_any = mask_any[0]
        elif mask_any.shape[-1] == 1:
            mask_any = mask_any[..., 0]
        elif mask_any.shape[-1] == 3 and mask_any.shape[:2] == image_rgb.shape[:2]:
            mask_any = mask_any[..., 0]
        elif mask_any.shape[0] == 3 and mask_any.shape[1:] == image_rgb.shape[:2]:
            mask_any = mask_any[0]
    mask_bool = mask_any.astype(bool)
    out = image_rgb.astype(np.float32).copy()
    out[mask_bool] = out[mask_bool] * (1 - alpha) + np.array([255, 0, 0], dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8), mask_bool


def _overlay_color(image_rgb: np.ndarray, mask_any: np.ndarray, color, alpha=0.5):
    mask_bool = np.asarray(mask_any).astype(bool)
    out = image_rgb.astype(np.float32).copy()
    out[mask_bool] = out[mask_bool] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8), mask_bool


def _overlay_masks_colored(image_rgb: np.ndarray, masks, colors, alpha=0.5):
    out = image_rgb.copy()
    for idx, mask in enumerate(masks):
        out, _ = _overlay_color(out, mask, colors[idx % len(colors)], alpha=alpha)
    return out


def _mask_path_to_bool(mask_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    mask_img = Image.open(mask_path).convert("L")
    target_h, target_w = target_shape
    if mask_img.size != (target_w, target_h):
        mask_img = mask_img.resize((target_w, target_h), Image.NEAREST)
    return np.array(mask_img) > 0


def _overlay_mask_paths_colored(image_rgb: np.ndarray, mask_paths: list[Path], alpha=0.5):
    masks = [_mask_path_to_bool(Path(mask_path), image_rgb.shape[:2]) for mask_path in mask_paths]
    return _overlay_masks_colored(image_rgb, masks, GSAM2_COLORS, alpha=alpha)


def _mask_union_from_arrays(mask_arrays):
    if not mask_arrays:
        return None
    arrays = [np.asarray(mask).astype(bool) for mask in mask_arrays]
    return np.any(np.stack(arrays, axis=0), axis=0)


def _save_mask_union_from_arrays(mask_arrays, out_root: Path, prefix: str):
    union = _mask_union_from_arrays(mask_arrays)
    if union is None:
        return None
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{prefix}_{time.time_ns()}.png"
    Image.fromarray(union.astype(np.uint8) * 255, mode="L").save(out_path)
    return str(out_path)


def _draw_points(image_rgb: np.ndarray, points, labels):
    out = image_rgb.copy()
    for (x, y), label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(out, (int(x), int(y)), 7, color, -1)
        cv2.circle(out, (int(x), int(y)), 10, (255, 255, 255), 2)
    return out


def _saved_sam3_mask_union():
    return _mask_union_from_arrays(SAM3_STATE.saved_masks)


def _sam3_preview_with_saved_masks():
    if SAM3_STATE.current_image_rgb is None:
        return None
    preview = SAM3_STATE.current_image_rgb.copy()
    if SAM3_STATE.saved_masks:
        preview = _overlay_masks_colored(preview, SAM3_STATE.saved_masks, GSAM2_COLORS, alpha=0.45)
    return _draw_points(preview, SAM3_STATE.points, SAM3_STATE.labels)


def _load_sam3_model():
    if SAM3_STATE.model is not None:
        return
    if str(SAM3_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(SAM3_ROOT))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = REPO_ROOT / "pretrained_models" / "sam3.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"SAM3 checkpoint is missing: {checkpoint_path}. "
            "Create this as a symlink to the local SAM3 checkpoint before using SAM3 click."
        )
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        enable_inst_interactivity=True,
        eval_mode=True,
    )
    if model.inst_interactive_predictor is None:
        raise RuntimeError("SAM3 interactive predictor is not available.")
    SAM3_STATE.model = model
    SAM3_STATE.processor = Sam3Processor(model, device=device)


def load_image_for_sam3(image):
    if image is None:
        raise gr.Error("Upload an RGB image first.")
    _load_sam3_model()
    image_rgb = _normalize_image(image)
    SAM3_STATE.current_image_rgb = image_rgb
    SAM3_STATE.reset_image()
    SAM3_STATE.inference_state = SAM3_STATE.processor.set_image(Image.fromarray(image_rgb))
    return image_rgb


def update_mask_processing_ui(mask_source, image):
    MASK_STATE.clear()
    GSAM2_STATE.reset_all()
    BBX_STATE.reset_all()
    show_sam3 = mask_source == "SAM3 click"
    show_gsam = mask_source == "Grounded SAM 2 text"
    show_bbx = mask_source == "Manual BBX"
    show_mask_actions = mask_source in ("Upload mask", "Trans4Trans auto")
    show_generate = mask_source == "Trans4Trans auto"
    show_refine = mask_source == "Trans4Trans auto"

    if show_bbx:
        bbx_img = _normalize_image(image) if image is not None else None
        if bbx_img is not None:
            BBX_STATE.current_image_rgb = bbx_img
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            None,
            bbx_img,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    if mask_source != "SAM3 click":
        return (
            gr.update(visible=False),
            gr.update(visible=show_gsam),
            gr.update(visible=False),
            gr.update(visible=show_mask_actions),
            None,
            None,
            None,
            None,
            gr.update(visible=show_generate),
            gr.update(visible=show_refine),
        )
    if image is None:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            None,
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
        )
    image_rgb = _normalize_image(image)
    SAM3_STATE.current_image_rgb = image_rgb
    SAM3_STATE.reset_image()
    SAM3_STATE.inference_state = None
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        image_rgb,
        None,
        None,
        None,
        gr.update(visible=False),
        gr.update(visible=False),
    )


def clear_generated_mask():
    MASK_STATE.clear()
    OPAQUE_STATE.clear()
    return None


def clear_generated_outputs():
    MASK_STATE.clear()
    OPAQUE_STATE.clear()
    return None, None, None, gr.update(value=None, visible=False), None, None, None


def mask_input_changed(image, mask_source, mask):
    if OPAQUE_STATE.is_uploaded and OPAQUE_STATE.layout is not None and OPAQUE_STATE.opaque_path is not None:
        if mask is not None:
            mask_paths = _save_upload_mask_instances(mask, OPAQUE_STATE.layout.mask_dir / OPAQUE_STATE.layout.stem)
            MASK_STATE.set(mask_paths, OPAQUE_STATE.image_signature, "Upload mask")
        else:
            MASK_STATE.clear()
        return (
            gr.update(),
            None,
            None,
            gr.update(value=OPAQUE_STATE.opaque_path, visible=True),
            None,
            None,
            None,
        )
    if mask_source == "Upload mask" and mask is not None and image is not None:
        image_signature = _image_signature(image)
        work_dir = Path(tempfile.mkdtemp(prefix="transparent_upload_mask_", dir=os.environ["GRADIO_TEMP_DIR"]))
        mask_paths = _save_upload_mask_instances(mask, work_dir / "upload_mask_instances")
        MASK_STATE.set(mask_paths, image_signature, "Upload mask")
        OPAQUE_STATE.clear()
        return None, None, None, gr.update(value=None, visible=False), None, None, None
    return clear_generated_outputs()


def clear_image_dependent_outputs():
    GPT_PROMPT_STATE.clear()
    MASK_STATE.clear()
    OPAQUE_STATE.clear()
    return None, None, None, None, gr.update(value=None, visible=False), None, None, None


def clear_opaque_outputs():
    OPAQUE_STATE.clear()
    return None, None, None, gr.update(value=None, visible=False), None, None, None


def generate_gpt_prompt(image):
    raise gr.Error("GPT prompt generation is disabled for this release.")
    image_signature = _image_signature(image)
    work_dir = Path(tempfile.mkdtemp(prefix="transparent_gpt_prompt_", dir=os.environ["GRADIO_TEMP_DIR"]))
    image_path = _save_upload_image(image, work_dir / "upload" / "demo.png")
    full_prompt, _, display_prompt = _gpt_transparent_object_prompt(image_path)
    GPT_PROMPT_STATE.set(full_prompt, display_prompt, image_signature)
    MASK_STATE.clear()
    OPAQUE_STATE.clear()
    return display_prompt


def _predict_current_sam3_mask(multimask):
    point_coords = np.asarray(SAM3_STATE.points, dtype=np.float32)
    point_labels = np.asarray(SAM3_STATE.labels, dtype=np.int32)
    masks, iou_preds, _ = SAM3_STATE.model.predict_inst(
        SAM3_STATE.inference_state,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=bool(multimask),
        return_logits=False,
        normalize_coords=True,
    )
    if masks.ndim == 2:
        best_mask = masks
        best_iou = float(iou_preds) if np.isscalar(iou_preds) else float(np.max(iou_preds))
    else:
        best_idx = int(np.argmax(iou_preds))
        best_mask = masks[best_idx]
        best_iou = float(iou_preds[best_idx])

    base_rgb = _sam3_preview_with_saved_masks()
    next_color = GSAM2_COLORS[len(SAM3_STATE.saved_masks) % len(GSAM2_COLORS)]
    overlay_rgb, mask_bool = _overlay_color(base_rgb, best_mask, next_color, alpha=0.55)
    SAM3_STATE.current_mask = mask_bool
    SAM3_STATE.current_iou = best_iou
    return _draw_points(overlay_rgb, SAM3_STATE.points, SAM3_STATE.labels)


def sam3_select_point(point_mode, multimask, image, evt: gr.SelectData):
    if SAM3_STATE.current_image_rgb is None:
        if image is None:
            raise gr.Error("Select SAM3 click and upload an RGB image first.")
        SAM3_STATE.current_image_rgb = _normalize_image(image)
        SAM3_STATE.reset_prompts()
        SAM3_STATE.inference_state = None
    if SAM3_STATE.inference_state is None:
        _load_sam3_model()
        SAM3_STATE.inference_state = SAM3_STATE.processor.set_image(Image.fromarray(SAM3_STATE.current_image_rgb))
    x_click, y_click = int(evt.index[0]), int(evt.index[1])
    label = 1 if point_mode == "positive" else 0
    SAM3_STATE.points.append([x_click, y_click])
    SAM3_STATE.labels.append(label)
    preview = _predict_current_sam3_mask(multimask)
    return preview


def sam3_clear():
    if SAM3_STATE.current_image_rgb is None:
        raise gr.Error("Select SAM3 click and upload an RGB image first.")
    SAM3_STATE.reset_prompts()
    SAM3_STATE.inference_state = SAM3_STATE.processor.set_image(Image.fromarray(SAM3_STATE.current_image_rgb))
    return _sam3_preview_with_saved_masks()


def sam3_undo(multimask):
    if SAM3_STATE.current_image_rgb is None:
        raise gr.Error("Select SAM3 click and upload an RGB image first.")
    if not SAM3_STATE.points:
        return _sam3_preview_with_saved_masks()
    SAM3_STATE.points.pop()
    SAM3_STATE.labels.pop()
    if not SAM3_STATE.points:
        SAM3_STATE.current_mask = None
        SAM3_STATE.current_iou = None
        return _sam3_preview_with_saved_masks()
    preview = _predict_current_sam3_mask(multimask)
    return preview


def _save_current_sam3_mask(path: Path) -> Path:
    if SAM3_STATE.current_mask is None:
        raise gr.Error("No SAM3 mask yet. Load the image and click positive/negative points first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(SAM3_STATE.current_mask.astype(np.uint8) * 255, mode="L").save(path)
    return path


def _save_mask_arrays(mask_arrays: list[np.ndarray], out_dir: Path) -> list[Path]:
    if not mask_arrays:
        raise gr.Error("No saved SAM3 object masks. Click points, then click 'Save object mask' for each object.")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, mask_array in enumerate(mask_arrays):
        mask_bool = np.asarray(mask_array).astype(bool)
        mask_path = out_dir / f"mask_{idx}.png"
        Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L").save(mask_path)
        paths.append(mask_path)
    return paths


def _copy_mask_paths(mask_paths: list[Path], out_dir: Path) -> list[Path]:
    if not mask_paths:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for idx, mask_path in enumerate(mask_paths):
        dest = out_dir / f"mask_{idx}.png"
        shutil.copy2(Path(mask_path), dest)
        copied.append(dest)
    return copied


def _save_sam3_saved_union_preview() -> str:
    out_dir = Path(os.environ["GRADIO_TEMP_DIR"]) / "sam3_saved_masks"
    out_path = out_dir / f"union_{time.time_ns()}.png"
    saved_paths = _save_mask_arrays(SAM3_STATE.saved_masks, out_dir / f"instances_{time.time_ns()}")
    union_path = _union_masks(saved_paths, out_path)
    return str(union_path)


def sam3_save_object_mask():
    if SAM3_STATE.current_image_rgb is None:
        raise gr.Error("Select SAM3 click and upload an RGB image first.")
    if SAM3_STATE.current_mask is None:
        raise gr.Error("No SAM3 candidate mask yet. Click positive/negative points first.")
    SAM3_STATE.saved_masks.append(SAM3_STATE.current_mask.copy())
    saved_paths = _save_mask_arrays(
        SAM3_STATE.saved_masks,
        Path(os.environ["GRADIO_TEMP_DIR"]) / "sam3_saved_masks" / f"state_{time.time_ns()}",
    )
    MASK_STATE.set(saved_paths, _image_array_signature(SAM3_STATE.current_image_rgb), "SAM3 click")
    OPAQUE_STATE.clear()
    SAM3_STATE.reset_prompts()
    SAM3_STATE.inference_state = SAM3_STATE.processor.set_image(Image.fromarray(SAM3_STATE.current_image_rgb))
    return _sam3_preview_with_saved_masks(), _save_sam3_saved_union_preview()


def sam3_clear_saved_masks():
    SAM3_STATE.saved_masks = []
    MASK_STATE.clear()
    OPAQUE_STATE.clear()
    return _sam3_preview_with_saved_masks(), None


def _union_masks(mask_paths: list[Path], out_path: Path) -> Path:
    if not mask_paths:
        raise gr.Error("No mask was produced.")
    arrays = []
    target_size = None
    for mask_path in mask_paths:
        img = Image.open(mask_path).convert("L")
        if target_size is None:
            target_size = img.size
        elif img.size != target_size:
            img = img.resize(target_size, Image.NEAREST)
        arrays.append(np.array(img) > 0)
    union = np.any(np.stack(arrays, axis=0), axis=0).astype(np.uint8) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(union, mode="L").save(out_path)
    return out_path


def _default_example_mask_path():
    mask_paths = collect_mask_paths(DEFAULT_EXAMPLE_MASK_DIR)
    if not DEFAULT_EXAMPLE_IMAGE.is_file() or not mask_paths:
        return None
    if DEFAULT_EXAMPLE_MASK_UNION.is_file():
        return DEFAULT_EXAMPLE_MASK_UNION
    return _union_masks(mask_paths, DEFAULT_EXAMPLE_MASK_UNION)


GSAM2_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
    (255, 99, 71),
    (46, 139, 87),
    (30, 144, 255),
    (218, 112, 214),
    (154, 205, 50),
    (255, 140, 0),
    (72, 61, 139),
    (0, 206, 209),
    (199, 21, 133),
    (107, 142, 35),
]


def _gsam2_overlay_image():
    """Returns np.ndarray overlay or None. Saved and current instance masks use distinct colors."""
    if GSAM2_STATE.current_image_rgb is None:
        return None
    preview = GSAM2_STATE.current_image_rgb.copy()
    if GSAM2_STATE.saved_masks:
        preview = _overlay_masks_colored(preview, GSAM2_STATE.saved_masks, GSAM2_COLORS, alpha=0.42)
    if GSAM2_STATE.current_masks:
        preview = _overlay_masks_colored(preview, GSAM2_STATE.current_masks, GSAM2_COLORS, alpha=0.58)
    return preview


def _gsam2_union_path():
    """Saves union of all saved Grounded SAM 2 masks to a temp file, returns path str or None."""
    if not GSAM2_STATE.saved_masks:
        return None
    ts = time.time_ns()
    out_dir = Path(os.environ["GRADIO_TEMP_DIR"]) / "gsam2_saved_masks"
    inst_dir = out_dir / f"inst_{ts}"
    saved_paths = _save_mask_arrays(GSAM2_STATE.saved_masks, inst_dir)
    union_path = _union_masks(saved_paths, out_dir / f"union_{ts}.png")
    return str(union_path)


def gsam2_run(image, prompt):
    if image is None:
        raise gr.Error("Upload an RGB image first.")
    image_rgb = _normalize_image(image)
    GSAM2_STATE.current_image_rgb = image_rgb
    GSAM2_STATE.current_masks = []
    paths = default_paths()
    prompt_str, _ = _manual_gsam_prompt(prompt)
    work_dir = Path(tempfile.mkdtemp(prefix="transparent_gsam2_", dir=os.environ["GRADIO_TEMP_DIR"]))
    image_path = work_dir / "upload" / "demo.png"
    _save_upload_image(image, image_path)
    out_dir = work_dir / "grounded_sam2_mask"
    cmd = build_gsam2_command(paths, image_path, prompt_str, out_dir, "demo", synonyms_per_object=1)
    logs = []
    _run_command(cmd, paths.repo_root, logs)
    mask_paths = collect_mask_paths(out_dir)
    if not mask_paths:
        raise gr.Error("Grounded SAM 2 produced no mask for that prompt.")
    arrays = [np.array(Image.open(p).convert("L")) > 0 for p in mask_paths]
    GSAM2_STATE.current_masks = arrays
    return _gsam2_overlay_image()


def gsam2_save_mask():
    if GSAM2_STATE.current_image_rgb is None or not GSAM2_STATE.current_masks:
        raise gr.Error("Run Grounded SAM 2 first to get a mask.")
    GSAM2_STATE.saved_masks.extend(mask.copy() for mask in GSAM2_STATE.current_masks)
    GSAM2_STATE.current_masks = []
    saved_paths = _save_mask_arrays(
        GSAM2_STATE.saved_masks,
        Path(os.environ["GRADIO_TEMP_DIR"]) / "gsam2_saved_masks" / f"state_{time.time_ns()}",
    )
    MASK_STATE.set(saved_paths, _image_array_signature(GSAM2_STATE.current_image_rgb), "Grounded SAM 2 text")
    OPAQUE_STATE.clear()
    return _gsam2_overlay_image(), _gsam2_union_path()


def gsam2_clear_saved():
    GSAM2_STATE.reset_saved()
    MASK_STATE.clear()
    OPAQUE_STATE.clear()
    return _gsam2_overlay_image(), None


def gsam2_reset_on_image_change():
    GSAM2_STATE.reset_all()
    return None, None


def _bbx_preview_image():
    if BBX_STATE.current_image_rgb is None:
        return None
    preview = BBX_STATE.current_image_rgb.copy()
    saved_union = _mask_union_from_arrays(BBX_STATE.saved_masks)
    if saved_union is not None:
        preview, _ = _overlay_red(preview, saved_union, alpha=0.42)
    if BBX_STATE.current_mask is not None:
        preview, _ = _overlay_color(preview, BBX_STATE.current_mask, (0, 180, 255), alpha=0.5)
    if BBX_STATE.points:
        pts = np.array(BBX_STATE.points, dtype=np.int32)
        for idx, point in enumerate(BBX_STATE.points):
            cv2.circle(preview, point, 6, (255, 80, 0), -1)
            cv2.putText(
                preview,
                str(idx + 1),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if len(pts) >= 2:
            cv2.polylines(preview, [pts], isClosed=len(pts) == 4, color=(255, 80, 0), thickness=2)
    return preview


def bbx_select_point(image, evt: gr.SelectData):
    if BBX_STATE.current_image_rgb is None:
        if image is None:
            raise gr.Error("Upload an RGB image first.")
        BBX_STATE.current_image_rgb = _normalize_image(image)
    x, y = int(evt.index[0]), int(evt.index[1])
    if len(BBX_STATE.points) >= 4:
        BBX_STATE.points = [(x, y)]
        BBX_STATE.current_mask = None
    else:
        BBX_STATE.points.append((x, y))
    if len(BBX_STATE.points) == 4:
        H, W = BBX_STATE.current_image_rgb.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        pts = np.array(BBX_STATE.points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        mask = mask.astype(bool)
        BBX_STATE.current_mask = mask
    return _bbx_preview_image()


def bbx_confirm_mask():
    if BBX_STATE.current_image_rgb is None or BBX_STATE.current_mask is None:
        raise gr.Error("Click four points to define the polygon first.")
    BBX_STATE.saved_masks.append(BBX_STATE.current_mask.copy())
    saved_paths = _save_mask_arrays(
        BBX_STATE.saved_masks,
        Path(os.environ["GRADIO_TEMP_DIR"]) / "bbx_mask" / f"state_{time.time_ns()}",
    )
    MASK_STATE.set(saved_paths, _image_array_signature(BBX_STATE.current_image_rgb), "Manual BBX")
    OPAQUE_STATE.clear()
    BBX_STATE.reset_current()
    return (
        _bbx_preview_image(),
        _save_mask_union_from_arrays(BBX_STATE.saved_masks, Path(os.environ["GRADIO_TEMP_DIR"]) / "bbx_mask", "union"),
    )


def bbx_clear():
    BBX_STATE.reset_current()
    return (
        _bbx_preview_image(),
        _save_mask_union_from_arrays(BBX_STATE.saved_masks, Path(os.environ["GRADIO_TEMP_DIR"]) / "bbx_mask", "union"),
    )


def _run_command(cmd: list, cwd: Path, logs: list[str]) -> None:
    logs.append("[RUN] " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    logs.append(proc.stdout)
    if proc.returncode != 0:
        raise gr.Error("\n".join(logs[-3:]))


def _get_opacification_runtime(args):
    global OPACIFICATION_RUNTIME, OPACIFICATION_RUNTIME_KEY
    key = (
        str(Path(args.opacification_ckpt).resolve()),
        str(Path(args.config).resolve()),
        str(Path(args.mask_refiner_path).resolve()),
    )
    if OPACIFICATION_RUNTIME is None or OPACIFICATION_RUNTIME_KEY != key:
        OPACIFICATION_RUNTIME = OpacificationRuntime.from_args(args)
        OPACIFICATION_RUNTIME_KEY = key
    return OPACIFICATION_RUNTIME


def _run_opacification_cached(paths, layout: PreparedLayout, seed: int, unipc_steps: int, logs: list[str]) -> list[str]:
    argv = [
        "--opacification_ckpt",
        str(paths.opacification_ckpt),
        "--config",
        str(paths.config),
        "--rgb_dir",
        str(layout.rgb_dir),
        "--mask_dir",
        str(layout.mask_dir),
        "--output_base",
        str(layout.output_base),
        "--mask_refiner_path",
        str(paths.mask_refiner_path),
        "--unipc_steps",
        str(unipc_steps),
        "--seeds",
        str(seed),
        "--batch_size",
        "8",
        "--prep_mode",
        "fast",
    ]
    cmd = ["python", str(paths.repo_root / "scripts" / "infer_opacification.py"), *argv]
    logs.append("[RUN_CACHED] " + " ".join(cmd))
    args = build_arg_parser().parse_args(argv)
    runtime = _get_opacification_runtime(args)
    run_opacification(args, runtime=runtime)
    return cmd


def _run_depth_cached(paths, image_path: Path, out_dir: Path, stem: str, source: str, logs: list[str]) -> Path:
    if source == "da3":
        logs.append(f"[RUN_CACHED] Depth Anything V3 {image_path}")
        return DEPTH_RUNTIME_CACHE.run_da3(paths.da3_root, paths.depth_model, image_path, out_dir)
    if source == "moge":
        logs.append(f"[RUN_CACHED] MoGe-2 {image_path}")
        return DEPTH_RUNTIME_CACHE.run_moge(paths.moge_root, image_path, out_dir, stem)
    raise ValueError(f"Unsupported depth source: {source}")


def _refine_with_sam3(paths, image_path: Path, mask_input: Path, work_dir: Path, logs: list[str]) -> list[Path]:
    out_dir = work_dir / "sam3_refined_mask"
    cmd = build_sam3_refine_command(paths, image_path, mask_input, out_dir, "demo")
    _run_command(cmd, paths.repo_root, logs)
    return collect_mask_paths(out_dir / "refined" / "demo")


def _mask_paths_for_source(
    mask_source: str,
    gsam_prompt: str,
    use_grouped_prompt: bool,
    paths,
    image_path: Path,
    mask,
    work_dir: Path,
    logs: list[str],
) -> list[Path]:
    if mask_source == "Trans4Trans auto":
        out_dir = work_dir / "trans4trans_mask"
        cmd = build_trans4trans_command(paths, image_path, out_dir)
        _run_command(cmd, paths.repo_root, logs)
        return collect_mask_paths(out_dir)

    if mask_source == "Grounded SAM 2 text":
        prompt = gsam_prompt
        synonyms_per_object = 3 if use_grouped_prompt else 1
        if not use_grouped_prompt:
            prompt, synonyms_per_object = _manual_gsam_prompt(gsam_prompt)
        logs.append(f"[Grounded SAM 2 prompt] {prompt}")
        out_dir = work_dir / "grounded_sam2_mask"
        cmd = build_gsam2_command(
            paths,
            image_path,
            prompt,
            out_dir,
            "demo",
            synonyms_per_object=synonyms_per_object,
        )
        _run_command(cmd, paths.repo_root, logs)
        return collect_mask_paths(out_dir)

    if mask_source == "SAM3 click":
        if not SAM3_STATE.saved_masks:
            raise gr.Error("SAM3 click needs saved object masks. Click points for one object, then click 'Save object mask'.")
        return _save_mask_arrays(SAM3_STATE.saved_masks, work_dir / "sam3_click")

    if mask_source == "Upload mask":
        if mask is None:
            raise gr.Error("Upload a transparent-object mask.")
    else:
        raise gr.Error(f"Unknown mask source: {mask_source}")

    return _save_upload_mask_instances(mask, work_dir / "upload_mask_instances")


def generate_mask(
    image,
    mask,
    mask_source: str,
    gsam_prompt: str,
    use_sam3_refine: bool,
):
    paths = default_paths()
    work_dir = Path(tempfile.mkdtemp(prefix="transparent_mask_demo_", dir=os.environ["GRADIO_TEMP_DIR"]))
    image_signature = _image_signature(image)
    image_path = _save_upload_image(image, work_dir / "upload" / "demo.png")
    prompt_for_run = gsam_prompt
    use_grouped_prompt = False
    if mask_source == "Grounded SAM 2 text" and GPT_PROMPT_STATE.matches(image_signature, gsam_prompt):
        prompt_for_run = GPT_PROMPT_STATE.full_prompt
        use_grouped_prompt = True
    logs = [f"[work_dir] {work_dir}"]
    mask_paths = _mask_paths_for_source(
        mask_source,
        prompt_for_run,
        use_grouped_prompt,
        paths,
        image_path,
        mask,
        work_dir,
        logs,
    )
    if not mask_paths:
        raise gr.Error(f"No mask was produced by {mask_source}.")
    if mask_source == "Trans4Trans auto" and use_sam3_refine:
        mask_input_dir = work_dir / "trans4trans_masks_for_refine"
        _copy_mask_paths(mask_paths, mask_input_dir)
        mask_paths = _refine_with_sam3(paths, image_path, mask_input_dir, work_dir, logs)
        if not mask_paths:
            raise gr.Error("SAM3 refine produced no masks.")
    MASK_STATE.set(mask_paths, image_signature, mask_source)
    OPAQUE_STATE.clear()
    overlay = _overlay_mask_paths_colored(_normalize_image(image), mask_paths, alpha=0.52)
    return overlay


def refine_mask_with_sam3(image):
    if not MASK_STATE.mask_paths:
        raise gr.Error("Generate a mask first.")
    if image is None:
        raise gr.Error("Upload an RGB image first.")
    paths = default_paths()
    work_dir = Path(tempfile.mkdtemp(prefix="transparent_refine_", dir=os.environ["GRADIO_TEMP_DIR"]))
    image_path = _save_upload_image(image, work_dir / "upload" / "demo.png")
    union_path = work_dir / "union_mask.png"
    _union_masks(MASK_STATE.mask_paths, union_path)
    logs = []
    refined_paths = _refine_with_sam3(paths, image_path, union_path, work_dir, logs)
    if not refined_paths:
        raise gr.Error("SAM3 refine produced no masks.")
    MASK_STATE.set(refined_paths, MASK_STATE.image_signature, MASK_STATE.source)
    OPAQUE_STATE.clear()
    refined_union_path = work_dir / "refined_mask_union.png"
    overlay = _overlay_mask_paths_colored(_normalize_image(image), refined_paths, alpha=0.52)
    return str(_union_masks(refined_paths, refined_union_path)), overlay


def generate_opaque(
    image,
    mask,
    mask_source: str,
    seed: int,
    unipc_steps: int,
):
    paths = default_paths()
    if not MASK_STATE.mask_paths and mask_source == "Upload mask" and mask is not None:
        image_signature = _image_signature(image)
        work_dir_for_mask = Path(tempfile.mkdtemp(prefix="transparent_upload_mask_", dir=os.environ["GRADIO_TEMP_DIR"]))
        mask_paths_for_upload = _save_upload_mask_instances(mask, work_dir_for_mask / "upload_mask_instances")
        MASK_STATE.set(mask_paths_for_upload, image_signature, "Upload mask")
    if not MASK_STATE.mask_paths:
        raise gr.Error("Generate a mask first.")
    if MASK_STATE.image_signature != _image_signature(image):
        raise gr.Error("The RGB image changed after mask generation. Click 'Generate mask' again.")
    work_dir = Path(tempfile.mkdtemp(prefix="transparent_opaque_demo_", dir=os.environ["GRADIO_TEMP_DIR"]))
    image_path = _save_upload_image(image, work_dir / "upload" / "demo.png")
    logs = [f"[work_dir] {work_dir}"]
    mask_paths = MASK_STATE.mask_paths

    layout = prepare_single_image_layout(
        image_path=image_path,
        mask_paths=mask_paths,
        work_dir=work_dir / "run",
        stem="demo",
    )
    commands = [_run_opacification_cached(paths, layout, int(seed), int(unipc_steps), logs)]
    commands_txt = work_dir / "commands.txt"
    write_commands(commands, commands_txt)

    logs.append(f"[commands] {commands_txt}")

    outputs = expected_outputs(layout)
    materialize_opaque_image(layout)
    OPAQUE_STATE.set(layout, MASK_STATE.image_signature, outputs["optimized_opaque"])
    opaque_path = str(outputs["optimized_opaque"])
    return opaque_path, None, None, gr.update(value=opaque_path, visible=True), None, None, None


def generate_depth(image, depth_source: str):
    paths = default_paths()
    if OPAQUE_STATE.layout is None or OPAQUE_STATE.opaque_path is None:
        raise gr.Error("Generate opaque image first.")
    if OPAQUE_STATE.image_signature is not None and OPAQUE_STATE.image_signature != _image_signature(image):
        raise gr.Error("The RGB image changed after opaque generation. Regenerate mask and opaque image.")
    if not Path(OPAQUE_STATE.opaque_path).is_file():
        raise gr.Error("Opaque output is missing. Generate opaque image again.")
    depth_key = "da3" if depth_source == "Depth Anything V3" else "moge"
    work_dir = Path(tempfile.mkdtemp(prefix="transparent_depth_demo_", dir=os.environ["GRADIO_TEMP_DIR"]))
    logs = [f"[work_dir] {work_dir}"]
    cmd = build_depth_command(paths, OPAQUE_STATE.layout, depth_key)
    write_commands([cmd], work_dir / "commands.txt")
    outputs = expected_outputs(OPAQUE_STATE.layout)
    stem = OPAQUE_STATE.layout.stem
    depth_dir = OPAQUE_STATE.layout.depth_dir
    _run_depth_cached(paths, result_image_path(OPAQUE_STATE.layout), depth_dir, stem, depth_key, logs)
    inst_pre_path = depth_dir / f"{stem}_instance_depth_pre.png"
    bl_color_path = depth_dir / f"{stem}_baseline_color.png"
    bl_inst_path  = depth_dir / f"{stem}_baseline_instance.png"

    mask_paths = MASK_STATE.mask_paths
    if not mask_paths and OPAQUE_STATE.layout is not None:
        mask_paths = collect_mask_paths(OPAQUE_STATE.layout.mask_dir / OPAQUE_STATE.layout.stem)

    if outputs["depth_npy"].is_file():
        save_depth_visualizations(outputs["depth_npy"], outputs["depth_gray"], outputs["depth_color"])
        if mask_paths:
            save_depth_per_instance_vis(outputs["depth_npy"], mask_paths, inst_pre_path)

    # Baseline: run same depth model on original RGB image
    if image is not None:
        bl_dir = depth_dir / "baseline"
        bl_dir.mkdir(parents=True, exist_ok=True)
        bl_img_path = bl_dir / f"{stem}.png"
        _save_upload_image(image, bl_img_path)
        bl_cmd = build_baseline_depth_command(paths, bl_img_path, bl_dir, stem, depth_key)
        write_commands([bl_cmd], work_dir / "commands_baseline.txt")
        _run_depth_cached(paths, bl_img_path, bl_dir, stem, depth_key, logs)
        bl_npy = bl_dir / f"{stem}.npy"
        if bl_npy.is_file():
            save_depth_color(bl_npy, bl_color_path)
            if mask_paths:
                save_depth_per_instance_vis(bl_npy, mask_paths, bl_inst_path)

    pre_str    = str(inst_pre_path) if inst_pre_path.is_file() else None
    bl_col_str = str(bl_color_path) if bl_color_path.is_file() else None
    bl_ins_str = str(bl_inst_path)  if bl_inst_path.is_file()  else None
    return str(outputs["depth_gray"]), str(outputs["depth_color"]), pre_str, bl_col_str, bl_ins_str


def save_all(image, save_dir_str):
    import datetime
    if not save_dir_str or not save_dir_str.strip():
        raise gr.Error("Specify a save directory.")
    out_dir = Path(save_dir_str.strip()) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved, missing = [], []

    if image is not None:
        _save_upload_image(image, out_dir / "rgb.png")
        saved.append("rgb.png")

    if OPAQUE_STATE.opaque_path and Path(OPAQUE_STATE.opaque_path).is_file():
        shutil.copy2(OPAQUE_STATE.opaque_path, out_dir / "opaque.png")
        saved.append("opaque.png")

    if OPAQUE_STATE.layout is not None:
        stem = OPAQUE_STATE.layout.stem
        outputs = expected_outputs(OPAQUE_STATE.layout)

        inst_mask_src_dir = OPAQUE_STATE.layout.mask_dir / stem
        inst_mask_files = sorted(inst_mask_src_dir.glob("*.png")) if inst_mask_src_dir.is_dir() else []
        if inst_mask_files:
            dst_mask_dir = out_dir / "instance_mask"
            dst_mask_dir.mkdir(parents=True, exist_ok=True)
            for src in inst_mask_files:
                shutil.copy2(src, dst_mask_dir / src.name)
            saved.append(f"instance_mask/ ({len(inst_mask_files)} file{'s' if len(inst_mask_files) > 1 else ''})")
        else:
            missing.append("instance_mask")

        color_src = outputs["depth_color"]
        if color_src.is_file():
            shutil.copy2(color_src, out_dir / "depth_color.png")
            saved.append("depth_color.png")
        else:
            missing.append("depth_color")
        inst_src = OPAQUE_STATE.layout.depth_dir / f"{stem}_instance_depth_pre.png"
        if inst_src.is_file():
            shutil.copy2(inst_src, out_dir / "depth_instance.png")
            saved.append("depth_instance.png")
        else:
            missing.append("depth_instance (Generate depth first)")
        for bl_name, bl_file in [
            ("baseline_color.png",    f"{stem}_baseline_color.png"),
            ("baseline_instance.png", f"{stem}_baseline_instance.png"),
        ]:
            bl_src = OPAQUE_STATE.layout.depth_dir / bl_file
            if bl_src.is_file():
                shutil.copy2(bl_src, out_dir / bl_name)
                saved.append(bl_name)
            else:
                missing.append(bl_name)

    lines = [f"Saved to: {out_dir}"] + [f"  ✓ {s}" for s in saved]
    if missing:
        lines += [f"  — missing: {', '.join(missing)}"]
    return "\n".join(lines)


def upload_opaque(image, mask, opaque_path):
    """User dragged in their own opaque image — build a synthetic PreparedLayout so generate_depth works."""
    if opaque_path is None:
        return
    stem = "demo"
    work_dir = Path(tempfile.mkdtemp(prefix="transparent_upload_opaque_", dir=os.environ["GRADIO_TEMP_DIR"]))
    output_base = work_dir / "opaque"
    blend_dir = output_base.with_name(f"{output_base.name}_0") / "blend"
    blend_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = work_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    dest = blend_dir / f"{stem}_result.png"
    shutil.copy2(opaque_path, dest)
    layout = PreparedLayout(
        rgb_dir=work_dir / "rgb",
        mask_dir=work_dir / "mask",
        output_base=output_base,
        depth_dir=depth_dir,
        stem=stem,
    )
    # image may be None if user uploads opaque without an RGB image — store None to skip signature check
    sig = _image_signature(image) if image is not None else None
    if mask is not None:
        mask_paths = _save_upload_mask_instances(mask, work_dir / "mask" / stem)
        MASK_STATE.set(mask_paths, sig, "Upload mask")
    elif MASK_STATE.mask_paths:
        mask_paths = _copy_mask_paths(MASK_STATE.mask_paths, work_dir / "mask" / stem)
        MASK_STATE.set(mask_paths, sig, MASK_STATE.source)
    else:
        (work_dir / "mask" / stem).mkdir(parents=True, exist_ok=True)
    OPAQUE_STATE.set(layout, sig, str(dest), is_uploaded=True)
    return gr.update(value=str(dest), visible=True)


with gr.Blocks(title="SeeClear Demo") as demo:
    gr.Markdown("# SeeClear")
    gr.Markdown(
        "Reliable Transparent Object Depth Estimation via Generative Opacification"
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            default_mask_path = _default_example_mask_path()
            image_in = gr.Image(
                label="RGB Image",
                type="pil",
                value=str(DEFAULT_EXAMPLE_IMAGE) if DEFAULT_EXAMPLE_IMAGE.is_file() else None,
            )
            mask_in = gr.Image(
                label="Optional uploaded mask",
                type="pil",
                image_mode="L",
                value=str(default_mask_path) if default_mask_path is not None else None,
            )
            mask_source = gr.Radio(
                [
                    "Upload mask",
                    "Manual BBX",
                    "SAM3 click",
                    "Grounded SAM 2 text",
                    "Trans4Trans auto",
                ],
                value="Trans4Trans auto",
                label="Mask source",
            )
            if DEFAULT_EXAMPLE_IMAGE.is_file() and default_mask_path is not None:
                gr.Examples(
                    examples=[[str(DEFAULT_EXAMPLE_IMAGE), str(default_mask_path), "Trans4Trans auto"]],
                    inputs=[image_in, mask_in, mask_source],
                    label="Example",
                )
        with gr.Column(scale=1):
            with gr.Group(visible=False) as sam3_group:
                sam3_preview = gr.Image(label="SAM3 interactive mask", type="numpy", interactive=True)
                with gr.Row():
                    point_mode = gr.Radio(
                        ["positive", "negative"],
                        value="positive",
                        label="Point mode",
                        container=False,
                        scale=2,
                        min_width=180,
                    )
                    multimask = gr.Checkbox(label="Multi-mask", value=True, container=False, scale=1, min_width=120)
                with gr.Row():
                    sam3_undo_btn = gr.Button("Undo point")
                    sam3_clear_btn = gr.Button("Clear points")
                with gr.Row():
                    sam3_save_btn = gr.Button("Save object mask")
                    sam3_clear_saved_btn = gr.Button("Clear saved masks")
                sam3_union_out = gr.Image(label="Saved masks union", type="filepath")
            with gr.Group(visible=False) as bbx_group:
                bbx_preview = gr.Image(label="Manual quadrilateral mask overlay", type="numpy", interactive=True)
                with gr.Row():
                    bbx_clear_btn = gr.Button("Clear polygon")
                    bbx_confirm_btn = gr.Button("Save polygon mask", variant="primary")
                bbx_union_out = gr.Image(label="Saved masks union", type="filepath")
            with gr.Group(visible=False) as gsam_group:
                gsam_prompt = gr.Textbox(
                    label="Grounded SAM 2 prompt",
                    placeholder="bottle, glass, cup",
                    lines=2,
                )
                # GPT prompt generation is disabled for this release.
                # gpt_prompt_btn = gr.Button("Generate GPT prompt", variant="secondary")
                gsam2_run_btn = gr.Button("Run Grounded SAM 2", variant="secondary")
                gsam2_preview = gr.Image(label="Grounded SAM 2 mask overlay", type="numpy")
                with gr.Row():
                    gsam2_save_btn = gr.Button("Save mask")
                    gsam2_clear_saved_btn = gr.Button("Clear saved masks")
                gsam2_union_out = gr.Image(label="Saved masks union", type="filepath")
            with gr.Group(visible=True) as mask_generation_group:
                sam3_refine_btn = gr.Checkbox(label="Use SAM3 refine", value=False)
                generate_mask_btn = gr.Button("Generate mask", variant="secondary", visible=True)
                mask_overlay_out = gr.Image(label="Mask overlay", type="numpy")
            with gr.Row():
                seed = gr.Number(label="Seed", value=42, precision=0)
                steps = gr.Slider(label="UniPC Steps", minimum=4, maximum=30, value=10, step=1)
            generate_opaque_btn = gr.Button("Generate opaque", variant="primary")
        with gr.Column(scale=1):
            optimized_out = gr.Image(label="Optimized opaque (or upload your own)", type="filepath", interactive=True)
            opaque_download_file = gr.File(label="Download opaque", visible=False)
            depth_source = gr.Radio(
                ["Depth Anything V3", "MoGe-2"],
                value="Depth Anything V3",
                label="Depth source",
            )
            generate_depth_btn = gr.Button("Generate depth", variant="primary")
            depth_gray_out = gr.Image(label="Depth gray", type="filepath", visible=False)
            depth_color_out = gr.Image(label="Depth color", type="filepath")
            depth_instance_out = gr.Image(label="Per-instance depth", type="filepath")
            gr.Markdown("**Baseline (Original)**")
            baseline_color_out    = gr.Image(label="Baseline color depth", type="filepath")
            baseline_instance_out = gr.Image(label="Baseline per-instance depth", type="filepath")
            gr.Markdown("---")
            save_dir_input = gr.Textbox(
                label="Save directory",
                value="/nas/xiaoyingwang/seeclear/demo_saves",
                placeholder="/nas/xiaoyingwang/seeclear/demo_saves",
            )
            save_all_btn = gr.Button("Save", variant="secondary")
            save_status_out = gr.Textbox(label="Save status", lines=5, interactive=False)

    image_in.change(
        update_mask_processing_ui,
        inputs=[mask_source, image_in],
        outputs=[sam3_group, gsam_group, bbx_group, mask_generation_group, sam3_preview, sam3_union_out, bbx_preview, bbx_union_out, generate_mask_btn, sam3_refine_btn],
    )
    mask_source.change(
        update_mask_processing_ui,
        inputs=[mask_source, image_in],
        outputs=[sam3_group, gsam_group, bbx_group, mask_generation_group, sam3_preview, sam3_union_out, bbx_preview, bbx_union_out, generate_mask_btn, sam3_refine_btn],
    )
    image_in.change(clear_image_dependent_outputs, outputs=[gsam_prompt, optimized_out, depth_gray_out, depth_color_out, opaque_download_file, depth_instance_out, baseline_color_out, baseline_instance_out])
    image_in.change(lambda: None, outputs=[mask_overlay_out])
    mask_in.change(mask_input_changed, inputs=[image_in, mask_source, mask_in], outputs=[optimized_out, depth_gray_out, depth_color_out, opaque_download_file, depth_instance_out, baseline_color_out, baseline_instance_out])
    mask_in.change(lambda: None, outputs=[mask_overlay_out])
    mask_source.change(clear_generated_outputs, outputs=[optimized_out, depth_gray_out, depth_color_out, opaque_download_file, depth_instance_out, baseline_color_out, baseline_instance_out])
    mask_source.change(lambda: None, outputs=[mask_overlay_out])
    gsam_prompt.change(clear_generated_outputs, outputs=[optimized_out, depth_gray_out, depth_color_out, opaque_download_file, depth_instance_out, baseline_color_out, baseline_instance_out])
    # gpt_prompt_btn.click(generate_gpt_prompt, inputs=[image_in], outputs=[gsam_prompt])
    seed.change(clear_opaque_outputs, outputs=[optimized_out, depth_gray_out, depth_color_out, opaque_download_file, depth_instance_out, baseline_color_out, baseline_instance_out])
    steps.change(clear_opaque_outputs, outputs=[optimized_out, depth_gray_out, depth_color_out, opaque_download_file, depth_instance_out, baseline_color_out, baseline_instance_out])
    depth_source.change(lambda: (None, None, None, None, None), outputs=[depth_gray_out, depth_color_out, depth_instance_out, baseline_color_out, baseline_instance_out])
    sam3_preview.select(
        sam3_select_point,
        inputs=[point_mode, multimask, image_in],
        outputs=[sam3_preview],
    )
    sam3_undo_btn.click(sam3_undo, inputs=[multimask], outputs=[sam3_preview])
    sam3_clear_btn.click(sam3_clear, outputs=[sam3_preview])
    sam3_save_btn.click(
        sam3_save_object_mask,
        outputs=[sam3_preview, sam3_union_out],
    )
    sam3_clear_saved_btn.click(
        sam3_clear_saved_masks,
        outputs=[sam3_preview, sam3_union_out],
    )
    gsam2_run_btn.click(
        gsam2_run,
        inputs=[image_in, gsam_prompt],
        outputs=[gsam2_preview],
    )
    gsam2_save_btn.click(
        gsam2_save_mask,
        outputs=[gsam2_preview, gsam2_union_out],
    )
    gsam2_clear_saved_btn.click(
        gsam2_clear_saved,
        outputs=[gsam2_preview, gsam2_union_out],
    )
    image_in.change(gsam2_reset_on_image_change, outputs=[gsam2_preview, gsam2_union_out])
    mask_source.change(gsam2_reset_on_image_change, outputs=[gsam2_preview, gsam2_union_out])
    bbx_preview.select(bbx_select_point, inputs=[image_in], outputs=[bbx_preview])
    bbx_clear_btn.click(bbx_clear, outputs=[bbx_preview, bbx_union_out])
    bbx_confirm_btn.click(bbx_confirm_mask, outputs=[bbx_preview, bbx_union_out])
    generate_mask_btn.click(
        generate_mask,
        inputs=[image_in, mask_in, mask_source, gsam_prompt, sam3_refine_btn],
        outputs=[mask_overlay_out],
    )

    generate_opaque_btn.click(
        generate_opaque,
        inputs=[
            image_in,
            mask_in,
            mask_source,
            seed,
            steps,
        ],
        outputs=[
            optimized_out,
            depth_gray_out,
            depth_color_out,
            opaque_download_file,
            depth_instance_out,
            baseline_color_out,
            baseline_instance_out,
        ],
    )
    generate_depth_btn.click(
        generate_depth,
        inputs=[
            image_in,
            depth_source,
        ],
        outputs=[
            depth_gray_out,
            depth_color_out,
            depth_instance_out,
            baseline_color_out,
            baseline_instance_out,
        ],
    )
    optimized_out.upload(
        upload_opaque,
        inputs=[image_in, mask_in, optimized_out],
        outputs=[opaque_download_file],
    )
    save_all_btn.click(
        save_all,
        inputs=[image_in, save_dir_input],
        outputs=[save_status_out],
    )


if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    demo.queue().launch(
        server_name=server_name,
        server_port=_select_server_port(server_name),
        allowed_paths=[os.environ["GRADIO_TEMP_DIR"], str(REPO_ROOT)],
    )
