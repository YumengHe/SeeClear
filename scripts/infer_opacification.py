import torch
import os, cv2, argparse, glob, sys, time, json, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch import autocast
from torchvision.transforms import Resize
import torchvision.transforms as T
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.uni_pc_sampler import UniPCSampler
from mask_refiner.model import PixelRefineHead


def _collect_input_masks(mask_path):
    path = Path(mask_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        )
    raise FileNotFoundError(f"Mask path does not exist: {mask_path}")


def _prepare_single_image_layout(image_path, mask_path, work_dir, stem):
    image_path = Path(image_path)
    work_dir = Path(work_dir)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    mask_paths = _collect_input_masks(mask_path)
    if not mask_paths:
        raise FileNotFoundError(f"No mask images found in: {mask_path}")

    rgb_dir = work_dir / 'rgb'
    mask_root = work_dir / 'mask'
    stem_mask_dir = mask_root / stem
    rgb_dir.mkdir(parents=True, exist_ok=True)
    stem_mask_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, rgb_dir / f"{stem}{image_path.suffix.lower()}")
    for idx, src in enumerate(mask_paths):
        shutil.copy2(src, stem_mask_dir / f"mask_{idx}.png")

    return str(rgb_dir), str(mask_root), str(work_dir / 'opaque')


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_original_blend_result_npy(img_np, out_dir, img_name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    np.save(out_dir / f"{img_name}_result.npy", np.ascontiguousarray(img_np))
    return 0.0, time.perf_counter() - t0


def get_tensor(normalize=True):
    if normalize:
        return T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    else:
        return T.ToTensor()

def to_pil(img_tensor):
    img = img_tensor.detach().cpu().float()
    img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
    img = img.permute(1, 2, 0).numpy() * 255
    return Image.fromarray(img.astype(np.uint8))

def detect_encoder_type(model):
    if hasattr(model, 'cond_stage_model'):
        name = model.cond_stage_model.__class__.__name__
        if 'Dino' in name or 'DINO' in name: return 'dinov2'
        if 'CLIP' in name or 'Clip' in name: return 'clip'
    return 'clip'

def prepare_ref_image_clip(img_tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1).to(img_tensor.device)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1).to(img_tensor.device)
    img_01 = (img_tensor + 1.0) / 2.0
    return (img_01 - mean) / std

def prepare_ref_image_dinov2(img_tensor):
    return (img_tensor + 1.0) / 2.0

def filter_and_postprocess(pred_bin, prompt_mask_bin):
    num_labels, labels = cv2.connectedComponents(pred_bin, connectivity=8)
    filtered = np.zeros_like(pred_bin)
    for i in range(1, num_labels):
        comp = (labels == i)
        if np.any(comp & (prompt_mask_bin > 0)): filtered[comp] = 255
    kernel_small  = np.ones((3,3), np.uint8)
    kernel_smooth = np.ones((5,5), np.uint8)
    temp  = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_small)
    final = cv2.morphologyEx(temp,     cv2.MORPH_OPEN,  kernel_smooth)
    return final

def get_patch_and_coords(full_image_pil, mask_pil, min_area=10, patch_ratio=0.6):
    mask_np = np.array(mask_pil)
    if mask_np.ndim == 3: mask_np = mask_np[:, :, 0]
    _, mask_bin = cv2.threshold(mask_np.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    if np.sum(mask_bin > 0) < min_area: return None
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours: return None
    all_points = np.concatenate(valid_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    cx, cy = x + w // 2, y + h // 2
    longest_side = max(w, h)
    patch_size = max(256, int(longest_side / patch_ratio))
    half_size  = patch_size // 2
    x1_ideal, y1_ideal = cx - half_size, cy - half_size
    x1_src = max(0, x1_ideal)
    y1_src = max(0, y1_ideal)
    x2_src = min(full_image_pil.width,  x1_ideal + patch_size)
    y2_src = min(full_image_pil.height, y1_ideal + patch_size)
    x1_dst = x1_src - x1_ideal
    y1_dst = y1_src - y1_ideal
    x2_dst = x1_dst + (x2_src - x1_src)
    y2_dst = y1_dst + (y2_src - y1_src)
    patch_image_np = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    patch_mask_np  = np.zeros((patch_size, patch_size),    dtype=np.uint8)
    if x1_src < x2_src and y1_src < y2_src:
        patch_image_np[y1_dst:y2_dst, x1_dst:x2_dst] = np.array(full_image_pil)[y1_src:y2_src, x1_src:x2_src]
        patch_mask_np [y1_dst:y2_dst, x1_dst:x2_dst] = mask_bin[y1_src:y2_src, x1_src:x2_src]
    info = {
        'paste_coords':           (x1_src, y1_src, x2_src, y2_src),
        'crop_coords_from_patch': (x1_dst, y1_dst, x2_dst, y2_dst),
        'actual_patch_size':      patch_size,
    }
    return Image.fromarray(patch_image_np), Image.fromarray(patch_mask_np), info

def _bbox_from_mask_np(mask_bin, min_area=10):
    if int(mask_bin.sum() // 255) < min_area:
        return None
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid: return None
    return cv2.boundingRect(np.concatenate(valid))


def _extract_patch_np(img_np, mask_bin, x, y, w, h, patch_ratio=0.6):
    H_full, W_full = img_np.shape[:2]
    cx, cy = x + w // 2, y + h // 2
    patch_size = max(256, int(max(w, h) / patch_ratio))
    half = patch_size // 2
    x1_i, y1_i = cx - half, cy - half
    x1s, y1s = max(0, x1_i), max(0, y1_i)
    x2s, y2s = min(W_full, x1_i + patch_size), min(H_full, y1_i + patch_size)
    x1d, y1d = x1s - x1_i, y1s - y1_i
    x2d, y2d = x1d + (x2s - x1s), y1d + (y2s - y1s)
    patch_img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    patch_msk = np.zeros((patch_size, patch_size), dtype=np.uint8)
    if x1s < x2s and y1s < y2s:
        patch_img[y1d:y2d, x1d:x2d] = img_np[y1s:y2s, x1s:x2s]
        patch_msk[y1d:y2d, x1d:x2d] = mask_bin[y1s:y2s, x1s:x2s]
    info = {'paste_coords': (x1s, y1s, x2s, y2s),
            'crop_coords_from_patch': (x1d, y1d, x2d, y2d),
            'actual_patch_size': patch_size}
    return patch_img, patch_msk, info


def _gen_heatmap_np(mask_patch_np, mode, sigma=10):
    h, w = mask_patch_np.shape[:2]
    if mode == 'mask':
        hm = np.zeros((h, w), dtype=np.uint8)
        hm[mask_patch_np > 127] = 255
        return hm
    if mode == 'bbox':
        ys, xs = np.where(mask_patch_np > 127)
        hm = np.zeros((h, w), dtype=np.uint8)
        if len(xs):
            hm[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 255
        return hm
    ys, xs = np.where(mask_patch_np > 127)
    if len(xs) == 0:
        cx, cy = w // 2, h // 2
    else:
        idx = np.random.randint(len(xs))
        cx, cy = int(xs[idx]), int(ys[idx])
    yg, xg = np.ogrid[:h, :w]
    return (np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2)) * 255).astype(np.uint8)


def _read_mask_np_resized(args):
    mf, target_wh = args
    mask_pil = Image.open(mf).convert('L')
    if mask_pil.size != target_wh:
        mask_pil = mask_pil.resize(target_wh, Image.NEAREST)
    return np.array(mask_pil)


def generate_heatmap_in_patch(mask_patch_np, mode, sigma=10):
    h, w = mask_patch_np.shape[:2]
    if mask_patch_np.ndim == 3: mask_patch_np = mask_patch_np[:, :, 0]
    ys, xs = np.where(mask_patch_np > 127)
    if mode == 'point':
        if len(xs) == 0: cx, cy = w//2, h//2
        else:
            idx = np.random.randint(len(xs))
            cx, cy = int(xs[idx]), int(ys[idx])
        y_g, x_g = np.ogrid[:h, :w]
        return (np.exp(-((x_g-cx)**2 + (y_g-cy)**2) / (2*sigma**2)) * 255).astype(np.uint8)
    elif mode == 'bbox':
        hm = np.zeros((h, w), dtype=np.uint8)
        if len(xs) > 0: hm[ys.min():ys.max()+1, xs.min():xs.max()+1] = 255
        return hm
    else:  # mask
        hm = np.zeros((h, w), dtype=np.uint8)
        hm[mask_patch_np > 127] = 255
        return hm

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dir',        type=str, default=None)
    parser.add_argument('--mask_dir',       type=str, default=None)
    parser.add_argument('--output_base',    type=str, default=None)
    parser.add_argument('--image',          type=str, default=None,
                        help='Single input image. If set, --mask and --work_dir are required.')
    parser.add_argument('--mask',           type=str, default=None,
                        help='Single mask image or a directory of mask images for --image.')
    parser.add_argument('--work_dir',       type=str, default=None,
                        help='Temporary/output directory used with --image and --mask.')
    parser.add_argument('--stem',           type=str, default='demo',
                        help='Output stem used with --image and --mask.')
    parser.add_argument('--opacification_ckpt', type=str, required=True)
    parser.add_argument('--config',         type=str, required=True)
    parser.add_argument('--mask_refiner_path', type=str, required=True)
    parser.add_argument('--unipc_steps',     type=int, default=10)
    parser.add_argument('--scale',          type=float, default=1.0)
    parser.add_argument('--seeds',          type=int, nargs='+', default=[42])
    parser.add_argument('--batch_size',     type=int, default=1,
                        help='Batch size for diffusion sampling within one image. '
                             'When >1, masks of the same image are stacked into one '
                             'sampler.sample call to amortize UNet forward cost.')
    parser.add_argument('--timing_json',    type=str, default=None,
                        help='If set, dump per-seed per-stem wall-clock timing '
                             '(t_prep, t_diff, t_head, t_blend, n_masks) as JSON; '
                             "and a top-level 't_load' (diffusion + mask head load wall-clock).")
    parser.add_argument('--phase3_workers', type=int, default=8,
                        help='ThreadPool worker count for phase 3 blend+save '
                             'across stems (each stem stays sequential internally).')
    parser.add_argument('--phase3_mask_workers', type=int, default=4,
                        help='Per-stem worker count for local patch resize/prep in '
                             'phase 3 before the final ordered canvas writes.')
    parser.add_argument('--overwrite_existing', action='store_true',
                        help='Recompute images even if blend result files already exist. '
                             'Use this for accurate per-image timing on repeated runs.')
    parser.add_argument('--prep_mode', choices=['legacy', 'fast'], default='fast',
                        help='legacy = original PIL LANCZOS + per-mask .cuda. '
                             'fast = threaded mask read + cv2 LANCZOS4 + numpy patch + batched H2D. '
                             '~3-4x faster on NAS-bound prep; pixel-level diff vs legacy ~0.25%%.')
    parser.add_argument('--prep_mask_workers', type=int, default=8,
                        help='Threads used for parallel mask file reads (fast mode only).')
    return parser


class OpacificationRuntime:
    def __init__(self, model, sampler, encoder_type, mask_refiner, load_time):
        self.model = model
        self.sampler = sampler
        self.encoder_type = encoder_type
        self.mask_refiner = mask_refiner
        self.load_time = load_time

    @classmethod
    def from_args(cls, args):
        print(f"[Phase 0] Loading opacification model: {args.opacification_ckpt}")
        t0 = time.perf_counter()
        model = instantiate_from_config(OmegaConf.load(args.config).model).cuda().eval()
        model.load_state_dict(torch.load(args.opacification_ckpt, map_location="cpu")["state_dict"], strict=False)
        sampler = UniPCSampler(model)
        encoder_type = detect_encoder_type(model)
        mask_refiner = PixelRefineHead().cuda().eval()
        mask_refiner.load_state_dict(torch.load(args.mask_refiner_path, map_location="cuda"))
        load_time = time.perf_counter() - t0
        print(f"  Encoder type: {encoder_type}")
        return cls(model, sampler, encoder_type, mask_refiner, load_time)

@torch.no_grad()
def run_opacification(args, runtime=None):

    single_image_mode = any([args.image, args.mask, args.work_dir])
    if single_image_mode:
        missing = [
            name for name, value in [
                ('--image', args.image),
                ('--mask', args.mask),
                ('--work_dir', args.work_dir),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} required when using single-image mode")
        args.rgb_dir, args.mask_dir, args.output_base = _prepare_single_image_layout(
            args.image, args.mask, args.work_dir, args.stem
        )
    else:
        missing = [
            name for name, value in [
                ('--rgb_dir', args.rgb_dir),
                ('--mask_dir', args.mask_dir),
                ('--output_base', args.output_base),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} required unless --image/--mask/--work_dir are used")

    # seed_idx -> { img_name -> {t_prep, t_diff, t_head, t_blend, n_masks} }
    all_timings = {}
    # Aggregate model-load wall-clock. For CLI this includes the one-time runtime
    # load. In the Gradio demo the runtime is cached by the server process.
    if runtime is None:
        runtime = OpacificationRuntime.from_args(args)
    t_load_total = runtime.load_time
    model = runtime.model
    sampler = runtime.sampler
    encoder_type = runtime.encoder_type
    m_head = runtime.mask_refiner

    diff_mode_type = 'mask'

    # =========================================================
    # =========================================================
    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n========== Seed {seed} (run {seed_idx}) ==========")
        np.random.seed(seed)
        torch.manual_seed(seed)

        cur_output_base = f"{args.output_base}_{seed_idx}"
        os.makedirs(cur_output_base, exist_ok=True)

        # Per-seed timing dict (img_name -> phase wall-clock seconds)
        timing = defaultdict(lambda: {'t_prep': 0.0, 't_diff': 0.0, 't_head': 0.0,
                                      't_blend': 0.0, 'n_masks': 0})
        all_timings[str(seed_idx)] = timing

        # =========================================================
        # =========================================================
        print(f"[Phase 1] Running diffusion...")
        entries = []

        stem_dirs = sorted([
            d for d in glob.glob(os.path.join(args.mask_dir, "*"))
            if os.path.isdir(d)
        ])
        print(f"  Found {len(stem_dirs)} images in mask_dir")

        for stem_dir in stem_dirs:
            img_name = os.path.basename(stem_dir)

            rgb_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                cand = os.path.join(args.rgb_dir, img_name + ext)
                if os.path.isfile(cand):
                    rgb_path = cand
                    break
            if rgb_path is None:
                print(f"  [WARN] No image found for {img_name}, skip")
                continue

            mask_files = sorted(glob.glob(os.path.join(stem_dir, "mask*.png")))
            if not mask_files:
                out_blend_dir = os.path.join(cur_output_base, 'blend')
                os.makedirs(out_blend_dir, exist_ok=True)
                img_np_fallback = np.array(Image.open(rgb_path).convert('RGB'))
                _, t_blend_save = save_original_blend_result_npy(img_np_fallback, out_blend_dir, img_name)
                timing[img_name]['t_blend'] += t_blend_save
                print(f"  [NO_MASK] {img_name}: saved original to blend/")
                continue

            out_blend_npy = os.path.join(cur_output_base, 'blend', f"{img_name}_result.npy")
            if not args.overwrite_existing and os.path.exists(out_blend_npy):
                print(f"  [SKIP] {img_name}")
                continue

            _t0_prep = time.perf_counter()
            img_pil = Image.open(rgb_path).convert('RGB')
            orig_w, orig_h = img_pil.size
            img_np = np.array(img_pil)

            # ---- CPU prep: gather per-mask patch tensors before batching ----
            W, H = 512, 512
            per_mask = []  # list of dicts ready for batched diffusion

            if args.prep_mode == 'legacy':
                for m_idx, mf in enumerate(mask_files):
                    mask_pil = Image.open(mf).convert('L')
                    if mask_pil.size != img_pil.size:
                        mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)

                    if np.sum(np.array(mask_pil) > 127) < 10:
                        continue

                    patch_data = get_patch_and_coords(img_pil, mask_pil, patch_ratio=0.6)
                    if patch_data is None:
                        continue
                    patch_pil, mask_patch_pil, info = patch_data

                    patch_512 = patch_pil.resize((W, H), Image.LANCZOS)
                    patch_tensor = get_tensor()(patch_512).unsqueeze(0).cuda()

                    mask_patch_np = np.array(mask_patch_pil)
                    heatmap_patch_np = generate_heatmap_in_patch(mask_patch_np, diff_mode_type)
                    heatmap_patch_512 = Image.fromarray(heatmap_patch_np).resize((W, H), Image.NEAREST)
                    hm_patch_tensor = get_tensor(normalize=False)(heatmap_patch_512).unsqueeze(0).cuda()

                    per_mask.append({
                        'm_idx':            m_idx,
                        'patch_tensor':     patch_tensor,
                        'hm_patch_tensor':  hm_patch_tensor,
                        'mask_patch_np':    mask_patch_np,
                        'mask_np_full':     np.array(mask_pil),
                        'info':             info,
                    })
            else:
                # fast: threaded mask read + numpy-only patch + cv2 LANCZOS4 + batched H2D
                target_wh = img_pil.size  # (W, H) for PIL
                worker_n = max(1, min(args.prep_mask_workers, len(mask_files)))
                with ThreadPoolExecutor(max_workers=worker_n) as ex:
                    masks_full_np = list(ex.map(_read_mask_np_resized,
                                                [(mf, target_wh) for mf in mask_files]))

                patch_arrs, hm_arrs, infos, mask_patches, mask_fulls, kept_idx = \
                    [], [], [], [], [], []
                for m_idx, mask_full_np in enumerate(masks_full_np):
                    if int((mask_full_np > 127).sum()) < 10:
                        continue
                    _, mask_bin = cv2.threshold(mask_full_np.astype(np.uint8),
                                                127, 255, cv2.THRESH_BINARY)
                    bbox = _bbox_from_mask_np(mask_bin)
                    if bbox is None:
                        continue
                    x, y, w, h = bbox
                    patch_img, patch_msk, info = _extract_patch_np(img_np, mask_bin, x, y, w, h)
                    patch_512 = cv2.resize(patch_img, (W, H), interpolation=cv2.INTER_LANCZOS4)
                    hm = _gen_heatmap_np(patch_msk, diff_mode_type)
                    hm_512 = cv2.resize(hm, (W, H), interpolation=cv2.INTER_NEAREST)
                    patch_arrs.append(patch_512)
                    hm_arrs.append(hm_512)
                    infos.append(info)
                    mask_patches.append(patch_msk)
                    mask_fulls.append(mask_full_np)
                    kept_idx.append(m_idx)

                if patch_arrs:
                    patch_stack = np.stack(patch_arrs, 0).astype(np.float32) / 255.0
                    patch_stack = patch_stack.transpose(0, 3, 1, 2)
                    patch_stack = (patch_stack - 0.5) / 0.5
                    hm_stack = np.stack(hm_arrs, 0).astype(np.float32) / 255.0
                    hm_stack = hm_stack[:, None, :, :]
                    pt = torch.from_numpy(patch_stack).contiguous().cuda(non_blocking=True)
                    ht = torch.from_numpy(hm_stack).contiguous().cuda(non_blocking=True)
                    for i, m_idx in enumerate(kept_idx):
                        per_mask.append({
                            'm_idx':            m_idx,
                            'patch_tensor':     pt[i:i+1],
                            'hm_patch_tensor':  ht[i:i+1],
                            'mask_patch_np':    mask_patches[i],
                            'mask_np_full':     mask_fulls[i],
                            'info':             infos[i],
                        })

            _cuda_sync()
            timing[img_name]['t_prep'] += time.perf_counter() - _t0_prep

            if not per_mask:
                out_blend_dir = os.path.join(cur_output_base, 'blend')
                _, t_blend_save = save_original_blend_result_npy(img_np, out_blend_dir, img_name)
                timing[img_name]['t_blend'] += t_blend_save
                print(f"  [NO_VALID_MASK] {img_name}: saved original to blend/")
                continue

            # ---- Batched diffusion: chunk per_mask into args.batch_size groups ----
            bs = max(1, args.batch_size)
            for chunk_start in range(0, len(per_mask), bs):
                chunk = per_mask[chunk_start:chunk_start + bs]
                N = len(chunk)
                patch_batch = torch.cat([d['patch_tensor']    for d in chunk], dim=0)  # (N,3,512,512)
                hm_batch    = torch.cat([d['hm_patch_tensor'] for d in chunk], dim=0)  # (N,1,512,512)

                _cuda_sync()
                _t0 = time.perf_counter()
                with autocast("cuda"), model.ema_scope():
                    ref_imgs = torch.nn.functional.interpolate(patch_batch, size=(224,224), mode='bilinear')
                    ref_imgs = prepare_ref_image_dinov2(ref_imgs) if encoder_type == 'dinov2' else prepare_ref_image_clip(ref_imgs)
                    c = model.get_learned_conditioning(ref_imgs.to(torch.float16))
                    if c.shape[-1] == 1024: c = model.proj_out(c)
                    if len(c.shape) == 2: c = c.unsqueeze(1)
                    uc = model.learnable_vector if args.scale != 1.0 else None
                    if uc is not None and uc.shape[0] == 1 and N > 1:
                        uc = uc.expand(N, *uc.shape[1:])
                    z_inpaint = model.get_first_stage_encoding(model.encode_first_stage(patch_batch)).detach()
                    heatmap_resized = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(hm_batch)
                    samples, _ = sampler.sample(
                        S=args.unipc_steps, conditioning=c, batch_size=N,
                        shape=[4, W//8, H//8],
                        unconditional_guidance_scale=args.scale,
                        unconditional_conditioning=uc,
                        test_model_kwargs={'inpaint_image': z_inpaint, 'heatmap': heatmap_resized}
                    )
                    generated_batch = model.decode_first_stage(samples)
                _cuda_sync()
                timing[img_name]['t_diff']  += time.perf_counter() - _t0
                timing[img_name]['n_masks'] += N

                for i, d in enumerate(chunk):
                    gen_i = generated_batch[i:i+1].detach()
                    entries.append({
                        'img_name':         img_name,
                        'm_idx':            d['m_idx'],
                        'img_np':           img_np,
                        'orig_w':           orig_w, 'orig_h': orig_h,
                        'mask_np_full':     d['mask_np_full'],
                        'generated_cpu':    gen_i.cpu(),
                        'patch_tensor_cpu': d['patch_tensor'].cpu(),
                        'mask_patch_np':    d['mask_patch_np'],
                        'generated_pil':    to_pil(gen_i[0]),
                        'info':             d['info'],
                    })

            print(f"  Diffusion done: {img_name}  (bs={bs}, masks={len(per_mask)})")

        if not entries:
            print("No entries for this seed.")
            continue

        # =========================================================
        # =========================================================
        print(f"[Phase 2] Running mask head over {len(entries)} entries...")
        head_results = []

        head_results = [None] * len(entries)
        bs_head = max(1, args.batch_size)
        # Group entry indices by img_name so phase 2 batching is per-image
        # (within an image: N masks stacked into one m_head forward; across
        #  images: serial). Mirrors phase 1's per-image batching.
        img_to_idxs = defaultdict(list)
        for idx, entry in enumerate(entries):
            img_to_idxs[entry['img_name']].append(idx)

        for img_name, idx_list in img_to_idxs.items():
            for chunk_start in range(0, len(idx_list), bs_head):
                chunk_idx = idx_list[chunk_start:chunk_start + bs_head]
                chunk = [entries[i] for i in chunk_idx]
                N = len(chunk)

                hm_np_list = []
                gen_list, p_list, hm_ts_list = [], [], []
                for entry in chunk:
                    hm_np = generate_heatmap_in_patch(entry['mask_patch_np'], 'mask')
                    hm_np_list.append(hm_np)
                    hm_pil = Image.fromarray(hm_np).resize((512, 512), Image.NEAREST)
                    gen_list.append(entry['generated_cpu'].cuda())
                    p_list.append(entry['patch_tensor_cpu'].cuda())
                    hm_ts_list.append(get_tensor(normalize=False)(hm_pil).unsqueeze(0).cuda())
                gen_batch = torch.cat(gen_list, dim=0)
                p_batch   = torch.cat(p_list,   dim=0)
                hm_batch  = torch.cat(hm_ts_list, dim=0)

                _cuda_sync()
                _t0_gpu = time.perf_counter()
                logits = m_head(gen_batch, p_batch, hm_batch)
                pred_raw_batch = (torch.sigmoid(logits).squeeze(1).cpu().numpy() * 255).astype(np.uint8)
                _cuda_sync()
                gpu_t_chunk = time.perf_counter() - _t0_gpu

                # All chunk time belongs to img_name (per-image batching).
                timing[img_name]['t_head'] += gpu_t_chunk

                for i, entry in enumerate(chunk):
                    _t0_cpu = time.perf_counter()
                    pred_raw = pred_raw_batch[i]
                    hm_mask_512 = cv2.resize(hm_np_list[i], (512, 512), interpolation=cv2.INTER_NEAREST)
                    pred_mask_512 = filter_and_postprocess(pred_raw, hm_mask_512)

                    info = entry['info']
                    actual_size = info['actual_patch_size']
                    px_s, py_s, px_e, py_e = info['paste_coords']
                    cx_s, cy_s, cx_e, cy_e = info['crop_coords_from_patch']
                    if (py_e > py_s) and (px_e > px_s):
                        pred_mask_patch = cv2.resize(
                            pred_mask_512,
                            (actual_size, actual_size),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        head_results[chunk_idx[i]] = {
                            'bbox': (px_s, py_s, px_e, py_e),
                            'crop': (cx_s, cy_s, cx_e, cy_e),
                            'mask_local': pred_mask_patch[cy_s:cy_e, cx_s:cx_e] > 127,
                        }
                    else:
                        head_results[chunk_idx[i]] = None
                    timing[img_name]['t_head'] += time.perf_counter() - _t0_cpu

        # =========================================================
        # Cross-image ThreadPool: different stems' blend+save run concurrently.
        # t_blend = composite (LANCZOS resize + localized mask copy) + npy save per image.
        # =========================================================
        n_total = len(set(e['img_name'] for e in entries))
        print(f"[Phase 3] Blend+save for {n_total} stems  "
              f"(workers={args.phase3_workers}, mask_workers={args.phase3_mask_workers})")
        grouped = defaultdict(list)
        for i, entry in enumerate(entries):
            grouped[entry['img_name']].append(i)

        out_d = os.path.join(cur_output_base, 'blend')
        os.makedirs(out_d, exist_ok=True)

        def blend_and_save(img_name, idx_list_local):
            idx_list_sorted = sorted(idx_list_local, key=lambda i: entries[i]['m_idx'])
            first = entries[idx_list_sorted[0]]

            t0_blend = time.perf_counter()
            canvas = first['img_np'].copy()  # (H, W, 3) uint8

            def prepare_local_blend(i):
                e = entries[i]
                head = head_results[i]
                if head is None:
                    return None
                actual_size = e['info']['actual_patch_size']
                px_s, py_s, px_e, py_e = head['bbox']
                cx_s, cy_s, cx_e, cy_e = head['crop']
                if px_e <= px_s or py_e <= py_s:
                    return None
                gen_patch_full = np.asarray(
                    e['generated_pil'].resize((actual_size, actual_size), Image.LANCZOS)
                )
                gen_local = gen_patch_full[cy_s:cy_e, cx_s:cx_e]
                mask_local = head['mask_local']
                return e['m_idx'], (px_s, py_s, px_e, py_e), gen_local, mask_local

            mask_workers = max(1, args.phase3_mask_workers)
            if mask_workers > 1 and len(idx_list_sorted) > 1:
                with ThreadPoolExecutor(max_workers=min(mask_workers, len(idx_list_sorted))) as local_exe:
                    local_items = list(local_exe.map(prepare_local_blend, idx_list_sorted))
            else:
                local_items = [prepare_local_blend(i) for i in idx_list_sorted]

            for item in sorted((x for x in local_items if x is not None), key=lambda x: x[0]):
                _, (px_s, py_s, px_e, py_e), gen_local, mask_local = item
                canvas_local = canvas[py_s:py_e, px_s:px_e]
                canvas_local[mask_local] = gen_local[mask_local]
            np.save(os.path.join(out_d, f"{img_name}_result.npy"), canvas)
            t_blend = time.perf_counter() - t0_blend
            return img_name, t_blend

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, args.phase3_workers)) as exe:
            futs = [exe.submit(blend_and_save, img_name, idx_list)
                    for img_name, idx_list in grouped.items()]
            for fut in as_completed(futs):
                img_name, t_blend = fut.result()
                timing[img_name]['t_blend'] += t_blend
                done += 1
                print(f"  [Phase 3] [{done:>3d}/{n_total}] {img_name}: "
                      f"blend={t_blend:.2f}s", flush=True)

        print(f"Finished seed {seed} (run {seed_idx}). Output: {cur_output_base}")

    if args.timing_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.timing_json)) or ".", exist_ok=True)
        plain = {seed_k: {k: dict(v) for k, v in stems_t.items()}
                 for seed_k, stems_t in all_timings.items()}
        plain['t_load'] = t_load_total
        with open(args.timing_json, 'w') as f:
            json.dump(plain, f, indent=2)
        print(f"timing dumped -> {args.timing_json}  t_load={t_load_total:.2f}s")

    print(f"\nAll seeds finished.")

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_opacification(args)


if __name__ == "__main__":
    main()
