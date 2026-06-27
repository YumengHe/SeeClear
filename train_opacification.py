import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:
    from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import calculate_psnr, calculate_ssim
import socket
from pytorch_lightning.plugins.environments import ClusterEnvironment,SLURMEnvironment
import shutil 

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=["configs/stable-diffusion/v1-inference-inpaint.yaml"],
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="",
        help="path to pretrained model",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--train_from_scratch",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Train from scratch",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, monitor="train/loss_simple"):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {}
        if hasattr(pl.loggers, "TestTubeLogger"):
            self.logger_log_images[pl.loggers.TestTubeLogger] = self._testtube
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.monitor = monitor
        self.best_loss = float('inf')
        self.best_folder = None
        self.last_folder = None
        self.all_saved_folders = set()

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        folder_name = f"epoch_{current_epoch:04d}"
        root = os.path.join(save_dir, "images", split, folder_name)
        os.makedirs(root, exist_ok=True)
        
        self.last_folder = root
        self.all_saved_folders.add(root)

        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            
            filename = "{}_gs-{:06}.png".format(k, global_step)
            path = os.path.join(root, filename)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    def on_train_epoch_end(self, trainer, pl_module):
        current_loss = trainer.callback_metrics.get(self.monitor)
        
        if current_loss is not None:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_folder = self.last_folder
                if trainer.global_rank == 0:
                    print(f"[ImageLogger] New Best Epoch! Loss: {current_loss:.6f}, Keeping: {self.best_folder}")

        if trainer.global_rank == 0:
            for folder in list(self.all_saved_folders):
                if folder != self.best_folder and folder != self.last_folder:
                    try:
                        print(f"[ImageLogger] Deleting non-best folder: {folder}")
                        shutil.rmtree(folder)
                        self.all_saved_folders.remove(folder)
                    except Exception as e:
                        print(f"Error deleting folder {folder}: {e}")


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def _device_index(self, trainer):
        device = getattr(getattr(trainer, "strategy", None), "root_device", None)
        if device is not None and device.index is not None:
            return device.index
        return getattr(trainer, "root_gpu", 0)

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        device_index = self._device_index(trainer)
        torch.cuda.reset_peak_memory_stats(device_index)
        torch.cuda.synchronize(device_index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        device_index = self._device_index(trainer)
        torch.cuda.synchronize(device_index)
        max_memory = torch.cuda.max_memory_allocated(device_index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            strategy = getattr(trainer, "strategy", None)
            reducer = getattr(strategy, "reduce", None)
            if reducer is None:
                reducer = getattr(getattr(trainer, "training_type_plugin", None), "reduce", None)
            if reducer is not None:
                max_memory = reducer(max_memory)
                epoch_time = reducer(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class DebugCallback(Callback):
    """
     every_n_steps , debug .
    :<save_dir>/images/step_XXXXXX/
    :<save_dir>/debug.log
    """

    def __init__(self, save_dir, every_n_steps=5, log_filename="debug.log"):
        super().__init__()
        self.save_dir   = save_dir
        self.every_n    = every_n_steps
        self.log_path   = os.path.join(save_dir, log_filename)
        self._f         = None          # log file handle
        self._t_train   = None          # training start time
        self._t_step    = None          # per-step start time

    # ──────────────────────────── helpers ────────────────────────────

    def _log(self, msg):
        if self._f is not None:
            self._f.write(msg + "\n")
            self._f.flush()

    @rank_zero_only
    def _save_batch_images(self, batch, pl_module, global_step):
        step_dir = os.path.join(self.save_dir, "images", f"step_{global_step:06d}")
        os.makedirs(step_dir, exist_ok=True)
        B = batch['GT'].shape[0]

        def to_01_rgb(t, mode='img'):
            """Convert tensor to uint8 HxWx3 numpy for PIL."""
            t = t.detach().cpu().float()
            if mode == 'img':           # [-1, 1] -> [0, 1]
                t = (t.clamp(-1., 1.) + 1.0) / 2.0
            elif mode == 'mask':        # [0, 1] gray -> [0, 1] RGB
                t = t.clamp(0., 1.)
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
            elif mode == 'clip':        # CLIP normalised -> [0, 1]
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                t = (t * std + mean).clamp(0., 1.)
            arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return arr

        def save(arr, fname):
            Image.fromarray(arr).save(os.path.join(step_dir, fname))

        for i in range(B):
            save(to_01_rgb(batch['GT'][i],           'img'),  f"s{i}_1_GT.png")
            save(to_01_rgb(batch['inpaint_image'][i], 'img'),  f"s{i}_2_ref_transparent.png")
            save(to_01_rgb(batch['inpaint_mask'][i],  'mask'), f"s{i}_3_mask_aug.png")
            save(to_01_rgb(batch['M_gt'][i],          'mask'), f"s{i}_4_mask_gt.png")
            save(to_01_rgb(batch['ref_imgs'][i],      'clip'), f"s{i}_5_ref224_clip.png")

        try:
            with torch.no_grad():
                gt  = batch['GT'].to(pl_module.device)
                z   = pl_module.get_first_stage_encoding(pl_module.encode_first_stage(gt))
                rec = pl_module.decode_first_stage(z[:, :4])
            for i in range(B):
                save(to_01_rgb(rec[i], 'img'), f"s{i}_6_vae_recon.png")
            self._log(f"    [img] VAE recon OK")
        except Exception as e:
            self._log(f"    [img][WARN] VAE recon failed: {e}")

        try:
            with torch.no_grad():
                x, c   = pl_module.get_input(batch, pl_module.first_stage_key)
                t_low  = torch.full((B,), int(0.1 * pl_module.num_timesteps),
                                    device=pl_module.device, dtype=torch.long)
                noise  = torch.randn_like(x[:, :4])
                xn     = pl_module.q_sample(x[:, :4], t=t_low, noise=noise)
                xn_full= torch.cat((xn, x[:, 4:]), dim=1)
                out    = pl_module.apply_model(xn_full, t_low, c)
                pred_z = pl_module.predict_start_from_noise(xn, t=t_low, noise=out)
                pred   = pl_module.decode_first_stage(pred_z)
            for i in range(B):
                save(to_01_rgb(pred[i], 'img'), f"s{i}_7_pred_t10pct.png")
            self._log(f"    [img] model pred OK")
        except Exception as e:
            self._log(f"    [img][WARN] model pred failed: {e}")

        try:
            gt_row   = batch['GT'].clamp(-1., 1.)
            ref_row  = batch['inpaint_image'].clamp(-1., 1.)
            maug_row = batch['inpaint_mask'].repeat(1, 3, 1, 1).clamp(0., 1.) * 2 - 1
            mgt_row  = batch['M_gt'].repeat(1, 3, 1, 1).clamp(0., 1.) * 2 - 1
            all_imgs = torch.cat([gt_row, ref_row, maug_row, mgt_row], dim=0)
            grid = torchvision.utils.make_grid(all_imgs, nrow=B, normalize=True, value_range=(-1., 1.))
            arr  = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(step_dir, "grid_GT_ref_maskaug_mgt.png"))
            self._log(f"    [img] grid saved -> {step_dir}")
        except Exception as e:
            self._log(f"    [img][WARN] grid failed: {e}")

    # ──────────────────────────── hooks ────────────────────────────

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        self._f       = open(self.log_path, 'a')
        self._t_train = time.time()
        self._log(f"\n{'='*60}")
        self._log(f"[DebugCallback] started  {datetime.datetime.now()}")
        self._log(f"  save_dir    : {self.save_dir}")
        self._log(f"  every_n_steps: {self.every_n}")
        self._log(f"{'='*60}")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        self._t_step = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if trainer.global_rank != 0:
            return

        gs        = pl_module.global_step
        step_time = time.time() - self._t_step if self._t_step else 0.0
        metrics   = trainer.callback_metrics

        # ── text log ──
        self._log(f"\n[step {gs:06d} | epoch {trainer.current_epoch} | batch {batch_idx} | {step_time:.2f}s]")

        # losses
        for key in ['train/loss', 'train/loss_simple', 'train/loss_vlb',
                    'train/lpips_loss', 'train/loss_shade_grad', 'train/loss_masked_l1']:
            if key in metrics:
                self._log(f"  {key}: {metrics[key].item():.6f}")

        # per-sample mask area + augmentation info
        aug_mask = batch['inpaint_mask']   # (B, 1, H, W)  [0,1]
        gt_mask  = batch['M_gt']           # (B, 1, H, W)  [0,1]
        aug_info = batch.get('augmentation_info', None)
        B = aug_mask.shape[0]
        for i in range(B):
            aug_area = aug_mask[i].mean().item()
            gt_area  = gt_mask[i].mean().item()
            info_str = aug_info[i] if (aug_info is not None) else 'N/A'
            # IoU between aug_mask and gt_mask for this sample
            inter = ((aug_mask[i] > 0.5) & (gt_mask[i] > 0.5)).float().mean().item()
            union = ((aug_mask[i] > 0.5) | (gt_mask[i] > 0.5)).float().mean().item()
            iou   = inter / (union + 1e-6)
            self._log(f"  sample[{i}]: gt_area={gt_area:.4f}  aug_area={aug_area:.4f}  "
                      f"IoU={iou:.4f}  info={info_str}")

        # ── image save ──
        if gs % self.every_n == 0:
            self._log(f"  >> saving images (step {gs})")
            self._save_batch_images(batch, pl_module, gs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if trainer.global_rank != 0:
            return

        ep = trainer.current_epoch
        metrics = trainer.callback_metrics

        # ── text log ──
        self._log(f"\n[VAL | epoch {ep} | batch {batch_idx}]")
        for key in ['val/loss', 'val/loss_simple', 'val/loss_vlb', 'val/psnr', 'val/ssim']:
            if key in metrics:
                self._log(f"  {key}: {metrics[key].item():.6f}")

        # per-sample mask stats
        aug_mask = batch['inpaint_mask']
        gt_mask  = batch['M_gt']
        aug_info = batch.get('augmentation_info', None)
        B = aug_mask.shape[0]
        for i in range(B):
            aug_area = aug_mask[i].mean().item()
            gt_area  = gt_mask[i].mean().item()
            info_str = aug_info[i] if aug_info is not None else 'N/A'
            inter = ((aug_mask[i] > 0.5) & (gt_mask[i] > 0.5)).float().mean().item()
            union = ((aug_mask[i] > 0.5) | (gt_mask[i] > 0.5)).float().mean().item()
            iou   = inter / (union + 1e-6)
            self._log(f"  sample[{i}]: gt_area={gt_area:.4f}  aug_area={aug_area:.4f}  "
                      f"IoU={iou:.4f}  info={info_str}")

        # ── image + error map ──
        val_dir = os.path.join(self.save_dir, "val_images", f"epoch_{ep:04d}", f"batch_{batch_idx:04d}")
        os.makedirs(val_dir, exist_ok=True)

        try:
            with torch.no_grad():
                x, c   = pl_module.get_input(batch, pl_module.first_stage_key)
                t_low  = torch.full((B,), int(0.1 * pl_module.num_timesteps),
                                    device=pl_module.device, dtype=torch.long)
                noise  = torch.randn_like(x[:, :4])
                xn     = pl_module.q_sample(x[:, :4], t=t_low, noise=noise)
                xn_full= torch.cat((xn, x[:, 4:]), dim=1)
                out    = pl_module.apply_model(xn_full, t_low, c)
                pred_z = pl_module.predict_start_from_noise(xn, t=t_low, noise=out)
                pred   = pl_module.decode_first_stage(pred_z)   # [-1,1]
                gt_img = batch['GT'].to(pl_module.device)        # [-1,1]

            pred_01 = (pred.clamp(-1., 1.) + 1.0) / 2.0
            gt_01   = (gt_img.clamp(-1., 1.) + 1.0) / 2.0

            def to_uint8(t):
                return (t.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            for i in range(B):
                # GT / ref / mask_aug / mask_gt
                Image.fromarray(to_uint8(gt_01[i])).save(
                    os.path.join(val_dir, f"s{i}_1_GT.png"))
                Image.fromarray(to_uint8(
                    (batch['inpaint_image'][i].clamp(-1.,1.) + 1.0) / 2.0)).save(
                    os.path.join(val_dir, f"s{i}_2_ref.png"))
                mask_aug_rgb = aug_mask[i].clamp(0.,1.).repeat(3,1,1)
                Image.fromarray(to_uint8(mask_aug_rgb)).save(
                    os.path.join(val_dir, f"s{i}_3_mask_aug.png"))
                mask_gt_rgb = gt_mask[i].clamp(0.,1.).repeat(3,1,1)
                Image.fromarray(to_uint8(mask_gt_rgb)).save(
                    os.path.join(val_dir, f"s{i}_4_mask_gt.png"))

                Image.fromarray(to_uint8(pred_01[i])).save(
                    os.path.join(val_dir, f"s{i}_5_pred.png"))

                err = (pred_01[i] - gt_01[i]).abs()           # [3,H,W] [0,1]
                err_vis = (err * 5.0).clamp(0., 1.)
                Image.fromarray(to_uint8(err_vis)).save(
                    os.path.join(val_dir, f"s{i}_6_err_L1x5.png"))

                m = gt_mask[i].clamp(0.,1.)                   # [1,H,W]
                err_masked = (err * m).clamp(0., 1.)
                Image.fromarray(to_uint8(err_masked)).save(
                    os.path.join(val_dir, f"s{i}_7_err_masked.png"))

                l1_full   = err.mean().item()
                l1_masked = (err * m).sum().item() / (m.sum().item() + 1e-6)
                psnr_s    = calculate_psnr(pred_01[i:i+1], gt_01[i:i+1]).item()
                ssim_s    = calculate_ssim(pred_01[i:i+1], gt_01[i:i+1]).item()
                self._log(f"    sample[{i}] l1_full={l1_full:.4f}  l1_masked={l1_masked:.4f}"
                          f"  psnr={psnr_s:.2f}  ssim={ssim_s:.4f}")

            all_imgs = torch.cat([gt_01, pred_01,
                                  (pred_01 - gt_01).abs().clamp(0.,1.) * 5], dim=0)
            grid = torchvision.utils.make_grid(all_imgs, nrow=B, normalize=False)
            arr  = (grid.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(val_dir, "grid_GT_pred_err.png"))

            self._log(f"    >> val images saved -> {val_dir}")

        except Exception as e:
            self._log(f"    [VAL][WARN] image save failed: {e}")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        elapsed = time.time() - self._t_train
        self._log(f"\n{'─'*60}")
        self._log(f"[Epoch {trainer.current_epoch} END]  elapsed={elapsed:.1f}s")
        for k, v in trainer.callback_metrics.items():
            try:
                self._log(f"  {k}: {v.item():.6f}")
            except Exception:
                pass
        self._log(f"{'─'*60}")

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank != 0 or self._f is None:
            return
        self._log(f"\n[DebugCallback] training finished  "
                  f"total={time.time()-self._t_train:.1f}s")
        self._f.close()


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # try:
        # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        if "accelerator" in trainer_config:
            del trainer_config["accelerator"]
        cpu = True
    else:
        if "accelerator" not in trainer_config:
            trainer_config["accelerator"] = "cuda"
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)
    if not opt.resume:
        if opt.train_from_scratch:
            ckpt_file=torch.load(opt.pretrained_model,map_location='cpu')['state_dict']
            ckpt_file={key:value for key,value in ckpt_file.items() if not ( key[:6]=='model.')}
            model.load_state_dict(ckpt_file,strict=False)
            print("Train from scratch!")
        else:
            model.load_state_dict(torch.load(opt.pretrained_model,map_location='cpu')['state_dict'],strict=False)
            print("Load Stable Diffusion v1-4!")

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.debug,
                "id": nowname,
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
    }
    default_logger_cfg = default_logger_cfgs["testtube"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train_opacification.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "train_opacification.ImageLogger",
            "params": {
                "batch_frequency": 84,
                "max_images": 4,
                "clamp": True,
                "increase_log_steps": False,
                "monitor": "train/loss_simple",
                "disabled": True
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
        "cuda_callback": {
            "target": "train_opacification.CUDACallback"
        },
        "debug_callback": {
            "target": "train_opacification.DebugCallback",
            "params": {
                "save_dir": os.path.join(logdir, "debug"),
                "every_n_steps": 5,
            }
        },
    }

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']

    for k in callbacks_cfg:
        if callbacks_cfg[k].get("target") == "pytorch_lightning.callbacks.ModelCheckpoint":
            print(f"Overriding dirpath for callback '{k}' to: {ckptdir}")
            callbacks_cfg[k].params.dirpath = ckptdir

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    if opt.resume:
        trainer_kwargs["resume_from_checkpoint"] = opt.resume_from_checkpoint
        
        trainer_opt.resume_from_checkpoint = opt.resume_from_checkpoint
        
        print(f"\n[Resume Info] Forcing resume path: {opt.resume_from_checkpoint}")
        print(f"[Resume Info] Target max epochs: {trainer_config.max_epochs}\n")

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    # trainer.plugins = [MyCluster()]
    trainer.logdir = logdir  ###

    # data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if cpu:
        ngpu = 1
    else:
        gpus = lightning_config.trainer.gpus
        if isinstance(gpus, int):
            ngpu = 1 if gpus >= 0 else 0
        else:
            ngpu = len(str(gpus).strip(",").split(','))
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    # if 'num_nodes' in lightning_config.trainer:
    #     num_nodes = lightning_config.trainer.num_nodes
    # else:
    num_nodes = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * num_nodes * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_nodes) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, num_nodes, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")


    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)


    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb;
            pudb.set_trace()


    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # run
    if opt.train:
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)
