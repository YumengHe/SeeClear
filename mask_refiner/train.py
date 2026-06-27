import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(current_dir)

from dataset import MyTransparentMaskHeadDataset
from model import PixelRefineHead, compute_mask_loss


def build_loader(state, dataset_base_cfg, batch_size, shuffle):
    dataset = MyTransparentMaskHeadDataset(state=state, **dataset_base_cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )
    return dataset, loader


@torch.no_grad()
def evaluate(model, loader, device, split_name, max_batches=0):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        x_gen_pixel = batch['x_gen_pixel'].to(device)
        x_rgb = batch['x_rgb'].to(device)
        m_cond = batch['m_cond'].to(device)
        m_gt = batch['M_gt'].to(device)

        logits = model(x_gen_pixel, x_rgb, m_cond)
        loss, _, _ = compute_mask_loss(logits, m_gt)
        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    print(f"{split_name} Avg Loss: {avg_loss:.5f}")
    return avg_loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    dataset_base_cfg = {
        'dataset_dir': args.data_dir,
        'img_size': args.img_size,
        'mask_strategy': args.strategy,
        'heatmap_sigma': 10,
        'bbox_wh_range': args.bbox_wh_range,
        'bbox_center_radius': args.bbox_center_radius,
    }
    
    train_dataset, train_loader = build_loader('train', dataset_base_cfg, args.batch_size, True)
    val_dataset, val_loader = build_loader('val', dataset_base_cfg, args.batch_size, False)
    
    print(
        f"Start Training: Strategy=S{args.strategy}, "
        f"Train={len(train_dataset)}, Val={len(val_dataset)}, "
        f"Save to: {args.save_dir}"
    )

    model = PixelRefineHead().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        resume_path = os.path.join(args.save_dir, "last.pth")
        if os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_loss = checkpoint.get('best_loss', float('inf'))
                print(f"Resumed from epoch {start_epoch}, best_loss: {best_loss:.5f}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model weights, starting from epoch 0")
        else:
            print(f"No checkpoint found at {resume_path}, starting from scratch")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            x_gen_pixel = batch['x_gen_pixel'].to(device)
            x_rgb = batch['x_rgb'].to(device)
            m_cond = batch['m_cond'].to(device)
            m_gt = batch['M_gt'].to(device)
            
            optimizer.zero_grad()
            logits = model(x_gen_pixel, x_rgb, m_cond)
            loss, bce, mid = compute_mask_loss(logits, m_gt)
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'BCE': f'{bce.item():.4f}'})
        
        train_batches = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
        avg_loss = total_epoch_loss / max(train_batches, 1)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.5f}")
        val_loss = evaluate(model, val_loader, device, "Val", args.max_eval_batches)
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'train_loss': avg_loss,
            'val_loss': val_loss,
        }
        torch.save(checkpoint, os.path.join(args.save_dir, "last.pth"))
        
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))
            print(f"*** Best Model Saved (Val Loss: {best_loss:.5f}) ***")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset/my_data")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--strategy", type=int, required=True, choices=[1,2,3,4,5,6,7,8,9,10,11,12], 
                        help="1-4: Point strategies, 5-8: Mask augmentation schemes 1-4, 9-12: BBox modes 1-4")
    parser.add_argument("--bbox_wh_range", type=float, default=0.3, help="BBox w/h random range for strategy 10,12")
    parser.add_argument("--bbox_center_radius", type=float, default=0.1, help="BBox center offset ratio for strategy 11,12")
    parser.add_argument("--resume", action='store_true', help="Resume from last.pth if exists")
    parser.add_argument("--max_train_batches", type=int, default=0, help="Limit train batches for smoke tests")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="Limit val/test batches for smoke tests")
    
    args = parser.parse_args()
    train(args)
