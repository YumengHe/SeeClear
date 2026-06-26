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
    
    train_dataset = MyTransparentMaskHeadDataset(
        state='train',
        **dataset_base_cfg
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Start Training: Strategy=S{args.strategy}, Images={len(train_dataset)}, Save to: {args.save_dir}")

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
        
        for batch in pbar:
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
        
        avg_loss = total_epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.5f}")
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        torch.save(checkpoint, os.path.join(args.save_dir, "last.pth"))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))
            print(f"*** Best Model Saved (Loss: {best_loss:.5f}) ***")

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
    
    args = parser.parse_args()
    train(args)
