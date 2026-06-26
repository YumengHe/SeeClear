import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelRefineHead(nn.Module):
    """
    : Concat [Image (3), Reference (3), Heatmap/Mask (1)] ->  7 
    : Mask Logits (1 )
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1),
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(32, 32),
            nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GroupNorm(16, 16),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )
    
    def forward(self, x_gen_pixel, x_rgb, m_cond):
        # x_gen_pixel: [B, 3, H, W] (GT Image/Generated Image)
        # x_rgb:       [B, 3, H, W] (Reference Image)
        # m_cond:      [B, 1, H, W] (Condition: Heatmap or Rough Mask)
        x = torch.cat([x_gen_pixel, x_rgb, m_cond], dim=1)
        return self.net(x)

def compute_mask_loss(mask_logits, gt_mask):
    """
    Loss: BCE + 0.1 * Mid_Penalty
    """
    if gt_mask.max() > 1.5:
        gt_mask = gt_mask.float() / 255.0

    # 1. BCE Loss
    bce_loss = F.binary_cross_entropy_with_logits(mask_logits, gt_mask, reduction='mean')
    
    # 2. Mid Penalty
    mask_soft = torch.sigmoid(mask_logits)
    mid_penalty = (mask_soft * (1 - mask_soft)).mean()
    
    # 3. Total Loss
    total_loss = bce_loss + 0.1 * mid_penalty
    
    return total_loss, bce_loss, mid_penalty