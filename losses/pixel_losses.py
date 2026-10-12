# losses/pixel_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    """
    Standard Cross-Entropy Loss for segmentation
    Supports ignore_index
    """
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,C,H,W]
        target: [B,H,W]
        """
        return self.ce(logits, target)


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for segmentation
    Supports multi-class segmentation
    """
    def __init__(self, smooth: float = 1e-6, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
    
        valid_mask = target != self.ignore_index
    
        # Replace ignore pixels with 0 temporarily (safe for one_hot)
        target_clean = target.clone()
        target_clean[~valid_mask] = 0
    
        target_one_hot = F.one_hot(target_clean, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
    
        mask = valid_mask.unsqueeze(1).float()
    
        probs = probs * mask
        target_one_hot = target_one_hot * mask
    
        intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


if __name__ == "__main__":
    # Quick test
    B, C, H, W = 2, 2, 64, 64
    logits = torch.rand(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))

    ce_loss = CELoss()
    dice_loss = DiceLoss()

    print("CE Loss:", ce_loss(logits, target).item())
    print("Dice Loss:", dice_loss(logits, target).item())
