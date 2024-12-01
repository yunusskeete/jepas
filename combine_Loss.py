import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, primary_loss=None, temporal_weight=0.1):
        """
        Initialize TemporalConsistencyLoss with a primary loss and a temporal weight.

        Args:
            primary_loss (torch.nn.Module): The primary loss function (e.g., MSELoss, L1Loss).
            temporal_weight (float): Weight for the temporal consistency loss.
        """
        super(TemporalConsistencyLoss, self).__init__()
        self.primary_loss = primary_loss if primary_loss else nn.MSELoss()
        self.temporal_weight = temporal_weight

    def forward(self, video_pred, video_target):
        """
        Compute the combined loss.

        Args:
            video_pred (torch.Tensor): Predicted video tensor of shape [B, C, T, H, W].
            video_target (torch.Tensor): Target video tensor of shape [B, C, T, H, W].

        Returns:
            torch.Tensor: Combined loss (primary loss + temporal consistency loss).
        """
        # Primary loss (e.g., MSE or L1)
        primary_loss = self.primary_loss(video_pred, video_target)

        # Compute temporal consistency loss
        pred_diff = (
            video_pred[:, :, 1:] - video_pred[:, :, :-1]
        )  # Frame-to-frame difference (pred)
        target_diff = (
            video_target[:, :, 1:] - video_target[:, :, :-1]
        )  # Frame-to-frame difference (target)
        temporal_loss = F.l1_loss(pred_diff, target_diff)

        # Combine losses
        combined_loss = primary_loss + self.temporal_weight * temporal_loss
        return combined_loss
