import torch

from .common import DiversityLoss


class OEDiversityLoss(DiversityLoss):
    """See "DEEP ANOMALY DETECTION WITH OUTLIER EXPOSURE" for more details."""
    def __call__(self, images, labels, logits, features, feature_maps):
        """Compute the loss.

        Args:
            images (unused): Input images with shape (N, B, *).
            labels (unused): Ground truth labels with shape (B).
            logits: Model outputs with shape (N, B, L).
            features (unused): List of model embeddings with shapes (B, D).
            feature_maps (unused): List of model activation maps with shapes (B, C, H, W).

        Returns:
            Model statistics with shape (N, B, *).
        """
        num_models, batch_size, num_classes = logits.shape  # (N, B, L).
        return -(logits.mean(2) - torch.logsumexp(logits, dim=2)).mean()
