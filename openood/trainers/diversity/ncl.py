import torch

from .common import DiversityLoss, cosine_similarity


class NCLDiversityLoss(DiversityLoss):
    """See https://arxiv.org/pdf/1802.07881.pdf for more details."""

    def __call__(self, images, labels, logits, features, feature_maps):
        """Compute the loss.

        Args:
            images (unused): Input images with shape (N, B, *).
            labels (unused): Ground truth labels with shape (B).
            logits: Model outputs with shape (N, B, L).
            features (unused): List of model embeddings with shapes (B, D).
            feature_maps (unused): List of model activation maps with shapes (B, C, H, W).
        """
        probs = torch.nn.functional.softmax(logits, 2)  # (N, B, L).
        means = probs.mean(0, keepdim=True)  # (1, B, L).
        means = means.detach()  # Follow the original paper.
        deltas = probs - means  # (N, B, L).
        return cosine_similarity(deltas, normalize=False)
