import torch
from .common import DiversityLoss


class DiceDiversityLoss(DiversityLoss):
    """Apply DICE diversification.

    Read https://openreview.net/pdf?id=R2ZlTVPx0Gk for more details.
    """
    def __init__(self, discriminator, tau=10.0):
        super().__init__()
        self.discriminator = discriminator
        self.tau = tau

    def pair_diversity(self, features1, features2, labels):
        log_probs = torch.nn.functional.logsigmoid(self.discriminator(features1, features2, labels))  # (B), joint distribution.
        log_ratio = -((-log_probs).exp() - 1 + 1e-6).log()
        log_f = self.tau * (log_ratio / self.tau).tanh()  # Tanh clipping for stability.
        return log_f.mean()

    def __call__(self, images, labels, logits, features, feature_maps):
        """Compute the loss.

        Args:
            images (unused): Input images with shape (N, B, *).
            labels: Ground truth labels with shape (B).
            logits (unused): Model outputs with shape (N, B, L).
            features: List of model embeddings with shapes (B, D).
            feature_maps (unused): List of model activation maps with shapes (B, C, H, W).
        """
        if not features:
            raise ValueError("Need embeddings for DICE diversity loss computation.")
        num_models = len(features)
        losses = [self.pair_diversity(features[i], features[j], labels)
                  for i in range(num_models) for j in range(i + 1, num_models)]
        return sum(losses) / len(losses)
