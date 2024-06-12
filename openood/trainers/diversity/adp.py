import torch

from .common import DiversityLoss, neglogdet_similarity


class ADPDiversityLoss(DiversityLoss):
    """See http://proceedings.mlr.press/v97/pang19a/pang19a.pdf for more details."""
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def __call__(self, images, labels, logits, features, feature_maps):
        """Compute the loss.

        Args:
            images (unused): Input images with shape (N, B, *).
            labels: Ground truth labels with shape (B).
            logits: Model outputs with shape (N, B, L).
            features (unused): List of model embeddings with shapes (B, D).
            feature_maps (unused): List of model activation maps with shapes (B, C, H, W).
        """
        num_models, batch_size, num_classes = logits.shape  # (N, B, L).
        probs = torch.nn.functional.softmax(logits, 2)  # (N, B, L).
        alt_mask = ~torch.nn.functional.one_hot(labels, num_classes).bool()  # (B, L).
        alt = torch.masked_select(probs, alt_mask[None]).reshape(num_models, batch_size, num_classes - 1)  # (N, B, L - 1).
        neglogdet = neglogdet_similarity(alt)
        mean_prediction = probs.mean(0)  # (B, L).
        neg_entropy = (mean_prediction * (mean_prediction + 1e-20).log()).sum(-1).mean()
        return self.alpha * neg_entropy + self.beta * neglogdet
