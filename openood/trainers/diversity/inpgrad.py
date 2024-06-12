import torch
from .common import DiversityLoss, cosine_similarity


class InputGradientDiversityLoss(DiversityLoss):
    """Diversify salience maps."""
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def get_salience_maps(self, images, labels, logits):
        num_models, batch_size, num_classes = logits.shape  # (N, B, L).

        if self.mode == "max":
            output = logits.max(dim=2)[0]  # (N, B).
        elif self.mode == "min":
            output = logits.min(dim=2)[0]  # (N, B).
        elif self.mode == "gt":
            output = logits.take_along_dim(labels[None, :, None], 2).squeeze(2)  # (N, B).
        elif self.mode == "sum":
            output = logits.sum(2)  # (N, B).
        else:
            raise ValueError(f"Unknown diversity loss mode {self.config.mode}.")
        assert output.ndim == 2

        image_grad = torch.autograd.grad(
            output.sum(), images, retain_graph=True, create_graph=True
        )[0]  # (N, B, *).
        return image_grad

    def __call__(self, images, labels, logits, features, feature_maps):
        """Compute the loss.

        Args:
            images: Input images with shape (N, B, *).
            labels: Ground truth labels with shape (B).
            logits: Model outputs with shape (N, B, L).
            features (unused): List of model embeddings with shapes (B, D).
            feature_maps (unused): List of model activation maps with shapes (B, C, H, W).
        """
        image_grad = self.get_salience_maps(images, labels, logits)
        return cosine_similarity(image_grad)
