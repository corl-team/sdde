import torch
from .common import DiversityLoss, cosine_similarity, neglogdet_similarity, iou_similarity, exp_cosine_similarity


class GradCAMDiversityLoss(DiversityLoss):
    """Diversify class activation maps."""
    def __init__(self, mode="gt_logit", similarity="cosine", padding=1, activation="relu"):
        super().__init__()
        self.mode = mode
        self.padding = padding
        self.similarity = similarity
        self.activation = activation

    def _get_alt_labels(self, labels, num_classes):
        assert labels.ndim == 1
        batch_size = len(labels)
        alt_labels = torch.arange(num_classes - 1, device=labels.device)[None].tile(batch_size, 1)  # (B, L - 1).
        alt_labels[alt_labels >= labels[:, None]] += 1  # (B, L - 1).
        return alt_labels

    def _get_less_quantile(self, logits, q, eps=1e-4):
        n, b, l = logits.shape
        # Add small noise to prevent repeated values.
        with torch.no_grad():
            noisy_logits = logits + torch.randn_like(logits) * eps
            quantiles = torch.quantile(noisy_logits, q, dim=2, keepdim=True)  # (N, B, 1).
            mask = noisy_logits < quantiles
        small_logits = logits[mask].reshape(n, b, l // 2)  # (N, B, L // 2).
        return small_logits

    def _get_outputs(self, logits, labels):
        """Choose output to compute salience map."""
        if self.mode == "gt_logit":
            outputs = logits.take_along_dim(labels[None, :, None], 2).squeeze(2).sum(1)  # (N).
        elif self.mode == "gt_prob":
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (N, B, L).
            outputs = probs.take_along_dim(labels[None, :, None], 2).squeeze(2).sum(1)  # (N).
        elif self.mode == "alt_logits":
            alt_labels = self._get_alt_labels(labels, logits.shape[2])  # (B, L - 1).
            alt_logits = logits.take_along_dim(alt_labels[None], 2)  # (N, B, L - 1)
            outputs = alt_logits.mean(2).sum(1)  # (N).
        elif self.mode == "alt_probs":
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (N, B, L).
            gt_probs = probs.take_along_dim(labels[None, :, None], 2).squeeze(2)  # (N, B).
            num_classes = logits.shape[2]
            outputs = (1 - gt_probs).sum(1) / (num_classes - 1)  # (N), alt probs mean.
        elif self.mode == "small_logits":
            small_logits = self._get_less_quantile(logits, 0.5)  # (N, B, L // 2).
            outputs = small_logits.mean(2).sum(1)  # (N).
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return outputs

    def get_salience_maps(self, images, labels, logits, feature_maps):
        # logits: (N, B, L).
        outputs = self._get_outputs(logits, labels)  # (N).
        assert outputs.shape == (len(logits),)
        feature_grads = [torch.autograd.grad(output, feature_map, retain_graph=True, create_graph=True)[0]  # (B, *).
                         for output, feature_map in zip(outputs, feature_maps)]
        feature_grads = torch.stack(feature_grads)  # (N, B, C, H, W).
        assert feature_grads.ndim == 5
        weights = feature_grads.mean(dim=(3, 4))  # (N, B, C).
        features = torch.stack(feature_maps)  # (N, B, C, H, W).
        salience_maps = (features * weights[..., None, None]).sum(2)  # (N, B, H, W).
        if self.activation == "relu":
            salience_maps = torch.nn.functional.relu(salience_maps)
        elif self.activation == "abs":
            salience_maps = salience_maps.abs()
        elif self.activation != "none":
            pass
            raise ValueError(f"Unknown activation {self.activation}")
        return salience_maps

    def __call__(self, images, labels, logits, features, feature_maps):
        """Compute the loss.

        Args:
            images (unused): Input images with shape (N, B, *).
            labels: Ground truth labels with shape (B).
            logits: Model outputs with shape (N, B, L).
            features (unused): List of model embeddings with shapes (B, D).
            feature_maps: List of model activation maps with shapes (B, C, H, W).

        Returns:
            Model statistics with shape (N, B, *).
        """
        if not feature_maps:
            raise ValueError("Need feature maps for CAM computation.")
        salience_maps = self.get_salience_maps(images, labels, logits, feature_maps)  # (N, B, H, W).
        if self.padding > 0:
            salience_maps = salience_maps[:, :, self.padding:-self.padding, self.padding:-self.padding]
        means = salience_maps.mean(dim=(2, 3), keepdim=True)  # (N, B, 1, 1).
        stds = salience_maps.std(dim=(2, 3), keepdim=True, correction=0)  # (N, B, 1, 1).
        normalized = (salience_maps - means) / (stds + 1e-6)
        if self.similarity == "cosine":
            loss = cosine_similarity(normalized)
        elif self.similarity == "logdet":
            loss = neglogdet_similarity(normalized)
        elif self.similarity == "iou":
            loss = iou_similarity(normalized)
        elif self.similarity == "expcosine":
            loss = exp_cosine_similarity(normalized)
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity}")
        return loss
