from abc import ABC, abstractmethod
import torch


def non_diag(a):
    """Get non-diagonal elements of matrices.

    Args:
        a: Matrices tensor with shape (..., N, N).

    Returns:
        Non-diagonal elements with shape (..., N, N - 1).
    """
    n = a.shape[-1]
    prefix = list(a.shape)[:-2]
    return a.reshape(*(prefix + [n * n]))[..., :-1].reshape(*(prefix + [n - 1, n + 1]))[..., 1:].reshape(*(prefix + [n, n - 1]))


def cosine_similarity(stats, normalize=True):
    """Compute cosine similarities for a tensor with shape (N, B, *)."""
    stats = stats.flatten(2).transpose(0, 1)  # (B, N, D).
    if normalize:
        stats = stats / (torch.linalg.norm(stats, dim=2, keepdim=True) + 1e-6)  # (B, N, D).
    cosines = stats @ stats.transpose(1, 2)  # (B, N, N - 1).
    return non_diag(cosines).mean()


def exp_cosine_similarity(stats, normalize=True):
    """Similarity from https://arxiv.org/pdf/2112.03615.pdf"""
    stats = stats.flatten(2).transpose(0, 1)  # (B, N, D).
    if normalize:
        stats = stats / (torch.linalg.norm(stats, dim=2, keepdim=True) + 1e-6)  # (B, N, D).
    cosines = non_diag(stats @ stats.transpose(1, 2))  # (B, N, N - 1).
    exp_cosines = torch.exp(cosines)
    return torch.log(torch.sum(exp_cosines) / 2).mean()


def neglogdet_similarity(stats):
    """Compute log determinant similarities for a tensor with shape (N, B, *)."""
    num_models = len(stats)
    stats = stats.flatten(2).transpose(0, 1)  # (B, N, D).
    stats = stats / (torch.linalg.norm(stats, dim=2, keepdim=True) + 1e-6)  # (B, N, D).
    products = stats @ stats.transpose(1, 2)  # (B, N, N).
    diagonal = torch.eye(num_models, dtype=products.dtype, device=products.device)
    logdet = torch.logdet(products + 1e-5 * diagonal).mean()
    return -logdet


def iou_similarity(stats, eps=1e-6):
    """Compute jaccard similarities for a tensor with shape (N, B, *)."""
    # Convert to 0-1 range.
    stats = stats.flatten(2)  # (N, B, D).
    low = stats.min(dim=2, keepdim=True)[0]  # (N, B, 1).
    high = stats.max(dim=2, keepdim=True)[0]  # (N, B, 1).
    stats = (stats - low) / (high - low + eps)  # (N, B, D).
    stats1 = stats[:, None]  # (N, 1, B, D).
    stats2 = stats[None]  # (1, N, B, D).
    prod = (stats1 * stats2).sum(3)  # (N, N, B).
    sum1 = (stats1 ** 2).sum(3)  # (N, 1, B).
    sum2 = (stats2 ** 2).sum(3)  # (1, N, B).
    similarity = (prod + eps) / (sum1 + sum2 - prod + eps)  # (N, N, B).
    assert similarity.ndim == 3
    similarity = similarity.permute(2, 0, 1)  # (B, N, N).
    pairs = non_diag(similarity)  # (B, N, N - 1).
    return pairs.mean()


class DiversityLoss(ABC):
    def __call__(self, images, labels, logits, features, feature_maps):
        """Compute the loss.

        Args:
            images: Input images with shape (N, B, *).
            labels: Ground truth labels with shape (B).
            logits: Model outputs with shape (N, B, L).
            features: List of model embeddings with shapes (B, D).
            feature_maps: List of model activation maps with shapes (B, C, H, W).
        """
        pass
