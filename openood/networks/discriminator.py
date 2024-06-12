import torch


def group_by_class(embeddings, labels):
    """Group embeddings into batch by label.

    Args:
        embeddings: Input tensor with shape (B, P).
        labels: Batch labels with shape (B). Labels must be balanced.

    Returns:
        A tuple of
            - grouped_embeddings with shape (B // L, L, P), where second dimension encodes label.
            - original indices with shape (B // L, L).
            - label_map with shape (L) which stores original label indices.
    """
    if embeddings.ndim != 2:
        raise ValueError("Expected tensor with shape (B, P).")
    counts = torch.bincount(labels)
    counts = counts[counts > 0]
    if (counts != counts[0]).any():
        raise ValueError("Need uniform balanced sampling: {}.".format(counts))
    unique_labels = torch.unique(labels)
    indices = torch.stack([torch.nonzero(labels == label).squeeze(-1) for label in unique_labels], dim=1)  # (B // L, L).
    by_class = torch.stack([embeddings[labels == label] for label in unique_labels], dim=1)  # (B // L, L, P).
    assert by_class.ndim == 3
    return by_class, indices, unique_labels


class Discriminator(torch.nn.Module):
    """Discriminator for the DICE model.

    Input contains two embeddings and condition label.
    Output is the logarithm of the true class probability
    """
    def __init__(self, num_classes, embedding_dim, units=(256, 256, 100), label_dim=64):
        super().__init__()
        self.label_embedder = torch.nn.Embedding(num_classes, label_dim)
        input_dim = embedding_dim * 2 + label_dim
        layers = []
        for n in units:
            layers.extend([
                torch.nn.Linear(input_dim, n),
                torch.nn.LeakyReLU(0.2, inplace=True)
            ])
            input_dim = n
        layers.append(torch.nn.Linear(input_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, embeddings1, embeddings2, labels):
        # Embeddings has the shape (B, D).
        label_embeddings = self.label_embedder(labels)
        inputs = torch.cat([embeddings1, embeddings2, label_embeddings], -1)  # (B, D').
        return self.net(inputs).squeeze(1)  # (B).

    def pair_loss(self, embeddings1, embeddings2, labels):
        by_class1, indices, label_map = group_by_class(embeddings1, labels)  # (B, L, D), (B, L), (L).
        by_class2, _, _ = group_by_class(embeddings2, labels)  # (B, L, D).
        b, l, d = by_class1.shape
        grouped_labels = label_map[None].tile(b, 1)  # (B, L).
        joint_logits = self(by_class1.reshape(b * l, d),
                            by_class2.reshape(b * l, d),
                            grouped_labels.reshape(b * l))  # (B * L).
        shifted2 = by_class2.roll(1, 0)  # (B, L, D).
        product_logits = self(by_class1.reshape(b * l, d),
                              shifted2.reshape(b * l, d),
                              grouped_labels.reshape(b * l))  # (B * L).
        loss = -0.5 * (torch.nn.functional.logsigmoid(joint_logits).mean() +
                       torch.nn.functional.logsigmoid(-product_logits).mean())
        with torch.no_grad():
            accuracy = 0.5 * ((torch.sigmoid(joint_logits) > 0.5).float().mean() + (torch.sigmoid(product_logits) < 0.5).float().mean())
        return loss, accuracy

    def loss(self, embeddings, labels):
        """Compute loss for training discriminator to classify samples from the joint distribution."""
        if not embeddings:
            raise ValueError("Need embeddings for DICE diversity loss computation.")
        num_models = len(embeddings)
        losses, accuracies = zip(*[self.pair_loss(embeddings[i], embeddings[j], labels)
                                   for i in range(num_models) for j in range(i + 1, num_models)])
        loss = sum(losses) / len(losses)
        accuracy = sum(accuracies) / len(accuracies)
        metrics = {
            "discriminator_accuracy": float(accuracy),
            "discriminator_ce": float(loss)
        }
        return loss, metrics
