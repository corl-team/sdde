import math
from unittest import TestCase, main

import torch

from openood.utils import Config, seed_everything
from openood.networks.discriminator import Discriminator
from openood.datasets import make_dataloader
from openood.datasets.sampler import ShuffledClassBalancedBatchSampler
from openood.trainers.deed_trainer import DEEDTrainer
from openood.trainers.diversity.common import non_diag


class DotNetEnsemble(torch.nn.Module):
    """A simple model.

    Predictions are equal to label + b. The random variable b is
    common for all models.

    The common_bias_std parameter reflects the degree of models dependency.

    """
    def __init__(self, input_size=(3, 2, 2), num_models=4, num_classes=5):
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        self.scale = torch.nn.Parameter(torch.ones([]))
        self.bias = torch.nn.Parameter(torch.ones([]))
        self.common_bias_std = torch.nn.Parameter(torch.ones([]))
        self.discriminator = Discriminator(num_classes, num_classes, units=(16, 16), label_dim=4)

    def forward(self, x, return_ensemble=False):
        # x: NBCHW.
        n, b, c, h, w = x.shape
        labels = x[:, :, 0, 0, 0].long()  # (N, B).
        logits = torch.nn.functional.one_hot(labels, self.num_classes).float()  # (N, B, C).
        common_bias =  torch.randn([b]) * self.common_bias_std.abs()  # (B).
        features = logits + self.bias + 0.1 * torch.randn_like(logits) + common_bias[None, :, None]  # (N, B, C).
        logits = features * self.scale
        agg_logits = logits.mean(0)
        if return_ensemble:
            return agg_logits, logits, None, list(features), None  # (N, B, C), None, N x (B, 1), None.
        else:
            return agg_logits


class WhiteDataset(torch.utils.data.Dataset):
    """Images with white pixels and different labels."""
    def __init__(self, image_size=2, size=1000, num_classes=5):
        super().__init__()
        self.image_size = image_size
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, index, only_label=False):
        assert index < self.size
        label = index % self.num_classes
        sample = {
            "label": label
        }
        if not only_label:
            sample["data"] = torch.full((3, self.image_size, self.image_size), float(label))
        return sample


class TestDiceLoss(TestCase):
    def _forward(self, config_patch):
        config = {
            "optimizer.lr": 0.01,
            "optimizer.discriminator_lr": 0.001,
            "optimizer.momentum": 0.9,
            "optimizer.weight_decay": 0.0,
            "optimizer.num_epochs": 10000,  # Large amount of epochs disables scheduler.
            "loss.cls_loss": "ce",
            "loss.ce": 0,  # Default. Can be overwritten with patch.
            "loss.eta": 0,  # Default. Can be overwritten with patch.
            "loss.mode": "sum",
            "trainer.same_batch": False,
            "trainer.diversity_loss_first_epochs": 100,
            "trainer.gradient_accumulation": 1
        }
        config = Config(config | config_patch)

        net = DotNetEnsemble()
        dataset = WhiteDataset()
        batch_sampler = ShuffledClassBalancedBatchSampler(dataset,
                                                          batch_size=8,
                                                          samples_per_class=2)
        loader = make_dataloader(dataset, batch_sampler=batch_sampler)  # Single batch.
        trainer = DEEDTrainer(net, loader, config=config)
        for i in range(3):
            _, metrics = trainer.train_epoch(i)
        return net, trainer, metrics

    def test_dice(self):
        seed_everything(0)
        net, _, metrics = self._forward({"loss.dice": 0.1, "loss.dice_ramp_up_epochs": 0})
        self.assertLess(net.common_bias_std, 0.1)


if __name__ == "__main__":
    main()
