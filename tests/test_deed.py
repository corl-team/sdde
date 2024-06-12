import math
from unittest import TestCase, main

import torch

from openood.utils.config import Config
from openood.networks.discriminator import Discriminator
from openood.trainers.deed_trainer import DEEDTrainer
from openood.trainers.diversity.common import non_diag
from openood.trainers.diversity.gradcam import GradCAMDiversityLoss
from openood.datasets import make_dataloader


class DotNetEnsemble(torch.nn.Module):
    """A simple model with linear projection and scaling.

    Gradient of the output w.r.t. input is equal to net.scale * net.weight.
    Gradient of the output w.r.t. features is equal to net.scale.
    """
    def __init__(self, input_size=(3, 7, 7), num_models=4, num_classes=5, use_discriminator=False):
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        self.weight = torch.nn.Parameter(torch.randn(num_models, *input_size))
        self.scale = torch.nn.Parameter(torch.randn(1))
        self.discriminator = Discriminator(num_classes, 1)

    def forward(self, x, return_ensemble=False):
        # x: NBCHW.
        feature_maps = list(x * self.weight[:, None, :])  # N * (B, C, H, W).
        feature = [(feature_map * self.scale).sum(dim=(1, 2, 3)).unsqueeze(1) for feature_map in feature_maps]  # N * (B, 1).
        logits = torch.stack(feature)  # (N, B, 1).
        logits = logits.repeat(1, 1, self.num_classes)  # (N, B, C).
        agg_logits = logits.mean(0)
        if return_ensemble:
            return agg_logits, logits, x, feature, feature_maps  # (N, B), None, N x (B, C, H, W).
        else:
            return agg_logits


class WhiteDataset(torch.utils.data.Dataset):
    """Images with white pixels and different labels."""
    def __init__(self, image_size=7, size=10, num_classes=5):
        super().__init__()
        self.image_size = image_size
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return {
            "data": torch.ones((3, self.image_size, self.image_size)),
            "label": index % self.num_classes
        }


class TestDiversityLoss(TestCase):
    def _forward(self, config_patch):
        net = DotNetEnsemble()
        dataset = WhiteDataset()
        loader = make_dataloader(dataset, batch_size=len(dataset))  # Single batch.
        batch = next(iter(loader))

        config = {
            "optimizer.lr": 1e-10,
            "optimizer.discriminator_lr": 1e-10,
            "optimizer.momentum": 0.9,
            "optimizer.weight_decay": 0.0001,
            "optimizer.num_epochs": 10,
            "loss.cls_loss": "ce",
            "loss.eta": 0,  # Default. Can be overwritten with patch.
            "loss.mode": "sum",
            "loss.similarity": "cosine",
            "loss.activation": "relu",
            "trainer.same_batch": False,
            "trainer.diversity_loss_first_epochs": 10,
            "trainer.gradient_accumulation": 1
        }
        trainer = DEEDTrainer(net, loader, config=Config(config | config_patch))
        _, metrics = trainer.train_epoch(0)
        metrics = {k: v / (1 - trainer.METRIC_SMOOTHING) for k, v in metrics.items()}

        if "loss_ce" in metrics:
            with torch.no_grad():
                prediction = net(batch["data"][None].repeat(net.num_models, 1, 1, 1, 1), return_ensemble=True)[1]
                gt_ce = torch.stack([torch.nn.functional.cross_entropy(p, batch["label"]) for p in prediction]).mean()
            self.assertAlmostEqual((gt_ce - metrics["loss_ce"]).item(), 0)

        return net, trainer, metrics

    def test_grads(self):
        net, _, metrics = self._forward({"loss.ce": 1})
        # CE gradients in our model are always zero.
        for name, p in net.named_parameters():
            if name.startswith("discriminator."):
                continue
            self.assertEqual(torch.count_nonzero(p.grad), 0)

    def test_gradcam(self):
        for padding in [0, 1, 2]:
            net, _, metrics = self._forward({"loss.gradcam_mode": "gt_logit", "loss.gradcam": 0.1, "loss.padding": padding})
            for p in net.parameters():
                self.assertGreater(torch.count_nonzero(p.grad), 0)
            maps = torch.nn.functional.relu((net.weight * net.scale).sum(1))  # (N, H, W).
            if padding > 0:
                maps = maps[:, padding:-padding, padding:-padding]
            maps = maps.flatten(1).detach()  # (N, D).
            maps = (maps - maps.mean(1, keepdim=True)) / (maps.std(dim=1, correction=0, keepdim=True) + 1e-20)
            maps /= (torch.linalg.norm(maps, dim=1, keepdim=True) + 1e-20)
            covs = maps @ maps.T  # (N, N).
            paircovs = non_diag(covs)  # (N, N - 1).
            gt_diversity = paircovs.mean()
            self.assertLess(abs(gt_diversity - metrics["loss_gradcam"]), 1e-6)

    def test_gradcam_modes(self):
        labels = torch.tensor([3, 0, 2])  # (B).

        # Test logits.
        logits1 = torch.tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]).float()  # (3, 4).
        logits2 = torch.tensor([
            [0, -1, 2, -3],
            [4, -5, 6, -7],
            [8, -9, 10, -11]
        ]).float()  # (3, 4).
        logits = torch.stack((logits1, logits2))  # (2, 3, 4) = (N, B, C).

        loss = GradCAMDiversityLoss()  # gt_logit.
        outputs_gt = torch.tensor([
            [3, 4, 10],
            [-3, 4, 10]
        ]).float().sum(1)  # (N, B).
        outputs = loss._get_outputs(logits, labels)
        self.assertLess((outputs - outputs_gt).abs().max(), 1e-6)

        loss = GradCAMDiversityLoss(mode="alt_logits")
        outputs_gt = torch.tensor([
            [[0, 1, 2], [5, 6, 7], [8, 9, 11]],
            [[0, -1, 2], [-5, 6, -7], [8, -9, -11]]
        ]).float().mean(2).sum(1)  # (N, B, C - 1).
        outputs = loss._get_outputs(logits, labels)
        self.assertLess((outputs - outputs_gt).abs().max(), 1e-6)

        loss = GradCAMDiversityLoss(mode="small_logits")
        outputs_gt = torch.tensor([
            [[0, 1], [4, 5], [8, 9]],
            [[-1, -3], [-5, -7], [-9, -11]]
        ]).float().mean(2).sum(1)  # (N, B, C - 1).
        outputs = loss._get_outputs(logits, labels)
        self.assertLess((outputs - outputs_gt).abs().max(), 1e-6)

        # Test probs.
        probs1 = torch.tensor([
            [0.1, 0.4, 0.2, 0.3],
            [0.4, 0.2, 0.2, 0.2],
            [0.0, 0.0, 1.0, 0.0]
        ]).float()  # (3, 4).
        probs2 = torch.tensor([
            [0.1, 0.2, 0.1, 0.6],
            [0.8, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.8, 0.2]
        ]).float()  # (3, 4).
        logits = torch.stack((probs1, probs2)).log()  # (2, 3, 4) = (N, B, C).

        loss = GradCAMDiversityLoss(mode="gt_prob")
        outputs_gt = torch.tensor([
            [0.3, 0.4, 1.0],
            [0.6, 0.8, 0.8]
        ]).float().sum(1)  # (N, B).
        outputs = loss._get_outputs(logits, labels)
        self.assertLess((outputs - outputs_gt).abs().max(), 1e-6)

        loss = GradCAMDiversityLoss(mode="alt_probs")
        outputs_gt = torch.tensor([
            [[0.1, 0.4, 0.2], [0.2, 0.2, 0.2], [0, 0, 0]],
            [[0.1, 0.2, 0.1], [0, 0, 0.2], [0, 0, 0.2]]
        ]).float().mean(2).sum(1)  # (N, B, C - 1).
        outputs = loss._get_outputs(logits, labels)
        self.assertLess((outputs - outputs_gt).abs().max(), 1e-6)

    def test_oe(self):
        net, _, metrics = self._forward({"loss.oe": 0.5})
        logits = (net.weight.flatten(1).sum(1) * net.scale).unsqueeze(1).tile(1, net.num_classes)  # (N, C).
        probs = torch.nn.functional.softmax(logits, 1)
        gt_diversity = (-probs.log().mean(1)).mean()
        self.assertLess(abs(gt_diversity - metrics["loss_oe"]), 1e-6)

    def test_inpgrad(self):
        net, _, metrics = self._forward({"loss.eta": 0.1})
        for p in net.parameters():
            self.assertGreater(torch.count_nonzero(p.grad), 0)
        grads = (net.weight.flatten(1) * net.scale).detach()  # (N, 3).
        grads /= torch.linalg.norm(grads, dim=1, keepdim=True)
        covs = grads @ grads.T  # (N, N).
        paircovs = non_diag(covs)  # (N, N - 1).
        gt_diversity = paircovs.mean()
        loss_diversity = metrics["loss_inpgrad"]
        self.assertLess(abs(gt_diversity - loss_diversity), 1e-6)

    def test_ncl(self):
        net, trainer, metrics = self._forward({"loss.ncl": 1})
        probs = torch.tensor([
            [0.2, 0.8],
            [0.6, 0.4],
        ]).unsqueeze(1)  # (N, B, L).
        labels = torch.tensor([
            1
        ]).reshape(1)  # (B).
        # Means: [0.4, 0.6].
        # Deltas: [
        #  [ -0.2,  0.2 ]
        #  [  0.2, -0.2 ]
        # ]
        # cov: -0.04 - 0.04 = -0.08
        ncl_gt = -0.08
        ncl = trainer.losses["ncl"][1](None, labels, probs.log(), None, None).item()
        self.assertAlmostEqual(ncl, ncl_gt)

    def test_adp(self):
        net, trainer, metrics = self._forward({"loss.adp": 1, "loss.adp_alpha": 0.2, "loss.adp_beta": 0.5})
        probs = torch.tensor([
            [0.2, 0.4, 0.4],
            [0.6, 0.1, 0.3],
        ])  # (N, L).
        labels = torch.tensor([
            2
        ]).reshape(1)  # (B).
        # Mean prediction: [0.4, 0.25, 0.35]
        entropy_gt = - 0.4 * math.log(0.4 + 1e-20) - 0.25 * math.log(0.25 + 1e-20) - 0.35 * math.log(0.35 + 1e-20)
        alt = probs[:, :2]
        normalized = alt / (torch.linalg.norm(alt, dim=1, keepdim=True) + 1e-6)  # (N, L - 1).
        log_det = torch.logdet(normalized @ normalized.T + torch.eye(2) * 1e-5).item()
        adp_gt = -0.2 * entropy_gt - 0.5 * log_det
        adp = trainer.losses["adp"][1](None, labels, probs.log().unsqueeze(1), None, None).item()
        self.assertAlmostEqual(adp, adp_gt)


if __name__ == "__main__":
    main()
