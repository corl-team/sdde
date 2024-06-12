import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from torch.nn import CrossEntropyLoss

import openood.utils.comm as comm
from openood.utils import Config, get_config_default
from openood.datasets.utils import clone_dataloader

from .diversity import ADPDiversityLoss, NCLDiversityLoss
from .diversity import InputGradientDiversityLoss, GradCAMDiversityLoss
from .diversity import DiceDiversityLoss, OEDiversityLoss
from .lr_scheduler import cosine_annealing
from .logitnorm_trainer import LogitNormLoss


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


class AMPOptimizerWrapper:
    def __init__(self, optimizer, disable=False):
        self.optimizer = optimizer
        self.params = sum([group["params"] for group in optimizer.param_groups], [])
        self.disable = disable
        if not disable:
            self.scaler = torch.cuda.amp.GradScaler()

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def backward(self, loss):
        if self.disable:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()

    def clip_grad_norm(self, max_norm):
        if not self.disable:
            self.scaler.unscale_(self.optimizer)
        return torch.nn.utils.clip_grad_norm_(self.params, max_norm)

    def step(self):
        if self.disable:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()


class DEEDTrainer:
    METRIC_SMOOTHING = 0.8
    CLS_LOSSES = {
        "ce": CrossEntropyLoss,
        "logitnorm": LogitNormLoss
    }

    def __init__(self, net, train_loader, config):
        self.net = net
        try:
            self.train_loader, self.train_unlabeled_loader = train_loader
        except ValueError:
            self.train_loader, self.train_unlabeled_loader = train_loader, None
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.amp_dtype = {
            "fp16": torch.float16,
            "bfp16": torch.bfloat16,
            "none": None
        }[get_config_default(config.trainer, "amp_dtype", "none")]
        self.use_amp = self.amp_dtype is not None
        if not self.use_amp:
            # Workaround for PyTorch checks.
            self.amp_dtype = torch.bfloat16 if self.device == "cpu" else torch.float16
        if isinstance(self.net, torch.nn.parallel.DistributedDataParallel):
            self.net = self.net.module

        if not self.config.trainer.same_batch:
            self.train_loaders = [self.train_loader] + [clone_dataloader(self.train_loader)
                                                        for _ in range(self.net.num_models - 1)]

        self.model_parameters = [p for name, p in self.net.named_parameters() if not name.startswith("discriminator.")]
        self.discriminator_parameters = [p for name, p in self.net.named_parameters() if name.startswith("discriminator.")]

        self.optimizer = torch.optim.SGD(
            self.model_parameters,
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(self.train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )
        self.optimizer = AMPOptimizerWrapper(self.optimizer, disable=not self.use_amp)

        self.cls_loss = self.CLS_LOSSES[self.config.loss.cls_loss]()

        self.losses = {}
        if get_config_default(config.loss, "eta", 0.0) > 0:
            self.losses["inpgrad"] = (config.loss.eta, InputGradientDiversityLoss(mode=config.loss.mode))
        if get_config_default(config.loss, "ncl", 0.0) > 0:
            self.losses["ncl"] = (config.loss.ncl, NCLDiversityLoss())
        if get_config_default(config.loss, "adp", 0.0) > 0:
            self.losses["adp"] = (config.loss.adp, ADPDiversityLoss(alpha=config.loss.adp_alpha, beta=config.loss.adp_beta))
        if get_config_default(config.loss, "gradcam", 0.0) > 0:
            self.losses["gradcam"] = (config.loss.gradcam, GradCAMDiversityLoss(
                mode=config.loss.gradcam_mode,
                similarity=config.loss.similarity,
                activation=config.loss.activation,
                padding=config.loss.padding))
        if get_config_default(config.loss, "dice", 0.0) > 0:
            if self.net.discriminator is None:
                raise ValueError("Need discriminator for the DICE loss.")
            self.losses["dice"] = (config.loss.dice, DiceDiversityLoss(self.net.discriminator))
        if get_config_default(config.loss, "oe", 0.0) > 0:
            self.losses["oe"] = (config.loss.oe, OEDiversityLoss())

        if self.net.discriminator is not None:
            assert self.discriminator_parameters
            self.discriminator_optimizer = torch.optim.RMSprop(
                self.discriminator_parameters,
                config.optimizer.discriminator_lr,
                momentum=config.optimizer.momentum,
                weight_decay=config.optimizer.weight_decay
            )
            self.discriminator_optimizer = AMPOptimizerWrapper(self.discriminator_optimizer, disable=not self.use_amp)

    def train_epoch(self, epoch_idx):
        metrics = defaultdict(float)
        self.net.train()
        if self.config.trainer.same_batch:
            train_dataiters = [iter(self.train_loader)]
        else:
            train_dataiters = [iter(train_loader) for train_loader in self.train_loaders]

        if self.train_unlabeled_loader is not None:
            unlabeled_dataiter = iter(self.train_unlabeled_loader)

        pbar = tqdm(range(1, len(train_dataiters[0]) + 1),
                    position=0,
                    leave=True,
                    disable=not comm.is_main_process())
        for step_idx in pbar:
            if self.config.trainer.same_batch:
                batch = next(train_dataiters[0])
                data = [batch['data'].to(self.device, non_blocking=True)] * self.net.num_models
                targets = [batch['label'].to(self.device, non_blocking=True)] * self.net.num_models
            else:
                batches = [next(i) for i in train_dataiters]
                data = [b['data'].to(self.device, non_blocking=True) for b in batches]
                targets = [b['label'].to(self.device, non_blocking=True) for b in batches]

            batch_metrics = {}

            weight_ce = get_config_default(self.config.loss, "ce", 1.0)
            apply_diversity = \
                epoch_idx <= self.config.trainer.diversity_loss_first_epochs or \
                epoch_idx >= self.config.optimizer.num_epochs - self.config.trainer.diversity_loss_last_epochs or \
                step_idx % self.config.trainer.diversity_loss_period == 0
            batch_metrics["apply_diversity"] = float(bool(apply_diversity))

            if apply_diversity:
                # Either use a data batch or an unlabeled OOD batch.
                if self.train_unlabeled_loader is not None:
                    try:
                        diversity_batch = next(unlabeled_dataiter)
                    except StopIteration:
                        unlabeled_dataiter = iter(self.train_unlabeled_loader)
                        diversity_batch = next(unlabeled_dataiter)
                    same_data = diversity_batch['data'].to(self.device)
                    same_target = None
                else:
                    batch_idx = random.randrange(len(batches))
                    same_data = data[batch_idx]
                    same_target = targets[batch_idx]

            # Forward pass.
            with torch.autocast(device_type=self.device, enabled=self.use_amp, dtype=self.amp_dtype):
                # Simple forward.
                if (weight_ce > 0) or (apply_diversity and self.config.trainer.same_batch):
                    images = torch.stack(data).requires_grad_(apply_diversity)
                    _, logits_classifier, inp, features, feature_maps = self.net(images, return_ensemble=True)

                # Forward diversity data.
                if apply_diversity and self.config.trainer.same_batch and (self.train_unlabeled_loader is None):
                    # Reuse outputs if there is no need for diversity batch forwarding.
                    same_logits_classifier, same_inp, same_features, same_feature_maps = (
                        logits_classifier, inp, features, feature_maps
                    )
                elif apply_diversity:
                    same_images = (
                        same_data.unsqueeze(0)
                        .repeat_interleave(self.net.num_models, dim=0)
                        .requires_grad_(True)
                    )  # (N, B, C, H, W).
                    assert same_images.ndim == 5
                    self.net.eval()
                    same_agg_logits, same_logits_classifier, same_inp, same_features, same_feature_maps = self.net(same_images,
                                                                                                                   return_ensemble=True)
                    self.net.train()
                    if same_target is None:
                        same_target = same_agg_logits.argmax(1)  # (B).
                        assert same_target.ndim == 1

                # Compute losses.
                total_loss = 0
                discriminator_loss = 0
                if weight_ce > 0:
                    loss_ce = sum([self.cls_loss(logits_classifier[i], targets[i])
                                   for i in range(self.net.num_models)]) / self.net.num_models
                    batch_metrics["loss_ce"] = float(loss_ce)
                    loss = loss_ce * weight_ce
                    total_loss = total_loss + loss
                    if get_config_default(self.config.loss, "adversarial", 0) > 0:
                        loss_adv = self._adversarial_loss(images, targets,
                                                        loss_ce, self.config.loss.adversarial)
                        loss = loss_adv * weight_ce
                        total_loss = total_loss + loss
                        batch_metrics["loss_adv"] = float(loss_adv)

                if apply_diversity:
                    assert (same_features is None) or (same_features[0].ndim == 2)  # N x (B, D).
                    assert (same_feature_maps is None) or (same_feature_maps[0].ndim == 4)  # N x (B, C, H, W).
                    div_loss = 0
                    self.net.eval()
                    for name, (weight, loss) in self.losses.items():
                        if (name == "dice") and (self.config.loss.dice_ramp_up_epochs > 0):
                            dice_scale = min((epoch_idx - 1) / self.config.loss.dice_ramp_up_epochs, 1)
                            weight = weight * dice_scale
                            batch_metrics["dice_scale"] = dice_scale
                        loss_diversity = loss(same_inp, same_target, same_logits_classifier,
                                              same_features, same_feature_maps)
                        div_loss = div_loss + weight * loss_diversity
                        batch_metrics[f"loss_{name}"] = float(loss_diversity)
                    batch_metrics["div_loss"] = float(div_loss)
                    total_loss = total_loss + div_loss
                    batch_metrics["loss"] = float(total_loss)
                    self.net.train()
                    if self.net.discriminator is not None:
                        discriminator_loss, disc_metrics = self.net.discriminator.loss([f.detach() for f in same_features],
                                                                                    same_target)
                        batch_metrics.update(disc_metrics)
            pbar_message = f'Epoch: {epoch_idx:03d}'
            # Backward and step.
            if isinstance(total_loss, torch.Tensor):
                pbar_message = pbar_message + f', loss: {total_loss.item():.4f}'
                if (step_idx + 1) % self.config.trainer.gradient_accumulation == 0 or (step_idx + 1) == len(self.train_loader):
                    self.optimizer.zero_grad(set_to_none=True)
                total_loss = total_loss / self.config.trainer.gradient_accumulation
                self.optimizer.backward(total_loss)
                if get_config_default(self.config.optimizer, "grad_clip", 0.0) > 0:
                    self.optimizer.clip_grad_norm(self.config.optimizer.grad_clip)
                if (step_idx + 1) % self.config.trainer.gradient_accumulation == 0 or (step_idx + 1) == len(self.train_loader):
                    self.optimizer.step()
            self.scheduler.step()
            if isinstance(discriminator_loss, torch.Tensor):
                if (step_idx + 1) % self.config.trainer.gradient_accumulation == 0 or (step_idx + 1) == len(self.train_loader):
                    self.discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_loss = discriminator_loss / self.config.trainer.gradient_accumulation
                self.discriminator_optimizer.backward(discriminator_loss)
                if get_config_default(self.config.optimizer, "grad_clip", 0.0) > 0:
                    self.discriminator_optimizer.clip_grad_norm(self.config.optimizer.grad_clip)
                if (step_idx + 1) % self.config.trainer.gradient_accumulation == 0 or (step_idx + 1) == len(self.train_loader):
                    self.discriminator_optimizer.step()

            with torch.no_grad():
                for k, v in batch_metrics.items():
                    metrics[k] = metrics[k] * self.METRIC_SMOOTHING + v * (1 - self.METRIC_SMOOTHING)

            pbar.set_description(pbar_message)

        metrics = {k: self.save_metrics(v) for k, v in metrics.items()}
        metrics.update({
            'epoch_idx': epoch_idx,
            'lr': self.optimizer.lr
        })

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced

    def _adversarial_loss(self, inputs, target, ce_loss, eps):
        requires_grad = inputs.requires_grad
        inputs.requires_grad_(True)
        image_grad = torch.autograd.grad(
            ce_loss, inputs, retain_graph=True, create_graph=True
        )
        inputs_adv = inputs + eps * torch.sign(image_grad[0])
        _, logits_classifier, _, _, _ = self.net(inputs_adv, return_ensemble=True)
        loss_adv = sum([F.cross_entropy(logits_classifier[i], target[i])
                        for i in range(self.net.num_models)]) / self.net.num_models
        inputs.requires_grad_(requires_grad)
        return loss_adv
