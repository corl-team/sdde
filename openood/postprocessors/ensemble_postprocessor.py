import os.path as osp
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from .base_postprocessor import BasePostprocessor


class EnsemblePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(EnsemblePostprocessor, self).__init__(config)
        self.config = config
        self.postprocess_config = config.postprocessor
        self.postprocessor_args = self.postprocess_config.postprocessor_args
        assert self.postprocessor_args.network_name == \
            self.config.network.name,\
            'checkpoint network type and model type do not align!'
        # get ensemble args
        self.checkpoint_root = self.postprocessor_args.checkpoint_root
        self.aggregation = self.postprocessor_args.aggregation

        # list of trained network checkpoints
        self.checkpoints = self.postprocessor_args.checkpoints
        # number of networks to esembel
        self.num_networks = self.postprocessor_args.num_networks
        # get networks
        self.checkpoint_dirs = [
            osp.join(self.checkpoint_root, path, 'best.ckpt')
            for path in self.checkpoints
        ]

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.networks = [deepcopy(net) for i in range(self.num_networks)]
        for i in range(self.num_networks):
            self.networks[i].load_state_dict(torch.load(
                self.checkpoint_dirs[i]),
                                             strict=False)
            self.networks[i].eval()

    def probabilities(self, net: nn.Module, data: Any):
        logits_list = [
            self.networks[i](data) for i in range(self.num_networks)
        ]
        logits = torch.stack(logits_list)
        probs = torch.nn.functional.softmax(logits, dim=-1)  # (N, B, L).
        return probs.mean(0)  # (B, L).

    def postprocess(self, net: nn.Module, data: Any):
        logits_list = [
            self.networks[i](data) for i in range(self.num_networks)
        ]
        logits = torch.stack(logits_list)
        logits_mean = sum(logits) / self.num_networks

        score = torch.softmax(logits_mean, dim=1)
        if self.aggregation == 'average':
            conf, pred = torch.max(score, dim=1)
        elif self.aggregation == 'minmax':
            logits_max = torch.max(logits, dim=2)[0]
            conf = torch.min(logits_max, dim=0)[0]
            pred = torch.max(score, dim=1)[1]
        elif self.aggregation == 'average-minmax':
            selected_classes = torch.max(logits_mean, dim=1)[1]
            selected_logits = logits.take_along_dim(selected_classes[None, :, None], 2).squeeze()
            conf = torch.min(selected_logits, dim=0)[0]
            pred = torch.max(score, dim=1)[1]
        else:
            raise ValueError(f'Unknown aggregation type {self.aggregation}.')

        return pred, conf
