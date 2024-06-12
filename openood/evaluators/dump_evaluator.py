import copy
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator


class DumpEvaluator(BaseEvaluator):
    """Save predictions to npy."""
    def eval_ood(self, net: nn.Module, id_data_loader: DataLoader,
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor):
        # load training in-distribution data
        for split in ['nearood', 'farood']:
            for dataset_name, ood_dl in ood_data_loaders[split].items():
                print(f'Performing inference on {dataset_name} dataset...', flush=True)
                name = split + '-' + dataset_name
                logits, labels = self._predict(net, ood_dl)
                self._save(logits, labels, name)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        logits, labels = self._predict(net, data_loader)
        print('Performing inference on test dataset...', flush=True)
        self._save(logits, labels, 'test')
        return {"acc": -1}

    def _predict(self, net, data_loader):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()

        device = next(iter(net.parameters())).device
        logits = []
        labels = []
        with torch.no_grad():
            for batch in data_loader:
                data = batch['data'].to(device)
                try:
                    logit = net(data, return_ensemble=True)[1]  # (N, B, C).
                except TypeError:
                    logit = net(data)  # (B, C).
                logits.append(logit.cpu())
                labels.append(batch['label'].cpu())
        assert logits, "Empty dataset"
        if logits[0].ndim == 3:
            logits = torch.cat(logits, dim=1)  # (N, B, C).
        else:
            logits = torch.cat(logits)  # (B, C).
        labels = torch.cat(labels)
        return logits, labels

    def _save(self, logits, labels, save_name):
        save_dir = os.path.join(self.config.output_dir, 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 logits=logits,
                 labels=labels)
