import os.path as osp
import numpy as np
import torch
from torch import nn
from typing import Any
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from ..networks.ensemble import ReactNetEnsemble, AshNetEnsemble


def find_react_threshold(net, react_percentile, loader, per_model=False):
    activation_log = []
    with torch.no_grad():
        for batch in tqdm(loader['val'],
                          desc='ReAct: ',
                          position=0,
                          leave=True):
            data = batch['data'].cuda()
            data = data.float()
            _, _, _, features, _ = net(data, return_ensemble=True)
            feature = torch.stack(features, dim=0)
            activation_log.append(feature)

    activation_log = torch.cat(activation_log, dim=1)
    activation_log = activation_log.flatten(1).cpu().numpy()
    if not per_model:
        react_threshold = np.percentile(activation_log, react_percentile)
    else:
        react_threshold = np.percentile(activation_log, react_percentile, axis=1)
    return react_threshold


class DEEDPostprocessor(BasePostprocessor):
    """
    Logits aggregation for DEED evaluation
    """
    def __init__(self, config):
        super(DEEDPostprocessor, self).__init__(config)
        self.config = config
        self.postprocess_config = config.postprocessor
        self.postprocessor_args = self.postprocess_config.postprocessor_args
        assert self.postprocessor_args.network_name == \
            self.config.network.name,\
            'checkpoint network type and model type do not align!'
        self.checkpoint_root = self.postprocessor_args.checkpoint_root
        self.checkpoint_path = osp.join(self.checkpoint_root, 'best.ckpt')
        self.num_networks = self.postprocessor_args.num_networks
        self.aggregation = self.postprocessor_args.aggregation
        self.net, self.react_percentile, self.react_threshold, self.ash_percentile \
            = None, None, None, None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.net = net
        self.net.load_state_dict(torch.load(self.checkpoint_path), strict=True)

        self.net.eval()
        if float(self.postprocessor_args.react_percentile) > 0 and \
                float(self.postprocessor_args.ash_percentile) > 0:
            raise ValueError("Can't apply ASH and ReAct simultaneously.")
        elif float(self.postprocessor_args.react_percentile) > 0:
            self.react_percentile = float(self.postprocessor_args.react_percentile)
            self.net = ReactNetEnsemble(self.net)
            self.react_threshold = find_react_threshold(
                self.net,
                float(self.postprocessor_args.react_percentile),
                id_loader_dict,
                self.postprocessor_args.react_per_model
            )
            print('ReAct threshold at percentile {:.2f} over id data is: {}'.format(
                self.react_percentile, self.react_threshold)
            )
        elif float(self.postprocessor_args.ash_percentile) > 0:
            self.ash_percentile = float(self.postprocessor_args.ash_percentile)
            self.net = AshNetEnsemble(self.net)
            print(f'Evaluating with ASH percentile {self.ash_percentile}')

    def postprocess(self, net: nn.Module, data: Any):
        if (self.react_percentile is not None) and (self.ash_percentile is not None):
            raise ValueError("Can't mix React and ASH inference.")
        if self.react_percentile is not None:
            agg_logits, logits = self.net.forward_threshold(data, self.react_threshold)[:2]  # (B, L), (N, B, L).
        elif self.ash_percentile is not None:
            agg_logits, logits = self.net.forward_threshold(data, self.ash_percentile)[:2]  # (B, L), (N, B, L).
        else:
            agg_logits, logits = net(data, return_ensemble=True)[:2]  # (B, L), (N, B, L).

        agg_logits = agg_logits.detach().cpu()
        logits = logits.detach().cpu()
        probs = torch.softmax(agg_logits, dim=1)
        pred_probs, pred = torch.max(probs, dim=1)
        if self.aggregation == 'average':
            conf = pred_probs
        elif self.aggregation == 'prob-average':
            mean_probs = torch.softmax(logits, dim=2).mean(0)  # (B, L).
            conf, _ = torch.max(mean_probs, dim=1)
        elif self.aggregation == 'prob-minmax':
            probs_max = torch.max(torch.softmax(logits, dim=2), dim=2)[0]  # (N, B).
            conf = torch.min(probs_max, dim=0)[0]
        elif self.aggregation == 'prob-maxstd':
            probs_max = torch.max(torch.softmax(logits, dim=2), dim=2)[0]  # (N, B).
            conf = - torch.std(probs_max, dim=0)
        elif self.aggregation == 'logit-average':
            conf, _ = torch.max(agg_logits, dim=1)
        elif self.aggregation == 'minmax':
            logits_max = torch.max(logits, dim=2)[0]
            conf = torch.min(logits_max, dim=0)[0]
        elif self.aggregation == 'stdmax':
            logits_std = torch.std(logits, dim=0)
            conf = - torch.max(logits_std, dim=1)[0]
        elif self.aggregation == 'maxstd':
            logits_max = torch.max(logits, dim=2)[0]
            conf = - torch.std(logits_max, dim=0)
        elif self.aggregation == 'average-minmax':
            selected_classes = torch.max(agg_logits, dim=1)[1]
            selected_logits = logits.take_along_dim(selected_classes[None, :, None], 2).squeeze()
            conf = torch.min(selected_logits, dim=0)[0]
        elif self.aggregation == "energy":
            conf = torch.logsumexp(agg_logits, dim=1)
        elif self.aggregation == "average-energy":
            energy = torch.logsumexp(logits, dim=2)
            conf = torch.mean(energy, dim=0)
        else:
            raise ValueError(f'Unknown aggregation type {self.aggregation}.')

        return pred, conf
