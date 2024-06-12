import torch
from .resnet50 import ResNet50


class ResNet50_Ensemble(torch.nn.Module):
    def __init__(self, num_classes=10, num_models=5, discriminator_cls=None):
        super(ResNet50_Ensemble, self).__init__()
        self.nets = torch.nn.ModuleList([ResNet50(num_classes=num_classes) for _ in range(num_models)])
        self.num_models = num_models
        self.num_classes = num_classes
        self.discriminator = discriminator_cls(num_classes, 512) if discriminator_cls is not None else None

    def forward(self, x):
        # x - N x B x C x H x W
        outs = []
        features = []
        feature_maps = []
        for i in range(len(self.nets)):
            logits, feature, feature_map = self.nets[i](x[i], return_feature_map=True)
            outs.append(logits)
            features.append(feature)
            feature_maps.append(feature_map)
            assert feature.ndim == 2, feature.shape  # (B, D).
            assert feature_map.ndim == 4, feature_map.shape  # (B, C, H, W).
        outs = torch.stack(outs)  # (N, B, L).
        return outs, None, features, feature_maps
