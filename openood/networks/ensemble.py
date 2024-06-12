import math
import torch
from .lenet import LeNet
from .resnet18_32x32 import ResNet18_32x32
from .resnet18_64x64 import ResNet18_64x64
from .resnet18_224x224 import ResNet18_224x224
from .resnet50 import ResNet50
from .react_net import ReactNet
from .ash_net import ASHNet


class BaseEnsemble(torch.nn.Module):
    """Either create_model or create_backbone + create_head must be redefined."""
    def __init__(self, num_classes=10, num_models=5, aggregate_logits=True, discriminator_cls=None):
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        self.aggregate_logits = aggregate_logits
        try:
            self.nets = torch.nn.ModuleList([self.create_model(num_classes=num_classes) for _ in range(num_models)])
            feature_size = self.nets[0].fc.in_features
            self.multihead = False
        except NotImplementedError:
            self.backbone = self.create_backbone()
            feature_size = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Identity()
            self.heads = torch.nn.ModuleList([self.create_head(feature_size, num_classes)
                                              for _ in range(num_models)])
            self.multihead = True
        self.discriminator = discriminator_cls(num_classes, feature_size) if discriminator_cls is not None else None

    def create_model(self, num_classes):
        raise NotImplementedError("The method is not implemented")

    def create_backbone(self):
        raise NotImplementedError("The method is not implemented")

    def create_head(self, num_classes):
        raise NotImplementedError("The method is not implemented")

    def forward(self, x, return_ensemble=False):
        # x: (B x C x H x W) or (N x B x C x H x W).
        if x.ndim == 4:
            single_input = True
            single_x = x
            x = [x] * self.num_models  # (N x B x C x H x W).
        elif x.ndim == 5:
            single_input = False
        else:
            raise ValueError(f"Expected input either (B x C x H x W) or (N x B x C x H x W), got {x.shape}")
        if len(x) != self.num_models:
            raise ValueError("Number of inputs and number of models mismatch")
        logits = []
        features = []
        feature_maps = []
        if self.multihead:
            if single_input:
                feature, feature_map = self.backbone(single_x, return_feature_map=True)[1:]
                features.extend([feature] * self.num_models)
                feature_maps.extend([feature_map] * self.num_models)
            else:
                for single_x in x:
                    feature, feature_map = self.backbone(x, return_feature_map=True)[1:]
                    features.append(feature)
                    feature_maps.append(feature_map)
            for feature, head in zip(features, self.heads):
                logits.append(head(feature))
        else:
            for single_x, net in zip(x, self.nets):
                logit, feature, feature_map = net(single_x, return_feature_map=True)
                logits.append(logit)
                features.append(feature)
                feature_maps.append(feature_map)
                assert feature.ndim == 2, feature.shape  # (B, D).
                assert feature_map.ndim == 4, feature_map.shape  # (B, C, H, W).
        logits = torch.stack(logits)  # (N, B, L).
        if self.aggregate_logits:
            agg_logits = logits.mean(0)  # (B, L).
        else:
            # Aggregate probs.
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (N, B, L).
            agg_logits = torch.logsumexp(log_probs, 0) - math.log(len(log_probs))  # (B, L).
        inp = features if self.multihead else x  # Diversify down to this tensor.
        if return_ensemble:
            return agg_logits, logits, inp, features, feature_maps
        else:
            return agg_logits


class LeNet_Ensemble(BaseEnsemble):
    def create_model(self, num_classes):
        return LeNet(num_classes=num_classes, num_channel=3)


class ResNet18_32x32_Ensemble(BaseEnsemble):
    def create_model(self, num_classes):
        return ResNet18_32x32(num_classes=num_classes)


class ResNet18_64x64_Ensemble(BaseEnsemble):
    def create_model(self, num_classes):
        return ResNet18_64x64(num_classes=num_classes)


class ResNet18_224x224_Ensemble(BaseEnsemble):
    def create_model(self, num_classes):
        return ResNet18_224x224(num_classes=num_classes)


class ResNet50_Ensemble(BaseEnsemble):
    def create_model(self, num_classes):
        return ResNet50(num_classes=num_classes)


class ResNet18_32x32_Multihead(BaseEnsemble):
    def create_backbone(self):
        return ResNet18_32x32(num_classes=1)

    def create_head(self, in_features, num_classes):
        return torch.nn.Sequential(*[
            torch.nn.Linear(in_features=in_features, out_features=in_features),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=in_features, out_features=num_classes)
        ])


class ReactNetEnsemble(torch.nn.Module):
    def __init__(self, ensemble: BaseEnsemble):
        super().__init__()
        self.ensemble = ensemble
        for i in range(len(self.ensemble.nets)):
            self.ensemble.nets[i] = ReactNet(self.ensemble.nets[i])

    def forward(self, x, return_ensemble=False):
        return self.ensemble(x, return_ensemble=return_ensemble)

    def forward_threshold(self, x, threshold):
        logits = []
        if isinstance(threshold, float):
            threshold = [threshold] * len(self.ensemble.nets)
        for i, net in enumerate(self.ensemble.nets):
            logits.append(net.forward_threshold(x, threshold[i]))
        agg_logits = sum(logits) / len(logits)
        logits = torch.stack(logits)
        return agg_logits, logits


class AshNetEnsemble(torch.nn.Module):
    def __init__(self, ensemble: BaseEnsemble):
        super().__init__()
        self.ensemble = ensemble
        for i in range(len(self.ensemble.nets)):
            self.ensemble.nets[i] = ASHNet(self.ensemble.nets[i])

    def forward(self, x, return_ensemble=False):
        return self.ensemble(x, return_ensemble=return_ensemble)

    def forward_threshold(self, x, threshold):
        logits = []
        if isinstance(threshold, float):
            threshold = [threshold] * len(self.ensemble.nets)
        for i, net in enumerate(self.ensemble.nets):
            logits.append(net.forward_threshold(x, threshold[i]))
        agg_logits = sum(logits) / len(logits)
        logits = torch.stack(logits)
        return agg_logits, logits
