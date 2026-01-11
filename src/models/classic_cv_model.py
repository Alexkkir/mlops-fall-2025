import torch
import torchvision
from torch import nn


class EfficientNetPointwise(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.efficientnet_b2(
            weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT
        )

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)

        self.register_buffer(
            "_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def prepare(self, image):
        image = (image - self._mean) / self._std
        return image

    def forward(self, x):
        x = self.prepare(x)
        return self.backbone(x)
