import pytest
import torch

from src.models.classic_cv_model import EfficientNetPointwise


def test_model_structure():
    model = EfficientNetPointwise()
    assert isinstance(model.backbone.classifier[1], torch.nn.Linear)
    assert model.backbone.classifier[1].out_features == 1


def test_model_forward():
    model = EfficientNetPointwise()
    model.eval()

    input_tensor = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (2, 1)
