import pytest
import torch
from src.models.classic_cv_model import EfficientNetPointwise

def test_model_structure():
    model = EfficientNetPointwise()
    # Check if last layer is modified to output 1 value
    # EfficientNet B2 classifier[1] is the Linear layer
    assert isinstance(model.backbone.classifier[1], torch.nn.Linear)
    assert model.backbone.classifier[1].out_features == 1

def test_model_forward():
    model = EfficientNetPointwise()
    model.eval()
    
    # Create dummy input (B, C, H, W) -> (2, 3, 224, 224)
    # Note: EfficientNet usually expects 224+, but works on smaller if fully convolutional or adapted.
    # However, standard B2 might complain if too small? 
    # Let's use 224 to be safe as per default, or verify if it handles variable.
    # EfficientNet adjusts to input size, but too small might reduce spatial dims to 0.
    input_tensor = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(input_tensor)
        
    # Expect output shape (B, 1)
    assert output.shape == (2, 1)
