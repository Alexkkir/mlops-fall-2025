import pytest
import torch
import numpy as np
from src.datasets import PairWiseDataLoader

def test_dataset_len(dummy_df, dummy_images_dir):
    dataset = PairWiseDataLoader(dummy_df, dummy_images_dir)
    assert len(dataset) == len(dummy_df)

def test_dataset_getitem(dummy_df, dummy_images_dir):
    image_size = 64
    dataset = PairWiseDataLoader(dummy_df, dummy_images_dir, image_size=image_size)
    item = dataset[0]
    
    assert "bad_image" in item
    assert "good_image" in item
    
    bad_img = item["bad_image"]
    good_img = item["good_image"]
    
    # Check shape: (C, H, W) -> (3, 64, 64)
    assert bad_img.shape == (3, image_size, image_size)
    assert good_img.shape == (3, image_size, image_size)
    
    # Check type
    assert isinstance(bad_img, torch.Tensor)
    assert bad_img.dtype == torch.float32
    
    # Check normalization (roughly, just max <= 1.0)
    assert bad_img.max() <= 1.0
    assert bad_img.min() >= 0.0

def test_dataset_missing_file(dummy_df, dummy_images_dir):
    # Modify dataframe to point to non-existent file
    dummy_df.iloc[0, 0] = "missing.jpg"
    dataset = PairWiseDataLoader(dummy_df, dummy_images_dir)
    
    with pytest.raises(ValueError):
        _ = dataset[0]
