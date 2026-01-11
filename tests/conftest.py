import os
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from omegaconf import OmegaConf

@pytest.fixture
def temp_data_dir():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Remove the directory after the test
    shutil.rmtree(temp_dir)

@pytest.fixture
def dummy_images_dir(temp_data_dir):
    images_dir = Path(temp_data_dir) / "images"
    images_dir.mkdir()
    
    # Create a few dummy images
    for i in range(5):
        # Create a random image 100x100
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img_{i}.jpg"), img)
        
    return images_dir

@pytest.fixture
def dummy_df(dummy_images_dir):
    data = {
        "bad_image": ["img_0.jpg", "img_1.jpg"],
        "good_image": ["img_2.jpg", "img_3.jpg"],
        "label": [1, 1] # dummy label, though not strictly used by dataset class as it assumes pairs
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_config(dummy_images_dir):
    conf = OmegaConf.create({
        "model": {
            "_target_": "src.models.classic_cv_model.EfficientNetPointwise"
        },
        "batch_size": 2,
        "lr": 1e-3,
        "epochs": 1,
        "margin": 1.0,
        "num_workers": 0,
        "device": "cpu",
        "data_path": "dummy.csv", # won't be used directly if we mock dataframe
        "images_dir": str(dummy_images_dir),
        "image_size": 64, # smaller size for tests
        "seed": 42
    })
    return conf
