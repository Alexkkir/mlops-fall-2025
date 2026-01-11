import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def dummy_images_dir(temp_data_dir):
    images_dir = Path(temp_data_dir) / "images"
    images_dir.mkdir()

    for i in range(5):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img_{i}.jpg"), img)

    return images_dir


@pytest.fixture
def dummy_df(dummy_images_dir):
    data = {
        "bad_image": ["img_0.jpg", "img_1.jpg"],
        "good_image": ["img_2.jpg", "img_3.jpg"],
        "label": [
            1,
            1,
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_config(dummy_images_dir):
    conf = OmegaConf.create(
        {
            "model": {"_target_": "src.models.classic_cv_model.EfficientNetPointwise"},
            "batch_size": 2,
            "lr": 1e-3,
            "epochs": 1,
            "margin": 1.0,
            "num_workers": 0,
            "device": "cpu",
            "data_path": "dummy.csv",
            "images_dir": str(dummy_images_dir),
            "image_size": 64,
            "seed": 42,
        }
    )
    return conf
