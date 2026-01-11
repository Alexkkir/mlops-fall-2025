from typing import *

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


class PairWiseDataLoader(Dataset):
    def __init__(self, df, images_root, image_size=224):
        super().__init__()
        self.df = df
        self.images_root = Path(images_root)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def _read_image(self, path: Path) -> np.array:
        path = str(path)
        image = cv2.imread(path)
        if image is None:
             # Handle missing image or bad path gracefully, though for training we might want to crash or skip
             raise ValueError(f"Failed to load image at {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.from_numpy(image).float() / 255
        image = image.permute(2, 0, 1)
        return image

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        row = self.df.iloc[index]
        bad_image = self._read_image(self.images_root / row["bad_image"])
        good_image = self._read_image(self.images_root / row["good_image"])
        return {"bad_image": bad_image, "good_image": good_image}
