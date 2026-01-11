import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import PairWiseDataLoader

logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            bad_img = batch["bad_image"].to(device)
            good_img = batch["good_image"].to(device)

            bad_out = model(bad_img)
            good_out = model(good_img)

            target = torch.ones_like(good_out)
            loss = criterion(good_out, bad_out, target)
            total_loss += loss.item()

            correct += (good_out > bad_out).sum().item()
            total += good_out.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(config: DictConfig):
    logger.info(f"Evaluation Config:\n{OmegaConf.to_yaml(config)}")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    val_path = Path(config.val_path)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found at {val_path}")

    val_df = pd.read_csv(val_path)
    logger.info(f"Validation samples: {len(val_df)}")

    val_dataset = PairWiseDataLoader(
        val_df, config.images_dir, image_size=config.get("image_size", 224)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    logger.info("Loading model...")
    model = hydra.utils.instantiate(config.model)

    model_path = "best_model.pth"
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded weights from {model_path}")
    else:
        logger.warning(
            f"No checkpoint found at {model_path}, evaluating with random weights!"
        )

    model.to(device)
    criterion = nn.MarginRankingLoss(margin=config.margin)

    loss, acc = evaluate_model(model, val_loader, device, criterion)
    logger.info(f"Validation Loss: {loss:.4f}")
    logger.info(f"Validation Accuracy: {acc:.4f}")

    with open("metrics.json", "w") as f:
        f.write(f'{{"val_loss": {loss}, "val_accuracy": {acc}}}')


if __name__ == "__main__":
    main()
