import logging
import os
import random
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets import PairWiseDataLoader

logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            bad_img = batch["bad_image"].to(device)
            good_img = batch["good_image"].to(device)

            bad_out = model(bad_img)
            good_out = model(good_img)

            # Target is 1 because we want good_out > bad_out
            target = torch.ones_like(good_out)
            loss = criterion(good_out, bad_out, target)
            total_loss += loss.item()

            # Accuracy: count how many times good_score > bad_score
            correct += (good_out > bad_out).sum().item()
            total += good_out.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def run_training(config: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    if "seed" in config:
        seed_everything(config.seed)

    # 1. Setup Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Load Data
    # Expecting train and val csv paths from config
    train_path = Path(config.train_path)
    val_path = Path(config.val_path)
    
    if not train_path.exists() or not val_path.exists():
         logger.info("Train/Val paths not found or not specified. Falling back to data_path splitting...")
         data_path = Path(config.data_path)
         df = pd.read_json(data_path, lines=True)
         
         train_size = int(0.8 * len(df))
         train_df = df.iloc[:train_size]
         val_df = df.iloc[train_size:]
    else:
        logger.info(f"Loading train data from {train_path}")
        train_df = pd.read_csv(train_path)
        
        logger.info(f"Loading val data from {val_path}")
        val_df = pd.read_csv(val_path)
    
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Create Datasets
    train_dataset = PairWiseDataLoader(train_df, config.images_dir, image_size=config.get("image_size", 224))
    val_dataset = PairWiseDataLoader(val_df, config.images_dir, image_size=config.get("image_size", 224))

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )

    # 3. Initialize Model
    logger.info("Initializing model...")
    model = hydra.utils.instantiate(config.model)
    model.to(device)

    # 4. Setup Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # MarginRankingLoss takes inputs (x1, x2, y) and computes max(0, -y * (x1 - x2) + margin)
    # With y=1, we want x1 > x2 (good > bad). Loss = max(0, -(good - bad) + margin)
    criterion = nn.MarginRankingLoss(margin=config.margin)

    # 5. TensorBoard Setup
    writer = SummaryWriter(log_dir=os.getcwd())  # Hydra changes cwd to outputs/...

    # MLflow Setup
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment("mlops-hw1")

    # 6. Training Loop
    global_step = 0
    best_acc = 0.0

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        
        for epoch in range(config.epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{config.epochs}")
            model.train()
            
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
            
            for batch in progress_bar:
                bad_img = batch["bad_image"].to(device)
                good_img = batch["good_image"].to(device)

                # Forward pass
                bad_out = model(bad_img)
                good_out = model(good_img)

                # Calculate loss
                target = torch.ones_like(good_out)
                loss = criterion(good_out, bad_out, target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Logging
                loss_val = loss.item()
                epoch_loss += loss_val
                global_step += 1
                
                writer.add_scalar("Train/Loss_Step", loss_val, global_step)
                mlflow.log_metric("train_loss_step", loss_val, step=global_step)
                progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})

            avg_train_loss = epoch_loss / len(train_loader)
            writer.add_scalar("Train/Loss_Epoch", avg_train_loss, epoch)
            mlflow.log_metric("train_loss_epoch", avg_train_loss, step=epoch)
            logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

            # Validation
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("Val/Accuracy", val_acc, epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            logger.info(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            # Save checkpoint if best
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")
                logger.info(f"New best model saved with accuracy: {best_acc:.4f}")
                mlflow.log_artifact("best_model.pth")
            
            # Save last checkpoint
            torch.save(model.state_dict(), "last_model.pth")
            
        mlflow.log_artifact("last_model.pth")

    writer.close()
    logger.info("Training finished!")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def train(config: DictConfig):
    run_training(config)


if __name__ == "__main__":
    train()
