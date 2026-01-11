from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def prepare(config: DictConfig):
    # This script will split the data into train and val
    # We use the same config for simplicity to get data_path, but ideally we'd have a separate config

    data_path = Path(config.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_json(data_path, lines=True)

    # Simple split
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Save to data/prepared
    output_dir = Path("data/prepared")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)

    print(f"Data split saved to {output_dir}")
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")


if __name__ == "__main__":
    prepare()
