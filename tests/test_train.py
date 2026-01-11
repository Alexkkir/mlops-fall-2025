from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from src.train import run_training


@patch("src.train.pd.read_json")
def test_train_one_epoch(mock_read_json, dummy_config, dummy_df, tmp_path):
    # Mock reading dataframe
    mock_read_json.return_value = dummy_df

    # Use CPU and 0 workers to avoid multiprocessing issues in tests
    dummy_config.device = "cpu"
    dummy_config.num_workers = 0
    dummy_config.epochs = 1

    # Change to temp dir so TensorBoard and checkpoints are written there
    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        run_training(dummy_config)

        # Check if files were created in temp dir
        assert os.path.exists("best_model.pth")
        assert os.path.exists("last_model.pth")

    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")
    finally:
        os.chdir(original_cwd)
