from copy import deepcopy
from typing import Tuple
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """Seed all libraries"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_train_val_sets(config: dict, val_fold: int) -> Tuple[np.ndarray]:
    """
    Get train and validation splits based on the defined folds from config.
    Args:
        config:     Configuration dictionary
        val_fold:   Index of validation fold
    """
    data = deepcopy(config["data"]["data_folds"])
    val_set = np.array(data.pop(val_fold))
    train_set = np.concatenate(data)

    return train_set, val_set
