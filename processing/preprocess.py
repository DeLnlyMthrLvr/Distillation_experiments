import torch
import numpy as np


def to_tensor(data, device):
    """Convert data to PyTorch tensor and move to device."""
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32).to(device)
    return data.to(device)