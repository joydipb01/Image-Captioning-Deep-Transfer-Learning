import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_xpu: bool = False) -> torch.device:
    # OpenCL path is handled by OpenCV DNN encoder; decoder uses CPU/CUDA.
    if prefer_xpu and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def worker_count(default: int = 4) -> int:
    cpu_count = os.cpu_count() or default
    return max(1, min(default, cpu_count))
