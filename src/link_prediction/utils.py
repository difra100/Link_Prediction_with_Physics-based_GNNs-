import numpy as np
import random
import torch
import torch_geometric
import pytorch_lightning as pl

def set_seed(seed_value):
    # Set seed for NumPy
    np.random.seed(seed_value)

    # Set seed for Python's random module
    random.seed(seed_value)

    # Set seed for PyTorch
    torch.manual_seed(seed_value)

    # Set seed for GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        # Set the deterministic behavior for cudNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set seed for PyTorch Geometric
    torch_geometric.seed_everything(seed_value)

    # Set seed for PyTorch Lightning
    pl.seed_everything(seed_value)
    print(f"{seed_value} have been correctly set!")
