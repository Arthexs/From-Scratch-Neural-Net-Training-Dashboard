"""Device configuration for training (CPU / CUDA)."""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")