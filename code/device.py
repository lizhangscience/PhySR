# -*- coding: utf-8 -*-
import torch

# Select CUDA if available, otherwise use the CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
