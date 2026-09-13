from dataclasses import dataclass
import torch
import torch.backends.mps as mps

@dataclass
class BigramConfig:
    #hyperparams
    num_iters: int = 10000
    block_size: int = 8
    batch_size: int = 4
    learning_rate: float = 1e-3
    dataset_path: str = 'dataset/input.txt' 
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')