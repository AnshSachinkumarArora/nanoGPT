from dataclasses import dataclass
import torch
from pathlib import Path

def get_root_dir():
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / '.git').exists() or (parent / 'requirements.txt').exists():
            return parent

    return current_path.parent

@dataclass
class BigramConfig:
    #hyperparams
    num_iters: int = 10000
    block_size: int = 8
    batch_size: int = 4
    learning_rate: float = 1e-3
    dataset_path: str = str(get_root_dir() / 'dataset' / 'input.txt') 
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')