from dataclasses import dataclass
import torch
import tiktoken
from pathlib import Path

def get_root_dir():
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / '.git').exists() or (parent / 'requirements.txt').exists():
            return parent

    return current_path.parent

@dataclass
class GPTConfig:
    #hyperparams
    vocab_size: int = 100256
    embed_dim: int = 512 ## C
    num_heads: int = 4 ## nh
    block_size: int = 256 ## T
    batch_size: int = 64 ## B
    mini_batch_size: int = 4
    head_dim: int = embed_dim//num_heads ## hs
    num_iters: int = 1000
    num_blocks: int = 1
    learning_rate: float = 6e-4
    dataset_path: str = str(get_root_dir() / 'dataset' / 'dataset.bin') 
    encoder: tiktoken.Encoding = tiktoken.get_encoding('cl100k_base')
    use_cache: bool = True
    use_flash_attention: bool = True
    max_seq_len: int = 1024
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')