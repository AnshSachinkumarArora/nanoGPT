from dataclasses import dataclass
import torch
import tiktoken

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
    dataset_path: str = 'dataset/dataset.bin'
    encoder: function = tiktoken.get_encoding('cl100k_base')
    max_seq_len: int = 1024
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')