import torch
from torch.optim.adamw import AdamW
import numpy as np
import time

from .model import (GenerateBatch, GPT)
from .config import GPTConfig

## Set default config
config = GPTConfig()

## training loop
def GPTModelTrain(config=None):
    if config == None:
        config = GPTConfig()

    print(f'device is: {config.device}')

    ## load dataset
    dataset = np.memmap(config.dataset_path, dtype=np.uint32, mode='r')
    ## train/val split
    split_size = int(0.9*len(dataset))
    train = dataset[:split_size]
    val = dataset[split_size:]

    model = GPT(config.num_blocks, config.batch_size, config.block_size, config.num_heads, config.embed_dim, config.max_seq_len, config.vocab_size, config.device)
    model = model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    ## synchronize GPU threads and time training loop
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    ## splitting into mini_batches due to gpu memory constraints
    grad_steps = int(config.batch_size/config.mini_batch_size)
    for step in range(config.num_iters):
        optimizer.zero_grad(set_to_none=True)
        loss_accumulator = 0.0
        for _ in range(grad_steps):
            xb, yb = GenerateBatch(train, config.device, config.block_size, config.mini_batch_size)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(xb, yb, use_flash_attention=config.use_flash_attention)
                loss = loss/grad_steps
            loss.backward()
            loss_accumulator += loss
        if step % 100 == 0: print(f'the loss is {loss_accumulator} on step {step}')
        optimizer.step()

    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")

    return model, config.encoder.decode

def GPTModelGenerate(model:GPT, device, max_seq_len):
    tokens = model.generate(idx=torch.zeros((1,1), dtype=torch.long, device=device), max_tokens=min(max_seq_len, 256), use_cache=config.use_cache, use_flash_attention=config.use_flash_attention)
    return tokens[0].tolist()