import torch
from torch.optim.adamw import AdamW
import numpy as np
from .model import (GenerateBatch, BigramModel)
from .config import BigramConfig

## training loop
def BigramModelTrain(config=None):
    if config == None:
        config = BigramConfig()

    print(f'device is: {config.device}')

    ## load dataset
    with open(config.dataset_path, 'r', encoding='utf-8') as file:
        text = file.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    ## basic encoding and decoding
    stoi = { ch: i for i, ch in enumerate(chars) }
    itos = { i: ch for i, ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    ## dataset creation
    dataset = torch.tensor(encode(text), dtype=torch.long)

    ## train/val split
    split_size = int(0.9*len(dataset))
    train = dataset[:split_size]
    val = dataset[split_size:]

    m = BigramModel(vocab_size)
    m = m.to(config.device)

    optimizer = AdamW(m.parameters(), lr=config.learning_rate)
    for step in range(config.num_iters):
        xb, yb = GenerateBatch(train, config.device, config.block_size, config.batch_size)
        logits, loss = m(xb, yb)
        if step % 100 == 0: print(f'the loss is {loss} on step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return m, decode

## inference
def BigramModelGenerate(m: BigramModel, device, max_tokens):
    tokens = m.generate(idx=torch.zeros((1,1), dtype=torch.long, device=device), max_tokens=max_tokens)
    return tokens[0].tolist()