import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
torch.manual_seed(117)

def GenerateBatch(data, device, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class BigramModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.embedding_table(idx) #(B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            #need to change shape to ensure compatibility with cross-entropy
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            #get the logits for the idx
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            #get the softmax probabilities
            probs = F.softmax(logits, dim=-1)
            #get the next token
            token = torch.multinomial(probs, num_samples=1)
            #attach generated token to sequence
            idx = torch.cat((idx, token), dim=1)
        return idx