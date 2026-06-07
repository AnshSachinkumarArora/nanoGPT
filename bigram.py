import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(117)

#hyperparams
training_steps = 10000
block_size = 8
batch_size = 4
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load dataset
with open('./dataset/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

#basic encoding and decoding
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
#dataset creation
dataset = torch.tensor(encode(text), dtype=torch.long)
#train/val split
split_size = int(0.9*len(dataset))
train = dataset[:split_size]
val = dataset[split_size:]

#data loader
def Generate_Batch(split):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#bigram model
class BigramModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.embedding_table(idx) #(B,T,C)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            #need to change shape to ensure compatibility with cross-entropy
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
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
    
m = BigramModel(vocab_size)
m = m.to(device)

#training loop
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for step in range(training_steps):
    xb, yb = Generate_Batch('train')
    logits, loss = m(xb, yb)
    if step % 100 == 0: print(f'the loss is {loss} on step {step}')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(m.generate(idx=torch.zeros((1,1), dtype=torch.long, device=device), max_tokens=500)[0].tolist()))