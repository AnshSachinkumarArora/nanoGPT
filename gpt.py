import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.mps as mps
from torch.optim.adamw import AdamW
torch.manual_seed(117)

#load dataset
with open('./dataset/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()
chars = sorted(list(set(text)))

#hyperparams
if torch.cuda.is_available():
    device = 'cuda'
elif mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
vocab_size = len(chars)
embed_dim = 128 #C
num_heads = 4 #nh
block_size = 256 #T
batch_size = 64 #B
head_dim = embed_dim//num_heads #hs
num_iters = 3000
num_blocks = 1
learning_rate = 6e-4

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

class LayerNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.epsilon = 1e-5

    def forward(self, idx):
        idx_var = torch.var(idx, dim=-1, keepdim=True, unbiased=False)
        idx_mean = torch.mean(idx, dim=-1, keepdim=True)
        idx_normalized = (idx - idx_mean)/torch.sqrt(idx_var + self.epsilon)
        output = idx_normalized * self.alpha + self.beta
        return output
        
class Embedding(nn.Module):
    '''
    This class will be used to perform the input and positional embedding
    returns a tensor of shape (block_size, embed_dim)
    '''
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = nn.Embedding(block_size, embed_dim)
    
    def forward(self, tokens):
        T = tokens.shape[1]
        embedded = self.embedding(tokens)
        positions = self.positional(torch.arange(T, device=device))
        combined = embedded + positions
        return combined

class CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.output_weights = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_weights.NANOGPT_SCALE_INIT = 1
        self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, tokens):
        B, T, C = tokens.shape
        wei = self.weights(tokens) #(B, T, 3C)

        #split into QKV matrices of shape (B, T, C)
        q = wei[:, :, :C]
        k = wei[:, :, C:C*2]
        v = wei[:, :, C*2:]

        #reshape for correct dimensions per head
        q = q.reshape(B, T, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, T, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, T, num_heads, head_dim).transpose(1, 2)

        #calculate scaled QK^t, shape (B, T, T)
        qkt = q @ k.transpose(-2, -1)
        qkt_scaled = qkt * k.shape[-1]**(-0.5)
        qkt_masked = qkt_scaled.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        qkt_softmax = F.softmax(qkt_masked, dim=-1)

        #calculating scaled dot product attention, shape (B, T, C)
        scaled_dot_attn = qkt_softmax @ v
        scaled_dot_attn = scaled_dot_attn.transpose(1, 2).reshape(B, T, C)

        #calculating final output 
        output = self.output_weights(scaled_dot_attn)

        return output

class MLP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.fc2.NANOGPT_SCALE_INIT = 1

    def forward(self, tokens):
        tokens = self.fc1(tokens)
        tokens = self.gelu(tokens)
        tokens = self.fc2(tokens)
        return tokens
    
class DecoderBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ln1 = LayerNorm()
        self.attn = CausalSelfAttention()
        self.ln2 = LayerNorm()
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, num_blocks, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = Embedding()
        self.mha = nn.Sequential(*[DecoderBlock() for _ in range(num_blocks)])
        self.ln = LayerNorm()
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self._init_weights(self._init_weights)

    def forward(self, x, y = None):
        x = self.embedding(x)
        x = self.mha(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            y = y.reshape(B*T)
            loss = F.cross_entropy(logits, y)
        
        return logits, loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx = idx if idx.shape[-1] <= block_size else idx[:, -block_size:]
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * num_blocks) ** (-0.5)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

model = GPT(num_blocks)
model = model.to(device)

#training loop
optimizer = AdamW(model.parameters(), lr=learning_rate)
for step in range(num_iters):
    xb, yb = Generate_Batch('train')
    logits, loss = model(xb, yb)
    if step % 100 == 0: print(f'the loss is {loss} on step {step}')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(model.generate(idx=torch.zeros((1,1), dtype=torch.long, device=device), max_tokens=500)[0].tolist()))
