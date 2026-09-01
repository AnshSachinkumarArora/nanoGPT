import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.mps as mps
from torch.optim.adamw import AdamW
torch.manual_seed(117)
import numpy as np
import tiktoken
from flash_attention.triton.flash_attention2 import custom_flash_attention_2
import time

#load dataset
'''with open('./dataset/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()
chars = sorted(list(set(text)))'''

#hyperparams
if torch.cuda.is_available():
    device = 'cuda'
elif mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
vocab_size = 100256
embed_dim = 512 #C
num_heads = 4 #nh
block_size = 256 #T
batch_size = 64 #B
mini_batch_size = 4
head_dim = embed_dim//num_heads #hs
num_iters = 1000
num_blocks = 1
learning_rate = 6e-4
dataset_path = 'dataset/dataset.bin'
encoder = tiktoken.get_encoding('cl100k_base')
max_seq_len = 1024

#basic encoding and decoding
'''stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])'''

#dataset creation
dataset = np.memmap(dataset_path, dtype=np.uint32, mode='r')
#train/val split
split_size = int(0.9*len(dataset))
train = dataset[:split_size]
val = dataset[split_size:]

#data loader
def Generate_Batch(split):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (mini_batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
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
    
    def forward(self, tokens):
        embedded = self.embedding(tokens)
        return embedded
    
class RoPE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #setting up positions and frequencies
        pos = torch.arange(max_seq_len, device=device)
        freq = 1/(10000**(torch.arange(0, head_dim, 2).float()/head_dim)).to(device=device)
        pos_freq = torch.outer(pos, freq)
        sin_cache = torch.sin(pos_freq)
        cos_cache = torch.cos(pos_freq)
        #setting up sin and cos caches
        self.register_buffer('sin_cache', torch.cat((sin_cache, sin_cache), dim=1))
        self.register_buffer('cos_cache', torch.cat((cos_cache, cos_cache), dim=1))

    def forward(self, q, k, start_pos=0):
        B, nh, T, hs = q.shape
        #splitting and negating
        q_split = self.neg_split(q)
        k_split = self.neg_split(k)
        #getting correct cache values
        sin = self.sin_cache[start_pos : start_pos + T, :]
        cos = self.cos_cache[start_pos : start_pos + T, :]
        #performing RoPE
        pos_embedded_q = (q * cos) + (q_split * sin)
        pos_embedded_k = (k * cos) + (k_split * sin)
        return pos_embedded_q, pos_embedded_k
    
    #helper function to negate second half of tensor dim -1 and flip it
    def neg_split(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        output = torch.cat((-x2, x1), dim=-1)
        return output


class CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.output_weights = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_weights.NANOGPT_SCALE_INIT = 1
        self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size)))
        self.register_buffer('k_cache', torch.zeros(batch_size, num_heads, block_size, head_dim))
        self.register_buffer('v_cache', torch.zeros(batch_size, num_heads, block_size, head_dim))
        self.rope = RoPE()

    def forward(self, tokens, use_cache=False, absolute_pos=0, use_flash_attention=False):
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

        #TRAINING 
        if use_cache is False:
            q, k = self.rope(q, k)
            sm_scale = k.shape[-1]**(-0.5)
            if use_flash_attention is False:
                #calculate scaled QK^t, shape (B, T, T)
                qkt = q @ k.transpose(-2, -1)
                qkt_scaled = qkt * sm_scale
                qkt_masked = qkt_scaled.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
                qkt_softmax = F.softmax(qkt_masked, dim=-1)
                #calculating scaled dot product attention, shape (B, T, C)
                scaled_dot_attn = qkt_softmax @ v
            else:
                scaled_dot_attn, _ = custom_flash_attention_2.apply(q, k, v, True, sm_scale)
        #INFERENCE 
        else:
            q, k = self.rope(q, k, start_pos=absolute_pos)
            cache_pos = absolute_pos % block_size
            #append to cache
            self.k_cache[:B, :, cache_pos:cache_pos+1, :] = k
            self.v_cache[:B, :, cache_pos:cache_pos+1, :] = v
            #get correct number of kv values
            if absolute_pos < block_size:
                k_hist = self.k_cache[:B, :, :absolute_pos+1, :]
                v_hist = self.v_cache[:B, :, :absolute_pos+1, :]
            else:
                k_hist = self.k_cache[:B, :, :, :]
                v_hist = self.v_cache[:B, :, :, :]

            #perform attention calculation
            qkt = q @ k_hist.transpose(-2, -1)
            qkt_scaled = qkt * k.shape[-1]**(-0.5)
            qkt_softmax = F.softmax(qkt_scaled, dim=-1)
            scaled_dot_attn = qkt_softmax @ v_hist

        #reshape attention tensor
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

    def forward(self, x, use_cache=False, absolute_pos=0, use_flash_attention=False):
        x = x + self.attn(self.ln1(x), use_cache, absolute_pos, use_flash_attention)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, num_blocks, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = Embedding()
        self.mha = nn.ModuleList([DecoderBlock() for _ in range(num_blocks)])
        self.ln = LayerNorm()
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.apply(self._init_weights)

    def forward(self, x, y=None, use_cache=False, absolute_pos=0, use_flash_attention=False):
        x = self.embedding(x)
        for layer in self.mha:
            x = layer(x, use_cache=use_cache, absolute_pos=absolute_pos, use_flash_attention=use_flash_attention)
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
    
    def generate(self, idx, max_tokens, use_cache=False):
        absolute_pos=0
        for _ in range(max_tokens):
            idx = idx if idx.shape[-1] <= block_size else idx[:, -block_size:]
            if use_cache is False:
                logits, _ = self(idx, use_cache=use_cache, absolute_pos=absolute_pos)
            else:
                latest_token = idx[:, -1:]
                logits, _ = self(latest_token, use_cache=use_cache, absolute_pos=absolute_pos)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            absolute_pos += 1
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
#splitting into mini_batches due to gpu memory constraints

#time training loop
start_time = time.perf_counter()

grad_steps = int(batch_size/mini_batch_size)
for step in range(num_iters):
    optimizer.zero_grad(set_to_none=True)
    loss_accumulator = 0.0
    for _ in range(grad_steps):
        xb, yb = Generate_Batch('train')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(xb, yb, use_flash_attention=True)
            loss = loss/grad_steps
        loss.backward()
        loss_accumulator += loss
    if step % 100 == 0: print(f'the loss is {loss_accumulator} on step {step}')
    optimizer.step()

end_time = time.perf_counter()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
print(encoder.decode(model.generate(idx=torch.zeros((1,1), dtype=torch.long, device=device), max_tokens=min(max_seq_len, 256), use_cache=True)[0].tolist()))
