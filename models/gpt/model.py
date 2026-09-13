import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from triton_flash_attention import custom_flash_attention_2
torch.manual_seed(117)

## data loader
def GenerateBatch(data, device, block_size, mini_batch_size):
    ix = torch.randint(len(data) - block_size, (mini_batch_size,)).tolist()
    x_np = np.stack([data[i:i+block_size] for i in ix]).astype(np.int64)
    y_np = np.stack([data[i+1:i+1+block_size] for i in ix]).astype(np.int64)
    x = torch.from_numpy(x_np).pin_memory().to(device, non_blocking=True)
    y = torch.from_numpy(y_np).pin_memory().to(device, non_blocking=True)
    return x, y

class LayerNorm(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
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
    def __init__(self, vocab_size, embed_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, tokens):
        embedded = self.embedding(tokens)
        return embedded
    
class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len, device):
        super().__init__()
        ## setting up positions and frequencies
        pos = torch.arange(max_seq_len, device=device)
        freq = 1/(10000**(torch.arange(0, head_dim, 2).float()/head_dim)).to(device=device)
        pos_freq = torch.outer(pos, freq)
        sin_cache = torch.sin(pos_freq)
        cos_cache = torch.cos(pos_freq)
        ## setting up sin and cos caches
        self.register_buffer('sin_cache', torch.cat((sin_cache, sin_cache), dim=1))
        self.register_buffer('cos_cache', torch.cat((cos_cache, cos_cache), dim=1))

    def forward(self, q, k, start_pos=0):
        B, nh, T, hs = q.shape
        ## splitting and negating
        q_split = self.neg_split(q)
        k_split = self.neg_split(k)
        ## getting correct cache values
        sin = self.sin_cache[start_pos : start_pos + T, :]
        cos = self.cos_cache[start_pos : start_pos + T, :]
        ## performing RoPE
        pos_embedded_q = (q * cos) + (q_split * sin)
        pos_embedded_k = (k * cos) + (k_split * sin)
        return pos_embedded_q, pos_embedded_k
    
    ## helper function to negate second half of tensor dim -1 and flip it
    def neg_split(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        output = torch.cat((-x2, x1), dim=-1)
        return output


class CausalSelfAttention(nn.Module):
    def __init__(self, batch_size, block_size, num_heads, embed_dim, max_seq_len, device) -> None:
        super().__init__()
        self.batch_size, self.block_size, self.num_heads, self.embed_dim = batch_size, block_size, num_heads, embed_dim
        self.head_dim = embed_dim//num_heads
        self.weights = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.output_weights = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_weights.NANOGPT_SCALE_INIT = 1
        self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size)))
        self.register_buffer('k_cache', torch.zeros(batch_size, num_heads, block_size, self.head_dim))
        self.register_buffer('v_cache', torch.zeros(batch_size, num_heads, block_size, self.head_dim))
        self.rope = RoPE(self.head_dim, max_seq_len, device)

    def forward(self, tokens, use_cache=False, absolute_pos=0, use_flash_attention=False):
        B, T, C = tokens.shape
        wei = self.weights(tokens) ## (B, T, 3C)

        ## split into QKV matrices of shape (B, T, C)
        q = wei[:, :, :C]
        k = wei[:, :, C:C*2]
        v = wei[:, :, C*2:]

        ## reshape for correct dimensions per head
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        ## TRAINING 
        if use_cache is False:
            q, k = self.rope(q, k)
            sm_scale = k.shape[-1]**(-0.5)
            if use_flash_attention is False:
                ## calculate scaled QK^t, shape (B, T, T)
                qkt = q @ k.transpose(-2, -1)
                qkt_scaled = qkt * sm_scale
                qkt_masked = qkt_scaled.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
                qkt_softmax = F.softmax(qkt_masked, dim=-1)
                ## calculating scaled dot product attention, shape (B, T, C)
                scaled_dot_attn = qkt_softmax @ v
            else:
                scaled_dot_attn, _ = custom_flash_attention_2.apply(q, k, v, True, sm_scale)
        ## INFERENCE 
        else:
            q, k = self.rope(q, k, start_pos=absolute_pos)
            cache_pos = absolute_pos % self.block_size
            ## append to cache
            self.k_cache[:B, :, cache_pos:cache_pos+1, :] = k
            self.v_cache[:B, :, cache_pos:cache_pos+1, :] = v
            ## get correct number of kv values
            if absolute_pos < self.block_size:
                k_hist = self.k_cache[:B, :, :absolute_pos+1, :]
                v_hist = self.v_cache[:B, :, :absolute_pos+1, :]
            else:
                k_hist = self.k_cache[:B, :, :, :]
                v_hist = self.v_cache[:B, :, :, :]

            ## perform attention calculation
            qkt = q @ k_hist.transpose(-2, -1)
            qkt_scaled = qkt * k.shape[-1]**(-0.5)
            qkt_softmax = F.softmax(qkt_scaled, dim=-1)
            scaled_dot_attn = qkt_softmax @ v_hist

        ## reshape attention tensor
        scaled_dot_attn = scaled_dot_attn.transpose(1, 2).reshape(B, T, C)

        ## calculating final output 
        output = self.output_weights(scaled_dot_attn)

        return output


class MLP(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
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
    def __init__(self, batch_size, block_size, num_heads, embed_dim, max_seq_len, device) -> None:
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(batch_size, block_size, num_heads, embed_dim, max_seq_len, device)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x, use_cache=False, absolute_pos=0, use_flash_attention=False):
        x = x + self.attn(self.ln1(x), use_cache, absolute_pos, use_flash_attention)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, num_blocks, batch_size, block_size, num_heads, embed_dim, max_seq_len, vocab_size, device) -> None:
        super().__init__()
        self.batch_size, self.block_size, self.num_heads, self.embed_dim = batch_size, block_size, num_heads, embed_dim
        self.embedding = Embedding(vocab_size, embed_dim)
        self.mha = nn.ModuleList([DecoderBlock(batch_size, block_size, num_heads, embed_dim, max_seq_len, device) for _ in range(num_blocks)])
        self.ln = LayerNorm(embed_dim)
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
            idx = idx if idx.shape[-1] <= self.block_size else idx[:, -self.block_size:]
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
                std *= (2 * self.num_blocks) ** (-0.5)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)