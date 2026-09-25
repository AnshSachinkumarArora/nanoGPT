import torch
import math
from triton_flash_attention import custom_flash_attention_2

# Setup hyperparams and tensors
B, H, N, D = 4, 32, 4096, 64
dtype = torch.float32
device = 'cuda'
causal = True
sm_scale = 1.0 / math.sqrt(D)

q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
sm_scale = 1.0 / math.sqrt(D)

def run_kernel():
    return custom_flash_attention_2.apply(q, k, v, causal, sm_scale)

# Kernel warmup
for _ in range(5):
    out, L = run_kernel()
    dO = torch.rand_like(out)
    out.backward(dO)
torch.cuda.synchronize()

# Setup random output grads
dO = torch.rand_like(out) 
torch.cuda.synchronize()

# Start profiler
torch.cuda.profiler.start()

# Forward pass
torch.cuda.nvtx.range_push('triton_fa2_forward')
out, L = run_kernel()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

# Backward pass
torch.cuda.nvtx.range_push('triton_fa2_backward')
out.backward(dO)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

# Stop profiler
torch.cuda.profiler.stop()