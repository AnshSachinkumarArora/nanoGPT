import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import math
from triton_flash_attention import custom_flash_attention_2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TestTritonFA2:
    # ---------------------------------------------------------------------------
    # Reference implementations (PyTorch)
    # ---------------------------------------------------------------------------
    
    def referenceAttentionNaive(self, q, k, v, causal, sm_scale):
        """
        Straightforward (non-fused) softmax attention, done in fp32 internally
        regardless of input dtype.
    
        q, k, v: (B, H, N, D)
        Returns: (out, L) where L = logsumexp(scores, dim=-1) per query row,
                matching what the kernel should be writing to its `L`/`M` buffer.
        """
        q32, k32, v32 = q.float(), k.float(), v.float()
        scores = torch.matmul(q32, k32.transpose(-2, -1)) * sm_scale  # (B,H,N,N)
    
        if causal:
            N = q.shape[-2]
            mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
    
        L = torch.logsumexp(scores, dim=-1)  # (B,H,N)
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v32)
        return out.to(q.dtype), L
    
    def referenceAttentionSdpa(self, q, k, v, causal, sm_scale):
        """
        PyTorch's fused SDPA. Faster reference, doesn't expose L, so use
        referenceAttentionNaive as the source of truth for L specifically.
        """
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)

    # ---------------------------------------------------------------------------
    # Comparison utilities
    # ---------------------------------------------------------------------------
    
    def compare(self, name, custom, ref, atol, rtol):
        diff = (custom.float() - ref.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        ok = torch.allclose(custom.float(), ref.float(), atol=atol, rtol=rtol)
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {name:<12} max_err={max_err:.3e}  mean_err={mean_err:.3e}  "
            f"(atol={atol:.0e}, rtol={rtol:.0e})")
        if not ok:
            # Show where the worst mismatch is, useful for spotting e.g. a single
            # bad row (mask bug) vs. uniform error (scale/precision bug).
            flat_idx = diff.argmax()
            idx = torch.unravel_index(flat_idx, diff.shape)
            print(f"           worst element at index {tuple(i.item() for i in idx)}: "
                f"custom={custom[idx].item():.6f}  ref={ref[idx].item():.6f}")
        return ok
    
    
    def runCase(self, B, H, N, D, causal, dtype, atol, rtol):
        q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True)
        # Detach and copy input tensors
        q_cust, k_cust, v_cust = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)
        sm_scale = 1.0 / math.sqrt(D)

        # Reference outputs
        ref_out, ref_L = self.referenceAttentionNaive(q, k, v, causal, sm_scale)
        sdpa_out = self.referenceAttentionSdpa(q, k, v, causal, sm_scale)

        # Triton kernel outputs
        custom_out, custom_L = custom_flash_attention_2.apply(q_cust, k_cust, v_cust, causal, sm_scale)

        # Generate random output grads
        dO = torch.randn_like(ref_out)

        # Reference grads
        sdpa_out.backward(dO)
        dQ_ref, dK_ref, dV_ref = q.grad, k.grad, v.grad
    
        # Triton kernel grads
        custom_out.backward(dO)
        dQ_cust, dK_cust, dV_cust = q_cust.grad, k_cust.grad, v_cust.grad

        # Compare all outputs
        out_ok = self.compare("output", custom_out, ref_out, atol=atol, rtol=rtol)
        L_ok = self.compare("logsumexp", custom_L, ref_L, atol=atol, rtol=rtol)
        dQ_ok = self.compare("dQ", dQ_cust, dQ_ref, atol=atol, rtol=rtol)
        dK_ok = self.compare("dK", dK_cust, dK_ref, atol=atol, rtol=rtol)
        dV_ok = self.compare("dV", dV_cust, dV_ref, atol=atol, rtol=rtol)
        return out_ok and L_ok and dQ_ok and dV_ok and dK_ok
    
    @pytest.mark.parametrize('B, H, N, D, causal, dtype, atol, rtol', [
        # (B, H, N, D, causal, dtype, atol, rtol)
        (1, 1, 64,  32,  False, torch.bfloat16, 3e-2, 3e-2),
        (1, 1, 64,  32,  True,  torch.float32, 1e-4, 1e-4),
        (1, 1, 65,  32,  True,  torch.float32, 1e-4, 1e-4),   # ragged tail
        (2, 4, 128, 64,  False, torch.float32, 1e-4, 1e-4),
        (2, 4, 128, 64,  True,  torch.float32, 1e-4, 1e-4),
        (2, 4, 257, 64,  True,  torch.float32, 1e-4, 1e-4),   # ragged tail
        (1, 2, 512, 128, True,  torch.float32, 1e-4, 1e-4),
        # fp16 needs much looser tolerance -- this is expected, not a bug
        (2, 4, 128, 64,  True,  torch.float16, 1e-2, 1e-2),
        (1, 2, 512, 128, True,  torch.float16, 2e-2, 2e-2),
    ])
    def testFlashAttention(self, B, H, N, D, causal, dtype, atol, rtol):
        """
        Small -> large, non-causal -> causal, fp32 -> fp16.
        Deliberately includes N values that are NOT multiples of BLOCK_M/BLOCK_N
        (e.g. 65, 130) to force your ragged-tail masking to actually get exercised.
        Also includes D=32 and D=128 to check you're not hardcoding head_dim.
        """
        assert self.runCase(B, H, N, D, causal, dtype, atol, rtol)