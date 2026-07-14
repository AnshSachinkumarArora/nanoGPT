import os
import torch
import torch.nn.functional as F
import math
import pytest 
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from typing import Optional
torch.manual_seed(0)

__all__ = ['custom_flash_attention_2']

def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def flash_attn_fwd_inner(q, o_i, m_i, l_i, N,
                        k, v, m_offs, n_offs,
                        qk_scale_log2,
                        STAGE: tl.constexpr):
    #calculate masks
    mask_n = n_offs[None, :] < N
    mask_m = m_offs[:, None] < N

    #calculate attention
    s_i = tl.dot(q, tl.trans(k), allow_tf32=False)
    s_i = tl.where(mask_n, s_i, float('-inf'))
    s_i = tl.where(mask_m, s_i, float('-inf'))

    #apply causal mask during training
    if STAGE == 2:
        mask_qkt = m_offs[:, None] < n_offs[None, :]
        s_i = tl.where(mask_qkt, float('-inf'), s_i)

    #resume attention calculation
    s_i *= qk_scale_log2
    m_ij = tl.maximum(m_i, tl.max(s_i, axis=1))
    s_i -= m_ij[:, None]
    p_i = tl.math.exp2(s_i)
    alpha = tl.math.exp2(m_i - m_ij)
    l_ij = alpha * l_i + tl.sum(p_i, axis=1)
    o_ij = alpha[:, None] * o_i + tl.dot(p_i.to(v.dtype), v, allow_tf32=False)

    return o_ij, l_ij, m_ij

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['N', 'causal', 'BLOCK_D'],
)
@triton.jit
def flash_attn_fwd_kernel_2(Q, K, V, O, L, #input/output matrices
                            B, H, N, D, #matrix shapes
                            stride_qb, stride_qh, stride_qn, stride_qd, #strides for matrix shapes
                            stride_kb, stride_kh, stride_kn, stride_kd,
                            stride_vb, stride_vh, stride_vn, stride_vd,
                            stride_ob, stride_oh, stride_on, stride_od,
                            stride_lb, stride_lh, stride_ln,
                            causal: tl.constexpr, #causal mask to switch between training/inference
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, #block sizes
                            qk_scale: tl.constexpr): #scaling for dot product attention
    #get B, H pid (start of M)
    pid_bh = tl.program_id(axis=1)
    #get starting m of Br block
    pid_m = tl.program_id(axis=0)

    #get batch, head
    batch = pid_bh // H
    head = pid_bh % H

    #get offsets
    base_q_offs = Q + (batch * stride_qb) + (head * stride_qh)
    base_k_offs = K + (batch * stride_kb) + (head * stride_kh)
    base_v_offs = V + (batch * stride_vb) + (head * stride_vh)
    base_o_offs = O + (batch * stride_ob) + (head * stride_oh)
    base_l_offs = L + (batch * stride_lb) + (head * stride_lh)
    m_start = pid_m * BLOCK_M
    m_start = tl.multiple_of(m_start, BLOCK_M)
    m_end = min((pid_m + 1) * BLOCK_M, N) if causal else N
    m_offs = m_start + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)

    #make tensor descriptors
    q_desc = tl.make_tensor_descriptor(base_q_offs, [N, D],
                                       [stride_qn, stride_qd], [BLOCK_M, BLOCK_D])
    k_desc = tl.make_tensor_descriptor(base_k_offs, [N, D],
                                       [stride_kn, stride_kd], [BLOCK_N, BLOCK_D])
    v_desc = tl.make_tensor_descriptor(base_v_offs, [N, D],
                                       [stride_vn, stride_vd], [BLOCK_N, BLOCK_D])

    #load data/setup matrices
    mask_q = m_offs[:, None] < N
    q = q_desc.load([m_start, 0])
    o_i = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + float('-inf')

    #scale qk to log2 for faster operations
    qk_scale_log2 = qk_scale * 1.4426950408889634

    for n_start in range(0, m_end, BLOCK_N):
        n_start = tl.multiple_of(n_start, BLOCK_N)
        #calculate stage
        STAGE = 1
        if causal is True:
            n_end = n_start + BLOCK_N
            STAGE = 1 if m_start >= n_end else 2

        #calculate offsets
        n_offs = n_start + tl.arange(0, BLOCK_N)
        k = k_desc.load([n_start, 0])
        v = v_desc.load([n_start, 0])

        #perform inner loop
        o_ij, l_ij, m_ij = flash_attn_fwd_inner(q, o_i, m_i, l_i, N,
                                                k, v, m_offs, n_offs,
                                                qk_scale_log2,
                                                STAGE)

        #update values
        o_i = o_ij
        l_i = l_ij
        m_i = m_ij
    
    #calculate final output and logsumexp
    o_i /= l_i[:, None]
    #revert log2 scaling
    logsumexp = (m_i + tl.math.log2(l_i))/1.4426950408889634

    #store output
    tl.store(base_o_offs + (m_offs[:, None]) * stride_on + (d_offs[None, :]) * stride_od, o_i, mask=mask_q)
    tl.store(base_l_offs + m_offs * stride_ln, logsumexp, mask=(m_offs < N))

@triton.jit
def flash_attn_bwd_dk_dv_inner(q, k, v, l, d, do, N,
                               m_start, offs_n, mask_n,
                               STAGE: tl.constexpr,
                               qk_scale: tl.constexpr,
                               BLOCK_M: tl.constexpr):
    #calculate s_i and mask
    s_i = tl.dot(q, tl.trans(k), allow_tf32=False)
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < N
    s_i = tl.where(mask_n, s_i, float('-inf'))
    s_i = tl.where(mask_m, s_i, float('-inf'))
    #apply causal mask
    if STAGE == 2:
        mask_causal = offs_m[:, None] >= offs_n[None, :]
        s_i = tl.where(mask_causal, s_i, float('-inf'))
    #calculate output
    p_i = tl.math.exp(s_i * qk_scale - l[:, None])
    #calculate gradients
    dv = tl.dot(tl.trans(p_i.to(do.dtype)), do, allow_tf32=False)
    dp = tl.dot(do.to(v.dtype), tl.trans(v), allow_tf32=False)
    ds = p_i * (dp - d[:, None])
    dk = tl.dot(tl.trans(ds.to(q.dtype)), q, allow_tf32=False) * qk_scale

    return dk, dv

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 16, 'BLOCK_M': 16}, num_warps=8, num_stages=2),
    ],
    key=['N', 'causal', 'BLOCK_D'],
)
@triton.jit
def flash_attn_bwd_dk_dv(Q, K, V, dO, dK, dV, L, delta,
                         B, H, N, D,
                         stride_qb, stride_qh, stride_qn, stride_qd, #strides for matrix shapes
                         stride_kb, stride_kh, stride_kn, stride_kd,
                         stride_vb, stride_vh, stride_vn, stride_vd,
                         stride_dob, stride_doh, stride_don, stride_dod,
                         stride_lb, stride_lh, stride_ln,
                         stride_dkb, stride_dkh, stride_dkn, stride_dkd,
                         stride_dvb, stride_dvh, stride_dvn, stride_dvd,
                         stride_del_b, stride_del_h, stride_del_n,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, #block sizes
                         qk_scale: tl.constexpr, causal: tl.constexpr):
    #get all pids
    pid_bh = tl.program_id(1)
    pid_n = tl.program_id(0)
    batch = pid_bh // H
    head = pid_bh % H

    #get base offsets and masks
    n_start = pid_n * BLOCK_N
    n_start = tl.multiple_of(n_start, BLOCK_N)
    n_end = min((pid_n + 1) * BLOCK_N, N)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n_store = offs_n[:, None] < N
    mask_n = offs_n[None, :] < N
    inner_loop_start = n_start if causal else 0
    offs_m = inner_loop_start + tl.arange(0, BLOCK_M)
    base_q_offs = Q + (batch * stride_qb) + (head * stride_qh)
    base_k_offs = K + (batch * stride_kb) + (head * stride_kh)
    base_v_offs = V + (batch * stride_vb) + (head * stride_vh)
    base_l_offs = L + (batch * stride_lb) + (head * stride_lh)
    base_delta_offs = delta + (batch * stride_del_b) + (head * stride_del_h)
    base_do_offs = dO + (batch * stride_dob) + (head * stride_doh)
    base_dk_offs = dK + (batch * stride_dkb) + (head * stride_dkh)
    base_dv_offs = dV + (batch * stride_dvb) + (head * stride_dvh)
    d_offs = base_delta_offs + (offs_m) * stride_del_n
    l_offs = base_l_offs + (offs_m) * stride_ln

    #make block pointers for TMA
    #make tensor descriptors for TMA
    q_desc = tl.make_tensor_descriptor(base_q_offs, [N, D],
                                       [stride_qn, stride_qd], [BLOCK_M, BLOCK_D])
    k_desc = tl.make_tensor_descriptor(base_k_offs, [N, D],
                                       [stride_kn, stride_kd], [BLOCK_N, BLOCK_D])
    v_desc = tl.make_tensor_descriptor(base_v_offs, [N, D],
                                       [stride_vn, stride_vd], [BLOCK_N, BLOCK_D])
    do_desc = tl.make_tensor_descriptor(base_do_offs, [N, D],
                                       [stride_don, stride_dod], [BLOCK_M, BLOCK_D])
    
    #load data
    k = k_desc.load([n_start, 0])
    v = v_desc.load([n_start, 0])
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for m_start in range(inner_loop_start, N, BLOCK_M):
        m_start = tl.multiple_of(m_start, BLOCK_M)
        #setup stage for causal mask and load blocks
        STAGE = 1
        if causal is True:
            STAGE = 2 if m_start < n_end else 1
        q = q_desc.load([m_start, 0])
        d = tl.load(d_offs, mask=offs_m < N, other=0.0)
        l = tl.load(l_offs, mask=offs_m < N, other=0.0)
        do = do_desc.load([m_start, 0])

        #calculate dK/dV
        dk_j, dv_j = flash_attn_bwd_dk_dv_inner(q, k, v, l, d, do, N,
                                                m_start, offs_n, mask_n,
                                                STAGE, qk_scale, BLOCK_M)
        
        #accumulate grads
        dk += dk_j
        dv += dv_j

        #advance pointers
        offs_m += BLOCK_M
        d_offs += BLOCK_M * stride_del_n
        l_offs += BLOCK_M * stride_ln
    
    #store grads
    tl.store(base_dk_offs + (offs_n[:, None]) * stride_dkn + (offs_d[None, :]) * stride_dkd, dk, mask=mask_n_store)
    tl.store(base_dv_offs + (offs_n[:, None]) * stride_dvn + (offs_d[None, :]) * stride_dvd, dv, mask=mask_n_store)


@triton.jit
def flash_attn_bwd_dq_inner(q, k, v, l, d, do,
                            N, n_start, offs_m,
                            STAGE: tl.constexpr,
                            qk_scale: tl.constexpr,
                            BLOCK_N: tl.constexpr):
    #calculate s_i and mask
    s_i = tl.dot(q, tl.trans(k), allow_tf32=False)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n[None, :] < N
    s_i = tl.where(mask_n, s_i, float('-inf'))
    s_i = tl.where((offs_m[:, None] < N), s_i, float('-inf'))
    #apply causal mask
    if STAGE == 2:
        mask_causal = offs_m[:, None] >= offs_n[None, :]
        s_i = tl.where(mask_causal, s_i, float('-inf'))
    #calculate output
    p_i = tl.math.exp(s_i * qk_scale - l[:, None])
    #calculate gradients
    dp = tl.dot(do.to(v.dtype), tl.trans(v), allow_tf32=False)
    ds = p_i * (dp - d[:, None])
    dq = tl.dot(ds.to(k.dtype), k, allow_tf32=False) * qk_scale

    return dq

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 16, 'BLOCK_M': 16}, num_warps=8, num_stages=2),
    ],
    key=['N', 'causal', 'BLOCK_D'],
)
@triton.jit
def flash_attn_bwd_dq(Q, K, V, dO, dQ, L, delta,
                      B, H, N, D,
                      stride_qb, stride_qh, stride_qn, stride_qd, #strides for matrix shapes
                      stride_kb, stride_kh, stride_kn, stride_kd,
                      stride_vb, stride_vh, stride_vn, stride_vd,
                      stride_dob, stride_doh, stride_don, stride_dod,
                      stride_lb, stride_lh, stride_ln,
                      stride_dqb, stride_dqh, stride_dqn, stride_dqd,
                      stride_del_b, stride_del_h, stride_del_n,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, #block sizes
                      qk_scale: tl.constexpr, causal:tl.constexpr):
    #get all pids
    pid_bh = tl.program_id(1)
    pid_m = tl.program_id(0)
    batch = pid_bh // H
    head = pid_bh % H
    m_start = pid_m * BLOCK_M
    m_start = tl.multiple_of(m_start, BLOCK_M)
    m_end = min((pid_m + 1) * BLOCK_M, N) if causal else N

    #get base offsets
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m[:, None] < N
    base_q_offs = Q + (batch * stride_qb) + (head * stride_qh)
    base_k_offs = K + (batch * stride_kb) + (head * stride_kh)
    base_v_offs = V + (batch * stride_vb) + (head * stride_vh)
    base_l_offs = L + (batch * stride_lb) + (head * stride_lh)
    base_delta_offs = delta + (batch * stride_del_b) + (head * stride_del_h)
    base_do_offs = dO + (batch * stride_dob) + (head * stride_doh)
    base_dq_offs = dQ + (batch * stride_dqb) + (head * stride_dqh)
    d_offs = base_delta_offs + (offs_m) * stride_del_n
    l_offs = base_l_offs + (offs_m) * stride_ln

    #make tensor descriptors for TMA
    q_desc = tl.make_tensor_descriptor(base_q_offs, [N, D],
                                       [stride_qn, stride_qd], [BLOCK_M, BLOCK_D])
    k_desc = tl.make_tensor_descriptor(base_k_offs, [N, D],
                                       [stride_kn, stride_kd], [BLOCK_N, BLOCK_D])
    v_desc = tl.make_tensor_descriptor(base_v_offs, [N, D],
                                       [stride_vn, stride_vd], [BLOCK_N, BLOCK_D])
    do_desc = tl.make_tensor_descriptor(base_do_offs, [N, D],
                                       [stride_don, stride_dod], [BLOCK_M, BLOCK_D])
    
    #load data
    q = q_desc.load([m_start, 0])
    do = do_desc.load([m_start, 0])
    d = tl.load(d_offs, mask=offs_m < N, other=0.0)
    l = tl.load(l_offs, mask=offs_m < N, other=0.0)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, m_end, BLOCK_N):
        n_start = tl.multiple_of(n_start, BLOCK_N)
        #setup stage for causal mask and load blocks
        STAGE = 1
        if causal is True:
            n_end = n_start + BLOCK_N
            STAGE = 2 if m_start < n_end else 1
        k = k_desc.load([n_start, 0])
        v = v_desc.load([n_start, 0])

        #calculate dq
        dq_j = flash_attn_bwd_dq_inner(q, k, v, l, d, do,
                                    N, n_start, offs_m,
                                    STAGE, qk_scale, BLOCK_N)
        
        #accumulate grads
        dq += dq_j

    #store dq
    tl.store(base_dq_offs + (offs_m[:, None]) * stride_dqn + (offs_d[None, :]) * stride_dqd, dq, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 512}, num_warps=8),
    ],
    key=['BLOCK_D'],
)
@triton.jit
def flash_attn_bwd_delta(O, dO, delta,
                         B, H, N, D,
                         stride_ob, stride_oh, stride_on, stride_od,
                         stride_dob, stride_doh, stride_don, stride_dod,
                         stride_del_b, stride_del_h, stride_del_n,
                         BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    #get pids
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // H
    head = pid_bh % H

    #calculate base offsets
    base_o_offs = O + (batch * stride_ob) + (head * stride_oh)
    base_delta_offs = delta + (batch * stride_del_b) + (head * stride_del_h)
    base_do_offs = dO + (batch * stride_dob) + (head * stride_doh)

    #make block ptrs
    m_start = pid_m * BLOCK_M
    do_desc = tl.make_tensor_descriptor(base_do_offs, [N, D],
                                       [stride_don, stride_dod], [BLOCK_M, BLOCK_D])
    o_desc = tl.make_tensor_descriptor(base_o_offs, [N, D],
                                       [stride_on, stride_od], [BLOCK_M, BLOCK_D])
    
    #load data
    o = o_desc.load([m_start, 0])
    do = do_desc.load([m_start, 0])

    #get delta and store
    _delta = o.to(tl.float32) * do.to(tl.float32)
    _delta = tl.sum(_delta, axis=1)
    m_offs = m_start + tl.arange(0, BLOCK_M)
    tl.store(base_delta_offs + (m_offs) * stride_del_n, _delta, mask=(m_offs < N))

class custom_flash_attention_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        triton.set_allocator(alloc_fn)

        B, H, N, D = q.shape
        O = torch.zeros_like(q)
        L = torch.zeros((B, H, N), device=q.device, dtype=torch.float32)
        grid = lambda META: (triton.cdiv(N, META['BLOCK_M']), B * H)
        flash_attn_fwd_kernel_2[grid](q, k, v, O, L,
                                    B, H, N, D,
                                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                                    O.stride(0), O.stride(1), O.stride(2), O.stride(3),
                                    L.stride(0), L.stride(1), L.stride(2),
                                    causal, #BLOCK_M=32, #BLOCK_N=32, 
                                    BLOCK_D=D,
                                    qk_scale=sm_scale)
        
        ctx.save_for_backward(q, k, v, O, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        return O, L

    @staticmethod
    def backward(ctx, grad_output, grad_L=None):
        triton.set_allocator(alloc_fn)

        q, k, v, o, L = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        dO = grad_output

        #get shape
        B, H, N, D = q.shape
        
        #create empty tensors for gradients/delta
        dQ = torch.zeros_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)
        delta = torch.zeros((B, H, N), device=q.device, dtype=torch.float32)

        #get strides
        stride_ob, stride_oh, stride_on, stride_od = o.stride(0), o.stride(1), o.stride(2), o.stride(3)
        stride_dob, stride_doh, stride_don, stride_dod = dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3)
        stride_del_b, stride_del_h, stride_del_n = delta.stride(0), delta.stride(1), delta.stride(2)
        stride_qb, stride_qh, stride_qn, stride_qd = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
        stride_kb, stride_kh, stride_kn, stride_kd = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
        stride_vb, stride_vh, stride_vn, stride_vd = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
        stride_lb, stride_lh, stride_ln = L.stride(0), L.stride(1), L.stride(2)
        stride_dqb, stride_dqh, stride_dqn, stride_dqd = dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3)
        stride_dkb, stride_dkh, stride_dkn, stride_dkd = dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3)
        stride_dvb, stride_dvh, stride_dvn, stride_dvd = dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3)

        #calculate delta
        grid_delta = lambda META: (triton.cdiv(N, META['BLOCK_M']), B * H)
        flash_attn_bwd_delta[grid_delta](o, dO, delta,
                                        B, H, N, D,
                                        stride_ob, stride_oh, stride_on, stride_od,
                                        stride_dob, stride_doh, stride_don, stride_dod,
                                        stride_del_b, stride_del_h, stride_del_n,
                                        BLOCK_D=D)
        
        #calculate dQ
        grid_dq = lambda META: (triton.cdiv(N, META['BLOCK_M']), B * H)
        flash_attn_bwd_dq[grid_dq](q, k, v, dO, dQ, L, delta,
                                B, H, N, D,
                                stride_qb, stride_qh, stride_qn, stride_qd, #strides for matrix shapes
                                stride_kb, stride_kh, stride_kn, stride_kd,
                                stride_vb, stride_vh, stride_vn, stride_vd,
                                stride_dob, stride_doh, stride_don, stride_dod,
                                stride_lb, stride_lh, stride_ln,
                                stride_dqb, stride_dqh, stride_dqn, stride_dqd,
                                stride_del_b, stride_del_h, stride_del_n,
                                BLOCK_D=D, qk_scale=sm_scale, causal=causal)
        
        #calculate dK/dV
        grid_dk_dv = lambda META: (triton.cdiv(N, META['BLOCK_N']), B * H)
        flash_attn_bwd_dk_dv[grid_dk_dv](q, k, v, dO, dK, dV, L, delta,
                                        B, H, N, D,
                                        stride_qb, stride_qh, stride_qn, stride_qd, #strides for matrix shapes
                                        stride_kb, stride_kh, stride_kn, stride_kd,
                                        stride_vb, stride_vh, stride_vn, stride_vd,
                                        stride_dob, stride_doh, stride_don, stride_dod,
                                        stride_lb, stride_lh, stride_ln,
                                        stride_dkb, stride_dkh, stride_dkn, stride_dkd,
                                        stride_dvb, stride_dvh, stride_dvn, stride_dvd,
                                        stride_del_b, stride_del_h, stride_del_n,
                                        BLOCK_D=D, qk_scale=sm_scale, causal=causal)
        
        return dQ, dK, dV, None, None
    
