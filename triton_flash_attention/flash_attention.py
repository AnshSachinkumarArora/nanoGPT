import triton
import torch
from .flash_attention_kernel import (flash_attn_fwd, flash_attn_bwd_delta, flash_attn_bwd_dk_dv, flash_attn_bwd_dq, alloc_fn)

class custom_flash_attention_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        triton.set_allocator(alloc_fn)

        B, H, N, D = q.shape
        O = torch.zeros_like(q)
        L = torch.zeros((B, H, N), device=q.device, dtype=torch.float32)
        grid = lambda META: (triton.cdiv(N, META['BLOCK_M']), B * H)
        flash_attn_fwd[grid](q, k, v, O, L,
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