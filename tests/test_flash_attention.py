from triton_flash_attention.flash_attention_kernel import *

# ---------------------------------------------------------------------------
# Reference implementations (PyTorch, no Triton)
# ---------------------------------------------------------------------------
 
def reference_attention_naive(q, k, v, causal, sm_scale):
    """
    Straightforward (non-fused) softmax attention, done in fp32 internally
    regardless of input dtype, so this is your numerical ground truth.
 
    q, k, v: (B, H, N, D)
    Returns: (out, L) where L = logsumexp(scores, dim=-1) per query row,
             matching what your kernel should be writing to its `L`/`M` buffer.
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
 
 
def reference_attention_sdpa(q, k, v, causal, sm_scale):
    """
    PyTorch's fused SDPA. Faster reference, doesn't expose L, so use
    reference_attention_naive as the source of truth for L specifically.
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)
 
 
# ---------------------------------------------------------------------------
# triton implementation (fwd/bwd)
# ---------------------------------------------------------------------------
 
def custom_attention_fwd(q, k, v, causal, sm_scale):
    """
    TODO: wire this up to your Triton kernel.
 
    Expected contract (match this to whatever you settle on in the kernel):
      q, k, v: (B, H, N, D) contiguous, same dtype
      returns: out (B, H, N, D) same dtype as input
               L   (B, H, N)    fp32, logsumexp per row (natural log, NOT log2 --
                                convert back if your kernel stores log2 internally)
 
    Example wiring once your kernel + Python launcher exist:
 
        from flash_attention2 import attention  # your launcher function
        out, L = attention(q, k, v, causal, sm_scale)
        return out, L
    """
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
                                  causal=causal, 
                                  BLOCK_D=D,
                                  qk_scale=sm_scale)
    return O, L

def custom_attention_bwd(q, k, v, L, o, dO, causal, sm_scale):
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
    
    return dQ, dK, dV

 
# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------
 
def compare(name, custom, ref, atol, rtol):
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
 
 
def run_case(B, H, N, D, causal, dtype, atol, rtol):
    print(f"\n=== B={B} H={H} N={N} D={D} causal={causal} dtype={dtype} ===")
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True) * 0.5
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True) * 0.5
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True) * 0.5
    sm_scale = 1.0 / math.sqrt(D)
 
    ref_out, ref_L = reference_attention_naive(q, k, v, causal, sm_scale)
    sdpa_out = reference_attention_sdpa(q, k, v, causal, sm_scale)
 
    # Sanity check: the two references should already agree closely.
    compare("naive_vs_sdpa", ref_out, sdpa_out, atol=atol, rtol=rtol)

    #generate random output grads
    dO = torch.randn_like(ref_out)

    #get reference grads
    dQ_ref, dK_ref, dV_ref = torch.autograd.grad(
        outputs=ref_out, inputs=[q, k, v], grad_outputs=dO
    )
 
    try:
        #detach and copy input tensors
        q_cust, k_cust, v_cust = q.detach().clone(), k.detach().clone(), v.detach().clone()
        #fwd pass
        custom_out, custom_L = custom_attention_fwd(q_cust, k_cust, v_cust, causal, sm_scale)
        #bwd pass
        dQ, dK, dV = custom_attention_bwd(q_cust, k_cust, v_cust, custom_L, custom_out,
                                          dO, causal, sm_scale)
    except NotImplementedError as e:
        print(f"    SKIPPED: {e}")
        return None
 
    out_ok = compare("output", custom_out, ref_out, atol=atol, rtol=rtol)
    L_ok = compare("logsumexp", custom_L, ref_L, atol=atol, rtol=rtol)
    dQ_ok = compare("dQ", dQ, dQ_ref, atol=atol, rtol=rtol)
    dK_ok = compare("dK", dK, dK_ref, atol=atol, rtol=rtol)
    dV_ok = compare("dV", dV, dV_ref, atol=atol, rtol=rtol)
    return out_ok and L_ok and dQ_ok and dV_ok and dK_ok
 
 
def run_correctness_suite():
    """
    Small -> large, non-causal -> causal, fp32 -> fp16.
    Deliberately includes N values that are NOT multiples of BLOCK_M/BLOCK_N
    (e.g. 65, 130) to force your ragged-tail masking to actually get exercised.
    Also includes D=32 and D=128 to check you're not hardcoding head_dim.
    """
    configs = [
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
    ]
 
    results = []
    for cfg in configs:
        results.append(run_case(*cfg))
 
    ran = [r for r in results if r is not None]
    print("\n" + "=" * 60)
    if not ran:
        print("No cases ran -- custom_attention_fwd is still a stub.")
    else:
        passed = sum(ran)
        print(f"Correctness: {passed}/{len(ran)} cases passed "
              f"({len(results) - len(ran)} skipped)")
    print("=" * 60)
 
 
def run_benchmark(B=4, H=8, N=2048, D=64, causal=True, dtype=torch.float16, iters=50):
    """
    Rough wall-clock comparison vs. PyTorch SDPA. Only meaningful once
    correctness passes -- a fast wrong kernel tells you nothing.
    """
    print(f"\n=== Benchmark: B={B} H={H} N={N} D={D} causal={causal} dtype={dtype} ===")
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True) * 0.5
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True) * 0.5
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True) * 0.5
    q_cust, k_cust, v_cust = q.detach().clone(), k.detach().clone(), v.detach().clone()
    sm_scale = 1.0 / math.sqrt(D)
 
    def timeit(fn, iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(5):  # warmup
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters  # ms per iter
    
    sdpa_out = reference_attention_sdpa(q, k, v, causal, sm_scale)
    custom_out, custom_L = custom_attention_fwd(q_cust, k_cust, v_cust, causal, sm_scale)
    #generate random output grads
    dO = torch.randn_like(sdpa_out)

    sdpa_ms = timeit(lambda: reference_attention_sdpa(q, k, v, causal, sm_scale), iters)
    sdpa_ms_bwd = timeit(lambda: torch.autograd.grad(outputs=sdpa_out, inputs=[q, k, v], grad_outputs=dO, retain_graph=True), iters)
    print(f"    PyTorch SDPA Fwd: {sdpa_ms:.3f} ms/iter")
    print(f"    PyTorch SDPA Bwd: {sdpa_ms_bwd:.3f} ms/iter")
 
    try:
        custom_ms = timeit(lambda: custom_attention_fwd(q_cust, k_cust, v_cust, causal, sm_scale), iters)
        custom_ms_bwd = timeit(lambda: custom_attention_bwd(q_cust, k_cust, v_cust, custom_L, custom_out, dO, causal, sm_scale), iters)
        print(f"    Custom kernel Fwd: {custom_ms:.3f} ms/iter  "
              f"    Custom kernel Bwd: {custom_ms_bwd:.3f} ms/iter  "
              f"({sdpa_ms / custom_ms:.2f}x vs SDPA Fwd)"
              f"({sdpa_ms_bwd / custom_ms_bwd:.2f}x vs SDPA Bwd)")
    except NotImplementedError as e:
        print(f"    SKIPPED: {e}")
 
 
if __name__ == "__main__":
    if DEVICE != "cuda":
        print("WARNING: no CUDA device found, running on CPU. Triton kernel "
              "comparisons will not work; only the two reference impls will run.")
 
    run_correctness_suite()
 
    if DEVICE == "cuda":
        run_benchmark()