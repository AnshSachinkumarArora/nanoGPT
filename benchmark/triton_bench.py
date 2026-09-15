import torch
import triton
import math
import os
from triton_flash_attention import custom_flash_attention_2

B, H = 4, 32
configs = []
triton_fa2 = custom_flash_attention_2.apply

for D in [64, 128]:
    for mode in ['fwd', 'bwd']:
        for causal in [True, False]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N"],
                    x_vals=[2**i for i in range(10, 15)],
                    line_arg="provider",
                    line_vals=["triton-fp16", "torch"],
                    line_names=["Triton [FP16]", "Torch [FP16]"],
                    styles=[("red", "-"), ("blue", "-")],
                    ylabel="TFLOPS",
                    plot_name=
                    f"fused-attention-batch={B}-head={H}-d={D}-mode={mode}-causal={causal}",
                    args={
                        "H": H,
                        "B": B,
                        "D": D,
                        "mode": mode,
                        "causal": causal,
                    },
                )
            )

@triton.testing.perf_report(configs)
def benchTritonFlashAttention(B, H, N, D, mode, causal, provider, device='cuda'):
    dtype = torch.float16
    assert mode in ['fwd', 'bwd']

    if 'triton' in provider:
        q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        sm_scale = 1.0 / math.sqrt(D)
        fn = lambda: triton_fa2(q, k, v, causal, sm_scale)
        if mode == 'bwd':
            o, l = fn()
            dO = torch.rand_like(o)
            fn = lambda: o.backward(dO, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if 'torch' in provider:
        q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        sm_scale = 1.0 / math.sqrt(D)
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)
        if mode == 'bwd':
            o = fn()
            dO = torch.rand_like(o)
            fn = lambda: o.backward(dO, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    matmul_flops = (2 * N * D * N) * B * H 
    total_flops = 2 * matmul_flops
    if causal:
        total_flops *= 0.5 # The implementation does stage based causal masking, meaning, upper triangular of the matrix is not calculated
    if mode == 'bwd':
        total_flops *= 2.5 # +2 for bwd +0.5 for recomputation

    return total_flops * 1e-12 / (ms * 1e-3)

def main():
    # Abort benchmark if cuda is not available
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available, aborting benchmark')

    # Create output dir
    path = './bench_run_outputs'
    os.makedirs(path, exist_ok=True)

    # Run benchmark
    benchTritonFlashAttention.run(save_path=path, print_data=True)

if __name__ == '__main__':
    main()