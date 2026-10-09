# GPT from Scratch with a Hand-Written Triton Flash-Attention 2

A GPT implementation built to understand transformer internals end to end, then extended with a from-scratch **Flash-Attention 2 (forward + backward) in Triton** that matches PyTorch's fused SDPA throughput at long sequence lengths on an RTX 5070.  
Bypassing PyTorch's native high-level abstractions, this repository integrates a custom Triton FlashAttention-2 forward and backward pass, out-of-core data streaming, and production inference primitives, benchmarked and profiled down to SASS instruction counters and memory sector traffic on NVIDIA's Blackwell architecture.  

<!-- Consider a project name: "nanoGPT" undersells the kernel work and invites comparison with Karpathy's repo. -->

<!-- > **Companion project:** the same architecture with every kernel (GEMM, GELU, softmax, attention) hand-written in CUDA: **[TODO: link to cuda-transformer repo]** -->

## Highlights

- **Triton Flash-Attention 2, forward and backward**, implemented from the paper (the official Triton tutorial was used as a reference for optimizations). Causal masking skips fully-masked blocks, loads go through TMA tensor descriptors, and block sizes are autotuned. The backward pass is three kernels (a `delta` precompute, `dQ`, and `dK/dV`) with no atomics.
- **Throughput matches PyTorch SDPA at long sequence lengths**: level at N=8192, +0.4% at N=16384, ~7% behind at N=1024. Peak is ~65 TFLOPS in FP16. See [Results](#results).
- **A complete GPT stack around it**: KV-caching for generation, RoPE, tiktoken BPE, and a bigram baseline.
- **Correctness-tested** against a PyTorch reference and `torch.autograd` across dtypes, causal/non-causal, and sequence lengths that are not multiples of the block size.

| Component                              | Status |
| :------------------------------------- | :----: |
| Triton Flash-Attention 2 (fwd/bwd)     |   ✅   |
| KV-caching                             |   ✅   |
| RoPE                                   |   ✅   |
| tiktoken BPE                           |   ✅   |
| Quantization                           |   ⬜   |
| LoRA                                   |   ⬜   |

## Results

### Setup

| | |
| :--- | :--- |
| GPU | RTX 5070 (sm_120) |
| dtype | FP16 |
| Baseline | `torch.nn.functional.scaled_dot_product_attention` |

<!-- | Software | **[TODO: PyTorch / Triton / CUDA / driver versions]** | -->
<!-- | Benchmark config | **[TODO: batch, heads, head_dim, causal?, forward or backward]** | -->

### Throughput vs. sequence length
<!-- 
![Triton FA2 vs PyTorch SDPA, TFLOPS vs sequence length](assets/fa2_tflops.png) -->

<!-- Replace the values below with the exact table printed by benchmark/triton_bench.py. These were read off the plot. -->

| N      | Triton FA2 (TFLOPS) | PyTorch SDPA (TFLOPS) | Triton / SDPA |
| -----: | ------------------: | --------------------: | ------------: |
| 1,024  | ~58.3               | ~62.5                 | 0.93          |
| 2,048  | ~62.4               | ~63.9                 | 0.98          |
| 4,096  | ~64.2               | ~64.9                 | 0.99          |
| 8,192  | ~65.4               | ~65.4                 | 1.00          |
| 16,384 | ~65.7               | ~65.5                 | 1.00          |

The gap is largest at short sequences and closes by N=8192.
<!-- 
**[TODO: backward-pass throughput vs. SDPA]** -->

### Kernel profiling (Nsight Compute)

<!-- Collected with `ncu --set full`. Run config: **[TODO: dtype, B, H, N, D, causal]**. -->

| Kernel                 | SM throughput | Memory throughput | L1/TEX throughput | Achieved occupancy | Registers / thread |
| :--------------------- | ------------: | ----------------: | ----------------: | -----------------: | -----------------: |
| `flash_attn_fwd`       | 32.2%         | 61.8%             | 56.9%             | 8.3%               | 255                |
| `flash_attn_bwd_delta` | 19.4%         | 95.4%             | 15.8%             | 15.7%              | 82                 |
| `flash_attn_bwd_dq`    | 18.5%         | 92.0%             | 92.7%             | 16.7%              | 255                |
| `flash_attn_bwd_dk_dv` | 31.3%         | 73.4%             | 80.3%             | 16.7%              | 128                |

"Memory throughput" is Nsight Compute's speed-of-light metric: the highest utilization across L1, L2, and DRAM.

### Observations

- `flash_attn_bwd_delta` is a pure streaming kernel. It reads `O` and `dO` once and does one row reduction, so 95% memory throughput against 19% SM throughput is what bandwidth-bound looks like.
- SM throughput is only 18–32% across the attention kernels, so none of them is compute-bound. The forward and `dQ` kernels compile to 255 registers/thread (the architectural cap) and have the lowest occupancy, so register pressure is the main lead to chase.
- A roofline analysis (analytic FLOPs, measured DRAM bytes) is on the [roadmap](#roadmap). It is not included yet because FFMA-based FLOP counting does not apply to FP16 tensor-core kernels.

## Development notes

- **An accidental fp32 upcast cost ~3× backward throughput.** Casting the dot-product operands up to fp32 instead of down to fp16 (with `allow_tf32=False`, that runs on CUDA cores instead of fp16 tensor cores) held backward at 0.21× of SDPA at N=8192, D=128. Fixing it brought it to 0.59×. The correctness tests could not catch it, because the fp32 path is at least as accurate.
- **TMA descriptors made little difference at small sizes.** Forward throughput moved ~1% (0.92× to 0.93× of SDPA) at N=2048, D=64, where load latency was not the bottleneck.

<!-- ## Repository layout

```
.
├── models/                    # GPT and bigram models, training entry point (train.py)
├── triton_flash_attention/    # FA2 kernels + torch.autograd.Function wrapper
├── tests/                     # pytest: FA2 correctness vs. PyTorch reference / autograd
├── benchmark/                 # throughput benchmark, ncu profiling script, report parser
├── dataset/                   # tokenized training data (generated by tokenizer.py)
├── tokenizer.py
└── requirements.txt
``` -->

## Setup

**Requirements:** an NVIDIA GPU with TMA support (Hopper, sm_90, or newer). Developed and tested on an RTX 5070 (sm_120). <!--Triton version: **[TODO]**--> .

1. Install requirements and the local Triton FA2 package:
   ```bash
   pip install -r requirements.txt
   pip install -e . --no-build-isolation
   ```
2. Tokenize the dataset:
   ```bash
   python tokenizer.py
   ```

### Train

```bash
cd models
python train.py --model gpt   # use --model bigram for the bigram baseline
```

### Test the Triton FA2 kernel

```bash
pytest -v tests/
```

### Benchmark

Throughput against PyTorch SDPA:

```bash
cd benchmark
python triton_bench.py
```

Kernel metrics with Nsight Compute (requires the CUDA Toolkit). `parse_ncu_report.py` imports `ncu_report`, which ships with Nsight Compute under `extras/python`. Add that directory to `PYTHONPATH`.

```bash
cd benchmark
ncu -f --profile-from-start off --set full --kernel-name regex:"flash_attn" -o triton_fa2_profile python triton_fa2_ncu_profiler.py
python parse_ncu_report.py --report_path triton_fa2_profile.ncu-repz
```

## Roadmap

- [ ] Investigate the 255-register kernels (spills? block-size / `num_warps` trade-offs)
- [ ] Roofline analysis with analytic FLOPs and the tensor-core roof
- [ ] Quantization
- [ ] LoRA

## References

- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* — https://arxiv.org/abs/2307.08691
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — https://arxiv.org/abs/2205.14135
- Triton fused attention tutorial — https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- Triton documentation — https://triton-lang.org/main/index.html
- nanoGPT by Andrej Karpathy — https://www.youtube.com/watch?v=kCc8FmEb1nY
- KV caching — https://huggingface.co/blog/not-lain/kv-caching
- RoPE — https://arxiv.org/abs/2104.09864
- tiktoken — https://github.com/openai/tiktoken
- The Illustrated Transformer — https://jalammar.github.io/illustrated-transformer/