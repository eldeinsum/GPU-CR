# NVIDIA vLLM Benchmark Results

Date: 2026-05-10

Machine:

- GPU: NVIDIA RTX A6000, 48 GiB
- Driver: 595.71.05
- CUDA: 13.2
- vLLM: 0.14.1
- Python: 3.12

Benchmark shape:

- Models from the GPU-CR README: Llama-8B, Phi-4-mini-instruct, pythia-1b, Qwen3-1.7B.
- The README's Llama path is local/gated. This run used `NousResearch/Meta-Llama-3-8B` as the accessible Llama-8B equivalent.
- vLLM settings: `enforce_eager=True`, `max_model_len=512`, `gpu_memory_utilization=0.80`, one prompt, 16 generated tokens.
- Checkpoint storage used `/mnt/huge-ckpt` on the regular filesystem. The machine has 64 GiB host RAM, so reserving the 50 GiB hugetlbfs region while loading the 8B model is not representative or reliable.

All Rust GPU-CR runs completed post-restore generation successfully.

| Model | Rust GPU-CR Total (s) | C++ GPU-CR Total (s) | CUDA Checkpoint Total (s) | Rust vs C++ |
|---|---:|---:|---:|---:|
| Llama-8B | 19.897 | 26.541 | 14.629 | 1.33x faster |
| Phi-4-mini-instruct | 18.582 | 25.526 | 15.234 | 1.37x faster |
| pythia-1b | 18.114 | 25.429 | 13.854 | 1.40x faster |
| Qwen3-1.7B | 18.973 | 25.511 | 14.034 | 1.34x faster |

Mean totals:

- Rust GPU-CR: 18.891 s
- C++ GPU-CR: 25.752 s
- CUDA Checkpoint: 14.438 s
- Rust GPU-CR is 1.36x faster than the current C++ GPU-CR implementation on this setup.

Rust data/control split:

| Model | Data Checkpoint (s) | Control Checkpoint (s) | Control Restore (s) | Data Restore (s) | VRAM Before (MiB) | VRAM After Data Checkpoint (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Llama-8B | 13.945 | 0.612 | 0.436 | 4.904 | 39190 | 342 |
| Phi-4-mini-instruct | 12.707 | 0.610 | 0.434 | 4.831 | 39216 | 342 |
| pythia-1b | 12.252 | 0.598 | 0.433 | 4.830 | 39194 | 342 |
| Qwen3-1.7B | 13.035 | 0.601 | 0.446 | 4.891 | 39230 | 342 |

Artifacts:

- Raw CSV: `rust/bench/results/vllm_advertised_nvidia_results.csv`
- Summary CSV: `rust/bench/results/vllm_advertised_nvidia_summary.csv`
- Per-run logs: `rust/bench/results/*.log`

Code fixes from the benchmark pass:

- `src/cr_client.cpp` now initializes `ret` to avoid undefined behavior in `-b` buffer-only mode.
- The Rust benchmark harness sets `GPU_VENDOR=NVIDIA` for C++ preload runs.
