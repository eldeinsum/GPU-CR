# NVIDIA Runtime

GPU-CR includes a Rust NVIDIA runtime alongside the original C++ implementation. It uses the same `LD_PRELOAD` workflow and a separate client binary.

Runtime names:

- C++ NVIDIA: `vGPU-NVIDIA.so` with `cr_client`
- Rust NVIDIA: `gpucr-nvidia.so` with `gpucr-client`

## Build

```bash
cmake -S . -B build-nvidia -DGPU_VENDOR=NVIDIA -DGPUCR_BUILD_CPP=OFF -DGPUCR_BUILD_RUST=ON -DGPUCR_RUST_RELEASE=ON
cmake --build build-nvidia -j$(nproc)
```

Artifacts:

- `build-nvidia/gpucr-nvidia.so`
- `build-nvidia/gpucr-client`

The legacy C++ NVIDIA build remains available with the default CMake options.

## Usage

```bash
LD_PRELOAD=/path/to/build-nvidia/gpucr-nvidia.so <application>
```

Initialize, checkpoint, and restore:

```bash
./gpucr-client -i -p <pid>
./gpucr-client -c -p <pid>
./gpucr-client -r -p <pid>
```

The client also supports positional commands:

```bash
./gpucr-client checkpoint <pid>
./gpucr-client restore <pid>
```

## Validation

Quick CUDA smoke test:

```bash
rust/smoke/run_cuda_ckpt_smoke.sh
```

vLLM benchmark harness:

```bash
source .venv-bench/bin/activate
python rust/bench/run_vllm_benchmarks.py --models llama-8b,phi-4-mini-instruct,pythia-1b,qwen3-1.7b
```

The current NVIDIA benchmark results are summarized in `docs/nvidia-vllm-benchmark-results.md`.

## Scope

The Rust runtime currently targets NVIDIA only and hooks `cudaMalloc`/`cudaFree`. Workloads that allocate through `cudaMallocAsync`, CUDA memory pools, managed memory, CUDA arrays, or direct driver allocation APIs may require additional hooks.
