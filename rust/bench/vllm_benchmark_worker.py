#!/usr/bin/env python3
import argparse
import os
import pathlib
import time

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--prompt", default="Explain GPU checkpoint restore in one short sentence.")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    gate = pathlib.Path(args.gate)
    if gate.exists():
        gate.unlink()

    print(f"BENCH_PARENT_PID {os.getpid()}", flush=True)
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    warmup = llm.generate([args.prompt], sampling)[0].outputs[0].text
    if not warmup:
        raise RuntimeError("warmup generation returned empty output")

    print(f"BENCH_READY {os.getpid()}", flush=True)
    while not gate.exists():
        time.sleep(0.1)

    restored = llm.generate([args.prompt], sampling)[0].outputs[0].text
    if not restored:
        raise RuntimeError("post-restore generation returned empty output")
    print("BENCH_VERIFY_OK", flush=True)


if __name__ == "__main__":
    main()
