#!/usr/bin/env python3
import argparse
import csv
import os
import pathlib
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import psutil


REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH_DIR = REPO / "rust" / "bench"
RESULTS_DIR = REPO / "rust" / "bench" / "results"
CUDA_CHECKPOINT = REPO / "cuda-checkpoint" / "bin" / "x86_64_Linux" / "cuda-checkpoint"


@dataclass
class ModelSpec:
    name: str
    model: str
    note: str = ""


DEFAULT_MODELS = [
    ModelSpec(
        "llama-8b",
        os.environ.get("GPUCR_LLAMA_MODEL", "NousResearch/Meta-Llama-3-8B"),
        "README uses a local/gated Llama-8B path; default is an accessible Llama-3 8B mirror.",
    ),
    ModelSpec("phi-4-mini-instruct", "microsoft/Phi-4-mini-instruct"),
    ModelSpec("pythia-1b", "EleutherAI/pythia-1b"),
    ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B"),
]


def run(args, *, env=None, timeout=900):
    start = time.perf_counter()
    completed = subprocess.run(
        args,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed after {elapsed:.3f}s: {' '.join(map(str, args))}\n{completed.stdout}"
        )
    return elapsed, completed.stdout


def nvidia_process_memory():
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    result = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        pid_s, mem_s = [part.strip() for part in line.split(",", 1)]
        result[int(pid_s)] = int(mem_s)
    return result


def descendant_pids(root_pid: int) -> set[int]:
    proc = psutil.Process(root_pid)
    return {root_pid, *(child.pid for child in proc.children(recursive=True))}


def select_gpu_pid(root_pid: int, timeout: float = 30.0) -> tuple[int, int]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pids = descendant_pids(root_pid)
        gpu = nvidia_process_memory()
        candidates = [(pid, gpu[pid]) for pid in pids if pid in gpu]
        if candidates:
            return max(candidates, key=lambda item: item[1])
        time.sleep(0.25)
    raise RuntimeError(f"no GPU process found for process tree rooted at {root_pid}")


def launch_worker(model: ModelSpec, impl: str, preload: pathlib.Path | None, args):
    gate = pathlib.Path("/tmp") / f"gpucr_vllm_gate_{impl}_{model.name}_{os.getpid()}"
    gate.unlink(missing_ok=True)
    log_path = RESULTS_DIR / f"{model.name}_{impl}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["GPU_VENDOR"] = "NVIDIA"
    env["VLLM_USE_V1"] = "1"
    if preload is not None:
        env["LD_PRELOAD"] = str(preload)
    cmd = [
        sys.executable,
        str(BENCH_DIR / "vllm_benchmark_worker.py"),
        "--model",
        model.model,
        "--gate",
        str(gate),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-tokens",
        str(args.max_tokens),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        bufsize=1,
    )
    lines: queue.Queue[str] = queue.Queue()
    ready = threading.Event()

    def reader():
        with log_path.open("w") as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                lines.put(line)
                if line.startswith("BENCH_READY"):
                    ready.set()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    if not ready.wait(timeout=args.load_timeout):
        terminate_process_group(proc)
        recent = []
        while not lines.empty():
            recent.append(lines.get_nowait())
        raise RuntimeError(f"{model.name}/{impl} did not become ready; see {log_path}\n{''.join(recent[-40:])}")
    return proc, gate, log_path


def terminate_process_group(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def wait_verify(proc, gate: pathlib.Path, timeout: float = 180.0) -> bool:
    gate.touch()
    deadline = time.time() + timeout
    log = RESULTS_DIR / "tmp-wait.log"
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            return rc == 0
        time.sleep(0.25)
    return False


def benchmark_gpucr(model, impl, preload, client, args):
    proc, gate, log_path = launch_worker(model, impl, preload, args)
    try:
        target_pid, mem_before = select_gpu_pid(proc.pid)
        ckpt_path = pathlib.Path("/mnt/huge-ckpt") / f"gpucr-bench-{impl}-{model.name}-{target_pid}"
        ckpt_path.unlink(missing_ok=True)

        if impl == "rust":
            run([client, "-i", "-p", str(target_pid), "-o", str(ckpt_path)], timeout=300)
        else:
            run([client, "-i", "-p", str(target_pid)], timeout=300)

        data_ckpt_s, data_ckpt_out = run([client, "-c", "-p", str(target_pid), "-b"], timeout=args.command_timeout)
        mem_after_data = nvidia_process_memory().get(target_pid, 0)
        control_ckpt_s, control_ckpt_out = run([CUDA_CHECKPOINT, "--toggle", "--pid", str(target_pid)], timeout=300)
        control_restore_s, control_restore_out = run([CUDA_CHECKPOINT, "--toggle", "--pid", str(target_pid)], timeout=300)
        data_restore_s, data_restore_out = run([client, "-r", "-p", str(target_pid), "-b"], timeout=args.command_timeout)
        verified = wait_verify(proc, gate, timeout=300)
        if not verified:
            raise RuntimeError(f"{model.name}/{impl} failed post-restore verification; see {log_path}")
        total = data_ckpt_s + control_ckpt_s + control_restore_s + data_restore_s
        return {
            "model_name": model.name,
            "model_id": model.model,
            "model_note": model.note,
            "implementation": impl,
            "target_pid": target_pid,
            "memory_before_mib": mem_before,
            "memory_after_data_ckpt_mib": mem_after_data,
            "data_checkpoint_s": data_ckpt_s,
            "control_checkpoint_s": control_ckpt_s,
            "control_restore_s": control_restore_s,
            "data_restore_s": data_restore_s,
            "total_checkpoint_restore_s": total,
            "verified": True,
            "log_path": str(log_path),
        }
    finally:
        terminate_process_group(proc)
        gate.unlink(missing_ok=True)


def benchmark_cuda(model, args):
    proc, gate, log_path = launch_worker(model, "cuda-checkpoint", None, args)
    try:
        target_pid, mem_before = select_gpu_pid(proc.pid)
        control_ckpt_s, _ = run([CUDA_CHECKPOINT, "--toggle", "--pid", str(target_pid)], timeout=300)
        mem_after = nvidia_process_memory().get(target_pid, 0)
        control_restore_s, _ = run([CUDA_CHECKPOINT, "--toggle", "--pid", str(target_pid)], timeout=300)
        verified = wait_verify(proc, gate, timeout=300)
        if not verified:
            raise RuntimeError(f"{model.name}/cuda-checkpoint failed post-restore verification; see {log_path}")
        return {
            "model_name": model.name,
            "model_id": model.model,
            "model_note": model.note,
            "implementation": "cuda-checkpoint",
            "target_pid": target_pid,
            "memory_before_mib": mem_before,
            "memory_after_data_ckpt_mib": mem_after,
            "data_checkpoint_s": "",
            "control_checkpoint_s": control_ckpt_s,
            "control_restore_s": control_restore_s,
            "data_restore_s": "",
            "total_checkpoint_restore_s": control_ckpt_s + control_restore_s,
            "verified": True,
            "log_path": str(log_path),
        }
    finally:
        terminate_process_group(proc)
        gate.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(model.name for model in DEFAULT_MODELS))
    parser.add_argument("--implementations", default="rust,cpp,cuda-checkpoint")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--load-timeout", type=float, default=1800)
    parser.add_argument("--command-timeout", type=float, default=1800)
    parser.add_argument("--output", default=str(RESULTS_DIR / "vllm_benchmark_results.csv"))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected_models = [model for model in DEFAULT_MODELS if model.name in set(args.models.split(","))]
    selected_impls = args.implementations.split(",")

    rust_preload = REPO / "rust" / "target" / "release" / "libgpucr_preload.so"
    rust_client = REPO / "rust" / "target" / "release" / "gpucr-client"
    cpp_preload = REPO / "build-nvidia-bench" / "vGPU-NVIDIA.so"
    cpp_client = REPO / "build-nvidia-bench" / "cr_client"

    rows = []
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for model in selected_models:
        for impl in selected_impls:
            try:
                if impl == "rust":
                    row = benchmark_gpucr(model, "rust", rust_preload, rust_client, args)
                elif impl == "cpp":
                    row = benchmark_gpucr(model, "cpp", cpp_preload, cpp_client, args)
                elif impl == "cuda-checkpoint":
                    row = benchmark_cuda(model, args)
                else:
                    raise ValueError(f"unknown implementation: {impl}")
            except Exception as exc:
                row = {
                    "model_name": model.name,
                    "model_id": model.model,
                    "model_note": model.note,
                    "implementation": impl,
                    "target_pid": "",
                    "memory_before_mib": "",
                    "memory_after_data_ckpt_mib": "",
                    "data_checkpoint_s": "",
                    "control_checkpoint_s": "",
                    "control_restore_s": "",
                    "data_restore_s": "",
                    "total_checkpoint_restore_s": "",
                    "verified": False,
                    "log_path": str(RESULTS_DIR / f"{model.name}_{impl}.log"),
                    "error": str(exc),
                }
                print(row, flush=True)
            rows.append(row)
            with output_path.open("w", newline="") as file:
                fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(row, flush=True)


if __name__ == "__main__":
    main()
