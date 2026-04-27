#!/usr/bin/env python3
"""Benchmark torch_fpsample CPU vs CUDA backends on generated point clouds."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Iterable

import torch
import torch_fpsample


DEFAULT_SIZES = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32)


def make_batched_shuffled(base: torch.Tensor, batch: int, seed: int) -> torch.Tensor:
    """Replicate a cloud by batch, shuffling point order independently per batch.

    `torch_fpsample.sample(..., start_idx=0)` starts from the first point in each
    batch. Independent per-batch permutations make those start points different
    while preserving the same point distribution for every batch.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    clouds = []
    n = base.shape[0]
    for b in range(batch):
        perm = torch.randperm(n, generator=generator)
        clouds.append(base[perm])
    return torch.stack(clouds, dim=0).contiguous()


def ms_avg(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


@torch.no_grad()
def run_cpu(x: torch.Tensor, k: int, h: int, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        torch_fpsample.sample(x, k, h=h, start_idx=0)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        torch_fpsample.sample(x, k, h=h, start_idx=0)
        times.append((time.perf_counter() - t0) * 1000.0)
    return ms_avg(times)


@torch.no_grad()
def run_gpu(x: torch.Tensor, k: int, h: int, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        torch_fpsample.sample(x, k, h=h, start_idx=0)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        torch_fpsample.sample(x, k, h=h, start_idx=0)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return ms_avg(times)


def print_table(rows: list[dict[str, float | int]]) -> None:
    headers = [
        "N",
        "K(25%)",
        "Batch",
        "CPU avg ms",
        "GPU avg ms",
        "Speedup",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join([" ---: " for _ in headers]) + "|")
    for row in rows:
        print(
            "| "
            f"{int(row['n']):,} | "
            f"{int(row['k']):,} | "
            f"{int(row['batch'])} | "
            f"{row['cpu_ms']:.3f} | "
            f"{row['gpu_ms']:.3f} | "
            f"{row['speedup']:.2f}x |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark_data"))
    parser.add_argument("--sizes", type=int, nargs="*", default=list(DEFAULT_SIZES))
    parser.add_argument("--batches", type=int, nargs="*", default=list(DEFAULT_BATCHES))
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--shuffle-seed", type=int, default=20260427)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu-only", action="store_true")
    args = parser.parse_args()

    if args.cpu_only and args.gpu_only:
        raise ValueError("--cpu-only and --gpu-only cannot both be set")
    if not torch.cuda.is_available() and not args.cpu_only:
        raise RuntimeError("CUDA is not available")

    print(f"torch: {torch.__version__}")
    print(f"cpu_threads: {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(
        f"height={args.height}, sample_rate=25%, warmup={args.warmup}, repeat={args.repeat}"
    )
    print()

    rows: list[dict[str, float | int]] = []
    for n in args.sizes:
        path = args.data_dir / f"pointcloud_{n}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run scripts/generate_pointclouds.py first")
        base = torch.load(path, map_location="cpu").to(torch.float32).contiguous()
        if base.shape != (n, 3):
            raise ValueError(f"{path} has unexpected shape {tuple(base.shape)}")
        k = n // 4

        for batch in args.batches:
            x_cpu = make_batched_shuffled(base, batch, args.shuffle_seed + n * 1000 + batch)
            x_gpu = None if args.cpu_only else x_cpu.cuda(non_blocking=False)
            torch.cuda.synchronize() if x_gpu is not None else None

            cpu_ms = float("nan") if args.gpu_only else run_cpu(
                x_cpu, k, args.height, args.warmup, args.repeat
            )
            gpu_ms = float("nan") if args.cpu_only else run_gpu(
                x_gpu, k, args.height, args.warmup, args.repeat
            )
            speedup = cpu_ms / gpu_ms if not args.cpu_only and not args.gpu_only else float("nan")
            rows.append(
                {
                    "n": n,
                    "k": k,
                    "batch": batch,
                    "cpu_ms": cpu_ms,
                    "gpu_ms": gpu_ms,
                    "speedup": speedup,
                }
            )
            print_table(rows[-1:])
            print()

            del x_cpu, x_gpu
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Final summary:")
    print_table(rows)


if __name__ == "__main__":
    main()
