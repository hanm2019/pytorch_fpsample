#!/usr/bin/env python3
"""Generate deterministic point cloud tensors for FPS benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_SIZES = (4096, 8192, 16384, 32768, 65536, 131072, 262144)


def make_cloud(n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + n)

    # A structured cloud is more representative than pure white noise: several
    # Gaussian clusters plus a small amount of uniform background.
    num_clusters = 8
    centers = torch.randn(num_clusters, 3, generator=generator) * 25.0
    cluster_id = torch.randint(num_clusters, (n,), generator=generator)
    local = torch.randn(n, 3, generator=generator) * torch.tensor([4.0, 2.0, 0.8])
    cloud = centers[cluster_id] + local

    background_count = max(1, n // 20)
    background = (torch.rand(background_count, 3, generator=generator) - 0.5) * 120.0
    cloud[:background_count] = background
    return cloud.contiguous().to(torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_data"))
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--sizes", type=int, nargs="*", default=list(DEFAULT_SIZES))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in args.sizes:
        cloud = make_cloud(n, args.seed)
        path = args.out_dir / f"pointcloud_{n}.pt"
        torch.save(cloud, path)
        print(f"generated {path} shape={tuple(cloud.shape)}")


if __name__ == "__main__":
    main()
