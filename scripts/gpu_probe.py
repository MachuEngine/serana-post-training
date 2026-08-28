#!/usr/bin/env python3
"""P0 done-criterion (CLAUDE.md build order): run this on the real L4
before trusting any VRAM budget in config/base.yaml. Reports device
properties, available VRAM, and a matmul throughput sanity check.

Targets the GCE GPU VM, not the local M5 (no CUDA there). `torch` is
intentionally not a project dependency yet (see pyproject.toml) --
`uv add torch` on the VM before running this.
"""

from __future__ import annotations

import time

try:
    import torch
except ImportError:
    raise SystemExit(
        "torch is not installed. This script targets the GPU VM -- "
        "`uv add torch` there before running it."
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible -- run this on the GCE L4 VM.")

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_vram_gb = props.total_memory / 1024**3
    print(f"device: {props.name}")
    print(f"total VRAM: {total_vram_gb:.1f} GB")
    print(f"compute capability: {props.major}.{props.minor}")

    torch.cuda.reset_peak_memory_stats()
    n = 4096
    iters = 20
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tflops = (2 * n**3 * iters) / elapsed / 1e12
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"matmul sanity check: {tflops:.1f} TFLOPS (bf16, {n}x{n}, {iters} iters)")
    print(f"peak VRAM during probe: {peak_vram_gb:.2f} GB")


if __name__ == "__main__":
    main()
