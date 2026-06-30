#!/usr/bin/env python3
"""
Tensor Fabric End-to-End Benchmark

Measures performance across all three layers vs a traditional stack.
Run: python benchmarks/e2e_benchmark.py --compare-baseline
"""
from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import click
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


@dataclass
class BenchmarkResult:
    name: str
    latencies_ms: list[float]
    throughput_gbs: float = 0.0
    notes: str = ""

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def p95_ms(self) -> float:
        sorted_l = sorted(self.latencies_ms)
        return sorted_l[int(len(sorted_l) * 0.95)]

    @property
    def p99_ms(self) -> float:
        sorted_l = sorted(self.latencies_ms)
        return sorted_l[int(len(sorted_l) * 0.99)]

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms)


# ── Storage Fabric Benchmarks ─────────────────────────────────────────────────

async def bench_storage_baseline(tensor_size_mb: float, iterations: int) -> BenchmarkResult:
    """Traditional: NVMe → CPU RAM → GPU VRAM (2 copies, CPU involved)."""
    import tempfile, os

    size_bytes = int(tensor_size_mb * 1024 * 1024)
    data = np.random.randn(size_bytes // 4).astype(np.float32)
    latencies = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(data.tobytes())
        tmp_path = f.name

    try:
        for _ in range(iterations):
            t0 = time.perf_counter()
            raw = open(tmp_path, "rb").read()
            arr = np.frombuffer(raw, dtype=np.float32)
            if CUPY_AVAILABLE:
                gpu_arr = cp.asarray(arr)
                cp.cuda.Stream.null.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
    finally:
        os.unlink(tmp_path)

    return BenchmarkResult(
        name="Storage (Baseline: CPU path)",
        latencies_ms=latencies,
        throughput_gbs=(tensor_size_mb / 1024) / (statistics.mean(latencies) / 1000),
        notes="NVMe → CPU RAM → cudaMemcpy → GPU VRAM",
    )


async def bench_storage_tensor_fabric(tensor_size_mb: float, iterations: int) -> BenchmarkResult:
    """Tensor Fabric: GPUDirect Storage path (simulated with host-pinned)."""
    import sys
    sys.path.insert(0, "src")

    from tensor_fabric.storage.gpudirect_bridge import GPUDirectBridge
    from tensor_fabric.common.tensor_descriptor import TensorDescriptor, TensorDtype, TensorRole
    import tempfile, os

    size_bytes = int(tensor_size_mb * 1024 * 1024)
    shape = (size_bytes // 2,)
    data = np.random.randn(*shape).astype(np.float16)
    latencies = []

    bridge = GPUDirectBridge()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tfs") as f:
        f.write(data.tobytes())
        tmp_path = f.name

    desc = TensorDescriptor(
        tensor_id=str(uuid.uuid4()),
        model_id="bench-model",
        role=TensorRole.MODEL_WEIGHT,
        shape=shape,
        dtype=TensorDtype.FLOAT16,
    )

    try:
        for _ in range(iterations):
            t0 = time.perf_counter()
            arr = await bridge.dma_to_gpu(tmp_path, desc, target_gpu=0, file_offset=0)
            if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
                cp.cuda.Stream.null.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
    finally:
        os.unlink(tmp_path)

    return BenchmarkResult(
        name=f"Storage (Tensor Fabric: {bridge._current_path_name()})",
        latencies_ms=latencies,
        throughput_gbs=(tensor_size_mb / 1024) / (statistics.mean(latencies) / 1000),
        notes="GPUDirect Storage path — NVMe DMA → GPU VRAM",
    )


# ── KV-Cache Benchmarks ───────────────────────────────────────────────────────

async def bench_kv_cache_miss(iterations: int) -> BenchmarkResult:
    """Baseline: KV-cache miss — full recompute every request."""
    latencies = []
    num_heads, head_dim, seq_len = 32, 128, 512

    for _ in range(iterations):
        t0 = time.perf_counter()
        if CUPY_AVAILABLE:
            keys = cp.random.randn(1, num_heads, seq_len, head_dim).astype(cp.float16)
            values = cp.random.randn(1, num_heads, seq_len, head_dim).astype(cp.float16)
            _ = cp.matmul(keys, values.transpose(0, 1, 3, 2)) / (head_dim ** 0.5)
            cp.cuda.Stream.null.synchronize()
        else:
            keys = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float16)
            values = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float16)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    return BenchmarkResult(
        name="KV-Cache (Baseline: miss/recompute)",
        latencies_ms=latencies,
        notes="No cross-pod sharing — every request recomputes attention",
    )


async def bench_kv_cache_hit(iterations: int) -> BenchmarkResult:
    """Tensor Fabric: Cross-pod KV-cache hit — return from GPU pool."""
    import sys
    sys.path.insert(0, "src")
    from tensor_fabric.inference.kv_cache_manager import KVCacheManager

    manager = KVCacheManager(capacity_per_gpu_mb=4096)
    manager.add_gpu(0)

    num_heads, head_dim, seq_len = 32, 128, 512
    sequence_id = str(uuid.uuid4())
    model_id = "llama3-8b"

    if CUPY_AVAILABLE:
        keys = cp.random.randn(1, num_heads, seq_len, head_dim).astype(cp.float16)
        values = cp.random.randn(1, num_heads, seq_len, head_dim).astype(cp.float16)
    else:
        keys = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float16)
        values = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float16)

    await manager.store_kv(sequence_id, model_id, 0, keys, values, 0, num_heads, head_dim)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = await manager.get_kv(sequence_id, model_id, 0)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    return BenchmarkResult(
        name="KV-Cache (Tensor Fabric: cross-pod hit)",
        latencies_ms=latencies,
        notes="KV tensors in shared GPU pool — zero recompute",
    )


# ── Routing Benchmarks ────────────────────────────────────────────────────────

async def bench_routing_baseline(iterations: int) -> BenchmarkResult:
    """Baseline: round-robin routing (no GPU state awareness)."""
    endpoints = [f"http://pod-{i}:8000" for i in range(8)]
    latencies = []
    counter = 0

    for _ in range(iterations):
        t0 = time.perf_counter()
        _endpoint = endpoints[counter % len(endpoints)]
        counter += 1
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    return BenchmarkResult(
        name="Routing (Baseline: round-robin)",
        latencies_ms=latencies,
        notes="No GPU state awareness — random distribution",
    )


async def bench_routing_tensor_fabric(iterations: int) -> BenchmarkResult:
    """Tensor Fabric: GPU-state-aware routing with KV-cache affinity."""
    import sys
    sys.path.insert(0, "src")
    from tensor_fabric.control_plane.gpu_state_manager import GPUStateManager
    from tensor_fabric.control_plane.routing_engine import RoutingEngine
    from tensor_fabric.common.tensor_descriptor import TensorDescriptor, TensorDtype, TensorRole

    manager = GPUStateManager()
    manager.initialize = lambda: None
    await manager.start()
    await asyncio.sleep(0.1)

    engine = RoutingEngine(manager)
    latencies = []

    for i in range(iterations):
        desc = TensorDescriptor(
            tensor_id=str(uuid.uuid4()),
            model_id="llama3-8b",
            role=TensorRole.INPUT,
            shape=(1, 512),
            dtype=TensorDtype.INT8,
        )
        t0 = time.perf_counter()
        _decision = engine.route(desc)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    await manager.stop()
    return BenchmarkResult(
        name="Routing (Tensor Fabric: GPU-aware)",
        latencies_ms=latencies,
        notes="pynvml + NVLink topology + access pattern prediction",
    )


# ── Main CLI ──────────────────────────────────────────────────────────────────

def print_results(results: list[tuple[BenchmarkResult, BenchmarkResult]], title: str) -> None:
    if RICH_AVAILABLE:
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Benchmark", style="bold")
        table.add_column("p50 (ms)", justify="right")
        table.add_column("p95 (ms)", justify="right")
        table.add_column("p99 (ms)", justify="right")
        table.add_column("Throughput (GB/s)", justify="right")
        table.add_column("Speedup", justify="right", style="bold green")

        for baseline, tf in results:
            speedup = baseline.p50_ms / max(tf.p50_ms, 0.001)
            table.add_row(
                "Baseline",
                f"{baseline.p50_ms:.2f}",
                f"{baseline.p95_ms:.2f}",
                f"{baseline.p99_ms:.2f}",
                f"{baseline.throughput_gbs:.2f}" if baseline.throughput_gbs else "—",
                "1.0x",
            )
            table.add_row(
                f"[green]{tf.name}[/green]",
                f"[green]{tf.p50_ms:.2f}[/green]",
                f"[green]{tf.p95_ms:.2f}[/green]",
                f"[green]{tf.p99_ms:.2f}[/green]",
                f"[green]{tf.throughput_gbs:.2f}[/green]" if tf.throughput_gbs else "—",
                f"[bold green]{speedup:.1f}x[/bold green]",
            )
            table.add_row("", "", "", "", "", "")

        console.print(table)
    else:
        print(f"\n{title}")
        print("-" * 80)
        for baseline, tf in results:
            speedup = baseline.p50_ms / max(tf.p50_ms, 0.001)
            print(f"Baseline: {baseline.name}")
            print(f"  p50={baseline.p50_ms:.2f}ms  p99={baseline.p99_ms:.2f}ms")
            print(f"Tensor Fabric: {tf.name}")
            print(f"  p50={tf.p50_ms:.2f}ms  p99={tf.p99_ms:.2f}ms  speedup={speedup:.1f}x")
            print()


@click.command()
@click.option("--iterations", default=100, help="Iterations per benchmark")
@click.option("--tensor-size-mb", default=256.0, help="Tensor size for storage benchmarks")
@click.option("--compare-baseline", is_flag=True, help="Show baseline vs Tensor Fabric comparison")
def main(iterations: int, tensor_size_mb: float, compare_baseline: bool) -> None:
    """Tensor Fabric End-to-End Performance Benchmark"""

    if RICH_AVAILABLE:
        console.rule("[bold cyan]Tensor Fabric Benchmark Suite[/bold cyan]")

    async def run():
        print("\n[1/3] Storage Fabric benchmarks...")
        storage_baseline = await bench_storage_baseline(tensor_size_mb, min(iterations, 20))
        storage_tf = await bench_storage_tensor_fabric(tensor_size_mb, min(iterations, 20))

        print("[2/3] KV-Cache benchmarks...")
        kv_miss = await bench_kv_cache_miss(iterations)
        kv_hit = await bench_kv_cache_hit(iterations)

        print("[3/3] Routing benchmarks...")
        routing_baseline = await bench_routing_baseline(iterations)
        routing_tf = await bench_routing_tensor_fabric(iterations)

        results = [
            (storage_baseline, storage_tf),
            (kv_miss, kv_hit),
            (routing_baseline, routing_tf),
        ]
        print_results(results, "Tensor Fabric vs Baseline")

        # Summary
        storage_speedup = storage_baseline.p50_ms / max(storage_tf.p50_ms, 0.001)
        kv_speedup = kv_miss.p50_ms / max(kv_hit.p50_ms, 0.001)
        routing_speedup = routing_baseline.p50_ms / max(routing_tf.p50_ms, 0.001)

        print(f"\nSummary:")
        print(f"  Storage: {storage_speedup:.1f}x faster  (GPUDirect path vs CPU memcpy)")
        print(f"  KV-Cache: {kv_speedup:.1f}x faster  (cross-pod hit vs recompute)")
        print(f"  Routing: {routing_speedup:.1f}x faster  (GPU-aware vs round-robin)")
        print(f"  GPU: {'CuPy GPU enabled' if CUPY_AVAILABLE else 'CPU simulation (install cupy-cuda12x for GPU)'}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
