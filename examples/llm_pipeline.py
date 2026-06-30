#!/usr/bin/env python3
"""
Tensor Fabric — End-to-End LLM Inference Pipeline Demo

Demonstrates the complete flow:
  Storage Fabric → Network Fabric → Inference Fabric → Response

Run: python examples/llm_pipeline.py --model llama3-8b --requests 20
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click
import structlog

from tensor_fabric.control_plane.api import _state_manager, _routing_engine
from tensor_fabric.inference.inference_mesh import InferenceMesh, InferenceMeshConfig, MeshRequest

log = structlog.get_logger()


DEMO_CONVERSATIONS = [
    {
        "sequence_id": "conv-001",
        "turns": [
            [{"role": "user", "content": "What is CUDA and how does it enable GPU computing?"}],
            [{"role": "user", "content": "How does NVLink differ from PCIe for GPU interconnects?"}],
            [{"role": "user", "content": "Explain how KV-cache works in transformer inference."}],
        ],
    },
    {
        "sequence_id": "conv-002",
        "turns": [
            [{"role": "user", "content": "What is GPUDirect Storage and when should I use it?"}],
            [{"role": "user", "content": "Compare Triton Inference Server vs vLLM for production serving."}],
        ],
    },
    {
        "sequence_id": "conv-003",
        "turns": [
            [{"role": "user", "content": "Design a high-throughput inference system for 1000 req/s."}],
        ],
    },
]


async def run_pipeline(
    model_id: str,
    num_requests: int,
    nim_url: str,
    num_gpus: int,
    show_stats: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Tensor Fabric — LLM Inference Pipeline Demo")
    print(f"  Model: {model_id} | GPUs: {num_gpus} | NIM: {nim_url}")
    print(f"{'='*60}\n")

    await _state_manager.start()
    await asyncio.sleep(0.2)

    config = InferenceMeshConfig(
        nim_base_url=nim_url,
        num_gpus=num_gpus,
        kv_cache_capacity_per_gpu_mb=8192,
        enable_kv_sharing=True,
        enable_predictive_prefetch=True,
    )
    mesh = InferenceMesh(_state_manager, config)
    await mesh.start()

    print("Infrastructure initialized:")
    summary = _state_manager.get_cluster_summary()
    print(f"  GPUs online:    {summary['gpu_count']}")
    print(f"  Total VRAM:     {summary['total_vram_gb']} GB")
    print(f"  NVSwitch:       {'yes' if summary['nvswitch_present'] else 'no'}")
    print(f"  KV-cache:       enabled ({config.kv_cache_capacity_per_gpu_mb/1024:.0f} GB/GPU)")
    print()

    _state_manager.register_model(model_id, 0)

    requests_sent = 0
    total_latency = 0.0
    kv_hits = 0
    results = []

    conversation_pool = DEMO_CONVERSATIONS * (num_requests // len(DEMO_CONVERSATIONS) + 1)

    for i in range(num_requests):
        conv = conversation_pool[i % len(conversation_pool)]
        turn_idx = i % len(conv["turns"])
        messages = conv["turns"][turn_idx]

        request = MeshRequest(
            request_id=str(uuid.uuid4()),
            model_id=model_id,
            messages=messages,
            max_new_tokens=256,
            temperature=0.7,
            sequence_id=conv["sequence_id"],
        )

        t0 = time.perf_counter()
        response = await mesh.handle(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        total_latency += elapsed_ms
        requests_sent += 1
        if response.kv_cache_hit:
            kv_hits += 1

        status = "[KV-HIT]" if response.kv_cache_hit else "       "
        print(
            f"  [{i+1:3d}/{num_requests}] {status} "
            f"GPU-{response.target_gpu} | "
            f"{response.routing_strategy:20s} | "
            f"{elapsed_ms:6.1f}ms | "
            f"{response.content[:50]}..."
        )

        results.append(response)
        await asyncio.sleep(0.01)

    print(f"\n{'='*60}")
    print(f"  Results ({requests_sent} requests)")
    print(f"{'='*60}")
    print(f"  Avg latency:      {total_latency/max(requests_sent,1):.1f} ms")
    print(f"  KV-cache hits:    {kv_hits}/{requests_sent} ({100*kv_hits/max(requests_sent,1):.0f}%)")
    print(f"  Throughput:       {requests_sent/(total_latency/1000):.1f} req/s")

    if show_stats:
        stats = mesh.stats
        print(f"\n  Inference Mesh Stats:")
        print(f"    KV hit rate:    {stats['kv_hit_rate']*100:.1f}%")
        print(f"    Batch stats:    {stats['batch_aggregator']}")
        print(f"    Router stats:   {stats['tensor_router']}")
        print(f"    GPUDirect:      {stats['gpudirect']}")

    await mesh.stop()
    await _state_manager.stop()


@click.command()
@click.option("--model", default="llama3-8b", help="Model ID to route")
@click.option("--requests", default=20, help="Number of requests to send")
@click.option("--nim-url", default="http://localhost:8000", help="NIM endpoint URL")
@click.option("--gpus", default=1, help="Number of simulated GPUs")
@click.option("--stats", is_flag=True, help="Show detailed stats")
def main(model: str, requests: int, nim_url: str, gpus: int, stats: bool) -> None:
    """Run the Tensor Fabric LLM inference pipeline demo."""
    asyncio.run(run_pipeline(model, requests, nim_url, gpus, stats))


if __name__ == "__main__":
    main()
