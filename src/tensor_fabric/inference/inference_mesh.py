from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog

from tensor_fabric.common.tensor_descriptor import TensorDescriptor, TensorDtype, TensorRole
from tensor_fabric.control_plane.gpu_state_manager import GPUStateManager
from tensor_fabric.control_plane.routing_engine import RoutingEngine
from tensor_fabric.inference.batch_aggregator import (
    BatchAggregator,
    BatchAggregatorConfig,
    InferenceRequest,
)
from tensor_fabric.inference.kv_cache_manager import KVCacheManager
from tensor_fabric.inference.nim_client import NIMClient, NIMCompletionRequest, NIMCompletionResponse
from tensor_fabric.network.tensor_router import TensorRouter
from tensor_fabric.storage.gpudirect_bridge import GPUDirectBridge

log = structlog.get_logger(__name__)


@dataclass
class InferenceMeshConfig:
    nim_base_url: str = "http://localhost:8000"
    nim_api_key: str = "not-needed-for-local"
    kv_cache_capacity_per_gpu_mb: float = 16384.0
    max_batch_size: int = 32
    max_batch_wait_ms: float = 5.0
    enable_continuous_batching: bool = True
    enable_kv_sharing: bool = True
    enable_predictive_prefetch: bool = True
    num_gpus: int = 1


@dataclass
class MeshRequest:
    """A single request flowing through the Inference Mesh."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    messages: list[dict] = field(default_factory=list)
    max_new_tokens: int = 512
    temperature: float = 0.7
    sequence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    arrived_at: float = field(default_factory=time.monotonic)


@dataclass
class MeshResponse:
    request_id: str
    content: str
    model_id: str
    target_gpu: int
    routing_strategy: str
    kv_cache_hit: bool
    time_to_first_token_ms: float
    total_latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    batch_size: int = 1
    batch_efficiency_pct: float = 100.0


class InferenceMesh:
    """
    The Inference Fabric — GPU-state-aware service mesh for AI.

    This is what runs on the BlueField DPU in production, intercepting
    inference requests before they ever reach a CPU process on the host.

    The Inference Mesh ties together all three Tensor Fabric layers:

    1. STORAGE: Knows where model weights and KV-caches live on NVMe
    2. NETWORK: Routes tensors across GPUs with optimal bandwidth
    3. INFERENCE: Makes final routing decision based on live GPU state

    Request flow through the mesh:
    ┌─────────────────────────────────────────────────────────────┐
    │  Client Request                                             │
    │       ↓                                                     │
    │  [KV-Cache Lookup] ← Check if sequence already in VRAM     │
    │       ↓                                                     │
    │  [Routing Engine] ← Select target GPU (affinity/load/cache)│
    │       ↓                                                     │
    │  [Batch Aggregator] ← Group with compatible requests       │
    │       ↓                                                     │
    │  [NIM Dispatch] ← Send batched request to NIM endpoint     │
    │       ↓                                                     │
    │  [KV-Cache Store] ← Cache new KV tensors for next turn     │
    │       ↓                                                     │
    │  Response to Client                                         │
    └─────────────────────────────────────────────────────────────┘

    The key innovation: KV-Cache Lookup happens BEFORE routing.
    If we find the cache, we route TO that GPU — not the other way around.
    This is the inversion that makes cross-pod KV sharing possible.
    """

    def __init__(
        self,
        state_manager: GPUStateManager,
        config: InferenceMeshConfig | None = None,
    ) -> None:
        self._config = config or InferenceMeshConfig()
        self._state = state_manager
        self._routing_engine = RoutingEngine(state_manager)

        self._kv_cache = KVCacheManager(
            capacity_per_gpu_mb=self._config.kv_cache_capacity_per_gpu_mb
        )
        self._batch_aggregator = BatchAggregator(
            config=BatchAggregatorConfig(
                max_batch_size=self._config.max_batch_size,
                max_wait_ms=self._config.max_batch_wait_ms,
                enable_continuous_batching=self._config.enable_continuous_batching,
            )
        )
        self._tensor_router = TensorRouter()
        self._gpudirect = GPUDirectBridge()
        self._nim = NIMClient(
            base_url=self._config.nim_base_url,
            api_key=self._config.nim_api_key,
        )

        self._request_count = 0
        self._kv_hit_count = 0
        self._prefetch_count = 0
        self._running = False

        self._routing_engine.register_prefetch_handler(self._handle_prefetch)

    async def start(self) -> None:
        for i in range(self._config.num_gpus):
            self._kv_cache.add_gpu(i)
            self._state.register_listener(self._on_gpu_event)

        await self._batch_aggregator.start()
        self._running = True
        log.info(
            "inference_mesh.started",
            gpus=self._config.num_gpus,
            kv_sharing=self._config.enable_kv_sharing,
            predictive_prefetch=self._config.enable_predictive_prefetch,
        )

    async def stop(self) -> None:
        await self._batch_aggregator.stop()
        self._running = False
        log.info("inference_mesh.stopped")

    async def handle(self, request: MeshRequest) -> MeshResponse:
        t0 = time.monotonic()
        self._request_count += 1

        # ── Step 1: KV-cache lookup (before routing) ───────────────────────
        kv_gpu: int | None = None
        kv_hit = False

        if self._config.enable_kv_sharing:
            kv_gpu = await self._kv_cache.lookup_sequence_gpu(
                request.sequence_id, request.model_id
            )
            if kv_gpu is not None:
                kv_hit = True
                self._kv_hit_count += 1
                log.info(
                    "inference_mesh.kv_hit",
                    sequence_id=request.sequence_id,
                    gpu=kv_gpu,
                )

        # ── Step 2: Route to target GPU ────────────────────────────────────
        descriptor = TensorDescriptor(
            tensor_id=request.request_id,
            model_id=request.model_id,
            role=TensorRole.INPUT,
            shape=(1, len(request.messages[-1].get("content", "").split()) if request.messages else 1),
            dtype=TensorDtype.INT8,
        )

        decision = self._routing_engine.route(
            descriptor=descriptor,
            kv_cache_gpu=kv_gpu,
        )
        target_gpu = decision.target_gpu
        routing_strategy = decision.strategy.value

        log.info(
            "inference_mesh.route",
            request_id=request.request_id,
            model=request.model_id,
            target_gpu=target_gpu,
            strategy=routing_strategy,
            kv_hit=kv_hit,
        )

        # ── Step 3: Dispatch to NIM ────────────────────────────────────────
        ttft = time.monotonic()
        nim_request = NIMCompletionRequest(
            model=request.model_id,
            messages=request.messages,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            request_id=request.request_id,
        )

        endpoint = self._nim.get_endpoint(request.model_id)
        nim_response = await self._dispatch_to_nim(nim_request, endpoint, target_gpu)

        time_to_first_token_ms = (time.monotonic() - ttft) * 1000
        total_latency_ms = (time.monotonic() - t0) * 1000

        # ── Step 4: Store KV-cache for future turns ────────────────────────
        if self._config.enable_kv_sharing and nim_response:
            asyncio.create_task(
                self._store_simulated_kv(request, target_gpu)
            )

        log.info(
            "inference_mesh.complete",
            request_id=request.request_id,
            total_ms=round(total_latency_ms, 1),
            ttft_ms=round(time_to_first_token_ms, 1),
            kv_hit=kv_hit,
            strategy=routing_strategy,
        )

        return MeshResponse(
            request_id=request.request_id,
            content=nim_response.content if nim_response else "[NIM unavailable — simulated response]",
            model_id=request.model_id,
            target_gpu=target_gpu,
            routing_strategy=routing_strategy,
            kv_cache_hit=kv_hit,
            time_to_first_token_ms=time_to_first_token_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=nim_response.input_tokens if nim_response else 0,
            output_tokens=nim_response.output_tokens if nim_response else 0,
        )

    async def handle_stream(self, request: MeshRequest) -> AsyncIterator[str]:
        kv_gpu = await self._kv_cache.lookup_sequence_gpu(
            request.sequence_id, request.model_id
        )
        descriptor = TensorDescriptor(
            tensor_id=request.request_id,
            model_id=request.model_id,
            role=TensorRole.INPUT,
            shape=(1, 1),
            dtype=TensorDtype.INT8,
        )
        decision = self._routing_engine.route(descriptor=descriptor, kv_cache_gpu=kv_gpu)

        nim_request = NIMCompletionRequest(
            model=request.model_id,
            messages=request.messages,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            stream=True,
        )
        endpoint = self._nim.get_endpoint(request.model_id)
        async with self._nim as client:
            async for chunk in client.stream(nim_request, endpoint_override=endpoint):
                yield chunk

    async def _dispatch_to_nim(
        self,
        request: NIMCompletionRequest,
        endpoint: str | None,
        target_gpu: int,
    ) -> NIMCompletionResponse | None:
        try:
            async with self._nim as client:
                return await client.complete(request, endpoint_override=endpoint)
        except Exception as exc:
            log.warning("inference_mesh.nim_error", error=str(exc))
            return NIMCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                content=f"[Simulated response for {request.model} — NIM not connected]",
                input_tokens=len(str(request.messages)),
                output_tokens=64,
                latency_ms=45.0,
            )

    async def _store_simulated_kv(self, request: MeshRequest, gpu: int) -> None:
        """Store a simulated KV-cache entry for demonstration."""
        try:
            import numpy as np
            seq_len = max(1, len(request.messages[-1].get("content", "")) // 4) if request.messages else 16
            num_heads, head_dim = 32, 128
            keys = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float16)
            values = np.random.randn(1, num_heads, seq_len, head_dim).astype(np.float16)

            try:
                import cupy as cp
                with cp.cuda.Device(gpu):
                    keys = cp.asarray(keys)
                    values = cp.asarray(values)
            except Exception:
                pass

            await self._kv_cache.store_kv(
                sequence_id=request.sequence_id,
                model_id=request.model_id,
                layer_index=0,
                keys=keys,
                values=values,
                gpu_device=gpu,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        except Exception as exc:
            log.debug("kv_store.error", error=str(exc))

    def _handle_prefetch(self, job) -> None:
        self._prefetch_count += 1
        log.debug("inference_mesh.prefetch", target_gpu=job.target_gpu)

    def _on_gpu_event(self, event) -> None:
        if event.event_type == "vram_low":
            log.warning(
                "inference_mesh.vram_pressure",
                device=event.device_id,
                free_mb=event.payload.get("vram_free_mb"),
            )
            asyncio.create_task(
                self._kv_cache.evict_sequence("__oldest__")
            )

    @property
    def stats(self) -> dict:
        return {
            "requests": self._request_count,
            "kv_hits": self._kv_hit_count,
            "kv_hit_rate": round(self._kv_hit_count / max(self._request_count, 1), 4),
            "prefetch_count": self._prefetch_count,
            "kv_cache": self._kv_cache.stats,
            "batch_aggregator": self._batch_aggregator.stats,
            "tensor_router": self._tensor_router.stats,
            "gpudirect": self._gpudirect.stats,
        }
