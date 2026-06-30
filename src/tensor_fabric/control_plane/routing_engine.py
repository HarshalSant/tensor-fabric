from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from tensor_fabric.common.tensor_descriptor import TensorDescriptor, TensorRole
from tensor_fabric.control_plane.gpu_state_manager import GPUStateManager

log = structlog.get_logger(__name__)


class RouteStrategy(str, Enum):
    VRAM_AFFINITY = "vram_affinity"      # Route to GPU that already has the model
    LEAST_LOADED = "least_loaded"         # Route to lowest-utilization GPU
    NVLINK_LOCALITY = "nvlink_locality"   # Route to NVLink peer of current GPU
    CACHE_HIT = "cache_hit"              # Route to GPU that has the KV-cache
    PREDICTIVE = "predictive"            # Pre-fetch based on access pattern


@dataclass
class RouteDecision:
    tensor_id: str
    target_gpu: int
    strategy: RouteStrategy
    confidence: float
    estimated_latency_us: float
    cache_hit: bool = False
    prefetch_triggered: bool = False
    reason: str = ""
    decided_at: float = field(default_factory=time.monotonic)


@dataclass
class PrefetchJob:
    tensor_descriptor: TensorDescriptor
    target_gpu: int
    triggered_at: float = field(default_factory=time.monotonic)
    completed: bool = False


class AccessPatternPredictor:
    """
    Tracks tensor access sequences to predict what will be needed next.
    Enables the predictive pre-fetch that is only possible when
    Storage, Network, and Inference layers share a control plane.
    """

    def __init__(self, window: int = 100) -> None:
        self._history: list[str] = []
        self._window = window
        self._ngrams: dict[str, dict[str, int]] = {}

    def record(self, cache_key: str) -> None:
        self._history.append(cache_key)
        if len(self._history) > self._window:
            self._history.pop(0)

        if len(self._history) >= 2:
            prev = self._history[-2]
            if prev not in self._ngrams:
                self._ngrams[prev] = {}
            self._ngrams[prev][cache_key] = self._ngrams[prev].get(cache_key, 0) + 1

    def predict_next(self, cache_key: str) -> str | None:
        successors = self._ngrams.get(cache_key, {})
        if not successors:
            return None
        return max(successors, key=successors.get)

    def confidence(self, cache_key: str) -> float:
        successors = self._ngrams.get(cache_key, {})
        if not successors:
            return 0.0
        total = sum(successors.values())
        best = max(successors.values())
        return best / total if total > 0 else 0.0


class RoutingEngine:
    """
    The routing brain of Tensor Fabric.

    Makes holistic routing decisions by combining:
    - Live GPU VRAM state (from GPUStateManager)
    - KV-cache hit state (from InferenceFabric's KVCacheManager)
    - Access pattern predictions (from AccessPatternPredictor)
    - NVLink topology (from GPUTopology)

    The key insight: routing decisions that span Storage, Network,
    and Inference simultaneously produce emergent optimizations
    impossible with per-layer routing.
    """

    PREFETCH_CONFIDENCE_THRESHOLD = 0.65
    MAX_PREFETCH_QUEUE = 32

    def __init__(self, state_manager: GPUStateManager) -> None:
        self._state = state_manager
        self._predictor = AccessPatternPredictor()
        self._prefetch_queue: list[PrefetchJob] = []
        self._prefetch_callbacks: list[Any] = []
        self._route_count = 0
        self._cache_hit_count = 0
        self._prefetch_hit_count = 0

    def register_prefetch_handler(self, fn) -> None:
        self._prefetch_callbacks.append(fn)

    def route(
        self,
        descriptor: TensorDescriptor,
        kv_cache_gpu: int | None = None,
        current_gpu: int | None = None,
    ) -> RouteDecision:
        self._route_count += 1
        self._predictor.record(descriptor.cache_key)

        # Loop 1: Check KV-cache hit — highest priority
        if kv_cache_gpu is not None:
            self._cache_hit_count += 1
            return self._route_cache_hit(descriptor, kv_cache_gpu)

        # Loop 2: Route model weights to GPU that already has the model
        if descriptor.role == TensorRole.MODEL_WEIGHT:
            gpus = self._state.gpus_with_model(descriptor.model_id)
            if gpus:
                return self._route_affinity(descriptor, gpus, current_gpu)

        # Loop 3: NVLink locality — route to peer of current GPU if bandwidth favors it
        topology = self._state.snapshot()
        if current_gpu is not None and topology:
            peers = topology.get_nvlink_peers(current_gpu)
            if peers:
                best_peer = self._best_nvlink_peer(peers, topology)
                if best_peer is not None:
                    return RouteDecision(
                        tensor_id=descriptor.tensor_id,
                        target_gpu=best_peer,
                        strategy=RouteStrategy.NVLINK_LOCALITY,
                        confidence=0.85,
                        estimated_latency_us=1.5,
                        reason=f"NVLink peer of GPU {current_gpu}",
                    )

        # Loop 4: Least-loaded GPU fallback
        target = self._least_loaded_healthy_gpu(
            required_vram_mb=descriptor.nbytes_mb * 1.2
        )
        if target is None:
            target = 0

        decision = RouteDecision(
            tensor_id=descriptor.tensor_id,
            target_gpu=target,
            strategy=RouteStrategy.LEAST_LOADED,
            confidence=0.70,
            estimated_latency_us=45.0,
            reason="least-loaded fallback",
        )

        # Trigger predictive prefetch as side effect
        self._maybe_prefetch(descriptor, target)

        return decision

    def _route_cache_hit(self, descriptor: TensorDescriptor, gpu: int) -> RouteDecision:
        return RouteDecision(
            tensor_id=descriptor.tensor_id,
            target_gpu=gpu,
            strategy=RouteStrategy.CACHE_HIT,
            confidence=1.0,
            estimated_latency_us=0.8,
            cache_hit=True,
            reason=f"KV-cache hit on GPU {gpu}",
        )

    def _route_affinity(
        self,
        descriptor: TensorDescriptor,
        candidate_gpus: list[int],
        current_gpu: int | None,
    ) -> RouteDecision:
        topology = self._state.snapshot()
        if topology is None:
            return RouteDecision(
                tensor_id=descriptor.tensor_id,
                target_gpu=candidate_gpus[0],
                strategy=RouteStrategy.VRAM_AFFINITY,
                confidence=0.90,
                estimated_latency_us=2.0,
            )

        best = min(
            (topology.nodes[g] for g in candidate_gpus if g in topology.nodes),
            key=lambda n: n.compute_util_pct,
            default=None,
        )
        target = best.device_id if best else candidate_gpus[0]

        return RouteDecision(
            tensor_id=descriptor.tensor_id,
            target_gpu=target,
            strategy=RouteStrategy.VRAM_AFFINITY,
            confidence=0.95,
            estimated_latency_us=1.2,
            reason=f"model already loaded on GPU {target}",
        )

    def _best_nvlink_peer(self, peers: list[int], topology) -> int | None:
        candidates = [
            topology.nodes[p] for p in peers
            if p in topology.nodes and topology.nodes[p].is_healthy
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.compute_util_pct).device_id

    def _least_loaded_healthy_gpu(self, required_vram_mb: float = 0) -> int | None:
        topology = self._state.snapshot()
        if topology is None:
            return None

        candidates = [
            node for node in topology.nodes.values()
            if node.is_healthy and node.vram_free_mb >= required_vram_mb
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.compute_util_pct).device_id

    def _maybe_prefetch(self, descriptor: TensorDescriptor, routed_to: int) -> None:
        if len(self._prefetch_queue) >= self.MAX_PREFETCH_QUEUE:
            return

        next_key = self._predictor.predict_next(descriptor.cache_key)
        if next_key is None:
            return

        confidence = self._predictor.confidence(descriptor.cache_key)
        if confidence < self.PREFETCH_CONFIDENCE_THRESHOLD:
            return

        log.debug(
            "routing_engine.prefetch_triggered",
            next_key=next_key,
            confidence=round(confidence, 3),
            target_gpu=routed_to,
        )

        job = PrefetchJob(tensor_descriptor=descriptor, target_gpu=routed_to)
        self._prefetch_queue.append(job)

        for fn in self._prefetch_callbacks:
            try:
                fn(job)
            except Exception:
                pass

    @property
    def stats(self) -> dict:
        total = max(self._route_count, 1)
        return {
            "total_routes": self._route_count,
            "cache_hit_rate": round(self._cache_hit_count / total, 4),
            "prefetch_hit_rate": round(self._prefetch_hit_count / total, 4),
            "prefetch_queue_depth": len(self._prefetch_queue),
        }
