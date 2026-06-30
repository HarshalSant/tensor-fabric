from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from tensor_fabric.control_plane.gpu_state_manager import get_state_manager
from tensor_fabric.control_plane.routing_engine import RoutingEngine
from tensor_fabric.common.tensor_descriptor import TensorDescriptor, TensorDtype, TensorRole

log = structlog.get_logger(__name__)

# ── Prometheus metrics ──────────────────────────────────────────────────────
ROUTE_REQUESTS = Counter("tf_route_requests_total", "Total routing requests", ["strategy"])
ROUTE_LATENCY = Histogram("tf_route_latency_seconds", "Routing decision latency")
GPU_FREE_VRAM = Gauge("tf_gpu_free_vram_mb", "Free VRAM per GPU", ["device_id"])
GPU_COMPUTE_UTIL = Gauge("tf_gpu_compute_util_pct", "Compute utilization per GPU", ["device_id"])
CACHE_HIT_RATE = Gauge("tf_kv_cache_hit_rate", "KV-cache hit rate (0–1)")


# ── Pydantic models ─────────────────────────────────────────────────────────
class RouteRequest(BaseModel):
    model_id: str
    tensor_role: str = "activation"
    shape: list[int]
    dtype: str = "float16"
    layer_index: int | None = None
    sequence_length: int | None = None
    batch_size: int = 1
    current_gpu: int | None = None
    kv_cache_gpu: int | None = None


class RouteResponse(BaseModel):
    tensor_id: str
    target_gpu: int
    strategy: str
    confidence: float
    estimated_latency_us: float
    cache_hit: bool
    prefetch_triggered: bool
    reason: str


class ModelRegistration(BaseModel):
    model_id: str
    device_id: int
    vram_mb: float = 0.0


class ClusterStatus(BaseModel):
    gpu_count: int
    total_vram_gb: float
    free_vram_gb: float
    vram_utilization_pct: float
    avg_compute_util_pct: float
    nvswitch_present: bool
    healthy_gpus: int
    loaded_models: dict[str, list[int]]
    uptime_seconds: float
    route_stats: dict[str, Any]


# ── Lifespan ─────────────────────────────────────────────────────────────────
_start_time = time.monotonic()
_state_manager = get_state_manager()
_routing_engine = RoutingEngine(_state_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _state_manager.start()
    log.info("tensor_fabric.control_plane.ready")
    yield
    await _state_manager.stop()
    log.info("tensor_fabric.control_plane.shutdown")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Tensor Fabric Control Plane",
    description=(
        "Unified GPU-native control plane for Storage, Network, and Inference layers. "
        "Routes tensor operations based on live GPU VRAM state, NVLink topology, "
        "and KV-cache availability — without CPU involvement in the data path."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    summary = _state_manager.get_cluster_summary()
    if summary.get("status") == "initializing":
        raise HTTPException(status_code=503, detail="Initializing GPU topology")
    return {"status": "healthy", "gpu_count": summary.get("gpu_count", 0)}


@app.get("/ready", status_code=status.HTTP_200_OK)
async def readiness():
    topology = _state_manager.snapshot()
    if topology is None or len(topology.nodes) == 0:
        raise HTTPException(status_code=503, detail="No GPUs discovered")
    return {"ready": True}


@app.post("/v1/route", response_model=RouteResponse)
async def route_tensor(req: RouteRequest):
    t0 = time.monotonic()

    descriptor = TensorDescriptor(
        tensor_id=f"req-{int(t0 * 1e9)}",
        model_id=req.model_id,
        role=TensorRole(req.tensor_role),
        shape=tuple(req.shape),
        dtype=TensorDtype(req.dtype),
        layer_index=req.layer_index,
        sequence_length=req.sequence_length,
        batch_size=req.batch_size,
    )

    decision = _routing_engine.route(
        descriptor=descriptor,
        kv_cache_gpu=req.kv_cache_gpu,
        current_gpu=req.current_gpu,
    )

    elapsed = time.monotonic() - t0
    ROUTE_LATENCY.observe(elapsed)
    ROUTE_REQUESTS.labels(strategy=decision.strategy.value).inc()

    log.info(
        "route.decision",
        model=req.model_id,
        target_gpu=decision.target_gpu,
        strategy=decision.strategy.value,
        cache_hit=decision.cache_hit,
        latency_us=round(elapsed * 1e6, 2),
    )

    return RouteResponse(
        tensor_id=decision.tensor_id,
        target_gpu=decision.target_gpu,
        strategy=decision.strategy.value,
        confidence=decision.confidence,
        estimated_latency_us=decision.estimated_latency_us,
        cache_hit=decision.cache_hit,
        prefetch_triggered=decision.prefetch_triggered,
        reason=decision.reason,
    )


@app.post("/v1/models/register", status_code=status.HTTP_201_CREATED)
async def register_model(reg: ModelRegistration):
    _state_manager.register_model(reg.model_id, reg.device_id)
    return {"registered": True, "model_id": reg.model_id, "device_id": reg.device_id}


@app.delete("/v1/models/{model_id}/gpu/{device_id}")
async def evict_model(model_id: str, device_id: int):
    _state_manager.evict_model(model_id, device_id)
    return {"evicted": True, "model_id": model_id, "device_id": device_id}


@app.get("/v1/status", response_model=ClusterStatus)
async def cluster_status():
    summary = _state_manager.get_cluster_summary()
    topology = _state_manager.snapshot()

    if topology:
        for dev_id, node in topology.nodes.items():
            GPU_FREE_VRAM.labels(device_id=str(dev_id)).set(node.vram_free_mb)
            GPU_COMPUTE_UTIL.labels(device_id=str(dev_id)).set(node.compute_util_pct)

    stats = _routing_engine.stats
    CACHE_HIT_RATE.set(stats.get("cache_hit_rate", 0))

    return ClusterStatus(
        gpu_count=summary.get("gpu_count", 0),
        total_vram_gb=summary.get("total_vram_gb", 0),
        free_vram_gb=summary.get("free_vram_gb", 0),
        vram_utilization_pct=summary.get("vram_utilization_pct", 0),
        avg_compute_util_pct=summary.get("avg_compute_util_pct", 0),
        nvswitch_present=summary.get("nvswitch_present", False),
        healthy_gpus=summary.get("healthy_gpus", 0),
        loaded_models=summary.get("loaded_models", {}),
        uptime_seconds=round(time.monotonic() - _start_time, 1),
        route_stats=stats,
    )


@app.get("/v1/topology")
async def get_topology_view():
    topology = _state_manager.snapshot()
    if topology is None:
        raise HTTPException(status_code=503, detail="Topology not yet discovered")

    return {
        "gpu_count": len(topology.nodes),
        "nvswitch_present": topology.nvswitch_present,
        "gpus": {
            dev_id: {
                "name": node.name,
                "vram_total_gb": round(node.vram_total_mb / 1024, 2),
                "vram_free_gb": round(node.vram_free_mb / 1024, 2),
                "compute_util_pct": node.compute_util_pct,
                "temperature_c": node.temperature_c,
                "nvlink_peers": node.nvlink_peers,
                "loaded_models": list(node.loaded_models),
                "is_healthy": node.is_healthy,
                "cuda_capability": list(node.cuda_capability),
            }
            for dev_id, node in topology.nodes.items()
        },
        "links": [
            {
                "src": link.src_gpu,
                "dst": link.dst_gpu,
                "type": link.link_type.value,
                "bandwidth_gbps": link.bandwidth_gbps,
            }
            for link in topology.links[:20]
        ],
    }


@app.get("/metrics")
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
