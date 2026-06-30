from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class LinkType(str, Enum):
    NVLINK = "nvlink"
    PCIE = "pcie"
    INFINIBAND = "infiniband"
    ROCE = "roce"


@dataclass
class GPULink:
    src_gpu: int
    dst_gpu: int
    link_type: LinkType
    bandwidth_gbps: float
    latency_us: float


@dataclass
class GPUNode:
    device_id: int
    uuid: str
    name: str
    vram_total_mb: float
    vram_free_mb: float
    compute_util_pct: float
    memory_util_pct: float
    temperature_c: float
    power_draw_w: float
    sm_count: int
    cuda_capability: tuple[int, int]
    loaded_models: set[str] = field(default_factory=set)
    nvlink_peers: list[int] = field(default_factory=list)

    @property
    def vram_used_mb(self) -> float:
        return self.vram_total_mb - self.vram_free_mb

    @property
    def vram_utilization_pct(self) -> float:
        if self.vram_total_mb == 0:
            return 0.0
        return (self.vram_used_mb / self.vram_total_mb) * 100.0

    @property
    def is_healthy(self) -> bool:
        return (
            self.temperature_c < 85.0
            and self.vram_utilization_pct < 95.0
            and self.compute_util_pct < 98.0
        )

    @property
    def free_vram_score(self) -> float:
        """Score for routing — higher means more room for tensors."""
        return self.vram_free_mb * (1.0 - self.compute_util_pct / 100.0)


@dataclass
class GPUTopology:
    """
    Live snapshot of the GPU cluster topology.
    The control plane reads this to make routing decisions
    without any CPU-side heuristics.
    """
    nodes: dict[int, GPUNode] = field(default_factory=dict)
    links: list[GPULink] = field(default_factory=list)
    nvswitch_present: bool = False
    bluefield_dpu_ids: list[str] = field(default_factory=list)

    def get_nvlink_peers(self, device_id: int) -> list[int]:
        return [
            link.dst_gpu
            for link in self.links
            if link.src_gpu == device_id and link.link_type == LinkType.NVLINK
        ]

    def best_gpu_for_model(self, model_vram_mb: float) -> int | None:
        candidates = [
            node for node in self.nodes.values()
            if node.vram_free_mb >= model_vram_mb * 1.1 and node.is_healthy
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.free_vram_score).device_id

    def gpus_with_model(self, model_id: str) -> list[int]:
        return [
            dev_id for dev_id, node in self.nodes.items()
            if model_id in node.loaded_models
        ]

    def total_free_vram_mb(self) -> float:
        return sum(n.vram_free_mb for n in self.nodes.values())

    def least_loaded_gpu(self) -> int | None:
        if not self.nodes:
            return None
        return min(self.nodes.values(), key=lambda n: n.compute_util_pct).device_id

    def iter_healthy_gpus(self) -> Iterator[GPUNode]:
        for node in self.nodes.values():
            if node.is_healthy:
                yield node


class GPUTopologyDiscovery:
    """
    Discovers and maintains live GPU topology using pynvml.
    Falls back to a mock topology for development without GPUs.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._topology = GPUTopology()

    def initialize(self) -> None:
        if not PYNVML_AVAILABLE:
            self._topology = self._mock_topology()
            self._initialized = True
            return

        try:
            pynvml.nvmlInit()
            self._initialized = True
        except pynvml.NVMLError:
            self._topology = self._mock_topology()
            self._initialized = True

    def refresh(self) -> GPUTopology:
        if not self._initialized:
            self.initialize()

        if not PYNVML_AVAILABLE:
            return self._topology

        try:
            return self._discover_live()
        except Exception:
            return self._topology

    def _discover_live(self) -> GPUTopology:
        gpu_count = pynvml.nvmlDeviceGetCount()
        nodes: dict[int, GPUNode] = {}
        links: list[GPULink] = []

        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

            try:
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                name = pynvml.nvmlDeviceGetName(handle)
            except Exception:
                uuid = f"GPU-{i}"
                name = "Unknown"

            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                cuda_cap = (major, minor)
            except Exception:
                cuda_cap = (8, 0)

            nvlink_peers: list[int] = []
            for j in range(gpu_count):
                if j == i:
                    continue
                try:
                    status = pynvml.nvmlDeviceGetNvLinkState(handle, j)
                    if status == pynvml.NVML_NVLINK_ACTIVE:
                        nvlink_peers.append(j)
                        links.append(GPULink(
                            src_gpu=i, dst_gpu=j,
                            link_type=LinkType.NVLINK,
                            bandwidth_gbps=600.0,
                            latency_us=1.5,
                        ))
                except Exception:
                    pass

            nodes[i] = GPUNode(
                device_id=i,
                uuid=str(uuid),
                name=str(name),
                vram_total_mb=mem_info.total / (1024 * 1024),
                vram_free_mb=mem_info.free / (1024 * 1024),
                compute_util_pct=float(util.gpu),
                memory_util_pct=float(util.memory),
                temperature_c=float(temp),
                power_draw_w=float(power),
                sm_count=0,
                cuda_capability=cuda_cap,
                nvlink_peers=nvlink_peers,
            )

        self._topology = GPUTopology(
            nodes=nodes,
            links=links,
            nvswitch_present=len(links) > 4,
        )
        return self._topology

    def _mock_topology(self) -> GPUTopology:
        nodes = {
            i: GPUNode(
                device_id=i,
                uuid=f"GPU-MOCK-{i:04d}",
                name=f"NVIDIA H100 SXM5 [mock-{i}]",
                vram_total_mb=81920.0,
                vram_free_mb=81920.0 - (i * 8192),
                compute_util_pct=float(i * 12),
                memory_util_pct=float(i * 10),
                temperature_c=55.0 + i * 3,
                power_draw_w=350.0,
                sm_count=132,
                cuda_capability=(9, 0),
                nvlink_peers=[j for j in range(8) if j != i],
            )
            for i in range(8)
        }
        links = [
            GPULink(i, j, LinkType.NVLINK, 900.0, 1.0)
            for i in range(8) for j in range(8) if i != j
        ]
        return GPUTopology(nodes=nodes, links=links, nvswitch_present=True)


_discovery = GPUTopologyDiscovery()


def get_topology() -> GPUTopology:
    return _discovery.refresh()


async def watch_topology(interval_seconds: float = 1.0):
    """Async generator that yields topology snapshots."""
    while True:
        yield _discovery.refresh()
        await asyncio.sleep(interval_seconds)
