from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from tensor_fabric.common.tensor_descriptor import TensorDescriptor

log = structlog.get_logger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


@dataclass
class TransferJob:
    descriptor: TensorDescriptor
    src_gpu: int
    dst_gpu: int
    array: Any
    submitted_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    bytes_transferred: int = 0

    @property
    def latency_ms(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.submitted_at) * 1000


class InNetworkCompute:
    """
    Simulates Nvidia Spectrum-X in-network compute capability.

    On real Spectrum-X hardware, packets transiting the switch can be
    processed by the switch ASIC. This is exposed via DOCA Flow + P4 programs.

    Current capability demonstrated here:
    - Tensor shard aggregation (AllReduce partial inside switch)
    - Attention score max-reduction (FlashAttention optimization)
    - In-flight compression/decompression

    This eliminates the need for data to reach a GPU just to be aggregated
    and sent onwards — the switch does it inline.
    """

    @staticmethod
    def allreduce_partial(
        shards: list[Any],
        operation: str = "sum",
    ) -> Any:
        """Aggregate tensor shards as they transit the switch."""
        if not shards:
            return None

        if CUPY_AVAILABLE and isinstance(shards[0], cp.ndarray):
            xp = cp
        else:
            xp = np

        if operation == "sum":
            result = xp.zeros_like(shards[0])
            for shard in shards:
                result += shard
            return result
        elif operation == "max":
            result = shards[0].copy()
            for shard in shards[1:]:
                result = xp.maximum(result, shard)
            return result
        elif operation == "mean":
            result = xp.zeros_like(shards[0], dtype=xp.float32)
            for shard in shards:
                result += shard.astype(xp.float32)
            return (result / len(shards)).astype(shards[0].dtype)

        raise ValueError(f"Unknown operation: {operation}")

    @staticmethod
    def compress_shard(array: Any, ratio: float = 0.5) -> tuple[Any, dict]:
        """
        In-switch lossy compression for activation tensors.
        Uses top-k sparsification — keeps the top-k% largest magnitude values.
        """
        if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            xp = cp
        else:
            xp = np

        flat = array.flatten()
        k = max(1, int(len(flat) * ratio))
        threshold = xp.partition(xp.abs(flat), -k)[-k]
        mask = xp.abs(flat) >= threshold
        sparse = flat * mask

        return sparse.reshape(array.shape), {
            "compression_ratio": ratio,
            "nonzero_pct": float(xp.mean(mask)),
        }


class TensorRouter:
    """
    GPU-aware tensor routing layer — the Network Fabric.

    Responsibilities:
    1. Route tensors between GPUs using the fastest available path
    2. Orchestrate in-network compute via InNetworkCompute
    3. Track transfer jobs and report bandwidth utilization
    4. Choose between NVLink, PCIe, or RDMA based on topology

    When NVSwitch is present (DGX H100, SuperPOD):
    - All GPUs are connected at full NVLink bandwidth (900 GB/s)
    - TensorRouter uses NCCL for collective operations

    When across servers:
    - Uses GPUDirect RDMA over InfiniBand/RoCE (Spectrum-X)
    - Bypasses CPU entirely for server-to-server transfers
    """

    def __init__(self, use_nccl: bool = True) -> None:
        self._use_nccl = use_nccl
        self._nccl_available = self._check_nccl()
        self._in_network = InNetworkCompute()
        self._pending: dict[str, TransferJob] = {}
        self._completed: list[TransferJob] = []
        self._total_bytes = 0

        log.info(
            "tensor_router.initialized",
            nccl=self._nccl_available,
            cupy=CUPY_AVAILABLE,
        )

    def _check_nccl(self) -> bool:
        if not CUPY_AVAILABLE:
            return False
        try:
            import cupy.cuda.nccl as nccl  # type: ignore
            return True
        except Exception:
            return False

    async def transfer(
        self,
        descriptor: TensorDescriptor,
        array: Any,
        src_gpu: int,
        dst_gpu: int,
    ) -> Any:
        """
        Transfer a tensor from src_gpu to dst_gpu.
        Chooses NVLink, PCIe P2P, or serialized copy depending on topology.
        """
        job = TransferJob(
            descriptor=descriptor,
            src_gpu=src_gpu,
            dst_gpu=dst_gpu,
            array=array,
        )
        self._pending[descriptor.tensor_id] = job

        if src_gpu == dst_gpu:
            result = array
        elif CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            result = await self._transfer_gpu_to_gpu(array, src_gpu, dst_gpu)
        else:
            result = array

        job.completed_at = time.monotonic()
        job.bytes_transferred = descriptor.nbytes
        self._total_bytes += descriptor.nbytes
        self._pending.pop(descriptor.tensor_id, None)
        self._completed.append(job)

        log.debug(
            "tensor_router.transfer",
            src=src_gpu,
            dst=dst_gpu,
            size_mb=round(descriptor.nbytes_mb, 2),
            latency_ms=round(job.latency_ms or 0, 3),
        )
        return result

    async def broadcast(
        self,
        descriptor: TensorDescriptor,
        array: Any,
        src_gpu: int,
        dst_gpus: list[int],
    ) -> dict[int, Any]:
        """
        Broadcast a tensor from one GPU to many.
        Uses NCCL broadcast when available (leverages NVSwitch).
        """
        if self._nccl_available and CUPY_AVAILABLE and len(dst_gpus) > 1:
            return await self._nccl_broadcast(array, src_gpu, dst_gpus, descriptor)

        tasks = [
            self.transfer(descriptor, array, src_gpu, dst)
            for dst in dst_gpus if dst != src_gpu
        ]
        results = await asyncio.gather(*tasks)
        output = {src_gpu: array}
        output.update(dict(zip([d for d in dst_gpus if d != src_gpu], results)))
        return output

    async def allreduce(
        self,
        arrays: dict[int, Any],
        descriptor: TensorDescriptor,
        operation: str = "sum",
    ) -> dict[int, Any]:
        """
        AllReduce across multiple GPUs.
        For Spectrum-X environments, the reduction happens in-switch.
        """
        shards = list(arrays.values())

        if self._nccl_available and CUPY_AVAILABLE and len(shards) > 1:
            return await self._nccl_allreduce(arrays, descriptor)

        reduced = self._in_network.allreduce_partial(shards, operation)
        return {gpu: reduced for gpu in arrays.keys()}

    async def _transfer_gpu_to_gpu(self, array: Any, src_gpu: int, dst_gpu: int) -> Any:
        """
        GPU-to-GPU transfer. CuPy handles NVLink vs PCIe automatically.
        With NVSwitch this runs at 900 GB/s; with PCIe ~64 GB/s.
        """
        await asyncio.sleep(0)
        with cp.cuda.Device(dst_gpu):
            return cp.array(array)

    async def _nccl_broadcast(
        self,
        array: Any,
        src_gpu: int,
        dst_gpus: list[int],
        descriptor: TensorDescriptor,
    ) -> dict[int, Any]:
        """
        NCCL broadcast — leverages NVSwitch for all-at-once transfer.
        On DGX H100: 900 GB/s aggregate, hardware-scheduled.
        """
        all_gpus = [src_gpu] + [g for g in dst_gpus if g != src_gpu]
        results = {}

        try:
            import cupy.cuda.nccl as nccl

            comms = nccl.NcclCommunicator.initAll(all_gpus)
            nccl.groupStart()

            arrays_per_gpu: dict[int, Any] = {}
            for i, gpu in enumerate(all_gpus):
                with cp.cuda.Device(gpu):
                    if gpu == src_gpu:
                        arrays_per_gpu[gpu] = array
                    else:
                        arrays_per_gpu[gpu] = cp.empty_like(array)
                    comms[i].broadcast(
                        arrays_per_gpu[gpu].data.ptr,
                        arrays_per_gpu[gpu].data.ptr,
                        array.size,
                        nccl.NCCL_FLOAT16 if descriptor.dtype.value == "float16" else nccl.NCCL_FLOAT,
                        all_gpus.index(src_gpu),
                        cp.cuda.Stream.null.ptr,
                    )

            nccl.groupEnd()
            for comm in comms:
                comm.destroy()

            return arrays_per_gpu

        except Exception as exc:
            log.warning("nccl_broadcast.failed", error=str(exc), fallback="sequential")
            results = {src_gpu: array}
            for dst in dst_gpus:
                if dst != src_gpu:
                    results[dst] = await self._transfer_gpu_to_gpu(array, src_gpu, dst)
            return results

    async def _nccl_allreduce(self, arrays: dict[int, Any], descriptor: TensorDescriptor) -> dict[int, Any]:
        try:
            import cupy.cuda.nccl as nccl

            gpu_list = list(arrays.keys())
            comms = nccl.NcclCommunicator.initAll(gpu_list)
            nccl.groupStart()

            for i, (gpu, arr) in enumerate(arrays.items()):
                with cp.cuda.Device(gpu):
                    comms[i].allReduce(
                        arr.data.ptr, arr.data.ptr, arr.size,
                        nccl.NCCL_FLOAT16, nccl.NCCL_SUM,
                        cp.cuda.Stream.null.ptr,
                    )

            nccl.groupEnd()
            for comm in comms:
                comm.destroy()

            return arrays

        except Exception as exc:
            log.warning("nccl_allreduce.failed", error=str(exc))
            shards = list(arrays.values())
            reduced = self._in_network.allreduce_partial(shards, "sum")
            return {gpu: reduced for gpu in arrays.keys()}

    @property
    def stats(self) -> dict:
        completed = self._completed[-1000:]
        if not completed:
            return {"transfers": 0, "total_gb": 0, "avg_latency_ms": 0}

        latencies = [j.latency_ms for j in completed if j.latency_ms is not None]
        return {
            "transfers": len(completed),
            "pending": len(self._pending),
            "total_gb": round(self._total_bytes / 1e9, 3),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 3),
            "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0, 3),
            "nccl_enabled": self._nccl_available,
        }
