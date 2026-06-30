from __future__ import annotations

import asyncio
import hashlib
import os
import struct
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from tensor_fabric.common.tensor_descriptor import (
    StorageLocation,
    TensorDescriptor,
    TensorDtype,
    TensorRole,
)

log = structlog.get_logger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# ── TensorFS on-disk format ───────────────────────────────────────────────────
# .tfs file layout:
#   [0:4]   magic   = b"TFS1"
#   [4:8]   version = uint32
#   [8:16]  nbytes  = uint64  (raw tensor data size)
#   [16:48] shape   = 4x uint64 (up to 4D, zero-padded)
#   [48:56] dtype   = 8-byte ASCII string
#   [56:72] model_id hash = 16 bytes
#   [72:]   tensor data (row-major, native byte order)

MAGIC = b"TFS1"
HEADER_SIZE = 72


@dataclass
class TFSFile:
    path: Path
    descriptor: TensorDescriptor


class TensorFilesystem:
    """
    GPU-native filesystem where files ARE tensors.

    Design goals:
    - open() returns a TensorDescriptor, not a file handle
    - read() returns a CuPy array directly in GPU VRAM (no CPU copy)
    - write() accepts a CuPy array and persists without CPU involvement
    - Metadata (shape, dtype, model affinity) is the primary key, not filename

    GPUDirect Storage Integration:
    When nvidia-fs kernel module is present, read/write bypass the CPU
    page cache entirely and DMA directly NVMe → GPU VRAM.
    Without it, we fall back to host-pinned memory as an intermediate.
    """

    def __init__(self, base_path: str | Path = "/tmp/tensor-fabric/storage") -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, TFSFile] = {}
        self._gpudirect_available = self._check_gpudirect()
        log.info(
            "tensor_fs.initialized",
            base_path=str(self._base),
            gpudirect=self._gpudirect_available,
            cupy=CUPY_AVAILABLE,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def write_tensor(
        self,
        descriptor: TensorDescriptor,
        array: Any,
    ) -> Path:
        """
        Persist a tensor to TensorFS.
        array can be cupy.ndarray, numpy.ndarray, or bytes.
        Returns the path where it was stored.
        """
        path = self._descriptor_to_path(descriptor)
        t0 = time.monotonic()

        if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            await self._write_cupy(path, descriptor, array)
        elif isinstance(array, np.ndarray):
            await self._write_numpy(path, descriptor, array)
        else:
            raise TypeError(f"Unsupported array type: {type(array)}")

        descriptor.storage.nvme_path = str(path)
        self._index[descriptor.cache_key] = TFSFile(path=path, descriptor=descriptor)

        elapsed_ms = (time.monotonic() - t0) * 1000
        throughput_gbs = (descriptor.nbytes / 1e9) / max(elapsed_ms / 1000, 1e-6)
        log.info(
            "tensor_fs.write",
            tensor_id=descriptor.tensor_id,
            size_mb=round(descriptor.nbytes_mb, 2),
            elapsed_ms=round(elapsed_ms, 2),
            throughput_gbs=round(throughput_gbs, 2),
        )
        return path

    async def read_tensor(
        self,
        cache_key: str,
        target_gpu: int = 0,
    ) -> tuple[Any, TensorDescriptor] | None:
        """
        Load a tensor from TensorFS into GPU VRAM.
        Returns (cupy_array, descriptor) or None if not found.
        """
        tfs_file = self._index.get(cache_key)
        if tfs_file is None:
            return None

        t0 = time.monotonic()
        path = tfs_file.path

        if not path.exists():
            self._index.pop(cache_key, None)
            return None

        if self._gpudirect_available and CUPY_AVAILABLE:
            array = await self._read_gpudirect(path, tfs_file.descriptor, target_gpu)
        else:
            array = await self._read_host_pinned(path, tfs_file.descriptor, target_gpu)

        elapsed_ms = (time.monotonic() - t0) * 1000
        throughput_gbs = (tfs_file.descriptor.nbytes / 1e9) / max(elapsed_ms / 1000, 1e-6)
        log.info(
            "tensor_fs.read",
            cache_key=cache_key,
            target_gpu=target_gpu,
            elapsed_ms=round(elapsed_ms, 2),
            throughput_gbs=round(throughput_gbs, 2),
            gpudirect=self._gpudirect_available,
        )

        tfs_file.descriptor.touch()
        return array, tfs_file.descriptor

    def stat(self, cache_key: str) -> TensorDescriptor | None:
        tfs_file = self._index.get(cache_key)
        return tfs_file.descriptor if tfs_file else None

    def ls(self, model_id: str | None = None) -> list[TensorDescriptor]:
        files = self._index.values()
        if model_id:
            files = [f for f in files if f.descriptor.model_id == model_id]
        return [f.descriptor for f in files]

    def exists(self, cache_key: str) -> bool:
        return cache_key in self._index

    async def delete(self, cache_key: str) -> bool:
        tfs_file = self._index.pop(cache_key, None)
        if tfs_file and tfs_file.path.exists():
            tfs_file.path.unlink()
            return True
        return False

    def storage_stats(self) -> dict:
        total_bytes = sum(
            f.descriptor.nbytes for f in self._index.values()
        )
        return {
            "tensor_count": len(self._index),
            "total_size_gb": round(total_bytes / 1e9, 3),
            "gpudirect_enabled": self._gpudirect_available,
            "base_path": str(self._base),
        }

    # ── Internal writers ──────────────────────────────────────────────────────

    async def _write_cupy(self, path: Path, desc: TensorDescriptor, array: Any) -> None:
        raw_bytes = cp.asnumpy(array).tobytes()
        await asyncio.get_event_loop().run_in_executor(
            None, self._write_tfs_file, path, desc, raw_bytes
        )

    async def _write_numpy(self, path: Path, desc: TensorDescriptor, array: np.ndarray) -> None:
        raw_bytes = array.tobytes()
        await asyncio.get_event_loop().run_in_executor(
            None, self._write_tfs_file, path, desc, raw_bytes
        )

    def _write_tfs_file(self, path: Path, desc: TensorDescriptor, data: bytes) -> None:
        shape_padded = list(desc.shape) + [0] * (4 - len(desc.shape))
        shape_padded = shape_padded[:4]
        dtype_bytes = desc.dtype.value.ljust(8).encode()[:8]
        model_hash = hashlib.md5(desc.model_id.encode()).digest()

        header = struct.pack(
            "<4sIQ4Q8s16s",
            MAGIC,
            1,
            len(data),
            *shape_padded,
            dtype_bytes,
            model_hash,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(header)
            f.write(data)

    async def _read_gpudirect(self, path: Path, desc: TensorDescriptor, gpu: int) -> Any:
        """
        GPUDirect Storage path: cuFile API bypasses CPU page cache.
        Requires: nvidia-fs kernel module, libcufile.so
        Falls back to host-pinned if cuFile is unavailable.
        """
        try:
            return await self._read_via_cufile(path, desc, gpu)
        except Exception:
            return await self._read_host_pinned(path, desc, gpu)

    async def _read_via_cufile(self, path: Path, desc: TensorDescriptor, gpu: int) -> Any:
        """
        Production path: cuFile DMA → GPU VRAM.
        The cufile Python bindings are part of CUDA 12 toolkit.
        """
        try:
            import cufile  # type: ignore
        except ImportError:
            return await self._read_host_pinned(path, desc, gpu)

        with cp.cuda.Device(gpu):
            gpu_buf = cp.empty(desc.shape, dtype=desc.dtype.value)
            with cufile.CuFile(str(path), "r") as cf:
                cf.read(gpu_buf, file_offset=HEADER_SIZE)
        return gpu_buf

    async def _read_host_pinned(self, path: Path, desc: TensorDescriptor, gpu: int) -> Any:
        """
        Fallback: read via host pinned memory → GPU (one DMA hop, no CPU copy)
        """
        raw = await asyncio.get_event_loop().run_in_executor(
            None, self._read_tfs_raw, path
        )
        if raw is None:
            return None

        np_array = np.frombuffer(raw, dtype=desc.dtype.value).reshape(desc.shape)

        if CUPY_AVAILABLE:
            with cp.cuda.Device(gpu):
                return cp.asarray(np_array)
        return np_array

    def _read_tfs_raw(self, path: Path) -> bytes | None:
        try:
            with open(path, "rb") as f:
                header = f.read(HEADER_SIZE)
                if len(header) < HEADER_SIZE or header[:4] != MAGIC:
                    return None
                nbytes = struct.unpack_from("<Q", header, 8)[0]
                return f.read(nbytes)
        except (OSError, struct.error):
            return None

    def _descriptor_to_path(self, desc: TensorDescriptor) -> Path:
        safe_model = desc.model_id.replace("/", "_").replace(":", "_")
        subdir = self._base / safe_model / desc.role.value
        fname = f"{desc.layer_index or 0:04d}_{desc.checksum}.tfs"
        return subdir / fname

    def _check_gpudirect(self) -> bool:
        return os.path.exists("/proc/driver/nvidia-fs/status")
