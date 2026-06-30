from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TensorDtype(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def itemsize(self) -> int:
        return {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1, "int4": 1}[self.value]


class TensorRole(str, Enum):
    MODEL_WEIGHT = "model_weight"
    KV_CACHE = "kv_cache"
    ACTIVATION = "activation"
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class StorageLocation:
    """Describes where a tensor lives physically."""
    nvme_path: str | None = None          # On-disk path (GPUDirect Storage)
    gpu_device_id: int | None = None      # Which GPU holds this in VRAM
    gpu_ptr: int | None = None            # Raw CUDA device pointer
    rdma_handle: bytes | None = None      # RDMA memory registration key
    is_pinned: bool = False               # Pinned host memory for fast DMA


@dataclass
class TensorDescriptor:
    """
    The universal currency of Tensor Fabric.
    Flows through all three layers (Storage → Network → Inference),
    enabling each layer to make GPU-native decisions without CPU involvement.
    """
    tensor_id: str
    model_id: str
    role: TensorRole
    shape: tuple[int, ...]
    dtype: TensorDtype

    # Transformer-specific metadata
    layer_index: int | None = None
    head_count: int | None = None
    sequence_length: int | None = None
    batch_size: int = 1

    # Placement hints
    gpu_affinity: list[int] = field(default_factory=list)
    storage: StorageLocation = field(default_factory=StorageLocation)

    # Lifecycle
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 0
    ttl_seconds: float | None = None

    # Routing
    checksum: str | None = None
    compressed: bool = False
    compression_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not self.checksum:
            key = f"{self.model_id}:{self.role}:{self.shape}:{self.layer_index}"
            self.checksum = hashlib.sha256(key.encode()).hexdigest()[:16]

    @property
    def nbytes(self) -> int:
        size = 1
        for dim in self.shape:
            size *= dim
        return size * self.dtype.itemsize

    @property
    def nbytes_mb(self) -> float:
        return self.nbytes / (1024 * 1024)

    @property
    def cache_key(self) -> str:
        return f"{self.model_id}:{self.role.value}:{self.layer_index}:{self.checksum}"

    def is_kv_cache(self) -> bool:
        return self.role == TensorRole.KV_CACHE

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.monotonic() - self.last_accessed) > self.ttl_seconds

    def touch(self) -> None:
        self.last_accessed = time.monotonic()
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_id": self.tensor_id,
            "model_id": self.model_id,
            "role": self.role.value,
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "layer_index": self.layer_index,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "gpu_affinity": self.gpu_affinity,
            "nbytes": self.nbytes,
            "cache_key": self.cache_key,
        }

    @classmethod
    def for_kv_cache(
        cls,
        model_id: str,
        layer_index: int,
        sequence_length: int,
        num_heads: int,
        head_dim: int,
        batch_size: int = 1,
        dtype: TensorDtype = TensorDtype.FLOAT16,
    ) -> "TensorDescriptor":
        import uuid
        shape = (batch_size, num_heads, sequence_length, head_dim)
        return cls(
            tensor_id=str(uuid.uuid4()),
            model_id=model_id,
            role=TensorRole.KV_CACHE,
            shape=shape,
            dtype=dtype,
            layer_index=layer_index,
            head_count=num_heads,
            sequence_length=sequence_length,
            batch_size=batch_size,
            ttl_seconds=300.0,
        )

    @classmethod
    def for_model_weight(
        cls,
        model_id: str,
        layer_index: int,
        shape: tuple[int, ...],
        dtype: TensorDtype = TensorDtype.BFLOAT16,
        nvme_path: str | None = None,
    ) -> "TensorDescriptor":
        import uuid
        return cls(
            tensor_id=str(uuid.uuid4()),
            model_id=model_id,
            role=TensorRole.MODEL_WEIGHT,
            shape=shape,
            dtype=dtype,
            layer_index=layer_index,
            storage=StorageLocation(nvme_path=nvme_path),
            ttl_seconds=None,
        )
