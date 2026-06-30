"""Tests for TensorDescriptor — the universal data currency of Tensor Fabric."""
import pytest
from tensor_fabric.common.tensor_descriptor import (
    TensorDescriptor, TensorDtype, TensorRole, StorageLocation
)


def test_basic_creation():
    desc = TensorDescriptor(
        tensor_id="t1",
        model_id="llama3-8b",
        role=TensorRole.MODEL_WEIGHT,
        shape=(4096, 4096),
        dtype=TensorDtype.BFLOAT16,
    )
    assert desc.tensor_id == "t1"
    assert desc.model_id == "llama3-8b"
    assert desc.shape == (4096, 4096)
    assert desc.dtype == TensorDtype.BFLOAT16


def test_nbytes_float16():
    desc = TensorDescriptor(
        tensor_id="t2", model_id="m", role=TensorRole.ACTIVATION,
        shape=(1, 32, 512, 128), dtype=TensorDtype.FLOAT16,
    )
    expected = 1 * 32 * 512 * 128 * 2
    assert desc.nbytes == expected


def test_nbytes_bfloat16():
    desc = TensorDescriptor(
        tensor_id="t3", model_id="m", role=TensorRole.MODEL_WEIGHT,
        shape=(4096, 4096), dtype=TensorDtype.BFLOAT16,
    )
    assert desc.nbytes == 4096 * 4096 * 2


def test_nbytes_mb():
    desc = TensorDescriptor(
        tensor_id="t4", model_id="m", role=TensorRole.KV_CACHE,
        shape=(1024 * 1024,), dtype=TensorDtype.FLOAT32,
    )
    assert abs(desc.nbytes_mb - 4.0) < 0.01


def test_checksum_auto_generated():
    desc = TensorDescriptor(
        tensor_id="t5", model_id="llama3", role=TensorRole.KV_CACHE,
        shape=(1, 32, 512, 128), dtype=TensorDtype.FLOAT16,
    )
    assert desc.checksum is not None
    assert len(desc.checksum) == 16


def test_checksum_deterministic():
    def make():
        return TensorDescriptor(
            tensor_id="tx", model_id="llama3", role=TensorRole.KV_CACHE,
            shape=(1, 32, 512, 128), dtype=TensorDtype.FLOAT16, layer_index=5,
        )
    assert make().checksum == make().checksum


def test_cache_key_format():
    desc = TensorDescriptor(
        tensor_id="t6", model_id="mistral-7b", role=TensorRole.KV_CACHE,
        shape=(1, 32, 512, 128), dtype=TensorDtype.FLOAT16, layer_index=3,
    )
    assert "mistral-7b" in desc.cache_key
    assert "kv_cache" in desc.cache_key


def test_is_kv_cache():
    kv = TensorDescriptor(
        tensor_id="t7", model_id="m", role=TensorRole.KV_CACHE,
        shape=(1,), dtype=TensorDtype.FLOAT16,
    )
    w = TensorDescriptor(
        tensor_id="t8", model_id="m", role=TensorRole.MODEL_WEIGHT,
        shape=(1,), dtype=TensorDtype.FLOAT16,
    )
    assert kv.is_kv_cache()
    assert not w.is_kv_cache()


def test_touch_updates_access():
    import time
    desc = TensorDescriptor(
        tensor_id="t9", model_id="m", role=TensorRole.INPUT,
        shape=(1,), dtype=TensorDtype.FLOAT16,
    )
    before = desc.last_accessed
    time.sleep(0.05)
    desc.touch()
    assert desc.last_accessed >= before
    assert desc.access_count == 1


def test_expiry():
    desc = TensorDescriptor(
        tensor_id="t10", model_id="m", role=TensorRole.KV_CACHE,
        shape=(1,), dtype=TensorDtype.FLOAT16, ttl_seconds=0.01,
    )
    import time; time.sleep(0.05)
    assert desc.is_expired()


def test_not_expired_with_none_ttl():
    desc = TensorDescriptor(
        tensor_id="t11", model_id="m", role=TensorRole.MODEL_WEIGHT,
        shape=(1,), dtype=TensorDtype.FLOAT16, ttl_seconds=None,
    )
    assert not desc.is_expired()


def test_factory_for_kv_cache():
    desc = TensorDescriptor.for_kv_cache(
        model_id="llama3-8b", layer_index=5,
        sequence_length=512, num_heads=32, head_dim=128, batch_size=2,
    )
    assert desc.role == TensorRole.KV_CACHE
    assert desc.shape == (2, 32, 512, 128)
    assert desc.layer_index == 5
    assert desc.ttl_seconds == 300.0


def test_factory_for_model_weight():
    desc = TensorDescriptor.for_model_weight(
        model_id="llama3-8b", layer_index=0,
        shape=(4096, 4096), nvme_path="/data/weights/layer0.tfs"
    )
    assert desc.role == TensorRole.MODEL_WEIGHT
    assert desc.storage.nvme_path == "/data/weights/layer0.tfs"


def test_to_dict():
    desc = TensorDescriptor(
        tensor_id="t12", model_id="m", role=TensorRole.OUTPUT,
        shape=(1, 512), dtype=TensorDtype.FLOAT32,
    )
    d = desc.to_dict()
    assert d["tensor_id"] == "t12"
    assert d["role"] == "output"
    assert d["shape"] == [1, 512]
    assert "nbytes" in d
