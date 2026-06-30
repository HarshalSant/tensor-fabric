"""Tests for KVCacheManager — the cross-pod shared KV-cache."""
import asyncio
import numpy as np
import pytest
from tensor_fabric.inference.kv_cache_manager import KVCacheManager, GPUKVCachePool


# ── GPUKVCachePool unit tests ─────────────────────────────────────────────────

def make_kv_entry(seq_id="seq1", model="llama3-8b", layer=0, seq_len=128,
                   num_heads=32, head_dim=128, gpu=0):
    from tensor_fabric.inference.kv_cache_manager import KVCacheEntry
    keys = np.zeros((1, num_heads, seq_len, head_dim), dtype=np.float16)
    values = np.zeros((1, num_heads, seq_len, head_dim), dtype=np.float16)
    return KVCacheEntry(
        sequence_id=seq_id, model_id=model, layer_index=layer,
        keys=keys, values=values, gpu_device=gpu,
        num_heads=num_heads, head_dim=head_dim, sequence_length=seq_len,
    )


def test_pool_put_and_get():
    pool = GPUKVCachePool(device_id=0, capacity_mb=512)
    entry = make_kv_entry()
    assert pool.put(entry)
    result = pool.get(entry.cache_key)
    assert result is not None
    assert result.sequence_id == "seq1"


def test_pool_miss_returns_none():
    pool = GPUKVCachePool(device_id=0, capacity_mb=512)
    assert pool.get("nonexistent:key:0") is None


def test_pool_lru_eviction():
    pool = GPUKVCachePool(device_id=0, capacity_mb=1)
    large_entry = make_kv_entry(seq_id="big", seq_len=4096)
    small_entry = make_kv_entry(seq_id="small", seq_len=64)
    pool.put(large_entry)
    pool.put(small_entry)
    assert pool.entry_count <= 2


def test_pool_evict_sequence():
    pool = GPUKVCachePool(device_id=0, capacity_mb=512)
    for layer in range(4):
        pool.put(make_kv_entry(seq_id="seq-A", layer=layer))
    pool.put(make_kv_entry(seq_id="seq-B", layer=0))
    removed = pool.evict_sequence("seq-A")
    assert removed == 4
    assert pool.get("llama3-8b:seq-A:0") is None
    assert pool.get("llama3-8b:seq-B:0") is not None


def test_pool_utilization():
    pool = GPUKVCachePool(device_id=0, capacity_mb=512)
    entry = make_kv_entry()
    pool.put(entry)
    assert pool.utilization_pct > 0
    assert pool.utilization_pct < 100


# ── KVCacheManager async tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_and_get():
    mgr = KVCacheManager(capacity_per_gpu_mb=1024)
    mgr.add_gpu(0)
    keys = np.zeros((1, 32, 128, 128), dtype=np.float16)
    values = np.zeros((1, 32, 128, 128), dtype=np.float16)
    ok = await mgr.store_kv("seq1", "llama3-8b", 0, keys, values, 0, 32, 128)
    assert ok
    result = await mgr.get_kv("seq1", "llama3-8b", 0)
    assert result is not None
    k, v, gpu = result
    assert k.shape == (1, 32, 128, 128)
    assert gpu == 0


@pytest.mark.asyncio
async def test_cache_miss():
    mgr = KVCacheManager(capacity_per_gpu_mb=1024)
    mgr.add_gpu(0)
    result = await mgr.get_kv("nonexistent", "llama3-8b", 0)
    assert result is None


@pytest.mark.asyncio
async def test_hit_rate_tracking():
    mgr = KVCacheManager(capacity_per_gpu_mb=1024)
    mgr.add_gpu(0)
    keys = np.zeros((1, 32, 64, 128), dtype=np.float16)
    values = np.zeros((1, 32, 64, 128), dtype=np.float16)
    await mgr.store_kv("s1", "m", 0, keys, values, 0, 32, 128)
    await mgr.get_kv("s1", "m", 0)   # hit
    await mgr.get_kv("s2", "m", 0)   # miss
    assert mgr.hit_rate == pytest.approx(0.5, abs=0.01)


@pytest.mark.asyncio
async def test_lookup_sequence_gpu():
    mgr = KVCacheManager(capacity_per_gpu_mb=1024)
    mgr.add_gpu(0)
    mgr.add_gpu(1)
    keys = np.zeros((1, 32, 64, 128), dtype=np.float16)
    values = np.zeros((1, 32, 64, 128), dtype=np.float16)
    await mgr.store_kv("myseq", "llama3-8b", 0, keys, values, 0, 32, 128)
    gpu = await mgr.lookup_sequence_gpu("myseq", "llama3-8b")
    assert gpu == 0


@pytest.mark.asyncio
async def test_extend_kv():
    mgr = KVCacheManager(capacity_per_gpu_mb=1024)
    mgr.add_gpu(0)
    k1 = np.zeros((1, 32, 64, 128), dtype=np.float16)
    v1 = np.zeros((1, 32, 64, 128), dtype=np.float16)
    await mgr.store_kv("s", "m", 0, k1, v1, 0, 32, 128)
    k2 = np.zeros((1, 32, 16, 128), dtype=np.float16)
    v2 = np.zeros((1, 32, 16, 128), dtype=np.float16)
    ok = await mgr.extend_kv("s", "m", 0, k2, v2)
    assert ok
    result = await mgr.get_kv("s", "m", 0)
    assert result is not None
    k, _, _ = result
    assert k.shape[-2] == 80


@pytest.mark.asyncio
async def test_evict_sequence():
    mgr = KVCacheManager(capacity_per_gpu_mb=1024)
    mgr.add_gpu(0)
    keys = np.zeros((1, 32, 64, 128), dtype=np.float16)
    values = np.zeros((1, 32, 64, 128), dtype=np.float16)
    await mgr.store_kv("evict-me", "m", 0, keys, values, 0, 32, 128)
    await mgr.evict_sequence("evict-me")
    result = await mgr.get_kv("evict-me", "m", 0)
    assert result is None


@pytest.mark.asyncio
async def test_stats_structure():
    mgr = KVCacheManager(capacity_per_gpu_mb=1024)
    mgr.add_gpu(0)
    stats = mgr.stats
    assert "hit_rate" in stats
    assert "hit_count" in stats
    assert "miss_count" in stats
    assert "pools" in stats
    assert 0 in stats["pools"]
