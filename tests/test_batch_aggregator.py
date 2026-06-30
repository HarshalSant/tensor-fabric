"""Tests for BatchAggregator — NIC-level sequence-bucketed batching."""
import asyncio
import pytest
from tensor_fabric.inference.batch_aggregator import (
    BatchAggregator, BatchAggregatorConfig, InferenceRequest, BatchedRequest
)


def make_request(model="llama3-8b", seq_len=128, priority=0):
    import uuid
    return InferenceRequest(
        request_id=str(uuid.uuid4()),
        model_id=model,
        input_ids=list(range(seq_len)),
        priority=priority,
    )


def test_request_creation():
    req = make_request(seq_len=256)
    assert req.sequence_length == 256
    assert req.model_id == "llama3-8b"


def test_bucket_sizes():
    agg = BatchAggregator()
    assert agg._get_bucket(64) == 128
    assert agg._get_bucket(128) == 128
    assert agg._get_bucket(129) == 256
    assert agg._get_bucket(512) == 512
    assert agg._get_bucket(1000) == 1024
    assert agg._get_bucket(5000) == 4096


def test_form_batches_groups_same_bucket():
    agg = BatchAggregator(config=BatchAggregatorConfig(max_batch_size=16))
    requests = [make_request(seq_len=100) for _ in range(4)]
    batches = agg._form_batches("llama3-8b", requests)
    assert len(batches) == 1
    assert batches[0].batch_size == 4


def test_form_batches_splits_different_buckets():
    agg = BatchAggregator(config=BatchAggregatorConfig(max_batch_size=32))
    short = [make_request(seq_len=64) for _ in range(4)]
    long_ = [make_request(seq_len=600) for _ in range(4)]
    batches = agg._form_batches("llama3-8b", short + long_)
    assert len(batches) >= 2


def test_batch_efficiency():
    agg = BatchAggregator()
    requests = [make_request(seq_len=128) for _ in range(8)]
    batches = agg._form_single_batch("llama3-8b", requests)
    assert batches is not None
    assert batches.efficiency_pct == pytest.approx(100.0, abs=1.0)


def test_batch_pads_to_bucket():
    agg = BatchAggregator()
    requests = [make_request(seq_len=100), make_request(seq_len=90)]
    batch = agg._form_single_batch("llama3-8b", requests)
    assert batch is not None
    assert batch.max_seq_len == 100


def test_batch_creates_numpy_tensors():
    import numpy as np
    agg = BatchAggregator()
    requests = [make_request(seq_len=64) for _ in range(3)]
    batch = agg._form_single_batch("llama3-8b", requests)
    assert batch is not None
    assert hasattr(batch.input_tensor, "shape")
    assert hasattr(batch.attention_mask, "shape")


@pytest.mark.asyncio
async def test_submit_and_stats():
    agg = BatchAggregator()
    await agg.start()
    for _ in range(5):
        await agg.submit(make_request())
    await asyncio.sleep(0.05)
    stats = agg.stats
    assert stats["requests_processed"] >= 0
    await agg.stop()


@pytest.mark.asyncio
async def test_dispatch_callback_called():
    dispatched = []

    def on_dispatch(batch: BatchedRequest):
        dispatched.append(batch)

    config = BatchAggregatorConfig(max_batch_size=4, max_wait_ms=10)
    agg = BatchAggregator(config=config, dispatch_fn=on_dispatch)
    await agg.start()
    for _ in range(4):
        await agg.submit(make_request(seq_len=64))
    await asyncio.sleep(0.1)
    await agg.stop()


def test_inference_request_sequence_length():
    req = InferenceRequest(
        request_id="r1", model_id="m",
        input_ids=[1, 2, 3, 4, 5],
    )
    assert req.sequence_length == 5
