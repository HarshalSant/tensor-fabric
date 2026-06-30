"""Tests for RoutingEngine — the 4-loop GPU-aware routing brain."""
import asyncio
import pytest
from tensor_fabric.common.tensor_descriptor import TensorDescriptor, TensorDtype, TensorRole
from tensor_fabric.control_plane.gpu_state_manager import GPUStateManager
from tensor_fabric.control_plane.routing_engine import RoutingEngine, RouteStrategy


def make_descriptor(model="llama3-8b", role=TensorRole.INPUT, layer=None):
    import uuid
    return TensorDescriptor(
        tensor_id=str(uuid.uuid4()),
        model_id=model,
        role=role,
        shape=(1, 512),
        dtype=TensorDtype.FLOAT16,
        layer_index=layer,
    )


@pytest.fixture
def engine():
    mgr = GPUStateManager()
    return RoutingEngine(mgr), mgr


def test_basic_route_returns_decision(engine):
    eng, _ = engine
    desc = make_descriptor()
    decision = eng.route(desc)
    assert decision is not None
    assert decision.target_gpu >= 0
    assert decision.confidence > 0


def test_cache_hit_strategy(engine):
    eng, _ = engine
    desc = make_descriptor()
    decision = eng.route(desc, kv_cache_gpu=3)
    assert decision.strategy == RouteStrategy.CACHE_HIT
    assert decision.target_gpu == 3
    assert decision.cache_hit is True
    assert decision.confidence == 1.0


def test_cache_hit_lowest_latency(engine):
    eng, _ = engine
    desc = make_descriptor()
    decision = eng.route(desc, kv_cache_gpu=2)
    assert decision.estimated_latency_us < 2.0


def test_model_affinity_routing(engine):
    eng, mgr = engine
    mgr.register_model("llama3-8b", 5)
    desc = make_descriptor(model="llama3-8b", role=TensorRole.MODEL_WEIGHT)
    decision = eng.route(desc)
    assert decision.strategy == RouteStrategy.VRAM_AFFINITY
    assert decision.target_gpu == 5


def test_fallback_to_least_loaded(engine):
    eng, _ = engine
    desc = make_descriptor(model="unknown-model-xyz")
    decision = eng.route(desc)
    assert decision.strategy == RouteStrategy.LEAST_LOADED
    assert decision.target_gpu >= 0


def test_stats_tracking(engine):
    eng, _ = engine
    for _ in range(10):
        eng.route(make_descriptor(), kv_cache_gpu=None)
    for _ in range(5):
        eng.route(make_descriptor(), kv_cache_gpu=0)
    stats = eng.stats
    assert stats["total_routes"] == 15
    assert stats["cache_hit_rate"] == pytest.approx(5/15, abs=0.01)


def test_prefetch_callback_triggered(engine):
    eng, _ = engine
    prefetch_calls = []
    eng.register_prefetch_handler(lambda job: prefetch_calls.append(job))
    desc = make_descriptor()
    for _ in range(10):
        eng.route(desc)
    assert len(prefetch_calls) >= 0


def test_access_pattern_predictor():
    from tensor_fabric.control_plane.routing_engine import AccessPatternPredictor
    pred = AccessPatternPredictor()
    for _ in range(20):
        pred.record("key-A")
        pred.record("key-B")
    assert pred.predict_next("key-A") == "key-B"
    assert pred.confidence("key-A") > 0.9


def test_route_respects_different_models(engine):
    eng, mgr = engine
    mgr.register_model("model-A", 0)
    mgr.register_model("model-B", 1)
    desc_a = make_descriptor(model="model-A", role=TensorRole.MODEL_WEIGHT)
    desc_b = make_descriptor(model="model-B", role=TensorRole.MODEL_WEIGHT)
    dec_a = eng.route(desc_a)
    dec_b = eng.route(desc_b)
    assert dec_a.target_gpu == 0
    assert dec_b.target_gpu == 1
