"""Tests for GPU topology discovery — mock mode (no GPU required)."""
import pytest
from tensor_fabric.common.gpu_topology import (
    GPUTopology, GPUTopologyDiscovery, GPUNode, GPULink, LinkType, get_topology
)


def test_mock_topology_has_8_gpus():
    disc = GPUTopologyDiscovery()
    disc.initialize()
    topo = disc.refresh()
    assert len(topo.nodes) == 8


def test_mock_topology_nvswitch():
    disc = GPUTopologyDiscovery()
    disc.initialize()
    topo = disc.refresh()
    assert topo.nvswitch_present is True


def test_mock_topology_all_healthy():
    disc = GPUTopologyDiscovery()
    disc.initialize()
    topo = disc.refresh()
    for node in topo.nodes.values():
        assert node.is_healthy


def test_gpu_node_vram_used():
    node = GPUNode(
        device_id=0, uuid="gpu-0", name="H100",
        vram_total_mb=81920, vram_free_mb=40960,
        compute_util_pct=50.0, memory_util_pct=50.0,
        temperature_c=60.0, power_draw_w=300.0,
        sm_count=132, cuda_capability=(9, 0),
    )
    assert node.vram_used_mb == 40960
    assert node.vram_utilization_pct == pytest.approx(50.0, abs=0.1)


def test_gpu_node_free_vram_score():
    high = GPUNode(
        device_id=0, uuid="g0", name="H100",
        vram_total_mb=81920, vram_free_mb=80000,
        compute_util_pct=5.0, memory_util_pct=5.0,
        temperature_c=50.0, power_draw_w=200.0,
        sm_count=132, cuda_capability=(9, 0),
    )
    low = GPUNode(
        device_id=1, uuid="g1", name="H100",
        vram_total_mb=81920, vram_free_mb=10000,
        compute_util_pct=90.0, memory_util_pct=90.0,
        temperature_c=80.0, power_draw_w=700.0,
        sm_count=132, cuda_capability=(9, 0),
    )
    assert high.free_vram_score > low.free_vram_score


def test_topology_best_gpu_for_model():
    disc = GPUTopologyDiscovery()
    disc.initialize()
    topo = disc.refresh()
    gpu = topo.best_gpu_for_model(model_vram_mb=1000)
    assert gpu is not None
    assert gpu in topo.nodes


def test_topology_best_gpu_returns_none_when_oom():
    disc = GPUTopologyDiscovery()
    disc.initialize()
    topo = disc.refresh()
    gpu = topo.best_gpu_for_model(model_vram_mb=999999999)
    assert gpu is None


def test_topology_gpus_with_model():
    disc = GPUTopologyDiscovery()
    disc.initialize()
    topo = disc.refresh()
    topo.nodes[0].loaded_models.add("llama3-8b")
    topo.nodes[2].loaded_models.add("llama3-8b")
    result = topo.gpus_with_model("llama3-8b")
    assert 0 in result
    assert 2 in result
    assert len(result) == 2


def test_topology_total_free_vram():
    disc = GPUTopologyDiscovery()
    disc.initialize()
    topo = disc.refresh()
    total = topo.total_free_vram_mb()
    assert total > 0


def test_get_nvlink_peers():
    disc = GPUTopologyDiscovery()
    disc.initialize()
    topo = disc.refresh()
    peers = topo.get_nvlink_peers(0)
    assert len(peers) == 7


def test_iter_healthy_gpus():
    disc = GPUTopologyDiscovery()
    disc.initialize()
    topo = disc.refresh()
    healthy = list(topo.iter_healthy_gpus())
    assert len(healthy) == 8
