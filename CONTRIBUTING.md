# Contributing to Tensor Fabric

Thank you for your interest in contributing to Tensor Fabric!

## How to Contribute

### 1. Fork and Clone
```bash
git fork https://github.com/YOUR_USERNAME/tensor-fabric
git clone https://github.com/YOUR_FORK/tensor-fabric
cd tensor-fabric
pip install -e ".[all]"
pip install pytest pytest-asyncio
```

### 2. Run Tests Before Changing Anything
```bash
pytest tests/ -v
```

### 3. Make Your Changes

Follow these principles:
- **No CPU in the critical path** — if your change adds a CPU hop between Storage → Network → Inference, find a way to eliminate it
- **Map to real Nvidia APIs** — every new feature must reference a shipping Nvidia SDK (DOCA, Magnum IO, cuFile, NCCL, pynvml)
- **Benchmark before and after** — run `python benchmarks/e2e_benchmark.py` and include numbers in your PR

### 4. Submit a PR

PR description must include:
- Which layer(s) this touches (Storage / Network / Inference / Control Plane)
- Which Nvidia SDK/hardware it targets
- Benchmark delta (before vs after)

## High-Value Contribution Areas

### Real GPUDirect Storage (`storage/gpudirect_bridge.py`)
The `_dma_gpudirect` path currently falls back to host-pinned if `cufile` is missing.
If you have access to a system with `nvidia-fs` kernel module, test and validate Path-A.

### BlueField DPU DOCA Programs
The Inference Mesh proxy (`inference/inference_mesh.py`) currently runs on the host CPU.
The production target is a DOCA-based C program on the BlueField DPU ARM cores.

### Spectrum-X P4 Programs
`network/tensor_router.py`'s `InNetworkCompute` class simulates what should be P4 programs
running on the Spectrum-X ASIC. Contributions of real P4 programs are highly valued.

### Benchmarks on Real H100/A100 Hardware
If you have access to DGX H100 or SuperPOD, run the benchmark suite and open an issue
with your results. This is the most impactful contribution you can make.

## Code Style

- Python 3.11+ type hints everywhere
- `structlog` for all logging (not `print`)
- `async/await` for all I/O
- Pydantic models for all data structures crossing API boundaries
- No silent fallbacks — log the path taken (`log.info("using path-B: host-pinned")`)

## Questions?

Open a GitHub Discussion or post in the Nvidia Developer Forums thread.
