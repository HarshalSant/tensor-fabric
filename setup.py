from setuptools import setup, find_packages

setup(
    name="tensor-fabric",
    version="1.0.0",
    description="World's First GPU-Native Unified Infrastructure Stack",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Tensor Fabric Contributors",
    license="Apache-2.0",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pynvml>=11.5.0",
        "httpx>=0.27.0",
        "numpy>=1.26.0",
        "pydantic>=2.6.0",
        "pydantic-settings>=2.2.0",
        "prometheus-client>=0.20.0",
        "structlog>=24.1.0",
        "tenacity>=8.2.0",
        "click>=8.1.0",
        "rich>=13.7.0",
    ],
    extras_require={
        "gpu": [
            "cupy-cuda12x>=13.0.0",
            "numba>=0.59.0",
        ],
        "triton": [
            "tritonclient[all]>=2.44.0",
        ],
        "all": [
            "cupy-cuda12x>=13.0.0",
            "numba>=0.59.0",
            "tritonclient[all]>=2.44.0",
            "grpcio>=1.62.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tensor-fabric=tensor_fabric.control_plane.api:run_server",
            "tf-bench=benchmarks.e2e_benchmark:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
