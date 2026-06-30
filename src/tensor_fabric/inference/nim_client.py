from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)


@dataclass
class NIMCompletionRequest:
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class NIMCompletionResponse:
    request_id: str
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool = False
    gpu_device: int = 0


@dataclass
class NIMModelInfo:
    model_id: str
    endpoint: str
    vram_mb: float
    loaded_on_gpus: list[int]
    requests_per_second: float = 0.0
    avg_latency_ms: float = 0.0


class NIMClient:
    """
    Client for Nvidia NIM (Nvidia Inference Microservices).

    NIM wraps optimized model engines (TensorRT-LLM, vLLM) behind
    a standard OpenAI-compatible API. Tensor Fabric integrates with NIM
    at the Inference Fabric layer, adding:

    - KV-cache awareness: route to NIM instance that has the KV-cache
    - Batch pre-aggregation: send already-batched tensors to NIM
    - GPU-state routing: control plane selects the NIM endpoint dynamically

    This client is used by InferenceMesh to dispatch batched requests
    to the appropriate NIM microservice.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "not-needed-for-local",
        timeout_s: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_s
        self._client: httpx.AsyncClient | None = None
        self._model_registry: dict[str, NIMModelInfo] = {}
        self._request_count = 0
        self._total_latency_ms = 0.0

    async def __aenter__(self) -> "NIMClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.2, max=1))
    async def complete(
        self,
        request: NIMCompletionRequest,
        endpoint_override: str | None = None,
    ) -> NIMCompletionResponse:
        t0 = time.monotonic()
        url = f"{endpoint_override or self._base_url}/v1/chat/completions"

        payload = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "X-Tensor-Fabric-Request-Id": request.request_id,
                },
            )
            response.raise_for_status()
            data = response.json()

            elapsed_ms = (time.monotonic() - t0) * 1000
            self._request_count += 1
            self._total_latency_ms += elapsed_ms

            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})

            return NIMCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                content=choice.get("message", {}).get("content", ""),
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_ms=elapsed_ms,
            )
        finally:
            if not self._client:
                await client.aclose()

    async def stream(
        self,
        request: NIMCompletionRequest,
        endpoint_override: str | None = None,
    ) -> AsyncIterator[str]:
        url = f"{endpoint_override or self._base_url}/v1/chat/completions"
        payload = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }

        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            async with client.stream("POST", url, json=payload, headers={
                "Authorization": f"Bearer {self._api_key}",
            }) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(chunk)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue
        finally:
            if not self._client:
                await client.aclose()

    async def list_models(self) -> list[str]:
        url = f"{self._base_url}/v1/models"
        client = self._client or httpx.AsyncClient(timeout=30.0)
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []
        finally:
            if not self._client:
                await client.aclose()

    async def health_check(self, endpoint: str | None = None) -> bool:
        url = f"{endpoint or self._base_url}/health"
        client = self._client or httpx.AsyncClient(timeout=5.0)
        try:
            response = await client.get(url)
            return response.status_code == 200
        except Exception:
            return False
        finally:
            if not self._client:
                await client.aclose()

    def register_model(self, info: NIMModelInfo) -> None:
        self._model_registry[info.model_id] = info
        log.info(
            "nim_client.model_registered",
            model=info.model_id,
            endpoint=info.endpoint,
            vram_mb=info.vram_mb,
        )

    def get_endpoint(self, model_id: str) -> str | None:
        info = self._model_registry.get(model_id)
        return info.endpoint if info else None

    @property
    def stats(self) -> dict:
        return {
            "requests": self._request_count,
            "avg_latency_ms": round(
                self._total_latency_ms / max(self._request_count, 1), 1
            ),
            "registered_models": list(self._model_registry.keys()),
        }
