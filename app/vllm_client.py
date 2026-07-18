import json
from collections.abc import AsyncIterator

import httpx

from app.config import settings
from app.schemas import GenerateRequest


class VLLMClient:
    """Minimal client for vLLM's OpenAI-compatible API.

    Learning TODOs:
    1. Add cancellation propagation when the caller disconnects.
    2. Parse token usage and return prompt/output token counts.
    3. Add bounded retries only for safe transient failures.
    """

    def __init__(self) -> None:
        self.base_url = settings.vllm_base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {settings.vllm_api_key}"}
        self.timeout = httpx.Timeout(settings.request_timeout_s)

    def _payload(self, req: GenerateRequest) -> dict:
        payload = {
            "model": settings.model_name,
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": req.stream,
        }
        if req.seed is not None:
            payload["seed"] = req.seed
        return payload

    async def complete(self, req: GenerateRequest) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/completions",
                headers=self.headers,
                json=self._payload(req),
            )
            response.raise_for_status()
            return response.json()

    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        payload = self._payload(req)
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/completions",
                headers=self.headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    yield event["choices"][0].get("text", "")
