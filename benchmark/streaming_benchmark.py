import asyncio
import json
import time

import httpx


async def measure_stream(base_url: str, prompt: str) -> dict:
    start = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    completion_tokens: int | None = None
    chunks: list[str] = []

    async with (
        httpx.AsyncClient(timeout=180.0) as client,
        client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/completions",
            headers={"Authorization": "Bearer local-token"},
            json={
                "model": "Qwen/Qwen3-8B",
                "prompt": prompt,
                "max_tokens": 128,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response,
    ):
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue

            raw = line[6:]

            if raw == "[DONE]":
                break

            now = time.perf_counter()

            event = json.loads(raw)

            usage = event.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")

            choices = event.get("choices")
            text = choices[0].get("text", "") if choices else ""

            if text:
                if first_token_at is None:
                    first_token_at = now

                last_token_at = now
                chunks.append(text)

    tpot_s = None

    if (
        completion_tokens is not None
        and completion_tokens > 1
        and first_token_at is not None
        and last_token_at is not None
    ):
        tpot_s = (last_token_at - first_token_at) / (
            completion_tokens - 1
        )

    return {
        "ttft_s": (
            None
            if first_token_at is None
            else first_token_at - start
        ),
        "tpot_s": tpot_s,
        "output_tokens": completion_tokens,
        "latency_s": time.perf_counter() - start,
        "text": "".join(chunks),
    }


if __name__ == "__main__":
    result = asyncio.run(
        measure_stream(
            "http://localhost:8000",
            "Explain continuous batching.",
        )
    )
    print(result)