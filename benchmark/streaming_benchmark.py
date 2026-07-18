"""Starter for measuring TTFT and inter-token latency using SSE streaming."""

import asyncio
import json
import time

import httpx


async def measure_stream(base_url: str, prompt: str) -> dict:
    start = time.perf_counter()
    first_token_at: float | None = None
    event_times: list[float] = []
    chunks: list[str] = []

    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/completions",
            headers={"Authorization": "Bearer local-token"},
            json={"model": "Qwen/Qwen3-1.7B", "prompt": prompt, "max_tokens": 128, "stream": True},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                now = time.perf_counter()
                event = json.loads(raw)
                text = event["choices"][0].get("text", "")
                if text:
                    first_token_at = first_token_at or now
                    event_times.append(now)
                    chunks.append(text)

    # TODO: Token chunks are not guaranteed to map one-to-one to tokens.
    # Use returned usage or the model tokenizer for accurate TPOT calculations.
    intervals = [b - a for a, b in zip(event_times, event_times[1:])]
    return {
        "ttft_s": None if first_token_at is None else first_token_at - start,
        "mean_chunk_interval_s": None if not intervals else sum(intervals) / len(intervals),
        "latency_s": time.perf_counter() - start,
        "text": "".join(chunks),
    }


if __name__ == "__main__":
    print(asyncio.run(measure_stream("http://localhost:8000", "Explain continuous batching.")))
