import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from benchmark.metrics import RequestResult, summarize
from benchmark.workloads import WORKLOADS, Workload


async def run_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    base_url: str,
    workload: Workload,
    request_id: int,
) -> RequestResult:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(
                f"{base_url.rstrip('/')}/generate",
                json={
                    "prompt": workload.prompt,
                    "max_tokens": workload.max_tokens,
                    "temperature": workload.temperature,
                    "top_p": workload.top_p,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            latency = time.perf_counter() - started
            # TODO: Return real token counts from the gateway instead of estimating text length.
            estimated_tokens = max(1, len(data["text"].split()))
            return RequestResult(request_id, True, latency, None, estimated_tokens)
        except Exception as exc:  # benchmark should retain every failure
            return RequestResult(request_id, False, time.perf_counter() - started, None, None, str(exc))


async def run(base_url: str, workload: Workload) -> tuple[list[RequestResult], dict]:
    semaphore = asyncio.Semaphore(workload.concurrency)
    timeout = httpx.Timeout(300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        started = time.perf_counter()
        results = await asyncio.gather(
            *[
                run_one(client, semaphore, base_url, workload, request_id)
                for request_id in range(workload.requests)
            ]
        )
        duration = time.perf_counter() - started
    return results, summarize(results, duration).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--workload", choices=WORKLOADS, default="smoke")
    parser.add_argument("--output", default="results/latest.json")
    args = parser.parse_args()

    results, summary = asyncio.run(run(args.base_url, WORKLOADS[args.workload]))
    payload = {"workload": args.workload, "summary": summary, "requests": [r.__dict__ for r in results]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
