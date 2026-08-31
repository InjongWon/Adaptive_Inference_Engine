
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
    semaphore:asyncio.Semaphore,
    base_url:str,
    workload:Workload,
    request_id: int,
)-> RequestResult:
    async with semaphore: #wait until its within concurrency.
        started = time.perf_counter()
        
        try:
            response = await client.post( # gpt <enter: sending request>
                f"{base_url.rstrip('/')}/generate",
                json = {
                    "prompt": workload.prompt,
                    "max_tokens":workload.max_tokens,
                    "temperature": workload.temperature,
                    "top_p": workload.top_p,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            latency = time.perf_counter() - started
            
            output_tokens = data.get("output_tokens")
            
            return RequestResult(
                request_id = request_id,
                success =True,
                latency_s = latency,
                output_tokens = output_tokens,
            )
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_s=time.perf_counter() - started,
                error=str(exc),
            )
            
async def run(
    base_url: str,
    workload: Workload,
) -> tuple[list[RequestResult], dict]:
    semaphore = asyncio.Semaphore(workload.concurrency)

    timeout = httpx.Timeout(300.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        benchmark_started = time.perf_counter()

        results = await asyncio.gather(
            *[
                run_one(
                    client=client,
                    semaphore=semaphore,
                    base_url=base_url,
                    workload=workload,
                    request_id=request_id,
                )
                for request_id in range(workload.requests)
            ]
        )

        duration_s = time.perf_counter() - benchmark_started

    summary = summarize(
        results=results,
        duration_s=duration_s,
    )

    return results, summary.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the Adaptive LLM Serving Gateway."
    )

    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL of the inference gateway.",
    )

    parser.add_argument(
        "--workload",
        choices=WORKLOADS,
        default="smoke",
        help="Benchmark workload to execute.",
    )

    parser.add_argument(
        "--output",
        default="results/latest.json",
        help="Path where benchmark results will be written.",
    )

    args = parser.parse_args()

    workload = WORKLOADS[args.workload]

    results, summary = asyncio.run(
        run(
            base_url=args.base_url,
            workload=workload,
        )
    )

    payload = {
        "workload": args.workload,
        "configuration": {
            "concurrency": workload.concurrency,
            "requests": workload.requests,
            "max_tokens": workload.max_tokens,
            "temperature": workload.temperature,
            "top_p": workload.top_p,
        },
        "summary": summary,
        "requests": [
            result.__dict__
            for result in results
        ],
    }

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            payload,
            indent=2,
        )
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

if __name__ == "__main__":
    main()