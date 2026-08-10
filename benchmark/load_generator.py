# 1. imports
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from benchmark.metrics import RequestResult, summarize  
from benchmark.workloads import WORKLOADS, Workload


# 2. run_one()
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
        except Exception as exc:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_s=time.perf_counter() - started,
                error=str(exc),
            )
            
# 3. concurrency control
# 4. run()
# 5. asyncio.gather()
# 6. benchmark duration
# 7. summary generation
# 8. command-line arguments
# 9. result serialization
# 10. file output
# 11. main entrypoint
# 12. later improvements:
#     real token counts
#     retries
#     per-request queue time
#     streaming benchmark separation