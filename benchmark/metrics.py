from dataclasses import asdict, dataclass
from statistics import mean

import numpy as np


@dataclass
class RequestResult:
        request_id: int
        success: bool
        latency_s: float
        
        ttft_s: float | None = None
        output_tokens: int | None = None
        error: str | None = None
        
        # streaming metrics
        tpot_s: float | None = None
        itl_ms: float | None = None
        queue_time_s: float | None = None

@dataclass
class BenchmarkSummary:
    total_requests: int
    successful_requests: int
    duration_s: float
    requests_per_second: float
    output_tokens_per_second: float | None
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    mean_ttft_s: float | None
    
    def to_dict(self) -> dict:
        return asdict(self)

def summarize(results: list[RequestResult], duration_s:float)->BenchmarkSummary:
    successful =  [r for r in results if r.success]
    
    if not successful:
        raise ValueError("empty result")
    
    latencies = np.asarray([r.latency_s for r in successful], dtype=float)
    token_counts = [r.output_tokens for r in successful if r.output_tokens is not None]
    ttft_s = [r.ttft_s for r in successful if r.ttft_s is not None]
    
    return BenchmarkSummary(
        total_requests=len(results),
        successful_requests=len(successful),
        duration_s=duration_s,
        requests_per_second=len(successful) / duration_s,
        output_tokens_per_second=(sum(token_counts) / duration_s if token_counts else None),
        mean_latency_s=float(mean(latencies)),
        p50_latency_s=float(np.percentile(latencies, 50)),
        p95_latency_s=float(np.percentile(latencies, 95)),
        p99_latency_s=float(np.percentile(latencies, 99)),
        mean_ttft_s=(float(mean(ttft_s)) if ttft_s else None),
    )