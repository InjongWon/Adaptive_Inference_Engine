from dataclasses import dataclass


@dataclass(frozen = True)
class Workload:
    name:str
    concurrency: int
    requests: int
    max_tokens:int
    temperature:float
    top_p:float
    prompt:str
    
SHORT_PROMPT = "explain KV cache in three concise sentence"
SMOKE = Workload( # baseline test   
    name="smoke",
    concurrency =1,
    requests = 2,
    max_tokens =32,
    temperature =0.0,
    top_p = 1.0,
    prompt = SHORT_PROMPT,
)

THROUGHPUT = Workload(# server's limit
    name = "throughput",
    concurrency = 16,
    requests = 128,
    max_tokens = 256,
    temperature =0.7,
    top_p = 0.95,
    prompt = SHORT_PROMPT
)

LONG_PROMPT = ( #long prompt affecting performance 
    "You are reviewing an LLM inference service. Explain how continuous batching, "
    "PagedAttention, request scheduling, and GPU memory pressure interact. " * 32
)
LONG_CONTEXT = Workload(
    name="long_context",
    concurrency=8,
    requests=32,
    max_tokens=128,
    temperature=0.7,
    top_p=0.95,
    prompt=LONG_PROMPT,
)
BASELINE_C1 = Workload(
    name="baseline_c1",
    concurrency=1,
    requests=32,
    max_tokens=128,
    temperature=0.0,
    top_p=1.0,
    prompt=SHORT_PROMPT,
)

BASELINE_C8 = Workload(
    name="baseline_c8",
    concurrency=8,
    requests=64,
    max_tokens=128,
    temperature=0.0,
    top_p=1.0,
    prompt=SHORT_PROMPT,
)

BASELINE_C32 = Workload(
    name="baseline_c32",
    concurrency=32,
    requests=128,
    max_tokens=128,
    temperature=0.0,
    top_p=1.0,
    prompt=SHORT_PROMPT,
)

WORKLOADS = {
    "smoke": SMOKE,
    "throughput": THROUGHPUT,
    "long_context": LONG_CONTEXT,
    "baseline_c1": BASELINE_C1,
    "baseline_c8": BASELINE_C8,
    "baseline_c32": BASELINE_C32,
}