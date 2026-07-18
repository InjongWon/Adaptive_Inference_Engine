from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    name: str
    concurrency: int
    requests: int
    max_tokens: int
    temperature: float
    top_p: float
    prompt: str


SHORT_PROMPT = "Explain KV cache in three concise sentences."
LONG_PROMPT = (
    "You are reviewing an LLM inference service. Explain how continuous batching, "
    "PagedAttention, request scheduling, and GPU memory pressure interact. " * 32
)

WORKLOADS = {
    "smoke": Workload("smoke", 1, 2, 32, 0.0, 1.0, SHORT_PROMPT),
    "throughput": Workload("throughput", 16, 64, 128, 0.7, 0.95, SHORT_PROMPT),
    "long_context": Workload("long_context", 8, 32, 128, 0.7, 0.95, LONG_PROMPT),
}

# TODO: Add a mixed-length workload to expose head-of-line blocking and scheduler behavior.
# TODO: Replace repeated text with tokenizer-controlled prompt lengths (128/512/2048 tokens).
