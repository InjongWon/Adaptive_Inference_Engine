from benchmark.metrics import RequestResult, summarize


def test_summarize_latency_and_throughput() -> None:
    results = [
        RequestResult(0, True, 1.0, 0.2, 10),
        RequestResult(1, True, 2.0, 0.3, 20),
    ]
    summary = summarize(results, duration_s=2.0)
    assert summary.successful_requests == 2
    assert summary.requests_per_second == 1.0
    assert summary.output_tokens_per_second == 15.0
    assert summary.mean_ttft_s == 0.25
