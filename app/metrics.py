from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "gateway_requests_total", "Requests received by the learning gateway", ["status"]
)
INFLIGHT = Gauge("gateway_inflight_requests", "Requests currently in flight")
LATENCY = Histogram(
    "gateway_request_latency_seconds",
    "Gateway end-to-end request latency",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)
