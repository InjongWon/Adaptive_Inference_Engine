# requests, current running, latency
# counter , Gauge, Histogram
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "gateway_requests_total",
    
    "Total generation requests served by gateway",
    ["status"],
)

LATENCY = Histogram(
    "latency_gateway_request",
    "e2e meausre of latency of generation requests",

)
INFLIGHT = Gauge(
    "gateway_inflight_requests",
    "Number of generation requests currently being processed.",

)