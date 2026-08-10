from prometheus_client import Counter, Gauge, Histogram
from app.metrics import INFLIGHT, LATENCY, REQUESTS

def test_metric_types():
    
    assert isinstance(REQUESTS, Counter)
    assert isinstance(INFLIGHT, Gauge)
    assert isinstance(LATENCY, Histogram)
    
def test_request_counter():
    assert REQUESTS._labelnames == ("status",)
    
def test_metric_names():
    assert REQUESTS._name == "gateway_requests"
    assert INFLIGHT._name == "gateway_inflight_requests"
    assert LATENCY._name == "latency_gateway_request"