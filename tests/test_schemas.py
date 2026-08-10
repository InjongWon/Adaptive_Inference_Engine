from app.schemas import GenerateRequest

def test_generate_request_uses_defaults():
    request = GenerateRequest(prompt = "Explain KV Cache")
    
    assert request.prompt == "Explain KV Cache"
    assert request.max_tokens == 128
    assert request.temperature == 0.7
    assert request.top_p == 0.95
    assert request.seed is None
    assert request.stream is False 