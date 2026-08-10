# with having api request check.
# we can you Counter to count total api requests. (positive for internal usage only)
# use buckets to fill and reject if empty follow refill time/requests
from app.schemas import GenerateRequest, GenerateResponse
import pytest
from pydantic import ValidationError

def test_generate_request():
    request = GenerateRequest(prompt="explain continuous batching")
    
    assert request.prompt == "explain continuous batching"
    assert request.max_tokens == 128
    assert request.temperature ==0.7
    assert request.top_p == 0.5
    assert request.stream == True
    assert request.seed is None

def test_generate_request_rejects_blank_prompt():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="   ")