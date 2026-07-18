import pytest
from pydantic import ValidationError

from app.schemas import GenerateRequest


def test_generate_request_defaults() -> None:
    req = GenerateRequest(prompt="hello")
    assert req.max_tokens == 128
    assert req.top_p == 0.95


def test_empty_prompt_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")
