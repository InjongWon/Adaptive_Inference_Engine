from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    seed: int | None = None
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    request_latency_s: float
    model: str
