# public access inference as a web API
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from prometheus_client import make_asgi_app

from app.config import settings
from app.metrics import INFLIGHT, LATENCY, REQUESTS
from app.schemas import GenerateRequest, GenerateResponse
from app.vllm_client import VLLMClient

app = FastAPI(
    title="LLM Serving Gateway", 
    version="0.1.0"
) # web API
app.mount("/metrics", make_asgi_app())
client = VLLMClient()


@app.post("/generate", response_model = GenerateResponse) #returned response   GenerateResponse Schema
async def generate(req:GenerateRequest):
    
    if req.stream:
        return StreamingResponse(
            content = client.stream(req),
            statu_code = 200,
            media_type = "text/plain",
        )
    start = time.perf_conter()
    INFLIGHT.inc()
    result = await client.complete(req)
    
    
    try:
        result = await client.complete(req)
        
        latency = time.perf_counter() - start
        LATENCY.observe(latency)
        REQUESTS.labels(status= "success").inc()
        
        return GenerateResponse(
            text=result["choices"][0]["text"],
            request_latency_s=0.0,
            model=result.get("model", settings.model_name),
            output_tokens=result.get("usage", {}).get("completion_tokens"),
        )
    except (httpx.HTTPError, KeyError) as exc:
        REQUESTS.labels(status="error").inc()
        
        raise HTTPException(
            status_code = 502,
            detail = str(exc),
        ) from exc 
        
    finally:
        INFLIGHT.dec()

    