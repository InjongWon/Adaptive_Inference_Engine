import json
from collections.abc import AsyncIterator

import httpx

from app.config import settings
from app.schemas import GenerateRequest


class VLLMClient:
    """Minimal client for vLLM's OpenAI-compatible API.
    
    OpenAI request payloads
    send synchrnous completion requests
    stream tokens response incrementaly
    hid HTTP implementation details 
    """
    def __init__(self) -> None:
        self.base_url = settings.vllm_base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {settings.vllm_api_key}"
        }
        self.timeout = httpx.Timeout(settings.request_timeout_s)
        
        
    
    def build_payload(self, req:GenerateRequest)->dict:
        """
        converting a validated GenerateRequest into JSON paybload for OpenAI API
        """
        
        load = {
            'model': settings.model_name,
            'prompt': req.prompt,
            'max_tokens': req.max_tokens,
            'temperature': req.temperature,
            'top_p': req.top_p,
            'stream': req.stream,
        }
        if req.seed is not None:
            load['seed'] = req.seed
        return load

    async def complete(self,req:GenerateRequest)->dict:
        """
        Send non streaming inference reqeust to the vLLM server
        
        It executes a single text generation request by translating validated GenerateRequest into HTTP reqwuest sending to 
        vLLM OpenAI API returns JSON response
        
        input:
        req: GenerateRequest
        output:
        dict: parsed JSON response from the server
        """
        
        payload = self.build_payload(req)
        
        # HTTP client using context manger to use connection
        
        async with httpx.AsyncClient(
            base_url = self.base_url,
            headers = self.headers,
            timeout= self.timeout,
        ) as client:
            
            # /POST send the inference requestthen wait receive
            response = await client.post('/v1/completions', json=payload)
            
            response.raise_for_status()
            
            return response.json()
    
    async def stream(self, req:GenerateRequest) -> AsyncIterator[str]:
        """
        Send a streaming inference request
        """
        payload = self.build_payload(req)
        
        
        payload['stream'] = True
        
        # HTTP /POST
        async with httpx.AsyncClient(
            base_url = self.base_url,
            headers = self.headers,
            timeout=self.timeout,
        ) as client, client.stream( #TCP connections 
            "POST", #open connection
            "/v1/completions",
            json=payload,
            
        ) as response: # in streaming context
            response.raise_for_status() 
            
            async for line in response.aiter_lines(): # read lines 
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break 
                
                #load data
                chunk = json.loads(data)
                text = chunk["choices"][0].get("text", "")
                
                if text:
                    yield text 
                    
        
        