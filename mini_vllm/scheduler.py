from collections import deque
from mini_vllm.request import Request, RequestStatus
from dataclasses import dataclass
from enum import Enum, auto

#define whether its prefill or decode incase of 1 token 
class SchedulingMode(Enum):
    PREFILL = auto()
    DECODE = auto()
    
@dataclass
class ScheduledRequest:
    request:Request
    num_tokens: int
    mode: SchedulingMode


    

class Scheduler:
    def __init__(self, max_num_seqs:int, max_num_batched_tokens):
        
        
        self.waiting = deque()
        self.running = []
        
        if max_num_seqs <=0:
            raise ValueError(
                "should be greater than 0"
            )
        if max_num_batched_tokens <=0:
            raise ValueError("max_num_batched_tokens should be greater than 0 to handle tokens per request")
    
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_computed_prompt_tokens:dict[int,int] = {} # chunked prefill
    
    def add_request(self,request:Request):
        request.status = RequestStatus.WAITING
        self.waiting.append(request)
        self.num_computed_prompt_tokens[request.request_id] = 0
    
    def _prompt_tokens_remaining(self,request:Request,):
        computed = self.num_computed_prompt_tokens[request.request_id]
        return len(request.prompt_tokens) - computed
    
    def _check_prefill(self,request:Request):
        return self._prompt_tokens_remaining(request)>0
    
    def mark_prefill_computed(self,request:Request,num_tokens):
        
        if num_tokens <=0:
            raise ValueError("has to greater than 0")
        current = self.num_computed_prompt_tokens[request.request_id]
        updated = current + num_tokens
        if updated > len(request.prompt_tokens):
            raise ValueError("computed tokens cannot exceeds prompt lenght")
        
        self.num_computed_prompt_tokens[request.request_id] = updated 

    def schedule(self)->list[ScheduledRequest]:
        
        self.running =[request for request in self.running if request.status != RequestStatus.FINISHED]

        available = self.max_num_seqs - len(self.running)
        for _ in range(available):
            if not self.waiting:
                break
            request = self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            self.running.append(request)
            
        capable_tokens = self.max_num_batched_tokens
        scheduled: list[ScheduledRequest] = []
        
        for request in self.running:
            if capable_tokens == 0:
                break
            
            remaining = self._prompt_tokens_remaining(request)
            if remaining >0:
                mode = SchedulingMode.PREFILL
                num_tokens = min(remaining, capable_tokens)
                

            else:
                #decoding parts since reamining !>0 so we have prefill all tokens
                #need to autoregressively decode one token each 
                num_tokens =1
                mode = SchedulingMode.DECODE
            scheduled.append(
                ScheduledRequest(
                    request = request,
                    num_tokens = num_tokens,
                    mode = mode
                )
            )
            capable_tokens -= num_tokens
            
                
        return scheduled
