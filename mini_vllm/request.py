from dataclasses import dataclass, field
from enum import Enum, auto

class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    
@dataclass
class Request:
    request_id : int
    prompt_tokens: list[int]
    max_tokens:int 
    
    output_tokens: list[int] = field(default_factory = list)
    status: RequestStatus = RequestStatus.WAITING
    
    def add_token(self,token_id:int, eos_token_id:int)->None:
        self.output_tokens.append(token_id)
        
        if (
            token_id == eos_token_id
            or len(self.output_tokens)>= self.max_tokens):
                self.status = RequestStatus.FINISHED
    @property
    def token_ids(self) -> list[int]:
        return self.prompt_tokens + self.output_tokens

    @property
    def finished(self)->bool:
        return self.status == RequestStatus.FINISHED