import torch
from transformers import AutoModelForCausalLM,PreTrainedTokenizerBase
from mini_vllm.request import Request

class ModelRunner:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device:str = "cpu"
    )->None:
            self.model = model.to(device)
            self.model.eval()
            
            self.tokenizer = tokenizer
            self.device = device
            
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is None:
                    raise ValueError(
                        "Tokenizer should have padding token or EOS"
                    )
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_batch(
        self,
        requests:list[Request],
    )->tuple[torch.Tensor,torch.Tensor]:
        
        if not requests:
            raise ValueError("empty Batch")
        
        all_tokens = [
            request.prompt_tokens + request.output_tokens for request in requests
        ]
        
        max_length = max(len(sequence) for sequence in all_tokens) 
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise RuntimeError("Tokenizer possess no padding token")
        
        input_ids = [] # tensors batch of token IDs
        attention_masks =[] #defining which positons with padding or not
        
        for sequence in all_tokens:
            padding_length = max_length - len(sequence) 
            padded_sequence =(
                [pad_token_id] * padding_length + sequence
            )
            attention_mask = (
                [0] * padding_length + [1]*len(sequence)
            )
            input_ids.append(padded_sequence)
            attention_masks.append(attention_mask)
        return (
            torch.tensor(input_ids,
                         dtype=torch.long,
                         device =self.device),
            torch.tensor(
                attention_masks,
                dtype=torch.long,
                device= self.device,
            ),
        )
        
    def forward(self, requests:list[Request])->torch.Tensor:
        input_ids, attention_mask = self.prepare_batch(requests)
        
        with torch.inference_mode():
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask,)
        return outputs.logits[:,-1,:]
    


