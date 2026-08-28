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
    

    def prefill(self, requests:list[Request]):
        input_ids, attention_mask = self.prepare_batch(requests)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                )
        return outputs.logits[:,-1,:], outputs.past_key_values, attention_mask

    def decode(self, requests:list[Request], past_key_values,  attention_mask: torch.Tensor):
        
        
        if attention_mask.shape[0] != len(requests):
            raise ValueError(
            "attention mask batch size must match number of requests"
        )

        for request in requests:
            if not request.output_tokens:
                raise ValueError(
                f"request {request.request_id} has no token to decode"
            )
        input_ids = torch.tensor(
            [[request.output_tokens[-1]] for request in requests],
            dtype=torch.long,
            device= self.device)
        
        new_column = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=self.device,
        )
        attention_mask = torch.cat(
            [attention_mask, new_column],
            dim=1,
        )
        with torch.inference_mode():
            outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        return (
            outputs.logits[:,-1,:],outputs.past_key_values, attention_mask,
        )
    
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "hf-internal-testing/tiny-random-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    runner = ModelRunner(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )

    token_ids = tokenizer.encode(
        "Hello world",
        add_special_tokens=False,
    )

    request = Request(
        request_id=1,
        prompt_tokens=token_ids,
        max_tokens=10,
    )

    logits, past_key_values, attention_mask= runner.prefill([request])

    print("LOGITS TYPE:", type(logits))
    print("LOGITS SHAPE:", logits.shape)
    print("KV TYPE:", type(past_key_values))
    
    next_token_id = logits.argmax(dim=-1).item()

    print("NEXT TOKEN:", next_token_id)


    request.add_token(
        next_token_id,
        tokenizer.eos_token_id,
    )

    print("OUTPUT TOKENS:", request.output_tokens)
    print("KV SHAPE BEFORE:",past_key_values.layers[0].keys.shape,
    )
    decode_logits, past_key_values, attention_mask = runner.decode(
        [request],
        past_key_values,
        attention_mask,
    )

    print(
        "KV SHAPE AFTER:",
        past_key_values.layers[0].keys.shape,
    )

    print("DECODE LOGITS SHAPE:", decode_logits.shape)
    print("ATTENTION MASK SHAPE:", attention_mask.shape)
    print("KV TYPE AFTER DECODE:", type(past_key_values))