import torch
import pytest

from mini_vllm.model_runner import ModelRunner
from mini_vllm.request import Request
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    pad_token = "<pad>"
    eos_token = "<eos>"


def test_prepare_batch():
    runner = ModelRunner.__new__(ModelRunner)
    runner.tokenizer = FakeTokenizer()
    runner.device = "cpu"

    requests = [
        Request(
            request_id=1,
            prompt_tokens=[10, 20, 30, 40],
            max_tokens=10,
        ),
        Request(
            request_id=2,
            prompt_tokens=[50, 60],
            max_tokens=10,
        ),
        Request(
            request_id=3,
            prompt_tokens=[70, 80, 90],
            max_tokens=10,
        ),
    ]

    input_ids, attention_mask = runner.prepare_batch(requests)

    expected_input_ids = torch.tensor([
        [10, 20, 30, 40],
        [0,  0,  50, 60],
        [0,  70, 80, 90],
    ])

    expected_attention_mask = torch.tensor([
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    ])

    assert torch.equal(input_ids, expected_input_ids)
    assert torch.equal(attention_mask, expected_attention_mask)

    assert input_ids.shape == (3, 4)
    assert attention_mask.shape == (3, 4)
    
class FakeModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        batch_size, sequence_length = input_ids.shape

        logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size,
            device=input_ids.device,
        )

        return SimpleNamespace(logits=logits)
    
def test_forward_returns_next_token_logits():
    runner = ModelRunner(
        model=FakeModel(vocab_size=100),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )

    requests = [
        Request(
            request_id=1,
            prompt_tokens=[10, 20, 30, 40],
            max_tokens=10,
        ),
        Request(
            request_id=2,
            prompt_tokens=[50, 60],
            max_tokens=10,
        ),
        Request(
            request_id=3,
            prompt_tokens=[70, 80, 90],
            max_tokens=10,
        ),
    ]

    logits = runner.forward(requests)

    assert logits.shape == (3, 100)

def test_forward_with_real_transformer():
    model_name = "hf-internal-testing/tiny-random-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    runner = ModelRunner(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )

    prompts = [
        "The capital of France is",
        "Machine learning is",
        "Hello",
    ]

    requests = []

    for i, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        requests.append(
            Request(
                request_id=i,
                prompt_tokens=token_ids,
                max_tokens=10,
            )
        )

    logits = runner.forward(requests)

    assert logits.shape[0] == 3
    assert logits.shape[1] == model.config.vocab_size
    

def test_prefill_then_decode():
    model_name = "hf-internal-testing/tiny-random-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    runner = ModelRunner(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )

    prompt_tokens = tokenizer.encode(
        "Prefill then Decode",
        add_special_tokens=False,
    )

    request = Request(
        request_id=1,
        prompt_tokens=prompt_tokens,
        max_tokens=10,
    )

    logits, past_key_values, attention_mask= runner.prefill([request])
    next_token = logits.argmax(dim=-1).item()
    request.add_token(
        next_token,
        tokenizer.eos_token_id,
    )
    prev_mask_length = attention_mask.shape[1]
    logits, past_key_values, attention_mask = runner.decode(
        [request],
        past_key_values,
        attention_mask,
    )
    assert logits.shape[0] == 1
    assert logits.shape[1] == model.config.vocab_size
    
    assert attention_mask.shape[1] == prev_mask_length + 1
    

def test_prefill_decode_updates_kv_cache():
    model_name = "hf-internal-testing/tiny-random-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    runner = ModelRunner(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )

    prompt_tokens = tokenizer.encode(
        "Hello world",
        add_special_tokens=False,
    )

    request = Request(
        request_id=1,
        prompt_tokens=prompt_tokens,
        max_tokens=10,
    )

    logits, past_key_values, attention_mask = runner.prefill(
        [request]
    )

    kv_length_before = past_key_values.layers[0].keys.shape[2]

    #greedy sampling
    next_token_id = logits.argmax(dim=-1).item()

    request.add_token(
        next_token_id,
        tokenizer.eos_token_id,
    )

    decode_logits, past_key_values, attention_mask = runner.decode(
        [request],
        past_key_values,
        attention_mask,
    )

    kv_length_after = past_key_values.layers[0].keys.shape[2]

    assert kv_length_after == kv_length_before + 1

    #Attention mask should match the updated KV length
    assert attention_mask.shape[1] == kv_length_after

    #One token distribution for one request
    assert decode_logits.shape == (
        1,
        model.config.vocab_size,
    )
    

def test_batched_prefill_then_decode():
    model_name = "hf-internal-testing/tiny-random-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    runner = ModelRunner(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )

    prompts = [
        "Prefill",
        "decode",
        "KV Cache",
    ]

    requests = [
        Request(
            request_id=i,
            prompt_tokens=tokenizer.encode(
                prompt,
                add_special_tokens=False,
            ),
            max_tokens=10,
        )
        for i, prompt in enumerate(prompts)
    ]

    #PREFILL all 3 requests together
    logits, past_key_values, attention_mask = runner.prefill(
        requests
    )

    kv_length_before = past_key_values.layers[0].keys.shape[2]

    #choose one token for EACH request
    next_token_ids = logits.argmax(dim=-1)

    assert next_token_ids.shape == (3,)

    #Store each generated token in its request
    for request, token_id in zip(requests, next_token_ids.tolist()):
        request.add_token(
            token_id,
            tokenizer.eos_token_id,
        )

    #Decode all 3 requests together
    decode_logits, past_key_values, attention_mask = runner.decode(
        requests,
        past_key_values,
        attention_mask,
    )

    kv_length_after = past_key_values.layers[0].keys.shape[2]

    #check cache grew by ONE position
    assert kv_length_after == kv_length_before + 1

    #only generated one token  per request
    assert decode_logits.shape == (
        3,
        model.config.vocab_size,
    )

    #Mask also grew by one position
    assert attention_mask.shape[1] == kv_length_after
    
def test_decode_empty_requests():
    runner = ModelRunner(
        model=FakeModel(vocab_size=100),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )
    attention_mask = torch.ones((1,3), dtype=torch.long)
    with pytest.raises(ValueError):
        runner.decode(
            [],
            past_key_values=None,
            attention_mask=attention_mask,
        )
    
def test_decode_empty_output_tokens():
    runner = ModelRunner(
        model = FakeModel(vocab_size=1000),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )
    attention_mask = torch.ones((1, 3), dtype=torch.long)
    request = Request(
        request_id=1,
        prompt_tokens=[10, 20, 30],
        max_tokens=10,
    )
    with pytest.raises(ValueError):
        runner.decode(
            [request],
            past_key_values=None,
            attention_mask=attention_mask,
        )