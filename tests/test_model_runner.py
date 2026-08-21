import torch

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