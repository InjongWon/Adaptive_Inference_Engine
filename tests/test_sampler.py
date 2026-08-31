import torch,pytest

from mini_vllm.sampler import Sampler, SamplingParams

def test_greedy_sampling():
    sampler = Sampler()

    logits = torch.tensor([
        [1.0, 5.0, 2.0],
        [8.0, 2.0, 3.0],
    ])

    params = SamplingParams(
        temperature=0,
    )

    tokens = sampler.sample(logits, params)

    assert tokens.tolist() == [1, 0]

def test_top_k_sampling():
    sampler = Sampler()
    logits = torch.tensor([
        [2.0,3.0,5.0,6.0],
        [4.0,7.0,8.0,9.0],
    ]
    )
    params = SamplingParams(
        top_k =1
    )
    tokens = sampler.sample(logits, params)
    print(tokens)
    assert tokens.tolist() == [3, 3]
    # python -m pytest tests/test_sampler.py -v

def test_top_p_sampling():
    sampler = Sampler()

    logits = torch.tensor([
        [10.0, 1.0, 0.0],
        [0.0, 1.0, 10.0],
    ])

    params = SamplingParams(
        top_p=0.5
    )

    tokens = sampler.sample(logits, params)

    assert tokens.tolist() == [0, 2]

def test_negative_temperature_raises():
    sampler = Sampler()

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
    ])

    params = SamplingParams(
        temperature=-1.0
    )

    with pytest.raises(ValueError):
        sampler.sample(logits, params)
        
def test_top_k_zero_raises():
    sampler = Sampler()

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
    ])

    params = SamplingParams(
        top_k=0
    )

    with pytest.raises(ValueError):
        sampler.sample(logits, params)


def test_top_k_larger_than_vocab_raises():
    sampler = Sampler()

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
    ])

    params = SamplingParams(
        top_k=4
    )

    with pytest.raises(ValueError):
        sampler.sample(logits, params)

def test_top_p_zero_raises():
    sampler = Sampler()

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
    ])

    params = SamplingParams(
        top_p=0.0
    )

    with pytest.raises(ValueError):
        sampler.sample(logits, params)


def test_top_p_greater_than_one_raises():
    sampler = Sampler()

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
    ])

    params = SamplingParams(
        top_p=1.1
    )

    with pytest.raises(ValueError):
        sampler.sample(logits, params)
        
def test_top_k_with_ties_keeps_exactly_k():
    sampler = Sampler()

    logits = torch.tensor([
        [8.0, 8.0, 8.0],
    ])

    params = SamplingParams(
        top_k=1
    )

    tokens = sampler.sample(logits, params)

    assert tokens.numel() == 1
    assert tokens.item() in [0, 1, 2]