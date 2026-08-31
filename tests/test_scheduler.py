from mini_vllm.request import Request, RequestStatus
from mini_vllm.scheduler import (
    Scheduler,
    ScheduledRequest,
    SchedulingMode,
)


def make_request(
    request_id: int,
    prompt_length: int,
    max_tokens: int = 10,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_tokens=list(range(prompt_length)),
        max_tokens=max_tokens,
    )
    
def test_add_and_schedule_request():
    scheduler = Scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=10,
    )

    request = make_request(1, 5)

    scheduler.add_request(request)

    assert request.status == RequestStatus.WAITING
    assert list(scheduler.waiting) == [request]

    scheduled = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert scheduler.running == [request]

    assert len(scheduled) == 1
    assert scheduled[0].request == request
    assert scheduled[0].mode == SchedulingMode.PREFILL
    assert scheduled[0].num_tokens == 5
    
def test_max_num_seqs():
    scheduler = Scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=100,
    )

    a = make_request(1, 5)
    b = make_request(2, 5)
    c = make_request(3, 5)

    scheduler.add_request(a)
    scheduler.add_request(b)
    scheduler.add_request(c)

    scheduler.schedule()

    assert scheduler.running == [a, b]
    assert list(scheduler.waiting) == [c]
    

    
def test_chunked_prefill():
    scheduler = Scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=2,
    )

    request = make_request(1, 5)
    scheduler.add_request(request)

    scheduled = scheduler.schedule()

    assert scheduled[0].mode == SchedulingMode.PREFILL
    assert scheduled[0].num_tokens == 2

    scheduler.mark_prefill_computed(
        request,
        scheduled[0].num_tokens,
    )

    assert scheduler.num_computed_prompt_tokens[1] == 2

    scheduled = scheduler.schedule()

    assert scheduled[0].mode == SchedulingMode.PREFILL
    assert scheduled[0].num_tokens == 2

    scheduler.mark_prefill_computed(
        request,
        scheduled[0].num_tokens,
    )

    assert scheduler.num_computed_prompt_tokens[1] == 4

    scheduled = scheduler.schedule()

    assert scheduled[0].mode == SchedulingMode.PREFILL
    assert scheduled[0].num_tokens == 1