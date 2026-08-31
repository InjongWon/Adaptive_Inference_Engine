from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from mini_vllm.request import Request, RequestStatus


class SchedulingMode(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclass
class ScheduledRequest:
    request: Request
    mode: SchedulingMode
    num_tokens: int


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledRequest] = field(default_factory=list)

    @property
    def requests(self) -> list[Request]:
        return [item.request for item in self.scheduled]

    @property
    def prefill_requests(self) -> list[Request]:
        return [
            item.request
            for item in self.scheduled
            if item.mode == SchedulingMode.PREFILL
        ]

    @property
    def decode_requests(self) -> list[Request]:
        return [
            item.request
            for item in self.scheduled
            if item.mode == SchedulingMode.DECODE
        ]

    @property
    def total_tokens(self) -> int:
        return sum(item.num_tokens for item in self.scheduled)


class Scheduler:
    """
    continuous-batching scheduler.

    Responsibilities:
    - Maintain waiting and running requests.
    - Admit new requests while capacity exists.
    - Prioritize already-running decode requests.
    - Schedule prefills for newly admitted requests.
    - Enforce both sequence and token budgets.
    - Support chunked prefill.
    - Remove finished requests immediately so new work can enter.

    This intentionally models the scheduling concepts used by
    production inference runtimes without reproducing vLLM/SGLang
    internals one-for-one.
    """

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_prefill_tokens_per_step: int | None = None,
    ) -> None:
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be greater than 0")

        if max_num_batched_tokens <= 0:
            raise ValueError(
                "max_num_batched_tokens must be greater than 0"
            )

        if (
            max_prefill_tokens_per_step is not None
            and max_prefill_tokens_per_step <= 0
        ):
            raise ValueError(
                "max_prefill_tokens_per_step must be greater than 0"
            )

        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_prefill_tokens_per_step = (
            max_prefill_tokens_per_step
            or max_num_batched_tokens
        )

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []


        self.num_computed_prompt_tokens: dict[int, int] = {}

    def add_request(self, request: Request) -> None:
        if request.finished:
            raise ValueError(
                "cannot add an already finished request"
            )

        if self.contains(request.request_id):
            raise ValueError(
                f"duplicate request_id: {request.request_id}"
            )

        request.status = RequestStatus.WAITING

        self.waiting.append(request)

        self.num_computed_prompt_tokens[request.request_id] = 0

    def contains(self, request_id: int) -> bool:
        return any(
            request.request_id == request_id
            for request in self.waiting
        ) or any(
            request.request_id == request_id
            for request in self.running
        )

    def _remove_finished_requests(self) -> list[Request]:
        finished: list[Request] = []
        still_running: list[Request] = []

        for request in self.running:
            if request.finished:
                finished.append(request)
            else:
                still_running.append(request)

        self.running = still_running

        for request in finished:
            self.num_computed_prompt_tokens.pop(
                request.request_id,
                None,
            )

        return finished

    def _admit_waiting_requests(self) -> None:
        while (
            self.waiting
            and len(self.running) < self.max_num_seqs
        ):
            request = self.waiting.popleft()

            request.status = RequestStatus.RUNNING

            self.running.append(request)

    def _prompt_tokens_remaining(
        self,
        request: Request,
    ) -> int:
        computed = self.num_computed_prompt_tokens[
            request.request_id
        ]

        return max(
            0,
            len(request.prompt_tokens) - computed,
        )

    def _needs_prefill(
        self,
        request: Request,
    ) -> bool:
        return self._prompt_tokens_remaining(request) > 0

    def _schedule_decode(
        self,
        request: Request,
        token_budget: int,
    ) -> int:
        if token_budget <= 0:
            return 0

        # Autoregressive decode processes one new token per step.
        return 1

    def _schedule_prefill(
        self,
        request: Request,
        token_budget: int,
    ) -> int:
        if token_budget <= 0:
            return 0

        remaining = self._prompt_tokens_remaining(request)

        if remaining <= 0:
            return 0

        return min(
            remaining,
            token_budget,
            self.max_prefill_tokens_per_step,
        )

    def schedule(self) -> SchedulerOutput:
        """
        Policy:
        1. Remove finished requests.
        2. Admit waiting requests into free sequence slots.
        3. Schedule decode tokens for active decoded requests first.
        4. Spend remaining token budget on prefills.
        """

        self._remove_finished_requests()
        self._admit_waiting_requests()

        output = SchedulerOutput()

        token_budget = self.max_num_batched_tokens

        for request in self.running:
            if token_budget <= 0:
                break

            if self._needs_prefill(request):
                continue

            num_tokens = self._schedule_decode(
                request,
                token_budget,
            )

            if num_tokens == 0:
                continue

            output.scheduled.append(
                ScheduledRequest(
                    request=request,
                    mode=SchedulingMode.DECODE,
                    num_tokens=num_tokens,
                )
            )

            token_budget -= num_tokens

        for request in self.running:
            if token_budget <= 0:
                break

            if not self._needs_prefill(request):
                continue

            num_tokens = self._schedule_prefill(
                request,
                token_budget,
            )

            if num_tokens == 0:
                continue

            output.scheduled.append(
                ScheduledRequest(
                    request=request,
                    mode=SchedulingMode.PREFILL,
                    num_tokens=num_tokens,
                )
            )

            token_budget -= num_tokens

        return output

    def mark_prefill_computed(
        self,
        request_id: int,
        num_tokens: int,
    ) -> None:
        if num_tokens <= 0:
            raise ValueError(
                "num_tokens must be greater than 0"
            )

        if request_id not in self.num_computed_prompt_tokens:
            raise KeyError(
                f"unknown request_id: {request_id}"
            )

        request = self.get_running_request(request_id)

        current = self.num_computed_prompt_tokens[
            request_id
        ]

        updated = current + num_tokens

        if updated > len(request.prompt_tokens):
            raise ValueError(
                "computed prompt tokens exceed prompt length"
            )

        self.num_computed_prompt_tokens[
            request_id
        ] = updated

    def get_running_request(
        self,
        request_id: int,
    ) -> Request:
        for request in self.running:
            if request.request_id == request_id:
                return request

        raise KeyError(
            f"request {request_id} is not running"
        )

    def mark_finished(
        self,
        request_id: int,
    ) -> None:
        request = self.get_running_request(request_id)

        request.status = RequestStatus.FINISHED

    def preempt(
        self,
        request_id: int,
    ) -> None:
  

        request = self.get_running_request(request_id)

        self.running.remove(request)

        request.status = RequestStatus.WAITING

        self.waiting.appendleft(request)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    @property
    def empty(self) -> bool:
        return not self.waiting and not self.running