"""Model calls with a bound on how long any one of them can take.

The benchmark's long tails were not slow models, they were unbounded calls. The
client timeout is a *read* timeout — it trips on silence, not on duration — so a
provider that keeps the connection alive, or a model that generates slowly and
never stops, is never cut off; the client's own retries then multiply that by
three, and the judge's by two more. A reply with no token budget, from a model
that reasons before it answers, can run for minutes and still come back empty.

`ResilientChatOpenAI` is `ChatOpenAI` with those holes closed and nothing else
changed — `bind_tools` and `with_structured_output` work as they always did:

- Every call streams, on a worker thread, and the caller waits at most
  `call_deadline_s` for it. Past that the stream is abandoned and closed — which
  closes the connection — and the call raises `LLMTimeoutError`. Silence is
  caught sooner, by the read timeout, which is `idle_timeout_s`.
- There is one retry layer, this one. The openai client is built with
  `max_retries=0`; a transient failure (timeout, dropped connection, 429, 5xx)
  is retried here, `retries` times, with exponential backoff and jitter.
- Each call carries a token budget. A reply that hits it (`finish_reason ==
  "length"`) is asked again with the budget grown by `length_growth`, up to
  `max_output_tokens_cap`; one that still hits it at the cap is returned as it
  is and recorded as truncated. Asking again with the same budget would be
  pointless at temperature zero: the same prompt truncates the same way.
- `reasoning_max_tokens` caps the reasoning share of that budget, through
  OpenRouter's `reasoning` request field.

Every call is recorded through `dma_bench.calls`, so the next outlier can be read
off `result.json` rather than guessed at.

Only the synchronous path is bounded. The benchmark never calls a model from
async code, and `ainvoke` behaves exactly as it does on `ChatOpenAI`.
"""

from __future__ import annotations

import contextvars
import queue
import random
import threading
import time
from typing import TYPE_CHECKING, Any, Literal, Self

import httpx
import openai
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_openai import ChatOpenAI
from pydantic import model_validator

from dma_bench.calls import record_call
from dma_bench.schema import CallRecord

if TYPE_CHECKING:
    from collections.abc import Generator

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

__all__ = [
    "EmptyReplyError",
    "LLMTimeoutError",
    "ResilientChatOpenAI",
    "is_transient",
]

_RETRYABLE_STATUS = frozenset({408, 409, 429})


class LLMTimeoutError(TimeoutError):
    """Raised when one model call outlives its deadline."""


class EmptyReplyError(RuntimeError):
    """Raised when a stream ends without producing a single chunk."""


_TIMEOUTS = (LLMTimeoutError, openai.APITimeoutError, httpx.TimeoutException)


def is_transient(exc: BaseException) -> bool:
    """Whether a failed call is worth sending again.

    Args:
        exc: What the call raised.

    Returns:
        `True` for timeouts, dropped connections, empty replies, 408/409/429 and
        5xx responses, and errors the server sent halfway through a stream —
        typically the provider behind a router failing, which a new attempt
        usually routes around. `False` for anything the same request would hit
        again: a bad request, bad credentials, an unknown model, a prompt that
        does not fit the context window.
    """
    if isinstance(exc, ContextOverflowError):
        return False
    if isinstance(
        exc,
        LLMTimeoutError
        | EmptyReplyError
        | httpx.TransportError
        | openai.APIConnectionError,
    ):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500
    return isinstance(exc, openai.APIError)


class ResilientChatOpenAI(ChatOpenAI):
    """`ChatOpenAI` with a hard deadline, one retry layer and a growing token budget.

    Takes every `ChatOpenAI` argument, plus the ones below. `max_retries` and
    `timeout` are set from them and cannot be overridden: a second retry layer
    or a longer read timeout underneath would bring the unbounded worst case
    straight back.

    Attributes:
        call_deadline_s: The most one attempt may take, stream included.
        idle_timeout_s: The most the server may stay silent — the connect and
            read timeout of the HTTP client.
        connect_timeout_s: The most establishing a connection may take.
        retries: Extra attempts after a transient failure.
        backoff_initial_s: Wait before the first retry; doubled on each one.
        backoff_max_s: The longest wait between retries.
        max_output_tokens: Token budget of the first attempt. `None` sends no
            budget at all and disables the escalation.
        max_output_tokens_cap: The largest budget an escalation may reach.
            `None` means no escalation: a truncated reply is returned at once.
        length_growth: Factor the budget grows by after a truncated reply.
        token_param: The request field the budget is sent as. `ChatOpenAI`
            renames its own `max_tokens` to `max_completion_tokens`, which not
            every OpenAI-compatible server reads; the budget is therefore sent
            through `extra_body` under this name. `max_tokens` suits
            OpenRouter and local servers; OpenAI's reasoning models want
            `max_completion_tokens`.
        reasoning_max_tokens: Cap on reasoning tokens, sent as OpenRouter's
            `reasoning.max_tokens`. Must be below `max_output_tokens`, since
            providers count reasoning against the output budget. `None` leaves
            the provider's default.
    """

    call_deadline_s: float = 120.0
    idle_timeout_s: float = 45.0
    connect_timeout_s: float = 10.0
    retries: int = 2
    backoff_initial_s: float = 2.0
    backoff_max_s: float = 30.0
    max_output_tokens: int | None = 4096
    max_output_tokens_cap: int | None = None
    length_growth: float = 2.0
    token_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"  # noqa: S105 - a request field name, not a secret
    reasoning_max_tokens: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _one_retry_layer(cls, values: Any) -> Any:
        """Build the client with no retries of its own and the idle timeout."""
        if not isinstance(values, dict):
            return values
        fields = cls.model_fields
        idle = values.get("idle_timeout_s", fields["idle_timeout_s"].default)
        connect = values.get("connect_timeout_s", fields["connect_timeout_s"].default)
        values = {
            key: value for key, value in values.items() if key != "request_timeout"
        }
        return {
            **values,
            "max_retries": 0,
            "timeout": httpx.Timeout(idle, connect=connect),
            # Every call goes through `_generate`, which streams on its own terms;
            # letting LangChain pick `_stream` would skip the deadline.
            "disable_streaming": True,
            "stream_usage": values.get("stream_usage", True),
        }

    @model_validator(mode="after")
    def _budgets_are_consistent(self) -> Self:
        """Reject budgets that could never be honoured."""
        problems = []
        if self.length_growth <= 1:
            problems.append(f"length_growth must exceed 1, got {self.length_growth}")
        budget, cap = self.max_output_tokens, self.max_output_tokens_cap
        if budget is not None and cap is not None and cap < budget:
            problems.append(
                f"max_output_tokens_cap ({cap}) is below max_output_tokens ({budget})"
            )
        reasoning = self.reasoning_max_tokens
        if reasoning is not None and budget is not None and reasoning >= budget:
            problems.append(
                f"reasoning_max_tokens ({reasoning}) must be below "
                f"max_output_tokens ({budget}), which it counts against"
            )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Run one call: deadline per attempt, retries, budget escalation."""
        started = time.monotonic()
        budget = self.max_output_tokens
        attempts = failures = 0
        while True:
            attempts += 1
            try:
                result = self._stream_within_deadline(
                    messages, stop, run_manager, self._with_budget(kwargs, budget)
                )
            except Exception as exc:
                if failures >= self.retries or not is_transient(exc):
                    self._record(started, attempts, budget, None, exc)
                    raise
                failures += 1
                time.sleep(self._backoff(failures))
                continue

            generation = result.generations[0]
            grown = self._grown(budget)
            if _finish_reason(generation) == "length" and grown is not None:
                budget = grown
                continue
            self._record(started, attempts, budget, generation, None)
            return result

    def _with_budget(
        self, kwargs: dict[str, Any], budget: int | None
    ) -> dict[str, Any]:
        """Return the call's kwargs with the token and reasoning budgets set."""
        extra = dict(kwargs.get("extra_body") or self.extra_body or {})
        if budget is not None:
            extra[self.token_param] = budget
        if self.reasoning_max_tokens is not None:
            extra["reasoning"] = {
                **extra.get("reasoning", {}),
                "max_tokens": self.reasoning_max_tokens,
            }
        return {**kwargs, "extra_body": extra} if extra else kwargs

    def _grown(self, budget: int | None) -> int | None:
        """Return the next budget after a truncated reply, or `None` at the cap."""
        cap = self.max_output_tokens_cap
        if budget is None or cap is None or budget >= cap:
            return None
        return min(max(budget + 1, int(budget * self.length_growth)), cap)

    def _backoff(self, failures: int) -> float:
        """Return the wait before retry number `failures`."""
        delay = min(self.backoff_max_s, self.backoff_initial_s * 2 ** (failures - 1))
        return delay * random.uniform(0.5, 1.0)

    def _stream_within_deadline(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        run_manager: CallbackManagerForLLMRun | None,
        kwargs: dict[str, Any],
    ) -> ChatResult:
        """Stream one attempt on a worker thread and wait for it, up to the deadline.

        The worker owns the stream from start to finish — a generator cannot be
        driven from two threads — and hands chunks over a queue, so the waiting
        side can give up at the deadline without waiting for the next chunk.
        When it does, the worker closes the stream at the next chunk it gets,
        or when the read timeout fires, whichever comes first.

        Raises:
            LLMTimeoutError: If the attempt outlived `call_deadline_s`.
            EmptyReplyError: If the stream ended without a chunk.
        """
        # Creating the generator runs none of it: the request is only sent once
        # the worker starts iterating.
        chunks = super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        inbox: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        abandoned = threading.Event()
        worker = threading.Thread(
            target=contextvars.copy_context().run,
            args=(_pump, chunks, inbox, abandoned),
            name="dma-bench-model-call",
            daemon=True,
        )
        worker.start()
        try:
            collected = _drain(inbox, self.call_deadline_s)
        finally:
            abandoned.set()
        if not collected:
            msg = "the stream ended without a single chunk"
            raise EmptyReplyError(msg)
        return generate_from_stream(iter(collected))

    def _record(
        self,
        started: float,
        attempts: int,
        budget: int | None,
        generation: ChatGeneration | None,
        error: Exception | None,
    ) -> None:
        """Record the finished call, successful or not."""
        usage = (
            getattr(generation.message, "usage_metadata", None) if generation else None
        )
        usage = usage or {}
        details = usage.get("output_token_details") or {}
        if error is not None:
            outcome = "timeout" if isinstance(error, _TIMEOUTS) else "error"
        elif _finish_reason(generation) == "length":
            outcome = "truncated"
        else:
            outcome = "ok"
        record_call(
            CallRecord(
                model=self.model_name or "",
                seconds=round(time.monotonic() - started, 2),
                attempts=attempts,
                outcome=outcome,
                finish_reason=_finish_reason(generation) if generation else None,
                max_output_tokens=budget,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                reasoning_tokens=details.get("reasoning", 0),
                error=None if error is None else repr(error),
            )
        )


def _pump(
    chunks: Generator[ChatGenerationChunk, None, None],
    inbox: queue.SimpleQueue[tuple[str, Any]],
    abandoned: threading.Event,
) -> None:
    """Drive a stream on this thread, handing each chunk to the waiting caller.

    Stops at the first chunk after the caller gave up, and closes the stream on
    the way out, which closes the connection under it.
    """
    try:
        for chunk in chunks:
            if abandoned.is_set():
                return
            inbox.put(("chunk", chunk))
    except Exception as exc:
        # Handed to the waiting caller, which raises it on its own thread.
        inbox.put(("error", exc))
    else:
        inbox.put(("done", None))
    finally:
        chunks.close()


def _drain(
    inbox: queue.SimpleQueue[tuple[str, Any]], deadline_s: float
) -> list[ChatGenerationChunk]:
    """Collect a stream's chunks from its worker, giving up at the deadline.

    Raises:
        LLMTimeoutError: If the stream has not ended `deadline_s` seconds in.
    """
    deadline = time.monotonic() + deadline_s
    collected: list[ChatGenerationChunk] = []
    while True:
        try:
            kind, item = inbox.get(timeout=max(deadline - time.monotonic(), 0))
        except queue.Empty:
            msg = f"model call outlived its {deadline_s:g}s deadline"
            raise LLMTimeoutError(msg) from None
        if kind == "error":
            raise item
        if kind == "done":
            return collected
        collected.append(item)


def _finish_reason(generation: ChatGeneration | None) -> str | None:
    """Return why the provider stopped generating, wherever it was reported."""
    if generation is None:
        return None
    info = generation.generation_info or {}
    return info.get("finish_reason") or generation.message.response_metadata.get(
        "finish_reason"
    )
