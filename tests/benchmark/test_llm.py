import time

import httpx
import openai
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

from dma_bench.calls import call_stage, record_calls
from dma_bench.llm import LLMTimeoutError, ResilientChatOpenAI, is_transient

REQUEST = httpx.Request("POST", "http://localhost/v1/chat/completions")


def build(**overrides):
    settings = {
        "model": "test-model",
        "api_key": "test",
        "base_url": "http://localhost/v1",
        "backoff_initial_s": 0.0,
        "call_deadline_s": 5.0,
    }
    return ResilientChatOpenAI(**{**settings, **overrides})


def chunk(text="", *, finish=None, usage=None):
    return ChatGenerationChunk(
        message=AIMessageChunk(content=text, usage_metadata=usage),
        generation_info={"finish_reason": finish} if finish else None,
    )


def reply(text="Enterprise", *, finish="stop", usage=None):
    return [chunk(text, finish=finish, usage=usage)]


def status_error(status):
    response = httpx.Response(status, request=REQUEST)
    return openai.APIStatusError("provider said no", response=response, body=None)


@pytest.fixture
def scripted(monkeypatch):
    """Replace the HTTP stream with a script, one step per attempt.

    A step is a list of chunks, an exception to raise, or a callable taking the
    attempt's kwargs and returning the chunks. Returns the kwargs each attempt
    was sent with.
    """
    sent = []

    def install(*steps):
        script = list(steps)

        def fake_stream(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG001
            sent.append(kwargs)
            step = script.pop(0)
            if isinstance(step, BaseException):
                raise step
            yield from step(kwargs) if callable(step) else step

        monkeypatch.setattr(ChatOpenAI, "_stream", fake_stream)
        return sent

    return install


def test_a_call_streams_and_returns_the_whole_reply(scripted):
    scripted([chunk("Enter"), chunk("prise", finish="stop")])

    assert build().invoke("Which plan?").content == "Enterprise"


def test_a_stalled_stream_is_abandoned_at_the_deadline(scripted):
    def stall(_kwargs):
        yield chunk("thinking")
        time.sleep(2)
        yield chunk(" too late", finish="stop")

    scripted(stall)
    started = time.monotonic()

    with pytest.raises(LLMTimeoutError):
        build(call_deadline_s=0.2, retries=0).invoke("Which plan?")

    assert time.monotonic() - started < 1.5


def test_a_transient_failure_is_retried(scripted):
    sent = scripted(openai.APIConnectionError(request=REQUEST), reply())

    assert build().invoke("Which plan?").content == "Enterprise"
    assert len(sent) == 2


def test_a_permanent_failure_is_not_retried(scripted):
    sent = scripted(status_error(400), reply())

    with pytest.raises(openai.APIStatusError):
        build(retries=3).invoke("Which plan?")

    assert len(sent) == 1


def test_retries_stop_at_their_count(scripted):
    sent = scripted(*[status_error(503)] * 3, reply())

    with pytest.raises(openai.APIStatusError):
        build(retries=2).invoke("Which plan?")

    assert len(sent) == 3


def test_a_truncated_reply_is_asked_again_with_a_grown_budget(scripted):
    def by_budget(kwargs):
        budget = kwargs["extra_body"]["max_tokens"]
        return reply(finish="length" if budget < 400 else "stop")

    sent = scripted(by_budget, by_budget, by_budget)

    build(max_output_tokens=100, max_output_tokens_cap=400).invoke("Which plan?")

    assert [kwargs["extra_body"]["max_tokens"] for kwargs in sent] == [100, 200, 400]


def test_a_reply_still_truncated_at_the_cap_is_returned_and_recorded(scripted):
    scripted(reply("Enter", finish="length"), reply("Enter", finish="length"))

    with record_calls() as log, call_stage("judge:lexical"):
        result = build(max_output_tokens=100, max_output_tokens_cap=200).invoke("q")

    assert result.content == "Enter"
    (record,) = log.records
    assert record.outcome == "truncated"
    assert record.attempts == 2
    assert record.max_output_tokens == 200
    assert record.stage == "judge:lexical"


def test_without_a_cap_a_truncated_reply_is_not_asked_again(scripted):
    sent = scripted(reply(finish="length"), reply())

    build(max_output_tokens=100).invoke("Which plan?")

    assert len(sent) == 1


def test_the_budgets_travel_in_the_request_body(scripted):
    sent = scripted(reply())

    build(
        max_output_tokens=4096,
        reasoning_max_tokens=1024,
        extra_body={"provider": {"sort": "throughput"}},
    ).invoke("Which plan?")

    assert sent[0]["extra_body"] == {
        "provider": {"sort": "throughput"},
        "max_tokens": 4096,
        "reasoning": {"max_tokens": 1024},
    }


def test_a_timeout_is_retried_and_the_call_records_both_attempts(scripted):
    def stall(_kwargs):
        time.sleep(1)
        yield chunk("too late", finish="stop")

    scripted(stall, reply())

    with record_calls() as log:
        build(call_deadline_s=0.2).invoke("Which plan?")

    (record,) = log.records
    assert record.outcome == "ok"
    assert record.attempts == 2


def test_a_failed_call_is_recorded_before_it_raises(scripted):
    scripted(status_error(401))

    with record_calls() as log, pytest.raises(openai.APIStatusError):
        build().invoke("Which plan?")

    (record,) = log.records
    assert record.outcome == "error"
    assert "provider said no" in (record.error or "")


def test_token_usage_is_recorded_reasoning_included(scripted):
    usage = {
        "input_tokens": 120,
        "output_tokens": 40,
        "total_tokens": 160,
        "output_token_details": {"reasoning": 25},
    }
    scripted(reply(usage=usage))

    with record_calls() as log:
        build().invoke("Which plan?")

    (record,) = log.records
    assert (record.input_tokens, record.output_tokens, record.reasoning_tokens) == (
        120,
        40,
        25,
    )


def test_calls_outside_a_record_block_are_not_kept(scripted):
    scripted(reply())

    build().invoke("Which plan?")

    with record_calls() as log:
        pass
    assert log.records == []


def test_the_client_gets_no_retry_layer_of_its_own():
    model = build(max_retries=5, idle_timeout_s=7.0)

    assert model.max_retries == 0
    assert model.request_timeout.read == 7.0


def test_a_reasoning_cap_at_or_above_the_budget_is_refused():
    with pytest.raises(ValueError, match="reasoning_max_tokens"):
        build(max_output_tokens=1024, reasoning_max_tokens=1024)


def test_a_cap_below_the_first_budget_is_refused():
    with pytest.raises(ValueError, match="max_output_tokens_cap"):
        build(max_output_tokens=1024, max_output_tokens_cap=512)


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (LLMTimeoutError("deadline"), True),
        (openai.APITimeoutError(request=REQUEST), True),
        (openai.APIConnectionError(request=REQUEST), True),
        (httpx.RemoteProtocolError("peer closed connection"), True),
        (status_error(429), True),
        (status_error(502), True),
        (status_error(400), False),
        (status_error(401), False),
        (
            openai.APIError("provider failed mid-stream", request=REQUEST, body=None),
            True,
        ),
        (ValueError("not a transport problem"), False),
    ],
)
def test_only_failures_a_new_attempt_can_fix_are_transient(error, transient):
    assert is_transient(error) is transient
