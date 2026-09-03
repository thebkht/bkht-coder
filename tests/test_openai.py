"""The OpenAI-compatible backend: SSE framing, tool-call deltas, conversion."""

import json

import pytest

from bkht.coder.openai import OpenAIProvider, _accumulate, _assemble, _convert
from bkht.coder.provider import ProviderError, build, collect


def sse(payload) -> str:
    return f"data: {json.dumps(payload)}"


def delta(**fields) -> str:
    return sse({"choices": [{"index": 0, "delta": fields}]})


def drain(provider, lines):
    """Every chunk the provider makes of ``lines``, as one stream would."""
    pending = {}
    chunks = [provider._parse_line(line, pending) for line in lines]
    return [c for c in chunks if c is not None], pending


@pytest.fixture
def provider():
    return OpenAIProvider(model="coder", host="http://localhost:8080")


# --- framing ---------------------------------------------------------------


def test_content_arrives_from_the_delta(provider):
    chunks, _ = drain(provider, [delta(content="Hel"), delta(content="lo")])
    assert "".join(chunk.content for chunk in chunks) == "Hello"


def test_blank_lines_and_keepalive_comments_say_nothing(provider):
    chunks, _ = drain(provider, ["", "   ", ": ping", "event: message"])
    assert chunks == []


def test_the_done_sentinel_is_not_json_and_is_not_treated_as_it(provider):
    # The literal that ends every stream. Parsed as JSON it is a list, and a
    # backend that forgets this reports a decode error on every completed turn.
    chunks, _ = drain(provider, ["data: [DONE]"])
    assert chunks == []


def test_a_truncated_payload_is_skipped_rather_than_fatal(provider):
    chunks, _ = drain(provider, ['data: {"choices": [{"delta": {"cont'])
    assert chunks == []


def test_usage_arrives_on_a_chunk_with_no_choices(provider):
    chunks, _ = drain(
        provider,
        [sse({"choices": [], "usage": {"prompt_tokens": 1200, "completion_tokens": 34}})],
    )
    assert (chunks[0].prompt_tokens, chunks[0].completion_tokens) == (1200, 34)


def test_finish_reason_marks_the_chunk_done(provider):
    chunks, _ = drain(provider, [sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})])
    assert chunks[0].done is True


def test_a_mid_stream_error_is_raised_not_swallowed(provider):
    # Some servers report a failure as an ordinary event rather than by closing
    # with a status, so the turn would otherwise end quietly with no answer.
    with pytest.raises(ProviderError, match="context length"):
        drain(provider, [sse({"error": {"message": "context length exceeded"}})])


# --- tool-call deltas ------------------------------------------------------


def test_arguments_are_assembled_across_deltas(provider):
    lines = [
        delta(tool_calls=[{"index": 0, "function": {"name": "read_file", "arguments": '{"pa'}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]),
    ]
    _, pending = drain(provider, lines)
    calls = _assemble(pending)
    assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "a.py"})]


def test_two_calls_keep_their_own_arguments():
    pending = {}
    _accumulate([{"index": 0, "function": {"name": "grep", "arguments": '{"pattern":'}}], pending)
    _accumulate([{"index": 1, "function": {"name": "glob", "arguments": '{"pattern":'}}], pending)
    _accumulate([{"index": 1, "function": {"arguments": ' "*.py"}'}}], pending)
    _accumulate([{"index": 0, "function": {"arguments": ' "x"}'}}], pending)
    assert [(c.name, c.arguments) for c in _assemble(pending)] == [
        ("grep", {"pattern": "x"}),
        ("glob", {"pattern": "*.py"}),
    ]


def test_a_call_whose_arguments_never_parsed_is_dropped():
    # Not passed on as an empty call: the loop answers a malformed call with a
    # correction the model can act on, and an empty one looks well-formed.
    assert _assemble({0: {"name": "read_file", "arguments": '{"path": "a.p'}}) == []


def test_a_call_with_no_name_is_dropped():
    assert _assemble({0: {"name": "", "arguments": "{}"}}) == []


def test_absent_arguments_mean_an_empty_object():
    calls = _assemble({0: {"name": "plan", "arguments": ""}})
    assert [(c.name, c.arguments) for c in calls] == [("plan", {})]


def test_a_whole_stream_collects_into_one_reply(provider):
    lines = [
        delta(content="Reading it.\n"),
        delta(tool_calls=[{"index": 0, "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]),
        sse({"choices": [], "usage": {"prompt_tokens": 90, "completion_tokens": 12}}),
        "data: [DONE]",
    ]
    chunks, pending = drain(provider, lines)
    reply = collect(chunks + [type(chunks[0])(done=True, tool_calls=_assemble(pending))])
    assert reply.prose == "Reading it."
    assert [(c.name, c.arguments) for c in reply.tool_calls] == [("read_file", {"path": "a.py"})]
    assert reply.prompt_tokens == 90


# --- message conversion ----------------------------------------------------


def test_the_tool_role_is_kept_rather_than_flattened():
    # Training and serving must assemble the same bytes. Qwen's chat template
    # renders a tool turn correctly, and a mismatch here is invisible until the
    # model starts emitting calls coder cannot parse.
    message = {"role": "tool", "name": "read_file", "content": "1\tx = 1"}
    assert _convert(message, vision=False) == message


def test_images_are_dropped_for_a_model_that_cannot_see():
    # A server that does not expect the content-part form rejects the whole
    # request, and losing the picture beats losing the turn.
    converted = _convert({"role": "user", "content": "look", "images": ["/tmp/a.png"]}, vision=False)
    assert converted == {"role": "user", "content": "look"}


def test_images_become_content_parts_when_the_model_can_see(tmp_path, monkeypatch):
    monkeypatch.setattr("bkht.coder.clipboard.encode", lambda path: "QUJD")
    converted = _convert(
        {"role": "user", "content": "look", "images": [str(tmp_path / "a.png")]}, vision=True
    )
    assert converted["content"][0] == {"type": "text", "text": "look"}
    assert converted["content"][1]["image_url"]["url"].endswith("QUJD")


def test_an_unreadable_image_costs_the_picture_not_the_turn(tmp_path, monkeypatch):
    def missing(path):
        raise OSError("gone")

    monkeypatch.setattr("bkht.coder.clipboard.encode", missing)
    converted = _convert(
        {"role": "user", "content": "look", "images": [str(tmp_path / "a.png")]}, vision=True
    )
    assert converted["content"] == [{"type": "text", "text": "look"}]


# --- wiring ----------------------------------------------------------------


def test_the_backend_is_reachable_by_name():
    assert isinstance(build("local"), OpenAIProvider)


def test_the_api_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("CODER_API_KEY", "secret")
    assert OpenAIProvider()._headers()["Authorization"] == "Bearer secret"


def test_no_key_means_no_authorization_header(monkeypatch):
    monkeypatch.delenv("CODER_API_KEY", raising=False)
    assert "Authorization" not in OpenAIProvider()._headers()


def test_a_trailing_slash_on_the_host_does_not_double_up():
    assert OpenAIProvider(host="http://box:8080/").host == "http://box:8080"


def test_review_pins_temperature_to_zero():
    from bkht.coder.provider import for_review

    review = for_review(OpenAIProvider(model="coder", temperature=0.7))
    assert review.temperature == 0.0 and review.model == "coder"
