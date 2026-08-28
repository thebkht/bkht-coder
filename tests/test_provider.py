"""Provider streaming, transport normalization, and usage accounting."""

import json

import pytest

from bkht.coder.parsing import ToolCall
from bkht.coder.provider import (
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
    Chunk,
    OllamaProvider,
    ProviderError,
    build,
    collect,
)


def test_collect_joins_content_and_usage():
    reply = collect(
        [
            Chunk(content="Hel"),
            Chunk(content="lo"),
            Chunk(done=True, prompt_tokens=120, completion_tokens=8),
        ]
    )
    assert reply.content == "Hello"
    assert reply.prompt_tokens == 120
    assert reply.completion_tokens == 8
    assert reply.tool_calls == []


def test_collect_parses_tool_calls_out_of_content():
    reply = collect([Chunk(content='{"name": "read_file", "arguments": {"path": "a.py"}}')])
    assert [c.name for c in reply.tool_calls] == ["read_file"]


def test_collect_accepts_native_tool_calls_too():
    reply = collect([Chunk(content="", tool_calls=[ToolCall("grep", {"pattern": "x"})])])
    assert [c.name for c in reply.tool_calls] == ["grep"]


def test_reply_prose_excludes_the_call():
    reply = collect(
        [Chunk(content='Looking.\n{"name": "read_file", "arguments": {"path": "a.py"}}')]
    )
    assert reply.prose == "Looking."


def test_parse_line_reads_usage_and_native_calls():
    provider = OllamaProvider()
    chunk = provider._parse_line(
        json.dumps(
            {
                "message": {
                    "content": "hi",
                    "tool_calls": [
                        {"function": {"name": "grep", "arguments": {"pattern": "x"}}}
                    ],
                },
                "done": True,
                "prompt_eval_count": 42,
                "eval_count": 3,
            }
        )
    )
    assert chunk.content == "hi"
    assert chunk.done is True
    assert chunk.prompt_tokens == 42
    assert [c.name for c in chunk.tool_calls] == ["grep"]


def test_parse_line_ignores_blank_and_garbage():
    provider = OllamaProvider()
    assert provider._parse_line("") is None
    assert provider._parse_line("not json") is None


def test_parse_line_raises_on_server_error():
    provider = OllamaProvider()
    with pytest.raises(ProviderError):
        provider._parse_line(json.dumps({"error": "model not found"}))


def test_native_string_arguments_are_decoded():
    provider = OllamaProvider()
    chunk = provider._parse_line(
        json.dumps(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "grep", "arguments": '{"pattern": "x"}'}}
                    ],
                }
            }
        )
    )
    assert chunk.tool_calls[0].arguments == {"pattern": "x"}


def test_num_ctx_is_always_sent():
    # The 2048 default silently truncates; this must never be omitted.
    provider = OllamaProvider(num_ctx=32768)
    assert provider.num_ctx == 32768


@pytest.mark.live
def test_live_streaming_completion():
    provider = OllamaProvider()
    if not provider.available():
        pytest.skip("Ollama is not reachable")
    reply = collect(provider.chat([{"role": "user", "content": "Say OK and nothing else."}]))
    assert reply.content.strip()
    assert reply.prompt_tokens


# --- hardening --------------------------------------------------------------


def test_a_too_small_num_ctx_is_refused_at_construction():
    # Ollama's 2048 default truncates silently; accepting it hides the failure.
    with pytest.raises(ValueError, match="too small to be useful"):
        OllamaProvider(num_ctx=2048)


def test_connect_timeout_is_much_shorter_than_read():
    # A dead server must fail in seconds; a loaded 14b legitimately takes
    # minutes to produce its first token.
    provider = OllamaProvider()
    assert provider.timeout.connect <= 10
    assert provider.timeout.read >= 60


def test_an_unreachable_host_fails_fast_with_a_useful_message():
    import time

    provider = OllamaProvider(host="http://127.0.0.1:9")
    started = time.time()
    with pytest.raises(ProviderError, match="cannot reach Ollama"):
        list(provider.chat([{"role": "user", "content": "hi"}]))
    assert time.time() - started < 15


def test_available_is_false_for_a_dead_host():
    assert OllamaProvider(host="http://127.0.0.1:9").available() is False


def test_temperature_defaults_low_for_tool_calls():
    # Ollama's own default of 0.8 is for prose. Every tool call here is a JSON
    # object that has to be exactly right, so the default is pulled down.
    assert OllamaProvider().temperature == DEFAULT_TEMPERATURE
    assert 0 < DEFAULT_TEMPERATURE < 0.8


def test_temperature_can_be_turned_off_entirely():
    assert OllamaProvider(temperature=None).temperature is None


def test_for_review_pins_temperature_to_zero():
    from bkht.coder.provider import for_review

    original = OllamaProvider(model="m", num_ctx=8192)
    review = for_review(original)
    assert review.temperature == 0.0
    assert review.model == "m" and review.num_ctx == 8192
    assert (
        original.temperature == DEFAULT_TEMPERATURE
    ), "the original must not be mutated"


def test_for_review_leaves_other_providers_alone():
    from bkht.coder.provider import for_review

    class Other:
        model = "x"

    other = Other()
    assert for_review(other) is other


def test_the_default_window_can_hold_a_file_and_still_think():
    """8192 is the fastest measured number and the wrong default.

    A real turn is a conversation, not one completion. At 8192 this project's
    own cli.py is ~85% of the window, so a turn reads a file, frees context to
    make room, loses the file, and reads it again until it runs out of
    iterations. The default pays about ten seconds a turn to be able to finish.
    """
    assert DEFAULT_NUM_CTX == 16384
    assert OllamaProvider().num_ctx == DEFAULT_NUM_CTX


def test_build_returns_the_named_backend():
    provider = build("ollama", model="m", host="http://h", num_ctx=8192)
    assert isinstance(provider, OllamaProvider)
    assert (provider.model, provider.num_ctx) == ("m", 8192)


def test_build_names_what_is_available_when_the_backend_is_unknown():
    with pytest.raises(ProviderError, match="ollama"):
        build("codex", model="m")
