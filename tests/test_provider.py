"""Provider streaming, transport normalization, and usage accounting."""

import json

import pytest

from bkht.coder.parsing import ToolCall
from bkht.coder.provider import Chunk, OllamaProvider, ProviderError, collect


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
