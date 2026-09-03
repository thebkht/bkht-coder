"""Backends that drive another agent's command line.

No subprocess is launched here. The two things worth pinning down are the flags
that turn an agent into a transport -- get one of those wrong and a frontier
model starts editing files behind coder's permission gate -- and the reading of
each tool's event stream. Both are pure given a fake process.
"""

import io
import subprocess

import pytest

from bkht.coder import external
from bkht.coder.external import ClaudeCodeProvider, CodexProvider, render
from bkht.coder.provider import (
    BACKENDS,
    CLAUDE_CODE_NUM_CTX,
    CODEX_NUM_CTX,
    DEFAULT_CLAUDE_CODE_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULTS,
    ProviderError,
    build,
    collect,
)

import json


# --- a process that never runs -----------------------------------------------


class Stdin(io.StringIO):
    """A pipe that remembers what was written after it is closed."""

    written = ""

    def close(self):
        self.written = self.getvalue()
        super().close()


class FakeProcess:
    def __init__(self, lines, returncode=0, stderr=""):
        self.stdin = Stdin()
        self.stdout = io.StringIO("".join(f"{line}\n" for line in lines))
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.killed = False

    def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


@pytest.fixture
def spawn(monkeypatch):
    """Capture the argv a provider would have run, and answer with fixed lines."""
    captured = {}

    def launcher(lines=(), returncode=0, stderr=""):
        def popen(argv, **kwargs):
            captured["argv"] = argv
            captured["env"] = kwargs.get("env", {})
            captured["process"] = FakeProcess(
                [json.dumps(line) if isinstance(line, dict) else line for line in lines],
                returncode=returncode,
                stderr=stderr,
            )
            return captured["process"]

        monkeypatch.setattr(subprocess, "Popen", popen)
        return captured

    monkeypatch.setattr(external.shutil, "which", lambda name: f"/usr/bin/{name}")
    return launcher


TASK = [
    {"role": "system", "content": "you are coder"},
    {"role": "user", "content": "fix it"},
]


# --- flattening the history --------------------------------------------------


def test_the_system_prompt_is_separated_from_the_conversation():
    system, prompt = render(TASK)
    assert system == "you are coder"
    assert prompt == "[user] fix it"


def test_several_system_messages_become_one():
    system, _ = render(
        [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "second"},
        ]
    )
    assert system == "first\n\nsecond"


def test_a_tool_result_is_labelled_with_the_tool_that_produced_it():
    _, prompt = render(
        [
            {"role": "user", "content": "read it"},
            {"role": "tool", "name": "read_file", "content": "def main():"},
            {"role": "assistant", "content": "found it"},
        ]
    )
    assert prompt.splitlines()[2] == "[read_file] def main():"
    assert prompt.splitlines()[4] == "[assistant] found it"


def test_an_unnamed_tool_result_is_still_labelled():
    _, prompt = render([{"role": "tool", "content": "out"}])
    assert prompt == "[tool] out"


def test_empty_messages_are_dropped():
    _, prompt = render(
        [
            {"role": "user", "content": "work"},
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": None},
        ]
    )
    assert prompt == "[user] work"


def test_a_history_with_nothing_left_in_it_says_so(spawn):
    spawn()
    with pytest.raises(ProviderError, match="nothing to send"):
        list(ClaudeCodeProvider().chat([{"role": "system", "content": "rules"}]))


# --- Claude Code -------------------------------------------------------------


def test_claude_code_runs_with_its_own_tools_switched_off(spawn):
    # The load-bearing flag. Without it Claude Code can edit files itself, and
    # an edit made that way never passes coder's permission gate or its
    # snapshot store -- so `/undo` could not take it back.
    captured = spawn()
    list(ClaudeCodeProvider(model="sonnet").chat(TASK))
    argv = captured["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    assert "--restricted" in argv
    assert "--strict-mcp-config" in argv
    assert "--disable-slash-commands" in argv


def test_claude_code_is_given_coders_system_prompt_and_model(spawn):
    captured = spawn()
    list(ClaudeCodeProvider(model="sonnet").chat(TASK))
    argv = captured["argv"]
    assert argv[0].endswith("claude")
    assert argv[argv.index("--system-prompt") + 1] == "you are coder"
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert captured["process"].stdin.written == "[user] fix it"


def test_claude_code_streams_text_deltas_and_ends_with_the_counts(spawn):
    spawn(
        lines=[
            {"type": "system", "subtype": "init"},
            _delta("he"),
            _delta("llo"),
            {
                "type": "result",
                "subtype": "success",
                "usage": {
                    "input_tokens": 2,
                    "cache_read_input_tokens": 300,
                    "cache_creation_input_tokens": 40,
                    "output_tokens": 9,
                },
            },
        ]
    )
    reply = collect(ClaudeCodeProvider().chat(TASK))
    assert reply.content == "hello"
    # Cached input is cheaper, not absent: it takes up the same room in the
    # window, which is the number the compactor is asking about.
    assert reply.prompt_tokens == 342
    assert reply.completion_tokens == 9


def test_claude_code_reports_a_failed_turn_rather_than_returning_nothing(spawn):
    spawn(lines=[{"type": "result", "is_error": True, "result": "Not logged in"}])
    with pytest.raises(ProviderError, match="Not logged in"):
        list(ClaudeCodeProvider().chat(TASK))


def _delta(text: str) -> dict:
    return {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
    }


# --- Codex -------------------------------------------------------------------


def test_codex_runs_read_only(spawn):
    # Codex has no switch that removes its shell, so the sandbox is the guard.
    captured = spawn()
    list(CodexProvider(model="gpt-5.5").chat(TASK))
    argv = captured["argv"]
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in argv
    assert "--ignore-user-config" in argv
    assert "--ignore-rules" in argv


def test_codex_carries_the_system_prompt_in_the_prompt(spawn):
    # There is no flag for it, so the instructions lead the conversation.
    captured = spawn()
    list(CodexProvider().chat(TASK))
    assert "--system-prompt" not in captured["argv"]
    assert captured["process"].stdin.written == "[user] you are coder\n\n[user] fix it"


def test_codex_emits_only_the_part_of_a_message_that_is_new(spawn):
    # Codex reports whole items and re-reports them as they grow. Emitting the
    # item each time would say everything twice.
    spawn(
        lines=[
            {"type": "thread.started"},
            {"type": "item.started", "item": {"id": "i0", "type": "agent_message", "text": "he"}},
            {"type": "item.updated", "item": {"id": "i0", "type": "agent_message", "text": "hell"}},
            {"type": "item.completed", "item": {"id": "i0", "type": "agent_message", "text": "hello"}},
            {"type": "turn.completed", "usage": {"input_tokens": 50, "output_tokens": 3}},
        ]
    )
    reply = collect(CodexProvider().chat(TASK))
    assert reply.content == "hello"
    assert (reply.prompt_tokens, reply.completion_tokens) == (50, 3)


def test_codex_ignores_items_that_are_not_the_answer(spawn):
    spawn(
        lines=[
            {"type": "item.completed", "item": {"id": "r", "type": "reasoning", "text": "hmm"}},
            {"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "done"}},
            {"type": "turn.completed", "usage": {}},
        ]
    )
    assert collect(CodexProvider().chat(TASK)).content == "done"


@pytest.mark.parametrize(
    "event",
    [
        {"type": "error", "message": "You've hit your usage limit."},
        {"type": "turn.failed", "error": {"message": "You've hit your usage limit."}},
    ],
)
def test_codex_passes_its_own_failure_through(spawn, event):
    spawn(lines=[event])
    with pytest.raises(ProviderError, match="usage limit"):
        list(CodexProvider().chat(TASK))


# --- failing before and after the turn ---------------------------------------


def test_a_tool_that_is_not_installed_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(external.shutil, "which", lambda name: None)
    with pytest.raises(ProviderError, match="npm i -g @openai/codex"):
        list(CodexProvider().chat(TASK))
    with pytest.raises(ProviderError, match="claude is not on PATH"):
        list(ClaudeCodeProvider().chat(TASK))


def test_a_non_zero_exit_carries_what_the_tool_printed(spawn):
    spawn(returncode=1, stderr="unknown flag --tools")
    with pytest.raises(ProviderError, match="exited 1: unknown flag"):
        list(ClaudeCodeProvider().chat(TASK))


def test_lines_that_are_not_json_are_skipped(spawn):
    # These tools print banners, warnings and progress bars around the stream.
    spawn(lines=["Warning: something", "{not json}", _delta("ok"), {"type": "result"}])
    assert collect(ClaudeCodeProvider().chat(TASK)).content == "ok"


def test_colour_is_switched_off_so_the_stream_stays_parseable(spawn):
    captured = spawn()
    list(ClaudeCodeProvider().chat(TASK))
    assert captured["env"]["NO_COLOR"] == "1"


# --- the registry ------------------------------------------------------------


def test_both_tools_can_be_named_as_providers():
    assert sorted(BACKENDS) == ["claude-code", "codex", "local", "ollama"]
    assert isinstance(build("claude-code"), ClaudeCodeProvider)
    assert isinstance(build("codex"), CodexProvider)


def test_each_backend_brings_its_own_model_and_window():
    # Switching provider without this would ask Claude Code for a model tag that
    # only a local Ollama has ever heard of.
    assert DEFAULTS["claude-code"] == {
        "model": DEFAULT_CLAUDE_CODE_MODEL, "host": "", "num_ctx": CLAUDE_CODE_NUM_CTX,
    }
    assert DEFAULTS["codex"] == {
        "model": DEFAULT_CODEX_MODEL, "host": "", "num_ctx": CODEX_NUM_CTX,
    }
