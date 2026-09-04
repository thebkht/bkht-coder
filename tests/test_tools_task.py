"""The `task` tool: a sub-agent's reading stays in the sub-agent's window."""

from __future__ import annotations

import pytest

from bkht.coder.parsing import ToolCall
from bkht.coder.tools import build_registry
from bkht.coder.tools.base import Registry, ToolError, ToolResult
from bkht.coder.tools.task import register_task_tool
from fakes import FakeProvider, call


class Recorder:
    """A listener that keeps what it was told, so a test can assert on it."""

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.calls: list[ToolCall] = []
        self.results: list[ToolResult] = []
        self.retries: list[str] = []

    def on_token(self, text: str) -> None:
        self.tokens.append(text)

    def on_tool_call(self, call: ToolCall) -> None:
        self.calls.append(call)

    def on_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        self.results.append(result)

    def on_retry(self, reason: str) -> None:
        self.retries.append(reason)


def delegated(project, script, listener=None):
    registry = Registry()
    register_task_tool(
        registry, project, FakeProvider(script), listener=listener, iterations=4
    )
    return registry.get("task")


def test_only_the_answer_comes_back(project):
    # The whole argument for delegating: the sub-agent reads a file, and the
    # parent's history grows by one paragraph rather than by the file.
    tool = delegated(
        project,
        [call("read_file", path="src/main.py"), "main() sums the numbers 0 to 9."],
    )
    result = tool.run(instruction="what does src/main.py do?")
    assert result.ok
    assert result.content == "main() sums the numbers 0 to 9."
    assert "def main" not in result.content


def test_a_sub_agent_that_answered_nothing_is_reported_as_a_failure(project):
    # Said as a fact rather than swallowed: the parent asked a question and got
    # nothing back, and it needs to know that so it goes and looks itself
    # instead of writing up an answer it never received.
    tool = delegated(project, [call("read_file", path="src/main.py")] * 6)
    with pytest.raises(ToolError, match="without an answer"):
        tool.run(instruction="what does it do?")


def test_an_empty_instruction_is_refused(project):
    tool = delegated(project, ["never reached"])
    with pytest.raises(ToolError, match="must say what to find out"):
        tool.run(instruction="   ")


def test_the_sub_agent_cannot_write_run_or_delegate(project):
    # Three bounds in one assertion, because they are one decision: delegation
    # buys context, and a nested agent that can change files or fan out further
    # is a different feature with a different set of questions to answer.
    # Built exactly as `task` builds it: read-only, no session, no delegate.
    registry, _ = build_registry(project, read_only=True)
    names = registry.names()
    assert "write_file" not in names
    assert "edit_file" not in names
    assert "bash" not in names
    assert "task" not in names
    assert "plan" not in names
    assert "read_file" in names


def test_delegation_is_offered_only_when_a_provider_is_handed_in(project):
    plain, _ = build_registry(project)
    assert "task" not in plain.names()

    with_task, _ = build_registry(
        project, delegate={"provider": FakeProvider([]), "listener": None}
    )
    assert "task" in with_task.names()


def test_the_sub_agents_tool_calls_are_shown_and_its_prose_is_not(project):
    # An agent that goes quiet for ninety seconds looks stuck, so the calls are
    # forwarded. The tokens are not: they would stream into the middle of the
    # parent's answer, which is the one place they do not belong.
    listener = Recorder()
    tool = delegated(
        project,
        [call("read_file", path="src/util.py"), "helper doubles its argument."],
        listener=listener,
    )
    tool.run(instruction="what does helper do?")

    assert "read_file" in [c.name for c in listener.calls]
    assert listener.tokens == []


def test_the_sub_agent_gets_its_own_session_and_leaves_no_file(project, tmp_path):
    # A transcript per delegated question would bury the sessions the user
    # actually had in a list they have to scroll.
    from bkht.coder import session as session_module

    tool = delegated(project, [call("read_file", path="README.md"), "A demo project."])
    before = list(session_module.SESSION_DIR.glob("*.jsonl")) if session_module.SESSION_DIR.is_dir() else []
    tool.run(instruction="what is this?")
    after = list(session_module.SESSION_DIR.glob("*.jsonl")) if session_module.SESSION_DIR.is_dir() else []
    assert len(after) == len(before)


def test_esc_travels_out_through_the_nested_loop(project):
    # `interrupt_main` raises KeyboardInterrupt on the thread running the loop,
    # and the sub-agent runs on that thread inside the tool call. `_execute`
    # catches Exception, not BaseException, so the cancellation has to keep
    # going up and out -- which is what the user pressing Esc meant.
    class Cancelling(FakeProvider):
        def chat(self, messages, tools=None):
            raise KeyboardInterrupt

    registry = Registry()
    register_task_tool(registry, project, Cancelling([]))
    with pytest.raises(KeyboardInterrupt):
        registry.get("task").run(instruction="anything")


def test_the_tool_needs_no_approval(project):
    # Reading only, so there is nothing to approve -- and a prompt raised from
    # inside a nested tool call is a question the user has no context for.
    registry = Registry()
    register_task_tool(registry, project, FakeProvider([]))
    assert registry.get("task").mutating is False


def test_the_sub_agent_gets_the_parents_hooks(project):
    """A gate the agent can get around by delegating is not a gate."""
    from bkht.coder import hooks as hooks_module

    class Fired:
        def __init__(self):
            self.events = []

        def fire(self, event, **context):
            self.events.append((event, context))
            return []

    fired = Fired()
    registry = Registry()
    provider = FakeProvider([call("read_file", path="src/util.py"), "it doubles"])
    register_task_tool(registry, project, provider=provider, hooks=fired)

    registry.get("task").run(instruction="what does helper do?")
    assert (
        hooks_module.PRE_TOOL,
        {"tool": "read_file", "arguments": {"path": "src/util.py"}},
    ) in fired.events
