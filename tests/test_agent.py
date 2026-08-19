"""The loop: tool dispatch, malformed-call recovery, and the bounds."""

import pytest

from bkht.coder.agent import Agent, MAX_RETRIES
from bkht.coder.prompts import system_prompt
from bkht.coder.provider import ProviderError
from bkht.coder.session import Session
from bkht.coder.tools import build_registry

from fakes import FakeProvider, call


@pytest.fixture
def loop(project):
    def build(script, **kwargs):
        registry, workspace = build_registry(project, read_only=True)
        provider = FakeProvider(script)
        session = Session(system=system_prompt(registry, str(workspace.root)))
        agent = Agent(provider, registry, session, **kwargs)
        return agent, provider

    return build


def test_answers_without_calling_a_tool(loop):
    agent, provider = loop(["It is a demo project."])
    outcome = agent.run("what is this?")
    assert outcome.answer == "It is a demo project."
    assert outcome.stopped == "answered"
    assert outcome.tool_calls == 0


def test_reads_a_file_then_answers(loop):
    agent, provider = loop(
        [call("read_file", path="src/util.py"), "helper doubles its argument."]
    )
    outcome = agent.run("what does helper do?")
    assert outcome.answer == "helper doubles its argument."
    assert outcome.tool_calls == 1

    # The tool output really reached the model on the second call.
    second = provider.calls[1]
    assert second[-1]["role"] == "tool"
    assert "def helper(x):" in second[-1]["content"]


def test_system_prompt_is_sent_first(loop):
    agent, provider = loop(["done"])
    agent.run("hi")
    assert provider.calls[0][0]["role"] == "system"
    assert "Calling a tool" in provider.calls[0][0]["content"]


def test_several_calls_in_one_reply_all_run(loop):
    agent, _ = loop(
        [
            call("read_file", path="src/util.py") + "\n" + call("read_file", path="README.md"),
            "read both.",
        ]
    )
    outcome = agent.run("read both files")
    assert outcome.tool_calls == 2


def test_prose_around_a_call_does_not_end_the_loop(loop):
    agent, _ = loop(
        ["Let me look.\n" + call("read_file", path="README.md"), "It is a demo."]
    )
    assert agent.run("what is it?").answer == "It is a demo."


# --- malformed-call recovery ------------------------------------------------


def test_unknown_tool_is_fed_back_and_recovered_from(loop):
    agent, provider = loop(
        [call("open_file", path="README.md"), call("read_file", path="README.md"), "done"]
    )
    outcome = agent.run("read it")
    assert outcome.answer == "done"

    correction = provider.calls[1][-1]["content"]
    assert "no tool named 'open_file'" in correction
    assert "read_file" in correction


def test_missing_argument_is_fed_back(loop):
    agent, provider = loop([call("read_file"), call("read_file", path="README.md"), "done"])
    assert agent.run("read it").answer == "done"
    assert "missing required argument" in provider.calls[1][-1]["content"]


def test_unknown_argument_is_fed_back(loop):
    agent, provider = loop(
        [call("read_file", filename="README.md"), call("read_file", path="README.md"), "done"]
    )
    assert agent.run("read it").answer == "done"
    assert "unknown argument" in provider.calls[1][-1]["content"]


def test_tool_error_is_fed_back_as_text_not_raised(loop):
    agent, provider = loop(
        [call("read_file", path="nope.py"), call("read_file", path="README.md"), "done"]
    )
    assert agent.run("read it").answer == "done"
    assert "file not found" in provider.calls[1][-1]["content"]


def test_path_escape_is_fed_back(loop):
    agent, provider = loop(
        [call("read_file", path="../../etc/passwd"), "I cannot read outside the workspace."]
    )
    outcome = agent.run("read /etc/passwd")
    assert "outside the workspace" in provider.calls[1][-1]["content"]
    assert outcome.stopped == "answered"


def test_empty_reply_triggers_a_retry_not_a_stop(loop):
    agent, provider = loop(["", "Here is the answer."])
    outcome = agent.run("hi")
    assert outcome.answer == "Here is the answer."
    assert "neither a tool call nor an answer" in provider.calls[1][-1]["content"]


def test_repeated_bad_replies_stop_at_the_retry_cap(loop):
    agent, _ = loop([""] * (MAX_RETRIES + 2))
    outcome = agent.run("hi")
    assert outcome.stopped == "retry-cap"
    assert outcome.iterations == MAX_RETRIES + 1


def test_retry_counter_resets_after_progress(loop):
    # Two bad replies, a good call, then two more bad ones: still recoverable.
    agent, _ = loop(
        ["", "", call("read_file", path="README.md"), "", "", "finally an answer"]
    )
    assert agent.run("hi").answer == "finally an answer"


def test_persistent_tool_failure_stops_at_the_retry_cap(loop):
    agent, _ = loop([call("read_file", path="nope.py")] * (MAX_RETRIES + 2))
    outcome = agent.run("read it")
    assert outcome.stopped == "retry-cap"


# --- bounds -----------------------------------------------------------------


def test_iteration_cap_stops_a_confused_model(loop):
    agent, _ = loop([call("read_file", path="README.md")] * 20, max_iterations=4)
    outcome = agent.run("loop forever")
    assert outcome.stopped == "iteration-cap"
    assert outcome.iterations == 4


def test_provider_error_is_reported_not_raised(loop):
    agent, _ = loop([ProviderError("cannot reach Ollama")])
    outcome = agent.run("hi")
    assert outcome.stopped == "provider-error"
    assert "cannot reach Ollama" in outcome.errors[0]


def test_usage_is_recorded(loop):
    agent, _ = loop([call("read_file", path="README.md"), "done"])
    agent.run("hi")
    assert agent.session.prompt_tokens == 100
    assert agent.session.completion_tokens > 0
