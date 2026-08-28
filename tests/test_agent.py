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
    # Distinct calls, so this bounds the model that keeps working and never
    # answers. A model repeating one call verbatim is stopped sooner, by the
    # repeat guard below.
    agent, _ = loop(
        [call("read_file", path="README.md", offset=i + 1) for i in range(20)],
        max_iterations=4,
    )
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


# --- permissions inside the loop --------------------------------------------


def test_denied_call_stops_the_loop_with_a_reason(project):
    from bkht.coder.permissions import ASK, Permissions

    registry, workspace = build_registry(project)
    permissions = Permissions(mode=ASK, workspace=workspace, prompt=lambda q, b: "n")
    provider = FakeProvider([call("write_file", path="a.py", content="x")])
    session = Session(system="")
    agent = Agent(provider, registry, session, permissions=permissions)

    outcome = agent.run("write a file")
    assert outcome.stopped == "denied"
    assert not (project / "a.py").exists()


def test_approved_call_writes_the_file(project):
    from bkht.coder.permissions import ASK, Permissions

    registry, workspace = build_registry(project)
    permissions = Permissions(mode=ASK, workspace=workspace, prompt=lambda q, b: "y")
    provider = FakeProvider([call("write_file", path="a.py", content="x = 1\n"), "done"])
    agent = Agent(provider, registry, Session(system=""), permissions=permissions)

    assert agent.run("write a file").answer == "done"
    assert (project / "a.py").read_text() == "x = 1\n"


def test_plan_mode_denial_is_fed_back_so_the_model_can_explain(project):
    from bkht.coder.permissions import PLAN, Permissions

    registry, workspace = build_registry(project)
    permissions = Permissions(mode=PLAN, workspace=workspace)
    provider = FakeProvider([call("write_file", path="a.py", content="x")])
    agent = Agent(provider, registry, Session(system=""), permissions=permissions)

    outcome = agent.run("write a file")
    assert outcome.stopped == "denied"
    assert "plan mode" in outcome.errors[-1]


def test_a_json_only_reply_that_is_not_a_call_is_a_final_answer(loop):
    # The review passes answer with a bare JSON array; stripping it would leave
    # nothing and send the loop into a pointless retry.
    agent, _ = loop(['[{"file": "a.py", "line": 3, "summary": "off by one"}]'])
    outcome = agent.run("review it")
    assert outcome.stopped == "answered"
    assert "off by one" in outcome.raw


def test_whitespace_only_reply_is_still_a_retry(loop):
    agent, provider = loop(["   \n  ", "done"])
    assert agent.run("hi").answer == "done"


# --- the scout ---------------------------------------------------------------


def scout_messages(payload: list[dict]) -> list[dict]:
    return [m for m in payload if m.get("name") == "codebase_search"]


def test_the_workspace_is_searched_before_the_model_is_asked(loop, project):
    agent, provider = loop(["helper doubles its argument."], scout_root=project)
    agent.run("what does helper do?")

    found = scout_messages(provider.calls[0])
    assert len(found) == 1
    assert "src/util.py" in found[0]["content"]
    # After the user message, so the model reads the request and then the
    # search made for it.
    assert provider.calls[0].index(found[0]) > provider.calls[0].index(
        {"role": "user", "content": "what does helper do?"}
    )


def test_nothing_is_searched_without_a_scout_root(loop):
    agent, provider = loop(["It is a demo project."])
    agent.run("what does helper do?")
    assert scout_messages(provider.calls[0]) == []


def test_a_conversational_turn_is_not_searched(loop, project):
    agent, provider = loop(["You're welcome."], scout_root=project)
    agent.run("thanks!")
    assert scout_messages(provider.calls[0]) == []


def test_a_broken_scout_does_not_lose_the_turn(loop, project, monkeypatch):
    from bkht.coder import agent as agent_module

    def explode(*args, **kwargs):
        raise RuntimeError("index on fire")

    monkeypatch.setattr(agent_module, "scout", explode)
    agent, provider = loop(["helper doubles its argument."], scout_root=project)

    outcome = agent.run("what does helper do?")
    assert outcome.answer == "helper doubles its argument."
    assert scout_messages(provider.calls[0]) == []


# --- the user's language ----------------------------------------------------


def test_an_uzbek_message_sets_the_language(loop):
    agent, provider = loop(["Salom! Nima qilay?"])
    agent.run("salom")
    assert agent.session.language == "Uzbek"
    assert "Uzbek" in provider.calls[0][-1]["content"]


def test_the_language_survives_a_message_that_says_nothing(loop):
    agent, provider = loop(["Salom!", "Xush kelibsiz."])
    agent.run("salom")
    agent.run("src/util.py")
    assert agent.session.language == "Uzbek"
    assert "Uzbek" in provider.calls[1][-1]["content"]


def test_an_english_message_gets_no_reminder(loop):
    agent, provider = loop(["It is a demo project."])
    agent.run("what is this?")
    assert agent.session.language == "English"
    assert provider.calls[0][-1]["role"] == "user"


def test_tracking_can_be_switched_off(loop):
    agent, provider = loop(["[]"], track_language=False)
    agent.run("salom")
    assert agent.session.language is None
    assert provider.calls[0][-1]["role"] == "user"


# --- what a turn cost -------------------------------------------------------


def test_a_turn_reports_how_long_it_took(loop):
    ticks = iter([10.0, 12.5, 13.0, 14.0])
    agent, _ = loop(["Done."], clock=lambda: next(ticks))
    assert agent.run("go").seconds == pytest.approx(4.0)


def test_the_duration_covers_the_whole_turn_not_just_the_model(loop):
    # run() restamps over resume()'s own measurement, so the scout's search of
    # the workspace is inside the number the user is shown.
    agent, _ = loop(["Done."])
    outcome = agent.run("go")
    assert outcome.seconds > 0


def test_generated_tokens_are_summed_across_every_round_trip(loop):
    agent, _ = loop([call("read_file", path="README.md"), "It is a demo."])
    outcome = agent.run("read it")
    # The fake reports one completion token per character of each reply.
    assert outcome.received == len(call("read_file", path="README.md")) + len("It is a demo.")


def test_the_prompt_size_reported_is_the_last_one_not_the_sum(loop):
    # The prompt grows every round; adding them up would claim a turn sent
    # several times what it ever held.
    agent, _ = loop([call("read_file", path="README.md"), "It is a demo."])
    assert agent.run("read it").sent == 100


def test_a_turn_that_never_reached_the_model_costs_nothing(loop):
    agent, _ = loop([ProviderError("ollama is not running")])
    outcome = agent.run("go")
    assert outcome.stopped == "provider-error"
    assert (outcome.sent, outcome.received) == (0, 0)


# --- what goes on the wire --------------------------------------------------


def test_tool_declarations_are_not_sent(loop):
    """The native ``tools`` array is deliberately omitted.

    Sending it makes Ollama's qwen2.5 template render its own
    ``<tool_call></tool_call>`` protocol into the prompt, contradicting the one
    the system prompt states -- and measurement says the model honours neither
    reliably when it is given both.
    """
    agent, provider = loop([call("read_file", path="src/util.py"), "done"])
    agent.run("read it")
    assert provider.tools_seen == [None, None]


def test_an_empty_reply_is_not_stored_in_history(loop):
    agent, provider = loop(["", "done"])
    outcome = agent.run("hi")
    assert outcome.answer == "done"
    assert not any(
        m["role"] == "assistant" and not m["content"].strip()
        for m in agent.session.messages
    ), "an empty reply teaches the format that produced it"


# --- reading the same thing forever -----------------------------------------


def test_an_exact_repeat_is_refused_not_run(loop):
    """The loop the user actually hit.

    Freeing context necessarily costs the model some of what it read, and a
    model that has lost a file reaches for it again -- which spends the window
    that made it forget, and loses the file again. Left alone the turn reads the
    same file until the iteration cap.
    """
    agent, _ = loop(
        [
            call("read_file", path="src/util.py"),
            call("read_file", path="src/util.py"),
            "done",
        ]
    )
    outcome = agent.run("what does helper do?")
    assert outcome.answer == "done"
    assert any("already ran" in e for e in outcome.errors)


def test_a_refused_repeat_says_what_to_do_instead(loop):
    agent, _ = loop([call("read_file", path="src/util.py")] * 2 + ["done"])
    agent.run("what does helper do?")
    refusal = agent.session.messages[-2]["content"]
    assert "offset" in refusal and "grep" in refusal


def test_repeating_one_call_ends_the_turn_as_looping(loop):
    # Named apart from the retry cap because it is a different diagnosis: the
    # model is being answered every time and asking again anyway.
    agent, _ = loop([call("read_file", path="src/util.py")] * 20, max_iterations=20)
    outcome = agent.run("loop forever")
    assert outcome.stopped == "looping"
    assert outcome.iterations < 20, "stopped by the repeat guard, not the iteration cap"


def test_the_same_call_is_allowed_again_in_a_later_turn(loop):
    agent, _ = loop(
        [call("read_file", path="src/util.py"), "first"]
        + [call("read_file", path="src/util.py"), "second"]
    )
    assert agent.run("read it").answer == "first"
    assert agent.run("read it again").answer == "second"


def test_different_arguments_are_not_a_repeat(loop):
    agent, _ = loop(
        [
            call("read_file", path="src/util.py"),
            call("read_file", path="src/util.py", offset=2),
            "done",
        ]
    )
    outcome = agent.run("read it twice")
    assert outcome.answer == "done"
    assert outcome.errors == []


# --- running out of room ----------------------------------------------------


def test_a_capped_turn_still_asks_for_an_answer(loop):
    """A bounded turn used to end with nothing at all.

    The CLI then filled the gap with the last tool error -- a message written
    for the model -- and showed it to the user as though it explained anything.
    """
    agent, _ = loop(
        [call("read_file", path="src/util.py", offset=i + 1) for i in range(3)]
        + ["I read most of src/util.py; I did not get to the end."],
        max_iterations=3,
    )
    outcome = agent.run("summarize it")
    assert outcome.stopped == "iteration-cap"
    assert outcome.answer.startswith("I read most of")


def test_the_final_ask_says_a_partial_answer_is_wanted(loop):
    agent, provider = loop(
        [call("read_file", path="src/util.py"), "partial"], max_iterations=1
    )
    agent.run("summarize it")
    asked = provider.calls[-1][-1]["content"]
    assert "run out of steps" in asked and "partial answer" in asked


def test_a_capped_turn_that_still_will_not_answer_reports_the_cap(loop):
    agent, _ = loop(
        [call("read_file", path="src/util.py", offset=i + 1) for i in range(5)],
        max_iterations=3,
    )
    outcome = agent.run("summarize it")
    assert outcome.stopped == "iteration-cap"
    assert outcome.answer == ""


def test_a_refused_repeat_hands_back_what_the_call_returned(loop):
    # The refusal used to return nothing, which left the model owing an answer
    # it had been told it could not have -- and a model in that position writes
    # down what it remembers instead of admitting it has nothing.
    agent, _ = loop([call("read_file", path="src/util.py")] * 2 + ["done"])
    agent.run("what does helper do?")
    refusal = agent.session.messages[-2]["content"]
    assert "def helper" in refusal, "the first result should be replayed verbatim"
    assert "Do not describe output you have not been given" in refusal


def test_a_call_that_was_never_run_is_not_replayed(loop):
    # A refused or malformed call never happened; handing back a result for it
    # would be inventing one, which is the thing this is meant to stop.
    agent, _ = loop([call("no_such_tool")] * 2 + ["done"])
    agent.run("go")
    refusal = agent.session.messages[-2]["content"]
    assert "returned the first time" not in refusal
