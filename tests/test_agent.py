"""The loop: tool dispatch, malformed-call recovery, and the bounds."""

import itertools

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
    # The first reading and the last are what the duration is made of; the ones
    # between belong to the loop's own bookkeeping and it may take more of them
    # than it does today, so the clock keeps answering rather than running out.
    ticks = itertools.chain([10.0, 12.5], itertools.repeat(14.0))
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


def test_a_turn_that_runs_too_long_is_asked_to_answer(loop):
    # Nothing else bounds the clock: the iteration cap counts round trips and
    # the retry cap counts only rounds where every call failed, so a turn making
    # distinct successful calls that go nowhere was bounded by neither.
    ticks = itertools.count(0.0, 1_000.0)  # every reading a thousand seconds on
    agent, _ = loop(
        [call("read_file", path=f"src/{n}.py") for n in range(20)] + ["done"],
        clock=lambda: next(ticks),
        max_iterations=20,
    )
    outcome = agent.run("go")
    assert outcome.stopped == "time-cap"
    assert outcome.iterations < 20


def test_the_clock_does_not_end_a_turn_before_it_starts(loop):
    # The budget is checked between iterations, so the first one always runs.
    ticks = itertools.count(0.0, 1_000.0)
    agent, _ = loop(["Done."], clock=lambda: next(ticks))
    outcome = agent.run("go")
    assert outcome.stopped == "answered"


# --- the verify loop --------------------------------------------------------


class Suite:
    """A scripted `verify.suite`, so the loop is tested without a test runner."""

    def __init__(self, *reports) -> None:
        from bkht.coder import verify

        self.reports = list(reports) or [verify.Report(verify.PASSED, "check")]
        self.runs: list[str] = []

    def __call__(self, command, root, timeout=None, runner=None):
        self.runs.append(command)
        return self.reports[min(len(self.runs), len(self.reports)) - 1]


def writing(project, script, monkeypatch, suite, **kwargs):
    """An agent that can write, with `verify.suite` replaced by ``suite``."""
    from bkht.coder import agent as agent_module

    monkeypatch.setattr(agent_module.verify, "suite", suite)
    registry, workspace = build_registry(project)
    provider = FakeProvider(script)
    made = Agent(
        provider, registry, Session(system=""),
        verify_command="check", verify_root=project, **kwargs,
    )
    return made, provider


def test_a_turn_that_edited_runs_the_command_before_it_answers(project, monkeypatch):
    from bkht.coder import verify

    suite = Suite(verify.Report(verify.PASSED, "check"))
    agent, _ = writing(
        project,
        [call("write_file", path="a.py", content="x = 1\n"), "done"],
        monkeypatch, suite,
    )
    outcome = agent.run("write a file")

    assert outcome.answer == "done"
    assert suite.runs == ["check"]


def test_a_turn_that_only_read_runs_nothing(project, monkeypatch):
    # The cheapness of this rests entirely on it: most turns are questions,
    # and a question has no edits to check.
    suite = Suite()
    agent, _ = writing(
        project, [call("read_file", path="src/util.py"), "it doubles"],
        monkeypatch, suite,
    )
    assert agent.run("what does helper do?").answer == "it doubles"
    assert suite.runs == []


def test_a_failure_goes_back_as_a_tool_result_and_the_turn_continues(project, monkeypatch):
    # The correction path, and the whole point of the feature: the model gets
    # to see what broke and fix it, in the same turn.
    from bkht.coder import verify

    suite = Suite(
        verify.Report(verify.FAILED, "check", "E   assert 3 == 4", 1),
        verify.Report(verify.PASSED, "check"),
    )
    agent, provider = writing(
        project,
        [
            call("write_file", path="a.py", content="x = 1\n"),
            "done",
            call("write_file", path="a.py", content="x = 2\n"),
            "fixed it",
        ],
        monkeypatch, suite,
    )
    outcome = agent.run("write a file")

    assert outcome.answer == "fixed it"
    assert suite.runs == ["check", "check"]
    fed = [m for m in agent.session.messages if m["role"] == "tool"]
    assert any("assert 3 == 4" in m["content"] for m in fed)


def test_the_suite_runs_at_most_twice_in_one_turn(project, monkeypatch):
    # The first run is the check, the second is the fix being checked. A third
    # would hand back a failure the model has already failed to fix once.
    from bkht.coder import verify

    suite = Suite(verify.Report(verify.FAILED, "check", "still broken", 1))
    agent, _ = writing(
        project,
        [
            call("write_file", path="a.py", content="x = 1\n"), "done",
            call("write_file", path="a.py", content="x = 2\n"), "try again",
            call("write_file", path="a.py", content="x = 3\n"), "and again",
        ],
        monkeypatch, suite,
    )
    outcome = agent.run("write a file")

    assert len(suite.runs) == verify.MAX_RUNS
    assert outcome.stopped == "answered"


def test_the_last_run_asks_for_an_account_rather_than_another_fix(project, monkeypatch):
    from bkht.coder import verify

    suite = Suite(verify.Report(verify.FAILED, "check", "still broken", 1))
    agent, _ = writing(
        project,
        [
            call("write_file", path="a.py", content="x = 1\n"), "done",
            call("write_file", path="a.py", content="x = 2\n"), "second try",
            # The account itself. Being asked to stop and explain still costs a
            # reply, which is the point -- the turn ends with the explanation
            # rather than with the failure nobody described.
            "test_add still fails; my change was not the cause.",
        ],
        monkeypatch, suite,
    )
    outcome = agent.run("write a file")

    fed = [m["content"] for m in agent.session.messages if m["role"] == "tool"]
    assert any("Stop editing and answer now" in text for text in fed)
    assert outcome.answer == "test_add still fails; my change was not the cause."


def test_a_timeout_ends_the_turn_instead_of_being_fed_back(project, monkeypatch):
    # Nothing about a timeout tells the model what to change, so handing it
    # back would spend an iteration on a message it cannot act on.
    from bkht.coder import verify

    suite = Suite(verify.Report(verify.TIMED_OUT, "check"))
    agent, _ = writing(
        project, [call("write_file", path="a.py", content="x = 1\n"), "done"],
        monkeypatch, suite,
    )
    outcome = agent.run("write a file")

    assert outcome.answer == "done"
    assert len(suite.runs) == 1
    assert any("timed out" in error for error in outcome.errors)


def test_no_command_configured_runs_nothing(project, monkeypatch):
    from bkht.coder import agent as agent_module

    suite = Suite()
    monkeypatch.setattr(agent_module.verify, "suite", suite)
    registry, _ = build_registry(project)
    made = Agent(
        FakeProvider([call("write_file", path="a.py", content="x = 1\n"), "done"]),
        registry, Session(system=""), verify_root=project,
    )
    assert made.run("write a file").answer == "done"
    assert suite.runs == []


def test_each_turn_gets_its_own_budget(project, monkeypatch):
    # Last turn's edits were checked last turn, and a second turn that edits
    # deserves the same two runs the first one had.
    from bkht.coder import verify

    suite = Suite(verify.Report(verify.PASSED, "check"))
    agent, provider = writing(
        project,
        [
            call("write_file", path="a.py", content="x = 1\n"), "done",
            call("write_file", path="b.py", content="y = 2\n"), "done again",
        ],
        monkeypatch, suite,
    )
    agent.run("write a file")
    agent.run("write another")
    assert suite.runs == ["check", "check"]


def test_the_check_is_reported_as_it_happens(project, monkeypatch):
    # A turn that goes quiet for two minutes running someone's test suite has
    # to say that is what it is doing.
    from bkht.coder import verify

    said = []

    class Listener:
        def on_token(self, text): pass
        def on_tool_call(self, call): pass
        def on_tool_result(self, call, result): pass
        def on_retry(self, reason): said.append(reason)

    suite = Suite(verify.Report(verify.PASSED, "check"))
    agent, _ = writing(
        project, [call("write_file", path="a.py", content="x = 1\n"), "done"],
        monkeypatch, suite, listener=Listener(),
    )
    agent.run("write a file")

    assert "running check" in said
    assert "check passed" in said


def test_a_bounded_turn_still_reports_whether_it_broke_the_tests(project, monkeypatch):
    # A turn that ran out mid-edit is the one most likely to have left the
    # tests broken, and ending in silence about that is the worst of both: the
    # work is half done and nothing says so.
    from bkht.coder import verify

    suite = Suite(verify.Report(verify.FAILED, "check", "1 failed", 1))
    agent, _ = writing(
        project,
        [call("write_file", path="a.py", content="x = 1\n")] * 3 + ["ran out"],
        monkeypatch, suite, max_iterations=3,
    )
    outcome = agent.run("write a file")

    assert outcome.stopped == "iteration-cap"
    assert suite.runs == ["check"]
    assert any("check failed" in error for error in outcome.errors)


def test_a_bounded_turn_is_not_handed_the_failure_to_fix(project, monkeypatch):
    # There is no room left to act on it -- that is what being bounded means --
    # so the message would only crowd out the wrap-up prose it is asked for.
    from bkht.coder import verify

    suite = Suite(verify.Report(verify.FAILED, "check", "1 failed", 1))
    agent, _ = writing(
        project,
        [call("write_file", path="a.py", content="x = 1\n")] * 3 + ["ran out"],
        monkeypatch, suite, max_iterations=3,
    )
    agent.run("write a file")

    fed = [m["content"] for m in agent.session.messages if m["role"] == "tool"]
    assert not any("was run to check the work" in text for text in fed)


def test_a_bounded_turn_that_wrote_nothing_runs_nothing(project, monkeypatch):
    suite = Suite()
    agent, _ = writing(
        project,
        [call("read_file", path="src/util.py")] * 3 + ["ran out"],
        monkeypatch, suite, max_iterations=3,
    )
    agent.run("read a file")
    assert suite.runs == []


def test_the_suite_is_not_rerun_when_nothing_changed_since_the_last_run(project, monkeypatch):
    # Seen live: the model read a failure, decided it was not about its change,
    # and answered without editing. Running the suite again there watches
    # identical code fail identically, at the price of the whole timeout.
    from bkht.coder import verify

    suite = Suite(verify.Report(verify.FAILED, "check", "unrelated failure", 1))
    agent, _ = writing(
        project,
        [
            call("write_file", path="a.py", content="x = 1\n"),
            "done",
            "that failure is not about my change",
        ],
        monkeypatch, suite,
    )
    outcome = agent.run("write a file")

    assert suite.runs == ["check"]
    assert outcome.answer == "that failure is not about my change"


def test_a_second_run_happens_when_the_model_actually_edited_again(project, monkeypatch):
    from bkht.coder import verify

    suite = Suite(
        verify.Report(verify.FAILED, "check", "broken", 1),
        verify.Report(verify.PASSED, "check"),
    )
    agent, _ = writing(
        project,
        [
            call("write_file", path="a.py", content="x = 1\n"), "done",
            call("write_file", path="a.py", content="x = 2\n"), "fixed",
        ],
        monkeypatch, suite,
    )
    assert agent.run("write a file").answer == "fixed"
    assert suite.runs == ["check", "check"]


# --- hooks --------------------------------------------------------------------


class Fired:
    """A ``Hooks`` that records rather than runs."""

    def __init__(self, *results) -> None:
        self.results = list(results)
        self.events: list[tuple] = []

    def fire(self, event, **context):
        self.events.append((event, context))
        return [r for r in self.results if r.event == event]


def hooked(project, script, results=()):
    """An agent that can write, with its hooks recorded rather than run."""
    registry, _ = build_registry(project)
    fired = Fired(*results)
    made = Agent(FakeProvider(script), registry, Session(system=""), hooks=fired)
    return made, fired


def test_a_turn_with_no_hooks_configured_fires_nothing(loop):
    agent, _ = loop(["done"])
    # `None` rather than an empty Hooks: the common case is one identity check
    # per call, not a dict lookup and a shell resolution.
    assert agent.hooks is None
    assert agent.run("hello").answer == "done"


def test_both_tool_events_fire_around_a_call(project):
    from bkht.coder import hooks as hooks_module

    agent, fired = hooked(
        project, [call("read_file", path="src/util.py"), "it doubles"]
    )
    agent.run("what does helper do?")
    events = [event for event, _ in fired.events]
    assert events == [hooks_module.PRE_TOOL, hooks_module.POST_TOOL, hooks_module.TURN_END]
    assert fired.events[0][1]["tool"] == "read_file"
    assert fired.events[1][1]["ok"] is True


def test_a_pre_tool_hook_can_refuse_a_call(project):
    from bkht.coder import hooks as hooks_module

    refusal = hooks_module.Result(
        hooks_module.PRE_TOOL, "gate", code=1, output="nothing under src/"
    )
    agent, fired = hooked(
        project,
        [call("read_file", path="src/util.py"), "I could not read it."],
        results=[refusal],
    )
    outcome = agent.run("read it")

    # The hook's own sentence reaches the model, which is the whole mechanism:
    # a refusal it cannot read is one it can only guess at.
    assert "nothing under src/" in agent.session.messages[-2]["content"]
    assert outcome.answer == "I could not read it."
    # And the tool never ran, so nothing after it was fired for it.
    assert [e for e, _ in fired.events].count(hooks_module.POST_TOOL) == 0


def test_a_post_tool_hook_hears_about_a_failed_call(project):
    from bkht.coder import hooks as hooks_module

    agent, fired = hooked(
        project, [call("read_file", path="nope.py"), "there is no such file"]
    )
    agent.run("read nope.py")
    post = [c for e, c in fired.events if e == hooks_module.POST_TOOL]
    # "The write did not happen" is exactly what a hook watching writes needs
    # to hear, and hearing nothing is indistinguishable from not configured.
    assert post and post[0]["ok"] is False


def test_the_turn_end_hook_hears_whether_anything_changed(project):
    from bkht.coder import hooks as hooks_module

    agent, fired = hooked(
        project, [call("write_file", path="a.py", content="x = 1\n"), "done"]
    )
    outcome = agent.run("write a file")
    end = [c for e, c in fired.events if e == hooks_module.TURN_END][0]
    assert end == {"stopped": outcome.stopped, "edited": True, "tool_calls": 1}


def test_a_turn_that_only_read_says_so(project):
    from bkht.coder import hooks as hooks_module

    agent, fired = hooked(project, [call("read_file", path="src/util.py"), "done"])
    agent.run("read it")
    end = [c for e, c in fired.events if e == hooks_module.TURN_END][0]
    assert end["edited"] is False


def test_the_turn_end_hook_fires_even_when_the_turn_did_not_answer(project):
    from bkht.coder import hooks as hooks_module

    agent, fired = hooked(project, ["", "", "", "", "", ""])
    outcome = agent.run("hello")
    assert outcome.stopped == "retry-cap"
    assert [c for e, c in fired.events if e == hooks_module.TURN_END]


def test_a_blocking_hook_is_not_also_announced(project):
    """Its sentence is about to be the tool result; twice is once too many.

    And in the wrong order: the listener prints a call together with its
    result, so a notice raised mid-call lands above the call it was fired for
    and reads as though it belonged to the one before.
    """
    from bkht.coder import hooks as hooks_module

    said: list[str] = []

    class Listens:
        def on_token(self, text): pass
        def on_tool_call(self, call): pass
        def on_tool_result(self, call, result): pass
        def on_retry(self, reason): said.append(reason)

    refusal = hooks_module.Result(hooks_module.PRE_TOOL, "gate", code=1, output="no")
    registry, _ = build_registry(project)
    agent = Agent(
        FakeProvider([call("read_file", path="src/util.py"), "blocked"]),
        registry, Session(system=""), listener=Listens(), hooks=Fired(refusal),
    )
    agent.run("read it")

    assert not [line for line in said if "gate" in line]
    assert "no" in agent.session.messages[-2]["content"]


def test_a_hook_that_went_wrong_and_blocks_nothing_is_still_said_out_loud(project):
    from bkht.coder import hooks as hooks_module

    said: list[str] = []

    class Listens:
        def on_token(self, text): pass
        def on_tool_call(self, call): pass
        def on_tool_result(self, call, result): pass
        def on_retry(self, reason): said.append(reason)

    # A formatter that silently rewrote the file the model just wrote is a hook
    # that makes the next tool result inexplicable.
    broke = hooks_module.Result(hooks_module.POST_TOOL, "fmt", code=2, output="boom")
    registry, _ = build_registry(project)
    agent = Agent(
        FakeProvider([call("read_file", path="src/util.py"), "done"]),
        registry, Session(system=""), listener=Listens(), hooks=Fired(broke),
    )
    agent.run("read it")
    assert [line for line in said if "fmt" in line and "exited 2" in line]
