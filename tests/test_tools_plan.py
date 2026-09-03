"""The `plan` tool, and the reminder it puts on every request."""

from __future__ import annotations

import pytest

from bkht.coder.session import Session
from bkht.coder.tools.base import Registry, ToolError
from bkht.coder.tools.plan import register_plan_tool


@pytest.fixture
def planning():
    session = Session()
    registry = Registry()
    register_plan_tool(registry, session)
    return registry.get("plan"), session


def test_writing_a_plan_reports_it_back_with_the_progress(planning):
    tool, session = planning
    result = tool.run(steps=["read it", "say what it does"])
    assert result.ok
    assert "1. [ ] read it" in result.content
    assert "0/2 done." in result.content


def test_ticking_a_step_names_it(planning):
    tool, session = planning
    tool.run(steps=["read it", "say what it does"])
    result = tool.run(done=1)
    assert result.ok
    assert "Ticked step 1: read it" in result.content
    assert "1/2 done." in result.content


def test_a_finished_plan_tells_the_model_to_stop_calling_tools(planning):
    # The failure without it: a model that has ticked every box goes round
    # again looking for something else to do, and spends the turn on it.
    tool, _ = planning
    tool.run(steps=["only step"])
    result = tool.run(done=1)
    assert "Answer the user now" in result.content


def test_a_call_with_neither_argument_says_what_to_pass(planning):
    tool, _ = planning
    with pytest.raises(ToolError, match="`steps`.*`done`"):
        tool.run()


def test_an_empty_step_list_is_refused_rather_than_silently_dropped(planning):
    # `set` throws blank lines away, so a plan of nothing but blanks would
    # otherwise be reported as a plan that was set.
    tool, session = planning
    with pytest.raises(ToolError, match="at least one step"):
        tool.run(steps=["", "   "])
    assert not session.plan


def test_steps_that_are_not_strings_are_refused(planning):
    tool, _ = planning
    with pytest.raises(ToolError, match="list of strings"):
        tool.run(steps=["fine", 3])


def test_a_bad_step_number_comes_back_with_the_plan_to_correct_from(planning):
    tool, _ = planning
    tool.run(steps=["one", "two"])
    with pytest.raises(ToolError, match="step 9 does not exist; the plan has 2"):
        tool.run(done=9)


def test_the_plan_is_appended_to_every_request_while_one_exists(planning):
    # The whole point: it rides on the payload, not in the history, so nothing
    # that frees context can take it.
    tool, session = planning
    assert session.payload() == []

    tool.run(steps=["read it", "say what it does"])
    payload = session.payload()
    assert len(payload) == 1
    assert payload[0]["role"] == "user"
    assert "1. [ ] read it" in payload[0]["content"]
    assert session.messages == []


def test_compaction_cannot_take_the_plan(planning):
    # `clear` is the most violent thing that happens to a history short of a
    # new session; the plan has to outlive everything short of that.
    from bkht.coder.context import elide_tool_results

    tool, session = planning
    tool.run(steps=["read it"])
    for _ in range(6):
        session.add_tool_result("read_file", "x" * 3000)
    elide_tool_results(session)
    assert "1. [ ] read it" in session.payload()[-1]["content"]


def test_clearing_the_session_clears_the_plan(planning):
    # A plan that survived `/clear` would meet the next turn with a checklist
    # for work nobody asked for -- the exact confusion it exists to prevent.
    tool, session = planning
    tool.run(steps=["read it"])
    session.clear()
    assert not session.plan
    assert session.payload() == []


def test_the_tool_needs_no_approval():
    # A plan the user has to approve is a plan the model stops writing.
    registry = Registry()
    register_plan_tool(registry, Session())
    assert registry.get("plan").mutating is False
