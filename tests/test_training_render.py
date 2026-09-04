"""Rendering, and the round-trip that makes the dataset trustworthy."""

import pytest

from bkht.coder.parsing import parse_tool_calls
from bkht.coder.training import render
from bkht.coder.training.ingest import Trajectory, call_json


@pytest.fixture(scope="module")
def allowed(tmp_path_factory):
    return render.schemas(render.registry_for(tmp_path_factory.mktemp("work")))


def made(messages, **fields):
    return Trajectory(source="test", origin="t", messages=messages, **fields)


# --- the invariant this whole module exists for -------------------------------


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("read_file", {"path": "a.py"}),
        ("bash", {"command": "grep -rn 'x' . | head -5"}),
        ("edit_file", {"path": "a.py", "old_string": "a\nb", "new_string": '"{}"'}),
        ("write_file", {"path": "s.py", "content": '{"nested": {"json": true}}'}),
        ("grep", {"pattern": r"def \w+\(", "path": "src"}),
        ("plan", {"steps": ["one", "two"]}),
    ],
)
def test_every_rendered_call_parses_back_to_the_call_it_was(name, arguments):
    # The one thing that must hold. A fine-tune whose calls coder cannot parse
    # is worse than no fine-tune: the model looks fluent, the loop answers every
    # reply with a correction, and no turn ever completes.
    calls = parse_tool_calls(call_json(name, arguments))
    assert len(calls) == 1
    assert calls[0].name == name and calls[0].arguments == arguments


def test_verify_passes_a_well_formed_example():
    example = render.render(
        made([
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": call_json("read_file", {"path": "a.py"})},
            {"role": "tool", "name": "read_file", "content": "x = 1"},
            {"role": "assistant", "content": "It sets x."},
        ]),
        system="SYS",
    )
    assert render.verify(example) == []


def test_verify_catches_a_reply_carrying_two_calls():
    # One call per reply is the protocol; two in one message is a shape the loop
    # half-executes and the model should never be shown.
    example = render.Example(
        messages=[{"role": "assistant", "content": '{"name": "a", "arguments": {}}\n{"name": "b", "arguments": {}}'}],
        source="test", origin="t", system_hash="",
    )
    assert render.verify(example)


def test_a_fenced_call_is_verified_rather_than_taken_for_prose():
    # A real session recorded this shape: qwen2.5-coder wrapped its call in
    # ```json ... ```, and every check keyed on content starting with `{` read
    # the fence as prose and waved it through. verify() is what makes the
    # dataset trustworthy; a code fence used to switch it off.
    fenced = '```json\n{"name": "a", "arguments": {}}\n{"name": "b", "arguments": {}}\n```'
    example = render.Example(
        messages=[{"role": "assistant", "content": fenced}],
        source="test", origin="t", system_hash="",
    )
    assert render.verify(example)


def test_a_fenced_call_to_a_tool_that_does_not_exist_ends_the_trajectory(allowed):
    # The same blindness in `render`: unconformable meant "not a call at all",
    # so an unknown tool inside a fence was exported as an assistant answer.
    fenced = "```json\n" + call_json("WebFetch", {"url": "x"}) + "\n```"
    example = render.render(
        made([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": call_json("read_file", {"path": "a.py"})},
            {"role": "tool", "name": "read_file", "content": "x = 1"},
            {"role": "assistant", "content": "It sets x."},
            {"role": "assistant", "content": fenced},
            {"role": "tool", "name": "WebFetch", "content": "html"},
            {"role": "assistant", "content": "and the page says so"},
        ]),
        system="SYS",
        allowed=allowed,
    )
    assert [m["content"] for m in example.messages][-1] == "It sets x."


def test_a_trajectory_ending_on_a_fenced_call_is_not_an_answer(allowed):
    # `_ending_in_an_answer` exists to stop the model being taught to halt
    # mid-work. A fenced call reading as prose was exactly that lesson.
    fenced = "```json\n" + call_json("read_file", {"path": "b.py"}) + "\n```"
    example = render.render(
        made([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": call_json("read_file", {"path": "a.py"})},
            {"role": "tool", "name": "read_file", "content": "x = 1"},
            {"role": "assistant", "content": "It sets x."},
            {"role": "assistant", "content": fenced},
        ]),
        system="SYS",
        allowed=allowed,
    )
    assert [m["content"] for m in example.messages][-1] == "It sets x."


# --- conforming calls to the tools that exist ---------------------------------


def test_an_argument_this_program_would_reject_is_stripped(allowed):
    # Claude Code's Read takes `pages`; coder's read_file never has. An unknown
    # argument is not ignored at runtime -- validate_arguments raises on it, so
    # every such call would fail.
    fixed = render.conform(call_json("read_file", {"path": "a.py", "pages": "1-5"}), allowed)
    assert fixed == call_json("read_file", {"path": "a.py"})


def test_a_call_to_a_tool_that_does_not_exist_cannot_be_conformed(allowed):
    assert render.conform(call_json("WebFetch", {"url": "x"}), allowed) is None


def test_a_call_missing_a_required_argument_cannot_be_conformed(allowed):
    assert render.conform(call_json("read_file", {"pages": "1-5"}), allowed) is None


def test_an_unconformable_call_ends_the_trajectory_there(allowed):
    example = render.render(
        made([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": call_json("read_file", {"path": "a.py"})},
            {"role": "tool", "name": "read_file", "content": "x = 1"},
            {"role": "assistant", "content": "It sets x."},
            {"role": "assistant", "content": call_json("WebFetch", {"url": "x"})},
            {"role": "tool", "name": "WebFetch", "content": "html"},
            {"role": "assistant", "content": "and the page says so"},
        ]),
        system="SYS",
        allowed=allowed,
    )
    assert [m["content"] for m in example.messages][-1] == "It sets x."


# --- what the model actually reads --------------------------------------------


def test_a_tool_result_is_labelled_with_the_tool_that_produced_it():
    example = render.render(
        made([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": call_json("grep", {"pattern": "x"})},
            {"role": "tool", "name": "grep", "content": "a.py:1:x"},
            {"role": "assistant", "content": "Found it."},
        ]),
        system="SYS",
    )
    assert example.messages[3]["content"].startswith("[grep]\n")


def test_a_tool_that_returned_nothing_still_gets_a_turn():
    # A call with nothing under it teaches the model not to wait for results,
    # which is the behaviour the loop has to guard against at runtime.
    example = render.render(
        made([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": call_json("bash", {"command": "true"})},
            {"role": "tool", "name": "bash", "content": ""},
            {"role": "assistant", "content": "Nothing to report."},
        ]),
        system="SYS",
    )
    assert render.NO_OUTPUT in example.messages[3]["content"]


def test_a_huge_tool_result_is_cut_to_what_a_session_would_deliver():
    # At serve time every result has been through `truncate`. Training on an
    # unbounded one teaches the model to answer from evidence it will not get.
    example = render.render(
        made([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": call_json("bash", {"command": "cat big"})},
            {"role": "tool", "name": "bash", "content": "x" * 80_000},
            {"role": "assistant", "content": "Read it."},
        ]),
        system="SYS",
        num_ctx=16384,
    )
    assert len(example.messages[3]["content"]) < 80_000
    assert "truncated" in example.messages[3]["content"]


def test_a_trajectory_that_never_answered_renders_to_nothing():
    assert render.render(made([{"role": "user", "content": "go"}]), system="SYS") is None


def test_an_example_ends_on_an_answer_not_a_call():
    example = render.render(
        made([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "Done."},
            {"role": "assistant", "content": call_json("bash", {"command": "ls"})},
        ]),
        system="SYS",
    )
    assert example.messages[-1]["content"] == "Done."


def test_a_session_that_recorded_its_own_prompt_keeps_it():
    # A coder session happened under a prompt worth training on; a foreign one
    # happened under another agent's, and the one it must be taught under is
    # coder's.
    example = render.render(
        made([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "Done."},
        ], system="ITS OWN"),
        system="THE DEFAULT",
    )
    assert example.messages[0]["content"] == "ITS OWN"


# --- fitting a long example ---------------------------------------------------


def test_an_example_that_fits_is_left_exactly_as_it_was():
    # Elision only under pressure. An example that fits is one whose results the
    # model really would have had.
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": call_json("bash", {"command": "ls"})},
        {"role": "tool", "name": "bash", "content": "a\nb"},
        {"role": "assistant", "content": "Two files."},
    ]
    assert render.fit(messages, 10_000) == messages


def long_work(results=6, size=4000):
    """One task, several calls, an answer -- the shape that does not fit."""
    messages = [{"role": "user", "content": "go"}]
    for index in range(results):
        messages.append({"role": "assistant", "content": call_json("bash", {"command": f"c{index}"})})
        messages.append({"role": "tool", "name": "bash", "content": "y" * size})
    messages.append({"role": "assistant", "content": "Done."})
    return messages


def test_an_over_long_example_loses_its_oldest_results_first():
    fitted = render.fit(long_work(), 3000)
    assert len(fitted) == 14, "elision alone should be enough at this budget"
    elided = [m for m in fitted if m["role"] == "tool" and "[elided]" in m["content"]]
    # Named, because the note the loop writes names the tool -- fold the label
    # into the text before eliding and every result claims to come from `tool`.
    assert elided and "`bash`" in elided[0]["content"]


def test_under_real_pressure_the_task_and_the_recent_work_survive():
    # What a compacted session looks like: what was asked, and the work near the
    # end. An example that does not say what was asked teaches nothing.
    fitted = render.fit(long_work(), 2000)
    assert fitted[0] == {"role": "user", "content": "go"}
    assert fitted[-1]["content"] == "Done."
    assert [m["role"] for m in fitted[1:-1]] == ["assistant", "tool"]


def test_a_cut_never_opens_on_a_result_to_a_call_that_is_gone():
    fitted = render.fit(long_work(results=10), 1500)
    for index, message in enumerate(fitted):
        if message["role"] == "tool":
            assert fitted[index - 1]["role"] == "assistant"
            assert fitted[index - 1]["content"].lstrip().startswith("{")


def test_an_example_that_cannot_be_reduced_is_given_up_on():
    huge = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "x" * 100_000},
    ]
    assert render.fit(huge, 100) == []


def test_the_default_system_prompt_is_the_one_a_session_sends(tmp_path):
    # Built off a real registry rather than a literal, so a tool added to coder
    # is a tool the next dataset knows about.
    prompt = render.default_system(tmp_path)
    assert "read_file" in prompt and '{"name": "<tool name>", "arguments": {<arguments>}}' in prompt
