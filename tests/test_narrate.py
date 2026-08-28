"""Turning a tool call into a sentence."""

from __future__ import annotations

import pytest

from bkht.coder import narrate
from bkht.coder.narrate import intent
from bkht.coder.parsing import ToolCall


@pytest.mark.parametrize(
    "call, expected",
    [
        (ToolCall("read_file", {"path": "avg.py"}), "Reading avg.py"),
        (ToolCall("write_file", {"path": "new.py"}), "Writing new.py"),
        (ToolCall("edit_file", {"path": "cli.py"}), "Editing cli.py"),
        (ToolCall("list_files", {"path": "src"}), "Listing src"),
        (ToolCall("grep", {"pattern": "TODO"}), "Searching for TODO"),
        (ToolCall("codebase_search", {"terms": "verbose, flag"}), "Looking for verbose, flag"),
        (ToolCall("bash", {"command": "pytest -q"}), "Running pytest -q"),
    ],
)
def test_each_tool_gets_a_sentence(call, expected):
    assert intent(call) == expected


def test_a_call_without_its_argument_still_reads_as_a_sentence():
    # A small model omits arguments constantly; the line must not become
    # "Reading None".
    assert intent(ToolCall("read_file", {})) == "Reading a file"
    assert intent(ToolCall("bash", {})) == "Running a command"


def test_an_unknown_tool_is_still_narrated():
    assert intent(ToolCall("teleport", {})) == "Calling teleport"


def test_long_arguments_are_cut_to_one_line():
    line = intent(ToolCall("bash", {"command": "echo " + "x" * 200}))
    assert len(line) < 80
    assert "\n" not in line


def test_a_multiline_argument_stays_on_one_line():
    assert "\n" not in intent(ToolCall("bash", {"command": "a\nb\nc"}))


# --- what came back -----------------------------------------------------------


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


def test_grep_counts_matches_and_the_files_they_are_in():
    content = "a.py:1: x\na.py:9: x\nb.py:3: x"
    assert narrate.outcome(call("grep"), content) == "3 matches in 2 files"


def test_one_match_is_not_pluralised():
    assert narrate.outcome(call("grep"), "a.py:1: x") == "1 match in 1 file"


def test_read_file_counts_lines():
    assert narrate.outcome(call("read_file"), "one\ntwo\nthree") == "3 lines"


def test_glob_counts_paths():
    assert narrate.outcome(call("glob"), "a.py\nb.py") == "2 paths"


def test_a_one_line_answer_is_repeated_as_it_stands():
    # `edit_file` answers in a sentence; a count of it would say less.
    assert narrate.outcome(call("edit_file"), "Edited bkht/coder/cli.py") == (
        "Edited bkht/coder/cli.py"
    )


def test_a_long_single_line_becomes_a_count():
    assert narrate.outcome(call("bash"), "x" * 200) == "200 characters"


def test_nothing_at_all_says_so():
    assert narrate.outcome(call("bash"), "") == "nothing"
