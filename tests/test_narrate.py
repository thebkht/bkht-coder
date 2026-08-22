"""Turning a tool call into a sentence."""

from __future__ import annotations

import pytest

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
