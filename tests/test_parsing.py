"""The content parser is the transport, so its shapes are tested exhaustively."""

import json

import pytest

from bkht.coder.parsing import (
    drop_empty_fences,
    extract_json_objects,
    is_tool_call,
    open_fence,
    parse_tool_calls,
    strip_json,
)


def test_bare_json_object():
    calls = parse_tool_calls('{"name": "read_file", "arguments": {"path": "src/main.py"}}')
    assert [(c.name, c.arguments) for c in calls] == [
        ("read_file", {"path": "src/main.py"})
    ]


def test_fenced_json():
    text = 'Sure.\n```json\n{"name": "grep", "arguments": {"pattern": "def main"}}\n```\n'
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["grep"]
    assert calls[0].arguments == {"pattern": "def main"}


def test_fenced_without_language_tag():
    text = '```\n{"name": "glob", "arguments": {"pattern": "**/*.py"}}\n```'
    assert [c.name for c in parse_tool_calls(text)] == ["glob"]


def test_json_wrapped_in_prose():
    text = (
        "I need to look at the file first.\n"
        '{"name": "read_file", "arguments": {"path": "a.py"}}\n'
        "Then I will decide what to change."
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].arguments["path"] == "a.py"


def test_multiple_calls_in_one_message():
    text = (
        '{"name": "read_file", "arguments": {"path": "a.py"}}\n'
        'and also\n'
        '{"name": "read_file", "arguments": {"path": "b.py"}}'
    )
    calls = parse_tool_calls(text)
    assert [c.arguments["path"] for c in calls] == ["a.py", "b.py"]


def test_top_level_array_is_flattened():
    text = '[{"name": "glob", "arguments": {"pattern": "*.py"}}, {"name": "list_files", "arguments": {}}]'
    assert [c.name for c in parse_tool_calls(text)] == ["glob", "list_files"]


def test_unparseable_text_yields_nothing():
    assert parse_tool_calls("I think the bug is in the parser, but I am not sure.") == []
    assert parse_tool_calls("") == []


def test_truncated_json_yields_nothing():
    assert parse_tool_calls('{"name": "read_file", "arguments": {"path": "a.py"') == []


def test_braces_inside_strings_do_not_confuse_the_scan():
    text = json.dumps(
        {"name": "write_file", "arguments": {"path": "x.py", "content": 'print("{}")'}}
    )
    calls = parse_tool_calls(text)
    assert calls[0].arguments["content"] == 'print("{}")'


def test_escaped_quote_inside_string():
    text = r'{"name": "bash", "arguments": {"command": "echo \"hi\""}}'
    calls = parse_tool_calls(text)
    assert calls[0].arguments["command"] == 'echo "hi"'


def test_nested_objects_survive():
    text = '{"name": "x", "arguments": {"a": {"b": {"c": 1}}}}'
    assert parse_tool_calls(text)[0].arguments == {"a": {"b": {"c": 1}}}


def test_arguments_as_json_string():
    text = '{"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}'
    assert parse_tool_calls(text)[0].arguments == {"path": "a.py"}


def test_missing_arguments_defaults_to_empty():
    assert parse_tool_calls('{"name": "list_files"}')[0].arguments == {}


@pytest.mark.parametrize("key", ["parameters", "args"])
def test_alternate_argument_keys(key):
    text = json.dumps({"name": "grep", key: {"pattern": "x"}})
    assert parse_tool_calls(text)[0].arguments == {"pattern": "x"}


def test_native_function_wrapper_shape():
    text = '{"function": {"name": "grep", "arguments": {"pattern": "x"}}}'
    assert parse_tool_calls(text)[0].name == "grep"


def test_non_tool_json_is_ignored_as_a_call():
    text = '{"severity": "high", "summary": "off by one"}'
    assert parse_tool_calls(text) == []
    assert extract_json_objects(text) == [{"severity": "high", "summary": "off by one"}]


def test_strip_json_leaves_prose():
    text = 'Reading it now.\n{"name": "read_file", "arguments": {"path": "a.py"}}\nDone.'
    assert strip_json(text) == "Reading it now.\n\nDone."


def test_strip_json_keeps_unparseable_braces():
    assert strip_json("use {placeholder} here") == "use {placeholder} here"


# --- fences left empty by a removed call --------------------------------------


def test_a_fence_around_a_tool_call_goes_with_the_call():
    # The model writes ```json around the call. Removing the call used to leave
    # the fence standing around nothing, which is the blank block that sat above
    # every tool call in a transcript.
    text = 'Here is the call:\n\n```json\n{"name": "bash", "arguments": {}}\n```\n'
    assert strip_json(text) == "Here is the call:"


def test_a_fence_with_something_in_it_is_kept():
    text = "Look:\n\n```py\nx = 1\n```"
    assert strip_json(text) == text


def test_an_unclosed_fence_is_left_alone():
    text = "```py\nx = 1"
    assert strip_json(text) == text


def test_open_fence_finds_one_still_waiting_for_its_contents():
    assert open_fence("a\n```json\n") == 2
    assert open_fence("a\n```json\nx\n```\n") is None


def test_drop_empty_fences_leaves_a_lone_fence_alone():
    assert drop_empty_fences("```json\n") == "```json\n"


# --- prose or a call ----------------------------------------------------------


def test_is_tool_call_sees_through_a_fence():
    # The shape a real session recorded. Testing the first character for `{`
    # reads this as prose, because a fence begins with a backtick.
    assert is_tool_call('```json\n{"name": "task", "arguments": {}}\n```')


def test_is_tool_call_accepts_a_bare_call_and_one_wrapped_in_prose():
    assert is_tool_call('{"name": "read_file", "arguments": {"path": "a.py"}}')
    assert is_tool_call('Let me look:\n{"name": "read_file", "arguments": {"path": "a.py"}}')


def test_is_tool_call_is_false_for_prose_and_for_two_calls():
    assert not is_tool_call("It sets x to one.")
    assert not is_tool_call("")
    # Two is not one: content carrying a pair cannot round-trip to the call it
    # was rendered from, so it is not something a caller may treat as a call.
    assert not is_tool_call('{"name": "a", "arguments": {}}{"name": "b", "arguments": {}}')
