"""Reading sessions, including the ones other agents wrote."""

from __future__ import annotations

import json

import pytest

from bkht.coder import transcripts


def write(path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


# --- coder --------------------------------------------------------------------


def test_a_coder_session_reads_back(tmp_path):
    path = tmp_path / "20260828-221324-945f6f.jsonl"
    write(path, [
        {"type": "session", "id": "20260828-221324-945f6f",
         "cwd": "/w", "model": "qwen", "created": 1_700_000_000.0},
        {"type": "message", "role": "user", "content": "hello"},
        {"type": "message", "role": "assistant", "content": "hi there"},
    ])
    made = transcripts.read_coder(path)
    assert (made.cwd, made.short) == ("/w", "945f6f")
    assert [(t.role, t.text) for t in made.turns] == [("user", "hello"), ("assistant", "hi there")]


def test_a_clear_record_drops_what_came_before(tmp_path):
    path = tmp_path / "s.jsonl"
    write(path, [
        {"type": "session", "id": "s", "cwd": "/w"},
        {"type": "message", "role": "user", "content": "forgotten"},
        {"type": "clear"},
        {"type": "message", "role": "user", "content": "kept"},
    ])
    assert [t.text for t in transcripts.read_coder(path).turns] == ["kept"]


def test_a_tool_call_is_taken_out_of_an_assistant_turn(tmp_path):
    # On this model a tool call travels as ordinary content. A transcript read
    # back with it left in is mostly JSON.
    path = tmp_path / "s.jsonl"
    write(path, [
        {"type": "session", "id": "s", "cwd": "/w"},
        {"type": "message", "role": "assistant",
         "content": 'Reading it.\n```json\n{"name": "read_file", "arguments": {"path": "a"}}\n```'},
        {"type": "message", "role": "assistant",
         "content": '{"name": "read_file", "arguments": {"path": "b"}}'},
    ])
    # The second turn was nothing but a call, so it drops out entirely.
    assert [t.text for t in transcripts.read_coder(path).turns] == ["Reading it."]


def test_tool_results_are_not_turns(tmp_path):
    path = tmp_path / "s.jsonl"
    write(path, [
        {"type": "session", "id": "s", "cwd": "/w"},
        {"type": "message", "role": "tool", "name": "grep", "content": "a.py:1: x"},
        {"type": "message", "role": "user", "content": "and?"},
    ])
    assert [t.role for t in transcripts.read_coder(path).turns] == ["user"]


# --- claude code --------------------------------------------------------------


def claude_records():
    return [
        {"type": "user", "uuid": "a", "parentUuid": None, "cwd": "/w",
         "timestamp": "2026-08-28T10:00:00.000Z", "message": {"role": "user", "content": "first"}},
        {"type": "assistant", "uuid": "b", "parentUuid": "a", "cwd": "/w",
         "timestamp": "2026-08-28T10:00:01.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "thinking", "thinking": "hmm"},
             {"type": "text", "text": "an answer"},
             {"type": "tool_use", "name": "Read", "input": {}},
         ]}},
        {"type": "user", "uuid": "c", "parentUuid": "b", "cwd": "/w",
         "timestamp": "2026-08-28T10:00:02.000Z",
         "message": {"role": "user", "content": [{"type": "tool_result", "content": "x"}]}},
        {"type": "last-prompt", "leafUuid": "c", "sessionId": "s"},
    ]


def test_a_claude_session_keeps_prose_and_drops_the_machinery(tmp_path):
    path = tmp_path / "1be46299-bddc-43f4-9a66-f56bc9d13b19.jsonl"
    write(path, claude_records())
    made = transcripts.read_claude(path)
    assert made.cwd == "/w"
    assert made.short == "1be46299"
    # Thinking and tool_use are not conversation; a block-list user message is
    # a tool result being fed back, not a person typing.
    assert [(t.role, t.text) for t in made.turns] == [
        ("user", "first"), ("assistant", "an answer"),
    ]


def test_only_the_live_branch_of_a_rewound_session_is_read(tmp_path):
    # The file is a tree: a rewind starts a sibling branch beside the one it
    # replaced, so replaying in order shows the same work twice.
    path = tmp_path / "s.jsonl"
    write(path, [
        {"type": "user", "uuid": "a", "parentUuid": None, "cwd": "/w",
         "message": {"role": "user", "content": "first"}},
        {"type": "assistant", "uuid": "abandoned", "parentUuid": "a",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "throw away"}]}},
        {"type": "assistant", "uuid": "kept", "parentUuid": "a",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "keep"}]}},
        {"type": "last-prompt", "leafUuid": "kept", "sessionId": "s"},
    ])
    assert [t.text for t in transcripts.read_claude(path).turns] == ["first", "keep"]


def test_a_session_with_no_leaf_is_read_straight_through(tmp_path):
    path = tmp_path / "s.jsonl"
    write(path, claude_records()[:2])
    assert len(transcripts.read_claude(path).turns) == 2


# --- codex --------------------------------------------------------------------


def test_a_codex_session_reads_the_event_view_only(tmp_path):
    # response_item and event_msg are two views of the same turn; reading both
    # yields every message twice.
    path = tmp_path / "rollout-2026-08-23T00-33-02-01a02a1a-9b76-7680-bfab-d00898eaa21a.jsonl"
    write(path, [
        {"timestamp": "2026-08-23T00:33:02.000Z", "type": "session_meta",
         "payload": {"session_id": "01a02a1a-9b76-7680", "cwd": "/w"}},
        {"timestamp": "2026-08-23T00:33:03.000Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": "do a thing"}},
        {"timestamp": "2026-08-23T00:33:04.000Z", "type": "response_item",
         "payload": {"type": "message", "role": "user", "content": [{"text": "do a thing"}]}},
        {"timestamp": "2026-08-23T00:33:05.000Z", "type": "event_msg",
         "payload": {"type": "agent_message", "message": "done"}},
        {"timestamp": "2026-08-23T00:33:06.000Z", "type": "event_msg",
         "payload": {"type": "token_count", "info": {}}},
    ])
    made = transcripts.read_codex(path)
    assert made.cwd == "/w"
    assert [(t.role, t.text) for t in made.turns] == [
        ("user", "do a thing"), ("assistant", "done"),
    ]


# --- robustness ---------------------------------------------------------------


def test_a_half_written_last_line_costs_that_line_only(tmp_path):
    # These files are appended to by another process while this reads them.
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"type": "session", "id": "s", "cwd": "/w"}) + "\n"
        + json.dumps({"type": "message", "role": "user", "content": "kept"}) + "\n"
        + '{"type": "message", "role": "user", "cont',
        encoding="utf-8",
    )
    assert [t.text for t in transcripts.read_coder(path).turns] == ["kept"]


def test_a_file_that_makes_no_sense_costs_its_own_session(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("not json at all\n", encoding="utf-8")
    assert transcripts.load(path, transcripts.CODER).turns == []


def test_a_missing_file_is_not_an_error(tmp_path):
    assert transcripts.read_coder(tmp_path / "gone.jsonl").turns == []


@pytest.mark.parametrize("source", transcripts.SOURCES)
def test_every_source_has_a_reader_and_a_header_reader(source):
    assert source in transcripts.READERS
    assert source in transcripts.HEADS
