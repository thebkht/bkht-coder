"""Reading trajectories out of every format that has them."""

import json


from bkht.coder.training import ingest
from bkht.coder.training.ingest import (
    Trajectory,
    drop_commentary,
    injected,
    pair,
    translate,
    unwrap,
)


def write(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


# --- tool translation ---------------------------------------------------------


def test_a_foreign_call_becomes_the_tool_coder_actually_has():
    # The whole reason translation exists: a model trained on Read(file_path=)
    # learns to emit a call coder has no tool for.
    assert translate("Read", {"file_path": "a.py"}) == ("read_file", {"path": "a.py"})


def test_an_argument_coder_does_not_take_is_dropped():
    # `glob` takes no path. Passing one through would train a call that fails
    # schema validation every single time.
    assert translate("Glob", {"pattern": "*.py", "path": "/w/src"}) == (
        "glob", {"pattern": "*.py"},
    )


def test_absolute_paths_become_workspace_relative():
    assert translate("Read", {"file_path": "/w/src/a.py"}, "/w")[1] == {"path": "src/a.py"}


def test_a_path_outside_the_workspace_is_left_alone():
    assert translate("Read", {"file_path": "/etc/hosts"}, "/w")[1] == {"path": "/etc/hosts"}


def test_an_argv_list_becomes_a_command_line():
    assert translate("shell", {"command": ["ls", "-la"]})[1] == {"command": "ls -la"}


def test_a_tool_with_no_equivalent_translates_to_nothing():
    assert translate("WebFetch", {"url": "https://x"}) is None


def test_a_plan_arrives_as_the_list_of_steps_coder_takes():
    # Codex sends [{step, status}] and coder's tool takes strings; the statuses
    # are dropped because coder marks one step done per call.
    assert translate("update_plan", {"plan": [{"step": "A", "status": "done"}]}) == (
        "plan", {"steps": ["A"]},
    )


def test_claude_spells_the_same_plan_differently():
    assert translate("TodoWrite", {"todos": [{"content": "B"}]}) == ("plan", {"steps": ["B"]})


def test_a_delegation_keeps_the_brief_and_drops_the_agent_type():
    # coder has exactly one sub-agent, so there is no type to carry.
    assert translate("Agent", {"prompt": "find it", "subagent_type": "Explore"}) == (
        "task", {"instruction": "find it"},
    )


def test_a_call_whose_arguments_carried_nothing_is_dropped():
    assert translate("TodoWrite", {"todos": []}) is None


# --- protocol shaping ---------------------------------------------------------


def test_parallel_calls_are_interleaved_with_their_own_results():
    # Both foreign agents call in parallel; coder's protocol is one call per
    # reply. Replayed as written, this trains the model to do what the system
    # prompt forbids.
    messages = [
        {"role": "assistant", "content": '{"a": 1}', "call_id": "1"},
        {"role": "assistant", "content": '{"a": 2}', "call_id": "2"},
        {"role": "tool", "name": "bash", "content": "one", "call_id": "1"},
        {"role": "tool", "name": "bash", "content": "two", "call_id": "2"},
    ]
    assert [m["content"] for m in pair(messages)] == ['{"a": 1}', "one", '{"a": 2}', "two"]


def test_a_call_whose_result_never_arrived_ends_the_trajectory():
    # Everything after it answers evidence that is not there.
    messages = [
        {"role": "assistant", "content": '{"a": 1}', "call_id": "1"},
        {"role": "assistant", "content": "done"},
    ]
    assert pair(messages) == []


def test_commentary_before_a_call_is_removed():
    messages = [
        {"role": "assistant", "content": "Let me look."},
        {"role": "assistant", "content": '{"name": "bash"}'},
    ]
    assert [m["content"] for m in drop_commentary(messages)] == ['{"name": "bash"}']


def test_two_paragraphs_of_commentary_both_go():
    # The bug a forward pass has: it drops the first, keeps the second, and
    # reports itself finished having created the adjacency it removes.
    messages = [
        {"role": "assistant", "content": "First I will."},
        {"role": "assistant", "content": "Then I will."},
        {"role": "assistant", "content": '{"name": "bash"}'},
    ]
    assert len(drop_commentary(messages)) == 1


def test_an_answer_is_not_commentary():
    messages = [
        {"role": "tool", "name": "bash", "content": "out"},
        {"role": "assistant", "content": "It sets x."},
        {"role": "user", "content": "thanks"},
    ]
    assert len(drop_commentary(messages)) == 3


# --- foreign framing ----------------------------------------------------------


def test_a_harness_preamble_is_not_a_user_turn():
    assert injected("<environment_context>\n  <cwd>/w</cwd>")
    assert injected("# AGENTS.md instructions for /w")
    assert not injected("read the file and tell me what it does")


def test_codex_exec_framing_is_stripped():
    wrapped = "Chunk ID: abc\nWall time: 0.1 seconds\nProcess exited with code 0\nOutput:\nhello"
    assert unwrap(wrapped) == "hello"


def test_the_other_codex_framing_is_stripped_too():
    wrapped = "Command: /bin/zsh -lc 'ls'\nChunk ID: abc\nOutput:\nhello"
    assert unwrap(wrapped) == "hello"


def test_a_result_that_merely_contains_the_word_output_is_untouched():
    # `Output:` is an ordinary word. Splitting on a later one would cut away the
    # first half of a result that was never wrapped.
    plain = "running tests\nOutput:\n3 passed"
    assert unwrap(plain) == plain


# --- coder's own sessions -----------------------------------------------------


def test_a_coder_session_is_read_back_in_its_own_schema(tmp_path):
    path = write(tmp_path / "s.jsonl", [
        {"type": "session", "id": "s", "cwd": "/w", "model": "m", "provider": "claude-code"},
        {"type": "prompt", "system": "SYS", "tools": ["read_file"]},
        {"type": "message", "role": "user", "content": "read it"},
        {"type": "message", "role": "assistant", "content": '{"name": "read_file"}'},
        {"type": "message", "role": "tool", "name": "read_file", "content": "x = 1"},
        {"type": "message", "role": "assistant", "content": "It sets x."},
        {"type": "outcome", "stopped": "answered", "iterations": 2},
    ])
    found = list(ingest.from_coder(path))
    assert len(found) == 1
    assert found[0].system == "SYS" and found[0].provider == "claude-code"
    assert len(found[0].messages) == 4


def test_a_turn_that_did_not_answer_is_cut_off_not_kept(tmp_path):
    # Dropping the whole session throws away the good prefix; keeping the bad
    # turn teaches the model to reproduce it.
    path = write(tmp_path / "s.jsonl", [
        {"type": "session", "id": "s", "cwd": "/w"},
        {"type": "prompt", "system": "SYS"},
        {"type": "message", "role": "user", "content": "one"},
        {"type": "message", "role": "assistant", "content": "done"},
        {"type": "outcome", "stopped": "answered"},
        {"type": "message", "role": "user", "content": "two"},
        {"type": "message", "role": "assistant", "content": "lost"},
        {"type": "outcome", "stopped": "iteration-cap"},
    ])
    found = list(ingest.from_coder(path))
    assert [m["content"] for m in found[0].messages] == ["one", "done"]
    assert found[0].truncated


def test_a_prompt_record_starts_a_new_example(tmp_path):
    # The tool set moved, so the messages after it are a different lesson from
    # the ones before.
    path = write(tmp_path / "s.jsonl", [
        {"type": "session", "id": "s", "cwd": "/w"},
        {"type": "prompt", "system": "FIRST"},
        {"type": "message", "role": "user", "content": "one"},
        {"type": "message", "role": "assistant", "content": "a"},
        {"type": "outcome", "stopped": "answered"},
        {"type": "prompt", "system": "SECOND"},
        {"type": "message", "role": "user", "content": "two"},
        {"type": "message", "role": "assistant", "content": "b"},
        {"type": "outcome", "stopped": "answered"},
    ])
    assert [t.system for t in ingest.from_coder(path)] == ["FIRST", "SECOND"]


def test_a_session_written_before_outcomes_existed_yields_nothing(tmp_path):
    # Correct, not a gap: without an outcome there is no way to know the turn
    # went anywhere, and 500 files on this machine are in that state.
    path = write(tmp_path / "s.jsonl", [
        {"type": "session", "id": "s", "cwd": "/w"},
        {"type": "message", "role": "user", "content": "one"},
        {"type": "message", "role": "assistant", "content": "a"},
    ])
    assert [t for t in ingest.from_coder(path) if t.messages] == []


# --- an outside corpus --------------------------------------------------------


def test_a_chat_jsonl_is_read_and_its_tools_translated(tmp_path):
    path = write(tmp_path / "c.jsonl", [
        {
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "read it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "Read", "arguments": '{"file_path": "a.py"}'}}
                    ],
                },
                {"role": "tool", "name": "Read", "content": "x = 1"},
                {"role": "assistant", "content": "It sets x."},
            ]
        }
    ])
    found = list(ingest.from_chat(path))
    assert found[0].system == "SYS"
    assert json.loads(found[0].messages[1]["content"])["name"] == "read_file"
    assert found[0].messages[2]["name"] == "read_file"


def test_a_line_that_is_not_a_conversation_is_skipped(tmp_path):
    path = write(tmp_path / "c.jsonl", [{"text": "not this"}, {"messages": "not this either"}])
    assert list(ingest.from_chat(path)) == []


def test_an_unreadable_line_costs_that_line_only(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"broken\n{"messages": [{"role": "user", "content": "hi"}]}\n')
    assert len(list(ingest.records(path))) == 1


def test_short_exchanges_are_not_collected(tmp_path, monkeypatch):
    # A question and an answer with no work in it is not a short example of
    # agentic work; the work is the thing being trained.
    trajectory = Trajectory(source="chat", origin="x", messages=[{"role": "user", "content": "hi"}])
    assert list(ingest._long_enough([trajectory], 4)) == []
