"""Session persistence and resume."""

import json

import pytest

from bkht.coder.session import Session


@pytest.fixture
def store(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    return directory


def records_in(session):
    return [json.loads(line) for line in session.path.read_text().splitlines()]


def test_messages_are_appended_as_they_happen(store):
    session = Session(system="sys", cwd="/work", model="m")
    session.start_file(store)
    session.add_user("hello")

    records = records_in(session)
    assert records[0]["type"] == "session" and records[0]["cwd"] == "/work"
    # The prompt comes between the header and the first message, and that order
    # is the point: every message in the file was answered under the prompt
    # above it, so a reader replaying the session never has to guess which.
    assert records[1]["type"] == "prompt" and records[1]["system"] == "sys"
    assert records[2] == {"type": "message", "role": "user", "content": "hello"}


def test_round_trip_preserves_the_conversation(store):
    session = Session(system="sys", cwd="/work", model="m")
    session.start_file(store)
    session.add_user("read it")
    session.add_assistant('{"name": "read_file", "arguments": {"path": "a.py"}}')
    session.add_tool_result("read_file", "1\tx = 1")
    session.add_assistant("It sets x.")

    loaded = Session.load(session.path, system="sys")
    assert loaded.messages == session.messages
    assert loaded.cwd == "/work" and loaded.model == "m"


def test_the_system_prompt_is_recorded_but_never_replayed(store):
    # Written down, because it is the input half of every exchange in the file
    # and cannot be reconstructed later -- it names the tool set of the run that
    # produced it. Not replayed, because this run has its own.
    session = Session(system="old prompt", cwd="/work")
    session.start_file(store)
    session.add_user("hi")

    loaded = Session.load(session.path, system="new prompt")
    assert loaded.system == "new prompt"
    assert [r["system"] for r in loaded.recorded["prompt"]] == ["old prompt"]


def test_a_resumed_session_records_the_prompt_it_was_given(store):
    # Two tool sets, one file. Appending rather than replacing is what lets a
    # reader pair each message with the prompt that was in force for it.
    session = Session(system="first", cwd="/work")
    session.start_file(store)
    session.add_user("one")

    resumed = Session.load(session.path)
    resumed.record_prompt("second", ["read_file"])
    resumed.add_user("two")

    kinds = [r.get("type") for r in records_in(resumed)]
    assert kinds == ["session", "prompt", "message", "prompt", "message"]
    assert [r["system"] for r in Session.load(session.path).recorded["prompt"]] == [
        "first", "second",
    ]


def test_an_outcome_says_how_the_turn_ended(store):
    class Turn:
        stopped, iterations, tool_calls = "iteration-cap", 25, 12
        errors, seconds, sent, received = ["gave up"], 91.5, 9000, 400

    session = Session(cwd="/work")
    session.start_file(store)
    session.record_outcome(Turn())

    written = records_in(session)[-1]
    assert written["type"] == "outcome" and written["stopped"] == "iteration-cap"
    assert written["iterations"] == 25 and written["errors"] == ["gave up"]


def test_the_header_names_the_backend_that_answered(store):
    session = Session(cwd="/work", model="m", provider="local")
    session.start_file(store)
    assert Session.load(session.path).provider == "local"


def test_a_session_written_before_any_of_this_still_loads(store):
    # The fields are all optional on read, because 500 files on this machine
    # were written without them.
    path = store / "old.jsonl"
    path.write_text(
        json.dumps({"type": "session", "id": "old", "cwd": "/work", "model": "m"})
        + "\n"
        + json.dumps({"type": "message", "role": "user", "content": "hi"})
        + "\n"
    )
    loaded = Session.load(path)
    assert loaded.provider == "" and loaded.recorded.get("prompt") is None
    assert [m["content"] for m in loaded.messages] == ["hi"]


def test_a_partial_final_line_is_skipped_not_fatal(store):
    session = Session(cwd="/work")
    session.start_file(store)
    session.add_user("hello")
    with session.path.open("a") as handle:
        handle.write('{"type": "message", "role": "assi')

    loaded = Session.load(session.path)
    assert [m["content"] for m in loaded.messages] == ["hello"]


def test_clear_is_persisted_and_replayed(store):
    session = Session(cwd="/work")
    session.start_file(store)
    session.add_user("first")
    session.clear()
    session.add_user("second")

    loaded = Session.load(session.path)
    assert [m["content"] for m in loaded.messages] == ["second"]


def test_latest_for_picks_the_newest_matching_directory(store):
    old = Session(cwd="/work", id="20200101-000000-aaaaaa")
    old.start_file(store)
    other = Session(cwd="/elsewhere", id="20990101-000000-cccccc")
    other.start_file(store)
    new = Session(cwd="/work", id="20300101-000000-bbbbbb")
    new.start_file(store)

    assert Session.latest_for("/work", store) == new.path


def test_latest_for_returns_none_when_nothing_matches(store):
    Session(cwd="/elsewhere").start_file(store)
    assert Session.latest_for("/work", store) is None
    assert Session.latest_for("/work", store / "missing") is None


def test_unsaved_session_still_works(tmp_path):
    session = Session(system="sys")
    session.add_user("hello")
    assert session.path is None
    assert session.payload()[0]["role"] == "system"


def test_an_unwritable_file_does_not_take_the_session_down(store):
    session = Session(cwd="/work")
    session.start_file(store)
    session.path = store / "missing-dir" / "x.jsonl"
    session.add_user("hello")  # must not raise
    assert session.path is None
    assert session.messages[-1]["content"] == "hello"


# --- the language reminder --------------------------------------------------


def test_no_reminder_without_a_language():
    session = Session(system="sys")
    session.add_user("hello")
    assert session.payload() == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]


def test_no_reminder_for_english():
    session = Session(system="sys", language="English")
    session.add_user("hello")
    assert len(session.payload()) == 2


def test_reminder_is_last_and_names_the_language():
    session = Session(system="sys", language="Uzbek")
    session.add_user("salom")
    last = session.payload()[-1]
    # A user turn, not a system one. A fresh system message arriving as the very
    # last thing before generation reads to a small model like the request it is
    # meant to answer, and it answers that instead of doing the work.
    assert last["role"] == "user"
    assert "Uzbek" in last["content"]


def test_reminder_is_never_stored_or_persisted(store):
    session = Session(system="sys", language="Uzbek", cwd="/work")
    session.start_file(store)
    session.add_user("salom")
    session.payload()
    session.payload()

    assert [m["content"] for m in session.messages] == ["salom"]
    reloaded = Session.load(session.path, system="sys")
    assert [m["content"] for m in reloaded.messages] == ["salom"]
    # A reloaded session has not been spoken to yet, so it has no language.
    assert reloaded.language is None
    assert len(reloaded.payload()) == 2


# --- the plan ---------------------------------------------------------------


def test_a_resumed_session_brings_its_plan_back(tmp_path):
    # The plan is persisted for exactly this: a session resumed after a crash,
    # or tomorrow, opens against the list it was working down rather than a
    # blank one.
    session = Session(cwd=str(tmp_path))
    path = session.start_file(tmp_path)
    session.add_user("review the review module")
    session.set_plan(["read reviewer.py", "read ci.py", "write it up"])
    session.tick_plan(1)

    back = Session.load(path)
    assert back.plan.render() == session.plan.render()
    assert back.plan.progress() == (1, 3)


def test_only_the_last_plan_written_is_the_plan(tmp_path):
    # The file is append-only, so it holds every version the plan ever had.
    # Replaying them in order is what makes the last one win.
    session = Session(cwd=str(tmp_path))
    path = session.start_file(tmp_path)
    session.set_plan(["first idea", "second"])
    session.set_plan(["it was wrong", "do this instead"])

    back = Session.load(path)
    assert [step.text for step in back.plan.steps] == ["it was wrong", "do this instead"]


def test_clearing_a_persisted_session_clears_its_plan_on_reload(tmp_path):
    session = Session(cwd=str(tmp_path))
    path = session.start_file(tmp_path)
    session.set_plan(["something"])
    session.clear()

    assert not Session.load(path).plan


def test_a_plan_made_before_the_file_existed_is_written_into_it(tmp_path):
    # Otherwise the plan would be the one piece of the session the transcript
    # did not describe.
    session = Session(cwd=str(tmp_path))
    session.set_plan(["made first"])
    path = session.start_file(tmp_path)

    assert Session.load(path).plan.render() == "1. [ ] made first"


def test_the_language_reminder_and_the_plan_both_ride_on_the_payload(tmp_path):
    # Two reminders, in order, and the plan is last: it is what the model reads
    # immediately before writing its reply.
    session = Session(system="sys")
    session.language = "Uzbek"
    session.set_plan(["one"])
    payload = session.payload()

    assert payload[0]["role"] == "system"
    assert "Uzbek" in payload[1]["content"]
    assert "1. [ ] one" in payload[2]["content"]
    assert session.messages == []
