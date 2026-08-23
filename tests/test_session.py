"""Session persistence and resume."""

import json

import pytest

from bkht.coder.session import Session


@pytest.fixture
def store(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    return directory


def test_messages_are_appended_as_they_happen(store):
    session = Session(system="sys", cwd="/work", model="m")
    session.start_file(store)
    session.add_user("hello")

    records = [json.loads(line) for line in session.path.read_text().splitlines()]
    assert records[0]["type"] == "session" and records[0]["cwd"] == "/work"
    assert records[1] == {"type": "message", "role": "user", "content": "hello"}


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


def test_system_prompt_is_not_persisted(store):
    # It is rebuilt on resume, so a session picks up the current tool set.
    session = Session(system="old prompt", cwd="/work")
    session.start_file(store)
    session.add_user("hi")
    assert "old prompt" not in session.path.read_text()
    assert Session.load(session.path, system="new prompt").system == "new prompt"


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
