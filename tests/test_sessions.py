"""Listing, finding, and inspecting saved sessions."""

from __future__ import annotations

import json
import os

import pytest

from bkht.coder import sessions
from bkht.coder.session import Session, find, headers, sessions_for


@pytest.fixture
def store(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    return directory


def write(store, identifier: str, cwd: str, *, model: str = "m", messages: int = 1,
          created: float = 1000.0) -> Session:
    session = Session(cwd=os.path.realpath(cwd), model=model)
    session.id = identifier
    session.start_file(store)
    # start_file stamps the real clock; the tests need a fixed one.
    lines = session.path.read_text().splitlines()
    header = json.loads(lines[0])
    header["created"] = created
    session.path.write_text("\n".join([json.dumps(header)] + lines[1:]) + "\n")
    for index in range(messages):
        session.add_user(f"message {index}")
    return session


# --- the store ---------------------------------------------------------------


def test_sessions_are_listed_newest_first_for_one_workspace(store, tmp_path):
    write(store, "20260101-000000-aaa", str(tmp_path / "a"))
    write(store, "20260102-000000-bbb", str(tmp_path / "a"))
    write(store, "20260103-000000-ccc", str(tmp_path / "b"))

    found = sessions_for(str(tmp_path / "a"), store)
    assert [info.id for info in found] == ["20260102-000000-bbb", "20260101-000000-aaa"]

    everywhere = sessions_for(None, store)
    assert len(everywhere) == 3


def test_the_count_is_what_survived_a_clear(store, tmp_path):
    session = write(store, "20260101-000000-aaa", str(tmp_path), messages=3)
    session.clear()
    session.add_user("after")

    assert sessions_for(str(tmp_path), store)[0].messages == 1


def test_a_truncated_last_line_does_not_lose_the_session(store, tmp_path):
    session = write(store, "20260101-000000-aaa", str(tmp_path), messages=2)
    with session.path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "message", "role": "us')

    assert sessions_for(str(tmp_path), store)[0].messages == 2


def test_a_file_that_is_not_a_session_is_skipped(store, tmp_path):
    write(store, "20260101-000000-aaa", str(tmp_path))
    (store / "notes.jsonl").write_text("not json at all\n")

    assert [path.name for path, _ in headers(None, store)] == ["20260101-000000-aaa.jsonl"]


def test_find_takes_an_exact_id_or_an_unambiguous_prefix(store, tmp_path):
    write(store, "20260101-000000-aaa", str(tmp_path))
    write(store, "20260101-000000-abb", str(tmp_path))

    assert find("20260101-000000-aaa", store).name == "20260101-000000-aaa.jsonl"
    assert find("20260101-000000-ab", store).name == "20260101-000000-abb.jsonl"
    # Ambiguous, so nothing: resuming the wrong conversation is the worse loss.
    assert find("20260101", store) is None
    assert find("nope", store) is None


def test_latest_for_still_answers_from_the_shared_scan(store, tmp_path):
    write(store, "20260101-000000-aaa", str(tmp_path))
    write(store, "20260102-000000-bbb", str(tmp_path))

    assert Session.latest_for(str(tmp_path), store).name == "20260102-000000-bbb.jsonl"


# --- ago ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [(5, "just now"), (300, "5m ago"), (7200, "2h ago"), (100000, "yesterday"),
     (400000, "4d ago")],
)
def test_ago_uses_the_coarsest_unit_that_still_says_it(seconds, expected):
    assert sessions.ago(1_000_000 - seconds, now=1_000_000) == expected


def test_a_session_with_no_creation_time_is_not_a_lie():
    assert sessions.ago(0) == "unknown"


# --- the commands ------------------------------------------------------------


def collect(store, monkeypatch):
    monkeypatch.setattr("bkht.coder.session.SESSION_DIR", store)
    lines = []
    return lines, lines.append


def test_report_lists_a_row_per_session(store, tmp_path, monkeypatch):
    write(store, "20260101-000000-aaa", str(tmp_path), model="qwen", messages=4)
    lines, out = collect(store, monkeypatch)

    assert sessions.report(tmp_path, out=out) == 0
    assert "20260101-000000-aaa" in lines[0]
    assert "qwen" in lines[0] and "4 msgs" in lines[0]


def test_report_says_so_rather_than_printing_nothing(store, tmp_path, monkeypatch):
    lines, out = collect(store, monkeypatch)
    assert sessions.report(tmp_path, out=out) == 0
    assert lines == [f"No saved sessions for {tmp_path}."]


def test_report_as_json_is_parseable(store, tmp_path, monkeypatch):
    write(store, "20260101-000000-aaa", str(tmp_path), messages=2)
    lines, out = collect(store, monkeypatch)

    assert sessions.report(tmp_path, as_json=True, out=out) == 0
    payload = json.loads(lines[0])
    assert payload[0]["id"] == "20260101-000000-aaa" and payload[0]["messages"] == 2


def test_show_prints_the_tail_of_the_transcript(store, tmp_path, monkeypatch):
    write(store, "20260101-000000-aaa", str(tmp_path), messages=sessions.PREVIEW + 3)
    lines, out = collect(store, monkeypatch)

    assert sessions.show(tmp_path, "last", out=out) == 0
    text = "\n".join(lines)
    assert "20260101-000000-aaa" in text
    assert "3 earlier messages" in text
    assert f"message {sessions.PREVIEW + 2}" in text
    assert "message 0" not in text


def test_show_reports_an_id_that_is_not_there(store, tmp_path, monkeypatch, capsys):
    lines, out = collect(store, monkeypatch)
    assert sessions.show(tmp_path, "nope", out=out) == 1
    assert "No session matches 'nope'" in capsys.readouterr().err
    assert lines == []
