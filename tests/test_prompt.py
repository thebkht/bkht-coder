"""Prompt history and slash-command completion."""

from __future__ import annotations

from bkht.coder import prompt
from bkht.coder.prompt import Completer, Reader, commands
from bkht.coder.repl import Repl


def test_commands_are_read_off_the_repl_itself():
    found = commands(Repl(agent=None, snapshots=None, permissions=None, workspace=None))
    # Whatever the table happens to hold, these are the ones users type most.
    for expected in ("/help", "/exit", "/diff", "/review", "/model", "/mode"):
        assert expected in found


def test_a_new_command_is_completable_without_a_second_list():
    class Extended(Repl):
        def do_teleport(self, argument):
            pass

    assert "/teleport" in commands(Extended(None, None, None, None))


def test_the_completer_offers_matching_commands():
    completer = Completer(["/help", "/diff", "/model", "/mode"])
    assert completer("/mo", 0) == "/model"
    assert completer("/mo", 1) == "/mode"
    assert completer("/mo", 2) is None


def test_the_completer_ignores_ordinary_prose():
    # "add a flag" must not be completed into a command.
    completer = Completer(["/help"])
    assert completer("add", 0) is None


def test_history_survives_a_restart(tmp_path):
    # Round-tripped rather than read as text: libedit (macOS) escapes spaces in
    # the file, so asserting on its contents would test the format, not that
    # the line comes back.
    history = tmp_path / "nested" / "history"
    reader = Reader(repl=None, history=history)
    if reader.readline is None:
        return  # no readline on this platform; nothing to assert

    readline = reader.readline
    readline.clear_history()
    readline.add_history("add a --verbose flag")
    reader.save()
    assert history.exists()

    readline.clear_history()
    Reader(repl=None, history=history)
    assert readline.get_history_item(1) == "add a --verbose flag"


def test_a_disabled_reader_touches_no_files(tmp_path):
    history = tmp_path / "history"
    Reader(repl=None, history=history, enabled=False).save()
    assert not history.exists()


def test_a_terminal_gets_the_editor_and_the_shortcut(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt.lineedit, "available", lambda *a, **k: True)
    reader = Reader(repl=None, history=tmp_path / "history")
    assert reader.cycles is True
    assert reader.editor is not None


def test_without_raw_mode_the_shortcut_is_not_offered(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt.lineedit, "available", lambda *a, **k: False)
    reader = Reader(repl=None, history=tmp_path / "history")
    assert reader.cycles is False


def test_the_editor_writes_history_the_readline_path_can_read(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt.lineedit, "available", lambda *a, **k: True)
    history = tmp_path / "nested" / "history"
    reader = Reader(repl=None, history=history)
    reader.editor.history.append("add a --verbose flag")
    reader.save()

    assert Reader(repl=None, history=history).editor.history == ["add a --verbose flag"]
