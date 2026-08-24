import io

from bkht.coder import lineedit
from bkht.coder.lineedit import Editor


def editor(**kwargs) -> Editor:
    kwargs.setdefault("stdout", io.StringIO())
    return Editor(**kwargs)


def type(ed: Editor, keys: str) -> str | None:
    return ed._consume(keys, "> ")


def test_typing_and_submitting_returns_the_line():
    ed = editor()
    assert type(ed, "hello") is None
    assert type(ed, "\r") == "hello"


def test_submitted_lines_join_the_history():
    ed = editor()
    type(ed, "first\r")
    type(ed, "second\r")
    assert ed.history == ["first", "second"]


def test_blank_lines_are_not_remembered():
    ed = editor()
    type(ed, "   \r")
    assert ed.history == []


def test_backspace_deletes_before_the_cursor():
    ed = editor()
    type(ed, "abc\x7f")
    assert ed.buffer == "ab"


def test_arrows_move_the_cursor_and_insert_lands_there():
    ed = editor()
    type(ed, "ac\x1b[D" "b")
    assert (ed.buffer, ed.cursor) == ("abc", 2)


def test_home_and_end():
    ed = editor()
    type(ed, "abc\x01")
    assert ed.cursor == 0
    type(ed, "\x05")
    assert ed.cursor == 3


def test_ctrl_u_kills_to_the_start_and_ctrl_k_to_the_end():
    ed = editor()
    type(ed, "one two\x15")
    assert ed.buffer == ""
    type(ed, "one two\x01\x0b")
    assert ed.buffer == ""


def test_ctrl_w_kills_a_word():
    ed = editor()
    type(ed, "one two\x17")
    assert ed.buffer == "one "


def test_up_recalls_history_and_down_gives_the_draft_back():
    ed = editor(history=["earlier"])
    ed.recall = len(ed.history)
    type(ed, "half")
    type(ed, "\x1b[A")
    assert ed.buffer == "earlier"
    type(ed, "\x1b[B")
    assert ed.buffer == "half"


def test_tab_completes_a_slash_command():
    ed = editor(completions=lambda: ["/model", "/mode"])
    type(ed, "/mod\t")
    assert ed.buffer == "/mode"


def test_tab_stops_at_the_shared_prefix_when_several_match():
    ed = editor(completions=lambda: ["/diff", "/doctor"])
    type(ed, "/d\t")
    assert ed.buffer == "/d"


def test_tab_leaves_a_plain_word_alone():
    ed = editor(completions=lambda: ["/model"])
    type(ed, "add a flag\t")
    assert ed.buffer == "add a flag"


def test_shift_tab_calls_cycle_and_keeps_the_draft():
    seen = []
    ed = editor(cycle=lambda: seen.append(True))
    type(ed, "half written\x1b[Z")
    assert seen == [True]
    assert ed.buffer == "half written"


def test_escape_tab_is_the_same_key():
    seen = []
    ed = editor(cycle=lambda: seen.append(True))
    type(ed, "\x1b\t")
    assert seen == [True]


def test_unknown_escape_sequences_are_swallowed_whole():
    ed = editor()
    type(ed, "\x1b[200~a")
    assert ed.buffer == "a"


def test_ctrl_d_on_an_empty_line_is_end_of_file():
    ed = editor()
    try:
        type(ed, "\x04")
    except EOFError:
        return
    raise AssertionError("expected EOFError")


def test_ctrl_d_with_text_typed_does_nothing():
    ed = editor()
    type(ed, "text\x04")
    assert ed.buffer == "text"


def test_the_submitted_line_is_left_in_the_scrollback():
    out = io.StringIO()
    ed = editor(stdout=out)
    type(ed, "a task\r")
    assert out.getvalue().endswith("> a task\n")


def test_footer_names_the_mode_and_the_shortcut():
    assert lineedit.footer("auto") == "▸▸ auto mode on (shift+tab to cycle)"
    assert lineedit.footer("ask", cycles=False) == "▸▸ ask mode on"


def test_available_is_false_off_a_terminal():
    assert lineedit.available(stdin=io.StringIO(), stdout=io.StringIO()) is False
