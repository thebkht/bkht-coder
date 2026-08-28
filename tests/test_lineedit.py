import io

from bkht.coder import lineedit, terminal
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


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


def test_each_mode_wears_its_own_colour():
    tty = FakeTTY()
    ask, auto, plan = (lineedit.footer(mode, stream=tty) for mode in ("ask", "auto", "plan"))
    assert terminal.ORANGE in auto and terminal.ACCENT in plan
    assert len({ask, auto, plan}) == 3


def test_an_unknown_mode_is_drawn_quietly_rather_than_not_at_all():
    tty = FakeTTY()
    assert "banana mode on" in lineedit.footer("banana", stream=tty)


# --- multi-line ---------------------------------------------------------------


def paste(ed: Editor, text: str) -> str | None:
    """Type `text` the way a terminal in bracketed paste delivers it."""
    return type(ed, f"\x1b[200~{text}\x1b[201~")


def test_a_pasted_newline_is_text_not_a_submission():
    ed = editor()
    assert paste(ed, "one\ntwo\nthree") is None
    assert ed.buffer == "one\ntwo\nthree"


def test_enter_after_a_paste_submits_the_whole_block():
    ed = editor()
    paste(ed, "one\ntwo")
    assert type(ed, "\r") == "one\ntwo"


def test_a_typed_newline_still_submits():
    ed = editor()
    type(ed, "hello")
    assert type(ed, "\r") == "hello"


def test_alt_enter_opens_a_line():
    ed = editor()
    type(ed, "one\x1b\rtwo")
    assert ed.buffer == "one\ntwo"


def test_a_trailing_backslash_opens_a_line():
    ed = editor()
    assert type(ed, "one\\\r") is None
    assert ed.buffer == "one\n"
    assert type(ed, "two\r") == "one\ntwo"


def test_a_pasted_tab_stays_a_tab():
    ed = editor(completions=lambda: ("/help",))
    paste(ed, "\tindented")
    assert ed.buffer == "\tindented"


def test_tab_outside_a_paste_still_completes():
    ed = editor(completions=lambda: ("/help",))
    type(ed, "/he\t")
    assert ed.buffer == "/help"


def test_up_and_down_move_between_lines():
    ed = editor()
    type(ed, "one\x1b\rtwo")
    type(ed, "\x1b[A")
    assert ed._locate() == (0, 3)
    type(ed, "\x1b[B")
    assert ed._locate() == (1, 3)


def test_up_on_the_first_line_still_recalls_history():
    ed = editor(history=["earlier"])
    type(ed, "one\x1b\rtwo")
    type(ed, "\x1b[A\x1b[A")  # onto line one, then past it
    assert ed.buffer == "earlier"


def test_the_column_is_kept_when_the_line_above_is_shorter():
    ed = editor()
    type(ed, "ab\x1b\rlonger")
    type(ed, "\x1b[A")
    assert ed.cursor == 2  # clamped to the end of "ab", not past it


def test_ctrl_a_and_ctrl_e_act_on_the_current_line():
    ed = editor()
    type(ed, "one\x1b\rtwo")
    type(ed, "\x01")
    assert ed.cursor == 4
    type(ed, "\x05")
    assert ed.cursor == 7


def test_ctrl_u_kills_to_the_start_of_the_line_only():
    ed = editor()
    type(ed, "one\x1b\rtwo\x15")
    assert ed.buffer == "one\n"


def test_ctrl_k_kills_to_the_end_of_the_line_only():
    ed = editor()
    type(ed, "one\x1b\rtwo\x01\x0b")
    assert ed.buffer == "one\n"


def test_a_long_paste_is_folded_to_one_numbered_line():
    ed = editor()
    paste(ed, "\n".join(f"line {n}" for n in range(40)))
    assert ed.buffer == "[Pasted text #1, 40 lines]"


def test_a_second_paste_gets_its_own_number():
    ed = editor()
    body = "\n".join(f"line {n}" for n in range(40))
    paste(ed, body)
    paste(ed, body)
    assert ed.buffer == "[Pasted text #1, 40 lines][Pasted text #2, 40 lines]"


def test_both_folded_pastes_are_put_back_in_order():
    ed = editor()
    first = "\n".join(f"one {n}" for n in range(40))
    second = "\n".join(f"two {n}" for n in range(40))
    paste(ed, first)
    paste(ed, second)
    assert type(ed, "\r") == first + second


def test_a_folded_paste_is_put_back_when_the_line_is_sent():
    ed = editor()
    body = "\n".join(f"line {n}" for n in range(40))
    paste(ed, body)
    type(ed, " please review")
    assert type(ed, "\r") == f"{body} please review"


def test_a_short_paste_is_left_alone():
    ed = editor()
    paste(ed, "one\ntwo")
    assert ed.buffer == "one\ntwo"
    assert ed.pastes == {}


def test_a_paste_at_the_threshold_is_still_shown_whole():
    ed = editor()
    body = "\n".join(f"line {n}" for n in range(lineedit.PASTE_LINES))
    paste(ed, body)
    assert ed.buffer == body


def test_a_folded_paste_grows_the_block_by_one_row_not_forty():
    ed = editor()
    paste(ed, "\n".join(f"line {n}" for n in range(40)))
    ed._redraw("> ")
    # rule, the chip, rule -- and no footer on this editor.
    assert ed.drawn == 3


def test_history_remembers_the_fold_not_the_whole_file():
    # Up should not paste forty lines back into a block that folded them.
    ed = editor()
    paste(ed, "\n".join(f"line {n}" for n in range(40)))
    type(ed, "\r")
    assert ed.history == ["[Pasted text #1, 40 lines]"]


def test_a_paste_split_across_two_reads_still_folds():
    # The end marker can be cut in half by the end of a read. Half of `[201~`
    # taken for text is a paste that never ends and a block that never folds.
    ed = editor()
    body = "\n".join(f"line {n}" for n in range(40))
    chunk = f"\x1b[200~{body}\x1b[201~"
    cut = len(chunk) - 3
    type(ed, chunk[:cut])
    type(ed, chunk[cut:])
    assert ed.buffer == "[Pasted text #1, 40 lines]"
    assert not ed.pasting


def test_nothing_is_drawn_while_a_paste_is_still_arriving():
    # Each frame is taller than the last; once the block outgrows the window
    # the erase cannot reach the top and every frame is left behind.
    out = io.StringIO()
    ed = editor(stdout=out)
    type(ed, "\x1b[200~" + "\n".join(f"line {n}" for n in range(40)))
    out.truncate(0)
    out.seek(0)
    ed._paint("> ")
    assert out.getvalue() == ""
    type(ed, "\x1b[201~")
    ed._paint("> ")
    assert out.getvalue() != ""


def test_escape_alone_does_not_swallow_the_next_key():
    ed = editor()
    type(ed, "\x1b")
    type(ed, "a")
    assert ed.buffer == "a"


# --- images -------------------------------------------------------------------


def test_ctrl_v_puts_a_chip_in_the_line_and_keeps_the_path():
    ed = editor(attach=lambda: "/tmp/one.png")
    type(ed, "what is this? \x16")
    assert ed.buffer == "what is this? [Image #1]"
    assert ed.images == ["/tmp/one.png"]


def test_a_second_image_is_numbered_after_the_first():
    ed = editor(attach=lambda: "/tmp/x.png")
    type(ed, "\x16 and \x16")
    assert ed.buffer == "[Image #1] and [Image #2]"
    assert len(ed.images) == 2


def test_an_empty_clipboard_leaves_the_line_alone():
    ed = editor(attach=lambda: None)
    type(ed, "hello\x16")
    assert ed.buffer == "hello"
    assert ed.images == []


def test_the_path_is_announced_when_it_is_attached():
    said = []
    ed = editor(attach=lambda: "/tmp/one.png", on_image=said.append)
    type(ed, "\x16")
    assert said == ["/tmp/one.png"]


def test_ctrl_v_does_nothing_when_there_is_no_clipboard_to_ask():
    ed = editor()
    type(ed, "hello\x16")
    assert ed.buffer == "hello"
