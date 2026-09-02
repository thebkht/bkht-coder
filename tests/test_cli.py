"""How loop events reach the screen, on a terminal and off one."""

from __future__ import annotations

import io
import re

from bkht.coder import cli
from bkht.coder.agent import Agent
from bkht.coder.cli import TerminalListener, report, run_turn, summarize
from bkht.coder.parsing import ToolCall
from bkht.coder.session import Session
from bkht.coder.tools.base import Registry, ToolResult

from fakes import FakeProvider, call


def listener(**kwargs) -> tuple[TerminalListener, io.StringIO]:
    stream = io.StringIO()
    return TerminalListener(stream=stream, **kwargs), stream


#: Everything the status line uses to take a row back: a line clear before each
#: row it paints, and a walk up to the top of the block followed by an erase to
#: the bottom of the screen. Both are boundaries between what was drawn and
#: what came next.
ERASED = re.compile(r"(?:\x1b\[A)*\r\x1b\[(?:2K|J)")


def visible_text(stream) -> str:
    """What survives on screen: spinner frames are erased, prose is not."""
    out = []
    for chunk in ERASED.split(stream.getvalue()):
        if not chunk.startswith("\x1b[2m"):  # a dim spinner frame
            out.append(re.sub(r"\x1b\[[0-9?]*[a-zA-Z]", "", chunk))
    return "".join(out).strip()


# --- off a terminal: nothing may change -------------------------------------


def test_a_pipe_gets_the_tool_lines_and_nothing_else():
    # This output is parsed by scripts and by the live suite; a spinner or a
    # streamed token appearing here is a regression, not a feature.
    listen, stream = listener(live=False)
    with listen.turn():
        listen.on_token("thinking out loud")
        listen.on_tool_call(ToolCall(name="read_file", arguments={"path": "a.py"}))
        listen.on_tool_result(
            ToolCall(name="read_file", arguments={"path": "a.py"}),
            ToolResult.success("contents"),
        )

    assert stream.getvalue() == "● read_file(a.py)\n"


def test_the_line_for_a_call_waits_until_its_result_is_known():
    # The mark says which way the call went, so it cannot be drawn before that
    # is known -- a line the terminal has scrolled cannot be repainted.
    listen, stream = listener(live=False)
    with listen.turn():
        listen.on_tool_call(ToolCall(name="read_file", arguments={"path": "a.py"}))
        assert stream.getvalue() == ""


def test_a_pipe_still_reports_tool_errors():
    listen, stream = listener(live=False)
    with listen.turn():
        listen.on_tool_result(ToolCall(name="read_file"), ToolResult.failure("no such file"))
    assert stream.getvalue() == "● read_file()\n  ! no such file\n"


def test_a_single_argument_call_drops_the_argument_name():
    assert summarize({"path": "a.py"}) == "a.py"
    assert summarize({"path": "a.py", "limit": 5}) == "path=a.py, limit=5"


def test_nothing_is_streamed_when_not_live_so_report_still_prints():
    listen, _ = listener(live=False)
    with listen.turn():
        listen.on_token("some prose")
    assert not listen.streamed


# --- on a terminal ----------------------------------------------------------


def test_prose_streams_but_the_tool_call_does_not():
    listen, stream = listener(live=True)
    with listen.turn():
        listen.on_token("I'll read the file. ")
        listen.on_token(call("read_file", path="a.py"))

    shown = stream.getvalue()
    assert "I'll read the file." in shown
    assert "read_file" not in shown, "the call must reach the tool line, not the screen"
    assert listen.streamed


def test_verbose_keeps_streaming_raw_and_skips_the_gate():
    listen, stream = listener(live=True, verbose=True)
    with listen.turn():
        listen.on_token('{"name": "read_file"}')
    assert '{"name": "read_file"}' in stream.getvalue()


def test_an_unclosed_brace_is_released_when_the_turn_ends():
    listen, stream = listener(live=True)
    with listen.turn():
        listen.on_token('Consider {"a": 1')
    assert visible_text(stream) == 'Consider {"a": 1'


def test_the_spinner_never_repaints_over_prose():
    # It redraws with a carriage return and a line clear, so sharing a line
    # with half-written prose would erase it rather than interleave.
    listen, stream = listener(live=True)
    with listen.turn():
        listen.on_token("half a sentence")
        listen.status._draw()
        listen.status._draw()
    assert visible_text(stream) == "half a sentence"


# --- report -----------------------------------------------------------------


def test_report_does_not_print_prose_that_was_already_streamed(capsys):
    class Outcome:
        answer, stopped, errors = "the answer", "answered", []

    assert report(Outcome(), streamed=True) == 0
    assert capsys.readouterr().out == ""

    assert report(Outcome(), streamed=False) == 0
    assert capsys.readouterr().out == "the answer\n"


def test_report_renders_the_markdown_it_prints(capsys):
    # A piped run never reaches the listener's renderer, and an answer full of
    # raw asterisks is no more readable for having been redirected.
    class Outcome:
        answer, stopped, errors = "### Head\n\n- **bold** item", "answered", []

    report(Outcome(), streamed=False)
    assert capsys.readouterr().out == "### Head\n\n• bold item\n"


def test_a_streamed_turn_shows_its_answer_exactly_once():
    # The whole point of the streamed=... handshake: prose on screen once.
    listen, stream = listener(live=True)
    provider = FakeProvider(["All done."])
    agent = Agent(
        provider=provider, registry=Registry([]), session=Session(), listener=listen
    )

    assert run_turn(agent, listen, "hi") == 0
    assert visible_text(stream).splitlines()[0] == "All done."


def test_a_finished_turn_says_what_it_cost():
    listen, stream = listener(live=True)
    provider = FakeProvider(["All done."])
    agent = Agent(
        provider=provider, registry=Registry([]), session=Session(), listener=listen
    )

    run_turn(agent, listen, "hi")
    # The fake reports one completion token per character, and a fixed prompt.
    assert visible_text(stream).splitlines()[-1] == "0.0s (↑100 ↓9)"


def test_a_turn_with_no_answer_gets_no_footer():
    listen, stream = listener(live=True)
    listen.footer(type("O", (), {"answer": "", "seconds": 1.0, "sent": 1, "received": 1})())
    assert visible_text(stream) == ""


# --- the rhythm of a turn ---------------------------------------------------


def blocks(stream) -> list[str]:
    """What the turn drew, split on the blank lines between its blocks."""
    return visible_text(stream).split("\n\n")


def test_every_block_of_a_turn_is_given_a_line_of_its_own():
    # Grouping here is done by proximity, and everything arriving flush leaves
    # nothing for the eye to group by.
    listen, stream = listener(live=True)
    with listen.turn():
        listen.on_tool_result(
            ToolCall("read_file", {"path": "a.py"}), ToolResult.success("x")
        )
        listen.on_tool_result(
            ToolCall("grep", {"pattern": "def"}), ToolResult.success("y")
        )
        listen.on_token("Here is the answer.")
    listen.footer(
        type("O", (), {"answer": "x", "seconds": 2.0, "sent": 10, "received": 5})()
    )

    assert blocks(stream) == [
        "● read_file(a.py)\n  1 line",
        "● grep(def)\n  1 match",
        "Here is the answer.",
        "2s (↑10 ↓5)",
    ]


def test_a_failure_stays_against_the_call_it_belongs_to():
    # Proximity is the grouping: a gap here would offer the error as its own
    # event rather than as what that call did.
    listen, stream = listener(live=True)
    with listen.turn():
        listen.on_tool_result(
            ToolCall("read_file", {"path": "a.py"}), ToolResult.failure("no such file")
        )
    assert blocks(stream) == ["● read_file(a.py)\n  ! no such file"]


def test_a_pipe_gets_no_gaps_because_they_are_chrome():
    listen, stream = listener(live=False)
    with listen.turn():
        listen.on_tool_result(
            ToolCall("read_file", {"path": "a.py"}), ToolResult.success("x")
        )
        listen.on_tool_result(
            ToolCall("grep", {"pattern": "def"}), ToolResult.success("y")
        )
    assert stream.getvalue() == "● read_file(a.py)\n● grep(def)\n"


# --- why the loop paused ----------------------------------------------------


def test_retry_reason_is_reported_not_replaced():
    """The reason is printed, not overwritten with a guess.

    This line used to read "retrying (malformed reply)" whatever the cause,
    including the caller that fires most often -- freeing context. A real
    diagnosis followed it in the wrong direction for two turns.
    """
    reporter, stream = listener()
    reporter.on_retry("compacted earlier turns to free context")
    assert "compacted earlier turns to free context" in stream.getvalue()
    assert "malformed" not in stream.getvalue()


def test_a_genuine_format_failure_still_says_so():
    reporter, stream = listener()
    reporter.on_retry("retrying: no tool call and no answer")
    assert "no tool call and no answer" in stream.getvalue()


def test_a_tool_result_is_summarised_under_its_call():
    # The call used to be printed and the result never was, so a model writing
    # out a file it had lost to compaction was indistinguishable, on screen,
    # from one that had really read it.
    listen, stream = listener(live=True)
    with listen.turn():
        listen.on_tool_result(
            ToolCall("grep", {"pattern": "def"}),
            ToolResult.success("a.py:1: def x\nb.py:2: def y"),
        )
    assert "2 matches in 2 files" in stream.getvalue()


def test_a_failed_call_says_why_and_not_what_it_returned():
    listen, stream = listener(live=True)
    with listen.turn():
        listen.on_tool_result(
            ToolCall("read_file", {"path": "gone.py"}),
            ToolResult.failure("no such file"),
        )
    written = stream.getvalue()
    assert "! no such file" in written
    assert "line" not in written


# --- the block pinned under a running turn ----------------------------------


class _Permissions:
    mode = "ask"


class _Line:
    """A StatusLine with nothing to look up."""

    def fields(self):
        return {
            "name": "bkht-coder", "branch": "main", "ratio": 0.24,
            "model": "qwen2.5-coder:7b", "spent": 3100, "note": "", "width": 100,
        }


def test_the_pinned_block_is_the_prompt_frame_with_nothing_in_it():
    rows = cli.pinned_block(_Permissions(), _Line(), stream=io.StringIO())()
    assert len(rows) == 5
    rule, prompt, rule_again, status, mode = rows
    assert set(rule.strip()) == {"─"} and rule == rule_again
    # Empty: this is not an editor, and a box with words in it would promise a
    # turn could read what was typed into it.
    assert prompt.strip() == "›"
    assert "bkht-coder" in status and "(main)" in status and "24% used" in status
    assert mode == "▸▸ ask mode on (shift+tab to cycle)"


def test_the_pinned_block_says_what_the_turn_is_changing():
    # Asked again on every frame rather than built once, so the meter and the
    # token count move while the turn runs.
    line = _Line()
    block = cli.pinned_block(_Permissions(), line, stream=io.StringIO())
    before = block()[3]
    line.fields = lambda: {**_Line().fields(), "ratio": 0.91, "spent": 12000}
    assert block()[3] != before
    assert "91% used" in block()[3]
