"""Rendering the model's markdown as it arrives.

The model writes markdown whether or not anything reads it, so without this
every `**`, backtick and `###` lands on the screen literally and a numbered
list arrives as a wall with asterisks in it.

Line-oriented, and that is the whole design. Prose reaches this module token by
token through :class:`streaming.Gate`, and buffering a turn to render it at the
end would undo the streaming it was worth building in the first place. What
makes it work is that every construct the model actually emits is settled by the
end of its own line: a heading, a bullet, a bold span. Only the fence state
crosses a line boundary, and one boolean carries it.

Not a markdown parser. It is a renderer for the subset a chat model writes,
which is why there is no nested-list model, no reference link, and no table:
those would be code that never runs against real output.
"""

from __future__ import annotations

import re

from . import highlight
from .terminal import ACCENT, BOLD, DIM, ITALIC, RESET, REVERSE, UNDERLINE, width

BULLET = "•"
QUOTE_BAR = "│"

# --- inline spans -----------------------------------------------------------

CODE = re.compile(r"(`+)(.+?)\1", re.DOTALL)
BOLD_SPAN = re.compile(r"\*\*(\S(?:.*?\S)?)\*\*|__(\S(?:.*?\S)?)__")
# A single asterisk is emphasis anywhere; an underscore is emphasis only
# between non-word characters, or `snake_case_name` would come back italic
# with its middle eaten.
ITALIC_SPAN = re.compile(
    r"\*(\S(?:.*?\S)?)\*"
    r"|(?<![A-Za-z0-9_])_(\S(?:.*?\S)?)_(?![A-Za-z0-9_])"
)
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")

#: Code spans are lifted out before anything else runs, so the markers inside
#: one stay literal. The sentinel is a control character the model cannot emit
#: through a JSON string body, which is what makes it safe as a placeholder.
SENTINEL = "\x00"
PLACEHOLDER = re.compile(f"{SENTINEL}(\\d+){SENTINEL}")


def inline(text: str, colour: bool = True) -> str:
    """Render the spans inside one line, markers removed.

    Order is load-bearing: code first, so ``**`` inside a code span is left
    alone; bold before italic, so ``**a**`` is not read as an emphasised ``*a*``.
    """
    spans: list[str] = []

    def stash(match: re.Match) -> str:
        body = match.group(2)
        spans.append(f"{REVERSE} {body} {RESET}" if colour else body)
        return f"{SENTINEL}{len(spans) - 1}{SENTINEL}"

    text = CODE.sub(stash, text)
    text = BOLD_SPAN.sub(lambda m: _wrap(m.group(1) or m.group(2), BOLD, colour), text)
    text = ITALIC_SPAN.sub(lambda m: _wrap(m.group(1) or m.group(2), ITALIC, colour), text)
    text = LINK.sub(lambda m: _wrap(m.group(1), UNDERLINE, colour), text)
    return PLACEHOLDER.sub(lambda m: spans[int(m.group(1))], text)


def _wrap(body: str, code: str, colour: bool) -> str:
    return f"{code}{body}{RESET}" if colour else body


# --- block shapes -----------------------------------------------------------

FENCE = re.compile(r"^\s*(?:```|~~~)")
HEADING = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
LIST_ITEM = re.compile(r"^(\s*)[-*+]\s+(.*)$")
ORDERED = re.compile(r"^(\s*)(\d+[.)])\s+(.*)$")
BLOCKQUOTE = re.compile(r"^\s*>\s?(.*)$")
# Three or more of the same marker, and nothing else on the line.
THEMATIC_BREAK = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")


class Renderer:
    """One complete source line to one rendered line, or to nothing at all.

    Stateful only in ``fenced``: everything else about a line is visible in the
    line itself.
    """

    def __init__(self, colour: bool = True, columns: int | None = None) -> None:
        self.colour = colour
        self.columns = columns
        self.fenced = False

    def reset(self) -> None:
        """Forget the block state, so the next turn starts outside a fence."""
        self.fenced = False

    def _rule(self) -> int:
        return self.columns if self.columns is not None else width()

    def line(self, text: str) -> str | None:
        """Render one line. ``None`` means the line is chrome and is dropped."""
        text = text.rstrip("\r")

        if FENCE.match(text):
            # The fence itself says nothing a reader needs: the indent and the
            # colouring below already mark the block as code.
            self.fenced = not self.fenced
            return None

        if self.fenced:
            body = highlight.code(text) if self.colour else text
            return f"  {body}"

        if not text.strip():
            return ""

        if THEMATIC_BREAK.match(text):
            return self._paint("─" * max(0, self._rule() - 1), DIM)

        if match := HEADING.match(text):
            # The hashes stay on screen. They are how a reader tells a second-
            # level heading from a third when both are the same colour.
            hashes, title = match.groups()
            return self._paint(f"{hashes} {title}", ACCENT + BOLD)

        if match := BLOCKQUOTE.match(text):
            return self._paint(f"{QUOTE_BAR} {match.group(1)}", DIM)

        if match := ORDERED.match(text):
            indent, marker, body = match.groups()
            return f"{indent}{marker} {inline(body, self.colour)}"

        if match := LIST_ITEM.match(text):
            indent, body = match.groups()
            return f"{indent}{BULLET} {inline(body, self.colour)}"

        return inline(text, self.colour)

    def _paint(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.colour else text


#: A line that could still turn out to be a heading, a bullet, a fence, a rule
#: or an indented block -- anything :meth:`Renderer.line` does not render with
#: :func:`inline` alone. Deliberately over-broad: a paragraph that happens to
#: open with a digit simply waits for its newline, which is no worse than every
#: line waiting.
BLOCKY = re.compile(r"^\s|^[#>*+\-`~_]|^\d")

#: Every character that can open an inline span. Text after an unclosed one is
#: not settled: `a *b` renders differently once its closing star arrives.
OPENERS = "`*_["


def settled(text: str) -> int:
    """How much of a partial line renders the same whatever arrives next.

    The prefix before the first still-open span. An opener that never closes on
    this line blocks the rest of it, which costs nothing worse than waiting for
    the newline -- the same thing that would have happened anyway.
    """
    index = 0
    while index < len(text):
        character = text[index]
        if character not in OPENERS:
            index += 1
            continue
        if character == "[":
            close = text.find(")", index + 1)
            step = 1
        else:
            marker = "**" if text.startswith("**", index) else character
            close = text.find(marker, index + len(marker))
            step = len(marker)
        if close < 0:
            return index
        index = close + step
    return len(text)


class Stream:
    """Renders fed text as it arrives, a settled piece at a time.

    Two things reach ``emit``: a finished line, and -- while one is still being
    written -- the part of it no later token can change. The second is what
    keeps a paragraph appearing as the model writes it rather than all at once
    when it finally reaches a newline, which is the responsiveness
    :mod:`streaming` exists to protect.

    What ``emit`` receives in total never depends on how the text was chunked:
    feeding a document one character at a time and feeding it whole produce the
    same output, only cut in different places.
    """

    def __init__(self, emit, colour: bool = True, renderer: Renderer | None = None) -> None:
        self.emit = emit
        self.renderer = renderer or Renderer(colour=colour)
        self.buffer = ""
        # Rendered characters of the line in progress that are already on
        # screen. Everything emitted afterwards is the delta past this.
        self.shown = 0

    def feed(self, text: str) -> None:
        if not text:
            return
        self.buffer += text
        while "\n" in self.buffer:
            line, _, self.buffer = self.buffer.partition("\n")
            self._write(line)
        self._preview()

    def finish(self) -> None:
        """Flush the unterminated tail and reset for the next turn."""
        if self.buffer:
            self._write(self.buffer)
        self.buffer = ""
        self.renderer.reset()

    def _preview(self) -> None:
        """Show the settled head of the line still being written."""
        # Inside a fence the body is syntax-coloured as a whole line, and a
        # block-shaped line has a prefix -- a bullet, a heading -- that is not
        # decided until more of it exists.
        if self.renderer.fenced or BLOCKY.match(self.buffer):
            return
        head = self.buffer[: settled(self.buffer)]
        if not head:
            return
        rendered = inline(head, self.renderer.colour)
        if len(rendered) > self.shown:
            self.emit(rendered[self.shown :])
            self.shown = len(rendered)

    def _write(self, line: str) -> None:
        rendered = self.renderer.line(line)
        shown, self.shown = self.shown, 0
        if rendered is None:
            return
        self.emit(rendered[shown:] + "\n")


def render(text: str, colour: bool = True) -> str:
    """A whole document, for callers that already have all of it."""
    out: list[str] = []
    stream = Stream(out.append, colour=colour)
    stream.feed(text)
    stream.finish()
    return "".join(out)
