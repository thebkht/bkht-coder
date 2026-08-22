"""What the model writes, and what a reader should see instead."""

import pytest

from bkht.coder import markdown
from bkht.coder.markdown import Renderer, Stream, inline, render


def plain(text: str) -> str:
    """Rendered with no colour: structure only, and no escape bytes."""
    return render(text, colour=False)


# --- inline spans -----------------------------------------------------------


def test_the_markers_never_reach_the_screen():
    # Measured without colour: an SGR sequence is full of brackets of its own,
    # and asking whether a `[` survived would find those instead.
    out = inline("**a** `b` [c](http://d) *e*", colour=False)
    for marker in ("**", "`", "[", "](", "http://d"):
        assert marker not in out
    assert out == "a b c e"


def test_each_span_gets_its_own_treatment():
    assert markdown.BOLD in inline("**a**")
    assert markdown.REVERSE in inline("`a`")
    assert markdown.UNDERLINE in inline("[a](b)")
    assert markdown.ITALIC in inline("*a*")


def test_an_underscore_inside_a_word_is_not_emphasis():
    # The whole reason the underscore rule is narrower than the asterisk one.
    assert inline("snake_case_name", colour=False) == "snake_case_name"
    assert inline("read_file and write_file", colour=False) == "read_file and write_file"


def test_an_underscore_between_words_still_is():
    assert inline("_really_ fast", colour=False) == "really fast"


def test_markers_inside_a_code_span_stay_literal():
    # Code is lifted out first, so nothing else gets a chance at its contents.
    assert "**" in inline("`**not bold**`", colour=False)


def test_bold_is_not_read_as_two_italics():
    assert inline("**a**", colour=False) == "a"


def test_an_unpaired_marker_is_left_alone():
    assert inline("2 * 3 * 4", colour=False) == "2 * 3 * 4"
    assert inline("a lone ` backtick", colour=False) == "a lone ` backtick"


# --- block shapes -----------------------------------------------------------


def test_a_heading_keeps_its_hashes():
    # They are how a reader tells one level from another when both are blue.
    assert plain("### Key Findings\n") == "### Key Findings\n"
    assert markdown.ACCENT in render("### Key Findings\n")


def test_a_list_item_becomes_a_bullet():
    assert plain("- one\n* two\n+ three\n") == "• one\n• two\n• three\n"


def test_a_nested_item_keeps_its_indent():
    assert plain("  - deeper\n") == "  • deeper\n"


def test_a_numbered_item_keeps_its_number():
    assert plain("1. one\n2. two\n") == "1. one\n2. two\n"


def test_a_blockquote_is_drawn_as_a_bar():
    assert plain("> quoted\n") == "│ quoted\n"


def test_a_thematic_break_becomes_a_rule():
    out = Renderer(colour=False, columns=11).line("---")
    assert out == "─" * 10


def test_a_break_is_not_mistaken_for_a_bullet():
    assert plain("---\n") != "• --\n"


def test_a_fence_is_dropped_and_its_body_is_indented():
    assert plain("```python\ncode = 1\n```\n") == "  code = 1\n"


def test_a_fenced_body_keeps_its_markers():
    # Inside a fence `**` is not bold, it is two asterisks the code contains.
    assert plain("```\na ** b\n- not a bullet\n```\n") == "  a ** b\n  - not a bullet\n"


def test_blank_lines_survive_because_they_are_the_paragraphs():
    assert plain("one\n\ntwo\n") == "one\n\ntwo\n"


def test_no_colour_means_no_escape_bytes_at_all():
    doc = "# h\n\n- **a** `b` [c](d)\n\n> q\n\n```\nx\n```\n\n---\n"
    assert "\033" not in plain(doc)
    assert "\x00" not in plain(doc)


# --- the streaming property -------------------------------------------------

DOCUMENTS = [
    "### Heading\n\nSome **bold** prose with `code` in it.\n",
    "- one\n- two\n\n1. first\n2. second\n",
    "before\n```py\ndef f():\n    return 1\n```\nafter\n",
    "> quoted\n\n---\n\ntrailing line with no newline",
    "",
]


@pytest.mark.parametrize("doc", DOCUMENTS)
def test_chunking_cannot_change_the_output(doc):
    """The property the whole line-oriented design exists to guarantee.

    Tokens arrive in whatever sizes the model and the network produce, so a
    renderer whose output depended on where the chunks fell would render the
    same answer differently on every run.
    """
    whole: list[str] = []
    stream = Stream(whole.append, colour=False)
    stream.feed(doc)
    stream.finish()

    piecemeal: list[str] = []
    stream = Stream(piecemeal.append, colour=False)
    for character in doc:
        stream.feed(character)
    stream.finish()

    assert "".join(piecemeal) == "".join(whole)


def test_a_line_is_emitted_as_soon_as_it_is_complete():
    # Not at the end of the turn: the point of streaming is seeing it early.
    seen: list[str] = []
    stream = Stream(seen.append, colour=False)
    stream.feed("first line\nsecond")
    assert seen == ["first line\n"]


def test_the_unterminated_tail_is_flushed_at_the_end():
    seen: list[str] = []
    stream = Stream(seen.append, colour=False)
    stream.feed("no trailing newline")
    stream.finish()
    assert seen == ["no trailing newline\n"]


def test_finishing_closes_an_unterminated_fence():
    # A turn that dies mid-code-block must not leave the next one indented.
    stream = Stream(lambda _: None, colour=False)
    stream.feed("```\ncode\n")
    stream.finish()
    assert not stream.renderer.fenced
