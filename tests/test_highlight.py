"""Diff colouring: markers, syntax, and staying out of the way."""

from __future__ import annotations

from bkht.coder.highlight import code, diff
from bkht.coder.terminal import BOLD, CYAN, GREEN, MAGENTA, RED, RESET, YELLOW


def strip(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_colour_never_changes_the_text_itself():
    # The thing being approved must read identically once the colour is gone,
    # or the preview is lying about the change.
    body = '@@ -1,3 +1,3 @@\n-def average(n):\n+def mean(n):  # renamed\n     return sum(n)'
    assert strip(diff(body)) == body


def test_disabled_colour_is_a_passthrough():
    body = "-a\n+b"
    assert diff(body, colour=False) == body


def test_added_and_removed_lines_take_their_marker_colour():
    assert diff("+new").startswith(GREEN)
    assert diff("-old").startswith(RED)


def test_hunk_and_file_headers_are_distinguished():
    assert diff("@@ -1 +1 @@").startswith(CYAN)
    assert diff("--- a/x.py").startswith(BOLD)


def test_a_removed_line_is_not_mistaken_for_a_file_header():
    # "---" starts a header, but "--x" is a deletion. Order matters.
    assert diff("-  x = 1").startswith(RED)


def test_strings_and_keywords_are_coloured_inside_a_line():
    coloured = code('return "hello"')
    assert MAGENTA in coloured, "return is a keyword"
    assert YELLOW in coloured, "the string literal is coloured"


def test_colour_returns_to_the_line_base_after_each_token():
    # Otherwise the rest of an added line loses its green after the first
    # string, and the diff stops reading as a diff.
    coloured = diff('+    x = "s"  # note')
    assert coloured.count(GREEN) > 1
    assert coloured.endswith(RESET)


def test_an_unterminated_string_does_not_swallow_the_line():
    # Diff hunks are fragments; half a string literal is normal input.
    assert strip(diff('+    text = "unterminated')) == '+    text = "unterminated'


def test_a_word_that_merely_contains_a_keyword_is_not_coloured():
    assert MAGENTA not in code("information = 1")


def test_an_empty_diff_stays_empty():
    assert diff("") == ""
