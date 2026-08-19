"""Colouring a unified diff before someone approves it.

Approval is a reading task under time pressure: the question is always "is this
the change I meant", and a wall of monochrome text is where a stray deletion
hides. Two things are coloured, and they answer different questions -- the
marker says what is happening to the line, the syntax says what the line is.

Not a parser. It is a token scanner with one regex covering the shapes common
to the languages this agent meets, because a diff hunk is a fragment: it has no
whole file to parse, half its lines are syntactically incomplete, and a real
parser would fail on exactly the input this always gets.
"""

from __future__ import annotations

import re

from .terminal import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RESET, YELLOW

KEYWORDS = {
    # Deliberately cross-language: previews are not always Python, and a word
    # highlighted in the wrong language is a smaller cost than a plain diff.
    "and", "as", "assert", "async", "await", "break", "case", "catch", "class",
    "const", "continue", "def", "del", "elif", "else", "except", "export",
    "extends", "finally", "fn", "for", "from", "func", "function", "global",
    "if", "impl", "import", "in", "is", "lambda", "let", "match", "mut", "new",
    "nonlocal", "not", "or", "pass", "pub", "raise", "return", "self", "struct",
    "switch", "try", "type", "var", "while", "with", "yield",
    "None", "True", "False", "null", "nil", "true", "false", "undefined",
}

TOKENS = re.compile(
    r"""
    (?P<comment>\#[^\n]*|//[^\n]*)
  | (?P<string>"(?:\\.|[^"\\])*"?|'(?:\\.|[^'\\])*'?)
  | (?P<number>\b\d[\d_]*(?:\.\d+)?\b)
  | (?P<word>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)

COLOURS = {"comment": DIM, "string": YELLOW, "number": CYAN, "keyword": MAGENTA}


def code(line: str, base: str = "") -> str:
    """Syntax-colour one line, returning to ``base`` after every token."""
    out, last = [], 0
    for match in TOKENS.finditer(line):
        kind = match.lastgroup
        if kind == "word":
            if match.group() not in KEYWORDS:
                continue
            kind = "keyword"
        out.append(line[last : match.start()])
        out.append(f"{COLOURS[kind]}{match.group()}{RESET}{base}")
        last = match.end()
    out.append(line[last:])
    return "".join(out)


def line(text: str) -> str:
    """Colour one diff line: its marker, then its contents."""
    if text.startswith(("+++", "---")):
        return f"{BOLD}{text}{RESET}"
    if text.startswith("@@"):
        return f"{CYAN}{text}{RESET}"
    if text.startswith("["):
        return f"{DIM}{text}{RESET}"  # the "[N more diff lines]" marker

    if text[:1] in ("+", "-"):
        base = GREEN if text[0] == "+" else RED
        return f"{base}{text[0]}{code(text[1:], base)}{RESET}"
    return code(text)


def diff(text: str, colour: bool = True) -> str:
    """A unified diff, coloured. ``colour=False`` returns it untouched."""
    if not colour or not text:
        return text
    return "\n".join(line(part) for part in text.split("\n"))
