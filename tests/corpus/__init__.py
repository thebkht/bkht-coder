"""A small corpus of diffs with known planted bugs and known-good code.

This exists because prompt tuning against a 14b model is most of the work in
code review, and without a fixed corpus there is no way to tell whether a
prompt change helped or hurt. Recall (are the planted bugs found?) and
precision (does the clean code stay quiet?) are the metric; a regression in
either is a failure, not noise.

Every case is one file changed by one commit. ``bug_line`` is the line in the
*new* file where the defect lives, or None for a case that must produce nothing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    filename: str
    before: str
    after: str
    bug_line: int | None
    category: str
    note: str


CASES: list[Case] = [
    Case(
        name="off-by-one-slice",
        filename="pager.py",
        before='''\
def page(items, size, number):
    start = number * size
    return items[start : start + size]
''',
        after='''\
def page(items, size, number):
    start = number * size
    return items[start : start + size + 1]
''',
        bug_line=3,
        category="correctness",
        note="the slice returns one item too many, so pages overlap",
    ),
    Case(
        name="empty-collection-division",
        filename="stats.py",
        before='''\
def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)
''',
        after='''\
def mean(values):
    return sum(values) / len(values)
''',
        bug_line=2,
        category="correctness",
        note="the empty-list guard was removed, so mean([]) now raises",
    ),
    Case(
        name="swallowed-exception",
        filename="loader.py",
        before='''\
import json


def load(path):
    with open(path) as handle:
        return json.load(handle)
''',
        after='''\
import json


def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception:
        return {}
''',
        bug_line=8,
        category="error-handling",
        note="a corrupt or missing file silently becomes an empty config",
    ),
    Case(
        name="path-traversal",
        filename="files.py",
        before='''\
import os

ROOT = "/srv/data"


def read(name):
    if "/" in name or ".." in name:
        raise ValueError(name)
    return open(os.path.join(ROOT, name)).read()
''',
        after='''\
import os

ROOT = "/srv/data"


def read(name):
    return open(os.path.join(ROOT, name)).read()
''',
        bug_line=7,
        category="security",
        note="the traversal guard was removed, so ../../etc/passwd is readable",
    ),
    Case(
        name="clean-rename",
        filename="greet.py",
        before='''\
def greet(n):
    return f"Hello, {n}!"
''',
        after='''\
def greet(name):
    return f"Hello, {name}!"
''',
        bug_line=None,
        category="",
        note="a pure rename; nothing about behaviour changed",
    ),
    Case(
        name="clean-guard-added",
        filename="divide.py",
        before='''\
def divide(a, b):
    return a / b
''',
        after='''\
def divide(a, b):
    if b == 0:
        raise ValueError("b must not be zero")
    return a / b
''',
        bug_line=None,
        category="",
        note="the diff fixes a problem; reporting it would be reporting the fix",
    ),
    Case(
        name="clean-extracted-helper",
        filename="fmt.py",
        before='''\
def show(items):
    return ", ".join(str(i) for i in items)
''',
        after='''\
def _render(item):
    return str(item)


def show(items):
    return ", ".join(_render(i) for i in items)
''',
        bug_line=None,
        category="",
        note="behaviour-preserving extraction",
    ),
]

BUGGY = [c for c in CASES if c.bug_line is not None]
CLEAN = [c for c in CASES if c.bug_line is None]
