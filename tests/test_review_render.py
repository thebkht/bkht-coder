"""Terminal, JSON, and Markdown output."""

import json

from bkht.coder.review.render import as_json, markdown, terminal
from bkht.coder.review.reviewer import CONFIRMED, PLAUSIBLE, Finding, ReviewResult


def result(findings=(), **kwargs):
    base = dict(files=["a.py", "b.py"], units=2, candidates=len(findings))
    base.update(kwargs)
    return ReviewResult(findings=list(findings), **base)


HIGH = Finding("a.py", 12, "high", "correctness", "divides by zero", "average([]) raises", "guard it", CONFIRMED)
LOW = Finding("a.py", 40, "low", "tests", "no test for the empty case", "", "", PLAUSIBLE)
OTHER = Finding("b.py", 7, "medium", "security", "path is not validated", "../ escapes the root", "", CONFIRMED)


# --- terminal ---------------------------------------------------------------


def test_terminal_states_what_was_reviewed(): 
    assert "Reviewed 2 files in 2 units" in terminal(result(), colour=False)


def test_empty_result_reports_what_was_dropped_not_a_bare_all_clear():
    # "refuted 6 candidates" and "looked at nothing" must not read the same.
    text = terminal(result(candidates=6, refuted=6), colour=False)
    assert "No findings." in text
    assert "6 refuted in verification" in text


def test_a_genuinely_quiet_review_says_so_explicitly():
    text = terminal(result(), colour=False)
    assert "Nothing was flagged in any pass." in text


def test_findings_are_grouped_by_file():
    text = terminal(result([HIGH, LOW, OTHER]), colour=False)
    assert text.index("a.py:12") < text.index("b.py:7")
    assert text.count("\na.py") == 1


def test_severity_orders_within_a_file():
    text = terminal(result([LOW, HIGH]), colour=False)
    assert text.index("a.py:12") < text.index("a.py:40")


def test_each_finding_shows_location_category_and_scenario():
    text = terminal(result([HIGH]), colour=False)
    assert "a.py:12" in text and "correctness" in text
    assert "divides by zero" in text
    assert "average([]) raises" in text


def test_plausible_findings_are_marked_and_confirmed_ones_are_not():
    text = terminal(result([HIGH, LOW]), colour=False)
    lines = [l for l in text.splitlines() if ":40" in l or ":12" in l]
    assert any("(plausible)" in l for l in lines)
    assert not any("(plausible)" in l and ":12" in l for l in lines)


def test_colour_is_optional():
    assert "\033[" not in terminal(result([HIGH]), colour=False)
    assert "\033[" in terminal(result([HIGH]), colour=True)


def test_errors_are_surfaced():
    assert "ollama is down" in terminal(result(errors=["ollama is down"]), colour=False)


# --- json -------------------------------------------------------------------


def test_json_round_trips_every_field():
    payload = json.loads(as_json(result([HIGH])))
    assert payload["findings"][0] == {
        "file": "a.py",
        "line": 12,
        "severity": "high",
        "category": "correctness",
        "summary": "divides by zero",
        "scenario": "average([]) raises",
        "suggestion": "guard it",
        "verdict": "confirmed",
    }


def test_json_carries_the_counts_ci_needs():
    payload = json.loads(as_json(result(candidates=9, refuted=8, malformed=1)))
    assert payload["candidates"] == 9
    assert payload["refuted"] == 8
    assert payload["malformed"] == 1
    assert payload["units"] == 2


def test_json_of_an_empty_review_is_still_valid():
    assert json.loads(as_json(result()))["findings"] == []


# --- markdown ---------------------------------------------------------------


def test_markdown_has_a_heading_per_file_and_finding():
    text = markdown(result([HIGH, OTHER]))
    assert "## `a.py`" in text and "## `b.py`" in text
    assert "### HIGH — `a.py:12` (correctness)" in text


def test_markdown_marks_plausible_findings():
    assert "_(plausible)_" in markdown(result([LOW]))


def test_markdown_of_an_empty_review_says_what_was_dropped():
    text = markdown(result(candidates=3, refuted=3))
    assert "No findings." in text and "3 refuted" in text


def test_markdown_ends_with_a_newline():
    assert markdown(result([HIGH])).endswith("\n")
