"""Accuracy of the review pipeline against the planted-bug corpus.

The corpus checks that need no model run always. The accuracy measurement is
live: it prints recall and precision so a prompt change can be judged, and
fails if either drops below the floor recorded here.
"""

from __future__ import annotations

import subprocess

import pytest

from bkht.coder.provider import OllamaProvider, for_review
from bkht.coder.review.diff import collect_diff
from bkht.coder.review.reviewer import Reviewer

from corpus import BUGGY, CASES, CLEAN, Case

# Floors, not targets. They record what this prompt set achieves on a 14b
# model; raise them when a change earns it, and treat a drop as a failure.
MIN_RECALL = 0.5
MIN_PRECISION = 0.6
LINE_TOLERANCE = 3


def build_repo(root, case: Case):
    """A repo whose single uncommitted change is this case's diff."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    (root / case.filename).write_text(case.before)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@localhost", "-c", "user.name=t", "commit", "-qm", "before"],
        cwd=root,
        check=True,
    )
    (root / case.filename).write_text(case.after)
    return root


# --- corpus integrity (no model) --------------------------------------------


def test_the_corpus_has_both_bugs_and_clean_cases():
    # Precision is unmeasurable without clean cases, recall without buggy ones.
    assert len(BUGGY) >= 3 and len(CLEAN) >= 3


def test_every_case_names_a_distinct_scenario():
    assert len({c.name for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_every_case_actually_changes_something(case, tmp_path):
    build_repo(tmp_path, case)
    files = collect_diff(tmp_path)
    assert files, f"{case.name} produces no diff"
    assert files[0].path == case.filename


@pytest.mark.parametrize("case", BUGGY, ids=lambda c: c.name)
def test_every_planted_bug_line_exists_in_the_new_file(case):
    lines = case.after.splitlines()
    assert 1 <= case.bug_line <= len(lines), f"{case.name}: bug_line is out of range"


@pytest.mark.parametrize("case", CLEAN, ids=lambda c: c.name)
def test_clean_cases_declare_no_category(case):
    assert case.category == ""


# --- accuracy (live) --------------------------------------------------------


@pytest.fixture(scope="module")
def provider(pytestconfig):
    from bkht.coder.provider import DEFAULT_MODEL

    provider = OllamaProvider(model=pytestconfig.getoption("--model") or DEFAULT_MODEL)
    if not provider.available():
        pytest.skip("Ollama is not reachable")
    # Deterministic sampling, or the measurement is noise.
    return for_review(provider)


def review_case(provider, root, case: Case):
    build_repo(root, case)
    reviewer = Reviewer(provider, root, dimensions=(case.category,) if case.category else ("correctness", "security"))
    return reviewer.review(collect_diff(root))


@pytest.mark.live
def test_recall_on_planted_bugs(provider, tmp_path_factory, record_property):
    found = []
    for case in BUGGY:
        result = review_case(provider, tmp_path_factory.mktemp(case.name), case)
        hit = any(
            f.file == case.filename and abs(f.line - case.bug_line) <= LINE_TOLERANCE
            for f in result.findings
        )
        found.append(hit)
        print(f"  {'HIT ' if hit else 'MISS'} {case.name}: {case.note}")
        for finding in result.findings:
            print(f"        reported {finding.file}:{finding.line} — {finding.summary}")

    recall = sum(found) / len(found)
    record_property("recall", recall)
    print(f"\nrecall = {sum(found)}/{len(found)} = {recall:.2f}")
    assert recall >= MIN_RECALL, f"recall {recall:.2f} is below the floor {MIN_RECALL}"


@pytest.mark.live
def test_precision_on_clean_code(provider, tmp_path_factory, record_property):
    quiet = []
    for case in CLEAN:
        result = review_case(provider, tmp_path_factory.mktemp(case.name), case)
        clean = not result.findings
        quiet.append(clean)
        print(f"  {'QUIET' if clean else 'NOISY'} {case.name}: {case.note}")
        for finding in result.findings:
            print(f"        false positive {finding.file}:{finding.line} — {finding.summary}")
        if result.refuted:
            print(f"        ({result.refuted} refuted in verification)")

    precision = sum(quiet) / len(quiet)
    record_property("precision", precision)
    print(f"\nprecision = {sum(quiet)}/{len(quiet)} = {precision:.2f}")
    assert precision >= MIN_PRECISION, (
        f"precision {precision:.2f} is below the floor {MIN_PRECISION}"
    )


@pytest.mark.live
def test_verification_removes_more_than_it_keeps_on_clean_code(provider, tmp_path_factory):
    # The point of the verify pass: on code with nothing wrong, candidates
    # should mostly be refuted rather than reported.
    case = CLEAN[0]
    result = review_case(provider, tmp_path_factory.mktemp("verify"), case)
    print(f"candidates={result.candidates} refuted={result.refuted} kept={len(result.findings)}")
    assert result.candidates == 0 or result.refuted >= len(result.findings)
