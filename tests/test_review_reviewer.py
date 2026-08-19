"""The find pass, de-duplication, and the refutation pass."""

import json

import pytest

from bkht.coder.review.diff import parse_diff
from bkht.coder.review.reviewer import (
    CONFIRMED,
    PLAUSIBLE,
    Finding,
    ReviewResult,
    Reviewer,
    coerce_finding,
    file_context,
)

from fakes import FakeProvider

DIFF = """\
diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,3 +1,4 @@ def average(numbers):
 def average(numbers):
     total = sum(numbers)
-    return total / len(numbers)
+    return total / len(numbers)
+
"""


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "calc.py").write_text(
        "def average(numbers):\n    total = sum(numbers)\n    return total / len(numbers)\n"
    )
    return tmp_path


def finding_json(**overrides):
    body = {
        "file": "calc.py",
        "line": 3,
        "severity": "high",
        "category": "correctness",
        "summary": "division by zero on an empty list",
        "scenario": "average([]) raises ZeroDivisionError",
        "suggestion": "return 0.0 when numbers is empty",
    }
    body.update(overrides)
    return json.dumps([body])


def verdict(refuted=False, certain=True, reason="checked the code"):
    return json.dumps({"refuted": refuted, "certain": certain, "reason": reason})


def review(repo, script, **kwargs):
    kwargs.setdefault("dimensions", ("correctness",))
    reviewer = Reviewer(FakeProvider(script), repo, **kwargs)
    return reviewer.review(parse_diff(DIFF))


# --- coercion ---------------------------------------------------------------


def test_a_well_formed_finding_is_accepted():
    finding = coerce_finding(
        {"file": "calc.py", "line": 3, "severity": "high", "summary": "boom", "scenario": "x"},
        "correctness",
        {"calc.py"},
    )
    assert finding.file == "calc.py" and finding.severity == "high"


def test_a_finding_about_an_unrelated_file_is_rejected():
    # A model that invents a path has also invented the problem.
    assert coerce_finding(
        {"file": "invented.py", "line": 3, "summary": "boom"}, "correctness", {"calc.py"}
    ) is None


def test_a_path_prefix_difference_is_tolerated():
    finding = coerce_finding(
        {"file": "./calc.py", "line": 3, "summary": "boom"}, "correctness", {"calc.py"}
    )
    assert finding.file == "calc.py"


def test_a_stringified_line_number_is_coerced():
    assert coerce_finding(
        {"file": "calc.py", "line": "3", "summary": "boom"}, "correctness", {"calc.py"}
    ).line == 3


def test_a_finding_with_no_line_is_rejected():
    assert coerce_finding({"file": "calc.py", "summary": "boom"}, "correctness", {"calc.py"}) is None


def test_a_finding_with_no_summary_is_rejected():
    assert coerce_finding({"file": "calc.py", "line": 2}, "correctness", {"calc.py"}) is None


def test_an_unknown_severity_falls_back_to_medium():
    assert coerce_finding(
        {"file": "calc.py", "line": 2, "severity": "catastrophic", "summary": "x"},
        "correctness",
        {"calc.py"},
    ).severity == "medium"


def test_findings_rank_high_severity_first():
    low = Finding("a.py", 1, "low", "correctness", "l", "")
    high = Finding("b.py", 9, "high", "correctness", "h", "")
    assert sorted([low, high], key=lambda f: f.rank)[0] is high


# --- verify context ---------------------------------------------------------


def test_context_is_numbered_around_the_line(repo):
    text = file_context(repo, "calc.py", 2, span=1)
    assert text.splitlines()[0].strip().startswith("1 def average")
    assert len(text.splitlines()) == 3


def test_context_of_a_line_past_the_end_says_so(repo):
    assert "does not exist" in file_context(repo, "calc.py", 900)


def test_context_of_a_missing_file_does_not_raise(repo):
    assert "could not be read" in file_context(repo, "gone.py", 1)


# --- the passes -------------------------------------------------------------


def test_a_confirmed_finding_is_reported(repo):
    result = review(repo, [finding_json(), verdict(refuted=False, certain=True)])
    assert len(result.findings) == 1
    assert result.findings[0].verdict == CONFIRMED
    assert result.candidates == 1 and result.refuted == 0


def test_an_uncertain_verifier_downgrades_to_plausible(repo):
    result = review(repo, [finding_json(), verdict(refuted=False, certain=False)])
    assert result.findings[0].verdict == PLAUSIBLE


def test_a_refuted_finding_is_dropped_and_counted(repo):
    result = review(repo, [finding_json(), verdict(refuted=True, reason="guarded upstream")])
    assert result.findings == []
    assert result.refuted == 1
    assert result.candidates == 1


def test_an_empty_array_reports_nothing_and_drops_nothing(repo):
    result = review(repo, ["[]"])
    assert result.findings == [] and result.candidates == 0 and result.malformed == 0


def test_a_verifier_with_no_verdict_keeps_the_finding(repo):
    # Silently discarding findings because the checker failed is worse than
    # reporting a false one.
    result = review(repo, [finding_json(), "I am not sure about this one."])
    assert len(result.findings) == 1
    assert result.findings[0].verdict == PLAUSIBLE


def test_the_same_finding_from_two_dimensions_is_reported_once(repo):
    reviewer = Reviewer(
        FakeProvider([finding_json(), finding_json(category="security"), verdict()]),
        repo,
        dimensions=("correctness", "security"),
    )
    result = reviewer.review(parse_diff(DIFF))
    assert len(result.findings) == 1
    assert result.duplicates == 1


def test_malformed_findings_are_retried_once_then_counted(repo):
    result = review(repo, ["Sorry, I cannot do that.", finding_json(), verdict()])
    assert result.malformed >= 1
    assert len(result.findings) == 1


def test_verification_can_be_skipped(repo):
    result = review(repo, [finding_json()], verify=False)
    assert len(result.findings) == 1
    assert result.findings[0].verdict == PLAUSIBLE
    assert result.refuted == 0


def test_a_provider_failure_is_recorded_not_raised(repo):
    from bkht.coder.provider import ProviderError

    result = review(repo, [ProviderError("ollama is down")])
    assert result.findings == []
    assert "ollama is down" in result.errors[0]


def test_the_unit_count_is_reported(repo):
    result = review(repo, ["[]"])
    assert result.units == 1
    assert result.files == ["calc.py"]


def test_the_review_agent_cannot_write(repo):
    reviewer = Reviewer(FakeProvider(["[]"]), repo, dimensions=("correctness",))
    agent = reviewer._agent("sys", 3)
    assert "write_file" not in agent.registry
    assert "bash" not in agent.registry


def test_each_dimension_gets_its_own_prompt(repo):
    provider = FakeProvider(["[]", "[]"])
    Reviewer(provider, repo, dimensions=("correctness", "security")).review(parse_diff(DIFF))
    first, second = provider.calls[0][0]["content"], provider.calls[1][0]["content"]
    assert "correctness" in first and "security" not in first.split("# Workspace")[0]
    assert "security" in second


def test_the_verifier_is_asked_a_positive_question(repo):
    """The verdict is asked for as "is this real?", never as "refute this".

    Measured on the corpus, the negative form made the model answer "refuted"
    for all four correctly-found bugs while its own stated reason described the
    defect happening. The negation flipped the field without touching the
    reasoning, so the question is now framed positively.
    """
    provider = FakeProvider([finding_json(), verdict()])
    Reviewer(provider, repo, dimensions=("correctness",)).review(parse_diff(DIFF))
    prompt = provider.calls[-1][0]["content"]
    assert "REFUTE" not in prompt
    assert '"real"' in prompt


def test_a_real_verdict_keeps_the_finding(repo):
    result = review(repo, [finding_json(), json.dumps(
        {"real": True, "certain": True, "reason": "read it"})])
    assert len(result.findings) == 1 and result.refuted == 0


def test_a_certain_contradiction_removes_the_finding(repo):
    result = review(repo, [finding_json(), json.dumps(
        {"real": False, "certain": True, "reason": "guarded at util.py:4"})])
    assert result.findings == [] and result.refuted == 1


def test_an_unsure_contradiction_keeps_the_finding_as_plausible(repo):
    # Dropping a real finding is the more expensive error: a missed bug reads
    # as a clean bill of health.
    result = review(repo, [finding_json(), json.dumps(
        {"real": False, "certain": False, "reason": "not sure"})])
    assert len(result.findings) == 1
    assert result.findings[0].verdict == "plausible"


def test_the_old_refuted_field_is_still_understood(repo):
    # A small model will sometimes answer the question it remembers rather
    # than the one it was asked.
    result = review(repo, [finding_json(), verdict(refuted=True, certain=True)])
    assert result.findings == [] and result.refuted == 1
