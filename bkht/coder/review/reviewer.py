"""The find pass and the verify pass.

The verify pass is not polish. A 14b model generates a high volume of
confident, wrong findings, and an unfiltered report is worse than no report
because it teaches the reader to ignore it. Every candidate gets a second,
independent call whose job is to refute it, and only survivors are reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .. import prompts
from ..agent import Agent, NullListener
from ..parsing import extract_json_objects
from ..session import Session
from ..tools import build_registry
from .diff import FileDiff, ReviewUnit, chunk

DIMENSIONS = ("correctness", "error-handling", "security", "tests")
SEVERITIES = ("high", "medium", "low")
SEVERITY_ORDER = {name: i for i, name in enumerate(SEVERITIES)}

CONFIRMED = "confirmed"
PLAUSIBLE = "plausible"

MAX_FIND_ITERATIONS = 12
MAX_VERIFY_ITERATIONS = 8
CONTEXT_LINES = 12


@dataclass
class Finding:
    """One reported problem."""

    file: str
    line: int
    severity: str
    category: str
    summary: str
    scenario: str
    suggestion: str = ""
    verdict: str = PLAUSIBLE

    @property
    def rank(self) -> tuple:
        """Sort key: high severity first, then by file and line."""
        return (SEVERITY_ORDER.get(self.severity, 99), self.file, self.line)

    @property
    def key(self) -> tuple:
        """Identity for de-duplication across dimensions."""
        return (self.file, self.line, self.summary.strip().lower()[:60])

    def as_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "severity": self.severity,
            "category": self.category,
            "summary": self.summary,
            "scenario": self.scenario,
            "suggestion": self.suggestion,
            "verdict": self.verdict,
        }


@dataclass
class ReviewResult:
    """Everything the report needs, including what was thrown away.

    The counts matter as much as the findings: a run that reports nothing after
    dropping eleven candidates says something quite different from one that
    found nothing to begin with.
    """

    findings: list[Finding] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    units: int = 0
    dimensions: tuple = DIMENSIONS
    candidates: int = 0
    refuted: int = 0
    malformed: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)


class ReviewListener:
    """Progress reporting for a review, which can take minutes."""

    def on_pass(self, unit: int, total: int, dimension: str) -> None:
        pass

    def on_candidates(self, dimension: str, count: int) -> None:
        pass

    def on_verify(self, index: int, total: int, finding: "Finding") -> None:
        pass

    def on_verdict(self, finding: "Finding", refuted: bool, reason: str) -> None:
        pass


def coerce_finding(obj: dict, dimension: str, valid_files: set[str]) -> Finding | None:
    """Build a :class:`Finding` from model JSON, or None if it is unusable.

    Rejects findings citing a file that was not part of the change: a model
    that invents a path has also invented the problem.
    """
    if not isinstance(obj, dict):
        return None

    file = obj.get("file")
    summary = obj.get("summary") or obj.get("title")
    scenario = obj.get("scenario") or obj.get("failure_scenario") or ""
    if not isinstance(file, str) or not isinstance(summary, str) or not summary.strip():
        return None

    file = file.lstrip("./")
    if valid_files and file not in valid_files:
        matches = [name for name in valid_files if name.endswith(file) or file.endswith(name)]
        if len(matches) != 1:
            return None
        file = matches[0]

    line = obj.get("line")
    if isinstance(line, str) and line.strip().isdigit():
        line = int(line)
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        return None

    severity = str(obj.get("severity", "medium")).strip().lower()
    if severity not in SEVERITIES:
        severity = "medium"

    return Finding(
        file=file,
        line=line,
        severity=severity,
        category=str(obj.get("category") or dimension),
        summary=summary.strip(),
        scenario=str(scenario).strip(),
        suggestion=str(obj.get("suggestion") or "").strip(),
    )


def file_context(root: Path, file: str, line: int, span: int = CONTEXT_LINES) -> str:
    """The lines around a finding, numbered, for the verify pass."""
    target = Path(root) / file
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(the file could not be read)"

    start = max(1, line - span)
    end = min(len(lines), line + span)
    if start > len(lines):
        return f"(the file has only {len(lines)} lines, so line {line} does not exist)"

    return "\n".join(f"{n:>5} {lines[n - 1]}" for n in range(start, end + 1))


class Reviewer:
    """Runs the find and verify passes over a set of review units."""

    def __init__(
        self,
        provider,
        root: Path,
        dimensions: tuple = DIMENSIONS,
        listener: ReviewListener | None = None,
        verify: bool = True,
    ) -> None:
        self.provider = provider
        self.root = Path(root)
        self.dimensions = dimensions
        self.listener = listener or ReviewListener()
        self.verify = verify

    # --- passes -------------------------------------------------------------

    def _agent(self, system: str, max_iterations: int) -> Agent:
        """A read-only agent. A review must never change the code it reviews."""
        registry, _ = build_registry(self.root, read_only=True)
        return Agent(
            provider=self.provider,
            registry=registry,
            session=Session(system=system),
            listener=NullListener(),
            max_iterations=max_iterations,
        )

    def find(self, unit: ReviewUnit, dimension: str, result: ReviewResult) -> list[Finding]:
        """One dimension over one unit."""
        registry, _ = build_registry(self.root, read_only=True)
        system = prompts.review_system(dimension, registry, str(self.root))
        agent = self._agent(system, MAX_FIND_ITERATIONS)

        outcome = agent.run(
            prompts.REVIEW_TASK.format(dimension=dimension, diff=unit.render())
        )
        if outcome.stopped == "provider-error":
            result.errors.extend(outcome.errors)
            return []

        findings = self._parse_findings(outcome.raw or outcome.answer, dimension, unit, result)

        # One retry on a reply that parsed to nothing usable, then give up and
        # count it, rather than letting a bad format silently look like "clean".
        if not findings and not self._looks_empty(outcome.raw or outcome.answer):
            result.malformed += 1
            outcome = agent.run(
                "That reply was not a valid JSON array of findings. Reply with "
                "only the JSON array, or [] if you found nothing."
            )
            findings = self._parse_findings(
                outcome.raw or outcome.answer, dimension, unit, result
            )

        self.listener.on_candidates(dimension, len(findings))
        return findings

    def _looks_empty(self, text: str) -> bool:
        """Whether the model plausibly said 'nothing found'."""
        stripped = (text or "").strip()
        return "[]" in stripped or not stripped

    def _parse_findings(
        self, text: str, dimension: str, unit: ReviewUnit, result: ReviewResult
    ) -> list[Finding]:
        valid = set(unit.paths)
        findings = []
        for obj in extract_json_objects(text or ""):
            # Verdict objects and stray JSON share the message; only things
            # shaped like a finding count.
            if "summary" not in obj and "title" not in obj:
                continue
            finding = coerce_finding(obj, dimension, valid)
            if finding is None:
                result.malformed += 1
            else:
                findings.append(finding)
        return findings

    def refute(self, finding: Finding, result: ReviewResult) -> bool:
        """Second, independent call whose job is to knock the finding down.

        Returns True if the finding survives. An unreachable or unusable
        verifier keeps the finding as *plausible* rather than dropping it --
        silently discarding findings because the checker failed is the one
        outcome worse than reporting a false one.
        """
        registry, _ = build_registry(self.root, read_only=True)
        agent = self._agent(
            prompts.verify_system(registry, str(self.root)), MAX_VERIFY_ITERATIONS
        )

        outcome = agent.run(
            prompts.VERIFY_TASK.format(
                file=finding.file,
                line=finding.line,
                category=finding.category,
                summary=finding.summary,
                scenario=finding.scenario or "(none given)",
                context=file_context(self.root, finding.file, finding.line),
            )
        )
        if outcome.stopped == "provider-error":
            result.errors.extend(outcome.errors)
            return True

        verdicts = [
            obj for obj in extract_json_objects(outcome.raw or outcome.answer or "")
            if "real" in obj or "refuted" in obj
        ]
        if not verdicts:
            self.listener.on_verdict(finding, False, "verifier gave no verdict")
            return True

        verdict = verdicts[-1]
        certain = bool(verdict.get("certain"))
        # The verdict is asked for as a positive ("is this real?"). The earlier
        # negative form asked the model whether to *refute*, and measured on the
        # corpus it answered "refuted" for all four correctly-found bugs while
        # its own stated reason described the defect happening -- the negation
        # flipped the field without touching the reasoning. ``refuted`` is still
        # accepted, because a small model will sometimes answer the question it
        # remembers rather than the one it was asked.
        real = bool(verdict.get("real")) if "real" in verdict else not bool(verdict.get("refuted"))
        reason = str(verdict.get("reason") or "").strip()
        self.listener.on_verdict(finding, not real, reason)

        # Only a confident contradiction removes a finding. Dropping a real one
        # is the more expensive error: a missed bug reads as a clean bill of
        # health, where a doubtful one merely reads as doubtful.
        if not real and certain:
            result.refuted += 1
            return False

        finding.verdict = CONFIRMED if real and certain else PLAUSIBLE
        return True

    # --- driver -------------------------------------------------------------

    def review(self, files: list[FileDiff], budget: int | None = None) -> ReviewResult:
        """Find, de-duplicate, verify, and rank."""
        units = chunk(files) if budget is None else chunk(files, budget)
        result = ReviewResult(
            files=[f.path for f in files], units=len(units), dimensions=self.dimensions
        )

        candidates: list[Finding] = []
        seen: set[tuple] = set()

        for index, unit in enumerate(units, start=1):
            for dimension in self.dimensions:
                self.listener.on_pass(index, len(units), dimension)
                for finding in self.find(unit, dimension, result):
                    result.candidates += 1
                    if finding.key in seen:
                        result.duplicates += 1
                        continue
                    seen.add(finding.key)
                    candidates.append(finding)

        if not self.verify:
            for finding in candidates:
                finding.verdict = PLAUSIBLE
            result.findings = sorted(candidates, key=lambda f: f.rank)
            return result

        survivors = []
        for index, finding in enumerate(candidates, start=1):
            self.listener.on_verify(index, len(candidates), finding)
            if self.refute(finding, result):
                survivors.append(finding)

        result.findings = sorted(survivors, key=lambda f: f.rank)
        return result
