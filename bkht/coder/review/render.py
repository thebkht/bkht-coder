"""Review output: terminal, JSON, and Markdown.

An empty result never prints a bare "no issues". It says what was reviewed and
how many candidates were dropped in verification, because "I looked at 4 files
and refuted 6 candidates" and "I looked at nothing" are very different results
that a bare all-clear renders identically.
"""

from __future__ import annotations

import json

from ..terminal import BLUE, BOLD, DIM, GREEN, RED, RESET, YELLOW
from .reviewer import CONFIRMED, ReviewResult

SEVERITY_COLOUR = {"high": RED, "medium": YELLOW, "low": BLUE}


def group_by_file(findings) -> dict:
    """Findings keyed by file, each list ordered high severity first.

    Sorted here rather than trusting the caller: ordering is part of what the
    report promises, and it is read by people scanning for the worst thing.
    """
    grouped: dict[str, list] = {}
    for finding in sorted(findings, key=lambda f: f.rank):
        grouped.setdefault(finding.file, []).append(finding)
    return grouped


def scope_line(result: ReviewResult) -> str:
    """One line saying what was actually examined."""
    files = len(result.files)
    return (
        f"Reviewed {files} file{'s' if files != 1 else ''} "
        f"in {result.units} unit{'s' if result.units != 1 else ''}, "
        f"across {len(result.dimensions)} dimension"
        f"{'s' if len(result.dimensions) != 1 else ''}."
    )


def dropped_line(result: ReviewResult) -> str:
    """What was thrown away, so an empty report is not mistaken for no work."""
    parts = []
    if result.refuted:
        parts.append(f"{result.refuted} refuted in verification")
    if result.duplicates:
        parts.append(f"{result.duplicates} duplicate")
    if result.malformed:
        parts.append(f"{result.malformed} unparseable")
    return f"Dropped {', '.join(parts)}." if parts else ""


def terminal(result: ReviewResult, colour: bool = True) -> str:
    """The default report: grouped by file, high severity first."""

    def paint(text, code):
        return f"{code}{text}{RESET}" if colour else text

    lines = [scope_line(result)]

    if not result.findings:
        lines.append("")
        lines.append(paint("No findings.", GREEN))
        dropped = dropped_line(result)
        if dropped:
            lines.append(dropped)
        elif result.candidates == 0:
            lines.append("Nothing was flagged in any pass.")
    else:
        count = len(result.findings)
        lines.append(f"{count} finding{'s' if count != 1 else ''}.")

        for file, findings in group_by_file(result.findings).items():
            lines.append("")
            lines.append(paint(file, BOLD))
            for finding in findings:
                colour_code = SEVERITY_COLOUR.get(finding.severity, "")
                tag = paint(f"{finding.severity:>6}", colour_code)
                mark = "" if finding.verdict == CONFIRMED else paint(" (plausible)", DIM)
                lines.append(f"  {tag}  {file}:{finding.line}  {finding.category}{mark}")
                lines.append(f"          {finding.summary}")
                if finding.scenario:
                    lines.append(paint(f"          → {finding.scenario}", DIM))
                if finding.suggestion:
                    lines.append(paint(f"          ↳ {finding.suggestion}", DIM))

        dropped = dropped_line(result)
        if dropped:
            lines.append("")
            lines.append(paint(dropped, DIM))

    if result.errors:
        lines.append("")
        lines.append(paint(f"Errors during review: {result.errors[-1]}", YELLOW))

    return "\n".join(lines)


def as_json(result: ReviewResult) -> str:
    """Machine-readable form, for CI."""
    return json.dumps(
        {
            "files": result.files,
            "units": result.units,
            "dimensions": list(result.dimensions),
            "candidates": result.candidates,
            "refuted": result.refuted,
            "duplicates": result.duplicates,
            "malformed": result.malformed,
            "errors": result.errors,
            "findings": [f.as_dict() for f in result.findings],
        },
        indent=2,
    )


def markdown(result: ReviewResult) -> str:
    """A report to save or paste into a pull request."""
    lines = ["# Code review", "", scope_line(result), ""]

    if not result.findings:
        lines.append("No findings.")
        dropped = dropped_line(result)
        if dropped:
            lines += ["", dropped]
    else:
        count = len(result.findings)
        lines.append(f"**{count} finding{'s' if count != 1 else ''}.**")

        for file, findings in group_by_file(result.findings).items():
            lines += ["", f"## `{file}`"]
            for finding in findings:
                mark = "" if finding.verdict == CONFIRMED else " _(plausible)_"
                lines += [
                    "",
                    f"### {finding.severity.upper()} — `{file}:{finding.line}` "
                    f"({finding.category}){mark}",
                    "",
                    finding.summary,
                ]
                if finding.scenario:
                    lines += ["", f"**How it fails.** {finding.scenario}"]
                if finding.suggestion:
                    lines += ["", f"**Suggested fix.** {finding.suggestion}"]

        dropped = dropped_line(result)
        if dropped:
            lines += ["", "---", "", dropped]

    if result.errors:
        lines += ["", f"> Errors during review: {result.errors[-1]}"]

    return "\n".join(lines) + "\n"
