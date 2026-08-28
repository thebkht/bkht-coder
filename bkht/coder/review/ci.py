"""Review output for CI, where nobody is watching a terminal.

Two things are wrong with the interactive :class:`~.cli.Progress` output once a
review runs in a pipeline. The escape codes are noise in a log file, and the
findings never reach the place a reviewer actually looks -- a run had to emit
``--json`` and have something downstream turn it into review comments.

So this module does both halves. Progress becomes flat, grepp-able lines
grouped into collapsible sections, and findings become each platform's native
annotation: workflow commands on GitHub Actions, which land inline on the pull
request diff, and a Code Quality report on GitLab, which lands inline on the
merge request. The finding is on the line it is about, which is the only place
it does any good.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

from . import render
from .reviewer import ReviewListener, ReviewResult

GITHUB = "github"
GITLAB = "gitlab"
GENERIC = "generic"
OFF = "off"
AUTO = "auto"

KINDS = (AUTO, GITHUB, GITLAB, GENERIC, OFF)

PREFIX = "[review]"

# GitHub renders these against the diff; the severity decides how loudly.
ANNOTATION = {"high": "error", "medium": "warning", "low": "notice"}

# GitLab's Code Quality widget knows five severities, not our three.
CODE_QUALITY_SEVERITY = {"high": "critical", "medium": "major", "low": "minor"}

NOT_SLUG = re.compile(r"[^A-Za-z0-9_]+")


def detect(env=None) -> str | None:
    """Which CI this is running under, or None for a plain shell.

    Checked most specific first: GitHub Actions and GitLab both also set the
    generic ``CI``, and picking that up first would cost the annotations.
    """
    env = os.environ if env is None else env
    if env.get("GITHUB_ACTIONS", "").lower() == "true":
        return GITHUB
    if env.get("GITLAB_CI", "").lower() == "true":
        return GITLAB
    if env.get("CI", "").lower() in ("true", "1", "yes"):
        return GENERIC
    return None


def resolve(requested: str | None, env=None) -> str | None:
    """Turn the ``--ci`` flag and the environment into one answer.

    ``None`` -- the flag was not given -- means detect. A bare ``--ci`` forces
    CI output even outside CI, falling back to the generic shape when there is
    no platform to detect.
    """
    if requested == OFF:
        return None
    if requested is None:
        return detect(env)
    if requested == AUTO:
        return detect(env) or GENERIC
    return requested


# --- progress ---------------------------------------------------------------


class CIListener(ReviewListener):
    """Flat progress for a log file: no colour, one event per line.

    Sections are opened and closed as the review moves between passes, so a
    thousand lines of verification collapse to one row on platforms that
    understand them. :meth:`finish` closes the last one -- without it the
    report that follows is swallowed into the final section.
    """

    def __init__(self, stream=None) -> None:
        self.stream = sys.stderr if stream is None else stream
        self._section: str | None = None

    # -- section handling, overridden per platform --

    def _open(self, title: str, slug: str) -> None:
        self._say(title)

    def _close(self) -> None:
        pass

    def _section_to(self, title: str, slug: str) -> None:
        if self._section == slug:
            return
        if self._section is not None:
            self._close()
        self._section = slug
        self._open(title, slug)

    def finish(self) -> None:
        """Close whatever section is open. Safe to call twice."""
        if self._section is not None:
            self._close()
            self._section = None

    # -- output --

    def _say(self, text: str) -> None:
        print(f"{PREFIX} {text}", file=self.stream, flush=True)

    def _write(self, text: str) -> None:
        print(text, file=self.stream, flush=True)

    # -- listener hooks --

    def on_pass(self, unit: int, total: int, dimension: str) -> None:
        self._section_to(f"unit {unit}/{total}: {dimension}", slug(f"unit{unit}-{dimension}"))

    def on_candidates(self, dimension: str, count: int) -> None:
        self._say(f"{count} candidate{'s' if count != 1 else ''}")

    def on_verify(self, index: int, total: int, finding) -> None:
        self._section_to("verification", "verify")
        self._say(f"verify {index}/{total}: {finding.file}:{finding.line}")

    def on_verdict(self, finding, refuted: bool, reason: str) -> None:
        if refuted:
            self._say(f"refuted - {reason or 'no reason given'}")


def slug(text: str) -> str:
    """A section name a CI log can carry: no spaces, no punctuation."""
    return NOT_SLUG.sub("-", text).strip("-").lower() or "section"


class GitHubActions(CIListener):
    """Sections as workflow ``::group::`` commands."""

    def _open(self, title: str, slug: str) -> None:
        self._write(f"::group::{title}")

    def _close(self) -> None:
        self._write("::endgroup::")


class GitLab(CIListener):
    """Sections as GitLab's ``section_start``/``section_end`` markers.

    The escape sequences are part of the protocol, not decoration: GitLab looks
    for ``\\033[0K`` around the marker and shows nothing collapsible without it.
    """

    def _open(self, title: str, slug: str) -> None:
        stamp = int(time.time())
        self._write(f"\033[0Ksection_start:{stamp}:{slug}[collapsed=true]\r\033[0K{title}")

    def _close(self) -> None:
        stamp = int(time.time())
        self._write(f"\033[0Ksection_end:{stamp}:{self._section}\r\033[0K")


LISTENERS = {GITHUB: GitHubActions, GITLAB: GitLab, GENERIC: CIListener}


def listener_for(kind: str, stream=None) -> CIListener:
    """The progress listener for a resolved CI kind."""
    return LISTENERS.get(kind, CIListener)(stream=stream)


# --- GitHub annotations -----------------------------------------------------


def escape_data(text: str) -> str:
    """Escape the message half of a workflow command.

    A raw newline ends the command, so an unescaped ``scenario`` does not
    render badly -- it silently truncates the annotation to its first line.
    """
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_property(text: str) -> str:
    """Escape a property value, where ``,`` and ``:`` are also delimiters."""
    return escape_data(text).replace(",", "%2C").replace(":", "%3A")


def message(finding) -> str:
    """The annotation body: what is wrong, how it fails, what to do."""
    parts = [finding.summary]
    if finding.scenario:
        parts.append(f"How it fails: {finding.scenario}")
    if finding.suggestion:
        parts.append(f"Suggested fix: {finding.suggestion}")
    return "\n".join(parts)


def annotate(result: ReviewResult, stream=None) -> None:
    """Emit one workflow command per finding, for the pull request diff."""
    stream = sys.stdout if stream is None else stream
    for finding in result.findings:
        command = ANNOTATION.get(finding.severity, "warning")
        title = escape_property(f"{finding.severity}: {finding.category}")
        print(
            f"::{command} file={escape_property(finding.file)},"
            f"line={finding.line},title={title}::{escape_data(message(finding))}",
            file=stream,
            flush=True,
        )


def summary(result: ReviewResult, env=None) -> str | None:
    """Append the Markdown report to the job summary, if there is one.

    Appended rather than written: ``$GITHUB_STEP_SUMMARY`` is one file shared by
    every step in the job, and truncating it would throw away another step's work.
    """
    env = os.environ if env is None else env
    path = env.get("GITHUB_STEP_SUMMARY")
    if not path:
        return None
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(render.markdown(result))
    return path


# --- GitLab Code Quality ----------------------------------------------------


def fingerprint(finding) -> str:
    """A stable identity for a finding, so GitLab can track it across runs.

    Deliberately over file, line and summary only. Anything that varies with
    run order or with what else the review found would make the same defect
    look new in every pipeline.
    """
    body = f"{finding.file}:{finding.line}:{finding.summary.strip().lower()}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def code_quality(result: ReviewResult) -> str:
    """The findings as a CodeClimate report, which GitLab renders on the MR."""
    return json.dumps(
        [
            {
                "description": message(finding),
                "check_name": finding.category,
                "fingerprint": fingerprint(finding),
                "severity": CODE_QUALITY_SEVERITY.get(finding.severity, "major"),
                "location": {
                    "path": finding.file,
                    "lines": {"begin": finding.line},
                },
            }
            for finding in result.findings
        ],
        indent=2,
    )
