"""Collecting a diff from git and splitting it into reviewable units.

Hunks carry real file line numbers rather than diff offsets, so a finding can
cite ``file:line`` and the user can go straight there. Getting this wrong makes
every finding untrustworthy even when the finding itself is correct.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@(.*)$")

# Units are sized in characters and converted to tokens at roughly 4:1. A 32K
# window has to hold the system prompt, the unit, and the model's reply, so the
# unit itself gets a fraction of it.
DEFAULT_UNIT_CHARS = 12_000


class GitError(RuntimeError):
    """git could not produce the requested diff."""


@dataclass
class Change:
    """One line of a hunk, with the line number it has in the new file."""

    kind: str  # ' ' context, '+' added, '-' removed
    line: int | None  # None for removed lines, which have no new-file number
    text: str


@dataclass
class Hunk:
    """A contiguous run of changes in one file."""

    start: int
    changes: list[Change] = field(default_factory=list)
    context: str = ""  # the function or section name git puts after @@

    @property
    def end(self) -> int:
        numbers = [c.line for c in self.changes if c.line is not None]
        return max(numbers) if numbers else self.start

    @property
    def added(self) -> int:
        return sum(1 for c in self.changes if c.kind == "+")

    def render(self) -> str:
        """The hunk as the model sees it: line numbers, then the marker.

        Every line is numbered, not just changed ones, because the model needs
        to cite the line a problem is *on*, which is often a context line.
        """
        out = []
        for change in self.changes:
            number = f"{change.line:>5}" if change.line is not None else "    -"
            out.append(f"{number} {change.kind}{change.text}")
        return "\n".join(out)


@dataclass
class FileDiff:
    """Every hunk in one file."""

    path: str
    hunks: list[Hunk] = field(default_factory=list)
    status: str = "modified"  # modified | added | deleted

    @property
    def added(self) -> int:
        return sum(h.added for h in self.hunks)

    def render(self) -> str:
        body = "\n\n".join(h.render() for h in self.hunks)
        return f"--- {self.path} ({self.status}) ---\n{body}"


@dataclass
class ReviewUnit:
    """A group of hunks small enough to review in one model call."""

    files: list[FileDiff] = field(default_factory=list)

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.files]

    def render(self) -> str:
        return "\n\n".join(f.render() for f in self.files)

    def size(self) -> int:
        return len(self.render())


def run_git(args: list[str], root: Path) -> str:
    """Run a git command in ``root``, raising :class:`GitError` on failure."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitError(f"could not run git: {exc}") from None

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise GitError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def is_repo(root: Path) -> bool:
    try:
        run_git(["rev-parse", "--git-dir"], root)
    except GitError:
        return False
    return True


def parse_diff(text: str) -> list[FileDiff]:
    """Parse unified diff output into per-file hunks with real line numbers."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    hunk: Hunk | None = None
    next_line = 0
    # git emits `--- /dev/null` before the `+++` line that names a new file,
    # so the status has to be carried forward rather than applied on sight.
    pending_added = False

    for raw in text.splitlines():
        if raw.startswith("diff --git "):
            current, hunk, pending_added = None, None, False
            continue

        # File headers are only recognised outside a hunk. Inside one, a
        # removed line reading "-- x" renders as "--- x" and an added "++ x"
        # as "+++ x", which would otherwise be mistaken for a new file and
        # silently drop the rest of the real one.
        if hunk is None and raw.startswith("--- "):
            pending_added = raw[4:].strip() == "/dev/null"
            continue

        if hunk is None and raw.startswith("+++ "):
            path = raw[4:].strip()
            if path == "/dev/null":
                current = None  # a deletion has nothing to review
                continue
            current = FileDiff(
                path=path[2:] if path.startswith("b/") else path,
                status="added" if pending_added else "modified",
            )
            files.append(current)
            hunk = None
            continue

        if current is None:
            continue

        match = HUNK_HEADER.match(raw)
        if match:
            next_line = int(match.group(1))
            hunk = Hunk(start=next_line, context=match.group(3).strip())
            current.hunks.append(hunk)
            continue

        if hunk is None:
            continue

        marker, body = (raw[0], raw[1:]) if raw else (" ", "")
        if marker == "+":
            hunk.changes.append(Change("+", next_line, body))
            next_line += 1
        elif marker == "-":
            hunk.changes.append(Change("-", None, body))
        elif marker == " ":
            hunk.changes.append(Change(" ", next_line, body))
            next_line += 1
        # '\' (no newline at end of file) and anything else is not a line.

    return [f for f in files if f.hunks]


def collect_diff(
    root: Path,
    base: str | None = None,
    revision_range: str | None = None,
    staged: bool = False,
) -> list[FileDiff]:
    """Collect the diff to review.

    ``base`` reviews everything on this branch by diffing from the merge base,
    not from the tip of ``base`` -- otherwise unrelated commits landing on the
    base branch show up as if this branch had made them.
    """
    root = Path(root)
    if not is_repo(root):
        raise GitError(f"{root} is not a git repository")

    if revision_range:
        return parse_diff(run_git(["diff", "--unified=3", revision_range], root))

    if base:
        merge_base = run_git(["merge-base", base, "HEAD"], root).strip()
        if not merge_base:
            raise GitError(f"no merge base between {base} and HEAD")
        return parse_diff(run_git(["diff", "--unified=3", f"{merge_base}..HEAD"], root))

    if staged:
        return parse_diff(run_git(["diff", "--unified=3", "--cached"], root))

    # Uncommitted work: staged and unstaged together, which is what "review my
    # changes" means to someone who has staged only part of them.
    return parse_diff(run_git(["diff", "--unified=3", "HEAD"], root))


def collect_files(root: Path, paths: list[str]) -> list[FileDiff]:
    """Present whole files as if every line were added, for ``--files``."""
    root = Path(root)
    files = []
    for name in paths:
        target = (root / name).resolve()
        if not target.is_file():
            raise GitError(f"not a file: {name}")

        text = target.read_text(encoding="utf-8", errors="replace")
        changes = [
            Change("+", number, line)
            for number, line in enumerate(text.splitlines(), start=1)
        ]
        if not changes:
            continue
        files.append(
            FileDiff(
                path=name,
                hunks=[Hunk(start=1, changes=changes)],
                status="whole file",
            )
        )
    return files


def chunk(files: list[FileDiff], budget: int = DEFAULT_UNIT_CHARS) -> list[ReviewUnit]:
    """Group hunks into units that fit the context budget.

    A hunk is never split across units: half a function is worse than no
    function, and a model shown half will confidently review the missing part.
    A single hunk larger than the budget therefore gets a unit of its own and
    goes over -- reporting that honestly beats silently truncating it.
    """
    units: list[ReviewUnit] = []
    current = ReviewUnit()
    used = 0

    for file in files:
        for hunk in file.hunks:
            size = len(hunk.render()) + len(file.path) + 32

            if used and used + size > budget:
                units.append(current)
                current, used = ReviewUnit(), 0

            # Keep hunks from the same file together within a unit.
            if current.files and current.files[-1].path == file.path:
                current.files[-1].hunks.append(hunk)
            else:
                current.files.append(
                    FileDiff(path=file.path, hunks=[hunk], status=file.status)
                )
            used += size

    if current.files:
        units.append(current)
    return units
