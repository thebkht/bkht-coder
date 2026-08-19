"""The ``coder review`` command.

Review itself is read-only. Fixing is a second phase you opt into with --fix,
and it goes through the normal agent loop and the normal permission gate, so
nothing is written without approval and /undo covers it like any other edit.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..agent import Agent
from ..context import file_tree
from ..permissions import ASK, AUTO, Permissions, ask_terminal
from ..prompts import system_prompt
from ..provider import OllamaProvider, for_review
from ..session import Session, Snapshots
from ..tools import build_registry
from . import render
from .diff import GitError, collect_diff, collect_files
from .reviewer import DIMENSIONS, ReviewListener, Reviewer

FIX_TASK = """\
Fix this code-review finding.

  file:     {file}:{line}
  severity: {severity}
  category: {category}
  problem:  {summary}
  fails by: {scenario}
  suggested: {suggestion}

Read the file first, make the smallest change that fixes the problem, and do
not change anything else. If after reading the code you believe the finding is
wrong, say so and change nothing."""


class Progress(ReviewListener):
    """Prints what the review is doing, since it can take minutes."""

    def __init__(self, stream=sys.stderr, colour: bool = True) -> None:
        self.stream = stream
        self.colour = colour

    def _say(self, text: str) -> None:
        prefix = f"{render.DIM}" if self.colour else ""
        suffix = render.RESET if self.colour else ""
        print(f"{prefix}{text}{suffix}", file=self.stream, flush=True)

    def on_pass(self, unit: int, total: int, dimension: str) -> None:
        self._say(f"  · unit {unit}/{total}: {dimension}")

    def on_candidates(self, dimension: str, count: int) -> None:
        if count:
            self._say(f"      {count} candidate{'s' if count != 1 else ''}")

    def on_verify(self, index: int, total: int, finding) -> None:
        self._say(f"  · verifying {index}/{total}: {finding.file}:{finding.line}")

    def on_verdict(self, finding, refuted: bool, reason: str) -> None:
        if refuted:
            self._say(f"      refuted — {reason or 'no reason given'}")


def add_arguments(parser) -> None:
    """Declare the review flags on a subparser."""
    parser.add_argument("range", nargs="?", help="Commit range, e.g. HEAD~3..HEAD")
    parser.add_argument("--base", help="Review everything on this branch, from its merge base with BASE.")
    parser.add_argument("--staged", action="store_true", help="Review staged changes only.")
    parser.add_argument("--files", nargs="+", help="Review whole files rather than a diff.")
    parser.add_argument("--dimension", action="append", choices=DIMENSIONS, help="Limit to one dimension. Repeatable.")
    parser.add_argument("--no-verify", action="store_true", help="Skip the refutation pass. Expect false positives.")
    parser.add_argument("--json", action="store_true", help="Emit the findings as JSON.")
    parser.add_argument("--output", help="Write a Markdown report to this path.")
    parser.add_argument("--fix", action="store_true", help="After reporting, offer to fix findings.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")


def gather(root: Path, args):
    """Collect the diff the flags describe."""
    if args.files:
        return collect_files(root, args.files)
    return collect_diff(
        root, base=args.base, revision_range=args.range, staged=args.staged
    )


def choose(findings, ask=input) -> list:
    """Ask which findings to fix. Accepts numbers, 'all', or nothing."""
    print(f"\nFix which findings? 1-{len(findings)}, comma separated, or 'all'.")
    try:
        answer = ask("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return []

    if not answer or answer in ("none", "n", "q"):
        return []
    if answer == "all":
        return list(findings)

    chosen = []
    for part in answer.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= len(findings):
            chosen.append(findings[int(part) - 1])
        elif part:
            print(f"Ignoring {part!r}: not a finding number.")
    return chosen


def fix(findings, provider, root: Path, mode: str = ASK, prompt=ask_terminal) -> int:
    """Hand selected findings to the normal agent loop, one at a time."""
    snapshots = Snapshots()
    registry, workspace = build_registry(root, snapshots=snapshots)
    permissions = Permissions(mode=mode, workspace=workspace, prompt=prompt)
    system = system_prompt(registry, str(workspace.root), file_tree(workspace.root))

    fixed = 0
    for index, finding in enumerate(findings, start=1):
        print(f"\n[{index}/{len(findings)}] {finding.file}:{finding.line} — {finding.summary}")
        agent = Agent(
            provider=provider,
            registry=registry,
            session=Session(system=system),
            permissions=permissions,
        )
        outcome = agent.run(
            FIX_TASK.format(
                file=finding.file,
                line=finding.line,
                severity=finding.severity,
                category=finding.category,
                summary=finding.summary,
                scenario=finding.scenario or "(not stated)",
                suggestion=finding.suggestion or "(none given)",
            )
        )
        if outcome.answer:
            print(outcome.answer)
        if outcome.stopped == "answered":
            fixed += 1
        else:
            print(f"[stopped: {outcome.stopped}]", file=sys.stderr)

    return fixed


def run(args) -> int:
    """Execute ``coder review``. Returns the process exit status."""
    root = Path(args.cwd).expanduser().resolve()

    try:
        files = gather(root, args)
    except GitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not files:
        print("Nothing to review: no changes were found.", file=sys.stderr)
        return 0

    provider = for_review(
        OllamaProvider(model=args.model, host=args.host, num_ctx=args.num_ctx)
    )
    listener = ReviewListener() if (args.quiet or args.json) else Progress()
    reviewer = Reviewer(
        provider,
        root,
        dimensions=tuple(args.dimension) if args.dimension else DIMENSIONS,
        listener=listener,
        verify=not args.no_verify,
    )
    result = reviewer.review(files)

    if args.json:
        print(render.as_json(result))
    else:
        print(render.terminal(result, colour=sys.stdout.isatty()))

    if args.output:
        Path(args.output).expanduser().write_text(render.markdown(result), encoding="utf-8")
        print(f"\nWrote {args.output}", file=sys.stderr)

    if args.fix and result.findings:
        # Numbered here so the selection matches what was just printed.
        print("\nFindings:")
        for index, finding in enumerate(result.findings, start=1):
            print(f"  {index}. {finding.file}:{finding.line} — {finding.summary}")
        selected = choose(result.findings)
        if selected:
            fix(selected, provider, root, mode=AUTO if args.auto else ASK)

    # Findings are a report, not a build failure, unless CI asked for JSON.
    if args.json and result.findings:
        return 1
    return 0
