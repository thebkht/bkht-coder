#!/usr/bin/env python3
"""Measure what a turn costs, so a change to what the model is sent can be argued about.

Every claim this project makes about model behaviour is measured rather than
asserted -- the tool-count comparison in 0.4.0's changelog is the standing
example. The thing that has been missing is a ruler for the number that matters
most: how long a task takes, and how many round trips it spends getting there.

The data already exists. `Outcome` carries `seconds`, `iterations`,
`tool_calls`, `sent`, `received` and `stopped`, and every turn's is written to
the session file. Nothing read it back. This does.

    scripts/benchmark.py                          # every task, the default backend
    scripts/benchmark.py --tasks 4                # the first four
    scripts/benchmark.py --provider ollama
    scripts/benchmark.py --out before.json
    scripts/benchmark.py --compare before.json after.json

The backend is whatever a session would use unless one is named, so this
measures the thing the user actually runs -- including the fall back to Ollama
when nothing is serving the default.

The tasks are read-only on purpose, which is the one place this differs from
`training/generate.py`. A task that edits files leaves a different repository
behind than it found, so the second run of a comparison is not measuring the
same work as the first; the numbers move and the change under test gets the
credit or the blame. Every task below can be run a hundred times over the same
tree and be asked the same thing each time.

Each run gets a fresh shallow clone, for the same reason.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bkht.coder.session import SESSION_DIR  # noqa: E402

#: Read-only questions about this repository, each needing several tools and
#: having an answer that is actually in the tree. Ordered roughly by how much
#: searching they need, so `--tasks N` stays a representative prefix rather
#: than N easy ones.
TASKS = [
    "what does the agent loop do when a tool call is malformed?",
    "which module decides whether a tool call needs permission?",
    "where is the context window size decided, and why is it that number?",
    "what stops a turn that runs forever?",
    "explain the difference between compacting and eliding",
    "find every place the default model name is written down",
    "which tools are left out in plan mode, and where is that decided?",
    "explain how a session is resumed, from the flag to the messages",
    "why are tool calls parsed out of message content instead of tool_calls?",
    "trace what happens between pressing Esc and the turn stopping",
]

#: Long enough for a local 14b to finish a search-heavy task, short enough that
#: a wedged run does not hold the whole benchmark. A task that hits this is
#: recorded as a timeout rather than dropped: how often a backend runs out of
#: road is part of what is being measured.
TIMEOUT = 900.0

FIELDS = ("seconds", "iterations", "tool_calls", "sent", "received")


def outcomes(path: Path) -> list[dict]:
    """Every `outcome` record in a session file, in order."""
    found = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a run killed mid-write; the rest is still good
                if record.get("type") == "outcome":
                    found.append(record)
    except OSError:
        return []
    return found


def run_task(task: str, provider: str | None, timeout: float) -> dict:
    """One task in its own clone, returning what the turn recorded about itself.

    The session file is found by watching the directory rather than by asking
    for its name: a session names itself after the clock, and the sub-agent a
    `task` call spawns keeps no file, so exactly one appears per run.
    """
    before = set(SESSION_DIR.glob("*.jsonl")) if SESSION_DIR.exists() else set()

    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch) / "repo"
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", str(REPO), str(workspace)],
            check=True, capture_output=True,
        )
        argv = [sys.executable, "-m", "bkht.coder.cli", "--auto", "--cwd", str(workspace)]
        if provider:
            argv += ["--provider", provider]
        try:
            finished = subprocess.run(
                argv + [task], cwd=REPO, capture_output=True, text=True, timeout=timeout
            )
            failure = "" if finished.returncode == 0 else (
                (finished.stderr or "").strip().splitlines() or ["exit " + str(finished.returncode)]
            )[-1]
        except subprocess.TimeoutExpired:
            failure = f"no answer within {int(timeout)}s"

    appeared = (set(SESSION_DIR.glob("*.jsonl")) - before) if SESSION_DIR.exists() else set()
    row = {"task": task, "failure": failure, "stopped": "", "turns": 0}
    for field in FIELDS:
        row[field] = 0.0 if field == "seconds" else 0

    # A run that never reached the model leaves no session and no outcome; that
    # is a result, not a gap, so the row stays and says what happened.
    for record in [r for path in appeared for r in outcomes(path)]:
        row["turns"] += 1
        row["stopped"] = record.get("stopped", "") or row["stopped"]
        for field in FIELDS:
            row[field] += record.get(field) or 0
    return row


def summarise(rows: list[dict]) -> dict:
    """The aggregate. Medians, not means: one timeout should not move the number."""
    answered = [r for r in rows if r["stopped"] == "answered"]
    stops: dict[str, int] = {}
    for row in rows:
        stops[row["stopped"] or row["failure"] or "no outcome"] = (
            stops.get(row["stopped"] or row["failure"] or "no outcome", 0) + 1
        )
    return {
        "tasks": len(rows),
        "answered": len(answered),
        "total_seconds": round(sum(r["seconds"] for r in rows), 1),
        "median_seconds": round(statistics.median([r["seconds"] for r in rows]), 1) if rows else 0,
        "median_iterations": statistics.median([r["iterations"] for r in rows]) if rows else 0,
        "median_tool_calls": statistics.median([r["tool_calls"] for r in rows]) if rows else 0,
        "stops": stops,
    }


def render(rows: list[dict], summary: dict) -> str:
    """The table, sized to the longest task so nothing wraps."""
    width = max([len(r["task"]) for r in rows] + [4])
    lines = [
        f"{'task'.ljust(width)}  {'secs':>7}  {'iters':>5}  {'calls':>5}  stopped",
        f"{'-' * width}  {'-' * 7}  {'-' * 5}  {'-' * 5}  {'-' * 14}",
    ]
    for row in rows:
        lines.append(
            f"{row['task'].ljust(width)}  {row['seconds']:>7.1f}  "
            f"{row['iterations']:>5}  {row['tool_calls']:>5}  "
            f"{row['stopped'] or row['failure'] or 'no outcome'}"
        )
    lines += [
        "",
        f"{summary['answered']}/{summary['tasks']} answered "
        f"in {summary['total_seconds']}s total",
        f"median: {summary['median_seconds']}s, "
        f"{summary['median_iterations']} iterations, "
        f"{summary['median_tool_calls']} tool calls",
        "stops: " + ", ".join(f"{name} {count}" for name, count in sorted(summary["stops"].items())),
    ]
    return "\n".join(lines)


def compare(before: dict, after: dict) -> str:
    """Two runs side by side, which is the only reason to keep the first one.

    Matched on the task text, so a run made with `--tasks 4` can still be
    compared against a full one -- only the tasks both did are reported, and
    the count says how many that was.
    """
    left = {row["task"]: row for row in before["rows"]}
    right = {row["task"]: row for row in after["rows"]}
    shared = [task for task in left if task in right]
    if not shared:
        return "The two runs share no tasks."

    width = max(len(task) for task in shared)
    lines = [
        f"{before.get('label', 'before')} -> {after.get('label', 'after')}"
        f"   ({len(shared)} shared task(s))",
        "",
        f"{'task'.ljust(width)}  {'secs':>16}  {'iters':>11}  {'calls':>11}",
        f"{'-' * width}  {'-' * 16}  {'-' * 11}  {'-' * 11}",
    ]
    for task in shared:
        a, b = left[task], right[task]
        lines.append(
            f"{task.ljust(width)}  "
            f"{a['seconds']:>6.1f} -> {b['seconds']:>6.1f}  "
            f"{a['iterations']:>4} -> {b['iterations']:>4}  "
            f"{a['tool_calls']:>4} -> {b['tool_calls']:>4}"
        )

    def total(rows, field):
        return sum(rows[task][field] for task in shared)

    seconds_before, seconds_after = total(left, "seconds"), total(right, "seconds")
    lines += ["", f"total: {seconds_before:.1f}s -> {seconds_after:.1f}s"]
    if seconds_before:
        change = 100 * (seconds_after - seconds_before) / seconds_before
        # Said plainly rather than dressed up: a benchmark whose own summary
        # rounds in the change's favour is not evidence of anything.
        lines.append(f"       {change:+.1f}%")
    lines.append(
        f"answered: {sum(1 for t in shared if left[t]['stopped'] == 'answered')}"
        f" -> {sum(1 for t in shared if right[t]['stopped'] == 'answered')}"
        f" of {len(shared)}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default=None, help="Backend. Default: whatever a session would use.")
    parser.add_argument("--tasks", type=int, default=0, help="How many tasks to run. 0 means all.")
    parser.add_argument("--timeout", type=float, default=TIMEOUT, help="Seconds per task.")
    parser.add_argument("--out", type=Path, default=None, help="Write the run to this JSON file.")
    parser.add_argument("--label", default="", help="A name for this run, shown when comparing.")
    parser.add_argument(
        "--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"),
        help="Report two saved runs against each other and exit. Runs nothing.",
    )
    args = parser.parse_args(argv)

    if args.compare:
        before, after = (json.loads(path.read_text()) for path in args.compare)
        print(compare(before, after))
        return 0

    if shutil.which("git") is None:
        print("git is needed to make a scratch copy of the repo.", file=sys.stderr)
        return 1

    tasks = TASKS[: args.tasks] if args.tasks else TASKS
    print(f"{len(tasks)} task(s) through {args.provider or 'the default backend'}\n", flush=True)

    rows = []
    for number, task in enumerate(tasks, start=1):
        print(f"  {number}/{len(tasks)} {task}", flush=True)
        row = run_task(task, args.provider, args.timeout)
        print(
            f"      {row['seconds']:.1f}s, {row['iterations']} iterations, "
            f"{row['tool_calls']} tool calls, "
            f"{row['stopped'] or row['failure'] or 'no outcome'}",
            flush=True,
        )
        rows.append(row)

    summary = summarise(rows)
    print("\n" + render(rows, summary))

    if args.out:
        args.out.write_text(json.dumps(
            {"label": args.label or args.out.stem, "provider": args.provider or "default",
             "summary": summary, "rows": rows},
            indent=2,
        ) + "\n")
        print(f"\nwritten to {args.out}")

    return 0 if summary["answered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
