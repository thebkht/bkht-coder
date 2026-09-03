#!/usr/bin/env python3
"""Produce trajectories by running real tasks through a frontier model.

The other four sources read work that already happened. This one makes work
happen, and it is the answer to the finding `coder dataset stats` reports
first: there is not much data. Existing transcripts are finite and were not
written with training in mind; this is unbounded and is.

Every task here is run by coder itself with `--provider claude-code`, which
matters more than it looks. The trajectory comes out already in coder's own
protocol -- its tools, its calling format, its system prompt -- because it *is*
a coder session; the frontier model is only the thing choosing the next call.
Nothing needs translating afterwards, so nothing can be lost in translating it.

The tasks below are deliberately about this repository. A model that has read
this codebase while learning the protocol is a better agent for it than one
that learned the protocol on someone else's code, and the repo is here.

    python training/generate.py                 # every task, into a temp clone
    python training/generate.py --tasks 5       # the first five
    python training/generate.py --provider codex

Each run appends to the ordinary session directory, so `coder dataset build`
picks the results up with no extra flag: they are coder sessions like any
other, and the `provider` in each header says which model produced them.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Tasks worth learning from: each one needs several tools, has a verifiable
#: answer in this repository, and is the kind of thing somebody actually asks.
#: Read-heavy on purpose -- a trajectory that finds the right file and explains
#: it is the common case, and it is what the local model is worst at.
TASKS = [
    "what does the agent loop do when a tool call is malformed?",
    "where is the context window size decided, and why is it that number?",
    "add a --quiet flag to the doctor subcommand that prints only failures",
    "which module decides whether a tool call needs permission?",
    "find every place the default model name is written down",
    "explain how a session is resumed, from the flag to the messages",
    "what stops a turn that runs forever?",
    "add a test that a session records the provider it ran under",
    "how does the reviewer decide a finding is real?",
    "trace what happens between pressing Esc and the turn stopping",
    "why are tool calls parsed out of message content instead of tool_calls?",
    "what would break if MAX_ITERATIONS were lowered to 5?",
    "add a `coder sessions --count` flag that prints just the number",
    "where does the file tree in the system prompt come from, and what bounds it?",
    "explain the difference between compacting and eliding",
    "which tools are left out in plan mode, and where is that decided?",
]


def run(task: str, workspace: Path, provider: str, timeout: float) -> bool:
    """One task, in its own copy of the repo. True when it answered.

    A copy rather than the repo itself: some of these tasks ask for a change,
    the run is unattended with `--auto`, and a training corpus is not worth a
    dirty working tree.
    """
    print(f"  {task}", flush=True)
    try:
        finished = subprocess.run(
            [
                sys.executable, "-m", "bkht.coder.cli",
                "--provider", provider, "--auto", "--cwd", str(workspace), task,
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print("    timed out", flush=True)
        return False

    if finished.returncode != 0:
        detail = (finished.stderr or "").strip().splitlines()
        print(f"    failed: {detail[-1] if detail else finished.returncode}", flush=True)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="claude-code", help="Backend to generate with.")
    parser.add_argument("--tasks", type=int, default=0, help="How many tasks to run. 0 means all.")
    parser.add_argument("--timeout", type=float, default=900.0, help="Seconds per task.")
    args = parser.parse_args()

    if shutil.which("git") is None:
        print("git is needed to make a scratch copy of the repo.", file=sys.stderr)
        return 1

    tasks = TASKS[: args.tasks] if args.tasks else TASKS
    print(f"{len(tasks)} task(s) through {args.provider}\n")

    answered = 0
    for task in tasks:
        with tempfile.TemporaryDirectory() as scratch:
            workspace = Path(scratch) / "repo"
            subprocess.run(
                ["git", "clone", "--quiet", "--depth", "1", str(REPO), str(workspace)],
                check=True, capture_output=True,
            )
            answered += run(task, workspace, args.provider, args.timeout)

    print(f"\n{answered}/{len(tasks)} answered. Now run: coder dataset build")
    return 0 if answered else 1


if __name__ == "__main__":
    raise SystemExit(main())
