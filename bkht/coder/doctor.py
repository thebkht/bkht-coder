"""Preflight checks: why this install is not working.

The README has always carried a list of the ways a local-model setup goes
wrong -- server not running, model not pulled, `num_ctx` larger than the RAM
can hold -- and a list is a thing the user has to read, remember, and work
through by hand while the thing in front of them is already broken.

Every check here answers one of those, and every failure carries the command
that fixes it. A check that reports a problem without naming the fix has only
moved the search, not ended it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from .instructions import load_instructions
from .provider import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_NUM_CTX
from .session import STATE_DIR
from .skills import discover as discover_skills
from .tools.shell import resolve_shell

PROBE_TIMEOUT = 5.0

OK = "ok"
WARN = "warn"
FAIL = "fail"

# Fitted to the measured table in the README -- 14b on a 16 GB machine, where
# 8192 holds at 10 GB, 16384 at 12 GB, and 32768 at 15 GB. The binding
# constraint is not RAM but what stays on the GPU: at 12 GB of 16 that machine
# already spills to CPU and a warm turn goes from 0.9s to 11s. Hence a usable
# fraction rather than the whole of it.
MODEL_FOOTPRINT_GB = 8.0
GB_PER_1K_CTX = 0.25
USABLE_FRACTION = 0.7


@dataclass
class Check:
    """One question, its answer, and -- when the answer is bad -- the fix."""

    name: str
    status: str
    detail: str
    fix: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL

    def payload(self) -> dict:
        record = {"name": self.name, "status": self.status, "detail": self.detail}
        if self.fix:
            record["fix"] = self.fix
        return record


def version() -> str:
    """The installed version, or nothing.

    A checkout run straight from source has no distribution metadata, and a
    greeting is not worth failing to start over.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version as installed
    except ImportError:  # pragma: no cover - stdlib since 3.8
        return ""
    try:
        return installed("bkht-coder")
    except PackageNotFoundError:
        return ""


def running_from() -> Path:
    """The directory the code executing right now was imported from."""
    return Path(__file__).resolve().parents[2]


def _is_this_project(root: Path) -> bool:
    """Whether ``root`` is a checkout of bkht-coder itself."""
    try:
        return 'name = "bkht-coder"' in (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return False


def check_version(root: Path, origin: Path | None = None) -> Check:
    """Which copy of coder is running, and whether it is the one being edited.

    `uv tool install` copies the package into an environment of its own, so the
    command on PATH keeps working while the checkout moves on without it. The
    symptom is a feature that exists in the source and not in the program, and
    nothing else in this report would explain it.
    """
    origin = Path(origin) if origin is not None else running_from()
    label = f"{version() or 'unknown version'} from {origin}"

    if _is_this_project(root) and root.resolve() != origin:
        return Check(
            "version", WARN,
            f"{label} -- not the checkout you are standing in ({root})",
            "That copy was installed separately and does not change when you edit here. "
            "Run `uv tool install --force --editable .` to point it at this checkout, "
            "or prefix commands with `uv run`.",
        )
    return Check("version", OK, label)


def _tags(host: str) -> tuple[list[str] | None, str]:
    """Model tags the server knows about, or why we could not ask."""
    try:
        response = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=PROBE_TIMEOUT)
        response.raise_for_status()
        models = response.json().get("models", [])
    except httpx.HTTPError as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, f"the server answered, but not with JSON ({exc})"
    return [str(entry.get("name", "")) for entry in models], ""


def check_server(host: str, tags: list[str] | None, problem: str) -> Check:
    if tags is None:
        return Check(
            "ollama",
            FAIL,
            f"{host} did not answer: {problem}",
            f"Start it with `ollama serve`, then check `curl {host}/api/tags`.",
        )
    return Check("ollama", OK, f"{host} answered with {len(tags)} model(s)")


def check_model(model: str, tags: list[str] | None) -> Check:
    if tags is None:
        return Check(
            "model", FAIL,
            "could not ask: the server is not answering",
            "Fix the check above first; this one cannot be answered until it passes.",
        )
    if model in tags:
        return Check("model", OK, f"{model} is pulled")

    # A tag without an explicit version resolves to :latest on the server, so a
    # bare name matching a pulled tag is not a problem to report.
    stem = model.split(":")[0]
    near = [tag for tag in tags if tag.split(":")[0] == stem]
    if near:
        return Check(
            "model", WARN,
            f"{model} is not pulled, but {', '.join(near)} is",
            f"Either run `ollama pull {model}` or pass `--model {near[0]}`.",
        )
    return Check(
        "model", FAIL,
        f"{model} is not pulled",
        f"Run `ollama pull {model}`.",
    )


def total_ram_gb() -> float | None:
    """Physical memory in GB, or None where we cannot ask portably."""
    try:
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, ValueError, OSError):
        pass

    try:
        completed = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, encoding="utf-8", timeout=PROBE_TIMEOUT,
        )
        return int(completed.stdout.strip()) / 1024**3
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _needed_gb(num_ctx: int) -> float:
    return MODEL_FOOTPRINT_GB + (num_ctx / 1024) * GB_PER_1K_CTX


def check_context(num_ctx: int, ram_gb: float | None) -> Check:
    """The single most common cause of a session where every turn takes minutes."""
    if ram_gb is None:
        return Check("num_ctx", OK, f"{num_ctx} tokens (could not read this machine's RAM)")

    needed = _needed_gb(num_ctx)
    usable = ram_gb * USABLE_FRACTION
    if needed <= usable:
        return Check("num_ctx", OK, f"{num_ctx} tokens, about {needed:.0f} GB of {ram_gb:.0f} GB")

    # The largest power-of-two context that still fits, so the fix is a value to
    # paste rather than a direction to experiment in.
    fits = max(
        (size for size in (4096, 8192, 16384, 32768) if _needed_gb(size) <= usable),
        default=None,
    )
    if fits is None:
        remedy = (
            "Even the smallest useful context is tight here; use "
            "`--model qwen2.5-coder:7b`."
        )
    elif num_ctx <= DEFAULT_NUM_CTX:
        # Naming the trade rather than just the smaller number. Lowering the
        # default does buy back the seconds, and it costs the ability to finish:
        # at 8192 this project's own cli.py is 85% of the window, so a turn
        # cannot hold a file and think at the same time. It reads, frees context
        # to make room, loses the file, and reads it again until it runs out of
        # iterations. Slower and finishing beats faster and looping, so the
        # default stays -- but a machine this tight deserves to be told.
        remedy = (
            f"Turns will be slower here. `--num-ctx {fits}` buys the speed back, "
            "at the cost of turns that run out of iterations on larger files. "
            "Try the default first."
        )
    else:
        remedy = f"Lower it with `--num-ctx {fits}`."

    return Check(
        "num_ctx", WARN,
        f"{num_ctx} tokens wants roughly {needed:.0f} GB, and this machine has {ram_gb:.0f} GB",
        "Past what stays on the GPU, the KV cache spills to CPU and every turn "
        f"pays for it. {remedy}",
    )


def check_shell() -> Check:
    shell = resolve_shell()
    if shell is None:
        return Check(
            "shell", FAIL, "no shell found",
            "Install Git for Windows, which provides bash, or run coder in WSL.",
        )
    return Check("shell", OK, f"{shell.label} ({shell.argv[0]})")


def check_git() -> Check:
    if shutil.which("git") is None:
        return Check(
            "git", WARN, "not on PATH",
            "`/diff` and `coder review` need it; everything else works without.",
        )
    return Check("git", OK, "on PATH")


def check_state_dir(directory: Path = None) -> Check:
    directory = Path(directory or STATE_DIR)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".doctor-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check(
            "state", FAIL, f"{directory} is not writable: {exc}",
            "Sessions, history and remembered approvals all live there.",
        )
    return Check("state", OK, f"{directory} is writable")


def check_instructions(root: Path) -> Check:
    loaded = load_instructions(root)
    if not loaded:
        return Check("instructions", OK, "none (no AGENTS.md or CLAUDE.md)")
    names = ", ".join(i.source + (" (truncated)" if i.truncated else "") for i in loaded)
    return Check("instructions", OK, names)


def check_skills(root: Path) -> Check:
    found = discover_skills(root)
    if found.problems:
        return Check(
            "skills",
            WARN,
            f"{len(found.skills)} loaded; {len(found.problems)} skipped: "
            + "; ".join(found.problems),
            "A skill needs both a name and a description in its frontmatter.",
        )
    if not found.skills:
        return Check("skills", OK, "none")
    return Check("skills", OK, ", ".join(skill.name for skill in found.skills))


def check_workspace(root: Path) -> Check:
    """Where the agent has been pointed, when that is somewhere too big.

    Started in a home directory, every search walks decades of files and every
    `.claude/skills` directory in it loads at once. Nothing here is broken, so
    the user is left with a working install that simply never answers.
    """
    root = Path(root).resolve()
    if root == Path.home().resolve():
        return Check(
            "workspace", WARN,
            f"{root} is your home directory",
            "Searches walk everything under the root, so start coder in the project "
            "you mean to work on instead.",
        )
    return Check("workspace", OK, str(root))


def run_checks(
    root: Path,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> list[Check]:
    """Every check, in the order a failing install should be read in.

    Which copy is running and where it was pointed come first: both can make
    every check below them describe a program the user is not actually running.
    Then the server, because nothing under it means anything if it is down.
    """
    tags, problem = _tags(host)
    return [
        check_version(root),
        check_workspace(root),
        check_server(host, tags, problem),
        check_model(model, tags),
        check_context(num_ctx, total_ram_gb()),
        check_shell(),
        check_git(),
        check_state_dir(),
        check_instructions(root),
        check_skills(root),
    ]


MARKERS = {OK: "ok  ", WARN: "warn", FAIL: "FAIL"}


def render(checks: list[Check]) -> str:
    """The report, one line per check plus the fix under any that need one."""
    lines = []
    for check in checks:
        lines.append(f"  {MARKERS[check.status]}  {check.name:<13} {check.detail}")
        if check.fix:
            lines.append(f"        {check.fix}")

    failed = [check for check in checks if check.failed]
    lines.append("")
    lines.append(
        f"{len(failed)} check(s) failed." if failed else "Everything checks out."
    )
    return "\n".join(lines)


def report(
    root: Path,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    num_ctx: int = DEFAULT_NUM_CTX,
    as_json: bool = False,
    out=print,
) -> int:
    """Run every check and print it. Non-zero exit when one failed, for CI."""
    checks = run_checks(root, model=model, host=host, num_ctx=num_ctx)
    out(json.dumps([c.payload() for c in checks], indent=2) if as_json else render(checks))
    return 1 if any(check.failed for check in checks) else 0


def add_arguments(parser) -> None:
    parser.add_argument("--json", action="store_true", help="Machine-readable output for CI.")
