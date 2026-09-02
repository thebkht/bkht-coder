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
from .provider import DEFAULT_PROVIDER, ProviderError, build
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
#
# These two are now only the fallback. The server knows what the weights weigh
# and what shape the KV cache is, and is asked first -- a single footprint is
# wrong the moment `--model` names anything but the 14b, and a single
# per-context figure is off by 3.4x between the two models this project ships
# with. They are still what the report degrades to when the server cannot say.
MODEL_FOOTPRINT_GB = 8.0
GB_PER_1K_CTX = 0.25
USABLE_FRACTION = 0.7

# A discrete card holds nothing but the display and the runtime, so its budget
# is the whole of it less a fixed reserve -- subtracted, not scaled. Scaling
# 8 GB by USABLE_FRACTION would leave 5.6 GB and reject a 7b at 16384, which is
# precisely the configuration that works on that card.
VRAM_HEADROOM_GB = 0.8


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


@dataclass(frozen=True)
class Budget:
    """How much memory a model may use here, and which memory that is.

    Two machines wear the same number very differently. On unified memory the
    model competes with everything else running, so only a fraction is really
    available; a discrete card is the model's alone. Which one was measured
    matters more than either number: an 8 GB card in a 32 GB box is the case
    this report used to get wrong -- it read the 32, passed, and left the user
    running two thirds of the weights on the CPU.
    """

    gb: float | None
    label: str = "memory"
    #: A GPU of its own, rather than memory shared with the rest of the machine.
    dedicated: bool = False

    @property
    def usable(self) -> float | None:
        if self.gb is None:
            return None
        if self.dedicated:
            return max(self.gb - VRAM_HEADROOM_GB, 0.0)
        return self.gb * USABLE_FRACTION


def version() -> str:
    """The installed version, or nothing.

    Two sources, because there are two ways to be running. An installed
    distribution carries the version in its metadata, which is the authority:
    it is what the wheel on PATH was built as, whatever the checkout beside it
    now says. A checkout run straight from source has no such metadata, and
    falls back to the file hatch-vcs writes at build time -- which describes an
    untagged commit as a dev version of the release it leads to, so a working
    copy never reports itself as a release.

    Neither is worth failing to start over: with both absent this returns
    nothing, and every caller already prints a shorter line for that.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version as installed
    except ImportError:  # pragma: no cover - stdlib since 3.8
        return ""
    try:
        return installed("bkht-coder")
    except PackageNotFoundError:
        pass
    try:
        from ._version import __version__
    except ImportError:
        return ""
    return str(__version__)


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


def _tags(host: str) -> tuple[list[dict] | None, str]:
    """Every model the server knows about, or why we could not ask.

    Whole entries rather than names: each one carries the size of the weights,
    which is the number this report used to hardcode.
    """
    try:
        response = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=PROBE_TIMEOUT)
        response.raise_for_status()
        models = response.json().get("models", [])
    except httpx.HTTPError as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, f"the server answered, but not with JSON ({exc})"
    return [entry for entry in models if isinstance(entry, dict)], ""


def _names(entries: list[dict] | None) -> list[str] | None:
    """Just the tags, for the checks that only ask whether one is present."""
    return None if entries is None else [str(entry.get("name", "")) for entry in entries]


def _show(host: str, model: str) -> dict | None:
    """One model's metadata, or nothing.

    Every caller has a fallback, so a server too old to answer this degrades the
    report rather than failing it.
    """
    try:
        response = httpx.post(
            f"{host.rstrip('/')}/api/show", json={"model": model}, timeout=PROBE_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _ps(host: str) -> list[dict] | None:
    """What is loaded right now, and how much of it is on the GPU."""
    try:
        response = httpx.get(f"{host.rstrip('/')}/api/ps", timeout=PROBE_TIMEOUT)
        response.raise_for_status()
        models = response.json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return None
    return [entry for entry in models if isinstance(entry, dict)]


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


def check_backend(name: str, model: str) -> Check:
    """The one check that replaces the three Ollama ones for a hosted backend.

    Nothing here sends a turn. Whether the tool is logged in cannot be asked
    without spending the user's own quota to find out, and a report that costs
    money to run is a report nobody runs -- so this answers the half that is
    free, and a login that has lapsed surfaces on the first turn instead, in the
    tool's own words.
    """
    try:
        backend = build(name, model=model)
    except ProviderError as exc:
        return Check("backend", FAIL, str(exc), f"Run `coder config set provider {DEFAULT_PROVIDER}`.")

    command = getattr(backend, "command", name)
    if not backend.available():
        return Check(
            "backend", FAIL, f"{command} is not installed",
            getattr(backend, "install_hint", ""),
        )
    return Check("backend", OK, f"{command} answers for {model}")


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


def _smi(argv: list[str]) -> str | None:
    """A GPU tool's output, or nothing if it is absent or unhappy."""
    if shutil.which(argv[0]) is None:
        return None
    try:
        completed = subprocess.run(
            argv, capture_output=True, encoding="utf-8", timeout=PROBE_TIMEOUT
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def gpu_vram_gb() -> float | None:
    """The largest discrete GPU's memory in GB, or None where there is none.

    The largest, not the sum: Ollama loads a model onto one card unless it is
    told otherwise, so adding two together would budget memory no single run can
    reach. None means no discrete card was found, which is the Apple Silicon
    case as much as the no-GPU one -- there the machine's memory is the answer.
    """
    output = _smi(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    )
    if output:
        sizes = []
        for line in output.splitlines():
            try:
                sizes.append(float(line.strip()) / 1024)  # nvidia-smi reports MiB
            except ValueError:
                continue
        if sizes:
            return max(sizes)

    output = _smi(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if output:
        try:
            payload = json.loads(output)
        except ValueError:
            payload = {}
        sizes = []
        for card in payload.values() if isinstance(payload, dict) else []:
            for key, value in (card or {}).items():
                if "total" in key.lower():
                    try:
                        sizes.append(float(value) / 1024**3)
                    except (TypeError, ValueError):
                        continue
        if sizes:
            return max(sizes)
    return None


def memory_budget() -> Budget:
    """The memory that actually binds here, and its name."""
    vram = gpu_vram_gb()
    if vram is not None:
        return Budget(vram, "VRAM", dedicated=True)
    return Budget(total_ram_gb(), "memory")


def weights_gb(model: str, entries: list[dict] | None) -> float | None:
    """What one model's weights weigh, as the server reports them."""
    for entry in entries or []:
        if str(entry.get("name", "")) == model:
            try:
                return float(entry["size"]) / 1024**3
            except (KeyError, TypeError, ValueError):
                return None
    return None


def _info(show: dict | None, suffix: str):
    """One field of a model's metadata, whatever family prefixes it.

    The keys are named for the architecture -- `qwen2.block_count` here,
    something else on the next model -- and the suffix is the part that is the
    same everywhere.
    """
    for key, value in ((show or {}).get("model_info") or {}).items():
        if str(key).split(".", 1)[-1] == suffix:
            return value
    return None


def kv_gb(show: dict | None, num_ctx: int) -> float | None:
    """The KV cache for ``num_ctx`` tokens, from the model's own metadata.

    Two for the key and the value, two again for float16. This is the number the
    single ``GB_PER_1K_CTX`` constant was standing in for, and it is not a
    property of a model's parameter count: 14b's cache is 3.4x 7b's because it
    has more layers and twice the KV heads, not because it is twice the model.
    """
    try:
        head_dim = float(_info(show, "embedding_length")) / float(
            _info(show, "attention.head_count")
        )
        layers = int(_info(show, "block_count"))
        kv_heads = int(_info(show, "attention.head_count_kv"))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return 2 * layers * kv_heads * head_dim * num_ctx * 2 / 1024**3


def _needed_gb(
    num_ctx: int,
    weights: float = MODEL_FOOTPRINT_GB,
    per_1k: float = GB_PER_1K_CTX,
) -> float:
    return weights + (num_ctx / 1024) * per_1k


def check_context(
    num_ctx: int,
    budget: "Budget | float | None",
    weights: float | None = None,
    per_1k: float | None = None,
    alternatives: list[tuple[str, float]] | None = None,
) -> Check:
    """The single most common cause of a session where every turn takes minutes.

    ``weights`` and ``per_1k`` come from the server when it answered, and fall
    back to the fitted constants when it did not. ``alternatives`` are the other
    pulled tags of the same model and what each would cost here, so a machine
    too small for any window can be sent to one that fits by name.
    """
    budget = budget if isinstance(budget, Budget) else Budget(budget)
    weights = MODEL_FOOTPRINT_GB if weights is None else weights
    per_1k = GB_PER_1K_CTX if per_1k is None else per_1k

    if budget.gb is None:
        return Check(
            "num_ctx", OK, f"{num_ctx} tokens (could not read this machine's memory)"
        )

    needed = _needed_gb(num_ctx, weights, per_1k)
    usable = budget.usable
    if needed <= usable:
        return Check(
            "num_ctx",
            OK,
            f"{num_ctx} tokens, about {needed:.0f} GB of "
            f"{budget.gb:.0f} GB of {budget.label}",
        )

    # The largest power-of-two context that still fits, so the fix is a value to
    # paste rather than a direction to experiment in.
    fits = max(
        (
            size
            for size in (4096, 8192, 16384, 32768)
            if _needed_gb(size, weights, per_1k) <= usable
        ),
        default=None,
    )
    if fits is None:
        # Named from what is actually pulled, so the advice stays true on a
        # machine and a model set this code has never seen. The 7b is the
        # fallback because it is what the README ships.
        smaller = [tag for tag, cost in sorted(alternatives or [], key=lambda p: -p[1]) if cost <= usable]
        remedy = (
            "Even the smallest useful context is tight here; use "
            f"`--model {smaller[0] if smaller else 'qwen2.5-coder:7b'}`."
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
        f"{num_ctx} tokens wants roughly {needed:.0f} GB, and this machine has "
        f"{budget.gb:.0f} GB of {budget.label}",
        "Past what stays on the GPU, the KV cache spills to CPU and every turn "
        f"pays for it. {remedy}",
    )


def check_placement(model: str, loaded: list[dict] | None) -> Check:
    """Where the weights actually are -- the one number here that is measured.

    Every other memory line in this report is arithmetic on a model's size, and
    arithmetic is what got the 8 GB card wrong for so long. This one is the
    server's own answer, and it is the answer that decides whether a turn takes
    a second or a minute: whatever did not fit on the GPU is walked through on
    the CPU once per token.

    Nothing here loads a model. A health check that pulled nine gigabytes into
    memory as a side effect would be a check nobody runs.
    """
    if loaded is None:
        return Check("placement", OK, "could not ask the server what is loaded")

    entry = next(
        (
            item
            for item in loaded
            if model in (str(item.get("name", "")), str(item.get("model", "")))
        ),
        None,
    )
    if entry is None:
        return Check(
            "placement", OK,
            f"{model} is not loaded; run a turn and re-run doctor to see the split",
        )

    try:
        total = float(entry["size"])
        resident = float(entry.get("size_vram", 0))
    except (KeyError, TypeError, ValueError):
        return Check("placement", OK, "the server did not say how it placed the model")
    if total <= 0:
        return Check("placement", OK, "the server reported a model of no size")

    share = resident / total
    if share >= 0.99:
        return Check(
            "placement", OK, f"100% on GPU ({total / 1024**3:.1f} GB resident)"
        )
    return Check(
        "placement", WARN,
        f"{share * 100:.0f}% on GPU -- {(total - resident) / 1024**3:.1f} GB of "
        f"{total / 1024**3:.1f} GB is on the CPU",
        "Every token walks the part that is not resident, which is the whole "
        "difference between a turn that takes a second and one that takes a "
        "minute. Lower `--num-ctx`, or run a smaller model.",
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


def _model_checks(provider: str, model: str, host: str, num_ctx: int) -> list[Check]:
    """The checks that only make sense for the backend actually in use.

    Four of them are about a local server: whether it is up, whether the weights
    are pulled, whether this machine has the memory for the window, and where
    the server actually put the model. None of that describes a model reached
    through somebody else's command line, and a report that fails on a server
    the user is not using is a report that teaches them to ignore it.
    """
    if provider != DEFAULT_PROVIDER:
        return [check_backend(provider, model)]

    entries, problem = _tags(host)
    names = _names(entries)
    weights = per_1k = None
    alternatives: list[tuple[str, float]] = []
    loaded: list[dict] | None = None
    if entries is not None:
        # Only worth asking once the server has answered at all; offline, every
        # one of these would be the same connection error three more times.
        weights = weights_gb(model, entries)
        per_1k = kv_gb(_show(host, model), 1024)
        alternatives = _alternatives(host, model, entries, num_ctx)
        loaded = _ps(host)

    return [
        check_server(host, names, problem),
        check_model(model, names),
        # The estimate first, the measurement directly under it.
        check_context(num_ctx, memory_budget(), weights, per_1k, alternatives),
        check_placement(model, loaded),
    ]


def _alternatives(
    host: str, model: str, entries: list[dict], num_ctx: int
) -> list[tuple[str, float]]:
    """Other pulled sizes of the same model, and what each would cost here.

    Same stem only. Offering to solve a memory problem by switching the user to
    an unrelated model they happen to have pulled is offering to change the
    answer, not the machine.
    """
    stem = model.split(":")[0]
    found = []
    for entry in entries:
        tag = str(entry.get("name", ""))
        if tag == model or tag.split(":")[0] != stem:
            continue
        weights = weights_gb(tag, entries)
        if weights is None:
            continue
        per_1k = kv_gb(_show(host, tag), 1024) or GB_PER_1K_CTX
        found.append((tag, _needed_gb(num_ctx, weights, per_1k)))
    return found


def run_checks(
    root: Path,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    num_ctx: int = DEFAULT_NUM_CTX,
    provider: str = DEFAULT_PROVIDER,
) -> list[Check]:
    """Every check, in the order a failing install should be read in.

    Which copy is running and where it was pointed come first: both can make
    every check below them describe a program the user is not actually running.
    Then the model, because nothing under it means anything if it cannot answer.
    """
    return [
        check_version(root),
        check_workspace(root),
        *_model_checks(provider, model, host, num_ctx),
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
    provider: str = DEFAULT_PROVIDER,
    as_json: bool = False,
    out=print,
) -> int:
    """Run every check and print it. Non-zero exit when one failed, for CI."""
    checks = run_checks(root, model=model, host=host, num_ctx=num_ctx, provider=provider)
    out(json.dumps([c.payload() for c in checks], indent=2) if as_json else render(checks))
    return 1 if any(check.failed for check in checks) else 0


def add_arguments(parser) -> None:
    parser.add_argument("--json", action="store_true", help="Machine-readable output for CI.")
