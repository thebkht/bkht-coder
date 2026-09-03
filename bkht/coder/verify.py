"""Cheap static checks on a Python file the model is about to write.

Nothing used to look at what an edit *meant*. `edit_file` checked that
`old_string` appeared exactly once, wrote the bytes, and reported "Edited" --
and every one of those statements was true of `from .session import STATE_DIR,
Input` even though `session.py` has never defined `Input`. The string matched.
The write succeeded. The mistake surfaced at the next import, which in a REPL
session is the next thing the user types.

A local model invents a name because the sentence reads well. That is not an
occasional slip, it is the failure mode, so it is worth a check that costs a
millisecond.

Both checks here are static. The check that would catch everything is "does the
module still import?", and running it means executing the model's new code to
find out whether the model's new code is safe to execute. These parse instead.

They are deliberately asymmetric:

* A syntax error is **refused**. A file that does not parse is never what
  anyone meant, and there is no case where writing it and reporting success is
  the more useful outcome.
* An unresolved import is **reported**, not refused. Names can arrive at
  runtime -- a star import, a conditional definition, a module ``__getattr__``
  -- and a check that blocks a correct edit is worse than one that misses an
  incorrect one. So this one warns, in both directions: to the model in the
  tool result, and to the human at the approval prompt, which is the moment
  someone is already deciding.

The second half of this module is the check the first half declined to make.
Running the model's new code to find out whether it works is exactly what the
static checks refuse to do -- but *the project's own test command* is a
different proposition from the module just written. It is a command the user
chose, wrote down, and already runs by hand; the agent running it is not the
agent deciding to execute something. So :func:`suite` runs it once a turn has
finished writing, and a failure goes back as a tool result the model can
correct from, which is the same correction path a malformed call takes.

Nothing runs until ``verify_command`` is set. :func:`detect` will suggest one,
and ``doctor`` will show you the suggestion, but an inferred command is never
run unasked: the whole safety of this rests on the command being the user's,
and a command coder guessed at is not.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path


def syntax_error(source: str, label: str) -> str | None:
    """The syntax error in ``source``, phrased for the model, or None."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        where = f" at line {exc.lineno}" if exc.lineno else ""
        return (
            f"This edit would leave {label} unparseable: {exc.msg}{where}. "
            "The file was not written. Read it again and correct the change."
        )
    except ValueError as exc:
        # Source containing null bytes, and anything else ast rejects outright.
        return f"This edit would leave {label} unreadable: {exc}. Not written."
    return None


def unknown_imports(source: str, path: Path, root: Path) -> list[str]:
    """Names imported from a workspace module that does not define them.

    Only relative imports resolving to a file inside ``root`` are checked. An
    absolute import names a package this cannot see, and guessing about one
    would produce exactly the false alarm that makes a warning worth ignoring.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []

    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue

        target = _resolve(path, root, node.level, node.module)
        if target is None:
            continue

        defined = _module_names(target)
        if defined is None:
            continue

        missing = [
            alias.name
            for alias in node.names
            if alias.name != "*"
            and alias.name not in defined
            and not _is_submodule(target, alias.name)
        ]
        if missing:
            names = ", ".join(f"`{name}`" for name in missing)
            problems.append(
                f"{_relative(target, root)} does not define {names}, "
                f"imported on line {node.lineno}."
            )
    return problems


def _resolve(path: Path, root: Path, level: int, module: str | None) -> Path | None:
    """The file a relative import points at, if it is one inside ``root``."""
    base = path.parent
    for _ in range(level - 1):
        base = base.parent
    if module:
        base = base.joinpath(*module.split("."))

    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def _is_submodule(target: Path, name: str) -> bool:
    """Whether ``name`` is a module inside the package ``target`` heads.

    `from . import prompts` imports a sibling module, not something `__init__`
    defines, and every such line in this project would otherwise be reported.
    One false alarm is enough to teach everybody to ignore the next true one.
    """
    if target.name != "__init__.py":
        return False
    package = target.parent
    return (package / f"{name}.py").is_file() or (
        package / name / "__init__.py"
    ).is_file()


def _module_names(target: Path) -> set[str] | None:
    """Every name ``target`` might expose, or None when that cannot be known.

    Deliberately over-inclusive: it walks the whole tree, so a local variable
    inside a function counts as a name the module defines. That direction is
    the safe one. Over-collecting means a real mistake occasionally goes
    unreported; under-collecting means warning about a correct import, and one
    false alarm is enough to teach everybody to ignore the next true one.
    """
    try:
        tree = ast.parse(target.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    # A star import brings in names this cannot enumerate, so
                    # nothing about this module can be said with confidence.
                    return None
                names.add(alias.asname or alias.name.split(".")[0])

    # A module that answers for arbitrary attributes answers for every name.
    if "__getattr__" in names:
        return None
    return names


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def check(source: str, path: Path, root: Path, label: str) -> tuple[str | None, list[str]]:
    """Both checks at once: ``(refusal, warnings)``.

    Non-Python files are not checked at all -- there is nothing here that would
    know what to say about them.
    """
    if path.suffix != ".py":
        return None, []
    refusal = syntax_error(source, label)
    if refusal:
        return refusal, []
    return None, unknown_imports(source, path, root)


# --- running the project's own test command ---------------------------------

#: How long the suite may run. Shorter than the shell tool's ceiling on
#: purpose: this runs on the way out of a turn, after the model has already
#: said it is finished, and a user waiting on an answer that is already written
#: will not wait five minutes for a test run they did not ask to watch. A suite
#: slower than this wants a narrower `verify_command` -- one package, one file.
TIMEOUT = 120

#: How many times one turn may run the suite. The first run is the check; the
#: second is the model's fix being checked. A third would mean feeding back a
#: failure the model has already failed to fix once, which spends iterations
#: on the least promising thing left to try.
MAX_RUNS = 2

#: What a suggestion is worth. Each of these is the command the project's own
#: contributors run, inferred from a file that is only there because somebody
#: set the project up that way -- which is enough to *offer*, and nowhere near
#: enough to run.
SUGGESTIONS = (
    ("pytest.ini", "pytest -q"),
    ("tox.ini", "pytest -q"),
    ("Cargo.toml", "cargo test"),
    ("go.mod", "go test ./..."),
    ("package.json", "npm test"),
    ("Gemfile", "bundle exec rspec"),
)

#: How much of a failing suite goes back to the model. A failure is the one
#: tool result worth being generous with -- the point is that the model can see
#: what broke -- but a suite that fails in three hundred places is a suite whose
#: first few failures are the story, and the rest is the same story repeated.
OUTPUT_LIMIT = 4000

PASSED, FAILED, TIMED_OUT, BROKEN = "passed", "failed", "timed out", "broken"


@dataclass(frozen=True)
class Report:
    """What one run of the suite produced."""

    status: str
    command: str
    output: str = ""
    code: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == PASSED

    def summary(self) -> str:
        """One line, for the transcript."""
        if self.status == PASSED:
            return f"{self.command} passed"
        if self.status == TIMED_OUT:
            return f"{self.command} timed out after {TIMEOUT}s"
        if self.status == BROKEN:
            return f"{self.command} could not be run"
        return f"{self.command} failed (exit {self.code})"


def detect(root: Path | str) -> str:
    """A test command this project probably uses, or ``""``.

    A suggestion, never a decision. `doctor` prints it and `coder config set
    verify_command` is how it becomes real -- the safety of running anything
    here rests entirely on the command being one the user chose, and a command
    inferred from the presence of a file is not one anybody chose.
    """
    root = Path(root)
    for marker, command in SUGGESTIONS:
        if (root / marker).is_file():
            return command
    # Last, because it is the weakest signal in the list: a `tests/` directory
    # says a project has tests, not what runs them. A project carrying both a
    # Cargo.toml and a tests/ directory matched above and is a Rust project.
    if (root / "tests").is_dir() and (root / "pyproject.toml").is_file():
        return "pytest -q"
    return ""


def suite(
    command: str,
    root: Path | str,
    timeout: int = TIMEOUT,
    runner=subprocess.run,
) -> Report:
    """Run ``command`` in ``root`` and say what happened.

    Esc does not reach this. The interrupt is a flag the main thread reads
    between bytecodes, and this thread is blocked in ``waitpid`` for the whole
    run -- the same shape as the provider read before it was moved to a worker.
    That is why the timeout above is short rather than generous: the bound on
    how long a user waits here is the timeout, not the key.

    ``runner`` is injected so the tests can assert what would be run without
    running anything.
    """
    from .tools.shell import resolve_shell

    shell = resolve_shell()
    if shell is None:
        return Report(BROKEN, command, "no shell is available to run it")

    try:
        completed = runner(
            [*shell.argv, command],
            cwd=str(root),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return Report(TIMED_OUT, command)
    except OSError as exc:
        return Report(BROKEN, command, str(exc))

    # stderr last: a failing suite prints its summary to stdout and its
    # tracebacks to stderr, and the summary is the half worth reading first.
    output = "\n".join(
        part.rstrip()
        for part in (completed.stdout or "", completed.stderr or "")
        if part.strip()
    )
    if completed.returncode == 0:
        return Report(PASSED, command, output, 0)
    return Report(FAILED, command, output, completed.returncode)
