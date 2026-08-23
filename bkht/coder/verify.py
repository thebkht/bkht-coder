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
"""

from __future__ import annotations

import ast
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
