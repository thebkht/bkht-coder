"""Tools written as files, and why they are off until you say otherwise.

The registry is a curated set: every extra tool measurably costs selection
accuracy on a small model, which is the argument :mod:`~bkht.coder.tools` makes
for keeping it short. That argument is about *this* project's tools. It says
nothing about the one integration a particular workspace lives inside -- the
ticket tracker, the deploy API, the internal search -- which no shipped tool
set can contain and which the model currently has to reach through a shell.

So: ``agent/tools/<name>.py``, one tool per file, named for the file. The
module exposes either ``TOOL``, a :class:`~bkht.coder.tools.base.Tool`, or
``tool(workspace)``, a factory given the workspace so a tool can stay inside
it.

**This runs code out of the workspace, in-process, before the first turn.** It
is a larger hazard than hooks, which at least are a command the user typed into
their own config file. Cloning a repository must never be enough to run its
Python, so three things have to be true before a single line of it is imported:

1. ``agent/`` is marked as ours -- see :mod:`~bkht.coder.layout`.
2. ``agent_tools`` is turned on, and it ships off. The marker alone is not
   consent: a workspace can be marked for its skills and instructions and have
   no wish to run anything.
3. ``--no-agent-tools`` was not passed, which is the same escape hatch
   ``--no-hooks`` is, for the same moment -- you are not sure what is in there.

And when they are, every tool that loaded is named by ``coder doctor`` and by
``/tools``, with the file it came from. Nothing here is ever invisible.

A user tool may not take the name of a built-in one. Shadowing ``write_file``
would redirect a call the permission layer has already approved by name, which
is not a tool -- it is a way around the gate.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import layout
from .tools.base import Tool, ToolError, ToolResult

#: The two ways a file may offer its tool.
ATTRIBUTE = "TOOL"
FACTORY = "tool"

#: Names are typed back by the model, and are the key the permission layer
#: remembers an approval under. Anything else is refused.
def _valid(name: str) -> bool:
    return bool(name) and name.replace("_", "").isalnum() and not name[0].isdigit()


@dataclass
class Found:
    """The tools that loaded, and every file that could not be one."""

    tools: list[Tool] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def listing(self) -> list[tuple[str, str]]:
        """``(name, file)`` for every loaded tool, for `doctor` to print."""
        return [(tool.name, self.sources.get(tool.name, "")) for tool in self.tools]


def _import(path: Path):
    """The module at ``path``, under a name that cannot collide with a package.

    Imported by location rather than by name: these files are not on the path,
    are not a package, and a workspace file called ``json.py`` must not become
    the ``json`` every other module then imports.
    """
    module_name = f"_bkht_coder_agent_tool_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} is not importable")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution, because a module that imports itself or uses
    # dataclasses looks itself up in `sys.modules` while it is still running.
    sys.modules[module_name] = module
    # And no bytecode: importing these would otherwise drop a __pycache__ into
    # the user's own agent/tools/, which is a directory we were invited to read
    # and not to write.
    written = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.dont_write_bytecode = written
    return module


def _guard(tool: Tool, source: str) -> Tool:
    """The tool with its ``run`` wrapped, so a bug in it stays a tool result.

    Every other tool in this package promises the loop that it raises
    ``ToolError`` and nothing else. These are somebody's afternoon's work, and
    a traceback out of one would end the turn rather than the call -- so the
    promise is kept here on their behalf.
    """
    inner = tool.run

    def run(**arguments) -> ToolResult:
        try:
            return inner(**arguments)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{tool.name} ({source}) failed: {exc}") from None

    return replace(tool, run=run)


def _load(path: Path, source: str, workspace) -> tuple[Tool | None, str]:
    """One file as a Tool, or the reason it is not one."""
    name = path.stem.lower()
    if not _valid(name):
        return None, f"{source}: '{name}' is not a usable tool name"

    try:
        module = _import(path)
    except BaseException as exc:
        # BaseException: a module body is arbitrary code, and one that calls
        # `sys.exit` on an unset variable must cost the user their tool, not
        # their session.
        return None, f"{source}: could not be imported ({exc})"

    offered = getattr(module, ATTRIBUTE, None)
    if offered is None:
        factory = getattr(module, FACTORY, None)
        if not callable(factory):
            return None, (
                f"{source}: defines neither {ATTRIBUTE} nor {FACTORY}(workspace)"
            )
        try:
            offered = factory(workspace)
        except Exception as exc:
            return None, f"{source}: {FACTORY}(workspace) raised ({exc})"

    if not isinstance(offered, Tool):
        return None, f"{source}: {ATTRIBUTE} is not a Tool"

    # The path is the name, as everywhere else under agent/. A file that
    # declares a different one is not refused for it -- it is simply the file
    # it is in, which is the name `doctor` printed and the user can find.
    return _guard(replace(offered, name=name), source), ""


def discover(root, workspace=None, include_global: bool = True) -> Found:
    """Every tool written under ``agent/tools/``, in root order.

    The caller decides *whether* to call this -- the setting is not read here.
    Discovery that consults its own permission is discovery nobody can test
    with the permission off.
    """
    root = Path(root)
    tools: dict[str, Tool] = {}
    sources: dict[str, str] = {}
    problems: list[str] = []

    for directory in layout.surface(root, include_global=include_global).slot("tools"):
        for path in sorted(directory.glob("*.py")):
            if not path.is_file() or path.name.startswith("_"):
                continue
            source = layout.label(path, root)
            tool, problem = _load(path, source, workspace)
            if tool is None:
                problems.append(problem)
            else:
                # Later root wins, as everywhere else under agent/.
                tools[tool.name] = tool
                sources[tool.name] = source

    return Found(
        tools=[tools[name] for name in sorted(tools)], sources=sources, problems=problems
    )


def register(registry, found: Found, read_only: bool = False) -> list[str]:
    """Add what loaded to ``registry``, and report what could not be added.

    Built-ins win every collision. A user tool answering to ``write_file``
    would take calls the permission layer approved under that name, which is
    not a tool but a way around the gate.

    **None of them load into a read-only registry**, whatever they declare.
    ``mutating`` is how a built-in tool is gated, and it works there because
    this package writes those tools and knows what they do. Here it is an
    assertion by the same code it would be gating: a file that writes to disk
    and says ``mutating=False`` would be handed to a ``--plan`` session, whose
    whole promise is that it refuses every change to the workspace, and to the
    sub-agent behind ``task``, which is documented as unable to write. A
    boundary the rest of the registry keeps structurally, by leaving tools out,
    cannot be left to the good faith of the thing on the other side of it.

    The cost is a genuinely read-only tool of yours being unavailable in plan
    mode. That is the right side to err on: the alternative is that the
    guarantee holds only for tools that chose to keep it.
    """
    if read_only:
        return []

    refused = []
    for tool in found.tools:
        if tool.name in registry:
            refused.append(
                f"{found.sources.get(tool.name, tool.name)}: '{tool.name}' is "
                f"already a built-in tool, so it was not loaded"
            )
            continue
        registry.add(tool)
    return refused


def summarize(found: Found) -> str:
    """One line for the startup banner, naming what loaded and what did not."""
    parts = []
    if found.tools:
        parts.append(
            "agent tools: " + ", ".join(f"{name} ({source})" for name, source in found.listing())
        )
    if found.problems:
        parts.append(f"{len(found.problems)} tool(s) skipped: " + "; ".join(found.problems))
    return "\n".join(parts)
