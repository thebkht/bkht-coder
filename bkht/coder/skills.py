"""Skills: standing instructions that only cost context when they are used.

``AGENTS.md`` charges for its text up front, on every turn, whether or not the
turn has anything to do with it. Even at ``num_ctx=16384`` that is a real price, and
it is why the instruction budget is capped at 4000 characters.

A skill splits the cost in two. Its name and one-line description go into the
prompt -- a few dozen characters -- and the body is fetched with the ``skill``
tool only when the model decides it applies. Twenty skills cost about what one
paragraph of ``AGENTS.md`` costs, and the twenty bodies cost nothing at all
until one of them is the right one.

A skill is a directory containing ``SKILL.md``, whose frontmatter carries a
``name`` and a ``description`` -- or, under ``agent/skills/``, a single
markdown file named for the skill, which is eve's shape and the right one for
a skill that is three paragraphs and ships nothing beside them:

    ---
    name: releasing
    description: How to cut and publish a release of this project.
    ---

    Bump the version in pyproject.toml, then ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import layout
from .session import STATE_DIR
from .tools.base import ToolError
from .tools.fs import read_text

SKILL_FILE = "SKILL.md"

# Least specific first: a later root wins a name collision, so a workspace can
# override a global skill the same way its AGENTS.md overrides the global one.
# `.claude/skills` is read for compatibility with a workspace already set up for
# another agent -- not owned by us, and never written to.
GLOBAL_ROOT = STATE_DIR / "skills"
WORKSPACE_ROOTS = (Path(".claude") / "skills", Path(".bkht-coder") / "skills")

MAX_NAME = 64
MAX_DESCRIPTION = 160
# The whole listing, in the system prompt, on every turn. Held down hard: the
# point of skills is that they are cheap when unused.
MAX_BLOCK = 600
# One body, as a tool result. Larger than the listing by two orders of
# magnitude, because this one the model asked for.
MAX_BODY = 4000

TRUNCATION_NOTE = "... [truncated: this skill is too long to include in full]"


@dataclass(frozen=True)
class Skill:
    """One discovered skill: what the model sees, and where the rest of it is."""

    name: str
    description: str
    path: Path
    source: str
    #: Whether this skill has a directory of its own. A flat file does not: its
    #: neighbours are other skills, not its resources, and reaching them would
    #: quietly widen the boundary :func:`directory` exists to draw.
    resources: bool = True

    @property
    def directory(self) -> Path:
        """The skill's own folder -- the only place a resource may come from."""
        return self.path.parent


@dataclass
class Discovery:
    """What a scan found, including what it refused.

    Problems are carried rather than raised. A skill that will never load is
    the user's to fix, and they can only fix it if they are told -- a silently
    ignored file looks exactly like a skill that the model chose not to use.
    """

    skills: list[Skill] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def get(self, name: str) -> Skill | None:
        wanted = (name or "").strip().lower()
        return next((s for s in self.skills if s.name == wanted), None)

    def __bool__(self) -> bool:
        return bool(self.skills)

    def __len__(self) -> int:
        return len(self.skills)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---`` frontmatter from the body, returning recognised keys.

    Hand-rolled rather than pulled from a YAML library: the only dependency
    this project has is ``httpx``, and what is needed here is two scalar keys
    and two block styles. Unknown keys are ignored so a skill written for
    another agent still loads.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        # No closing delimiter: the whole file is body. Treating the rest as
        # frontmatter would swallow the skill and report nothing useful.
        return {}, text

    meta: dict[str, str] = {}
    index = 1
    while index < end:
        raw = lines[index]
        index += 1
        if not raw.strip() or raw.lstrip().startswith("#") or ":" not in raw:
            continue
        if raw[:1].isspace():  # a continuation of something we skipped
            continue

        key, _, value = raw.partition(":")
        key, value = key.strip().lower(), value.strip()

        if value in (">", "|", ">-", "|-"):
            block, index = _read_block(lines, index, end)
            meta[key] = block
        else:
            meta[key] = _unquote(value)

    return meta, "\n".join(lines[end + 1 :]).strip()


def _read_block(lines: list[str], index: int, end: int) -> tuple[str, int]:
    """Indented lines following a ``>`` or ``|`` marker, joined into one line.

    Folded and literal blocks are treated alike: a description is one line by
    the time it reaches the prompt either way.
    """
    collected = []
    while index < end and (not lines[index].strip() or lines[index][:1].isspace()):
        collected.append(lines[index].strip())
        index += 1
    return " ".join(part for part in collected if part), index


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _valid_name(name: str) -> bool:
    """Names are an argument the model has to type back exactly.

    So: no spaces, no path separators, short. A name the model cannot reproduce
    is a skill it cannot open.
    """
    return (
        bool(name)
        and len(name) <= MAX_NAME
        and not any(character.isspace() for character in name)
        and "/" not in name
        and "\\" not in name
        and name not in (".", "..")
    )


def _load(path: Path, source: str, resources: bool = True) -> tuple[Skill | None, str]:
    """One skill file as a Skill, or a reason it could not be one."""
    try:
        text = read_text(path)
    except (ToolError, OSError) as exc:
        return None, f"{source}: could not be read ({exc})"

    meta, _ = parse_frontmatter(text)
    name = meta.get("name", "").strip().lower()
    description = " ".join(meta.get("description", "").split())

    # The path is the name, which is the whole of eve's convention: a file at
    # skills/summarize.md is the skill `summarize` because of where it is.
    # Frontmatter still wins where it disagrees, so a skill borrowed from
    # another agent keeps the name it was written under.
    if not name and not resources:
        name = path.stem.strip().lower()

    if not name:
        return None, f"{source}: no 'name' in its frontmatter"
    if not _valid_name(name):
        return None, f"{source}: '{name}' is not a usable skill name"
    if not description:
        return None, f"{source}: no 'description' in its frontmatter"

    if len(description) > MAX_DESCRIPTION:
        description = description[: MAX_DESCRIPTION - 1].rstrip() + "…"

    return Skill(
        name=name, description=description, path=path, source=source, resources=resources
    ), ""


def _label(path: Path, root: Path) -> str:
    """How a skill is named in listings: short, but traceable to a file."""
    for base, prefix in ((root, ""), (Path.home(), "~/")):
        try:
            return prefix + str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def discover(root, include_global: bool = True) -> Discovery:
    """Every skill that applies to ``root``, least specific root first.

    One level deep, no recursion, in the same handful of places every time --
    the same rule the instruction files follow. A discovery mechanism that has
    to be explained is one the user will not trust.
    """
    root = Path(root)
    surface = layout.surface(root, include_global=include_global)

    directories = []
    if include_global:
        directories.append(GLOBAL_ROOT)
    directories.extend(surface.slot("skills", layout.GLOBAL))
    directories.extend(root / relative for relative in WORKSPACE_ROOTS)
    directories.extend(surface.slot("skills", layout.WORKSPACE))

    return scan(directories, root)


def scan(directories, root) -> Discovery:
    """Every skill in the given directories, least specific first.

    Split out of :func:`discover` for the one other caller that has skills but
    not the layered roots they usually come from: a subagent, whose ``skills/``
    is its own and is not layered over anything.
    """
    found: dict[str, Skill] = {}
    problems: list[str] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            # A directory holding a SKILL.md, or a markdown file that is the
            # skill. Anything else is somebody's notes, and is not ours to
            # complain about.
            if child.is_dir():
                path, resources = child / SKILL_FILE, True
                if not path.is_file():
                    continue
            elif child.suffix == ".md" and child.is_file():
                path, resources = child, False
            else:
                continue

            skill, problem = _load(path, _label(path, root), resources)
            if skill is None:
                problems.append(problem)
            else:
                # Later root wins: the more specific skill is the one meant.
                found[skill.name] = skill

    return Discovery(skills=sorted(found.values(), key=lambda s: s.name), problems=problems)


HEADER = """\
# Skills

Reusable instructions the user has written for this workspace. Each line is a
name and what it covers. When one applies to the task, call `skill` with its
name to read it *before* acting, and follow what it says.
"""


def render(discovery: Discovery, budget: int = MAX_BLOCK) -> str:
    """The skills section of the system prompt, or nothing when there are none.

    Over budget, the listing is cut and the cut is stated. A prompt that quietly
    omits half the skills teaches the model that a name it was never shown does
    not exist.
    """
    if not discovery.skills:
        return ""

    lines = []
    spent = 0
    for skill in discovery.skills:
        entry = f"- {skill.name} — {skill.description}"
        if spent + len(entry) > budget and lines:
            dropped = len(discovery.skills) - len(lines)
            lines.append(f"[{dropped} more skill(s) omitted: the listing is full]")
            break
        lines.append(entry)
        spent += len(entry) + 1

    return HEADER + "\n" + "\n".join(lines) + "\n"


def body(skill: Skill, limit: int = MAX_BODY) -> str:
    """A skill's instructions, without its frontmatter, fitted to ``limit``."""
    _, text = parse_frontmatter(read_text(skill.path))
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n" + TRUNCATION_NOTE


def summarize(discovery: Discovery) -> str:
    """One line for the startup banner, naming what loaded and what did not."""
    parts = []
    if discovery.skills:
        parts.append("skills: " + ", ".join(s.name for s in discovery.skills))
    if discovery.problems:
        parts.append(f"{len(discovery.problems)} skill(s) skipped: " + "; ".join(discovery.problems))
    return "\n".join(parts)
