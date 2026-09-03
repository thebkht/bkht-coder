"""Settings that survive a restart.

Every knob in coder is a flag with a built-in default, which is right for a one
-off run and wrong for a preference. Somebody who wants the 7b model, or a
smaller window, or plan mode by default, should say so once.

There are two files, and the same global-then-workspace shadowing that skills
and slash commands already use:

    ~/.bkht-coder/config.json     personal defaults, everywhere
    .bkht-coder/config.json       this project's defaults, committed or not

Resolution runs lowest to highest -- built-in default, global file, workspace
file, then the flag typed on the command line. A flag always wins, because the
flag is the thing the user is holding.

``provider`` names the backend, and the three settings that mean something
different to each backend -- the model tag, the server URL and the window size
-- follow it. Switching to ``claude`` without also naming a model would
otherwise ask Anthropic for a model tag that only a local Ollama has ever heard
of, so any of those three the user has not set themselves is taken from the
provider's own defaults instead of from a constant.

Reading never raises. A config file that cannot be parsed leaves the session on
its defaults and hands back a sentence to print -- refusing to start would take
away the session that could fix the file. Writing does raise, because a write is
something the user just asked for and a silent failure would be a lie.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .agent import MAX_ITERATIONS
from .permissions import ASK, AUTO, MODES, PLAN
from .provider import settle
from .provider import (
    BACKENDS,
    DEFAULTS,
    DEFAULT_NUM_CTX,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    MIN_USEFUL_NUM_CTX,
)
from .session import STATE_DIR

GLOBAL_PATH = STATE_DIR / "config.json"
WORKSPACE_NAME = Path(".bkht-coder") / "config.json"

#: Where a value came from. Also the two scopes a write can target.
DEFAULT = "default"
GLOBAL = "global"
WORKSPACE = "workspace"
SCOPES = (GLOBAL, WORKSPACE)

#: What the program defaults to, which is the default *backend's* model and
#: server rather than any one server's. Read from the same table
#: ``_follow_provider`` uses, so the declared default and the effective one
#: cannot drift apart -- before this they could, and `coder config` would show
#: a model the session was not running.
DEFAULT_MODEL = DEFAULTS[DEFAULT_PROVIDER]["model"]
DEFAULT_HOST = DEFAULTS[DEFAULT_PROVIDER]["host"]

#: The highest temperature Ollama accepts as meaningful. Above this the model
#: is not sampling, it is guessing, and every tool call it emits is malformed.
MAX_TEMPERATURE = 2.0


class ConfigError(ValueError):
    """A key that does not exist, or a value that key cannot hold."""


@dataclass(frozen=True)
class Field:
    """One setting: how to read it, what it means, and whether it can change."""

    name: str
    kind: str  # str | int | float | bool
    default: object
    description: str
    #: Whether a running session can adopt a new value. The ones that cannot are
    #: baked into the provider or the system prompt before the first turn.
    live: bool = True


FIELDS: tuple[Field, ...] = (
    Field("provider", "str", DEFAULT_PROVIDER, f"Model backend: {', '.join(sorted(BACKENDS))}.", live=False),
    Field("model", "str", DEFAULT_MODEL, "Model to run."),
    Field("host", "str", DEFAULT_HOST, "Model server URL."),
    Field("num_ctx", "int", DEFAULT_NUM_CTX, "Context window to plan for."),
    Field("temperature", "float", DEFAULT_TEMPERATURE, "Sampling temperature; low keeps tool calls valid."),
    Field("mode", "str", ASK, f"Permission mode: {', '.join(MODES)}."),
    Field("scout", "bool", True, "Search the workspace before each task."),
    Field("max_iterations", "int", MAX_ITERATIONS, "Cap on loop iterations per task."),
    Field("instructions", "bool", True, "Read AGENTS.md and CLAUDE.md.", live=False),
    Field("skills", "bool", True, "Load skills, and offer the skill tool.", live=False),
    # Two tools, two switches. The registry's standing argument -- every extra
    # tool costs selection accuracy on a small model -- applies to these as
    # much as to anything else, so both are turned off from one line.
    Field("planning", "bool", True, "Offer the plan tool.", live=False),
    Field("delegation", "bool", True, "Offer the task tool, which runs a sub-agent.", live=False),
    # The one setting that governs a request leaving this machine. Not live:
    # the check is started once, before the first turn.
    Field("update_check", "bool", True, "Check for a new release once a day.", live=False),
)

BY_NAME = {f.name: f for f in FIELDS}

TRUTHY = ("true", "yes", "on", "1")
FALSEY = ("false", "no", "off", "0")


def _coerce(spec: Field, raw: object) -> object:
    """``raw`` as the type ``spec`` declares, from JSON or from a command line.

    Both callers land here: a value read out of a file arrives already typed,
    and one typed at a prompt arrives as a string. Booleans are checked before
    integers on purpose -- ``bool`` is a subclass of ``int``, so ``scout = true``
    would otherwise be accepted as ``num_ctx``.
    """
    if spec.kind == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in TRUTHY:
            return True
        if text in FALSEY:
            return False
        raise ConfigError(
            f"{spec.name} is a true/false setting; got {raw!r}. "
            f"Use one of {', '.join(TRUTHY)} or {', '.join(FALSEY)}."
        )

    if spec.kind in ("int", "float"):
        if isinstance(raw, bool):
            raise ConfigError(f"{spec.name} is a number, not true/false.")
        try:
            return int(str(raw).strip()) if spec.kind == "int" else float(str(raw).strip())
        except ValueError:
            raise ConfigError(f"{spec.name} must be a number; got {raw!r}.") from None

    if not isinstance(raw, str):
        raise ConfigError(f"{spec.name} must be text; got {raw!r}.")
    text = raw.strip()
    if not text:
        raise ConfigError(f"{spec.name} cannot be empty.")
    return text


def _validate(spec: Field, value: object) -> object:
    """Reject a well-typed value that would still break the session.

    The point of doing it here rather than at startup is that the error lands on
    the keystroke that caused it, naming what would have been accepted -- rather
    than three days later, on a session that will not start.
    """
    if spec.name == "provider" and value not in BACKENDS:
        raise ConfigError(
            f"unknown provider {value!r}. Available: {', '.join(sorted(BACKENDS))}."
        )
    if spec.name == "mode" and value not in MODES:
        raise ConfigError(f"unknown mode {value!r}. Expected one of {', '.join(MODES)}.")
    if spec.name == "num_ctx" and value < MIN_USEFUL_NUM_CTX:
        raise ConfigError(
            f"num_ctx of {value} is too small to be useful; Ollama's own default "
            f"of 2048 silently truncates the prompt. Use at least {MIN_USEFUL_NUM_CTX}."
        )
    if spec.name == "temperature" and not 0.0 <= value <= MAX_TEMPERATURE:
        raise ConfigError(f"temperature must be between 0 and {MAX_TEMPERATURE}; got {value}.")
    if spec.name == "max_iterations" and value < 1:
        raise ConfigError(f"max_iterations must be at least 1; got {value}.")
    return value


def parse(key: str, raw: object) -> object:
    """One key and one value, coerced and validated, or ``ConfigError``."""
    spec = BY_NAME.get(key)
    if spec is None:
        raise ConfigError(f"unknown setting {key!r}. Known: {', '.join(BY_NAME)}.")
    return _validate(spec, _coerce(spec, raw))


def format_value(value: object) -> str:
    """A value as it would be typed back in, so a listing round-trips."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def path_for(scope: str, root=None) -> Path:
    """The file a scope is stored in.

    ``GLOBAL_PATH`` is read through the module rather than captured, so the
    suite can move the personal file somewhere harmless.
    """
    if scope == GLOBAL:
        return GLOBAL_PATH
    if scope == WORKSPACE:
        if root is None:
            raise ConfigError("the workspace config needs a workspace to live in.")
        return Path(root) / WORKSPACE_NAME
    raise ConfigError(f"unknown scope {scope!r}. Expected one of {', '.join(SCOPES)}.")


def _read(path: Path) -> tuple[dict, str]:
    """The raw object stored at ``path``, and a reason if there was not one."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, ""
    except (OSError, ValueError) as exc:
        return {}, f"could not read {path}: {exc}; continuing with the defaults"
    if not isinstance(raw, dict):
        return {}, f"{path} is not a JSON object; continuing with the defaults"
    return raw, ""


@dataclass
class Settings:
    """Every setting, resolved, with the layer each one came from."""

    values: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    #: What could not be read, ready to print. Empty when everything loaded.
    error: str = ""
    #: A sentence about a decision this resolution made on the user's behalf --
    #: today, only falling back from an unserved default. Separate from
    #: ``error`` because nothing went wrong: the session is starting, and this
    #: says on what.
    notice: str = ""

    def __getattr__(self, name: str):
        # Only reached for attributes the dataclass does not define, so the
        # three fields above still resolve normally.
        try:
            return self.__dict__["values"][name]
        except KeyError:
            raise AttributeError(name) from None

    def source(self, key: str) -> str:
        return self.sources.get(key, DEFAULT)

    def listing(self) -> list[tuple[str, str, str]]:
        """``(key, value, source)`` per setting, in declaration order."""
        return [(f.name, format_value(self.values[f.name]), self.source(f.name)) for f in FIELDS]

    def payload(self) -> dict:
        """The listing as JSON-shaped data, for ``--json``."""
        return {
            f.name: {"value": self.values[f.name], "source": self.source(f.name)}
            for f in FIELDS
        }

    def apply(self, args) -> None:
        """Fill in every flag the user did not type.

        The flags parse with ``default=None`` precisely so that this can tell
        "the user asked for the default" from "argparse filled it in" -- without
        that distinction a config file could never win over a flag that is
        always present.
        """
        # `--provider` is resolved first, and it moves the three settings whose
        # meaning depends on it. Otherwise `--provider claude-code` would run
        # Claude Code against the model tag the config file picked for Ollama.
        typed = getattr(args, "provider", "missing")
        if typed is None:
            args.provider = self.values["provider"]
        chosen = getattr(args, "provider", self.values["provider"])

        # Nobody named this backend -- not on the command line, not in either
        # config file. That is the one case where finding nothing at the other
        # end is worth answering rather than failing on: a machine that has only
        # ever run Ollama should not have its first session fail against a
        # server it was never told to start.
        if typed is None and self.source("provider") == DEFAULT:
            chosen, self.notice = settle(chosen, self.values["host"])
            args.provider = chosen
        follows = {
            key: value
            for key, value in DEFAULTS.get(chosen, {}).items()
            if self.source(key) == DEFAULT
        }

        for name in ("model", "host", "num_ctx", "temperature", "max_iterations"):
            if getattr(args, name, "missing") is None:
                setattr(args, name, follows.get(name, self.values[name]))

        # The negative switches carry only one bit: present means off, absent
        # means "whatever was configured". So they are filled from the inverse.
        for name, dest in (
            ("scout", "no_scout"), ("instructions", "no_instructions"), ("skills", "no_skills"),
            ("planning", "no_planning"), ("delegation", "no_delegation"),
        ):
            if getattr(args, dest, "missing") is None:
                setattr(args, dest, not self.values[name])

        # `--auto` and `--plan` are two switches over one setting. Neither given
        # means the configured mode; either given is left exactly as typed, so
        # `--auto --plan` still contradicts itself out loud.
        if not getattr(args, "auto", None) and not getattr(args, "plan", None):
            mode = self.values["mode"]
            if hasattr(args, "auto"):
                args.auto = mode == AUTO
            if hasattr(args, "plan"):
                args.plan = mode == PLAN


def load(root=None) -> Settings:
    """Defaults, then the global file, then the workspace file.

    Never raises. An unreadable file at either layer is reported through
    ``error`` and skipped, so a typo in a committed config cannot stop a session
    from starting.
    """
    settings = Settings(values={f.name: f.default for f in FIELDS})
    problems: list[str] = []

    layers = [(GLOBAL, GLOBAL_PATH)]
    if root is not None:
        layers.append((WORKSPACE, Path(root) / WORKSPACE_NAME))

    for scope, path in layers:
        stored, problem = _read(path)
        if problem:
            problems.append(problem)
        for key, raw in stored.items():
            try:
                settings.values[key] = parse(key, raw)
            except ConfigError as exc:
                problems.append(f"{path}: {exc}")
                continue
            settings.sources[key] = scope

    _follow_provider(settings)
    settings.error = "; ".join(problems)
    return settings


def _follow_provider(settings: Settings) -> None:
    """Re-default the settings whose meaning depends on the backend.

    Only the ones still sitting on a built-in default are touched. A value the
    user wrote down is theirs -- if they have pinned a model, switching provider
    should fail loudly on that model rather than quietly run a different one.
    """
    for key, value in DEFAULTS.get(settings.values["provider"], {}).items():
        if settings.source(key) == DEFAULT:
            settings.values[key] = value


def _write(path: Path, stored: dict) -> None:
    """Replace ``path`` with ``stored``, atomically.

    Written to a sibling and renamed for the same reason the permission store
    is: a crash mid-write would otherwise leave a truncated file, and the next
    session would start having silently forgotten every setting.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        raise ConfigError(f"could not write {path}: {exc}") from exc


def set_value(key: str, raw: object, scope: str = GLOBAL, root=None) -> object:
    """Store one setting in one scope, and return what was stored.

    Everything else in the file is carried across untouched -- including keys
    this version does not know about, so a config written by a newer coder is
    not quietly emptied by an older one.
    """
    value = parse(key, raw)
    path = path_for(scope, root)
    stored, problem = _read(path)
    if problem:
        raise ConfigError(problem.split(";")[0])
    stored[key] = value
    _write(path, stored)
    return value


def unset(key: str, scope: str = GLOBAL, root=None) -> bool:
    """Remove one setting from one scope. False when it was not set there."""
    if key not in BY_NAME:
        raise ConfigError(f"unknown setting {key!r}. Known: {', '.join(BY_NAME)}.")
    path = path_for(scope, root)
    stored, problem = _read(path)
    if problem:
        raise ConfigError(problem.split(";")[0])
    if key not in stored:
        return False
    del stored[key]
    _write(path, stored)
    return True


def render(settings: Settings) -> str:
    """The table ``coder config`` and ``/config`` both print."""
    width = max(len(f.name) for f in FIELDS)
    return "\n".join(
        f"  {key:<{width}}  {value:<24}  {source}"
        for key, value, source in settings.listing()
    )


USAGE = """\
Usage: coder config [list]
       coder config get <key>
       coder config set <key> <value> [--workspace]
       coder config unset <key> [--workspace]
       coder config path"""


def run(args) -> int:
    """Execute ``coder config``. Returns the process exit status.

    Every failure is the user's own typing -- an unknown key, a value the key
    cannot hold -- so it goes to stderr with a message that names what would
    have worked, and exits non-zero for anything reading the status.
    """
    root = Path(args.cwd).expanduser().resolve()
    scope = WORKSPACE if args.workspace else GLOBAL
    action = args.action or "list"
    rest = list(args.rest)

    try:
        if action == "list":
            return _list(root, as_json=args.json)
        if action == "path":
            return _paths(root)
        if action == "get":
            return _get(root, rest)
        if action == "set":
            return _set(root, scope, rest)
        if action == "unset":
            return _unset(root, scope, rest)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"error: unknown action {action!r}.\n{USAGE}", file=sys.stderr)
    return 2


def _announce(settings: Settings) -> None:
    if settings.error:
        print(f"warning: {settings.error}", file=sys.stderr)


def _list(root: Path, as_json: bool = False) -> int:
    settings = load(root)
    _announce(settings)
    print(json.dumps(settings.payload(), indent=2) if as_json else render(settings))
    return 0


def _paths(root: Path) -> int:
    for scope in SCOPES:
        path = path_for(scope, root)
        print(f"  {scope:<10}  {path}{'' if path.exists() else '  (not written yet)'}")
    return 0


def _get(root: Path, rest: list[str]) -> int:
    if len(rest) != 1:
        raise ConfigError(f"get takes one key.\n{USAGE}")
    key = rest[0]
    if key not in BY_NAME:
        raise ConfigError(f"unknown setting {key!r}. Known: {', '.join(BY_NAME)}.")
    settings = load(root)
    _announce(settings)
    print(format_value(settings.values[key]))
    return 0


def _set(root: Path, scope: str, rest: list[str]) -> int:
    if len(rest) < 2:
        raise ConfigError(f"set takes a key and a value.\n{USAGE}")
    # Joined rather than indexed: a value with a space in it -- a host with a
    # path, a model name someone quoted oddly -- should not be silently halved.
    key, value = rest[0], " ".join(rest[1:])
    stored = set_value(key, value, scope=scope, root=root)
    print(f"{key} = {format_value(stored)}  ({scope}: {path_for(scope, root)})")
    return 0


def _unset(root: Path, scope: str, rest: list[str]) -> int:
    if len(rest) != 1:
        raise ConfigError(f"unset takes one key.\n{USAGE}")
    key = rest[0]
    if unset(key, scope=scope, root=root):
        print(f"Unset {key} in the {scope} config. Now {format_value(load(root).values[key])}.")
    else:
        print(f"{key} was not set in the {scope} config.")
    return 0
