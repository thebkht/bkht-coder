"""Reading trajectories out of whatever wrote them.

Five sources, one shape. A :class:`Trajectory` is a system prompt and a list of
messages in *coder's* schema -- which means an assistant tool call is the
literal JSON string the runtime parser reads back, not a native ``tool_calls``
field. Everything downstream works on that and never learns where it came from.

The three foreign sources need their tools translated, and the translation is
not cosmetic. A model trained on `Read(file_path=...)` learns to emit a call
coder has no tool for; trained on the mapped `read_file(path=...)` it learns
the one that exists. Paths are relativized against the session's working
directory for the same reason -- coder's tools take workspace-relative paths,
and absolute ones from somebody else's machine are not merely useless, they are
a pattern worth unlearning.

A call with no coder equivalent truncates the trajectory rather than being
dropped from the middle of it. A conversation with a hole in it teaches a
non-sequitur: an assistant turn that answers a question nobody asked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..session import SESSION_DIR
from ..transcripts import CLAUDE_DIR, CODEX_DIR

#: Every source name that can be asked for, in the order a mixed build reads
#: them: the ones already speaking coder's protocol first.
CODER = "coder"
CLAUDE = "claude"
CODEX = "codex"
CHAT = "chat"
SOURCES = (CODER, CLAUDE, CODEX, CHAT)

#: The turn outcomes worth imitating. Everything else -- the iteration cap, the
#: retry cap, a turn that went round in circles, a denied call -- is a turn the
#: model should not be taught to reproduce.
GOOD_STOPS = frozenset({"answered"})

def _steps(arguments: dict) -> dict:
    """A plan from either agent's spelling, as coder's `plan` takes it.

    Codex sends ``plan: [{step, status}]`` and Claude Code sends
    ``todos: [{content, status}]``; coder's tool takes a list of strings and
    tracks doneness itself. The statuses are dropped rather than translated,
    because coder's plan marks one step done per call and these mark the whole
    list every time -- a faithful translation would be a call the tool rejects.
    """
    raw = arguments.get("plan") or arguments.get("todos") or []
    steps = []
    for entry in raw if isinstance(raw, list) else []:
        if isinstance(entry, str):
            steps.append(entry)
        elif isinstance(entry, dict):
            text = entry.get("step") or entry.get("content") or entry.get("activeForm")
            if text:
                steps.append(str(text))
    return {"steps": steps} if steps else {}


def _instruction(arguments: dict) -> dict:
    """A delegated task as coder's `task` takes it: one written question.

    Claude Code splits the brief across ``prompt`` and a short ``description``,
    and names an agent type coder has no equivalent for -- it has exactly one
    sub-agent. Only the brief survives.
    """
    text = str(arguments.get("prompt") or arguments.get("instruction") or "").strip()
    return {"instruction": text} if text else {}


def _skill(arguments: dict) -> dict:
    name = str(arguments.get("skill") or arguments.get("name") or "").strip()
    return {"name": name} if name else {}


#: Foreign tool name to coder's, and how each one's arguments are spelled.
#: ``None`` for an argument means "drop it": coder's `glob` takes no path, and
#: passing one through would train the model to send an argument that fails
#: schema validation on every call. A callable replaces the renaming entirely,
#: for the calls whose shape differs rather than only their spelling.
TOOLS: dict[str, tuple[str, dict[str, str | None] | object]] = {
    # Claude Code
    "Read": ("read_file", {"file_path": "path", "offset": "offset", "limit": "limit"}),
    "Write": ("write_file", {"file_path": "path", "content": "content"}),
    "Edit": (
        "edit_file",
        {
            "file_path": "path",
            "old_string": "old_string",
            "new_string": "new_string",
            "replace_all": "replace_all",
        },
    ),
    "Bash": ("bash", {"command": "command", "timeout": "timeout"}),
    "Glob": ("glob", {"pattern": "pattern", "path": None}),
    "Grep": ("grep", {"pattern": "pattern", "path": "path", "glob": "glob"}),
    "LS": ("list_files", {"path": "path"}),
    "Task": ("task", _instruction),
    "Agent": ("task", _instruction),
    "TodoWrite": ("plan", _steps),
    "Skill": ("skill", _skill),
    # Codex
    "exec_command": ("bash", {"cmd": "command", "workdir": None, "timeout_ms": None}),
    "shell": ("bash", {"command": "command", "workdir": None}),
    "local_shell": ("bash", {"command": "command", "workdir": None}),
    "update_plan": ("plan", _steps),
}

#: Openings that mean a harness wrote this message, not a person. Both agents
#: inject context as ordinary user turns -- an AGENTS.md dump, an environment
#: block, a note that the last turn was interrupted -- and coder sends none of
#: it. Training on them teaches the model to expect a preamble that will never
#: arrive, and to read the first real instruction as the third thing it is told.
INJECTED = (
    "<turn_aborted>",
    "<environment_context>",
    "<user_instructions>",
    "# AGENTS.md instructions",
    "<INSTRUCTIONS>",
    "<system-reminder>",
    "Caveat: The messages below",
)

#: Arguments every mapping silently drops. Named once so the tables above stay
#: about the arguments that carry meaning.
IGNORED = frozenset({"description", "caller", "max_output_tokens", "yield_time_ms"})


@dataclass
class Trajectory:
    """One conversation, in coder's schema, ready to be rendered.

    ``system`` is empty for a foreign source: those conversations happened under
    another agent's prompt, and the one that matters is the one coder will
    actually send. :mod:`.render` supplies it.
    """

    source: str
    origin: str
    messages: list[dict] = field(default_factory=list)
    system: str = ""
    provider: str = ""
    cwd: str = ""
    #: How each turn in this trajectory ended, in order. Empty for a source that
    #: does not record it, which is every source but coder's own.
    outcomes: list[str] = field(default_factory=list)
    #: Why this trajectory stops where it does, when it was cut short.
    truncated: str = ""

    def __len__(self) -> int:
        return len(self.messages)

    @property
    def turns(self) -> int:
        """How many things the user asked for."""
        return sum(1 for m in self.messages if m.get("role") == "user")


# --- helpers ------------------------------------------------------------------


def records(path: Path) -> Iterator[dict]:
    """Every JSON object in a JSONL file, skipping what will not parse.

    These files belong to other programs and to killed processes, so a line that
    does not decode is ordinary rather than exceptional.
    """
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def call_json(name: str, arguments: dict) -> str:
    """A tool call as the system prompt says to emit one.

    The single place the wire format is written down for training. It matches
    ``prompts.TOOL_PROTOCOL`` exactly, and :mod:`.render` proves the match by
    parsing every one of these back through the runtime parser.
    """
    return json.dumps({"name": name, "arguments": arguments})


def relative(value: str, cwd: str) -> str:
    """An absolute path under ``cwd`` as coder's tools would be given it.

    Anything outside the workspace, and anything that is not a path at all, is
    returned untouched: guessing is worse than leaving a string alone.
    """
    if not cwd or not isinstance(value, str) or not value.startswith("/"):
        return value
    try:
        return str(Path(value).relative_to(Path(cwd)))
    except ValueError:
        return value


def translate(name: str, arguments: dict, cwd: str = "") -> tuple[str, dict] | None:
    """One foreign call as a coder call, or None when there is no equivalent."""
    mapping = TOOLS.get(name)
    if mapping is None:
        return None
    coder_name, spelling = mapping

    # A call whose shape differs rather than only its spelling. An empty result
    # means the arguments carried nothing coder's tool could be given, which is
    # a call to drop rather than one to send empty.
    if callable(spelling):
        rebuilt = spelling(arguments or {})
        return (coder_name, rebuilt) if rebuilt else None

    translated: dict = {}
    for key, value in (arguments or {}).items():
        if key in IGNORED:
            continue
        renamed = spelling.get(key, key)
        if renamed is None:
            continue
        if renamed in ("path", "file_path"):
            value = relative(value, cwd)
        translated[renamed] = value

    # Codex's `shell` takes an argv list where coder's `bash` takes a line.
    if coder_name == "bash" and isinstance(translated.get("command"), list):
        translated["command"] = " ".join(str(part) for part in translated["command"])
    return coder_name, translated


def injected(text: str) -> bool:
    """Whether this user turn was written by a harness rather than a person."""
    stripped = text.lstrip()
    return any(stripped.startswith(marker) for marker in INJECTED)


def pair(messages: list[dict]) -> list[dict]:
    """Calls and their results interleaved, one call per assistant turn.

    Both foreign agents call tools in parallel: several ``tool_use`` blocks in
    one assistant message, and every result afterwards. Coder's protocol is the
    opposite and says so in the system prompt -- one call per reply, wait for
    the result before the next. Replaying the foreign order would train the
    model to do exactly what the prompt forbids, and a model emitting four calls
    at once against this loop gets one executed and three ignored.

    So each result is moved up behind the call it answers, by id. A call whose
    result never arrives ends the trajectory: everything after it is a reply to
    evidence that is not there.
    """
    results = {
        message["call_id"]: message
        for message in messages
        if message.get("role") == "tool" and message.get("call_id")
    }

    ordered: list[dict] = []
    for message in messages:
        if message.get("role") == "tool":
            continue
        if message.get("role") == "assistant" and message.get("call_id"):
            answer = results.get(message["call_id"])
            if answer is None:
                break
            ordered.append(message)
            ordered.append(answer)
            continue
        ordered.append(message)
    return ordered


#: Codex wraps every command result in a header of its own -- a chunk id, a
#: wall time, an exit line, a token count -- and then `Output:`. Coder's `bash`
#: returns what the command printed. Everything above the marker is another
#: harness's furniture, and a model trained to expect it is a model reading for
#: a line that never comes.
CODEX_OUTPUT = "\nOutput:\n"
#: Either of these near the top means the framing above is Codex's, not the
#: command's. It writes the chunk id first for some calls and echoes the whole
#: shell invocation first for others.
CODEX_HEADERS = ("Chunk ID:", "Command:")


def unwrap(text: str) -> str:
    """One tool result with a foreign harness's framing removed.

    Bounded to the head of the string on purpose: `Output:` is an ordinary word
    that appears in real command output, and splitting on a later one would cut
    away the first half of a result that was never wrapped at all.
    """
    head = text[:400]
    if not any(head.startswith(marker) for marker in CODEX_HEADERS):
        return text
    if CODEX_OUTPUT not in head:
        return text
    return text.split(CODEX_OUTPUT, 1)[1]


def drop_commentary(messages: list[dict]) -> list[dict]:
    """Assistant prose that only introduces the call after it, removed.

    The protocol in the system prompt is explicit: a reply that calls a tool is
    the JSON object and nothing else. Codex narrates before acting -- its own
    messages are marked `commentary` -- and Claude Code often does the same. Fed
    in as-is, that is a worked example of the one thing the prompt forbids, in
    the position the model imitates most readily.

    The rule is that an assistant turn is either a call or the answer. So any
    assistant prose with another assistant turn behind it is commentary, and
    only the last one -- the answer, or the call it was narrating -- is kept.
    Prose that follows a tool result and is followed by a user turn is an
    answer, and answers are the thing being learned.

    Walked backwards, because what matters is what *survives* after this
    message, not what was written after it. Two paragraphs before a call is the
    ordinary shape, and a forward pass drops the first, leaves the second, and
    reports itself finished having created exactly the adjacency it removes.
    """
    kept: list[dict] = []
    for message in reversed(messages):
        following = kept[-1] if kept else None
        if (
            message.get("role") == "assistant"
            and not message["content"].lstrip().startswith("{")
            and following is not None
            and following.get("role") == "assistant"
        ):
            continue
        kept.append(message)
    kept.reverse()
    return kept


def text_of(content) -> str:
    """The readable text of a content field, whatever shape it arrived in."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(str(block.get("text") or block.get("content") or ""))
    return "\n".join(part for part in parts if part).strip()


# --- coder's own sessions -----------------------------------------------------


def from_coder(path: Path) -> Iterator[Trajectory]:
    """Coder's own transcripts, which are already in the target schema.

    One trajectory per segment rather than per file. A `clear` starts a new
    conversation in the same file, and a `prompt` record means the tool set
    moved -- both make the messages after them a different example from the ones
    before, and running them together would train the model to answer a question
    from a conversation it never saw.

    A turn that ended badly truncates the trajectory at the end of the last turn
    that did not. Dropping the whole session for one bad turn throws away the
    good prefix; keeping the bad turn teaches the model to reproduce it.
    """
    header: dict = {}
    system = ""
    messages: list[dict] = []
    outcomes: list[str] = []
    good_upto = 0  # messages as of the end of the last turn that answered

    def flush() -> Trajectory | None:
        if not messages:
            return None
        kept = messages[:good_upto] if good_upto else []
        return Trajectory(
            source=CODER,
            origin=f"{path.stem}#{len(outcomes)}",
            messages=kept,
            system=system,
            provider=header.get("provider", ""),
            cwd=header.get("cwd", ""),
            outcomes=list(outcomes),
            truncated="" if good_upto == len(messages) else "turn did not answer",
        )

    for record in records(path):
        kind = record.get("type")
        if kind == "session":
            header = record
        elif kind == "prompt":
            if messages and (made := flush()):
                yield made
            messages, outcomes, good_upto = [], [], 0
            system = record.get("system") or ""
        elif kind == "clear":
            if made := flush():
                yield made
            messages, outcomes, good_upto = [], [], 0
        elif kind == "outcome":
            stopped = record.get("stopped") or ""
            outcomes.append(stopped)
            if stopped in GOOD_STOPS:
                good_upto = len(messages)
        elif kind == "message":
            message = {k: v for k, v in record.items() if k not in ("type", "images")}
            if message.get("role") and message.get("content"):
                messages.append(message)

    if made := flush():
        yield made


# --- Claude Code --------------------------------------------------------------


def from_claude(path: Path) -> Iterator[Trajectory]:
    """One Claude Code session, with its tools translated.

    The file is a tree rather than a list -- a rewind starts a sibling branch --
    so only the live branch is read, which is what ``transcripts._claude_branch``
    already works out for the session listing.

    Thinking blocks are dropped. Coder's protocol has nowhere to put them, and a
    model trained to emit reasoning that the loop then feeds back as a tool call
    is a model trained to produce something unparseable.
    """
    from ..transcripts import _claude_branch

    every = list(records(path))
    if not every:
        return

    cwd = next((r.get("cwd") for r in every if r.get("cwd")), "") or ""
    messages: list[dict] = []
    truncated = ""
    # Which coder tool answered each Claude tool_use id, so its result can be
    # labelled with the name coder would have used.
    called: dict[str, str] = {}

    for record in _claude_branch(every):
        kind = record.get("type")
        if kind not in ("user", "assistant"):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")

        if kind == "user" and isinstance(content, str):
            if not injected(content):
                messages.append({"role": "user", "content": content})
            continue

        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "text" and (text := str(block.get("text") or "").strip()):
                messages.append({"role": kind, "content": text})
            elif block_type == "tool_use":
                mapped = translate(str(block.get("name") or ""), block.get("input") or {}, cwd)
                if mapped is None:
                    truncated = f"no equivalent for {block.get('name')}"
                    break
                name, arguments = mapped
                identifier = str(block.get("id"))
                called[identifier] = name
                messages.append(
                    {
                        "role": "assistant",
                        "content": call_json(name, arguments),
                        "call_id": identifier,
                    }
                )
            elif block_type == "tool_result":
                identifier = str(block.get("tool_use_id"))
                name = called.get(identifier)
                if name is None:
                    continue
                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": text_of(block.get("content")),
                        "call_id": identifier,
                    }
                )
        if truncated:
            break

    if trimmed := _ending_in_an_answer(drop_commentary(pair(messages))):
        yield Trajectory(
            source=CLAUDE, origin=path.stem, messages=trimmed, cwd=cwd,
            truncated=truncated,
        )


# --- Codex --------------------------------------------------------------------


def from_codex(path: Path) -> Iterator[Trajectory]:
    """One Codex rollout, read from the response items rather than the events.

    ``transcripts.read_codex`` reads ``event_msg`` because that is the view
    written for a person. The calls are only in ``response_item``, so this reads
    that one instead -- the two are the same turn told twice, and mixing them
    would double every message.
    """
    cwd = ""
    messages: list[dict] = []
    truncated = ""
    called: dict[str, str] = {}

    for record in records(path):
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if record.get("type") == "session_meta":
            cwd = payload.get("cwd") or cwd
            continue
        if record.get("type") != "response_item":
            continue

        kind = payload.get("type")
        if kind == "message":
            role = payload.get("role")
            # `developer` is Codex's own instruction block, and the first user
            # message carries AGENTS.md and an environment dump. Neither is
            # something a person typed, and both would train the model to expect
            # a preamble coder never sends.
            if role not in ("user", "assistant"):
                continue
            text = text_of(payload.get("content"))
            if text and not (role == "user" and injected(text)):
                messages.append({"role": role, "content": text})
        elif kind == "function_call":
            try:
                arguments = json.loads(payload.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            mapped = translate(str(payload.get("name") or ""), arguments, cwd)
            if mapped is None:
                truncated = f"no equivalent for {payload.get('name')}"
                break
            name, translated = mapped
            identifier = str(payload.get("call_id"))
            called[identifier] = name
            messages.append(
                {
                    "role": "assistant",
                    "content": call_json(name, translated),
                    "call_id": identifier,
                }
            )
        elif kind == "function_call_output":
            identifier = str(payload.get("call_id"))
            name = called.get(identifier)
            if name is None:
                continue
            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": unwrap(text_of(payload.get("output"))),
                    "call_id": identifier,
                }
            )

    if trimmed := _ending_in_an_answer(drop_commentary(pair(messages))):
        yield Trajectory(
            source=CODEX, origin=path.stem, messages=trimmed, cwd=cwd, truncated=truncated
        )


# --- a dataset somebody else made ---------------------------------------------


def from_chat(path: Path) -> Iterator[Trajectory]:
    """OpenAI-style ``{"messages": [...]}`` JSONL, one trajectory per line.

    The escape hatch for a corpus this package has never heard of. Tool names
    are put through the same table as everything else, so a dataset written
    against Claude's tools arrives speaking coder's.
    """
    for index, record in enumerate(records(path)):
        raw = record.get("messages")
        if not isinstance(raw, list):
            continue

        system = ""
        messages: list[dict] = []
        for message in raw:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                system = str(message.get("content") or "")
                continue
            content = text_of(message.get("content"))

            # A native tool_calls field is the other way a dataset may spell a
            # call. Moved into content, because that is where coder's protocol
            # puts one and where its parser looks.
            for call in message.get("tool_calls") or []:
                function = (call or {}).get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                mapped = translate(str(function.get("name") or ""), arguments)
                if mapped is None:
                    continue
                messages.append({"role": "assistant", "content": call_json(*mapped)})
            if not content:
                continue
            if role == "tool":
                name = str(message.get("name") or "")
                mapped = TOOLS.get(name)
                messages.append(
                    {"role": "tool", "name": mapped[0] if mapped else name, "content": content}
                )
            elif role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        if messages:
            yield Trajectory(
                source=CHAT, origin=f"{path.stem}#{index}", messages=messages, system=system
            )


def _ending_in_an_answer(messages: list[dict]) -> list[dict]:
    """``messages`` cut back to the last plain assistant answer.

    A trajectory that stops on a tool call has no lesson in its tail: the model
    is being shown a call and never shown what to do with the result. A
    trajectory that stops on prose is a finished piece of work.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        if message["content"].lstrip().startswith("{"):
            continue
        return messages[: index + 1]
    return []


# --- collecting ---------------------------------------------------------------

#: Where each source keeps its files, and which reader opens one.
READERS = {
    CODER: (lambda: sorted(SESSION_DIR.glob("*.jsonl")), from_coder),
    CLAUDE: (lambda: sorted(CLAUDE_DIR.glob("*/*.jsonl")), from_claude),
    CODEX: (lambda: sorted(CODEX_DIR.glob("*/*/*/rollout-*.jsonl")), from_codex),
}


def collect(
    sources=SOURCES, files: list[Path] | None = None, min_messages: int = 4
) -> Iterator[Trajectory]:
    """Every trajectory the named sources can produce.

    ``min_messages`` is the one filter applied here rather than in
    :mod:`.export`, because a two-message exchange is not a short example of
    agentic work -- it is a question and an answer with no work in it, and the
    thing being trained is the work.
    """
    for source in sources:
        if source == CHAT:
            for path in files or []:
                yield from _long_enough(from_chat(Path(path)), min_messages)
            continue
        entry = READERS.get(source)
        if entry is None:
            continue
        listing, reader = entry
        for path in listing():
            try:
                found = list(reader(path))
            except (OSError, ValueError):
                continue
            yield from _long_enough(found, min_messages)


def _long_enough(found, min_messages: int):
    for trajectory in found:
        if len(trajectory) >= min_messages:
            yield trajectory
