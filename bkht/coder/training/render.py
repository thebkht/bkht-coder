"""Trajectories to the exact bytes the model is trained on.

The whole file exists to protect one invariant: **what the model is trained to
emit must be what the runtime parser reads back.** A fine-tune whose calls
coder cannot parse is worse than no fine-tune, and the failure is invisible --
the model looks fluent, the loop answers every reply with a correction, and no
turn ever completes.

So the rendering does not describe the protocol, it *uses* it.
:func:`prompts.system_prompt` builds the same system prompt a session sends, off
a real registry; and :func:`verify` runs every rendered assistant call back
through :func:`parsing.parse_tool_calls` and refuses the ones that do not
survive. Both are cheap, and both would otherwise be assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import prompts
from ..parsing import parse_tool_calls
from ..session import prompt_hash
from ..skills import Discovery
from ..context import KEEP_TOOL_RESULTS, elide_tool_results
from ..tools import build_registry
from ..tools.base import output_budget, truncate
from .ingest import Trajectory, call_json

#: The roles the chat format accepts. Coder's `tool` role is kept as itself --
#: Qwen's chat template renders it, and flattening it into a user turn here
#: would mean training on one shape and serving another.
ROLES = frozenset({"user", "assistant", "tool"})

#: What stands in for a tool that returned nothing. A call with no result under
#: it is the worst thing this file could emit: the model would be shown a call
#: and then, immediately, another one -- which is exactly the behaviour the loop
#: has to guard against at runtime. An empty result is a real thing tools
#: produce, so it is written down as one rather than skipped.
NO_OUTPUT = "(no output)"


@dataclass(frozen=True)
class Example:
    """One trajectory as ``mlx_lm.lora`` takes it, plus what it came from."""

    messages: list[dict]
    source: str
    origin: str
    system_hash: str

    def as_record(self) -> dict:
        """The line written to train.jsonl. Nothing but the conversation.

        The provenance above is deliberately not in it: the trainer would read
        it as part of the example, and a model that has learned to predict
        `"source": "codex"` has spent capacity on the wrong thing.
        """
        return {"messages": self.messages}


def registry_for(root: Path | str = "."):
    """The tool set a default session offers, which is the one to train for.

    `session` and `delegate` are passed because a default session passes them --
    they are what `plan` and `task` are registered on, and both are on by
    default. Without them the dataset would drop every plan and every delegation
    as a call to a tool that does not exist, which is the opposite of true.

    Skills are left empty. Which skills a workspace has is a property of that
    workspace, so a `skill` tool here would name whichever ones happened to be
    installed on the machine that built the dataset.
    """
    from ..session import Session

    registry, _ = build_registry(
        Path(root),
        skills=Discovery(),
        session=Session(),
        delegate={"provider": None, "skills": Discovery(), "listener": None},
    )
    return registry


def schemas(registry) -> dict[str, tuple[set[str], set[str]]]:
    """Each tool's argument names and required ones, by tool name.

    Read off the live registry rather than written down here, so that the
    dataset can never teach a call the running program would reject. It very
    nearly did: Claude Code's `Read` takes a `pages` argument coder's
    `read_file` has never had, and an unknown argument is not ignored at
    runtime -- `validate_arguments` raises on it, so every such call would fail.
    """
    found = {}
    for tool in registry:
        properties = set(tool.parameters.get("properties", {}))
        found[tool.name] = (properties, set(tool.parameters.get("required", [])))
    return found


def conform(content: str, allowed: dict[str, tuple[set[str], set[str]]]) -> str | None:
    """One rendered call with arguments this program would actually accept.

    Returns the call re-rendered without its unknown arguments, or None when
    there is no honest way to keep it: an unknown tool, or a required argument
    that is missing once the unknown ones are gone. None means the trajectory
    stops here -- see :func:`render`.
    """
    calls = parse_tool_calls(content)
    if len(calls) != 1:
        return None
    call = calls[0]
    if call.name not in allowed:
        return None
    properties, required = allowed[call.name]
    kept = {k: v for k, v in call.arguments.items() if k in properties}
    if required - set(kept):
        return None
    return call_json(call.name, kept)


def default_system(root: Path | str = ".") -> str:
    """The system prompt a default session sends, for a trajectory with none.

    Built off a real registry rather than a literal, so that a tool added to
    coder is a tool the next dataset knows about -- and so that a prompt this
    file invents can never drift from the one the session actually sends.

    The tree is left out on purpose. It names the files of whichever workspace
    happened to be current, and a model trained to expect a listing that matches
    its task is a model that has learned the wrong thing about the prompt.
    """
    return prompts.system_prompt(registry_for(root), str(root))


def render(
    trajectory: Trajectory,
    system: str = "",
    allowed: dict[str, tuple[set[str], set[str]]] | None = None,
    num_ctx: int = 0,
    budget: int = 0,
) -> Example | None:
    """One trajectory as an example, or None when there is nothing to learn.

    The system prompt comes from the trajectory when it recorded its own -- a
    coder session did -- and from ``system`` otherwise. That asymmetry is the
    point: a Claude Code conversation happened under Claude Code's prompt, and
    the prompt it has to be *taught* under is coder's.
    """
    system = trajectory.system or system
    cap = output_budget(num_ctx)
    messages = [{"role": "system", "content": system}] if system else []

    for message in trajectory.messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role not in ROLES:
            continue

        # A tool result is labelled with the tool that produced it, the way the
        # session's own history labels one. The name is part of what the model
        # reads: "this came back from grep" is most of what makes the next call
        # predictable. An empty one is still written: a call with nothing under
        # it teaches the model not to wait for results.
        if role == "tool":
            # Bounded by coder's own runtime cap, not left as the foreign agent
            # wrote it. At serve time every tool result reaching this model will
            # have been through `truncate` -- a frontier agent's unbounded
            # 40,000-character `cat` is evidence coder would never deliver, and
            # training on it teaches the model to answer from what it will not
            # be given.
            #
            # `name` is carried on the message rather than folded into the text
            # here, because elision reads it: fold it in first and every elided
            # result says it came from a tool called "tool".
            messages.append(
                {
                    "role": "tool",
                    "name": message.get("name") or "tool",
                    "content": truncate(content, max_chars=cap) if content else NO_OUTPUT,
                }
            )
            continue

        if not content:
            continue
        if role == "assistant" and content.lstrip().startswith("{") and allowed is not None:
            fixed = conform(content, allowed)
            if fixed is None:
                # Nothing after this call can be learned from -- every message
                # below it answers a call that was not made.
                break
            content = fixed
        messages.append({"role": role, "content": content})

    messages = _ending_in_an_answer(messages)
    if budget:
        messages = fit(messages, budget)
    messages = [_labelled(message) for message in messages]

    # An example has to contain the thing being learned: at least one exchange
    # where the model produced something. A list that is all user turns is a
    # transcript of somebody typing.
    if not any(m["role"] == "assistant" for m in messages):
        return None

    return Example(
        messages=messages,
        source=trajectory.source,
        origin=trajectory.origin,
        system_hash=prompt_hash(system) if system else "",
    )


def _labelled(message: dict) -> dict:
    """One message in the chat format, with the tool name folded into the text.

    The last step, after elision has had its use of the name. The chat format
    has no field for it, and "this came back from grep" is most of what makes
    the next call predictable -- so it goes into the content rather than being
    lost.
    """
    if message["role"] != "tool":
        return {"role": message["role"], "content": message["content"]}
    return {
        "role": "tool",
        "content": f"[{message.get('name') or 'tool'}]\n{message['content']}",
    }


def _ending_in_an_answer(messages: list[dict]) -> list[dict]:
    """``messages`` cut back to the last plain assistant answer.

    Applied again here rather than trusted from ingest, because conforming the
    calls above can shorten a trajectory -- and one that now ends on a tool call
    would be teaching the model to stop mid-work.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message["role"] != "assistant":
            continue
        if message["content"].lstrip().startswith("{"):
            continue
        return messages[: index + 1]
    return []


@dataclass
class _Elidable:
    """The two attributes ``context.elide_tool_results`` reaches for.

    Reused rather than reimplemented: the elision a long session gets at
    runtime and the elision an over-long example gets here must be the same
    text, or the model is trained to read a note the loop does not write.
    """

    messages: list[dict]


def fit(messages: list[dict], budget: int) -> list[dict]:
    """``messages`` cut down to fit ``budget`` tokens, the way a session would.

    Two levers, in the order the running loop pulls them.

    First elision, and only under pressure. That "only" is the whole reason
    this is not applied to everything: the loop elides when the window fills,
    so an example that fits is an example whose results the model really would
    have had, and eliding it anyway would teach the model to work from less
    evidence than it will actually be given.

    Then, still over, the middle goes. What is kept is the task and the recent
    work: the first user turn, because an example that does not say what was
    asked teaches nothing, and a suffix beginning at a later call. Cutting only
    at a call keeps every result behind the call it answers -- start anywhere
    else and the example opens on a result to a call that is not there.

    That shape is not invented for training. It is what a long session looks
    like after the loop has compacted it: the request, and the work near the
    end.

    Returns an empty list when nothing that fits is worth training on.
    """
    if _tokens(messages) <= budget:
        return messages

    # Deep enough to edit: elision rewrites message content in place, and these
    # dicts are shared with the trajectory this was rendered from.
    working = [dict(message) for message in messages]
    for keep in (KEEP_TOOL_RESULTS + 2, KEEP_TOOL_RESULTS):
        elide_tool_results(_Elidable(working), keep_recent=keep)
        if _tokens(working) <= budget:
            return working

    head = [m for m in working if m["role"] == "system"]
    body = [m for m in working if m["role"] != "system"]
    if not body:
        return []

    # The task, then the oldest cut that fits. A user turn is preferred where
    # there is one -- a later request stands on its own and needs no preamble.
    task = body[0] if body[0]["role"] == "user" else None
    for start in range(1, len(body)):
        message = body[start]
        starts_here = message["role"] == "user" or (
            message["role"] == "assistant" and message["content"].lstrip().startswith("{")
        )
        if not starts_here:
            continue
        kept = _ending_in_an_answer(body[start:])
        if not kept:
            continue
        opening = [] if (task is None or message["role"] == "user") else [task]
        candidate = head + opening + kept
        if _tokens(candidate) <= budget:
            return candidate
    return []


def _tokens(messages: list[dict]) -> int:
    """The same estimate ``export`` buckets by, so the two cannot disagree."""
    from .export import CHARS_PER_TOKEN

    return int(sum(len(m.get("content") or "") for m in messages) / CHARS_PER_TOKEN)


def verify(example: Example) -> list[str]:
    """Every way this example would fail to round-trip, named.

    An empty list means every assistant tool call in it parses back to exactly
    the call it was rendered from. This is the check that makes the dataset
    trustworthy, so it runs over every example on the way out rather than as a
    test somebody remembers to run.
    """
    problems = []
    for index, message in enumerate(example.messages):
        if message["role"] != "assistant":
            continue
        content = message["content"]
        if not content.lstrip().startswith("{"):
            # Prose. The other half of the protocol, and a valid thing to emit.
            continue
        calls = parse_tool_calls(content)
        if len(calls) != 1:
            problems.append(
                f"message {index}: parsed {len(calls)} calls out of what should be one"
            )
            continue
        if not calls[0].name:
            problems.append(f"message {index}: parsed a call with no name")
    return problems


def render_all(
    trajectories,
    system: str = "",
    strict: bool = True,
    allowed: dict[str, tuple[set[str], set[str]]] | None = None,
    num_ctx: int = 0,
    budget: int = 0,
):
    """Every trajectory that renders and survives :func:`verify`.

    Yields ``(example, problems)`` so a caller can report what it dropped.
    ``strict`` is what decides whether a failing example is dropped or kept; it
    is on by default because a call the parser cannot read is not a lesson, it
    is noise with a plausible shape.
    """
    for trajectory in trajectories:
        example = render(trajectory, system, allowed, num_ctx, budget)
        if example is None:
            continue
        problems = verify(example)
        if problems and strict:
            yield None, problems
            continue
        yield example, problems
