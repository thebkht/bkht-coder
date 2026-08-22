"""Context discipline: file-tree summary, token accounting, and compaction.

The system prompt carries a compact file list rather than a full repo map.
It is cheap, and the model can ``grep`` and ``glob`` its way to everything
else -- which it does more reliably than it reads a large map.
"""

from __future__ import annotations

from pathlib import Path

from .provider import ProviderError, collect

COMPACT_AT = 0.75
KEEP_RECENT = 6
MAX_TREE_ENTRIES = 200
# The walk is bounded separately from the listing: a monorepo can hold hundreds
# of thousands of files, and counting them all to report an exact remainder is
# not worth the seconds it costs on every startup.
MAX_TREE_SCAN = 20_000

SUMMARY_INSTRUCTION = """\
Summarise the conversation so far so that it can replace the original messages.

Keep, in this order:
1. What the user asked for, including anything they corrected or ruled out.
2. Which files were read or changed, by path, and what was found in them.
3. Decisions made and why.
4. What is still outstanding.

Be specific: keep file paths, function names and exact values. Drop tool output
that has already been acted on. Write plain prose under 400 words. Do not add
anything that is not in the conversation."""


def estimate_tokens(text: str) -> int:
    """Rough token count, used only before the model has reported a real one."""
    return max(1, len(text) // 4)


def file_tree(root: Path, max_entries: int = MAX_TREE_ENTRIES) -> str:
    """A flat list of the workspace's files, shallowest first.

    Flat rather than indented on purpose: the model uses these as arguments to
    ``read_file``, and a full relative path is directly usable where a tree
    drawing has to be reassembled first.

    Which files get the budget is the part that matters. Taken in alphabetical
    order, the first 200 paths of a monorepo are whatever sorts earliest --
    caches, vendored checkouts, a worktree full of somebody else's SDK -- and
    the model is left to guess where the project actually keeps its code. It
    guesses badly, then reads files that do not exist. Shallowest first spends
    the budget where the answer to "what is this project" lives: the README,
    the manifests, the top of each package.
    """
    root = Path(root)
    paths: list[Path] = []
    scanned = 0
    complete = True

    for path in _walk(root):
        scanned += 1
        if scanned > MAX_TREE_SCAN:
            complete = False
            break
        paths.append(path.relative_to(root))

    if not paths:
        return "(no files)"

    kept = sorted(paths, key=lambda p: (len(p.parts), p.parts))[:max_entries]
    listing = sorted(str(p) for p in kept)

    omitted = len(paths) - len(kept)
    if omitted or not complete:
        # "at least" when the walk was cut short: an exact count would be a
        # claim about files we never looked at.
        count = f"at least {omitted}" if not complete else str(omitted)
        listing.append(f"... and {count} more files, not listed. Use `glob` to find them.")
    return "\n".join(listing)


def _walk(root: Path):
    """Every file worth showing, ignored and hidden directories skipped.

    Hidden directories are skipped by every part of the path, not just the
    file's own name: `.aider.tags.cache.v4/07/15/b5e4.val` is not a dotfile by
    its basename, and a listing that fills up with cache entries is worse than
    no listing at all.
    """
    from .tools.search import iter_files

    for path in iter_files(root, root):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        yield path


def usage_ratio(session, num_ctx: int) -> float:
    """How full the context window is, as a fraction."""
    if not num_ctx:
        return 0.0
    used = session.prompt_tokens or estimate_tokens(
        "".join(m.get("content", "") for m in session.payload())
    )
    return used / num_ctx


def should_compact(session, num_ctx: int, threshold: float = COMPACT_AT) -> bool:
    """Whether the history is close enough to the window to need compacting."""
    if len(session.messages) <= KEEP_RECENT:
        return False
    return usage_ratio(session, num_ctx) >= threshold


def transcript(messages: list[dict]) -> str:
    """The messages rendered as text for the summarizer."""
    lines = []
    for message in messages:
        role = message.get("role", "?")
        label = message.get("name") or role
        content = (message.get("content") or "").strip()
        if len(content) > 2000:
            content = content[:2000] + " ...[truncated]"
        lines.append(f"[{label}] {content}")
    return "\n\n".join(lines)


def compact(session, provider, keep_recent: int = KEEP_RECENT) -> str | None:
    """Replace the oldest messages with a summary of them.

    Uses a separate model call rather than truncating, because dropping the
    early turns loses exactly the thing a long session depends on: what the
    user originally asked for. Returns the summary, or None if nothing was done.

    The system prompt and the most recent turns are untouched, and a failed
    summarization leaves the history exactly as it was -- degrading to a full
    context is better than silently forgetting the task.
    """
    if len(session.messages) <= keep_recent:
        return None

    older, recent = session.messages[:-keep_recent], session.messages[-keep_recent:]

    try:
        reply = collect(
            provider.chat(
                [
                    {"role": "system", "content": SUMMARY_INSTRUCTION},
                    {"role": "user", "content": transcript(older)},
                ]
            )
        )
    except ProviderError:
        return None

    summary = reply.content.strip()
    if not summary:
        return None

    session.messages = [
        {
            "role": "user",
            "content": (
                "Summary of the earlier part of this conversation, which has "
                f"been compacted to save context:\n\n{summary}"
            ),
        },
        *recent,
    ]
    # The reported count belongs to the pre-compaction prompt; clearing it
    # avoids compacting again immediately on the stale number.
    session.prompt_tokens = 0
    return summary
