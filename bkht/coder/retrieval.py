"""Deterministic codebase retrieval, run before the model's first turn.

The model is given `grep` and `glob`, and the system prompt tells it to use
them. A 14b model frequently does not: it answers, or edits, from the flat file
list in the system prompt without ever looking. Rather than trusting it to
search, we search first -- in Python, from the words the user actually used --
and hand it the result as though a search had already been made.

This is a keyword match, not an index. It is meant to save the model its first
two or three turns, not to replace reading the files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .tools.base import truncate
from .tools.fs import read_text
from .tools.search import MAX_SCANNED_BYTES, iter_files

MAX_TERMS = 8
MAX_FILES = 6
MAX_LINES_PER_FILE = 4
BUDGET_CHARS = 2000

# A whole-repo scan happens on every command, so it needs a ceiling that does
# not depend on the repo being small.
MAX_SCANNED_FILES = 2000
MAX_SCANNED_TOTAL = 20_000_000

PATH_SCORE = 5
DEFINITION_SCORE = 3
MENTION_SCORE = 1
COVERAGE_SCORE = 4

# Mentions per term are capped, and matching a second term is worth more than
# matching the first one twenty times: without this a file that merely says
# "depth" a lot outranks the one that actually implements "undo depth".
MAX_COUNTED_MENTIONS = 3

SOURCE_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".rb", ".java",
    ".kt", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".php", ".sh",
    ".sql", ".html", ".css", ".scss", ".json", ".toml", ".yaml", ".yml", ".md",
}

# English filler plus the verbs and nouns that appear in almost every task,
# which match everywhere and therefore rank nothing.
STOPWORDS = {
    "the", "this", "that", "these", "those", "there", "here", "what", "when",
    "where", "which", "who", "why", "how", "and", "but", "for", "with", "from",
    "into", "about", "have", "has", "had", "was", "were", "are", "you", "your",
    "our", "not", "can", "could", "should", "would", "will", "please", "then",
    "than", "also", "some", "any", "all", "its", "his", "her", "them", "they",
    "add", "adds", "added", "fix", "fixes", "fixed", "make", "makes", "made",
    "use", "uses", "used", "using", "get", "gets", "set", "sets", "run", "runs",
    "does", "did", "doing", "done", "need", "needs", "want", "wants", "let",
    "file", "files", "code", "codebase", "project", "repo", "repository",
    "function", "functions", "method", "methods", "change", "changes", "work",
    "works", "working", "check", "show", "tell", "look", "find", "write",
    # Asking to be told about something is not a term to search for. Left in,
    # "explain me this project" searches the repository for "explain" and
    # hands the model whatever happens to say it.
    "explain", "explains", "explained", "describe", "describes", "summarise",
    "summarize", "summary", "overview", "understand", "give", "gives", "walk",
    "mean", "means",
    "new", "old", "one", "two", "now", "just", "like", "sure", "yes", "yeah",
    "thanks", "thank", "ok", "okay", "good", "ahead", "sounds", "great",
}

_QUOTED = re.compile(r"[\"'`]([^\"'`\n]{2,})[\"'`]")
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DEFINITION_KEYWORDS = "async def|def|class|function|const|let|var|type|struct|fn|interface"


def _is_pathlike(token: str) -> bool:
    return "/" in token or Path(token).suffix.lower() in SOURCE_SUFFIXES


def _specificity(term: str) -> tuple[int, int]:
    """Sort key: distinctive tokens first, longer before shorter."""
    distinctive = int("/" in term or "_" in term or "." in term or term[1:] != term[1:].lower())
    return (distinctive, len(term))


def terms(message: str) -> list[str]:
    """The search terms in a user message, most specific first.

    Returns an empty list when the message carries nothing worth searching for
    -- "thanks", "yes go ahead" -- so that conversational turns cost nothing.
    """
    candidates: list[str] = []

    def keep(value: str) -> None:
        value = value.strip().strip(".,:;!?()[]{}")
        if len(value) < 3 or value.lower() in STOPWORDS:
            return
        if value not in candidates:
            candidates.append(value)

    for quoted in _QUOTED.findall(message):
        keep(quoted)

    for token in _TOKEN.findall(message):
        token = token.rstrip("./-")
        if not token:
            continue
        keep(token)
        # A path or a compound identifier is also worth searching for by its
        # parts: "tools/search.py" should find `search`, `handle_undo` `undo`.
        if _is_pathlike(token) or "_" in token or token[1:] != token[1:].lower():
            for part in _IDENTIFIER.findall(re.sub(r"(?<!^)(?=[A-Z])", " ", token.replace("/", " ").replace(".", " ").replace("_", " "))):
                keep(part)

    candidates.sort(key=_specificity, reverse=True)
    return candidates[:MAX_TERMS]


@dataclass
class Hit:
    """One file that matched, and the lines worth showing from it."""

    path: str
    score: int = 0
    lines: list[tuple[int, str, bool]] = field(default_factory=list)  # number, text, is-definition

    def best_lines(self) -> list[tuple[int, str]]:
        """Definitions first, then earliest mentions."""
        ordered = sorted(self.lines, key=lambda line: (not line[2], line[0]))
        return [(number, text) for number, text, _ in ordered[:MAX_LINES_PER_FILE]]


def _definition_pattern(term: str) -> re.Pattern:
    escaped = re.escape(term)
    return re.compile(
        rf"^\s*(?:{_DEFINITION_KEYWORDS})\s+{escaped}\b|^{escaped}\s*[:=]",
    )


def search(root: Path, wanted: list[str]) -> list[Hit]:
    """Rank the workspace's files against ``wanted``, best first."""
    if not wanted:
        return []

    root = Path(root)
    mentions = [re.compile(re.escape(term), re.IGNORECASE) for term in wanted]
    definitions = [_definition_pattern(term) for term in wanted]
    lowered = [term.lower() for term in wanted]

    hits: list[Hit] = []
    scanned = 0
    total_bytes = 0

    for path in iter_files(root, root):
        if scanned >= MAX_SCANNED_FILES or total_bytes >= MAX_SCANNED_TOTAL:
            break
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_SCANNED_BYTES:
            continue

        relative = str(path.relative_to(root))
        hit = Hit(path=relative)

        try:
            content = read_text(path)
        except Exception:
            continue  # binary or unreadable; not an error for a search
        scanned += 1
        total_bytes += size

        counts = [0] * len(wanted)
        defines = [False] * len(wanted)
        for number, line in enumerate(content.splitlines(), start=1):
            matched = False
            defined = False
            for index, (mention, definition) in enumerate(zip(mentions, definitions)):
                if not mention.search(line):
                    continue
                matched = True
                counts[index] += 1
                if definition.search(line):
                    defined = defines[index] = True
            if not matched:
                continue
            if len(hit.lines) < MAX_LINES_PER_FILE * 3:
                hit.lines.append((number, line.strip()[:200], defined))

        covered = 0
        for index, term in enumerate(lowered):
            in_path = term in relative.lower()
            if not (in_path or counts[index]):
                continue
            covered += 1
            hit.score += PATH_SCORE if in_path else 0
            hit.score += DEFINITION_SCORE if defines[index] else 0
            hit.score += MENTION_SCORE * min(counts[index], MAX_COUNTED_MENTIONS)

        # Every term after the first earns a bonus, so breadth of match beats
        # sheer repetition of one common word.
        hit.score += COVERAGE_SCORE * max(0, covered - 1)

        # A file named after the search term but never mentioning it is still
        # the answer -- ".gitignore" says "gitignore" nowhere inside itself --
        # so it is shown by its opening lines rather than dropped for having no
        # matching line.
        if hit.score and not hit.lines:
            hit.lines = [
                (number, line.strip()[:200], False)
                for number, line in enumerate(content.splitlines(), start=1)
                if line.strip()
            ][:MAX_LINES_PER_FILE]

        if hit.score and hit.lines:
            hits.append(hit)

    hits.sort(key=lambda hit: (hit.score, -len(hit.path)), reverse=True)
    return hits[:MAX_FILES]


HEADER = """\
Automatic keyword search of the workspace for this request. You did not call this --
it is provided to save you a step, and it is a keyword match, not a complete answer.
Terms: {terms}"""

# The same search, when the model asked for it. The scouted header's "you did
# not call this" is a lie once it has.
CALLED_HEADER = """\
Keyword search of the workspace. This is a keyword match, not a complete answer.
Terms: {terms}"""

FOOTER = """\
Read the files that matter before changing anything, and grep further if this missed something."""


def render(
    wanted: list[str],
    hits: list[Hit],
    budget: int = BUDGET_CHARS,
    header: str = HEADER,
) -> str:
    """The block as the model sees it, in the same shape ``grep`` returns."""
    if not hits:
        return ""

    body: list[str] = []
    used = 0
    for hit in hits:
        block = "\n".join(
            f"{hit.path}:{number}: {text}" for number, text in hit.best_lines()
        )
        if used + len(block) > budget:
            break
        body.append(block)
        used += len(block) + 1

    if not body:
        return ""

    return truncate(
        header.format(terms=", ".join(wanted))
        + "\n\n"
        + "\n\n".join(body)
        + "\n\n"
        + FOOTER
    )


#: A request that names something the workspace does not contain. Searching the
#: files for "review github run 33185669396" matched `.github/` and `review/`,
#: because both are real directories -- and then the turn went reading the
#: workflow file, which says what the job would do and nothing about what it
#: did. The answer was one `gh run view` away.
#:
#: Narrow on purpose: an identifier with a number attached, or a URL. A task
#: that merely mentions running something still gets its search.
EXTERNAL = re.compile(
    r"""(?xi)
    https?://                       # any URL
    | \#\d{2,}                       # #1234
    | \b(?:run|pr|pull|issue|build|job|pipeline|workflow|deploy|release)
      \s+\#?\d{3,}\b                 # run 33185669396, issue 4210
    """
)


def about_something_else(message: str) -> bool:
    """True when the request names something outside this workspace.

    The scout is a keyword match made before anything has read the request. On
    a question about a CI run it will match something regardless -- every repo
    has a `.github` -- and what it matches then is worse than nothing, because
    it is the first thing in the turn and it points the wrong way.
    """
    return bool(EXTERNAL.search(message))


def scout(root: Path, message: str) -> str:
    """Search the workspace for what ``message`` is about.

    Returns the rendered block, or ``""`` when there is nothing useful to say --
    which is the normal outcome for a conversational turn.
    """
    if about_something_else(message):
        return ""
    wanted = terms(message)
    if not wanted:
        return ""
    return render(wanted, search(root, wanted))


SEARCH_TERM_SPLIT = re.compile(r"[,\s]+")


def register_retrieval_tool(registry, workspace) -> None:
    """Register ``codebase_search`` as a tool the model can actually call.

    It could already see one. :meth:`Agent._scout` opens every turn by injecting
    a ``tool`` message named ``codebase_search``, and a model that has just read
    the result of a tool reasonably concludes it may call that tool again -- so
    it did, and was told there is no such tool, which cost it a retry and taught
    it that the format was wrong when the format was fine.

    Registering it is the honest fix. The scouted block and this tool are now
    the same thing said twice: once unprompted at the start of the turn, and
    once on request when the model wants to look for something else.
    """
    from .tools.base import Tool, ToolError, ToolResult

    def codebase_search(terms: str) -> ToolResult:
        wanted = [term for term in SEARCH_TERM_SPLIT.split(terms.strip()) if term]
        if not wanted:
            raise ToolError("terms is empty; give one or more words to search for")
        wanted = wanted[:MAX_TERMS]
        block = render(
            wanted, search(workspace.root, wanted), header=CALLED_HEADER
        )
        if not block:
            return ToolResult.success(
                f"No files matched {', '.join(wanted)}. "
                "Try `grep` with a narrower pattern, or `glob` to find the file."
            )
        return ToolResult.success(block)

    registry.add(
        Tool(
            name="codebase_search",
            description=(
                "Find where the workspace talks about something, by keyword. "
                "Returns the most relevant files with matching lines. Use it to "
                "locate an area; use `grep` when you know the exact string."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "string",
                        "description": "Words to look for, separated by commas.",
                    },
                },
                "required": ["terms"],
            },
            run=codebase_search,
        )
    )
