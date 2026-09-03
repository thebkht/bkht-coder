"""``coder dataset`` -- build, inspect and read the training corpus.

Three verbs, and the middle one is the point. `build` writes the files; `stats`
says what is in them; `show` prints one example exactly as the model will read
it. A dataset nobody has looked at is a training run nobody can explain, and
the cheapest moment to discover there is not enough data is before the
afternoon spent on a LoRA rather than after it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..provider import DEFAULT_NUM_CTX
from . import export as exporting
from . import ingest, render

USAGE = """\
Usage: coder dataset build [--source a,b] [--out DIR] [--file PATH]...
       coder dataset stats [--out DIR]
       coder dataset show <n> [--out DIR]"""

#: Where a build lands unless it is told otherwise. Under the workspace rather
#: than the state directory: a dataset is an artifact of a project, it is large,
#: and it belongs next to the training config that consumes it.
DEFAULT_OUT = Path("training/data")


def add_arguments(parser) -> None:
    parser.add_argument("action", nargs="?", default="build", help="build, stats, or show.")
    parser.add_argument("rest", nargs="*", help="Arguments for the action.")
    parser.add_argument(
        "--source", default=",".join(ingest.SOURCES),
        help=f"Comma-separated: {', '.join(ingest.SOURCES)}.",
    )
    parser.add_argument("--file", action="append", default=[], help="A chat JSONL to include. Repeatable.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Directory to write to.")
    parser.add_argument(
        "--max-tokens", type=int, default=exporting.DEFAULT_MAX_TOKENS,
        help="Longest example to keep, in tokens.",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=DEFAULT_NUM_CTX,
        help="Window to size tool results for, as a session would.",
    )
    parser.add_argument("--cwd", default=".", help="Workspace the prompt is built against.")
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")


def run(args) -> int:
    """Execute ``coder dataset``. Returns the process exit status."""
    action = args.action or "build"
    out = Path(args.out).expanduser()

    if action == "build":
        return _build(args, out)
    if action == "stats":
        return _stats(out, as_json=args.json)
    if action == "show":
        return _show(args, out)

    print(f"error: unknown action {action!r}.\n{USAGE}", file=sys.stderr)
    return 2


def _build(args, out: Path) -> int:
    sources = [name.strip() for name in args.source.split(",") if name.strip()]
    unknown = [name for name in sources if name not in ingest.SOURCES]
    if unknown:
        print(
            f"error: unknown source(s) {', '.join(unknown)}. "
            f"Known: {', '.join(ingest.SOURCES)}.",
            file=sys.stderr,
        )
        return 2

    root = Path(args.cwd).expanduser().resolve()
    system = render.default_system(root)
    allowed = render.schemas(render.registry_for(root))

    trajectories = ingest.collect(sources, files=[Path(f) for f in args.file])
    pairs = render.render_all(
        trajectories, system, allowed=allowed,
        num_ctx=args.num_ctx, budget=args.max_tokens,
    )
    kept, report = exporting.select(pairs, max_tokens=args.max_tokens)

    if not kept:
        print(
            "No examples. Every trajectory was empty, too long, or used tools "
            "coder has no equivalent for.",
            file=sys.stderr,
        )
        return 1

    report.splits = exporting.write(exporting.split(kept), out)
    exporting.manifest(report, out, sources, args.max_tokens)

    if args.json:
        print(json.dumps(json.loads((out / "manifest.json").read_text()), indent=2))
    else:
        print(f"Wrote {out}/")
        print(report.render())
    return 0


def _stats(out: Path, as_json: bool = False) -> int:
    """What a built dataset contains, read back from the files.

    Read rather than recomputed, so this describes the data that will actually
    be trained on -- including a directory built last week by a version of this
    code that has since changed.
    """
    manifest = out / "manifest.json"
    if not manifest.exists():
        print(f"error: no dataset at {out}. Run `coder dataset build` first.", file=sys.stderr)
        return 2

    stored = json.loads(manifest.read_text(encoding="utf-8"))
    if as_json:
        print(json.dumps(stored, indent=2))
        return 0

    print(f"  {out}")
    print(f"  {stored['examples']} examples, about {stored['tokens']:,} tokens")
    if mix := stored.get("by_source"):
        print("  from " + ", ".join(f"{count} {name}" for name, count in mix.items()))
    for name, size in (stored.get("splits") or {}).items():
        print(f"  {name:<6} {size}")
    for why, count in (stored.get("dropped") or {}).items():
        if count:
            print(f"  dropped {count} {why}")
    return 0


def _show(args, out: Path) -> int:
    """One example, printed as the model reads it.

    The check no summary can stand in for. A histogram cannot tell you the
    conversation is incoherent; reading one can, and it takes a minute.
    """
    index = int(args.rest[0]) if args.rest and args.rest[0].isdigit() else 0
    path = out / exporting.FILES["train"]
    if not path.exists():
        print(f"error: no dataset at {out}. Run `coder dataset build` first.", file=sys.stderr)
        return 2

    with path.open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if position != index:
                continue
            for message in json.loads(line).get("messages", []):
                print(f"--- {message['role']} ---")
                print(message["content"])
            return 0

    print(f"error: no example {index} in {path}.", file=sys.stderr)
    return 2
