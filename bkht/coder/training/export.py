"""Examples to the files ``mlx_lm.lora`` reads.

Three jobs, in order: throw away what should not be trained on, say plainly how
much is left and what shape it is, and split it. The middle one is the reason
this is a command rather than a script -- a dataset nobody has looked at is a
training run nobody can explain, and the histogram is where "there is not
enough data" gets found out before an afternoon is spent on a LoRA.

Token counts are estimated, not tokenized. Pulling in a tokenizer would make
this package depend on the training extra, which the shipped agent must not do;
and the number is wanted for bucketing, where a fixed ratio is close enough to
be honest about.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

#: Characters per token. Code and paths run denser than prose, so this is a
#: little under the usual 4:1 -- an estimate that errs toward "this example is
#: longer than you think", which is the safe direction next to a memory limit.
CHARS_PER_TOKEN = 3.5

#: The sequence length the 14b LoRA is configured for on a 16 GB machine.
#: Anything longer is not trained on a truncated version of itself: a trajectory
#: cut in half mid-way through a tool result teaches the model to answer from
#: evidence it never saw.
DEFAULT_MAX_TOKENS = 4096

#: How the three files divide. Small holdouts, because the corpus is small and
#: every example spent on measurement is one not spent on learning.
VALID_SHARE = 0.05
TEST_SHARE = 0.05

#: Fixed, so that rebuilding a dataset from the same transcripts twice puts the
#: same examples in the same splits. A model evaluated against a test set it was
#: trained on in a previous build is a model with a made-up score.
SEED = 20260903

FILES = {"train": "train.jsonl", "valid": "valid.jsonl", "test": "test.jsonl"}


def tokens_of(example) -> int:
    """Roughly how many tokens one example is."""
    total = sum(len(m.get("content") or "") for m in example.messages)
    return int(total / CHARS_PER_TOKEN)


def fingerprint(example) -> str:
    """What makes two examples the same example.

    The conversation, not the provenance. The same task run twice through two
    different agents is one lesson, and training on it twice only teaches the
    model that this particular file matters more than the others.
    """
    joined = "\n".join(
        f"{m.get('role')}:{m.get('content')}" for m in example.messages
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass
class Report:
    """What a build did, in the terms someone deciding whether to train needs."""

    kept: int = 0
    duplicates: int = 0
    too_long: int = 0
    unparseable: int = 0
    by_source: Counter = field(default_factory=Counter)
    tokens: list[int] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens)

    def histogram(self, buckets=(512, 1024, 2048, 4096, 8192)) -> list[tuple[str, int]]:
        """How long the kept examples are, in buckets.

        The one number that decides whether ``max_seq_length`` is set right: a
        corpus whose mass sits above the configured window is a corpus most of
        which will be truncated away during training, and nothing else in this
        report would say so.
        """
        counted = Counter()
        for count in self.tokens:
            label = next(
                (f"<{bucket}" for bucket in buckets if count < bucket),
                f">={buckets[-1]}",
            )
            counted[label] += 1
        order = [f"<{bucket}" for bucket in buckets] + [f">={buckets[-1]}"]
        return [(label, counted[label]) for label in order if counted[label]]

    def render(self) -> str:
        """The report as it is printed. Written, not generated from the fields:
        what matters is different per line, and a table of every attribute would
        bury the two numbers a decision actually rests on."""
        lines = [
            f"  {self.kept} examples, about {self.total_tokens:,} tokens",
        ]
        if self.by_source:
            mix = ", ".join(
                f"{count} {source}" for source, count in self.by_source.most_common()
            )
            lines.append(f"  from {mix}")
        dropped = [
            (self.duplicates, "duplicate"),
            (self.too_long, "over the token budget"),
            (self.unparseable, "with a call the parser could not read back"),
        ]
        for count, why in dropped:
            if count:
                lines.append(f"  dropped {count} {why}")
        if self.tokens:
            lines.append("")
            width = max(count for _, count in self.histogram())
            for label, count in self.histogram():
                bar = "#" * max(1, round(20 * count / width))
                lines.append(f"  {label:>8} {bar} {count}")
        if self.splits:
            lines.append("")
            lines.append(
                "  " + "  ".join(f"{name} {size}" for name, size in self.splits.items())
            )
        return "\n".join(lines)


def select(pairs, max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[list, Report]:
    """The examples worth keeping, and the account of what was not.

    ``pairs`` is what :func:`render.render_all` yields -- ``(example, problems)``
    -- so a dropped round-trip is counted here rather than swallowed there.
    """
    report = Report()
    seen: set[str] = set()
    kept = []

    for example, problems in pairs:
        if example is None:
            report.unparseable += 1
            continue
        if problems:
            report.unparseable += 1
            continue
        mark = fingerprint(example)
        if mark in seen:
            report.duplicates += 1
            continue
        seen.add(mark)

        count = tokens_of(example)
        if count > max_tokens:
            report.too_long += 1
            continue

        kept.append(example)
        report.kept += 1
        report.by_source[example.source] += 1
        report.tokens.append(count)

    return kept, report


def split(examples: list, seed: int = SEED) -> dict[str, list]:
    """Train, valid and test, shuffled once and cut.

    A holdout is never empty when there is anything to put in it: a valid set of
    zero makes ``mlx_lm.lora`` fail at the end of the first epoch, which is a
    long way to travel for a mistake made here.
    """
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    valid_size = min(max(1, round(total * VALID_SHARE)), max(0, total - 2)) if total > 2 else 0
    test_size = min(max(1, round(total * TEST_SHARE)), max(0, total - valid_size - 1)) if total > 2 else 0

    return {
        "valid": shuffled[:valid_size],
        "test": shuffled[valid_size : valid_size + test_size],
        "train": shuffled[valid_size + test_size :],
    }


def write(splits: dict[str, list], out: Path) -> dict[str, int]:
    """The three files, and how many lines went into each."""
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ("train", "valid", "test"):
        examples = splits.get(name, [])
        path = out / FILES[name]
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example.as_record(), ensure_ascii=False) + "\n")
        written[name] = len(examples)
    return written


def manifest(report: Report, out: Path, sources, max_tokens: int) -> Path:
    """A record beside the data of what produced it.

    A dataset directory with no note of where its contents came from becomes
    unusable within a week: two builds look identical, and the difference
    between them is the whole reason one model is better than the other.
    """
    path = out / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "sources": list(sources),
                "max_tokens": max_tokens,
                "examples": report.kept,
                "tokens": report.total_tokens,
                "by_source": dict(report.by_source),
                "dropped": {
                    "duplicate": report.duplicates,
                    "too_long": report.too_long,
                    "unparseable": report.unparseable,
                },
                "splits": report.splits,
                "seed": SEED,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
