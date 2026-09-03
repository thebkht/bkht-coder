"""The turn's record of what it is doing, kept where compaction cannot reach it.

The failure this exists for is in the README already: a turn reads a file, the
window fills, `compact` and `elide_tool_results` free space by throwing away
what it read -- and the first thing they throw away is the model's own account
of what it was doing. So it starts again. A review of five files ended at the
time cap with the model still opening files, because by the fourth it no longer
remembered there had been a plan.

A plan lives on the :class:`~bkht.coder.session.Session` beside the messages
rather than in them. Nothing that frees context can take it, and it is appended
to every request as a reminder, so the model reads its own list on every single
round trip whether or not the turn that wrote it survived.

Kept deliberately blunt: a numbered list of short lines, each done or not. It is
not a task tracker. It is the one sentence a model that has lost the thread
needs in order to find it again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: A plan longer than this is not a plan, it is the work. A 14b that writes
#: twelve steps has decomposed the task instead of doing it, and every step
#: costs tokens on every subsequent request.
MAX_STEPS = 8

#: Steps are one line each. The detail belongs in the step being done, not in
#: the list of steps not started yet.
MAX_STEP_CHARS = 120

DONE, TODO = "x", " "


@dataclass
class Step:
    """One line of the plan."""

    text: str
    done: bool = False


@dataclass
class Plan:
    """A short numbered list the turn ticks its way down.

    Empty until the model writes one, and an empty plan renders as nothing at
    all -- a turn that needs no plan should not be paying for the words that
    say it has none.
    """

    steps: list[Step] = field(default_factory=list)

    def set(self, steps: list[str]) -> None:
        """Replace the plan. Ticks are lost, because the plan changed.

        Rewriting is how the model revises: it discovered the second step was
        wrong and says so by writing a new list. Trying to carry ticks across a
        rewrite means guessing which of the new steps the old ones were, and a
        wrong guess reports work as finished that nobody did.
        """
        cleaned = [
            line.strip()[:MAX_STEP_CHARS] for line in steps if str(line).strip()
        ]
        self.steps = [Step(text) for text in cleaned[:MAX_STEPS]]

    def tick(self, number: int) -> Step:
        """Mark step ``number`` (1-based, as it is rendered) done."""
        if not self.steps:
            raise IndexError("there is no plan yet")
        if not 1 <= number <= len(self.steps):
            raise IndexError(
                f"step {number} does not exist; the plan has {len(self.steps)}"
            )
        step = self.steps[number - 1]
        step.done = True
        return step

    # --- rendering ----------------------------------------------------------

    def render(self) -> str:
        """The plan as the model is shown it, on every request."""
        return "\n".join(
            f"{index}. [{DONE if step.done else TODO}] {step.text}"
            for index, step in enumerate(self.steps, start=1)
        )

    def progress(self) -> tuple[int, int]:
        """How many steps are done, out of how many."""
        return sum(1 for step in self.steps if step.done), len(self.steps)

    def finished(self) -> bool:
        done, total = self.progress()
        return bool(total) and done == total

    # --- persistence --------------------------------------------------------

    def as_record(self) -> list[dict]:
        return [{"text": step.text, "done": step.done} for step in self.steps]

    @classmethod
    def from_record(cls, record) -> "Plan":
        """Rebuild from a session file, skipping anything malformed.

        The transcript is append-only and a session killed mid-write can leave
        a partial line; a plan is not worth refusing to resume a session over.
        """
        steps = []
        for entry in record or []:
            if isinstance(entry, dict) and str(entry.get("text", "")).strip():
                steps.append(Step(str(entry["text"]), bool(entry.get("done"))))
        return cls(steps[:MAX_STEPS])

    def __len__(self) -> int:
        return len(self.steps)

    def __bool__(self) -> bool:
        return bool(self.steps)
