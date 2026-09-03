"""Turning transcripts into training data.

A coding agent produces exactly the thing a coding agent has to be trained on:
a task, the calls made to work it out, what each one returned, and an answer.
This package collects those from every source on the machine that has them,
normalizes each into coder's own protocol, and writes the result in the chat
format ``mlx_lm.lora`` reads.

Nothing here runs during a session. It is offline tooling that happens to live
inside the package, because it depends on the two things the session defines --
:mod:`bkht.coder.prompts`, which states the wire protocol, and
:mod:`bkht.coder.parsing`, which reads it back. Keeping them together is what
makes the round-trip check in :mod:`.render` possible at all, and that check is
the difference between a fine-tune that emits calls coder can execute and one
that emits calls it cannot.
"""

from .ingest import Trajectory, collect

__all__ = ["Trajectory", "collect"]
