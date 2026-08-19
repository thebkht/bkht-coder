"""Reading a line at the prompt, with history and slash completion.

``input()`` gives none of this: no recall of the last thing you asked, no
completion, and a typo means retyping the whole line. ``readline`` gives all
three for the cost of importing it -- but it does not exist on Windows, so
every entry point here degrades to plain ``input()`` rather than requiring one.
"""

from __future__ import annotations

import atexit
from pathlib import Path

from .session import STATE_DIR

HISTORY = STATE_DIR / "history"
HISTORY_LIMIT = 1000


def commands(repl) -> list[str]:
    """The slash commands ``repl`` actually answers to.

    Read off the object rather than kept in a second list, so a new ``do_``
    method is completable the moment it exists -- the same lookup
    :meth:`Repl.dispatch` does.
    """
    return sorted(f"/{name[3:]}" for name in dir(repl) if name.startswith("do_"))


class Completer:
    """Completes slash commands, and nothing else."""

    def __init__(self, options: list[str]) -> None:
        self.options = options
        self.matches: list[str] = []

    def __call__(self, text: str, state: int) -> str | None:
        if state == 0:
            self.matches = [o for o in self.options if o.startswith(text)] if text.startswith("/") else []
        try:
            return self.matches[state]
        except IndexError:
            return None


class Reader:
    """One prompt line at a time, with whatever the platform can give us."""

    def __init__(self, repl=None, history: Path | None = None, enabled: bool = True) -> None:
        self.history = HISTORY if history is None else history
        self.readline = None
        if enabled:
            self._setup(repl)

    def _setup(self, repl) -> None:
        try:
            import readline
        except ImportError:
            # Windows without pyreadline3. Plain input() still works; history
            # and completion simply are not there.
            return

        self.readline = readline
        try:
            self.history.parent.mkdir(parents=True, exist_ok=True)
            if self.history.exists():
                readline.read_history_file(str(self.history))
        except (OSError, ValueError):
            pass

        readline.set_history_length(HISTORY_LIMIT)
        if repl is not None:
            readline.set_completer(Completer(commands(repl)))
            readline.set_completer_delims(" \t\n")
            # libedit ships as readline on macOS and takes a different binding.
            libedit = "libedit" in (getattr(readline, "__doc__", "") or "")
            readline.parse_and_bind("bind ^I rl_complete" if libedit else "tab: complete")
        atexit.register(self.save)

    def save(self) -> None:
        if self.readline is None:
            return
        try:
            self.history.parent.mkdir(parents=True, exist_ok=True)
            self.readline.write_history_file(str(self.history))
        except (OSError, ValueError):
            pass

    def read(self, prompt: str = "> ") -> str:
        """Read one line. Raises EOFError on Ctrl-D, as ``input`` does."""
        return input(prompt)
