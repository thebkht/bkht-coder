"""Reading a line at the prompt, with history and slash completion.

``input()`` gives none of this: no recall of the last thing you asked, no
completion, and a typo means retyping the whole line. Two things can give all
three -- :mod:`~bkht.coder.lineedit`, which draws and edits the line itself, and
``readline``, which does it for us but cannot answer Shift+Tab or keep a live
footer under the input.

So the editor is what a terminal gets, readline is what everything else gets,
and ``input()`` alone is what is left on a Windows box without either. Each
step down loses a feature and none of them loses a prompt.
"""

from __future__ import annotations

import atexit
from pathlib import Path

from . import lineedit
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

    def __init__(
        self, repl=None, history: Path | None = None, enabled: bool = True,
        footer=None, cycle=None, attach=None, on_image=None, header=None,
    ) -> None:
        self.history = HISTORY if history is None else history
        self.readline = None
        self.editor = None
        if not enabled:
            return
        if lineedit.available():
            self._edit(repl, footer, cycle, attach, on_image, header)
        else:
            self._setup(repl)

    @property
    def cycles(self) -> bool:
        """True when Shift+Tab reaches us, and so when the footer may offer it."""
        return self.editor is not None

    def _edit(self, repl, footer, cycle, attach=None, on_image=None, header=None) -> None:
        """Our own editor, which is the only path that can see Shift+Tab."""
        self.editor = lineedit.Editor(
            completions=(lambda: commands(repl)) if repl is not None else None,
            footer=footer,
            header=header,
            cycle=cycle,
            attach=attach,
            on_image=on_image,
            history=self._recall(),
        )
        atexit.register(self.save)

    def _recall(self) -> list[str]:
        """The history file as a list, or an empty one if it cannot be read."""
        try:
            return self.history.read_text(encoding="utf-8").splitlines()[-HISTORY_LIMIT:]
        except (OSError, UnicodeDecodeError):
            return []

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
        """Write the history back, in the one format both paths read."""
        try:
            self.history.parent.mkdir(parents=True, exist_ok=True)
            if self.editor is not None:
                lines = self.editor.history[-HISTORY_LIMIT:]
                self.history.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
            elif self.readline is not None:
                self.readline.write_history_file(str(self.history))
        except (OSError, ValueError):
            pass

    def read(self, prompt: str = "> ") -> str:
        """Read one line. Raises EOFError on Ctrl-D, as ``input`` does."""
        if self.editor is not None:
            return self.editor.read(prompt)
        return input(prompt)

    def images(self) -> list[str]:
        """Paths attached to the line just read, and never to an earlier one."""
        return list(self.editor.images) if self.editor is not None else []
