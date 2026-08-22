"""Remembered permission decisions.

The approval prompt is the right place to decide *this* call. It is the wrong
place to decide the same call for the twentieth time -- and the honest response
to being asked twenty times is ``--auto``, which is the outcome the permission
system exists to avoid.

So a decision can be remembered. What gets remembered is one exact call: this
command, or this path, in this workspace. Not the tool -- approving one ``rm``
must never approve the next one, and a rule that widens itself is worse than no
rule because the user believes they know what they granted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .session import STATE_DIR

RULES_PATH = STATE_DIR / "permissions.json"

ALLOW = "allow"
DENY = "deny"
DECISIONS = (ALLOW, DENY)

# The argument that identifies a call, per tool. Everything else about the call
# is noise for matching: two `edit_file` calls on the same path are the same
# decision even though their diffs differ, and the diff was already shown once.
IDENTIFYING = {
    "bash": "command",
    "powershell": "command",
    "background": "command",
}

MAX_LABEL = 60


def signature(tool_name: str, arguments: dict) -> str:
    """The string that decides whether two calls are the same call.

    Path-taking tools key on the path, shells key on the command, and anything
    else falls back to all of its arguments -- canonically ordered, so a model
    that emits the same call with its keys shuffled still matches.
    """
    key = IDENTIFYING.get(tool_name)
    if key is None and "path" in arguments:
        key = "path"

    if key is not None and isinstance(arguments.get(key), str):
        return arguments[key]

    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def rule_id(scope: str, tool_name: str, sig: str) -> str:
    """A short id that is stable across runs, so it can be revoked later."""
    digest = hashlib.sha256(f"{scope}\x00{tool_name}\x00{sig}".encode()).hexdigest()
    return digest[:8]


@dataclass(frozen=True)
class Rule:
    """One remembered decision about one exact call in one workspace."""

    scope: str
    tool: str
    signature: str
    decision: str

    @property
    def id(self) -> str:
        return rule_id(self.scope, self.tool, self.signature)

    def label(self) -> str:
        """One line, for ``/permissions``."""
        sig = self.signature
        if len(sig) > MAX_LABEL:
            sig = sig[: MAX_LABEL - 1] + "…"
        return f"{self.id}  {self.decision:<5} {self.tool}  {sig}"

    def payload(self) -> dict:
        return {
            "scope": self.scope,
            "tool": self.tool,
            "signature": self.signature,
            "decision": self.decision,
        }


@dataclass
class Rules:
    """Every remembered decision, scoped to one workspace.

    Loading never raises. A permissions file that cannot be read leaves the
    session with no rules and a reason to print -- refusing to start would take
    away the only tool that could fix it, and silently granting nothing while
    claiming to have loaded rules would be worse still.
    """

    scope: str
    path: Path | None = None
    rules: dict[str, Rule] = None
    error: str = ""

    def __post_init__(self) -> None:
        if self.rules is None:
            self.rules = {}
        # Resolved here rather than as a default argument: a default binds at
        # import time, and the tests need to move the store somewhere harmless.
        self.path = Path(self.path) if self.path is not None else RULES_PATH

    @classmethod
    def load(cls, scope: str, path: Path | None = None) -> "Rules":
        """Every rule stored for ``scope``, keyed by id."""
        store = cls(scope=str(scope), path=path)
        try:
            raw = json.loads(store.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return store
        except (OSError, ValueError) as exc:
            store.error = f"could not read {store.path}: {exc}; continuing with no stored rules"
            return store

        for record in raw if isinstance(raw, list) else []:
            rule = _rule_from(record)
            if rule is not None and rule.scope == store.scope:
                store.rules[rule.id] = rule
        return store

    def decide(self, tool_name: str, arguments: dict) -> str | None:
        """``allow``, ``deny``, or None when nothing has been remembered."""
        found = self.rules.get(rule_id(self.scope, tool_name, signature(tool_name, arguments)))
        return found.decision if found else None

    def remember(self, tool_name: str, arguments: dict, decision: str) -> Rule:
        """Store a decision about one exact call and return the rule."""
        if decision not in DECISIONS:
            raise ValueError(f"unknown decision {decision!r}; expected one of {', '.join(DECISIONS)}")
        rule = Rule(
            scope=self.scope,
            tool=tool_name,
            signature=signature(tool_name, arguments),
            decision=decision,
        )
        self.rules[rule.id] = rule
        self._save()
        return rule

    def revoke(self, identifier: str) -> Rule | None:
        """Remove a rule by id, returning it, or None if there was no such id."""
        removed = self.rules.pop(identifier, None)
        if removed is not None:
            self._save()
        return removed

    def listing(self) -> list[Rule]:
        """Rules for this workspace, in a stable order."""
        return sorted(self.rules.values(), key=lambda r: (r.tool, r.signature))

    def _save(self) -> None:
        """Rewrite the file, preserving rules belonging to other workspaces.

        Re-read rather than cached: another session in another directory may
        have stored a rule since this one started, and dropping it because we
        were not watching would be a silent revocation.
        """
        others = []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for record in raw if isinstance(raw, list) else []:
                rule = _rule_from(record)
                if rule is not None and rule.scope != self.scope:
                    others.append(rule)
        except (OSError, ValueError):
            others = []

        payload = [r.payload() for r in others] + [r.payload() for r in self.listing()]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            # Replaced rather than written in place: a crash mid-write would
            # otherwise leave a truncated file, and the next session would start
            # having silently forgotten every rule.
            tmp.replace(self.path)
        except OSError as exc:
            self.error = f"could not write {self.path}: {exc}"


def _rule_from(record: object) -> Rule | None:
    """One stored record as a Rule, or None if it is not one."""
    if not isinstance(record, dict):
        return None
    fields = ("scope", "tool", "signature", "decision")
    if not all(isinstance(record.get(name), str) for name in fields):
        return None
    if record["decision"] not in DECISIONS:
        return None
    return Rule(**{name: record[name] for name in fields})
