"""The `github` tool: read GitHub through the `gh` CLI.

The session that put this here was asked to review a CI run. It had no way to
fetch one, so it wrote `curl -H 'Authorization: token YOUR_GITHUB_TOKEN'`,
authenticated as nobody, and spent the rest of the turn diagnosing the
credentials problem it had just invented -- while `gh`, installed and logged
in, sat on the PATH the whole time.

A shell tool could reach it, and did not, because nothing had said so. Naming
the capability is most of what makes it get used.
"""

from __future__ import annotations

from .base import Registry
from .forge import Forge, register

GITHUB = Forge(
    name="gh",
    label="GitHub",
    commands=frozenset({
        "run", "pr", "issue", "repo", "release", "workflow", "search",
        "api", "label", "browse", "status", "cache", "variable", "ruleset",
    }),
    examples=(
        "Examples:\n"
        "  run view 33185669396 --log-failed   why a CI run failed\n"
        "  run list --limit 5                  recent runs\n"
        "  pr view 42 --comments               a pull request and its review\n"
        "  pr diff 42                          what it changes\n"
        "  issue view 17                       an issue\n"
        "  api repos/{owner}/{repo}/commits    anything else, read-only"
    ),
    login="gh auth login",
)


def register_github_tool(registry: Registry) -> Registry:
    """Add `gh` to the registry when it is installed."""
    return register(registry, GITHUB)
