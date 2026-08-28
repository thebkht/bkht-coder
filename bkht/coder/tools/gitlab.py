"""The `gitlab` tool: read GitLab through the `glab` CLI.

The same tool as `github.py`, against the other host. This package already
reviews changes inside GitLab CI -- `coder review --ci gitlab` writes a code
quality report for the merge request widget -- so a session on a GitLab project
should be able to read the pipeline it is running under.

`glab` is not installed on most machines. That is handled by not registering
the tool at all, rather than by offering one that fails on first use.
"""

from __future__ import annotations

from .base import Registry
from .forge import Forge, register

GITLAB = Forge(
    name="glab",
    label="GitLab",
    commands=frozenset({
        "ci", "mr", "issue", "repo", "release", "pipeline", "job",
        "api", "label", "snippet", "cluster", "schedule",
    }),
    examples=(
        "Examples:\n"
        "  ci status                    the pipeline for this branch\n"
        "  ci get --pipeline-id 123     one pipeline\n"
        "  ci trace <job>               a job's log\n"
        "  mr view 42 --comments        a merge request and its review\n"
        "  mr diff 42                   what it changes\n"
        "  issue view 17                an issue"
    ),
    login="glab auth login",
)


def register_gitlab_tool(registry: Registry) -> Registry:
    """Add `glab` to the registry when it is installed."""
    return register(registry, GITLAB)
