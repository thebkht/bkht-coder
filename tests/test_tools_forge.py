"""The `github` and `gitlab` tools: read-only wrappers around a logged-in CLI."""

from __future__ import annotations

import pytest

from bkht.coder.tools import forge
from bkht.coder.tools.base import Registry, ToolError
from bkht.coder.tools.github import GITHUB
from bkht.coder.tools.gitlab import GITLAB


def check(command: str, which=GITHUB):
    return forge.check(which, command)


# --- what gets through --------------------------------------------------------


@pytest.mark.parametrize("command", [
    "run view 33185669396 --log-failed",
    "run list --limit 5",
    "pr view 42 --comments",
    "pr diff 42",
    "issue view 17",
    "repo view",
    "api repos/owner/name/commits",
])
def test_a_read_is_allowed(command):
    assert check(command)[0] in GITHUB.commands


def test_the_program_name_may_be_written_in_as_well():
    # A model told the tool is called `gh` sometimes writes `gh` again. Harmless.
    assert check("gh run list") == ["run", "list"]


def test_gitlab_has_its_own_vocabulary():
    assert check("ci status", GITLAB) == ["ci", "status"]
    with pytest.raises(ToolError):
        check("run list", GITLAB)  # a GitHub word, not a GitLab one


# --- what does not ------------------------------------------------------------


@pytest.mark.parametrize("command", [
    "pr merge 42",
    "run rerun 123",
    "repo delete owner/name",
    "release create v1",
    "issue close 17",
    "pr review 42 --approve",
])
def test_anything_that_writes_is_refused(command):
    # Refused here rather than left to the permission gate: merging a pull
    # request is not a thing to approve one keypress at a time in the middle of
    # a turn that was only supposed to read a log.
    with pytest.raises(ToolError, match="only reads"):
        check(command)


@pytest.mark.parametrize("command", [
    "api --method DELETE repos/x",
    "api -X POST repos/x",
    "api repos/x -f name=value",
])
def test_a_flag_that_turns_a_read_into_a_write_is_refused(command):
    with pytest.raises(ToolError):
        check(command)


def test_an_unknown_top_level_command_is_refused_by_name():
    # An allow-list as well as a deny-list: a deny-list alone would pass every
    # verb the CLI gains after this was written.
    with pytest.raises(ToolError, match="not one of the GitHub commands"):
        check("auth token")


def test_an_empty_command_says_so():
    for command in ("", "   "):
        with pytest.raises(ToolError, match="must not be empty"):
            check(command)


def test_unbalanced_quotes_are_reported_rather_than_guessed_at():
    with pytest.raises(ToolError, match="could not parse"):
        check("pr view 'unclosed")


def test_shell_punctuation_is_never_shell():
    # Split into an argument list and run without a shell, so a `;` is a `;`.
    assert check("run view 1; rm -rf /") == ["run", "view", "1;", "rm", "-rf", "/"]


def test_a_backtick_is_just_a_character():
    assert "`whoami`" in check("run view `whoami`")


# --- registration -------------------------------------------------------------


def test_a_missing_cli_registers_nothing(monkeypatch):
    # A tool the model can see and cannot use is a turn spent finding that out.
    monkeypatch.setattr(forge.shutil, "which", lambda name: None)
    assert forge.register(Registry(), GITHUB).names() == []


def test_an_installed_cli_registers_a_read_only_tool(monkeypatch):
    monkeypatch.setattr(forge.shutil, "which", lambda name: f"/usr/bin/{name}")
    registry = forge.register(Registry(), GITHUB)
    assert registry.names() == ["gh"]
    assert registry.get("gh").mutating is False, "reading a CI log is not a mutation"


def test_the_description_tells_the_model_when_to_reach_for_it(monkeypatch):
    # The session this exists for had `gh` on the PATH the whole time and wrote
    # a curl with a placeholder token instead, because nothing had said so.
    monkeypatch.setattr(forge.shutil, "which", lambda name: f"/usr/bin/{name}")
    described = forge.register(Registry(), GITHUB).get("gh").description
    assert "not a file in the workspace" in described
    assert "--log-failed" in described


def test_both_forges_are_offered_in_plan_mode(monkeypatch):
    from bkht.coder.tools import build_registry

    monkeypatch.setattr(forge.shutil, "which", lambda name: f"/usr/bin/{name}")
    registry, _ = build_registry(".", read_only=True)
    assert "gh" in registry.names() and "glab" in registry.names()
