"""Slash commands and the shell escape."""

import pytest

from bkht.coder.agent import Agent
from bkht.coder.permissions import ASK, AUTO, PLAN, Permissions
from bkht.coder.repl import Repl
from bkht.coder.session import Session, Snapshots
from bkht.coder.tools import build_registry

from fakes import FakeProvider


@pytest.fixture
def repl(project):
    snapshots = Snapshots()
    registry, workspace = build_registry(project, snapshots=snapshots)
    permissions = Permissions(mode=ASK, workspace=workspace, prompt=lambda q, b: "n")
    session = Session(system="sys", cwd=str(project), model="fake")
    agent = Agent(FakeProvider([]), registry, session, permissions=permissions)

    lines = []
    return Repl(agent, snapshots, permissions, workspace, out=lines.append), lines


def test_plain_text_is_a_task_not_a_command(repl):
    r, _ = repl
    command = r.dispatch("fix the bug in main.py")
    assert not command.handled and command.task == "fix the bug in main.py"


def test_blank_input_is_ignored(repl):
    r, _ = repl
    assert r.dispatch("   ").handled and r.dispatch("   ").task is None


def test_unknown_command_says_so(repl):
    r, lines = repl
    r.dispatch("/nope")
    assert "Unknown command /nope" in lines[0]


def test_bare_exit_and_quit_leave_without_the_slash(repl):
    r, _ = repl
    for line in ("exit", "quit", "EXIT", "  exit  "):
        assert r.dispatch(line).quit, f"{line!r} should leave"


def test_exit_inside_a_sentence_is_still_a_task(repl):
    # "exit" alone leaves; a request that merely mentions it is work.
    r, _ = repl
    command = r.dispatch("add an exit command")
    assert not command.quit
    assert command.task == "add an exit command"


def test_exit_quits(repl):
    r, _ = repl
    assert r.dispatch("/exit").quit


def test_help_lists_every_command(repl):
    r, lines = repl
    r.dispatch("/help")
    for name in ("/tools", "/context", "/clear", "/undo", "/diff", "/model", "/mode", "/exit"):
        assert name in lines[0]


def test_tools_marks_the_mutating_ones(repl):
    r, lines = repl
    r.dispatch("/tools")
    text = "\n".join(lines)
    assert "read_file" in text
    assert "write_file (needs permission)" in text


def test_context_reports_usage_and_mode(repl):
    r, lines = repl
    r.agent.session.record_usage(4096, 100)
    r.dispatch("/context")
    text = "\n".join(lines)
    assert "4096 / 32768" in text and "12%" in text
    assert "ask" in text


def test_context_reports_whether_the_scout_is_on(repl, project):
    r, lines = repl
    r.dispatch("/context")
    assert "scout      off" in "\n".join(lines)

    r.agent.scout_root = project
    lines.clear()
    r.dispatch("/context")
    assert "scout      on" in "\n".join(lines)


def test_clear_drops_history_but_not_the_workspace(repl, project):
    r, lines = repl
    r.agent.session.add_user("hello")
    r.dispatch("/clear")
    assert r.agent.session.messages == []
    assert (project / "README.md").exists()


def test_undo_restores_the_last_change(repl, project):
    r, lines = repl
    before = (project / "README.md").read_text()
    r.snapshots.capture(project / "README.md")
    (project / "README.md").write_text("clobbered")

    r.dispatch("/undo")
    assert (project / "README.md").read_text() == before
    assert lines[-1] == "restored README.md"


def test_undo_with_nothing_to_undo(repl):
    r, lines = repl
    r.dispatch("/undo")
    assert lines[-1] == "Nothing to undo."


def test_model_shows_and_switches(repl):
    r, lines = repl
    r.dispatch("/model")
    assert lines[-1] == "fake"
    r.dispatch("/model qwen2.5-coder:7b")
    assert r.agent.provider.model == "qwen2.5-coder:7b"
    assert r.agent.session.model == "qwen2.5-coder:7b"


def test_mode_shows_and_switches(repl):
    r, lines = repl
    r.dispatch("/mode")
    assert lines[-1] == "ask"
    r.dispatch("/mode plan")
    assert r.permissions.mode == PLAN


def test_mode_rejects_an_unknown_value(repl):
    r, lines = repl
    r.dispatch("/mode yolo")
    assert "Unknown mode" in lines[-1]
    assert r.permissions.mode == ASK


def test_remembered_rules_survive_a_mode_switch(repl):
    # Unlike the old session-wide grant, a stored rule is a deliberate decision
    # about one call. Switching modes is not a reason to forget it.
    r, _ = repl
    r.permissions.rules.remember("bash", {"command": "ls"}, "allow")
    r.dispatch("/mode auto")
    r.dispatch("/mode ask")
    assert r.permissions.rules.decide("bash", {"command": "ls"}) == "allow"


def test_permissions_lists_rules_with_ids(repl):
    r, lines = repl
    rule = r.permissions.rules.remember("bash", {"command": "ls"}, "allow")
    lines.clear()
    r.dispatch("/permissions")
    assert rule.id in lines[0] and "ls" in lines[0]


def test_permissions_with_no_rules_says_so(repl):
    r, lines = repl
    r.dispatch("/permissions")
    assert "No remembered decisions" in lines[0]


def test_permissions_remember_stores_without_running(repl):
    r, lines = repl
    r.dispatch('/permissions remember allow bash {"command": "ls"}')
    assert r.permissions.rules.decide("bash", {"command": "ls"}) == "allow"
    assert "Remembered" in lines[0]


def test_permissions_remember_rejects_an_unknown_tool(repl):
    r, lines = repl
    r.dispatch('/permissions remember allow nope {"command": "ls"}')
    assert "No tool named nope" in lines[0]


def test_permissions_remember_rejects_arguments_the_tool_would_reject(repl):
    # A rule whose arguments can never match a real call is worse than no rule:
    # it looks like a grant and behaves like nothing.
    r, lines = repl
    r.dispatch('/permissions remember allow bash {"cmd": "ls"}')
    assert "unknown argument" in lines[0]
    assert r.permissions.rules.listing() == []


def test_permissions_remember_rejects_malformed_json(repl):
    r, lines = repl
    r.dispatch("/permissions remember allow bash {not json}")
    assert "must be JSON" in lines[0]


def test_permissions_revoke_removes_a_rule(repl):
    r, lines = repl
    rule = r.permissions.rules.remember("bash", {"command": "ls"}, "allow")
    lines.clear()
    r.dispatch(f"/permissions revoke {rule.id}")
    assert "Revoked" in lines[0]
    assert r.permissions.rules.decide("bash", {"command": "ls"}) is None


def test_permissions_revoke_of_an_unknown_id_says_so(repl):
    r, lines = repl
    r.dispatch("/permissions revoke deadbeef")
    assert "No rule with id deadbeef" in lines[0]


def test_permissions_usage_on_a_bad_verb(repl):
    r, lines = repl
    r.dispatch("/permissions forget everything")
    assert "Usage: /permissions" in lines[0]


def test_diff_on_a_non_repo_reports_the_failure(repl):
    r, lines = repl
    r.dispatch("/diff")
    assert lines[-1]  # git says something; the point is it does not raise


def test_shell_escape_runs_without_the_model(repl, project):
    r, _ = repl
    r.dispatch("!touch made-by-hand.txt")
    assert (project / "made-by-hand.txt").exists()


def test_shell_escape_uses_the_resolved_shell_not_the_platform_default(repl, monkeypatch):
    # `shell=True` means cmd.exe on Windows, which would disagree with the
    # shell the model was told about.
    import bkht.coder.repl as repl_module

    seen = {}
    monkeypatch.setattr(
        repl_module.subprocess, "run", lambda argv, **kw: seen.update(argv=argv)
    )
    r, _ = repl
    r.dispatch("!echo hi")
    assert seen["argv"] == [*repl_module.resolve_shell().argv, "echo hi"]


def test_bare_bang_explains_itself(repl):
    r, lines = repl
    r.dispatch("!")
    assert "Usage: !<command>" in lines[-1]


# --- argument routing -------------------------------------------------------


def test_a_free_text_prompt_is_not_mistaken_for_a_subcommand():
    from bkht.coder.cli import build_agent_parser

    # `coder "add a --verbose flag"` used to fail as an invalid subcommand
    # choice, which broke the documented one-shot form.
    args = build_agent_parser().parse_args(["what does calc.py do?"])
    assert args.prompt == ["what does calc.py do?"]


def test_review_still_reaches_its_subparser():
    from bkht.coder.cli import build_parser

    args = build_parser().parse_args(["review", "--staged"])
    assert args.command == "review" and args.staged
