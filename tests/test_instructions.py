"""Instruction files: discovery, precedence, and the context budget."""

from pathlib import Path

import pytest

from bkht.coder import instructions as I
from bkht.coder.instructions import (
    Instruction,
    load_instructions,
    render,
    summarize,
)


@pytest.fixture
def no_global(monkeypatch, tmp_path):
    """Point the global source at an empty directory.

    Without this the developer's own ~/.bkht-coder/AGENTS.md would leak into
    the tests and make them pass or fail depending on whose machine they run on.
    """
    monkeypatch.setattr(I, "GLOBAL_DIR", tmp_path / "nowhere")


def test_no_files_found_is_empty(tmp_path, no_global):
    assert load_instructions(tmp_path) == []
    assert render([]) == ""
    assert summarize([]) == ""


def test_reads_agents_md(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_text("Use tabs.\n")
    loaded = load_instructions(tmp_path)
    assert [i.text for i in loaded] == ["Use tabs."]
    assert loaded[0].source.endswith("AGENTS.md")


def test_reads_claude_md(tmp_path, no_global):
    (tmp_path / "CLAUDE.md").write_text("Never touch generated files.\n")
    assert [i.text for i in load_instructions(tmp_path)] == [
        "Never touch generated files."
    ]


def test_both_workspace_files_load_in_order(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_text("first")
    (tmp_path / "CLAUDE.md").write_text("second")
    assert [i.text for i in load_instructions(tmp_path)] == ["first", "second"]


def test_global_comes_before_workspace(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "AGENTS.md").write_text("global rule")
    monkeypatch.setattr(I, "GLOBAL_DIR", home)

    root = tmp_path / "project"
    root.mkdir()
    (root / "CLAUDE.md").write_text("project rule")

    # Least specific first: the workspace's own rules are read last, so they
    # are what the model sees most recently.
    assert [i.text for i in load_instructions(root)] == ["global rule", "project rule"]


def test_global_can_be_disabled(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "AGENTS.md").write_text("global rule")
    monkeypatch.setattr(I, "GLOBAL_DIR", home)

    root = tmp_path / "project"
    root.mkdir()
    assert load_instructions(root, include_global=False) == []


def test_empty_file_is_not_a_source(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_text("   \n\n  ")
    assert load_instructions(tmp_path) == []


def test_binary_file_is_skipped_not_fatal(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_bytes(b"rules\x00binary")
    (tmp_path / "CLAUDE.md").write_text("readable")
    # The unreadable source drops out; the agent still starts, with the rules
    # it could read.
    assert [i.text for i in load_instructions(tmp_path)] == ["readable"]


def test_directory_named_like_an_instruction_file_is_skipped(tmp_path, no_global):
    (tmp_path / "AGENTS.md").mkdir()
    assert load_instructions(tmp_path) == []


def test_per_source_truncation_is_announced(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_text("x" * 5000)
    loaded = load_instructions(tmp_path, per_source=100)
    assert loaded[0].truncated is True
    assert I.TRUNCATION_NOTE in loaded[0].text
    assert len(loaded[0].text) < 5000


def test_total_budget_is_spent_from_the_most_specific_source_back(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "AGENTS.md").write_text("g" * 200)
    monkeypatch.setattr(I, "GLOBAL_DIR", home)

    root = tmp_path / "project"
    root.mkdir()
    (root / "CLAUDE.md").write_text("p" * 200)

    # Only the workspace file fits. A long global file must not crowd out the
    # rules that belong to the project actually being worked on.
    loaded = load_instructions(root, budget=200, per_source=1000)
    assert [i.text for i in loaded] == ["p" * 200]


def test_a_source_that_partly_fits_is_kept_and_marked(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "AGENTS.md").write_text("g" * 200)
    monkeypatch.setattr(I, "GLOBAL_DIR", home)

    root = tmp_path / "project"
    root.mkdir()
    (root / "CLAUDE.md").write_text("p" * 100)

    loaded = load_instructions(root, budget=150, per_source=1000)
    assert len(loaded) == 2
    assert loaded[1].text == "p" * 100
    assert loaded[0].truncated is True


def test_home_relative_sources_are_labelled_with_a_tilde(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "AGENTS.md").write_text("global rule")
    monkeypatch.setattr(I, "GLOBAL_DIR", home)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    loaded = load_instructions(tmp_path / "project")
    assert loaded[0].source == "~/home/AGENTS.md"


def test_render_names_every_source(tmp_path, no_global):
    (tmp_path / "AGENTS.md").write_text("alpha")
    (tmp_path / "CLAUDE.md").write_text("beta")
    text = render(load_instructions(tmp_path))
    assert "alpha" in text and "beta" in text
    assert text.count("From ") == 2


def test_summarize_marks_truncation():
    assert summarize([Instruction("a.md", "x")]) == "instructions: a.md"
    assert "truncated" in summarize([Instruction("a.md", "x", truncated=True)])


# --- wiring -----------------------------------------------------------------


def _repl(project, use_instructions=True):
    from bkht.coder.agent import Agent
    from bkht.coder.permissions import ASK, Permissions
    from bkht.coder.repl import Repl
    from bkht.coder.session import Session, Snapshots
    from bkht.coder.tools import build_registry

    from fakes import FakeProvider

    snapshots = Snapshots()
    registry, workspace = build_registry(project, snapshots=snapshots)
    permissions = Permissions(mode=ASK, workspace=workspace, prompt=lambda q, b: "n")
    session = Session(system="sys", cwd=str(project), model="fake")
    agent = Agent(FakeProvider([]), registry, session, permissions=permissions)

    lines = []
    repl = Repl(
        agent, snapshots, permissions, workspace,
        out=lines.append, use_instructions=use_instructions,
    )
    return repl, lines


def test_system_prompt_omits_the_section_when_there_is_nothing_to_say(project):
    from bkht.coder.prompts import system_prompt
    from bkht.coder.tools import build_registry

    registry, workspace = build_registry(project)
    assert "Project instructions" not in system_prompt(registry, str(project))


def test_system_prompt_carries_instructions_and_survives_braces(project):
    from bkht.coder.prompts import system_prompt
    from bkht.coder.tools import build_registry

    registry, workspace = build_registry(project)
    # Instruction files are prose and routinely contain braces; formatting them
    # would raise or, worse, silently substitute.
    prompt = system_prompt(registry, str(project), "", "Prefer f'{x}' over format().")
    assert "Prefer f'{x}' over format()." in prompt

    # The tool protocol has to stay last: drifting off the emission format is
    # this model's characteristic failure.
    assert prompt.index("Project instructions") < prompt.index("# Calling a tool")


def test_make_agent_loads_workspace_instructions(project, monkeypatch, tmp_path, no_global):
    from bkht.coder import session as session_module
    from bkht.coder.cli import build_parser, make_agent

    monkeypatch.setattr(session_module, "STATE_DIR", tmp_path / "state")
    (project / "CLAUDE.md").write_text("MARKER-ALPHA")

    args = build_parser().parse_args(["--cwd", str(project)])
    agent, *_ = make_agent(args)
    assert "MARKER-ALPHA" in agent.session.system


def test_make_agent_honours_no_instructions(project, monkeypatch, tmp_path, no_global):
    from bkht.coder import session as session_module
    from bkht.coder.cli import build_parser, make_agent

    monkeypatch.setattr(session_module, "STATE_DIR", tmp_path / "state")
    (project / "CLAUDE.md").write_text("MARKER-ALPHA")

    args = build_parser().parse_args(["--cwd", str(project), "--no-instructions"])
    agent, *_ = make_agent(args)
    assert "MARKER-ALPHA" not in agent.session.system


def test_instructions_command_reports_nothing_found(project, no_global):
    repl, lines = _repl(project)
    repl.dispatch("/instructions")
    assert "No AGENTS.md or CLAUDE.md found" in lines[0]


def test_instructions_command_lists_sources(project, no_global):
    (project / "CLAUDE.md").write_text("MARKER-BETA")
    repl, lines = _repl(project)
    repl.dispatch("/instructions")
    assert any("CLAUDE.md" in line for line in lines)
    assert any("MARKER-BETA" in line for line in lines)


def test_instructions_command_says_when_disabled(project, no_global):
    (project / "CLAUDE.md").write_text("MARKER-BETA")
    repl, lines = _repl(project, use_instructions=False)
    repl.dispatch("/instructions")
    assert "disabled" in lines[0]
    assert not any("MARKER-BETA" in line for line in lines)


def test_reload_rebuilds_the_system_prompt(project, no_global):
    repl, lines = _repl(project)
    assert "MARKER-GAMMA" not in repl.agent.session.system

    # The file appears after the session started, which is the case reload
    # exists for: editing the rules without losing the conversation.
    (project / "CLAUDE.md").write_text("MARKER-GAMMA")
    repl.dispatch("/instructions reload")
    assert "MARKER-GAMMA" in repl.agent.session.system
    assert any("Reloaded" in line for line in lines)


def test_workspace_sources_are_named_relative_to_the_root(tmp_path, no_global):
    (tmp_path / "CLAUDE.md").write_text("rule")
    assert load_instructions(tmp_path)[0].source == "CLAUDE.md"
