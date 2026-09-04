"""Subagents: discovery, what is refused, and how `task` offers them."""

import pytest

from bkht.coder import subagents as subagents_module
from bkht.coder.subagents import discover, roster, summarize
from bkht.coder.tools import build_registry
from bkht.coder.tools.base import ToolError


@pytest.fixture(autouse=True)
def no_global_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(subagents_module.layout, "GLOBAL_ROOT", tmp_path / "nowhere")


def write_subagent(root, name, description="Reviews a diff.", instructions=None):
    directory = root / "agent" / "subagents" / name
    directory.mkdir(parents=True, exist_ok=True)
    (root / "agent" / "agent.json").write_text("{}")
    (directory / "agent.md").write_text(f"---\ndescription: {description}\n---\n")
    if instructions is not None:
        (directory / "instructions.md").write_text(instructions)
    return directory


# --- discovery ------------------------------------------------------------


def test_the_directory_names_the_subagent(tmp_path):
    write_subagent(tmp_path, "reviewer")

    found = discover(tmp_path)
    assert found.names() == ["reviewer"]
    assert found.get("reviewer").description == "Reviews a diff."


def test_a_subagent_without_a_description_is_refused_and_said_so(tmp_path):
    # Without one the model has nothing to choose on.
    directory = tmp_path / "agent" / "subagents" / "reviewer"
    directory.mkdir(parents=True)
    (tmp_path / "agent" / "agent.json").write_text("{}")
    (directory / "agent.md").write_text("Just prose.\n")

    found = discover(tmp_path)
    assert not found.agents and "description" in found.problems[0]


def test_a_directory_without_an_agent_file_is_not_a_subagent(tmp_path):
    (tmp_path / "agent" / "subagents" / "notes").mkdir(parents=True)
    (tmp_path / "agent" / "agent.json").write_text("{}")

    assert discover(tmp_path) == subagents_module.Found()


def test_its_instructions_and_skills_are_its_own(tmp_path):
    directory = write_subagent(tmp_path, "reviewer", instructions="Read the diff first.")
    (directory / "skills").mkdir()
    (directory / "skills" / "house.md").write_text(
        "---\ndescription: House review rules.\n---\n\nSmall diffs.\n"
    )

    reviewer = discover(tmp_path).get("reviewer")
    assert reviewer.instructions == "Read the diff first."
    assert [skill.name for skill in reviewer.skills.skills] == ["house"]


def test_the_workspaces_own_skills_are_not_inherited(tmp_path):
    # A reviewer that quietly got every skill in the project would be the
    # parent agent with a different name on it.
    write_subagent(tmp_path, "reviewer")
    workspace_skill = tmp_path / "agent" / "skills" / "releasing"
    workspace_skill.mkdir(parents=True)
    (workspace_skill / "SKILL.md").write_text(
        "---\nname: releasing\ndescription: Cut a release.\n---\n\nTag it.\n"
    )

    assert discover(tmp_path).get("reviewer").skills.skills == []


def test_an_unmarked_agent_directory_contributes_no_subagents(tmp_path):
    directory = tmp_path / "agent" / "subagents" / "theirs"
    directory.mkdir(parents=True)
    (directory / "agent.md").write_text("---\ndescription: Somebody else's.\n---\n")

    assert not discover(tmp_path).agents


def test_the_roster_is_what_the_model_chooses_on(tmp_path):
    write_subagent(tmp_path, "reviewer", description="Reviews a diff.")
    assert roster(discover(tmp_path)) == "- reviewer: Reviews a diff."


def test_the_summary_names_what_loaded_and_what_did_not(tmp_path):
    write_subagent(tmp_path, "reviewer")
    assert "subagents: reviewer" in summarize(discover(tmp_path))


# --- the task tool --------------------------------------------------------


def registry_with(tmp_path, found):
    registry, _ = build_registry(
        tmp_path,
        delegate={"provider": object(), "subagents": found},
    )
    return registry


def test_the_task_tool_offers_the_specialists_by_name(tmp_path):
    write_subagent(tmp_path, "reviewer")
    task = registry_with(tmp_path, discover(tmp_path)).get("task")

    assert task.parameters["properties"]["agent"]["enum"] == ["reviewer"]
    assert "reviewer: Reviews a diff." in task.description


def test_with_no_subagents_the_schema_is_untouched(tmp_path):
    # A parameter offering a choice of nothing can only be got wrong.
    task = registry_with(tmp_path, discover(tmp_path)).get("task")
    assert "agent" not in task.parameters["properties"]


def test_an_unknown_specialist_is_named_back_with_the_list(tmp_path):
    write_subagent(tmp_path, "reviewer")
    task = registry_with(tmp_path, discover(tmp_path)).get("task")

    with pytest.raises(ToolError, match="Available: reviewer"):
        task.run(instruction="review this", agent="reviwer")


def test_a_specialists_instructions_and_skills_reach_its_prompt(tmp_path, monkeypatch):
    # The whole point of naming one: the delegated turn runs on that agent's
    # standing rules rather than on an empty prompt.
    from bkht.coder.tools.task import register_task_tool
    from bkht.coder.tools.base import Registry
    from fakes import FakeProvider

    directory = write_subagent(tmp_path, "reviewer", instructions="Read the diff first.")
    (directory / "skills").mkdir()
    (directory / "skills" / "house.md").write_text(
        "---\ndescription: House review rules.\n---\n\nSmall diffs.\n"
    )

    provider = FakeProvider(["The diff is fine."])
    registry = Registry()
    register_task_tool(
        registry, tmp_path, provider, subagents=discover(tmp_path), iterations=2
    )

    result = registry.get("task").run(instruction="review it", agent="reviewer")
    assert result.ok
    system = provider.calls[0][0]["content"]
    assert "Read the diff first." in system and "house" in system


def test_a_general_delegation_still_runs_on_the_sessions_skills(tmp_path):
    from bkht.coder.tools.task import register_task_tool
    from bkht.coder.tools.base import Registry
    from bkht.coder.skills import discover as discover_skills
    from fakes import FakeProvider

    write_subagent(tmp_path, "reviewer")
    workspace_skill = tmp_path / "agent" / "skills" / "releasing"
    workspace_skill.mkdir(parents=True)
    (workspace_skill / "SKILL.md").write_text(
        "---\nname: releasing\ndescription: Cut a release.\n---\n\nTag it.\n"
    )

    provider = FakeProvider(["It builds the registry."])
    registry = Registry()
    register_task_tool(
        registry, tmp_path, provider,
        skills=discover_skills(tmp_path, include_global=False),
        subagents=discover(tmp_path), iterations=2,
    )

    registry.get("task").run(instruction="what builds the registry?")
    assert "releasing" in provider.calls[0][0]["content"]
