"""Skill discovery, the prompt listing, and the tool that fetches a body."""

import pytest

from bkht.coder import skills as skills_module
from bkht.coder.session import Snapshots
from bkht.coder.skills import (
    MAX_BODY,
    Discovery,
    body,
    discover,
    parse_frontmatter,
    render,
    summarize,
)
from bkht.coder.tools import build_registry
from bkht.coder.tools.base import ToolError


def write_skill(root, name, description="Does a thing.", text="Do the thing.", where=".bkht-coder/skills"):
    directory = root / where / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{text}\n"
    )
    return directory


# --- frontmatter ----------------------------------------------------------


def test_frontmatter_and_body_are_separated():
    meta, text = parse_frontmatter("---\nname: a\ndescription: b\n---\n\nBody here.\n")
    assert meta == {"name": "a", "description": "b"}
    assert text == "Body here."


def test_quotes_are_stripped():
    meta, _ = parse_frontmatter('---\nname: "a"\ndescription: \'b\'\n---\nx')
    assert meta == {"name": "a", "description": "b"}


def test_a_folded_description_becomes_one_line():
    meta, _ = parse_frontmatter("---\nname: a\ndescription: >\n  one\n  two\n---\nx")
    assert meta["description"] == "one two"


def test_unknown_keys_are_ignored_not_fatal():
    # A skill written for another agent still loads.
    meta, _ = parse_frontmatter("---\nname: a\nallowed-tools: bash\ndescription: b\n---\nx")
    assert meta["name"] == "a" and meta["description"] == "b"


def test_a_file_without_frontmatter_is_all_body():
    meta, text = parse_frontmatter("Just prose.\n")
    assert meta == {} and text == "Just prose.\n"


def test_an_unclosed_delimiter_does_not_swallow_the_file():
    meta, text = parse_frontmatter("---\nname: a\n\nstill prose\n")
    assert meta == {} and "still prose" in text


# --- discovery ------------------------------------------------------------


def test_a_workspace_skill_is_found(project):
    write_skill(project, "releasing", description="How to cut a release.")
    found = discover(project)
    assert [s.name for s in found.skills] == ["releasing"]
    assert found.get("releasing").description == "How to cut a release."


def test_a_claude_skill_is_read_for_compatibility(project):
    write_skill(project, "testing", where=".claude/skills")
    assert discover(project).get("testing") is not None


def test_the_workspaces_own_skill_wins_a_name_collision(project):
    write_skill(project, "testing", description="from claude", where=".claude/skills")
    write_skill(project, "testing", description="from bkht")
    assert discover(project).get("testing").description == "from bkht"


def test_a_global_skill_applies_everywhere(project, tmp_path, monkeypatch):
    monkeypatch.setattr(skills_module, "GLOBAL_ROOT", tmp_path / "global")
    write_skill(tmp_path / "global", "everywhere", where=".")
    assert discover(project).get("everywhere") is not None


def test_a_global_skill_can_be_left_out(project, tmp_path, monkeypatch):
    monkeypatch.setattr(skills_module, "GLOBAL_ROOT", tmp_path / "global")
    write_skill(tmp_path / "global", "everywhere", where=".")
    assert discover(project, include_global=False).get("everywhere") is None


def test_a_skill_without_a_description_is_skipped_and_named(project):
    directory = project / ".bkht-coder" / "skills" / "broken"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: broken\n---\n\nBody.\n")

    found = discover(project)
    assert found.skills == []
    assert "no 'description'" in found.problems[0]
    assert "broken" in found.problems[0]


def test_a_skill_with_an_unusable_name_is_skipped(project):
    # The name is an argument the model has to type back exactly.
    directory = project / ".bkht-coder" / "skills" / "bad"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: two words\ndescription: b\n---\nx")
    assert "not a usable skill name" in discover(project).problems[0]


def test_a_directory_without_a_skill_file_is_not_a_skill(project):
    (project / ".bkht-coder" / "skills" / "empty").mkdir(parents=True)
    found = discover(project)
    assert found.skills == [] and found.problems == []


def test_discovery_does_not_recurse(project):
    write_skill(project, "top")
    nested = project / ".bkht-coder" / "skills" / "top" / "deeper"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text("---\nname: deeper\ndescription: b\n---\nx")
    assert [s.name for s in discover(project).skills] == ["top"]


def test_a_workspace_with_no_skills_finds_nothing(project):
    assert not discover(project)


def test_summarize_names_what_loaded_and_what_did_not(project):
    write_skill(project, "good")
    directory = project / ".bkht-coder" / "skills" / "bad"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: bad\n---\nx")

    line = summarize(discover(project))
    assert "good" in line and "skipped" in line


# --- the prompt listing ---------------------------------------------------


def test_the_listing_carries_names_and_descriptions(project):
    write_skill(project, "releasing", description="How to cut a release.")
    block = render(discover(project))
    assert "releasing" in block and "How to cut a release." in block


def test_no_skills_means_no_block(project):
    assert render(discover(project)) == ""


def test_the_listing_is_capped_and_says_so(project):
    for index in range(40):
        write_skill(project, f"skill{index}", description="x" * 100)

    block = render(discover(project))
    assert "omitted" in block
    assert len(block) < 1200


def test_a_long_description_is_shortened(project):
    write_skill(project, "wordy", description="y" * 500)
    assert len(discover(project).get("wordy").description) <= skills_module.MAX_DESCRIPTION


# --- the tool -------------------------------------------------------------


@pytest.fixture
def registry_with(project):
    def build():
        return build_registry(project, snapshots=Snapshots(), skills=discover(project))[0]

    return build


def test_the_tool_is_absent_when_there_are_no_skills(registry_with):
    # No skills, no tool: an extra tool the model can never use successfully is
    # one more wrong answer available at every step.
    assert "skill" not in registry_with()


def test_the_tool_appears_once_a_skill_exists(project, registry_with):
    write_skill(project, "releasing")
    assert "skill" in registry_with()


def test_the_tool_returns_the_body_without_frontmatter(project, registry_with):
    write_skill(project, "releasing", text="Bump the version first.")
    result = registry_with().get("skill").run(name="releasing")
    assert "Bump the version first." in result.content
    assert "description:" not in result.content


def test_the_name_is_matched_case_insensitively(project, registry_with):
    write_skill(project, "releasing")
    assert registry_with().get("skill").run(name="Releasing").ok


def test_an_unknown_name_lists_the_real_ones(project, registry_with):
    write_skill(project, "releasing")
    with pytest.raises(ToolError, match="releasing"):
        registry_with().get("skill").run(name="nope")


def test_a_resource_beside_the_skill_can_be_read(project, registry_with):
    directory = write_skill(project, "releasing")
    (directory / "checklist.md").write_text("1. tag it\n")
    result = registry_with().get("skill").run(name="releasing", resource="checklist.md")
    assert "tag it" in result.content


def test_a_resource_cannot_escape_the_skill_directory(project, registry_with):
    write_skill(project, "releasing")
    (project / "secret.txt").write_text("nope\n")
    with pytest.raises(ToolError, match="outside the skill"):
        registry_with().get("skill").run(name="releasing", resource="../../../secret.txt")


def test_a_missing_resource_says_which_one(project, registry_with):
    write_skill(project, "releasing")
    with pytest.raises(ToolError, match="no file named"):
        registry_with().get("skill").run(name="releasing", resource="absent.md")


def test_a_huge_body_is_truncated_and_the_cut_announced(project):
    write_skill(project, "long", text="x" * (MAX_BODY * 2))
    text = body(discover(project).get("long"))
    assert len(text) < MAX_BODY + 200
    assert "truncated" in text


def test_the_tool_is_not_permission_gated(project, registry_with):
    # Reading instructions the user wrote changes nothing.
    write_skill(project, "releasing")
    assert not registry_with().get("skill").mutating


def test_an_empty_discovery_is_falsy():
    assert not Discovery()


# --- placement in the system prompt ---------------------------------------


def test_the_block_sits_above_the_tool_protocol(project, registry_with):
    from bkht.coder.prompts import system_prompt

    write_skill(project, "releasing", description="How to cut a release.")
    found = discover(project)
    prompt = system_prompt(
        registry_with(), str(project), "", "Use pytest.", render(found)
    )

    assert "# Skills" in prompt and "releasing" in prompt
    # Instructions, then skills, then the protocol -- which has to stay last,
    # because drifting off the emission format is this model's characteristic
    # failure.
    assert prompt.index("Project instructions") < prompt.index("# Skills")
    assert prompt.index("# Skills") < prompt.index("# Calling a tool")


def test_a_description_with_braces_survives_the_prompt(project, registry_with):
    from bkht.coder.prompts import system_prompt

    write_skill(project, "formatting", description="Prefer f'{x}' over format().")
    prompt = system_prompt(
        registry_with(), str(project), "", "", render(discover(project))
    )
    assert "Prefer f'{x}' over format()." in prompt


def test_no_skills_means_no_section(project, registry_with):
    from bkht.coder.prompts import system_prompt

    assert "# Skills" not in system_prompt(registry_with(), str(project))


# --- the agent/ slot ------------------------------------------------------


@pytest.fixture
def no_global_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(skills_module.layout, "GLOBAL_ROOT", tmp_path / "nowhere")


def marked(root):
    directory = root / "agent"
    (directory / "skills").mkdir(parents=True, exist_ok=True)
    (directory / "agent.json").write_text("{}")
    return directory / "skills"


def test_a_skill_under_agent_is_found(tmp_path, no_global_agent):
    marked(tmp_path)
    write_skill(tmp_path, "releasing", where="agent/skills")

    found = discover(tmp_path, include_global=False)
    assert [skill.name for skill in found.skills] == ["releasing"]


def test_a_flat_markdown_file_is_a_skill_named_for_its_path(tmp_path, no_global_agent):
    # eve's shape: the file is the skill, and where it is decides its name.
    (marked(tmp_path) / "summarize.md").write_text(
        "---\ndescription: Summarise a diff.\n---\n\nRead it, then say what changed.\n"
    )

    found = discover(tmp_path, include_global=False)
    assert [skill.name for skill in found.skills] == ["summarize"]
    assert body(found.get("summarize")) == "Read it, then say what changed."


def test_a_flat_skill_still_needs_a_description(tmp_path, no_global_agent):
    (marked(tmp_path) / "summarize.md").write_text("Just prose, no frontmatter.\n")

    found = discover(tmp_path, include_global=False)
    assert not found.skills and found.problems


def test_frontmatter_wins_over_the_filename(tmp_path, no_global_agent):
    # A skill borrowed from another agent keeps the name it was written under.
    (marked(tmp_path) / "summarize.md").write_text(
        "---\nname: digest\ndescription: Summarise a diff.\n---\n\nRead it.\n"
    )

    assert [s.name for s in discover(tmp_path, include_global=False).skills] == ["digest"]


def test_agent_skills_shadow_the_older_root(tmp_path, no_global_agent):
    marked(tmp_path)
    write_skill(tmp_path, "releasing", description="The old one.")
    write_skill(tmp_path, "releasing", description="The new one.", where="agent/skills")

    found = discover(tmp_path, include_global=False)
    assert found.get("releasing").description == "The new one."


def test_an_unmarked_agent_directory_contributes_no_skills(tmp_path, no_global_agent):
    (tmp_path / "agent" / "skills" / "theirs").mkdir(parents=True)
    (tmp_path / "agent" / "skills" / "theirs" / "SKILL.md").write_text(
        "---\nname: theirs\ndescription: Somebody else's.\n---\n\nx\n"
    )

    assert not discover(tmp_path, include_global=False).skills


def test_a_flat_skill_refuses_to_read_its_neighbours(tmp_path, no_global_agent):
    # Its neighbours are other skills, not its resources.
    skills = marked(tmp_path)
    (skills / "summarize.md").write_text("---\ndescription: Summarise.\n---\n\nDo it.\n")
    (skills / "secret.md").write_text("---\ndescription: Other.\n---\n\nElsewhere.\n")

    found = discover(tmp_path, include_global=False)
    registry, _ = build_registry(tmp_path, skills=found)
    with pytest.raises(ToolError, match="single file"):
        registry.get("skill").run(name="summarize", resource="secret.md")


def test_a_directory_skill_still_ships_its_files(tmp_path, no_global_agent):
    marked(tmp_path)
    directory = write_skill(tmp_path, "releasing", where="agent/skills")
    (directory / "checklist.md").write_text("1. Tag it.\n")

    found = discover(tmp_path, include_global=False)
    registry, _ = build_registry(tmp_path, skills=found)
    result = registry.get("skill").run(name="releasing", resource="checklist.md")
    assert "Tag it" in result.content
