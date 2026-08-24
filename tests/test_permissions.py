"""The permission gate, including what the user is actually shown."""

import pytest

from bkht.coder.permissions import ASK, AUTO, PLAN, Permissions, cycle, preview, truncate
from bkht.coder.session import Snapshots
from bkht.coder.tools import build_registry


@pytest.fixture
def parts(project):
    registry, workspace = build_registry(project, snapshots=Snapshots())
    return registry, workspace


class Recorder:
    """A scripted prompt that records what it was shown."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.shown = []

    def __call__(self, question, body):
        self.shown.append(body)
        return self.answers.pop(0) if self.answers else "n"


def test_read_only_tools_never_prompt(parts):
    registry, workspace = parts
    recorder = Recorder()
    policy = Permissions(mode=ASK, workspace=workspace, prompt=recorder)
    assert policy.check(registry.get("read_file"), {"path": "README.md"}).allowed
    assert recorder.shown == []


def test_auto_allows_mutations_without_asking(parts):
    registry, workspace = parts
    recorder = Recorder()
    policy = Permissions(mode=AUTO, workspace=workspace, prompt=recorder)
    assert policy.check(registry.get("bash"), {"command": "rm -rf x"}).allowed
    assert recorder.shown == []


def test_plan_denies_mutations_and_says_why(parts):
    registry, workspace = parts
    policy = Permissions(mode=PLAN, workspace=workspace)
    decision = policy.check(registry.get("write_file"), {"path": "a.py", "content": "x"})
    assert not decision.allowed
    assert "plan mode" in decision.reason


def test_plan_still_allows_reads(parts):
    registry, workspace = parts
    policy = Permissions(mode=PLAN, workspace=workspace)
    assert policy.check(registry.get("grep"), {"pattern": "x"}).allowed


def test_yes_allows_once_and_asks_again(parts):
    registry, workspace = parts
    recorder = Recorder("y", "n")
    policy = Permissions(mode=ASK, workspace=workspace, prompt=recorder)
    tool = registry.get("write_file")
    assert policy.check(tool, {"path": "a.py", "content": "1"}).allowed
    assert not policy.check(tool, {"path": "a.py", "content": "2"}).allowed


def test_always_remembers_the_same_call(parts):
    # The same path again is the same decision: the diff was shown once and
    # answered once, and asking again is the friction that drives people to
    # --auto.
    registry, workspace = parts
    recorder = Recorder("a")
    policy = Permissions(mode=ASK, workspace=workspace, prompt=recorder)
    tool = registry.get("write_file")
    assert policy.check(tool, {"path": "a.py", "content": "1"}).allowed
    assert policy.check(tool, {"path": "a.py", "content": "2"}).allowed
    assert len(recorder.shown) == 1


def test_always_does_not_widen_to_other_calls(parts):
    # The whole point of the rule store: approving one call must never approve
    # a different one made with the same tool.
    registry, workspace = parts
    recorder = Recorder("a", "n")
    policy = Permissions(mode=ASK, workspace=workspace, prompt=recorder)
    tool = registry.get("write_file")
    assert policy.check(tool, {"path": "a.py", "content": "1"}).allowed
    assert not policy.check(tool, {"path": "b.py", "content": "2"}).allowed


def test_always_is_per_tool_not_global(parts):
    registry, workspace = parts
    recorder = Recorder("a", "n")
    policy = Permissions(mode=ASK, workspace=workspace, prompt=recorder)
    assert policy.check(registry.get("write_file"), {"path": "a.py", "content": "1"}).allowed
    assert not policy.check(registry.get("bash"), {"command": "ls"}).allowed


def test_a_remembered_rule_survives_a_new_policy(parts):
    # Persistence is the difference between a rule and a session preference:
    # tomorrow's session must not re-ask what today's already answered.
    registry, workspace = parts
    call = {"command": "uv run pytest -q"}
    first = Permissions(mode=ASK, workspace=workspace, prompt=Recorder("a"))
    assert first.check(registry.get("bash"), call).allowed

    second = Permissions(mode=ASK, workspace=workspace, prompt=Recorder("n"))
    assert second.check(registry.get("bash"), call).allowed
    assert not second.check(registry.get("bash"), {"command": "rm -rf /"}).allowed


def test_a_remembered_denial_refuses_without_asking(parts):
    registry, workspace = parts
    recorder = Recorder()
    policy = Permissions(mode=ASK, workspace=workspace, prompt=recorder)
    policy.rules.remember("bash", {"command": "rm -rf /"}, "deny")

    decision = policy.check(registry.get("bash"), {"command": "rm -rf /"})
    assert not decision.allowed and "Do not retry" in decision.reason
    assert recorder.shown == []


def test_denial_tells_the_model_not_to_retry(parts):
    registry, workspace = parts
    policy = Permissions(mode=ASK, workspace=workspace, prompt=Recorder("n"))
    decision = policy.check(registry.get("bash"), {"command": "ls"})
    assert "Do not retry" in decision.reason


def test_unknown_mode_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown permission mode"):
        Permissions(mode="yolo")


# --- what the user is shown -------------------------------------------------


def test_bash_preview_is_the_command_line(parts):
    registry, workspace = parts
    assert preview(registry.get("bash"), {"command": "pytest -q"}, workspace) == "$ pytest -q"


def test_edit_preview_is_a_diff(parts):
    registry, workspace = parts
    body = preview(
        registry.get("edit_file"),
        {"path": "src/util.py", "old_string": "x * 2", "new_string": "x * 3"},
        workspace,
    )
    assert "-    return x * 2" in body
    assert "+    return x * 3" in body


def test_write_preview_of_a_new_file_is_all_additions(parts):
    registry, workspace = parts
    body = preview(registry.get("write_file"), {"path": "new.py", "content": "a\nb\n"}, workspace)
    assert "+a" in body and "+b" in body


def test_preview_of_a_no_op_says_so(parts):
    registry, workspace = parts
    body = preview(
        registry.get("write_file"),
        {"path": "README.md", "content": (workspace.root / "README.md").read_text()},
        workspace,
    )
    assert "no change" in body


def test_huge_diff_preview_is_bounded_for_display(parts):
    # preview() now returns the whole diff -- the prompt is what bounds it, so
    # that a keypress can expand it -- but what reaches a screen is still small.
    registry, workspace = parts
    body = truncate(
        preview(
            registry.get("write_file"),
            {"path": "big.py", "content": "\n".join(str(i) for i in range(500))},
            workspace,
        )
    )
    assert "more diff lines" in body
    assert len(body.splitlines()) < 50


# --- what the approval prompt shows -----------------------------------------


@pytest.fixture
def package(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "session.py").write_text("STATE_DIR = '/tmp'\n")
    (tmp_path / "pkg" / "commands.py").write_text("from .session import STATE_DIR\n")
    registry, workspace = build_registry(tmp_path)
    return registry, workspace


def test_the_prompt_warns_about_a_name_that_does_not_exist(package):
    """The one moment a person is already deciding.

    "session.py does not define Input" is the whole of what someone needs in
    order to say no, and the prompt is where they are being asked.
    """
    registry, workspace = package
    body = preview(
        registry.get("edit_file"),
        {
            "path": "pkg/commands.py",
            "old_string": "from .session import STATE_DIR",
            "new_string": "from .session import STATE_DIR, Input",
        },
        workspace,
    )
    assert "session.py does not define `Input`" in body


def test_the_diff_still_leads(package):
    registry, workspace = package
    body = preview(
        registry.get("edit_file"),
        {
            "path": "pkg/commands.py",
            "old_string": "from .session import STATE_DIR",
            "new_string": "from .session import STATE_DIR, Input",
        },
        workspace,
    )
    assert body.startswith("---")
    assert body.index("does not define") > body.index("+from .session")


def test_a_sound_edit_shows_only_the_diff(package):
    registry, workspace = package
    body = preview(
        registry.get("edit_file"),
        {
            "path": "pkg/commands.py",
            "old_string": "from .session import STATE_DIR",
            "new_string": "from .session import STATE_DIR as WHERE",
        },
        workspace,
    )
    assert "does not define" not in body


def test_the_modes_cycle_in_order_and_wrap():
    assert cycle(ASK) == AUTO
    assert cycle(AUTO) == PLAN
    assert cycle(PLAN) == ASK


def test_cycling_from_nonsense_lands_on_the_careful_mode():
    # Shift+Tab is not a place to raise: whatever the mode was, the next one
    # has to be a mode, and the safe one is the one to arrive at.
    assert cycle("banana") == ASK
