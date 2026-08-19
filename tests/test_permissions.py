"""The permission gate, including what the user is actually shown."""

import pytest

from bkht.coder.permissions import ASK, AUTO, PLAN, Permissions, preview, truncate
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


def test_always_is_remembered_for_the_session(parts):
    registry, workspace = parts
    recorder = Recorder("a")
    policy = Permissions(mode=ASK, workspace=workspace, prompt=recorder)
    tool = registry.get("write_file")
    assert policy.check(tool, {"path": "a.py", "content": "1"}).allowed
    assert policy.check(tool, {"path": "b.py", "content": "2"}).allowed
    assert len(recorder.shown) == 1


def test_always_is_per_tool_not_global(parts):
    registry, workspace = parts
    recorder = Recorder("a", "n")
    policy = Permissions(mode=ASK, workspace=workspace, prompt=recorder)
    assert policy.check(registry.get("write_file"), {"path": "a.py", "content": "1"}).allowed
    assert not policy.check(registry.get("bash"), {"command": "ls"}).allowed


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
