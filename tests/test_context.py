"""File-tree summary, token accounting, and compaction."""

import pytest

from bkht.coder.agent import Agent
from bkht.coder.context import (
    KEEP_RECENT,
    compact,
    file_tree,
    should_compact,
    transcript,
    usage_ratio,
)
from bkht.coder.provider import ProviderError
from bkht.coder.session import Session
from bkht.coder.tools import build_registry

from fakes import FakeProvider, call


def history(n: int) -> list[dict]:
    return [{"role": "user", "content": f"message {i}"} for i in range(n)]


# --- file tree --------------------------------------------------------------


def test_tree_lists_full_relative_paths(project):
    # Flat paths are directly usable as read_file arguments.
    assert "src/main.py" in file_tree(project).splitlines()


def test_tree_skips_ignored_directories_and_dotfiles(project):
    listing = file_tree(project)
    assert "node_modules" not in listing and ".git" not in listing


def test_tree_is_bounded(project):
    for i in range(50):
        (project / f"f{i}.py").write_text("x\n")
    listing = file_tree(project, max_entries=10)
    assert "more files" in listing
    assert len(listing.splitlines()) == 11


def test_tree_counts_what_it_left_out(project):
    for i in range(50):
        (project / f"f{i}.py").write_text("x\n")
    # 50 planted files plus the fixture's own three, minus the ten shown.
    assert "and 43 more files" in file_tree(project, max_entries=10)


def test_tree_skips_hidden_directories_at_any_depth(project):
    cache = project / ".aider.tags.cache.v4" / "07" / "15"
    cache.mkdir(parents=True)
    (cache / "b5e4.val").write_text("x\n")
    assert ".aider" not in file_tree(project)


def test_tree_spends_its_budget_on_the_shallowest_files(project):
    """A deep vendored tree must not crowd out the manifests above it."""
    deep = project / "vendor" / "sdk" / "sources" / "internal"
    deep.mkdir(parents=True)
    for i in range(50):
        (deep / f"gen{i}.swift").write_text("x\n")

    listing = file_tree(project, max_entries=3).splitlines()
    assert listing[:3] == ["README.md", "src/main.py", "src/util.py"]
    assert "and 50 more files" in listing[-1]


def test_tree_of_an_empty_directory(tmp_path):
    assert file_tree(tmp_path) == "(no files)"


# --- accounting -------------------------------------------------------------


def test_usage_ratio_uses_the_reported_count():
    session = Session(messages=history(4))
    session.record_usage(8192, 10)
    assert usage_ratio(session, 32768) == pytest.approx(0.25)


def test_usage_ratio_falls_back_to_an_estimate():
    session = Session(system="x" * 4000, messages=history(2))
    assert usage_ratio(session, 32768) > 0


def test_no_compaction_while_there_is_room():
    session = Session(messages=history(20))
    session.record_usage(1000, 0)
    assert not should_compact(session, 32768)


def test_compaction_triggers_past_the_threshold():
    session = Session(messages=history(20))
    session.record_usage(26000, 0)
    assert should_compact(session, 32768)


def test_a_short_history_is_never_compacted():
    # Nothing to gain, and compacting would drop the whole conversation.
    session = Session(messages=history(KEEP_RECENT))
    session.record_usage(31000, 0)
    assert not should_compact(session, 32768)


# --- compaction -------------------------------------------------------------


def test_compaction_replaces_the_old_and_keeps_the_recent():
    session = Session(messages=history(20))
    provider = FakeProvider(["The user wants the parser fixed in src/parse.py."])

    summary = compact(session, provider)
    assert summary.startswith("The user wants")
    assert len(session.messages) == KEEP_RECENT + 1
    assert "compacted to save context" in session.messages[0]["content"]
    assert session.messages[-1]["content"] == "message 19"


def test_compaction_does_not_send_the_system_prompt_to_the_summarizer():
    session = Session(system="TOOLS AND PROTOCOL", messages=history(20))
    provider = FakeProvider(["summary"])
    compact(session, provider)
    sent = "".join(m["content"] for m in provider.calls[0])
    assert "TOOLS AND PROTOCOL" not in sent


def test_compaction_resets_the_stale_token_count():
    session = Session(messages=history(20))
    session.record_usage(30000, 0)
    compact(session, FakeProvider(["summary"]))
    assert not should_compact(session, 32768)


def test_a_failed_summary_leaves_history_untouched():
    # Degrading to a full context beats silently forgetting the task.
    session = Session(messages=history(20))
    before = list(session.messages)
    assert compact(session, FakeProvider([ProviderError("down")])) is None
    assert session.messages == before


def test_an_empty_summary_is_refused():
    session = Session(messages=history(20))
    before = list(session.messages)
    assert compact(session, FakeProvider(["   "])) is None
    assert session.messages == before


def test_transcript_labels_tool_output_by_tool_name():
    text = transcript([{"role": "tool", "name": "read_file", "content": "x = 1"}])
    assert text.startswith("[read_file]")


def test_transcript_bounds_a_huge_tool_result():
    text = transcript([{"role": "tool", "name": "bash", "content": "y" * 9000}])
    assert "[truncated]" in text and len(text) < 3000


# --- inside the loop --------------------------------------------------------


def test_the_loop_compacts_before_asking(project):
    registry, workspace = build_registry(project, read_only=True)
    session = Session(system="sys", messages=history(20))
    session.record_usage(30000, 0)

    # First scripted reply answers the compaction call, the second the task.
    provider = FakeProvider(["a summary of earlier work", "done"])
    agent = Agent(provider, registry, session, max_iterations=2)

    assert agent.run("carry on").answer == "done"
    assert "compacted to save context" in session.messages[0]["content"]
