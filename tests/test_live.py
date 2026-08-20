"""End-to-end checks against a real Ollama server.

Mirrors the shape of ~/agent/scripts/verify.sh: build a throwaway git repo
containing a real bug, ask the agent to fix it, and assert the corrected code
behaves. Skipped entirely unless Ollama is reachable.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from bkht.coder.agent import Agent
from bkht.coder.context import file_tree
from bkht.coder.permissions import AUTO, Permissions
from bkht.coder.prompts import system_prompt
from bkht.coder.provider import DEFAULT_MODEL, OllamaProvider, collect
from bkht.coder.session import Session, Snapshots
from bkht.coder.tools import build_registry

pytestmark = pytest.mark.live

BUGGY = '''\
def average(numbers):
    """Return the mean of a list of numbers."""
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)
'''


@pytest.fixture(scope="module")
def provider(pytestconfig):
    provider = OllamaProvider(model=pytestconfig.getoption("--model") or DEFAULT_MODEL)
    if not provider.available():
        pytest.skip("Ollama is not reachable")
    return provider


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo containing one buggy function."""
    (tmp_path / "buggy.py").write_text(BUGGY)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@localhost", "-c", "user.name=t", "commit", "-qm", "scratch"],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def build(provider, root, mode=AUTO, snapshots=None):
    snapshots = snapshots if snapshots is not None else Snapshots()
    registry, workspace = build_registry(root, snapshots=snapshots)
    session = Session(
        system=system_prompt(registry, str(workspace.root), file_tree(workspace.root))
    )
    permissions = Permissions(mode=mode, workspace=workspace)
    agent = Agent(provider, registry, session, permissions=permissions)
    return agent, snapshots


def test_the_model_is_present(provider):
    import httpx

    names = [
        m["name"]
        for m in httpx.get(f"{provider.host}/api/tags", timeout=10).json()["models"]
    ]
    assert provider.model in names, f"{provider.model} is not pulled; have {names}"


def test_num_ctx_is_honoured(provider):
    # The 2048 default is the single most common cause of a bad local session.
    # The exact ceiling is a property of the machine, not of this code, so the
    # check is that we are clear of the default rather than at any fixed value.
    assert provider.num_ctx > 2048
    reply = collect(provider.chat([{"role": "user", "content": "Reply with OK."}]))
    assert reply.content.strip()


def test_answers_a_question_by_reading_the_code(provider, repo):
    agent, _ = build(provider, repo)
    outcome = agent.run(
        "What does the function in buggy.py compute? Read the file before answering."
    )
    assert outcome.tool_calls >= 1, "the model never called a tool"
    assert outcome.stopped == "answered"
    assert "average" in outcome.answer.lower() or "mean" in outcome.answer.lower()


def test_a_greeting_is_answered_without_running_anything(provider, repo):
    # A 14b model reads a prompt full of tools as an instruction to use one, and
    # will shell out to `echo hello` rather than say hello back. The prompt has
    # to say that a conversational message is not a task.
    agent, _ = build(provider, repo)
    outcome = agent.run("hello")
    assert outcome.tool_calls == 0, f"the model called a tool to greet: {outcome.errors}"
    assert outcome.stopped == "answered"
    assert outcome.answer.strip()


def test_answers_in_the_language_it_was_asked_in(provider, repo):
    # Russian rather than Uzbek only because the script makes the assertion
    # exact: Cyrillic in the reply cannot be anything but an answer in Russian.
    agent, _ = build(provider, repo)
    outcome = agent.run("что делает функция в buggy.py?")
    assert outcome.stopped == "answered"
    cyrillic = sum("\u0400" <= c <= "\u04ff" for c in outcome.answer)
    assert cyrillic > 10, f"answered a Russian question in another language: {outcome.answer}"


def test_reads_the_file_instead_of_asking_for_it(provider, repo):
    # The flip side of "not every message is a task": told to lean on prose, the
    # model starts replying "paste the file and I will look" to questions it has
    # a read_file tool for.
    agent, _ = build(provider, repo)
    outcome = agent.run("what does buggy.py do?")
    assert outcome.tool_calls >= 1, f"asked instead of reading: {outcome.answer}"
    assert "average" in outcome.answer.lower() or "mean" in outcome.answer.lower()


def test_fixes_a_real_bug_and_the_fixed_code_runs(provider, repo):
    agent, snapshots = build(provider, repo)
    outcome = agent.run(
        "buggy.py raises ZeroDivisionError when average() is called with an empty "
        "list. Edit buggy.py so that average([]) returns 0.0 instead. Keep the "
        "behaviour for non-empty lists unchanged."
    )
    assert outcome.stopped in ("answered", "iteration-cap")

    source = (repo / "buggy.py").read_text()
    assert source != BUGGY, f"the file was never changed (stopped: {outcome.stopped})"

    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import buggy; assert buggy.average([]) == 0.0; "
            "assert buggy.average([1, 2, 3]) == 2.0; print('ok')",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, f"fixed code misbehaves:\n{source}\n{check.stderr}"

    assert snapshots.undo() is not None
    assert (repo / "buggy.py").read_text() == BUGGY
