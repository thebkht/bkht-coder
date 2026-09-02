"""The one question git is asked: which branch this is."""

from __future__ import annotations

import subprocess

import pytest

from bkht.coder import git


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True)


@pytest.fixture
def repo(tmp_path):
    _git("init", "-q", "-b", "main", cwd=tmp_path)
    _git("config", "user.email", "t@example.com", cwd=tmp_path)
    _git("config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "file.txt").write_text("one\n")
    _git("add", ".", cwd=tmp_path)
    _git("commit", "-qm", "first", cwd=tmp_path)
    return tmp_path


def test_the_branch_is_the_one_checked_out(repo):
    assert git.branch(repo) == "main"
    _git("checkout", "-q", "-b", "feature", cwd=repo)
    assert git.branch(repo) == "feature"


def test_a_detached_head_answers_with_the_commit_it_sits_on(repo):
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    _git("checkout", "-q", "--detach", cwd=repo)
    # Not the literal "HEAD" git answers with, which names every detached
    # checkout there has ever been and so identifies none of them.
    assert git.branch(repo) == head


def test_a_directory_that_is_not_a_repository_has_no_branch(tmp_path):
    assert git.branch(tmp_path) == ""


def test_a_git_that_is_not_installed_is_not_an_error(monkeypatch, repo):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)
    assert git.branch(repo) == ""


def test_a_git_that_hangs_is_not_a_hung_prompt(monkeypatch, repo):
    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired("git", 0.001)

    monkeypatch.setattr(subprocess, "run", slow)
    assert git.branch(repo) == ""
