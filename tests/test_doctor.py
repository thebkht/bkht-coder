"""Preflight checks: each one's pass and fail branch, and the CI contract."""

import json

import httpx
import pytest

from bkht.coder import doctor
from bkht.coder.doctor import (
    FAIL,
    OK,
    WARN,
    check_context,
    check_git,
    check_model,
    check_server,
    check_state_dir,
    check_skills,
    render,
    report,
)


# --- the server -----------------------------------------------------------


def test_a_reachable_server_passes():
    check = check_server("http://localhost:11434", ["qwen2.5-coder:14b"], "")
    assert check.status == OK


def test_an_unreachable_server_fails_and_names_the_fix():
    check = check_server("http://localhost:11434", None, "connection refused")
    assert check.status == FAIL
    assert "ollama serve" in check.fix


# --- the model ------------------------------------------------------------


def test_a_pulled_model_passes():
    assert check_model("qwen2.5-coder:14b", ["qwen2.5-coder:14b"]).status == OK


def test_a_missing_model_fails_with_the_pull_command():
    check = check_model("qwen2.5-coder:14b", ["llama3:8b"])
    assert check.status == FAIL
    assert "ollama pull qwen2.5-coder:14b" in check.fix


def test_a_different_size_of_the_same_model_is_only_a_warning():
    # It will not run as asked, but the user is one flag away rather than one
    # nine-gigabyte download away.
    check = check_model("qwen2.5-coder:14b", ["qwen2.5-coder:7b"])
    assert check.status == WARN
    assert "--model qwen2.5-coder:7b" in check.fix


def test_the_model_cannot_be_checked_without_a_server():
    assert check_model("qwen2.5-coder:14b", None).status == FAIL


# --- context against memory -----------------------------------------------


def test_the_documented_default_fits_the_documented_machine():
    # The README measures 8192 on a 16 GB machine at 10 GB, fully on the GPU.
    # If this check warned about it, it would be arguing with the default.
    assert check_context(8192, 16).status == OK


def test_a_context_too_large_for_the_machine_warns():
    # Also measured: 32768 on the same machine timed the turn out entirely.
    check = check_context(32768, 16)
    assert check.status == WARN
    assert "--num-ctx" in check.fix


def test_the_suggested_context_is_one_that_actually_fits():
    check = check_context(32768, 16)
    suggested = int(check.fix.split("--num-ctx ")[1].split("`")[0])
    assert check_context(suggested, 16).status == OK


def test_a_tiny_machine_is_told_to_change_model_not_context():
    check = check_context(8192, 4)
    assert check.status == WARN and "7b" in check.fix


def test_unknown_memory_is_not_reported_as_a_problem():
    assert check_context(8192, None).status == OK


# --- the rest -------------------------------------------------------------


def test_an_unwritable_state_directory_fails(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("I am a file, not a directory\n")
    assert check_state_dir(blocked).status == FAIL


def test_a_writable_state_directory_passes(tmp_path):
    assert check_state_dir(tmp_path / "state").status == OK


def test_git_is_only_ever_a_warning(monkeypatch):
    # Everything but /diff and review works without it.
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    assert check_git().status == WARN


def test_a_broken_skill_is_surfaced_here_too(project):
    directory = project / ".bkht-coder" / "skills" / "broken"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: broken\n---\n\nBody.\n")

    check = check_skills(project)
    assert check.status == WARN and "broken" in check.detail


# --- the report -----------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    """A machine with no Ollama on it."""

    def refuse(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(doctor.httpx, "get", refuse)


def test_a_clean_report_exits_zero(project, monkeypatch):
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda *a, **k: [doctor.Check("ollama", OK, "fine")],
    )
    lines = []
    assert report(project, out=lines.append) == 0
    assert "Everything checks out." in lines[0]


def test_a_failing_report_exits_one(project, offline):
    # The contract `coder review --json` already has: a non-zero exit is what
    # CI reads.
    lines = []
    assert report(project, out=lines.append) == 1
    assert "check(s) failed" in lines[0]


def test_json_output_is_parseable_and_carries_the_fix(project, offline):
    lines = []
    report(project, as_json=True, out=lines.append)

    checks = json.loads(lines[0])
    server = next(check for check in checks if check["name"] == "ollama")
    assert server["status"] == FAIL and "ollama serve" in server["fix"]


def test_every_failure_carries_a_fix(project, offline):
    for check in doctor.run_checks(project):
        if check.status != OK:
            assert check.fix, f"{check.name} reports a problem without naming the fix"


def test_the_rendered_report_shows_fixes_under_their_check():
    text = render([doctor.Check("model", FAIL, "not pulled", "Run `ollama pull x`.")])
    assert text.index("not pulled") < text.index("ollama pull x")
