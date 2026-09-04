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
    check_version,
    check_workspace,
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


# --- what the server says a model costs -----------------------------------

# Recorded from a live Ollama. Sizes are `/api/tags`, shapes are `/api/show`.
TAGS = [
    {"name": "qwen2.5-coder:7b", "size": 4683087561},
    {"name": "qwen2.5-coder:14b", "size": 8988124298},
    {"name": "llama3:8b", "size": 4661224676},
]
SHOW_7B = {
    "model_info": {
        "qwen2.block_count": 28,
        "qwen2.attention.head_count": 28,
        "qwen2.attention.head_count_kv": 4,
        "qwen2.embedding_length": 3584,
    }
}
SHOW_14B = {
    "model_info": {
        "qwen2.block_count": 48,
        "qwen2.attention.head_count": 40,
        "qwen2.attention.head_count_kv": 8,
        "qwen2.embedding_length": 5120,
    }
}


def test_the_weights_come_from_the_server_not_a_constant():
    assert doctor.weights_gb("qwen2.5-coder:7b", TAGS) == pytest.approx(4.36, abs=0.01)
    assert doctor.weights_gb("qwen2.5-coder:14b", TAGS) == pytest.approx(8.37, abs=0.01)
    assert doctor.weights_gb("qwen2.5-coder:32b", TAGS) is None
    assert doctor.weights_gb("qwen2.5-coder:7b", None) is None


def test_the_kv_cache_is_derived_from_the_model_shape():
    # The constant this replaces claimed one figure for both. It is out by 3.4x:
    # the 14b has more layers and twice the KV heads, which is not the same
    # thing as being twice the model.
    assert doctor.kv_gb(SHOW_7B, 16384) == pytest.approx(0.875)
    assert doctor.kv_gb(SHOW_14B, 16384) == pytest.approx(3.0)


def test_an_unknown_architecture_falls_back_rather_than_raising():
    assert doctor.kv_gb(None, 16384) is None
    assert doctor.kv_gb({"model_info": {"mystery.block_count": 4}}, 16384) is None


# --- the machine this report used to get wrong ----------------------------


def _vram(monkeypatch, mib: str | None):
    """A machine with a discrete card of the given size, or without one."""
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None if mib is None else f"/usr/bin/{name}")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": mib})(),
    )


def test_vram_is_read_from_the_card_and_the_largest_one_wins(monkeypatch):
    # Ollama loads onto one card unless told otherwise, so two 8 GB cards are an
    # 8 GB budget, not a 16 GB one.
    _vram(monkeypatch, "8192\n")
    assert doctor.gpu_vram_gb() == pytest.approx(8.0)
    _vram(monkeypatch, "8192\n24564\n")
    assert doctor.gpu_vram_gb() == pytest.approx(23.99, abs=0.01)


def test_no_gpu_tool_means_no_discrete_card(monkeypatch):
    _vram(monkeypatch, None)
    assert doctor.gpu_vram_gb() is None


def test_the_budget_prefers_the_card_and_says_which_it_read(monkeypatch):
    _vram(monkeypatch, "8192\n")
    budget = doctor.memory_budget()
    assert budget.label == "VRAM" and budget.gb == pytest.approx(8.0)
    # Subtracted, not scaled: 0.7 * 8 would leave 5.6 GB and reject the 7b at
    # 16384, which is the configuration that actually works on that card.
    assert budget.usable == pytest.approx(7.2)

    _vram(monkeypatch, None)
    monkeypatch.setattr(doctor, "total_ram_gb", lambda: 16.0)
    unified = doctor.memory_budget()
    assert unified.label == "memory" and unified.usable == pytest.approx(11.2)


def test_an_eight_gb_card_in_a_large_box_is_not_told_it_has_the_box():
    """The bug this whole check exists for.

    32 GB of system RAM and an 8 GB card: the report read the 32, passed, and
    left the user running a third of the weights on the CPU at a few tokens a
    second. The binding number is the card.
    """
    card = doctor.Budget(8.0, "VRAM", dedicated=True)
    check = check_context(
        16384,
        card,
        doctor.weights_gb("qwen2.5-coder:14b", TAGS),
        doctor.kv_gb(SHOW_14B, 1024),
        [("qwen2.5-coder:7b", 5.24)],
    )
    assert check.status == WARN
    assert "8 GB of VRAM" in check.detail
    assert "--model qwen2.5-coder:7b" in check.fix


def test_the_configuration_that_works_on_that_card_is_not_warned_about():
    # 4.36 GB of weights and 0.875 GB of cache: fully resident, and the point of
    # the whole exercise. A check that warned here would be noise.
    check = check_context(
        16384,
        doctor.Budget(8.0, "VRAM", dedicated=True),
        doctor.weights_gb("qwen2.5-coder:7b", TAGS),
        doctor.kv_gb(SHOW_7B, 1024),
    )
    assert check.status == OK
    assert "of 8 GB of VRAM" in check.detail


def test_a_smaller_model_is_named_from_what_is_actually_pulled():
    check = check_context(
        16384, doctor.Budget(8.0, "VRAM", dedicated=True), 8.37, 3.0 / 16,
        [("qwen2.5-coder:14b-q8", 20.0), ("qwen2.5-coder:7b", 5.24)],
    )
    # The largest one that fits, and never one that does not.
    assert "--model qwen2.5-coder:7b" in check.fix


def test_only_the_same_model_is_offered_as_an_alternative(monkeypatch):
    monkeypatch.setattr(doctor, "_show", lambda host, model: SHOW_7B)
    offered = doctor._alternatives("http://x", "qwen2.5-coder:14b", TAGS, 16384)
    assert [tag for tag, _ in offered] == ["qwen2.5-coder:7b"]


# --- where the weights actually are ---------------------------------------

# Recorded from `/api/ps` with the 14b loaded at 16384 on a 16 GB machine: the
# 91% GPU row the README measured, in the server's own words.
SPILLING = [{"name": "qwen2.5-coder:14b", "size": 12486495434, "size_vram": 11421058333}]
RESIDENT = [{"name": "qwen2.5-coder:7b", "size": 5628107561, "size_vram": 5628107561}]


def test_a_fully_resident_model_passes():
    check = doctor.check_placement("qwen2.5-coder:7b", RESIDENT)
    assert check.status == OK and "100% on GPU" in check.detail


def test_a_model_spilling_to_the_cpu_warns_with_the_share():
    check = doctor.check_placement("qwen2.5-coder:14b", SPILLING)
    assert check.status == WARN
    assert "91% on GPU" in check.detail and "1.0 GB" in check.detail
    assert "--num-ctx" in check.fix


def test_a_model_that_is_not_loaded_is_not_a_problem_and_is_not_loaded():
    # A health check that pulled nine gigabytes into memory as a side effect is
    # a health check nobody runs.
    check = doctor.check_placement("qwen2.5-coder:14b", RESIDENT)
    assert check.status == OK and "not loaded" in check.detail


def test_a_server_that_cannot_say_is_not_a_problem_either():
    assert doctor.check_placement("qwen2.5-coder:14b", None).status == OK
    assert doctor.check_placement("x", [{"name": "x"}]).status == OK
    assert doctor.check_placement("x", [{"name": "x", "size": 0}]).status == OK


# --- which copy, and where it was pointed ---------------------------------


def _checkout(root):
    (root / "pyproject.toml").write_text('[project]\nname = "bkht-coder"\n')
    return root


def test_running_the_checkout_you_are_in_passes(project):
    assert check_version(_checkout(project), origin=project).status == OK


def test_a_separately_installed_copy_is_flagged(project, tmp_path):
    # The exact confusion this check exists for: a feature is in the source and
    # not in the program, because the program is a copy installed elsewhere.
    check = check_version(_checkout(project), origin=tmp_path / "tools" / "bkht-coder")
    assert check.status == WARN
    assert "--editable" in check.fix


def test_a_workspace_that_is_not_coder_itself_is_not_flagged(project, tmp_path):
    # Everywhere else, an installed copy is exactly what should be running.
    assert check_version(project, origin=tmp_path / "elsewhere").status == OK


def test_the_version_check_names_where_it_ran_from(project):
    assert str(project) in check_version(project, origin=project).detail


def test_the_home_directory_is_a_poor_workspace(monkeypatch, tmp_path):
    # Not broken, just unusable: every search walks the whole of it.
    monkeypatch.setenv("HOME", str(tmp_path))
    check = check_workspace(tmp_path)
    assert check.status == WARN and "home directory" in check.detail


def test_an_ordinary_workspace_passes(project):
    assert check_workspace(project).status == OK


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
    """A machine with no model server on it.

    `doctor.httpx` is the httpx module itself, so patching an attribute on it
    reaches every backend that reaches for it -- which is what makes this
    hermetic on a machine that happens to have something on port 8080.
    """

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
    server = next(check for check in checks if check["name"] == "server")
    assert server["status"] == FAIL and "mlx_lm.server" in server["fix"]


def test_each_backend_is_checked_as_itself_not_as_the_default(project, offline):
    # This branch used to ask whether the backend *was* the default, which read
    # correctly only while the default happened to be Ollama. The moment it
    # moved, Ollama would have been checked for a command on PATH.
    def names(provider):
        return [c.name for c in doctor.run_checks(project, provider=provider)]

    assert "ollama" in names("ollama") and "placement" in names("ollama")
    assert "server" in names("local") and "placement" not in names("local")
    assert "backend" in names("codex")


def test_an_unlisted_model_is_a_warning_because_one_model_servers_answer_anyway():
    # A server started with a single model reports it under whatever name it was
    # given, and answers to any name asked for. Refusing to start over that
    # would be wrong far more often than right.
    check = doctor.check_served("coder", ["/models/qwen-fused"])
    assert check.status == WARN and "--model /models/qwen-fused" in check.fix


def test_a_served_model_that_matches_is_fine():
    assert doctor.check_served("coder", ["coder"]).status == OK


def test_a_silent_server_cannot_answer_the_model_question():
    assert doctor.check_served("coder", None).status == FAIL


def test_every_failure_carries_a_fix(project, offline):
    for check in doctor.run_checks(project):
        if check.status != OK:
            assert check.fix, f"{check.name} reports a problem without naming the fix"


def test_the_rendered_report_shows_fixes_under_their_check():
    text = render([doctor.Check("model", FAIL, "not pulled", "Run `ollama pull x`.")])
    assert text.index("not pulled") < text.index("ollama pull x")


def test_a_tight_machine_is_told_what_lowering_costs():
    """The doctor must not simply contradict its own default.

    16384 does spill to CPU on a 16 GB machine, and that is worth saying. But
    the remedy is a trade, not a correction: 8192 buys the seconds back and
    costs the ability to finish a turn on a larger file.
    """
    check = check_context(16384, 16)
    assert check.status == WARN
    assert "--num-ctx 8192" in check.fix
    assert "run out of iterations" in check.fix
    assert "Try the default first" in check.fix


def test_asking_for_more_than_the_default_is_still_told_to_lower_it():
    check = check_context(32768, 16)
    assert check.status == WARN
    assert check.fix.endswith("Lower it with `--num-ctx 8192`.")


def test_a_machine_too_small_for_any_window_is_sent_to_the_7b():
    assert "qwen2.5-coder:7b" in check_context(8192, 4).fix


# --- a backend that is not a local server ------------------------------------


def test_a_hosted_backend_replaces_the_three_ollama_checks(monkeypatch, tmp_path):
    # Nothing about a local server describes a model reached through somebody
    # else's command line, and a report that fails on a server the user is not
    # running is a report they learn to ignore.
    monkeypatch.setattr(doctor, "_tags", lambda host: pytest.fail("probed Ollama"))
    checks = doctor.run_checks(tmp_path, provider="claude-code", model="opus")
    names = [check.name for check in checks]
    assert "ollama" not in names and "num_ctx" not in names
    assert "backend" in names


def test_an_installed_backend_passes(monkeypatch, tmp_path):
    monkeypatch.setattr("bkht.coder.external.shutil.which", lambda name: f"/usr/bin/{name}")
    check = doctor.check_backend("codex", "gpt-5.5")
    assert check.status == doctor.OK
    assert "codex" in check.detail and "gpt-5.5" in check.detail


def test_a_missing_backend_fails_with_the_way_to_install_it(monkeypatch, tmp_path):
    monkeypatch.setattr("bkht.coder.external.shutil.which", lambda name: None)
    check = doctor.check_backend("claude-code", "opus")
    assert check.status == doctor.FAIL
    assert "claude.com/claude-code" in check.fix


def test_a_backend_that_does_not_exist_fails_rather_than_raising(tmp_path):
    check = doctor.check_backend("gemini", "whatever")
    assert check.status == doctor.FAIL
    assert "ollama" in check.detail


def test_the_update_check_reports_a_newer_release(monkeypatch, tmp_path):
    from bkht.coder import update

    monkeypatch.setattr(update, "editable", lambda: None)
    monkeypatch.setattr(update, "refresh", lambda: None)
    monkeypatch.setattr(update, "cached", lambda: "0.9.0")
    monkeypatch.setattr(update, "available", lambda: "0.9.0")

    check = doctor.check_update(tmp_path)
    assert check.status == WARN and "0.9.0" in check.detail
    assert "coder update" in check.fix


def test_the_update_check_is_quiet_when_current(monkeypatch, tmp_path):
    from bkht.coder import update

    monkeypatch.setattr(update, "editable", lambda: None)
    monkeypatch.setattr(update, "refresh", lambda: None)
    monkeypatch.setattr(update, "cached", lambda: "0.9.0")
    monkeypatch.setattr(update, "available", lambda: None)

    assert doctor.check_update(tmp_path).status == OK


def test_the_update_check_never_fails_a_report(monkeypatch, tmp_path):
    # Being a release behind does not stop a turn from running, and this report
    # is read to find out why one will not.
    from bkht.coder import update

    monkeypatch.setattr(update, "editable", lambda: None)
    monkeypatch.setattr(update, "refresh", lambda: None)
    monkeypatch.setattr(update, "cached", lambda: None)

    assert doctor.check_update(tmp_path).status == WARN


def test_the_update_check_sends_a_checkout_to_git_pull(monkeypatch, tmp_path):
    from bkht.coder import update

    monkeypatch.setattr(update, "editable", lambda: tmp_path)
    monkeypatch.setattr(update, "refresh", lambda: pytest.fail("asked anyway"))

    check = doctor.check_update(tmp_path)
    assert check.status == OK and "git pull" in check.detail


# --- the verify check -------------------------------------------------------


def test_a_configured_command_is_reported(tmp_path):
    check = doctor.check_verify(tmp_path, "pytest -q")
    assert check.status == doctor.OK
    assert "pytest -q" in check.detail
    assert not check.fix


def test_an_unset_command_suggests_one_and_says_how_to_set_it(tmp_path):
    # The suggestion is why this check exists. Without somewhere that says out
    # loud what this project looks like, the feature is one nobody finds.
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "tests").mkdir()
    check = doctor.check_verify(tmp_path)

    assert check.status == doctor.OK
    assert "not set" in check.detail
    assert "coder config set verify_command" in check.fix
    assert "pytest -q" in check.fix


def test_a_project_with_no_signal_is_not_offered_a_guess(tmp_path):
    check = doctor.check_verify(tmp_path)
    assert check.status == doctor.OK
    assert not check.fix


def test_an_unset_command_never_fails_the_report(tmp_path):
    # Not configuring it is a choice, not a broken install, and `doctor`'s exit
    # status is what CI reads.
    assert not doctor.check_verify(tmp_path).failed


# --- the hooks check --------------------------------------------------------


def test_no_hooks_is_a_sentence_not_a_silence():
    check = doctor.check_hooks({})
    assert check.status == doctor.OK
    assert "none configured" in check.detail


def test_every_configured_hook_is_named():
    # Listed rather than counted. The whole safety argument for hooks is that
    # they are never invisible, and this is where they stop being invisible.
    check = doctor.check_hooks({"pre_tool": ["gate.sh"], "turn_end": ["make docs"]})
    assert "2 configured" in check.detail
    assert "gate.sh" in check.fix and "make docs" in check.fix
    assert "without asking" in check.fix


def test_hooks_never_fail_the_report():
    # Configuring one is a choice, not a broken install, and `doctor`'s exit
    # status is what CI reads.
    assert not doctor.check_hooks({"pre_tool": ["gate.sh"]}).failed


def test_the_report_carries_the_hooks_it_was_given(tmp_path, offline):
    checks = doctor.run_checks(tmp_path, hooks={"turn_end": ["make"]})
    named = [c for c in checks if c.name == "hooks"]
    assert named and "make" in named[0].fix


# --- the agent/ surface ---------------------------------------------------


def test_the_agent_check_names_the_slots_it_found(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor.layout, "GLOBAL_ROOT", tmp_path / "nowhere")
    (tmp_path / "agent" / "skills").mkdir(parents=True)
    (tmp_path / "agent" / "agent.json").write_text("{}")

    check = doctor.check_agent_surface(tmp_path)
    assert check.status == doctor.OK and "skills" in check.detail


def test_the_agent_check_is_quiet_about_somebody_elses_agent_directory(tmp_path, monkeypatch):
    # An eve project's agent/ is not ours and is not a problem.
    monkeypatch.setattr(doctor.layout, "GLOBAL_ROOT", tmp_path / "nowhere")
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "instructions.md").write_text("You are a data agent.\n")

    check = doctor.check_agent_surface(tmp_path)
    assert check.status == doctor.OK and "no agent/ surface" in check.detail


def test_a_broken_marker_warns_and_says_what_it_should_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor.layout, "GLOBAL_ROOT", tmp_path / "nowhere")
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.json").write_text("{not json")

    check = doctor.check_agent_surface(tmp_path)
    assert check.status == doctor.WARN and "agent.json" in check.fix
