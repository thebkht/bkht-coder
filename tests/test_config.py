"""Persisted settings: what wins, what is rejected, and what survives a write."""

import argparse
import json

import pytest

from bkht.coder import config
from bkht.coder.config import ConfigError, Settings


@pytest.fixture
def home(tmp_path, monkeypatch):
    """The global config file, moved somewhere harmless."""
    path = tmp_path / "home" / "config.json"
    monkeypatch.setattr(config, "GLOBAL_PATH", path)
    return path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return root


def write(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding="utf-8")


# --- layering ---------------------------------------------------------------


def test_nothing_stored_is_every_default(home, project):
    settings = config.load(project)
    assert settings.model == config.DEFAULT_MODEL
    assert settings.source("model") == config.DEFAULT
    assert settings.error == ""


def test_the_global_file_beats_the_default(home, project):
    write(home, model="qwen2.5-coder:7b")
    settings = config.load(project)
    assert settings.model == "qwen2.5-coder:7b"
    assert settings.source("model") == config.GLOBAL


def test_the_workspace_file_beats_the_global_one(home, project):
    write(home, model="qwen2.5-coder:7b", num_ctx=8192)
    write(project / config.WORKSPACE_NAME, model="qwen2.5-coder:14b")
    settings = config.load(project)

    assert settings.model == "qwen2.5-coder:14b"
    assert settings.source("model") == config.WORKSPACE
    # Untouched by the workspace, so the global value still stands.
    assert settings.num_ctx == 8192
    assert settings.source("num_ctx") == config.GLOBAL


def test_without_a_workspace_only_the_global_file_is_read(home, project):
    write(home, model="global-one")
    write(project / config.WORKSPACE_NAME, model="workspace-one")
    assert config.load().model == "global-one"


# --- reading never raises ---------------------------------------------------


def test_an_unparseable_file_leaves_the_defaults_and_a_reason(home, project):
    home.parent.mkdir(parents=True, exist_ok=True)
    home.write_text("{not json", encoding="utf-8")

    settings = config.load(project)
    assert settings.model == config.DEFAULT_MODEL
    assert str(home) in settings.error


def test_a_file_that_is_not_an_object_is_skipped(home, project):
    write_path = home
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text("[1, 2, 3]", encoding="utf-8")

    settings = config.load(project)
    assert settings.model == config.DEFAULT_MODEL
    assert "not a JSON object" in settings.error


def test_one_bad_key_does_not_discard_the_good_ones(home, project):
    write(home, model="qwen2.5-coder:7b", num_ctx=64)
    settings = config.load(project)

    assert settings.model == "qwen2.5-coder:7b"
    assert settings.num_ctx == config.DEFAULT_NUM_CTX
    assert "too small" in settings.error


def test_an_unknown_key_is_reported_not_adopted(home, project):
    write(home, nonsense=True)
    settings = config.load(project)
    assert "unknown setting" in settings.error
    assert "nonsense" not in settings.values


# --- coercion and validation ------------------------------------------------


@pytest.mark.parametrize(
    "key, raw, expected",
    [
        ("num_ctx", "8192", 8192),
        ("temperature", "0.5", 0.5),
        ("temperature", 1, 1.0),
        ("scout", "off", False),
        ("scout", "YES", True),
        ("model", "  spaced  ", "spaced"),
    ],
)
def test_values_are_coerced_to_their_declared_type(key, raw, expected):
    assert config.parse(key, raw) == expected


@pytest.mark.parametrize(
    "key, raw",
    [
        ("provider", "claude"),
        ("mode", "yolo"),
        ("num_ctx", 2048),
        ("num_ctx", "lots"),
        ("temperature", 9),
        ("max_iterations", 0),
        ("scout", "maybe"),
        ("model", ""),
        ("nope", "x"),
    ],
)
def test_bad_values_are_refused(key, raw):
    with pytest.raises(ConfigError):
        config.parse(key, raw)


def test_the_unknown_provider_message_names_what_exists():
    with pytest.raises(ConfigError, match="ollama"):
        config.parse("provider", "gemini")


def test_every_registered_backend_can_be_named():
    for name in ("ollama", "claude-code", "codex"):
        assert config.parse("provider", name) == name


def test_switching_provider_brings_its_model_and_window_with_it(home, project):
    # Otherwise `provider = claude-code` would ask Claude Code for a model tag
    # only a local Ollama has ever heard of.
    write(home, provider="claude-code")
    settings = config.load(project)
    assert settings.values["model"] == "opus"
    assert settings.values["num_ctx"] == 1_000_000
    assert settings.source("model") == config.DEFAULT


def test_a_model_the_user_pinned_survives_a_provider_switch(home, project):
    # A value they wrote down is theirs. Running a different model than the one
    # they named would be worse than failing on the one they did.
    write(home, provider="codex", model="qwen2.5-coder:7b")
    assert config.load(project).values["model"] == "qwen2.5-coder:7b"


def test_true_is_not_accepted_as_a_number():
    # bool is a subclass of int, so this has to be refused deliberately.
    with pytest.raises(ConfigError):
        config.parse("num_ctx", True)


# --- writing ----------------------------------------------------------------


def test_set_then_load(home, project):
    config.set_value("model", "qwen2.5-coder:7b")
    assert config.load(project).model == "qwen2.5-coder:7b"


def test_set_targets_the_workspace_when_asked(home, project):
    config.set_value("num_ctx", "8192", scope=config.WORKSPACE, root=project)
    assert (project / config.WORKSPACE_NAME).exists()
    assert not home.exists()
    assert config.load(project).source("num_ctx") == config.WORKSPACE


def test_setting_one_key_keeps_the_others(home, project):
    config.set_value("model", "a-model")
    config.set_value("num_ctx", "8192")
    stored = json.loads(home.read_text())
    assert stored == {"model": "a-model", "num_ctx": 8192}


def test_a_key_this_version_does_not_know_is_carried_across(home, project):
    write(home, model="a-model", from_the_future="keep me")
    config.set_value("num_ctx", "8192")
    assert json.loads(home.read_text())["from_the_future"] == "keep me"


def test_set_refuses_a_bad_value_before_touching_the_file(home, project):
    with pytest.raises(ConfigError):
        config.set_value("num_ctx", "12")
    assert not home.exists()


def test_unset_removes_only_that_key(home, project):
    config.set_value("model", "a-model")
    config.set_value("num_ctx", "8192")
    assert config.unset("model") is True
    assert json.loads(home.read_text()) == {"num_ctx": 8192}


def test_unset_reports_a_key_that_was_not_set(home, project):
    config.set_value("model", "a-model")
    assert config.unset("num_ctx") is False


def test_unset_refuses_an_unknown_key(home, project):
    with pytest.raises(ConfigError):
        config.unset("nonsense")


def test_an_unknown_scope_is_refused():
    with pytest.raises(ConfigError):
        config.path_for("elsewhere")


def test_the_workspace_scope_needs_a_workspace():
    with pytest.raises(ConfigError):
        config.path_for(config.WORKSPACE)


# --- applying to parsed arguments -------------------------------------------


def namespace(**overrides):
    """A namespace shaped like the one argparse hands back, all flags unset."""
    values = dict(
        provider=None, model=None, host=None, num_ctx=None, temperature=None,
        max_iterations=None, no_scout=None, no_instructions=None, no_skills=None,
        no_planning=None, no_delegation=None, auto=None, plan=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_apply_fills_the_flags_nobody_typed(home, project):
    write(home, model="qwen2.5-coder:7b", num_ctx=8192, scout=False)
    args = namespace()
    config.load(project).apply(args)

    assert args.model == "qwen2.5-coder:7b"
    assert args.num_ctx == 8192
    assert args.no_scout is True


def test_a_flag_beats_the_file(home, project):
    write(home, model="qwen2.5-coder:7b")
    args = namespace(model="qwen2.5-coder:14b")
    config.load(project).apply(args)
    assert args.model == "qwen2.5-coder:14b"


def test_apply_turns_a_configured_mode_into_the_two_switches(home, project):
    write(home, mode="plan")
    args = namespace()
    config.load(project).apply(args)
    assert (args.auto, args.plan) == (False, True)


def test_a_mode_flag_beats_a_configured_mode(home, project):
    write(home, mode="plan")
    args = namespace(auto=True)
    config.load(project).apply(args)
    # Left exactly as typed, so --auto --plan still contradicts itself.
    assert (args.auto, args.plan) == (True, None)


def test_apply_tolerates_a_namespace_without_every_flag(home, project):
    # `coder review` has --auto but no --plan; nothing here may assume a shape.
    args = argparse.Namespace(model=None, auto=None)
    config.load(project).apply(args)
    assert args.model == config.DEFAULT_MODEL


# --- rendering --------------------------------------------------------------


def test_the_listing_names_every_setting_and_where_it_came_from(home, project):
    write(home, model="qwen2.5-coder:7b")
    settings = config.load(project)
    listed = dict((key, (value, source)) for key, value, source in settings.listing())

    assert listed["model"] == ("qwen2.5-coder:7b", config.GLOBAL)
    assert listed["scout"] == ("true", config.DEFAULT)
    assert len(listed) == len(config.FIELDS)


def test_a_rendered_value_can_be_typed_back_in(home, project):
    settings = config.load(project)
    for key, value, _ in settings.listing():
        assert config.parse(key, value) == settings.values[key]


def test_the_payload_is_json_shaped(home, project):
    payload = config.load(project).payload()
    assert json.loads(json.dumps(payload))["scout"] == {"value": True, "source": "default"}


def test_a_missing_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError):
        Settings(values={}).nonsense


# --- the command ------------------------------------------------------------


def coder(capsys, *argv):
    """Run `coder ...` and return its exit status with what it printed."""
    from bkht.coder.cli import main

    status = main(list(argv))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_config_lists_every_setting(home, project, capsys):
    status, out, _ = coder(capsys, "config", "--cwd", str(project))
    assert status == 0
    for spec in config.FIELDS:
        assert spec.name in out


def test_config_set_then_get(home, project, capsys):
    coder(capsys, "config", "set", "model", "qwen2.5-coder:7b", "--cwd", str(project))
    status, out, _ = coder(capsys, "config", "get", "model", "--cwd", str(project))
    assert (status, out.strip()) == (0, "qwen2.5-coder:7b")


def test_config_set_names_the_file_it_wrote(home, project, capsys):
    status, out, _ = coder(capsys, "config", "set", "num_ctx", "8192", "--cwd", str(project))
    assert status == 0
    assert str(home) in out


def test_config_set_workspace_writes_the_workspace_file(home, project, capsys):
    coder(capsys, "config", "set", "--workspace", "num_ctx", "8192", "--cwd", str(project))
    assert json.loads((project / config.WORKSPACE_NAME).read_text()) == {"num_ctx": 8192}


def test_config_list_says_where_each_value_came_from(home, project, capsys):
    coder(capsys, "config", "set", "--workspace", "num_ctx", "8192", "--cwd", str(project))
    _, out, _ = coder(capsys, "config", "--cwd", str(project))
    line = next(row for row in out.splitlines() if row.strip().startswith("num_ctx"))
    assert "8192" in line and config.WORKSPACE in line


def test_config_json_is_machine_readable(home, project, capsys):
    _, out, _ = coder(capsys, "config", "--json", "--cwd", str(project))
    assert json.loads(out)["provider"] == {"value": "ollama", "source": "default"}


def test_config_unset_falls_back_to_the_layer_below(home, project, capsys):
    coder(capsys, "config", "set", "model", "a-model", "--cwd", str(project))
    status, out, _ = coder(capsys, "config", "unset", "model", "--cwd", str(project))
    assert status == 0
    assert config.DEFAULT_MODEL in out


def test_config_unset_says_when_there_was_nothing_to_unset(home, project, capsys):
    status, out, _ = coder(capsys, "config", "unset", "model", "--cwd", str(project))
    assert (status, "was not set" in out) == (0, True)


def test_config_path_names_both_files(home, project, capsys):
    _, out, _ = coder(capsys, "config", "path", "--cwd", str(project))
    assert str(home) in out
    assert str(project / config.WORKSPACE_NAME) in out


def test_a_bad_value_exits_two_and_says_what_would_work(home, project, capsys):
    status, _, err = coder(capsys, "config", "set", "num_ctx", "12", "--cwd", str(project))
    assert status == 2
    assert str(config.MIN_USEFUL_NUM_CTX) in err
    assert not home.exists()


def test_an_unknown_key_exits_two(home, project, capsys):
    status, _, err = coder(capsys, "config", "get", "nonsense", "--cwd", str(project))
    assert status == 2
    assert "unknown setting" in err


def test_an_unknown_action_exits_two_with_the_usage(home, project, capsys):
    status, _, err = coder(capsys, "config", "frobnicate", "--cwd", str(project))
    assert status == 2
    assert "coder config" in err


def test_set_without_a_value_exits_two(home, project, capsys):
    status, _, err = coder(capsys, "config", "set", "model", "--cwd", str(project))
    assert status == 2
    assert "key and a value" in err


def test_a_value_with_spaces_survives(home, project, capsys):
    coder(capsys, "config", "set", "model", "a model", "--cwd", str(project))
    _, out, _ = coder(capsys, "config", "get", "model", "--cwd", str(project))
    assert out.strip() == "a model"


def test_an_unreadable_file_is_warned_about_not_swallowed(home, project, capsys):
    home.parent.mkdir(parents=True, exist_ok=True)
    home.write_text("{not json", encoding="utf-8")
    status, _, err = coder(capsys, "config", "--cwd", str(project))
    assert status == 0
    assert "warning" in err


# --- reaching a session -----------------------------------------------------


def test_a_configured_model_reaches_the_parsed_arguments(home, project):
    from bkht.coder.cli import build_agent_parser, configured

    write(home, model="qwen2.5-coder:7b", num_ctx=8192, mode="plan")
    args = configured(build_agent_parser().parse_args(["--cwd", str(project)]))

    assert (args.model, args.num_ctx) == ("qwen2.5-coder:7b", 8192)
    assert args.plan is True


def test_a_flag_still_beats_a_configured_value(home, project):
    from bkht.coder.cli import build_agent_parser, configured

    write(home, model="qwen2.5-coder:7b")
    args = configured(
        build_agent_parser().parse_args(["--cwd", str(project), "--model", "other"])
    )
    assert args.model == "other"


def test_a_configured_mode_becomes_the_sessions_mode(home, project):
    from bkht.coder.cli import build_agent_parser, configured, resolve_mode
    from bkht.coder.permissions import AUTO

    write(home, mode="auto")
    args = configured(build_agent_parser().parse_args(["--cwd", str(project)]))
    assert resolve_mode(args) == AUTO


def test_an_unreadable_config_does_not_stop_a_session_starting(home, project, capsys):
    from bkht.coder.cli import build_agent_parser, configured

    home.parent.mkdir(parents=True, exist_ok=True)
    home.write_text("{not json", encoding="utf-8")

    args = configured(build_agent_parser().parse_args(["--cwd", str(project)]))
    assert args.model == config.DEFAULT_MODEL
    assert str(home) in capsys.readouterr().err


def test_a_typed_provider_brings_its_model_with_it(home, project):
    # `--provider claude-code` alone has to be enough. Without this it would run
    # Claude Code against the model tag the file picked for Ollama.
    args = namespace(provider="codex")
    config.load(project).apply(args)
    assert (args.model, args.num_ctx) == ("gpt-5.5", 400_000)


def test_a_typed_provider_does_not_overrule_a_pinned_model(home, project):
    write(home, model="qwen2.5-coder:7b")
    args = namespace(provider="claude-code")
    config.load(project).apply(args)
    assert args.model == "qwen2.5-coder:7b"


def test_a_typed_model_still_wins_over_the_backend_default(home, project):
    args = namespace(provider="claude-code", model="sonnet")
    config.load(project).apply(args)
    assert args.model == "sonnet"


# --- argv normalization -------------------------------------------------------
#
# argparse before CPython 3.12.7 drops a positional that follows a flag, which
# made `config set --workspace <key> <value>` fail on a distro python while
# passing everywhere else. These pin the reordering rather than the interpreter.


def test_config_argv_moves_positionals_ahead_of_flags():
    from bkht.coder.cli import config_argv

    line = ["config", "set", "--workspace", "num_ctx", "8192", "--cwd", "/tmp/x"]
    assert config_argv(line) == [
        "config", "set", "num_ctx", "8192", "--workspace", "--cwd", "/tmp/x",
    ]


def test_config_argv_keeps_a_valued_flag_with_its_value():
    from bkht.coder.cli import config_argv

    # --cwd's value must not be mistaken for a positional and hoisted away
    # from the flag it belongs to.
    assert config_argv(["config", "--cwd", "/tmp/x", "get", "model"]) == [
        "config", "get", "model", "--cwd", "/tmp/x",
    ]


def test_config_argv_leaves_an_already_ordered_line_alone():
    from bkht.coder.cli import config_argv

    line = ["config", "set", "num_ctx", "8192", "--workspace"]
    assert config_argv(line) == line


def test_config_set_takes_the_flag_before_the_key(home, project, capsys):
    status, _, err = coder(
        capsys, "config", "set", "--workspace", "num_ctx", "8192", "--cwd", str(project),
    )
    assert (status, err) == (0, "")
    assert json.loads((project / config.WORKSPACE_NAME).read_text()) == {"num_ctx": 8192}


def test_the_update_check_is_on_by_default_and_can_be_turned_off(tmp_path, monkeypatch):
    # The one setting that governs a request leaving the machine, so both
    # directions are asserted rather than assumed from the field list.
    monkeypatch.setattr(config, "GLOBAL_PATH", tmp_path / "config.json")
    assert config.load(tmp_path).update_check is True

    config.set_value("update_check", "false", root=tmp_path)
    settings = config.load(tmp_path)
    assert settings.update_check is False
    assert settings.source("update_check") == config.GLOBAL


# --- the two tool switches --------------------------------------------------


def test_planning_and_delegation_are_on_unless_turned_off(home, project):
    args = namespace()
    config.load(project).apply(args)
    assert args.no_planning is False
    assert args.no_delegation is False


def test_turning_them_off_in_a_file_sets_the_negative_switches(home, project):
    # Two tools, one line each. The registry's standing argument -- every extra
    # tool costs selection accuracy on a small model -- is why they can go.
    write(home, planning=False, delegation=False)
    args = namespace()
    config.load(project).apply(args)
    assert args.no_planning is True
    assert args.no_delegation is True


def test_the_setting_named_planning_does_not_touch_the_plan_mode_switch(home, project):
    # `--plan` is the permission mode and `planning` is the tool. They were one
    # word apart, and a config key called `plan` would have silently rewritten
    # the mode flag.
    write(home, planning=False, mode="auto")
    args = namespace()
    config.load(project).apply(args)
    assert args.no_planning is True
    assert (args.auto, args.plan) == (True, False)
