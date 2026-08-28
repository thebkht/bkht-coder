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
        config.parse("provider", "codex")


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
        auto=None, plan=None,
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
