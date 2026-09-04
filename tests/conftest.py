"""Shared fixtures. Nothing here needs a model."""

from pathlib import Path

import pytest

from bkht.coder.tools.base import Workspace, set_output_budget


@pytest.fixture(autouse=True)
def default_output_budget():
    """Restore the tool-output cap around every test.

    The cap is process-wide: it is sized once at startup from ``num_ctx``,
    because ``truncate`` is called from six tools in five modules. That is right
    for a run and wrong for a suite, where one test sizing it for a small window
    would silently shorten the output of every test after it.
    """
    set_output_budget(0)
    yield
    set_output_budget(0)


@pytest.fixture(autouse=True)
def no_backend_probe(monkeypatch):
    """Answer the default-backend probe without asking the network.

    Resolving a provider nobody named asks whether anything is serving on the
    default endpoint. That is right for a session and wrong for a suite: the
    answer would depend on what happens to be running on the machine, so the
    same test would pass on a laptop with Ollama up and fail on CI.

    Stubbed to "the configured backend, no notice", which is the shape every
    test but the fallback's own expects. Those override it.
    """
    monkeypatch.setattr(
        "bkht.coder.config.settle", lambda name, host: (name, "")
    )


@pytest.fixture(autouse=True)
def no_personal_agent_directory(monkeypatch, tmp_path):
    """Point the global ``agent/`` root somewhere empty.

    It is a real directory in a real home, so without this the suite would
    load whatever the developer running it happens to keep there -- and a test
    that passes on one machine and fails on the next is worse than no test.
    """
    from bkht.coder import layout

    monkeypatch.setattr(layout, "GLOBAL_ROOT", tmp_path / "no-such-home" / "agent")


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """An empty workspace rooted at a temp directory."""
    return Workspace(tmp_path)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A small project tree used across the tool and loop tests."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def main():\n    total = 0\n    for i in range(10):\n        total += i\n"
        "    return total\n"
    )
    (tmp_path / "src" / "util.py").write_text("def helper(x):\n    return x * 2\n")
    (tmp_path / "README.md").write_text("# demo\n\nA demo project.\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("module.exports = {}\n")
    return tmp_path


@pytest.fixture(autouse=True)
def neutral_ci_environment(monkeypatch):
    """Hide the runner's own CI variables from the suite.

    `coder review` detects GitHub Actions and GitLab from the environment, and
    the suite runs *inside* GitHub Actions. Without this a test asserting what a
    plain run does would assert it on a machine where no run is plain -- and it
    would pass on a laptop and fail only once pushed. Tests that want CI pass an
    environment in explicitly.
    """
    for name in ("CI", "GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY", "GITLAB_CI"):
        monkeypatch.delenv(name, raising=False)


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        default=None,
        help="Ollama model for the live tests. Defaults to the provider default.",
    )


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Keep the suite out of the developer's real ``~/.bkht-coder``.

    Autouse because these stores load themselves: ``Permissions`` reads the rule
    file whenever it is given a workspace, and skill and command discovery read
    their global directories on every scan. Any of them would otherwise pick up
    -- and in the rules' case, write -- whatever is on the machine running the
    tests.
    """
    from bkht.coder import commands, config, rules, skills

    monkeypatch.setattr(rules, "RULES_PATH", tmp_path / "permissions.json")
    monkeypatch.setattr(config, "GLOBAL_PATH", tmp_path / "config.json")
    monkeypatch.setattr(skills, "GLOBAL_ROOT", tmp_path / "global-skills")
    monkeypatch.setattr(commands, "GLOBAL_ROOT", tmp_path / "global-commands")


@pytest.fixture(autouse=True)
def no_update_check(monkeypatch, tmp_path):
    """Keep the release check off the network and out of the real cache.

    The check is deliberately easy to reach -- `doctor` runs it, and so does
    starting a session -- so without this a test that never mentions updates
    would still make a request to GitHub, and write its answer to the
    developer's own ~/.bkht-coder. Tests about the check patch over this with
    what they want the request to have returned.
    """
    from bkht.coder import update

    monkeypatch.setattr(update, "CACHE", tmp_path / "update.json")
    monkeypatch.setattr(update, "refresh", lambda: None)
