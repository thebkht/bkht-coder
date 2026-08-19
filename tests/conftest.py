"""Shared fixtures. Nothing here needs a model."""

from pathlib import Path

import pytest

from bkht.coder.tools.base import Workspace


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
