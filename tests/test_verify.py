"""Static checks on a file the model is about to write."""

import subprocess
from pathlib import Path

import pytest

from bkht.coder import verify


@pytest.fixture
def pkg(tmp_path: Path) -> Path:
    """A small package, so relative imports have something real to resolve to."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "session.py").write_text(
        "STATE_DIR = '/tmp'\n\n\nclass Session:\n    pass\n\n\ndef load():\n    return 1\n"
    )
    (package / "commands.py").write_text("from .session import STATE_DIR\n")
    return tmp_path


def unknown(pkg: Path, source: str, where: str = "pkg/commands.py") -> list[str]:
    return verify.unknown_imports(source, pkg / where, pkg)


# --- the failure this exists for ---------------------------------------------


def test_a_name_the_module_does_not_define_is_reported(pkg):
    """The exact edit that broke the package.

    `old_string` matched, the write succeeded, the result said "Edited", and
    every one of those was true of a name that has never existed.
    """
    problems = unknown(pkg, "from .session import STATE_DIR, Input\n")
    assert len(problems) == 1
    assert "session.py" in problems[0] and "`Input`" in problems[0]


def test_the_report_names_the_line_it_is_on(pkg):
    problems = unknown(pkg, "import os\n\nfrom .session import Input\n")
    assert "line 3" in problems[0]


def test_several_missing_names_are_reported_together(pkg):
    problems = unknown(pkg, "from .session import Input, Output\n")
    assert len(problems) == 1
    assert "`Input`" in problems[0] and "`Output`" in problems[0]


def test_names_that_do_exist_are_silent(pkg):
    assert unknown(pkg, "from .session import STATE_DIR, Session, load\n") == []


# --- not crying wolf ---------------------------------------------------------


def test_a_submodule_import_is_not_a_missing_name(pkg):
    """`from . import session` imports a module, not something __init__ defines.

    Every such line in this project would otherwise be reported, and one false
    alarm is enough to teach everybody to ignore the next true one.
    """
    assert unknown(pkg, "from . import session\n") == []


def test_a_subpackage_import_is_not_a_missing_name(pkg):
    (pkg / "pkg" / "review").mkdir()
    (pkg / "pkg" / "review" / "__init__.py").write_text("")
    assert unknown(pkg, "from . import review\n") == []


def test_an_absolute_import_is_never_checked(pkg):
    # There is no way to see the package it names, and guessing produces exactly
    # the false alarm that makes a warning worth ignoring.
    assert unknown(pkg, "from json import NoSuchThing\n") == []


def test_a_module_with_a_star_import_is_not_judged(pkg):
    (pkg / "pkg" / "session.py").write_text("from os.path import *\n")
    assert unknown(pkg, "from .session import anything\n") == []


def test_a_module_with_getattr_answers_for_every_name(pkg):
    (pkg / "pkg" / "session.py").write_text("def __getattr__(name):\n    return name\n")
    assert unknown(pkg, "from .session import anything\n") == []


def test_a_name_bound_anywhere_counts_as_defined(pkg):
    """Deliberately over-inclusive: missing a real mistake beats a false alarm."""
    (pkg / "pkg" / "session.py").write_text(
        "import typing\n\nif typing.TYPE_CHECKING:\n    Handle = int\n"
    )
    assert unknown(pkg, "from .session import Handle\n") == []


def test_an_import_pointing_outside_the_workspace_is_skipped(pkg):
    assert unknown(pkg, "from ...elsewhere import thing\n") == []


def test_a_module_that_does_not_exist_is_skipped(pkg):
    assert unknown(pkg, "from .nothing import thing\n") == []


def test_unparseable_source_reports_no_import_problems(pkg):
    # The syntax error is the finding; a list of imports read out of a file that
    # does not parse would be noise on top of it.
    assert unknown(pkg, "from .session import (\n") == []


# --- syntax ------------------------------------------------------------------


def test_a_syntax_error_is_described_with_its_line():
    message = verify.syntax_error("def go(:\n    pass\n", "x.py")
    assert "x.py" in message and "line 1" in message and "not written" in message


def test_valid_source_has_no_syntax_error():
    assert verify.syntax_error("x = 1\n", "x.py") is None


# --- the two together --------------------------------------------------------


def test_check_refuses_syntax_and_reports_nothing_else(pkg):
    refusal, warnings = verify.check(
        "from .session import Input\ndef go(:\n", pkg / "pkg/commands.py", pkg, "c.py"
    )
    assert refusal and warnings == []


def test_check_warns_when_the_file_parses(pkg):
    refusal, warnings = verify.check(
        "from .session import Input\n", pkg / "pkg/commands.py", pkg, "c.py"
    )
    assert refusal is None and len(warnings) == 1


def test_a_file_that_is_not_python_is_not_checked(pkg):
    assert verify.check("def go(:", pkg / "notes.md", pkg, "notes.md") == (None, [])


# --- running the project's own test command ---------------------------------


class Ran:
    """Stands in for ``subprocess.run``, recording what it was asked to do."""

    def __init__(self, code=0, stdout="", stderr="", raises=None) -> None:
        self.code, self.stdout, self.stderr, self.raises = code, stdout, stderr, raises
        self.calls: list[tuple] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(argv, self.code, self.stdout, self.stderr)


def test_a_passing_suite_says_so(tmp_path):
    report = verify.suite("pytest -q", tmp_path, runner=Ran(0, "3 passed"))
    assert report.ok
    assert report.status == verify.PASSED
    assert report.summary() == "pytest -q passed"


def test_a_failing_suite_carries_the_output_and_the_code(tmp_path):
    # The output is the whole point: a failure the model cannot read is a
    # failure it can only guess at.
    ran = Ran(1, "1 failed, 2 passed", "E   assert 3 == 4")
    report = verify.suite("pytest -q", tmp_path, runner=ran)
    assert not report.ok
    assert report.code == 1
    assert "1 failed" in report.output
    assert "assert 3 == 4" in report.output
    assert report.summary() == "pytest -q failed (exit 1)"


def test_stdout_comes_before_stderr(tmp_path):
    # A failing suite prints its summary to stdout and its tracebacks to
    # stderr, and the summary is the half worth reading first.
    ran = Ran(1, "SUMMARY", "TRACEBACK")
    report = verify.suite("x", tmp_path, runner=ran)
    assert report.output.index("SUMMARY") < report.output.index("TRACEBACK")


def test_a_suite_that_never_returns_is_a_timeout_not_a_failure(tmp_path):
    # Distinct from a failure on purpose: nothing about a timeout tells the
    # model what to change, so the loop must not hand it back as if it did.
    ran = Ran(raises=subprocess.TimeoutExpired("pytest", 120))
    report = verify.suite("pytest", tmp_path, runner=ran)
    assert report.status == verify.TIMED_OUT
    assert not report.ok
    assert "timed out" in report.summary()


def test_a_command_that_cannot_start_is_broken_not_failed(tmp_path):
    ran = Ran(raises=OSError("no such file"))
    report = verify.suite("nope", tmp_path, runner=ran)
    assert report.status == verify.BROKEN
    assert "no such file" in report.output


def test_the_command_runs_in_the_workspace_root(tmp_path):
    ran = Ran(0)
    verify.suite("pytest -q", tmp_path, runner=ran)
    argv, kwargs = ran.calls[0]
    assert kwargs["cwd"] == str(tmp_path)
    assert argv[-1] == "pytest -q"
    # Bounded, always. Esc cannot reach a blocked waitpid, so the timeout is
    # the only thing that ends a suite that hangs.
    assert kwargs["timeout"] == verify.TIMEOUT


@pytest.mark.parametrize(
    "marker, expected",
    [
        ("pytest.ini", "pytest -q"),
        ("Cargo.toml", "cargo test"),
        ("go.mod", "go test ./..."),
        ("package.json", "npm test"),
    ],
)
def test_a_marker_file_suggests_the_command_its_project_uses(tmp_path, marker, expected):
    (tmp_path / marker).write_text("")
    assert verify.detect(tmp_path) == expected


def test_a_tests_directory_alone_suggests_nothing(tmp_path):
    # It says a project has tests, not what runs them, and a wrong suggestion
    # is worse than none -- the user is being asked to trust this enough to
    # write it into their config.
    (tmp_path / "tests").mkdir()
    assert verify.detect(tmp_path) == ""

    (tmp_path / "pyproject.toml").write_text("")
    assert verify.detect(tmp_path) == "pytest -q"


def test_a_rust_project_that_also_has_tests_is_still_a_rust_project(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "Cargo.toml").write_text("")
    assert verify.detect(tmp_path) == "cargo test"


def test_an_empty_directory_suggests_nothing(tmp_path):
    assert verify.detect(tmp_path) == ""
