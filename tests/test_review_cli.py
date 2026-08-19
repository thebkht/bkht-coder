"""The review command: flag routing, selection, and the fix phase."""

import json
import subprocess

import pytest

from bkht.coder.cli import build_parser
from bkht.coder.review import cli as review_cli
from bkht.coder.review.reviewer import CONFIRMED, Finding

from fakes import FakeProvider, call


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "user.email=t@localhost", "-c", "user.name=t", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "calc.py").write_text("def average(n):\n    return sum(n) / len(n)\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "initial")
    (tmp_path / "calc.py").write_text("def average(n):\n    return sum(n) / len(n)\n\n")
    return tmp_path


def parse(*argv):
    return build_parser().parse_args(["review", *argv])


# --- flags ------------------------------------------------------------------


def test_review_is_a_subcommand():
    assert parse().command == "review"


def test_defaults_are_uncommitted_and_verified():
    args = parse()
    assert not args.staged and not args.base and args.range is None
    assert not args.no_verify and not args.json and not args.fix


@pytest.mark.parametrize(
    "argv,attribute,value",
    [
        (["--staged"], "staged", True),
        (["--base", "main"], "base", "main"),
        (["HEAD~3..HEAD"], "range", "HEAD~3..HEAD"),
        (["--json"], "json", True),
        (["--output", "r.md"], "output", "r.md"),
        (["--fix"], "fix", True),
        (["--no-verify"], "no_verify", True),
    ],
)
def test_each_documented_flag_parses(argv, attribute, value):
    assert getattr(parse(*argv), attribute) == value


def test_files_takes_several_paths():
    assert parse("--files", "a.py", "b.py").files == ["a.py", "b.py"]


def test_dimension_is_repeatable_and_validated():
    assert parse("--dimension", "security", "--dimension", "tests").dimension == [
        "security",
        "tests",
    ]
    with pytest.raises(SystemExit):
        parse("--dimension", "vibes")


def test_gather_routes_files_before_the_diff(repo):
    args = parse("--files", "calc.py")
    args.cwd = str(repo)
    [file] = review_cli.gather(repo, args)
    assert file.status == "whole file"


# --- selection --------------------------------------------------------------


FINDINGS = [
    Finding("a.py", 1, "high", "correctness", "first", "", "", CONFIRMED),
    Finding("a.py", 2, "low", "tests", "second", "", "", CONFIRMED),
    Finding("b.py", 3, "medium", "security", "third", "", "", CONFIRMED),
]


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("all", ["first", "second", "third"]),
        ("2", ["second"]),
        ("1,3", ["first", "third"]),
        ("1, 3", ["first", "third"]),
        ("", []),
        ("none", []),
    ],
)
def test_selection_accepts_the_documented_forms(answer, expected, capsys):
    chosen = review_cli.choose(FINDINGS, ask=lambda _: answer)
    assert [f.summary for f in chosen] == expected


def test_out_of_range_and_junk_are_ignored_not_fatal(capsys):
    chosen = review_cli.choose(FINDINGS, ask=lambda _: "1,99,banana")
    assert [f.summary for f in chosen] == ["first"]
    assert "Ignoring" in capsys.readouterr().out


def test_an_abandoned_prompt_selects_nothing():
    def refuse(_):
        raise EOFError

    assert review_cli.choose(FINDINGS, ask=refuse) == []


# --- fixing -----------------------------------------------------------------


def test_fix_goes_through_the_permission_gate(repo):
    # Review is read-only; fixing is a second phase, gated like any other edit.
    asked = []
    finding = Finding("calc.py", 2, "high", "correctness", "divides by zero", "", "", CONFIRMED)
    provider = FakeProvider([call("write_file", path="calc.py", content="broken")])

    review_cli.fix(
        [finding], provider, repo, prompt=lambda q, b: asked.append(q) or "n"
    )
    assert asked, "the fix phase wrote without asking"
    assert "broken" not in (repo / "calc.py").read_text()


def test_fix_applies_an_approved_change(repo):
    finding = Finding("calc.py", 2, "high", "correctness", "divides by zero", "", "", CONFIRMED)
    provider = FakeProvider(
        [
            call("edit_file", path="calc.py", old_string="sum(n) / len(n)", new_string="sum(n) / len(n) if n else 0.0"),
            "Guarded the empty case.",
        ]
    )

    assert review_cli.fix([finding], provider, repo, prompt=lambda q, b: "y") == 1
    assert "if n else 0.0" in (repo / "calc.py").read_text()


def test_the_fix_prompt_carries_the_whole_finding(repo):
    finding = Finding("calc.py", 2, "high", "correctness", "divides by zero", "average([]) raises", "guard it", CONFIRMED)
    provider = FakeProvider(["I would guard the empty case."])

    review_cli.fix([finding], provider, repo, prompt=lambda q, b: "n")
    task = provider.calls[0][-1]["content"]
    assert "calc.py:2" in task
    assert "divides by zero" in task
    assert "average([]) raises" in task
    assert "guard it" in task


# --- end to end through run() -----------------------------------------------


def run_review(repo, monkeypatch, script, *argv):
    args = parse(*argv)
    args.cwd = str(repo)
    monkeypatch.setattr(
        "bkht.coder.review.cli.OllamaProvider", lambda **kw: FakeProvider(script)
    )
    return review_cli.run(args)


def test_a_clean_tree_is_reported_not_an_error(tmp_path, monkeypatch, capsys):
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "a.py").write_text("x = 1\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "initial")

    assert run_review(tmp_path, monkeypatch, []) == 0
    assert "no changes were found" in capsys.readouterr().err


def test_not_a_repository_exits_two(tmp_path, monkeypatch, capsys):
    assert run_review(tmp_path, monkeypatch, []) == 2
    assert "not a git repository" in capsys.readouterr().err


def test_json_output_is_machine_readable(repo, monkeypatch, capsys):
    run_review(repo, monkeypatch, ["[]"] * 4, "--json", "--no-verify")
    payload = json.loads(capsys.readouterr().out)
    assert payload["findings"] == [] and payload["units"] == 1


def test_json_exits_one_when_there_are_findings(repo, monkeypatch, capsys):
    script = [
        json.dumps([{"file": "calc.py", "line": 3, "severity": "high",
                     "summary": "trailing blank line changes nothing",
                     "scenario": "x"}]),
        "[]", "[]", "[]",
    ]
    assert run_review(repo, monkeypatch, script, "--json", "--no-verify") == 1


def test_findings_alone_do_not_fail_a_plain_run(repo, monkeypatch, capsys):
    # A report is not a build failure unless CI asked for JSON.
    script = [
        json.dumps([{"file": "calc.py", "line": 3, "severity": "high",
                     "summary": "trailing blank line", "scenario": "x"}]),
        "[]", "[]", "[]",
    ]
    assert run_review(repo, monkeypatch, script, "--no-verify", "--quiet") == 0


def test_output_writes_markdown(repo, monkeypatch, tmp_path, capsys):
    report = tmp_path / "report.md"
    run_review(repo, monkeypatch, ["[]"] * 4, "--no-verify", "--quiet", "--output", str(report))
    assert report.read_text().startswith("# Code review")


def test_one_dimension_runs_one_pass(repo, monkeypatch, capsys):
    args = parse("--dimension", "correctness", "--no-verify", "--quiet")
    args.cwd = str(repo)
    provider = FakeProvider(["[]"])
    monkeypatch.setattr("bkht.coder.review.cli.OllamaProvider", lambda **kw: provider)

    review_cli.run(args)
    assert len(provider.calls) == 1
