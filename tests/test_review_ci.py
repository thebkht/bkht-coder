"""Review output for CI: detection, log sections, and native annotations."""

import json
import subprocess

import pytest

from bkht.coder.cli import build_parser
from bkht.coder.review import ci
from bkht.coder.review import cli as review_cli
from bkht.coder.review.reviewer import CONFIRMED, PLAUSIBLE, Finding, ReviewResult

HIGH = Finding("calc.py", 12, "high", "correctness", "divides by zero", "average([]) raises", "guard it", CONFIRMED)
MEDIUM = Finding("api.py", 7, "medium", "security", "path is not validated", "../ escapes the root", "", CONFIRMED)
LOW = Finding("calc.py", 40, "low", "tests", "no test for the empty case", "", "", PLAUSIBLE)


def result(findings=(), **kwargs):
    base = dict(files=["calc.py", "api.py"], units=2, candidates=len(findings))
    base.update(kwargs)
    return ReviewResult(findings=list(findings), **base)


def parse(*argv):
    return build_parser().parse_args(["review", *argv])


class Stream:
    """Collects written lines, so a listener can be read back."""

    def __init__(self):
        self.lines = []

    def write(self, text):
        if text.strip("\n"):
            self.lines.append(text.rstrip("\n"))

    def flush(self):
        pass


# --- detection --------------------------------------------------------------


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"GITHUB_ACTIONS": "true"}, ci.GITHUB),
        ({"GITLAB_CI": "true"}, ci.GITLAB),
        ({"CI": "true"}, ci.GENERIC),
        ({"CI": "1"}, ci.GENERIC),
        ({}, None),
        ({"CI": "false"}, None),
    ],
)
def test_detect_reads_the_environment(env, expected):
    assert ci.detect(env) == expected


def test_the_specific_platform_wins_over_the_generic_flag():
    # Both are set on a real Actions runner; picking CI first costs annotations.
    assert ci.detect({"GITHUB_ACTIONS": "true", "CI": "true"}) == ci.GITHUB
    assert ci.detect({"GITLAB_CI": "true", "CI": "true"}) == ci.GITLAB


def test_resolve_honours_the_flag_over_the_environment():
    assert ci.resolve(None, {"GITLAB_CI": "true"}) == ci.GITLAB
    assert ci.resolve("off", {"GITLAB_CI": "true"}) is None
    assert ci.resolve("github", {"GITLAB_CI": "true"}) == ci.GITHUB


def test_a_bare_ci_flag_forces_output_outside_ci():
    assert ci.resolve("auto", {}) == ci.GENERIC
    assert ci.resolve("auto", {"GITHUB_ACTIONS": "true"}) == ci.GITHUB


# --- flags and listener selection -------------------------------------------


def test_the_new_flags_parse():
    assert parse().ci is None
    assert parse("--ci").ci == "auto"
    assert parse("--ci", "gitlab").ci == "gitlab"
    assert parse("--code-quality", "r.json").code_quality == "r.json"
    with pytest.raises(SystemExit):
        parse("--ci", "jenkins")


def test_ci_output_is_chosen_inside_ci():
    listener, kind = review_cli.choose_listener(parse(), {"GITHUB_ACTIONS": "true"})
    assert kind == ci.GITHUB
    assert isinstance(listener, ci.GitHubActions)


def test_off_beats_a_ci_environment():
    listener, kind = review_cli.choose_listener(parse("--ci", "off"), {"CI": "true"})
    assert kind is None
    assert isinstance(listener, review_cli.Progress)


def test_quiet_beats_ci():
    # Someone who asked for silence gets it, pipeline or not.
    listener, kind = review_cli.choose_listener(parse("--quiet"), {"CI": "true"})
    assert kind == ci.GENERIC
    assert type(listener) is review_cli.ReviewListener


def test_a_plain_shell_still_gets_the_interactive_progress():
    listener, kind = review_cli.choose_listener(parse(), {})
    assert kind is None
    assert isinstance(listener, review_cli.Progress)


# --- log sections -----------------------------------------------------------


def drive(listener):
    """One review's worth of events, ending in verification."""
    listener.on_pass(1, 2, "correctness")
    listener.on_candidates("correctness", 2)
    listener.on_pass(1, 2, "security")
    listener.on_verify(1, 1, HIGH)
    listener.on_verdict(HIGH, True, "guarded above")
    listener.finish()


def test_github_groups_are_balanced():
    stream = Stream()
    drive(ci.GitHubActions(stream=stream))
    opens = [line for line in stream.lines if line.startswith("::group::")]
    closes = [line for line in stream.lines if line == "::endgroup::"]
    assert len(opens) == len(closes) == 3
    assert stream.lines[-1] == "::endgroup::"


def test_gitlab_sections_are_balanced_and_uniquely_named():
    stream = Stream()
    drive(ci.GitLab(stream=stream))
    starts = [line for line in stream.lines if "section_start:" in line]
    ends = [line for line in stream.lines if "section_end:" in line]
    assert len(starts) == len(ends) == 3
    names = [line.split(":")[2].split("[")[0] for line in starts]
    assert len(set(names)) == 3
    assert all(" " not in name for name in names)


def test_finish_is_safe_to_call_twice():
    stream = Stream()
    listener = ci.GitHubActions(stream=stream)
    listener.on_pass(1, 1, "correctness")
    listener.finish()
    listener.finish()
    assert stream.lines.count("::endgroup::") == 1


def test_the_generic_listener_writes_plain_lines():
    stream = Stream()
    drive(ci.CIListener(stream=stream))
    assert all(not line.startswith("::") and "\033" not in line for line in stream.lines)
    assert f"{ci.PREFIX} verify 1/1: calc.py:12" in stream.lines
    assert f"{ci.PREFIX} refuted - guarded above" in stream.lines


def test_a_verdict_that_survives_is_not_announced():
    stream = Stream()
    ci.CIListener(stream=stream).on_verdict(HIGH, False, "")
    assert stream.lines == []


# --- GitHub annotations -----------------------------------------------------


def test_severity_picks_the_annotation_level():
    stream = Stream()
    ci.annotate(result([HIGH, MEDIUM, LOW]), stream)
    assert stream.lines[0].startswith("::error ")
    assert stream.lines[1].startswith("::warning ")
    assert stream.lines[2].startswith("::notice ")


def test_an_annotation_carries_the_file_and_line():
    stream = Stream()
    ci.annotate(result([HIGH]), stream)
    assert "file=calc.py,line=12," in stream.lines[0]
    assert "divides by zero" in stream.lines[0]
    assert "How it fails: average([]) raises" in stream.lines[0].replace("%0A", "\n")


def test_data_escaping_keeps_a_command_on_one_line():
    # A raw newline ends the command, truncating the annotation silently.
    assert ci.escape_data("a\nb") == "a%0Ab"
    assert ci.escape_data("100%") == "100%25"
    assert ci.escape_data("a\r\nb") == "a%0D%0Ab"


def test_property_escaping_also_covers_the_delimiters():
    assert ci.escape_property("a,b") == "a%2Cb"
    assert ci.escape_property("file.py:12") == "file.py%3A12"


def test_a_multiline_finding_still_emits_one_command():
    nasty = Finding("calc.py", 3, "high", "a,b:c", "50% off\nsecond line", "", "", CONFIRMED)
    stream = Stream()
    ci.annotate(result([nasty]), stream)
    assert len(stream.lines) == 1
    assert "\n" not in stream.lines[0]
    assert "title=high%3A a%2Cb%3Ac" in stream.lines[0]
    assert "50%25 off%0Asecond line" in stream.lines[0]


# --- job summary ------------------------------------------------------------


def test_the_summary_is_appended_not_overwritten(tmp_path):
    path = tmp_path / "summary.md"
    path.write_text("an earlier step wrote this\n")
    written = ci.summary(result([HIGH]), {"GITHUB_STEP_SUMMARY": str(path)})
    assert written == str(path)
    text = path.read_text()
    assert "an earlier step wrote this" in text
    assert "# Code review" in text and "divides by zero" in text


def test_no_summary_variable_is_not_an_error():
    assert ci.summary(result([HIGH]), {}) is None


# --- GitLab Code Quality ----------------------------------------------------


def test_the_report_has_the_codeclimate_shape():
    [entry] = json.loads(ci.code_quality(result([HIGH])))
    assert entry["check_name"] == "correctness"
    assert entry["severity"] == "critical"
    assert entry["location"] == {"path": "calc.py", "lines": {"begin": 12}}
    assert entry["description"].startswith("divides by zero")


def test_severities_map_onto_gitlabs_own():
    entries = json.loads(ci.code_quality(result([HIGH, MEDIUM, LOW])))
    assert [e["severity"] for e in entries] == ["critical", "major", "minor"]


def test_a_fingerprint_is_stable_across_runs():
    # GitLab tracks a finding by this; a value that moves makes every defect new.
    once = json.loads(ci.code_quality(result([HIGH, MEDIUM])))
    twice = json.loads(ci.code_quality(result([HIGH])))
    assert once[0]["fingerprint"] == twice[0]["fingerprint"]
    assert once[0]["fingerprint"] != once[1]["fingerprint"]


def test_a_different_line_is_a_different_finding():
    moved = Finding("calc.py", 13, "high", "correctness", "divides by zero", "", "", CONFIRMED)
    assert ci.fingerprint(HIGH) != ci.fingerprint(moved)


# --- the command end to end -------------------------------------------------


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


@pytest.fixture
def canned(monkeypatch):
    """Replace the review itself, so these tests exercise only the wiring."""

    def install(findings):
        class Stub:
            def __init__(self, *a, **kw):
                pass

            def review(self, files, budget=None):
                return result(findings, files=[f.path for f in files])

        monkeypatch.setattr(review_cli, "Reviewer", Stub)

    return install


def run(repo, *argv):
    args = parse(*argv)
    args.cwd = str(repo)
    return review_cli.run(args)


def test_findings_fail_the_job_in_ci(repo, canned, capsys):
    canned([HIGH])
    assert run(repo, "--ci", "github") == 1
    assert "::error file=calc.py,line=12" in capsys.readouterr().out


def test_a_clean_review_passes(repo, canned, capsys):
    canned([])
    assert run(repo, "--ci", "github") == 0


def test_gitlab_writes_the_report_where_gitlab_looks(repo, canned):
    canned([HIGH])
    run(repo, "--ci", "gitlab")
    [entry] = json.loads((repo / "gl-code-quality-report.json").read_text())
    assert entry["location"]["path"] == "calc.py"


def test_code_quality_can_be_pointed_anywhere(repo, canned, tmp_path):
    canned([HIGH])
    target = tmp_path / "elsewhere.json"
    run(repo, "--ci", "github", "--code-quality", str(target))
    assert json.loads(target.read_text())


def test_quiet_in_ci_still_reports(repo, canned, capsys, tmp_path):
    # --quiet suppresses progress, not the report: it once returned a listener
    # with no finish() while still taking the CI path, and crashed on it.
    canned([HIGH])
    target = tmp_path / "q.json"
    assert run(repo, "--ci", "github", "--quiet", "--code-quality", str(target)) == 1
    out = capsys.readouterr()
    assert "::error file=calc.py,line=12" in out.out
    assert "::group::" not in out.out and "::group::" not in out.err
    assert json.loads(target.read_text())


def test_finish_is_part_of_the_listener_protocol():
    # run() calls it on whatever listener it was handed.
    from bkht.coder.review.reviewer import ReviewListener

    assert ReviewListener().finish() is None
    assert review_cli.Progress().finish() is None


def test_fix_is_skipped_in_ci(repo, canned, capsys):
    # --fix asks which findings to fix; there is nobody there to answer.
    canned([HIGH])
    assert run(repo, "--ci", "generic", "--fix") == 1
    assert "Skipping --fix" in capsys.readouterr().err
