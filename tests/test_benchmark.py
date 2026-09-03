"""The benchmark's reading and reporting. Nothing here needs a model."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"


@pytest.fixture(scope="module")
def benchmark():
    """Loaded by path: `scripts/` is not a package, and should not become one."""
    spec = importlib.util.spec_from_file_location("benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session(path: Path, *records: dict) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_only_outcome_records_are_read(benchmark, tmp_path):
    path = session(
        tmp_path / "s.jsonl",
        {"type": "session", "id": "s"},
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "outcome", "stopped": "answered", "seconds": 12.5, "iterations": 3},
    )
    assert [r["stopped"] for r in benchmark.outcomes(path)] == ["answered"]


def test_a_file_cut_off_mid_write_still_yields_what_it_has(benchmark, tmp_path):
    """A killed run leaves a half-written line; the turns before it are real."""
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"type": "outcome", "stopped": "answered", "seconds": 1.0}) + "\n"
        + '{"type": "outcome", "stopp'
    )
    assert len(benchmark.outcomes(path)) == 1


def test_a_session_that_was_never_written_is_not_an_error(benchmark, tmp_path):
    assert benchmark.outcomes(tmp_path / "absent.jsonl") == []


def test_the_summary_uses_medians_so_one_timeout_cannot_move_it(benchmark):
    rows = [
        {"task": "a", "seconds": 10.0, "iterations": 3, "tool_calls": 4, "stopped": "answered", "failure": ""},
        {"task": "b", "seconds": 12.0, "iterations": 3, "tool_calls": 4, "stopped": "answered", "failure": ""},
        {"task": "c", "seconds": 900.0, "iterations": 25, "tool_calls": 30, "stopped": "time-cap", "failure": ""},
    ]
    summary = benchmark.summarise(rows)
    assert summary["median_seconds"] == 12.0
    assert summary["answered"] == 2 and summary["tasks"] == 3
    assert summary["stops"] == {"answered": 2, "time-cap": 1}


def test_a_run_that_never_reached_the_model_is_reported_not_dropped(benchmark):
    rows = [{"task": "a", "seconds": 0.0, "iterations": 0, "tool_calls": 0,
             "stopped": "", "failure": "connection refused"}]
    summary = benchmark.summarise(rows)
    assert summary["stops"] == {"connection refused": 1}
    assert "connection refused" in benchmark.render(rows, summary)


def run(label, **seconds):
    return {"label": label, "rows": [
        {"task": t, "seconds": s, "iterations": 2, "tool_calls": 2, "stopped": "answered", "failure": ""}
        for t, s in seconds.items()
    ]}


def test_comparing_reports_the_change_in_the_direction_it_happened(benchmark):
    said = benchmark.compare(run("before", a=10.0, b=10.0), run("after", a=8.0, b=7.0))
    assert "20.0s -> 15.0s" in said
    assert "-25.0%" in said


def test_a_regression_is_not_dressed_up(benchmark):
    said = benchmark.compare(run("before", a=10.0), run("after", a=15.0))
    assert "+50.0%" in said


def test_only_the_tasks_both_runs_did_are_compared(benchmark):
    said = benchmark.compare(run("before", a=10.0, b=10.0), run("after", a=5.0))
    assert "1 shared task" in said
    assert "10.0s -> 5.0s" in said


def test_two_runs_with_nothing_in_common_say_so(benchmark):
    said = benchmark.compare(run("before", a=1.0), run("after", z=1.0))
    assert "share no tasks" in said


def test_every_benchmark_task_is_read_only(benchmark):
    """A task that edits the tree makes the second run measure different work."""
    writes = ("add ", "create ", "fix ", "rename ", "delete ", "remove ", "write ")
    for task in benchmark.TASKS:
        assert not task.lower().startswith(writes), task
