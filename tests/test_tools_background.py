"""Background processes: starting without waiting, reading later, and cleanup."""

import time

import pytest

from bkht.coder.session import Snapshots
from bkht.coder.tools import build_registry
from bkht.coder.tools.background import Jobs
from bkht.coder.tools.base import ToolError
from bkht.coder.tools.shell import resolve_shell

pytestmark = pytest.mark.skipif(
    resolve_shell() is None, reason="no shell on this machine to run a job with"
)


@pytest.fixture
def jobs(tmp_path):
    store = Jobs(directory=tmp_path / "jobs")
    yield store
    store.stop_all()


@pytest.fixture
def tool(project, jobs):
    registry, _ = build_registry(project, snapshots=Snapshots(), jobs=jobs)
    return registry.get("background")


def wait_for(predicate, timeout: float = 5.0):
    """Poll until true, so the tests do not race a process that is starting."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# --- the store ------------------------------------------------------------


def test_start_returns_before_the_command_finishes(jobs, project):
    # The whole point: a command that never ends must not end the turn.
    began = time.time()
    job = jobs.start("sleep 30", project)
    assert time.time() - began < 5
    assert job.state() == "running"


def test_output_is_captured_to_a_log(jobs, project):
    job = jobs.start("printf 'hello from the job\\n'", project)
    assert wait_for(lambda: "hello from the job" in job.tail())


def test_a_finished_job_reports_its_exit_code(jobs, project):
    job = jobs.start("exit 3", project)
    assert wait_for(lambda: job.state() == "exited 3")


def test_output_survives_the_process_exiting(jobs, project):
    job = jobs.start("printf 'done\\n'", project)
    assert wait_for(lambda: job.state() != "running")
    assert "done" in job.tail()


def test_a_long_log_returns_its_end_not_its_beginning(jobs, project):
    # For a server, the last thing it said is the only thing worth reading.
    job = jobs.start("for i in $(seq 1 500); do echo line$i; done", project)
    assert wait_for(lambda: job.state() != "running")

    tail = job.tail(limit=10)
    assert "line500" in tail and "line1\n" not in tail
    assert "omitted" in tail


def test_stopping_is_idempotent(jobs, project):
    job = jobs.start("sleep 30", project)
    assert "stopped" in jobs.stop(job.id)
    assert "already" in jobs.stop(job.id)


def test_stop_kills_a_process_that_ignores_termination(jobs, project):
    job = jobs.start("trap '' TERM; sleep 30", project)
    assert wait_for(lambda: job.state() == "running")
    jobs.stop(job.id)
    assert job.state() != "running"


def test_teardown_stops_everything_still_running(tmp_path, project):
    store = Jobs(directory=tmp_path / "jobs")
    first = store.start("sleep 30", project)
    second = store.start("sleep 30", project)

    store.stop_all()
    assert first.state() != "running" and second.state() != "running"


def test_an_unknown_id_names_the_ones_that_exist(jobs, project):
    jobs.start("sleep 30", project)
    with pytest.raises(ToolError, match="no background job with id"):
        jobs.get("99")


def test_ids_are_listed_in_order(jobs, project):
    for _ in range(11):
        jobs.start("sleep 30", project)
    assert [job.id for job in jobs.listing()][:3] == ["1", "2", "3"]
    assert jobs.listing()[-1].id == "11"


# --- the tool -------------------------------------------------------------


def test_the_tool_is_absent_in_plan_mode(project, jobs):
    # Plan mode is read-only, and starting a process is not reading.
    registry, _ = build_registry(project, read_only=True, snapshots=Snapshots(), jobs=jobs)
    assert "background" not in registry


def test_the_tool_is_absent_without_a_job_store(project):
    registry, _ = build_registry(project, snapshots=Snapshots())
    assert "background" not in registry


def test_the_tool_needs_permission(tool):
    assert tool.mutating


def test_start_then_output(tool, jobs):
    started = tool.run(action="start", command="printf 'from the tool\\n'")
    assert "started job 1" in started.content
    assert wait_for(lambda: "from the tool" in tool.run(action="output", job_id="1").content)


def test_list_reports_what_is_running(tool):
    tool.run(action="start", command="sleep 30")
    assert "sleep 30" in tool.run(action="list").content


def test_list_with_nothing_running_says_so(tool):
    assert "no background jobs" in tool.run(action="list").content


def test_stop_through_the_tool(tool, jobs):
    tool.run(action="start", command="sleep 30")
    assert "stopped job 1" in tool.run(action="stop", job_id="1").content
    assert jobs.get("1").state() != "running"


def test_an_unknown_action_lists_the_real_ones(tool):
    with pytest.raises(ToolError, match="action must be one of"):
        tool.run(action="restart")


def test_start_without_a_command_says_so(tool):
    with pytest.raises(ToolError, match="start needs a command"):
        tool.run(action="start")


def test_output_without_a_job_id_says_so(tool):
    with pytest.raises(ToolError, match="needs a job_id"):
        tool.run(action="output")
