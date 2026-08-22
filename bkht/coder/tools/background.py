"""Long-running commands, kept off the turn.

The shell tool waits, and times out at sixty seconds. That is right for the
commands an agent mostly runs -- a test suite, a build, a `git log` -- and
useless for the ones that are supposed to never finish. A dev server started
through it either kills the turn or kills itself.

So those get their own tool, which returns as soon as the process is up and
hands back an id. Output goes to a file the model can ask for later, because a
server that logs a line a second would otherwise eat the context window while
nobody was reading it.

One tool with four actions rather than four tools: the tool set is the scarce
resource on a small model, and `start`, `output`, `stop`, `list` cost far less
inside one schema than spread across four.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..session import STATE_DIR
from .base import Tool, ToolError, ToolResult, Workspace
from .shell import NO_SHELL, resolve_shell

JOBS_DIR = STATE_DIR / "jobs"

START, OUTPUT, STOP, LIST = "start", "output", "stop", "list"
ACTIONS = (START, OUTPUT, STOP, LIST)

# How long a terminated process is given to go quietly before it is killed.
GRACE_SECONDS = 5
# Log tail returned by `output`. The end, not the beginning: for a server, the
# last thing it said is the only thing worth reading.
MAX_OUTPUT_LINES = 200

RUNNING = "running"


@dataclass
class Job:
    """One background process and the file its output is going to."""

    id: str
    command: str
    process: subprocess.Popen
    log: Path

    def state(self) -> str:
        code = self.process.poll()
        return RUNNING if code is None else f"exited {code}"

    def tail(self, limit: int = MAX_OUTPUT_LINES) -> str:
        try:
            text = self.log.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"(could not read the log: {exc})"

        lines = text.splitlines()
        if len(lines) <= limit:
            return text.rstrip()
        return "\n".join([f"[earlier {len(lines) - limit} line(s) omitted]"] + lines[-limit:])


@dataclass
class Jobs:
    """Every background process this session started.

    A session owns its jobs and outlives none of them. An agent that leaves a
    server running after the user has quit is a bug -- the user did not start
    it, cannot see it, and has no obvious way to find it again.
    """

    directory: Path = None
    jobs: dict[str, Job] = field(default_factory=dict)
    _next: int = 1

    def __post_init__(self) -> None:
        if self.directory is None:
            self.directory = JOBS_DIR / str(os.getpid())

    def start(self, command: str, cwd: Path) -> Job:
        shell = resolve_shell()
        if shell is None:
            raise ToolError(NO_SHELL)

        self.directory.mkdir(parents=True, exist_ok=True)
        identifier = str(self._next)
        self._next += 1
        log = self.directory / f"{identifier}.log"

        try:
            handle = log.open("w", encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolError(f"could not open a log file for the job: {exc}") from None

        try:
            process = subprocess.Popen(
                [*shell.argv, command],
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                # Its own process group, so a server that forks can be stopped
                # whole. Killing only the shell would leave the thing it started.
                **_group_flags(),
            )
        except OSError as exc:
            handle.close()
            raise ToolError(f"could not start the command: {exc}") from None
        finally:
            # The child holds its own descriptor; ours would otherwise keep the
            # file open for the life of the session.
            handle.close()

        job = Job(id=identifier, command=command, process=process, log=log)
        self.jobs[identifier] = job
        return job

    def get(self, identifier: str) -> Job:
        job = self.jobs.get(str(identifier).strip())
        if job is None:
            known = ", ".join(sorted(self.jobs)) or "none"
            raise ToolError(f"no background job with id '{identifier}'. Running jobs: {known}")
        return job

    def stop(self, identifier: str) -> str:
        """Terminate a job, killing it if it will not go. Safe to repeat."""
        job = self.get(identifier)
        if job.process.poll() is not None:
            return f"job {job.id} had already {job.state()}"

        _signal_group(job.process, signal.SIGTERM)
        try:
            job.process.wait(timeout=GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_group(job.process, signal.SIGKILL)
            job.process.wait(timeout=GRACE_SECONDS)
        return f"stopped job {job.id} ({job.command})"

    def stop_all(self) -> None:
        """Teardown. Never raises: it runs while the process is on its way out."""
        for identifier in list(self.jobs):
            try:
                self.stop(identifier)
            except (ToolError, OSError, subprocess.TimeoutExpired):
                continue

    def running(self) -> list[Job]:
        return [job for job in self.listing() if job.state() == RUNNING]

    def listing(self) -> list[Job]:
        return [self.jobs[identifier] for identifier in sorted(self.jobs, key=int)]

    def summary(self) -> str:
        if not self.jobs:
            return "no background jobs"
        return "\n".join(f"  {job.id}  {job.state():<12} {job.command}" for job in self.listing())


def _group_flags() -> dict:
    """Put the child in its own group, in whichever way this platform has."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _signal_group(process: subprocess.Popen, sig) -> None:
    """Signal the whole group, falling back to the one process we know about."""
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), sig)
            return
    except (OSError, AttributeError):
        pass

    try:
        process.kill() if sig == getattr(signal, "SIGKILL", None) else process.terminate()
    except OSError:
        pass


def register_background_tools(registry, workspace: Workspace, jobs: Jobs):
    """Add the background tool, backed by ``jobs``."""

    def background(action: str, command: str = "", job_id: str = "") -> ToolResult:
        if action not in ACTIONS:
            raise ToolError(f"action must be one of {', '.join(ACTIONS)}")

        if action == LIST:
            return ToolResult.success(jobs.summary())

        if action == START:
            if not command.strip():
                raise ToolError("start needs a command")
            job = jobs.start(command, workspace.root)
            return ToolResult.success(
                f"started job {job.id}. It is running in the background; use "
                f"action 'output' with job_id {job.id} to read what it prints."
            )

        if not job_id.strip():
            raise ToolError(f"{action} needs a job_id. Use action 'list' to see them.")

        if action == STOP:
            return ToolResult.success(jobs.stop(job_id))

        job = jobs.get(job_id)
        return ToolResult.success(f"job {job.id} is {job.state()}\n{job.tail()}".rstrip())

    registry.add(
        Tool(
            name="background",
            description=(
                "Run a command that is not expected to finish, such as a dev "
                "server, without waiting for it. Use action 'start' with a "
                "command, then 'output' with the returned job_id to read what "
                "it printed, 'stop' to end it, or 'list' to see them all. For "
                "commands that do finish, use the shell tool instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": f"One of: {', '.join(ACTIONS)}.",
                    },
                    "command": {"type": "string", "description": "The command to start."},
                    "job_id": {"type": "string", "description": "Which job to read or stop."},
                },
                "required": ["action"],
            },
            run=background,
            mutating=True,
        )
    )
    return registry


__all__ = ["Job", "Jobs", "register_background_tools"]
