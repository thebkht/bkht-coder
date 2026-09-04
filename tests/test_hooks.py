"""User commands fired on tool events."""

import subprocess
from pathlib import Path

import pytest

from bkht.coder import hooks
from bkht.coder import hooks as hooks_module
from bkht.coder.hooks import POST_TOOL, Hooks


class Ran:
    """Stands in for ``subprocess.run``, recording what it was asked to do.

    One entry per command, so a hook list can be asserted in order.
    """

    def __init__(self, code=0, stdout="", stderr="", raises=None) -> None:
        self.code, self.stdout, self.stderr, self.raises = code, stdout, stderr, raises
        self.calls: list[tuple] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(argv, self.code, self.stdout, self.stderr)


def build(commands, root=".", **kwargs) -> hooks.Hooks:
    return hooks.Hooks(commands=commands, root=root, **kwargs)


# --- reading them out of a config file ---------------------------------------


def test_a_string_is_read_as_one_command():
    # Nobody writing a single formatter wants to type a list around it.
    parsed, problems = hooks.parse({"post_tool": "ruff format ."})
    assert parsed == {"post_tool": ["ruff format ."]}
    assert not problems


def test_a_list_keeps_its_order():
    parsed, _ = hooks.parse({"pre_tool": ["one", "two"]})
    assert parsed["pre_tool"] == ["one", "two"]


def test_an_unknown_event_is_reported_and_skipped():
    # Reported rather than raised: this is read on the way into a session, and
    # a typo should cost the hooks, not the session that could fix them.
    parsed, problems = hooks.parse({"pre_write": ["x"], "turn_end": ["y"]})
    assert parsed == {"turn_end": ["y"]}
    assert "pre_write" in problems[0]


def test_something_that_is_not_an_object_at_all():
    parsed, problems = hooks.parse(["ruff format ."])
    assert parsed == {}
    assert problems


def test_no_hooks_at_all_is_not_a_problem():
    assert hooks.parse(None) == ({}, [])


def test_blank_commands_are_dropped():
    parsed, _ = hooks.parse({"turn_end": ["  ", "make"]})
    assert parsed["turn_end"] == ["make"]


# --- firing them -------------------------------------------------------------


def test_nothing_configured_runs_nothing(tmp_path):
    ran = Ran()
    assert build({}, tmp_path, runner=ran).fire(hooks.PRE_TOOL, tool="write_file") == []
    assert ran.calls == []


def test_each_command_runs_in_the_workspace_root(tmp_path):
    ran = Ran()
    build({"turn_end": ["make", "date"]}, tmp_path, runner=ran).fire(hooks.TURN_END)
    assert [argv[-1] for argv, _ in ran.calls] == ["make", "date"]
    assert ran.calls[0][1]["cwd"] == str(tmp_path)
    # Bounded, always. A hung formatter must not be indistinguishable from a
    # hung model.
    assert ran.calls[0][1]["timeout"] == hooks.TIMEOUT


def test_the_hook_is_told_what_it_was_fired_for(tmp_path):
    ran = Ran()
    build({"pre_tool": ["x"]}, tmp_path, runner=ran).fire(
        hooks.PRE_TOOL, tool="write_file", arguments={"path": "a b.py", "content": "1"}
    )
    environment = ran.calls[0][1]["env"]
    assert environment["CODER_EVENT"] == hooks.PRE_TOOL
    assert environment["CODER_TOOL"] == "write_file"
    assert environment["CODER_ROOT"] == str(tmp_path)
    assert '"path": "a b.py"' in environment["CODER_ARGS"]
    # Lifted out of the JSON on purpose: a path with a space in it is exactly
    # what a command line would have got wrong, and every hook anybody writes
    # wants this one value.
    assert environment["CODER_PATH"] == "a b.py"


def test_the_parent_environment_survives(tmp_path, monkeypatch):
    # A formatter that cannot see PATH is a formatter that does not run.
    monkeypatch.setenv("SOME_TOOLCHAIN", "yes")
    ran = Ran()
    build({"turn_end": ["x"]}, tmp_path, runner=ran).fire(hooks.TURN_END)
    assert ran.calls[0][1]["env"]["SOME_TOOLCHAIN"] == "yes"


def test_booleans_reach_the_shell_as_one_and_zero(tmp_path):
    ran = Ran()
    build({"post_tool": ["x"]}, tmp_path, runner=ran).fire(
        hooks.POST_TOOL, tool="edit_file", ok=False
    )
    assert ran.calls[0][1]["env"]["CODER_OK"] == "0"


def test_a_turn_end_hook_hears_how_the_turn_stopped(tmp_path):
    ran = Ran()
    build({"turn_end": ["x"]}, tmp_path, runner=ran).fire(
        hooks.TURN_END, stopped="answered", edited=True, tool_calls=3
    )
    environment = ran.calls[0][1]["env"]
    assert environment["CODER_STOPPED"] == "answered"
    assert environment["CODER_EDITED"] == "1"
    assert environment["CODER_TOOL_CALLS"] == "3"


def test_a_huge_argument_does_not_go_into_the_environment_whole(tmp_path):
    ran = Ran()
    build({"pre_tool": ["x"]}, tmp_path, runner=ran).fire(
        hooks.PRE_TOOL, tool="write_file", arguments={"content": "x" * 50_000}
    )
    assert len(ran.calls[0][1]["env"]["CODER_ARGS"]) <= hooks.ARGS_LIMIT


# --- what a result means ------------------------------------------------------


def test_a_non_zero_pre_tool_hook_blocks_the_call(tmp_path):
    ran = Ran(code=1, stderr="no writes under vendor/")
    fired = build({"pre_tool": ["gate"]}, tmp_path, runner=ran).fire(
        hooks.PRE_TOOL, tool="write_file"
    )
    assert fired[0].blocked
    # The hook's own sentence, because that is the thing the model can act on.
    assert fired[0].reason == "no writes under vendor/"


def test_a_pre_tool_hook_that_says_nothing_still_says_why(tmp_path):
    fired = build({"pre_tool": ["gate"]}, tmp_path, runner=Ran(code=3)).fire(
        hooks.PRE_TOOL, tool="write_file"
    )
    assert "exited 3" in fired[0].reason


def test_a_zero_exit_blocks_nothing(tmp_path):
    fired = build({"pre_tool": ["gate"]}, tmp_path, runner=Ran(0)).fire(hooks.PRE_TOOL)
    assert not fired[0].blocked


def test_a_failing_post_tool_hook_cannot_block_anything(tmp_path):
    # The call already happened. There is nothing left to refuse.
    fired = build({"post_tool": ["fmt"]}, tmp_path, runner=Ran(code=2)).fire(
        hooks.POST_TOOL, tool="write_file"
    )
    assert not fired[0].blocked
    assert "exited 2" in fired[0].summary()


def test_a_pre_tool_hook_that_hangs_blocks(tmp_path):
    # The one place failing open would be worse than failing loudly: a gate
    # nobody heard from is not a gate.
    ran = Ran(raises=subprocess.TimeoutExpired("gate", 30))
    fired = build({"pre_tool": ["gate"]}, tmp_path, runner=ran).fire(hooks.PRE_TOOL)
    assert fired[0].timed_out and fired[0].blocked
    assert "did not finish" in fired[0].reason


def test_a_machine_with_no_shell_blocks_nothing(tmp_path):
    # Nothing was heard from because nothing could be spawned. Blocking every
    # call there would turn a config nobody can run into an agent that cannot
    # work -- unlike a command the shell looked for and did not find, below.
    ran = Ran(raises=OSError("no such file"))
    fired = build({"pre_tool": ["nope"]}, tmp_path, runner=ran).fire(hooks.PRE_TOOL)
    assert fired[0].broken and not fired[0].blocked
    assert "could not run" in fired[0].summary()


def test_stderr_comes_first(tmp_path):
    # The opposite of a test run: a hook that refuses says why on stderr, and
    # that sentence is the whole result.
    ran = Ran(code=1, stdout="checking...", stderr="REFUSED")
    fired = build({"pre_tool": ["gate"]}, tmp_path, runner=ran).fire(hooks.PRE_TOOL)
    assert fired[0].output.index("REFUSED") < fired[0].output.index("checking")


def test_a_chatty_hook_is_truncated(tmp_path):
    ran = Ran(code=1, stderr="x" * 20_000)
    fired = build({"pre_tool": ["gate"]}, tmp_path, runner=ran).fire(hooks.PRE_TOOL)
    assert len(fired[0].output) < 20_000
    assert "truncated" in fired[0].output


def test_every_hook_runs_even_after_one_refuses(tmp_path):
    # Two gates, and the second is not skipped because the first said no: a
    # hook that only sometimes runs is a hook nobody can reason about.
    ran = Ran(code=1, stderr="no")
    fired = build({"pre_tool": ["a", "b"]}, tmp_path, runner=ran).fire(hooks.PRE_TOOL)
    assert len(fired) == 2 and len(ran.calls) == 2


# --- what doctor prints -------------------------------------------------------


def test_the_listing_is_in_event_order():
    configured = build({"turn_end": ["c"], "pre_tool": ["a", "b"]})
    assert configured.listing() == [
        (hooks.PRE_TOOL, "a"), (hooks.PRE_TOOL, "b"), (hooks.TURN_END, "c"),
    ]


@pytest.mark.parametrize(
    "commands, expected", [({}, False), ({"pre_tool": []}, False), ({"turn_end": ["x"]}, True)]
)
def test_emptiness(commands, expected):
    assert bool(build(commands)) is expected


def test_a_timeout_report_names_the_timeout_that_was_enforced(tmp_path):
    # It used to name the module default whatever the caller had passed, which
    # is a report about a number nobody used.
    ran = Ran(raises=subprocess.TimeoutExpired("gate", 1))
    fired = build({"pre_tool": ["gate"]}, tmp_path, runner=ran, timeout=1.0).fire(
        hooks.PRE_TOOL
    )
    assert "1s" in fired[0].summary() and "1s" in fired[0].reason


def test_a_command_the_shell_cannot_find_fails_closed(tmp_path):
    # Not `broken`: the shell ran and said 127. A gate whose script has been
    # deleted is a gate that should stop the call, not wave it through.
    ran = Ran(code=127, stderr="bash: gate.sh: command not found")
    fired = build({"pre_tool": ["gate.sh"]}, tmp_path, runner=ran).fire(hooks.PRE_TOOL)
    assert not fired[0].broken
    assert fired[0].blocked


# --- hooks written as files -----------------------------------------------


def write_hook(root, event, name="run.sh", executable=True):
    directory = root / "agent" / "hooks" / event
    directory.mkdir(parents=True, exist_ok=True)
    (root / "agent" / "agent.json").write_text("{}")
    path = directory / name
    path.write_text("#!/bin/sh\necho ran\n")
    if executable:
        path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def no_global_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(hooks_module.layout, "GLOBAL_ROOT", tmp_path / "nowhere")


def test_the_directory_names_the_event(tmp_path):
    write_hook(tmp_path, "post_tool", "format.sh")

    found, problems = hooks_module.discover(tmp_path)
    assert found == {"post_tool": ["agent/hooks/post_tool/format.sh"]}
    assert problems == []


def test_a_file_without_the_execute_bit_is_reported_not_skipped(tmp_path):
    # Silently ignoring it looks exactly like a hook that fired and did nothing.
    write_hook(tmp_path, "post_tool", "format.sh", executable=False)

    found, problems = hooks_module.discover(tmp_path)
    assert found == {}
    assert "not executable" in problems[0]


def test_an_unknown_event_directory_is_reported(tmp_path):
    write_hook(tmp_path, "post_turn", "build.sh")

    found, problems = hooks_module.discover(tmp_path)
    assert found == {} and "unknown hook event" in problems[0]


def test_hooks_on_one_event_keep_their_name_order(tmp_path):
    write_hook(tmp_path, "post_tool", "20-lint.sh")
    write_hook(tmp_path, "post_tool", "10-format.sh")

    found, _ = hooks_module.discover(tmp_path)
    assert [Path(command).name for command in found["post_tool"]] == [
        "10-format.sh", "20-lint.sh",
    ]


def test_a_path_with_a_space_in_it_is_quoted(tmp_path):
    write_hook(tmp_path, "post_tool", "run it.sh")

    found, _ = hooks_module.discover(tmp_path)
    assert found["post_tool"] == ["'agent/hooks/post_tool/run it.sh'"]


def test_an_unmarked_agent_directory_contributes_no_hooks(tmp_path):
    # Arbitrary commands out of somebody else's repository, fired without
    # asking, is the exact thing the marker exists to prevent.
    directory = tmp_path / "agent" / "hooks" / "post_tool"
    directory.mkdir(parents=True)
    (directory / "theirs.sh").write_text("#!/bin/sh\n")
    (directory / "theirs.sh").chmod(0o755)

    assert hooks_module.discover(tmp_path) == ({}, [])


def test_config_hooks_and_file_hooks_both_run_config_first(tmp_path):
    combined = hooks_module.combine(
        {"post_tool": ["ruff format"]}, {"post_tool": ["agent/hooks/post_tool/x.sh"]}
    )
    assert combined == {"post_tool": ["ruff format", "agent/hooks/post_tool/x.sh"]}


def test_a_file_hook_actually_fires(tmp_path):
    path = write_hook(tmp_path, "post_tool", "format.sh")
    path.write_text("#!/bin/sh\necho formatted\n")
    path.chmod(0o755)

    found, _ = hooks_module.discover(tmp_path)
    [result] = Hooks(commands=found, root=tmp_path).fire(POST_TOOL, tool="write_file")
    assert result.code == 0 and "formatted" in result.output
