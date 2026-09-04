"""Tools written as files: what loads, what is refused, and the gate."""

import pytest

from bkht.coder import usertools
from bkht.coder.tools import build_registry
from bkht.coder.tools.base import ToolError, Workspace


@pytest.fixture(autouse=True)
def no_global_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(usertools.layout, "GLOBAL_ROOT", tmp_path / "nowhere")


SIMPLE = '''
from bkht.coder.tools.base import Tool, ToolResult

TOOL = Tool(
    name="whatever",
    description="Says hello.",
    parameters={"type": "object", "properties": {}},
    run=lambda: ToolResult.success("hello"),
)
'''


def write_tool(root, name="greet", body=SIMPLE):
    directory = root / "agent" / "tools"
    directory.mkdir(parents=True, exist_ok=True)
    (root / "agent" / "agent.json").write_text("{}")
    path = directory / f"{name}.py"
    path.write_text(body)
    return path


def discover(root):
    return usertools.discover(root, workspace=Workspace(root))


# --- loading --------------------------------------------------------------


def test_the_file_is_the_tool_and_its_name(tmp_path):
    write_tool(tmp_path, "greet")

    found = discover(tmp_path)
    assert [tool.name for tool in found.tools] == ["greet"]
    assert found.tools[0].run().content == "hello"


def test_a_factory_is_given_the_workspace(tmp_path):
    write_tool(tmp_path, "where", body='''
from bkht.coder.tools.base import Tool, ToolResult

def tool(workspace):
    return Tool(
        name="where",
        description="Says where it is.",
        parameters={"type": "object", "properties": {}},
        run=lambda: ToolResult.success(str(workspace.root)),
    )
''')

    found = discover(tmp_path)
    assert found.tools[0].run().content == str(tmp_path)


def test_a_module_offering_neither_is_reported(tmp_path):
    write_tool(tmp_path, "empty", body="x = 1\n")

    found = discover(tmp_path)
    assert not found.tools and "defines neither" in found.problems[0]


def test_a_module_that_raises_on_import_costs_its_tool_not_the_session(tmp_path):
    write_tool(tmp_path, "broken", body="raise SystemExit(2)\n")
    write_tool(tmp_path, "greet")

    found = discover(tmp_path)
    assert [tool.name for tool in found.tools] == ["greet"]
    assert "could not be imported" in found.problems[0]


def test_an_underscored_file_is_a_helper_not_a_tool(tmp_path):
    write_tool(tmp_path, "_shared", body="x = 1\n")

    assert discover(tmp_path) == usertools.Found()


def test_an_unmarked_agent_directory_runs_nothing(tmp_path):
    # Cloning a repository must never be enough to import its Python.
    directory = tmp_path / "agent" / "tools"
    directory.mkdir(parents=True)
    (directory / "theirs.py").write_text("raise AssertionError('imported')\n")

    assert discover(tmp_path) == usertools.Found()


# --- the promise the loop is made -----------------------------------------


def test_an_exception_inside_a_tool_becomes_a_tool_error(tmp_path):
    write_tool(tmp_path, "boom", body='''
from bkht.coder.tools.base import Tool

def blow_up():
    raise ValueError("nope")

TOOL = Tool(
    name="boom",
    description="Fails.",
    parameters={"type": "object", "properties": {}},
    run=blow_up,
)
''')

    tool = discover(tmp_path).tools[0]
    with pytest.raises(ToolError, match="nope"):
        tool.run()


# --- registration ---------------------------------------------------------


def test_a_user_tool_may_not_take_a_built_ins_name(tmp_path):
    # It would take calls the permission layer approved under that name.
    write_tool(tmp_path, "write_file")

    found = discover(tmp_path)
    registry, _ = build_registry(tmp_path, agent_tools=found)
    assert registry.get("write_file").mutating
    assert "already a built-in" in found.problems[0]


def test_a_mutating_user_tool_is_left_out_of_a_read_only_registry(tmp_path):
    write_tool(tmp_path, "deploy", body=SIMPLE.replace(
        'run=lambda: ToolResult.success("hello"),',
        'run=lambda: ToolResult.success("hello"),\n    mutating=True,',
    ))

    registry, _ = build_registry(tmp_path, read_only=True, agent_tools=discover(tmp_path))
    assert "deploy" not in registry


def test_no_user_tool_loads_into_a_read_only_registry_whatever_it_declares(tmp_path):
    # `mutating` gates a built-in because this package wrote it and knows what
    # it does. Here it is an assertion by the code it would be gating, and a
    # file that writes to disk while claiming otherwise must not reach a --plan
    # session, whose whole promise is that it refuses every change.
    write_tool(tmp_path, "sneaky", body=SIMPLE.replace(
        'run=lambda: ToolResult.success("hello"),',
        'run=lambda: ToolResult.success("hello"),  # writes, and does not say so',
    ))

    registry, _ = build_registry(tmp_path, read_only=True, agent_tools=discover(tmp_path))
    assert "sneaky" not in registry


def test_a_delegated_sub_agent_gets_none_of_them_either(tmp_path):
    # The sub-agent behind `task` is documented as unable to write, and it is
    # given a read-only registry to make that true.
    write_tool(tmp_path, "greet")

    registry, _ = build_registry(tmp_path, read_only=True, agent_tools=discover(tmp_path))
    assert "greet" not in registry


def test_nothing_is_registered_when_nothing_was_found(tmp_path):
    before, _ = build_registry(tmp_path)
    after, _ = build_registry(tmp_path, agent_tools=discover(tmp_path))
    assert before.names() == after.names()


def test_the_summary_names_each_tool_and_its_file(tmp_path):
    write_tool(tmp_path, "greet")
    assert "greet (agent/tools/greet.py)" in usertools.summarize(discover(tmp_path))
