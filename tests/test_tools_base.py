"""Path confinement, schema validation, and output truncation."""

import pytest

from bkht.coder.tools.base import (
    Registry,
    Tool,
    ToolError,
    ToolResult,
    Workspace,
    truncate,
    validate_arguments,
)


def _tool(**kwargs) -> Tool:
    defaults = dict(
        name="demo",
        description="",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "count": {"type": "integer"},
                "flag": {"type": "boolean"},
            },
            "required": ["path"],
        },
        run=lambda **kw: ToolResult.success(""),
    )
    defaults.update(kwargs)
    return Tool(**defaults)


# --- path confinement -------------------------------------------------------


def test_relative_path_resolves_under_root(workspace):
    assert workspace.resolve("src/main.py") == workspace.root / "src" / "main.py"


@pytest.mark.parametrize(
    "path",
    [
        "../../etc/passwd",
        "../outside.txt",
        "src/../../escape.py",
        "/etc/passwd",
    ],
)
def test_escapes_are_rejected(workspace, path):
    with pytest.raises(ToolError, match="outside the workspace"):
        workspace.resolve(path)


def test_absolute_path_inside_root_is_allowed(workspace):
    inside = workspace.root / "a.py"
    assert workspace.resolve(str(inside)) == inside


def test_symlink_escape_is_rejected(workspace, tmp_path):
    outside = tmp_path.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    (workspace.root / "link").symlink_to(outside)
    with pytest.raises(ToolError, match="outside the workspace"):
        workspace.resolve("link/secret.txt")


def test_empty_path_is_rejected(workspace):
    with pytest.raises(ToolError):
        workspace.resolve("   ")


def test_root_itself_resolves(workspace):
    assert workspace.resolve(".") == workspace.root


# --- schema validation ------------------------------------------------------


def test_valid_arguments_pass_through():
    assert validate_arguments(_tool(), {"path": "a.py", "count": 3}) == {
        "path": "a.py",
        "count": 3,
    }


def test_missing_required_argument_names_it():
    with pytest.raises(ToolError, match="missing required argument"):
        validate_arguments(_tool(), {"count": 1})


def test_unknown_argument_names_it_and_the_expected_set():
    with pytest.raises(ToolError) as exc:
        validate_arguments(_tool(), {"path": "a.py", "nope": 1})
    assert "nope" in str(exc.value) and "path" in str(exc.value)


def test_wrong_type_is_rejected():
    with pytest.raises(ToolError, match="must be a integer"):
        validate_arguments(_tool(), {"path": "a.py", "count": []})


def test_stringified_number_is_coerced():
    # Small models routinely stringify numbers; accept that narrowly.
    assert validate_arguments(_tool(), {"path": "a.py", "count": "7"})["count"] == 7


def test_unparseable_stringified_number_is_rejected():
    with pytest.raises(ToolError):
        validate_arguments(_tool(), {"path": "a.py", "count": "many"})


def test_bool_is_not_an_integer():
    with pytest.raises(ToolError):
        validate_arguments(_tool(), {"path": "a.py", "count": True})


def test_integer_is_not_a_bool():
    with pytest.raises(ToolError):
        validate_arguments(_tool(), {"path": "a.py", "flag": 1})


def test_non_object_arguments_are_rejected():
    with pytest.raises(ToolError, match="must be a JSON object"):
        validate_arguments(_tool(), ["a.py"])


# --- truncation -------------------------------------------------------------


def test_short_output_is_untouched():
    assert truncate("a\nb\nc") == "a\nb\nc"


def test_line_truncation_is_announced():
    out = truncate("\n".join(str(i) for i in range(50)), max_lines=10)
    assert out.splitlines()[-1] == "[truncated 40 lines]"
    assert len(out.splitlines()) == 11


def test_character_truncation_is_announced():
    out = truncate("x" * 40_000)
    assert "[truncated 10000 characters]" in out


# --- results and registry ---------------------------------------------------


def test_failure_is_rendered_as_actionable_text():
    assert ToolResult.failure("no match").as_message() == "ERROR: no match"


def test_registry_declares_tools_sorted():
    registry = Registry()
    registry.add(_tool(name="zeta"))
    registry.add(_tool(name="alpha"))
    assert registry.names() == ["alpha", "zeta"]
    assert [d["function"]["name"] for d in registry.declarations()] == ["alpha", "zeta"]
    assert "alpha" in registry and len(registry) == 2
