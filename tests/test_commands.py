"""Slash commands the user writes as files."""

from bkht.coder import commands as commands_module
from bkht.coder.commands import discover, summarize


def write_command(root, name, text="Review the tests in this project.", where=".bkht-coder/commands"):
    directory = root / where
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    path.write_text(text)
    return path


def test_a_workspace_command_is_found(project):
    write_command(project, "review-tests")
    assert "review-tests" in discover(project)


def test_the_body_becomes_the_task(project):
    write_command(project, "audit", text="Audit the error handling.")
    assert discover(project)["audit"].expand("") == "Audit the error handling."


def test_arguments_are_substituted_where_the_file_asks(project):
    write_command(project, "explain", text="Explain $ARGUMENTS in plain terms.")
    assert discover(project)["explain"].expand("the retry loop") == (
        "Explain the retry loop in plain terms."
    )


def test_arguments_are_appended_when_there_is_no_placeholder(project):
    # Otherwise a file written without a placeholder would silently discard
    # whatever the user typed after the command.
    write_command(project, "audit", text="Audit the error handling.")
    assert discover(project)["audit"].expand("in provider.py").endswith("in provider.py")


def test_no_arguments_leaves_no_trailing_blank(project):
    write_command(project, "audit", text="Audit the error handling.")
    assert discover(project)["audit"].expand("  ") == "Audit the error handling."


def test_frontmatter_is_stripped_and_its_description_used(project):
    write_command(
        project, "release",
        text="---\ndescription: Cut a release.\n---\n\nBump the version, then tag.\n",
    )
    command = discover(project)["release"]
    assert command.body == "Bump the version, then tag."
    assert command.description == "Cut a release."


def test_without_frontmatter_the_first_line_describes_it(project):
    write_command(project, "audit", text="Audit the error handling.\nThen report.")
    assert discover(project)["audit"].description == "Audit the error handling."


def test_an_empty_file_is_not_a_command(project):
    write_command(project, "blank", text="   \n\n")
    assert discover(project) == {}


def test_a_global_command_applies_everywhere(project, tmp_path, monkeypatch):
    monkeypatch.setattr(commands_module, "GLOBAL_ROOT", tmp_path / "global")
    (tmp_path / "global").mkdir()
    (tmp_path / "global" / "everywhere.md").write_text("Do the usual.")
    assert "everywhere" in discover(project)


def test_a_workspace_command_shadows_a_global_one(project, tmp_path, monkeypatch):
    monkeypatch.setattr(commands_module, "GLOBAL_ROOT", tmp_path / "global")
    (tmp_path / "global").mkdir()
    (tmp_path / "global" / "audit.md").write_text("global version")
    write_command(project, "audit", text="workspace version")
    assert discover(project)["audit"].body == "workspace version"


def test_only_markdown_files_are_commands(project):
    directory = project / ".bkht-coder" / "commands"
    directory.mkdir(parents=True)
    (directory / "notes.txt").write_text("not a command")
    assert discover(project) == {}


def test_summarize_is_empty_without_commands(project):
    assert summarize(discover(project)) == ""


def test_summarize_names_each_command(project):
    write_command(project, "audit", text="Audit the error handling.")
    text = summarize(discover(project))
    assert "/audit" in text and "Audit the error handling." in text

