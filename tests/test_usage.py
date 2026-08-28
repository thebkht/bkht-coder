"""The written help pages, and the commands that reach them."""

from __future__ import annotations

import pytest

from bkht.coder import cli, usage

PAGES = [usage.HELP, usage.SESSIONS_HELP, usage.SESSION_HELP, usage.REVIEW_HELP,
         usage.DOCTOR_HELP, usage.CONFIG_HELP]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_is_written_rather_than_generated(page):
    # argparse's own headings are the tell that a page fell back to the
    # generated rendering, which is the thing this module exists to replace.
    assert "positional arguments:" not in page
    assert "optional arguments:" not in page
    assert "options:" not in page
    assert "Usage:" in page and "Examples:" in page


@pytest.mark.parametrize("page", PAGES)
def test_no_page_wraps_an_eighty_column_terminal(page):
    assert [line for line in page.splitlines() if len(line) > 80] == []


def test_the_top_level_page_names_every_command():
    for command in ("sessions", "session", "config", "review", "doctor", "help"):
        assert command in usage.HELP
    assert usage.HELP.startswith(usage.title())
    assert usage.TAGLINE in usage.HELP


def test_each_subcommand_page_is_the_one_its_parser_serves():
    parser = cli.build_parser()
    assert parser.format_help() == usage.HELP

    pages = {
        "review": usage.REVIEW_HELP,
        "doctor": usage.DOCTOR_HELP,
        "sessions": usage.SESSIONS_HELP,
        "session": usage.SESSION_HELP,
        "config": usage.CONFIG_HELP,
    }
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    for name, page in pages.items():
        assert subparsers.choices[name].format_help() == page


def test_a_parser_without_a_page_still_generates_one():
    assert usage.Parser(prog="x").format_help().startswith("usage: x")


def test_the_agent_parser_serves_the_same_page_as_the_subcommand_one():
    assert cli.build_agent_parser().format_help() == usage.HELP


# --- dispatch ---------------------------------------------------------------


def test_help_as_a_word_prints_the_page(capsys):
    assert cli.main(["help"]) == 0
    assert capsys.readouterr().out.strip() == usage.HELP.strip()


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_as_a_flag_prints_the_same_page(flag, capsys):
    assert cli.main([flag]) == 0
    assert capsys.readouterr().out.strip() == usage.HELP.strip()


def test_version_prints_the_version_and_the_copy_that_ran(capsys):
    with pytest.raises(SystemExit) as exit:
        cli.main(["--version"])
    assert exit.value.code == 0
    assert capsys.readouterr().out.startswith("coder ")


def test_a_short_v_is_the_version_not_the_verbose_flag(capsys):
    with pytest.raises(SystemExit):
        cli.main(["-v"])
    assert capsys.readouterr().out.startswith("coder ")


# --- session resume ---------------------------------------------------------


@pytest.mark.parametrize(
    "rest, expected",
    [
        ([], ["--resume", "last"]),
        (["last"], ["--resume", "last"]),
        (["20260822-101455"], ["--resume", "20260822-101455"]),
        (["--plan"], ["--resume", "last", "--plan"]),
        (["20260822-101455", "--auto"], ["--resume", "20260822-101455", "--auto"]),
    ],
)
def test_resume_becomes_the_flag_it_has_always_been(rest, expected):
    assert cli.resume_argv(rest) == expected
