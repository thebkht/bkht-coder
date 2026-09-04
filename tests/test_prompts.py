"""The system prompt: what it must say, and what it must not."""

from pathlib import Path

import pytest

from bkht.coder.prompts import SYSTEM, language_reminder, system_prompt
from bkht.coder.tools import build_registry


@pytest.fixture
def registry_and_root(project: Path):
    registry, workspace = build_registry(project, read_only=True)
    return registry, str(workspace.root)


# --- language ---------------------------------------------------------------

def test_the_system_prompt_names_no_language(registry_and_root):
    """No language example survives in the always-on prompt.

    It used to spell out the Uzbek/Russian distinction and quote `salom`, which
    put a vivid greeting in the most-attended region of every request. A model
    that has lost the thread answers with the most salient thing near it, and
    one did: an English task came back as `Salom! Sizga qanday yordam bera
    olishim mumkin?`. Which language to answer in is decided in
    ``language.detect`` and delivered only when there is something to say.
    """
    registry, root = registry_and_root
    prompt = system_prompt(registry, root)
    for word in ("Uzbek", "Russian", "salom"):
        assert word not in prompt


def test_the_system_prompt_still_says_to_match_the_user():
    assert "language the user wrote to you in" in SYSTEM


def test_the_reminder_marks_itself_as_an_aside():
    reminder = language_reminder("Uzbek")
    assert "not a new request" in reminder
    assert "Uzbek" in reminder


# --- how the answer is written ----------------------------------------------

def test_the_prompt_says_how_to_answer_not_only_how_to_work():
    """The prompt used to describe the work and say nothing about the reply.

    A procedural prompt with no output contract leaves the register to the
    model, and the register is most of what a user means when they say an agent
    feels wrong: a preamble announcing the plan, a summary repeating the tool
    calls they just watched, a whole file pasted back to make one point.
    """
    assert "# Answering" in SYSTEM
    assert "No preamble" in SYSTEM
    assert "path/to/file.py:42" in SYSTEM


def test_the_prompt_rules_out_the_unasked_for_artifacts():
    """Comments, docs, and commits nobody asked for."""
    assert "Do not add\ncomments explaining what your change does" in SYSTEM
    assert "documentation nobody asked for" in SYSTEM
    assert "do not commit or push unless you\nwere asked to" in SYSTEM


def test_the_answering_section_sits_above_the_tool_protocol(registry_and_root):
    """Ordering is load-bearing: the protocol has to be read last.

    Drifting off the emission format is this model's characteristic failure, so
    anything added to the prompt goes above the tool section, never below it.
    """
    registry, root = registry_and_root
    prompt = system_prompt(registry, root)
    assert prompt.index("# Answering") < prompt.index("# Calling a tool")
