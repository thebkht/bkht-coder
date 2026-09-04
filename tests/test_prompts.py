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


# --- batching ---------------------------------------------------------------

def test_the_default_protocol_is_one_call_at_a_time():
    """Serial is the default, and the training exporter depends on it.

    `TOOL_PROTOCOL` is what `training.ingest.call_json` claims to match, so it
    has to keep meaning the format a default local session actually sends.
    """
    from bkht.coder.prompts import TOOL_PROTOCOL, tool_protocol

    assert TOOL_PROTOCOL == tool_protocol()
    assert "One tool call per reply" in TOOL_PROTOCOL


def test_a_roomy_window_is_allowed_to_batch_independent_calls():
    from bkht.coder.prompts import tool_protocol

    protocol = tool_protocol(parallel=True)
    assert "One tool call per reply" not in protocol
    assert "Several calls in one reply is allowed" in protocol


def test_batching_still_forbids_a_call_that_needs_another_call_s_result():
    """The one ordering rule that batching cannot relax.

    Reading three files at once is free; editing a file in the same reply that
    reads it is an edit against text the model has not seen.
    """
    from bkht.coder.prompts import tool_protocol

    protocol = tool_protocol(parallel=True)
    assert "goes in your next reply" in protocol
    assert "cannot edit a file in the same reply you read it in" in protocol


def test_the_session_prompt_carries_whichever_protocol_it_was_built_with(
    registry_and_root,
):
    registry, root = registry_and_root
    assert "One tool call per reply" in system_prompt(registry, root)
    assert "One tool call per reply" not in system_prompt(registry, root, parallel=True)


def test_the_language_reminder_ends_on_the_work_not_the_prose():
    """It is the last thing a planless turn reads, so its last line matters.

    Four sentences about writing sentences, read immediately before
    generation, is an instruction a small model will follow: asked in Uzbek to
    study this repo, `qwen2.5-coder:7b` answered in fluent Uzbek having called
    nothing at all. The language rule goes first; the closing line says an
    unfinished task is answered with a tool call.
    """
    reminder = language_reminder("Uzbek")
    assert reminder.rstrip().endswith("When it is finished, answer.)")
    assert reminder.index("Uzbek") < reminder.index("not finished")


def test_the_reminder_does_not_contradict_the_final_answer_request():
    """It is appended to every request, `_final_answer`'s included.

    That turn has just been told to stop calling tools and answer from what it
    has. A reminder saying an unfinished task is answered with a call and
    nothing else is read after it, so it wins -- and the turn is recorded with
    no answer at all. The rule has to name when to stop as well as when to
    call.
    """
    reminder = language_reminder("Uzbek")
    assert "not prose" not in reminder
    assert "When it is finished, answer." in reminder


def test_the_reminder_still_marks_itself_as_an_aside_and_names_the_language():
    reminder = language_reminder("Uzbek")
    assert "not a new request" in reminder
    assert reminder.count("Uzbek") == 2
