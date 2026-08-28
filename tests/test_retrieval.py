"""The scout: term extraction, ranking, and the bounds on what it emits."""

import pytest

from bkht.coder import retrieval
from bkht.coder.retrieval import BUDGET_CHARS, scout, search, terms
from bkht.coder.tools import build_registry
from bkht.coder.tools.base import ToolError


# --- terms -------------------------------------------------------------------


def test_a_conversational_message_yields_nothing_to_search_for():
    assert terms("thanks!") == []
    assert terms("yes go ahead") == []
    assert terms("ok, do it") == []


def test_identifiers_and_paths_beat_plain_words():
    found = terms("fix the off-by-one in tools/search.py iter_files")
    assert found[0] in {"tools/search.py", "iter_files"}
    assert "iter_files" in found
    assert "fix" not in found  # a stopword: it matches everything


def test_asking_to_be_told_about_something_searches_for_nothing():
    """"explain me this project" has no term in it worth a search."""
    assert terms("explain me this project") == []
    assert terms("give me an overview") == []
    # The verb goes, what it was asked about stays.
    assert terms("explain the auth flow") == ["auth", "flow"]


def test_compound_names_are_searched_by_their_parts_too():
    found = terms("the handle_undo helper")
    assert "handle_undo" in found and "handle" in found and "undo" in found

    found = terms("readFileSync is throwing")
    assert "readFileSync" in found and "read" in found and "Sync" in found
    # "file" is a stopword even when it arrives as part of a camelCase name.
    assert "File" not in found


def test_quoted_text_is_kept_whole():
    assert "no such file" in terms('why does it print "no such file"?')


def test_the_term_list_is_bounded():
    from bkht.coder.retrieval import MAX_TERMS

    message = " ".join(f"symbol_{i}" for i in range(40))
    assert len(terms(message)) <= MAX_TERMS


# --- ranking -----------------------------------------------------------------


def test_the_defining_file_ranks_first(project):
    hits = search(project, terms("what does helper do?"))
    assert hits[0].path == "src/util.py"
    assert any("def helper" in line for _, line in hits[0].best_lines())


def test_breadth_of_match_beats_repetition(project):
    (project / "noisy.py").write_text("total = 1\n" * 50)
    (project / "both.py").write_text("def total():\n    return helper()\n")

    ranked = [hit.path for hit in search(project, ["total", "helper"])]
    assert ranked.index("both.py") < ranked.index("noisy.py")


def test_ignored_directories_are_never_searched(project):
    (project / "node_modules" / "junk.js").write_text("function helper() {}\n")
    (project / ".git" / "helper").write_text("helper\n")

    paths = [hit.path for hit in search(project, terms("what does helper do?"))]
    assert not any(path.startswith(("node_modules", ".git")) for path in paths)


def test_nothing_matching_produces_no_hits(project):
    assert search(project, ["quorum_reconciler"]) == []


# --- rendering ---------------------------------------------------------------


def test_the_block_names_its_terms_and_cites_file_and_line(project):
    block = scout(project, "what does helper do?")
    assert "Terms:" in block and "helper" in block
    assert "src/util.py:1:" in block
    # It has to say it is a keyword match, or the model treats it as the answer.
    assert "not a complete answer" in block


def test_the_block_stays_within_its_budget(project):
    for i in range(40):
        (project / f"file_{i}.py").write_text("def helper():\n    return helper\n" * 30)

    block = scout(project, "what does helper do?")
    assert 0 < len(block) < BUDGET_CHARS + 500  # header and footer sit outside


def test_a_message_with_no_terms_is_not_searched(project):
    assert scout(project, "thanks!") == ""


def test_a_file_named_after_the_term_is_shown_even_if_it_never_says_it(project):
    # ".gitignore" is the answer to "what does .gitignore exclude?" and the word
    # appears nowhere inside it.
    (project / ".gitignore").write_text("__pycache__/\n*.pyc\n.venv/\n")

    hits = search(project, terms("what does .gitignore exclude?"))
    assert hits and hits[0].path == ".gitignore"
    assert hits[0].best_lines()[0] == (1, "__pycache__/")


# --- the tool the scout advertises ------------------------------------------


def search_tool(project):
    registry, _ = build_registry(project, read_only=True)
    return registry.get("codebase_search")


def test_codebase_search_is_a_real_tool(project):
    """The scout opens every turn with a `codebase_search` result.

    A model that has just read the output of a tool will call that tool. It used
    to be told there is no such tool, which cost a retry and taught it that its
    formatting was wrong when the formatting was fine.
    """
    assert search_tool(project) is not None


def test_codebase_search_finds_what_the_scout_would(project):
    result = search_tool(project).run(terms="helper")
    assert result.ok and "src/util.py" in result.content


def test_codebase_search_splits_terms_on_commas_and_spaces(project):
    result = search_tool(project).run(terms="helper, main")
    assert result.ok and "helper" in result.content


def test_a_called_search_does_not_claim_it_was_automatic(project):
    result = search_tool(project).run(terms="helper")
    assert "You did not call this" not in result.content


def test_codebase_search_says_so_when_nothing_matches(project):
    result = search_tool(project).run(terms="zzzznotpresentzzzz")
    assert result.ok and "No files matched" in result.content


def test_codebase_search_rejects_empty_terms(project):
    with pytest.raises(ToolError):
        search_tool(project).run(terms="   ")


# --- requests that are not about this workspace -------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "review github run 33185669396",
        "why did run 4210 fail",
        "look at issue #42",
        "check https://example.com/thing",
        "deploy 12345 broke the site",
    ],
)
def test_a_request_naming_something_external_is_not_scouted(message):
    assert retrieval.about_something_else(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "add a --verbose flag",
        "why does run_tests fail?",
        "look at the workflow file",
        "rename handle_undo to undo",
        "fix the off-by-one in truncate",
    ],
)
def test_an_ordinary_task_still_gets_its_search(message):
    assert retrieval.about_something_else(message) is False


def test_the_scout_says_nothing_about_a_ci_run(tmp_path):
    # Every repository has a `.github`, so the keyword search matched one and
    # pointed the turn at the workflow file -- which says what the job would do
    # and nothing at all about what it did.
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "review.yml").write_text("name: review\n")
    assert retrieval.scout(tmp_path, "review github run 33185669396") == ""
