"""Remembered permission decisions: what matches, and what survives a restart."""

import json

import pytest

from bkht.coder.rules import ALLOW, DENY, Rules, rule_id, signature


@pytest.fixture
def store(tmp_path):
    return Rules.load("/work/one", path=tmp_path / "permissions.json")


def test_a_shell_call_is_identified_by_its_command():
    assert signature("bash", {"command": "ls", "timeout": 30}) == "ls"


def test_a_file_call_is_identified_by_its_path():
    # Not by the content: the same file written twice is the same decision, and
    # the diff was already shown once.
    assert signature("write_file", {"path": "a.py", "content": "x"}) == "a.py"


def test_argument_order_does_not_change_the_signature():
    # A small model emits the same call with its keys shuffled all the time.
    one = signature("odd_tool", {"a": 1, "b": 2})
    two = signature("odd_tool", {"b": 2, "a": 1})
    assert one == two


def test_ids_are_stable_across_runs():
    first = rule_id("/work", "bash", "ls")
    assert first == rule_id("/work", "bash", "ls")
    assert first != rule_id("/other", "bash", "ls")


def test_remembering_then_deciding(store):
    store.remember("bash", {"command": "ls"}, ALLOW)
    assert store.decide("bash", {"command": "ls"}) == ALLOW
    assert store.decide("bash", {"command": "rm -rf /"}) is None


def test_a_rule_does_not_cross_tools(store):
    store.remember("write_file", {"path": "a.py", "content": "1"}, ALLOW)
    assert store.decide("edit_file", {"path": "a.py"}) is None


def test_rules_persist_and_reload(tmp_path):
    path = tmp_path / "permissions.json"
    Rules.load("/work/one", path=path).remember("bash", {"command": "ls"}, ALLOW)
    assert Rules.load("/work/one", path=path).decide("bash", {"command": "ls"}) == ALLOW


def test_rules_do_not_leak_between_workspaces(tmp_path):
    # One store per file, many workspaces in it. A grant earned in one project
    # firing in another is the failure this scoping exists to prevent.
    path = tmp_path / "permissions.json"
    Rules.load("/work/one", path=path).remember("bash", {"command": "ls"}, ALLOW)
    assert Rules.load("/work/two", path=path).decide("bash", {"command": "ls"}) is None


def test_saving_preserves_another_workspaces_rules(tmp_path):
    # Another session, in another directory, may have written since we loaded.
    # Dropping its rule because we were not watching is a silent revocation.
    path = tmp_path / "permissions.json"
    Rules.load("/work/one", path=path).remember("bash", {"command": "one"}, ALLOW)
    Rules.load("/work/two", path=path).remember("bash", {"command": "two"}, DENY)

    assert Rules.load("/work/one", path=path).decide("bash", {"command": "one"}) == ALLOW
    assert Rules.load("/work/two", path=path).decide("bash", {"command": "two"}) == DENY


def test_revoking_removes_the_rule(tmp_path):
    path = tmp_path / "permissions.json"
    store = Rules.load("/work/one", path=path)
    rule = store.remember("bash", {"command": "ls"}, ALLOW)

    assert store.revoke(rule.id) == rule
    assert store.revoke(rule.id) is None
    assert Rules.load("/work/one", path=path).decide("bash", {"command": "ls"}) is None


def test_a_missing_file_is_not_an_error(tmp_path):
    store = Rules.load("/work/one", path=tmp_path / "nothing.json")
    assert store.listing() == [] and store.error == ""


def test_a_corrupt_file_degrades_to_no_rules(tmp_path):
    # Refusing to start would take away the only tool that could fix the file.
    path = tmp_path / "permissions.json"
    path.write_text("{ not json")

    store = Rules.load("/work/one", path=path)
    assert store.listing() == []
    assert "could not read" in store.error


def test_unparseable_records_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "permissions.json"
    path.write_text(
        json.dumps(
            [
                {"scope": "/work/one", "tool": "bash", "signature": "ls", "decision": "allow"},
                {"scope": "/work/one", "tool": "bash"},
                {"scope": "/work/one", "tool": "bash", "signature": "x", "decision": "maybe"},
                "junk",
            ]
        )
    )

    store = Rules.load("/work/one", path=path)
    assert [r.signature for r in store.listing()] == ["ls"]


def test_an_unknown_decision_is_rejected(store):
    with pytest.raises(ValueError, match="unknown decision"):
        store.remember("bash", {"command": "ls"}, "maybe")


def test_a_long_signature_is_shortened_for_display(store):
    rule = store.remember("bash", {"command": "echo " + "x" * 200}, ALLOW)
    assert len(rule.label()) < 120 and rule.id in rule.label()
