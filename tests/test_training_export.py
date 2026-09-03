"""Selection, splitting, and the report a decision to train rests on."""

import json


from bkht.coder.training import export
from bkht.coder.training.render import Example


def example(*contents, source="claude", origin="t"):
    messages = [{"role": "user", "content": c} for c in contents]
    return Example(messages=messages, source=source, origin=origin, system_hash="h")


def pairs(*examples):
    return [(e, []) for e in examples]


# --- what gets kept -----------------------------------------------------------


def test_the_same_conversation_twice_is_kept_once():
    # The same task through two agents is one lesson; training on it twice only
    # teaches that this particular file matters more than the others.
    kept, report = export.select(pairs(example("a", "b"), example("a", "b")))
    assert len(kept) == 1 and report.duplicates == 1


def test_provenance_does_not_make_two_examples_different():
    kept, _ = export.select(
        pairs(example("a", "b", source="claude"), example("a", "b", source="codex"))
    )
    assert len(kept) == 1


def test_an_example_past_the_budget_is_dropped_not_truncated():
    # A trajectory cut in half mid-way through a tool result teaches the model
    # to answer from evidence it never saw.
    kept, report = export.select(pairs(example("x" * 100_000)), max_tokens=100)
    assert kept == [] and report.too_long == 1


def test_an_example_that_failed_its_round_trip_is_counted_and_dropped():
    kept, report = export.select([(None, ["message 2: parsed 0 calls"])])
    assert kept == [] and report.unparseable == 1


def test_the_source_mix_is_reported():
    _, report = export.select(
        pairs(example("a"), example("b"), example("c", source="codex"))
    )
    assert dict(report.by_source) == {"claude": 2, "codex": 1}


# --- the histogram ------------------------------------------------------------


def test_the_histogram_says_where_the_mass_sits():
    # The one number that decides whether max_seq_length is set right: a corpus
    # whose mass sits above the window is one that will be truncated away.
    report = export.Report(tokens=[100, 200, 3000, 9000])
    assert dict(report.histogram()) == {"<512": 2, "<4096": 1, ">=8192": 1}


def test_a_report_with_nothing_in_it_renders_without_failing():
    assert "0 examples" in export.Report().render()


def test_the_report_names_what_it_dropped():
    report = export.Report(kept=2, duplicates=1, too_long=3, tokens=[10, 20])
    text = report.render()
    assert "dropped 1 duplicate" in text and "dropped 3 over the token budget" in text


# --- splitting ----------------------------------------------------------------


def test_every_example_lands_in_exactly_one_split():
    kept, _ = export.select(pairs(*(example(f"m{i}") for i in range(40))))
    splits = export.split(kept)
    assert sum(len(v) for v in splits.values()) == 40
    seen = [tuple(m["content"] for m in e.messages) for v in splits.values() for e in v]
    assert len(set(seen)) == 40


def test_the_split_is_the_same_every_time():
    # A model evaluated against a test set it was trained on in a previous build
    # is a model with a made-up score.
    kept, _ = export.select(pairs(*(example(f"m{i}") for i in range(40))))
    first = [e.messages[0]["content"] for e in export.split(kept)["test"]]
    second = [e.messages[0]["content"] for e in export.split(kept)["test"]]
    assert first == second


def test_the_holdouts_are_never_empty_when_there_is_anything_to_put_in_them():
    # An empty valid set makes mlx_lm.lora fail at the end of the first epoch,
    # which is a long way to travel for a mistake made here.
    kept, _ = export.select(pairs(*(example(f"m{i}") for i in range(10))))
    splits = export.split(kept)
    assert splits["valid"] and splits["test"] and splits["train"]


def test_a_corpus_too_small_to_split_still_trains():
    kept, _ = export.select(pairs(example("only")))
    splits = export.split(kept)
    assert len(splits["train"]) == 1


# --- the files ----------------------------------------------------------------


def test_the_written_lines_carry_the_conversation_and_nothing_else(tmp_path):
    # Provenance in the line would be read by the trainer as part of the
    # example, and a model that has learned to predict "source": "codex" has
    # spent capacity on the wrong thing.
    kept, _ = export.select(pairs(example("hello")))
    export.write({"train": kept, "valid": [], "test": []}, tmp_path)
    line = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert line == {"messages": [{"role": "user", "content": "hello"}]}


def test_all_three_files_are_written_even_when_a_split_is_empty(tmp_path):
    export.write({"train": [], "valid": [], "test": []}, tmp_path)
    assert all((tmp_path / name).exists() for name in export.FILES.values())


def test_the_manifest_records_what_produced_the_data(tmp_path):
    # Two builds look identical, and the difference between them is the whole
    # reason one model is better than the other.
    report = export.Report(kept=5, tokens=[100] * 5, splits={"train": 5})
    report.by_source.update({"codex": 5})
    path = export.manifest(report, tmp_path, ["codex"], 4096)
    stored = json.loads(path.read_text())
    assert stored["sources"] == ["codex"] and stored["max_tokens"] == 4096
    assert stored["examples"] == 5 and stored["by_source"] == {"codex": 5}


def test_non_ascii_survives_the_round_trip(tmp_path):
    kept, _ = export.select(pairs(example("o'zgartir — привет")))
    export.write({"train": kept, "valid": [], "test": []}, tmp_path)
    written = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert written["messages"][0]["content"] == "o'zgartir — привет"
