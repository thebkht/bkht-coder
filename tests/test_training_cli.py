"""`coder dataset` -- the three verbs, end to end."""

import json

import pytest

from bkht.coder.cli import main


@pytest.fixture
def corpus(tmp_path):
    """A chat JSONL holding one usable trajectory."""
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "what does a.py do"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "Read", "arguments": '{"file_path": "a.py"}'}}
                        ],
                    },
                    {"role": "tool", "name": "Read", "content": "x = 1"},
                    {"role": "assistant", "content": "It sets x to one."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def build(tmp_path, corpus, *extra):
    return main(
        ["dataset", "build", "--source", "chat", "--file", str(corpus),
         "--out", str(tmp_path / "data"), "--cwd", str(tmp_path), *extra]
    )


def test_a_build_writes_the_three_files_and_a_manifest(tmp_path, corpus, capsys):
    assert build(tmp_path, corpus) == 0
    out = tmp_path / "data"
    assert {p.name for p in out.iterdir()} == {
        "train.jsonl", "valid.jsonl", "test.jsonl", "manifest.json",
    }
    assert "1 examples" in capsys.readouterr().out


def test_the_written_example_speaks_coders_protocol(tmp_path, corpus, capsys):
    build(tmp_path, corpus)
    line = json.loads((tmp_path / "data" / "train.jsonl").read_text().splitlines()[0])
    roles = [m["role"] for m in line["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    # Translated on the way in: the model must learn the tool coder has.
    assert json.loads(line["messages"][2]["content"])["name"] == "read_file"
    assert line["messages"][0]["content"].startswith("You are `coder`")


def test_an_unknown_source_is_refused_by_name(tmp_path, corpus, capsys):
    assert main(["dataset", "build", "--source", "nonesuch", "--out", str(tmp_path / "d")]) == 2
    assert "nonesuch" in capsys.readouterr().err


def test_a_build_that_found_nothing_says_so_rather_than_writing_empty_files(tmp_path, capsys):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert build(tmp_path, empty) == 1
    assert "No examples" in capsys.readouterr().err


def test_stats_reads_the_data_that_would_actually_be_trained_on(tmp_path, corpus, capsys):
    build(tmp_path, corpus)
    capsys.readouterr()
    assert main(["dataset", "stats", "--out", str(tmp_path / "data")]) == 0
    assert "1 examples" in capsys.readouterr().out


def test_stats_on_a_directory_with_no_dataset_says_what_to_run(tmp_path, capsys):
    assert main(["dataset", "stats", "--out", str(tmp_path / "nothing")]) == 2
    assert "coder dataset build" in capsys.readouterr().err


def test_show_prints_one_example_as_the_model_reads_it(tmp_path, corpus, capsys):
    build(tmp_path, corpus)
    capsys.readouterr()
    assert main(["dataset", "show", "0", "--out", str(tmp_path / "data")]) == 0
    printed = capsys.readouterr().out
    assert "--- system ---" in printed and "--- tool ---" in printed


def test_show_past_the_end_of_the_file_is_an_error_not_a_traceback(tmp_path, corpus, capsys):
    build(tmp_path, corpus)
    capsys.readouterr()
    assert main(["dataset", "show", "99", "--out", str(tmp_path / "data")]) == 2


def test_a_flag_before_the_positional_still_parses(tmp_path, corpus, capsys):
    # argparse before 3.12.7 cannot fill an optional positional that appears
    # after a flag, which is why the argv is reordered first.
    build(tmp_path, corpus)
    capsys.readouterr()
    assert main(["dataset", "show", "--out", str(tmp_path / "data"), "0"]) == 0


def test_json_output_is_the_manifest(tmp_path, corpus, capsys):
    build(tmp_path, corpus, "--json")
    stored = json.loads(capsys.readouterr().out)
    assert stored["sources"] == ["chat"] and stored["examples"] == 1
