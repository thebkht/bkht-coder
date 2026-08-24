# Contributing

Bug reports, questions and patches are all welcome. Open an issue before a
large change so we can agree on the shape of it first; small fixes can go
straight to a pull request.

## Setup

```sh
git clone https://github.com/thebkht/bkht-coder
cd bkht-coder
uv sync --extra dev
```

That is enough to run the tests. You only need [Ollama](https://ollama.com/download)
and a pulled model if you want to run the live suite.

## Tests

```sh
uv run pytest -q                # unit + loop tests, no model needed
uv run pytest -q -m live        # end-to-end against a running Ollama
./scripts/verify.sh             # preflight checks, then both suites
```

The unit suite runs in a few seconds and needs no model — it is the one CI
runs, and the one to run before you push. `tests/fakes.py` and
`tests/conftest.py` are the seams that make that possible: `FakeProvider`
stands in for the model, so a turn can be driven end to end without Ollama.
Reach for those before adding a new kind of stub.

New behaviour should come with a test. Bug fixes should come with the test
that fails without the fix.

## Changing prompts

Prompt changes are the easy thing to get wrong, because they look fine and
regress quietly. `tests/corpus/` holds diffs with planted bugs and known-good
code, and the live suite reports recall and precision against it:

```sh
uv run pytest -q -m live
```

That number is how a prompt change gets judged rather than guessed at. If you
touch `bkht/coder/prompts.py`, say what happened to it in your pull request.

## Style

Match the code around you — naming, comment density, and the way modules are
laid out are fairly consistent already, and that consistency is worth more
than any individual preference.

Ruff's default rules must pass:

```sh
uv run ruff check .
```

There is deliberately no autoformatter. Please don't reformat code you aren't
otherwise changing; it buries the actual diff and rewrites `git blame`.

## Commits

Write the subject line as what the change does for a reader of the code, not
what you did to the file. The existing log is the guide.
