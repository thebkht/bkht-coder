# Changelog

Notable changes to bkht-coder, newest first.

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
versions are [semantic](https://semver.org/spec/v2.0.0.html). The git tag is the
only place a version number is written down: `vX.Y.Z` is cut from the section
below it, and the release workflow refuses a tag this file does not describe.

## [Unreleased]

### Changed

- A bare `pytest` no longer runs the live tests. The `live` marker already
  existed and was already applied; what was missing was a default, so the
  command CONTRIBUTING tells you to run before a commit collected the model-
  backed suite along with everything else. `pytest -m live` runs them, and
  pytest reports the deselected count on every run, so nothing is skipped
  quietly.

### Measured

Three full runs of the old default, on `qwen2.5-coder:14b`: **41, 46 and 49
minutes**. The same tree with the live tests deselected: **9.1 seconds**, 1372
passed. Coverage is identical -- 11 tests moved from "run by default" to "run
on request", and the slowest test remaining in the default arm takes 1.0 s.

Where the 46 minutes went, from `--durations`:

    1118s  test_review_corpus.py::test_recall_on_planted_bugs
     787s  test_review_corpus.py::test_precision_on_clean_code
     297s  test_live.py::test_reads_the_file_instead_of_asking_for_it
     222s  test_live.py::test_fixes_a_real_bug_and_the_fixed_code_runs
     171s  test_live.py::test_answers_a_question_by_reading_the_code

Five tests, 45 of the 46 minutes.

## [0.4.0] - 2026-09-03

### Added

- A `plan` tool: a short numbered list the model writes and ticks off. It is
  kept on the session rather than in the message history, so the two things
  that free context — summarising and eliding — cannot take it, and it is
  appended to every request. A turn whose history has just been compacted still
  reads its own list before it decides what to do next. Persisted, so
  `--resume` resumes the plan; dropped by `/clear`; shown in full in the
  transcript and by `/context`; and kept in plan mode's read-only tool set,
  because producing a plan is what plan mode is for.
- A `task` tool: one self-contained question handed to a second agent with its
  own session, a read-only tool set, and its own clock — three minutes and
  eight round trips. Only its prose comes back, so the files it read cost the
  parent nothing. It cannot write, run commands, delegate further, or touch the
  parent's plan. Its tool calls are shown as they happen; its prose is not, so
  it cannot stream into the middle of the parent's answer. Esc reaches it.
- `planning` and `delegation` settings, both on, with `--no-planning` and
  `--no-delegation` to match.

### Measured

The tool set is small on purpose: the registry's standing claim is that every
extra tool measurably costs selection accuracy on a small model. Two were
added, so the claim was tested rather than waved at.

Twenty-four single-tool requests against `qwen2.5-coder:14b` at temperature
0.1, scoring the first call each one makes, with nine tools and then with
eleven. **Both arms chose identically on all twenty-four** — including the six
where both disagreed with the answer key, preferring `list_files` to `glob`
and `codebase_search` to `grep`. Neither arm produced an unparseable call, and
on a single-file request the two replies were byte-identical at 22 completion
tokens. The extra tools cost nothing in selection here.

What they do cost is prompt. The system prompt grows from 9,038 to 10,425
characters, about 347 tokens, or 15% — roughly 2% of a 16k window, paid on
every request. Wall-clock was not attributable to the tools: the per-call
spread on this machine, 3.5–11.4s for the same request, is far wider than any
difference between the arms.

### Known: the opening search suppresses both tools

Delegation fires when the automatic workspace search does not run, and does not
fire when it does. Asked to summarise a module and say who calls it,
`qwen2.5-coder:14b` under `--no-scout` called `task` and got its answer from a
sub-agent that read the files; the identical request with the search block
present was answered straight out of the search snippet, with no file opened
and no tool called.

This is the scout's own long-standing hazard rather than something these tools
introduced — a 7b does the same thing without them, answering from the snippet
and reading nothing — but it is what decides how often either new tool is
reached for, so it is stated here rather than left to be discovered. `--no-scout`
is the lever today. A footer reworded to tell the model to read before it
*answers*, not only before it changes anything, was tried and did not shift the
behaviour; it was reverted rather than shipped unvalidated. Making the two
features agree is the next thing to look at.

## [0.3.0] - 2026-09-03

### Added

- A row under the prompt naming the directory, the branch, how full the context
  window is, the model and what the session has spent. It is redrawn on every
  keystroke, and the meter turns orange at the point compaction becomes the
  next thing to happen. A narrow terminal drops fields from the right rather
  than wrapping.
- The prompt block stays on screen while a turn runs, pinned under the spinner,
  instead of the session emptying out to one line until the answer arrives. It
  stays up while the answer streams too: a half-written sentence keeps its own
  line and the block sits below it, rather than being taken down for every
  fragment that does not end on a newline — which is most of them.
- A release waiting to be installed is shown on its own row above the frame.

### Changed

- Esc stops a turn during the wait before the first token, which on a local
  model is most of a turn. The read now happens on a worker thread, so the
  interrupt lands when the key is pressed rather than whenever the model next
  speaks.
- The greeting no longer names the permission mode or the context count. Both
  changed while the session ran and the greeting could not be redrawn, so it
  went on reporting the mode the session started in; they are on the row under
  the prompt now.

## [0.2.0] - 2026-09-02

### Added

- `coder update` installs the newest release, and `--check` reports one without
  installing it. An interactive session asks GitHub for a version number at most
  once a day, in the background, and says so in one line of the greeting.
  `coder doctor` reports the same thing, asked live.
- `update_check` config setting, on by default, turns the check off for good.

### Changed

- The version comes from the git tag rather than a number written in
  `pyproject.toml`. An untagged commit builds as a dev version of the release it
  leads to, so a checkout can no longer report itself as a release.
- `scripts/install.sh` and `scripts/install.ps1` install the newest release tag
  rather than the default branch. `BKHT_CODER_REF` still overrides it.

## [0.1.0]

The first version anyone installed. An interactive REPL and one-shot runs
against a local Ollama, with permission-gated file, shell, search, background
job, GitHub and GitLab tools; resumable sessions; skills and slash commands;
`AGENTS.md` and `CLAUDE.md` instructions; `coder review` with CI annotations and
GitLab Code Quality reports; `coder doctor`; `coder config`; and Claude Code and
Codex as borrowed model backends.

[Unreleased]: https://github.com/thebkht/bkht-coder/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/thebkht/bkht-coder/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/thebkht/bkht-coder/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/thebkht/bkht-coder/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/thebkht/bkht-coder/releases/tag/v0.1.0
