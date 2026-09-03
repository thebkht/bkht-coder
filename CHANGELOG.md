# Changelog

Notable changes to bkht-coder, newest first.

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
versions are [semantic](https://semver.org/spec/v2.0.0.html). The git tag is the
only place a version number is written down: `vX.Y.Z` is cut from the section
below it, and the release workflow refuses a tag this file does not describe.

## [Unreleased]

### Changed

- A pasted image is encoded once instead of once per round trip. The provider
  base64s every image on the message it sends, and a turn sends its whole
  history on each of up to twenty-five iterations, so one screenshot was read
  off disk and encoded twenty-five times to produce twenty-five identical
  strings. Keyed on the file's identity rather than its name, so an image
  re-saved at the same path is encoded again.

### Measured

Two other candidates were built, measured and dropped, which is the more useful
half of the result:

- **Running a round's tool calls concurrently.** The protocol is one call per
  reply and says so in the prompt; across 558 sessions on this machine, 187 of
  189 replies carrying a call carried exactly one. Concurrency in the loop
  would apply to 1% of rounds, so it was not added.
- **Caching the workspace read that scouting does before every turn.** Built
  and measured at **8%** of scout on this repository -- 94 ms to 87 ms -- because
  reading is not where scout's time goes; the per-line regex scan is, and that
  depends on the terms, which change every turn. Seven milliseconds against an
  eleven-second turn does not pay for a cache that can serve a stale file, so
  it was reverted.

The image change was kept on the opposite reasoning: 4.2 ms to 0.012 ms on the
repeated call, and unlike the scout cache its saving grows with the file rather
than staying a fixed fraction of it.

## [0.5.0] - 2026-09-03

### Added

- **An OpenAI-compatible backend, and it is now the default.** `mlx_lm.server`,
  llama.cpp's `llama-server`, vLLM and Ollama's own `/v1` all take
  `POST /v1/chat/completions`, so one backend covers every way a model gets
  served on hardware somebody owns -- including the fine-tune this project is
  heading for, which Ollama cannot host without being taught about it first.
  The promise was never Ollama; it is that the weights stay on a machine the
  user controls, and `local` points at localhost, so it holds unchanged.
  Pointing it at another machine on the network is the arrangement the
  fine-tune needs: the Mac with the memory serves, everything else drives.

  Three places a naive port breaks, each with a test: the stream is SSE and
  ends with a literal `[DONE]` that is not JSON; tool-call arguments arrive as
  a string split across deltas keyed by index, so appending rather than
  assigning is load-bearing; and usage appears at all only with
  `stream_options.include_usage`.

- **`coder dataset`** -- a training corpus built out of the agent sessions
  already on the machine. coder's own, Claude Code's and Codex's, read into one
  shape and written as `mlx_lm.lora` takes it.

  Translation is the substance rather than the plumbing. A model trained on
  `Read(file_path=)` learns a call coder has no tool for; mapped to
  `read_file(path=)` it learns the one that exists. So the live registry
  decides what an argument is, not a table; parallel calls are serialised into
  coder's one-call-per-reply protocol; narration before a call is dropped,
  because under a protocol that says a call reply is the JSON and nothing else
  it is a worked example of the forbidden thing in the position a model
  imitates most readily; and harness furniture goes with it. Length is handled
  the way the loop handles it -- results capped at coder's own output budget,
  and an over-long example elided by `context.elide_tool_results` itself, which
  took the corpus from 33 examples to 148 without truncating one. Every
  rendered call is parsed back through `parsing.parse_tool_calls`, and one that
  does not survive is dropped: a fine-tune whose calls coder cannot read is
  worse than none.

- **A training pipeline under `training/`** -- a LoRA on a 4-bit 14b, fused
  into one directory and served on `0.0.0.0`. Every number in `lora.yaml` is a
  memory decision before it is a quality one, and the README says which to turn
  first when a run does not fit rather than leaving it to be discovered three
  hours in. `max_seq_length` and `dataset build --max-tokens` have to agree,
  and the comment in both places says so. `mlx-lm` is an optional extra: the
  machine that trains a model is not the machine that runs one.

- **The prompt and the outcome are written to the transcript.** A transcript is
  only a training example if the input half survives, and the input half is the
  system prompt -- assembled per session from the registry, the tree, the
  instructions and the skills, and reconstructable from none of them later. A
  resumed session appends a second `prompt` record rather than replacing the
  first, so a file spanning two tool sets says so in order. Beside it, an
  `outcome` record per turn, because the message list cannot tell a turn that
  answered from one that hit the iteration cap, and anything choosing
  trajectories to imitate needs exactly that difference. Recorded, never
  replayed: resume still rebuilds against the tools it has now.

- **A fallback to Ollama when nothing is serving the default.** A first session
  that fails to connect to a server nobody was told to start teaches nothing
  except that this does not work. Only the built-in default falls back --
  `--provider local` against a dead server stays an error, and so does one
  pinned in a config file, because quietly answering with a different model
  than the one somebody typed is the worse failure. Announced on every path,
  never silent.

### Changed

- A bare `pytest` no longer runs the live tests. The `live` marker already
  existed and was already applied; what was missing was a default, so the
  command CONTRIBUTING tells you to run before a commit collected the model-
  backed suite along with everything else. `pytest -m live` runs them, and
  pytest reports the deselected count on every run, so nothing is skipped
  quietly.

- `doctor` chooses its checks by what a backend *is* rather than by whether it
  is the one listed first. The old `provider != DEFAULT_PROVIDER` test read
  correctly only for as long as the default happened to be Ollama; with the
  default moved it inverted exactly, checking Ollama for a command on PATH and
  asking the new default for `/api/tags`. `local` gets three checks where
  Ollama gets four -- the missing one is placement, and it stays missing,
  because the OpenAI API has no notion of where a model is resident and this
  file already spent a release learning not to invent that number.
- `for_review` asks whether a backend can turn sampling off instead of what
  type it is. Determinism is a property a backend either offers or does not,
  and the `isinstance` check silently stopped applying the moment the default
  moved.
- `coder config`'s declared defaults for model and host now come from the same
  table `_follow_provider` reads, so it cannot show a model the session is not
  running.

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

[Unreleased]: https://github.com/thebkht/bkht-coder/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/thebkht/bkht-coder/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/thebkht/bkht-coder/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/thebkht/bkht-coder/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/thebkht/bkht-coder/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/thebkht/bkht-coder/releases/tag/v0.1.0
