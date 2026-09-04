# Changelog

Notable changes to bkht-coder, newest first.

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
versions are [semantic](https://semver.org/spec/v2.0.0.html). The git tag is the
only place a version number is written down: `vX.Y.Z` is cut from the section
below it, and the release workflow refuses a tag this file does not describe.

## [Unreleased]

### Fixed

- **Installing behind a TLS-inspecting proxy failed with no explanation.** uv
  verifies against roots it bundles itself, so on a machine whose HTTPS is
  re-signed by a corporate proxy or an antivirus -- where git, curl and the
  browser all work -- `uv tool install` dies fetching a Python interpreter with
  `invalid peer certificate: UnknownIssuer`, and both installers reported only
  "uv tool install failed". They now retry against the platform's certificate
  store and, if that fails too, name the two other ways out
  (`--no-python-downloads`, `SSL_CERT_FILE`). `coder update` does the same.

  The setting is passed as an environment variable rather than a flag, in both
  spellings (`UV_NATIVE_TLS` and `UV_SYSTEM_CERTS`), because the flag was
  renamed between uv versions: an environment variable uv does not know is
  ignored, where a flag it does not know is a hard error that would replace the
  real failure with a worse one.

## [0.7.1] - 2026-09-04

### Added

- **Hooks can be files.** `agent/hooks/<event>/format.sh` -- the directory names
  the event, the file is the command, and a diff can review it, where the same
  hook written into `config.json` is a JSON string with escaped quotes in it.
  Both sources fire, config first, appended rather than shadowed: a setting is
  one value and the specific one wins, but a hook is a thing that happens, and a
  project asking for a formatter has not asked for your own hook to stop. The
  execute bit is required and a file without it is reported, because everything
  in that directory is a candidate to run and a hook that silently never fires
  is worse than one that was never written.

- **Subagents: a delegated task handed to somebody in particular.** `task`
  already ran a second agent, but every delegation got the same empty prompt and
  the same skills -- right for "find where the registry is built", wrong for
  "review this diff the way we review diffs here". A subagent is a directory
  under `agent/subagents/` with a description to choose it by, its own
  instructions, and its own skills. Its own, not the workspace's: a reviewer
  that inherited every skill in the project would be the parent agent under
  another name.

  With none written the `task` tool has the schema it always had, because a
  parameter offering a choice of nothing can only be got wrong. There is no
  per-subagent model: on a machine serving one, naming a second means evicting
  the first to load it and evicting it back, which costs more than the
  delegation saves.

- **Tools a workspace writes for itself, behind two switches.**
  `agent/tools/<name>.py`, one tool per file, named for the file, exposing
  `TOOL` or `tool(workspace)`. The tool set is short on purpose -- every extra
  tool costs selection accuracy on a small model -- but that argument is about
  *this* project's tools, not about the one integration a workspace lives inside
  and otherwise reaches through a shell. A user tool's `run` is wrapped, because
  every other tool here promises the loop it raises `ToolError` and nothing
  else, and a traceback out of one would end the turn rather than the call.

  This imports the workspace's Python into the agent's own process before the
  first turn, which is a larger hazard than hooks. Cloning a repository must not
  be enough to run it, so three things have to be true: `agent/` is marked with
  an `agent.json`, `agent_tools` is on (it ships **off**), and
  `--no-agent-tools` was not passed. Then `doctor` and `/tools` name every tool
  and its file. A user tool cannot take a built-in's name -- one answering to
  `write_file` would take calls the permission layer had already approved under
  that name, which is not a tool but a way around the gate.

  None of them load in plan mode or into the sub-agent behind `task`, whatever
  they declare. `mutating` gates a built-in because this package wrote it and
  knows what it does; on a user tool it is an assertion by the code it would be
  gating, and a boundary the rest of the registry keeps structurally cannot be
  left to the good faith of the thing on the other side of it.

- **`coder info` says what loaded, not only what is on disk**, and `/agent`
  prints the same view mid-session. The tree alone cannot show a skill refused
  for want of a description; the summary alone cannot show which file that skill
  is sitting in.

## [0.7.0] - 2026-09-04

### Added

- **`agent/`: one directory for everything this project authors.** Instructions,
  skills, commands and hooks each landed somewhere different, by a rule of their
  own. `agent/` is the floor of one rule instead -- the slot a file lands in
  decides how it loads -- with `instructions`, `skills`, `commands`, `hooks`,
  `subagents` and `tools` as the slots.

  Layered, not moved. `AGENTS.md`, `.claude/skills` and `.bkht-coder/commands`
  all still load, and `agent/` goes last, where the most specific source
  belongs. A workspace root needs an `agent.json` marker and the global root
  does not: `agent/` is exactly what an eve project calls its own agent, and
  adopting one unasked would make its system prompt ours.

  Skills gain the flat form, where the file is the skill and its path is its
  name. A flat skill ships no resources -- its neighbours are other skills, not
  its files -- so `skill(resource=...)` refuses one by name rather than reaching
  sideways into somebody else's directory.

  `coder info` prints the surface before anything reads from it, and `doctor`
  reports it, for the same reason hooks are listed: a source that shapes every
  answer silently is worse than no source at all.

- **Hooks: your own commands, fired on tool events.** `permissions.json`
  remembers what you allowed but cannot *do* anything. A `hooks` block in
  `config.json` can -- run the formatter after a write, refuse a call whose
  shape this project never wants, kick off a build when the turn ends.

  Three events. `pre_tool` fires before a call runs, after validation and
  permission -- after, because firing a hook for a call the user is about to
  refuse would run it for a call that never happens. `post_tool` fires once the
  call has run, including when it failed: "the write did not happen" is exactly
  what a hook watching writes needs to hear, and hearing nothing is
  indistinguishable from not being configured. `turn_end` fires once, however
  the turn stopped.

  Only `pre_tool` can say no. A non-zero exit blocks the call and what the hook
  printed becomes the tool result, so the model reads the refusal and works
  around it -- the same correction path a malformed call already takes. A
  `post_tool` exit code is reported and ignored; the call already happened. A
  hook that went wrong without blocking is said out loud, because a formatter
  that silently rewrote the file the model just wrote makes the next tool
  result inexplicable; a hook that blocked is not, because its sentence is
  already the tool result.

  Everything a hook might want is in the environment rather than on the command
  line, because a command line means quoting and quoting means a hook that
  silently does the wrong thing on a path with a space in it. `CODER_PATH` is
  lifted out of the arguments JSON on its own, because every hook anybody
  actually writes wants exactly that value and making each of them parse JSON in
  a shell would make hooks a thing only people who like `jq` can use.

  Bounded, and asymmetrically. Every hook is timed out at 30 seconds -- a
  formatter that hangs must not be indistinguishable from a model that hangs. A
  `pre_tool` hook that times out **blocks**: it is the one place failing open is
  worse than failing loudly, because a gate nobody heard from is not a gate. A
  gate whose script has gone missing blocks too -- the shell answered `127`, and
  a gate that waves calls through because it can no longer find itself is not
  one. The only case that blocks nothing is a machine with no shell at all,
  where nothing was heard from because nothing could be spawned.

  The sub-agent behind `task` gets the parent's hooks. A `pre_tool` gate is not
  a gate if delegating the read is the way around it.

  Hooks are arbitrary commands out of a config file that fire without anyone
  asking, so they are never invisible: `coder doctor` names every one it finds,
  `/context` counts them, `SECURITY.md` says what they are, and `--no-hooks`
  runs a session with none of them.

### Fixed

- **A fresh session's context bar read a fifth full before anything was sent.**
  `usage_ratio` falls back to estimating from the payload when the backend has
  not reported a token count, and the payload includes the system prompt -- so
  with no messages at all it answered with the size of a prompt the model had
  not been given, and `/clear` could not put the bar back to zero. On this
  repository that was 19% before a word had been typed. It now reads 0% until
  there is something in the history, which is the question the row was always
  asking.

## [0.6.0] - 2026-09-04

### Added

- **The project's own test command, run over what a turn wrote.** `verify.py`
  has always argued that running the model's new code to find out whether the
  model's new code is safe to run is not a check -- it is the thing the check
  exists to avoid. That holds for the module just written. It does not hold for
  a command the user chose, wrote down, and already runs by hand: the agent
  running it is not the agent deciding what to run.

  It runs on the answered path, the one moment a turn is known to have stopped
  writing. After every write it would put the test runner inside the edit loop
  and spend the iteration budget on it; here a turn that edited costs one run
  and a turn that only read costs none, which is most of them. A failure goes
  back as a tool result the model corrects from -- the same path a malformed
  call takes -- bounded at two runs, where the first is the check and the second
  is the fix being checked. The second failure asks for an account rather than a
  third attempt, because a turn that ends naming the failing test is worth more
  than one that ends having tried again and run out mid-edit.

  Four outcomes, not two. A timeout and a command that could not start are
  distinct from a failure: neither says anything the model could act on, so
  neither is fed back. Esc cannot reach a blocked `waitpid`, which is why the
  timeout is 120 seconds rather than generous -- what bounds the wait here is
  the timeout, not the key.

  Nothing runs until `verify_command` is set, and nothing infers its way into
  it. `doctor` suggests one from a marker file and prints the command that
  would turn it on; a bare `tests/` directory suggests nothing, because it says
  a project has tests and not what runs them. `--verify-command` sets it for one
  run, `--no-verify` turns it off for a session without costing what was
  configured, and `/context` says which it is.

  Two things came out of running it rather than reasoning about it. A bounded
  turn never reached the check at all -- it ends through `_final_answer`, and a
  turn that ran out mid-edit is precisely the one most likely to have left the
  tests broken; it now runs there too, reported and never fed back. And the
  "did this turn write anything" flag meant *at some point* rather than *since
  the last check*, so a model that read a failure, correctly judged it
  unrelated, and answered without editing paid for a second identical run.

- `verify_command` and `verify` settings. `verify_command` is the one text
  setting permitted to be empty: for every other one an empty value is a
  session that cannot start, and refusing it puts the error on the keystroke,
  but empty is exactly how this one says "run nothing".

- `scripts/benchmark.py` -- what a turn costs, per task: seconds, iterations,
  tool calls and how it stopped, with `--out` to save a run and `--compare` to
  put two of them side by side. All of it was already recorded; `Outcome`
  carries every field and every turn writes one to the session file, and
  nothing read them back. Until something did, a change to what the model is
  sent could only be argued about.

  The backend is whatever a session would use unless one is named, so it
  measures what the user runs, fallback included. The tasks are read-only,
  which is the one place this differs from `training/generate.py`: a task that
  edits the tree leaves a different repository behind than it found, so the
  second run of a comparison is not measuring the same work as the first, and
  the change under test gets credit or blame that belongs to the edit.

### Changed

- `grep` handed a glob where a path goes now says which argument the pattern
  belongs in, and writes out the call that would have worked. `path not found:
  .github/workflows/*.yml` is true and useless -- the string was never meant to
  be a path, so the model saw nothing to change and resent the same call until
  the loop's bounds ended the turn. Found in a real session that spent its
  whole iteration budget that way and answered from vendored workflows it had
  stumbled into. A path that is simply missing says only that, as before.
- A pasted image is encoded once instead of once per round trip. The provider
  base64s every image on the message it sends, and a turn sends its whole
  history on each of up to twenty-five iterations, so one screenshot was read
  off disk and encoded twenty-five times to produce twenty-five identical
  strings. Keyed on the file's identity rather than its name, so an image
  re-saved at the same path is encoded again.

### Measured

The first thing the benchmark was pointed at was the scout, because its own
first run kept landing on it: two read-heavy tasks both came back
`1 iteration, 0 tool calls, answered`, the model having answered out of the
injected search snippet without opening a file. 0.4.0 recorded that shape as a
hazard and left it there. With `--no-scout` to compare against, on
`qwen2.5-coder:14b`:

    task                                with scout        without
    what does the agent loop do when
      a tool call is malformed?      112.6s  1 iter    167.6s  1 iter
    which module decides whether a
      tool call needs permission?     43.3s  1 iter    309.6s  3 iters, 2 calls

    total                            155.9s            477.2s   (+206%)

So the snippet is not only a shortcut past reading -- it is the difference
between one round trip and three, and on the second task it saved four and a
half minutes. That is the opposite of what "hazard" suggests, and it is worth
having the number before anyone reaches for the lever.

Two limits on the reading, stated because the figure invites more weight than
it can carry. It is two tasks on one machine, and this one is spilling 1 GB of
KV cache to CPU, so the per-turn cost is high and its variance with it. And the
benchmark measures what a turn *costs*, never whether it was right -- the first
task answered a question about the agent loop without opening a file in either
arm, which is fast in both and only useful if the answer is true. Correctness
is a different instrument, and this is not it.

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

[Unreleased]: https://github.com/thebkht/bkht-coder/compare/v0.7.1...HEAD
[0.7.1]: https://github.com/thebkht/bkht-coder/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/thebkht/bkht-coder/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/thebkht/bkht-coder/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/thebkht/bkht-coder/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/thebkht/bkht-coder/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/thebkht/bkht-coder/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/thebkht/bkht-coder/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/thebkht/bkht-coder/releases/tag/v0.1.0
