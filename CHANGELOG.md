# Changelog

Notable changes to bkht-coder, newest first.

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
versions are [semantic](https://semver.org/spec/v2.0.0.html). The git tag is the
only place a version number is written down: `vX.Y.Z` is cut from the section
below it, and the release workflow refuses a tag this file does not describe.

## [Unreleased]

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

[Unreleased]: https://github.com/thebkht/bkht-coder/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/thebkht/bkht-coder/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/thebkht/bkht-coder/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/thebkht/bkht-coder/releases/tag/v0.1.0
