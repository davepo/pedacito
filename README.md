# Pedacito

*Installs the `pedacito` command, plus `pdc` as a shorter alias — the two are interchangeable everywhere in this document.*

**Pedacito lets a small local LLM work on a codebase bigger than its context
window.** Models in the 9B–27B range typically can't hold a real project in
context. Instead of pasting your whole repo at the model, Pedacito builds a
lookup index and lets the model request only the specific files, symbols, or
snippets it needs for the task at hand — piece by piece (*pedacito* is Spanish
for "little piece") — until it has enough to answer or edit. It's aimed at the
hardware most people running local models actually have: a single consumer GPU
with 16–24GB of VRAM, not a cluster.

- Works with any server exposing an OpenAI-compatible `/v1` API — tested
  against **LM Studio and Ollama** — on Windows, Linux, and macOS. Hosted APIs
  (DeepSeek, OpenAI, and similar) work too.
- AST-based chunking for Python, tree-sitter for JavaScript, TypeScript, TSX,
  and Ruby; structural indexing for everything else useful as context.
- Edits come back as search/replace blocks rather than whole files or diffs.
- It can also start a project from scratch (`pedacito new`) and run an
  autonomous task loop (`pedacito auto`).

### What this is (and isn't)

Built by one person, for one person's use case: running local models over
Tailscale against a single home GPU box, mostly for narrow, localized edits
rather than sweeping refactors. It was designed and coded almost entirely
through conversations with AI pair programmers — DeepSeek, Qwen, Claude, and
local models — iterating against real bugs on real hardware. That background is
worth knowing: it's had one user and one set of models exercising it so far, so
edge cases outside that path are more likely to be unpolished.

If you have the hardware or budget for a frontier hosted model, you'll almost
certainly get better results from a tool built around those models (see
"Alternatives" below). Pedacito exists for the case where you specifically want
to stay local and small.

### Alternatives

If you're not committed to a small local model, these are generally more
capable and polished:

- **[opencode](https://opencode.ai)** — an open-source, terminal-based coding
  agent that works with a wide range of providers, hosted and local.
- **[DeepSeek Harness](https://github.com/deepseek-ai/harness)** — DeepSeek's
  tooling for running and working with their models.
- **[Qwen Code](https://github.com/QwenLM/qwen-code)** — an agentic coding CLI
  built around Qwen's models; works well with the same local Qwen weights
  Pedacito can run against.
- **[Claude Code](https://claude.com/claude-code)** — Anthropic's agentic
  coding tool, the flagship option if you can use Claude models.

Pedacito is worth reaching for when none of those fit: no internet access to a
hosted API, a preference for keeping code entirely on your own hardware, or
just wanting to see what a 9B–27B model can do on a real project when it's not
fighting its context window.

---

## Install

Requires Python 3.11+.

**Windows (PowerShell):**

```powershell
cd C:\path\to\pedacito
py -m pip install -e .
```

**Linux / macOS:**

```bash
cd ~/pedacito
pip install -e .          # or: pipx install -e .
```

The editable install puts `pedacito` on your `PATH`, along with `pdc` (a shorter
alias for the same command), and pulls in `requests` and `tree-sitter` (used for
JS/TS/Ruby symbol indexing). If the command isn't found on `PATH` afterwards
(common with per-user installs on Windows), `python -m pedacito` / `py -m
pedacito` works identically.

Uninstall with `pip uninstall pedacito-local` (that's the distribution name in
`pyproject.toml`). It removes the commands but leaves your config and every
project's `.pedacito/` untouched.

---

## Model server setup

Point Pedacito at any server that speaks the OpenAI chat-completions API.

- **LM Studio** — load a model and start the server (Developer tab). Default
  port `1234`. If Pedacito runs on another machine, enable "Serve on Local
  Network".
- **Ollama** — `ollama serve`, then `ollama pull <model>`. Default port `11434`.
  For a remote setup, set `OLLAMA_HOST=0.0.0.0` before starting it.
- **Hosted OpenAI-compatible APIs** — DeepSeek, OpenAI, Groq, Mistral and
  similar. Set `base_url`, `api_key`, and `model`; Pedacito detects a remote
  host and switches to `json_mode = "object"` automatically.

Verify the connection from wherever Pedacito runs:

```bash
pedacito config --check
```

This distinguishes "server unreachable" from "model not found or won't load"
(usually a name mismatch, or not enough free VRAM) — different problems with
different fixes.

---

## Configure

```bash
pedacito config --init
```

writes a starter config to `~/.config/pedacito/config.toml` (Linux/macOS) or
`%APPDATA%\pedacito\config.toml` (Windows) — `pedacito config` always prints the
exact path it uses. `pedacito config --setup` builds it interactively, finding
LM Studio or Ollama for you. The three settings that matter:

```toml
base_url = "http://localhost:1234/v1"   # or :11434/v1 for Ollama
model    = "your-model-id"              # exact id/name the server reports
vram_gb  = 8                            # GPU VRAM on the model server, in GB
```

`vram_gb` is mandatory for local servers (Pedacito uses it, plus the model's
file size, to size the context window); hosted configs don't need it. Settings
resolve in layers, each overriding the one before: built-in defaults → your
user config → the project's `.pedacito.toml` → `--config FILE` → a selected
`--profile` → command-line flags. `[server]` and `[lmstudio]` are both accepted
as section names for the connection settings.

Run `pedacito config` with no arguments to see every resolved setting and which
layer set it; typos are rejected with a suggestion. State for each project lives
in `.pedacito/` at that project's root — see "Undo and workspace layout" below.

---

## Built-in help: `pedacito how`

`pedacito how` is built-in help that answers in your own words, using this
README, the [Quick Start](QUICKSTART.md), and Pedacito's source code as its
references. It needs no project and builds no index — just a model connection:

```bash
pedacito how "how do I set up a config file?"
pedacito how "what command indexes a project?"
pedacito how "how do I undo an apply?"
pedacito how                          # interactive help prompt
```

Answers are written for a first-time user and include exact commands (or a
minimal TOML snippet) whenever the question is about doing something. It uses
your normal config automatically — secrets such as `api_key` are never sent to
the model — so you can ask "what should I run with my setup?" and get an answer
that reflects your real server, model, and limits.

It also accepts per-session overrides (`--url`/`--server`, `--model`,
`--api-key`, `--vram`/`--vram-gb`, `--config`, `--profile`). Unlike the coding
commands, `how` does not require `vram_gb`.

---

## The basic workflow

Three commands cover most day-to-day work. The project path is the first
argument; state lands in `<project>/.pedacito/`.

```bash
pedacito index   ~/code/anki                                    # build the index (once; cached)
pedacito ask     ~/code/anki -t "make the leech threshold configurable"
pedacito explain ~/code/anki -q "how does the scheduler pick the next card?"
```

### `index` — build the lookup index

Parses the project into a map of files, symbols, and line ranges, then
optionally adds one-line LLM summaries. `--no-summaries` builds the static index
only (instant, no LLM calls); `--resummarise` discards cached summaries and
regenerates them with the current model. Summaries are rationed (units with
docstrings and methods covered by their class are skipped) and cached by content
hash, so re-indexing only costs what actually changed. `ask` and `chat` build or
refresh the index automatically, so you only run `index` explicitly the first
time.

Three inspection commands help before and after indexing, none of which need a
working model:

- `pedacito files ~/code/anki` — every candidate file, and why it was kept or
  dropped. Free.
- `pedacito cost ~/code/anki` — what a first index will cost (LLM calls, time).
  Free.
- `pedacito map ~/code/anki` — prints the current index, no model.

### `ask` — run one task

`ask` is the core loop: it gathers the source the task needs (the model requests
files, symbols, and snippets through the index), then produces edits as
search/replace blocks. By default it's a dry run — you see a diff and nothing is
written:

```bash
pedacito ask ~/code/anki -t "..." --apply     # write immediately
pedacito ask ~/code/anki -t "..." --review    # side-by-side sign-off page
```

Every write snapshots originals first; `pedacito restore <project>` undoes the
last one.

### `explain` — answer questions about the code

`explain` answers questions about the indexed project — how a function works,
what a file is responsible for, how data or control flows between parts — in
prose, with no edits:

```bash
pedacito explain ~/code/anki -q "trace the flow from answer to schedule"
pedacito explain ~/code/anki        # interactive prompt
```

It's always scoped to the project's index and never builds one; if there's no
index yet, it tells you to run `pedacito index` (or navigate to a project you've
already indexed).

---

## More ways to work

### `chat` — interactive session

```bash
pedacito chat ~/code/anki
```

Keeps the index warm between tasks instead of reloading it each time. Type a
task, or `/help` for the slash commands: `/profile`, `/model`, `/apply`,
`/steps`, `/status`, `/profiles`, `/quit`.

### `new` — start a project from scratch

`ask` and `chat` edit an existing codebase; `new` builds one. Give it a one-line
description (`-t`) or a spec document (`--spec`, also via stdin), and it plans
the file tree, scaffolds it, then fills in each file:

```bash
pedacito new my-app -t "a CLI todo manager with a sqlite backend" --apply
pedacito new my-api --spec docs/spec.md --review      # sign off in the browser
pedacito new my-app -t "..." --plan-only --apply      # stop after the skeleton
```

Dry run by default. `--plan-only` stops after the skeleton so you can adjust the
plan yourself; `--complex` hands implementation to the autonomous loop.

### `auto` — autonomous loop over a task list

`auto` breaks a goal into an ordered checklist and implements each task in a
fresh context window, with optional verification and repair:

```bash
pedacito auto . -t "add retry logic to the sync client"     # plan only, nothing written
pedacito auto . -t "..." --apply                            # plan, then implement every task
pedacito auto . --apply --verify "pytest -q"                # verify each task, repair on failure
pedacito auto . --apply --plan TODO.md                      # use an existing checklist instead
```

State lives in a plain markdown plan file (`PLAN.md` by default, or `--plan
FILE`), so a run is resumable and editable mid-flight. `--verify CMD` runs after
each task and feeds failures back for repair (`--repair` rounds); a missing
Python dependency is reported, or installed with `--auto-install` (off by
default). `--max-tasks N` caps the run. Without `--apply`, `auto` only generates
and prints the plan.

### `gui` — browser control panel

```bash
pedacito gui                # opens in a browser at an OS-chosen port
pedacito gui --no-open      # print the URL instead of opening a browser
```

A local control panel instead of the terminal: open a project, build the index,
check the model, run tasks with the review page inline, and ask `explain`
questions. One process, one page, bound to `127.0.0.1` like every other server
Pedacito starts.

---

## What gets indexed

**Python** (via the standard-library `ast`) and **JavaScript, TypeScript, TSX,
and Ruby** (via `tree-sitter`, installed by default) are parsed into functions,
classes, and methods with exact line ranges. Everything else useful as context
— `.md .txt .rst .csv .tsv .json .toml .yaml .yml .ini .cfg .conf .env .sql
.sh` — is indexed *structurally*: the map shows a CSV's columns, a JSON file's
keys, a TOML file's sections, a markdown file's headings. Contents stay out of
the map, but `read` and `grep` reach them normally.

Anything else is invisible until you add its extension explicitly:

```bash
pedacito ask . -t "..." --ext .go
```

which routes it through the same structural chunker — `read`/`grep` work,
there's just no symbol map for it.

**Excluding files** works in three layers, increasing precedence:

1. `.pedacitoignore` in the project root (gitignore syntax) — the project's
   `.gitignore` is read too.
2. Command-line flags: `--exclude 'deprecated/'`, `--include
   'deprecated/still_used.py'`, `--no-refs` (code only), `--no-gitignore`.
3. Naming a path directly always wins — typing it is an explicit request.

`pedacito files` shows every candidate and the reason it was kept or dropped.

## Index freshness, tiers, and summaries

The index stores a content hash per file and self-corrects on every command:
edits Pedacito makes are re-chunked immediately, and edits you make in your
editor are detected and refreshed automatically on the next run (line ranges are
only meaningful against the bytes they were computed from). Run a full
`pedacito index` after a large refactor or a `git pull`.

The map is tiered so it survives real projects: a flat listing of every function
fits small projects; larger ones get one *card* per file (imports, a summary,
and top-level definitions with line ranges), and the model descends with
`outline` and `read` on demand. `pedacito map <project> --tier auto` shows which
side of the line a project landed on.

## Editing: search/replace, not diffs or whole files

Edits are search/replace blocks anchored on text the model actually read:

```
--- FILE: src/app.py
<<<<<<< SEARCH
    def is_leech(self, lapses):
        return lapses >= 8
=======
    def is_leech(self, lapses):
        return lapses >= self.leech_threshold
>>>>>>> REPLACE
```

A block either matches the file exactly or it fails cleanly — no partial or
silent corruption. Every edit is validated before anything is written
(`ast.parse()` for Python, `tree-sitter` for JS/TS/Ruby), and a failed block's
error is fed back to the model (up to two rounds) so it can re-anchor with more
context.

## Review page

`--review` (or `review = true` in config) opens a side-by-side sign-off page
instead of writing directly: current file on the left, proposed version on the
right, changed words highlighted, long unchanged stretches folded. It's
self-contained HTML — no CDN, no external fonts — so it opens and works offline.
Applying takes two deliberate clicks, and Enter never triggers a write.

The property that matters more than the buttons: the page pins the SHA-256 of
every file as it stood when the preview was built, and applying re-hashes and
refuses on any mismatch. Editing a file in your own editor while the review sits
open causes the write to be refused rather than silently overwriting your
change. The review server binds to `127.0.0.1` only.

## Undo and workspace layout

Every write (from `--apply` or the review page) snapshots originals into
`.pedacito/backups/<stamp>/` first:

```bash
pedacito restore <project> --list
pedacito restore <project> --dry-run
pedacito restore <project>
pedacito restore <project> --snapshot 20260830-020724
```

Restoration is byte-for-byte. Everything Pedacito writes for a project lives
under that project's root:

```
<project>/.pedacito/
    index.json           the map
    summaries.json       summary cache, keyed by content hash
    backups/<stamp>/     pre-write snapshots, mirroring the tree
    sessions/<stamp>.md  what was asked, looked up, and changed
    reviews/<stamp>.html saved copies of review pages
    log.txt              a flat, timestamped event trace
```

`sessions/` is the file worth opening when an edit turns out wrong days later;
`log.txt` is the quick scan for *where* a failure happened. On first run Pedacito
also creates a commented `.pedacitoignore` and adds its workspace entries to
`.gitignore` (both switchable via `create_ignore_file` and `manage_gitignore`).

---

## Profiles, tiers, and token budgets

Define a profile per model instead of editing `model =` each time:

```toml
[profiles.small]
model = "qwen2.5-coder:7b"

[profiles.big]
model = "some-27b-model"
max_steps = 8
```

```bash
pedacito ask anki -t "..." --profile big
```

Select a profile with `default_profile =` in config, the `PEDACITO_PROFILE`
environment variable, `--profile` on a command, or `/profile` inside `chat`.
A profile can override any setting, not just the model.

Add `tier = "small" | "medium" | "large" | "hosted"` and Pedacito fills in all
the token-budget knobs (`max_tokens_summary/step/answer/plan/build`,
`flat_map_token_budget`, `max_gathered_chars`, `max_read_lines`,
`max_spec_chars`, `max_new_files`) from a preset — tight for a 9B on an 8k
local window, generous for DeepSeek-class APIs. You rarely need to tune any of
this manually: Pedacito auto-detects the server's real context window and the
model's file size (using `vram_gb`) and sizes every budget to fit (`auto_fit`,
on by default). Explicit values you write always win. `pedacito config` shows
which tier is active and marks every non-default value.

Reasoning models (Qwen3 and similar) are supported — `<think>` blocks are
stripped before parsing, so they never corrupt a tool call, an edit, or a
session log. `disable_thinking` and `thinking_suffix` turn thinking off. The
client also handles Mistral-family templates (strict role alternation) and
Gemma (no system role) transparently.

---

## Configuration reference

`pedacito config` lists every setting with its current value and origin;
`pedacito config --reference` prints the full annotated template. The important
ones:

| Setting | Default | |
|---|---|---|
| `base_url` | `http://localhost:1234/v1` | Server address. |
| `model` | `google/gemma-3-27b` | Exact id/name the server reports. |
| `vram_gb` | *(mandatory)* | GPU VRAM on the machine running the model (local only). |
| `json_mode` | `"schema"` | `"schema"` for local llama.cpp/LM Studio, `"object"` for hosted APIs (auto-switched). |
| `tier` | `""` | `small`/`medium`/`large`/`hosted` preset for token budgets. |
| `max_steps` | 30 | Lookups before the model must answer. |
| `max_write_lines` | 120 | Lines in one write step; larger files are written in chunks. |
| `temperature` | 0.2 | Higher makes tool selection noticeably worse. |
| `backup` | True | Snapshot originals before writing. |
| `log_sessions` | True | Write a session record per task. |
| `review` | False | Always open the side-by-side reviewer. |
| `allow_write` | False | Let the gather loop overwrite whole files (prefer search/replace). |
| `disable_thinking` | False | Suppress `<think>` blocks. |
| `default_profile` | `""` | Profile applied when `--profile` isn't given. |

The remaining knobs (chunking, per-budget limits, review timeouts, auto-loop
behavior, and more) all have working defaults and are documented in
`pedacito config --reference`.

---

## Known limits

- Symbol-level parsing covers Python, JavaScript/TypeScript/TSX, and Ruby.
  Other extensions are structural, or invisible until added via
  `--ext`/`extra_extensions`.
- The call graph matches on bare names, so two unrelated `run()` methods are
  conflated — fine for locating code, not sound for automated refactoring.
- No embeddings — `grep` plus the summary map covers most of what semantic
  search would, without a second model resident in memory.
- Broad, structural tasks ("modernize this codebase") are a poor fit regardless
  of scaffolding — that's a limit of small local models. Narrow, localized
  tasks are where this is genuinely useful.
- The browser GUI covers ask/explain/review/build-index/model-check;
  `how`, `new`, `auto`, `restore`, and the advanced flags are CLI-only.

---

## Contributing

Issues and pull requests are welcome — bug reports with a repro, support for
other model servers or languages, and fixes for anything that breaks outside the
setup this was built and tested on. Given the small scope, please open an issue
before a large PR so the direction can be agreed on first.

## License

MIT — see [LICENSE](LICENSE).
