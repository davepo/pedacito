# Pedacito

**Pedacito lets a small local LLM work on a codebase bigger than its context
window.** Models from 4B to 27B can't hold a real project in context.
Instead of pasting your repo at the model, Pedacito builds a lookup index and
lets the model request only the files, symbols, and snippets it needs
(*pedacito* is Spanish for "little piece") until it can answer or edit. It
targets the hardware most people actually have: one consumer GPU with 16–24 GB
of VRAM.

- Works with any server that exposes an OpenAI-compatible `/v1` API — tested
  with **LM Studio, Ollama, and Lemonade** — on Windows, Linux, and macOS.
  Hosted APIs (DeepSeek, OpenAI, and similar) work too.
- Python is parsed with `ast`; JavaScript, TypeScript, TSX, Go, Rust, Java, C,
  C++, C#, and Ruby with tree-sitter. Other useful text files are indexed
  structurally.
- Edits are search/replace blocks, validated before anything is written.
- Beyond `ask`, it can start a project from scratch (`new`) and run an
  autonomous task loop (`auto`).

The install provides two interchangeable commands: `pedacito` and `pdc`.

### What this is (and isn't)

Pedacito was built by one person for one setup: local models over Tailscale on a
single home GPU box, mostly for narrow, localized edits. It has had one user and
one set of models exercising it, so edge cases outside that path may be rough.

If you can use a frontier hosted model, a tool built around those will do
better: [opencode](https://opencode.ai),
[Qwen Code](https://github.com/QwenLM/qwen-code), and
[Claude Code](https://claude.com/claude-code) are good options. Pedacito is for
when you specifically want to stay local and small.

---

## Install

Requires Python 3.11+.

```bash
cd /path/to/pedacito
pip install -e .            # Windows: py -m pip install -e .
```

This puts `pedacito` and `pdc` on your `PATH` and installs `requests` and
`tree-sitter`. If the command isn't found (common with per-user installs on
Windows), `python -m pedacito` (`py -m pedacito`) is identical.

Optional extras: `pip install -e ".[vision]"` (screenshots) and
`pip install -e ".[browser]"` (headless browser; then run
`python -m playwright install chromium`).

Uninstall with `pip uninstall pedacito-local`. Your config and every project's
`.pedacito/` folder are left untouched.

---

## Model server setup

Start any server that speaks the OpenAI chat-completions API, then give
Pedacito its base URL and a model name.

| Server | Start it | Base URL |
|---|---|---|
| **LM Studio** | Load a model, then start the server (Developer tab). For another machine, enable "Serve on Local Network". | `http://localhost:1234/v1` |
| **Ollama** | `ollama serve` and `ollama pull <model>`. For another machine, set `OLLAMA_HOST=0.0.0.0` first. | `http://localhost:11434/v1` |
| **Lemonade** | Start the server and load a model. | The URL it prints (ends in `/v1` or `/api/v1`) |
| **Hosted API** | Get an API key. | e.g. `https://api.deepseek.com/v1` |

For hosted APIs set `base_url`, `api_key`, and `model`. Pedacito detects a
remote host and switches to `json_mode = "object"` automatically.

Verify the connection from the machine where Pedacito runs:

```bash
pedacito config --check
```

It tells "server unreachable" apart from "model not found or won't load"
(usually a name mismatch or not enough free VRAM).

`pedacito config --doctor` goes further and calibrates the setup: it runs the
same check, then probes which structured-output modes the server really
supports (`json_schema`, `json_object`, plain prompting), prints a recommended
`json_mode`, reports the server's real context window and model weight size,
and calibrates token math against the server's own tokenizer. Verdicts are
remembered per server and model for a week, so a server without grammar
support automatically gets a few-shot JSON example in the prompt on later runs.

---

## Configure

```bash
pedacito config --setup     # interactive: finds LM Studio/Ollama, lists models
pedacito config --init      # or write a commented starter file
pedacito config             # show every resolved setting and where it came from
pedacito config --reference # print the full annotated template
```

The file is `~/.config/pedacito/config.toml` (Linux/macOS) or
`%APPDATA%\pedacito\config.toml` (Windows). The minimum:

```toml
[universal]
vram_gb = 8                             # GPU VRAM on the model server, in GB

[profiles.default]
base_url = "http://localhost:1234/v1"
model    = "your-model-id"              # exact id/name the server reports
```

`vram_gb` is required for local servers (with the model's file size it sizes
the context window) and not needed for hosted APIs.

Settings resolve in layers, each overriding the previous: built-in defaults →
user config → the project's `.pedacito.toml` → `--config FILE` → `--profile` →
command-line flags. Environment variables `PEDACITO_<KEY>` (for example
`PEDACITO_MODEL`) and `PEDACITO_PROFILE` are also honored. Unknown keys are
rejected with a suggestion. The legacy `[lmstudio]` section is still accepted as
an alias for `[server]`.

---

## Built-in help: `pedacito how`

`pedacito how` answers questions about using Pedacito, drawing on this README,
the [Quick Start](QUICKSTART.md), and the source. It needs no project and no
index, only a model connection, and it uses your config (never sending
`api_key`) so answers reflect your setup:

```bash
pedacito how "how do I undo an apply?"
pedacito how                          # interactive
```

It accepts `--url`/`--server`, `--model`, `--api-key`, `--vram`/`--vram-gb`,
`--config`, and `--profile`.

---

## Everyday workflow

The project path is the first argument; state lands in `<project>/.pedacito/`.

```bash
pedacito index   ~/code/anki                       # build the index (cached)
pedacito ask     ~/code/anki -t "make the leech threshold configurable"
pedacito explain ~/code/anki -q "how does the scheduler pick the next card?"
```

### `index`

Parses the project into files, symbols, and line ranges, and adds one-line LLM
summaries. `--no-summaries` builds the static index only (no model calls);
`--resummarise` regenerates cached summaries. Summaries are cached by content
hash, so re-indexing costs only what changed. `ask` and `chat` refresh the index
automatically; edits you make in your editor are detected on the next run.

Free inspection commands (no model needed):

- `pedacito files PROJ` — every candidate file and why it was kept or skipped.
- `pedacito cost PROJ` — estimated LLM calls and time for a first index
  (`--tps` sets your tokens/second).
- `pedacito map PROJ` — print the index (`--tier auto|flat|cards|dirs`).

### `ask`

Gathers the source the task needs, then proposes edits. **It is a dry run by
default**: you see a diff and nothing is written.

```bash
pedacito ask PROJ -t "..." --apply      # write immediately
pedacito ask PROJ -t "..." --review     # side-by-side sign-off page
pedacito ask PROJ -t "..." --reindex    # force a fresh index first
```

Other useful flags: `--steps N` (lookup limit), `--model`, `--url`, `--profile`,
`--config`, `--root`, `--exclude`, `--include`, `--ext`.

Every write snapshots the originals first; `pedacito restore PROJ` undoes the
last one.

### `explain`

Answers questions about the indexed project in prose, with no edits. It never
builds an index; run `pedacito index` first.

```bash
pedacito explain PROJ -q "trace the flow from answer to schedule"
pedacito explain PROJ                   # interactive
```

---

## What the model can do while gathering

The model works through a small set of actions:

| Action | Purpose |
|---|---|
| `read` | Numbered lines of a file, or a definition by symbol name. A mid-file read annotates the enclosing function/class and the module's imports, so the model always knows where it is. |
| `grep` | Regex or literal search (project, file, directory, or glob), hits labelled with their enclosing symbol. Falls back to literal and case-insensitive matching, and suggests similar symbols on a miss. |
| `find` | Search indexed definitions by name, signature, or summary — navigation by concept, where `grep` needs the exact source spelling. One line per hit with exact line ranges; a miss suggests similar names. |
| `callers` | Where a symbol is defined, what it calls, and who calls it. |
| `outline` | Every unit in a file (or the files in a directory/glob) with signatures and line ranges. |
| `sessions` | Search or read logs of earlier runs. |
| `diff` | What changed since Pedacito's own pre-write snapshots (`allow_diff`, on by default). |
| `todo` | Short `- [ ]` checklist saved to `.pedacito/todo.md` and shown in the answer prompt. |
| `done` | Stop gathering and answer. |

Opt-in actions, each behind a config flag because they widen what a model can
do:

| Action | Enable with | Purpose |
|---|---|---|
| `write` / `append` / `move` / `delete` | `allow_write = true` | Whole-file operations; `append` builds a file larger than one reply. |
| `run` | `allow_shell = true` | Run one command (e.g. `pytest -q`), without a shell. A quiet GUI/server is stopped after `shell_idle_seconds`. |
| `diagnose` | `allow_shell = true` | Run the detected linter/type checker (`ruff`, `pyflakes`, `flake8`, `npm lint`, `tsc`, `cargo check`, `go vet`, `rubocop`). Also available as `pedacito diagnose`. |
| `screenshot` | `allow_screenshots = true` | Capture the screen as text: size, brightness, blank-screen verdict, OCR, optional `vision_model` description. Needs the `vision` extra. |
| `browser` | `allow_browser = true` | Headless browser: start a server command, open the page, click/type, and report page text, canvas sizes, and console errors. Needs the `browser` extra. |
| `mcp` | `allow_mcp = true` + `mcp_servers = [...]` | Call tools from MCP servers. One live session per server is kept warm and reused; a dead session is restarted once. |
| semantic search | `semantic_search = true` | A `grep` with no hits falls back to the server's `/v1/embeddings` endpoint. |

`run` and `mcp` execute model-chosen commands in your project. Treat them as a
trust decision, not a sandbox.

### Seeing a GUI or web page

A text model can't see a window. `screenshot` and `browser` turn it into
evidence it can act on:

```toml
allow_screenshots = true
allow_browser = true
allow_shell = true                  # only needed to launch the app or server
vision_model = "your-vl-model-id"   # optional: multimodal model on the same server
```

`browser` takes a URL or a server command that prints one, plus optional
actions:

```text
browser  command="python main.py"  pattern="press Enter; wait 3"  start=4
```

This reliably catches black screens, blank pages, missing text, and console
errors; it does not judge visual quality. PNGs are saved under
`.pedacito/screenshots/`.

### Project instructions

Pedacito reads one conventions file into every prompt: the first found of
`PEDACITO.md`, `AGENTS.md`, `CLAUDE.md`, `.pedacito/instructions.md`, or
`.pedacito/PEDACITO.md` (the older spelling `PEDACEITO.md` is also accepted).
Put build/test commands, style rules, and off-limits directories there. It is
capped by `max_instruction_chars` (2000) and disabled with
`use_project_instructions = false`.

---

## More commands

### `chat` — interactive session

```bash
pedacito chat ~/code/anki [--apply | --review]
```

Keeps the index warm between tasks. Slash commands: `/profile NAME`,
`/model ID`, `/apply on|off`, `/steps N`, `/status`, `/profiles`, `/help`,
`/quit`. Switching profile mid-session is the cheapest way to compare two models
on one task; a failed switch leaves the session on the previous model.

### `new` — start a project from scratch

```bash
pedacito new my-app -t "a CLI todo manager with a sqlite backend" --apply
pedacito new my-api --spec docs/spec.md --review
pedacito new my-app -t "..." --plan-only --apply     # skeleton only
```

The model plans a file tree, the skeleton is written, then each file is filled
in. Dry run by default. The target directory must be empty or new unless
`--force`. `--complex` hands implementation to the `auto` loop.

### `auto` — autonomous loop over a task list

Breaks a goal into an ordered checklist and implements each task in a fresh
context, with optional verification and repair:

```bash
pedacito auto . -t "add retry logic to the sync client"      # plan only
pedacito auto . -t "..." --apply                             # plan, then implement
pedacito auto . --apply --verify "pytest -q"                 # verify and repair each task
pedacito auto . --apply --diagnostics                        # lint/typecheck each task
pedacito auto . --apply --worktree refactor                  # run in an isolated git worktree
pedacito auto . --apply --plan TODO.md                       # use an existing checklist
```

The plan is a markdown checklist (`PLAN.md` by default), so a run is resumable
and editable. There is no default cap on tasks, steps, or time; add one with
`--max-tasks N`, `--max-minutes N`, or `--steps N`. `--repair N` sets repair
rounds per failed verification, `--keep-going` continues past failed tasks, and
`--auto-install` installs a missing Python package instead of reporting it.

Guards for unattended runs: a repair round that makes the check *worse* (more
test failures, more findings) is rolled back from that round's own snapshot
and the task is marked failed, instead of compounding the damage. Plan titles
are linted at startup — one title should be one change, and a title bundling
several edits gets a warning. `auto_allowlist = true` restricts each task's
writes: existing files can only be modified, moved, or deleted when the task
title names them or the lookup phase read them; creating new files is always
allowed.

`--worktree NAME` runs in `.pedacito/worktrees/NAME` on a `pedacito/NAME`
branch, leaving your checkout untouched. Manage worktrees directly with
`pedacito worktree create|list|remove NAME`.

### `gui` — browser control panel

```bash
pedacito gui [--port N] [--no-open]
```

A local page (bound to `127.0.0.1`) to open a project, build the index, check
the model, run tasks with the review page inline, ask `explain` questions, and
restore snapshots. `how`, `new`, `auto`, and the advanced flags are CLI-only.

---

## What gets indexed

Code files are parsed into functions, classes, and methods with exact line
ranges. A unit larger than `max_unit_lines` (220), `max_unit_chars` (12000), or
`max_unit_tokens` (3000) is split into `name#2`, `name#3`, ... parts so one huge
function can't swamp a small model. Other useful files (`.md .txt .rst .csv
.tsv .json .toml .yaml .yml .ini .cfg .conf .env .sql .sh`) are indexed
structurally: the map shows a CSV's columns, a JSON file's keys, a markdown
file's headings, and so on, while `read` and `grep` reach their contents.

Other extensions are invisible until you add them: `--ext .kt` (or
`extra_extensions` in config) routes them through the structural chunker.

**Excluding files**, in increasing precedence:

1. `.pedacitoignore` in the project root (gitignore syntax); `.gitignore` is
   read too.
2. Flags: `--exclude 'deprecated/'`, `--include 'deprecated/still_used.py'`,
   `--no-refs` (code only), `--no-gitignore`.
3. Naming a path directly always wins.

**Map tiers.** Small projects get a flat listing of every function; larger ones
get one card per file; very large ones get one card per directory, which the
model descends with `outline <dir>`. `read` and `outline` reach exact text from
any tier. File cards also cross-link modules and their tests (`tested by:` /
`tests:`) whenever the `foo.py ↔ test_foo.py` names match unambiguously, so the
model finds the tests for a change without searching.

---

## Editing: search/replace, not diffs

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

A block matches the file exactly (ignoring trailing whitespace) or it fails
cleanly — and a failure shows the closest matching window in the file right
now, with line numbers and a similarity score, so the retry re-anchors on real
text instead of guessing. A FILE header can also pin a block from the start:
`--- FILE: src/app.py LINES 40-60` replaces that exact range, and
`SYMBOL is_leech` replaces that whole definition — useful when a snippet
appears more than once or the text keeps shifting under the model.

Before writing, Python is checked with `ast.parse` and tree-sitter
languages with their parsers: an edit that breaks a clean file is refused, while
a file that was already broken is edited with a warning. A failed block's error
goes back to the model for up to `repair_rounds` retries (default 2).

After a write, `post_edit_check` (on by default) runs the cheap check — the
detected linter, else a syntax parse — on the changed files and feeds back only
findings that were *not* there before the task, giving the model up to
`repair_rounds` fix rounds to clean up its own mess without chasing
pre-existing issues.

Set `fuzzy_match = true` to let near-miss blocks anchor to the most similar
text (`fuzzy_min_ratio`, default 0.85). Such edits are flagged as approximate.

## Review page

`--review` (or `review = true`) opens an offline, side-by-side page: current file
left, proposed version right, changed words highlighted. To write, tick the
attestation box, click **Sign off & write**, then confirm. Enter never writes.

The page records each file's SHA-256 when built, and applying re-checks it: if
you edit a file in the meantime, the write is refused instead of overwriting
your change. The server binds to `127.0.0.1` only and rejects foreign `Host`
and `Origin` headers.

## Undo and workspace layout

```bash
pedacito restore PROJ --list
pedacito restore PROJ --dry-run
pedacito restore PROJ                        # newest snapshot, byte-for-byte
pedacito restore PROJ --snapshot 20260830-020724
```

Files created after a snapshot are left in place. Everything Pedacito writes for
a project lives under `<project>/.pedacito/`:

```
index.json            the map
summaries.json        summary cache, keyed by content hash
embeddings.json       semantic-search cache (if enabled)
backups/<stamp>/      pre-write snapshots
sessions/<stamp>.md   what was asked, looked up, and changed
reviews/<stamp>.html  saved review pages
screenshots/          captures (if enabled)
todo.md               current task checklist
log.txt               timestamped event trace
```

`sessions/` is the place to look when an edit turns out wrong days later;
`log.txt` shows where a failure happened. On first run Pedacito creates a
commented `.pedacitoignore` and adds its entries to `.gitignore` (switch off
with `create_ignore_file = false` / `manage_gitignore = false`). Deleting
`.pedacito/` is always safe; it only costs a re-index.

---

## Profiles, tiers, and token budgets

Define a profile per model instead of editing `model =` each time:

```toml
[profiles.small]
model = "qwen2.5-coder:7b"

[profiles.big]
model = "some-27b-model"
tier  = "large"
```

```bash
pedacito ask PROJ -t "..." --profile big
```

Choose a profile with `default_profile =`, `PEDACITO_PROFILE`, `--profile`, or
`/profile` in `chat` (increasing precedence). A profile can override any
setting, not just the model.

`tier = "tiny" | "small" | "medium" | "large" | "hosted"` fills in all token
budgets (`max_tokens_*`, `flat_map_token_budget`, `max_gathered_chars`, ...)
from a preset; `tiny` suits ≤4B models with a ~4k window. When unset, the tier
is guessed from the model name (≤4B → tiny, ≤10B → small, ≤20B → medium, else
large; a hosted keyword wins). You rarely need to tune these: Pedacito detects
the server's real context window and the model's file size and fits every
budget to your VRAM (`auto_fit`, on by default). Explicit values always win,
and `pedacito config` marks every non-default value.

If your runtime uses a quantized KV cache, set `k_cache_quant` and
`v_cache_quant` (e.g. `"q8_0"`, `"q5_1"`) so the VRAM estimate isn't too
pessimistic. If the server reports a stale default window instead of your
configured one, set `context_tokens` explicitly.

Switching models makes the server load the new one; Pedacito waits up to
`load_wait_seconds` (240).

Reasoning models (Qwen3 and similar) are supported: `<think>` blocks are
stripped before parsing, and `disable_thinking` / `thinking_suffix` turn
thinking off. Mistral-style role alternation and Gemma's missing system role are
handled automatically.

---

## Configuration reference

`pedacito config` lists every setting with its value and origin; `pedacito
config --reference` prints the annotated template. Key settings:

| Setting | Default | Meaning |
|---|---|---|
| `base_url` | `http://localhost:1234/v1` | Server address. |
| `model` | `google/gemma-3-27b` | Exact id/name the server reports. |
| `api_key` | `""` | For hosted APIs and secured servers. |
| `vram_gb` | *(required for local)* | GPU VRAM; set in `[universal]`, overridable per profile. |
| `context_tokens` | auto | Server-side context window; set when detection is wrong. |
| `k_cache_quant` / `v_cache_quant` | `""` | KV-cache quantization the runtime uses. |
| `json_mode` | `"schema"` | `"schema"` for local servers, `"object"` for hosted (auto-switched). |
| `tier` | `""` | Budget preset: `tiny`/`small`/`medium`/`large`/`hosted`; guessed from the model name when unset. |
| `max_steps` | 0 | Lookup limit per task; 0 = none. |
| `temperature` | 0.2 | Higher values make tool selection worse. |
| `backup` | true | Snapshot originals before writing. |
| `log_sessions` | true | Write a session record per task. |
| `review` | false | Always use the review page. |
| `fuzzy_match` | false | Anchor near-miss search blocks approximately. |
| `post_edit_check` | true | After applied edits, feed back only NEW linter/syntax findings on the changed files. |
| `auto_allowlist` | false | Restrict `auto` task writes to files the title names or the lookup read (new files always allowed). |
| `allow_write` | false | Whole-file write/append/move/delete during gathering. |
| `allow_shell` | false | `run` and `diagnose`. |
| `allow_screenshots` / `allow_browser` | false | Screen and headless-browser capture. |
| `vision_model` | `""` | Multimodal model used to describe captures. |
| `semantic_search` | false | Embeddings fallback for `grep`. |
| `allow_diff` | true | Read-only `diff` action. |
| `disable_thinking` | false | Suppress `<think>` blocks. |
| `default_profile` | `""` | Profile used when `--profile` isn't given. |

Everything else (chunking, budgets, browser timeouts, auto-loop behavior) has a
working default and is documented in `pedacito config --reference`.

---

## Known limits

- The call graph matches bare names, so two unrelated `run()` methods are
  conflated — fine for locating code, not for automated refactoring.
- Semantic search needs an embeddings endpoint; the first query embeds the whole
  index once and caches it.
- A text-only model can't see graphics; `screenshot` and `browser` give it text
  evidence, not visual judgement.
- Broad, structural tasks ("modernize this codebase") are a poor fit for small
  models. Narrow, localized tasks are where this works well.

---

## Contributing

Issues and pull requests are welcome: bug reports with a repro, support for
other model servers or languages, and fixes for anything that breaks outside the
setup this was built on. Please open an issue before a large PR.

## License

MIT — see [LICENSE](LICENSE).
