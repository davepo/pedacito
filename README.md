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
- The map is **ranked for the task**: files the task names, matches, imports,
  or is tested by come first, so a small model's first lookup is usually the
  right one. Import/call graph, BM25 search, and test links work in every
  supported language.
- Edits are search/replace blocks, validated before anything is written —
  and checked first for the classic small-model mistakes (editing code it never
  read, leaving a removed function's callers behind, `TODO` stubs).
- It **measures** your model instead of guessing from its name
  (`pedacito calibrate`), keeps the server's prompt cache warm (one fold point,
  answer continues the lookup conversation), and reports tokens, cache-hit
  ratio and cost per task.
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

Once per model, run `pedacito calibrate`. It spends about a minute on a scripted
mini-suite — tool choice, JSON adherence, edit format, planted-injection and
honesty checks, long-context retrieval, continuation, capabilities — and saves
a **model card**. Settings you left at their defaults then follow the card
(prompt pack, answer mode, verifier, best-of-N, fold depth); anything you set
explicitly still wins. `pedacito config` shows the card.

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
pedacito ask PROJ -t "..." --plan       # split a multi-part task into steps
pedacito ask PROJ -t "..." --jsonl      # one JSON event per line on stdout
```

Other useful flags: `--steps N` (lookup limit), `--model`, `--url`, `--profile`,
`--role planner=big` (route one job to another profile for this run),
`--config`, `--root`, `--exclude`, `--include`, `--ext`.

**Plan mode.** A task that changes three things in one context window usually
gets the first done, the second half done and the third forgotten. `--plan`
(or `plan_threshold`, which a model card sets for weaker models) gives each
change its own window, lookups, evidence checks and rollback. A task written as
bullets or several clear requirements is split deterministically, with no model
call; anything else goes to the `planner` role (which may declare
`depends_on`). The plan is a markdown checklist in `.pedacito/plans/` you can
read and edit (`--review-plan` pauses for that). After a failed step the planner
is asked once to revise the remaining steps (`max_replans`).

**Machine-readable runs.** `--jsonl` prints every event (steps, model calls with
token usage, the final answer and the diffs) as one JSON object per line on
stdout and moves the human text to stderr, so an editor plugin or script can
drive Pedacito. Writing runs (`--apply`) take a per-project lock
(`.pedacito/run.lock`); a second writer is refused while the first is alive.

**Steering and questions.** In `chat`, anything you type while the model is
working is delivered at its next step ("stop reading tests, it's in
`sync.py`") without disturbing the cached prompt prefix (`allow_steering`). In
an interactive `ask`/`chat` the model also gets a `question` action for genuine
ambiguity, at most twice per task; in `auto` and with `--jsonl` it never asks
and states its assumption instead.

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
| `find` | Search indexed definitions by name, signature, or summary — navigation by concept, where `grep` needs the exact source spelling. Substring hits first, then BM25 (so a multi-word concept like `retry backoff` works). One line per hit with exact line ranges; a miss suggests similar names. |
| `callers` | Where a symbol is defined, what it calls, who calls it, and the files and tests a change to it can affect (transitive importers/callers, two hops). |
| `context` | One reply instead of four: a definition's numbered source, its class header, the signatures it calls, its callers with call sites, its tests, and the likely blast radius. Also works on a file. |
| `outline` | Every unit in a file (or the files in a directory/glob) with signatures and line ranges. |
| `sessions` | Search or read logs of earlier runs. |
| `diff` | What changed since Pedacito's own pre-write snapshots (`allow_diff`, on by default). |
| `git` | Read-only git: `status`, `diff`, `log`, `show`, `blame` (`allow_git`, on in a repository). |
| `remember` | Save one durable fact (a build command, a convention, a trap) to `.pedacito/memory.md` for future runs. |
| `skill` | Load a named procedure from the skills list (`add-a-test`, `fix-a-failing-test`, ...). |
| `todo` | Short `- [ ]` checklist saved to `.pedacito/todo.md` and shown in the answer prompt. |
| `question` | Ask *you* one short question (interactive sessions only, at most twice a task). |
| `done` | Stop gathering and answer. |

Repeating a read-only lookup (even many steps later) that nothing has changed
since gets a pointer to the earlier result instead of a second copy, and any
non-read observation (a broad `grep`, a chatty command, an MCP reply) is capped
at a fraction of the window (`max_observation_tokens`) with a note on how to
narrow it.

The action list is **filtered to what is enabled and to the model's size**: a
tiny profile sees ten actions and a ~430-token prompt, not two dozen tools and
browser-eval advice it cannot use.

Opt-in actions, each behind a config flag because they widen what a model can
do:

| Action | Enable with | Purpose |
|---|---|---|
| `write` / `append` / `move` / `delete` | `allow_write = true` | Whole-file operations; `append` builds a file larger than one reply. |
| `rename` | `allow_write = true` | Mechanically rename one definition and its references in the files that import it (strings and comments untouched, every file re-parsed, snapshotted). Refuses a name defined in several places and lists the references it left alone. |
| `run` | `allow_shell = true` | Run one command (e.g. `pytest -q`), without a shell. A quiet GUI/server is stopped after `shell_idle_seconds`. |
| `diagnose` | `allow_shell = true` | Run the detected linter/type checker (`ruff`, `pyflakes`, `flake8`, `mypy`/`pyright` when configured, `npm lint`, `tsc`, `cargo check`, `go vet`, `rubocop`). Also available as `pedacito diagnose`. |
| `screenshot` | `allow_screenshots = true` | Capture the screen as text: size, brightness, blank-screen verdict, OCR, optional `vision_model` description. Needs the `vision` extra. |
| `browser` | `allow_browser = true` | Headless browser: start a server command, open the page, click/type, and report page text, canvas sizes, and console errors. Needs the `browser` extra. |
| `mcp` | `allow_mcp = true` + `mcp_servers = [...]` | Call tools from MCP servers. The catalog shows each tool's arguments, calls are validated against the tool's schema, one live session per server is kept warm, and a server that was down at startup is retried. |
| `task` | `allow_subagents = true` | Hand a bounded, read-only research question to a helper agent with a fresh context window; only a short digest comes back. Depth is capped (1 local, 2 hosted). |
| `fetch` | `allow_fetch = true` + `fetch_allowed_hosts` | URL → readable text, host-allowlisted, size-capped, cached, labelled untrusted. Private/loopback addresses are refused. |
| semantic search | `semantic_search = true` | A `grep` with no hits falls back to the server's `/v1/embeddings` endpoint. |

`run` and `mcp` execute model-chosen commands in your project. Treat them as a
trust decision — but see [Safety](#safety-and-trust): `run` is sandboxed by
default where the OS allows, and the rest of the blast radius is limited too.

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

### Memory, rules, and skills

Pedacito can remember what earlier runs learned — as plain markdown you can
edit or delete:

- **Memory** (`.pedacito/memory.md`, plus a user-level `memory.md` next to your
  config) holds short facts under Commands / Conventions / Landmines / Notes.
  The model adds one with `remember`; you can too:
  `pedacito memory PROJ --add "command: run tests with pytest -x"`,
  `--list`, `--forget TEXT`. Lines are de-duplicated, capped per category,
  secrets are masked, and anything that reads like an instruction to an
  assistant is refused. It is shown to the model as *hints, not truth*.
- **Rules** (`.pedacito/rules/*.md` with a `paths: src/api/*.py` frontmatter
  line, and nested `AGENTS.md`/`CLAUDE.md` files) are shown only when the model
  reads a matching file.
- **Skills** are named procedures (`.pedacito/skills/NAME/SKILL.md`, or
  `skills/NAME/SKILL.md` next to your config). The model sees a one-line
  catalog and loads a skill's text only when it needs it. Five starters ship
  built in; `pedacito skills` lists them. Skills are prompt text — nothing in a
  skill folder is ever executed.

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
`/model ID`, `/apply on|off`, `/steps N`, `/status` (pack, answer mode,
verifier, window, model card), `/cost` (tokens, cache hits and spend by call
site), `/memory`, `/skills`, `/profiles`, `/help`, `/quit`. **Ctrl-C cancels the
running task but keeps the session open.** Switching profile mid-session is the
cheapest way to compare two models on one task; a failed switch leaves the
session on the previous model.

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
pedacito gui [PROJECT] [--profile NAME] [--port N] [--no-open]
```

A local page (bound to `127.0.0.1`, no CDN or extra dependencies, so it works
offline) that runs the same engine as the commands (`ask`, `explain`, `auto`,
`new`), so planning, preflight checks, escalation and post-edit tests behave
identically.

| Area | What it does |
|---|---|
| **Header** | Open a project (type a path, **Browse…** the machine's folders, or pick a recent one), choose its model profile, see the model server's status light and the session's token/cache/cost totals. |
| **Chat** (left) | Four modes next to the box: **Task** (edit code; *review* or *write directly*, planning auto/always/never, max lookups), **Explain** (read-only questions, answer streams in), **Auto** (goal → plan → implement; plan-only or implement, verify command, repair rounds, task/minute caps, worktree, keep-going, pip-install), **New** (start a project from scratch in a folder: preview, build, build-then-review, skeleton only, and *complex* for the autonomous loop). **Attach** (or drag in) a text file as the task, question, or spec. While a task runs, type to **steer** the model, answer its clarifying questions inline, or **Stop** it. |
| **Changes** (middle) | The proposed edits as *Old code | Updated code* side by side with word-level highlights. Accept or reject the whole set, untick files to accept only some, or **Edit updated** to hand-edit a file before accepting. Backups are made first and a file that changed on disk since the preview is refused. |
| **Plan** (middle) | The `auto`/plan checklist with live progress; edit it, then **Run this plan**. |
| **File** (middle) | Read-only viewer for any indexed file (click it in the tree). |
| **Terminal** (bottom) | Everything pedacito prints, live, with ANSI colours — the same text the launching terminal shows. |
| **Files** (right) | Filterable tree of the indexed files; files in the pending review are highlighted. |
| **Settings** | See below. |

Panels resize (drag the dividers), the layout and colour scheme are
remembered, and on a narrow window the panels become tabs.

**Settings** edits the user config or the open project's `.pedacito.toml`
without touching the rest of the file — comments survive, every edit is checked
with the same rules the loader uses before it is written, and the previous file
is kept as `config.toml.bak`. Tabs: **Profiles** (create, duplicate, delete,
make default; server URL, key, model with a *Test connection* that lists the
server's models, tier, VRAM, JSON mode, and the common switches), **All
settings** (every setting, searchable, with its help text and where its value
comes from, applied globally or to one profile), and **Raw file**. With no
config at all, a first-run form (with *Find servers* for LM Studio/Ollama) builds
one. A project config's execution settings stay ignored until you trust it
(*Review & trust…*, the same as `pedacito trust`).

How it stays safe: it refuses requests with a foreign `Host`/`Origin`, accepts
only JSON `POST`s, sends a strict Content-Security-Policy, and serves a project's
files only from inside that project. API keys are never shown in the profile
forms (leave the box blank to keep one); the **Raw file** tab does show the file
as written, key included. Only one job runs at a time, and jobs that write directly take the
same project lock as the command line.

### Measurement and upkeep

```bash
pedacito calibrate                 # measure the model; save a model card
pedacito eval [--task ID] [--out report.json]   # score the harness on fixtures
pedacito map PROJ -t "fix X"       # see the map a task would get (ranked)
pedacito tests PROJ --files src/a.py   # detected test command + impacted tests
pedacito memory PROJ [--add T | --forget T]
pedacito skills PROJ
pedacito prune PROJ --keep-sessions 50 --keep-backups 20 [--dry-run]
pedacito trust PROJ                # approve a project's .pedacito.toml
pedacito ask PROJ -t "..." --record          # save the model's responses
pedacito ask PROJ -t "..." --replay FILE     # re-run the harness, no model needed
```

`calibrate` also measures **native tool calls** against the JSON grammar on your
model (same eight scenarios, both modes) and the card switches `tool_mode` to
`native` only when it picks the right action at least 0.1 more often with no
loss of well-formed steps. Native mode sends the step schema as one forced
function and reads the model's tool-call arguments, which suits models trained
on tool calling; a server that rejects `tools` falls back to the grammar once
and remembers. Its short probes (instruction following, long-context needle)
get a generous token budget and retry once with thinking off, so a reasoning
model that spends a tiny budget on `<think>` is not scored as failing the skill.

`--record` writes the ordered model responses to `.pedacito/replay/<stamp>.jsonl`
(prompts as a hash and a short redacted preview, never in full); `--replay`
feeds them back with no server, running the whole harness (gather loop, folding,
preflight, edit parsing, repair) deterministically and free. It reports which
calls were asked a *different* prompt than when recorded: that is how you test a
change to the harness rather than to the model.

`pedacito eval` runs eight tasks over three fixture repositories (Python,
TypeScript, Go) — each with a gold file set, a substring check, a real test for
Python, and a planted prompt injection — and prints locate rate, first-apply
success, repair calls, tokens, cache-hit ratio, wall time and a failure
taxonomy (wrong file, unread edit, format error, truncation, refusal,
hallucination, test regression, injection followed). Use it to A/B any setting:
run it, change `prompt_pack` or `answer_mode`, run it again.

---

## What gets indexed

Code files are parsed into functions, classes, and methods with exact line
ranges (Go methods are qualified by their receiver, Rust `impl` blocks and
nested `mod`s and C# namespaces are chunked, and a script with no functions is
split at statement boundaries instead of becoming one giant unit). A unit larger than `max_unit_lines` (220), `max_unit_chars` (12000), or
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
get one card per file; very large ones get a **ranked map**: the cards of the
files most likely relevant to the task (named in it, matching its words, close
to those in the import graph, tested by or touching them, uncommitted or
recently changed), each with a `why listed:` line, above a hierarchical
directory skeleton of everything else. A map that fits is listed alphabetically
and stays byte-identical across tasks so the server's prompt cache survives;
ranking only decides *what* to show when something has to be left out.
`read`, `outline`, and `find` reach exact text from any tier. A repo brief
(stack, entry points, test/lint/build commands, directory roles) sits above the
map, built from your manifests with no model call.

File cards carry blast radius — `imported by: 7 file(s)`, or a labelled
"no importers or callers found (evidence, not proof)" — and cross-link modules
with their tests (`tested by:` / `tests:`) for Python, JS/TS (`x.test.ts`,
`__tests__/`), Go (`x_test.go`), Rust (`tests/`), Java/C# (`XTest`), and Ruby
(`x_spec.rb`) when the match is unambiguous. `pedacito map PROJ -t "task"`
prints exactly what a task would see.

**Secrets are never indexed.** `.env`, `.env.*` (but not `.env.example`),
`*.pem`, `*.key`, `id_rsa*`, `.npmrc`, `.pypirc`, `credentials.json` and
similar are skipped by default; re-include one deliberately with `!.env` in
`.pedacitoignore`. Excluded directories (`node_modules`, `build`, ...) are
pruned before they are walked, and nested `.gitignore` files are honoured.

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
A block the model *started* but never finished (a missing `=======` or
`>>>>>>> REPLACE`, a block with no `--- FILE:` header) is reported back as a
failure instead of being silently dropped.

### Checking an answer before it is written

A small model's usual failure is not bad syntax but a plausible edit to the
wrong place. Before anything is written, `verify_answer = "deterministic"` (the
default) dry-applies the blocks and runs free checks — no model call:

| Check | Catches |
|---|---|
| unread | an edit to a file, or a region of one, the lookup phase never `read` |
| placeholder | `TODO` / `NotImplementedError` / bare `pass` instead of working code (unless the task asks for a stub) |
| stale references | a removed or renamed definition that is still referenced elsewhere — judged against what *all* of the reply's edits leave behind, so a correct multi-file rename passes |
| moved imports | a moved file still imported by its old path |
| citations | `file:line` claims about files never read |
| coverage | a requirement from the task (it extracts up to five) no edit or sentence touches — a warning |

Errors cause one corrected re-ask that shows the model its own blocks and the
findings; warnings are printed and logged. `verify_answer = "full"` adds one
adversarial verifier call (on the `verifier` role, if you set one), and
`best_of = 3` samples extra candidates at a higher temperature and keeps the
best-scoring one — both cost model calls, and `pedacito calibrate` turns them
on for models that measure weak.

With `verify_tests = true` and `allow_shell = true`, the tests *impacted* by an
edit (the tests named after the file, tests of its importers and callers) run
after it is written; new failures — compared with a run before the edit — go
back to the model for up to `repair_rounds` fix rounds. `test_command`
overrides detection (`pytest`, `npm test`, `cargo test`, `go test ./...`,
`make test`, ...).

## How context is managed

A small window is spent three ways: the prompt's fixed text, the map, and the
conversation. Pedacito spends each on purpose.

- **Prompt packs.** The instructions come in `full`, `small` and `tiny`
  density (`prompt_pack`, else the model card, else the tier) and mention only
  the actions that are enabled. With browser, shell and write off, the default
  lookup prompt is about 1,700 tokens instead of 3,000; `tiny` is about 430.
  Override any section with `.pedacito/prompts/{gather,edit_format,answer}.md`.
- **One fold point.** A server caches the prompt *prefix*; rewriting an old
  turn throws the cache away from that turn on. So the lookup conversation is
  left alone until the window is really full, then the whole older band is
  folded into **one** `STATE` digest (what was looked up, one line each) and the
  last few steps stay verbatim. One cache break, not one per step. A model can
  write the digest instead of the extractive default (`compaction = "model"`).
- **The answer continues the lookup conversation** (`answer_mode`): the final
  instruction is appended, so the server reuses the whole warm prefix instead
  of prefilling the map and the source a second time. If the conversation was
  folded, or the continued reply is empty or still shaped like a tool call, it
  falls back to a rebuilt prompt. Hosted APIs default to `auto`; local models to
  `rebuild` until a model card says otherwise.
- **Degrade, don't refuse.** An answer prompt over the window drops project
  notes, then shrinks the map, then drops the oldest *whole* source items
  (leaving a re-read pointer for each) before it ever errors.
- **Stuck recovery.** A repeated step, a run of failed steps, or repeated
  reads get one targeted recovery prompt ("you sent the identical `read` four
  times; do something different") before gather gives up.
- **Visible accounting.** Every task prints calls, tokens in/out, the
  cache-hit ratio and (with `price_*_per_mtok` set) cost; `/cost` breaks it down
  by call site; `max_total_tokens` / `max_cost` (or `--max-tokens-budget` /
  `--max-cost`) stop the run before the next call.

## Safety and trust

Pedacito is local-first and sends nothing anywhere except to the model server
you configured. Within that:

- **Tool output is data, never instructions.** The system prompt says so;
  output from `run`, `browser`, `fetch`, `mcp`, `git` and `screenshot` is fenced
  as untrusted, control characters are stripped, and instruction-like wording
  ("ignore previous instructions...") is flagged in the observation.
  `pedacito eval` plants injections to measure this.
- **Commands get a minimal environment.** A command the model chooses (and dev
  servers it starts) sees `PATH`, `HOME`, locale and toolchain variables — not
  your API keys or tokens. Add what a project needs with `env_passthrough`
  (`["DATABASE_URL", "AWS_*"]`); `scrub_env = false` restores the old behaviour.
- **Protected paths.** The model can never write `.git/`, `.pedacito/`,
  `.pedacito.toml`, the autonomous loop's plan file (only the loop ticks its
  boxes), or anything in `protected_paths`. It will not edit a file that has
  unresolved git merge conflicts. `shell_deny = ["git push*", "rm -rf *"]`
  refuses matching commands even when the executable is allowed.
- **OS sandbox (on by default).** `run` and the tests an edit triggers run under
  an unprivileged OS sandbox: they can read the filesystem but write only to the
  project and a private temp directory; `.git/` (hooks!), `.pedacito/` and the
  project config are read-only; `~/.ssh`, `~/.aws`, `~/.config/pedacito` and
  similar are hidden; and the **network is off**. Backends: bubblewrap on Linux
  (`bwrap`), `sandbox-exec` on macOS; Windows has none that works without admin
  rights. Where no backend works (not installed, user namespaces disabled,
  Windows) commands run unsandboxed after ONE notice, so Pedacito behaves the
  same everywhere. Opt out with `sandbox = false`; allow the network with
  `sandbox_network = true` (needed for tests that talk to a server or install
  packages); add writable directories with `sandbox_writable = ["~/.cache/pip"]`;
  `sandbox_strict = true` refuses to run a command when no sandbox can be
  applied. A failed command's observation says it ran sandboxed, so the model
  does not chase a blocked write as a code bug. `pedacito config --doctor` and
  `/status` show whether it is active. Your own hooks and `auto_verify` are
  never sandboxed.
- **Browser `eval`** runs model-written JavaScript, so it is limited to local
  pages (`browser_eval_hosts`).
- **Secrets stay out.** Secret-bearing files are not indexed (above); secret-
  looking values are masked in session files, logs and events
  (`redact_secrets`); a task that appears to contain one warns before going to
  a hosted API.
- **A repository cannot run commands behind your back.** A project's own
  `.pedacito.toml` may not set `hooks`, `mcp_servers`, `auto_verify` or
  `test_command` until you read it and run `pedacito trust`; editing the file
  revokes the approval.
- **Hooks** (`hooks = { post_edit = ["ruff format {file}"], pre_edit = [...] }`)
  run your own program at `session_start`, `pre_task`/`post_task`,
  `pre_tool`/`post_tool`, `pre_run`/`post_run`, `pre_edit`/`post_edit`,
  `post_compact`, `notification`: a JSON event on stdin, an optional JSON
  decision on stdout, exit status 2 to deny a `pre_*` event. No shell, a
  timeout, the scrubbed environment; a failing hook is a warning.

`browser` and MCP servers still execute unsandboxed on your machine (a server is
a program you chose to run). The sandbox contains `run` and test commands; it is
defence in depth, not a promise against a determined adversary or a kernel bug.

## Roles: a different model per job

```toml
[universal]
roles = { planner = "big", verifier = "big", summarizer = "tiny" }

[profiles.local]                  # the default worker: lookups and edits
model = "qwen3-9b"

[profiles.big]
base_url = "https://api.deepseek.com/v1"
api_key  = "sk-..."
model    = "deepseek-chat"

[profiles.tiny]
model = "qwen3-1.7b"
```

Roles are `planner` (turns a goal into the `auto` checklist and plan-mode
steps), `verifier`, `summarizer` (the index's one-line summaries and
compaction), and `repair`, plus two recovery rungs: `escalate` (one more
attempt on that model when a task ends with no edit applied) and `fallback`
(takes over when the worker errors out). Each rung is used at most once per
task and only when configured, so a local model's bad day cannot silently burn
a hosted budget. An unset role uses the active model; a role whose server is
unreachable falls back to it with one warning. Usage is counted per call site
either way. `--role planner=big` sets one for a single run.

A task that ends without a usable edit is also remembered in
`.pedacito/attempts.json` (files looked at, why); a similar later task is told
what was already tried. A success clears it.

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
pedacito restore PROJ --delete-created       # also remove files that run created
```

Files created after a snapshot are left in place unless you pass
`--delete-created` (the run's created files are listed in a sidecar
`backups/<stamp>.created`; directories left empty are removed too). Everything Pedacito writes for
a project lives under `<project>/.pedacito/`:

```
index.json            the map (versioned, written atomically)
summaries.json        summary cache, keyed by content hash
embeddings.json       semantic-search cache (if enabled)
backups/<stamp>/      pre-write snapshots (never merged: -2, -3 on a clash)
sessions/<stamp>.md   what was asked, looked up, and changed (secrets masked)
events/<stamp>.jsonl  one JSON line per step, call, fold, verify, task
reviews/<stamp>.html  saved review pages
screenshots/          captures (if enabled)
memory.md             remembered facts (hints for the model)
rules/*.md            path-scoped rules       skills/*/SKILL.md   your skills
prompts/*.md          prompt overrides        fetch/              fetch cache
todo.md               current task checklist
log.txt               timestamped event trace
```

`sessions/` is the place to look when an edit turns out wrong days later;
`log.txt` shows where a failure happened. On first run Pedacito creates a
commented `.pedacitoignore` and adds its entries to `.gitignore` (switch off
with `create_ignore_file = false` / `manage_gitignore = false`). Deleting
`.pedacito/` is always safe; it only costs a re-index. `pedacito prune` keeps
the newest N sessions, backups and event files and trims `log.txt`.

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

Reasoning models are supported across families: `<think>`/`<thinking>`/
`<reasoning>`, Magistral's `[THINK]`, Seed-OSS's `<seed:think>`, GPT-OSS
harmony `analysis` channels, and Gemma 4's `<|channel>thought` block are
stripped before parsing, while separate `reasoning_content`/`reasoning`/
`thinking` fields are read directly. The
`disable_thinking` switch is family-aware: local templates get the common
`enable_thinking=false` and `thinking=false` kwargs (unknown template variables
are ignored), plus `reasoning_effort=none` for Magistral and
`thinking_budget=0` for Seed-OSS. Recognised hosted providers get their own
spelling (Kimi's `thinking={"type":"disabled"}`, DashScope's
`enable_thinking`, DeepSeek's template kwarg, `reasoning_effort` on
OpenAI/Cerebras/GPT-OSS); an unknown hosted model gets no extra field, so a
strict API cannot reject it. `thinking_suffix` adds a prompt soft switch, and
auto-selects `/no_think` for Qwen3/SmolLM3. The Qwen, Gemma, Nanbeige, and
LiquidAI families all use `enable_thinking` and are covered automatically.
Some families have no off switch (GPT-OSS, Kimi K2.7/K3, LFM2.5-24B-A2B); their
reasoning blocks are still stripped, but the tokens are unavoidable. For an
unrecognised family, add the switch your model expects under the profile's
`sampler` table (user keys survive and merge with the automatic ones).
Mistral-style role alternation and Gemma's missing system role are handled
automatically.

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
| `disable_thinking` | false | Ask for non-thinking mode using the model family's switch; output blocks are stripped regardless. |
| `default_profile` | `""` | Profile used when `--profile` isn't given. |
| `prompt_pack` | `""` | `tiny` / `small` / `full` instruction density; empty = model card, then tier. |
| `answer_mode` | `""` | `rebuild` / `continue` / `auto`; empty = model card, then `auto` (hosted) or `rebuild` (local). |
| `compaction` | `"deterministic"` | How folded lookups are summarised: `deterministic`, `model`, or `off` (legacy per-turn clipping). |
| `rank_map` | true | Rank the map for the task when it doesn't all fit. |
| `repo_brief` / `max_brief_tokens` | true / 500 | Deterministic orientation block above the map. |
| `verify_answer` | `"deterministic"` | `off` / `deterministic` / `full`: check edits before writing them. |
| `best_of` | 1 | Sample N candidate edit replies and keep the best-scoring one. |
| `verify_tests` / `test_command` | false / `""` | Run impacted tests after applied edits (needs `allow_shell`). |
| `use_memory` / `use_skills` | true | Push memory, rules, and the skills catalog. |
| `allow_git` / `git_context` | true | Read-only `git` action; branch/dirty/recent-commits preamble. |
| `allow_subagents` | false | The `task` action (read-only helper with a fresh window). |
| `allow_fetch` / `fetch_allowed_hosts` | false / `[]` | The `fetch` action and the hosts it may read. |
| `roles` | `{}` | Per-job profiles: `planner`, `verifier`, `summarizer`, `repair`. |
| `hooks` | `{}` | Your commands at events (project-level hooks need `pedacito trust`). |
| `scrub_env` / `env_passthrough` | true / `[]` | Minimal environment for model-chosen commands. |
| `protected_paths` / `shell_deny` | `[]` | Paths the model may not write; command patterns `run` refuses. |
| `wrap_observations` / `redact_secrets` | true | Fence untrusted output; mask secrets in logs and sessions. |
| `max_total_tokens` / `max_cost` | 0 | Stop before the next call once spent (0 = no limit). |
| `price_input_per_mtok` / `price_output_per_mtok` / `price_cached_per_mtok` | 0 | USD per million tokens, for cost reporting. |
| `stuck_recovery` | true | One targeted recovery prompt before gather gives up. |
| `use_model_card` | true | Let `pedacito calibrate`'s card fill unset settings. |
| `log_events` / `keep_events` | true / 20 | Local JSONL telemetry under `.pedacito/events/`. |
| `summary_workers` / `refresh_summaries` | 0 / 12 | Concurrent summaries (0 = auto: hosted 4, local 1); units re-summarised after a task. |
| `summary_batch` | 1 | Short units summarised N per request while indexing (fewer round-trips; unparsed lines fall back to single requests). |
| `max_observation_tokens` | 0 | Ceiling on one non-read observation, in tokens (0 = auto: a quarter of the window, 300-2200). |
| `plan_threshold` / `max_replans` | 0 / 1 | Auto-plan tasks naming this many files or requirements (0 = only with `--plan`); planner revisions after a failed step. |
| `allow_steering` | true | Deliver lines typed mid-task to the model in `chat`. |
| `tool_mode` | `grammar` | `native` sends lookup steps through the model's tool-call channel; `calibrate` picks. |
| `sandbox` / `sandbox_network` | true / false | OS sandbox for `run` and test commands; allow network inside it. |
| `sandbox_writable` / `sandbox_strict` | `[]` / false | Extra writable dirs; refuse to run when no sandbox can be applied. |
| `atomic_edits` | false | Write a multi-file reply all-or-nothing (any failure writes nothing). |

Everything else (chunking, budgets, browser timeouts, auto-loop behavior) has a
working default and is documented in `pedacito config --reference`.

---

## Known limits

- The call graph matches bare names, so two unrelated `run()` methods are
  conflated — fine for locating code, not for automated refactoring. Import
  edges are resolved per language (relative paths, packages, `crate::`), but an
  import it cannot tie to an indexed file is treated as external, and the
  "no importers" hint is evidence, not proof (dynamic loading, entry points).
- `py-tree-sitter` 0.26 can crash CPython 3.14 on files with several hundred
  top-level statements (an upstream bug; reproducible without Pedacito). CI
  covers 3.11–3.13.
- `run` and test commands are sandboxed where bubblewrap/sandbox-exec work (not
  on Windows, not where user namespaces are disabled -- you get a notice, not a
  silent downgrade); `browser`, `fetch` and MCP are guarded (scrubbed
  environment, protected paths, host allowlists, trust prompts), not sandboxed.
- `rename` is mechanical and conservative: it follows imports, not types, so a
  method renamed on a class used through another name is reported as "not
  changed" rather than guessed. Several edit blocks across files are applied
  per file unless `atomic_edits = true` (the snapshot and `pedacito restore` are
  the safety net either way).
- Languages: Python by AST; JS/TS, Ruby, Go, Rust, Java, C, C++, C#, PHP,
  Kotlin, Swift, Scala and Lua through tree-sitter (a grammar quirk can drop an
  occasional Swift call edge); everything else is searchable reference text.
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
