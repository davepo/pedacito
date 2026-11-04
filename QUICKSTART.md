# Pedacito Quick Start

Pedacito lets a small local LLM (4B–27B and similar) work on a codebase bigger
than its context window. It builds a searchable index of your project, then
lets the model fetch only the pieces it needs to answer questions or propose
edits. See the [README](README.md) for background and the full reference.

Commands are identical on Windows, Linux, and macOS except where noted. Every
command uses `pedacito`; `pdc` is an identical shorter alias, and
`python -m pedacito` (`py -m pedacito` on Windows) works if neither is on your
`PATH`.

---

## 1. Start your model server

Pedacito talks to any server with an OpenAI-compatible `/v1` API.

- **LM Studio:** load a model and start the server (Developer tab). Default
  port `1234`. If Pedacito runs on a different machine, enable **Serve on Local
  Network**.
- **Ollama:** `ollama serve`, then `ollama pull <model>`. Default port `11434`.
  For a different machine, set `OLLAMA_HOST=0.0.0.0` before starting it.
- **Lemonade:** start the server and load a model; use the base URL it shows.
- **llama.cpp:** run the server; use the URL it prints (often `:8080`).
- **Hosted API (DeepSeek, OpenAI, ...):** you only need the URL, an API key, and
  a model name.

Those ports are defaults, not rules: any port works, and the scan identifies a
server by its replies — so two servers can run on one machine (for example
Lemonade on 11434 next to LM Studio on 1234).

## 2. Install

Requires Python 3.11+.

```bash
cd /path/to/pedacito
pip install -e .            # Windows: py -m pip install -e .
```

`-e` (editable) means a later `git pull` takes effect without reinstalling.

## 3. Configure and check

```bash
pedacito config --setup
```

The wizard finds running local model servers (LM Studio, Ollama, Lemonade,
llama.cpp) on any port, lists their models, and offers to write a profile for
every model and pick a default. For a server on another machine or a hosted
API, run `pedacito models scan --url <url>` (with `--api-key-env VAR` and
`--match PATTERN` for a hosted catalogue). It writes
`~/.config/pedacito/config.toml` (`%APPDATA%\pedacito\config.toml` on Windows).
Prefer to edit by hand? `pedacito config --init` writes a commented starter.
The essentials:

```toml
[universal]
vram_gb = 8                             # GPU VRAM on the server machine, in GB

[profiles.default]
base_url = "http://localhost:1234/v1"   # or :11434/v1 for Ollama
model    = "qwen3.5-9b"                 # exact id/name the server reports
```

`vram_gb` is needed for local servers only. Token budgets need no tuning:
Pedacito asks the server for its real context window and fits every limit to it.

Then verify:

```bash
pedacito config --check
```

```
Connecting to http://localhost:1234/v1 ...
  reachable. Models available: qwen3.5-9b
  'qwen3.5-9b' is listed. Testing that it actually generates...
  generation works. You're good.
```

If it fails, there are two cases. **Not reachable** means the server isn't
running, the port is wrong, or (remote setup) it isn't listening on the network.
**Reachable but the model isn't listed or won't generate** means `model =`
doesn't match the server's name for it (copy it from the `--check` output), or
the server couldn't load it (usually not enough free VRAM). Run `--check`
whenever something stops working.

Once per server, also run:

```bash
pedacito config --doctor
```

It repeats the check, then probes which structured-output modes your server
really supports, prints a recommended `json_mode`, reports the real context
window, calibrates token math against the server's own tokenizer, and reports
whether tree-sitter parsing is active. The answers are remembered, so a server
without grammar support falls back to JSON prompting automatically on later
runs.

Then, once per model:

```bash
pedacito calibrate
```

It spends about a minute measuring the model — does it pick the right lookup,
emit valid edit blocks, resist a planted prompt injection, admit what it has
not read, find a needle in a long prompt, answer without its whole budget
going to hidden reasoning? — and saves a model card. Settings you left at
their defaults (prompt pack, answer mode, verifier, best-of-N, the thinking
switch) then follow the card; `pedacito config` shows it.

Stuck on anything else? `pedacito how "how do I ...?"` answers from these docs
and the source, using your config. It needs no project or index.

---

## 4. See what would be indexed

```bash
pedacito files ~/code/anki
```

```
  created ~/code/anki/.pedacitoignore
  created ~/code/anki/.gitignore with .pedacito/
  ref   data/intervals.csv
  code  src/scheduler.py
  skip  deprecated/   [ignored directory, 1 file(s)]
```

No model call. If something is being indexed that shouldn't be (old scripts,
vendored dependencies, build output), add it to the `.pedacitoignore` this
created (gitignore syntax, forward slashes on every OS):

```
deprecated/
*_old.py
```

Re-run `pedacito files` to confirm. Trimming generously leaves more map space
for the code you care about.

## 5. Check the cost, then index

```bash
pedacito cost ~/code/anki        # free: estimated LLM calls and time
pedacito index ~/code/anki
```

On a large project, `cost` tells you whether the first index takes 2 minutes or
20. It happens once; afterward only changed code is re-summarised. You never
need to run `index` again explicitly, since `ask` and `chat` refresh it.
`pedacito map ~/code/anki` prints the index with no model, a quick sanity check.

Curious what the model will actually see? `pedacito map ~/code/anki -t "make the
leech threshold configurable"` prints the map a task would get: on a big project,
the likeliest files first (each with a `why listed:` line) over a directory
skeleton.

## 6. Ask for something (nothing is written)

```bash
pedacito ask ~/code/anki -t "make the leech threshold configurable"
```

```
--- looking things up ---
  step 1: read src/scheduler.py L11-21   (see the scheduler)
  step 2: done                           (have enough)
--- answering ---
...
[ok ] src/scheduler.py: 2 block(s) would apply
--- a/src/scheduler.py
+++ b/src/scheduler.py
```

**Without `--apply` or `--review`, nothing is written**: you get a diff and
decide. Stay in this mode for your first dozen tasks with a new model or project
to learn how reliably its edits anchor.

Every task ends with a usage line — calls, tokens in/out, the server's
prompt-cache hit ratio, and cost if you set prices — and before any edit is
written Pedacito checks it for the usual small-model mistakes (editing code it
never read, a `TODO` instead of working code, a removed function whose callers
were left behind). Findings are printed; errors trigger one corrected re-ask.

To ask a question instead of changing code, use `explain`:

```bash
pedacito explain ~/code/anki -q "how does the scheduler pick the next card?"
```

## 7. Review or apply

**Review page** — an offline page with your file on the left and the proposed
version on the right:

```bash
pedacito ask ~/code/anki -t "make the leech threshold configurable" --review
```

To write, tick the attestation box, click **Sign off & write**, then confirm.
Enter never writes. If you edit the file while the review is open, the write is
refused instead of overwriting your change. **Discard** changes nothing. Set
`review = true` in your config to make this the default (`--no-review` skips it
for one run).

**Direct apply** — write immediately:

```bash
pedacito ask ~/code/anki -t "make the leech threshold configurable" --apply
```

Either way, originals are snapshotted first.

## 8. Undo

```bash
pedacito restore ~/code/anki --list
pedacito restore ~/code/anki --dry-run      # what the newest snapshot would restore
pedacito restore ~/code/anki                # restore it, byte-for-byte
pedacito restore ~/code/anki --snapshot 20260830-005535
```

Inside a task the model can do the same to its own writes with an `undo`
action (needs `allow_write` and backups, on by default): this task's snapshots
come back, files it created go away.

---

## More ways to work

**Browser GUI.** `pedacito gui [PROJECT]` opens a local page for everything
above: open a project and profile, run a **Task** (with the diff shown old vs
updated, accept/reject per file), **Explain** questions, **Auto** (plan, then
implement), or start a **New** project from scratch (including *complex*).
Attach a task or spec file, steer or stop a running job, watch the live
terminal, and edit your settings and profiles in **Settings** — no TOML by hand
(it also builds your first config if you have none). Full tour in the README.

**Interactive session.** `pedacito chat ~/code/anki` keeps the index warm
between tasks (add `--apply` or `--review` for the whole session). Slash
commands: `/profile NAME`, `/model ID`, `/apply on|off`, `/steps N`, `/status`,
`/cost`, `/memory`, `/skills`, `/profiles`, `/help`, `/quit`. Ctrl-C cancels
the running task and keeps the session. Switching profile and re-running a task
is the cheapest way to compare two models.

**Memory and skills.** The model can `remember` a fact about the project (a
build command, a convention, a trap) into `.pedacito/memory.md`; you can edit
it, or `pedacito memory ~/code/anki --add "..."`. Built-in skills
(`pedacito skills`) give it recipes like *add-a-test* and *fix-a-failing-test*;
write your own in `.pedacito/skills/NAME/SKILL.md`.

**Measure it.** `pedacito eval` runs eight fixture tasks (Python, TypeScript,
Go, with a planted prompt injection) and reports locate rate, first-apply
success, tokens, cache hits and a failure taxonomy — change a setting, run it
again, compare.

**New project.** `pedacito new ~/code/my-app -t "a CLI todo manager with a
sqlite backend" --apply` plans a file tree, writes a skeleton, then fills in
each file. Use `--spec FILE` for a longer description, `--review` to sign off
first, and `--plan-only` to stop after the skeleton. It is a dry run without
`--apply`, and the target must be new or empty unless you pass `--force`.

**Autonomous loop.** `pedacito auto . -t "add retry logic to the sync client"`
prints a task checklist; add `--apply` to implement it task by task. Useful
flags: `--verify "pytest -q"` (check and repair each task), `--diagnostics`
(lint/typecheck each task), `--worktree NAME` (work in an isolated git
worktree), `--plan FILE` (use your own checklist), `--max-tasks N` /
`--max-minutes N` (set limits; there are none by default), and `--strict`
(exit 1 when an applied task left a failed edit block or impacted tests
failing — for CI). A repair round that
makes the check worse is rolled back automatically and the task marked failed,
so an unattended run can't compound damage.

**Several changes in one task.** `pedacito ask . -t "..." --plan` runs each
requirement in its own fresh context (bullets split with no model call) and
re-plans once if a step fails. Weak models get this automatically after
`pedacito calibrate` (which also tests whether your model does better with
native tool calls), and any task naming five or more files gets it too — one
window rarely holds that many lookups. While `chat` is working, type a line to steer it ("it's in
sync.py"); it may also ask you one short question when a task is truly
ambiguous. `--jsonl` turns a run into one JSON event per line for scripts and
editors; `--role planner=big` sends one job to another profile for a run.

**Optional abilities** (all off by default; set in your config):
`allow_write = true` (create/overwrite/move/delete files, a mechanical
`rename` of a symbol across the files that import it, and an `undo` that
reverts this task's own writes), `allow_shell = true`
(run tests and linters; add `verify_tests = true` to run the tests an edit can
break), `allow_screenshots = true` and `allow_browser = true` (let the model
see a GUI or web page as text), `allow_subagents = true` (a read-only helper
for long investigations), `allow_fetch = true` (read allow-listed web pages),
and — the other direction — `auto_allowlist = true` (each `auto` task may only
write existing files its title names or its lookups read; creating new files
stays allowed). See the README for details.

**Sandboxed commands.** Where your OS has bubblewrap (Linux) or sandbox-exec
(macOS), `run` and post-edit tests execute in a sandbox: writes only to the
project and a temp dir, `.git/` read-only, credentials hidden, **no network**.
If a test needs the network, set `sandbox_network = true`; `sandbox = false`
turns it off; on Windows (no backend) commands run as before after one notice.
`pedacito config --doctor` shows whether it is active.

**Undo, including new files.** `pedacito restore . --delete-created` also
removes the files that run created. `pedacito ask . -t "..." --record` saves the
model's responses and `--replay FILE` re-runs the harness against them with no
model, which is how to test a harness change.

**Safe by default.** Commands the model runs get a scrubbed environment (no
ambient API keys), `.env`/key files are never indexed, tool output is fenced as
data, the model can't write `.git/`, `.pedacito/` or the `auto` plan file, and a
cloned project's `.pedacito.toml` cannot set hooks, MCP servers or verify
commands until you read it and run `pedacito trust`.

## Several models: profiles

For VRAM-based model selection, thinking/reasoning advice, and role routing, see
the [model guide](MODEL_GUIDE.md). `pedacito models scan` finds a server and
writes one profile per model; `pedacito profiles --recommend` shows measured
scores and per-role suggestions.

```toml
[profiles.tiny]
model = "qwen2.5-coder:1.5b"
tier  = "tiny"           # ~4k window budgets for models of 4B and under

[profiles.small]
model = "qwen2.5-coder:7b"
tier  = "small"          # tight token budgets for an 8k window

[profiles.big]
model = "some-27b-model"
tier  = "large"

[profiles.api]
base_url = "https://api.deepseek.com/v1"
api_key  = "sk-..."
model    = "deepseek-chat"      # remote hosts are detected automatically

[profiles.reasoning]
model = "qwen3:32b"
disable_thinking = true         # family-aware: faster, less reasoning
```

```bash
pedacito ask anki -t "..." --profile big
pedacito config --profile big        # show what it resolves to, and why
```

Pick a profile with `default_profile =` in your config, the `PEDACITO_PROFILE`
environment variable, `--profile`, or `/profile` inside `chat` (increasing
precedence). `tier` can usually stay unset: Pedacito guesses it from the model
name (4B and under → `tiny`, up to 10B → `small`, up to 20B → `medium`,
bigger → `large`). Model names differ by server: LM Studio and Lemonade use
the id that `pedacito config --check` prints; Ollama uses the name from
`ollama list` (e.g. `qwen2.5-coder:14b`). A mismatch is the most common setup
error.

Switching models makes the server load the new one, which can take up to a
minute or two; Pedacito waits automatically (`load_wait_seconds`, default 240).

## Writing good tasks

The scaffolding gets a small model to the right 300 lines out of 12,000; it
doesn't make the model smarter once it's there.

- **Works well (narrow, local):** "add retry-with-backoff to the sync client",
  "is_leech has an off-by-one, fix it", "add type hints to Scheduler".
- **Works badly (broad, structural):** "refactor the scheduler", "modernise
  this codebase", "add tests" (nothing to anchor on).

Name the file or symbol when you know it: "fix the off-by-one in is_leech" beats
"fix the leech bug". Questions work too; if no code change is needed you get an
answer and no edits.

## When something looks wrong

| Symptom | Cause and fix |
|---|---|
| `Cannot reach the server` | Server not running, wrong port, or (remote) not listening on the network. See step 1. |
| Model listed but won't load or generate | Usually VRAM: another model is resident. Free it or lower the context length. |
| `search block not found` | The model copied line-number prefixes or the file shifted under it. The failure shows the closest matching text with its current line numbers, and the model retries (twice) re-anchored on it — often pinning a `LINES a-b` range in the block header. If it keeps failing, narrow the task. |
| Model says a function doesn't exist | It isn't indexed. Check `pedacito files`; an ignore rule may be too broad. |
| `Model would not produce valid JSON` | Usually a truncated reply. Pedacito retries with more room; if it persists, raise `max_tokens_step`. A thinking model that spends its whole budget reasoning returns no JSON; turn thinking off. |
| Every request fails or "kept hitting limits" | The request exceeded the server's context window. Check the `[auto-fit]` line printed at startup. If the server reports a stale default window, set `context_tokens` (and `k_cache_quant`/`v_cache_quant` for a quantized cache). |
| Answer is nonsense on a local server | Some servers (Ollama) silently truncate an oversized prompt. Raise the server-side context length. |
| Model seems lost, reads the wrong things | Name the file in the task. If you set `--steps`/`max_steps`, raise it. |
| Answers turn vague | The map may have switched from a full listing to per-file cards. `pedacito map PROJ --tier auto` shows which. |
| Stale or misleading summaries | A docstring replaces a generated summary. Set `summarise_documented = true` and re-index. |
| No review page appeared | Nothing anchored, so there was nothing to review. Check the printed failures. |
| Everything is slow | Throughput has likely collapsed because the model spilled out of VRAM. |

To force a clean rebuild, run `pedacito reset PROJ --reindex` (or pass
`--reindex` to `ask`/`chat`). `reset` wipes the project's `.pedacito/` runtime
state without touching your code or `.pedacito.toml`; `--dry-run` previews it,
`--keep-memory` keeps remembered facts, and `--keep-config` keeps authored
rules/skills/prompts/instructions. A plain `--reindex` is the lighter option
when only the map is stale.

## The commands, briefly

| Command | What it does |
|---|---|
| `pedacito config --check` | Is the server up, and does the model work? |
| `pedacito how "..."` | Ask how to use Pedacito. No project needed. |
| `pedacito files PROJ` | What would be indexed, and why. Free. |
| `pedacito cost PROJ` | What indexing will cost. Free. |
| `pedacito map PROJ [-t "task"]` | Print the index (ranked for a task). No model needed. |
| `pedacito calibrate` | Measure the model; save a model card that tunes defaults. |
| `pedacito eval` | Score the harness on built-in fixture repos. |
| `pedacito tests PROJ --files F` | Detected test command; tests an edit to F can break. |
| `pedacito webcheck PROJ` | Static contract checks for web files: DOM ids, ESM in classic scripts, duplicate globals, missing script srcs. |
| `pedacito memory PROJ` | List / add / forget remembered facts. |
| `pedacito skills PROJ` | List the available skills. |
| `pedacito prune PROJ` | Delete old sessions/backups/events; trim the log. |
| `pedacito reset PROJ [--reindex]` | Wipe `.pedacito/` runtime state; code and config stay. |
| `pedacito trust PROJ` | Approve a project's `.pedacito.toml`. |
| `pedacito index PROJ` | Build or refresh the index. |
| `pedacito ask PROJ -t "..."` | One task; dry run by default. Add `--review` or `--apply`. |
| `pedacito explain PROJ -q "..."` | Answer a question about the code. No edits. |
| `pedacito chat PROJ` | Interactive session; the index stays warm. |
| `pedacito new DIR -t "..."` | Start a project from scratch (or `--spec FILE`). |
| `pedacito auto PROJ -t "..."` | Plan a goal, then implement it task by task. |
| `pedacito diagnose` | Run the project's linter/type checker. |
| `pedacito worktree create\|list\|remove NAME` | Manage isolated git worktrees. |
| `pedacito restore PROJ` | Undo the last write. |
| `pedacito gui [PROJECT]` | Browser control panel: tasks, explain, auto, new, review, settings. |

Flags available on most commands: `--exclude 'pattern'`, `--include 'pattern'`,
`--ext .kt`, `--steps N`, `--model ID`, `--url URL`, `--profile NAME`,
`--config FILE`, `--root PATH`.

## Where things live

Per project, under `<project>/.pedacito/`: `index.json` (the map),
`summaries.json` (summary cache), `backups/<stamp>/` (pre-write snapshots),
`sessions/<stamp>.md` (what was asked, looked up, and changed — open this when an
edit turns out wrong later), `events/*.jsonl` (one JSON line per step, model
call, fold and task: tokens, cache hits, latency), `memory.md`, `reviews/`
(saved review pages), and `log.txt` (event trace). Deleting the folder is always
safe; it only costs a re-index. `pedacito reset PROJ` clears it for you (and
`--reindex` rebuilds the map immediately), while `pedacito prune` trims the old
files.

Your settings live in `~/.config/pedacito/config.toml` (`%APPDATA%\pedacito\`
on Windows); `pedacito config` shows every resolved value and which file or
profile set it. Settings resolve in layers: defaults → that file → a project's
`.pedacito.toml` → `--config FILE` → `--profile` → command-line flags. Next to
the config file, the GUI remembers recent projects in `gui_recent.json` and
`server_modes.json` stores what each server + model supports (JSON modes,
tokenizer rate; refreshed by `--doctor`), and `model_cards.json` holds the
`pedacito calibrate` results (they expire after 30 days). All are safe to
delete.
