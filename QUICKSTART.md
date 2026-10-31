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
- **Hosted API (DeepSeek, OpenAI, ...):** you only need the URL, an API key, and
  a model name.

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

The wizard probes for a running LM Studio or Ollama server, lists its models,
asks which to use, optionally takes a hosted API key, and writes
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
window, and calibrates token math against the server's own tokenizer. The
answers are remembered, so a server without grammar support falls back to
JSON prompting automatically on later runs.

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

---

## More ways to work

**Browser GUI.** `pedacito gui` opens a local page where you open a project,
build the index, check the model, run tasks with the review page inline, ask
`explain` questions, and restore snapshots. Install, configuration, and the
other commands stay in the terminal.

**Interactive session.** `pedacito chat ~/code/anki` keeps the index warm
between tasks (add `--apply` or `--review` for the whole session). Slash
commands: `/profile NAME`, `/model ID`, `/apply on|off`, `/steps N`, `/status`,
`/profiles`, `/help`, `/quit`. Switching profile and re-running a task is the
cheapest way to compare two models.

**New project.** `pedacito new ~/code/my-app -t "a CLI todo manager with a
sqlite backend" --apply` plans a file tree, writes a skeleton, then fills in
each file. Use `--spec FILE` for a longer description, `--review` to sign off
first, and `--plan-only` to stop after the skeleton. It is a dry run without
`--apply`, and the target must be new or empty unless you pass `--force`.

**Autonomous loop.** `pedacito auto . -t "add retry logic to the sync client"`
prints a task checklist; add `--apply` to implement it task by task. Useful
flags: `--verify "pytest -q"` (check and repair each task), `--diagnostics`
(lint/typecheck each task), `--worktree NAME` (work in an isolated git
worktree), `--plan FILE` (use your own checklist), and `--max-tasks N` /
`--max-minutes N` (set limits; there are none by default). A repair round that
makes the check worse is rolled back automatically and the task marked failed,
so an unattended run can't compound damage.

**Optional abilities** (all off by default; set in your config):
`allow_write = true` (create/overwrite/move/delete files), `allow_shell = true`
(run tests and linters), `allow_screenshots = true` and `allow_browser = true`
(let the model see a GUI or web page as text), and — the other direction —
`auto_allowlist = true` (each `auto` task may only write existing files its
title names or its lookups read; creating new files stays allowed). See the
README for details.

## Several models: profiles

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
disable_thinking = true         # skip the <think> block: faster
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

To force a clean rebuild, delete `<project>/.pedacito/` (always safe) or pass
`--reindex` to `ask`/`chat`.

## The commands, briefly

| Command | What it does |
|---|---|
| `pedacito config --check` | Is the server up, and does the model work? |
| `pedacito how "..."` | Ask how to use Pedacito. No project needed. |
| `pedacito files PROJ` | What would be indexed, and why. Free. |
| `pedacito cost PROJ` | What indexing will cost. Free. |
| `pedacito map PROJ` | Print the index. No model needed. |
| `pedacito index PROJ` | Build or refresh the index. |
| `pedacito ask PROJ -t "..."` | One task; dry run by default. Add `--review` or `--apply`. |
| `pedacito explain PROJ -q "..."` | Answer a question about the code. No edits. |
| `pedacito chat PROJ` | Interactive session; the index stays warm. |
| `pedacito new DIR -t "..."` | Start a project from scratch (or `--spec FILE`). |
| `pedacito auto PROJ -t "..."` | Plan a goal, then implement it task by task. |
| `pedacito diagnose` | Run the project's linter/type checker. |
| `pedacito worktree create\|list\|remove NAME` | Manage isolated git worktrees. |
| `pedacito restore PROJ` | Undo the last write. |
| `pedacito gui` | Browser control panel. |

Flags available on most commands: `--exclude 'pattern'`, `--include 'pattern'`,
`--ext .kt`, `--steps N`, `--model ID`, `--url URL`, `--profile NAME`,
`--config FILE`, `--root PATH`.

## Where things live

Per project, under `<project>/.pedacito/`: `index.json` (the map),
`summaries.json` (summary cache), `backups/<stamp>/` (pre-write snapshots),
`sessions/<stamp>.md` (what was asked, looked up, and changed — open this when an
edit turns out wrong later), `reviews/` (saved review pages), and `log.txt`
(event trace). Deleting the folder is always safe; it only costs a re-index.

Your settings live in `~/.config/pedacito/config.toml` (`%APPDATA%\pedacito\`
on Windows); `pedacito config` shows every resolved value and which file or
profile set it. Settings resolve in layers: defaults → that file → a project's
`.pedacito.toml` → `--config FILE` → `--profile` → command-line flags. Next to
the config file, the GUI remembers recent projects in `gui_recent.json` and
`server_modes.json` stores what each server + model supports (JSON modes,
tokenizer rate; refreshed by `--doctor`). Both are safe to delete.
