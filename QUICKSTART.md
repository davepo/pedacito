# Pedacito quickstart

Pedacito lets a small local model work on a codebase that is far bigger than its
context window. It builds an index of your project, then lets the model fetch
only the pieces it needs to answer a question or propose an edit. For the
background, see the [README](README.md). For how the small-model techniques
work, see [TECHNIQUES.md](TECHNIQUES.md). For choosing a model, see the
[model guide](MODEL_GUIDE.md).

There are two ways in, and they run the same engine and read the same config:

- **[Part A: the GUI](#part-a-the-gui).** A control panel in your browser. Best if
  you want to see diffs, plans, and settings side by side.
- **[Part B: the CLI](#part-b-the-cli).** Commands in a terminal. Best for
  scripts, editors, and quick one-off tasks.

Every command works the same on Windows, Linux, and macOS. Use `pedacito`, the
shorter alias `pdc`, or `python -m pedacito` (`py -m pedacito` on Windows) if
neither is on your `PATH`.

---

## Before you start

### 1. A model server

Pedacito talks to any server with an OpenAI-compatible `/v1` API.

- **LM Studio:** load a model, then start the server from the Developer tab. The
  default port is `1234`. To use it from another machine, enable **Serve on Local
  Network**.
- **Ollama:** run `ollama serve`, then `ollama pull <model>`. The default port is
  `11434`. For another machine, set `OLLAMA_HOST=0.0.0.0` before starting it.
- **Lemonade Server:** start the server and load a model. Use the base URL it
  shows.
- **llama.cpp:** start `llama-server` with your model. Use the URL it prints.
- **A hosted API** (Claude, ChatGPT, DeepSeek, Qwen, Gemini, OpenRouter): you need
  an API key. Put it in an environment variable, such as `DEEPSEEK_API_KEY`, and
  Pedacito reads it from there so the key stays out of your config file.

Ports are defaults, not rules. The scanner identifies each server by its replies,
so two servers can run on one machine.

### 2. Install

Requires Python 3.11 or newer.

```bash
cd /path/to/pedacito
pip install -e .            # Windows: py -m pip install -e .
```

That one command installs everything the core needs: the tree-sitter AST chunker
and its grammars (the exact syntax check before every edit), plus `pytest` and
`ruff` for verification, and `requests` for the model connection. Nothing else
is needed for the default setup. `pip install -r requirements.txt` installs the
same core set without installing Pedacito itself.

`-e` means "editable": a later `git pull` takes effect without reinstalling.
Optional extras: `pip install -e ".[full]"` adds screenshots and the headless
browser, or pick one with `pip install -e ".[vision]"` (screenshots) or
`pip install -e ".[browser]"` for the headless browser (then run
`python -m playwright install chromium`).

---

## Part A: the GUI

### A1. Open the control panel

```bash
pedacito gui ~/code/your-project
```

Pedacito starts a local server on `127.0.0.1` and opens it in your browser. The
address it prints is the one to use if the tab does not open by itself. Pass
`--port 8787` to pin the port, `--no-open` to skip launching a browser, and
`--profile NAME` to pick a model profile for the project.

If you have no config file yet, a first-run form appears. Its **Find servers**
button looks for LM Studio and Ollama on this machine.

### A2. Connect a model

You can add models two ways. Both write profiles to your config, so you only do
this once per model.

**From a local server.** Open **Settings** (the gear) and go to **Model
discovery**. Click **Scan local servers**. Pedacito asks how much GPU memory the
machine running the server has, in GB. Then tick only the models you want. The
name box lets you rename a profile. Click **Import Selected**. Each model becomes a
profile named after it, and one is set as the default. Each profile in the list
has a **Calibrate** button that runs a quick measurement of that model.

**From a hosted provider.** In the same tab, click **+ Add Hosted Provider**.
Choose a preset: **Claude (Anthropic)**, **ChatGPT (OpenAI)**, **DeepSeek**,
**Qwen (DashScope)**, **Google Gemini**, or **OpenRouter**. Provide the key, or
name the environment variable that holds it. Pedacito connects to the provider,
lists its models, and imports the ones you tick, one profile each.

Back in the top bar, the profile picker switches the model for the next task.
Its status light shows whether the server is reachable. The **no-think** or
**thinking** chip next to it shows whether reasoning is allowed for that profile.

### A3. Run the Auto-Tune wizard

Settings are the numbers that decide how much of a model's context Pedacito
uses for reads, steps, and verification. The wizard sets them from your
hardware, not from guesswork.

1. Open **Settings → Auto-Tune**.
2. Answer four questions: your hardware, the model class, the size of your
   codebase, and what you care about most (speed or thoroughness).
3. Read the preview. It lists each recommended budget and the reason for it.
   Nothing is written yet, and the preview uses no tokens.
4. Click **Apply Recommended Universal Settings**. The previous file is kept as
   `config.toml.bak`.

You can run the wizard again whenever you change your GPU or your model.

### A4. Turn thinking on or off for one profile

Some models reason before they answer. That helps on hard problems but can use up
the answer budget on a small model. Set it per profile:

1. Open **Settings → Profiles** and select the profile.
2. Under **Reasoning & Execution Overrides**, set `disable_thinking`.
3. Save. The chip in the top bar changes from **thinking** to **no-think**.

For a model that needs a prompt switch instead, set `thinking_suffix`, for
example `/no_think`. [MODEL_GUIDE.md](MODEL_GUIDE.md#thinking-per-profile)
explains when each setting helps.

### A5. Ask for a change (Task/Edit)

1. Pick your profile in the top bar, and select the project if it is not open.
2. Choose **Task/Edit** and describe the change. Be specific, and name a file or
   function when you know one. For example: `make the retry count in
   client.py configurable`.
3. Check the options next to the box:
   - **Writing:** *Review before writing* (the default) or *Write directly
     (backup first)*.
   - **Planning:** *Automatic* splits a multi-part request into steps. *Always
     plan* and *Never* override it.
   - **Max lookups:** a cap on how many reads the model may make. `0` uses the
     config value.
4. Click **Run**. The **Engine V2** panel shows the current phase, the requirement
   ledger (one line per clause of your request, from unlocated to verified), and
   recent lookups.
5. When the run finishes, the **Changes** panel shows each file's old code and
   new code side by side, with the changed words highlighted. You can accept or
   reject the whole set, untick individual files, or click **Edit updated** to
   change the new version by hand before accepting.

Pedacito backs up each file before it writes. If a file changed on disk after
the preview, the write is refused, so your own edits are never overwritten.

**Short on detail?** Type a one-line idea and click **✦ Refine Prompt** above the
composer. Pedacito replaces the line with a full spec: an objective, the files to
create or modify (using real paths from your project), interface contracts,
requirements and edge cases, and verification. Read it, edit it if you like, then
click **Run**. **↶ Undo Refine** puts your original text back. Drag the top edge of
the composer to make it taller while you read.

### A6. Ask a question (Ask)

Choose **Ask** to ask how the code works. The answer streams in and nothing is
written. Use it to understand a module before you change it.

### A7. Run an autonomous plan (Auto)

Choose **Auto**, then describe a goal, such as `add CSV export to the report
page, with tests`.

- Tick **Plan only** under the options to receive a checklist first. You can
  edit it in the **Plan** panel, then click **Run this plan**.
- Without **Plan only**, Pedacito plans, then applies and verifies each task.
  Options include a verify command (for example `pytest -q`), repair rounds, a
  cap on tasks and minutes, a separate git worktree, and keep-going after a
  failed task.

Auto writes files and runs commands without asking for each one. Starting an
Auto run is your approval of those tools for that run, so use **Plan only** until
you trust the project. A repair round that makes things worse is rolled back
automatically.

Each step is also checked before the next one starts. Pedacito compares every
function and class signature in the project with the calls and imports across
files, and runs a fast check for undefined names, so a step that breaks a name is
caught before its tests run. If a step still fails after its repair rounds, the
green ratchet rolls it back to the last passing state. The Activity log then
shows `ROLLED BACK`, with the files that were restored or deleted.

### A8. Start a new project (New)

Choose **New**, describe the project, and choose a folder. You can preview the
file tree, build it, build and review it, or stop after the skeleton. The
*complex* option hands the implementation to the autonomous loop.

### A9. Ask how to do something (How ?)

Choose **How ?** to ask where a setting or feature lives, or how to run something
from the CLI. The answer comes with copyable commands and **Open in Settings**
buttons. How answers stay out of your coding thread, and they work even before a
model is configured.

### A10. Compare models with the benchmarks

Click **▦ Benchmarks** in the top bar. Choose a profile, and optionally a folder
for the output, then click **Run Benchmark**. Each run builds one of six small
projects with the normal Pedacito loop, then grades it with a hidden holdout
rubric. The full report is `BENCHMARK_REPORT.md` in the new project folder. A
run can take several minutes. [MODEL_GUIDE.md](MODEL_GUIDE.md#compare-models-with-the-diagnostic-benchmarks)
describes the six projects and how to read the results. The pelican benchmark
also draws the SVG the model produced, inline in the scorecard.

### A11. Undo a write

In the right-hand panel, open the **Backups** tab. It lists the snapshots from
Task/Edit and Auto writes. Choose one and click **Restore**. Files that changed
since the snapshot are overwritten, and files created since then stay on disk.

### A12. Approve or deny a permission request

Sometimes the model asks for something you have not allowed yet: web access, a
package install, a shell command that is not on the allowlist, or a sensitive
file such as a `.env`. The run pauses, and a **Permission requested** dialog
shows what it wants, why, and the exact target.

- **Allow Once** lets this one request through.
- **Allow for This Run** stops the prompts for that kind of request until the run
  ends.
- **Always Allow** stops them from now on. For web access it turns on
  `enable_web`, and for shell commands it turns on `allow_shell`.
- **Deny & Redirect** refuses the request, and the model carries on with the local
  project files.

If you do not answer, the request is denied when the dialog times out or you press
Stop. A denial never crashes the run. The CLI has no dialog, so there a gated tool
reports that it is disabled in the config.

### A13. Turn on web access (optional)

Web access is off by default. Two built-in tools use it: `fetch_url` reads a web
page, and `web_search` looks up documentation. To allow them without asking each
time, open **⚙ Settings → All settings**, search for `enable_web`, set it to on
for every profile, and save. Or leave it off and approve each call in the
permission dialog. Loopback, link-local and private-network addresses are refused
unless you list them in `fetch_allowed_hosts`.

### A14. Other things to know

- **Settings** edits your user config or the project's `.pedacito.toml`. Your
  comments survive, and every edit is checked before it is saved. The **Raw
  file / reset** tab shows the file as written.
- **Themes:** pick a color scheme from the theme menu in the top bar. There are
  nine. The choice is saved to your config as `gui_theme`.
- **Attach** a text file, or drag it into the chat box, to use it as the task, a
  question, or a spec.
- **Steer:** while a task runs, type a note to redirect the model. You can also
  answer its clarifying questions inline, or click **Stop**.
- **Terminal:** the bottom panel shows everything Pedacito prints, live.

---

## Part B: the CLI

### B1. Check the connection

First-time setup. This finds local servers, lists their models, and writes a
profile for each one:

```bash
pedacito config --setup
```

Then verify the connection:

```bash
pedacito config --check
```

```
Connecting to http://localhost:1234/v1 ...
  reachable. Models available: qwen3.5-9b
  'qwen3.5-9b' is listed. Testing that it actually generates...
  generation works. You're good.
```

If it fails, there are two cases. **Not reachable** means the server is not
running, the port is wrong, or (for a remote server) it is not listening on the
network. **Reachable, but the model is not listed or will not generate** means
`model =` does not match the name the server uses (copy it from the `--check`
output), or the server could not load the model, usually because there is not
enough free VRAM.

Once per server, also run:

```bash
pedacito config --doctor
```

It repeats the check, then probes which structured-output modes the server
supports, recommends a `json_mode`, reports the real context window, and checks
whether tree-sitter is active. The answers are remembered for a week.

### B2. Find models and measure them

```bash
pedacito models scan
```

This finds local servers on any port, lets you pick models, writes one profile
per model, and measures each one with a quick calibration. The calibration
probes tool choice, JSON output, edit format, resistance to planted prompt
injections, honesty about what was read, long-context retrieval, and whether
hidden reasoning eats the answer budget. The result is saved as a model card.

Useful flags:

- `--match 'qwen*'` only considers models whose id matches the pattern.
- `--loaded` only considers models the server has loaded.
- `--calibrate none|quick|full` chooses how much measuring to do.
- `-y` selects every matching model without asking.

For a hosted provider, name the URL and the environment variable that holds the
key. `--match` is required, because hosted catalogues can hold hundreds of models:

```bash
pedacito models scan --url https://api.deepseek.com/v1 \
  --api-key-env DEEPSEEK_API_KEY --match 'deepseek-chat'
```

To measure a model again later, run `pedacito calibrate`. To see the results side
by side, run `pedacito profiles`, or `pedacito profiles --recommend` for suggested
roles.

### B3. See what would be indexed

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

This makes no model call. If something should not be indexed (old scripts,
vendored code, build output), add it to `.pedacitoignore` in gitignore syntax,
using forward slashes on every OS:

```
deprecated/
*_old.py
```

Run `pedacito files` again to confirm. Leaving out what you do not need gives the
model more room for the code you care about.

### B4. Check the cost, then index

```bash
pedacito cost ~/code/anki        # free: estimated model calls and time
pedacito index ~/code/anki
```

`cost` tells you whether the first index takes two minutes or twenty. It happens
once. After that, only changed code is summarized again. You do not have to run
`index` yourself, because `ask` and `chat` refresh it.

To see the map a task would get, with no model call:

```bash
pedacito map ~/code/anki -t "make the leech threshold configurable"
```

On a large project, the files most likely to matter come first, each with a
`why listed:` line, above a skeleton of the rest.

### B5. Ask for a change (nothing is written)

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

**Without `--apply` or `--review`, nothing is written.** You get a diff, and you
decide. Stay in this mode for your first dozen tasks with a new model or project,
so you learn how reliably its edits land.

Before any edit is written, Pedacito checks it for common small-model mistakes:
editing code it never read, leaving a `TODO` instead of working code, or removing
a function that still has callers. Errors trigger one corrected retry. Every task
ends with a usage line that shows calls, tokens in and out, the server's
prompt-cache hit ratio, and the cost if you set prices.

### B6. Review, then apply

Review first. This opens an offline page with your current file on the left and
the proposed version on the right:

```bash
pedacito ask ~/code/anki -t "make the leech threshold configurable" --review
```

To write, tick the attestation box, click **Sign off & write**, then confirm.
Pressing Enter never writes. If you edit the file while the review is open, the
write is refused instead of overwriting your change. **Discard** changes nothing.
Set `review = true` in your config to make review the default, and use
`--no-review` to skip it for one run.

To write immediately:

```bash
pedacito ask ~/code/anki -t "make the leech threshold configurable" --apply
```

Either way, the originals are snapshotted first.

### B7. Undo

```bash
pedacito restore ~/code/anki --list
pedacito restore ~/code/anki --dry-run      # what the newest snapshot would restore
pedacito restore ~/code/anki                # restore it, byte for byte
pedacito restore ~/code/anki --snapshot 20260830-005535
pedacito restore ~/code/anki --delete-created   # also remove files the run created
```

During a task, the model can undo its own writes with the `undo` action (this
needs `allow_write`, and backups are on by default). That reverts this task's
snapshots and removes the files it created.

### B8. Ask a question (no edits)

```bash
pedacito explain ~/code/anki -q "how does the scheduler pick the next card?"
```

`explain` never builds an index, so run `pedacito index` first.

### B9. Run an autonomous plan

```bash
pedacito auto . -t "add retry logic to the sync client"          # plan only
pedacito auto . -t "add retry logic to the sync client" --apply  # plan, then implement
```

The plan is a markdown checklist in `PLAN.md`. You can edit it before you apply
it, and a run can be resumed. Useful flags:

- `--verify "pytest -q"` checks and repairs each task.
- `--diagnostics` runs the project's linter and type checker on each task.
- `--worktree NAME` works in an isolated git worktree and leaves your checkout
  alone.
- `--plan FILE` uses your own checklist instead of generating one.
- `--max-tasks N`, `--max-minutes N`, or `--steps N` set limits. There are none by
  default.
- `--strict` exits with status 1 if a task applied edits but left a failed edit or
  failing tests, which is useful in CI.
- `--refine` expands a one-line goal into a full spec before planning.

A repair round that makes the check worse is rolled back automatically, and that
task is marked failed. An unattended run cannot compound the damage. A step that
still fails after its repair rounds is rolled back to the last passing checkpoint
(`auto_green_ratchet`, on by default), so later steps never build on it.

### B10. Interactive session

```bash
pedacito chat ~/code/anki            # add --apply or --review for the whole session
```

The index stays warm between tasks. Type `/help` for the commands. The most
useful ones are `/profile NAME` and `/model ID`. Ctrl-C cancels the running
task and keeps the session open. Switching profiles and running the same task
again is the cheapest way to compare two models.

### B11. Benchmarks

The six-project benchmark suite runs from the GUI (the **▦ Benchmarks** button in
Part A10). There is no CLI command for it yet. The CLI measures a model with
`pedacito calibrate` and `pedacito eval`, which are described in the
[model guide](MODEL_GUIDE.md#how-to-measure).

---

## Several models: profiles

Each model you use is a profile. `pedacito models scan` and the GUI create them
for you, but you can also write them yourself:

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
environment variable, `--profile`, or `/profile` in `chat`. The later choices
win. `tier` can usually stay unset: Pedacito guesses it from the model name (4B
and under is `tiny`, up to 10B is `small`, up to 20B is `medium`, and anything
bigger is `large`).

Model names differ by server. LM Studio and Lemonade use the id that
`pedacito config --check` prints. Ollama uses the name from `ollama list`, such as
`qwen2.5-coder:14b`. A mismatch is the most common setup error.

Switching models makes the server load the new one, which can take a minute or
two. Pedacito waits for it (`load_wait_seconds`, default 240).

---

## Writing good tasks

The scaffolding gets a small model to the right 300 lines out of 12,000. It does
not make the model smarter once it is there.

- **Works well (narrow, local):** "add retry-with-backoff to the sync client",
  "`is_leech` has an off-by-one, fix it", "add type hints to `Scheduler`".
- **Works badly (broad, structural):** "refactor the scheduler", "modernise this
  codebase", "add tests" (nothing specific to anchor on).

Name the file or symbol when you know it. "Fix the off-by-one in `is_leech`" beats
"fix the leech bug". Questions work too. If no code change is needed, you get an
answer and no edits.

Several changes in one task work better when split. Use `--plan`, or bullets in
the task text. Each requirement then gets its own fresh context window, and
Pedacito tracks each one until it has evidence that it was done. Tasks that name
five or more files are planned automatically.

A one-line request can also be expanded first. In the GUI, click **✦ Refine
Prompt**. On the command line, add `--refine` to `ask`, `new` or `auto`. The
expanded spec names the real files and the function signatures, so each step has
the same names to work from.

---

## When something looks wrong

| Symptom | Cause and fix |
|---|---|
| `Cannot reach the server` | The server is not running, the port is wrong, or (remote) it is not listening on the network. See "A model server" above. |
| Model is listed but will not load or generate | Usually VRAM. Another model is loaded. Unload it, or lower the context length. |
| `search block not found` | The model copied line-number prefixes, or the file changed under it. The failure shows the closest matching text with current line numbers, and the model retries (twice) anchored on that text. If it keeps failing, narrow the task. |
| Model says a function does not exist | It is not indexed. Check `pedacito files`. An ignore rule may be too broad. |
| `Model would not produce valid JSON` | Usually a truncated reply. Pedacito retries with more room. If it persists, raise `max_tokens_step`. A thinking model that spends its whole budget reasoning returns no JSON, so turn thinking off for that profile. |
| Every request fails, or "kept hitting limits" | The request is bigger than the server's context window. Check the `[auto-fit]` line printed at startup. If the server reports a stale window, set `context_tokens` (and `k_cache_quant` / `v_cache_quant` for a quantized cache). |
| The answer is nonsense on a local server | Some servers (Ollama) silently truncate an oversized prompt. Raise the server's context length. |
| The model seems lost or reads the wrong things | Name the file in the task. If you set `--steps` or `max_steps`, raise it. |
| Answers are vague | The map may have switched from a full listing to per-file cards. `pedacito map PROJ --tier auto` shows which. |
| Stale or misleading summaries | A docstring replaces a generated summary. Set `summarise_documented = true` and re-index. |
| No review page appeared | Nothing anchored, so there was nothing to review. Check the printed failures. |
| Everything is slow | The model has likely spilled out of VRAM. |

To force a clean rebuild, run `pedacito reset PROJ --reindex`, or pass `--reindex`
to `ask` or `chat`. `reset` clears the project's `.pedacito/` state without
touching your code or `.pedacito.toml`. Use `--dry-run` to preview it,
`--keep-memory` to keep remembered facts, and `--keep-config` to keep your rules,
skills, prompts, and instructions. A plain `--reindex` is the lighter option when
only the map is stale.

---

## Commands at a glance

| Command | What it does |
|---|---|
| `pedacito gui [PROJECT]` | Browser control panel: Task/Edit, Ask, Auto, New, How ?, Benchmarks, Settings |
| `pedacito config --setup` | First-time setup: find servers and write profiles |
| `pedacito config --check` | Is the server up, and does the model work? |
| `pedacito config --doctor` | Also probe structured output, context window, tokenizer, and sandbox |
| `pedacito models scan` | Find local servers or a hosted API, pick models, write profiles, measure them |
| `pedacito calibrate` | Measure the model and save a model card that tunes defaults |
| `pedacito profiles [--recommend]` | List profiles and their scores; suggest roles |
| `pedacito how "..."` | Ask how to use Pedacito. No project needed |
| `pedacito files PROJ` | What would be indexed, and why. Free |
| `pedacito cost PROJ` | What indexing will cost. Free |
| `pedacito index PROJ` | Build or refresh the index |
| `pedacito map PROJ [-t "task"]` | Print the index, ranked for a task. No model needed |
| `pedacito ask PROJ -t "..."` | One task. Dry run by default. Add `--review`, `--apply`, or `--refine` |
| `pedacito explain PROJ -q "..."` | Answer a question about the code. No edits |
| `pedacito chat PROJ` | Interactive session; the index stays warm |
| `pedacito new DIR -t "..."` | Start a project from scratch (or `--spec FILE`) |
| `pedacito auto PROJ -t "..."` | Plan a goal, then implement it task by task |
| `pedacito restore PROJ` | Undo the last write |
| `pedacito eval` | Score the harness on fixture repositories. Use `--baseline` as a gate |
| `pedacito memory PROJ` | List, add, or forget remembered facts |
| `pedacito skills PROJ` | List the available skills |
| `pedacito tests PROJ --files F` | Detected test command, and the tests an edit to F can break |
| `pedacito webcheck PROJ` | Static checks for web files: DOM ids, script loading, missing scripts |
| `pedacito diagnose` | Run the project's linter or type checker |
| `pedacito prune PROJ` | Delete old sessions, backups, and events |
| `pedacito reset PROJ [--reindex]` | Wipe `.pedacito/` runtime state. Code and config stay |
| `pedacito trust PROJ` | Approve a project's `.pedacito.toml` |
| `pedacito worktree create\|list\|remove NAME` | Manage isolated git worktrees |

Flags available on most commands: `--exclude 'pattern'`, `--include 'pattern'`,
`--ext .kt`, `--steps N`, `--model ID`, `--url URL`, `--profile NAME`,
`--config FILE`, `--root PATH`.

---

## Where things live

Per project, under `<project>/.pedacito/`:

- `index.json`: the map.
- `summaries.json`: the summary cache.
- `backups/<stamp>/`: pre-write snapshots.
- `sessions/<stamp>.md`: what was asked, looked up, and changed. Open this when an
  edit turns out wrong later.
- `events/*.jsonl`: one JSON line per step, call, fold, and task, with tokens,
  cache hits, and latency.
- `memory.md`, `reviews/` (saved review pages), and `log.txt` (event trace).

Deleting the folder is always safe. You only pay for a re-index. `pedacito reset`
clears it for you, and `pedacito prune` trims the old files.

Your settings live in `~/.config/pedacito/config.toml` (`%APPDATA%\pedacito\` on
Windows). `pedacito config` shows every resolved value and which file or profile
set it. Settings resolve in layers: built-in defaults, then that file, then the
project's `.pedacito.toml`, then `--config FILE`, then `--profile`, then
command-line flags. Next to the config file, the GUI remembers recent projects in
`gui_recent.json`. `server_modes.json` stores what each server and model supports
(JSON modes and tokenizer rate, refreshed by `--doctor`). `model_cards.json` holds
the `pedacito calibrate` results. All of these are safe to delete.
