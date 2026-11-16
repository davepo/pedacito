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

Not sure which model size to load? The [VRAM quick reference](README.md#which-model-fits-your-gpu)
maps 6–8 GB, 12–16 GB, and 24 GB+ cards to the model sizes that work best on
each. Start with one local model; add a hosted planner or verifier later if you
want one, and it is optional.

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
discovery**. Click **Scan local servers**. The scan is capped at 20 seconds and
returns whatever it found when the budget is spent, so a slow server cannot
block it. Pedacito asks how much GPU memory the
machine running the server has, in GB. Then tick only the models you want. The
name box lets you rename a profile. Click **Import Selected**. Each model becomes a
profile named after it, and one is set as the default. Each profile in the list
has a **Calibrate** button that runs a quick measurement of that model, and a
calibration badge: ✓ Calibrated (with the date), ⚠ stale, or ○ Not calibrated.
Hosted provider profiles show ☁ Hosted instead, and their Calibrate button is
disabled — calibration is only for local models; hosted models use
provider-table defaults automatically.

**From a hosted provider.** In the same tab, click **+ Add Hosted Provider**.
Choose a preset: **Claude (Anthropic)**, **ChatGPT (OpenAI)**, **DeepSeek**,
**Qwen (DashScope)**, **Google Gemini**, or **OpenRouter**. Provide the key, or
name the environment variable that holds it. Pedacito connects to the provider,
lists its models, and imports the ones you tick, one profile each.

Back in the top bar, the profile picker switches the model for the next task.
Its status light shows whether the server is reachable. The **no-think** or
**thinking** chip next to it shows whether reasoning is allowed for that profile,
and a calibration badge shows whether the active model has a fresh model card.

To measure every local model at once, open **Project ▾ → Calibrate All Local
Models…** (also in **Settings → Model discovery**). A confirmation lists the
local profiles — hosted ones are excluded — and one background job then loads
and measures each in turn, ticking them off as their cards are saved. Tick
**Re-calibrate models that already have a fresh card** to redo them all.

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
   change the new version by hand before accepting. Each file shows clause
   badges (`R1`, `R2`, ...) for the parts of your request it serves; accepting
   only some files flags any clause left unaccepted.

Pedacito backs up each file before it writes. If a file changed on disk after
the preview, the write is refused, so your own edits are never overwritten.

**Short on detail?** Type a one-line idea and click **✦ Refine Prompt** above the
composer. Pedacito replaces the line with a full spec: an objective, the files to
create or modify (using real paths from your project), interface contracts,
requirements and edge cases, and verification. Read it, edit it if you like, then
click **Run**. **↶ Undo Refine** puts your original text back. Drag the top edge of
the composer to make it taller while you read.

**Building something visual?** Click **✨ Creative** above the composer to force
creative/design (Builder) mode for the next Task/Auto/New run: a Senior Product
Engineer & Visual Designer persona, warmer sampling and whole-file HTML/CSS
rewrites instead of the smallest-diff discipline. With the toggle off, design
tasks (websites, restyles, `.html`/`.css`/UI-component work) are still
auto-detected — that is `creative_auto`; `creative_mode = true` in the config
forces it for every task (A21).

### A6. Ask a question (Ask)

Choose **Ask** to ask how the code works. The answer streams in and nothing is
written. Use it to understand a module before you change it.

### A7. Run an autonomous plan (Auto)

Choose **Auto**, then describe a goal, such as `add CSV export to the report
page, with tests`.

- Tick **Plan only** under the options to receive a checklist first. You can
  edit it in the **Plan** panel (**Edit**, then **Save**), then click **Run**.
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
caught before its tests run. In a web project it also checks that every local script
and stylesheet exists and that every DOM id a script looks up is in the page. If a
step still fails after its repair rounds, the
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
unless you list them in `fetch_allowed_hosts`. A public name that *resolves* to a
private address is refused too (DNS rebinding), and the connection is pinned to
the checked IP so a second DNS answer cannot swap it mid-request.

### A14. Other things to know

- **Settings** edits your user config or the project's `.pedacito.toml`. Your
  comments survive, and every edit is checked before it is saved. The **Raw
  file / reset** tab shows the file with every stored secret masked as
  `***REDACTED***`; save the mask unchanged and the stored key is kept.
- **MCP servers:** **⚙ Settings → MCP Servers** adds and manages named
  `[mcp_servers.<name>]` tool servers (test, enable, trust, delete) without
  hand-editing TOML.
- **Project ▾** also holds **Calibrate All Local Models…** and the upkeep
  dialogs: **Memory & repo lessons…**, **Skills library & creator…** (A19),
  **Consolidate session…** (A20), **Background servers…**, **Run project
  checks…**, **Prune old sessions/backups…** and **Reset project state…**
  (prune and reset preview before they write).
- **Telemetry footer:** under the workspace, a live bar shows the context
  window in use (green under 70%, amber to 85%, red above), turn and session
  tokens, generation speed, KV cache hits and folds, and the active model.
  Click it to expand the full drawer. Its **Consolidate Session** button
  compacts a long chat into a handoff checkpoint (A20). The CLI prints the
  same numbers as a one-line ribbon after each task; `show_telemetry = false`
  hides it.
- **Themes:** pick a color scheme from the theme menu in the top bar. There are
  nine. The choice is saved to your config as `gui_theme`.
- **Attach** a text file, or drag it into the chat box, to use it as the task, a
  question, or a spec.
- **Steer:** while a task runs, type a note to redirect the model. You can also
  answer its clarifying questions inline, or click **Stop**.
- **Terminal:** the bottom panel shows everything Pedacito prints, live.
- **🌐 Preview:** when the project has `.html` pages, a Live Preview tab joins
  Files / Backups / Logs and renders the page in a read-only iframe. It reloads
  when a run finishes — and while a run writes — so you see the result
  immediately (A22).

### A15. Rename a symbol across the project

Renaming a function or variable everywhere it is used does not need a model.

1. Open **Project ▾** in the top bar and choose **Rename symbol…**.
2. Type the current name in **Rename**, the new name in **To**, and optionally one
   file in **Scope file** to limit the rename to that file and the files that import
   it.
3. Click **Preview**. The per-file diffs appear and nothing is written yet.
4. Click **Rename** to write the change. The originals are backed up first, so you
   can undo it from **Project ▾ → Restore from a backup…**.

Strings and comments that happen to contain the old name are left alone. A parameter
in an unrelated function that has the same name is renamed too, so read the preview.
On the CLI, `pedacito rename OLD NEW --strings` also renames whole-word occurrences
inside string literals (comments stay untouched), and `--include-labels` moves only
test-label strings (`test`/`it`/`describe`/`suite`); the GUI dialog has no such option.

### A16. Let the model run one scoped test

While it investigates, the model can run a single scoped test and read the live
traceback before it writes a fix. Python targets run under pytest; `*.test.*` /
`*.spec.*` files run under Vitest (or Node's built-in runner), `*_test.go` under
`go test`, and `tests/*.rs` under `cargo test`. This is on by default. Nothing is
run through a shell, and each run is capped at 15 seconds.

If you would rather approve each run, open **⚙ Settings → All settings**, search for
`enable_run_test_in_gather`, set it to off, and save. Each test run then appears in
the **Permission requested** dialog under **Test Execution**.

### A17. Use a scout model for lookups (two GPUs)

If you have a small model on a second GPU, or a lightweight server, let it do the
lookups while the large model plans and edits.

The quick path: **⚙ Settings → Profiles → Dual-GPU / Scout + Architect Pairing**.
Pick the architect (your planner/editor profile) and a scout, or choose
**➕ Create a starter Scout profile…** and give the second server's URL, model id,
VRAM and tier. **Save pairing** writes `default_profile` + `scout_profile` in one
comment-preserving edit. **⚙ Settings → Auto-Tune** also asks a dual-GPU question
and, when a second card fits, offers **Create starter Scout & pair…**.

The manual path:

1. Add the small model as its own profile in **⚙ Settings → Profiles**, with its own
   server address and model.
2. Open **⚙ Settings → All settings**, search for `scout_profile`, and choose that
   profile. Choose **None (use active profile)** to turn the scout off.
3. Keep the large model as the active profile. The top bar then shows a
   `Scout: <profile>` badge beside the model chip.

If the scout cannot be reached, Pedacito falls back to the active model and shows a
warning. The sizes to pair, and the one-GPU caveat, are in the
[model guide](MODEL_GUIDE.md#dual-gpu--dual-model-scout--architect-routing-scout_profile).

### A18. Route each task to a specialist model

A scout splits one task by *job*; specialist routing splits a run by *task*: the
coder profile takes "refactor the parser", the vision profile takes "fix the button
layout", the reasoner takes "optimize the sorting algorithm".

1. Tag your profiles: **⚙ Settings → Profiles → [select one] → Connection &
   Capabilities → specialties** — toggle the tag pills (Coding, Frontend / UI,
   3D / graphics / SVG, Math / algorithms, Testing / QA, SQL / data,
   Translation / i18n), or type a custom lowercase tag and press Enter. Leave it
   empty to let the model card decide: `pedacito calibrate` detects tags from the
   model name, the architecture library, and six micro-probes (SVG, math,
   translation, SQL, testing/QA, planning), and the form shows what it detected.
2. Turn routing on with the **✦ Auto-route** checkbox in the topbar (or
   `auto_route_specialties = true` in the config).

Switches happen between tasks, never mid-investigation. On one GPU the outgoing
model is unloaded before the specialist loads, so both fit in VRAM one after the
other; on two GPUs (or with a hosted specialist) both stay loaded. The Activity
panel shows which specialist is running, and an unreachable one falls back to the
active model with a warning. On the CLI: `--auto-route` per run, `--no-auto-route`
to override the config.

### A19. Browse and create skills

Skills are short checklists the model follows for one kind of task. Eighteen
starters ship built in (testing, refactoring, FastAPI endpoints, canvas/SVG,
SQLite migrations, async safety, security hardening, i18n, parsers,
profiling, web design, desktop GUIs & games, …); the top-2 trigger matches
for your task are injected automatically, capped at 1,200 chars.

1. Open **Project ▾ → Skills library & creator…** (also linked under
   **⚙ Settings → Quick links**). Every skill shows a scope badge:
   **bundled**, **user** (all your projects) or **project** (this repo).
2. Type a task in the preview box and press Enter to see which skills it
   would inject; click any skill to read its triggers and checklist.
3. Click **+ New Custom Skill**, fill in the name (a slug), description,
   comma-separated triggers and the checklist body — or just write the
   description and click **✦ Auto-Draft from Description** to let the active
   model draft the whole skill. Review the draft, then **Save to Project**
   (`.pedacito/skills/<name>.md`) or **Save to Global**
   (`~/.config/pedacito/skills/<name>.md`).
4. Custom skills can be edited or deleted from the same dialog; built-ins
   are immutable. A project skill may shadow a built-in of the same name.

The CLI twins are `pedacito skills list|show|add|draft|rm` (B13).

### A20. Consolidate a long session

When a chat grows long, consolidate it: the transcript is distilled into a
handoff of under 400 tokens with four sections — Goal & Context, Files
Modified & Key Decisions, Verified Test Status, Open Next Steps — saved to
`.pedacito/consolidated_session.md`. Durable decisions are pinned to
`memory.md`.

1. Click **Consolidate Session** in the telemetry footer (or
   **Project ▾ → Consolidate session…**) and confirm. A running job must be
   stopped first.
2. The chat restarts from the single handoff card, and the Context Meter
   drops to the handoff's size immediately — the next turn starts small.

On the CLI: `/consolidate` (alias `/compact`) inside `pedacito chat`, or
`pedacito memory PROJ --consolidate` over the newest session transcripts.
Over RPC: `{"id": 1, "op": "consolidate"}`.

### A21. Creative / design mode (the Builder persona)

A bug fix wants the smallest possible diff; a landing page wants a designer.
Pedacito tells them apart automatically: with `creative_auto = true` (the
default), a task that targets UI files (`.html`, `.htm`, `.css`, `.svg`,
`.jsx`, `.tsx`, `.vue`, `.svelte`) or reads like design work — website,
portfolio, landing page, hero, dashboard, redesign, restyle — runs as a
**Senior Product Engineer & Visual Designer**:

- the answer temperature rises to at least `0.6` on local servers, so the
  model explores instead of repeating the first safe idea;
- the reply budget floors at 8192 tokens (your `request_max_tokens_cap`
  still caps it), so a whole page is not cut off mid-`<div>`;
- whole-file `.html`/`.css` rewrites are allowed: the fragment veto and the
  small-window oversized-rewrite brake stand down (the wipe refusal and the
  JS export guard still apply).

To force it for **every** task, click **✨ Creative** above the composer, pass
`--creative` on the CLI, type `/creative on` in the TUI, or set
`creative_mode = true` in the config. `/creative auto` restores detection and
`/creative off` (both settings false) keeps surgical edits everywhere;
`/status` shows the current state. On a hosted API the persona and budget
still apply, but the provider keeps its own sampling (see "Several models:
profiles" below).

### A22. Live Preview: watch the page as you build it

When the open project has at least one `.html` file, the workspace panel grows
a **🌐 Preview** tab beside Files / Backups / Logs.

1. Open the tab and pick the page in the dropdown (`index.html` is the
   default). Relative CSS/JS/image links resolve through the same read-only
   route, so the page looks the way it will when served.
2. Run a website task. The preview reloads automatically when the run
   finishes — and while it writes, when the tab is open.
3. **⟳ Refresh** reloads the page by hand; **↗ New tab** opens it in a full
   browser tab.

The preview is read-only and never leaves the project root: `..` segments and
escaping symlinks are refused, sensitive files are refused, and only the app's
own origin may frame the page. Pair it with ✨ Creative (A21) for the full
design loop.

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

After upgrading Pedacito, `pedacito config --upgrade` appends any newly
documented options to your config, commented out, without touching your
existing settings or comments. `pedacito config --upgrade --dry-run` lists
what would be added without writing.

### B2. Find models and measure them

```bash
pedacito models scan
```

This finds local servers on any port, lets you pick models, writes one profile
per model, and measures each one with a quick calibration. The calibration
probes tool choice, JSON output, edit format, resistance to planted prompt
injections, honesty about what was read, long-context retrieval, and whether
hidden reasoning eats the answer budget. It also detects the model's
**specialties** (name heuristics, the architecture library, plus six
micro-probes: SVG/UI, math, translation, SQL, testing/QA, planning) for
specialist routing — see A18. The result is saved as a model card. Hosted
imports are never calibrated: they use provider-table defaults automatically.

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

To measure a model again later, run `pedacito calibrate`. To measure every local
profile in one go, run `pedacito calibrate --all` — it asks first (`-y` skips the
question), loads each model in turn, unloads it before the next, and skips
profiles with a fresh card unless you add `--recalibrate`. Hosted profiles are
refused: calibration is only for local models. To see the results side
by side, run `pedacito profiles`, or `pedacito profiles --recommend` for suggested
roles.

After each task the CLI prints a one-line telemetry ribbon on stderr — context
bar, turn tokens, speed, cache hit. Set `show_telemetry = false` to hide it;
`--jsonl` and rpc modes never print it.

Two more things help when a hosted model is in the loop. `--max-cost USD` on a
task command stops before the next model call once the estimated spend reaches
that cap (it needs `price_*_per_mtok` set for the model). In the GUI footer,
**KV Cache Hit %** shows how much of the prompt your local server reused, which
is the number to watch if a long session starts feeling slow.

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

For design work, add `--creative` (creative/design Builder mode: designer
persona, warmer sampling, whole-file HTML/CSS rewrites — A21):

```bash
pedacito ask ~/code/site -t "redesign the landing page hero" --creative
```

Design tasks are auto-detected even without the flag (`creative_auto`);
`--creative` forces the mode for every task in the run.

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
it, and a run can be resumed; the file also stores the full goal, which a
resumed run hands back to each task. Useful flags:

- `--verify "pytest -q"` checks and repairs each task.
- `--verify-final` runs that check once, after the last unchecked task, instead
  of after every one.
- `--diagnostics` runs the project's linter and type checker on each task.
- `--worktree NAME` works in an isolated git worktree and leaves your checkout
  alone.
- `--plan FILE` uses your own checklist instead of generating one.
- `--max-tasks N`, `--max-minutes N`, or `--steps N` set limits. There are none by
  default.
- `--strict` exits with status 1 if a task applied edits but left a failed edit or
  failing tests, which is useful in CI.
- `--refine` expands a one-line goal into a full spec before planning.
- `--force-auto` skips the Task-vs-Auto suitability nudge (below); `-y`/`--yes`
  does the same and accepts other confirmations.

**Single-pass edits get a nudge.** When the goal looks like one-pass Task work
(it targets 1–2 existing files, uses localized edit verbs like `fix`, `restyle`
or `rename`, or tweaks content/styling in a tiny project), `auto` asks
`Run as a single-pass Task instead? [Y/n]` before building a multi-step
`PLAN.md`. Enter or `y` runs the whole prompt as one fast `ask` pass; `n` keeps
the Auto loop. Pipes and CI are never blocked on the question — they get a
one-line note and Auto proceeds. Prompts that say "multi-step plan",
"create a plan", "from scratch", or name ≥4 new files/modules always keep Auto.
In the TUI the same choice appears on `/auto GOAL` (`/auto! GOAL` forces the
loop); in the GUI a "💡 Looks like a single-pass edit" banner offers
**Switch to Task mode (Recommended)**, and clicking Run on such a prompt asks
**Run as a single-pass Task instead?** with **Run as Task (Recommended)** /
**Run Auto Anyway** buttons.

A repair round that makes the check worse is rolled back automatically, and that
task is marked failed. An unattended run cannot compound the damage. A repair
also cannot "fix" a failing check by editing a pre-existing test file -- those
edits are refused and aimed at the code under test instead (a task that asks
for test changes is exempt). A step that
still fails after its repair rounds is rolled back to the last passing checkpoint
(`auto_green_ratchet`, on by default), so later steps never build on it.

### B10. Interactive session

```bash
pedacito chat ~/code/anki            # add --apply or --review for the whole session
```

The index stays warm between tasks. Type `/help` for the commands. The most
useful ones are `/profile NAME` and `/model ID`. When the session grows long,
`/consolidate` (alias `/compact`) compacts it into a handoff checkpoint and
the next task starts from that handoff alone (A20). Ctrl-C cancels the running
task and keeps the session open. Switching profiles and running the same task
again is the cheapest way to compare two models.

### B10b. Split-screen session (`pedacito tui`)

```bash
pedacito tui ~/code/anki                  # apply-on, with an undo backup
pedacito tui . --no-apply                 # preview edits only
pedacito tui . --headless < commands.txt  # scripted/CI mode, no PTY
```

`pedacito tui` keeps the index warm and splits the terminal into four zones:
a header with the project, profile/model, calibration badge, Auto-Route,
Scout and git state; a transcript pane beside a collapsible **Activity &
Plan** sidebar (100+ columns, `/sidebar` toggles it); a two-row telemetry
footer; and the `pdc tui (profile) >` prompt.

The footer is live: the **Context Meter** turns green below 70%, amber from
70–85% and red above 85%, next to `Turn: +in/+out`, session totals,
`Speed: 44.2 tok/s (avg 41.8 tok/s)`, KV cache, folds and the active
model/role. The **Burn Rate** (token burn rate) and the sidebar follow every `read`/`grep`/`mcp_call` step,
`model_swap`, verification receipt and `PLAN.md` checkbox while a task runs.

Keys work inside the TUI instead of echoing escape codes: **PgUp/PgDn** scroll
the transcript half a page, **Shift+Up/Down** scroll three lines,
**Alt+PgUp/PgDn** scroll the Activity/Plan sidebar, and the **mouse wheel**
scrolls three lines. The default wheel mode (`?1007h` Alternate Scroll) keeps
native left-click selection and OS copy/paste; **F2** or `/mouse on` enables
optional SGR wheel capture, and `/mouse off` restores native selection.
**Up/Down** recall command history, **Left/Right/Home/End** edit the prompt,
**Tab** completes slash commands and profile names, and **Esc/End** jump to the
live bottom. A scrolled-back view shows `[ ↑ Scrolled back N lines ... ]` and
stays stationary as new log lines arrive.

Type a task as usual, or use `/help`, `/keys`, `/status`, `/auto GOAL`
(single-pass edits get a Task-mode suggestion first; `/auto! GOAL` forces the
multi-step loop), `/plan`, `/consolidate`, `/profile`, `/auto-route`, `/think`,
`/creative [on|off|auto]` (design mode: on = every task, auto = detect design
tasks, off = surgical edits), `/apply`,
`/skills`, `/memory`, `/procs`, `/diff`, `/undo`, `/reset`, `/sidebar`,
`/scroll up|down|top|bottom|N`, `/mouse [on|off]`, `/clear` and `/quit`.
Permission prompts appear in the prompt bar (`y` once, `a` for the session,
`n` deny).

### B11. Benchmarks

The six-project benchmark suite runs from the GUI (the **▦ Benchmarks** button in
Part A10) or from the CLI:

```bash
pedacito bench --list                                  # ids, checks, deliverables
pedacito bench --recent                                # past runs, no model needed
pedacito bench rpn_expression_calc                     # one benchmark
pedacito bench all --json --strict                     # all six, CI-friendly
pedacito bench rpn_expression_calc --dir ~/benchmarks  # choose the output root
```

Each run builds the project through the normal Pedacito loop, then grades the
model's own tests and the hidden holdout rubric. Progress streams to stderr; the
scorecard (tests passed, rubric checks, tokens, duration and report path) goes to
stdout. `--json` prints a structured result for scripts, and `--strict` exits 1
unless both the model tests and the holdout rubric pass. `--list` and `--recent`
need no model server or `vram_gb`.

The CLI also measures a model with `pedacito calibrate` and `pedacito eval`,
which are described in the [model guide](MODEL_GUIDE.md#how-to-measure).

### B12. Rename a symbol (no model call)

```bash
pedacito rename apply_discount discounted_price                        # preview the diffs
pedacito rename apply_discount discounted_price --file pkg/core.py     # that file and its importers
pedacito rename apply_discount discounted_price --strings              # also whole-word string literals
pedacito rename toggleStep flipStep --include-labels                   # also test-label strings only
pedacito rename apply_discount discounted_price --apply                # write it
pedacito restore ~/code/anki                                           # undo it, if needed
```

The preview is the default. With `--apply`, the originals are backed up first, so
`pedacito restore` can undo the rename. Python, JavaScript, TypeScript, Go and Rust
identifiers are renamed; a string or comment containing the old name is left alone
unless you pass `--strings`, which also renames whole-word occurrences inside
string literals (comments still stay), or `--include-labels`, which moves only
test-label strings (the first argument of `test`/`it`/`describe`/`suite`). Every
changed file must pass the syntax check, or nothing is written.

### B13. Manage skills

```bash
pedacito skills .                          # every skill with its scope badge
pedacito skills show add-a-test            # one skill's triggers + checklist
pedacito skills add deploy-checklist \
  --description "Ship a release safely" \
  --triggers "deploy,release" --body "1. Run the suite. 2. Tag. 3. Publish."
pedacito skills draft "Skill for Alpine.js dark-mode components"   # model writes it
pedacito skills rm deploy-checklist        # delete a custom skill
pedacito skills list --global              # the user-level library instead
```

`add` and `draft` write `.pedacito/skills/<name>.md` in the project, or
`~/.config/pedacito/skills/<name>.md` with `--global`. Built-in skills are
immutable: `rm` refuses them, but a custom skill that shadows a built-in can
be deleted to let the built-in show through again. The GUI twin is
**Project ▾ → Skills library & creator…** (A19).

### B14. Consolidate session history

```bash
pedacito memory ~/code/anki --consolidate
```

Distils the newest session transcripts into a handoff checkpoint under 400
tokens (Goal & Context, Files Modified & Key Decisions, Verified Test Status,
Open Next Steps), writes `.pedacito/consolidated_session.md`, pins durable
decisions to `memory.md`, and prints the before/after token counts. Inside
`pedacito chat` the same operation is `/consolidate` (alias `/compact`); over
RPC it is the `consolidate` method (alias `compact`). The GUI button is
**Consolidate Session** in the telemetry footer (A20).

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

**Hosted APIs keep their own sampling.** A profile with a non-local `base_url`
(the `[profiles.api]` above) receives no `temperature`, `top_p`, `top_k`,
`min_p` or `repeat_penalty` by default — including keys from the `sampler`
table — so the provider's tuned defaults apply. To send the configured
temperature anyway, set `send_temperature_to_hosted = true`; to pin one value
for hosted endpoints (e.g. `1.0` for Claude), set `hosted_temperature = 1.0`
globally or in the profile. Either way only `temperature` is sent; the other
four keys stay omitted. Local servers (loopback or LAN) keep every configured
value: `temperature_answer` defaults to `0.3` (cool, not greedy), creative
tasks run at ≥ 0.6, and a reply that repeats itself gets one gentle
`presence_penalty = 0.15` nudge on the retry instead of a cold temperature.

To send only the lookup steps to a smaller profile, set `scout_profile` to its name,
for example `scout_profile = "scout"` in `[universal]`. The active profile still
plans, answers and edits. The [model guide](MODEL_GUIDE.md#dual-gpu--dual-model-scout--architect-routing-scout_profile)
explains how to pair the sizes.

To send whole *tasks* to specialist profiles instead, tag each profile with
`specialties = ["coding", ...]` (or let `pedacito calibrate` detect the tags) and
set `auto_route_specialties = true`, or pass `--auto-route` to `ask`, `auto` or
`new --complex`. Each task then runs on the best-matching profile, switching only
at task boundaries; on one local server the previous model is unloaded first so
the next one fits in VRAM. `--no-auto-route` keeps everything on the active
profile. A task title prefixed `[specialty: TAG]` forces that specialty.

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
| `search block not found` | The model copied line-number prefixes, or the file changed under it -- or its text carries bytes the file does not (a smart quote, mojibake, a tab, CRLF). The failure shows the closest matching text with current line numbers, and the retry is offered the range anchor (`--- FILE: path LINES 40-46`), which ignores the quoted text entirely and applies by number. If it keeps failing, narrow the task. |
| Model says a function does not exist | It is not indexed. Check `pedacito files`. An ignore rule may be too broad. |
| `EXIT: 5` / `no tests ran` from `run_test` | Nothing was collected: not a failing test, and the code may be fine. The target is probably a standalone script (`def main()`, `if __name__ == "__main__"`). Run it with `run` instead, or write real `def test_...` functions if it was meant to hold tests. Renaming working functions to chase the exit code is the wrong fix, and the tool says so. |
| A new browser project's page is blank when opened by double-clicking | ES modules are refused over `file://` (the console shows a CORS error). Serve the folder: `python -m http.server 8000`, then open `http://localhost:8000/index.html`. The generated README's `## Run` block says so. |
| `Model would not produce valid JSON` | Usually a truncated reply. Pedacito retries with more room. If it persists, raise `max_tokens_step`. A thinking model that spends its whole budget reasoning returns no JSON, so turn thinking off for that profile. |
| Every request fails, or "kept hitting limits" | The request is bigger than the server's context window. Check the `[auto-fit]` line printed at startup. If the server reports a stale window, set `context_tokens` (and `k_cache_quant` / `v_cache_quant` for a quantized cache). |
| The answer is nonsense on a local server | Some servers (Ollama) silently truncate an oversized prompt. Raise the server's context length. |
| A hosted API seems to ignore `temperature` | By design: hosted endpoints get no sampling keys by default, so the provider's tuned defaults apply. Set `send_temperature_to_hosted = true` to send the configured temperature, or pin `hosted_temperature` (e.g. `1.0` for Claude). Local servers keep every configured value. |
| A local model repeats itself verbatim | The answer phase runs at `temperature_answer = 0.3` (not greedy) to avoid exactly this, and a repetition retry gets a gentle `presence_penalty = 0.15` nudge. If it persists, raise `temperature_answer` slightly or switch profiles; a design task can also use `--creative` (≥ 0.6). |
| The model seems lost or reads the wrong things | Name the file in the task. If you set `--steps` or `max_steps`, raise it. |
| Answers are vague | The map may have switched from a full listing to per-file cards. `pedacito map PROJ --tier auto` shows which. |
| Stale or misleading summaries | A docstring replaces a generated summary. Set `summarise_documented = true` and re-index. |
| No review page appeared | Nothing anchored, so there was nothing to review. Check the printed failures. |
| `FrontendContractError` after an edit | The page cannot load as written: a script or stylesheet path that does not exist, a `getElementById` or `querySelector('#id')` for an id no HTML defines, JavaScript that does not parse, a module-level name bound twice (a second `import { X }` on a file that already imports it), or a call to a name no module declares or imports. The message names the file, line and the exact fix. `pedacito webcheck PROJ` lists every such problem. |
| The page passes `webcheck` but stays blank after an edit | Static checks cannot see runtime failures (a module that throws on load, a mount that never happens). `pedacito domcheck PROJ` loads the entry page in headless Chromium and reports page errors, mount regressions and unbound controls; it needs the optional Playwright install (`pip install playwright && playwright install chromium`). To run it after every applied web edit, set `verify_dom = true` and `allow_shell = true`, or pass `ask --verify-dom`. |
| A sequencer/synth passes every check but plays no sound | Web Audio fails silently: a context never resumed inside a user gesture, a graph that never reaches `ctx.destination`, oscillators never `start()`ed, gains pinned at 0, NaN automation. `pedacito domcheck PROJ` instruments the audio API, clicks Play and sweeps the sliders, and reports these as errors (non-audio pages are unaffected). |
| The server rejects structured output, or steps keep failing to parse | Pedacito steps down to a simpler structured mode by itself. Run `pedacito config --doctor` to see which modes the server supports. Set `enforce_json_schema = false` only if the server misbehaves under constrained sampling. |
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
| `pedacito config --upgrade [--dry-run]` | Add newly documented options to your config, commented out; `--dry-run` only lists them |
| `pedacito models scan` | Find local servers or a hosted API, pick models, write profiles, measure them |
| `pedacito calibrate` | Measure the model and save a model card that tunes defaults; `--all` walks every local profile (hosted models are refused) |
| `pedacito profiles [--recommend]` | List profiles and their scores; suggest roles |
| `pedacito how "..."` | Ask how to use Pedacito. No project needed |
| `pedacito files PROJ` | What would be indexed, and why. Free |
| `pedacito cost PROJ` | What indexing will cost. Free |
| `pedacito index PROJ` | Build or refresh the index |
| `pedacito map PROJ [-t "task"]` | Print the index, ranked for a task. No model needed |
| `pedacito ask PROJ -t "..."` | One task. Dry run by default. Add `--review`, `--apply`, `--refine`, or `--creative` (design mode) |
| `pedacito explain PROJ -q "..."` | Answer a question about the code. No edits |
| `pedacito chat PROJ` | Interactive session; the index stays warm |
| `pedacito tui PROJ` | Split-screen terminal UI: transcript + activity/plan sidebar + Context Meter, token burn and tok/s telemetry. `/help` lists the slash commands; `--headless` is scriptable |
| `pedacito rpc [PROJ]` | JSONL-RPC over stdin/stdout for editors and agents; the index stays warm |
| `pedacito mcp list` | Show the tools of the named MCP servers and their trust flags |
| `pedacito new DIR -t "..."` | Start a project from scratch (or `--spec FILE`) |
| `pedacito auto PROJ -t "..."` | Plan a goal, then implement it task by task |
| `pedacito restore PROJ` | Undo the last write |
| `pedacito rename OLD NEW [--file F] [--strings] [--include-labels] [--apply]` | Rename a symbol across the project. No model call; preview by default; `--strings` also renames whole-word string literals, `--include-labels` only test-label strings |
| `pedacito eval` | Score the harness on fixture repositories. Use `--baseline` as a gate |
| `pedacito bench [BENCHMARK\|all]` | Run the six diagnostic benchmarks and holdout rubric. `--list`/`--recent` are free; `--json`/`--strict` for CI |
| `pedacito memory PROJ` | List, add, or forget remembered facts; `--consolidate` compacts session history into a handoff |
| `pedacito skills [list\|show\|add\|draft\|rm] [PROJ]` | Browse and manage the skill library (18 built-ins + custom); `--global` targets the user library |
| `pedacito tests PROJ --files F` | Detected test command, and the tests an edit to F can break |
| `pedacito webcheck PROJ` | Static checks for web files: DOM ids, script loading, missing scripts, ES module bindings and links |
| `pedacito domcheck PROJ` | Headless DOM smoke check: load the entry page in headless Chromium, prove the module graph mounts, report console errors, unbound controls and silent Web Audio failures (needs Playwright) |
| `pedacito diagnose` | Run the project's linter or type checker |
| `pedacito prune PROJ` | Delete old sessions, backups, and events |
| `pedacito reset PROJ [--reindex]` | Wipe `.pedacito/` runtime state. Code and config stay |
| `pedacito trust PROJ` | Approve a project's `.pedacito.toml` so its execution settings and capability toggles apply |
| `pedacito worktree create\|list\|remove NAME` | Manage isolated git worktrees |
| `pedacito serve --root PROJ --name web -- npm run dev` | Start a named background server for the project |
| `pedacito procs --root PROJ [logs\|stop NAME] [--all]` | List, tail or stop the project's background servers. The list hides superseded stopped entries; `--all` shows them |
| `pedacito lsp --root PROJ --file F --line N --op definition` | Ask the project's language server for definitions, references or hover |

Flags available on most commands: `--exclude 'pattern'`, `--include 'pattern'`,
`--ext .kt`, `--steps N`, `--model ID`, `--url URL`, `--profile NAME`,
`--config FILE`, `--root PATH`, `--creative` (creative/design Builder mode).

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
- `consolidated_session.md`: the newest session-consolidation handoff (A20).
- `skills/`: custom skills, as `NAME/SKILL.md` or `NAME.md` (A19, B13).
- `micro_facts.json`: commands confirmed to work in this repository, and repair
  lessons under `anti_patterns`. A lesson is written only after a fix is verified.
  A lesson confirmed twice whose files share one directory is promoted into an
  auto-managed path-scoped rule (`rules/auto_*.md`) and leaves `anti_patterns`;
  those files clean themselves up when the files they name are deleted. Lessons
  carry `runs` / `recurred` / `avoided` counters; one that never avoids its
  failure kind across five observed runs moves to `retired_lessons`.
  Safe to read, edit, or delete.

Deleting the folder is always safe. You only pay for a re-index. `pedacito reset`
clears it for you, and `pedacito prune` trims the old files.

Your settings live in `~/.config/pedacito/config.toml` (`%APPDATA%\pedacito\` on
Windows; a set `XDG_CONFIG_HOME` replaces `~/.config`). `pedacito config` shows
every resolved value and which file or profile set it. Settings resolve in
layers: built-in defaults, then that file, then the project's `.pedacito.toml`,
then `--config FILE`, then `--profile`, then command-line flags. Next to the
config file, the GUI remembers recent projects in
`gui_recent.json`. `server_modes.json` stores what each server and model supports
(JSON modes and tokenizer rate, refreshed by `--doctor`). `model_cards.json` holds
the `pedacito calibrate` results. All of these are safe to delete.
