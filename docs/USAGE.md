🌐 **English** | [Español](es/USAGE.md)

# Usage

This guide covers day-to-day use of Pedacito: a first run, the browser GUI, the command line, the
interactive and autonomous modes, and how to pin context into a task. For installing, see
[INSTALLATION.md](INSTALLATION.md). For settings, see [CONFIG.md](CONFIG.md). For the security
model and how to undo an autonomous run, see [SECURITY.md](SECURITY.md).

To extend Pedacito with skills, MCP servers, hooks and prompt overrides, see
[SKILLS_AND_TOOLS.md](SKILLS_AND_TOOLS.md). To work on Pedacito itself, see
[CONTRIBUTING.md](CONTRIBUTING.md).

Step-by-step walkthroughs with the output you should expect are in
[../QUICKSTART.md](../QUICKSTART.md). Choosing and tuning models is in
[../MODEL_GUIDE.md](../MODEL_GUIDE.md).

## Quickstart in 60 seconds

1. **Start a model server.** Load a model in LM Studio, Ollama, Lemonade, or
   llama.cpp. A hosted API works too, if you have a key.
2. **Install** from PyPI (requires Python 3.11+). Add the browser and vision extras
   if you want them, or see [Install](INSTALLATION.md#install) for pipx and source installs:

   ```bash
   pip install pedacito                     # Windows: py -m pip install pedacito
   ```

3. **Pick your way in.**

   **In the browser:**

   ```bash
   pedacito gui ~/code/your-project
   ```

   Open **Settings → Model discovery**, click **Scan local servers**, tick the
   models you want, and import them. Then ask for a change in **Task/Edit**.

   **In the terminal:**

   ```bash
   pedacito models scan                       # find servers, write a profile per model, measure each one
   pedacito ask ~/code/app -t "make the retry count configurable"   # dry run: shows a diff, writes nothing
   pedacito ask ~/code/app -t "make the retry count configurable" --review   # side-by-side page, you sign off
   ```

Step-by-step versions, with what you should see at each point, are in
[../QUICKSTART.md](../QUICKSTART.md).

## Switching language (EN | ES)

Pedacito's interface is available in English and Spanish. Switch it where you work:

| Surface | How |
|---|---|
| GUI | The **EN \| ES** switcher in the top bar; the choice is saved to `language`. |
| TUI | `F3` toggles EN/ES, or `/lang es` / `/lang en`. |
| CLI | `pedacito --lang es ...`, or `PEDACITO_LANG=es` (alias `PEDACITO_LANGUAGE`). |
| Config | `language = "es"` and `llm_reply_language = "match_ui"` (see [CONFIG.md](CONFIG.md#language--localization-language-llm_reply_language)). |

`llm_reply_language` only changes the language of the model's prose explanations. In the TUI,
`/profiles` lists every configured profile, and `/profile NAME` switches to one.

## What you get

| Area | What it does | Read more |
|---|---|---|
| **Small-model techniques** | A harness-run retrieval loop, a requirement state machine (`UNLOCATED → LOCATED → READ → EDITED → VERIFIED`), symbol-anchored edit microloops, an AST syntax gate before any test runs, traceback repair packets, KV prompt-cache-friendly conversations, and exact-span rehydration | [../TECHNIQUES.md](../TECHNIQUES.md) |
| **Engine V2 telemetry** | The GUI shows the live phase, the requirement ledger, and recent lookups as a task runs | [../QUICKSTART.md](../QUICKSTART.md#part-a-the-gui) |
| **Repository memory** | Confirmed commands in `.pedacito/micro_facts.json`, with contradicted facts retired automatically. Unused tool schemas are pruned to save tokens | [../TECHNIQUES.md](../TECHNIQUES.md) §14 and §15 |
| **Local server scanner** | Finds LM Studio, Ollama, Lemonade, llama.cpp, and Strata / Avalon (preset at port 8080, `STRATA_API_KEY` fallback) on any port, then imports only the models you tick. See [Strata / Avalon onboarding](INSTALLATION.md#model-server-setup) | [Model server setup](INSTALLATION.md#model-server-setup) |
| **Hosted provider presets** | One-click setup for Claude, ChatGPT, DeepSeek, Qwen, Gemini, and OpenRouter. Keys can come from an environment variable instead of the config file | [Configure](CONFIG.md#configure) |
| **Role routing** | Send planner, verifier, or escalation work to a different profile, per job or per project | [../MODEL_GUIDE.md](../MODEL_GUIDE.md#hybrid-local--cloud-role-routing) |
| **Auto-Tune wizard** | Answer four questions about your hardware and codebase, preview the budgets, then apply them | [../QUICKSTART.md](../QUICKSTART.md#part-a-the-gui) |
| **Per-profile thinking** | Turn reasoning on or off for each model profile, from the GUI or `disable_thinking` | [../MODEL_GUIDE.md](../MODEL_GUIDE.md#thinking-per-profile) |
| **Model benchmarks** | Six small projects with a hidden holdout rubric, scored per profile, including a parametric SVG drawing test whose result is shown inline, so you can compare models on your own hardware | [../MODEL_GUIDE.md](../MODEL_GUIDE.md#compare-models-with-the-diagnostic-benchmarks) |
| **Prompt architect** | `✦ Refine Prompt` turns a one-line idea into a repo-aware spec with locked file targets and interface signatures. `↶ Undo Refine` restores your text | [../TECHNIQUES.md](../TECHNIQUES.md#17-the-promptspec-architect) |
| **Interface contracts** | Auto locks the project's public signatures into every step, and checks cross-file imports and call arity before the tests run | [../TECHNIQUES.md](../TECHNIQUES.md#18-contract-first-interface-locking--cross-file-verification) |
| **Static gate** | A fast undefined-name and syntax check (`ruff` F821 and E999, with a pure-Python fallback) runs before any test subprocess | [../TECHNIQUES.md](../TECHNIQUES.md#19-the-static-analysis-gate-a-linter-shaped-repair-trigger-for-free) |
| **Green ratchet** | Auto snapshots the project before each step and rolls a step that still fails back to the last passing state | [../TECHNIQUES.md](../TECHNIQUES.md#20-green-ratchet-step-checkpoints-for-the-autonomous-loop) |
| **Permission broker** | Web access, package installs, unapproved shell commands and sensitive file reads pause the run for an Allow Once, Allow for This Run, Always Allow, or Deny & Redirect decision | [Safety and trust](SECURITY.md) |
| **Safe writes** | Dry run by default, side-by-side review, backups before every write, `pedacito restore` to undo | [Review page](#review-page) |
| **Live test introspection** | During investigation the model can run one scoped test check (pytest, Node/Vitest, Go, or Cargo) and read the live traceback before it writes a fix. On by default (`enable_run_test_in_gather`) | [Scoped test runs](ARCHITECTURE.md#scoped-test-runs-during-investigation) |
| **Frontend contract checks** | Before tests run, local `<script src>` and `<link href>` files must exist, and literal `getElementById` and `querySelector('#id')` lookups must match the HTML ids | [Frontend contract checks](ARCHITECTURE.md#frontend-contract-checks) |
| **Creative / design mode** | Design tasks (websites, restyles, `.html`/`.css`/UI-component work) run as a Senior Product Engineer & Visual Designer: warmer sampling, an ≥8192-token reply budget and whole-file HTML/CSS rewrites. `--creative`, `/creative` in the TUI, or the GUI's ✨ Creative toggle force it for every task | [Creative / design mode](#creative--design-mode) |
| **Live Preview** | A project with `.html` pages grows a 🌐 Preview tab in the GUI: a read-only live view of the page that reloads when a run finishes — or while it writes | [Live Preview](#live-preview) |
| **Hosted sampling respect** | Hosted APIs receive no `temperature`/`top_p`/`top_k`/`min_p`/`repeat_penalty` by default, so the provider's tuned defaults apply; opt back in with `send_temperature_to_hosted` or pin `hosted_temperature` | [Hosted APIs keep their own sampling](CONFIG.md#hosted-apis-keep-their-own-sampling) |
| **Repair lessons** | A repair that fixes a failed check leaves a deduplicated, repo-specific lesson in `.pedacito/micro_facts.json`; a lesson a directory proves twice is promoted into an auto-managed path-scoped rule (`.pedacito/rules/auto_*.md`) that only shows up when the model reads that directory | [Repair lessons](ARCHITECTURE.md#repair-lessons-and-repo-gotchas) |
| **Symbol-level edits and rename** | Replace a whole function by name without echoing its old text, or rename a symbol across the project with no model call (`pedacito rename`) | [Editing](ARCHITECTURE.md#editing-searchreplace-not-diffs) |
| **Engine-enforced JSON and scout routing** | Lookup steps are constrained by the server's JSON schema support, and `scout_profile` sends the lookups to a fast model while the big model plans and edits | [Model guide](../MODEL_GUIDE.md#dual-gpu--dual-model-scout--architect-routing-scout_profile) |
| **Specialist auto-routing** | `auto_route_specialties` sends each task to the profile whose specialty tags fit it best (coder, UI/SVG, math, ...), at task boundaries only, with a clean VRAM handoff on one local server; `pedacito calibrate` detects the tags | [Specialist routing](ARCHITECTURE.md#specialist-auto-routing) |
| **Calibration badges & live telemetry** | Profiles show ✓ Calibrated / ⚠ stale / ○ Not calibrated / ☁ Hosted badges; `calibrate --all` measures every local model in turn; a GUI footer and a CLI ribbon track context use, tokens, speed and cache hits live | [Measurement and upkeep](#measurement-and-upkeep) |

### The GUI workspace

`pedacito gui` serves a local page on `127.0.0.1` with no CDN and no extra
dependencies, so it works offline. It runs the same engine as the commands.

- A two-row toolbar with the project path, a live **profile and model** picker,
  the server's status light, a calibration badge for the active profile
  (✓ Calibrated, ⚠ stale, ○ Not calibrated, or ☁ Hosted API), and session totals
  for tokens, cache hits, and cost.
- A **telemetry footer** under the workspace: the context window in use with a
  green / amber / red bar, turn and session tokens, generation speed, KV cache
  hits and folds, and which model holds the card. Click it to expand the full
  drawer.
- **Color schemes** from the Theme Manager (the palette button in the top bar). The nine built-in
  schemes and any custom themes come from the shared theme engine — the same set the TUI's
  `/theme` offers. See [Color schemes](CONFIG.md#color-schemes).
- **Five modes** next to the chat box: **Task/Edit** for changes, **Ask** for
  read-only questions, **Auto** for goal-to-checklist runs, **New** for starting
  a project, and **How ?** for the built-in reference.
- **✦ Refine Prompt** above the composer turns a one-line idea into a full spec
  in place: objective, files to create or modify, interface contracts,
  requirements and edge cases, and verification. **↶ Undo Refine** puts your
  original text back. Drag the composer's top edge to make it taller while you
  read the spec.
- **✨ Creative** above the composer forces the Builder persona for the next
  Task/Auto/New run: a designer's eye, warmer sampling and whole-file HTML/CSS
  rewrites. With it off, design tasks are still auto-detected (`creative_auto`).
  See [Creative / design mode](#creative--design-mode).
- The **✦ Auto-route** (specialist routing) and **Post-edit verify** (runs `auto_verify` after each Auto task) toggles now sit beside **✨ Creative**, under the text box, instead of in the top bar, which stays uncrowded next to the `EN | ES` switcher.
- **Project ▾ → 🎨 Generate image…** opens a dialog that draws an image with your image profile (see [Local image generation](#local-image-generation)).
- **Changes** show proposed edits as old and new code side by side, with
  per-file accept, reject, or hand edits.
- **Permission requested** dialogs pause a run when the model asks for web
  access, a package install, an unapproved shell command, or a sensitive file.
  See [Safety and trust](SECURITY.md).
- **Settings** edits your profiles, every setting, the Auto-Tune wizard, named
  MCP servers, and the raw config file, without deleting your comments.
- **Benchmarks** runs the six diagnostic projects against any profile. The
  pelican SVG result is drawn inline in its scorecard.
- **Project ▾** holds project actions: **Rename symbol…** renames an identifier
  across the project with no model call, and shows the per-file diffs before it
  writes. **Calibrate All Local Models…** measures every local profile in turn
  (hosted profiles are excluded; fresh cards are skipped unless you tick
  re-calibrate). **Restore from a backup…**, **Check model (loads it)** and
  **Clear terminal** are also here, plus the upkeep dialogs: **Memory & repo lessons…**,
  **Skills library & creator…** (browse the built-in, user and project skills,
  preview a task's trigger matches, write or ✦ auto-draft custom skills),
  **Consolidate session…** (compact the transcript into a handoff checkpoint),
  **Background servers…**, **Run project checks…**, **Prune old
  sessions/backups…** and **Reset project state…** (prune and reset preview
  before they write). A `Scout: <profile>` badge appears beside the model chip
  when a scout profile is active.

#### Profile availability in the picker

The profile picker validates local profiles against the `/api/servers` poll, which runs every
15 seconds. The hardware badge has its own faster poll of `/api/hardware`, every
3 seconds and also pushed live over SSE, so RAM and VRAM stay current on their own. A local profile whose server answers but
whose model is not in the server's model list is marked **· unavailable** and
cannot be selected; the switch API also refuses it with a 409. The tooltip
names the model and server. A profile whose server is not reachable is marked
**· server offline** and becomes selectable again on a later poll once the
server starts. Hosted profiles are never probed and stay selectable, as does a
profile whose server state is not known yet. If the active profile becomes
unavailable it stays shown with its state; the GUI never switches away from it
on its own.

Chat threads are saved with timestamps and can be renamed. With no project
open they are kept in `~/.local/state/pedacito/threads/global/` (see the
`.pedacito/` layout below); once a project is open they live inside that
project. The **Files** panel lists the indexed project files, and **Backups**
lists snapshots you can restore. When the project has `.html` pages, a
**🌐 Preview** tab joins them: a read-only live view of the selected page that
reloads when a run finishes, so a website task shows its result immediately
(see [Live Preview](#live-preview)).

### The CLI at a glance

The package installs `pedacito` and a shorter alias, `pdc`. Both accept the same
commands. Run `pedacito <command> --help` for every flag.

| Command | What it does |
|---|---|
| `pedacito gui [PROJECT]` | Open the browser control panel |
| `pedacito models scan` | Find local servers, pick models, write profiles, and measure them |
| `pedacito models [--profile NAME] [--model ID]` | Print the active (or a named) profile's tuned numbers; `--model` names the model id for that view |
| `pedacito models refresh` | Refresh the cached models.dev catalog (hosted context windows and prices) now; a failed fetch keeps the old cache |
| `pedacito config --check` | Test the connection and generation. `--doctor` also probes structured output and reports the resolved project environment. `--setup` runs first-time setup; `--upgrade [--dry-run]` adds newly documented options to your file |
| `pedacito doctor [PROJECT]` | Alias of `pedacito config --doctor`: model/structured-output probes plus the sandbox, detected test command and offline-resolved environment (interpreter, PATH, package manager, evidence). Detection never runs an environment manager |
| `pedacito calibrate` | Measure the active model (or `--all` local profiles) and save a model card that tunes defaults. Hosted models are refused |
| `pedacito profiles [--recommend]` | List profiles with their scores, and suggest role routing |
| `pedacito profile [--show\|--run\|--reset]` | Show the hardware snapshot, last speed probe and active auto-tune; `--run` forces one probe, `--reset` drops cached measurements |
| `pedacito ask PROJ -t "..."` | Run one task. Dry run by default. Add `--review`, `--apply`, `--plan`, `--refine`, `--jsonl`, or `--json` |
| `pedacito explain PROJ -q "..."` | Answer a question about the code. Never edits |
| `pedacito auto PROJ -t "..."` | Turn a goal into a checklist, then implement it task by task |
| `pedacito new DIR -t "..."` | Start a project from scratch |
| `pedacito chat PROJ` | Interactive session that keeps the index warm (`--rpc` turns the session into a machine protocol) |
| `pedacito tui PROJ` | Split-screen terminal UI: transcript, live activity/plan sidebar, and a Context Meter / token burn / tok/s telemetry footer. Full slash-command suite; `--headless` is scriptable |
| `pedacito rpc [PROJ]` | JSONL-RPC over stdin/stdout for editors and other agents: one request per line, streamed events, steering, permission prompts, cancel -- index, profiles and session warm the whole time. `--events off` keeps only terminal frames; `--describe` prints the protocol and exits |
| `pedacito acp [PROJ]` | Agent Client Protocol (JSON-RPC 2.0) over stdin/stdout: `initialize` (advertising protocol `0.4.0` and its capabilities), `session/new`, `session/prompt` with text/`@file`/image blocks, streamed `session/update` notifications and `session/request_permission` round-trips; the same warm engine as `rpc` |
| `pedacito daemon [start\|stop\|status\|sessions]` | Headless loopback-only daemon: a warm index and model leases shared by isolated workspace sessions, with HTTP/SSE attach, detach, prompt, cancel, permission and review endpoints. Bearer-token authenticated; `status` exits 0 cleanly when no daemon runs |
| `pedacito how "..."` | Built-in help for the CLI, the GUI, and settings |
| `pedacito undo PROJ` | Restore the workspace to the state immediately before the last run; `--list` shows the checkpoint tree |
| `pedacito checkpoints` | List time-travel checkpoints and the active head; `--limit` and `--json` shape the output |
| `pedacito checkout ID` | Restore the whole workspace or one `--file` from a checkpoint id (a unique prefix works) |
| `pedacito recover ACTION ...` | `list [PROJECT]` shows unfinished journals by UUID; `rollback TASK_ID [PROJECT] [--apply]` previews the bound pre-run checkpoint and writes only with `--apply`. `resume` and `cleanup` refuse when safety cannot be proved |
| `pedacito update [--run]` | Check PyPI for a newer Pedacito and optionally upgrade this install |
| `pedacito restore PROJ` | Undo the last write |
| `pedacito commit [--all] [-m MSG] [-y]` | Scan the staged diff for secrets, then create a local commit. Never pushes; `--no-model` drafts the message offline and `--allow-secrets` is the explicit override |
| `pedacito pr [--base main] [--draft] [--dry-run] [-y]` | Draft a grounded title/body from the branch diff, commit log and recent checkpoints, then open the pull request with `gh`. Never pushes the branch |
| `pedacito rename OLD NEW` | Multi-file symbol rename through the AST and the tokenizer, no model call. `--file` scopes it, `--strings` also renames whole-word string literals, `--include-labels` only test-label strings, `--apply` writes (default: diff preview) |
| `pedacito image PROMPT` | Generate one image with a configured image-model profile (`--out`, `--size WxH`, `--steps`, `--seed`, `--negative`, `--json`). Saved under `.pedacito/images/` |
| `pedacito index`, `map`, `files`, `cost` | Build and inspect the index. These need no model call, except `index` summaries |
| `pedacito diag [PATH]` | Alias of `pedacito diagnose`: run the project's detected linter or typechecker |
| `pedacito parsers prefetch [--languages ...]` | Pre-warm the tree-sitter grammars for offline or air-gapped use and report ok or failed per language |
| `pedacito eval` | Score the harness on fixture repositories. Use `--baseline` as a regression gate |
| `pedacito bench [BENCHMARK\|all]` | Run the six-project diagnostic benchmark suite and hidden holdout rubric. `--list` and `--recent` need no model; `--json` and `--strict` fit CI |
| `pedacito bench compare BASELINE CURRENT` | Gate two eval scorecards written by `pedacito eval --scorecard`: exit 0 pass, 1 regression, 2 invalid or incomparable. Needs no model; `--max-prompt-growth` defaults to 0 |
| `pedacito memory`, `skills`, `prune`, `reset`, `trust`, `worktree`, `serve`, `procs`, `lsp`, `mcp`, `diagnose`, `tests`, `webcheck`, `domcheck` | Upkeep and specialist tools. `mcp list` shows the named MCP servers' tools and trust flags |

Benchmarks also run from the GUI through the **Benchmarks** button; the CLI
command builds the same projects and writes the same reports.

### Recovering an interrupted task

Each run appends a journal under `.pedacito/journal/`.
`pedacito recover list [PROJECT]` shows every unfinished or uncertain journal
by UUID, with its title, session, status and recorded checkpoint. Pre-v0.4.1
title-named journals stay visible for inspection, but they are read-only:
recovery never guesses ownership from a title.

Every action previews first and writes nothing unless `--apply` is present.
`pedacito recover rollback TASK_ID PROJECT` previews restoring the journal's
durable pre-run checkpoint; `--apply` takes the project write lock, rechecks
maintenance safety, revalidates the journal and the checkpoint's WAL binding
under that lock, and captures dirty files in a safety checkpoint before the
restore. The safety checkpoint must be complete, and recovery verifies a
versioned manifest that records the exact snapshot paths and SHA-256 hashes:
an absent, incomplete or altered manifest, a missing or damaged snapshot file,
or a symlink/junction escape is refused before any source file is touched.
`resume` and `cleanup` refuse with an explanation in this release, because no
validated durable resume boundary or authoritative cross-process
ownership/liveness proof exists. A refusal exits 1 and leaves the journal and
your files untouched.

## Built-in help: `pedacito how`

For model selection, VRAM sizing, thinking settings, and role routing, see the
[model guide](../MODEL_GUIDE.md). It has concrete starting points for under 8 GB,
8 to 16 GB, and 24 GB of VRAM, plus hybrid local and cloud role routing.

`pedacito how` answers questions about using Pedacito from a built-in
**structured reference index**: every CLI command and flag, the exact GUI
navigation path for each feature (including ✦ Refine Prompt, the permission
dialog, web access, green ratchet rollbacks and the benchmarks), and every
config setting with its default, GUI location and `config.toml` form. It needs no project and no index. A model
connection is optional: when a fast model server is reachable it rewrites the
retrieved facts into a short answer; otherwise the deterministic GUI/CLI/setting
cards print immediately.

```bash
pedacito how "how do I disable thinking?"       # model synthesis if available
pedacito how --offline "how do I import models?" # instant, no model needed
pedacito how --json "how do I reset my config?"  # structured matches + answer
pedacito how                                      # interactive
```

It accepts `--url`/`--server`, `--model`, `--api-key`, `--vram`/`--vram-gb`,
`--config`, and `--profile`, plus `--offline`/`--fast` and `--json`. The GUI's
**How ?** composer tab exposes the same reference with topic chips, copyable CLI
commands and **Open in Settings** buttons.

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

- `pedacito files PROJ`, every candidate file and why it was kept or skipped.
- `pedacito cost PROJ`, estimated LLM calls and time for a first index
  (`--tps` sets your tokens/second).
- `pedacito map PROJ`, print the index (`--tier auto|flat|cards|dirs|ranked`).

### Multimodal chat input

Send images with a task to a vision-capable model: up to **4 images, 10 MB
each**, as PNG, JPEG, WebP or GIF.

- **CLI**: `--image PATH` (repeatable), e.g. `pedacito ask . -t "match this mockup" --image mock.png`; `new` accepts it too.
- **GUI**: paste with `Ctrl+V` / `Cmd+V`, drag and drop onto the composer, or click `📎 Attach`.
- **TUI**: `/attach [PATH | list | clear]` or `@image.png`.

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

`--refine` runs one Prompt/Spec Architect pass first: a one-line idea becomes a
five-section spec (objective, files, interface contracts, requirements, and
verification) grounded in the real file tree, and the run uses that spec. If no
model answers, a deterministic fallback writes the same sections, so the flag
never stops the run. See [../TECHNIQUES.md §17](../TECHNIQUES.md#17-the-promptspec-architect).

Other useful flags: `--steps N` (lookup limit), `--strict` (exit 1 when a task
applied edits but left a failed edit block or impacted tests still failing
after the repair rounds), `--model`, `--url`, `--profile`,
`--role planner=big` (route one job to another profile for this run),
`--auto-route` / `--no-auto-route` (specialist routing per run; see
[Specialist auto-routing](ARCHITECTURE.md#specialist-auto-routing)),
`--creative` (creative/design Builder mode for this run; see
[Creative / design mode](#creative--design-mode)),
`--config`, `--root`, `--exclude`, `--include`, `--ext`.

**Plan mode.** A task that changes three things in one context window usually
gets the first done, the second half done and the third forgotten. `--plan`
(or `plan_threshold`, which a model card sets for weaker models) gives each
change its own window, lookups, evidence checks and rollback. A task that
names five or more files is planned even when `plan_threshold` is 0, the
default, and what a strong model's card picks, because width is a
window-capacity problem, not a skill problem; an explicit `plan_threshold` in
your config still governs, and `--no-plan` opts one task out. A task written as
bullets or several clear requirements is split deterministically, with no model
call; anything else goes to the `planner` role (which may declare
`depends_on`). The plan is a markdown checklist in `.pedacito/plans/` you can
read and edit (`--review-plan` pauses for that). After a failed step the planner
is asked once to revise the remaining steps (`max_replans`).

**Machine-readable runs.** `--jsonl` prints every event (steps, model calls with
token usage, the final answer and the diffs) as one JSON object per line on
stdout and moves the human text to stderr, so an editor plugin or script can
drive Pedacito. For one closing verdict object instead of an event stream, use
`--json` (see [Headless and automation](#headless-and-automation)). Writing
runs (`--apply`) take a per-project lock
(`.pedacito/run.lock`); a second writer is refused while the first is alive.

**Steering and questions.** In `chat`, anything you type while the model is
working is delivered at its next step ("stop reading tests, it's in
`sync.py`") without disturbing the cached prompt prefix (`allow_steering`). In
an interactive `ask`/`chat` the model also gets a `question` action for genuine
ambiguity, at most twice per task; in `auto`, `--jsonl` and `--json` it never
asks and states its assumption instead.

Every write snapshots the originals first; `pedacito restore PROJ` undoes the
last one.

### `explain`

Answers questions about the indexed project in prose, with no edits. It never
builds an index; run `pedacito index` first.

```bash
pedacito explain PROJ -q "trace the flow from answer to schedule"
pedacito explain PROJ                   # interactive
```

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

The `pattern` grammar also covers `hold ArrowUp 1.5` (a key kept DOWN for N
seconds, an instant `press` can fall between two samples of a fixed-timestep
game loop), `click #start`, `fill #name Ada`, `type hello`, and `eval` (JS in
the page). This reliably catches black screens, blank pages, missing text, and
console errors; it does not judge visual quality. PNGs are saved under
`.pedacito/screenshots/`. For your own eyes, the GUI's 🌐 Preview tab renders
the project's HTML live as a run writes it (see [Live Preview](#live-preview)).

### Session consolidation

Long sessions fill the context window. Consolidation distils the recent
history into one structured handoff of under 400 tokens — **Goal & Context**,
**Files Modified & Key Decisions**, **Verified Test Status**, **Open Next
Steps** — built from harness facts (tasks, applied/failed edits, test results,
decisions), never invented prose. It saves `.pedacito/consolidated_session.md`,
pins durable decisions to `memory.md`, and resets the live buffer so the next
turn starts from the handoff alone:

- **GUI**: the **Consolidate Session** button in the telemetry footer (or
  **Project ▾ → Consolidate session…**). The transcript restarts from the
  handoff card and the Context Meter updates immediately.
- **CLI chat**: `/consolidate` (alias `/compact`); the next task is seeded
  with the handoff. `pedacito memory PROJ --consolidate` runs the same
  distillation over the newest session transcripts and prints the checkpoint.
- **RPC**: `{"id": 1, "op": "consolidate"}` (or `"method": "consolidate"`,
  alias `compact`) answers with `before_tokens`, `after_tokens` and the
  handoff; refused while a task runs.

## Headless and automation

`pdc ask` is already the non-interactive one-task entry point: it needs no TTY,
takes the task on the command line, and leaves state under `.pedacito/`.
`--json` turns the run into one machine-readable verdict for CI and eval
harnesses; `--jsonl` remains the streaming alternative (one event per line).
`--json` and `--jsonl` are mutually exclusive and are rejected together with a
clear error.

```bash
# One summary object on stdout; every human line goes to stderr.
pdc ask . -t "add retries to the sync client" --apply --json

# Same, with a one-run verification command; verify_passed reports its result.
pdc ask . -t "fix the failing parser tests" --apply --json --verify-cmd "pytest -q"
```

The stdout shape is exactly this object (no extra keys, no prose, one object at
run end):

```json
{
  "ok": true,
  "model": "qwen2.5-coder-14b",
  "provider": "LM Studio",
  "turns": 7,
  "elapsed_sec": 12.345,
  "tokens": {"prompt": 8123, "completion": 1456, "tok_per_sec": 23.4},
  "files_changed": ["src/sync.py"],
  "verify_passed": null,
  "diff": "--- a/src/sync.py\n+++ b/src/sync.py\n@@ -1,3 +1,4 @@\n...",
  "diff_truncated": false
}
```

| Key | Type | Meaning |
|---|---|---|
| `ok` | bool | Task success. Not the exit code: a run without `--strict` can exit 0 while a red verification or a failed edit block makes `ok` false. The process exit code keeps its existing semantics. |
| `model`, `provider` | string | The model and endpoint label that answered, from the run telemetry. |
| `turns` | int | **Model calls** in this run, exactly `events.RunTelemetry.summary()["calls"]`: every gather step, answer, planning and repair request counts once. |
| `elapsed_sec` | float | Model-call time accumulated by the run telemetry (`elapsed_s`), not wall-clock time. |
| `tokens.prompt` / `tokens.completion` | int | Prompt and completion tokens across those calls. |
| `tokens.tok_per_sec` | float | Completion tokens per second of model time. |
| `files_changed` | list[string] | Project-relative paths whose bytes the applied run changed (empty on a dry run). |
| `verify_passed` | bool or null | The run's verification outcome. `null` means no verification ran. With `--verify-cmd` (or a configured `auto_verify`) it is that command's result; otherwise it is the post-edit impacted-test verdict when those tests ran. |
| `diff` | string | Combined unified diff of the applied changes, capped at 200,000 characters. |
| `diff_truncated` | bool | `true` when the diff was cut at that cap. |

`--json` implies `--no-review`: it never opens the review page or a pager and
never prompts for a plan, a question or a permission decision. The task text
itself is never echoed into the object, and `diff`/string fields pass the same
`redact_secrets` masking as logs, so a secret planted in the task cannot come
back out on stdout (set `redact_secrets = false` only if you deliberately want
unmasked output).

`--verify-cmd "COMMAND"` overrides `auto_verify` for this single run **in
memory only**: the live config gets the command plus a `sources` entry of
`--verify-cmd`, and no config file is written or rewritten. It runs once after
an applied task; a skipped dry run therefore leaves `verify_passed` null.

### Saving trajectories for SFT/GRPO

`--save-trajectory` writes one full-prompt JSON transcript per task run to
`.pedacito/trajectories/`, for dataset filtering and fine-tuning. It is the
opt-in counterpart to `replay` (which stores only a hash and a short redacted
preview), so it contains the real prompts; every string is redacted and the
file is re-scanned for secrets before it is written. The same capture is
available everywhere through `record_trajectories = true`; the flag turns it
on for this `ask` run only without touching the config file.

```bash
pdc ask . -t "fix the retry loop" --apply --save-trajectory
```

Each file is `{"metadata": {...}, "messages": [...]}`, where `messages` is the
ChatML/OpenAI-shaped transcript actually exchanged (system/user/assistant
turns, tool calls and tool results). Keep only the runs whose verification
passed before building a dataset:

```python
import json
from pathlib import Path

kept = []
for path in Path(".pedacito/trajectories").glob("*.json"):
    record = json.loads(path.read_text(encoding="utf-8"))
    meta = record["metadata"]
    if meta["verify_passed"] is True:          # null = no check ran
        kept.append({"messages": record["messages"], **{
            key: meta[key] for key in (
                "task_hash", "model_id", "provider", "edit_format",
                "harness_version", "started_at", "elapsed_s", "tokens")
        }})
Path("dataset.jsonl").write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in kept),
    encoding="utf-8")
```

`verify_passed` is the task's own post-edit impacted-test verdict
(`true`/`false`/`null`); `null` means no in-task check ran, so `is True`
deliberately excludes unverified runs. A `--verify-cmd` runs once after the
task, so its result is reported by `--json`; the trajectory records the
in-loop verification that existed when the task finished. The full metadata
contract and the privacy trade are in
[SECURITY.md — Trajectory export](SECURITY.md#trajectory-export).

## Pinning context with `@`

Mention a file or a symbol with `@` in your prompt, and Pedacito pins its exact source into the
run. The pinned context is carried into every phase of the task (lookup, answer and edit), so the
model cannot miss it or quietly work from a different version.

```text
Make @src/app.py retry on timeout, and keep the signature that @Store.save uses.
```

### What `@` accepts

| Form | Resolves to |
|---|---|
| `@path/to/file.py` | An exact relative path |
| `@file.py` | A unique basename, when only one file has that name |
| `@app` | An unambiguous case-insensitive prefix or fuzzy match on a file name |
| `@Class.method` | An exact qualified name |
| `@file.py:symbol` | A symbol in a specific file |
| `@symbol` | A unique indexed member name |

Two things are ignored on purpose: email addresses (`name@domain.com`), and decorator lines in
fenced Python blocks. A mention that matches no unique file or symbol is not pinned.

### Manage pins in the TUI

`/pin` shows the pins for this session. `/pin add REF` adds one explicitly, `/pin remove REF`
removes one, and `/pin clear` removes them all.

### Skills: catalog on demand, and manual injection

Skills are markdown checklists with YAML frontmatter (`name`, `description`, optional
`triggers`). Pedacito keeps the catalog in the prompt (names and one-line descriptions) and
loads a full body only when the model asks for it with the `skill` action. Up to two trigger
matches are selected automatically for a task. Project skills override user skills, which
override the bundled library; `pedacito skills list|show|add|draft|rm` manages the files.

To force one into the next task yourself, without waiting for a trigger match:

```text
/skill add-a-test          # TUI: load the full body into the next task
```

The body is validated now, then carried through the same **Explicitly Pinned Context** block
as `@file`/`@symbol` pins, above the map, and it is never evicted by token pruning.
`/skills` lists names, `/pin remove skill:add-a-test` unpins one, and `/pin clear` unpins
everything. In the GUI open **Skills library & creator**, use **Use in next task** on any
skill; the pinned list in the dialog (and the 📌 Pinned row above the composer) shows what
will be injected, with Remove and Clear skills actions. A skill pin is a normal
session pin (`skill:add-a-test`), so it rides every task until you remove it.

## More commands

### `chat`: interactive session

```bash
pedacito chat ~/code/anki [--apply | --review]
```

Keeps the index warm between tasks. Slash commands: `/profile NAME`,
`/model ID`, `/apply on|off`, `/steps N`, `/status` (pack, answer mode,
verifier, window, model card), `/cost` (tokens, cache hits and spend by call
site), `/memory`, `/skills`, `/consolidate` (alias `/compact`: compact the
session into a handoff; see [Session consolidation](#session-consolidation)),
`/profiles`, `/help`, `/quit`. **Ctrl-C cancels the
running task but keeps the session open.** Switching profile mid-session is the
cheapest way to compare two models on one task; a failed switch leaves the
session on the previous model.

With `--rpc` the same warm session speaks a machine protocol instead of prose:
stdin takes one JSON request per line and stdout answers with JSONL events --
see `pedacito rpc` below.

### `tui`: split-screen terminal session

```bash
pedacito tui ~/code/anki                 # apply-on by default, with undo backup
pedacito tui . --no-apply                # preview edits only
pedacito tui . -t "add retries to the sync client"
pedacito tui . --headless < commands.txt  # scriptable CI mode, no PTY
```

The TUI keeps the index warm like `chat`, but paints four zones instead of a
scrolling REPL: a header (project, profile/model, calibration badge,
Auto-Route, Scout, git branch/dirty state), a main transcript pane beside a
collapsible **Activity & Plan** sidebar, a two-row telemetry footer, and the
`pdc tui (profile) >` prompt bar. The telemetry footer repaints live: a 150 ms
watcher tails the project's event log while each task runs on a worker thread.
Text from repo files, logs and tool results is stripped of terminal escape
sequences before it is painted, so a file cannot retitle your terminal or write
to your clipboard. The sidebar appears at 100+ columns and
toggles with `/sidebar`; it shows `PLAN.md` checkboxes, live `read`/`grep`/
`mcp_call`/`task` steps, specialist `model_swap` routing, and the files this
session changed.

Interactive input is a cbreak/`msvcrt` raw reader, so navigation keys edit the
prompt instead of echoing `^[[5~`: `PgUp`/`PgDn` scroll the transcript half a
page, `Shift+Up`/`Shift+Down` scroll three lines, `Alt+PgUp`/`Alt+PgDn` scroll
the Activity/Plan sidebar, and the mouse wheel scrolls three lines. The wheel
uses xterm Alternate Scroll (`?1007h`) by default, so left-click-drag selection
and the OS copy/paste keep working; `/mouse on` (or `F2`) enables optional SGR
wheel capture and `/mouse off` restores native selection. `Up`/`Down` recall
command history, `Left`/`Right`/`Home`/`End` edit the prompt, `Tab` completes
slash commands and profile names, and `Esc`/`End` jump back to the live bottom.
A scrolled-back transcript ([ ↑ Scrolled back N lines ... ]) stays stationary
while background log lines arrive. Repaints use DEC synchronized output and
place the hardware cursor inside `pdc tui (profile) > ...`.

**Multi-line input.** `Shift+Enter`, `Alt+Enter` and `Ctrl+J` insert a newline in
the prompt; a trailing `\` followed by `Enter` does the same on terminals that
cannot tell `Shift+Enter` apart. Pasted text arrives through bracketed paste
(`?2004h`), so a multi-line paste stays in the prompt as one block and is not
run line by line. Plain `Enter` submits.

**Image attachments.** `/attach [PATH | list | clear]` (or `@image.png` in the
prompt) stages PNG/JPEG/WebP/GIF images for the next task; `list` shows what is
staged and `clear` drops it. See [Multimodal input](#multimodal-chat-input).

**Rich view (v0.3.3).** Still zero-dependency. Edits render as inline
syntax-highlighted unified diffs, and every tool call is a collapsible one-line
card: `Ctrl+O` toggles the latest card, and `/expand [N|all]` and
`/collapse [N|all]` work on any card. Type `@` in the prompt to open an
interactive six-row fuzzy picker over project files, symbols and images: Up and
Down move through it, Tab or Enter inserts the path, and Esc dismisses it. SGR
mouse zones are clickable too: `[EN|es]` switches language, the sidebar tabs
switch panels, a tool card expands or collapses, and the scrollbar jumps. Mouse
capture stays opt-in (`/mouse on`, or `F2`).

The footer updates after every model call and event: a green/amber/red
**Context Meter** (`Ctx: [████████░░░░░░░░] 6,420 / 32,768 (19.6%)`), turn and
session **Burn Rate** (token burn rate), `Speed: 44.2 tok/s (avg 41.8 tok/s)`,
KV cache hit rate, fold count, and `Active: <profile> [<role>/<specialty>]` with `$cost`
when priced. `pedacito tui` registers the live permission broker, so an untrusted
MCP tool, sensitive file read or shell command asks in the prompt bar:
`y` allows once, `a` allows that key for the session, `n` denies.

Slash commands: `/help`, `/keys`, `/status` (`/telemetry`), `/new`, `/auto GOAL`,
`/auto! GOAL`, `/plan [GOAL]`, `/consolidate` (`/compact`), `/profile [NAME]`,
`/profiles`, `/model ID`, `/auto-route [on|off]`, `/think [on|off]`,
`/effort [low|medium|high]`, `/creative [on|off|auto]`, `/servers`,
`/unload [NAME]`, `/apply [on|off]`, `/verify on|off|status`, `/skills`,
`/skill NAME`, `/memory`, `/procs`, `/security`, `/theme`, `/pin`, `/rules`,
`/hybrid`, `/arena`, `/diff`, `/expand [N|all]`, `/collapse [N|all]`, `/git [status|stage PATH|unstage PATH|commit
[MSG]|branch NAME]`, `/checkpoints`, `/undo`, `/checkout ID`, `/run`,
`/fix`, `/image PROMPT`, `/update`, `/reset`, `/sidebar`, `/scroll up|down|top|bottom|N`,
`/mouse [on|off]`, `/clear`, `/quit` (`/exit`). A plain line is a
task and runs through the normal gather/edit/verify loop. `/creative on` forces
the Builder persona for every task, `auto` (the default) detects design tasks,
`off` keeps surgical edits everywhere; `/verify` controls the post-edit
verification command (see [the three tiers](#the-three-verification-tiers-as-shipped));
`/status` shows the current state.

### `rpc`: drive pedacito from an editor or another agent

```bash
pedacito rpc ~/code/anki          # or: pedacito chat ~/code/anki --rpc
echo '{"id":1,"method":"status"}' | pedacito rpc ~/code/anki
```

Bidirectional JSONL-RPC over stdio, the same engine as `ask` and `chat`: each
stdin line is `{"id": ..., "method": ..., "params": {...}}`; every stdout line
is one JSON frame, so a client never parses prose. `ask` streams the task's
events tagged with its `req_id` and ends with a `result` frame (`applied`,
`changed`, `ok`, cost, and `blockers`: the gate findings the run ended on --
`applied: true` beside non-empty `blockers` means the code landed and was not
verified clean, which is the difference a calling agent needs); `steer` pushes
a note into the running task's queue; `permission_request` events arrive when
a gated tool wants to act and are answered by `permission_reply` (`allow_once`,
`allow_run`, `allow_always`, `deny`); `cancel` stops the task and keeps the
process warm; `consolidate` (alias `compact`, also accepted as `"op"`)
compacts the session into a handoff checkpoint and resets the resume buffer,
so the next `ask` starts from the handoff alone; `switch_profile` and `status`
work mid-session too. The index,
the profiles and the session ledger stay warm across requests -- that is the
whole point. Human-facing diagnostics go to stderr; stdout stays pure JSONL.
Protocol details in `pedacito/rpc.py`, rationale in
[Techniques §31](../TECHNIQUES.md#31-bidirectional-stdio-jsonl-rpc-a-warm-engine-an-editor-can-drive).

### `acp`: Agent Client Protocol bridge

```bash
pedacito acp ~/code/anki                                  # editor or agent host over stdio
pedacito acp ~/code/anki --profile fast --execution-mode interleaved
```

`pedacito acp` speaks the Agent Client Protocol (JSON-RPC 2.0) on stdin and
stdout, for editors and agent hosts such as Zed. It wraps the same warm engine
as `rpc`, so nothing about the harness is duplicated. `initialize` advertises
the capabilities, and `session/new` binds a project (a foreign working
directory is refused rather than read) with an optional profile and execution
mode. `session/prompt` accepts text, `@file` resource and image blocks; `@file`
paths go through the same containment check as every other read, so traversal
and symlink escapes are refused. Tasks stream `session/update` notifications
(`agent_message_chunk`, `agent_thought_chunk`, `tool_call`, `tool_call_update`,
`plan`), and gated tools round-trip through `session/request_permission`.
`session/cancel` maps to the rpc cancel and keeps the process alive. stdout is
pure JSON-RPC; every human line goes to stderr, and stdin EOF drains the active
task and exits 0.

### `new`: start a project from scratch

```bash
pedacito new my-app -t "a CLI todo manager with a sqlite backend" --apply
pedacito new my-api --spec docs/spec.md --review
pedacito new my-app -t "..." --plan-only --apply     # skeleton only
pedacito new my-app -t "..." --apply --no-complex    # one pass, no Complex escalation
```

`--no-complex` (alias `--single`) turns off the automatic escalation to Complex
scaffolding for prompts that name several modules. `--complex` now defaults
`verify_final` on, and a scaffold whose pytest run collects no tests (exit 5)
is treated as a valid scaffold rather than a failure. The starter
`pyproject.toml` includes `[tool.pytest.ini_options] pythonpath = ["."]`, so
`pytest` finds the package from the project root.

The model plans a file tree, the skeleton is written, then each file is filled
in. Dry run by default. The target directory must be empty or new, or hold
nothing but Pedacito's own files (`.git`, `.gitignore`, `.pedacito/`,
`.pedacito.toml`, `.pedacitoignore`), unless `--force`. `--complex` hands
implementation to the `auto` loop, one task per planned file: test files are
implemented after the sources, so each is written against the real API.

Each file is written by its own call, so the finished tree is checked as a
whole before the build reports success: a browser scaffold whose scripts cannot
link, or whose `getElementById` names an id no template renders, gets repair
rounds and then prints `BUILD INCOMPLETE` and exits 1. A web project's generated
`README` also says how to actually start it -- ES modules are refused over
`file://`, so it names `python -m http.server 8000` and the URL to open instead
of a `python main.py` line for a file the project does not contain. For a
website or UI-heavy scaffold, `--creative` runs every task in creative/design
mode (see [Creative / design mode](#creative--design-mode)).

### `auto`: autonomous loop over a task list

Breaks a goal into an ordered checklist and implements each task in a fresh
context, with optional verification and repair:

```bash
pedacito auto . -t "add retry logic to the sync client"      # plan only
pedacito auto . -t "..." --apply                             # plan, then implement
pedacito auto . --apply --verify "pytest -q"                 # verify and repair each task
pedacito auto . --apply --verify "pytest -q" --verify-final  # one whole-goal check at the end
pedacito auto . --apply --diagnostics                        # lint/typecheck each task
pedacito auto . --apply --worktree refactor                  # run in an isolated git worktree
pedacito auto . --apply --plan TODO.md                       # use an existing checklist
pedacito auto . -t "restyle the buttons in index.html" --force-auto  # skip the Task-mode nudge
```

**Task-vs-Auto suitability nudge.** Before spinning up a multi-step `PLAN.md`,
`auto -t "..."` runs a fast, deterministic check (`evaluate_mode_suitability`,
zero model calls): a prompt that targets 1–2 existing files, uses localized
edit verbs (`fix`, `restyle`, `rename`, `remove`, `add a link`, …), or tweaks
content/styling in a tiny project (≤5 non-ignored source files) looks like
single-pass **Task** work. On a terminal, Pedacito prints the recommendation
and asks `Run as a single-pass Task instead? [Y/n]` — Enter or `y` runs the
whole prompt as one `ask` pass (~20s, no checklist); `n` keeps the Auto loop.
A non-TTY stdin (CI pipe) is never blocked: it logs a one-line note and
proceeds with Auto. `--force-auto` (or `-y`/`--yes`) bypasses the check, and
prompts that say "multi-step plan", "create a plan", "from scratch", or name
≥4 distinct new files/modules always keep Auto. The TUI mirrors this on
`/auto GOAL` (`/auto! GOAL` forces the loop), and the GUI shows a live
"💡 Looks like a single-pass edit" banner above the composer plus a
"Run as a single-pass Task instead?" choice on Run, with **Run as Task
(Recommended)** and **Run Auto Anyway** buttons.

The plan is a markdown checklist (`PLAN.md` by default), so a run is resumable
and editable; the file also keeps the full goal (a `Goal:` line with `> `
continuation lines), which a resumed run hands back to each task. There is
no default cap on tasks, steps, or time; add one with `--max-tasks N`,
`--max-minutes N`, or `--steps N`. `--max-minutes` bounds the loop's own clock:
planning, scaffolding and per-step gate subprocesses run outside it, so a
caller that needs a wall-clock ceiling should keep its own margin.
`--repair N` sets repair
rounds per failed verification, `--keep-going` continues past failed tasks, and
`--auto-install` installs a missing Python package instead of reporting it.
`--verify-final` runs the `--verify` command once, after the last unchecked
task, instead of after every task: a whole-goal check cannot pass mid-plan, so
the per-task runs would roll back valid partial fixes (`auto_verify_final`,
off by default). After the last task, `--apply` also exercises the project
once (`auto_smoke`, on by default): the Python entry points are exec'd and
the receipt says `verify: smoke passed` -- that proves the project BOOTS, not
that the suite is green, so a plan should end with a run-the-tests task or
you should pass `--verify "pytest -q"`. With no `--verify` and no `main.py`
to smoke-run, the receipt says `verify: not run` instead of claiming a pass. `--strict` exits 1 when a task applied edits but left a
failed edit block or impacted tests still failing after the repair rounds,
for CI and unattended pipelines (without it both are reported and the run
exits 0 -- "reported" means the red verdict is named where you read the
result: an `ask` closes with a `[verification: RED ...]` note listing the
blockers the repair did not clear, and the `auto` finish line counts the
tasks that ticked done while the last gate round was red, the way it
counts partial applications; ../TECHNIQUES.md §33).

Guards for unattended runs: a repair round that makes the check *worse* (more
test failures, more findings) is rolled back from that round's own snapshot
and the task is marked failed, instead of compounding the damage. Plan titles
are linted at startup, one title should be one change, and a title bundling
several edits gets a warning. `auto_allowlist = true` restricts each task's
writes: existing files can only be modified, moved, or deleted when the task
title names them or the lookup phase read them; creating new files is always
allowed.

Three more guards run without a model call:

- **Interface lock** (`interface_lock`, on by default). Before each task, the
  project's public signatures (`def parse_line(text, sep=',')`, class
  constructors, JS/TS functions) are injected into the prompt under a
  "do not break signatures" heading, together with a `conftest.py` fixtures
  subsection (exact fixture signatures and scopes) whenever the repository
  has fixtures or the task touches tests. After each applied task, every
  local import and call -- Python and JS/TS ES-module exports alike -- is
  checked against those signatures. A mismatch fails the task
  before any test runs, and the exact message goes into the repair prompt.
- **Static gate** (`static_gate`, on by default). Changed Python files get a
  fast check for undefined names and syntax errors (`ruff` rules F821, F822,
  F823, F402, F404, with E999 for syntax). Ruff gets a 3-second ceiling
  (`static_gate_timeout`), and a built-in checker takes over if ruff is missing or
  slow. Only errors the project did not already have count. And when a new
  undefined name provably has exactly one import -- a standard-library module,
  or a symbol only one local file exports -- Pedacito inserts that import
  itself before the model is ever asked (`auto_heal_imports`, on by default).
- **Green ratchet** (`auto_green_ratchet`, on by default). Before each step,
  Pedacito snapshots the source files into `.pedacito/checkpoints/`. A step that
  still fails after its repair rounds is rolled back to that snapshot, and the
  files it created are removed. The next step starts from the last passing state,
  and the Activity log says `ROLLED BACK`. See
  [../TECHNIQUES.md §18–§20](../TECHNIQUES.md#18-contract-first-interface-locking--cross-file-verification).

#### The three verification tiers, as shipped

Edits move through three tiers, cheapest first:

1. **Syntax and static gate (in-process, no model call).** The edited Python
   files are parsed, then checked with `ruff` rules `F821`, `F822`, `F823`,
   `F402`, `F404` and `E999`; a built-in AST resolver takes over when ruff is
   missing, fails, or exceeds its ceiling. The ceiling is
   **`static_gate_timeout = 3.0` seconds** for one ruff run. Only findings the
   edit introduced are reported: the project's pre-edit state is the baseline,
   so old breakage never vetoes a new task. `auto_heal_imports = true` inserts
   the one import that provably fixes a new undefined name before the model is
   asked.
2. **Scoped impacted tests.** When `verify_tests = true` and the test
   subprocess is allowed (`allow_shell`, or a user-granted test command such as
   `auto_verify`/`test_command`), the import graph selects the tests a change
   can break — tests named after the file, tests of its importers/callers, and
   direct importers — and runs a narrowed `pytest FILE...`, `npm test --
   FILE...` or `go test ./dir/...`. One run is bounded by **`test_timeout =
   300` seconds**. The full suite is deliberately not run here.
3. **The full suite is your verify command.** `auto_verify` (or `--verify` on
   one run) is the user-chosen command run after each Auto task; with
   `auto_verify_final = true` it runs once after the last unchecked task
   instead. A failure feeds its output into **`auto_repair_rounds = 2`** fix
   rounds. After an `auto --apply` run, **`auto_verify_rounds = 2`** bounds the
   runtime verification. The key is a command string: empty means off.

The TUI's `/verify on|off|status` and the GUI's **Post-edit verify** checkbox
(plus the Auto Options → verify command field) control tier 3. `on` leaves an
existing command alone; with an empty `auto_verify` it first asks
`testsel.detect_test_command` for the project's runner and reports what it
found. When nothing is detectable the command stays empty, while tiers 1 and 2
and the interface lock still run. The green ratchet
(`auto_green_ratchet = true`) rolls a failed Auto step back to the last
verified-green snapshot, and `interface_lock = true` checks cross-file imports
and call arity after every edit.

`--worktree NAME` runs in `.pedacito/worktrees/NAME` on a `pedacito/NAME`
branch, leaving your checkout untouched. Manage worktrees directly with
`pedacito worktree create|list|remove NAME`.

### git: commit and pull requests

`pedacito commit` is opt-in: it never stages or commits without you asking, and
it refuses when `security.scan_text_for_secrets` finds a credential in the
staged diff. Review the message, then confirm (or pass `-y`). `--all` stages
every change first; `--no-model` drafts the Conventional Commit message
deterministically, with no server call; `--allow-secrets` is the explicit
override for a confirmed false positive. The commit is local — Pedacito never
runs `git push`.

```bash
pedacito commit --all                       # draft a message from the staged diff
pedacito commit -m "feat: add login" -y     # commit; still scans first
```

`pedacito pr` uses the same secret scan on the branch diff, gathers the commit
log and recent checkpoint labels, and drafts a grounded title and body.
`--dry-run` prints them without calling `gh`; `--draft` opens a draft PR.
It requires the GitHub CLI (`gh`) and never pushes the branch itself.

```bash
pedacito pr --base main --dry-run
pedacito pr --base main --draft -y
```

The TUI exposes the local half as `/git [status|stage PATH|unstage PATH|commit
[MSG]|branch NAME]`, and the GUI's **Git** button (or **Project → Git: commit &
pull request…**) offers stage/unstage rows, a message box with **Draft message**
and **Commit**, and a secret-scan warning banner. Both call the same `gitops`
functions and refuse on findings. The GUI Git drawer and `/git` never push, and
`commit` and `pr` refuse to proceed on a secret-scan match.

### Offline parsers: `parsers prefetch` and `diag`

```bash
pedacito parsers prefetch                          # the common grammar set, while online
pedacito parsers prefetch --languages python go    # just these grammars
pedacito diag .                                    # alias of `pedacito diagnose`
```

Tree-sitter grammars load on first use. `parsers prefetch` warms them ahead of
time and prints `ok` or `failed` for each language. A grammar that cannot be
fetched is retried and then falls back to the built-in parser, so an air-gapped
machine keeps working. `diag` is the short name for `diagnose`, the project's
detected linter or typechecker, and it takes the same `--file` scope.

### Windows PowerShell 5.1: capturing stderr (`2>&1`)

Pedacito writes its human text to stderr, and under `--jsonl`, `rpc` and `acp`
stdout must stay pure JSON. On Windows PowerShell 5.1, `2>&1` wraps every
stderr line as an error record, so a successful run can print red
`NativeCommandError` text and a piped JSON stream can be corrupted. For clean
machine output, discard stderr (`pedacito ask . -t "..." --json 2>$null`), or
run the command under PowerShell 7 or `cmd.exe`, which merge stderr as plain
text.

### `gui`: browser control panel

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
| **Chat** (left) | Five modes next to the box: **Task/Edit** (edit code; *review* or *write directly*, planning auto/always/never, max lookups), **Ask** (read-only questions, answer streams in), **Auto** (goal → plan → implement; plan-only or implement, verify command, repair rounds, task/minute caps, worktree, keep-going, pip-install; applying Auto accepts the tool trust boundary and enables write/run tools for the run), **New** (start a project from scratch in a folder: preview, build, build-then-review, skeleton only, and *complex* for the autonomous loop), and **How ?** (the built-in CLI/GUI/setting reference with topic chips, copyable commands and *Open in Settings* links; answers stay out of the saved coding thread). **Attach** (or drag in) a text file as the task, question, or spec. **✨ Creative** above the composer forces the Builder persona for Task/Auto/New runs; with it off, design tasks are still auto-detected. While a task runs, type to **steer** the model, answer its clarifying questions inline, or **Stop** it. |
| **Changes** (middle) | The proposed edits as *Old code \| Updated code* side by side with word-level highlights. Accept or reject the whole set, untick files or single hunks to accept only some, or **Edit updated** to hand-edit a file before accepting. Files carry clause badges (`R1`, `R2`, ...) showing which part of the task they serve; a partial accept flags clauses left unaccepted. Backups are made first and a file that changed on disk since the preview is refused. |
| **Activity** and **Plan** (right, top) | Live phase and activity for the running task. The `auto`/plan checklist with progress; click **Edit** to change it, then **Run** to implement it. |
| **Files**, **Backups**, **Logs** (right, bottom tabs) | Files: a filterable tree of the indexed files, with files in the pending review highlighted. Backups: restorable snapshots. Logs: the terminal, everything pedacito prints, live, with ANSI colours. |
| **🌐 Preview** (right, bottom tab) | Appears when the project has `.html` pages: a read-only iframe of the selected page (default `index.html`), served from inside the project root. ⟳ Refresh reloads it, ↗ New tab opens it in a full browser tab, and it reloads automatically when a run finishes or changes files. See [Live Preview](#live-preview). |
| **File preview** | Clicking a file in the Files tab opens a read-only preview modal. |
| **Settings** | See below. |

Panels resize (drag the dividers), the layout and colour scheme are
remembered, and on a narrow window the panels become tabs.

**Settings** edits the user config or the open project's `.pedacito.toml`
without touching the rest of the file, comments survive, every edit is checked
with the same rules the loader uses before it is written, and the previous file
is kept as `config.toml.bak`. Tabs: **Profiles** (create, duplicate, delete,
make default; server URL, key, model with a *Test connection* that lists the
server's models, tier, VRAM, JSON mode, and the common switches; a **Dual-GPU /
Scout + Architect Pairing** form writes `default_profile` + `scout_profile` in
one comment-preserving edit and can create a starter scout profile for the
second GPU), **Model discovery**, **Auto-Tune** (hardware, model class,
codebase size, priority and a dual-GPU question; a dual-GPU answer recommends
a scout and offers *Create starter Scout & pair…*), **All settings** (every
setting, searchable, with its help text and where its value comes from,
applied globally or to one profile), **MCP Servers** (add, edit, test, enable,
trust and delete named `[mcp_servers.<name>]` servers), and **Raw file /
reset**. With no config at all, a first-run form (with *Find servers* for LM
Studio/Ollama) builds one. A project config's execution settings stay ignored
until you trust it (*Review & trust…*, the same as `pedacito trust`).

How it stays safe: it refuses requests with a foreign `Host`/`Origin`, accepts
only JSON `POST`s, sends a strict Content-Security-Policy, and serves a project's
files only from inside that project. API keys are never shown in the profile
forms (leave the box blank to keep one), and the **Raw file** tab masks every
stored secret as `***REDACTED***`: saving the mask unchanged restores the
stored value, while a mask for a key the file does not have is refused.
Sensitive files (`.env*`, keys, credentials) are refused in the file preview
too. Only one job runs at a time, and jobs that write directly take the
same project lock as the command line.

### Live Preview

When the open project has at least one `.html` file, the workspace panel grows
a **🌐 Preview** tab beside Files / Backups / Logs: the page itself, rendered
in a read-only iframe (`#live-preview-frame`) while you work on it.

- Pick the page in the dropdown (`index.html` is the default). Relative CSS,
  JS and image links resolve through the same read-only route, so the page
  looks the way it will when served.
- **⟳ Refresh** reloads the page; **↗ New tab** opens it in a full browser tab.
- The preview iframe is sandboxed (`allow-scripts`, no same-origin access): scripts
  run, but the page cannot read the GUI's storage or drive the agent API. Use
  **↗ New tab** when you want the page to behave as a normal same-origin document.
- The preview reloads automatically when an Ask/Auto/New/Apply run finishes —
  and while a run is writing, when the preview tab is open — so a website task
  shows its result immediately.

The route is `GET /preview/<path>` and never leaves the project root: `..`
segments and symlinks that escape are refused, sensitive files are refused the
same way as the file preview, and only the app's own origin may frame the page
(`X-Frame-Options: SAMEORIGIN` plus a `frame-ancestors 'self'` CSP). It is
read-only: nothing the page does can write back through it.

### Measurement and upkeep

```bash
pedacito calibrate [--quick] [--repeats N]   # measure the model; save a model card
pedacito calibrate --all [-y] [--recalibrate]  # every local profile, one after another
pedacito eval [--task ID] [--out report.json]   # score the harness on fixtures
pedacito map PROJ -t "fix X"       # see the map a task would get (ranked)
pedacito tests PROJ --files src/a.py   # detected test command + impacted tests
pedacito webcheck PROJ             # static contract checks for web files
pedacito domcheck PROJ             # headless DOM smoke check (needs Playwright)
pedacito memory PROJ [--add T | --forget T | --consolidate]
pedacito skills [list | show NAME | add NAME | draft "PROMPT" | rm NAME] [PROJ] [--global]
pedacito prune PROJ --keep-sessions 50 --keep-backups 20 [--dry-run]
pedacito reset PROJ [--dry-run] [--keep-memory] [--keep-config] [--reindex]
pedacito trust PROJ                # approve a project's .pedacito.toml
pedacito ask PROJ -t "..." --record          # save the model's responses
pedacito ask PROJ -t "..." --replay FILE     # re-run the harness, no model needed
pedacito ask PROJ -t "..." --replay FILE --strict-replay  # fail on prompt drift
pedacito ask PROJ -t "..." --continue        # seed the task from the newest ask checkpoint
pedacito eval --out new.json --baseline base.json         # CI A/B regression gate
pedacito eval --seed 1 --scorecard card-before.json       # schema-v2 scorecard (full task set, explicit seed)
pedacito bench compare card-before.json card-after.json   # scorecard gate: 0 pass, 1 regression, 2 invalid/incomparable
pedacito eval --repeats 3 --seed 1 --out new.json   # 3 seeded runs: pass counts, mean ± spread
pedacito new app -t "..." --design           # interfaces/data/tests before code
pedacito serve --root PROJ --name web -- npm run dev
pedacito procs --root PROJ [logs|stop NAME] [--all]
pedacito lsp --root PROJ --file src/a.py --line 12 --op definition
```

Optional diagnostics gate (off by default; needs `allow_lsp = true` and
`lsp_diagnostics_gate = true`, which a project file must have trusted): for a
symbol-batch edit, Pedacito asks the language server for diagnostics on the
files before and after the edit, within one 2.9 s budget, and sends only new
findings on changed lines to repair. If the server is missing, slow, silent, or
cannot prove its diagnostics match the current file, the gate reports
*unverified* and repairs nothing. It is an event (`lsp_gate`), not a pass.

`webcheck` runs static cross-file contract checks on a web project, no browser
needed: DOM ids that exist in no HTML file, ESM syntax loaded through a classic
`<script src>` (a silent SyntaxError), duplicate globals across classic
scripts, `<script src>` files that do not exist, bare imports package.json
never declared, inline handlers calling members nothing defines, a module-level
name bound twice (`Identifier 'X' has already been declared`, which kills the
whole module graph), and a call whose callee no module in the graph binds. It is
regex-level on purpose, so its findings are warnings (printed by `auto` after
tasks that touched web files, and in the `new` scaffold summary), never hard
failures. Those last two shapes are also enforced by the vetting frontend gate,
which reads the same names through the tree-sitter grammar instead of regexes.

`domcheck` is the runtime half `webcheck` cannot be: it serves the project on a
loopback HTTP server and loads the entry page in headless Chromium (Playwright),
proving the module graph actually MOUNTS. It reports uncaught page errors,
failed same-origin loads, and unbound controls (conservatively -- any delegated
or inline handler suppresses the finding), and it can diff the mounted surface
against an earlier snapshot (`--snapshot before.json`, then `--baseline
before.json`) to catch the BUG-216 shape where a replaced entry module silently
stops rendering. It also catches SILENT Web Audio failures: it instruments the
API (contexts, resume calls, the connect graph, source starts, gain envelopes,
invalid AudioParam automation), then enables a step if the grid is empty, clicks
Play, and sweeps the sliders -- flagging a graph that never reaches
`ctx.destination`, a context never resumed inside a user gesture, a Play button
that starts zero sources, gains pinned at zero, and NaN/non-positive automation.
Pages that never touch Web Audio get zero audio findings. It needs `pip install
playwright && playwright install chromium`; when that is missing it says so
instead of pretending the page is fine. Running project code is an execution
decision, so the in-run gate needs `verify_dom = true` AND `allow_shell = true`
(or `ask --verify-dom`); an explicit `pedacito domcheck` invocation is itself the
consent. See ../TECHNIQUES.md #45.

`calibrate` also measures **native tool calls** against the JSON grammar on your
model (same eight scenarios, both modes) and the card switches `tool_mode` to
`native` only when it picks the right action at least 0.1 more often with no
loss of well-formed steps. Native mode sends the step schema as one forced
function and reads the model's tool-call arguments, which suits models trained
on tool calling; a server that rejects `tools` falls back to the grammar once
and remembers. Its short probes (instruction following, long-context needle)
get a generous token budget and retry once with thinking off, so a reasoning
model that spends a tiny budget on `<think>` is not scored as failing the skill.
The thinking A/B then asks the same short edit question at the real answer
budget, once with thinking on and once off: a model that is only usable with
thinking off gets `disable_thinking = true` persisted on the card, so later
runs start in the winning mode instead of rediscovering it one empty reply
(one wasted round-trip) at a time.

### Live telemetry

While a task runs, the GUI's **telemetry footer** shows the context window in
use — `Context: 6,420 / 32,768 (19.6%)` with a bar that is green under 70%,
amber from 70–85%, red above — plus turn and session tokens, generation speed
(`Speed: 44.2 tok/s (avg 41.8 tok/s)`), KV cache hit rate and fold count, and
the active model with its specialty (`Active: deepseek-coder (coding)`) and
the orchestrator profile. Click the footer to expand a drawer with the full
per-call numbers. While a specialist holds the card, the last-call numbers
are the specialist's; session totals stay the shared ledger's. The footer
also carries the **Consolidate Session** button: one click distils the
transcript into a handoff checkpoint and restarts the chat from it, and the
context bar drops to the handoff's size right away (see
[Session consolidation](#session-consolidation)).

The CLI prints the same picture as a one-line ribbon on stderr after each
task:

```
[telemetry | ctx: ████░░░░░░ 38% (6,210/16,384) | turn: 6,210 in / 185 out | 43.5 tok/s | cache: 82%]
```

`show_telemetry = false` hides it; `--jsonl` and rpc modes never print it, so
machine consumers see clean streams. In `chat`, `/status` and `/cost` include
the context bar and speeds, and the rpc `status` reply carries
`context_utilization_pct`, `last_tok_per_s` and `avg_tok_per_s`.

`--record` writes the ordered model responses to `.pedacito/replay/<stamp>.jsonl`
(prompts as a hash and a short redacted preview, never in full); `--replay`
feeds them back with no server, running the whole harness (gather loop, folding,
preflight, edit parsing, repair) deterministically and free. It reports which
calls were asked a *different* prompt than when recorded: that is how you test a
change to the harness rather than to the model.

`pedacito eval` runs 15 tasks over four fixture repositories, a small Python
shop, a TypeScript app, a Go CLI, and a medium multi-file Python service with
decoy modules, each with a gold file set, a substring check, a real test for
Python, and planted prompt injections, and prints locate rate, first-apply
success, repair calls, tokens, cache-hit ratio, wall time and a failure
taxonomy, for edits: wrong file, unread edit, format error, truncation,
refusal, hallucination, test regression, injection followed; for questions:
retrieval miss (never read the gold file), ungrounded citation (a file or line
the project does not have), invented behaviour (read it and still got it
wrong), over-refusal (read it and declined). The summary also reports each
kind (edit, question) separately, and the gate checks them separately. Use it to A/B any setting:
run it, change `prompt_pack` or `answer_mode`, run it again. With
`--baseline base.json`, the new report is compared with the old one and the
command exits nonzero on a quality regression, more followed injections, or
token/time growth beyond the configured tolerances, a CI gate over a replay
or a live run. Rates are compared per task over the tasks both reports ran: a
drop of up to one task-equivalent (one task, even across every repeat) is noise
and only noted, a larger drop fails, and a failure label the baseline's taxonomy could not emit is reported
as information rather than a regression. `--repeats N` runs the task set N times, repeat *i*
with `sampler.seed = S + i` (`--seed S`; default your own sampler seed, else 0),
and the report gains per-task pass counts and the mean ± standard deviation of
calls, tokens, seconds and cache-hit across repeats; the gate widens its margin
to twice the standard error when the repeats themselves are noisy. Repeats
reuse the server's warm prompt cache, so `cache_hit` reads a little higher than
for a single run. Two reports that ran with different `disable_thinking` settings
are flagged instead of compared on growth: thinking on/off moves tokens more
than any harness change, and the gate must not blame (or bless) the harness
for the config. `ask --replay FILE --strict-replay` does the same for prompt
drift: it exits 1 when the harness asks a different prompt than the recording.

### GUI Run tab and Fix This Crash

The GUI's **Run** tab previews `index.html` / `public/index.html` /
`dist/index.html` in a live iframe and runs detected entrypoints (`python
main.py`, `pytest -q`, a shell launcher or a custom `run_cmd` in
`.pedacito/config.json`) without a shell. A non-zero exit or a Python traceback
shows a **Fix This Crash** button that puts the exact command, exit code and
traceback into the Task composer. In the TUI, `/run [cmd]` runs the same
targets and `/fix` sends the last failure into a repair task.

## Creative / design mode

The surgical-edit discipline — smallest possible diff, cool sampling, a tight
reply budget — is right for a bug fix and wrong for a landing page. Design
tasks therefore run in a different mode: the model answers as a **Senior
Product Engineer & Visual Designer**, with

- **warmer sampling** on local servers: the answer temperature is at least
  `0.6` (`max(temperature_answer, 0.6)`), so the model explores layouts and
  copy instead of repeating the first safe idea;
- **room to finish**: the answer budget floors at **8192 tokens** (your
  `request_max_tokens_cap` still caps it), so a whole page is not cut off
  mid-`<div>`;
- **whole-file rewrites**: a full `.html`/`.htm`/`.css`/`.md` replacement is
  allowed, and the small-window oversized-rewrite brake and the fragment veto
  stand down for creative tasks and restructure prompts (redesign, restyle,
  reorganize, ...). The wipe refusal and the JS export guard still apply — a
  rewrite may replace a page, never silently delete its substance.

**Detection is automatic.** With `creative_auto = true` (the default), a task
is creative when it targets UI files (`.html`, `.htm`, `.css`, `.svg`, `.jsx`,
`.tsx`, `.vue`, `.svelte`) or reads like design work — website, portfolio,
landing page, hero, dashboard, redesign, restyle, typography, color scheme —
or asks to build/create a web or design artifact. A plain backend task keeps
the surgical discipline even with detection on.

**Force it, or turn it off:**

| Where | Control |
|---|---|
| GUI | the **✨ Creative** button above the composer (Task/Auto/New runs); off = auto-detect |
| TUI | `/creative on` (every task), `/creative auto` (detect, the default), `/creative off` (surgical everywhere); `/status` shows the state |
| CLI | `--creative` on `ask`, `auto`, `new`, `chat`, `tui` |
| Config | `creative_mode = true` forces every task; `creative_mode = false` + `creative_auto = false` disables detection |

On a hosted API the persona and the reply budget still apply, but sampling
stays with the provider — no temperature key is sent (see
[Hosted APIs keep their own sampling](CONFIG.md#hosted-apis-keep-their-own-sampling)).
Pair the mode with the GUI's [Live Preview](#live-preview) to watch the page
change as the run writes it.

## Local image generation

With a local image server (Lemonade Server's `sd-cpp` recipe, Stable Diffusion, Flux), `pedacito models scan`
tags it `modality = "image"` and lists it as **🎨 Image**. Set `image_profile` (or leave it empty to use the
first one) and generate from wherever you work:

```bash
pedacito image "a lighthouse at dusk, watercolor" --size 1024x1024 --out assets/lighthouse.png
```

In the TUI use `/image PROMPT [--out FILE] [--size WxH] [--steps N] [--seed N] [--profile NAME]`; in the GUI,
**Project ▾ → 🎨 Generate image…**. Without `--out` the file is written to `.pedacito/images/`. To let the agent
call the `generate_image` action itself, turn on `allow_image_gen`
([details](CONFIG.md#local-image-generation-image_profile-allow_image_gen)).

## Review page

`--review` (or `review = true`) opens an offline, side-by-side page: current file
left, proposed version right, changed words highlighted. To write, tick the
attestation box, click **Sign off & write**, then confirm. Enter never writes.

The page records each file's SHA-256 when built, and applying re-checks it: if
you edit a file in the meantime, the write is refused instead of overwriting
your change. The server binds to `127.0.0.1` only and rejects foreign `Host`
and `Origin` headers.

When the run tracked requirement clauses, each file carries a badge (`R1`,
`R2`, ...) naming the clauses its edits served; the badge's title is the clause
text. If you accept only some files, any clause whose files were all rejected
is flagged as unaccepted in the sign-off detail, so a partial accept cannot
silently drop half the task.

**Hunk-level sign-off.** In the GUI Changes pane, each change block in a file
is listed as **Hunk N** with an *accept this hunk only* checkbox. Accepting some
hunks writes only those; the rest of the file keeps its current text. The
SHA-256 check still covers the whole file before anything is written.
