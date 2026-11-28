🌐 **English** | [Español](es/ARCHITECTURE.md)

# Architecture

This document describes how Pedacito is built: the local-first engine, the server lifecycle
manager that treats your model servers as one memory pool, execution profiles (including the
`local-moe` tier for tiered MoE servers), and multi-model routing.

For day-to-day use, see [USAGE.md](USAGE.md). For security boundaries, see
[SECURITY.md](SECURITY.md). For settings, see [CONFIG.md](CONFIG.md). For skills, MCP servers and hooks, see
[SKILLS_AND_TOOLS.md](SKILLS_AND_TOOLS.md); to contribute, see [CONTRIBUTING.md](CONTRIBUTING.md).

**Docs ownership.** `docs/` (INSTALLATION, ARCHITECTURE, USAGE, CONFIG, SKILLS_AND_TOOLS, SECURITY and CONTRIBUTING) is the canonical reference for settings, commands and guarantees. The three root guides, [QUICKSTART.md](../QUICKSTART.md), [MODEL_GUIDE.md](../MODEL_GUIDE.md) and [TECHNIQUES.md](../TECHNIQUES.md), remain the canonical task-oriented guides: first runs, model pairing and tuning, and small-model techniques. This README is the index, and [CHANGELOG.md](../CHANGELOG.md) is the history. New reference prose goes to `docs/`, new walkthroughs go to the root guides, and each set links to the other instead of copying text.

## Engine overview

The CLI, the TUI, the browser GUI and the JSONL-RPC interface all drive the same engine. A
task moves through these stages:

1. **Index.** The project is parsed into files, symbols and exact line ranges. Python uses
   `ast`. Other languages use tree-sitter. Summaries are cached by content hash, so
   re-indexing costs only what changed. See [What gets indexed](#what-gets-indexed).
2. **Map.** A repository brief (stack, entry points, test and build commands) and a map of
   the code sit above the conversation. Small projects get a flat listing. Large projects get
   a ranked map of the files most relevant to the task.
3. **Gather.** The worker model takes small, typed steps (`read`, `grep`, `find`, `callers`,
   `context`, and others) until it calls `done`. Where the server supports it, each step is
   constrained by a JSON schema, so a malformed step cannot be sampled. See
   [What the model can do while gathering](#what-the-model-can-do-while-gathering).
4. **Answer and edit.** The model writes search-and-replace blocks, or symbol-level
   replacements. Blocks are dry-applied and checked before anything is written. See
   [Editing](#editing-searchreplace-not-diffs).
5. **Gate.** After writes, Pedacito runs cheap checks first: syntax, static undefined-name
   analysis, frontend contracts, then impacted tests. Repair rounds feed back only findings
   that are new for this task.
6. **Record.** A checkpoint, a session log and telemetry events are written. See
   [SECURITY.md, pillar 4](SECURITY.md#pillar-4-immutable-audit-trail-and-time-travel).

The order is deliberate. A server caches the prompt prefix, so the engine keeps the early
turns byte-identical and folds older lookups only when the window is really full. Every
deterministic gate runs before a model call is spent.

## Why "Pedacito"?

A large codebase is too much to hand a small model in one go, so Pedacito hands
it over in pieces:

- **Chunks, not files.** Python is parsed with `ast`. JavaScript, TypeScript,
  Go, Rust, Java, C, C++, C#, Ruby, PHP, Kotlin, Swift, Scala, and Lua are parsed with tree-sitter. Each function
  or class is a unit with an exact line range.
- **Skeletons, not dumps.** A 500-line class comes back as its signatures and
  line pointers, so the model reads the one method it intends to change.
- **Search before reading.** Ranked maps, lexical and hybrid search, an import
  and call graph, and test links decide which pieces matter for this task.
- **Budgets everywhere.** Reads, observations and answers are capped to token
  budgets fitted to your model's real context window.

## Who it is for

- Developers with one consumer GPU, from a 6 GB laptop to a 24 GB desktop, who
  want an agent that runs on their own machine.
- People who want local models by default, and a hosted model only for the hard
  parts, with a cap on what it can spend.
- People who want to see what the model is doing. Every edit is a dry-run diff
  until you apply it, and every write is snapshotted first.

Pedacito is **local-first and hybrid-ready**. It runs entirely on your own model
server by default, and it can bring in a hosted model for the parts that need
one: as a planner, as a verifier, or as an escalation target, with a spend cap.
Every mode is described in [Why Pedacito?](#why-pedacito-how-it-differs-from-cloud-first-harnesses).

It was built and tested by one person on one home GPU box, so edge cases on other
hardware may be rough. Bug reports with a repro are very welcome.

## Why Pedacito? (How It Differs from Cloud-First Harnesses)

**Why cloud-first harnesses struggle on local GPUs.** Coding agents built for
200k-token cloud APIs send 15k+ tokens of static tool schemas on every request,
and many rewrite earlier turns each step (re-summarising, reordering, or editing
the system prompt). A cloud API absorbs that. A single-slot local server does not:
`llama.cpp`, Ollama, and LM Studio reuse their KV cache only while the prompt
prefix stays byte-identical. Mutate the prefix and the server re-evaluates the
whole prompt on every turn, so the GPU spends its time re-reading context instead
of generating.

**How Pedacito solves it:**

1. **Byte-identical KV-cache prefix (`fold_middle`).** Middle turns fold
   deterministically into hash-pinned `EvidenceStore` span receipts. The first two
   messages (`messages[0:2]`) are never touched, so your server's KV cache stays
   hot. The GUI footer shows the result as **KV Cache Hit %**.
2. **Deterministic gates before any LLM call.** Static checks (`verify`,
   `webcheck`, `domcheck`, `apicheck`), automatic missing-import healing, and
   impacted-test selection catch many mistakes in milliseconds, without spending
   model tokens.
3. **A small dependency set.** The harness needs Python 3.11+ and installs
   `requests` and tree-sitter; there is no Node.js requirement for the harness
   itself. Playwright (browser checks) and LSP servers are optional, used only
   when you have them installed.
4. **Zero phone-home.** "Telemetry" in Pedacito means *your* local Context Meter,
   Burn Rate, and `tok/s` speed figures, written to `.pedacito/events/` in your
   project. Pedacito sends no analytics and no crash reports. The only release
   check is one anonymous GET of the public PyPI metadata at startup (2 s
   timeout, cached 24 h, never blocking), and `check_for_updates = false`
   disables even that — zero network calls. Otherwise the network is used only
   for the model servers and hosted APIs you configure, and for web tools you
   turn on.

### Local-first, hybrid-ready architecture

Pedacito gives you four modes. Use the first alone, or add the others as your
hardware and budget allow.

1. **100% local.** Run fully offline against `llama.cpp`, Lemonade Server,
   LM Studio, Ollama, or vLLM, with no cloud dependency.
2. **Specialist auto-routing (`--auto-route`).** A fast orchestrator model (for
   example, an 8B-class model) plans the work and dispatches each task to the
   best calibrated local specialist: coding, math, SVG/3D, SQL, or i18n. Between
   tasks, the server unloads one model and loads the next, so VRAM stays clean.
   See [Specialist auto-routing](#specialist-auto-routing).
3. **Dual-GPU scout + architect (`scout_profile`).** A fast 4B-9B scout on a
   second GPU or port gathers files and indexes at high speed, while a 24B-35B
   architect does the planning and editing. The architect's KV cache is not
   disturbed by the lookups.
4. **Hybrid escalation and role routing (`planner`, `verifier`, `escalate`).**
   Pair a free local worker with a hosted frontier model (DeepSeek, Qwen, OpenAI,
   or Anthropic) for upfront planning, for verification, or for automatic
   escalation when a small local model stalls. Keep cloud spend small with
   **Session Consolidation** (`/consolidate`) and the **`--max-cost`** budget cap.
   See [Role routing](#roles-a-different-model-per-job).

## Server lifecycle manager

A tiered MoE server keeps most of the GPU's VRAM and a large DDR5 working set resident. If
you load a dense model on another local server beside it, the two servers compete for the
same memory pages, and one of them thrashes. Pedacito therefore treats all of your local model
servers as **one pool with explicit lifecycle controls**. The code is in `pedacito/servers.py`.

| Capability | What happens |
|---|---|
| **Discovery** | `configured_local_servers` lists the local endpoints in your config. `list_server_states` probes each one with a 1.5-second timeout and reports whether it is online and which models are resident. |
| **Memory view** | `get_hardware_memory` reads system RAM from `/proc/meminfo` and VRAM from the DRM sysfs counters for every card that exposes them (AMD amdgpu, Intel xe, and other drivers). Its shape is unchanged. `profiler.snapshot()` layers NVIDIA (`nvidia-smi`), macOS (`sysctl` + `vm_stat`) and Windows (`GlobalMemoryStatusEx`) readers on top, with a source per number. |
| **Load and unload** | Each runtime's native API is used: Lemonade `/v1/load` and `/v1/unload`, llama.cpp `/models/load` and `/models/unload`, and the generic endpoints used by tiered MoE servers. Ollama is unloaded with `keep_alive: 0`. For a single-model daemon, `load_cmd` and `unload_cmd` can run your own command instead. |
| **Pin / unpin** | Lemonade only: `pin_server_model` calls `POST /internal/pin` with `{"model_name": "...", "pinned": true|false}` so LRU eviction skips a loaded model. `/api/v1/internal/pin` is the fallback spelling; other runtimes treat pinning as a no-op. |
| **Auto-evict** | Before a run starts on one local server, the resident models on the *other* local servers are unloaded. Then the memory is polled briefly, so the target's weights can claim it. Controlled by `auto_evict_servers` (default on). The GUI has a toggle. |
| **Best effort** | A sleeping server is a normal state, never an exception. Hosted APIs are never touched. |

The manager is used at task boundaries, by specialist routing and by hybrid routing. It
never unloads a model in the middle of a gather step, so the prompt cache on the server
that is working stays warm.

### What the memory badge means

The toolbar's memory badge shows system RAM, plus VRAM where the driver exposes it. The
numbers are informational and are **totals only**: Pedacito does not attribute VRAM to a
process, and the badge labels say so. Eviction works on server state: models resident on
the other local servers are unloaded before the run, and memory is then polled briefly so
the target can load. A card whose VRAM Pedacito cannot read still gets that handoff.

### Advanced telemetry and auto-tuning

`pedacito/profiler.py` adds a vendor-neutral snapshot without changing
`servers.get_hardware_memory()`:

- **Linux AMD/Intel**: the existing `/proc/meminfo` + `/sys/class/drm` reader, tagged
  `drm-sysfs`. Its glob covers every driver exposing `mem_info_vram_total`/`_used`, which
  includes amdgpu and Intel xe.
- **Linux NVIDIA**: `nvidia-smi --query-gpu=index,memory.total,memory.used
  --format=csv,noheader,nounits`, run as an argv list and only when the binary resolves,
  tagged `nvidia-smi`.
- **macOS**: `sysctl -n hw.memsize` total and `vm_stat` free pages, tagged `vm_stat`.
  Apple Silicon unified memory counts as RAM; there is no separate VRAM claim.
- **Windows**: `kernel32.GlobalMemoryStatusEx` through stdlib `ctypes` (the same API on
  x86-64 and ARM64), tagged `winapi`. NVIDIA cards still use `nvidia-smi`; AMD/Intel on
  Windows have no vendor-neutral VRAM reader.

Each reader is guarded by a platform check or `shutil.which`, time-boxed to two seconds,
and skipped silently when absent. A missing number carries an empty source and is
**unknown**, never zero free memory.

The optional speed probe runs one tiny completion plus one prefill measurement from SSE
timings. It is explicit (`pedacito profile --run`, TUI `/profile run`) or scheduled once
per server+model on a daemon thread after a successful explicit server load. The probe
uses the client's public one-shot entry point and timing record
(`LLMClient.complete_once()`, `LLMClient.timing_record()`), so it does not depend on
private stream bookkeeping. Results are cached for a week in `profiler.json` beside
the user config; failures are silent.

`auto_tune` (default **false**) is applied at the same point as the `local-moe` defaults
in `config.load()`. When enabled it uses only numbers and their sources: free VRAM (or
unified RAM), the estimated weights and KV cache, and the measured prefill speed. It
shrinks a too-large context to a 4,096-token safety floor, proposes growth when there is
spare headroom (never forces it), scales retrieval budgets with the context, keeps the
600 s timeout and concurrency 1 for tiered MoE servers, and abstains when no telemetry is
available. Explicit user values always win. The full rule table and the platform matrix
are in [Configuration](CONFIG.md#advanced-system-profiling-and-auto-tune).

## Execution profiles and the `local-moe` tier

Every model is classed into one of three **execution profiles**. The class decides the output
budgets, timeouts and concurrency that Pedacito uses for it.

| Profile | Applies to | Behaviour |
|---|---|---|
| `local-dense` | A conventional local model (under about 35B parameters) | One GPU queue, modest output budgets, a short timeout. |
| `local-moe` | A tiered mixture-of-experts server (Strata, Avalon, or a matching model) | Frontier-style reasoning on local hardware, with the budgets and timeouts below. |
| `hosted` | A remote API | The provider's own sampling, budgets and reasoning controls. |

### How a model is classed as `local-moe`

Detection needs either signal:

- **The runtime or provider label, or the profile name,** contains `strata`, `avalon`,
  `local-moe`, `local_moe`, `localmoe` or `tiered-moe`.
- **A local endpoint's model id** matches a tiered-MoE pattern. Examples are
  `qwen3.8-flash`, `deepseek-v4.1-flash`, `qwen3-35b-a3b`, or any id containing `moe`.

A remote endpoint is `local-moe` only when its provider label says so. A hosted model whose
name looks like a MoE stays `hosted`.

### What changes for `local-moe`

Applying the tier only **raises** budgets. It never lowers a value you or a preset set.

- **Reply budgets.** Steps and answers get at least 8,192 tokens. Plan and build calls get at
  least 16,384. Reasoning blocks count against the reply budget, so a budget that fits a dense
  9B model cuts a reasoning MoE off mid-thought.
- **Timeout.** At least 600 seconds, for weight loading and a slow 30,000-token prefill.
- **Concurrency.** Exactly **one stream**. These servers hold most of VRAM and DDR5, and a second
  concurrent prefill can evict the resident weights.
- **Reasoning effort.** `reasoning_effort` is `default` (send nothing), or `low`, `medium` or
  `high`, which are forwarded as `reasoning_effort`. It never changes sampling. The local model
  keeps its own temperature.

### Configure a `local-moe` profile

```toml
[profiles.strata]
base_url         = "http://127.0.0.1:8000/v1"   # example port; use your server's address
provider         = "strata"
model            = "your-moe-model-id"
tier             = "local-moe"
reasoning_effort = "high"
```

The `tier` key accepts `tiny`, `small`, `medium`, `large`, `local-moe` and `hosted`. Setting
`tier = "local-moe"` fills the budgets above. Detection covers the case where you leave it
unset.

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
attempt on that model) and `fallback` (takes over when the worker errors
out). Each rung is used at most once per task and only when configured, so a
local model's bad day cannot silently burn a hosted budget. An unset role
uses the active model; a role whose server is unreachable falls back to it
with one warning. Usage is counted per call site either way. `--role
planner=big` sets one for a single run.

`escalate` fires on a recorded reason, not just on a missing edit
(`auto_escalate`, on by default): a task that ends with no edit applied, two
no-progress leases on one task, tests still failing with runtime/semantic
errors (a TypeError, a failed assertion, a loop that never ends) after the
repair rounds, or `escalation_zero_edit_runs` consecutive zero-edit runs on
the active profile (default 3, counted per profile in
`.pedacito/profile_stats.json`; 0 disables the statistic). A reason earns the
retry even when edits DID land -- a patch that applies but does not work is
still a failed task -- and the first attempt is kept only when it was green
and the escalated one came back red. Every reason also emits an
`escalation_suggested` event, so the signal reaches a human or an outer
orchestrator even when no `escalate` role is configured. When no distinct
`escalate` role exists, a fired reason buys ONE same-model retry with the edit
style flipped (anchor <-> quote, unset -> anchor) instead of nothing -- a
`strategy_shift` event marks it, green runs are never retried, and
`auto_escalate = false` disables it. A success clears the
profile's zero-edit count and any queued suggestion.

`pedacito profiles --recommend` suggests a profile per role (worker, escalate,
verifier, planner, summarizer) with a one-line reason, from the measured
harness-fit and eval evidence; `--apply-roles` writes the roles after you
confirm.

Gather has one split that is not a role: `scout_profile = "tiny"` runs the
lookup steps of the gather loop (`read`, `grep`, `find`, `outline`, `callers`, `context`, `usages`, `run_test`) on
that profile's server while the active *architect* profile keeps planning,
synthesis, answers and edits. A scout that is unknown or unreachable falls
back to the architect with one warning, its usage still lands in the same
session ledger, and the GUI shows an active scout as a `Scout: <profile>`
badge beside the model chip. On one GPU it keeps the big model's context
free during investigation; on two it is the natural division: cheap card for
the chatty loop, expensive card loaded once for the answer. In Settings this
is a dropdown over the configured profiles (or "None (use active
profile)"). Pairing sizes and the one-GPU caveat are in
[../MODEL_GUIDE.md](../MODEL_GUIDE.md#dual-gpu--dual-model-scout--architect-routing-scout_profile).

### Specialist auto-routing

Roles split one task by *job*; specialist routing splits a run by *task*.
With `auto_route_specialties = true` (the topbar's **✦ Auto-route** checkbox,
or `--auto-route` per run; `--no-auto-route` overrides the config), each task
goes to the profile whose `specialties` tags fit it best: a coder model for
"refactor the parser", a vision model for "fix the button layout in
styles.css", a reasoner for "optimize the sorting algorithm". Tasks are
scored by keyword and by the extensions of the files they name; a
`[specialty: TAG]` prefix in a task title forces the tag, and the planner is
told to group same-specialty tasks so dependencies allow it.

Tags come from two places, and an explicit list always wins:

- `specialties = ["coding", "testing_qa"]` in a profile (in the GUI: the tag
  pills under **⚙ Settings → Profiles → Connection & Capabilities**, plus a
  custom-slug input). Canonical tags: `general`, `coding`, `frontend_ui`,
  `3d_graphics_svg`, `math_algorithms`, `testing_qa`, `sql_data`,
  `translation_i18n`; any lowercase slug (`[a-z0-9_-]+`) works too.
- `pedacito calibrate` detects them: model-name heuristics (coder families,
  `-vl`/vision models, `r1`/QwQ reasoners), the architecture library's
  recommended tags, plus six micro-probes — a well-formed SVG with exact
  attributes, one integer computation, a placeholder-preserving Spanish
  translation, a SQL join graded by executing it against a fixture database,
  a testing/QA question answered as strict JSON (`avg([])` raises
  `ZeroDivisionError`), and a three-step plan that must order parser → CLI →
  tests — each one tiny call with thinking off. The tags land on the card
  (`measured.specialties`), show in `pedacito profiles` and the card render,
  and fill a profile's unset `specialties` exactly like `prompt_pack` does.

Switches happen **at task boundaries only** — never mid-gather — so the
gather conversation's KV prefix stays cacheable on whichever server owns it
(../TECHNIQUES.md §2 and §50). On one local server the handoff is clean: the
outgoing model is unloaded first (Ollama `keep_alive: 0`, LM Studio
best-effort) so the specialist fits in VRAM, then loaded and warmed; profiles
on different servers (two GPUs, or a hosted specialist) stay warm together
with no unload. Every switch emits `specialist_routed` and `model_swap`
events, the specialist's tokens and cost merge into the main session ledger,
and an unreachable specialist falls back to the worker with one warning. A
task that matches nothing — or whose winning tag the active profile already
carries — stays put; when a previous specialist still holds the card, it is
handed back to the worker first.

A task that ends without a usable edit is also remembered in
`.pedacito/attempts.json` (files looked at, why); a similar later task is told
what was already tried. A success clears it. The same runs are counted per
profile in `.pedacito/profile_stats.json`, which feeds the escalation
statistic above.

## Hybrid routing: an architect plans, an editor implements

Hybrid routing splits one task across two models. A strong **architect** plans it. A fast
**editor**, the active profile, implements it. The architect is usually a larger model,
possibly hosted. The editor is usually a local model you run all day.

The task runs in two stages:

1. **Stage 1, architect.** One focused call returns a `HYBRID_BLUEPRINT`: the root cause,
   the target files, the exact signatures and contracts to keep, and the edge cases.
2. **Stage 2, editor.** The editor gathers context and writes the edits. The blueprint is
   injected ahead of the gathered context.

Between the stages, the server that held the architect is evicted if the two models live on
different local servers (`auto_evict_servers`). On one GPU, a 27B architect and a 9B editor
can take turns on the card. A hosted architect coexists with a local editor and needs no
eviction.

Telemetry is combined. Architect and editor tokens fold into one run total, with a per-stage
breakdown. If the architect cannot be reached, the task degrades to a plain editor run and
prints one warning.

### Turn it on

In the GUI, use the **Hybrid Architect** toggle in the Options bar and choose a provider and model.
In the TUI, use `/hybrid off` or `/hybrid <provider> <model>`. In the config:

```toml
[universal]
hybrid_routing     = true
architect_provider = "big"                  # a [profiles.*] name, a provider label, or a base URL
architect_model    = "your-architect-model-id"
```

Hybrid routing stays off while `architect_model` is empty. An architect model equal to the
active model is a no-op, because there is nothing to split.

Hybrid routing is separate from [role routing](#roles-a-different-model-per-job). Roles pick
a model for a job, such as planning or verification. Hybrid routing is a two-stage split of
a single task.

## Two-model sandbox arena

The arena compares two models on the same prompt without touching your project. Each
contestant works in its own isolated copy of the workspace. The copy excludes `.git`, `.venv`,
`node_modules`, checkpoints, `dist` and `build`, and the real project is never modified.

The result is a side-by-side scorecard for each contestant: whether it succeeded, files
changed, a unified diff, elapsed time, tokens per second, total tokens and a summary. The last
result is stored in `.pedacito/arena_last.json`.

Applying a winner is checkpointed before and after, so the adoption is undoable with
`pdc undo`. Use the GUI's **Arena (A/B)** button, or the TUI's `/arena <a> <b> <prompt>` and
`/arena apply a|b`. The arena has no separate CLI command.

## What the model can do while gathering

The model works through a small set of actions:

| Action | Purpose |
|---|---|
| `read` | Numbered lines of a file, or a definition by symbol name. A mid-file read annotates the enclosing function/class and the module's imports, so the model always knows where it is. |
| `grep` | Regex or literal search (project, file, directory, or glob), hits labelled with their enclosing symbol. Falls back to literal and case-insensitive matching, and suggests similar symbols on a miss. |
| `find` | Search indexed definitions by name, signature, or summary, navigation by concept, where `grep` needs the exact source spelling. Substring hits first, then BM25 (so a multi-word concept like `retry backoff` works). One line per hit with exact line ranges; a miss suggests similar names. |
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
| `run_test` | Run one scoped test check (pytest, Node/Vitest, Go, or Cargo, chosen from the target) and read the condensed traceback. On by default; see [Scoped test runs](#scoped-test-runs-during-investigation) |
| `done` | Stop gathering and answer. |

Repeating a read-only lookup (even many steps later) that nothing has changed
since gets a pointer to the earlier result instead of a second copy, and any
non-read observation (a broad `grep`, a chatty command, an MCP reply) is capped
at a fraction of the window (`max_observation_tokens`) with a note on how to
narrow it.

The action list is **filtered to what is enabled and to the model's size**: a
tiny profile sees eleven actions and a ~450-token prompt, not two dozen tools and
browser-eval advice it cannot use.

Opt-in actions, each behind a config flag because they widen what a model can
do:

| Action | Enable with | Purpose |
|---|---|---|
| `write` / `append` / `move` / `delete` | `allow_write = true` | Whole-file operations; `append` builds a file larger than one reply. |
| `rename` | `allow_write = true` | Mechanically rename one definition and its references in the files that import it (strings and comments untouched, every file re-parsed, snapshotted). Refuses a name defined in several places and lists the references it left alone. |
| `undo` | `allow_write = true` (+ `backup`, on by default) | Self-recovery mid-task: reverts what this task's `write`/`append`/`move`/`delete` actions did so far (or just one of their files), pre-task snapshots come back, files the task created go away. |
| `run` | `allow_shell = true` | Run one command (e.g. `pytest -q`), without a shell, so `&` and `nohup` are rejected and long-lived processes go to `serve`. A quiet GUI/server is stopped after `shell_idle_seconds`. |
| `serve` | `allow_serve = true` | Start, list, log, or stop a background dev server. The registry and logs live under `.pedacito/procs/`; also `pedacito serve` / `pedacito procs`. The list collapses superseded stopped entries (restart history) behind a count; `procs --all` shows them. |
| `diagnose` | `allow_shell = true` | Run the detected linter/type checker (`ruff`, `pyflakes`, `flake8`, `mypy`/`pyright` when configured, `npm lint`, `tsc`, `cargo check`, `go vet`, `rubocop`). Also available as `pedacito diagnose`. |
| `lsp` | `allow_lsp = true` | Exact definition/references/hover/diagnostics through pyright/pylsp/typescript-language-server/rust-analyzer/gopls, or commands in `lsp_servers`. Also `pedacito lsp`. |
| `screenshot` | `allow_screenshots = true` | Capture the screen as text: size, brightness, blank-screen verdict, OCR, optional `vision_model` description. Needs the `vision` extra. |
| `browser` | `allow_browser = true` | Headless browser: start a server command, open the page, click/type/hold keys, and report page text, canvas sizes, and console errors. Needs the `browser` extra. |
| `mcp` | `allow_mcp = true` + `mcp_servers = [...]` | Call tools, prompts (`prompt:server/name`), or resources (`resource:server/uri`) from MCP servers. The catalog shows arguments, calls are validated, one live session per server is kept warm, and a server that was down at startup is retried. |
| `mcp_call` | `enable_mcp = true` + `[mcp_servers.<name>]` | Call a tool on a *named* MCP server. Only the up-to-four tools whose names and descriptions match the task are ever shown, the action itself appears only when one matches, and untrusted servers ask the permission broker per call. See [Named MCP servers](CONFIG.md#named-mcp-servers). |
| `task` | `allow_subagents = true` | Hand a bounded, read-only research question to a helper agent with a fresh context window; only a short digest comes back. Depth is capped (1 local, 2 hosted). |
| `fetch` | `allow_fetch = true` + `fetch_allowed_hosts` | URL → readable text, host-allowlisted, size-capped, cached, labelled untrusted. Private/loopback addresses are refused — including public names that *resolve* to them — unless you name them explicitly, e.g. `fetch_allowed_hosts = ["127.0.0.1", "localhost"]` to read your own dev server; the connection is pinned to the checked IP. |
| semantic search | `semantic_search = true` | A `grep` with no hits falls back to the server's `/v1/embeddings` endpoint. |

`run` and the two MCP actions execute model-chosen commands in your project.
Treat them as a
trust decision, but see [Safety](SECURITY.md): `run` is sandboxed by
default where the OS allows, and the rest of the blast radius is limited too.

### Scoped test runs during investigation

A model that only reads code has to guess what a fix will do. `run_test` lets it
check the guess first: it runs one scoped test check during investigation, so
the live traceback is in front of the model before it writes the patch.

```json
{"action": "run_test", "target": "tests/test_app.py::test_save", "filter": "save or retry"}
```

- **Scoped.** `target` is a test file, a node id, or a directory. `filter` is an
  optional `-k` expression (a name pattern for the other runners). The model
  never supplies a shell command line.
- **Follows the project's runner.** Python targets use pytest. A
  `*.test.js`/`*.spec.js`/`*.test.ts`/`*.spec.ts` file uses Vitest when the
  package or detected command names it, otherwise Node's built-in `node --test`.
  `*_test.go` (or a directory holding one) uses `go test ./dir`, and
  `tests/*.rs` uses `cargo test --test NAME`.
- **Offered when useful.** It appears when the project has tests (`tests/`,
  `test_*.py`, `*.test.js`) or the task names a failure (test, fail, bug, error,
  traceback, reproduce, broken).
- **Bounded.** Each run is capped at 15 seconds and at the same token budget as a
  read. Output is a tally line plus up to four condensed failure tracebacks that
  show only project frames. A green run is reported as green, so the model does
  not re-run it to watch it pass again.
- **Sandboxed.** It runs under the same sandbox as `run` (no network, writes only
  inside the project), and the answer phase has no tools at all. The run is recorded as a repository fact, so a test command
  that keeps working becomes a remembered command.
- **Honest about what an exit code means.** The reply names the verdict in
  words -- `EXIT: 5 (NOT COLLECTED)` -- and explains it: 5 means nothing was
  collected, which says nothing about whether the code works. A `target` that
  declares no `def test_...` and no `class Test...` is refused before pytest
  starts, with the working alternative spelled out (`run` it as a script),
  because a bare `no tests ran` once sent a model rewriting a passing
  standalone smoke script's functions to `test_*` to make the number go away.

It is on by default (`enable_run_test_in_gather = true`). Set it to `false` to make
every run ask first in the **Permission requested** dialog under **Test Execution**.
A plain CLI run has no dialog, so there the tool reports that it is disabled.

### Repair lessons and repo gotchas

Every time a repair round fixes a failed check, Pedacito records what the failure
was and that the fix worked. The lesson goes to `.pedacito/micro_facts.json` under
`anti_patterns`, and later runs in that repository see it as a hint:

```
[REPO GOTCHAS & LESSONS] (repair lessons distilled from earlier runs; hints, not truth):
- dom_id_mismatch: '#add-task-btn' was queried by JS but no rendered element carried that id
```

- **Only proven fixes.** A lesson is written only when a gate (syntax, static,
  frontend contract) or a test went red during the task and is green again after
  the task's own repair rounds. An unverified guess is never recorded.
- **Plain text, no model prose.** Each lesson comes from a fixed template that
  fills in the names from the failure, so it can be checked by the same tools.
- **Bounded and self-cleaning.** At most eight lessons are kept. Repeats are merged
  and counted, not added. A lesson about files that no longer exist is dropped.
  Each lesson also carries `runs` / `recurred` / `avoided` counters from observed
  check runs: one that faced five runs without ever avoiding its failure kind is
  retired into `retired_lessons` (auto-promoted rules are exempt). These are
  harness facts, never model-written reflection.
- **Scoped, not shouted, when they repeat.** A lesson confirmed twice whose
  files all live under one directory (`src/api/...`, `frontend/...`) is promoted
  into an auto-managed path-scoped rule at `.pedacito/rules/auto_<dir>.md`
  (`paths: <dir>/*`, `auto_generated: true`, up to five deduplicated lines). It
  then shows up only when the model reads a file in that directory -- not in
  every prompt of every task. When the files a promoted lesson names are all
  deleted, its line is dropped and an empty auto rule is removed. Files without
  the `auto_generated: true` flag are yours; Pedacito never edits or deletes them.
- **Hints, never truth.** The loop re-checks everything live, so a wrong lesson
  costs a glance, not a wrong change.

The file also holds the commands Pedacito confirmed in this repository (see
[micro-facts](../TECHNIQUES.md#14-repository-micro-facts-with-contradiction-retirement)).
It is plain JSON. Read it, edit it, or delete it to forget everything it has
learned. `micro_facts = false` turns recording and injection off, and
`max_micro_fact_tokens` caps how much of it reaches the prompt.

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

**Repo map and the KV prefix.** The brief and map are placed in **user0** — the
first user turn — never in the system prompt. `messages[0:2]` is therefore the
cached prefix every gather step shares within a task, and the system prompt stays
byte-identical across tasks. Ranking is deterministic for a given index and task,
so repeated calls return byte-identical map text. `flat_map_token_budget`
(default 4,000, auto-tuned by the profiler) is the absolute key for the flat
tier; larger projects fall back to file cards, then a ranked map over a
directory skeleton. Explicit pins (including a manually injected skill) are
resolved before automatic retrieval and sit ahead of the map, so they are never
scored or evicted.

File cards carry blast radius, `imported by: 7 file(s)`, or a labelled
"no importers or callers found (evidence, not proof)", and cross-link modules
with their tests (`tested by:` / `tests:`) for Python, JS/TS (`x.test.ts`,
`__tests__/`), Go (`x_test.go`), Rust (`tests/`), Java/C# (`XTest`), and Ruby
(`x_spec.rb`) when the match is unambiguous. `pedacito map PROJ -t "task"`
prints exactly what a task would see.

**Secrets are never indexed.** `.env`, `.env.*` (but not `.env.example`),
`*.pem`, `*.key`, `id_rsa*`, `.npmrc`, `.pypirc`, `credentials.json` and
similar are skipped by default; re-include one deliberately with `!.env` in
`.pedacitoignore`. Excluded directories (`node_modules`, `build`, ...) are
pruned before they are walked, and nested `.gitignore` files are honoured.

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
cleanly, and a failure shows the closest matching window in the file right
now, with line numbers and a similarity score, so the retry re-anchors on real
text instead of guessing. A FILE header can also pin a block from the start:
`--- FILE: src/app.py LINES 40-60` replaces that exact range, and
`SYMBOL is_leech` replaces that whole definition, useful when a snippet
appears more than once or the text keeps shifting under the model.

A definition can also be replaced *without* echoing its old text at all, which
is what the model usually gets wrong. Name the symbol instead of searching for
it, and write only the new body:

```
--- FILE: src/app.py
<<<<<<< SYMBOL src/app.py :: Report.summarise
    @staticmethod
    def summarise(rows):
        return sum(rows)
>>>>>>> END_SYMBOL
```

The span comes from the file's own AST (`ast` for Python, tree-sitter for
JavaScript/TypeScript/Go/Rust), decorators included, so a block cannot land in
the middle of something. Indentation is normalised to the definition's real
column, and leading decorators are carried over when the new body does not
re-echo them. `<<<<<<< AFTER_SYMBOL file :: name` splices the body *after* that
definition instead of replacing it (add a method without re-emitting the
class). The same two operations exist as structured payloads for constrained
sampling: `{"edit_action": "replace_symbol", "file": ..., "symbol": ...,
"replacement": ...}` and `{"edit_action": "insert_after_symbol", ...}`. Plain
SEARCH/REPLACE blocks are unaffected and both shapes may appear in one reply.
The result faces the same syntax gate: a body that does not parse is refused,
not written, and an unknown or ambiguous symbol name comes back with an
instruction to `outline` the file or re-emit the block with `LINES a-b`.

Before writing, Python is checked with `ast.parse` and tree-sitter
languages with their parsers: an edit that breaks a clean file is refused, while
a file that was already broken is edited with a warning. A failed block's error
goes back to the model for up to `repair_rounds` retries (default 2).

After a write, `post_edit_check` (on by default) runs the cheap check, the
detected linter, else a syntax parse, on the changed files and feeds back only
findings that were *not* there before the task, giving the model up to
`repair_rounds` fix rounds to clean up its own mess without chasing
pre-existing issues.

Set `fuzzy_match = true` to let near-miss blocks anchor to the most similar
text (`fuzzy_min_ratio`, default 0.85). Such edits are flagged as approximate.
A block the model *started* but never finished (a missing `=======` or
`>>>>>>> REPLACE`, a block with no `--- FILE:` header) is reported back as a
failure instead of being silently dropped.

### Renaming a symbol across the project

A rename does not need a model, so Pedacito does it mechanically: every exact
identifier is rewritten, and comments are left alone. String literals are
left alone too unless you pass `--strings` (all of them) or `--include-labels`
(test-label strings only).

```bash
pedacito rename apply_discount discounted_price                       # preview the diffs
pedacito rename apply_discount discounted_price --file pkg/core.py    # that file and its importers
pedacito rename apply_discount discounted_price --strings             # also whole-word string literals
pedacito rename toggleStep flipStep --include-labels                  # also test/it/describe/suite labels only
pedacito rename apply_discount discounted_price --apply               # write it
```

A qualified name narrows the rename to one class: `pedacito rename Store.save save_record`
renames `Store.save` and the call sites whose receiver is provably a `Store` (Python),
and lists the look-alikes it left alone. The preview is the default, so no flag is needed to get one. `--apply` writes the
change and snapshots the originals first, so `pedacito restore` undoes it.

- **Identifiers first.** Python names move through `tokenize`. JavaScript,
  TypeScript, Go and Rust move through their tree-sitter identifier nodes. A
  string or comment that merely contains the old name is never touched by
  default; `--strings` additionally renames whole-word occurrences inside
  string literals (test labels, docs that name the API), while comments stay
  untouched. `--include-labels` is the narrow middle option: only string
  literals that are test labels (the first argument of `test`/`it`/`describe`/
  `suite`, `.only`/`.skip` included) move; every other string stays.
- **All or nothing.** Every changed file must pass the syntax check before any
  file is written. If one would break, nothing is written.
- **Name-based, on purpose.** A parameter in an unrelated function that shares the
  name is renamed too. The preview shows it, so you can judge before writing.
- **Also available to the model.** With `allow_write = true`, the gather loop has a
  `rename_symbol` action that calls the same engine, and it goes through the same
  permission and snapshot rules as other writes.

In the GUI, **Project ▾ → Rename symbol…** takes the old name, the new name and an
optional scope file. **Preview** shows the diffs, and **Rename** writes them.
The dialog renames identifiers only; `--strings` is a CLI flag.

### Checking an answer before it is written

A small model's usual failure is not bad syntax but a plausible edit to the
wrong place. Before anything is written, `verify_answer = "deterministic"` (the
default) dry-applies the blocks and runs free checks, no model call:

| Check | Catches |
|---|---|
| unread | an edit to a file, or a region of one, the lookup phase never `read` |
| placeholder | `TODO` / `NotImplementedError` / bare `pass` instead of working code (unless the task asks for a stub) |
| stale references | a removed or renamed definition that is still referenced elsewhere, judged against what *all* of the reply's edits leave behind, so a correct multi-file rename passes |
| moved imports | a moved file still imported by its old path |
| citations | `file:line` claims about files never read |
| coverage | a requirement from the task (it extracts up to five) no edit or sentence touches; a warning, and in the incremental edit loop it seeds one bounded follow-up pass |

Errors cause one corrected re-ask that shows the model its own blocks and the
findings; warnings are printed and logged. `verify_answer = "full"` adds one
adversarial verifier call (on the `verifier` role, if you set one), and
`best_of = 3` samples extra candidates at a higher temperature and keeps the
best-scoring one, both cost model calls, and `pedacito calibrate` turns them
on for models that measure weak.

With `verify_tests = true` and `allow_shell = true`, the tests *impacted* by an
edit (the tests named after the file, tests of its importers and callers) run
after it is written; new failures, compared with a run before the edit, go
back to the model for up to `repair_rounds` fix rounds. A failure the command
ledger already recorded before this run is labelled *pre-existing failure
(recorded before this run)* and does not trigger repair -- unless this run
captured its own baseline, which supersedes the history. `test_command`
overrides detection (`pytest`, `npm test`, `cargo test`, `go test ./...`,
`make test`, ...). Those two flags only ever switch the **subprocess** checks
off. In a mixed tree (Python tests beside `*.test.js`), the runner follows the
language of the change: a Python-only edit runs `pytest` even when a
`package.json` test script would otherwise win detection, and a JavaScript
edit runs the JS runner. An explicit `test_command` is never overridden. The syntax, static, frontend and cross-file contract gates read files in
this process, so they vet every applied edit even in the default
`allow_shell = false` profile: a run that cannot execute tests still cannot
ship a module graph that would not link.

### Frontend contract checks

Passing tests do not prove that a web page loads. After an applied edit, and
before the test command runs, Pedacito checks five things with no model call:

- **Asset links.** Every local `<script src>` and `<link href>` in an HTML file
  must point to a file that exists. External URLs, `data:` URIs and template
  placeholders (`{{ }}`, `<%= %>`, `${...}`) are skipped, because their real path
  cannot be known statically.
- **DOM ids.** Every literal `getElementById('x')` and `querySelector('#x')` must
  match an `id="x"` in some HTML file, or an id the JavaScript creates itself. A
  lookup built from an interpolated id is not checked, since its value is not
  known until runtime.
- **Syntax.** Changed JavaScript must parse. `node --check` is used when Node is
  installed, and the tree-sitter grammar otherwise. Parsing is not linking: a
  file can parse and still be dead on arrival, so two more shapes are checked
  against the grammar.
- **Module bindings.** A name bound twice at module level -- the second
  `import { WAYPOINTS }` an incremental edit appends to a file that already
  imports it -- is a `SyntaxError` the browser refuses the whole graph for.
  Every ES module's top level is scanned for it, `var`/`function` pairs excepted
  because those are legal re-declarations.
- **Unreachable calls.** A module sees only the names it declares or imports. A
  call to a name no module binds is a `ReferenceError` the moment the line
  runs, and the finding says which fix applies: the sibling that exports it
  (add the import), the file that declares it without exporting it (add the
  `export`), or nothing at all (define it or drop the call). Browser globals,
  CDN bundles and classic-script globals are never reported. An arrow
  callback's own parameter counts as bound, so `rows.forEach(m => m.name)`
  is not a call to an undefined `m`.

A finding is reported in the same shape as the other gates, for example:

```
FrontendContractError in app.js:42: queries DOM element '#add-task-btn',
but index.html only defines ids ['board', 'new-task-btn', 'todo-col']
```

Only findings that are new for the current task block the edit, so existing
problems elsewhere in the project do not stop unrelated work. If you edit an HTML
file, the JavaScript that looks up its ids is checked again. The gate is on by
default and is turned off with `frontend_contracts = false`. `pedacito webcheck PROJ`
runs the same vocabulary plus the heuristic checks (ESM syntax in classic scripts,
duplicate globals, undeclared imports, member calls on page globals) on demand,
as warnings only, and `auto` prints its findings after tasks that touch web
files. Both `pedacito new --apply` and `pedacito new --complex` run the vetting
half over the finished tree -- the autonomous loop most of all, since each of
its tasks writes one file in a fresh context and never sees the whole page:
a scaffold whose page cannot link gets repair rounds, and if the violations
survive them the build prints `BUILD INCOMPLETE` and exits 1 instead of
reporting a project that cannot start.

### Headless DOM smoke check (opt-in)

The frontend gate proves the contracts a browser enforces at *load*; it cannot
prove the page actually *mounts*. With `verify_dom = true` AND `allow_shell =
true` (or `ask --verify-dom`), Pedacito also loads the entry page in headless
Chromium after applied web edits and diffs the mounted surface against a
snapshot taken before the edit. A replaced entry module that silently stops
rendering -- named elements or controls gone, visible text collapsed -- is a
`dom` gate finding that feeds a repair round, alongside uncaught page errors and
assets/modules that no longer load. The same check catches silent Web Audio
failures: an init script records contexts, resume calls, the connect graph,
source starts, and gain automation; a smoke probe enables a step if the grid is
empty, clicks Play, and sweeps the sliders. A never-resumed context, a graph
that never reaches `ctx.destination`, a Play button that starts zero sources,
gains pinned at zero, or invalid AudioParam automation are `dom` gate errors
too -- Web Audio fails without console errors, so nothing else sees them.
Non-audio pages stay dormant (zero audio findings). Pre-existing page errors
are not blamed on the task. It needs the optional Playwright install (`pip
install playwright && playwright install chromium`); without it the check
reports an install hint and the gate steps aside. `dom_entry` overrides the
detected entry page and `dom_timeout` caps one load. `pedacito domcheck PROJ`
runs it on demand (that invocation is the consent to run project code);
`--snapshot`/`--baseline` diff two runs. The session runs in a child process
with a wall-clock bound, so a page whose script never yields is stopped and
reported as *not verified* instead of blocking the run. Unlike `webcheck`'s notice-level regex
findings, this tool's errors can drive repair. See ../TECHNIQUES.md #45.

## How context is managed

A small window is spent three ways: the prompt's fixed text, the map, and the
conversation. Pedacito spends each on purpose.

- **Prompt packs.** The instructions come in `full`, `small` and `tiny`
  density (`prompt_pack`, else the model card, else the tier) and mention only
  the actions that are enabled. With browser, shell and write off, the default
  lookup prompt is about 1,700 tokens instead of over 3,000; `tiny` is about 450.
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

## Project state on disk

Everything Pedacito writes for a project lives under `<project>/.pedacito/`:

```
index.json            the map (versioned, written atomically)
summaries.json        summary cache, keyed by content hash
embeddings.json       semantic-search cache (if enabled)
backups/<stamp>/      pre-write snapshots (never merged: -2, -3 on a clash)
sessions/<stamp>.md   what was asked, looked up, and changed (secrets masked)
checkpoints/          time-travel tree (index.json + snapshots/) and ask/chat history
events/<stamp>.jsonl  one JSON line per step, call, fold, verify, task
reviews/<stamp>.html  saved review pages
screenshots/          captures (if enabled)
procs/                background process records and logs
design.json/.md       spec-driven design (interfaces, data, tests)
memory.md             remembered facts (hints for the model)
consolidated_session.md  the newest session-consolidation handoff
micro_facts.json      commands confirmed to work here, and repair lessons
rules/*.md            path-scoped rules       skills/           your skills
                                             (NAME/SKILL.md or NAME.md)
prompts/*.md          prompt overrides        fetch/              fetch cache
todo.md               current task checklist
log.txt               timestamped event trace
```

`sessions/` is the place to look when an edit turns out wrong days later;
`log.txt` shows where a failure happened. With no project open, the GUI's
project-less modes (How and New) have no `.pedacito/` folder, so their chats
and session logs go to a per-user state folder instead:

```
~/.local/state/pedacito/            Linux/macOS ($XDG_STATE_HOME replaces
                                    ~/.local/state)
  threads/global/<id>.json          chats opened with no project
  sessions/global/<stamp>.md        task logs written with no project
%LOCALAPPDATA%\pedacito\            Windows
```

Opening a project switches to that project's own threads under
`<project>/.pedacito/`; the project-less chat stays in the state folder and
comes back when the scope returns to no project. Nothing is copied between
the two.

On first run Pedacito creates a
commented `.pedacitoignore` and adds its entries to `.gitignore` (switch off
with `create_ignore_file = false` / `manage_gitignore = false`). Deleting
`.pedacito/` is always safe; it only costs a re-index. `pedacito reset PROJ`
does that in one step: it stops registered background servers, then clears the
runtime state (index, summaries, plans, sessions, checkpoints, backups, events,
reviews, screenshots, fetch cache, procs, `log.txt`) while leaving your code,
`.pedacito.toml`, `.pedacitoignore` and `.gitignore` alone. `--keep-memory`
preserves remembered facts, `--keep-config` preserves authored `rules/`,
`skills/`, `prompts/` and instruction files, and `--reindex` rebuilds the map
(`--no-summaries` makes that rebuild static, with no model calls). `--dry-run`
shows what would go; live git worktrees under `.pedacito/worktrees/` make it
refuse unless you pass `--force`. `pedacito prune` keeps the newest N sessions,
backups and event files and trims `log.txt`.

## v0.3.3 modules at a glance

- **`pedacito/gitops.py`** — guided git. Every git call is argv-only. It runs the
  secret pre-flight scan on the staged diff (commit) or the branch diff (pr) and
  never pushes or force-pushes. `pedacito commit`, `pedacito pr`, the TUI `/git`
  and the GUI Git drawer all call the same functions.
- **`pedacito/acp.py`** — the Agent Client Protocol bridge (`pedacito acp`). It
  is a JSON-RPC 2.0 framing layer over the warm `rpc.py` engine, so the task
  logic is not duplicated. `session/update` notifications come from the same
  event stream that `rpc` emits.
- **`pedacito/rpc.py`** — `params.question` on `explain`, `--events on|off` and
  `--describe`. The caller's `id` is kept next to `req_id` on streamed events.
- **`pedacito/config.py: _endpoint_is_local`** — the single locality predicate
  (loopback, `.local`, RFC 1918, RFC 6598 CGNAT / Tailscale, link-local, IPv6
  ULA, local provider labels). `config`, `llm`, `modelcard`, `servers`,
  `model_catalog` and `cli` all delegate to it.
- **`pedacito/tools.py` scratch spill** — oversized observations from the
  observation-producing actions are written to `.pedacito/scratch/*.log`. The
  model reads and greps them through `read` and `grep`, and cannot write them.
- **Tool format `tool_mode = "plain"`** — a plain-word gather format next to the
  grammar and native formats. `calibrate` A/B-tests it.
- **Router (`proactive_escalation`)** — an opt-in two-path router. A zero-token
  structural gate escalates hard tasks; a boundary-fewshot Scout keeps simple
  single-file tasks on the worker. Off by default.
- **`pedacito/tui.py`** — the zero-dependency rich view: inline diffs, collapsible
  tool-call cards (`Ctrl+O`, `/expand`, `/collapse`), the six-row `@` picker and
  SGR mouse zones. It is still a stdlib-only terminal layer.
- **`pedacito/i18n.py` + `pedacito/profiles.py`** — `pedacito profiles` headers
  and the recommendations title are looked up through `i18n.t`, so
  `pedacito --lang es profiles` prints Spanish labels. English stays the default.

## Provider registry and capability evidence

`pedacito/providers.py` is the one registry behind hosted compatibility quirks,
the GUI quick-add presets and the capability matrix. It lists 24 hosted
providers plus LiteLLM and three gateway-only placeholders, and 13 local
runtimes (LM Studio, Ollama, llama.cpp, vLLM, Lemonade/lemond, Strata, Avalon,
MLX-LM, SGLang, TabbyAPI, KoboldCpp, LocalAI). Host matching is equality or a
dot-boundary suffix / `*` label, never a substring, so a look-alike host cannot
inherit a provider's caching or capability claims.

`ProviderCapabilities` keeps every feature tri-state: `None` is unknown and is
never presented as supported. `get_capabilities` is offline by default and
merges catalog evidence, cached models.dev metadata and refusals observed live
by `llm._JSON_CAPS` / `_ACCEL_CAPS`. `probe_capabilities` runs only after
explicit consent, makes one bounded minimal request, refuses redirects so auth
cannot cross a host boundary, and stores a redacted 24 h cache entry keyed by
normalized URL + exact model + auth-identity hash (never the secret). Azure
OpenAI, AWS Bedrock and Google Vertex are gateway-only entries whose capability
source says `unverified: requires configured OpenAI-compatible gateway`; they
are not native adapters.

The transports preserve opaque reasoning across assistant tool-call turns:
OpenAI-compatible `reasoning_content`, Anthropic signed thinking blocks and
Gemini thought parts / `thoughtSignature` are carried as separate protocol
fields, not visible content, and are excluded from logs and capability caches.

## v0.4.0 session isolation, daemon and parallel subagents

v0.4.0 lets several sessions run in one process, or behind `pedacito daemon`,
without sharing runtime state. The contract below is the set of rules that the
acceptance gates in `tests/test_v040_acceptance_gates.py` prove end to end.

### The Session Isolation Contract

| ID | Rule | Where it lives |
|---|---|---|
| D5 | Local model servers are shared through leases, never evicted under a live session | `leases.py` |
| D6 / D15 | Project state is guarded by a lock, and maintenance waits for live sessions and daemons | `workspace.py` |
| D7 | Every task writes a durable write-ahead journal that survives a crash | `journal.py` |
| D8 | A session that uses a worktree gets its own branch and directory; a dirty tree is refused, carried or stashed, never silently mixed | `worktree.py` |
| D9 | Review routing and the review queue belong to one session | `review.py`, `session_ctx.py` |
| D10 | Sensitive artifacts never reach checkpoints, ratchet, backups, mirrors or review | `security.py`, `edits.py`, `workspace_overlay.py` |
| D11-D12 | The session protocol carries its version and per-session features | `rpc.py`, `daemon.py` |
| D13 | Cancelling a session runs one deterministic cascade over its own resources only | `resources.py`, `session_ctx.py` |
| D14 | Unattended execution runs inside an OS sandbox or is refused | `sandbox.py`, `sandbox_win.py` |

Two numbering notes. The D-numbers are the work-package labels used in the code
and tests; a few numbers (D1-D4) are used by other test areas for unrelated
work and are not part of this contract. The gate suite is the executable
definition of the contract.

### SessionContext and context propagation

A `SessionContext` (`session_ctx.py`) owns everything a session mutates: its
`cfg` clone, `idx`, `client`, `event_bus`, `permission_broker`, `model_lease`,
`budget_ledger`, `worktree`, `policy`, `review_queue`, `resource_registry`,
`run_telemetry`, and the `tool_caches` namespaces. The active context is a
`contextvars` value. `use_session(ctx)` binds it for a block,
`spawn_thread_with_context(...)` starts a thread with a copy of the caller's
context, and `ContextThreadPoolExecutor` copies the context on every `submit`.
Threads started from a session therefore see that session and no other, and
concurrent tasks never race on one another's budget, telemetry or cache.

### Events, permissions and the daemon

`SessionEventBus` (`events.py`) is a per-session, monotonic stream with
bounded `replay(after_seq)`. Every envelope carries its `session_id` and `seq`.
Events are also written to the root's JSONL log under
`<root>/.pedacito/events/`, tagged the same way. `SessionPermissionBroker`
(`security.py`) keys pending prompts by `(session_id, request_id)`: a decision
for another session is refused with `session mismatch`. A non-interactive or
`approval_policy = "never"` prompt is denied at once and emits a
`permission_decision` event.

`DaemonServer` (`daemon.py`) is a loopback-only HTTP/SSE service that owns
isolated `WorkspaceSession`s. Each route is scoped to one session id:
`/v1/sessions/<id>/events`, `/prompt`, `/cancel`, `/permission` and `/review`.
Every request needs the per-run bearer token. The daemon's `quiesce` hook makes
`check_maintenance_safe` refuse `reset` and `prune` while a daemon session is
active.

### ModelLeaseManager

Local models are a shared, expensive resource. `ModelLeaseManager` (`leases.py`)
grants a lease per `(server, model)`. A lease marks its model as in use:
`can_evict()` is False while any lease is held, and `servers.evict_other_local_servers()`
skips leased models, so a session's model is never unloaded under it. A session that needs a different local model queues. An
`exclusive=True` lease (for example `calibrate`, at `LeasePriority.EXCLUSIVE`)
drains active leases first and then runs, without preempting a session that is
already running. Hosted endpoints bypass the pool.

### ProjectLockManager and maintenance

`ProjectLockManager` (`workspace.py`) holds one writer per project root. A second
exclusive claim from another session raises `ProjectBusy`, and shared claims
coexist. `atomic_update_json` performs read-modify-write under a file lock, so
concurrent updates are not lost. `check_maintenance_safe(root, action=...)`
returns `(False, reason)` while a registered quiesce hook reports active sessions
or the run lock is held, and `force=True` is the only override.

### TaskJournal (WAL)

`TaskJournal` (`journal.py`) writes versioned UUID-named WAL files
(`.pedacito/journal/<wal_id>.jsonl`, separate from the redacted task title),
preserves monotonic sequence numbers across explicit `wal_id` reopens, and
fsyncs each line by default. Write or fsync failures mark durability lost rather
than succeeding silently; torn, corrupt, or non-durable journals are surfaced as
non-recoverable rather than silently ignored. `list_recovery` inspects both
v2 UUID journals and legacy title-named journals conservatively, and `recover`
(`pedacito recover list|resume|rollback|cleanup`, preview-by-default, `--apply`
to mutate) verifies cross-process locks, maintenance safety, ownership, and
SHA-256 checkpoint manifests before restoring or cleaning up, refusing to mark a
task terminal if restore or terminal durability fails. A stale `run.lock` left
by a dead process is taken over by `acquire_lock`.

### SessionResourceRegistry and cancellation

Anything a session starts registers a cleanup with its `SessionResourceRegistry`
(`resources.py`): dev servers, LSP and MCP clients, temporary mirrors, ephemeral
worktrees and review servers. `SessionContext.cancel()` denies pending prompts,
releases the model lease, runs `teardown_all` in reverse order, and wakes any
review the session is waiting on. Only that session's resources are touched. A
`proc.run` child is stopped through the session's cancel event, and its process
tree is reaped before the cancel returns.

### Bounded parallel subagents

`subagent_max_parallel` (default 3) caps how many subagents run at once. They run
on a `ContextThreadPoolExecutor`, so each one has its own context. A child is
given a `PolicyEngine.child_policy(role=...)` that can only intersect with its
parent. Read-only roles (`explore`, `verifier`) are forced into `plan` mode.

### Package manifests

An extension, skill, command set or model pack declares itself in a
`pedacito-package.toml` at its root: its `kind`, the `capabilities` it asks for
and the minimum Pedacito version it needs. `pkg_manifest.py` validates the file,
and a person reviews it before anything in the package runs. See
[SKILLS_AND_TOOLS.md](SKILLS_AND_TOOLS.md#package-manifests-pedacito-packagetoml).

## Known limits

These are the real boundaries. Missing harness features are not on this list:
the TUI, `pedacito bench`, MCP, LSP, subagents, and multi-file projects are all
shipped.

- **Model size sets the ceiling, not the harness.** A 4B to 8B model is an
  excellent scout and a reliable single-file editor. Multi-file architecture work
  is where 14B to 30B+ local models, or a hybrid `planner`, earn their keep. Broad
  structural tasks such as "modernize this codebase" strain every small model, so
  split them into narrow steps (`pedacito auto` is built for exactly that).
- The call graph records receiver-qualified targets (`Store.get`, `this.x()`,
  typed locals) where the receiver is provable, but an untyped call still matches
  its bare name, so two unrelated `run()` methods can be conflated when the receiver
  is unknown. Refactoring should use the receiver-qualified rename (see
  [Renaming a symbol](#renaming-a-symbol-across-the-project)). Import edges are resolved per language
  (relative paths, packages, `crate::`), but an import it cannot tie to an
  indexed file is treated as external, and the "no importers" hint is evidence,
  not proof (dynamic loading, entry points).
- `py-tree-sitter` 0.26 regressed Node lifetime handling and can segfault
  CPython 3.12/3.14 (upstream py-tree-sitter#472; reproducible without
  Pedacito). The dependency is pinned to `<0.26` (0.25.2 has cp314 wheels);
  CI covers 3.11–3.14. The chunker also pauses the cyclic GC around live
  nodes as a second line of defence.
- `run` and test commands are sandboxed where bubblewrap/sandbox-exec work (not
  on Windows, not where user namespaces are disabled -- you get a notice, not a
  silent downgrade); `browser`, `fetch`, MCP, background `serve` processes and
  LSP servers are guarded (scrubbed environment, protected paths, allowlists,
  trust prompts), not sandboxed.
- `rename` is mechanical and conservative. A bare name renames only when it has
  exactly one definition anywhere, then follows imports, not types. The agent
  tool additionally takes a receiver-qualified query (`Store.save`): Python call
  sites are renamed when the receiver is deterministically bound to that class
  (`x = Store()`, `s: Store`), and everything it cannot prove -- `y.save` on
  another class, untyped receivers, files that never import the definition -- is
  listed as "not renamed" rather than guessed; receiver tracking is
  Python-only. Every multi-file write is transactional: a snapshot or disk
  error partway rolls back the files already written in that batch, so no
  half-renamed tree survives. Several edit blocks across files are still
  applied per file unless `atomic_edits = true` (the snapshot and
  `pedacito restore` are the safety net either way).
- Languages: Python by AST; JS/TS, Ruby, Go, Rust, Java, C, C++, C#, PHP,
  Kotlin, Swift, Scala and Lua through tree-sitter (a grammar quirk can drop an
  occasional Swift call edge); everything else is searchable reference text.
- Semantic search needs an embeddings endpoint. It runs candidate-first: a
  no-hit query embeds only the bounded lexical shortlist (top 50 units) and the
  two rankings are merged with reciprocal-rank fusion -- never one pass over
  the whole project. Vectors are cached by content hash and reused until the
  unit changes or the embedding model is switched.
- A text-only model can't see graphics; `screenshot` and `browser` give it text
  evidence, not visual judgement.
- Narrow, localized tasks are where small models work best. Start there, then
  widen the scope as your model size and hardware allow.
