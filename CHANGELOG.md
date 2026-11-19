# Changelog

All notable changes to Pedacito (`pdc`) are documented here.

## v0.3.1 — unreleased

### Context-Aware Auto-Update (opt-out, local-first)

- **`pedacito/update.py`** — one anonymous GET of the public PyPI metadata at
  GUI/TUI/agent startup, on a daemon thread (2 s timeout, never blocks
  startup), cached 24 h beside the user config (`update_check.json`).
  Offline, timeout, non-200 or malformed JSON is a silent no-op. Semver-aware
  comparison (pre-releases sort before their final release), yanked releases
  ignored, never nags when installed >= latest.
- **Install-method detection** — pipx / editable / venv / user-pip / unknown,
  from local heuristics (package path, `__editable__` finders, egg-links,
  `pyvenv.cfg`, guarded `pipx list --json`). The upgrade is always an argv
  list run with `shell=False`; editable checkouts get `git pull && pip
  install -e .` instructions instead of a command, and PEP 668
  externally-managed Pythons get a pipx/venv note.
- **GUI** — slim dismissible banner (*Pedacito vX.Y.Z is available*), Update
  streams the upgrade into an output drawer, exit 0 offers Restart Pedacito
  (re-exec on the same port); failure or editable shows the manual command.
  In-flow, never covers controls, no modals.
- **CLI** — `pedacito update [--run]`; **TUI** — `/update [run]`.
- **`check_for_updates`** (default true) — the single opt-out: false means
  zero network calls and zero upgrade subprocesses everywhere; the app
  behaves exactly as before. Also `PEDACITO_CHECK_FOR_UPDATES=false`.
- Docs: INSTALLATION "Upgrading", CONFIG reference row, README feature row,
  QUICKSTART B1, SECURITY network-boundary note, ARCHITECTURE phone-home
  clarification; how-cards for the CLI command, the GUI topic and the setting.

## v0.3.0 — 2026-11-18

The release built around **time travel, live feedback and two-model
workflows**: every run is checkpointed and undoable, secrets are guarded
before they leave, tokens stream live, and a strong architect can plan for a
fast editor — or two models can compete in isolated sandboxes.

### Split-Stage Hybrid Routing (Architect + Editor)

- **`hybrid_routing` + `architect_provider` / `architect_model`** — Stage 1
  asks a strong architect model for a `HYBRID_BLUEPRINT` (root cause, target
  files, exact signatures and contracts, edge cases); Stage 2 injects the
  blueprint ahead of the fast editor's gathered context.
- **Native VRAM handoff between stages** — when both models live on different
  local servers, the outgoing server is evicted at the Stage 1 → 2 transition
  (`auto_evict_servers`); hosted endpoints coexist.
- **Combined + per-stage telemetry** — architect and editor tokens fold into
  one run aggregate with a per-stage breakdown; an unreachable architect
  degrades to a plain editor run with one warning.
- **GUI Options bar** — Hybrid Architect toggle + provider/model selector;
  **TUI** — `/hybrid [off | <provider> <model>]`.

### 2-Model Sandbox Arena

- **`pedacito/arena.py`** — one prompt, two contestants, each in an isolated
  sandbox copy of the workspace (no `.git`, `.venv`, `node_modules`,
  checkpoints, `dist` or `build`). The real project is never touched.
- **Side-by-side scorecard** — per contestant: ok, files changed, unified
  diff, elapsed time, tok/s, total tokens and a summary; stored in
  `.pedacito/arena_last.json`.
- **Checkpointed apply** — `apply_arena_winner` snapshots before and after
  adopting the winner (`Arena winner: <model>`), so every arena result is
  undoable.
- **GUI** — Arena (A/B) button and modal with Apply buttons
  (`/api/arena/run|last|apply`); **TUI** — `/arena <a> <b> <prompt>` and
  `/arena apply a|b`.

### Time-Travel Checkpoints & Live Preview (Phase 3A)

- Interactive checkpoint tree in the GUI and TUI (`/checkpoints`,
  `/checkout`, `/undo`), with per-run baselines and manual pins.
- Live app preview runner: detected entrypoints and web targets run from the
  GUI with output streaming.

### Context Pinning & Project Rules (Phase 2B)

- `@file` and `@symbol` context pins in the composer and TUI (`/pin`),
  carried into every phase of a run.
- Persistent project rules editor (`.pedacito/rules.md`) in both surfaces.

### Custom Color Schemes (Phase 2A)

- Unified GUI + TUI theme engine with shared custom color schemes
  (`/theme`, GUI Theme Manager).

### Live Streaming & Instant Cancel (Phase 1C)

- SSE token streaming with a live reasoning drawer, per-call heartbeat and
  millisecond-scale Stop.

### Secret & Credential Guard (Phase 1B)

- Non-blocking outbound secret scanning with acknowledge/redact choices,
  post-run workspace notices and `.env` gitignore helpers.

### Cross-Server Memory & Strata (Phase 1A)

- Cross-server memory manager for local model servers and local-moe Strata
  support.

## v0.2.0 — 2026-11-15

The first release built around **specialist routing and skill-driven workflows**:
models are matched to the work they actually measure well at, sessions survive
across turns, and the CLI, TUI and GUI now expose the same telemetry and
controls.

### Specialist Auto-Routing (`--auto-route`) & Dual-GPU Scout/Architect Pairing

- **Composite specialist scoring** — each task is scored against the
  `SPECIALTIES` vocabulary (keywords plus file extensions) and each profile
  against its measured model card: `0.40 × domain probe + 0.35 × edit_format +
  0.25 × overall`, with a small tier tie-break.
- **`edit_format >= 0.65` competence gate** — a one-shot probe answer is not
  enough to route editing work. Every editing specialty (coding, frontend/UI,
  3D/SVG, math, testing, SQL) additionally requires a measured edit format and
  tool-choice score.
- **Orchestrator tie-breaking** — an active router/orchestrator profile never
  hoards specialist work it cannot do itself; a candidate that leads the
  active profile by a composite margin takes the task. Ties prefer a resident
  model, then a profile on a separate server.
- **Zero-edit coding fallback** — when a routed specialist finishes with
  0 applied edits / 0 files written (hidden-reasoning runaway, empty answer),
  the task is retried once on the top-scoring local coding specialist that
  clears the edit-format gate, instead of being marked failed.
- **Native VRAM handoff** — on a shared local server the outgoing model is
  unloaded before the specialist loads, using the native Lemonade, LM Studio,
  or Ollama unload API; profiles on different GPUs stay warm together.
- **Dual-GPU scout/architect pairing** — `scout_profile` runs the gather-phase
  lookups on a fast model while the architect keeps planning, synthesis and
  edits.

### Base Architecture Library (`arch_defaults.py`) & Calibration Upgrades

- **Empirical micro-probes** for `sql_data`, `testing_qa`, `svg_ui`,
  `math_algo`, `translation` and `orchestrator_plan`, recorded on every model
  card with plain-language notes.
- **`pedacito calibrate --all`** — calibrate every discovered local profile in
  one pass, persisting per-profile windows and specialty tags.
- **Architecture-aware defaults** — context-window and capability hints are
  derived from the model architecture family when the server does not report
  them, so a stale card cannot silently cap a large model.
- **Hosted model calibration lockout** — hosted APIs are protected from
  accidental calibration spend; hosted cards are marked, not probed.
- **GUI calibration status badges** — model discovery shows measured,
  stale, hosted and uncalibrated state at a glance.

### 16 Built-In Skills & Custom Skill Creator

- **Expanded playbook library** — 16 built-in skills covering common
  engineering workflows, loaded automatically when a task matches.
- **`pedacito skills add/draft/show/rm`** — create a skill from a draft, add
  it to the library, inspect its playbook, or remove it.
- **GUI Skills Manager** — browse, edit and delete skills without leaving the
  browser UI.

### Session Consolidation (`/consolidate`)

- Compacts a multi-turn session into `.pedacito/consolidated_session.md`,
  preserving goals, decisions, findings and open work while dropping
  redundant tool churn.
- Available from the CLI, TUI (`/consolidate`), GUI and RPC, so long-running
  sessions can continue with a small, durable context.

### Interactive Split-Screen TUI (`pedacito tui`)

- Zero-dependency, cross-platform ANSI/VT100 interface with a transcript pane
  and live telemetry pane.
- Native-selection mouse-wheel scrolling using `?1007h`, plus `PgUp`/`PgDn`
  keyboard paging and a scrollback buffer.
- `Tab` completion for slash commands, paths and task prompts.
- Live 150 ms telemetry tailing of the background server, token burn and model
  handoffs.

### Live Telemetry & Full GUI–CLI Parity

- **Visual Context Meter**, **Token Burn Rate**, **`tok/s` speed metrics** and
  **KV Cache Hit %** in the GUI footer, CLI ribbon and `/status`.
- **GUI MCP Server Editor** — add, edit and test MCP server definitions.
- **Project Reset** — restore a project to its baseline state from the GUI.
- **Memory & Repo Lessons modal** — inspect and clear failure memory and
  repository lessons.
- **Background Server (`procs`) manager** — start, stop and inspect background
  model servers and jobs from the GUI.
- GUI actions now map to the same underlying CLI/RPC operations, so both
  surfaces report identical state.

### Security & Reliability Hardening

- **Bug Hunts #1 through #14** fixes across the edit loop, routing, retrieval,
  TUI, GUI, RPC and model-server integrations.
- **2,840+ deterministic tests** — unit, property/fuzz, security and
  cross-platform portability suites, plus the encoding-warning gate.

### Upgrading

```bash
pip install -e ".[dev]"
pedacito config upgrade
pedacito calibrate --all   # optional: refresh model cards and specialty tags
```

No configuration migration is required for `v0.1.0` users; `pedacito config
upgrade` adds the new defaults.
