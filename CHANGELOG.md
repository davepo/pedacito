# Changelog

All notable changes to Pedacito (`pdc`) are documented here.

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
