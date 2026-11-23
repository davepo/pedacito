# Changelog

All notable changes to Pedacito (`pdc`) are documented here.

## v0.3.2 — 2026-11-23

### Interleaved Execution, Native Providers, and Strata Onboarding

- **Execution engine (`execution_mode`)** — `two_phase` (the existing
  gather-then-answer loop, still the default), `interleaved` (one native
  tool-calling conversation: read, edit, run a scoped test, read the traceback,
  edit, verify; supports OpenAI `tool_calls`, Anthropic `tool_use`, JSON
  fallback, and Qwen3.5/3.8 / Strata XML `<tool_call><function=...>` blocks)
  and `auto` (interleaved for hosted APIs, local-MoE servers and calibrated
  large local models with native tool calls; two-phase for tiny, small and
  medium local models and for uncalibrated ones). Set it in config, per run
  with `--execution-mode {two_phase,interleaved,auto}` on `ask`, `new`, `auto`
  and `eval`, or live in the TUI with `/mode` (alias `/exec-mode`). The GUI
  shows it under Profiles → Reasoning & Execution Overrides.
- **Shadow workspace for interleaved runs** — `pedacito/workspace_overlay.py`
  (`VirtualWorkspace`) stages every edit in memory for dry runs and `--review`.
  Tests and linters run against a temporary mirror with the overlay applied, so
  the repository on disk is untouched until you apply. Applied writes are atomic
  and journalled, so a rollback restores the exact bytes. The loop itself is in
  `pedacito/interleaved.py`.
- **Native Gemini and OpenAI Responses transports** — `gemini_native` (default
  `true`) sends Google Gemini through the native `generateContent` API: thought
  parts, inline images, `thinkingConfig` and `cachedContent`. `openai_responses`
  (default `false`, or `provider = "openai-responses"`) uses `/v1/responses` for
  o3, o4 and GPT-5 reasoning models, keeping encrypted reasoning items and
  `previous_response_id` across tool loops.
- **Three-tier provider model list and pricing** — context windows and $/Mtok
  come from the provider's own `/models` metadata (tier 1), a 24-hour
  `models.dev` cache (tier 2, `pedacito/model_catalog.py`) and a built-in price
  table (tier 3). `auto_refresh_model_catalog` (default `true`) refreshes the
  cache quietly. `pedacito models refresh`, Add Hosted Provider → Refresh catalog
  and `POST /api/providers/refresh` force a refresh. Hosted rows show context and
  price columns.
- **Optional OS keychain** — `api_key = "keyring:<service>/<account>"` or
  `api_key_keyring = "<service>/<account>"` read the key from the system
  keychain once `pip install "pedacito[keyring]"` is installed
  (`pedacito/secrets_store.py`). Core installs have no keychain dependency.
- **Strata / Avalon onboarding** — the server list has a Strata / Avalon (Local
  MoE) preset at `http://127.0.0.1:8080/v1`. The scan has an API key field and
  reads `STRATA_API_KEY` (or `AVALON_API_KEY`) when it is left empty. A server
  that requires a key shows an **API key required** badge. Model listing works
  through the public `/health` endpoint when `/v1/models` answers 401. Lemonade
  and Strata can share one machine across GUI, TUI, and CLI: `auto_evict_servers`
  (default `true`) synchronously unloads the other server's unpinned model
  (`POST /v1/unload?wait=15` `{}` for Strata) and re-runs `auto_fit` on the freed
  VRAM before loading the target profile.
- **VRAM auto-detection & LAN-safe `auto_fit`** — Reset to Baseline and loopback
  Model Scanner imports auto-detect this machine's local GPU VRAM instead of
  hardcoding `8 GB`; `auto_fit` warns if loopback `vram_gb` is below the real
  local GPU, while LAN server profiles (`192.168.x.x` / `*.local`) ignore the
  client's `[universal] vram_gb` and trust the remote server's reported context
  unless the profile explicitly sets its own `vram_gb`.
- **Live GUI RAM/VRAM telemetry** — a new `/api/hardware` endpoint, a 3-second
  poll and an SSE `hardware` push keep the top-bar memory badge current. Server
  probes run in parallel, so one slow server cannot stall the badge. The badge
  shows totals only, with no per-process GPU attribution.
- **Collapsible Dual-GPU / Scout + Architect Pairing** — the pairing panel in
  Profiles is collapsed by default and its header shows Paired or Not
  configured. Saving writes `default_profile` and `scout_profile` in one
  comment-preserving edit, and a starter Scout profile can be created from it.

### Calibrated Temperatures, GGUF Architecture Detection, and Creative Mode

- **GGUF & runtime base-family detection** — calibration infers the underlying
  model family of custom fine-tunes (such as `DaisyRose-27B` → `qwen3.5`) from
  Ollama/LM Studio/llama-server metadata, local GGUF v2/v3 headers
  (`general.architecture` / `general.base_model.0.name`), or a 30-day cached
  anonymous HuggingFace Hub metadata fallback for LAN servers. LiquidAI
  (`lfm2.5-24b-a2b`, `lfm3`, `lfm`) and `starcoder` families are included,
  always-thinking warmup gets 512 tokens, `Calibrate All` skips individual
  failing profiles cleanly, and the GUI adds a Quick/Full calibration mode
  picker while preserving Strata's full `max_context`.
- **Architecture temperature override** — a calibrated local model card's
  architecture-library temperature replaces only default-level values, is shown
  directly in `pedacito config --check` and GUI Settings (Profiles & All
  settings), and still yields to an explicit per-profile, env, or CLI override.
- **`creative_temperature_delta` & moderate-window fit** — the Creative/design
  sway is relative to each answer baseline (+0.20 for tiny/small, +0.30
  otherwise, capped at 0.95 / 1.0 explicit). On moderate-context local models
  (e.g. 13k), `--creative` reclaims headroom from its 8,192-token reply boost
  before failing over-budget.
- **CLI & verification hardening** — `pedacito diagnose` accepts a positional
  project path, `models scan` normalizes localhost/127.0.0.1 URLs to prevent
  duplicate profiles, `ask --apply` exits non-zero when budget exhausts with
  failed edit blocks, and the placeholder gate catches single unfinished
  top-level `pass` / `...` stubs.

### Hunk-Level Review and Scoped Shell Allow-Rules

- **Hunk-level sign-off** — the Changes pane lists each change block in a file as
  **Hunk N** with an *accept this hunk only* checkbox. Accepting some hunks
  writes only those, and the SHA-256 stale-file check still covers the whole file.
- **`shell_allowlist_by_project`** — per-project shell allow-rules for `run`,
  keyed by absolute project root. **Allow '<pattern>' in this project** (GUI) and
  `p` (TUI) save one narrow rule for the current project only. Package installs,
  shells, never-allow executables (`rm`, `sudo`, `curl` and similar) and chained
  commands are never covered. Like `shell_allowlist`, a project file's rules
  apply only after `pedacito trust`.
- **Discovery auth** — per-origin auth is passed during discovery, idle status is
  parsed, and a 401 is surfaced when a model list loads instead of an empty list.

### Bilingual Parity and Documentation

- **EN / ES parity for every new surface** — each new string (CLI help, TUI `/mode`
  and `p`, GUI labels and tooltips, the hunk and pairing panels, the hosted
  catalog columns and the Strata auth hints) ships in `en.json` and `es.json`.
- **`pedacito how` covers v0.3.2** — a `models refresh` card, and topics for the
  hunk sign-off, project shell rules, Strata onboarding, the model catalog
  refresh, the hardware badge and the creative sway setting. The calibrated
  temperature rule and the native transports are in the existing topics.
- **Docs mirrors** — `docs/CONFIG.md`, `docs/USAGE.md`, `docs/INSTALLATION.md`,
  `docs/SECURITY.md`, `MODEL_GUIDE.md`, `QUICKSTART.md` and their Spanish mirrors
  describe the same behaviour, and each is covered by the link and config-field
  guards.

### Release Hygiene

- Version is `0.3.2` in `pyproject.toml` and `pedacito/__init__.py`.
- `RELEASE.md` describes the current procedure: release gates, the wheel bundle
  check, and the verification steps for the release.

## v0.3.1 — 2026-11-18

### Bilingual EN / ES, Multimodal Input, Native Anthropic Caching

- **Bilingual EN/ES** — GUI top-bar toggle, TUI `F3` / `/lang`, CLI `--lang es`
  / `PEDACITO_LANG`; `llm_reply_language` sets the model's reply language.
  Every guide, `how` card and locale has a Spanish edition (`docs/es/`,
  `*.es.md`), guarded by a locale/link/anchor integrity suite.
- **Local image generation** — `pedacito image`, TUI `/image` and a GUI
  generator dialog drive a local Lemonade Server (`sd-cpp`); model scanning
  detects image models (`modality = "image"`) and calibration covers them.
- **TUI fixes and multi-line composer** — the cursor is restored on exit;
  `Shift+Enter`, `Alt+Enter`, `Ctrl+J` and a trailing `\` + `Enter` insert
  newlines, and bracketed paste (`?2004h`) keeps pasted blocks intact.
- **GUI polish** — uncrowded top bar and a collapsible, closable model-scan
  list.
- **Multimodal image input** — `--image PATH` (CLI, `new`), GUI paste
  (`Ctrl+V` / `Cmd+V`), drag-and-drop and `📎 Attach`, and TUI
  `/attach [PATH | list | clear]` / `@image.png`; PNG/JPEG/WebP/GIF, max 4
  images of 10 MB each.
- **Native Anthropic `/v1/messages`** — `anthropic_native` (default `true`)
  uses the native transport with explicit `cache_control` breakpoints,
  `5m`/`1h` TTL, `cache_read_input_tokens` / `cache_creation_input_tokens`
  accounting, native extended thinking, tool calls and images; `false` falls
  back to the OpenAI-compatible layer.

### Unified GUI + TUI Color Schemes

- **One shared set of nine built-in schemes** — the theme engine
  (`themes.py`) now ships exactly the GUI's crafted `gui_theme` schemes:
  `slate` (the default, GitHub Dark), `light`, `daylight`, `nord`,
  `midnight`, `jazz_solo`, `trapper_keeper`, `memphis_arcade` and
  `windbreaker`. The TUI's `/theme` and the GUI Theme Manager offer the
  same names and the same palettes; the Theme Manager shows one pill per
  scheme and paints built-ins from their full crafted CSS blocks, not the
  flattened ten-token derivation.
- **Legacy presets removed** — `pedacito-dark`, `catppuccin-mocha`,
  `tokyo-night`, `gruvbox-dark` and `solarized-dark` are gone; a saved
  preference naming one of them falls back to `slate`. `nord` keeps its
  name but now uses the crafted modern-dark palette.
- Docs: CONFIG's Color schemes section, USAGE, QUICKSTART, README and the
  built-in `how` change-theme card name the new set.

### Docs Ownership Split

- **Ownership decision** — `docs/` (INSTALLATION, ARCHITECTURE, USAGE, CONFIG,
  SECURITY) is the canonical reference. QUICKSTART, MODEL_GUIDE and TECHNIQUES
  remain the canonical task-oriented guides, README is the index, and this
  CHANGELOG is the history. The decision is written once in README's
  Documentation section and echoed in `docs/ARCHITECTURE.md`. New reference
  prose goes to `docs/`, new walkthroughs go to the root guides, and each set
  links to the other instead of copying text.
- **Cross-links** — each root guide now points into `docs/` (MODEL_GUIDE and
  TECHNIQUES previously did not), with no guide content moved or rewritten.
- **Test guard** — `tests/test_doc_sync.py` pins the root-guide-to-docs links,
  README's links to all three guides and `docs/`, the echoed paragraph, and
  this entry.

### Manual Skill Injection and Repo-Map Regression Pins

- **TUI `/skill NAME`** — loads the full checklist through
  `skills.load(root, cfg, NAME)` and pins it as a normal session pin
  (`skill:NAME`), so it reaches the next task through the existing
  **Explicitly Pinned Context** block, above the map, not through a parallel
  injection channel. A receipt names the source and body size; an unknown name
  prints the loader's error plus every available skill, and no argument prints
  usage. Registered in the slash-command list and `/help`.
- **GUI “Use in next task”** — the existing Skills library & creator dialog
  gains a per-skill action plus a pinned-for-the-next-task box showing each
  selected skill, its scope, description and body size, with Remove and Clear
  skills actions. It writes `skill:NAME` into the composer's existing `pinned`
  request list, so it uses the same pinned-context path as @file/@symbol; no
  new slash-command parser was added.
- **Regression pins** — tests cover byte-identical map text for the same
  index + task (including a cleared cache and a fresh index), byte-identical
  system + user0 across gather steps, the flat map's
  `flat_map_token_budget` estimate, `.venv`/`node_modules`/`.pedacito` and
  ignore-file exclusions, project > user > bundled skill precedence, the
  `skill` tool returning the full body, and `/skill` landing in pinned context
  and surviving `_trim_gathered` pruning.
- Docs: USAGE's skills/injection section and TUI command list, and
  ARCHITECTURE's repo-map/KV-prefix note (map in user0, deterministic ranking,
  the absolute budget key). Version unchanged.

### Headless Task Summary (`pdc ask --json`)

- **`pdc ask --json`** — one machine-readable summary object on stdout at run
  end, with every human/log line moved to stderr. Shape: `{ok, model,
  provider, turns, elapsed_sec, tokens{prompt, completion, tok_per_sec},
  files_changed, verify_passed, diff, diff_truncated}`. `turns` is the run's
  model-call count (`events.RunTelemetry.summary()["calls"]`), tokens and
  timing come from the same run telemetry, `files_changed`/`diff` come from
  the applied changes (capped at 200,000 characters), and `verify_passed` is
  `null` when no verification ran. `ok` reports task success while the
  process exit code keeps its current semantics. `--json` implies
  `--no-review` (no browser, pager or prompts) and is mutually exclusive with
  `--jsonl`.
- **`pdc ask --verify-cmd "COMMAND"`** — overrides `auto_verify` for that one
  run in memory, with a `sources` entry of `--verify-cmd`; it is never written
  to a config file. The command runs once after an applied task and its
  outcome is the JSON summary's `verify_passed`.
- Docs: USAGE's Headless and automation section with the exact shape and two
  runnable examples, a CONFIG verification cross-reference, and the built-in
  `how` ask card. Tests cover the one-object stdout contract, the
  `--verify-cmd` override (config bytes untouched), `verify_passed: null`, the
  diff cap, `--json` + `--jsonl` rejection, and secret redaction. Also keeps
  `safety.redact` linear on very long unbroken lines (a multi-hundred-kilobyte
  diff no longer stalls on the assignment regex).

### Trajectory Export (`pdc ask --save-trajectory`)

- **`pedacito/trajectories.py`** — opt-in full-prompt ChatML/OpenAI
  transcripts for SFT/GRPO dataset filtering, written one JSON file per task
  run to `.pedacito/trajectories/<UTC timestamp>_<short id>.json` as
  `{"metadata": {...}, "messages": [...]}`. The messages are the real turns
  sent and received (system, user, assistant, model tool calls and tool
  results), not a reconstruction. Metadata carries `verify_passed` (bool or
  null), `harness_version` (`pedacito.__version__`), `edit_format`,
  `model_id`, `provider`, a SHA-256 `task_hash` (never the task text),
  `started_at`, `elapsed_s` and token totals from the run telemetry.
- **`record_trajectories = false`** config knob, plus
  **`pdc ask --save-trajectory`** as a one-run override in memory (live
  `sources` entry `--save-trajectory`; the config key and file are untouched).
  Capture hooks `loop.run_task`, so the knob covers `ask`, `chat`, `auto`,
  `rpc` and `eval`; every capture path is exception-safe and a write failure
  leaves the task unaffected.
- **Mandatory scrubbing** — every string value passes `safety.redact`; the
  serialised transcript is then scanned with
  `security.scan_text_for_secrets`, and a residual finding is masked or the
  write is refused with a warning. A detected secret is never persisted.
  Local files only: no network and no telemetry, under the already-gitignored
  `.pedacito/`.
- Docs: SECURITY's "Trajectory export" section (opt-in, full prompts unlike
  replay's hash + preview, same redaction pipeline, local-only, user-owned
  files), USAGE's metadata filtering example, and a CONFIG reference row.
  Tests cover the default-off no-write path, the flag and config-knob
  captures, metadata/ChatML shape, a planted API key in a tool result, and a
  capture failure that does not fail the task.

### Post-Edit Verification Controls

- **TUI `/verify on|off|status`** — `on` detects the project's test runner
  (`testsel.detect_test_command`) and fills an empty `auto_verify` command
  string, `off` clears it, and `status` reports the command, static gate,
  repair rounds, scoped-test behavior, green ratchet and interface lock.
  Session-only, like the existing `/think` and `/apply` toggles, with a
  `cfg.sources` entry. Registered in the slash-command list and `/help`.
- **GUI “Post-edit verify” checkbox** — sits beside the ✦ Auto-route toggle
  and writes `auto_verify` through the existing `/api/settings/set` plumbing,
  using the same detect-assist on enable. Its tooltip shows the detected test
  command, and `/api/state` now exposes `auto_verify` (the command string) and
  `detected_test_command` so the checkbox reflects the live config.
- **Unchanged gates** — `static_gate`/`static_gate_timeout`, `test_timeout`,
  `auto_repair_rounds`, `auto_verify_rounds`, `auto_green_ratchet` and
  `interface_lock` keep their existing defaults and behavior; no boolean
  `auto_verify` key was added.
- Docs: CONFIG's verification-tier section and reference rows, USAGE's
  three-tier description with the real 3 s/`test_timeout`/verify-command
  boundaries, and tests for both surfaces plus byte-identical untouched
  configs.

### Advanced System Profiling & Model Auto-Tuning

- **`pedacito/profiler.py`** — a vendor-neutral snapshot layered over the
  unchanged `servers.get_hardware_memory()` shape: Linux DRM sysfs (AMD,
  Intel xe, any `mem_info_vram_*` driver), NVIDIA `nvidia-smi` via argv,
  macOS `sysctl` + `vm_stat` (Apple Silicon unified memory as RAM), and
  Windows `kernel32.GlobalMemoryStatusEx` through stdlib `ctypes`. Every
  reader is guarded, time-boxed to 2 s and silent when absent; every number
  carries its source.
- **Footprint estimation** — dense and MoE weight residency, active-expert
  VRAM, KV GB per 1k tokens from layers × KV heads × head dim × quant bytes,
  architecture metadata from explicit config, llama.cpp `/props` + `/models`,
  Ollama `/api/show`, then model-name heuristics. Missing fields lower
  `confidence`; they are never presented as measured.
- **Speed/prefill probe** — one 16-token completion plus a ~2k-token filler
  (512 when the window is smaller), timed from SSE. Explicit (`pdc profile
  --run`, TUI `/profile run`) or once per server+model on a daemon thread
  after a successful load; cached for a week in `profiler.json`.
- **`auto_tune`** (default **false**) — opts into context/retrieval/timeout
  tuning from free VRAM or unified RAM. Tight pools shrink to a 4,096-token
  floor, spare headroom is proposed but never forced, explicit values win,
  tiered local-MoE keeps concurrency 1 and its 600 s timeout, and missing
  telemetry abstains instead of assuming zero free memory.
- **Surfaces** — `pdc profile --show|--run|--reset`, `pdc models` tuned
  numbers, TUI `/profile show|run|reset`, and the GUI live telemetry bar now
  populated on NVIDIA/macOS/Windows (totals only, no per-process GPU claim).
- Docs: CONFIG profiling section and reference rows, ARCHITECTURE memory
  manager, README/QUICKSTART, and a how-card.

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

### README Modularization into `docs/`

- **README split** — the detailed reference moved out of `README.md` into
  `docs/INSTALLATION.md`, `docs/ARCHITECTURE.md`, `docs/SECURITY.md`,
  `docs/CONFIG.md` and `docs/USAGE.md`. The README is now the summary, the
  30-second install, the release tables and the doc index.
- **Doc audit fixes** — Strata/Avalon rows in the INSTALLATION server table and
  the `local-moe` tier note; SECURITY filler-detection wording matches
  `safety.redact`; the trusted-project-config list now names `load_cmd`,
  `unload_cmd`, `enable_mcp`, `allow_mcp`, the sandbox and scrub keys and
  `verify_tests`, `verify_dom`, `dom_entry`; the checkpoint section covers
  `⑂ Branch Prompt From Here`, `pedacito restore --snapshot` and empty-directory
  cleanup on `--delete-created`; the permission dialog button is named
  **Deny & Redirect**; the INSTALLATION `config --init` starter is documented.
- **Test guard** — `tests/test_doc_sync.py` checks the modular corpus
  (README plus `docs/*.md`) for every CLI subcommand and feature needle, and
  the how-cards against the same features.

### Profile Availability in the GUI Picker

- **`/api/servers`** now includes `profile_availability`
  (`available` / `unavailable` / `offline` / `unknown`) computed from the
  server states that poll already fetched; the latest map is cached on
  `STATE`. No new interval and no new network call, and hosted endpoints are
  never probed.
- **`POST /api/profile`** refuses a profile whose model is no longer listed by
  its server with a 409 and an actionable detail. Unknown, offline and
  not-yet-polled states fail open; TUI and CLI switching are unchanged.
- **GUI profile picker** — unavailable options are disabled and suffixed
  **· unavailable** with a model-not-found tooltip; offline options are
  disabled and suffixed **· server offline** until a later poll. The
  **(default)** option stays selectable; an active profile that becomes
  unavailable remains shown with its state, and the GUI never auto-switches
  away from it.
- Docs: USAGE explains the states and their refresh via the existing 15 s
  telemetry poll.

### Project Website and Contact

- **`https://www.pedacito.dev`** and **`contact@pedacito.dev`** now appear in
  the GUI Settings dialog footer, the TUI `/help` output, the README and the
  SECURITY vulnerability-reporting path, and as `[project.urls]` entries for
  PyPI. Interface furniture and docs only; task prompts, agent output and
  telemetry are unchanged.

### Agent Security Guide (`docs/SECURITY.md`)

- A new guide for running coding agents safely, organized around four pillars:
  **credential isolation and pre-flight scanning**, **strict workspace
  sandboxing**, **least-privilege execution** (OS sandbox, timeouts, command
  policy, permission broker) and an **immutable audit trail with time travel**
  (checkpoints, inspect-before-undo, roll back, retention).
- It also carries the threat model, a summary checklist, known limits and the
  vulnerability reporting path.

### Run-tab Environment Credential Scrub

- **`pedacito/runner.py`** — Run-tab child processes no longer inherit
  credential variables from the exported shell. Names matching `*_API_KEY`,
  `*SECRET*`, `*TOKEN*`, `*PASSWORD*`, `*ACCESS_KEY*` or `*CREDENTIAL*`
  (case-insensitive), plus `GITHUB_TOKEN` and `PYPI_API_KEY`, are dropped from
  the child environment before the process starts.

### Bundled Safe Commit Skill

- **`safe-git-commit`** — a nineteenth bundled skill for commit requests:
  inspect the uncommitted diff with `git_diff`, match the repo's message style
  with read-only `git log`, stage only the files the task touched, and hand the
  user the exact `git add` / `git commit` / `git push` commands because the
  sandbox keeps `.git` read-only. It forbids destructive git commands in favour
  of `pdc undo` and checkpoints, and says so plainly when git is missing or the
  project is not a repository — git remains optional.

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
pedacito config --upgrade
pedacito calibrate --all   # optional: refresh model cards and specialty tags
```

No configuration migration is required for `v0.1.0` users; `pedacito config
--upgrade` adds the new defaults.
