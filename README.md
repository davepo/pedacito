🌐 **English** | [Español](README.es.md)

# Pedacito

[![PyPI](https://img.shields.io/badge/PyPI-v0.3.3-blue)](https://pypi.org/project/pedacito/)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)
[![MIT License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
![Local-First](https://img.shields.io/badge/Local--First-yes-brightgreen)
![EN / ES](https://img.shields.io/badge/Docs-EN%20%2F%20ES-orange)

**Pedacito** (`pdc`) is a local-first AI coding agent for the models you can run on the GPU you
already own, from 4B to 27B parameters. It cuts your codebase into small, precisely bounded
pieces, so a small model reads and edits only the lines that matter, instead of needing a
frontier model with a 200k-token window.

- **Local and hybrid servers:** LM Studio, Ollama, Lemonade Server, llama.cpp, vLLM, Strata/Avalon,
  and any OpenAI-compatible endpoint.
- **Hosted when you want it:** Claude, ChatGPT, DeepSeek, Qwen, Gemini and OpenRouter for planning,
  verification or escalation, with a spend cap.
- **Safe by default:** secret scanning, OS sandbox, checkpoints you can roll back.

![Pedacito GUI workspace: chat with a staged diff, live activity and plan panes, the Composer with Auto-route, Post-edit verify and Creative toggles, and the telemetry bar](docs/assets/gui-workspace.png)

![Pedacito split-screen terminal UI: transcript beside the Activity & Plan sidebar, with the context meter and the EN|ES language switch in the header](docs/assets/tui-split.png)

## 30-second quickstart

```bash
pip install pedacito                 # Windows: py -m pip install pedacito
pdc gui ~/code/your-project          # browser control panel
pdc tui ~/code/your-project          # split-screen terminal
pdc ask "where is login handled?"    # one-shot CLI, from the project folder
```

Start a model server first (LM Studio, Ollama, Lemonade or llama.cpp), then in the GUI open
**Settings → Model discovery → Scan local servers**, or run `pdc models scan`.

> [!TIP]
> Prefer Spanish? Use the `EN | ES` switcher in the GUI top bar, press `F3` in the TUI, or run
> `pdc --lang es gui`. See [Language & localization](docs/CONFIG.md#language--localization-language-llm_reply_language).

## What's new in v0.4.0

| Feature | What you get | Read more |
|---|---|---|
| **Isolated sessions** | Concurrent sessions in one process share no config, index, events, permissions or budgets | [ARCHITECTURE.md](docs/ARCHITECTURE.md#v040-session-isolation-daemon-and-parallel-subagents) |
| **Permission modes & approvals** | `permission_mode` (plan, ask, accept-edits, workspace-write, danger) and `approval_policy`; a project's `.pedacito/policy.toml` only narrows, and only when trusted | [SECURITY.md](docs/SECURITY.md#sensitive-artifacts-policy-trust-and-approvals-v040) |
| **Windows sandbox** | Commands run in a Job Object with a restricted token (`win_restricted`); unattended runs without a sandbox are refused | [SECURITY.md](docs/SECURITY.md#pillar-3-least-privilege-execution) |
| **Session daemon** | `pedacito daemon` is a token-protected loopback service with per-session event replay | [USAGE.md](docs/USAGE.md#the-cli-at-a-glance) |
| **Parallel subagents** | `subagent_max_parallel` bounds concurrent subagents; read-only roles stay read-only | [CONFIG.md](docs/CONFIG.md) |
| **Package manifest** | `pedacito-package.toml` declares what an extension or skill package needs | [SKILLS_AND_TOOLS.md](docs/SKILLS_AND_TOOLS.md#package-manifests-pedacito-packagetoml) |

Full notes: [CHANGELOG.md](CHANGELOG.md#v040--2026-11-26).

## What's new in v0.3.3

| Feature | What you get | Read more |
|---|---|---|
| **Agent Client Protocol bridge** | `pedacito acp` speaks JSON-RPC 2.0 over stdio for editors and agent hosts: `session/new`, `session/prompt` with text, `@file` and image blocks, streamed `session/update` events and `session/request_permission`. It wraps the same warm engine as `rpc`, which gains `--events` and `--describe`. | [ACP bridge](docs/USAGE.md#acp-agent-client-protocol-bridge) |
| **Headless session daemon** | `pedacito daemon` keeps one warm index and the model leases in a loopback-only HTTP/SSE server with isolated workspace sessions, bearer-token auth, attach/detach, permission and review routing, and clean quiesce/shutdown. `pedacito daemon status` exits 0 offline. | [CLI at a glance](docs/USAGE.md#the-cli-at-a-glance) |
| **Rich zero-dependency TUI** | Inline syntax-highlighted diffs, collapsible tool-call cards (`Ctrl+O`, `/expand`, `/collapse`), an `@` fuzzy picker for files, symbols and images, and clickable SGR mouse zones for the language switch, sidebar tabs and scrollbar. | [`tui`](docs/USAGE.md#tui-split-screen-terminal-session) |
| **Local, LAN & Tailscale endpoints** | One locality rule covers loopback, `.local`, RFC 1918, Tailscale/CGNAT (`100.64.0.0/10`), link-local and IPv6 ULA. Local cards without a calibrated temperature are marked stale for recalibration. | [Local endpoints](docs/CONFIG.md#local-lan-and-tailscale-endpoints) |
| **Small-model & sparse-MoE upgrades** | `tool_mode = "plain"` (plain-word gather steps, A/B-tested by `calibrate`), an opt-in `proactive_escalation` router, shadow-tested `best_of`, and an interleaved pre-stop verifier. Oversized tool output spills to `.pedacito/scratch/`, which the model can read but never write. | [Configuration](docs/CONFIG.md#configuration-reference) |
| **Guided git, hardened** | `pedacito commit` and `pedacito pr` run a secret pre-flight scan before every commit and PR, never push or force-push, and are shared by the TUI `/git` and the GUI Git drawer. `undo` refuses to rewind past a moved git HEAD without `--force`. | [Git and pull requests](docs/USAGE.md#git-commit-and-pull-requests) |
| **Reliable runs & CLI polish** | Pre-run checkpoints for divergent trees, `restore --checkpoint`, offline tree-sitter fallback with `parsers prefetch`, `pedacito diag`, `models --profile/--model`, `new --no-complex`, honest partial-batch semantics and a hard `--max-minutes` clamp. | [Offline parsers and diag](docs/USAGE.md#offline-parsers-parsers-prefetch-and-diag) |
| **GUI hardening** | Partial-hunk review sign-off with per-hunk provenance, persistent thread cards, deterministic `/compact` handoffs, a responsive narrow layout, and quiet client disconnects. | [Review page](docs/USAGE.md#review-page) |
| **Windows PowerShell 5.1 note** | On PowerShell 5.1, `2>&1` wraps stderr lines as error records; use `2>$null` for clean JSON output. | [Windows stderr](docs/USAGE.md#windows-powershell-51-capturing-stderr-21) |

## What's new in v0.3.2

| Feature | What you get | Read more |
|---|---|---|
| **Interleaved execution** | `execution_mode` picks `two_phase`, `interleaved` or `auto`. Interleaved runs keep one native tool-calling conversation (OpenAI `tool_calls`, Anthropic `tool_use`, JSON fallback, and Qwen3.5/3.8 / Strata XML `<tool_call>`), with dry runs staged in a shadow workspace. Set it per run with `--execution-mode`, or live with the TUI `/mode`. | [Configuration](docs/CONFIG.md) |
| **Native Gemini & OpenAI Responses** | `gemini_native` (on by default) uses Google's `generateContent`; `openai_responses` uses `/v1/responses` for o3, o4 and GPT-5 reasoning models. | [Configuration](docs/CONFIG.md) |
| **Three-tier model catalog & keychain** | Provider metadata, a 24 h `models.dev` cache and a built-in table supply context windows and prices. `pedacito models refresh` forces an update. `api_key_keyring` reads your OS keychain once `pedacito[keyring]` is installed. | [Configuration](docs/CONFIG.md) |
| **Strata / Avalon onboarding & coexistence** | A one-step preset at port 8080, an API key field in the scan, `STRATA_API_KEY` fallback, an **API key required** badge, and synchronous Lemonade ↔ Strata auto-evict across GUI, TUI and CLI. | [Model server setup](docs/INSTALLATION.md#model-server-setup) |
| **Live RAM/VRAM badge & pairing panel** | The GUI hardware badge refreshes every 3 seconds and is pushed live. Local GPU VRAM is auto-detected on baseline reset/loopback scans while LAN profiles stay client-independent. The Dual-GPU / Scout + Architect Pairing panel starts collapsed. | [GUI workspace](docs/USAGE.md) |
| **Hunk review & project shell rules** | Accept individual hunks in the Changes pane. **Allow in this project** saves a narrow shell rule (`shell_allowlist_by_project`) for the current project only. | [Review page](docs/USAGE.md#review-page) |
| **Guided git & pull requests** | `pedacito commit` scans the staged diff for secrets and commits locally, and `pedacito pr` drafts a grounded title/body and opens it with `gh`. Both never push. The TUI `/git` and the GUI **Git** drawer use the same pre-flight. | [Git and pull requests](docs/USAGE.md#git-commit-and-pull-requests) |
| **GGUF calibration, temperatures & creative sway** | Calibration infers base families for custom fine-tunes from runtime/GGUF headers (or cached HF metadata over LAN), adds a Quick/Full GUI mode picker, surfaces calibrated temperatures in Settings/`config --check`, and adds relative `creative_temperature_delta` sway. | [Configuration](docs/CONFIG.md) |

## What's new in v0.3.1

| Feature | What you get | Read more |
|---|---|---|
| **Bilingual EN / ES UI, TUI & docs** | Switch the GUI with the EN/ES top-bar toggle, the TUI with `F3` or `/lang`, the CLI with `--lang es` or `PEDACITO_LANG`. `llm_reply_language` controls the language of the model's explanations. Every guide has a Spanish edition. | [Language & localization](docs/CONFIG.md#language--localization-language-llm_reply_language) |
| **Advanced system profiling & auto-tune** | `pedacito profile` shows a vendor-neutral RAM/VRAM snapshot with a source per number (Linux sysfs, NVIDIA `nvidia-smi`, macOS `vm_stat`, Windows `winapi`), a cached prefill/generation probe and the active profile's tuning. `auto_tune = true` (opt-in) fits context, retrieval budgets and timeout to free memory; missing telemetry makes it abstain, and existing configs do not change. | [System profiling and auto-tune](docs/CONFIG.md#advanced-system-profiling-and-auto-tune) |
| **Context-aware auto-update** | A quiet, opt-out release check: one anonymous PyPI GET at startup (2 s timeout, cached 24 h), a dismissible GUI banner with a streamed upgrade and Restart, `pdc update --run`, and the TUI `/update`. The command matches your install method (pipx, venv, pip --user, or git for editable checkouts). `check_for_updates = false` keeps Pedacito fully offline. | [Upgrading](docs/INSTALLATION.md#upgrading) |
| **Local image generation** | `pedacito image` and the GUI/TUI `/image` generate images on a local Lemonade Server, with calibration for image models. | [Local image generation](docs/USAGE.md#local-image-generation) |
| **Multimodal image input** | Send PNG/JPEG/WebP/GIF to a vision model: `--image PATH`, GUI paste/drag-drop/`📎 Attach`, TUI `/attach` (4 images, 10 MB each). | [Multimodal chat input](docs/USAGE.md#multimodal-chat-input) |
| **Native Anthropic prompt caching** | `anthropic_native = true` (default) uses `/v1/messages` with explicit `cache_control` breakpoints, `5m`/`1h` TTLs, native thinking, tools and images. | [Hosted prompt caching](docs/CONFIG.md#hosted-prompt-caching) |
| **TUI multi-line composer** | `Shift+Enter`, `Alt+Enter`, `Ctrl+J` or a trailing `\` add newlines; bracketed paste keeps pasted blocks intact. | [`tui`](docs/USAGE.md#tui-split-screen-terminal-session) |

## What's new in v0.3.0

| Feature | What you get | Read more |
|---|---|---|
| **Cross-server VRAM manager** | Treats all your local model servers as one pool. It shows which models are resident, loads and unloads them through each runtime's API, and evicts other servers before a run. | [Server lifecycle manager](docs/ARCHITECTURE.md#server-lifecycle-manager) |
| **Strata local-moe** | Tiered mixture-of-experts servers (Strata, Avalon) get their own execution profile: larger reply budgets, a 10-minute timeout and one stream at a time. | [local-moe tier](docs/ARCHITECTURE.md#execution-profiles-and-the-local-moe-tier) |
| **SSE streaming** | Tokens stream live over server-sent events, with a live reasoning drawer, a per-call heartbeat and a near-instant Stop. | [Live telemetry](docs/USAGE.md#live-telemetry) |
| **Security Guard** | Scans for secrets before any hosted request (`sk-`, `pypi-`, `ghp_`, AWS and Google keys, PEM private keys), keeps sensitive files out of the model's reach, asks before risky actions, and runs commands in an OS sandbox. | [Security guide](docs/SECURITY.md) |
| **Custom themes** | Ten-token color schemes shared by the GUI and the TUI. Nine presets, your own saved schemes, and a per-project pin. | [Color schemes](docs/CONFIG.md#color-schemes) |
| **`@` pinning** | Write `@file.py` or `@Class.method` in a prompt to pin that exact code into every phase of the run. | [Pinning context](docs/USAGE.md#pinning-context-with-) |
| **Checkpoint tree** | Every run is checkpointed. Inspect each change as a unified diff, then roll back with `pdc undo` or `pdc checkout`. | [Checkpoints and rollback](docs/SECURITY.md#pillar-4-immutable-audit-trail-and-time-travel) |
| **Hybrid routing** | A strong architect model plans a task and a fast local editor implements it. On one GPU, the architect's server is evicted between the two stages. | [Hybrid routing](docs/ARCHITECTURE.md#hybrid-routing-an-architect-plans-an-editor-implements) |


## Documentation

| Guide | What it covers |
|---|---|
| [Installation](docs/INSTALLATION.md) | pip, pipx, extras, model servers, GPU inference servers, upgrading |
| [Usage](docs/USAGE.md) | GUI, CLI, TUI, autonomous runs, `@` pinning, language switching |
| [Configuration](docs/CONFIG.md) | Config files, rules, themes, profiles, language, every setting |
| [Architecture](docs/ARCHITECTURE.md) | Engine, server lifecycle manager, `local-moe`, hybrid routing, limits |
| [Skills and tools](docs/SKILLS_AND_TOOLS.md) | 19 bundled skills, custom skills, MCP servers, hooks, prompt overrides |
| [Security](docs/SECURITY.md) | Credential scanning, sandboxing, checkpoints, reporting |
| [Contributing](docs/CONTRIBUTING.md) | Dev setup, tests, module map, docs and locale rules |
| [QUICKSTART.md](QUICKSTART.md) | Step-by-step first runs with expected output |
| [MODEL_GUIDE.md](MODEL_GUIDE.md) | Choosing, pairing and tuning models |
| [TECHNIQUES.md](TECHNIQUES.md) | How Pedacito gets small models to work well |

**Docs ownership.** `docs/` (INSTALLATION, ARCHITECTURE, USAGE, CONFIG, SKILLS_AND_TOOLS, SECURITY and CONTRIBUTING) is the canonical reference for settings, commands and guarantees. The three root guides, [QUICKSTART.md](QUICKSTART.md), [MODEL_GUIDE.md](MODEL_GUIDE.md) and [TECHNIQUES.md](TECHNIQUES.md), remain the canonical task-oriented guides: first runs, model pairing and tuning, and small-model techniques. This README is the index, and [CHANGELOG.md](CHANGELOG.md) is the history. New reference prose goes to `docs/`, new walkthroughs go to the root guides, and each set links to the other instead of copying text.

Run `pdc --help` for every command, or `pdc how "your question"` for built-in help.
Machine clients can drive one warm session over stdio: `pedacito rpc` speaks
newline-delimited JSONL-RPC, and `pedacito acp` speaks JSON-RPC 2.0 Agent
Client Protocol for editors and agent hosts.

## Contributing

Issues and pull requests are welcome; please open an issue before a large PR. See
[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). The project website is
[pedacito.dev](https://www.pedacito.dev), and the maintainer can be reached at
contact@pedacito.dev. Report security issues privately, as described in
[docs/SECURITY.md](docs/SECURITY.md#reporting-a-vulnerability).

## License

MIT, see [LICENSE](LICENSE).
