# Pedacito

**Pedacito** (`pdc`) is a local-first AI coding agent built for the models you can run on the
GPU you already own, from 4B to 27B parameters. Most coding agents assume a frontier model with a
200k-token window. Pedacito cuts your codebase into small, precisely bounded pieces, so a small
model reads and edits only the lines that matter.

It works with **LM Studio, Ollama, Lemonade Server, llama.cpp, vLLM**, and any OpenAI-compatible
server. When you need more power, it can route planning, verification or escalation to hosted models
such as **Claude, ChatGPT, DeepSeek, Qwen, Gemini and OpenRouter**, with a spend cap.

## 30-second install

```bash
pip install pedacito                 # Windows: py -m pip install pedacito
pdc gui ~/code/your-project          # opens the browser control panel
```

Start a model server first (LM Studio, Ollama, Lemonade or llama.cpp). Then, in the GUI, open
**Settings → Model discovery** and click **Scan local servers**. In a terminal, run `pdc models scan`.

Pipx, optional extras and GPU server setup are in [docs/INSTALLATION.md](docs/INSTALLATION.md).

## What's new in v0.3.1

| Feature | What you get | Read more |
|---|---|---|
| **Advanced system profiling & auto-tune** | `pedacito profile` shows a vendor-neutral RAM/VRAM snapshot with a source per number (Linux sysfs, NVIDIA `nvidia-smi`, macOS `vm_stat`, Windows `winapi`), a cached prefill/generation probe and the active profile's tuning. `auto_tune = true` (opt-in) fits context, retrieval budgets and timeout to free memory; missing telemetry makes it abstain, and existing configs do not change. | [System profiling and auto-tune](docs/CONFIG.md#advanced-system-profiling-and-auto-tune) |
| **Context-aware auto-update** | A quiet, opt-out release check: one anonymous PyPI GET at startup (2 s timeout, cached 24 h), a dismissible GUI banner with a streamed upgrade and Restart, `pdc update --run`, and the TUI `/update`. The command matches your install method (pipx, venv, pip --user, or git for editable checkouts). `check_for_updates = false` keeps Pedacito fully offline. | [Upgrading](docs/INSTALLATION.md#upgrading) |

## What's new in v0.3.0

| Feature | What you get | Read more |
|---|---|---|
| **Cross-server VRAM manager** | Treats all your local model servers as one pool. It shows which models are resident, loads and unloads them through each runtime's API, and evicts other servers before a run. | [Server lifecycle manager](docs/ARCHITECTURE.md#server-lifecycle-manager) |
| **Strata local-moe** | Tiered mixture-of-experts servers (Strata, Avalon) get their own execution profile: larger reply budgets, a 10-minute timeout and one stream at a time. | [local-moe tier](docs/ARCHITECTURE.md#execution-profiles-and-the-local-moe-tier) |
| **SSE streaming** | Tokens stream live over server-sent events, with a live reasoning drawer, a per-call heartbeat and a near-instant Stop. | [Live telemetry](docs/USAGE.md#live-telemetry) |
| **Security Guard** | Scans for secrets before any hosted request (`sk-`, `pypi-`, `ghp_`, AWS and Google keys, PEM private keys), keeps sensitive files out of the model's reach, asks before risky actions, and runs commands in an OS sandbox. | [Security guide](docs/SECURITY.md) |
| **Custom themes** | Ten-token color schemes shared by the GUI and the TUI. Six presets, your own saved schemes, and a per-project pin. | [Color schemes](docs/CONFIG.md#color-schemes) |
| **`@` pinning** | Write `@file.py` or `@Class.method` in a prompt to pin that exact code into every phase of the run. | [Pinning context](docs/USAGE.md#pinning-context-with-) |
| **Checkpoint tree** | Every run is checkpointed. Inspect each change as a unified diff, then roll back with `pdc undo` or `pdc checkout`. | [Checkpoints and rollback](docs/SECURITY.md#pillar-4-immutable-audit-trail-and-time-travel) |
| **Hybrid routing** | A strong architect model plans a task and a fast local editor implements it. On one GPU, the architect's server is evicted between the two stages. | [Hybrid routing](docs/ARCHITECTURE.md#hybrid-routing-an-architect-plans-an-editor-implements) |

## Documentation

- **[Installation](docs/INSTALLATION.md)**: pip, pipx, optional extras, model servers, and ROCm or
  CUDA GPU inference servers.
- **[Architecture](docs/ARCHITECTURE.md)**: the local-first engine, the server lifecycle manager,
  the `local-moe` tier, hybrid routing, and the known limits.
- **[Security](docs/SECURITY.md)**: credential scanning, workspace sandboxing, least-privilege
  execution, and checkpoints you can inspect and roll back.
- **[Configuration](docs/CONFIG.md)**: `.pedacito/rules.md`, color scheme tokens, model profiles,
  and the full settings reference.
- **[Usage](docs/USAGE.md)**: the GUI, the CLI, the TUI, autonomous runs and `@` pinning.

The step-by-step guides from before v0.3.0 are still current:
[QUICKSTART.md](QUICKSTART.md) for first runs, [MODEL_GUIDE.md](MODEL_GUIDE.md) for choosing and tuning
models, and [TECHNIQUES.md](TECHNIQUES.md) for how Pedacito gets small models to work well.

**Docs ownership.** `docs/` (INSTALLATION, ARCHITECTURE, USAGE, CONFIG and SECURITY) is the canonical reference for settings, commands and guarantees. The three root guides, [QUICKSTART.md](QUICKSTART.md), [MODEL_GUIDE.md](MODEL_GUIDE.md) and [TECHNIQUES.md](TECHNIQUES.md), remain the canonical task-oriented guides: first runs, model pairing and tuning, and small-model techniques. This README is the index, and [CHANGELOG.md](CHANGELOG.md) is the history. New reference prose goes to `docs/`, new walkthroughs go to the root guides, and each set links to the other instead of copying text.

Run `pdc --help` for every command, or `pdc how "your question"` for built-in help.

## Contributing

Issues and pull requests are welcome: bug reports with a repro, support for other model servers or
languages, and fixes for anything that breaks outside the setup this was built on. Please open an
issue before a large PR. The project website is [pedacito.dev](https://www.pedacito.dev), and the
maintainer can be reached at contact@pedacito.dev.

Pedacito was built and tested by one person on one home GPU box, so edge cases on other hardware
may be rough. Report security issues privately, as described in
[docs/SECURITY.md](docs/SECURITY.md#reporting-a-vulnerability).

## License

MIT, see [LICENSE](LICENSE).
