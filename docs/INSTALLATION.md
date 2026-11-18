# Installation

This guide covers installing Pedacito (`pedacito` or the short alias `pdc`) from PyPI,
choosing optional extras, and connecting a GPU inference server. If you only want the
30-second version, see the [README](../README.md#30-second-install).

Pedacito needs **Python 3.11 or newer**. It does not need Node.js for the harness itself.

## Contents

- [Install from PyPI](#install)
- [Choose a model for your GPU](#which-model-fits-your-gpu)
- [Connect a model server](#model-server-setup)
- [GPU inference servers: ROCm and CUDA](#gpu-inference-servers-rocm-and-cuda)

## Install

Requires Python 3.11+.

Quick install from PyPI (0.3.0):

```bash
pip install pedacito                     # Windows: py -m pip install pedacito
pipx install pedacito                    # isolated CLI install, recommended for a tool
pip install "pedacito[browser,vision]"   # browser verification (Playwright) + screenshot analysis (Pillow)
python -m playwright install chromium    # one-time, after the browser extra
```

With pipx, install the extras with `pipx install "pedacito[browser,vision]"`, then run `pipx run --spec playwright playwright install chromium` to download the browser.

This puts `pedacito` and `pdc` on your `PATH` and installs the full core:

- `tree-sitter` with the `tree-sitter-language-pack` grammars, which power AST
  chunking and the exact syntax check that runs before an edit is written.
- `pytest` and `ruff`, the verification tools. Pedacito runs `ruff check` for
  diagnostics and post-edit checks, and `pytest` for the benchmark holdout
  rubric and the tests an edit can break.
- `requests`, the HTTP client for every model server and hosted API.

You don't need a separate step for any of these. If the command isn't found
(common with per-user installs on Windows), `python -m pedacito`
(`py -m pedacito`) is identical.

`pip install -r requirements.txt` installs the same core set without installing
Pedacito itself. Use it when you only want the dependencies.

Optional features: `pip install -e ".[full]"` adds screenshots (Pillow) and the
headless browser (playwright). Or pick one: `pip install -e ".[vision]"` or
`pip install -e ".[browser]"`. After installing the browser extra, run
`python -m playwright install chromium`.

<details>
<summary>Install from source (development)</summary>

```bash
git clone https://github.com/davepo/pedacito.git
cd pedacito
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

</details>

Uninstall with `pip uninstall pedacito`. Your config and every project's
`.pedacito/` folder are left untouched.

## Which model fits your GPU?

| VRAM | Pedacito tier | Models (Q4 to Q8) | What it is best at |
|---|---|---|---|
| **6–8 GB** | `tiny` / `small` | 4B to 8B | Single-file surgical edits with `pedacito ask`, a fast `scout_profile`, or the local worker paired with a larger `planner` or `escalate` model |
| **12–16 GB** | `medium` | 9B to 14B | Multi-file tasks, test authoring, and `pedacito auto` workflows with a 16k–32k context window |
| **24 GB+** (e.g. RX 7900 XTX, RTX 3090, RTX 4090) | `large` | 24B to 35B | Autonomous multi-file scaffolding, specialist auto-routing across 24B–30B specialists, and 32k context windows |
| **Two GPUs or two servers** | scout + architect | 4B–9B scout, 24B–35B architect | Zero-swap `scout_profile` pairing across separate ports |

Sizes are for the weights alone; add the KV cache and about 1.5 to 2.5 GB of
overhead. The full table, with examples and per-tier thinking advice, is in
[../MODEL_GUIDE.md](../MODEL_GUIDE.md#hardware-tiers).

## Model server setup

Start any server that speaks the OpenAI chat-completions API, then give
Pedacito its base URL and a model name.

| Server | Start it | Base URL |
|---|---|---|
| **LM Studio** | Load a model, then start the server (Developer tab). For another machine, enable "Serve on Local Network". | `http://localhost:1234/v1` |
| **Ollama** | `ollama serve` and `ollama pull <model>`. For another machine, set `OLLAMA_HOST=0.0.0.0` first. | `http://localhost:11434/v1` |
| **Lemonade** | Start the server and load a model. | The URL it prints (ends in `/v1` or `/api/v1`) |
| **Strata / Avalon** | Start the tiered MoE server and load a model, following its own documentation. | Its OpenAI-compatible URL, e.g. `http://127.0.0.1:8000/v1` |
| **Hosted API** | Get an API key. | e.g. `https://api.deepseek.com/v1` |

The default ports above are hints, not rules: any port works, and the scan
identifies a server by its replies, so two servers can coexist on one host
(for example Lemonade on 11434 next to LM Studio on 1234).

Strata and Avalon are tiered mixture-of-experts servers: they keep most of the
GPU's VRAM resident and reason before answering. Give the profile
`provider = "strata"` (or `"avalon"`), or a name containing `strata`/`avalon`,
and Pedacito applies the `local-moe` execution tier: larger reply budgets, a
timeout of at least 600 seconds for weight loading, and exactly one concurrent
stream so a second prefill cannot evict the resident weights. Pedacito also drives their
model load/unload lifecycle between runs. See
[Execution profiles and the local-moe tier](ARCHITECTURE.md#execution-profiles-and-the-local-moe-tier)
for the budgets and a profile example.

For hosted APIs set `base_url`, `api_key`, and `model` (or `api_key_env`, the
name of an environment variable holding the key, so the secret stays out of the
file). Pedacito detects a remote host and switches to `json_mode = "object"`
automatically.

Verify the connection from the machine where Pedacito runs:

```bash
pedacito config --check
```

It tells "server unreachable" apart from "model not found or won't load"
(usually a name mismatch or not enough free VRAM).

`pedacito config --doctor` goes further and calibrates the setup: it runs the
same check, then probes which structured-output modes the server really
supports (`json_schema`, `json_object`, plain prompting), prints a recommended
`json_mode`, reports the server's real context window and model weight size,
calibrates token math against the server's own tokenizer, and reports whether
tree-sitter parsing is active (with its version, or why it is disabled).
Verdicts are remembered per server and model for a week, so a server without
grammar support automatically gets a few-shot JSON example in the prompt on
later runs.

Pedacito also asks the server to constrain every lookup step to the action JSON
(`enforce_json_schema = true`, on by default). Servers that support a JSON schema
enforce it while sampling, so a malformed step cannot be produced. A server that
refuses the request is downgraded automatically, first to `json_object` and then
to plain prompting, and the choice is remembered per model. How each server
behaves is in [../MODEL_GUIDE.md](../MODEL_GUIDE.md#engine-enforced-json-by-server-enforce_json_schema).

Once per model, run `pedacito calibrate`. It runs a scripted
mini-suite, tool choice, JSON adherence, edit format, planted-injection and
honesty checks, long-context retrieval, continuation, a thinking A/B,
capabilities, and saves a **model card**. Each probe runs 3 times (1 with
`--quick`; `--repeats N` overrides) and the card records the mean and spread,
so the choices come from a mean, not one noisy sample; a card from before
this (version 1) is shown with a re-calibrate note and is not applied. Settings you left at their defaults
then follow the card (prompt pack, answer mode, verifier, best-of-N, fold
depth, the thinking switch); anything you set explicitly still wins.
`pedacito config` shows the card.

Calibration is for **local models only**: a hosted provider profile is refused
(CLI exit 2, GUI button disabled) because hosted models use provider-table
defaults automatically, and `models scan` never calibrates hosted imports.
For local models the card also records **architecture defaults** from a
built-in library — recommended temperature, sampler and context window per
family (Qwen coder, Qwen3, DeepSeek coder/R1, Nemotron, Gemma, Mistral,
Granite, Phi-4, Llama-3) — capped by the server's reported window and your
VRAM, with the long-context needle stepping the window down (16384, then
8192) when it misses. `pedacito calibrate --all` walks every local profile in
turn, unloading each model before the next loads; `-y` skips the confirmation
and fresh cards are skipped unless `--recalibrate` is given. In the GUI the
same batch lives under **Project ▾ → Calibrate All Local Models…**, and every
profile row carries a calibration badge: ✓ Calibrated (with the date),
⚠ Calibration stale (older than 30 days or an old card version),
○ Not calibrated, or ☁ Hosted API.

## GPU inference servers: ROCm and CUDA

Pedacito never runs the model itself. It talks to a server that does, so the GPU
backend is a property of the server you install. Pick the server for your card, then
point Pedacito at it.

### Pick a backend

| Your GPU | Good server choices | Backend |
|---|---|---|
| **AMD Radeon** on Linux (check AMD's ROCm release notes for your card's support) | Lemonade Server, llama.cpp built with HIP, Ollama, LM Studio | ROCm (HIP). Vulkan is a fallback that works on more cards. |
| **AMD Radeon** on Windows | Lemonade Server, LM Studio, llama.cpp built with Vulkan | Vulkan or ROCm, depending on the server build |
| **NVIDIA GeForce or RTX** | Ollama, LM Studio, llama.cpp built with CUDA, vLLM | CUDA |
| **No discrete GPU** | Ollama or llama.cpp built for CPU | CPU. Expect small models only. |

If a model fails to load or produces garbage on one backend, try the other one. Pedacito's
own error text suggests this when it sees a runtime mismatch.

### AMD: ROCm

1. Install the ROCm runtime for your distribution, following AMD's own install guide for
   your release. Confirm the card is visible:

   ```bash
   rocm-smi --showmeminfo vram
   ```

2. Start a server. With **llama.cpp**, build with HIP support, then run the server. The
   flags below are a starting point. Check the llama.cpp build documentation for your
   version, and set `AMDGPU_TARGETS` to your card's architecture (for example `gfx1100`
   for RX 7900 XTX).

   ```bash
   cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
   cmake --build build --config Release -j
   ./build/bin/llama-server -m ./models/your-model.gguf \
     --host 127.0.0.1 --port 8080 -ngl 99 -c 16384
   ```

3. Point Pedacito at it. llama.cpp's OpenAI-compatible endpoint is `/v1`:

   ```toml
   [universal]
   vram_gb = 24                              # your card's VRAM, in GB

   [profiles.rocm]
   base_url = "http://127.0.0.1:8080/v1"
   model    = "your-model.gguf"              # the id the server reports
   ```

Pedacito's cross-server memory view reads AMD VRAM from the kernel's ROCm counters
(`mem_info_vram_total` and `mem_info_vram_used`). That is why the memory badge shows real
numbers on ROCm. If the counters are missing, the badge shows system RAM only.

### NVIDIA: CUDA

1. Install the NVIDIA driver and CUDA toolkit for your distribution. Confirm the card is
   visible:

   ```bash
   nvidia-smi
   ```

2. Start a server. With **llama.cpp**, build with CUDA support:

   ```bash
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j
   ./build/bin/llama-server -m ./models/your-model.gguf \
     --host 127.0.0.1 --port 8080 -ngl 99 -c 16384
   ```

   **Ollama** and **LM Studio** detect CUDA automatically. Start them as described in
   [Model server setup](#model-server-setup).

3. Set `vram_gb` by hand. Pedacito does not read NVIDIA VRAM counters, so the memory view
   cannot see your card. The value sizes the context window, so set it to your card's real
   VRAM.

### Quantized KV cache and context

If your runtime quantizes the KV cache, set `k_cache_quant` and `v_cache_quant` in the
profile (for example `"q8_0"`), so Pedacito's VRAM estimate is not too pessimistic. If
the server reports a stale context window, set `context_tokens` explicitly. See
[CONFIG.md](CONFIG.md#profiles-tiers-and-token-budgets) for the full list of profile keys.

### Verify the GPU path end to end

```bash
pedacito config --check     # reachable? model found? generation works?
pedacito config --doctor    # also probes structured output and the real context window
pedacito models scan        # lists the models the server reports and writes a profile for each
```

Then run `pedacito calibrate` once per model. It saves a model card that tunes defaults
to what that model does on your hardware.
