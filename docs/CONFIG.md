# Configuration

Pedacito is configured with TOML files, a small set of project files, and environment
variables. This guide covers where each setting lives, the project rules file, the color
scheme tokens, model profiles, and the full settings reference.

## Where settings live

| Location | Scope | What it holds |
|---|---|---|
| `~/.config/pedacito/config.toml` (Linux and macOS), or `%APPDATA%\pedacito\config.toml` (Windows) | You, for every project | Servers, profiles, API keys, defaults. `XDG_CONFIG_HOME` replaces `~/.config`. |
| `<project>/.pedacito.toml` | One project | Per-project settings. Execution settings are ignored until you run `pedacito trust`. |
| `<project>/.pedacito/rules.md` | One project | Hard requirements shown in every prompt. See [Project rules](#project-rules-pedacitorulesmd). |
| `<project>/.pedacito/config.json` | One project | Optional theme pin, such as `{"theme": "nord"}`. |
| `themes.json`, beside the user config | You | Custom color schemes and the selected theme. See [Color schemes](#color-schemes). |
| `PEDACITO.md`, `AGENTS.md`, `CLAUDE.md`, `.pedacito/instructions.md` | One project | Conventions read into every prompt. See [Project instructions](#project-instructions). |

Settings resolve in layers. Each layer overrides the one before it: built-in defaults, then the
user config, then the project's `.pedacito.toml`, then `--config FILE`, then `--profile`, then
command-line flags. Environment variables `PEDACITO_<KEY>` (for example `PEDACITO_MODEL`) and
`PEDACITO_PROFILE` are also honoured. Unknown keys are rejected with a suggestion.

To see the value of every setting and where it came from, run `pedacito config`. To print the
annotated template, run `pedacito config --reference`.

## Configure

```bash
pedacito config --setup     # finds local servers, writes a profile per model
pedacito models scan        # re-run later to add models or servers
pedacito profiles           # see scores and role recommendations
pedacito config --init      # or write a commented starter file
pedacito config             # show every resolved setting and where it came from
pedacito config --reference # print the full annotated template
pedacito config --upgrade   # add newly documented options, commented out
pedacito config --upgrade --dry-run  # preview what would be added
```

`config --upgrade` appends options that a newer Pedacito documents but your
file lacks, commented out, keeping every existing setting and comment;
`--dry-run` lists them without writing.

`config --setup` (and `models scan`) finds local model servers, LM Studio,
Ollama, Lemonade, llama.cpp, **on any port**, identifies each by its replies
rather than its port (so two servers can share one machine), lists its models,
and writes one identity-only profile per model, then sets a default. Pass
`--url` to point at a server on another machine or a hosted API; `models scan
--match PATTERN --calibrate quick|full|none` re-runs it. Prefer to edit by
hand? The file is `~/.config/pedacito/config.toml` (Linux/macOS) or
`%APPDATA%\pedacito\config.toml` (Windows); a set `XDG_CONFIG_HOME` replaces
`~/.config`. The minimum:

```toml
[universal]
vram_gb = 8                             # GPU VRAM on the model server, in GB

[profiles.default]
base_url = "http://localhost:1234/v1"
model    = "your-model-id"              # exact id/name the server reports
```

`vram_gb` is required for local servers (with the model's file size it sizes
the context window) and not needed for hosted APIs.

Settings resolve in layers, each overriding the previous: built-in defaults →
user config → the project's `.pedacito.toml` → `--config FILE` → `--profile` →
command-line flags. Environment variables `PEDACITO_<KEY>` (for example
`PEDACITO_MODEL`) and `PEDACITO_PROFILE` are also honored. Unknown keys are
rejected with a suggestion. The legacy `[lmstudio]` section is still accepted as
an alias for `[server]`.

## Project rules: `.pedacito/rules.md`

`.pedacito/rules.md` is a short, persistent list of requirements for one project. Pedacito
inserts it into every task as **hard requirements**, so the model sees them whether or not it
reads the affected files. Use it for things that must always hold, such as "all HTTP calls go
through `client.py`" or "never add a dependency without asking".

```markdown
- Use `pytest -q` for tests. Do not change `tests/conftest.py` fixtures.
- Public functions in `pedacito/api/` keep their signatures.
- Do not write to `legacy/`.
```

### How it is used

- The text is added to the prompt as `PROJECT RULES (from .pedacito/rules.md; these are hard
  requirements for every task)`.
- The text is capped at **6,000 characters**. A longer file is cut with a note that points to
  the file.
- It is cached by file modification time, so editing it takes effect on the next task.

### Edit it

- **GUI:** open the project rules editor, which saves the file.
- **TUI:** `/rules` shows the current rules. `/rules set TEXT` replaces them, and `/rules clear`
  empties the file.
- **Any editor:** the file is plain Markdown.

### Share it with your team

Pedacito adds `.pedacito/` to the project's `.gitignore` on first run, so `rules.md` is ignored
by default. To commit it, replace the `.pedacito/` line with these two lines:

```gitignore
.pedacito/*
!.pedacito/rules.md
```

### Rules compared with other instruction files

| File | Shown when | Use it for |
|---|---|---|
| `.pedacito/rules.md` | Every task, as hard requirements | Invariants that must always hold |
| `PEDACITO.md`, `AGENTS.md`, `CLAUDE.md` | Every task, capped at 2,000 characters | Build and test commands, style, directories to avoid |
| `.pedacito/rules/*.md` | Only when the model reads a matching file | Rules for one area, with `paths:` frontmatter |
| `.pedacito/memory.md` | Every task, as hints rather than truth | Facts learned from earlier runs |

## Color schemes

The GUI and the TUI share one theme engine (`pedacito/themes.py`). A theme is exactly **ten
hex color tokens**. Each token is `#rgb` or `#rrggbb`. Every token is required, and an unknown
token is rejected, so a typo cannot silently drop a colour.

| Token | Paints |
|---|---|
| `bg` | The primary background |
| `surface` | Panels and cards |
| `surface_alt` | Inputs, hover states and elevated surfaces |
| `text` | Primary text |
| `muted` | Secondary text |
| `accent` | Brand and primary action colour |
| `border` | Dividers and borders |
| `success` | Passing status |
| `warning` | Caution status |
| `danger` | Errors and failures |

The engine ships six built-in presets: `pedacito-dark` (the default), `catppuccin-mocha`,
`tokyo-night`, `gruvbox-dark`, `nord` and `solarized-dark`. In the TUI, `/theme` lists them and
`/theme NAME` switches and saves the choice. In the GUI, the palette button in the top bar opens
the Theme Manager, which lists the same presets and your custom themes.

The `gui_theme` key in `config.toml` comes from the earlier GUI picker, which had nine legacy palette
names. The GUI still reads that key for compatibility, but the colours on screen come from the
selected theme in the Theme Manager. Use the Theme Manager, not `gui_theme`, to change the scheme.

### Create a custom scheme

Custom schemes live in `themes.json`, beside your user config. You can create one from the GUI's
Theme Manager, or write the file by hand:

```json
{
  "active_theme": "midnight-ink",
  "custom_themes": {
    "midnight-ink": {
      "bg": "#0f1020",
      "surface": "#171a2e",
      "surface_alt": "#22263f",
      "text": "#e6e8f5",
      "muted": "#9aa0c3",
      "accent": "#8ab4ff",
      "border": "#2c3152",
      "success": "#8fd3a0",
      "warning": "#f0c674",
      "danger": "#ff7b8a"
    }
  }
}
```

A custom theme cannot reuse a built-in name. A hand-edited entry that fails validation is skipped,
so one bad entry does not stop the app from starting.

### Pin a theme to one project

To use a different scheme for one repository, create `<project>/.pedacito/config.json`:

```json
{"theme": "nord"}
```

A valid project pin overrides the globally selected theme in that project only. The global
selection is unchanged for every other project.

## Hosted prompt caching

Pedacito keeps the prompt prefix stable (the system prompt and first user turn
are byte-identical within a gather), so a hosted provider that caches on its own
reuses it across steps automatically. A provider table matches on the host of
`base_url` and decides the cache mode, plus a few compatibility quirks:

| Host | Caching | What Pedacito sets |
|---|---|---|
| `api.deepseek.com` | Automatic (implicit) | nothing extra; reads hit/miss tokens |
| `api.openai.com` | Automatic for prompts ≥ 1024 tokens | `prompt_cache_key` routes to a warm cache |
| Qwen / DashScope | Implicit automatic; explicit opt-in | `cache_control` markers in `explicit` mode |
| `api.anthropic.com` (OpenAI-compatible) | Not supported on this surface | `json_mode = "schema"`, `tool_mode = "grammar"` (sampling keys are omitted for every hosted API; see [Hosted APIs keep their own sampling](#hosted-apis-keep-their-own-sampling)) |
| Moonshot, Gemini, xAI, Groq | Automatic / implicit | nothing extra; hits are counted when returned |

Set `prompt_cache` to override the table per profile:

```toml
[profiles.qwen]
prompt_cache = "explicit"   # auto (the table decides) | implicit | explicit | off
prompt_cache_ttl = "5m"     # explicit-mode breakpoint TTL: "5m" or "1h"
prompt_cache_key = "auto"   # a stable hash of project + profile + model, sent
                            # only where the provider accepts it; "" = nothing
```

Cache writes are billed separately from reads: set
`price_cache_write_per_mtok` to count them in cost. `pedacito config --doctor`
probes the active provider and reports the cache as *observed*, *not observed*,
or *explicit-only* (DeepSeek-style caches persist asynchronously, so "not
observed" is not "unsupported").

## Hosted APIs keep their own sampling

A hosted provider tunes sampling per model itself, so Pedacito stops dictating
it: any profile whose `base_url` is not a local server (loopback or LAN) gets
**no `temperature`, `top_p`, `top_k`, `min_p` or `repeat_penalty`** by default —
including keys from the `sampler` table. The provider's tuned defaults apply,
which is what its own chat UI uses. Non-sampling keys such as `seed` still pass
through, and `presence_penalty` is never stripped.

Two opt-ins, both sending **only** `temperature` (the other four keys stay
omitted either way):

```toml
send_temperature_to_hosted = true   # send the configured temperature anyway
hosted_temperature = 1.0            # or pin one value for hosted endpoints,
                                    # globally or in a profile; it wins over
                                    # phase overrides (e.g. 1.0 for Claude)
```

Local servers keep every configured value. The answer phase defaults to
`temperature_answer = 0.3` — cool enough to copy SEARCH text exactly, warm
enough to escape the degenerate verbatim loops a greedy `0.0` froze weak local
models into — and creative/design tasks run at ≥ 0.6 (see
[Creative / design mode](USAGE.md#creative--design-mode)). When a local reply still
repeats itself, the retry gets one gentle `presence_penalty = 0.15` nudge
instead of a cold temperature or a heavy repeat penalty.

## Local inference acceleration

Hosted caching reuses a provider-side prefix; a self-hosted server can do the
same thing locally, and Pedacito asks it to with two llama.cpp server keys
(understood by LM Studio, llama.cpp server, Lemonade, vLLM, and Ollama's
OpenAI surface):

```toml
[profiles.local]
enable_prompt_cache = true      # send cache_prompt (default: true)
enable_prompt_lookup = false    # send prompt_lookup_num_tokens (default)
prompt_lookup_num_tokens = 10   # draft span length when speculation is on
```

`cache_prompt` keeps the KV prefix warm between gather steps, so the repeated
conversation prefix is not re-prefilled on every lookup call.
`prompt_lookup_num_tokens` turns on n-gram prompt-lookup decoding: the draft
comes from the prompt itself, which pays off on copy-heavy output (SEARCH
blocks quote source lines that are sitting right there in the context).

Three safety properties hold: the keys are attached only to endpoints you
self-host (loopback, a LAN address in RFC1918 / CGNAT / IPv6 unique-local
range, a `.local` name, or a provider entry naming a local runtime), so a
hosted API never sees them; a 400 or 422 that lands while either key is
attached is treated as a refusal, the request is re-sent once without them,
and the denial is remembered per server+model pair so later calls start
stripped (one wasted request, ever, on servers that predate the keys); and a
key you spell yourself in the profile's `[sampler]` table is the user's, not
ours, and is left alone.

## Named MCP servers

Beyond the legacy `mcp_servers = ["command", ...]` list behind the `mcp`
action, Pedacito can reach **named** servers declared as tables, and hands
them to the model through a single `mcp_call` action:

```toml
enable_mcp = true                       # the whole bridge; default true

[mcp_servers.docs]
command = "python"                      # stdio transport: argv, no shell
args = ["-m", "docs_server"]
env = "DOCS_DB=~/docs.db"               # or a JSON object { "K": "V" }
trusted = false                         # true: calls run without asking
enabled = true                          # false: park this server

[mcp_servers.jira]
url = "http://127.0.0.1:8000/mcp"       # Streamable HTTP instead of stdio
```

Everything is **pruned per task**. Once per session the bridge lists each
enabled server's tools (a 5-second budget per server; a server that is down
simply contributes nothing), scores tool names and descriptions against the
task text, and injects at most four matches as one-line signatures into the
gather prompt. `mcp_call` is offered in the action schema only when at least
one tool matches, and the model may call only what it was shown: irrelevant
tools on a busy server are invisible, and the default schema for people who
configure nothing stays byte-identical.

Everything is **permission-gated per call**. A server marked `trusted = true`
runs silently; on any other server each call asks the permission broker with
the server, tool, and exact arguments in the dialog (**MCP Tool** category),
and a decline returns fixed text telling the model to proceed with local
repository tools only. Returned text is capped at about `max_read_tokens`
tokens with a truncation note. Project `.pedacito.toml` files may not define
servers at all (the same trust gate as `hooks` and shell settings): named
servers come from your user config, or from a project you explicitly trusted.

```bash
pedacito mcp list        # what the bridge sees for this project, pruned view
```

In the GUI, **⚙ Settings → MCP Servers** adds, edits, tests, enables, trusts
and deletes named servers without hand-editing TOML; they are written to your
user config (a project file needs trust first). The same tool listing is
served at `GET /api/mcp/tools`. See
[Techniques §30](../TECHNIQUES.md#30-pruned-permission-gated-mcp-bridges-named-servers--mcp_call)
for why the pruning is shaped this way.

## Hooks

Hooks run your own program at well-defined moments:

```toml
hooks = { post_edit = ["ruff format {file}"], pre_edit = ["./scripts/guard.sh"] }
```

Events: `session_start`, `pre_task`/`post_task`, `pre_tool`/`post_tool`,
`pre_edit`/`post_edit`, `pre_run`/`post_run`, `pre_compact`/`post_compact`,
`notification`.

The protocol is plain JSON, no plugin ABI: the hook receives one JSON object
on stdin (`{"event", "cwd", ...event fields}`) and may answer on stdout with
`{"decision": "allow" | "deny", "message": "...", "context": "..."}`. A
`pre_*` hook denies with a `deny` decision, or with exit status 2 and the
reason on stderr. Text a hook prints (the `context` field, or plain stdout
after a `post_*` event) is fed back to the model.

Hooks run as an argv — never through a shell — in the project root, with the
scrubbed environment plus `PEDACITO_EVENT`, and a timeout (`hook_timeout`,
10 seconds by default). `{file}` / `{files}` placeholders expand from the
event. A hook that fails or times out is a warning, not an error. Hooks from
a project's own `.pedacito.toml` only run after `pedacito trust`.

## Memory, skills, and instructions

### Memory, rules, and skills

Pedacito can remember what earlier runs learned, as plain markdown you can
edit or delete:

- **Memory** (`.pedacito/memory.md`, plus a user-level `memory.md` next to your
  config) holds short facts under Commands / Conventions / Landmines / Notes.
  The model adds one with `remember`; you can too:
  `pedacito memory PROJ --add "command: run tests with pytest -x"`.
  `pedacito memory PROJ` lists what is remembered, `--forget TEXT` removes
  matching lines, and `--global` switches either to the user-level file.
  Lines are de-duplicated, capped per category,
  secrets are masked, and anything that reads like an instruction to an
  assistant is refused. It is shown to the model as *hints, not truth*.
- **Rules** (`.pedacito/rules/*.md` with a `paths: src/api/*.py` frontmatter
  line, and nested `AGENTS.md`/`CLAUDE.md` files) are shown only when the model
  reads a matching file. Pedacito also auto-manages `auto_*.md` rule files here
  (repair lessons promoted from `micro_facts.json`; see
  [Repair lessons](ARCHITECTURE.md#repair-lessons-and-repo-gotchas)) -- anything without the
  `auto_generated: true` frontmatter flag is never written or deleted by it.
- **Skills** are named procedures (`.pedacito/skills/NAME/SKILL.md` or
  `.pedacito/skills/NAME.md`, and the same two layouts under `skills/` next to
  your config). The model sees a one-line catalog and loads a skill's text only
  when it needs it; additionally, the top-2 trigger matches for a task are
  injected up front, capped at 1,200 chars of checklist text. Eighteen starters
  ship built in: `add-a-test`, `fix-a-failing-test`, `refactor-safely`,
  `review-my-diff`, `find-dead-code`, `surgical-symbol-edit`,
  `frontend-dom-audio-debug`, `safe-multi-file-rename`, `fastapi-rest-endpoint`,
  `canvas-svg-2d-3d`, `sqlite-sql-migrations`, `async-concurrency-safety`,
  `security-ssrf-xss-hardening`, `i18n-localization-catalog`,
  `parser-lexer-state-machine`, `performance-memory-profiling`,
  `modern-web-design-ui` and `desktop-gui-and-games`.
  `pedacito skills list` shows them all with a scope badge (bundled / user /
  project). Skills are prompt text, nothing in a skill folder is ever executed.
- **Custom skills** are yours to write: `pedacito skills add NAME
  --description "..." --triggers "k1,k2" --body "1. ..."` saves one to the
  project (or `--global` for the user library), `pedacito skills draft "Skill
  for Alpine.js dark-mode components"` lets the active model write the whole
  skill from one sentence, `pedacito skills show NAME` prints its triggers and
  checklist, and `pedacito skills rm NAME` deletes a custom skill (built-ins
  are immutable; a project skill may shadow a built-in of the same name).
  The GUI twin is **Project ▾ → Skills library & creator…** (also linked from
  ⚙ Settings): browse every scope, preview which skills a task would inject,
  and create skills with the **✦ Auto-Draft from Description** button, saving
  to the project or the global library.

### Project instructions

Pedacito reads one conventions file into every prompt: the first found of
`PEDACITO.md`, `AGENTS.md`, `CLAUDE.md`, `.pedacito/instructions.md`, or
`.pedacito/PEDACITO.md` (the older spelling `PEDACEITO.md` is also accepted).
Put build/test commands, style rules, and off-limits directories there. It is
capped by `max_instruction_chars` (2000) and disabled with
`use_project_instructions = false`.

## Profiles, tiers, and token budgets

`pedacito profiles` lists every profile with its measured harness-fit (mean ±
spread), eval pass rate when a report exists, generation speed, the context
that fits its VRAM, and card status; `--recommend` adds per-role suggestions
and `--json` emits the same as JSON. `pedacito models scan` writes an
identity-only profile per model a server lists.

Define a profile per model instead of editing `model =` each time:

```toml
[profiles.small]
model = "qwen2.5-coder:7b"

[profiles.big]
model = "some-27b-model"
tier  = "large"
```

```bash
pedacito ask PROJ -t "..." --profile big
```

Choose a profile with `default_profile =`, `PEDACITO_PROFILE`, `--profile`, or
`/profile` in `chat` (increasing precedence). A profile can override any
setting, not just the model.

`tier = "tiny" | "small" | "medium" | "large" | "hosted"` fills in all token
budgets (`max_tokens_*`, `flat_map_token_budget`, `max_gathered_chars`, ...)
from a preset; `tiny` suits ≤4B models with a ~4k window. When unset, the tier
is guessed from the model name (≤4B → tiny, ≤10B → small, ≤20B → medium, else
large; a hosted keyword wins). You rarely need to tune these: Pedacito detects
the server's real context window and the model's file size and fits every
budget to your VRAM (`auto_fit`, on by default). Explicit values always win,
and `pedacito config` marks every non-default value.

If your runtime uses a quantized KV cache, set `k_cache_quant` and
`v_cache_quant` (e.g. `"q8_0"`, `"q5_1"`) so the VRAM estimate isn't too
pessimistic. If the server reports a stale default window instead of your
configured one, set `context_tokens` explicitly.

Switching models makes the server load the new one; Pedacito waits up to
`load_wait_seconds` (240).

Reasoning models are supported across families: `<think>`/`<thinking>`/
`<reasoning>`, Magistral's `[THINK]`, Seed-OSS's `<seed:think>`, GPT-OSS
harmony `analysis` channels, and Gemma 4's `<|channel>thought` block are
stripped before parsing, while separate `reasoning_content`/`reasoning`/
`thinking` fields are read directly. The
`disable_thinking` switch is family-aware: local templates get the common
`enable_thinking=false` and `thinking=false` kwargs (unknown template variables
are ignored), plus `reasoning_effort=none` for Magistral and
`thinking_budget=0` for Seed-OSS. Recognised hosted providers get their own
spelling (Kimi's `thinking={"type":"disabled"}`, DashScope's
`enable_thinking`, DeepSeek's template kwarg, `reasoning_effort` on
OpenAI/Cerebras/GPT-OSS); an unknown hosted model gets no extra field, so a
strict API cannot reject it. `thinking_suffix` adds a prompt soft switch, and
auto-selects `/no_think` for Qwen3/SmolLM3. The Qwen, Gemma, Nanbeige, and
LiquidAI families all use `enable_thinking` and are covered automatically.
Some families have no off switch (GPT-OSS, Kimi K2.7/K3, LFM2.5-24B-A2B); their
reasoning blocks are still stripped, but the tokens are unavoidable. For an
unrecognised family, add the switch your model expects under the profile's
`sampler` table (user keys survive and merge with the automatic ones).
Mistral-style role alternation and Gemma's missing system role are handled
automatically.

## Advanced system profiling and auto-tune

`pedacito profile` shows the hardware snapshot, the last cached speed measurement and the
active profile's tuning. `pedacito models` (with no action) prints the same numbers beside
the active model; `pedacito models scan` is unchanged.

```bash
pedacito profile             # snapshot + last probe + active tuning (same as --show)
pedacito profile --show
pedacito profile --run       # force one prefill/generation probe now
pedacito profile --reset     # drop cached measurements, turn off auto_tune = true
pedacito profile --show --json
```

The TUI exposes the same three actions as `/profile show`, `/profile run` and
`/profile reset`. Plain `/profile NAME` still switches profiles. Both `--reset` and
`/profile reset` drop the cache and turn off an explicit `auto_tune = true` in the same
selected config file (the `--config` file when one was passed, otherwise the user config).

### What the telemetry reads

Every number carries its source. A reader that is absent on this platform contributes
nothing; it is never treated as zero free memory. Each reader is time-boxed to two seconds
and every failure is silent (log-level only). The snapshot is memoized for two seconds, so
a burst of callers (config load, role derivation, GUI payload) shares one reader pass.

| Platform | System RAM | GPU VRAM | Source tag |
|---|---|---|---|
| Linux with AMD or Intel (amdgpu, Intel xe, any driver exposing `mem_info_vram_*`) | `/proc/meminfo` | `/sys/class/drm/card*/device/mem_info_vram_total` and `_used` | `drm-sysfs` |
| Linux with NVIDIA | `/proc/meminfo` | `nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits` (argv, never a shell string) | `nvidia-smi` |
| macOS | `sysctl -n hw.memsize` total, `vm_stat` free/inactive/speculative pages | none: Apple Silicon unified memory counts as RAM, never as a separate VRAM claim | `vm_stat` |
| Windows | `kernel32.GlobalMemoryStatusEx` through stdlib `ctypes` (same API on x86-64 and ARM64) | NVIDIA through `nvidia-smi`; AMD/Intel have no vendor-neutral reader, so auto-tune abstains there | `winapi` |

The GUI hardware badge shows **totals only**. There is no per-process GPU attribution,
and its labels say so. `servers.get_hardware_memory()` keeps its original shape and values;
the profiler snapshot is a superset, so older GUI/TUI consumers keep working.

### Footprint and confidence

`estimate_model_footprint` estimates two things:

- **Resident weights** = parameters × bytes per parameter. The accepted quantizations are
  f16 = 2, q8 = 1, q6 = 0.75, q5 = 0.625, q4 = 0.5 and q2 = 0.25 bytes per parameter. An
  unknown quantization uses the conservative 0.6.
- **KV cache** uses the same size model as `auto_fit`: an explicit
  `kv_cache_bytes_per_token` is used verbatim; otherwise a known model file
  size gives a 32k-token cache roughly the size of the weights, scaled by
  `k_cache_quant`/`v_cache_quant` (unset = f16). Without a file size the
  architecture formula is 2 (K and V) × layers × KV heads × head dim, using
  the weight quantization when no cache quant is set. The profiler reports
  GB per 1,000 tokens.

A dense model keeps all weights VRAM-resident. A mixture-of-experts model reports both
numbers: all experts in RAM (`resident_gb`) and only the active experts VRAM-resident
(`active_gb`/`vram_resident_gb`), so a tiered Strata/Avalon profile can be reasoned about.

Architecture metadata is read in this order: explicit config overrides
(`model_params_b`, `model_active_params_b`, `model_layers`, `model_kv_heads`,
`model_head_dim`, `model_quant`, `model_arch`), llama.cpp `/props` and `/models`, Ollama
`/api/show`, then model-name heuristics. `confidence` is highest when every field came
from explicit config or server metadata (0.95 / 0.85), and drops for each inferred field
(an unknown quantization, guessed layer count, guessed KV head count or guessed head
dim). A confidence of `0.00` means the estimate is unusable. The report never presents a
guess as measured; `estimated_fields` lists exactly what was inferred.

### `auto_tune` rules

`auto_tune` is **false by default**. Existing users see no change — no hardware reader
or budget adjustment runs until they opt in. The automatic post-load speed probe is a
separate behavior, controlled by `profile_probe_on_load` (default true); set it false to
keep even that from starting after a load.

When enabled for a local profile:

- **context_tokens** is capped by `(free memory − weights − safety reserve) / KV GB per
  1,000 tokens`. A 4,096-token safety floor (or the user's smaller value) always remains.
  When the pool is tight, the window shrinks to that floor. When the pool is roomy, growth
  is *proposed* up to the server-reported window and is never forced. When the model file
  size is known, the proposal is also capped by the same VRAM window `auto_fit` derives,
  so `k_cache_quant` / `v_cache_quant` can never make the tuner suggest a window the
  server cannot hold.
- **Confidence scales the action.** Below a footprint confidence of 0.50 (for example,
  all KV geometry guessed from the model name), the tuner still computes the would-be
  context and retrieval-budget numbers but reports them only as notes: no value is
  written. At 0.50 and above the same calculations apply as before. When a KV-geometry
  field was estimated, the notes say so and point at `model_layers` / `model_kv_heads` /
  `model_head_dim` / `model_quant` or `pdc profile --run`.
- **Retrieval budgets** (`max_gathered_chars`, `flat_map_token_budget`) scale with the
  tuned context; memory-heavy local-MoE profiles get a smaller window than dense profiles
  at the same context.
- **timeout** stays at 600 s for tiered/local-MoE. For local-dense it becomes
  `max(60, context / measured prefill tok/s × 1.5)` when a probe exists. A hosted profile
  gets the 30 s cloud default; its context, retrieval and hardware-derived budgets are
  never touched.
- **max_concurrency** stays 1 for tiered/local-MoE and is never raised. Local-dense and
  hosted profiles keep their value.
- **User-pinned values win.** Any field whose provenance is an explicit config file,
  profile, environment variable or CLI flag is left alone. A model with every tunable
  field pinned is untouched.
- **Multiple cards are not summed for headroom.** VRAM totals remain a summed system
  inventory for the GUI badge and the report header, but when the cards differ in size
  (for example a 0.5 GB iGPU beside a 24 GB discrete card) tuning uses the largest
  card's total and free numbers and names the ignored cards in a note
  (`tuned against card1 (24.0 GB); card0 ignored`). `pdc profile` marks the per-card
  tuning basis. Equally sized cards and single-card systems keep the summed path; a
  card with unknown free memory is never treated as zero.
- **No telemetry means abstain.** With no VRAM reader, auto-tune uses RAM headroom only on
  a unified-memory platform (macOS). Everywhere else it leaves the config untouched; a
  missing reader is never interpreted as zero free memory.

**Who owns a value.** A value you set (config file, profile, environment or CLI) is
explicit to both `auto_fit` and the tuner, and neither overwrites it. A value written by
auto-tune is tagged `auto-tune (...)`; `auto_fit` treats that tag as explicit, so its VRAM
window will not replace the tuned number on a later load, while the tuner treats its own
tag as soft and may refine a previous tuning pass when the hardware snapshot or metadata
changes. That one-way pinning is deliberate: `auto_fit` honors the tuned window, and the
tuner remains able to correct its earlier output. To pin a number against both, set it
explicitly in the config or profile.

The applied per-field origins use the same `sources` provenance convention as the
`local-moe` defaults, for example
`auto-tune (VRAM 19.8 GB free via nvidia-smi; weights 7.0 GB; KV 0.0458 GB/1k tok; confidence 0.85; floor 4096)`.

### Speed probe

`pedacito profile --run` (and, once per server+model, a successful explicit model load)
sends one tiny completion (`max_tokens = 16`, "Reply with just: OK") plus one prefill
measurement: about 2,000 filler tokens when the model's window allows, otherwise 512.
Set `profile_probe_on_load = false` to keep the automatic post-load probe from starting
at all; explicit probes still work. Prefill and generation tok/s come from the server's
SSE timings. The probe is best-effort
and silent on failure; a cached result lives in `profiler.json` beside the user config for
one week and is keyed by server+model, so changing either one invalidates it. Probes never
run for hosted endpoints: both explicit paths (`pdc profile --run` and `/profile run`)
refuse with a "hosted model — refusing to probe" message before sending a request, and
the automatic post-load probe is skipped for a hosted URL.

## Configuration reference

`pedacito config` lists every setting with its value and origin; `pedacito
config --reference` prints the annotated template. Key settings:

| Setting | Default | Meaning |
|---|---|---|
| `base_url` | `http://localhost:1234/v1` | Server address. |
| `model` | `google/gemma-3-27b` | Exact id/name the server reports. |
| `api_key` / `api_key_env` | `"lm-studio"` / `""` | For hosted APIs and secured servers. `api_key_env` stores the *name* of an environment variable holding the key, so the secret never lands in the config file. |
| `vram_gb` | *(required for local)* | GPU VRAM; set in `[universal]`, overridable per profile. |
| `context_tokens` | `8192` | Server-side context window. With `auto_detect_context` on (the default) the server's reported window replaces it; set it explicitly when detection is wrong. |
| `auto_tune` | false | Opt in to hardware/probe-driven tuning of context, retrieval budgets and timeout. Off by default: nothing changes until it is true. See [Advanced system profiling and auto-tune](#advanced-system-profiling-and-auto-tune). |
| `profile_probe_on_load` | true | Schedule the background speed probe once per server+model after a successful explicit load. `false` = no probe thread starts after a load; `pdc profile --run` still works. |
| `model_params_b` / `model_active_params_b` | 0 / 0 | Explicit parameter counts (billions) for the footprint estimate; active is the MoE experts used per token. 0 = discover, then fall back to name heuristics with lower confidence. |
| `model_layers` / `model_kv_heads` / `model_head_dim` | 0 / 0 / 0 | Explicit KV-cache geometry. Missing fields lower the estimate's `confidence` instead of being reported as measured. |
| `model_quant` / `model_arch` | `""` / `""` | Explicit weight quantization (`q4`, `Q4_K_M`, `q8`, `f16`, ...) and `dense`/`moe`. Empty = infer from server metadata, then the model id. |
| `small_context_tokens` / `max_whole_file_lines` | 16384 / 120 | A window at or below `small_context_tokens` counts as "small": the answer brief then demands chunked/anchored edits, and a whole-file rewrite of a file bigger than `max_whole_file_lines` is refused before it lands (unless the task explicitly asks to rewrite that file, is a creative/design task, or targets `.html`/`.htm`/`.css`/`.md` — see [Creative / design mode](USAGE.md#creative--design-mode)). `small_context_tokens = 0` turns the discipline off. |
| `k_cache_quant` / `v_cache_quant` | `""` | KV-cache quantization the runtime uses. |
| `json_mode` | `"schema"` | `"schema"` for local servers, `"object"` for hosted (auto-switched). |
| `enforce_json_schema` | true | Ask the engine to constrain gather steps with `response_format` json_schema (strict); servers that refuse are downgraded to json_object, then plain prompting, per model. |
| `enable_run_test_in_gather` | true | Let the gather loop run one scoped test check (pytest, Node/Vitest, Go, or Cargo, chosen from the target) to see live tracebacks. `false` makes each run ask the permission broker. |
| `frontend_contracts` | true | Before tests run, check local script/stylesheet links, literal DOM id lookups and changed JavaScript syntax. |
| `auto_heal_imports` | true | Before a static-gate finding reaches the model, insert the one import that provably resolves it (stdlib name, or a symbol exactly one local module exports), verified by the AST checker; ambiguous names are never guessed. |
| `collapse_repair_turns` | true | When a repair succeeds, drop the failed drafts from the conversation tail and keep a compact receipt, so the server's prompt-prefix cache is reused instead of re-prefilling the broken trajectory. |
| `micro_facts` / `max_micro_fact_tokens` | true / 150 | Record confirmed commands and repair lessons in `.pedacito/micro_facts.json` and inject them as hints; the token cap applies to the injected block. A lesson confirmed twice for one directory is promoted into an auto-managed path-scoped rule in `.pedacito/rules/auto_*.md` instead of sitting in every prompt. |
| `tier` | `""` | Budget preset: `tiny`/`small`/`medium`/`large`/`hosted`; guessed from the model name when unset. |
| `max_steps` | 0 | Lookup limit per task; 0 = none. |
| `prefetch_targets` | 0 | Read up to this many task-named files/symbols through the normal tool path before the first lookup step (and once more after gather when a named target is still unread). Prefetch reads don't count toward `max_steps`. |
| `temperature` | 0.2 | Higher values make tool selection worse. Local servers only: hosted APIs get no sampling keys by default (see [Hosted APIs keep their own sampling](#hosted-apis-keep-their-own-sampling)). |
| `temperature_answer` / `temperature_repair` | 0.3 / unset | Sampling temperature for the answer and for the repair re-asks, which copy SEARCH text character for character. The answer defaults to `0.3` — cool enough to copy exactly, warm enough to escape verbatim repetition loops (a greedy `0` froze weak local models); creative/design tasks run at ≥ 0.6. The repair re-asks are unset (they use `temperature`). Gather keeps `temperature`; `best_of` candidates run at the answer temperature + 0.4. |
| `send_temperature_to_hosted` | false | Send the configured `temperature` to hosted APIs anyway. The other sampling keys (`top_p`, `top_k`, `min_p`, `repeat_penalty`) stay omitted either way. |
| `hosted_temperature` | unset | Pin one temperature for hosted endpoints, globally or per profile (e.g. `1.0` for Claude); it wins over phase overrides and is the only sampling key sent. Unset: the provider's own default applies. Local servers ignore it. |
| `backup` | true | Snapshot originals before writing. |
| `log_sessions` | true | Write a session record per task. |
| `review` | false | Always use the review page. |
| `fuzzy_match` | false | Anchor near-miss search blocks approximately. |
| `edit_style` | unset | `quote` (default brief) or `anchor` (prefer `SYMBOL` + full definition, SEARCH only for small edits); unset lets the model card decide. |
| `creative_mode` | false | Creative/design (Builder) mode for EVERY task: designer persona, answer temperature ≥ 0.6 on local servers, an ≥ 8192-token reply budget, whole-file HTML/CSS rewrites. `--creative`, TUI `/creative on`, or the GUI's ✨ Creative toggle set it per run. See [Creative / design mode](USAGE.md#creative--design-mode). |
| `creative_auto` | true | Auto-detect design tasks (website/portfolio/redesign prompts, `.html`/`.css`/`.svg`/`.jsx`/`.tsx`/`.vue`/`.svelte` targets) when `creative_mode` is off. Both false = surgical edits everywhere (`/creative off`). |
| `post_edit_check` | true | After applied edits, feed back only NEW linter/syntax findings on the changed files. |
| `auto_allowlist` | false | Restrict `auto` task writes to files the title names or the lookup read (new files always allowed). |
| `allow_write` | false | Whole-file write/append/move/delete during gathering. From a project's own config: ignored until `pedacito trust`. |
| `allow_shell` | false | `run` and `diagnose`. From a project's own config: ignored until `pedacito trust`. |
| `allow_serve` / `serve_wait` | false / 3.0 | Background dev servers (`serve` action, `pedacito serve`); seconds to wait for a printed URL. A server that prints one later still gets it recorded, on the next `procs`/`logs` look. |
| `allow_lsp` / `lsp_servers` / `lsp_timeout` | false / `{}` / 10 | Exact definition/references/hover/diagnostics via a language server; per-extension commands override PATH detection. |
| `allow_screenshots` / `allow_browser` | false | Screen and headless-browser capture. From a project's own config: ignored until `pedacito trust`. |
| `vision_model` | `""` | Multimodal model used to describe captures. |
| `semantic_search` | false | Embeddings fallback for `grep`. |
| `circuit_breaker` / `circuit_failures` / `circuit_cooldown` | true / 4 / 60 | Fail fast after repeated transport/server failures; rate limits, context overflows and 4xx errors do not trip it. |
| `allow_diff` | true | Read-only `diff` action. |
| `disable_thinking` | false | Ask for non-thinking mode using the model family's switch; output blocks are stripped regardless. |
| `default_profile` | `""` | Profile used when `--profile` isn't given. |
| `scout_profile` | `""` | Profile whose client runs gather's lookup steps while the active (architect) profile plans and edits; empty = no scout. |
| `specialties` | `()` | What this profile's model is good at, e.g. `["coding", "testing_qa"]`; empty = the model card detects them (name heuristics + architecture library + six micro-probes). An explicit list always beats the card. See [Specialist auto-routing](ARCHITECTURE.md#specialist-auto-routing). |
| `auto_route_specialties` | false | Route each task to the best-matching specialist profile at task boundaries (clean VRAM handoff on one local server); `--auto-route` / `--no-auto-route` flip it per run. |
| `prompt_pack` | `""` | `tiny` / `small` / `full` instruction density; empty = model card, then tier. |
| `answer_mode` | `""` | `rebuild` / `continue` / `auto`; empty = model card, then `auto` (hosted) or `rebuild` (local). |
| `compaction` | `"deterministic"` | How folded lookups are summarised: `deterministic`, `model`, or `off` (legacy per-turn clipping). |
| `rank_map` | true | Rank the map for the task when it doesn't all fit. |
| `repo_brief` / `max_brief_tokens` | true / 500 | Deterministic orientation block above the map. |
| `verify_answer` | `"deterministic"` | `off` / `deterministic` / `full`: check edits before writing them. |
| `best_of` | 1 | Sample N candidate edit replies and keep the best-scoring one. |
| `verify_tests` / `test_command` | false / `""` | Run impacted tests after applied edits (needs `allow_shell`). |
| `verify_dom` / `dom_entry` / `dom_timeout` | false / `""` / 45 | Headless DOM smoke check after applied web edits (needs `allow_shell` and the optional Playwright install): load the entry page in headless Chromium, prove the module graph mounts, and diff the mounted surface against a pre-edit snapshot. `dom_entry` overrides the detected entry HTML; `dom_timeout` caps one page load in seconds. `ask --verify-dom` flips it per run; `pedacito domcheck` runs it on demand. |
| `auto_smoke` / `auto_smoke_timeout` | true / 20 | After an `auto --apply` run, exercise the project once (the `--verify` command and/or a boot of its Python entry points) and feed a failure back for repair. `verify: passed` proves the project BOOTS -- it is not the test suite; end plans with a run-the-tests task or pass `--verify`. |
| `use_memory` / `use_skills` | true | Push memory, rules, and the skills catalog. |
| `allow_git` / `git_context` | true | Read-only `git` action; branch/dirty/recent-commits preamble. |
| `allow_subagents` | false | The `task` action (read-only helper with a fresh window). |
| `allow_fetch` / `fetch_allowed_hosts` | false / `[]` | The `fetch` action and the hosts it may read. |
| `roles` | `{}` | Per-job profiles: `planner`, `verifier`, `summarizer`, `repair`. |
| `auto_escalate` / `escalation_zero_edit_runs` | true / 3 | Widen the `escalate` retry beyond "no edit landed": a semantic repair failure, two no-progress leases, or this many consecutive zero-edit runs on one profile earn one retry even when edits applied. Every reason emits an `escalation_suggested` event even without a configured role; 0 disables the zero-edit statistic. |
| `hooks` / `hook_timeout` | `{}` / 10 | Your commands at events, each capped at this many seconds (see [Hooks](#hooks); project-level hooks need `pedacito trust`). |
| `scrub_env` / `env_passthrough` | true / `[]` | Minimal environment for model-chosen commands. |
| `protected_paths` / `shell_deny` | `[]` | Paths the model may not write; command patterns `run` refuses. |
| `wrap_observations` / `redact_secrets` | true | Fence untrusted output; mask secrets in logs and sessions. |
| `max_total_tokens` / `max_cost` | 0 | Stop before the next call once spent (0 = no limit). |
| `price_input_per_mtok` / `price_output_per_mtok` / `price_cached_per_mtok` / `price_cache_write_per_mtok` | 0 | USD per million tokens, for cost reporting; the write price covers explicit-cache creation tokens. |
| `prompt_cache` / `prompt_cache_ttl` / `prompt_cache_key` | `auto` / `5m` / `""` | Hosted prompt caching (see [Hosted prompt caching](#hosted-prompt-caching)): `auto` lets the provider table decide, `implicit`/`explicit`/`off` override it, `prompt_cache_key = "auto"` derives a stable cache-affinity key. |
| `enable_prompt_cache` / `enable_prompt_lookup` / `prompt_lookup_num_tokens` | true / false / 10 | Local server acceleration (see [Local inference acceleration](#local-inference-acceleration)): send `cache_prompt` and/or `prompt_lookup_num_tokens` to self-hosted endpoints only; a 400/422 refusal strips the key and is remembered per server+model. |
| `enable_mcp` / `mcp_servers` | true / `{}` | The named-server MCP bridge (see [Named MCP servers](#named-mcp-servers)): `[mcp_servers.<name>]` tables with `command`/`args`/`url`/`env`/`enabled`/`trusted`; the legacy `mcp_servers = ["cmd", ...]` list keeps feeding the `mcp` action. Servers defined by an untrusted project's `.pedacito.toml` are ignored. |
| `stuck_recovery` | true | One targeted recovery prompt before gather gives up. |
| `use_model_card` | true | Let `pedacito calibrate`'s card fill unset settings. |
| `show_telemetry` | true | One-line telemetry ribbon on stderr after each task (context bar, turn tokens, speed, cache hit); never shown in `--jsonl` / rpc modes. See [Live telemetry](USAGE.md#live-telemetry). |
| `check_for_updates` | true | Quiet PyPI release check at startup: one anonymous GET (2 s timeout, cached 24 h beside the user config), a dismissible GUI banner with streamed upgrade + Restart, `pedacito update [--run]`, TUI `/update`. `false` = zero network calls and zero upgrade subprocesses everywhere. See [Upgrading](INSTALLATION.md#upgrading). |
| `update_channel` | `"stable"` | Which PyPI releases count as "latest": `"stable"` ignores pre/dev/rc/alpha/beta tags; `"any"` includes them. `PEDACITO_UPDATE_CHANNEL` overrides it like any other setting. See [Upgrading](INSTALLATION.md#upgrading). |
| `log_events` / `keep_events` | true / 20 | Local JSONL telemetry under `.pedacito/events/`. |
| `summary_workers` / `refresh_summaries` | 0 / 12 | Concurrent summaries (0 = auto: hosted 4, local 1); units re-summarised after a task. |
| `summary_batch` | 1 | Short units summarised N per request while indexing (fewer round-trips; unparsed lines fall back to single requests). |
| `max_observation_tokens` | 0 | Ceiling on one non-read observation, in tokens (0 = auto: a quarter of the window, 300-2200). |
| `plan_threshold` / `max_replans` | 0 / 1 | Auto-plan tasks naming this many files or requirements; tasks naming 5+ files auto-plan even at 0 unless this is set explicitly (`--plan` forces one task, `--no-plan` opts out); planner revisions after a failed step. |
| `allow_steering` | true | Deliver lines typed mid-task to the model in `chat`. |
| `tool_mode` | `grammar` | `native` sends lookup steps through the model's tool-call channel; `calibrate` picks. |
| `step_reasoning_first` | true | `true` puts `reasoning` before the action in each lookup step (the model says what it needs, then picks); unset lets the model card decide. `calibrate` measures both orders with the same adopt-only-if-0.1-better rule as `tool_mode`. |
| `fewshot_steps` | unset | `true` shows three worked steps from your project (grep a symbol, read it, done) once in the first user turn, even under grammar or native tool calls; unset lets the model card decide. About 70 extra prompt tokens on every lookup call (the server caches them after the first, but a prompt-token count such as `eval`'s still includes them on each call). `calibrate` adopts it only if it picks the right action 0.1 more often with no loss of well-formed steps. |
| `edit_plan_brief` | true | The edit brief asks for 1-3 plan lines before the blocks (file, lines or symbol, why) and no other text between or inside blocks. A prompt override (`edit_format.md`, `answer.md`) bypasses it. |
| `sandbox` / `sandbox_network` | true / false | OS sandbox for `run` and test commands; allow network inside it. |
| `sandbox_writable` / `sandbox_strict` | `[]` / false | Extra writable dirs; refuse to run when no sandbox can be applied. |
| `atomic_edits` | false | Write a multi-file reply all-or-nothing (any failure writes nothing). |

Everything else (chunking, budgets, browser timeouts, auto-loop behavior) has a
working default and is documented in `pedacito config --reference`.
