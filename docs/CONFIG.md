🌐 **English** | [Español](es/CONFIG.md)

# Configuration

Pedacito is configured with TOML files, a small set of project files, and environment
variables. This guide covers where each setting lives, the project rules file, the color
scheme tokens, model profiles, language selection, and the full settings reference.

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

The engine ships nine built-in schemes — the same set the GUI offers: `slate` (the default,
GitHub Dark), `light` (GitHub Light), `daylight` (warm paper), `nord` (modern dark), `midnight`
(high contrast), and four 1990s palettes: `jazz_solo`, `trapper_keeper`, `memphis_arcade` and
`windbreaker`. In the TUI, `/theme` lists them and `/theme NAME` switches and saves the choice.
In the GUI, the palette button in the top bar opens the Theme Manager, which lists the same
schemes and your custom themes.

The `gui_theme` key in `config.toml` holds one of the same nine names; the GUI's Settings picker
and the Theme Manager's scheme pills both write it. A browser with no `gui_theme` preference
follows the theme selected in the Theme Manager or the TUI (`themes.json`); custom palettes are
stored there too. A saved preference naming a removed legacy preset falls back to `slate`.

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
| `api.anthropic.com` | Explicit opt-in (native) | native `/v1/messages` with `cache_control` markers (up to 4 breakpoints, `5m`/`1h` TTLs), `json_mode = "schema"`; temperature below 1 is omitted (Claude 4.7+). See `anthropic_native` to fall back to the OpenAI-compatible shim |
| Moonshot, Gemini, xAI, Groq | Automatic / implicit | nothing extra; hits are counted when returned |

With `anthropic_native = true` (the default) Anthropic's own API uses the native
`/v1/messages` transport instead of the OpenAI-compatible shim. Pedacito places
explicit `cache_control` breakpoints on the stable prefix (up to 4), honours the
`5m` / `1h` `prompt_cache_ttl`, and reports `cache_read_input_tokens` and
`cache_creation_input_tokens` from the response as cache hits and writes.
Extended thinking, tool calls and image inputs also use the native wire format.
Set `anthropic_native = false` to fall back to the OpenAI-compatible layer.

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

## Local, LAN and Tailscale endpoints

Every decision that depends on "is this a local server?" goes through one
predicate, so the tier heuristic, the sampling rules, the acceleration keys and
the calibration stale check all agree. An endpoint counts as local when its host
is loopback, a `.local` name, an RFC 1918 address (`10/8`, `172.16/12`,
`192.168/16`), a Tailscale or other RFC 6598 CGNAT address (`100.64.0.0/10`), a
link-local address, an IPv6 unique-local address (`fc00::/7`), or a provider label
that names a local runtime:

```toml
[profiles.homebox]
base_url = "http://100.101.102.103:1234/v1"   # a Tailscale address counts as local
model = "qwen3.5-27b"
```

Local endpoints keep the local sampler, `cache_prompt` and the local-MoE rules.
A local model card that has no calibrated temperature is marked **stale**, so
run `pedacito calibrate --recalibrate` once after upgrading to v0.3.3.

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

### Server lifecycle commands

A local profile's `load_cmd` / `unload_cmd` are the fallback when the runtime's
own lifecycle API cannot load or unload the model. They run as an argv — never
through a shell — with the same scrubbed environment as hooks
(`scrub_env` / `env_passthrough`), a 120-second timeout, and the
`{model}` / `{server}` textual substitutions. They also receive:

| Variable | Value |
|---|---|
| `PEDACITO_SERVER_NAME` | the configured server name, or its base URL when unnamed |
| `PEDACITO_MODEL_PATH` | the model id or path passed to the load/unload action |

Neither name is a Pedacito setting, so it cannot shadow a `PEDACITO_<KEY>`
config override. With `scrub_env = false` these commands inherit the ambient
environment; adding a name or `PREFIX*` pattern to `env_passthrough` exposes
just that variable while the allow-list scrub stays on. A project file needs
`pedacito trust` before its `load_cmd` / `unload_cmd` run.

On **Lemonade Server**, a loaded model can be pinned so its LRU eviction skips
it. In the GUI open **Servers**, pick the resident model in a Lemonade row and
click **Pin** or **Unpin**; in the TUI use `/servers pin SERVER [MODEL]` or
`/servers unpin SERVER [MODEL]`. Pedacito calls Lemonade's
`POST /internal/pin` with `{"model_name": "...", "pinned": true|false}` and
tries `/api/v1/internal/pin` when a build only answers that spelling. When
`LEMONADE_ADMIN_API_KEY` is set it is used for this internal endpoint;
otherwise the normal Lemonade key or the profile's `api_key` applies. The
model must already be loaded. Other local runtimes have no pin concept, so the
action is a no-op there.

When `auto_evict_servers` is on (the default), Pedacito unloads residents on
the *other* local servers before a run starts. It first reads each Lemonade
server's `/v1/health`; a pinned resident is never unloaded by that handshake.
Only unpinned residents on the same server are unloaded, and the pinned ids
are reported in the handshake's `skipped` list. Other runtimes expose no pin
state, so they keep the existing unload-everything behavior.

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
  injected up front, capped at 1,200 chars of checklist text. Nineteen starters
  ship built in: `add-a-test`, `fix-a-failing-test`, `refactor-safely`,
  `review-my-diff`, `find-dead-code`, `surgical-symbol-edit`,
  `frontend-dom-audio-debug`, `safe-multi-file-rename`, `fastapi-rest-endpoint`,
  `canvas-svg-2d-3d`, `sqlite-sql-migrations`, `async-concurrency-safety`,
  `security-ssrf-xss-hardening`, `i18n-localization-catalog`,
  `parser-lexer-state-machine`, `performance-memory-profiling`,
  `modern-web-design-ui`, `desktop-gui-and-games` and `safe-git-commit`.
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

`tier = "tiny" | "small" | "medium" | "large" | "local-moe" | "hosted"` fills in all token
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

`reasoning_effort` accepts `default`, `low`, `medium` and `high`. `default` is
the shipped spelling of the plan's `off`: it sends no `reasoning_effort`
parameter at all. The other three values are forwarded verbatim on local-MoE
and reasoning-capable hosted models, never on models that do not expose the
control, and they never change sampling. `disable_thinking = true` wins over
any effort level.

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
SSE timings. The probe drives one public one-shot call (`LLMClient.complete_once()`)
and reads the result through the public timing record
(`LLMClient.timing_record()`): request/first-token anchors, the server timing block
and token counts, never the stream's private bookkeeping. The probe is best-effort
and silent on failure; a cached result lives in `profiler.json` beside the user config for
one week and is keyed by server+model, so changing either one invalidates it. Probes never
run for hosted endpoints: both explicit paths (`pdc profile --run` and `/profile run`)
refuse with a "hosted model — refusing to probe" message before sending a request, and
the automatic post-load probe is skipped for a hosted URL.

## Verification tiers and the `auto_verify` family

Every applied edit passes up to three verification tiers, cheapest first. The
first two are deterministic and need no model call; the third is the command
you choose.

| Tier | What runs | Bounds and defaults |
|---|---|---|
| 1. Syntax / static gate | Changed Python files are checked for syntax (`E999`) and undefined names / bad `__all__` entries / misplaced imports (`F821`, `F822`, `F823`, `F402`, `F404`) with `ruff`; a built-in AST resolver takes over when ruff is missing, fails, or exceeds its ceiling. Pre-existing findings are baselined out, so only errors this edit introduced feed repair. | `static_gate = true`; `static_gate_timeout = 3.0` seconds for one ruff run. |
| 2. Scoped impacted tests | `testsel.run_impacted` selects tests from the import graph (tests named after the changed file, tests of its importers/callers, direct importers) and runs a narrowed `pytest FILE...` / `npm test -- FILE...` / `go test ./dir/...`. | Runs after edits when `verify_tests = true` and the test subprocess is allowed (`allow_shell` or a user-granted test command); one run is bounded by `test_timeout = 300` seconds. |
| 3. Full suite | Your verify command runs after each Auto task (or once at the end with `auto_verify_final`), and its failure output feeds the repair rounds. The full suite is deliberately never selected automatically. | `auto_verify = ""` (off) is a **command string**, never a bool; `auto_repair_rounds = 2`; `auto_verify_rounds = 2` for post-run runtime verification. |

`auto_verify` is the command the GUI's **Post-edit verify** checkbox and the
TUI's `/verify on` turn on (both first try
`testsel.detect_test_command`, and leave the key empty when no runner is
evident). `auto_verify_final = true` defers that command to the plan's last
unchecked task, so a whole-goal check cannot roll back valid partial fixes.
`auto_repair_rounds` caps fix rounds per failing per-task check;
`auto_verify_rounds` caps the runtime verification performed after an
`auto --apply` run.

For one **single-task** run, `pedacito ask --verify-cmd "COMMAND"` overrides
`auto_verify` in memory (the live config's `sources` entry becomes
`--verify-cmd`; no config file is written) and runs the command once after an
applied task. `pedacito ask --json` reports that outcome as `verify_passed`
(`null` when no verification ran) in its one-object summary; see
[USAGE.md — Headless and automation](USAGE.md#headless-and-automation) for the
exact JSON shape.

Trajectory export is a separate, opt-in dataset setting:
`record_trajectories = false` by default. When true, every `loop.run_task`
call writes one redacted ChatML transcript under
`.pedacito/trajectories/` (see
[SECURITY.md — Trajectory export](SECURITY.md#trajectory-export)).
`pedacito ask --save-trajectory` forces it for that run only, with a live
`sources` entry of `--save-trajectory`; like `--verify-cmd`, it never writes
the config file.

`auto_verify` and `test_command` are execution settings: in a project's
`.pedacito.toml` they are honored only after `pedacito trust` (see
[SECURITY.md](SECURITY.md#project-config-is-not-trusted-by-default)).

The tiers connect to two existing guards:

- **Green ratchet** (`auto_green_ratchet = true`): before each Auto step the
  source is snapshotted; a step still failing after its repair rounds is
  rolled back exactly, so later steps build only on the last verified-green
  workspace. `auto_keep_checkpoints = 25` verified-green snapshots are kept.
- **Interface lock** (`interface_lock = true`): the project's public signatures
  are injected into every Auto task, and a deterministic cross-file contract
  check (local import existence and call arity) runs after each edit before
  the test subprocess.

## Language & localization (`language`, `llm_reply_language`)

Pedacito ships an English and a Spanish interface.

| Setting | Values | Default | Meaning |
|---|---|---|---|
| `language` | `"en"` \| `"es"` | `"en"` | Interface language for menus, messages and the GUI/TUI chrome (catalogs in `pedacito/locales/<code>.json`). |
| `llm_reply_language` | `"match_ui"` \| `"auto"` \| `"en"` \| `"es"` | `"match_ui"` | Language of the model's prose explanations. `match_ui` follows `language`, `auto` follows the language of your prompt, `en`/`es` force one. Code, identifiers and edit blocks are never translated. |

Ways to choose the language, from highest to lowest precedence:

- **CLI:** `pedacito --lang es ...` (also accepted by `gui`, `config`, `how`, `models`, `profiles`, `update`).
- **Environment:** `PEDACITO_LANG=es` (alias `PEDACITO_LANGUAGE`).
- **Config:** `language = "es"` in your user or project config.
- **GUI:** the `EN | ES` switcher in the top bar; the choice is saved to `language`.
- **TUI:** `F3` toggles EN/ES, or type `/lang es`.

## Local image generation (`image_profile`, `allow_image_gen`)

Image generation is an opt-in modality, not a chat model. A profile with
`modality = "image"` (set automatically when `models scan` or the GUI sees a
Lemonade Server `sd-cpp` recipe, a Stable Diffusion / Flux server, or an image
model id) is routed to `/images/generations` instead of the text tool loop.

```toml
[profiles.flux]
base_url = "http://127.0.0.1:8000/api/v1"
model = "SD-Turbo"
modality = "image"
image_default_steps = 4

[universal]
image_profile = "flux"       # empty = first image profile found
allow_image_gen = true       # expose the generate_image agent action
```

- `pedacito image PROMPT`, the TUI `/image` command and the GUI **Project ▾ →
  🎨 Generate image…** dialog all use `image_profile`. Files land in
  `.pedacito/images/` unless you pass `--out`; the path must stay inside the
  workspace.
- `allow_image_gen` only controls the agent's `generate_image` action. With it
  off, the GUI permission broker can still approve one call at a time. It is
  an execution switch, so a project's own `.pedacito.toml` cannot turn it on
  until you run `pedacito trust`.
- `image_default_size` and `image_default_steps` are the fallbacks for any
  request that names neither.
- `pedacito calibrate` on an image profile runs a single 512×512 readiness
  probe instead of the text probes; the GUI's **Probe image** button does the
  same. See the [Model guide](../MODEL_GUIDE.md#image-models-modality--image).

## Unified policy & capability sandbox (`permission_mode`, `approval_policy`, `policy_file`)

Every capability decision — `run`, `write`, `delete`, hooks, MCP, network, and
subagent inheritance — funnels through one policy engine (`pedacito/policy.py`).
Two keys set the default stance, and a per-project policy file can tighten it:

| Setting | Values | Default | Meaning |
|---|---|---|---|
| `permission_mode` | `"plan"` \| `"ask"` \| `"accept-edits"` \| `"workspace-write"` \| `"danger"` | `"ask"` | The five-mode capability matrix. `plan` is read-only (reads + read-only subagents only); `ask` prompts for every mutation/execution; `accept-edits` auto-approves workspace writes/edits but still asks to run; `workspace-write` adds sandboxed `run` and configured hooks/subagents; `danger` allows everything the deny rules, safe-path and budget guards still permit. |
| `approval_policy` | `"ask"` \| `"never"` \| `"auto-safe"` | `"ask"` | How an `ask` verdict resolves: ask the human, refuse (unattended-safe), or auto-approve known safe read-only/test commands (`pytest`, `ruff`, `git status`, `git diff`, or a trusted `[commands].allow`). |
| `policy_file` | a path | `.pedacito/policy.toml` | The trust-gated capability policy file. Its `[policy]`, `[paths]`, `[commands]`, `[network]`, `[mcp]` and `[budget]` rules apply only after `pedacito trust`; an untrusted file can never widen permissions. |
| `daemon_port` | integer | `0` | Loopback port for the optional background daemon; `0` lets the OS choose an ephemeral one. |
| `daemon_auto_start` | bool | false | Auto-start the background daemon for a run. An execution switch: a project config cannot turn it on until `pedacito trust`. |
| `subagent_max_parallel` | integer | `3` | Cap on subagents that may run in parallel. |

The five modes are a strict lattice: `plan` ⊂ `ask` ⊂ `accept-edits` ⊂
`workspace-write` ⊂ `danger`, and a subagent can only ever *intersect* (never
widen) its parent's capabilities. `approval_policy = "never"` converts every
`ask` into a `deny`, so a background or headless run can never be upgraded
without a human. On Windows, `workspace-write`/`danger` command execution is a
**hard gate**: it runs under the built-in Job Object + restricted-token
(Low Integrity) sandbox, and `sandbox_strict` (or `approval_policy = "never"`
with `sandbox` on) refuses it outright when no OS sandbox is active.

A `.pedacito/policy.toml` looks like:

```toml
[policy]
permission_mode = "ask"               # plan | ask | accept-edits | workspace-write | danger
approval_policy = "ask"               # ask | never | auto-safe
subagent_inheritance = "narrow_only"  # narrow_only | inherit_readonly

[paths]
allow = ["src/**", "tests/**"]
deny = [".env*", "**/*.pem", "secrets/**"]

[commands]
allow = ["pytest *", "ruff *"]
deny = ["git push*", "rm -rf *", "curl *"]

[network]
allow_domains = ["127.0.0.1", "localhost"]
deny_domains = ["*"]

[mcp]
allow_servers = []
deny_servers = ["*"]

[budget]
max_cost_usd = 0.0
max_tokens = 0
max_steps = 0
```

## Environment resolution

Pedacito resolves the project's environment without executing anything.
`resolve_environment` reads local markers (`.venv`, `uv.lock`,
`poetry.lock`, `Pipfile`, `conda-meta`, `tox.ini`/`.tox`,
`noxfile.py`/`.nox`, pnpm/yarn/bun lockfiles, `Cargo.toml`, `go.mod`) and
existing interpreters, then applies a fixed precedence: an existing project
interpreter first, an explicitly active `VIRTUAL_ENV`/`CONDA_PREFIX` next,
then package-manager metadata, then the ambient `PATH`. Ambiguity is reported
in `pedacito doctor`'s evidence line; the choice is still deterministic. On
Windows, `Scripts` is used instead of `bin` and `PATHEXT` executable
extensions are checked.

`test_command` remains authoritative: an explicit command is never rewritten.
Detected commands, `run_test`, `diagnose`, LSP servers and the `run`
launches receive the resolved bin directories prepended to `PATH` and the
usual scrubbed child environment (`scrub_env` / `env_passthrough` still
apply), so project tools win without leaking ambient secrets. An explicit
`test_command` may still say `python`; the child `PATH` keeps the project's
venv directories, but the command text is unchanged. `pedacito doctor` (the
documented alias of `pedacito config --doctor`) prints the resolved kind,
interpreter, `PATH` directories, package manager and evidence.

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
| `load_cmd` / `unload_cmd` | `""` / `""` | Fallback argv for a runtime with no usable lifecycle endpoint. Scrubbed environment plus `PEDACITO_SERVER_NAME` / `PEDACITO_MODEL_PATH`. See [Server lifecycle commands](#server-lifecycle-commands). |
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
| `tier` | `""` | Budget preset: `tiny`/`small`/`medium`/`large`/`local-moe`/`hosted`; guessed from the model name when unset. |
| `max_steps` | 0 | Lookup limit per task; 0 = none. |
| `prefetch_targets` | 0 | Read up to this many task-named files/symbols through the normal tool path before the first lookup step (and once more after gather when a named target is still unread). Prefetch reads don't count toward `max_steps`. |
| `temperature` | 0.2 | Higher values make tool selection worse. A calibrated model card may replace this default; a value set on a profile, a `PEDACITO_*` variable or a CLI flag always wins. Local servers only: hosted APIs get no sampling keys by default (see [Hosted APIs keep their own sampling](#hosted-apis-keep-their-own-sampling)). |
| `temperature_answer` / `temperature_repair` | 0.3 / unset | Sampling temperature for the answer and for the repair re-asks, which copy SEARCH text character for character. The answer defaults to `0.3` — cool enough to copy exactly, warm enough to escape verbatim repetition loops (a greedy `0` froze weak local models); creative/design tasks run at ≥ 0.6. The repair re-asks are unset (they use `temperature`). Gather keeps `temperature`; `best_of` candidates run at the answer temperature + 0.4. |
| `send_temperature_to_hosted` | false | Send the configured `temperature` to hosted APIs anyway. The other sampling keys (`top_p`, `top_k`, `min_p`, `repeat_penalty`) stay omitted either way. |
| `hosted_temperature` | unset | Pin one temperature for hosted endpoints, globally or per profile (e.g. `1.0` for Claude); it wins over phase overrides and is the only sampling key sent. Unset: the provider's own default applies. Local servers ignore it. |
| `anthropic_native` | true | Route Anthropic's own API through the native `/v1/messages` transport (prompt caching, streaming usage, extended thinking, tools and images); `false` uses the OpenAI-compatible shim. |
| `gemini_native` | true | Route Google Gemini (`generativelanguage.googleapis.com` without `/openai`, or `provider = "gemini"`) through the native `/v1beta` `generateContent` transport (thought parts, inline images, function declarations, `thinkingConfig`, and `cachedContent`); `false` uses the OpenAI-compatible shim. |
| `openai_responses` | false | Opt-in OpenAI Responses API (`/v1/responses`) transport for reasoning models (`o3`/`o4`/`GPT-5`), preserving encrypted reasoning items and `previous_response_id` across tool loops; also enabled by `provider = "openai-responses"`. |
| `auto_refresh_model_catalog` | true | Quietly refresh the local `models.dev` cache (`model_catalog.json`, 24 h TTL) when probing or refreshing hosted providers so context windows and pricing stay current; `false` keeps everything strictly offline. Lookups use three tiers: the provider's own `/models` metadata (tier 1), this cache (tier 2), then the built-in price table (tier 3); explicit `price_*` settings always win. `pedacito models refresh` forces a refresh. |
| `api_key_keyring` | `""` | Optional OS keychain reference `<service>/<account>` (or `<account>`, defaulting service to `pedacito`) resolved lazily via `pedacito[keyring]`; alternative to `api_key = "keyring:<service>/<account>"`. |
| `backup` | true | Snapshot originals before writing. |
| `log_sessions` | true | Write a session record per task. |
| `record_trajectories` | false | Opt-in full-prompt trajectory export for SFT/GRPO dataset filtering: one JSON file per task under `.pedacito/trajectories/`, holding the real ChatML/OpenAI-shaped messages (system, user, assistant, tool calls and tool results) plus metadata (`verify_passed`, `harness_version`, `edit_format`, `model_id`, `provider`, `task_hash`, `started_at`, `elapsed_s`, token totals). Unlike `replay` it contains full prompts; every string is passed through `safety.redact` and the serialised file is re-scanned with `security.scan_text_for_secrets`, with a residual finding masked or the write refused. Local only, no network or telemetry. `pdc ask --save-trajectory` forces capture for one run in memory (live `sources` entry `--save-trajectory`) without writing the key. See [SECURITY.md — Trajectory export](SECURITY.md#trajectory-export). |
| `review` | false | Always use the review page. |
| `fuzzy_match` | false | Anchor near-miss search blocks approximately. |
| `edit_style` | unset | `quote` (default brief) or `anchor` (prefer `SYMBOL` + full definition, SEARCH only for small edits); unset lets the model card decide. |
| `creative_mode` | false | Creative/design (Builder) mode for EVERY task: designer persona, answer temperature ≥ 0.6 on local servers, an ≥ 8192-token reply budget, whole-file HTML/CSS rewrites. `--creative`, TUI `/creative on`, or the GUI's ✨ Creative toggle set it per run. See [Creative / design mode](USAGE.md#creative--design-mode). |
| `creative_auto` | true | Auto-detect design tasks (website/portfolio/redesign prompts, `.html`/`.css`/`.svg`/`.jsx`/`.tsx`/`.vue`/`.svelte` targets) when `creative_mode` is off. Both false = surgical edits everywhere (`/creative off`). |
| `creative_temperature_delta` | unset | Extra temperature added to the answer baseline in Creative/design mode (relative sway, not a static floor). Unset = automatic by tier (`+0.20` tiny/small, `+0.30` otherwise); an explicit float in `0.0`–`0.8` overrides it. Capped at `0.95` (`1.0` for an explicit delta) and never below the baseline, so a calibrated `0.6` Gemma becomes `0.9` while a `0.2` tiny model nudges to `0.4`. |
| `post_edit_check` | true | After applied edits, feed back only NEW linter/syntax findings on the changed files. |
| `auto_allowlist` | false | Restrict `auto` task writes to files the title names or the lookup read (new files always allowed). |
| `allow_write` | false | Whole-file write/append/move/delete during gathering. From a project's own config: ignored until `pedacito trust`. |
| `allow_shell` | false | `run` and `diagnose`. From a project's own config: ignored until `pedacito trust`. |
| `allow_serve` / `serve_wait` | false / 3.0 | Background dev servers (`serve` action, `pedacito serve`); seconds to wait for a printed URL. A server that prints one later still gets it recorded, on the next `procs`/`logs` look. |
| `allow_lsp` / `lsp_servers` / `lsp_timeout` | false / `{}` / 10 | Exact definition/references/hover/diagnostics via a language server; per-extension commands override PATH detection. |
| `allow_screenshots` / `allow_browser` | false | Screen and headless-browser capture. From a project's own config: ignored until `pedacito trust`. |
| `vision_model` | `""` | Multimodal model used to describe captures. |
| `modality` | `""` | What kind of model a profile serves: `""` (auto-detect from the model id and server metadata), `"text"`, `"image"`, `"audio"` or `"embedding"`. Only `"image"` opens the [local image generation](#local-image-generation-image_profile-allow_image_gen) path. |
| `image_profile` | `""` | The `[profiles.*]` entry that serves `/images/generations`. Empty = use the first image-model profile found. |
| `allow_image_gen` | false | Expose the opt-in `generate_image` agent action. From a project's own config: ignored until `pedacito trust`. |
| `image_default_size` | `"1024x1024"` | `size` sent to `/images/generations` when the caller passes none. |
| `image_default_steps` | 0 | Diffusion steps sent when the caller passes none (0 = the server's own default). |
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
| `verify_tests` / `test_command` | false / `""` | Run impacted tests after applied edits (needs `allow_shell`); `test_command` is the runner detection override. |
| `auto_verify` | `""` | Command string run after each Auto task; a failure feeds its output back for repair. Empty = off; never a bool. The TUI `/verify on` and GUI **Post-edit verify** detect and fill it. `pdc ask --verify-cmd "..."` overrides it for one run, in memory (see [USAGE.md](USAGE.md#headless-and-automation)). |
| `auto_verify_final` | false | Run `auto_verify` once after the plan's last unchecked task instead of after every task (useful when the whole suite cannot pass mid-plan). |
| `auto_repair_rounds` / `auto_verify_rounds` | 2 / 2 | Repair rounds per failing per-task check; runtime-verification rounds after an `auto --apply` run. |
| `static_gate` / `static_gate_timeout` | true / 3.0 | Changed Python files get a ruff-or-AST static check (new errors only) before any test subprocess; seconds allowed for one ruff run. |
| `auto_green_ratchet` / `auto_keep_checkpoints` | true / 25 | Snapshot each Auto step and roll a step that still fails after repair back to the last verified-green workspace; verified-green snapshots kept per project. |
| `interface_lock` | true | Inject public signatures into Auto tasks and gate each edit on local import existence and call arity. |
| `verify_dom` / `dom_entry` / `dom_timeout` | false / `""` / 45 | Headless DOM smoke check after applied web edits (needs `allow_shell` and the optional Playwright install): load the entry page in headless Chromium, prove the module graph mounts, and diff the mounted surface against a pre-edit snapshot. `dom_entry` overrides the detected entry HTML; `dom_timeout` caps one page load in seconds. `ask --verify-dom` flips it per run; `pedacito domcheck` runs it on demand. |
| `auto_smoke` / `auto_smoke_timeout` | true / 20 | After an `auto --apply` run, exercise the project once (the `--verify` command and/or a boot of its Python entry points) and feed a failure back for repair. `verify: passed` proves the project BOOTS -- it is not the test suite; end plans with a run-the-tests task or pass `--verify`. |
| `use_memory` / `use_skills` | true | Push memory, rules, and the skills catalog. |
| `allow_git` / `git_context` | true | Read-only `git` action; branch/dirty/recent-commits preamble. |
| `allow_subagents` | false | The `task` action (read-only helper with a fresh window). |
| `allow_fetch` / `fetch_allowed_hosts` | false / `[]` | The `fetch` action and the hosts it may read. |
| `roles` | `{}` | Per-job profiles: `planner`, `verifier`, `summarizer`, `repair`, `escalate`, `fallback`. |
| `proactive_escalation` | false | Opt-in (v0.3.3) two-path router that runs once per task. A zero-token structural gate (import fan-out, files touched, module boundaries, open-ended architecture words) sends a hard task straight to the `escalate` role; a boundary-fewshot Scout classifier keeps well-defined single-file tasks local. Needs `roles.escalate`; `false` leaves reactive escalation unchanged. |
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
| `show_update_reminder` | true | Show a concise 1-line active-development reminder on interactive TUI/chat exit (never in `--headless`, `--json`, `rpc`, `acp` or non-TTY pipes) and a subtle dismissible notice in the GUI reminding you to run `pedacito update` regularly. `false` opts out. |
| `picard_mode` | false | Cosmetic Captain Picard Mode: relabels GUI/TUI submit, OK, Save, Apply, Run, and Commit buttons to `'Make it so.'` (with the original label preserved in the button tooltip); TUI `/picard [on\|off]` toggles it live. |
| `log_events` / `keep_events` | true / 20 | Local JSONL telemetry under `.pedacito/events/`. |
| `summary_workers` / `refresh_summaries` | 0 / 12 | Concurrent summaries (0 = auto: hosted 4, local 1); units re-summarised after a task. |
| `summary_batch` | 1 | Short units summarised N per request while indexing (fewer round-trips; unparsed lines fall back to single requests). |
| `max_observation_tokens` | 0 | Ceiling on one non-read observation, in tokens (0 = auto: a quarter of the window, 300-2200). An observation over the ceiling (run, run_test, grep, diff, fetch, diagnose, serve) is spilled to `.pedacito/scratch/*.log` (v0.3.3) and the model gets a head, a tail and the file name to `read` or `grep` on demand. |
| `plan_threshold` / `max_replans` | 0 / 1 | Auto-plan tasks naming this many files or requirements; tasks naming 5+ files auto-plan even at 0 unless this is set explicitly (`--plan` forces one task, `--no-plan` opts out); planner revisions after a failed step. |
| `allow_steering` | true | Deliver lines typed mid-task to the model in `chat`. |
| `tool_mode` | `grammar` | `native` sends lookup steps through the model's tool-call channel; `plain` (v0.3.3) sends them as short plain words (`read app.py 1 40`), which suits small models that stumble on JSON; `calibrate` A/B-tests the formats and keeps the one that scores better. Valid values: `grammar`, `native`, `plain`. |
| `execution_mode` | `two_phase` | The task engine: `two_phase` (gather then answer), `interleaved` (one agentic read/edit/test conversation) or `auto` (interleaved for hosted APIs, local-MoE servers and calibrated large local models whose card chose native tool calls; `two_phase` for tiny/small/medium and uncalibrated local models). `--execution-mode` overrides it per run; the TUI's `/mode` switches it live. |
| `step_reasoning_first` | true | `true` puts `reasoning` before the action in each lookup step (the model says what it needs, then picks); unset lets the model card decide. `calibrate` measures both orders with the same adopt-only-if-0.1-better rule as `tool_mode`. |
| `fewshot_steps` | unset | `true` shows three worked steps from your project (grep a symbol, read it, done) once in the first user turn, even under grammar or native tool calls; unset lets the model card decide. About 70 extra prompt tokens on every lookup call (the server caches them after the first, but a prompt-token count such as `eval`'s still includes them on each call). `calibrate` adopts it only if it picks the right action 0.1 more often with no loss of well-formed steps. |
| `edit_plan_brief` | true | The edit brief asks for 1-3 plan lines before the blocks (file, lines or symbol, why) and no other text between or inside blocks. A prompt override (`edit_format.md`, `answer.md`) bypasses it. |
| `sandbox` / `sandbox_network` | true / false | OS sandbox for `run` and test commands; allow network inside it. |
| `sandbox_writable` / `sandbox_strict` | `[]` / false | Extra writable dirs; refuse to run when no sandbox can be applied. |
| `atomic_edits` | false (auto for local small tiers) | Staged all-or-nothing multi-file apply: every file is validated and its preimage snapshotted to a durable WAL before the first write; a write failure rolls back the whole batch and a crash mid-commit is recovered on the next run. An explicit `true`/`false` always wins; left unset it turns itself on only for a local endpoint whose tier is tiny/small/medium. |

Every remaining public setting is listed below, grouped by subsystem, with its real default.
`pedacito config --reference` prints the same keys as an annotated template.

<details>
<summary>Complete advanced settings reference (Connection, budgets & profile selectors)</summary>

| Setting | Default | Meaning |
|---|---|---|
| `timeout` | `600.0` | Seconds to wait for one request before failing. |
| `max_concurrency` | `0` | Concurrent requests this endpoint can serve. |
| `max_consecutive_failures` | `3` | Stop a batch after this many failures in a row. |
| `max_tokens_summary` | `220` | Output-token budget for one indexing summary call. |
| `max_tokens_step` | `700` | One tool call plus a short note; the schema puts the action first so a truncated reply loses only the explanation. |
| `max_tokens_answer` | `3000` | Output-token budget for the answer phase. |
| `max_total_model_calls` | `30` | Action-independent global safety bounds that still apply when max_steps = 0 removes the lookup cap: a hard ceiling on total LLM invocations per task, and a ceiling on consecutive gather steps that show no progress (ad... |
| `max_no_progress_steps` | `4` | Consecutive gather steps that add no new code or symbol before gather ends cleanly. |
| `max_task_completion_tokens` | `60000` | Task-level budget guard (BUG-218): ceilings on ONE task's total completion tokens and wall-clock seconds, gather + answer + repair together. |
| `max_task_seconds` | `900` | Wall-clock ceiling for one task; 0 disables it. |
| `max_output_tokens` | `0` | Per-message OUTPUT ceiling the provider honours (hosted APIs cap one reply far below their context window, e.g. ~8k tokens). |
| `reserve_tokens` | `2048` | Tokens held back from context_tokens for the reply itself; auto_fit lowers it when the window is too small. |
| `request_max_tokens_cap` | `32768` | max_tokens sent on generation calls; auto_fit clamps it to the window. |
| `max_tokens_plan` | `2000` | Budgets for the two scaffold calls with no existing answer budget. |
| `max_tokens_build` | `8000` | Output-token budget for the scaffold build call. |
| `stream_usage` | `true` | Ask the server to report usage on streamed replies too (OpenAI `stream_options.include_usage`); without it the answer phase's token, cost, and cache-hit numbers are unknown. |
| `language` | `"en"` | Interface language: "en" or "es". |
| `llm_reply_language` | `"match_ui"` | What language the model writes its prose explanations in. |
| `benchmarks_dir` | `""` | Parent folder for the built-in model benchmark projects. |
| `profiles` | `{}` | Named model profiles (`[profiles.NAME]` tables); managed by the Profiles tab and `pedacito models`. |
| `active_profile` | `''` | Name of the profile selected for this run (set by `--profile` / `default_profile`). |
| `active_tier` | `''` | Budget tier resolved for this run (from `tier` or the model name). |
| `model_size_bytes` | `0` | Model weight size reported by the server, used by the footprint estimate; 0 when unknown. |

</details>

<details>
<summary>Complete advanced settings reference (Indexing & Chunking)</summary>

| Setting | Default | Meaning |
|---|---|---|
| `file_card_line_ranges` | `true` | Attach L<start>-<end> to names on file cards (saves an `outline` call). |
| `class_split_lines` | `60` | Classes shorter than this are indexed as one chunk instead of per-method. |
| `max_unit_lines` | `220` | A function/method (or module preamble/trailing block) larger than either limit is split at top-level statement boundaries into name#2, name#3, ... |
| `max_unit_chars` | `12000` | Split a function, method or preamble larger than this many characters into name#2, name#3, ... |
| `max_unit_tokens` | `3000` | Token guard (~3.6 chars/token) for small contexts; 0 disables. |
| `unit_overlap_lines` | `0` | Lines repeated between line windows when one giant statement must be windowed. |
| `max_chunk_chars` | `6000` | Chunks larger than this get their source truncated before summarising. |
| `min_lines_to_summarise` | `4` | Skip summarising trivially short chunks; the signature says enough. |
| `summarise_methods` | `false` | Summarisation rationing: a docstring is already a summary, and a method is covered by its class summary plus its signature. |
| `summarise_documented` | `false` | Also summarise units that already have a docstring. |
| `include_references` | `true` | Index non-Python files (.md .csv .json .toml .yaml ...) structurally: the map shows a CSV's columns or a config's sections; contents are never pulled into the map. |
| `extra_extensions` | `[]` | Extra extensions to treat as reference files, e.g. [".jinja", ".proto"]. |
| `max_reference_bytes` | `2000000` | Above this size a reference file is recorded by name and header only. |
| `summarise_references` | `false` | Summarise reference files (markdown, JSON, ...) instead of listing only their structure. |
| `use_project_instructions` | `true` | Include the repository's PEDACITO.md / AGENTS.md / CLAUDE.md in every prompt. |
| `manage_gitignore` | `true` | On first run, add .pedacito/ to the project's .gitignore. |
| `create_ignore_file` | `true` | On first run, create a commented .pedacitoignore. |
| `use_ripgrep` | `true` | Use ripgrep for `grep` when installed (same output format). |

</details>

<details>
<summary>Complete advanced settings reference (Gather & Loop Leases)</summary>

| Setting | Default | Meaning |
|---|---|---|
| `max_todo_chars` | `1500` | Cap on the model's own checklist when injected into a prompt. |
| `dynamic_tool_pruning` | `true` | Dynamic tool schema pruning: keep the core navigation actions always available, but omit an enabled optional/specialized action from the prompt schema once the repository's tool-use stats show it has gone unused for `... |
| `tool_prune_after_tasks` | `10` | Consecutive tasks a tool may go unused before pruning considers it. |
| `max_completion_rounds` | `0` | Limit on follow-up answer turns when the model defers remaining work; 0 = no limit (stops on NOTHING MORE or when no blocks are produced). |
| `max_continuation_rounds` | `0` | Limit on continuation turns when a reply is cut off mid-block; 0 = no limit (the continuation char budget is the real bound). |
| `max_rehydration_rounds` | `2` | Rounds of `NEED path/to/file.py:120-168` exact-span rehydration allowed per answer, repair or verifier generation: the harness fetches ONLY those spans (hash-checked, token-clipped like a `read`), APPENDS them at the... |
| `max_read_lines` | `200` | Maximum lines returned by one `read` call. |
| `read_context_hint` | `true` | Attach the enclosing signature and module imports when a `read` starts mid- file. |
| `read_skeleton` | `true` | When a `read` targets a class/module larger than the read budget, return a deterministic AST skeleton (signatures + docstrings + [lines X-Y] pointers, bodies folded) instead of the first page of the giant unit; contin... |
| `compact_step_schema` | `false` | EXPERIMENTAL: require only `action` and `reasoning` in the gather step schema so grammar-constrained decoding may omit unused fields (~20-40 output tokens per step). |
| `fold_tail_steps` | `0` | Recent gather steps kept verbatim when older ones are folded into a STATE digest at window pressure; 0 = chosen from the tier. |
| `max_memory_chars` | `1500` | Cap on project and user memory text included in prompts. |
| `max_skill_catalog_chars` | `900` | Cap on the skills catalog block in prompts. |
| `subagent_max_steps` | `8` | Lookup-step limit for one `task` subagent. |
| `max_fetch_chars` | `6000` | Cap on fetched page text fed back to the model. |

</details>

<details>
<summary>Complete advanced settings reference (Editing & Repair)</summary>

| Setting | Default | Meaning |
|---|---|---|
| `auto_diagnostics` | `false` | After each `auto` task (when no --verify command is set), run the project's detected linter/type checker and feed failures into the repair rounds. |
| `diagnostics_timeout` | `90` | Seconds allowed for one auto-diagnostics linter/typechecker run. |
| `max_diagnostics_chars` | `8000` | Cap on diagnostics text fed back into repair. |
| `max_write_chars` | `200000` | Ceiling on one `write` call's contents, in characters. |
| `max_write_lines` | `120` | Maximum lines one reply may emit for one file (gather `write` or answer-phase edit block). |
| `allow_rename` | `true` | Treat a creation of a new path paired with a block that empties an existing same-extension file as a real move (old file deleted). |
| `fuzzy_min_ratio` | `0.85` | Minimum similarity for a fuzzy edit anchor; below this the edit fails instead of guessing. |
| `repair_rounds` | `2` | Repair round-trips per apply when a search/replace block fails to match. |
| `incremental_edit_loop` | `true` | Symbol-anchored incremental edit-verify microloops. |
| `max_new_files` | `24` | Cap on files a plan may propose; the excess is reported. |
| `max_spec_chars` | `60000` | Spec documents longer than this are truncated, with a warning. |
| `test_timeout` | `300` | Seconds allowed for one test-command run. |

</details>

<details>
<summary>Complete advanced settings reference (Auto Loop & Checkpoints)</summary>

| Setting | Default | Meaning |
|---|---|---|
| `auto_plan_file` | `"PLAN.md"` | A goal becomes a checklist; each task is implemented in a fresh context with the plan file as the only surviving state. |
| `auto_max_tasks` | `0` | Cap on tasks per `auto` run; 0 = no cap. |
| `auto_max_minutes` | `0` | Wall-clock cap per `auto` run in minutes; 0 = no cap. |
| `auto_checkpoint_max_file_bytes` | `1000000` | Per-step snapshot bounds: a file larger than this is recorded but not copied (rollback leaves it alone); the walk stops after this many files, so one snapshot cannot fill the disk on a vendored monorepo. |
| `auto_checkpoint_max_files` | `1500` | Maximum files recorded in one auto-step snapshot; the walk stops there. |
| `auto_split_lines` | `120` | `new --complex` / `auto`: a source file whose scaffold stub is longer than this is split into several tasks (one per group of top-level definitions). |
| `auto_install_deps` | `false` | pip-install a missing third-party module into the running interpreter when --verify fails on it, then retry. |
| `review_port` | `0` | Port for the review page; 0 lets the OS choose a free one. |
| `open_browser` | `true` | Open the review page in a browser automatically. |
| `review_timeout` | `1800` | Seconds before an unanswered review page expires. |
| `hybrid_routing` | `false` | Run each task in two stages: a strong architect model plans first (root cause, target files, exact signatures, edge cases), then the active editor model implements the blueprint. |
| `architect_provider` | `""` | The profile whose server/model plans in hybrid Stage 1. |
| `architect_model` | `""` | The model id the architect runs on. |
| `auto_route_allow_hosted` | `false` | Allow specialist routing to pick a HOSTED (remote API) profile. |

</details>

<details>
<summary>Complete advanced settings reference (Web, Browser, Vision & Semantic)</summary>

| Setting | Default | Meaning |
|---|---|---|
| `allow_mcp` | `false` | Optional MCP client: exposes one `mcp` action plus a short tool catalog. |
| `mcp_timeout` | `30` | Seconds allowed for one MCP server request. |
| `screenshot_wait_seconds` | `5` | Seconds to wait after launching before capturing; the action's `start` value overrides it. |
| `browser_wait_seconds` | `3` | Seconds to wait after navigation/interaction before capturing. |
| `browser_server_wait_seconds` | `15` | Seconds to wait for a launched server to print its localhost URL. |
| `browser_headless` | `true` | Run the browser action headless (no visible window). |
| `browser_timeout_seconds` | `20` | Per-navigation/action timeout in seconds. |
| `enable_web` | `false` | Allow the agent to fetch URLs or search documentation online. |
| `vision_max_tokens` | `400` | Output-token cap for the vision model's screenshot description. |
| `embedding_model` | `""` | Embedding model id; empty = the chat model. |
| `semantic_top_k` | `5` | Number of nearest indexed units semantic search reports. |
| `semantic_min_score` | `0.2` | Cosine similarity below this is not shown; scales differ by model. |
| `browser_eval_hosts` | `[localhost, 127.0.0.1, ::1, 0.0.0.0]` | Hosts where the browser `eval` action may run model-authored JavaScript. |

</details>

<details>
<summary>Complete advanced settings reference (Security, Sandbox & Retention)</summary>

| Setting | Default | Meaning |
|---|---|---|
| `shell_timeout` | `120` | Seconds before a `run` command is killed. |
| `shell_idle_seconds` | `10` | Kill a `run` command after this many seconds with NO output (keeps GUI apps and servers from blocking the agent); 0 disables the idle rule. |
| `shell_allowlist` | `[]` | Optional executable allowlist for `run`, e.g. ["pytest", "python"]; empty = any executable once allow_shell is on. |
| `shell_allowlist_by_project` | `{}` | Per-project shell allow-rules for `run`: a table mapping absolute project roots to pattern lists, e.g. { "/home/me/proj" = ["pytest *", "cargo test *"] }. A matching command skips the permission prompt in that project only. Rules never cover package installs, shells, or commands carrying chaining metacharacters (semicolons, ampersands, pipes, backticks, command substitution). The GUI/TUI "allow in this project" answer writes them here. Trust-gated like `shell_allowlist`: a project config's rules apply only after `pedacito trust`. |
| `max_shell_output_chars` | `8000` | Cap on command output fed back to the model, in characters. |
| `security_outbound_mode` | `"warn"` | Outbound cloud-provider guard: before a request leaves for a hosted endpoint, scan its messages for unacknowledged secrets. |
| `keep_sessions` | `0` | `pedacito prune` defaults (0 = keep everything). |
| `keep_backups` | `0` | Backups kept by `pedacito prune`; 0 keeps everything. |
| `log_max_bytes` | `2000000` | Rotate the event log after this many bytes. |

</details>
