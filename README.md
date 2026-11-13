# Pedacito

**Pedacito** is Spanish for *a little piece* or *a small chunk*. It is a coding
agent built for the small local models that run on the GPU you already own,
from 4B to 27B parameters.

Most coding agents assume you have a frontier model with a 200k-token window and
an open cloud budget. Point one at a 9B model on a 12 GB card and it tends to
stuff whole files into context until the model loses the thread, invents a
function, or breaks a diff. Pedacito works the other way around. It cuts your
codebase into small, precisely bounded pieces, so the model only reads and edits
the exact lines that matter.

It works with **LM Studio, Ollama, Lemonade Server, llama.cpp, vLLM**, and any
other OpenAI-compatible server. When you want more power, it can route work to
hosted models such as **Claude, ChatGPT, DeepSeek, Qwen, Google Gemini, and
OpenRouter**, with a one-click provider preset for each. A common setup is a fast
local model doing the edits and a larger model planning or verifying.

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

A fair warning: Pedacito was built by one person for one setup (local models on
a single home GPU box), so edge cases outside that path may be rough. If you can
use a frontier hosted model, [opencode](https://opencode.ai),
[Qwen Code](https://github.com/QwenLM/qwen-code), and
[Claude Code](https://claude.com/claude-code) are good choices. Pedacito is for
when you want to stay small and local, or you want to control exactly which
model sees what.

## Quickstart in 60 seconds

1. **Start a model server.** Load a model in LM Studio, Ollama, Lemonade, or
   llama.cpp. A hosted API works too, if you have a key.
2. **Install** from the repository folder (requires Python 3.11+):

   ```bash
   pip install -e .            # Windows: py -m pip install -e .
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
[QUICKSTART.md](QUICKSTART.md).

## What you get

| Area | What it does | Read more |
|---|---|---|
| **Small-model techniques** | A harness-run retrieval loop, a requirement state machine (`UNLOCATED → LOCATED → READ → EDITED → VERIFIED`), symbol-anchored edit microloops, an AST syntax gate before any test runs, traceback repair packets, KV prompt-cache-friendly conversations, and exact-span rehydration | [TECHNIQUES.md](TECHNIQUES.md) |
| **Engine V2 telemetry** | The GUI shows the live phase, the requirement ledger, and recent lookups as a task runs | [QUICKSTART.md](QUICKSTART.md#part-a-the-gui) |
| **Repository memory** | Confirmed commands in `.pedacito/micro_facts.json`, with contradicted facts retired automatically. Unused tool schemas are pruned to save tokens | [TECHNIQUES.md](TECHNIQUES.md) §14 and §15 |
| **Local server scanner** | Finds LM Studio, Ollama, Lemonade, and llama.cpp on any port, then imports only the models you tick | [Model server setup](#model-server-setup) |
| **Hosted provider presets** | One-click setup for Claude, ChatGPT, DeepSeek, Qwen, Gemini, and OpenRouter. Keys can come from an environment variable instead of the config file | [Configure](#configure) |
| **Role routing** | Send planner, verifier, or escalation work to a different profile, per job or per project | [MODEL_GUIDE.md](MODEL_GUIDE.md#hybrid-local--cloud-role-routing) |
| **Auto-Tune wizard** | Answer four questions about your hardware and codebase, preview the budgets, then apply them | [QUICKSTART.md](QUICKSTART.md#part-a-the-gui) |
| **Per-profile thinking** | Turn reasoning on or off for each model profile, from the GUI or `disable_thinking` | [MODEL_GUIDE.md](MODEL_GUIDE.md#thinking-per-profile) |
| **Model benchmarks** | Six small projects with a hidden holdout rubric, scored per profile, including a parametric SVG drawing test whose result is shown inline, so you can compare models on your own hardware | [MODEL_GUIDE.md](MODEL_GUIDE.md#compare-models-with-the-diagnostic-benchmarks) |
| **Prompt architect** | `✦ Refine Prompt` turns a one-line idea into a repo-aware spec with locked file targets and interface signatures. `↶ Undo Refine` restores your text | [TECHNIQUES.md](TECHNIQUES.md#17-the-promptspec-architect) |
| **Interface contracts** | Auto locks the project's public signatures into every step, and checks cross-file imports and call arity before the tests run | [TECHNIQUES.md](TECHNIQUES.md#18-contract-first-interface-locking--cross-file-verification) |
| **Static gate** | A fast undefined-name and syntax check (`ruff` F821 and E999, with a pure-Python fallback) runs before any test subprocess | [TECHNIQUES.md](TECHNIQUES.md#19-the-static-analysis-gate-a-linter-shaped-repair-trigger-for-free) |
| **Green ratchet** | Auto snapshots the project before each step and rolls a step that still fails back to the last passing state | [TECHNIQUES.md](TECHNIQUES.md#20-green-ratchet-step-checkpoints-for-the-autonomous-loop) |
| **Permission broker** | Web access, package installs, unapproved shell commands and sensitive file reads pause the run for an Allow Once, Allow for This Run, Always Allow, or Deny & Redirect decision | [Safety and trust](#safety-and-trust) |
| **Safe writes** | Dry run by default, side-by-side review, backups before every write, `pedacito restore` to undo | [Review page](#review-page) |
| **Live test introspection** | During investigation the model can run one scoped `pytest` check and read the live traceback before it writes a fix. On by default (`enable_run_test_in_gather`) | [Scoped test runs](#scoped-test-runs-during-investigation) |
| **Frontend contract checks** | Before tests run, local `<script src>` and `<link href>` files must exist, and literal `getElementById` and `querySelector('#id')` lookups must match the HTML ids | [Frontend contract checks](#frontend-contract-checks) |
| **Repair lessons** | A repair that fixes a failed check leaves a deduplicated, repo-specific lesson in `.pedacito/micro_facts.json`; a lesson a directory proves twice is promoted into an auto-managed path-scoped rule (`.pedacito/rules/auto_*.md`) that only shows up when the model reads that directory | [Repair lessons](#repair-lessons-and-repo-gotchas) |
| **Symbol-level edits and rename** | Replace a whole function by name without echoing its old text, or rename a symbol across the project with no model call (`pedacito rename`) | [Editing](#editing-searchreplace-not-diffs) |
| **Engine-enforced JSON and scout routing** | Lookup steps are constrained by the server's JSON schema support, and `scout_profile` sends the lookups to a fast model while the big model plans and edits | [Model guide](MODEL_GUIDE.md#dual-gpu--dual-model-scout--architect-routing-scout_profile) |

### The GUI workspace

`pedacito gui` serves a local page on `127.0.0.1` with no CDN and no extra
dependencies, so it works offline. It runs the same engine as the commands.

- A two-row toolbar with the project path, a live **profile and model** picker,
  the server's status light, and session totals for tokens, cache hits, and cost.
- **Nine color schemes** under the theme menu: GitHub Light and Dark, Daylight,
  Nord, Midnight, and four 1990s themes (Jazz Cup, Trapper Keeper, Memphis
  Arcade, Windbreaker).
- **Five modes** next to the chat box: **Task/Edit** for changes, **Ask** for
  read-only questions, **Auto** for goal-to-checklist runs, **New** for starting
  a project, and **How ?** for the built-in reference.
- **✦ Refine Prompt** above the composer turns a one-line idea into a full spec
  in place: objective, files to create or modify, interface contracts,
  requirements and edge cases, and verification. **↶ Undo Refine** puts your
  original text back. Drag the composer's top edge to make it taller while you
  read the spec.
- **Changes** show proposed edits as old and new code side by side, with
  per-file accept, reject, or hand edits.
- **Permission requested** dialogs pause a run when the model asks for web
  access, a package install, an unapproved shell command, or a sensitive file.
  See [Safety and trust](#safety-and-trust).
- **Settings** edits your profiles, every setting, the Auto-Tune wizard, and the
  raw config file, without deleting your comments.
- **Benchmarks** runs the six diagnostic projects against any profile. The
  pelican SVG result is drawn inline in its scorecard.
- **Project ▾** holds project actions: **Rename symbol…** renames an identifier
  across the project with no model call, and shows the per-file diffs before it
  writes. **Restore from a backup…**, **Check model (loads it)** and **Clear
  terminal** are also here. A `Scout: <profile>` badge appears beside the model
  chip when a scout profile is active.

Chat threads are saved with timestamps and can be renamed. The **Files** panel
lists the indexed project files, and **Backups** lists snapshots you can
restore.

### The CLI at a glance

The package installs `pedacito` and a shorter alias, `pdc`. Both accept the same
commands. Run `pedacito <command> --help` for every flag.

| Command | What it does |
|---|---|
| `pedacito gui [PROJECT]` | Open the browser control panel |
| `pedacito models scan` | Find local servers, pick models, write profiles, and measure them |
| `pedacito config --check` | Test the connection and generation. `--doctor` also probes structured output. `--setup` runs first-time setup |
| `pedacito calibrate` | Measure the active model and save a model card that tunes defaults |
| `pedacito profiles [--recommend]` | List profiles with their scores, and suggest role routing |
| `pedacito ask PROJ -t "..."` | Run one task. Dry run by default. Add `--review`, `--apply`, `--plan`, `--refine`, or `--jsonl` |
| `pedacito explain PROJ -q "..."` | Answer a question about the code. Never edits |
| `pedacito auto PROJ -t "..."` | Turn a goal into a checklist, then implement it task by task |
| `pedacito new DIR -t "..."` | Start a project from scratch |
| `pedacito chat PROJ` | Interactive session that keeps the index warm (`--rpc` turns the session into a machine protocol) |
| `pedacito rpc [PROJ]` | JSONL-RPC over stdin/stdout for editors and other agents: one request per line, streamed events, steering, permission prompts, cancel -- index, profiles and session warm the whole time |
| `pedacito how "..."` | Built-in help for the CLI, the GUI, and settings |
| `pedacito restore PROJ` | Undo the last write |
| `pedacito rename OLD NEW` | Multi-file symbol rename through the AST and the tokenizer, no model call. `--file` scopes it, `--apply` writes (default: diff preview) |
| `pedacito index`, `map`, `files`, `cost` | Build and inspect the index. These need no model call, except `index` summaries |
| `pedacito eval` | Score the harness on fixture repositories. Use `--baseline` as a regression gate |
| `pedacito memory`, `skills`, `prune`, `reset`, `trust`, `worktree`, `serve`, `procs`, `lsp`, `mcp`, `diagnose`, `tests`, `webcheck`, `domcheck` | Upkeep and specialist tools. `mcp list` shows the named MCP servers' tools and trust flags |

Benchmarks are in the GUI only, through the **Benchmarks** button.

### Documentation map

- **[QUICKSTART.md](QUICKSTART.md)** is the step-by-step guide for GUI users and
  CLI users.
- **[MODEL_GUIDE.md](MODEL_GUIDE.md)** covers choosing and tuning models by
  hardware tier, thinking settings, calibration, and benchmarks.
- **[TECHNIQUES.md](TECHNIQUES.md)** explains how Pedacito gets small models to
  work well, and which popular techniques it deliberately leaves out.
- The rest of this README is the full reference: install, configuration, every
  mode and setting, safety, and known limits.

---

## Install

Requires Python 3.11+.

```bash
cd /path/to/pedacito
pip install -e .            # Windows: py -m pip install -e .
```

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

Uninstall with `pip uninstall pedacito-local`. Your config and every project's
`.pedacito/` folder are left untouched.

---

## Model server setup

Start any server that speaks the OpenAI chat-completions API, then give
Pedacito its base URL and a model name.

| Server | Start it | Base URL |
|---|---|---|
| **LM Studio** | Load a model, then start the server (Developer tab). For another machine, enable "Serve on Local Network". | `http://localhost:1234/v1` |
| **Ollama** | `ollama serve` and `ollama pull <model>`. For another machine, set `OLLAMA_HOST=0.0.0.0` first. | `http://localhost:11434/v1` |
| **Lemonade** | Start the server and load a model. | The URL it prints (ends in `/v1` or `/api/v1`) |
| **Hosted API** | Get an API key. | e.g. `https://api.deepseek.com/v1` |

The default ports above are hints, not rules: any port works, and the scan
identifies a server by its replies, so two servers can coexist on one host
(for example Lemonade on 11434 next to LM Studio on 1234).

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
behaves is in [MODEL_GUIDE.md](MODEL_GUIDE.md#engine-enforced-json-by-server-enforce_json_schema).

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

---

## Configure

```bash
pedacito config --setup     # finds local servers, writes a profile per model
pedacito models scan        # re-run later to add models or servers
pedacito profiles           # see scores and role recommendations
pedacito config --init      # or write a commented starter file
pedacito config             # show every resolved setting and where it came from
pedacito config --reference # print the full annotated template
```

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
| `api.anthropic.com` (OpenAI-compatible) | Not supported on this surface | `json_mode = "schema"`, omits `temperature`, `tool_mode = "grammar"` |
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

---

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

The GUI shows the same list at `GET /api/mcp/tools` and settings expose the
acceleration and bridge switches. See
[Techniques §30](TECHNIQUES.md#30-pruned-permission-gated-mcp-bridges-named-servers--mcp_call)
for why the pruning is shaped this way.

---

## Built-in help: `pedacito how`

For model selection, VRAM sizing, thinking settings, and role routing, see the
[model guide](MODEL_GUIDE.md). It has concrete starting points for under 8 GB,
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

---

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
never stops the run. See [TECHNIQUES.md §17](TECHNIQUES.md#17-the-promptspec-architect).

Other useful flags: `--steps N` (lookup limit), `--strict` (exit 1 when a task
applied edits but left a failed edit block or impacted tests still failing
after the repair rounds), `--model`, `--url`, `--profile`,
`--role planner=big` (route one job to another profile for this run),
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
drive Pedacito. Writing runs (`--apply`) take a per-project lock
(`.pedacito/run.lock`); a second writer is refused while the first is alive.

**Steering and questions.** In `chat`, anything you type while the model is
working is delivered at its next step ("stop reading tests, it's in
`sync.py`") without disturbing the cached prompt prefix (`allow_steering`). In
an interactive `ask`/`chat` the model also gets a `question` action for genuine
ambiguity, at most twice per task; in `auto` and with `--jsonl` it never asks
and states its assumption instead.

Every write snapshots the originals first; `pedacito restore PROJ` undoes the
last one.

### `explain`

Answers questions about the indexed project in prose, with no edits. It never
builds an index; run `pedacito index` first.

```bash
pedacito explain PROJ -q "trace the flow from answer to schedule"
pedacito explain PROJ                   # interactive
```

---

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
| `run_test` | Run one scoped `pytest` check (a test file or node id) and read the condensed traceback. On by default; see [Scoped test runs](#scoped-test-runs-during-investigation) |
| `done` | Stop gathering and answer. |

Repeating a read-only lookup (even many steps later) that nothing has changed
since gets a pointer to the earlier result instead of a second copy, and any
non-read observation (a broad `grep`, a chatty command, an MCP reply) is capped
at a fraction of the window (`max_observation_tokens`) with a note on how to
narrow it.

The action list is **filtered to what is enabled and to the model's size**: a
tiny profile sees ten actions and a ~430-token prompt, not two dozen tools and
browser-eval advice it cannot use.

Opt-in actions, each behind a config flag because they widen what a model can
do:

| Action | Enable with | Purpose |
|---|---|---|
| `write` / `append` / `move` / `delete` | `allow_write = true` | Whole-file operations; `append` builds a file larger than one reply. |
| `rename` | `allow_write = true` | Mechanically rename one definition and its references in the files that import it (strings and comments untouched, every file re-parsed, snapshotted). Refuses a name defined in several places and lists the references it left alone. |
| `undo` | `allow_write = true` (+ `backup`, on by default) | Self-recovery mid-task: reverts what this task's `write`/`append`/`move`/`delete` actions did so far (or just one of their files), pre-task snapshots come back, files the task created go away. |
| `run` | `allow_shell = true` | Run one command (e.g. `pytest -q`), without a shell, so `&` and `nohup` are rejected and long-lived processes go to `serve`. A quiet GUI/server is stopped after `shell_idle_seconds`. |
| `serve` | `allow_serve = true` | Start, list, log, or stop a background dev server. The registry and logs live under `.pedacito/procs/`; also `pedacito serve` / `pedacito procs`. |
| `diagnose` | `allow_shell = true` | Run the detected linter/type checker (`ruff`, `pyflakes`, `flake8`, `mypy`/`pyright` when configured, `npm lint`, `tsc`, `cargo check`, `go vet`, `rubocop`). Also available as `pedacito diagnose`. |
| `lsp` | `allow_lsp = true` | Exact definition/references/hover/diagnostics through pyright/pylsp/typescript-language-server/rust-analyzer/gopls, or commands in `lsp_servers`. Also `pedacito lsp`. |
| `screenshot` | `allow_screenshots = true` | Capture the screen as text: size, brightness, blank-screen verdict, OCR, optional `vision_model` description. Needs the `vision` extra. |
| `browser` | `allow_browser = true` | Headless browser: start a server command, open the page, click/type/hold keys, and report page text, canvas sizes, and console errors. Needs the `browser` extra. |
| `mcp` | `allow_mcp = true` + `mcp_servers = [...]` | Call tools, prompts (`prompt:server/name`), or resources (`resource:server/uri`) from MCP servers. The catalog shows arguments, calls are validated, one live session per server is kept warm, and a server that was down at startup is retried. |
| `mcp_call` | `enable_mcp = true` + `[mcp_servers.<name>]` | Call a tool on a *named* MCP server. Only the up-to-four tools whose names and descriptions match the task are ever shown, the action itself appears only when one matches, and untrusted servers ask the permission broker per call. See [Named MCP servers](#named-mcp-servers). |
| `task` | `allow_subagents = true` | Hand a bounded, read-only research question to a helper agent with a fresh context window; only a short digest comes back. Depth is capped (1 local, 2 hosted). |
| `fetch` | `allow_fetch = true` + `fetch_allowed_hosts` | URL → readable text, host-allowlisted, size-capped, cached, labelled untrusted. Private/loopback addresses are refused, including public names that *resolve* to them; the connection is pinned to the checked IP. |
| semantic search | `semantic_search = true` | A `grep` with no hits falls back to the server's `/v1/embeddings` endpoint. |

`run` and the two MCP actions execute model-chosen commands in your project.
Treat them as a
trust decision, but see [Safety](#safety-and-trust): `run` is sandboxed by
default where the OS allows, and the rest of the blast radius is limited too.

### Scoped test runs during investigation

A model that only reads code has to guess what a fix will do. `run_test` lets it
check the guess first: it runs one scoped `pytest` check during investigation, so
the live traceback is in front of the model before it writes the patch.

```json
{"action": "run_test", "target": "tests/test_app.py::test_save", "filter": "save or retry"}
```

- **Scoped.** `target` is a test file or a node id. `filter` is an optional `-k`
  expression. The model never supplies a shell command line.
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
`.pedacito/screenshots/`.

### Memory, rules, and skills

Pedacito can remember what earlier runs learned, as plain markdown you can
edit or delete:

- **Memory** (`.pedacito/memory.md`, plus a user-level `memory.md` next to your
  config) holds short facts under Commands / Conventions / Landmines / Notes.
  The model adds one with `remember`; you can too:
  `pedacito memory PROJ --add "command: run tests with pytest -x"`,
  `--list`, `--forget TEXT`. Lines are de-duplicated, capped per category,
  secrets are masked, and anything that reads like an instruction to an
  assistant is refused. It is shown to the model as *hints, not truth*.
- **Rules** (`.pedacito/rules/*.md` with a `paths: src/api/*.py` frontmatter
  line, and nested `AGENTS.md`/`CLAUDE.md` files) are shown only when the model
  reads a matching file. Pedacito also auto-manages `auto_*.md` rule files here
  (repair lessons promoted from `micro_facts.json`; see
  [Repair lessons](#repair-lessons-and-repo-gotchas)) -- anything without the
  `auto_generated: true` frontmatter flag is never written or deleted by it.
- **Skills** are named procedures (`.pedacito/skills/NAME/SKILL.md`, or
  `skills/NAME/SKILL.md` next to your config). The model sees a one-line
  catalog and loads a skill's text only when it needs it. Five starters ship
  built in; `pedacito skills` lists them. Skills are prompt text, nothing in a
  skill folder is ever executed.

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
[micro-facts](TECHNIQUES.md#14-repository-micro-facts-with-contradiction-retirement)).
It is plain JSON. Read it, edit it, or delete it to forget everything it has
learned. `micro_facts = false` turns recording and injection off, and
`max_micro_fact_tokens` caps how much of it reaches the prompt.

### Project instructions

Pedacito reads one conventions file into every prompt: the first found of
`PEDACITO.md`, `AGENTS.md`, `CLAUDE.md`, `.pedacito/instructions.md`, or
`.pedacito/PEDACITO.md` (the older spelling `PEDACEITO.md` is also accepted).
Put build/test commands, style rules, and off-limits directories there. It is
capped by `max_instruction_chars` (2000) and disabled with
`use_project_instructions = false`.

---

## More commands

### `chat`: interactive session

```bash
pedacito chat ~/code/anki [--apply | --review]
```

Keeps the index warm between tasks. Slash commands: `/profile NAME`,
`/model ID`, `/apply on|off`, `/steps N`, `/status` (pack, answer mode,
verifier, window, model card), `/cost` (tokens, cache hits and spend by call
site), `/memory`, `/skills`, `/profiles`, `/help`, `/quit`. **Ctrl-C cancels the
running task but keeps the session open.** Switching profile mid-session is the
cheapest way to compare two models on one task; a failed switch leaves the
session on the previous model.

With `--rpc` the same warm session speaks a machine protocol instead of prose:
stdin takes one JSON request per line and stdout answers with JSONL events --
see `pedacito rpc` below.

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
process warm; `switch_profile` and `status` work mid-session too. The index,
the profiles and the session ledger stay warm across requests -- that is the
whole point. Human-facing diagnostics go to stderr; stdout stays pure JSONL.
Protocol details in `pedacito/rpc.py`, rationale in
[Techniques §31](TECHNIQUES.md#31-bidirectional-stdio-jsonl-rpc-a-warm-engine-an-editor-can-drive).

### `new`: start a project from scratch

```bash
pedacito new my-app -t "a CLI todo manager with a sqlite backend" --apply
pedacito new my-api --spec docs/spec.md --review
pedacito new my-app -t "..." --plan-only --apply     # skeleton only
```

The model plans a file tree, the skeleton is written, then each file is filled
in. Dry run by default. The target directory must be empty or new, or hold
nothing but Pedacito's own files (`.git`, `.pedacito/`, `.pedacito.toml`,
`.pedacitoignore`), unless `--force`. `--complex` hands implementation to the
`auto` loop.

Each file is written by its own call, so the finished tree is checked as a
whole before the build reports success: a browser scaffold whose scripts cannot
link, or whose `getElementById` names an id no template renders, gets repair
rounds and then prints `BUILD INCOMPLETE` and exits 1. A web project's generated
`README` also says how to actually start it -- ES modules are refused over
`file://`, so it names `python -m http.server 8000` and the URL to open instead
of a `python main.py` line for a file the project does not contain.

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
```

The plan is a markdown checklist (`PLAN.md` by default), so a run is resumable
and editable; the file also keeps the full goal (a `Goal:` line with `> `
continuation lines), which a resumed run hands back to each task. There is
no default cap on tasks, steps, or time; add one with `--max-tasks N`,
`--max-minutes N`, or `--steps N`. `--repair N` sets repair
rounds per failed verification, `--keep-going` continues past failed tasks, and
`--auto-install` installs a missing Python package instead of reporting it.
`--verify-final` runs the `--verify` command once, after the last unchecked
task, instead of after every task: a whole-goal check cannot pass mid-plan, so
the per-task runs would roll back valid partial fixes (`auto_verify_final`,
off by default). After the last task, `--apply` also exercises the project
once (`auto_smoke`, on by default): the Python entry points are exec'd and
the receipt prints `verify: passed` -- that proves the project BOOTS, not
that the suite is green, so a plan should end with a run-the-tests task or
you should pass `--verify "pytest -q"`. `--strict` exits 1 when a task applied edits but left a
failed edit block or impacted tests still failing after the repair rounds,
for CI and unattended pipelines (without it both are reported and the run
exits 0 -- "reported" means the red verdict is named where you read the
result: an `ask` closes with a `[verification: RED ...]` note listing the
blockers the repair did not clear, and the `auto` finish line counts the
tasks that ticked done while the last gate round was red, the way it
counts partial applications; TECHNIQUES.md §33).

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
  [TECHNIQUES.md §18–§20](TECHNIQUES.md#18-contract-first-interface-locking--cross-file-verification).

`--worktree NAME` runs in `.pedacito/worktrees/NAME` on a `pedacito/NAME`
branch, leaving your checkout untouched. Manage worktrees directly with
`pedacito worktree create|list|remove NAME`.

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
| **Chat** (left) | Five modes next to the box: **Task/Edit** (edit code; *review* or *write directly*, planning auto/always/never, max lookups), **Ask** (read-only questions, answer streams in), **Auto** (goal → plan → implement; plan-only or implement, verify command, repair rounds, task/minute caps, worktree, keep-going, pip-install; applying Auto accepts the tool trust boundary and enables write/run tools for the run), **New** (start a project from scratch in a folder: preview, build, build-then-review, skeleton only, and *complex* for the autonomous loop), and **How ?** (the built-in CLI/GUI/setting reference with topic chips, copyable commands and *Open in Settings* links; answers stay out of the saved coding thread). **Attach** (or drag in) a text file as the task, question, or spec. While a task runs, type to **steer** the model, answer its clarifying questions inline, or **Stop** it. |
| **Changes** (middle) | The proposed edits as *Old code | Updated code* side by side with word-level highlights. Accept or reject the whole set, untick files to accept only some, or **Edit updated** to hand-edit a file before accepting. Files carry clause badges (`R1`, `R2`, ...) showing which part of the task they serve; a partial accept flags clauses left unaccepted. Backups are made first and a file that changed on disk since the preview is refused. |
| **Activity** and **Plan** (right, top) | Live phase and activity for the running task. The `auto`/plan checklist with progress; click **Edit** to change it, then **Run** to implement it. |
| **Files**, **Backups**, **Logs** (right, bottom tabs) | Files: a filterable tree of the indexed files, with files in the pending review highlighted. Backups: restorable snapshots. Logs: the terminal, everything pedacito prints, live, with ANSI colours. |
| **File preview** | Clicking a file in the Files tab opens a read-only preview modal. |
| **Settings** | See below. |

Panels resize (drag the dividers), the layout and colour scheme are
remembered, and on a narrow window the panels become tabs.

**Settings** edits the user config or the open project's `.pedacito.toml`
without touching the rest of the file, comments survive, every edit is checked
with the same rules the loader uses before it is written, and the previous file
is kept as `config.toml.bak`. Tabs: **Profiles** (create, duplicate, delete,
make default; server URL, key, model with a *Test connection* that lists the
server's models, tier, VRAM, JSON mode, and the common switches), **Model
discovery**, **Auto-Tune**, **All settings** (every setting, searchable, with its help text and where its value
comes from, applied globally or to one profile), and **Raw file / reset**. With no
config at all, a first-run form (with *Find servers* for LM Studio/Ollama) builds
one. A project config's execution settings stay ignored until you trust it
(*Review & trust…*, the same as `pedacito trust`).

How it stays safe: it refuses requests with a foreign `Host`/`Origin`, accepts
only JSON `POST`s, sends a strict Content-Security-Policy, and serves a project's
files only from inside that project. API keys are never shown in the profile
forms (leave the box blank to keep one); the **Raw file** tab does show the file
as written, key included. Only one job runs at a time, and jobs that write directly take the
same project lock as the command line.

### Measurement and upkeep

```bash
pedacito calibrate [--quick] [--repeats N]   # measure the model; save a model card
pedacito eval [--task ID] [--out report.json]   # score the harness on fixtures
pedacito map PROJ -t "fix X"       # see the map a task would get (ranked)
pedacito tests PROJ --files src/a.py   # detected test command + impacted tests
pedacito webcheck PROJ             # static contract checks for web files
pedacito domcheck PROJ             # headless DOM smoke check (needs Playwright)
pedacito memory PROJ [--add T | --forget T]
pedacito skills PROJ
pedacito prune PROJ --keep-sessions 50 --keep-backups 20 [--dry-run]
pedacito reset PROJ [--dry-run] [--keep-memory] [--keep-config] [--reindex]
pedacito trust PROJ                # approve a project's .pedacito.toml
pedacito ask PROJ -t "..." --record          # save the model's responses
pedacito ask PROJ -t "..." --replay FILE     # re-run the harness, no model needed
pedacito ask PROJ -t "..." --replay FILE --strict-replay  # fail on prompt drift
pedacito ask PROJ -t "..." --continue        # seed the task from the newest ask checkpoint
pedacito eval --out new.json --baseline base.json         # CI A/B regression gate
pedacito eval --repeats 3 --seed 1 --out new.json   # 3 seeded runs: pass counts, mean ± spread
pedacito new app -t "..." --design           # interfaces/data/tests before code
pedacito serve --root PROJ --name web -- npm run dev
pedacito procs --root PROJ [logs|stop NAME]
pedacito lsp --root PROJ --file src/a.py --line 12 --op definition
```

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
consent. See TECHNIQUES.md #45.

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

---

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

---

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
identifier is rewritten, and strings and comments are left alone.

```bash
pedacito rename apply_discount discounted_price                       # preview the diffs
pedacito rename apply_discount discounted_price --file pkg/core.py    # that file and its importers
pedacito rename apply_discount discounted_price --apply               # write it
```

A qualified name narrows the rename to one class: `pedacito rename Store.save save_record`
renames `Store.save` and the call sites whose receiver is provably a `Store` (Python),
and lists the look-alikes it left alone. The preview is the default, so no flag is needed to get one. `--apply` writes the
change and snapshots the originals first, so `pedacito restore` undoes it.

- **Identifiers only.** Python names move through `tokenize`. JavaScript,
  TypeScript, Go and Rust move through their tree-sitter identifier nodes. A
  string or comment that merely contains the old name is never touched.
- **All or nothing.** Every changed file must pass the syntax check before any
  file is written. If one would break, nothing is written.
- **Name-based, on purpose.** A parameter in an unrelated function that shares the
  name is renamed too. The preview shows it, so you can judge before writing.
- **Also available to the model.** With `allow_write = true`, the gather loop has a
  `rename_symbol` action that calls the same engine, and it goes through the same
  permission and snapshot rules as other writes.

In the GUI, **Project ▾ → Rename symbol…** takes the old name, the new name and an
optional scope file. **Preview** shows the diffs, and **Rename** writes them.

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
off. The syntax, static, frontend and cross-file contract gates read files in
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
  CDN bundles and classic-script globals are never reported.

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
`--snapshot`/`--baseline` diff two runs. Unlike `webcheck`'s notice-level regex
findings, this tool's errors can drive repair. See TECHNIQUES.md #45.

## How context is managed

A small window is spent three ways: the prompt's fixed text, the map, and the
conversation. Pedacito spends each on purpose.

- **Prompt packs.** The instructions come in `full`, `small` and `tiny`
  density (`prompt_pack`, else the model card, else the tier) and mention only
  the actions that are enabled. With browser, shell and write off, the default
  lookup prompt is about 1,700 tokens instead of 3,000; `tiny` is about 430.
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

## Safety and trust

Pedacito is local-first and sends nothing anywhere except to the model server
you configured. Within that:

- **Tool output is data, never instructions.** The system prompt says so;
  output from `run`, `browser`, `fetch`, `mcp`, `git` and `screenshot` is fenced
  as untrusted, control characters are stripped, and instruction-like wording
  ("ignore previous instructions...") is flagged in the observation.
  `pedacito eval` plants injections to measure this.
- **Commands get a minimal environment.** A command the model chooses (and dev
  servers it starts) sees `PATH`, `HOME`, locale and toolchain variables, not
  your API keys or tokens. Add what a project needs with `env_passthrough`
  (`["DATABASE_URL", "AWS_*"]`); `scrub_env = false` restores the old behaviour.
- **Protected paths.** The model can never write `.git/`, `.pedacito/`,
  `.pedacito.toml`, the autonomous loop's plan file (only the loop ticks its
  boxes), or anything in `protected_paths`. It will not edit a file that has
  unresolved git merge conflicts. `shell_deny = ["git push*", "rm -rf *"]`
  refuses matching commands even when the executable is allowed.
- **OS sandbox (on by default).** `run` and the tests an edit triggers run under
  an unprivileged OS sandbox: they can read the filesystem but write only to the
  project and a private temp directory; `.git/` (hooks!), `.pedacito/` and the
  project config are read-only; `~/.ssh`, `~/.aws`, `~/.config/pedacito` and
  similar are hidden; and the **network is off**. Backends: bubblewrap on Linux
  (`bwrap`), `sandbox-exec` on macOS; Windows has none that works without admin
  rights. Where no backend works (not installed, user namespaces disabled,
  Windows) commands run unsandboxed after ONE notice, so Pedacito behaves the
  same everywhere. Opt out with `sandbox = false`; allow the network with
  `sandbox_network = true` (needed for tests that talk to a server or install
  packages); add writable directories with `sandbox_writable = ["~/.cache/pip"]`;
  `sandbox_strict = true` refuses to run a command when no sandbox can be
  applied. A failed command's observation says it ran sandboxed, so the model
  does not chase a blocked write as a code bug. `pedacito config --doctor` and
  `/status` show whether it is active. Your own hooks and `auto_verify` are
  never sandboxed.
- **Browser `eval`** runs model-written JavaScript, so it is limited to local
  pages (`browser_eval_hosts`).
- **Secrets stay out.** Secret-bearing files are not indexed (above); secret-
  looking values are masked in session files, logs and events
  (`redact_secrets`); a task that appears to contain one warns before going to
  a hosted API.
- **A repository cannot run commands behind your back.** A project's own
  `.pedacito.toml` may not set `hooks`, `mcp_servers`, `auto_verify`, `test_command`,
  `allow_serve`, `allow_lsp`, `lsp_servers`, any sandbox setting, or the capability
  toggles that unlock those tools (`allow_shell`, `allow_write`, `allow_browser`,
  `allow_screenshots`) until you read it
  and run `pedacito trust`; editing the file revokes the approval. Harmless
  settings from the same file still apply, and the toggles stay honoured from
  your user config, from `--config FILE` and from `PEDACITO_*`.
- **Hooks** (`hooks = { post_edit = ["ruff format {file}"], pre_edit = [...] }`)
  run your own program at `session_start`, `pre_task`/`post_task`,
  `pre_tool`/`post_tool`, `pre_run`/`post_run`, `pre_edit`/`post_edit`,
  `post_compact`, `notification`: a JSON event on stdin, an optional JSON
  decision on stdout, exit status 2 to deny a `pre_*` event. No shell, a
  timeout, the scrubbed environment; a failing hook is a warning.

- **Workspace containment.** Every path a tool touches is resolved with its
  symlinks and must stay inside the project. A `..` component, an absolute path
  outside the project, or a symlink that leads out is refused, so a link inside
  your repository cannot reach `~/.ssh` or another project.
- **Credential guard.** `.env` files (except `.env.example`-style templates),
  private keys, SSH and TLS material, `.npmrc`, `.pypirc`, `.netrc` and
  `credentials.json` are never indexed. The model can read one only after you
  approve that read in the permission dialog.
- **Localhost CSRF checks.** The GUI server refuses a state-changing request whose
  `Origin` or `Referer` is not a loopback address. A web page open in the same
  browser cannot drive it, even if it guesses the port.
- **Web access is off by default.** With `enable_web = false`, the built-in
  `fetch_url` and `web_search` tools are offered only when a task names a URL or
  web words, and each call asks first in the permission dialog. Set
  `enable_web = true` to allow them without asking. Loopback, link-local
  (including cloud metadata) and private-network addresses are refused unless you
  list them in `fetch_allowed_hosts`. The check also runs on the *resolved*
  addresses: a public name that resolves to a private IP (DNS rebinding) is
  refused, and the connection is pinned to the validated address so a second DNS
  answer cannot swap it mid-request. Redirects are re-checked at every hop, and a
  wildcard allow-list entry never exempts a private range.
- **Permission requests.** Some actions pause the run until you decide. The
  **Permission requested** dialog shows the category, the action, the reason and
  the exact target, with four buttons:
  - **Allow Once** approves this request only.
  - **Allow for This Run** approves that kind of request until the run ends.
  - **Always Allow** approves it from now on. For web access it sets
    `enable_web`, and for shell commands it sets `allow_shell`. The choice is
    saved to your config when one exists.
  - **Deny & Redirect** refuses the request, and the model is told to continue
    with local project files only.

  A request you do not answer (the dialog times out, or you press Stop) is
  denied. The CLI has no dialog, so there a gated tool reports that it is
  disabled in the config.

`browser` and MCP servers still execute unsandboxed on your machine (a server is
a program you chose to run). The sandbox contains `run` and test commands; it is
defence in depth, not a promise against a determined adversary or a kernel bug.

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
[MODEL_GUIDE.md](MODEL_GUIDE.md#dual-gpu--dual-model-scout--architect-routing-scout_profile).

A task that ends without a usable edit is also remembered in
`.pedacito/attempts.json` (files looked at, why); a similar later task is told
what was already tried. A success clears it. The same runs are counted per
profile in `.pedacito/profile_stats.json`, which feeds the escalation
statistic above.

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

## Undo and workspace layout

```bash
pedacito restore PROJ --list
pedacito restore PROJ --dry-run
pedacito restore PROJ                        # newest snapshot, byte-for-byte
pedacito restore PROJ --snapshot 20260830-020724
pedacito restore PROJ --delete-created       # also remove files that run created
```

Files created after a snapshot are left in place unless you pass
`--delete-created` (the run's created files are listed in a sidecar
`backups/<stamp>.created`; directories left empty are removed too). The model
can trigger the same rewind mid-task: the gather `undo` action (`allow_write`,
backups on) reverts everything its own writes did in the current task, so it
recovers from a compounding bad edit instead of ending the run on one.
Everything Pedacito writes for a project lives under `<project>/.pedacito/`:

```
index.json            the map (versioned, written atomically)
summaries.json        summary cache, keyed by content hash
embeddings.json       semantic-search cache (if enabled)
backups/<stamp>/      pre-write snapshots (never merged: -2, -3 on a clash)
sessions/<stamp>.md   what was asked, looked up, and changed (secrets masked)
checkpoints/          ask/chat conversation checkpoints (JSON)
events/<stamp>.jsonl  one JSON line per step, call, fold, verify, task
reviews/<stamp>.html  saved review pages
screenshots/          captures (if enabled)
procs/                background process records and logs
design.json/.md       spec-driven design (interfaces, data, tests)
memory.md             remembered facts (hints for the model)
micro_facts.json      commands confirmed to work here, and repair lessons
rules/*.md            path-scoped rules       skills/*/SKILL.md   your skills
prompts/*.md          prompt overrides        fetch/              fetch cache
todo.md               current task checklist
log.txt               timestamped event trace
```

`sessions/` is the place to look when an edit turns out wrong days later;
`log.txt` shows where a failure happened. On first run Pedacito creates a
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

---

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

---

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
| `small_context_tokens` / `max_whole_file_lines` | 16384 / 120 | A window at or below `small_context_tokens` counts as "small": the answer brief then demands chunked/anchored edits, and a whole-file rewrite of a file bigger than `max_whole_file_lines` is refused before it lands (unless the task explicitly asks to rewrite that file). `small_context_tokens = 0` turns the discipline off. |
| `k_cache_quant` / `v_cache_quant` | `""` | KV-cache quantization the runtime uses. |
| `json_mode` | `"schema"` | `"schema"` for local servers, `"object"` for hosted (auto-switched). |
| `enforce_json_schema` | true | Ask the engine to constrain gather steps with `response_format` json_schema (strict); servers that refuse are downgraded to json_object, then plain prompting, per model. |
| `enable_run_test_in_gather` | true | Let the gather loop run one scoped pytest check (a file or node id) to see live tracebacks. `false` makes each run ask the permission broker. |
| `frontend_contracts` | true | Before tests run, check local script/stylesheet links, literal DOM id lookups and changed JavaScript syntax. |
| `auto_heal_imports` | true | Before a static-gate finding reaches the model, insert the one import that provably resolves it (stdlib name, or a symbol exactly one local module exports), verified by the AST checker; ambiguous names are never guessed. |
| `collapse_repair_turns` | true | When a repair succeeds, drop the failed drafts from the conversation tail and keep a compact receipt, so the server's prompt-prefix cache is reused instead of re-prefilling the broken trajectory. |
| `micro_facts` / `max_micro_fact_tokens` | true / 150 | Record confirmed commands and repair lessons in `.pedacito/micro_facts.json` and inject them as hints; the token cap applies to the injected block. A lesson confirmed twice for one directory is promoted into an auto-managed path-scoped rule in `.pedacito/rules/auto_*.md` instead of sitting in every prompt. |
| `tier` | `""` | Budget preset: `tiny`/`small`/`medium`/`large`/`hosted`; guessed from the model name when unset. |
| `max_steps` | 0 | Lookup limit per task; 0 = none. |
| `prefetch_targets` | 0 | Read up to this many task-named files/symbols through the normal tool path before the first lookup step (and once more after gather when a named target is still unread). Prefetch reads don't count toward `max_steps`. |
| `temperature` | 0.2 | Higher values make tool selection worse. |
| `temperature_answer` / `temperature_repair` | 0 / unset | Sampling temperature for the answer and for the repair re-asks, which copy SEARCH text character for character. The answer defaults to `0` (greedy, so copied text is exact); the repair re-asks are unset (they use `temperature`). Gather keeps `temperature`; `best_of` candidates run at the answer temperature + 0.4. |
| `backup` | true | Snapshot originals before writing. |
| `log_sessions` | true | Write a session record per task. |
| `review` | false | Always use the review page. |
| `fuzzy_match` | false | Anchor near-miss search blocks approximately. |
| `edit_style` | unset | `quote` (default brief) or `anchor` (prefer `SYMBOL` + full definition, SEARCH only for small edits); unset lets the model card decide. |
| `post_edit_check` | true | After applied edits, feed back only NEW linter/syntax findings on the changed files. |
| `auto_allowlist` | false | Restrict `auto` task writes to files the title names or the lookup read (new files always allowed). |
| `allow_write` | false | Whole-file write/append/move/delete during gathering. From a project's own config: ignored until `pedacito trust`. |
| `allow_shell` | false | `run` and `diagnose`. From a project's own config: ignored until `pedacito trust`. |
| `allow_serve` / `serve_wait` | false / 3.0 | Background dev servers (`serve` action, `pedacito serve`); seconds to wait for a printed URL. |
| `allow_lsp` / `lsp_servers` / `lsp_timeout` | false / `{}` / 10 | Exact definition/references/hover/diagnostics via a language server; per-extension commands override PATH detection. |
| `allow_screenshots` / `allow_browser` | false | Screen and headless-browser capture. From a project's own config: ignored until `pedacito trust`. |
| `vision_model` | `""` | Multimodal model used to describe captures. |
| `semantic_search` | false | Embeddings fallback for `grep`. |
| `circuit_breaker` / `circuit_failures` / `circuit_cooldown` | true / 4 / 60 | Fail fast after repeated transport/server failures; rate limits, context overflows and 4xx errors do not trip it. |
| `allow_diff` | true | Read-only `diff` action. |
| `disable_thinking` | false | Ask for non-thinking mode using the model family's switch; output blocks are stripped regardless. |
| `default_profile` | `""` | Profile used when `--profile` isn't given. |
| `scout_profile` | `""` | Profile whose client runs gather's lookup steps while the active (architect) profile plans and edits; empty = no scout. |
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
| `hooks` | `{}` | Your commands at events (project-level hooks need `pedacito trust`). |
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

---

## Known limits

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
- Broad, structural tasks ("modernize this codebase") are a poor fit for small
  models. Narrow, localized tasks are where this works well.

---

## Contributing

Issues and pull requests are welcome: bug reports with a repro, support for
other model servers or languages, and fixes for anything that breaks outside the
setup this was built on. Please open an issue before a large PR.

## License

MIT, see [LICENSE](LICENSE).
