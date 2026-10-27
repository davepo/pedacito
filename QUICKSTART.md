# Pedacito Quick Start

Pedacito lets a small local LLM (9B–27B and similar) work on a codebase
bigger than its context window. It builds a searchable index of your
project, then lets the model retrieve only the relevant pieces of your
codebase to answer questions or propose edits — instead of requiring the
whole project to fit in context. See the [README](README.md) for the full
background, including why this exists and how it compares to tools like
Aider, Cline, or Claude Code.

This guide works whether you're running **LM Studio or Ollama**, on
**Windows or Linux** (or macOS). Commands are identical everywhere except
where a step is genuinely OS-specific, which is called out explicitly.

Every command below uses `pedacito`. Installing also puts `pdc` on your
`PATH` as a shorter alias for the exact same command — use whichever you
prefer, everywhere. If neither is found after installing, use
`python -m pedacito` (or `py -m pedacito` on Windows) instead — identical.

---

## Before you start: is your model server up?

Pedacito talks to any server exposing an OpenAI-compatible `/v1` API. LM Studio
and Ollama both qualify, and nothing else about Pedacito changes between them —
only how you start the server and which model name you use.

**LM Studio:** load a model, start the server (Developer tab → Status:
Running), and if Pedacito runs on a *different machine* than LM Studio, enable
**"Serve on Local Network"** in that same tab — without it the server only
listens on `127.0.0.1` and a remote machine gets refused. Default port
`1234`.

**Ollama:** run `ollama serve` (or let the background service handle it),
and `ollama pull <model>` for whatever you intend to use. Ollama listens on
`127.0.0.1` by default; if Pedacito runs on a different machine, set
`OLLAMA_HOST=0.0.0.0` (or the machine's address) in Ollama's environment
before starting it. Default port `11434`.

Either way, confirm from wherever Pedacito runs:

```
pedacito config --check
```

```
Connecting to http://box.tail1234.ts.net:1234/v1 ...
  reachable. Models available: qwen3.5-9b
  currently loaded: (none)
  'qwen3.5-9b' is not resident; it will be loaded on first use (up to 240s).
  'qwen3.5-9b' is listed. Testing that it actually generates...
  generation works. You're good.
```

Two distinct failures to tell apart. **Not reachable** is a network/firewall
or server-not-running problem — see the platform notes above. **Reachable
but the model isn't listed, or is listed but won't generate** means your
`model =` string doesn't match what the server has, or the server couldn't
load it (usually not enough free VRAM). Copy the exact name from what
`--check` prints.

Run this whenever something stops working. It's the fastest way to tell "the
server isn't right" from "Pedacito is broken."

---

## Step 1 — Install

Requires Python 3.11+ (Pedacito uses the standard-library `tomllib`).

**Windows (PowerShell):**

```powershell
cd C:\path\to\pedacito
py -m pip install -e .
```

**Linux / macOS:**

```bash
cd ~/pedacito
pip install -e .          # or: pipx install -e .
```

`-e` (editable) means the installed commands point straight at this folder,
so a later `git pull` or file replacement here takes effect immediately —
no reinstall needed. This installs `pedacito`, its shorthand `pdc`, and
symbol-level indexing support for JavaScript, TypeScript, and Ruby
(`tree-sitter`, prebuilt wheels, no compiler needed) right out of the box —
no separate step if your project uses those languages.

**Without installing**, from inside the folder: `pip install requests` then
`python -m pedacito ...` (`py -m pedacito ...` on Windows). Works, but only from
inside this folder, so installing is worth it.

If neither `pedacito` nor `pdc` is found after installing, pip may have put
it in a directory that isn't on your `PATH` (common on Windows with a
per-user install). Use `python -m pedacito` / `py -m pedacito` instead —
same result, no `PATH` needed.

---

## Step 2 — Configure

```bash
pedacito config --setup
```

An interactive wizard: it probes for a running LM Studio or Ollama server,
lists the models it finds, asks which to use (Enter accepts the
auto-detected one), optionally takes a hosted API key (DeepSeek and
friends), and writes a minimal config to `~/.config/pedacito/config.toml`
(Linux/macOS) or `%APPDATA%\pedacito\config.toml` (Windows). Three settings
are mandatory — `base_url`, `model`, and the server machine's GPU memory
(`vram_gb`) — and the wizard fills them in for you. Prefer doing it by hand?
`pedacito config --init` writes the same starter; `pedacito config
--reference` shows every knob with explanations. Hosted APIs
(DeepSeek/OpenAI) do not need the VRAM value.

Whatever path you chose, token budgets need no attention: pedacito asks
your server its real context window and the model's file size at startup,
then fits every limit to the VRAM actually available (see "Token budgets"
in the README). A DeepSeek-style hosted profile pointed at an 8k local model
is now safe — the numbers get squeezed to fit instead of the server
rejecting every request. If you still want per-model profiles, add
`tier = "small"|"medium"|"large"|"hosted"` to a profile and all ten budget
knobs come from that preset. On a local server VRAM caps the context window
so the KV cache does not spill; the tier token values are used when they fit
inside it, otherwise auto-fit uses the VRAM-derived allocation. A value you
write explicitly overrides both. If your runtime uses quantized K/V caches
(LM Studio's cache quantization, llama.cpp `--cache-type-k/-v`, Ollama
`OLLAMA_KV_CACHE_TYPE`), add `k_cache_quant = "q8_0"` and
`v_cache_quant = "q5_1"` (or whatever you loaded) so the VRAM estimate uses
the smaller cache. If the server reports its default window instead of your
configured one, set `context_tokens` explicitly in that profile.

Then:

```toml
[server]
base_url = "http://localhost:1234/v1"   # or :11434/v1 for Ollama
model = "qwen3.5-9b"            # exact id/name the server reports
vram_gb = 8                     # GPU VRAM on the server machine, in GB
```

Whatever you do by hand, the file only ever needs those three lines. The old
skeleton with four fully-spelled-out profile blocks is still available via
`pedacito config --reference`. If you run several models and want to switch
with `--profile NAME`, each profile can carry a one-word preset —
`tier = "small"|"medium"|"large"|"hosted"` — instead of ten budget knobs;
values you write explicitly still win. And remember budgets self-fit to the
server's real window and the GPU's VRAM at startup, so a wrong tier costs
efficiency, not correctness.

Next, verify:

```bash
pedacito config --check
```

Settings resolve in layers, each overriding the one before: built-in
defaults → the file above → a `.pedacito.toml` in a specific project (for
per-project overrides) → `--config FILE` → command-line flags. Run
`pedacito config` with no flags any time to see every resolved value and which
layer set it.

---

## Stuck? Ask Pedacito itself

```bash
pedacito how "how do I set up a config file?"
pedacito how "what command do I run to index a project?"
```

`pedacito how` is smart help: it answers from this guide, the README, and
Pedacito's source code, and gives you the exact command or config snippet to
use. Run it with no question for an interactive prompt. It does not need a
project or an index, and it uses your current config automatically — so you
can ask "what should I run with my setup?". To point it at a different
server for one session:

```bash
pedacito how --url http://localhost:11434/v1 --model qwen3:8b --vram 8 \
             "how do I make the index ignore a folder?"
```

`--api-key` is available too when the server needs one. `--server` and
`--vram-gb` are aliases for `--url` and `--vram`.

---

## Step 3 — See what would be indexed

```bash
pedacito files ~/code/anki
```

First run in a project creates its workspace:

```
  created ~/code/anki/.pedacitoignore
  created ~/code/anki/.gitignore with .pedacito/
  state for this project lives in ~/code/anki/.pedacito
  ref   data/intervals.csv
  code  src/render.py
  code  src/scheduler.py
  skip  deprecated/   [ignored directory, 1 file(s)]

2 code + 1 reference indexed, 1 skipped
```

Costs nothing — no model call. Read the list. If something is being indexed
that shouldn't be (an old version of a script, a vendored dependency, build
output), fix it in step 4 before spending time or tokens indexing it.

## Step 4 — Trim what gets indexed

Open the `.pedacitoignore` step 3 created and add lines, gitignore syntax:

```
deprecated/
*_old.py
```

Forward slashes on every OS, including Windows. Re-run `pedacito files` to
confirm. Trim generously — everything excluded here is map space freed for
the code you actually want the model looking at.

## Step 5 — Check the cost, then build the index

```bash
pedacito cost ~/code/anki
```

```
6 units across 4 files (3 code, 1 reference)
  eligible for a summary : 4
  skipped, has docstring : 4
  skipped, is a method   : 1
  reference (never summ.): 1
  LLM calls              : 0 units + 3 files = 3
  first index @ 25 tok/s : ~0 min (cached after)

map size, flat  : 173 tokens
map size, cards : 108 tokens   <- used above 4,000
```

Also free — no model call. On a real project this is where you find out the
first index is 20 minutes rather than 2. If that's unpleasant, go back to
step 4 and exclude more, or accept it: it only happens once, and after that
only genuinely changed code gets re-summarised.

```bash
pedacito index ~/code/anki
```

You never have to run this explicitly again — `ask`/`chat` build or refresh
the index automatically. `pedacito map ~/code/anki` needs no model at all and
is the fastest sanity check that the index looks right.

---

## Understanding the code without changing it

`pedacito explain` answers questions about the indexed project — how a
function works, what a file does, how data or control flows between parts —
without proposing any edits:

```bash
pedacito explain ~/code/anki -q "how does the scheduler pick the next card?"
pedacito explain ~/code/anki -q "trace the flow from answer to schedule"
pedacito explain ~/code/anki        # interactive prompt
```

It is always scoped to the project's index: it reads the relevant source
through the same lookup index `ask` uses, then answers in prose. It never
builds an index, so run `pedacito index ~/code/anki` first (or navigate to a
project you've already indexed) — otherwise it tells you exactly that.

---

## Step 6 — Ask for something, without changing anything

```bash
pedacito ask ~/code/anki -t "make the leech threshold configurable"
```

```
--- looking things up ---
  [map: flat, ~173 tokens, 3 files]
  step 1: read src/scheduler.py L11-21   (see the scheduler)
  step 2: done                           (have enough)
--- answering ---

The leech threshold was hard-coded. I moved it to a constructor argument...

[ok ] src/scheduler.py: 2 block(s) would apply
--- a/src/scheduler.py
+++ b/src/scheduler.py
@@ -11,11 +11,12 @@
-    def __init__(self, intervals):
+    def __init__(self, intervals, leech_threshold=8):
```

**No `--apply` or `--review` means nothing is written.** You get a diff and
decide. Stay in this mode for your first dozen tasks on a new project or a
new model — you're learning how reliably its edits anchor against your
actual code, which is far cheaper to learn from a printed diff than from a
half-mangled file.

## Step 7 — Review side by side, or apply directly

Two ways to actually write a change.

**Review page** — a local, offline HTML page: your file on the left,
proposed version on the right, changed words highlighted, long unchanged
stretches folded away.

```bash
pedacito ask ~/code/anki -t "make the leech threshold configurable" --review
```

Applying takes three deliberate actions: tick the attestation box, click
**Sign off & write**, then click again to confirm. Enter never triggers a
write. If the file changes on disk while the review sits open — you edited
it yourself in the meantime — the write is refused rather than clobbering
your change:

```
Not written: src/scheduler.py (changed on disk). Re-run the task to rebuild the review.
```

The page is saved under `.pedacito/reviews/` and served from `127.0.0.1`
only; it can write files, so it never listens on a network interface.
**Discard** closes it and changes nothing. To make review the default for
every task, add `review = true` under `[review]` in your config, then use
`--no-review` for the occasional terminal-only run.

**Direct apply** — skip the page, write immediately, from the terminal:

```bash
pedacito ask ~/code/anki -t "make the leech threshold configurable" --apply
```

Either way, originals are snapshotted first and nothing is written beside
your source files.

## Step 8 — Undo, if it went wrong

```bash
pedacito restore ~/code/anki --list
```

```
  20260830-005535   1 file(s): src/scheduler.py
```

```bash
pedacito restore ~/code/anki --dry-run     # what the newest snapshot would put back
pedacito restore ~/code/anki               # put it back, byte-for-byte
pedacito restore ~/code/anki --snapshot 20260830-005535
```

---

## Or: do steps 3–7 in a browser

```bash
pedacito gui
```

Opens a local page at `http://127.0.0.1:<port>/`. Paste your project path
and pick a profile — the dropdown fills in as you type, before you even
click Open — hit **Build index** the first time, then **Check model**,
then type a task and hit **Run**. The review page from step 7 renders
right there inline; sign off or discard exactly like the standalone
version. Tick "write directly" to skip review and apply immediately, the
GUI's equivalent of `--apply`. An **Explain** box answers questions about
the project's code in prose — no edits, no review — the GUI's equivalent of
`pedacito explain`.

A sidebar lists every file currently in the index with a filter box, so
you can sanity-check step 4's `.pedacitoignore` changes without switching
to `pedacito files`. Pick a color scheme from the header if the default
doesn't suit you, and use the **Exit** button when you're done instead of
closing the tab and hunting down the terminal to Ctrl-C.

Steps 1, 2, and 8 — install, configure, and restore — still go through the
terminal; the GUI doesn't cover those yet.

---

## Starting a project from scratch

Everything above edits an existing codebase. `pedacito new` builds one from
a description — a one-liner with `-t`, or a bigger spec document with
`--spec` (a `.md`/`.txt` file, stdin, or both):

```bash
pedacito new ~/code/my-app -t "a CLI todo manager with a sqlite backend"          # dry run: plan + preview only
pedacito new ~/code/my-app -t "..." --apply                                       # write it all
pedacito new ~/code/my-api --spec docs/spec.md --review                           # sign off in the browser first
pedacito new ~/code/my-app -t "..." --plan-only --apply                           # skeleton only; finish with ordinary ask runs
```

Three stages: the model plans a file tree (JSON, capped at
`max_new_files`), the skeleton is written (each file just a header comment
stating its purpose), then one build pass per file fills in the
implementation with the neighbouring files visible. Dry run by default,
backed up like any `--apply` (`pedacito restore` undoes a whole run), and
the target directory must not exist or be empty unless you pass `--force`.

---

## Working continuously

```bash
pedacito chat ~/code/anki
```

Keeps the index warm between tasks instead of reloading it each time. Add
`--apply` or `--review` if you want writes enabled for the whole session.
`quit` (or Ctrl-D) to leave. A handful of slash commands work mid-session:

| | |
|---|---|
| `/profile NAME` | switch model profile (waits for it to load if needed) |
| `/model ID` | switch to a model name directly |
| `/apply on\|off` | toggle whether edits get written |
| `/steps N` | how many lookups the model gets per task |
| `/status` | current profile, model, and write mode |
| `/profiles` | list the profiles in your config |
| `/help`, `/quit` | |

This is the cheapest way to compare two models on the same task: run it,
`/profile other-model`, run it again, compare the two diffs. The index stays
warm in between — no rebuild. A failed switch leaves the session on the
model you already had, rather than in a broken state.

---

## Running more than one local model

If you keep several models around — different sizes, a coder-tuned one, one
with thinking on and one with it off — define a profile per model instead of
editing `model =` by hand each time:

```toml
[profiles.small]
model = "qwen2.5-coder:7b"        # example Ollama name
tier = "small"                     # tight token limits: 1500-token answers,
                                   # 12k-char gathers — fits an 8k context window

[profiles.big]
model = "devstral-small-2-24b-instruct-2512"   # example LM Studio id
tier = "large"                     # roomy budgets for a big local model
max_steps = 8                      # optional hard lookup cap (0/default = none)

[profiles.api]
base_url = "https://api.deepseek.com/v1"
model = "deepseek-chat"
json_mode = "object"               # hosted APIs reject json_schema
# no tier line needed: ".../api.deepseek.com" + "deepseek-chat" infer "hosted"
# -> 40000-token answers, 7000-token steps, 100-step gather budget.
# Hosted profiles are never clamped to a local window: raise limits only if
# your provider's real caps require it (explicit keys always win).

[profiles.reasoning]
model = "qwen3:32b"
disable_thinking = true          # skip the <think> block: faster, cheaper
thinking_suffix = "/no_think"    # a second, more universal way to ask; harmless to set both
```

Every profile carries its own **token limits** — summaries, tool steps,
final answers, scaffold plan/build passes, gather ceiling, map budget.
`tier = "small"|"medium"|"large"|"hosted"` sets all of them at once from a
preset matched to the model class (or pedacito infers the tier from the
model/server name); individual keys written in the profile win over the
preset. On a local server these token values are fallbacks: when the server
reports the model's file size, VRAM caps the context window and the tier
token values are used only when they fit inside it; otherwise auto-fit uses
the VRAM-derived allocation. Hosted tiers keep their generous values and are
never hardware-capped.
That's what lets one config serve a 9B on laptop hardware and a DeepSeek API
call without hand-tuning seven knobs per model.

```bash
pedacito ask anki -t "..." --profile big
pedacito config --profile big         # see exactly what it resolves to, and where from
```

Four ways to choose a profile, in increasing precedence: `default_profile =`
in your config (a lasting default) → `PEDACITO_PROFILE` env var (one shell
session) → `--profile` on one command → `/profile` inside a running `chat`
session.

**Model names differ by server.** LM Studio wants the id shown by
`/v1/models` (visible in `pedacito config --check`'s output, or the LM Studio
UI). Ollama wants the name shown by `ollama list`, e.g. `qwen2.5-coder:14b`.
Copy it exactly — a mismatch is the single most common setup error.

**Switching models means the server has to load the new one**, which can
take anywhere from instant to a minute or two depending on size and whether
something else is still resident. Pedacito waits for this automatically
(`load_wait_seconds`, default 240) and tells you what's happening:

```
[server currently has qwen3.5-9b loaded; requesting 'qwen3:32b']
[waiting for 'qwen3:32b' to load -- switching models can take a minute or two]
[model ready after 29s]
```

If a model never finishes loading, that's a server-side problem (usually
not enough free VRAM for both models at once) — see the "Before you start"
section above.

---

## Writing good tasks

This is a small local model, not a frontier hosted one. The scaffolding gets
it to the right 300 lines out of 12,000; it doesn't make it smarter once
it's there.

**Works well** — narrow and local:

- `"add retry-with-backoff to the sync client"`
- `"is_leech has an off-by-one, fix it"`
- `"add type hints to Scheduler"`
- `"why does next_due return a float when the table has ints?"`

**Works badly** — broad and structural:

- `"refactor the scheduler"`
- `"modernise this codebase"`
- `"add tests"` (no anchor for it to search from)

Name the file or symbol if you know it — `"fix the off-by-one in is_leech"`
beats `"fix the leech bug"`, because the first gives the lookup phase
something exact to grab. Questions work too: if the task doesn't need a
code change, you get an answer and no edit blocks.

---

## When something looks wrong

| Symptom | What's happening |
|---|---|
| `Cannot reach the server` | Server isn't running, wrong port, or (remote setup) not listening on the network — see "Before you start." |
| Model listed but won't load / generate | Usually VRAM: another model is still resident. Free it up, or lower context length. |
| `search block not found` | The model copied line-number prefixes into its edit text, or picked a snippet that appears twice. Gets two automatic retries; if it keeps failing, narrow the task. |
| Model claims a function doesn't exist | It's not in the index — check `pedacito files`, an ignore rule is probably too broad. |
| `Model would not produce valid JSON` | Usually a truncated reply (the model wrote too much before finishing its tool call), not actually bad JSON. Pedacito retries with more room automatically; if it persists, raise `max_tokens_step`. A thinking model that spends its whole budget on reasoning returns no JSON at all — pedacito now says so explicitly; turn thinking off or give it more room. |
| Every request fails / "kept hitting limits" | Budgets exceeded the server's context window. This is fixed automatically at startup (auto-detect + auto-fit) — check the `[auto-fit]` line it prints, and see "Token budgets" in the README. If you set `auto_fit = false`, remove that. If the server reports a stale/default window, set `context_tokens` explicitly (and `k_cache_quant`/`v_cache_quant` if the cache is quantized). |
| Local run finishes but the answer is nonsense | The prompt was silently truncated by the server (Ollama does this instead of erroring) so the model never saw the task. Update pedacito — oversized answer prompts are now refused up front with instructions, and history is trimmed to fit mid-task. Raising the server-side context length helps too. |
| Model reads the wrong things / seems lost | There is no default lookup-step cap; if you set `--steps`/`max_steps`, raise it. Also name the file directly in the task. |
| Answers turn vague and generic | The project may have crossed the map's token budget and switched from a full listing to per-file cards. `pedacito map ~/code/anki --tier auto` shows which mode is active. |
| Stale or misleading summaries | Your docstrings are inaccurate — a docstring substitutes for a generated summary by default. Set `summarise_documented = true` and re-index. |
| No review page appeared | Nothing anchored, so there was nothing to review — check the printed failures. |
| Everything is slow | Check the machine hosting the model: if throughput has collapsed, it's likely spilled out of VRAM. |

To force a clean rebuild, delete `<project>/.pedacito/` — always safe, just
costs a re-index. Or pass `--reindex` to `ask`/`chat`.

---

## The commands, briefly

| | |
|---|---|
| `pedacito config --check` | Is the server up, and does the model actually work? |
| `pedacito how "..."` | Smart help: ask how to use Pedacito. No project needed. |
| `pedacito files PROJ` | What would be indexed, and why. Free. |
| `pedacito cost PROJ` | What indexing will cost. Free. |
| `pedacito map PROJ` | Print the index. No model needed. |
| `pedacito index PROJ` | Build or refresh the index. |
| `pedacito ask PROJ -t "..."` | One task, dry run by default. |
| `pedacito ask PROJ -t "..." --review` | Same, with a side-by-side sign-off page. |
| `pedacito ask PROJ -t "..." --apply` | Same, writes immediately. |
| `pedacito chat PROJ` | Interactive; index stays warm between tasks. |
| `pedacito explain PROJ -q "..."` | Answer a question about the project's code. No edits. |
| `pedacito new DIR -t "..."` | Start a project from scratch: plan, scaffold, build. |
| `pedacito new DIR --spec FILE` | Same, from a larger spec document. |
| `pedacito restore PROJ` | Undo the last write. |
| `pedacito gui` | Browser control panel: pick a project, check the model, ask, explain, review inline. |

Useful flags on most commands: `--exclude 'pattern'`, `--include 'pattern'`,
`--steps N`, `--model ID`, `--url URL`, `--profile NAME`, `--root PATH`.

## Where things live

```
<project>/.pedacito/
    index.json           the map
    summaries.json       summary cache, keyed by content hash
    backups/<stamp>/     pre-write snapshots of every file touched
    sessions/<stamp>.md  what was asked, looked up, and changed
    reviews/<stamp>.html saved copies of review pages
```

Session files are worth knowing about — when an edit turns out wrong days
later, that file shows exactly what the model was looking at when it made
the call. Deleting `.pedacito/` is always safe; it just costs a re-index.

Your personal settings live in `~/.config/pedacito/config.toml`
(`%APPDATA%\pedacito\config.toml` on Windows). Run `pedacito config` any time
to see every resolved value and which file or profile set it.

The browser GUI remembers recently opened projects in `gui_recent.json`
next to that same config file — safe to delete, it just empties the
dropdown.
