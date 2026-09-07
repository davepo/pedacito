# Pedacito

*Installs the `pedacito` command, plus `pdc` as a shorter alias — the two are interchangeable everywhere in this document.*

**Pedacito lets a small local LLM work on a codebase bigger than its context
window.** Models in the 9B–27B range typically can't hold a real project in
context, so instead of pasting your whole repo at the model, Pedacito builds
a lookup index and lets the model request only the specific files, symbols,
or snippets it needs for the task at hand — piece by piece (*pedacito* is
Spanish for "little piece"), back and forth, until it has enough to answer
or edit. It's aimed at the kind of hardware most people running local models
actually have: a single consumer GPU with 16–24GB of VRAM, not a cluster.

Works with any server exposing an OpenAI-compatible `/v1` API — tested
against **LM Studio and Ollama** — on **Windows, Linux, or macOS**. Nothing
in this document is specific to one server or one OS unless it says so.

Chunking is AST-based for Python and tree-sitter-based for JavaScript,
TypeScript, and Ruby — parser output, not line-based guessing, for most of
the index — and edits come back as search/replace blocks rather than whole
files.

### What this is (and isn't)

This was built by one person, for one person's use case: running local
models over Tailscale against a single home GPU box, mostly for narrow,
localized edits rather than sweeping refactors. It was designed and coded
almost entirely through conversations with Claude (Anthropic's AI model) —
this isn't a hand-rolled framework, it's what came out of iterating with an
AI pair programmer against real bugs on real hardware. That background is
worth knowing before you point it at your own workflow: it's had one user
and one set of models exercising it so far, edge cases outside that path are
more likely to be unpolished, and the code and docs may still read like
something built for personal use first and a public tool second.

If you have the hardware or budget for a frontier hosted model, you will
almost certainly get better results from tools built around those models —
see "Alternatives" below. Pedacito exists for the case where you specifically
want to stay local and small, and are willing to trade some capability for
that.

### Alternatives

If you're not committed to a small local model, these are generally more
capable and more polished:

- **[Claude Code](https://claude.com/claude-code)** — Anthropic's
  agentic coding tool. The flagship option if you can use Claude models.
- **[Qwen Code](https://github.com/QwenLM/qwen-code)** — a similar
  agentic CLI built around Qwen's models, works well with the same local
  Qwen weights Pedacito can run against.
- **[Aider](https://aider.chat)** — a mature, actively developed CLI
  pair-programming tool with git-aware edits, repo maps, and support for
  both hosted and local models. Closest in spirit to what Pedacito does,
  and far more battle-tested.
- **[Cline](https://github.com/cline/cline)** (and its fork
  **[Roo Code](https://github.com/RooCodeInc/Roo-Code)**) — VS Code
  extensions that turn Claude, GPT, or a local model into an autonomous
  agent with file and terminal access, right inside the editor.
- **[Continue](https://continue.dev)** — an open-source VS Code/JetBrains
  extension for chat and autocomplete against hosted or local models.
- **[Cursor](https://cursor.com)** and **[Windsurf](https://windsurf.com)**
  — full AI-native editors, hosted-model-first, generally the smoothest
  experience if you don't need to stay local.

Pedacito is worth reaching for specifically when none of those fit: no
internet access to a hosted API, a preference for keeping code entirely
on your own hardware, or just wanting to see what a 9B–27B model can do
on a real project when it's not fighting its context window.

---

## Install

Requires Python 3.11+ (uses the standard-library `tomllib`).

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

`-e` (editable) installs a pointer to this folder rather than copying files,
so `git pull` or replacing a file here takes effect immediately. This pulls
in `requests` — the only required dependency — and puts a `pedacito`
command on your `PATH` — along with `pdc`, a shorter alias for the same
command. Everywhere this document shows `pedacito <command>`, `pdc <command>`
works identically.

**JavaScript, TypeScript, and Ruby** get symbol-level indexing (a real map,
not just `read`/`grep`) out of the box — `tree-sitter` and
`tree-sitter-language-pack` (prebuilt wheels, no compiler needed) install
alongside `requests` as part of the base install, no extra step needed.

**Without installing**, from inside this folder: `pip install requests`,
then `python -m pedacito --help` (`py -m pedacito --help` on Windows). That works
but only from inside the folder, so installing is worth the one command.

If `pedacito` (or `pdc`) isn't found on `PATH` after installing (common with
a per-user pip install, especially on Windows), use `python -m pedacito` /
`py -m pedacito` instead — identical result, no `PATH` needed.

## Uninstall

```bash
pip uninstall pedacito-local
```

`pedacito-local` is the distribution name in `pyproject.toml` — not `pedacito`,
which is just the command it installs. Confirmed by testing an actual
install/uninstall cycle:

| | Removed by uninstall? |
|---|---|
| The `pedacito` command on `PATH` | Yes |
| The `pdc` command on `PATH` | Yes |
| pip's pointer file in site-packages | Yes |
| The source folder itself (wherever you cloned it) | No |
| Per-project `.pedacito/` directories (index, backups, sessions) | No |
| Your personal config (`~/.config/pedacito/` or `%APPDATA%\pedacito\`) | No |

Reinstalling later (`pip install -e .` from inside the folder) picks up
existing config and every project's `.pedacito/` untouched — nothing about
uninstalling wipes them. For a genuinely clean slate, delete those three
yourself: the source folder, the config directory above, and any `.pedacito/`
folders in projects you'd tested against.

One thing worth knowing since the install is editable: **don't delete or
move the source folder while it's installed.** The `pedacito` and `pdc`
commands are both pointers to that exact path, not a copy — move the folder
and they start failing with import errors until you reinstall from the new
location.

---

## Model server setup

Point Pedacito at any server that speaks the OpenAI chat-completions API.

**LM Studio.** Load a model, start the server (Developer tab → Status:
Running). If Pedacito runs on a different machine than LM Studio, enable
**"Serve on Local Network"** in that tab — without it the server only
listens on `127.0.0.1` and a remote machine is refused. Default port `1234`.

**Ollama.** `ollama serve`, then `ollama pull <model>`. Listens on
`127.0.0.1` by default; for a remote setup, set `OLLAMA_HOST=0.0.0.0` (or
the host's address) in Ollama's environment before starting it. Default
port `11434`.

Pedacito's tool-selection loop depends on the server enforcing
`response_format: {"type": "json_schema", ...}` — grammar-constrained JSON
output, so a tool choice always parses. Every GGUF/llama.cpp model in LM
Studio supports this. Ollama added support for it on `/v1/chat/completions`
for self-hosted instances; Ollama Cloud currently accepts the field without
enforcing it, so a Cloud endpoint should be treated as unverified until
tested. Without schema support, the gather loop falls back to salvaging JSON
from free text, which is markedly less reliable.

Verify the connection from wherever Pedacito runs:

```bash
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

This distinguishes "server unreachable" (network/firewall/not running) from
"model not found or won't load" (usually a name mismatch, or not enough
free VRAM) — different problems with different fixes.

---

## Configure

```bash
pedacito config --init
```

Writes a starter config to `~/.config/pedacito/config.toml` (Linux/macOS) or
`%APPDATA%\pedacito\config.toml` (Windows) — `pedacito config` always prints the
exact path it's using. Edit the two lines that matter:

```toml
[server]
base_url = "http://localhost:1234/v1"   # or :11434/v1 for Ollama
model = "qwen3.5-9b"            # exact id/name the server reports
```

`[server]` is the current section name; `[lmstudio]` still works as an
alias for older config files.

Settings resolve in layers, each overriding the one before:

| | |
|---|---|
| built-in defaults | |
| `~/.config/pedacito/config.toml` (`%APPDATA%\pedacito\config.toml` on Windows) | your machine: url, model |
| `<project>/.pedacito.toml` | per-project overrides |
| `--config FILE` | an explicit file |
| command-line flags | one-off overrides |

Run `pedacito config` with no arguments to see every resolved setting and
which layer set it:

```
Resolved settings (non-default marked):
  * base_url    'http://box.tail1234.ts.net:1234/v1'   <- user config (...)
  * max_steps   4                                      <- project config (...)
    temperature 0.2
```

Typos are rejected rather than silently ignored, with a suggestion:

```
Config error: /home/you/code/proj/.pedacito.toml: unknown setting(s):
  server.base_ur1   did you mean 'base_url'?
```

State for each project lives in `.pedacito/` at that project's root — see
"Workspace layout" below. Deleting it is always safe; it just forces a
re-index.

---

## Running multiple models

Define a profile per model instead of editing `model =` each time:

```toml
[profiles.small]
model = "qwen2.5-coder:7b"        # example Ollama name

[profiles.big]
model = "devstral-small-2-24b-instruct-2512"   # example LM Studio id
max_steps = 8

[profiles.reasoning]
model = "qwen3:32b"
disable_thinking = true          # skip the <think> block: faster, cheaper
thinking_suffix = "/no_think"    # a second, more universal way to ask; harmless to set both
```

```bash
pedacito ask anki -t "..." --profile big
pedacito config --profile big        # confirm exactly what it resolves to
```

A profile can override any setting, not just the model — useful when a
larger or slower model needs more lookups per task than a smaller one.

Model names are server-specific: LM Studio wants the id from `/v1/models`
(shown by `pedacito config --check`); Ollama wants the name from `ollama
list`. Copy it exactly — a mismatched name is the most common setup
mistake.

Four ways to select a profile, lowest to highest precedence:
`default_profile =` in your config → `PEDACITO_PROFILE` environment variable
→ `--profile` on one command → `/profile` inside a running `chat` session.
Inside `chat`, `/profile NAME` switches models mid-session (waiting for the
new one to load) while keeping the index warm — the cheapest way to compare
two models on the same task.

Switching models means the server has to load the new one, which can take
anywhere from instant to a couple of minutes. Pedacito waits for this
automatically (`load_wait_seconds`, default 240s) and reports progress:

```
[server currently has qwen3.5-9b loaded; requesting 'qwen3:32b']
[waiting for 'qwen3:32b' to load -- switching models can take a minute or two]
[model ready after 29s]
```

A model that never finishes loading is a server-side problem, usually not
enough free VRAM for both models at once.

### Reasoning models

Models that emit a `<think>...</think>` block before answering (Qwen3 and
similar) are supported: thinking is stripped before JSON parsing and before
edit extraction, so it never corrupts a tool call, an edit, or a session
log. Turning thinking off saves time and tokens but isn't required for
correctness. Two independent switches exist because which one a given model
honors varies: `disable_thinking` sends the server a
`chat_template_kwargs: {"enable_thinking": false}` hint (LM Studio/llama.cpp
convention); `thinking_suffix` appends literal text like `/no_think` to the
prompt instead. Setting both is harmless.

### Chat template compatibility

Chat templates differ in what message sequences they accept, and Pedacito
handles the two common failure modes transparently:

- **Mistral-family templates** (Devstral, Mistral Small) require strictly
  alternating user/assistant turns and raise an error otherwise. The gather
  loop is built to alternate; the client also collapses any adjacent
  same-role messages before sending, as a second line of defense.
- **Gemma** has no system role. If a template rejects one, the client folds
  the system prompt into the first user turn and retries automatically,
  remembering the result so it doesn't rediscover this on every call.

---

## What gets indexed

**Python, JavaScript/TypeScript/TSX, and Ruby** are parsed and chunked
semantically — functions, classes, methods, with exact line ranges —
Python via the standard-library `ast` module, the other three via
`tree-sitter`, installed by default (see "Install" above).

Everything else that's useful as context — `.md .rst .txt .csv .tsv .json
.toml .yaml .yml .ini .cfg .conf .env .sql .sh` — is indexed *structurally*:
the map shows a CSV's columns, a JSON file's top-level keys, a TOML file's
sections, a markdown file's headings, each with real line ranges. Contents
stay out of the map, but `read` and `grep` reach them normally — so one
`grep` can cross code and config together:

```
data/config.csv:2: min_score,0.75,ratio,cutoff for the run filter
src/app.py:18: return [i for i in items if i > self.cfg["min_score"]]
```

**Anything else is invisible by default** — not parsed, not structural, not
in the map, and `read`/`grep` will report it as not indexed. A language
without a symbol-level chunker (Go, Rust, Java, C, ...) only gets *any*
coverage once you add its extension explicitly:

```bash
pedacito ask . -t "..." --ext .go
```

which routes it through the same structural (fixed-size block) chunker as
an unrecognized reference file — `read`/`grep` work, there's just no
symbol map for it.

Reference files (structural, of any kind) are never sent to the model for
summarizing; their extracted shape already says more than a generated
sentence would, for free.

`grep`/`outline` locate code; only `read` returns text exact enough to
paste into an edit. The gather loop enforces this — it will not accept
`done` until the model has actually read something, so an edit can't be
built from a trimmed `grep` snippet that would never match the file.

## Excluding and including files

Three mechanisms, increasing precedence.

**1. `.pedacitoignore`** in the project root, gitignore syntax:

```
deprecated/           # a folder of old script versions
*_old.py
/scratch              # anchored to the root only
experiments/**/tmp.py
!deprecated/still_used.py   # negation; last match wins
```

The project's existing `.gitignore` is read too (`--no-gitignore` to skip
it), plus built-in defaults: `.git`, `__pycache__`, `.venv`, `node_modules`,
`build`, `dist`, `*.egg-info`, and Pedacito's own `.pedacito/`.

**2. Command-line flags**, repeatable:

```bash
pedacito ask src/ -t "..." --exclude 'deprecated/' --exclude '*_v1.py'
pedacito ask src/ -t "..." --include 'deprecated/still_used.py'
pedacito ask . -t "..." --ext .jinja --ext .proto   # index extra file types
pedacito ask . -t "..." --no-refs                   # code only (Python/JS/TS/Ruby), no .md/.json/etc.
```

**3. Naming a path directly always wins.** `pedacito ask deprecated/app_v1.py`
indexes that file even though `deprecated/` is ignored — typing it is an
explicit request.

`pedacito files` shows every candidate with the reason it was kept or
dropped, at no cost:

```
$ pedacito files .
  code  src/app.py
  ref   data/config.csv
  skip  deprecated/app_v1.py   [ignored]
  skip  .venv/lib/vendored.py  [ignored]
```

Two caveats on the pattern matcher, since it's a practical subset of
gitignore rather than a full implepedacitoion: character ranges like `[abc]`
aren't supported, and a negation always wins if it matches last — even if a
parent directory was excluded. Real gitignore refuses to resurrect files
under an excluded directory; this doesn't, because that's what most people
expect when they write a negation.

Files over `max_reference_bytes` (2MB default) are recorded by name and
header only, so a large CSV still shows up in the map without being read in
full.

---

## Index freshness

The index stores a content hash per file and checks it on every command.
Edits Pedacito makes are re-chunked immediately, whether applied from the
terminal or from the review page. Edits made outside Pedacito — in an editor,
between sessions — are detected and corrected automatically on the next
command:

```
index refreshed (2 changed, 1 new)
```

This matters because line ranges are only meaningful against the bytes they
were computed from. Without this check, editing three lines at the top of a
file would silently shift every range below it, and the model would read
the wrong function while believing it read the right one. Deleted files are
dropped from the index; new files are picked up structurally right away but
arrive without a generated summary until the next `pedacito index`.

Run a full `pedacito index` after a large refactor or a `git pull` that
touched many files. Day to day, it isn't necessary — every command
self-corrects. `pedacito map <project>` is always the fastest way to see
exactly what Pedacito currently believes about a project, with no model
involved.

---

## The map is tiered

A flat listing of every function does not survive a real project. Measured
on 452KB of Python stdlib source (`inspect.py`, `typing.py`, `tarfile.py`,
`argparse.py` — 570 units):

| | tokens in every prompt |
|---|---|
| Flat map with summaries | ~35,500 |
| Tier 0, file cards | **~1,500** |
| Tier 1, one file's outline (on demand) | ~2,400 |

Tier 0 is one card per file: line count, imports, a two-sentence summary,
and top-level definitions with their line ranges attached — with ranges on
the card, the model can often skip straight to `read` without an `outline`
round trip. Projects small enough that the flat map fits under
`flat_map_token_budget` get it automatically, since when it fits it is
strictly more useful.

When more detail is needed, the model descends: `outline` gives one file's
full unit list, `read` gives exact lines. Outlines are treated as
navigation rather than evidence — they stay in the gather conversation but
are dropped before the answer prompt, since carrying a multi-thousand-token
outline into the answer phase is pure waste once the model has decided what
to read.

## Summarizing is rationed

Of those 570 units, 442 cleared the length threshold for a summary — but
261 already had a docstring, and 281 were methods whose class summary plus
signature already says enough. Summarizing only undocumented top-level
units left 76 calls, plus one per file:

```
$ pedacito cost src/
570 units across 4 files
  eligible for a summary : 442
  skipped, has docstring : 261
  skipped, is a method   : 197
  LLM calls              : 76 units + 4 files = 80
  first index @ 25 tok/s : ~4 min (cached after)
```

`pedacito cost` builds only the static index, so it's instant and free — run
it before `pedacito index` on anything large. File summaries are generated
from a file's *skeleton* (imports plus signatures) rather than its raw
source, since a 100KB module doesn't fit in a prompt but its skeleton does.

---

## Editing: search/replace, not diffs or whole files

Small models get unified-diff hunk headers and line offsets wrong
frequently, and a wrong header can silently corrupt a file. Whole-file
rewrites are slow at local inference speeds and models tend to quietly drop
a function somewhere in the middle of a long regeneration.

Instead, edits are search/replace blocks anchored on text the model
actually read:

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

A block either matches the file exactly or it fails cleanly — no partial or
silent corruption. Every edit is validated before anything is written —
`ast.parse()` for Python, `tree-sitter`'s parse-error detection for JS/TS/Ruby
(when that extra is installed) — and a result that doesn't parse cleanly is
rejected, nothing written. A failed block's error is fed back to the model
(up to two rounds) so it can re-anchor with more context.

## Review page

`--review` opens a side-by-side page instead of writing directly: current
file on the left, proposed version on the right, changed words
highlighted, long unchanged stretches folded. Self-contained HTML — no
CDN, no external fonts — so it opens and works offline. Applying takes two
deliberate clicks (tick an attestation, then confirm), and Enter never
triggers a write.

The property that matters more than the buttons: the page carries the
SHA-256 of every file as it stood when the preview was built, and applying
re-hashes and refuses on any mismatch. Editing a file in your own editor
while the review sits open causes the write to be refused rather than
silently overwriting your change:

```
Not written: src/scheduler.py (changed on disk). Re-run the task to rebuild the review.
```

The review server binds to `127.0.0.1` only and shuts down after sign-off,
discard, or a timeout (`review_timeout`, default 1800s) — it writes files,
so it has no business listening on a network interface. Make it the default
for every task with `review = true` under `[review]` in your config, then
use `--no-review` for terminal-only runs.

## Browser GUI

```bash
pedacito gui                # opens in a browser at an OS-chosen port
pedacito gui --port 8756    # fixed port instead
pedacito gui --no-open      # print the URL instead of launching a browser
```

A local control panel instead of the terminal: pick a project, check the
model, type a task, and watch the same review page above render inline —
one process, one page, bound to `127.0.0.1` like every other server
Pedacito starts.

- **Project** — type or paste a path; a small recent-projects list is
  remembered between sessions, and the profile dropdown fills in as soon
  as you finish typing, before Open is ever clicked.
- **Build index** only appears when the project has none yet — never
  triggered silently, since indexing spends real model calls.
- **Check model** runs the same reachability-plus-generation check as
  `pedacito config --check`.
- **Ask** runs a task with review on by default; tick "write directly" to
  go straight to `--apply` instead.
- A sidebar lists every file currently in the index, with a filter box.
- Five color schemes (Parchment, Sage, Mocha, Clay, Nightshade), remembered
  in the browser between sessions.
- **Exit** stops the server from the page itself.

Current limits: one project open at a time (switching directories reloads
the warm index/agent rather than keeping several resident); detailed
step-by-step gather/answer progress still prints to the terminal
`pedacito gui` was launched from, not into the page; and `restore`, the
advanced per-task flags (`--steps`, `--exclude`/`--include`, `--ext`,
`--no-refs`, `--no-gitignore`, a one-off `--url`/`--model` override), and
narrowing to a subset of paths are CLI-only for now.

The recent-projects list lives in `gui_recent.json` next to your user
config (see "Configure" above) — deleting it just empties the dropdown.

## Undo

Every write (from `--apply` or the review page) snapshots originals into
`.pedacito/backups/<timestamp>/` first:

```bash
pedacito restore <project> --list
pedacito restore <project> --dry-run
pedacito restore <project>
pedacito restore <project> --snapshot 20260830-020724
```

Restoration is byte-for-byte, including the original line-ending style.
Each run also writes a session record — the task, every lookup the model
made with its stated reason, the reply, and which edits landed — to
`.pedacito/sessions/<timestamp>.md`. That's the file worth opening when an
edit turns out wrong days later.

---

## Workspace layout

Nothing Pedacito writes lands beside your source files. Everything for one
project lives under that project's root:

```
<project>/.pedacito/
    index.json           the map
    summaries.json       summary cache, keyed by content hash
    backups/<stamp>/     pre-write snapshots, mirroring the tree
    sessions/<stamp>.md  what was asked, looked up, and changed
    reviews/<stamp>.html saved copies of review pages
```

On first run in a project, Pedacito also creates a commented `.pedacitoignore`
and adds `.pedacito/` to that project's `.gitignore` — but only if it's
already a git checkout, never inventing a `.gitignore` where one wasn't
there. Both behaviors are switchable (`create_ignore_file`,
`manage_gitignore`).

### The project root is inferred

`.pedacito/` goes into the *project*, not wherever the terminal happens to be:

```bash
cd ~/pedacito
pedacito ask ~/code/anki -t "..."     # state lands in ~/code/anki/.pedacito/
```

The root is the common ancestor of the paths given. Pass `--root` to
override — for example, indexing one subdirectory of a larger repository
while keeping paths relative to the repository root.

---

## Cross-platform correctness

Every path a project file takes through Pedacito is handled explicitly for
Windows, since naive text I/O will silently corrupt files there:

- **Line endings are preserved exactly.** A CRLF file stays CRLF; an LF
  file stays LF — even though matching and chunking work on LF internally.
  This matters more than it sounds: default Python text I/O on Windows
  translates line endings on write, which turns a three-line edit into a
  whole-file rewrite the moment it touches an LF file.
- **A UTF-8 byte-order mark**, if present, is stripped before parsing and
  restored on write.
- Both `src\app.py` and `src/app.py` are accepted anywhere a path is typed.
  The index always stores POSIX-style paths internally, so `.pedacito/` built
  on one machine reads correctly on another.
- Ignore-pattern matching follows the platform, the same way git does:
  case-insensitive on Windows, case-sensitive on Linux/macOS.
- Paths on different Windows drive letters have no common ancestor for
  root inference; pass `--root` explicitly if a run spans drives.

---

## Configuration reference

Set any of these under the matching section in your config, or as
`[profiles.NAME]` overrides. `pedacito config` lists every field with its
current value and where it came from.

| Setting | Default | |
|---|---|---|
| `base_url` | `http://localhost:1234/v1` | Server address. |
| `model` | *(none)* | Exact id/name as the server reports it. |
| `timeout` | 600.0 | Seconds per request; raise for very large files. |
| `load_wait_seconds` | 240 | How long to wait for a model to load. |
| `max_consecutive_failures` | 3 | Aborts an indexing batch after this many failures in a row. |
| `temperature` | 0.2 | Higher makes tool selection noticeably worse. |
| `max_tokens_step` | 700 | Budget for one tool call; raise for models that narrate at length. |
| `max_tokens_answer` | 3000 | Budget for the final answer/edit reply. |
| `max_steps` | 6 | Lookups allowed before the model must answer. |
| `max_gathered_chars` | 24000 | Ceiling on accumulated source per task. |
| `flat_map_token_budget` | 4000 | Above this, the map switches from a full listing to per-file cards. |
| `summarise_methods` | False | On if your methods lack useful docstrings. |
| `summarise_documented` | False | On if your existing docstrings are stale or wrong. |
| `include_references` | True | Off indexes only Python/JS/TS/Ruby (code), not the other reference file types listed under "What gets indexed". |
| `extra_extensions` | `()` | Extra reference file types, e.g. `(".jinja", ".proto")`. |
| `max_reference_bytes` | 2MB | Above this, a reference file is header-only. |
| `disable_thinking` | False | Sends a template hint to suppress `<think>` blocks. |
| `thinking_suffix` | `""` | Text appended to the prompt for the same purpose, e.g. `/no_think`. |
| `backup` | True | Snapshot originals before writing. |
| `log_sessions` | True | Write a session record per task. |
| `review` | False | Always open the side-by-side reviewer. |
| `open_browser` | True | Off prints the review URL instead of launching a browser. |
| `manage_gitignore` | True | Add `.pedacito/` to the project's `.gitignore`. |
| `create_ignore_file` | True | Create a starter `.pedacitoignore` on first run. |
| `default_profile` | `""` | Profile applied when `--profile` isn't given. |

## Two things to watch when pointing this at your own code

**`summarise_documented=False` assumes your docstrings are accurate.** A
docstring substitutes for a generated summary entirely — that's most of the
indexing-time saving. If yours are stale or misleading, the map will be
confidently wrong in exactly the places you'd least expect it. Set it to
`true` and accept the longer first index.

**`flat_map_token_budget=4000` is the tier switch.** If a mid-sized project
feels like it's navigating worse than it should, run `pedacito map <project>
--tier auto` to see which side of the line it landed on — a project just
above the threshold gets file cards when the flat map would have fit and
served it better, and raising the budget is the fix.

## Known limits

- **Symbol-level parsing covers Python, JavaScript/TypeScript/TSX, and
  Ruby** (the last three via `tree-sitter`, installed by default — see
  "Install"). Any other extension either falls through to *structural*
  indexing (the reference file types listed under "What gets indexed",
  plus anything added via `--ext`/`extra_extensions`) or, if never
  recognized or added, isn't indexed at all — no map entry, and `read`/
  `grep` will report it as not found. `pedacito files` always shows which
  bucket a given file landed in.
- Ruby's nested `module`/`class` namespacing (`module App; class Foo; ...
  end; end`) gets its own recursive handling distinct from the JS/TS
  chunker, since that idiom has no real equivalent in the other two
  languages. A small, unsplit container (under `class_split_lines`) is
  captured whole rather than recursed into — the same trade-off
  `chunker.py` already makes for a short Python class — so very deep
  nesting inside a short file won't get a separate symbol per level.
- The call graph matches on bare names, so two unrelated classes with a
  `run` method are conflated. Fine for locating code, not sound for
  automated refactoring.
- No embeddings — `grep` plus the summary map covers most of what semantic
  search would, without a second model resident in memory.
- Directory walking follows symlinks and is cycle-safe (a symlink pointing
  back at an ancestor is visited once, not recursed forever), so a
  vendored or shared directory reached only through a symlink is actually
  indexed rather than silently skipped.
- Broad, structural tasks ("modernize this codebase") are a poor fit
  regardless of scaffolding — that's a limit of small local models, not
  something retrieval can paper over. Narrow, localized tasks are where
  this is genuinely useful.
- The browser GUI (`pedacito gui`) covers ask/chat/review/build-index/
  model-check; see "Browser GUI" above for what's still CLI-only.

---

## Contributing

This started as a personal tool built for one setup (LM Studio over
Tailscale, a single home GPU box), so it hasn't been exercised against the
range of models, servers, and projects a wider audience would throw at it.
Issues and pull requests are welcome — bug reports with a repro, support for
other model servers or languages, and fixes for anything that breaks outside
the one setup this was built and tested on are all useful. Given the small
scope, please open an issue before a large PR so the direction can be agreed
on first.

## License

MIT — see [LICENSE](LICENSE). Short version: do essentially anything you
want with this code (use, modify, redistribute, include in a commercial
project), as long as the copyright notice stays attached; the software
comes with no warranty.

MIT was picked over Apache-2.0 because Apache's extra provisions (an
explicit patent grant, contributor-notice tracking) mostly matter for
larger projects with multiple contributors and real patent exposure —
overhead this small, single-author tool doesn't need. MIT gives the same
practical freedom with less text to read.
