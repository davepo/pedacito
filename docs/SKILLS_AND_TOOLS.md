🌐 **English** | [Español](es/SKILLS_AND_TOOLS.md)

# Skills and tools

This guide covers the ways you extend what Pedacito knows and can do: the 19 bundled skills,
custom skills, named MCP servers, JSON hooks, and prompt overrides. Settings live in
[CONFIG.md](CONFIG.md); the trust model behind them is in [SECURITY.md](SECURITY.md).

## Contents

- [Bundled skills](#bundled-skills)
- [Custom skills](#custom-skills)
- [Named MCP servers](#named-mcp-servers)
- [JSON hooks](#json-hooks)
- [Image generation action](#image-generation-action)
- [Gather formats and oversized observations (v0.3.3)](#gather-formats-and-oversized-observations-v033)
- [Prompt overrides](#prompt-overrides)

## Bundled skills

Skills are short markdown checklists. The model sees a one-line catalog and loads a body only
when it needs it; the two best trigger matches for a task are also injected up front. Skills
are prompt text: nothing in a skill is ever executed. `pedacito skills list` shows every skill
with a scope badge (bundled, user or project).

| Skill | What it is for | Example triggers |
|---|---|---|
| `add-a-test` | Add a focused test for a function that has none | test, tests, coverage, untested, pytest |
| `fix-a-failing-test` | Diagnose and fix a failing test | failing test, failed test, test failure, broken test, assertionerror |
| `refactor-safely` | Rename or restructure code without breaking callers | refactor, restructure, signature, callers, extract |
| `review-my-diff` | Review the current uncommitted changes for defects | review, diff, uncommitted, defect, audit |
| `find-dead-code` | Check whether a file or function is unused | dead code, unused, unreferenced, orphan |
| `surgical-symbol-edit` | Replace or insert functions and methods in large files using zero-anchor SYMBOL splices or LINES ranges without echoing old code. | symbol, splice, large file, outline, lines range |
| `frontend-dom-audio-debug` | Verify and fix web UIs, HTML/JS module contracts, canvas rendering, and silent Web Audio failures. | dom, getelementbyid, web audio, audiocontext, es module |
| `safe-multi-file-rename` | Rename a function, class, method, or variable across the project deterministically without manual multi-file edits. | rename, multi-file, usages, identifier |
| `fastapi-rest-endpoint` | Add or fix a FastAPI REST endpoint with Pydantic validation and tests | fastapi, pydantic, apireouter, testclient, endpoint |
| `canvas-svg-2d-3d` | Debug and build canvas/SVG/WebGL rendering and animation loops | canvas, svg, viewbox, webgl, three.js |
| `sqlite-sql-migrations` | Write safe SQLite schema migrations and correct SQL queries | sqlite, sqlite3, sql, schema, migration |
| `async-concurrency-safety` | Fix asyncio/threading races, deadlocks and subprocess handling | asyncio, async, await, threading, lock |
| `security-ssrf-xss-hardening` | Harden code against SSRF, XSS, CSRF, injection and path traversal | ssrf, xss, csrf, sanitize, path traversal |
| `i18n-localization-catalog` | Externalize UI strings and keep translation catalogs in sync | i18n, locale, translate, translation, spanish |
| `parser-lexer-state-machine` | Build lexers, parsers, ASTs and state machines that cannot loop | lexer, tokenizer, parser, ast, rpn |
| `performance-memory-profiling` | Find and fix bottlenecks, unbounded caches and memory leaks | lru, ttl, cache, evict, complexity |
| `modern-web-design-ui` | Design modern websites, landing pages and bio/portfolio pages with high-craft CSS | website, webpage, landing page, personal site, bio |
| `desktop-gui-and-games` | Build desktop GUIs and 2D games that install nothing and never hang headless tests | calculator, tkinter, pyqt, pyside, gui |
| `safe-git-commit` | Inspect changes and prepare a safe commit the user can run | commit, git, stage, push, message |

Project skills override user skills, which override the bundled library.

## Custom skills

Create one from the command line:

```bash
pedacito skills add NAME --description "..." --triggers "k1,k2" --body "1. ..."   # this project
pedacito skills add NAME --global ...                                              # your user library
pedacito skills draft "Skill idea in plain words"    # the model drafts it for you to review
pedacito skills list | show NAME | rm NAME
```

Files are saved as `.pedacito/skills/NAME.md` (project) or in the `skills/` folder next to your
config (user), with YAML frontmatter (`name`, `description`, optional `triggers`).

In the GUI open **Project ▾ → Skills library & creator…**, fill the form or press
**✦ Auto-Draft from Description**. To force a skill into the next task without waiting for a
trigger match, use **Use in next task** in that dialog or, in the TUI, `/skill NAME`. A pinned
skill rides every task as `skill:NAME` until you run `/pin remove skill:NAME` or `/pin clear`.
Details: [USAGE.md](USAGE.md#skills-catalog-on-demand-and-manual-injection).

## Named MCP servers

Declare servers as `[mcp_servers.<name>]` tables and the model reaches them through one
`mcp_call` action:

```toml
enable_mcp = true

[mcp_servers.docs]
command = "python"
args = ["-m", "docs_server"]
trusted = false          # true: calls run without asking

[mcp_servers.jira]
url = "http://127.0.0.1:8000/mcp"
```

Tools are pruned per task (at most four matching signatures reach the prompt) and every call
is permission-gated unless the server is `trusted`. Project `.pedacito.toml` files cannot
define servers until you run `pedacito trust`. Manage them in **⚙ Settings → MCP Servers** or
check the pruned view with `pedacito mcp list`. Full reference:
[CONFIG.md](CONFIG.md#named-mcp-servers).

## JSON hooks

Hooks run your own argv (never a shell) at events such as `pre_task`, `post_edit`, `pre_run`
or `post_compact`. A hook gets one JSON object on stdin and may answer with
`{"decision": "allow" | "deny", "message": "...", "context": "..."}`; a `pre_*` hook can also
deny with exit status 2. Project-level hooks only run after `pedacito trust`.

```toml
hooks = { post_edit = ["ruff format {file}"], pre_edit = ["./scripts/guard.sh"] }
hook_timeout = 10
```

See [CONFIG.md](CONFIG.md#hooks) for the full event list and protocol.

## Image generation action

`generate_image` is an opt-in agent action: `prompt` (what to draw), optional `size` (such as `1024x1024`) and `file`
(a workspace-relative target; the default is `.pedacito/images/`). It appears in the model's tool list only when an
image profile is configured (`image_profile`, or an auto-detected `modality = "image"` profile) **and** either
`allow_image_gen = true` or a live permission broker (GUI/TUI) can approve each call. With no image profile the tool
schema is unchanged. The write uses the same workspace containment and atomic write as every other file write.
See [Local image generation](CONFIG.md#local-image-generation-image_profile-allow_image_gen).

## Gather formats and oversized observations (v0.3.3)

The gather loop can ask for each lookup step in one of three formats, set by
`tool_mode`:

- `grammar` (default): a JSON step, constrained by the server when `enforce_json_schema` is on.
- `native`: the model's own tool-call channel.
- `plain`: short plain words, for example `read app.py 1 40` or `grep parse_args`. It suits small
  models that stumble on JSON. `calibrate` A/B-tests it against `grammar` and keeps the format that
  scores better.

Any observation over `max_observation_tokens` from `run`, `run_test`, `grep`, `diff`, `git_diff`,
`fetch`, `web_search`, `diagnose` or `serve` is spilled to `.pedacito/scratch/<id>.log`. The model
gets the head, the tail, the line count and the file name. It can then `read` or `grep` exact lines
from that file, and only from direct `*.log` files under `.pedacito/scratch/`. The model can never
write to that directory. See [Security](SECURITY.md#git-commits-pull-requests-undo-and-scratch-files-v033).

## Prompt overrides

Any built-in prompt section can be replaced wholesale by a markdown file named after it:
`.pedacito/prompts/*.md` in the project, then `prompts/*.md` next to your user config
(for example `gather.md`, `edit_format.md`, `answer.md`). An override bypasses the section's
built-in guidance, so keep the format instructions the engine relies on. See
[ARCHITECTURE.md](ARCHITECTURE.md) for how the prompt is assembled.
