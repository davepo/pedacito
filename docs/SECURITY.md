🌐 **English** | [Español](es/SECURITY.md)

# Security: AI coding harness and agent best practices

Pedacito is a local-first coding agent. A model reads your files, proposes edits,
and can ask to run commands. This guide explains how Pedacito contains that
behaviour, what it does **not** protect against, and what you should configure.

It is written for two audiences: people deciding whether to trust an agent on
their machine, and people building their own harness who want a checklist of
patterns that work.

For how hooks, MCP servers and skills fit the trust model, see
[SKILLS_AND_TOOLS.md](SKILLS_AND_TOOLS.md). To contribute fixes, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Threat model

The agent is only as trustworthy as the least trusted text it reads. Pedacito
assumes each of these can be hostile or simply wrong:

| Source | Risk |
|---|---|
| **Model output** | Proposes a path outside the project, a destructive shell command, or an edit to the wrong file. |
| **Repository contents** | A `.pedacito.toml`, a `Makefile`, or a test file can contain commands or instructions. Symlinks can point outside the project. |
| **Tool output** | Command output, web pages, MCP replies and LSP results can contain "ignore previous instructions" text. |
| **Secrets in the workspace** | `.env` files and private keys can be pulled into a prompt and sent to a hosted provider. |
| **Other websites** | A page open in your browser can try to reach the local GUI server. |

The defences fall into four pillars. Each pillar limits the damage if the layer
above it fails.

| Pillar | Question it answers | Main code | Default |
|---|---|---|---|
| 1. Credential isolation and pre-flight scanning | Can a secret leave the machine? | `pedacito/security.py`, `pedacito/safety.py`, `pedacito/llm.py` | On (`warn`) |
| 2. Strict workspace sandboxing | Can a path escape the project? | `security.resolve_safe_path`, `pedacito/index.py`, tool handlers | Always on |
| 3. Least-privilege execution | What can a command do if it runs? | `pedacito/sandbox.py`, `pedacito/proc.py`, `safety.child_env` | Sandbox on, shell off |
| 4. Audit trail and time travel | Can every change be seen and reverted? | `pedacito/checkpoints.py` | On for every run |

Pedacito is defence in depth, not a guarantee against a determined attacker or a
kernel bug. Read [Known limits and gaps](#known-limits-and-gaps) before relying on
it for untrusted code.

---

## Pillar 1: Credential isolation and pre-flight scanning

The goal is that a key sitting in your repository does not reach a hosted model
provider, a log file, or a git commit by accident.

### What is detected

Pedacito recognises these credential shapes. The patterns are in
`pedacito/security.py` (`_TOKEN_RULES`, `_GENERIC_ASSIGNMENT_RE`) and
`pedacito/safety.py` (`_TOKEN_RES`, `_PRIVATE_KEY_RE`):

| Kind | Pattern (simplified) |
|---|---|
| OpenAI, Anthropic and similar `sk-` keys | `sk-` followed by 20 or more of `[A-Za-z0-9_-]` (also `sk-proj-`, `sk-ant-`, `sk-live-`, `sk-or-v1-`) |
| PyPI tokens | `pypi-` followed by 20 or more characters |
| GitHub tokens | `ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_` with 30 or more characters, and `github_pat_` |
| GitLab tokens | `glpat-` with 16 or more characters |
| AWS access key IDs | `AKIA` followed by 16 uppercase letters and digits |
| Google API keys | `AIza` followed by 35 characters |
| Slack tokens | `xox[baprs]-` with 10 or more characters |
| Hugging Face tokens | `hf_` with 30 or more characters |
| JWTs | three dot-separated base64url segments starting `eyJ` |
| PEM private keys | `-----BEGIN ... PRIVATE KEY-----` through its `END` line (or end of text) |
| Generic assignments | `password`, `secret`, `token`, `api_key`, `authorization` and similar names, assigned a quoted value of 16 or more characters |

Specific token formats win over the generic assignment rule when both match the
same text.

### How values are filtered

Pedacito does not compute Shannon entropy. It has no entropy threshold, so a long
random string under an unrelated name is not flagged. Instead it applies three
cheap filters that cut false positives:

- **Length floors.** Generic assignment values must be at least 16 characters
  (in `safety.redact`, at least 12), without spaces.
- **Placeholder detection.** `_looks_like_placeholder` ignores values such as
  `your_key_here`, `changeme`, `<token>`, `${API_KEY}`, `os.environ[...]`,
  `process.env.X`, `xxxx`, and values containing `...` or `…`. Test and example
  markers (`test_`, `fake_`, `sample`, `dummy`) are also ignored.
- **Filler detection.** `safety.redact` skips a quoted value that lacks a letter
  or a digit **and** has fewer than six distinct characters. This removes
  repeated filler such as `aaaaaaaaaaaaaaaa`.

If you need entropy scoring for a custom token format, add it as a new rule in
`_TOKEN_RULES` and test it against your real key shapes.

### Where scanning runs

| Point | What happens | Setting |
|---|---|---|
| **Before every hosted request** | Pedacito scans the outgoing messages for unacknowledged secrets. Replayed reasoning fields (`reasoning_content`, thinking blocks, thought parts) are scanned too; a finding drops that opaque reasoning rather than sending it. Loopback and LAN endpoints are never scanned, because those keys do not leave the machine. | `security_outbound_mode` |
| **Workspace scan** | `scan_workspace_secrets` checks indexed files. Results appear as a post-run notice and in the GUI security status (`/security` in the TUI). | Always on |
| **Git hygiene** | `check_git_secret_hygiene` reports sensitive files that exist in the repository but are not excluded by `.gitignore`. `ensure_gitignore_entries` can add the missing lines. | `manage_gitignore` |
| **Indexing** | Sensitive files are never indexed (see below). | Always on |
| **Session files, logs, events** | Secret-looking values are masked before they are written to `.pedacito/sessions/`, `log.txt`, events and checkpoints. | `redact_secrets` (on) |

`security_outbound_mode` takes three values:

- `warn` (default). The GUI and TUI ask whether to **Acknowledge and send**,
  **Redact and send**, or **Cancel**. A non-interactive CLI run redacts automatically.
- `redact`. Every detected secret is replaced with `<REDACTED_SECRET>` before the
  request is sent.
- `allow`. Messages are sent unchanged. Use this only if you accept the risk.

Findings never expose the raw value. A finding carries the file, line, kind, a
masked preview (for example `sk-ab...wxyz`), and a SHA-256 fingerprint. Only the
preview and fingerprint are serialised. When you acknowledge a finding, its
fingerprint is stored in `.pedacito/security_ack.json`, so the same key is not
raised again.

### Provider capability probes

The capability cache and the opt-in probe follow the same outbound rules:

- Importing `pedacito.providers` never contacts a network. `get_capabilities`
  is offline unless the caller passes `allow_probe=True`, which is the explicit
  consent signal; startup and ordinary model calls never probe.
- A probe sends one minimal, bounded request to the configured base URL and
  refuses redirects, so an `Authorization` or `x-api-key` header never follows a
  redirect to another host.
- The probe prompt is fixed (`ping`) and passes the existing outbound secret
  scan. Project files, prompts and tool output are never included.
- `provider_capabilities.json` stores only the normalized URL, exact model,
  auth-identity hash and capability values, with a 24 hour TTL. It contains no
  API key, credential, prompt or tool output.
- Azure OpenAI, AWS Bedrock and Google Vertex are gateway-only entries and are
  never presented as native auth adapters; their capability source says
  `unverified` until a configured OpenAI-compatible gateway proves otherwise.

### Sensitive files are never indexed

These files are excluded from the index and from model reads unless you approve a
specific read in the permission dialog:

- `.env`, `.env.local`, `.env.production`, and any `*.env` file. Templates
  (`.env.example`, `.env.sample`, `.env.template`, `.env.dist`, `.env.defaults`)
  stay readable.
- Private keys and key containers: `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.jks`,
  `*.kdbx`, `id_rsa*`, `id_dsa*`, `id_ecdsa*`, `id_ed25519*`, `id_sk_ed25519*`.
  Public keys (`*.pub`) stay readable.
- Credential files: `.npmrc`, `.pypirc`, `.netrc`, `.git-credentials`,
  `credentials.json`/`.yaml`/`.yml`, `secrets.json`/`.yaml`/`.yml`,
  `*.credentials`, and `service-account*.json`.
- Anything under `.ssh`, `.gnupg`, `.aws`, `.docker`, or `.config/gcloud`.

Re-include a file deliberately with `!name` in `.pedacitoignore`, or reconsider
whether it belongs in the repository at all.

### Checklist

- Keep keys out of the repository. Either store `api_key` in your user config, which
  lives outside the project, or set `api_key_env` to the **name** of an environment
  variable, never the key itself.
- Keep `security_outbound_mode = "warn"` or `"redact"` for any hosted profile.
- Run `pedacito config --check` or `/security` and read the findings before the
  first hosted run on a new repository.
- Add `.env` and key files to `.gitignore`, or let Pedacito add them with
  `manage_gitignore`.
- If a key was ever sent to a hosted provider, rotate it. Redaction protects future
  requests, not requests already sent.

---

## Pillar 2: Strict workspace sandboxing

The goal is that every file the model touches is inside the project you opened,
and that a symlink, an absolute path, or a `..` segment cannot change that.

### Path containment: `resolve_safe_path`

Every tool path goes through `security.resolve_safe_path(root, target)`:

1. The project root is resolved with `Path.resolve()`, which follows symlinks.
2. The raw path text is checked for any `..` component. A literal `..` is refused
   even when it would collapse back inside the root, because a model-supplied path
   should never need the parent directory.
3. A relative path is joined to the root. An absolute path is used as given.
4. The candidate is resolved with `Path.resolve()`, so a symlink that points at
   `/etc/passwd` or `~/.ssh` is followed to its real target.
5. The resolved path must satisfy `is_relative_to(resolved_root)`. Otherwise the
   call raises `SecurityError`.

Callers include the read, edit, write and rename tools (`pedacito/tools.py`,
`pedacito/edits.py`, `pedacito/loop.py`, `pedacito/workspace.py`), the plan
checker (`pedacito/plan.py`, which resolves the plan path and every file mention
before Auto or Complex runs), the skill loader (`pedacito/skills.py`), the
checkpoint restore path (`pedacito/checkpoints.py`), and every GUI file route
(`pedacito/gui/server.py`). The plan checker only reads: a traversal, symlink or
junction escape is reported as a definite error and stops the run before the
first task writes.

**A note on `os.path.commonpath`.** Pedacito uses `os.path.commonpath` in
`pedacito/index.py`, but only to work out the project root when you pass several
paths. It does not decide whether access is allowed. It is a lexical comparison and
does not resolve symlinks, so it cannot be the containment check on its own.
Containment uses `Path.resolve()` followed by `is_relative_to`, which is stricter
because symlinks are resolved first.

### Search is bounded by the index

`grep` searches the index's file list, and the other lookup tools work from the same index.
That list is built from the project root and excludes ignored and sensitive paths. A search
query therefore cannot read a file outside the project, even with a pattern such as
`../../`. Scope arguments (`src/`, `tests/*.py`) are matched against indexed paths
only.

### Writes are guarded

- **Protected paths.** The model can never write `.git/`, `.pedacito/`,
  `.pedacito.toml`, `PLAN.md`, `todo.md`, or anything in `protected_paths`.
- **Merge conflicts.** A file with unresolved git conflict markers is not edited.
- **Stale-file check.** The review page records each file's SHA-256 when it is
  built. Applying refuses the write if the file changed on disk in the meantime.
- **Syntax gate.** An edit that breaks a previously clean file is refused before it
  is written. Pedacito uses `ast.parse` for Python and tree-sitter for the other
  supported languages.
- **Snapshot first.** Every write is preceded by a backup and a checkpoint (see
  Pillar 4).
- **Allow-list mode.** `auto_allowlist = true` restricts an autonomous task to the
  files its title names or that the lookup read. New files are always allowed.

### Project config is not trusted by default

A repository can ship a `.pedacito.toml`. That file cannot enable execution on its
own. These settings are ignored from a project file until you read it and run
`pedacito trust PROJECT`:

- Commands and hooks: `hooks`, `mcp_servers`, `auto_verify`, `test_command`,
  `allow_serve`, `allow_lsp`, `lsp_servers`, `lsp_diagnostics_gate`, `load_cmd`,
  `unload_cmd`.
- Capability switches: `allow_shell`, `allow_write`, `allow_browser`,
  `allow_screenshots`, `allow_fetch`, `allow_image_gen`, `enable_mcp`, `allow_mcp`.
- Sandbox settings (`sandbox`, `sandbox_network`, `sandbox_writable`,
  `sandbox_strict`) and the environment scrub (`scrub_env`, `env_passthrough`).
- Endpoint redirection: `base_url`, `api_key`, `api_key_env`. A project cannot send
  your prompts to a different server.
- Repo-code verification switches: `verify_tests`, `verify_dom`, `dom_entry`.
- Anything that widens where data can go: `fetch_allowed_hosts`,
  `browser_eval_hosts`, `shell_allowlist`.

Editing a trusted file revokes the approval. Harmless settings from the same
file still apply, and these fields stay honoured from your user config, from
`--config FILE` and from `PEDACITO_*` environment variables. The starter
written by `pedacito config --init --path <project>/.pedacito.toml` keeps those
lines commented, so trusting it neither warns nor revokes what you granted.

### GUI and network boundary

The GUI server is local, but other websites can still send requests to it. Pedacito
guards it in four ways:

- It binds to `127.0.0.1` and refuses requests with a foreign `Host` header.
- A state-changing request whose `Origin` or `Referer` is not loopback is refused.
  A page open in the same browser cannot drive the server, even if it guesses the port.
- Only JSON `POST` bodies are accepted, and a strict Content-Security-Policy is sent.
- Project files are served only from inside the project root. The `/preview/` route
  refuses `..` segments and escaping symlinks, and frames only from the same origin.

The preview iframes (`/preview/` and `/api/runner/preview/`) are additionally
sandboxed with `sandbox="allow-scripts"` and deliberately **without**
`allow-same-origin`. A previewed page therefore runs in an opaque origin: it can
execute its own scripts, but it cannot read the GUI's storage or call the agent API
as the app itself. A state-changing request from the preview arrives with
`Origin: null`, which the loopback-only POST guard refuses. Project pages are also
served without `'unsafe-eval'` in their CSP (the GUI's own page CSP is unchanged;
inline scripts still run, so a self-contained static page previews normally).
**↗ New tab** opens the page as a normal same-origin document, outside the sandbox;
that is the trusted escape hatch, so use it only for projects you would run locally.

The auto-update check (`pedacito/update.py`) stays inside the same boundary:
the browser never contacts PyPI itself (the CSP's `connect-src 'self'` forbids
it) — the local server makes the one anonymous GET, and the upgrade runs as an
argv list with `shell=False`. `check_for_updates = false` turns the whole
feature off: no request, no subprocess, no banner.

The `fetch` and `web_search` tools (`pedacito/webfetch.py`) add an SSRF guard:

- Only full `http` and `https` URLs are fetched.
- Loopback (`127.0.0.1`, `localhost`, `::1`), link-local and cloud-metadata
  addresses (`169.254.169.254`), and private LAN ranges are refused unless the host is
  named in `fetch_allowed_hosts`.
- The check runs on the **resolved** addresses, so a public name that resolves to a
  private IP (DNS rebinding) is refused. The connection is pinned to the validated
  address, and redirects are re-checked at every hop.

### Tool output is data, not instructions

Output from `run`, `browser`, `fetch`, `mcp`, `git`, `lsp` and `screenshot` is wrapped
as untrusted observation text. Control characters and ANSI escapes are stripped.
Wording that addresses the assistant, such as "ignore previous instructions", "you
are now", or "reveal the system prompt", is flagged in the observation. `pedacito
eval` plants injections to measure how often they are followed.

This is a mitigation. It is not a guarantee, which is why Pillars 3 and 4 exist.

---

## Pillar 3: Least-privilege execution

The goal is that a command the model chooses runs with no more power than it needs,
can be stopped, and cannot reach your credentials or the network.

### Runs as you, never as root

Pedacito never asks for elevated privileges, never calls `sudo`, and never changes
its user ID. Every command, server and test runs with the permissions of the account
that started Pedacito. Do not run it as root. A root-owned process has no
sandbox-free safe path for a model-chosen command.

### The OS sandbox

Commands that the model causes to run (`run`, and the test commands used by
`verify_tests`, `run_test` and the test selector) are wrapped by `pedacito/sandbox.py`:

| Platform | Backend | How it is used |
|---|---|---|
| Linux | `bwrap` (bubblewrap) | Unprivileged user namespaces, no setuid helper |
| macOS | `sandbox-exec` (Seatbelt) | A generated profile applied to the child process |
| Windows | `win_restricted` (Job Object + restricted token) | Built in since v0.4.0; see below |

**Windows sandbox (`win_restricted`, v0.4.0).** `pedacito/sandbox_win.py` runs
the child inside a Windows Job Object and a restricted token at Low integrity
(`ctypes`, loaded lazily behind `os.name == "nt"`; no third-party packages). The
job is set to kill every process when it closes, so a cancelled, timed-out or
crashed command takes its whole process tree with it, and Pedacito waits for the
child before it removes any temporary directory. The Low-integrity label is
applied to the project and temporary directories only, and the relabel walker
skips the guarded `.git`, `.pedacito` and `.pedacito.toml` (and never follows
NTFS junctions or symlinks), so a sandboxed command cannot rewrite a git hook,
the policy file or its own guard rails. Pedacito probes the backend at
start-up with a real no-op child; if the Job Object, restricted token or child
launch fails, the backend reports `unavailable` and is never claimed.
`sandbox.status()` reports `active`, `unavailable` or `off`.

`win_restricted` isolates process lifetime (Job Object) and restricts writes to
the relabeled project and temporary directories (Low Integrity, skipping the
guarded paths above), but it does **not** filter outbound network traffic or
hide user-readable files: outbound network filtering and read-hiding require an
AppContainer on Windows, or the POSIX `bwrap` / `sandbox-exec` backends.
`sandbox = false` turns wrapping off entirely.

**Unattended hard gate (D14).** Unattended `workspace-write` and `danger` modes,
background tasks, and `approval_policy = "never"` with `sandbox` on must run inside
an active OS sandbox. When no backend is active they raise `SandboxRefused` and
the command does not run. Interactive `ask` runs are not blocked by the gate, but
they still use the sandbox when one exists.

The Linux `bwrap` arguments give the following guarantees:

- `--die-with-parent`, `--new-session` and `--unshare-pid`: the child cannot outlive
  Pedacito, cannot take over its terminal, and cannot see other processes.
- The root filesystem is read-only (`--ro-bind / /`). `/tmp` is a private tmpfs.
- The project is the only writable location, along with `sandbox_writable`
  directories.
- `.git`, `.pedacito`, `.pedacito.toml`, `pedacito.toml` and `.pedacitoignore` are
  mounted read-only. A command cannot plant a git hook or rewrite its own guard rails.
- Secret directories (`~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.azure`, `~/.kube`,
  `~/.docker`, `~/.config/gcloud`, `~/.config/gh`, `~/.config/pedacito`) are replaced
  with empty tmpfs mounts. Secret files (`~/.netrc`, `~/.git-credentials`,
  `~/.npmrc`, `~/.pypirc`) are bound to `/dev/null`.
- The network is unshared (`--unshare-net`) unless you set `sandbox_network = true`.

On macOS the Seatbelt profile denies network access, denies file writes except to
the project, a temporary directory and `sandbox_writable`, and denies reads of the
same secret paths.

**Probe before trust.** Pedacito starts a no-op command inside the sandbox once and
caches the result. If the backend cannot start (for example, user namespaces are
disabled, or the process is inside a container that forbids nesting), the command
runs unsandboxed after one notice. Set `sandbox_strict = true` to refuse to run
instead. `pedacito config --doctor` and `/status` show whether the sandbox is
active. A failed command's observation says whether it ran sandboxed, so the model
does not chase a blocked write as if it were a code bug.

### What is and is not sandboxed

| Execution path | Sandboxed | Environment | Timeout |
|---|---|---|---|
| `run` action (model-chosen command) | Yes | Scrubbed allow-list | `shell_timeout` (120 s), `shell_idle_seconds` (10 s) |
| Test commands from `run_test` | Yes | Scrubbed | 15 s ceiling per run |
| `verify_tests` after an edit | Yes | Scrubbed | Test timeout |
| `pedacito serve` (background dev servers) | **No** | Scrubbed | Kept running until stopped |
| `browser`, `lsp`, MCP servers | **No** | Scrubbed | `lsp_timeout` (10 s), MCP discovery 5 s per server; the diagnostics gate uses one 2.9 s budget per symbol batch and reaps a timed-out server |
| Hooks and `auto_verify` (your own commands) | **No** (by design) | Scrubbed for hooks (+ `PEDACITO_EVENT`); `auto_verify` also scrubbed | Hook timeout |
| Server lifecycle `load_cmd` / `unload_cmd` (trusted config) | **No** (by design) | Scrubbed allow-list + `PEDACITO_SERVER_NAME` / `PEDACITO_MODEL_PATH` | 120 s |
| GUI **Run** tab (you click it) | **No** | Credential variables removed (see below) | 15 s default, or the value you set |

Your own hooks and `auto_verify` command are yours, so they are not sandboxed.
Server lifecycle commands are in the same bucket: argv-only, not sandboxed,
not shelled, and given the two documented variables above.

### Process control and timeouts

Commands are started without a shell. `&` and `nohup` are rejected, so a
background process cannot escape the timeout. Long-lived programs go through `serve`,
which records them under `.pedacito/procs/`.

`pedacito/proc.py` enforces two limits on every command:

- A **hard timeout** (wall-clock).
- An **idle timeout** (no output for N seconds), which stops a silent hung process.

When either limit is reached, the process is terminated. On POSIX the whole process
group is killed, so grandchildren cannot keep running. If the child still does not
exit, Pedacito escalates to a forced kill.

Other limits:

- `auto_smoke_timeout` (20 s) bounds the boot check after an autonomous run.
- `dom_timeout` (45 s) bounds one headless page load.
- `serve_wait` (3 s) bounds how long Pedacito waits for a dev server to print its URL.

### Command policy and environment

- **Shell is off by default.** `allow_shell = false` disables `run` and `diagnose`.
  Turning it on is a trust decision, and a project file cannot turn it on.
- **Pattern deny list.** `shell_deny = ["git push*", "rm -rf *"]` refuses matching
  commands even when the executable is allowed.
- **Scrubbed environment.** With `scrub_env = true` (the default), model-chosen commands
  and dev servers receive only `PATH`, `HOME`, locale and toolchain variables. API keys
  and tokens are not inherited. Add what a project needs with
  `env_passthrough = ["DATABASE_URL", "AWS_*"]`.
- **Browser `eval`** runs model-written JavaScript, so it is restricted to the hosts in
  `browser_eval_hosts`.
- **Environment resolution is detect-only.** `env_resolve` reads local markers
  (`.venv`, uv/poetry/pipenv/conda metadata, tox/nox environments,
  pnpm/yarn/bun lockfiles, Cargo/Go manifests) and checks whether an
  interpreter or tool exists. It never sources an activation script, runs
  `uv`/`poetry`/`conda`/`tox`/`nox`/`npx`, installs a package or makes a
  network request. Project-discovered paths pass `security.resolve_safe_path`,
  so a `.venv` symlink/junction that escapes the project is rejected; an
  explicitly set `VIRTUAL_ENV`/`CONDA_PREFIX` remains user state and may live
  outside. `command_env` composes `safety.child_env`, so prepending the
  project's bin directories cannot re-add a variable the scrub removed.
  Resolver-discovered scripts are data: nothing runs because a manifest names
  it, and `allow_shell`/policy/sandbox still gate every actual execution.

The environment policy has three per-surface tiers:

| Tier | Surfaces | Policy |
|---|---|---|
| Allow-list scrub | agent tools, tests, dev servers, browser/LSP/MCP, hooks, `auto_verify`, server `load_cmd`/`unload_cmd` | `safety.child_env`: only `PATH`, `HOME`, locale and toolchain variables; `env_passthrough` adds names or `PREFIX*` patterns; `scrub_env = false` disables the scrub for trusted setups |
| Run-tab deny-list | GUI/TUI **Run** tab | Only credential-shaped names are removed; other ambient variables still pass (see Known limits) |
| Trusted config | hooks, `auto_verify`, server lifecycle commands | Not sandboxed by design. Environment uses the allow-list scrub above; hooks also get `PEDACITO_EVENT`, lifecycle commands get `PEDACITO_SERVER_NAME` / `PEDACITO_MODEL_PATH` |

### The permission broker

Some actions pause the run until you decide: web access, package installs, shell
commands that are not otherwise allowed, reads of sensitive files, untrusted MCP
calls, and `run_test` when `enable_run_test_in_gather = false`. The dialog (GUI),
prompt (TUI) or JSON event (RPC) shows the category, the exact target and the reason.

| Choice | Effect |
|---|---|
| **Allow Once** | Approves this request only. |
| **Allow for This Run** | Approves that kind of request until the run ends. |
| **Always Allow** | Approves it from now on. Sets `enable_web` for web access, or `allow_shell` for shell commands, and saves the choice to your config. |
| **Allow '<pattern>' in this project** | Shell commands only: saves one narrow rule to `shell_allowlist_by_project` for this project root. Matching commands skip the prompt in that project only. Package installs, shells and chained commands are never covered. |
| **Deny & Redirect** | Refuses the request. The model is told to continue with local project files only. |

A request you do not answer, because the dialog times out or you press Stop, is
denied. The CLI has no dialog, so a gated tool reports that it is disabled in the
config instead of waiting for a person who is not there.

Web access is off by default (`enable_web = false`). With it off, `fetch_url` and
`web_search` are offered only when a task names a URL or web terms, and each call asks
first.

### Hardening profile

A conservative configuration for a repository you do not fully trust:

```toml
[universal]
sandbox = true
sandbox_strict = true       # refuse to run instead of running unsandboxed
sandbox_network = false
scrub_env = true
allow_shell = false         # turn on per project, after reading the code
allow_write = false
enable_web = false
security_outbound_mode = "warn"
redact_secrets = true
shell_deny = ["git push*", "rm -rf *", "curl *", "wget *"]
```

---

## Pillar 4: Immutable audit trail and time travel

The goal is that every change an autonomous run makes can be inspected, compared and
reverted, and that the history survives a failed or rolled-back run.

### What is recorded

Every Task, Auto step, New scaffold and manual checkpoint writes a checkpoint to
`<project>/.pedacito/checkpoints/`:

```
.pedacito/checkpoints/
  index.json              checkpoint tree: ids, parents, labels, modes, timestamps, head
  snapshots/<id>/         full copy of the tracked source files at that checkpoint
```

Each checkpoint records its id (for example `cp_0003_a1b2c3`), its parent id, its
mode, a label, a timestamp, and `files_changed`. The modes are `task`, `auto`, `new`,
`manual`, and `restore`. The tracked set excludes build and cache directories such as
`.git`, `node_modules`, `.venv` and `dist`, and skips binary artifacts over 2 MB.

Checkpoints are linked by parent, so the file is a tree. A create after a restore
starts a new branch from the restored point instead of overwriting history.

Some logs have their own audit trail. `sessions/<stamp>.md` records what was asked, what
was looked up, and what changed. `events/<stamp>.jsonl` records each step, model call,
fold and verification. `log.txt` is a timestamped trace. All of them mask secrets.

### Inspect before you undo

`pdc` is a short alias for `pedacito`, and both accept the same commands.

```bash
pedacito checkpoints PROJ                 # the tree: ids, parents, modes, timestamps
pedacito checkpoints PROJ --json          # the same tree for scripts
```

The GUI **Checkpoints / Undo** button shows the timeline with a unified diff for each
file, and offers one-click restore for a file or the whole workspace. **⑂ Branch Prompt
From Here** restores the workspace to a checkpoint (current files are saved in a safety
checkpoint first) and prepares the prompt box so the next task branches from there. The TUI has
`/checkpoints`, `/undo` and `/checkout ID`. Each diff comes from
`difflib.unified_diff` over the snapshot and the current file, so you see exactly what
an autonomous run changed before you accept it.

### Roll back

```bash
pedacito undo PROJ                        # state immediately before the last run
pedacito checkout cp_0003_a1b2c3 PROJ     # restore the whole workspace to a checkpoint
pedacito checkout cp_0003 PROJ --file app.py   # restore one file only
pedacito restore PROJ                     # restore the newest backup snapshot
pedacito restore PROJ --snapshot NAME     # restore a specific snapshot
pedacito restore PROJ --list              # file-level backups from before checkpoints existed
pedacito restore PROJ --dry-run           # preview a restore without writing
```

Rules that keep rollback safe:

- **Dirty changes are captured first.** Before a restore, any uncommitted change in the
  working tree is saved as a `restore` checkpoint. A restore can always be undone.
- **Restoring makes the target the active head.** The next run branches from it, so
  the history is kept.
- **A whole-workspace restore is complete.** It also removes tracked files that did not
  exist in the target checkpoint. A single-file restore only touches that file.
- **Undo falls back gracefully.** `pedacito undo` prefers the active checkpoint's parent. A
  project created before checkpoints existed falls back to its newest backup snapshot.
- **File-level restores are conservative.** `pedacito restore` leaves files created after the
  snapshot in place. Add `--delete-created` to remove them too; directories left empty by the
  removal are removed as well. The run's created files are listed in a sidecar
  `backups/<stamp>.created`.

Inside a run, the model can rewind its own work with the `undo` action, which reverts
what that task's writes did so far. The green ratchet in autonomous mode rolls a step
that still fails after its repair rounds back to the last passing snapshot, and removes
files the step created.

### Immutability: what it does and does not guarantee

Checkpoints are **append-only by design and written atomically**. The index is written
to a temporary file and moved into place with `os.replace`, so a crash cannot leave a
half-written index. Snapshots are not rewritten after they are taken. Restores create
new checkpoints rather than editing old ones.

They are **not tamper-evident**. The index has no hash chain or signature, and
snapshots are plain files. Anyone who can write to `.pedacito/` can edit the history.
If you need an audit record that resists tampering, commit your work to git, or copy
`.pedacito/checkpoints/` to storage the agent cannot reach.

### Reset and retention

`pedacito prune` keeps the newest N sessions, backups and event files. `pedacito reset`
clears runtime state, including checkpoints, in one step. It leaves your code,
`.pedacito.toml` and `.pedacitoignore` alone. `reset --dry-run` shows what would go, and
live git worktrees under `.pedacito/worktrees/` make the reset refuse unless you pass
`--force`.

---

## Git commits, pull requests, undo and scratch files (v0.3.3)

- **Secret pre-flight before every commit.** `pedacito commit` (and the TUI
  `/git commit`, and the GUI Git drawer) scans the staged diff with the same
  pattern set as `security.scan_text_for_secrets` before it runs `git commit`.
  A match refuses the commit and names the file and the pattern. Only
  `--allow-secrets` overrides it, and that override is explicit on the command
  line, never saved to config.
- **Secret pre-flight before every pull request.** `pedacito pr` scans the branch
  diff the same way before it drafts the title and body, so a credential cannot
  reach a PR description either. `--dry-run` prints the draft and never calls `gh`.
- **No automatic push and no force-push.** Nothing in `gitops` runs `git push`
  or `--force`. `commit` leaves a local commit, and `pr` opens a pull request for
  a branch you have already published. Git calls are argv-only, with no shell.
- **`undo` checks git HEAD.** When the repository's HEAD has moved since the
  checkpoint was taken (for example, you committed in between), `pedacito undo`
  refuses to rewind the working tree and asks for `--force`, so a commit is not
  silently undone. Checkpoints created before a divergent edit are taken first.
- **`.pedacito/scratch/` is contained and read-only to the model.** Large tool
  output is written there as `*.log` files. The model can `read` and `grep`
  only those direct `*.log` files, through the same path containment as any
  other read. It cannot write, rename or traverse out of the directory.
- **ACP paths are contained.** In `pedacito acp`, a `@file` mention or resource
  block goes through `security.resolve_safe_path`, so traversal and symlink
  escapes are refused. The project is bound at `session/new`, and a foreign
  working directory is refused rather than read from.

## Trajectory export

Trajectory export is an **opt-in** dataset feature, off by default
(`record_trajectories = false`). It is the one place Pedacito writes a
**full-prompt** record: `pedacito/trajectories.py` captures the messages the
harness actually sent and received during a run — system, user and assistant
turns, model tool calls and tool results — and writes one JSON file per task
under `.pedacito/trajectories/`:

```text
.pedacito/trajectories/20261120T101500Z_4f2a9c1d.json
```

That is deliberately a different privacy trade from `replay.py`, which stores
prompts as a **hash plus a short redacted preview and never the full text**
(see `--record` / `--replay` in `pdc ask --help`). A trajectory is intended for
SFT/GRPO dataset filtering, so it needs the real transcript. Treat a
trajectory file like the source code it quotes.

Enable it for one run or for a project:

```bash
pedacito ask . -t "fix the retry loop" --apply --save-trajectory
```

```toml
# user config, .pedacito.toml, or a profile -- a local logging flag,
# honoured like log_sessions without `pedacito trust`
record_trajectories = true
```

What protects the file:

- **Same redaction pipeline as logs and sessions.** Every string value passes
  `safety.redact` before serialisation, regardless of `redact_secrets`; the
  serialised transcript is then scanned with
  `security.scan_text_for_secrets`. A residual finding is masked in place; if
  a finding survives or masking would break the JSON, the write is refused
  with a warning and no file is created. A detected secret is never persisted.
- **Local files only.** There is no network call and no telemetry; the files
  live under the project's `.pedacito/`, which `workspace.py` adds to
  `.gitignore` on first run.
- **Metadata is not a second copy of the task.** It stores a SHA-256
  `task_hash`, never the task text, alongside `verify_passed`,
  `harness_version`, `edit_format`, `model_id`, `provider`, `started_at`,
  `elapsed_s` and token totals.
- **Capture cannot break a run.** The hook in `loop.run_task` is
  exception-safe: a full disk, permissions error or malformed message leaves
  the task unaffected (the warning goes to stderr).

You own the files. They are plain JSON; delete `.pedacito/trajectories/`
whenever you want, and do not copy a dataset into a repository or a shared
service without reviewing it first. Secret detection is pattern-based, so a
credential with no known shape can still survive redaction — the same
[known limit](#known-limits-and-gaps) applies here.

---

## Summary checklist

- [ ] Hosted profiles use `security_outbound_mode = "warn"` or `"redact"`.
- [ ] Keys live in your user config or in environment variables (`api_key_env`), never in the repository.
- [ ] `.env`, key files and credential files are in `.gitignore`.
- [ ] `sandbox = true`, and `sandbox_strict = true` where a silent fallback is unacceptable.
- [ ] `sandbox_network = false` unless a test genuinely needs the network.
- [ ] `allow_shell`, `allow_write` and `enable_web` are off until you need them.
- [ ] You ran `pedacito trust` only on projects whose `.pedacito.toml` you read.
- [ ] Before a large autonomous run, you know how to run `pedacito checkpoints` and `pedacito undo`.
- [ ] Checkpoints you care about are also in git or another store the agent cannot write.
- [ ] `record_trajectories` is on only where you are prepared to keep the full prompts locally.
- [ ] You never run Pedacito as root.

## Sensitive artifacts, policy trust and approvals (v0.4.0)

**Sensitive-artifact hygiene (D10).** Credential material never leaves the project through
Pedacito's own stores. The shared gate, `security.check_sensitive_write`, refuses `.env`
variants, `id_rsa*`, `*.pem`, `*.key`, `*.p12`, `credentials.json` and the other names in
`security.SENSITIVE_NAMES` and `SENSITIVE_PATTERNS`, and the `~/.ssh`-style directories. It
applies in both engines: the two-phase engine (`edits.apply_edits`, `tools.tool_write`) and
the interleaved engine (`workspace_overlay.VirtualWorkspace`) refuse the same writes with the
same message. The following never carry the bytes of a sensitive file:

- checkpoints, the ratchet snapshots and the backups (`fileio.backup_copy` refuses and
  creates nothing);
- dry-run mirrors (`VirtualWorkspace.exec_root`);
- review diffs, review pages and GUI JSON previews. Embedded tokens are redacted as
  `[REDACTED]` in the review HTML and rows, but the text that gets applied is unchanged.

Template and public-key names stay usable: `.env.example`, `.env.sample`, `.env.template`
and `id_rsa.pub` are not treated as secrets.

**Policy file trust (`.pedacito/policy.toml`).** A project's policy file can narrow what the
agent may do, but it cannot widen it. The file is read only when the project is trusted
(`pedacito trust PROJECT`). An untrusted file is ignored
entirely, so its `permission_mode` or `approval_policy` never takes effect. The policy
engine (`pedacito/policy.py`) resolves every capability against the permission mode and
the approval policy, and a subagent gets a `child_policy()` that can only intersect with
its parent, never add to it.

**Permission modes and approvals.** `permission_mode` is one of `plan` (read-only),
`ask` (every mutating capability asks), `accept-edits` (file writes are automatic, other
capabilities ask), `workspace-write` (writes and runs are automatic inside the sandbox)
and `danger` (everything is automatic). `approval_policy` is `ask`, `never` (a prompt is
denied at once, with a `permission_decision` event) or `auto-safe` (`pytest`, `ruff`,
`git status` and `git diff` are approved, everything else asks). Background and headless sessions are denied
deterministically instead of waiting for a person.

## Known limits and gaps

These are real boundaries. Read them before you rely on Pedacito for untrusted code.

- **No entropy scoring.** Secret detection is pattern-based, with length and placeholder
  filters. A high-entropy secret that matches no known shape and sits under an unrelated
  name is not detected.
- **The GUI Run tab is not sandboxed, and its environment scrub is a deny-list.**
  `runner.run_project_command` removes variables whose names match `*_API_KEY`, `*SECRET*`,
  `*TOKEN*`, `*PASSWORD*`, `*ACCESS_KEY*`, `*CREDENTIAL*`, `GITHUB_TOKEN` or `PYPI_API_KEY`,
  matched case-insensitively. Any credential under a name outside those patterns still
  reaches the program you start from that tab. Examples are a `DATABASE_URL` that embeds a
  password, or a vendor variable without a conventional suffix. The deny-list is also not a
  sandbox. Use `run` with the sandbox, or keep untrusted projects off the Run tab.
- **Server lifecycle commands are not sandboxed.** `load_cmd` / `unload_cmd` from a
  trusted config run as an argv with the allow-list-scrubbed environment plus
  `PEDACITO_SERVER_NAME` / `PEDACITO_MODEL_PATH`. They are not shelled, but they are
  also not sandboxed, and `scrub_env = false` hands them the ambient environment.
  Keep that switch off in configs you have not read.
- **Dev servers, browser, LSP and MCP servers are not sandboxed.** They get a scrubbed
  environment, protected paths, allow-lists and trust prompts. A server you configure is a
  program you chose to run.
- **Windows sandbox is process-level, not a container.** `win_restricted` uses a Job Object
  and a Low-integrity restricted token. It does not isolate the file system the way a VM or
  AppContainer does, so a command can still read what the user can read, and it does not
  filter outbound network traffic (that needs an AppContainer). Where the backend
  is unavailable, interactive commands run unsandboxed after one notice, and unattended runs
  are refused (see the unattended hard gate above).
- **Container and CI environments may disable the sandbox.** If user namespaces are off,
  Pedacito falls back as described in Pillar 3.
- **Checkpoints are not tamper-evident.** See Pillar 4.
- **Path containment is only as good as the tools that call it.** Every built-in file tool
  uses `resolve_safe_path`. A custom hook or an external MCP server that writes files
  directly is outside that boundary.
- **The sandbox is defence in depth, not a promise against a determined adversary or a
  kernel bug.**

## Reporting a vulnerability

Please open a private security advisory on the GitHub repository, or email
contact@pedacito.dev, rather than a public issue. Include the Pedacito version
(`pip show pedacito`), your platform, the steps to reproduce, and whether the
sandbox was active (`pedacito config --doctor`).
