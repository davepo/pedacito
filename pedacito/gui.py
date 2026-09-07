"""A small local web GUI: pick a project, check the model, type a task, and
review the diff inline -- one page, one local server, no new dependencies.

Usage
-----
    pedacito gui                    # opens http://127.0.0.1:<port>/ in a browser
    pedacito gui --port 8756        # fixed port instead of an OS-chosen one
    pedacito gui --no-open          # print the URL instead of opening a browser

Design
------
This reuses the same local-server pattern `review.py` already established
(ThreadingHTTPServer bound to 127.0.0.1 only, since this endpoint writes
files and has no business on a network interface) rather than introducing a
second GUI toolkit. The review page itself is unmodified: `review.build_page`
is served at GET /review and loaded in an <iframe>, so its existing
Apply/Cancel JavaScript -- which POSTs to "/apply" and "/cancel" as relative
paths -- lands on *this* server's routes instead of spinning up a second
one. One process, one port, one warm Agent/Index/client, like `pedacito
chat` but driven by HTTP requests instead of stdin.

Only one project is "open" at a time. Switching directories or profiles
tears down and reloads the warm state rather than juggling several -- the
same tradeoff `chat` makes implicitly by being one process per invocation.

Long-running work (opening a project, building the index, checking the
model, running a task) happens on a background thread guarded by a single
"busy" flag, so the HTTP handler never blocks and the page can poll
GET /state for progress instead of hanging on a slow local-model call.
Detailed step-by-step progress still goes to the terminal `pedacito gui`
was launched from (the same prints `gather()`/`answer()` already produce) --
piping that into the browser is a reasonable fast-follow, not done here.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from . import cli as cli_mod
from . import index as index_mod
from . import review as review_mod
from . import workspace
from .agent import Agent
from .config import USER_CONFIG, ConfigError
from .llm import LLMError, LMStudio

RECENT_FILE = USER_CONFIG.parent / "gui_recent.json"
MAX_RECENT = 8


# --------------------------------------------------------------------------
# Reusing cli.py's internals
# --------------------------------------------------------------------------
def _args_for(root_str: str, profile: str | None) -> SimpleNamespace:
    """Build a stand-in for the argparse Namespace that cli.py's internal
    helpers (_config/_matcher/_root/_refresh) expect, so the GUI can reuse
    them unmodified instead of re-implementing config loading, file
    selection, and index staleness handling."""
    return SimpleNamespace(
        paths=[root_str], root=root_str, config=None, profile=profile or None,
        exclude=[], include=[], ext=[], no_refs=False, no_gitignore=False,
        url=None, model=None, steps=None,
    )


# --------------------------------------------------------------------------
# Recent-projects list
# --------------------------------------------------------------------------
def _load_recent() -> list[str]:
    """Return the most-recently-opened project paths, newest first."""
    try:
        data = json.loads(RECENT_FILE.read_text())
        return [str(p) for p in data][:MAX_RECENT]
    except (OSError, ValueError):
        return []


def _remember_recent(root: str) -> None:
    """Move `root` to the front of the recent-projects list, persisted
    alongside the user config so it survives between GUI sessions."""
    recent = [root] + [r for r in _load_recent() if r != root]
    try:
        RECENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        RECENT_FILE.write_text(json.dumps(recent[:MAX_RECENT]))
    except OSError:
        pass  # a lost recents list is not worth failing the request over


# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------
class State:
    """Everything held between HTTP requests. One instance, one active
    project -- see the module docstring for why."""

    def __init__(self):
        self.root: Path | None = None
        self.args: SimpleNamespace | None = None
        self.cfg = None
        self.client: LMStudio | None = None
        self.idx = None
        self.agent: Agent | None = None

        self.busy = False
        self.busy_what = ""
        self.error = ""

        self.model_checked = False
        self.model_status = ""

        # The most recent run, staged for review or already applied.
        self.last_task = ""
        self.last_reply = ""
        self.last_results: list = []
        self.last_backup_dir: Path | None = None
        self.last_stamp = ""

        self.review_page = ""
        self.review_nonce = ""
        self.review_diffs: list = []
        self.review_outcome = ""     # "", "applied", "cancelled"
        self.review_detail = ""

    def clear_review(self) -> None:
        """Discard whatever review is currently staged, without touching
        disk. Called both on /cancel and whenever a new project or task
        makes the old review meaningless."""
        self.review_page = ""
        self.review_nonce = ""
        self.review_diffs = []
        self.review_outcome = ""
        self.review_detail = ""

    def status(self) -> dict:
        """The JSON payload the page polls to update itself."""
        return {
            "root": str(self.root) if self.root else "",
            "profile": self.cfg.active_profile if self.cfg else "",
            "profiles": sorted(self.cfg.profiles) if self.cfg else [],
            "model": self.cfg.model if self.cfg else "",
            "apply_default": bool(self.cfg and not self.cfg.review),
            "has_index": self.idx is not None,
            "index_files": len(self.idx.files) if self.idx else 0,
            "files": sorted(self.idx.files) if self.idx else [],
            "busy": self.busy,
            "busy_what": self.busy_what,
            "error": self.error,
            "model_checked": self.model_checked,
            "model_status": self.model_status,
            "has_review": bool(self.review_diffs),
            "review_outcome": self.review_outcome,
            "review_detail": self.review_detail,
            "last_task": self.last_task,
            "recent": _load_recent(),
        }


STATE = State()


def _spawn(fn, what: str) -> bool:
    """Run `fn` on a background thread, refusing to start a second one
    while the model, the index, or the agent are already busy -- they are
    single-user resources and running two tasks against them at once would
    corrupt whichever finishes first."""
    if STATE.busy:
        return False
    STATE.busy, STATE.busy_what, STATE.error = True, what, ""

    def worker():
        try:
            fn()
        except (ConfigError, LLMError) as e:
            STATE.error = str(e)
        except Exception as e:  # surface it rather than hang the UI forever
            STATE.error = f"{type(e).__name__}: {e}"
        finally:
            STATE.busy, STATE.busy_what = False, ""

    threading.Thread(target=worker, daemon=True).start()
    return True


# --------------------------------------------------------------------------
# Actions (run on the background thread via _spawn)
# --------------------------------------------------------------------------
def _open_project(path: str, profile: str) -> None:
    """Resolve a directory as the active project: load its config (which is
    what makes the profile dropdown meaningful -- profiles live in a config
    resolved relative to root), ensure workspace scaffolding, and load
    (never silently build) its index."""
    p = Path(path).expanduser()
    if not p.is_dir():
        raise ConfigError(f"not a directory: {p}")
    args = _args_for(str(p), profile)
    cfg = cli_mod._config(args)
    root = cli_mod._root(args)
    workspace.ensure(root, cfg.manage_gitignore, cfg.create_ignore_file)

    idx = index_mod.Index.load(root)
    if idx is not None:
        idx = cli_mod._refresh(idx, args, cfg, root, verbose=False)

    STATE.root, STATE.args, STATE.cfg = root, args, cfg
    STATE.client = LMStudio(cfg)
    STATE.idx = idx
    STATE.agent = Agent(idx, STATE.client, cfg) if idx is not None else None
    STATE.model_checked, STATE.model_status = False, ""
    STATE.clear_review()
    _remember_recent(str(root))


def _peek_profiles(path: str) -> list[str]:
    """Read just the profile names for a directory, without opening it as
    the active project. Cheap (TOML parsing only, no LLM/index work), so
    it runs synchronously in the request handler rather than through
    _spawn -- it lets the profile dropdown fill in *before* you commit to
    Open, instead of forcing a blind first open on the default profile."""
    p = Path(path).expanduser()
    if not p.is_dir():
        raise ConfigError(f"not a directory: {p}")
    args = _args_for(str(p), None)
    cfg = cli_mod._config(args)
    return sorted(cfg.profiles)


def _build_index() -> None:
    """Build (or rebuild) the index with summaries -- the same work
    `pedacito index` does from the CLI. Kept as an explicit, visible action
    rather than something that fires silently on first Run, since it makes
    one LLM call per summarised unit and is not free."""
    if STATE.root is None:
        raise ConfigError("no project selected")
    idx = index_mod.build(
        STATE.args.paths, STATE.cfg, client=STATE.client, root=STATE.root,
        summarise=True, matcher=cli_mod._matcher(STATE.args, STATE.root))
    STATE.idx = idx
    STATE.agent = Agent(idx, STATE.client, STATE.cfg)


def _check_model() -> None:
    """Verify the server is reachable and the configured model actually
    generates -- the same two checks as `pedacito config --check`."""
    if STATE.client is None:
        raise ConfigError("no project selected")
    loaded = STATE.client.check(warn=False)
    STATE.client.warmup()
    STATE.model_checked = True
    STATE.model_status = f"reachable \u2014 models available: {loaded}"


def _run_task(task: str, apply_now: bool) -> None:
    """Gather source, get an answer or edits, and either stage a review
    (the default) or write immediately if `apply_now` was checked.

    `stream=False` on the answer call is deliberate: LMStudio.chat's
    streaming mode writes tokens straight to this process's stdout, which
    is meaningless in an HTTP handler with no terminal on the other end.
    """
    if STATE.agent is None:
        raise ConfigError("no index yet -- build the index first")
    agent, cfg, root = STATE.agent, STATE.cfg, STATE.root
    when = workspace.stamp()
    reviewing = not apply_now
    backup_dir = workspace.new_backup_dir(root, when) if cfg.backup else None

    gathered = agent.gather(task, verbose=True)
    reply = agent.answer(task, gathered, stream=False)
    results, reply = agent.apply(
        reply, dry_run=reviewing, task=task, gathered=gathered,
        backup_dir=None if reviewing else backup_dir)

    STATE.last_task, STATE.last_reply = task, reply
    STATE.last_results, STATE.last_backup_dir, STATE.last_stamp = (
        results, backup_dir, when)
    applied = False

    STATE.clear_review()
    if reviewing:
        diffs = review_mod.diffs_from_results(results)
        STATE.review_diffs = diffs
        if diffs:
            STATE.review_nonce = review_mod.new_nonce()
            STATE.review_page = review_mod.build_page(
                diffs, task, cfg.model, cfg.active_profile,
                STATE.review_nonce, interactive=True, stamp=when)
            review_mod.save(root, STATE.review_page, when)
    else:
        applied = any(r.ok for r in results)

    if applied:
        idx = STATE.idx
        for r in results:
            if r.ok:
                idx = index_mod.reindex_file(idx, r.file, cfg)
        STATE.idx, agent.index = idx, idx

    if cfg.log_sessions:
        workspace.write_session(root, task, getattr(agent, "steps_log", []),
                                reply, results, backup_dir, when)


def _apply_review() -> tuple[bool, str]:
    """Write the currently staged review's files to disk (the review
    page's Sign off button, handled on this server rather than a second
    one -- see the module docstring)."""
    if not STATE.review_diffs:
        return False, "No review is staged."
    ok, detail = review_mod.write_files(STATE.root, STATE.review_diffs,
                                        STATE.last_backup_dir)
    if ok:
        STATE.review_outcome, STATE.review_detail = "applied", detail
        idx = STATE.idx
        for fd in STATE.review_diffs:
            idx = index_mod.reindex_file(idx, fd.rel, STATE.cfg)
        STATE.idx = idx
        if STATE.agent is not None:
            STATE.agent.index = idx
    return ok, detail


# --------------------------------------------------------------------------
# Page: CSS + HTML + JS (self-contained, no CDN -- same reasoning as
# review.py: this opens on a machine that may have no other browser tab
# open, and should not depend on anything reachable over the network).
# --------------------------------------------------------------------------
CSS = """
:root {
  --bg: #f7f7f5; --panel: #ffffff; --line: #e4e2dc; --text: #2a2822;
  --muted: #86816f; --accent: #b5651d; --ok: #3c6e47; --err: #a4342a;
  font: 15px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif;
}
/* Sage: light, muted eucalyptus green -- soft and easy on the eyes for
   long sessions, warm-neutral text stays high-contrast on the pale bg. */
html[data-theme="sage"] {
  --bg: #eef0ea; --panel: #ffffff; --line: #d7ddd0; --text: #283228;
  --muted: #74806d; --accent: #5b7a52; --ok: #3f8f6e; --err: #b1543a;
}
/* Mocha: light, warm neutral (the Pantone "Mocha Mousse" family of
   trending warm-taupe palettes) -- cream background, cocoa text. */
html[data-theme="mocha"] {
  --bg: #f3ece4; --panel: #fffdfa; --line: #e2d2c2; --text: #3a2c22;
  --muted: #8a7360; --accent: #a56a3f; --ok: #4f7a52; --err: #b1493a;
}
/* Clay: dark, warm terracotta -- deep brown bg, clay-orange accent. */
html[data-theme="clay"] {
  --bg: #221a16; --panel: #2c221c; --line: #4a362c; --text: #f1e6dc;
  --muted: #b9997f; --accent: #c9703a; --ok: #8fbf8a; --err: #e5776a;
}
/* Nightshade: dark, cool plum/navy with a coral accent for contrast. */
html[data-theme="nightshade"] {
  --bg: #1b1a24; --panel: #242232; --line: #3a3650; --text: #e9e6f2;
  --muted: #a79cc0; --accent: #c96a5a; --ok: #7ec98a; --err: #e2726a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); }
header {
  padding: 16px 24px; border-bottom: 1px solid var(--line); background: var(--panel);
  display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
}
header h1 { margin: 0; font-size: 17px; }
header .sub { color: var(--muted); font-weight: normal; }
header select { font-size: 13px; }
main {
  max-width: 1520px; margin: 0 auto; padding: 20px 24px 60px;
  display: grid; grid-template-columns: 1fr 300px; gap: 16px; align-items: start;
}
main .col { min-width: 0; }
@media (max-width: 820px) { main { grid-template-columns: 1fr; } }
section.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 16px 18px; margin-bottom: 16px;
}
aside.card { position: sticky; top: 20px; max-height: calc(100vh - 40px); display: flex; flex-direction: column; }
aside.card h2 { flex: 0 0 auto; }
#fileSearch {
  width: 100%; margin-bottom: 8px; padding: 1px 6px; height: 1.9em;
  font-size: 12.5px; line-height: 1.9em; flex: 0 0 auto; min-width: 0;
}
#fileList { list-style: none; margin: 0; padding: 0; overflow-y: auto; flex: 1 1 auto; font: 12.5px/1.6 ui-monospace, "SF Mono", Menlo, monospace; }
#fileList li { padding: 2px 4px; border-radius: 4px; color: var(--text); word-break: break-all; }
#fileList li:hover { background: var(--bg); }
#fileCount { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
section.card h2 { margin: 0 0 10px; font-size: 13px; text-transform: uppercase;
  letter-spacing: .04em; color: var(--muted); }
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
input[type=text], select, textarea {
  font: inherit; padding: 7px 9px; border: 1px solid var(--line); border-radius: 5px;
  background: #fff; color: var(--text);
}
input[type=text] { flex: 1 1 260px; min-width: 200px; }
textarea { width: 100%; min-height: 84px; resize: vertical; }
button {
  font: inherit; padding: 7px 14px; border-radius: 5px; border: 1px solid var(--line);
  background: #fff; cursor: pointer;
}
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button.danger { background: transparent; border-color: var(--err); color: var(--err); }
button.danger:hover { background: var(--err); color: #fff; }
button:disabled { opacity: .5; cursor: default; }
label.chk { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 13px; }
.status { display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; color: var(--muted); margin-top: 10px; }
.status b { color: var(--text); }
.busy { color: var(--accent); }
.busy::before { content: "\\25CF"; margin-right: 6px; animation: pulse 1s infinite; }
@keyframes pulse { 50% { opacity: .25; } }
.error { color: var(--err); font-size: 13px; margin-top: 8px; }
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
iframe#reviewFrame { width: 100%; height: 70vh; border: 1px solid var(--line);
  border-radius: 8px; background: #fff; }
.placeholder { color: var(--muted); text-align: center; padding: 60px 0; }
"""

JS = """
var $ = function (id) { return document.getElementById(id); };

function post(path, body) {
  return fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}) }).then(function (r) { return r.json(); });
}

var lastReviewKey = '';
var lastFilesKey = '';
var lastState = {};

function fillProfiles(list, wanted) {
  var sel = $('profile');
  var cur = wanted !== undefined ? wanted : sel.value;
  sel.innerHTML = '<option value="">(default)</option>' +
    list.map(function (p) {
      return '<option value="' + p + '"' + (p === cur ? ' selected' : '') + '>' + p + '</option>';
    }).join('');
}

function renderFiles(files) {
  var key = files.join('\\n');
  if (key === lastFilesKey) return;
  lastFilesKey = key;
  $('fileCount').textContent = files.length + ' file' + (files.length === 1 ? '' : 's');
  var ul = $('fileList');
  ul.innerHTML = '';
  var frag = document.createDocumentFragment();
  files.forEach(function (f) {
    var li = document.createElement('li');
    li.textContent = f;
    li.dataset.name = f.toLowerCase();
    frag.appendChild(li);
  });
  ul.appendChild(frag);
  filterFiles();
}

function filterFiles() {
  var q = $('fileSearch').value.trim().toLowerCase();
  Array.prototype.forEach.call($('fileList').children, function (li) {
    li.hidden = q !== '' && li.dataset.name.indexOf(q) === -1;
  });
}

function render(s) {
  lastState = s;
  if (document.activeElement !== $('path')) { $('path').value = s.root || $('path').value; }
  var recent = $('recent');
  recent.innerHTML = '<option value="">recent projects\\u2026</option>' +
    s.recent.map(function (r) { return '<option value="' + r.replace(/"/g, '&quot;') + '">' + r + '</option>'; }).join('');

  // Once a project is open, its config is the authority on which profiles
  // exist. Before that, leave whatever /peek populated alone -- s.profiles
  // is empty pre-open and would otherwise wipe out the picker.
  if (s.root) { fillProfiles(s.profiles, s.profile); }

  $('modelLbl').textContent = s.model || '\\u2014';
  $('indexLbl').textContent = s.has_index ? (s.index_files + ' files indexed') : 'no index yet';
  $('buildBtn').hidden = s.has_index;
  $('modelStatus').textContent = s.model_checked ? s.model_status : '';
  renderFiles(s.files || []);

  var busy = s.busy;
  ['openBtn', 'buildBtn', 'checkBtn', 'runBtn'].forEach(function (id) { $(id).disabled = busy || !s.root && id !== 'openBtn'; });
  $('runBtn').disabled = busy || !s.has_index;
  $('busyLbl').textContent = busy ? s.busy_what + '\\u2026' : '';
  $('busyLbl').className = busy ? 'busy' : '';
  $('errLbl').textContent = s.error || '';

  var key = s.last_task + '|' + s.has_review + '|' + s.review_outcome;
  if (s.has_review && key !== lastReviewKey) {
    $('reviewWrap').innerHTML = '<iframe id="reviewFrame" src="/review?t=' + Date.now() + '"></iframe>';
  } else if (!s.has_review && key !== lastReviewKey) {
    $('reviewWrap').innerHTML = '<p class="placeholder">Run a task to see a review here.</p>';
  }
  lastReviewKey = key;
}

function poll() {
  if (stopped) return;
  fetch('/state').then(function (r) { return r.json(); }).then(render).finally(function () {
    if (!stopped) { setTimeout(poll, 1200); }
  });
}
poll();

$('openBtn').addEventListener('click', function () {
  var path = $('path').value.trim();
  if (!path) return;
  post('/project', { path: path, profile: $('profile').value });
});
$('recent').addEventListener('change', function () {
  if (this.value) { $('path').value = this.value; peekTimer = setTimeout(schedulePeek, 0); }
});

// Populate the profile dropdown for whatever directory is typed, before
// Open is ever clicked -- so a profile can be picked up front instead of
// opening once on the default profile and again after switching.
var peekTimer = null;
var peekedPath = '';
function schedulePeek() {
  clearTimeout(peekTimer);
  peekTimer = setTimeout(function () {
    var path = $('path').value.trim();
    if (!path || path === peekedPath) return;
    post('/peek', { path: path }).then(function (res) {
      if (res.ok) { peekedPath = path; fillProfiles(res.profiles, $('profile').value); }
    });
  }, 500);
}
$('path').addEventListener('input', schedulePeek);

$('profile').addEventListener('change', function () {
  // Only an already-open project should reopen on profile change; before
  // that, this just stages the choice for the next Open click.
  if (lastState.root) {
    var path = $('path').value.trim();
    if (path) { post('/project', { path: path, profile: this.value }); }
  }
});
$('fileSearch').addEventListener('input', filterFiles);
$('buildBtn').addEventListener('click', function () { post('/build_index', {}); });
$('checkBtn').addEventListener('click', function () { post('/check_model', {}); });
$('runBtn').addEventListener('click', function () {
  var task = $('task').value.trim();
  if (!task) return;
  post('/run', { task: task, apply: $('applyNow').checked });
});

var THEME_KEY = 'pedacito_theme';
var themeSel = $('theme');
themeSel.value = localStorage.getItem(THEME_KEY) || 'parchment';
themeSel.addEventListener('change', function () {
  if (this.value === 'parchment') { document.documentElement.removeAttribute('data-theme'); }
  else { document.documentElement.setAttribute('data-theme', this.value); }
  try { localStorage.setItem(THEME_KEY, this.value); } catch (e) {}
});

var stopped = false;
$('exitBtn').addEventListener('click', function () {
  if (!confirm('Stop the pedacito gui server? You will need to run "pedacito gui" again to reopen it.')) return;
  stopped = true;
  post('/shutdown', {}).catch(function () {}).then(function () {
    document.body.innerHTML =
      '<div style="padding:80px 20px;text-align:center;font:16px sans-serif;color:var(--muted)">' +
      'pedacito gui has stopped. You can close this tab.</div>';
  });
});
"""

PAGE_HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pedacito</title>
<style>{CSS}</style>
<script>(function(){{try{{var t=localStorage.getItem('pedacito_theme');
  if(t&&t!=='parchment'){{document.documentElement.setAttribute('data-theme',t);}}
}}catch(e){{}}}})();</script>
</head>
<body>
<header>
  <h1>pedacito <span class="sub">local control panel</span></h1>
  <div class="row" style="flex-wrap:nowrap">
    <select id="theme" title="color scheme">
      <option value="parchment">Parchment</option>
      <option value="sage">Sage</option>
      <option value="mocha">Mocha</option>
      <option value="clay">Clay</option>
      <option value="nightshade">Nightshade</option>
    </select>
    <button class="danger" id="exitBtn" title="stop the pedacito gui server">Exit</button>
  </div>
</header>
<main>
<div class="col">

<section class="card">
  <h2>Project</h2>
  <div class="row">
    <input type="text" id="path" placeholder="/path/to/your/project">
    <select id="recent"></select>
    <select id="profile"></select>
    <button class="primary" id="openBtn">Open</button>
  </div>
  <div class="status">
    <span>model: <b id="modelLbl">\u2014</b></span>
    <span>index: <b id="indexLbl">\u2014</b></span>
    <button id="buildBtn" hidden>Build index</button>
    <button id="checkBtn">Check model</button>
    <span id="modelStatus"></span>
  </div>
  <div class="status"><span id="busyLbl"></span></div>
  <div class="error" id="errLbl"></div>
</section>

<section class="card">
  <h2>Ask</h2>
  <textarea id="task" placeholder="Describe the change you want\u2026"></textarea>
  <div class="row" style="margin-top:8px">
    <button class="primary" id="runBtn">Run</button>
    <label class="chk"><input type="checkbox" id="applyNow"> write directly (skip review)</label>
  </div>
  <p class="note">Review is on by default. Detailed step-by-step progress prints in the
    terminal this GUI was launched from.</p>
</section>

<section class="card">
  <h2>Review</h2>
  <div id="reviewWrap"><p class="placeholder">Run a task to see a review here.</p></div>
</section>

</div>
<aside class="card">
  <h2>Indexed files</h2>
  <div id="fileCount">\u2014</div>
  <input type="text" id="fileSearch" placeholder="filter\u2026">
  <ul id="fileList"></ul>
</aside>
</main>
<script>{JS}</script>
</body>
</html>"""

_NO_REVIEW_HTML = ("<!doctype html><html><body style='font:14px sans-serif;"
                   "color:#86816f;padding:40px;text-align:center'>"
                   "No review staged.</body></html>")


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    """Serves the control page, its polling endpoints, and the embedded
    review page's own /apply and /cancel routes -- all on one server so the
    review page's existing same-origin fetch calls work unmodified inside
    the <iframe>."""

    server_version = "PedacitoGUI"

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/state":
            self._json(200, STATE.status())
        elif path == "/review":
            page = STATE.review_page or _NO_REVIEW_HTML
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b'{"ok":false}', "application/json")

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._body()

        if path == "/peek":
            try:
                profiles = _peek_profiles(body.get("path", ""))
                self._json(200, {"ok": True, "profiles": profiles})
            except ConfigError as e:
                self._json(200, {"ok": False, "detail": str(e)})
        elif path == "/project":
            ok = _spawn(lambda: _open_project(body.get("path", ""),
                                              body.get("profile", "")),
                       "opening project")
            self._json(202 if ok else 409, {"ok": ok})
        elif path == "/build_index":
            self._json(202 if _spawn(_build_index, "building index") else 409,
                       {"ok": True})
        elif path == "/check_model":
            self._json(202 if _spawn(_check_model, "checking model") else 409,
                       {"ok": True})
        elif path == "/run":
            task = (body.get("task") or "").strip()
            if not task:
                self._json(400, {"ok": False, "detail": "empty task"})
                return
            ok = _spawn(lambda: _run_task(task, bool(body.get("apply"))),
                       "running task")
            self._json(202 if ok else 409, {"ok": ok})
        elif path == "/apply":
            if body.get("nonce") != STATE.review_nonce:
                self._json(403, {"ok": False,
                                 "detail": "This review is no longer valid."})
                return
            ok, detail = _apply_review()
            self._json(200, {"ok": ok, "detail": detail})
        elif path == "/cancel":
            STATE.review_outcome, STATE.review_detail = "cancelled", "No files were changed."
            STATE.clear_review()
            self._json(200, {"ok": True})
        elif path == "/shutdown":
            self._json(200, {"ok": True})
            threading.Thread(target=_shutdown_server, daemon=True).start()
        else:
            self._json(404, {"ok": False})


_SERVER: ThreadingHTTPServer | None = None


def _shutdown_server() -> None:
    """Stop serve_forever() from a thread other than the one running it,
    as required by http.server -- called via the Exit button's /shutdown
    request, on its own thread so this request's response is already on
    the wire before the server actually stops."""
    if _SERVER is not None:
        _SERVER.shutdown()


def run(port: int = 0, open_browser: bool = True) -> None:
    """Start the GUI server and block until interrupted (Ctrl-C or the
    page's Exit button)."""
    global _SERVER
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    _SERVER = srv
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"pedacito gui running at {url}  (Ctrl-C to stop)", file=sys.stderr)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(file=sys.stderr)
    finally:
        srv.shutdown()
        srv.server_close()
