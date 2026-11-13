"""Headless DOM smoke check (`pedacito domcheck`, TECHNIQUES.md #45).

webcheck proves the STATIC contracts a browser enforces at load (asset
links, DOM ids, ESM syntax); it cannot prove the page actually MOUNTS. This
module is the runtime half: it serves the project over a loopback HTTP
server (ES modules need a real origin), loads the entry page in headless
Chromium via Playwright, and reports

  * uncaught page errors and failed same-origin loads (a module that 404s
    takes the whole graph down with it),
  * interactive controls with no listener, inline handler, or delegation
    (conservative: any ancestor/document binding suppresses the finding),
  * a snapshot of the mounted surface (elements, controls, text volume)
    that can be diffed against a pre-edit snapshot -- the BUG-216 shape,
    where replacing the entry module leaves the page silently unmounted,
  * silent Web Audio failures: a graph that never reaches ctx.destination,
    an AudioContext that is never resumed inside a user gesture, a Play
    button that starts zero sources, gains pinned at zero, and invalid
    AudioParam automation. Web Audio fails WITHOUT console errors, so the
    check instruments the API (window.__pdcAudio) and runs a transport
    smoke probe (enable a step, click Play, watch what actually starts).
    Pages that never touch Web Audio get zero audio findings.

Opt-in and permission-gated: running project code in a browser is an
execution decision, so the in-run gate needs `verify_dom` AND `allow_shell`
(the same boundary as the test subprocess). An explicit `pedacito domcheck`
invocation is itself the consent, like `webcheck`. Playwright + Chromium is
an optional install; when it is missing the check reports an honest install
hint instead of pretending the page is fine.
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

__all__ = ["DomFinding", "DomResult", "available", "check", "find_entry",
           "load_snapshot", "save_snapshot"]

_SKIP_DIRS = {"node_modules", ".git", "dist", "build", "out", ".venv", "venv",
              "__pycache__", ".pedacito", "coverage", ".next", "vendor",
              ".cache", ".tox"}
_HTML_EXT = (".html", ".htm")

_INSTALL_HINT = ("Playwright is not installed. The DOM smoke check needs it:"
                 "\n  pip install playwright"
                 "\n  playwright install chromium")
_BROWSER_HINT = ("Playwright's Chromium browser is missing. Install it with:"
                 "\n  playwright install chromium")

_AVAIL_CACHE: tuple[bool, str] | None = None


def available() -> tuple[bool, str]:
    """(usable, hint): is Playwright importable AND its Chromium present?
    Cached per process; the hint is the honest install instruction."""
    global _AVAIL_CACHE
    if _AVAIL_CACHE is not None:
        return _AVAIL_CACHE
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _AVAIL_CACHE = (False, _INSTALL_HINT)
        return _AVAIL_CACHE
    try:
        with sync_playwright() as p:
            exe = p.chromium.executable_path
        ok = bool(exe) and Path(exe).exists()
    except Exception:
        ok = False
    _AVAIL_CACHE = (True, "") if ok else (False, _BROWSER_HINT)
    return _AVAIL_CACHE


@dataclass(frozen=True)
class DomFinding:
    kind: str        # page_error | load_error | console_error |
    #                # unbound_control | mount_regression |
    #                # audio_graph_disconnected | audio_context_suspended |
    #                # audio_dead_transport | audio_param_invalid |
    #                # audio_gain_zeroed
    severity: str    # "error" gates a run; "warn" is reported, not gated
    message: str

    @property
    def key(self) -> str:
        """Stable identity for baseline diffing: digits (line numbers,
        counts) shift between runs and must not fake a new finding."""
        return self.kind + ":" + re.sub(r"\d+", "#", self.message)[:160]

    def __str__(self) -> str:
        return f"[{self.severity}] {self.kind}: {self.message}"


@dataclass
class DomResult:
    entry: str                       # project-relative entry page, "" if none
    findings: list = field(default_factory=list)
    snapshot: dict | None = None     # the mounted surface (see _SNAPSHOT_JS)
    error: str = ""                  # why the check could not run at all

    @property
    def ok(self) -> bool:
        return not self.error and not any(
            f.severity == "error" for f in self.findings)


def find_entry(root, override: str = "") -> str:
    """The entry HTML: an explicit override, else root index.html, else the
    shallowest index page, preferring pages that load ES modules."""
    root = Path(root)
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = root / p
        if not p.is_file():
            return ""
        try:
            return p.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return ""
    candidates: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in _SKIP_DIRS and not d.startswith("."))
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() in _HTML_EXT:
                rel = (Path(dirpath) / fn).relative_to(root).as_posix()
                candidates.append(rel)
        if len(candidates) >= 200:
            break
    if not candidates:
        return ""

    def score(rel: str):
        name = rel.rsplit("/", 1)[-1].lower()
        is_index = name in ("index.html", "index.htm")
        try:
            text = (root / rel).read_text(encoding="utf-8",
                                          errors="replace")[:200_000]
        except OSError:
            text = ""
        has_module = bool(re.search(
            r"<script[^>]*type\s*=\s*[\"']module[\"']", text, re.I))
        return (0 if rel == "index.html" else 1 if is_index else 2,
                0 if has_module else 1, rel.count("/"), rel)

    return min(candidates, key=score)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):      # the check is not an access log
        pass


@contextmanager
def _local_server(root: Path):
    """A loopback-only static server rooted at the project: ES modules refuse
    to load from file://, so the page needs a real http origin. Port 0 lets
    the OS pick a free one; the thread dies with the process."""
    handler = functools.partial(_QuietHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


# Wrap addEventListener BEFORE any project script runs, so every binding the
# app registers (direct or delegated) is visible to the unbound-control check.
_INIT_SCRIPT = """
(() => {
  const bindings = [];
  const desc = (el) => {
    try {
      if (el === window) return 'window';
      if (el === document) return 'document';
      if (typeof Element !== 'undefined' && el instanceof Element) {
        let s = el.tagName.toLowerCase();
        if (el.id) s += '#' + el.id;
        else if (el.classList.length)
          s += '.' + Array.from(el.classList).slice(0, 3).join('.');
        return s;
      }
    } catch (e) {}
    return 'other';
  };
  const refs = [];
  const orig = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function (type, fn, opts) {
    try {
      // BUG-325: addEventListener(type, undefined) is a no-op -- the browser
      // binds nothing, yet the wrapper used to record a binding anyway, so a
      // dead control looked bound. Only a callable listener (or an object with
      // a handleEvent method) is a real binding.
      const real = typeof fn === 'function' ||
                   (fn && typeof fn.handleEvent === 'function');
      if (real && bindings.length < 2000) {
        bindings.push({ type: String(type), target: desc(this) });
        refs.push(this);
      }
    } catch (e) {}
    return orig.call(this, type, fn, opts);
  };
  window.__pdcBindings = bindings;
  // BUG-325: a listener registered on a node that is later removed (e.g.
  // innerHTML = '') is dead. Report only targets still in the document (or
  // window/document themselves), so a detached node cannot mask an unbound
  // control. Element refs live in `refs`, never in the serialized bindings.
  window.__pdcBoundTargets = () => {
    const out = [];
    for (let i = 0; i < refs.length; i++) {
      const t = refs[i];
      if (t === window || t === document || (t && t.isConnected))
        out.push(bindings[i].target);
    }
    return out;
  };
})();
"""

# Web Audio telemetry. Web Audio fails SILENTLY: a suspended context, an
# unstarted oscillator, a graph that never reaches ctx.destination or a gain
# pinned at 0 produce no console error, so load-time checks cannot see them.
# This init script wraps the API before any project script runs and records
# what the page actually DOES (contexts, edges, source starts, automation).
# It is dormant by construction: a page that never touches Web Audio records
# nothing and gets zero audio findings. Headless Chromium does not enforce
# the autoplay policy (contexts report "running" even when a real browser
# would suspend them), so the suspended-context gate is behavioral: it asks
# whether resume() was ever called and whether the context was created
# inside a user-gesture handler (window.event is set during ANY handler --
# addEventListener, onclick property, inline attribute, delegation -- and is
# undefined inside setTimeout/setInterval/rAF callbacks).
_AUDIO_INIT_SCRIPT = """
(() => {
  if (typeof AudioNode === 'undefined' ||
      typeof BaseAudioContext === 'undefined') return;
  const A = {
    connectCalls: 0,
    nodes: [],        // {id, ctx, type, started}
    edges: [],        // {from, to, param}
    starts: [],       // {node, ctx, time}
    paramIssues: [],  // {nodeType, method, kind, value}
    gains: {},        // nodeId -> {value, auto: [{m, v}]}
    contexts: []      // {id, kind, resumeCalls, resumeInGesture, createdInGesture, dest}
  };
  const ctxRefs = [];          // live context objects, parallel to A.contexts
  const nodeIds = new WeakMap();
  const nodeById = new Map();
  const ctxIds = new WeakMap();
  const paramOwner = new WeakMap();
  let nId = 0, cId = 0;
  const MAX_NODES = 4000, MAX_EDGES = 8000, MAX_STARTS = 2000, MAX_ISSUES = 60;
  const PARAM_NAMES = ['gain', 'frequency', 'Q', 'detune', 'delayTime',
                       'playbackRate', 'offset', 'pan', 'threshold', 'knee',
                       'ratio', 'attack', 'release'];
  const GESTURE_RE = /^(click|dblclick|pointerdown|pointerup|mousedown|mouseup|keydown|keyup|touchstart|touchend|input|change|submit)$/i;
  const inGesture = () => {
    try { return !!(window.event && GESTURE_RE.test(window.event.type)); }
    catch (e) { return false; }
  };
  const finite = (v) => typeof v === 'number' && isFinite(v);

  const ctxOf = (n) => {
    try { return ctxIds.get(n.context) || 0; } catch (e) { return 0; }
  };
  const nodeId = (n) => {
    let id = nodeIds.get(n);
    if (id !== undefined) return id;
    id = ++nId;
    nodeIds.set(n, id);
    if (A.nodes.length < MAX_NODES) {
      let type = 'AudioNode';
      try { type = (n.constructor && n.constructor.name) || 'AudioNode'; }
      catch (e) {}
      const rec = { id, ctx: ctxOf(n), type, started: false };
      A.nodes.push(rec);
      nodeById.set(id, rec);
      if (type === 'GainNode') A.gains[id] = { value: 1, auto: [] };
      for (const k of PARAM_NAMES) {
        try {
          const p = n[k];
          if (typeof AudioParam !== 'undefined' && p instanceof AudioParam)
            paramOwner.set(p, id);
        } catch (e) {}
      }
    }
    return id;
  };
  const regCtx = (c, kind) => {
    if (ctxIds.has(c)) return;
    const id = ++cId;
    ctxIds.set(c, id);
    let dest = 0;
    try { dest = nodeId(c.destination); } catch (e) {}
    A.contexts.push({ id, kind, resumeCalls: 0, resumeInGesture: 0,
                      createdInGesture: inGesture(), dest });
    ctxRefs.push(c);
  };

  // Context constructors: record origin (gesture or not) and kind. Realtime
  // contexts need resume() inside a gesture in real browsers; offline ones
  // render on demand and are exempt.
  const wrapCtor = (name, kind) => {
    const Orig = window[name];
    if (typeof Orig !== 'function') return;
    const W = function (...args) {
      if (!new.target) throw new TypeError(
        "Failed to construct '" + Orig.name +
        "': Please use the 'new' operator, this DOM object constructor " +
        'cannot be called as a function.');
      const inst = Reflect.construct(Orig, args, new.target);
      try { regCtx(inst, kind); } catch (e) {}
      return inst;
    };
    W.prototype = Orig.prototype;
    try { Object.defineProperty(W, 'name', { value: Orig.name }); } catch (e) {}
    window[name] = W;
  };
  // resume(): count calls and whether they happened inside a gesture.
  const patchResume = (Ctor) => {
    if (!Ctor || !Ctor.prototype ||
        typeof Ctor.prototype.resume !== 'function') return;
    const origResume = Ctor.prototype.resume;
    Ctor.prototype.resume = function () {
      try {
        const id = ctxIds.get(this);
        const rec = A.contexts.find(c => c.id === id);
        if (rec) {
          rec.resumeCalls++;
          if (inGesture()) rec.resumeInGesture++;
        }
      } catch (e) {}
      return origResume.apply(this, arguments);
    };
  };
  const RealAC = window.AudioContext;
  patchResume(RealAC);
  if (window.webkitAudioContext && window.webkitAudioContext !== RealAC)
    patchResume(window.webkitAudioContext);
  wrapCtor('AudioContext', 'realtime');
  wrapCtor('webkitAudioContext', 'realtime');
  wrapCtor('OfflineAudioContext', 'offline');
  wrapCtor('webkitOfflineAudioContext', 'offline');

  // Node factories: register every created node with its context.
  for (const key of Object.getOwnPropertyNames(BaseAudioContext.prototype)) {
    if (!/^create/.test(key)) continue;
    const d = Object.getOwnPropertyDescriptor(BaseAudioContext.prototype, key);
    if (!d || typeof d.value !== 'function') continue;
    const orig = d.value;
    Object.defineProperty(BaseAudioContext.prototype, key, {
      configurable: true, writable: true, enumerable: d.enumerable,
      value: function (...args) {
        const r = orig.apply(this, args);
        try { if (r instanceof AudioNode) nodeId(r); } catch (e) {}
        return r;
      }
    });
  }
  if (typeof window.AudioWorkletNode === 'function') {
    const OWN = window.AudioWorkletNode;
    const WW = function (...args) {
      if (!new.target) throw new TypeError(
        "Failed to construct 'AudioWorkletNode': Please use the 'new' operator.");
      const inst = Reflect.construct(OWN, args, new.target);
      try { nodeId(inst); } catch (e) {}
      return inst;
    };
    WW.prototype = OWN.prototype;
    window.AudioWorkletNode = WW;
  }

  // Directed graph: connect records edges (AudioNode AND AudioParam
  // targets); disconnect removes them. Edges are recorded only after the
  // native call succeeds, so a refused connect (e.g. across contexts) does
  // not fake a path.
  const origConnect = AudioNode.prototype.connect;
  AudioNode.prototype.connect = function (target) {
    const r = origConnect.apply(this, arguments);
    try {
      A.connectCalls++;
      const from = nodeId(this);
      if (target instanceof AudioNode) {
        if (A.edges.length < MAX_EDGES)
          A.edges.push({ from, to: nodeId(target), param: false });
      } else if (typeof AudioParam !== 'undefined' &&
                 target instanceof AudioParam) {
        const owner = paramOwner.get(target);
        if (owner !== undefined && A.edges.length < MAX_EDGES)
          A.edges.push({ from, to: owner, param: true });
      }
    } catch (e) {}
    return r;
  };
  const origDisconnect = AudioNode.prototype.disconnect;
  AudioNode.prototype.disconnect = function () {
    const r = origDisconnect.apply(this, arguments);
    try {
      const from = nodeIds.get(this);
      if (from !== undefined) {
        const t = arguments[0];
        if (arguments.length === 0) {
          A.edges = A.edges.filter(e => e.from !== from);
        } else if (t instanceof AudioNode) {
          const to = nodeIds.get(t);
          A.edges = A.edges.filter(e => !(e.from === from && e.to === to));
        } else if (typeof AudioParam !== 'undefined' &&
                   t instanceof AudioParam) {
          const to = paramOwner.get(t);
          A.edges = A.edges.filter(e => !(e.from === from && e.to === to &&
                                          e.param));
        }
        // disconnect(outputIndex): per-output edges are not tracked; keep
        // them (under-removal cannot fake a disconnection finding).
      }
    } catch (e) {}
    return r;
  };

  // Source starts: OscillatorNode / AudioBufferSourceNode /
  // ConstantSourceNode all inherit start() from AudioScheduledSourceNode.
  const SS = window.AudioScheduledSourceNode;
  if (SS && SS.prototype && typeof SS.prototype.start === 'function') {
    const origStart = SS.prototype.start;
    SS.prototype.start = function () {
      const r = origStart.apply(this, arguments);
      try {
        const id = nodeId(this);
        const rec = nodeById.get(id);
        if (rec && !rec.started) {
          rec.started = true;
          if (A.starts.length < MAX_STARTS)
            A.starts.push({ node: id, ctx: rec.ctx,
                            time: finite(arguments[0]) ? arguments[0] : 0 });
        }
      } catch (e) {}
      return r;
    };
  }

  // AudioParam automation: record gain envelopes and invalid arguments
  // (NaN / undefined / non-finite, exponential ramps to <= 0). Recording
  // happens BEFORE the native call so a swallowed exception still leaves
  // evidence; the native call then runs untouched (same throws as always).
  const issue = (nodeType, method, kind, value) => {
    if (A.paramIssues.length >= MAX_ISSUES) return;
    if (A.paramIssues.some(i => i.nodeType === nodeType &&
                                i.method === method && i.kind === kind))
      return;
    A.paramIssues.push({ nodeType, method, kind,
                         value: finite(value) ? value : String(value) });
  };
  const gainAuto = (owner, m, v) => {
    const g = A.gains[owner];
    if (g && g.auto.length < 200) g.auto.push({ m, v: finite(v) ? v : 0 });
  };
  for (const m of ['setValueAtTime', 'linearRampToValueAtTime',
                   'exponentialRampToValueAtTime', 'setTargetAtTime']) {
    const orig = AudioParam.prototype[m];
    if (typeof orig !== 'function') continue;
    AudioParam.prototype[m] = function (value, time, extra) {
      try {
        const owner = paramOwner.get(this);
        const rec = owner !== undefined ? nodeById.get(owner) : null;
        const nodeType = rec ? rec.type : 'AudioParam';
        if (!finite(value)) issue(nodeType, m, 'nonfinite', value);
        else if (m === 'exponentialRampToValueAtTime' && value <= 0)
          issue(nodeType, m, 'nonpositive', value);
        if (m === 'setTargetAtTime') {
          if (!finite(time)) issue(nodeType, m, 'nonfinite_time', time);
          if (finite(extra) && extra < 0)
            issue(nodeType, m, 'negative_constant', extra);
        } else if (time !== undefined && !finite(time)) {
          issue(nodeType, m, 'nonfinite_time', time);
        }
        if (rec && rec.type === 'GainNode') gainAuto(owner, m, value);
      } catch (e) {}
      return orig.apply(this, arguments);
    };
  }
  const vd = Object.getOwnPropertyDescriptor(AudioParam.prototype, 'value');
  if (vd && vd.set) {
    Object.defineProperty(AudioParam.prototype, 'value', {
      configurable: true, enumerable: vd.enumerable, get: vd.get,
      set: function (v) {
        try {
          const owner = paramOwner.get(this);
          const rec = owner !== undefined ? nodeById.get(owner) : null;
          if (!finite(v))
            issue(rec ? rec.type : 'AudioParam', 'value', 'nonfinite', v);
          if (rec && rec.type === 'GainNode' && A.gains[owner])
            A.gains[owner].value = finite(v) ? v : 0;
        } catch (e) {}
        vd.set.call(this, v);
      }
    });
  }

  window.__pdcAudio = A;
  window.__pdcAudioReport = () => {
    const contexts = A.contexts.map((c, i) => {
      let state = 'unknown';
      try { state = ctxRefs[i].state; } catch (e) {}
      return { id: c.id, kind: c.kind, state, resumeCalls: c.resumeCalls,
               resumeInGesture: c.resumeInGesture,
               createdInGesture: c.createdInGesture, dest: c.dest };
    });
    return { contexts, nodes: A.nodes, edges: A.edges, starts: A.starts,
             paramIssues: A.paramIssues, gains: A.gains,
             connectCalls: A.connectCalls };
  };
})();
"""

# The transport smoke probe, part 1: mark the controls to click. A sequencer
# whose default grid is empty plays nothing even when healthy, so if step
# toggles exist and NONE is active, the first one is marked for a click.
# The primary Play/Start/Trigger control is marked next (id first, then a
# word-anchored text match -- "Restart" must not match "start").
_AUDIO_MARK_JS = """
() => {
  const out = { stepMarked: false, stepsActive: 0, stepCandidates: 0,
                playMarked: false, playLabel: '' };
  const isActive = (el) =>
    /(?:^|\\s)(?:[\\w-]*(?:active|lit|pressed|selected)[\\w-]*|on)(?:\\s|$)/i
      .test(el.className || '') ||
    el.getAttribute('aria-pressed') === 'true';
  const steps = Array.from(document.querySelectorAll(
    'button[data-step], [role="button"][data-step], button.step, ' +
    'button.pf-cell, button[class*="step"], button[class*="pad"], ' +
    'button[class*="cell"]'));
  out.stepCandidates = steps.length;
  out.stepsActive = steps.filter(isActive).length;
  if (steps.length && out.stepsActive === 0) {
    steps[0].setAttribute('data-pdc-probe', 'step');
    out.stepMarked = true;
  }
  let play = null;
  for (const id of ['play', 'start', 'play-btn', 'playBtn', 'play-pause',
                    'trigger', 'start-btn', 'startBtn']) {
    play = document.getElementById(id);
    if (play) break;
  }
  if (!play) {
    const re = /(?:^|[\\s\\u25b6\\u25ba\\u23f5])(play|start|trigger)\\b/i;
    for (const b of document.querySelectorAll('button, [role="button"]')) {
      const t = ((b.innerText || '') + ' ' +
                 (b.getAttribute('aria-label') || '')).trim();
      if (re.test(t) && !b.disabled) { play = b; break; }
    }
  }
  if (play && !play.disabled) {
    play.setAttribute('data-pdc-probe', 'play');
    out.playMarked = true;
    out.playLabel = ((play.innerText ||
                      play.getAttribute('aria-label') || '') + '')
                      .trim().slice(0, 40);
  }
  return out;
}
"""

# The transport smoke probe, part 2: sweep up to six range sliders through
# their midpoint and dispatch 'input'. A handler that calls a method the
# synth class does not define (setCutoff vs setFilterCutoff) throws here --
# a TypeError the load-time check never sees because nobody moved a slider.
_AUDIO_SLIDER_JS = """
() => {
  const sliders = Array.from(
      document.querySelectorAll('input[type="range"]')).slice(0, 6);
  for (const s of sliders) {
    try {
      const min = Number(s.min || 0), max = Number(s.max || 100);
      s.value = String(min + (max - min) / 2);
      s.dispatchEvent(new Event('input', { bubbles: true }));
    } catch (e) {}
  }
  return sliders.length;
}
"""

# The mounted surface: every element's identity, the interactive controls
# (with their ancestor chain, for delegation matching), and the text volume.
_SNAPSHOT_JS = """
() => {
  const desc = (el) => {
    let s = el.tagName.toLowerCase();
    if (el.id) s += '#' + el.id;
    else if (el.classList.length)
      s += '.' + Array.from(el.classList).slice(0, 3).join('.');
    return s;
  };
  const elements = [];
  const all = document.querySelectorAll('*');
  for (let i = 0; i < all.length && i < 4000; i++) {
    const el = all[i];
    elements.push({ tag: el.tagName.toLowerCase(), id: el.id || '',
                    classes: Array.from(el.classList).sort().join(' ') });
  }
  const controls = [];
  const sel = 'button, a[href], input, select, textarea, ' +
              '[role="button"], [onclick]';
  for (const el of document.querySelectorAll(sel)) {
    if (controls.length >= 400) break;
    const chain = [];
    let n = el;
    while (n && n.nodeType === 1) { chain.push(desc(n)); n = n.parentElement; }
    controls.push({
      tag: el.tagName.toLowerCase(), id: el.id || '',
      classes: Array.from(el.classList).sort().join(' '),
      text: ((el.innerText || el.value ||
              el.getAttribute('aria-label') || '') + '').trim().slice(0, 40),
      inline: Array.from(el.attributes).some(a => /^on/i.test(a.name)),
      disabled: !!el.disabled,
      type: (el.getAttribute('type') || '').toLowerCase(),
      chain: chain
    });
  }
  return {
    title: document.title || '',
    elements: elements,
    text_len: ((document.body && document.body.innerText) || '').length,
    controls: controls,
    module_scripts: Array.from(
        document.querySelectorAll('script[type="module"][src]'))
        .map(s => s.getAttribute('src')),
    body_children: document.body ? document.body.children.length : 0
  };
}
"""


def _ident(e: dict) -> tuple:
    return (e.get("tag", ""), e.get("id", ""), e.get("classes", ""))


def _label(c: dict) -> str:
    if c.get("id"):
        return f"#{c['id']}"
    if c.get("text"):
        return f"\"{c['text']}\""
    return c.get("tag", "control")


def _diff_mounted(before: dict, after: dict) -> list[DomFinding]:
    """The BUG-216 regression shape: UI that was mounted before the edit is
    gone after it. Conservative -- anonymous layout churn alone is not a
    finding; named elements, lost controls, a large shrink, or collapsed text
    volume are."""
    out: list[DomFinding] = []
    b_el = Counter(_ident(e) for e in before.get("elements", []))
    a_el = Counter(_ident(e) for e in after.get("elements", []))
    removed = b_el - a_el
    n_removed = sum(removed.values())
    named = [f"{t}#{i}" if i else (f"{t}.{c}" if c else t)
             for (t, i, c) in removed if i][:8]
    b_ctrl = {_ident(c) for c in before.get("controls", [])}
    a_ctrl = {_ident(c) for c in after.get("controls", [])}
    lost_ctrl = sorted(b_ctrl - a_ctrl)
    lost_names = [f"{t}#{i}" if i else (f"{t}.{c}" if c else t)
                  for (t, i, c) in lost_ctrl][:8]
    total_before = sum(b_el.values())
    big_shrink = n_removed >= 10 and n_removed > 0.3 * max(1, total_before)
    b_text = int(before.get("text_len", 0))
    a_text = int(after.get("text_len", 0))
    text_gone = b_text >= 200 and a_text < b_text / 2
    if named or lost_names or big_shrink or text_gone:
        bits = []
        if named:
            bits.append("named elements gone: " + ", ".join(named))
        if lost_names:
            bits.append("controls gone: " + ", ".join(lost_names))
        if big_shrink:
            bits.append(f"{n_removed} of {total_before} elements removed")
        if text_gone:
            bits.append(f"visible text collapsed ({b_text} -> {a_text} chars)")
        out.append(DomFinding(
            "mount_regression", "error",
            "the mounted DOM regressed against the pre-edit snapshot ("
            + "; ".join(bits)
            + "); the entry module may no longer mount the UI"))
    return out


# --- Web Audio telemetry analysis -------------------------------------------

_SOURCE_TYPES = {"OscillatorNode", "AudioBufferSourceNode",
                 "MediaElementAudioSourceNode", "MediaStreamAudioSourceNode",
                 "AudioWorkletNode", "ConstantSourceNode"}

_AUDIO_STATIC_RE = re.compile(
    r"\b(?:AudioContext|webkitAudioContext|OfflineAudioContext|"
    r"AudioWorkletNode|createOscillator|createBufferSource|createGain|"
    r"createBiquadFilter|createConstantSource|createMediaElementSource)\b")


def _audio_static_presence(root: Path) -> bool:
    """Does any local script/HTML reference the Web Audio API? Catches apps
    that create their context lazily (first Play click), where the runtime
    telemetry is still empty at load. Bounded like webcheck's scan."""
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in _SKIP_DIRS and not d.startswith("."))
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() not in (
                    ".js", ".mjs", ".html", ".htm"):
                continue
            try:
                text = (Path(dirpath) / fn).read_text(
                    encoding="utf-8", errors="replace")[:262_144]
            except OSError:
                continue
            if _AUDIO_STATIC_RE.search(text):
                return True
            n += 1
            if n >= 400:
                return False
    return False


def _run_audio_probe(page) -> dict:
    """The active transport smoke probe: enable a step if the grid is empty,
    click Play with a TRUSTED click (so gesture-gated code paths run), give
    lookahead schedulers ~300ms to tick, then sweep the sliders. Never
    raises: a probe that cannot click records what it managed to do."""
    probe = {"playClicked": False, "playLabel": "", "stepToggled": False,
             "stepsActiveBefore": 0, "stepCandidates": 0,
             "startedBefore": 0, "startedAfter": 0, "mediaPlaying": 0}
    try:
        probe["startedBefore"] = page.evaluate(
            "window.__pdcAudioReport ? "
            "window.__pdcAudioReport().starts.length : 0")
        marks = page.evaluate(_AUDIO_MARK_JS)
    except Exception:
        return probe
    probe["stepsActiveBefore"] = marks.get("stepsActive", 0)
    probe["stepCandidates"] = marks.get("stepCandidates", 0)
    try:
        if marks.get("stepMarked"):
            page.click('[data-pdc-probe="step"]', timeout=2000)
            probe["stepToggled"] = True
    except Exception:
        pass
    try:
        if marks.get("playMarked"):
            page.click('[data-pdc-probe="play"]', timeout=2000)
            probe["playClicked"] = True
            probe["playLabel"] = marks.get("playLabel", "")
            # lookahead schedulers (setInterval / rAF / setTimeout chains)
            # need at least one tick at a typical 120-160 BPM (125-94ms)
            page.wait_for_timeout(300)
    except Exception:
        pass
    try:
        page.evaluate(_AUDIO_SLIDER_JS)
        page.wait_for_timeout(150)
    except Exception:
        pass
    try:
        rep = page.evaluate(
            "window.__pdcAudioReport ? window.__pdcAudioReport() : null")
        probe["startedAfter"] = len(rep["starts"]) if rep else 0
        probe["mediaPlaying"] = page.evaluate(
            "Array.from(document.querySelectorAll('audio,video'))"
            ".filter(m => !m.paused && !m.ended).length")
    except Exception:
        pass
    return probe


def _audio_findings(report: dict | None, probe: dict) -> list[DomFinding]:
    """Deterministic verdicts over the recorded Web Audio telemetry. Pure
    function of (report, probe) so every failure shape is unit-testable
    without a browser. Messages name the shape, not shifting ids/counts, so
    DomFinding.key stays stable across runs."""
    if not report:
        return []
    nodes = {n["id"]: n for n in report.get("nodes", [])}
    edges = report.get("edges", [])
    contexts = report.get("contexts", [])
    starts = report.get("starts", [])
    gains = report.get("gains", {})
    out: list[DomFinding] = []

    # 1. Invalid AudioParam automation: NaN/undefined/non-finite values,
    #    exponential ramps to <= 0, bad times. Recorded even when the app
    #    swallows the exception inside a timer callback.
    for i in report.get("paramIssues", []):
        method = i.get("method", "?")
        kind = i.get("kind", "")
        nt = i.get("nodeType", "AudioParam")
        if kind == "nonfinite":
            desc = "a non-finite value (NaN/undefined/Infinity)"
        elif kind == "nonpositive":
            desc = f"{i.get('value')} (exponential ramps must be > 0)"
        elif kind == "nonfinite_time":
            desc = "a non-finite time"
        elif kind == "negative_constant":
            desc = "a negative time constant"
        else:
            desc = str(i.get("value"))
        out.append(DomFinding(
            "audio_param_invalid", "error",
            f"AudioParam.{method} on {nt} received {desc}; the automation "
            "is rejected or silent -- pass finite numbers (and a positive "
            "target for exponentialRampToValueAtTime)"))

    adj: dict = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])
    source_nodes = [n for n in nodes.values() if n["type"] in _SOURCE_TYPES]
    any_started = bool(starts)
    play_clicked = bool(probe.get("playClicked"))

    def reaches(start_id: int, dest: int, skip: set | None = None) -> bool:
        seen = {start_id}
        stack = [start_id]
        while stack:
            cur = stack.pop()
            if cur == dest:
                return True
            for nxt in adj.get(cur, []):
                if nxt in seen or (skip and nxt in skip):
                    continue
                seen.add(nxt)
                stack.append(nxt)
        return False

    # 2. Disconnected graph: sources exist (or were started) but no path
    #    reaches their context's destination. Evaluated per context so an
    #    offline render graph is judged against its own destination.
    if source_nodes and (any_started or play_clicked):
        disconnected = False
        for c in contexts:
            ctx_sources = [n for n in source_nodes if n["ctx"] == c["id"]]
            if not ctx_sources or not c.get("dest"):
                continue
            if not any(reaches(n["id"], c["dest"]) for n in ctx_sources):
                disconnected = True
                break
        if disconnected:
            types = ", ".join(sorted({n["type"] for n in source_nodes}))
            out.append(DomFinding(
                "audio_graph_disconnected", "error",
                f"audio sources ({types}) were created/started but no "
                "connected path reaches ctx.destination; connect the "
                "output chain (e.g. source -> gain -> ctx.destination)"))

    # 3. Suspended context: created outside a user-gesture handler and
    #    resume() never called, while the page actually attempts playback.
    #    Headless Chromium masks the suspended STATE (autoplay policy is
    #    not enforced), so the gate is the observable behavior: no resume
    #    call + non-gesture origin. A context created inside a click
    #    handler starts running in real browsers and is exempt.
    for c in contexts:
        if c.get("kind") != "realtime":
            continue
        cid = c["id"]
        used = any(n["ctx"] == cid for n in source_nodes) or any(
            nodes.get(e["from"], {}).get("ctx") == cid for e in edges)
        if c.get("resumeCalls", 0) > 0:
            # resume() ran, but only outside a gesture: flag it only when
            # the probe's Play click leaves the context suspended. Apps
            # that unlock audio on the first interaction (a game with no
            # Play button) are exempt -- the probe never triggered that
            # gesture, and the load-time jingle staying silent until the
            # first click is standard browser behavior, not a bug.
            if (play_clicked and used
                    and c.get("state") == "suspended"
                    and c.get("resumeInGesture", 0) == 0):
                out.append(DomFinding(
                    "audio_context_suspended", "error",
                    "AudioContext is still suspended: resume() was only "
                    "called outside a user gesture; call ctx.resume() "
                    "inside the Play/Start click handler"))
            continue
        if c.get("createdInGesture"):
            continue
        if not used or not (any_started or play_clicked):
            continue
        state = c.get("state", "unknown")
        extra = (f" (state observed: {state})"
                 if state == "suspended" else "")
        out.append(DomFinding(
            "audio_context_suspended", "error",
            "AudioContext was created outside a user-gesture handler and "
            "resume() was never called; browsers keep it suspended and "
            f"silent{extra}. Call ctx.resume() inside the Play/Start click "
            "handler (or create the context there)"))

    # 4. Dead transport: Play was clicked (with a step enabled if the grid
    #    was empty) yet zero sources started. Media-element playback and
    #    routed AudioWorklet synthesis do not call start() and are exempt.
    started_delta = probe.get("startedAfter", 0) - probe.get("startedBefore", 0)
    audio_wired = (source_nodes or report.get("connectCalls", 0)
                   or probe.get("stepCandidates", 0))
    worklet_routed = any(n["type"] == "AudioWorkletNode" and adj.get(n["id"])
                         for n in source_nodes)
    if (play_clicked and audio_wired and started_delta <= 0
            and not probe.get("mediaPlaying") and not worklet_routed):
        out.append(DomFinding(
            "audio_dead_transport", "error",
            "clicking Play started zero audio sources (OscillatorNode / "
            "AudioBufferSourceNode): the scheduler never triggers the "
            "synth. Check the play-handler wiring, the default pattern "
            "(no active steps), mute/solo defaults, and the synth trigger "
            "method signature"))

    # 5. Zeroed gain: every started source reaches the destination ONLY
    #    through GainNodes pinned at 0 (value 0 and no automation raising
    #    it). Connected but silent.
    if starts:
        pinned = {int(gid) for gid, g in gains.items()
                  if g.get("value") == 0
                  and not any(a.get("v", 0) > 0 for a in g.get("auto", []))}
        routed = [s for s in starts
                  if nodes.get(s["node"]) and any(
                      c["id"] == nodes[s["node"]]["ctx"] and c.get("dest")
                      and reaches(s["node"], c["dest"]) for c in contexts)]
        if routed and all(
                not reaches(s["node"], next(
                    c["dest"] for c in contexts
                    if c["id"] == nodes[s["node"]]["ctx"]), skip=pinned)
                for s in routed):
            out.append(DomFinding(
                "audio_gain_zeroed", "error",
                "every started audio source routes through a GainNode "
                "pinned at 0 (no automation raises it); the graph is "
                "connected but silent -- set a non-zero gain or schedule "
                "an envelope"))
    return out


def check(root, *, entry: str = "", timeout: int = 45,
          baseline: dict | None = None) -> DomResult:
    """Load the entry page headless and report what the browser sees. Never
    raises: a check that cannot run returns `error` with the reason (missing
    install, no entry page, launch failure) instead of a false green.

    The Playwright session runs in a child process: `page.evaluate()` has no
    timeout, so a page script that loops forever would block the caller
    forever (a `new --complex` task stalled for eleven minutes before the
    driver was killed). `timeout` bounds navigation; the child gets a short
    grace for launch and reporting, then its whole process group is stopped.
    """
    root = Path(root)
    ok, hint = available()
    if not ok:
        return DomResult(entry="", error=hint)
    rel = find_entry(root, entry)
    if not rel:
        return DomResult(entry="", error=(
            "no HTML entry page found (set dom_entry to the page to load)"))
    timeout = max(5, int(timeout or 45))
    from . import proc

    cmd = [sys.executable, "-m", "pedacito.domcheck", "--worker",
           str(root), "--entry", rel, "--timeout", str(timeout)]
    cap = timeout + 10
    rc, out, stopped, _secs = proc.run(cmd, cwd=root, timeout=cap,
                                       max_output=400_000)
    if stopped == "timeout":
        return DomResult(entry=rel, error=(
            f"the headless DOM check did not finish within {cap}s; a page "
            "script may be looping, so the browser was stopped. The page is "
            "NOT verified."))
    if stopped == "error":
        return DomResult(entry=rel, error=(
            f"headless DOM check could not run: {out.strip()[:300]}"))
    payload = None
    for line in (out or "").splitlines():
        if line.startswith(_WORKER_MARK):
            try:
                payload = json.loads(line[len(_WORKER_MARK):])
            except ValueError:
                payload = None
    if payload is None:
        tail = " | ".join((out or "").strip().splitlines()[-4:])[:600]
        return DomResult(entry=rel, error=(
            "headless DOM check ended without a result "
            f"(exit {rc}): {tail}"))
    findings = [DomFinding(**f) for f in payload.get("findings") or []
                if isinstance(f, dict)]
    snap = payload.get("snapshot")
    if isinstance(snap, dict) and baseline:
        findings.extend(_diff_mounted(baseline, snap))
    return DomResult(entry=payload.get("entry") or rel, findings=findings,
                     snapshot=snap if isinstance(snap, dict) else None,
                     error=payload.get("error") or "")


_WORKER_MARK = "__PEDACITO_DOM_JSON__"


def _worker_main(argv: list[str]) -> int:
    """Child entry: run the Playwright session and print one JSON line."""
    import argparse

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("root")
    ap.add_argument("--entry", default="")
    ap.add_argument("--timeout", type=int, default=45)
    a = ap.parse_args(argv)
    res = _check_inproc(a.root, entry=a.entry, timeout=a.timeout)
    payload = {
        "entry": res.entry,
        "error": res.error,
        "findings": [{"kind": f.kind, "severity": f.severity,
                      "message": f.message} for f in res.findings],
        "snapshot": res.snapshot,
    }
    print(_WORKER_MARK + json.dumps(payload))
    return 0


def _check_inproc(root, *, entry: str = "", timeout: int = 45,
                  baseline: dict | None = None) -> DomResult:
    """The actual Playwright session, called by the worker child (and by
    tests that want to observe it directly)."""
    root = Path(root)
    rel = find_entry(root, entry)
    if not rel:
        return DomResult(entry="", error=(
            "no HTML entry page found (set dom_entry to the page to load)"))
    timeout = max(5, int(timeout or 45))
    page_errors: list[str] = []
    console_errors: list[str] = []
    failed: dict[str, str] = {}
    try:
        from playwright.sync_api import TimeoutError as PWTimeout
        from playwright.sync_api import sync_playwright
        with _local_server(root) as base, sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"])
            try:
                page = browser.new_page()
                page.on("pageerror",
                        lambda e: page_errors.append(str(e)))
                page.on("console", lambda m: (
                    console_errors.append(m.text)
                    if m.type == "error" else None))
                page.on("response", lambda r: (
                    failed.setdefault(r.url, str(r.status))
                    if r.status >= 400 and r.url.startswith(base) else None))
                page.on("requestfailed", lambda r: (
                    failed.setdefault(r.url, str(r.failure))
                    if r.url.startswith(base) else None))
                page.add_init_script(_INIT_SCRIPT)
                page.add_init_script(_AUDIO_INIT_SCRIPT)
                try:
                    page.goto(base + "/" + quote(rel),
                              wait_until="networkidle",
                              timeout=timeout * 1000)
                except PWTimeout:
                    # A polling/websocket page never reaches networkidle;
                    # whatever mounted by now is what we check.
                    pass
                page.wait_for_timeout(250)
                snap = page.evaluate(_SNAPSHOT_JS)
                bindings = page.evaluate("window.__pdcBindings || []")
                bound_targets = page.evaluate(
                    "window.__pdcBoundTargets ? window.__pdcBoundTargets() : []")
                # Web Audio: probe only when the page touches the API --
                # runtime activity, or a static reference (lazy contexts).
                # Non-audio pages stay completely dormant.
                audio_report, audio_probe = None, {}
                audio_active = page.evaluate(
                    "window.__pdcAudio ? (window.__pdcAudio.contexts.length"
                    " > 0 || window.__pdcAudio.connectCalls > 0 || "
                    "window.__pdcAudio.nodes.length > 0) : false")
                if audio_active or _audio_static_presence(root):
                    audio_probe = _run_audio_probe(page)
                    audio_report = page.evaluate(
                        "window.__pdcAudioReport ? "
                        "window.__pdcAudioReport() : null")
            finally:
                browser.close()
    except Exception as e:
        return DomResult(entry=rel, error=(
            f"headless DOM check could not run: {type(e).__name__}: {e}"))

    findings: list[DomFinding] = []
    for msg in page_errors[:10]:
        findings.append(DomFinding(
            "page_error", "error",
            "uncaught page error: " + msg.splitlines()[0][:300]))
    for url, why in list(failed.items())[:10]:
        path = url[len(base):] if url.startswith(base) else url
        if path.endswith("favicon.ico"):
            continue
        findings.append(DomFinding(
            "load_error", "error", f"{path or '/'} failed to load ({why})"))
    for msg in console_errors[:6]:
        findings.append(DomFinding(
            "console_error", "warn", "console.error: " + msg[:200]))

    # Unbound controls: conservative. A binding on the control itself, any
    # ancestor (delegation), document/window, or an inline handler all count.
    # BUG-325: count only REAL listeners on still-CONNECTED targets -- the init
    # script drops no-op (undefined) handlers and nodes removed after binding,
    # so neither can mask a dead control.
    targets = set(bound_targets or [])
    delegation = bool(targets & {"document", "window"})
    n_unbound = 0
    for c in snap.get("controls", []):
        if n_unbound >= 8:
            break
        if c.get("tag") == "input" and c.get("type") == "hidden":
            continue
        if c.get("disabled") or c.get("inline") or delegation:
            continue
        if targets & set(c.get("chain", [])):
            continue
        n_unbound += 1
        findings.append(DomFinding(
            "unbound_control", "warn",
            f"<{c.get('tag', '?')}> {_label(c)} has no event listener "
            "(no direct binding, inline handler, or delegation)"))

    if snap.get("module_scripts") and not snap.get("body_children"):
        findings.append(DomFinding(
            "mount_regression", "error",
            "the entry page loads ES modules but <body> is empty after "
            "load; the module graph did not mount any UI"))
    findings.extend(_audio_findings(audio_report, audio_probe))
    if audio_report and (audio_report.get("contexts")
                         or audio_report.get("nodes")):
        snap["audio"] = {
            "contexts": len(audio_report.get("contexts", [])),
            "sources_started": len(audio_report.get("starts", [])),
            "connect_calls": audio_report.get("connectCalls", 0),
            "play_clicked": bool(audio_probe.get("playClicked")),
        }
    if baseline:
        findings.extend(_diff_mounted(baseline, snap))
    return DomResult(entry=rel, findings=findings, snapshot=snap)


def save_snapshot(path, snapshot: dict) -> None:
    Path(path).write_text(json.dumps(snapshot, indent=1), encoding="utf-8")


def load_snapshot(path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
