// pedacito gui -- the Preact + HTM workspace.
//
// vendor.js is the vendored `htm/preact/standalone` bundle (no CDN, no npm at
// run time). Everything renders declaratively from one store: Server-Sent
// Events mutate the store, `emit()` schedules a re-render, and Preact's keyed
// diff updates text nodes in place -- so input focus, scroll positions and
// open dropdowns survive live updates. Model-written text always becomes
// vnodes / text, never innerHTML.
import { html, render, useState, useEffect, useRef, useMemo, useCallback } from './vendor.js';

// ------------------------------------------------------------------ helpers ---
const $ = (id) => document.getElementById(id);
const PDC = window.PDC = {};

function h(tag, props) {                    // plain-DOM builder (modals, term)
  const el = document.createElement(tag);
  const p = props || {};
  Object.keys(p).forEach((k) => {
    const v = p[k];
    if (v == null || v === false) return;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k.slice(0, 2) === 'on') el.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k === 'style') el.style.cssText = v;
    else if (k in el && k !== 'list') el[k] = v;
    else el.setAttribute(k, v === true ? '' : v);
  });
  for (let i = 2; i < arguments.length; i++) add(el, arguments[i]);
  return el;
}
function add(el, kid) {
  if (kid == null || kid === false) return;
  if (Array.isArray(kid)) { kid.forEach((k) => add(el, k)); return; }
  el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
}
const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };

PDC.h = h; PDC.$ = $; PDC.clear = clear;

PDC.api = (path, body, method) => {
  const opts = body === undefined && !method
    ? { method: 'GET' }
    : { method: method || 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {}) };
  return fetch(path, opts).then((r) => r.json().then(
    (j) => { j._status = r.status; return j; },
    () => ({ ok: false, detail: 'bad response', _status: r.status })));
};

PDC.toast = (text, isErr) => {
  const t = h('div', { class: 'toast' + (isErr ? ' err' : ''), text: text });
  $('toast').appendChild(t);
  setTimeout(() => t.remove(), isErr ? 7000 : 3500);
};

PDC.fmtInt = (n) => Number(n || 0).toLocaleString();
PDC.fmtTokens = (n) => {
  n = Number(n || 0);
  return n >= 10000 ? (n / 1000).toFixed(1) + 'k' : String(n);
};

// ------------------------------------------------------------------- modal ---
PDC.modal = (title, body, buttons, opts) => {
  opts = opts || {};
  const overlay = h('div', { class: 'overlay' });
  const close = () => { overlay.remove(); document.removeEventListener('keydown', onKey); if (opts.onClose) opts.onClose(); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  const foot = h('div', { class: 'modal-foot' });
  (buttons || []).forEach((b) => {
    foot.appendChild(h('button', {
      class: b.primary ? 'primary' : (b.danger ? 'danger' : ''), text: b.label,
      onclick: () => { const keep = b.onclick && b.onclick(close); if (keep !== false && !b.stay) close(); }
    }));
  });
  const box = h('div', { class: 'modal' + (opts.narrow ? ' narrow' : '') + (opts.wide ? ' wide' : '') },
    h('div', { class: 'modal-head' }, h('h3', { text: title }),
      h('button', { class: 'ghost', text: '\u2715', onclick: close })),
    h('div', { class: 'modal-body' }, body),
    (buttons && buttons.length) ? foot : null);
  overlay.appendChild(box);
  overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) close(); });
  document.addEventListener('keydown', onKey);
  document.body.appendChild(overlay);
  return { close: close, box: box, body: box.querySelector('.modal-body') };
};

PDC.confirm = (text, okLabel, danger) => new Promise((resolve) => {
  let done = false;
  PDC.modal('Please confirm', h('p', { text: text, style: 'white-space:pre-wrap;margin:0' }), [
    { label: 'Cancel', onclick: () => { done = true; resolve(false); } },
    { label: okLabel || 'OK', primary: !danger, danger: !!danger, onclick: () => { done = true; resolve(true); } }
  ], { narrow: true, onClose: () => { if (!done) resolve(false); } });
});

// Folder browser (used by the header, settings and the New-project option).
PDC.browse = (opts) => {
  const st = { path: opts.start || '', hidden: false, picked: '' };
  const pathIn = h('input', { type: 'text', class: 'grow mono', spellcheck: false });
  const list = h('div', { class: 'fb-list' });
  const info = h('div', { class: 'muted small-text', style: 'margin-top:6px' });
  let m;
  const go = (path) => {
    PDC.api('/api/fs?path=' + encodeURIComponent(path || '') + '&files=' + (opts.files ? 1 : 0) + '&hidden=' + (st.hidden ? 1 : 0)).then((j) => {
      st.path = j.path; pathIn.value = j.path; st.picked = '';
      clear(list);
      if (j.parent) list.appendChild(item('\u21B0', '..', () => go(j.parent)));
      j.entries.forEach((e) => {
        const full = j.path.replace(/[\\/]$/, '') + (j.path.indexOf('\\') >= 0 && j.path.indexOf('/') < 0 ? '\\' : '/') + e.name;
        if (e.dir) list.appendChild(item('\uD83D\uDCC1', e.name, () => go(full), e.project ? 'project' : ''));
        else list.appendChild(item('\uD83D\uDCC4', e.name, (el) => pick(el, full), ''));
      });
      info.textContent = j.error || (j.truncated ? 'List truncated.' : (j.entries.length + ' entries'));
    });
  };
  const pick = (el, full) => {
    Array.prototype.forEach.call(list.children, (c) => c.classList.remove('sel'));
    el.classList.add('sel'); st.picked = full;
    if (opts.files) { opts.onPick(full); m.close(); }
  };
  const item = (icon, name, fn, tag) => {
    const el = h('div', { class: 'fb-item' }, h('span', { class: 'ico', text: icon }), h('span', { text: name }), tag ? h('span', { class: 'proj', text: tag }) : null);
    el.addEventListener('click', () => fn(el));
    return el;
  };
  pathIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(pathIn.value); });
  const body = h('div', {}, h('div', { class: 'row', style: 'margin-bottom:8px' }, pathIn,
    h('button', { text: 'Go', onclick: () => go(pathIn.value) }),
    h('label', { class: 'chk' }, h('input', { type: 'checkbox', onchange: (e) => { st.hidden = e.target.checked; go(st.path); } }), 'hidden')), list, info);
  const buttons = [{ label: 'Cancel' }];
  if (!opts.files) buttons.push({ label: opts.pickLabel || 'Use this folder', primary: true, onclick: () => opts.onPick(st.path) });
  m = PDC.modal(opts.title || 'Choose', body, buttons);
  go(st.path);
};

// ------------------------------------------------------- markdown to vnodes ---
function inlineNodes(text) {
  const out = [];
  const re = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\[[^\]\n]+\]\(https?:\/\/[^)\s]+\))/g;
  let last = 0, m, k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1]) out.push(html`<code key=${k++}>${m[1].slice(1, -1)}</code>`);
    else if (m[2]) out.push(html`<b key=${k++}>${m[2].slice(2, -2)}</b>`);
    else {
      const mm = /\[([^\]]+)\]\(([^)]+)\)/.exec(m[3]);
      out.push(html`<a key=${k++} href=${mm[2]} target="_blank" rel="noopener noreferrer">${mm[1]}</a>`);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
function md(text) {
  const blocks = [];
  const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
  let i = 0, para = [];
  let list = null;                        // {items: [], ordered}
  const flushPara = () => {
    if (para.length) { blocks.push(html`<p key=${blocks.length}>${inlineNodes(para.join('\n'))}</p>`); para = []; }
  };
  const flushList = () => {
    if (!list) return;
    const items = list.items.map((it, n) => html`<li key=${n}>${inlineNodes(it)}</li>`);
    blocks.push(html`<${list.ordered ? 'ol' : 'ul'} key=${blocks.length}>${items}<//>`);
    list = null;
  };
  while (i < lines.length) {
    const ln = lines[i];
    const fence = /^\s*```\s*([\w+-]*)\s*$/.exec(ln);
    if (fence) {
      flushPara(); flushList();
      const buf = []; i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++;
      blocks.push(html`<pre key=${blocks.length}>${fence[1] ? html`<span class="lang">${fence[1]}</span>` : null}<code>${buf.join('\n')}</code></pre>`);
      continue;
    }
    const hd = /^(#{1,4})\s+(.*)$/.exec(ln);
    if (hd) { flushPara(); flushList(); blocks.push(html`<${'h' + hd[1].length} key=${blocks.length}>${inlineNodes(hd[2])}<//>`); i++; continue; }
    const li = /^\s*([-*]|\d+[.)])\s+(.*)$/.exec(ln);
    if (li) {
      flushPara();
      const ordered = /\d/.test(li[1]);
      if (!list || list.ordered !== ordered) { flushList(); list = { items: [], ordered: ordered }; }
      list.items.push(li[2]); i++; continue;
    }
    if (!ln.trim()) { flushPara(); flushList(); i++; continue; }
    flushList(); para.push(ln); i++;
  }
  flushPara(); flushList();
  return blocks;
}
PDC.md = md;

// ------------------------------------------------------------- the store ------
const S = {
  status: {},
  msgs: [], msgTick: 0,                   // msgTick bumps on any message change
  plan: { path: '', exists: false, text: '', tasks: [] },
  review: null, diff: null, diffRel: '', sel: {}, editing: false,
  editAfterLoad: false,
  ws: { tab: localStorage.getItem('pedacito_ws_tab') || 'files', files: [],
        indexed: [], indexedSet: null, sig: '', filter: '', truncated: false,
        loading: false, backups: [], backupsAt: 0 },
  mode: localStorage.getItem('pedacito_mode') || 'task',
  drafts: {}, attachments: [],
  opts: (() => { try { return JSON.parse(localStorage.getItem('pedacito_opts') || '{}'); } catch (e) { return {}; } })(),
  ui: { diffOpen: false, wsCollapsed: false, termFollow: true, optsOpen: false,
        diffMode: localStorage.getItem('pedacito_diff_mode') || 'split',
        view: 'chat', folds: {}, planEditing: false, pendingProfile: null },
  peek: null,
  how: { query: '', answer: null, loading: false, error: '', useModel: true },
};
PDC.S = S;

let subs = new Set();
function emit() {                         // Preact batches these per frame
  subs.forEach((f) => f());
}

const job = () => S.status.job || null;

// ----------------------------------------------------------- ANSI terminal ----
function Term(el) {
  this.el = el; this.lines = []; this.cur = null; this.style = ''; this.fresh = false;
  this.max = 4000; this.raf = 0; this.follow = true; this.newLine();
}
Term.prototype.newLine = function () {
  const div = h('div', { class: 'tl' });
  this.el.appendChild(div);
  this.lines.push(div);
  if (this.lines.length > this.max) { const old = this.lines.shift(); old.remove(); }
};
Term.prototype.span = function (text) {
  if (!text) return;
  if (this.fresh) { clear(this.cur); this.fresh = false; }
  if (this.style) this.cur.appendChild(h('span', { class: this.style, text: text }));
  else this.cur.appendChild(document.createTextNode(text));
};
Term.prototype.sgr = function (codes) {
  let cls = [];
  if (!codes.length) codes = [0];
  codes.forEach((c) => {
    c = Number(c);
    if (c === 0) cls = [];
    else if (c === 1) cls.push('b');
    else if (c === 2) cls.push('dim');
    else if (c >= 30 && c <= 37) cls.push('c' + c);
    else if (c >= 90 && c <= 97) cls.push('c' + (c - 60));
  });
  this.style = cls.join(' ');
};
Term.prototype.write = function (text) {
  const re = /\x1b\[([0-9;?]*)([A-Za-z])|\r\n|\n|\r|\x1b[^[]/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    this.span(text.slice(last, m.index));
    last = m.index + m[0].length;
    if (m[2] === 'm') this.sgr(m[1] ? m[1].split(';') : []);
    else if (m[2] === 'K' || m[2] === 'J') clear(this.cur);
    else if (m[0] === '\n' || m[0] === '\r\n') { this.newLine(); this.fresh = false; }
    else if (m[0] === '\r') this.fresh = true;
  }
  this.span(text.slice(last));
  this.schedule();
};
Term.prototype.schedule = function () {
  if (this.raf) return;
  this.raf = requestAnimationFrame(() => {
    this.raf = 0;
    if (this.follow) this.el.scrollTop = this.el.scrollHeight;
  });
};
Term.prototype.reset = function (text) {
  clear(this.el); this.lines = []; this.style = ''; this.fresh = false; this.newLine();
  if (text) this.write(text);
};
let term = null;
let termBuf = '', termQueued = false;
function termFeed(text) {                 // batch raw SSE terminal traffic
  termBuf += text;
  if (termQueued) return;
  termQueued = true;
  requestAnimationFrame(() => { termQueued = false; if (term && termBuf) { term.write(termBuf); termBuf = ''; } });
}

// ------------------------------------------------------------- run modes ------
const MODES = {
  task: { label: 'Task/Edit', send: 'Run',
    hint: 'Describe a change. Pedacito looks things up, edits, and shows you the diff.',
    ph: 'e.g. make the sync client retry three times and update its callers\u2026' },
  explain: { label: 'Ask', send: 'Ask',
    hint: 'Ask how the code works. Read-only \u2014 nothing is written.',
    ph: 'e.g. how does the sync client retry?' },
  auto: { label: 'Auto', send: 'Apply & run',
    hint: 'Give a goal. Pedacito plans a checklist, then applies and verifies each change automatically. Tick \u201cPlan only\u201d in Options to review the plan first. By using Auto mode you accept the tool trust boundary: it may write files and run commands in this project.',
    ph: 'e.g. add CSV export to the report page, with tests' },
  'new': { label: 'New', send: 'Create',
    hint: 'Describe a project to start from scratch in a new folder.',
    ph: 'e.g. a CLI todo manager with sqlite storage and a --json flag' },
  how: { label: 'How ?', send: 'Ask',
    hint: 'Ask how to do anything in the CLI or where a setting/feature lives in the GUI. Answers stay out of your coding thread and work even before a model is configured.',
    ph: 'e.g. how do I disable thinking?' }
};
const MODE_ORDER = ['explain', 'task', 'auto', 'new', 'how'];
const HOW_TOPICS = [
  ['Switch or import models', 'How do I switch models or import a new model?'],
  ['Tune settings for small models', 'How do I tune settings for small models?'],
  ['Toggle reasoning / thinking', 'How do I disable model thinking or reasoning?'],
  ['CLI vs GUI cheat sheet', 'Show me a CLI vs GUI cheat sheet'],
  ['Undo / restore backups', 'How do I undo an Auto or Edit run and restore a backup?'],
  ['Reset config to baseline', 'How do I reset my settings or config to the baseline?'],
  ['Reference a file in chat', 'How do I reference a file in chat?']
];
const OPTS = {
  task: [
    { k: 'write', label: 'Writing', type: 'select', def: 'review', opts: [['review', 'Review before writing'], ['apply', 'Write directly (backup first)']] },
    { k: 'plan', label: 'Planning', type: 'select', def: 'auto', opts: [['auto', 'Automatic'], ['always', 'Always plan'], ['never', 'Never']], note: 'Planning runs only with direct writes.' },
    { k: 'steps', label: 'Max lookups (0 = config)', type: 'number', def: '' },
    { k: 'reindex', label: 'Refresh index first', type: 'check', def: false }
  ],
  explain: [],
  auto: [
    { k: 'plan_only', label: 'Plan only (write the checklist, change no files)', type: 'check', def: false, span2: true },
    { k: 'plan_file', label: 'Use an existing plan file', type: 'text', browse: 'file', span2: true, ph: 'optional (default PLAN.md)' },
    { k: 'verify', label: 'Verify command after each task', type: 'text', span2: true, ph: 'e.g. pytest -q' },
    { k: 'repair', label: 'Repair rounds', type: 'number', def: '' },
    { k: 'max_tasks', label: 'Max tasks (0 = no cap)', type: 'number', def: '' },
    { k: 'max_minutes', label: 'Max minutes (0 = no cap)', type: 'number', def: '' },
    { k: 'worktree', label: 'Git worktree name', type: 'text', ph: 'optional' },
    { k: 'keep_going', label: 'Keep going after a failed task', type: 'check', def: false },
    { k: 'auto_install', label: 'Pip-install missing modules', type: 'check', def: false },
    { k: 'diagnostics', label: 'Run linter/type checker', type: 'check', def: false },
    { k: 'force', label: 'Always start a new plan', type: 'check', def: false }
  ],
  'new': [
    { k: 'project', label: 'New project folder', type: 'text', browse: 'dir', span2: true, ph: '/path/to/new-project' },
    { k: 'build', label: 'What to do', type: 'select', def: 'preview', opts: [['preview', 'Preview the plan (writes nothing)'], ['build', 'Build it'], ['review', 'Build, then review'], ['plan', 'Header skeleton only']], span2: true },
    { k: 'complex', label: 'Complex: hand it to the autonomous loop', type: 'check', def: false, span2: true },
    { k: 'verify', label: 'Verify command', type: 'text', span2: true, ph: 'e.g. python -m pytest -q' },
    { k: 'repair', label: 'Repair rounds', type: 'number', def: '' },
    { k: 'max_tasks', label: 'Max tasks', type: 'number', def: '' },
    { k: 'max_minutes', label: 'Max minutes', type: 'number', def: '' },
    { k: 'force', label: 'Allow a non-empty folder', type: 'check', def: false },
    { k: 'keep_going', label: 'Keep going after a failure', type: 'check', def: false },
    { k: 'auto_install', label: 'Pip-install missing modules', type: 'check', def: false }
  ],
  how: []
};
if (!MODES[S.mode]) S.mode = 'task';
function opt(mode, k) {
  const o = (S.opts[mode] = S.opts[mode] || {});
  if (!(k in o)) {
    const d = OPTS[mode].filter((x) => x.k === k)[0];
    o[k] = d ? d.def : '';
  }
  return o[k];
}
function saveOpts() { try { localStorage.setItem('pedacito_opts', JSON.stringify(S.opts)); } catch (e) {} }

// --------------------------------------------------------------- actions ------
let lastErr = '', lastReviewKey = '', lastJobId = 0, lastFilesSig = '';
let inputEl = null;                       // the composer textarea (uncontrolled)

function refreshState() {
  return PDC.api('/api/state').then(applyStatus, () => {});
}
PDC.refreshState = refreshState;

function applyStatus(s) {
  const prev = S.status;
  S.status = s;
  if (prev.active_thread_id && s.active_thread_id && s.active_thread_id !== prev.active_thread_id) {
    S.msgs = [];                       // a different conversation: the ids
    S.msgTick++;                       // restart, so merging would mix threads
  }
  if (Array.isArray(s.messages)) reconcileMessages(s.messages);
  if (s.error && s.error !== lastErr && !s.job) PDC.toast(s.error.split('\n')[0], true);
  lastErr = s.error || '';
  const rkey = s.review ? s.review.id + ':' + s.review.outcome : '';
  if (rkey !== lastReviewKey) { lastReviewKey = rkey; loadReview(); }
  const j = s.job;
  if (j && j.id !== lastJobId) loadPlan();
  if (!j && lastJobId) { collapseDone(); loadPlan(); loadReview(); loadFiles(); loadBackups(); }
  lastJobId = j ? j.id : 0;
  if ((s.files_sig || '') !== lastFilesSig) {   // index or workspace changed
    lastFilesSig = s.files_sig || '';
    if (S.ws.tab === 'files' && s.root) loadFiles();
  }
  if (S.ui.pendingProfile !== null && s.profile === S.ui.pendingProfile) S.ui.pendingProfile = null;
  // Colorscheme: config.toml (gui_theme) is the source of truth. On the
  // first poll the server wins only over an untouched browser (a browser-
  // local pick that could not be persisted \u2014 no user config file \u2014 must
  // not be erased on every reload). Afterwards we follow the server only
  // while nobody changed the theme here since the last known server value.
  if (typeof s.gui_theme === 'string' && s.gui_theme && s.gui_theme !== lastServerTheme) {
    let local = 'slate';
    try { local = localStorage.getItem('pedacito_theme') || 'slate'; } catch (err) {}
    const follow = lastServerTheme === null ? local === 'slate' : local === lastServerTheme;
    if (follow) applyTheme(s.gui_theme, false);
    lastServerTheme = s.gui_theme;
  }
  if (s.needs_setup && !sessionStorage.getItem('pedacito_setup_shown')) {
    sessionStorage.setItem('pedacito_setup_shown', '1');
    callSettings();
  }
  emit();
}

function collapseDone() {
  S.msgs.forEach((m) => {
    if ((m.role === 'stream' || m.role === 'activity') && m.job && !jobActive(m.job)) m.uiOpen = false;
  });
  S.msgTick++;
}
function jobActive(id) { return !!(job() && job().id === id); }

// ---- snapshot + SSE stream ----
let es = null;
function addMsg(m) {
  if (!m || m.id == null) return;
  const i = S.msgs.findIndex((x) => x.id === m.id);
  if (i >= 0) {                           // already reconciled from state
    const prev = S.msgs[i];
    const merged = { ...prev, ...m, rev: (prev.rev || 0) + 1 };
    if (m.role === 'stream' && prev.role !== 'stream') {
      merged.role = prev.role;            // don't undo a final promotion
      merged.text = prev.text;
    }
    S.msgs = S.msgs.slice();
    S.msgs[i] = merged;
  } else {
    m.rev = 0;
    if (m.role === 'stream' || m.role === 'activity') m.uiOpen = jobActive(m.job);
    S.msgs = S.msgs.concat([m]);
  }
  S.msgTick++;
}

// Fold /api/state's transcript into the live one. The server list is
// authoritative, but the response can predate an SSE event already applied in
// the page; keep those newer messages (id greater than the newest server id)
// instead of dropping them. Existing messages keep their UI state (open/closed
// folds, rendered markdown) and only re-render when their text or role changed.
function reconcileMessages(incoming) {
  if (!Array.isArray(incoming)) return;
  if (!incoming.length) {                  // an explicitly cleared transcript
    if (S.msgs.length) { S.msgs = []; S.msgTick++; }
    return;
  }
  const old = new Map(S.msgs.map((m) => [m.id, m]));
  const seen = new Set();
  let changed = incoming.length !== S.msgs.length;
  const merged = incoming.map((m) => {
    seen.add(m.id);
    const prev = old.get(m.id);
    if (!prev) { changed = true; return { ...m, rev: 0 }; }
    // A poll response can predate the msg_set that promoted a stream to the
    // final assistant answer; never let that stale copy fold it back.
    const staleStream = m.role === 'stream' && prev.role !== 'stream';
    let text = staleStream ? (prev.text || '')
      : (typeof m.text === 'string' ? m.text : (prev.text || ''));
    // Never replace a longer live stream with an older, shorter snapshot.
    if (m.role === 'stream' && prev.role === 'stream'
        && (prev.text || '').length > text.length) text = prev.text;
    const role = staleStream ? prev.role : (m.role || prev.role);
    const dirty = text !== prev.text || role !== prev.role
      || Object.keys(m).some((k) => k !== 'text' && k !== 'role'
        && k !== 'rev' && k !== 'uiOpen'
        && JSON.stringify(m[k]) !== JSON.stringify(prev[k]));
    if (dirty) changed = true;
    return { ...prev, ...m, text, role, rev: (prev.rev || 0) + (dirty ? 1 : 0) };
  });
  const newest = incoming.length ? incoming[incoming.length - 1].id : 0;
  S.msgs.forEach((m) => {
    if (!seen.has(m.id) && m.id > newest) { merged.push(m); changed = true; }
  });
  if (!changed) return;
  merged.sort((a, b) => a.id - b.id);
  S.msgs = merged;
  S.msgTick++;
}
function loadSnapshot() {
  return PDC.api('/api/snapshot').then((snap) => {
    if (snap._status !== 200) return;
    S.msgs = (snap.messages || []).map((m) => ({ ...m, rev: 0 }));
    S.msgTick++;
    S.status = snap.status || {};
    lastJobId = S.status.job ? S.status.job.id : 0;
    lastReviewKey = S.status.review ? S.status.review.id + ':' + S.status.review.outcome : '';
    lastFilesSig = S.status.files_sig || '';
    if (term) term.reset(snap.term || '');
    connect(snap.seq || 0);
    loadReview();
    loadPlan();
    if (S.status.root && S.ws.tab === 'files') loadFiles(true);
    emit();
  });
}
function connect(seq) {
  if (es) { es.close(); es = null; }
  const stream = new EventSource('/events?since=' + seq);
  es = stream;
  const on = (type, fn) => stream.addEventListener(type, (e) => fn(JSON.parse(e.data)));
  on('msg_new', (e) => { addMsg(e.msg); emit(); });
  on('msg_append', (e) => {
    const m = S.msgs.find((x) => x.id === e.id);
    if (!m) { refreshState(); return; }   // fell out of the local view
    m.text += e.text; m.rev = (m.rev || 0) + 1; S.msgTick++;
    emit();
  });
  on('msg_set', (e) => {
    const i = S.msgs.findIndex((x) => x.id === e.id);
    if (i < 0) { refreshState(); return; }
    const prev = S.msgs[i];
    const fields = { ...e.fields };
    if (prev.role === 'assistant' && fields.role === 'stream') {
      delete fields.role;                 // never undo a final promotion
    }
    S.msgs = S.msgs.slice();
    S.msgs[i] = { ...prev, ...fields, rev: (prev.rev || 0) + 1 };
    S.msgTick++; emit();
  });
  on('msgs_clear', () => { S.msgs = []; S.msgTick++; emit(); });
  on('term', (e) => termFeed(e.text));
  on('term_clear', () => { if (term) term.reset(''); });
  on('usage', () => refreshState());
  on('job', () => refreshState());
  ['status', 'review', 'review_done'].forEach((t) => on(t, () => refreshState()));
  stream.addEventListener('resync', () => { stream.close(); es = null; loadSnapshot(); });
}

// ---- project ----
function openProject(path, profile) {
  if (!path) { PDC.toast('Enter a project folder (or use Browse).', true); return; }
  PDC.api('/api/project', { path: path, profile: profile || '' }).then((j) => {
    if (!j.ok) PDC.toast(j.detail || 'busy', true);
    refreshState();
  });
}
let peekTimer = 0, peeked = '';
function peek(path) {                     // list profiles of a folder not yet
  clearTimeout(peekTimer);                // opened (header only; no state change)
  peekTimer = setTimeout(() => {
    if (!path || path === peeked || path === S.status.root) return;
    PDC.api('/api/peek', { path: path }).then((j) => {
      if (!j.ok) return;
      peeked = path;
      S.peek = j;
      emit();
    });
  }, 450);
}
function switchProfile(name) {
  S.ui.pendingProfile = name;
  emit();
  PDC.api('/api/profile', { profile: name }).then((j) => {
    if (!j.ok) {
      S.ui.pendingProfile = null;
      PDC.toast(j.detail || 'busy', true);
    }
    refreshState(); emit();
  });
}
function buildIndex(summarise) {
  PDC.api('/api/index', { summarise: summarise }).then((j) => {
    if (!j.ok) PDC.toast(j.detail || 'busy', true);
    refreshState();
  });
}

// ---- chat threads (the project's saved conversations) ----
// Threads live in `.pedacito/gui/threads/`; the switcher in the Chat head
// lists them, `+ New` starts another one without losing this one, and the
// page resyncs after any switch (message ids restart per thread).
function swapTranscript() {
  S.msgs = [];
  S.msgTick++;
  emit();
  refreshState();
}
function newThread() {
  PDC.api('/api/threads/new', { profile: S.status.profile || '' }).then((j) => {
    if (!j.ok) { PDC.toast(j.detail || 'could not start a new chat', true); return; }
    swapTranscript();
  });
}
function selectThread(id) {
  if (!id || id === S.status.active_thread_id) return;
  PDC.api('/api/threads/select', { id: id }).then((j) => {
    if (!j.ok) { PDC.toast(j.detail || 'could not open that chat', true); return; }
    swapTranscript();
  });
}
function deleteCurrentThread() {
  const s = S.status;
  PDC.confirm('Delete \u201c' + (s.thread_title || 'this conversation') +
              '\u201d?\n\nIts thread file under .pedacito/gui/threads/ is removed. This cannot be undone.',
              'Delete', true).then((ok) => {
    if (!ok) return;
    PDC.api('/api/threads/delete', { id: s.active_thread_id }).then((j) => {
      if (!j.ok) { PDC.toast(j.detail || 'could not delete', true); return; }
      PDC.toast('Conversation deleted.');
      swapTranscript();
    });
  });
}

// ---- run / stop / steer ----
PDC.startRun = (payload) => PDC.api('/api/run', payload).then((j) => {
  if (!j.ok) PDC.toast(j.detail || 'could not start', true);
  refreshState();
  return j;
});
const stick = { current: true };          // chat auto-follow flag
function stop() { PDC.api('/api/stop', {}).then(refreshState); }

// ---- How / Help -------------------------------------------------------------
function askHow(query) {
  query = (query || '').trim();
  if (!query) return;
  S.how.query = query;
  S.how.loading = true;
  S.how.error = '';
  S.how.answer = null;
  emit();
  PDC.api('/api/how', { query: query, use_model: !!S.how.useModel }).then((j) => {
    S.how.loading = false;
    if (j._status !== 200 || !j.ok) {
      S.how.error = j.detail || 'Could not build a help answer.';
    } else {
      S.how.answer = j;
    }
    emit();
  }, () => {
    S.how.loading = false;
    S.how.error = 'The GUI server did not answer.';
    emit();
  });
}
function copyText(text) {
  const done = () => PDC.toast('Copied to the clipboard.');
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
  } else fallbackCopy(text, done);
}
function fallbackCopy(text, done) {
  const ta = h('textarea', { value: text, style: 'position:fixed;left:-9999px;top:0' });
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand('copy'); done(); }
  catch (e) { PDC.toast('Copy failed; select the command manually.', true); }
  ta.remove();
}
function openSetting(name) {
  if (name === 'default_profile') { callSettings({ tab: 'profiles' }); return; }
  callSettings({ tab: 'all', search: name });
}

function send() {
  const s = S.status, j = s.job;
  const text = (S.drafts[S.mode] || '').trim();
  if (S.mode === 'how') {
    if (!text) { PDC.toast('Type a question first.', true); return; }
    S.drafts.how = '';
    if (inputEl) inputEl.value = '';
    askHow(text);
    return;
  }
  if (j) {
    if (!text) return;
    if (j.kind === 'task' && s.allow_steering) {
      PDC.api('/api/steer', { text: text }).then((r) => { if (!r.ok) PDC.toast(r.detail || 'not delivered', true); });
      S.drafts[S.mode] = '';
      if (inputEl) inputEl.value = '';
    }
    return;
  }
  if (!text && !S.attachments.length && !(S.mode === 'auto' && opt('auto', 'plan_file'))) { PDC.toast('Type something first.', true); return; }
  if (S.mode !== 'new' && !s.root) { PDC.toast('Open a project first.', true); return; }
  if (S.mode !== 'new' && !s.has_index) { PDC.toast('Build the index first.', true); return; }
  const body = { mode: S.mode, text: text, attachments: S.attachments.slice() };
  OPTS[S.mode].forEach((d) => { body[d.k] = opt(S.mode, d.k); });
  if (S.mode === 'auto') {
    body.plan_only = !!body.plan_only;
    body.apply = !body.plan_only;      // Auto applies unless Plan only is ticked
  }
  PDC.startRun(body).then((r) => {
    if (r.ok) {
      S.drafts[S.mode] = '';
      if (inputEl) inputEl.value = '';
      S.attachments = []; stick.current = true; emit();
    }
  });
}
function attachFiles(files) {
  Array.prototype.forEach.call(files, (f) => {
    if (f.size > 400000) { PDC.toast(f.name + ' is larger than 400 KB.', true); return; }
    f.text().then((t) => {
      if (t.indexOf('\u0000') >= 0) { PDC.toast(f.name + ' looks binary.', true); return; }
      S.attachments.push({ name: f.name, text: t }); emit();
    });
  });
}

// ---- review (staged diff) ----
function loadReview() {
  return PDC.api('/api/review').then((j) => {
    const meta = j.review || null;
    const isNew = !!meta && (!S.review || meta.id !== S.review.id);
    const changed = meta ? (isNew || meta.outcome !== S.review.outcome)
                         : !!S.review;
    S.review = meta;
    if (!meta) { S.diff = null; S.editing = false; emit(); return; }
    if (!changed) { emit(); return; }
    S.editing = false;
    S.editAfterLoad = false;
    S.sel = {};
    meta.files.forEach((f) => { S.sel[f.rel] = true; });
    S.ui.folds = {};
    const fresh = !meta.files.some((f) => f.rel === S.diffRel);
    if (fresh) S.diffRel = ((meta.files.filter((f) => !f.source)[0] || meta.files[0] || {}).rel) || '';
    if (meta.pending && (isNew || fresh)) {
      S.ui.diffOpen = true;               // stage it, expand it: chat beside it
      if (isNew) PDC.toast('Staged review \u2014 '
        + meta.files.filter((f) => !f.source).length + ' file(s) awaiting your decision.');
    }
    if (S.diffRel) loadDiffFile(S.diffRel); else { S.diff = null; emit(); }
  });
}
function loadDiffFile(rel) {
  if (!S.review) return;
  S.diffRel = rel; S.diff = null; S.ui.folds = {};
  emit();
  PDC.api('/api/review/file?id=' + S.review.id + '&rel=' + encodeURIComponent(rel)).then((d) => {
    S.diff = d._status === 200 ? d : null;
    if (S.editAfterLoad) {                // the \u270E on a file-strip row
      S.editAfterLoad = false;
      S.editing = !!S.diff;
    }
    emit();
  });
}
function decide(action) {
  if (!S.review) return;
  if (S.editing) { PDC.toast('Click "\u2716 Show diff" to keep your edit first.', true); return; }
  const files = S.review.partial_ok
    ? S.review.files.filter((f) => !f.source && S.sel[f.rel] !== false).map((f) => f.rel)
    : null;
  PDC.api('/api/review/decide', { id: S.review.id, action: action, files: files }).then((j) => {
    if (j.ok) PDC.toast(action === 'apply' ? j.detail || 'Changes written.' : 'Rejected. No files were changed.');
    else PDC.toast(j.detail || 'failed', true);
    loadReview(); refreshState();
  });
}
const editBox = { current: null };
function toggleEdit() {
  if (!S.review || !S.diff) return;
  if (!S.editing) { S.editing = true; emit(); return; }
  const text = editBox.current ? editBox.current.value : S.diff.after;
  PDC.api('/api/review/rediff', { id: S.review.id, rel: S.diffRel, text: text }).then((d) => {
    S.editing = false;
    if (d._status === 200) {
      S.diff = d;
      S.review.files.forEach((f) => { if (f.rel === S.diffRel) { f.added = d.added; f.removed = d.removed; f.edited = true; } });
    } else PDC.toast(d.detail || 'could not update', true);
    emit();
  });
}

// ---- workspace panel (files / backups / logs) ----
function loadFiles(force) {
  if (!S.status.root) { S.ws.files = []; S.ws.indexed = []; return Promise.resolve(); }
  if (!force && S.ws.sig === (S.status.files_sig || '')) { emit(); return Promise.resolve(); }
  S.ws.loading = true; emit();
  return PDC.api('/api/files').then((j) => {
    S.ws.loading = false;
    if (j._status !== 200 || !Array.isArray(j.files)) { emit(); return; }
    S.ws.files = j.files;
    S.ws.indexed = j.indexed || [];
    S.ws.indexedSet = new Set(S.ws.indexed);
    S.ws.sig = j.sig || '';
    S.ws.truncated = !!j.truncated;
    emit();
  }, () => { S.ws.loading = false; emit(); });
}
function loadBackups(force) {
  if (!S.status.root) { S.ws.backups = []; emit(); return Promise.resolve(); }
  if (!force && S.ws.backups.length && Date.now() - S.ws.backupsAt < 4000) return Promise.resolve();
  return PDC.api('/api/backups').then((j) => {
    if (j._status === 200) { S.ws.backups = j.backups || []; S.ws.backupsAt = Date.now(); }
    emit();
  }, () => {});
}
function setWsTab(t) {
  if (S.ws.tab === t && !S.ui.wsCollapsed) {      // the tab already showing:
    if (t === 'files') loadFiles(true);           // click again = rescan
    else if (t === 'backups') loadBackups(true);
    return;
  }
  S.ws.tab = t;
  S.ui.wsCollapsed = false;
  try { localStorage.setItem('pedacito_ws_tab', t); } catch (e) {}
  if (t === 'files') loadFiles();
  if (t === 'backups') loadBackups();
  if (t === 'logs' && term) { term.follow = S.ui.termFollow; term.schedule(); }
  emit();
}
function fmtTime(t) {
  return t ? new Date(t * 1000).toLocaleString() : '';
}
function insertRef(rel) {                          // "@" the file into the
  const el = inputEl || $('input');                // chat composer
  const tok = '@' + rel;
  if (el) {
    el.value = (el.value ? (/\s$/.test(el.value) ? el.value : el.value + ' ') : '') + tok + ' ';
    S.drafts[S.mode] = el.value;
    el.focus();
    el.selectionStart = el.selectionEnd = el.value.length;
  } else {
    S.drafts[S.mode] = ((S.drafts[S.mode] || '') + (S.drafts[S.mode] && !/\s$/.test(S.drafts[S.mode]) ? ' ' : '') + tok + ' ');
  }
  PDC.toast('Added ' + tok + ' to the chat input.');
  emit();
}
function previewFile(rel) {                        // read-only file preview
  PDC.api('/api/file?path=' + encodeURIComponent(rel)).then((j) => {
    if (j._status !== 200) { PDC.toast(j.detail || 'could not read ' + rel, true); return; }
    let body;
    if (j.binary) {
      body = h('div', { class: 'tip', text: 'Binary file \u2014 ' + PDC.fmtInt(j.size) + ' bytes. No text preview.' });
    } else {
      const lines = String(j.text || '').replace(/\r\n/g, '\n').split('\n');
      const box = h('div', { class: 'pv' });
      lines.forEach((ln, i) => box.appendChild(h('div', { class: 'pv-line' },
        h('span', { class: 'pv-n', text: String(i + 1) }),
        h('span', { class: 'pv-t', text: ln }))));
      body = h('div', {}, box,
        j.truncated ? h('div', { class: 'muted small-text', text: 'Preview truncated at 600 KB.' }) : null);
    }
    PDC.modal(rel + ' \u00B7 ' + PDC.fmtInt(j.size) + ' bytes', body,
      [{ label: '\u2295 Use in chat', onclick: () => insertRef(rel) }], { wide: true });
  });
}
function restoreBackup(name) {
  if (!name) { PDC.toast('There is no backup snapshot to restore.', true); return; }
  PDC.confirm('Restore \u201c' + name + '\u201d?\n\nCopies the snapshot back over the working tree. Files changed since then are overwritten; files created since stay on disk.',
    'Restore', true).then((ok) => {
      if (!ok) return;
      PDC.api('/api/restore', { snapshot: name }).then((j) => {
        PDC.toast(j.detail || (j.ok ? 'restored' : 'failed'), !j.ok);
        refreshState(); loadFiles(true); loadBackups(true);
      });
    });
}

// ---- plan ----
const planEdit = { current: null };
function loadPlan() {
  return PDC.api('/api/plan').then((j) => {
    if (j._status !== 200) return;
    S.plan = j;
    emit();
  });
}
function savePlan() {
  const text = planEdit.current ? planEdit.current.value : '';
  PDC.api('/api/plan', { text: text }).then((j) => {
    if (j.ok) { PDC.toast('Plan saved.'); S.ui.planEditing = false; loadPlan(); }
    else PDC.toast(j.detail || 'could not save', true);
  });
}
PDC.runPlan = function () {
  const info = S.plan || {};
  if (!info.exists) { PDC.toast('There is no plan to run yet.', true); return; }
  const a = (S.opts && S.opts.auto) || {};
  const s = S.status;
  const goal = (s.last_auto && s.last_auto.text && info.path === s.plan_file) ? s.last_auto.text : '';
  PDC.startRun({ mode: 'auto', text: goal, plan_file: info.path, plan_only: false, apply: true, verify: a.verify || '', repair: a.repair || '',
    max_tasks: a.max_tasks || '', max_minutes: a.max_minutes || '', keep_going: !!a.keep_going, auto_install: !!a.auto_install, diagnostics: !!a.diagnostics });
};

// ---- menu actions ----
function checkModel() { PDC.api('/api/check_model', {}).then(refreshState); }

// Quick-calibrate the profile shown in the header from the top bar. Same
// endpoint and wording as the Settings -> Model discovery card button.
function calibrateProfile(name) {
  if (!name) { PDC.toast('No active profile to calibrate.', true); return; }
  PDC.confirm('Calibrate \u201c' + name + '\u201d?\n\nA short probe: a few timed calls to measure how fast this model answers, how long a prompt it can actually hold, and how reliably it uses tools. The result is stored in its model card and the tuned limits are written to the profile.', 'Run it').then((ok) => {
    if (!ok) return;
    PDC.api('/api/models/calibrate', { profile: name, mode: 'quick' }).then((j) => {
      if (!j.ok) { PDC.toast(j.detail || 'could not start the calibration', true); return; }
      PDC.toast('Calibrating "' + name + '" \u2014 watch the Output tab.');
      refreshState();
    });
  });
}

// The GUI colorscheme. The browser always gets it at once (localStorage, so
// theme.js can apply it before first paint); `userPick` additionally persists
// it as gui_theme in the user config through /api/settings/theme — purely
// cosmetic, so the project is never reloaded for it. If the write fails (no
// user config file yet, server gone), the choice stays browser-local.
let lastServerTheme = null;
function applyTheme(t, userPick) {
  if (!t) return;
  if (t === 'slate') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', t);
  try { localStorage.setItem('pedacito_theme', t); } catch (err) {}
  const sel = $('theme');
  if (sel && Array.prototype.some.call(sel.options, (o) => o.value === t)) sel.value = t;
  if (!userPick) return;
  PDC.api('/api/settings/theme', { gui_theme: t }).then((j) => {
    if (j && j.ok) {
      lastServerTheme = t;
      if (!j.persisted) PDC.toast('Theme kept in this browser: create a user config to store it in config.toml.');
    } else PDC.toast('Theme kept in this browser (config not written): ' + ((j && j.detail) || 'server error'), true);
  }).catch(() => PDC.toast('Theme kept in this browser (server unreachable).', true));
}
function clearTerm() { PDC.api('/api/term_clear', {}); }
function restoreDialog() {
  const snaps = S.status.snapshots || [];
  const sel = h('select', {}, snaps.map((n, i) => h('option', { value: n, text: n + (i === 0 ? '  (newest)' : '') })));
  PDC.modal('Restore a backup', h('div', {}, h('p', { text: 'Copies the chosen snapshot back over the working tree. Files changed since then are overwritten; files created since stay on disk.' }), sel), [
    { label: 'Cancel' },
    { label: 'Restore', danger: true, onclick: () => {
      PDC.api('/api/restore', { snapshot: sel.value }).then((j) => { PDC.toast(j.detail || (j.ok ? 'restored' : 'failed'), !j.ok); refreshState(); });
    } }], { narrow: true });
}

// ------------------------------------------------------------------ layout ----
function setVar(name, px) { document.documentElement.style.setProperty(name, px + 'px'); }
function startDrag(e, onMove) {
  e.preventDefault();
  const move = (ev) => onMove(ev);
  const up = () => {
    document.body.style.userSelect = '';
    document.removeEventListener('mousemove', move);
    document.removeEventListener('mouseup', up);
    saveLayout();
  };
  document.body.style.userSelect = 'none';
  document.addEventListener('mousemove', move);
  document.addEventListener('mouseup', up);
}
function saveLayout() {
  const cs = getComputedStyle(document.documentElement);
  try {
    localStorage.setItem('pedacito_layout', JSON.stringify({
      left: parseInt(cs.getPropertyValue('--left')) || 460,
      right: parseInt(cs.getPropertyValue('--right')) || 340,
      term: parseInt(cs.getPropertyValue('--term-h')) || 220 }));
  } catch (e) {}
}
function dragChat(appRef) {
  return (e) => startDrag(e, (ev) => {
    const r = appRef.current.getBoundingClientRect();
    setVar('--left', Math.max(300, Math.min(ev.clientX - r.left, r.width - 340)));
  });
}
function dragRight(appRef) {
  return (e) => startDrag(e, (ev) => {
    const r = appRef.current.getBoundingClientRect();
    setVar('--right', Math.max(240, Math.min(r.right - ev.clientX, 620)));
  });
}
function dragTerm(paneRef) {
  return (e) => startDrag(e, (ev) => {
    const r = paneRef.current.getBoundingClientRect();
    setVar('--term-h', Math.max(60, Math.min(r.bottom - ev.clientY, r.height - 120)));
  });
}
function setView(v) {
  S.ui.view = v;
  document.body.dataset.view = v;
  emit();
}

// =============================================================== components ===

function ProfileSelect(props) {           // header dropdown; uncontrolled
  const s = S.status;
  const peek = props.peek;
  const profiles = peek ? peek.profiles : (s.profiles || []);
  const defName = peek ? peek.default : s.default_profile;
  const active = S.ui.pendingProfile !== null ? S.ui.pendingProfile
    : (s.profile && s.profile !== defName ? s.profile : '');
  const ref = useRef(null);
  useEffect(() => {                       // sync only when it truly differs,
    const el = ref.current;                // and never while it is open/focused
    if (el && el.value !== active && document.activeElement !== el) el.value = active;
  }, [active, profiles.join(',')]);
  return html`
    <select id="profile" ref=${ref} title="active model profile -- switch it any time"
            disabled=${!!props.busy}
            onchange=${(e) => props.onChange(e.target.value)}>
      <option value="">(default${defName ? ': ' + defName : ''})</option>
      ${profiles.map((p) => html`<option key=${p} value=${p}>${p}${(s.profile_models || {})[p] ? ' \u00B7 ' + s.profile_models[p] : ''}</option>`)}
    </select>`;
}

function TopBar() {
  const s = S.status, j = job();
  const pathRef = useRef(null);
  useEffect(() => {                       // mirror the open project into the
    const el = pathRef.current;            // path box without stealing focus
    if (el && s.root && document.activeElement !== el && el.value !== s.root) el.value = s.root;
  }, [s.root]);
  const peekOn = S.peek && pathRef.current && pathRef.current.value.trim() &&
    pathRef.current.value.trim() !== s.root && !j;
  const indexed = s.indexed || s.has_index;
  const idxTitle = indexed
    ? (s.index_time ? 'index from ' + new Date(s.index_time * 1000).toLocaleString() : 'index present')
    : 'no index yet -- click Index to build one (Shift-click for a quick, no-summaries index)';
  const open = () => openProject((pathRef.current && pathRef.current.value || '').trim(), S.ui.pendingProfile || '');
  const pending = S.ui.pendingProfile;
  const canSwitch = pending !== null && pending !== s.profile;
  const activeProf = s.profile || s.default_profile || '';
  return html`
    <header id="topbar">
      <div class="tb-row" id="tbRow1">
        <div class="brand">pe<span>da</span>cito</div>
        <div id="projectBar">
          <input type="text" id="path" ref=${pathRef} placeholder="/path/to/your/project"
                 spellcheck="false" autocomplete="off"
                 oninput=${(e) => { S.peek = null; peeked = ''; peek(e.target.value.trim()); }}
                 onkeydown=${(e) => { if (e.key === 'Enter') open(); }} />
          <button id="browseBtn" title="browse folders on this machine"
                  onClick=${() => PDC.browse({ title: 'Choose a project folder',
                    start: ((pathRef.current && pathRef.current.value) || s.root || '').replace(/[\\/][^\\/]*[\\/]$/, ''),
                    pickLabel: 'Open this folder',
                    onPick: (p) => { if (pathRef.current) pathRef.current.value = p; openProject(p, S.ui.pendingProfile || ''); } })}>Browse…</button>
          <select id="recent" title="recent projects"
                  onchange=${(e) => { const v = e.target.value; e.target.value = ''; if (v) { if (pathRef.current) pathRef.current.value = v; openProject(v, ''); } }}>
            <option value="">recent\u2026</option>
            ${(s.recent || []).map((r) => html`<option key=${r} value=${r}>${r}</option>`)}
          </select>
          <button class="primary" id="openBtn" disabled=${!!j} onClick=${open}>Open</button>
          <${Menu} button=${html`<button id="projMenuBtn" title="index, restore and other project actions">Project ▾</button>`}>
            <${MenuItem} label="Check model (loads it)" disabled=${!!j || !s.root} action=${checkModel} />
            <${MenuItem} label="Restore from a backup\u2026" disabled=${!!j || !(s.snapshots || []).length} action=${restoreDialog} />
            <${MenuItem} label="Clear terminal" action=${clearTerm} />
            <${MenuItem} label="Clear chat" action=${() => PDC.api('/api/clear', {})} />
          <//>
        </div>
        <div id="topRight">
          <select id="theme" title="color scheme (saved as gui_theme in config.toml)"
                  onchange=${(e) => applyTheme(e.target.value, true)}>
            <optgroup label="Light">
              <option value="light">Light \u2014 GitHub</option>
              <option value="daylight">Daylight \u2014 warm paper</option>
            </optgroup>
            <optgroup label="Dark">
              <option value="slate">Slate \u2014 GitHub Dark</option>
              <option value="nord">Nord \u2014 Catppuccin</option>
              <option value="midnight">Midnight \u2014 high contrast</option>
            </optgroup>
            <optgroup label="1990s">
              <option value="jazz_solo">Jazz Cup \u2014 '91 teal/purple</option>
              <option value="trapper_keeper">Trapper Keeper \u2014 cobalt/coral</option>
              <option value="memphis_arcade">Memphis Arcade \u2014 neon geometry</option>
              <option value="windbreaker">Windbreaker \u2014 color-block nylon</option>
            </optgroup>
          </select>
          <button id="settingsBtn" title="profiles and settings" onClick=${() => callSettings()}>\u2699 Settings</button>
          <button class="danger" id="exitBtn" title="stop the pedacito gui server"
                  onClick=${() => PDC.confirm('Stop the pedacito gui server? Run "pedacito gui" again to reopen it.', 'Stop server', true).then((ok) => {
                    if (!ok) return;
                    PDC.api('/api/shutdown', {}).catch(() => {}).then(() => {
                      document.body.innerHTML = '<div style="padding:80px 20px;text-align:center;font:16px sans-serif;color:#8b91a0">pedacito gui has stopped. You can close this tab.</div>';
                    });
                  })}>Exit</button>
        </div>
      </div>
      <div class="tb-row" id="tbRow2">
        <${ProfileSelect} busy=${j} peek=${peekOn ? S.peek : null}
                          onChange=${(name) => { S.ui.pendingProfile = name; emit(); }} />
        <button id="switchBtn" class="small" disabled=${!!j || !s.root || !canSwitch}
                title=${s.root ? 'switch the open project to the profile picked above'
                               : 'remembered: applied when you open a project'}
                onClick=${() => { if (s.root) switchProfile(pending === null ? '' : pending); }}>Switch</button>
        <button id="calibrateBtn" class="small" disabled=${!!j || !activeProf}
                title=${'quick-calibrate "' + (activeProf || '') + '" (measured scores and limits go into its model card; output streams to the terminal)'}
                onClick=${() => calibrateProfile(activeProf)}>Calibrate</button>
        <span class="chip" id="modelChip"
              title=${(s.model_status || 'click to check the model server') + (s.base_url ? '\n' + s.base_url : '') + (s.context_tokens ? '\nwindow ' + PDC.fmtInt(s.context_tokens) + ' tokens (' + s.context_origin + ')' : '')}
              onClick=${() => { if (s.root && !j) checkModel(); }}>
          <i class=${'dot ' + (j ? 'busy' : s.model_ok === true ? 'ok' : s.model_ok === false ? 'bad' : 'idle')}></i>
          <b>${s.root ? (s.model || '\u2014') : 'no project'}</b>
          ${s.root && s.model ? html`<span class="tag ${s.disable_thinking ? '' : 'act'}"
            title=${s.disable_thinking ? 'thinking is disabled for this profile (disable_thinking = true)'
                                       : 'thinking is allowed (disable_thinking = false) \u2014 turn it off in Settings \u2192 Profiles'}>${s.disable_thinking ? 'no-think' : 'thinking'}</span>` : null}
        </span>
        <div class="tb-right">
          <div id="indexBox">
            <span class="chip" id="indexChip" title=${idxTitle}>
              <i class=${'dot ' + (indexed ? 'ok' : 'idle')}></i>
              <b>${indexed ? 'indexed' : 'no index'}</b>
              ${indexed ? html`<span class="muted small-text">${PDC.fmtInt(s.index_files)} files \u00B7 ${PDC.fmtInt(s.index_chunks)} units</span>` : null}
            </span>
            <button id="indexBtn" class="small" disabled=${!!j || !s.root}
                    onClick=${(e) => buildIndex(!e.shiftKey)}>Index</button>
          </div>
          ${s.root ? html`<span class="chip" id="threadChip"
            title=${'active conversation: ' + (s.thread_title || '') + ' \u2014 switch threads in the Chat panel'}>
            <b class="ellip">${s.thread_title || 'chat'}</b>
          </span>` : null}
          ${(s.usage && s.usage.calls) ? html`
            <span class="chip" id="usageChip" title=${'model usage this session (' + Math.round(s.usage.seconds) + 's)'}>
              <b>${s.usage.calls + ' calls \u00B7 ' + PDC.fmtTokens(s.usage.tokens_in) + ' in / ' + PDC.fmtTokens(s.usage.tokens_out) + ' out'
                + (s.usage.cache_hit ? ' \u00B7 cache ' + Math.round(s.usage.cache_hit * 100) + '%' : '')
                + (s.usage.cost ? ' \u00B7 $' + Number(s.usage.cost).toFixed(4) : '')}</b>
            </span>` : null}
        </div>
      </div>
    </header>`;
}

function Menu(props) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);
  useEffect(() => {
    if (!open) return;
    const closeOn = (e) => { if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    const t = setTimeout(() => document.addEventListener('mousedown', closeOn), 0);
    document.addEventListener('keydown', onKey);
    return () => { clearTimeout(t); document.removeEventListener('mousedown', closeOn); document.removeEventListener('keydown', onKey); };
  }, [open]);
  return html`
    <span class="menuwrap" ref=${wrapRef}>
      <span class="menubtn" onClick=${() => setOpen(!open)}>${props.button}</span>
      ${open ? html`<div class="dropdown" role="menu" onClick=${() => setOpen(false)}>${props.children}</div>` : null}
    </span>`;
}
function MenuItem(props) {
  return html`<button class="menuitem" disabled=${!!props.disabled}
                       onClick=${() => { if (!props.disabled) props.action(); }}>${props.label}</button>`;
}

// ---------------------------------------------------------------- how tab ----
function PathBadges(props) {
  const parts = String(props.text || '').split(/\s*→\s*/);
  return html`<span class="how-path">${parts.map((p, i) =>
    html`<span key=${i} class="how-step">${p}</span>`)}</span>`;
}

function CopyBlock(props) {
  return html`<div class=${'how-cmd' + (props.lang ? ' ' + props.lang : '')}>
    <pre><code>${props.text}</code></pre>
    <button class="ghost small" title="copy to the clipboard"
            onClick=${() => copyText(props.text)}>Copy</button>
  </div>`;
}

function HowFlags(props) {
  const flags = props.flags || [];
  const common = props.common || [];
  if (!flags.length && !common.length) return null;
  return html`<details class="how-flags">
    <summary>Flags (${flags.length}${common.length ? ' + ' + common.length + ' common' : ''})</summary>
    <ul>
      ${flags.map((f, i) => html`<li key=${'f' + i}><code>${f.flag}</code> — ${f.help}</li>`)}
      ${common.length ? html`<li class="muted"><b>Common project flags</b></li>` : null}
      ${common.map((f, i) => html`<li key=${'c' + i}><code>${f.flag}</code> — ${f.help}</li>`)}
    </ul>
  </details>`;
}

function HowAnswer(props) {
  const r = props.r || {};
  const M = r.matches || {};
  const gui = M.gui || [], cli = M.cli || [], settings = M.settings || [];
  return html`<div class="how-answer">
    <div class="how-meta">
      <span class=${'tag ' + (r.used_model ? 'act' : '')}>${r.used_model ? 'model synthesis' : 'deterministic reference'}</span>
      ${r.model_error ? html`<span class="muted small-text">model skipped: ${r.model_error}</span>` : null}
    </div>
    ${r.used_model && r.answer ? html`<div class="how-model md">${md(r.answer)}</div>` : null}
    ${gui.length ? html`<section class="how-sec">
      <h4>In the GUI</h4>
      ${gui.map((item, i) => html`<div key=${'g' + i} class="how-card">
        <div class="how-card-title">${item.title}</div>
        <div class="how-desc">${item.summary}</div>
        ${(item.steps || []).map((s, n) => html`<div key=${n} class="how-step-row">
          <${PathBadges} text=${s} /></div>`)}
      </div>`)}
    </section>` : null}
    ${cli.length ? html`<section class="how-sec">
      <h4>In the Terminal (CLI)</h4>
      ${cli.map((item, i) => html`<div key=${'c' + i} class="how-card">
        <div class="how-card-title"><code>pedacito ${item.name}</code></div>
        <div class="how-desc">${item.summary}</div>
        <div class="muted small-text"><code>${item.usage}</code></div>
        ${(item.examples || []).map((ex, n) => html`<${CopyBlock} key=${n} text=${ex} lang="sh" />`)}
        <${HowFlags} flags=${item.flags} common=${item.common ? (r.common_flags || []) : []} />
      </div>`)}
    </section>` : null}
    ${settings.length ? html`<section class="how-sec">
      <h4>Relevant Settings (<code>config.toml</code>)</h4>
      ${settings.map((item, i) => html`<div key=${'s' + i} class="how-card">
        <div class="how-card-title"><code>${item.key}</code>
          <span class="tag">${item.default_text}</span></div>
        <div class="how-desc">${item.help}</div>
        <div class="how-step-row"><${PathBadges} text=${item.gui} /></div>
        <div class="row how-actions">
          <button class="small" onClick=${() => openSetting(item.key)}>Open in Settings</button>
          ${item.profile_gui ? html`<span class="muted small-text">${item.profile_gui}</span>` : null}
        </div>
        <${CopyBlock} text=${item.toml} lang="toml" />
      </div>`)}
    </section>` : null}
    ${!gui.length && !cli.length && !settings.length ? html`
      <div class="tip">No reference item matched that question. Try one of the
      topic chips above, or ask about a command, a GUI panel, or a setting
      name.</div>` : null}
  </div>`;
}

function HowPane() {
  const hw = S.how;
  const ask = (q) => {
    S.drafts.how = q;
    if (inputEl) {
      inputEl.value = q;
      inputEl.focus();
      inputEl.selectionStart = inputEl.selectionEnd = q.length;
    }
    askHow(q);
  };
  return html`<div id="howPane">
    <div class="how-hero">
      <h3>How / Help</h3>
      <div class="how-lede">Ask how to do anything in the CLI, where a feature
        lives in the GUI, or what a setting does. This pane never touches your
        coding thread, and the reference answer works without a model.</div>
      <div class="how-chips">
        ${HOW_TOPICS.map((t, i) => html`<button key=${i} class="how-chip"
          onClick=${() => ask(t[1])}>${t[0]}</button>`)}
      </div>
      <label class="chk how-model"><input type="checkbox" checked=${hw.useModel}
        onchange=${(e) => { S.how.useModel = e.target.checked; emit(); }} />
        Use model synthesis when a model server is reachable (falls back to the
        built-in reference automatically)</label>
    </div>
    ${hw.query ? html`<div class="how-question"><span class="tag">question</span>
      <b>${hw.query}</b></div>` : null}
    ${hw.loading ? html`<div class="tip">Searching the CLI, GUI navigation and
      settings reference…</div>` : null}
    ${hw.error ? html`<div class="note-box err">${hw.error}</div>` : null}
    ${hw.answer && !hw.loading ? html`<${HowAnswer} r=${hw.answer} />` : null}
  </div>`;
}

// ------------------------------------------------------------------- chat ----
function MessageList() {
  const listRef = useRef(null);
  useEffect(() => {
    const el = listRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;   // follow, but
  }, [S.msgTick]);                                              // never hijack
  const onscroll = () => {
    const el = listRef.current;
    stick.current = !el || el.scrollHeight - el.scrollTop - el.clientHeight < 90;
  };
  if (S.mode === 'how') return html`<${HowPane} />`;
  const msgs = S.msgs;
  return html`
    <div id="messages" ref=${listRef} onscroll=${onscroll}>
      ${!msgs.length ? html`<${Hello}/>` : null}
      ${msgs.map((m) => html`<${MessageView} key=${m.id} m=${m}/>`)}
    </div>`;
}

function Hello() {
  const s = S.status;
  return html`
    <div class="hello">
      <h3>Welcome to pedacito</h3>
      <div>A small local model, working on a codebase bigger than its window.</div>
      <ul>
        <li><b>Ask</b> answers questions, read-only.</li>
        <li><b>Task/Edit</b> changes code and stages the diff beside the chat.</li>
        <li><b>Auto</b> plans a goal and works through the checklist.</li>
        <li><b>New</b> starts a project from scratch.</li>
        <li><b>How ?</b> explains CLI commands, GUI paths and settings.</li>
      </ul>
      ${s.needs_setup
        ? html`<button class="primary" onClick=${() => callSettings()}>Set up your model server</button>`
        : 'Open a project above (Browse\u2026 helps).'}
    </div>`;
}

function MessageView(props) {             // pure router: hooks live in the leaves
  const m = props.m;
  if (m.role === 'user') return html`<${UserMsg} m=${m} />`;
  if (m.role === 'assistant') return html`<${AssistantMsg} m=${m} />`;
  if (m.role === 'stream' || m.role === 'activity') return html`<${FoldMsg} m=${m} />`;
  if (m.role === 'question') return html`<div class=${'msg question'}><${QuestionCard} m=${m} /></div>`;
  if (m.role === 'review') return html`<div class=${'msg review'}><${ReviewCard} m=${m} /></div>`;
  if (m.role === 'error') return html`<div class="msg error"><div class="bubble">${m.text}</div></div>`;
  return html`<div class=${'msg ' + m.role + (m.kind ? ' ' + m.kind : '')}><${SystemMsg} m=${m} /></div>`;
}

function UserMsg(props) {
  const m = props.m;
  return html`
    <div class=${'msg user' + (m.kind === 'steer' ? ' steer' : '')}>
      <div class="bubble">
        <div class="tag">${m.kind === 'steer' ? 'note to the model' : (m.mode || 'task')}</div>
        ${m.text}
        ${(m.attachments && m.attachments.length) ? html`
          <div class="chiprow">
            ${m.attachments.map((a, i) => html`<span key=${i} class="fchip">\uD83D\uDCCE ${a.name} (${PDC.fmtInt(a.chars)} chars)</span>`)}
          </div>` : null}
      </div>
    </div>`;
}

function AssistantMsg(props) {
  const m = props.m;
  const parsed = useMemo(() => md(m.text), [m.rev]);
  return html`
    <div class=${'msg assistant' + (m.kind ? ' ' + m.kind : '')}>
      <div class="bubble">
        ${m.head ? html`<div class=${'head ' + (m.level || '')}>${m.head}</div>` : null}
        ${m.text ? html`<div class="md">${parsed}</div>` : null}
        ${(m.results && m.results.length) ? html`
          <ul class="results">
            ${m.results.map((x, i) => html`
              <li key=${i} class=${x.ok ? (/^not written/.test(x.message || '') ? 'skip' : 'ok') : 'no'}>
                ${x.file + ' \u2014 ' + x.message}</li>`)}
          </ul>` : null}
        ${m.kind === 'plan_ready' ? html`
          <div class="row">
            <button class="small" onClick=${() => setView('right')}>Open plan</button>
            <button class="small primary" onClick=${() => PDC.runPlan()}>Implement it now</button>
          </div>` : null}
        ${(m.stats && m.stats.calls) ? html`
          <div class="footer">${m.stats.calls + ' calls \u00B7 ' + PDC.fmtTokens(m.stats.tokens_in) + ' in / ' + PDC.fmtTokens(m.stats.tokens_out) + ' out'
            + (m.stats.cost ? ' \u00B7 $' + Number(m.stats.cost).toFixed(4) : '')
            + (m.stats.seconds ? ' \u00B7 ' + Math.round(m.stats.seconds) + 's' : '')}</div>` : null}
      </div>
    </div>`;
}

function FoldMsg(props) {                 // stream / activity transcripts
  const m = props.m;
  const preRef = useRef(null);
  if (m.uiOpen === undefined) m.uiOpen = jobActive(m.job);
  const open = m.uiOpen;
  useEffect(() => {
    const el = preRef.current;
    if (el && open) el.scrollTop = el.scrollHeight;
  }, [m.rev, open]);
  const summary = m.role === 'stream'
    ? 'Model output \u00B7 ' + (m.site || 'answer')
    : 'Lookups \u00B7 ' + (m.text ? m.text.split('\n').length : 0);
  return html`
    <div class=${'msg ' + m.role}>
      <details class=${m.role} open=${open} ontoggle=${(e) => { m.uiOpen = e.target.open; }}>
        <summary>${summary}</summary>
        <pre ref=${preRef}>${m.text}</pre>
      </details>
    </div>`;
}

function SystemMsg(props) {
  const m = props.m;
  const j = job();
  return html`
    <div>${m.text}</div>
    ${m.kind === 'opened' && m.needs_index && !S.status.has_index ? html`
      <div class="row" style="justify-content:center;margin-top:6px">
        <button class="small primary" disabled=${!!j} onClick=${() => buildIndex(true)}>Build index (with summaries)</button>
        <button class="small" disabled=${!!j} onClick=${() => buildIndex(false)}>Quick index (no summaries)</button>
      </div>` : null}`;
}

function QuestionCard(props) {
  const m = props.m;
  const inpRef = useRef(null);
  useEffect(() => { if (!m.answered && inpRef.current) inpRef.current.focus(); }, [m.answered]);
  const answer = (t) => PDC.api('/api/answer', { qid: m.qid, text: t }).then(refreshState);
  return html`
    <div class=${'card' + (m.answered ? ' done' : '')}>
      <div class="title">The model asks</div>
      <div style="white-space:pre-wrap">${m.text}</div>
      ${m.answered
        ? html`<div class="muted small-text" style="margin-top:4px">
            ${m.answer ? 'Your answer: ' + m.answer : 'No answer \u2014 the model chose.'}</div>`
        : html`<div class="row">
            <input type="text" class="grow" ref=${inpRef} placeholder="your answer\u2026"
                   onkeydown=${(e) => { if (e.key === 'Enter') answer(inpRef.current.value); }} />
            <button class="small primary" onClick=${() => answer(inpRef.current.value)}>Answer</button>
            <button class="small" onClick=${() => answer('')}>Let it choose</button>
          </div>`}
    </div>`;
}

function ReviewCard(props) {
  const m = props.m;
  const pending = !m.outcome;
  const title = pending ? 'Review ready \u2014 ' + m.files + ' file' + (m.files === 1 ? '' : 's')
    : m.outcome === 'applied' ? 'Changes accepted'
    : m.outcome === 'timeout' ? 'Review timed out' : 'Changes rejected';
  return html`
    <div class=${'card ' + (pending ? '' : (m.outcome === 'applied' ? 'applied' : 'rejected'))}>
      <div class="title">${title}</div>
      <div class="muted small-text">${m.detail || m.text}</div>
      ${pending ? html`<div class="row">
        <button class="small primary" onClick=${() => { S.ui.diffOpen = true; setView('diff'); }}>Open review</button>
      </div>` : null}
    </div>`;
}

// --------------------------------------------------------------- composer ----
function Composer() {
  const s = S.status, j = job(), m = MODES[S.mode] || MODES.task;
  const taRef = useRef(null);
  useEffect(() => {                       // swap drafts on mode change (the DOM
    if (taRef.current) {                   // textarea is never re-created, so its
      taRef.current.value = S.drafts[S.mode] || '';
      inputEl = taRef.current;
    }
  }, [S.mode]);
  const needIndex = s.root && !s.has_index && S.mode !== 'new' && S.mode !== 'how';
  const steerable = j && j.kind === 'task' && s.allow_steering;
  const isHow = S.mode === 'how';
  const setMode = (k) => {
    if (taRef.current) S.drafts[S.mode] = taRef.current.value;
    S.mode = k;
    try { localStorage.setItem('pedacito_mode', k); } catch (e) {}
    S.ui.optsOpen = false;
    emit();
  };
  return html`
    <div id="composer"
         ondragenter=${(e) => { e.preventDefault(); e.currentTarget.classList.add('drop'); }}
         ondragover=${(e) => { e.preventDefault(); }}
         ondragleave=${(e) => { e.currentTarget.classList.remove('drop'); }}
         ondrop=${(e) => { e.preventDefault(); e.currentTarget.classList.remove('drop'); if (e.dataTransfer && e.dataTransfer.files.length) attachFiles(e.dataTransfer.files); }}>
      <div id="modeTabs">
        ${MODE_ORDER.map((k) => html`
          <button key=${k} class=${k === S.mode ? 'on' : ''} onClick=${() => setMode(k)}>${MODES[k].label}</button>`)}
      </div>
      <p id="modeHint">
        ${!s.root && S.mode !== 'new' && S.mode !== 'how' ? 'Open a project first. ' : null}
        ${needIndex
          ? html`No index yet \u2014 build one to start:
              <button class="small primary" disabled=${!!j} onClick=${() => buildIndex(true)}>Build index</button>
              <button class="small" disabled=${!!j} onClick=${() => buildIndex(false)}>Quick (no summaries)</button>`
          : m.hint}
      </p>
      <div id="attachRow">
        ${S.attachments.map((a, i) => html`
          <span key=${i} class="attach">\uD83D\uDCCE ${a.name} \u00B7 ${PDC.fmtInt(a.text.length)} chars
            <button title="remove" onClick=${() => { S.attachments.splice(i, 1); emit(); }}>\u2715</button>
          </span>`)}
      </div>
      <textarea id="input" rows="3" spellcheck="true" ref=${taRef}
                placeholder=${isHow ? m.ph : (j ? (steerable ? 'Type a note to steer the model (Enter)\u2026' : 'Working\u2026 press Stop to cancel.') : m.ph)}
                disabled=${!!j && !steerable && !isHow}
                oninput=${(e) => { S.drafts[S.mode] = e.target.value; }}
                onkeydown=${(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } }} />
      <div id="optsBox" key=${S.mode} hidden=${!S.ui.optsOpen}>
        <div class="opts-grid">${OPTS[S.mode].filter((o) => o.type !== 'check').map((o, idx) => html`<${OptField} key=${'g' + idx} o=${o} />`)}</div>
        ${OPTS[S.mode].some((o) => o.type === 'check') ? html`<div class="opts-checks">${OPTS[S.mode].filter((o) => o.type === 'check').map((o, idx) => html`<${OptField} key=${'c' + idx} o=${o} />`)}</div>` : null}
      </div>
      <div id="composerBtns">
        <button id="attachBtn" title="attach a text file (or drop one here)"
                hidden=${isHow}
                disabled=${!!j} onClick=${() => $('filePick').click()}>\uD83D\uDCCE Attach</button>
        <button id="optsBtn" class="ghost" hidden=${!OPTS[S.mode].length}
                onClick=${() => { S.ui.optsOpen = !S.ui.optsOpen; emit(); }}>Options \u25BE</button>
        <span class="grow muted small-text" id="sendHint">${isHow ? 'answers stay out of this chat thread' : (steerable ? 'delivered at the next lookup step' : '')}</span>
        <button id="sendBtn" class=${isHow ? 'primary' : (j ? 'danger' : 'primary')}
                disabled=${!!(j && j.cancelled && !isHow)}
                onClick=${() => { if (isHow) send(); else if (j) stop(); else send(); }}>
          ${isHow ? (S.how.loading ? 'Asking\u2026' : 'Ask')
              : (j ? (j.cancelled ? 'Stopping\u2026' : 'Stop')
              : (S.mode === 'auto' ? (opt('auto', 'plan_only') ? 'Plan only' : 'Apply & run') : m.send))}
        </button>
        <input type="file" id="filePick" multiple hidden
               onchange=${(e) => { attachFiles(e.target.files); e.target.value = ''; }} />
      </div>
    </div>`;
}

function OptField(props) {
  const o = props.o, mode = S.mode;
  const val = opt(mode, o.k);
  const set = useCallback((v) => { S.opts[mode][o.k] = v; saveOpts(); emit(); }, [o.k, mode]);
  let el;
  if (o.type === 'select') {
    el = html`<select value=${val} onchange=${(e) => set(e.target.value)}>
      ${o.opts.map((x) => html`<option key=${x[0]} value=${x[0]}>${x[1]}</option>`)}
    </select>`;
  } else if (o.type === 'check') {
    return html`<label class="chk"><input type="checkbox" checked=${!!val}
                   onchange=${(e) => set(e.target.checked)} />${o.label}</label>`;
  } else if (o.type === 'number') {
    el = html`<input type="number" min="0" defaultValue=${String(val == null ? '' : val)}
              oninput=${(e) => set(e.target.value)} />`;
  } else {
    el = html`<input type="text" defaultValue=${String(val == null ? '' : val)} placeholder=${o.ph || ''}
              spellcheck="false" oninput=${(e) => set(e.target.value)} />`;
    if (o.browse) {
      el = html`<div class="row" style="flex-wrap:nowrap">
        <div class="grow">${el}</div>
        <button class="small" onClick=${() => PDC.browse({
          title: o.browse === 'file' ? 'Choose a plan file' : 'Choose the parent folder',
          files: o.browse === 'file', pickLabel: 'Use this folder', start: S.status.root || '',
          onPick: (p) => {
            if (o.browse === 'dir') {
              const name = (String(S.opts[mode][o.k] || '').split(/[\\/]/).pop()) || 'my-project';
              p = p.replace(/[\\/]$/, '') + '/' + name;
            }
            set(p);
          } })}>Browse\u2026</button>
      </div>`;
    }
  }
  return html`
    <label class=${'f' + (o.span2 ? ' span2' : '')}>${o.label}
      ${el}
      ${o.note && mode === 'task' && (S.opts.task || {}).write === 'review' ? html`<span class="muted">${o.note}</span>` : null}
    </label>`;
}

// ---------------------------------------------------------------- diff pane --
function setDiffMode(m) {
  S.ui.diffMode = m;
  try { localStorage.setItem('pedacito_diff_mode', m); } catch (e) {}
  emit();
}
function markedCells(text, spans) {
  if (!spans || !spans.length) return [text];
  const out = [];
  let pos = 0;
  spans.forEach((s, i) => {
    if (s[0] > pos) out.push(text.slice(pos, s[0]));
    out.push(html`<mark key=${i}>${text.slice(s[0], s[1])}</mark>`);
    pos = s[1];
  });
  if (pos < text.length) out.push(text.slice(pos));
  return out;
}
function codeLine(kind, no, cells, show, key) {
  return html`<div key=${key} class=${'l ' + kind} hidden=${!show}>
    <span class="n">${no == null ? '' : String(no)}</span>
    <span class="t">${cells}</span>
  </div>`;
}
function uniLine(kind, no, cells, show, key) {   // unified row with +/\u2212 mark
  const pm = kind === 'add' ? '+' : (kind === 'del' ? '\u2212' : '\u00A0');
  return html`<div key=${key} class=${'l u ' + kind} hidden=${!show}>
    <span class="n">${no == null ? '' : String(no)}</span>
    <span class="pm">${pm}</span>
    <span class="t">${cells}</span>
  </div>`;
}

function DiffPane() {
  const R = S.review, D = S.diff;
  const unified = S.ui.diffMode === 'unified';
  const oldRef = useRef(null), newRef = useRef(null);
  useEffect(() => {                       // keep the two panes in step
    const a = oldRef.current, b = newRef.current;
    if (!a || !b) return;
    const sa = () => { if (b.scrollTop !== a.scrollTop) b.scrollTop = a.scrollTop; };
    const sb = () => { if (a.scrollTop !== b.scrollTop) a.scrollTop = b.scrollTop; };
    a.addEventListener('scroll', sa); b.addEventListener('scroll', sb);
    return () => { a.removeEventListener('scroll', sa); b.removeEventListener('scroll', sb); };
  }, [D, S.diffRel, S.editing]);
  useEffect(() => {                       // fill the editor when entering it
    if (S.editing && editBox.current) editBox.current.value = (S.diff && S.diff.after) || '';
  }, [S.editing, D]);

  if (!R) {
    return html`
      <div class="placeholder">
        <b>No proposed changes yet.</b><br />
        Run a <b>Task/Edit</b> in <i>Review</i> mode: the model's edits appear
        here, old code beside updated code, for you to accept or reject.
      </div>`;
  }
  const pending = R.pending;
  const editable = R.files.filter((f) => !f.source);
  const nSel = R.partial_ok ? editable.filter((f) => S.sel[f.rel] !== false).length : editable.length;
  const rows = (D && D.rows) || [];
  const oldRows = [], newRows = [], uniRows = [];
  rows.forEach((r, i) => {
    const key = 'r' + i;
    if (r[0] === 'skip') {
      const foldKey = S.diffRel + '#' + r[2];
      const opened = S.ui.folds[foldKey] === true;
      const label = '\u22EF ' + r[1] + ' unchanged line' + (r[1] === 1 ? '' : 's') + ' \u2014 click to show';
      const mk = () => html`<div key=${key} class="skip" hidden=${opened}
        onClick=${() => { S.ui.folds[foldKey] = true; emit(); }}>${label}</div>`;
      oldRows.push(mk()); newRows.push(mk()); uniRows.push(mk());
      return;
    }
    const kind = r[0], show = !r[7] || S.ui.folds[S.diffRel + '#' + r[7]] === true;
    if (kind === 'equal') {
      oldRows.push(codeLine('equal', r[1], r[2], show, key));
      newRows.push(codeLine('equal', r[3], r[4], show, key));
      uniRows.push(uniLine('equal', r[1], r[2], show, key));
    } else if (kind === 'replace') {
      oldRows.push(codeLine('replace del', r[1], markedCells(r[2], r[5]), show, key));
      newRows.push(codeLine('replace add', r[3], markedCells(r[4], r[6]), show, key));
      uniRows.push(uniLine('del', r[1], markedCells(r[2], r[5]), show, key + 'a'));
      uniRows.push(uniLine('add', r[3], markedCells(r[4], r[6]), show, key + 'b'));
    } else if (kind === 'delete') {
      oldRows.push(codeLine('delete', r[1], r[2], show, key));
      newRows.push(codeLine('blank', null, '', show, key));
      uniRows.push(uniLine('del', r[1], r[2], show, key));
    } else {
      oldRows.push(codeLine('blank', null, '', show, key));
      newRows.push(codeLine('insert', r[3], r[4], show, key));
      uniRows.push(uniLine('add', r[3], r[4], show, key));
    }
  });
  const acceptLabel = !R.partial_ok || nSel === editable.length
    ? (editable.length > 1 ? 'Accept all' : 'Accept')
    : 'Accept ' + nSel + ' of ' + editable.length;
  return html`
    <div class="diffwrap">
      <div id="reviewBar">
        <b id="reviewTitle" title=${R.task || ''}>${R.task || 'Proposed changes'}</b>
        ${!pending ? html`<span class=${'badge-out ' + R.outcome}>
            ${R.outcome === 'applied' ? 'accepted' : R.outcome === 'cancelled' ? 'rejected' : R.outcome}</span>` : null}
        <div id="reviewActions">
          <button class="ghost small" title="collapse the diff (Chat stays full-width)"
                  onClick=${() => { S.ui.diffOpen = false; emit(); }}>\u29C9 Collapse</button>
          <span id="diffViewToggle" title="diff layout: two panes or one unified column">
            <button class=${'small' + (unified ? '' : ' on')} onClick=${() => setDiffMode('split')}>Side by side</button>
            <button class=${'small' + (unified ? ' on' : '')} onClick=${() => setDiffMode('unified')}>Unified</button>
          </span>
          ${pending ? html`
            <button class="small" id="editBtn" hidden=${!R.partial_ok}
                    title="edit the proposed code for the selected file by hand before accepting"
                    onClick=${toggleEdit}>${S.editing ? '\u2716 Show diff' : '\u270E Edit proposed'}</button>
            <button class="small danger" onClick=${() => decide('reject')}>Reject${R.files.length > 1 ? ' all' : ''}</button>
            <button class="small primary" disabled=${nSel === 0}
                    onClick=${() => decide('apply')}>${acceptLabel}</button>` : null}
        </div>
      </div>
      <div id="fileStrip">
        ${R.files.map((f) => html`
          <div key=${f.rel} class=${'fileitem' + (f.rel === S.diffRel ? ' on' : '') + (f.source ? ' source' : '')}
               title=${f.message || ''} onClick=${() => { S.editing = false; loadDiffFile(f.rel); }}>
            ${pending && R.partial_ok && !f.source ? html`
              <input type="checkbox" checked=${S.sel[f.rel] !== false} title="include in Accept"
                     onclick=${(e) => e.stopPropagation()}
                     onchange=${(e) => { S.sel[f.rel] = e.target.checked; emit(); }} />` : null}
            <span>${f.rel}</span>
            ${f.edited ? html`<span class="muted">(edited)</span>` : null}
            <span class="a">+${f.added}</span><span class="r">\u2212${f.removed}</span>
            ${pending && !f.source && R.partial_ok ? html`
              <button class="fedit" title="edit this file's proposed code"
                      onClick=${(e) => {
                        e.stopPropagation();
                        if (f.rel === S.diffRel && S.diff) { toggleEdit(); }
                        else { S.editAfterLoad = true; loadDiffFile(f.rel); }
                      }}>\u270E</button>` : null}
          </div>`)}
      </div>
      ${!D ? html`<div class="tip">${R.files.length ? 'Loading\u2026' : 'No files in this review.'}</div>` : (unified ? html`
        <div id="panes" class="unified">
          <div class="pane">
            <div class="pane-head"><span>${S.diffRel}</span><span>\u2212${D.removed} / +${D.added} lines</span></div>
            <div class="code" hidden=${S.editing}>${uniRows}</div>
            <textarea id="editBox" spellcheck="false" hidden=${!S.editing}
                      ref=${(el) => { editBox.current = el; }}></textarea>
          </div>
        </div>` : html`
        <div id="panes">
          <div class="pane">
            <div class="pane-head"><span>Old code</span><span>\u2212${D.removed} lines</span></div>
            <div class="code" ref=${oldRef}>${oldRows}</div>
          </div>
          <div class="pane">
            <div class="pane-head"><span>Updated code</span><span>+${D.added} lines</span></div>
            <div class="code" ref=${newRef} hidden=${S.editing}>${newRows}</div>
            <textarea id="editBox" spellcheck="false" hidden=${!S.editing}
                      ref=${(el) => { editBox.current = el; }}></textarea>
          </div>
        </div>`)}
    </div>`;
}

// ------------------------------------------------- right pane: ACTIVITY & PLAN -
// The right column is one ACTIVITY & PLAN panel: this pane carries the Engine
// V2 telemetry (`status.engine` -- live phase, counters, the requirement
// ledger, recent gather/microloop actions) and the Plan checklist pane sits
// right below it. Without engine data the classic activity-steps feed renders
// exactly as before, so Auto-mode checklists and transcripts are unchanged.
function actParts(a) {                 // "app.py L12-40" -> file + line badges
  const d = String(a.detail || '');
  const m = /^(.+?)\s+L(\d+)\s*-\s*(\d+)(.*)$/.exec(d);
  if (m) {
    return [html`<span key="f" class="tx">${m[1]}</span>`,
            html`<span key="l" class="aline">L${m[2]}-${m[3]}</span>`,
            m[4].trim() ? html`<span key="r" class="tx2">${m[4].trim()}</span>` : null];
  }
  return [html`<span key="f" class="tx">${d}</span>`];
}

function ActivityPane() {
  const s = S.status, j = s.job, eng = s.engine || {};
  const lastActivity = useMemo(() => {
    for (let i = S.msgs.length - 1; i >= 0; i--) {
      const m = S.msgs[i];
      if (m.role === 'activity' && (!j || m.job === j.id)) return m;
    }
    for (let i = S.msgs.length - 1; i >= 0; i--) {
      if (S.msgs[i].role === 'activity') return S.msgs[i];
    }
    return null;
  }, [S.msgTick, j && j.id]);
  const lastStream = useMemo(() => {
    for (let i = S.msgs.length - 1; i >= 0; i--) {
      const m = S.msgs[i];
      if (m.role === 'stream' && j && m.job === j.id) return m;
    }
    return null;
  }, [S.msgTick, j && j.id]);
  const lines = (lastActivity && lastActivity.text ? lastActivity.text.split('\n') : []);
  const tail = lines.slice(-120);
  const acts = (eng.actions || []).slice(-12).reverse();
  const reqs = eng.requirements || [];
  const phase = eng.phase || (j ? 'gather' : 'idle');
  return html`
    <section class="panel" id="activityPanel">
      <div class="panel-head">
        <h2>Activity</h2>
        <span class="chip">
          <i class=${'dot ' + (j ? 'busy' : 'idle')}></i><b>${phase}</b></span>
        <span class="muted small-text grow" id="engCount">
          ${PDC.fmtInt(eng.calls || 0)} calls \u00B7 ${PDC.fmtInt(eng.steps || 0)} steps</span>
        ${j ? html`<button class="ghost small" onClick=${() => stop()}>Stop</button>` : null}
      </div>
      <div id="activityBody">
        <div class="eng-bar" title="Engine V2 state (during and after runs)">
          <span class=${'phase ' + phase}>${phase}</span>
          <span class="estat" title="model calls this session">calls
            <b>${PDC.fmtInt(eng.calls || 0)}</b></span>
          <span class="estat" title="gather steps in the last run">step
            <b>${PDC.fmtInt(eng.steps || 0)}</b></span>
          <span class="estat" title="read spans recorded in the ledger">read spans
            <b>${PDC.fmtInt(eng.read_spans || 0)}</b></span>
        </div>
        ${reqs.length ? html`
          <div class="eng-sec"><b>Requirement ledger</b><ol class="reqs">
            ${reqs.map((rq, i) => html`
              <li key=${i} class="reqrow" title=${rq.text}>
                <span class=${'rq st-' + String(rq.state || 'UNLOCATED').toLowerCase()}>${rq.state}</span>
                <span class="tx">${rq.text}</span>
                <span class="muted small-text">${rq.reads || 0}r${rq.edits ? ' \u00B7 ' + rq.edits + 'e' : ''}${rq.anchors ? ' \u00B7 ' + rq.anchors + 'a' : ''}</span>
              </li>`)}
          </ol></div>` : null}
        ${acts.length ? html`
          <div class="eng-sec"><b>Recent actions</b><ul class="acts">
            ${acts.map((a, i) => html`
              <li key=${i} class=${'actrow' + (a.ok === false ? ' bad' : '')}>
                <span class=${'atool k-' + String(a.tool || 'step').toLowerCase().replace(/[^a-z0-9-]+/g, '')}>${a.tool}</span>
                ${actParts(a)}</li>`)}
          </ul></div>` : null}
        ${tail.length ? html`<ol class="steps" hidden=${!!acts.length}>
            ${tail.map((l, i) => html`<li key=${'l' + (lines.length - tail.length + i)}>
              <span class="tx">${l}</span></li>`)}
          </ol>` : (acts.length ? null : html`<div class="tip">No lookups yet \u2014 run something.</div>`)}
        ${lastStream && lastStream.text ? html`
          <div class="streamtail" title="live model output"><b>${lastStream.site || 'answer'}</b>
          <pre>${lastStream.text.slice(-2000)}</pre></div>` : null}
      </div>
    </section>`;
}

function PlanPane() {
  const p = S.plan;
  const tasks = p.tasks || [];
  const done = tasks.filter((t) => t.status === 'done').length;
  const openN = tasks.filter((t) => t.status === 'todo').length;
  const pct = tasks.length ? Math.round(100 * done / tasks.length) : 0;
  const running = !!job();
  const icons = { todo: '\u25CB', done: '\u2713', failed: '\u2717', skipped: '\u2013' };
  useEffect(() => {
    if (S.ui.planEditing && planEdit.current) planEdit.current.value = p.text || '';
  }, [S.ui.planEditing]);
  let firstTodo = true;
  return html`
    <section class="panel" id="planPanel">
      <div class="panel-head">
        <h2>Plan</h2>
        ${tasks.length ? html`<span class="badge">${done}/${tasks.length}</span>` : null}
        <span class="muted small-text grow mono" id="planPath">${p.exists ? p.path : ''}</span>
        ${p.exists ? html`
          <button class="small" onClick=${() => { S.ui.planEditing = !S.ui.planEditing; emit(); }}>
            ${S.ui.planEditing ? 'Cancel' : 'Edit'}</button>
          ${S.ui.planEditing ? html`<button class="small primary" onClick=${() => savePlan()}>Save</button>` : null}
          <button class="small primary" disabled=${running || !openN} title="run this plan now"
                  onClick=${() => PDC.runPlan()}>Run</button>` : null}
      </div>
      <div id="planBody" hidden=${S.ui.planEditing}>
        ${!p.exists ? html`
          <div class="tip">No plan yet. Use <b>Auto</b> to turn a goal into a
          checklist; its progress shows here as it runs.</div>` : html`
          <div class="muted small-text">${done} of ${tasks.length} tasks done</div>
          <div class="progress"><div style=${'width:' + pct + '%'}></div></div>
          ${tasks.map((t, i) => {
            const cur = t.status === 'todo' && firstTodo && running;
            if (t.status === 'todo') firstTodo = false;
            return html`<div key=${i} class=${'ptask ' + t.status + (cur ? ' current' : '')}>
              <span class="st">${cur ? '\u25B6' : (icons[t.status] || '\u25CB')}</span>
              <span>${t.title}</span></div>`;
          })}`}
      </div>
      <textarea id="planEdit" spellcheck="false" hidden=${!S.ui.planEditing}
                ref=${(el) => { planEdit.current = el; }}></textarea>
    </section>`;
}

// ------------------------------------------------------- workspace pane ----
// The bottom-right panel: three tabs replacing the old static TERMINAL box.
// FILES browses the project (index + workspace scan), BACKUPS restores a
// snapshot with one click, LOGS keeps the raw terminal output. The #term
// element stays mounted in every tab so the live Term buffer never resets.
const WS_TABS = [['files', 'Files'], ['backups', 'Backups'], ['logs', 'Logs']];

function WorkspacePane() {
  const s = S.status, j = s.job;
  const termRef = useRef(null);
  useEffect(() => {
    if (termRef.current && !term) term = new Term(termRef.current);
  }, []);
  const tab = S.ws.tab;
  const idxSet = S.ws.indexedSet || new Set();
  const flt = S.ws.filter.trim().toLowerCase();
  const all = S.ws.files || [];
  const shown = (flt ? all.filter((f) => f.toLowerCase().indexOf(flt) >= 0) : all).slice(0, 600);
  const fileRow = (f) => {
    const i = f.lastIndexOf('/');
    const base = i < 0 ? f : f.slice(i + 1);
    const dir = i < 0 ? '' : f.slice(0, i);
    return html`
      <div key=${f} class="wsrow" title=${f}>
        <button class="ws-open" title="preview"
                onClick=${() => previewFile(f)}>
          <span class="ws-base">${base}</span>${dir ? html`<span class="ws-dir">${dir}/</span>` : null}
        </button>
        ${idxSet.has(f) ? html`<span class="tag idx" title="in the index">indexed</span>` : null}
        <span class="grow"></span>
        <button class="ws-at" title="insert @${f} into the chat input"
                onClick=${() => insertRef(f)}>@</button>
      </div>`;
  };
  const snap = S.ws.backups[0] || (s.snapshots || [])[0] || null;
  const snapName = snap ? (snap.name || snap) : '';
  return html`
    <section class=${'panel' + (S.ui.wsCollapsed ? ' collapsed' : '')} id="wsPanel">
      <div class="panel-head">
        <div id="wsTabs">
          ${WS_TABS.map(([k, lbl]) => html`
            <button key=${k} class=${tab === k ? 'on' : ''}
                    onClick=${() => setWsTab(k)}>${lbl}</button>`)}
        </div>
        <span class="muted small-text grow ws-sub">
          ${tab === 'files' ? (s.root ? PDC.fmtInt(S.ws.indexed.length) + ' of ' + PDC.fmtInt(all.length) + ' files indexed' : 'no project open')
          : tab === 'backups' ? 'undo a Task/Edit or Auto write'
          : 'live output'}
        </span>
        ${tab === 'logs' ? html`
          <label class="chk"><input type="checkbox" checked=${S.ui.termFollow}
                 onchange=${(e) => { S.ui.termFollow = e.target.checked; if (term) { term.follow = e.target.checked; if (e.target.checked) term.schedule(); } }} />follow</label>
          <button class="ghost small" onClick=${() => clearTerm()}>Clear</button>` : null}
        <button class="ghost small" title="collapse / expand"
                onClick=${() => { S.ui.wsCollapsed = !S.ui.wsCollapsed; emit(); }}>${S.ui.wsCollapsed ? '\u25B2' : '\u25BC'}</button>
      </div>
      <div class="ws-body" id="wsFiles" hidden=${tab !== 'files'}>
        <div class="ws-tools">
          <input type="text" class="grow" placeholder="filter files\u2026"
                 spellcheck="false" value=${S.ws.filter}
                 oninput=${(e) => { S.ws.filter = e.target.value; emit(); }} />
          <button class="small" disabled=${S.ws.loading || !s.root} title="rescan the project folder"
                  onClick=${() => loadFiles(true)}>\u27F3</button>
        </div>
        ${!s.root ? html`<div class="tip">Open a project to browse its files.</div>`
        : S.ws.loading && !all.length ? html`<div class="tip">Scanning the workspace\u2026</div>`
        : !shown.length ? html`<div class="tip">${all.length ? 'No file matches the filter.' : 'No files found.'}</div>`
        : html`<div class="ws-list">${shown.map(fileRow)}</div>`}
        <div class="muted small-text ws-foot" hidden=${!all.length}>
          ${'Showing ' + PDC.fmtInt(shown.length) + ' of ' + PDC.fmtInt(all.length) + ' files'
            + (S.ws.truncated ? ' \u00B7 scan capped at ' + PDC.fmtInt(all.length) : '')}
        </div>
      </div>
      <div class="ws-body" id="wsBackups" hidden=${tab !== 'backups'}>
        <div class="ws-tools">
          <button class="small danger" disabled=${!!j || !snapName}
                  title=${snapName ? 'restore ' + snapName : 'no snapshot yet'}
                  onClick=${() => restoreBackup(snapName)}>\u21BA Restore last backup</button>
          <button class="small" onClick=${() => loadBackups(true)} title="relist snapshots">\u27F3</button>
          <span class="muted small-text grow">${'backups taken before writes (' + PDC.fmtInt(S.ws.backups.length) + ')'}</span>
        </div>
        ${!S.ws.backups.length ? html`<div class="tip">No snapshots yet \u2014 direct writes and reviewed accepts back up the originals first.</div>`
        : html`<div class="ws-list">${S.ws.backups.map((b, i) => html`
          <div key=${b.name} class="wsrow" title=${b.name}>
            <span class="ws-base mono">${b.name}</span>
            ${i === 0 ? html`<span class="tag idx" title="newest snapshot">newest</span>` : null}
            <span class="muted small-text">${PDC.fmtInt(b.files)} file(s) \u00B7 ${fmtTime(b.time)}</span>
            <span class="grow"></span>
            <button class="small" disabled=${!!j} title="copy this snapshot back over the working tree"
                    onClick=${() => restoreBackup(b.name)}>Restore</button>
          </div>`)}</div>`}
      </div>
      <div class="ws-body" id="wsLogs" hidden=${tab !== 'logs'}>
        <div id="term" ref=${termRef}
             onwheel=${() => { const el = termRef.current;
               if (term && term.follow && el && el.scrollTop + el.clientHeight < el.scrollHeight - 40) {
                 term.follow = false; S.ui.termFollow = false; emit(); } }}></div>
      </div>
    </section>`;
}

// --------------------------------------------------------------------- app ---
function diffMode() {
  if (!S.review) return 'none';
  return S.ui.diffOpen ? 'expanded' : 'collapsed';
}

function App() {
  const [, force] = useState(0);
  const appRef = useRef(null);
  const rightRef = useRef(null);
  useEffect(() => {
    const fn = () => force((x) => x + 1);
    subs.add(fn);
    return () => { subs.delete(fn); };
  }, []);
  const s = S.status;
  const dm = diffMode();
  return html`
    <div id="shell">
      <${TopBar} />
      <main id="app" ref=${appRef} data-diff=${dm}>
        <section class="panel" id="chatPanel">
          <div class="panel-head">
            <h2>${S.mode === 'how' ? 'How / Help' : 'Chat'}</h2>
            <span class="muted small-text grow" id="busyLbl">
              ${s.job ? (s.job.cancelled ? 'stopping\u2026' : s.job.title + '\u2026') : ''}</span>
            ${S.mode === 'how' ? html`<span class="muted small-text">CLI commands \u00b7 GUI paths \u00b7 settings</span>` : null}
            ${dm === 'collapsed' ? html`
              <button class="small primary" id="diffPill" title="show the staged diff beside the chat"
                      onClick=${() => { S.ui.diffOpen = true; emit(); }}>
                \u29C6 ${s.review.pending ? 'Review staged' : 'Review'}
                ${(s.review.files || []).length ? ' \u00B7 ' + s.review.files.filter((f) => !f.source).length + ' file(s)' : ''} \u2014 expand</button>` : null}
            <select id="threadSel" title="this project's saved conversations"
                    value=${s.active_thread_id || ''} disabled=${!!s.job}
                    onchange=${(e) => selectThread(e.target.value)}>
              ${(s.threads || []).map((t) => html`
                <option key=${t.id} value=${t.id}>${t.title} \u00b7 ${t.message_count} msg${t.message_count === 1 ? '' : 's'}</option>`)}
            </select>
            <button class="ghost small" id="newThread" title="start a new conversation (this one stays saved in the project)"
                    disabled=${!!s.job} onClick=${newThread}>+ New</button>
            <button class="ghost small danger" id="delThread" title="delete this conversation and its saved thread"
                    disabled=${!!s.job} onClick=${deleteCurrentThread}>Delete</button>
            <button class="ghost small" id="clearChat" title="clear the conversation"
                    onClick=${() => PDC.api('/api/clear', {})}>Clear</button>
          </div>
          <${MessageList} />
          <${Composer} />
        </section>
        <div class="splitter" title="drag to resize" onMousedown=${dragChat(appRef)}></div>
        <section class="panel" id="diffPanel">
          <${DiffPane} />
        </section>
        <div class="splitter" title="drag to resize" onMousedown=${dragRight(appRef)}></div>
        <aside class="panelcol" id="rightPane" ref=${rightRef}>
          <${ActivityPane} />
          <${PlanPane} />
          <div class="splitter h" id="split2" title="drag to resize" onMousedown=${dragTerm(rightRef)}></div>
          <${WorkspacePane} />
        </aside>
      </main>
      <nav id="mobileTabs">
        <button data-view="chat" class=${S.ui.view === 'chat' ? 'on' : ''} onClick=${() => setView('chat')}>Chat</button>
        <button data-view="diff" class=${S.ui.view === 'diff' ? 'on' : ''}
                disabled=${dm === 'none'} onClick=${() => { S.ui.diffOpen = true; setView('diff'); }}>Changes</button>
        <button data-view="right" class=${S.ui.view === 'right' ? 'on' : ''} onClick=${() => setView('right')}>Activity</button>
      </nav>
    </div>`;
}

// --------------------------------------------------------------- settings ----
// settings.js is the legacy (non-module-style) editor UI; load it once, after
// PDC exists, so its Settings dialog stays available without a rewrite.
let settingsReady = false;
const settingsPromise = import('./settings.js').then(() => { settingsReady = true; }, () => {});
function callSettings(opts) {
  if (settingsReady) { if (PDC.openSettings) PDC.openSettings(opts); }
  else settingsPromise.then(() => { if (PDC.openSettings) PDC.openSettings(opts); });
}

// ------------------------------------------------------------------- boot ----
function initThemeSelect() {
  const sel = $('theme');
  if (sel) {
    let t = localStorage.getItem('pedacito_theme') || 'slate';
    // the Sprint-E legacy names (mocha/sage/parchment/clay/nightshade) no
    // longer exist: an unknown saved value falls back to the slate default
    if (!Array.prototype.some.call(sel.options, (o) => o.value === t)) t = 'slate';
    sel.value = t;
  }
  if (S.ui.view) document.body.dataset.view = S.ui.view;
}

render(html`<${App}/>`, $('root'));
initThemeSelect();
refreshState().then(() => loadSnapshot());
setInterval(() => { if (!document.hidden) refreshState(); }, 2500);
