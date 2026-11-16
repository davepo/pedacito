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

// Built-in hosted-provider catalog. settings.js seeds its "Add Hosted
// Provider" dialog from this immediately, so the provider selector renders
// even before (or if) GET /api/providers/presets ever answers; the server's
// reply replaces it when it arrives. Mirrors settings_api.CLOUD_PROVIDERS
// exactly -- if a preset is added or renamed there, change it here too.
// The dialog probes the chosen provider live (POST /api/providers/probe) and
// imports each ticked model as a profile NAMED AFTER THE MODEL, not the
// vendor (POST /api/providers/import).
PDC.FALLBACK_PROVIDER_CATALOG = [
  { id: 'anthropic', label: 'Claude (Anthropic)', provider: 'anthropic',
    base_url: 'https://api.anthropic.com/v1', api_key_env: 'ANTHROPIC_API_KEY',
    suggest: 'claude', context_tokens: 200000,
    models: ['claude-sonnet-4-5', 'claude-haiku-4-5', 'claude-opus-4-5'] },
  { id: 'openai', label: 'ChatGPT (OpenAI)', provider: 'openai',
    base_url: 'https://api.openai.com/v1', api_key_env: 'OPENAI_API_KEY',
    suggest: 'openai', context_tokens: 272000,
    models: ['gpt-5.4', 'gpt-5.4-mini', 'o3-mini', 'gpt-4.1', 'gpt-4.1-mini'] },
  { id: 'deepseek', label: 'DeepSeek', provider: 'openai',
    base_url: 'https://api.deepseek.com/v1', api_key_env: 'DEEPSEEK_API_KEY',
    suggest: 'deepseek', context_tokens: 128000,
    models: ['deepseek-chat', 'deepseek-reasoner'] },
  { id: 'qwen', label: 'Qwen (DashScope)', provider: 'openai',
    base_url: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    api_key_env: 'DASHSCOPE_API_KEY', suggest: 'qwen', context_tokens: 256000,
    models: ['qwen3-coder-plus', 'qwen-max', 'qwen-plus', 'qwen-flash'] },
  { id: 'gemini', label: 'Google Gemini', provider: 'openai',
    base_url: 'https://generativelanguage.googleapis.com/v1beta/openai',
    api_key_env: 'GEMINI_API_KEY', suggest: 'gemini', context_tokens: 200000,
    models: ['gemini-2.5-pro', 'gemini-2.5-flash'] },
  { id: 'openrouter', label: 'OpenRouter', provider: 'openai',
    base_url: 'https://openrouter.ai/api/v1', api_key_env: 'OPENROUTER_API_KEY',
    suggest: 'openrouter', context_tokens: 200000,
    models: ['deepseek/deepseek-chat', 'qwen/qwen3-coder', 'anthropic/claude-3.7-sonnet'] },
];

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

// ------------------------------------------------ live permission broker ---
// A gated action (web fetch, pip install, unapproved shell/test command,
// sensitive file read) parks its worker thread server-side and surfaces here
// as `status.pending_permission`. The modal is the only way forward: one of
// the four buttons POSTs /api/permission/decide and the server releases the
// waiting thread (Allow Once / Allow for This Run / Always Allow /
// Deny & Redirect).
const PERM_LABELS = {
  web: 'Web Access', package_install: 'Package Install',
  shell: 'Shell Command', sensitive_file: 'Sensitive File Read',
};
let permModal = null, permShownId = '';

PDC.decidePermission = (id, decision) => {
  if (permModal) { const m = permModal; permModal = null; permShownId = ''; m.close(); }
  PDC.api('/api/permission/decide', { id: id, decision: decision }).then(() => refreshState());
};

function syncPermissionModal(s) {
  const pp = s.pending_permission || null;
  if (pp && pp.id !== permShownId) {          // a new request: open the modal
    permShownId = pp.id;
    showPermissionModal(pp);
  } else if (!pp && permShownId) {            // answered/timed out: close it
    permShownId = '';
    if (permModal) { const m = permModal; permModal = null; m.close(); }
  }
}

function showPermissionModal(pp) {
  const label = PERM_LABELS[pp.category] || pp.category;
  const body = h('div', { class: 'perm' },
    h('div', { class: 'perm-badge pb-' + pp.category, text: label }),
    h('p', { class: 'perm-sum', text: pp.action_summary || 'The model requests this action.' }),
    pp.reason ? h('p', { class: 'perm-why' }, h('b', { text: 'Why: ' }), pp.reason) : null,
    h('pre', { class: 'perm-detail', text: pp.detail || '(no target shown)' }),
    h('p', { class: 'perm-risk', text: 'Risk: ' + (pp.risk_level || 'medium') +
      ' \u00B7 the run is paused until you decide. Deny & Redirect tells the ' +
      'model to proceed with local project files only.' }));
  const btn = (label, decision, extra) => Object.assign(
    { label: label, stay: true,
      onclick: () => { PDC.decidePermission(pp.id, decision); return false; } },
    extra || {});
  permModal = PDC.modal('Permission requested', body, [
    btn('Allow Once', 'allow_once', { primary: true }),
    btn('Allow for This Run', 'allow_run', {}),
    btn('Always Allow', 'allow_always', {}),
    btn('Deny & Redirect', 'deny', { danger: true }),
  ], { narrow: true, onClose: () => { permModal = null; permShownId = ''; } });
}

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
// The composer's draft survives a browser refresh. `pedacito_draft_prompt` is
// written on every input and cleared only when the user submits (or empties)
// the box -- F5 must never cost someone the prompt they just typed.
const DRAFT_KEY = 'pedacito_draft_prompt';
function readDraft() {
  try { return localStorage.getItem(DRAFT_KEY) || ''; } catch (e) { return ''; }
}
function writeDraft(text) {
  try {
    if (text) localStorage.setItem(DRAFT_KEY, text);
    else localStorage.removeItem(DRAFT_KEY);
  } catch (e) {}
}
function clearDraft() { writeDraft(''); }
const savedDraft = readDraft();
const savedMode = (() => {
  try { return localStorage.getItem('pedacito_mode') || 'task'; }
  catch (e) { return 'task'; }
})();

const S = {
  status: {},
  msgs: [], msgTick: 0,                   // msgTick bumps on any message change
  plan: { path: '', exists: false, text: '', tasks: [] },
  review: null, diff: null, diffRel: '', sel: {}, editing: false,
  editAfterLoad: false,
  ws: { tab: localStorage.getItem('pedacito_ws_tab') || 'files', files: [],
        indexed: [], indexedSet: null, sig: '', filter: '', truncated: false,
        loading: false, backups: [], backupsAt: 0 },
  mode: savedMode,
  drafts: savedDraft ? { [savedMode]: savedDraft } : {}, attachments: [],
  opts: (() => { try { return JSON.parse(localStorage.getItem('pedacito_opts') || '{}'); } catch (e) { return {}; } })(),
  ui: { diffOpen: false, wsCollapsed: false, termFollow: true, optsOpen: false,
        diffMode: localStorage.getItem('pedacito_diff_mode') || 'split',
        view: 'chat', folds: {}, planEditing: false, pendingProfile: null,
        refining: false, previousPrompt: null, switchingProfile: false,
        actionBusy: false, planGenerating: false, planGenFrom: null },
  peek: null,
  how: { query: '', answer: null, loading: false, error: '', useModel: true },
};
PDC.S = S;

let subs = new Set();
function emit() {                         // Preact batches these per frame
  subs.forEach((f) => f());
}

const job = () => S.status.job || null;

// Local action state only: a profile switch or another quick header action
// must never look like a running task. `setBusy(false)` is guaranteed in the
// caller's `finally`, so a failed request cannot leave the Run button stuck
// as Stop.
function setBusy(on) {
  S.ui.actionBusy = !!on;
  if (!on) S.ui.switchingProfile = false;
  emit();
}
PDC.setBusy = setBusy;

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
    { k: 'build', label: 'What to do', type: 'select', def: 'build', opts: [['build', 'Build it'], ['preview', 'Preview the plan (writes nothing)'], ['review', 'Build, then review'], ['plan', 'Header skeleton only']], span2: true },
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
  syncPermissionModal(s);               // live permission requests open the modal
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
  // The plan file is written/ticks WHILE Auto or a planned Task runs. Use the
  // live payload carried by /api/state every poll (and fall back to /api/plan
  // for an older server), so PLAN.md appears as soon as the planner writes it
  // and each [x] lands immediately -- not when the whole run finishes.
  if (s.plan) applyPlanPayload(s.plan);
  else if (j && (j.kind === 'auto' || j.kind === 'task')) loadPlan();
  if (j && j.id !== lastJobId) loadPlan();
  if (!j && lastJobId) { collapseDone(); loadPlan(); loadReview(); loadFiles(); loadBackups(); }
  if (!j && S.ui.planGenerating && !S.ui.planGenFrom) {
    S.ui.planGenerating = false;
    S.ui.planGenFrom = null;
  }
  if (!j && S.ui.planGenerating && lastJobId) {
    // The planner/run ended without writing a replacement: stop claiming a
    // new plan is coming and show whatever is on disk.
    S.ui.planGenerating = false;
    S.ui.planGenFrom = null;
  }
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
    syncPermissionModal(S.status);
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
  on('permission', () => refreshState());         // a gated action is waiting
  on('permission_decided', () => refreshState()); // someone answered it
  // PLAN.md was created or a box was ticked: refresh the checklist now.
  on('plan', () => loadPlan());
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
async function switchProfile(name) {
  S.ui.pendingProfile = name;
  S.ui.switchingProfile = true;
  emit();
  try {
    const j = await PDC.api('/api/profile', { profile: name });
    if (!j.ok) {
      S.ui.pendingProfile = null;
      PDC.toast(j.detail || 'could not switch profile', true);
    }
  } catch (e) {
    S.ui.pendingProfile = null;
    PDC.toast('Could not switch profile.', true);
  } finally {
    // /api/profile is synchronous and never claims the job slot. Make that
    // visible here too: no loading state may survive a profile switch.
    setBusy(false);
    S.ui.switchingProfile = false;
    await refreshState();
    emit();
  }
}
function startNewProject() {
  S.mode = 'new';
  if (!opt('new', 'project')) {
    const s = S.status;
    S.opts.new.project = s.suggested_new_project || '';
  }
  try { localStorage.setItem('pedacito_mode', 'new'); } catch (e) {}
  S.ui.optsOpen = false;
  emit();
  setTimeout(() => { const el = $('input'); if (el) el.focus(); }, 30);
}
function buildIndex(summarise) {
  PDC.api('/api/index', { summarise: summarise }).then((j) => {
    if (!j.ok) PDC.toast(j.detail || 'busy', true);
    refreshState();
  });
}
function setAutoRoute(on) {           // topbar toggle -> user config + reload
  PDC.api('/api/settings/set', { file: 'user', changes: { auto_route_specialties: !!on } })
    .then((j) => {
      if (!j.ok) { PDC.toast(j.detail || 'could not change auto-route', true); return; }
      PDC.toast(on ? 'Specialist auto-routing on: tasks switch to the profile whose specialty fits.'
                   : 'Specialist auto-routing off: every task stays on the active profile.');
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
function renameThread() {
  const s = S.status;
  if (!s.active_thread_id) { PDC.toast('There is no open conversation to rename.', true); return; }
  const inp = h('input', { type: 'text', class: 'grow', value: s.thread_title || '',
                            maxlength: '120', spellcheck: false, autocomplete: 'off',
                            placeholder: 'a name for this conversation' });
  const go = () => {
    const t = (inp.value || '').trim();
    if (!t) { PDC.toast('A conversation needs a name.', true); return; }
    PDC.api('/api/threads/rename', { id: s.active_thread_id, title: t }).then((j) => {
      if (!j.ok) { PDC.toast(j.detail || 'could not rename', true); return; }
      PDC.toast('Conversation renamed to \u201c' + j.title + '\u201d.');
      swapTranscript();                     // re-poll: list, status and titles refresh
    });
  };
  const m = PDC.modal('Rename conversation', h('div', {},
      inp,
      h('div', { class: 'muted small-text', style: 'margin-top:8px',
        text: 'Hand-picked names stick: neither the first message nor Clear chat will overwrite this one.' })),
    [{ label: 'Cancel' }, { label: 'Rename', primary: true, onclick: go }],
    { narrow: true });
  inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); m.close(); go(); } });
  setTimeout(() => { inp.focus(); inp.select(); }, 30);
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
function planIsComplete() {
  const p = S.plan || {};
  const tasks = p.tasks || [];
  return !!p.exists && tasks.length > 0
    && tasks.every((t) => t.status !== 'todo');
}
function markPlanGenerating(payload) {
  // A new Auto goal against a fully-ticked plan replaces it. Show that intent
  // immediately; applyPlanPayload swaps in the new checklist the moment the
  // planner writes it (before Task 1 begins).
  if (!payload || payload.mode !== 'auto' || !planIsComplete()) return;
  S.ui.planGenerating = true;
  S.ui.planGenFrom = { path: S.plan.path || '', text: S.plan.text || '',
                       mtime: S.plan.mtime || 0 };
  emit();
}
PDC.startRun = (payload) => {
  markPlanGenerating(payload);
  return PDC.api('/api/run', payload).then((j) => {
    if (!j.ok) {
      PDC.toast(j.detail || 'could not start', true);
      if (!job()) { S.ui.planGenerating = false; S.ui.planGenFrom = null; }
    }
    refreshState();
    return j;
  }, (e) => {
    if (!job()) { S.ui.planGenerating = false; S.ui.planGenFrom = null; }
    throw e;
  });
};
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
    clearDraft();
    askHow(text);
    return;
  }
  if (j) {
    if (!text) return;
    if (j.kind === 'task' && s.allow_steering) {
      PDC.api('/api/steer', { text: text }).then((r) => { if (!r.ok) PDC.toast(r.detail || 'not delivered', true); });
      S.drafts[S.mode] = '';
      if (inputEl) inputEl.value = '';
      clearDraft();
    }
    return;
  }
  if (!text && !S.attachments.length && !(S.mode === 'auto' && opt('auto', 'plan_file'))) { PDC.toast('Type something first.', true); return; }
  // Ask works with no project (the scratchpad); task/edit and auto need one.
  const needsProject = S.mode !== 'new' && S.mode !== 'explain' && S.mode !== 'how';
  if (needsProject && !s.root) { PDC.toast('Open a project first.', true); return; }
  if (needsProject && !s.has_index) { PDC.toast('Build the index first.', true); return; }
  if (S.mode === 'new' && !(opt('new', 'project') || '').trim()) {
    PDC.toast('Choose a target directory for the new project.', true); return;
  }
  const body = { mode: S.mode, text: text, attachments: S.attachments.slice() };
  OPTS[S.mode].forEach((d) => { body[d.k] = opt(S.mode, d.k); });
  // Send the topbar's model selection explicitly so a no-project New/Ask uses
  // the profile the user just picked, even before /api/state catches up.
  body.profile = S.ui.pendingProfile !== null ? S.ui.pendingProfile : (s.profile || '');
  if (S.mode === 'auto') {
    body.plan_only = !!body.plan_only;
    body.apply = !body.plan_only;      // Auto applies unless Plan only is ticked
  }
  PDC.startRun(body).then((r) => {
    if (r.ok) {
      S.drafts[S.mode] = '';
      if (inputEl) inputEl.value = '';
      clearDraft();
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
function applyPlanPayload(j) {
  if (!j || typeof j !== 'object') return false;
  const prev = S.plan || {};
  const changed = j.path !== prev.path || j.text !== prev.text
    || !!j.exists !== !!prev.exists || j.mtime !== prev.mtime;
  if (S.ui.planGenerating && j.exists) {
    const from = S.ui.planGenFrom;
    // The planner replaced the old checklist: stop the placeholder and show
    // the brand-new plan right away. `mtime` catches a rewrite whose text
    // happens to be identical.
    if (!from || j.path !== from.path || j.text !== from.text
        || (j.mtime && from.mtime && j.mtime !== from.mtime)) {
      S.ui.planGenerating = false;
      S.ui.planGenFrom = null;
    }
  }
  if (changed) {
    S.plan = j;
    emit();
  }
  return changed;
}
PDC.applyPlanPayload = applyPlanPayload;
function loadPlan() {
  return PDC.api('/api/plan').then((j) => {
    if (j._status !== 200) return;
    applyPlanPayload(j);
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
// Hosted provider profiles are refused: calibration is local-only.
function calibrateProfile(name) {
  if (!name) { PDC.toast('No active profile to calibrate.', true); return; }
  const cal = (S.status.calibration || {})[name] || {};
  if (cal.is_hosted) {
    const model = (S.status.profile_models || {})[name] || S.status.model || '';
    PDC.toast('Calibration is only for local models. Profile \'' + name + '\' (' + model + ') is a hosted provider model and uses provider-table defaults automatically.', true);
    return;
  }
  PDC.confirm('Calibrate \u201c' + name + '\u201d?\n\nA short probe: a few timed calls to measure how fast this model answers, how long a prompt it can actually hold, and how reliably it uses tools. The result is stored in its model card and the tuned limits are written to the profile.', 'Run it').then((ok) => {
    if (!ok) return;
    PDC.api('/api/models/calibrate', { profile: name, mode: 'quick' }).then((j) => {
      if (!j.ok) { PDC.toast(j.detail || 'could not start the calibration', true); return; }
      PDC.toast('Calibrating "' + name + '" \u2014 watch the Output tab.');
      refreshState();
    });
  });
}

// Calibrate All lives in settings.js (it owns the settings payload); load
// that module on demand, exactly like the Settings dialog itself.
function calibrateAllProfiles() {
  if (PDC.calibrateAll) { PDC.calibrateAll(); return; }
  settingsPromise.then(() => { if (PDC.calibrateAll) PDC.calibrateAll(); });
}

// ---------------------------------------------------------------- benchmarks ---
// Model Benchmark Runner: pick a configured profile and one of the six
// built-in diagnostic projects. The server runs generation + both verification
// passes as a background job; this dialog polls /api/benchmarks and then shows
// the scorecard, the report, and a one-click switch into the generated folder.
function openBenchmarks() {
  const st = { catalog: [], profiles: [], profileModels: {}, recent: [],
               bench: '', profile: '', base: '', jobId: 0, timer: 0,
               running: false, last: null };
  const benchSel = h('select', { id: 'benchSelect' });
  const profSel = h('select', { id: 'benchProfile' });
  const baseIn = h('input', { type: 'text', class: 'mono grow', spellcheck: false,
    placeholder: 'optional output parent (defaults to a benchmarks/ folder)' });
  const desc = h('div', { class: 'bench-desc' });
  const statusLine = h('div', { class: 'bench-status' });
  const resultBox = h('div', { class: 'bench-result' });
  const recentBox = h('div', { class: 'bench-recent' });
  const runBtn = h('button', { class: 'primary', text: 'Run Benchmark' });
  const browseBtn = h('button', { text: 'Browse\u2026', onclick: () => {
    PDC.browse({ title: 'Choose an output parent folder', start: baseIn.value || st.base || '',
      pickLabel: 'Use this folder', onPick: (p) => { baseIn.value = p; } });
  } });

  const currentBench = () => st.catalog.find((b) => b.id === benchSel.value) || st.catalog[0];
  const refreshDesc = () => {
    const b = currentBench();
    clear(desc);
    if (!b) return;
    desc.appendChild(h('div', { class: 'bench-title' }, h('b', { text: b.title }),
      h('span', { class: 'tag', text: b.category })));
    desc.appendChild(h('div', { class: 'muted small-text', text: b.description }));
    desc.appendChild(h('div', { class: 'bench-tests', text: 'Exposes: ' + b.what_it_tests }));
    desc.appendChild(h('div', { class: 'muted small-text',
      text: b.checks + ' hidden rubric checks \u00b7 deliverables: ' + (b.deliverables || []).join(', ') }));
  };
  const fmt = (n) => PDC.fmtInt(n);
  const scorecard = (run) => {
    clear(resultBox);
    if (!run) return;
    const rub = run.rubric || { passed: 0, total: 0 };
    const mt = run.model_tests || { passed: 0, total: 0, ran: false, ok: false };
    const ok = rub.passed === rub.total && mt.ok;
    const warn = !ok && (rub.passed > 0 || mt.passed > 0);
    const box = h('div', { class: 'bench-score ' + (ok ? 'ok' : warn ? 'warn' : 'bad') });
    box.appendChild(h('div', { class: 'bench-score-head' },
      h('span', { class: 'bench-big', text: rub.passed + '/' + rub.total }),
      h('span', { class: 'muted', text: ' holdout rubric checks passed' }),
      h('span', { class: 'tag ' + (ok ? 'ok' : warn ? 'act' : ''), text: ok ? 'PASS' : warn ? 'PARTIAL' : 'FAIL' })));
    const grid = h('div', { class: 'bench-grid' });
    const metric = (label, value) => {
      grid.appendChild(h('div', { class: 'bench-metric' },
        h('span', { class: 'muted small-text', text: label }),
        h('b', { text: value })));
    };
    metric('Model tests', mt.ran ? (mt.passed + '/' + mt.total + (mt.ok ? ' passed' : ' failed')) : 'none generated');
    metric('Duration', Number(run.duration || 0).toFixed(1) + 's');
    metric('Tokens', fmt((run.stats || {}).total_tokens || 0));
    metric('Calls', fmt((run.stats || {}).calls || 0));
    metric('Repairs', fmt(run.repair_rounds || 0));
    metric('Profile', run.profile || 'default');
    box.appendChild(grid);
    if (run.svg_preview) {
      box.appendChild(h('figure', { class: 'bench-svg' },
        h('img', { src: run.svg_preview, alt: 'SVG drawing produced by the model' }),
        h('figcaption', { class: 'muted small-text', text: 'Rendered from the generated SVG (sandboxed image, scripts disabled).' })));
    }
    const shortcomings = run.shortcomings || [];
    if (shortcomings.length) {
      const list = h('ul', { class: 'bench-short' });
      shortcomings.slice(0, 10).forEach((s) => list.appendChild(h('li', { text: s })));
      box.appendChild(h('div', { class: 'bench-label', text: 'Shortcomings detected' }));
      box.appendChild(list);
    } else {
      box.appendChild(h('div', { class: 'tip', text: 'No shortcomings: every model test and holdout check passed.' }));
    }
    const actions = h('div', { class: 'row gap bench-actions' },
      h('button', { class: 'primary', text: 'Open Generated Project', onclick: () => {
        m.close();
        openProject(run.project_dir, run.profile || '');
      } }),
      h('button', { text: 'Copy project path', onclick: () => copyText(run.project_dir) }));
    box.appendChild(actions);
    box.appendChild(h('div', { class: 'muted small-text mono bench-path', text: run.project_dir || '' }));
    if (run.report) {
      const details = h('details', { class: 'bench-report' },
        h('summary', { text: 'BENCHMARK_REPORT.md' }),
        h('div', { class: 'md' }));
      render(html`<div class="md">${md(run.report)}</div>`, details.lastChild);
      box.appendChild(details);
    }
    resultBox.appendChild(box);
  };
  const showRecent = (runs) => {
    clear(recentBox);
    if (!runs || !runs.length) return;
    recentBox.appendChild(h('div', { class: 'bench-label', text: 'Recent runs' }));
    runs.slice(0, 8).forEach((r) => {
      const row = h('div', { class: 'bench-recent-row' },
        h('span', { class: 'tag', text: (r.rubric || {}).passed + '/' + (r.rubric || {}).total }),
        h('span', { class: 'bench-recent-title', text: (r.title || r.benchmark_id || '') + ' \u00b7 ' + (r.profile || 'default') }),
        h('span', { class: 'muted small-text', text: (r.started_at || '').replace('T', ' ').slice(0, 19) }));
      row.addEventListener('click', () => { st.last = r; scorecard(r); });
      recentBox.appendChild(row);
    });
  };
  const finish = (run) => {
    st.running = false;
    st.jobId = 0;
    runBtn.disabled = false;
    runBtn.textContent = 'Run Benchmark';
    st.last = run;
    statusLine.textContent = 'Finished.';
    scorecard(run);
    PDC.api('/api/benchmarks').then((j) => { if (j.ok) showRecent(j.recent || []); });
  };
  const poll = () => {
    clearTimeout(st.timer);
    PDC.api('/api/benchmarks').then((j) => {
      if (!j.ok) { statusLine.textContent = j.detail || 'Could not read benchmark status.'; return; }
      if (j.last_run && st.jobId && j.last_run.job_id === st.jobId) { finish(j.last_run); return; }
      if (!st.jobId) return;
      if (!j.job || j.job.id !== st.jobId) {
        st.running = false;
        runBtn.disabled = false;
        runBtn.textContent = 'Run Benchmark';
        statusLine.textContent = 'The benchmark job ended without a report; see Activity/Chat for the error.';
        return;
      }
      statusLine.textContent = 'Running\u2026 generation and verification can take a few minutes. Watch Activity/Chat for live progress.';
      st.timer = setTimeout(poll, 1500);
    }).catch(() => { st.timer = setTimeout(poll, 2500); });
  };
  const run = () => {
    const b = currentBench();
    if (!b) { PDC.toast('No benchmark selected.', true); return; }
    if (st.running) return;
    st.running = true;
    st.last = null;
    runBtn.disabled = true;
    runBtn.textContent = 'Running\u2026';
    clear(resultBox);
    statusLine.textContent = 'Starting\u2026';
    PDC.api('/api/benchmarks/run', {
      profile: profSel.value, benchmark_id: b.id, base_dir: baseIn.value.trim(),
    }).then((j) => {
      if (!j.ok) {
        st.running = false; runBtn.disabled = false; runBtn.textContent = 'Run Benchmark';
        statusLine.textContent = j.detail || 'Could not start the benchmark.';
        PDC.toast(j.detail || 'benchmark did not start', true);
        return;
      }
      st.jobId = j.job;
      statusLine.textContent = 'Running\u2026 watch Activity/Chat for live progress.';
      poll();
    }).catch((e) => {
      st.running = false; runBtn.disabled = false; runBtn.textContent = 'Run Benchmark';
      statusLine.textContent = 'Could not start the benchmark: ' + e;
    });
  };
  benchSel.addEventListener('change', () => { refreshDesc(); emit(); });
  runBtn.addEventListener('click', run);

  const body = h('div', { class: 'bench-wrap' },
    h('div', { class: 'bench-intro' },
      h('h4', { text: 'Test a profile against a diagnostic mini-project' }),
      h('div', { class: 'muted small-text', text: 'Pedacito generates the project with the selected profile, runs the model\u2019s own tests, then runs a hidden deterministic rubric. The generated folder is kept on disk.' })),
    h('div', { class: 'bench-form' },
      h('label', { class: 'bench-field' }, h('span', { text: 'Profile / model' }), profSel),
      h('label', { class: 'bench-field' }, h('span', { text: 'Benchmark task' }), benchSel)),
    desc,
    h('label', { class: 'bench-field' }, h('span', { text: 'Output parent directory' }),
      h('div', { class: 'row gap' }, baseIn, browseBtn)),
    h('div', { class: 'row gap bench-runrow' }, runBtn, statusLine),
    resultBox, recentBox);
  const m = PDC.modal('Model Benchmark Runner', body, [{ label: 'Close' }], { wide: true });
  PDC.api('/api/benchmarks').then((j) => {
    if (!j.ok) { statusLine.textContent = j.detail || 'Could not load benchmarks.'; return; }
    st.catalog = j.benchmarks || [];
    st.profiles = j.profiles || [];
    st.profileModels = S.status.profile_models || {};
    st.base = j.default_base_dir || '';
    baseIn.value = st.base;
    clear(benchSel);
    st.catalog.forEach((b) => benchSel.appendChild(h('option', { value: b.id, text: b.title })));
    clear(profSel);
    const active = S.status.profile || '';
    const names = st.profiles;
    profSel.appendChild(h('option', { value: '', text: '(default' + (S.status.default_profile ? ': ' + S.status.default_profile : '') + ')' }));
    names.forEach((p) => profSel.appendChild(h('option', { value: p,
      text: p + (st.profileModels[p] ? ' \u00b7 ' + st.profileModels[p] : '') })));
    if (active && names.includes(active)) profSel.value = active;
    if (!st.catalog.length) { statusLine.textContent = 'No benchmarks are available.'; return; }
    benchSel.value = st.catalog[0].id;
    refreshDesc();
    showRecent(j.recent || []);
    if (j.last_run) scorecard(j.last_run);
  }).catch(() => { statusLine.textContent = 'Could not load the benchmark catalog.'; });
}
PDC.openBenchmarks = openBenchmarks;

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

// Deterministic symbol rename: the server moves every exact
// identifier through the tokenizer / tree-sitter, so strings and comments that
// merely contain the old name survive and no model tokens are spent. Preview
// first (nothing written), then Rename (originals snapshotted, so "Restore
// from a backup" undoes it).
function renameSymbolDialog() {
  if (!S.status.root) { PDC.toast('Open a project first.', true); return; }
  const oldInp = h('input', { type: 'text', class: 'grow', spellcheck: false,
    autocomplete: 'off', placeholder: 'current name, e.g. compute_total' });
  const newInp = h('input', { type: 'text', class: 'grow', spellcheck: false,
    autocomplete: 'off', placeholder: 'new name' });
  const scopeInp = h('input', { type: 'text', class: 'grow', spellcheck: false,
    autocomplete: 'off', placeholder: 'optional: src/orders.py (only it plus its importers)' });
  const out = h('pre', { class: 'diffwrap', style: 'max-height:46vh;overflow:auto;margin:8px 0 0' });
  const row = (label, input) => h('div', { style: 'display:flex;gap:8px;align-items:center;margin-bottom:6px' },
    h('label', { class: 'muted small-text', style: 'min-width:88px', text: label }), input);
  const body = h('div', {},
    row('Rename', oldInp), row('To', newInp), row('Scope file', scopeInp),
    h('div', { class: 'muted small-text', text: 'Exact identifier nodes only (definitions, imports, call sites). Strings and comments are never touched. Python, JavaScript/TypeScript, Go and Rust.' }),
    out);
  const payload = (apply) => {
    const old = (oldInp.value || '').trim(), nw = (newInp.value || '').trim();
    if (!old || !nw) { PDC.toast('Both names are required.', true); return null; }
    if (old === nw) { PDC.toast('The new name equals the old one.', true); return null; }
    const p = { old: old, new: nw, apply: apply };
    const scope = (scopeInp.value || '').trim();
    if (scope) p.file = scope;
    return p;
  };
  const show = (j) => {
    if (!j.ok) { out.textContent = j.detail || j.error || 'rename failed'; return; }
    if (!(j.files_modified || []).length) {
      out.textContent = 'No exact identifier occurrence found' +
        (j.skipped && j.skipped.length ? '\nskipped: ' + j.skipped.slice(0, 8).join('\n') : '');
      return;
    }
    const head = (j.dry_run ? 'would rename ' : 'renamed ') + j.total_replacements +
      ' occurrence(s) in ' + j.files_modified.length + ' file(s)\n';
    const diffs = j.files_modified.map((rel) => rel + ' (' + (j.per_file[rel] || 0) + ')\n' +
      (j.diffs[rel] || '')).join('\n');
    out.textContent = head + '\n' + diffs;
  };
  const m = PDC.modal('Rename a symbol across the project', body, [
    { label: 'Cancel' },
    { label: 'Preview', stay: true, onclick: () => {
      const p = payload(false);
      if (p) PDC.api('/api/refactor/rename', p).then(show).catch(() => PDC.toast('server unreachable', true));
    } },
    { label: 'Rename', danger: true, onclick: (close) => {
      const p = payload(true);
      if (!p) return false;
      PDC.api('/api/refactor/rename', p).then((j) => {
        show(j);
        if (j.ok && (j.files_modified || []).length) {
          PDC.toast(j.total_replacements + ' occurrence(s) renamed in ' + j.files_modified.length +
            ' file(s)' + (j.backup ? '; backup: ' + j.backup : '') + '.');
          refreshState();
        } else if (!j.ok) PDC.toast(j.detail || j.error || 'rename refused', true);
      }).catch(() => PDC.toast('server unreachable', true));
      return false;                      // keep the dialog open to read the result
    } }], { wide: true });
  setTimeout(() => { oldInp.focus(); }, 30);
  return m;
}

// ------------------------------------------------------------------ project dialogs ---
function _needRoot() {
  if (!S.status.root) { PDC.toast('Open a project first.', true); return false; }
  return true;
}
function _bytesText(n) {
  n = Number(n || 0);
  return n >= 1048576 ? (n / 1048576).toFixed(1) + ' MB'
    : n >= 1024 ? (n / 1024).toFixed(1) + ' KB' : n + ' B';
}

// Project memory (`.pedacito/memory.md`) plus the auto-learned repair lessons
// in `micro_facts.json`. Adding and forgetting are the same operations the CLI
// exposes; deleting a lesson only removes that one ledger row.
function memoryDialog() {
  // Memory is workspace-backed: with no project open the server reads and
  // writes the persistent scratchpad instead of refusing.
  const body = h('div', {});
  const list = h('div', {});
  const note = h('div', { class: 'muted small-text' });
  const text = h('input', { type: 'text', class: 'grow', spellcheck: false,
    placeholder: 'a fact about this repo, e.g. tests run with pytest -q' });
  const scope = h('select', {}, [
    h('option', { value: 'project', text: 'this project' }),
    h('option', { value: 'global', text: 'all projects' })]);
  const add = h('button', { class: 'primary', text: 'Remember', onclick: () => {
    const t = (text.value || '').trim();
    if (!t) { PDC.toast('Type a fact first.', true); return; }
    PDC.api('/api/memory', { action: 'add', text: t, scope: scope.value }).then((j) => {
      if (!j.ok && !j.stored) PDC.toast(j.detail || 'not stored', true);
      else PDC.toast('Remembered.');
      text.value = '';
      paint(j);
    });
  } });
  function facts(rows) {
    const box = h('div', {});
    if (!rows.length) { box.appendChild(h('div', { class: 'tip', text: 'No remembered facts yet.' })); return box; }
    rows.forEach((e) => {
      box.appendChild(h('div', { class: 'row small-text', style: 'padding:2px 0' },
        h('span', { class: 'tag', text: e.scope + '/' + e.category }),
        h('span', { class: 'grow', text: e.text }),
        h('button', { class: 'ghost small danger', text: 'Forget', onclick: () => {
          PDC.api('/api/memory', { action: 'forget', text: e.text, scope: e.scope })
            .then((j) => { PDC.toast('Forgot ' + (j.forgot || 0) + ' fact(s).'); paint(j); });
        } })));
    });
    return box;
  }
  function lessons(rows, label) {
    const box = h('div', {});
    if (!rows.length) return box;
    box.appendChild(h('div', { class: 'muted small-text', style: 'margin-top:8px', text: label }));
    rows.forEach((l) => {
      const id = l.lesson_id || '';
      box.appendChild(h('div', { class: 'row small-text', style: 'padding:2px 0' },
        h('span', { class: 'tag', text: l.kind || 'lesson' }),
        h('span', { class: 'grow', text: l.text || '' }),
        id ? h('button', { class: 'ghost small danger', text: 'Delete', onclick: () => {
          PDC.api('/api/memory', { action: 'forget_lesson', id: id })
            .then((j) => { PDC.toast('Lesson deleted.'); paint(j); });
        } }) : null));
    });
    return box;
  }
  function paint(j) {
    PDC.clear(list);
    if (!j || !j.ok) { note.textContent = (j && j.detail) || 'Could not load project memory.'; return; }
    list.appendChild(h('b', { text: 'Remembered facts' }));
    list.appendChild(facts(j.entries || []));
    list.appendChild(h('b', { text: 'Auto-learned repair lessons', style: 'display:block;margin-top:12px' }));
    list.appendChild(h('div', { class: 'muted small-text',
      text: 'Distilled from broken -> fixed runs. Hints, not truth; delete one that no longer applies.' }));
    list.appendChild(lessons(j.lessons || [], 'Live lessons'));
    list.appendChild(lessons(j.promoted || [], 'Promoted into path-scoped rules'));
    list.appendChild(lessons(j.retired || [], 'Retired (zero measured effect)'));
    note.textContent = j.project_file || '';
  }
  body.appendChild(h('div', { class: 'row', style: 'margin-bottom:10px' },
    text, scope, add));
  body.appendChild(note);
  body.appendChild(list);
  PDC.modal('Project memory & repo lessons', body, [{ label: 'Close' }], { wide: true });
  PDC.api('/api/memory').then(paint, () => {
    note.textContent = 'The GUI server did not answer.';
  });
}

// Skills library & creator: browse the bundled starters plus user/project
// custom skills, preview which checklists a task would inject, and write new
// skills by hand or with a one-call model draft. The same operations as
// `pedacito skills list|show|add|draft|rm`.
function skillsDialog() {
  const body = h('div', {});
  const list = h('div', {});
  const detail = h('div', {});
  const editor = h('div', {});
  const note = h('div', { class: 'muted small-text' });
  let rows = [], matchNames = [];
  const filter = h('input', { type: 'text', class: 'grow', spellcheck: false,
    placeholder: 'filter by name or description\u2026',
    oninput: () => paintList() });
  const task = h('input', { type: 'text', class: 'grow', spellcheck: false,
    placeholder: 'type a task to preview which skills match (Enter)',
    onkeydown: (e) => { if (e.key === 'Enter') previewMatches(); } });
  const newBtn = h('button', { class: 'primary', text: '+ New Custom Skill',
    onclick: () => openEditor(null) });

  function previewMatches() {
    const t = (task.value || '').trim();
    if (!t) { matchNames = []; note.textContent = ''; paintList(); return; }
    PDC.api('/api/skills?task=' + encodeURIComponent(t)).then((j) => {
      if (!j.ok) { PDC.toast(j.detail || 'could not preview matches', true); return; }
      matchNames = (j.matches || []).map((m) => m.name);
      note.textContent = matchNames.length
        ? 'This task injects: ' + matchNames.join(', ')
          + ' (top 2 matches, \u22641,200 chars of checklist).'
        : 'No skill triggers match this task.';
      paintList();
    });
  }

  function paintList() {
    PDC.clear(list);
    const needle = (filter.value || '').trim().toLowerCase();
    const shown = rows.filter((r) => !needle
      || r.name.indexOf(needle) >= 0
      || (r.description || '').toLowerCase().indexOf(needle) >= 0);
    if (!shown.length) {
      list.appendChild(h('div', { class: 'tip', text: 'No skills match the filter.' }));
      return;
    }
    shown.forEach((r) => {
      const matched = matchNames.indexOf(r.name) >= 0;
      const custom = r.scope !== 'bundled';
      list.appendChild(h('div', { class: 'row small-text', style: 'padding:3px 0;cursor:pointer',
        onclick: () => showDetail(r) },
        h('span', { class: 'tag' + (matched ? ' act' : ''), text: r.scope }),
        h('b', { text: r.name, style: 'min-width:200px' }),
        matched ? h('span', { class: 'tag act', text: 'matches task' }) : null,
        h('span', { class: 'grow muted', text: r.description }),
        custom ? h('button', { class: 'ghost small', text: 'Edit', onclick: (e) => {
          e.stopPropagation(); openEditor(r);
        } }) : null,
        custom ? h('button', { class: 'ghost small danger', text: 'Delete', onclick: (e) => {
          e.stopPropagation();
          PDC.confirm('Delete the ' + r.scope + ' skill \u201c' + r.name + '\u201d?', 'Delete', true).then((ok) => {
            if (!ok) return;
            PDC.api('/api/skills/delete', { name: r.name, scope: r.scope }).then((j) => {
              if (!j.ok) { PDC.toast(j.detail || 'could not delete', true); return; }
              PDC.toast('Skill deleted.');
              rows = j.skills || []; paintList(); PDC.clear(detail);
            });
          });
        } }) : null));
    });
  }

  function showDetail(r) {
    PDC.clear(detail);
    detail.appendChild(h('div', { class: 'note-box', style: 'margin-top:10px' },
      h('div', { class: 'row' },
        h('b', { text: r.name }),
        h('span', { class: 'tag', text: r.scope }),
        h('span', { class: 'muted small-text grow', text: r.source || '' })),
      h('div', { class: 'small-text', style: 'margin:4px 0',
        text: 'triggers: ' + ((r.triggers || []).join(', ') || '(none)') }),
      h('pre', { class: 'diffwrap', style: 'max-height:26vh;overflow:auto;margin:0;white-space:pre-wrap',
        text: r.body || '' })));
  }

  // -- the custom-skill editor (new or editing an existing custom skill) ----
  const nameInp = h('input', { type: 'text', class: 'grow', spellcheck: false,
    placeholder: 'slug, e.g. alpine-dark-mode' });
  const descInp = h('input', { type: 'text', class: 'grow',
    placeholder: 'one line: when to use this skill (also the draft prompt)' });
  const trigInp = h('input', { type: 'text', class: 'grow', spellcheck: false,
    placeholder: 'comma-separated keywords, e.g. alpine, dark mode, toggle' });
  const bodyInp = h('textarea', { rows: 7, spellcheck: false, style: 'width:100%',
    placeholder: 'the numbered checklist the model follows (prompt text, never executed)' });
  const draftBtn = h('button', { text: '\u2726 Auto-Draft from Description', onclick: () => {
    const prompt = (descInp.value || '').trim();
    if (!prompt) { PDC.toast('Describe what the skill should teach first.', true); return; }
    draftBtn.disabled = true; draftBtn.textContent = '\u2726 Drafting\u2026';
    PDC.api('/api/skills/draft', { prompt: prompt }).then((j) => {
      draftBtn.disabled = false; draftBtn.textContent = '\u2726 Auto-Draft from Description';
      if (!j.ok) { PDC.toast(j.detail || 'draft failed', true); return; }
      const d = j.draft || {};
      if (!nameInp.value.trim()) nameInp.value = d.name || '';
      descInp.value = d.description || descInp.value;
      trigInp.value = (d.triggers || []).join(', ');
      bodyInp.value = d.body || '';
      PDC.toast('Draft ready \u2014 review it, then save.');
    }, () => {
      draftBtn.disabled = false; draftBtn.textContent = '\u2726 Auto-Draft from Description';
      PDC.toast('The GUI server did not answer.', true);
    });
  } });
  function save(scope) {
    const payload = { name: nameInp.value.trim(), description: descInp.value.trim(),
      triggers: trigInp.value, body: bodyInp.value, scope: scope };
    if (!payload.name) { PDC.toast('Give the skill a slug name.', true); return; }
    PDC.api('/api/skills', payload).then((j) => {
      if (!j.ok) { PDC.toast(j.detail || 'could not save', true); return; }
      PDC.toast('Saved \u201c' + payload.name + '\u201d to the ' + scope + ' library.');
      rows = j.skills || []; paintList(); PDC.clear(editor);
    });
  }
  function openEditor(r) {
    PDC.clear(editor);
    nameInp.value = r ? r.name : '';
    descInp.value = r ? r.description : '';
    trigInp.value = r ? (r.triggers || []).join(', ') : '';
    bodyInp.value = r ? r.body : '';
    editor.appendChild(h('div', { class: 'note-box', style: 'margin-top:10px' },
      h('b', { text: r ? 'Edit skill: ' + r.name : 'New custom skill' }),
      h('div', { class: 'form', style: 'margin-top:6px' },
        h('label', { text: 'Name (slug)' }), nameInp,
        h('label', { text: 'Description' }), descInp,
        h('label', { text: 'Triggers' }), trigInp,
        h('label', { text: 'Checklist body' }), bodyInp),
      h('div', { class: 'row', style: 'margin-top:8px' },
        draftBtn,
        h('span', { class: 'grow' }),
        h('button', { text: 'Cancel', onclick: () => PDC.clear(editor) }),
        h('button', { class: 'primary', text: 'Save to Project',
          title: S.status.root ? '' : 'No project open: saves into the scratchpad project',
          onclick: () => save('project') }),
        h('button', { class: 'primary', text: 'Save to Global', onclick: () => save('user') }))));
    nameInp.focus();
  }

  body.appendChild(h('div', { class: 'row', style: 'margin-bottom:8px' }, filter, newBtn));
  body.appendChild(h('div', { class: 'row', style: 'margin-bottom:8px' },
    task, h('button', { text: 'Preview matches', onclick: previewMatches })));
  body.appendChild(note);
  body.appendChild(list);
  body.appendChild(detail);
  body.appendChild(editor);
  PDC.modal('Skills library & creator', body, [{ label: 'Close' }], { wide: true });
  PDC.api('/api/skills').then((j) => {
    if (!j.ok) { note.textContent = j.detail || 'Could not load the skills library.'; return; }
    rows = j.skills || [];
    paintList();
  }, () => { note.textContent = 'The GUI server did not answer.'; });
}
PDC.skillsDialog = skillsDialog;

// Session consolidation: distil the transcript into a handoff checkpoint and
// restart the chat from it, so the next turn starts with a small prompt. The
// server resets the transcript and the Context Meter follows immediately.
function consolidateSession() {
  if (job()) { PDC.toast('Stop the running job first.', true); return; }
  PDC.confirm('Consolidate this session?\n\nThe transcript is distilled into a short handoff checkpoint (.pedacito/consolidated_session.md) and the chat restarts from that handoff alone, so the next turn uses far fewer context tokens.',
    'Consolidate').then((ok) => {
    if (!ok) return;
    PDC.api('/api/session/consolidate', {}).then((j) => {
      if (!j.ok) { PDC.toast(j.detail || 'could not consolidate', true); return; }
      PDC.toast('Consolidated: ~' + PDC.fmtInt(j.before_tokens) + ' \u2192 ~'
        + PDC.fmtInt(j.after_tokens) + ' tokens.');
      if (j.status) applyStatus(j.status);
    }, () => PDC.toast('The GUI server did not answer.', true));
  });
}
PDC.consolidateSession = consolidateSession;

// Background dev servers: the `.pedacito/procs/` registry, with a live log
// tail and a Stop button, and the same start form `pedacito serve` uses.
function procsDialog() {
  // Workspace-backed: background servers live under the scratchpad when no
  // project is open.
  const body = h('div', {});
  const list = h('div', {});
  const logs = h('pre', { class: 'diffwrap', style: 'max-height:32vh;overflow:auto;margin:10px 0 0' });
  const command = h('input', { type: 'text', class: 'grow', spellcheck: false,
    placeholder: 'npm run dev  (or: python -m http.server 8000)' });
  const name = h('input', { type: 'text', placeholder: 'name (optional)', style: 'width:150px' });
  const status = h('span', { class: 'muted small-text grow' });
  const start = h('button', { class: 'primary', text: 'Start server', onclick: () => {
    const cmd = (command.value || '').trim();
    if (!cmd) { PDC.toast('Type a command first.', true); return; }
    start.disabled = true; status.textContent = 'Starting\u2026';
    PDC.api('/api/procs', { action: 'start', command: cmd, name: (name.value || '').trim() })
      .then((j) => {
        start.disabled = false;
        if (!j.ok) { status.textContent = j.detail || 'could not start it'; return; }
        command.value = '';
        status.textContent = j.process.name + ' [' + j.process.status + ']'
          + (j.process.url ? '  ' + j.process.url : '');
        paint(j);
      });
  } });
  function paint(j) {
    PDC.clear(list);
    if (!j || !j.ok) { status.textContent = (j && j.detail) || 'Could not read the registry.'; return; }
    const rows = j.processes || [];
    if (!rows.length) { list.appendChild(h('div', { class: 'tip', text: 'No background servers. Start one above.' })); return; }
    rows.forEach((p) => {
      const live = p.status === 'running';
      list.appendChild(h('div', { class: 'note-box', style: 'margin-bottom:8px' },
        h('div', { class: 'row' },
          h('b', { text: p.name }),
          h('span', { class: 'tag' + (live ? ' act' : ''), text: p.status || '?' }),
          h('span', { class: 'muted small-text', text: 'pid ' + (p.pid || 0) }),
          p.url ? h('span', { class: 'tag', text: p.url }) : null,
          h('span', { class: 'grow' }),
          h('button', { class: 'small', text: 'Logs', onclick: () => {
            PDC.api('/api/procs', { action: 'logs', name: p.name, tail: 6000 })
              .then((x) => { logs.textContent = x.logs || x.detail || '(no output)'; });
          } }),
          h('button', { class: 'small danger', text: 'Stop', disabled: !live, onclick: () => {
            PDC.api('/api/procs', { action: 'stop', name: p.name }).then((x) => {
              PDC.toast(x.detail || 'stopped'); paint(x);
            });
          } })),
        h('div', { class: 'mono small-text', style: 'margin-top:4px', text: p.command || '' })));
    });
  }
  body.appendChild(h('div', { class: 'row', style: 'margin-bottom:8px' }, command, name, start));
  body.appendChild(status);
  body.appendChild(list);
  body.appendChild(logs);
  PDC.modal('Background servers', body, [{ label: 'Close' }], { wide: true });
  PDC.api('/api/procs').then(paint, () => { status.textContent = 'The GUI server did not answer.'; });
}

// On-demand project checks: diagnose (linter/typecheck), webcheck (static web
// contracts) and domcheck (headless DOM smoke). The server runs the check
// synchronously and returns its report.
function checksDialog() {
  if (!_needRoot()) return;
  const out = h('pre', { class: 'diffwrap', style: 'max-height:52vh;overflow:auto;margin:8px 0 0' });
  const status = h('div', { class: 'muted small-text' });
  const buttons = h('div', { class: 'row', style: 'margin-bottom:8px' });
  const kinds = [
    ['diagnose', 'Linter / Typecheck (diagnose)', 'Runs the project\u2019s own ruff/pyflakes/mypy/tsc/cargo/... checks.'],
    ['webcheck', 'Static Web Contracts (webcheck)', 'Missing <script src> targets, DOM ids, ESM behind classic scripts, undeclared imports.'],
    ['domcheck', 'Headless DOM Smoke Check (domcheck)', 'Loads the entry page in headless Chromium and proves the module graph mounts (needs Playwright).'],
  ];
  function run(kind, label) {
    status.textContent = 'Running ' + label + '\u2026';
    out.textContent = '';
    PDC.api('/api/project/check', { kind: kind }).then((j) => {
      if (!j.ok) { status.textContent = j.detail || 'the check could not run'; return; }
      status.textContent = (j.passed ? '\u2713 PASS' : '\u2717 findings') + ' \u2014 ' + kind
        + (j.count !== undefined ? ' (' + j.count + ' finding(s))' : '');
      out.textContent = j.report || '(no output)';
      PDC.refreshState();
    }, () => { status.textContent = 'The GUI server did not answer.'; });
  }
  kinds.forEach((k) => {
    buttons.appendChild(h('button', { class: 'small', text: k[1], title: k[2],
      onclick: () => run(k[0], k[1]) }));
  });
  PDC.modal('Run project checks', h('div', {}, buttons, status, out),
    [{ label: 'Close' }], { wide: true });
}

// Retention for sessions, backups, events and log.txt (same as `pedacito prune`).
function pruneDialog() {
  // Workspace-backed: prunes the scratchpad when no project is open.
  const sessions = h('input', { type: 'number', min: 0, placeholder: 'keep newest N' });
  const backups = h('input', { type: 'number', min: 0, placeholder: 'keep newest N' });
  const out = h('div', { class: 'note-box', style: 'margin-top:10px' });
  const m = PDC.modal('Prune old sessions/backups', h('div', {},
    h('p', { class: 'small-text', text: 'Blank means the config\u2019s keep_sessions / keep_backups value (0 leaves that kind alone). Backups are the undo path: only the oldest beyond the limit go.' }),
    h('div', { class: 'row' },
      h('label', { text: 'Keep sessions' }), sessions,
      h('label', { text: 'Keep backups' }), backups),
    out), [
    { label: 'Cancel' },
    { label: 'Preview', onclick: () => {
      PDC.api('/api/project/prune', { keep_sessions: sessions.value || undefined,
        keep_backups: backups.value || undefined, dry_run: true })
        .then((j) => { out.className = 'note-box'; out.textContent = j.ok ? j.summary : (j.detail || 'failed'); });
      return false;
    } },
    { label: 'Prune', danger: true, onclick: () => {
      PDC.api('/api/project/prune', { keep_sessions: sessions.value || undefined,
        keep_backups: backups.value || undefined })
        .then((j) => {
          out.className = 'note-box';
          out.textContent = j.ok ? j.summary : (j.detail || 'failed');
          if (j.ok) { PDC.toast(j.summary); refreshState(); }
        });
      return false;
    } }], { wide: true });
  return m;
}

// `pedacito reset` from the GUI: preview first (files/KB and servers that would
// stop), then execute with the authored-state checkboxes.
function resetDialog() {
  // Workspace-backed: resets the scratchpad's .pedacito/ state when no
  // project is open.
  const keepMemory = h('input', { type: 'checkbox', checked: true });
  const keepConfig = h('input', { type: 'checkbox', checked: true });
  const reindex = h('input', { type: 'checkbox', checked: false });
  const out = h('div', { class: 'note-box', style: 'margin-top:10px' });
  const confirm = h('input', { type: 'checkbox' });
  const body = h('div', {},
    h('p', { class: 'small-text', text: 'Deletes this project\u2019s .pedacito/ runtime state (index, sessions, backups, events, memory unless kept, generated rules without the authored checkbox). Source code and .pedacito.toml are never touched. Running background servers are stopped first.' }),
    h('div', { class: 'form' },
      h('label', { class: 'chk' }, keepMemory, ' Keep remembered facts (memory.md)'),
      h('label', { class: 'chk' }, keepConfig, ' Keep authored rules, skills & prompts'),
      h('label', { class: 'chk' }, reindex, ' Rebuild the static index after reset'),
      h('label', { class: 'chk' }, confirm, ' I understand the index and backups will be removed')),
    out);
  const options = () => ({ keep_memory: keepMemory.checked,
                           keep_config: keepConfig.checked,
                           reindex: reindex.checked });
  const show = (j, dry) => {
    if (!j.ok) { out.className = 'note-box err'; out.textContent = j.detail || 'reset failed'; return; }
    out.className = 'note-box';
    PDC.clear(out);
    out.appendChild(h('b', { text: dry ? 'Preview (nothing deleted)' : 'Reset complete' }));
    out.appendChild(h('div', { class: 'small-text',
      text: (j.files || 0) + ' file(s), ' + _bytesText(j.bytes)
        + ' under .pedacito/ ' + (dry ? 'would be cleared.' : 'were cleared.') }));
    if ((j.running || []).length) {
      out.appendChild(h('div', { class: 'small-text',
        text: (dry ? 'Would stop' : 'Stopped') + ' background server(s): '
          + (dry ? j.running : j.stopped).join(', ') }));
    }
    out.appendChild(h('div', { class: 'small-text',
      text: 'Kept: ' + ((j.kept || []).join(', ') || '(nothing)') }));
    out.appendChild(h('div', { class: 'small-text',
      text: 'Removed: ' + ((j.removed || []).join(', ') || '(nothing)') }));
    if (j.reindex_started) out.appendChild(h('div', { class: 'small-text', text: 'Index rebuild started.' }));
  };
  PDC.modal('Reset project state', body, [
    { label: 'Cancel' },
    { label: 'Preview (Dry Run)', onclick: () => {
      PDC.api('/api/project/reset', { ...options(), dry_run: true })
        .then((j) => show(j, true));
      return false;
    } },
    { label: 'Reset State', danger: true, onclick: () => {
      if (!confirm.checked) { PDC.toast('Tick the confirmation first.', true); return false; }
      PDC.api('/api/project/reset', options()).then((j) => {
        show(j, false);
        if (j.ok) { PDC.toast('Project state reset.'); refreshState(); }
      });
      return false;
    } }], { wide: true });
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
// Composer splitter: the prompt textarea's height is the `--composer-h` CSS
// variable (app.css default 110px, restored before first paint by theme.js),
// clamped between 64px and 65% of the viewport so the bottom buttons always
// stay on screen. Dragging trades height with the chat transcript above; the
// final size persists under its own key, double-click toggles 320/96, and a
// fresh refined spec auto-expands a too-small composer so it can be read.
let composerH = 110;
try {
  const stored = parseInt(localStorage.getItem('pedacito_composer_height') || '', 10);
  if (stored) composerH = stored;
} catch (e) {}
function clampComposerH(px) {
  return Math.max(64, Math.min(Math.round(px), Math.round(window.innerHeight * 0.65)));
}
function setComposerH(px, save) {
  composerH = clampComposerH(px);
  setVar('--composer-h', composerH);
  const el = $('input');
  if (el) el.style.height = '';        // a native resize grip must not win
  if (save) {
    try { localStorage.setItem('pedacito_composer_height', String(composerH)); } catch (e) {}
  }
}
function composerHeightNow() {
  const el = $('input');
  return (el && el.offsetHeight) || composerH;
}
function dragComposer(e) {
  e.preventDefault();
  const el = e.currentTarget;
  const start = composerHeightNow(), y0 = e.clientY;
  const move = (ev) => {
    el.classList.add('dragging');
    setComposerH(start + (y0 - ev.clientY));       // up = taller
  };
  const up = () => {
    el.classList.remove('dragging');
    document.body.style.userSelect = '';
    document.removeEventListener('mousemove', move);
    document.removeEventListener('mouseup', up);
    setComposerH(composerHeightNow(), true);       // keep the final height
  };
  document.body.style.userSelect = 'none';
  document.addEventListener('mousemove', move);
  document.addEventListener('mouseup', up);
}
function toggleComposer() {
  setComposerH(composerHeightNow() >= 200 ? 96 : 320, true);
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
            disabled=${!!props.busy && !!s.root}
            onchange=${(e) => props.onChange(e.target.value)}>
      <option value="">(default${defName ? ': ' + defName : ''})</option>
      ${profiles.map((p) => html`<option key=${p} value=${p}>${p}${(s.profile_models || {})[p] ? ' \u00B7 ' + s.profile_models[p] : ''}</option>`)}
    </select>`;
}

// ------------------------------------------------- calibration & telemetry ---
// The toolbar badge for the active profile's model card: green when fresh,
// amber when stale, muted when absent, cloud for hosted APIs (which never
// calibrate). Clicking opens Settings -> Model discovery, where the
// Calibrate buttons live.
function calChoicesTip(ch) {
  if (!ch) return '';
  const keys = ['arch_family', 'prompt_pack', 'answer_mode', 'best_of',
                'temperature', 'temperature_answer', 'context_tokens'];
  const bits = keys.filter((k) => ch[k] !== undefined && ch[k] !== null)
                   .map((k) => k + ': ' + ch[k]);
  return bits.length ? '\n' + bits.join('\n') : '';
}
function CalibrationBadge(props) {
  const s = S.status;
  const name = props.name || '';
  if (!name && !s.root) return null;
  const cal = (s.calibration || {})[name] || {};
  const openProfiles = () => callSettings({ tab: 'discovery' });
  if (cal.is_hosted) {
    return html`<span class="chip cal-chip hosted" id="calChip"
      title=${'Hosted provider model \u2014 uses provider-table defaults automatically.\nCalibration disabled for hosted models.'}>
      <b>\u2601 Hosted API</b></span>`;
  }
  if (cal.calibrated && !cal.calibrated_stale) {
    const age = cal.calibrated_age_days || 0;
    const ago = age <= 0 ? 'today' : age + 'd ago';
    return html`<span class="chip cal-chip ok" id="calChip" onClick=${openProfiles}
      title=${'Calibrated ' + (cal.calibrated_at || '') + '\nscore: '
        + Number(cal.card_score || 0).toFixed(2) + calChoicesTip(cal.card_choices)
        + '\nClick to open Settings \u2192 Model discovery'}>
      <b>\u2713 Calibrated (${ago})</b></span>`;
  }
  if (cal.calibrated && cal.calibrated_stale) {
    return html`<span class="chip cal-chip stale" id="calChip" onClick=${openProfiles}
      title=${'The model card is older than 30 days (or from an older calibrate version).\nRe-calibrate to refresh the tuned defaults.\nClick to open Settings \u2192 Model discovery'}>
      <b>\u26A0 Calibration stale (${(cal.calibrated_age_days || 0) + 'd ago'})</b></span>`;
  }
  return html`<span class="chip cal-chip none" id="calChip" onClick=${openProfiles}
    title=${'No model card yet \u2014 calibrate to measure this model and tune its defaults.\nClick to open Settings \u2192 Model discovery'}>
    <b>\u25CB Not calibrated</b></span>`;
}

// The live telemetry footer: context window fill, last turn and session
// tokens, generation speed, KV cache hit, folds, and who is answering.
// Click it to unfold the drawer with the full breakdown.
function TelemetryBar() {
  const s = S.status;
  const [open, setOpen] = useState(false);
  const t = s.telemetry || {};
  if (!s.root || !t.context_tokens) return null;
  const pct = Number(t.context_utilization_pct || 0);
  const heat = pct > 85 ? 'hot' : pct >= 70 ? 'warm' : 'cool';
  const role = t.active_model_role || {};
  const activeName = role.profile || role.model || s.model || '\u2014';
  const activeTag = role.specialty || role.role || 'worker';
  const u = s.usage || {};
  const fmt1 = (n) => Number(n || 0).toFixed(1);
  const drawer = open ? html`
    <div class="tele-drawer" id="teleDrawer">
      <div class="tele-grid">
        <div><span class="muted">Last turn</span><b>${PDC.fmtInt(t.last_prompt_tokens)} in / ${PDC.fmtInt(t.last_completion_tokens)} out</b></div>
        <div><span class="muted">Latency</span><b>${fmt1(t.last_elapsed_s)} s</b></div>
        <div><span class="muted">Speed (last / avg)</span><b>${fmt1(t.last_tok_per_s)} / ${fmt1(t.avg_tok_per_s)} tok/s</b></div>
        <div><span class="muted">Context window</span><b>${PDC.fmtInt(t.context_tokens)} tokens (${s.context_origin || 'default'})</b></div>
        <div><span class="muted">Session</span><b>${PDC.fmtInt(u.calls)} calls \u00B7 ${PDC.fmtTokens(u.tokens_in)} in / ${PDC.fmtTokens(u.tokens_out)} out \u00B7 ${Math.round(u.seconds || 0)}s</b></div>
        <div><span class="muted">KV cache hit</span><b>${fmt1(t.cache_hit_pct)}%</b></div>
        <div><span class="muted">Context folds</span><b>${s.folds || 0}</b></div>
        <div><span class="muted">Cost</span><b>${u.cost ? '$' + Number(u.cost).toFixed(4) : '\u2014'}</b></div>
        <div><span class="muted">Active model</span><b>${activeName} (${activeTag})</b></div>
        <div><span class="muted">Orchestrator</span><b>${s.planner_profile || '\u2014'}</b></div>
        ${s.scout_profile ? html`<div><span class="muted">Scout</span><b>${s.scout_profile}</b></div>` : null}
      </div>
    </div>` : null;
  return html`
    <footer id="telemetryBar" class=${open ? 'open' : ''}>
      ${drawer}
      <div class="tele-row" onClick=${() => setOpen(!open)}
           title=${'Live model telemetry \u2014 click to ' + (open ? 'collapse' : 'expand')}>
        <span class=${'tele-bar ' + heat} title=${'context utilization ' + pct + '%'}>
          <i style=${'width:' + Math.max(0, Math.min(100, pct)) + '%'}></i>
        </span>
        <span class="tele-item">Context: <b>${PDC.fmtInt(t.last_prompt_tokens)} / ${PDC.fmtInt(t.context_tokens)}</b> (${pct}%)</span>
        <span class="tele-item">Turn: <b>${PDC.fmtInt(t.last_prompt_tokens)} in / ${PDC.fmtInt(t.last_completion_tokens)} out</b></span>
        <span class="tele-item">Session: <b>${PDC.fmtTokens(t.session_prompt_tokens)} in / ${PDC.fmtTokens(t.session_completion_tokens)} out</b></span>
        <span class="tele-item">Speed: <b>${fmt1(t.last_tok_per_s)} tok/s</b> (avg ${fmt1(t.avg_tok_per_s)} tok/s) \u00B7 ${fmt1(t.last_elapsed_s)}s</span>
        <span class="tele-item">KV Cache Hit: <b>${Math.round(t.cache_hit_pct || 0)}%</b> | Folds: <b>${s.folds || 0}</b></span>
        <span class="tele-item">Active: <b>${activeName}</b> (${activeTag})</span>
        ${s.planner_profile ? html`<span class="tele-item">Orchestrator: <b>${s.planner_profile}</b></span>` : null}
        <button id="consolidateBtn" class="small ghost"
                title="Compact this session into a handoff checkpoint; the next turn starts from it alone"
                onClick=${(e) => { e.stopPropagation(); consolidateSession(); }}>Consolidate Session</button>
        <i class="tele-caret">${open ? '\u25BE' : '\u25B4'}</i>
      </div>
    </footer>`;
}

function TopBar() {
  // Two dense rows: row 1 is the project side -- brand, a compact
  // path box, Browse/Recent/Open and the Project menu -- against the global
  // actions (Benchmarks, Settings, theme, Exit). Row 2 is a tinted status
  // strip that only reports: profile picker, model + thinking, index, usage
  // and the live engine phase. Everything that *does* something to the
  // project lives in the Project menu, so the strip never needs a resize.
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
    ? (s.index_time ? 'indexed ' + new Date(s.index_time * 1000).toLocaleString()
                    : 'index present') + ' \u2014 Click to re-index project (Shift-click for a quick, no-summaries index)'
    : 'no index yet \u2014 Click to index the project (Shift-click for a quick, no-summaries index)';
  const open = () => openProject((pathRef.current && pathRef.current.value || '').trim(), S.ui.pendingProfile || '');
  const activeProf = s.profile || s.default_profile || '';
  const activeCal = (s.calibration || {})[activeProf] || {};
  const activeHosted = !!activeCal.is_hosted;
  const eng = s.engine || {};
  const phase = eng.phase || (j ? 'gather' : 'idle');
  const phaseName = { idle: 'Idle', gather: 'Gathering', edit: 'Editing',
                      verify: 'Verifying', repair: 'Repairing', done: 'Done' }[phase] || 'Idle';
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
                    onPick: (p) => { if (pathRef.current) pathRef.current.value = p; openProject(p, S.ui.pendingProfile || ''); } })}>Browse\u2026</button>
          <${Menu} button=${html`<button id="recent" title="projects you opened here before">Recent \u25BE</button>`}>
            ${(s.recent || []).length ? (s.recent || []).map((r) => html`
              <button key=${r} class="menuitem ellip" title=${r}
                      onClick=${() => { if (pathRef.current) pathRef.current.value = r; openProject(r, ''); }}>${r}</button>`)
              : html`<button class="menuitem" disabled>no recent projects yet</button>`}
          <//>
          <button class="primary" id="openBtn" disabled=${!!j} onClick=${open}>Open</button>
          <button id="newProjectBtn" title="start a new project from a description"
                  disabled=${!!j} onClick=${startNewProject}>+ New</button>
          <${Menu} button=${html`<button id="projMenuBtn" title="calibrate, index, restore and other project actions">Project \u25BE</button>`}>
            <${MenuItem} label="New project\u2026" disabled=${!!j} action=${startNewProject} />
            <${MenuItem} label="Calibrate active profile" disabled=${!!j || !activeProf || activeHosted}
                       title=${activeHosted ? 'Calibration disabled for hosted models' : ''}
                       action=${() => calibrateProfile(activeProf)} />
            <${MenuItem} label="Calibrate All Local Models\u2026" disabled=${!!j}
                       title="Calibrate every local profile in turn (hosted models are skipped)"
                       action=${calibrateAllProfiles} />
            <${MenuItem} label="Check model (loads it)" disabled=${!!j || !s.root} action=${checkModel} />
            <${MenuItem} label="Index project (with summaries)" disabled=${!!j || !s.root} action=${() => buildIndex(true)} />
            <${MenuItem} label="Quick index (no summaries)" disabled=${!!j || !s.root} action=${() => buildIndex(false)} />
            <${MenuItem} label="Restore from a backup\u2026" disabled=${!!j || !(s.snapshots || []).length} action=${restoreDialog} />
            <${MenuItem} label="Rename symbol\u2026" disabled=${!!j || !s.root} action=${renameSymbolDialog} />
            <${MenuItem} label="Memory & repo lessons\u2026" action=${memoryDialog}
                       title="scratchpad memory while no project is open" />
            <${MenuItem} label="Skills library & creator\u2026" action=${skillsDialog}
                       title="browse the built-in, user and project skills; write or auto-draft custom ones" />
            <${MenuItem} label="Background servers\u2026" action=${procsDialog}
                       title="servers live in the scratchpad while no project is open" />
            <${MenuItem} label="Run project checks\u2026" disabled=${!s.root} action=${checksDialog} />
            <${MenuItem} label="Prune old sessions/backups\u2026" action=${pruneDialog}
                       title="prunes the scratchpad while no project is open" />
            <${MenuItem} label="Reset project state\u2026" disabled=${!!j} action=${resetDialog}
                       title="resets the scratchpad state while no project is open" />
            <${MenuItem} label="Consolidate session\u2026" disabled=${!!j} action=${consolidateSession}
                       title="compact the transcript into a handoff checkpoint (also in the telemetry bar)" />
            <${MenuItem} label="Provider settings\u2026" action=${() => callSettings()} />
            <${MenuItem} label="Clear terminal" action=${clearTerm} />
            <${MenuItem} label="Clear chat" action=${() => PDC.api('/api/clear', {})} />
          <//>
        </div>
        <div id="topRight">
          <button id="benchBtn" title="run a diagnostic benchmark against a configured profile" onClick=${() => openBenchmarks()}>\u25A6 Benchmarks</button>
          <button id="settingsBtn" title="profiles and settings" onClick=${() => callSettings()}>\u2699 Settings</button>
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
        <label class="tb-label" for="profile">Profile</label>
        <${ProfileSelect} busy=${j} peek=${peekOn ? S.peek : null}
                          onChange=${(name) => switchProfile(name)} />
        <span class="chip" id="modelChip"
              title=${(s.model_status || 'click to check the model server') + (s.base_url ? '\n' + s.base_url : '') + (s.context_tokens ? '\nwindow ' + PDC.fmtInt(s.context_tokens) + ' tokens (' + s.context_origin + ')' : '')}
              onClick=${() => { if (s.root && !j) checkModel(); }}>
          <i class=${'dot ' + (j ? 'busy' : s.model_ok === true ? 'ok' : s.model_ok === false ? 'bad' : 'idle')}></i>
          <b>${s.needs_setup ? 'no model selected' : (s.model || (s.root ? '\u2014' : 'no model selected'))}</b>
          ${s.model && !s.needs_setup ? html`<span class="tag ${s.disable_thinking ? '' : 'act'}"
            title=${s.disable_thinking ? 'thinking is disabled for this profile (disable_thinking = true)'
                                       : 'thinking is allowed (disable_thinking = false) \u2014 turn it off in Settings \u2192 Profiles'}>${s.disable_thinking ? 'no-think' : 'thinking'}</span>` : null}
        </span>
        ${!s.root ? html`<span class="chip" id="workspaceChip"
              title="Questions run on the selected profile in a scratchpad until you open or create a project">
          <b>${s.workspace_label || 'Workspace: Scratchpad (no project open)'}</b>
        </span>` : null}
        <${CalibrationBadge} name=${activeProf} />
        ${s.scout_profile ? html`<span class="chip" id="scoutChip"
              title=${'Scout/architect split: lookup steps during gather run on profile \''
                + s.scout_profile + '\' (a fast model); planning, answers and edits stay on \''
                + (s.profile || 'active') + '\'. Click to change scout_profile.'}
              onClick=${() => callSettings({ tab: 'all', search: 'scout_profile' })}>
          <b>Scout: ${s.scout_profile}</b>
        </span>` : null}
        <label class="chip" id="autoRouteChip"
              title=${'Specialist auto-routing (auto_route_specialties): each task runs on the profile'
                + ' whose specialty tags fit it best (model-card tags, or explicit specialties in'
                + ' Settings \u2192 Profiles). Switches happen at task boundaries only, with a clean'
                + ' VRAM handoff when the profiles share one local server.'}>
          <input type="checkbox" checked=${!!s.auto_route_specialties}
                 onChange=${(e) => setAutoRoute(e.target.checked)} />
          <b>\u2726 Auto-route</b>
        </label>
        ${eng.specialist && eng.specialist.profile ? html`<span class="chip" id="specialistChip"
              title=${'Specialist routed: profile \'' + eng.specialist.profile + '\' ('
                + (eng.specialist.model || '?') + ') for specialty \''
                + (eng.specialist.specialty || '?') + '\''}>
          <b>\u2726 ${eng.specialist.profile}</b>
          <span class="tag act">${eng.specialist.specialty}</span>
        </span>` : null}
        <div class="tb-right">
          <button class="chip" id="indexChip" title=${idxTitle} disabled=${!!j || !s.root}
                  onClick=${(e) => { if (s.root && !j) buildIndex(!e.shiftKey); }}>
            <i class=${'dot ' + (indexed ? 'ok' : 'idle')}></i>
            <b>${indexed ? 'indexed' : 'no index'}</b>
            ${indexed ? html`<span class="muted small-text">${PDC.fmtInt(s.index_files)} files \u00B7 ${PDC.fmtInt(s.index_chunks)} units</span>` : null}
            <i class="reidx" aria-hidden="true">\u21BB</i>
          </button>
          ${(s.usage && s.usage.calls) ? html`
            <span class="chip" id="usageChip" title=${'model usage this session (' + Math.round(s.usage.seconds) + 's)'}>
              <b>${s.usage.calls + ' calls \u00B7 ' + PDC.fmtTokens(s.usage.tokens_in) + ' in / ' + PDC.fmtTokens(s.usage.tokens_out) + ' out'
                + (s.usage.cache_hit ? ' \u00B7 cache ' + Math.round(s.usage.cache_hit * 100) + '%' : '')
                + (s.usage.cost ? ' \u00B7 $' + Number(s.usage.cost).toFixed(4) : '')}</b>
            </span>` : null}
          <span class=${'phase ' + phase} id="phaseChip"
                title=${'engine phase' + (j ? ' of the running job' : '')}>${phaseName}</span>
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
  return html`<button class="menuitem" disabled=${!!props.disabled} title=${props.title || ''}
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
      ${flags.map((f, i) => html`<li key=${'f' + i}><code>${f.flag}</code>: ${f.help}</li>`)}
      ${common.length ? html`<li class="muted"><b>Common project flags</b></li>` : null}
      ${common.map((f, i) => html`<li key=${'c' + i}><code>${f.flag}</code>: ${f.help}</li>`)}
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
  if (m.kind === 'handoff') {
    return html`<div class="md">${md(m.text)}</div>`;
  }
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

// ---- Prompt/Spec Architect ----
// One click turns the composer brief into a sectioned spec (objective, files,
// interface contracts, requirements, edge cases, verification). The original
// text is kept in S.ui.previousPrompt so the step can be undone, and repeated
// refines keep the FIRST original -- undo always returns the untouched brief.
function refinePrompt() {
  const el = $('input');
  const text = ((el && el.value) || '').trim();
  if (!text || S.ui.refining) return;
  S.ui.refining = true;
  emit();
  const profile = S.ui.pendingProfile !== null ? S.ui.pendingProfile : (S.status.profile || '');
  PDC.api('/api/spec/refine', { prompt: text, mode: S.mode, profile: profile }).then((j) => {
    S.ui.refining = false;
    if (j._status !== 200 || !j.ok || !j.refined) {
      PDC.toast(j.detail || 'The spec architect did not answer.', true);
      emit();
      return;
    }
    if (S.ui.previousPrompt == null && el) S.ui.previousPrompt = el.value;
    if (el) el.value = j.refined;
    S.drafts[S.mode] = j.refined;
    writeDraft(j.refined);
    // A full spec is many lines: if the composer is compact, grow it just
    // enough to read what came back (never shrink a user's larger setting).
    if ((j.refined.match(/\n/g) || []).length > 8 && composerHeightNow() < 220) {
      setComposerH(260, true);
    }
    PDC.toast(j.contracts && j.contracts.length
      ? 'Refined \u00b7 ' + j.contracts.length + ' interface contract(s) fixed'
      : 'Refined into a full spec.');
    emit();
  }, () => {
    S.ui.refining = false;
    PDC.toast('The GUI server did not answer.', true);
    emit();
  });
}
function undoRefine() {
  if (S.ui.previousPrompt == null) return;
  const el = $('input');
  if (el) el.value = S.ui.previousPrompt;
  S.drafts[S.mode] = S.ui.previousPrompt;
  writeDraft(S.ui.previousPrompt);
  S.ui.previousPrompt = null;
  emit();
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
    if (taRef.current) {
      S.drafts[S.mode] = taRef.current.value;
      writeDraft(taRef.current.value);
    }
    S.mode = k;
    if (k === 'new' && !opt('new', 'project')) {
      S.opts.new.project = s.suggested_new_project || '';
    }
    S.ui.previousPrompt = null;
    try { localStorage.setItem('pedacito_mode', k); } catch (e) {}
    writeDraft(S.drafts[k] || '');
    S.ui.optsOpen = false;
    emit();
  };
  return html`
    <div id="composer"
         ondragenter=${(e) => { e.preventDefault(); e.currentTarget.classList.add('drop'); }}
         ondragover=${(e) => { e.preventDefault(); }}
         ondragleave=${(e) => { e.currentTarget.classList.remove('drop'); }}
         ondrop=${(e) => { e.preventDefault(); e.currentTarget.classList.remove('drop'); if (e.dataTransfer && e.dataTransfer.files.length) attachFiles(e.dataTransfer.files); }}>
      <div class="composer-resizer"
           title="Drag to resize composer (double-click to toggle expand/compact)"
           onMousedown=${dragComposer} onDblclick=${toggleComposer}></div>
      <div id="modeTabs">
        ${MODE_ORDER.map((k) => html`
          <button key=${k} class=${k === S.mode ? 'on' : ''} onClick=${() => setMode(k)}>${MODES[k].label}</button>`)}
      </div>
      <p id="modeHint">
        ${!s.root && (S.mode === 'task' || S.mode === 'auto') ? 'Open a project first. ' : null}
        ${!s.root && S.mode === 'explain'
          ? html`Workspace: <b>Scratchpad (no project open)</b> \u2014 answers run on <b>${s.model || 'the selected profile'}</b>. `
          : null}
        ${!s.root && S.mode === 'new' ? 'The new project becomes your workspace when it is created. ' : null}
        ${needIndex
          ? html`No index yet \u2014 build one to start:
              <button class="small primary" disabled=${!!j} onClick=${() => buildIndex(true)}>Build index</button>
              <button class="small" disabled=${!!j} onClick=${() => buildIndex(false)}>Quick (no summaries)</button>`
          : m.hint}
      </p>
      ${S.mode === 'new' ? html`<div id="newProjectRow">
        <label for="newProjectPath">Target directory for new project</label>
        <input id="newProjectPath" type="text" spellcheck="false" autocomplete="off"
               defaultValue=${opt('new', 'project')}
               placeholder=${s.suggested_new_project || '/path/to/new-project'}
               oninput=${(e) => { S.opts.new.project = e.target.value; saveOpts(); }} />
        <button class="small" onClick=${() => PDC.browse({
          title: 'Choose the parent folder for the new project',
          files: false, pickLabel: 'Use this folder',
          start: S.status.root || '',
          onPick: (p) => {
            const current = String(opt('new', 'project') || '').trim();
            const name = (current.split(/[\\/]/).filter(Boolean).pop()) || 'my-project';
            const target = p.replace(/[\\/]$/, '') + '/' + name;
            S.opts.new.project = target;
            const el = $('newProjectPath');
            if (el) el.value = target;
            saveOpts();
          } })}>Browse\u2026</button>
      </div>` : null}
      <div id="attachRow">
        ${S.attachments.map((a, i) => html`
          <span key=${i} class="attach">\uD83D\uDCCE ${a.name} \u00B7 ${PDC.fmtInt(a.text.length)} chars
            <button title="remove" onClick=${() => { S.attachments.splice(i, 1); emit(); }}>\u2715</button>
          </span>`)}
      </div>
      <textarea id="input" rows="3" spellcheck="true" ref=${taRef}
                placeholder=${isHow ? m.ph : (j ? (steerable ? 'Type a note to steer the model (Enter)\u2026' : 'Working\u2026 press Stop to cancel.') : m.ph)}
                disabled=${!!j && !steerable && !isHow}
                oninput=${(e) => { S.drafts[S.mode] = e.target.value;
                  writeDraft(e.target.value);       // F5 must not lose it
                  const rb = $('refineBtn');
                  if (rb) rb.disabled = S.ui.refining || !e.target.value.trim(); }}
                onkeydown=${(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } }} />
      <div id="optsBox" key=${S.mode} hidden=${!S.ui.optsOpen}>
        <div class="opts-grid">${OPTS[S.mode].filter((o) => o.type !== 'check' && !(S.mode === 'new' && o.k === 'project')).map((o, idx) => html`<${OptField} key=${'g' + idx} o=${o} />`)}</div>
        ${OPTS[S.mode].some((o) => o.type === 'check') ? html`<div class="opts-checks">${OPTS[S.mode].filter((o) => o.type === 'check').map((o, idx) => html`<${OptField} key=${'c' + idx} o=${o} />`)}</div>` : null}
      </div>
      <div id="composerBtns">
        <button id="attachBtn" title="attach a text file (or drop one here)"
                hidden=${isHow}
                disabled=${!!j} onClick=${() => $('filePick').click()}>\uD83D\uDCCE Attach</button>
        <button id="optsBtn" class="ghost" hidden=${!OPTS[S.mode].length}
                onClick=${() => { S.ui.optsOpen = !S.ui.optsOpen; emit(); }}>Options \u25BE</button>
        <button id="refineBtn" class="ghost${S.ui.refining ? ' busy' : ''}" hidden=${isHow}
                title="expand this brief into a full spec with interface contracts (one architect call)"
                disabled=${!!j || S.ui.refining || !(S.drafts[S.mode] || '').trim()}
                onClick=${refinePrompt}>${S.ui.refining ? '\u2726 Refining\u2026' : '\u2726 Refine Prompt'}</button>
        <button id="undoRefineBtn" class="ghost" hidden=${isHow || S.ui.previousPrompt == null}
                title="restore the prompt as it was before refining"
                onClick=${undoRefine}>↶ Undo Refine</button>
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
        ${!pending && (R.unaccepted || []).length ? html`
          <span class="badge-out unaccepted"
                title=${R.unaccepted.map((c) => c + ': ' + ((R.clause_texts || {})[c] || '')).join('\n')}>
            clause(s) left open: ${R.unaccepted.join(', ')}</span>` : null}
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
            ${(f.clauses || []).map((c) => html`
              <span key=${c} class="clause" title=${(R.clause_texts || {})[c] || c}>${c}</span>`)}
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
          ${eng.specialist && eng.specialist.profile ? html`
            <span class="estat" title=${'specialist routed: ' + eng.specialist.profile
              + ' (' + (eng.specialist.model || '?') + ') for specialty \''
              + (eng.specialist.specialty || '?') + '\''}>specialist
              <b>${eng.specialist.profile}</b></span>` : null}
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
        ${S.ui.planGenerating ? html`<span class="badge">new plan\u2026</span>`
          : tasks.length ? html`<span class="badge">${done}/${tasks.length}</span>` : null}
        <span class="muted small-text grow mono" id="planPath">${p.exists ? p.path : ''}</span>
        ${p.exists && !S.ui.planGenerating ? html`
          <button class="small" onClick=${() => { S.ui.planEditing = !S.ui.planEditing; emit(); }}>
            ${S.ui.planEditing ? 'Cancel' : 'Edit'}</button>
          ${S.ui.planEditing ? html`<button class="small primary" onClick=${() => savePlan()}>Save</button>` : null}
          <button class="small primary" disabled=${running || !openN} title="run this plan now"
                  onClick=${() => PDC.runPlan()}>Run</button>` : null}
      </div>
      <div id="planBody" hidden=${S.ui.planEditing}>
        ${S.ui.planGenerating ? html`
          <div class="tip"><b>Generating new plan\u2026</b><br>The previous
          checklist is complete; the planner is replacing it for this goal.</div>`
        : !p.exists ? html`
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
                <option key=${t.id} value=${t.id}>${t.title_custom ? '\u270E ' : ''}${t.title} \u00b7 ${t.message_count} msg${t.message_count === 1 ? '' : 's'}</option>`)}
            </select>
            <button class="ghost small" id="newThread" title="start a new conversation (this one stays saved in the project)"
                    disabled=${!!s.job} onClick=${newThread}>+ New</button>
            <button class="ghost small" id="renameThreadBtn" title=${'rename \u201c' + (s.thread_title || 'this conversation') + '\u201d'}
                    disabled=${!!s.job || !s.active_thread_id} onClick=${renameThread}>Rename</button>
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
      <${TelemetryBar} />
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
    // the legacy theme names (mocha/sage/parchment/clay/nightshade) no
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
