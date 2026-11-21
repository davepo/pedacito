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
let modalSeq = 0;
PDC.modal = (title, body, buttons, opts) => {
  opts = opts || {};
  const overlay = h('div', { class: 'overlay' });
  const close = () => { overlay.remove(); document.removeEventListener('keydown', onKey); if (opts.onClose) opts.onClose(); };
  const onKey = (e) => { if (e.key === 'Escape' && !opts.locked) close(); };
  const foot = h('div', { class: 'modal-foot' });
  (buttons || []).forEach((b) => {
    foot.appendChild(h('button', {
      class: b.primary ? 'primary' : (b.danger ? 'danger' : ''), text: b.label,
      onclick: () => { const keep = b.onclick && b.onclick(close); if (keep !== false && !b.stay) close(); }
    }));
  });
  const titleId = 'modalTitle-' + (++modalSeq);
  const box = h('div', {
    class: 'modal' + (opts.narrow ? ' narrow' : '') + (opts.wide ? ' wide' : ''),
    role: 'dialog',
    'aria-modal': 'true',
    'aria-labelledby': titleId,
  },
    h('div', { class: 'modal-head' }, h('h3', { id: titleId, text: title }),
      opts.locked ? null : h('button', { class: 'ghost', text: '\u2715', onclick: close })),
    h('div', { class: 'modal-body' }, body),
    (buttons && buttons.length) ? foot : null);
  overlay.appendChild(box);
  if (!opts.locked) {
    overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) close(); });
  }
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
  ], { narrow: true, locked: true, onClose: () => { permModal = null; permShownId = ''; } });
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
// The dismissed update version survives reloads, beside the other
// pedacito_* preferences. A strictly newer latest release re-shows the banner;
// the same (or an older cached) latest stays hidden.
const UPDATE_DISMISSED_KEY = 'pedacito_update_dismissed';
function readUpdateDismissed() {
  try { return localStorage.getItem(UPDATE_DISMISSED_KEY) || null; }
  catch (e) { return null; }
}
function writeUpdateDismissed(version) {
  try {
    if (version) localStorage.setItem(UPDATE_DISMISSED_KEY, version);
    else localStorage.removeItem(UPDATE_DISMISSED_KEY);
  } catch (e) {}
}
const UPDATE_STAGE_RANK = { dev: 0, a: 1, alpha: 1, b: 2, beta: 2,
                            c: 3, rc: 3, pre: 3, preview: 3,
                            post: 5, rev: 5, r: 5 };
function updateVersionKey(text) {
  const m = /^\s*v?(\d+(?:\.\d+)*)(?:[-_.]?(a|alpha|b|beta|c|rc|pre|preview|dev|post|rev|r)[-_.]?(\d*))?/i
    .exec(String(text || ''));
  if (!m) return null;
  const rank = UPDATE_STAGE_RANK[(m[2] || '').toLowerCase()];
  return [m[1].split('.').map(Number),
          rank === undefined ? 4 : rank,
          Number(m[3] || 0)];
}
function isNewerVersion(candidate, base) {
  const a = updateVersionKey(candidate), b = updateVersionKey(base);
  if (!a) return false;
  if (!b) return true;
  const n = Math.max(a[0].length, b[0].length);
  for (let i = 0; i < n; i++) {
    const x = a[0][i] || 0, y = b[0][i] || 0;
    if (x !== y) return x > y;
  }
  return a[1] > b[1] || (a[1] === b[1] && a[2] > b[2]);
}
const savedDraft = readDraft();
const savedUpdateDismissed = readUpdateDismissed();
const savedMode = (() => {
  try { return localStorage.getItem('pedacito_mode') || 'task'; }
  catch (e) { return 'task'; }
})();
const savedCreative = (() => {
  try { return localStorage.getItem('pedacito_creative') === '1'; }
  catch (e) { return false; }
})();

const S = {
  status: {},
  profile_availability: {},               // profile -> available|unavailable|offline|unknown
  msgs: [], msgTick: 0,                   // msgTick bumps on any message change
  plan: { path: '', exists: false, text: '', tasks: [] },
  review: null, diff: null, diffRel: '', sel: {}, editing: false,
  editAfterLoad: false,
  ws: { tab: localStorage.getItem('pedacito_ws_tab') || 'files', files: [],
        indexed: [], indexedSet: null, sig: '', filter: '', truncated: false,
        loading: false, backups: [], backupsAt: 0,
        pages: [], page: '', previewRev: 0,
        targets: null, targetsAt: 0, runPage: '', runRev: 0, runCmd: '',
        runCustom: '', runOut: null, runRunning: false },
  cp: { items: [], head_id: '', loaded: false },
  mode: savedMode,
  creative: savedCreative,
  drafts: savedDraft ? { [savedMode]: savedDraft } : {}, attachments: [],
  pinned: [],                              // explicit @file/@symbol picks (session)
  opts: (() => { try { return JSON.parse(localStorage.getItem('pedacito_opts') || '{}'); } catch (e) { return {}; } })(),
  ui: { diffOpen: false, wsCollapsed: false, termFollow: true, optsOpen: false,
        diffMode: localStorage.getItem('pedacito_diff_mode') || 'split',
        view: 'chat', folds: {}, planEditing: false, pendingProfile: null,
        refining: false, previousPrompt: null, switchingProfile: false,
        actionBusy: false, planGenerating: false, planGenFrom: null,
        // `@` autocomplete: the open floating menu, its items and the
        // [start,end) text range the chosen insert replaces.
        suggest: null, suggestSeq: 0,
        // Task-vs-Auto suitability: the debounced /api/mode-check verdict for
        // the current Auto draft, the pre-run confirmation bar, and the draft
        // text the user already acknowledged with "Continue with Auto".
        modeCheck: null, modeConfirm: null, autoAck: null,
        // Auto-update notice (pedacito.update): the dismissed version string
        // (a newer release re-shows the banner), the upgrade output drawer,
        // and manual instructions when no upgrade command can be run.
        updateDismissed: savedUpdateDismissed,
        updateDrawerOpen: false, updateManual: '', updateRunHidden: false },
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
// Each mode carries a `title` tooltip and a one-line `hint` under the mode
// selector so Task (one pass) vs Auto (plan + per-item loop) is unmistakable.
const PLAN_HINT = 'Generates or updates PLAN.md without editing code yet.';
const MODES = {
  task: { label: 'Task/Edit', send: 'Run',
    title: 'Task (ask): single-pass edit \u2014 one turn reads and applies your whole prompt',
    hint: 'Single-pass edit \u2014 reads & applies all parts of your prompt together in 1 turn (best for tweaking 1\u20133 files, HTML/CSS, & bug fixes).',
    ph: 'e.g. make the sync client retry three times and update its callers\u2026' },
  explain: { label: 'Ask', send: 'Ask',
    title: 'Ask: read-only questions about the code \u2014 nothing is written',
    hint: 'Ask how the code works. Read-only \u2014 nothing is written.',
    ph: 'e.g. how does the sync client retry?' },
  auto: { label: 'Auto', send: 'Apply & run',
    title: 'Auto: autonomous multi-step loop \u2014 PLAN.md checklist, a separate turn + test check per item',
    hint: 'Autonomous multi-step loop \u2014 splits big goals into a PLAN.md checklist and runs a separate turn + test check per item (best for large multi-file features). Tick \u201cPlan only\u201d in Options to review the plan first. By using Auto mode you accept the tool trust boundary: it may write files and run commands in this project.',
    ph: 'e.g. add CSV export to the report page, with tests' },
  'new': { label: 'New', send: 'Create',
    title: 'New: scaffold a brand-new project in a target directory',
    hint: 'Scaffolds a brand-new project in a target directory.',
    ph: 'e.g. a CLI todo manager with sqlite storage and a --json flag' },
  how: { label: 'How ?', send: 'Ask',
    title: 'How: help about pedacito itself \u2014 CLI commands, GUI features, settings',
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
  ['Reference a file in chat', 'How do I reference a file in chat?'],
  ['Creative / design mode', 'How do I turn on creative mode for design tasks?']
];
const OPTS = {
  task: [
    { k: 'write', label: 'Writing', type: 'select', def: 'review', opts: [['review', 'Review before writing'], ['apply', 'Write directly (backup first)']] },
    { k: 'plan', label: 'Planning', type: 'select', def: 'auto', opts: [['auto', 'Automatic'], ['always', 'Always plan'], ['never', 'Never']], note: 'Planning runs only with direct writes.' },
    { k: 'steps', label: 'Max lookups (0 = config)', type: 'number', def: '' },
    { k: 'reindex', label: 'Refresh index first', type: 'check', def: false },
    { k: 'hybrid', label: 'Hybrid Architect (a strong model plans, the editor implements)', type: 'check', def: false, span2: true },
    { k: 'architect_provider', label: 'Architect profile (Stage 1 planner)', type: 'select', def: '', opts: [], span2: true }
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
let lastNewRoot = null;                 // last completed New project adopted
let inputEl = null;                       // the composer textarea (uncontrolled)

function refreshState() {
  return PDC.api('/api/state').then(applyStatus, () => {});
}
PDC.refreshState = refreshState;

// A New job finished: the server already adopted the project it created
// (active root, index, session log carrying the creation turn). Follow it
// without a manual browser refresh -- sync the path box, drop the mode
// selector back to Task so follow-up edits are one keystroke away, and
// resync transcript, file tree, index status and sessions from /api/state.
function adoptNewProject(root) {
  if (!root) return;
  lastNewRoot = root;
  const el = document.getElementById('path');
  if (el && document.activeElement !== el) el.value = root;
  if (S.mode === 'new') {
    S.mode = 'task';
    try { localStorage.setItem('pedacito_mode', 'task'); } catch (e) {}
  }
  S.ui.optsOpen = false;
  emit();
  refreshState();
  setTimeout(() => { const ta = $('input'); if (ta) ta.focus(); }, 30);
}

function applyStatus(s) {
  const prev = S.status;
  S.status = s;
  if (prev && prev.root !== s.root && S.pinned.length) S.pinned = [];
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
  // Fallback for a missed SSE job-end: the status carries the last completed
  // New project's root. The first sighting only records it -- a page reload
  // must not yank a mode the user deliberately picked after that build.
  if (s.new_project_root && s.new_project_root !== lastNewRoot) {
    if (lastNewRoot === null) lastNewRoot = s.new_project_root;
    else if (!j) adoptNewProject(s.new_project_root);
  }
  // The plan file is written/ticks WHILE Auto or a planned Task runs. Use the
  // live payload carried by /api/state every poll (and fall back to /api/plan
  // for an older server), so PLAN.md appears as soon as the planner writes it
  // and each [x] lands immediately -- not when the whole run finishes.
  if (s.plan) applyPlanPayload(s.plan);
  else if (j && (j.kind === 'auto' || j.kind === 'task')) loadPlan();
  if (j && j.id !== lastJobId) loadPlan();
  if (!j && lastJobId) {
    collapseDone(); loadPlan(); loadReview(); loadFiles(); loadBackups();
    loadPreview(true); loadRunnerTargets(true); loadCheckpoints(true);
  }
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
  if ((s.root || '') !== (prev.root || '')) {
    loadPreview();                              // project opened/switched
    S.ws.targets = null; S.ws.runOut = null;
    loadRunnerTargets(true);
    S.cp.loaded = false;
    loadCheckpoints(true);
  }
  if ((s.files_sig || '') !== lastFilesSig) {   // index or workspace changed
    lastFilesSig = s.files_sig || '';
    if (S.ws.tab === 'files' && s.root) loadFiles();
    // Live Preview stays live WHILE a run writes: re-list pages and reload
    // the framed page whenever the workspace signature moves.
    if (S.ws.tab === 'preview' && s.root) loadPreview(true);
    if (S.ws.tab === 'runner' && s.root) { loadRunnerTargets(true); S.ws.runRev++; emit(); }
  }
  if (S.ui.pendingProfile !== null && s.profile === S.ui.pendingProfile) S.ui.pendingProfile = null;
  // Colorscheme: config.toml (gui_theme) is the source of truth. On the
  // first poll the server wins only over an untouched browser (a browser-
  // local pick that could not be persisted \u2014 no user config file \u2014 must
  // not be erased on every reload). Afterwards a changed server value is a
  // real Settings/config edit: follow it even while an engine palette is
  // active, so the edit repaints (applyTheme clears the engine tokens). A
  // browser-local pick that could not be persisted leaves the server value
  // unchanged, so it is never overwritten here.
  if (typeof s.gui_theme === 'string' && s.gui_theme && s.gui_theme !== lastServerTheme) {
    let local = 'slate';
    try { local = localStorage.getItem('pedacito_theme') || 'slate'; } catch (err) {}
    const follow = lastServerTheme === null ? local === 'slate' : true;
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

// Live "Thinking & Stream" drawer: while a Task/Auto/New/Refine job runs, the
// throttled `llm_stream` SSE deltas are folded into job.live_reasoning and
// job.live_content. The drawer shows the active phase, token count and speed,
// with two auto-scrolling panes for the reasoning block and the raw output.
function LiveStreamDrawer() {
  const j = job();
  const [open, setOpen] = useState(true);
  const [tab, setTab] = useState('reasoning');
  const paneRef = useRef(null);
  const reasoning = (j && j.live_reasoning) || '';
  const content = (j && j.live_content) || '';
  const text = tab === 'reasoning' ? reasoning : content;
  useEffect(() => {
    const el = paneRef.current;
    if (el && open) el.scrollTop = el.scrollHeight;
  }, [text, tab, open, j && j.id]);
  if (!j) return null;
  const phase = j.stream_phase || '';
  const stopped = !!j.cancelled;
  const label = stopped ? 'Stopped by user'
    : phase === 'reasoning' ? 'Thinking...'
    : phase === 'content' ? 'Writing code...'
    : 'Waiting for the model...';
  const tokens = Number(j.live_tokens || 0);
  const speed = Number(j.live_tok_per_sec || 0);
  return html`
    <div class=${'live-drawer' + (open ? ' open' : '') + (stopped ? ' stopped' : '')}
         id="liveStreamDrawer">
      <div class="live-head">
        <button class="live-toggle" title=${open ? 'collapse live stream' : 'expand live stream'}
                onClick=${() => setOpen(!open)}>${open ? '\u25BE' : '\u25B8'}</button>
        <span class=${'live-phase ' + (stopped ? 'stopped' : phase || 'waiting')}>
          <i class="live-dot"></i>${label}</span>
        <span class="live-stat"><b>${PDC.fmtInt(tokens)}</b> tok</span>
        <span class="live-stat"><b>${speed.toFixed(1)}</b> tok/s</span>
        <span class="grow"></span>
        ${stopped ? null : html`<button class="ghost small live-stop"
          onClick=${() => stop()}>Stop</button>`}
      </div>
      ${open ? html`
        <div class="live-tabs">
          <button class=${'live-tab' + (tab === 'reasoning' ? ' on' : '')}
                  onClick=${() => setTab('reasoning')}>Reasoning ${'<think>'}</button>
          <button class=${'live-tab' + (tab === 'content' ? ' on' : '')}
                  onClick=${() => setTab('content')}>Live Output</button>
          <span class="muted small-text grow">${stopped ? 'Stopped by user' : 'auto-scrolling'}</span>
        </div>
        <pre class=${'live-pane ' + (tab === 'reasoning' ? 'reason-text' : 'output-text')}
             ref=${paneRef}>${text || (tab === 'reasoning'
               ? 'No reasoning yet. Models that expose hidden thinking stream it here.'
               : 'Output starts as the model writes.')}</pre>` : null}
    </div>`;
}

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
    const status = snap.status || {};
    lastJobId = status.job ? status.job.id : 0;
    lastReviewKey = status.review ? status.review.id + ':' + status.review.outcome : '';
    lastFilesSig = status.files_sig || '';
    applyStatus(status);
    if (term) term.reset(snap.term || '');
    connect(snap.seq || 0);
    loadReview();
    loadPlan();
    if (S.status.root && S.ws.tab === 'files') loadFiles(true);
    loadPreview();
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
  on('stream', (e) => {
    // Live token deltas from the model. Append locally so the drawer moves
    // several times a second without a /api/state round trip; the poll
    // snapshot remains authoritative after a reconnect.
    const j = S.status.job;
    if (!j || (e && e.job != null && e.job !== j.id)) return;
    if (e.delta_reasoning) j.live_reasoning = (j.live_reasoning || '') + e.delta_reasoning;
    if (e.delta_content) j.live_content = (j.live_content || '') + e.delta_content;
    if (e.phase) j.stream_phase = e.phase;
    if (e.tokens != null) j.live_tokens = e.tokens;
    if (e.tok_per_sec != null) j.live_tok_per_sec = e.tok_per_sec;
    emit();
  });
  on('security', (e) => {
    if (e && e.message) PDC.toast(e.message);
    refreshState();
  });
  // Auto-update: the check's notice and the upgrade run's start/end resync
  // state; per-line subprocess output appends locally (like 'stream' does).
  on('update', () => refreshState());
  on('update_run', (e) => {
    if (e && e.running) S.ui.updateRunHidden = false;  // a new run re-shows it
    refreshState();
  });
  on('update_output', (e) => {
    const r = S.status.update_run;
    if (r && e && typeof e.line === 'string') {
      r.output = ((r.output || '') + e.line + '\n').slice(-100000);
      emit();
    }
  });
  on('job', (e) => {
    // A finished New job carries the project it created: adopt the workspace
    // (path box, Task mode, session view) right here, then resync everything.
    // The end event fires once per job, so even rebuilding the SAME folder
    // drops the mode back to Task; lastNewRoot only dedupes the poll fallback.
    if (e.state === 'end' && e.kind === 'new' && e.project_root) adoptNewProject(e.project_root);
    refreshState();
  });
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
    else if (j.detail) PDC.toast(j.detail);   // synchronous empty-workspace reply
    refreshState();
  }).catch(() => PDC.toast('Could not reach the GUI server to build the index.', true));
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
function setPostEditVerify(on) {      // topbar toggle -> user config + reload
  const s = S.status || {};
  let cmd = String(s.auto_verify || '').trim();
  if (on && !cmd) cmd = String(s.detected_test_command || '').trim();
  if (on && !cmd) {
    // No runner is evident: leave auto_verify empty (the cheap gates still
    // run) instead of writing a key that enables nothing.
    emit();
    PDC.toast('No test command detected, so auto_verify stays empty. Set test_command or auto_verify in config.toml; the syntax, static and contract gates and scoped impacted tests still run.', true);
    return;
  }
  PDC.api('/api/settings/set', { file: 'user', changes: { auto_verify: on ? cmd : '' } })
    .then((j) => {
      if (!j.ok) { PDC.toast(j.detail || 'could not change post-edit verification', true); return; }
      S.status.auto_verify = on ? cmd : '';
      emit();
      if (!on) {
        PDC.toast('Post-edit verify off: auto_verify cleared. The syntax, static and contract gates still run on every edit.');
      } else {
        PDC.toast('Post-edit verify on: ' + cmd + ' runs after each Auto task; failures feed the repair rounds.');
      }
      refreshState();
    });
}
function setReasoningEffort(level) {  // composer Options bar -> live + user config
  PDC.api('/api/effort', { level: level }).then((j) => {
    if (!j.ok) { PDC.toast(j.detail || 'could not set reasoning effort', true); return; }
    S.status.reasoning_effort = j.reasoning_effort || level;
    emit();
    PDC.toast(level === 'default'
      ? 'Reasoning effort: default (no reasoning_effort field is sent)'
      : 'Reasoning effort: ' + level + ' (sent as reasoning_effort to local-moe and reasoning APIs)');
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
function stop() {
  const j = S.status.job;
  if (j && !j.cancelled) { j.cancelled = true; emit(); }
  PDC.api('/api/stop', {}).then(refreshState);
}

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

// ---- Task-vs-Auto suitability: live banner + pre-run choice ------------------
// While Auto is selected, a debounced POST /api/mode-check classifies the
// draft (the same deterministic helper the CLI/TUI nudge uses). A mismatch
// shows a clickable banner; clicking Run on a mismatched prompt opens the
// "Run as a single-pass Task instead?" confirmation unless the user already
// chose "Continue with Auto" for this exact draft.
let modeCheckTimer = 0;
function scheduleModeCheck(text) {
  clearTimeout(modeCheckTimer);
  const trimmed = (text || '').trim();
  if (S.ui.modeConfirm && S.ui.modeConfirm.text !== trimmed) {
    S.ui.modeConfirm = null;               // editing the draft drops the choice
    emit();
  }
  if (S.mode !== 'auto' || !trimmed) {
    if (S.ui.modeCheck) { S.ui.modeCheck = null; emit(); }
    return;
  }
  modeCheckTimer = setTimeout(() => {
    PDC.api('/api/mode-check', { text: trimmed, mode: 'auto' }).then((j) => {
      if (S.mode !== 'auto' || (S.drafts.auto || '').trim() !== trimmed) return;
      S.ui.modeCheck = { text: trimmed, mismatch: !!j.mismatch,
                         reason: j.reason || '' };
      if (!j.mismatch) S.ui.autoAck = null;
      emit();
    }, () => {});
  }, 350);
}
function carryDraftToTask() {
  const text = S.drafts.auto || '';
  S.drafts.task = text;
  S.drafts.auto = '';
  S.ui.modeCheck = null; S.ui.modeConfirm = null; S.ui.autoAck = null;
  S.mode = 'task';
  try { localStorage.setItem('pedacito_mode', 'task'); } catch (e) {}
  writeDraft(text);
  if (inputEl) inputEl.value = text;
  emit();
}
function switchToTaskMode() { carryDraftToTask(); }
function runAsTaskFromConfirm() {
  const text = S.ui.modeConfirm ? S.ui.modeConfirm.text : (S.drafts.auto || '');
  S.drafts.auto = text;
  carryDraftToTask();
  send();                                  // runs immediately in Task mode
}
function runAutoAnyway() {
  S.ui.autoAck = S.ui.modeConfirm ? S.ui.modeConfirm.text : (S.drafts.auto || '').trim();
  S.ui.modeConfirm = null;
  emit();
  send(true);                              // POST /api/auto with force_auto
}

// ---- Non-blocking secret guard: preflight + 3-way choice --------------------
// Before a request can leave for an external provider, POST /api/security/scan
// returns only masked findings. Localhost/LAN endpoints never ask. The user
// gets exactly the three choices the warning promises: acknowledge, redact,
// or cancel. Acknowledged fingerprints persist server-side.
function askSecurityChoice(scan) {
  return new Promise((resolve) => {
    let settled = false;
    const done = (v) => { if (!settled) { settled = true; resolve(v); } };
    const rows = (scan.findings || []).map((f) => h('li', {
      key: f.fingerprint, class: 'security-row'
    },
      h('code', { text: f.file + ':' + f.line }),
      h('span', { class: 'mask', text: '[' + f.kind + '] ' + f.masked_preview })));
    const body = h('div', {},
      h('p', { class: 'muted', style: 'margin-top:0',
        text: 'These credentials are about to be sent to ' +
          (scan.provider || 'an external provider') + ':' }),
      h('ul', { class: 'security-list' }, rows),
      h('p', { class: 'muted', text: 'Acknowledge & Send As-Is keeps the request unchanged and will not ask again for these keys. Redact Secrets & Send replaces each with <REDACTED_SECRET>.' }));
    PDC.modal('Secrets detected in this request', body, [
      { label: 'Cancel', onclick: () => done(null) },
      { label: 'Redact Secrets & Send', primary: true, onclick: () => done('redact') },
      { label: 'Acknowledge & Send As-Is', onclick: () => done('ack') }
    ], { wide: true, onClose: () => done(null) });
  });
}

async function securityPreflight(body) {
  let scan = null;
  try {
    scan = await PDC.api('/api/security/scan', {
      text: body.text || '', mode: body.mode || '',
      attachments: (body.attachments || []).map((a) => ({ name: a.name, text: a.text }))
    });
  } catch (e) { return { send: true, choice: '' }; }
  if (!scan || !scan.needs_confirmation) return { send: true, choice: '' };
  const choice = await askSecurityChoice(scan);
  if (!choice) return { send: false, choice: '' };
  if (choice === 'ack') {
    // Persist the prompt's fingerprints immediately: an acknowledged key
    // must not be raised again, even if this run is interrupted.
    const fps = (scan.findings || []).map((f) => f.fingerprint);
    if (fps.length) PDC.api('/api/security/ack', { fingerprints: fps }).catch(() => {});
  }
  return { send: true, choice: choice };
}

async function send(forceAuto) {
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
  // Explicit context pins from the `@` picker ride as a request-level list so
  // they survive edits to the prompt text. Prompt @mentions are parsed again
  // server-side; duplicates collapse there.
  if (S.pinned.length && (S.mode === 'task' || S.mode === 'auto'
                          || S.mode === 'explain')) {
    body.pinned = S.pinned.map((p) => p.insert || p.label)
      .filter(Boolean).map((s) => String(s).replace(/^@/, ''));
  }
  // The ✨ toggle forces Creative mode for this run; when off, the agent
  // still auto-detects design/UI tasks (creative_auto).
  if (S.mode === 'task' || S.mode === 'auto' || S.mode === 'new') body.creative = !!S.creative;
  // Split-stage hybrid: the architect's model rides along from the selected
  // profile; with the toggle off both architect fields clear so a previous
  // run's planner never leaks into this one.
  if (S.mode === 'task') {
    const ap = body.hybrid ? (body.architect_provider || s.architect_provider || '') : '';
    if (body.hybrid && !ap) { PDC.toast('Pick an Architect profile in Options first.', true); return; }
    body.architect_provider = ap;
    body.architect_model = ap ? ((s.profile_models || {})[ap] || '') : '';
  }
  // Send the topbar's model selection explicitly so a no-project New/Ask uses
  // the profile the user just picked, even before /api/state catches up.
  body.profile = S.ui.pendingProfile !== null ? S.ui.pendingProfile : (s.profile || '');
  if (S.mode === 'auto') {
    body.plan_only = !!body.plan_only;
    body.apply = !body.plan_only;      // Auto applies unless Plan only is ticked
    // Pre-run choice: a single-pass-looking prompt gets the Task-vs-Auto
    // confirmation before the multi-step loop starts (skip it when the user
    // already chose "Continue with Auto", forced Auto, or just wants a plan).
    if (!body.plan_only && !forceAuto && S.ui.autoAck !== text) {
      let mc = S.ui.modeCheck;
      if (!mc || mc.text !== text) {
        try {
          const j = await PDC.api('/api/mode-check', { text: text, mode: 'auto' });
          mc = { text: text, mismatch: !!j.mismatch, reason: j.reason || '' };
          S.ui.modeCheck = mc;
        } catch (e) { mc = null; }
      }
      if (mc && mc.mismatch) {
        S.ui.modeConfirm = { text: text, reason: mc.reason };
        emit();
        return;
      }
    }
    if (forceAuto) body.force_auto = true;
  }
  // Outbound secret preflight (external providers only): acknowledge, redact
  // or cancel before any bytes leave the machine. Local endpoints skip it.
  const pre = await securityPreflight(body);
  if (!pre.send) return;
  if (pre.choice === 'ack') body.acknowledge_secrets = true;
  else if (pre.choice === 'redact') body.redact_secrets = true;
  // "Run Auto Anyway" addresses the Auto alias directly with force_auto:true;
  // every other run goes through the shared /api/run starter.
  const starter = (S.mode === 'auto' && forceAuto)
    ? PDC.api('/api/auto', body).then((j) => {
        if (!j.ok) PDC.toast(j.detail || 'could not start', true);
        refreshState();
        return j;
      })
    : PDC.startRun(body);
  starter.then((r) => {
    if (r.ok) {
      // A real New build switches the workspace on the server IMMEDIATELY
      // (before scaffolding): follow it now -- path box + state resync -- so
      // the build's progress streams into the new project's session, and
      // reset the one-shot Complex checkbox (it never carries to a next run).
      if (S.mode === 'new' && body.build !== 'preview') {
        const el = document.getElementById('path');
        if (el && document.activeElement !== el) el.value = body.project;
        S.opts.new = S.opts.new || {};
        S.opts.new.complex = false;
        saveOpts();
        refreshState();
      }
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
// ---- live preview (project HTML framed read-only from /preview/) ----
function previewSrc(page, rev) {
  return '/preview/' + String(page).split('/').map(encodeURIComponent).join('/')
    + '?v=' + rev;                       // cache-buster: refresh must reload
}
function loadPreview(force) {
  if (!S.status.root) {
    if (S.ws.pages.length || S.ws.page) { S.ws.pages = []; S.ws.page = ''; emit(); }
    return Promise.resolve();
  }
  return PDC.api('/api/preview').then((j) => {
    if (j._status !== 200 || !Array.isArray(j.files)) return;
    S.ws.pages = j.files;
    if (!j.files.length) { S.ws.page = ''; emit(); return; }
    let page = S.ws.page;
    if (!page || j.files.indexOf(page) < 0) {
      let saved = '';
      try { saved = localStorage.getItem('pedacito_preview_page') || ''; } catch (e) {}
      page = j.files.indexOf(saved) >= 0 ? saved
        : (j.files.indexOf('index.html') >= 0 ? 'index.html' : j.files[0]);
    }
    const changed = page !== S.ws.page;
    S.ws.page = page;
    if (force || changed) S.ws.previewRev++;
    emit();
  }, () => {});
}
function setPreviewPage(p) {
  S.ws.page = p;
  try { localStorage.setItem('pedacito_preview_page', p); } catch (e) {}
  emit();
}
function openPreviewTab() {
  if (S.ws.page) window.open(previewSrc(S.ws.page, S.ws.previewRev), '_blank', 'noopener,noreferrer');
}

// ---- live runner: detected commands + a static web preview ----
function runnerSrc(page, rev) {
  return '/api/runner/preview/' + String(page).split('/').map(encodeURIComponent).join('/')
    + '?v=' + rev;
}
function loadRunnerTargets(force) {
  if (!S.status.root) {
    S.ws.targets = null; S.ws.runOut = null;
    if (S.ws.runPage) { S.ws.runPage = ''; emit(); }
    return Promise.resolve();
  }
  if (!force && S.ws.targets && Date.now() - S.ws.targetsAt < 4000) return Promise.resolve();
  return PDC.api('/api/runner/targets').then((j) => {
    if (j._status !== 200 || !j.ok) return;
    S.ws.targets = j;
    S.ws.targetsAt = Date.now();
    const pages = (j.web || []).map((p) => p.path);
    if (!S.ws.runPage || pages.indexOf(S.ws.runPage) < 0) {
      S.ws.runPage = pages.indexOf('index.html') >= 0 ? 'index.html' : (pages[0] || '');
      S.ws.runRev++;
    }
    const cmds = j.commands || [];
    if (!S.ws.runCmd || !cmds.some((c) => c.id === S.ws.runCmd)) {
      S.ws.runCmd = cmds.length ? cmds[0].id : '';
    }
    emit();
  }, () => {});
}
function setRunPage(p) { S.ws.runPage = p; S.ws.runRev++; emit(); }
function reloadRunnerPreview() { S.ws.runRev++; emit(); }
function openRunnerPreview() {
  if (S.ws.runPage) window.open(runnerSrc(S.ws.runPage, S.ws.runRev), '_blank', 'noopener,noreferrer');
}
function selectedRunnerCommand() {
  const t = S.ws.targets || {};
  return (t.commands || []).find((c) => c.id === S.ws.runCmd) || null;
}
function runProjectCommand() {
  if (!S.status.root || S.ws.runRunning) return;
  const custom = String(S.ws.runCustom || '').trim();
  const target = selectedRunnerCommand();
  const cmd = custom || (target && target.command) || '';
  if (!cmd) { PDC.toast('Pick a detected command or type one first.', true); return; }
  S.ws.runRunning = true; S.ws.runOut = null; emit();
  PDC.api('/api/runner/exec', { command: cmd }).then((j) => {
    S.ws.runRunning = false;
    S.ws.runOut = j;
    emit();
    if (j.ok) PDC.toast('Run finished: ' + (j.command || cmd));
  }, () => {
    S.ws.runRunning = false;
    PDC.toast('Runner request failed.', true);
    emit();
  });
}
function setPromptDraft(text, mode) {
  if (mode) S.mode = mode;
  S.drafts[S.mode] = text;
  writeDraft(text);
  try { localStorage.setItem('pedacito_mode', S.mode); } catch (e) {}
  emit();
  setTimeout(() => {
    if (inputEl && S.mode !== 'how') {
      inputEl.value = text;
      inputEl.focus();
      inputEl.selectionStart = inputEl.selectionEnd = text.length;
    }
  }, 30);
}
function fixRunnerCrash() {
  const out = S.ws.runOut;
  if (!out || out.ok) return;
  const parts = [
    'Fix this crash.', '',
    'Command: ' + (out.command || '(unknown command)'),
    'Exit code: ' + (out.exit_code == null ? '?' : out.exit_code),
  ];
  if (out.timed_out) parts.push('The command timed out.');
  if (out.traceback_summary) {
    parts.push('', 'Traceback:', String(out.traceback_summary));
  } else if (out.stderr) {
    parts.push('', 'stderr:', String(out.stderr).slice(-6000));
  } else if (out.stdout) {
    parts.push('', 'stdout:', String(out.stdout).slice(-4000));
  }
  parts.push('', 'Reproduce the failing command, find the cause, and fix it. '
    + 'Re-run the command afterwards and report the result.');
  setPromptDraft(parts.join('\n'), 'task');
  PDC.toast('Crash context put in the Task composer.');
}

// ---- time-travel checkpoints ----
function cpTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return isNaN(d.getTime()) ? String(ts) : d.toLocaleString();
}
function fetchCheckpoints() {
  if (!S.status.root) {
    S.cp = { items: [], head_id: '', loaded: true };
    emit();
    return Promise.resolve({ ok: true, checkpoints: [], head_id: '' });
  }
  return PDC.api('/api/checkpoints').then((j) => {
    if (j._status === 200 && j.ok) {
      S.cp.items = j.checkpoints || [];
      S.cp.head_id = j.head_id || '';
      S.cp.loaded = true;
      emit();
    }
    return j;
  }, () => ({ ok: false, checkpoints: [], head_id: '' }));
}
function loadCheckpoints(force) {
  if (force || !S.cp.loaded) return fetchCheckpoints();
  return Promise.resolve({ ok: true, checkpoints: S.cp.items, head_id: S.cp.head_id });
}
function restoreCheckpoint(id, files) {
  return PDC.api('/api/checkpoints/restore', { id: id, files: files || null })
    .then((j) => {
      if (j.ok) {
        PDC.toast(j.detail || 'Restored.');
        refreshState(); loadFiles(true); loadRunnerTargets(true);
        loadCheckpoints(true);
      } else PDC.toast(j.detail || 'Restore failed.', true);
      return j;
    }, () => ({ ok: false, detail: 'Restore failed.' }));
}
function branchFromCheckpoint(cp) {
  PDC.confirm('Branch from ' + cp.id + '?\n\nThe workspace is restored to '
    + 'that checkpoint and the next prompt branches a new run from there. '
    + 'Your current files are saved in a safety checkpoint first.', 'Branch', true)
    .then((ok) => {
      if (!ok) return;
      restoreCheckpoint(cp.id, null).then((j) => {
        if (!j.ok) return;
        setPromptDraft('Branch from checkpoint ' + cp.id + ' (' + (cp.label || '') + '): ', 'task');
        PDC.toast('Workspace restored \u2014 type the new prompt.');
      });
    });
}
function openCheckpointsDialog() {
  const body = h('div', { class: 'cp-dialog' });
  let selected = S.cp.head_id || '';
  let against = 'parent';
  let diff = null;

  const loadDiff = () => {
    if (!selected) { diff = null; renderBody(); return; }
    diff = null;
    renderBody();
    PDC.api('/api/checkpoints/diff?id=' + encodeURIComponent(selected)
      + '&against=' + against).then((j) => {
      if (j._status !== 200) {
        diff = null;
        renderBody();
        return;
      }
      diff = j;
      renderBody();
    });
  };
  function pick(cp) {
    selected = cp.id;
    loadDiff();
  }
  function renderBody() {
    clear(body);
    const cp = S.cp.items.find((c) => c.id === selected) || null;
    const name = h('input', { type: 'text', class: 'grow cp-name',
      placeholder: 'name this checkpoint (e.g. before refactor)' });
    body.appendChild(h('div', { class: 'cp-tools' },
      name,
      h('button', { class: 'small primary', text: '+ Checkpoint', onclick: () => {
        const label = String(name.value || '').trim();
        if (!label) { PDC.toast('Name the checkpoint first.', true); return; }
        PDC.api('/api/checkpoints/create', { label: label }).then((j) => {
          PDC.toast(j.ok ? 'Checkpoint ' + j.checkpoint.id + ' recorded.'
            : (j.detail || 'Could not create the checkpoint.'), !j.ok);
          if (j.ok) { selected = j.checkpoint.id; fetchCheckpoints().then(loadDiff); }
        });
      } }),
      h('button', { class: 'small', text: '\u21BB', title: 'reload the tree',
        onclick: () => fetchCheckpoints().then(loadDiff) })));
    if (!S.cp.items.length) {
      body.appendChild(h('div', { class: 'tip', text: 'No checkpoints yet. '
        + 'One is recorded automatically after every Task, Auto step and New '
        + 'scaffold, or make one above.' }));
      return;
    }
    if (!selected || !S.cp.items.some((c) => c.id === selected)) {
      selected = S.cp.head_id || S.cp.items[S.cp.items.length - 1].id;
    }
    const list = h('div', { class: 'cp-list' });
    S.cp.items.forEach((item) => {
      const node = h('button', {
        class: 'cp-node' + (item.id === selected ? ' on' : ''),
        title: item.id,
        onclick: () => pick(item),
      });
      node.appendChild(h('span', { class: 'cp-mark', text: item.head ? '\u25CF' : (item.parent_id ? '\u25CB' : '\u25C7') }));
      node.appendChild(h('span', { class: 'cp-id mono', text: item.id }));
      node.appendChild(h('span', { class: 'tag', text: item.mode || 'task' }));
      node.appendChild(h('span', { class: 'cp-label ellip', text: item.label || '(untitled)' }));
      node.appendChild(h('span', { class: 'muted small-text cp-meta',
        text: (item.files_changed || []).length + ' file(s) \u00B7 ' + cpTime(item.timestamp) }));
      if (item.parent_id) {
        node.appendChild(h('span', { class: 'cp-parent muted small-text',
          text: '\u21B3 ' + item.parent_id }));
      }
      list.appendChild(node);
    });
    body.appendChild(list);
    if (!cp) return;
    body.appendChild(h('div', { class: 'cp-actions' },
      h('button', { class: 'small', text: against === 'parent'
          ? 'Base: parent \u21C4' : 'Base: workspace \u21C4',
        title: 'toggle the diff base',
        onclick: () => { against = against === 'parent' ? 'workspace' : 'parent'; diff = null; loadDiff(); } }),
      h('button', { class: 'small danger', text: '\u21BA Restore Entire Workspace',
        onclick: () => {
          PDC.confirm('Restore the whole workspace to ' + cp.id + '?\n\n'
            + 'Dirty changes are saved in a safety checkpoint first. Files '
            + 'created after that checkpoint are removed.', 'Restore', true)
            .then((ok) => { if (ok) restoreCheckpoint(cp.id, null).then(loadDiff); });
        } }),
      h('button', { class: 'small primary', text: '\u2442 Branch Prompt From Here',
        onclick: () => branchFromCheckpoint(cp) })));
    if (!diff) {
      body.appendChild(h('div', { class: 'tip', text: 'Loading diff\u2026' }));
      return;
    }
    const files = h('div', { class: 'cp-files' });
    const pre = h('pre', { class: 'cp-diff' });
    const showFile = (f) => {
      Array.from(files.children).forEach((row) => row.classList.remove('on'));
      if (f && f.row) f.row.classList.add('on');
      pre.textContent = f ? (f.diff || '(binary file or no textual diff)')
        : (diff.diff || 'No differences against ' + against + '.');
    };
    files.appendChild(h('button', { class: 'cp-file-all small', text: 'All files ('
      + (diff.changed_files || []).length + ')', onclick: () => showFile(null) }));
    (diff.files || []).forEach((f) => {
      const row = h('div', { class: 'cp-file' + (f.status === 'deleted' ? ' gone' : ''),
        title: f.path, onclick: () => showFile(f) });
      f.row = row;
      row.appendChild(h('span', { class: 'cp-status ' + f.status, text: f.status }));
      row.appendChild(h('span', { class: 'cp-path mono ellip', text: f.path }));
      row.appendChild(h('span', { class: 'muted small-text',
        text: '+' + (f.added || 0) + ' / -' + (f.removed || 0) }));
      row.appendChild(h('button', { class: 'small', text: 'Restore Selected File',
        onclick: (e) => {
          e.stopPropagation();
          PDC.confirm('Restore only \u201c' + f.path + '\u201d from ' + cp.id + '?',
            'Restore file', true).then((ok) => {
              if (ok) restoreCheckpoint(cp.id, [f.path]).then(loadDiff);
            });
        } }));
      files.appendChild(row);
    });
    pre.textContent = diff.diff || 'No differences against ' + against + '.';
    body.appendChild(files);
    body.appendChild(pre);
  }

  S.cp.head_id = S.cp.head_id || '';
  fetchCheckpoints().then(() => {
    selected = selected || S.cp.head_id
      || (S.cp.items.length ? S.cp.items[S.cp.items.length - 1].id : '');
    loadDiff();
  });
  PDC.modal('Checkpoints / Undo', body,
    [{ label: 'Close' }], { wide: true });
}

function setWsTab(t) {
  if (S.ws.tab === t && !S.ui.wsCollapsed) {      // the tab already showing:
    if (t === 'files') loadFiles(true);           // click again = rescan
    else if (t === 'backups') loadBackups(true);
    else if (t === 'preview') loadPreview(true);  // click again = reload page
    else if (t === 'runner') { loadRunnerTargets(true); S.ws.runRev++; emit(); }
    return;
  }
  S.ws.tab = t;
  S.ui.wsCollapsed = false;
  try { localStorage.setItem('pedacito_ws_tab', t); } catch (e) {}
  if (t === 'files') loadFiles();
  if (t === 'backups') loadBackups();
  if (t === 'preview') loadPreview();
  if (t === 'runner') loadRunnerTargets(true);
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
    max_tasks: a.max_tasks || '', max_minutes: a.max_minutes || '', keep_going: !!a.keep_going, auto_install: !!a.auto_install, diagnostics: !!a.diagnostics,
    creative: !!S.creative });
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

// ------------------------------------------------------------------- arena ---
// The 2-model sandbox arena: one prompt, two contestants, each in an isolated
// copy of the workspace. The scorecard compares their diffs side by side; an
// Apply button adopts the winner between two checkpoints (undoable).
function openArena() {
  const st = { jobId: 0, timer: 0, running: false, initialId: '', state: null };
  const selA = h('select', { id: 'arenaA' });
  const selB = h('select', { id: 'arenaB' });
  const promptIn = h('textarea', { rows: 3, spellcheck: false,
    placeholder: 'the task both contestants should run, e.g. add a --json flag to the CLI' });
  const statusLine = h('div', { class: 'bench-status' });
  const resultBox = h('div', { class: 'bench-result' });
  const runBtn = h('button', { class: 'primary', text: 'Run Arena' });

  const fillProfiles = () => {
    const s = S.status || {};
    const names = s.profiles || [];
    const models = s.profile_models || {};
    [selA, selB].forEach((sel, i) => {
      clear(sel);
      sel.appendChild(h('option', { value: '',
        text: '(active' + (s.profile ? ': ' + s.profile : '') + ')' }));
      names.forEach((n) => sel.appendChild(h('option', { value: n,
        text: n + (models[n] ? ' \u00b7 ' + models[n] : '') })));
      if (names.length > i) sel.value = names[i];
    });
  };
  const spec = (sel) => sel.value || (S.status.profile || S.status.model || '');
  const metric = (grid, label, value) => {
    grid.appendChild(h('div', { class: 'bench-metric' },
      h('span', { class: 'muted small-text', text: label }),
      h('b', { text: value })));
  };
  const applyWinner = (side) => {
    const state = st.state;
    if (!state || state.winner) return;
    PDC.api('/api/arena/apply', { arena_id: state.arena_id, winner: side }).then((j) => {
      if (!j.ok) { PDC.toast(j.detail || 'could not apply the winner', true); return; }
      state.winner = side;
      PDC.toast('Arena winner applied: ' + (j.model || side.toUpperCase())
        + ' (' + (j.files_changed || []).length + ' file(s)). Undo via Checkpoints.');
      scorecard(state);
      PDC.refreshState();
    });
  };
  const sideBox = (state, side) => {
    const rec = (state.contestants || {})[side] || {};
    const ok = !!rec.ok;
    const won = state.winner === side;
    const box = h('div', { class: 'bench-score arena-side ' + (ok ? 'ok' : 'bad') + (won ? ' won' : '') });
    box.appendChild(h('div', { class: 'bench-score-head' },
      h('span', { class: 'bench-big', text: side.toUpperCase() }),
      h('b', { text: (rec.provider ? rec.provider + ' / ' : '') + (rec.model || '?') }),
      h('span', { class: 'tag ' + (ok ? 'ok' : ''), text: ok ? 'EDITS LANDED' : 'NO EDITS' }),
      won ? h('span', { class: 'tag ok', text: 'WINNER APPLIED' }) : null));
    const grid = h('div', { class: 'bench-grid' });
    metric(grid, 'Files changed', String((rec.files_changed || []).length));
    metric(grid, 'Duration', Number(rec.elapsed_sec || 0).toFixed(1) + 's');
    metric(grid, 'Speed', Number(rec.tok_per_sec || 0).toFixed(1) + ' tok/s');
    metric(grid, 'Tokens', PDC.fmtInt(rec.total_tokens || 0));
    box.appendChild(grid);
    if (rec.summary) {
      box.appendChild(h('div', { class: 'muted small-text arena-summary', text: rec.summary }));
    }
    const files = rec.files_changed || [];
    if (files.length) {
      box.appendChild(h('div', { class: 'bench-label', text: 'Changed files' }));
      box.appendChild(h('div', { class: 'mono small-text arena-files', text: files.join('\n') }));
    }
    box.appendChild(h('div', { class: 'bench-label', text: 'Diff' }));
    const pre = h('pre', { class: 'cp-diff arena-diff' });
    pre.textContent = rec.diff || '(no file changes)';
    box.appendChild(pre);
    if (!state.winner) {
      box.appendChild(h('div', { class: 'bench-actions' },
        h('button', { class: 'primary', text: 'Apply ' + side.toUpperCase() + '\u2019s changes',
          onclick: () => applyWinner(side) })));
    }
    return box;
  };
  const scorecard = (state) => {
    clear(resultBox);
    st.state = state;
    if (!state) return;
    resultBox.appendChild(h('div', { class: 'muted small-text',
      text: 'Arena ' + state.arena_id + ' \u00b7 ' + (state.prompt || '') }));
    resultBox.appendChild(h('div', { class: 'arena-grid' },
      sideBox(state, 'a'), sideBox(state, 'b')));
  };
  const finish = (state) => {
    st.running = false;
    st.jobId = 0;
    runBtn.disabled = false;
    runBtn.textContent = 'Run Arena';
    statusLine.textContent = 'Finished. Compare the two runs, then apply a winner.';
    scorecard(state);
  };
  const poll = () => {
    clearTimeout(st.timer);
    PDC.api('/api/arena/last').then((j) => {
      const state = j.ok ? j.arena : null;
      if (state && state.arena_id && state.arena_id !== st.initialId) { finish(state); return; }
      if (!st.jobId) return;
      const job = S.status.job;
      if (!job || job.id !== st.jobId) {
        // The job slot freed: one last read covers the save-then-end race.
        if (state && state.arena_id && state.arena_id !== st.initialId) { finish(state); return; }
        st.running = false;
        runBtn.disabled = false;
        runBtn.textContent = 'Run Arena';
        statusLine.textContent = 'The arena job ended without a result; see Activity/Chat for the error.';
        return;
      }
      statusLine.textContent = 'Running\u2026 both contestants work in isolated sandboxes. Watch Activity for live progress.';
      st.timer = setTimeout(poll, 1500);
    }).catch(() => { st.timer = setTimeout(poll, 2500); });
  };
  const run = () => {
    if (st.running) return;
    const text = promptIn.value.trim();
    if (!text) { PDC.toast('Describe the task both contestants should run.', true); return; }
    const a = spec(selA), b = spec(selB);
    if (!a || !b) { PDC.toast('Pick two contestants.', true); return; }
    if (S.status.job) { PDC.toast('Another job is running. Wait for it or press Stop.', true); return; }
    st.running = true;
    st.initialId = (st.state || {}).arena_id || '';
    runBtn.disabled = true;
    runBtn.textContent = 'Running\u2026';
    clear(resultBox);
    statusLine.textContent = 'Starting\u2026';
    PDC.api('/api/arena/run', { text: text, model_a: a, model_b: b }).then((j) => {
      if (!j.ok) {
        st.running = false; runBtn.disabled = false; runBtn.textContent = 'Run Arena';
        statusLine.textContent = j.detail || 'Could not start the arena.';
        PDC.toast(j.detail || 'arena did not start', true);
        return;
      }
      st.jobId = j.job;
      statusLine.textContent = 'Running\u2026 watch Activity for live progress.';
      poll();
    }).catch((e) => {
      st.running = false; runBtn.disabled = false; runBtn.textContent = 'Run Arena';
      statusLine.textContent = 'Could not start the arena: ' + e;
    });
  };
  runBtn.addEventListener('click', run);

  const body = h('div', { class: 'bench-wrap' },
    h('div', { class: 'bench-intro' },
      h('h4', { text: 'Run one prompt on two models, side by side' }),
      h('div', { class: 'muted small-text', text: 'Each contestant works in an isolated sandbox copy of the workspace \u2014 the real project stays untouched until you apply a winner (checkpointed, undoable).' })),
    h('div', { class: 'bench-form' },
      h('label', { class: 'bench-field' }, h('span', { text: 'Contestant A' }), selA),
      h('label', { class: 'bench-field' }, h('span', { text: 'Contestant B' }), selB)),
    h('label', { class: 'bench-field' }, h('span', { text: 'Task prompt' }), promptIn),
    h('div', { class: 'row gap bench-runrow' }, runBtn, statusLine),
    resultBox);
  PDC.modal('Model Arena (A/B)', body, [{ label: 'Close' }],
    { wide: true, onClose: () => clearTimeout(st.timer) });
  fillProfiles();
  PDC.api('/api/arena/last').then((j) => {
    if (j.ok && j.arena) {
      st.initialId = j.arena.arena_id || '';
      scorecard(j.arena);
      statusLine.textContent = 'Last arena shown. Run a new one above.';
    }
  }).catch(() => {});
}
PDC.openArena = openArena;

// The nine CSS-native gui_theme schemes (config.py's closed enum). Each has
// a full crafted palette block in app.css -- ansi colors, terminal, code and
// diff surfaces included -- and paints via the data-theme attribute. The
// engine's inline ten-token derivation must never overpaint them.
const CSS_SCHEMES = ['slate', 'light', 'daylight', 'nord', 'midnight',
  'jazz_solo', 'trapper_keeper', 'memphis_arcade', 'windbreaker'];
const CSS_SCHEME_META = {
  slate: { label: 'Slate \u00b7 GitHub Dark', bg: '#0d1117', accent: '#58a6ff' },
  light: { label: 'Light \u00b7 GitHub Light', bg: '#f6f8fa', accent: '#0969da' },
  daylight: { label: 'Daylight \u00b7 warm paper', bg: '#fcfaf2', accent: '#b54900' },
  nord: { label: 'Nord \u00b7 modern dark', bg: '#181825', accent: '#89b4fa' },
  midnight: { label: 'Midnight \u00b7 high contrast', bg: '#05070a', accent: '#38bdf8' },
  jazz_solo: { label: 'Jazz Cup \u00b7 1991 light', bg: '#f4f1ea', accent: '#009688' },
  memphis_arcade: { label: 'Memphis Arcade \u00b7 dark', bg: '#121324', accent: '#00e5ff' },
  trapper_keeper: { label: 'Trapper Keeper \u00b7 light', bg: '#eef2f9', accent: '#2541b2' },
  windbreaker: { label: 'Windbreaker \u00b7 1992 dark', bg: '#191326', accent: '#00d2b8' },
};
function isCssScheme(name) { return CSS_SCHEMES.indexOf(name) >= 0; }

// The GUI colorscheme. The browser always gets it at once (localStorage, so
// theme.js can apply it before first paint); `userPick` additionally persists
// it as gui_theme in the user config through /api/settings/theme — purely
// cosmetic, so the project is never reloaded for it. If the write fails (no
// user config file yet, server gone), the choice stays browser-local.
let lastServerTheme = null;
function applyTheme(t, userPick) {
  if (!t) return;
  // A CSS-native scheme is taking over: drop any engine inline tokens and
  // cache so the crafted app.css block paints -- and survives the next reload.
  clearThemeTokens();
  if (t === 'slate') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', t);
  document.documentElement.dataset.themeName = t;
  try { localStorage.setItem('pedacito_theme', t); } catch (err) {}
  const btn = $('themeBtn');
  if (btn) btn.title = 'theme: ' + t + ' \u2014 open the Theme Manager';
  if (!userPick) return;
  PDC.api('/api/settings/theme', { gui_theme: t }).then((j) => {
    if (j && j.ok) {
      lastServerTheme = t;
      if (!j.persisted) PDC.toast('Theme kept in this browser: create a user config to store it in config.toml.');
    } else PDC.toast('Theme kept in this browser (config not written): ' + ((j && j.detail) || 'server error'), true);
  }).catch(() => PDC.toast('Theme kept in this browser (server unreachable).', true));
}
// ----------------------------------------------------- unified theme engine ---
// The ten-token schema in pedacito/themes.py drives every CSS variable below.
// Presets and user palettes arrive from /api/themes; the modal edits a local
// draft and applies it live before anything is saved.
const THEME_FIELDS = [
  ['bg', 'Background'], ['surface', 'Surface'],
  ['surface_alt', 'Surface alt'], ['text', 'Text'], ['muted', 'Muted text'],
  ['accent', 'Accent'], ['border', 'Border'], ['success', 'Success'],
  ['warning', 'Warning'], ['danger', 'Danger'],
];
const THEME_TOKEN_KEYS = THEME_FIELDS.map((f) => f[0]);

function themeContrast(hex) {
  const raw = String(hex || '').replace('#', '');
  if (raw.length !== 6) return '#ffffff';
  const r = parseInt(raw.slice(0, 2), 16);
  const g = parseInt(raw.slice(2, 4), 16);
  const b = parseInt(raw.slice(4, 6), 16);
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 > 0.55
    ? '#111111' : '#ffffff';
}

function themeCssProperties(t) {
  const mix = (a, b, pct) => 'color-mix(in srgb, ' + a + ' ' + pct + '%, ' + b + ')';
  return {
    '--bg': t.bg, '--surface': t.surface, '--surface-alt': t.surface_alt,
    '--text': t.text, '--muted': t.muted, '--accent': t.accent,
    '--border': t.border, '--success': t.success, '--warning': t.warning,
    '--danger': t.danger,
    '--panel': t.surface, '--panel2': t.surface_alt, '--panel3': t.surface_alt,
    '--line': t.border, '--line2': t.border, '--soft': t.surface_alt,
    '--ok': t.success, '--err': t.danger, '--warn': t.warning,
    '--accent-soft': mix(t.accent, t.surface, 22),
    '--accent-fg': themeContrast(t.accent),
    '--danger-fg': themeContrast(t.danger),
    '--add-bg': t.success, '--del-bg': t.danger,
    '--term-bg': t.bg, '--term-text': t.text, '--code-bg': t.surface,
    '--ansi-0': t.muted, '--ansi-1': t.danger, '--ansi-2': t.success,
    '--ansi-3': t.warning, '--ansi-4': t.accent, '--ansi-5': t.accent,
    '--ansi-6': t.accent, '--ansi-7': t.text,
  };
}

function applyThemeTokens(tokens, name) {
  if (!tokens) return;
  const css = themeCssProperties(tokens);
  const root = document.documentElement;
  Object.keys(css).forEach((key) => {
    if (css[key]) root.style.setProperty(key, css[key]);
  });
  root.style.colorScheme = themeContrast(tokens.bg) === '#111111' ? 'light' : 'dark';
  document.documentElement.dataset.themeName = name || '';
  const btn = $('themeBtn');
  if (btn && name) btn.title = 'theme: ' + name + ' \u2014 open the Theme Manager';
  try {
    localStorage.setItem('pedacito_theme_tokens', JSON.stringify(tokens));
    localStorage.setItem('pedacito_theme_css', JSON.stringify(css));
    localStorage.setItem('pedacito_theme_name', name || '');
  } catch (err) {}
}

function clearThemeTokens() {
  const root = document.documentElement;
  Object.keys(themeCssProperties({
    bg: '#000000', surface: '#000000', surface_alt: '#000000',
    text: '#000000', muted: '#000000', accent: '#000000',
    border: '#000000', success: '#000000', warning: '#000000',
    danger: '#000000',
  })).forEach((key) => root.style.removeProperty(key));
  root.style.colorScheme = '';
  try {
    localStorage.removeItem('pedacito_theme_tokens');
    localStorage.removeItem('pedacito_theme_css');
    localStorage.removeItem('pedacito_theme_name');
  } catch (err) {}
}

// Resolve which palette the GUI paints. gui_theme (config) is the source of
// truth for the nine CSS-native schemes: an explicit browser marker, or the
// server-injected data-theme attribute when no marker exists, wins over
// whatever palette sits in themes.json -- the engine must not flatten a
// crafted scheme into its ten-token derivation. `explicitEngine` marks a
// Theme Manager pick/save/delete, which always switches to the engine palette.
function applyThemePayload(payload, explicitEngine) {
  if (!payload || !payload.themes) return;
  const active = payload.active_theme;
  const tokens = payload.themes[active];
  if (!tokens) return;
  let marker = '';
  try { marker = localStorage.getItem('pedacito_theme') || ''; } catch (err) {}
  const attr = document.documentElement.getAttribute('data-theme') || '';
  if (!explicitEngine) {
    if (isCssScheme(marker)) {
      clearThemeTokens();
      if (marker === 'slate') document.documentElement.removeAttribute('data-theme');
      else document.documentElement.setAttribute('data-theme', marker);
      document.documentElement.dataset.themeName = marker;
      return;
    }
    if (!marker && isCssScheme(attr)) {
      clearThemeTokens();
      document.documentElement.dataset.themeName = attr;
      try { localStorage.setItem('pedacito_theme', attr); } catch (err) {}
      return;
    }
  }
  // Engine-native: an explicit Theme Manager pick, or no CSS preference
  // anywhere. The built-in presets ARE the nine crafted gui_theme schemes,
  // so let their app.css blocks paint the full palette instead of the
  // flattened token derivation; only custom palettes use inline tokens.
  if (isCssScheme(active)) {
    clearThemeTokens();
    if (active === 'slate') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', active);
  } else {
    applyThemeTokens(tokens, active);
    document.documentElement.removeAttribute('data-theme');
  }
  document.documentElement.dataset.themeName = active;
  try { localStorage.setItem('pedacito_theme', active); } catch (err) {}
}

function loadActiveTheme() {
  return PDC.api('/api/themes').then((j) => {
    if (j && j.ok) applyThemePayload(j);
    return j;
  }).catch(() => null);
}

function expandThemeHex(value) {
  let text = String(value || '').trim();
  if (!text.startsWith('#')) text = '#' + text;
  if (/^#[0-9a-f]{3}$/i.test(text)) {
    text = '#' + text.slice(1).split('').map((c) => c + c).join('');
  }
  return /^#[0-9a-f]{6}$/i.test(text) ? text.toLowerCase() : null;
}

function openThemeManager() {
  PDC.api('/api/themes').then((j) => {
    if (!j || !j.ok) {
      PDC.toast('Could not load themes: ' + ((j && j.detail) || 'server error'), true);
      return;
    }
    showThemeManager(j);
  }).catch(() => PDC.toast('Could not load themes (server unreachable).', true));
}

function showThemeManager(initial) {
  let marker0 = '';
  try { marker0 = localStorage.getItem('pedacito_theme') || ''; } catch (err) {}
  const attr0 = document.documentElement.getAttribute('data-theme') || '';
  const st = {
    themes: initial.themes || {},
    builtin: new Set(initial.builtin_names || []),
    active: initial.active_theme,
    project: initial.project_theme || null,
    draft: { ...(initial.themes[initial.active_theme] || {}) },
    cssSchemes: initial.css_schemes || [],
    // Which family the GUI currently paints: a CSS-native gui_theme scheme
    // (crafted app.css block) or an engine palette (inline ten tokens).
    mode: isCssScheme(marker0) ? 'css'
      : (marker0 ? 'engine'
        : (isCssScheme(attr0) ? 'css'
          : (isCssScheme(initial.active_theme) ? 'css' : 'engine'))),
    cssActive: '',
  };
  st.cssActive = st.mode === 'css'
    ? (marker0 || attr0 || initial.active_theme || 'slate') : '';
  const grid = h('div', { class: 'theme-grid', id: 'themeGrid' });
  const editor = h('div', { class: 'theme-editor' });
  const status = h('div', { class: 'theme-status', id: 'themeStatus' });
  const nameInp = h('input', { type: 'text', class: 'theme-name',
    id: 'themeName', placeholder: 'custom theme name', spellcheck: 'false',
    autocomplete: 'off' });
  const rows = {};
  let selected = st.active;

  const say = (text, tone) => {
    status.textContent = text || '';
    status.className = 'theme-status' + (tone ? ' ' + tone : '');
  };
  const preview = () => applyThemeTokens(st.draft, nameInp.value.trim() || selected);

  function syncInputs(tokens) {
    THEME_FIELDS.forEach(([key]) => {
      if (!rows[key]) return;
      rows[key].color.value = tokens[key] || '#000000';
      rows[key].text.value = tokens[key] || '';
    });
  }

  function renderGrid() {
    clear(grid);
    // The nine crafted gui_theme schemes first: they paint their full app.css
    // palette and persist as gui_theme, exactly like the Settings enum.
    st.cssSchemes.forEach((name) => {
      const meta = CSS_SCHEME_META[name] || {};
      const sw = h('span', { class: 'sw',
        style: 'background:' + (meta.bg || '#808080')
          + ';border-color:' + (meta.accent || '#808080') });
      grid.appendChild(h('button', {
        class: 'theme-pill'
          + (st.mode === 'css' && name === st.cssActive ? ' on' : ''),
        title: name,
        onclick: () => selectCssScheme(name),
      }, sw, h('span', { text: meta.label || name })));
    });
    const custom = Object.keys(st.themes).filter((n) => !st.builtin.has(n)).sort();
    // The built-in presets are the crafted schemes above -- one pill each,
    // not a second engine pill for the same name.
    const names = Array.from(st.builtin)
      .filter((n) => st.cssSchemes.indexOf(n) < 0).concat(custom);
    names.forEach((name) => {
      const t = st.themes[name];
      if (!t) return;
      const sw = h('span', { class: 'sw',
        style: 'background:' + t.accent + ';border-color:' + t.border });
      grid.appendChild(h('button', {
        class: 'theme-pill'
          + (st.mode === 'engine' && name === st.active ? ' on' : ''),
        title: name,
        onclick: () => selectTheme(name),
      }, sw, h('span', { text: name }),
        st.builtin.has(name) ? null : h('span', {
          class: 'small-text', text: 'custom' })));
    });
  }

  function selectCssScheme(name) {
    // applyTheme clears the engine tokens, paints the crafted CSS block and
    // persists gui_theme through /api/settings/theme (browser-local fallback).
    applyTheme(name, true);
    st.mode = 'css';
    st.cssActive = name;
    renderGrid();
    say('Active: ' + ((CSS_SCHEME_META[name] || {}).label || name), 'ok');
  }

  function selectTheme(name) {
    say('Switching to ' + name + '\u2026');
    PDC.api('/api/themes/select', { name }).then((j) => {
      if (!j || !j.ok) { say((j && j.detail) || 'Could not switch theme.', 'err'); return; }
      st.themes = j.themes || st.themes;
      st.builtin = new Set(j.builtin_names || []);
      st.active = j.active_theme;
      st.project = j.project_theme || null;
      selected = st.active;
      st.draft = { ...(st.themes[selected] || {}) };
      nameInp.value = st.builtin.has(selected) ? '' : selected;
      syncInputs(st.draft);
      applyThemePayload(j, true);
      st.mode = 'engine';
      renderGrid();
      delBtn.disabled = st.builtin.has(selected);
      say('Active: ' + selected + (st.project ? ' (project override)' : ''), 'ok');
    }).catch(() => say('Could not switch theme.', 'err'));
  }

  THEME_FIELDS.forEach(([key, label]) => {
    const color = h('input', { type: 'color', value: st.draft[key] || '#000000',
      title: label });
    const text = h('input', { type: 'text', value: st.draft[key] || '',
      spellcheck: 'false', autocomplete: 'off' });
    color.addEventListener('input', () => {
      text.value = color.value;
      st.draft[key] = color.value;
      preview();
    });
    text.addEventListener('input', () => {
      const value = expandThemeHex(text.value);
      if (value) {
        color.value = value;
        st.draft[key] = value;
        preview();
      }
    });
    text.addEventListener('blur', () => {
      const value = expandThemeHex(text.value);
      if (value) {
        text.value = value;
        color.value = value;
        st.draft[key] = value;
        preview();
      } else {
        text.value = st.draft[key] || '';
      }
    });
    rows[key] = { color, text };
    editor.append(h('div', { class: 'theme-row' },
      h('span', { class: 'theme-key', text: label }), color, text));
  });

  const saveBtn = h('button', { class: 'primary', text: 'Save Custom Theme',
    id: 'themeSave', onclick: () => {
      const name = nameInp.value.trim();
      if (!name) { say('Enter a custom theme name first.', 'err'); nameInp.focus(); return; }
      const tokens = {};
      for (const key of THEME_TOKEN_KEYS) {
        const value = expandThemeHex(st.draft[key]);
        if (!value) { say('Invalid color for ' + key + '.', 'err'); return; }
        tokens[key] = value;
      }
      PDC.api('/api/themes/save', { name, tokens }).then((j) => {
        if (!j || !j.ok) { say((j && j.detail) || 'Could not save theme.', 'err'); return; }
        st.themes = j.themes || st.themes;
        st.builtin = new Set(j.builtin_names || []);
        st.active = j.active_theme;
        st.project = j.project_theme || null;
        selected = j.active_theme;
        st.draft = { ...(st.themes[selected] || tokens) };
        nameInp.value = selected;
        syncInputs(st.draft);
        applyThemePayload(j, true);
        st.mode = 'engine';
        renderGrid();
        delBtn.disabled = false;
        say('Saved and activated ' + selected + '.', 'ok');
      }).catch(() => say('Could not save theme.', 'err'));
    } });

  const delBtn = h('button', { class: 'danger', text: 'Delete Custom Theme',
    id: 'themeDelete', onclick: () => {
      const name = nameInp.value.trim() || selected;
      if (st.builtin.has(name) || !st.themes[name] || st.builtin.has(selected)
          && name === selected) {
        say('Built-in presets cannot be deleted.', 'err');
        return;
      }
      PDC.confirm('Delete the custom theme \u201c' + name + '\u201d?', 'Delete', true)
        .then((ok) => {
          if (!ok) return;
          PDC.api('/api/themes/delete', { name }).then((j) => {
            if (!j || !j.ok) { say((j && j.detail) || 'Could not delete theme.', 'err'); return; }
            st.themes = j.themes || st.themes;
            st.builtin = new Set(j.builtin_names || []);
            st.active = j.active_theme;
            st.project = j.project_theme || null;
            selected = j.active_theme;
            st.draft = { ...(st.themes[selected] || {}) };
            nameInp.value = st.builtin.has(selected) ? '' : selected;
            syncInputs(st.draft);
            applyThemePayload(j, true);
            st.mode = 'engine';
            renderGrid();
            delBtn.disabled = st.builtin.has(selected);
            say('Deleted ' + name + '. Active: ' + selected + '.', 'ok');
          }).catch(() => say('Could not delete theme.', 'err'));
        });
    } });

  const importBox = h('textarea', { class: 'theme-import', id: 'themeImport',
    placeholder: '{"name":"my-theme","tokens":{"bg":"#101418", ...}}\n'
      + 'or just the ten-token object',
    spellcheck: 'false' });
  const importBtn = h('button', { text: 'Import / Paste JSON',
    id: 'themeImportBtn', onclick: () => {
      try {
        const parsed = JSON.parse(importBox.value || '{}');
        const tokens = parsed.tokens || parsed.theme || parsed;
        const clean = {};
        for (const key of THEME_TOKEN_KEYS) {
          const value = expandThemeHex(tokens[key]);
          if (!value) throw new Error('missing or invalid ' + key);
          clean[key] = value;
        }
        st.draft = clean;
        if (parsed.name) nameInp.value = String(parsed.name);
        syncInputs(clean);
        preview();
        say('Imported palette \u2014 review it, then Save Custom Theme.', 'ok');
      } catch (err) {
        say('Could not import: ' + (err && err.message ? err.message : 'invalid JSON'), 'err');
      }
    } });

  const content = h('div', {},
    h('div', { class: 'theme-hint',
      text: 'GUI schemes paint their full crafted palette and save as gui_theme. '
        + 'Presets and custom palettes are shared with the terminal TUI. '
        + 'Edit any of the ten colors for a live preview.' }),
    grid,
    h('div', { class: 'theme-hint', text: 'Active palette \u2014 ten shared tokens' }),
    editor,
    h('div', { class: 'theme-actions' }, nameInp, saveBtn, delBtn),
    status,
    h('div', { class: 'theme-hint', text: 'Import / Paste JSON' }),
    importBox,
    h('div', { class: 'theme-actions' }, importBtn));
  PDC.modal('Theme Manager & Custom Color Editor', content, [{ label: 'Close' }],
    { wide: true, onClose: () => loadActiveTheme() });
  syncInputs(st.draft);
  renderGrid();
  if (!st.builtin.has(selected)) nameInp.value = selected;
  delBtn.disabled = st.builtin.has(selected);
  if (st.project) say('Active: ' + st.active + ' (project override)', 'ok');
  else say('Active: ' + st.active, 'ok');
}
PDC.openThemes = openThemeManager;

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
  const pinBox = h('div', {});
  let rows = [], matchNames = [];
  const filter = h('input', { type: 'text', class: 'grow', spellcheck: false,
    placeholder: 'filter by name or description\u2026',
    oninput: () => paintList() });
  const task = h('input', { type: 'text', class: 'grow', spellcheck: false,
    placeholder: 'type a task to preview which skills match (Enter)',
    onkeydown: (e) => { if (e.key === 'Enter') previewMatches(); } });
  const newBtn = h('button', { class: 'primary', text: '+ New Custom Skill',
    onclick: () => openEditor(null) });

  // Manual injection: a pinned skill rides the same request-level `pinned`
  // list as @file/@symbol picks, so the server's existing pinned-context
  // block receives its full checklist body.
  function skillRef(name) { return 'skill:' + name; }
  function isSkillPinned(name) {
    return S.pinned.some((p) => p.insert === skillRef(name));
  }
  function pinSkill(r) {
    if (!isSkillPinned(r.name)) {
      S.pinned.push({ kind: 'skill', label: r.name, insert: skillRef(r.name),
        skillName: r.name,
        detail: (r.scope || 'skill') + ' skill: ' + (r.description || '') });
    }
    emit();
    paintPins();
    paintList();
    PDC.toast('Skill \u201c' + r.name + '\u201d will be injected into the next task.');
  }
  function unpinSkill(name) {
    S.pinned = S.pinned.filter((p) => p.insert !== skillRef(name));
    emit();
    paintPins();
    paintList();
    PDC.toast('Removed \u201c' + name + '\u201d from the next task.');
  }
  function paintPins() {
    PDC.clear(pinBox);
    const pins = S.pinned.filter((p) => String(p.insert || '').indexOf('skill:') === 0);
    if (!pins.length) return;
    const box = h('div', { class: 'note-box', style: 'margin:6px 0' },
      h('div', { class: 'row' },
        h('b', { text: 'Pinned for the next task' }),
        h('span', { class: 'grow muted small-text',
          text: 'full checklist body, injected through the pinned-context block' }),
        h('button', { class: 'ghost small', text: 'Clear skills', onclick: () => {
          S.pinned = S.pinned.filter((p) => String(p.insert || '').indexOf('skill:') !== 0);
          emit(); paintPins(); paintList();
        } })));
    pins.forEach((p) => {
      const name = p.skillName || p.label;
      const row = rows.find((x) => x.name === name) || {};
      const chars = (row.body || '').length;
      box.appendChild(h('div', { class: 'row small-text', style: 'padding:2px 0' },
        h('span', { class: 'tag', text: row.scope || 'skill' }),
        h('b', { text: name }),
        h('span', { class: 'grow muted',
          text: (row.description || '') + (chars ? ' \u00b7 ' + chars + ' chars' : '') }),
        h('button', { class: 'ghost small danger', text: 'Remove',
          onclick: () => unpinSkill(name) })));
    });
    pinBox.appendChild(box);
  }

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
      const pinned = isSkillPinned(r.name);
      list.appendChild(h('div', { class: 'row small-text', style: 'padding:3px 0;cursor:pointer',
        onclick: () => showDetail(r) },
        h('span', { class: 'tag' + (matched ? ' act' : ''), text: r.scope }),
        h('b', { text: r.name, style: 'min-width:200px' }),
        matched ? h('span', { class: 'tag act', text: 'matches task' }) : null,
        h('span', { class: 'grow muted', text: r.description }),
        h('button', { class: pinned ? 'ghost small' : 'primary small',
          text: pinned ? 'Pinned for next task' : 'Use in next task',
          onclick: (e) => {
            e.stopPropagation();
            if (pinned) unpinSkill(r.name); else pinSkill(r);
          } }),
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
    const pinned = isSkillPinned(r.name);
    detail.appendChild(h('div', { class: 'note-box', style: 'margin-top:10px' },
      h('div', { class: 'row' },
        h('b', { text: r.name }),
        h('span', { class: 'tag', text: r.scope }),
        h('span', { class: 'muted small-text grow', text: r.source || '' }),
        h('button', { class: pinned ? 'ghost small' : 'primary small',
          text: pinned ? 'Pinned for next task' : 'Use in next task',
          onclick: () => { if (pinned) unpinSkill(r.name); else pinSkill(r); } })),
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
  body.appendChild(pinBox);
  body.appendChild(list);
  body.appendChild(detail);
  body.appendChild(editor);
  paintPins();
  PDC.modal('Skills library & creator', body, [{ label: 'Close' }], { wide: true });
  PDC.api('/api/skills').then((j) => {
    if (!j.ok) { note.textContent = j.detail || 'Could not load the skills library.'; return; }
    rows = j.skills || [];
    paintList();
    paintPins();
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
// Background servers, model-server lifecycle, and live VRAM/RAM telemetry.
// The badge polls slowly (15 s); the modal refreshes after every action.
function _gb(n) { return Number(n || 0).toFixed(1); }
function _memText(mem) {
  const v = (mem && mem.vram) || {}, r = (mem && mem.ram) || {};
  return 'VRAM: ' + _gb(v.used_gb) + ' / ' + _gb(v.total_gb) + ' GB | RAM: '
    + _gb(r.used_gb) + ' / ' + _gb(r.total_gb) + ' GB';
}

function HardwareBadge() {
  const [mem, setMem] = useState(null);
  useEffect(() => {
    let alive = true;
    const load = () => PDC.api('/api/servers').then((j) => {
      if (!alive || !j) return;
      if (j.memory) setMem(j.memory);
      // The same 15 s cycle refreshes the profile picker's disabled states.
      if (j.profile_availability) {
        S.profile_availability = j.profile_availability;
        emit();
      }
    }, () => {});
    load();
    const timer = setInterval(load, 15000);
    return () => { alive = false; clearInterval(timer); };
  }, []);
  if (!mem) return null;
  const v = mem.vram || {}, r = mem.ram || {};
  const src = mem.sources || {};
  const cards = (v.cards || []).map((c) => c.card + ' ' + _gb(c.used_gb)
    + '/' + _gb(c.total_gb) + ' GB (' + (c.source || '?') + ')').join('\n');
  return html`<span class="chip" id="hwChip"
      title=${'hardware totals only \u2014 no per-process GPU attribution'
        + '\nRAM source: ' + (src.ram_total_gb || 'unavailable')
        + '\nVRAM source: ' + (src.vram_total_gb || 'unavailable')
        + (cards ? '\n' + cards : '')}
      onClick=${() => serversDialog()}>
    <b>${_memText(mem)}</b>
  </span>`;
}

function serversDialog() {
  // One registry row per endpoint: every profile pointing at the same URL
  // shares it. The row is titled after the server itself (kind + host) --
  // naming it after the first profile (s.name) reads like a model sitting
  // unloaded on the server. s.name stays the registry key the actions route
  // by; the profile names ride as a secondary line.
  const KIND_LABELS = {
    lemonade: 'Lemonade', lemond: 'Lemonade', 'llama-server': 'llama.cpp',
    strata: 'Strata', avalon: 'Avalon', 'local-moe': 'Local MoE',
    local: 'Local server', custom: 'Local server',
  };
  function serverTitle(s) {
    const host = String(s.base_url || '')
      .replace(/^[a-z][a-z0-9+.-]*:\/\//i, '').split('/')[0];
    const kind = KIND_LABELS[String(s.kind || '').toLowerCase()]
      || 'Local server';
    return host ? kind + ' \u00b7 ' + host : (s.name || kind);
  }
  const status = h('div', { class: 'muted small-text', style: 'margin-bottom:8px' });
  const list = h('div', {});
  const check = h('input', { type: 'checkbox', checked: !!S.status.auto_evict_servers });

  function setAutoEvict(on) {
    PDC.api('/api/servers/auto-evict', { enabled: on }).then((j) => {
      if (!j.ok) { PDC.toast(j.detail || 'could not save the setting', true); return; }
      S.status.auto_evict_servers = !!j.auto_evict_servers;
      emit();
      PDC.toast(on ? 'Auto-unload other local servers on switch: on'
                   : 'Auto-unload other local servers on switch: off');
    });
  }
  check.addEventListener('change', () => setAutoEvict(!!check.checked));

  function act(path, payload) {
    status.textContent = 'Working\u2026';
    PDC.api(path, payload).then((j) => {
      if (j.skipped) {
        status.textContent = j.detail || 'nothing to do';
      } else if (!j.ok) {
        status.textContent = (j.detail || 'the server refused the action');
      } else {
        const verb = path.indexOf('pin') >= 0
          ? (payload.pinned ? 'Pinned ' : 'Unpinned ')
          : (path.indexOf('unload') >= 0 ? 'Unloaded ' : 'Loaded ');
        status.textContent = verb
          + (payload.model || payload.server || 'model')
          + ' \u00B7 ' + _memText(j.memory || {});
      }
      paint(j);
    }, () => { status.textContent = 'The GUI server did not answer.'; });
  }

  function paint(j) {
    PDC.clear(list);
    if (!j) {
      status.textContent = 'Could not read the server registry.';
      return;
    }
    if (!j.ok) {
      status.textContent = j.detail || 'Could not read the server registry.';
      // A refused load/unload/pin still answers 502 WITH fresh telemetry:
      // keep the registry visible instead of blanking the dialog.
      if (!(j.servers || []).length) return;
    } else if (!j.detail) {
      status.textContent = _memText(j.memory || {});
    }
    const rows = j.servers || [];
    if (!rows.length) {
      list.appendChild(h('div', { class: 'tip',
        text: 'No local servers are configured. Add a local profile in Settings \u2192 Profiles.' }));
      return;
    }
    rows.forEach((s) => {
      const loaded = s.loaded_models || [];
      const loading = s.loading_models || [];
      const available = (s.models || []).filter(
        (m) => loaded.indexOf(m) < 0 && loading.indexOf(m) < 0);
      const sel = h('select', { title: 'model to load' });
      available.forEach((m) => sel.appendChild(h('option', { value: m, text: m })));
      const pinSel = h('select', { title: 'loaded model to pin' });
      loaded.forEach((m) => pinSel.appendChild(h('option', { value: m, text: m })));
      const canPin = ['lemonade', 'lemond'].indexOf(
        String(s.kind || '').toLowerCase()) >= 0;
      const loadBtn = h('button', { class: 'small primary', text: 'Load',
        disabled: !available.length,
        onclick: () => act('/api/servers/load', { server: s.name, model: sel.value }) });
      const unloadBtn = h('button', { class: 'small danger', text: loaded.length > 1 ? 'Eject all' : 'Unload / Eject',
        disabled: !loaded.length,
        onclick: () => act('/api/servers/unload',
          { server: s.name, model: loaded.length === 1 ? loaded[0] : '' }) });
      list.appendChild(h('div', { class: 'note-box', style: 'margin-bottom:8px' },
        h('div', { class: 'row' },
          h('b', { text: serverTitle(s) }),
          h('span', { class: 'tag' + (s.online ? ' act' : ''), text: s.online ? 'online' : 'offline' }),
          h('span', { class: 'tag', text: s.kind || 'custom' }),
          h('span', { class: 'grow' }),
          unloadBtn),
        h('div', { class: 'mono small-text', style: 'margin-top:4px', text: s.base_url || '' }),
        (s.profiles && s.profiles.length) ? h('div', {
          class: 'muted small-text', style: 'margin-top:2px',
          title: s.profiles.join(', '),
          text: 'profiles: ' + s.profiles.slice(0, 3).join(', ')
            + (s.profiles.length > 3
               ? ' + ' + (s.profiles.length - 3) + ' more' : '') }) : null,
        h('div', { class: 'small-text', style: 'margin-top:4px',
          text: loaded.length ? 'resident: ' + loaded.join(', ')
            : (loading.length ? 'loading: ' + loading.join(', ')
                              : 'no model resident') }),
        (canPin && loaded.length) ? h('div', { class: 'row', style: 'margin-top:6px' },
          pinSel,
          h('button', { class: 'small', text: 'Pin',
            title: 'prevent Lemonade from evicting this model',
            onclick: () => act('/api/servers/pin',
              { server: s.name, kind: s.kind, model: pinSel.value, pinned: true }) }),
          h('button', { class: 'small', text: 'Unpin',
            title: 'allow Lemonade to evict this model again',
            onclick: () => act('/api/servers/pin',
              { server: s.name, kind: s.kind, model: pinSel.value, pinned: false }) })) : null,
        (available.length || !s.online) ? h('div', { class: 'row', style: 'margin-top:6px' },
          sel, loadBtn) : null));
    });
  }

  function refresh() {
    status.textContent = 'Probing local servers\u2026';
    PDC.api('/api/servers').then(paint,
      () => { status.textContent = 'The GUI server did not answer.'; });
  }

  const body = h('div', {},
    status,
    list,
    h('label', { class: 'chk', style: 'display:block;margin-top:10px' },
      check, ' Auto-unload other local servers on switch'),
    h('div', { class: 'muted small-text', style: 'margin-top:6px',
      text: 'Before a run, resident models on other configured local servers are unloaded and VRAM/RAM is polled briefly. A sleeping server is never touched.' }),
    h('div', { class: 'muted small-text', style: 'margin-top:6px',
      text: 'Load / unload acts on the server itself (Lemonade: POST /v1/load, /v1/unload) \u2014 it creates no profile and switches nothing. The dropdown is the live catalog the server reports, including models no profile references; running a model still needs a profile pointing at this server with that model id.' }));
  PDC.modal('Servers & memory', body,
    [{ label: 'Refresh', onclick: () => { refresh(); return false; } },
     { label: 'Close' }], { wide: true });
  refresh();
}

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
    h('p', { class: 'small-text', text: 'Deletes this project\u2019s .pedacito/ runtime state (index, sessions, consolidated handoff, backups, events, memory unless kept, generated rules without the authored checkbox) and the auto-plan file (PLAN.md). Source code and .pedacito.toml are never touched. Running background servers are stopped first.' }),
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
    if (j.plan_file && j.plan_removed) {
      out.appendChild(h('div', { class: 'small-text',
        text: (dry ? 'Would remove' : 'Removed') + ' plan file: ' + j.plan_file }));
    }
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
        if (j.ok) { PDC.toast(j.summary || 'Project state reset.'); refreshState(); }
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
  // Peek lists another folder's profiles, which the current server map does
  // not describe, so the annotations only apply to the open config.
  const availability = peek ? {} : (S.profile_availability || {});
  const optionFor = (p) => {
    const model = (s.profile_models || {})[p] || '';
    const label = p + (model ? ' \u00B7 ' + model : '');
    const state = availability[p] || 'unknown';
    if (state === 'unavailable') return {
      label: label + ' \u00B7 unavailable', disabled: true,
      title: 'model ' + (model || '?') + ' was not found on its server \u2014 '
        + 'it may have been deleted; update the profile in Settings' };
    if (state === 'offline') return {
      label: label + ' \u00B7 server offline', disabled: true,
      title: 'the model server is not reachable \u2014 start it; this profile '
        + 'becomes selectable again after the next telemetry poll' };
    return { label: label, disabled: false, title: '' };
  };
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
      ${profiles.map((p) => {
        const o = optionFor(p);
        return html`<option key=${p} value=${p} disabled=${o.disabled}
                            title=${o.title}>${o.label}</option>`;
      })}
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
function tuneTip() {
  const s = S.status || {};
  if (!s.auto_tune) return '';
  const src = (s.tuning_sources || {}).context_tokens;
  return '\nauto-tune: ' + PDC.fmtInt(s.context_tokens || 0) + ' context tokens, '
    + Number(s.timeout || 0).toFixed(0) + 's timeout, max_concurrency '
    + (s.max_concurrency || 0) + (src && src !== 'default' ? '\ncontext: ' + src : '');
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
        + tuneTip()
        + '\nClick to open Settings \u2192 Model discovery'}>
      <b>\u2713 Calibrated (${ago})</b></span>`;
  }
  if (cal.calibrated && cal.calibrated_stale) {
    return html`<span class="chip cal-chip stale" id="calChip" onClick=${openProfiles}
      title=${'The model card is older than 30 days (or from an older calibrate version).\nRe-calibrate to refresh the tuned defaults.'
        + tuneTip() + '\nClick to open Settings \u2192 Model discovery'}>
      <b>\u26A0 Calibration stale (${(cal.calibrated_age_days || 0) + 'd ago'})</b></span>`;
  }
  return html`<span class="chip cal-chip none" id="calChip" onClick=${openProfiles}
    title=${'No model card yet \u2014 calibrate to measure this model and tune its defaults.'
      + tuneTip() + '\nClick to open Settings \u2192 Model discovery'}>
    <b>\u25CB Not calibrated</b></span>`;
}

// The live telemetry footer: context window fill, last turn and session
// tokens, generation speed, KV cache hit, folds, and who is answering.
// Click it to unfold the drawer with the full breakdown.
// Live LLM heartbeat: while a request is in flight the server reports
// llm_call = {provider, model, phase, started}; a pulsing dot plus a 1s JS
// timer shows who is thinking and for how long ("🧠 DeepSeek (deepseek-
// reasoner) is thinking & generating... (24s elapsed)"). The matching `call`
// event clears it, so it never outlives the request it describes.
function LlmHeartbeat() {
  const call = S.status.llm_call;
  const started = call && call.started;
  const [, tick] = useState(0);
  useEffect(() => {
    if (!started) return undefined;
    const id = setInterval(() => tick((x) => x + 1), 1000);
    return () => clearInterval(id);
  }, [started]);
  if (!started) return null;
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - started));
  const who = call.provider || 'The model';
  const model = call.model ? ' (' + call.model + ')' : '';
  return html`<span class="llm-heartbeat" id="llmHeartbeat"
                    title=${'phase: ' + (call.phase || 'call')}>
    <i class="llm-pulse"></i>\u{1F9E0} ${who}${model} is thinking & generating... (${secs}s elapsed)</span>`;
}

function TelemetryBar() {
  const s = S.status;
  const [open, setOpen] = useState(false);
  const t = s.telemetry || {};
  // Visible during any live run or in-flight LLM call even before the first
  // usage lands (a fresh New project has no context window reported yet).
  const live = !!s.job || !!(s.llm_call && s.llm_call.started);
  if ((!s.root || !t.context_tokens) && !live) return null;
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
        <div><span class="muted">Last turn total</span><b>${PDC.fmtInt(t.last_total_tokens)} tokens \u00B7 ${PDC.fmtInt(t.last_reasoning_tokens)} reasoning</b></div>
        <div><span class="muted">Latency</span><b>${fmt1(t.last_elapsed_s)} s</b></div>
        <div><span class="muted">Speed (last / avg)</span><b>${fmt1(t.last_tok_per_s)} / ${fmt1(t.avg_tok_per_s)} tok/s</b></div>
        <div><span class="muted">Context window</span><b>${PDC.fmtInt(t.context_tokens)} tokens (${s.context_origin || 'default'})</b></div>
        <div><span class="muted">Session</span><b>${PDC.fmtInt(u.calls)} calls \u00B7 ${PDC.fmtTokens(u.tokens_in)} in / ${PDC.fmtTokens(u.tokens_out)} out \u00B7 ${Math.round(u.seconds || 0)}s</b></div>
        <div><span class="muted">Session total</span><b>${PDC.fmtInt(t.session_total_tokens)} tokens \u00B7 ${PDC.fmtInt(t.session_reasoning_tokens)} reasoning</b></div>
        <div><span class="muted">KV cache hit</span><b>${fmt1(t.cache_hit_pct)}%</b></div>
        <div><span class="muted">Context folds</span><b>${s.folds || 0}</b></div>
        <div><span class="muted">Cost</span><b>${u.cost ? '$' + Number(u.cost).toFixed(4) : '\u2014'}</b></div>
        <div><span class="muted">Provider</span><b>${t.provider || '\u2014'}</b></div>
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
            <${MenuItem} label="Checkpoints / Undo\u2026" disabled=${!s.root} action=${openCheckpointsDialog} />
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
          <button id="checkpointsBtn" title="time-travel: inspect, diff, restore or branch from any checkpoint"
                  disabled=${!s.root} onClick=${openCheckpointsDialog}>\u23F1 Checkpoints / Undo</button>
          <button id="rulesBtn" title="edit persistent project rules (.pedacito/rules.md) applied to every Task, Auto and Refine call"
                  onClick=${openRulesDialog}>\uD83D\uDCDC Rules</button>
          <button id="serversBtn" title="live VRAM/RAM and local model-server lifecycle"
                  onClick=${() => serversDialog()}>Servers</button>
          <button id="benchBtn" title="run a diagnostic benchmark against a configured profile" onClick=${() => openBenchmarks()}>\u25A6 Benchmarks</button>
          <button id="arenaBtn" title="compare two models on one prompt in isolated sandboxes, then apply the winner"
                  disabled=${!s.root} onClick=${() => openArena()}>\u2694 Arena</button>
          <button id="settingsBtn" title="profiles and settings" onClick=${() => callSettings()}>\u2699 Settings</button>
          <button id="themeBtn" title="theme manager and custom color editor"
                  onClick=${() => openThemeManager()}>\uD83C\uDFA8 Theme</button>
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
              title=${(s.model_status || 'click to check the model server') + (s.base_url ? '\n' + s.base_url : '') + (s.context_tokens ? '\nwindow ' + PDC.fmtInt(s.context_tokens) + ' tokens (' + s.context_origin + ')' : '') + (s.auto_tune ? '\nauto-tune: timeout ' + Number(s.timeout || 0).toFixed(0) + 's, max_concurrency ' + (s.max_concurrency || 0) + '\ncontext: ' + ((s.tuning_sources || {}).context_tokens || s.context_origin || 'default') : '')}
              onClick=${() => { if (s.root && !j) checkModel(); }}>
          <i class=${'dot ' + (j ? 'busy' : s.model_ok === true ? 'ok' : s.model_ok === false ? 'bad' : 'idle')}></i>
          <b>${s.needs_setup ? 'no model selected' : (s.model || (s.root ? '\u2014' : 'no model selected'))}</b>
          ${s.model && !s.needs_setup ? html`<span class="tag ${s.disable_thinking ? '' : 'act'}"
            title=${s.disable_thinking ? 'thinking is disabled for this profile (disable_thinking = true)'
                                       : 'thinking is allowed (disable_thinking = false) \u2014 turn it off in Settings \u2192 Profiles'}>${s.disable_thinking ? 'no-think' : 'thinking'}</span>` : null}
        </span>
        <${HardwareBadge} />
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
        <label class="chip" id="postEditVerifyChip"
              title=${'Post-edit verification (auto_verify): '
                + (s.auto_verify ? 'ON \u2014 ' + s.auto_verify : 'off')
                + '. Detected test command: '
                + (s.detected_test_command || 'none detected')
                + '. The syntax, static and interface-contract gates and the scoped'
                + ' impacted tests always run on every edit; the full suite is this'
                + ' verify command.'}>
          <input type="checkbox" checked=${!!s.auto_verify}
                 onChange=${(e) => setPostEditVerify(e.target.checked)} />
          <b>Post-edit verify</b>
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
// ✨ Creative / UI-polish mode: forces the warm designer persona for the next
// run. Off is not "cold" -- creative_auto still detects design tasks on its own.
function toggleCreative() {
  S.creative = !S.creative;
  try { localStorage.setItem('pedacito_creative', S.creative ? '1' : '0'); } catch (e) {}
  emit();
}

// ---- @file / @symbol autocomplete -------------------------------------------
// The menu is deliberately tiny: the server already fuzzy-matches the warm
// index, so each keystroke is one GET and no model work. Boundary matching
// keeps an email (`user@example.com`) from opening it.
let ctxSuggestTimer = 0;
function contextMention(text, caret) {
  const before = String(text || '').slice(0, caret == null ? String(text || '').length : caret);
  const m = before.match(/(^|[\s([{,;:])@([A-Za-z0-9_][A-Za-z0-9_./:\\-]*)$/);
  if (!m) return null;
  return { query: m[2], start: m.index + m[1].length, end: caret };
}
function clearContextSuggest() {
  S.ui.suggestSeq = (S.ui.suggestSeq || 0) + 1;
  if (S.ui.suggest) { S.ui.suggest = null; emit(); }
}
function requestContextSuggest(text, caret) {
  const m = contextMention(text, caret);
  if (!m) { clearContextSuggest(); return; }
  const seq = (S.ui.suggestSeq || 0) + 1;
  S.ui.suggestSeq = seq;
  clearTimeout(ctxSuggestTimer);
  ctxSuggestTimer = setTimeout(() => {
    PDC.api('/api/context/suggest?q=' + encodeURIComponent(m.query)).then((j) => {
      if (seq !== S.ui.suggestSeq) return;      // a newer keystroke won
      const items = (j.items || j.suggestions || []).slice(0, 20);
      S.ui.suggest = items.length
        ? { items: items, index: 0, start: m.start, end: m.end, query: m.query }
        : null;
      emit();
    }, () => { if (seq === S.ui.suggestSeq) { S.ui.suggest = null; emit(); } });
  }, 70);
}
function insertPinned(item) {
  const ta = inputEl || $('input');
  const st = S.ui.suggest;
  if (!ta || !st) return;
  const text = ta.value || '';
  const token = String(item.insert || ('@' + item.label)) + ' ';
  const before = text.slice(0, st.start), after = text.slice(st.end);
  ta.value = before + token + after;
  const pos = (before + token).length;
  try { ta.setSelectionRange(pos, pos); } catch (e) {}
  ta.focus();
  S.drafts[S.mode] = ta.value;
  writeDraft(ta.value);
  S.ui.suggest = null;
  S.ui.suggestSeq = (S.ui.suggestSeq || 0) + 1;
  if (!S.pinned.some((p) => p.insert === item.insert)) {
    S.pinned.push({ kind: item.kind, label: item.label,
                    insert: item.insert, detail: item.detail || '' });
  }
  emit();
}
function removePinned(i) {
  S.pinned.splice(i, 1);
  emit();
}

// ---- persistent project rules (.pedacito/rules.md) --------------------------
function openRulesDialog() {
  if (!S.status.root) { PDC.toast('Open a project first.', true); return; }
  PDC.api('/api/rules').then((j) => {
    if (j._status !== 200 || !j.ok) {
      PDC.toast(j.detail || 'Could not read project rules.', true);
      return;
    }
    const ta = h('textarea', { class: 'rules-editor', rows: '16',
      spellcheck: false, placeholder:
        'Project-specific rules every Task, Auto and Refine call must follow.\n' +
        'Examples:\n- Use pathlib, never os.path\n- Tests live in tests/ and use pytest\n' +
        '- Never edit generated files under build/',
      value: j.text || '' });
    const info = h('p', { class: 'muted small-text', style: 'margin:0 0 8px',
      text: 'Saved as .pedacito/rules.md and injected into the static prompt prefix, ' +
            'so it rides the cached KV prefix on local and local-moe servers.' });
    const note = h('p', { class: 'muted small-text', style: 'margin:8px 0 0',
      text: 'Empty content clears the file.' });
    PDC.modal('Project Rules (.pedacito/rules.md)',
      h('div', {}, info, ta, note), [
        { label: 'Cancel' },
        { label: 'Clear', danger: true, stay: true, onclick: () => { ta.value = ''; ta.focus(); } },
        { label: 'Save rules', primary: true, stay: true, onclick: (close) => {
            PDC.api('/api/rules', { text: ta.value }).then((r) => {
              if (r._status !== 200 || !r.ok) {
                PDC.toast(r.detail || 'Could not save rules.', true);
                return;
              }
              PDC.toast(r.exists ? 'Project rules saved.' : 'Project rules cleared.');
              close();
            }, () => PDC.toast('The GUI server did not answer.', true));
          } },
      ], { wide: true });
  });
}

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
    S.ui.modeConfirm = null;               // a mode switch drops the choice
    if (k !== 'auto') S.ui.modeCheck = null;
    else scheduleModeCheck(S.drafts.auto || '');
    try { localStorage.setItem('pedacito_mode', k); } catch (e) {}
    writeDraft(S.drafts[k] || '');
    S.ui.optsOpen = false;
    emit();
  };
  const autoText = (S.drafts.auto || '').trim();
  const nudge = S.mode === 'auto' && !j && S.ui.modeCheck
    && S.ui.modeCheck.mismatch && S.ui.modeCheck.text === autoText
    && S.ui.autoAck !== autoText;
  const planning = (S.mode === 'auto' && !!opt('auto', 'plan_only'))
    || (S.mode === 'task' && opt('task', 'plan') === 'always');
  return html`
    <div id="composer"
         ondragenter=${(e) => { e.preventDefault(); e.currentTarget.classList.add('drop'); }}
         ondragover=${(e) => { e.preventDefault(); }}
         ondragleave=${(e) => { e.currentTarget.classList.remove('drop'); }}
         ondrop=${(e) => { e.preventDefault(); e.currentTarget.classList.remove('drop'); if (e.dataTransfer && e.dataTransfer.files.length) attachFiles(e.dataTransfer.files); }}>
      <div class="composer-resizer"
           title="Drag to resize composer (double-click to toggle expand/compact)"
           onMousedown=${dragComposer} onDblclick=${toggleComposer}></div>
      ${S.ui.modeConfirm ? html`
        <div id="modeConfirm" role="alertdialog" aria-label="Run as a single-pass Task instead?">
          <b>Run as a single-pass Task instead?</b>
          <span class="muted">Task mode applies all clauses in one fast pass without creating a 7-step PLAN.md.${S.ui.modeConfirm.reason ? ' ' + S.ui.modeConfirm.reason : ''}</span>
          <span class="modeConfirmBtns">
            <button class="small primary" onClick=${runAsTaskFromConfirm}>Run as Task (Recommended)</button>
            <button class="small" onClick=${runAutoAnyway}>Run Auto Anyway</button>
          </span>
        </div>` : (nudge ? html`
        <div id="modeNudge" role="status">
          <span>\uD83D\uDCA1 Looks like a single-pass edit \u2014</span>
          <button class="small primary" onClick=${switchToTaskMode}>Switch to Task mode (Recommended)</button>
          <button class="small ghost" title="Keep Auto mode for this prompt"
                  onClick=${() => { S.ui.autoAck = autoText; emit(); }}>Continue with Auto</button>
        </div>` : null)}
      <div id="modeTabs">
        ${MODE_ORDER.map((k) => html`
          <button key=${k} class=${k === S.mode ? 'on' : ''} title=${MODES[k].title}
                  onClick=${() => setMode(k)}>${MODES[k].label}</button>`)}
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
      ${planning ? html`<p id="planHint" title=${PLAN_HINT}>${PLAN_HINT}</p>` : null}
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
      ${S.pinned.length ? html`
        <div id="pinnedRow" title="Explicit context pinned for this session (also sent as request pins)">
          <span class="pinned-title">\uD83D\uDCCC Pinned</span>
          ${S.pinned.map((p, i) => html`
            <span key=${'p' + i} class=${'pin ' + (p.kind === 'symbol' ? 'symbol' : 'file')}
                  title=${p.detail || p.label}>
              ${p.kind === 'symbol' ? '\u0192' : '\u25A4'} ${p.label}
              <button title="unpin" onClick=${() => removePinned(i)}>\u2715</button>
            </span>`)}
          <button class="pin-clear" title="unpin everything" onClick=${() => { S.pinned = []; emit(); }}>clear</button>
        </div>` : null}
      <textarea id="input" rows="3" spellcheck="true" ref=${taRef}
                placeholder=${isHow ? m.ph : (j ? (steerable ? 'Type a note to steer the model (Enter)\u2026' : 'Working\u2026 press Stop to cancel.') : m.ph)}
                disabled=${!!j && !steerable && !isHow}
                oninput=${(e) => { S.drafts[S.mode] = e.target.value;
                  writeDraft(e.target.value);       // F5 must not lose it
                  scheduleModeCheck(e.target.value);
                  requestContextSuggest(e.target.value, e.target.selectionStart);
                  const rb = $('refineBtn');
                  if (rb) rb.disabled = S.ui.refining || !e.target.value.trim(); }}
                onclick=${(e) => requestContextSuggest(e.target.value, e.target.selectionStart)}
                onblur=${() => setTimeout(() => { if (S.ui.suggest) clearContextSuggest(); }, 180)}
                onkeydown=${(e) => {
                  const sg = S.ui.suggest;
                  if (sg && sg.items.length) {
                    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                      e.preventDefault();
                      const n = sg.items.length;
                      sg.index = (sg.index + (e.key === 'ArrowDown' ? 1 : n - 1)) % n;
                      emit();
                      return;
                    }
                    if (e.key === 'Enter' || e.key === 'Tab') {
                      e.preventDefault();
                      insertPinned(sg.items[sg.index] || sg.items[0]);
                      return;
                    }
                    if (e.key === 'Escape') {
                      e.preventDefault(); clearContextSuggest(); return;
                    }
                  }
                  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
                    e.preventDefault(); clearContextSuggest(); send();
                  }
                }} />
      ${(() => {
        const sg = S.ui.suggest;
        if (!sg || !sg.items.length) return null;
        const ta = taRef.current;
        const style = ta
          ? ('left:' + ta.offsetLeft + 'px;width:' + ta.offsetWidth +
             'px;bottom:calc(100% - ' + Math.max(0, ta.offsetTop - 4) + 'px)')
          : '';
        return html`
          <div id="ctxMenu" role="listbox" aria-label="Pin file or symbol context" style=${style}>
            <div class="ctx-head">Pin context \u00B7 ${sg.query ? '@' + sg.query : 'files & symbols'}</div>
            ${sg.items.map((x, i) => html`
              <button key=${'ctx' + i} type="button" role="option"
                      aria-selected=${i === sg.index ? 'true' : 'false'}
                      class=${'ctx-item' + (i === sg.index ? ' on' : '')}
                      onMouseDown=${(ev) => ev.preventDefault()}
                      onClick=${() => insertPinned(x)}>
                <span class="ctx-kind">${x.kind === 'symbol' ? '\u0192' : '\u25A4'}</span>
                <span class="ctx-label">${x.insert || ('@' + x.label)}</span>
                <span class="ctx-detail">${x.detail || ''}</span>
              </button>`)}
            <div class="ctx-foot">\u2191\u2193 move \u00B7 Enter/Tab insert \u00B7 Esc dismiss</div>
          </div>`;
      })()}
      <div id="optsBox" key=${S.mode} hidden=${!S.ui.optsOpen}>
        <div class="opts-grid">${OPTS[S.mode].filter((o) => o.type !== 'check' && !(S.mode === 'new' && o.k === 'project')).map((o, idx) => html`<${OptField} key=${'g' + idx} o=${o} />`)}</div>
        ${OPTS[S.mode].some((o) => o.type === 'check') ? html`<div class="opts-checks">${OPTS[S.mode].filter((o) => o.type === 'check').map((o, idx) => html`<${OptField} key=${'c' + idx} o=${o} />`)}</div>` : null}
      </div>
      <div id="composerBtns">
        <label id="effortControl" class="effort"
               title="Reasoning effort forwarded as reasoning_effort to local-moe (Strata/Avalon) and reasoning-capable hosted models. Default sends no field.">
          <span>Effort</span>
          <select value=${s.reasoning_effort || 'default'}
                  onChange=${(e) => setReasoningEffort(e.target.value)}>
            <option value="default">Default</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </label>
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
        <button id="creativeBtn" class=${'ghost' + (S.creative ? ' on' : '')}
                hidden=${isHow || S.mode === 'explain'}
                aria-pressed=${S.creative ? 'true' : 'false'}
                title="Creative / UI-polish mode: warmer sampling, a designer persona and full-page rewrites for UI work. Off = design tasks are still auto-detected."
                onClick=${toggleCreative}>${S.creative ? '✨ Creative: on' : '✨ Creative'}</button>
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
  if (o.k === 'architect_provider') {
    // Hybrid Stage-1 planner: the choices are the live profile list (the
    // static OPTS table cannot know them), each labeled with its model.
    const s = S.status || {};
    const names = s.profiles || [];
    const models = s.profile_models || {};
    const cur = val || s.architect_provider || '';
    el = html`<select value=${cur} onchange=${(e) => set(e.target.value)}>
      <option value="">None (single-model runs)</option>
      ${names.map((n) => html`<option key=${n} value=${n}>${n + (models[n] ? ' \u00b7 ' + models[n] : '')}</option>`)}
    </select>`;
  } else if (o.type === 'select') {
    el = html`<select value=${val} onchange=${(e) => set(e.target.value)}>
      ${o.opts.map((x) => html`<option key=${x[0]} value=${x[0]}>${x[1]}</option>`)}
    </select>`;
  } else if (o.type === 'check') {
    return html`<label class="chk" title=${o.k === 'plan_only' ? PLAN_HINT : null}><input type="checkbox" checked=${!!val}
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
  const [editDraft, setEditDraft] = useState('');
  useEffect(() => {                       // keep the two panes in step
    const a = oldRef.current, b = newRef.current;
    if (!a || !b) return;
    const sa = () => { if (b.scrollTop !== a.scrollTop) b.scrollTop = a.scrollTop; };
    const sb = () => { if (a.scrollTop !== b.scrollTop) a.scrollTop = b.scrollTop; };
    a.addEventListener('scroll', sa); b.addEventListener('scroll', sb);
    return () => { a.removeEventListener('scroll', sa); b.removeEventListener('scroll', sb); };
  }, [D, S.diffRel, S.editing]);
  useEffect(() => {                       // fill the editor when entering it
    if (S.editing) {
      const next = (D && D.after) || '';
      setEditDraft(next);
      if (editBox.current) editBox.current.value = next;
    }
  }, [S.editing, D]);
  useEffect(() => {                       // keep textarea synced across split/unified toggle
    if (S.editing && editBox.current && editBox.current.value !== editDraft) {
      editBox.current.value = editDraft;
    }
  }, [unified, S.editing]);
  const switchDiffFile = useCallback((rel, editAfter) => {
    if (!R) return;
    if (rel === S.diffRel) {
      if (editAfter && !S.editing && S.diff) toggleEdit();
      return;
    }
    if (!S.editing) {
      S.editAfterLoad = !!editAfter;
      loadDiffFile(rel);
      return;
    }
    const currentText = editBox.current ? editBox.current.value : editDraft;
    const dirty = D && currentText !== (D.after || '');
    if (!dirty) {
      S.editing = false;
      S.editAfterLoad = !!editAfter;
      loadDiffFile(rel);
      return;
    }
    const fromRel = S.diffRel;
    PDC.modal('Unsaved changes',
      h('p', { style: 'margin:0', text: 'Save your edits to ' + fromRel + ' before switching to ' + rel + '?' }),
      [
        { label: 'Cancel' },
        { label: 'Discard', danger: true, onclick: () => {
          S.editing = false;
          S.editAfterLoad = !!editAfter;
          loadDiffFile(rel);
        } },
        { label: 'Save & switch', primary: true, onclick: () => {
          PDC.api('/api/review/rediff', { id: R.id, rel: fromRel, text: currentText }).then((d) => {
            S.editing = false;
            if (d._status === 200) {
              const item = (S.review && S.review.files || []).find((x) => x.rel === fromRel);
              if (item) { item.added = d.added; item.removed = d.removed; item.edited = true; }
              S.editAfterLoad = !!editAfter;
              loadDiffFile(rel);
            } else {
              PDC.toast(d.detail || 'could not update', true);
              emit();
            }
          });
        } },
      ], { narrow: true });
  }, [R, D, editDraft]);
  useEffect(() => {
    if (!R || !R.files || R.files.length < 2) return;
    const onKey = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key !== '[' && e.key !== ']') return;
      const t = e.target;
      const tag = (t && t.tagName || '').toUpperCase();
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable)) return;
      if (document.querySelector('.overlay') || !S.ui.diffOpen) return;
      const idx = R.files.findIndex((f) => f.rel === S.diffRel);
      const len = R.files.length;
      const next = idx < 0 ? 0 : (idx + (e.key === ']' ? 1 : -1) + len) % len;
      e.preventDefault();
      switchDiffFile(R.files[next].rel, false);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [R, S.diffRel, S.ui.diffOpen, switchDiffFile]);
  const foldSig = Object.keys(S.ui.folds).filter((k) => k.startsWith(S.diffRel + '#')).join('|');
  const { oldRows, newRows, uniRows } = useMemo(() => {
    const oldR = [], newR = [], uniR = [];
    const rows = (D && D.rows) || [];
    rows.forEach((r, i) => {
      const key = 'r' + i;
      if (r[0] === 'skip') {
        const foldKey = S.diffRel + '#' + r[2];
        const opened = S.ui.folds[foldKey] === true;
        const label = '\u22EF ' + r[1] + ' unchanged line' + (r[1] === 1 ? '' : 's') + ' \u2014 click to show';
        const mk = () => html`<div key=${key} class="skip" hidden=${opened}
          onClick=${() => { S.ui.folds[foldKey] = true; emit(); }}>${label}</div>`;
        oldR.push(mk()); newR.push(mk()); uniR.push(mk());
        return;
      }
      const kind = r[0], show = !r[7] || S.ui.folds[S.diffRel + '#' + r[7]] === true;
      if (kind === 'equal') {
        oldR.push(codeLine('equal', r[1], r[2], show, key));
        newR.push(codeLine('equal', r[3], r[4], show, key));
        uniR.push(uniLine('equal', r[1], r[2], show, key));
      } else if (kind === 'replace') {
        oldR.push(codeLine('replace del', r[1], markedCells(r[2], r[5]), show, key));
        newR.push(codeLine('replace add', r[3], markedCells(r[4], r[6]), show, key));
        uniR.push(uniLine('del', r[1], markedCells(r[2], r[5]), show, key + 'a'));
        uniR.push(uniLine('add', r[3], markedCells(r[4], r[6]), show, key + 'b'));
      } else if (kind === 'delete') {
        oldR.push(codeLine('delete', r[1], r[2], show, key));
        newR.push(codeLine('blank', null, '', show, key));
        uniR.push(uniLine('del', r[1], r[2], show, key));
      } else {
        oldR.push(codeLine('blank', null, '', show, key));
        newR.push(codeLine('insert', r[3], r[4], show, key));
        uniR.push(uniLine('add', r[3], r[4], show, key));
      }
    });
    return { oldRows: oldR, newRows: newR, uniRows: uniR };
  }, [D, S.diffRel, foldSig]);

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
               title=${f.message || ''} onClick=${() => switchDiffFile(f.rel, false)}>
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
                        switchDiffFile(f.rel, true);
                      }}>\u270E</button>` : null}
          </div>`)}
      </div>
      ${!D ? html`<div class="tip">${R.files.length ? 'Loading\u2026' : 'No files in this review.'}</div>` : (unified ? html`
        <div id="panes" class="unified">
          <div class="pane">
            <div class="pane-head"><span>${S.diffRel}</span><span>\u2212${D.removed} / +${D.added} lines</span></div>
            <div class="code" hidden=${S.editing}>${uniRows}</div>
            <textarea id="editBox" spellcheck="false" hidden=${!S.editing}
                      value=${editDraft}
                      onInput=${(e) => setEditDraft(e.target.value)}
                      ref=${(el) => { editBox.current = el; if (el && S.editing && el.value !== editDraft) el.value = editDraft; }}></textarea>
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
                      value=${editDraft}
                      onInput=${(e) => setEditDraft(e.target.value)}
                      ref=${(el) => { editBox.current = el; if (el && S.editing && el.value !== editDraft) el.value = editDraft; }}></textarea>
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
        ${j ? html`<button class="ghost small" disabled=${!!j.cancelled} onClick=${() => stop()}>${j.cancelled ? 'Stopping\u2026' : 'Stop'}</button>` : null}
      </div>
      <div id="activityBody">
        <${LiveStreamDrawer} />
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
// The bottom-right panel: Files / Backups / Logs, plus a 🌐 Preview tab that
// appears whenever the project has HTML pages (live, read-only iframe fed by
// /preview/). The #term element and the preview iframe stay mounted in every
// tab so the live Term buffer never resets and the page keeps its state.
const WS_TABS = [['files', 'Files'], ['backups', 'Backups'], ['logs', 'Logs']];

function WorkspacePane() {
  const s = S.status, j = s.job;
  const termRef = useRef(null);
  useEffect(() => {
    if (termRef.current && !term) term = new Term(termRef.current);
  }, []);
  const tabs = [WS_TABS[0], ['runner', '\u25B6 Run']]
    .concat(S.ws.pages.length ? [['preview', '\uD83C\uDF10 Preview']] : [])
    .concat(WS_TABS.slice(1));
  let tab = S.ws.tab;
  if (tab === 'preview' && !S.ws.pages.length) tab = 'runner';  // saved tab, no pages
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
        <div id="wsTabs" role="tablist">
          ${tabs.map(([k, lbl]) => html`
            <button key=${k} role="tab" aria-selected=${tab === k ? 'true' : 'false'}
                    class=${tab === k ? 'on' : ''}
                    onClick=${() => setWsTab(k)}>${lbl}</button>`)}
        </div>
        <span class="muted small-text grow ws-sub">
          ${tab === 'files' ? (s.root ? PDC.fmtInt(S.ws.indexed.length) + ' of ' + PDC.fmtInt(all.length) + ' files indexed' : 'no project open')
          : tab === 'backups' ? 'undo a Task/Edit or Auto write'
          : tab === 'runner' ? 'preview + one-click commands'
          : tab === 'preview' ? 'read-only \u00B7 reloads when a run finishes'
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
      <div class="ws-body" id="wsPreview" hidden=${tab !== 'preview'}>
        <div class="ws-tools">
          <select class="grow" title="page to preview" value=${S.ws.page}
                  onchange=${(e) => setPreviewPage(e.target.value)}>
            ${S.ws.pages.map((p) => html`<option key=${p} value=${p}>${p}</option>`)}
          </select>
          <button class="small" title="reload the page"
                  onClick=${() => loadPreview(true)}>\u27F3 Refresh</button>
          <button class="small" title="open this page in a new browser tab"
                  onClick=${openPreviewTab}>\u2197 New tab</button>
        </div>
        ${S.ws.page
          ? html`<iframe id="live-preview-frame" sandbox="allow-scripts"
                         title=${'Live preview of ' + S.ws.page}
                         src=${previewSrc(S.ws.page, S.ws.previewRev)}></iframe>`
          : html`<div class="tip">No HTML pages in this project yet \u2014 create one and it shows up here.</div>`}
      </div>
      <div class="ws-body" id="wsRunner" hidden=${tab !== 'runner'}>
        ${!s.root ? html`<div class="tip">Open a project to preview and run it.</div>`
        : (() => {
          const t = S.ws.targets || { web: [], commands: [] };
          const web = t.web || [];
          const cmds = t.commands || [];
          const out = S.ws.runOut;
          return html`
            <div class="runner-preview">
              <div class="ws-tools">
                <select class="grow" title="static page to preview" value=${S.ws.runPage}
                        disabled=${!web.length}
                        onchange=${(e) => setRunPage(e.target.value)}>
                  ${web.length ? web.map((p) => html`<option key=${p.path} value=${p.path}>${p.path}</option>`)
                    : html`<option value="">no HTML entrypoint</option>`}
                </select>
                <button class="small" disabled=${!S.ws.runPage}
                        title="reload the live preview" onClick=${reloadRunnerPreview}>\u27F3 Reload Preview</button>
                <button class="small" disabled=${!S.ws.runPage}
                        title="open this page in a browser tab" onClick=${openRunnerPreview}>\u2197 New tab</button>
              </div>
              ${S.ws.runPage
                ? html`<iframe class="runner-frame" sandbox="allow-scripts"
                               title=${'Live app preview of ' + S.ws.runPage}
                               src=${runnerSrc(S.ws.runPage, S.ws.runRev)}></iframe>`
                : html`<div class="tip">No <code>index.html</code>, <code>public/index.html</code> or <code>dist/index.html</code> yet \u2014 create one and it appears here live.</div>`}
            </div>
            <div class="runner-cmd">
              <select title="detected runnable target" value=${S.ws.runCmd}
                      onchange=${(e) => { S.ws.runCmd = e.target.value; emit(); }}>
                ${cmds.length ? cmds.map((c) => html`<option key=${c.id} value=${c.id}>${c.label}</option>`)
                  : html`<option value="">no entrypoint detected</option>`}
              </select>
              <input type="text" class="grow" spellcheck="false" value=${S.ws.runCustom}
                     placeholder="or type a command (argv, no shell)"
                     oninput=${(e) => { S.ws.runCustom = e.target.value; }} />
              <button class="small primary" disabled=${S.ws.runRunning}
                      onClick=${runProjectCommand}>${S.ws.runRunning ? 'Running\u2026' : '\u25B6 Run'}</button>
            </div>
            ${out ? html`
              <div class=${'runner-out' + (out.ok ? '' : ' failed')}>
                <div class="runner-out-head">
                  <b class=${out.ok ? 'ok' : 'bad'}>${out.ok ? 'OK' : 'FAILED'} \u00B7 exit ${out.exit_code}</b>
                  <span class="muted small-text ellip">${out.command} \u00B7 ${PDC.fmtInt(out.duration_ms)} ms</span>
                  <span class="grow"></span>
                  ${!out.ok ? html`<button class="small danger" title="put the command, exit code and traceback into the Task composer"
                        onClick=${fixRunnerCrash}>\u2691 Fix This Crash</button>` : null}
                </div>
                ${out.stdout ? html`<pre class="runner-pre">${out.stdout}</pre>` : null}
                ${out.stderr ? html`<pre class="runner-pre err">${out.stderr}</pre>` : null}
                ${out.traceback_summary && String(out.stderr || '').indexOf(out.traceback_summary) < 0
                  ? html`<pre class="runner-pre err">${out.traceback_summary}</pre>` : null}
              </div>` : null}
          `;
        })()}
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

// ---- Non-blocking Security Notice banner -----------------------------------
// Shown above the workspace whenever /api/state reports unacknowledged
// secrets or unignored sensitive files. It never blocks the composer: every
// action is an explicit HTTP call, and Dismiss hides it for this page session.
function SecurityBanner() {
  const sec = S.status.security || {};
  const findings = sec.findings || [];
  const files = sec.unignored_sensitive_files || [];
  if ((!findings.length && !files.length) || S.ui.securityDismissed) return null;
  const ack = (fingerprint) => {
    PDC.api('/api/security/ack', { fingerprints: [fingerprint] })
      .then(refreshState, () => PDC.toast('Could not acknowledge that finding.', true));
  };
  const ackAll = () => {
    PDC.api('/api/security/ack', { all: true })
      .then(refreshState, () => PDC.toast('Could not acknowledge findings.', true));
  };
  const gitignore = () => {
    PDC.api('/api/security/gitignore', {})
      .then((j) => { PDC.toast('Added ' + ((j.added || []).join(', ') || '.env') + ' to .gitignore.'); refreshState(); },
        () => PDC.toast('Could not update .gitignore.', true));
  };
  return html`
    <div id="securityNotice" role="status" aria-live="polite">
      <div class="security-head">
        <span>\u26A0 Security Notice</span>
        <span class="grow">Potential credentials are present in this workspace. Nothing was sent or changed without your choice.</span>
        <button class="ghost small" title="dismiss this notice for now"
                onClick=${() => { S.ui.securityDismissed = true; emit(); }}>Dismiss</button>
      </div>
      ${findings.length ? html`
        <ul class="security-list">
          ${findings.map((f) => html`
            <li key=${f.fingerprint} class="security-row">
              <code>${f.file}:${f.line}</code>
              <span class="mask">[${f.kind}] ${f.masked_preview}</span>
              <button class="small" title="ignore this exact key from now on"
                      onClick=${() => ack(f.fingerprint)}>Acknowledge (Ignore)</button>
            </li>`)}
        </ul>
        ${findings.length > 1 ? html`
          <div class="security-actions">
            <button class="small" onClick=${ackAll}>Acknowledge all (${findings.length})</button>
          </div>` : null}` : null}
      ${files.length ? html`
        <ul class="security-list">
          ${files.map((f) => html`
            <li key=${'file:' + f.file} class="security-row">
              <code>${f.file}</code>
              <span class="mask">${f.kind === 'unignored_env' ? 'environment secrets file' : 'credential file'} is not ignored by .gitignore</span>
            </li>`)}
        </ul>` : null}
      ${(sec.unignored_env || []).length ? html`
        <div class="security-actions">
          <button class="small primary" onClick=${gitignore}>Add .env to .gitignore</button>
        </div>` : null}
    </div>`;
}

// ---- Non-blocking Update notice (pedacito.update) ---------------------------
// A slim, dismissible banner when the startup check found a newer release on
// PyPI. Update streams the detected upgrade command into a small output
// drawer; exit 0 offers Restart, a failure (or an editable install) shows the
// output plus the manual command. In-flow like SecurityBanner: it pushes the
// workspace down, never covers a control, and never opens a modal.
function UpdateBanner() {
  const info = S.status.update;
  const run = S.status.update_run || {};
  const manual = S.ui.updateManual || '';
  const dismissed = S.ui.updateDismissed;
  const showBanner = !!(info && info.update_available
    && (!dismissed || isNewerVersion(info.latest, dismissed)));
  const showRun = !S.ui.updateRunHidden
    && !!(run.running || run.ok === true || run.ok === false || manual);
  if (!showBanner && !showRun) return null;
  const startUpgrade = () => {
    S.ui.updateManual = '';
    S.ui.updateRunHidden = false;
    PDC.api('/api/update/run', {}).then((j) => {
      if (j.ok) { S.ui.updateDrawerOpen = true; refreshState(); }
      else {
        // Editable install or opt-out: no command to run, show instructions.
        S.ui.updateManual = j.detail || 'Upgrade unavailable.';
        S.ui.updateDrawerOpen = true; emit();
      }
    }, () => PDC.toast('Could not start the upgrade.', true));
  };
  const restart = () => {
    PDC.api('/api/update/restart', {}).then((j) => {
      if (!j.ok) PDC.toast(j.detail || 'Could not restart.', true);
    });
  };
  const stopUpgrade = () => {
    PDC.api('/api/update/cancel', {}).then((j) => {
      if (!j.ok) PDC.toast(j.detail || 'Could not stop the upgrade.', true);
      refreshState();
    }, () => PDC.toast('Could not stop the upgrade.', true));
  };
  const runLabel = run.running ? 'Upgrading\u2026'
    : run.ok === true ? 'Upgrade finished'
    : run.ok === false ? 'Upgrade failed' : 'Upgrade';
  const runNote = run.running ? (run.command || []).join(' ')
    : run.ok === true ? 'Restart Pedacito to run the new version.'
    : run.ok === false ? (run.detail || 'See the output below.')
    : manual;
  return html`
    <div id="updateNotice" role="status" aria-live="polite">
      ${showBanner ? html`
        <div class="update-head">
          <span>\u2B06 Pedacito v${info.latest} is available</span>
          <span class="grow">(you have ${info.current})</span>
          <button class="small primary" disabled=${!!run.running}
                  title="run the detected upgrade command"
                  onClick=${startUpgrade}>Update</button>
          <button class="ghost small" title="hide this notice until a newer release"
                  onClick=${() => { S.ui.updateDismissed = info.latest; writeUpdateDismissed(info.latest); emit(); }}>Dismiss</button>
        </div>` : null}
      ${showRun ? html`
        <div class="update-run">
          <div class="update-head">
            <span>${runLabel}</span>
            <span class="grow">${runNote}</span>
            ${run.ok === true && !run.running ? html`
              <button class="small primary" onClick=${restart}>Restart Pedacito</button>` : null}
            ${run.running ? html`
              <button class="small danger" disabled=${!!run.cancelling}
                      title="stop the upgrade process and its children"
                      onClick=${stopUpgrade}>${run.cancelling ? 'Stopping\u2026' : 'Stop'}</button>` : null}
            <button class="ghost small"
                    onClick=${() => { S.ui.updateDrawerOpen = !S.ui.updateDrawerOpen; emit(); }}>
              ${S.ui.updateDrawerOpen ? 'Hide output' : 'Show output'}</button>
            ${!run.running ? html`
              <button class="ghost small" title="hide this upgrade panel"
                      onClick=${() => { S.ui.updateRunHidden = true; S.ui.updateManual = ''; emit(); }}>Close</button>` : null}
          </div>
          ${run.ok === false && info && info.instructions ? html`
            <p class="update-manual">${info.instructions}</p>` : null}
          ${S.ui.updateDrawerOpen ? html`
            <pre class="update-out">${run.output || manual || '(no output yet)'}</pre>` : null}
        </div>` : null}
    </div>`;
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
      <${SecurityBanner} />
      <${UpdateBanner} />
      <main id="app" ref=${appRef} data-diff=${dm}>
        <section class="panel" id="chatPanel">
          <div class="panel-head">
            <h2>${S.mode === 'how' ? 'How / Help' : 'Chat'}</h2>
            <span class="muted small-text grow" id="busyLbl">
              ${s.job ? (s.job.cancelled ? 'Stopping\u2026' : s.job.title + '\u2026') : ''}</span>
            <${LlmHeartbeat} />
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
      <nav id="mobileTabs" role="tablist">
        <button role="tab" aria-selected=${S.ui.view === 'chat' ? 'true' : 'false'}
                data-view="chat" class=${S.ui.view === 'chat' ? 'on' : ''} onClick=${() => setView('chat')}>Chat</button>
        <button role="tab" aria-selected=${S.ui.view === 'diff' ? 'true' : 'false'}
                data-view="diff" class=${S.ui.view === 'diff' ? 'on' : ''}
                disabled=${dm === 'none'} onClick=${() => { S.ui.diffOpen = true; setView('diff'); }}>Changes</button>
        <button role="tab" aria-selected=${S.ui.view === 'right' ? 'true' : 'false'}
                data-view="right" class=${S.ui.view === 'right' ? 'on' : ''} onClick=${() => setView('right')}>Activity</button>
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
function initViewState() {
  if (S.ui.view) document.body.dataset.view = S.ui.view;
}

render(html`<${App}/>`, $('root'));
initViewState();
loadActiveTheme();
loadSnapshot();
setInterval(() => { if (!document.hidden) refreshState(); }, 2500);
