// Chat, composer, header, live updates (SSE + polling), and layout.
(function () {
  'use strict';
  var PDC = window.PDC, h = PDC.h, $ = PDC.$, S = PDC.S;

  var MODES = {
    task: { label: 'Task', send: 'Run',
      hint: 'Describe a change. Pedacito looks things up, edits, and shows you the diff.',
      ph: 'e.g. make the sync client retry three times and update its callers\u2026' },
    explain: { label: 'Explain', send: 'Ask',
      hint: 'Ask how the code works. Read-only \u2014 nothing is written.',
      ph: 'e.g. how does the sync client retry?' },
    auto: { label: 'Auto', send: 'Start',
      hint: 'Give a goal. Pedacito plans a checklist, then implements it task by task, each in a fresh context.',
      ph: 'e.g. add CSV export to the report page, with tests' },
    'new': { label: 'New', send: 'Create',
      hint: 'Describe a project to start from scratch in a new folder.',
      ph: 'e.g. a CLI todo manager with sqlite storage and a --json flag' }
  };
  var OPTS = {
    task: [
      { k: 'write', label: 'Writing', type: 'select', def: 'review', opts: [['review', 'Review before writing'], ['apply', 'Write directly (backup first)']] },
      { k: 'plan', label: 'Planning', type: 'select', def: 'auto', opts: [['auto', 'Automatic'], ['always', 'Always plan'], ['never', 'Never']], note: 'Planning runs only with direct writes.' },
      { k: 'steps', label: 'Max lookups (0 = config)', type: 'number', def: '' },
      { k: 'reindex', label: 'Refresh index first', type: 'check', def: false }
    ],
    explain: [],
    auto: [
      { k: 'run', label: 'Run', type: 'select', def: 'plan', opts: [['plan', 'Write the plan only (review it first)'], ['apply', 'Plan and implement it']], span2: true },
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
    ]
  };

  S.opts = {};
  try { S.opts = JSON.parse(localStorage.getItem('pedacito_opts') || '{}'); } catch (e) { S.opts = {}; }
  S.mode = localStorage.getItem('pedacito_mode') || 'task';
  if (!MODES[S.mode]) S.mode = 'task';
  function opt(mode, k) { var o = (S.opts[mode] = S.opts[mode] || {}); if (!(k in o)) { var d = OPTS[mode].filter(function (x) { return x.k === k; })[0]; o[k] = d ? d.def : ''; } return o[k]; }
  function saveOpts() { try { localStorage.setItem('pedacito_opts', JSON.stringify(S.opts)); } catch (e) {} }

  // ------------------------------------------------------------- state / header ---
  var refreshTimer = 0, lastErr = '', lastJob = null, lastReviewKey = '';
  PDC.refreshState = function () {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(function () { PDC.api('/api/state').then(applyStatus); }, 60);
  };

  function applyStatus(st) {
    var prev = S.status; S.status = st;
    var job = st.job;
    // header
    var path = $('path');
    if (document.activeElement !== path && st.root) path.value = st.root;
    fillSelect($('recent'), [['', 'recent\u2026']].concat((st.recent || []).map(function (r) { return [r, r]; })), '');
    if (st.root) fillSelect($('profile'), [['', '(default' + (st.default_profile ? ': ' + st.default_profile : '') + ')']].concat((st.profiles || []).map(function (p) { return [p, p]; })), st.profile === st.default_profile ? '' : st.profile);
    $('modelLbl').textContent = st.root ? (st.model || '\u2014') : 'no project';
    var dot = $('modelDot');
    dot.className = 'dot ' + (job ? 'busy' : st.model_ok === true ? 'ok' : st.model_ok === false ? 'bad' : '');
    $('modelChip').title = (st.model_status || 'click to check the model server') + (st.base_url ? '\n' + st.base_url : '') + (st.context_tokens ? '\nwindow ' + PDC.fmtInt(st.context_tokens) + ' tokens (' + st.context_origin + ')' : '');
    var u = st.usage || {};
    $('usageChip').hidden = !u.calls;
    if (u.calls) {
      $('usageChip').textContent = u.calls + ' calls \u00B7 ' + PDC.fmtTokens(u.tokens_in) + ' in / ' + PDC.fmtTokens(u.tokens_out) + ' out' + (u.cache_hit ? ' \u00B7 cache ' + Math.round(u.cache_hit * 100) + '%' : '') + (u.cost ? ' \u00B7 $' + u.cost.toFixed(4) : '');
      $('usageChip').title = 'model usage this session (' + Math.round(u.seconds) + 's)';
    }
    $('busyLbl').textContent = job ? (job.cancelled ? 'stopping\u2026' : job.title + '\u2026') : '';
    $('openBtn').disabled = !!job; $('profile').disabled = !!job;
    if (st.error && st.error !== lastErr) PDC.toast(st.error.split('\n')[0], true);
    lastErr = st.error || '';
    renderComposerState();
    PDC.loadFiles(st.files_sig);
    // review / plan follow-ups
    var rkey = st.review ? st.review.id + ':' + st.review.outcome : '';
    if (rkey !== lastReviewKey) { lastReviewKey = rkey; PDC.loadReview(); }
    if (job || (prev.job && !job)) { if (S.tab === 'plan' || (job && (job.kind === 'auto' || job.kind === 'task'))) PDC.loadPlan(); }
    if (prev.job && !job) { collapseDone(); }
    if (prev.has_index !== undefined && prev.has_index !== st.has_index) {
      Array.prototype.forEach.call(msgsEl.querySelectorAll('.msg.opened'), function (el) { fillMsg(el, S.msgs[Number(el.dataset.id)]); });
    }
    if (st.needs_setup && !sessionStorage.getItem('pedacito_setup_shown')) {
      sessionStorage.setItem('pedacito_setup_shown', '1');
      PDC.openSettings();
    }
    lastJob = job;
  }
  function fillSelect(sel, opts, value) {
    var key = JSON.stringify(opts) + '|' + value;
    if (sel.dataset.key === key) return;
    sel.dataset.key = key;
    PDC.clear(sel);
    opts.forEach(function (o) { sel.appendChild(h('option', { value: o[0], text: o[1] })); });
    sel.value = value;
  }

  function openProject() {
    var path = $('path').value.trim();
    if (!path) { PDC.toast('Enter a project folder (or use Browse).', true); return; }
    PDC.api('/api/project', { path: path, profile: $('profile').value }).then(function (j) {
      if (!j.ok) PDC.toast(j.detail || 'busy', true);
      PDC.refreshState();
    });
  }
  var peekTimer = 0, peeked = '';
  function peek() {
    clearTimeout(peekTimer);
    peekTimer = setTimeout(function () {
      var path = $('path').value.trim();
      if (!path || path === peeked || path === S.status.root) return;
      PDC.api('/api/peek', { path: path }).then(function (j) {
        if (!j.ok) return;
        peeked = path;
        fillSelect($('profile'), [['', '(default' + (j.default ? ': ' + j.default : '') + ')']].concat(j.profiles.map(function (p) { return [p, p]; })), '');
      });
    }, 450);
  }

  // ----------------------------------------------------------------------- chat ---
  var msgsEl, stick = true;
  function nearBottom() { return msgsEl.scrollHeight - msgsEl.scrollTop - msgsEl.clientHeight < 90; }
  function scrollDown() { if (stick) msgsEl.scrollTop = msgsEl.scrollHeight; }

  function jobActive(id) { return S.curJob === id || !!(S.status.job && S.status.job.id === id); }

  function buildMsg(m) {
    var el = h('div', { dataset: { id: m.id } });
    fillMsg(el, m);
    return el;
  }
  function fillMsg(el, m) {
    PDC.clear(el);
    el.className = 'msg ' + m.role + (m.kind ? ' ' + m.kind : '');
    var r = m.role;
    if (r === 'user') {
      el.appendChild(h('div', { class: 'bubble' },
        h('div', { class: 'tag', text: m.kind === 'steer' ? 'note to the model' : (m.mode || 'task') }), m.text,
        (m.attachments && m.attachments.length) ? h('div', { class: 'chiprow' }, m.attachments.map(function (a) { return h('span', { class: 'fchip', text: '\uD83D\uDCCE ' + a.name + ' (' + PDC.fmtInt(a.chars) + ' chars)' }); })) : null));
    } else if (r === 'assistant') {
      var b = h('div', { class: 'bubble' });
      if (m.head) b.appendChild(h('div', { class: 'head ' + (m.level || ''), text: m.head }));
      if (m.text) b.appendChild(PDC.md(m.text));
      if (m.results && m.results.length) b.appendChild(h('ul', { class: 'results' }, m.results.map(function (x) { return h('li', { class: x.ok ? (/^not written/.test(x.message || '') ? 'skip' : 'ok') : 'no', text: x.file + ' \u2014 ' + x.message }); })));
      if (m.kind === 'plan_ready') {
        b.appendChild(h('div', { class: 'row' },
          h('button', { class: 'small', text: 'Open plan', onclick: function () { PDC.showTab('plan'); } }),
          h('button', { class: 'small primary', text: 'Implement it now', onclick: implementPlan })));
      }
      if (m.stats && m.stats.calls) b.appendChild(h('div', { class: 'footer', text: m.stats.calls + ' calls \u00B7 ' + PDC.fmtTokens(m.stats.tokens_in) + ' in / ' + PDC.fmtTokens(m.stats.tokens_out) + ' out' + (m.stats.cost ? ' \u00B7 $' + Number(m.stats.cost).toFixed(4) : '') + (m.stats.seconds ? ' \u00B7 ' + Math.round(m.stats.seconds) + 's' : '') }));
      el.appendChild(b);
    } else if (r === 'stream') {
      var d = h('details', { class: 'stream', open: jobActive(m.job) }, h('summary', { text: 'Model output \u00B7 ' + (m.site || 'answer') }), h('pre', { text: m.text }));
      el.appendChild(d);
    } else if (r === 'activity') {
      var lines = m.text ? m.text.split('\n').length : 0;
      el.appendChild(h('details', { class: 'activity', open: jobActive(m.job) }, h('summary', { text: 'Lookups \u00B7 ' + lines }), h('pre', { text: m.text })));
    } else if (r === 'error') {
      el.appendChild(h('div', { class: 'bubble', text: m.text }));
    } else if (r === 'question') {
      el.appendChild(questionCard(m));
    } else if (r === 'review') {
      el.appendChild(reviewCard(m));
    } else {
      el.appendChild(h('div', { text: m.text }));
      if (m.kind === 'opened' && m.needs_index && !S.status.has_index) {
        el.appendChild(h('div', { class: 'row', style: 'justify-content:center;margin-top:6px' },
          h('button', { class: 'small primary', text: 'Build index (with summaries)', onclick: function () { buildIndex(true); } }),
          h('button', { class: 'small', text: 'Quick index (no summaries)', onclick: function () { buildIndex(false); } })));
      }
    }
  }
  function questionCard(m) {
    var card = h('div', { class: 'card' + (m.answered ? ' done' : '') }, h('div', { class: 'title', text: 'The model asks' }), h('div', { text: m.text, style: 'white-space:pre-wrap' }));
    if (m.answered) { card.appendChild(h('div', { class: 'muted small-text', style: 'margin-top:4px', text: m.answer ? 'Your answer: ' + m.answer : 'No answer \u2014 the model chose.' })); return card; }
    var inp = h('input', { type: 'text', class: 'grow', placeholder: 'your answer\u2026' });
    var send = function (t) { PDC.api('/api/answer', { qid: m.qid, text: t }); };
    inp.addEventListener('keydown', function (e) { if (e.key === 'Enter') send(inp.value); });
    card.appendChild(h('div', { class: 'row' }, inp, h('button', { class: 'small primary', text: 'Answer', onclick: function () { send(inp.value); } }),
      h('button', { class: 'small', text: 'Let it choose', onclick: function () { send(''); } })));
    setTimeout(function () { inp.focus(); }, 30);
    return card;
  }
  function reviewCard(m) {
    var pending = !m.outcome;
    var card = h('div', { class: 'card ' + (pending ? '' : (m.outcome === 'applied' ? 'applied' : 'rejected')) },
      h('div', { class: 'title', text: pending ? 'Review ready \u2014 ' + m.files + ' file' + (m.files === 1 ? '' : 's') : m.outcome === 'applied' ? 'Changes accepted' : m.outcome === 'timeout' ? 'Review timed out' : 'Changes rejected' }),
      h('div', { class: 'muted small-text', text: m.detail || m.text }));
    if (pending) card.appendChild(h('div', { class: 'row' }, h('button', { class: 'small primary', text: 'Open review', onclick: function () { PDC.showTab('changes'); } })));
    return card;
  }
  function collapseDone() {
    Array.prototype.forEach.call(msgsEl.querySelectorAll('details.stream, details.activity'), function (d) {
      var id = Number(d.parentNode.dataset.id), m = S.msgs[id];
      if (m && !jobActive(m.job)) d.open = false;
    });
  }
  function implementPlan() { PDC.runPlan(); }

  function addMsg(m) {
    if (S.msgs[m.id]) return;
    S.msgs[m.id] = m; S.order.push(m.id);
    var hello = msgsEl.querySelector('.hello'); if (hello) hello.remove();
    msgsEl.appendChild(buildMsg(m));
    scrollDown();
  }
  function msgEl(id) { return msgsEl.querySelector('[data-id="' + id + '"]'); }
  function resetMsgs(list) {
    S.msgs = {}; S.order = []; PDC.clear(msgsEl);
    if (!list.length) { msgsEl.appendChild(hello()); return; }
    list.forEach(function (m) { S.msgs[m.id] = m; S.order.push(m.id); msgsEl.appendChild(buildMsg(m)); });
    stick = true; scrollDown();
  }
  function hello() {
    var st = S.status;
    return h('div', { class: 'hello' }, h('h3', { text: 'Welcome to pedacito' }),
      h('div', { text: 'A small local model, working on a codebase bigger than its window.' }),
      h('ul', {}, h('li', {}, h('b', { text: 'Open a project' }), ' above (Browse\u2026 helps).'),
        h('li', {}, h('b', { text: 'Task' }), ' edits code and shows the diff to accept or reject.'),
        h('li', {}, h('b', { text: 'Explain' }), ' answers questions, read-only.'),
        h('li', {}, h('b', { text: 'Auto' }), ' plans a goal and works through it.'),
        h('li', {}, h('b', { text: 'New' }), ' starts a project from scratch.')),
      st.needs_setup ? h('button', { class: 'primary', text: 'Set up your model server', onclick: function () { PDC.openSettings(); } }) : null);
  }

  // -------------------------------------------------------------------- live events ---
  function connect(seq) {
    var es = new EventSource('/events?since=' + seq);
    var on = function (type, fn) { es.addEventListener(type, function (e) { fn(JSON.parse(e.data)); }); };
    on('msg_new', function (e) { addMsg(e.msg); });
    on('msg_append', function (e) {
      var m = S.msgs[e.id]; if (!m) return;
      m.text += e.text;
      var el = msgEl(e.id); if (!el) return;
      var pre = el.querySelector('pre');
      if (pre) { pre.append(e.text); pre.scrollTop = pre.scrollHeight;
        if (m.role === 'activity') el.querySelector('summary').textContent = 'Lookups \u00B7 ' + m.text.split('\n').length; }
      else fillMsg(el, m);
      scrollDown();
    });
    on('msg_set', function (e) {
      var m = S.msgs[e.id]; if (!m) return;
      Object.assign(m, e.fields);
      var el = msgEl(e.id); if (el) fillMsg(el, m);
      scrollDown();
    });
    on('msgs_clear', function () { resetMsgs([]); });
    on('term', function (e) { term.write(e.text); });
    on('term_clear', function () { term.reset(''); });
    on('usage', function () { PDC.refreshState(); });
    on('job', function (e) { S.curJob = e.state === 'start' ? e.job : (e.state === 'end' ? 0 : S.curJob); PDC.refreshState(); });
    ['status', 'review', 'review_done'].forEach(function (t) { on(t, function () { PDC.refreshState(); }); });
    es.addEventListener('resync', function () { es.close(); load(); });
  }
  var term, es0;
  function load() {
    PDC.api('/api/snapshot').then(function (snap) {
      S.status = snap.status; S.curJob = snap.status.job ? snap.status.job.id : 0; resetMsgs(snap.messages); term.reset(snap.term);
      applyStatus(snap.status);
      connect(snap.seq);
    });
  }

  // --------------------------------------------------------------------- composer ---
  function renderModeTabs() {
    var tabs = PDC.clear($('modeTabs'));
    Object.keys(MODES).forEach(function (k) {
      tabs.appendChild(h('button', { class: k === S.mode ? 'on' : '', text: MODES[k].label, onclick: function () { S.mode = k; localStorage.setItem('pedacito_mode', k); $('optsBox').hidden = true; renderModeTabs(); renderOpts(); renderComposerState(); } }));
    });
    $('input').placeholder = MODES[S.mode].ph;
    $('optsBtn').hidden = !OPTS[S.mode].length;
  }
  function renderOpts() {
    var box = PDC.clear($('optsBox'));
    OPTS[S.mode].forEach(function (o) {
      var val = opt(S.mode, o.k), el;
      var set = function (v) { S.opts[S.mode][o.k] = v; saveOpts(); renderComposerState(); };
      if (o.type === 'select') {
        el = h('select', {}, o.opts.map(function (x) { return h('option', { value: x[0], text: x[1] }); }));
        el.value = val; el.addEventListener('change', function () { set(el.value); if (S.mode === 'task') renderOpts(); });
      } else if (o.type === 'check') {
        el = h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: !!val, onchange: function (e) { set(e.target.checked); } }), o.label);
        box.appendChild(el); return;
      } else {
        var field = h('input', { type: o.type === 'number' ? 'number' : 'text', value: val, placeholder: o.ph || '', min: o.type === 'number' ? '0' : null });
        field.addEventListener('input', function () { set(field.value); });
        el = field;
        if (o.browse) {
          el = h('div', { class: 'row', style: 'flex-wrap:nowrap' }, h('div', { class: 'grow' }, field),
            h('button', { class: 'small', text: 'Browse\u2026', onclick: function () {
              PDC.browse({ title: o.browse === 'file' ? 'Choose a plan file' : 'Choose the parent folder', files: o.browse === 'file', pickLabel: 'Use this folder',
                start: S.status.root || '', onPick: function (p) {
                  if (o.browse === 'dir') { var name = (field.value.split(/[\\/]/).pop()) || 'my-project'; p = p.replace(/[\\/]$/, '') + '/' + name; }
                  field.value = p; set(p); } });
            } }));
        }
      }
      var lab = h('label', { class: 'f' + (o.span2 ? ' span2' : ''), text: o.label }, el);
      if (o.note && S.mode === 'task' && S.opts.task.write === 'review') lab.appendChild(h('span', { class: 'muted', text: o.note }));
      box.appendChild(lab);
    });
  }

  function renderComposerState() {
    var st = S.status, job = st.job, m = MODES[S.mode];
    var needIndex = st.root && !st.has_index && S.mode !== 'new';
    var hint = $('modeHint'); PDC.clear(hint);
    if (!st.root && S.mode !== 'new') hint.append('Open a project first. ');
    if (needIndex) {
      hint.append('No index yet \u2014 build one to start: ',
        h('button', { class: 'small primary', text: 'Build index', disabled: !!job, onclick: function () { buildIndex(true); } }), ' ',
        h('button', { class: 'small', text: 'Quick (no summaries)', disabled: !!job, onclick: function () { buildIndex(false); } }));
    } else hint.append(m.hint);
    var send = $('sendBtn'), input = $('input');
    var steerable = job && job.kind === 'task' && st.allow_steering;
    if (job) {
      send.textContent = job.cancelled ? 'Stopping\u2026' : 'Stop'; send.className = 'danger'; send.disabled = !!job.cancelled;
      input.placeholder = steerable ? 'Type a note to steer the model (Enter)\u2026' : 'Working\u2026 press Stop to cancel.';
      input.disabled = !steerable;
      $('sendHint').textContent = steerable ? 'delivered at the next lookup step' : '';
    } else {
      send.textContent = S.mode === 'auto' && opt('auto', 'run') === 'apply' ? 'Plan & run' : m.send; send.className = 'primary'; send.disabled = false;
      input.disabled = false; input.placeholder = m.ph;
      $('sendHint').textContent = ''; input.title = 'Enter sends \u00B7 Shift+Enter adds a line';
    }
    $('attachBtn').disabled = !!job;
  }

  function buildIndex(summarise) {
    PDC.api('/api/index', { summarise: summarise }).then(function (j) { if (!j.ok) PDC.toast(j.detail || 'busy', true); PDC.refreshState(); });
  }

  PDC.startRun = function (payload) {
    return PDC.api('/api/run', payload).then(function (j) {
      if (!j.ok) PDC.toast(j.detail || 'could not start', true);
      PDC.refreshState();
      return j;
    });
  };

  function send() {
    var st = S.status, job = st.job;
    var text = $('input').value.trim();
    if (job) {
      if (!text) return;
      if (job.kind === 'task' && st.allow_steering) {
        PDC.api('/api/steer', { text: text }).then(function (j) { if (!j.ok) PDC.toast(j.detail || 'not delivered', true); });
        $('input').value = '';
      }
      return;
    }
    if (!text && !S.attachments.length && !(S.mode === 'auto' && opt('auto', 'plan_file'))) { PDC.toast('Type something first.', true); return; }
    if (S.mode !== 'new' && !st.root) { PDC.toast('Open a project first.', true); return; }
    if (S.mode !== 'new' && !st.has_index) { PDC.toast('Build the index first.', true); return; }
    var body = { mode: S.mode, text: text, attachments: S.attachments.slice() };
    var o = S.opts[S.mode] || {};
    OPTS[S.mode].forEach(function (d) { body[d.k] = opt(S.mode, d.k); });
    if (S.mode === 'auto') body.apply = body.run === 'apply';
    PDC.startRun(body).then(function (j) {
      if (j.ok) { $('input').value = ''; S.attachments = []; renderAttach(); stick = true; }
    });
  }

  function stop() { PDC.api('/api/stop', {}).then(function () { PDC.refreshState(); }); }

  // attachments
  function renderAttach() {
    var row = PDC.clear($('attachRow'));
    S.attachments.forEach(function (a, i) {
      row.appendChild(h('span', { class: 'attach' }, '\uD83D\uDCCE ' + a.name + ' \u00B7 ' + PDC.fmtInt(a.text.length) + ' chars',
        h('button', { text: '\u2715', title: 'remove', onclick: function () { S.attachments.splice(i, 1); renderAttach(); } })));
    });
  }
  function attachFiles(files) {
    Array.prototype.forEach.call(files, function (f) {
      if (f.size > 400000) { PDC.toast(f.name + ' is larger than 400 KB.', true); return; }
      f.text().then(function (t) {
        if (t.indexOf('\u0000') >= 0) { PDC.toast(f.name + ' looks binary.', true); return; }
        S.attachments.push({ name: f.name, text: t }); renderAttach();
      });
    });
  }

  // ------------------------------------------------------------------ project menu ---
  function projectMenu() {
    var old = document.getElementById('projMenu'); if (old) { old.remove(); return; }
    var st = S.status, job = !!st.job;
    var rect = $('projMenuBtn').getBoundingClientRect();
    var item = function (label, fn, disabled) { return h('button', { text: label, disabled: !!disabled, style: 'display:block;width:100%;text-align:left;border:none;border-radius:0', onclick: function () { menu.remove(); fn(); } }); };
    var menu = h('div', { id: 'projMenu', style: 'position:fixed;z-index:60;background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:0 10px 30px rgba(0,0,0,.25);min-width:260px;overflow:hidden;top:' + (rect.bottom + 4) + 'px;right:' + Math.max(8, window.innerWidth - rect.right) + 'px' },
      item('Build index (with summaries)', function () { buildIndex(true); }, job || !st.root),
      item('Quick index (no summaries)', function () { buildIndex(false); }, job || !st.root),
      item('Check model (loads it)', function () { PDC.api('/api/check_model', {}).then(PDC.refreshState); }, job || !st.root),
      item('Restore from a backup\u2026', restoreDialog, job || !(st.snapshots || []).length),
      item(S.filesHidden ? 'Show files panel' : 'Hide files panel', function () { $('app').classList.toggle('no-files'); S.filesHidden = !S.filesHidden; }),
      item('Clear terminal', function () { PDC.api('/api/term_clear', {}); }));
    document.body.appendChild(menu);
    setTimeout(function () { document.addEventListener('mousedown', function close(e) { if (!menu.contains(e.target)) { menu.remove(); document.removeEventListener('mousedown', close); } }); }, 0);
  }
  function restoreDialog() {
    var snaps = S.status.snapshots || [];
    var sel = h('select', {}, snaps.map(function (n, i) { return h('option', { value: n, text: n + (i === 0 ? '  (newest)' : '') }); }));
    PDC.modal('Restore a backup', h('div', {}, h('p', { text: 'Copies the chosen snapshot back over the working tree. Files changed since then are overwritten; files created since stay on disk.' }), sel), [
      { label: 'Cancel' },
      { label: 'Restore', danger: true, onclick: function () {
        PDC.api('/api/restore', { snapshot: sel.value }).then(function (j) { PDC.toast(j.detail || (j.ok ? 'restored' : 'failed'), !j.ok); PDC.refreshState(); PDC.api('/api/files'); });
      } }], { narrow: true });
  }

  // ----------------------------------------------------------------------- layout ---
  function splitter(el, onmove) {
    el.addEventListener('mousedown', function (e) {
      e.preventDefault(); el.classList.add('drag'); document.body.style.userSelect = 'none';
      var move = function (ev) { onmove(ev); };
      var up = function () { el.classList.remove('drag'); document.body.style.userSelect = ''; document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); saveLayout(); };
      document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
    });
  }
  function saveLayout() {
    var cs = getComputedStyle(document.documentElement);
    try { localStorage.setItem('pedacito_layout', JSON.stringify({ left: parseInt(cs.getPropertyValue('--left')), right: parseInt(cs.getPropertyValue('--right')), term: parseInt(cs.getPropertyValue('--term-h')) })); } catch (e) {}
  }
  PDC.setView = function (v) {
    document.body.dataset.view = v;
    Array.prototype.forEach.call($('mobileTabs').children, function (b) { b.classList.toggle('on', b.dataset.view === v); });
  };

  // ------------------------------------------------------------------------ boot ---
  document.addEventListener('DOMContentLoaded', function () {
    msgsEl = $('messages');
    term = new PDC.Term($('term'));
    msgsEl.addEventListener('scroll', function () { stick = nearBottom(); });
    renderModeTabs(); renderOpts(); renderComposerState();

    $('openBtn').addEventListener('click', openProject);
    $('path').addEventListener('keydown', function (e) { if (e.key === 'Enter') openProject(); });
    $('path').addEventListener('input', peek);
    $('browseBtn').addEventListener('click', function () {
      PDC.browse({ title: 'Choose a project folder', start: ($('path').value.trim() || S.status.root || '').replace(/[\\/][^\\/]*[\\/]?$/, ''), pickLabel: 'Open this folder', onPick: function (p) { $('path').value = p; peek(); openProject(); } });
    });
    $('recent').addEventListener('change', function () { if (this.value) { $('path').value = this.value; peek(); openProject(); } });
    $('profile').addEventListener('change', function () { if (S.status.root) openProject(); });
    $('modelChip').addEventListener('click', function () { if (S.status.root && !S.status.job) PDC.api('/api/check_model', {}).then(PDC.refreshState); });
    $('projMenuBtn').addEventListener('click', projectMenu);
    $('settingsBtn').addEventListener('click', function () { PDC.openSettings(); });
    $('clearChat').addEventListener('click', function () { PDC.api('/api/clear', {}); });

    $('sendBtn').addEventListener('click', function () { if (S.status.job) stop(); else send(); });
    $('input').addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
    });
    $('optsBtn').addEventListener('click', function () { var b = $('optsBox'); b.hidden = !b.hidden; });
    $('attachBtn').addEventListener('click', function () { $('filePick').click(); });
    $('filePick').addEventListener('change', function () { attachFiles(this.files); this.value = ''; });
    var comp = $('composer');
    ['dragenter', 'dragover'].forEach(function (t) { comp.addEventListener(t, function (e) { e.preventDefault(); comp.classList.add('drop'); }); });
    ['dragleave', 'drop'].forEach(function (t) { comp.addEventListener(t, function (e) { e.preventDefault(); comp.classList.remove('drop'); }); });
    comp.addEventListener('drop', function (e) { if (e.dataTransfer && e.dataTransfer.files.length) attachFiles(e.dataTransfer.files); });

    $('termFollow').addEventListener('change', function () { term.follow = this.checked; if (this.checked) term.schedule(); });
    $('termClear').addEventListener('click', function () { PDC.api('/api/term_clear', {}); });
    $('termToggle').addEventListener('click', function () { $('termPanel').classList.toggle('collapsed'); });
    $('term').addEventListener('wheel', function () { if (term.follow && $('term').scrollTop + $('term').clientHeight < $('term').scrollHeight - 40) { term.follow = false; $('termFollow').checked = false; } });

    var themeSel = $('theme');
    themeSel.value = localStorage.getItem('pedacito_theme') || 'parchment';
    themeSel.addEventListener('change', function () {
      if (this.value === 'parchment') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', this.value);
      try { localStorage.setItem('pedacito_theme', this.value); } catch (e) {}
    });
    $('exitBtn').addEventListener('click', function () {
      PDC.confirm('Stop the pedacito gui server? Run "pedacito gui" again to reopen it.', 'Stop server', true).then(function (ok) {
        if (!ok) return;
        PDC.api('/api/shutdown', {}).catch(function () {}).then(function () {
          document.body.innerHTML = '<div style="padding:80px 20px;text-align:center;font:16px sans-serif;color:#86816f">pedacito gui has stopped. You can close this tab.</div>';
        });
      });
    });

    var app = $('app');
    splitter($('split1'), function (e) { var r = app.getBoundingClientRect(); document.documentElement.style.setProperty('--left', Math.max(280, Math.min(e.clientX - r.left, r.width * 0.6)) + 'px'); });
    splitter($('split3'), function (e) { var r = app.getBoundingClientRect(); document.documentElement.style.setProperty('--right', Math.max(150, Math.min(r.right - e.clientX, 520)) + 'px'); });
    splitter($('split2'), function (e) { var r = $('workPanel').getBoundingClientRect(); document.documentElement.style.setProperty('--term-h', Math.max(40, Math.min(r.bottom - e.clientY, r.height * 0.75)) + 'px'); });
    Array.prototype.forEach.call($('mobileTabs').children, function (b) { b.addEventListener('click', function () { PDC.setView(b.dataset.view); }); });

    load();
    setInterval(function () { if (!document.hidden) PDC.refreshState(); }, 2500);
  });
})();
