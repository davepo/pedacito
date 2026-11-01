// The work area: proposed changes (old | updated), the plan, the file viewer,
// the file tree, and the folder browser.
(function () {
  'use strict';
  var PDC = window.PDC, h = PDC.h, $ = PDC.$;

  // ---------------------------------------------------------------- tabs -----
  var TABS = ['changes', 'plan', 'file'];
  PDC.showTab = function (name, quiet) {
    TABS.forEach(function (t) { $('tab-' + t).hidden = t !== name; });
    Array.prototype.forEach.call($('codeTabs').children, function (b) { b.classList.toggle('on', b.dataset.tab === name); });
    PDC.S.tab = name;
    if (name === 'plan') PDC.loadPlan();
    if (!quiet && document.body.dataset.view !== 'work' && window.innerWidth <= 980) PDC.setView('work');
  };

  // ---------------------------------------------------------- code lines -----
  function marked(text, spans) {
    var frag = document.createDocumentFragment();
    if (!spans || !spans.length) { frag.append(text); return frag; }
    var pos = 0;
    spans.forEach(function (s) {
      if (s[0] > pos) frag.append(text.slice(pos, s[0]));
      frag.append(h('mark', { text: text.slice(s[0], s[1]) }));
      pos = s[1];
    });
    if (pos < text.length) frag.append(text.slice(pos));
    return frag;
  }
  function line(cls, no, content, fold) {
    var el = h('div', { class: 'l ' + cls }, h('span', { class: 'n', text: no == null ? '' : String(no) }),
      h('span', { class: 't' }, content));
    if (fold) { el.dataset.fold = fold; el.hidden = true; }
    return el;
  }
  function skipRow(n, fold) {
    return h('div', {
      class: 'skip', dataset: { skip: fold }, text: '\u22EF ' + n + ' unchanged line' + (n === 1 ? '' : 's') + ' \u2014 click to show',
      onclick: function () {
        var host = $('panes');
        Array.prototype.forEach.call(host.querySelectorAll('[data-fold="' + fold + '"]'), function (e) { e.hidden = false; });
        Array.prototype.forEach.call(host.querySelectorAll('[data-skip="' + fold + '"]'), function (e) { e.remove(); });
      }
    });
  }
  function renderRows(rows) {
    var oldF = document.createDocumentFragment(), newF = document.createDocumentFragment();
    rows.forEach(function (r) {
      if (r[0] === 'skip') { oldF.append(skipRow(r[1], r[2])); newF.append(skipRow(r[1], r[2])); return; }
      var kind = r[0], fold = r[7];
      if (kind === 'equal') {
        oldF.append(line('equal', r[1], r[2], fold)); newF.append(line('equal', r[3], r[4], fold));
      } else if (kind === 'replace') {
        oldF.append(line('replace del', r[1], marked(r[2], r[5]))); newF.append(line('replace add', r[3], marked(r[4], r[6])));
      } else if (kind === 'delete') {
        oldF.append(line('delete', r[1], r[2])); newF.append(line('blank', null, ''));
      } else {
        oldF.append(line('blank', null, '')); newF.append(line('insert', r[3], r[4]));
      }
    });
    return [oldF, newF];
  }

  // -------------------------------------------------------------- review -----
  var R = { meta: null, cur: '', sel: {}, data: null, editing: false, seenId: 0, seenOutcome: '' };

  PDC.loadReview = function () {
    return PDC.api('/api/review').then(function (j) {
      var meta = j.review || null;
      var changed = (meta ? meta.id : 0) !== R.seenId || (meta ? meta.outcome : '') !== R.seenOutcome;
      R.meta = meta;
      if (!changed) return;
      R.seenId = meta ? meta.id : 0; R.seenOutcome = meta ? meta.outcome : '';
      R.editing = false;
      var newReview = !R.cur || !meta || !meta.files.some(function (f) { return f.rel === R.cur; });
      if (meta) {
        meta.files.forEach(function (f) { if (!(f.rel in R.sel)) R.sel[f.rel] = true; });
        if (newReview) R.cur = (meta.files.filter(function (f) { return !f.source; })[0] || meta.files[0] || {}).rel || '';
      }
      renderReviewChrome();
      if (R.cur) loadReviewFile(R.cur); else renderPanes(null);
      if (meta && meta.pending) PDC.showTab('changes');
      PDC.markChanged(meta ? meta.files.map(function (f) { return f.rel; }) : []);
    });
  };

  function renderReviewChrome() {
    var m = R.meta;
    $('reviewBar').hidden = !m; $('fileStrip').hidden = !m; $('panes').hidden = !m; $('noReview').hidden = !!m;
    var badge = $('chgBadge');
    badge.hidden = !(m && m.pending);
    if (m && m.pending) badge.textContent = String(m.files.filter(function (f) { return !f.source; }).length);
    if (!m) return;
    $('reviewTitle').textContent = m.task || 'Proposed changes';
    $('reviewTitle').title = m.task || '';
    var out = $('reviewOutcome');
    out.hidden = m.pending; out.className = 'badge-out ' + m.outcome;
    out.textContent = m.outcome === 'applied' ? 'accepted' : m.outcome === 'cancelled' ? 'rejected' : m.outcome;
    out.title = m.detail || '';
    var pending = m.pending;
    $('acceptBtn').hidden = !pending; $('rejectBtn').hidden = !pending;
    $('editBtn').hidden = !(pending && m.partial_ok);
    $('rejectBtn').textContent = 'Reject' + (m.files.length > 1 ? ' all' : '');
    updateAcceptLabel();
    var strip = PDC.clear($('fileStrip'));
    m.files.forEach(function (f) {
      var item = h('div', { class: 'fileitem' + (f.rel === R.cur ? ' on' : '') + (f.source ? ' source' : ''), title: f.message || '' });
      if (pending && m.partial_ok && !f.source) {
        item.appendChild(h('input', {
          type: 'checkbox', checked: R.sel[f.rel] !== false, title: 'include in Accept',
          onclick: function (e) { e.stopPropagation(); R.sel[f.rel] = e.target.checked; updateAcceptLabel(); }
        }));
      }
      item.appendChild(h('span', { text: f.rel }));
      if (f.edited) item.appendChild(h('span', { class: 'muted', text: '(edited)' }));
      item.appendChild(h('span', { class: 'a', text: '+' + f.added }));
      item.appendChild(h('span', { class: 'r', text: '\u2212' + f.removed }));
      item.addEventListener('click', function () { R.editing = false; R.cur = f.rel; renderReviewChrome(); loadReviewFile(f.rel); });
      strip.appendChild(item);
    });
  }
  function updateAcceptLabel() {
    var m = R.meta; if (!m) return;
    var files = m.files.filter(function (f) { return !f.source; });
    var n = files.filter(function (f) { return R.sel[f.rel] !== false; }).length;
    var all = n === files.length;
    $('acceptBtn').textContent = !m.partial_ok || all ? (files.length > 1 ? 'Accept all' : 'Accept') : 'Accept ' + n + ' of ' + files.length;
    $('acceptBtn').disabled = n === 0;
  }

  function loadReviewFile(rel) {
    if (!R.meta) return;
    return PDC.api('/api/review/file?id=' + R.meta.id + '&rel=' + encodeURIComponent(rel)).then(function (d) {
      if (d._status !== 200) { renderPanes(null); return; }
      R.data = d; renderPanes(d);
    });
  }

  function renderPanes(d) {
    var oldC = PDC.clear($('oldCode')), newC = PDC.clear($('newCode'));
    $('editBox').hidden = !R.editing; newC.hidden = R.editing;
    $('editBtn').textContent = R.editing ? '\u2714 Show diff' : '\u270E Edit updated';
    if (!d) { $('oldInfo').textContent = ''; $('newInfo').textContent = ''; return; }
    var parts = renderRows(d.rows);
    oldC.appendChild(parts[0]); newC.appendChild(parts[1]);
    $('oldInfo').textContent = '\u2212' + d.removed + ' lines';
    $('newInfo').textContent = '+' + d.added + ' lines';
    if (R.editing) $('editBox').value = d.after;
    oldC.scrollTop = newC.scrollTop = 0;
  }

  function syncScroll() {
    var a = $('oldCode'), b = $('newCode');
    a.addEventListener('scroll', function () { if (b.scrollTop !== a.scrollTop) b.scrollTop = a.scrollTop; });
    b.addEventListener('scroll', function () { if (a.scrollTop !== b.scrollTop) a.scrollTop = b.scrollTop; });
  }

  function toggleEdit() {
    if (!R.meta || !R.data) return;
    if (!R.editing) { R.editing = true; renderPanes(R.data); $('editBox').focus(); return; }
    var text = $('editBox').value;
    PDC.api('/api/review/rediff', { id: R.meta.id, rel: R.cur, text: text }).then(function (d) {
      R.editing = false;
      if (d._status === 200) {
        R.data = d;
        R.meta.files.forEach(function (f) { if (f.rel === R.cur) { f.added = d.added; f.removed = d.removed; f.edited = true; } });
        renderReviewChrome();
      } else PDC.toast(d.detail || 'could not update', true);
      renderPanes(R.data);
    });
  }

  function decide(action) {
    if (!R.meta) return;
    if (R.editing) { PDC.toast('Click "Show diff" to keep your edit first.', true); return; }
    var files = R.meta.partial_ok ? R.meta.files.filter(function (f) { return !f.source && R.sel[f.rel] !== false; }).map(function (f) { return f.rel; }) : null;
    PDC.api('/api/review/decide', { id: R.meta.id, action: action, files: files }).then(function (j) {
      if (j.ok) PDC.toast(action === 'apply' ? j.detail || 'Changes written.' : 'Rejected. No files were changed.');
      else PDC.toast(j.detail || 'failed', true);
      PDC.loadReview(); PDC.refreshState();
    });
  }

  // ---------------------------------------------------------------- plan -----
  var P = { text: '', editing: false };
  PDC.loadPlan = function () {
    return PDC.api('/api/plan').then(function (j) {
      P.info = j;
      if (!P.editing) renderPlan(j);
      var open = j.tasks ? j.tasks.filter(function (t) { return t.status === 'todo'; }).length : 0;
      var badge = $('planBadge');
      badge.hidden = !(j.tasks && j.tasks.length && open);
      badge.textContent = j.tasks ? j.tasks.filter(function (t) { return t.status === 'done'; }).length + '/' + j.tasks.length : '';
    });
  };
  function renderPlan(j) {
    var body = PDC.clear($('planBody'));
    $('planPath').textContent = j.exists ? j.path : '';
    var running = !!(PDC.S.status.job);
    $('planEditBtn').hidden = !j.exists; $('planEditBtn').disabled = running;
    var open = (j.tasks || []).filter(function (t) { return t.status === 'todo'; });
    $('planRunBtn').hidden = !(j.exists && open.length); $('planRunBtn').disabled = running;
    if (!j.exists) {
      body.appendChild(h('div', { class: 'placeholder', style: 'margin:auto' }, h('b', { text: 'No plan yet.' }), h('br'),
        'Use ', h('b', { text: 'Auto' }), ' to turn a goal into a task list, or a Task with planning on. The checklist and its progress show here.'));
      return;
    }
    var done = j.tasks.filter(function (t) { return t.status === 'done'; }).length;
    var pct = j.tasks.length ? Math.round(100 * done / j.tasks.length) : 0;
    body.appendChild(h('div', { class: 'muted small-text', text: done + ' of ' + j.tasks.length + ' tasks done' }));
    body.appendChild(h('div', { class: 'progress' }, h('div', { style: 'width:' + pct + '%' })));
    var firstTodo = true;
    var icons = { todo: '\u25CB', done: '\u2713', failed: '\u2717', skipped: '\u2013' };
    j.tasks.forEach(function (t) {
      var cur = t.status === 'todo' && firstTodo && running; if (t.status === 'todo') firstTodo = false;
      body.appendChild(h('div', { class: 'ptask ' + t.status + (cur ? ' current' : '') },
        h('span', { class: 'st', text: cur ? '\u25B6' : icons[t.status] || '\u25CB' }), h('span', { text: t.title })));
    });
  }
  function planEdit() {
    var j = P.info; if (!j || !j.exists) return;
    P.editing = !P.editing;
    $('planEdit').hidden = !P.editing; $('planBody').hidden = P.editing; $('planSaveBtn').hidden = !P.editing;
    $('planEditBtn').textContent = P.editing ? 'Cancel' : 'Edit';
    if (P.editing) $('planEdit').value = j.text; else renderPlan(j);
  }
  function planSave() {
    PDC.api('/api/plan', { text: $('planEdit').value }).then(function (j) {
      if (j.ok) { PDC.toast('Plan saved.'); P.editing = true; planEdit(); PDC.loadPlan(); }
      else PDC.toast(j.detail || 'could not save', true);
    });
  }
  function planRun() {
    var st = PDC.S.status, info = P.info || {};
    if (!info.exists) { PDC.toast('There is no plan to run yet.', true); return; }
    var a = (PDC.S.opts && PDC.S.opts.auto) || {};
    var goal = (st.last_auto && st.last_auto.text && info.path === st.plan_file) ? st.last_auto.text : '';
    PDC.startRun({ mode: 'auto', text: goal, plan_file: info.path, apply: true, verify: a.verify || '', repair: a.repair || '',
      max_tasks: a.max_tasks || '', max_minutes: a.max_minutes || '', keep_going: !!a.keep_going, auto_install: !!a.auto_install, diagnostics: !!a.diagnostics });
  }
  PDC.runPlan = function () { PDC.loadPlan().then(planRun); };

  // ---------------------------------------------------------------- files -----
  PDC.openFile = function (rel) {
    PDC.api('/api/file?path=' + encodeURIComponent(rel)).then(function (j) {
      PDC.showTab('file');
      var box = PDC.clear($('fileCode'));
      if (j._status !== 200) { $('filePath').textContent = j.detail || 'cannot open'; return; }
      $('filePath').textContent = rel + (j.truncated ? '  (first 600 KB)' : '') + '  \u00B7 ' + PDC.fmtInt(j.size) + ' bytes';
      if (j.binary) { box.appendChild(h('div', { class: 'tip', text: 'Binary file.' })); return; }
      var frag = document.createDocumentFragment();
      j.text.split('\n').forEach(function (t, i) { frag.append(line('equal', i + 1, t)); });
      box.appendChild(frag);
      $('fileView').scrollTop = 0;
      PDC.S.openFile = rel; renderTree();
    });
  };

  var T = { files: [], sig: '', collapsed: {}, filter: '', changed: {} };
  PDC.loadFiles = function (sig) {
    if (sig === T.sig) return;
    PDC.api('/api/files').then(function (j) { T.files = j.files || []; T.sig = j.sig; renderTree(); });
  };
  PDC.markChanged = function (list) { T.changed = {}; list.forEach(function (f) { T.changed[f] = true; }); renderTree(); };
  function renderTree() {
    var tree = PDC.clear($('tree'));
    var q = T.filter.trim().toLowerCase();
    var files = q ? T.files.filter(function (f) { return f.toLowerCase().indexOf(q) !== -1; }) : T.files;
    $('fileCount').textContent = T.files.length ? (q ? files.length + ' of ' : '') + T.files.length + ' file' + (T.files.length === 1 ? '' : 's') : 'no index yet';
    var frag = document.createDocumentFragment(), seen = {};
    var limit = 3000, shown = 0;
    for (var i = 0; i < files.length && shown < limit; i++) {
      var parts = files[i].split('/'), hide = false;
      if (!q) {
        for (var d = 1; d < parts.length; d++) {
          var dir = parts.slice(0, d).join('/');
          if (T.collapsed[dir]) { hide = true; }
          if (!seen[dir]) {
            seen[dir] = true;
            if (!isHidden(parts, d)) {
              frag.append(dirRow(dir, parts[d - 1], d - 1)); shown++;
            }
          }
        }
        if (hide || isHidden(parts, parts.length)) continue;
      }
      var rel = files[i];
      var row = h('div', { class: 'it' + (rel === PDC.S.openFile ? ' sel' : '') + (T.changed[rel] ? ' chg' : ''), title: rel,
        style: q ? '' : 'padding-left:' + (8 + (parts.length - 1) * 12) + 'px', text: q ? rel : parts[parts.length - 1],
        onclick: function () { PDC.openFile(this.title); } });
      frag.append(row); shown++;
    }
    if (shown >= limit) frag.append(h('div', { class: 'tip', text: 'Showing the first ' + limit + ' entries; use the filter.' }));
    tree.appendChild(frag);
  }
  function isHidden(parts, depth) {
    for (var d = 1; d < depth; d++) if (T.collapsed[parts.slice(0, d).join('/')]) return true;
    return false;
  }
  function dirRow(dir, name, depth) {
    return h('div', { class: 'it dir', style: 'padding-left:' + (8 + depth * 12) + 'px', title: dir,
      text: (T.collapsed[dir] ? '\u25B8 ' : '\u25BE ') + name + '/',
      onclick: function () { T.collapsed[dir] = !T.collapsed[dir]; renderTree(); } });
  }

  // ---------------------------------------------------------- folder browser -----
  PDC.browse = function (opts) {
    var st = { path: opts.start || '', hidden: false, picked: '' };
    var pathIn = h('input', { type: 'text', class: 'grow mono', spellcheck: false });
    var list = h('div', { class: 'fb-list' });
    var info = h('div', { class: 'muted small-text', style: 'margin-top:6px' });
    var m;
    function go(path) {
      PDC.api('/api/fs?path=' + encodeURIComponent(path || '') + '&files=' + (opts.files ? 1 : 0) + '&hidden=' + (st.hidden ? 1 : 0)).then(function (j) {
        st.path = j.path; pathIn.value = j.path; st.picked = '';
        PDC.clear(list);
        if (j.parent) list.appendChild(item('\u21B0', '..', function () { go(j.parent); }));
        j.entries.forEach(function (e) {
          var full = j.path.replace(/[\\/]$/, '') + (j.path.indexOf('\\') >= 0 && j.path.indexOf('/') < 0 ? '\\' : '/') + e.name;
          if (e.dir) list.appendChild(item('\uD83D\uDCC1', e.name, function () { go(full); }, e.project ? 'project' : ''));
          else list.appendChild(item('\uD83D\uDCC4', e.name, function (el) { pick(el, full); }, ''));
        });
        info.textContent = j.error || (j.truncated ? 'List truncated.' : (j.entries.length + ' entries'));
      });
    }
    function pick(el, full) {
      Array.prototype.forEach.call(list.children, function (c) { c.classList.remove('sel'); });
      el.classList.add('sel'); st.picked = full;
      if (opts.files) { opts.onPick(full); m.close(); }
    }
    function item(icon, name, fn, tag) {
      var el = h('div', { class: 'fb-item' }, h('span', { class: 'ico', text: icon }), h('span', { text: name }), tag ? h('span', { class: 'proj', text: tag }) : null);
      el.addEventListener('click', function () { fn(el); });
      return el;
    }
    pathIn.addEventListener('keydown', function (e) { if (e.key === 'Enter') go(pathIn.value); });
    var body = h('div', {}, h('div', { class: 'row', style: 'margin-bottom:8px' }, pathIn,
      h('button', { text: 'Go', onclick: function () { go(pathIn.value); } }),
      h('label', { class: 'chk' }, h('input', { type: 'checkbox', onchange: function (e) { st.hidden = e.target.checked; go(st.path); } }), 'hidden')), list, info);
    var buttons = [{ label: 'Cancel' }];
    if (!opts.files) buttons.push({ label: opts.pickLabel || 'Use this folder', primary: true, onclick: function () { opts.onPick(st.path); } });
    m = PDC.modal(opts.title || 'Choose', body, buttons);
    go(st.path);
  };

  // ------------------------------------------------------------------ wire -----
  document.addEventListener('DOMContentLoaded', function () {
    Array.prototype.forEach.call($('codeTabs').children, function (b) { b.addEventListener('click', function () { PDC.showTab(b.dataset.tab); }); });
    syncScroll();
    $('editBtn').addEventListener('click', toggleEdit);
    $('acceptBtn').addEventListener('click', function () { decide('apply'); });
    $('rejectBtn').addEventListener('click', function () { decide('reject'); });
    $('planEditBtn').addEventListener('click', planEdit);
    $('planSaveBtn').addEventListener('click', planSave);
    $('planRunBtn').addEventListener('click', planRun);
    $('fileFilter').addEventListener('input', function () { T.filter = this.value; renderTree(); });
    $('hideFiles').addEventListener('click', function () { $('app').classList.add('no-files'); PDC.S.filesHidden = true; });
    PDC.showTab('changes', true);
    PDC.renderTree = renderTree;
  });
})();
