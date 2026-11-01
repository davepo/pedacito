// Settings: edit profiles and every setting in the user or project config
// file. The server patches the TOML text in place (comments survive), checks
// it with the loader's own rules, and keeps the old file as <name>.bak.
(function () {
  'use strict';
  var PDC = window.PDC, h = PDC.h;

  var COMMON = ['base_url', 'api_key', 'model', 'tier', 'vram_gb', 'json_mode', 'tool_mode', 'timeout',
    'context_tokens', 'allow_write', 'allow_shell', 'allow_browser', 'allow_screenshots', 'semantic_search', 'review'];
  var X = null;      // {data, which, tab, profile, target, pending, modal, body}

  PDC.openSettings = function () {
    X = { data: null, which: 'user', tab: 'profiles', profile: '', target: '', pending: {}, models: [], search: '' };
    X.modal = PDC.modal('Settings', h('div', { class: 'tip', text: 'Loading\u2026' }), [{ label: 'Close' }]);
    X.body = X.modal.body;
    PDC.api('/api/settings').then(function (d) { X.data = d; pickDefaults(); render(); });
  };

  function file() { return X.data[X.which]; }
  function schemaItem(name) { return X.data.schema.filter(function (s) { return s.name === name; })[0]; }
  function pickDefaults() {
    var f = file(), names = Object.keys(f.profiles || {});
    if (!X.profile || names.indexOf(X.profile) < 0) X.profile = names.indexOf(X.data.active_profile) >= 0 ? X.data.active_profile : (names[0] || '');
    if (!X.target) X.target = '';
  }

  function save(payload, url, okMsg) {
    return PDC.api(url, payload).then(function (j) {
      if (!j.ok) { PDC.toast(j.detail || 'could not save', true); return j; }
      X.data = j; X.pending = {}; pickDefaults(); render();
      PDC.toast(okMsg || 'Saved.' + (j.reload_pending ? ' Applies after the running job.' : ''));
      PDC.refreshState();
      return j;
    });
  }

  // ----- typed controls ----------------------------------------------------------
  // `value === undefined` means "not set here" (inherit). onchange(newValue|null)
  function control(item, value, onchange, placeholder) {
    var t = item.type, el;
    if (t === 'bool' || item.choices) {
      var opts = t === 'bool' ? [['', '(inherit)'], ['true', 'on'], ['false', 'off']]
        : [['__inherit__', '(inherit)']].concat(item.choices.map(function (c) { return [c, c === '' ? '(none)' : c]; }));
      el = h('select', {}, opts.map(function (o) { return h('option', { value: o[0], text: o[1] }); }));
      el.value = value === undefined ? (t === 'bool' ? '' : '__inherit__') : (t === 'bool' ? String(value) : value);
      el.addEventListener('change', function () {
        var v = el.value;
        if (v === '' && t === 'bool' || v === '__inherit__') onchange(null);
        else onchange(t === 'bool' ? v === 'true' : v);
      });
      return el;
    }
    if (t === 'table') return h('span', { class: 'muted small-text', text: 'edit in the Raw file tab' });
    var shown = value === undefined ? '' : (t === 'list' ? (value || []).join(', ') : String(value));
    el = h('input', { type: item.secret ? 'password' : (t === 'int' || t === 'float' ? 'number' : 'text'), value: shown,
      placeholder: placeholder || '', step: t === 'float' ? 'any' : null, autocomplete: 'off', spellcheck: false });
    el.addEventListener('input', function () {
      var raw = el.value;
      if (raw === '' && !item.secret) { onchange(null); return; }
      if (raw === '') { onchange(undefined); return; }
      if (t === 'int') onchange(parseInt(raw, 10));
      else if (t === 'float') onchange(parseFloat(raw));
      else if (t === 'list') onchange(raw.split(',').map(function (s) { return s.trim(); }).filter(Boolean));
      else onchange(raw);
    });
    return el;
  }
  function placeholderFor(item) {
    if (item.secret) {
      var body = (file().profiles || {})[X.profile] || {};
      return (body._secret_set || []).indexOf(item.name) >= 0 ? '\u2022\u2022\u2022\u2022 set \u2014 leave blank to keep' : '';
    }
    // what applies when this layer says nothing: the file's global value,
    // else the open project's resolved value, else the built-in default
    var g = (file().settings || {})[item.name], r = X.data.resolved[item.name], v, tag = '';
    if (g !== undefined) { v = g; tag = '  (global)'; }
    else if (r && r.source !== 'default') { v = r.value; tag = '  (' + r.source.replace(/\s*\(.*$/, '') + ')'; }
    else { v = item.default; tag = '  (default)'; }
    if (v === '' || v == null) return 'default';
    return (Array.isArray(v) ? v.join(', ') : String(v)) + tag;
  }

  // ----- render ---------------------------------------------------------------------
  function render() {
    var d = X.data, f = file();
    var body = PDC.clear(X.body);
    var which = h('div', { class: 'row', style: 'margin-bottom:10px' },
      h('b', { text: 'Editing:' }),
      radio('user', 'User config'), radio('project', 'This project (.pedacito.toml)', !d.project_open),
      h('span', { class: 'mono muted small-text grow', text: f.available ? f.path : '' }));
    body.appendChild(which);
    if (!f.available) { body.appendChild(h('div', { class: 'note-box', text: f.error })); return; }
    if (X.which === 'user' && !f.exists) { body.appendChild(wizard()); return; }
    if (X.which === 'project' && !f.exists) {
      body.appendChild(h('div', { class: 'note-box', text: 'This project has no config file yet. Saving a setting creates .pedacito.toml in the project root (it layers on top of your user config).' }));
    }
    if (f.error) body.appendChild(h('div', { class: 'note-box err', text: f.error }));
    if (X.which === 'project' && f.exists && f.trusted === false) {
      body.appendChild(h('div', { class: 'note-box' },
        'This project config is not trusted yet, so its execution settings (' + f.exec_fields.join(', ') + ') are ignored until you approve it. ',
        h('button', { class: 'small', text: 'Review & trust\u2026', onclick: trust })));
    }
    var tabs = h('div', { class: 'stabs' });
    [['profiles', 'Profiles'], ['all', 'All settings'], ['raw', 'Raw file']].forEach(function (t) {
      tabs.appendChild(h('button', { class: X.tab === t[0] ? 'on' : '', text: t[1], onclick: function () { X.tab = t[0]; X.pending = {}; render(); } }));
    });
    body.appendChild(tabs);
    if (X.tab === 'profiles') body.appendChild(profilesTab());
    else if (X.tab === 'all') body.appendChild(allTab());
    else body.appendChild(rawTab());
  }
  function radio(which, label, disabled) {
    return h('label', { class: 'chk' }, h('input', { type: 'radio', name: 'which', checked: X.which === which, disabled: !!disabled,
      onchange: function () { X.which = which; X.pending = {}; X.profile = ''; X.target = ''; pickDefaults(); render(); } }), label);
  }
  function trust() {
    var f = file();
    PDC.confirm('Trust ' + f.path + '?\n\nIt will be allowed to set: ' + f.exec_fields.join(', ') + '. These can run commands, so only trust a file you have read. Trust lapses if the file changes.', 'Trust it').then(function (ok) {
      if (ok) save({ file: 'project' }, '/api/settings/trust', 'Trusted.');
    });
  }

  // ----- first-run wizard ------------------------------------------------------------
  function wizard() {
    var url = h('input', { type: 'text', value: 'http://localhost:1234/v1', class: 'grow' });
    var model = h('input', { type: 'text', value: '', class: 'grow', list: 'modelOptions', placeholder: 'model id as the server reports it' });
    var vram = h('input', { type: 'number', value: '', placeholder: 'GB of GPU VRAM (blank for hosted APIs)' });
    var list = h('datalist', { id: 'modelOptions' });
    var found = h('div', { class: 'row', style: 'margin-top:6px' });
    var status = h('div', { class: 'muted small-text', style: 'margin-top:6px' });
    function probe() {
      status.textContent = 'Testing\u2026';
      PDC.api('/api/settings/models', { base_url: url.value, api_key: '' }).then(function (j) {
        PDC.clear(list);
        if (!j.ok) { status.textContent = '\u2717 ' + j.error; return; }
        j.models.forEach(function (m) { list.appendChild(h('option', { value: m })); });
        if (j.models.length && !model.value) model.value = j.models[0];
        status.textContent = '\u2713 reachable \u2014 ' + j.models.length + ' model(s): ' + j.models.slice(0, 6).join(', ');
      });
    }
    function discover() {
      PDC.clear(found); found.appendChild(h('span', { class: 'muted small-text', text: 'Looking for LM Studio and Ollama\u2026' }));
      PDC.api('/api/settings/discover', {}).then(function (j) {
        PDC.clear(found);
        if (!j.servers.length) { found.appendChild(h('span', { class: 'muted small-text', text: 'No local server found; enter its URL.' })); return; }
        j.servers.forEach(function (s) {
          found.appendChild(h('button', { class: 'small', text: s.label + ' (' + s.models.length + ')', onclick: function () {
            url.value = s.base_url; PDC.clear(list); s.models.forEach(function (m) { list.appendChild(h('option', { value: m })); });
            if (s.models.length) model.value = s.models[0]; status.textContent = '';
          } }));
        });
      });
    }
    var box = h('div', {}, h('div', { class: 'note-box', text: 'No config file yet \u2014 pedacito is running on built-in defaults. Create one: it only needs the server URL and model; everything else has a working default (and can be tuned in the other tabs afterwards).' }),
      h('div', { class: 'form' },
        h('label', { text: 'Server URL' }), h('div', { class: 'row' }, url, h('button', { text: 'Test', onclick: probe }), h('button', { text: 'Find servers', onclick: discover })),
        h('label', { text: '' }), found,
        h('label', { text: 'Model' }), h('div', { class: 'row' }, model, list),
        h('label', { text: 'vram_gb' }), vram),
      status,
      h('div', { class: 'row', style: 'margin-top:12px' }, h('button', { class: 'primary', text: 'Create config', onclick: function () {
        save({ base_url: url.value, model: model.value, vram_gb: vram.value }, '/api/settings/setup', 'Config created.');
      } })));
    return box;
  }

  // ----- profiles ----------------------------------------------------------------------
  function profilesTab() {
    var f = file(), names = Object.keys(f.profiles || {}).sort();
    var defName = (f.settings || {}).default_profile || '';
    var list = h('div', { class: 'plist' });
    names.forEach(function (n) {
      list.appendChild(h('div', { class: 'pitem' + (n === X.profile ? ' on' : ''), onclick: function () { X.profile = n; X.pending = {}; render(); } },
        h('span', { class: 'grow', text: n }),
        n === X.data.active_profile ? h('span', { class: 'tag act', text: 'active' }) : null,
        n === defName ? h('span', { class: 'tag', text: 'default' }) : null));
    });
    if (!names.length) list.appendChild(h('div', { class: 'tip', text: 'No profiles in this file.' }));
    var side = h('div', {}, list, h('div', { class: 'row', style: 'margin-top:8px' },
      h('button', { class: 'small', text: '+ New', onclick: newProfile }),
      h('button', { class: 'small', text: 'Duplicate', disabled: !X.profile, onclick: dupProfile }),
      h('button', { class: 'small danger', text: 'Delete', disabled: !X.profile, onclick: delProfile })));
    return h('div', { class: 'set-layout' }, side, X.profile ? profileForm(f.profiles[X.profile] || {}) : h('div', { class: 'tip', text: 'Create a profile to point pedacito at a model.' }));
  }
  function askName(title, initial, then) {
    var inp = h('input', { type: 'text', value: initial || '', class: 'grow', placeholder: 'e.g. qwen-27b' });
    PDC.modal(title, h('div', { class: 'row' }, inp), [{ label: 'Cancel' }, { label: 'OK', primary: true, onclick: function () { then(inp.value.trim()); } }], { narrow: true });
    setTimeout(function () { inp.focus(); }, 30);
  }
  function newProfile() {
    askName('New profile name', '', function (name) {
      if (!name) return;
      save({ file: X.which, action: 'create', name: name, values: { base_url: 'http://localhost:1234/v1', model: 'model-id' } }, '/api/settings/profile', 'Profile created \u2014 fill in its model.').then(function (j) { if (j.ok) { X.profile = name; render(); } });
    });
  }
  function dupProfile() {
    askName('Duplicate "' + X.profile + '" as', X.profile + '-copy', function (name) {
      if (!name) return;
      save({ file: X.which, action: 'duplicate', name: name, source: X.profile }, '/api/settings/profile', 'Profile duplicated.').then(function (j) { if (j.ok) { X.profile = name; render(); } });
    });
  }
  function delProfile() {
    PDC.confirm('Delete profile "' + X.profile + '" from this file? (A .bak copy of the file is kept.)', 'Delete', true).then(function (ok) {
      if (ok) save({ file: X.which, action: 'delete', name: X.profile }, '/api/settings/profile', 'Profile deleted.').then(function () { X.profile = ''; pickDefaults(); render(); });
    });
  }

  function pendingFor(key) { return X.pending[key]; }
  function setPending(key, v, row) {
    if (v === undefined) delete X.pending[key]; else X.pending[key] = v;
    if (row) row.classList.toggle('mod', key in X.pending);
    var b = document.getElementById('savePending'); if (b) { var n = Object.keys(X.pending).length; b.disabled = !n; b.textContent = n ? 'Save ' + n + ' change' + (n === 1 ? '' : 's') : 'No changes'; }
  }
  function commit(profile) {
    var changes = {}, removes = [];
    Object.keys(X.pending).forEach(function (k) { if (X.pending[k] === null) removes.push(k); else changes[k] = X.pending[k]; });
    return save({ file: X.which, profile: profile || null, changes: changes, removes: removes }, '/api/settings/set');
  }

  function profileForm(body) {
    var form = h('div', { class: 'form' });
    var name = X.profile;
    var apiKeyEl;
    COMMON.forEach(function (key) {
      var item = schemaItem(key); if (!item) return;
      var cur = key === 'api_key' ? undefined : body[key];
      var el = control(item, cur, function (v) { setPending(key, v); }, placeholderFor(item));
      if (key === 'api_key') apiKeyEl = el;
      if (key === 'model') el.setAttribute('list', 'modelOptions');
      form.appendChild(h('label', { text: key, title: item.help }));
      form.appendChild(key === 'model' ? h('div', { class: 'row' }, h('div', { class: 'grow' }, el), h('datalist', { id: 'modelOptions' }, X.models.map(function (m) { return h('option', { value: m }); }))) : el);
      if (item.help) form.appendChild(h('div', { class: 'help', text: item.help.length > 170 ? item.help.slice(0, 167) + '\u2026' : item.help }));
    });
    var result = h('div', { class: 'muted small-text', style: 'margin-top:8px' });
    var extras = Object.keys(body).filter(function (k) { return COMMON.indexOf(k) < 0 && k.charAt(0) !== '_'; });
    var adv = h('div', { style: 'margin-top:14px' });
    if (extras.length) {
      adv.appendChild(h('div', { class: 'muted small-text', text: 'Other overrides set in this profile:' }));
      extras.forEach(function (k) {
        var v = body[k];
        adv.appendChild(h('div', { class: 'row small-text' }, h('code', { text: k }), h('span', { class: 'muted grow mono', text: typeof v === 'object' ? JSON.stringify(v) : String(v) }),
          h('button', { class: 'ghost small', text: 'remove', onclick: function () { X.pending = {}; X.pending[k] = null; commit(name); } })));
      });
    }
    var mkDefault = h('button', { text: 'Make default', onclick: function () {
      save({ file: X.which, profile: null, changes: { default_profile: name } }, '/api/settings/set', '"' + name + '" is now the default profile.'); } });
    return h('div', {}, form, adv, result,
      h('div', { class: 'row', style: 'margin-top:12px' },
        h('button', { id: 'savePending', class: 'primary', text: 'No changes', disabled: true, onclick: function () { commit(name); } }),
        h('button', { text: 'Test connection', onclick: function () {
          var url = X.pending.base_url !== undefined && X.pending.base_url !== null ? X.pending.base_url : (body.base_url || (X.data.resolved.base_url || {}).value || '');
          var key = X.pending.api_key || '';
          result.textContent = 'Testing ' + url + '\u2026';
          PDC.api('/api/settings/models', { base_url: url, api_key: key, file: X.which, profile: name }).then(function (j) {
            if (!j.ok) { result.textContent = '\u2717 ' + j.error; return; }
            X.models = j.models;
            var dl = document.getElementById('modelOptions'); if (dl) { PDC.clear(dl); j.models.forEach(function (m) { dl.appendChild(h('option', { value: m })); }); }
            var want = X.pending.model || body.model;
            result.textContent = '\u2713 reachable \u2014 ' + j.models.length + ' model(s) (offered in the model box): ' + j.models.slice(0, 8).join(', ') +
              (want && j.models.indexOf(want) < 0 ? '\n\u26A0 "' + want + '" is not in that list.' : '');
            result.style.whiteSpace = 'pre-wrap';
          });
        } }),
        mkDefault));
  }

  // ----- all settings ----------------------------------------------------------------
  function allTab() {
    var f = file();
    var targets = [['', 'Global (every profile)']].concat(Object.keys(f.profiles || {}).sort().map(function (n) { return [n, 'Profile: ' + n]; }));
    var targetSel = h('select', {}, targets.map(function (t) { return h('option', { value: t[0], text: t[1] }); }));
    targetSel.value = X.target;
    targetSel.addEventListener('change', function () { X.target = targetSel.value; X.pending = {}; render(); });
    var search = h('input', { type: 'text', placeholder: 'search settings\u2026', value: X.search, class: 'grow' });
    var holder = h('div');
    var saveBtn = h('button', { id: 'savePending', class: 'primary', text: 'No changes', disabled: true, onclick: function () { commit(X.target || null); } });
    var cur = X.target ? (f.profiles[X.target] || {}) : (f.settings || {});
    function fill() {
      PDC.clear(holder);
      var q = search.value.trim().toLowerCase(), groups = {}, order = [];
      X.data.schema.forEach(function (it) {
        if (q && (it.name + ' ' + it.help + ' ' + it.group).toLowerCase().indexOf(q) < 0) return;
        if (!groups[it.group]) { groups[it.group] = []; order.push(it.group); }
        groups[it.group].push(it);
      });
      order.forEach(function (g) {
        var det = h('details', { class: 'setgrp', open: !!q || order.length === 1 || g === 'Server connection' }, h('summary', { text: g + ' (' + groups[g].length + ')' }));
        groups[g].forEach(function (it) {
          var set = cur[it.name];
          var res = X.data.resolved[it.name] || {};
          var src = set !== undefined ? 'set here' : ((res.source || 'default').replace(/\s*\(.*$/, '') || 'default');
          var row = h('div', { class: 'set-row' });
          row.appendChild(h('span', { class: 'name', text: it.name, title: it.help }));
          var shown = X.pending[it.name] !== undefined ? (X.pending[it.name] === null ? undefined : X.pending[it.name]) : set;
          row.appendChild(control(it, shown, function (v) { setPending(it.name, v, row); }, placeholderFor(it)));
          row.appendChild(h('span', { class: 'src', text: src }));
          if (it.help) row.appendChild(h('div', { class: 'set-help', text: it.help.length > 260 ? it.help.slice(0, 257) + '\u2026' : it.help }));
          det.appendChild(row);
        });
        holder.appendChild(det);
      });
      if (!order.length) holder.appendChild(h('div', { class: 'tip', text: 'No setting matches.' }));
    }
    search.addEventListener('input', function () { X.search = search.value; fill(); });
    fill();
    return h('div', {}, h('div', { class: 'row', style: 'margin-bottom:8px' }, h('span', { class: 'muted', text: 'Apply to:' }), targetSel, search, saveBtn),
      h('div', { class: 'muted small-text', style: 'margin-bottom:6px', text: 'A blank box means "not set here" \u2014 the value comes from the layer below it (the badge on the right says which). Clearing a box removes the key from the file.' }), holder);
  }

  // ----- raw ---------------------------------------------------------------------------
  function rawTab() {
    var f = file();
    var ta = h('textarea', { id: 'rawText', spellcheck: false, value: f.text || '' });
    var msg = h('div', { class: 'muted small-text', style: 'margin-top:6px', text: 'Saved edits are checked with the same rules the loader uses; the previous file is kept as ' + (f.path || '') + '.bak.' });
    return h('div', {}, ta, msg, h('div', { class: 'row', style: 'margin-top:8px' },
      h('button', { class: 'primary', text: 'Validate & save', onclick: function () { save({ file: X.which, text: ta.value }, '/api/settings/raw', 'Config file saved.'); } }),
      h('button', { text: 'Revert', onclick: function () { ta.value = f.text || ''; } })));
  }
})();
