// Settings: edit profiles and every setting in the user or project config
// file. The server patches the TOML text in place (comments survive), checks
// it with the loader's own rules, and keeps the old file as <name>.bak.
(function () {
  'use strict';
  var PDC = window.PDC, h = PDC.h;

  var COMMON = ['base_url', 'api_key', 'model', 'tier', 'vram_gb', 'json_mode', 'timeout',
    'allow_write', 'allow_shell', 'allow_browser', 'allow_screenshots', 'semantic_search', 'review',
    'scout_profile', 'enforce_json_schema'];
  // Profile overrides that the editor surfaces up front, in their own boxed
  // group: how the model thinks, plans, and lands edits (see modelcard.choose
  // \u2014 these are exactly the keys a calibration writes into a profile).
  var REASONING = ['disable_thinking', 'thinking_suffix', 'step_reasoning_first',
    'incremental_edit_loop', 'edit_plan_brief', 'dynamic_tool_pruning',
    'prompt_pack', 'answer_mode', 'edit_style', 'tool_mode', 'best_of',
    'fuzzy_match', 'temperature', 'temperature_answer',
    'max_output_tokens', 'context_tokens'];
  var X = null;      // {data, which, tab, profile, target, pending, modal, body}
  // Model discovery lives outside X: its state survives switching tabs, and a
  // scan is read-only, so re-rendering must not throw the results away.
  var SCAN = { busy: false, rows: [], err: '', custom: '', vram: '',
               pick: {}, name: {}, def: '', importing: false };
  // Same for the Auto-Tune wizard: the answers and the preview survive
  // switching to another settings tab and back.
  var TUNE = { hardware: 'hosted', model: 'medium', codebase: 'medium',
               priority: 'balanced', result: null, err: '', busy: false };
  // And for the hosted-provider dialog: the catalog starts as the built-in
  // fallback (always renders instantly) and is replaced once
  // GET /api/providers/presets answers. HC holds the dialog's own state, so
  // a half-finished connect/import survives re-renders and reopen.
  var CLOUD = { presets: (PDC.FALLBACK_PROVIDER_CATALOG || []).slice(),
                fetched: false };
  var HC = { preset: '', base: '', key: '', env: '', busy: false,
             probed: false, verified: false, err: '', models: [],
             pick: {}, name: {}, def: '', search: '', importing: false,
             modal: null, body: null };

  PDC.openSettings = function (opts) {
    opts = opts || {};
    X = { data: null, which: opts.which || 'user',
          tab: opts.tab || 'discovery', profile: opts.profile || '',
          target: opts.target || '', pending: {}, models: [],
          search: opts.search || '', focusSearch: !!opts.search };
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
  // The "(inherit)" option only exists when the caller passes an
  // `inheritLabel` — that is, while editing a PROFILE override. Universal
  // settings are the top-level user config (nothing above them to inherit
  // from), so they render concrete values only: booleans as on/off, enums as
  // their real choices. Clearing a universal key still removes it, restoring
  // the built-in default.
  function control(item, value, onchange, placeholder, inheritLabel) {
    var t = item.type, el;
    if (item.name === 'scout_profile') {
      // Scout/architect routing: pick the fast profile that answers gather's
      // lookup steps. The server-side schema is cached across projects, so
      // the profile choices are read here from the settings payload instead.
      var names = [];
      ['user', 'project'].forEach(function (k) {
        var pf = (X.data && X.data[k] && X.data[k].profiles) || {};
        Object.keys(pf).forEach(function (n) { if (names.indexOf(n) < 0) names.push(n); });
      });
      if (value && names.indexOf(value) < 0) names.push(value);
      var sopts = [['', 'None (use active profile)']]
        .concat(names.map(function (n) { return [n, n]; }));
      if (inheritLabel) sopts = [['__inherit__', inheritLabel]].concat(sopts);
      el = h('select', {}, sopts.map(function (o) { return h('option', { value: o[0], text: o[1] }); }));
      el.value = (value === undefined || value === null)
        ? (inheritLabel ? '__inherit__' : '') : String(value);
      el.addEventListener('change', function () {
        onchange(el.value === '__inherit__' ? null : el.value);
      });
      return el;
    }
    if (t === 'bool' || item.choices) {
      var opts = t === 'bool' ? [['true', 'on'], ['false', 'off']]
        : item.choices.map(function (c) { return [c, c === '' ? '(none)' : c]; });
      if (inheritLabel) opts = [['__inherit__', inheritLabel]].concat(opts);
      el = h('select', {}, opts.map(function (o) { return h('option', { value: o[0], text: o[1] }); }));
      var unset = value === undefined || value === null;
      // without an inherit option, a genuinely unset optional (fewshot_steps,
      // edit_style, …) shows no preselection until the user picks one
      el.value = unset ? (inheritLabel ? '__inherit__' : '\u0000')
                       : (t === 'bool' ? String(value) : value);
      el.addEventListener('change', function () {
        var v = el.value;
        if (v === '__inherit__') onchange(null);
        else if (v !== '\u0000') onchange(t === 'bool' ? v === 'true' : v);
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
  // The concrete value that applies when the file's own layer says nothing:
  // the value written in the file, else the effective resolved one (always a
  // concrete value — the payload falls back to the Config() defaults), else
  // the schema default. Universal settings render this, never "(inherit)".
  function universalValue(item) {
    var g = (file().settings || {})[item.name];
    if (g !== undefined) return g;
    var r = X.data.resolved[item.name];
    if (r && r.value !== undefined && r.value !== null) return r.value;
    return item.default === null ? undefined : item.default;
  }
  // What the profile picker shows for the inherit option, so the user can
  // see which universal value the profile takes when left unset.
  function inheritLabel(item) {
    var v = universalValue(item);
    if (item.type === 'bool') v = v ? 'on' : 'off';
    else if (v === undefined || v === null) v = '';
    var s = Array.isArray(v) ? v.join(', ') : String(v);
    return '(inherit \u2014 universal: ' + (s === '' ? 'unset' : s) + ')';
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
    [['discovery', 'Model discovery'], ['profiles', 'Profiles'], ['all', 'All settings'],
     ['autotune', 'Auto-Tune'], ['raw', 'Raw file / reset']].forEach(function (t) {
      tabs.appendChild(h('button', { class: X.tab === t[0] ? 'on' : '', text: t[1], onclick: function () { X.tab = t[0]; X.pending = {}; render(); } }));
    });
    body.appendChild(tabs);
    if (X.tab === 'discovery') body.appendChild(discoveryTab());
    else if (X.tab === 'profiles') body.appendChild(profilesTab());
    else if (X.tab === 'all') body.appendChild(allTab());
    else if (X.tab === 'autotune') body.appendChild(autotuneTab());
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

  // ----- model discovery, selective import, calibration -------------------------------
  // Canonical URL spelling, mirroring `discovery.normalize_url` on the server: one
  // local server is often written several ways (localhost vs 127.0.0.1, trailing
  // /v1 or slash), and each spelling would key its own checkbox and group header.
  function normUrl(u) {
    u = String(u || '').trim().replace(/\/+$/, '');
    u = u.replace(/^([a-zA-Z][a-zA-Z0-9+.\-]*:\/\/)localhost(?=[:/]|$)/i, '$1127.0.0.1');
    var auth = u.match(/^([a-zA-Z][a-zA-Z0-9+.\-]*:\/\/[^/]*)([\s\S]*)$/);
    if (auth) u = auth[1].toLowerCase() + auth[2];   // scheme+host are case-insensitive; the path is not
    var sfx = u.match(/(\/api\/v1|\/v1)$/);
    if (sfx) u = u.slice(0, -sfx[0].length).replace(/\/+$/, '');
    return u;
  }
  // Mirrors `settings_api._sanitize_profile_name`: version dots (and : @ space)
  // become _, names lower-case, separator runs collapse and trim, a -gguf tail
  // drops, max 32 chars. A '/' is left in place on purpose: that is a name a
  // human typed, and the import check should refuse it rather than guess.
  function sanitizeName(s) {
    s = String(s || '').trim().toLowerCase().replace(/[^a-z0-9_\-\/]+/g, '_');
    s = s.replace(/([-_]){2,}/g, '$1');
    s = s.replace(/^[-_]+|[-_]+$/g, '');
    while (s.length > 4 && /[-_]gguf$/.test(s)) s = s.replace(/[-_]gguf$/, '').replace(/^[-_]+|[-_]+$/g, '');
    if (s.length > 32) s = s.slice(0, 32).replace(/[-_]+$/, '');
    return s;
  }
  // ----- hosted provider dialog: connect & verify, then model-named import ----
  // Pedacito stays local-first: a hosted key is only trusted after it has been
  // used from this machine. Step A POSTs /api/providers/probe (one live GET to
  // the provider's own /models endpoint); step B ticks the models that came
  // back and imports each as a profile NAMED AFTER THE MODEL (`deepseek_chat`,
  // `gemini_2_5_flash`) — never after the vendor. A failed key still offers
  // the curated preset list, so the dialog works while the key gets fixed.
  function hcPreset() {
    var ps = CLOUD.presets || [];
    for (var i = 0; i < ps.length; i++) if (ps[i].id === HC.preset) return ps[i];
    return null;                        // 'custom' — any OpenAI-compatible URL
  }
  function hcApplyPreset() {           // a provider switch starts a fresh connect
    var p = hcPreset();
    HC.err = ''; HC.probed = false; HC.verified = false; HC.models = [];
    HC.pick = {}; HC.name = {}; HC.def = ''; HC.search = '';
    HC.base = p ? p.base_url : '';     // editable: regional endpoints and proxies
    HC.env = p ? p.api_key_env : '';
  }
  function hcSeed() {
    if (HC.preset) return;             // the session keeps its last provider
    var p = (CLOUD.presets || [])[0];
    HC.preset = p ? p.id : 'custom';
    hcApplyPreset();
  }
  function loadPresets() {
    if (CLOUD.fetched) return;
    CLOUD.fetched = true;                     // one attempt per session; the
    PDC.api('/api/providers/presets').then(function (j) {   // fallback is live
      // Accept every sane shape: a bare array, {presets: [...]}, or the
      // {providers: [...]} spelling. No `ok` gate: a missing flag must not
      // throw away a perfectly good catalog. An empty answer keeps the fallback.
      var list = Array.isArray(j) ? j : ((j && (j.presets || j.providers)) || []);
      if (!list.length) return;
      CLOUD.presets = list;
    }, function () {});                       // unreachable route: fallback stands
  }
  function hcVisible() {              // rows matching the step-B filter box
    var flt = String(HC.search || '').trim().toLowerCase();
    return HC.models.filter(function (m) {
      return !flt || String(m.id).toLowerCase().indexOf(flt) >= 0;
    });
  }
  function hcPicked() {
    return HC.models.filter(function (m) { return HC.pick[m.id] && !m.already_imported; });
  }
  function hcSetPicks(on) {
    hcVisible().forEach(function (m) { if (!m.already_imported) HC.pick[m.id] = on; });
    if (HC.def && !HC.pick[HC.def]) HC.def = '';
    if (on && !HC.def) { var first = hcPicked()[0]; if (first) HC.def = first.id; }
    hcRenderRows();
  }
  function hcProbe() {
    if (HC.busy) return;
    if (!String(HC.base || '').trim()) {
      HC.err = 'Give the provider base URL first.'; hcRender(); return;
    }
    HC.busy = true; HC.err = ''; hcRender();
    PDC.api('/api/providers/probe', {
      preset_id: HC.preset, base_url: String(HC.base).trim(),
      api_key: HC.key, api_key_env: HC.env, file: X ? X.which : 'user'
    }).then(function (j) {
      HC.busy = false;
      if (j._status === 400) {        // the server refused the request itself
        HC.err = j.detail || 'the GUI server refused the probe'; hcRender(); return;
      }
      HC.probed = true;
      HC.verified = !!j.ok;           // false: the list below is the curated one
      HC.err = j.ok ? '' : (j.error || j.detail || 'the provider rejected the key');
      HC.models = (j.models || []).map(function (m) {
        if (typeof m === 'string') {
          return { id: m, suggested_profile_name: sanitizeName(m.split('/').pop()) };
        }
        return m;
      });
      HC.pick = {}; HC.name = {}; HC.def = '';
      HC.models.forEach(function (m) {
        HC.pick[m.id] = !m.already_imported;
        HC.name[m.id] = m.suggested_profile_name
          || sanitizeName(String(m.id).split('/').pop());
      });
      var first = hcPicked()[0];
      if (first) HC.def = first.id;   // the first ticked model is the default
      hcRender();
    });
  }
  function hcImport() {
    if (HC.importing) return;
    var sel = hcPicked();
    if (!sel.length) return;
    var list = sel.map(function (m) {
      return { model: m.id,
               profile_name: HC.name[m.id] || m.suggested_profile_name || '',
               set_default: HC.def === m.id };
    });
    var bad = list.filter(function (m) { return !/^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$/.test(m.profile_name); });
    if (bad.length) { PDC.toast('Profile names may use letters, digits, - and _: ' + bad.map(function (m) { return '"' + m.profile_name + '"'; }).join(', '), true); return; }
    HC.importing = true; hcRenderRows();
    PDC.api('/api/providers/import', {
      preset_id: HC.preset, base_url: String(HC.base).trim(),
      api_key: HC.key, api_key_env: HC.env, file: X ? X.which : 'user',
      models: list
    }).then(function (j) {
      HC.importing = false;
      if (!j.ok) {
        HC.err = 'Import failed: ' + (j.detail || 'bad response');
        hcRender(); PDC.toast(HC.err, true); return;
      }
      var done = {};
      (j.imported || []).concat(j.skipped || []).forEach(function (nm) { done[nm] = 1; });
      HC.models.forEach(function (m) {
        if (done[HC.name[m.id] || m.suggested_profile_name]) {
          m.already_imported = true; HC.pick[m.id] = false;
          if (HC.def === m.id) HC.def = '';
        }
      });
      if (X) { X.data = j; X.pending = {}; pickDefaults(); render(); }
      PDC.toast((j.imported || []).length + ' model-named profile(s) added'
        + ((j.skipped || []).length ? ', ' + j.skipped.length + ' already there' : '')
        + (j.default_profile ? '; default is "' + j.default_profile + '"' : ''));
      PDC.refreshState();
      hcRenderRows();
    });
  }
  function hcRenderRows() {              // rebuild only the model list: the
    if (!HC.rows) return;                // filter box keeps focus and state
    var box = PDC.clear(HC.rows);
    var head = h('tr', {}, h('th', { text: 'import' }), h('th', { text: 'default' }),
      h('th', { text: 'profile name' }), h('th', { text: 'model' }), h('th', { text: '' }));
    var tbody = h('tbody', {});
    hcVisible().forEach(function (m) {
      var nm = HC.name[m.id] === undefined
        ? (m.suggested_profile_name || sanitizeName(String(m.id).split('/').pop()))
        : HC.name[m.id];
      var name = h('input', { type: 'text', class: 'mono', value: nm, disabled: !!m.already_imported,
        autocomplete: 'off', spellcheck: false });
      name.addEventListener('input', function () {
        var fixed = sanitizeName(name.value);
        if (fixed !== name.value) name.value = fixed;   // "3.8" → "3_8" as you type
        HC.name[m.id] = name.value;
      });
      var cb = h('input', { type: 'checkbox', checked: !!HC.pick[m.id], disabled: !!m.already_imported,
        onchange: function (e) { HC.pick[m.id] = e.target.checked; if (!e.target.checked && HC.def === m.id) HC.def = ''; hcRenderRows(); } });
      var rd = h('input', { type: 'radio', name: 'hcdefault', checked: HC.def === m.id, disabled: !!m.already_imported,
        onchange: function () { HC.def = m.id; if (!HC.pick[m.id]) HC.pick[m.id] = true; hcRenderRows(); } });
      tbody.appendChild(h('tr', { class: m.already_imported ? 'imp' : '' },
        h('td', {}, cb), h('td', {}, rd), h('td', {}, name),
        h('td', { class: 'mono small-text', text: m.id, title: m.id }),
        h('td', {}, m.already_imported ? h('span', { class: 'tag', text: 'already imported' }) : null)));
    });
    box.appendChild(h('div', { class: 'hp-models' },
      h('table', { class: 'scan-table' }, h('thead', {}, head), tbody)));
    var n = hcPicked().length;
    box.appendChild(h('div', { class: 'hp-foot' },
      h('button', { class: 'primary', text: HC.importing ? 'Importing\u2026' : 'Import Selected (' + n + ')',
        disabled: !n || HC.importing, onclick: hcImport }),
      h('span', { class: 'muted small-text grow',
        text: 'one profile per model, named after the model (the radio picks the default); an existing name gets \u201c_2\u201d' })));
  }
  function hcRender() {
    if (!HC.body) return;
    var box = PDC.clear(HC.body);
    HC.rows = null;
    // Step A: connect & verify — the key is tried live, nothing is saved yet.
    var psel = h('select', {}, (CLOUD.presets || []).map(function (x) {
      return h('option', { value: x.id, text: x.label });
    }).concat([h('option', { value: 'custom', text: 'Custom OpenAI-Compatible\u2026' })]));
    psel.value = HC.preset;
    psel.addEventListener('change', function () {
      HC.preset = psel.value; hcApplyPreset(); hcRender();
    });
    var bIn = h('input', { type: 'text', class: 'mono', value: HC.base, autocomplete: 'off',
      spellcheck: false, placeholder: 'https://api.provider.com/v1' });
    bIn.addEventListener('input', function () { HC.base = bIn.value; });
    var kIn = h('input', { type: 'password', class: 'mono', value: HC.key, autocomplete: 'off',
      placeholder: 'paste the API key \u2014 or leave blank to read it from the env var' });
    kIn.addEventListener('input', function () { HC.key = kIn.value; });
    var eIn = h('input', { type: 'text', class: 'mono', value: HC.env, autocomplete: 'off',
      placeholder: 'env var holding the key, e.g. DEEPSEEK_API_KEY' });
    eIn.addEventListener('input', function () { HC.env = eIn.value.trim(); });
    box.appendChild(h('div', { class: 'form hp-form' },
      h('label', { text: 'Provider' }), psel,
      h('label', { text: 'Base URL' }), bIn,
      h('label', { text: 'API key' }), kIn,
      h('label', { text: 'or env var' }), eIn));
    box.appendChild(h('div', { class: 'hp-foot' },
      h('button', { class: 'primary', text: HC.busy ? 'Connecting\u2026' : 'Connect & Fetch Models',
        disabled: HC.busy, onclick: hcProbe }),
      h('span', { class: 'muted small-text grow',
        text: HC.probed ? 'live model list from ' + HC.base.trim()
                        : 'one GET to <base URL>/models (8s timeout) \u2014 nothing is written yet' })));
    if (HC.err) box.appendChild(h('div', { class: 'note-box err',
      text: HC.err + (HC.probed ? ' \u2014 the key could not be verified; the curated list below stays importable.' : '') }));
    if (HC.verified) box.appendChild(h('div', { class: 'hp-badge ok',
      text: '\u2713 Connected \u2014 ' + HC.models.length + ' model(s) available' }));
    if (HC.busy) { box.appendChild(h('div', { class: 'tip', text: 'Asking the provider what it has\u2026' })); return box; }
    if (!HC.probed) return box;
    // Step B: tick models, edit slugs, import as model-named profiles.
    if (!HC.models.length) {
      box.appendChild(h('div', { class: 'tip', text: 'That server reported no chat models.' }));
      return box;
    }
    var search = h('input', { type: 'text', class: 'grow hp-search', value: HC.search,
      placeholder: 'filter models\u2026', spellcheck: false, autocomplete: 'off' });
    search.addEventListener('input', function () { HC.search = search.value; hcRenderRows(); });
    box.appendChild(h('div', { class: 'hp-tools' }, search,
      h('button', { class: 'small ghost', text: 'Select All', onclick: function () { hcSetPicks(true); } }),
      h('button', { class: 'small ghost', text: 'Unselect All', onclick: function () { hcSetPicks(false); } }),
      h('span', { class: 'muted small-text', text: 'embedding / speech / moderation models are filtered out' })));
    HC.rows = h('div', {});
    box.appendChild(HC.rows);
    hcRenderRows();
    return box;
  }
  function openHostedDialog() {
    loadPresets();                            // at most one HTTP attempt; the
    hcSeed();                                 // fallback catalog is live
    HC.body = h('div', { class: 'hp-wrap' });
    HC.modal = PDC.modal('Add Hosted Provider', HC.body, [{ label: 'Close' }], { wide: true });
    hcRender();
    var f = HC.modal.box && HC.modal.box.querySelector('.hp-search');
    if (f) f.focus();
  }

  function rowKey(r) { return normUrl(r.server_url) + '::' + r.model_id; }
  function pickedRows() { return SCAN.rows.filter(function (r) { return SCAN.pick[rowKey(r)]; }); }
  function groupRows(skey) { return SCAN.rows.filter(function (r) { return normUrl(r.server_url) === skey; }); }
  function setPicks(rows, on) {
    rows.forEach(function (r) {
      if (!r.already_imported) SCAN.pick[rowKey(r)] = on;
    });
    if (SCAN.def && !SCAN.pick[SCAN.def]) SCAN.def = '';
    if (on && !SCAN.def) { var first = pickedRows()[0]; if (first) SCAN.def = rowKey(first); }
    render();
  }
  function fmtSize(bytes) {
    if (!bytes) return '';
    var gb = bytes / 1073741824;
    return gb >= 1 ? gb.toFixed(1) + ' GB' : Math.round(bytes / 1048576) + ' MB';
  }
  function discoveryTab() {
    var f = file(), names = Object.keys(f.profiles || {}).sort();
    var box = h('div', {});
    // Local-first: the tab leads with scanning this machine. Hosted providers
    // are reachable through the dialog button below — never as a card that
    // competes with the local scanner for attention.
    box.appendChild(h('div', { class: 'muted small-text', style: 'margin:0 0 6px',
      text: 'Local servers \u2014 scan the models running on this machine and import them:' }));
    box.appendChild(scanner());
    box.appendChild(h('div', { class: 'muted small-text', style: 'margin:14px 0 6px', text: 'Profiles in ' + (X.which === 'user' ? 'the user config' : 'this project') + ' \u2014 calibrate the ones you actually use:' }));
    if (!names.length) {
      box.appendChild(h('div', { class: 'tip', text: 'No profiles yet. Scan a local server above, add a hosted provider below, or create one manually \u2014 then import the models you want to use.' }));
    } else {
    var defName = (f.settings || {}).default_profile || '';
    names.forEach(function (n) {
      var p = f.profiles[n] || {};
      var sub = (p.model || '(no model set)') + (p.base_url ? ' @ ' + p.base_url : '');
      // Two-part card: the identity block (name + pills over the model line)
      // shrinks and ellipsises; the action buttons never wrap mid-label.
      var row = h('div', { class: 'prow' },
        h('div', { class: 'prow-main' },
          h('div', { class: 'prow-head' },
            h('b', { class: 'prow-name ellip', text: n, title: 'open this profile', style: 'cursor:pointer', onclick: function () { gotoProfile(n); } }),
            n === X.data.active_profile ? h('span', { class: 'tag act', text: 'active' }) : null,
            n === defName ? h('span', { class: 'tag', text: 'default' }) : null),
          h('div', { class: 'prow-sub mono ellip', text: sub, title: sub })),
        h('span', { class: 'prow-actions' },
          h('button', { class: 'small', text: 'Calibrate', title: 'Measure this model and store the scores in its model card', onclick: function () { calibrate(n, 'quick'); } }),
          n === defName ? h('button', { class: 'small ghost', disabled: true, text: 'Default' })
            : h('button', { class: 'small', text: 'Set default', onclick: function () { setDefaultProfile(n); } }),
          h('button', { class: 'small', text: 'Rename', title: 'Rename this profile (the default-profile and role references follow it)', onclick: function () { renameProfileNamed(n); } }),
          h('button', { class: 'small danger', text: 'Delete', onclick: function () { delProfileNamed(n); } })));
      box.appendChild(row);
    });
    }
    box.appendChild(h('div', { class: 'row', style: 'margin-top:10px' },
      h('button', { class: 'small', text: '+ New profile', onclick: newProfile }),
      h('button', { class: 'small', text: '+ Add Hosted Provider\u2026', onclick: openHostedDialog,
        title: 'Connect a hosted API key, verify it live, and import its models as profiles' }),
      h('span', { class: 'muted small-text grow', text: 'Calibrating takes about a minute and streams to the Output tab.' })));
    return box;
  }
  function scanner() {
    var box = h('div', {});
    var url = h('input', { type: 'text', class: 'grow mono', placeholder: 'another server URL (optional)', value: SCAN.custom });
    url.addEventListener('input', function () { SCAN.custom = url.value; });
    var vram = h('input', { type: 'number', min: '1', step: '1', placeholder: 'e.g. 12', value: SCAN.vram });
    vram.addEventListener('input', function () { SCAN.vram = vram.value; });
    box.appendChild(h('div', { class: 'note-box' },
      'Find the models your local servers have loaded, then import only the ones you want. A scan reads nothing: nothing is written to the config until you press Import.',
      h('div', { class: 'row scan-controls', style: 'margin-top:8px' },
        h('button', { class: 'primary', text: SCAN.busy ? 'Scanning\u2026' : (SCAN.rows.length ? 'Scan again' : 'Scan local servers'), disabled: SCAN.busy, onclick: doScan }),
        url,
        h('label', { class: 'scan-vram' }, 'VRAM (GB)', vram)),
      // the hint gets its own line: inline in the row it competed with the
      // controls for width and clipped to half a sentence
      h('div', { class: 'muted small-text scan-hint', text: 'probes LM Studio, Ollama, llama.cpp and your existing profiles' })));
    if (SCAN.err) box.appendChild(h('div', { class: 'note-box err', text: SCAN.err }));
    if (SCAN.busy) { box.appendChild(h('div', { class: 'tip', text: 'Asking each server what it has\u2026' })); return box; }
    if (!SCAN.rows.length) return box;
    // pick whole sets at once: with 15+ GGUFs on one server you tick the two
    // you want by unselecting everything, not by hunting fourteen checkboxes
    box.appendChild(h('div', { class: 'row', style: 'margin:10px 0 4px' },
      h('button', { class: 'small ghost', text: 'Select all', onclick: function () { setPicks(SCAN.rows, true); } }),
      h('button', { class: 'small ghost', text: 'Unselect all', onclick: function () { setPicks(SCAN.rows, false); } }),
      h('span', { class: 'muted small-text grow', text: 'tick only the models to import; the name box fixes version dots (\u201c3.8\u201d \u2192 \u201c3_8\u201d) as you type' })));
    var head = h('tr', {}, h('th', { text: 'import' }), h('th', { text: 'default' }), h('th', { text: 'profile name' }),
      h('th', { text: 'model' }), h('th', { text: 'ctx' }), h('th', { text: 'size' }), h('th', { text: '' }));
    var tbody = h('tbody', {});
    var last = '';
    SCAN.rows.forEach(function (r, i) {
      var k = rowKey(r);
      var skey = normUrl(r.server_url);
      if (skey !== last) {
        last = skey;
        var grp = groupRows(skey);
        tbody.appendChild(h('tr', { class: 'scan-server' }, h('td', { colspan: '7' },
          h('div', { class: 'row sg-group' },
            h('span', { class: 'grow mono small-text ellip', text: (r.provider || 'server') + '  \u00b7  ' + r.server_url }),
            h('button', { class: 'small ghost', text: 'select all', onclick: function () { setPicks(grp, true); } }),
            h('button', { class: 'small ghost', text: 'unselect', onclick: function () { setPicks(grp, false); } })))));
      }
      var name = h('input', { type: 'text', class: 'mono', value: SCAN.name[k] === undefined ? r.suggested_profile_name : SCAN.name[k], disabled: r.already_imported });
      name.addEventListener('input', function () {
        var fixed = sanitizeName(name.value);
        if (fixed !== name.value) name.value = fixed;   // "qwen3.8" becomes "qwen3_8" live, so the import never trips on it
        SCAN.name[k] = name.value;
      });
      var cb = h('input', { type: 'checkbox', checked: !!SCAN.pick[k], disabled: !!r.already_imported,
        onchange: function (e) { SCAN.pick[k] = e.target.checked; if (!e.target.checked && SCAN.def === k) SCAN.def = ''; render(); } });
      var rd = h('input', { type: 'radio', name: 'importdefault', checked: SCAN.def === k, disabled: !!r.already_imported,
        onchange: function () { SCAN.def = k; if (!SCAN.pick[k]) { SCAN.pick[k] = true; } render(); } });
      tbody.appendChild(h('tr', { class: r.already_imported ? 'imp' : '' },
        h('td', {}, cb), h('td', {}, rd), h('td', {}, name),
        h('td', { class: 'mono small-text', text: r.model_id, title: r.model_id }),
        h('td', { class: 'small-text', text: r.context_length ? PDC.fmtInt(r.context_length) : '\u2014' }),
        h('td', { class: 'small-text muted', text: fmtSize(r.size_bytes) + (r.quant ? ' ' + r.quant : '') }),
        h('td', {}, r.already_imported ? h('span', { class: 'tag', text: 'already imported' })
          : r.loaded ? h('span', { class: 'tag act', text: 'loaded' }) : null)));
    });
    box.appendChild(h('table', { class: 'scan-table' }, h('thead', {}, head), tbody));
    var n = pickedRows().length;
    box.appendChild(h('div', { class: 'row', style: 'margin-top:10px' },
      h('button', { class: 'primary', text: SCAN.importing ? 'Importing\u2026' : 'Import Selected (' + n + ')', disabled: !n || SCAN.importing, onclick: doImport }),
      h('span', { class: 'muted small-text grow', text: 'each import becomes an editable profile, so you can tune per-model limits afterwards' })));
    return box;
  }
  function doScan() {
    SCAN.busy = true; SCAN.err = ''; SCAN.rows = []; render();
    var body = {};
    var extra = (SCAN.custom || '').trim();
    if (extra) body.urls = [extra];
    if (SCAN.vram) body.vram_gb = parseFloat(SCAN.vram) || 0;
    PDC.api('/api/models/scan', body).then(function (j) {
      SCAN.busy = false;
      if (!j.ok) { SCAN.err = 'Scan failed: ' + (j.detail || 'bad response'); render(); return; }
      var seenRow = {};
      SCAN.rows = (j.models || []).filter(function (r) {   // one row per server+model, whatever URL spelling it arrived under
        var rk = rowKey(r);
        if (seenRow[rk]) return false;
        seenRow[rk] = 1; return true;
      });
      SCAN.pick = {}; SCAN.name = {}; SCAN.def = '';
      SCAN.rows.forEach(function (r) {
        var k = rowKey(r);
        SCAN.pick[k] = !r.already_imported;
        SCAN.name[k] = r.suggested_profile_name;
      });
      if (!SCAN.rows.length) SCAN.err = 'No server answered. Start LM Studio / Ollama / llama.cpp, or type its URL above (it must end in /v1 for OpenAI-style servers).';
      else {
        var first = SCAN.rows.filter(function (r) { return SCAN.pick[rowKey(r)]; })[0];
        if (first) SCAN.def = rowKey(first);
      }
      render();
    });
  }
  function doImport() {
    var sel = pickedRows();
    if (!sel.length) return;
    var vram = parseFloat(SCAN.vram) || 0;
    var list = sel.map(function (r) {
      var k = rowKey(r);
      var pname = sanitizeName(SCAN.name[k] || r.suggested_profile_name);
      SCAN.name[k] = pname;
      return { profile_name: pname, base_url: r.server_url,
        model: r.model_id, context_length: r.context_length || 0, vram_gb: vram,
        set_default: SCAN.def === k };
    });
    var bad = list.filter(function (m) { return !/^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$/.test(m.profile_name); });
    if (bad.length) { PDC.toast('Profile names may use letters, digits, - and _: ' + bad.map(function (m) { return '"' + m.profile_name + '"'; }).join(', '), true); return; }
    SCAN.importing = true; render();
    PDC.api('/api/models/import', { models: list }).then(function (j) {
      SCAN.importing = false;
      if (!j.ok) { SCAN.err = 'Import failed: ' + (j.detail || 'bad response'); render(); return; }
      var done = {};
      (j.imported || []).concat(j.skipped || []).forEach(function (nm) { done[nm] = 1; });
      SCAN.rows.forEach(function (r) {
        var k = rowKey(r);
        if (done[SCAN.name[k] || r.suggested_profile_name]) { r.already_imported = true; SCAN.pick[k] = false; if (SCAN.def === k) SCAN.def = ''; }
      });
      X.data = j; pickDefaults(); render();
      PDC.toast((j.imported || []).length + ' profile(s) imported' + ((j.skipped || []).length ? ', ' + j.skipped.length + ' already there' : '') + (j.default_profile ? '; default is "' + j.default_profile + '"' : ''));
      PDC.refreshState();
    });
  }
  function calibrate(name, mode) {
    PDC.confirm('Calibrate \u201c' + name + '\u201d?\n\n' + (mode === 'quick'
      ? 'A short probe: a few timed calls to measure how fast this model answers, how long a prompt it can actually hold, and how reliably it uses tools. The result is stored in its model card and the tuned limits are written to the profile.'
      : 'The full suite \u2014 slower and more calls, but it also checks tool calls that must happen in order.'), 'Run it').then(function (ok) {
      if (!ok) return;
      PDC.api('/api/models/calibrate', { profile: name, mode: mode }).then(function (j) {
        if (!j.ok) { PDC.toast(j.detail || 'could not start the calibration', true); return; }
        PDC.toast('Calibrating "' + name + '" \u2014 watch the Output tab.');
        PDC.refreshState();
      });
    });
  }
  function setDefaultProfile(name) {
    save({ file: X.which, profile: null, changes: { default_profile: name } }, '/api/settings/set', '"' + name + '" is now the default profile.');
  }
  function delProfileNamed(name) {
    PDC.confirm('Delete profile \u201c' + name + '\u201d from ' + (X.which === 'user' ? 'the user config' : 'this project') + '?\n\nA .bak copy of the file is kept, so this can be undone.', 'Delete', true).then(function (ok) {
      if (ok) save({ file: X.which, action: 'delete', name: name }, '/api/settings/profile', 'Profile deleted.').then(function () { X.profile = ''; pickDefaults(); render(); });
    });
  }
  function renameProfileNamed(oldName) {
    askName('Rename profile \u201c' + oldName + '\u201d to', oldName, function (raw) {
      var name = sanitizeName(raw);
      if (!name || name === oldName) return;
      if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$/.test(name)) {
        PDC.toast('Profile names may use letters, digits, - and _: "' + name + '"', true);
        return;
      }
      // The backend retargets default_profile / roles and reloads STATE;
      // save() re-renders the modal and PDC.refreshState() updates the
      // header Profile dropdown in one shot.
      save({ file: X.which, old_name: oldName, new_name: name }, '/api/profiles/rename',
        'Renamed \u201c' + oldName + '\u201d to \u201c' + name + '\u201d.')
        .then(function (j) { if (j && j.ok) { if (X.profile === oldName) X.profile = name; render(); } });
    });
  }
  function gotoProfile(name) { X.profile = name; X.pending = {}; X.tab = 'profiles'; render(); }

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
      h('button', { class: 'small', text: 'Rename', disabled: !X.profile, title: 'Rename this profile (the default-profile and role references follow it)', onclick: function () { renameProfileNamed(X.profile); } }),
      h('button', { class: 'small danger', text: 'Delete', disabled: !X.profile, onclick: delProfile }),
      // The hosted-provider dialog lives right in this toolbar (it used to be
      // a card on top of both tabs): whoever manages profiles can connect a
      // hosted API and import its models without detouring anywhere.
      h('button', { class: 'small', text: '+ Add Hosted Provider\u2026', onclick: openHostedDialog,
        title: 'Connect a hosted API key, verify it live, and import its models as profiles' })));
    return h('div', {},
      h('div', { class: 'set-layout' }, side,
        X.profile ? profileForm(f.profiles[X.profile] || {})
          : h('div', { class: 'tip', text: 'Create a profile to point pedacito at a model.' })));
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

  function profileControl(form, key, body) {
    var item = schemaItem(key); if (!item) return;
    var cur = key === 'api_key' ? undefined : body[key];
    var el = control(item, cur, function (v) { setPending(key, v); }, placeholderFor(item), inheritLabel(item));
    if (key === 'model') el.setAttribute('list', 'modelOptions');
    form.appendChild(h('label', { text: key, title: item.help }));
    form.appendChild(key === 'model' ? h('div', { class: 'row' }, h('div', { class: 'grow' }, el), h('datalist', { id: 'modelOptions' }, X.models.map(function (m) { return h('option', { value: m }); }))) : el);
    if (item.help) form.appendChild(h('div', { class: 'help', text: item.help.length > 170 ? item.help.slice(0, 167) + '\u2026' : item.help }));
  }
  function profileForm(body) {
    var name = X.profile;
    // Reasoning & execution overrides go FIRST: on a small local model these
    // decide the quality of every answer, and they are exactly what a
    // calibration writes; the generic form lists the connection settings.
    var ovr = h('div', { class: 'form' });
    REASONING.forEach(function (key) { profileControl(ovr, key, body); });
    var form = h('div', { class: 'form' });
    COMMON.forEach(function (key) { profileControl(form, key, body); });
    var result = h('div', { class: 'muted small-text', style: 'margin-top:8px' });
    var extras = Object.keys(body).filter(function (k) {
      return COMMON.indexOf(k) < 0 && REASONING.indexOf(k) < 0 && k.charAt(0) !== '_'; });
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
    return h('div', {},
      h('div', { class: 'ovr-group' },
        h('div', { class: 'ovr-head' },
          h('b', { text: 'Reasoning & Execution Overrides' }),
          h('span', { class: 'muted small-text', text: 'how this model thinks and lands edits \u2014 each box offers \u201c(inherit \u2014 universal: \u2026)\u201d to drop the override again' })),
        ovr),
      h('div', { class: 'ovr-group' },
        h('div', { class: 'ovr-head' }, h('b', { text: 'Connection & Capabilities' })),
        form),
      adv, result,
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

  // ----- auto-tune universal settings -------------------------------------------
  // Four questions, one deterministic backend computation (no LLM call):
  // /api/settings/autotune previews the recommendation, and only "Apply"
  // writes the changes into the user config as [universal] values.
  var TUNE_Q = [
    ['hardware', 'Model hardware', [
      ['lt8', '\u2264 8 GB VRAM \u2014 small GPU / iGPU / CPU-only'],
      ['8to16', '8\u201316 GB VRAM \u2014 one mid GPU'],
      ['16to24', '16\u201324 GB VRAM \u2014 strong single GPU'],
      ['gt24', '> 24 GB VRAM \u2014 multi-GPU or server card'],
      ['hosted', 'no local GPU \u2014 hosted API (token money, not VRAM)']]],
    ['model', 'Model class', [
      ['small', 'small (up to \u2248 8B) \u2014 format slips, needs guard rails'],
      ['medium', 'mid-size (\u2248 9\u201330B) \u2014 solid with structure'],
      ['large', 'large (\u2265 \u2248 70B) \u2014 holds whole patches unaided'],
      ['hosted', 'frontier / hosted model']],
    ],
    ['codebase', 'Typical codebase', [
      ['small', 'small \u2014 tens of files'],
      ['medium', 'medium \u2014 hundreds of files'],
      ['large', 'large \u2014 1000+ files, many packages']]],
    ['priority', 'What should settings optimise for?', [
      ['token_saver', 'token/latency savings \u2014 finish cheap and fast'],
      ['balanced', 'balance \u2014 the recommended middle'],
      ['thorough', 'answer quality \u2014 spend more model calls']]]
  ];
  function autotuneTab() {
    var box = h('div', {});
    box.appendChild(h('div', { class: 'note-box' },
      h('b', { text: 'Auto-Tune Universal Settings' }),
      h('div', { class: 'small-text', text: 'Answer four questions and pedacito computes the universal budgets and editing guard rails that fit that machine: read and gather sizes, step and call caps, answer verification, the incremental edit loop. Nothing is written until you press Apply, and the current file is kept as a .bak copy.' })));
    var form = h('div', { class: 'form' });
    TUNE_Q.forEach(function (q) {
      var sel = h('select', {}, q[2].map(function (o) { return h('option', { value: o[0], text: o[1] }); }));
      sel.value = TUNE[q[0]];
      sel.addEventListener('change', function () { TUNE[q[0]] = sel.value; TUNE.result = null; });
      form.appendChild(h('label', { text: q[1] }));
      form.appendChild(sel);
    });
    var run = h('button', { class: 'primary', text: TUNE.busy ? 'Computing\u2026' : (TUNE.result ? 'Update preview' : 'Preview recommendation'), disabled: TUNE.busy, onclick: function () {
      TUNE.busy = true; TUNE.err = ''; render();
      PDC.api('/api/settings/autotune', { hardware: TUNE.hardware, model: TUNE.model, codebase: TUNE.codebase, priority: TUNE.priority })
        .then(function (j) {
          TUNE.busy = false;
          if (!j.ok) { TUNE.err = j.detail || 'the server refused the request'; TUNE.result = null; }
          else TUNE.result = j;
          render();
        });
    } });
    box.appendChild(form);
    box.appendChild(h('div', { class: 'row', style: 'margin-top:10px' }, run,
      h('span', { class: 'muted small-text grow', text: 'pure arithmetic \u2014 the preview costs no tokens and touches no file' })));
    if (TUNE.err) box.appendChild(h('div', { class: 'note-box err', text: TUNE.err }));
    if (!TUNE.result) return box;
    var rows = TUNE.result.changes || [];
    var changed = rows.filter(function (r) { return String(r.value) !== String(r.current); });
    var table = h('table', { class: 'tune-table' },
      h('thead', {}, h('tr', {}, h('th', { text: 'setting' }), h('th', { text: 'current' }),
        h('th', { text: 'recommended' }), h('th', { text: 'why' }))),
      h('tbody', {}, rows.map(function (r) {
        var diff = String(r.value) !== String(r.current);
        return h('tr', { class: diff ? 'diff' : 'same' },
          h('td', { class: 'mono small-text', text: r.key }),
          h('td', { class: 'mono small-text muted', text: String(r.current) }),
          h('td', { class: 'mono small-text', text: String(r.value) }),
          h('td', { class: 'small-text muted', text: r.reason }));
      })));
    box.appendChild(h('div', { class: 'muted small-text', style: 'margin:10px 0 4px',
      text: changed.length ? changed.length + ' of ' + rows.length + ' settings would change:'
                           : 'Your universal settings already match this recommendation.' }));
    box.appendChild(table);
    box.appendChild(h('div', { class: 'row', style: 'margin-top:12px' },
      h('button', { class: 'primary', text: 'Apply Recommended Universal Settings', disabled: !changed.length, onclick: function () {
        PDC.confirm('Write these ' + changed.length + ' universal settings into the user config?\n\nThe file as it stands is kept as config.toml.bak first. Profile-level overrides keep winning over these values.', 'Apply it').then(function (ok) {
          if (!ok) return;
          TUNE.busy = true; render();
          PDC.api('/api/settings/autotune', { hardware: TUNE.hardware, model: TUNE.model, codebase: TUNE.codebase, priority: TUNE.priority, apply: true })
            .then(function (j) {
              TUNE.busy = false;
              if (!j.ok) { TUNE.err = j.detail || 'the server refused the request'; render(); return; }
              if (j.payload) { X.data = j.payload; X.pending = {}; }
              TUNE.result = null;   // re-preview to confirm the new "current" column
              render();
              PDC.toast('Universal settings applied (' + (j.changes || []).length + ' written).');
              PDC.refreshState();
            });
        });
      } }),
      h('span', { class: 'muted small-text grow', text: changed.length ? 'applies the rows above to [universal] in ' + (file().path || 'the user config') : 'nothing to apply' })));
    return box;
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
          var pending = pendingFor(it.name);
          var shown = pending !== undefined ? (pending === null ? undefined : pending)
            : (X.target ? set : universalValue(it));
          // "inherit" is a profile-override concept only: the global layer is
          // the top of the user config and always shows a concrete value.
          row.appendChild(control(it, shown, function (v) { setPending(it.name, v, row); },
                                  placeholderFor(it), X.target ? inheritLabel(it) : null));
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
    if (X.focusSearch) {
      X.focusSearch = false;
      setTimeout(function () { search.focus(); search.select(); }, 30);
    }
    return h('div', {}, h('div', { class: 'row', style: 'margin-bottom:8px' }, h('span', { class: 'muted', text: 'Apply to:' }), targetSel, search, saveBtn),
      h('div', { class: 'muted small-text', style: 'margin-bottom:6px', text: X.target
        ? 'A blank box means "not set here" \u2014 the profile inherits the universal value the control names (the badge on the right says where each value comes from). Clearing a box removes the override from the file.'
        : 'Universal settings always show the value that applies (the file\u0027s own value, else the effective one, else the built-in default \u2014 the badge on the right says which). Clearing a box removes the key from the file and restores the default.' }), holder);
  }

  // ----- raw ---------------------------------------------------------------------------
  function rawTab() {
    var f = file();
    var ta = h('textarea', { id: 'rawText', spellcheck: false, value: f.text || '' });
    var msg = h('div', { class: 'muted small-text', style: 'margin-top:6px', text: 'Saved edits are checked with the same rules the loader uses; the previous file is kept as ' + (f.path || '') + '.bak.' });
    return h('div', {}, ta, msg, h('div', { class: 'row', style: 'margin-top:8px' },
      h('button', { class: 'primary', text: 'Validate & save', onclick: function () { save({ file: X.which, text: ta.value }, '/api/settings/raw', 'Config file saved.'); } }),
      h('button', { text: 'Revert', onclick: function () { ta.value = f.text || ''; } })),
      resetBox());
  }

  // ----- reset to the recommended baseline ---------------------------------------------
  function resetBox() {
    return h('div', { class: 'reset-box' },
      h('b', { text: 'Reset to baseline defaults' }),
      h('div', { class: 'small-text', text: 'Start over from the settings pedacito recommends for a local GPU: every profile and every hand-edited value in the user config is removed, the tuned defaults are written back, and the file you have now is kept as config.toml.reset.<date>.bak. Project settings are not touched.' }),
      h('div', { class: 'row', style: 'margin-top:8px' },
        h('button', { class: 'danger small', disabled: X.which !== 'user', onclick: function () {
          PDC.confirm('Reset the user config to the recommended baseline?\n\nAll profiles are removed and your current settings are replaced by the defaults pedacito recommends for a local GPU. The file as it stands is saved first as config.toml.reset.<date>.bak, so this can be undone.', 'Reset it', true).then(function (ok) {
            if (!ok) return;
            PDC.api('/api/settings/reset', {}).then(function (j) {
              if (!j.ok) { PDC.toast(j.detail || 'could not reset', true); return; }
              X.data = j; X.which = 'user'; X.profile = ''; X.pending = {}; pickDefaults(); render();
              PDC.toast('Reset done. Backup: ' + (j.backup || ''));
              PDC.refreshState();
            });
          });
        } }, 'Reset to baseline defaults'),
        X.which !== 'user' ? h('span', { class: 'muted small-text', text: 'switch to the user config to reset it' }) : null));
  }
})();
