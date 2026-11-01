// Shared helpers: DOM builder, API calls, toasts, modals, a safe markdown
// renderer, and the terminal emulator. Everything model-written is inserted
// with textContent / text nodes, never innerHTML.
(function () {
  'use strict';
  var PDC = window.PDC = { S: { status: {}, msgs: {}, order: [], mode: 'task', attachments: [] } };

  function h(tag, props) {
    var el = document.createElement(tag);
    var p = props || {};
    Object.keys(p).forEach(function (k) {
      var v = p[k];
      if (v == null || v === false) return;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k.slice(0, 2) === 'on') el.addEventListener(k.slice(2), v);
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k === 'style') el.style.cssText = v;
      else if (k in el && k !== 'list') el[k] = v;
      else el.setAttribute(k, v === true ? '' : v);
    });
    for (var i = 2; i < arguments.length; i++) add(el, arguments[i]);
    return el;
  }
  function add(el, kid) {
    if (kid == null || kid === false) return;
    if (Array.isArray(kid)) { kid.forEach(function (k) { add(el, k); }); return; }
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  PDC.h = h;
  PDC.$ = function (id) { return document.getElementById(id); };
  PDC.clear = function (el) { while (el.firstChild) el.removeChild(el.firstChild); return el; };

  PDC.api = function (path, body, method) {
    var opts = body === undefined && !method
      ? { method: 'GET' }
      : { method: method || 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) };
    return fetch(path, opts).then(function (r) {
      return r.json().then(function (j) { j._status = r.status; return j; },
        function () { return { ok: false, detail: 'bad response', _status: r.status }; });
    });
  };

  PDC.toast = function (text, isErr) {
    var t = h('div', { class: 'toast' + (isErr ? ' err' : ''), text: text });
    PDC.$('toast').appendChild(t);
    setTimeout(function () { t.remove(); }, isErr ? 7000 : 3500);
  };

  PDC.fmtInt = function (n) { return Number(n || 0).toLocaleString(); };
  PDC.fmtTokens = function (n) {
    n = Number(n || 0);
    return n >= 10000 ? (n / 1000).toFixed(1) + 'k' : String(n);
  };

  // --- modal --------------------------------------------------------------------
  PDC.modal = function (title, body, buttons, opts) {
    opts = opts || {};
    var overlay = h('div', { class: 'overlay' });
    var close = function () { overlay.remove(); document.removeEventListener('keydown', onKey); if (opts.onClose) opts.onClose(); };
    var onKey = function (e) { if (e.key === 'Escape') close(); };
    var foot = h('div', { class: 'modal-foot' });
    (buttons || []).forEach(function (b) {
      foot.appendChild(h('button', {
        class: b.primary ? 'primary' : (b.danger ? 'danger' : ''), text: b.label,
        onclick: function () { var keep = b.onclick && b.onclick(close); if (keep !== false && !b.stay) close(); }
      }));
    });
    var box = h('div', { class: 'modal' + (opts.narrow ? ' narrow' : '') },
      h('div', { class: 'modal-head' }, h('h3', { text: title }),
        h('button', { class: 'ghost', text: '\u2715', onclick: close })),
      h('div', { class: 'modal-body' }, body),
      (buttons && buttons.length) ? foot : null);
    overlay.appendChild(box);
    overlay.addEventListener('mousedown', function (e) { if (e.target === overlay) close(); });
    document.addEventListener('keydown', onKey);
    document.body.appendChild(overlay);
    return { close: close, box: box, body: box.querySelector('.modal-body') };
  };

  PDC.confirm = function (text, okLabel, danger) {
    return new Promise(function (resolve) {
      var done = false;
      PDC.modal('Please confirm', h('p', { text: text, style: 'white-space:pre-wrap;margin:0' }), [
        { label: 'Cancel', onclick: function () { done = true; resolve(false); } },
        { label: okLabel || 'OK', primary: !danger, danger: !!danger, onclick: function () { done = true; resolve(true); } }
      ], { narrow: true, onClose: function () { if (!done) resolve(false); } });
    });
  };

  // --- markdown (safe: builds nodes, never parses HTML) ---------------------------
  function inline(text) {
    var frag = document.createDocumentFragment();
    var re = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\[[^\]\n]+\]\(https?:\/\/[^)\s]+\))/g;
    var last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) frag.append(text.slice(last, m.index));
      if (m[1]) frag.append(h('code', { text: m[1].slice(1, -1) }));
      else if (m[2]) frag.append(h('b', { text: m[2].slice(2, -2) }));
      else {
        var mm = /\[([^\]]+)\]\(([^)]+)\)/.exec(m[3]);
        frag.append(h('a', { href: mm[2], text: mm[1], target: '_blank', rel: 'noopener noreferrer' }));
      }
      last = m.index + m[0].length;
    }
    if (last < text.length) frag.append(text.slice(last));
    return frag;
  }
  PDC.md = function (text) {
    var root = h('div', { class: 'md' });
    var lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
    var i = 0, para = [], list = null;
    function flushPara() { if (para.length) { var p = h('p'); p.append(inline(para.join('\n'))); root.append(p); para = []; } }
    function flushList() { list = null; }
    while (i < lines.length) {
      var ln = lines[i];
      var fence = /^\s*```\s*([\w+-]*)\s*$/.exec(ln);
      if (fence) {
        flushPara(); flushList();
        var buf = []; i++;
        while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        root.append(h('pre', {}, fence[1] ? h('span', { class: 'lang', text: fence[1] }) : null, h('code', { text: buf.join('\n') })));
        continue;
      }
      var hd = /^(#{1,4})\s+(.*)$/.exec(ln);
      if (hd) { flushPara(); flushList(); var hh = h('h' + hd[1].length); hh.append(inline(hd[2])); root.append(hh); i++; continue; }
      var li = /^\s*([-*]|\d+[.)])\s+(.*)$/.exec(ln);
      if (li) {
        flushPara();
        var ordered = /\d/.test(li[1]);
        if (!list || list.ordered !== ordered) { list = { el: h(ordered ? 'ol' : 'ul'), ordered: ordered }; root.append(list.el); }
        var item = h('li'); item.append(inline(li[2])); list.el.append(item); i++; continue;
      }
      if (!ln.trim()) { flushPara(); flushList(); i++; continue; }
      flushList(); para.push(ln); i++;
    }
    flushPara();
    return root;
  };

  // --- terminal -----------------------------------------------------------------------
  // A small ANSI-aware line buffer: SGR colours, carriage-return overwrite, and
  // a line cap. Rendering is batched per animation frame.
  function Term(el) {
    this.el = el; this.lines = []; this.cur = null; this.style = ''; this.fresh = false;
    this.max = 4000; this.raf = 0; this.follow = true; this.newLine();
  }
  Term.prototype.newLine = function () {
    var div = h('div', { class: 'tl' });
    this.el.appendChild(div);
    this.cur = div; this.lines.push(div);
    if (this.lines.length > this.max) { var old = this.lines.shift(); old.remove(); }
  };
  Term.prototype.span = function (text) {
    if (!text) return;
    if (this.fresh) { PDC.clear(this.cur); this.fresh = false; }
    if (this.style) this.cur.appendChild(h('span', { class: this.style, text: text }));
    else this.cur.appendChild(document.createTextNode(text));
  };
  Term.prototype.sgr = function (codes) {
    var cls = [];
    if (!codes.length) codes = [0];
    codes.forEach(function (c) {
      c = Number(c);
      if (c === 0) { cls = []; }
      else if (c === 1) cls.push('b');
      else if (c === 2) cls.push('dim');
      else if ((c >= 30 && c <= 37)) cls.push('c' + c);
      else if (c >= 90 && c <= 97) cls.push('c' + (c - 60));
    });
    this.style = cls.join(' ');
  };
  Term.prototype.write = function (text) {
    var re = /\x1b\[([0-9;?]*)([A-Za-z])|\r\n|\n|\r|\x1b[^[]/g;
    var last = 0, m;
    while ((m = re.exec(text))) {
      this.span(text.slice(last, m.index));
      last = m.index + m[0].length;
      if (m[2] === 'm') this.sgr(m[1] ? m[1].split(';') : []);
      else if (m[2] === 'K' || m[2] === 'J') { PDC.clear(this.cur); }
      else if (m[0] === '\n' || m[0] === '\r\n') { this.newLine(); this.fresh = false; }
      else if (m[0] === '\r') { this.fresh = true; }
    }
    this.span(text.slice(last));
    this.schedule();
  };
  Term.prototype.schedule = function () {
    var self = this;
    if (this.raf) return;
    this.raf = requestAnimationFrame(function () {
      self.raf = 0;
      if (self.follow) self.el.scrollTop = self.el.scrollHeight;
    });
  };
  Term.prototype.reset = function (text) {
    PDC.clear(this.el); this.lines = []; this.style = ''; this.fresh = false; this.newLine();
    if (text) this.write(text);
  };
  PDC.Term = Term;
})();
