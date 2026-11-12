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
    where replacing the entry module leaves the page silently unmounted.

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
    #                # unbound_control | mount_regression
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
  const orig = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function (type, fn, opts) {
    try {
      if (bindings.length < 2000)
        bindings.push({ type: String(type), target: desc(this) });
    } catch (e) {}
    return orig.call(this, type, fn, opts);
  };
  window.__pdcBindings = bindings;
})();
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


def check(root, *, entry: str = "", timeout: int = 45,
          baseline: dict | None = None) -> DomResult:
    """Load the entry page headless and report what the browser sees. Never
    raises: a check that cannot run returns `error` with the reason (missing
    install, no entry page, launch failure) instead of a false green."""
    root = Path(root)
    ok, hint = available()
    if not ok:
        return DomResult(entry="", error=hint)
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
    targets = {b.get("target", "") for b in bindings}
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