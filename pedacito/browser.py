"""Optional headless-browser driver for the gather-loop `browser` action.

Navigates to a URL, presses keys/clicks/evaluates JS, waits, and captures the
page using Playwright (an optional extra)::

    pip install "pedacito-local[browser]"
    python -m playwright install chromium

Without Playwright, a system Chrome/Chromium `--headless --screenshot` fallback
can load and capture a page but cannot interact; with neither, the tool
returns an install hint.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import proc

# A URL printed by a dev server, including the bind-all 0.0.0.0 spelling
# (`normalise_url` makes it openable).
_URL_RE = re.compile(
    r"https?://(?:\[[0-9a-fA-F:]+\]|localhost|127\.0\.0\.1|0\.0\.0\.0|"
    r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)(?::\d{1,5})?"
    r"(?:/[^\s'\"<>]*)?", re.IGNORECASE)

_CHROME_NAMES = (
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "chrome", "msedge", "microsoft-edge",
)
# GUI installs not on PATH (macOS app bundles, Windows defaults).
_CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def normalise_url(url: str) -> str:
    """Make a server-printed URL openable by a browser on this machine."""
    u = (url or "").strip().strip("`'\"")
    u = re.sub(r"^(https?://)0\.0\.0\.0(?=[:/]|$)", r"\g<1>127.0.0.1", u,
               flags=re.IGNORECASE)
    u = re.sub(r"^(https?://)\[::\](?=[:/]|$)", r"\g<1>127.0.0.1", u,
               flags=re.IGNORECASE)
    return u


def find_server_url(text: str) -> str:
    """First plausible server URL in command output, or ''.

    Trailing punctuation is stripped because dev servers often print
    `Running on http://127.0.0.1:5000 (Press CTRL+C to quit)`.
    """
    text = _ANSI_RE.sub("", text or "")
    for m in _URL_RE.finditer(text):
        url = normalise_url(m.group(0).rstrip(".,;:)]}"))
        if url:
            return url
    return ""


def available() -> bool:
    """True when Playwright's Python package can be imported."""
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except Exception:
        return False


def chrome_executable() -> str:
    """Path to a Chrome/Chromium binary usable for a navigation-only fallback."""
    for name in _CHROME_NAMES:
        path = shutil.which(name)
        if path:
            return path
    for path in _CHROME_PATHS:
        if Path(path).is_file():
            return path
    return ""


def install_hint() -> str:
    """Actionable error text when no browser backend is present."""
    return ("Browser automation is not installed. Install Playwright and a "
            "Chromium build:\n"
            "  pip install \"pedacito-local[browser]\"\n"
            "  python -m playwright install chromium\n"
            "A system Chrome/Chromium enables a navigation-only fallback "
            "(no key presses or clicks). Without either, do NOT treat a "
            "desktop screenshot as evidence of the web page: use `run` with "
            "a console/headless check instead, or report that the UI could "
            "not be driven.")


def parse_actions(spec: str) -> list[tuple]:
    """Parse the step's `pattern` field into (kind, arg) actions. A tiny
    grammar, separated by semicolons:

        Enter                    -> press Enter
        press Enter              -> press a key
        click #start             -> click a CSS/text selector
        fill #name Ada           -> fill a field
        type hello               -> type into the focused element
        wait 2                   -> wait two seconds
        eval x.y = 3             -> run JS in the page and report its value;
                                     an eval consumes the REST of the spec
                                     (put it last; `;` inside it is fine)
    """
    out: list[tuple] = []
    parts = [raw.strip() for raw in (spec or "").split(";")]
    for i, part in enumerate(parts):
        if not part:
            continue
        head, _, rest = part.partition(" ")
        kind = head.lower()
        rest = rest.strip()
        if kind in ("eval", "js"):
            # The rest of the pattern is one JS expression: splitting on `;`
            # would shred multi-statement code.
            expr = "; ".join(p for p in (rest, *parts[i + 1:]) if p)
            if expr:
                out.append(("eval", expr))
            break
        if kind in ("press", "key"):
            out.append(("press", rest or "Enter"))
        elif kind == "click":
            if rest:
                out.append(("click", rest))
        elif kind == "fill":
            selector, _, text = rest.partition(" ")
            if selector:
                out.append(("fill", (selector, text)))
        elif kind == "type":
            out.append(("type", rest))
        elif kind == "wait":
            try:
                out.append(("wait", max(0.0, min(float(rest), 60.0))))
            except (TypeError, ValueError):
                pass
        elif len(part.split()) == 1:
            out.append(("press", part))
    return out


class _OutputReader:
    """Drain a child's combined stdout/stderr into a bounded buffer (the first
    `limit` bytes) on a background thread."""

    def __init__(self, stream, limit: int = 20_000):
        self.stream = stream
        self.limit = limit
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        if self.stream is None:
            return
        while True:
            try:
                chunk = self.stream.read1(65536)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            with self._lock:
                if len(self._buf) < self.limit:
                    self._buf.extend(chunk[:self.limit - len(self._buf)])

    def text(self) -> str:
        with self._lock:
            return bytes(self._buf).decode("utf-8", "replace")


def launch_server(argv: list[str], *, cwd, env: dict | None = None,
                  wait_seconds: int = 15) -> tuple:
    """Start a server and wait for it to print a URL.

    Returns (process, url, output); the process is left RUNNING and the caller
    must `proc.terminate` it. Returns (None, "", error) if it cannot start.
    """
    try:
        p = proc.spawn(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    except OSError as e:
        return None, "", str(e)
    reader = _OutputReader(p.stdout)
    deadline = time.monotonic() + max(1, int(wait_seconds or 1))
    url = ""
    while time.monotonic() < deadline:
        url = find_server_url(reader.text())
        if url:
            break
        if p.poll() is not None:
            break
        time.sleep(0.1)
    if url:
        _wait_until_listening(url)
    return p, url, reader.text()


def _wait_until_listening(url: str, timeout: float = 3.0) -> None:
    """Give the HTTP listener up to `timeout` seconds to bind after it prints
    its URL."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0):
                return
        except urllib.error.HTTPError:
            return                       # server is up; 4xx is still a server
        except Exception:
            time.sleep(0.15)


def _page_text(page) -> str:
    try:
        text = page.inner_text("body")
    except Exception:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    return text[:4000]


def _canvas_info(page) -> str:
    try:
        canvases = page.evaluate(
            "() => Array.from(document.querySelectorAll('canvas'))"
            ".slice(0, 5).map(c => ({w: c.width, h: c.height}))")
    except Exception:
        return ""
    if not canvases:
        return ""
    return ", ".join(f"{c.get('w')}x{c.get('h')}" for c in canvases)


def run(url: str, *, actions: str = "", wait: float = 3.0, out=None,
        timeout: int = 20, headless: bool = True) -> tuple[bool, dict | str]:
    """Open `url`, perform `actions`, wait, and capture a screenshot.

    Returns (True, info) where info has url/title/text/console/canvas/method
    on success, or (False, error_text). The caller formats the observation.
    """
    url = normalise_url(url)
    out = Path(out) if out is not None else None
    if out is not None:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return False, f"could not create the screenshot directory: {e}"

    if available():
        return _run_playwright(url, actions=actions, wait=wait, out=out,
                               timeout=timeout, headless=headless)

    exe = chrome_executable()
    if not exe:
        return False, install_hint()
    if parse_actions(actions):
        return False, ("A system Chrome/Chromium can load and capture a page, "
                       "but cannot press keys, click, or run scripts. "
                       "Install Playwright for interactions:\n"
                       "  pip install \"pedacito-local[browser]\"\n"
                       "  python -m playwright install chromium")
    return _run_chrome(exe, url, wait=wait, out=out, timeout=timeout)


def _raced_eval(expr: str, timeout_s: int) -> str:
    """Wrap a model-authored eval in an in-page race against a timer.

    page.evaluate() has no timeout, so a Promise that never settles would hang
    the whole loop. Racing inside the page keeps the completion-value
    semantics, adopts a returned Promise, calls a bare function result, and
    turns a never-settling eval into a rejection reported as an action error.
    A synchronous infinite loop still starves the timer (not handled).
    """
    ms = max(1, int(timeout_s)) * 1000
    src = json.dumps(expr)
    return ("Promise.race([Promise.resolve().then(() => eval(" + src + "))"
            ".then(v => typeof v === 'function' ? v() : v), "
            "new Promise((_, reject) => setTimeout(() => reject(new Error("
            f"'eval did not settle within {ms // 1000}s -- a Promise whose "
            "callback threw never resolves; wrap risky page code in "
            "try/catch and always resolve')), " + str(ms) + "))])")


def _unclosed_brackets(expr: str) -> str:
    """Describe the brackets still open at the end of a JS expression.

    V8 reports an unclosed construct as a bare "Unexpected end of input" with
    no position. This scanner tracks (), [] and {} outside strings and
    comments and names the openers left on the stack, with position and
    context. Regex literals are not distinguished from division, which is fine
    since the hint is only attached to an eval that already failed to parse.
    """
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    quote: str | None = None
    i, n = 0, len(expr)
    while i < n:
        c = expr[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
            elif quote != "`" and c == "\n":
                quote = None      # unterminated one-line string; JS flags it
        elif c in "'\"`":
            quote = c
        elif c == "/" and expr[i + 1:i + 2] == "/":
            j = expr.find("\n", i)
            i = n if j < 0 else j
            continue
        elif c == "/" and expr[i + 1:i + 2] == "*":
            j = expr.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        elif c in "([{":
            stack.append((c, i))
        elif c in ")]}":
            if stack and stack[-1][0] == pairs[c]:
                stack.pop()
            elif stack:
                ch, pos = stack[-1]
                return (f"brackets cross at char {i}: found `{c}` but the "
                        f"innermost still-open bracket is `{ch}` from char "
                        f"{pos} (`{expr[max(0, pos - 16):pos + 16]}`)")
        i += 1
    if quote:
        return (f"a {quote}-quoted string is still open at end of the "
                "expression (unterminated string literal)")
    if stack:
        bits = []
        for ch, pos in stack[-3:]:
            ctx = expr[max(0, pos - 16):pos + 16].replace("\n", " ")
            bits.append(f"`{ch}` from char {pos} (`{ctx}`)")
        extra = f" (+{len(stack) - 3} more)" if len(stack) > 3 else ""
        return ("unclosed at end of the expression: " + "; ".join(bits)
                + extra + " -- the source ends before these close; the "
                "missing closer belongs near the tail")
    return ""


def _eval_syntax_error(arg: str, msg: str) -> str:
    """Action-error text for an eval that failed to parse: the length received
    (so the model does not blame truncation), the bracket report V8 omits, and
    the source tail where a missing closer belongs."""
    first = msg.split("\n", 1)[0].strip()[:120]
    text = (f"eval SYNTAX ERROR -- {first} -- the tool received all "
            f"{len(arg)} chars of the source intact; nothing was "
            "truncated in transit.")
    hint = _unclosed_brackets(arg)
    if hint:
        text += " " + hint + "."
    return text + f" Source tail: {arg[-70:]}"


def _run_playwright(url: str, *, actions: str, wait: float, out,
                    timeout: int, headless: bool) -> tuple[bool, dict | str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False, install_hint()

    console: list[str] = []
    action_log: list[str] = []
    action_errors: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless)
            try:
                page = browser.new_page(
                    viewport={"width": 1280, "height": 800})
                page.set_default_timeout(max(1, timeout) * 1000)

                def on_console(msg):
                    if msg.type in ("error", "warning"):
                        console.append(f"{msg.type}: {msg.text}"[:240])

                page.on("console", on_console)
                page.on("pageerror",
                        lambda exc: console.append(f"pageerror: {exc}"[:240]))
                page.goto(url, wait_until="domcontentloaded",
                          timeout=max(1, timeout) * 1000)

                for kind, arg in parse_actions(actions):
                    try:
                        if kind == "press":
                            page.keyboard.press(arg)
                            action_log.append(f"pressed {arg}")
                        elif kind == "click":
                            page.click(arg)
                            action_log.append(f"clicked {arg}")
                        elif kind == "fill":
                            selector, text = arg
                            page.fill(selector, text)
                            action_log.append(f"filled {selector}")
                        elif kind == "type":
                            page.keyboard.type(arg)
                            action_log.append(f"typed {arg[:40]}")
                        elif kind == "wait":
                            page.wait_for_timeout(int(float(arg) * 1000))
                            action_log.append(f"waited {arg:g}s")
                        elif kind == "eval":
                            try:
                                value = page.evaluate(
                                    _raced_eval(arg, timeout))
                            except Exception as ee:
                                msg = str(ee)
                                low2 = msg.lower()
                                if ("content security policy" in low2
                                        or "refused to evaluate" in low2):
                                    # A strict CSP forbids the in-page
                                    # eval() the race needs; fall back to the
                                    # CSP-immune, unbounded CDP evaluate.
                                    value = page.evaluate(arg)
                                elif "syntaxerror" in low2:
                                    action_errors.append(
                                        _eval_syntax_error(arg, msg))
                                    continue
                                else:
                                    raise
                            action_log.append(
                                f"eval'd {arg[:60]} -> {str(value)[:2000]}")
                    except Exception as e:
                        action_errors.append(
                            f"{kind} {str(arg)[:60]}: {str(e)[:160]}")

                page.wait_for_timeout(int(max(0.0, float(wait or 0)) * 1000))
                if out is not None:
                    page.screenshot(path=str(out), full_page=False)
                info = {
                    "method": "Playwright/Chromium",
                    "url": page.url,
                    "title": page.title(),
                    "text": _page_text(page),
                    "canvas": _canvas_info(page),
                    "console": console[-10:],
                    "actions": action_log,
                    "action_errors": action_errors,
                }
                return True, info
            finally:
                browser.close()
    except Exception as e:
        msg = str(e)
        hint = ""
        low = msg.lower()
        if "executable doesn't exist" in low or "browsertype.launch" in low:
            hint = ("\nPlaywright is installed but Chromium is not. Run:\n"
                    "  python -m playwright install chromium")
        return False, f"browser launch failed: {msg[:400]}{hint}"


def _run_chrome(exe: str, url: str, *, wait: float, out,
                timeout: int) -> tuple[bool, dict | str]:
    if out is None:
        return False, "the Chrome fallback needs an output screenshot path"
    budget = max(1000, int(float(wait or 0) * 1000) + 3000)
    detail = ""
    # Chrome 109+ prefers --headless=new; older builds only know --headless.
    for mode in ("--headless=new", "--headless"):
        cmd = [
            exe, mode, "--disable-gpu", "--hide-scrollbars",
            "--window-size=1280,800", f"--virtual-time-budget={budget}",
            f"--screenshot={out}", url,
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=max(5, int(timeout) + 15))
        except Exception as e:
            return False, f"could not run {Path(exe).name}: {e}"
        if out.is_file():
            return True, {
                "method": f"{Path(exe).name} {mode} (navigation only)",
                "url": url,
                "title": "",
                "text": "",
                "canvas": "",
                "console": [],
                "actions": [],
                "action_errors": [],
            }
        detail = ((p.stdout or "") + (p.stderr or "")).strip()
    return False, ("Chrome did not produce a screenshot: "
                   + (detail[:400] or "no output"))
