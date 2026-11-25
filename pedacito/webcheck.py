"""Static contract checks for multi-file web projects (no browser needed).

Small models building browser games and pages reliably make the same
cross-file contract mistakes, and every one of them stays silent until
somebody opens the page:

  - JS reaches for a DOM id that exists in no HTML file
    (``getElementById`` / ``querySelector("#...")`` returns null);
  - HTML loads a module through a classic ``<script src>`` (no
    ``type="module"``) while the file uses ``import``/``export`` -- the
    browser aborts it with a SyntaxError and nothing runs;
  - two classic scripts on one page define the same top-level global -- a
    let/const/class redeclaration is a SyntaxError, a function/var
    redeclaration silently replaces the earlier definition;
  - a ``<script src>`` points at a file that does not exist (404);
  - a JS file imports a bare specifier that package.json never declared --
    ``node --test`` dies with ERR_MODULE_NOT_FOUND before any assertion ran;
  - an inline handler calls ``Game.pasue()`` but ``pasue`` appears nowhere
    in the one classic script that defines ``Game`` (TypeError);
  - a module binds the same name twice -- the ``import { WAYPOINTS }`` an
    incremental edit appends next to the import that already binds it
    (``SyntaxError: Identifier 'WAYPOINTS' has already been declared``);
  - a module calls a name it neither declares nor imports, while a sibling
    module exports it or nothing in the project defines it at all
    (``ReferenceError``, and the scene never renders);
  - a ``const`` binding is reassigned later -- parses clean, then throws
    ``TypeError: Assignment to constant variable`` the first time the line
    runs;
  - a handler calls ``engine.setBpm(...)`` on a ``new SynthEngine()``
    instance but the class defines no ``setBpm`` (``TypeError: ... is not a
    function`` on the first slider move);
  - a spam honeypot input (``name="_gotcha"``, ``class="honeypot"``,
    ``aria-hidden="true"``) is not actually hidden -- real visitors see and
    fill it, so the trap catches humans instead of bots;
  - a visible text/email input or textarea inside a ``<form>`` has no
    ``<label for>``, wrapping ``<label>`` or ``aria-label`` -- screen
    readers announce an unnamed field;
  - a stylesheet the page links carries raw ``@tailwind base`` /
    ``components`` / ``utilities`` build directives -- the browser cannot
    parse them, ignores the sheet and the page renders unstyled;
  - an HTML page wears Tailwind utility classes (``bg-[#141518]``,
    ``min-h-screen max-w-5xl mx-auto`` ...) but loads neither the Tailwind
    CDN script nor a stylesheet that compiles those utilities -- every
    class is inert and the page renders completely unstyled.

Every check is regex-level on purpose: no JS parser, no execution, no
network, no dependencies. That trades recall for speed, so the results are
wired in as *notices* -- ``pedacito webcheck``, the scaffold summary, a
WARN after auto tasks that touched web files -- never as hard failures.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Finding", "notice", "run"]

_SKIP_DIRS = {"node_modules", ".git", "dist", "build", "out", ".venv", "venv",
              "__pycache__", ".pedacito", "coverage", ".next", "vendor",
              ".cache", ".tox"}
_HTML_EXT = (".html", ".htm")
_JS_EXT = (".js", ".mjs", ".cjs")
_WEB_EXT = _HTML_EXT + _JS_EXT
_MAX_FILES = 400
_MAX_BYTES = 2 * 1024 * 1024

# Node builtins: a bare import of one of these always resolves.
_NODE_BUILTINS = {
    "assert", "async_hooks", "buffer", "child_process", "cluster", "console",
    "constants", "crypto", "dgram", "diagnostics_channel", "dns", "domain",
    "events", "fs", "http", "http2", "https", "inspector", "module", "net",
    "os", "path", "perf_hooks", "process", "punycode", "querystring",
    "readline", "repl", "stream", "string_decoder", "timers", "tls",
    "trace_events", "tty", "url", "util", "v8", "vm", "worker_threads",
    "zlib",
}

# Members every JS object/function carries: calling them is fine even when
# the defining file never mentions the word.
_BUILTIN_MEMBERS = {"name", "length", "call", "apply", "bind", "prototype",
                    "constructor", "toString", "valueOf", "hasOwnProperty"}

_SCRIPT_TAG_RE = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>",
                            re.I | re.S)
_SRC_ATTR_RE = re.compile(r"""\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_TYPE_ATTR_RE = re.compile(r"""\btype\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_HANDLER_ATTR_RE = re.compile(r"""\bon[a-z]+\s*=\s*(?:"([^"]*)"|'([^']*)')""",
                              re.I)
_ID_ATTR_RE = re.compile(r"""\bid\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)

# A <select> and its body, and an <option> tag (BUG-324: an empty dropdown is a
# dead control). _OPTION_POPULATION_RE is evidence that JS builds options at
# runtime -- when it fires anywhere, an empty <select> may be filled dynamically
# and the finding is suppressed (webcheck stays conservative).
_SELECT_TAG_RE = re.compile(r"<select\b([^>]*)>(.*?)</select\s*>", re.I | re.S)
_OPTION_TAG_RE = re.compile(r"<option\b", re.I)
_OPTION_POPULATION_RE = re.compile(
    r"new\s+Option\s*\(|createElement\s*\(\s*[\"']option|<option\b"
    r"|\.options\s*\.\s*add\s*\(", re.I)

# Form controls: honeypot traps that are not hidden, and visible fields with
# no label. Regex-level and conservative, like every webcheck notice.
_INPUT_TAG_RE = re.compile(r"<input\b([^>]*?)/?>", re.I | re.S)
_TEXTAREA_TAG_RE = re.compile(r"<textarea\b([^>]*)>", re.I | re.S)
_FORM_SPAN_RE = re.compile(r"<form\b[^>]*>.*?</form\s*>", re.I | re.S)
_LABEL_SPAN_RE = re.compile(r"<label\b[^>]*>.*?</label\s*>", re.I | re.S)
_LABEL_FOR_RE = re.compile(
    r"""<label\b[^>]*\bfor\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_CLASS_ATTR_RE = re.compile(r"""\bclass\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_NAME_ATTR_RE = re.compile(r"""\bname\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_STYLE_ATTR_RE = re.compile(r"""\bstyle\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_ARIA_HIDDEN_RE = re.compile(
    r"""\baria-hidden\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_ARIA_LABEL_RE = re.compile(
    r"""\baria-label(?:ledby)?\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
# The boolean `hidden` attribute -- never the `hidden` inside `aria-hidden`
# (the lookarounds exclude a preceding word char or hyphen).
_HIDDEN_ATTR_RE = re.compile(r"(?<![\w-])hidden(?![\w-])", re.I)
_ATTR_VALUE_RE = re.compile(r"""=\s*(?:"[^"]*"|'[^']*')""")
_INLINE_HIDDEN_RE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)
_STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>(.*?)</style\s*>", re.I | re.S)
_LINK_TAG_RE = re.compile(r"<link\b([^>]*)>", re.I)
_REL_ATTR_RE = re.compile(r"""\brel\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_HREF_ATTR_RE = re.compile(r"""\bhref\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
# One CSS rule (selector list + flat body); nested @media bodies carry braces
# and are skipped, their inner rules match on their own.
_CSS_RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}", re.S)
_CSS_CLASS_RE = re.compile(r"\.([A-Za-z_][\w-]*)")
# Uncompiled Tailwind build directives at the start of a line. A browser
# cannot parse them: the whole linked sheet is ignored and the page renders
# unstyled. (The Play CDN compiles utility classes at runtime; it never
# processes a linked .css file, so its presence does not excuse them.)
_TAILWIND_DIRECTIVE_RE = re.compile(
    r"^[ \t]*@tailwind\s+(base|components|utilities)\b[ \t]*;?", re.M)
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)

# --- check 13: Tailwind utility classes with no compiler in sight ------------
# A model can dress a page in Tailwind utilities while linking a 20-line
# hand-written styles.css and omitting the Play CDN: every utility class is
# then inert and the page renders unstyled. Two unmistakable signals:
#   * an arbitrary-value class -- `bg-[#141518]`, `text-[#e9e7e2]`,
#     `w-[32rem]`, `grid-cols-[1fr_2fr]` -- or an opacity modifier such as
#     `border-white/10` / `bg-black/50`. No hand-written semantic CSS uses
#     these shapes, so ONE is enough to prove Tailwind intent.
#   * a cluster of >= 4 distinct layout/spacing utilities (`min-h-screen`,
#     `max-w-5xl`, `mx-auto`, `flex-col`, `md:flex-row`, `tracking-tight`...).
#     A variant prefix (`md:`, `hover:`) is stripped before the lookup.
_TW_ARBITRARY_RE = re.compile(r"^[a-z][a-z0-9-]*-\[[^\]\s]+\]$")
_TW_OPACITY_RE = re.compile(r"^[a-z][a-z0-9-]*-[a-z0-9]+/\d{1,3}$")
_TW_VARIANT_RE = re.compile(r"^(?:[a-z0-9-]+:)+")
# Utilities whose names are Tailwind-specific enough that a cluster of four
# distinct ones is not a coincidence. Deliberately not exhaustive: recall is
# traded for the same conservatism every webcheck notice keeps.
_TW_LAYOUT_UTILS = frozenset("""
    min-h-screen min-h-dvh h-screen max-w-5xl max-w-4xl max-w-6xl max-w-7xl
    max-w-screen-xl mx-auto my-auto flex-col flex-row items-center items-start
    items-end justify-center justify-between justify-around justify-evenly
    tracking-tight tracking-wide gap-1 gap-2 gap-3 gap-4 gap-6 gap-8
    px-2 px-4 px-6 px-8 py-2 py-4 py-6 py-8 p-4 p-6 p-8
    space-x-2 space-x-4 space-y-2 space-y-4 rounded-lg rounded-xl rounded-2xl
    rounded-full text-xs text-sm text-base text-lg text-xl text-2xl text-3xl
    font-medium font-semibold font-bold leading-relaxed leading-tight
    grid-cols-1 grid-cols-2 grid-cols-3 grid-cols-4 col-span-2 col-span-3
    flex-1 flex-grow shrink-0 grow w-full h-full min-w-0 overflow-hidden
    overflow-auto relative absolute fixed sticky inset-0 top-0 left-0 right-0
    bottom-0 z-10 z-20 z-50 opacity-0 opacity-50 opacity-100 transition
    transition-all transition-colors duration-200 duration-300 ease-in-out
    shadow-sm shadow-md shadow-lg shadow-xl ring-1 ring-2 backdrop-blur
    bg-gradient-to-r bg-gradient-to-br from-gray-900 to-gray-800
    text-gray-100 text-gray-300 text-gray-400 text-gray-500 text-white
    bg-white bg-gray-900 bg-gray-800 bg-slate-900 border border-gray-700
    border-gray-800 border-white divide-y divide-gray-800 antialiased
    select-none pointer-events-none cursor-pointer whitespace-nowrap
    truncate line-clamp-2 line-clamp-3 sr-only not-sr-only
""".split())
_TW_MIN_CLUSTER = 4
# The Play CDN (v3 `cdn.tailwindcss.com`, v4 `@tailwindcss/browser`) compiles
# utility classes at runtime; its presence anywhere in the document excuses
# the utilities. It never processes a linked .css file (that is check 12).
_TW_CDN_RE = re.compile(
    r"<script\b[^>]*\bsrc\s*=\s*[\"'][^\"']*"
    r"(?:cdn\.tailwindcss\.com|@tailwindcss/browser)[^\"']*[\"']", re.I)
# Input types the honeypot check never flags: they are not fillable text
# fields, so a trap marker on one is not a visible honeypot.
_HONEYPOT_EXEMPT_TYPES = frozenset({"hidden", "submit", "button", "reset"})

# ``import``/``export`` statements (never the dynamic ``import(``, which is
# legal in classic scripts). Only horizontal space before the keyword, so
# the match starts on the keyword's own line (line numbers stay honest).
_ESM_RE = re.compile(
    r"^[ \t]*(?:export\s+(?:default\b|const\b|let\b|var\b|function\b|class\b"
    r"|async\b|\*|\{)|import\b(?!\s*\())", re.M)

# Top-level declarations at column 0: the classic-script global scope.
_TOPLEVEL_RE = re.compile(
    r"^(?:function\s*\*?\s*([A-Za-z_$][\w$]*)"
    r"|class\s+([A-Za-z_$][\w$]*)"
    r"|(const|let)\s+([A-Za-z_$][\w$]*)"
    r"|var\s+([A-Za-z_$][\w$]*))", re.M)

_GET_BY_ID_RE = re.compile(
    r"""getElementById\(\s*(?:"([^"]+)"|'([^']+)')\s*[,)]""")
_QS_ID_RE = re.compile(r"""querySelector(?:All)?\(\s*["']#([A-Za-z0-9_-]+)""")
# Ids JS creates at runtime count as existing (dynamic UI built in code).
_JS_ID_RE = re.compile(r"""\.id\s*=\s*(?:"([^"]+)"|'([^']+)')""")
_SET_ID_RE = re.compile(
    r"""setAttribute\(\s*(?:"id"|'id')\s*,\s*(?:"([^"]+)"|'([^']+)')""")

_FROM_RE = re.compile(r"""\bfrom\s*(?:"([^"]+)"|'([^']+)')""")
_BARE_IMPORT_RE = re.compile(r"""^[ \t]*import\s*(?:"([^"]+)"|'([^']+)')""",
                             re.M)
_REQUIRE_RE = re.compile(r"""\brequire\(\s*(?:"([^"]+)"|'([^']+)')\s*\)""")

# Globals a classic script exposes, and calls that reach for their members.
_OBJ_DEF_RE = re.compile(
    r"^(?:const|let|var)\s+([A-Z][\w$]*)\s*=\s*(?:\{|Object\.assign|class\b)",
    re.M)
_FN_DEF_RE = re.compile(r"^function\s+([A-Z][\w$]*)\s*\(", re.M)
_CLASS_DEF_RE = re.compile(r"^class\s+([A-Z][\w$]*)\b", re.M)
_WIN_DEF_RE = re.compile(r"\b(?:window|globalThis)\.([A-Z][\w$]*)\s*=")
_MEMBER_CALL_RE = re.compile(r"\b([A-Z][\w$]*)\.([a-z_$][\w$]*)\s*\(")


@dataclass(frozen=True)
class Finding:
    """One contract violation. `peer` is the other file in the contract
    (the page that loads the script, the script that defines the global),
    which keeps the `files=` filter honest: editing either side of a broken
    contract is editing the bug."""
    rel: str
    line: int
    message: str
    peer: str | None = None

    def __str__(self) -> str:
        return f"{self.rel}:{self.line}: {self.message}"


@dataclass
class _Scope:
    """One executable script: a standalone JS file or an inline <script>."""
    rel: str                  # file findings report against
    text: str                 # the JavaScript itself
    offset: int               # lines to add for positions inside `text`
    module: bool
    inline: bool = False
    peer: str | None = None   # page that loads (or contains) this scope


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _first_group(m) -> str:
    return next((g for g in m.groups() if g), "")


def _collect(root: Path):
    """(html files, js files, css files) under root, each a (rel, text) pair,
    all sorted by rel. Skips build/vendored dirs and caps count and size so a
    post-edit pass stays at regex speed. CSS rides along as hiding evidence
    for the form-control checks (a `.honeypot{display:none}` rule)."""
    htmls: list[tuple[str, str]] = []
    jss: list[tuple[str, str]] = []
    csss: list[tuple[str, str]] = []
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in _SKIP_DIRS and not d.startswith("."))
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _WEB_EXT and ext != ".css":
                continue
            p = Path(dirpath) / fn
            try:
                if p.stat().st_size > _MAX_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = p.relative_to(root).as_posix()
            bucket = (htmls if ext in _HTML_EXT
                      else csss if ext == ".css" else jss)
            bucket.append((rel, text))
            n += 1
            if n >= _MAX_FILES:
                htmls.sort(key=lambda t: t[0])
                jss.sort(key=lambda t: t[0])
                csss.sort(key=lambda t: t[0])
                return htmls, jss, csss
    htmls.sort(key=lambda t: t[0])
    jss.sort(key=lambda t: t[0])
    csss.sort(key=lambda t: t[0])
    return htmls, jss, csss


def _resolve_src(root: Path, html_rel: str, src: str):
    """(project-relative target, False) for a local script src; (None, True)
    when the src is a URL or escapes the project (nothing we can check)."""
    if not src or src.startswith(("data:", "blob:")):
        return None, True
    if re.match(r"^(?:[A-Za-z][\w.+-]*:)?//", src):
        return None, True
    base = root if src.startswith("/") else (root / html_rel).parent
    try:
        rel = (base / src.lstrip("/")).resolve() \
            .relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None, True
    return rel, False


def _pkg_deps(root: Path) -> set[str] | None:
    """The declared package names, or None when there is no (parseable)
    package.json -- no manifest, no contract to check imports against."""
    p = root / "package.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    deps: set[str] = set()
    for key in ("dependencies", "devDependencies", "peerDependencies",
                "optionalDependencies"):
        block = data.get(key)
        if isinstance(block, dict):
            deps.update(str(k) for k in block)
    return deps


def _bare_spec(spec: str) -> str | None:
    """The package name a specifier must resolve to, or None when it is
    relative, absolute, a URL or a Node builtin (always fine)."""
    if not spec or spec.startswith((".", "/", "node:")):
        return None
    if re.match(r"^[A-Za-z][\w.+-]*:", spec):        # http:, data:, ...
        return None
    parts = spec.split("/")
    if spec.startswith("@"):
        return "/".join(parts[:2]) if len(parts) >= 2 else None
    return parts[0] or None


# A DOM lookup whose null result the surrounding code handles. Regex-level
# like every check here, and used ONLY to word a finding: an id no HTML file
# renders is a broken contract either way, but telling the model "the next
# property access throws" at a call site that reads `... || fallback` or
# `if (!el) return` is a claim it can disprove by looking -- and a notice
# caught lying once gets ignored everywhere else.
_GUARD_TAIL_RE = re.compile(
    r"\s*(?:\|\||&&|\?\.|\?|!?={2,3}\s*(?:null|undefined))")
# `const el = ` / `el = ` immediately before the lookup (receiver stripped).
_ASSIGN_HEAD_RE = re.compile(
    r"(?:\b(?:const|let|var)\s+)?([A-Za-z_$][\w$]*)\s*=\s*$")
# The lookup is itself the condition: `if (document.getElementById('x'))`.
_TEST_HEAD_RE = re.compile(
    r"(?:\b(?:if|while)\s*\(\s*!?|\bBoolean\s*\(\s*|!)\s*$")
# `document.` / `window.document.` between the assignment and the call.
_RECEIVER_TAIL_RE = re.compile(r"(?:[A-Za-z_$][\w$]*\s*\.\s*)+$")


def _call_end(text: str, m) -> int:
    """Position just past the lookup's closing paren. `_GET_BY_ID_RE` already
    consumes it; `_QS_ID_RE` stops at the id, so the guard scan has to look
    past the `')` that follows."""
    if text[m.end() - 1:m.end()] == ")":
        return m.end()
    close = text.find(")", m.end())
    return close + 1 if 0 <= close <= m.end() + 60 else m.end()


def _null_guarded(text: str, m) -> bool:
    """True when the code around a DOM lookup handles the null result: a
    fallback / optional chain / ternary right after the call, the call used as
    a condition, or an `if (!el)` style check on the variable it was assigned
    to within the next couple of statements."""
    head = _RECEIVER_TAIL_RE.sub(
        "", text[max(0, m.start() - 80):m.start()])
    if _TEST_HEAD_RE.search(head):
        return True
    if _GUARD_TAIL_RE.match(text, _call_end(text, m)):
        return True
    am = _ASSIGN_HEAD_RE.search(head)
    if not am:
        return False
    var = re.escape(am.group(1))
    tail = text[_call_end(text, m):][:200]
    return bool(re.search(
        rf"\b(?:if|while)\s*\(\s*!?\s*{var}\b"
        rf"|\b{var}\s*(?:\?|&&|\|\|)"
        rf"|\b{var}\s*!?={2,3}\s*(?:null|undefined)", tail))


def _null_consequence(text, m, method: str) -> str:
    """What a missing id does AT THIS call site: an unguarded lookup throws on
    the next property access; a guarded one silently takes the fallback, so
    the code it feeds never runs."""
    if _null_guarded(text, m):
        return (f"{method} returns null here; this call site handles the null "
                f"(a fallback or a guard), so nothing throws -- but whatever "
                f"the lookup feeds can only run if the id is created "
                f"dynamically at runtime")
    return f"{method} returns null and the next property access throws"


def _id_findings(rel, text, offset, ids, peer):
    """Check 1: DOM id references that no HTML file in the project has."""
    out = []
    for rx, method in ((_GET_BY_ID_RE, "getElementById"),
                       (_QS_ID_RE, "querySelector")):
        for m in rx.finditer(text):
            name = _first_group(m)
            if not name or name in ids:
                continue
            if method == "querySelector" and name.endswith(("-", "_")):
                continue          # "#row-" + i: a dynamic id prefix, not a ref
            out.append(Finding(
                rel, offset + _line_of(text, m.start()),
                f"no element with id '{name}' exists in any HTML file -- "
                + _null_consequence(text, m, method),
                peer=peer))
    return out


def _empty_select_findings(htmls, jss) -> list[Finding]:
    """BUG-324: a <select> with no <option> children is a dead control -- the
    user can open the dropdown but choose nothing, so the feature it feeds is
    unusable even though every id and module binding checks out. Flag empty
    selects unless some script builds options at runtime (then the markup may
    be filled dynamically and we cannot prove it is dead). Regex-level and
    conservative, like every webcheck notice."""
    # Population evidence lives in JS only -- searching the HTML markup would
    # match the selects' own <option> tags and suppress everything. Gather the
    # standalone JS plus every inline <script> body.
    js_texts = [text for _rel, text in jss]
    for _rel, text in htmls:
        for sm in _SCRIPT_TAG_RE.finditer(text):
            js_texts.append(sm.group(2))
    if any(_OPTION_POPULATION_RE.search(t) for t in js_texts):
        return []
    out: list[Finding] = []
    for rel, text in htmls:
        for m in _SELECT_TAG_RE.finditer(text):
            attrs, body = m.group(1), m.group(2)
            if _OPTION_TAG_RE.search(body):
                continue
            idm = _ID_ATTR_RE.search(attrs)
            name = _first_group(idm) if idm else ""
            ident = f"#{name}" if name else "<select>"
            out.append(Finding(
                rel, _line_of(text, m.start()),
                f"{ident} has no <option> children and no script builds any "
                "-- the dropdown is empty, so the feature it feeds cannot be "
                "used"))
    return out


def _attr_value(rx, attrs: str) -> str:
    """One attribute's value from a tag's attribute text ('' when absent)."""
    m = rx.search(attrs)
    return _first_group(m).strip() if m else ""


def _input_hidden(attrs: str, classes: set[str], css_hidden: set[str]) -> bool:
    """Evidence a control is not user-visible: the boolean `hidden`
    attribute, inline display:none / visibility:hidden, or a class some
    collected stylesheet hides."""
    # Strip quoted values first: the `hidden` inside `aria-hidden="true"`
    # or `type="hidden"` must not read as the boolean hidden attribute.
    bare = _ATTR_VALUE_RE.sub("", attrs)
    if _HIDDEN_ATTR_RE.search(bare):
        return True
    style = _attr_value(_STYLE_ATTR_RE, attrs)
    if style and _INLINE_HIDDEN_RE.search(style):
        return True
    return bool(classes & css_hidden)


def _css_hidden_classes(css_texts) -> set[str]:
    """Class names some stylesheet rule hides (display:none or
    visibility:hidden): `.honeypot{display:none}` is the canonical way to
    arm a honeypot without an inline style."""
    out: set[str] = set()
    for text in css_texts:
        for m in _CSS_RULE_RE.finditer(text):
            if _INLINE_HIDDEN_RE.search(m.group(2)):
                out.update(_CSS_CLASS_RE.findall(m.group(1)))
    return out


def _form_input_findings(htmls, csss, root: Path) -> list[Finding]:
    """Check 11: a spam honeypot input nothing hides (real visitors see and
    fill it, so the trap catches humans instead of bots), and a visible
    text/email input or textarea inside a <form> with no <label for>,
    wrapping <label> or aria-label (screen readers announce an unnamed
    field). Hiding evidence: the `hidden` attribute, inline display:none /
    visibility:hidden, or a class hidden by any collected, inline or linked
    stylesheet. Regex-level and conservative, like every webcheck notice."""
    css_texts = [text for _rel, text in csss]
    for rel, text in htmls:
        css_texts.extend(m.group(1) for m in _STYLE_BLOCK_RE.finditer(text))
        for lm in _LINK_TAG_RE.finditer(text):
            attrs = lm.group(1)
            rm = _REL_ATTR_RE.search(attrs)
            if not rm or "stylesheet" not in _first_group(rm).lower():
                continue
            hm = _HREF_ATTR_RE.search(attrs)
            if not hm:
                continue
            href = _first_group(hm).strip().split("?")[0].split("#")[0]
            css_rel, external = _resolve_src(root, rel, href)
            if external:
                continue
            p = root / css_rel
            try:
                if p.is_file() and p.stat().st_size <= _MAX_BYTES:
                    css_texts.append(
                        p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    css_hidden = _css_hidden_classes(css_texts)
    out: list[Finding] = []
    for rel, text in htmls:
        label_fors = {_first_group(m) for m in _LABEL_FOR_RE.finditer(text)}
        label_fors.discard("")
        label_spans = [(m.start(), m.end())
                       for m in _LABEL_SPAN_RE.finditer(text)]
        form_spans = [(m.start(), m.end())
                      for m in _FORM_SPAN_RE.finditer(text)]
        controls = [(m, "input") for m in _INPUT_TAG_RE.finditer(text)]
        controls += [(m, "textarea")
                     for m in _TEXTAREA_TAG_RE.finditer(text)]
        controls.sort(key=lambda t: t[0].start())
        for m, tag in controls:
            attrs = m.group(1)
            typ = (_attr_value(_TYPE_ATTR_RE, attrs).lower()
                   if tag == "input" else "textarea")
            classes = set(_attr_value(_CLASS_ATTR_RE, attrs).split())
            name = _attr_value(_NAME_ATTR_RE, attrs)
            idm = _ID_ATTR_RE.search(attrs)
            fid = _first_group(idm) if idm else ""
            ident = (f"#{fid}" if fid else
                     f'name="{name}"' if name else f"<{tag}>")
            hidden = _input_hidden(attrs, classes, css_hidden)
            line = _line_of(text, m.start())
            # --- a honeypot trap that nothing hides ------------------------
            if tag == "input" and typ not in _HONEYPOT_EXEMPT_TYPES:
                aria = _attr_value(_ARIA_HIDDEN_RE, attrs).lower()
                signals = []
                if aria == "true":
                    signals.append('aria-hidden="true"')
                if name == "_gotcha":
                    signals.append('name="_gotcha"')
                if "honeypot" in classes:
                    signals.append('class="honeypot"')
                if signals and not hidden:
                    out.append(Finding(
                        rel, line,
                        f"{ident} looks like a spam honeypot "
                        f"({', '.join(signals)}) but nothing hides it -- "
                        "real visitors will see and fill it, so the trap "
                        "catches humans instead of bots; add the `hidden` "
                        "attribute, inline display:none, or a CSS rule "
                        "that hides its class"))
            # --- a visible form field with no label ------------------------
            if hidden:
                continue
            if tag == "input" and typ not in ("", "text", "email"):
                continue
            if not any(a <= m.start() < b for a, b in form_spans):
                continue
            if any(a <= m.start() < b for a, b in label_spans):
                continue                     # a wrapping <label>
            if fid and fid in label_fors:
                continue                     # <label for="..."> elsewhere
            if _attr_value(_ARIA_LABEL_RE, attrs):
                continue
            out.append(Finding(
                rel, line,
                f"{ident} inside <form> has no label -- no <label for>, "
                "wrapping <label>, or aria-label; screen readers announce "
                "it as an unnamed field"))
    return out


def _tailwind_directive_findings(htmls, csss, root: Path) -> list[Finding]:
    """Check 12: a stylesheet linked via <link rel="stylesheet"> carries raw
    `@tailwind base/components/utilities` build directives. Those are input
    for the Tailwind build step, not CSS: the browser cannot parse them,
    ignores the sheet and the page renders unstyled. Only LINKED sheets are
    flagged -- an unlinked source file may legitimately feed a PostCSS
    build, and an inline `<style type="text/tailwindcss">` is the Play CDN's
    own sanctioned input. One finding per directive line, peer = the page
    that links the sheet, so editing either side surfaces it."""
    css_texts = dict(csss)
    out: list[Finding] = []
    seen: set[tuple[str, int]] = set()
    for rel, text in htmls:
        for lm in _LINK_TAG_RE.finditer(text):
            attrs = lm.group(1)
            rm = _REL_ATTR_RE.search(attrs)
            if not rm or "stylesheet" not in _first_group(rm).lower():
                continue
            hm = _HREF_ATTR_RE.search(attrs)
            if not hm:
                continue
            href = _first_group(hm).strip().split("?")[0].split("#")[0]
            css_rel, external = _resolve_src(root, rel, href)
            if external:
                continue
            css = css_texts.get(css_rel)
            if css is None:
                continue
            comments = [(m.start(), m.end())
                        for m in _CSS_COMMENT_RE.finditer(css)]
            for m in _TAILWIND_DIRECTIVE_RE.finditer(css):
                if any(a <= m.start() < b for a, b in comments):
                    continue        # a commented-out directive is inert
                line = _line_of(css, m.start())
                key = (css_rel, line)
                if key in seen:
                    continue        # two pages linking one sheet: one bug
                seen.add(key)
                out.append(Finding(
                    css_rel, line,
                    f"raw `@tailwind {m.group(1)}` build directive in the "
                    f"stylesheet {rel} links -- browsers cannot parse it, "
                    "so the sheet is ignored and the page renders unstyled; "
                    "write standard browser-valid CSS in the linked "
                    "stylesheet, or use the Tailwind CDN script "
                    "(https://cdn.tailwindcss.com) with utility classes "
                    "instead of raw @tailwind build directives",
                    peer=rel))
    return out


def _tw_class_tokens(text: str) -> list[tuple[int, str]]:
    """(position, class-token) for every whitespace-separated token of every
    `class="..."` attribute in an HTML document. Position is the attribute's
    start, so a finding points at the line that wears the utilities."""
    out: list[tuple[int, str]] = []
    for m in _CLASS_ATTR_RE.finditer(text):
        for tok in _first_group(m).split():
            out.append((m.start(), tok))
    return out


def _tw_utility_tokens(text: str) -> tuple[list[str], int]:
    """(distinct Tailwind utility tokens, first position) worn by an HTML
    document. A token counts when it is an arbitrary-value / opacity-modifier
    class, or a known layout utility (variant prefix stripped)."""
    seen: dict[str, int] = {}
    for pos, tok in _tw_class_tokens(text):
        base = _TW_VARIANT_RE.sub("", tok)
        if (_TW_ARBITRARY_RE.match(base) or _TW_OPACITY_RE.match(base)
                or base in _TW_LAYOUT_UTILS):
            seen.setdefault(tok, pos)
    if not seen:
        return [], 0
    return list(seen), min(seen.values())


def _tw_is_tailwind(tokens: list[str]) -> bool:
    """True when the token set is unmistakably Tailwind: one arbitrary-value
    or opacity-modifier class, or a cluster of >= 4 distinct layout utilities
    (counted by their variant-stripped base name)."""
    bases = set()
    for tok in tokens:
        base = _TW_VARIANT_RE.sub("", tok)
        if _TW_ARBITRARY_RE.match(base) or _TW_OPACITY_RE.match(base):
            return True
        if base in _TW_LAYOUT_UTILS:
            bases.add(base)
    return len(bases) >= _TW_MIN_CLUSTER


def _tw_compiled_selector(token: str) -> re.Pattern:
    r"""A regex matching the CSS class selector a Tailwind build emits for
    `token`. Arbitrary values escape their specials (`.bg-\[\#141518\]`), so
    each non-word character may carry a leading backslash; the selector must
    end at a combinator, brace, comma or colon -- never mid-name."""
    parts = []
    for ch in token:
        if ch.isalnum() or ch in "-_":
            parts.append(re.escape(ch))
        else:
            parts.append(r"\\?" + re.escape(ch))
    return re.compile(r"\." + "".join(parts) + r"(?![\w-])")


def _tw_compiled_in_css(tokens: list[str], css_texts: list[str]) -> bool:
    """True when the linked/inline stylesheets actually define the utility
    classes the page wears -- a real compiled Tailwind sheet (or a hand-built
    equivalent), not a 20-line semantic stylesheet that leaves them inert.
    Every detected utility must have a rule; one missing class is enough to
    keep the notice (the page would still render partly unstyled)."""
    if not css_texts or not tokens:
        return False
    blob = "\n".join(css_texts)
    return all(_tw_compiled_selector(t).search(blob) for t in tokens)


def _linked_css_texts(root: Path, rel: str, text: str,
                      csss: dict[str, str]) -> tuple[list[str], str | None]:
    """(stylesheet texts this page links or inlines, first linked css rel).
    Local linked sheets resolve through the same path logic as check 12;
    inline <style> blocks ride along so a compiled-in-<style> page passes."""
    texts: list[str] = []
    first_rel: str | None = None
    texts.extend(m.group(1) for m in _STYLE_BLOCK_RE.finditer(text))
    for lm in _LINK_TAG_RE.finditer(text):
        attrs = lm.group(1)
        rm = _REL_ATTR_RE.search(attrs)
        if not rm or "stylesheet" not in _first_group(rm).lower():
            continue
        hm = _HREF_ATTR_RE.search(attrs)
        if not hm:
            continue
        href = _first_group(hm).strip().split("?")[0].split("#")[0]
        css_rel, external = _resolve_src(root, rel, href)
        if external:
            continue
        if first_rel is None:
            first_rel = css_rel
        css = csss.get(css_rel)
        if css is None:
            p = root / css_rel
            try:
                if p.is_file() and p.stat().st_size <= _MAX_BYTES:
                    css = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                css = None
        if css is not None:
            texts.append(css)
    return texts, first_rel


def _tailwind_utility_findings(htmls, csss, root: Path) -> list[Finding]:
    """Check 13: an HTML page wears Tailwind utility classes but loads neither
    the Tailwind CDN script nor a stylesheet that compiles those utilities.
    The browser then ignores every utility class and the page renders
    completely unstyled -- the silent twin of check 12's raw @tailwind
    directives. One finding per page, peer = the linked stylesheet (editing
    either side of the broken contract surfaces it)."""
    css_texts = dict(csss)
    out: list[Finding] = []
    for rel, text in htmls:
        tokens, pos = _tw_utility_tokens(text)
        if not tokens or not _tw_is_tailwind(tokens):
            continue
        if _TW_CDN_RE.search(text):
            continue                 # the Play CDN compiles them at runtime
        linked, peer = _linked_css_texts(root, rel, text, css_texts)
        if _tw_compiled_in_css(tokens, linked):
            continue                 # a real compiled stylesheet defines them
        examples = ", ".join(tokens[:3])
        out.append(Finding(
            rel, _line_of(text, pos),
            f"HTML uses Tailwind utility classes (e.g. {examples}) but "
            "neither loads the Tailwind CDN script in <head> nor links a "
            "compiled stylesheet. Either write standard semantic CSS rules "
            'in styles.css or include <script src="https://cdn.tailwindcss.com">'
            "</script>.",
            peer=peer))
    return out


def _member_findings(rel, text, offset, single, texts):
    """Check 5: `Name.member(` where exactly one classic script defines
    `Name` and the word `member` appears nowhere in that file."""
    out = []
    for m in _MEMBER_CALL_RE.finditer(text):
        name, member = m.group(1), m.group(2)
        def_rel = single.get(name)
        if def_rel is None or member in _BUILTIN_MEMBERS:
            continue
        if re.search(rf"\b{re.escape(member)}\b", texts.get(def_rel, "")):
            continue          # the word exists in the definer: assume defined
        out.append(Finding(
            rel, offset + _line_of(text, m.start()),
            f"{name}.{member}() is called here but '{member}' appears "
            f"nowhere in {def_rel}, the one classic script that defines "
            f"{name} -- TypeError at runtime", peer=def_rel))
    return out


# --- ES module wiring: what a module had to declare to run at all -----------
#
# A `<script type="module">` is strict and file-scoped: every name it calls
# has to be declared there or imported from a sibling module, and the browser
# stays silent about it until the page loads -- then it shows one
# `SyntaxError: Identifier 'X' has already been declared` or a bare
# `ReferenceError: X is not defined` with no frame to point at. Both are
# visible statically: a duplicate binding is a second declaration in the same
# file, a missing one is a call site whose name exists in no scope the module
# can see. `spec._js_module_link_errors` makes the same two checks veto; the
# lists below are the vocabulary both share.

# Names every script may use without declaring them: ECMAScript plus the
# browser and Node hosts. A free identifier here is not a missing import.
JS_GLOBAL_NAMES = frozenset("""
    Array ArrayBuffer AsyncGenerator AsyncFunction Atomics BigInt
    BigInt64Array BigUint64Array Boolean Buffer DataView Date Error EvalError
    FinalizationRegistry Float16Array Float32Array Float64Array Function
    Generator Infinity Int16Array Int32Array Int8Array Intl JSON Map Math NaN Number Object Promise Proxy
    RangeError ReferenceError Reflect RegExp Set SharedArrayBuffer String
    Symbol SyntaxError TypeError Uint16Array Uint32Array Uint8Array
    Uint8ClampedArray Uint8ClampedArray BigUint64Array URIError WeakMap
    WeakRef WeakSet arguments console decodeURI decodeURIComponent encodeURI
    encodeURIComponent escape global globalThis isFinite isNaN module exports
    require parseFloat parseInt process unescape undefined
    AbortController AbortSignal Audio AudioBuffer AudioBufferSourceNode
    AudioChannelMerger AudioChannelSplitter AudioContext AudioData AudioNode
    AudioParam AudioParamMap AudioScheduledSourceNode AudioWorklet
    AudioWorkletNode BaseAudioContext BeforeUnloadEvent BiquadFilterNode Blob
    BlobEvent BroadcastChannel ByteCounter CanInteractResult CanvasGradient
    CanvasPattern CanvasRenderingContext2D Chromium CSS CSSContainerRule
    CSSCounterStyleRule CSSFontFaceRule CSSGroupingRule CSSImportRule
    CSSKeyframeRule CSSKeyframesRule CSSMediaRule CSSNamespaceRule CSSPageRule
    CSSParser CSSRule CSSStyleDeclaration CSSStyleRule CSSStyleSheet
    CSSSupportsRule Cache CacheStorage CaptionCue CharacterData Clipboard
    ClipboardEvent CloseEvent Comment CompositionEvent ConstantSourceNode
    ContainerCookieStore ContainerStorage CountQueuingStrategy Crypto CryptoKey
    CustomElementRegistry DOMException DOMParser DOMStringMap DOMTokenList
    DataTransfer DataTransferItem DataTransferItemList DelayNode
    DeleteCookiesRequest DeviceMotionEvent DeviceOrientationEvent Document
    DocumentFragment DocumentPictureInPicture DragEvent Element ElementInternals
    EnterElementExitElement Event EventSource EventTarget ExtendableEvent
    ExtendableMessageEvent External FetchEvent FieldSet File FileList
    Filesystem FilesystemDirectoryEntry FilesystemEntry FilesystemFileEntry
    FileSystem FileSystemDirectoryEntry FileSystemEntry FileSystemFileEntry
    FileSystemHandle FocusEvent FontFace FontFaceSet
    FontFaceSetLoadEvent FormData Gamepad GamepadButton GamepadEvent
    Geolocation GeolocationPosition GeolocationPositionError GetReaderBySerial
    GlobalThis HTMLAllCollection HTMLAnchorElement HTMLAreaElement
    HTMLAudioElement HTMLBaseElement HTMLBodyElement HTMLBRElement
    HTMLButtonElement HTMLCanvasElement HTMLCollection HTMLDataElement
    HTMLDataListElement HTMLDetailsElement HTMLDialogElement HTMLDivElement
    HTMLDListElement HTMLElement HTMLEmbedElement HTMLFieldSetElement
    HTMLFormElement HTMLHeadElement HTMLHeadingElement HTMLHRElement
    HTMLHtmlElement HTMLIFrameElement HTMLImageElement HTMLInputElement
    HTMLLabelElement HTMLLegendElement HTMLLiElement HTMLLinkElement
    HTMLMapElement HTMLMediaElement HTMLMenuElement HTMLMetaElement
    HTMLMeterElement HTMLModElement HTMLObjectElement HTMLOListElement
    HTMLOptGroupElement HTMLOptionElement HTMLOutputElement HTMLParagraphElement
    HTMLParamElement HTMLPictureElement HTMLPreElement HTMLProgressElement
    HTMLQuoteElement HTMLScriptElement HTMLSelectElement HTMLSlotElement
    HTMLSourceElement HTMLSpanElement HTMLStyleElement HTMLTableCaptionElement
    HTMLTableCellElement HTMLTableColElement HTMLTableElement HTMLTableRowElement
    HTMLTableSectionElement HTMLTemplateElement HTMLTextAreaElement HTMLTimeElement
    HTMLTitleElement HTMLTrackElement HTMLUListElement HTMLUnknownElement
    HTMLVideoElement IDBCursor IDBCursorWithValue IDBDatabase IDBFactory IDBIndex
    IDBKeyRange IDBObjectStore IDBOpenDBRequest IDBRequest IDBTransaction
    IDBVersionChangeEvent Image ImageBitmap ImageData InputEvent
    IntersectionObserver IntersectionObserverEntry KeyboardEvent Location
    Lock LockManager MediaDevices MediaKeyStatusMap MediaList
    MediaMetadata MediaQueryList MediaQueryListEvent MediaSource
    MediaStream MediaStreamTrack Memory MIDIConnectionEvent MIDIInput
    MIDIMessageEvent MIDIOutput MIDIPort MutationObserver MutationRecord
    NamedNodeMap Navigator NavigatorUAData NetworkInformation Node NodeFilter
    NodeIterator NodeList Notification OffscreenCanvas OffscreenCanvasRenderingContext2D
    Option OscillatorNode OverconstrainedError PageTransitionEvent PannerNode
    Path2D PaymentAddress Performance PerformanceEntry PerformanceMark
    PerformanceMeasure PerformanceNavigation PerformanceNavigationTiming
    PerformanceObserver PerformanceResourceTiming Permissions PermissionStatus
    Plugin PluginArray Point3D PointerEvent PopStateEvent PortraitOrientation
    PostMessageOptions Presentation Prefix PrerenderingChange Private
    ProgressEvent PromiseRejectionEvent PushManager PushMessageData
    RTCCertificate RTCDataChannel RTCDataChannelEvent RTCIceCandidate
    RTCIceGatherer RTCIceTransport RTCPeerConnection RTCPeerConnectionIceEvent
    RTCRtpContributingSource RTCRtpReceiver RTCRtpSender RTCSctpTransport
    RTCStatsReport RTCTrackEvent RadioNodeList Range ReadableByteStreamController
    ReadableStream ReadableStreamBYOBReader ReadableStreamBYOBRequest
    ReadableStreamDefaultController ReadableStreamDefaultReader
    RelativeIndexingMethod RemotePlayback Request ResizeEntry ResizeObserver
    Resource Timing ResourceContentResourceTiming Response SVGAElement SVGAngle
    SVGAnimatedAngle SVGAnimatedBoolean SVGAnimatedEnumeration SVGAnimatedInteger
    SVGAnimatedLength SVGAnimatedLengthList SVGAnimatedNumber SVGAnimatedNumberList
    SVGAnimatedPreserveAspectRatio SVGAnimatedRect SVGAnimatedString
    SVGAnimatedTransformList SVGAnimateElement SVGAnimateMotionElement
    SVGAnimateTransformElement SVGAnimationElement SVGCircleElement
    SVGClipPathElement SVGComponentTransferFunctionElement SVGDefsElement
    SVGDescElement SVGDiscardElement SVGElement SVGEllipseElement
    SVGFEBlendElement SVGFEColorMatrixElement SVGFEComponentTransferElement
    SVGFECompositeElement SVGFEConvolveMatrixElement SVGFEDiffuseLightingElement
    SVGFEDisplacementMapElement SVGFEDistantLightingElement SVGFEDropShadowElement
    SVGFEFloodElement SVGFEFuncAElement SVGFEFuncBElement SVGFEFuncGElement
    SVGFEFuncRElement SVGFEGaussianBlurElement SVGFEImageElement
    SVGFEMergeElement SVGFEMergeNodeElement SVGFEMorphologyElement
    SVGFEOffsetElement SVGFEPointLightElement SVGFESpecularLightingElement
    SVGFESpotLightElement SVGFETileElement SVGFETurbulenceElement
    SVGFilterElement SVGFilterPrimitiveStandardAttributes SVGFitToViewBox
    SVGFontElement SVGFontFaceElement SVGForeignObjectElement SVGGElement
    SVGGeometryElement SVGGradientElement SVGGraphicsElement SVGImageElement
    SVGLength SVGLengthList SVGLinearGradientElement SVGLineElement
    SVGMarkerElement SVGMaskElement SVGMatrix SVGMetadataElement
    SVGMPathElement SVGNumber SVGNumberList SVGPathElement SVGPatternElement
    SVGPoint SVGPointList SVGPolygonElement SVGPolylineElement
    SVGPreserveAspectRatio SVGRadialGradientElement SVGRect SVGRectElement
    SVGScriptElement SVGSetElement SVGStopElement SVGStringList SVGStyleElement
    SVGSVGElement SVGSwitchElement SVGSymbolElement SVGTextContentElement
    SVGTextElement SVGTextPathElement SVGTextPositioningElement SVGTitleElement
    SVGTransform SVGTSpansList SVGURIReference SVGUnitTypes SVGUseElement
    SVGViewElement Screen ScreenAvailQuota ScreenOrientation ScrollTimeline
    ScrollY Serialization Session ServiceWorker ServiceWorkerContainer
    ServiceWorkerRegistration SharedWorker ShadowRoot SpeechSynthesis
    SpeechSynthesisEvent SpeechSynthesisUtterance SpeechSynthesisVoice StaticRange
    Storage StorageManager Stream Stub StyleMedia StyleSheet CSSStyleSheet2
    SubtleCrypto Text TextDecoder TextEncoder TextMetrics TextTrack
    TextTrackCue TextTrackCueList TextTrackList TimeRanges Touch TouchEvent
    TouchList TrackEvent TransitionEvent TreeWalker UIEvent URL URLSearchParams
    Uint8Array UTF8Decoder UTF8Encoder UserActivation VTTCue ValidityState
    VideoFrame View Transitions WaveShaperNode WebGLActiveInfo WebGLBuffer
    WebGLContextEvent WebGLFramebuffer WebGLProgram WebGLQuery
    WebGLRenderbuffer WebGLRenderingContext WebGLSampler WebGLShader
    WebGLShaderPrecisionFormat WebGLSync WebGLTexture WebGLTransformFeedback
    WebGLUniformLocation WebGLVertexArrayObject WebSocket WebSocketStream
    Widget Window WindowControlsOverwrite Worker Worklet WorkletGlobalScope
    WritableStream WritableStreamDefaultController WritableStreamDefaultWriter
    XMLDocument XmlHttpRequest XMLHttpRequest XMLHttpRequestEventTarget
    XMLHttpRequestUpload XSLTProcessor atob btoa createImageBitmap crypto
    customElements devicePixelRatio document fetch frameElement frames
    getComputedStyle history innerHeight innerWidth indexedDB localStorage
    location matchMedia name navigator onafterprint onbeforeprint
    onbeforeunload onerror onhashchange onlanguagechange onload onmessage
    onoffline ononline onpagehide onpageshow onpopstate onrejectionhandled
    onresize onstorage onunhandledrejection onunload open outerHeight
    outerWidth parent performance postMessage print prompt queueMicrotask
    registerProtocolHandler releaseEvents reportError requestAnimationFrame
    cancelAnimationFrame screen screenX screenY scroll scrollBy scrollTo
    self sessionStorage setTimeout setInterval clearTimeout clearInterval
    setImmediate clearImmediate structuredClone top trustedTypes visualViewport
    window worker localStorage
""".split())

# The second browser dump above is broad but not complete, and the frontend
# gate carried its own shorter list before this one became the only source.
# Merging it here added the Web API classes and window methods it had and the
# dump missed. What did NOT survive the merge were its HTML *tag* names and
# invented spellings (`Menu`, `Table`, `Form`, `Title`, `Data`, `Slot`,
# `WriteableStream`, `WebGLShaderPrecisionStatus`): they belong to no scope at
# all, so a call to one of them is precisely the bug these checks exist to
# find -- a helper the page was supposed to define and nobody did.
JS_GLOBAL_NAMES = JS_GLOBAL_NAMES | frozenset("""
    ByteLengthQueuingStrategy ChannelMergerNode ChannelSplitterNode CookieStore
    ErrorEvent FileReader GainNode History IndexedDB MessageChannel
    MessageEvent MessagePort OfflineAudioContext Selection SourceBuffer
    SourceBufferList StereoPannerNode StyleSheetList VisualViewport
    addEventListener alert close confirm dispatchEvent removeEventListener
""".split())

# Identifier shapes that are host globals by naming convention.
JS_GLOBAL_PREFIXES = ("HTML", "SVG", "MathMLElement", "WebGL", "OES_",
                      "ANGLE_", "EXT_", "KHR_", "DOM", "IDB", "RTC", "CSS",
                      "AudioWorklet", "Canvas", "TextTrack", "ServiceWorker")

# Reserved words and self-references: they look like names, never are ones.
JS_NON_NAMES = frozenset("""
    if else for while do return throw new typeof instanceof in of void delete
    function class const let var this super break continue switch case default
    try catch finally await async yield null true false get set static import
    export from as with debugger extends constructor keyof readonly declare
""".split())

# Names the usual CDN bundles put on the page as classic globals. A project
# that loads one of these can legitimately call into it from a module without
# an import, so the checks below refuse to guess about them.
JS_LIBRARY_GLOBALS = frozenset("""
    THREE Phaser PIXI Pixi GSAP TweenMax TimelineMax TimelineLite Power0
    Power1 Power2 Power3 Power4 Back Bounce Elastic Expo Circ Sine Quint
    SlowMo SteppedEase RoughEase EasePack CustomEase CustomWiggle
    CustomBounce ScrollTrigger Draggable ThrowPropsPlugin InertiaPlugin
    MorphSVGPlugin DrawSVGPlugin SplitText TextPlugin Flip gsap d3 jQuery $
    Chart chartjs moment dayjs marked showdown turndown hljs Prism Matter
    MatterJS Planck planck Tone ToneJS Howler Howl _ lodash underscore
    immutai Immutable React ReactDOM Vue Vuex Pinia Alpine animejs anime
    AOS Sortable SortableJS Dropzone dropzone p5 P5 ml5 babylon BJS
    CANNON cannon Ammo ammo Box2D box2d SocketIO io socket am4core am4charts
    am4maps am4plugins highcharts Highcharts Hammer Hammerjs fabric Fabric
    Kontra kontra Kha phaser PixiJS PixiGL three BabylonJS gl mat4 mat3 mat2
    vec4 vec3 vec2 quat quat2 glMatrix random Random seedrandom SimplexNoise
    noise ToneJS Howl Tone jsyt SpeechRecognition System SystemJS
    requirejs define modernizr Modernizr L Leaflet leaflet topojson
    turf d3scale chroma chromajs N3 RDFStore $
""".split())

# Same merge as the host list above: the analytics and loader bundles the gate
# also suppressed. The gate's list carried a few module members and one-off
# names as if they were globals (`createApp`, `ref`, `computed`, `charts`,
# `core`, `Note`, `mersenneTwister`); a file that calls one of those has
# either imported it or is broken, so they were left out on purpose.
JS_LIBRARY_GLOBALS = JS_LIBRARY_GLOBALS | frozenset("""
    Intercom dataLayer fbq google googleapis gtag
""".split())

# Test runners inject these into every test file's scope.
JS_TEST_GLOBALS = frozenset("""
    after afterEach afterAll before beforeEach beforeAll context describe it
    test expect assert chai jest mocha sinon should suite suiteSetup
    suiteTeardown teardown vi vitest xdescribe xit xtest step Given When Then
    assertStrictEquals assertDeepEqual fail pending skip only
""".split())

# import ... from './x.js' / export { a } from './y.js' (never `import(`).
_ESM_IMPORT_RE = re.compile(
    r"""^[ \t]*import(?!\s*[(.])(?P<clause>[^'"]*?)"""
    r"""(?:from\s*)?['"](?P<spec>[^'"]+)['"]""", re.M)
_BRACE_GROUP_RE = re.compile(r"\{(?P<body>[^{}]*)\}", re.S)
_NS_IMPORT_RE = re.compile(r"\*\s+as\s+([A-Za-z_$][\w$]*)")
_ESM_EXPORT_DECL_RE = re.compile(
    r"^[ \t]*export\s+(?:(?:default|async)\s+)?"
    r"(?:function\s*\*?\s*([A-Za-z_$][\w$]*)|class\s+([A-Za-z_$][\w$]*)"
    r"|(?:const|let|var)\s+([A-Za-z_$][\w$]*))", re.M)
_ESM_EXPORT_LIST_RE = re.compile(r"^[ \t]*export\s*\{(?P<body>[^{}]*\}?"
                                 r")", re.M | re.S)

# Anything that looks like a definition, at any depth: the generous "this name
# does get bound somewhere in this file" pool. Over-inclusive on purpose --
# skipping a real reference costs a finding, flagging a bound one costs trust.
_JS_DEF_ANYWHERE_RE = re.compile(
    r"\b(?:function|class)\s*\*?\s*([A-Za-z_$][\w$]*)"
    r"|\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)"
    r"|\b(?:const|let|var)\s*[\[{]([^}\]]*)[\]}]"
    r"|\b([A-Za-z_$][\w$]*)\s*=(?![=>])"
    r"|\b(?:window|globalThis|self|global)\s*(?:\.\s*|\[\s*['\"])([A-Za-z_$][\w$]*)",
)
# `spawnBurst(x, y) {` -- a method definition, in a class or an object literal.
_JS_METHOD_DEF_RE = re.compile(
    r"\b([A-Za-z_$][\w$]*)\s*\([^()]*\)\s*\{")

# Comments, strings and template literals: code a checker must not read.
# Blanking them (with the newlines kept, so line numbers stay honest) is what
# keeps `// Distant mountains (parallax 0.1)` and `'./game-core.js'` from
# being reported as calls to `mountains()` and `core()`.
_JS_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)

# What may stand before a `/` for it to open a regex literal instead of
# division. Skipping a regex body is not decoration: `/['"]/` would otherwise
# start a string that never ends and blank the rest of the file.
_JS_REGEX_PRECEDERS = "(,=:[!&|?{};+-*%~^<>\n"
_JS_STRING_OPENERS = ("'", '"', "`")


def _blank(m) -> str:
    body = m.group(0)
    if "\n" in body:                # keep the line count identical
        return "".join("\n" if ch == "\n" else " " for ch in body)
    return " " * len(body)


def _js_no_comments(text: str) -> str:
    """`text` with comments blanked, strings intact: an `import` statement
    still has to be readable, so its specifier stays in the text."""
    return _JS_COMMENT_RE.sub(_blank, text)


def _js_code(text: str) -> str:
    """`text` with comments, strings and template literals blanked to spaces.

    One regex pass cannot do this job. Lit-html-style code nests templates
    inside interpolations (`html`...${ok ? html`x` : null}...`), and a scanner
    that stops at the first inner backtick hands the rest of the markup back as
    code -- which is how the prose "when a model server is reachable (falls
    back to the ...)" was reported as a call to `reachable()` on a file that
    parses and runs fine. So the walk keeps a stack instead: quotes, templates,
    and the `${ }` regions that are code again, with regex literals skipped so a
    character class cannot open a string. Length and every newline are
    preserved, so an offset handed to `_line_of` still names its own line, and
    an unterminated quote recovers at the line break rather than blanking the
    rest of the file.
    """
    out: list[str] = []
    stack: list[list[str]] = []       # ["'" | \" | `], or ["${", brace depth]
    i, n = 0, len(text)
    prev = ""                         # last character still in code position
    while i < n:
        ch = text[i]
        top = stack[-1] if stack else None
        if top is not None and top[0] in ('\'', '"', "`"):
            quote = top[0]
            if ch == "\\":
                out.append(" ")
                if i + 1 < n:
                    out.append("\n" if text[i + 1] == "\n" else " ")
                    i += 1
                i += 1
                continue
            if ch == quote:
                stack.pop()
                out.append(" ")
                prev = quote
                i += 1
                continue
            if quote != "`" and ch == "\n":
                stack.pop()           # unterminated: recover, do not swallow
                out.append("\n")
                prev = "\n"
                i += 1
                continue
            if quote == "`" and ch == "$" and i + 1 < n and text[i + 1] == "{":
                stack.append(["${", "0"])
                out.append("  ")
                prev = "{"          # the interpolation opens in code position
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            body = text[i:j]
            out.append("".join("\n" if c == "\n" else " " for c in body))
            i = j
            prev = "*"
            continue
        if ch == "/" and (not prev or prev in _JS_REGEX_PRECEDERS):
            j, cls, i2 = i + 1, False, i + 1
            while i2 < n:
                c = text[i2]
                if c == "\\":
                    i2 += 2
                    continue
                if c == "\n":
                    break             # not a regex after all: leave it alone
                if cls:
                    if c == "]":
                        cls = False
                elif c == "[":
                    cls = True
                elif c == "/":
                    j = i2 + 1
                    break
                i2 += 1
            if j > i + 1:
                out.append(" " * (j - i))
                prev = ")"
                i = j
                continue
        if ch in _JS_STRING_OPENERS:
            stack.append([ch])
            out.append(" ")
            prev = ch
            i += 1
            continue
        # A `${ }` region is code again, so its own braces nest -- but only
        # once strings, comments and regex literals above have claimed their
        # characters. Counting `}` inside `/[}]/` or `// }` as the closing
        # brace ended the interpolation early and blanked every later real
        # call in it, hiding the ReferenceError this pass exists to catch.
        if top is not None and top[0] == "${":
            if ch == "{":
                top[1] = str(int(top[1]) + 1)
                out.append("{")
                prev = ch
                i += 1
                continue
            if ch == "}":
                if int(top[1]) > 0:
                    top[1] = str(int(top[1]) - 1)
                    out.append("}")
                else:
                    stack.pop()       # back into the template body
                    out.append(" ")
                prev = ch
                i += 1
                continue
        out.append(ch)
        if not ch.isspace():
            prev = ch
        i += 1
    return "".join(out)


# A destructuring head binds every name inside it (`const [a, b] =`,
# `const {x, y: z} =`), and a checker that reads the payload as one token
# reports `b` and `z` as unreachable calls.
_JS_BIND_TOKEN_RE = re.compile(r"[A-Za-z_$][\w$]*")


def _js_def_names(m) -> set[str]:
    """Every name one `_JS_DEF_ANYWHERE_RE` match binds."""
    groups = m.groups()
    if groups[2]:                     # `const|let|var [..]` / `{..}` payload
        return {t for t in _JS_BIND_TOKEN_RE.findall(groups[2])
                if t not in ("let", "const", "var")}
    for g in groups:
        if g:
            return {g}
    return set()


# Parameters: `(e) => ...`, `function f(a, b)`, `catch (err)`, `m(x) {`.
# Every short paren group in the file is read as a parameter list -- grossly
# over-inclusive (`if (ready)` binds `ready`), which is the right direction:
# an extra local name only ever costs a finding, never a false one.
_JS_PAREN_GROUP_RE = re.compile(r"\(([^()]*)\)")
_JS_ARROW_PARAM_RE = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*=>")


def _js_paren_bound(text: str) -> set[str]:
    """Identifiers bound by parameter and catch lists anywhere in `text`."""
    names: set[str] = set()
    for m in _JS_PAREN_GROUP_RE.finditer(text):
        for part in m.group(1).split(","):
            part = part.strip().lstrip("{[").rstrip("}]").strip()
            if not part:
                continue
            head = re.split(r"[=:?]", part, maxsplit=1)[0].strip()
            for token in re.findall(r"[A-Za-z_$][\w$]*", head):
                if token not in ("let", "const", "var"):
                    names.add(token)
        for token in re.findall(r"[A-Za-z_$][\w$]*", m.group(1)):
            if "{" in m.group(1) and token not in ("let", "const", "var"):
                names.add(token)      # destructuring: every key is a binding
    # A single-parameter arrow (`m => ...`) has no paren group, and the call
    # form matches `m.name` too: without this, every `rows.forEach(m => ...)`
    # callback using `m.something` was reported as calling an undefined `m()`.
    for m in _JS_ARROW_PARAM_RE.finditer(text):
        names.add(m.group(1))
    return names


# The call form of a name: `NAME(...)`, `new NAME(...)`, `NAME.member(...)`.
# The lookbehind keeps member reads (`a.NAME(`) out of it: the base is the
# only part a module scope has to provide.
_JS_CALL_FORM_RE = re.compile(
    r"(?<![\w$.])(?:new\s+)?([A-Za-z_$][\w$]*)\s*(?:\(|\.[A-Za-z_$])")


def _js_is_host_name(name: str) -> bool:
    """True when a bare `name` needs no declaration: host, library, keyword."""
    if (name in JS_GLOBAL_NAMES or name in JS_LIBRARY_GLOBALS
            or name in JS_TEST_GLOBALS or name in JS_NON_NAMES):
        return True
    return any(name.startswith(p) for p in JS_GLOBAL_PREFIXES)


def _js_import_names(clause: str) -> set[str]:
    """Locals one `import ... from '...'` statement binds."""
    names: set[str] = set()
    body, brace_pos = (clause.split("{", 1)[0], clause.find("{"))
    head = body.replace("from", "").strip()
    for part in head.split(","):
        part = part.strip().rstrip("*").strip()
        if part and re.fullmatch(r"(?:type\s+)?[A-Za-z_$][\w$]*", part or ""):
            names.add(part.split()[-1])
    if brace_pos >= 0:
        braces = _BRACE_GROUP_RE.search("{" + clause[brace_pos:])
        if braces:
            for item in braces.group("body").split(","):
                item = item.strip()
                if not item or item in ("type", "*"):
                    continue
                names.add(item.split()[-1] if " as " in item
                          else item.replace("type ", "").strip())
    for m in _NS_IMPORT_RE.finditer(clause):
        names.add(m.group(1))
    return {n for n in names if n and n not in JS_NON_NAMES}


def _js_exported_names(text: str) -> set[str]:
    """Names an ES module puts on its public surface (not `default`)."""
    exported: set[str] = set()
    text = _js_no_comments(text)
    for m in _ESM_EXPORT_DECL_RE.finditer(text):
        exported.add(_first_group(m))
    for m in _ESM_EXPORT_LIST_RE.finditer(text):
        for item in m.group("body").split(","):
            item = item.strip().rstrip(";").strip()
            if not item or item == "default":
                continue
            if " as " in item:
                exported.add(item.split(" as ")[1].strip())
            elif item != "*":
                exported.add(item.replace("type ", "").strip())
    return {n for n in exported if n and re.fullmatch(r"[A-Za-z_$][\w$]*", n)}


def _js_module_links(htmls, jss, scopes, texts, root):
    """Checks 6 and 7: duplicate module bindings and unreachable call targets."""
    module_rels = {sc.rel for sc in scopes if sc.module and not sc.inline}
    classic_top: set[str] = set()
    exported_by: dict[str, list[str]] = {}
    declared_in: dict[str, list[str]] = {}
    defined_anywhere: set[str] = set()
    for rel, text in jss:
        for m in _JS_DEF_ANYWHERE_RE.finditer(text):
            defined_anywhere.update(_js_def_names(m))
        for m in _JS_METHOD_DEF_RE.finditer(text):
            defined_anywhere.add(m.group(1))
    for rel, text in htmls:
        for m in _JS_DEF_ANYWHERE_RE.finditer(text):
            defined_anywhere.update(_js_def_names(m))
        for m in _JS_METHOD_DEF_RE.finditer(text):
            defined_anywhere.add(m.group(1))
    exports_of: dict[str, set[str]] = {}
    declared_by: dict[str, set[str]] = {}
    for rel, text in jss:
        if rel not in module_rels:
            classic_top.update(m.group(1) or _first_group(m)
                               for m in _TOPLEVEL_RE.finditer(text))
            continue
        exported = _js_exported_names(text)
        exports_of[rel] = exported
        for name in exported:
            exported_by.setdefault(name, []).append(rel)
        bound = {m.group(1) or _first_group(m) for m in
                 _TOPLEVEL_RE.finditer(text)} | _js_bound_by_imports(text)
        for name in bound - exported:
            declared_in.setdefault(name, []).append(rel)
        declared_by[rel] = bound | exported
    declared_in.pop("default", None)
    externals = any(
        _resolve_src(root, rel, _first_group(sm))[1]
        for rel, text in htmls for sm in _SRC_ATTR_RE.finditer(text))

    findings: list[Finding] = []
    told: set[tuple[str, str]] = set()
    for rel, text in jss:
        if rel not in module_rels:
            continue
        # 6: a name bound twice at module level. Only `var` may repeat; a
        # function, a lexical binding, a class or an import may not, and any
        # mix of the two is the same SyntaxError.
        seen: dict[str, tuple[int, str]] = {}
        code = _js_code(text)
        stmts = _js_no_comments(text)
        for m in _TOPLEVEL_RE.finditer(code):
            g = m.groups()
            name = g[0] or g[1] or g[3] or g[4]
            kind = ("function" if g[0] else
                    "lexical" if (g[1] or g[3]) else "hoisted")
            _report_duplicate(findings, seen, name,
                              _line_of(text, m.start()), kind, rel)
        for m in _ESM_IMPORT_RE.finditer(stmts):
            line = _line_of(stmts, m.start())
            for name in sorted(_js_import_names(m.group("clause"))):
                _report_duplicate(findings, seen, name, line, "import", rel)
        # 7: calls to names this module cannot reach. Uses are read out of the
        # code (`_js_code`), bindings out of the raw file -- a name defined in
        # a string or comment was never a definition, an over-wide parameter
        # sweep only ever skips a reference.
        code = _js_code(text)
        bound = ({n for m in _JS_DEF_ANYWHERE_RE.finditer(text)
                  for n in _js_def_names(m)}
                 | {m.group(1) for m in _JS_METHOD_DEF_RE.finditer(text)}
                 | _js_paren_bound(code) | _js_bound_by_imports(stmts))
        for m in _JS_CALL_FORM_RE.finditer(code):
            name = m.group(1)
            if (name in bound or (rel, name) in told
                    or _js_is_host_name(name)):
                continue
            told.add((rel, name))
            line = _line_of(text, m.start())
            holders = [h for h in exported_by.get(name, []) if h != rel]
            if holders:
                findings.append(Finding(
                    rel, line,
                    f"calls {name}() but this module never imports it -- "
                    f"{holders[0]} exports it and a module only sees names it "
                    f"declares or imports, so this line is a ReferenceError "
                    f"(add `import {{ {name} }} from "
                    f"'{_sibling_spec(rel, holders[0])}'`)",
                    peer=holders[0]))
                continue
            hidden = [h for h in declared_in.get(name, []) if h != rel]
            if hidden and name not in classic_top:
                findings.append(Finding(
                    rel, line,
                    f"calls {name}(), which {hidden[0]} declares but never "
                    f"exports -- modules do not share a scope, so this file "
                    f"cannot reach it (export it in {hidden[0]} and import it "
                    f"here)", peer=hidden[0]))
                continue
            if name in defined_anywhere or name in classic_top or externals:
                continue   # a page global, or a CDN bundle: not provable
            findings.append(Finding(
                rel, line,
                f"calls {name}() but nothing in the project defines it and no "
                "module exports it -- the ReferenceError stops this file the "
                "first time the line runs (declare it here, import it from "
                "the file that should own it, or remove the call)"))
    return findings


# --- checks 8+9: runtime-fatal JS that parses clean -------------------------
# Regex twins of spec._js_const_reassign_errors / _js_instance_method_errors
# (the tree-sitter gates behind `verify_frontend_contracts`). Same
# conservatism: anything shadowed, redeclared, reassigned or unreadable stays
# quiet -- a missed notice costs nothing, a false one costs trust.

_JS_CONST_DECL_RE = re.compile(r"\bconst\s+([A-Za-z_$][\w$]*)\s*(?:=|\bof\b|\bin\b)")
_JS_REDECL_RE = re.compile(r"\b(?:let|var|function|class)\s+([A-Za-z_$][\w$]*)")
# `x =`, `x +=`, `x++` -- but not `x ==`, `x ===`, `x =>`, `a.x =`
_JS_ASSIGN_RE = re.compile(
    r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*(?:[+\-*/%&|^]?=(?!=|>)|\+\+|--)")
# Parameter lists only: `function f(a) {`, `m(a) {`, `catch (e) {`,
# `(a, b) =>`, `e =>`. Deliberately narrower than `_js_paren_bound`, which
# reads every paren group as parameters: a const reassignment usually sits
# inside a callback body, and counting the callback's own argument list
# would hide exactly the bug check 8 exists for.
_JS_PARAM_LIST_RE = re.compile(
    r"(?:\bfunction\s*[A-Za-z_$\w]*\s*|\bcatch\s*"
    r"|(?<![\w$.])(?!(?:if|for|while|switch|with)\b)[A-Za-z_$][\w$]*\s*)"
    r"\(([^()]*)\)(?=\s*(?:\{|=>))"
    r"|\(([^()]*)\)\s*=>"
    r"|(?<![\w$.])([A-Za-z_$][\w$]*)\s*=>")


def _js_param_names(code: str) -> set[str]:
    """Names bound as a parameter or catch variable anywhere in `code`."""
    names: set[str] = set()
    for m in _JS_PARAM_LIST_RE.finditer(code):
        if m.group(3):
            names.add(m.group(3))
            continue
        payload = m.group(1) if m.group(1) is not None else m.group(2)
        for part in (payload or "").split(","):
            part = part.strip().lstrip("{[").rstrip("}]").strip()
            if not part:
                continue
            head = re.split(r"[=:?]", part, maxsplit=1)[0].strip()
            for token in re.findall(r"[A-Za-z_$][\w$]*", head):
                if token not in ("let", "const", "var"):
                    names.add(token)
    return names


def _const_reassign_findings(scopes) -> list:
    """`const x` followed by an assignment to x: parses clean, then throws
    `TypeError: Assignment to constant variable` the first time the line
    runs (BUG-210)."""
    findings: list[Finding] = []
    for sc in scopes:
        code = _js_code(sc.text)
        decls: dict[str, int] = {}
        for m in _JS_CONST_DECL_RE.finditer(code):
            decls.setdefault(m.group(1),
                             _line_of(sc.text, m.start()) + sc.offset)
        if not decls:
            continue
        redeclared = set(_JS_REDECL_RE.findall(code))
        params = _js_param_names(code)
        for m in _JS_ASSIGN_RE.finditer(code):
            name = m.group(1)
            line = _line_of(sc.text, m.start()) + sc.offset
            if name not in decls or line < decls[name]:
                continue
            if name in redeclared or name in params:
                continue
            # a declaration in the same statement (`const x = 1, y = 2`,
            # `const x = 1; x = 2` -- only the first is a binding)
            start = max(code.rfind("\n", 0, m.start()),
                        code.rfind(";", 0, m.start()),
                        code.rfind("{", 0, m.start()),
                        code.rfind("}", 0, m.start()))
            if re.search(r"\b(?:const|let|var)\b", code[start + 1:m.start()]):
                continue
            findings.append(Finding(
                sc.rel, line,
                f"'{name}' is declared const (line {decls[name]}) and then "
                f"reassigned here -- JavaScript raises `TypeError: Assignment "
                f"to constant variable` the moment this line runs; declare "
                f"'{name}' with let if its value must change"))
    return findings


_JS_CLASS_HEAD_RE = re.compile(r"\bclass\s+([A-Za-z_$][\w$]*)")
# One member head: `m(`, `get m(`, `static async m(`, `*gen(`, `#priv(`,
# or a field `f =`. Matched at a member boundary, never inside a body.
_JS_MEMBER_RE = re.compile(
    r"(?:static\s+)?(?:(?:get|set)\s+)?(?:async\s+)?(?:\*\s*)?"
    r"(#?[A-Za-z_$][\w$]*)\s*(\(|=)")
_JS_NEW_INST_RE = re.compile(
    r"\bconst\s+([A-Za-z_$][\w$]*)\s*=\s*new\s+([A-Za-z_$][\w$]*)\s*\(")
_JS_THIS_NEW_RE = re.compile(
    r"\bthis\.([A-Za-z_$][\w$]*)\s*=\s*new\s+([A-Za-z_$][\w$]*)\s*\(")
_JS_THIS_ASSIGN_RE = re.compile(
    r"\bthis\.([A-Za-z_$][\w$]*)\s*(?:[+\-*/%&|^]?=(?!=|>)|\+\+|--)")
_JS_MEMBER_CALL_RE = re.compile(
    r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\.\s*([A-Za-z_$][\w$]*)\s*\(")
_JS_THIS_CALL_RE = re.compile(
    r"\bthis\.([A-Za-z_$][\w$]*)\s*\.\s*([A-Za-z_$][\w$]*)\s*\(")
_JS_METHOD_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "function", "new",
    "typeof", "do", "else", "with", "await", "yield", "case", "delete",
    "void", "in", "of", "instanceof", "constructor",
}
# Every object answers these, whatever its class defines (twin of
# spec._JS_OBJECT_PROTO; kept local so webcheck never imports spec).
_JS_OBJECT_PROTO = {
    "constructor", "hasOwnProperty", "isPrototypeOf", "propertyIsEnumerable",
    "toLocaleString", "toString", "valueOf", "__defineGetter__",
    "__defineSetter__", "__lookupGetter__", "__lookupSetter__", "__proto__",
}


def _js_class_bodies(code: str):
    """(name, body_start, body_end) for each parentless class declaration,
    brace-scanned on string/comment-blanked code. A class with `extends`
    is skipped: the parent could define any method the child lacks."""
    out = []
    for m in _JS_CLASS_HEAD_RE.finditer(code):
        i, depth = m.end(), 0
        while i < len(code):
            c = code[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth = max(0, depth - 1)
            elif c == "{" and depth == 0:
                break
            i += 1
        else:
            continue
        if re.search(r"\bextends\b", code[m.end():i]):
            continue
        j, d = i + 1, 1
        while j < len(code) and d:
            if code[j] == "{":
                d += 1
            elif code[j] == "}":
                d -= 1
            j += 1
        out.append((m.group(1), i + 1, j - 1))
    return out


def _js_class_surface(code: str, start: int, end: int):
    """(member names, opaque) for one class body: method and field heads at
    brace depth 0, outside any parameter list. A computed member, a
    decorator or an unreadable member shape makes the surface unprovable,
    exactly as in spec._js_class_surface."""
    methods: set[str] = set()
    depth = parens = 0
    member_start = True
    i = start
    while i < end:
        c = code[i]
        if c == "(":
            parens += 1
        elif c == ")":
            parens = max(0, parens - 1)
        elif c == "{":
            depth += 1
            member_start = False
        elif c == "}":
            depth -= 1
            member_start = depth == 0 and parens == 0
        elif depth == 0 and parens == 0:
            if c.isspace():
                pass
            elif c == ";":
                member_start = True
            elif member_start:
                if c in "[@":
                    return methods, True
                if re.match(r"static\s*\{", code[i:end]):
                    member_start = False   # static block: shapes the class,
                    i += 1                 # not the instance surface
                    continue
                m = _JS_MEMBER_RE.match(code, i)
                if m is None:
                    return methods, True   # unexpected member: never guess
                if m.group(1) not in _JS_METHOD_KEYWORDS:
                    methods.add(m.group(1))
                member_start = False
        i += 1
    return methods, False


def _instance_method_findings(scopes) -> list:
    """`x.method()` where x is a provable project-class instance and the
    class defines no such method: `TypeError: x.method is not a function`
    on the first call (BUG-211). Only the shapes spec.py proves are
    reported: a top-level `const x = new Class()` whose binding is never
    shadowed or reassigned, or `this.x = new Class()` inside a parentless
    class. Calls through a parameter (`wireTransport(el, engine)`) are a
    dataflow question the regex level cannot answer and stay quiet."""
    classes: dict[str, tuple] = {}     # name -> (rel, methods)
    ambiguous: set[str] = set()
    bodies: list[list] = []            # per scope: [(name, start, end)]
    codes: list[str] = []
    for sc in scopes:
        code = _js_code(sc.text)
        codes.append(code)
        found = _js_class_bodies(code)
        bodies.append(found)
        for name, start, end in found:
            methods, opaque = _js_class_surface(code, start, end)
            if opaque or name in classes:
                ambiguous.add(name)
                continue
            classes[name] = (sc.rel, methods)
    for name in ambiguous:
        classes.pop(name, None)

    findings: list[Finding] = []
    for idx, sc in enumerate(scopes):
        code = codes[idx]
        params = _js_param_names(code)
        redeclared = set(_JS_REDECL_RE.findall(code))
        # top-level `const x = new Class()`: brace depth 0 at the declaration
        instances: dict[str, tuple] = {}   # x -> (class, decl name pos)
        new_matches = {m.start(): m for m in _JS_NEW_INST_RE.finditer(code)}
        depth = 0
        top: dict[int, object] = {}
        for i, c in enumerate(code):
            if i in new_matches and depth == 0:
                top[i] = new_matches[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
        unstable: set[str] = set()
        for m in top.values():
            x, cname = m.group(1), m.group(2)
            if cname not in classes:
                continue
            if x in params or x in redeclared or x in instances:
                unstable.add(x)
                continue
            instances[x] = (cname, m.start(1))
        for m in _JS_ASSIGN_RE.finditer(code):
            x = m.group(1)
            if x in instances and m.start() != instances[x][1]:
                unstable.add(x)
        for x in unstable:
            instances.pop(x, None)

        def emit(receiver: str, method: str, cname: str, pos: int):
            if method in _JS_OBJECT_PROTO or method in _BUILTIN_MEMBERS:
                return
            def_rel, methods = classes[cname]
            if method in methods:
                return
            shown = ", ".join(sorted(methods)[:8]) or "(none)"
            findings.append(Finding(
                sc.rel, _line_of(sc.text, pos) + sc.offset,
                f"calls {receiver}.{method}(...), but class {cname} "
                f"({def_rel}) defines no such method (it defines: {shown}) "
                f"-- `TypeError: {receiver}.{method} is not a function` the "
                f"first time this line runs; add {method} to {cname} or call "
                f"a method it defines", peer=def_rel))

        for m in _JS_MEMBER_CALL_RE.finditer(code):
            x, method = m.group(1), m.group(2)
            if x in instances:
                emit(x, method, instances[x][0], m.start())
        # this.x instances, per parentless class body
        for cname, start, end in bodies[idx]:
            body = code[start:end]
            this_inst: dict[str, str] = {}
            this_bad: set[str] = set()
            for m in _JS_THIS_ASSIGN_RE.finditer(body):
                x = m.group(1)
                nm = _JS_THIS_NEW_RE.match(body, m.start())
                if nm is None or nm.group(2) not in classes:
                    this_bad.add(x)
                elif x in this_inst and this_inst[x] != nm.group(2):
                    this_bad.add(x)
                else:
                    this_inst[x] = nm.group(2)
            for m in _JS_THIS_CALL_RE.finditer(body):
                x, method = m.group(1), m.group(2)
                if x in this_inst and x not in this_bad:
                    emit("this." + x, method, this_inst[x], start + m.start())
    return findings


def _js_bound_by_imports(text: str) -> set[str]:
    names: set[str] = set()
    for m in _ESM_IMPORT_RE.finditer(_js_no_comments(text)):
        names |= _js_import_names(m.group("clause"))
    return names


def _report_duplicate(findings, seen, name, line, kind, rel) -> None:
    """One finding per re-bound module-level name, worded for the second one."""
    prev = seen.get(name)
    if prev is None:
        seen[name] = (line, kind)
        return
    if kind == "hoisted" and prev[1] == "hoisted":
        return               # two `var f`s are one binding; a module refuses
                             # repeated functions and var/function mixes
    this = {"import": "imported again here", "lexical": "declared again here",
            "function": "defined again here", "hoisted": "defined again here"}
    that = {"import": f"imported at line {prev[0]}",
            "lexical": f"declared at line {prev[0]}",
            "function": f"defined at line {prev[0]}",
            "hoisted": f"defined at line {prev[0]}"}
    findings.append(Finding(
        rel, line,
        f"'{name}' is {this[kind]}, but it is already {that[prev[1]]} -- both "
        f"live in this module's own scope, so the browser refuses the file "
        f"with `SyntaxError: Identifier '{name}' has already been declared` "
        f"and none of its exports exist; keep exactly one binding for it"))


def _sibling_spec(from_rel: str, target_rel: str) -> str:
    """The ES specifier `from_rel` writes to reach `target_rel`."""
    import posixpath
    spec = posixpath.relpath(target_rel, posixpath.dirname(from_rel) or ".")
    return spec if spec.startswith((".", "/")) else "./" + spec


def _comment_only(text: str, html: bool = False) -> bool:
    """True when `text` holds nothing but comments and whitespace."""
    s = text
    if html:
        s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
        s = re.sub(r"<!doctype[^>]*>", "", s, flags=re.I)
    else:
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
        s = re.sub(r"//[^\n]*", "", s)
    return not s.strip()


def _empty_file_findings(htmls, jss, csss) -> list[Finding]:
    """Effectively empty or comment-only HTML/JS/CSS files: a page, script or
    stylesheet that ships nothing is a silent, broken deliverable."""
    out: list[Finding] = []
    for rel, text in htmls:
        if _comment_only(text, html=True):
            out.append(Finding(rel, 1, "empty or comment-only HTML file -- the "
                               "page renders nothing"))
    for rel, text in jss:
        if _comment_only(text):
            out.append(Finding(rel, 1, "empty or comment-only JS file -- no "
                               "code runs when it loads"))
    for rel, text in csss:
        if _comment_only(text):
            out.append(Finding(rel, 1, "empty or comment-only CSS file -- the "
                               "stylesheet styles nothing"))
    return out


def _css_slash_comment_findings(csss) -> list[Finding]:
    """`//` is not a CSS comment (CSS uses /* */): a line starting with `//`
    is invalid CSS and the browser skips the rule that follows."""
    out: list[Finding] = []
    for rel, text in csss:
        for m in re.finditer(r"^[ \t]*//[^\n]*", text, re.M):
            out.append(Finding(
                rel, _line_of(text, m.start()),
                "`//` is not a CSS comment (CSS uses /* */); the browser "
                "treats this line as invalid and skips the rule"))
    return out


def _unreferenced_sibling_findings(root: Path, htmls, jss, csss
                                   ) -> list[Finding]:
    """Sibling .js/.css files no .html references (directly or through another
    JS file's import): they never load, so their content never runs."""
    if not htmls:
        return []
    referenced: set[str] = set()
    for rel, text in htmls:
        for m in _SCRIPT_TAG_RE.finditer(text):
            sm = _SRC_ATTR_RE.search(m.group(1))
            if sm:
                src = _first_group(sm).strip().split("?")[0].split("#")[0]
                js_rel, external = _resolve_src(root, rel, src)
                if not external and js_rel:
                    referenced.add(js_rel)
        for m in _LINK_TAG_RE.finditer(text):
            rm = _REL_ATTR_RE.search(m.group(1))
            if not rm or "stylesheet" not in _first_group(rm).lower():
                continue
            hm = _HREF_ATTR_RE.search(m.group(1))
            if not hm:
                continue
            href = _first_group(hm).strip().split("?")[0].split("#")[0]
            css_rel, external = _resolve_src(root, rel, href)
            if not external and css_rel:
                referenced.add(css_rel)
    # A JS file another JS file imports is part of the module graph even when
    # no <script src> names it directly.
    for rel, text in jss:
        for m in _ESM_IMPORT_RE.finditer(text):
            spec = m.group("spec")
            for cand in (spec, spec + ".js", spec + ".mjs"):
                target, external = _resolve_src(root, rel, cand)
                if not external and target:
                    referenced.add(target)
                    break
    out: list[Finding] = []
    for rel, _text in [*jss, *csss]:
        if rel in referenced:
            continue
        kind = "JS" if rel.endswith((".js", ".mjs", ".cjs")) else "CSS"
        out.append(Finding(
            rel, 1,
            f"this {kind} file is not referenced by any .html file (and no "
            "sibling imports it) -- it never loads"))
    return out


def run(root, files: list[str] | None = None) -> list[Finding]:
    """Every contract finding for the project under `root`, sorted by file
    and line. With `files`, only findings whose own file OR whose contract
    peer is among them -- a post-edit pass cares about both sides of a
    broken contract, and nothing else."""
    root = Path(root)
    htmls, jss, csss = _collect(root)
    if not htmls and not jss:
        return []
    texts = dict(htmls)
    texts.update(dict(jss))
    findings: list[Finding] = []

    # --- ids available anywhere in the project (HTML attrs + JS-built) ----
    ids: set[str] = set()
    for _rel, text in htmls:
        for m in _ID_ATTR_RE.finditer(text):
            if _first_group(m):
                ids.add(_first_group(m))
    for text in texts.values():
        for rx in (_JS_ID_RE, _SET_ID_RE):
            for m in rx.finditer(text):
                if _first_group(m):
                    ids.add(_first_group(m))

    # --- script tags: load map, inline scopes, dangling src ---------------
    loads: dict[str, list] = {}          # js rel -> [(html, line, module, order)]
    page_scopes: dict[str, list] = {}    # html rel -> [(order, classic scope)]
    scopes: list[_Scope] = []
    order = 0
    for rel, text in htmls:
        for m in _SCRIPT_TAG_RE.finditer(text):
            attrs, content = m.group(1), m.group(2)
            tm = _TYPE_ATTR_RE.search(attrs)
            module = bool(tm) and _first_group(tm).strip().lower() == "module"
            tag_line = _line_of(text, m.start())
            sm = _SRC_ATTR_RE.search(attrs)
            if sm:
                src = _first_group(sm).strip().split("?")[0].split("#")[0]
                js_rel, external = _resolve_src(root, rel, src)
                if external:
                    continue
                if js_rel in texts:
                    loads.setdefault(js_rel, []).append(
                        (rel, tag_line, module, order))
                    order += 1
                elif not (root / js_rel).exists():
                    findings.append(Finding(
                        rel, tag_line,
                        f'<script src="{src}"> points at a file that does not '
                        "exist -- the browser 404s on it and none of its "
                        "code runs", peer=js_rel))
                continue
            if not content.strip():
                continue
            scope = _Scope(rel=rel, text=content,
                           offset=_line_of(text, m.start(2)) - 1,
                           module=module, inline=True, peer=rel)
            scopes.append(scope)
            if not module:
                page_scopes.setdefault(rel, []).append((order, scope))
            order += 1
    for rel, text in jss:
        ls = loads.get(rel, [])
        if ls:
            module = all(l[2] for l in ls)
            peer = ls[0][0]
        else:
            # Never referenced by a page: judge by syntax (an ESM file is a
            # module; anything else may still be injected as a classic one).
            module, peer = bool(_ESM_RE.search(text)), None
        sc = _Scope(rel=rel, text=text, offset=0, module=module,
                    inline=False, peer=peer)
        scopes.append(sc)
        if not module:
            # A classic load puts this file into that page's global scope,
            # in document order (a file loaded as a module elsewhere keeps
            # its own scope on that page).
            for l in ls:
                if not l[2]:
                    page_scopes.setdefault(l[0], []).append((l[3], sc))

    # --- check 2: ESM syntax behind a classic <script> ---------------------
    for sc in scopes:
        m = _ESM_RE.search(sc.text)
        if sc.module or not m:
            continue
        kw = m.group(0).strip().split()[0]
        if sc.inline:
            findings.append(Finding(
                sc.rel, sc.offset + _line_of(sc.text, m.start()),
                f'inline <script> uses ESM `{kw}` without type="module" -- '
                "the browser aborts it with a SyntaxError", peer=sc.rel))
            continue
        classic = [l for l in loads.get(sc.rel, []) if not l[2]]
        if not classic:
            continue
        extra = (f" (and {len(classic) - 1} other page(s))"
                 if len(classic) > 1 else "")
        findings.append(Finding(
            sc.rel, _line_of(sc.text, m.start()),
            f"uses ESM `{kw}` but {classic[0][0]} loads it through a classic "
            f'<script src>{extra} -- add type="module" to the tag; browsers '
            "abort classic scripts on import/export with a SyntaxError",
            peer=classic[0][0]))

    # --- check 3: duplicate top-level globals on a shared page -------------
    for page, entries in sorted(page_scopes.items()):
        seen: dict[str, tuple] = {}
        for _ord, sc in sorted(entries, key=lambda e: e[0]):
            for m in _TOPLEVEL_RE.finditer(sc.text):
                g = m.groups()
                name = g[0] or g[1] or g[3] or g[4]
                kind = ("function" if g[0] else
                        "lexical" if (g[1] or g[3]) else "var")
                line = sc.offset + _line_of(sc.text, m.start())
                prev = seen.get(name)
                if prev is None:
                    seen[name] = (sc, line, kind)
                elif prev[0] is not sc:
                    why = ("redeclaring a top-level let/const/class in one "
                           "global scope is a SyntaxError that kills this "
                           "whole file"
                           if "lexical" in (kind, prev[2]) else
                           "the later classic script silently replaces the "
                           "earlier definition")
                    findings.append(Finding(
                        sc.rel, line,
                        f"top-level '{name}' is already declared in "
                        f"{prev[0].rel}:{prev[1]} -- both load as classic "
                        f"scripts on {page} and share one global scope: {why}",
                        peer=prev[0].rel))

    # --- check 1: DOM id references ----------------------------------------
    for sc in scopes:
        findings.extend(_id_findings(sc.rel, sc.text, sc.offset, ids,
                                     sc.peer))
    for rel, text in htmls:
        for m in _HANDLER_ATTR_RE.finditer(text):
            findings.extend(_id_findings(rel, _first_group(m),
                                         _line_of(text, m.start()) - 1, ids,
                                         rel))

    # --- check 10: empty <select> controls (BUG-324) ------------------------
    findings.extend(_empty_select_findings(htmls, jss))

    # --- check 11: unhidden honeypots and unlabelled form fields ------------
    findings.extend(_form_input_findings(htmls, csss, root))

    # --- check 12: raw @tailwind directives in a linked stylesheet ----------
    findings.extend(_tailwind_directive_findings(htmls, csss, root))

    # --- check 13: Tailwind utility classes with no CDN / compiled sheet ----
    findings.extend(_tailwind_utility_findings(htmls, csss, root))

    # --- check 4: bare imports vs package.json ------------------------------
    deps = _pkg_deps(root)
    if deps is not None:
        for sc in scopes:
            if sc.inline:
                continue      # an inline module may resolve via an import map
            seen_spec: set[str] = set()
            for rx in (_FROM_RE, _BARE_IMPORT_RE, _REQUIRE_RE):
                for m in rx.finditer(sc.text):
                    spec = _first_group(m)
                    bare = _bare_spec(spec)
                    if not bare or bare in deps or bare in _NODE_BUILTINS \
                            or spec in seen_spec:
                        continue
                    seen_spec.add(spec)
                    findings.append(Finding(
                        sc.rel, _line_of(sc.text, m.start()),
                        f"'{spec}' is imported but package.json declares no "
                        "such dependency -- Node and bundlers fail with "
                        "ERR_MODULE_NOT_FOUND (add it to dependencies or "
                        "devDependencies)"))

    # --- check 5: member calls on singly-defined globals ---------------------
    defs: dict[str, set[str]] = {}
    for sc in scopes:
        if sc.module:
            continue
        for rx in (_OBJ_DEF_RE, _FN_DEF_RE, _CLASS_DEF_RE, _WIN_DEF_RE):
            for m in rx.finditer(sc.text):
                defs.setdefault(m.group(1), set()).add(sc.rel)
    single = {n: next(iter(rels)) for n, rels in defs.items()
              if len(rels) == 1}
    for sc in scopes:
        findings.extend(_member_findings(sc.rel, sc.text, sc.offset, single,
                                         texts))
    for rel, text in htmls:
        for m in _HANDLER_ATTR_RE.finditer(text):
            findings.extend(_member_findings(
                rel, _first_group(m), _line_of(text, m.start()) - 1, single,
                texts))

    # --- checks 6+7: module bindings and module links ----------------------
    # (regex twins of spec._js_module_link_errors, which is the vetoing gate)
    findings.extend(_js_module_links(htmls, jss, scopes, texts, root))

    # --- checks 8+9: const reassignment and missing instance methods -------
    # (regex twins of spec._js_const_reassign_errors /
    #  _js_instance_method_errors -- runtime-fatal, parse-cleans JS)
    findings.extend(_const_reassign_findings(scopes))
    findings.extend(_instance_method_findings(scopes))

    # --- check 14: empty/comment-only files, `//` CSS comments, orphans -----
    findings.extend(_empty_file_findings(htmls, jss, csss))
    findings.extend(_css_slash_comment_findings(csss))
    findings.extend(_unreferenced_sibling_findings(root, htmls, jss, csss))

    # --- dedupe, filter, sort ----------------------------------------------
    uniq: list[Finding] = []
    seen_keys: set = set()
    for f in findings:
        key = (f.rel, f.line, f.message)
        if key not in seen_keys:
            seen_keys.add(key)
            uniq.append(f)
    if files:
        keep = {Path(f).as_posix() for f in files}
        uniq = [f for f in uniq if f.rel in keep or f.peer in keep]
    uniq.sort(key=lambda f: (f.rel, f.line))
    return uniq


def notice(root, files: list[str] | None = None, limit: int = 8) -> int:
    """Print the findings as `[webcheck]` warnings on stderr and return how
    many there were. Best-effort: a broken project tree must never break the
    caller (the scaffold summary, the auto loop), so this swallows every
    error."""
    try:
        found = run(root, files)
    except Exception:
        return 0
    for f in found[:limit]:
        print(f"  [webcheck] {f}", file=sys.stderr)
    if len(found) > limit:
        print(f"  [webcheck] ... and {len(found) - limit} more "
              "(pedacito webcheck)", file=sys.stderr)
    return len(found)
