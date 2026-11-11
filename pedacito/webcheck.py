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
    in the one classic script that defines ``Game`` (TypeError).

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
    """(html files, js files) under root, each a (rel, text) pair, both
    sorted by rel. Skips build/vendored dirs and caps count and size so a
    post-edit pass stays at regex speed."""
    htmls: list[tuple[str, str]] = []
    jss: list[tuple[str, str]] = []
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in _SKIP_DIRS and not d.startswith("."))
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _WEB_EXT:
                continue
            p = Path(dirpath) / fn
            try:
                if p.stat().st_size > _MAX_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = p.relative_to(root).as_posix()
            (htmls if ext in _HTML_EXT else jss).append((rel, text))
            n += 1
            if n >= _MAX_FILES:
                htmls.sort(key=lambda t: t[0])
                jss.sort(key=lambda t: t[0])
                return htmls, jss
    htmls.sort(key=lambda t: t[0])
    jss.sort(key=lambda t: t[0])
    return htmls, jss


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


def run(root, files: list[str] | None = None) -> list[Finding]:
    """Every contract finding for the project under `root`, sorted by file
    and line. With `files`, only findings whose own file OR whose contract
    peer is among them -- a post-edit pass cares about both sides of a
    broken contract, and nothing else."""
    root = Path(root)
    htmls, jss = _collect(root)
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
