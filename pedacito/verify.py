"""Deterministic answer verification, a requirement ledger, and candidate scoring.

    findings = check(agent, task, reply, results, gathered)
    errors = [f for f in findings if f.severity == "error"]

The dominant failure of a small model is not bad syntax (the edit path already
refuses that) but a plausible edit to the wrong place: code it never read, a
renamed function whose callers were left behind, a `TODO` standing in for the
work, a requirement from the task's second half that quietly vanished. Every
check here is free -- no model call -- and runs on the DRY-RUN result of the
edit blocks, before anything is written:

  placeholder   the edit introduces `TODO` / `NotImplementedError` / a bare
                `pass` body the task did not ask for
  unread        the edit changes a file (or a region of it) the model never
                read: it is guessing, not editing
  stale refs    a definition was removed or renamed but is still referenced
  moved import  a moved file is still imported by its old path
  citations     the prose cites `file:line` in files that were never read
  coverage      requirements extracted from the task that no edit or sentence
                of the reply touches (a warning: heuristic); each requirement
                also carries an evidence state (UNLOCATED -> LOCATED -> READ
                -> EDITED -> VERIFIED, or READ -> REPORTED for investigations)

Sprint 3 makes that coverage ledger a run-scoped state machine
(`RequirementTracker`): search results, displayed reads and clean patches move
each clause forward, and the agent uses the oldest unmet clause to pick its
next deterministic lookup. No transition costs a model call.

Errors feed one bounded repair round; warnings are shown and logged. A model
verifier (`verify_answer = "full"`) is one extra call, used only when the
deterministic pass found nothing to fix.

Sprint 2 adds the two hooks the harness uses to hand source back WITHOUT ever
rewriting history (TECHNIQUES.md section 11): `find_needs` parses the
explicit `NEED path/to/file.py:120-168` directive from an answer, repair or
verifier reply, and `evidence_rehydrations` selects the unread / fuzzy
findings whose flagged region was recorded in the agent's EvidenceStore (or
sits within ±30 lines of a recorded span) so the repair prompt can append
those exact lines from disk at its tail.
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass, field

from . import fileio
from .edits import locate, parse_edits


@dataclass
class Finding:
    kind: str                  # placeholder | unread | stale | import | citation | coverage | model
    severity: str              # "error" (fed to repair) | "warn"
    file: str
    message: str
    line: int = 0
    # The 1-based inclusive (start, end) region the finding flags, when the
    # check knows one (unread regions, fuzzy-anchored blocks). Sprint 2B's
    # repair auto-rehydration re-fetches exactly these lines.
    lines: tuple = ()

    def render(self) -> str:
        loc = f"{self.file}:{self.line}" if self.line else self.file
        return f"[{self.severity}] {self.kind}: {loc} -- {self.message}"


# ------------------------------------------------------- immediate syntax gate
# A candidate patch is parsed in memory BEFORE any linter or test subprocess is
# started. Python uses `ast.parse`; every other language chunker_ts knows uses
# its tree-sitter grammar. The check compares the dry-applied `after` text with
# the original `before` text, so a file that was already broken stays editable
# (the edit path's recovery case) while a NEW syntax error rejects the patch in
# microseconds. The repair prompt gets the parser diagnostic plus a single
# 10-line window, never the whole file or a subprocess transcript.
_SYNTAX_WINDOW_LINES = 10


def _file_suffix(rel: str) -> str:
    """Lowercase extension of a path without importing Path for every check."""
    name = str(rel or "").replace("\\", "/").rsplit("/", 1)[-1]
    return ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""


def _syntax_problem(rel: str, text: str) -> tuple[bool, int | None, str]:
    """(has_error, 1-based line, parser diagnostic) for `text` as `rel`'s
    language. Unsupported extensions and unavailable parsers are not errors."""
    suffix = _file_suffix(rel)
    body = text if isinstance(text, str) else ""
    if suffix == ".py":
        try:
            ast.parse(body)
        except SyntaxError as e:
            line = int(e.lineno or 0) or None
            col = int(e.offset or 0)
            return (True, line,
                    f"ast.parse: {e.msg} (line {line or '?'}"
                    + (f", column {col}" if col else "") + ")")
        except (ValueError, TypeError) as e:      # null bytes / bad source type
            return True, None, f"ast.parse: {type(e).__name__}: {e}"
        return False, None, ""

    try:
        from .chunker_ts import LANG_BY_EXT, TS_AVAILABLE, has_syntax_error
    except Exception:                              # pragma: no cover - import guard
        return False, None, ""
    language = LANG_BY_EXT.get(suffix)
    if not language or not TS_AVAILABLE:
        return False, None, ""
    try:
        bad, line = has_syntax_error(body, language)
    except Exception:                              # unsupported/broken grammar
        return False, None, ""
    if not bad:
        return False, None, ""
    line = int(line or 0) or None
    return (True, line,
            f"tree-sitter {language}: ERROR node near line {line or '?'}")


def _syntax_window(text: str, line: int | None,
                   max_lines: int = _SYNTAX_WINDOW_LINES
                   ) -> tuple[str, int, int]:
    """(numbered window, first line, last line) around `line`; at most
    `max_lines` source lines, with the error line marked by `>>`."""
    lines = str(text or "").split("\n")
    if lines and lines[-1] == "" and len(lines) > 1:
        lines.pop()
    if not lines:
        return "", 0, 0
    err = max(1, min(int(line or 1), len(lines)))
    before = max(1, max_lines // 2)
    start = max(1, err - before)
    end = min(len(lines), start + max_lines - 1)
    start = max(1, end - max_lines + 1)
    shown = []
    for i in range(start, end + 1):
        mark = ">>" if i == err else "  "
        shown.append(f"{mark}{i:>4}| {lines[i - 1]}")
    return "\n".join(shown), start, end


def check_syntax(results) -> list[Finding]:
    """New syntax errors in dry-applied results, checked in-process.

    `Result.before`/`Result.after` are the in-memory buffers before and after a
    file's blocks. A parse failure that already existed in `before` is not a
    new failure (repairing a broken file must stay possible); only a transition
    from parseable to unparseable rejects the patch. Each finding carries the
    parser diagnostic and exactly the error's 10-line window, so callers can
    feed repair without running a linter or test subprocess first."""
    out: list[Finding] = []
    for r in results or ():
        if not getattr(r, "ok", False) or not getattr(r, "file", ""):
            continue
        after = getattr(r, "after", None)
        if after is None:
            continue
        before = getattr(r, "before", None)
        if before is None:
            before = ""
        was_bad, _, _ = _syntax_problem(r.file, before)
        if was_bad:
            continue
        is_bad, line, diagnostic = _syntax_problem(r.file, after)
        if not is_bad:
            continue
        window, start, end = _syntax_window(after, line)
        message = (
            f"this edit introduces a syntax error at {r.file}:"
            f"{line or '?'}; {diagnostic}. No lint or test process was run. "
            f"Fix the syntax first; only the {_SYNTAX_WINDOW_LINES}-line "
            f"parser window is shown:\n"
            f"lines {start}-{end}:\n{window}")
        out.append(Finding("syntax", "error", r.file, message,
                           line=line or 0, lines=(start, end)))
    return out


# Readable alias for callers that describe the result rather than the check.
syntax_findings = check_syntax


# ------------------------------------------------------------ placeholders
_PLACEHOLDER_RES = (
    re.compile(r"\braise\s+NotImplementedError\b"),
    re.compile(r"\bthrow\s+new\s+(?:NotImplemented|UnsupportedOperation)\w*"),
    re.compile(r"\btodo!\s*\(|\bunimplemented!\s*\("),
    re.compile(r"(?i)(?:#|//|/\*|--)\s*(?:TODO|FIXME|implement me|to be implemented)\b"),
    re.compile(r"(?m)^\s*\.\.\.\s*$"),
    re.compile(r"(?m)^\s*pass\s*(?:#.*)?$"),
)
_STUB_OK_RE = re.compile(
    r"(?i)\b(stub|placeholder|skeleton|scaffold|todo|not implemented|"
    r"unimplemented|abstract|interface|protocol)\b")


def _count(text: str) -> int:
    return sum(len(rx.findall(text or "")) for rx in _PLACEHOLDER_RES)


def check_placeholders(task: str, edits) -> list[Finding]:
    if _STUB_OK_RE.search(task or ""):
        return []
    out: list[Finding] = []
    for e in edits:
        added = _count(e.replace) - _count(e.search)
        if added > 0 and e.replace.strip():
            out.append(Finding(
                "placeholder", "error", e.file,
                f"the edit adds {added} placeholder(s) (TODO / "
                "NotImplementedError / bare `pass` / `...`) instead of "
                "working code; the task did not ask for a stub"))
    return out


# --------------------------------------------------------------- unread edits
def _ledger(agent) -> dict:
    return getattr(agent, "_read_spans", None) or {}


def check_unread(agent, edits, results) -> list[Finding]:
    """Edits to files (or line regions) the gather phase never read."""
    spans = _ledger(agent)
    if not spans and not getattr(agent, "read_any", False):
        # nothing was read at all: the answer-phase refusal logic owns that
        return []
    # A file this run CREATED during lookup (a `write` step) was never
    # "read", but the model authored every byte of it, so a later edit block
    # for it is not an unread guess. The run's backup dir records creations.
    created: set[str] = set()
    bdir = getattr(getattr(agent, "cfg", None), "_backup_dir", None)
    if bdir:
        try:
            created = {fileio.path_key(x) for x in
                       fileio.created_manifest(bdir).read_text(
                           encoding="utf-8").splitlines() if x.strip()}
        except OSError:
            created = set()
    out: list[Finding] = []
    root = agent.index.root
    seen_files: set[str] = set()
    for e in edits:
        rel = fileio.normalise_rel(e.file)
        key = fileio.path_key(rel)
        path = root / rel
        if not path.is_file():
            continue                      # a creation: nothing to have read
        if key in created:
            continue                      # created by this run's own write
        read = spans.get(key) or next(
            (v for k, v in spans.items() if k.endswith("/" + key)
             or key.endswith("/" + k)), None)
        if not read:
            if key not in seen_files:
                out.append(Finding(
                    "unread", "error", rel,
                    "this file was never `read` in the lookup phase, so the "
                    "edit's SEARCH text cannot have come from the file"))
                seen_files.add(key)
            continue
        # region check for quoted (non-anchored) edits
        if e.lines:
            a, b = e.lines
        elif e.search.strip():
            try:
                text, _style = fileio.read(path, errors="replace")
            except OSError:
                continue
            span = locate(text, e.search, agent.cfg)
            if span is None:
                continue
            a = text.count("\n", 0, span.start) + 1
            b = text.count("\n", 0, max(span.start, span.stop)) + 1
        else:
            continue
        covered = any(s <= a and b <= t for s, t in _merge(read))
        if not covered:
            overlap = any(not (t < a or s > b) for s, t in read)
            out.append(Finding(
                "unread", "warn" if overlap else "error", rel,
                f"lines {a}-{b} were not (fully) among the lines read "
                f"({_fmt(_merge(read))}); the edit may be a guess",
                line=a, lines=(a, b)))
    return out


def _merge(spans) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _fmt(spans) -> str:
    return ", ".join(f"{a}-{b}" for a, b in spans[:4]) or "none"


# ------------------------------------------------------------ stale references
_DEF_RES = (
    re.compile(r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)"),
    re.compile(r"(?m)^\s*(?:export\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"),
    re.compile(r"(?m)^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
               r"(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*=>"),
    re.compile(r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)"),
    re.compile(r"(?m)^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"),
    re.compile(r"(?m)^\s*(?:class|struct|interface|trait|enum)\s+([A-Za-z_]\w*)"),
)
_IGNORE_NAMES = {"main", "init", "new", "test", "setup", "run", "get", "set",
                 "self", "cls", "constructor", "toString", "__init__"}


def _defs(text: str) -> set[str]:
    names: set[str] = set()
    for rx in _DEF_RES:
        names.update(rx.findall(text or ""))
    return {n for n in names if len(n) > 2 and n not in _IGNORE_NAMES
            and not (n.startswith("__") and n.endswith("__"))}


def check_stale_references(agent, results) -> list[Finding]:
    """Definitions present before an edit and gone after it that other code
    still references (callers, importers, same-file uses)."""
    out: list[Finding] = []
    index = agent.index
    # what each file will contain once ALL of this reply's edits land: a
    # correct multi-file rename updates the callers in the same answer
    after_by_file = {r.file: (r.after or "") for r in results
                     if r.ok and r.file and r.after is not None}
    for r in results:
        if not r.ok or not r.file:
            continue
        before, after = r.before or "", r.after or ""
        if not before:
            continue
        gone = sorted(_defs(before) - _defs(after))
        if not gone:
            continue
        # files that could still use them: the file itself + importers + callers
        candidates = {r.file}
        try:
            candidates |= set(index.graph.dependents(r.file))
        except Exception:
            pass
        for name in gone[:6]:
            rx = re.compile(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])")
            # moved elsewhere and imported back in: not a loss
            if re.search(r"(?m)^\s*(?:from\s+\S+\s+import[^\n]*|import\s+[^\n]*|"
                         r"(?:const|let|var)[^\n]*require\([^\n]*|use\s+[^\n]*|"
                         r"import\s*\{[^\n]*)(?<![\w$])" + re.escape(name)
                         + r"(?![\w$])", after):
                continue
            hits: list[tuple[str, int]] = []
            for i, line in enumerate((after or "").splitlines(), 1):
                if rx.search(line):
                    hits.append((r.file, i))
                    break
            for f in sorted(candidates):
                if f == r.file:
                    continue
                if f in after_by_file:
                    lines = after_by_file[f].splitlines()
                else:
                    try:
                        lines = fileio.read_lines_cached(index.root / f)
                    except OSError:
                        continue
                for i, line in enumerate(lines, 1):
                    if rx.search(line):
                        hits.append((f, i))
                        break
            if hits:
                shown = ", ".join(f"{f}:{n}" for f, n in hits[:3])
                out.append(Finding(
                    "stale", "error", r.file,
                    f"`{name}` is defined in the original but not in the "
                    f"edited file, and is still referenced at {shown}; "
                    "update those references in the same answer (or keep "
                    "the definition)"))
    return out


def check_moved_imports(agent, results) -> list[Finding]:
    out: list[Finding] = []
    for r in results:
        for old in getattr(r, "renames_from", ()) or ():
            try:
                users = [f for f in agent.index.graph.importers_of(old)
                         if f != r.file]
            except Exception:
                users = []
            if users:
                out.append(Finding(
                    "import", "error", old,
                    f"moved to {r.file}, but still imported by "
                    f"{', '.join(users[:4])}; update those imports in the "
                    "same answer"))
    return out


# ------------------------------------------------------------------ citations
_CITE_RES = (
    re.compile(r"(?<![\w/.-])([\w./\\-]+\.[A-Za-z]{1,5}):(\d+)"),
    re.compile(r"(?<![\w/.-])([\w./\\-]+\.[A-Za-z]{1,5})\s+(?:lines?|L)\s*(\d+)",
               re.IGNORECASE),
)
_BLOCK_STRIP_RE = re.compile(r"<{5,}\s*SEARCH.*?>{5,}\s*REPLACE", re.DOTALL)


def _prose(reply: str) -> str:
    return _BLOCK_STRIP_RE.sub("", reply or "")


def check_citations(agent, reply: str) -> list[Finding]:
    """`file:line` claims in the reply about files that were never read (or
    lines outside what was read): the model is describing code it has not
    seen. An unresolvable path is reported too, rather than silently skipped."""
    spans = _ledger(agent)
    if not spans:
        return []
    out: list[Finding] = []
    seen: set[str] = set()
    for rx in _CITE_RES:
        for m in rx.finditer(_prose(reply)):
            name = m.group(1)
            rel = agent.index.resolve_file(name)
            if not rel:
                if name not in seen:
                    seen.add(name)
                    out.append(Finding(
                        "citation", "warn", name,
                        f"the reply cites {name}:{m.group(2)}, a file that does "
                        "not resolve in the project"))
                continue
            if rel in seen:
                continue
            seen.add(rel)
            key = fileio.path_key(rel)
            read = spans.get(key)
            line = int(m.group(2))
            if not read:
                out.append(Finding(
                    "citation", "warn", rel,
                    f"the reply cites line {line} of a file the lookup phase "
                    "never read", line=line))
            elif not any(s <= line <= t for s, t in read):
                out.append(Finding(
                    "citation", "warn", rel,
                    f"the reply cites line {line}, which was not among the "
                    "lines read", line=line))
    return out[:4]


# ------------------------------------------- shadow-mode prose verification
_SNAKE_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_CAMEL_RE = re.compile(r"\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*\b")


def check_ungrounded_identifiers(agent, reply: str, task: str) -> list[Finding]:
    """Identifier-shaped tokens (snake_case or camelCase) in the prose that
    resolve to nothing in the index and are not named in the task -- a signal
    of an invented symbol. Shadow only: findings are logged, never acted on."""
    index = getattr(agent, "index", None)
    if index is None:
        return []
    task_words = {w.lower() for w in _WORD_RE.findall(task or "")}
    out: list[Finding] = []
    seen: set[str] = set()
    for rx in (_SNAKE_RE, _CAMEL_RE):
        for w in rx.findall(_prose(reply)):
            low = w.lower()
            if low in seen or low in task_words:
                continue
            seen.add(low)
            if not index.find_symbol(w):
                out.append(Finding(
                    "identifier", "warn", "",
                    f"'{w}' looks like a symbol but is neither in the index "
                    "nor named in the task"))
    return out[:4]


def prose_shadow(agent, task: str, reply: str, edits=None) -> list[Finding]:
    """Shadow-mode prose verification (P1a): the checks run on a no-edit
    answer, and every finding is a `warn` that is logged for later labelling --
    never promoted to an error or fed to repair."""
    findings: list[Finding] = []
    try:
        findings += check_citations(agent, reply)
        findings += check_ungrounded_identifiers(agent, reply, task)
        findings += check_coverage(task, reply, edits or [])
    except Exception:
        pass
    for f in findings:
        f.severity = "warn"
    return findings


# ----------------------------------------------------- requirement ledger
_REQ_SPLIT_RE = re.compile(
    r"(?:\n\s*(?:[-*]|\d+[.)])\s+|;\s+|\.\s+(?=[A-Z])|,?\s+and then\s+|"
    r",?\s+and\s+(?=(?:also\s+)?(?:add|make|fix|update|change|remove|rename|"
    r"create|write|ensure|check|handle|support|use|return|move|delete|"
    r"implement|show|print|log)\b)|,?\s+then\s+)", re.IGNORECASE)
_VERBS = {"add", "make", "fix", "update", "change", "remove", "rename", "create",
          "write", "ensure", "check", "handle", "support", "use", "return",
          "move", "delete", "implement", "show", "print", "log", "should",
          "must", "also", "please", "when", "with", "that", "this", "into",
          "from", "have", "make", "need", "the", "and", "for", "all", "any",
          "each", "every", "not", "does", "will", "would", "then", "than",
          "your", "our", "its", "are", "can", "get", "set", "new", "old"}
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")

# Sprint 3: requirement evidence state machine. An edit requirement has to
# travel UNLOCATED -> LOCATED -> READ -> EDITED -> VERIFIED; an investigation /
# explain requirement travels UNLOCATED -> (LOCATED) -> READ -> REPORTED. A
# search/find/outline result supplies LOCATED evidence, a displayed read span
# supplies READ, a clean patch supplies EDITED/VERIFIED, and prose supplies
# REPORTED. The tracker is cheap (no model calls) and monotonic.
REQ_UNLOCATED = "UNLOCATED"
REQ_LOCATED = "LOCATED"
REQ_READ = "READ"
REQ_EDITED = "EDITED"
REQ_VERIFIED = "VERIFIED"
REQ_REPORTED = "REPORTED"

# Short aliases so callers can spell a state the way the plan does.
UNLOCATED = REQ_UNLOCATED
LOCATED = REQ_LOCATED
READ = REQ_READ
EDITED = REQ_EDITED
VERIFIED = REQ_VERIFIED
REPORTED = REQ_REPORTED

_REQ_ORDER = {
    REQ_UNLOCATED: 0, REQ_LOCATED: 1, REQ_READ: 2,
    REQ_EDITED: 3, REQ_REPORTED: 3, REQ_VERIFIED: 4,
}
_REQ_EDIT_RE = re.compile(
    r"\b(?:add|make|fix|update|change|remove|rename|create|write|ensure|"
    r"handle|support|use|return|move|delete|implement|refactor|replace|"
    r"edit|modify|adjust|correct|bump|upgrade|migrate|introduce|apply|"
    r"keep|allow|accept|raise|lower|increase|decrease|reduce|extend|"
    r"prevent|avoid|guard|validate|normalize|normalise|convert|switch|"
    r"wire|hook|register|append|insert|set)\b", re.IGNORECASE)
_REQ_REPORT_RE = re.compile(
    r"\b(?:explain|describe|summari[sz]e|investigate|trace|find|locate|"
    r"identify|report|why|how|what|where|which|who|tell|show|list|answer|"
    r"determine|clarify|review|audit)\b|\?", re.IGNORECASE)
_REQ_LOCATION_RE = re.compile(
    r"(?<![\w/.-])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]{1,8}):(\d+)"
    r"(?:-(\d+))?")


@dataclass
class Requirement:
    """One task clause plus the evidence the harness has gathered for it."""
    text: str
    terms: list[str]
    kind: str = "edit"                 # edit | report
    state: str = REQ_UNLOCATED
    anchors: list[tuple] = field(default_factory=list)    # (file, start, end)
    read_spans: list[tuple] = field(default_factory=list)
    edit_spans: list[tuple] = field(default_factory=list)

    def __post_init__(self):
        if self.kind not in ("edit", "report"):
            self.kind = "edit"
        if self.state not in _REQ_ORDER:
            self.state = REQ_UNLOCATED
        self.terms = list(dict.fromkeys(self.terms))

    @property
    def is_edit(self) -> bool:
        return self.kind == "edit"

    @property
    def final(self) -> bool:
        return self.state == (REQ_VERIFIED if self.is_edit else REQ_REPORTED)

    @property
    def open(self) -> bool:
        return not self.final

    def add_anchor(self, rel: str, start: int, end: int) -> None:
        span = (fileio.normalise_rel(rel or ""), int(start or 1),
                int(end or 0))
        if span[0] and span not in self.anchors:
            self.anchors.append(span)

    def add_read(self, rel: str, start: int, end: int) -> None:
        span = (fileio.normalise_rel(rel or ""), int(start or 1),
                int(end or 0))
        if span[0] and span not in self.read_spans:
            self.read_spans.append(span)

    def add_edit(self, rel: str) -> None:
        rel = fileio.normalise_rel(rel or "")
        if rel:
            span = (rel, 0, 0)
            if span not in self.edit_spans:
                self.edit_spans.append(span)

    def search_term(self) -> str:
        """The most specific term to grep/find when this requirement has no
        located anchor yet (identifiers and file names win over prose)."""
        generic = {
            "explain", "describe", "summarize", "summarise", "investigate",
            "trace", "find", "locate", "identify", "report", "works", "work",
            "logic", "thing", "things", "stuff", "code", "file", "files",
            "update", "change", "make", "fix", "add", "new", "old", "the",
            "and", "for", "with", "that", "this", "into", "from",
        }

        def rank(t: str):
            score = len(t)
            if "_" in t:
                score += 8
            if "." in t or "/" in t:
                score += 6
            if any(c.isdigit() for c in t):
                score += 3
            if t.isupper():
                score += 2
            if t.isdigit():
                score -= 4
            if t.lower() in generic:
                score -= 10
            return score
        terms = [t for t in self.terms if len(t) >= 3]
        return max(terms, key=rank) if terms else ""


def _term_in(term: str, text: str) -> bool:
    """Whether `term` appears as a word (or a simple plural stem) in `text`."""
    term = (term or "").strip().lower()
    if not term or not text:
        return False
    low = text.lower()
    if re.search(r"(?<![\w])" + re.escape(term) + r"(?![\w])", low):
        return True
    stem = term.rstrip("s")
    return stem != term and bool(
        re.search(r"(?<![\w])" + re.escape(stem) + r"(?![\w])", low))


def _match_score(terms, text: str) -> int:
    return sum(1 for t in terms or () if _term_in(t, text))


def _resolve_rel(name: str, index=None) -> str:
    name = (name or "").strip().strip("`'\"")
    if not name:
        return ""
    if index is not None:
        try:
            rel = index.resolve_file(name)
            if rel:
                return rel
        except Exception:
            pass
    return fileio.normalise_rel(name)


def _anchors_from_text(text: str, index=None) -> list[tuple]:
    """Exact `file:line` / `file:start-end` anchors in a tool observation."""
    out: list[tuple] = []
    seen: set[tuple] = set()
    for m in _REQ_LOCATION_RE.finditer(text or ""):
        name = m.group(1)
        if name.lower().startswith("http"):
            continue
        start = int(m.group(2))
        end = int(m.group(3) or start)
        if start <= 0 or end < start:
            continue
        rel = _resolve_rel(name, index)
        if not rel:
            continue
        span = (rel, start, end)
        if span not in seen:
            seen.add(span)
            out.append(span)
        if len(out) >= 40:
            break
    return out


def _clause_kind(clause: str, default: str) -> str:
    if _REQ_EDIT_RE.search(clause or ""):
        return "edit"
    if _REQ_REPORT_RE.search(clause or ""):
        return "report"
    return default


def _requirements_from_task(task: str, limit: int = 5) -> list[Requirement]:
    text = (task or "").split("\nTASK\n")[-1].strip()
    if not text:
        return []
    parts = [p.strip(" \t\n.,;") for p in _REQ_SPLIT_RE.split(text) if p]
    task_kind = "edit" if _REQ_EDIT_RE.search(text) else (
        "report" if _REQ_REPORT_RE.search(text) else "edit")
    reqs: list[Requirement] = []
    for p in parts:
        if len(p) < 8:
            continue
        terms = []
        for w in _WORD_RE.findall(p):
            if w.lower() in _VERBS:
                continue
            terms.append(w)
        terms += re.findall(r"\b\d+\b", p)
        terms += re.findall(r"[\"'`]([^\"'`]{2,40})[\"'`]", p)
        if terms:
            reqs.append(Requirement(
                p[:140], list(dict.fromkeys(terms))[:8],
                _clause_kind(p, task_kind)))
    return reqs[:limit]


def requirement_ledger(task: str, limit: int = 5) -> list[tuple[str, list[str]]]:
    """[(requirement text, key terms)] for up to `limit` clauses of the task.
    Heuristic and free: split on sentence/bullet/`and add ...` boundaries, keep
    each clause's distinctive words (identifiers, file names, numbers,
    quoted text)."""
    return [(r.text, r.terms) for r in _requirements_from_task(task, limit)]


class RequirementTracker:
    """Run-scoped requirement evidence state machine.

    The tracker is deliberately deterministic: every transition comes from a
    tool result or a proposed patch the harness already has, never from another
    model call. It powers both the gather completion gate (oldest unmet
    requirement) and the answer-phase source packer (pinned located spans).
    """

    def __init__(self, task: str = "", limit: int = 5):
        self.task = task or ""
        self.items = _requirements_from_task(self.task, limit)

    # ---- container conveniences
    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    @property
    def requirements(self) -> list[Requirement]:
        return self.items

    def states(self) -> list[tuple[str, str]]:
        return [(r.text, r.state) for r in self.items]

    def unmet(self, states=(REQ_UNLOCATED, REQ_LOCATED)) -> list[Requirement]:
        return [r for r in self.items if r.state in states]

    def next_unmet(self, states=(REQ_UNLOCATED, REQ_LOCATED)):
        found = self.unmet(states)
        return found[0] if found else None

    # Alias kept explicit: the gather gate asks for the OLDEST requirement.
    oldest_unmet = next_unmet

    def _advance(self, req: Requirement, target: str) -> bool:
        """Promote `req` monotonically to `target`, filling in the states a
        stronger piece of evidence implies (a verified patch also proves the
        span was read). Returns True when the state changed."""
        if _REQ_ORDER.get(req.state, 0) >= _REQ_ORDER.get(target, 0):
            return False
        req.state = target
        return True

    def _locate(self, req: Requirement, anchors=(), files=()) -> None:
        for rel, a, b in anchors:
            req.add_anchor(rel, a, b)
        for rel in files:
            req.add_anchor(rel, 1, 0)
        self._advance(req, REQ_LOCATED)

    # ---- evidence transitions
    def observe_search(self, action: str, step: dict | None = None,
                       observation: str = "", index=None) -> list[Requirement]:
        """A grep/find/outline/callers/context result that names a requirement
        term (or a file/line anchor) advances it to LOCATED."""
        step = step or {}
        obs = observation or ""
        if obs.startswith("ERROR"):
            return []
        query = " ".join(str(step.get(k) or "") for k in
                         ("pattern", "symbol", "file", "content", "command"))
        anchors = _anchors_from_text(obs, index)
        changed = []
        for req in self.items:
            if req.state not in (REQ_UNLOCATED, REQ_LOCATED):
                continue
            q_hit = _match_score(req.terms, query)
            o_hit = _match_score(req.terms, obs)
            # A grep only locates when its RESULT mentions the term; find /
            # outline / callers may locate from the query itself because their
            # output is a definition list for exactly that query. Anchors alone
            # are generic to the observation, so they do not locate every
            # requirement in the task.
            evidence = o_hit or (
                q_hit and action in ("find", "outline", "callers", "context"))
            if not evidence:
                continue
            files = []
            for key in ("file", "pattern", "symbol"):
                val = step.get(key)
                if not isinstance(val, str) or not _match_score(req.terms, val):
                    continue
                rel = _resolve_rel(val, index)
                if rel and "." in val:
                    files.append(rel)
            self._locate(req, anchors, files)
            changed.append(req)
        return changed

    def observe_read(self, file: str, start: int, end: int, text: str = "",
                     index=None) -> list[Requirement]:
        """A displayed read/context span that touches requirement terms or a
        previously located anchor advances the requirement to READ."""
        rel = _resolve_rel(file, index)
        if not rel or (text or "").startswith("ERROR"):
            return []
        changed = []
        for req in self.items:
            if req.final:
                continue
            hit = _match_score(req.terms, rel + "\n" + (text or ""))
            overlap = any(
                _same_file(rel, a_rel) and _spans_overlap(start, end, a, b)
                for a_rel, a, b in req.anchors)
            if not (hit or overlap):
                continue
            req.add_read(rel, start, end)
            if req.state != REQ_READ and self._advance(req, REQ_READ):
                changed.append(req)
        return changed

    def observe_edit(self, edits=None, *, file: str = "", search: str = "",
                     replace: str = "", ok: bool = True) -> list[Requirement]:
        """A clean patch whose file/text touches a requirement advances it to
        EDITED (passing through the earlier states, which the patch proves)."""
        if edits is None:
            edits = [{"file": file, "search": search,
                      "replace": replace, "ok": ok}]
        elif isinstance(edits, (str, dict)) or hasattr(edits, "file"):
            edits = [edits]
        changed = []
        for e in edits or ():
            if isinstance(e, str):
                e = {"file": e, "search": "", "replace": "", "ok": True}
            if isinstance(e, dict):
                rel = e.get("file", "")
                before = e.get("search", "")
                after = e.get("replace", "")
                good = e.get("ok", True)
            else:
                rel = getattr(e, "file", "")
                before = getattr(e, "search", "") or getattr(e, "before", "")
                after = getattr(e, "replace", "") or getattr(e, "after", "")
                good = getattr(e, "ok", True)
            if not good or not rel:
                continue
            rel = _resolve_rel(rel)
            text = f"{rel}\n{before}\n{after}"
            for req in self.items:
                if not req.is_edit or req.final:
                    continue
                hit = _match_score(req.terms, text)
                overlap = any(
                    _same_file(rel, a_rel)
                    for a_rel, _a, _b in req.anchors + req.read_spans)
                if not (hit or overlap):
                    continue
                req.add_edit(rel)
                if self._advance(req, REQ_EDITED):
                    changed.append(req)
        return changed

    def observe_patch(self, file: str, text: str = "", ok: bool = True
                      ) -> list[Requirement]:
        """A `write`/`append`/`move` tool result is an edit too."""
        return self.observe_edit(file=file, replace=text, ok=ok)

    def observe_verified(self, files=None) -> list[Requirement]:
        """Promote EDITED edit requirements to VERIFIED after the checks and
        tests accept the patch. `files` narrows it to the touched files."""
        changed = []
        wanted = {_resolve_rel(f) for f in files} if files else None
        for req in self.items:
            if not req.is_edit or req.state != REQ_EDITED:
                continue
            if wanted is not None and not any(
                    _same_file(f, rel) for rel in wanted
                    for f, _a, _b in req.edit_spans):
                continue
            if self._advance(req, REQ_VERIFIED):
                changed.append(req)
        return changed

    def observe_report(self, reply: str) -> list[Requirement]:
        """Prose that answers a READ investigation requirement reports it."""
        changed = []
        for req in self.items:
            if req.kind != "report" or req.state != REQ_READ:
                continue
            if _match_score(req.terms, _prose(reply)):
                self._advance(req, REQ_REPORTED)
                changed.append(req)
        return changed

    def observe_answer(self, reply: str, edits=None, results=None
                       ) -> list[Requirement]:
        changed = []
        if results is not None:
            changed += self.observe_edit(results)
        elif edits is not None:
            changed += self.observe_edit(edits)
        changed += self.observe_report(reply)
        return changed

    # ---- gather completion / packing helpers
    def next_action(self, index=None) -> tuple[dict | None, Requirement | None]:
        """The cheapest deterministic action for the oldest requirement that
        is still UNLOCATED or LOCATED, or (None, None). A located anchor is
        read directly; otherwise a specific term is searched."""
        for req in self.items:
            if req.state not in (REQ_UNLOCATED, REQ_LOCATED):
                continue
            if req.state == REQ_LOCATED:
                for rel, a, b in req.anchors:
                    if any(_same_file(rel, r) and _spans_overlap(a, b or a, x, y)
                           for r, x, y in req.read_spans):
                        continue
                    start, end = max(1, a), max(0, b)
                    # A grep hit names a line; when it lands inside an indexed
                    # definition, read that whole definition so the eventual
                    # SEARCH block can quote real body lines.
                    if index is not None:
                        try:
                            units = [c for c in index.in_file(rel)
                                     if c.kind not in ("module", "reference")
                                     and c.start <= start <= c.end]
                            if units:
                                unit = min(units, key=lambda c: c.nlines)
                                start = max(1, unit.start)
                                end = max(start, unit.scope_end or unit.end)
                        except Exception:
                            pass
                    return {"action": "read", "file": rel,
                            "start": start, "end": end}, req
            term = req.search_term()
            if term:
                return {"action": "grep", "pattern": term}, req
        return None, None

    # Explicit alias for callers that describe the choice, not the ordering.
    cheapest_action = next_action

    def prompt(self, req: Requirement | None = None) -> str:
        """A specific harness nudge naming the oldest unmet requirement."""
        req = req or self.next_unmet()
        if req is None:
            return ""
        if req.state == REQ_LOCATED and req.anchors:
            rel, a, b = req.anchors[0]
            span = f"{rel}:{a}-{b}" if b else f"{rel}:{a}"
            return (f"[Harness: requirement still {req.state}: '{req.text}'. "
                    f"You located it at {span} but have not read it; `read` "
                    f"that exact span before answering.]")
        term = req.search_term()
        how = f"`grep`/`find` for '{term}'" if term else "locate the code"
        return (f"[Harness: requirement still {req.state}: '{req.text}'. "
                f"Use {how} and then `read` the matching span before "
                f"answering.]")

    def pinned_spans(self) -> list[tuple]:
        """Exact spans the answer packer must never evict: located anchors and
        displayed reads for every requirement that got past UNLOCATED."""
        out: list[tuple] = []
        for req in self.items:
            if req.state == REQ_UNLOCATED:
                continue
            for span in req.anchors + req.read_spans:
                if span not in out:
                    out.append(span)
        return out

    def packing_terms(self) -> list[tuple[list[str], int]]:
        """(terms, weight) for the source packer; unread requirements weigh
        most because their evidence is what the answer still needs."""
        out = []
        for req in self.items:
            if req.final:
                continue
            weight = 3 if req.state in (REQ_UNLOCATED, REQ_LOCATED) else (
                2 if req.state == REQ_READ else 1)
            out.append((req.terms, weight))
        return out


def requirement_tracker(task: str, limit: int = 5) -> RequirementTracker:
    """Convenience constructor for the run-scoped state machine."""
    return RequirementTracker(task, limit)


def _same_file(a: str, b: str) -> bool:
    a, b = fileio.path_key(a or ""), fileio.path_key(b or "")
    return bool(a and b and (a == b or a.endswith("/" + b)
                             or b.endswith("/" + a)))


def _spans_overlap(a1: int, b1: int, a2: int, b2: int) -> bool:
    """Overlap for 1-based inclusive spans; an end of 0 means whole file."""
    if not a1 or not a2:
        return True
    b1 = b1 or 10 ** 9
    b2 = b2 or 10 ** 9
    return not (b1 < a2 or b2 < a1)


def check_coverage(task: str, reply: str, edits,
                   tracker: RequirementTracker | None = None) -> list[Finding]:
    """Requirements whose key terms appear nowhere in the edits, their file
    names, or the reply's prose. Runs for prose-only replies too (P1a), not
    only when edit blocks are present. When a live RequirementTracker is
    supplied, its states are authoritative for what counts as addressed."""
    if tracker is None or tracker.task != task:
        tracker = RequirementTracker(task)
    reqs = tracker.items
    if len(reqs) < 2:
        return []
    tracker.observe_edit(edits)
    tracker.observe_report(reply)
    haystack = "\n".join([_prose(reply)] + [f"{e.file}\n{e.replace}\n{e.search}"
                                            for e in edits]).lower()
    out = []
    for req in reqs:
        text, terms = req.text, req.terms
        hit = sum(1 for t in terms if t.lower() in haystack
                  or t.lower().rstrip("s") in haystack)
        if "test" in text.lower() and any("test" in e.file.lower() for e in edits):
            continue
        if req.state in (REQ_EDITED, REQ_VERIFIED, REQ_REPORTED):
            continue
        if hit == 0:
            out.append(Finding(
                "coverage", "warn", "", f"no edit or sentence addresses: "
                f"'{text}' (still {req.state})"))
    return out[:3]



def _fuzzy_regions(before: str, after: str) -> list[tuple[int, int]]:
    """1-based inclusive line ranges of `before` that `after` changes: where
    the approximately-anchored blocks landed, computed free with difflib so a
    fuzzy finding carries the region rehydration should re-fetch."""
    if not before or not after or before == after:
        return []
    sm = difflib.SequenceMatcher(
        None, before.splitlines(), after.splitlines(), autojunk=False)
    out: list[tuple[int, int]] = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a = i1 + 1
        out.append((a, max(i2, a)))
        if len(out) >= 4:
            break
    return out


def check_fuzzy(results) -> list[Finding]:
    """A block anchored approximately (fuzzy_match) is a warning, not an
    error: it landed somewhere the SEARCH text did not exactly match, so the
    diff needs a human's eye before it is trusted. The finding carries the
    region it anchored on (Sprint 2B: near recorded evidence, the repair
    re-fetches exactly those lines)."""
    out: list[Finding] = []
    for r in (results or []):
        if not getattr(r, "fuzzy", False):
            continue
        a = b = 0
        try:
            regions = _fuzzy_regions(getattr(r, "before", ""),
                                     getattr(r, "after", ""))
            if regions:
                a, b = regions[0]
        except Exception:
            a = b = 0
        out.append(Finding(
            "fuzzy", "warn", r.file,
            "one or more blocks matched approximately (not exactly); "
            "review the diff carefully", line=a,
            lines=(a, b) if a else ()))
    return out


# --------------------------------------------- Sprint 2: exact-span rehydration
# `NEED shop/pricing.py:120-168` -- the model asks the harness for exactly
# the lines it needs, and the harness APPENDS them at the conversation tail
# so the prompt-prefix cache survives the round (TECHNIQUES.md #2). A plain
# `file:line` citation is not a request: the path needs an extension and a
# `:` or space before the line numbers.
_NEED_RE = re.compile(
    r"(?im)^[ \t]*(?:[-*]|\d+[.)])?[ \t]*NEED[ \t]+[`'\"]?[ \t]*"
    r"([\w./\\-]+\.[A-Za-z][A-Za-z0-9]{0,7})"
    r"(?:[ \t]*:[ \t]*|[ \t]+)"
    r"(\d+)(?:[ \t]*(?:-|\u2013|[ \t]to)[ \t]*(\d+))?[ \t]*[`'\"]?[ \t]*$")


def find_needs(reply: str) -> list[tuple[str, int, int]]:
    """`(file, start, end)` for each `NEED path:120-168` directive line in a
    reply (deduplicated, order preserved, capped). A single line number asks
    for one line; a reversed range is swapped. Anything without an extension
    and a number is prose about needing something, not a directive."""
    out: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for m in _NEED_RE.finditer(reply or ""):
        name = m.group(1).strip()
        a = int(m.group(2))
        b = int(m.group(3)) if m.group(3) else a
        if a > b:
            a, b = b, a
        key = (name, a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= 8:
            break
    return out


_MAX_REHYDRATE_SPANS = 4


def _finding_region(f) -> tuple[int, int]:
    """The line region a finding flags: its `lines` pair, else its single
    `line`, else (0, 0) when the check knows no position."""
    if len(getattr(f, "lines", ()) or ()) == 2:
        return int(f.lines[0]), int(f.lines[1])
    if getattr(f, "line", 0):
        a = int(f.line)
        return a, a
    return 0, 0


def evidence_rehydrations(agent, findings) -> list[tuple[str, int, int]]:
    """`(rel, start, end)` spans the repair prompt should re-fetch from disk.

    Only `unread` and `fuzzy` findings qualify, and only when the flagged
    region is covered by, or within the evidence window (±30 lines) of, a
    span the EvidenceStore recorded as DISPLAYED to the model: the model
    misquoted source it saw, so handing back its fresh exact bytes fixes the
    patch for free. An unread region far from any recorded span is source it
    never saw at all -- rehydrating there would dump an unseen file on the
    model (against token economy #1) instead of grounding a correction."""
    store = getattr(agent, "evidence", None)
    if store is None or not findings:
        return []
    out: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for f in findings:
        if getattr(f, "kind", "") not in ("unread", "fuzzy") or not f.file:
            continue
        a, b = _finding_region(f)
        if a <= 0:
            continue
        try:
            rel = agent.index.resolve_file(f.file) or fileio.normalise_rel(f.file)
        except Exception:
            rel = fileio.normalise_rel(f.file)
        if not rel or not store.nearby(rel, a, b):
            continue
        key = (rel, a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= _MAX_REHYDRATE_SPANS:
            break
    return out


# --------------------------------------------------------------- entry points
def check(agent, task: str, reply: str, results, gathered: str = ""
          ) -> list[Finding]:
    """All deterministic findings for a proposed reply (never raises)."""
    findings: list[Finding] = []
    try:
        edits = parse_edits(reply)
        findings += check_syntax(results)
        findings += check_placeholders(task, edits)
        findings += check_unread(agent, edits, results)
        findings += check_stale_references(agent, results)
        findings += check_moved_imports(agent, results)
        findings += check_citations(agent, reply)
        tracker = getattr(agent, "requirements", None)
        if tracker is not None and getattr(tracker, "task", "") != task:
            tracker = None
        findings += check_coverage(task, reply, edits, tracker=tracker)
        findings += check_fuzzy(results)
    except Exception as e:                  # a checker bug must not block a task
        findings.append(Finding("model", "warn", "",
                                f"verification skipped ({type(e).__name__})"))
    return findings


def render(findings, severity: str | None = None) -> str:
    return "\n".join(f.render() for f in findings
                     if severity is None or f.severity == severity)


def score_candidate(agent, task: str, reply: str, results, gathered: str = ""
                    ) -> tuple[float, list[Finding]]:
    """A comparable quality score for one candidate edit reply (higher is
    better): blocks that parse and dry-apply cleanly count for it; failed
    blocks, placeholder/unread/stale errors and unaddressed requirements
    count against it."""
    edits = parse_edits(reply)
    findings = check(agent, task, reply, results, gathered)
    if not edits:
        return (0.0 if reply.strip() else -5.0), findings
    ok = sum(1 for r in results if r.ok)
    bad = sum(1 for r in results if not r.ok)
    score = 10.0 + 4.0 * ok - 6.0 * bad
    score -= 8.0 * sum(1 for f in findings if f.severity == "error")
    score -= 2.0 * sum(1 for f in findings if f.severity == "warn")
    return score, findings


VERIFIER_SYSTEM = """\
You are a strict code reviewer. You are given a TASK, the SOURCE the author
read, and the author's proposed search/replace edits. Judge the edits ONLY
against that source.

Check: (1) every SEARCH region really appears in the SOURCE; (2) the edit
does what the TASK says -- including its LAST requirement; (3) nothing the
edit removes or renames is still used elsewhere in the SOURCE; (4) no claim
about code that is not in the SOURCE.

Reply with either the single word PASS, or one line per concrete defect in
exactly this form (at most 5 lines, no preamble):
DEFECT: <file>:<line> -- <what is wrong and what it should be>

If the SOURCE is missing a region you must inspect to judge it, you may put
`NEED <file>:<start>-<end>` lines (at most two) before your verdict: the
harness fetches those exact lines and asks you again."""

_DEFECT_RE = re.compile(r"(?im)^\s*DEFECT\s*:\s*(?P<where>[^\s]+)\s*(?:--|-|:)\s*(?P<what>.+)$")


def parse_verdict(text: str, resolve=None, root=None) -> list[Finding]:
    """Defects from a verifier reply ([] for PASS or anything unparseable).

    `resolve` is a file-name -> indexed-path resolver (``index.resolve_file``);
    `root` is the project directory. With both, a DEFECT is kept as an `error`
    only when the file resolves and the cited line is inside it; otherwise it
    is a `warn` -- a verifier hallucinating a path or a line number must not
    feed the repair re-ask. Without `resolve`, every defect is an error (the
    caller has no index to check against)."""
    out = []
    for m in _DEFECT_RE.finditer(text or ""):
        where = m.group("where")
        f, _, line = where.partition(":")
        name = f.strip("`")
        line_no = int(line) if line.isdigit() else 0
        severity = "error"
        file = name
        if resolve is not None:
            rel = resolve(name)
            if rel is None or line_no <= 0:
                severity = "warn"
            else:
                file = rel
                if root is not None:
                    try:
                        if line_no > len(fileio.read_lines_cached(root / rel)):
                            severity = "warn"
                    except OSError:
                        severity = "warn"
        out.append(Finding("model", severity, file,
                           m.group("what").strip()[:240], line_no))
    return out[:5]
