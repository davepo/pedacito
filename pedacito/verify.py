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
                of the reply touches (a warning: heuristic)

Errors feed one bounded repair round; warnings are shown and logged. A model
verifier (`verify_answer = "full"`) is one extra call, used only when the
deterministic pass found nothing to fix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import fileio
from .edits import locate, parse_edits


@dataclass
class Finding:
    kind: str                  # placeholder | unread | stale | import | citation | coverage | model
    severity: str              # "error" (fed to repair) | "warn"
    file: str
    message: str
    line: int = 0

    def render(self) -> str:
        loc = f"{self.file}:{self.line}" if self.line else self.file
        return f"[{self.severity}] {self.kind}: {loc} -- {self.message}"


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
                f"({_fmt(_merge(read))}); the edit may be a guess", line=a))
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


def requirement_ledger(task: str, limit: int = 5) -> list[tuple[str, list[str]]]:
    """[(requirement text, key terms)] for up to `limit` clauses of the task.
    Heuristic and free: split on sentence/bullet/`and add ...` boundaries, keep
    each clause's distinctive words (identifiers, file names, numbers,
    quoted text)."""
    text = (task or "").split("\nTASK\n")[-1].strip()
    if not text:
        return []
    parts = [p.strip(" \t\n.,;") for p in _REQ_SPLIT_RE.split(text) if p]
    reqs: list[tuple[str, list[str]]] = []
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
            reqs.append((p[:140], list(dict.fromkeys(terms))[:8]))
    return reqs[:limit]


def check_coverage(task: str, reply: str, edits) -> list[Finding]:
    """Requirements whose key terms appear nowhere in the edits, their file
    names, or the reply's prose. Runs for prose-only replies too (P1a), not
    only when edit blocks are present."""
    reqs = requirement_ledger(task)
    if len(reqs) < 2:
        return []
    haystack = "\n".join([_prose(reply)] + [f"{e.file}\n{e.replace}\n{e.search}"
                                            for e in edits]).lower()
    out = []
    for text, terms in reqs:
        hit = sum(1 for t in terms if t.lower() in haystack
                  or t.lower().rstrip("s") in haystack)
        if "test" in text.lower() and any("test" in e.file.lower() for e in edits):
            continue
        if hit == 0:
            out.append(Finding(
                "coverage", "warn", "", f"no edit or sentence addresses: "
                f"'{text}'"))
    return out[:3]


def check_fuzzy(results) -> list[Finding]:
    """A block anchored approximately (fuzzy_match) is a warning, not an
    error: it landed somewhere the SEARCH text did not exactly match, so the
    diff needs a human's eye before it is trusted."""
    return [Finding("fuzzy", "warn", r.file,
                    "one or more blocks matched approximately (not exactly); "
                    "review the diff carefully")
            for r in (results or []) if getattr(r, "fuzzy", False)]


# --------------------------------------------------------------- entry points
def check(agent, task: str, reply: str, results, gathered: str = ""
          ) -> list[Finding]:
    """All deterministic findings for a proposed reply (never raises)."""
    findings: list[Finding] = []
    try:
        edits = parse_edits(reply)
        findings += check_placeholders(task, edits)
        findings += check_unread(agent, edits, results)
        findings += check_stale_references(agent, results)
        findings += check_moved_imports(agent, results)
        findings += check_citations(agent, reply)
        findings += check_coverage(task, reply, edits)
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
DEFECT: <file>:<line> -- <what is wrong and what it should be>"""

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
