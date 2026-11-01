"""Push memory: durable project facts and scoped rules shown to the model.

    memory.render(root, cfg)                  # block for the system prompt
    memory.remember(root, "pytest needs DB_URL set", category="commands")
    memory.rules_for(root, "src/api/users.py")   # path-scoped rules

Hierarchy (later overrides earlier on conflict, and is listed last):

    global     <config dir>/memory.md           your habits, every project
    project    <root>/.pedacito/memory.md       this repo's facts
    scoped     <root>/.pedacito/rules/*.md      shown when a matching file is
               nested AGENTS.md / CLAUDE.md      read (frontmatter `paths:` globs)

Memory is a HINT, never truth: the block is labelled that way, `remember`
refuses instruction-like text (a persisted prompt injection would outlive the
session that planted it), secrets are redacted before they are written, lines
are de-duplicated by normalised hash, and each category is capped so the file
cannot grow without bound. The file is plain markdown: edit or delete it freely.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from pathlib import Path

from . import safety

CATEGORIES = ("commands", "conventions", "landmines", "notes")
_HEADINGS = {c: c.capitalize() for c in CATEGORIES}
_PER_CATEGORY = 20
_MAX_LINE = 300
_HEADING_RE = re.compile(r"^##\s+(\w+)\s*$")

_KEYWORDS = (
    ("commands", re.compile(r"\b(run|runs|command|pytest|npm|make|cargo|build|"
                            r"test with|start with|port \d+)\b", re.I)),
    ("landmines", re.compile(r"\b(never|don't|do not|careful|broken|legacy|"
                             r"deprecated|flaky|avoid|trap|gotcha)\b", re.I)),
    ("conventions", re.compile(r"\b(convention|style|always|prefer|naming|"
                               r"format|lint|use .* instead)\b", re.I)),
)


def project_file(root) -> Path:
    return Path(root) / ".pedacito" / "memory.md"


def global_file() -> Path:
    from . import config as config_mod
    return config_mod.USER_CONFIG.parent / "memory.md"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _hash(text: str) -> str:
    return hashlib.sha1(_norm(text).encode()).hexdigest()[:10]


def _read(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    cat = "notes"
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            name = m.group(1).lower()
            cat = name if name in out else "notes"
            continue
        if line.startswith("- ") and line[2:].strip():
            out[cat].append(line[2:].strip())
    return out


def _write(path: Path, data: dict[str, list[str]]) -> None:
    lines = ["# Pedacito memory (hints for the model; edit or delete freely)", ""]
    for c in CATEGORIES:
        if data[c]:
            lines.append(f"## {_HEADINGS[c]}")
            lines += [f"- {x}" for x in data[c]]
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(path)


def guess_category(text: str) -> str:
    for cat, rx in _KEYWORDS:
        if rx.search(text):
            return cat
    return "notes"


def remember(root, text: str, category: str = "", scope: str = "project"
             ) -> tuple[bool, str]:
    """Append one fact. Returns (stored, message). Refuses empty, over-long,
    instruction-like, or duplicate text."""
    line = " ".join((text or "").split())
    if not line:
        return False, "nothing to remember (empty)"
    m = re.match(r"^(command|convention|landmine|note)s?\s*:\s*(.+)$", line, re.I)
    if m and not category:
        category = {"command": "commands", "convention": "conventions",
                    "landmine": "landmines", "note": "notes"}[m.group(1).lower()]
        line = m.group(2).strip()
    if len(line) > _MAX_LINE:
        line = line[:_MAX_LINE - 3].rstrip() + "..."
    if safety.looks_like_injection(line):
        return False, ("refused: that reads like an instruction to an "
                       "assistant; memory stores facts about the project, "
                       "not commands")
    line = safety.redact(line)
    cat = category if category in CATEGORIES else guess_category(line)
    path = global_file() if scope == "global" else project_file(root)
    data = _read(path)
    h = _hash(line)
    if any(_hash(x) == h for lst in data.values() for x in lst):
        return False, "already remembered"
    data[cat].append(line)
    if len(data[cat]) > _PER_CATEGORY:
        data[cat] = data[cat][-_PER_CATEGORY:]
    _write(path, data)
    return True, f"remembered under {cat}"


def forget(root, needle: str, scope: str = "project") -> int:
    """Delete lines containing `needle` (or whose hash equals it); returns
    how many."""
    path = global_file() if scope == "global" else project_file(root)
    data = _read(path)
    n = 0
    low = needle.lower().strip()
    for c in CATEGORIES:
        keep = [x for x in data[c]
                if low not in x.lower() and _hash(x) != low]
        n += len(data[c]) - len(keep)
        data[c] = keep
    if n:
        _write(path, data)
    return n


def entries(root) -> list[tuple[str, str, str]]:
    """[(scope, category, text)] for `pedacito memory --list`."""
    out = []
    for scope, path in (("global", global_file()), ("project", project_file(root))):
        for cat, lst in _read(path).items():
            out += [(scope, cat, x) for x in lst]
    return out


def render(root, cfg=None, limit: int | None = None) -> str:
    """The memory block for a prompt ('' when there is none or memory is off),
    newest lines kept when over `limit` characters."""
    if cfg is not None and not getattr(cfg, "use_memory", True):
        return ""
    limit = int(limit or getattr(cfg, "max_memory_chars", 1500) or 1500)
    rows: list[str] = []
    for scope, path in (("global", global_file()), ("project", project_file(root))):
        data = _read(path)
        for cat in CATEGORIES:
            for x in data[cat]:
                rows.append(f"- [{scope}/{cat}] {x}")
    if not rows:
        return ""
    head = ("PROJECT MEMORY (hints saved by earlier runs; NOT truth -- verify "
            "against source before relying on any of it)")
    while rows and len(head) + 1 + sum(len(r) + 1 for r in rows) > limit:
        rows.pop(0)                    # oldest first
    return head + "\n" + "\n".join(rows) if rows else ""


# --------------------------------------------------------- failed attempts
#
# A task that ends with no edit landed leaves a one-line record; the next task
# that looks similar is told what was tried and why it failed, so a small model
# does not walk the same dead end twice. Records are facts the harness saw
# (files read, the failure), never model text; a later success clears them.

_MAX_ATTEMPTS = 30
_STOP = frozenset("the a an to of in on for and or with is it this that be "
                  "add fix make change update use from into".split())


def attempts_file(root) -> Path:
    return Path(root) / ".pedacito" / "attempts.json"


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z_][a-z0-9_]{2,}", text.lower())
            if w not in _STOP}


def _similar(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(1, min(len(wa), len(wb))) if wa and wb else 0.0


def _read_attempts(root) -> list[dict]:
    import json
    try:
        data = json.loads(attempts_file(root).read_text(encoding="utf-8"))
        return [d for d in data if isinstance(d, dict) and d.get("task")]
    except (OSError, ValueError):
        return []


def _write_attempts(root, rows: list[dict]) -> None:
    import json
    path = attempts_file(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows[-_MAX_ATTEMPTS:], indent=1),
                        encoding="utf-8")
    except OSError:
        pass


def note_attempt(root, task: str, why: str, files: list[str] | None = None) -> None:
    task = " ".join((task or "").split())[:200]
    if not task:
        return
    rows = [r for r in _read_attempts(root) if _similar(r["task"], task) < 0.9]
    rows.append({"task": safety.redact(task), "why": safety.redact(why)[:200],
                 "files": list(files or [])[:6]})
    _write_attempts(root, rows)


def clear_attempts(root, task: str) -> None:
    rows = _read_attempts(root)
    keep = [r for r in rows if _similar(r["task"], task) < 0.6]
    if len(keep) != len(rows):
        _write_attempts(root, keep)


def attempts_note(root, task: str, cfg=None) -> str:
    """A short block about earlier failed attempts at a similar task ('' if none)."""
    if cfg is not None and not getattr(cfg, "use_memory", True):
        return ""
    hits = [r for r in _read_attempts(root) if _similar(r["task"], task) >= 0.6]
    if not hits:
        return ""
    lines = ["EARLIER FAILED ATTEMPT(S) at a similar task (do not repeat them):"]
    for r in hits[-2:]:
        files = f" (looked at {', '.join(r['files'])})" if r.get("files") else ""
        lines.append(f"- {r['task'][:120]}: {r.get('why') or 'no edit landed'}{files}")
    return "\n".join(lines)


# ------------------------------------------------------------ scoped rules
def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[3:end].splitlines():
        k, _, v = line.partition(":")
        if k.strip() and v.strip():
            meta[k.strip().lower()] = v.strip().strip("\"'")
    return meta, text[end + 4:].lstrip("\n")


def rules_for(root, rel: str, max_chars: int = 600) -> list[tuple[str, str]]:
    """[(source, text)] of rules that apply to file `rel`: `.pedacito/rules/*.md`
    whose `paths:` globs match it, and the nearest nested AGENTS.md /
    CLAUDE.md in the file's ancestor directories (the root one is global)."""
    root = Path(root)
    rel = rel.replace("\\", "/")
    out: list[tuple[str, str]] = []
    rules_dir = root / ".pedacito" / "rules"
    if rules_dir.is_dir():
        for p in sorted(rules_dir.glob("*.md")):
            try:
                meta, body = _frontmatter(p.read_text(encoding="utf-8"))
            except OSError:
                continue
            globs = [g.strip() for g in re.split(r"[,\s]+", meta.get("paths", "")) if g.strip()]
            if globs and any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(
                    Path(rel).name, g) for g in globs):
                out.append((f".pedacito/rules/{p.name}", body.strip()[:max_chars]))
    d = Path(rel).parent
    while str(d) not in (".", ""):
        for name in ("AGENTS.md", "CLAUDE.md", "PEDACITO.md"):
            p = root / d / name
            if p.is_file():
                try:
                    out.append((f"{d.as_posix()}/{name}",
                                p.read_text(encoding="utf-8").strip()[:max_chars]))
                except OSError:
                    pass
                return out
        d = d.parent
    return out


# ---------------------------------------------------------------- the tool
def tool_remember(index, cfg, content: str) -> str:
    """`remember` action: save one durable fact (build command, convention,
    trap) for future runs."""
    if not getattr(cfg, "use_memory", True):
        return "ERROR: memory is disabled (use_memory = false)."
    ok, msg = remember(index.root, content)
    return (f"MEMORY: {msg}." if ok
            else f"MEMORY: not saved -- {msg}.")
