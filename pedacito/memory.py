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
import json
import re
import time
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


# ------------------------------------------------- repository micro-facts
#
# TokenEconomizer's Micro-Skills are zero-LLM-call facts a session derives
# deterministically instead of rediscovering every run. Pedacito keeps only
# the environment/command slice: a command that succeeds twice (or succeeds
# after an earlier failure) becomes a confirmed fact and is injected, capped,
# into the stable prompt prefix. A later failure is a contradiction: the fact
# is retired immediately so stale advice cannot linger.

MICRO_FACTS_FILE = "micro_facts.json"
TOOL_STATS_FILE = "tool_stats.json"
_MAX_MICRO_FACTS = 24
_MAX_PENDING_FACTS = 24


def micro_facts_file(root) -> Path:
    return Path(root) / ".pedacito" / MICRO_FACTS_FILE


def tool_stats_file(root) -> Path:
    return Path(root) / ".pedacito" / TOOL_STATS_FILE


def _read_json(path: Path, default):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, type(default)) else default
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def _facts_doc(root) -> dict:
    """The micro-fact document: {facts, pending, retired, anti_patterns}.

    Rebuilt from a whitelist of keys on every read, so any writer sees the
    same shape; `anti_patterns` (the repair lesson ledger) must be listed
    here or another writer's round-trip would silently drop it."""
    raw = _read_json(micro_facts_file(root), {})
    facts = [f for f in raw.get("facts", ())
             if isinstance(f, dict) and f.get("fact_id") and f.get("text")]
    pending = [f for f in raw.get("pending", ())
               if isinstance(f, dict) and f.get("fact_id") and f.get("text")]
    retired = [f for f in raw.get("retired", ())
               if isinstance(f, dict) and f.get("fact_id")]
    anti = [a for a in raw.get("anti_patterns", ())
            if isinstance(a, dict) and a.get("lesson_id") and a.get("text")]
    return {"facts": facts[-_MAX_MICRO_FACTS:],
            "pending": pending[-_MAX_PENDING_FACTS:],
            "retired": retired[-_MAX_MICRO_FACTS:],
            "anti_patterns": anti[-_MAX_ANTI_PATTERNS:]}


# ------------------------------------------------- repair lesson ledger ---
#
# A round of the auto loop that went broken -> fixed is the cheapest lesson
# the repository will ever produce: the tooling already knows what failed,
# what the repair was, and that it worked. Most of that knowledge evaporates
# when the session ends, so the next session pays the same tuition. Each
# lesson is ONE line, distilled mechanically from the check findings (never
# from LLM prose: the distilled kind and file set are what make the line
# trustworthy), capped at eight, and auto-retired once the files it names
# are gone. These are hints injected under [REPO GOTCHAS & LESSONS], not
# truth; the loop still re-checks everything live.

_MAX_ANTI_PATTERNS = 8


def _lesson_identity(kind: str, text: str) -> str:
    norm = " ".join(str(text or "").split()).lower()
    digest = hashlib.sha1(norm.encode("utf-8", "replace")).hexdigest()
    return f"ap:{str(kind or '').strip().lower()}:{digest[:12]}"


def record_anti_pattern(root, kind: str, text: str,
                        files=(), cfg=None) -> dict | None:
    """Add one distilled repair lesson; duplicates bump `confirmations`
    instead of growing the ledger. Oldest lessons are evicted past the cap.
    Returns the stored row, or None when refused (feature off, empty or
    injection-shaped text)."""
    if cfg is not None and not getattr(cfg, "micro_facts", True):
        return None
    kind = str(kind or "").strip().lower()[:32]
    text = " ".join(str(text or "").split())[:300]
    if not kind or not text:
        return None
    # No injection screen here, unlike `remember`: lessons are distilled
    # mechanically by loop.distill_repair_lesson from gate findings and test
    # output, and its capture regexes (ids, names, module paths) admit no
    # free text at all -- an instruction-shaped lesson cannot be produced.
    lesson_id = _lesson_identity(kind, text)
    now = time.time()
    doc = _facts_doc(root)
    existing = next((a for a in doc["anti_patterns"]
                     if a.get("lesson_id") == lesson_id), None)
    if existing is not None:
        existing["confirmations"] = int(existing.get("confirmations", 1) or 1) + 1
        existing["last_seen"] = now
        existing["files"] = sorted({str(f) for f in (files or ())}
                                   | {str(f) for f in existing.get("files", ())})[:8]
        _write_json(micro_facts_file(root), doc)
        return existing
    row = {"lesson_id": lesson_id, "kind": kind, "text": text,
           "files": sorted({str(f) for f in (files or ())})[:8],
           "confirmations": 1, "last_seen": now}
    doc["anti_patterns"].append(row)
    doc["anti_patterns"] = doc["anti_patterns"][-_MAX_ANTI_PATTERNS:]
    _write_json(micro_facts_file(root), doc)
    return row


def load_anti_patterns(root, prune: bool = True) -> list[dict]:
    """Live lessons; entries whose every named file is gone are retired and
    dropped from the document (a lesson about a deleted component would
    otherwise outlive the thing it warns about)."""
    root = Path(root) if root else None
    doc = _facts_doc(root)
    live, gone = [], []
    for a in doc["anti_patterns"]:
        files = [f for f in a.get("files", ()) if str(f).strip()]
        if files and root is not None and prune \
                and not any((root / str(f)).exists() for f in files):
            gone.append(a)
        else:
            live.append(a)
    if gone and root is not None:
        doc["anti_patterns"] = live
        _write_json(micro_facts_file(root), doc)
    return live


def render_anti_patterns(root, cfg=None, limit: int | None = None) -> str:
    """The lesson ledger as a short stable-prefix block ('' when empty)."""
    if cfg is not None and not getattr(cfg, "micro_facts", True):
        return ""
    rows = load_anti_patterns(root)
    if not rows:
        return ""
    rows = sorted(rows, key=lambda a: (-float(a.get("last_seen", 0) or 0),
                                       str(a.get("kind", ""))))
    tokens = int(limit if limit is not None
                 else getattr(cfg, "max_micro_fact_tokens", 150) or 150)
    budget = max(160, tokens * 4)
    head = ("[REPO GOTCHAS & LESSONS] (repair lessons distilled from earlier "
            "runs; hints, not truth):")
    lines = [f"- {a.get('kind', 'note')}: {a['text']}" for a in rows]
    while lines and len(head) + 1 + sum(len(x) + 1 for x in lines) > budget:
        lines.pop()                    # oldest lessons first to go
    return head + "\n" + "\n".join(lines) if lines else ""


def load_micro_facts(root, include_retired: bool = False) -> list[dict]:
    """Confirmed facts (or every record, including retired ones) for tests
    and inspection."""
    doc = _facts_doc(root)
    rows = list(doc["facts"])
    if include_retired:
        rows += list(doc["retired"])
    return rows


def _command_identity(command: str) -> tuple[str, str]:
    norm = " ".join(str(command or "").split())
    digest = hashlib.sha1(norm.lower().encode("utf-8", "replace")).hexdigest()
    return norm, f"cmd:{digest[:12]}"


def _fact_text(command: str, kind: str, confirmations: int) -> str:
    runs = "run" if confirmations == 1 else "runs"
    if kind == "test":
        return f"`{command}` works as the test command here ({confirmations} {runs})."
    return f"`{command}` works here ({confirmations} {runs})."


def observe_command(root, command: str, ok: bool, cfg=None,
                    kind: str = "command") -> dict | None:
    """Record one executed command outcome and promote/retire a micro-fact.

    Promotion needs two successes, or one success after an earlier failure.
    An active fact contradicted by a failure is moved to `retired` with its
    contradiction count incremented; retired facts are never injected.
    Returns the promoted fact, or None.
    """
    if cfg is not None and not getattr(cfg, "micro_facts", True):
        return None
    command, fact_id = _command_identity(command)
    if not command or len(command) > 240:
        return None
    if safety.looks_like_injection(command):
        return None
    command = safety.redact(command)
    now = time.time()
    doc = _facts_doc(root)

    for fact in list(doc["facts"]):
        if fact.get("fact_id") != fact_id:
            continue
        confirmations = int(fact.get("confirmations", 0) or 0)
        if ok:
            fact["confirmations"] = confirmations + 1
            fact["last_seen"] = now
            fact["text"] = _fact_text(command, fact.get("kind", kind),
                                      fact["confirmations"])
            _write_json(micro_facts_file(root), doc)
            return fact
        # Contradiction: the command that used to work now fails. Retire it
        # at once -- a stale environment fact is worse than none.
        fact["contradictions"] = int(fact.get("contradictions", 0) or 0) + 1
        fact["last_seen"] = now
        doc["facts"].remove(fact)
        doc["retired"] = (doc["retired"] + [fact])[-_MAX_MICRO_FACTS:]
        _write_json(micro_facts_file(root), doc)
        return None

    pending = next((p for p in doc["pending"]
                    if p.get("fact_id") == fact_id), None)
    if pending is None:
        pending = {"fact_id": fact_id, "text": _fact_text(command, kind, 0),
                   "successes": 0, "failures": 0, "last_seen": now,
                   "kind": kind}
        doc["pending"].append(pending)
    pending["last_seen"] = now
    if ok:
        pending["successes"] = int(pending.get("successes", 0) or 0) + 1
    else:
        pending["failures"] = int(pending.get("failures", 0) or 0) + 1

    promoted = None
    if ok and (pending["successes"] >= 2 or pending["failures"] >= 1):
        promoted = {
            "fact_id": fact_id,
            "text": _fact_text(command, kind, pending["successes"]),
            "confirmations": pending["successes"],
            "contradictions": 0,
            "last_seen": now,
            "kind": kind,
        }
        doc["facts"].append(promoted)
        doc["pending"].remove(pending)
    doc["facts"] = doc["facts"][-_MAX_MICRO_FACTS:]
    doc["pending"] = doc["pending"][-_MAX_PENDING_FACTS:]
    _write_json(micro_facts_file(root), doc)
    return promoted


def render_micro_facts(root, cfg=None, limit: int | None = None) -> str:
    """Confirmed micro-facts as a short stable-prefix block ('' if none)."""
    if cfg is not None and not getattr(cfg, "micro_facts", True):
        return ""
    facts = _facts_doc(root)["facts"]
    if not facts:
        return ""
    facts = sorted(facts, key=lambda f: (-int(f.get("confirmations", 0) or 0),
                                         -float(f.get("last_seen", 0) or 0),
                                         str(f.get("text", ""))))[:10]
    tokens = int(limit if limit is not None
                 else getattr(cfg, "max_micro_fact_tokens", 150) or 150)
    budget = max(160, tokens * 4)
    head = ("REPO MICRO-FACTS (confirmed commands from earlier runs; hints, "
            "not truth):")
    rows = [f"- {f['text']}" for f in facts
            if f.get("text") and not safety.looks_like_injection(str(f["text"]))]
    while rows and len(head) + 1 + sum(len(r) + 1 for r in rows) > budget:
        rows.pop()                     # least-confirmed first
    return head + "\n" + "\n".join(rows) if rows else ""


# ------------------------------------------------------------ tool stats
#
# Dynamic tool schema pruning needs one cheap signal: how many repository
# tasks have run since each optional action was last used. Task starts bump
# the counter; dispatch records actual uses. A never-used tool therefore
# becomes prunable after `tool_prune_after_tasks` tasks, and a recently used
# one stays visible. All local, deterministic, and capped.

def _tool_stats_doc(root) -> dict:
    raw = _read_json(tool_stats_file(root), {})
    uses = raw.get("uses") if isinstance(raw.get("uses"), dict) else {}
    last = raw.get("last_task") if isinstance(raw.get("last_task"), dict) else {}
    return {"tasks": max(0, int(raw.get("tasks", 0) or 0)),
            "uses": {str(k): int(v or 0) for k, v in uses.items()},
            "last_task": {str(k): int(v or 0) for k, v in last.items()}}


def begin_tool_task(root, cfg=None) -> int:
    """Count one task start; returns the repository's task counter."""
    if cfg is not None and not getattr(cfg, "dynamic_tool_pruning", True):
        return 0
    doc = _tool_stats_doc(root)
    doc["tasks"] += 1
    _write_json(tool_stats_file(root), doc)
    if cfg is not None:
        try:
            cfg._track_tool_use = True
        except Exception:
            pass
    return doc["tasks"]


def record_tool_use(root, action: str, cfg=None) -> None:
    """Record one dispatched action for dynamic schema pruning."""
    if cfg is not None and not getattr(cfg, "dynamic_tool_pruning", True):
        return
    if cfg is not None and not getattr(cfg, "_track_tool_use", False):
        return
    action = str(action or "").strip()
    if not action:
        return
    doc = _tool_stats_doc(root)
    doc["uses"][action] = int(doc["uses"].get(action, 0)) + 1
    doc["last_task"][action] = doc["tasks"]
    _write_json(tool_stats_file(root), doc)


def tool_unused_tasks(root, action: str) -> int:
    """Tasks since `action` was last used (0 when never used)."""
    doc = _tool_stats_doc(root)
    return max(0, doc["tasks"] - int(doc["last_task"].get(action, 0) or 0))


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
