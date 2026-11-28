"""Prompt budget provenance: what a run actually sent, without storing it.

A `PromptReport` records the sections a request was assembled from -- in
assembly order -- with a hash, a token count, the counting method, and a
disposition (`kept`, `truncated`, `dropped`, `pruned`, `disabled`, `sent`).
The report never stores the section text, the prompt, source files, tool
output, or credentials: only hashes, lengths and rule/skill/memory IDs.

Counting is explicit about its evidence. A character heuristic is always
labelled ``estimate:<name>``; only a provider-reported usage number may be
labelled ``exact:<name>``. Nothing here guesses that a tokenizer was used
when none was.

Reports are session-scoped project artifacts under
``.pedacito/prompt_reports/``. Every path goes through
``security.resolve_safe_path``; a report from another session is never
returned by `latest_report(root, session_id=...)`. The explain/compare paths
read local files only and never contact a model server.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import fileio, safety, security

VERSION = 1
REPORT_DIR = ".pedacito/prompt_reports"
# A report is small metadata; a runaway section list is capped rather than
# allowed to grow a project file without bound.
MAX_REPORTS = 40
MAX_SECTIONS = 160
MAX_PRUNED = 40
MAX_NAME = 120
MAX_REASON = 240
MAX_ID = 160
MAX_MODEL = 160
MAX_ACTIONS = 64
# llm.CHARS_PER_TOKEN is 3.6; using the same divisor keeps the inspector's
# estimate aligned with the answer-prompt budget checks. It is still an
# estimate: the method label says so.
CHARS_PER_TOKEN = 3.6
ESTIMATE_METHOD = "estimate:chars-div-3.6"
EXACT_USAGE_METHOD = "exact:provider-usage"
TRANSPORT_SECTION = "transport:serialized-request"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.:/-]+")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------- dataclasses

@dataclass(frozen=True)
class PromptSection:
    """One assembled prompt section, identified by hash rather than text."""

    name: str
    ordinal: int
    token_count: int | None
    count_method: str
    content_hash: str
    disposition: str
    reason: str


@dataclass(frozen=True)
class PromptReport:
    """The provenance of one run's prompt assembly and final serialization."""

    version: int
    run_id: str
    model: str
    sections: tuple[PromptSection, ...]
    prefix_message_count: int
    prefix_hash: str
    schema_actions: tuple[str, ...]
    pruned: dict[str, str]


# --------------------------------------------------------------- primitives

def sha256_text(text: str) -> str:
    """Stable content hash for a prompt section (UTF-8, hex SHA-256)."""
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """The same character heuristic the prompt budgets use (never "exact")."""
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / CHARS_PER_TOKEN)))


def count_text(text: str, *, token_count: int | None = None,
               count_method: str | None = None) -> tuple[int | None, str]:
    """`(token_count, count_method)` for a section.

    An explicit exact count is used only when the caller supplies both a
    number and an ``exact:`` method; otherwise the character heuristic is
    used and labelled as an estimate.
    """
    method = str(count_method or "").strip()
    if token_count is not None and method.startswith(("exact:", "estimate:")):
        return max(0, int(token_count)), method
    return estimate_tokens(text), ESTIMATE_METHOD


def new_run_id(prefix: str = "pr") -> str:
    """A sortable ASCII run id: ``pr-YYYYmmdd-HHMMSS-<uuid8>``."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def valid_run_id(run_id: str) -> bool:
    """True for an id this module is willing to turn into a filename."""
    return bool(_RUN_ID_RE.match(str(run_id or "").strip()))


def _bounded(value: Any, cap: int) -> str:
    """One-line, redacted, control-free metadata text."""
    text = safety.strip_control(safety.redact(str(value or "")))
    return _WS_RE.sub(" ", text).strip()[:cap]


def _safe_name(name: Any) -> str:
    text = _NAME_SAFE_RE.sub("-", _bounded(name, MAX_NAME))
    return text.strip("-") or "section"


def _section_id(value: Any) -> str:
    return _bounded(value, MAX_ID)


def _unique(items: Iterable[str], cap: int = 12) -> tuple[str, ...]:
    out: list[str] = []
    for item in items or ():
        text = _section_id(item)
        if text and text not in out:
            out.append(text)
        if len(out) >= cap:
            break
    return tuple(out)


_MEMORY_ID_RE = re.compile(r"^\s*-\s*\[([^\]\n]{1,80})\]", re.M)
_SKILL_ID_RE = re.compile(r"^\s*-\s*([A-Za-z0-9][A-Za-z0-9._-]{0,80}):", re.M)


def memory_ids(text: str) -> tuple[str, ...]:
    """`[scope/category]` IDs from a rendered project-memory block."""
    return _unique(m.group(1).strip() for m in _MEMORY_ID_RE.finditer(text or ""))


def skill_ids(text: str) -> tuple[str, ...]:
    """Skill names from a rendered skills catalog (no descriptions)."""
    return _unique(m.group(1).strip() for m in _SKILL_ID_RE.finditer(text or ""))


# ------------------------------------------------------------------- report

def build_report(*, run_id: str, model: str,
                 sections: Iterable[PromptSection],
                 prefix_message_count: int = 0, prefix_hash: str = "",
                 schema_actions: Iterable[str] = (),
                 pruned: dict[str, str] | None = None,
                 version: int = VERSION) -> PromptReport:
    """Build a bounded, deterministic-order report from section records."""
    ordered = sorted(sections, key=lambda s: (int(s.ordinal), s.name))
    if len(ordered) > MAX_SECTIONS:
        ordered = ordered[:MAX_SECTIONS]
    actions = tuple(_section_id(a) for a in (schema_actions or ()))[:MAX_ACTIONS]
    safe_pruned: dict[str, str] = {}
    for key, value in list((pruned or {}).items())[:MAX_PRUNED]:
        safe_pruned[_safe_name(key)] = _bounded(value, MAX_REASON)
    return PromptReport(
        version=int(version),
        run_id=str(run_id or ""),
        model=_bounded(model, MAX_MODEL),
        sections=tuple(ordered),
        prefix_message_count=max(0, int(prefix_message_count or 0)),
        prefix_hash=_bounded(prefix_hash, 64),
        schema_actions=actions,
        pruned=safe_pruned,
    )


def section_to_dict(section: PromptSection) -> dict[str, Any]:
    return {
        "name": section.name,
        "ordinal": int(section.ordinal),
        "token_count": (None if section.token_count is None
                        else int(section.token_count)),
        "count_method": section.count_method,
        "content_hash": section.content_hash,
        "disposition": section.disposition,
        "reason": section.reason,
    }


def report_to_dict(report: PromptReport) -> dict[str, Any]:
    return {
        "version": int(report.version),
        "run_id": report.run_id,
        "model": report.model,
        "sections": [section_to_dict(s) for s in report.sections],
        "prefix_message_count": int(report.prefix_message_count),
        "prefix_hash": report.prefix_hash,
        "schema_actions": list(report.schema_actions),
        "pruned": dict(report.pruned),
    }


def report_from_dict(data: Any) -> PromptReport | None:
    """Parse an envelope (``{"report": {...}}``) or a bare report object."""
    if not isinstance(data, dict):
        return None
    raw = data.get("report") if isinstance(data.get("report"), dict) else data
    sections: list[PromptSection] = []
    for item in list(raw.get("sections") or [])[:MAX_SECTIONS]:
        if not isinstance(item, dict):
            continue
        method = str(item.get("count_method") or ESTIMATE_METHOD)
        if not method.startswith(("exact:", "estimate:")):
            method = ESTIMATE_METHOD
        count = item.get("token_count")
        try:
            count = None if count is None else max(0, int(count))
        except (TypeError, ValueError):
            count = None
        sections.append(PromptSection(
            name=_safe_name(item.get("name")),
            ordinal=int(item.get("ordinal") or 0),
            token_count=count,
            count_method=method[:80],
            content_hash=str(item.get("content_hash") or "")[:64],
            disposition=_safe_name(item.get("disposition") or "unknown"),
            reason=_bounded(item.get("reason"), MAX_REASON),
        ))
    pruned_raw = raw.get("pruned") if isinstance(raw.get("pruned"), dict) else {}
    actions = raw.get("schema_actions")
    if not isinstance(actions, (list, tuple)):
        actions = ()
    return build_report(
        run_id=str(raw.get("run_id") or ""),
        model=str(raw.get("model") or ""),
        sections=sections,
        prefix_message_count=int(raw.get("prefix_message_count") or 0),
        prefix_hash=str(raw.get("prefix_hash") or ""),
        schema_actions=actions,
        pruned={str(k): str(v) for k, v in pruned_raw.items()},
        version=int(raw.get("version") or VERSION),
    )


# ------------------------------------------------------------ persistence

def reports_dir(root) -> Path | None:
    """The resolved report directory, or None when it is outside the root."""
    if root is None:
        return None
    try:
        return security.resolve_safe_path(root, REPORT_DIR)
    except (security.SecurityError, OSError, ValueError):
        return None


def report_path(root, run_id: str) -> Path | None:
    """Resolved path for one run report; None for an invalid/escaping id."""
    if not valid_run_id(run_id):
        return None
    try:
        return security.resolve_safe_path(
            root, f"{REPORT_DIR}/{str(run_id).strip()}.json")
    except (security.SecurityError, OSError, ValueError):
        return None


def _envelope(report: PromptReport, session_id: str,
              created: float | None = None) -> dict[str, Any]:
    payload = report_to_dict(report)
    payload["session_id"] = _bounded(session_id, 120)
    payload["saved_at"] = round(float(created if created is not None
                                     else time.time()), 3)
    return payload


def save_report(root, report: PromptReport, *, session_id: str = "",
                created: float | None = None) -> Path | None:
    """Write one report atomically; return its path (None when refused).

    Only metadata is serialized: sections carry hashes and counts, never
    their text. Report files are pruned to `MAX_REPORTS` newest entries.
    """
    target = report_path(root, report.run_id)
    if target is None:
        return None
    text = json.dumps(_envelope(report, session_id, created),
                      ensure_ascii=True, indent=2) + "\n"
    try:
        fileio.write(target, text, fileio.TextStyle())
    except (OSError, ValueError):
        return None
    _prune_reports(target.parent, keep=target, session_id=session_id)
    return target


# A familiar alias for callers that think in terms of "write the report".
write_report = save_report


def _prune_reports(directory: Path, *, keep: Path, session_id: str = "") -> None:
    """Keep the newest `MAX_REPORTS` for one session; never another's."""
    kept = 0
    try:
        files = sorted(
            (p for p in directory.glob("*.json")
             if p.is_file() and p != keep and valid_run_id(p.stem)),
            key=lambda p: p.name, reverse=True)
    except OSError:
        return
    for old in files:
        try:
            data = json.loads(fileio.read_text(old, errors="replace"))
        except (OSError, ValueError):
            continue
        if str((data or {}).get("session_id") or "") != str(session_id or ""):
            continue
        kept += 1
        if kept < MAX_REPORTS:
            continue
        try:
            old.unlink()
        except OSError:
            pass


def load_report(root, run_id: str, *,
                session_id: str | None = None) -> PromptReport | None:
    """Read one report; with `session_id`, refuse another session's report."""
    path = report_path(root, run_id)
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(fileio.read_text(path, errors="replace"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if session_id is not None \
            and str(data.get("session_id") or "") != str(session_id):
        return None
    report = report_from_dict(data)
    if report is None or report.run_id != str(run_id):
        return None
    return report


def list_reports(root, *, session_id: str | None = None,
                 limit: int = 20) -> list[str]:
    """Newest run ids first (optionally limited to one session)."""
    directory = reports_dir(root)
    if directory is None or not directory.is_dir():
        return []
    out: list[str] = []
    try:
        files = sorted((p for p in directory.glob("*.json")
                        if p.is_file() and valid_run_id(p.stem)),
                       key=lambda p: p.name, reverse=True)
    except OSError:
        return []
    for path in files[:max(1, int(limit or 20)) * 4]:
        try:
            data = json.loads(fileio.read_text(path, errors="replace"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if session_id is not None \
                and str(data.get("session_id") or "") != str(session_id):
            continue
        run_id = str(data.get("run_id") or path.stem)
        if valid_run_id(run_id) and run_id not in out:
            out.append(run_id)
        if len(out) >= max(1, int(limit or 20)):
            break
    return out


def latest_report(root, *, session_id: str | None = None
                  ) -> PromptReport | None:
    """The newest report for the project (and optionally this session)."""
    ids = list_reports(root, session_id=session_id, limit=1)
    if not ids:
        return None
    return load_report(root, ids[0], session_id=session_id)


# ------------------------------------------------------------------ compare

def _section_summary(section: PromptSection) -> dict[str, Any]:
    return {
        "name": section.name,
        "ordinal": int(section.ordinal),
        "token_count": (None if section.token_count is None
                        else int(section.token_count)),
        "count_method": section.count_method,
        "content_hash": section.content_hash,
        "disposition": section.disposition,
        "reason": section.reason,
    }


def _total_tokens(report: PromptReport) -> int:
    return sum(int(s.token_count or 0) for s in report.sections)


def compare_reports(base: PromptReport, other: PromptReport) -> dict[str, Any]:
    """Diff two reports by section IDs, hashes and counts -- never by text.

    `base` is the newer/explained run, `other` the one it is compared with.
    Section text was never stored, so no source can appear in the diff.
    """
    before = {s.name: s for s in other.sections}
    after = {s.name: s for s in base.sections}
    added = [_section_summary(after[n]) for n in after if n not in before]
    removed = [_section_summary(before[n]) for n in before if n not in after]
    changed: list[dict[str, Any]] = []
    for name in (n for n in after if n in before):
        old, new = before[name], after[name]
        fields: list[str] = []
        if old.ordinal != new.ordinal:
            fields.append("ordinal")
        if old.token_count != new.token_count:
            fields.append("token_count")
        if old.count_method != new.count_method:
            fields.append("count_method")
        if old.content_hash != new.content_hash:
            fields.append("content_hash")
        if old.disposition != new.disposition:
            fields.append("disposition")
        if old.reason != new.reason:
            fields.append("reason")
        if fields:
            changed.append({"name": name, "fields": fields,
                            "before": _section_summary(old),
                            "after": _section_summary(new)})
    before_order = [s.name for s in other.sections]
    after_order = [s.name for s in base.sections]
    common_before = [n for n in before_order if n in after]
    common_after = [n for n in after_order if n in before]
    pruned_before = dict(other.pruned)
    pruned_after = dict(base.pruned)
    pruned = {
        "added": {k: v for k, v in pruned_after.items()
                  if k not in pruned_before},
        "removed": {k: v for k, v in pruned_before.items()
                    if k not in pruned_after},
        "changed": {k: {"before": pruned_before[k], "after": pruned_after[k]}
                    for k in pruned_after
                    if k in pruned_before
                    and pruned_before[k] != pruned_after[k]},
    }
    old_total = _total_tokens(other)
    new_total = _total_tokens(base)
    return {
        "base_run_id": base.run_id,
        "compare_run_id": other.run_id,
        "model": {"base": base.model, "compare": other.model},
        "prefix": {
            "base": {"count": base.prefix_message_count,
                     "hash": base.prefix_hash},
            "compare": {"count": other.prefix_message_count,
                        "hash": other.prefix_hash},
            "changed": (base.prefix_message_count != other.prefix_message_count
                        or base.prefix_hash != other.prefix_hash),
        },
        "schema_actions": {
            "base": list(base.schema_actions),
            "compare": list(other.schema_actions),
            "added": sorted(set(base.schema_actions)
                            - set(other.schema_actions)),
            "removed": sorted(set(other.schema_actions)
                              - set(base.schema_actions)),
        },
        "sections": {
            "added": added,
            "removed": removed,
            "changed": changed,
            "order_changed": bool(common_before != common_after),
            "unchanged": len([n for n in after
                              if n in before and before[n] == after[n]]),
        },
        "pruned": pruned,
        "totals": {
            "base": new_total,
            "compare": old_total,
            "delta": new_total - old_total,
        },
    }


# --------------------------------------------------------------- rendering

def _short_hash(value: str) -> str:
    return (value or "-")[:12]


def render_report(report: PromptReport, translate=None) -> list[str]:
    """Human lines for one report (English literals for a plain caller)."""
    def t(key: str, fallback: str, **kw) -> str:
        if translate is not None:
            try:
                return translate(key, **kw)
            except Exception:
                pass
        return fallback.format(**kw) if kw else fallback

    lines = [t("cli.prompt.header",
               "Prompt report {run_id}", run_id=report.run_id),
             t("cli.prompt.model", "model: {model}", model=report.model or "-"),
             t("cli.prompt.prefix",
               "prefix: {count} message(s), sha256:{hash}",
               count=report.prefix_message_count,
               hash=_short_hash(report.prefix_hash))]
    actions = ", ".join(report.schema_actions[:12])
    lines.append(t("cli.prompt.schema", "schema actions: {count} ({names})",
                   count=len(report.schema_actions), names=actions or "-"))
    lines.append(t("cli.prompt.sections", "sections (assembly order):"))
    for section in report.sections:
        count = ("-" if section.token_count is None
                 else str(section.token_count))
        lines.append(
            f"  {section.ordinal:>3}  {section.name:<34} "
            f"{count:>7}  {section.count_method:<24} "
            f"{section.disposition:<9} #{_short_hash(section.content_hash)}"
            + (f"  {section.reason}" if section.reason else ""))
    if report.pruned:
        lines.append(t("cli.prompt.pruned", "pruned/dropped:"))
        for name, reason in report.pruned.items():
            lines.append(f"  {name}: {reason}")
    lines.append(t("cli.prompt.total",
                   "total: {tokens} tokens (estimated section counts)",
                   tokens=_total_tokens(report)))
    return lines


def render_diff(diff: dict[str, Any], translate=None) -> list[str]:
    """Human lines for `compare_reports`, showing only IDs/hashes/counts."""
    def t(key: str, fallback: str, **kw) -> str:
        if translate is not None:
            try:
                return translate(key, **kw)
            except Exception:
                pass
        return fallback.format(**kw) if kw else fallback

    lines = [t("cli.prompt.compare_header",
               "Prompt diff: {base} -> {target}",
               base=diff.get("compare_run_id", "?"),
               target=diff.get("base_run_id", "?"))]
    prefix = diff.get("prefix") or {}
    if prefix.get("changed"):
        lines.append(t(
            "cli.prompt.compare_prefix",
            "prefix changed: {old} -> {new}",
            old=_short_hash((prefix.get("compare") or {}).get("hash", "")),
            new=_short_hash((prefix.get("base") or {}).get("hash", ""))))
    else:
        lines.append(t("cli.prompt.compare_prefix_same",
                       "prefix unchanged: {hash}",
                       hash=_short_hash((prefix.get("base") or {}).get("hash", ""))))
    sections = diff.get("sections") or {}
    for label, entries in (("added", sections.get("added") or []),
                           ("removed", sections.get("removed") or [])):
        if not entries:
            continue
        lines.append(t(f"cli.prompt.compare_{label}",
                       f"{label} sections:", label=label))
        for entry in entries:
            lines.append(
                f"  {entry['name']}  {entry.get('token_count') or 0} tok  "
                f"{entry.get('count_method', '')}  "
                f"{entry.get('disposition', '')}  "
                f"#{_short_hash(entry.get('content_hash', ''))}")
    changed = sections.get("changed") or []
    if changed:
        lines.append(t("cli.prompt.compare_changed", "changed sections:"))
        for entry in changed:
            old = entry.get("before") or {}
            new = entry.get("after") or {}
            fields = ",".join(entry.get("fields") or [])
            lines.append(
                f"  ~ {entry['name']}  fields={fields}  "
                f"tokens {old.get('token_count')} -> {new.get('token_count')}  "
                f"#{_short_hash(old.get('content_hash', ''))}"
                f" -> #{_short_hash(new.get('content_hash', ''))}")
    pruned = diff.get("pruned") or {}
    if any(pruned.get(k) for k in ("added", "removed", "changed")):
        lines.append(t("cli.prompt.compare_pruned", "pruned changes:"))
        for key in ("added", "removed"):
            for name, reason in (pruned.get(key) or {}).items():
                lines.append(f"  {key} {name}: {reason}")
        for name, pair in (pruned.get("changed") or {}).items():
            lines.append(f"  changed {name}: {pair.get('before')} -> "
                         f"{pair.get('after')}")
    totals = diff.get("totals") or {}
    lines.append(t("cli.prompt.compare_total",
                   "total: {old} -> {new} tokens ({delta:+d})",
                   old=totals.get("compare", 0),
                   new=totals.get("base", 0),
                   delta=int(totals.get("delta") or 0)))
    return lines


# ----------------------------------------------------------------- capture

class PromptCapture:
    """Collects section metadata for one run, then writes it as a report.

    The capture keeps only derived metadata (hash, length, method, reason).
    `add()` hashes the text immediately and does not retain it. Failed
    writes never raise into a model run: inspection must not break a task.
    """

    def __init__(self, *, model: str = "", root=None, run_id: str | None = None,
                 session_id: str = "", schema_actions: Iterable[str] = (),
                 version: int = VERSION):
        self.root = Path(root) if root is not None else None
        self.model = _bounded(model, MAX_MODEL)
        self.run_id = (run_id if run_id and valid_run_id(run_id)
                       else new_run_id())
        self.session_id = _bounded(session_id, 120)
        self.version = int(version)
        self.schema_actions: tuple[str, ...] = tuple(
            _section_id(a) for a in (schema_actions or ()))[:MAX_ACTIONS]
        self.prefix_message_count = 0
        self.prefix_hash = ""
        self._sections: dict[str, PromptSection] = {}
        self._order: list[str] = []
        self._pruned: dict[str, str] = {}
        self._lock = threading.Lock()
        self._last_save = 0.0
        self.saved_path: Path | None = None
        # Set by note_request() so an embedding/vision POST cannot overwrite
        # the chat request's transport provenance.
        self._expect_request_payload = False

    # -- additions ---------------------------------------------------------
    def add(self, name: str, text: str | None, *, disposition: str = "kept",
            reason: str = "", matched: Iterable[str] = (),
            token_count: int | None = None,
            count_method: str | None = None,
            content_hash: str | None = None) -> PromptSection:
        """Record one section; repeated names update in place (same ordinal)."""
        safe = _safe_name(name)
        ids = _unique(matched)
        why = _bounded(reason, MAX_REASON)
        if ids:
            why = _bounded((why + "; " if why else "") + "matched="
                           + ",".join(ids), MAX_REASON)
        body = "" if text is None else str(text)
        count, method = count_text(body, token_count=token_count,
                                   count_method=count_method)
        digest = (str(content_hash)[:64] if content_hash is not None
                  else sha256_text(body) if text is not None else "")
        with self._lock:
            existing = self._sections.get(safe)
            ordinal = existing.ordinal if existing is not None else len(self._order)
            section = PromptSection(
                name=safe,
                ordinal=ordinal,
                token_count=None if text is None and token_count is None
                else count,
                count_method=method,
                content_hash=digest,
                disposition=_safe_name(disposition),
                reason=why,
            )
            self._sections[safe] = section
            if existing is None:
                self._order.append(safe)
        return section

    add_section = add

    def disabled(self, name: str, reason: str = "",
                 matched: Iterable[str] = ()) -> PromptSection:
        """Record a feature that is off/absent (no token claim)."""
        return self.add(name, None, disposition="disabled", reason=reason,
                        matched=matched)

    def drop(self, name: str, reason: str = "",
             matched: Iterable[str] = ()) -> PromptSection:
        return self.add(name, None, disposition="dropped", reason=reason,
                        matched=matched)

    def truncate(self, name: str, text: str, reason: str = "",
                 matched: Iterable[str] = ()) -> PromptSection:
        return self.add(name, text, disposition="truncated", reason=reason,
                        matched=matched)

    def prune(self, name: str, reason: str, text: str | None = None,
              matched: Iterable[str] = ()) -> PromptSection:
        """Mark a previously recorded section pruned/dropped for this run."""
        section = self.add(name, text, disposition="pruned", reason=reason,
                           matched=matched)
        with self._lock:
            self._pruned[section.name] = _bounded(reason, MAX_REASON)
        return section

    def set_pruned(self, name: str, reason: str) -> None:
        with self._lock:
            if len(self._pruned) < MAX_PRUNED or _safe_name(name) in self._pruned:
                self._pruned[_safe_name(name)] = _bounded(reason, MAX_REASON)

    def set_schema_actions(self, actions: Iterable[str]) -> None:
        with self._lock:
            merged = list(self.schema_actions)
            for action in actions or ():
                text = _section_id(action)
                if text and text not in merged:
                    merged.append(text)
            self.schema_actions = tuple(merged[:MAX_ACTIONS])

    def expect_chat_payload(self) -> None:
        """The next transport POST is the chat request just recorded."""
        with self._lock:
            self._expect_request_payload = True

    def take_chat_payload(self) -> bool:
        """True once for the chat POST announced by `expect_chat_payload`."""
        with self._lock:
            expected = self._expect_request_payload
            self._expect_request_payload = False
            return expected

    # -- request provenance ------------------------------------------------
    def note_prefix(self, messages: Iterable[dict] | None) -> None:
        """Record the stable first messages once per capture."""
        with self._lock:
            if self.prefix_hash:
                return
        msgs = list(messages or ())
        if not msgs:
            return
        prefix = msgs[:min(2, len(msgs))]
        try:
            encoded = json.dumps(prefix, ensure_ascii=False, sort_keys=True,
                                 default=str)
        except (TypeError, ValueError):
            encoded = repr(prefix)
        with self._lock:
            if not self.prefix_hash:
                self.prefix_message_count = len(prefix)
                self.prefix_hash = sha256_text(encoded)

    def note_transport(self, payload: Any, *, provider: str = "",
                       message_count: int = 0) -> None:
        """Hash the actual serialized request (after outbound redaction)."""
        try:
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                 default=str)
        except (TypeError, ValueError):
            encoded = repr(payload)
        digest = sha256_text(encoded)
        reason = (f"provider={_bounded(provider, 80) or '-'}; "
                  f"bytes={len(encoded.encode('utf-8'))}; "
                  f"messages={int(message_count or 0)}")
        self.add(TRANSPORT_SECTION, encoded, disposition="sent",
                 reason=reason, content_hash=digest)
        self.save()

    def note_exact_usage(self, prompt_tokens: int | None) -> None:
        """Upgrade the transport section to provider-reported usage."""
        try:
            count = int(prompt_tokens or 0)
        except (TypeError, ValueError):
            return
        if count <= 0:
            return
        with self._lock:
            current = self._sections.get(TRANSPORT_SECTION)
        if current is None:
            return
        self.add(TRANSPORT_SECTION, None, disposition="sent",
                 reason=(current.reason + "; usage=provider-reported"),
                 token_count=count, count_method=EXACT_USAGE_METHOD,
                 content_hash=current.content_hash)
        self.save(force=True)

    # -- finalization ------------------------------------------------------
    def report(self) -> PromptReport:
        with self._lock:
            sections = [self._sections[n] for n in self._order
                        if n in self._sections]
            pruned = dict(self._pruned)
            prefix_count = self.prefix_message_count
            prefix_hash = self.prefix_hash
            actions = self.schema_actions
        return build_report(
            run_id=self.run_id,
            model=self.model,
            sections=sections,
            prefix_message_count=prefix_count,
            prefix_hash=prefix_hash,
            schema_actions=actions,
            pruned=pruned,
            version=self.version,
        )

    def save(self, force: bool = False) -> Path | None:
        """Best-effort write; throttled unless `force`."""
        if self.root is None:
            return None
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_save < 1.0:
                return self.saved_path
            self._last_save = now
        try:
            path = save_report(self.root, self.report(),
                               session_id=self.session_id)
        except Exception:
            return self.saved_path
        if path is not None:
            self.saved_path = path
        return path


# ----------------------------------------------------------- context hook

_CURRENT: contextvars.ContextVar[PromptCapture | None] = contextvars.ContextVar(
    "pedacito_prompt_capture", default=None)


def current_capture() -> PromptCapture | None:
    """The capture active in this execution context, if any."""
    return _CURRENT.get()


def activate(capture: PromptCapture | None):
    """Make `capture` current; returns the reset token."""
    return _CURRENT.set(capture)


def deactivate(token) -> None:
    """Undo `activate` with its token (ignoring a stale/misordered token)."""
    try:
        _CURRENT.reset(token)
    except (ValueError, RuntimeError):
        _CURRENT.set(None)


def clear_current() -> None:
    """Drop the active capture (used when a run phase ends)."""
    _CURRENT.set(None)


def begin_capture(*, model: str = "", root=None, run_id: str | None = None,
                  session_id: str = "", schema_actions: Iterable[str] = (),
                  version: int = VERSION) -> PromptCapture:
    """Create and activate one session-scoped capture."""
    capture = PromptCapture(model=model, root=root, run_id=run_id,
                            session_id=session_id,
                            schema_actions=schema_actions, version=version)
    activate(capture)
    return capture


def note_section(name: str, text: str | None, *, disposition: str = "kept",
                 reason: str = "", matched: Iterable[str] = (),
                 token_count: int | None = None,
                 count_method: str | None = None) -> None:
    """Record a section on the active capture; no-op outside a run."""
    capture = _CURRENT.get()
    if capture is None:
        return
    capture.add(name, text, disposition=disposition, reason=reason,
                matched=matched, token_count=token_count,
                count_method=count_method)


def note_disabled(name: str, reason: str = "",
                  matched: Iterable[str] = ()) -> None:
    capture = _CURRENT.get()
    if capture is not None:
        capture.disabled(name, reason, matched)


def note_request(messages: Iterable[dict] | None, schema: dict | None = None
                 ) -> None:
    """Record the stable prefix and the actual schema at a transport call."""
    capture = _CURRENT.get()
    if capture is None:
        return
    capture.note_prefix(messages)
    capture.expect_chat_payload()
    if isinstance(schema, dict):
        try:
            from .tools import schema_action_names
            capture.set_schema_actions(schema_action_names(schema))
        except Exception:
            pass


def note_transport(payload: Any, *, provider: str = "",
                   message_count: int = 0) -> None:
    """Hook for the HTTP boundary: hash-only serialized request metadata."""
    capture = _CURRENT.get()
    if capture is not None and capture.take_chat_payload():
        capture.note_transport(payload, provider=provider,
                               message_count=message_count)


def note_exact_usage(prompt_tokens: int | None) -> None:
    capture = _CURRENT.get()
    if capture is not None:
        capture.note_exact_usage(prompt_tokens)


def event_payload(report: PromptReport) -> dict[str, Any]:
    """Bounded, redacted metadata for the `prompt_report` session event."""
    return {
        "run_id": _section_id(report.run_id),
        "version": int(report.version),
        "model": _bounded(report.model, MAX_MODEL),
        "section_count": len(report.sections),
        "prefix_message_count": int(report.prefix_message_count),
        "prefix_hash": _bounded(report.prefix_hash, 64),
        "schema_actions": list(report.schema_actions[:24]),
        "pruned_count": len(report.pruned),
        "sections": [
            {"name": s.name, "ordinal": int(s.ordinal),
             "tokens": s.token_count, "method": s.count_method,
             "disposition": s.disposition,
             "hash": s.content_hash[:16]}
            for s in report.sections[:40]
        ],
    }
