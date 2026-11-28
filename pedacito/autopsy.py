"""Deterministic autopsy of a finished run: why did it fail?

    from pedacito import autopsy
    result = autopsy.autopsy(root, "last")       # Autopsy(category, ...)

The autopsy reads only the typed events Pedacito already writes to
`<root>/.pedacito/events/*.jsonl` and never calls a model. Every event that
matches a category is a typed signal (a `step` whose observation carried a
known marker, a `gate_error`, a `task_budget`, ...); the category is the most
severe signal in the run, ties broken by the earliest `seq`, then by name.
When nothing matches, the answer is `unknown`: absence of evidence is never
turned into a diagnosis, and no model intent is inferred.

Evidence records are redacted and bounded: an id, seq, relative clock `t`,
kind, and a short whitelist of scalar fields. Raw observations, prompts,
tool output and free text never leave the log. Missing or old logs yield
`unknown` with `incomplete=True`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import events as events_mod
from . import safety

CATEGORIES = (
    "unread_edit", "format_error", "truncation", "schema_rejection",
    "test_regression", "flake", "api_hallucination", "context_overflow",
    "permission_denial", "timeout", "budget_stop", "partial_batch", "unknown",
)

# Higher is more severe. Only the category ranks; the evidence order is by seq.
SEVERITY = {
    "timeout": 8,
    "budget_stop": 7,
    "permission_denial": 6,
    "context_overflow": 6,
    "partial_batch": 6,
    "format_error": 5,
    "schema_rejection": 5,
    "test_regression": 5,
    "unread_edit": 4,
    "api_hallucination": 4,
    "truncation": 3,
    "flake": 2,
}

# Kinds that close a run. Without one the run is incomplete (crashed,
# cancelled mid-flight, or its log was pruned).
TERMINAL_KINDS = frozenset({
    "exit", "finish", "prompt_finished", "prompt_error", "run_cancelled",
})

# Observation markers produced by the tools themselves. Each is a stable
# harness message, not model text, so matching them is deterministic.
_OBS_MARKERS = (
    ("this task never read the file", "unread_edit"),
    ("denied by policy", "permission_denial"),
    ("timed out after", "timeout"),
    ("[truncated: context budget reached]", "truncation"),
)

_SCALAR_FIELDS = (
    "file", "action", "gate", "n", "step", "field", "stopped", "ok",
    "applied", "failed", "errors", "code", "obs_code", "phase", "rc",
    "steps", "signal", "reason_code", "timeout_s",
)

_NEXT_STEPS = {
    "unread_edit": ("read_file_before_edit",),
    "format_error": ("fix_syntax_then_retry",),
    "truncation": ("send_shorter_payload",),
    "schema_rejection": ("check_structured_output_support",),
    "test_regression": ("run_failing_test_scope",),
    "flake": ("rerun_or_quarantine_test",),
    "api_hallucination": ("verify_api_with_index",),
    "context_overflow": ("lower_context_or_fold",),
    "permission_denial": ("review_permissions",),
    "timeout": ("narrow_scope_or_raise_timeout",),
    "budget_stop": ("raise_budget_or_narrow_task",),
    "partial_batch": ("review_partial_edits",),
    "unknown": ("no_typed_evidence",),
}

_MAX_EVIDENCE = 12
_MAX_EVENTS_PER_FILE = 20000
_MAX_FILES = 8


@dataclass(frozen=True)
class Autopsy:
    category: str
    evidence: tuple[dict, ...]
    next_steps: tuple[str, ...]
    incomplete: bool


def classify_observation(text: str) -> str:
    """The category of one tool observation, or "" when it carries no known
    marker. Used at the gather step so the log records the cause, not the
    prose."""
    low = str(text or "")
    for marker, category in _OBS_MARKERS:
        if marker in low:
            return category
    return ""


def typed_category(ev: dict) -> str:
    """The category one typed event supports, or "" for an untyped event."""
    kind = str(ev.get("kind") or "")
    if kind in ("step", "tool_result"):
        code = str(ev.get("obs_code") or "")
        return code if code in SEVERITY else ""
    if kind == "payload_cut":
        return "truncation"
    if kind == "step_rejected":
        return "schema_rejection" if ev.get("reason") == "malformed" else ""
    if kind == "gate_error":
        return "format_error" if ev.get("gate") == "syntax" else ""
    if kind == "verify":
        if int(ev.get("errors") or 0) <= 0:
            return ""
        found = set(ev.get("kinds") or [])
        if "unread" in found:
            return "unread_edit"
        if "syntax" in found:
            return "format_error"
        return ""
    if kind == "task_budget":
        return "budget_stop"
    if kind in ("budget_stop", "timeout"):
        return kind
    if kind == "incremental_partial":
        return "partial_batch"
    if kind == "llm_error":
        code = str(ev.get("code") or "")
        return code if code in SEVERITY else ""
    if kind == "test_result":
        if ev.get("stopped") == "timeout":
            return "timeout"
        if ev.get("ok") is False and not ev.get("stopped"):
            return "test_regression"
        return ""
    if kind == "test_classified":
        return "flake" if ev.get("kind_label") == "flake" else ""
    return ""


def _events_dir(root: Path) -> Path:
    return Path(root) / ".pedacito" / events_mod.EVENT_DIR


def _read_file(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    for line in lines[:_MAX_EVENTS_PER_FILE]:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _select_run(root: Path, run_id: str) -> tuple[list[dict], str] | None:
    """Events of one run and its id. `last` is the newest log file's final
    session; any other value matches a session id, a session prefix, or a log
    file stem (the run's process)."""
    d = _events_dir(root)
    if not d.is_dir():
        return None
    try:
        files = sorted(d.glob("*.jsonl"), key=lambda p: p.name, reverse=True)
    except OSError:
        return None
    wanted = str(run_id or "last").strip() or "last"
    for path in files[:_MAX_FILES]:
        evs = _read_file(path)
        if not evs:
            continue
        keys = [str(ev.get("session_id") or path.stem) for ev in evs]
        if wanted == "last":
            key = keys[-1]
        else:
            matched = [k for k in keys
                       if k == wanted or k.startswith(wanted)
                       or path.stem == wanted]
            if not matched:
                continue
            key = matched[-1]
        group = [ev for ev, k in zip(evs, keys) if k == key]
        return group, key
    return None


def _has_seq_gap(evs: list[dict]) -> bool:
    seqs = sorted({int(ev["seq"]) for ev in evs
                   if isinstance(ev.get("seq"), int) and ev["seq"] > 0})
    if not seqs:
        return False
    return len(seqs) != seqs[-1] - seqs[0] + 1


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 3)
    text = str(value)
    return safety.redact(text)[:120]


def _entry(ev: dict, category: str, position: int, run_key: str) -> dict:
    seq = ev.get("seq")
    seq_val = int(seq) if isinstance(seq, int) and seq > 0 else None
    ident = f"{run_key}:{seq_val if seq_val is not None else position}"
    out: dict[str, Any] = {
        "id": safety.redact(ident)[:200],
        "seq": seq_val,
        "t": _clean_scalar(ev.get("t")),
        "kind": str(ev.get("kind") or "")[:60],
        "category": category,
    }
    for key in _SCALAR_FIELDS:
        if key in ev:
            value = _clean_scalar(ev[key])
            if isinstance(value, (list, dict)):
                continue
            out[key] = value
    return out


def autopsy(root: Path, run_id: str = "last") -> Autopsy:
    """Classify one run from its typed events. Pure with respect to models:
    it reads the local event logs and nothing else."""
    selected = _select_run(Path(root), run_id)
    if selected is None:
        return Autopsy("unknown", (), _NEXT_STEPS["unknown"], True)
    evs, run_key = selected
    ranked: list[tuple[int, int, str, int, dict]] = []
    for position, ev in enumerate(evs):
        category = typed_category(ev)
        if not category:
            continue
        seq = ev.get("seq")
        seq_val = int(seq) if isinstance(seq, int) and seq > 0 else position
        ranked.append((-SEVERITY[category], seq_val, category, position, ev))
    incomplete = (not any(str(ev.get("kind") or "") in TERMINAL_KINDS
                          for ev in evs)) or _has_seq_gap(evs)
    if not ranked:
        return Autopsy("unknown", (), _NEXT_STEPS["unknown"], True)
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    chosen = ranked[0][2]
    matching = [row for row in ranked if row[2] == chosen]
    matching.sort(key=lambda row: (row[1], row[3]))
    evidence = tuple(_entry(row[4], chosen, row[3], run_key)
                     for row in matching[:_MAX_EVIDENCE])
    steps = _NEXT_STEPS[chosen]
    if incomplete:
        steps = steps + ("check_run_log_completeness",)
    return Autopsy(chosen, evidence, steps, incomplete)


_RENDER_FIELDS = ("file", "action", "gate", "n", "field", "stopped",
                  "obs_code", "code", "errors", "applied", "failed", "ok",
                  "phase", "signal", "rc")


def render_evidence(item: dict) -> str:
    """One evidence record as a plain line (protocol values, not localized)."""
    seq = item.get("seq")
    bits = [f"#{seq if seq is not None else '-'}", str(item.get("kind") or "")]
    for key in _RENDER_FIELDS:
        if key in item:
            bits.append(f"{key}={item[key]}")
    return " ".join(bits)


def payload(result: Autopsy) -> dict:
    """JSON-ready form of an autopsy (stable ASCII ids; render labels with i18n)."""
    return {
        "category": result.category,
        "evidence": [dict(item) for item in result.evidence],
        "next_steps": list(result.next_steps),
        "incomplete": bool(result.incomplete),
    }
