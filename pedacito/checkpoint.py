"""Conversation checkpoints for `ask` / `chat`.

    from . import checkpoint
    checkpoint.save(root, "chat", task, agent._messages, reply, cfg)
    cp = checkpoint.latest(root, "chat")
    print(checkpoint.render(cp))

A checkpoint is a small JSON record under `.pedacito/checkpoints/`: the task,
the visible conversation messages, the final reply, model/profile, and a few
task-result flags. It is enough to resume with `ask --continue` or
`chat --resume` without replaying the whole transcript. Files stay local and
are redacted by the caller when session logging/redaction is enabled.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import safety, workspace
from .index import state_dir

CHECKPOINT_DIR = "checkpoints"
VERSION = 1


def _dir(root) -> Path:
    return state_dir(Path(root), CHECKPOINT_DIR)


def _safe_cmd(cmd: str) -> str:
    return "".join(c for c in (cmd or "task").lower()
                   if c.isalnum() or c in "-_") or "task"


def save(root, cmd: str, task: str, messages, reply: str,
         cfg=None, extra: dict | None = None) -> Path | None:
    """Write one checkpoint. Returns its path, or None when session logging is
    disabled. Existing files are never overwritten (a numeric suffix is added)."""
    if cfg is not None and not getattr(cfg, "log_sessions", True):
        return None
    root = Path(root).resolve()
    d = _dir(root)
    d.mkdir(parents=True, exist_ok=True)
    stamp = workspace.stamp()
    path = d / f"{stamp}-{_safe_cmd(cmd)}.json"
    n = 2
    while path.exists():
        path = d / f"{stamp}-{_safe_cmd(cmd)}-{n}.json"
        n += 1
    redact = bool(getattr(cfg, "redact_secrets", True)) if cfg else False

    def clean(text) -> str:
        text = str(text or "")
        return safety.redact(text) if redact else text

    msgs = []
    for m in list(messages or [])[-60:]:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(p.get("text") or "") for p in content
                if isinstance(p, dict))
        msgs.append({"role": str(m.get("role") or ""),
                     "content": clean(content)[:6000]})
    rec = {
        "version": VERSION,
        "cmd": cmd,
        "task": clean(task)[:4000],
        "model": getattr(cfg, "model", "") if cfg else "",
        "profile": getattr(cfg, "active_profile", "") if cfg else "",
        "ts": time.time(),
        "messages": msgs,
        "reply": clean(reply)[:20000],
    }
    if extra:
        for k, v in extra.items():
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                rec[k] = v
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def paths(root, cmd: str = "") -> list[Path]:
    """Checkpoint files, newest first; optionally filtered by command."""
    d = _dir(Path(root))
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json"), reverse=True):
        if not cmd:
            out.append(p)
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if rec.get("cmd") == cmd:
            out.append(p)
    return out


def load(root, name: str = "") -> dict | None:
    """Load a named checkpoint (basename only) or the newest one."""
    root = Path(root).resolve()
    if name:
        base = Path(str(name)).name
        p = _dir(root) / base
        if not p.is_file() and not base.endswith(".json"):
            p = _dir(root) / (base + ".json")
    else:
        found = paths(root)
        p = found[0] if found else None
    if p is None or not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or rec.get("version") != VERSION:
        return None
    rec["_path"] = str(p)
    return rec


def latest(root, cmd: str = "") -> dict | None:
    """Newest checkpoint, optionally for one command (`ask` / `chat`)."""
    found = paths(root, cmd)
    return load(root, found[0].name) if found else None


def render(cp: dict | None, max_chars: int = 3000) -> str:
    """Compact continuation context: the prior task, reply, and the last few
    conversation turns. Empty for a missing checkpoint."""
    if not cp:
        return ""
    bits = [("PRIOR CONVERSATION (resume/continue; verify current source "
             "before editing):")]
    if cp.get("task"):
        bits.append("Previous task: " + str(cp["task"]))
    reply = str(cp.get("reply") or "").strip()
    if reply:
        bits.append("Previous reply:\n" + reply)
    # The last few turns add continuity without replaying a huge transcript.
    turns = []
    for m in (cp.get("messages") or [])[-6:]:
        role = str(m.get("role") or "?")
        text = str(m.get("content") or "").strip()
        if not text:
            continue
        if len(text) > 700:
            text = text[:700] + "..."
        turns.append(f"[{role}] {text}")
    if turns:
        bits.append("Recent turns:\n" + "\n".join(turns))
    text = "\n\n".join(bits).strip()
    limit = max(300, int(max_chars or 3000))
    if len(text) > limit:
        suffix = "\n[... checkpoint truncated ...]"
        text = text[:max(0, limit - len(suffix))].rstrip() + suffix
    return text


def render_list(root, limit: int = 10) -> str:
    """Human-readable list of recent checkpoints."""
    found = paths(root)[:max(1, int(limit or 10))]
    if not found:
        return ("No conversation checkpoints. They are written after "
                "`ask`/`chat` tasks when log_sessions is on.")
    lines = []
    for p in found:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        task = str(rec.get("task") or "").splitlines()[0][:80]
        lines.append(f"{p.name:<34} {rec.get('cmd', '?'):<5} {task}")
    return "\n".join(lines)
