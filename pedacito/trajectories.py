"""Opt-in full-prompt trajectory export for SFT/GRPO dataset filtering.

    pedacito ask . -t "..." --save-trajectory
    # or, in the config file: record_trajectories = true

This is deliberately a different privacy trade from `replay.py`. A replay
recording keeps prompts as a hash plus a short redacted preview so a
recording can be committed without becoming a copy of your source. A
trajectory keeps the *real* conversation the harness sent and received --
system and user prompts, assistant replies, model tool calls and tool results
-- as ChatML/OpenAI-shaped messages, because that is what an SFT or GRPO
dataset needs. It is therefore opt-in and local only:

* one JSON file per task run under ``.pedacito/trajectories/``;
* every string value passes :func:`pedacito.safety.redact`, then the
  serialised transcript is scanned with
  :func:`pedacito.security.scan_text_for_secrets`; a residual finding is
  masked, and when that is not possible the write is refused;
* nothing is sent anywhere and no telemetry is emitted.

The capture hook lives in :func:`pedacito.loop.run_task`, so the config knob
covers every surface (`ask`, `chat`, `auto`, `rpc`, `eval`) and a capture
failure can never fail the task itself.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import events, fileio, imagegen, safety, security

TRAJECTORY_DIR = "trajectories"
_MASK = "[REDACTED]"
_TOKEN_KEYS = ("calls", "prompt_tokens", "completion_tokens", "total_tokens",
               "reasoning_tokens")


def enabled(cfg) -> bool:
    """True when this run should capture a trajectory.

    ``record_trajectories`` is the persistent knob; ``_save_trajectory`` is the
    per-run override set by ``pdc ask --save-trajectory`` (never written back
    to a config file)."""
    return bool(getattr(cfg, "record_trajectories", False)
                or getattr(cfg, "_save_trajectory", False))


def directory(root) -> Path:
    """The trajectory directory for a project (created on the first write)."""
    return Path(root) / ".pedacito" / TRAJECTORY_DIR


def task_hash(task) -> str:
    """SHA-256 of the task text; the text itself is never stored in metadata."""
    return hashlib.sha256(str(task or "").encode("utf-8", "replace")).hexdigest()


def _warn(message: str) -> None:
    try:
        print(f"[trajectories: {message}]", file=sys.stderr)
    except Exception:
        pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _harness_version() -> str:
    try:
        from . import __version__
        return str(__version__)
    except Exception:
        return ""


def _edit_format(cfg) -> str:
    """The resolved edit dialect: ``anchor`` or the default ``quote``."""
    style = str(getattr(cfg, "edit_style", "") or "").strip().lower()
    return style or "quote"


def _scrub_text(text: str) -> str:
    """One string through the same redaction pipeline as logs and sessions,
    plus a value-level masking pass for anything the detector still sees."""
    out = safety.redact(str(text))
    try:
        for finding in security.scan_text_for_secrets(out):
            raw = finding.secret or finding.raw_match
            if raw:
                out = out.replace(raw, _MASK)
    except Exception:
        pass
    return out


def _scrub(value):
    """Recursively redact every string value before serialisation."""
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        return {_scrub_text(str(key)): _scrub(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _scrub_text(str(value))


def _message(value) -> dict:
    """Normalise one conversation entry to a ChatML/OpenAI-shaped dict while
    keeping a native tool call or tool result id when the harness used one."""
    if not isinstance(value, dict):
        return {"role": "user", "content": str(value)}
    content = value.get("content", "")
    if isinstance(content, list):
        content = imagegen.sanitize_multimodal_content(content)
    elif content is not None and not isinstance(content, str):
        content = str(content)
    out = {"role": str(value.get("role") or "user"), "content": content}
    for key in ("name", "tool_calls", "tool_call_id", "function_call"):
        if key in value:
            out[key] = value[key]
    return out


def _reply_message(reply):
    """The assistant turn a call returned, or None when there was no reply."""
    if reply is None:
        return None
    if isinstance(reply, str):
        return {"role": "assistant", "content": reply}
    try:
        text = json.dumps(reply, ensure_ascii=False, default=str)
    except Exception:
        text = str(reply)
    return {"role": "assistant", "content": text}


def _common_prefix(left: list, right: list) -> int:
    n = 0
    while n < len(left) and n < len(right) and left[n] == right[n]:
        n += 1
    return n


def _transcript(calls) -> list:
    """Flatten the calls' sent messages into one ordered conversation.

    Every call normally re-sends the previous conversation plus the new
    assistant step and tool result, so appending the unseen suffix is enough.
    A fresh prompt (answer rebuild) or a rewritten history (folding, repair
    collapse) starts a new segment, with the previous reply kept in place so
    the transcript remains chronological. The final reply is appended when no
    later prompt carried it."""
    out: list = []
    pending = None
    for call in calls:
        messages = [_message(m) for m in call.messages]
        if not out:
            out.extend(messages)
        else:
            n = _common_prefix(out, messages)
            if n < len(out):
                if pending is not None:
                    out.append(pending)
                out.extend(messages[n:] if n else messages)
            else:
                out.extend(messages[n:])
        # A later call usually carries this reply in its prompt; keep it as
        # the candidate for when no later prompt does (the last call, or a
        # fresh answer prompt that discarded the gather conversation).
        pending = _reply_message(call.reply)
    if pending is not None:
        out.append(pending)
    return out


class _Call:
    __slots__ = ("kind", "messages", "reply")

    def __init__(self, kind: str, messages, reply):
        self.kind = kind
        self.messages = messages
        self.reply = reply


class _ClientProxy:
    """A transparent client wrapper that records the messages a call sent and
    the reply it received, then delegates everything else."""

    def __init__(self, client, capture: "Capture"):
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_capture", capture)

    def _call(self, kind: str, messages, fn, *args, **kwargs):
        reply = fn(messages, *args, **kwargs)
        try:
            self._capture.note(kind, messages, reply)
        except Exception:
            pass                # capture is diagnostics, never the call
        return reply

    def chat(self, messages, *args, **kwargs):
        return self._call("chat", messages, self._client.chat,
                          *args, **kwargs)

    def chat_json(self, messages, *args, **kwargs):
        return self._call("chat_json", messages, self._client.chat_json,
                          *args, **kwargs)

    def complete_once(self, messages, *args, **kwargs):
        # One-shot completions are model calls too. The profiler's probe uses
        # its own raw client, but a captured client must not lose a call just
        # because it took the `complete_once` entry point instead of `chat`.
        return self._call("chat", messages, self._client.complete_once,
                          *args, **kwargs)

    def __getattr__(self, name):
        # Special methods are looked up on the class, so chat/chat_json/
        # complete_once above always intercept; every other attribute is a
        # plain delegation (stats, cfg, cancel_event, ...).
        return getattr(self._client, name)

    def __setattr__(self, name, value):
        setattr(self._client, name, value)


class Capture:
    """One run's captured calls, metadata and final write."""

    def __init__(self, root, cfg, task, client):
        self.root = Path(root)
        self.task = str(task or "")
        self.started = _now()
        self._started_mono = time.monotonic()
        self._calls: list[_Call] = []
        self._closed = False
        self._tokens0 = self._event_totals()
        self.model_id = str(getattr(getattr(client, "cfg", None), "model", "")
                            or getattr(cfg, "model", "") or "")
        self.provider = self._provider(client, cfg)

    @staticmethod
    def _provider(client, cfg) -> str:
        try:
            label = getattr(client, "provider_label", None)
            if callable(label):
                return str(label() or "")
        except Exception:
            pass
        role = getattr(client, "active_model_role", None)
        if isinstance(role, dict):
            return str(role.get("profile") or "")
        return str(getattr(cfg, "provider", "")
                   or getattr(cfg, "active_profile", "") or "")

    @staticmethod
    def _event_totals() -> dict:
        try:
            run = events.current_run()
            return run.summary() if run is not None else {}
        except Exception:
            return {}

    def install(self, client, agent):
        """Wrap `client` (and the agent that owns it) and return both."""
        proxy = self.wrap(client)
        try:
            if agent is not None and getattr(agent, "client", None) is client:
                agent.client = proxy
        except Exception:
            pass
        return proxy, agent

    def wrap(self, client):
        """Wrap another client for this capture (the hybrid architect). The
        caller wraps at the call site, never at a cached-client site, so a
        later run starts a fresh proxy for its own capture."""
        return _ClientProxy(client, self)

    def note(self, kind: str, messages, reply) -> None:
        """Record one completed call. Snapshotting must never hurt the call:
        the agent mutates its message list in place between steps."""
        try:
            snapshot = copy.deepcopy(messages)
        except Exception:
            snapshot = messages
        try:
            self._calls.append(_Call(kind, snapshot, reply))
        except Exception:
            pass

    def _tokens(self, tr) -> dict:
        now = self._event_totals()
        if now:
            out = {}
            for key in _TOKEN_KEYS:
                out[key] = max(0, int(now.get(key, 0) or 0)
                               - int(self._tokens0.get(key, 0) or 0))
            return out
        stats = getattr(tr, "stats", None) or {}
        prompt = int(stats.get("prompt_tokens", 0) or 0)
        completion = int(stats.get("completion_tokens", 0) or 0)
        return {
            "calls": int(stats.get("calls", 0) or 0),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": int(stats.get("total_tokens", 0) or 0)
            or prompt + completion,
            "reasoning_tokens": int(stats.get("reasoning_tokens", 0) or 0),
        }

    def write(self, root=None, cfg=None, tr=None) -> Path | None:
        if self._closed:
            return None
        self._closed = True
        try:
            messages = _transcript(self._calls)
            verify = getattr(tr, "tests_ok", None) if tr is not None else None
            provider = self.provider or str(
                self._event_totals().get("provider") or "")
            metadata = {
                "verify_passed": None if verify is None else bool(verify),
                "harness_version": _harness_version(),
                "edit_format": _edit_format(cfg),
                "model_id": self.model_id,
                "provider": provider,
                "task_hash": task_hash(self.task),
                "started_at": self.started.isoformat(timespec="seconds"),
                "elapsed_s": round(time.monotonic() - self._started_mono, 3),
                "tokens": self._tokens(tr),
            }
            payload = _scrub({"metadata": metadata, "messages": messages})
            return _write_payload(
                directory(root or self.root), payload,
                stamp=self.started.strftime("%Y%m%dT%H%M%SZ"))
        except Exception as e:
            _warn(f"could not write trajectory "
                  f"({type(e).__name__}: {e})")
            return None


def begin(root, cfg, task, client=None) -> Capture | None:
    """Start capturing one run, or return None when the knob is off."""
    if not enabled(cfg) or root is None or client is None:
        return None
    try:
        return Capture(root, cfg, task, client)
    except Exception:
        return None


def finish(capture: Capture | None, root=None, cfg=None, tr=None):
    """Write the trajectory. Never raises: capture is diagnostics, not the
    task, and a full disk must not turn a finished edit into a failure."""
    if capture is None:
        return None
    try:
        return capture.write(root, cfg, tr)
    except Exception as e:                 # belt and braces around the hook
        _warn(f"could not write trajectory ({type(e).__name__}: {e})")
        return None


def _write_payload(target_dir: Path, payload: dict,
                   stamp: str | None = None) -> Path | None:
    """Serialise, re-scan the serialised transcript, then write atomically.

    The per-value pass already ran; this second pass catches a finding that
    only appears after JSON escaping. A value that still matches is masked in
    the serialised text; if the result is not valid JSON (or a finding
    survives), no file is written and a warning names the failure without ever
    echoing the secret."""
    text = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    findings = security.scan_text_for_secrets(text)
    if findings:
        for finding in findings:
            for raw in (finding.secret, finding.raw_match):
                if raw:
                    text = text.replace(raw, _MASK)
        remaining = security.scan_text_for_secrets(text)
        if remaining:
            _warn("refusing to write a trajectory: a detected secret "
                  f"survived redaction in {len(remaining)} place(s)")
            return None
        try:
            json.loads(text)
        except ValueError:
            _warn("refusing to write a trajectory: masking a residual "
                  "secret left invalid JSON")
            return None

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or _now().strftime("%Y%m%dT%H%M%SZ")
    path = target_dir / f"{stamp}_{uuid.uuid4().hex[:8]}.json"
    fileio.write_bytes(path, text.encode("utf-8"))
    return path
