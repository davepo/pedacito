"""User-command hooks: run your own program at well-defined moments.

    hooks = { post_edit = ["ruff format {file}"], pre_edit = ["./scripts/guard.sh"] }

Events: session_start, pre_task, post_task, pre_tool, post_tool, pre_edit,
post_edit, pre_run, post_run, pre_compact, post_compact, notification.

Protocol (no plugin ABI): the hook receives ONE JSON object on stdin
(`{"event", "cwd", ...event fields}`) and may answer on stdout with JSON:

    {"decision": "allow" | "deny", "message": "...", "context": "..."}

`pre_*` hooks can deny (a `deny` decision, or exit status 2 with the reason on
stderr); text a hook prints (the `context` field, or plain stdout) is fed back
to the model after `post_*` events. Hooks run as an argv -- never through a
shell -- in the project root, with the scrubbed environment, a timeout
(`hook_timeout`), and `{file}` / `{files}` placeholders expanded from the
event. A hook that fails or times out is a warning, not an error. Hooks from a
project's own `.pedacito.toml` only run after `pedacito trust` (config.py).
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from . import proc, safety

EVENTS = ("session_start", "pre_task", "post_task", "pre_tool", "post_tool",
          "pre_edit", "post_edit", "pre_run", "post_run", "pre_compact",
          "post_compact", "notification")


@dataclass
class HookResult:
    decision: str = "allow"        # allow | deny
    message: str = ""
    context: str = ""

    @property
    def denied(self) -> bool:
        return self.decision == "deny"


def _commands(cfg, event: str) -> list[str]:
    table = getattr(cfg, "hooks", None) or {}
    raw = table.get(event)
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(c) for c in raw if str(c).strip()]


def _argv(command: str, payload: dict) -> list[str]:
    try:
        parts = shlex.split(command) if sys.platform != "win32" else \
            [p.strip("\"'") for p in shlex.split(command, posix=False)]
    except ValueError:
        return []
    files = payload.get("files") or ([payload["file"]] if payload.get("file") else [])
    out: list[str] = []
    for p in parts:
        if p == "{files}":
            out.extend(files)
        else:
            out.append(p.replace("{file}", str(payload.get("file") or
                                               (files[0] if files else "")))
                        .replace("{event}", str(payload.get("event", ""))))
    return out


def has_hooks(cfg, event: str) -> bool:
    return bool(_commands(cfg, event))


def fire(cfg, root, event: str, **fields) -> HookResult:
    """Run every command registered for `event`; the first deny wins, and
    contexts are concatenated. Never raises."""
    result = HookResult()
    cmds = _commands(cfg, event)
    if not cmds:
        return result
    payload = {"event": event, "cwd": str(root), **fields}
    stdin = json.dumps(payload, default=str)
    timeout = max(1, int(getattr(cfg, "hook_timeout", 10) or 10))
    contexts: list[str] = []
    for command in cmds:
        argv = _argv(command, payload)
        if not argv:
            print(f"[hook] {event}: could not parse `{command}`",
                  file=sys.stderr)
            continue
        try:
            rc, out, stopped, _el = proc.run(
                argv, cwd=Path(root), timeout=timeout, idle_timeout=0,
                max_output=4000, stdin_text=stdin,
                env=dict(safety.child_env(cfg), PEDACITO_EVENT=event))
        except Exception as e:              # a hook must never kill a run
            print(f"[hook] {event}: `{command}` failed ({e})", file=sys.stderr)
            continue
        if stopped:
            print(f"[hook] {event}: `{command}` stopped ({stopped}) after "
                  f"{timeout}s", file=sys.stderr)
            continue
        text = (out or "").strip()
        decision, message, context = "allow", "", ""
        if text.startswith("{") or text.endswith("}"):
            obj = None
            for candidate in (text, text.splitlines()[-1] if text else ""):
                try:
                    obj = json.loads(candidate)
                    break
                except ValueError:
                    continue
            if isinstance(obj, dict):
                decision = str(obj.get("decision") or "allow").lower()
                message = str(obj.get("message") or "")
                context = str(obj.get("context") or "")
                text = ""
        if rc == 2 and event.startswith("pre_"):
            decision, message = "deny", message or text or "denied by hook"
            text = ""
        elif rc not in (0, 2) and rc is not None:
            print(f"[hook] {event}: `{command}` exited {rc}", file=sys.stderr)
        if text and event.startswith("post_"):
            context = context or text
        if decision == "deny" and event.startswith("pre_"):
            return HookResult("deny", message or "denied by hook",
                              "\n".join(contexts))
        if context:
            contexts.append(context[:800])
    result.context = "\n".join(contexts)
    return result
