"""Session Resource Registry & deterministic cancellation cascade (D13).

Every session owns a set of resources it spawned or created: subprocesses,
dev servers, browser sessions, LSP clients, MCP server children, temp mirrors,
git worktrees, and review servers. Cancelling a session must reap EXACTLY its
own resources -- in the right order -- while a sibling session's resources stay
alive.

The registry is deliberately tiny and dependency-free (stdlib only): callers
register a `cleanup` callable when they acquire a resource, and `teardown_all`
runs every cleanup in priority-grouped LIFO order, catching every error so one
stuck resource can never block the others.

Windows teardown ordering matters: ``git worktree remove`` / ``shutil.rmtree``
on a temp mirror FAIL while a child process inside it still holds a file handle
or its CWD. So phase 1 tears down every active process/server first (LIFO),
then phase 2 removes filesystem artifacts (temp_mirror, worktree) after all the
processes are reaped.
"""

from __future__ import annotations

import atexit
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import Any, Callable

RESOURCE_KINDS = (
    "subprocess", "dev_server", "browser", "lsp", "mcp",
    "temp_mirror", "worktree", "review_server", "custom",
)

# Teardown priority groups: lower first. Phase 1 = live processes/servers that
# may hold file handles or a CWD inside a mirror/worktree; phase 2 = the
# filesystem artifacts those processes must stop touching before removal.
_PHASE = {
    "subprocess": 1,
    "dev_server": 1,
    "browser": 1,
    "lsp": 1,
    "mcp": 1,
    "review_server": 1,
    "custom": 1,
    "temp_mirror": 2,
    "worktree": 2,
}


@dataclass
class ManagedResource:
    """One registered resource and the callable that releases it."""

    resource_id: str
    session_id: str
    kind: str
    label: str
    cleanup: Callable[[], None]
    created_at: float


class SessionResourceRegistry:
    """Per-session resource tracker with deterministic, idempotent teardown."""

    def __init__(self, session_id: str) -> None:
        self.session_id = str(session_id)
        self._lock = threading.RLock()
        self._entries: list[ManagedResource] = []
        # Every live registry joins the global weak set so an unclean process
        # exit (a kill mid-run, an unhandled exception) still reaps resources.
        with _ALL_LOCK:
            _ALL_REGISTRIES.add(self)

    def register(
        self,
        kind: str,
        label: str,
        cleanup: Callable[[], None],
        *,
        resource_id: str | None = None,
    ) -> str:
        """Add `cleanup` and return the resource id it is stored under."""
        rid = str(resource_id) if resource_id is not None else (
            f"{kind}-{uuid.uuid4().hex[:12]}")
        res = ManagedResource(
            resource_id=rid,
            session_id=self.session_id,
            kind=str(kind),
            label=str(label or ""),
            cleanup=cleanup,
            created_at=time.time(),
        )
        with self._lock:
            self._entries.append(res)
        return rid

    def unregister(self, resource_id: str) -> None:
        """Forget one resource WITHOUT running its cleanup (it was released)."""
        rid = str(resource_id)
        with self._lock:
            for i, res in enumerate(self._entries):
                if res.resource_id == rid:
                    self._entries.pop(i)
                    return

    def list_resources(self) -> list[dict[str, Any]]:
        """JSON-safe snapshot of everything currently registered."""
        with self._lock:
            return [
                {
                    "resource_id": r.resource_id,
                    "session_id": r.session_id,
                    "kind": r.kind,
                    "label": r.label,
                    "created_at": r.created_at,
                }
                for r in self._entries
            ]

    def teardown_all(self, reason: str = "cancelled") -> dict[str, Any]:
        """Run every cleanup in priority-grouped LIFO order; never raises.

        Phase 1 (active processes/servers) runs before phase 2 (filesystem
        artifacts), LIFO within each phase, so a child process is reaped before
        the mirror or worktree it lives in is removed. Every exception is
        caught into ``errors`` and the entries are cleared, so a repeated call
        is a harmless no-op.
        """
        with self._lock:
            entries = list(self._entries)
            self._entries.clear()

        # sort: phase ascending, then registration index DESCENDING (LIFO).
        ordered = sorted(
            enumerate(entries),
            key=lambda item: (_PHASE.get(item[1].kind, 1), -item[0]),
        )
        torn_down: list[str] = []
        errors: list[dict[str, Any]] = []
        for _, res in ordered:
            try:
                res.cleanup()
                torn_down.append(res.resource_id)
            except Exception as exc:  # noqa: BLE001 - one stuck resource
                errors.append({
                    "resource_id": res.resource_id,
                    "kind": res.kind,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return {"torn_down": torn_down, "errors": errors}


# ---------------------------------------------------------------------------
# Global registry set + single atexit handler
# ---------------------------------------------------------------------------
_ALL_LOCK = threading.Lock()
_ALL_REGISTRIES: "weakref.WeakSet[SessionResourceRegistry]" = weakref.WeakSet()


def _teardown_all_registries() -> None:
    """Unclean-exit safety net: reap every still-live session's resources."""
    with _ALL_LOCK:
        registries = list(_ALL_REGISTRIES)
    for registry in registries:
        try:
            registry.teardown_all(reason="atexit")
        except Exception:           # pragma: no cover - best effort
            pass


atexit.register(_teardown_all_registries)
