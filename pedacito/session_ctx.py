"""Per-session execution context, budget ledger, and contextvars propagation.

Every concurrent session (GUI tab, daemon run, RPC/ACP connection, subagent, or
headless task) carries its own `SessionContext` bound via `contextvars`. Worker
threads and thread pools MUST propagate context explicitly through
`spawn_thread_with_context` and `ContextThreadPoolExecutor`, copying a fresh
`contextvars.Context` per task so two threads never enter the same Context
concurrently.
"""

from __future__ import annotations

import contextvars
import copy
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")

__all__ = [
    "BudgetLedger",
    "ContextThreadPoolExecutor",
    "SessionContext",
    "clone_config",
    "current_session",
    "require_session",
    "run_in_context",
    "spawn_thread_with_context",
    "use_session",
]


@dataclass
class BudgetLedger:
    """Thread-safe cost, token, and step budget tracker for one session."""

    max_cost_usd: float = 0.0
    max_tokens: int = 0
    max_steps: int = 0
    spent_cost_usd: float = 0.0
    spent_tokens: int = 0
    spent_steps: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def record(
        self,
        *,
        cost_usd: float = 0.0,
        tokens: int = 0,
        steps: int = 0,
    ) -> None:
        """Accumulate spent cost, tokens, and steps safely from any thread."""
        with self._lock:
            if cost_usd:
                self.spent_cost_usd += float(cost_usd)
            if tokens:
                self.spent_tokens += int(tokens)
            if steps:
                self.spent_steps += int(steps)

    def _check_locked(self) -> tuple[bool, str]:
        if self.max_cost_usd > 0.0 and self.spent_cost_usd >= self.max_cost_usd:
            return (
                False,
                f"cost budget exceeded (${self.spent_cost_usd:.4f} / ${self.max_cost_usd:.4f})",
            )
        if self.max_tokens > 0 and self.spent_tokens >= self.max_tokens:
            return (
                False,
                f"token budget exceeded ({self.spent_tokens} / {self.max_tokens})",
            )
        if self.max_steps > 0 and self.spent_steps >= self.max_steps:
            return (
                False,
                f"step budget exceeded ({self.spent_steps} / {self.max_steps})",
            )
        return True, ""

    def check(self) -> tuple[bool, str]:
        """Return `(ok, reason_if_exceeded)`."""
        with self._lock:
            return self._check_locked()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of limits, usage, and status."""
        with self._lock:
            ok, reason = self._check_locked()
            return {
                "max_cost_usd": float(self.max_cost_usd),
                "max_tokens": int(self.max_tokens),
                "max_steps": int(self.max_steps),
                "spent_cost_usd": round(float(self.spent_cost_usd), 6),
                "spent_tokens": int(self.spent_tokens),
                "spent_steps": int(self.spent_steps),
                "ok": ok,
                "reason": reason,
            }


def clone_config(cfg: Any) -> Any:
    """Create a shallow per-run copy of `cfg` so run-local attributes never
    mutate sibling sessions. Mutable top-level dict fields (`sources`,
    `shell_allowlist_by_project`) are also shallow-copied when present."""
    if cfg is None:
        return None
    try:
        cloned = copy.copy(cfg)
    except Exception:
        return cfg
    for attr in ("sources", "shell_allowlist_by_project"):
        val = getattr(cloned, attr, None)
        if isinstance(val, dict):
            try:
                setattr(cloned, attr, dict(val))
            except Exception:
                pass
    return cloned


@dataclass
class SessionContext:
    """All runtime state owned by a single isolated session."""

    session_id: str = field(default_factory=lambda: f"ses_{uuid.uuid4().hex[:12]}")
    root: Path | None = None
    cfg: Any = None
    idx: Any = None
    client: Any = None
    event_bus: Any = None
    permission_broker: Any = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    model_lease: Any = None
    budget_ledger: BudgetLedger = field(default_factory=BudgetLedger)
    worktree: Any = None
    policy: Any = None
    review_queue: Any = None
    resource_registry: Any = None
    run_telemetry: Any = None
    tool_caches: dict[str, dict[Any, Any]] = field(default_factory=dict)
    parent_session_id: str = ""
    mode: str = "interactive"  # interactive | rpc | acp | daemon | subagent | headless
    created_at: float = field(default_factory=time.time)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.root is not None and not isinstance(self.root, Path):
            self.root = Path(self.root)
        if not self.created_at:
            self.created_at = time.time()

    def cache(self, namespace: str) -> dict[Any, Any]:
        """Return a session-scoped cache dict for `namespace`."""
        key = str(namespace)
        with self._lock:
            return self.tool_caches.setdefault(key, {})

    def cancel(self, reason: str = "cancelled") -> dict[str, Any]:
        """Cancel the session, denying pending permissions and releasing leases
        and registered resources. Safe to call repeatedly."""
        self.cancel_event.set()
        report: dict[str, Any] = {
            "session_id": self.session_id,
            "reason": str(reason or "cancelled"),
            "permissions_denied": 0,
            "client_cancelled": False,
            "lease_released": False,
            "resources_closed": None,
        }
        with self._lock:
            broker = self.permission_broker
            client = self.client
            lease = self.model_lease
            registry = self.resource_registry
            bus = self.event_bus

        if broker is not None:
            deny_all = getattr(broker, "deny_all", None)
            if callable(deny_all):
                try:
                    report["permissions_denied"] = int(deny_all(reason) or 0)
                except Exception:
                    pass

        if client is not None:
            cancel_fn = getattr(client, "cancel", None)
            if callable(cancel_fn):
                try:
                    cancel_fn()
                    report["client_cancelled"] = True
                except Exception:
                    pass

        if lease is not None:
            release_fn = getattr(lease, "release", None)
            if callable(release_fn):
                try:
                    release_fn()
                    report["lease_released"] = True
                except Exception:
                    pass

        if registry is not None:
            close_fn = getattr(registry, "close_all", None) or getattr(
                registry, "close", None
            )
            if callable(close_fn):
                try:
                    report["resources_closed"] = close_fn()
                except Exception:
                    pass

        if bus is not None and not getattr(bus, "closed", False):
            emit_fn = getattr(bus, "emit", None)
            if callable(emit_fn):
                try:
                    emit_fn("session_cancelled", reason=report["reason"])
                except Exception:
                    pass

        return report

    def close(self, reason: str = "closed") -> dict[str, Any]:
        """Cancel the session and close its event bus."""
        report = self.cancel(reason=reason)
        with self._lock:
            bus = self.event_bus
        if bus is not None:
            close_fn = getattr(bus, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        report["closed"] = True
        return report


_CURRENT_SESSION: contextvars.ContextVar[SessionContext | None] = (
    contextvars.ContextVar("pedacito_current_session", default=None)
)


def current_session() -> SessionContext | None:
    """Return the active `SessionContext` for the current execution context."""
    return _CURRENT_SESSION.get()


def require_session() -> SessionContext:
    """Return the active `SessionContext`, or raise `RuntimeError` if unbound."""
    ctx = _CURRENT_SESSION.get()
    if ctx is None:
        raise RuntimeError("No active SessionContext is bound in this context.")
    return ctx


@contextmanager
def use_session(ctx: SessionContext) -> Iterator[SessionContext]:
    """Bind `ctx` as the active `SessionContext` for the duration of the block."""
    token = _CURRENT_SESSION.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT_SESSION.reset(token)


def run_in_context(
    ctx: contextvars.Context, fn: Callable[..., T], *args: Any, **kwargs: Any
) -> T:
    """Execute `fn(*args, **kwargs)` inside `ctx`."""
    return ctx.run(fn, *args, **kwargs)


def spawn_thread_with_context(
    target: Callable[..., Any],
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    name: str | None = None,
    daemon: bool = True,
    ctx: contextvars.Context | None = None,
) -> threading.Thread:
    """Spawn and start a `threading.Thread` with an isolated copy of the
    current (or supplied) `contextvars.Context`."""
    captured = contextvars.copy_context() if ctx is None else ctx.copy()
    kw = dict(kwargs) if kwargs else {}
    thread = threading.Thread(
        target=run_in_context,
        args=(captured, target, *args),
        kwargs=kw,
        name=name,
        daemon=daemon,
    )
    thread.start()
    return thread


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that copies `contextvars.copy_context()` on every
    `submit` (and therefore every `map`) call so each task gets its own
    Context copy without concurrent-entry errors."""

    def submit(
        self, fn: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> Future[T]:
        captured = contextvars.copy_context()
        return super().submit(captured.run, fn, *args, **kwargs)
