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
    "RunBudget",
    "RunBudgetError",
    "SessionContext",
    "clone_config",
    "current_run_budget",
    "current_session",
    "require_session",
    "run_in_context",
    "set_run_budget",
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


class RunBudgetError(RuntimeError):
    """A run budget refused a call or was exhausted (deadline or strict cost).

    Raised by :class:`RunBudget` when a deadline has passed, a strict cost cap
    can no longer be guaranteed (unknown pricing/usage, unbounded output, an
    unknown reservation), or a reservation would exceed the cap. Callers that
    already understand ``llm.BudgetExceeded`` catch this type through a small
    adapter (``llm.py`` re-raises it as ``BudgetExceeded`` at the call site).
    """

    kind = "budget"


def _positive_price(value: Any) -> float | None:
    """A strictly-positive USD-per-million-token price, else None."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0.0 else None


@dataclass
class RunBudget:
    """One run's deadline and strict cost reservation state.

    Session-scoped and synchronized: every method takes an internal lock, so a
    worker thread, the GUI stop thread, and the teardown path can all consult
    the same budget without corrupting its counters. The dataclass is mutable
    *by design* -- ``reserved_usd`` and ``spent_usd`` are live counters, and
    the reservation table and poison flag are runtime state -- but the
    mutation surface is exactly the three methods below, all under the lock.

    ``deadline_monotonic`` is a ``time.monotonic()`` timestamp that starts
    *before* model load; ``remaining_seconds()`` and ``check()`` read it so a
    deadline governs leases, backoff, streaming, gates, tests, teardown and
    child launches without any wall-clock drift.

    A strict cost cap is ``max_cost_usd > 0``. In strict mode every potentially
    billable call must be reserved first with known pricing and a bounded
    output, released only once usage is known, and any unknown pricing, unknown
    usage, unbounded output or over-budget reservation fails closed. With
    ``max_cost_usd <= 0`` (uncapped) ``reserve_call``/``settle_call`` are
    permissive and the run retains legacy (estimate-only) behavior.
    """

    deadline_monotonic: float | None = None
    max_cost_usd: float = 0.0
    reserved_usd: float = 0.0
    spent_usd: float = 0.0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False)
    _reservations: dict[str, float] = field(
        default_factory=dict, repr=False, compare=False)
    _poisoned: bool = field(default=False, repr=False, compare=False)

    @property
    def strict(self) -> bool:
        """True when this run enforces a hard (reservation-based) cost cap."""
        return float(self.max_cost_usd or 0.0) > 0.0

    def remaining_seconds(self) -> float | None:
        """Seconds until the deadline (floor 0), or None when uncapped."""
        if self.deadline_monotonic is None:
            return None
        return max(0.0, float(self.deadline_monotonic) - time.monotonic())

    def _check_locked(self) -> None:
        if self._poisoned:
            raise RunBudgetError(
                "The strict cost cap can no longer be guaranteed (a call ended "
                "with unknown pricing or usage); refusing to keep running a "
                "strict cap on estimates.")
        if self.deadline_monotonic is not None \
                and time.monotonic() >= float(self.deadline_monotonic):
            raise RunBudgetError(
                "The run deadline has passed; no further billable work is "
                "allowed.")
        if self.strict and self.spent_usd + self.reserved_usd \
                >= float(self.max_cost_usd):
            raise RunBudgetError(
                f"Cost budget exhausted (${self.spent_usd + self.reserved_usd:.4f}"
                f" spent+reserved of ${float(self.max_cost_usd):.4f}).")

    def check(self) -> None:
        """Raise :class:`RunBudgetError` when the deadline passed or (strict)
        the cost cap is exhausted."""
        with self._lock:
            self._check_locked()

    def reserve_call(self, input_tokens: int, max_output_tokens: int,
                     prices: dict) -> str:
        """Reserve a conservative maximum cost for one billable call.

        ``prices`` maps ``"input"`` / ``"output"`` (and optional ``"cached"``
        / ``"cache_write"``) to USD per million tokens. In strict mode the
        conservative maximum assumes every input token bills at the most
        expensive input price and every output token bills at the output price;
        unknown pricing, an unbounded (non-positive) output cap, or a
        reservation that would exceed the cap raise :class:`RunBudgetError`
        (and poison a strict budget) instead of estimating. Returns an opaque
        reservation token for :meth:`settle_call`.
        """
        with self._lock:
            self._check_locked()
            token = f"res_{uuid.uuid4().hex}"
            if not self.strict:
                self._reservations[token] = 0.0
                return token
            pin = _positive_price((prices or {}).get("input"))
            pout = _positive_price((prices or {}).get("output"))
            if pin is None or pout is None:
                self._poisoned = True
                raise RunBudgetError(
                    "Strict cost cap: input/output pricing is unknown, so a "
                    "conservative maximum cannot be reserved. Set the price_* "
                    "fields (USD per million tokens) or drop the strict cap.")
            mot = int(max_output_tokens or 0)
            if mot <= 0:
                self._poisoned = True
                raise RunBudgetError(
                    "Strict cost cap: the call has an unbounded (or unknown) "
                    "output limit, so its maximum cost cannot be reserved.")
            most = max(
                pin,
                _positive_price((prices or {}).get("cache_write")) or pin,
                _positive_price((prices or {}).get("cached")) or pin,
            )
            cost = (max(0, int(input_tokens or 0)) * most
                    + mot * pout) / 1_000_000
            if self.spent_usd + self.reserved_usd + cost \
                    > float(self.max_cost_usd):
                raise RunBudgetError(
                    f"Strict cost cap: reserving this call "
                    f"(~${cost:.6f}) would exceed the ${float(self.max_cost_usd):.4f}"
                    f" budget (${self.spent_usd + self.reserved_usd:.4f} already "
                    f"spent or reserved).")
            self._reservations[token] = cost
            self.reserved_usd += cost
            return token

    def settle_call(self, reservation: str, actual_usd: float | None) -> None:
        """Release a reservation, charging ``actual_usd`` when known.

        Strict mode refuses ``actual_usd=None`` (unknown usage after a call)
        and an unknown reservation token: either poisons the budget so no
        further call proceeds on an estimate. Uncapped runs only record known
        usage for observability and never refuse.
        """
        with self._lock:
            cost = self._reservations.pop(str(reservation), None)
            if cost is None:
                if self.strict:
                    self._poisoned = True
                    raise RunBudgetError(
                        "Strict cost cap: cannot settle an unknown reservation "
                        "token; refusing to keep running a strict cap.")
                return
            self.reserved_usd = max(0.0, self.reserved_usd - float(cost))
            if actual_usd is None:
                if self.strict:
                    self._poisoned = True
                    raise RunBudgetError(
                        "Strict cost cap: the call reported no usage, so its "
                        "actual cost is unknown; refusing to keep running a "
                        "strict cap on estimates.")
                return
            self.spent_usd += max(0.0, float(actual_usd))

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe summary of the budget's limits and live state."""
        with self._lock:
            return {
                "deadline_monotonic": (
                    float(self.deadline_monotonic)
                    if self.deadline_monotonic is not None else None),
                "remaining_seconds": self.remaining_seconds(),
                "max_cost_usd": float(self.max_cost_usd),
                "reserved_usd": round(float(self.reserved_usd), 6),
                "spent_usd": round(float(self.spent_usd), 6),
                "strict": self.strict,
                "ok": not self._poisoned,
                "poisoned": self._poisoned,
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
    run_budget: RunBudget | None = None
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
        if self.resource_registry is None:
            # Lazy import: resources imports nothing from session_ctx, so this
            # stays cycle-free even at first-import time.
            from . import resources
            self.resource_registry = resources.SessionResourceRegistry(
                self.session_id)
        if self.review_queue is None:
            # D9: every session owns its own review routing state. The import
            # is lazy (review imports security, which imports this module).
            try:
                from . import review as review_mod
                self.review_queue = review_mod.ReviewQueue(self.session_id)
            except Exception:           # pragma: no cover - import cycle guard
                self.review_queue = None

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
            "reviews_cancelled": 0,
        }
        with self._lock:
            broker = self.permission_broker
            client = self.client
            lease = self.model_lease
            registry = self.resource_registry
            bus = self.event_bus
            review_queue = self.review_queue

        # D9: cancelling a session unblocks any review a worker is waiting on
        # (the queue's `done` event is set by `cancel_all`). A sibling session's
        # queue is untouched.
        if review_queue is not None:
            cancel_fn = getattr(review_queue, "cancel_all", None)
            if callable(cancel_fn):
                try:
                    report["reviews_cancelled"] = int(
                        cancel_fn(report["reason"]) or 0)
                except Exception:
                    pass

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
            # WP-4003: the deterministic cascade is `teardown_all` (priority
            # grouped LIFO). Keep the WP-4001 `close_all`/`close` names as a
            # fallback for any older or mocked registry.
            teardown_fn = getattr(registry, "teardown_all", None)
            if callable(teardown_fn):
                try:
                    report["resources_closed"] = teardown_fn(reason=report["reason"])
                except Exception:
                    pass
            else:
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


def current_run_budget(cfg: Any = None) -> RunBudget | None:
    """The active run's :class:`RunBudget`, or None when none is bound.

    Checks the bound session first (the session-scoped source of truth), then a
    ``cfg._run_budget`` fallback set by headless CLI runs that never bind a
    ``SessionContext``. Used by the LLM client, lease manager and process runner
    to clamp request timeouts, backoff and child launches to the run's
    remaining deadline, and to reserve/settle strict cost caps. Never raises.
    """
    ctx = _CURRENT_SESSION.get()
    if ctx is not None and ctx.run_budget is not None:
        return ctx.run_budget
    if cfg is not None:
        budget = getattr(cfg, "_run_budget", None)
        if budget is not None:
            return budget
    return None


def set_run_budget(budget: RunBudget, cfg: Any = None) -> None:
    """Bind ``budget`` as the active run budget: on the bound session when one
    exists (session-scoped), and on ``cfg._run_budget`` as a headless fallback.
    """
    ctx = _CURRENT_SESSION.get()
    if ctx is not None:
        ctx.run_budget = budget
    if cfg is not None:
        try:
            cfg._run_budget = budget
        except Exception:
            pass


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
