"""Local-GPU model lease manager (D5).

Concurrent sessions can thrash a single GPU's KV cache by loading and
unloading models mid-conversation. This module is the local control plane
that serialises access to one local model pool:

    mgr = default_lease_manager()
    with mgr.acquire(session_id, base_url, model_id) as lease:
        ...   # the model is resident for this session until release

Rules (all best-effort, all local, no network in the hot path):

  * Hosted endpoints (``not servers.is_local_url``) bypass GPU serialization
    entirely and return an immediate lightweight lease, tracked only for
    accounting/telemetry.
  * Local endpoints share the pool lock (one pool when ``single_gpu_pool`` is
    true, else one pool per ``server_url``).
  * An active *exclusive* lease (calibrate, models scan) blocks every new
    local request until it releases; it never preempts an active lease.
  * The requested ``ModelKey`` already resident and non-exclusive is shared:
    the refcount increments immediately (cache affinity). A caller for a
    *different* model waits until the resident model's refcount drops to
    zero, so an active session's model is never evicted from under it.
  * Among queued waiters, those matching the last-resident model get a
    cache-affinity priority boost, bounded against starvation by a wait-time
    aging term.
  * ``do_not_evict`` and ``pin``/``unpin`` protect models from the auto-evict
    handshake (``can_evict``), which ``servers.evict_other_local_servers`` and
    ``servers.unload_server_model`` consult.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class LeasePriority(IntEnum):
    LOW = 10          # background indexing / probes
    NORMAL = 20       # interactive / rpc / acp tasks
    HIGH = 30         # user-triggered urgent repair / foreground
    EXCLUSIVE = 100   # calibrate, models scan, explicit unload


@dataclass(frozen=True)
class ModelKey:
    """One canonical (server, model) identity for a lease."""

    server_url: str      # servers.normalize_url(base_url)
    model_id: str


# Cache-affinity priority boost for waiters matching the last-resident model,
# and the anti-starvation aging terms (priority points).
CACHE_AFFINITY_BOOST = 10.0
AGING_RATE = 10.0        # points per second of waiting
AGING_CAP = 100.0
# A default acquire times out after this many seconds.
DEFAULT_TIMEOUT_S = 300.0


@dataclass
class LeaseHandle:
    """A held lease. Release it (or use it as a context manager) to give the
    pool back. Hosted leases are no-ops for scheduling but still tracked."""

    lease_id: str
    session_id: str
    key: ModelKey
    priority: LeasePriority
    exclusive: bool
    do_not_evict: bool
    pinned: bool
    acquired_at: float
    _manager: "ModelLeaseManager" = field(repr=False)

    def release(self) -> None:
        if self._manager is not None:
            self._manager.release(self.lease_id)

    def __enter__(self) -> "LeaseHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


@dataclass
class _LeaseRecord:
    """Internal bookkeeping for one granted lease."""

    lease_id: str
    session_id: str
    key: ModelKey
    priority: LeasePriority
    exclusive: bool
    do_not_evict: bool
    pinned: bool
    acquired_at: float
    pool_key: str | None = None     # None -> hosted (no pool)


@dataclass
class _Waiter:
    """A queued acquisition request waiting for the pool."""

    key: ModelKey
    priority: LeasePriority
    exclusive: bool
    do_not_evict: bool
    pinned: bool
    session_id: str
    mc: int
    enqueued_at: float
    seq: int
    granted: bool = False
    record: "_LeaseRecord | None" = None


@dataclass
class _Pool:
    """One serialisation domain (a single GPU pool or one server)."""

    key: str
    exclusive: "_LeaseRecord | None" = None
    active: list = field(default_factory=list)   # non-exclusive _LeaseRecord
    resident: "ModelKey | None" = None
    resident_mc: int = 0
    last_resident: "ModelKey | None" = None
    waiters: list = field(default_factory=list)  # _Waiter


class ModelLeaseManager:
    """Thread-safe lease manager for local model servers.

    ``single_gpu_pool=True`` (the default) treats every local endpoint as one
    GPU: at most one distinct model is resident at a time. ``False`` gives
    each ``server_url`` its own pool.
    """

    def __init__(self, *, single_gpu_pool: bool = True) -> None:
        self.single_gpu_pool = bool(single_gpu_pool)
        self._cond = threading.Condition()
        self._pools: dict[str, _Pool] = {}
        self._leases: dict[str, _LeaseRecord] = {}
        self._pinned: set[ModelKey] = set()
        self._seq = 0
        self._clock = time.monotonic

    # ------------------------------------------------------------- identity
    def _new_id(self) -> str:
        return f"lease_{uuid.uuid4().hex}"

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _pool_key(self, server_url: str) -> str:
        return "_gpu_pool" if self.single_gpu_pool else server_url

    def _pool(self, pool_key: str) -> _Pool:
        pool = self._pools.get(pool_key)
        if pool is None:
            pool = _Pool(pool_key)
            self._pools[pool_key] = pool
        return pool

    def _aging(self, waiter: _Waiter, now: float) -> float:
        return min(AGING_CAP, max(0.0, now - waiter.enqueued_at) * AGING_RATE)

    # ------------------------------------------------------------- acquire
    def _max_concurrency(self, cfg: Any, key: ModelKey) -> int:
        mc = 0
        if cfg is not None:
            mc = int(getattr(cfg, "max_concurrency", 0) or 0)
        if mc > 0:
            return mc
        # A tiered local-MoE server is single-stream by design.
        try:
            from . import config as config_mod
            if config_mod.is_local_moe(cfg, model=key.model_id,
                                       base_url=key.server_url):
                return 1
        except Exception:
            pass
        return 0          # 0 -> shared access for the same ModelKey

    def acquire(
        self,
        session_id: str,
        base_url: str,
        model_id: str,
        *,
        cfg: Any = None,
        priority: LeasePriority = LeasePriority.NORMAL,
        exclusive: bool = False,
        do_not_evict: bool = False,
        pin: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        cancel_event: threading.Event | None = None,
    ) -> LeaseHandle:
        """Acquire a lease for ``(base_url, model_id)``.

        Hosted endpoints return immediately. Local endpoints serialize on the
        pool; ``exclusive`` drains all active leases first. Raises
        ``TimeoutError`` after ``timeout_s`` and ``RuntimeError`` when
        ``cancel_event`` is set.
        """
        from . import servers as servers_mod

        url = servers_mod.normalize_url(base_url)
        key = ModelKey(url, model_id)
        if not servers_mod.is_local_url(base_url):
            return self._acquire_hosted(session_id, key, priority, exclusive,
                                        do_not_evict, pin)
        return self._acquire_local(session_id, key, priority, exclusive,
                                   do_not_evict, pin, timeout_s, cancel_event,
                                   cfg)

    def _acquire_hosted(self, session_id: str, key: ModelKey,
                        priority: LeasePriority, exclusive: bool,
                        do_not_evict: bool, pin: bool) -> LeaseHandle:
        with self._cond:
            rec = _LeaseRecord(self._new_id(), session_id, key, priority,
                               exclusive, do_not_evict, pin, self._clock(), None)
            self._leases[rec.lease_id] = rec
        return self._handle(rec)

    def _acquire_local(self, session_id: str, key: ModelKey,
                       priority: LeasePriority, exclusive: bool,
                       do_not_evict: bool, pin: bool, timeout_s: float,
                       cancel_event: threading.Event | None,
                       cfg: Any) -> LeaseHandle:
        mc = self._max_concurrency(cfg, key)
        deadline = None
        if timeout_s and timeout_s > 0:
            deadline = self._clock() + float(timeout_s)
        with self._cond:
            pool = self._pool(self._pool_key(key.server_url))
            waiter = _Waiter(key, priority, exclusive, do_not_evict, pin,
                             session_id, mc, self._clock(), self._next_seq())
            pool.waiters.append(waiter)
            self._drain(pool)
            while not waiter.granted:
                if cancel_event is not None and cancel_event.is_set():
                    self._drop_waiter(pool, waiter)
                    self._drain(pool)
                    raise RuntimeError("lease acquisition cancelled")
                if deadline is not None and self._clock() >= deadline:
                    self._drop_waiter(pool, waiter)
                    self._drain(pool)
                    raise TimeoutError(
                        f"timed out waiting for model lease on "
                        f"{key.server_url}/{key.model_id}")
                wait = 0.05
                if deadline is not None:
                    wait = min(wait, max(0.0, deadline - self._clock()))
                self._cond.wait(timeout=wait)
            rec = waiter.record
            assert rec is not None
            self._leases[rec.lease_id] = rec
        return self._handle(rec)

    def _handle(self, rec: _LeaseRecord) -> LeaseHandle:
        return LeaseHandle(
            lease_id=rec.lease_id,
            session_id=rec.session_id,
            key=rec.key,
            priority=rec.priority,
            exclusive=rec.exclusive,
            do_not_evict=rec.do_not_evict,
            pinned=rec.pinned,
            acquired_at=rec.acquired_at,
            _manager=self,
        )

    def _drop_waiter(self, pool: _Pool, waiter: _Waiter) -> None:
        if waiter in pool.waiters:
            pool.waiters.remove(waiter)

    # ------------------------------------------------------------ scheduling
    def _drain(self, pool: _Pool) -> None:
        """Grant as many queued waiters as the current pool state allows."""
        while True:
            if pool.exclusive is not None:
                return                       # exclusive holds the pool
            if not pool.active:
                waiter = self._pick_waiter(pool)
                if waiter is None:
                    return
                self._grant(pool, waiter)
                continue
            # A resident model is loaded: only same-key shared waiters proceed.
            resident = pool.resident
            mc = pool.resident_mc
            if mc == 0 or len(pool.active) < mc:
                same = self._first_same_key(pool, resident)
                if same is not None:
                    self._grant(pool, same)
                    continue
            return

    def _pick_waiter(self, pool: _Pool) -> "_Waiter | None":
        now = self._clock()
        # Exclusive waiters (calibrate / models scan) win when the pool frees.
        exclusive = [w for w in pool.waiters if w.exclusive]
        if exclusive:
            return max(exclusive,
                       key=lambda w: (self._aging(w, now), -w.seq))
        best: _Waiter | None = None
        best_score: float | None = None
        for w in pool.waiters:
            score = self._waiter_score(w, pool, now)
            if best is None or score > best_score:
                best, best_score = w, score
        return best

    def _waiter_score(self, w: _Waiter, pool: _Pool, now: float) -> float:
        score = float(int(w.priority))
        if w.key == pool.last_resident:
            score += CACHE_AFFINITY_BOOST
        score += self._aging(w, now)
        return score

    def _first_same_key(self, pool: _Pool, resident: ModelKey) -> "_Waiter | None":
        for w in pool.waiters:
            if not w.exclusive and w.key == resident:
                return w
        return None

    def _grant(self, pool: _Pool, waiter: _Waiter) -> None:
        pool.waiters.remove(waiter)
        rec = _LeaseRecord(self._new_id(), waiter.session_id, waiter.key,
                           waiter.priority, waiter.exclusive,
                           waiter.do_not_evict, waiter.pinned, self._clock(),
                           pool.key)
        waiter.granted = True
        waiter.record = rec
        if waiter.exclusive:
            pool.exclusive = rec
        else:
            pool.active.append(rec)
            pool.resident = waiter.key
            pool.resident_mc = waiter.mc
            pool.last_resident = waiter.key

    # -------------------------------------------------------------- release
    def release(self, lease_id: str) -> None:
        with self._cond:
            self._release_locked(lease_id)
            self._cond.notify_all()

    def _release_locked(self, lease_id: str) -> None:
        rec = self._leases.pop(lease_id, None)
        if rec is None or rec.pool_key is None:
            return
        pool = self._pools.get(rec.pool_key)
        if pool is None:
            return
        if rec.exclusive:
            if pool.exclusive is rec:
                pool.exclusive = None
        else:
            if rec in pool.active:
                pool.active.remove(rec)
            if not pool.active:
                pool.resident = None
        self._drain(pool)

    def release_session(self, session_id: str) -> int:
        """Release every lease held by ``session_id`` (hosted included) and
        drop its queued waiters. Returns the number of leases released."""
        with self._cond:
            ids = [lid for lid, rec in self._leases.items()
                   if rec.session_id == session_id]
            for lid in ids:
                self._release_locked(lid)
            for pool in self._pools.values():
                before = len(pool.waiters)
                pool.waiters = [w for w in pool.waiters
                                if w.session_id != session_id]
                if len(pool.waiters) != before:
                    self._drain(pool)
            self._cond.notify_all()
        return len(ids)

    # -------------------------------------------------------------- eviction
    def can_evict(self, base_url: str, model_id: str = "") -> bool:
        """False when a model is protected from the auto-evict handshake.

        With ``model_id`` given, only leases of that model block eviction;
        with ``model_id == ""``, any lease on the server does. A
        ``do_not_evict`` or ``pinned`` lease, and any *non-exclusive* lease
        (an active session's model), protect the model. An exclusive lease
        (calibrate / models scan) only blocks *new* acquisitions, never the
        holder's own load/unload, so it does not veto eviction here.
        """
        from . import servers as servers_mod

        url = servers_mod.normalize_url(base_url)
        with self._cond:
            for rec in self._leases.values():
                if rec.key.server_url != url:
                    continue
                if model_id and rec.key.model_id != model_id:
                    continue
                if rec.do_not_evict or rec.pinned:
                    return False
                if not rec.exclusive:
                    return False
            for key in self._pinned:
                if key.server_url == url and (not model_id
                                              or key.model_id == model_id):
                    return False
        return True

    def pin(self, base_url: str, model_id: str, *,
            cfg: Any = None) -> dict[str, Any]:
        """Pin a model so the evict handshake leaves it resident.

        Passes through to Lemonade / the server pin API when available and
        gracefully no-ops offline / on non-Lemonade runtimes.
        """
        from . import servers as servers_mod

        url = servers_mod.normalize_url(base_url)
        with self._cond:
            self._pinned.add(ModelKey(url, model_id))
        server_result: dict[str, Any] = {}
        try:
            server_result = servers_mod.pin_server_model(
                base_url, model_id, pinned=True, config=cfg)
        except Exception:
            pass
        return {"ok": True, "server_url": url, "model_id": model_id,
                "pinned": True, "server": server_result}

    def unpin(self, base_url: str, model_id: str, *,
              cfg: Any = None) -> dict[str, Any]:
        from . import servers as servers_mod

        url = servers_mod.normalize_url(base_url)
        with self._cond:
            self._pinned.discard(ModelKey(url, model_id))
        server_result: dict[str, Any] = {}
        try:
            server_result = servers_mod.pin_server_model(
                base_url, model_id, pinned=False, config=cfg)
        except Exception:
            pass
        return {"ok": True, "server_url": url, "model_id": model_id,
                "pinned": False, "server": server_result}

    # ------------------------------------------------------------- reconcile
    def reconcile(self, cfg: Any) -> dict[str, Any]:
        """Inspect server health / loaded models and reconcile stale pinned
        tracking. Never raises and never blocks on a sleeping server."""
        from . import servers as servers_mod

        out: dict[str, Any] = {"ok": True, "servers": [], "stale_cleared": 0}
        try:
            states = servers_mod.list_server_states(cfg) if cfg is not None else []
        except Exception:
            states = []
        for st in states:
            url = servers_mod.normalize_url(str(st.get("base_url") or ""))
            loaded = {str(m) for m in (st.get("loaded_models") or []) if m}
            out["servers"].append({
                "url": url,
                "online": bool(st.get("online")),
                "loaded_models": sorted(loaded),
            })
            with self._cond:
                stale = [k for k in self._pinned
                         if k.server_url == url and k.model_id not in loaded]
            for k in stale:
                with self._cond:
                    self._pinned.discard(k)
                out["stale_cleared"] += 1
        return out

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe accounting of active leases and pinned models."""
        with self._cond:
            leases = [{
                "lease_id": rec.lease_id,
                "session_id": rec.session_id,
                "server_url": rec.key.server_url,
                "model_id": rec.key.model_id,
                "priority": int(rec.priority),
                "exclusive": rec.exclusive,
                "do_not_evict": rec.do_not_evict,
                "pinned": rec.pinned,
                "acquired_at": rec.acquired_at,
                "hosted": rec.pool_key is None,
            } for rec in self._leases.values()]
            pinned = [{"server_url": k.server_url, "model_id": k.model_id}
                      for k in sorted(self._pinned, key=lambda k: (k.server_url,
                                                                   k.model_id))]
        return {"active_leases": leases, "pinned": pinned,
                "single_gpu_pool": self.single_gpu_pool}


# ------------------------------------------------------------------ singleton
_default_manager: ModelLeaseManager | None = None
_default_lock = threading.Lock()


def default_lease_manager() -> ModelLeaseManager:
    """The process-default lease manager (one GPU pool)."""
    global _default_manager
    if _default_manager is None:
        with _default_lock:
            if _default_manager is None:
                _default_manager = ModelLeaseManager()
    return _default_manager


def reset_default_lease_manager() -> None:
    """Drop the process-default manager (for tests)."""
    global _default_manager
    with _default_lock:
        _default_manager = None


__all__ = [
    "LeasePriority", "ModelKey", "LeaseHandle", "ModelLeaseManager",
    "default_lease_manager", "reset_default_lease_manager",
]
