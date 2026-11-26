"""The `task` action: a read-only helper agent with a fresh context window.

    tool_task(index, cfg, client, "who ends up calling retry() from the CLI?")

A long investigation (trace a call chain across six files, triage a test log)
fills a small model's window with material the parent does not need. The
parent hands the question to a CHILD agent instead: a new conversation, a
narrow brief, a step cap, READ-ONLY tools, and only a short digest comes back.
The child shares the index (and the client, so usage is counted once) but not
the transcript, and runs on its own copy of the config, so the parent's
per-task state (write allowlist, read ledger) cannot be disturbed.

Depth is capped (1 for local models, 2 for hosted ones); the deepest child
cannot spawn another. Writes, shell, browser and MCP are always off in a child.
"""

from __future__ import annotations

import copy
import fnmatch
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from . import session_ctx


DIGEST_MAX_CHARS = 1500

SUBAGENT_BASE_NOTE = (
    "You are a HELPER agent. Another agent asked you ONE bounded research "
    "question. You cannot edit anything. Find the answer with as few lookups "
    "as you can, then say `done`.")

DIGEST_SYSTEM = """\
You are a helper agent reporting back to another agent. Answer the question
from the SOURCE YOU REQUESTED only, in at most 12 short lines. Cite every claim
as file:line. If you could not verify something, say 'unverified:' and what.
No code blocks longer than 3 lines, no edit blocks, no preamble."""


def max_depth(cfg) -> int:
    from .config import is_hosted
    return 2 if is_hosted(cfg) else 1


def _tier(cfg) -> str:
    """Effective tier for the single-slot guard: an applied profile tier wins,
    otherwise the same model/base-url guess config.py uses for profiles."""
    tier = str(getattr(cfg, "active_tier", "")
               or getattr(cfg, "tier", "") or "").strip().lower()
    if tier:
        return tier
    try:
        from .config import _guess_tier
        return str(_guess_tier(
            str(getattr(cfg, "active_profile", "") or ""),
            {"model": getattr(cfg, "model", ""),
             "base_url": getattr(cfg, "base_url", "")}) or "").lower()
    except Exception:
        return ""


def _shared_prefix_eligible(cfg) -> bool:
    """True when a child may fork the parent's prefix on a single-slot local
    server: no separate scout, a local endpoint, and a tiny/small profile."""
    if str(getattr(cfg, "scout_profile", "") or "").strip():
        return False
    try:
        from .llm import _is_local_server
        if not _is_local_server(cfg):
            return False
    except Exception:
        return False
    tier = _tier(cfg)
    if tier:
        return tier in ("tiny", "small")
    try:
        from .prompts import pack_for
        return pack_for(cfg) in ("tiny", "small")
    except Exception:
        return False


def child_config(cfg):
    """A read-only copy of `cfg` for a child agent."""
    child = copy.copy(cfg)
    depth = int(getattr(cfg, "_depth", 0) or 0) + 1
    child._depth = depth
    for flag in ("allow_write", "allow_shell", "allow_browser",
                 "allow_screenshots", "allow_mcp", "enable_mcp",
                 "allow_fetch", "enable_web",
                 "semantic_search", "verify_tests", "auto_allowlist"):
        setattr(child, flag, False)
    child.allow_subagents = depth < max_depth(cfg)
    child.max_steps = max(1, int(getattr(cfg, "subagent_max_steps", 8) or 8))
    child.answer_mode = "rebuild"
    child.git_context = False
    child._interactive = False        # a helper never interrupts the user
    child.__dict__.pop("_steer", None)
    child.log_events = getattr(cfg, "log_events", True)
    for scratch in ("_backup_dir", "_task_reads", "_task_allowlist",
                    "_task_mutations", "_task_preexisting", "_agent"):
        child.__dict__.pop(scratch, None)
    # WP-4005: a subagent can only ever NARROW the parent's capabilities. The
    # parent's policy (cfg.policy or the bound session's) is intersected into a
    # read-only child policy; the `task` action is always the "explore" role.
    try:
        from . import session_ctx
        parent_policy = getattr(cfg, "policy", None)
        if parent_policy is None:
            sess = session_ctx.current_session()
            if sess is not None:
                parent_policy = getattr(sess, "policy", None)
        if parent_policy is not None and hasattr(parent_policy, "child_policy"):
            child.policy = parent_policy.child_policy(role="explore")
        else:
            child.policy = parent_policy
    except Exception:
        child.policy = getattr(cfg, "policy", None)
    return child


def tool_task(index, cfg, client, brief: str, parent=None) -> str:
    """Run a child agent on `brief` and return its digest.

    `parent`, when given, receives the child's best 1-2 content-addressed
    `EvidenceSpan`s (hashed exact source the helper really read) in its own
    evidence store and read ledger, plus matching requirement-state advances.
    The digest observation is extended with the bounded numbered spans so the
    parent can edit them without paying for a second `read`.
    """
    if not getattr(cfg, "allow_subagents", False):
        return ("ERROR: `task` is disabled (config allow_subagents = false). "
                "Research the question yourself with `read`/`grep`/`callers`.")
    brief = (brief or "").strip()
    if not brief:
        return "ERROR: task needs a question in `content`."
    if client is None:
        return "ERROR: task needs a model client."
    if int(getattr(cfg, "_depth", 0) or 0) >= max_depth(cfg):
        return "ERROR: sub-agent depth limit reached; do this yourself."
    from .agent import Agent
    from .explain import EXPLAIN_GATHER_BASE

    # Scout routing: when a separate fast profile is configured and reachable,
    # the child's gather (and its small digest) runs there, so the architect's
    # KV cache is not evicted by helper lookups.
    gather_client = client
    scout_active = False
    if str(getattr(cfg, "scout_profile", "") or "").strip():
        try:
            from . import roles
            resolved = roles.scout_client(cfg, client)
            if resolved is not None and resolved is not client:
                gather_client = resolved
                scout_active = True
        except Exception:
            gather_client = client

    child_cfg = child_config(cfg)
    # Single-slot shared-prefix forking: with no scout on a local tiny/small
    # profile, reuse the parent's System Prompt + First User Turn byte-for-byte
    # and append the helper brief, instead of wiping the server's cached KV.
    seed: list[dict] = []
    if parent is not None and not scout_active and _shared_prefix_eligible(cfg):
        parent_msgs = getattr(parent, "_messages", None) or []
        if (len(parent_msgs) >= 2
                and parent_msgs[0].get("role") == "system"
                and parent_msgs[1].get("role") == "user"):
            seed = [dict(m) for m in parent_msgs[:2]]

    child = Agent(index, gather_client, child_cfg)
    prev_site = getattr(gather_client, "site", "")
    try:
        gather_client.site = "subagent"
        gathered = child.gather(brief, verbose=False,
                                base=EXPLAIN_GATHER_BASE
                                + "\n\n" + SUBAGENT_BASE_NOTE,
                                prefix_messages=seed or None)
        digest = child.answer(brief, gathered, system=DIGEST_SYSTEM,
                              include_edit_format=False, stream=False,
                              max_tokens=700)
    except Exception as e:                          # a child never kills the parent
        return f"ERROR: helper agent failed ({type(e).__name__}: {str(e)[:160]})."
    finally:
        gather_client.site = prev_site
        # Drop the child's ephemeral turns: only its digest + evidence cross
        # the boundary back to the parent.
        try:
            del child._messages[len(seed):]
        except (AttributeError, TypeError):
            pass
    digest = (digest or "").strip()
    spans = []
    if parent is not None and hasattr(parent, "absorb_subagent_evidence"):
        try:
            spans = parent.absorb_subagent_evidence(child)
        except Exception:
            spans = []
    if not digest:
        steps = "; ".join(getattr(child, "steps_log", [])[:6]) or "no steps"
        return f"HELPER RESULT: the helper produced no answer ({steps})."
    if len(digest) > DIGEST_MAX_CHARS:
        digest = digest[:DIGEST_MAX_CHARS].rsplit("\n", 1)[0] + "\n[digest clipped]"
    if spans and hasattr(parent, "append_subagent_evidence"):
        try:
            digest = parent.append_subagent_evidence(digest, spans)
        except Exception:
            pass
    return ("HELPER RESULT (a read-only helper's digest; verify what you will "
            "rely on):\n" + digest)


# --------------------------------------------------------------------------- #
# Bounded parallel subagents (WP-4006)
# --------------------------------------------------------------------------- #
SUBAGENT_ROLES = ("explore", "general", "verifier")


@dataclass(frozen=True)
class SubagentSpec:
    """One bounded worker in a parallel fan-out."""

    role: str                       # "explore" | "general" | "verifier"
    brief: str
    evidence_scope: tuple[str, ...] = ()   # disjoint path/glob partition
    profile: str = ""               # verifier: prefer a distinct model family


@dataclass(frozen=True)
class SubagentOutcome:
    """What one worker proved, and where it ran."""

    role: str
    session_id: str
    ok: bool
    digest: str
    spans: list[Any]
    worktree_branch: str = ""
    verifier_family: str = ""
    cross_family: bool = False
    error: str = ""


def _norm_scope_entry(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _scope_prefix(value: str) -> str:
    """The literal path prefix before the first glob metacharacter."""
    text = _norm_scope_entry(value)
    cut = len(text)
    for token in ("*", "?", "["):
        idx = text.find(token)
        if idx >= 0:
            cut = min(cut, idx)
    prefix = text[:cut]
    return prefix


def _scopes_overlap(left: str, right: str) -> bool:
    """True when two path/glob entries can name the same file."""
    a = _norm_scope_entry(left)
    b = _norm_scope_entry(right)
    if not a or not b:
        return True                      # an empty scope is unpartitioned
    if a == b:
        return True
    pa = _scope_prefix(a).rstrip("/")
    pb = _scope_prefix(b).rstrip("/")
    if not pa or not pb:
        return True                      # a bare "**"/"*" covers everything
    if pa == pb or pa.startswith(pb + "/") or pb.startswith(pa + "/"):
        return True
    # Globs can still match the other literal path (e.g. src/*.py vs src/a.py).
    if any(ch in a for ch in "*?[") and fnmatch.fnmatch(b, a):
        return True
    if any(ch in b for ch in "*?[") and fnmatch.fnmatch(a, b):
        return True
    return False


def validate_disjoint_scopes(specs: list[SubagentSpec]) -> None:
    """Refuse an unpartitioned or overlapping explore/general fan-out.

    Two sibling workers may run in parallel only when each declares an
    evidence partition and no two partitions can name the same path or glob.
    Verifier specs are read-only cross-checks and are not part of the write
    partition.
    """
    relevant = [s for s in specs if s.role in ("explore", "general")]
    if len(relevant) <= 1:
        return
    empty = [s.role for s in relevant if not s.evidence_scope]
    if empty:
        raise ValueError(
            "parallel explore/general subagents need a disjoint evidence_scope "
            f"per worker; {len(empty)} worker(s) declared none")
    for i, left in enumerate(relevant):
        for right in relevant[i + 1:]:
            for a in left.evidence_scope:
                for b in right.evidence_scope:
                    if _scopes_overlap(a, b):
                        raise ValueError(
                            "overlapping subagent evidence scopes: "
                            f"{left.role} {a!r} and {right.role} {b!r}; "
                            "partition the work by disjoint paths or globs")


def model_family(model_name: str) -> str:
    """Bucket a model id into a coarse family (for cross-family verification)."""
    text = str(model_name or "").strip().lower()
    if not text:
        return ""
    # "provider/model" -> the model segment; "org:model" likewise.
    for sep in ("/",):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    text = text.strip()
    checks = (
        (("claude",), "claude"),
        (("gemini", "gemma"), "gemini" if text.startswith("gemini") else "gemma"),
        (("deepseek",), "deepseek"),
        (("qwen",), "qwen"),
        (("llama",), "llama"),
        (("mistral", "mixtral", "codestral"), "mistral"),
        (("gpt-", "gpt", "o1", "o3", "o4"), "openai"),
    )
    for prefixes, family in checks:
        if any(text.startswith(prefix) for prefix in prefixes):
            return family
    return text.split("-", 1)[0].split("_", 1)[0] or ""


def _is_git_repo(root: Path | None) -> bool:
    if root is None:
        return False
    try:
        from . import worktree
        worktree._root(root)
        return True
    except Exception:
        return False


def _resolve_model(cfg: Any, ref: str) -> str:
    """Resolve a profile reference (or a raw model id) to a model name."""
    ref = str(ref or "").strip()
    if not ref:
        return ""
    profiles = getattr(cfg, "profiles", {}) or {}
    entry = profiles.get(ref) if isinstance(profiles, dict) else None
    if isinstance(entry, dict):
        return str(entry.get("model") or ref)
    return ref


def _parent_policy(ctx: session_ctx.SessionContext) -> Any:
    policy = getattr(ctx, "policy", None)
    if policy is None:
        cfg = getattr(ctx, "cfg", None)
        policy = getattr(cfg, "policy", None)
    return policy


def _narrow_policy(ctx: session_ctx.SessionContext, role: str) -> Any:
    from . import policy as policy_mod
    parent = _parent_policy(ctx)
    if parent is not None and hasattr(parent, "child_policy"):
        return parent.child_policy(role=role)
    cfg = getattr(ctx, "cfg", None)
    root = getattr(ctx, "root", None)
    if role in ("explore", "verifier"):
        return policy_mod.PolicyEngine(cfg, root=root, _forced_mode="plan")
    return policy_mod.PolicyEngine(cfg, root=root)


def _event_sink(root: Path | None, cfg: Any) -> Any:
    try:
        from . import events as events_mod
        log = events_mod.get(root, cfg)
        if hasattr(log, "_emit_direct"):
            return log
        return getattr(log, "_log", None)
    except Exception:
        return None


def _child_context(
    parent_ctx: session_ctx.SessionContext,
    spec: SubagentSpec,
    *,
    root: Path | None = None,
) -> session_ctx.SessionContext:
    from . import events as events_mod
    from . import security

    cfg = session_ctx.clone_config(getattr(parent_ctx, "cfg", None))
    approval = str(getattr(cfg, "approval_policy", "ask") or "ask").lower()
    run_root = Path(root) if root is not None else getattr(parent_ctx, "root", None)
    child = session_ctx.SessionContext(
        root=run_root,
        cfg=cfg,
        idx=getattr(parent_ctx, "idx", None),
        client=getattr(parent_ctx, "client", None),
        permission_broker=None,
        policy=None,
        parent_session_id=str(parent_ctx.session_id),
        mode="subagent",
    )
    sink = _event_sink(run_root, cfg)
    bus = events_mod.SessionEventBus(child.session_id, sink=sink)
    child.event_bus = bus
    child.permission_broker = security.SessionPermissionBroker(
        child.session_id,
        timeout_s=30.0,
        approval_policy=approval,
        cancel_event=child.cancel_event,
        event_bus=bus,
        interactive=False,             # a worker never blocks the parent on a human
    )
    child.policy = _narrow_policy(parent_ctx, spec.role)
    # A child can only ever narrow the parent: force the read-only flags for
    # explore/verifier and keep the shared index/client (usage counted once).
    if spec.role in ("explore", "verifier"):
        for flag in ("allow_write", "allow_shell", "allow_browser",
                     "allow_screenshots", "allow_mcp", "enable_mcp",
                     "allow_fetch", "enable_web"):
            try:
                setattr(cfg, flag, False)
            except Exception:
                pass
    return child


def _in_scope(path: str, scope: tuple[str, ...]) -> bool:
    if not scope:
        return True
    rel = str(path or "").replace("\\", "/").lstrip("./")
    for entry in scope:
        pattern = _norm_scope_entry(entry).rstrip("/")
        if not pattern:
            return True
        if fnmatch.fnmatch(rel, pattern) or rel == pattern:
            return True
        if rel.startswith(pattern + "/"):
            return True
        if pattern.endswith("/**") and rel.startswith(pattern[:-3] + "/"):
            return True
    return False


def _default_runner(
    ctx: session_ctx.SessionContext, spec: SubagentSpec
) -> SubagentOutcome:
    """Real default: a small Agent gather/answer when a client is present."""
    cfg = getattr(ctx, "cfg", None)
    client = getattr(ctx, "client", None)
    root = getattr(ctx, "root", None)
    if client is None:
        digest = ""
        if root is not None and spec.evidence_scope:
            digest = "scope: " + ", ".join(spec.evidence_scope)
        return SubagentOutcome(spec.role, ctx.session_id, True, digest, [])
    try:
        from .agent import Agent
        from .index import Index
        idx = getattr(ctx, "idx", None)
        if idx is None and root is not None:
            idx = Index.load(root) or Index(root, [])
        agent = Agent(idx, client, cfg)
        gathered = agent.gather(spec.brief, verbose=False)
        digest = agent.answer(spec.brief, gathered, stream=False)
        spans = [s for s in getattr(agent.evidence, "spans", [])
                 if _in_scope(getattr(s, "file", ""), spec.evidence_scope)]
        return SubagentOutcome(spec.role, ctx.session_id, True,
                               str(digest or ""), spans)
    except Exception as exc:  # a worker never kills the fan-out
        return SubagentOutcome(spec.role, ctx.session_id, False, "", [],
                               error=f"{type(exc).__name__}: {exc}"[:300])


def run_parallel_subagents(
    parent_ctx: session_ctx.SessionContext,
    specs: list[SubagentSpec],
    *,
    max_workers: int | None = None,
    runner: Callable[[session_ctx.SessionContext, SubagentSpec], SubagentOutcome]
    | None = None,
) -> list[SubagentOutcome]:
    """Run `specs` concurrently with a hard worker cap and isolated contexts.

    Every worker receives a fresh `SessionContext` (cloned cfg, session-scoped
    event bus, permission broker, policy and resource registry). `general`
    workers run in a dedicated git worktree when the parent root is a repo;
    `explore` and `verifier` are read-only. Results keep the input order.
    """
    specs = list(specs or [])
    if not specs:
        return []
    validate_disjoint_scopes(specs)
    cfg = getattr(parent_ctx, "cfg", None)
    configured = int(getattr(cfg, "subagent_max_parallel", 3) or 3) \
        if cfg is not None else 3
    cap = int(max_workers or configured or 3)
    if cap <= 0:
        cap = 1
    workers = max(1, min(len(specs), cap))
    run = runner or _default_runner
    parent_root = getattr(parent_ctx, "root", None)
    git_repo = _is_git_repo(parent_root)

    def _run_one(index_spec: tuple[int, SubagentSpec]) -> SubagentOutcome:
        idx, spec = index_spec
        child = _child_context(parent_ctx, spec)
        handle = None
        try:
            if spec.role == "general" and git_repo and parent_root is not None:
                from . import worktree
                handle = worktree.create_session_worktree(
                    parent_root, f"sub-{idx + 1}",
                    session_id=child.session_id,
                    dirty_policy="refuse", ephemeral=False)
                child.root = handle.path
                child.worktree = handle
                # Re-point the child's policy at the worktree root.
                child.policy = _narrow_policy_for_root(parent_ctx, spec,
                                                       handle.path)
            with session_ctx.use_session(child):
                outcome = run(child, spec)
            if outcome is None:
                outcome = SubagentOutcome(spec.role, child.session_id, False, "",
                                          [], error="runner returned None")
            if not isinstance(outcome, SubagentOutcome):
                outcome = SubagentOutcome(
                    spec.role, child.session_id, True, str(outcome), [])
            branch = handle.branch if handle is not None else ""
            if spec.role == "verifier":
                outcome = _with_verifier_family(child, parent_ctx, spec, outcome)
            outcome = replace(
                outcome,
                role=outcome.role or spec.role,
                session_id=outcome.session_id or child.session_id,
                worktree_branch=outcome.worktree_branch or branch,
            )
            _emit(child, parent_ctx, "subagent_finished",
                  role=spec.role, session_id=child.session_id,
                  ok=bool(outcome.ok))
            return outcome
        except Exception as exc:
            if handle is not None:
                try:
                    from . import worktree
                    worktree.remove_session_worktree(handle, force=True)
                except Exception:
                    pass
            err = f"{type(exc).__name__}: {exc}"[:300]
            _emit(child, parent_ctx, "subagent_failed",
                  role=spec.role, session_id=child.session_id, error=err)
            return SubagentOutcome(spec.role, child.session_id, False, "", [],
                                   worktree_branch=(handle.branch
                                                    if handle is not None else ""),
                                   error=err)

    from .session_ctx import ContextThreadPoolExecutor
    ordered = list(enumerate(specs))
    with ContextThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_run_one, ordered))
    return results


def _narrow_policy_for_root(
    parent_ctx: session_ctx.SessionContext, spec: SubagentSpec, root: Path
) -> Any:
    from . import policy as policy_mod
    parent = _parent_policy(parent_ctx)
    if parent is not None and hasattr(parent, "child_policy"):
        child = parent.child_policy(role=spec.role)
        child.root = Path(root)
        return child
    cfg = getattr(parent_ctx, "cfg", None)
    forced = "plan" if spec.role in ("explore", "verifier") else ""
    return policy_mod.PolicyEngine(cfg, root=root, _forced_mode=forced)


def _with_verifier_family(
    child: session_ctx.SessionContext,
    parent_ctx: session_ctx.SessionContext,
    spec: SubagentSpec,
    outcome: SubagentOutcome,
) -> SubagentOutcome:
    cfg = getattr(parent_ctx, "cfg", None)
    worker_model = str(getattr(cfg, "model", "") or "")
    worker_family = model_family(worker_model)
    verifier_model = _resolve_model(cfg, spec.profile)
    if not verifier_model:
        roles = getattr(cfg, "roles", {}) or {}
        verifier_model = _resolve_model(
            cfg, str(roles.get("verifier") or "") if isinstance(roles, dict) else "")
    if not verifier_model:
        verifier_model = worker_model
    verifier_family = model_family(verifier_model)
    cross = bool(verifier_family and worker_family
                 and verifier_family != worker_family)
    if not cross:
        payload = {
            "role": spec.role,
            "session_id": child.session_id,
            "worker_family": worker_family,
            "verifier_family": verifier_family,
            "reason": "verifier must use a distinct model family",
        }
        _emit(child, parent_ctx, "verifier_same_family", **payload)
    return replace(outcome, verifier_family=verifier_family,
                   cross_family=cross)


def _emit(child: session_ctx.SessionContext,
          parent_ctx: session_ctx.SessionContext, kind: str,
          **fields: Any) -> None:
    for bus in (getattr(child, "event_bus", None),
                getattr(parent_ctx, "event_bus", None)):
        emit = getattr(bus, "emit", None)
        if not callable(emit):
            continue
        try:
            emit(kind, **fields)
        except Exception:
            pass
