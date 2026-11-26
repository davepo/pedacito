"""Unified capability-decision policy engine (WP-4005, §1.2).

`PolicyEngine.evaluate()` is the single place every capability decision --
`run`, `hooks`, `mcp`, `write`, `network`, and subagent inheritance -- is
resolved. It implements the normative five-mode matrix over
`permission_mode`, the `approval_policy` post-processing, the trust-gated
`.pedacito/policy.toml` allow/deny rules, the budget ledger, and strict
subagent intersection (a child can only ever narrow its parent).

The engine is *inactive* for the historical defaults (`permission_mode =
"ask"`, `approval_policy = "ask"`, no session policy, no trusted policy file):
in that case `enforce()` returns `None` and the legacy `allow_*` /
`security.request_permission()` paths keep their exact behaviour, so the
hundreds of existing tool/loop tests are untouched.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import session_ctx

PERMISSION_MODES = ("plan", "ask", "accept-edits", "workspace-write", "danger")
APPROVAL_POLICIES = ("ask", "never", "auto-safe")

# Capabilities that mutate state or execute something: the budget gate and the
# `plan` mode hard-deny apply to exactly these.
_MUTATING_CAPABILITIES = (
    "write", "delete", "run", "hook", "mcp", "network", "subagent",
    "sensitive_write",
)

# Subagent roles that are read-only by construction.
_READ_ONLY_ROLES = ("explore", "verifier")

_SAFE_COMMAND_PREFIXES = ("pytest", "ruff", "git status", "git diff")

_STRICTNESS = {"deny": 3, "ask": 2, "allow": 1}


@dataclass(frozen=True)
class PolicyRequest:
    """One capability decision to resolve."""

    capability: str
    target: str
    session_id: str = ""
    is_subagent: bool = False
    is_background: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    """The engine's verdict."""

    decision: str          # "allow" | "ask" | "deny"
    matched_rule: str
    reason: str
    requires_sandbox: bool = False


# --------------------------------------------------------------------------- #
# policy file loading (trust-gated)                                          #
# --------------------------------------------------------------------------- #

def policy_path(root: Path | str, cfg: Any = None) -> Path:
    """The resolved path of the policy file for `root`."""
    rel = str(getattr(cfg, "policy_file", ".pedacito/policy.toml")
              or ".pedacito/policy.toml")
    return Path(root) / rel


def policy_file_exists(root: Path | str, cfg: Any = None) -> bool:
    try:
        return policy_path(root, cfg).is_file()
    except (OSError, ValueError):
        return False


def load_policy_file(root: Path | str, cfg: Any = None) -> dict[str, Any]:
    """The parsed `.pedacito/policy.toml`, or `{}`.

    Returns `{}` when the file does not exist, cannot be parsed, or -- the
    security-critical case -- is NOT trusted (`config.is_trusted`). An
    untrusted policy file can therefore never widen permissions; it is
    silently ignored (the caller may still consult `cfg.permission_mode`).
    """
    path = policy_path(root, cfg)
    if not path.is_file():
        return {}
    try:
        from . import config as config_mod
        if not config_mod.is_trusted(path):
            return {}
    except Exception:
        return {}
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _effective(root: Path | str | None, cfg: Any) -> tuple[str, str]:
    """(permission_mode, approval_policy) from cfg, defensively validated."""
    mode = str(getattr(cfg, "permission_mode", "ask") or "ask").strip().lower()
    approval = str(getattr(cfg, "approval_policy", "ask") or "ask") \
        .strip().lower()
    if mode not in PERMISSION_MODES:
        mode = "ask"
    if approval not in APPROVAL_POLICIES:
        approval = "ask"
    return mode, approval


def policy_active(cfg: Any, root: Path | str | None = None) -> bool:
    """True when the policy engine should govern this config/root.

    The engine is consulted only when something explicitly opts in: a policy
    attached to `cfg` or the session, `permission_mode`/`approval_policy`
    differing from their defaults, or a *trusted* policy file under `root`.
    """
    if getattr(cfg, "policy", None) is not None:
        return True
    sess = session_ctx.current_session()
    if sess is not None and getattr(sess, "policy", None) is not None:
        return True
    mode, approval = _effective(root, cfg)
    if mode != "ask" or approval != "ask":
        return True
    if root is not None:
        try:
            from . import config as config_mod
            if policy_file_exists(root, cfg) and \
                    config_mod.is_trusted(policy_path(root, cfg)):
                return True
        except Exception:
            pass
    return False


def engine_for(cfg: Any, root: Path | str | None = None):
    """The active `PolicyEngine` for this config/root, or None (legacy)."""
    if not policy_active(cfg, root):
        return None
    attached = getattr(cfg, "policy", None)
    if attached is not None and hasattr(attached, "evaluate"):
        return attached
    sess = session_ctx.current_session()
    if sess is not None and getattr(sess, "policy", None) is not None:
        return sess.policy
    return PolicyEngine(cfg, root=root)


def enforce(cfg: Any, capability: str, target: str, *,
            root: Path | str | None = None, is_subagent: bool = False,
            is_background: bool = False, metadata: dict[str, Any] | None = None,
            session_id: str = "") -> PolicyDecision | None:
    """Resolve one capability, or None when the policy engine is inactive.

    Returns `None` for the legacy default configuration (no policy), so the
    caller's existing `allow_*` / `security.request_permission()` flow is
    unchanged. Otherwise returns the normative `PolicyDecision`.
    """
    engine = engine_for(cfg, root)
    if engine is None:
        return None
    req = PolicyRequest(capability, target, session_id=session_id,
                        is_subagent=is_subagent, is_background=is_background,
                        metadata=metadata or {})
    return engine.evaluate(req)


# --------------------------------------------------------------------------- #
# matching helpers                                                            #
# --------------------------------------------------------------------------- #

def _norm_rel(rel: str) -> str:
    return str(rel or "").replace("\\", "/").strip("/")


def _path_match(rel: str, pattern: str) -> bool:
    r = _norm_rel(rel)
    p = str(pattern or "").strip().strip("/")
    if not p:
        return False
    if fnmatch.fnmatch(r, p):
        return True
    # A leading "**/" also matches a top-level file.
    if p.startswith("**/") and fnmatch.fnmatch(r, p[3:]):
        return True
    # A bare name glob (".env*", "*.pem") matches the basename anywhere in the
    # tree, so `deny = [".env*"]` covers `sub/.env` and `keys/x.pem` alike.
    if "/" not in p:
        basename = r.rsplit("/", 1)[-1]
        if fnmatch.fnmatch(basename, p):
            return True
    return False


def _command_match(command: str, pattern: str) -> bool:
    c = str(command or "").strip().lower()
    p = str(pattern or "").strip().lower()
    if not p:
        return False
    if fnmatch.fnmatch(c, p):
        return True
    # "pytest *" / "git push*" style prefix rules.
    core = p.rstrip("*").rstrip()
    if core and c.startswith(core):
        return True
    return False


def _domain_match(domain: str, pattern: str) -> bool:
    d = str(domain or "").strip().lower().rstrip(".")
    p = str(pattern or "").strip().lower().rstrip(".")
    if p == "*":
        return True
    if not p:
        return False
    return d == p or d.endswith("." + p) or fnmatch.fnmatch(d, p)


def _mcp_server(target: str) -> str:
    return str(target or "").strip().split("/", 1)[0]


def _server_match(server: str, pattern: str) -> bool:
    s = str(server or "").strip().lower()
    p = str(pattern or "").strip().lower()
    if not p:
        return False
    if p == "*":
        return True
    return fnmatch.fnmatch(s, p)


# --------------------------------------------------------------------------- #
# the engine                                                                  #
# --------------------------------------------------------------------------- #

class PolicyEngine:
    """The single capability-decision engine for one session/config."""

    def __init__(
        self,
        cfg: Any,
        root: Path | None = None,
        *,
        policy_doc: dict[str, Any] | None = None,
        parent_policy: "PolicyEngine | None" = None,
        budget_ledger: Any = None,
        event_bus: Any = None,
        _forced_mode: str = "",
    ) -> None:
        self.cfg = cfg
        self.root = Path(root) if root is not None else None
        self.parent_policy = parent_policy
        self.budget_ledger = budget_ledger
        self.event_bus = event_bus

        if policy_doc is None and self.root is not None:
            policy_doc = load_policy_file(self.root, cfg)
        self.policy_doc = policy_doc if isinstance(policy_doc, dict) else {}

        cfg_mode, cfg_approval = _effective(root, cfg)
        doc = self.policy_doc.get("policy") or {}
        mode = _forced_mode or str(doc.get("permission_mode") or "") \
            .strip().lower() or cfg_mode
        approval = str(doc.get("approval_policy") or "").strip().lower() \
            or cfg_approval
        if mode not in PERMISSION_MODES:
            mode = cfg_mode
        if approval not in APPROVAL_POLICIES:
            approval = cfg_approval
        self.mode = mode
        self.approval_policy = approval

    # -- public API ----------------------------------------------------------
    def evaluate(self, req: PolicyRequest | str | None = None,
                 *args: Any, **kwargs: Any) -> PolicyDecision:
        """Resolve one capability. Accepts a `PolicyRequest`, a bare
        capability string, `(capability, target)` positionally, or keyword
        form (`capability=..., target=.../path=...`), for callers such as
        `security.check_sensitive_write`."""
        request = self._coerce_request(req, args, kwargs)
        decision = self._evaluate_self(request)
        if self.parent_policy is not None:
            parent = self.parent_policy.evaluate(request)
            decision = _stricter(parent, decision)
        self._emit(request, decision)
        return decision

    def child_policy(self, *, role: str = "explore") -> "PolicyEngine":
        """A subagent policy that can only INTERSECT (never widen) this one.

        `explore`/`verifier` roles are forced read-only (`plan`) regardless of
        the parent's mode.
        """
        mode = self.mode
        if role in _READ_ONLY_ROLES:
            mode = "plan"
        return PolicyEngine(
            self.cfg, self.root,
            policy_doc=self.policy_doc,
            parent_policy=self,
            budget_ledger=self.budget_ledger,
            event_bus=self.event_bus,
            _forced_mode=mode,
        )

    # -- input coercion ------------------------------------------------------
    @staticmethod
    def _coerce_request(req, args, kwargs) -> PolicyRequest:
        if isinstance(req, PolicyRequest):
            return req
        capability = str(kwargs.get("capability")
                         or (req if isinstance(req, str) else "") or "")
        if not capability and args and isinstance(args[0], str):
            capability = args[0]
        target = str(kwargs.get("target") or kwargs.get("path")
                     or kwargs.get("command") or kwargs.get("domain")
                     or kwargs.get("server") or "")
        if not target and args:
            if isinstance(req, str):
                if args and isinstance(args[0], str) and args[0] != capability:
                    target = args[0]
            elif len(args) >= 2 and isinstance(args[1], str):
                target = args[1]
        metadata = dict(kwargs.get("metadata") or {})
        for extra in ("action", "root"):
            if kwargs.get(extra) is not None:
                metadata.setdefault(extra, kwargs[extra])
        return PolicyRequest(
            capability=capability or "read",
            target=target,
            session_id=str(kwargs.get("session_id") or ""),
            is_subagent=bool(kwargs.get("is_subagent", False)),
            is_background=bool(kwargs.get("is_background", False)),
            metadata=metadata,
        )

    # -- evaluation steps ----------------------------------------------------
    def _evaluate_self(self, req: PolicyRequest) -> PolicyDecision:
        d = self._budget_denied(req)
        if d is not None:
            return d
        if req.capability == "sensitive_write":
            return self._sensitive_write(req)
        if req.capability == "sensitive_read":
            return self._sensitive_read(req)
        d = self._explicit_deny(req)
        if d is not None:
            return d
        d = self._mode_decision(req)
        return self._apply_approval_policy(req, d)

    def _budget_denied(self, req: PolicyRequest) -> PolicyDecision | None:
        if req.capability not in _MUTATING_CAPABILITIES:
            return None
        ledger = self.budget_ledger
        if ledger is None:
            return None
        if hasattr(ledger, "check"):
            try:
                ok, reason = ledger.check()
            except Exception:
                ok, reason = True, ""
            if not ok:
                return PolicyDecision("deny", "budget:ledger",
                                      reason or "budget exceeded", False)
        caps = self.policy_doc.get("budget") or {}
        for key, attr, label in (
                ("max_cost_usd", "spent_cost_usd", "max_cost_usd"),
                ("max_tokens", "spent_tokens", "max_tokens"),
                ("max_steps", "spent_steps", "max_steps")):
            raw = caps.get(key)
            if raw is None:
                continue
            try:
                cap = float(raw)
            except (TypeError, ValueError):
                continue
            if cap <= 0:
                continue
            try:
                spent = float(getattr(ledger, attr, 0.0) or 0.0)
            except (TypeError, ValueError):
                spent = 0.0
            if spent >= cap:
                return PolicyDecision(
                    "deny", f"budget:{label}",
                    f"{label} budget exceeded ({spent} / {cap})", False)
        return None

    def _sensitive_write(self, req: PolicyRequest) -> PolicyDecision:
        reason = ("sensitive writes are always refused in plan/ask, "
                  "background, subagent or unattended (approval_policy="
                  "never) modes")
        if (self.mode in ("plan", "ask") or req.is_subagent
                or req.is_background or self.approval_policy == "never"):
            return PolicyDecision("deny", "policy:sensitive_write", reason,
                                  False)
        if self._paths_allowed(req.target)[0] and \
                not self._paths_denied(req.target)[0]:
            return PolicyDecision(
                "allow", "policy.toml:paths.allow",
                "explicitly allowed sensitive write (interactive human "
                "approval is still required)", False)
        return PolicyDecision(
            "deny", "policy:sensitive_write",
            "sensitive write not in trusted [paths].allow", False)

    def _sensitive_read(self, req: PolicyRequest) -> PolicyDecision:
        if self._paths_allowed(req.target)[0] and \
                not self._paths_denied(req.target)[0]:
            return PolicyDecision(
                "allow", "policy.toml:paths.allow",
                "explicitly allowed sensitive read", False)
        return PolicyDecision(
            "deny", "policy:sensitive_read",
            "sensitive read not allowed by trusted policy", False)

    def _explicit_deny(self, req: PolicyRequest) -> PolicyDecision | None:
        cap = req.capability
        if cap in ("write", "delete"):
            denied, rule = self._paths_denied(req.target)
            if denied:
                return PolicyDecision("deny", rule,
                                      f"path denied by policy: {req.target}",
                                      False)
        if cap == "run":
            denied, rule = self._commands_denied(req.target)
            if denied:
                return PolicyDecision("deny", rule,
                                      f"command denied by policy: {req.target}",
                                      False)
        if cap == "network":
            denied, rule = self._network_denied(req.target)
            if denied:
                return PolicyDecision("deny", rule,
                                      f"network denied by policy: {req.target}",
                                      False)
        if cap == "mcp":
            denied, rule = self._mcp_denied(req.target)
            if denied:
                return PolicyDecision("deny", rule,
                                      f"MCP denied by policy: {req.target}",
                                      False)
        return None

    def _mode_decision(self, req: PolicyRequest) -> PolicyDecision:
        cap = req.capability
        role = str(req.metadata.get("role") or req.target or "")
        read_only_sub = cap == "subagent" and role in _READ_ONLY_ROLES

        if cap == "read":
            return PolicyDecision("allow", f"mode:{self.mode}",
                                  "non-sensitive reads are always allowed",
                                  False)

        mode = self.mode
        if mode == "plan":
            if read_only_sub:
                return PolicyDecision("allow", "mode:plan",
                                      "read-only subagent allowed in plan mode",
                                      False)
            return PolicyDecision("deny", "mode:plan",
                                  f"capability {cap!r} is refused in plan mode",
                                  False)

        if mode == "ask":
            if cap == "subagent" and read_only_sub:
                return PolicyDecision("allow", "mode:ask",
                                      "read-only subagent allowed", False)
            if cap in _MUTATING_CAPABILITIES:
                return PolicyDecision("ask", "mode:ask",
                                      f"capability {cap!r} needs approval",
                                      False)
            return PolicyDecision("allow", "mode:ask",
                                  f"capability {cap!r} allowed", False)

        if mode == "accept-edits":
            if cap in ("write", "delete"):
                return PolicyDecision("allow", "mode:accept-edits",
                                      "workspace write/delete auto-approved",
                                      False)
            if cap == "subagent" and read_only_sub:
                return PolicyDecision("allow", "mode:accept-edits",
                                      "read-only subagent allowed", False)
            if cap in _MUTATING_CAPABILITIES:
                return PolicyDecision("ask", "mode:accept-edits",
                                      f"capability {cap!r} needs approval",
                                      False)
            return PolicyDecision("allow", "mode:accept-edits",
                                  f"capability {cap!r} allowed", False)

        if mode == "workspace-write":
            if cap in ("write", "delete"):
                return PolicyDecision("allow", "mode:workspace-write",
                                      "workspace write/delete auto-approved",
                                      False)
            if cap == "run":
                return PolicyDecision("allow", "mode:workspace-write",
                                      "sandboxed run allowed",
                                      requires_sandbox=True)
            if cap == "hook":
                return PolicyDecision("allow", "mode:workspace-write",
                                      "configured hooks allowed", False)
            if cap == "subagent":
                return PolicyDecision("allow", "mode:workspace-write",
                                      "configured subagent allowed", False)
            if cap == "network":
                allowed, rule = self._network_allowed(req.target)
                if allowed:
                    return PolicyDecision("allow", rule,
                                          "network target allowlisted", False)
                return PolicyDecision("ask", "mode:workspace-write",
                                      "network needs an allowlist or approval",
                                      False)
            if cap == "mcp":
                allowed, rule = self._mcp_allowed(req.target)
                if allowed:
                    return PolicyDecision("allow", rule,
                                          "MCP server allowlisted", False)
                return PolicyDecision("ask", "mode:workspace-write",
                                      "MCP needs an allowlist or approval",
                                      False)
            return PolicyDecision("ask", "mode:workspace-write",
                                  f"capability {cap!r} needs approval", False)

        # mode == "danger"
        if cap == "run":
            sandbox = bool(getattr(self.cfg, "sandbox", False)
                           or getattr(self.cfg, "sandbox_strict", False))
            return PolicyDecision("allow", "mode:danger",
                                  "run allowed in danger mode",
                                  requires_sandbox=sandbox)
        return PolicyDecision("allow", "mode:danger",
                              f"capability {cap!r} allowed in danger mode",
                              False)

    def _apply_approval_policy(self, req: PolicyRequest,
                               d: PolicyDecision) -> PolicyDecision:
        if d.decision != "ask":
            return d
        if self.approval_policy == "never" or req.is_background:
            return PolicyDecision("deny", "approval_policy:never",
                                  "unattended/never policy refuses any ask",
                                  d.requires_sandbox)
        if self.approval_policy == "auto-safe" and req.capability == "run":
            if self._is_safe_command(req.target):
                return PolicyDecision("allow", "approval_policy:auto-safe",
                                      "known safe read-only/test command",
                                      d.requires_sandbox)
        return d

    def _is_safe_command(self, command: str) -> bool:
        if self._commands_allowed(command)[0]:
            return True
        cmd = str(command or "").strip().lower()
        for prefix in _SAFE_COMMAND_PREFIXES:
            if cmd == prefix or cmd.startswith(prefix + " "):
                return True
        return False

    # -- policy.toml allow/deny helpers --------------------------------------
    @property
    def _paths_deny(self) -> tuple[str, ...]:
        return tuple(self.policy_doc.get("paths", {}).get("deny") or ())

    @property
    def _paths_allow(self) -> tuple[str, ...]:
        return tuple(self.policy_doc.get("paths", {}).get("allow") or ())

    @property
    def _commands_deny(self) -> tuple[str, ...]:
        return tuple(self.policy_doc.get("commands", {}).get("deny") or ())

    @property
    def _commands_allow(self) -> tuple[str, ...]:
        return tuple(self.policy_doc.get("commands", {}).get("allow") or ())

    def _paths_denied(self, rel: str) -> tuple[bool, str]:
        for i, pat in enumerate(self._paths_deny):
            if _path_match(rel, pat):
                return True, f"policy.toml:paths.deny[{i}]"
        return False, ""

    def _paths_allowed(self, rel: str) -> tuple[bool, str]:
        for i, pat in enumerate(self._paths_allow):
            if _path_match(rel, pat):
                return True, f"policy.toml:paths.allow[{i}]"
        return False, ""

    def _commands_denied(self, command: str) -> tuple[bool, str]:
        for i, pat in enumerate(self._commands_deny):
            if _command_match(command, pat):
                return True, f"policy.toml:commands.deny[{i}]"
        return False, ""

    def _commands_allowed(self, command: str) -> tuple[bool, str]:
        for i, pat in enumerate(self._commands_allow):
            if _command_match(command, pat):
                return True, f"policy.toml:commands.allow[{i}]"
        return False, ""

    def _network_denied(self, domain: str) -> tuple[bool, str]:
        deny = tuple(self.policy_doc.get("network", {}).get("deny_domains")
                     or ())
        allow = tuple(self.policy_doc.get("network", {}).get("allow_domains")
                      or ())
        for i, d in enumerate(deny):
            if d != "*" and _domain_match(domain, d):
                return True, f"policy.toml:network.deny_domains[{i}]"
        if allow and any(_domain_match(domain, a) for a in allow):
            return False, ""
        if allow:
            return True, "policy.toml:network.allow_domains"
        if "*" in deny:
            return True, "policy.toml:network.deny_domains[wildcard]"
        return False, ""

    def _network_allowed(self, domain: str) -> tuple[bool, str]:
        allow = tuple(self.policy_doc.get("network", {}).get("allow_domains")
                      or ())
        for i, a in enumerate(allow):
            if _domain_match(domain, a):
                return True, f"policy.toml:network.allow_domains[{i}]"
        return False, ""

    def _mcp_denied(self, target: str) -> tuple[bool, str]:
        server = _mcp_server(target)
        deny = tuple(self.policy_doc.get("mcp", {}).get("deny_servers") or ())
        allow = tuple(self.policy_doc.get("mcp", {}).get("allow_servers") or ())
        for i, d in enumerate(deny):
            if d != "*" and _server_match(server, d):
                return True, f"policy.toml:mcp.deny_servers[{i}]"
        if allow and any(_server_match(server, a) for a in allow):
            return False, ""
        if allow:
            return True, "policy.toml:mcp.allow_servers"
        if "*" in deny:
            return True, "policy.toml:mcp.deny_servers[wildcard]"
        return False, ""

    def _mcp_allowed(self, target: str) -> tuple[bool, str]:
        server = _mcp_server(target)
        allow = tuple(self.policy_doc.get("mcp", {}).get("allow_servers") or ())
        for i, a in enumerate(allow):
            if _server_match(server, a):
                return True, f"policy.toml:mcp.allow_servers[{i}]"
        return False, ""

    # -- events --------------------------------------------------------------
    def _emit(self, req: PolicyRequest, decision: PolicyDecision) -> None:
        bus = self.event_bus
        if bus is None:
            sess = session_ctx.current_session()
            if sess is not None:
                bus = getattr(sess, "event_bus", None)
        if bus is None:
            return
        emit = getattr(bus, "emit", None)
        if not callable(emit):
            return
        try:
            emit(
                "permission_decision",
                session_id=req.session_id or "",
                capability=req.capability,
                target=req.target,
                decision=decision.decision,
                matched_rule=decision.matched_rule,
                reason=decision.reason,
                requires_sandbox=decision.requires_sandbox,
            )
        except Exception:
            pass


def _stricter(a: PolicyDecision, b: PolicyDecision) -> PolicyDecision:
    """The stricter of two decisions (deny > ask > allow). A child policy can
    only ever narrow its parent, so the stricter verdict wins; `requires_sandbox`
    is OR-ed so a sandbox requirement is never dropped."""
    base = a if _STRICTNESS[a.decision] >= _STRICTNESS[b.decision] else b
    if base.requires_sandbox or a.requires_sandbox or b.requires_sandbox:
        if not base.requires_sandbox:
            return PolicyDecision(base.decision, base.matched_rule,
                                  base.reason, True)
    return base
