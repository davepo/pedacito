"""WP-4005 (§1.2): the unified policy engine, its normative matrix, and the
integration proofs that `run` / `hooks` / `mcp` / `write` / `network` /
subagents all funnel through it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import hooks, tools  # noqa: E402
from pedacito.config import Config, EXEC_FIELDS  # noqa: E402
from pedacito.index import Index  # noqa: E402
from pedacito.policy import (  # noqa: E402
    PolicyDecision,
    PolicyEngine,
    PolicyRequest,
    load_policy_file,
    policy_active,
    policy_path,
)
from pedacito.session_ctx import BudgetLedger  # noqa: E402

MUTATING_SUBAGENT = PolicyRequest("subagent", "general", metadata={"role": "general"})
READ_ONLY_SUBAGENT = PolicyRequest("subagent", "explore",
                                   metadata={"role": "explore"})


def _engine(mode: str, approval: str = "ask", **kw):
    cfg = Config()
    cfg.permission_mode = mode
    cfg.approval_policy = approval
    return PolicyEngine(cfg, None, **kw)


# --------------------------------------------------------------------------- #
# the normative 5-mode matrix                                                 #
# --------------------------------------------------------------------------- #

# Expected decision per (mode, capability) with approval_policy="ask", no
# policy file, no parent, no budget, and a mutating subagent ("general").
_MATRIX = {
    "plan": {
        "read": "allow", "write": "deny", "delete": "deny", "run": "deny",
        "hook": "deny", "mcp": "deny", "network": "deny",
        "subagent": "deny", "sensitive_write": "deny",
    },
    "ask": {
        "read": "allow", "write": "ask", "delete": "ask", "run": "ask",
        "hook": "ask", "mcp": "ask", "network": "ask",
        "subagent": "ask", "sensitive_write": "deny",
    },
    "accept-edits": {
        "read": "allow", "write": "allow", "delete": "allow", "run": "ask",
        "hook": "ask", "mcp": "ask", "network": "ask",
        "subagent": "ask", "sensitive_write": "deny",
    },
    "workspace-write": {
        "read": "allow", "write": "allow", "delete": "allow", "run": "allow",
        "hook": "allow", "mcp": "ask", "network": "ask",
        "subagent": "allow", "sensitive_write": "deny",
    },
    "danger": {
        "read": "allow", "write": "allow", "delete": "allow", "run": "allow",
        "hook": "allow", "mcp": "allow", "network": "allow",
        "subagent": "allow", "sensitive_write": "deny",
    },
}

_CAPABILITIES = ("read", "write", "delete", "run", "hook", "mcp", "network",
                 "subagent", "sensitive_write")


@pytest.mark.parametrize("mode", list(_MATRIX))
@pytest.mark.parametrize("capability", _CAPABILITIES)
def test_matrix(mode, capability):
    expected = _MATRIX[mode][capability]
    req = MUTATING_SUBAGENT if capability == "subagent" \
        else PolicyRequest(capability, "target")
    d = _engine(mode).evaluate(req)
    assert d.decision == expected, (mode, capability, d)


@pytest.mark.parametrize("mode", list(_MATRIX))
def test_read_only_subagent_allowed_in_every_mode(mode):
    d = _engine(mode).evaluate(READ_ONLY_SUBAGENT)
    assert d.decision == "allow", (mode, d)


def test_workspace_write_run_requires_sandbox():
    d = _engine("workspace-write").evaluate(PolicyRequest("run", "pytest -q"))
    assert d.decision == "allow" and d.requires_sandbox is True


def test_danger_run_requires_sandbox_when_cfg_sandbox_on():
    cfg = Config()
    cfg.permission_mode = "danger"
    cfg.sandbox = True
    d = PolicyEngine(cfg, None).evaluate(PolicyRequest("run", "pytest"))
    assert d.decision == "allow" and d.requires_sandbox is True


def test_matched_rules_are_stable():
    assert _engine("plan").evaluate(
        PolicyRequest("write", "x")).matched_rule == "mode:plan"
    assert _engine("ask").evaluate(
        PolicyRequest("run", "x")).matched_rule == "mode:ask"


# --------------------------------------------------------------------------- #
# approval_policy + background                                                #
# --------------------------------------------------------------------------- #

def test_approval_policy_never_converts_ask_to_deny():
    d = _engine("ask", approval="never").evaluate(PolicyRequest("write", "x"))
    assert d.decision == "deny"
    assert d.matched_rule == "approval_policy:never"


def test_background_converts_ask_to_deny():
    req = PolicyRequest("write", "x", is_background=True)
    d = _engine("ask").evaluate(req)
    assert d.decision == "deny"
    assert d.matched_rule == "approval_policy:never"


def test_auto_safe_approves_safe_run_commands():
    d = _engine("ask", approval="auto-safe").evaluate(
        PolicyRequest("run", "pytest -q"))
    assert d.decision == "allow"
    assert d.matched_rule == "approval_policy:auto-safe"


def test_auto_safe_still_asks_unsafe_run_commands():
    d = _engine("ask", approval="auto-safe").evaluate(
        PolicyRequest("run", "rm -rf /"))
    assert d.decision == "ask"


# --------------------------------------------------------------------------- #
# budget caps                                                                 #
# --------------------------------------------------------------------------- #

def test_budget_ledger_denies_mutating_capabilities():
    ledger = BudgetLedger(max_cost_usd=1.0, spent_cost_usd=1.0)
    eng = PolicyEngine(Config(), None, budget_ledger=ledger)
    d = eng.evaluate(PolicyRequest("run", "pytest"))
    assert d.decision == "deny" and d.matched_rule == "budget:ledger"
    # reads are never budget-gated
    assert eng.evaluate(PolicyRequest("read", "x")).decision == "allow"


def test_budget_caps_from_policy_doc():
    ledger = BudgetLedger(spent_tokens=100)
    doc = {"budget": {"max_tokens": 50}}
    eng = PolicyEngine(Config(), None, policy_doc=doc, budget_ledger=ledger)
    d = eng.evaluate(PolicyRequest("write", "x"))
    assert d.decision == "deny" and d.matched_rule == "budget:max_tokens"


# --------------------------------------------------------------------------- #
# policy.toml parsing & trust gating                                          #
# --------------------------------------------------------------------------- #

_POLICY_TOML = """\
[paths]
deny = [".env*", "**/*.pem"]

[commands]
deny = ["rm -rf *"]

[network]
allow_domains = ["127.0.0.1"]
deny_domains = ["*"]

[mcp]
allow_servers = []
deny_servers = ["*"]

[budget]
max_cost_usd = 0.0
"""


def test_load_policy_file_returns_empty_when_missing(tmp_path):
    assert load_policy_file(tmp_path, Config()) == {}


def test_load_policy_file_returns_empty_when_untrusted(tmp_path, monkeypatch):
    (tmp_path / ".pedacito").mkdir()
    (tmp_path / ".pedacito" / "policy.toml").write_text(_POLICY_TOML,
                                                        encoding="utf-8")
    from pedacito import config as config_mod
    monkeypatch.setattr(config_mod, "is_trusted", lambda p: False)
    assert load_policy_file(tmp_path, Config()) == {}


def test_load_policy_file_parses_when_trusted(tmp_path, monkeypatch):
    (tmp_path / ".pedacito").mkdir()
    (tmp_path / ".pedacito" / "policy.toml").write_text(_POLICY_TOML,
                                                        encoding="utf-8")
    from pedacito import config as config_mod
    monkeypatch.setattr(config_mod, "is_trusted", lambda p: True)
    doc = load_policy_file(tmp_path, Config())
    assert doc["paths"]["deny"] == [".env*", "**/*.pem"]


def test_untrusted_policy_cannot_widen(tmp_path, monkeypatch):
    (tmp_path / ".pedacito").mkdir()
    (tmp_path / ".pedacito" / "policy.toml").write_text(_POLICY_TOML,
                                                        encoding="utf-8")
    from pedacito import config as config_mod
    monkeypatch.setattr(config_mod, "is_trusted", lambda p: False)
    cfg = Config()
    assert policy_active(cfg, tmp_path) is False
    eng = PolicyEngine(cfg, tmp_path)
    # the untrusted [commands].deny is ignored: "run" stays at the "ask" mode
    d = eng.evaluate(PolicyRequest("run", "rm -rf /tmp"))
    assert d.decision == "ask"


def test_trusted_policy_enforces_deny_rules(tmp_path, monkeypatch):
    (tmp_path / ".pedacito").mkdir()
    (tmp_path / ".pedacito" / "policy.toml").write_text(_POLICY_TOML,
                                                        encoding="utf-8")
    from pedacito import config as config_mod
    monkeypatch.setattr(config_mod, "is_trusted", lambda p: True)
    cfg = Config()
    cfg.permission_mode = "danger"
    eng = PolicyEngine(cfg, tmp_path)

    d = eng.evaluate(PolicyRequest("write", "sub/.env"))
    assert d.decision == "deny" and "paths.deny" in d.matched_rule

    d = eng.evaluate(PolicyRequest("run", "rm -rf /tmp"))
    assert d.decision == "deny" and "commands.deny" in d.matched_rule

    d = eng.evaluate(PolicyRequest("network", "example.com"))
    assert d.decision == "deny" and "network" in d.matched_rule

    d = eng.evaluate(PolicyRequest("network", "127.0.0.1"))
    assert d.decision == "allow"

    d = eng.evaluate(PolicyRequest("mcp", "docs/tool"))
    assert d.decision == "deny" and "mcp" in d.matched_rule


def test_policy_file_overrides_permission_mode(tmp_path, monkeypatch):
    (tmp_path / ".pedacito").mkdir()
    (tmp_path / ".pedacito" / "policy.toml").write_text(
        '[policy]\npermission_mode = "plan"\n', encoding="utf-8")
    from pedacito import config as config_mod
    monkeypatch.setattr(config_mod, "is_trusted", lambda p: True)
    cfg = Config()               # default "ask"
    eng = PolicyEngine(cfg, tmp_path)
    assert eng.mode == "plan"
    assert eng.evaluate(PolicyRequest("write", "x")).decision == "deny"


def test_policy_path_honours_cfg_policy_file(tmp_path):
    cfg = Config()
    cfg.policy_file = "custom/policy.toml"
    assert policy_path(tmp_path, cfg) == tmp_path / "custom" / "policy.toml"


# --------------------------------------------------------------------------- #
# subagent inheritance                                                        #
# --------------------------------------------------------------------------- #

def test_child_never_widens_parent():
    parent = _engine("danger")
    child = parent.child_policy(role="general")
    # a child of a "danger" parent still intersects; "general" keeps danger
    d = child.evaluate(PolicyRequest("run", "pytest"))
    assert d.decision == "allow"


def test_explore_child_is_read_only_even_under_danger_parent():
    parent = _engine("danger")
    child = parent.child_policy(role="explore")
    assert child.evaluate(PolicyRequest("read", "x")).decision == "allow"
    d = child.evaluate(PolicyRequest("write", "x"))
    assert d.decision == "deny" and d.matched_rule == "mode:plan"


def test_child_cannot_turn_parent_ask_into_allow():
    parent = _engine("ask")
    child = parent.child_policy(role="general")
    assert child.evaluate(PolicyRequest("write", "x")).decision == "ask"


def test_child_cannot_turn_parent_deny_into_allow():
    parent = _engine("plan")
    child = parent.child_policy(role="general")
    assert child.evaluate(PolicyRequest("write", "x")).decision == "deny"


def test_child_intersects_parent_deny_rule():
    cfg = Config()
    cfg.permission_mode = "danger"
    parent = PolicyEngine(cfg, None, policy_doc={
        "commands": {"deny": ["rm *"]}})
    child = parent.child_policy(role="general")
    assert child.evaluate(PolicyRequest("run", "rm -rf /")).decision == "deny"


# --------------------------------------------------------------------------- #
# event emission                                                              #
# --------------------------------------------------------------------------- #

class _FakeBus:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))


def test_evaluate_emits_permission_decision():
    bus = _FakeBus()
    eng = PolicyEngine(Config(), None, event_bus=bus)
    eng.evaluate(PolicyRequest("write", "src/a.py", session_id="ses_x"))
    assert bus.events
    kind, fields = bus.events[-1]
    assert kind == "permission_decision"
    assert fields["capability"] == "write"
    assert fields["target"] == "src/a.py"
    assert fields["session_id"] == "ses_x"
    assert fields["decision"] == "ask"
    assert fields["matched_rule"] == "mode:ask"
    assert "requires_sandbox" in fields


# --------------------------------------------------------------------------- #
# integration proofs                                                          #
# --------------------------------------------------------------------------- #

def test_hooks_fire_refuses_in_plan_mode(tmp_path):
    cfg = Config()
    cfg.permission_mode = "plan"
    cfg.hooks = {"pre_task": ["echo x"]}
    res = hooks.fire(cfg, tmp_path, "pre_task")
    assert res.denied and "policy" in res.message


def test_hooks_fire_still_runs_in_ask_mode(tmp_path, monkeypatch):
    cfg = Config()
    cfg.hooks = {"post_task": ["echo x"]}
    monkeypatch.setattr(hooks.proc, "run",
                        lambda *a, **k: (0, "ok", "", 0.0))
    res = hooks.fire(cfg, tmp_path, "post_task")
    assert not res.denied


def test_tool_write_enforces_plan_mode(tmp_path):
    idx = Index(tmp_path, [])
    cfg = Config()
    cfg.allow_write = True
    cfg.permission_mode = "plan"
    obs = tools.tool_write(idx, cfg, "newfile.py", "x = 1\n")
    assert "denied by policy" in obs and "mode:plan" in obs


def test_tool_run_enforces_plan_mode(tmp_path):
    idx = Index(tmp_path, [])
    cfg = Config()
    cfg.allow_shell = True
    cfg.permission_mode = "plan"
    obs = tools.tool_run(idx, cfg, "pytest -q")
    assert "denied by policy" in obs and "mode:plan" in obs


def test_tool_run_enforces_never_approval(tmp_path, monkeypatch):
    idx = Index(tmp_path, [])
    cfg = Config()
    cfg.allow_shell = True
    cfg.approval_policy = "never"
    monkeypatch.setattr(tools.proc, "run",
                        lambda *a, **k: (0, "ok", "", 0.0))
    obs = tools.tool_run(idx, cfg, "pytest -q")
    assert "denied by policy" in obs and "approval_policy:never" in obs


def test_legacy_default_tool_run_is_unchanged(tmp_path, monkeypatch):
    idx = Index(tmp_path, [])
    cfg = Config()               # permission_mode="ask", approval_policy="ask"
    cfg.allow_shell = True
    monkeypatch.setattr(tools.proc, "run",
                        lambda *a, **k: (0, "ok", "", 0.0))
    obs = tools.tool_run(idx, cfg, "echo hi")
    assert "denied by policy" not in obs


def test_exec_fields_include_policy_keys():
    for key in ("permission_mode", "approval_policy", "policy_file",
                "daemon_auto_start"):
        assert key in EXEC_FIELDS, key
    # the non-execution fields must NOT be trust-gated
    assert "daemon_port" not in EXEC_FIELDS
    assert "subagent_max_parallel" not in EXEC_FIELDS


def test_decision_dataclass_shape():
    d = PolicyDecision("deny", "mode:plan", "reason", True)
    assert d.decision == "deny" and d.matched_rule == "mode:plan"
    assert d.requires_sandbox is True
