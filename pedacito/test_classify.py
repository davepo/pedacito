"""Flake and infrastructure classification for a red test run (WP-4105D).

A red post-edit test run is only worth a model repair when a specific test
failed for a reason this change can own. This module decides that:

    initial red run ─┬─ timeout / collection error / missing dependency /
                     │  infrastructure ──────────────────► stop, unverified
                     └─ one attributable node ──► one rerun of that node
                          passes ─────────────► flake      (stop, unverified)
                          same assertion ─────► real_failure (repair)
                          anything else ──────► unknown    (stop, unverified)

Only `real_failure` may reach the repair loop. Nothing here turns a red run
green: a quarantined failure stays red and is only routed differently.
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass

from . import security

KINDS = ("flake", "timeout", "collection_error", "missing_dependency",
         "infra", "real_failure", "unknown")
INFRA_KINDS = frozenset({"timeout", "collection_error", "missing_dependency",
                         "infra"})
REPAIRABLE_KINDS = frozenset({"real_failure"})
RERUN_STATUSES = ("passed", "failed", "inconclusive", "infra", "not_run")
BASELINE_STATUSES = ("failed", "not_failing", "absent")

# At most this many distinct failing nodes are rerun per verify run.
MAX_RERUN_NODES = 3
QUARANTINE_FILE = ".pedacito/flakes.json"
_MAX_QUARANTINE_BYTES = 64 * 1024
_MAX_NODE_ID = 300
_MAX_REASON = 200

_NODE_ID_RE = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9_./\-]*\.py"
    r"(?:::[A-Za-z0-9_][A-Za-z0-9_.\[\]\-,=+]*)+$")

_TIMEOUT_RE = re.compile(
    r"was stopped \((?:timeout|idle)\b|\bTimeoutExpired\b"
    r"|^\+{3,} Timeout \+{3,}", re.M)
_COLLECTION_RE = re.compile(
    r"error(?:s)? during collection|ERROR collecting\b|collection error",
    re.IGNORECASE)
_MISSING_RE = re.compile(
    r"ModuleNotFoundError: No module named|No module named ['\"]?\w"
    r"|command not found|is not recognized as an internal or external command"
    r"|executable file not found"
    r"|pytest is not installed for this interpreter"
    r"|No such file or directory: ['\"]?(?:pytest|node|npm|cargo|go)\b",
    re.IGNORECASE)
_INFRA_RE = re.compile(
    r"was stopped \(|could not run `|could not run the |ERROR: could not run"
    r"|ERROR: denied by policy|SandboxRefused|MemoryError",
    re.IGNORECASE)
_PASS_EVIDENCE_RE = re.compile(
    r"\b[1-9]\d*\s+passed\b|^#\s*pass\s+[1-9]\d*\b|^ℹ\s*pass\s+[1-9]\d*\b",
    re.M)
_FAIL_TALLY_RE = re.compile(
    r"\b[1-9]\d*\s+(?:failed|errors?)\b|^#\s*fail\s+[1-9]|^ℹ\s*fail\s+[1-9]",
    re.M)


@dataclass(frozen=True)
class TestClassification:
    kind: str
    node_id: str
    baseline: str
    rerun: str
    reason: str


@dataclass(frozen=True)
class NodeRerun:
    """Outcome of rerunning one failing node under the original command."""
    node_id: str
    passed: bool
    output: str


@dataclass(frozen=True)
class Quarantine:
    """An explicitly, temporarily tolerated failing node. Expiry is required."""
    test_id: str
    reason: str
    expires: str


def has_test_node(identities: dict) -> bool:
    """True when some identity names a test node. A bare traceback keyed as
    `output::<signature>` or `signature::...` is not a node."""
    return any("::" in key and not key.startswith(("output::", "signature::"))
               for key in identities)


def classify_output_kind(output: str, *, has_node_failures: bool = False) -> str:
    """The infrastructure kind a red run's output names, or '' when it names none.

    A timeout, a collection error or a missing dependency stops the run before
    any test verdict exists, so it wins even when some node ids are printed.
    Missing-dependency and generic infrastructure text only counts when no node
    failed: a real test whose traceback mentions a module is still a real test.
    """
    text = output or ""
    if _TIMEOUT_RE.search(text):
        return "timeout"
    if _COLLECTION_RE.search(text):
        return "collection_error"
    if has_node_failures:
        return ""
    if _MISSING_RE.search(text):
        return "missing_dependency"
    if _INFRA_RE.search(text):
        return "infra"
    return ""


def valid_node_id(node_id: str) -> bool:
    """A pytest node id safe to pass as one argv element: a relative `.py`
    path with `::` parts, no traversal and no whitespace or shell syntax."""
    nid = str(node_id or "")
    if not nid or len(nid) > _MAX_NODE_ID or not _NODE_ID_RE.match(nid):
        return False
    file_part = nid.split("::", 1)[0]
    return ".." not in re.split(r"[/\\]", file_part)


def is_pytest_command(command: str) -> bool:
    low = (command or "").lower()
    return "pytest" in low and "unittest" not in low


def rerun_command(base: str, node_id: str) -> str:
    """The original runner invoked on exactly one node, or '' when the runner
    has no stable node-id syntax (only pytest's is used; nothing is invented)."""
    if not base or not valid_node_id(node_id) or not is_pytest_command(base):
        return ""
    return f"{base} {node_id}"


def node_file_inside(root, node_id: str) -> bool:
    """True when the node's file resolves to an existing file inside `root`."""
    file_part = str(node_id or "").split("::", 1)[0]
    try:
        resolved = security.resolve_safe_path(root, file_part)
    except security.SecurityError:
        return False
    return resolved.is_file()


def _rerun_passed_ok(output: str) -> bool:
    """A zero exit only counts as a pass when the output shows the node ran."""
    text = output or ""
    return bool(_PASS_EVIDENCE_RE.search(text)) and not _FAIL_TALLY_RE.search(text)


def baseline_status(baseline: str, base_ids: dict, node: str) -> str:
    """How the pre-edit run saw `node`: failed, not_failing or absent."""
    if not (baseline or "").strip():
        return "absent"
    return "failed" if node in base_ids else "not_failing"


def classify_test_failure(initial: str, rerun: NodeRerun | None,
                          baseline: str) -> TestClassification:
    """Classify one red run's failing node from its rerun and the baseline.

    `initial` is the red run's output, `rerun` the single node rerun (None when
    no rerun was made) and `baseline` the pre-edit output ('' when absent).
    Identities and baseline diffs come from `loop.failure_identities`, so the
    same node key and assertion signature drive both repair and this verdict.
    """
    from . import loop

    ids = loop.failure_identities(initial or "")
    base_ids = loop.failure_identities(baseline or "") if baseline else {}
    kind = classify_output_kind(initial, has_node_failures=has_test_node(ids))
    node = rerun.node_id if rerun is not None else (
        next(iter(ids)) if len(ids) == 1 else "")
    bstatus = baseline_status(baseline, base_ids, node)
    if kind:
        return TestClassification(
            kind, node, bstatus, "not_run",
            f"{kind}: the run stopped before a test verdict; no rerun")
    if not node:
        return TestClassification(
            "unknown", "", bstatus, "not_run",
            "ambiguous: no single failing test to attribute")
    if node not in ids:
        return TestClassification(
            "unknown", node, bstatus, "not_run",
            "the rerun node is not one of the initial failures")
    if node in base_ids and base_ids[node].signature == ids[node].signature:
        return TestClassification(
            "unknown", node, bstatus, "not_run",
            "pre-existing: the same failure is already in the baseline")
    if rerun is None:
        return TestClassification(
            "unknown", node, bstatus, "not_run",
            "no rerun was possible; the failure is unverified")
    if rerun.passed:
        if _rerun_passed_ok(rerun.output):
            return TestClassification(
                "flake", node, bstatus, "passed",
                "the node passed on one rerun under the same command")
        return TestClassification(
            "unknown", node, bstatus, "inconclusive",
            "rerun exited 0 but showed no evidence that the node ran")

    rkind = classify_output_kind(
        rerun.output,
        has_node_failures=has_test_node(loop.failure_identities(rerun.output or "")))
    if rkind:
        return TestClassification(
            rkind, node, bstatus, "infra",
            f"{rkind}: the rerun stopped before a test verdict")
    rids = loop.failure_identities(rerun.output or "")
    if node in rids and rids[node].signature == ids[node].signature:
        return TestClassification(
            "real_failure", node, bstatus, "failed",
            "the same assertion failed on rerun")
    if node in rids:
        return TestClassification(
            "unknown", node, bstatus, "failed",
            "the failure changed on rerun; ambiguous")
    return TestClassification(
        "unknown", node, bstatus, "inconclusive",
        "the rerun failed without an attributable failing test")


def _parse_date(text: str) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(str(text).strip())
    except ValueError:
        return None


def load_quarantine(root, *, today: datetime.date | None = None
                    ) -> tuple[dict[str, Quarantine], tuple[str, ...]]:
    """(active quarantine entries by test id, warnings) from `.pedacito/flakes.json`.

    Shape: {"quarantine": [{"test_id", "reason", "expires": "YYYY-MM-DD"}]}.
    Every entry needs an explicit future-or-today expiry; expired and malformed
    entries are dropped with a warning, never silently honoured. The file is
    resolved inside the project first, so a symlink or junction out is refused.
    """
    warnings: list[str] = []
    try:
        path = security.resolve_safe_path(root, QUARANTINE_FILE)
    except security.SecurityError as exc:
        return {}, (f"{QUARANTINE_FILE} refused: {exc}",)
    if not path.exists():
        return {}, ()
    try:
        if not path.is_file() or path.stat().st_size > _MAX_QUARANTINE_BYTES:
            return {}, (f"{QUARANTINE_FILE} is not a small regular file; ignored",)
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, (f"{QUARANTINE_FILE} unreadable: {type(exc).__name__}",)
    rows = data.get("quarantine") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {}, (f"{QUARANTINE_FILE} needs a 'quarantine' list; ignored",)
    now = today or datetime.date.today()
    active: dict[str, Quarantine] = {}
    for row in rows:
        if not isinstance(row, dict):
            warnings.append("quarantine entry is not an object; ignored")
            continue
        test_id = str(row.get("test_id") or "").strip()
        reason = str(row.get("reason") or "").strip()[:_MAX_REASON]
        expires = _parse_date(row.get("expires") or "")
        if not test_id or len(test_id) > _MAX_NODE_ID or not reason:
            warnings.append("quarantine entry needs test_id and reason; ignored")
            continue
        if expires is None:
            warnings.append(f"quarantine for {test_id[:120]} needs an "
                            "expires date (YYYY-MM-DD); ignored")
            continue
        if expires < now:
            warnings.append(f"quarantine for {test_id[:120]} expired "
                            f"{expires.isoformat()}; the failure counts again")
            continue
        active[test_id] = Quarantine(test_id, reason, expires.isoformat())
    return active, tuple(warnings)
