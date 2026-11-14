"""Engine upgrades: receiver-aware qualified calls, incoming test-caller
arity gate, and hardlink fast-path ratchet checkpoints.

Chunkers now record `ClassName.method` next to bare `method` for calls made
through a receiver known to be a class instance (self/cls/this, constructor
binding, type annotation, Go receiver pairs, Rust impls). callers/usages rank
exact qualified matches first and exclude sites explicitly qualified to a
different class. The contract gate surfaces bad-arity TEST call sites
statically -- before pytest ever spawns -- and ratchet checkpoints are cheap
hardlinks backed by fileio.write's atomic replacement.
"""
from __future__ import annotations

import stat
import types

import pytest

from pedacito import fileio, loop, ratchet, spec
from pedacito.chunker import chunk_source
from pedacito.config import Config
from pedacito.edits import Result
from pedacito.index import Index
from pedacito.tools import tool_callers, tool_usages

TS_AVAILABLE = True
try:
    from pedacito.chunker_ts import chunk_file_ts
except Exception:
    TS_AVAILABLE = False


def _pw(path, text: str) -> None:
    """Write the way the loop does: fileio.write's atomic replace, so a
    ratchet hardlink checkpoint keeps its own inode."""
    fileio.write(path, text)


def _py_index(tmp_path, files):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    chunks = []
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
        chunks.extend(chunk_source(rel, src, 60))
    return Index(root, chunks)


def _calls_of(idx, rel):
    return [c for ch in idx.chunks if ch.file == rel for c in ch.calls]


def _arity_errors(root):
    """Only the incoming-test-caller messages (the other passes have
    their own, already-tested shapes)."""
    return [e for e in spec.verify_cross_file_contracts(root)
            if "but updated signature in" in e]


# ---------------------------------------------------------------------------
# 1) Python chunker records qualified targets


def test_python_self_calls_are_qualified(tmp_path):
    idx = _py_index(tmp_path, {"svc.py": (
        "class Server:\n"
        "    def stop(self):\n"
        "        pass\n"
        "    def run(self):\n"
        "        self.stop()\n"
    )})
    calls = _calls_of(idx, "svc.py")
    assert "stop" in calls          # bare kept: base-name lookups still work
    assert "Server.stop" in calls   # qualified: exact, ranked lookups


def test_python_ctor_and_annotation_receivers_are_qualified(tmp_path):
    idx = _py_index(tmp_path, {"app.py": (
        "def main():\n"
        "    server = Server()\n"
        "    server.start()\n"
        "    logger: Logger = make()\n"
        "    logger.info('hi')\n"
    )})
    calls = _calls_of(idx, "app.py")
    assert {"Server.start", "Logger.info"} <= set(calls)
    assert "start" in calls and "info" in calls


def test_python_local_reassignment_not_attributed(tmp_path):
    # `server` rebound to another type must not keep the old binding.
    idx = _py_index(tmp_path, {"app.py": (
        "def main():\n"
        "    server = Server()\n"
        "    server = Other()\n"
        "    server.start()\n"
    )})
    calls = _calls_of(idx, "app.py")
    assert "Other.start" in calls
    assert "Server.start" not in calls


def test_callers_of_ranks_exact_qualified_first(tmp_path):
    idx = _py_index(tmp_path, {
        "b.py": "class X:\n    def g(self):\n        pass\n",
        "a.py": "def f():\n    g()\n",                       # bare only
        "c.py": "def h(x: X):\n    x.g()\n",                 # exact qualified
    })
    res = idx.callers_of("X.g")
    assert res and res[0].file == "c.py"
    assert [c.file for c in res][0] == "c.py"
    assert idx.callers_of("nope") == []


def test_tool_callers_excludes_other_class_and_labels_unqualified(tmp_path):
    idx = _py_index(tmp_path, {
        "k.py": "class K1:\n    def run(self):\n        pass\n\n"
                "class K2:\n    def run(self):\n        pass\n",
        "callers.py": (
            "from k import K1\n\n\n"
            "def use(a):\n"
            "    K1.run(a)\n"    # exact qualified receiver -> counted, first
            "    K2.run()\n"     # receiver is a DIFFERENT class -> dropped
            "    a.run()\n"      # receiver unknown -> kept, labelled
        ),
    })
    out = tool_callers(idx, "K1.run")
    assert "Defined at:" in out
    assert "Called from:" in out
    assert "K1.run(a)" in out
    assert "K2.run" not in out          # different-class site excluded entirely
    assert "(unqualified match)" in out
    assert "a.run()" in out
    assert "class K2" not in out        # the other class never pollutes
    # exact qualified sites are listed before the ambiguous ones
    assert out.index("K1.run(a)") < out.index("(unqualified match)")


def test_tool_usages_bare_query_still_shows_everything(tmp_path):
    idx = _py_index(tmp_path, {
        "u.py": "class A:\n    def m(self):\n        pass\n\n"
                "class B:\n    def m(self):\n        pass\n",
        "use.py": "from u import A\n\n\ndef go(a, b):\n    A.m(a)\n    b.m(a)\n",
    })
    out = tool_callers(idx, "A.m")
    assert "A.m(a)" in out
    assert "b.m(a)" in out and "(unqualified match)" in out
    assert "B.m" not in out
    cfg = Config()
    us = tool_usages(idx, cfg, "m")     # bare base-name query
    assert "use.py" in us and "u.py" in us


# ---------------------------------------------------------------------------
# 2) tree-sitter chunkers qualify receivers too

JS = """\
class Server {
  stop() {}
  run() {
    this.stop();
    other.stop();
  }
}
"""

GO = """\
package main

type Server struct{}

func (s *Server) Stop() {}

func (s *Server) Run() {
\ts.Stop()
}

func Use() {
\tvar srv Server
\tsrv.Stop()
\tc := &Server{}
\tc.Ping()
}
"""

RUST = """\
struct Server;

impl Server {
    fn stop(&self) {}
    fn run(&self) {
        self.stop();
    }
}
"""


@pytest.mark.skipif(not TS_AVAILABLE, reason="tree-sitter grammars not installed")
@pytest.mark.parametrize("rel,src,expected", [
    ("x.js", JS, {"stop", "Server.stop"}),
    ("x.go", GO, {"Stop", "Server.Stop", "Ping", "Server.Ping"}),
    ("x.rs", RUST, {"stop", "Server.stop"}),
])
def test_ts_chunkers_record_qualified_calls(rel, src, expected):
    ext = "." + rel.rsplit(".", 1)[1]
    chunks = chunk_file_ts(rel, src, ext, 60, 220, 12000, 3000, 0)
    calls = [c for ch in chunks for c in ch.calls]
    assert expected <= set(calls)


@pytest.mark.skipif(not TS_AVAILABLE, reason="tree-sitter grammars not installed")
def test_ts_tool_callers_excludes_other_class(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    src = ("class K1 {\n  run() {}\n}\n"
           "class K2 {\n  run() {}\n}\n"
           "function go(a) {\n  K1.run(a);\n  K2.run();\n  a.run();\n}\n")
    (root / "a.js").write_text(src, encoding="utf-8")
    idx = Index(root, chunk_file_ts("a.js", src, ".js", 60, 220, 12000,
                                    3000, 0))
    out = tool_callers(idx, "K1.run")
    assert "Called from:" in out
    assert "K1.run(a)" in out
    assert "K2.run" not in out
    assert "(unqualified match)" in out


# ---------------------------------------------------------------------------
# 3) incoming test-caller arity gate (before pytest ever spawns)


def _contract_project(tmp_path):
    root = tmp_path / "proj"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
    return root


def test_gate_flags_dotted_test_caller(tmp_path):
    root = _contract_project(tmp_path)
    (root / "app" / "logic.py").write_text(
        "def add(a, b, c):\n    return a + b + c\n", encoding="utf-8")
    (root / "tests" / "test_logic.py").write_text(
        "import app.logic\n\n"
        "def test_add():\n"
        "    assert app.logic.add(1) == 6\n", encoding="utf-8")
    errs = _arity_errors(root)
    assert len(errs) == 1
    assert "ContractError in tests/test_logic.py:4" in errs[0]
    assert "with 1 arg(s)" in errs[0]
    assert "app/logic.py requires 3 positional argument(s)" in errs[0]


def test_gate_flags_from_submodule_and_range(tmp_path):
    root = _contract_project(tmp_path)
    (root / "app" / "logic.py").write_text(
        "def add(a, b, c=0):\n    return a + b + c\n", encoding="utf-8")
    (root / "tests" / "test_logic.py").write_text(
        "from app import logic\n\n"
        "def test_add():\n"
        "    logic.add(1)\n", encoding="utf-8")
    errs = _arity_errors(root)
    assert len(errs) == 1
    assert "requires 2-3 positional argument(s)" in errs[0]


def test_gate_flags_ctor_bound_instance_method(tmp_path):
    root = _contract_project(tmp_path)
    (root / "app" / "svc.py").write_text(
        "class Server:\n"
        "    def start(self, host, port):\n"
        "        return host, port\n", encoding="utf-8")
    (root / "tests" / "test_svc.py").write_text(
        "from app.svc import Server\n\n"
        "def test_start():\n"
        "    s = Server()\n"
        "    s.start('h')\n", encoding="utf-8")
    errs = _arity_errors(root)
    assert len(errs) == 1
    assert "with 1 arg(s)" in errs[0]
    assert "app/svc.py requires 2 positional argument(s)" in errs[0]


def test_gate_annotated_var_ok_and_bad(tmp_path):
    root = _contract_project(tmp_path)
    (root / "app" / "svc.py").write_text(
        "class Srv:\n"
        "    def m(self, a, b):\n        return a + b\n", encoding="utf-8")
    (root / "tests" / "test_mixed.py").write_text(
        "from app.svc import Srv\n\n"
        "def make():\n    return Srv()\n\n"
        "def test_ok():\n"
        "    s: Srv = make()\n"
        "    s.m(1, 2)\n\n"          # right arity -> no error
        "def test_bad():\n"
        "    t: Srv = make()\n"
        "    t.m(1)\n", encoding="utf-8")
    errs = _arity_errors(root)
    assert len(errs) == 1
    assert "test_mixed.py" in errs[0]
    assert "with 1 arg(s)" in errs[0]
    assert "requires 2 positional argument(s)" in errs[0]


def test_gate_skips_shadowed_and_unbound_receivers(tmp_path):
    root = _contract_project(tmp_path)
    (root / "app" / "svc.py").write_text(
        "class Srv:\n"
        "    def m(self, a, b):\n        return a + b\n", encoding="utf-8")
    (root / "tests" / "test_shadow.py").write_text(
        "from app.svc import Srv\n\n"
        "def test_local_class():\n"
        "    class Srv:\n"
        "        pass\n"
        "    s = Srv()\n"
        "    s.m(1)\n\n"              # local Srv shadows the import: skipped
        "def test_unknown_var():\n"
        "    z.m(1)\n", encoding="utf-8")
    assert _arity_errors(root) == []


def test_gate_only_scans_test_files(tmp_path):
    root = _contract_project(tmp_path)
    (root / "app" / "logic.py").write_text(
        "def add(a, b, c):\n    return a + b + c\n", encoding="utf-8")
    # the same violation in a NON-test file is out of scope for this phase
    (root / "app" / "user.py").write_text(
        "import app.logic\napp.logic.add(1)\n", encoding="utf-8")
    assert _arity_errors(root) == []


def test_contract_gate_blocks_pytest_spawn(tmp_path, monkeypatch):
    """`_post_edit_tests` vetoes the subprocess: a changed signature plus a
    bad-arity test caller produces a `contract` repair request with ZERO
    pytest runs."""
    root = _contract_project(tmp_path)
    (root / "app" / "logic.py").write_text(
        "def add(a, b, c=0):\n    return a + b + c\n", encoding="utf-8")
    (root / "tests" / "test_logic.py").write_text(
        "import app.logic\n\n"
        "def test_add():\n    assert app.logic.add(1) == 6\n", encoding="utf-8")

    cfg = Config()
    cfg.verify_tests = True
    cfg.allow_shell = True
    cfg.repair_rounds = 1
    cfg._static_baseline = set()

    class _Agent:
        def __init__(self):
            self.calls = []

        def _repair_from_findings(self, task, reply, errors, results):
            self.calls.append([e.kind for e in errors])
            self.messages = [e.message for e in errors]
            return "REPAIR"

        def _apply_once(self, reply, dry_run, backup_dir):
            return []          # repair did not land: the loop stops

    runs = []
    monkeypatch.setattr(loop, "_run_tests",
                        lambda idx, cfg_, files: (runs.append(list(files))
                                                  or (True, "ok", "pytest")))
    agent = _Agent()
    results = [Result(True, "app/logic.py", "applied", before="",
                      after="def add(a, b, c=0):\n    return a + b + c\n")]
    out_results, reply, tests_ok, _preexisting = loop._post_edit_tests(
        root, cfg, agent, types.SimpleNamespace(root=root), "task",
        results, "reply", None, None, ["app/logic.py"])
    assert runs == []                        # pytest never spawned
    assert agent.calls == [["contract"]]
    assert "updated signature" in agent.messages[0]
    assert tests_ok is False


# ---------------------------------------------------------------------------
# 4) ratchet hardlink checkpoints + atomic writes


def test_ratchet_begin_links_then_rollback_restores(tmp_path):
    _pw(tmp_path / "a.py", "orig\n")
    ratchet.begin(tmp_path, 1, "step one")
    snap = ratchet.step_dir(tmp_path, 1) / "a.py"
    if snap.stat().st_ino == (tmp_path / "a.py").stat().st_ino:
        assert (tmp_path / "a.py").stat().st_nlink == 2      # true hardlink
    else:
        pytest.skip("filesystem forbids hard links (copy2 fallback)")
    (tmp_path / "a.py").unlink()             # simulate a step that deleted it
    report = ratchet.rollback(tmp_path, 1)
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "orig\n"
    assert "a.py" in report["restored"]


def test_ratchet_snapshot_survives_later_edits(tmp_path):
    _pw(tmp_path / "a.py", "v1\n")
    ratchet.begin(tmp_path, 1)
    fileio.write(tmp_path / "a.py", "v2\n")  # atomic replace: snapshot holds v1
    assert (ratchet.step_dir(tmp_path, 1) / "a.py").read_text(encoding="utf-8") == "v1\n"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "v2\n"
    report = ratchet.rollback(tmp_path, 1)
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "v1\n"
    assert "a.py" in report["restored"]


def test_ratchet_rollback_deletes_step_creations(tmp_path):
    _pw(tmp_path / "a.py", "v1\n")
    ratchet.begin(tmp_path, 1)
    (tmp_path / "new.py").write_text("x = 1\n", encoding="utf-8")
    ratchet.rollback(tmp_path, 1)
    assert not (tmp_path / "new.py").exists()


def test_fileio_write_is_atomic(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("v1", encoding="utf-8")
    f.chmod(0o640)
    inode_before = f.stat().st_ino
    # Windows only honours the read-only bit, so 0o640 reads back as 0o666;
    # the invariant is that write_bytes preserves the mode the platform reports.
    mode_before = stat.S_IMODE(f.stat().st_mode)
    fileio.write(f, "v2")
    assert f.read_text(encoding="utf-8") == "v2"
    assert f.stat().st_ino != inode_before          # fresh inode, not mutated
    assert stat.S_IMODE(f.stat().st_mode) == mode_before  # permissions preserved
    assert not list(tmp_path.glob("*.pedacito-tmp"))


def test_fileio_write_through_symlink_updates_target(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("v1", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    fileio.write(link, "v2")
    assert target.read_text(encoding="utf-8") == "v2"
    assert link.is_symlink()
