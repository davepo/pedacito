"""WP-R402 regression tests: scaffold `.gitignore` hygiene (R4-BUG-02) and the
`new --complex --max-minutes` guard on the optional README doc pass
(R4-BUG-06)."""

from types import SimpleNamespace

from pedacito import config, loop, scaffold


class _StubClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "seconds": 0.0}

    def check(self, warn=True):
        return {}

    def chat_json(self, messages, schema, max_tokens=0, **kw):
        return {"files": []}


def _plan(*, no_extras=False):
    return scaffold.Plan("Pond", "a tiny game",
                         [scaffold.PlannedFile("main.py", "entry point")],
                         spec=scaffold.Spec("a tiny pond game"),
                         no_extras=no_extras)


def _cfg():
    cfg = config.load(require_hardware=False)
    cfg.backup = False
    cfg.log_sessions = False
    return cfg


def test_scaffold_gitignore_gets_pedacito_entries_without_git_dir(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr(scaffold, "build_plan",
                        lambda *a, **k: _plan())
    monkeypatch.setattr(scaffold, "build_interface",
                        lambda *a, **k: {})
    root = tmp_path / "pond"
    cfg = _cfg()
    code = scaffold.run_new(root, cfg, _StubClient(cfg),
                            scaffold.Spec("a tiny pond game"),
                            plan_only=True, verbose=False)
    assert code == 0
    assert not (root / ".git").exists()
    text = (root / ".gitignore").read_text(encoding="utf-8")
    lines = {ln.strip() for ln in text.splitlines()}
    assert {".pedacito/", ".pedacitoignore", "PLAN.md"} <= lines
    assert "__pycache__/" in lines


def test_dry_run_scaffold_leaves_disk_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(scaffold, "build_plan",
                        lambda *a, **k: _plan())
    cfg = _cfg()

    fresh = tmp_path / "fresh"
    assert scaffold.run_new(fresh, cfg, _StubClient(cfg),
                            scaffold.Spec("a tiny pond game"),
                            verbose=False) == 0
    assert not fresh.exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    assert scaffold.run_new(empty, cfg, _StubClient(cfg),
                            scaffold.Spec("a tiny pond game"),
                            verbose=False) == 0
    assert list(empty.iterdir()) == []


def test_complex_expired_max_minutes_skips_readme_doc_pass(tmp_path,
                                                           monkeypatch,
                                                           capsys):
    clock = {"now": 1000.0}
    monkeypatch.setattr(scaffold, "time", SimpleNamespace(
        monotonic=lambda: clock["now"]))
    monkeypatch.setattr(scaffold, "build_plan",
                        lambda *a, **k: _plan())
    monkeypatch.setattr(scaffold, "build_interface",
                        lambda *a, **k: {})
    monkeypatch.setattr(scaffold, "verify_and_fix",
                        lambda *a, **k: (True, ""))
    doc_calls = []

    def fake_docs(client, cfg, idx, doc_plan, verbose=True, backup_dir=None):
        doc_calls.append([f.path for f in doc_plan.files])
        return {"done": [], "noop": [], "failed": [], "skipped": []}

    def slow_loop(*a, **k):
        clock["now"] += 120      # the build consumes more than the 1-minute budget
        return 0

    monkeypatch.setattr(scaffold, "build_docs", fake_docs)
    monkeypatch.setattr(loop, "run_loop", slow_loop)

    root = tmp_path / "pond"
    cfg = _cfg()
    code = scaffold.run_new(root, cfg, _StubClient(cfg),
                            scaffold.Spec("a tiny pond game"),
                            apply=True, complex=True, max_minutes=1,
                            verbose=False)
    assert code == 0
    assert (root / "README.md").is_file()
    assert doc_calls == [["main.py"]]
    err = capsys.readouterr().err
    assert ("[max-minutes reached; skipping optional README documentation "
            "pass]") in err


def test_complex_within_budget_keeps_readme_doc_pass(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(scaffold, "build_plan",
                        lambda *a, **k: _plan())
    monkeypatch.setattr(scaffold, "build_interface",
                        lambda *a, **k: {})
    monkeypatch.setattr(scaffold, "verify_and_fix",
                        lambda *a, **k: (True, ""))
    doc_calls = []

    def fake_docs(client, cfg, idx, doc_plan, verbose=True, backup_dir=None):
        doc_calls.append([f.path for f in doc_plan.files])
        return {"done": [], "noop": [], "failed": [], "skipped": []}

    monkeypatch.setattr(scaffold, "build_docs", fake_docs)
    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: 0)

    root = tmp_path / "pond"
    cfg = _cfg()
    scaffold.run_new(root, cfg, _StubClient(cfg),
                     scaffold.Spec("a tiny pond game"),
                     apply=True, complex=True, max_minutes=60, verbose=False)
    assert any("README.md" in c for c in doc_calls)
    assert "max-minutes reached" not in capsys.readouterr().err
