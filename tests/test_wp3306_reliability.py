"""WP-3306 reliability fixes: checkpoint data-loss prevention, resilient
tree-sitter loading, honest task outcomes, circuit-breaker isolation, budget
enforcement, and the CLI/UX polish items from Rounds 1 & 2 bug hunts.

Every test is offline and deterministic (tmp_path + monkeypatch); no model
server or network is touched.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import checkpoints, cli, loop, webcheck  # noqa: E402
from pedacito import chunker_ts, profiles  # noqa: E402
from pedacito.config import Config  # noqa: E402


# --------------------------------------------------------------- checkpoints
def test_ensure_pre_run_checkpoint_creates_baseline_when_empty(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    cp = checkpoints.ensure_pre_run_checkpoint(tmp_path, "pre", mode="task")
    assert cp["parent_id"] is None
    assert "a.py" in cp["files_changed"]


def test_ensure_pre_run_checkpoint_detects_drift(tmp_path):
    (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
    first = checkpoints.create_checkpoint(tmp_path, "first")
    (tmp_path / "a.py").write_text("two\n", encoding="utf-8")  # drift
    pre = checkpoints.ensure_pre_run_checkpoint(tmp_path, "pre", mode="task")
    assert pre["id"] != first["id"]
    assert pre["parent_id"] == first["id"]
    assert "a.py" in pre["files_changed"]


def test_ensure_pre_run_checkpoint_returns_head_when_identical(tmp_path):
    (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
    first = checkpoints.create_checkpoint(tmp_path, "first")
    pre = checkpoints.ensure_pre_run_checkpoint(tmp_path, "pre", mode="task")
    assert pre["id"] == first["id"]


def test_undo_refuses_to_clobber_external_changes(tmp_path):
    (tmp_path / "a.py").write_text("before\n", encoding="utf-8")
    checkpoints.ensure_baseline(tmp_path, "baseline")
    (tmp_path / "a.py").write_text("after\n", encoding="utf-8")
    checkpoints.create_checkpoint(tmp_path, "the run")
    # An external change the run did not make (its delta is only a.py).
    (tmp_path / "b.py").write_text("external\n", encoding="utf-8")
    result = checkpoints.undo_last_run(tmp_path)
    assert result["ok"] is False
    assert result.get("clobber") is True
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "after\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "external\n"


def test_undo_force_overrides_the_clobber_guard(tmp_path):
    (tmp_path / "a.py").write_text("before\n", encoding="utf-8")
    checkpoints.ensure_baseline(tmp_path, "baseline")
    (tmp_path / "a.py").write_text("after\n", encoding="utf-8")
    checkpoints.create_checkpoint(tmp_path, "the run")
    (tmp_path / "b.py").write_text("external\n", encoding="utf-8")
    result = checkpoints.undo_last_run(tmp_path, force=True)
    assert result["ok"] is True
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "before\n"


def test_cli_undo_force_flag_is_parsed(tmp_path):
    parser = cli.build_parser()
    args = parser.parse_args(["undo", str(tmp_path), "--force"])
    assert args.force is True


def test_render_timestamp_is_local_with_offset():
    ts = checkpoints._render_timestamp("2026-01-02T03:04:05Z")
    assert "T" not in ts                      # local time, not raw ISO
    assert ts[-5] in "+-"                     # timezone offset present
    assert "Z" not in ts


def test_clean_label_folds_repair_boilerplate():
    assert checkpoints._clean_label("Your last change failed this check:") \
        == "repair"
    assert checkpoints._clean_label("  your last change failed this check  ") \
        == "repair"
    assert checkpoints._clean_label("Implement the parser") == \
        "Implement the parser"


def test_restore_notes_byte_identical_files(tmp_path):
    (tmp_path / "a.py").write_text("v1\n", encoding="utf-8")
    cp = checkpoints.create_checkpoint(tmp_path, "base")
    result = checkpoints.restore_checkpoint(tmp_path, cp["id"])
    assert "a.py" in result["identical"]


def test_cli_restore_accepts_a_checkpoint_id(tmp_path, monkeypatch, capsys):
    from pedacito import config as config_mod
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text("vram_gb = 8\nlog_sessions = false\n",
                        encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "unused.toml")
    (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
    cp = checkpoints.create_checkpoint(tmp_path, "base")
    (tmp_path / "a.py").write_text("two\n", encoding="utf-8")
    code = cli.main(["restore", str(tmp_path), "--snapshot", cp["id"],
                     "--config", str(cfg_file)])
    assert code == 0, capsys.readouterr().err
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "one\n"


def test_restore_parser_accepts_checkpoint_flag():
    parser = cli.build_parser()
    args = parser.parse_args(["restore", ".", "--checkpoint", "cp_0001_abc"])
    assert args.checkpoint == "cp_0001_abc"


# -------------------------------------------------------------- tree-sitter
def test_get_parser_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(chunker_ts, "get_parser", None)
    chunker_ts._clear_parsers()
    with pytest.raises(chunker_ts.TreeSitterUnavailableError):
        chunker_ts._get_parser("python")


def test_get_parser_retries_transient_download(monkeypatch):
    calls = []

    def fake(lang):
        calls.append(lang)
        if len(calls) < 3:
            raise OSError("grammar download failed")
        return "parser-object"

    monkeypatch.setattr(chunker_ts, "get_parser", fake)
    monkeypatch.setattr(chunker_ts.time, "sleep", lambda _s: None)
    chunker_ts._clear_parsers()
    assert chunker_ts._get_parser("python") == "parser-object"
    assert len(calls) == 3


# ------------------------------------------------------------------ scaffold
def test_header_css_is_a_block_comment():
    from pedacito.scaffold import _header
    h = _header("styles.css", "styles")
    assert h == "/* styles.css -- styles */\n"
    assert "//" not in h


def test_starter_pyproject_has_pytest_pythonpath():
    from pedacito.scaffold import STARTER_PYPROJECT
    text = STARTER_PYPROJECT.format(name="demo", summary="s")
    assert "[tool.pytest.ini_options]" in text
    assert 'pythonpath = ["."]' in text


def test_apply_skeleton_populates_a_planned_pyproject(tmp_path):
    from pedacito.scaffold import Plan, PlannedFile, apply_skeleton
    plan = Plan("demo", "a demo", [PlannedFile("pyproject.toml", "packaging"),
                                   PlannedFile("main.py", "entry")])
    apply_skeleton(tmp_path, plan, verbose=False)
    text = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert "pytest" in text and "pythonpath" in text
    assert not text.strip().startswith("#")


def test_extract_plan_json_from_markdown_fences():
    from pedacito.scaffold import _extract_plan_json
    text = ('```json\n{"project_name": "x", '
            '"files": [{"path": "a.py", "purpose": "core"}]}\n```')
    obj = _extract_plan_json(text)
    assert obj["files"][0]["path"] == "a.py"


def test_extract_plan_json_tolerates_prose():
    from pedacito.scaffold import _extract_plan_json
    obj = _extract_plan_json(
        'Here is the plan: {"files": [{"path": "a.py", "purpose": "c"}]}')
    assert obj["files"][0]["path"] == "a.py"


def test_redact_snippet_scrubs_secrets_and_bounds():
    from pedacito.scaffold import _redact_snippet
    out = _redact_snippet(
        {"files": [], "summary": "use key sk-abcdefghijklmnop for auth"},
        n=200)
    assert "sk-abcdefghijklmnop" not in out
    assert len(out) <= 200


def test_new_parser_has_no_complex_single_flag():
    parser = cli.build_parser()
    args = parser.parse_args(["new", "proj", "--single"])
    assert args.no_complex is True
    args2 = parser.parse_args(["new", "proj", "--no-complex"])
    assert args2.no_complex is True


def test_diagnose_has_diag_alias():
    parser = cli.build_parser()
    args = parser.parse_args(["diag", "src"])
    assert args.cmd in ("diagnose", "diag")


def test_models_parser_has_profile_and_model():
    parser = cli.build_parser()
    args = parser.parse_args(["models", "show", "--json", "--profile", "p",
                              "--model", "m"])
    assert args.profile == "p" and args.model == "m"


# -------------------------------------------------------------- verification
def test_run_verify_treats_pytest_no_tests_as_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(loop.proc_mod, "run",
                        lambda *a, **k: (5, "collected 0 items", "", 0.0))
    ok, out = loop.run_verify(tmp_path, "pytest -q", cfg=Config())
    assert ok is True
    assert "no tests" in out.lower()


def test_clean_task_for_coverage_strips_the_evidence_hint():
    from pedacito.verify import clean_task_for_coverage
    task = ("Fix the button.\n\nEVIDENCE: use the `run`, `screenshot`, and "
            "`browser` tools to observe it.")
    clean = clean_task_for_coverage(task)
    assert "EVIDENCE" not in clean
    assert "`run`" not in clean
    assert "Fix the button" in clean


def test_clean_task_for_coverage_strips_the_goal_prefix():
    from pedacito.verify import clean_task_for_coverage
    task = ("ORIGINAL USER GOAL (background)\nbig goal\n\nCURRENT TASK\n"
            "Rename foo to bar\n\nEVIDENCE: ...")
    clean = clean_task_for_coverage(task)
    assert "ORIGINAL USER GOAL" not in clean
    assert clean.startswith("Rename foo to bar")


# ------------------------------------------------------------ budget clamp
def test_effective_timeout_clamps_to_remaining_budget(monkeypatch):
    from pedacito.llm import LLMClient
    cfg = Config()
    cfg.timeout = 120.0
    cfg._task_deadline = time.monotonic() + 30.0
    client = LLMClient(cfg)
    assert client._effective_timeout() == pytest.approx(30.0, abs=1.0)
    cfg._task_deadline = time.monotonic() - 5.0   # budget already expired
    assert client._effective_timeout() == 5.0
    cfg._task_deadline = 0.0
    assert client._effective_timeout() == 120.0


# --------------------------------------------------------------- profiles
def test_profiles_context_fits_falls_back_to_declared_context():
    cfg = Config()
    cfg.profiles = {"local": {"model": "m",
                              "base_url": "http://localhost:1234/v1",
                              "tier": "local", "context_tokens": 16384}}
    cfg.active_profile = "local"
    recs = profiles.list_profiles(cfg)
    assert recs[0]["context_fits"] == 16384


# ---------------------------------------------------------------- webcheck
def _w(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_webcheck_flags_comment_only_css(tmp_path):
    _w(tmp_path, "index.html", '<link rel="stylesheet" href="styles.css">\n'
                               "<body>hi</body>\n")
    _w(tmp_path, "styles.css", "/* nothing yet */\n")
    found = webcheck.run(tmp_path)
    assert any("comment-only CSS" in f.message for f in found)


def test_webcheck_flags_slash_comments_in_css(tmp_path):
    _w(tmp_path, "index.html", '<link rel="stylesheet" href="styles.css">\n'
                               "<body>hi</body>\n")
    _w(tmp_path, "styles.css", "// this is not valid css\nbody { color: red }\n")
    found = webcheck.run(tmp_path)
    assert any("`//` is not a CSS comment" in f.message for f in found)


def test_webcheck_flags_unreferenced_sibling(tmp_path):
    _w(tmp_path, "index.html", "<body>hi</body>\n")
    _w(tmp_path, "orphan.css", "body { color: red }\n")
    found = webcheck.run(tmp_path)
    assert any("not referenced by any .html" in f.message for f in found)


# -------------------------------------------------------------- CLI polish
def test_parsers_prefetch_reports_per_language(capsys, monkeypatch):
    monkeypatch.setattr(chunker_ts, "get_parser", lambda lang: "parser")
    chunker_ts._clear_parsers()
    args = SimpleNamespace(parsers_action="prefetch",
                           languages=["python", "javascript"])
    code = cli._cmd_parsers(args)
    out = capsys.readouterr().out
    assert code == 0
    assert "ok     python" in out
    assert "ok     javascript" in out


def test_parsers_prefetch_reports_failure(capsys, monkeypatch):
    def boom(lang):
        raise OSError("nope")

    monkeypatch.setattr(chunker_ts, "get_parser", boom)
    chunker_ts._clear_parsers()
    args = SimpleNamespace(parsers_action="prefetch", languages=["python"])
    code = cli._cmd_parsers(args)
    out = capsys.readouterr().out
    assert code == 1
    assert "FAILED python" in out


def test_ask_json_report_has_sandbox_effective():
    cfg = Config()
    report = cli._ask_json_report(cfg, {}, ok=True, files_changed=[],
                                  diff="", verify_passed=None)
    assert "sandbox_effective" in report
    assert isinstance(report["sandbox_effective"], bool)
