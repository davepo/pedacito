"""`pedacito bench`: the CLI surface for the six-project benchmark suite."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import benchmarks
from pedacito.cli import main


def _isolate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_bench_list_needs_no_model_or_vram(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    code = main(["bench", "--list"])
    assert code == 0
    out = capsys.readouterr().out
    assert "rpn_expression_calc" in out
    assert "pelican_bicycle_svg" in out
    assert "12 checks" in out and "71" not in out
    assert "calc.py" in out


def test_bench_list_json(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    code = main(["bench", "--list", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    ids = {row["id"] for row in payload["benchmarks"]}
    assert len(ids) == 6 and "rpn_expression_calc" in ids


def _fake_run(base_dir: Path, benchmark_id: str, report: bool = True):
    run = base_dir / f"pedacito_bench_{benchmark_id}_tiny_20260101_120000"
    run.mkdir(parents=True)
    data = {
        "benchmark_id": benchmark_id,
        "title": "Fake benchmark",
        "profile": "tiny",
        "model": "fake-model",
        "started_at": "2026-01-01T12:00:00+00:00",
        "duration": 1.0,
        "score": "1/1 rubric checks",
        "rubric": {"ok": True, "passed": 1, "total": 1},
        "model_tests": {"ok": True, "passed": 1, "total": 1},
    }
    (run / "benchmark_run.json").write_text(json.dumps(data), encoding="utf-8")
    if report:
        (run / "BENCHMARK_REPORT.md").write_text("# report\n", encoding="utf-8")
    return run


def test_bench_recent_uses_the_default_root(monkeypatch, tmp_path, capsys):
    # The default root is a benchmarks/ folder BESIDE the current project.
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    _fake_run(tmp_path / "benchmarks", "rpn_expression_calc")
    code = main(["bench", "--recent"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Recent benchmark runs" in out
    assert "rpn_expression_calc" in out
    assert "BENCHMARK_REPORT.md" in out


def test_bench_recent_json(monkeypatch, tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    _fake_run(tmp_path / "benchmarks", "ttl_lru_cache")
    code = main(["bench", "--recent", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recent"][0]["benchmark_id"] == "ttl_lru_cache"
    assert payload["base_dir"].endswith("benchmarks")


def _config_file(tmp_path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        'base_url = "http://127.0.0.1:9/v1"\n'
        'model = "fake-model"\n'
        'vram_gb = 8\n', encoding="utf-8")
    return path


def test_bench_one_benchmark_json_with_injected_generator(
        monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    cfgfile = _config_file(tmp_path)
    real = benchmarks.run_benchmark

    class Client:
        def __init__(self):
            self.stats = {}

    def generator(project, cfg, client):
        project.mkdir(parents=True, exist_ok=True)
        (project / "calc.py").write_text(
            "def add(a, b):\n    return a + b\n", encoding="utf-8")
        (project / "test_calc.py").write_text(
            "from calc import add\n\n"
            "def test_add():\n    assert add(1, 2) == 3\n",
            encoding="utf-8")
        return 0

    def wrapper(*args, **kwargs):
        kwargs["generator"] = generator
        kwargs["client"] = Client()
        return real(*args, **kwargs)

    monkeypatch.setattr(benchmarks, "run_benchmark", wrapper)
    out_dir = tmp_path / "runs"
    code = main([
        "bench", "rpn_expression_calc", "--json", "--config", str(cfgfile),
        "--dir", str(out_dir), "--model-timeout", "60",
        "--holdout-timeout", "60",
    ])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["benchmark_id"] == "rpn_expression_calc"
    assert payload["model_tests"]["passed"] == 1
    assert payload["rubric"]["total"] == 12
    assert payload["ok"] is False          # the toy calc fails the rubric
    assert Path(payload["report_path"]).is_file()
    assert "Generating the project" in captured.err
    assert (out_dir / Path(payload["project_dir"]).name).is_dir()
