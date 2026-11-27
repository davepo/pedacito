"""`pedacito bench`: the CLI surface for the six-project benchmark suite."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


# ------------------------------------------------- bench compare (WP-4106A) --
import copy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_scorecard_v2 import _card, _corpus_rows  # noqa: E402


def _write(path: Path, card) -> Path:
    path.write_text(json.dumps(card), encoding="utf-8")
    return path


def test_bench_compare_identical_scorecards_pass(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    base = _write(tmp_path / "base.json", _card())
    cur = _write(tmp_path / "cur.json", _card())
    code = main(["bench", "compare", str(base), str(cur)])
    assert code == 0
    assert "result: PASS" in capsys.readouterr().out


def test_bench_compare_regression_exits_one(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    base = _write(tmp_path / "base.json", _card())
    rows = _corpus_rows()
    for r in rows:
        r["prompt_tokens"] += 1
    cur = _write(tmp_path / "cur.json", _card(rows=rows))
    code = main(["bench", "compare", str(base), str(cur), "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"] == "regression"
    assert any("prompt tokens" in r for r in payload["regressions"])
    assert any(n.startswith("stratum edit_format") for n in payload["notes"])


def test_bench_compare_growth_flags_are_passed_through(monkeypatch, tmp_path,
                                                       capsys):
    _isolate(monkeypatch, tmp_path)
    base = _write(tmp_path / "base.json", _card())
    rows = _corpus_rows()
    for r in rows:
        r["prompt_tokens"] += 1
    cur = _write(tmp_path / "cur.json", _card(rows=rows))
    code = main(["bench", "compare", str(base), str(cur),
                 "--max-prompt-growth", "0.5"])
    assert code == 0


def test_bench_compare_incomparable_exits_two(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    base = _write(tmp_path / "base.json", _card())
    cur = _write(tmp_path / "cur.json", _card(seeds=(7, 8),
                                              rows=_corpus_rows(seeds=(7, 8))))
    assert main(["bench", "compare", str(base), str(cur)]) == 2
    assert "incomparable" in capsys.readouterr().err


def test_bench_compare_invalid_card_exits_two(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    bad = copy.deepcopy(_card())
    bad["rows"].pop()
    base = _write(tmp_path / "base.json", bad)
    cur = _write(tmp_path / "cur.json", _card())
    assert main(["bench", "compare", str(base), str(cur)]) == 2
    assert "invalid baseline" in capsys.readouterr().err


def test_bench_compare_legacy_eval_report_is_refused(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    legacy = _write(tmp_path / "legacy.json",
                    {"summary": {"pass_rate": 1.0}, "rows": []})
    cur = _write(tmp_path / "cur.json", _card())
    assert main(["bench", "compare", str(legacy), str(cur)]) == 2


def test_bench_compare_unreadable_or_bad_json_exits_two(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    cur = _write(tmp_path / "cur.json", _card())
    assert main(["bench", "compare", str(tmp_path / "missing.json"),
                 str(cur)]) == 2
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main(["bench", "compare", str(broken), str(cur)]) == 2


def test_bench_compare_needs_exactly_two_paths(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    cur = _write(tmp_path / "cur.json", _card())
    assert main(["bench", "compare", str(cur)]) == 2
    assert "usage: pedacito bench compare" in capsys.readouterr().err


def test_bench_still_rejects_two_benchmark_ids(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    assert main(["bench", "rpn_expression_calc", "ttl_lru_cache"]) == 2
    assert "only `bench compare` takes two" in capsys.readouterr().err


def test_eval_parser_accepts_scorecard_and_bench_compare_parses():
    from pedacito.cli import build_parser
    parser = build_parser()
    ns = parser.parse_args(["eval", "--seed", "3", "--scorecard", "c.json"])
    assert ns.scorecard == "c.json" and ns.seed == 3
    ns = parser.parse_args(["eval"])
    assert ns.scorecard is None
    ns = parser.parse_args(["bench", "compare", "a.json", "b.json",
                            "--max-time-growth", "0.2"])
    assert ns.benchmark == "compare" and ns.bench_args == ["a.json", "b.json"]
    assert ns.max_time_growth == 0.2 and ns.max_prompt_growth == 0.0


@pytest.mark.parametrize("flag,value", [
    ("--max-prompt-growth", "nan"),
    ("--max-prompt-growth", "inf"),
    ("--max-time-growth", "-Infinity"),
    ("--max-time-growth", "-0.1"),
    ("--max-time-growth", "abc"),
])
def test_bench_compare_refuses_bad_tolerances_at_parse_time(
        monkeypatch, tmp_path, flag, value):
    _isolate(monkeypatch, tmp_path)
    base = _write(tmp_path / "base.json", _card())
    cur = _write(tmp_path / "cur.json", _card())
    with pytest.raises(SystemExit) as exc:
        main(["bench", "compare", str(base), str(cur), flag, value])
    assert exc.value.code == 2


@pytest.mark.parametrize("flag,value", [
    ("--max-token-growth", "nan"),
    ("--max-time-growth", "inf"),
])
def test_eval_refuses_bad_tolerances_at_parse_time(monkeypatch, tmp_path,
                                                   flag, value):
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["eval", flag, value])
    assert exc.value.code == 2


def test_bench_compare_accepts_a_finite_zero_tolerance(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    base = _write(tmp_path / "base.json", _card())
    cur = _write(tmp_path / "cur.json", _card())
    assert main(["bench", "compare", str(base), str(cur),
                 "--max-prompt-growth", "0", "--max-time-growth", "0.0"]) == 0


def test_bench_compare_refuses_a_nan_literal_in_a_card(monkeypatch, tmp_path,
                                                       capsys):
    _isolate(monkeypatch, tmp_path)
    text = json.dumps(_card())
    assert '"seconds": 1.5' in text
    poisoned = tmp_path / "nan.json"
    poisoned.write_text(text.replace('"seconds": 1.5', '"seconds": NaN', 1),
                        encoding="utf-8")
    cur = _write(tmp_path / "cur.json", _card())
    assert main(["bench", "compare", str(poisoned), str(cur)]) == 2
    assert "non-finite" in capsys.readouterr().err
