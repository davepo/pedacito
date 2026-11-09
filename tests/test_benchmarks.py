"""The built-in model benchmarking suite: catalog, folder naming, hidden
rubric scoring, report generation, and the GUI endpoints.

The generation phase is injectable (`generator=`), so these tests exercise the
real grader and report writer without a model server or GPU. The calculator
fixture is a deliberately small golden implementation: the hidden rubric must
grade it 12/12, and a broken variant must produce listed shortcomings.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import benchmarks  # noqa: E402
from pedacito import config as config_mod, scaffold  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.gui import core, jobs, server, settings_api  # noqa: E402

GOLDEN_CALC = r'''"""A small arithmetic calculator (golden test fixture)."""
import ast
import re
import sys


def _eval(node, variables):
    if isinstance(node, ast.Expression):
        return _eval(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"unknown variable {node.id!r}")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval(node.operand, variables)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, variables)
        right = _eval(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
    raise ValueError("syntax error")


def evaluate(expression, variables=None):
    variables = {} if variables is None else variables
    text = expression.strip()
    m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(?!=)(.*)$", text)
    if m:
        value = evaluate(m.group(2), variables)
        variables[m.group(1)] = value
        return value
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"syntax error: {exc.msg}") from exc
    return _eval(tree, variables)


def evaluate_rpn(expression, variables=None):
    variables = {} if variables is None else variables
    stack = []
    for token in expression.split():
        if token in ("+", "-", "*", "/", "**"):
            if len(stack) < 2:
                raise ValueError("syntax error")
            b, a = stack.pop(), stack.pop()
            if token == "+":
                stack.append(a + b)
            elif token == "-":
                stack.append(a - b)
            elif token == "*":
                stack.append(a * b)
            elif token == "/":
                if b == 0:
                    raise ZeroDivisionError("division by zero")
                stack.append(a / b)
            else:
                stack.append(a ** b)
        else:
            stack.append(float(token) if "." in token else int(token))
    if len(stack) != 1:
        raise ValueError("syntax error")
    return stack[0]


def _show(value):
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rpn = bool(argv and argv[0] == "--rpn")
    if rpn:
        argv.pop(0)
    if not argv:
        print("usage: calc.py [--rpn] EXPR ...", file=sys.stderr)
        return 2
    variables = {}
    try:
        for expr in argv:
            value = evaluate_rpn(expr, variables) if rpn else evaluate(expr, variables)
            print(_show(value))
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

GOLDEN_CALC_TEST = """\
from calc import evaluate


def test_precedence():
    assert evaluate("2 + 3 * 4") == 14
"""


def _write_calc(project: Path, calc: str = GOLDEN_CALC,
                test: str = GOLDEN_CALC_TEST) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / "calc.py").write_text(calc, encoding="utf-8")
    (project / "test_calc.py").write_text(test, encoding="utf-8")


class FakeClient:
    """Only the usage counters the runner reads; the injected generator does
    the file work, so no chat call is made."""

    def __init__(self):
        self.stats = {"calls": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "cached_tokens": 0,
                      "cache_write_tokens": 0, "cost": 0.0,
                      "seconds": 0.0, "by_site": {}}

    def record_usage(self):
        self.stats.update({"calls": 7, "prompt_tokens": 700,
                           "completion_tokens": 300, "cached_tokens": 100,
                           "cost": 0.01, "seconds": 2.0,
                           "by_site": {"repair": {"calls": 2}}})


def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.model = "golden-test-model"
    cfg.active_profile = "golden"
    cfg.benchmarks_dir = str(tmp_path / "runs")
    cfg.sources = {}
    cfg.log_events = False
    return cfg


# ------------------------------------------------------------------ catalog ----
def test_catalog_has_the_five_diagnostic_benchmarks():
    catalog = benchmarks.list_benchmarks()
    assert [b["id"] for b in catalog] == [
        "rpn_expression_calc", "markdown_table_fmt", "ttl_lru_cache",
        "log_anomaly_analyzer", "mini_kanban_html",
    ]
    for item in catalog:
        assert item["title"] and item["category"] and item["description"]
        assert item["what_it_tests"] and item["checks"] >= 10
        assert item["deliverables"]
    assert benchmarks.get_benchmark("ttl_lru_cache").id == "ttl_lru_cache"
    with pytest.raises(config_mod.ConfigError):
        benchmarks.get_benchmark("does-not-exist")


def test_every_holdout_source_is_valid_python():
    for benchmark in benchmarks.BENCHMARKS.values():
        source = benchmarks.holdout_source(benchmark)
        compile(source, f"<{benchmark.id}>", "exec")
        assert f"test_{benchmark.rubric[0].id}" in source


# ------------------------------------------------------------- directory ------
def test_project_dir_name_and_fresh_unique_folder(tmp_path):
    base = tmp_path / "benchmarks"
    first = benchmarks.benchmark_project_dir(
        "rpn_expression_calc", "My Profile / 1", base,
        stamp="20260101_120000")
    assert first.name == (
        "pedacito_bench_rpn_expression_calc_My_Profile_1_20260101_120000")
    assert first.is_dir()
    second = benchmarks.benchmark_project_dir(
        "rpn_expression_calc", "My Profile / 1", base,
        stamp="20260101_120000")
    assert second != first and second.name.endswith("_2")
    assert second.is_dir()
    assert benchmarks.project_dir_name("x", "y", "z") == "pedacito_bench_x_y_z"


def test_default_benchmarks_root_prefers_config_then_project_parent(tmp_path):
    cfg = Config()
    project = tmp_path / "work" / "app"
    project.mkdir(parents=True)
    assert benchmarks.default_benchmarks_root(
        cfg, current_root=project) == tmp_path / "work" / "benchmarks"
    cfg.benchmarks_dir = str(tmp_path / "my-benches")
    assert benchmarks.default_benchmarks_root(
        cfg, current_root=project) == tmp_path / "my-benches"


# --------------------------------------------------------------- grading ------
def test_holdout_rubric_scores_a_correct_calculator(tmp_path):
    project = tmp_path / "calc"
    _write_calc(project)
    suite = benchmarks.run_holdout(project, "rpn_expression_calc")
    assert suite.total == 12 and suite.passed == 12 and suite.ok
    assert all(c.passed for c in suite.checks)

    model = benchmarks.run_model_tests(project)
    assert model.ok and model.passed == 1 and model.total == 1


def test_holdout_rubric_names_specific_failures(tmp_path):
    project = tmp_path / "broken"
    _write_calc(project, calc="def evaluate(expression, variables=None):\n"
                              "    return 0\n")
    suite = benchmarks.run_holdout(project, "rpn_expression_calc")
    assert not suite.ok and suite.passed < suite.total
    failed = {c.id for c in suite.checks if not c.passed}
    assert "exponent_right_associative" in failed
    assert "variables_persist" in failed


# ------------------------------------------------------ runner and report -----
def test_run_benchmark_writes_report_and_keeps_project(tmp_path):
    cfg = _cfg(tmp_path)
    project_box: list[Path] = []

    def generator(project: Path, _cfg, client):
        project_box.append(project)
        _write_calc(project)
        client.record_usage()
        return 0

    result = benchmarks.run_benchmark(
        "golden", "rpn_expression_calc", cfg=cfg,
        client=FakeClient(), generator=generator,
        current_root=tmp_path / "projects" / "app",
        stamp="20260102_030405")

    assert project_box == [result.project_dir]
    assert result.project_dir.parent == tmp_path / "runs"
    assert result.project_dir.is_dir()
    assert result.rubric.ok and result.rubric.passed == 12
    assert result.model_tests.ok
    assert result.profile == "golden"
    assert result.model == "golden-test-model"
    assert result.stats["total_tokens"] == 1000
    assert result.repair_rounds >= 2

    report = result.report_path
    assert report == result.project_dir / "BENCHMARK_REPORT.md"
    text = report.read_text(encoding="utf-8")
    for needle in ("BENCHMARK REPORT", "golden", "golden-test-model",
                   "12/12", "Run It Yourself", "Requirement Ledger"):
        assert needle in text
    assert (result.project_dir / "BENCHMARK_TASK.md").is_file()
    meta = json.loads((result.project_dir / "benchmark_run.json").read_text())
    assert meta["report_path"] == str(report)
    assert meta["project_path"] == str(result.project_dir)
    assert not any(result.project_dir.rglob("__pycache__"))

    data = result.to_dict()
    assert data["score"] == "12/12 rubric checks"
    assert data["shortcomings"] == []
    assert data["project_dir"] == str(result.project_dir)


def test_default_generator_uses_scaffold_apply_and_the_benchmark_prompt(
        tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls = []

    def fake_run_new(root, run_cfg, client, spec, **kwargs):
        calls.append((root, spec, kwargs))
        _write_calc(root)
        return 0

    monkeypatch.setattr(scaffold, "run_new", fake_run_new)
    result = benchmarks.run_benchmark(
        "golden", "rpn_expression_calc", cfg=cfg, client=FakeClient(),
        stamp="20260102_030408")
    assert calls and calls[0][0] == result.project_dir
    assert calls[0][1].text == benchmarks.get_benchmark("rpn_expression_calc").task
    assert calls[0][2]["apply"] is True
    assert result.rubric.ok


def test_report_lists_shortcomings_for_a_broken_generation(tmp_path):
    cfg = _cfg(tmp_path)

    def generator(project: Path, _cfg, _client):
        project.mkdir(parents=True, exist_ok=True)
        (project / "calc.py").write_text(
            "def evaluate(expression, variables=None):\n    return 0\n",
            encoding="utf-8")
        (project / "test_calc.py").write_text(
            "def test_wrong():\n    assert False\n", encoding="utf-8")
        return 1

    result = benchmarks.run_benchmark(
        "broken", "rpn_expression_calc", cfg=cfg, client=FakeClient(),
        generator=generator, stamp="20260102_030406")
    assert not result.rubric.ok
    assert result.generation_rc == 1
    assert any("exponent" in s.lower() for s in result.shortcomings)
    text = result.report_path.read_text(encoding="utf-8")
    assert "Shortcomings Detected" in text
    assert "Model's own tests failed" in text


def test_recent_runs_reads_report_metadata(tmp_path):
    cfg = _cfg(tmp_path)

    def generator(project: Path, _cfg, _client):
        _write_calc(project)
        return 0

    result = benchmarks.run_benchmark(
        "golden", "rpn_expression_calc", cfg=cfg, client=FakeClient(),
        generator=generator, stamp="20260102_030407")
    rows = benchmarks.recent_runs(tmp_path / "runs")
    assert rows and rows[0]["benchmark_id"] == "rpn_expression_calc"
    assert rows[0]["project_dir"] == str(result.project_dir)
    assert rows[0]["score"] == "12/12 rubric checks"


# ------------------------------------------------------------- GUI API --------
class _Client:
    def __init__(self, port: int):
        self.port = port

    def request(self, method: str, path: str, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        hdrs = {"Content-Type": "application/json"}
        data = None if body is None and method == "GET" else json.dumps(body or {})
        conn.request(method, path, body=data, headers=hdrs)
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        return response.status, json.loads(payload or b"{}")

    def get(self, path: str):
        return self.request("GET", path)

    def post(self, path: str, body=None):
        return self.request("POST", path, body or {})

    def wait_idle(self, timeout: float = 15.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if not self.get("/api/state")[1].get("job"):
                return True
            time.sleep(0.05)
        return False


@pytest.fixture
def gui(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg" / "config.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(settings_api, "USER_CONFIG", cfg_path)
    monkeypatch.setattr(core, "RECENT_FILE", tmp_path / "cfg" / "gui_recent.json")
    for var in [k for k in os.environ if k.startswith("PEDACITO_")]:
        monkeypatch.delenv(var)
    core.STATE.__init__()
    srv = server.create_server(0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    core.install_runner()
    client = _Client(srv.server_address[1])
    client.tmp = tmp_path
    yield client
    core.STATE.bus.close()
    srv.shutdown()
    srv.server_close()
    core.uninstall_runner()
    job = core.STATE.job
    if job is not None and job.thread is not None:
        job.thread.join(5)


def test_benchmarks_endpoint_lists_catalog_and_profiles(gui):
    status, data = gui.get("/api/benchmarks")
    assert status == 200 and data["ok"] is True
    assert len(data["benchmarks"]) == 5
    assert data["benchmarks"][0]["id"] == "rpn_expression_calc"
    assert "profiles" in data and "recent" in data
    assert data["default_base_dir"].endswith("benchmarks")
    assert data["last_run"] is None


def test_benchmark_run_endpoint_rejects_unknown_task(gui):
    status, data = gui.post("/api/benchmarks/run", {
        "profile": "", "benchmark_id": "not-a-benchmark"})
    assert status == 400 and data["ok"] is False
    assert "Unknown benchmark" in data["detail"]


def test_benchmark_run_endpoint_spawns_job_and_exposes_result(gui, monkeypatch):
    def fake_run(job, opts):
        with core.STATE.lock:
            core.STATE.benchmark = {
                "job_id": job.id,
                "benchmark_id": opts["benchmark_id"],
                "title": "Fake benchmark result",
                "profile": opts["profile"] or "default",
                "model": "fake-model",
                "project_dir": str(gui.tmp / "generated"),
                "started_at": "2026-01-01T00:00:00+00:00",
                "duration": 1.25,
                "stats": {"total_tokens": 10, "calls": 2},
                "repair_rounds": 0,
                "shortcomings": [],
                "score": "12/12 rubric checks",
                "rubric": {"passed": 12, "total": 12, "checks": []},
                "model_tests": {"passed": 1, "total": 1, "ok": True,
                                "ran": True},
                "report": "# report",
                "report_path": str(gui.tmp / "generated" /
                                   "BENCHMARK_REPORT.md"),
            }
        core.STATE.add_message("assistant", "fake benchmark finished",
                               kind="benchmark")

    monkeypatch.setattr(jobs, "run_benchmark", fake_run)
    status, data = gui.post("/api/benchmarks/run", {
        "profile": "", "benchmark_id": "rpn_expression_calc"})
    assert status == 202 and data["ok"] is True and data["job"] > 0
    assert gui.wait_idle()

    status, out = gui.get("/api/benchmarks")
    assert status == 200
    assert out["last_run"] is not None
    assert out["last_run"]["job_id"] == data["job"]
    assert out["last_run"]["project_dir"].endswith("generated")
    assert out["last_run"]["score"] == "12/12 rubric checks"


def test_benchmark_run_endpoint_can_wait_for_report(gui, monkeypatch):
    def fake_run(job, opts):
        with core.STATE.lock:
            core.STATE.benchmark = {
                "job_id": job.id, "project_dir": str(gui.tmp / "waited"),
                "project_path": str(gui.tmp / "waited"),
                "title": "Waited run", "profile": "default",
                "report": "# report", "score": "1/1 rubric checks",
            }

    monkeypatch.setattr(jobs, "run_benchmark", fake_run)
    status, data = gui.post("/api/benchmarks/run", {
        "profile": "", "benchmark_id": "rpn_expression_calc", "wait": True})
    assert status == 200 and data["ok"] is True
    assert data["project_path"].endswith("waited")
    assert data["report"] == "# report"
