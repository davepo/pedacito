"""Built-in model benchmarking suite: five small projects that expose the
failure modes of small local models, plus an objective holdout grader.

A benchmark run creates a fresh directory, asks the selected profile to build
one of the mini-projects through the ordinary scaffolding loop, then verifies
the result twice:

* the model's own tests (``pytest -q``, or its generated verification script);
* a hidden, deterministic holdout rubric defined here.

The result is written to ``BENCHMARK_REPORT.md`` inside the generated project
and returned as a :class:`BenchmarkResult` for the GUI. Generated projects are
never deleted: the user can open, run and inspect them.

The holdout tests intentionally use only the public API promised in each task
prompt (plus the task's CLI), so they are fair to every implementation while
still catching the mistakes small models commonly make: right-associative
exponentiation, Unicode cell widths, expired-item eviction order,
``functools.wraps``, percentile maths, sliding time windows, and mismatched
DOM ids.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .config import ConfigError


# --------------------------------------------------------------------------- #
# Holdout checks
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RubricCheck:
    """One deterministic holdout check.

    ``body`` is the body of a pytest test function. The runner writes all
    check bodies into a temporary test file and maps pytest's per-test verdict
    back to ``description`` for the report.
    """

    id: str
    description: str
    body: str

    @property
    def test_name(self) -> str:
        return f"test_{self.id}"


@dataclass(frozen=True)
class Benchmark:
    id: str
    title: str
    category: str
    description: str
    what_it_tests: str
    task: str
    rubric: tuple[RubricCheck, ...]
    deliverables: tuple[str, ...]
    manual: str

    @property
    def checks(self) -> int:
        return len(self.rubric)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "what_it_tests": self.what_it_tests,
            "deliverables": list(self.deliverables),
            "checks": self.checks,
        }


@dataclass
class CheckResult:
    id: str
    description: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "description": self.description,
                "passed": self.passed, "detail": self.detail}


@dataclass
class SuiteResult:
    kind: str
    ok: bool
    passed: int
    total: int
    returncode: int
    output: str = ""
    ran: bool = True
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return max(0, self.total - self.passed)

    def to_dict(self, *, output_limit: int | None = None) -> dict:
        text = self.output
        if output_limit is not None and len(text) > output_limit:
            text = text[-output_limit:]
        return {"kind": self.kind, "ok": self.ok, "passed": self.passed,
                "failed": self.failed, "total": self.total,
                "returncode": self.returncode, "ran": self.ran,
                "output": text,
                "checks": [c.to_dict() for c in self.checks]}


# --------------------------------------------------------------------------- #
# Benchmark 1: calculator
# --------------------------------------------------------------------------- #

_RPN_TASK = r"""\
Build a Python command-line expression calculator in this project.

Deliverables:
- `calc.py` -- the calculator module and CLI entry point.
- `test_calc.py` -- pytest tests for the behaviours below.

Public API in calc.py:
- `evaluate(expression: str, variables: dict | None = None) -> int | float`
  parses and evaluates one infix expression. `variables` is a mutable dict;
  an assignment such as `x = 5` must store `x` in it and return the assigned
  value, and later expressions must see it. When `variables` is None, an
  assignment is still accepted and returns the value.
- Support +, -, *, /, **, parentheses, integer/float literals, unary plus and
  minus, and variables. `**` is right-associative, so `2 ** 3 ** 2` is 512.
  `-3 * -2` is 6. Division is true division (`7 / 2` is 3.5).
- Invalid syntax, unknown variables, and division by zero must raise a Python
  exception (ValueError/ZeroDivisionError or a subclass) with a clear message.

CLI:
- `python calc.py EXPR [EXPR ...]` evaluates the expressions left to right,
  sharing variables, and prints one result per line. Assignment prints the
  assigned value.
- `python calc.py --rpn EXPR [EXPR ...]` evaluates whitespace-separated RPN
  tokens (numbers and `+ - * / **`), one expression per argument.
- On any error print a clear message to stderr and exit with a non-zero code
  (2 is the suggested code). On success exit 0.

Use only the Python standard library (no third-party packages). Include a
`if __name__ == "__main__": main()` guard. Write real pytest tests that cover
operator precedence, unary minus, variables, RPN mode, and the error exit.
"""

_RPN_CHECKS = (
    RubricCheck(
        "infix_precedence", "Infix precedence: 2 + 3 * 4 == 14",
        r"""
calc = _module("calc")
assert _num(calc.evaluate("2 + 3 * 4")) == 14
"""),
    RubricCheck(
        "exponent_right_associative",
        "Right-associative exponentiation: 2 ** 3 ** 2 == 512",
        r"""
calc = _module("calc")
assert _num(calc.evaluate("2 ** 3 ** 2")) == 512
"""),
    RubricCheck(
        "unary_minus", "Unary minus on both operands: -3 * -2 == 6",
        r"""
calc = _module("calc")
assert _num(calc.evaluate("-3 * -2")) == 6
"""),
    RubricCheck(
        "parentheses", "Parentheses override precedence: (2 + 3) * 4 == 20",
        r"""
calc = _module("calc")
assert _num(calc.evaluate("(2 + 3) * 4")) == 20
"""),
    RubricCheck(
        "variables_persist",
        "Variable assignment persists in the supplied dict",
        r"""
calc = _module("calc")
variables = {}
assert _num(calc.evaluate("x = 5", variables)) == 5
assert _num(calc.evaluate("x * 2", variables)) == 10
"""),
    RubricCheck(
        "true_division", "True division: 7 / 2 == 3.5",
        r"""
calc = _module("calc")
assert _num(calc.evaluate("7 / 2")) == 3.5
"""),
    RubricCheck(
        "cli_infix", "CLI evaluates an infix expression and exits 0",
        r"""
proc = _run("calc.py", "2 + 3 * 4")
assert proc.returncode == 0, _failure(proc)
assert proc.stdout.strip() == "14", _failure(proc)
"""),
    RubricCheck(
        "cli_rpn", "CLI --rpn evaluates 2 3 + 4 * as 20",
        r"""
proc = _run("calc.py", "--rpn", "2 3 + 4 *")
assert proc.returncode == 0, _failure(proc)
assert proc.stdout.strip() == "20", _failure(proc)
"""),
    RubricCheck(
        "cli_variables_across_args",
        "CLI expressions share variables across arguments",
        r"""
proc = _run("calc.py", "x = 5", "x * 2")
assert proc.returncode == 0, _failure(proc)
lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
assert lines == ["5", "10"], _failure(proc)
"""),
    RubricCheck(
        "cli_division_by_zero",
        "CLI reports division by zero on stderr with a non-zero exit",
        r"""
proc = _run("calc.py", "1 / 0")
assert proc.returncode != 0, _failure(proc)
text = (proc.stderr + proc.stdout).lower()
assert "zero" in text or "division" in text, _failure(proc)
"""),
    RubricCheck(
        "cli_syntax_error",
        "CLI rejects malformed syntax with a non-zero exit",
        r"""
proc = _run("calc.py", "2 + * 3")
assert proc.returncode != 0, _failure(proc)
"""),
    RubricCheck(
        "own_tests_exist",
        "Model generated test_calc.py with pytest tests",
        r"""
text = (PROJECT / "test_calc.py").read_text(encoding="utf-8", errors="replace")
assert "def test_" in text
"""),
)


# --------------------------------------------------------------------------- #
# Benchmark 2: markdown / box table formatter
# --------------------------------------------------------------------------- #

_MDTABLE_TASK = r"""\
Build a Python Markdown and box-drawing table formatter in this project.

Deliverables:
- `mdtable.py` -- parser, formatter, and CLI.
- `test_mdtable.py` -- pytest tests.

Public API in mdtable.py:
- `parse_table(text: str) -> list[list[str]]`: parse a messy Markdown table or
  comma-separated (CSV) table. Strip outer pipes and surrounding whitespace,
  drop the Markdown separator row (`---`, `:---`, `:---:`, `---:`), and
  unescape `\|` inside cells. The first row is the header.
- `format_markdown(rows, aligns=None, max_width=0) -> str`: emit a standard
  Markdown table. `aligns` is a list of "left"/"center"/"right" per column
  (missing entries default to "left"). The separator row must use `:---`,
  `:---:`, and `---:` for those alignments. Pad every cell to the column's
  display width. Escape literal pipes in cell text as `\|`.
- `format_box(rows, aligns=None, max_width=0) -> str`: emit a Unicode
  box-drawing table using `┌ ┬ ┐`, `├ ┼ ┤`, `└ ┴ ┘`, `─`, and `│`.
- `max_width > 0` limits each cell's displayed width: truncate long cell text
  (the usual `…` ellipsis is fine) and still align the table.
- Display width must count East Asian wide characters (for example `漢字`) as
  2 columns and combining marks as 0, not `len()`.

CLI:
- `python mdtable.py --input FILE [--format markdown|box] [--align left,center,right] [--max-width N]`.
  `--input -` reads stdin. Print the formatted table to stdout and exit 0.
  Default format is markdown.

Use only the Python standard library. Include a `main()` guard. Write real
pytest tests covering Markdown parsing, CSV parsing, escaped pipes, alignment
markers, padding, and box output.
"""

_MDTABLE_CHECKS = (
    RubricCheck(
        "parse_markdown_basic", "Parses a normal Markdown table into rows",
        r"""
md = _module("mdtable")
rows = md.parse_table("| Name | Age |\n| --- | --- |\n| Ada | 36 |\n")
assert [[str(c).strip() for c in row] for row in rows] == [
    ["Name", "Age"], ["Ada", "36"]]
"""),
    RubricCheck(
        "parse_escaped_pipe", "Keeps an escaped pipe inside a Markdown cell",
        r"""
md = _module("mdtable")
rows = md.parse_table("| expr | note |\n| --- | --- |\n| a \\| b | ok |\n")
assert [[str(c).strip() for c in row] for row in rows] == [
    ["expr", "note"], ["a | b", "ok"]]
"""),
    RubricCheck(
        "parse_csv", "Parses comma-separated input",
        r"""
md = _module("mdtable")
rows = md.parse_table("name,score\nAda,10\nGrace,9\n")
assert [[str(c).strip() for c in row] for row in rows] == [
    ["name", "score"], ["Ada", "10"], ["Grace", "9"]]
"""),
    RubricCheck(
        "markdown_alignment_markers",
        "Writes :---, :---:, and ---: alignment markers",
        r"""
md = _module("mdtable")
out = md.format_markdown([["a", "b", "c"], ["1", "2", "3"]],
                         aligns=["left", "center", "right"])
assert ":---" in out and ":---:" in out and "---:" in out, out
"""),
    RubricCheck(
        "markdown_padding", "Pads all Markdown output lines to one display width",
        r"""
md = _module("mdtable")
out = md.format_markdown([["name", "score"], ["Ada", "10"], ["Grace Hopper", "9"]])
lines = [line.rstrip() for line in out.splitlines() if line.strip()]
lines = [line for line in lines if not set(line.strip()) <= set("|-: ")]
widths = {_display_width(line) for line in lines}
assert len(widths) == 1, out
"""),
    RubricCheck(
        "max_width_truncates", "Respects max_width for long cells",
        r"""
md = _module("mdtable")
out = md.format_markdown([["abcdefghij"], ["klmnopqrst"]], max_width=5)
assert "abcdefghij" not in out, out
cell = out.splitlines()[2].strip().strip("|").strip()
assert _display_width(cell) <= 5, out
"""),
    RubricCheck(
        "box_characters", "Emits Unicode box-drawing borders",
        r"""
md = _module("mdtable")
out = md.format_box([["a", "b"], ["1", "2"]])
for ch in "┌┬┐├┼┤└┴┘│─":
    assert ch in out, (ch, out)
"""),
    RubricCheck(
        "box_padding", "Pads all box-drawing lines to one display width",
        r"""
md = _module("mdtable")
out = md.format_box([["name", "score"], ["Ada", "10"]])
widths = {_display_width(line.rstrip()) for line in out.splitlines() if line.strip()}
assert len(widths) == 1, out
"""),
    RubricCheck(
        "unicode_display_width", "Uses display width for wide CJK characters",
        r"""
md = _module("mdtable")
out = md.format_markdown([["h"], ["漢字"]])
widths = {_display_width(line.rstrip()) for line in out.splitlines() if line.strip()}
assert len(widths) == 1, out
"""),
    RubricCheck(
        "cli_markdown", "CLI formats a Markdown file with alignment",
        r"""
p = tmp_path / "in.md"
p.write_text("| Name | Score |\n| --- | --- |\n| Ada | 10 |\n", encoding="utf-8")
proc = _run("mdtable.py", "--input", str(p), "--format", "markdown",
            "--align", "left,right")
assert proc.returncode == 0, _failure(proc)
assert "Ada" in proc.stdout and ":---" in proc.stdout and "---:" in proc.stdout, _failure(proc)
"""),
    RubricCheck(
        "cli_box", "CLI can emit a box-drawing table",
        r"""
p = tmp_path / "in.csv"
p.write_text("name,score\nAda,10\n", encoding="utf-8")
proc = _run("mdtable.py", "--input", str(p), "--format", "box")
assert proc.returncode == 0, _failure(proc)
assert "┌" in proc.stdout and "Ada" in proc.stdout, _failure(proc)
"""),
    RubricCheck(
        "own_tests_exist", "Model generated test_mdtable.py with pytest tests",
        r"""
text = (PROJECT / "test_mdtable.py").read_text(encoding="utf-8", errors="replace")
assert "def test_" in text
"""),
)


# --------------------------------------------------------------------------- #
# Benchmark 3: TTL + LRU cache
# --------------------------------------------------------------------------- #

_CACHE_TASK = r"""\
Build a two-file thread-safe TTL + LRU cache package in this project.

Deliverables:
- `cache.py` -- the cache implementation.
- `decorators.py` -- a memoization decorator that imports the cache.
- `test_cache.py` -- pytest tests.

Public API in cache.py:
- `class TTLCache`:
  - `TTLCache(maxsize=128, ttl=None, clock=None)`. `maxsize <= 0` means
    unlimited. `ttl` is seconds (None = never expires). `clock` is an
    injectable zero-argument callable returning seconds; default
    `time.monotonic`.
  - `set(key, value)` and `get(key, default=None)`. A successful `get`
    refreshes LRU recency; setting an existing key replaces its value and
    refreshes its TTL timestamp. `__getitem__`, `__setitem__`, `__contains__`,
    and `__len__` should also work.
  - O(1) average-time ordered LRU (an OrderedDict or dict + linked order is
    fine). When capacity is exceeded, evict the least-recently-used item.
  - Per-item TTL: an item is expired when `clock() - stored_at >= ttl`.
    Expired items are removed on access. Expired items MUST be evicted before
    live items when making room for a new key.
  - `stats` is a dict with integer keys `hits`, `misses`, `evictions`, and
    `expirations`, updated in place.
- In decorators.py:
  - `memoize(maxsize=128, ttl=None, clock=None)` returns a decorator. It must
    use `TTLCache`, preserve the wrapped function's metadata with
    `functools.wraps` (so `__name__`, `__doc__`, and `__wrapped__` work), and
    expose the cache as `wrapper.cache` (so `wrapper.cache.stats` is
    available).
  - Only hashable positional/keyword arguments are required.

Thread safety: concurrent `get`/`set` calls must not corrupt the structure or
raise. Use a `threading.RLock`. Use only the Python standard library. Include
real pytest tests with a fake clock for TTL and an eviction-order test.
"""

_CACHE_CHECKS = (
    RubricCheck(
        "basic_set_get", "TTLCache stores and returns a value",
        r"""
cache = _module("cache")
c = cache.TTLCache(maxsize=2, clock=lambda: 0)
c.set("a", 1)
assert c.get("a") == 1
assert c.get("missing") is None
"""),
    RubricCheck(
        "lru_eviction", "Evicts the least-recently-used key",
        r"""
cache = _module("cache")
c = cache.TTLCache(maxsize=2, clock=lambda: 0)
c.set("a", 1)
c.set("b", 2)
assert c.get("a") == 1          # a is now most recent
c.set("c", 3)
assert c.get("b") is None, "b should have been evicted"
assert c.get("a") == 1 and c.get("c") == 3
"""),
    RubricCheck(
        "ttl_expiry", "Expires an item at or after its TTL",
        r"""
cache = _module("cache")
now = [0.0]
c = cache.TTLCache(maxsize=4, ttl=10, clock=lambda: now[0])
c.set("a", 1)
now[0] = 10.0
assert c.get("a") is None
assert c.stats["expirations"] >= 1
"""),
    RubricCheck(
        "ttl_recency_refresh", "set() refreshes an existing key's value and TTL",
        r"""
cache = _module("cache")
now = [0.0]
c = cache.TTLCache(maxsize=4, ttl=10, clock=lambda: now[0])
c.set("a", 1)
now[0] = 6
c.set("a", 2)
now[0] = 12
assert c.get("a") == 2, "set should refresh the TTL timestamp"
"""),
    RubricCheck(
        "expired_before_capacity",
        "Removes expired items before evicting a live item",
        r"""
cache = _module("cache")
now = [0.0]
c = cache.TTLCache(maxsize=2, ttl=5, clock=lambda: now[0])
c.set("a", 1)
c.set("b", 2)
now[0] = 6
c.set("c", 3)                   # both a and b are expired
assert len(c) == 1
assert c.get("c") == 3
"""),
    RubricCheck(
        "update_existing_key", "Updating an existing key does not grow the cache",
        r"""
cache = _module("cache")
c = cache.TTLCache(maxsize=2, clock=lambda: 0)
c.set("a", 1)
c.set("a", 2)
assert len(c) == 1
assert c.get("a") == 2
"""),
    RubricCheck(
        "decorator_wraps_metadata",
        "memoize preserves __name__, __doc__, and __wrapped__",
        r"""
decorators = _module("decorators")
@decorators.memoize(maxsize=8)
def add(a, b):
    "add two numbers"
    return a + b
assert add.__name__ == "add"
assert add.__doc__ == "add two numbers"
assert getattr(add, "__wrapped__", None) is not None
"""),
    RubricCheck(
        "decorator_caches", "memoize computes each argument list once",
        r"""
decorators = _module("decorators")
calls = []
@decorators.memoize(maxsize=8)
def double(n):
    calls.append(n)
    return n * 2
assert double(3) == 6 and double(3) == 6
assert calls == [3]
assert double(4) == 8
assert calls == [3, 4]
"""),
    RubricCheck(
        "decorator_ttl", "memoize honours TTL with an injected clock",
        r"""
decorators = _module("decorators")
now = [0.0]
calls = []
@decorators.memoize(maxsize=8, ttl=5, clock=lambda: now[0])
def value():
    calls.append(1)
    return len(calls)
assert value() == 1
now[0] = 5
assert value() == 2
"""),
    RubricCheck(
        "stats_keys", "TTLCache.stats exposes hits, misses, evictions, expirations",
        r"""
cache = _module("cache")
c = cache.TTLCache(maxsize=1, clock=lambda: 0)
c.set("a", 1)
c.get("a")
c.get("nope")
c.set("b", 2)
for key in ("hits", "misses", "evictions", "expirations"):
    assert key in c.stats and isinstance(c.stats[key], int), c.stats
assert c.stats["hits"] >= 1 and c.stats["misses"] >= 1 and c.stats["evictions"] >= 1
"""),
    RubricCheck(
        "decorator_cache_attribute", "memoize wrapper exposes wrapper.cache",
        r"""
decorators = _module("decorators")
@decorators.memoize(maxsize=4)
def square(n):
    return n * n
assert hasattr(square, "cache")
square(2)
for key in ("hits", "misses", "evictions", "expirations"):
    assert key in square.cache.stats
"""),
    RubricCheck(
        "thread_safety", "Concurrent get/set calls do not raise or corrupt length",
        r"""
import threading
cache = _module("cache")
c = cache.TTLCache(maxsize=50, clock=lambda: 0)
errors = []
def worker(n):
    try:
        for i in range(100):
            c.set((n, i), i)
            c.get((n, i - 1))
    except Exception as exc:      # pragma: no cover - reported as a failure
        errors.append(exc)
threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join(10)
assert not errors, errors
assert len(c) <= 50
"""),
    RubricCheck(
        "own_tests_exist", "Model generated test_cache.py with pytest tests",
        r"""
text = (PROJECT / "test_cache.py").read_text(encoding="utf-8", errors="replace")
assert "def test_" in text
"""),
)


# --------------------------------------------------------------------------- #
# Benchmark 4: streaming log analyzer
# --------------------------------------------------------------------------- #

_LOG_TASK = r"""\
Build a Python streaming log parser and JSON report generator.

Deliverables:
- `logstat.py` -- parser, aggregation, and CLI.
- `test_logstat.py` -- pytest tests.

Input lines look like:

    [2024-01-01T12:00:00Z] INFO [api] ip=10.0.0.1 status=200 latency_ms=12.5
    [2024-01-01T12:00:05Z] ERROR [api] ip=10.0.0.2 status=503 latency_ms=40

Public API in logstat.py:
- `parse_line(line: str) -> dict | None`: return a dict with keys
  `timestamp` (a timezone-aware datetime in UTC), `level` (str), `service`
  (str), `ip` (str), `status` (int), and `latency_ms` (float). Return None for
  malformed lines.
- `percentile(values: list[float], p: float) -> float`: nearest-rank
  percentile on a sorted copy: index = ceil(p / 100 * n) - 1, clamped to
  the valid range. `percentile([10, 20, 30, 40], 50) == 20` and
  `percentile([10, 20, 30, 40], 95) == 40`.
- `analyze(lines: iterable[str]) -> dict`: return a JSON-ready report:
  ```
  {
    "total_lines": int, "parsed": int, "skipped": int,
    "services": {
      "<service>": {"count": int, "p50_ms": float, "p95_ms": float}
    },
    "error_spikes": [
      {"ip": str, "count": int, "start": ISO-8601 str, "end": ISO-8601 str}
    ]
  }
  ```
  Compute p50/p95 latency per service. Flag an error spike when the same IP
  produces MORE THAN THREE 5xx responses (status >= 500) within any rolling
  60-second window (end - start <= 60 seconds). Report each spike with the IP,
  count, and window start/end as ISO-8601 strings. Use a sliding window, not
  a fixed bucket.

CLI:
- `python logstat.py FILE` reads the file and prints `json.dumps(report,
  sort_keys=True)` to stdout, exit 0. `-` reads stdin. The JSON must be strict
  (no Python repr, no extra prose).

Use only the Python standard library. Include a `main()` guard. Write real
pytest tests covering parsing, percentiles on a small list, per-service
aggregation, and a 60-second spike window.
"""

_LOG_CHECKS = (
    RubricCheck(
        "parse_line", "parse_line extracts every field from a valid log line",
        r"""
logstat = _module("logstat")
row = logstat.parse_line(
    "[2024-01-01T12:00:00Z] INFO [api] ip=10.0.0.1 status=200 latency_ms=12.5")
assert row is not None, "valid line was rejected"
assert row["service"] == "api" and row["ip"] == "10.0.0.1"
assert int(row["status"]) == 200 and abs(float(row["latency_ms"]) - 12.5) < 1e-9
assert row["level"].upper() == "INFO"
assert row["timestamp"].year == 2024 and row["timestamp"].month == 1
"""),
    RubricCheck(
        "parse_line_rejects_malformed", "parse_line returns None for malformed input",
        r"""
logstat = _module("logstat")
assert logstat.parse_line("not a log line") is None
assert logstat.parse_line("") is None
"""),
    RubricCheck(
        "percentile_nearest_rank", "Nearest-rank percentile matches the definition",
        r"""
logstat = _module("logstat")
assert abs(logstat.percentile([10, 20, 30, 40], 50) - 20) < 1e-9
assert abs(logstat.percentile([10, 20, 30, 40], 95) - 40) < 1e-9
"""),
    RubricCheck(
        "per_service_aggregation", "analyze groups counts and p50/p95 by service",
        r"""
logstat = _module("logstat")
lines = [
    "[2024-01-01T12:00:00Z] INFO [api] ip=10.0.0.1 status=200 latency_ms=10",
    "[2024-01-01T12:00:01Z] INFO [api] ip=10.0.0.1 status=200 latency_ms=20",
    "[2024-01-01T12:00:02Z] INFO [web] ip=10.0.0.2 status=200 latency_ms=100",
    "[2024-01-01T12:00:03Z] INFO [web] ip=10.0.0.2 status=200 latency_ms=200",
]
report = logstat.analyze(lines)
api = report["services"]["api"]
web = report["services"]["web"]
assert api["count"] == 2 and web["count"] == 2
assert abs(api["p50_ms"] - 10) < 1e-9
assert abs(api["p95_ms"] - 20) < 1e-9
assert abs(web["p50_ms"] - 100) < 1e-9
assert abs(web["p95_ms"] - 200) < 1e-9
"""),
    RubricCheck(
        "error_spike_flagged", "Flags four 5xx errors from one IP inside 60 seconds",
        r"""
logstat = _module("logstat")
lines = [
    "[2024-01-01T12:00:00Z] ERROR [api] ip=10.0.0.9 status=500 latency_ms=1",
    "[2024-01-01T12:00:10Z] ERROR [api] ip=10.0.0.9 status=502 latency_ms=1",
    "[2024-01-01T12:00:20Z] ERROR [api] ip=10.0.0.9 status=503 latency_ms=1",
    "[2024-01-01T12:00:30Z] ERROR [api] ip=10.0.0.9 status=504 latency_ms=1",
]
spikes = logstat.analyze(lines)["error_spikes"]
assert any(s["ip"] == "10.0.0.9" and int(s["count"]) >= 4 for s in spikes), spikes
"""),
    RubricCheck(
        "error_spike_threshold", "Does not flag exactly three 5xx errors",
        r"""
logstat = _module("logstat")
lines = [
    "[2024-01-01T12:00:00Z] ERROR [api] ip=10.0.0.9 status=500 latency_ms=1",
    "[2024-01-01T12:00:10Z] ERROR [api] ip=10.0.0.9 status=502 latency_ms=1",
    "[2024-01-01T12:00:20Z] ERROR [api] ip=10.0.0.9 status=503 latency_ms=1",
]
assert logstat.analyze(lines)["error_spikes"] == []
"""),
    RubricCheck(
        "error_spike_window", "Does not flag errors spread beyond 60 seconds",
        r"""
logstat = _module("logstat")
lines = [
    "[2024-01-01T12:00:00Z] ERROR [api] ip=10.0.0.9 status=500 latency_ms=1",
    "[2024-01-01T12:00:30Z] ERROR [api] ip=10.0.0.9 status=500 latency_ms=1",
    "[2024-01-01T12:01:01Z] ERROR [api] ip=10.0.0.9 status=500 latency_ms=1",
    "[2024-01-01T12:01:30Z] ERROR [api] ip=10.0.0.9 status=500 latency_ms=1",
]
assert logstat.analyze(lines)["error_spikes"] == []
"""),
    RubricCheck(
        "ignores_non_5xx", "Counts only 5xx responses as errors for spikes",
        r"""
logstat = _module("logstat")
lines = [
    "[2024-01-01T12:00:00Z] ERROR [api] ip=10.0.0.9 status=404 latency_ms=1",
    "[2024-01-01T12:00:01Z] ERROR [api] ip=10.0.0.9 status=404 latency_ms=1",
    "[2024-01-01T12:00:02Z] ERROR [api] ip=10.0.0.9 status=404 latency_ms=1",
    "[2024-01-01T12:00:03Z] ERROR [api] ip=10.0.0.9 status=404 latency_ms=1",
]
assert logstat.analyze(lines)["error_spikes"] == []
"""),
    RubricCheck(
        "line_counts", "Counts parsed and skipped lines",
        r"""
logstat = _module("logstat")
lines = [
    "[2024-01-01T12:00:00Z] INFO [api] ip=10.0.0.1 status=200 latency_ms=1",
    "garbage",
    "",
    "[2024-01-01T12:00:01Z] INFO [web] ip=10.0.0.2 status=200 latency_ms=2",
]
report = logstat.analyze(lines)
assert report["total_lines"] == 4
assert report["parsed"] == 2 and report["skipped"] == 2
"""),
    RubricCheck(
        "cli_json", "CLI emits a strict JSON report",
        r"""
import json
p = tmp_path / "server.log"
p.write_text(
    "[2024-01-01T12:00:00Z] INFO [api] ip=10.0.0.1 status=200 latency_ms=10\n"
    "[2024-01-01T12:00:01Z] INFO [api] ip=10.0.0.1 status=200 latency_ms=20\n",
    encoding="utf-8")
proc = _run("logstat.py", str(p))
assert proc.returncode == 0, _failure(proc)
report = json.loads(proc.stdout)
assert report["services"]["api"]["count"] == 2
assert "error_spikes" in report
"""),
    RubricCheck(
        "own_tests_exist", "Model generated test_logstat.py with pytest tests",
        r"""
text = (PROJECT / "test_logstat.py").read_text(encoding="utf-8", errors="replace")
assert "def test_" in text
"""),
)


# --------------------------------------------------------------------------- #
# Benchmark 5: single-file Kanban board
# --------------------------------------------------------------------------- #

_KANBAN_TASK = r"""\
Build a zero-dependency, single-page interactive Kanban board in this project.

Deliverables:
- `index.html` -- the page, with three columns and the controls.
- `app.js` -- all behaviour, loaded with `<script defer src="app.js">`.
- `styles.css` -- the three-column layout and card styling.
- `verify_kanban.py` -- a standard-library verification script (see below).

Required HTML ids and structure:
- Three columns with `data-column="todo"`, `data-column="progress"`, and
  `data-column="done"`, titled `Todo`, `In Progress`, and `Done`. Each column
  contains a task list with ids `todo-list`, `progress-list`, `done-list`.
- A creation form with id `task-form`, text input `task-title`, and a select
  `task-priority` with options `low`, `medium`, `high`.
- A filter select with id `priority-filter` and options `all`, `low`,
  `medium`, `high`.
- Buttons with ids `export-btn` and `import-btn`, plus a hidden file input with
  id `import-file`.
- Each task element has class `task`, `data-id`, and `data-priority`, with
  `move-left` and `move-right` buttons inside it.

Behaviour in app.js:
- Create a task from the form (default priority medium) and append it to Todo.
- `move-left` / `move-right` move the task between the three columns and stop
  at the ends. Event delegation on the board is fine.
- The priority filter shows only tasks with the selected priority (`all` shows
  everything).
- Persist tasks to `localStorage` under the exact key `pedacito-kanban-v1`,
  and load them on startup. Saving must happen after every mutation.
- `export-btn` downloads the task list as JSON (Blob + object URL is fine).
- `import-btn` opens `import-file`; reading a JSON array replaces the board
  state and persists it.

styles.css must give the board a three-column layout using CSS grid or flex
(the columns should sit side by side on a normal desktop width). No CDNs, no
frameworks, no external assets: everything must work offline by opening
index.html.

verify_kanban.py:
- A standard-library-only Python script, no arguments, that checks the three
  files contain the required ids/classes and exits 0 with `print("OK")` on
  success or non-zero with a clear message on failure.
"""

_KANBAN_CHECKS = (
    RubricCheck(
        "required_files", "All four deliverables exist",
        r"""
for name in ("index.html", "app.js", "styles.css", "verify_kanban.py"):
    assert (PROJECT / name).is_file(), f"missing {name}"
"""),
    RubricCheck(
        "assets_linked", "index.html links styles.css and app.js",
        r"""
html = (PROJECT / "index.html").read_text(encoding="utf-8", errors="replace")
assert "styles.css" in html and "app.js" in html
"""),
    RubricCheck(
        "three_columns", "The page defines the three Kanban columns",
        r"""
html = (PROJECT / "index.html").read_text(encoding="utf-8", errors="replace")
low = html.lower()
for word in ("todo", "progress", "done"):
    assert word in low, f"missing column {word}"
assert low.count("data-column") >= 3, html[:2000]
"""),
    RubricCheck(
        "required_dom_ids", "All control/list ids used by the UI exist in HTML",
        r"""
html = (PROJECT / "index.html").read_text(encoding="utf-8", errors="replace")
ids = _html_ids(html)
required = {
    "todo-list", "progress-list", "done-list", "task-form", "task-title",
    "task-priority", "priority-filter", "export-btn", "import-btn",
    "import-file",
}
missing = sorted(required - ids)
assert not missing, f"missing ids: {missing}"
"""),
    RubricCheck(
        "js_ids_match_html", "Every id referenced by app.js exists in index.html",
        r"""
html = (PROJECT / "index.html").read_text(encoding="utf-8", errors="replace")
js = (PROJECT / "app.js").read_text(encoding="utf-8", errors="replace")
ids = _html_ids(html)
referenced = set(re.findall(r"getElementById\(\s*[\"']([^\"']+)", js))
referenced |= set(re.findall(
    r"querySelector(?:All)?\(\s*[\"']#([A-Za-z0-9_-]+)", js))
missing = sorted(referenced - ids)
assert not missing, f"app.js references ids absent from index.html: {missing}"
"""),
    RubricCheck(
        "init_wiring", "app.js initialises after the DOM is ready",
        r"""
js = (PROJECT / "app.js").read_text(encoding="utf-8", errors="replace")
html = (PROJECT / "index.html").read_text(encoding="utf-8", errors="replace")
ready = ("DOMContentLoaded" in js or "window.onload" in js
         or re.search(r"<script[^>]*\bdefer\b", html, re.I))
assert ready, "no DOM-ready hook and no deferred script"
"""),
    RubricCheck(
        "move_buttons", "app.js handles move-left/move-right",
        r"""
js = (PROJECT / "app.js").read_text(encoding="utf-8", errors="replace")
assert "move-left" in js and "move-right" in js, "missing move button wiring"
assert "addEventListener" in js
"""),
    RubricCheck(
        "localstorage", "Persists tasks under pedacito-kanban-v1",
        r"""
js = (PROJECT / "app.js").read_text(encoding="utf-8", errors="replace")
assert "localStorage" in js
assert "pedacito-kanban-v1" in js
assert "setItem" in js and "getItem" in js
"""),
    RubricCheck(
        "priority_filter", "Priority filtering is implemented",
        r"""
html = (PROJECT / "index.html").read_text(encoding="utf-8", errors="replace").lower()
js = (PROJECT / "app.js").read_text(encoding="utf-8", errors="replace").lower()
for value in ("low", "medium", "high"):
    assert value in html or value in js, f"missing priority {value}"
assert "priority" in js and "filter" in js
"""),
    RubricCheck(
        "export_import", "JSON export and import are implemented",
        r"""
js = (PROJECT / "app.js").read_text(encoding="utf-8", errors="replace")
assert "JSON.stringify" in js and "JSON.parse" in js
assert "Blob" in js or "createObjectURL" in js or "download" in js
assert "FileReader" in js or ".text()" in js
"""),
    RubricCheck(
        "css_columns", "styles.css lays the board out in three columns",
        r"""
css = (PROJECT / "styles.css").read_text(encoding="utf-8", errors="replace")
assert re.search(r"grid-template-columns|display\s*:\s*(grid|flex)", css, re.I)
assert re.search(r"repeat\(\s*3|1fr[^;]*1fr[^;]*1fr", css, re.I) or "flex" in css.lower()
"""),
    RubricCheck(
        "no_external_assets", "No CDN or external runtime dependency",
        r"""
html = (PROJECT / "index.html").read_text(encoding="utf-8", errors="replace")
assert not re.search(r"(?:src|href)\s*=\s*[\"']https?://", html, re.I), \
    "external URL found in index.html"
"""),
    RubricCheck(
        "verify_script_runs", "verify_kanban.py exits 0",
        r"""
proc = _run("verify_kanban.py")
assert proc.returncode == 0, _failure(proc)
"""),
)


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #

BENCHMARKS: dict[str, Benchmark] = {
    b.id: b for b in (
        Benchmark(
            id="rpn_expression_calc",
            title="Stateful CLI Expression & RPN Calculator",
            category="Parsers, arithmetic, CLI",
            description=("A Python CLI/module for infix expressions with "
                         "precedence, variables, RPN mode, and clear errors."),
            what_it_tests=("Parser state machines, right-associative "
                           "exponentiation, unary minus, variable persistence, "
                           "CLI argument handling and exit codes."),
            task=_RPN_TASK,
            rubric=_RPN_CHECKS,
            deliverables=("calc.py", "test_calc.py"),
            manual=textwrap.dedent("""\
                ```bash
                cd <this folder>
                python calc.py "2 + 3 * 4"
                python calc.py "x = 5" "x ** 2"
                python calc.py --rpn "2 3 + 4 *"
                python calc.py "1 / 0"     # expected: error + non-zero exit
                pytest -q
                ```
                """),
        ),
        Benchmark(
            id="markdown_table_fmt",
            title="Markdown Table & Box-Drawing Formatter",
            category="Formatting, Unicode, CLI",
            description=("Parse messy Markdown/CSV tables and render aligned "
                         "Markdown or Unicode box-drawing tables."),
            what_it_tests=("Off-by-one padding, escaped pipes, East Asian "
                           "display widths, alignment markers, truncation, and "
                           "remembering all formatting requirements."),
            task=_MDTABLE_TASK,
            rubric=_MDTABLE_CHECKS,
            deliverables=("mdtable.py", "test_mdtable.py"),
            manual=textwrap.dedent("""\
                ```bash
                cd <this folder>
                python mdtable.py --input sample.md --format markdown
                python mdtable.py --input sample.csv --format box --max-width 8
                pytest -q
                ```
                """),
        ),
        Benchmark(
            id="ttl_lru_cache",
            title="Thread-Safe TTL + LRU Cache with Decorator & Stats",
            category="Data structures, time, concurrency",
            description=("An ordered LRU cache with per-item TTL, injectable "
                         "clock, memoize decorator, and statistics."),
            what_it_tests=("Eviction order, expired-before-capacity eviction, "
                           "functools.wraps metadata, cross-file imports, "
                           "injectable time, and thread safety."),
            task=_CACHE_TASK,
            rubric=_CACHE_CHECKS,
            deliverables=("cache.py", "decorators.py", "test_cache.py"),
            manual=textwrap.dedent("""\
                ```bash
                cd <this folder>
                python -c "from cache import TTLCache; c=TTLCache(2); c.set('a',1); print(c.get('a'), c.stats)"
                pytest -q
                ```
                """),
        ),
        Benchmark(
            id="log_anomaly_analyzer",
            title="Streaming Log Parser & JSON Report Generator",
            category="Regex, time windows, JSON",
            description=("Parse mixed server logs, compute p50/p95 per service, "
                         "flag 5xx spikes, and emit strict JSON."),
            what_it_tests=("Regex accuracy, nearest-rank percentile maths on "
                           "small lists, rolling 60-second windows, and JSON "
                           "schema discipline."),
            task=_LOG_TASK,
            rubric=_LOG_CHECKS,
            deliverables=("logstat.py", "test_logstat.py"),
            manual=textwrap.dedent("""\
                ```bash
                cd <this folder>
                printf '[2024-01-01T12:00:00Z] INFO [api] ip=10.0.0.1 status=200 latency_ms=12.5\\n' > sample.log
                python logstat.py sample.log
                pytest -q
                ```
                """),
        ),
        Benchmark(
            id="mini_kanban_html",
            title="Single-File Interactive Kanban Board",
            category="HTML/CSS/JS, DOM, storage",
            description=("A zero-dependency three-column Kanban board with "
                         "filters, move buttons, localStorage, and JSON I/O."),
            what_it_tests=("DOM id consistency between HTML and JS, event "
                           "wiring, state persistence, CSS layout, and offline "
                           "zero-dependency delivery."),
            task=_KANBAN_TASK,
            rubric=_KANBAN_CHECKS,
            deliverables=("index.html", "app.js", "styles.css",
                          "verify_kanban.py"),
            manual=textwrap.dedent("""\
                ```bash
                cd <this folder>
                python verify_kanban.py
                # open index.html in a browser, add/move/filter tasks,
                # then reload to confirm localStorage persistence
                ```
                """),
        ),
    )
}


def list_benchmarks() -> list[dict]:
    """The GUI/API catalog, in stable order."""
    return [b.to_dict() for b in BENCHMARKS.values()]


def get_benchmark(benchmark_id: str) -> Benchmark:
    try:
        return BENCHMARKS[str(benchmark_id)]
    except KeyError as exc:
        known = ", ".join(BENCHMARKS)
        raise ConfigError(f"Unknown benchmark {benchmark_id!r}. "
                          f"Choose one of: {known}") from exc


benchmark_catalog = list_benchmarks


# --------------------------------------------------------------------------- #
# Holdout test source
# --------------------------------------------------------------------------- #

_HOLDOUT_HEADER = r'''"""Hidden holdout rubric generated by pedacito.benchmarks (do not edit)."""
from __future__ import annotations

import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

PROJECT = Path(os.environ.get("PEDC_BENCH_PROJECT", ".")).resolve()
PY = sys.executable


def _module(name):
    """Import a generated project module, even when cwd is the temp dir."""
    sys.path.insert(0, str(PROJECT))
    return importlib.import_module(name)


def _num(value):
    return float(value)


def _display_width(text):
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(text))
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def _run(script, *args, timeout=30):
    return subprocess.run(
        [PY, str(PROJECT / script), *map(str, args)],
        cwd=str(PROJECT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout)


def _failure(proc):
    return (f"exit={proc.returncode}\nstdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-2000:]}")


def _html_ids(html):
    return set(re.findall(r"""\bid\s*=\s*["']([^"']+)["']""", html))


'''


def holdout_source(benchmark: Benchmark | str) -> str:
    """The complete pytest file for a benchmark's hidden rubric."""
    if isinstance(benchmark, str):
        benchmark = get_benchmark(benchmark)
    parts = [_HOLDOUT_HEADER]
    for check in benchmark.rubric:
        body = textwrap.indent(textwrap.dedent(check.body).strip(), "    ")
        # Every check receives tmp_path even when it does not use it: pytest
        # only injects fixtures declared as parameters, and several checks
        # write sample input files.
        parts.append(f"def {check.test_name}(tmp_path):\n{body}\n")
    return "\n\n".join(parts).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Running suites
# --------------------------------------------------------------------------- #

def _pytest_command(test_path: str | Path) -> list[str]:
    return [sys.executable, "-m", "pytest", str(test_path), "-q", "-rA",
            "--tb=short", "-p", "no:cacheprovider"]


def _pytest_counts(output: str) -> tuple[int, int, int]:
    def count(pattern: str) -> int:
        m = re.search(r"(\d+)\s+" + pattern, output)
        return int(m.group(1)) if m else 0
    return (count(r"passed"), count(r"failed"),
            count(r"errors?\b"))


def _suite_from_pytest(kind: str, proc: subprocess.CompletedProcess,
                       checks: list[CheckResult] | None = None) -> SuiteResult:
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    passed_n, failed_n, error_n = _pytest_counts(output)
    checks = checks or []
    if checks:
        passed = sum(1 for c in checks if c.passed)
        total = len(checks)
        return SuiteResult(kind=kind, ok=passed == total,
                           passed=passed, total=total,
                           returncode=proc.returncode, output=output,
                           ran=True, checks=checks)
    total = passed_n + failed_n + error_n
    if total == 0:
        # A collection error, an empty test run, or a crashed runner still
        # counts as one failed attempt rather than 0/0.
        total = passed_n or 1
    return SuiteResult(kind=kind, ok=proc.returncode == 0 and passed_n > 0,
                       passed=passed_n, total=total,
                       returncode=proc.returncode, output=output,
                       ran=True)


def run_model_tests(project_dir: Path, timeout: int = 180) -> SuiteResult:
    """Run the tests the model generated (`pytest -q`), or its verification
    script for the HTML benchmark when it generated no pytest file."""
    project_dir = Path(project_dir)
    test_files = [
        p for p in project_dir.rglob("*.py")
        if p.name.startswith("test_") or p.name.endswith("_test.py")
    ]
    test_files = [p for p in test_files
                  if ".pedacito" not in p.parts and ".git" not in p.parts
                  and "node_modules" not in p.parts]
    if test_files:
        cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short",
               "-p", "no:cacheprovider"]
        try:
            proc = subprocess.run(cmd, cwd=str(project_dir),
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return SuiteResult(
                "model_tests", False, 0, 0, -1,
                output=f"Model tests timed out after {timeout}s.\n{exc}",
                ran=True)
        return _suite_from_pytest("model_tests", proc)
    verify = project_dir / "verify_kanban.py"
    if verify.is_file():
        try:
            proc = subprocess.run(
                [sys.executable, str(verify)], cwd=str(project_dir),
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return SuiteResult("model_tests", False, 0, 0, -1,
                               output=f"verify_kanban.py timed out.\n{exc}")
        output = (proc.stdout or "") + (("\n" + proc.stderr)
                                       if proc.stderr else "")
        return SuiteResult("model_tests", proc.returncode == 0,
                           1 if proc.returncode == 0 else 0, 1,
                           proc.returncode, output=output, ran=True)
    return SuiteResult(
        "model_tests", False, 0, 0, -1, ran=False,
        output="No model-generated test files were found "
               "(expected test_*.py, or verify_kanban.py for the HTML task).")


def _parse_holdout_verdicts(output: str,
                            checks: Iterable[RubricCheck]) -> dict[str, str]:
    """Map ``test_name`` -> the pytest short-summary line."""
    found: dict[str, str] = {}
    line_re = re.compile(
        r"^(PASSED|FAILED|ERROR)\s+\S+::(test_[A-Za-z0-9_]+)(?:\s+-\s+(.*))?$")
    for raw in output.splitlines():
        line = raw.strip()
        m = line_re.match(line)
        if m:
            status, name, detail = m.group(1), m.group(2), m.group(3) or ""
            found[name] = f"{status}" + (f": {detail}" if detail else "")
    return found


def run_holdout(project_dir: Path, benchmark: Benchmark | str,
                timeout: int = 180) -> SuiteResult:
    """Run the hidden rubric against a generated project."""
    project_dir = Path(project_dir)
    if isinstance(benchmark, str):
        benchmark = get_benchmark(benchmark)
    source = holdout_source(benchmark)
    with tempfile.TemporaryDirectory(prefix="pedacito_bench_") as td:
        test_path = Path(td) / f"test_holdout_{benchmark.id}.py"
        test_path.write_text(source, encoding="utf-8")
        env = os.environ.copy()
        old_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(project_dir) + (
            os.pathsep + old_path if old_path else "")
        env["PEDC_BENCH_PROJECT"] = str(project_dir)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        cmd = _pytest_command(test_path)
        try:
            proc = subprocess.run(cmd, cwd=td, env=env, capture_output=True,
                                  text=True, encoding="utf-8",
                                  errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            output = (f"Holdout rubric timed out after {timeout}s.\n{exc}\n")
            results = [CheckResult(c.id, c.description, False,
                                   "timed out") for c in benchmark.rubric]
            return SuiteResult("holdout", False, 0, len(results), -1,
                               output=output, ran=True, checks=results)
    output = (proc.stdout or "") + (("\n" + proc.stderr)
                                   if proc.stderr else "")
    verdicts = _parse_holdout_verdicts(output, benchmark.rubric)
    checks: list[CheckResult] = []
    for check in benchmark.rubric:
        verdict = verdicts.get(check.test_name, "")
        passed = verdict.startswith("PASSED")
        detail = ""
        if verdict and not passed:
            detail = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
        elif not verdict:
            detail = "not run (pytest collection or setup error)"
        checks.append(CheckResult(check.id, check.description, passed,
                                  detail[:500]))
    return _suite_from_pytest("holdout", proc, checks)


def run_rubric(project_dir: Path, benchmark: Benchmark | str,
               timeout: int = 180) -> SuiteResult:
    """Alias used by callers that want only the hidden score."""
    return run_holdout(project_dir, benchmark, timeout=timeout)


# --------------------------------------------------------------------------- #
# Project directories & default location
# --------------------------------------------------------------------------- #

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str, fallback: str = "default") -> str:
    text = _SAFE_NAME.sub("_", str(value or "").strip()).strip("._-")
    return text[:80] or fallback


def project_dir_name(benchmark_id: str, profile_name: str,
                     stamp: str | None = None) -> str:
    """The unique-by-construction folder name for one run.

    ``pedacito_bench_<benchmark_id>_<profile>_<short_timestamp>``; the caller
    adds a numeric suffix if that exact name already exists.
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return (f"pedacito_bench_{_slug(benchmark_id)}_{_slug(profile_name)}_"
            f"{_slug(stamp, 'now')}")


def default_benchmarks_root(cfg=None, current_root: Path | str | None = None,
                            cwd: Path | str | None = None) -> Path:
    """Where benchmark runs go when the user does not pick a parent folder.

    Order: the config's ``benchmarks_dir``; a ``benchmarks/`` folder beside the
    currently open project; ``./benchmarks``. The folder is not created here.
    """
    configured = str(getattr(cfg, "benchmarks_dir", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    if current_root:
        try:
            return Path(current_root).expanduser().resolve().parent / "benchmarks"
        except OSError:
            pass
    return Path(cwd or Path.cwd()).expanduser().resolve() / "benchmarks"


def benchmark_project_dir(benchmark_id: str, profile_name: str,
                          base_dir: Path | str, *, stamp: str | None = None,
                          create: bool = True) -> Path:
    """Create (unless ``create=False``) a fresh, uniquely named run folder."""
    base = Path(base_dir).expanduser()
    name = project_dir_name(benchmark_id, profile_name, stamp=stamp)
    candidate = base / name
    n = 2
    while candidate.exists():
        candidate = base / f"{name}_{n}"
        n += 1
    if create:
        candidate.mkdir(parents=True, exist_ok=False)
    return candidate


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

@dataclass
class BenchmarkResult:
    benchmark_id: str
    title: str
    category: str
    profile: str
    model: str
    project_dir: Path
    base_dir: Path
    started_at: str
    duration: float
    generation_rc: int
    generation_error: str
    stats: dict
    repair_rounds: int
    requirement_ledger: list[dict]
    model_tests: SuiteResult
    rubric: SuiteResult
    report_path: Path | None = None
    report_text: str = ""
    job_id: int = 0

    @property
    def score(self) -> str:
        return f"{self.rubric.passed}/{self.rubric.total} rubric checks"

    @property
    def shortcomings(self) -> list[str]:
        out: list[str] = []
        if self.generation_error:
            out.append(f"Generation failed: {self.generation_error}")
        if not self.model_tests.ok:
            if not self.model_tests.ran:
                out.append("Model's own tests: none were generated.")
            else:
                out.append(
                    f"Model's own tests failed ({self.model_tests.passed}/"
                    f"{self.model_tests.total} passed).")
        for check in self.rubric.checks:
            if not check.passed:
                text = check.description
                if check.detail and not check.detail.startswith("not run"):
                    text += f" ({check.detail.splitlines()[0][:240]})"
                out.append(text)
        return out

    def to_dict(self, *, include_report: bool = True,
                output_limit: int = 12000) -> dict:
        data = {
            "benchmark_id": self.benchmark_id,
            "title": self.title,
            "category": self.category,
            "profile": self.profile,
            "model": self.model,
            "project_dir": str(self.project_dir),
            "project_path": str(self.project_dir),
            "base_dir": str(self.base_dir),
            "started_at": self.started_at,
            "duration": round(self.duration, 2),
            "generation_rc": self.generation_rc,
            "generation_error": self.generation_error,
            "stats": dict(self.stats),
            "repair_rounds": self.repair_rounds,
            "requirement_ledger": [dict(r) for r in self.requirement_ledger],
            "model_tests": self.model_tests.to_dict(output_limit=output_limit),
            "rubric": self.rubric.to_dict(output_limit=output_limit),
            "score": self.score,
            "shortcomings": self.shortcomings,
            "report_path": str(self.report_path) if self.report_path else "",
            "job_id": self.job_id,
        }
        if include_report:
            data["report"] = self.report_text
        return data


class _LineTee:
    """Write through to the real stderr while keeping a copy and emitting
    complete lines to a progress callback."""

    def __init__(self, real, capture: io.StringIO,
                 on_line: Callable[[str], None] | None = None):
        self.real = real
        self.capture = capture
        self.on_line = on_line
        self.pending = ""

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        try:
            self.real.write(text)
        except (OSError, ValueError):
            pass
        self.capture.write(text)
        if self.on_line:
            self.pending += text
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                if line.strip():
                    try:
                        self.on_line(line.rstrip())
                    except Exception:
                        pass
        return len(text)

    def flush(self):
        try:
            self.real.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self.real, name)


def _stats_delta(before: dict, after: dict) -> dict:
    keys = ("calls", "prompt_tokens", "completion_tokens", "cached_tokens",
            "cache_write_tokens", "cost", "seconds")
    out = {k: after.get(k, 0) - before.get(k, 0) for k in keys}
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    before_site = before.get("by_site") if isinstance(
        before.get("by_site"), dict) else {}
    after_site = after.get("by_site") if isinstance(
        after.get("by_site"), dict) else {}
    before_repair = int(((before_site.get("repair") or {})
                         ).get("calls", 0))
    after_repair = int(((after_site.get("repair") or {})
                        ).get("calls", 0))
    out["repair_calls"] = max(0, after_repair - before_repair)
    return out


def _delete_bytecode(project: Path) -> None:
    for cache in list(project.rglob("__pycache__")):
        if ".pedacito" in cache.parts:
            continue
        for p in cache.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            cache.rmdir()
        except OSError:
            pass


def _count_repairs(captured: str) -> int:
    n = len(re.findall(r"\[verify failed .*?repairing", captured))
    n += len(re.findall(r"fixed \d+ import issue", captured))
    n += len(re.findall(r"\bAPI mismatch", captured, re.I))
    return n


def _write_task_file(project: Path, benchmark: Benchmark) -> None:
    try:
        (project / "BENCHMARK_TASK.md").write_text(
            f"# Benchmark Task: {benchmark.title}\n\n"
            f"- **Benchmark id:** `{benchmark.id}`\n"
            f"- **Category:** {benchmark.category}\n"
            f"- **Holdout checks:** {benchmark.checks}\n\n"
            "## Prompt given to the model\n\n"
            "~~~~markdown\n" + benchmark.task.strip() + "\n~~~~\n",
            encoding="utf-8")
    except OSError:
        pass


def _generate_with_scaffold(project: Path, cfg, client,
                            benchmark: Benchmark) -> int:
    """Run the normal plan -> skeleton -> build loop for one benchmark."""
    from . import scaffold
    spec = scaffold.Spec(text=benchmark.task,
                         name_hint=benchmark.title)
    return scaffold.run_new(
        project, cfg, client, spec, apply=True,
        repair_rounds=getattr(cfg, "auto_repair_rounds", 2),
        verbose=True,
    )


def run_benchmark(profile: str = "", benchmark_id: str = "",
                  base_dir: Path | str | None = None, *,
                  cfg=None, client=None, config_root: Path | str | None = None,
                  current_root: Path | str | None = None,
                  generator: Callable[[Path, object, object], int | None] | None = None,
                  model_timeout: int = 180, holdout_timeout: int = 180,
                  log: Callable[[str], None] | None = None,
                  stamp: str | None = None,
                  job_id: int = 0,
                  profile_name: str | None = None) -> BenchmarkResult:
    """Run one benchmark end to end and write its report.

    ``generator`` is injectable for tests/dry runs: it receives
    ``(project_dir, cfg, client)`` and returns an exit code (or None). The
    default generator uses :func:`pedacito.scaffold.run_new` with
    ``apply=True``.
    """
    benchmark = get_benchmark(benchmark_id)
    if profile_name is not None:
        profile = profile_name
    if cfg is None:
        from . import config as config_mod
        cfg = config_mod.load(
            root=Path(config_root).expanduser() if config_root else None,
            profile=profile or None, require_hardware=False,
            warn_budgets=False)
    profile_name = str(getattr(cfg, "active_profile", "") or profile or "default")
    model = str(getattr(cfg, "model", "") or "")

    if base_dir is None or not str(base_dir).strip():
        base = default_benchmarks_root(cfg, current_root=current_root)
    else:
        base = Path(base_dir).expanduser()
    project = benchmark_project_dir(benchmark.id, profile_name, base,
                                    stamp=stamp)
    started_wall = time.time()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def say(line: str) -> None:
        if log is not None:
            try:
                log(line)
            except Exception:
                pass

    say(f"Benchmark {benchmark.id} (`{benchmark.title}`) for profile "
        f"'{profile_name}' ({model})")
    say(f"Project directory: {project}")

    if client is None:
        from .llm import LLMClient
        client = LLMClient(cfg)
    before = dict(getattr(client, "stats", {}) or {})
    if generator is None:
        def gen(project, cfg, client):
            return _generate_with_scaffold(project, cfg, client, benchmark)
    else:
        gen = generator
    captured = io.StringIO()
    generation_error = ""
    rc = 0
    say("Generating the project (scaffold plan -> skeleton -> build)...")
    tee = _LineTee(sys.stderr, captured, on_line=log)
    try:
        with contextlib.redirect_stderr(tee):
            result_rc = gen(project, cfg, client)
        rc = 0 if result_rc is None else int(result_rc)
    except KeyboardInterrupt:
        raise
    except Exception as exc:                     # report the failure, keep grading
        generation_error = f"{type(exc).__name__}: {exc}"
        rc = 1
        say(f"Generation failed: {generation_error}")
    finally:
        try:
            tee.flush()
        except Exception:
            pass
    after = dict(getattr(client, "stats", {}) or {})
    stats = _stats_delta(before, after)
    stats["repair_calls"] = max(stats.get("repair_calls", 0),
                                _count_repairs(captured.getvalue()))
    _write_task_file(project, benchmark)

    say("Running the model's own tests...")
    model_tests = run_model_tests(project, timeout=model_timeout)
    say(f"Model tests: {model_tests.passed}/{model_tests.total} "
        f"({'passed' if model_tests.ok else 'failed'})")

    say("Running the hidden holdout rubric...")
    rubric = run_holdout(project, benchmark, timeout=holdout_timeout)
    say(f"Holdout rubric: {rubric.passed}/{rubric.total} checks passed")
    _delete_bytecode(project)

    ledger = [{"requirement": c.description,
               "state": "VERIFIED" if c.passed else "FAILED"}
              for c in rubric.checks]
    duration = time.time() - started_wall
    result = BenchmarkResult(
        benchmark_id=benchmark.id, title=benchmark.title,
        category=benchmark.category, profile=profile_name, model=model,
        project_dir=project, base_dir=base, started_at=started_at,
        duration=duration, generation_rc=rc,
        generation_error=generation_error, stats=stats,
        repair_rounds=int(stats.get("repair_calls", 0)),
        requirement_ledger=ledger, model_tests=model_tests, rubric=rubric,
        job_id=job_id)
    result.report_path, result.report_text = write_report(result, benchmark)
    say(f"Report written: {result.report_path}")
    return result


def generate_report(result: BenchmarkResult,
                    benchmark: Benchmark | str | None = None) -> str:
    """Render ``BENCHMARK_REPORT.md`` (pure function, easy to test)."""
    if benchmark is None:
        benchmark = get_benchmark(result.benchmark_id)
    elif isinstance(benchmark, str):
        benchmark = get_benchmark(benchmark)
    stats = result.stats or {}
    mt, rub = result.model_tests, result.rubric
    overall = ("PASS" if rub.ok and mt.ok else
               "PARTIAL" if rub.passed or mt.passed else "FAIL")
    lines = [
        f"# BENCHMARK REPORT — {result.title}",
        "",
        "> Generated by Pedacito's built-in model benchmarking suite. "
        "The project is left on disk so you can inspect and run it yourself.",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Benchmark | `{result.benchmark_id}` — {result.title} |",
        f"| Category | {result.category} |",
        f"| Profile | `{result.profile}` |",
        f"| Model ID | `{result.model}` |",
        f"| Started (UTC) | {result.started_at} |",
        f"| Duration | {result.duration:.1f} s |",
        f"| Project | `{result.project_dir}` |",
        f"| Generation exit | {result.generation_rc} |",
        "",
        "## Score",
        "",
        "| Verification | Result |",
        "| --- | --- |",
    ]
    if mt.ran:
        mt_mark = "✅" if mt.ok else "❌"
        mt_text = f"{mt_mark} {mt.passed}/{mt.total} passed"
        if mt.returncode:
            mt_text += f" (exit {mt.returncode})"
    else:
        mt_text = "⚠️ no model-generated tests found"
    rub_mark = "✅" if rub.ok else ("⚠️" if rub.passed else "❌")
    lines += [
        f"| Model's own tests (`pytest -q`) | {mt_text} |",
        f"| Harness holdout rubric | {rub_mark} {rub.passed}/{rub.total} "
        f"checks passed |",
        f"| Overall | **{overall}** |",
        "",
        "## Usage",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Model calls | {int(stats.get('calls', 0))} |",
        f"| Prompt tokens | {int(stats.get('prompt_tokens', 0)):,} |",
        f"| Completion tokens | {int(stats.get('completion_tokens', 0)):,} |",
        f"| Total tokens | {int(stats.get('total_tokens', 0)):,} |",
        f"| Cached tokens | {int(stats.get('cached_tokens', 0)):,} |",
        f"| Repair rounds / calls | {result.repair_rounds} |",
        f"| Wall clock | {result.duration:.1f} s |",
    ]
    if stats.get("cost"):
        lines.append(f"| Reported cost | ${float(stats['cost']):.4f} |")
    lines += [
        "",
        "## Shortcomings Detected",
        "",
    ]
    shortcomings = result.shortcomings
    if shortcomings:
        lines += [f"- {item}" for item in shortcomings]
    else:
        lines.append("- None. Every model test and every holdout check passed.")
    lines += [
        "",
        "## Harness Holdout Rubric",
        "",
        "| Check | Description | Result |",
        "| --- | --- | --- |",
    ]
    for check in rub.checks:
        mark = "✅" if check.passed else "❌"
        detail = ("passed" if check.passed else
                  (check.detail.splitlines()[0][:180] if check.detail else "failed"))
        detail = str(detail).replace("|", "\\|")
        lines.append(f"| `{check.id}` | {check.description} | {mark} {detail} |")
    lines += [
        "",
        "## Requirement Ledger",
        "",
        "| Requirement | State |",
        "| --- | --- |",
    ]
    for row in result.requirement_ledger:
        lines.append(f"| {row['requirement']} | `{row['state']}` |")
    lines += [
        "",
        "## Model Test Output",
        "",
        "~~~~text",
        (mt.output or "(no output)").strip()[-6000:],
        "~~~~",
        "",
        "## Holdout Rubric Output",
        "",
        "~~~~text",
        (rub.output or "(no output)").strip()[-6000:],
        "~~~~",
        "",
        "## Run It Yourself",
        "",
        f"```bash\ncd {shlex.quote(str(result.project_dir))}\n```",
        "",
        benchmark.manual.strip(),
        "",
        "The hidden holdout rubric is not part of the project, so the tests "
        "above are exactly the ones the model wrote. Use the scorecard to see "
        "which behaviours it missed.",
        "",
        "## Generated Files",
        "",
    ]
    files = []
    if result.project_dir.is_dir():
        files = sorted(
            p.relative_to(result.project_dir).as_posix()
            for p in result.project_dir.rglob("*")
            if p.is_file() and ".pedacito" not in p.parts
            and ".git" not in p.parts
        )
    if files:
        lines += [f"- `{name}`" for name in files[:120]]
        if len(files) > 120:
            lines.append(f"- … and {len(files) - 120} more")
    else:
        lines.append("- (no files were generated)")
    lines.append("")
    return "\n".join(lines)


def write_report(result: BenchmarkResult,
                 benchmark: Benchmark | str | None = None) -> tuple[Path, str]:
    """Write ``BENCHMARK_REPORT.md`` and ``benchmark_run.json`` in the project."""
    if benchmark is None:
        benchmark = get_benchmark(result.benchmark_id)
    text = generate_report(result, benchmark)
    report = result.project_dir / "BENCHMARK_REPORT.md"
    report.write_text(text, encoding="utf-8")
    result.report_path = report
    result.report_text = text
    try:
        data = result.to_dict(include_report=False, output_limit=4000)
        (result.project_dir / "benchmark_run.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    except OSError:
        pass
    return report, text


# --------------------------------------------------------------------------- #
# Recent runs for the GUI
# --------------------------------------------------------------------------- #

# Convenient alias for callers that talk about a benchmark "run" record.
BenchmarkRun = BenchmarkResult


def recent_runs(base_dir: Path | str, limit: int = 8) -> list[dict]:
    """Summaries of the newest runs under one benchmarks root."""
    base = Path(base_dir).expanduser()
    if not base.is_dir():
        return []
    found: list[tuple[float, dict]] = []
    try:
        dirs = [p for p in base.iterdir()
                if p.is_dir() and p.name.startswith("pedacito_bench_")]
    except OSError:
        return []
    for d in dirs:
        meta = d / "benchmark_run.json"
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            mtime = meta.stat().st_mtime
        except OSError:
            mtime = 0.0
        data = {k: data.get(k) for k in (
            "benchmark_id", "title", "category", "profile", "model",
            "project_dir", "started_at", "duration", "score",
            "generation_rc", "stats", "shortcomings", "rubric",
            "model_tests", "report_path")}
        data["project_dir"] = str(d)
        data["project_path"] = str(d)
        report = d / "BENCHMARK_REPORT.md"
        data["report_path"] = str(report)
        try:
            data["report"] = report.read_text(encoding="utf-8",
                                              errors="replace")[-60_000:]
        except OSError:
            data["report"] = ""
        found.append((mtime, data))
    found.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _mtime, row in found[:max(0, int(limit))]]


__all__ = [
    "BENCHMARKS", "Benchmark", "BenchmarkResult", "BenchmarkRun", "CheckResult",
    "RubricCheck", "SuiteResult", "benchmark_catalog", "benchmark_project_dir",
    "default_benchmarks_root", "generate_report", "get_benchmark",
    "holdout_source", "list_benchmarks", "project_dir_name", "recent_runs",
    "run_benchmark", "run_holdout", "run_model_tests", "run_rubric",
    "write_report",
]
