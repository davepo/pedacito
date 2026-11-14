"""Import resolution, the dependency graph, test links, BM25 search, ranked
map selection, the repo brief, and the index's incremental updates."""
from __future__ import annotations

import os
import shutil
import stat
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import brief, index as index_mod, rank, testlinks, tools
from pedacito.config import Config
from pedacito.imports import FileSet, raw_imports, resolve
from pedacito.search import tokenize


def _project(tmp_path: Path, files: dict[str, str], cfg=None):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return index_mod.build([str(tmp_path)], cfg or Config(), client=None,
                           summarise=False, root=tmp_path, verbose=False)


# ------------------------------------------------------------------ imports
def test_raw_imports_per_language():
    assert raw_imports("a.py", "import os\ndef f():\n    from . import b\n    import x.y\n") \
        == ["os", ".", ".b", "x.y"]
    assert raw_imports("a.ts", 'import a from "./a";\nconst b = require("../b");\n'
                       'export * from "./c";\nawait import("./d");\n') \
        == ["./a", "./c", "../b", "./d"]
    assert raw_imports("m.go", 'import (\n "fmt"\n u "github.com/x/y/util"\n)\n') \
        == ["fmt", "github.com/x/y/util"]
    assert raw_imports("l.rs", "use crate::a::{b, c::d};\nmod e;\n") \
        == ["crate::a::b", "crate::a::c::d", "self::e"]
    assert raw_imports("E.java", "import com.acme.Helper;\nimport static a.B.c;\n") \
        == ["com.acme.Helper", "a.B.c"]
    assert raw_imports("f.rb", 'require "json"\nrequire_relative "lib/u"\n') \
        == ["json", "./lib/u"]


def test_resolution_ties_imports_to_indexed_files():
    files = ["src/app.ts", "src/lib/util.ts", "src/lib/index.ts", "pkg/__init__.py",
             "pkg/mod.py", "pkg/sub/leaf.py", "store/store.go", "store/util.go",
             "src/main.rs", "src/util/mod.rs", "src/lib.rs", "com/acme/Helper.java"]
    fs = FileSet(files)
    assert resolve("src/app.ts", "./lib/util", fs) == ["src/lib/util.ts"]
    assert resolve("src/app.ts", "./lib", fs) == ["src/lib/index.ts"]
    assert resolve("src/app.ts", "react", fs) == []
    assert resolve("pkg/mod.py", ".sub.leaf", fs) == ["pkg/sub/leaf.py"]
    assert resolve("pkg/mod.py", "pkg.sub.leaf", fs) == ["pkg/sub/leaf.py"]
    assert resolve("pkg/mod.py", "os", fs) == []
    assert sorted(resolve("main.go", "example.com/x/store", fs)) \
        == ["store/store.go", "store/util.go"]
    assert resolve("src/main.rs", "self::util", fs) == ["src/util/mod.rs"]
    assert resolve("src/main.rs", "crate::util", fs) == ["src/util/mod.rs"]
    assert resolve("x/Y.java", "com.acme.Helper", fs) == ["com/acme/Helper.java"]


# ------------------------------------------------------------------- graph
PKG = {
    "pkg/__init__.py": "",
    "pkg/core.py": "def core():\n    return 1\n",
    "pkg/a.py": "from pkg import core\n\ndef a():\n    return core.core()\n",
    "pkg/b.py": "from pkg.core import core\n\ndef b():\n    return core()\n",
    "pkg/leaf.py": "from pkg import a\n\ndef leaf():\n    return a.a()\n",
    "tests/test_core.py": "from pkg.core import core\n\ndef test_core():\n    assert core() == 1\n",
}


def test_graph_importers_pagerank_and_impact(tmp_path):
    idx = _project(tmp_path, PKG)
    g = idx.graph
    assert g.importers_of("pkg/core.py") == ["pkg/a.py", "pkg/b.py", "tests/test_core.py"]
    assert idx.importers_of("pkg/core.py") == g.importers_of("pkg/core.py")
    pr = g.pagerank()
    assert max(pr, key=pr.get) == "pkg/core.py"          # the most imported
    assert pr["pkg/core.py"] > pr["pkg/leaf.py"]
    reach = g.impact("pkg/core.py", depth=2)
    assert reach["pkg/a.py"] == 1 and reach["pkg/leaf.py"] == 2


def test_graph_follows_index_mutations(tmp_path):
    idx = _project(tmp_path, PKG)
    assert "pkg/new.py" not in idx.graph.importers_of("pkg/core.py")
    (tmp_path / "pkg/new.py").write_text("from pkg.core import core\n", encoding="utf-8")
    index_mod.reindex_file(idx, "pkg/new.py", Config(), save=False)
    assert "pkg/new.py" in idx.graph.importers_of("pkg/core.py")
    idx.forget("pkg/new.py")
    assert "pkg/new.py" not in idx.graph.importers_of("pkg/core.py")


# -------------------------------------------------------------- test links
@pytest.mark.parametrize("rel,expect", [
    ("tests/test_foo.py", (True, "foo", "py")),
    ("foo_test.py", (True, "foo", "py")),
    ("web/foo.test.ts", (True, "foo", "js")),
    ("web/foo.spec.jsx", (True, "foo", "js")),
    ("web/__tests__/foo.ts", (True, "foo", "js")),
    ("store/store_test.go", (True, "store", "go")),
    ("tests/foo.rs", (True, "foo", "rs")),
    ("src/test/java/FooTest.java", (True, "foo", "java")),
    ("spec/foo_spec.rb", (True, "foo", "rb")),
    ("Foo.cs", (False, "foo", "cs")),
    ("src/foo.py", (False, "foo", "py")),
    ("src/testimony.py", (False, "testimony", "py")),
    ("README.md", (False, "", "")),
])
def test_test_conventions_across_languages(rel, expect):
    assert testlinks.classify(rel) == expect


def test_cards_link_tests_for_every_language(tmp_path):
    from pedacito.chunker_ts import TS_AVAILABLE
    if not TS_AVAILABLE:
        pytest.skip("tree-sitter unavailable (JS/Go chunking falls back to lines)")
    idx = _project(tmp_path, {
        "web/app.js": "function a() { return 1; }\n",
        "web/app.test.js": "test('a', () => a());\n",
        "store/store.go": "package store\nfunc Get() {}\n",
        "store/store_test.go": "package store\nfunc TestGet() {}\n",
    })
    assert "tested by: web/app.test.js" in idx.render_cards(["web/app.js"])
    assert "tests: web/app.js" in idx.render_cards(["web/app.test.js"])
    assert "tested by: store/store_test.go" in idx.render_cards(["store/store.go"])


def test_a_test_with_two_candidate_modules_gets_no_link(tmp_path):
    idx = _project(tmp_path, {"a/foo.py": "x = 1\n", "b/foo.py": "y = 2\n",
                              "tests/test_foo.py": "z = 3\n"})
    assert "tests:" not in idx.render_cards(["tests/test_foo.py"])


# ------------------------------------------------------------------- BM25
def test_tokenize_splits_identifiers_and_stems():
    assert "retri" not in tokenize("retries")
    assert tokenize("retries") == tokenize("retry")
    toks = tokenize("retryWithBackoff snake_case_name")
    assert {"retry", "backoff", "snake", "case", "name", "retrywithbackoff"} <= set(toks)


def test_bm25_ranks_names_above_prose_and_updates_incrementally(tmp_path):
    idx = _project(tmp_path, {
        "a.py": ('def retry_with_backoff(fn):\n    """Call fn again."""\n    return fn()\n\n\n'
                 'def other(fn):\n    """Sometimes we retry here."""\n    return fn()\n'),
        "b.py": "def unrelated():\n    return 1\n",
    })
    bm = idx.bm25
    hits = bm.query("retry backoff", limit=5)
    assert hits[0][1].qualname == "retry_with_backoff"
    assert all(c.qualname != "unrelated" for _s, c in hits)
    bm.remove_file("a.py")
    assert bm.query("retry backoff") == []
    # the Index keeps its BM25 in step with mutations
    (tmp_path / "a.py").write_text("def new_retry():\n    return 1\n", encoding="utf-8")
    index_mod.reindex_file(idx, "a.py", Config(), save=False)
    assert idx.bm25.query("retry")[0][1].qualname == "new_retry"


def test_find_handles_multi_word_concepts_after_substring_hits(tmp_path):
    idx = _project(tmp_path, {
        "net.py": "def retry_with_delay(fn):\n    return fn()\n\n\n"
                  "def fetch_page(url):\n    return url\n",
    })
    out = tools.tool_find(idx, "retry backoff delay")
    assert "retry_with_delay" in out and "matches:" in out
    assert "retry_with_delay" in tools.tool_find(idx, "retry")
    assert tools.tool_find(idx, "zzzzqq").startswith("No definition matches")


def test_grep_miss_suggests_definitions_by_bm25(tmp_path):
    idx = _project(tmp_path, {"net.py": "def retry_with_delay(fn):\n    return fn()\n"})
    out = tools.tool_grep(idx, "retry delay", cfg=Config())
    assert "No matches" in out and "retry_with_delay" in out


# ----------------------------------------------------------------- grep: rg
def _fake_rg(bin_dir: Path) -> Path:
    exe = bin_dir / "rg"
    exe.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import re, sys
        a = sys.argv[1:]
        pat = a[a.index("-e") + 1]
        files = a[a.index("--") + 1:]
        for f in files:
            try:
                if re.search(pat, open(f, encoding="utf-8").read(), re.I):
                    print(f)
            except OSError:
                pass
        """), encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_ripgrep_only_prefilters_and_output_is_unchanged(tmp_path, monkeypatch):
    files = {f"m{i}.py": f"def f{i}():\n    return {i}\n" for i in range(60)}
    files["m7.py"] = "def f7():\n    return NEEDLE_TOKEN\n"
    idx = _project(tmp_path, files)
    cfg = Config()
    without = tools.tool_grep(idx, "NEEDLE_TOKEN", cfg=cfg)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_rg(bin_dir)
    # Windows `shutil.which` only looks at PATHEXT extensions, so it cannot
    # find a shebang script named plain `rg`; skip rather than testing the
    # system ripgrep (which may not exist on the runner).
    if shutil.which("rg", path=str(bin_dir)) is None:
        pytest.skip("the fake rg script needs a POSIX shebang")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    calls = []
    real_scan = tools._scan_grep

    def spy(index, scan_files, *a, **k):
        calls.append(list(scan_files))
        return real_scan(index, scan_files, *a, **k)

    monkeypatch.setattr(tools, "_scan_grep", spy)
    with_rg = tools.tool_grep(idx, "NEEDLE_TOKEN", cfg=cfg)
    assert with_rg == without
    assert "m7.py:2: [f7]" in with_rg
    assert calls[-1] == ["m7.py"]                   # rg narrowed 60 files to 1
    cfg.use_ripgrep = False
    tools.tool_grep(idx, "NEEDLE_TOKEN", cfg=cfg)
    assert len(calls[-1]) == 60                     # the Python scan covers all


def test_literal_grep_fast_path_matches_regex_semantics(tmp_path):
    idx = _project(tmp_path, {"a.py": "x = foo.bar(1)\ny = fooXbar(2)\nZ = Foo.Bar\n"})
    out = tools.tool_grep(idx, "foo.bar", cfg=Config())     # '.' is regex: both
    assert "a.py:1:" in out and "a.py:2:" in out
    out = tools.tool_grep(idx, "FOO.BAR(", cfg=Config())    # invalid regex -> literal, ci retry
    assert "retried case-insensitively" in out and "a.py:1:" in out


# ------------------------------------------------------------------ ranking
BIG = {f"mods/m{i}.py": f"def unused_{i}():\n    return {i}\n" for i in range(40)}
BIG.update({
    "core/engine.py": "from core import util\n\ndef run_engine():\n    return util.helper()\n",
    "core/util.py": "def helper():\n    return 'leech threshold'\n",
    "core/sched.py": ('"""Scheduler."""\n\ndef is_leech(lapses):\n    return lapses >= 8\n'),
    "tests/test_sched.py": "from core.sched import is_leech\n\ndef test_x():\n    assert is_leech(9)\n",
})


def test_rank_files_prefers_named_matching_and_central_files(tmp_path):
    idx = _project(tmp_path, BIG)
    ranked = rank.rank_files(idx, "make the leech threshold in core/sched.py configurable",
                             Config(), use_git=False)
    top = [r.file for r in ranked[:3]]
    assert top[0] == "core/sched.py"
    assert ranked[0].reasons[0] == "named in task"
    # the test that covers it outranks unrelated leaves
    assert ranked.index(next(r for r in ranked if r.file == "tests/test_sched.py")) < 10
    # with no task text, widely imported modules lead
    plain = rank.rank_files(idx, "", Config(), use_git=False)
    assert plain[0].file == "core/util.py"          # imported by engine.py


def test_ranked_tier_lists_likely_files_over_a_directory_skeleton(tmp_path):
    idx = _project(tmp_path, BIG)
    cfg = Config()
    cfg.flat_map_token_budget = 450
    text, tier = idx.map_for_prompt(cfg, task="fix is_leech in core/sched.py")
    assert tier == "ranked"
    assert text.startswith("LIKELY RELEVANT FILES")
    assert "### core/sched.py" in text
    assert "why listed: named in task" in text
    assert "DIRECTORY SKELETON" in text and "mods/" in text
    assert text.index("### core/sched.py") < text.index("DIRECTORY SKELETON")
    # cached per (generation, task, budget): same object back
    assert idx.map_for_prompt(cfg, task="fix is_leech in core/sched.py") == (text, tier)
    # a different task re-ranks
    other, _ = idx.map_for_prompt(cfg, task="improve run_engine in core/engine.py")
    assert other.index("### core/engine.py") < other.index("DIRECTORY SKELETON")
    # a project that fits stays alphabetical and stable across tasks
    small = _project(tmp_path / "small", {"b.py": "def b():\n    return 1\n",
                                           "a.py": "def a():\n    return 2\n"})
    s1, t1 = small.map_for_prompt(Config(), task="anything")
    s2, _ = small.map_for_prompt(Config(), task="something else")
    assert t1 == "flat" and s1 == s2 and s1.index("### a.py") < s1.index("### b.py")


def test_cards_show_blast_radius(tmp_path):
    idx = _project(tmp_path, PKG)
    card = idx.render_cards(["pkg/core.py"])
    assert "imported by: 3 file(s)" in card
    orphan = _project(tmp_path / "o", {**PKG, "pkg/orphan.py": "def lonely():\n    return 0\n"})
    assert "no importers or callers found" in orphan.render_cards(["pkg/orphan.py"])
    assert "no importers" not in orphan.render_cards(["pkg/core.py"])


def test_repo_brief_reports_stack_entry_points_and_commands(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\ndependencies = ["requests>=2"]\n'
        '[project.scripts]\ndemo = "demo.cli:main"\n', encoding="utf-8")
    idx = _project(tmp_path, {"demo/__init__.py": "", "demo/cli.py": "def main():\n    return 1\n",
                              "demo/core.py": "def f():\n    return 1\n",
                              "tests/test_core.py": "from demo.core import f\n"})
    text = brief.build_brief(idx, Config(), max_tokens=400)
    assert text.startswith("REPO BRIEF")
    assert "Python" in text and "demo" in text
    assert "entry points:" in text and "demo.cli:main" in text
    assert "python -m pytest -q" in text or "pytest -q" in text
    assert "layout:" in text


# ----------------------------------------------------------- index mechanics
def test_load_is_lazy_and_hydrates_source_on_demand(tmp_path):
    idx = _project(tmp_path, PKG)
    idx.save()
    loaded = index_mod.Index.load(tmp_path)
    c = loaded.find_symbol("core")[0]
    assert c.source == ""                       # nothing read at load time
    assert "return 1" in loaded.hydrate(c).source
    assert loaded.file_imports == idx.file_imports


def test_index_file_is_versioned_atomic_and_tolerates_bad_data(tmp_path, capsys):
    idx = _project(tmp_path, PKG)
    p = idx.save()
    assert not (p.parent / "index.json.tmp").exists()
    import json
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["version"] == index_mod.INDEX_VERSION
    data["version"] = 999
    p.write_text(json.dumps(data), encoding="utf-8")
    assert index_mod.Index.load(tmp_path) is None
    p.write_text("{not json", encoding="utf-8")
    assert index_mod.Index.load(tmp_path) is None
    data["version"] = 2
    data["chunks"][0]["field_from_the_future"] = 1
    p.write_text(json.dumps(data), encoding="utf-8")
    assert index_mod.Index.load(tmp_path) is not None     # unknown fields ignored


def test_reindex_keeps_old_summary_until_refreshed(tmp_path):
    cfg = Config()
    cfg.min_lines_to_summarise = 1
    idx = _project(tmp_path, {"a.py": "def work():\n    x = 1\n    return x\n"}, cfg)
    c = idx.find_symbol("work")[0]
    c.summary = "Does the work."
    (tmp_path / "a.py").write_text("def work():\n    x = 2\n    return x + 1\n", encoding="utf-8")
    index_mod.reindex_file(idx, "a.py", cfg, save=False)
    new = idx.find_symbol("work")[0]
    assert new.summary == "Does the work."              # not blanked
    assert new.key in idx.stale_summaries

    class Client:
        def chat(self, messages, max_tokens, **kw):
            return "Does the work, differently."

    n = index_mod.refresh_summaries(idx, Client(), cfg, limit=5, verbose=False)
    assert n == 1
    assert idx.find_symbol("work")[0].summary == "Does the work, differently."
    assert not idx.stale_summaries


def test_parallel_summaries_are_bounded_and_all_land(tmp_path):
    import threading
    cfg = Config()
    cfg.summary_workers = 3
    cfg.min_lines_to_summarise = 1
    files = {f"m{i}.py": f"def fn{i}():\n    a = {i}\n    return a\n" for i in range(9)}
    live = {"now": 0, "max": 0}
    lock = threading.Lock()

    class Client:
        def warmup(self):
            pass

        def chat(self, messages, max_tokens, **kw):
            with lock:
                live["now"] += 1
                live["max"] = max(live["max"], live["now"])
            import time
            time.sleep(0.02)
            with lock:
                live["now"] -= 1
            return "Does something."

    for rel, text in files.items():
        (tmp_path / rel).write_text(text, encoding="utf-8")
    idx = index_mod.build([str(tmp_path)], cfg, client=Client(), root=tmp_path,
                          summarise=True, verbose=False)
    assert all(c.summary for c in idx.chunks if c.kind == "function")
    assert 1 < live["max"] <= 3
