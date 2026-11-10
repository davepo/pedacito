"""Golden chunker tests: exact qualnames, kinds, line ranges, calls and imports
per language, plus the reference-file fixes (setext/RST headings, fenced code,
SQL comments and dollar quotes, shell function ranges, duplicate headings)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import index as index_mod
from pedacito.chunker import chunk_source
from pedacito.chunker_ts import TS_AVAILABLE, chunk_file_ts
from pedacito.config import Config
from pedacito.references import chunk_reference

needs_ts = pytest.mark.skipif(not TS_AVAILABLE, reason="tree-sitter unavailable")


def _ts(rel, src):
    ext = "." + rel.rsplit(".", 1)[1]
    return chunk_file_ts(rel, src, ext, 60, 220, 12000, 3000, 0)


def _rows(chunks):
    return [(c.qualname, c.kind, c.start, c.end) for c in chunks]


# ----------------------------------------------------------------- Python
def test_python_imports_include_local_nested_and_guarded():
    src = ("import os\n"
           "def f():\n"
           "    import json\n"
           "    from . import sibling\n"
           "    try:\n"
           "        import yaml\n"
           "    except ImportError:\n"
           "        yaml = None\n"
           "    return json\n")
    imps = chunk_source("pkg/g.py", src, 60)[0].imports
    assert set(imps) >= {"os", "json", "yaml", ".sibling"}


# ---------------------------------------------------------------------- Go
@needs_ts
def test_go_methods_are_qualified_by_receiver():
    src = ('package main\n\nimport (\n\t"fmt"\n\t"github.com/a/b/util"\n)\n\n'
           "type Server struct{}\n\n"
           "func (s *Server) Start() { fmt.Println(util.Do()); s.stop() }\n"
           "func (s *Server) stop() {}\nfunc main() { s := &Server{}; s.Start() }\n")
    ch = _ts("b.go", src)
    rows = {c.qualname: c for c in ch}
    assert set(rows) >= {"Server", "Server.Start", "Server.stop", "main"}
    assert rows["Server.Start"].kind == "method"
    assert rows["main"].kind == "function"
    assert (rows["Server.Start"].start, rows["Server.Start"].end) == (10, 10)
    assert set(rows["Server.Start"].calls) == {"Do", "Println", "stop",
                                               "Server.stop"}
    assert "github.com/a/b/util" in rows["main"].imports


# -------------------------------------------------------------------- Rust
RUST = ("use std::io;\nuse crate::util::helper;\n"
        "mod inner { pub fn deep() { super::top(); } }\n"
        "pub struct Foo;\n"
        "impl Foo { pub fn new() -> Foo { io::stdout(); Foo } pub fn bar(&self) { helper(); } }\n"
        "impl Drop for Foo { fn drop(&mut self) { } }\n"
        "fn top() { let f = Foo::new(); f.bar(); }\n")


@needs_ts
def test_rust_scoped_calls_nested_mod_and_impl_labels():
    ch = _ts("c.rs", RUST)
    by = {c.qualname: c for c in ch}
    # nested `mod` contents are chunked, qualified by the module
    assert "inner.deep" in by and by["inner.deep"].kind == "function"
    assert "top" in by["inner.deep"].calls
    # scoped_identifier callees (`io::stdout()`, `Foo::new()`) are captured
    assert "stdout" in by["impl Foo"].calls
    assert "new" in by["top"].calls and "bar" in by["top"].calls
    # struct, inherent impl and trait impl stay distinguishable
    assert {"Foo", "impl Foo", "impl Drop for Foo"} <= set(by)
    assert by["impl Foo"].members == ["new", "bar"]


@needs_ts
def test_rust_multiple_impls_get_unique_qualnames():
    src = ("pub struct A;\nimpl A { fn x(&self) {} }\nimpl A { fn y(&self) {} }\n")
    ch = index_mod._unique_qualnames(_ts("a.rs", src))
    names = [c.qualname for c in ch]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------- C#
@needs_ts
def test_csharp_block_namespace_is_chunked_and_calls_are_found():
    src = ("using System;\nnamespace Acme.App\n{\n    public class Runner\n    {\n"
           "        public void Go() { Console.WriteLine(\"x\"); Helper.Do(); this.Stop(); }\n"
           "        void Stop() {}\n    }\n}\n")
    ch = _ts("D.cs", src)
    by = {c.qualname: c for c in ch}
    assert "Runner" in by                       # not one `module` blob
    assert by["Runner"].kind == "class"
    assert (by["Runner"].start, by["Runner"].end) == (4, 8)
    assert {"WriteLine", "Do", "Stop"} <= set(by["Runner"].calls)
    assert by["Runner"].members == ["Go", "Stop"]


# ---------------------------------------------------------------- JS / TS
@needs_ts
def test_js_script_without_scopes_is_split_not_one_giant_chunk():
    # (230 statements: enough to exceed max_unit_lines. Far more live
    # tree-sitter nodes at once can crash py-tree-sitter 0.26 on CPython 3.14
    # -- an upstream bug independent of this code.)
    stmts = "\n".join(f"var v{i} = compute({i});" for i in range(230))
    ch = _ts("big.js", stmts)
    assert [(c.qualname, c.start, c.end) for c in ch] == [
        ("module", 1, 220), ("module#2", 221, 230)]
    assert all(c.nlines <= 220 for c in ch)


@needs_ts
def test_js_imports_see_require_and_dynamic_import():
    src = ('import a from "./a";\nconst fs = require("fs");\n'
           'export async function f() { return import("./lazy"); }\n')
    imps = _ts("m.js", src)[0].imports
    assert {"a", "fs", "lazy"} <= set(imps)


# ------------------------------------------------------------- references
def _ref(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return chunk_reference(name, p)


def test_markdown_setext_headings_and_preamble(tmp_path):
    ch = _ref(tmp_path, "d.md", "intro text\n\nTitle\n=====\nbody\n\nSub\n---\nmore\n")
    assert [(c.qualname, c.start, c.end) for c in ch] == [
        ("preamble", 1, 2), ("Title", 3, 6), ("Sub", 7, 9)]


def test_markdown_headings_inside_fenced_code_are_not_headings(tmp_path):
    text = "# Real\n\n```bash\n# just a comment\necho hi\n```\n\n## Next\n"
    names = [c.qualname for c in _ref(tmp_path, "d.md", text)]
    assert names == ["Real", "Next"]


def test_rst_headings_are_recognised(tmp_path):
    text = ("Project\n=======\n\nintro\n\nUsage\n-----\n\nrun it\n\n"
            "Deep\n~~~~\n\nx\n")
    ch = _ref(tmp_path, "d.rst", text)
    assert [c.qualname for c in ch] == ["Project", "Usage", "Deep"]
    assert [c.signature.split()[0] for c in ch] == ["#", "##", "###"]


def test_duplicate_headings_get_unique_keys(tmp_path):
    p = tmp_path / "d.md"
    p.write_text("# A\n## Usage\nx\n## Usage\ny\n", encoding="utf-8")
    cfg = Config()
    chunks = index_mod._chunk_file("d.md", p, cfg)
    keys = [c.key for c in chunks]
    assert len(keys) == len(set(keys))


def test_sql_ignores_semicolons_in_comments_quotes_and_dollar_bodies(tmp_path):
    sql = ("-- header; with a semicolon\n"
           "CREATE TABLE a (id int); -- trailing; comment\n"
           "/* block; comment */\n"
           "CREATE FUNCTION f() RETURNS int AS $$\nBEGIN\n  SELECT 1;\n  RETURN 2;\nEND;\n$$ LANGUAGE plpgsql;\n"
           "INSERT INTO a VALUES ('it''s; fine');\n")
    ch = _ref(tmp_path, "s.sql", sql)
    assert [c.qualname for c in ch] == ["create table a", "create function f",
                                         "INSERT INTO a VALUES ('it''s; fine');"]
    fn = ch[1]
    assert fn.start == 3 and fn.end == 9   # leading comment + the whole $$ body


def test_shell_function_ranges_stop_at_the_closing_brace(tmp_path):
    sh = ("#!/bin/sh\nset -e\n\nfoo() {\n  echo a\n}\n\necho between\n\nbar() {\n  echo b\n}\n")
    ch = _ref(tmp_path, "x.sh", sh)
    by = {c.qualname: (c.start, c.end) for c in ch}
    assert by["foo"] == (4, 6)
    assert by["bar"] == (10, 12)
    # the code between the functions is its own chunk, not swallowed by foo
    assert any(c.qualname.startswith("script") and c.start == 7 for c in ch) \
        or any(c.qualname.startswith("script_L") for c in ch)
