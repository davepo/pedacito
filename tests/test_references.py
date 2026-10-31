"""Structural chunkers for YAML, SQL, shell, and the generic tree-sitter
language families (Go, Rust, Java, C, C++, C#)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pedacito.chunker_ts import TS_AVAILABLE, chunk_file_ts
from pedacito.references import chunk_reference


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_yaml_top_level_keys_become_sections(tmp_path):
    p = _write(tmp_path, "config.yaml", (
        "# header\n"
        "name: demo\n"
        "server:\n"
        "  host: 127.0.0.1\n"
        "  port: 8080\n"
        "logging:\n"
        "  level: info\n"))
    chunks = chunk_reference("config.yaml", p)
    quals = [c.qualname for c in chunks]
    assert "name" in quals and "server" in quals and "logging" in quals
    server = next(c for c in chunks if c.qualname == "server")
    assert server.start == 3 and server.end == 5


def test_sql_statements_become_sections(tmp_path):
    p = _write(tmp_path, "schema.sql",
               "CREATE TABLE users (id INT);\n"
               "SELECT * FROM users;\n")
    chunks = chunk_reference("schema.sql", p)
    joined = " ".join(c.qualname for c in chunks)
    assert "create table users" in joined
    assert any(c.start == 2 for c in chunks)


def test_shell_functions_become_sections(tmp_path):
    p = _write(tmp_path, "run.sh",
               "#!/bin/sh\n"
               "greet() {\n  echo hello\n}\n"
               "main() {\n  greet\n}\n")
    chunks = chunk_reference("run.sh", p)
    quals = [c.qualname for c in chunks]
    assert "greet" in quals and "main" in quals
    assert "script" in quals


@pytest.mark.skipif(not TS_AVAILABLE, reason="tree-sitter unavailable")
@pytest.mark.parametrize("ext,src,expected", [
    (".go", "package main\nfunc Add(a int, b int) int {\n  return a+b\n}\n",
     "Add"),
    (".rs", "struct P { x: i32 }\nimpl P {\n  fn get(&self) -> i32 {\n"
            "    self.x\n  }\n}\n", "P.get"),
    (".java", "class A {\n  int add(int a, int b) {\n    return a+b;\n  }\n}\n",
     "A.add"),
    (".c", "#include <stdio.h>\nint add(int a, int b) {\n  return a+b;\n}\n",
     "add"),
    (".cpp", "class A {\npublic:\n  int add(int a, int b) {\n"
             "    return a+b;\n  }\n};\n", "A.add"),
    (".cs", "class A {\n  int Add(int a, int b) {\n    return a+b;\n  }\n}\n",
     "A.Add"),
])
def test_generic_tree_sitter_languages(ext, src, expected, tmp_path):
    chunks = chunk_file_ts("x" + ext, src, ext, class_split_lines=2)
    assert expected in [c.qualname for c in chunks], \
        [(c.qualname, c.kind) for c in chunks]
