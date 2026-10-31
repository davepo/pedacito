"""Chunker tests: semantic units, oversized-unit splitting, line ranges.

No LLM and no index are needed here; these are pure source-to-chunk checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito.chunker import chunk_source
from pedacito.chunker_ts import chunk_file_ts, TS_AVAILABLE


def _py_function(name: str, body_lines: int) -> str:
    body = "\n".join(f"    step_{i} = {i}" for i in range(body_lines))
    return f"def {name}():\n{body}\n    return step_0\n"


def test_python_long_function_is_split_at_statement_boundaries():
    src = "import os\n\n" + _py_function("big", 400)
    chunks = chunk_source("big.py", src, max_unit_lines=50, max_unit_chars=10000)
    big = [c for c in chunks if c.qualname == "big" or c.qualname.startswith("big#")]
    assert len(big) > 1, "a 400-line function was left as one retrieval unit"
    assert big[0].qualname == "big"
    assert [c.qualname for c in big[1:]] == [f"big#{i}" for i in range(2, len(big) + 1)]
    # Every part is a real, ordered slice of the file, and the union covers
    # the whole function body without overlapping headers.
    assert big[0].start == 3
    assert big[-1].end == len(src.splitlines())
    for a, b in zip(big, big[1:]):
        assert a.end < b.start
    assert all("part " in c.signature for c in big)
    assert all(c.imports == ["os"] for c in big)


def test_python_short_function_stays_one_chunk():
    src = _py_function("small", 10)
    chunks = chunk_source("small.py", src, max_unit_lines=220, max_unit_chars=12000)
    assert [c.qualname for c in chunks] == ["small"]
    assert "part " not in chunks[0].signature


def test_python_char_budget_splits_long_lines():
    body = "\n".join(f'    x_{i} = "{"a" * 120}"' for i in range(30))
    src = f"def wide():\n{body}\n"
    chunks = chunk_source("wide.py", src, max_unit_lines=220, max_unit_chars=1000)
    assert len([c for c in chunks if c.qualname.startswith("wide")]) > 1


def test_python_single_giant_statement_falls_back_to_line_windows():
    literal = ",\n".join(f'    "k{i}": {i}' for i in range(300))
    src = "DATA = {\n" + literal + "\n}\n"
    chunks = chunk_source("data.py", src, max_unit_lines=20, max_unit_chars=1000)
    data = [c for c in chunks if c.qualname == "module"
            or c.qualname.startswith("module#")]
    assert len(data) > 1, "a giant dict literal stayed one unreadable unit"
    assert data[0].start == 1
    assert data[-1].end == len(src.splitlines())
    assert all(1 <= c.start <= c.end for c in data)


def test_giant_statement_windows_can_overlap(tmp_path):
    literal = ",\n".join(f'    "k{i}": {i}' for i in range(200))
    src = "DATA = {\n" + literal + "\n}\n"
    chunks = chunk_source("data.py", src, max_unit_lines=20,
                          max_unit_chars=1000, unit_overlap_lines=3)
    data = [c for c in chunks if c.qualname == "module"
            or c.qualname.startswith("module#")]
    assert len(data) > 1
    assert any(b.start <= a.end for a, b in zip(data, data[1:]))


def test_token_budget_can_split_before_line_or_char_limits():
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    src = f"def f():\n{body}\n"
    chunks = chunk_source("tok.py", src, max_unit_lines=1000,
                          max_unit_chars=100000, max_unit_tokens=50)
    assert len([c for c in chunks if c.qualname == "f"
                or c.qualname.startswith("f#")]) > 1


def test_python_long_preamble_is_split():
    src = "import os\n" + "\n".join(f"CONST_{i} = {i}" for i in range(250)) + \
        "\ndef f():\n    return CONST_0\n"
    chunks = chunk_source("constants.py", src, max_unit_lines=40, max_unit_chars=10000)
    mods = [c for c in chunks if c.kind == "module"]
    assert len(mods) > 1
    assert mods[0].qualname == "module"
    assert all(c.imports == ["os"] for c in mods)
    assert mods[-1].end < len(src.splitlines())


def test_section_aware_splitting_prefers_blank_line_boundaries():
    first = "\n".join(f"    a{i} = {i}" for i in range(40))
    second = "\n".join(f"    b{i} = {i}" for i in range(40))
    src = f"def f():\n{first}\n\n{second}\n    return a0\n"
    chunks = chunk_source("sections.py", src, max_unit_lines=60,
                          max_unit_chars=100000, max_unit_tokens=0)
    parts = [c for c in chunks if c.qualname == "f"
             or c.qualname.startswith("f#")]
    assert len(parts) > 1
    # The first part should end at the blank-line section boundary, not at
    # the hard 60-line edge in the middle of the second section.
    assert parts[0].end < 50


def test_python_semicolon_statements_never_produce_invalid_ranges():
    src = "def f():\n    a = 1; b = 2; c = 3\n    return a\n"
    chunks = chunk_source("semi.py", src, max_unit_lines=1, max_unit_chars=5)
    assert all(1 <= c.start <= c.end for c in chunks)
    assert [c.qualname for c in chunks][0] == "f"


def test_python_large_class_without_methods_is_split():
    src = "class C:\n" + "\n".join(f"    A_{i} = {i}" for i in range(300)) + "\n"
    chunks = chunk_source("c.py", src, max_unit_lines=40, max_unit_chars=10000)
    parts = [c for c in chunks if c.qualname == "C" or c.qualname.startswith("C#")]
    assert len(parts) > 1
    assert all("part " in c.signature for c in parts)


def test_python_large_class_header_with_methods_is_split():
    attrs = "\n".join(f"    A_{i} = {i}" for i in range(250))
    src = "class C:\n" + attrs + "\n\n    def m(self):\n        return 1\n"
    chunks = chunk_source("c.py", src, class_split_lines=2,
                          max_unit_lines=40, max_unit_chars=10000)
    headers = [c for c in chunks if c.kind == "classheader"]
    assert len(headers) > 1
    assert any(c.qualname == "C.m" for c in chunks)
    assert all("part " in c.signature for c in headers)


def test_duplicate_definition_qualnames_are_made_unique(tmp_path):
    """A property getter and setter share a name. The index cannot store two
    chunks under one key, so the later body gets a #2 suffix instead of
    silently replacing the first."""
    from pedacito import index as index_mod
    from pedacito.config import Config

    p = tmp_path / "obj.py"
    p.write_text(
        "class C:\n"
        "    @property\n"
        "    def value(self):\n"
        "        return self._v\n"
        "\n"
        "    @value.setter\n"
        "    def value(self, v):\n"
        "        self._v = v\n", encoding="utf-8")
    # Force the class to split into methods so the duplicate method names are
    # visible as separate chunks.
    cfg = Config()
    cfg.class_split_lines = 2
    chunks = index_mod._chunk_file("obj.py", p, cfg)
    quals = [c.qualname for c in chunks]
    assert "C.value" in quals
    assert "C.value#2" in quals
    assert len(quals) == len(set(quals))


def test_ts_long_function_is_split_when_tree_sitter_is_available():
    if not TS_AVAILABLE:
        return
    body = "\n".join(f"  const step{i} = {i};" for i in range(120))
    src = "export function big() {\n" + body + "\n  return step0;\n}\n"
    chunks = chunk_file_ts("big.ts", src, ".ts", class_split_lines=60,
                           max_unit_lines=20, max_unit_chars=10000)
    big = [c for c in chunks if c.qualname == "big" or c.qualname.startswith("big#")]
    assert len(big) > 1
    assert big[0].qualname == "big"
    assert big[-1].end == len(src.splitlines())
    assert all("part " in c.signature for c in big)


def test_ts_short_function_stays_whole():
    if not TS_AVAILABLE:
        return
    src = "export function small() {\n  return 1;\n}\n"
    chunks = chunk_file_ts("small.ts", src, ".ts")
    assert [c.qualname for c in chunks] == ["small"]
    assert "part " not in chunks[0].signature


def test_ts_long_arrow_component_is_split():
    if not TS_AVAILABLE:
        return
    body = "\n".join(f"  const step{i} = {i};" for i in range(80))
    src = "export const Widget = () => {\n" + body + "\n  return null;\n};\n"
    chunks = chunk_file_ts("widget.tsx", src, ".tsx", max_unit_lines=20,
                           max_unit_chars=10000)
    parts = [c for c in chunks
             if c.qualname == "Widget" or c.qualname.startswith("Widget#")]
    assert len(parts) > 1
    assert all("part " in c.signature for c in parts)


def test_ts_class_field_and_object_literal_methods_are_chunks():
    if not TS_AVAILABLE:
        return
    src = (
        "class A {\n"
        "  handle = (x) => {\n    return x;\n  };\n"
        "  value = 1;\n"
        "}\n"
        "const api = {\n"
        "  foo(a) {\n    return a;\n  },\n"
        "  bar: () => 1,\n"
        "};\n")
    chunks = chunk_file_ts("x.ts", src, ".ts", class_split_lines=2)
    quals = [c.qualname for c in chunks]
    assert "A.handle" in quals
    assert "api.foo" in quals and "api.bar" in quals
    assert "A.value" not in quals          # a data field is not a method


def test_ruby_long_method_is_split_when_tree_sitter_is_available():
    if not TS_AVAILABLE:
        return
    body = "\n".join(f"    x{i} = {i}" for i in range(60))
    src = "class C\n  def big\n" + body + "\n  end\nend\n"
    chunks = chunk_file_ts("big.rb", src, ".rb", class_split_lines=2,
                           max_unit_lines=20, max_unit_chars=10000)
    big = [c for c in chunks
           if c.qualname == "C.big" or c.qualname.startswith("C.big#")]
    assert len(big) > 1
    assert all("part " in c.signature for c in big)


def test_form_feed_does_not_shift_line_ranges():
    src = "import os\n\x0c\ndef f():\n    return 1\n\ndef g():\n    return 2\n"
    chunks = {c.qualname: c for c in chunk_source("x.py", src)}
    assert (chunks["g"].start, chunks["g"].end) == (6, 7)
    assert chunks["g"].source.startswith("def g")
