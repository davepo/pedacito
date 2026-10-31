"""F2 edit anchoring: `LINES a-b` / `SYMBOL name` FILE-header anchors, the
closest-window failure hint, and the agent-side SYMBOL resolution."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito.agent import Agent
from pedacito.chunker import chunk_source
from pedacito.config import Config
from pedacito.edits import apply_edits, parse_edits
from pedacito.index import Index

SRC = """\
def alpha():
    return 1


def beta():
    return 2


class Gamma:
    def method(self):
        return 3
"""


def _write(tmp_path: Path, text: str = SRC, name: str = "app.py") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _index(tmp_path: Path, files: dict[str, str]) -> Index:
    chunks = []
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
        chunks.extend(chunk_source(rel, src, 60))
    return Index(tmp_path, chunks)


# ------------------------------------------------------------------- parsing
def test_parse_lines_anchor():
    reply = ("--- FILE: app.py LINES 5-6\n"
             "<<<<<<< SEARCH\n=======\ndef beta():\n    return 22\n"
             ">>>>>>> REPLACE\n")
    edits = parse_edits(reply)
    assert len(edits) == 1
    assert edits[0].file == "app.py"
    assert edits[0].lines == (5, 6)
    assert edits[0].search == ""
    assert edits[0].replace == "def beta():\n    return 22"


def test_parse_lines_anchor_normalises_reversed_range():
    edits = parse_edits("--- FILE: a.py LINES 9-3\n"
                        "<<<<<<< SEARCH\n=======\nx\n>>>>>>> REPLACE\n")
    assert edits[0].lines == (3, 9)


def test_parse_symbol_anchor():
    edits = parse_edits("--- FILE: app.py SYMBOL Gamma.method\n"
                        "<<<<<<< SEARCH\n=======\n    def method(self):\n"
                        "        return 33\n>>>>>>> REPLACE\n")
    assert edits[0].symbol == "Gamma.method"
    assert edits[0].lines is None


def test_anchored_empty_replace_survives_parse():
    """An anchored delete (empty REPLACE) is a real edit, not a no-op and not
    a whole-file clear."""
    edits = parse_edits("--- FILE: app.py LINES 3-4\n"
                        "<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n")
    assert len(edits) == 1
    assert edits[0].replace == ""


# -------------------------------------------------------------------- apply
def test_apply_lines_anchor_replaces_exact_range(tmp_path):
    _write(tmp_path)
    reply = ("--- FILE: app.py LINES 5-6\n"
             "<<<<<<< SEARCH\n=======\ndef beta():\n    return 22\n"
             ">>>>>>> REPLACE\n")
    results = apply_edits(tmp_path, parse_edits(reply), backup=False)
    assert [r.ok for r in results] == [True]
    assert "return 22" in (tmp_path / "app.py").read_text()
    assert "return 1" in (tmp_path / "app.py").read_text()   # rest untouched


def test_apply_lines_anchor_empty_replace_deletes_range(tmp_path):
    _write(tmp_path)
    reply = ("--- FILE: app.py LINES 5-6\n"
             "<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n")
    results = apply_edits(tmp_path, parse_edits(reply), backup=False)
    assert [r.ok for r in results] == [True]
    text = (tmp_path / "app.py").read_text()
    assert "def beta" not in text
    assert "def alpha():" in text and "class Gamma:" in text


def test_apply_lines_out_of_bounds_fails(tmp_path):
    _write(tmp_path)
    reply = ("--- FILE: app.py LINES 900-950\n"
             "<<<<<<< SEARCH\n=======\nx = 1\n>>>>>>> REPLACE\n")
    results = apply_edits(tmp_path, parse_edits(reply), backup=False)
    assert not results[0].ok
    assert "out of bounds" in results[0].message


def test_apply_blank_lines_anchor_is_refused(tmp_path):
    _write(tmp_path)
    reply = ("--- FILE: app.py LINES 3-4\n"
             "<<<<<<< SEARCH\n=======\n\n\n>>>>>>> REPLACE\n")
    # Lines 3-4 are blank in SRC; a replacement of only newlines is refused
    # only when the NEEDLE is blank -- here the needle is blank, so refuse.
    results = apply_edits(tmp_path, parse_edits(reply), backup=False)
    assert not results[0].ok
    assert "blank lines" in results[0].message


def test_apply_unresolved_symbol_fails_with_guidance(tmp_path):
    _write(tmp_path)
    reply = ("--- FILE: app.py SYMBOL NoSuchThing\n"
             "<<<<<<< SEARCH\n=======\nx = 1\n>>>>>>> REPLACE\n")
    results = apply_edits(tmp_path, parse_edits(reply), backup=False)
    assert not results[0].ok
    assert "SYMBOL anchor" in results[0].message
    assert "LINES" in results[0].message


def test_failed_search_shows_closest_window_hint(tmp_path):
    _write(tmp_path)
    reply = ("--- FILE: app.py\n"
             "<<<<<<< SEARCH\ndef beta():\n    return 3\n=======\n"
             "def beta():\n    return 22\n>>>>>>> REPLACE\n")
    results = apply_edits(tmp_path, parse_edits(reply), backup=False,
                          cfg=Config())
    assert not results[0].ok
    msg = results[0].message
    assert "closest text in the file right now is at lines 5-6" in msg
    assert "5 | def beta():" in msg
    assert "LINES 5-6" in msg          # the hint offers the anchor form


# ------------------------------------------------------- symbol resolution
def _agent(tmp_path: Path, files: dict[str, str]) -> Agent:
    return Agent(_index(tmp_path, files), object(), Config())


def test_resolve_symbol_anchor_sets_lines(tmp_path):
    agent = _agent(tmp_path, {"app.py": SRC})
    edits = parse_edits("--- FILE: app.py SYMBOL beta\n"
                        "<<<<<<< SEARCH\n=======\ndef beta():\n    return 9\n"
                        ">>>>>>> REPLACE\n")
    failed, left = agent._resolve_symbol_anchors(edits)
    assert failed == []
    assert len(left) == 1
    assert left[0].lines == (5, 6)


def test_resolve_symbol_anchor_class_covers_whole_scope(tmp_path):
    agent = _agent(tmp_path, {"app.py": SRC})
    edits = parse_edits("--- FILE: app.py SYMBOL Gamma\n"
                        "<<<<<<< SEARCH\n=======\nclass Gamma:\n    pass\n"
                        ">>>>>>> REPLACE\n")
    failed, left = agent._resolve_symbol_anchors(edits)
    assert failed == []
    a, b = left[0].lines
    assert a == 9 and b >= 11          # the class header through its body


def test_resolve_symbol_anchor_missing_fails_whole_file(tmp_path):
    agent = _agent(tmp_path, {"app.py": SRC})
    edits = parse_edits(
        "--- FILE: app.py SYMBOL nope\n<<<<<<< SEARCH\n=======\nx\n"
        ">>>>>>> REPLACE\n\n"
        "--- FILE: app.py\n<<<<<<< SEARCH\ndef alpha():\n=======\n"
        "def alpha():  # kept out too\n>>>>>>> REPLACE\n")
    failed, left = agent._resolve_symbol_anchors(edits)
    assert left == []                  # the whole file drops out of the pass
    assert len(failed) == 1
    assert "no definition by that name" in failed[0].message
    assert "outline" in failed[0].message


def test_resolve_symbol_anchor_method_of_whole_class_needs_qualname(tmp_path):
    """A method name that only matches via the class's members list is not a
    replaceable definition of its own."""
    big = "class Big:\n" + "".join(
        f"    def m{i}(self):\n        return {i}\n" for i in range(3))
    agent = _agent(tmp_path, {"big.py": big})
    edits = parse_edits("--- FILE: big.py SYMBOL m1\n"
                        "<<<<<<< SEARCH\n=======\n    def m1(self):\n"
                        "        return 99\n>>>>>>> REPLACE\n")
    failed, left = agent._resolve_symbol_anchors(edits)
    if left:                           # small class indexed whole: resolved
        assert left[0].lines is not None
    else:
        assert "LINES" in failed[0].message


def test_failed_block_texts_echo_anchors():
    from pedacito.edits import Result
    agent = Agent(Index(Path("."), []), object(), Config())
    reply = ("--- FILE: a.py LINES 3-4\n<<<<<<< SEARCH\n=======\nx\n"
             ">>>>>>> REPLACE\n\n"
             "--- FILE: a.py SYMBOL foo\n<<<<<<< SEARCH\n=======\ny\n"
             ">>>>>>> REPLACE\n")
    out = agent._failed_block_texts(
        reply, [Result(False, "a.py", "boom")])
    assert "--- FILE: a.py LINES 3-4" in out
    assert "--- FILE: a.py SYMBOL foo" in out
