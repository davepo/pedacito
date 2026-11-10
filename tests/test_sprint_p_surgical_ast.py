"""Sprint P: zero-anchor AST surgical splicing and the deterministic
multi-file symbol rename engine.

Two techniques, both without a SEARCH block and both without model tokens:

1. Symbol splicing (`<<<<<<< SYMBOL file :: Class.method` and the
   `{"edit_action": "replace_symbol" | "insert_after_symbol"}` payloads): the
   definition's span comes from the real AST (or tree-sitter), the body's
   indentation is re-normalised to the definition's own column, decorators are
   carried over, and the result passes the same syntax gate as every edit.
2. `rename_symbol_across_project`: every EXACT identifier token moves, through
   Python's `tokenize` or tree-sitter identifier nodes, so strings and comments
   that merely contain the old name survive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_context import Scripted, _agent  # noqa: E402

from pedacito import cli, edits, refactor, tools  # noqa: E402
from pedacito.chunker import locate_symbol_span  # noqa: E402
from pedacito.chunker_ts import TS_AVAILABLE  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.index import Index  # noqa: E402

APP = '''\
import json


def load(path):
    return open(path).read()


class Report:
    @staticmethod
    def summarise(rows):
        total = 0
        for r in rows:
            total += r
        return total

    def title(self):
        return "report"


def save(path, data):
    return json.dump(data, open(path, "w"))
'''


def _line_of(text: str, needle: str) -> int:
    """1-based line number of `needle` in `text` (for span assertions)."""
    for i, ln in enumerate(text.splitlines(), 1):
        if needle in ln:
            return i
    raise AssertionError(f"{needle!r} not found")


# ------------------------------------------------------------ locate_symbol_span

def test_locate_python_toplevel_span(tmp_path):
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    got = locate_symbol_span(p, "load")
    assert got == (_line_of(APP, "def load(path):"),
                   _line_of(APP, "return open(path).read()"), "")
    assert locate_symbol_span(p, "save") == (
        _line_of(APP, "def save(path, data):"), len(APP.splitlines()), "")


def test_locate_python_method_includes_decorators_and_indent(tmp_path):
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    start, end, indent = locate_symbol_span(p, "Report.summarise")
    # the decorator line belongs to the definition: replacing the symbol must
    # not orphan it
    assert start == _line_of(APP, "@staticmethod")
    assert end == _line_of(APP, "return total")
    assert indent == "    "                       # the method's own column


def test_locate_python_missing_and_ambiguous_return_none(tmp_path):
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    assert locate_symbol_span(p, "no_such_function") is None
    assert locate_symbol_span(p, "Report.nope") is None
    assert locate_symbol_span(p, "") is None
    dup = tmp_path / "dup.py"
    src = "class A:\n    def m(self):\n        return 1\n\n\n" \
          "class B:\n    def m(self):\n        return 2\n"
    dup.write_text(src, encoding="utf-8")
    # bare `m` matches two definitions: ambiguous, so refuse to guess
    assert locate_symbol_span(dup, "m") is None
    assert locate_symbol_span(dup, "B.m") == (_line_of(src, "class B:") + 1,
                                              len(src.splitlines()), "    ")


def test_locate_python_broken_file_returns_none(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("def oops(:\n    return\n", encoding="utf-8")
    assert locate_symbol_span(p, "oops") is None


@pytest.mark.skipif(not TS_AVAILABLE, reason="tree-sitter not installed")
def test_locate_treesitter_languages(tmp_path):
    js = tmp_path / "a.js"
    js.write_text("class Report {\n  summarise(rows) {\n    let t = 0;\n"
                  "    return t;\n  }\n}\n\nfunction helper(x) { return x; }\n",
                  encoding="utf-8")
    assert locate_symbol_span(js, "helper") == (8, 8, "")
    assert locate_symbol_span(js, "Report.summarise") == (2, 5, "  ")
    assert locate_symbol_span(js, "Report.missing") is None

    go = tmp_path / "m.go"
    go.write_text("package main\n\ntype Calc struct{}\n\n"
                  "func (c Calc) Total(xs []int) int {\n\treturn len(xs)\n}\n",
                  encoding="utf-8")
    assert locate_symbol_span(go, "Calc.Total") == (5, 7, "")

    rs = tmp_path / "lib.rs"
    rs.write_text("pub struct Calc;\n\nimpl Calc {\n"
                  "    pub fn total(&self) -> i32 {\n        1\n    }\n}\n",
                  encoding="utf-8")
    assert locate_symbol_span(rs, "Calc.total") == (4, 6, "    ")


# ------------------------------------------------------------- symbol splicing

def _splice_reply(symbol: str, body: str, header: str = "--- FILE: app.py",
                  kind: str = "SYMBOL") -> str:
    return (f"{header}\n"
            f"<<<<<<< {kind} app.py :: {symbol}\n"
            f"{body}\n"
            f">>>>>>> END_SYMBOL\n")


def test_symbol_block_parses_without_a_search_needle():
    parsed = edits.parse_edits(_splice_reply("Report.summarise",
                                             "def summarise(rows):\n    x = 1"))
    assert len(parsed) == 1
    e = parsed[0]
    assert e.file == "app.py"
    assert e.symbol == "Report.summarise"
    assert e.search == "" and e.insert_after == ""
    assert edits._is_anchored(e)


def test_symbol_block_replaces_method_and_reindents_and_keeps_decorators(tmp_path):
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    # the body arrives at column 0 (the model's habitual shape) with no
    # decorator: the engine aligns it to the method's column and carries the
    # @staticmethod over.
    reply = _splice_reply("Report.summarise",
                          "def summarise(rows):\n    return sum(rows)")
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert len(res) == 1 and res[0].ok, res[0].message
    after = p.read_text(encoding="utf-8")
    assert "@staticmethod\n    def summarise(rows):\n        return sum(rows)" \
        in after
    assert "total = 0" not in after and "for r in rows" not in after
    # untouched neighbours survive
    assert "def load(path):" in after and "def title(self):" in after


def test_symbol_block_explicit_decorator_wins_over_carryover(tmp_path):
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    reply = _splice_reply("Report.summarise",
                          "@classmethod\n"
                          "def summarise(cls, rows):\n    return len(rows)")
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert res[0].ok, res[0].message
    after = p.read_text(encoding="utf-8")
    assert "@classmethod" in after and "@staticmethod" not in after
    assert "    def summarise(cls, rows):\n        return len(rows)" in after


def test_symbol_block_empty_replace_deletes_definition(tmp_path):
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    reply = _splice_reply("Report.title", "")
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert res[0].ok, res[0].message
    after = p.read_text(encoding="utf-8")
    assert "def title(self)" not in after and "summarise" in after


def test_symbol_block_syntax_gate_refuses_and_writes_nothing(tmp_path):
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    reply = _splice_reply("Report.summarise",
                          "def summarise(rows):\n    return sum(rows")   # unclosed
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert not res[0].ok
    assert "syntax error" in res[0].message and "not written" in res[0].message
    assert p.read_text(encoding="utf-8") == APP       # all-or-nothing


def test_symbol_block_unknown_symbol_says_what_to_do(tmp_path):
    (tmp_path / "app.py").write_text(APP, encoding="utf-8")
    reply = _splice_reply("Report.nonexistent", "def x():\n    return 1")
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert not res[0].ok
    # the failure contract other tests rely on: name the anchor and the escape
    assert "SYMBOL anchor" in res[0].message and "LINES" in res[0].message


def test_after_symbol_block_inserts_a_sibling(tmp_path):
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    reply = ("--- FILE: app.py\n"
             "<<<<<<< AFTER_SYMBOL app.py :: load\n"
             "def cached(path):\n    return load(path)\n"
             ">>>>>>> END_SYMBOL\n")
    parsed = edits.parse_edits(reply)
    assert len(parsed) == 1 and parsed[0].insert_after == "load"
    res = edits.apply_edits(tmp_path, parsed, backup=False)
    assert res[0].ok, res[0].message
    after = p.read_text(encoding="utf-8")
    lines = after.splitlines()
    assert _line_of(after, "def cached(path):") > _line_of(after, "return open(path).read()")
    assert _line_of(after, "def cached(path):") < _line_of(after, "class Report:")
    assert "class Report:" in after and lines.count("def load(path):") == 1


def test_after_symbol_indent_is_normalised_to_the_anchor(tmp_path):
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    reply = ("--- FILE: app.py\n"
             "<<<<<<< INSERT_AFTER_SYMBOL app.py :: Report.title\n"
             "def subtitle(self):\n    return \"sub\"\n"
             ">>>>>>> END_SYMBOL\n")
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert res[0].ok, res[0].message
    after = p.read_text(encoding="utf-8")
    assert "    def subtitle(self):\n        return \"sub\"" in after
    assert _line_of(after, "def subtitle") > _line_of(after, "def title")


def test_unterminated_symbol_block_is_reported_malformed():
    text = _splice_reply("load", "def load(p):\n    return 1").replace(
        ">>>>>>> END_SYMBOL\n", "")
    bad = edits.malformed_blocks(text, len(edits.parse_edits(text)))
    assert bad == [("app.py", bad[0][1])]
    assert "END_SYMBOL" in bad[0][1]


def test_well_formed_symbol_block_is_not_reported_malformed():
    """A splice block under its own `--- FILE:` header is ONE block: counting
    the header as a lost SEARCH block would nag the model about a good reply."""
    text = _splice_reply("load", "def load(p):\n    return 1")
    assert edits.malformed_blocks(text, len(edits.parse_edits(text))) == []


def test_classic_search_replace_blocks_still_work_unchanged(tmp_path):
    """Backward compatibility: the plain SEARCH/REPLACE grammar is untouched by
    the SYMBOL parser, and mixed in one reply the blocks apply in reply order."""
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    reply = ("--- FILE: app.py\n"
             "<<<<<<< SEARCH\n"
             "def load(path):\n    return open(path).read()\n"
             "=======\n"
             "def load(path):\n    return Path(path).read_text()\n"
             ">>>>>>> REPLACE\n"
             "--- FILE: app.py\n"
             "<<<<<<< SYMBOL app.py :: save\n"
             "def save(path, data):\n    Path(path).write_text(str(data))\n"
             ">>>>>>> END_SYMBOL\n")
    parsed = edits.parse_edits(reply)
    assert len(parsed) == 2
    res = edits.apply_edits(tmp_path, parsed, backup=False)
    assert res[0].ok, res[0].message
    after = p.read_text(encoding="utf-8")
    assert "return Path(path).read_text()" in after        # SEARCH block
    assert "Path(path).write_text(str(data))" in after     # SYMBOL splice
    assert "json.dump" not in after


# ------------------------------------------------------------ JSON edit actions

def test_replace_symbol_json_payload(tmp_path):
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    payload = {"edit_action": "replace_symbol", "file": "app.py",
               "symbol": "load",
               "replacement": "def load(path):\n    return json.loads(path)"}
    reply = "here you go:\n```json\n" + json.dumps(payload) + "\n```\n"
    found = edits.parse_edit_actions(reply)
    assert len(found) == 1 and found[0][1].symbol == "load"
    res = edits.apply_edits(tmp_path, edits.parse_edits(reply), backup=False)
    assert res[0].ok, res[0].message
    assert "return json.loads(path)" in p.read_text(encoding="utf-8")


def test_insert_after_symbol_json_payload(tmp_path):
    p = tmp_path / "app.py"
    p.write_text(APP, encoding="utf-8")
    payload = {"edit_action": "insert_after_symbol", "file": "app.py",
               "symbol": "load",
               "replacement": "def load_json(path):\n    return json.loads(load(path))"}
    res = edits.apply_edits(tmp_path, edits.parse_edits(json.dumps(payload)),
                            backup=False)
    assert res[0].ok, res[0].message
    after = p.read_text(encoding="utf-8")
    assert _line_of(after, "def load_json") > _line_of(after, "def load(path)")
    assert after.count("def load(path):") == 1


def test_unknown_or_broken_json_payload_is_ignored():
    for text in ['{"edit_action": "shrink_file", "file": "a.py"}',
                 '{"edit_action": "replace_symbol", "file": "a.py"}',
                 '{"edit_action": "replace_symbol"',
                 'unrelated {"note": "edit_action"} prose']:
        assert edits.parse_edit_actions(text) == []


# -------------------------------------------------------------- rename engine

PY_PKG = {
    "pkg/__init__.py": "",
    "pkg/core.py": "def apply_discount(price):\n    return price * 0.9\n",
    "pkg/shop.py": ("from pkg.core import apply_discount\n\n\n"
                    "def checkout(p):\n    # apply_discount here\n"
                    "    note = 'apply_discount'\n    return apply_discount(p)\n"),
    "pkg/util.py": ("def cents(apply_discount):\n"
                    "    return apply_discount * 100\n"),
}


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_rename_across_files_leaves_strings_and_comments_alone(tmp_path):
    root = _tree(tmp_path, PY_PKG)
    res = refactor.rename_symbol_across_project(root, "apply_discount",
                                                "markdown_price")
    assert res["ok"] and not res["dry_run"]
    assert sorted(res["files_modified"]) == ["pkg/core.py", "pkg/shop.py",
                                             "pkg/util.py"]
    assert res["total_replacements"] == 5       # def, import, call, param, body
    assert res["per_file"] == {"pkg/core.py": 1, "pkg/shop.py": 2,
                               "pkg/util.py": 2}
    core = (root / "pkg/core.py").read_text(encoding="utf-8")
    shop = (root / "pkg/shop.py").read_text(encoding="utf-8")
    assert "def markdown_price(price):" in core
    assert "from pkg.core import markdown_price" in shop
    assert "return markdown_price(p)" in shop
    # the two places the old name is DATA, not a reference, must survive
    assert "# apply_discount here" in shop
    assert "note = 'apply_discount'" in shop
    # `cents(apply_discount)` is a different binding but the SAME name: a
    # project-wide rename follows names, so it moves too, and the file parses
    util = (root / "pkg/util.py").read_text(encoding="utf-8")
    assert "def cents(markdown_price):" in util
    assert "return markdown_price * 100" in util
    for rel in res["files_modified"]:
        assert res["diffs"][rel].startswith("--- ")


def test_rename_dry_run_writes_nothing(tmp_path):
    root = _tree(tmp_path, PY_PKG)
    before = (root / "pkg/core.py").read_text(encoding="utf-8")
    res = refactor.rename_symbol_across_project(root, "apply_discount",
                                               "markdown_price", dry_run=True)
    assert res["ok"] and res["dry_run"] and res["files_modified"]
    assert "def markdown_price" in res["diffs"]["pkg/core.py"]
    assert (root / "pkg/core.py").read_text(encoding="utf-8") == before
    assert list(root.rglob("*.pyc")) == []


def test_rename_scope_limits_to_the_file_and_its_importers(tmp_path):
    root = _tree(tmp_path, PY_PKG)
    res = refactor.rename_symbol_across_project(root, "apply_discount",
                                               "off", scope_file="pkg/core.py")
    assert "def off(price):" in (root / "pkg/core.py").read_text(encoding="utf-8")
    assert "from pkg.core import off" in (root / "pkg/shop.py").read_text(encoding="utf-8")
    # util.py has its own `apply_discount` parameter and does not import core
    assert "def cents(apply_discount):" in (root / "pkg/util.py").read_text(encoding="utf-8")


def test_rename_validates_names_and_refuses_nonsense(tmp_path):
    root = _tree(tmp_path, PY_PKG)
    for old, new, why in [("apply discount", "x", "is not a valid identifier"),
                          ("apply_discount", "2fast", "is not a valid identifier"),
                          ("apply_discount", "class", "keyword"),
                          ("apply_discount", "apply_discount", "same")]:
        res = refactor.rename_symbol_across_project(root, old, new)
        assert not res["ok"] and why in res["error"], res
        assert res["files_modified"] == []
    assert "def apply_discount(price):" in (root / "pkg/core.py").read_text(encoding="utf-8")


def test_rename_is_all_or_nothing_on_a_gate_failure(tmp_path):
    """A `before_write` refusal (permission gate, or a failed snapshot) means
    no file lands, even though every candidate already validated."""
    root = _tree(tmp_path, PY_PKG)
    seen = []

    def gate(rel, path):
        seen.append(rel)
        return f"{rel} is not in this task's allowlist" if rel == "pkg/shop.py" else None

    res = refactor.rename_symbol_across_project(root, "apply_discount",
                                               "markdown_price",
                                               before_write=gate)
    assert not res["ok"] and "allowlist" in res["error"]
    assert set(seen) == {"pkg/core.py", "pkg/shop.py"}
    assert "def apply_discount(price):" in (root / "pkg/core.py").read_text(encoding="utf-8")
    assert "# apply_discount here" in (root / "pkg/shop.py").read_text(encoding="utf-8")


def test_rename_skips_venv_and_protected_files(tmp_path):
    files = dict(PY_PKG)
    files["venv/lib/site-packages/copy_of_core.py"] = files["pkg/core.py"]
    files[".pedacito/backups/20260101-000000/pkg/core.py"] = files["pkg/core.py"]
    root = _tree(tmp_path, files)
    res = refactor.rename_symbol_across_project(root, "apply_discount",
                                               "markdown_price")
    assert sorted(res["files_modified"]) == ["pkg/core.py", "pkg/shop.py",
                                             "pkg/util.py"]
    vendored = root / "venv/lib/site-packages/copy_of_core.py"
    assert "def apply_discount" in vendored.read_text(encoding="utf-8")
    stored = root / ".pedacito/backups/20260101-000000/pkg/core.py"
    assert "def apply_discount" in stored.read_text(encoding="utf-8")


@pytest.mark.skipif(not TS_AVAILABLE, reason="tree-sitter not installed")
def test_rename_javascript_uses_identifier_nodes(tmp_path):
    js = ("// apply_discount is a note in prose\n"
          "export function apply_discount(price) {\n"
          "  const label = 'apply_discount';\n"
          "  return apply_discount === null ? price : price * 0.9;\n"
          "}\n"
          "console.log('apply_discount');\n")
    root = _tree(tmp_path, {"src/cart.js": js})
    res = refactor.rename_symbol_across_project(root, "apply_discount",
                                               "markdown_price")
    assert res["ok"] and res["files_modified"] == ["src/cart.js"]
    after = (root / "src/cart.js").read_text(encoding="utf-8")
    assert "export function markdown_price(price)" in after
    assert "return markdown_price === null" in after
    assert "const label = 'apply_discount';" in after        # string kept
    assert "// apply_discount is a note in prose" in after   # comment kept
    assert "console.log('apply_discount');" in after         # string kept


# ---------------------------------------------------------------- tool layer

def _cfg(**kw):
    cfg = Config()
    cfg.backup = False
    cfg.log_events = False
    cfg.allow_write = True
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _index(tmp_path: Path, files: dict[str, str], cfg=None) -> Index:
    from pedacito import index as index_mod
    root = _tree(tmp_path, files)
    return index_mod.build([str(root)], cfg or _cfg(), client=None,
                           summarise=False, root=root, verbose=False)


def test_rename_symbol_is_offered_only_when_writing_is_allowed(tmp_path):
    ro = tools.schema_for(_cfg(allow_write=False))["properties"]["action"]["enum"]
    rw = tools.schema_for(_cfg(allow_write=True))["properties"]["action"]["enum"]
    assert "rename_symbol" not in ro
    assert "rename_symbol" in rw
    # the legacy index-based `rename` stays, and both appear once writing is
    # allowed (they are mutating actions, so neither shows up read-only)
    assert "rename" in rw and "rename" not in ro


def test_dispatch_rename_symbol(tmp_path):
    cfg = _cfg()
    index = _index(tmp_path, PY_PKG, cfg)
    out = tools.dispatch(index, cfg, {"action": "rename_symbol",
                                      "file": "pkg/core.py",
                                      "symbol": "apply_discount",
                                      "to": "markdown_price"})
    assert out.startswith("RENAMED"), out
    assert "def markdown_price(price):" in (tmp_path / "pkg/core.py").read_text(encoding="utf-8")
    assert "# apply_discount here" in (tmp_path / "pkg/shop.py").read_text(encoding="utf-8")


def test_tool_rename_symbol_preview_then_apply(tmp_path):
    root = _tree(tmp_path, PY_PKG)
    index = Index(root, [])
    cfg = _cfg()
    preview = refactor.tool_rename_symbol(index, cfg, "apply_discount",
                                          "markdown_price", dry_run=True)
    assert preview.startswith("RENAME PREVIEW") and "dry run" in preview
    assert "def apply_discount(price):" in (root / "pkg/core.py").read_text(encoding="utf-8")
    done = refactor.tool_rename_symbol(index, cfg, "apply_discount",
                                       "markdown_price", file="pkg/core.py")
    assert done.startswith("RENAMED")
    assert "def markdown_price(price):" in (root / "pkg/core.py").read_text(encoding="utf-8")
    assert "from pkg.core import markdown_price" in (root / "pkg/shop.py").read_text(encoding="utf-8")
    assert "# apply_discount here" in (root / "pkg/shop.py").read_text(encoding="utf-8")
    cfg_ro = _cfg(allow_write=False)
    assert "allow_write" in refactor.tool_rename_symbol(index, cfg_ro, "a", "b")


def test_rename_symbol_records_task_mutations(tmp_path):
    root = _tree(tmp_path, PY_PKG)
    index = Index(root, [])
    cfg = _cfg()
    refactor.tool_rename_symbol(index, cfg, "apply_discount", "markdown_price",
                                file="pkg/core.py")
    assert set(cfg._task_mutations) >= {"pkg/core.py", "pkg/shop.py"}


# ------------------------------------------------------------------ GUI route

def test_gui_rename_route_preview_and_apply(tmp_path, monkeypatch):
    from pedacito.gui import core, server

    root = _tree(tmp_path, PY_PKG)
    st = core.STATE
    monkeypatch.setattr(st, "root", root)
    monkeypatch.setattr(st, "cfg", _cfg())
    monkeypatch.setattr(st, "idx", None)
    monkeypatch.setattr(st, "job", None)

    status, j = server.POST_ROUTES["/api/refactor/rename"](
        {}, {"old": "apply_discount", "new": "markdown_price"})
    assert status == 200 and j["ok"] and j["dry_run"]
    assert j["files_modified"] == ["pkg/core.py", "pkg/shop.py", "pkg/util.py"]
    assert "def apply_discount" in (root / "pkg/core.py").read_text(encoding="utf-8")

    status, j = server.POST_ROUTES["/api/refactor/rename"](
        {}, {"old": "apply_discount", "new": "markdown_price", "apply": True})
    assert status == 200 and not j["dry_run"]
    assert "def markdown_price(price):" in (root / "pkg/core.py").read_text(encoding="utf-8")
    assert "from pkg.core import markdown_price" in (root / "pkg/shop.py").read_text(encoding="utf-8")
    assert j.get("backup"), "applying through the GUI must snapshot originals"
    assert Path(root / j["backup"]).is_dir()

    for bad, code in (({"old": "x"}, 400),
                      ({"old": "a", "new": "b", "file": "../outside/core.py"}, 400)):
        status, j = server.POST_ROUTES["/api/refactor/rename"]({}, bad)
        assert status == code and not j["ok"]
    monkeypatch.setattr(st, "root", None)
    status, j = server.POST_ROUTES["/api/refactor/rename"]({}, {"old": "a", "new": "b"})
    assert status == 409


def test_gui_app_js_exposes_rename():
    src = (Path(cli.__file__).resolve().parent / "gui/static/app.js"
           ).read_text(encoding="utf-8")
    assert "function renameSymbolDialog()" in src
    assert 'label="Rename symbol\\u2026"' in src
    assert "/api/refactor/rename" in src
    # the new dialog's own copy must stay free of em-dashes (UI strings rule)
    start = src.index("// Deterministic symbol rename (Sprint P)")
    block = src[start:start + 4000]
    assert "\u2014" not in block


# ------------------------------------------------------------------ CLI layer

def test_cli_rename_parser_flags(tmp_path, capsys):
    # `rename` is a light command (no config, no model, no index build needed)
    # and takes both names; argparse's exit code for a missing one is 2.
    assert callable(cli._cmd_rename)
    with pytest.raises(SystemExit) as e:
        cli.main(["rename", "only_one", "--root", str(tmp_path)])
    assert e.value.code == 2
    capsys.readouterr()
    # a project without the name exits quietly with 0 (nothing to undo)
    assert cli.main(["rename", "nada", "x", "--root", str(tmp_path)]) == 0
    assert "nothing renamed" in capsys.readouterr().out


def test_cli_rename_dry_run_then_apply(tmp_path, capsys):
    root = _tree(tmp_path, PY_PKG)
    assert cli.main(["rename", "apply_discount", "markdown_price",
                     "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "would rename" in out and "dry run" in out
    assert "apply_discount" in out                      # the string survives
    assert cli.main(["rename", "apply_discount", "markdown_price",
                     "--file", "pkg/core.py", "--apply", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "renamed" in out and "snapshotted" in out.lower()
    assert "def markdown_price(price):" in (root / "pkg/core.py").read_text(encoding="utf-8")
    assert (root / "pkg/shop.py").read_text(encoding="utf-8").count("markdown_price") == 2


# ------------------------------------------------------- the agent's apply path

def test_agent_applies_a_zero_anchor_splice(tmp_path):
    agent = _agent(tmp_path, Scripted(), PY_PKG, allow_write=True)
    reply = ("--- FILE: pkg/core.py\n"
             "<<<<<<< SYMBOL pkg/core.py :: apply_discount\n"
             "def discounted_price(price):\n    return price * 0.85\n"
             ">>>>>>> END_SYMBOL\n")
    res = agent._apply_once(reply, dry_run=False)
    assert res[0].ok, res[0].message
    assert "def discounted_price(price):" in (tmp_path / "pkg/core.py").read_text(encoding="utf-8")
    # the index never saw the new name, and the old definition is gone
    assert "apply_discount" not in (tmp_path / "pkg/core.py").read_text(encoding="utf-8")


def test_agent_splices_a_file_the_index_never_saw(tmp_path):
    """A file created mid-run (so absent from the index) still resolves: the
    anchor falls back to the file's own AST at apply time."""
    agent = _agent(tmp_path, Scripted(), PY_PKG, allow_write=True)
    (tmp_path / "pkg" / "new.py").write_text(
        "def draft(text):\n    return text.strip()\n", encoding="utf-8")
    assert "pkg/new.py" not in agent.index.files
    reply = ("--- FILE: pkg/new.py\n"
             "<<<<<<< SYMBOL pkg/new.py :: draft\n"
             "def draft(text):\n    return text.strip().lower()\n"
             ">>>>>>> END_SYMBOL\n")
    res = agent._apply_once(reply, dry_run=False)
    assert res[0].ok, res[0].message
    assert "return text.strip().lower()" in (tmp_path / "pkg" / "new.py").read_text(encoding="utf-8")


def test_agent_reports_an_unterminated_splice_as_malformed(tmp_path):
    agent = _agent(tmp_path, Scripted(), PY_PKG, allow_write=True)
    reply = ("--- FILE: pkg/core.py\n"
             "<<<<<<< SYMBOL pkg/core.py :: apply_discount\n"
             "def discounted_price(price):\n    return price\n")
    res = agent._apply_once(reply, dry_run=False)
    assert any(r.file == "pkg/core.py" and "malformed edit block" in r.message
               and "END_SYMBOL" in r.message for r in res), [r.message for r in res]
    assert "def apply_discount" in (tmp_path / "pkg/core.py").read_text(encoding="utf-8")


def test_gather_rename_symbol_action_renames_for_real(tmp_path):
    """The rename engine is one step in the loop: a scripted `rename_symbol`
    action moves the definition, its importers and the mutation ledger, with
    no edit block and no model tokens spent on rewriting files."""
    client = Scripted(steps=[
        {"action": "rename_symbol", "file": "pkg/core.py",
         "symbol": "apply_discount", "to": "discounted_price"},
        {"action": "done"}])
    agent = _agent(tmp_path, client, PY_PKG, allow_write=True)
    agent.gather("rename apply_discount to discounted_price", verbose=False)
    assert "def discounted_price(price):" in (tmp_path / "pkg/core.py").read_text(encoding="utf-8")
    assert "from pkg.core import discounted_price" in (tmp_path / "pkg/shop.py").read_text(encoding="utf-8")
    assert "# apply_discount here" in (tmp_path / "pkg/shop.py").read_text(encoding="utf-8")
    assert set(agent.cfg._task_mutations) >= {"pkg/core.py", "pkg/shop.py"}
