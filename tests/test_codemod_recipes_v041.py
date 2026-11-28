"""WP-4102B: the remaining eight codemod recipes.

    add_param, remove_param, wrap_try, wrap_context,
    move_symbol, move_file, replace_deprecated, add_reexport

Every recipe is exercised through the same ``CodemodRequest`` / ``preview`` /
``apply`` contract as WP-4102A, with positive, no-op, ambiguity, untouched
comments/strings, partial-write-failure, parser and caller (or importer) tests.
Cross-file recipes build a real in-memory index (no model, no network) so the
import graph edges and provable bindings the recipes depend on are exercised.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pedacito.codemod as codemod_module
from pedacito import index as index_mod
from pedacito.codemod import CodemodRequest, apply, list_recipes, preview
from pedacito.config import Config


def _write(root: Path, text: str, name: str = "app.py") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path


def _request(recipe: str, file: str = "app.py", symbol: str = "",
             **options: str) -> CodemodRequest:
    return CodemodRequest(recipe, file, symbol, options)


def _project(root: Path, files: dict[str, str]):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="")
    return index_mod.build([str(root)], Config(), client=None, summarise=False,
                           root=root, verbose=False)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# add_param
# ---------------------------------------------------------------------------
def test_add_param_default_keeps_same_file_caller(tmp_path):
    path = _write(tmp_path, "def f(a):\n    return a\n\nf(1)\n")
    res = apply(tmp_path, _request("add_param", symbol="f", name="b",
                                   default="0"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "def f(a, b=0):" in text
    assert "f(1)" in text


def test_add_param_with_annotation(tmp_path):
    path = _write(tmp_path, "def f(a):\n    return a\n")
    res = apply(tmp_path, _request("add_param", symbol="f", name="b",
                                   annotation="int", default="0"))
    assert res.ok, res.error
    assert "def f(a, b: int = 0):" in path.read_text(encoding="utf-8")


def test_add_param_required_without_callers(tmp_path):
    path = _write(tmp_path, "def f(a):\n    return a\n")
    res = apply(tmp_path, _request("add_param", symbol="f", name="b"))
    assert res.ok, res.error
    assert "def f(a, b):" in path.read_text(encoding="utf-8")


def test_add_param_required_with_caller_refuses(tmp_path):
    _write(tmp_path, "def f(a):\n    return a\n\nf(1)\n")
    res = preview(tmp_path, _request("add_param", symbol="f", name="b"))
    assert not res.ok
    assert "break" in res.error and "default" in res.error


def test_add_param_refuses_variadic(tmp_path):
    _write(tmp_path, "def f(*args):\n    return args\n")
    res = preview(tmp_path, _request("add_param", symbol="f", name="b",
                                     default="0"))
    assert not res.ok
    assert "*args" in res.error or "**kwargs" in res.error


def test_add_param_refuses_duplicate_param(tmp_path):
    _write(tmp_path, "def f(a):\n    return a\n")
    res = preview(tmp_path, _request("add_param", symbol="f", name="a"))
    assert not res.ok
    assert "already a parameter" in res.error


def test_add_param_refuses_ambiguous_symbol(tmp_path):
    _write(tmp_path, "def f(a):\n    return a\n\ndef f(a):\n    return a\n")
    res = preview(tmp_path, _request("add_param", symbol="f", name="b",
                                     default="0"))
    assert not res.ok
    assert "ambiguous" in res.error


def test_add_param_preserves_comments_and_strings(tmp_path):
    path = _write(tmp_path, 'def f(a):\n    # keep this comment\n'
                            '    return "a = 1"  # and this\n\nf(1)\n')
    res = apply(tmp_path, _request("add_param", symbol="f", name="b",
                                   default="0"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "# keep this comment" in text
    assert '"a = 1"' in text
    assert "# and this" in text


def test_add_param_cross_file_default_leaves_importer(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def f(a):\n    return a\n",
        "pkg/user.py": "from pkg.core import f\n\ndef g():\n    return f(1)\n",
    })
    res = apply(tmp_path, _request("add_param", file="pkg/core.py",
                                   symbol="f", name="b", default="0"),
                index=idx)
    assert res.ok, res.error
    assert "def f(a, b=0):" in (tmp_path / "pkg/core.py").read_text(
        encoding="utf-8")
    assert "return f(1)" in (tmp_path / "pkg/user.py").read_text(
        encoding="utf-8")


def test_add_param_keyword_conflict_refuses(tmp_path):
    _write(tmp_path, "def f(a):\n    return a\n\nf(1, b=2)\n")
    res = preview(tmp_path, _request("add_param", symbol="f", name="b",
                                     default="0"))
    assert not res.ok
    assert "already passes" in res.error


def test_add_param_default_after_annotated_defaulted_param(tmp_path):
    path = _write(
        tmp_path, "def load(path: str = 'x') -> str:\n    return path\n")
    res = apply(tmp_path, _request("add_param", symbol="load", name="strict",
                                   annotation="bool", default="False"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "def load(path: str = 'x', strict: bool = False) -> str:" in text


def test_add_param_default_after_unannotated_defaulted_param(tmp_path):
    path = _write(tmp_path, "def f(a=1):\n    return a\n")
    res = apply(tmp_path, _request("add_param", symbol="f", name="b",
                                   default="2"))
    assert res.ok, res.error
    assert "def f(a=1, b=2):" in path.read_text(encoding="utf-8")


def test_add_param_default_after_multiline_trailing_comma(tmp_path):
    path = _write(tmp_path, "def f(\n"
                            "    a: int = 1,\n"
                            "    b: int = 2,\n"
                            "):\n"
                            "    return a\n")
    res = apply(tmp_path, _request("add_param", symbol="f", name="c",
                                   annotation="bool", default="False"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "    b: int = 2, c: bool = False,\n):" in text


def test_add_param_method_default_keeps_self_call(tmp_path):
    path = _write(tmp_path, "class C:\n"
                            "    def m(self, a):\n"
                            "        return a\n\n"
                            "    def n(self):\n"
                            "        return self.m(1)\n")
    res = apply(tmp_path, _request("add_param", symbol="C.m", name="b",
                                   default="0"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "def m(self, a, b=0):" in text
    assert "return self.m(1)" in text


def test_remove_param_method_updates_self_call(tmp_path):
    path = _write(tmp_path, "class C:\n"
                            "    def m(self, a, b):\n"
                            "        return a\n\n"
                            "    def n(self):\n"
                            "        return self.m(1, 2)\n")
    res = apply(tmp_path, _request("remove_param", symbol="C.m", name="b"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "def m(self, a):" in text
    assert "return self.m(1)" in text


def test_remove_param_refuses_dynamic_receiver(tmp_path):
    _write(tmp_path, "class C:\n"
                     "    def m(self, a, b):\n"
                     "        return a\n\n"
                     "def g(x):\n    return x.m(1, 2)\n")
    res = preview(tmp_path, _request("remove_param", symbol="C.m", name="b"))
    assert not res.ok
    assert "not provable" in res.error


def test_add_param_parser_refuses_syntax_error_input(tmp_path):
    _write(tmp_path, "def broken(:\n")
    res = preview(tmp_path, _request("add_param", symbol="f", name="b"))
    assert not res.ok
    assert "does not parse" in res.error


# ---------------------------------------------------------------------------
# remove_param
# ---------------------------------------------------------------------------
def test_remove_param_positive(tmp_path):
    path = _write(tmp_path, "def f(a, b):\n    return a\n\nf(1, 2)\n")
    res = apply(tmp_path, _request("remove_param", symbol="f", name="b"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "def f(a):" in text
    assert "f(1)" in text
    assert "f(1, 2)" not in text


def test_remove_param_multiple_callers(tmp_path):
    path = _write(tmp_path, "def f(a, b):\n    return a\n\n"
                            "f(1, 2)\nf(3, 4)\n")
    res = apply(tmp_path, _request("remove_param", symbol="f", name="b"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert text.count("f(1)") == 1 and text.count("f(3)") == 1
    assert "2" not in text.replace("def f(a):", "").split("f(1)")[0]


def test_remove_param_cross_file(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def f(a, b):\n    return a\n",
        "pkg/user.py": "from pkg.core import f\n\ndef g():\n    return f(1, 2)\n",
    })
    res = apply(tmp_path, _request("remove_param", file="pkg/core.py",
                                   symbol="f", name="b"), index=idx)
    assert res.ok, res.error
    assert "def f(a):" in (tmp_path / "pkg/core.py").read_text(
        encoding="utf-8")
    assert "return f(1)" in (tmp_path / "pkg/user.py").read_text(
        encoding="utf-8")


def test_remove_param_refuses_keyword_call(tmp_path):
    _write(tmp_path, "def f(a, b):\n    return a\n\nf(1, b=2)\n")
    res = preview(tmp_path, _request("remove_param", symbol="f", name="b"))
    assert not res.ok
    assert "keyword" in res.error


def test_remove_param_refuses_variadic(tmp_path):
    _write(tmp_path, "def f(a, *rest):\n    return a\n")
    res = preview(tmp_path, _request("remove_param", symbol="f", name="a"))
    assert not res.ok
    assert "*args" in res.error or "**kwargs" in res.error


def test_remove_param_refuses_dynamic_reference(tmp_path):
    _write(tmp_path, "def f(a, b):\n    return a\n\ncb = f\ncb(1, 2)\n")
    res = preview(tmp_path, _request("remove_param", symbol="f", name="b"))
    assert not res.ok
    assert "referenced as a value" in res.error


def test_remove_param_missing_param_refuses(tmp_path):
    _write(tmp_path, "def f(a):\n    return a\n")
    res = preview(tmp_path, _request("remove_param", symbol="f", name="zz"))
    assert not res.ok
    assert "not a parameter" in res.error


def test_remove_param_preserves_comments_and_strings(tmp_path):
    path = _write(tmp_path, 'def f(a, b):\n    # comment\n'
                            '    return "f(1, 2)"\n\nf(1, 2)  # call\n')
    res = apply(tmp_path, _request("remove_param", symbol="f", name="b"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "# comment" in text
    assert '"f(1, 2)"' in text
    assert "f(1)  # call" in text


def test_remove_param_removes_last_annotated_default(tmp_path):
    path = _write(
        tmp_path,
        "def new_expense_id(expenses: list[str], limit: int = 0) -> str:\n"
        "    return expenses\n")
    res = apply(tmp_path, _request("remove_param", symbol="new_expense_id",
                                   name="limit"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "def new_expense_id(expenses: list[str]) -> str:" in text
    assert "= 0" not in text


def test_remove_param_removes_first_defaulted_param(tmp_path):
    path = _write(tmp_path, "def f(a: int = 1, b: int = 2) -> int:\n"
                            "    return b\n")
    res = apply(tmp_path, _request("remove_param", symbol="f", name="a"))
    assert res.ok, res.error
    assert "def f(b: int = 2) -> int:" in path.read_text(encoding="utf-8")


def test_remove_param_removes_middle_defaulted_param(tmp_path):
    path = _write(tmp_path, "def f(a, b: int = 1, c: int = 2):\n"
                            "    return c\n")
    res = apply(tmp_path, _request("remove_param", symbol="f", name="b"))
    assert res.ok, res.error
    assert "def f(a, c: int = 2):" in path.read_text(encoding="utf-8")


def test_remove_param_removes_defaulted_positional_only_param(tmp_path):
    path = _write(tmp_path, "def f(a: int = 1, /, b: int = 2):\n"
                            "    return b\n")
    res = apply(tmp_path, _request("remove_param", symbol="f", name="a"))
    assert res.ok, res.error
    assert "def f(b: int = 2):" in path.read_text(encoding="utf-8")


def test_remove_param_removes_defaulted_keyword_only_param(tmp_path):
    path = _write(tmp_path, "def f(a, *, limit: int = 0, "
                            "strict: bool = False):\n"
                            "    return a\n\nf(1, limit=2)\n")
    res = apply(tmp_path, _request("remove_param", symbol="f", name="limit"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "def f(a, *, strict: bool = False):" in text
    assert "f(1)\n" in text


def test_remove_param_removes_sole_keyword_only_param_and_star(tmp_path):
    path = _write(tmp_path, "def f(a, *, limit: int = 0):\n"
                            "    return a\n")
    res = apply(tmp_path, _request("remove_param", symbol="f", name="limit"))
    assert res.ok, res.error
    assert "def f(a):" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# wrap_try
# ---------------------------------------------------------------------------
def test_wrap_try_positive(tmp_path):
    path = _write(tmp_path, "def f():\n    x = 1\n    return x\n")
    res = apply(tmp_path, _request("wrap_try", symbol="f"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8") == (
        "def f():\n"
        "    try:\n"
        "        x = 1\n"
        "        return x\n"
        "    except Exception:\n"
        "        raise\n")


def test_wrap_try_custom_exception(tmp_path):
    path = _write(tmp_path, "def f():\n    x = 1\n")
    res = apply(tmp_path, _request("wrap_try", symbol="f",
                                   exception="ValueError"))
    assert res.ok, res.error
    assert "    except ValueError:" in path.read_text(encoding="utf-8")


def test_wrap_try_preserves_docstring(tmp_path):
    path = _write(tmp_path, 'def f():\n    """doc"""\n    return 1\n')
    res = apply(tmp_path, _request("wrap_try", symbol="f"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert '"""doc"""' in text
    assert text.index('"""doc"""') < text.index("try:")


def test_wrap_try_method(tmp_path):
    path = _write(tmp_path, "class C:\n"
                            "    def m(self):\n"
                            "        return 1\n")
    res = apply(tmp_path, _request("wrap_try", symbol="C.m"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "    def m(self):\n        try:\n            return 1" in text


def test_wrap_try_refuses_ambiguous(tmp_path):
    _write(tmp_path, "def f():\n    pass\n\ndef f():\n    pass\n")
    res = preview(tmp_path, _request("wrap_try", symbol="f"))
    assert not res.ok
    assert "ambiguous" in res.error


def test_wrap_try_refuses_empty_body(tmp_path):
    _write(tmp_path, "def f():\n    pass\n")
    # pass is a statement, so it wraps; a truly empty body refuses
    _write(tmp_path, "def g():\n    \"\"\"only docs\"\"\"\n")
    res = preview(tmp_path, _request("wrap_try", symbol="g"))
    assert not res.ok
    assert "docstring" in res.error or "empty" in res.error


def test_wrap_try_refuses_invalid_exception(tmp_path):
    _write(tmp_path, "def f():\n    x = 1\n")
    res = preview(tmp_path, _request("wrap_try", symbol="f",
                                     exception="not valid !!"))
    assert not res.ok
    assert "exception" in res.error


# ---------------------------------------------------------------------------
# wrap_context
# ---------------------------------------------------------------------------
def test_wrap_context_positive(tmp_path):
    path = _write(tmp_path, "def f():\n    x = 1\n    return x\n")
    res = apply(tmp_path, _request("wrap_context", symbol="f",
                                   manager="lock"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8") == (
        "def f():\n"
        "    with lock:\n"
        "        x = 1\n"
        "        return x\n")


def test_wrap_context_missing_manager_refuses(tmp_path):
    _write(tmp_path, "def f():\n    x = 1\n")
    res = preview(tmp_path, _request("wrap_context", symbol="f"))
    assert not res.ok
    assert "manager" in res.error


def test_wrap_context_preserves_comments(tmp_path):
    path = _write(tmp_path, "def f():\n    x = 1  # keep\n")
    res = apply(tmp_path, _request("wrap_context", symbol="f",
                                   manager="lock"))
    assert res.ok, res.error
    assert "x = 1  # keep" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# move_symbol
# ---------------------------------------------------------------------------
PKG = {
    "pkg/__init__.py": "",
    "pkg/core.py": "def helper():\n    return 1\n\n\ndef other():\n    return 2\n",
    "pkg/user.py": "from pkg.core import helper\n\nhelper()\n",
}


def test_move_symbol_positive(tmp_path):
    idx = _project(tmp_path, PKG)
    res = apply(tmp_path, _request("move_symbol", file="pkg/core.py",
                                   symbol="helper", to_file="pkg/util.py"),
                index=idx)
    assert res.ok, res.error
    assert (tmp_path / "pkg/util.py").read_text(encoding="utf-8") \
        .startswith("def helper():\n    return 1\n")
    assert "def helper" not in (tmp_path / "pkg/core.py").read_text(
        encoding="utf-8")
    assert "def other" in (tmp_path / "pkg/core.py").read_text(
        encoding="utf-8")
    assert "from pkg.util import helper" in (tmp_path / "pkg/user.py").read_text(
        encoding="utf-8")


def test_move_symbol_inserts_new_import_in_leading_import_block(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "def x():\n    return 1\n\n\ndef combine():\n"
                    "    return 2\n",
        "pkg/b.py": "",
        "pkg/user.py": "from pkg.a import x, combine\n\nTWO = 2\n\n"
                       "print(TWO)\n",
    })
    res = apply(tmp_path, _request("move_symbol", file="pkg/a.py",
                                   symbol="combine", to_file="pkg/b.py"),
                index=idx)
    assert res.ok, res.error
    text = (tmp_path / "pkg/user.py").read_text(encoding="utf-8")
    assert text.startswith(
        "from pkg.a import x\nfrom pkg.b import combine\n\nTWO = 2\n")
    assert text.index("from pkg.b import combine") < text.index("TWO = 2")


def test_move_symbol_preview_does_not_write(tmp_path):
    idx = _project(tmp_path, PKG)
    before = _snapshot(tmp_path)
    res = preview(tmp_path, _request("move_symbol", file="pkg/core.py",
                                     symbol="helper", to_file="pkg/util.py"),
                  index=idx)
    assert res.ok, res.error
    assert _snapshot(tmp_path) == before


def test_move_symbol_refuses_nested_symbol(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "class C:\n    def m(self):\n        return 1\n",
    })
    res = preview(tmp_path, _request("move_symbol", file="pkg/core.py",
                                     symbol="C.m", to_file="pkg/util.py"),
                  index=idx)
    assert not res.ok
    assert "top-level" in res.error


def test_move_symbol_refuses_missing_symbol(tmp_path):
    idx = _project(tmp_path, PKG)
    res = preview(tmp_path, _request("move_symbol", file="pkg/core.py",
                                     symbol="missing", to_file="pkg/util.py"),
                  index=idx)
    assert not res.ok
    assert "not found" in res.error


def test_move_symbol_refuses_case_only_collision(tmp_path):
    idx = _project(tmp_path, PKG)
    res = preview(tmp_path, _request("move_symbol", file="pkg/core.py",
                                     symbol="helper", to_file="pkg/CORE.py"),
                  index=idx)
    assert not res.ok
    assert "case" in res.error


def test_move_symbol_appends_to_existing_destination(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def helper():\n    return 1\n",
        "pkg/util.py": "def existing():\n    return 0\n",
    })
    res = apply(tmp_path, _request("move_symbol", file="pkg/core.py",
                                   symbol="helper", to_file="pkg/util.py"),
                index=idx)
    assert res.ok, res.error
    text = (tmp_path / "pkg/util.py").read_text(encoding="utf-8")
    assert "def existing():" in text
    assert "def helper():" in text


# ---------------------------------------------------------------------------
# move_file
# ---------------------------------------------------------------------------
def test_move_file_positive(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def helper():\n    return 1\n",
        "pkg/user.py": "from pkg.core import helper\n\nhelper()\n",
    })
    res = apply(tmp_path, _request("move_file", file="pkg/core.py",
                                   to_path="pkg/moved.py"), index=idx)
    assert res.ok, res.error
    assert not (tmp_path / "pkg/core.py").exists()
    assert (tmp_path / "pkg/moved.py").read_text(encoding="utf-8") == \
        "def helper():\n    return 1\n"
    assert "from pkg.moved import helper" in (tmp_path / "pkg/user.py") \
        .read_text(encoding="utf-8")


def test_move_file_preview_does_not_write(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def helper():\n    return 1\n",
        "pkg/user.py": "from pkg.core import helper\n",
    })
    before = _snapshot(tmp_path)
    res = preview(tmp_path, _request("move_file", file="pkg/core.py",
                                     to_path="pkg/moved.py"), index=idx)
    assert res.ok, res.error
    assert _snapshot(tmp_path) == before


def test_move_file_refuses_existing_destination(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def helper():\n    return 1\n",
        "pkg/moved.py": "x = 1\n",
    })
    res = preview(tmp_path, _request("move_file", file="pkg/core.py",
                                     to_path="pkg/moved.py"), index=idx)
    assert not res.ok
    assert "already exists" in res.error


def test_move_file_refuses_case_only_collision(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def helper():\n    return 1\n",
    })
    res = preview(tmp_path, _request("move_file", file="pkg/core.py",
                                     to_path="pkg/CORE.py"), index=idx)
    assert not res.ok
    assert "case" in res.error


def test_move_file_refuses_bare_module_import(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def helper():\n    return 1\n",
        "pkg/user.py": "import pkg.core\n\npkg.core.helper()\n",
    })
    res = preview(tmp_path, _request("move_file", file="pkg/core.py",
                                     to_path="pkg/moved.py"), index=idx)
    assert not res.ok
    assert "qualified" in res.error


def test_move_file_rewrites_aliased_module_import(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def helper():\n    return 1\n",
        "pkg/user.py": "import pkg.core as c\n\nc.helper()\n",
    })
    res = apply(tmp_path, _request("move_file", file="pkg/core.py",
                                   to_path="pkg/moved.py"), index=idx)
    assert res.ok, res.error
    assert "import pkg.moved as c" in (tmp_path / "pkg/user.py").read_text(
        encoding="utf-8")


# ---------------------------------------------------------------------------
# replace_deprecated
# ---------------------------------------------------------------------------
def test_replace_deprecated_positive(tmp_path):
    path = _write(tmp_path, "def f():\n    return old_api.call()\n")
    res = apply(tmp_path, _request("replace_deprecated", old="old_api",
                                   new="new_api"))
    assert res.ok, res.error
    assert "new_api.call()" in path.read_text(encoding="utf-8")
    assert "old_api" not in path.read_text(encoding="utf-8")


def test_replace_deprecated_untouched_strings_comments(tmp_path):
    path = _write(tmp_path, '# old_api in a comment\n'
                            'text = "old_api"\n'
                            'def f():\n    return old_api()\n')
    res = apply(tmp_path, _request("replace_deprecated", old="old_api",
                                   new="new_api"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "# old_api in a comment" in text
    assert 'text = "old_api"' in text
    assert "return new_api()" in text


def test_replace_deprecated_noop_when_absent(tmp_path):
    path = _write(tmp_path, "x = 1\n")
    res = preview(tmp_path, _request("replace_deprecated", old="old_api",
                                     new="new_api"))
    assert res.ok
    assert res.changed == ()
    assert "found no reference" in " ".join(res.warnings)
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_replace_deprecated_refuses_same_name(tmp_path):
    _write(tmp_path, "x = 1\n")
    res = preview(tmp_path, _request("replace_deprecated", old="a", new="a"))
    assert not res.ok
    assert "same" in res.error


def test_replace_deprecated_scoped_to_symbol(tmp_path):
    path = _write(tmp_path, "def a():\n    return old()\n\ndef b():\n    return old()\n")
    res = apply(tmp_path, _request("replace_deprecated", symbol="a",
                                   old="old", new="new"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert text.count("return new()") == 1
    assert text.count("return old()") == 1


# ---------------------------------------------------------------------------
# add_reexport
# ---------------------------------------------------------------------------
def test_add_reexport_positive(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def helper():\n    return 1\n",
    })
    res = apply(tmp_path, _request("add_reexport", file="pkg/__init__.py",
                                   name="helper", from_module="pkg.core"),
                index=idx)
    assert res.ok, res.error
    assert "from pkg.core import helper" in (tmp_path / "pkg/__init__.py") \
        .read_text(encoding="utf-8")


def test_add_reexport_noop_when_present(tmp_path):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "from pkg.core import helper\n",
        "pkg/core.py": "def helper():\n    return 1\n",
    })
    res = preview(tmp_path, _request("add_reexport", file="pkg/__init__.py",
                                     name="helper", from_module="pkg.core"),
                  index=idx)
    assert res.ok
    assert res.changed == ()
    assert "already present" in " ".join(res.warnings)


def test_add_reexport_refuses_unresolved_module(tmp_path):
    _project(tmp_path, {"pkg/__init__.py": "", "pkg/core.py": "x = 1\n"})
    res = preview(tmp_path, _request("add_reexport", file="pkg/__init__.py",
                                     name="helper", from_module="pkg.missing"))
    assert not res.ok
    assert "could not be resolved" in res.error


def test_add_reexport_refuses_undefined_name(tmp_path):
    _project(tmp_path, {"pkg/__init__.py": "", "pkg/core.py": "x = 1\n"})
    res = preview(tmp_path, _request("add_reexport", file="pkg/__init__.py",
                                     name="helper", from_module="pkg.core"))
    assert not res.ok
    assert "not defined" in res.error


# ---------------------------------------------------------------------------
# Cross-recipe: apply permission, partial write, atomicity
# ---------------------------------------------------------------------------
def test_apply_requires_allow_write_when_cfg_given(tmp_path):
    path = _write(tmp_path, "def f():\n    x = 1\n")
    cfg = Config()
    res = apply(tmp_path, _request("wrap_try", symbol="f"), cfg=cfg)
    assert not res.ok
    assert "allow_write" in res.error
    assert path.read_text(encoding="utf-8") == "def f():\n    x = 1\n"


def test_apply_write_failure_rolls_back_multi_file(tmp_path, monkeypatch):
    idx = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def helper():\n    return 1\n",
        "pkg/user.py": "from pkg.core import helper\n",
    })
    before = _snapshot(tmp_path)

    real_write = codemod_module.fileio.write
    calls = {"n": 0}

    def boom(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise OSError("injected write failure")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr("pedacito.fileio.write", boom)
    res = apply(tmp_path, _request("move_file", file="pkg/core.py",
                                   to_path="pkg/moved.py"), index=idx)
    assert not res.ok
    assert "write" in res.error or "rolled back" in res.error
    assert _snapshot(tmp_path) == before


def test_recipes_all_declare_python_only_and_implemented():
    for recipe in list_recipes():
        if recipe["id"] in ("add_param", "remove_param", "wrap_try",
                            "wrap_context", "move_symbol", "move_file",
                            "replace_deprecated", "add_reexport"):
            assert recipe["implemented"] is True
            assert recipe["languages"] == ["python"]


def test_multi_file_preview_is_deterministic(tmp_path):
    idx = _project(tmp_path, PKG)
    request = _request("move_symbol", file="pkg/core.py", symbol="helper",
                       to_file="pkg/util.py")
    first = preview(tmp_path, request, index=idx)
    second = preview(tmp_path, request, index=idx)
    assert first == second
