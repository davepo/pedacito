"""WP-4102A: codemod catalog, preview/apply framework and first five recipes.

The tests are deliberately local and deterministic: no model, network, Node or
LSP server is involved. They cover the exact ``CodemodRequest`` /
``CodemodResult`` contract, option metadata, AST/token precision (including
Unicode byte offsets), CRLF/BOM preservation, ambiguity refusals with no
writes, policy/protection gates and the atomic apply transaction.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pedacito.codemod as codemod_module
from pedacito.codemod import CodemodRequest, CodemodResult, apply, list_recipes, preview
from pedacito.config import Config
from pedacito.edits import Edit, apply_edits_atomic

EXPECTED_IDS = [
    "add_param", "remove_param", "add_import", "remove_import", "wrap_try",
    "wrap_context", "print_to_logger", "extract_constant", "add_annotation",
    "move_symbol", "move_file", "replace_deprecated", "add_reexport",
]


def _write(root: Path, text: str, name: str = "app.py",
           encoding: str = "utf-8") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding, newline="")
    return path


def _request(recipe: str, file: str = "app.py", symbol: str = "",
             **options: str) -> CodemodRequest:
    return CodemodRequest(recipe, file, symbol, options)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# Catalog and contract
# ---------------------------------------------------------------------------
def test_catalog_is_exact_and_documented():
    recipes = list_recipes()
    assert [r["id"] for r in recipes] == EXPECTED_IDS
    for recipe in recipes:
        assert recipe["title"]
        assert recipe["description"]
        assert recipe["status"] in ("implemented", "unverified")
        assert recipe["implemented"] is (recipe["status"] == "implemented")
        assert recipe["languages"] == ["python"]
        assert recipe["symbol"] in ("required", "optional", "ignored")
        assert recipe["option_names"] == [o["name"] for o in recipe["options"]]
        assert recipe["options_doc"] == {
            o["name"]: o["description"] for o in recipe["options"]}
        assert len(recipe["option_names"]) == len(set(recipe["option_names"]))
        for option in recipe["options"]:
            assert option["name"]
            assert isinstance(option["required"], bool)
            assert option["description"]


def test_first_five_recipes_are_implemented_and_rest_are_unverified():
    by_id = {r["id"]: r for r in list_recipes()}
    for recipe_id in ("add_import", "remove_import", "print_to_logger",
                      "extract_constant", "add_annotation"):
        assert by_id[recipe_id]["implemented"] is True
        assert by_id[recipe_id]["status"] == "implemented"
    for recipe_id in ("add_param", "remove_param", "wrap_try", "wrap_context",
                      "move_symbol", "move_file", "replace_deprecated",
                      "add_reexport"):
        assert by_id[recipe_id]["implemented"] is False
        assert by_id[recipe_id]["status"] == "unverified"


def test_catalog_returns_a_deep_copy():
    recipes = list_recipes()
    recipes[0]["options"][0]["name"] = "mutated"
    recipes[0]["id"] = "mutated"
    fresh = list_recipes()
    assert fresh[0]["id"] == "add_param"
    assert fresh[0]["options"][0]["name"] == "name"


def test_codemod_result_is_frozen():
    result = CodemodResult(True, {}, (), ())
    with pytest.raises(Exception):
        result.ok = False                   # type: ignore[misc]


def test_unknown_recipe_and_unimplemented_recipe_refuse(tmp_path):
    _write(tmp_path, "x = 1\n")
    unknown = preview(tmp_path, _request("does_not_exist"))
    assert not unknown.ok
    assert "unknown codemod recipe" in unknown.error
    assert unknown.changed == () and unknown.diffs == {}

    pending = preview(tmp_path, _request("add_param", symbol="f",
                                         name="x"))
    assert not pending.ok
    assert "not implemented" in pending.error
    assert "unverified" in pending.error


def test_unknown_option_refuses(tmp_path):
    _write(tmp_path, "x = 1\n")
    res = preview(tmp_path, _request("add_import", module="os", bogus="1"))
    assert not res.ok
    assert "unknown option" in res.error
    assert "module" in res.error


def test_required_option_refuses(tmp_path):
    _write(tmp_path, "x = 1\n")
    res = preview(tmp_path, _request("add_import"))
    assert not res.ok
    assert "requires option" in res.error and "module" in res.error


def test_preview_never_writes(tmp_path):
    src = '"""Doc."""\nimport os\n\n\ndef f():\n    print("x")\n'
    _write(tmp_path, src)
    before = _snapshot(tmp_path)
    res = preview(tmp_path, _request("print_to_logger"))
    assert res.ok and res.changed == ("app.py",)
    assert res.diffs["app.py"].startswith("--- a/app.py\n+++ b/app.py\n")
    assert _snapshot(tmp_path) == before


def test_preview_is_deterministic(tmp_path):
    _write(tmp_path, 'name = "x"\nprint(name)\n')
    request = _request("print_to_logger")
    first = preview(tmp_path, request)
    second = preview(tmp_path, request)
    assert first == second


# ---------------------------------------------------------------------------
# add_import
# ---------------------------------------------------------------------------
def test_add_import_after_existing_imports_preserves_comments(tmp_path):
    _write(tmp_path, '"""Doc."""\n\nimport os\n\n# keep this comment\n'
                     'def f():\n    return os.name\n')
    res = preview(tmp_path, _request("add_import", module="typing",
                                     name="Any"))
    assert res.ok, res.error
    assert res.changed == ("app.py",)
    assert "+from typing import Any" in res.diffs["app.py"]
    assert " # keep this comment" in res.diffs["app.py"]


def test_add_import_apply_preserves_docstring_and_comments(tmp_path):
    path = _write(tmp_path, '"""Doc."""\n\nimport os\n\n# keep\nx = os.name\n')
    res = apply(tmp_path, _request("add_import", module="typing",
                                   name="Any"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert text == ('"""Doc."""\n\nimport os\nfrom typing import Any\n\n'
                    '# keep\nx = os.name\n')


def test_add_import_module_and_alias(tmp_path):
    path = _write(tmp_path, "x = 1\n")
    res = apply(tmp_path, _request("add_import", module="os",
                                   alias="operating_system"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "import os as operating_system" in text


def test_add_import_already_present_is_a_noop(tmp_path):
    _write(tmp_path, "from typing import Any\n\nx: Any = 1\n")
    res = preview(tmp_path, _request("add_import", module="typing",
                                     name="Any"))
    assert res.ok
    assert res.changed == () and res.diffs == {}
    assert "already present" in " ".join(res.warnings)


def test_add_import_rebinding_existing_module_name_refuses(tmp_path):
    _write(tmp_path, "def os():\n    return 1\n")
    res = preview(tmp_path, _request("add_import", module="os"))
    assert not res.ok
    assert "rebind" in res.error
    assert res.changed == ()


def test_add_import_rejects_injected_text(tmp_path):
    _write(tmp_path, "x = 1\n")
    for module, name in (("os\nimport evil", ""), ("os; import evil", ""),
                         ("os", "Any\nimport evil")):
        res = preview(tmp_path, _request("add_import", module=module,
                                         name=name))
        assert not res.ok
        assert res.changed == ()


def test_add_import_empty_file(tmp_path):
    path = _write(tmp_path, "")
    res = apply(tmp_path, _request("add_import", module="os"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8") == "import os\n"


def test_add_import_no_trailing_newline(tmp_path):
    path = _write(tmp_path, "x = 1")
    res = apply(tmp_path, _request("add_import", module="os"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8").startswith("import os\n")


def test_add_import_before_a_decorated_first_statement(tmp_path):
    path = _write(tmp_path, "@decorator\ndef f():\n    return 1\n")
    res = apply(tmp_path, _request("add_import", module="os"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8") == (
        "import os\n\n@decorator\ndef f():\n    return 1\n")


def test_add_import_after_import_without_trailing_newline(tmp_path):
    path = _write(tmp_path, "import os")
    res = apply(tmp_path, _request("add_import", module="sys"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8") == "import os\nimport sys\n"


# ---------------------------------------------------------------------------
# remove_import
# ---------------------------------------------------------------------------
def test_remove_unused_import(tmp_path):
    path = _write(tmp_path, "import os\nimport sys\n\nprint(sys.version)\n")
    res = apply(tmp_path, _request("remove_import", module="os"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "import os" not in text
    assert "import sys" in text
    assert "sys.version" in text


def test_remove_import_refuses_when_binding_is_used(tmp_path):
    _write(tmp_path, "import os\n\nprint(os.name)\n")
    res = preview(tmp_path, _request("remove_import", module="os"))
    assert not res.ok
    assert "still used" in res.error
    assert res.changed == ()


def test_remove_one_name_from_from_import_preserves_comment(tmp_path):
    path = _write(tmp_path, "from pkg import a, b  # keep\n\nx = a\n")
    res = apply(tmp_path, _request("remove_import", module="pkg", name="b"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8") == \
        "from pkg import a  # keep\n\nx = a\n"


def test_remove_last_name_removes_the_statement(tmp_path):
    path = _write(tmp_path, "from pkg import a\n\nx = 1\n")
    res = apply(tmp_path, _request("remove_import", module="pkg", name="a"))
    assert res.ok, res.error
    assert "from pkg" not in path.read_text(encoding="utf-8")


def test_remove_one_module_from_multi_import(tmp_path):
    path = _write(tmp_path, "import os, sys\n\nprint(sys.version)\n")
    res = apply(tmp_path, _request("remove_import", module="os"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8").startswith("import sys\n")


def test_remove_import_keeps_trailing_comment(tmp_path):
    path = _write(tmp_path, "import os  # keep me\n\nx = 1\n")
    res = apply(tmp_path, _request("remove_import", module="os"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8") == "# keep me\n\nx = 1\n"


def test_remove_import_ambiguous_statements_refuse(tmp_path):
    _write(tmp_path, "import os\nimport os\n\nx = 1\n")
    res = preview(tmp_path, _request("remove_import", module="os"))
    assert not res.ok
    assert "ambiguous" in res.error


def test_remove_import_alias_disambiguates(tmp_path):
    path = _write(tmp_path, "import os as first\nimport os as second\n\nx = 1\n")
    res = apply(tmp_path, _request("remove_import", module="os",
                                   alias="first"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "first" not in text and "second" in text


def test_remove_from_import_without_name_explains_option(tmp_path):
    _write(tmp_path, "from pkg import a\n\nx = 1\n")
    res = preview(tmp_path, _request("remove_import", module="pkg"))
    assert not res.ok
    assert "name the imported symbol" in res.error


def test_remove_import_at_end_of_file_applies(tmp_path):
    path = _write(tmp_path, "x = 1\ny = 2\nimport os\n")
    shown = preview(tmp_path, _request("remove_import", module="os"))
    assert shown.ok, shown.error
    applied = apply(tmp_path, _request("remove_import", module="os"))
    assert applied.ok, applied.error
    assert applied.diffs == shown.diffs
    assert path.read_text(encoding="utf-8") == "x = 1\ny = 2\n"


def test_remove_import_can_empty_the_file(tmp_path):
    path = _write(tmp_path, "import os\n")
    shown = preview(tmp_path, _request("remove_import", module="os"))
    assert shown.ok, shown.error
    applied = apply(tmp_path, _request("remove_import", module="os"))
    assert applied.ok, applied.error
    assert applied.diffs == shown.diffs
    assert path.read_bytes() == b""


def test_remove_absent_import_is_a_noop(tmp_path):
    path = _write(tmp_path, "x = 1\n")
    request = _request("remove_import", module="os")
    res = apply(tmp_path, request)
    assert res.ok
    assert res.changed == ()
    assert "nothing to remove" in " ".join(res.warnings)
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_remove_import_refuses_exported_name(tmp_path):
    _write(tmp_path, "from pkg import a\n\n__all__ = [\"a\"]\n")
    res = preview(tmp_path, _request("remove_import", module="pkg", name="a"))
    assert not res.ok
    assert "__all__" in res.error


def test_remove_import_refuses_name_appended_to_all(tmp_path):
    _write(tmp_path, "from pkg import a\n\n__all__ = []\n__all__.append(\"a\")\n")
    res = preview(tmp_path, _request("remove_import", module="pkg", name="a"))
    assert not res.ok
    assert "__all__" in res.error


def test_remove_import_multiline_from_import_line(tmp_path):
    path = _write(tmp_path, "from pkg import (\n"
                            "    a,\n"
                            "    b,  # b docs\n"
                            "    c,\n"
                            ")\n\nx = a + c\n")
    res = apply(tmp_path, _request("remove_import", module="pkg", name="b"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8") == (
        "from pkg import (\n"
        "    a,\n"
        "    # b docs\n"
        "    c,\n"
        ")\n\nx = a + c\n")


# ---------------------------------------------------------------------------
# print_to_logger
# ---------------------------------------------------------------------------
def test_print_to_logger_single_argument_preserves_comment(tmp_path):
    path = _write(tmp_path, "logger = object()\n\n"
                            "def f(x):\n    print(x)  # trace\n")
    res = apply(tmp_path, _request("print_to_logger", symbol="f"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "logger.info(x)  # trace" in text
    assert "print(" not in text
    assert "import logging" not in text


def test_print_to_logger_multiple_arguments_use_lazy_format(tmp_path):
    path = _write(tmp_path, "logger = object()\n\nprint(a, b)\n")
    res = apply(tmp_path, _request("print_to_logger"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8").endswith(
        'logger.info("%s %s", a, b)\n')


def test_print_to_logger_zero_arguments(tmp_path):
    path = _write(tmp_path, "logger = object()\n\nprint()\n")
    res = apply(tmp_path, _request("print_to_logger"))
    assert res.ok, res.error
    assert 'logger.info("")' in path.read_text(encoding="utf-8")


def test_print_to_logger_without_calls_is_a_noop(tmp_path):
    path = _write(tmp_path, "x = 1\n")
    res = apply(tmp_path, _request("print_to_logger"))
    assert res.ok
    assert res.changed == ()
    assert "no print()" in " ".join(res.warnings)
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_print_to_logger_creates_logger_when_missing(tmp_path):
    path = _write(tmp_path, '"""Doc."""\n\n\ndef f():\n    print("hello")\n')
    res = apply(tmp_path, _request("print_to_logger"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert text.startswith('"""Doc."""\n\nimport logging\n'
                           'logger = logging.getLogger(__name__)\n')
    assert 'logger.info("hello")' in text


def test_print_to_logger_handles_an_aliased_logging_import(tmp_path):
    path = _write(tmp_path, "import logging as log\n\nprint(1)\n")
    res = apply(tmp_path, _request("print_to_logger"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "import logging as log\nimport logging\n" in text
    assert "logger = logging.getLogger(__name__)" in text
    assert "logger.info(1)" in text


def test_print_to_logger_scope_leaves_other_calls(tmp_path):
    path = _write(tmp_path, "logger = object()\n\n"
                            "def a():\n    print(1)\n\n"
                            "def b():\n    print(2)\n")
    res = apply(tmp_path, _request("print_to_logger", symbol="a"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "logger.info(1)" in text
    assert "print(2)" in text


def test_print_to_logger_accepts_a_local_logger_binding(tmp_path):
    path = _write(tmp_path, "def f(logger):\n    print(1)\n")
    res = apply(tmp_path, _request("print_to_logger", symbol="f"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "logger.info(1)" in text
    assert "import logging" not in text


def test_print_to_logger_refuses_ambiguous_symbol(tmp_path):
    _write(tmp_path, "logger = object()\n\n"
                     "def f():\n    print(1)\n\n"
                     "def f():\n    print(2)\n")
    res = preview(tmp_path, _request("print_to_logger", symbol="f"))
    assert not res.ok
    assert "ambiguous" in res.error


def test_print_to_logger_refuses_shadowed_print(tmp_path):
    _write(tmp_path, "def print(*args):\n    pass\n\nprint(1)\n")
    res = preview(tmp_path, _request("print_to_logger"))
    assert not res.ok
    assert "rebound" in res.error


def test_print_to_logger_refuses_keyword_arguments(tmp_path):
    _write(tmp_path, "logger = object()\n\nprint(1, sep=\"\")\n")
    res = preview(tmp_path, _request("print_to_logger"))
    assert not res.ok
    assert "keyword" in res.error


def test_print_to_logger_refuses_star_args(tmp_path):
    _write(tmp_path, "logger = object()\n\nprint(*items)\n")
    res = preview(tmp_path, _request("print_to_logger"))
    assert not res.ok
    assert "*args" in res.error


def test_print_to_logger_refuses_unknown_level(tmp_path):
    _write(tmp_path, "logger = object()\n\nprint(1)\n")
    res = preview(tmp_path, _request("print_to_logger", level="loud"))
    assert not res.ok
    assert "level" in res.error


def test_print_to_logger_unicode_before_call(tmp_path):
    path = _write(tmp_path, 'logger = object()\n\nx = "é"; print(x)\n')
    res = apply(tmp_path, _request("print_to_logger"))
    assert res.ok, res.error
    assert 'x = "é"; logger.info(x)' in path.read_text(encoding="utf-8")


def test_print_to_logger_does_not_touch_strings_or_comments(tmp_path):
    path = _write(tmp_path, 'logger = object()\n\n'
                            '# print(1) in a comment\n'
                            'text = "print(2)"\n'
                            'print(3)\n')
    res = apply(tmp_path, _request("print_to_logger"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "# print(1) in a comment" in text
    assert 'text = "print(2)"' in text
    assert "logger.info(3)" in text


# ---------------------------------------------------------------------------
# extract_constant
# ---------------------------------------------------------------------------
def test_extract_constant_replaces_all_matching_literals(tmp_path):
    path = _write(tmp_path, "def f():\n    return \"timeout\"\n\n"
                            "def g():\n    return \"timeout\"\n")
    res = apply(tmp_path, _request("extract_constant", name="TIMEOUT",
                                   value='"timeout"'))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert text.startswith('TIMEOUT = "timeout"\n\n')
    assert text.count("return TIMEOUT") == 2
    assert 'return "timeout"' not in text


def test_extract_constant_preserves_comments(tmp_path):
    path = _write(tmp_path, "x = 42  # answer\n")
    res = apply(tmp_path, _request("extract_constant", name="ANSWER",
                                   value="42"))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert "ANSWER = 42" in text
    assert "x = ANSWER  # answer" in text


def test_extract_constant_inserts_before_decorators(tmp_path):
    path = _write(tmp_path, "@decorator\ndef f():\n    return \"x\"\n")
    res = apply(tmp_path, _request("extract_constant", name="X",
                                   value='"x"'))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert text.startswith('X = "x"\n\n@decorator\n')
    assert "return X" in text


def test_extract_constant_scope_limits_occurrences(tmp_path):
    path = _write(tmp_path, "def a():\n    return \"x\"\n\n"
                            "def b():\n    return \"x\"\n")
    res = apply(tmp_path, _request("extract_constant", symbol="a",
                                   name="X", value='"x"'))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert text.count("return X") == 1
    assert text.count('return "x"') == 1


def test_extract_constant_missing_literal_is_a_noop(tmp_path):
    path = _write(tmp_path, "x = 1\n")
    res = preview(tmp_path, _request("extract_constant", name="Y",
                                     value='"missing"'))
    assert res.ok
    assert res.changed == () and res.diffs == {}
    assert "found no occurrence" in " ".join(res.warnings)
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_extract_constant_refuses_existing_binding(tmp_path):
    _write(tmp_path, "TIMEOUT = 1\nx = \"timeout\"\n")
    res = preview(tmp_path, _request("extract_constant", name="TIMEOUT",
                                     value='"timeout"'))
    assert not res.ok
    assert "already exists with a different value" in res.error


def test_extract_constant_skips_docstrings_and_annotations(tmp_path):
    path = _write(tmp_path, '"""T"""\n\n'
                            'def f(x: "T") -> "T":\n'
                            '    """T"""\n'
                            '    return "T"\n')
    res = apply(tmp_path, _request("extract_constant", name="T_VALUE",
                                   value='"T"'))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert text.count('"""T"""') == 2
    assert 'x: "T"' in text and '-> "T":' in text
    assert "return T_VALUE" in text


def test_extract_constant_skips_fstrings_and_match_patterns(tmp_path):
    _write(tmp_path, 'def f(x):\n    return f"{x}"\n')
    res = preview(tmp_path, _request("extract_constant", name="VALUE",
                                     value='"x"'))
    assert res.ok
    assert res.changed == ()
    assert "found no occurrence" in " ".join(res.warnings)


def test_extract_constant_negative_number_semantics(tmp_path):
    _write(tmp_path, "a = -1\nb = 1\n")
    res = preview(tmp_path, _request("extract_constant", name="ONE",
                                     value="1"))
    assert res.ok, res.error
    diff = res.diffs["app.py"]
    assert "-b = 1" in diff
    assert "-a = -1" not in diff
    assert "+a = -1" not in diff


def test_extract_constant_unicode_offsets(tmp_path):
    path = _write(tmp_path, 'label = "café"\nvalue = "café"\n')
    res = apply(tmp_path, _request("extract_constant", name="LABEL",
                                   value='"café"'))
    assert res.ok, res.error
    text = path.read_text(encoding="utf-8")
    assert 'LABEL = "café"' in text
    assert 'label = LABEL' in text
    assert 'value = LABEL' in text


def test_extract_constant_refuses_non_literal_value(tmp_path):
    _write(tmp_path, "x = 1\n")
    res = preview(tmp_path, _request("extract_constant", name="V",
                                     value="compute()"))
    assert not res.ok
    assert "not a literal" in res.error


def test_extract_constant_reuses_an_existing_equal_constant(tmp_path):
    path = _write(tmp_path, "ANSWER = 42\ny = 42\n")
    res = apply(tmp_path, _request("extract_constant", name="ANSWER",
                                   value="42"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8") == "ANSWER = 42\ny = ANSWER\n"
    assert "reused" in " ".join(res.warnings)


def test_extract_constant_is_idempotent_after_success(tmp_path):
    path = _write(tmp_path, "x = 42\n")
    request = _request("extract_constant", name="ANSWER", value="42")
    first = apply(tmp_path, request)
    assert first.ok, first.error
    second = apply(tmp_path, request)
    assert second.ok
    assert second.changed == ()
    assert path.read_text(encoding="utf-8") == "ANSWER = 42\n\nx = ANSWER\n"


def test_extract_constant_skips_reexport_names(tmp_path):
    path = _write(tmp_path, "from pkg import a\n\n__all__ = [\"a\"]\n")
    res = preview(tmp_path, _request("extract_constant", name="EXPORT",
                                     value='"a"'))
    assert res.ok
    assert res.changed == ()
    assert "found no occurrence" in " ".join(res.warnings)
    assert path.read_text(encoding="utf-8").count('"a"') == 1


# ---------------------------------------------------------------------------
# add_annotation
# ---------------------------------------------------------------------------
def test_add_annotation_parameter_with_name_option(tmp_path):
    path = _write(tmp_path, "def greet(name):\n    return name\n")
    res = apply(tmp_path, _request("add_annotation", symbol="greet",
                                   annotation="str", target="param",
                                   name="name"))
    assert res.ok, res.error
    assert "def greet(name: str):" in path.read_text(encoding="utf-8")


def test_add_annotation_parameter_with_dotted_symbol(tmp_path):
    path = _write(tmp_path, "class Greeter:\n"
                            "    def greet(self, name):\n"
                            "        return name\n")
    res = apply(tmp_path, _request("add_annotation",
                                   symbol="Greeter.greet.name",
                                   annotation="str", target="param"))
    assert res.ok, res.error
    assert "def greet(self, name: str):" in path.read_text(encoding="utf-8")


def test_add_annotation_return(tmp_path):
    path = _write(tmp_path, "def f(x):\n    return x\n")
    res = apply(tmp_path, _request("add_annotation", symbol="f",
                                   annotation="int", target="return"))
    assert res.ok, res.error
    assert "def f(x) -> int:" in path.read_text(encoding="utf-8")


def test_add_annotation_return_on_one_line_function(tmp_path):
    path = _write(tmp_path, "def f(x): return x\n")
    res = apply(tmp_path, _request("add_annotation", symbol="f",
                                   annotation="int", target="return"))
    assert res.ok, res.error
    assert "def f(x) -> int: return x" in path.read_text(encoding="utf-8")


def test_add_annotation_async_return_with_unicode_default(tmp_path):
    path = _write(tmp_path, 'async def f(x="é"):\n    return x\n')
    res = apply(tmp_path, _request("add_annotation", symbol="f",
                                   annotation="str", target="return"))
    assert res.ok, res.error
    assert 'async def f(x="é") -> str:' in path.read_text(encoding="utf-8")


def test_add_annotation_variable_module_scope(tmp_path):
    path = _write(tmp_path, "x = 1\n")
    res = apply(tmp_path, _request("add_annotation", symbol="x",
                                   annotation="int", target="variable"))
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8") == "x: int = 1\n"


def test_add_annotation_variable_function_scope(tmp_path):
    path = _write(tmp_path, "def f():\n    y = 2\n    return y\n")
    res = apply(tmp_path, _request("add_annotation", symbol="f.y",
                                   annotation="int", target="variable"))
    assert res.ok, res.error
    assert "y: int = 2" in path.read_text(encoding="utf-8")


def test_add_annotation_variable_class_attribute(tmp_path):
    path = _write(tmp_path, "class C:\n    value = 1\n")
    res = apply(tmp_path, _request("add_annotation", symbol="C.value",
                                   annotation="int", target="variable"))
    assert res.ok, res.error
    assert "value: int = 1" in path.read_text(encoding="utf-8")


def test_add_annotation_existing_same_is_noop(tmp_path):
    _write(tmp_path, "def f(x: int) -> int:\n    return x\n")
    res = preview(tmp_path, _request("add_annotation", symbol="f",
                                     annotation="int", target="return"))
    assert res.ok
    assert res.changed == ()
    assert "already" in " ".join(res.warnings)


def test_add_annotation_different_existing_refuses_without_overwrite(tmp_path):
    _write(tmp_path, "def f(x: float) -> float:\n    return x\n")
    refused = preview(tmp_path, _request("add_annotation", symbol="f",
                                         annotation="int", target="return"))
    assert not refused.ok
    assert "overwrite=true" in refused.error


def test_add_annotation_overwrite_replaces(tmp_path):
    path = _write(tmp_path, "def f(x: float) -> float:\n    return x\n")
    res = apply(tmp_path, _request("add_annotation", symbol="f",
                                   annotation="int", target="return",
                                   overwrite="true"))
    assert res.ok, res.error
    assert "-> int:" in path.read_text(encoding="utf-8")


def test_add_annotation_missing_parameter_refuses(tmp_path):
    _write(tmp_path, "def f(x):\n    return x\n")
    res = preview(tmp_path, _request("add_annotation", symbol="f",
                                     annotation="int", target="param",
                                     name="missing"))
    assert not res.ok
    assert "not a parameter" in res.error


def test_add_annotation_invalid_annotation_refuses(tmp_path):
    _write(tmp_path, "def f(x):\n    return x\n")
    res = preview(tmp_path, _request("add_annotation", symbol="f",
                                     annotation="not valid !!",
                                     target="return"))
    assert not res.ok
    assert "annotation" in res.error


def test_add_annotation_ambiguous_symbol_refuses(tmp_path):
    _write(tmp_path, "def f(x):\n    return x\n\n"
                     "def f(x):\n    return x\n")
    res = preview(tmp_path, _request("add_annotation", symbol="f",
                                     annotation="int", target="return"))
    assert not res.ok
    assert "ambiguous" in res.error


def test_add_annotation_unicode_preceding_parameter(tmp_path):
    path = _write(tmp_path, "def f(é, x):\n    return x\n")
    res = apply(tmp_path, _request("add_annotation", symbol="f",
                                   annotation="int", target="param",
                                   name="x"))
    assert res.ok, res.error
    assert "def f(é, x: int):" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Refusals: language, path, policy, syntax
# ---------------------------------------------------------------------------
def test_unsupported_language_refuses_without_writing(tmp_path):
    path = _write(tmp_path, "const x = 1;\n", name="app.js")
    before = path.read_bytes()
    res = preview(tmp_path, _request("add_import", file="app.js",
                                     module="os"))
    assert not res.ok
    assert "supports only Python" in res.error
    assert path.read_bytes() == before


def test_syntax_error_input_refuses(tmp_path):
    _write(tmp_path, "def broken(:\n")
    res = preview(tmp_path, _request("add_import", module="os"))
    assert not res.ok
    assert "does not parse" in res.error


def test_invalid_recipe_output_is_refused(tmp_path, monkeypatch):
    path = _write(tmp_path, "x = 1\n")

    def broken(source, symbol, options):
        return codemod_module._Outcome("def broken(:\n")

    monkeypatch.setitem(codemod_module._HANDLERS, "add_import", broken)
    res = preview(tmp_path, _request("add_import", module="os"))
    assert not res.ok
    assert "syntax error" in res.error
    assert res.changed == ()
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_missing_file_and_directory_refuse(tmp_path):
    missing = preview(tmp_path, _request("add_import", file="missing.py",
                                         module="os"))
    assert not missing.ok
    assert "not a regular file" in missing.error
    (tmp_path / "pkg.py").mkdir()
    directory = preview(tmp_path, _request("add_import", file="pkg.py",
                                           module="os"))
    assert not directory.ok
    assert "not a regular file" in directory.error


def test_path_traversal_refuses(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    res = preview(tmp_path, _request("add_import", file="../outside.py",
                                     module="os"))
    assert not res.ok
    assert "traversal" in res.error or "escapes" in res.error
    assert outside.read_text(encoding="utf-8") == "x = 1\n"


def test_protected_and_sensitive_paths_refuse(tmp_path):
    protected = _write(tmp_path, "x = 1\n", name=".pedacito/state.py")
    res = preview(tmp_path, _request("add_import", file=".pedacito/state.py",
                                     module="os"))
    assert not res.ok
    assert "refused" in res.error
    assert protected.read_text(encoding="utf-8") == "x = 1\n"

    sensitive = _write(tmp_path, "x = 1\n", name=".env.production")
    res = preview(tmp_path, _request("add_import", file=".env.production",
                                     module="os"))
    assert not res.ok
    assert "sensitive" in res.error
    assert sensitive.read_text(encoding="utf-8") == "x = 1\n"


def test_symlink_escape_refuses(tmp_path):
    outside = tmp_path.parent / "outside-target.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    link = tmp_path / "link.py"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    res = preview(tmp_path, _request("add_import", file="link.py",
                                     module="os"))
    assert not res.ok
    assert "escapes" in res.error or "refused" in res.error
    assert outside.read_text(encoding="utf-8") == "x = 1\n"


# ---------------------------------------------------------------------------
# Apply: permission, snapshot, atomicity, encoding
# ---------------------------------------------------------------------------
def test_apply_requires_allow_write_when_cfg_is_given(tmp_path):
    path = _write(tmp_path, "x = 1\n")
    cfg = Config()
    res = apply(tmp_path, _request("add_import", module="os"), cfg=cfg)
    assert not res.ok
    assert "allow_write" in res.error
    assert path.read_text(encoding="utf-8") == "x = 1\n"


class _DenyPolicy:
    def evaluate(self, request):
        class Decision:
            decision = "deny"
            matched_rule = "test-deny"
            reason = "codemods are denied here"
        return Decision()


def test_policy_denial_refuses_preview_and_apply(tmp_path):
    path = _write(tmp_path, "x = 1\n")
    cfg = Config()
    cfg.allow_write = True
    cfg.policy = _DenyPolicy()
    request = _request("add_import", module="os")
    shown = preview(tmp_path, request, cfg=cfg)
    assert not shown.ok
    assert "denied by policy" in shown.error
    applied = apply(tmp_path, request, cfg=cfg)
    assert not applied.ok
    assert "denied by policy" in applied.error
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_apply_with_allow_write_snapshots_and_changes(tmp_path):
    path = _write(tmp_path, "x = 1\n")
    cfg = Config()
    cfg.allow_write = True
    res = apply(tmp_path, _request("add_annotation", symbol="x",
                                   annotation="int", target="variable"),
                cfg=cfg)
    assert res.ok, res.error
    assert path.read_text(encoding="utf-8") == "x: int = 1\n"
    snapshots = sorted((tmp_path / ".pedacito" / "backups").glob("*/app.py"))
    assert snapshots
    assert snapshots[0].read_text(encoding="utf-8") == "x = 1\n"


def test_apply_is_idempotent_after_success(tmp_path):
    path = _write(tmp_path, "x = 1\n")
    request = _request("add_annotation", symbol="x", annotation="int",
                       target="variable")
    first = apply(tmp_path, request)
    assert first.ok, first.error
    second = apply(tmp_path, request)
    assert second.ok
    assert second.changed == () and second.diffs == {}
    assert path.read_text(encoding="utf-8") == "x: int = 1\n"


def test_apply_write_failure_rolls_back(tmp_path, monkeypatch):
    path = _write(tmp_path, "x = 1\n")

    def boom(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr("pedacito.fileio.write", boom)
    res = apply(tmp_path, _request("add_annotation", symbol="x",
                                   annotation="int", target="variable"))
    assert not res.ok
    assert "write" in res.error
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_apply_preserves_crlf_and_bom(tmp_path):
    path = tmp_path / "app.py"
    path.write_bytes(b"\xef\xbb\xbfx = 1\r\n")
    res = apply(tmp_path, _request("add_annotation", symbol="x",
                                   annotation="int", target="variable"))
    assert res.ok, res.error
    data = path.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf")
    assert data.endswith(b"x: int = 1\r\n")
    assert b"\n" not in data.replace(b"\r\n", b"")


def test_apply_refuses_stale_preimage_hash(tmp_path):
    path = _write(tmp_path, "alpha\n")
    edit = Edit("app.py", "alpha\n", "beta\n")
    stale = hashlib.sha256(b"not the current file\n").hexdigest()
    results = apply_edits_atomic(tmp_path, [edit],
                                 expected_sha256={"app.py": stale})
    assert not results[0].ok
    assert "changed after the preview" in results[0].message
    assert path.read_text(encoding="utf-8") == "alpha\n"


def test_refactor_facade_delegates(tmp_path):
    from pedacito.refactor import codemod_apply, codemod_preview

    path = _write(tmp_path, "x = 1\n")
    request = _request("add_import", module="os")
    shown = codemod_preview(tmp_path, request)
    assert shown.ok and shown.changed == ("app.py",)
    applied = codemod_apply(tmp_path, request)
    assert applied.ok, applied.error
    assert path.read_text(encoding="utf-8").startswith("import os\n")
