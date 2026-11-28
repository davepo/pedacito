"""WP-4101A: the strict structured-edit parser, guard and dry-run materializer.

Covers the five actions (replace_symbol / insert_after_symbol / search_replace
/ append / delete_symbol), strict JSON schema with ``additionalProperties:
false``, exact (file, symbol) intent matching, unique symbol/search resolution,
read-scope and allowlist guards, no-op/overlap/whole-file refusal, Unicode and
decorators, CRLF/BOM preservation, and the guarantee that validation never
mutates the filesystem.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pedacito import fileio
from pedacito.chunker import file_symbol_spans, locate_symbol_span
from pedacito.config import Config
from pedacito.edits import (
    Edit,
    StructuredEdit,
    StructuredIntent,
    StructuredReplyError,
    materialize_structured_edits,
    parse_structured_reply,
)
from pedacito.verify import validate_structured_reply

SRC = (
    "def alpha():\n"
    "    return 1\n"
    "\n"
    "\n"
    "def beta():\n"
    "    return 2\n"
)


def _write(tmp_path: Path, text: str = SRC, name: str = "app.py",
           *, newline: str | None = None, bom: bool = False) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if newline == "\r\n":
        data = data.replace(b"\n", b"\r\n")
    if bom:
        data = b"\xef\xbb\xbf" + data
    p.write_bytes(data)
    return p


def _reply(intents, edits, version: int = 1) -> str:
    return json.dumps({"version": version, "intents": intents,
                       "edits": edits})


def _intent(file="app.py", symbol="beta", why="fix it"):
    return {"file": file, "symbol": symbol, "why": why}


def _edit(action, file="app.py", symbol="beta", search="", replacement=""):
    return {"edit_action": action, "file": file, "symbol": symbol,
            "search": search, "replacement": replacement}


def _all(tmp_path):
    return tmp_path, {"app.py"}, {"app.py": [(1, 7)]}


# ------------------------------------------------------------------ parsing
def test_parse_no_change_reply():
    intents, edits = parse_structured_reply(
        '{"version":1,"intents":[],"edits":[]}')
    assert intents == [] and edits == []


def test_parse_round_trips_dataclasses():
    intents, edits = parse_structured_reply(_reply(
        [_intent()], [_edit("replace_symbol", replacement="def beta():\n    return 9")]))
    assert intents == [StructuredIntent("app.py", "beta", "fix it")]
    assert edits == [StructuredEdit("replace_symbol", "app.py", "beta", "",
                                    "def beta():\n    return 9")]
    assert isinstance(intents[0], StructuredIntent)
    assert isinstance(edits[0], StructuredEdit)


def test_parse_strips_a_wrapping_fence():
    text = "```json\n" + _reply([_intent()], [_edit("delete_symbol")]) + "\n```"
    intents, edits = parse_structured_reply(text)
    assert len(intents) == 1 and len(edits) == 1


@pytest.mark.parametrize("bad", [
    "", "not json", "[1, 2, 3]", '{"version":1}', "null", '"str"',
])
def test_parse_rejects_non_object(bad):
    with pytest.raises(StructuredReplyError):
        parse_structured_reply(bad)


def test_parse_rejects_salvage_of_embedded_json():
    # prose wrapping the object is not a strict reply: no salvage
    with pytest.raises(StructuredReplyError):
        parse_structured_reply("Sure, here:\n" + _reply([_intent()],
                                                        [_edit("delete_symbol")]))


def test_parse_rejects_unknown_top_level_field():
    raw = json.dumps({"version": 1, "intents": [], "edits": [], "extra": 1})
    with pytest.raises(StructuredReplyError) as exc:
        parse_structured_reply(raw)
    assert "unknown field" in str(exc.value)


def test_parse_rejects_missing_required_field():
    with pytest.raises(StructuredReplyError):
        parse_structured_reply('{"version":1,"intents":[]}')


@pytest.mark.parametrize("version", [0, 2, "1", 1.0, None, True])
def test_parse_rejects_bad_version(version):
    with pytest.raises(StructuredReplyError):
        parse_structured_reply(_reply([], [], version=version))


def test_parse_rejects_unknown_edit_action():
    with pytest.raises(StructuredReplyError) as exc:
        parse_structured_reply(_reply([_intent()], [_edit("rewrite_file")]))
    assert "unknown edit_action" in str(exc.value)


def test_parse_rejects_unknown_intent_field():
    raw = json.dumps({"version": 1,
                      "intents": [{"file": "a.py", "symbol": "x", "why": "w",
                                   "bogus": 1}],
                      "edits": []})
    with pytest.raises(StructuredReplyError) as exc:
        parse_structured_reply(raw)
    assert "unknown field" in str(exc.value)


def test_parse_rejects_blank_why():
    with pytest.raises(StructuredReplyError) as exc:
        parse_structured_reply(_reply([_intent(why="   ")], [_edit("delete_symbol")]))
    assert "nonblank" in str(exc.value)


def test_parse_rejects_why_over_300_chars():
    with pytest.raises(StructuredReplyError) as exc:
        parse_structured_reply(_reply([_intent(why="x" * 301)],
                                      [_edit("delete_symbol")]))
    assert "capped at 300" in str(exc.value)


def test_parse_rejects_too_many_entries():
    intents = [_intent(symbol=f"s{i}") for i in range(9)]
    with pytest.raises(StructuredReplyError) as exc:
        parse_structured_reply(_reply(intents, []))
    assert "at most 8" in str(exc.value)


def test_parse_rejects_edit_missing_field():
    raw = json.dumps({"version": 1,
                      "intents": [_intent()],
                      "edits": [{"edit_action": "replace_symbol",
                                 "file": "app.py", "symbol": "beta"}]})
    with pytest.raises(StructuredReplyError):
        parse_structured_reply(raw)


def test_parse_rejects_non_string_field():
    raw = json.dumps({"version": 1, "intents": [_intent()],
                      "edits": [{"edit_action": "replace_symbol", "file": 3,
                                 "symbol": "beta", "search": "",
                                 "replacement": ""}]})
    with pytest.raises(StructuredReplyError):
        parse_structured_reply(raw)


# ---------------------------------------------------------- field constraints
def test_replace_symbol_requires_nonblank_replacement(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "fix")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "", "")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files=allowed, read_spans=read,
                                     cfg=Config())
    assert any("delete_symbol" in e for e in errs)


def test_append_requires_empty_symbol(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "add")]
    edits = [StructuredEdit("append", "app.py", "beta", "", "x = 1\n")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files=allowed, read_spans=read,
                                     cfg=Config())
    assert any("symbol' must be empty" in e for e in errs)


def test_delete_symbol_requires_empty_replacement(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "remove")]
    edits = [StructuredEdit("delete_symbol", "app.py", "beta", "", "oops")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files=allowed, read_spans=read,
                                     cfg=Config())
    assert any("replacement' must be empty" in e for e in errs)


# --------------------------------------------------------------- intent match
def test_edit_must_match_a_declared_intent(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "alpha", "fix")]   # wrong symbol
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 9")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files=allowed, read_spans=read,
                                     cfg=Config())
    assert any("does not match any declared intent" in e for e in errs)


def test_intent_and_edits_must_be_consistent(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    errs = validate_structured_reply(root, [StructuredIntent("app.py", "beta", "x")],
                                     [], allowed_files=allowed,
                                     read_spans=read, cfg=Config())
    assert any("both be empty" in e for e in errs)


# ----------------------------------------------------------------- apply paths
def test_replace_symbol_applies(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "fix return")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 22")]
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files=allowed, read_spans=read,
                                       cfg=Config())
    assert mat.ok
    assert "return 22" in mat.buffers["app.py"]
    assert "return 1" in mat.buffers["app.py"]          # alpha untouched
    # no filesystem mutation
    assert (root / "app.py").read_text(encoding="utf-8") == SRC


def test_multiple_actions_compose_sequentially(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "alpha", "fix return"),
               StructuredIntent("app.py", "beta", "fix return")]
    edits = [StructuredEdit("search_replace", "app.py", "alpha", "return 1",
                            "return 11"),
             StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 22")]
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files=allowed, read_spans=read,
                                       cfg=Config())
    assert mat.ok
    buf = mat.buffers["app.py"]
    assert "return 11" in buf and "return 22" in buf


def test_insert_after_symbol(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "add a helper after")]
    edits = [StructuredEdit("insert_after_symbol", "app.py", "beta", "",
                            "def gamma():\n    return 3")]
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files=allowed, read_spans=read,
                                       cfg=Config())
    assert mat.ok
    buf = mat.buffers["app.py"]
    assert buf.index("def beta():") < buf.index("def gamma():")


def test_search_replace_within_symbol(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "change the constant")]
    edits = [StructuredEdit("search_replace", "app.py", "beta", "return 2",
                            "return 42")]
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files=allowed, read_spans=read,
                                       cfg=Config())
    assert mat.ok
    assert "return 42" in mat.buffers["app.py"]
    assert "return 2" not in mat.buffers["app.py"]


def test_search_replace_must_be_unique_within_symbol(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root, "def foo():\n    return 1\n    return 1\n")
    intents = [StructuredIntent("app.py", "foo", "change one")]
    edits = [StructuredEdit("search_replace", "app.py", "foo", "return 1",
                            "return 2")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files={"app.py"},
                                     read_spans={"app.py": [(1, 3)]},
                                     cfg=Config())
    assert any("exactly once" in e for e in errs)


def test_search_replace_search_not_in_symbol_is_rejected(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    # the whole file (an accidental whole-file SEARCH quote) is not inside beta
    intents = [StructuredIntent("app.py", "beta", "fix")]
    edits = [StructuredEdit("search_replace", "app.py", "beta", SRC, "x")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files=allowed, read_spans=read,
                                     cfg=Config())
    assert any("exactly once" in e for e in errs)


def test_delete_symbol_removes_the_whole_definition(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "remove it")]
    edits = [StructuredEdit("delete_symbol", "app.py", "beta", "", "")]
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files=allowed, read_spans=read,
                                       cfg=Config())
    assert mat.ok
    assert "def beta" not in mat.buffers["app.py"]
    assert "def alpha" in mat.buffers["app.py"]


def test_append_to_end_of_file(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "", "add a main guard")]
    edits = [StructuredEdit("append", "app.py", "", "",
                            "if __name__ == '__main__':\n    alpha()")]
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files=allowed, read_spans=read,
                                       cfg=Config())
    assert mat.ok
    buf = mat.buffers["app.py"]
    assert buf.rstrip().endswith("alpha()")
    assert buf.startswith("def alpha():")


def test_append_creates_an_authorized_new_file(tmp_path):
    root = tmp_path
    intents = [StructuredIntent("new.py", "", "create it")]
    edits = [StructuredEdit("append", "new.py", "", "", "x = 1")]
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files={"new.py"},
                                       read_spans={}, cfg=Config())
    assert mat.ok
    assert mat.buffers["new.py"].rstrip() == "x = 1"
    assert not (root / "new.py").exists()               # still not written


def test_new_file_requires_authorization(tmp_path):
    root = tmp_path
    intents = [StructuredIntent("new.py", "", "create it")]
    edits = [StructuredEdit("append", "new.py", "", "", "x = 1")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files={"other.py"},
                                     read_spans={}, cfg=Config())
    assert any("authorized file set" in e for e in errs)


def test_non_append_action_on_missing_file_is_rejected(tmp_path):
    root = tmp_path
    intents = [StructuredIntent("new.py", "beta", "fix it")]
    edits = [StructuredEdit("replace_symbol", "new.py", "beta", "",
                            "def beta():\n    return 1")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files={"new.py"},
                                     read_spans={"new.py": [(1, 2)]},
                                     cfg=Config())
    assert any("does not exist yet" in e for e in errs)


# ------------------------------------------------------------------- guards
def test_unapproved_existing_file_is_rejected(tmp_path):
    root = tmp_path
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "fix")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 9")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files={"unrelated.py"},
                                     read_spans={"app.py": [(1, 7)]},
                                     cfg=Config())
    assert any("authorized file set" in e for e in errs)


def test_path_traversal_is_rejected(tmp_path):
    root = tmp_path
    _write(root)
    intents = [StructuredIntent("../secret.py", "beta", "fix")]
    edits = [StructuredEdit("replace_symbol", "../secret.py", "beta", "",
                            "def beta():\n    return 9")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files={"../secret.py"},
                                     read_spans={"../secret.py": [(1, 7)]},
                                     cfg=Config())
    assert any("traversal" in e or "escapes" in e or ".." in e for e in errs)


def test_sensitive_file_is_rejected(tmp_path):
    root = tmp_path
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    intents = [StructuredIntent(".env", "", "add")]
    edits = [StructuredEdit("append", ".env", "", "", "MORE=2")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files={".env"},
                                     read_spans={".env": [(1, 1)]},
                                     cfg=Config())
    assert any("sensitive" in e for e in errs)


def test_unread_symbol_is_rejected(tmp_path):
    root = tmp_path
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "fix")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 9")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files={"app.py"},
                                     read_spans={"app.py": [(1, 2)]},  # only alpha
                                     cfg=Config())
    assert any("outside the lines read" in e for e in errs)


def test_never_read_file_is_rejected(tmp_path):
    root = tmp_path
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "fix")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 9")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files={"app.py"},
                                     read_spans={"app.py": []}, cfg=Config())
    assert any("never read" in e for e in errs)


def test_overlapping_edits_are_rejected(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "fix"),
               StructuredIntent("app.py", "beta", "fix again")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 22"),
             StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 33")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files=allowed, read_spans=read,
                                     cfg=Config())
    assert any("overlap" in e for e in errs)


def test_noop_edits_are_rejected(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "no change")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 2")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files=allowed, read_spans=read,
                                     cfg=Config())
    assert any("no-op" in e for e in errs)


def test_syntax_error_is_rejected(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "break it")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta(:\n    return 2")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files=allowed, read_spans=read,
                                     cfg=Config())
    assert any("syntax" in e for e in errs)


def test_static_undefined_name_is_rejected(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "use a helper")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return does_not_exist()")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files=allowed, read_spans=read,
                                     cfg=Config())
    assert any("do not exist" in e or "F821" in e for e in errs)


# ------------------------------------------------- Unicode, decorators, whole-file
def test_unicode_content_round_trips(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root, 'def beta():\n    return "caf\u00e9 \u4f60\u597d"\n')
    intents = [StructuredIntent("app.py", "beta", "translate")]
    edits = [StructuredEdit("search_replace", "app.py", "beta",
                            'caf\u00e9 \u4f60\u597d', "hello world")]
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files=allowed, read_spans=read,
                                       cfg=Config())
    assert mat.ok
    assert "hello world" in mat.buffers["app.py"]


def test_replace_symbol_preserves_decorators(tmp_path):
    root = tmp_path
    _write(root, "@decorator\ndef beta():\n    return 2\n")
    intents = [StructuredIntent("app.py", "beta", "fix")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 99")]
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files={"app.py"},
                                       read_spans={"app.py": [(1, 3)]},
                                       cfg=Config())
    assert mat.ok
    assert "@decorator" in mat.buffers["app.py"]          # decorator carried over


def test_whole_file_replace_via_empty_symbol_is_rejected(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "", "rewrite the file")]
    edits = [StructuredEdit("replace_symbol", "app.py", "", "", "x = 1")]
    errs = validate_structured_reply(root, intents, edits,
                                     allowed_files=allowed, read_spans=read,
                                     cfg=Config())
    assert any("symbol' must name a symbol" in e for e in errs)


# ----------------------------------------------------- CRLF/BOM preservation
def test_crlf_and_bom_style_is_preserved(tmp_path):
    root = tmp_path
    _write(root, SRC, newline="\r\n", bom=True)
    intents = [StructuredIntent("app.py", "beta", "fix")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 22")]
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files={"app.py"},
                                       read_spans={"app.py": [(1, 7)]},
                                       cfg=Config())
    assert mat.ok
    style = mat.styles["app.py"]
    assert style.newline == "\r\n" and style.bom is True
    assert "\r" not in mat.buffers["app.py"]              # LF-normalised buffer
    # writing the buffer back with its style reproduces the original encoding
    fileio.write(root / "app.py", mat.buffers["app.py"], style)
    raw = (root / "app.py").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in raw
    assert b"return 22" in raw


# --------------------------------------------------------------- chunker API
def test_file_symbol_spans_lists_definitions_and_decorators():
    spans = file_symbol_spans(Path("x.py"), text="@deco\ndef foo():\n    pass\n")
    assert spans == [("foo", 1, 3)]


def test_file_symbol_spans_returns_none_for_unparsable():
    assert file_symbol_spans(Path("x.py"), text="def :") is None


def test_locate_symbol_span_resolves_uniquely():
    assert locate_symbol_span(Path("x.py"), "beta", text=SRC) == (5, 6, "")


# ------------------------------------------------------- materialized outputs
def test_materialization_exposes_buffers_results_and_edits(tmp_path):
    root, allowed, read = _all(tmp_path)
    _write(root)
    intents = [StructuredIntent("app.py", "beta", "fix")]
    edits = [StructuredEdit("replace_symbol", "app.py", "beta", "",
                            "def beta():\n    return 22")]
    mat = materialize_structured_edits(root, intents, edits,
                                       allowed_files=allowed, read_spans=read,
                                       cfg=Config())
    assert mat.files == ("app.py",)
    assert isinstance(mat.edits[0], Edit)
    assert mat.edits[0].file == "app.py" and mat.edits[0].replace == mat.buffers["app.py"]
    assert len(mat.results) == 1
    r = mat.results[0]
    assert r.ok and r.before == SRC and r.after == mat.buffers["app.py"]
    assert r.sha and r.diff


def test_legacy_parse_edits_unchanged(tmp_path):
    # parse_structured_reply must not affect the legacy block parser
    from pedacito.edits import parse_edits
    edits = parse_edits("--- FILE: app.py\n<<<<<<< SEARCH\nreturn 2\n"
                        "=======\nreturn 3\n>>>>>>> REPLACE\n")
    assert len(edits) == 1 and edits[0].search == "return 2"
    assert edits[0].replace == "return 3"
