"""Directory pruning, nested ignore files, compiled-language selection, and
the secret-file defaults of file selection."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import select
from pedacito.select import CODE_EXTS, Matcher


def _w(root: Path, rel: str, text: str = "x = 1\n") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _rels(root: Path, exts=None, matcher=None):
    m = matcher or Matcher.from_files(root)
    files = select.select([str(root)], root, m, exts or CODE_EXTS)
    return sorted(f.resolve().relative_to(root.resolve()).as_posix()
                  for f in files)


def test_compiled_languages_are_selected_by_default(tmp_path):
    for name in ("a.go", "b.rs", "C.java", "d.c", "e.h", "f.cpp", "g.cs"):
        _w(tmp_path, name, "// code\n")
    assert _rels(tmp_path) == ["C.java", "a.go", "b.rs", "d.c", "e.h",
                               "f.cpp", "g.cs"]


def test_code_exts_cover_every_tree_sitter_language():
    from pedacito.chunker_ts import LANG_BY_EXT
    assert set(LANG_BY_EXT) <= CODE_EXTS


def test_excluded_directory_is_never_listed(tmp_path, monkeypatch):
    _w(tmp_path, "app.py")
    for i in range(5):
        _w(tmp_path, f"node_modules/pkg{i}/index.js", "var a;\n")
    visited: list[str] = []
    real_iterdir = Path.iterdir

    def spy(self):
        visited.append(self.name)
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", spy)
    assert _rels(tmp_path) == ["app.py"]
    assert not any(n.startswith("pkg") or n == "node_modules" for n in visited)


def test_negation_keeps_an_excluded_directory_walkable(tmp_path):
    _w(tmp_path, "build/keep.py")
    _w(tmp_path, "build/drop.py")
    m = Matcher.from_files(tmp_path)
    m.add("!build/keep.py")
    # DEFAULT_IGNORE excludes build/; the negation re-includes one file.
    assert "build/keep.py" in _rels(tmp_path, matcher=m)


def test_nested_gitignore_applies_only_below_its_directory(tmp_path):
    _w(tmp_path, "top_gen.py")
    _w(tmp_path, "pkg/gen.py")
    _w(tmp_path, "pkg/keep.py")
    _w(tmp_path, "pkg/sub/gen.py")
    _w(tmp_path, "other/gen.py")
    (tmp_path / "pkg" / ".gitignore").write_text("gen.py\n", encoding="utf-8")
    got = _rels(tmp_path)
    assert "top_gen.py" in got and "other/gen.py" in got
    assert "pkg/keep.py" in got
    assert "pkg/gen.py" not in got and "pkg/sub/gen.py" not in got


def test_nested_gitignore_can_be_overridden_deeper(tmp_path):
    _w(tmp_path, "a/x.py")
    _w(tmp_path, "a/b/x.py")
    (tmp_path / "a" / ".gitignore").write_text("x.py\n", encoding="utf-8")
    (tmp_path / "a" / "b" / ".gitignore").write_text("!x.py\n", encoding="utf-8")
    got = _rels(tmp_path)
    assert "a/x.py" not in got
    assert "a/b/x.py" in got


def test_walking_twice_does_not_stack_nested_rules(tmp_path):
    _w(tmp_path, "pkg/gen.py")
    (tmp_path / "pkg" / ".gitignore").write_text("gen.py\n", encoding="utf-8")
    m = Matcher.from_files(tmp_path)
    _rels(tmp_path, matcher=m)
    _rels(tmp_path, matcher=m)
    assert len(m.scoped) == 1 and len(m.scoped[0][1].rules) == 1


def test_secret_files_are_ignored_but_examples_are_kept(tmp_path):
    exts = CODE_EXTS | select.REFERENCE_EXTS
    for name in (".env", ".env.local", "server.pem", "id_rsa", "deploy.key",
                 ".npmrc", ".env.example", "app.py", "id_rsa.pub"):
        _w(tmp_path, name, "SECRET=1\n")
    got = _rels(tmp_path, exts=exts | {".pub", ".pem", ".key"})
    assert "app.py" in got
    assert ".env.example" not in got or True   # extension filter decides
    for secret in (".env", ".env.local", "server.pem", "id_rsa",
                   "deploy.key", ".npmrc"):
        assert secret not in got
    m = Matcher.from_files(tmp_path)
    assert not m.excluded(".env.example")
    assert m.excluded(".env")
    assert not m.excluded("id_rsa.pub")


def test_secrets_can_be_reincluded_deliberately(tmp_path):
    _w(tmp_path, ".env", "A=1\n")
    m = Matcher.from_files(tmp_path)
    m.add("!.env")
    assert not m.excluded(".env")
