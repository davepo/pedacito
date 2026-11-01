"""Which file tests which: naming conventions across languages.

    is_test, stem, family = classify("web/src/api.test.ts")   # (True, "api", "js")
    link = link_for(rel, files)                                # ("tested by", "...")

Conventions understood (stem = the module name a test's affixes hide):

  Python   test_x.py  x_test.py  tests/x.py
  JS/TS    x.test.ts  x.spec.js  __tests__/x.ts  tests/x.ts
  Go       x_test.go
  Rust     tests/x.rs   (inline #[cfg(test)] modules need no link)
  Java/C#  XTest.java  XTests.java  TestX.java  XTests.cs   (src/test/...)
  Ruby     x_spec.rb  test_x.rb  spec/x.rb
  C/C++    x_test.cc  test_x.cpp  tests/x.c

Links are evidence for the map, never proof: a test links to a module only
when the stem match is unambiguous within the same language family.
"""

from __future__ import annotations

import posixpath
import re

_FAMILY = {
    ".py": "py", ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "js", ".tsx": "js", ".go": "go", ".rs": "rs", ".java": "java",
    ".cs": "cs", ".rb": "rb", ".c": "c", ".h": "c", ".cc": "c", ".cpp": "c",
    ".cxx": "c", ".hpp": "c", ".hh": "c", ".php": "php", ".kt": "kt",
    ".kts": "kt", ".swift": "swift", ".scala": "scala", ".lua": "lua",
}
_TEST_DIRS = {"tests", "test", "__tests__", "spec", "specs", "__test__"}
_JS_SUFFIX_RE = re.compile(r"^(?P<stem>.+)\.(?:test|spec)$", re.IGNORECASE)


def family_of(rel: str) -> str:
    """Language family of a path ('' when it is not a source file)."""
    return _FAMILY.get(posixpath.splitext(rel)[1].lower(), "")


def classify(rel: str) -> tuple[bool, str, str]:
    """(is_test, stem, family) for a project-relative path. `stem` is lower
    case; non-source files return family ''."""
    fam = family_of(rel)
    if not fam:
        return False, "", ""
    p = rel.replace("\\", "/")
    dirs = {d.lower() for d in posixpath.dirname(p).split("/")}
    base = posixpath.splitext(posixpath.basename(p))[0]
    low = base.lower()
    in_test_dir = bool(dirs & _TEST_DIRS) or "src/test/" in p.lower()
    if fam == "js":
        m = _JS_SUFFIX_RE.match(base)
        if m:
            return True, m.group("stem").lower(), fam
    if low.startswith("test_") and fam in ("py", "rb", "c"):
        return True, low[len("test_"):], fam
    if low.endswith("_test") and fam in ("py", "go", "c", "rb", "rs"):
        return True, low[:-len("_test")], fam
    if low.endswith("_spec") and fam == "rb":
        return True, low[:-len("_spec")], fam
    if fam in ("java", "cs"):
        for suf in ("tests", "test", "it"):
            if low.endswith(suf) and len(low) > len(suf):
                return True, low[:-len(suf)], fam
        if low.startswith("test") and len(low) > 4:
            return True, low[4:], fam
    if in_test_dir:
        # conftest / helpers inside tests/ link to nothing
        if low in ("conftest", "setup", "helpers", "__init__", "fixtures",
                   "utils", "common", "mod", "index"):
            return True, "", fam
        return True, low, fam
    return False, low, fam


def build_maps(files) -> tuple[dict, dict]:
    """({(stem, family): [module paths]}, {(stem, family): [test paths]})."""
    mods: dict[tuple, list[str]] = {}
    tests: dict[tuple, list[str]] = {}
    for f in files:
        is_t, stem, fam = classify(f)
        if not fam or not stem:
            continue
        (tests if is_t else mods).setdefault((stem, fam), []).append(f)
    return mods, tests


def link_for(rel: str, mods: dict, tests: dict) -> tuple[str, str] | None:
    """('tested by', test paths) for a module, ('tests', module path) for a
    test file, when the match is unambiguous."""
    is_t, stem, fam = classify(rel)
    if not fam or not stem:
        return None
    key = (stem, fam)
    if is_t:
        cands = [m for m in mods.get(key, ()) if m != rel]
        return ("tests", cands[0]) if len(cands) == 1 else None
    cands = sorted(t for t in tests.get(key, ()) if t != rel)
    return ("tested by", ", ".join(cands[:3])) if cands else None


def tests_of(rel: str, tests: dict) -> list[str]:
    """Test files linked to module `rel` (by name)."""
    is_t, stem, fam = classify(rel)
    if is_t or not fam or not stem:
        return []
    return sorted(t for t in tests.get((stem, fam), ()) if t != rel)
