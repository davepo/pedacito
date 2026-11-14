"""Windows CI portability guards that run deterministically on Linux.

The GitHub Windows runner fails differently from Linux: cp1252 is the default
text encoding, `\\` separators and `C:` drive letters break naive path
parsing, CRLF files shift byte/line offsets, and `os.replace()`/`os.unlink()`
raise WinError 32 while a handle is still open. Each test below simulates one
of those shapes here, before a push can discover it on the runner.

Zero extra dependencies: stdlib AST, pathlib, tempfile and pytest only.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (  # noqa: E402
    agent,
    browser,
    checkpoint,
    chunker,
    chunker_ts,
    config,
    diagnostics,
    edits,
    fileio,
    proc,
    ratchet,
    sandbox,
    spec,
    verify,
    webcheck,
)
from pedacito.config import Config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = (REPO_ROOT / "pedacito", REPO_ROOT / "tests")


# ---------------------------------------------------------------------------
# 1. static encoding audit: no implicit locale encoding anywhere
# ---------------------------------------------------------------------------
def _python_files():
    for base in SCAN_DIRS:
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def _rel_label(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _root_name(node) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _keyword(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _open_mode(call):
    """Mode string, 'r' when omitted, or None when it cannot be proven."""
    mode = "r"
    if len(call.args) >= 2:
        arg = call.args[1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            mode = arg.value
        else:
            return None
    kw = _keyword(call, "mode")
    if kw is not None:
        if isinstance(kw, ast.Constant) and isinstance(kw.value, str):
            mode = kw.value
        else:
            return None
    return mode


def _is_subprocess_call(call, aliases) -> bool:
    fn = call.func
    if isinstance(fn, ast.Attribute):
        return (fn.attr in ("run", "Popen", "call", "check_call", "check_output")
                and _root_name(fn) in aliases)
    return isinstance(fn, ast.Name) and fn.id in aliases - {"subprocess"}


def _subprocess_aliases(tree: ast.AST) -> set[str]:
    aliases = {"subprocess"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "subprocess":
                    aliases.add(a.asname or "subprocess")
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            aliases.update(a.asname or a.name for a in node.names)
    return aliases


def _has_text_mode(call) -> bool:
    for name in ("text", "universal_newlines"):
        value = _keyword(call, name)
        if value is None:
            continue
        if isinstance(value, ast.Constant) and value.value is False:
            continue
        return True                 # True, or a runtime value we cannot prove
    return False


def _encoding_violations(path: Path, tree: ast.AST) -> list[str]:
    bad: list[str] = []
    subprocess_aliases = _subprocess_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        where = f"{_rel_label(path)}:{node.lineno}"
        # builtins.open / io.open: text mode must name its encoding.
        is_builtin_open = (
            (isinstance(fn, ast.Name) and fn.id == "open")
            or (isinstance(fn, ast.Attribute) and fn.attr == "open"
                and _root_name(fn) in ("builtins", "io"))
        )
        if is_builtin_open:
            mode = _open_mode(node)
            if mode is not None and "b" in mode:
                continue
            if _keyword(node, "encoding") is None:
                bad.append(f"{where}: open() without encoding= (mode={mode!r})")
            continue
        # pathlib.Path.read_text / write_text. `fileio.read_text` is the
        # project's byte-level wrapper (it owns encoding internally); its
        # signature intentionally takes only (path, errors).
        if isinstance(fn, ast.Attribute) and fn.attr in ("read_text", "write_text"):
            if _root_name(fn) == "fileio":
                continue
            if _keyword(node, "encoding") is None:
                bad.append(f"{where}: .{fn.attr}() without encoding=")
            continue
        if _is_subprocess_call(node, subprocess_aliases) and _has_text_mode(node):
            if _keyword(node, "encoding") is None:
                bad.append(f"{where}: subprocess text mode without encoding=")
    return bad


def test_no_implicit_locale_encoding():
    problems: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:      # a broken file is a different failure
            pytest.fail(f"{path.relative_to(REPO_ROOT)} does not parse: {exc}")
        problems.extend(_encoding_violations(path, tree))
    assert not problems, (
        "implicit locale encoding would crash on Windows cp1252:\n"
        + "\n".join(problems))


# ---------------------------------------------------------------------------
# 2. drive-letter colon and backslash path parsing
# ---------------------------------------------------------------------------
WINDOWS_ABS = r"C:\Users\runneradmin\AppData\Local\Temp\pytest-0\proj\src\app.py"


def test_windows_relative_paths_normalise_to_one_key():
    variants = (r"src\sub\app.py", r".\src\sub\app.py", "src/sub/app.py")
    assert [fileio.normalise_rel(v) for v in variants] == ["src/sub/app.py"] * 3
    assert len({fileio.path_key(v) for v in variants}) == 1
    assert fileio.path_key(variants[0]) == "src/sub/app.py"

    # A drive letter is a prefix, never a filename fragment.
    normalised = fileio.normalise_rel(WINDOWS_ABS)
    assert normalised.startswith("C:/Users/runneradmin/")
    assert normalised.endswith("/src/app.py")


def test_windows_path_regexes_and_gate_parsers(tmp_path):
    ruff_line = (r"C:\Users\runner\proj\src\app.py:42:5: "
                 "F821 Undefined name 'foo'")

    matches = list(agent._PATH_LINE_RE.finditer(ruff_line))
    assert matches, "the path:line regex missed a Windows absolute path"
    assert matches[0].group(1) == r"C:\Users\runner\proj\src\app.py"
    assert matches[0].group(1) != r"\Users\runner\proj\src\app.py"

    pins = agent._diagnostic_pins({"src\\app.py": [ruff_line]})
    assert pins == [("src/app.py", 42, 42)]
    abs_pins = agent._diagnostic_pins({WINDOWS_ABS: [ruff_line]})
    assert abs_pins == [("C:/Users/runneradmin/AppData/Local/Temp/"
                          "pytest-0/proj/src/app.py", 42, 42)]

    assert diagnostics._norm_finding(ruff_line) == "F821 Undefined name 'foo'"
    assert diagnostics._norm_finding(
        r"src\app.py:12: ContractError in src\app.py:12"
    ) == r"ContractError in src\app.py:12"

    m = spec._RUFF_LINE_RE.match(ruff_line)
    assert m is not None
    assert m.group("file") == r"C:\Users\runner\proj\src\app.py"
    assert (m.group("line"), m.group("col"), m.group("code")) == ("42", "5", "F821")
    assert spec._gate_label(tmp_path, r"src\app.py") == "src/app.py"
    # A Windows-spelled absolute path under the project also becomes relative.
    win_abs = f"{tmp_path}\\src\\app.py"
    assert spec._gate_label(tmp_path, win_abs) == "src/app.py"

    assert verify.find_needs(r"NEED src\pricing.py:120-168") == [
        ("src/pricing.py", 120, 168)]
    assert verify.find_needs(
        r"NEED C:\Users\runner\proj\src\pricing.py:120-168") == [
        ("C:/Users/runner/proj/src/pricing.py", 120, 168)]
    # A colon in the drive prefix is never read as the line separator.
    assert verify.find_needs(r"NEED C:\notes.txt:9") == [
        ("C:/notes.txt", 9, 9)]


def test_static_gate_output_is_posix_relative(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def f():\n    return missing_name\n", encoding="utf-8")
    errors = spec.run_fast_static_gate(tmp_path, [Path("src/app.py")])
    assert errors, "the static gate missed an undefined name"
    assert all(e.startswith("src/app.py:") for e in errors), errors
    assert all("\\" not in e.split(":", 1)[0] for e in errors), errors


# ---------------------------------------------------------------------------
# 3. CRLF byte/line parity through chunkers, webcheck and the edit engine
# ---------------------------------------------------------------------------
PY_SRC = '''"""Module 漢字 — doc."""


class Greeter:
    @staticmethod
    def greet(name):
        """Say 漢字 — hello."""
        return "hello " + name


def shout(text):
    return text.upper() + " — 漢字"
'''
JS_SRC = '''export class Greeter {
  greet(name) {
    return `hello ${name} 漢字 —`;
  }
}

export function shout(text) {
  return text.toUpperCase() + " — 漢字";
}

const el = document.getElementById("ghost-id");
'''
TS_SRC = '''export interface Item { name: string; }

export function label(item: Item): string {
  return item.name + " — 漢字";
}
'''
GO_SRC = '''package main

func Greet(name string) string {
    return "hello " + name + " — 漢字"
}
'''
RS_SRC = '''pub fn greet(name: &str) -> String {
    format!("hello {} — 漢字", name)
}
'''

SAMPLES = {
    "python": {
        "rel": "app.py", "ext": ".py", "src": PY_SRC,
        "symbol": "Greeter.greet",
        "search": '        return "hello " + name',
        "replace": '        return "hi " + name',
        "symbol_body": ('def greet(name):\n'
                        '    """Say 漢字 — hello."""\n'
                        '    return "hi " + name\n'),
    },
    "javascript": {
        "rel": "app.js", "ext": ".js", "src": JS_SRC,
        "symbol": "shout",
        "search": '  return text.toUpperCase() + " — 漢字";',
        "replace": '  return text.toLowerCase() + " — 漢字";',
        "symbol_body": ('export function shout(text) {\n'
                        '  return text.trim() + " — 漢字";\n}\n'),
    },
    "typescript": {
        "rel": "app.ts", "ext": ".ts", "src": TS_SRC,
        "symbol": "label",
        "search": '  return item.name + " — 漢字";',
        "replace": '  return item.name.trim() + " — 漢字";',
        "symbol_body": ('export function label(item: Item): string {\n'
                        '  return item.name.trim();\n}\n'),
    },
    "go": {
        "rel": "app.go", "ext": ".go", "src": GO_SRC,
        "symbol": "Greet",
        "search": '    return "hello " + name + " — 漢字"',
        "replace": '    return "hi " + name + " — 漢字"',
        "symbol_body": ('func Greet(name string) string {\n'
                        '    return "hi " + name\n}\n'),
    },
    "rust": {
        "rel": "app.rs", "ext": ".rs", "src": RS_SRC,
        "symbol": "greet",
        "search": '    format!("hello {} — 漢字", name)',
        "replace": '    format!("hi {} — 漢字", name)',
        "symbol_body": ('pub fn greet(name: &str) -> String {\n'
                        '    format!("hi {}", name)\n}\n'),
    },
}


def _chunk_facts(rel: str, ext: str, src: str) -> list[tuple]:
    if ext == ".py":
        chunks = chunker.chunk_source(rel, src)
    else:
        chunks = chunker_ts.chunk_file_ts(rel, src, ext)
    return [(c.qualname, c.kind, c.start, c.end, c.signature,
             tuple(c.calls)) for c in chunks]


def _line_of(src: str, needle: str) -> int:
    for n, line in enumerate(src.split("\n"), 1):
        if line == needle:
            return n
    raise AssertionError(f"sample line not found: {needle!r}")


def _assert_newline_style(data: bytes, newline: str) -> None:
    if newline == "\r\n":
        assert b"\r\n" in data
        assert data.count(b"\r\n") == data.count(b"\n")
        assert b"\r" not in data.replace(b"\r\n", b"")
    else:
        assert b"\r" not in data
        assert b"\r\n" not in data


def _write_pair(tmp_path: Path, rel: str, src: str):
    lf = tmp_path / "lf" / rel
    crlf = tmp_path / "crlf" / rel
    lf.parent.mkdir(parents=True, exist_ok=True)
    crlf.parent.mkdir(parents=True, exist_ok=True)
    lf.write_bytes(src.encode("utf-8"))
    crlf.write_bytes(src.replace("\n", "\r\n").encode("utf-8"))
    return lf, crlf


@pytest.mark.parametrize("name", list(SAMPLES))
def test_crlf_chunking_and_edits_parity(tmp_path, name):
    sample = SAMPLES[name]
    src, rel, ext = sample["src"], sample["rel"], sample["ext"]
    if ext != ".py" and not chunker_ts.TS_AVAILABLE:
        pytest.skip("tree-sitter grammars are not installed")
    lf_path, crlf_path = _write_pair(tmp_path, rel, src)

    assert fileio.read_text(lf_path) == src
    assert fileio.read_text(crlf_path) == src       # normalised on read
    _assert_newline_style(crlf_path.read_bytes(), "\r\n")

    # Chunk identities, line ranges and signatures are style-independent.
    assert _chunk_facts(rel, ext, src) == \
        _chunk_facts(rel, ext, src.replace("\n", "\r\n"))

    # Symbol location is byte/line style-independent too.
    lf_span = chunker.locate_symbol_span(lf_path, sample["symbol"])
    crlf_span = chunker.locate_symbol_span(crlf_path, sample["symbol"])
    assert lf_span is not None and lf_span == crlf_span

    if ext == ".js":
        lf_findings = [(f.rel, f.line, f.message)
                       for f in webcheck.run(tmp_path / "lf")]
        crlf_findings = [(f.rel, f.line, f.message)
                         for f in webcheck.run(tmp_path / "crlf")]
        assert lf_findings == crlf_findings
        assert any("ghost-id" in f[2] for f in lf_findings), lf_findings

    line_no = _line_of(src, sample["search"])
    symbol_body = sample["symbol_body"]

    for root, newline in ((tmp_path / "lf", "\n"),
                          (tmp_path / "crlf", "\r\n")):
        target = root / rel

        # SEARCH / REPLACE
        target.write_bytes(src.replace("\n", newline).encode("utf-8"))
        reply = (f"--- FILE: {rel}\n<<<<<<< SEARCH\n{sample['search']}\n"
                 f"=======\n{sample['replace']}\n>>>>>>> REPLACE\n")
        results = edits.apply_edits(root, edits.parse_edits(reply), backup=False)
        assert results and results[0].ok, (name, newline, results)
        assert fileio.read_text(target) == src.replace(sample["search"],
                                                       sample["replace"])
        _assert_newline_style(target.read_bytes(), newline)

        # LINES a-b
        target.write_bytes(src.replace("\n", newline).encode("utf-8"))
        reply = (f"--- FILE: {rel} LINES {line_no}-{line_no}\n"
                 "<<<<<<< SEARCH\n\n=======\n"
                 f"{sample['replace']}\n>>>>>>> REPLACE\n")
        results = edits.apply_edits(root, edits.parse_edits(reply), backup=False)
        assert results and results[0].ok, (name, newline, results)
        assert sample["replace"].strip() in fileio.read_text(target)
        _assert_newline_style(target.read_bytes(), newline)

        # <<<<<<< SYMBOL splice
        target.write_bytes(src.replace("\n", newline).encode("utf-8"))
        reply = (f"<<<<<<< SYMBOL {rel} :: {sample['symbol']}\n"
                 f"{symbol_body}>>>>>>> END_SYMBOL\n")
        results = edits.apply_edits(root, edits.parse_edits(reply), backup=False)
        assert results and results[0].ok, (name, newline, results)
        text = fileio.read_text(target)
        assert sample["symbol_body"].splitlines()[0] in text
        _assert_newline_style(target.read_bytes(), newline)
        if ext == ".py":
            assert "@staticmethod" in text       # decorator carried over


# ---------------------------------------------------------------------------
# 4. NamedTemporaryFile and atomic-write handle audits
# ---------------------------------------------------------------------------
def _named_tempfile_violations(path: Path, tree: ast.AST) -> list[str]:
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            fname = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else "")
            if fname != "NamedTemporaryFile":
                continue
            var = item.optional_vars
            if not isinstance(var, ast.Name):
                continue
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if (isinstance(sub, ast.Attribute) and sub.attr == "name"
                            and isinstance(sub.value, ast.Name)
                            and sub.value.id == var.id):
                        bad.append(
                            f"{_rel_label(path)}:{sub.lineno}: "
                            f"uses {var.id}.name while "
                            "NamedTemporaryFile is still open "
                            "(WinError 32; pass delete=False, close, then "
                            "open by name)")
    return bad


def test_no_unsafe_named_temporary_file():
    problems: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        problems.extend(_named_tempfile_violations(path, tree))
    assert not problems, "\n".join(problems)

    # The detector itself must catch the unsafe shape and allow the safe one.
    unsafe = ast.parse(
        "with tempfile.NamedTemporaryFile() as f:\n"
        "    open(f.name)\n")
    safe = ast.parse(
        "f = tempfile.NamedTemporaryFile(delete=False)\n"
        "f.close()\n"
        "open(f.name)\n")
    assert _named_tempfile_violations(Path("x.py"), unsafe)
    assert not _named_tempfile_violations(Path("x.py"), safe)


def _open_fd_paths() -> set[str]:
    """Real paths of every open file descriptor in this process."""
    out: set[str] = set()
    fd_dir = Path("/proc/self/fd")
    for entry in fd_dir.iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if not target.startswith("/"):
            continue
        out.add(os.path.realpath(target.removesuffix(" (deleted)")))
    return out


def test_atomic_writes_close_handles_before_replace(tmp_path, monkeypatch):
    if not Path("/proc/self/fd").is_dir():
        pytest.skip("needs /proc/self/fd to observe open handles")

    real_replace = os.replace
    checked: list[tuple[str, str]] = []

    def spy(src, dst):
        open_now = _open_fd_paths()
        for label, path in (("temp source", src), ("target", dst)):
            real = os.path.realpath(str(path))
            assert real not in open_now, (
                f"{label} {path} still has an open file descriptor when "
                "os.replace() runs -- Windows raises WinError 32")
        checked.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)

    root = tmp_path
    live = root / "app.py"
    fileio.write(live, "def f():\n    return 1\n")

    reply = ("--- FILE: app.py\n<<<<<<< SEARCH\n"
             "def f():\n    return 1\n=======\n"
             "def f():\n    return 2\n>>>>>>> REPLACE\n")
    results = edits.apply_edits(root, edits.parse_edits(reply), backup=False)
    assert results and results[0].ok, results

    cfg = Config()
    cfg.log_sessions = True
    cp = checkpoint.save(root, "ask", "task", [], "reply", cfg=cfg)
    assert cp is not None and cp.is_file()

    ratchet.begin(root, 1)
    fileio.write(live, "def f():\n    return 3\n")
    report = ratchet.rollback(root, 1)
    assert report["error"] is None, report
    assert "app.py" in report["restored"], report

    assert checked, "none of the atomic writers called os.replace()"
    assert len(checked) >= 4, checked


# ---------------------------------------------------------------------------
# 5. simulated win32: sandbox, proc tree kill, browser, config/state paths
# ---------------------------------------------------------------------------
def test_sandbox_degrades_cleanly_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sandbox, "_noticed", set())
    cfg = Config()

    state, detail = sandbox.status(cfg, root=tmp_path)
    assert state == "unavailable"
    assert "Windows" in detail

    argv, note = sandbox.wrap([sys.executable, "-c", "print(1)"], cfg, tmp_path)
    assert argv == [sys.executable, "-c", "print(1)"]
    assert note == ""

    cfg.sandbox_strict = True
    with pytest.raises(sandbox.SandboxRefused):
        sandbox.wrap([sys.executable, "-c", "print(1)"], cfg, tmp_path)


def test_proc_kill_tree_uses_taskkill_on_windows(monkeypatch):
    calls: list[list[str]] = []

    class FakeRun:
        returncode = 0

    def fake_run(argv, **_kw):
        calls.append(list(argv))
        return FakeRun()

    class FakeProc:
        pid = 4242

        def __init__(self):
            self.killed = False

        def kill(self):
            self.killed = True

    monkeypatch.setattr(proc.subprocess, "run", fake_run)
    monkeypatch.setattr(proc.os, "name", "nt")
    child = FakeProc()
    proc.kill_tree(child)

    assert calls, "taskkill was not invoked"
    assert calls[0][:5] == ["taskkill", "/F", "/T", "/PID", "4242"]
    assert child.killed


def test_browser_resolves_a_mocked_windows_chrome_path(tmp_path, monkeypatch):
    win_path = (r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    fake = tmp_path / win_path
    fake.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(browser, "_CHROME_NAMES", ())
    monkeypatch.setattr(browser, "_CHROME_PATHS", (str(fake),))
    monkeypatch.setattr(browser.shutil, "which", lambda _name: None)
    assert browser.chrome_executable() == str(fake)


def test_windows_config_and_gui_state_dirs(tmp_path, monkeypatch):
    appdata = tmp_path / "Roaming"
    localappdata = tmp_path / "Local"
    posix_xdg = tmp_path / "must-not-be-used"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setenv("LOCALAPPDATA", str(localappdata))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(posix_xdg / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(posix_xdg / "state"))

    cfg_dir = config._user_config_dir()
    state_dir = config._user_state_dir()
    assert cfg_dir == appdata / "pedacito"
    assert state_dir == localappdata / "pedacito"
    assert "must-not-be-used" not in cfg_dir.parts
    assert "must-not-be-used" not in state_dir.parts

    monkeypatch.setattr(config, "USER_STATE_DIR", state_dir)
    from pedacito.gui import core
    assert core.threads_dir(None) == state_dir / "threads" / "global"
