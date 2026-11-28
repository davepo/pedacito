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
    index,
    proc,
    processes,
    ratchet,
    sandbox,
    spec,
    verify,
    webcheck,
)
from pedacito.config import Config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = (REPO_ROOT / "pedacito", REPO_ROOT / "tests")

# The concrete path class of the REAL filesystem, captured before any test
# patches os.name. The Windows fakes force this class so registry records land
# in tmp_path: PosixPath on Linux, WindowsPath on a real nt host. Hardcoding
# PosixPath would raise NotImplementedError under Wine/Windows.
_REAL_PATH = Path(".").__class__


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
    # WP-4005: Windows now has a Job Object sandbox when the probe succeeds, so
    # force it unavailable to exercise the degradation path deterministically.
    monkeypatch.setattr(sandbox, "_win_ready", lambda: False)
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


# --- the background-server registry: a job object is its process group ------
# `taskkill /T` can only walk a LIVE parent-child chain. When a launcher exits
# first (`npm run dev` spawning the real server) the child survives `stop` and
# keeps the port. The crucible test that proves this needs a Windows runner;
# these tests pin the Win32 call sequence here, so a wrong argument or a leaked
# handle fails on the first Linux job instead of on the runner.
#
# The runner itself is the hostile case: GitHub Actions wraps every Windows
# step in a job object, and a child that already belongs to a job can be
# refused nested assignment -- so the fakes also model refusal, and the
# snapshot walk that must reap the orphans when the job is missing.
class _FakeKernel32:
    """The job-object entry points `processes` uses, recording every call."""

    def __init__(self, *, create_ok=True, assign_ok=True, open_ok=True,
                 assign_error=0, processes=(), threads=(), resume_ok=True,
                 dead_pids=(), active_processes=0):
        self.calls: list[tuple] = []
        self.create_ok = create_ok
        self.assign_ok = assign_ok
        self.open_ok = open_ok
        self.assign_error = assign_error
        self.resume_ok = resume_ok
        # Pids OpenProcess must refuse (modeling already-exited processes).
        self.dead_pids = set(dead_pids)
        # Answer to QueryInformationJobObject for ActiveProcesses.
        self.active_processes = active_processes
        # The Win32 error behind the last failed call. `_pretend_windows`
        # points `_last_error` here, because ctypes' get/set_last_error only
        # exist on Windows and these fakes run on every platform.
        self.last_error = 0
        # (pid, ppid, creation FILETIME) and (tid, owner pid) tables the
        # snapshot walks read; empty means "no such processes/threads".
        self.proc_table = list(processes)
        self.thread_table = list(threads)
        self._next = 100
        self.opened: list[int] = []
        self.closed: list[int] = []
        self._pid_of_handle: dict[int, int] = {}
        self._tid_of_handle: dict[int, int] = {}
        self._proc_iter: iter = iter(())
        self._thread_iter: iter = iter(())

    @property
    def leaked(self) -> list[int]:
        """Handles opened and never closed: a leak that outlives the server."""
        return [h for h in self.opened if h not in self.closed]

    def _handle(self, ok: bool) -> int:
        if not ok:
            return 0                    # the Win32 failure return
        self._next += 1
        self.opened.append(self._next)
        return self._next

    def CreateJobObjectW(self, _attrs, name):
        handle = self._handle(self.create_ok)
        if not handle:
            self.last_error = 5         # ERROR_ACCESS_DENIED
        self.calls.append(("create", handle, name))
        return handle

    def OpenProcess(self, access, _inherit, pid):
        if pid in self.dead_pids:
            self.last_error = 87           # ERROR_INVALID_PARAMETER: no process
            self.calls.append(("open_process", 0, access, pid))
            return 0
        handle = self._handle(self.create_ok)
        if not handle:
            self.last_error = 5
        self.calls.append(("open_process", handle, access, pid))
        if handle:
            self._pid_of_handle[handle] = pid
        return handle

    def AssignProcessToJobObject(self, hjob, hproc):
        ok = 1 if self.assign_ok else 0
        if not ok:
            self.last_error = self.assign_error
        self.calls.append(("assign", hjob, hproc, ok))
        return ok

    def OpenJobObjectW(self, access, _inherit, name):
        handle = self._handle(self.open_ok)
        if not handle:
            self.last_error = 2         # ERROR_FILE_NOT_FOUND
        self.calls.append(("open_job", handle, access, name))
        return handle

    def TerminateJobObject(self, hjob, code):
        self.calls.append(("terminate", hjob, code))
        return 1

    def TerminateProcess(self, handle, code):
        self.calls.append(("terminate_process", handle, code))
        return 1

    def QueryInformationJobObject(self, handle, cls, info_ptr, size, _retlen):
        import ctypes
        info = ctypes.cast(
            info_ptr,
            ctypes.POINTER(processes._job_accounting_struct())).contents
        info.ActiveProcesses = self.active_processes
        self.calls.append(("query_job", handle, cls))
        return 1

    def CloseHandle(self, handle):
        self.calls.append(("close", handle))
        self.closed.append(handle)
        return 1

    # -- the suspended-child resume and the orphan snapshot walk ------------

    def CreateToolhelp32Snapshot(self, flags, _pid):
        handle = self._handle(True)
        self.calls.append(("snapshot", handle, flags))
        if flags == 0x2:                # TH32CS_SNAPPROCESS
            self._proc_iter = iter(self.proc_table)
        elif flags == 0x4:              # TH32CS_SNAPTHREAD
            self._thread_iter = iter(self.thread_table)
        return handle

    def _next_proc(self, ptr):
        import ctypes
        entry = ctypes.cast(
            ptr, ctypes.POINTER(processes._win_structs()[1])).contents
        try:
            pid, ppid, _born = next(self._proc_iter)
        except StopIteration:
            return 0
        entry.th32ProcessID = pid
        entry.th32ParentProcessID = ppid
        return 1

    def Process32FirstW(self, _snap, ptr):
        return self._next_proc(ptr)

    def Process32NextW(self, _snap, ptr):
        return self._next_proc(ptr)

    def GetProcessTimes(self, handle, born_ptr, _e, _k, _u):
        import ctypes
        from ctypes import wintypes
        pid = self._pid_of_handle.get(handle, 0)
        born = next((b for p, _pp, b in self.proc_table if p == pid), 0)
        ft = ctypes.cast(born_ptr, ctypes.POINTER(wintypes.FILETIME)).contents
        ft.dwLowDateTime = born & 0xFFFFFFFF
        ft.dwHighDateTime = born >> 32
        self.calls.append(("times", handle, born))
        return 1

    def _next_thread(self, ptr):
        import ctypes
        entry = ctypes.cast(
            ptr, ctypes.POINTER(processes._win_structs()[0])).contents
        try:
            tid, owner = next(self._thread_iter)
        except StopIteration:
            return 0
        entry.th32ThreadID = tid
        entry.th32OwnerProcessID = owner
        return 1

    # No W suffix: the real thread-snapshot API has no wide variant.
    def Thread32First(self, _snap, ptr):
        return self._next_thread(ptr)

    def Thread32Next(self, _snap, ptr):
        return self._next_thread(ptr)

    def OpenThread(self, access, _inherit, tid):
        handle = self._handle(True)
        self.calls.append(("open_thread", handle, access, tid))
        if handle:
            self._tid_of_handle[handle] = tid
        return handle

    def ResumeThread(self, handle):
        tid = self._tid_of_handle.get(handle, 0)
        self.calls.append(("resume", handle, tid))
        return 1 if self.resume_ok else 0xFFFFFFFF


def _pretend_windows(monkeypatch, k32):
    monkeypatch.setattr(processes.os, "name", "nt")
    monkeypatch.setattr(processes, "_kernel32", lambda: k32)
    # ctypes' get/set_last_error exist only on Windows; the fake carries the
    # error code instead, so the diagnostics are testable on every platform.
    monkeypatch.setattr(processes, "_last_error",
                        lambda k: getattr(k, "last_error", 0))
    # A fresh job-handle registry per test: `_job_wrap` now RETAINS the job
    # handle it created, and that state must not leak between tests.
    monkeypatch.setattr(processes, "_JOB_HANDLES", {})
    # pathlib chooses WindowsPath from os.name; keep the REAL filesystem's
    # path class so registry records still land in tmp_path. `state_dir`
    # re-wraps its root with index's own Path, and fileio re-wraps the final
    # path again for its atomic write, so patch those namespaces too.
    from pedacito import security
    monkeypatch.setattr(processes, "Path", _REAL_PATH)
    monkeypatch.setattr(index, "Path", _REAL_PATH)
    monkeypatch.setattr(fileio, "Path", _REAL_PATH)
    monkeypatch.setattr(security, "Path", _REAL_PATH)
    runs: list[list[str]] = []

    def fake_run(argv, **_kw):
        runs.append(list(argv))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(processes.subprocess, "run", fake_run)
    return runs


def test_job_wrap_assigns_the_leader_and_retains_the_job_handle(monkeypatch):
    """The job handle is OWNED, not closed: a named job's name vanishes the
    moment its last handle closes -- even while members still run -- so the
    old "close it, reopen it by name later" plan made `stop` a no-op
    (ERROR_FILE_NOT_FOUND) and leaked the orphan. Only the transient process
    handle is closed; the job handle is retained for `_job_kill`."""
    k32 = _FakeKernel32()
    _pretend_windows(monkeypatch, k32)

    job, note = processes._job_wrap(4242, "web")

    assert note == ""
    assert job.startswith("Local\\pedacito-web-4242-"), job
    # create + open_process + assign, then only the process handle closes.
    assert [c[0] for c in k32.calls] == [
        "create", "open_process", "assign", "close"], k32.calls
    # OpenProcess asks for exactly SET_QUOTA|TERMINATE, the rights
    # AssignProcessToJobObject requires; anything less fails on the runner.
    access = [c for c in k32.calls if c[0] == "open_process"][0][2]
    assert access == processes._PROCESS_SET_QUOTA | processes._PROCESS_TERMINATE
    # The job handle is retained, not leaked: it is registered for later
    # termination and has not been closed.
    retained = processes._JOB_HANDLES.get(job)
    assert retained, "the job handle was not retained for later termination"
    assert retained not in k32.closed, "the retained handle was closed"
    # `_job_kill` terminates through the retained handle and then closes it.
    assert processes._job_kill(job) is True
    assert job not in processes._JOB_HANDLES
    assert not k32.leaked, "job/process handles were not closed"


def test_job_kill_reopens_by_name_only_when_no_handle_is_held(monkeypatch):
    """Cross-process stop (no retained handle) still opens the job by name --
    which works while the starting process still holds its own handle."""
    k32 = _FakeKernel32()
    _pretend_windows(monkeypatch, k32)

    assert processes._job_kill("Local\\pedacito-web-4242-abcd1234") is True

    opened = [c for c in k32.calls if c[0] == "open_job"]
    assert opened, "no open-by-name attempt was made"
    assert opened[0][3] == "Local\\pedacito-web-4242-abcd1234"
    assert opened[0][2] == processes._JOB_TERMINATE | processes._SYNCHRONIZE
    assert not k32.leaked


def test_job_wrap_is_a_no_op_when_the_api_refuses(monkeypatch):
    for kwargs in ({"create_ok": False},          # name already taken
                   {"assign_ok": False}):         # child already exited
        k32 = _FakeKernel32(**kwargs)
        _pretend_windows(monkeypatch, k32)
        job, note = processes._job_wrap(4242, "web")
        assert job == "" and note, kwargs
        assert not k32.leaked, kwargs
    # POSIX never loads kernel32 at all: `_kernel32()` returns None there.
    monkeypatch.setattr(processes.os, "name", "posix")
    monkeypatch.setattr(processes, "_kernel32", lambda: None)
    assert processes._job_wrap(4242, "web") == ("", "")


def test_job_wrap_says_which_step_failed_and_why(monkeypatch):
    """A silent '' is how the runner failure hid: the record carried no job
    and no reason, so `stop` looked like it worked while the orphan lived.
    Error 5 is the runner's refusal of nested assignment -- the note must
    name it, because that is the case the orphan walk exists for."""
    k32 = _FakeKernel32(assign_ok=False, assign_error=5)
    _pretend_windows(monkeypatch, k32)
    job, note = processes._job_wrap(4242, "web")
    assert job == ""
    assert "AssignProcessToJobObject failed (error 5)" in note, note
    assert "orphan walk" in note, note
    assert not k32.leaked

    k32 = _FakeKernel32(create_ok=False)
    _pretend_windows(monkeypatch, k32)
    job, note = processes._job_wrap(4242, "web")
    assert job == "" and "CreateJobObject failed" in note, note


def test_stop_reaps_the_orphaned_tree_through_the_job_on_windows(monkeypatch):
    """The failure the crucible test saw: leader gone, server alive.

    `stop` must prefer the job object. Taskkill /T on a pid whose children have
    been reparented kills nothing at all, and the registry still reports
    success -- which is how a dev server ends up holding a port across runs.
    """
    k32 = _FakeKernel32()
    runs = _pretend_windows(monkeypatch, k32)

    processes._terminate_pid(4242, "Local\\pedacito-web-4242-abcd1234")

    opened = [c for c in k32.calls if c[0] == "open_job"][0]
    assert opened[3] == "Local\\pedacito-web-4242-abcd1234"
    assert opened[2] == processes._JOB_TERMINATE | processes._SYNCHRONIZE
    terminated = [c for c in k32.calls if c[0] == "terminate"][0]
    assert terminated[1] == opened[1], terminated      # the job we just opened
    assert terminated[2] == 1, terminated              # nonzero exit code
    assert not runs, "taskkill ran even though the job was reaped"
    assert not k32.leaked


def test_stop_without_a_job_and_no_birth_time_kills_nothing(monkeypatch):
    """Old records carry no exact birth time: without provable identity the
    fallback must not kill the leader (or anything) by pid -- it reports an
    honest partial note instead."""
    k32 = _FakeKernel32(open_ok=False)       # OpenJobObjectW: ERROR_FILE_NOT_FOUND
    _pretend_windows(monkeypatch, k32)

    note = processes._terminate_pid(4242, "Local\\pedacito-web-4242-abcd1234")

    assert "no recorded birth time" in note, note
    assert not _terminated_pids(k32), "an unverifiable leader was killed"
    assert not k32.leaked


def test_stop_uses_the_job_recorded_for_the_process(monkeypatch, tmp_path):
    """`stop` must hand the record's job to the terminator, not just the pid.

    The pid-only call is what shipped first and what the Windows runner
    rejected: the leader was already gone, so a pid alone could not reach the
    surviving server. A record written by any pedacito process carries the job
    name, which is the whole point of naming the kernel object.
    """
    seen: list[tuple] = []
    monkeypatch.setattr(processes, "_terminate_pid",
                        lambda pid, job="", **_kw: seen.append((pid, job)))
    monkeypatch.setattr(processes, "_alive", lambda _pid: False)  # leader exited
    root = tmp_path / "proj"
    root.mkdir()
    processes._write_record(root, {
        "name": "web", "command": "npm run dev", "pid": 4242,
        "cwd": str(root), "log": str(root / "web.log"), "url": "",
        "job": "Local\\pedacito-web-4242-abcd1234",
        "started": 1.0, "status": "running"})

    out = processes.stop(root, "web")

    assert out.startswith("stopped web"), out
    assert seen == [(4242, "Local\\pedacito-web-4242-abcd1234")], seen
    stopped = processes._read_record(root / ".pedacito" / "procs" / "web.json")
    assert stopped["status"] == "stopped"


def test_stop_hands_the_creation_time_to_the_terminator(monkeypatch, tmp_path):
    """The orphan walk needs the leader's creation time to tell family from
    pid reuse. `stop` must pass what the record carries -- and old records
    without `born` still have `started` as a floor."""
    seen: list[dict] = []
    monkeypatch.setattr(
        processes, "_terminate_pid",
        lambda pid, job="", **kw: seen.append({"pid": pid, "job": job, **kw}))
    monkeypatch.setattr(processes, "_alive", lambda _pid: False)
    root = tmp_path / "proj"
    root.mkdir()
    processes._write_record(root, {
        "name": "web", "command": "npm run dev", "pid": 4242,
        "cwd": str(root), "log": str(root / "web.log"), "url": "",
        "job": "", "job_note": "AssignProcessToJobObject failed (error 5)",
        "started": 100.0, "born": 99.5, "status": "running"})

    processes.stop(root, "web")

    assert seen == [{"pid": 4242, "job": "", "started": 100.0, "born": 99.5}], seen


def test_stop_hands_exact_identity_and_descendants_to_the_terminator(
        monkeypatch, tmp_path):
    """`stop` passes the exact FILETIME (`born_ft`) and the captured
    `[pid, FILETIME]` descendants, which the fail-closed fallback needs."""
    seen: list[dict] = []
    monkeypatch.setattr(
        processes, "_terminate_pid",
        lambda pid, job="", **kw: seen.append({"pid": pid, "job": job, **kw}))
    monkeypatch.setattr(processes, "_alive", lambda _pid: False)
    root = tmp_path / "proj"
    root.mkdir()
    ft = 133000000000000000
    processes._write_record(root, {
        "name": "web", "command": "npm run dev", "pid": 4242,
        "cwd": str(root), "log": str(root / "web.log"), "url": "",
        "job": "", "started": 100.0, "born": 99.5, "born_ft": ft,
        "descendants": [[5000, ft + 1]],
        "status": "running"})

    processes.stop(root, "web")

    assert seen == [{"pid": 4242, "job": "", "started": 100.0, "born": 99.5,
                     "born_ft": ft, "win_descendants": [[5000, ft + 1]]}], seen


def test_stop_reports_partial_when_descendants_cannot_be_verified(
        monkeypatch, tmp_path):
    """`stop` must not claim success when a descendant could not be reaped: it
    returns an honest incomplete note and persists `stop_incomplete` (not a
    clean `stopped`) with the note."""
    k32 = _FakeKernel32(processes=[
        (4242, 1, _ft(_ORPHAN_BORN)),               # leader still ours
        (5000, 1, _ft(_ORPHAN_BORN + 9999)),        # captured pid now reused
    ])
    _pretend_windows(monkeypatch, k32)
    monkeypatch.setattr(processes, "_alive", lambda _pid: False)
    root = tmp_path / "proj"
    root.mkdir()
    processes._write_record(root, {
        "name": "web", "command": "npm run dev", "pid": 4242,
        "cwd": str(root), "log": str(root / ".pedacito" / "procs" / "web.log"),
        "url": "", "job": "", "started": _ORPHAN_BORN, "born": _ORPHAN_BORN,
        "born_ft": _ft(_ORPHAN_BORN),
        "descendants": [[5000, _ft(_ORPHAN_BORN + 10)]],
        "status": "running"})

    out = processes.stop(root, "web")

    assert out.startswith("stopped web (pid 4242) [incomplete:"), out
    assert "descendant pid 5000 not verified" in out, out
    rec = processes._read_record(root / ".pedacito" / "procs" / "web.json")
    assert rec["status"] == "stop_incomplete", rec
    assert rec["stop_note"], rec
    assert "stopped" not in rec, "no clean `stopped` timestamp on partial"


def test_stop_partial_keeps_cleanup_and_child_and_supports_retry(
        monkeypatch, tmp_path):
    """A still-live but unverified leader must not be recorded as cleanly
    stopped: keep the cleanup registration and child tracking, persist
    `stop_incomplete` + `stop_note` (no `stopped` timestamp), and let a later
    stop re-attempt safely."""
    k32 = _FakeKernel32(processes=[
        (4242, 1, _ft(_ORPHAN_BORN + 9999)),   # leader alive but not ours
    ])
    _pretend_windows(monkeypatch, k32)
    monkeypatch.setattr(processes, "_alive", lambda _pid: True)

    unregistered: list = []
    monkeypatch.setattr(processes, "_unregister_dev_server",
                        lambda rid: unregistered.append(rid))

    class Child:
        pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(processes, "_CHILDREN", {4242: Child()})

    root = tmp_path / "proj"
    root.mkdir()
    processes._write_record(root, {
        "name": "web", "command": "npm run dev", "pid": 4242,
        "cwd": str(root), "log": str(root / ".pedacito" / "procs" / "web.log"),
        "url": "", "job": "", "started": _ORPHAN_BORN, "born": _ORPHAN_BORN,
        "born_ft": _ft(_ORPHAN_BORN), "resource_id": "dev-server-web",
        "status": "running"})

    out = processes.stop(root, "web")

    assert out.startswith("stopped web (pid 4242) [incomplete:"), out
    rec = processes._read_record(root / ".pedacito" / "procs" / "web.json")
    assert rec["status"] == "stop_incomplete", rec
    assert rec["stop_note"], rec
    assert "stopped" not in rec, "no clean `stopped` timestamp on partial"
    assert not unregistered, "cleanup registration was dropped on partial"
    assert 4242 in processes._CHILDREN, "child tracking was dropped on partial"

    # A retry re-attempts safely and stays incomplete while the leader is
    # still unverified.
    out2 = processes.stop(root, "web")
    assert out2.startswith("stopped web (pid 4242) [incomplete:"), out2
    rec2 = processes._read_record(root / ".pedacito" / "procs" / "web.json")
    assert rec2["status"] == "stop_incomplete", rec2


def test_stop_complete_unregisters_and_forgets_child(monkeypatch, tmp_path):
    """A clean stop preserves the existing complete bookkeeping: it drops the
    cleanup registration, reaps/forgets the child, and records `stopped`."""
    monkeypatch.setattr(processes, "_terminate_pid", lambda *a, **k: "")
    monkeypatch.setattr(processes, "_alive", lambda _pid: True)
    unregistered: list = []
    monkeypatch.setattr(processes, "_unregister_dev_server",
                        lambda rid: unregistered.append(rid))

    class Child:
        pid = 4242

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(processes, "_CHILDREN", {4242: Child()})

    root = tmp_path / "proj"
    root.mkdir()
    processes._write_record(root, {
        "name": "web", "command": "npm run dev", "pid": 4242,
        "cwd": str(root), "log": str(root / ".pedacito" / "procs" / "web.log"),
        "url": "", "job": "Local\\x", "started": 1.0, "born": 1.0,
        "resource_id": "dev-server-web", "status": "running"})

    out = processes.stop(root, "web")

    assert out == "stopped web (pid 4242)", out
    rec = processes._read_record(root / ".pedacito" / "procs" / "web.json")
    assert rec["status"] == "stopped", rec
    assert rec["stopped"], rec
    assert "stop_note" not in rec
    assert unregistered == ["dev-server-web"], unregistered
    assert 4242 not in processes._CHILDREN


# The runner's refusal leaves no job, and the launcher exits first: the walk
# below is the ONLY thing that reaches the orphaned server there.
_ORPHAN_BORN = 1_700_000_000.0


def _ft(secs: float) -> int:
    """Unix seconds -> the FILETIME the snapshot reports (100ns since 1601)."""
    return int((secs + 11644473600) * 10_000_000)


def _terminated_pids(k32) -> list[int]:
    """Pids the fake was asked to TerminateProcess, in call order."""
    h2pid = {c[1]: c[3] for c in k32.calls if c[0] == "open_process" and c[1]}
    return [h2pid[c[1]] for c in k32.calls if c[0] == "terminate_process"]


def _orphan_proc_table():
    # The leader 4242 is DEAD (absent from the snapshot), exactly like the
    # launcher that spawned the real server and exited.
    return [
        (5000, 4242, _ft(_ORPHAN_BORN + 10)),   # the orphaned server
        (5001, 5000, _ft(_ORPHAN_BORN + 20)),   # its worker
        (6000, 1, _ft(_ORPHAN_BORN + 30)),      # unrelated: another parent
        (7000, 4242, _ft(_ORPHAN_BORN - 10)),   # pid reuse: predates leader
    ]


def test_windows_descendants_walk_past_a_dead_leader(monkeypatch):
    """Windows keeps ParentProcessId after the parent dies, so the snapshot
    chain outlives the leader -- the property `taskkill /T` lacks. But a pid
    can be REUSED after its holder exits: a candidate created before the
    leader is not family, it is a stranger wearing the same pid."""
    k32 = _FakeKernel32(processes=_orphan_proc_table())
    _pretend_windows(monkeypatch, k32)

    assert processes._windows_descendants(4242, _ft(_ORPHAN_BORN)) == [5000, 5001]
    assert not k32.leaked, "snapshot/process handles were not closed"


def test_windows_descendant_identities_returns_exact_filetimes(monkeypatch):
    """`start` stores `[pid, FILETIME]` pairs -- exact, not a lossy float --
    so `stop` can later tell a recycled pid from the real descendant."""
    k32 = _FakeKernel32(processes=_orphan_proc_table())
    _pretend_windows(monkeypatch, k32)

    ids = processes._windows_descendant_identities(4242, _ft(_ORPHAN_BORN))

    assert ids == [
        [5000, _ft(_ORPHAN_BORN + 10)],
        [5001, _ft(_ORPHAN_BORN + 20)],
    ], ids
    assert not k32.leaked


def test_stop_with_a_live_leader_reaps_only_its_walk_found_orphans(monkeypatch):
    """No job, leader alive and proven ours: the live walk reaches its real
    descendants and reaps each by a validated handle, leaving an unrelated
    process and a pid that predates the leader untouched."""
    k32 = _FakeKernel32(processes=[
        (4242, 1, _ft(_ORPHAN_BORN)),           # the live leader (ours)
        (5000, 4242, _ft(_ORPHAN_BORN + 10)),   # its server
        (5001, 5000, _ft(_ORPHAN_BORN + 20)),   # its worker
        (6000, 1, _ft(_ORPHAN_BORN + 30)),      # unrelated
        (7000, 4242, _ft(_ORPHAN_BORN - 10)),   # pid reuse: predates leader
    ])
    _pretend_windows(monkeypatch, k32)

    note = processes._terminate_pid(4242, "", born_ft=_ft(_ORPHAN_BORN))

    assert note and "not provably fully reaped" in note, note
    assert _terminated_pids(k32) == [4242, 5000, 5001]
    assert not k32.leaked


def test_stop_kills_captured_descendants_through_vanished_intermediates(
        monkeypatch):
    """The venvlauncher chain: leader 4242 -> launcher 4243 -> real server.
    By stop time 4242 and 4243 are DEAD, so the live walk finds nothing -- the
    identities captured at start, while the chain was alive, are the only
    thing that reaches the survivors, and only after an exact re-validation."""
    k32 = _FakeKernel32(processes=[
        (5000, 4243, _ft(_ORPHAN_BORN + 10)),   # the orphaned server
        (5001, 5000, _ft(_ORPHAN_BORN + 20)),   # its worker
        (6000, 1, _ft(_ORPHAN_BORN + 30)),      # unrelated: another parent
    ], dead_pids={4242, 4243})
    _pretend_windows(monkeypatch, k32)

    note = processes._terminate_pid(
        4242, "", born_ft=_ft(_ORPHAN_BORN),
        win_descendants=[[5000, _ft(_ORPHAN_BORN + 10)],
                         [5001, _ft(_ORPHAN_BORN + 20)]])

    assert note and "not provably fully reaped" in note, note
    assert _terminated_pids(k32) == [5000, 5001]
    assert not k32.leaked


def test_stop_rejects_a_pid_reused_within_one_second(monkeypatch):
    """The old 1-second tolerance could kill a stranger whose pid was reused
    almost immediately; the exact FILETIME compare must reject it."""
    k32 = _FakeKernel32(processes=[
        (4242, 1, _ft(_ORPHAN_BORN)),             # our leader, still valid
        (5000, 1, _ft(_ORPHAN_BORN + 10) + 1),    # reused: 100ns off
    ])
    _pretend_windows(monkeypatch, k32)

    note = processes._terminate_pid(
        4242, "", born_ft=_ft(_ORPHAN_BORN),
        win_descendants=[[5000, _ft(_ORPHAN_BORN + 10)]])

    assert "descendant pid 5000 not verified" in note, note
    assert _terminated_pids(k32) == [4242]     # leader only; 5000 rejected
    assert not k32.leaked


def test_stop_skips_a_stale_leader_whose_pid_was_reused(monkeypatch):
    """A stale record whose leader pid now belongs to a stranger: the leader
    fails exact validation, its walk is skipped, and nothing is killed."""
    k32 = _FakeKernel32(processes=[
        (4242, 1, _ft(_ORPHAN_BORN + 9999)),        # unrelated now
        (5000, 4242, _ft(_ORPHAN_BORN + 10000)),    # its (unrelated) child
    ])
    _pretend_windows(monkeypatch, k32)

    note = processes._terminate_pid(4242, "", born_ft=_ft(_ORPHAN_BORN))

    assert "leader pid 4242 not verified" in note, note
    assert not _terminated_pids(k32), "a stale leader's tree was killed"
    assert not k32.leaked


def test_stop_reports_partial_when_leader_vanished_with_no_captured_descendants(
        monkeypatch):
    """A vanished leader with no captured identities cannot prove the tree is
    empty (an uncaptured descendant behind a dead intermediate may remain), so
    the fallback must report partial, not a clean stop."""
    k32 = _FakeKernel32(processes=[
        (6000, 1, _ft(_ORPHAN_BORN + 30)),   # unrelated
    ], dead_pids={4242})
    _pretend_windows(monkeypatch, k32)

    note = processes._terminate_pid(4242, "", born_ft=_ft(_ORPHAN_BORN))

    assert note and "not provably fully reaped" in note, note
    assert not _terminated_pids(k32)
    assert not k32.leaked


def test_leader_is_walked_before_it_is_terminated(monkeypatch):
    """The live walk must capture descendants while the validated leader handle
    is still open (before TerminateProcess), so a pid reused after the kill
    cannot masquerade as our tree."""
    k32 = _FakeKernel32(processes=[
        (4242, 1, _ft(_ORPHAN_BORN)),           # the live leader
        (5000, 4242, _ft(_ORPHAN_BORN + 10)),   # its child
    ])
    _pretend_windows(monkeypatch, k32)

    processes._terminate_pid(4242, "", born_ft=_ft(_ORPHAN_BORN))

    h2pid = {c[1]: c[3] for c in k32.calls if c[0] == "open_process" and c[1]}
    leader_handle = next(h for h, pid in h2pid.items() if pid == 4242)
    snap_idx = next(i for i, c in enumerate(k32.calls) if c[0] == "snapshot")
    term_idx = next(i for i, c in enumerate(k32.calls)
                    if c[0] == "terminate_process" and c[1] == leader_handle)
    assert snap_idx < term_idx, (
        "the snapshot walk must happen before the leader is terminated; "
        f"calls={k32.calls}")
    assert _terminated_pids(k32) == [4242, 5000]
    assert not k32.leaked


def test_terminate_verified_refuses_unknown_birth(monkeypatch):
    k32 = _FakeKernel32(processes=[(5000, 1, _ft(_ORPHAN_BORN + 10))])
    _pretend_windows(monkeypatch, k32)

    assert processes._terminate_verified(5000, 0) == "unverified"
    assert not _terminated_pids(k32)


def test_terminate_verified_refuses_unreadable_creation_time(monkeypatch):
    k32 = _FakeKernel32(processes=[(5000, 1, 0)])  # unreadable creation time
    _pretend_windows(monkeypatch, k32)

    assert processes._terminate_verified(
        5000, _ft(_ORPHAN_BORN + 10)) == "unverified"
    assert not _terminated_pids(k32)


def test_terminate_verified_validates_and_kills_on_one_handle(monkeypatch):
    """The check and the kill share ONE handle, so a pid-reuse window cannot
    be straddled: GetProcessTimes and TerminateProcess both target the handle
    that OpenProcess returned, and nothing else is opened."""
    k32 = _FakeKernel32(processes=[(5000, 1, _ft(_ORPHAN_BORN + 10))])
    _pretend_windows(monkeypatch, k32)

    assert processes._terminate_verified(
        5000, _ft(_ORPHAN_BORN + 10)) == "reaped"

    opens = [c for c in k32.calls if c[0] == "open_process"]
    assert len(opens) == 1, k32.calls
    h = opens[0][1]
    assert opens[0][2] == (processes._PROCESS_QUERY_LIMITED_INFORMATION
                           | processes._PROCESS_TERMINATE)
    times = [c for c in k32.calls if c[0] == "times"]
    assert times and times[0][1] == h and times[0][2] == _ft(_ORPHAN_BORN + 10)
    terms = [c for c in k32.calls if c[0] == "terminate_process"]
    assert terms == [("terminate_process", h, 1)], terms
    assert h in k32.closed
    assert not k32.leaked


def test_forget_job_closes_the_retained_handle(monkeypatch):
    k32 = _FakeKernel32()
    _pretend_windows(monkeypatch, k32)

    processes._JOB_HANDLES["Local\\pedacito-x"] = 101
    processes._forget_job("Local\\pedacito-x")

    assert "Local\\pedacito-x" not in processes._JOB_HANDLES
    assert 101 in k32.closed


def test_list_procs_closes_the_job_handle_only_when_the_job_is_empty(
        monkeypatch, tmp_path):
    """A fully-exited server (empty job) releases its handle; a leader that
    exited while descendants remain must keep it (see the next test)."""
    k32 = _FakeKernel32(active_processes=0)
    _pretend_windows(monkeypatch, k32)
    monkeypatch.setattr(processes, "_alive", lambda _pid: False)

    root = tmp_path / "proj"
    root.mkdir()
    job = "Local\\pedacito-web-4242-x"
    processes._JOB_HANDLES[job] = 101
    processes._write_record(root, {
        "name": "web", "command": "npm run dev", "pid": 4242,
        "cwd": str(root), "log": str(root / ".pedacito" / "procs" / "web.log"),
        "url": "", "job": job, "started": 1.0, "born": 1.0, "status": "running"})

    recs = processes.list_procs(root)

    assert recs and recs[0]["status"] == "exited"
    assert job not in processes._JOB_HANDLES
    assert 101 in k32.closed


def test_list_procs_keeps_the_job_handle_while_members_remain(
        monkeypatch, tmp_path):
    """The leader exiting is exactly when orphans may still be members: the
    handle must survive list_procs so a later stop can still terminate them."""
    k32 = _FakeKernel32(active_processes=2)
    _pretend_windows(monkeypatch, k32)
    monkeypatch.setattr(processes, "_alive", lambda _pid: False)

    root = tmp_path / "proj"
    root.mkdir()
    job = "Local\\pedacito-web-4242-x"
    processes._JOB_HANDLES[job] = 101
    processes._write_record(root, {
        "name": "web", "command": "npm run dev", "pid": 4242,
        "cwd": str(root), "log": str(root / ".pedacito" / "procs" / "web.log"),
        "url": "", "job": job, "started": 1.0, "born": 1.0, "status": "running"})

    recs = processes.list_procs(root)

    assert recs and recs[0]["status"] == "exited"
    assert job in processes._JOB_HANDLES      # members remain: keep it
    assert 101 not in k32.closed


def test_list_procs_then_stop_uses_the_retained_job_handle(
        monkeypatch, tmp_path):
    """After list_procs reconciles a dead leader (job non-empty), a later stop
    terminates through the retained handle -- not an open-by-name that would
    fail once the starter process is gone."""
    k32 = _FakeKernel32(active_processes=2)
    _pretend_windows(monkeypatch, k32)
    monkeypatch.setattr(processes, "_alive", lambda _pid: False)

    root = tmp_path / "proj"
    root.mkdir()
    job = "Local\\pedacito-web-4242-x"
    processes._JOB_HANDLES[job] = 101
    processes._write_record(root, {
        "name": "web", "command": "npm run dev", "pid": 4242,
        "cwd": str(root), "log": str(root / ".pedacito" / "procs" / "web.log"),
        "url": "", "job": job, "started": 1.0, "born": 1.0,
        "status": "running"})

    processes.list_procs(root)
    assert job in processes._JOB_HANDLES

    out = processes.stop(root, "web")

    assert out.startswith("stopped web"), out
    terminated = [c for c in k32.calls if c[0] == "terminate"]
    assert terminated and terminated[0][1] == 101, k32.calls
    assert not [c for c in k32.calls if c[0] == "open_job"]
    assert job not in processes._JOB_HANDLES
    assert not k32.leaked


def test_resume_child_resumes_only_the_suspended_childs_threads(monkeypatch):
    """Popen drops the initial thread handle, so the resume finds the thread
    by owner pid in the snapshot -- and must not resume anyone else's."""
    class P:
        pid = 4242

    k32 = _FakeKernel32(threads=[(900, 4242), (901, 1111)])
    _pretend_windows(monkeypatch, k32)
    assert processes._resume_child(P()) is True
    resumed = [c for c in k32.calls if c[0] == "resume"]
    assert [c[2] for c in resumed] == [900], resumed
    opened = [c for c in k32.calls if c[0] == "open_thread"]
    assert opened and all(c[2] == 0x2 for c in opened), opened
    assert not k32.leaked

    k32 = _FakeKernel32(threads=[(900, 4242)], resume_ok=False)
    _pretend_windows(monkeypatch, k32)
    assert processes._resume_child(P()) is False
    assert not k32.leaked

    k32 = _FakeKernel32(threads=[])                 # nothing to resume
    _pretend_windows(monkeypatch, k32)
    assert processes._resume_child(P()) is False

    monkeypatch.setattr(processes.os, "name", "posix")
    assert processes._resume_child(P()) is True     # no-op off Windows


def test_start_wraps_the_child_before_it_runs_and_records_the_diagnostics(
        monkeypatch, tmp_path):
    """Order is the fix: spawn SUSPENDED, wrap in the job, then resume. A
    child that runs first can spawn descendants outside the job forever,
    because membership is inherited only at creation."""
    order: list[tuple] = []
    ft = _ft(_ORPHAN_BORN)
    k32 = _FakeKernel32(processes=[(4242, 1, ft)])
    _pretend_windows(monkeypatch, k32)

    class P:
        pid = 4242

        def poll(self):
            return None

    def fake_spawn(argv, *, cwd, env, stdout, stderr, suspended=False):
        order.append(("spawn", suspended))
        return P()

    def fake_wrap(pid, name):
        order.append(("wrap", pid))
        return "Local\\pedacito-web-4242-x", ""

    def fake_resume(p):
        order.append(("resume", p.pid))
        return True

    monkeypatch.setattr(processes.proc, "spawn", fake_spawn)
    monkeypatch.setattr(processes, "_job_wrap", fake_wrap)
    monkeypatch.setattr(processes, "_resume_child", fake_resume)
    root = tmp_path / "proj"
    root.mkdir()

    rec = processes.start(root, "npm run dev", name="web", wait=0)

    assert order == [("spawn", True), ("wrap", 4242), ("resume", 4242)], order
    assert rec["job"] == "Local\\pedacito-web-4242-x"
    assert rec["job_note"] == ""
    assert rec["born"] == pytest.approx(_ORPHAN_BORN, abs=1e-3)
    processes._forget_child(rec["pid"])


def test_start_respawns_running_when_the_suspended_child_cannot_resume(
        monkeypatch, tmp_path):
    """A frozen server is worse than a racy job wrap: kill the suspended
    child and respawn it running, wrapping best effort. The record must
    describe the SECOND child, including its own creation time."""
    ft = _ft(_ORPHAN_BORN)
    k32 = _FakeKernel32(processes=[(4242, 1, ft),
                                   (5555, 1, ft + 10_000_000)])
    _pretend_windows(monkeypatch, k32)
    spawns: list[tuple[int, bool]] = []
    wraps: list[int] = []

    class P:
        def __init__(self, pid):
            self.pid = pid
            self.killed = False

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return None

    children = [P(4242), P(5555)]

    def fake_spawn(argv, *, cwd, env, stdout, stderr, suspended=False):
        p = children[len(spawns)]
        spawns.append((p.pid, suspended))
        return p

    def fake_wrap(pid, name):
        wraps.append(pid)
        return f"Local\\pedacito-web-{pid}-x", ""

    monkeypatch.setattr(processes.proc, "spawn", fake_spawn)
    monkeypatch.setattr(processes, "_job_wrap", fake_wrap)
    monkeypatch.setattr(processes, "_resume_child", lambda p: False)
    root = tmp_path / "proj"
    root.mkdir()

    rec = processes.start(root, "npm run dev", name="web", wait=0)

    assert spawns == [(4242, True), (5555, False)], spawns
    assert children[0].killed, "the frozen child was left behind"
    assert wraps == [4242, 5555], wraps
    assert rec["pid"] == 5555
    assert rec["born"] == pytest.approx(_ORPHAN_BORN + 1, abs=1e-3)
    processes._forget_child(rec["pid"])


def test_browser_resolves_a_mocked_windows_chrome_path(tmp_path, monkeypatch):
    # Joined under tmp_path: an absolute C:\ string would escape tmp_path on
    # native Windows and create real Program Files directories.
    fake = tmp_path / "Program Files" / "Google" / "Chrome" / "Application" / "chrome.exe"
    fake.parent.mkdir(parents=True, exist_ok=True)
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
