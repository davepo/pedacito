"""WP-4005 (D14): the Windows OS sandbox (Job Object + Restricted Token) and
the unattended hard gate.

The mocked tests run on every platform; the live tests are Windows-only and
guarded by ``skipif``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import sandbox, sandbox_win  # noqa: E402
from pedacito.config import Config  # noqa: E402
from pedacito.sandbox import SandboxRefused  # noqa: E402


# --------------------------------------------------------------------------- #
# pure helpers                                                                #
# --------------------------------------------------------------------------- #

def test_env_block_is_double_nul_terminated_utf16():
    data = sandbox_win._env_block({"A": "1", "B": "2"})
    assert data.decode("utf-16-le") == "A=1\0B=2\0\0"


def test_command_line_uses_list2cmdline():
    argv = ["python", "-c", "print('a b')"]
    assert sandbox_win._command_line(argv) == subprocess.list2cmdline(argv)


# --------------------------------------------------------------------------- #
# probe (mocked)                                                              #
# --------------------------------------------------------------------------- #

def _mock_ok(monkeypatch):
    monkeypatch.setattr(sandbox_win, "_check_job_object", lambda: (True, ""))
    monkeypatch.setattr(sandbox_win, "_check_restricted_token",
                        lambda: (True, "", True))
    monkeypatch.setattr(sandbox_win, "_check_appcontainer", lambda: True)
    monkeypatch.setattr(sandbox_win, "_check_child_launch", lambda: (True, ""))


def test_probe_windows_success(monkeypatch):
    _mock_ok(monkeypatch)
    caps = sandbox_win._probe_windows()
    assert caps.available is True
    assert caps.job_object is True
    assert caps.restricted_token is True
    assert caps.low_integrity is True
    assert caps.appcontainer is True
    assert caps.reason == ""


def test_probe_windows_privilege_not_held(monkeypatch):
    _mock_ok(monkeypatch)
    monkeypatch.setattr(
        sandbox_win, "_check_child_launch",
        lambda: (False, "CreateProcessAsUserW failed (winerror 1314)"))
    caps = sandbox_win._probe_windows()
    assert caps.available is False
    assert "1314" in caps.reason


def test_probe_windows_access_denied(monkeypatch):
    _mock_ok(monkeypatch)
    monkeypatch.setattr(
        sandbox_win, "_check_child_launch",
        lambda: (False, "CreateProcessAsUserW failed (winerror 5)"))
    caps = sandbox_win._probe_windows()
    assert caps.available is False
    assert "5" in caps.reason


def test_probe_windows_job_object_failure(monkeypatch):
    monkeypatch.setattr(sandbox_win, "_check_job_object",
                        lambda: (False, "CreateJobObjectW failed (winerror 5)"))
    monkeypatch.setattr(sandbox_win, "_check_restricted_token",
                        lambda: (True, "", True))
    monkeypatch.setattr(sandbox_win, "_check_appcontainer", lambda: False)
    monkeypatch.setattr(sandbox_win, "_check_child_launch", lambda: (True, ""))
    caps = sandbox_win._probe_windows()
    assert caps.available is False
    assert caps.job_object is False


def test_probe_non_windows_reports_unavailable():
    if os.name == "nt":
        pytest.skip("runs the non-Windows branch")
    caps = sandbox_win.probe()
    assert caps.available is False
    assert "not Windows" in caps.reason


# --------------------------------------------------------------------------- #
# enforce_unattended_gate (mocked)                                            #
# --------------------------------------------------------------------------- #

def _no_sandbox(monkeypatch):
    monkeypatch.setattr(sandbox, "backend_name", lambda: "")


def test_gate_raises_for_unattended_workspace_write(monkeypatch):
    _no_sandbox(monkeypatch)
    cfg = Config()
    cfg.sandbox_strict = True
    with pytest.raises(SandboxRefused):
        sandbox.enforce_unattended_gate(cfg, permission_mode="workspace-write")


def test_gate_raises_for_unattended_danger(monkeypatch):
    _no_sandbox(monkeypatch)
    cfg = Config()
    cfg.sandbox_strict = True
    with pytest.raises(SandboxRefused):
        sandbox.enforce_unattended_gate(cfg, permission_mode="danger")


def test_gate_raises_for_background(monkeypatch):
    _no_sandbox(monkeypatch)
    cfg = Config()
    cfg.sandbox_strict = True
    with pytest.raises(SandboxRefused):
        sandbox.enforce_unattended_gate(cfg, background=True)


def test_gate_raises_for_never_approval_with_sandbox_on(monkeypatch):
    _no_sandbox(monkeypatch)
    cfg = Config()
    cfg.sandbox = True
    with pytest.raises(SandboxRefused):
        sandbox.enforce_unattended_gate(cfg, permission_mode="workspace-write",
                                        approval_policy="never")


def test_gate_passes_when_sandbox_active(monkeypatch):
    monkeypatch.setattr(sandbox, "backend_name", lambda: "win_restricted")
    cfg = Config()
    cfg.sandbox_strict = True
    sandbox.enforce_unattended_gate(cfg, permission_mode="workspace-write")


def test_gate_passes_for_interactive_ask(monkeypatch):
    _no_sandbox(monkeypatch)
    cfg = Config()
    cfg.sandbox_strict = True
    sandbox.enforce_unattended_gate(cfg, permission_mode="ask")


def test_gate_passes_when_sandbox_off(monkeypatch):
    _no_sandbox(monkeypatch)
    cfg = Config()
    cfg.sandbox = False
    sandbox.enforce_unattended_gate(cfg, permission_mode="workspace-write",
                                    approval_policy="never")


def test_backend_name_reset_helper(monkeypatch):
    monkeypatch.setattr(sandbox, "_win_probed", True)
    monkeypatch.setattr(sandbox, "_win_available", True)
    sandbox.reset_win_probe()
    assert sandbox._win_probed is False
    assert sandbox._win_available is False


# --------------------------------------------------------------------------- #
# run_sandboxed orchestration (mocked)                                        #
# --------------------------------------------------------------------------- #

class _FakeChild:
    def __init__(self, stdout_chunks=(), exit_after=1, never_exit=False):
        self.stdout_chunks = list(stdout_chunks)
        self.exit_after = exit_after
        self.never_exit = never_exit
        self._polls = 0
        self.terminated = False
        self.closed = False

    def poll(self):
        self._polls += 1
        if self.never_exit:
            return None
        if self.exit_after is not None and self._polls >= self.exit_after:
            return 0
        return None

    def read_stdout(self):
        return self.stdout_chunks.pop(0) if self.stdout_chunks else b""

    def read_stderr(self):
        return b""

    def wait(self, timeout):
        return 0

    def terminate(self):
        self.terminated = True

    def close(self):
        self.closed = True


def _run(argv, child, timeout=None, cancel_event=None, idle_timeout=0.0,
         monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(sandbox_win, "_spawn_child",
                            lambda *a, **k: child)
    return sandbox_win._run_nt(
        argv, cwd=Path(tempfile.mkdtemp()), env={"A": "1"},
        timeout=timeout, idle_timeout=idle_timeout, max_output=8000,
        stdin_text=None, cancel_event=cancel_event)


def test_run_happy_path(monkeypatch):
    child = _FakeChild(stdout_chunks=[b"hello_sandbox\n"], exit_after=1)
    rc, out, err, elapsed, stopped = _run(
        ["python", "-c", "print(1)"], child, monkeypatch=monkeypatch)
    assert rc == 0
    assert "hello_sandbox" in out
    assert elapsed >= 0.0
    assert stopped == ""
    assert child.closed is True


def test_run_timeout_terminates_child(monkeypatch):
    child = _FakeChild(never_exit=True)
    rc, out, err, _el, stopped = _run(["python", "-c", "x"], child,
                                      timeout=0.05, monkeypatch=monkeypatch)
    assert rc is None
    assert stopped == "timeout"
    assert child.terminated is True
    assert child.closed is True


def test_run_cancel_event_terminates_child(monkeypatch):
    child = _FakeChild(never_exit=True)
    ev = threading.Event()
    ev.set()
    rc, out, err, _el, stopped = _run(["python", "-c", "x"], child,
                                      cancel_event=ev, monkeypatch=monkeypatch)
    assert rc is None
    assert stopped == "cancelled"
    assert child.terminated is True
    assert child.closed is True


def test_run_idle_timeout_terminates_child(monkeypatch):
    child = _FakeChild(never_exit=True)
    rc, out, err, _el, stopped = _run(["python", "-c", "x"], child,
                                      idle_timeout=0.05,
                                      monkeypatch=monkeypatch)
    assert rc is None
    assert stopped == "idle"
    assert child.terminated is True


def test_run_spawn_failure_returns_error(monkeypatch):
    def boom(*a, **k):
        raise OSError(5, "CreateProcessAsUserW failed (winerror 5)")
    monkeypatch.setattr(sandbox_win, "_spawn_child", boom)
    rc, out, err, _el, stopped = sandbox_win._run_nt(
        ["python"], cwd=Path(tempfile.mkdtemp()), env={}, timeout=None,
        idle_timeout=0.0, max_output=8000, stdin_text=None, cancel_event=None)
    assert rc is None and "winerror" in err
    assert stopped == "error"


# --------------------------------------------------------------------------- #
# live Windows probe + run (only on Windows)                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(sys.platform != "win32",
                    reason="Windows-only live sandbox test")
def test_live_probe_returns_valid_capabilities():
    caps = sandbox_win.probe()
    assert caps.available in (True, False)
    assert isinstance(caps.job_object, bool)
    assert isinstance(caps.restricted_token, bool)
    assert isinstance(caps.low_integrity, bool)
    if not caps.available:
        assert caps.reason


@pytest.mark.skipif(sys.platform != "win32",
                    reason="Windows-only live sandbox test")
def test_live_run_and_cancel():
    caps = sandbox_win.probe()
    if not caps.available:
        cfg = Config()
        cfg.sandbox_strict = True
        with pytest.raises(SandboxRefused):
            sandbox.enforce_unattended_gate(cfg,
                                            permission_mode="workspace-write")
        return
    cwd = Path(tempfile.mkdtemp())
    env = dict(os.environ)
    rc, out, err, _el = sandbox_win.run_sandboxed(
        [sys.executable, "-c", "print('hello_sandbox')"], cwd=cwd, env=env,
        timeout=30)
    assert rc == 0
    assert "hello_sandbox" in out

    # cancellation kills a long-running child cleanly
    ev = threading.Event()
    ev.set()
    rc, out, err, _el = sandbox_win.run_sandboxed(
        [sys.executable, "-c", "import time; time.sleep(30)"], cwd=cwd,
        env=env, timeout=30, cancel_event=ev)
    assert rc is None


# --------------------------------------------------------------------------- #
# C1: safe relabel guard + probe never touches os.getcwd()                    #
# --------------------------------------------------------------------------- #

def test_is_safe_to_relabel_rejects_roots_and_home(tmp_path):
    assert sandbox_win._is_safe_to_relabel(Path.home()) is False
    # The filesystem/drive root.
    assert sandbox_win._is_safe_to_relabel(Path(Path.home().anchor)) is False
    # A private directory under the temp dir is safe to relabel.
    proj = tmp_path / "proj"
    proj.mkdir()
    assert sandbox_win._is_safe_to_relabel(proj) is True


def test_is_safe_to_relabel_rejects_system_dirs(monkeypatch, tmp_path):
    if os.name == "nt":
        base = r"C:\PedacitoFakeSystemRoot"
    else:
        base = "/opt/pedacito-fake-system-root"
    monkeypatch.setitem(os.environ, "SystemRoot", base)
    # A system directory and a descendant are both refused, even if they do
    # not exist on disk (the guard refuses before touching them).
    assert sandbox_win._is_safe_to_relabel(Path(base)) is False
    assert sandbox_win._is_safe_to_relabel(Path(base) / "sub") is False


@pytest.mark.skipif(os.name == "nt",
                    reason="POSIX symlink (Windows covered by live junction test)")
def test_is_safe_to_relabel_rejects_symlink(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    assert sandbox_win._is_safe_to_relabel(link) is False


@pytest.mark.skipif(os.name != "nt", reason="Windows-only probe path")
def test_probe_runs_in_private_tempdir_not_cwd(monkeypatch, tmp_path):
    captured = {}

    def fake_spawn(argv, *, cwd, env, stdin_text=None, writable_dirs=()):
        captured["cwd"] = str(cwd)
        raise RuntimeError("stop")

    monkeypatch.setattr(sandbox_win, "_spawn_child", fake_spawn)
    monkeypatch.chdir(tmp_path)
    ok, err = sandbox_win._check_child_launch()
    assert not ok and "RuntimeError" in err
    # The probe never uses os.getcwd(): it runs in a private temp directory.
    assert captured.get("cwd") and captured["cwd"] != str(tmp_path)
    assert captured["cwd"].startswith(str(Path(tempfile.gettempdir())))


# --------------------------------------------------------------------------- #
# C2/H1 + MEDIUM 2: live Windows junction / guarded paths / stdin             #
# --------------------------------------------------------------------------- #

_WRITE_PROBE = (
    "import sys\n"
    "try:\n"
    "    open(sys.argv[1], 'w').write('pwned')\n"
    "    print('WROTE')\n"
    "except PermissionError:\n"
    "    print('DENIED')\n"
    "except OSError:\n"
    "    print('DENIED')\n"
)

_STDIN_PROBE = (
    "import sys\n"
    "print('STDIN:' + repr(sys.stdin.read()))\n"
)


@pytest.mark.skipif(sys.platform != "win32",
                    reason="Windows-only live junction test")
def test_live_junction_not_relabeled_and_child_cannot_write_outside(tmp_path):
    caps = sandbox_win.probe()
    if not caps.available:
        pytest.skip("Windows sandbox unavailable")
    proj = tmp_path / "proj"
    outside = tmp_path / "outside"
    proj.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    jn = proj / "jn"
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(jn), str(outside)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not jn.exists():
        pytest.skip("cannot create NTFS junction (mklink unavailable)")
    env = dict(os.environ)
    rc, out, err, _el = sandbox_win.run_sandboxed(
        [sys.executable, "-c", _WRITE_PROBE, str(jn / "secret.txt")],
        cwd=proj, env=env, timeout=30)
    assert rc == 0
    assert "DENIED" in out, out
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "secret"


@pytest.mark.skipif(sys.platform != "win32",
                    reason="Windows-only live guarded-paths test")
def test_live_child_cannot_write_guarded_paths(tmp_path):
    caps = sandbox_win.probe()
    if not caps.available:
        pytest.skip("Windows sandbox unavailable")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".git" / "hooks").mkdir(parents=True)
    (proj / ".git" / "hooks" / "pre-commit").write_text(
        "#!/bin/sh\nexit 0\n", encoding="utf-8")
    (proj / ".pedacito").mkdir()
    (proj / ".pedacito" / "policy.toml").write_text(
        "[policy]\n", encoding="utf-8")
    (proj / ".pedacito.toml").write_text(
        "sandbox = true\n", encoding="utf-8")
    env = dict(os.environ)
    for rel in (".git/hooks/pre-commit", ".pedacito/policy.toml",
                ".pedacito.toml"):
        target = proj / rel
        before = target.read_text(encoding="utf-8")
        rc, out, err, _el = sandbox_win.run_sandboxed(
            [sys.executable, "-c", _WRITE_PROBE, str(target)],
            cwd=proj, env=env, timeout=30)
        assert rc == 0, (rel, out, err)
        assert "DENIED" in out, (rel, out)
        assert target.read_text(encoding="utf-8") == before, rel


@pytest.mark.skipif(sys.platform != "win32",
                    reason="Windows-only live stdin test")
def test_live_stdin_text_passed_and_eof(tmp_path):
    caps = sandbox_win.probe()
    if not caps.available:
        pytest.skip("Windows sandbox unavailable")
    cwd = tmp_path / "proj"
    cwd.mkdir()
    env = dict(os.environ)
    rc, out, err, _el = sandbox_win.run_sandboxed(
        [sys.executable, "-c", _STDIN_PROBE], cwd=cwd, env=env, timeout=30,
        stdin_text="hello_stdin")
    assert rc == 0
    assert "STDIN:'hello_stdin'" in out
