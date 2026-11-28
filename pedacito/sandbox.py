"""OS-level sandbox for commands the model (or its edits) cause to run.

    argv, note = sandbox.wrap(argv, cfg, root)      # then run argv as usual

On by default (`sandbox = true`). A wrapped command can read the filesystem but
write only to the project and a private temporary directory; `.git/`,
`.pedacito/` and the project config stay read-only (a command must not plant a
git hook or rewrite its own guard rails); the user's secret directories
(`~/.ssh`, `~/.aws`, `~/.config/pedacito`, ...) are hidden; and the network is
off unless `sandbox_network = true`.

Backends are the unprivileged tools each platform already ships:

    Linux    bubblewrap (`bwrap`)
    macOS    `sandbox-exec` (Seatbelt)
    Windows  `win_restricted` (Job Object + restricted token, Low Integrity)

Where no backend is usable (not installed, user namespaces disabled, Windows,
a Flatpak/containers that forbid nesting) commands run unsandboxed with ONE
notice, so Pedacito behaves the same everywhere; `sandbox_strict = true` turns
that notice into a refusal. A backend is probed once before it is trusted.
Hooks and `auto_verify` are YOUR commands and are never sandboxed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

# Directories under $HOME that hold credentials; hidden inside the sandbox.
SECRET_DIRS = (".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker",
               ".config/gcloud", ".config/gh", ".config/pedacito",
               ".var/app/ai.opencode.opencode/config/pedacito")
SECRET_FILES = (".netrc", ".git-credentials", ".npmrc", ".pypirc",
                ".pgpass", ".my.cnf", ".boto", ".cargo/credentials.toml",
                ".config/git/credentials", ".bash_history", ".zsh_history")
# Inside the project: readable, never writable by a sandboxed command.
GUARDED = (".git", ".pedacito", ".pedacito.toml", "pedacito.toml",
           ".pedacitoignore")

_probe_cache: dict[tuple[str, bool], tuple[bool, str]] = {}
_noticed: set[str] = set()

# Windows Job Object sandbox probe: probed once, cached, reset by tests.
_win_probe_lock = threading.Lock()
_win_probed = False
_win_available = False


def _win_ready() -> bool:
    """True when the built-in Windows Job Object sandbox is usable here."""
    global _win_probed, _win_available
    with _win_probe_lock:
        if not _win_probed:
            try:
                from . import sandbox_win
                _win_available = bool(sandbox_win.probe().available)
            except Exception:
                _win_available = False
            _win_probed = True
        return _win_available


def reset_win_probe() -> None:
    """Forget the cached Windows probe result (tests)."""
    global _win_probed, _win_available
    with _win_probe_lock:
        _win_probed = False
        _win_available = False


def backend_name() -> str:
    if sys.platform == "win32" and _win_ready():
        return "win_restricted"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "bwrap"
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "seatbelt"
    return ""


def windows_sandbox_active(cfg=None) -> bool:
    """True when a wrapped command runs under the Windows Job Object sandbox
    (so `proc.run` should dispatch to `sandbox_win.run_sandboxed`). Respects
    `cfg.sandbox`, so a config with `sandbox = false` never sandboxes."""
    if cfg is not None and not getattr(cfg, "sandbox", True):
        return False
    return backend_name() == "win_restricted"


def shadow_network_isolation(cfg=None) -> bool:
    """Whether the active sandbox can filter outbound network for shadow tests.

    ``bwrap`` (``--unshare-net``) and ``seatbelt`` (``deny network*``) can;
    the Windows ``win_restricted`` backend cannot (its restricted token does
    not filter outbound traffic), and a run with no backend cannot. Port
    selection is not a network namespace, so this is never implied by merely
    allocating loopback ports.
    """
    if cfg is not None and not getattr(cfg, "sandbox", True):
        return False
    return backend_name() in ("bwrap", "seatbelt")


def _home() -> Path:
    # Resolve symlinks (macOS homes, /tmp -> /private/tmp) so the canonical
    # path matches what Seatbelt/bwrap rules compare against.
    return Path(os.path.expanduser("~")).resolve()


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _existing_secrets(root: Path) -> tuple[list[Path], list[Path]]:
    home = _home()
    dirs, files = [], []
    for rel in SECRET_DIRS:
        p = home / rel
        if p.is_dir() and not _under(root, p) and not _under(p, root):
            dirs.append(p)
    for rel in SECRET_FILES:
        p = home / rel
        if p.is_file() and not _under(p, root):
            files.append(p)
    return dirs, files


def _bwrap_args(root: Path, network: bool, extra_rw: list[Path]) -> list[str]:
    args = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid",
            "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp"]
    # The tmpfs hides any interpreter installed under /tmp (a venv, a
    # standalone build): re-bind its prefixes read-only so it still starts.
    tmp = Path("/tmp").resolve()
    for prefix in dict.fromkeys((sys.prefix, sys.base_prefix)):
        p = Path(prefix).resolve()
        if p != tmp and _under(p, tmp):
            args += ["--ro-bind", str(p), str(p)]
    if not network:
        args.append("--unshare-net")
    for p in [root, *extra_rw]:
        args += ["--bind", str(p), str(p)]
    for name in GUARDED:
        g = root / name
        if g.exists():
            args += ["--ro-bind", str(g), str(g)]
    # Secret mounts come AFTER `extra_rw`: a writable path the user listed
    # must never be able to re-bind over a hidden credential (bwrap applies
    # mounts in order, last one wins).
    dirs, files = _existing_secrets(root)
    for d in dirs:
        args += ["--tmpfs", str(d)]
    for f in files:
        args += ["--ro-bind", "/dev/null", str(f)]
    return args + ["--chdir", str(root), "--"]


def _sbpl(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def seatbelt_profile(root: Path, network: bool, extra_rw: list[Path]) -> str:
    """The Seatbelt profile (last matching rule wins)."""
    writable = [root, *extra_rw, Path("/private/tmp"), Path("/private/var/folders")]
    lines = ["(version 1)", "(allow default)"]
    if not network:
        lines.append("(deny network*)")
    lines.append("(deny file-write*)")
    lines.append("(allow file-write* "
                 + " ".join(f"(subpath {_sbpl(str(p))})" for p in writable)
                 + ' (literal "/dev/null") (literal "/dev/tty")'
                 + ' (regex #"^/dev/ttys[0-9]+$"))')
    for name in GUARDED:
        g = root / name
        lines.append(f"(deny file-write* (subpath {_sbpl(str(g))}))")
    dirs, files = _existing_secrets(root)
    for p in dirs:
        lines.append(f"(deny file-read* (subpath {_sbpl(str(p))}))")
    for p in files:
        lines.append(f"(deny file-read* (literal {_sbpl(str(p))}))")
    return "\n".join(lines)


def _prefix(backend: str, root: Path, network: bool, extra_rw: list[Path]
            ) -> list[str]:
    if backend == "bwrap":
        return _bwrap_args(root, network, extra_rw)
    # `--` ends sandbox-exec's own options, so an argv[0] starting with `-`
    # can never be read as a Seatbelt flag.
    return ["sandbox-exec", "-p", seatbelt_profile(root, network, extra_rw), "--"]


def _probe(backend: str, network: bool) -> tuple[bool, str]:
    """Whether the backend can really start a process here (user namespaces
    can be disabled, nesting forbidden); cached."""
    key = (backend, network)
    if key in _probe_cache:
        return _probe_cache[key]
    true = shutil.which("true") or "/usr/bin/true"
    probe_root = Path(os.getcwd()).resolve()
    try:
        argv = _prefix(backend, probe_root, network, []) + [true]
        r = subprocess.run(argv, capture_output=True, timeout=10,
                           stdin=subprocess.DEVNULL)
        ok = r.returncode == 0
        why = "" if ok else (r.stderr.decode("utf-8", "replace").strip()
                             .splitlines() or ["failed to start"])[-1][:160]
    except (OSError, subprocess.SubprocessError) as e:
        ok, why = False, str(e)[:160]
    _probe_cache[key] = (ok, why)
    return ok, why


def status(cfg, root=None) -> tuple[str, str]:
    """(state, detail): state is 'off', 'active' or 'unavailable'."""
    if not getattr(cfg, "sandbox", True):
        return "off", "sandbox = false"
    backend = backend_name()
    net = bool(getattr(cfg, "sandbox_network", False))
    if not backend:
        why = ("no sandbox backend on this system "
               "(Linux: install bubblewrap; macOS: sandbox-exec ships with the OS; "
               "Windows: the built-in Job Object sandbox is unavailable here)")
        return "unavailable", why
    if backend == "win_restricted":
        return "active", (
            "win_restricted: Job Object + restricted token (Low Integrity; "
            "writes restricted to project/temp; outbound network and "
            "user-readable files not blocked without AppContainer)")
    ok, why = _probe(backend, net)
    if not ok:
        return "unavailable", f"{backend} cannot start here ({why})"
    return "active", (f"{backend}: writes limited to the project and a temporary "
                      f"directory; network {'on' if net else 'off'}")


class SandboxRefused(RuntimeError):
    """The command could not be sandboxed (`sandbox_strict`, or an argv the
    sandbox backend could misread as its own flags)."""


def enforce_unattended_gate(cfg, *, permission_mode: str = "",
                            approval_policy: str = "",
                            background: bool = False,
                            requires_sandbox: bool = False) -> None:
    """Refuse unattended execution that has no OS sandbox behind it.

    The hard gate (D14): unattended `workspace-write`, `danger`,
    `background=True`, or `requires_sandbox` command execution must run inside
    an active OS sandbox. `sandbox` defaults to True (matching `status()`), so
    a config that never mentions it still opts into the gate.
    """
    mode = str(permission_mode
               or getattr(cfg, "permission_mode", "ask") or "ask").lower()
    approval = str(approval_policy
                   or getattr(cfg, "approval_policy", "ask") or "ask").lower()
    sandbox_on = bool(getattr(cfg, "sandbox", True))
    unattended = (background or requires_sandbox
                  or mode in ("workspace-write", "danger"))
    if not unattended:
        return
    active = bool(backend_name()) and sandbox_on
    if active:
        return                      # an OS sandbox is active and enabled
    strict = bool(getattr(cfg, "sandbox_strict", False))
    if (strict or background or requires_sandbox
            or (approval == "never" and sandbox_on)):
        raise SandboxRefused(
            "unattended execution (permission_mode="
            f"{mode!r}, background={background}) requires an OS sandbox, but "
            "no backend is active here")
    # `sandbox = false` is the explicit opt-out: an unattended, headless
    # (`approval = "never"`) run with the sandbox switched off is allowed.


def wrap(argv: list[str], cfg, root) -> tuple[list[str], str]:
    """(argv to run, note). The note is '' when unsandboxed, else a short
    description the caller can show the model on failure."""
    state, detail = status(cfg)
    if state == "off":
        return argv, ""
    if state == "unavailable":
        if getattr(cfg, "sandbox_strict", False):
            raise SandboxRefused(f"sandbox_strict is on but {detail}")
        if detail not in _noticed:
            _noticed.add(detail)
            print(f"[sandbox] {detail}; commands run unsandboxed. Set "
                  "sandbox = false to silence this.", file=sys.stderr)
        return argv, ""
    root = Path(root).resolve()
    extra = [Path(p).expanduser().resolve()
             for p in (getattr(cfg, "sandbox_writable", ()) or ())
             if Path(p).expanduser().exists()]
    net = bool(getattr(cfg, "sandbox_network", False))
    argv = list(argv)
    if not argv:
        raise SandboxRefused("refusing to sandbox an empty command")
    head = str(argv[0])
    # An argv[0] beginning with `-` would be parsed as a backend flag even
    # behind `--` on some platforms, and is a sign of a confused caller.
    # Allow only an explicit path that actually exists on disk.
    if head.startswith("-") and not Path(head).exists():
        raise SandboxRefused(
            f"refusing to sandbox a command whose program name starts with '-' "
            f"({head!r}); pass an explicit existing path instead")
    backend = backend_name()
    if backend == "win_restricted":
        # The Job Object + restricted token are applied inside
        # proc.run -> sandbox_win.run_sandboxed (no argv prefix).
        return argv, detail
    return _prefix(backend, root, net, extra) + argv, detail
