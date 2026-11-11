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
    Windows  none that works without admin rights

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
from pathlib import Path

# Directories under $HOME that hold credentials; hidden inside the sandbox.
SECRET_DIRS = (".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker",
               ".config/gcloud", ".config/gh", ".config/pedacito",
               ".var/app/ai.opencode.opencode/config/pedacito")
SECRET_FILES = (".netrc", ".git-credentials", ".npmrc", ".pypirc")
# Inside the project: readable, never writable by a sandboxed command.
GUARDED = (".git", ".pedacito", ".pedacito.toml", "pedacito.toml",
           ".pedacitoignore")

_probe_cache: dict[tuple[str, bool], tuple[bool, str]] = {}
_noticed: set[str] = set()


def backend_name() -> str:
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "bwrap"
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "seatbelt"
    return ""


def _home() -> Path:
    return Path(os.path.expanduser("~"))


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
    dirs, files = _existing_secrets(root)
    for d in dirs:
        args += ["--tmpfs", str(d)]
    for f in files:
        args += ["--ro-bind", "/dev/null", str(f)]
    for p in [root, *extra_rw]:
        args += ["--bind", str(p), str(p)]
    for name in GUARDED:
        g = root / name
        if g.exists():
            args += ["--ro-bind", str(g), str(g)]
    return args + ["--chdir", str(root), "--"]


def _sbpl(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def seatbelt_profile(root: Path, network: bool, extra_rw: list[Path]) -> str:
    """The Seatbelt profile (last matching rule wins)."""
    writable = [root, *extra_rw, Path("/private/tmp"), Path("/private/var/folders")]
    lines = ["(version 1)", "(allow default)"]
    if not network:
        lines.append("(deny network*)")
        lines.append('(allow network* (remote unix-socket))')
    lines.append("(deny file-write*)")
    lines.append("(allow file-write* "
                 + " ".join(f"(subpath {_sbpl(str(p))})" for p in writable)
                 + ' (literal "/dev/null") (literal "/dev/tty")'
                 + ' (regex #"^/dev/ttys[0-9]+$"))')
    for name in GUARDED:
        g = root / name
        if g.exists():
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
    return ["sandbox-exec", "-p", seatbelt_profile(root, network, extra_rw)]


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
               "Windows has none)")
        return "unavailable", why
    ok, why = _probe(backend, net)
    if not ok:
        return "unavailable", f"{backend} cannot start here ({why})"
    return "active", (f"{backend}: writes limited to the project and a temporary "
                      f"directory; network {'on' if net else 'off'}")


class SandboxRefused(RuntimeError):
    """`sandbox_strict` is set and no sandbox could be applied."""


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
    backend = backend_name()
    return _prefix(backend, root, net, extra) + list(argv), detail
