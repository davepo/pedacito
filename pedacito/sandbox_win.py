"""Windows OS sandbox: ctypes Job Object + Restricted Token (Low Integrity).

`run_sandboxed()` runs an argv under a Windows Job Object (KILL_ON_JOB_CLOSE +
die-on-unhandled-exception) and a restricted token (DISABLE_MAX_PRIVILEGE plus
the Low Integrity SID `S-1-16-4096`), so a model-chosen command cannot outlive
the agent, spawn escaping descendants, or read/write medium-integrity user
state. AppContainer is detected best-effort (never left behind on disk).

Pure stdlib `ctypes` (kernel32 / advapi32, optional userenv). Every
`ctypes.WinDLL` load is lazy and inside an `os.name == "nt"` guard, so this
module imports and the mocked tests run cleanly on Linux and macOS.
"""

from __future__ import annotations

import ctypes
import os
import stat as _stat
import subprocess
import tempfile
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# capabilities                                                                #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WinSandboxCapabilities:
    available: bool
    job_object: bool
    restricted_token: bool
    low_integrity: bool
    appcontainer: bool
    reason: str


# --------------------------------------------------------------------------- #
# constants                                                                   #
# --------------------------------------------------------------------------- #

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x0400
JOB_OBJECT_UILIMIT_HANDLES = 0x0001
JOB_OBJECT_UILIMIT_READCLIPBOARD = 0x0002
JOB_OBJECT_UILIMIT_WRITECLIPBOARD = 0x0004
JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS = 0x0008
JOB_OBJECT_UILIMIT_DISPLAYSETTINGS = 0x0010
JOB_OBJECT_UILIMIT_GLOBALATOMS = 0x0020
JOB_OBJECT_UILIMIT_DESKTOP = 0x0040

JobObjectExtendedLimitInformation = 9
JobObjectBasicUIRestrictions = 4

TOKEN_ASSIGN_PRIMARY = 0x0001
TOKEN_DUPLICATE = 0x0002
TOKEN_QUERY = 0x0008
TOKEN_ADJUST_DEFAULT = 0x0080
TOKEN_ADJUST_SESSIONID = 0x0100
_OPEN_TOKEN_ACCESS = (TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY
                      | TOKEN_ADJUST_DEFAULT | TOKEN_ADJUST_SESSIONID)
DISABLE_MAX_PRIVILEGE = 0x1
TokenIntegrityLevel = 25
SE_GROUP_INTEGRITY = 0x00000020

CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
STARTF_USESTDHANDLES = 0x00000100
HANDLE_FLAG_INHERIT = 0x00000001

ERROR_PRIVILEGE_NOT_HELD = 1314
ERROR_ACCESS_DENIED = 5
ERROR_BROKEN_PIPE = 109
STILL_ACTIVE = 259
WAIT_TIMEOUT = 0x00000102
WAIT_OBJECT_0 = 0x00000000
INFINITE = 0xFFFFFFFF

# NTFS reparse point (symlink / junction / mount point) attribute bit.
FILE_ATTRIBUTE_REPARSE_POINT = 0x400

# Project paths the Low-Integrity child must never be able to write: they stay
# Medium Integrity so `.git/hooks`, `.pedacito/policy.toml` and
# `.pedacito.toml` cannot be rewritten by the child. Mirrors the guarded set in
# `sandbox.py` plus the extra config filenames and cache dirs.
GUARDED = (".git", ".pedacito", ".pedacito.toml", ".pedacito.yml",
           ".pedacito.yaml", "node_modules", ".venv")

# Directory trees larger than this are not walked to relabel; the root is still
# relabeled, and the child simply cannot write to the unrelabeled remainder
# (fail closed) rather than stalling `run` on an enormous tree.
_MAX_RELABEL_ENTRIES = 5000

# Environment variables naming protected system directories.
_SYSTEM_ENV_VARS = ("SystemRoot", "WINDIR", "ProgramFiles", "ProgramFiles(x86)",
                    "ProgramData", "APPDATA", "LOCALAPPDATA")

# --------------------------------------------------------------------------- #
# win32 structs                                                               #
# --------------------------------------------------------------------------- #

class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class JOBOBJECT_BASIC_UI_RESTRICTIONS(ctypes.Structure):
    _fields_ = [
        ("UIRestrictionsClass", wintypes.DWORD),
    ]


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Sid", ctypes.c_void_p),
        ("Attributes", wintypes.DWORD),
    ]


class TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [
        ("Label", SID_AND_ATTRIBUTES),
    ]


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", wintypes.LPBYTE),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


# --------------------------------------------------------------------------- #
# lazy DLL loading                                                            #
# --------------------------------------------------------------------------- #

_kernel32_dll = None
_advapi32_dll = None
_userenv_dll = None
_load_lock = threading.Lock()


def _kernel32():
    global _kernel32_dll
    if _kernel32_dll is None:
        with _load_lock:
            if _kernel32_dll is None:
                k = ctypes.WinDLL("kernel32", use_last_error=True)
                _configure_kernel32(k)
                _kernel32_dll = k
    return _kernel32_dll


def _advapi32():
    global _advapi32_dll
    if _advapi32_dll is None:
        with _load_lock:
            if _advapi32_dll is None:
                a = ctypes.WinDLL("advapi32", use_last_error=True)
                _configure_advapi32(a)
                _advapi32_dll = a
    return _advapi32_dll


def _userenv():
    global _userenv_dll
    if _userenv_dll is None:
        with _load_lock:
            if _userenv_dll is None:
                try:
                    u = ctypes.WinDLL("userenv", use_last_error=True)
                    u.CreateAppContainerProfile.restype = wintypes.BOOL
                    u.CreateAppContainerProfile.argtypes = [
                        wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
                        wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID,
                    ]
                    u.DeleteAppContainerProfile.restype = wintypes.BOOL
                    u.DeleteAppContainerProfile.argtypes = [wintypes.LPCWSTR]
                    _userenv_dll = u
                except OSError:
                    _userenv_dll = None
    return _userenv_dll


def _configure_kernel32(k):
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k.SetInformationJobObject.restype = wintypes.BOOL
    k.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.AssignProcessToJobObject.restype = wintypes.BOOL
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetCurrentProcess.argtypes = []
    k.TerminateProcess.restype = wintypes.BOOL
    k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k.TerminateJobObject.restype = wintypes.BOOL
    k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k.WaitForSingleObject.restype = wintypes.DWORD
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.GetExitCodeProcess.restype = wintypes.BOOL
    k.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.CreatePipe.restype = wintypes.BOOL
    k.CreatePipe.argtypes = [
        ctypes.POINTER(wintypes.HANDLE), ctypes.POINTER(wintypes.HANDLE),
        ctypes.c_void_p, wintypes.DWORD]
    k.SetHandleInformation.restype = wintypes.BOOL
    k.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                       wintypes.DWORD]
    k.ReadFile.restype = wintypes.BOOL
    k.ReadFile.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    k.WriteFile.restype = wintypes.BOOL
    k.WriteFile.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    k.CreateProcessAsUserW.restype = wintypes.BOOL
    k.CreateProcessAsUserW.argtypes = [
        wintypes.HANDLE, wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p,
        ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p,
        wintypes.LPCWSTR, ctypes.c_void_p, ctypes.c_void_p]
    k.ResumeThread.restype = wintypes.DWORD
    k.ResumeThread.argtypes = [wintypes.HANDLE]
    k.LocalFree.restype = wintypes.HLOCAL
    k.LocalFree.argtypes = [wintypes.HLOCAL]


def _configure_advapi32(a):
    a.OpenProcessToken.restype = wintypes.BOOL
    a.OpenProcessToken.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    a.CreateRestrictedToken.restype = wintypes.BOOL
    a.CreateRestrictedToken.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
        ctypes.POINTER(wintypes.HANDLE)]
    a.ConvertStringSidToSidW.restype = wintypes.BOOL
    a.ConvertStringSidToSidW.argtypes = [
        wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    a.SetTokenInformation.restype = wintypes.BOOL
    a.SetTokenInformation.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    a.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    a.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.DWORD)]
    a.GetSecurityDescriptorSacl.restype = wintypes.BOOL
    a.GetSecurityDescriptorSacl.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL)]
    a.SetNamedSecurityInfoW.restype = wintypes.DWORD
    a.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


# --------------------------------------------------------------------------- #
# pure helpers (testable on every platform)                                   #
# --------------------------------------------------------------------------- #

def _env_block(env: dict[str, str]) -> bytes:
    """A double-NUL-terminated UTF-16LE environment block."""
    items = [f"{k}={v}" for k, v in env.items()]
    return ("\0".join(items) + "\0\0").encode("utf-16-le")


def _command_line(argv: list[str]) -> str:
    return subprocess.list2cmdline([str(a) for a in argv])


def _under(path: Path, parent: Path) -> bool:
    """True when `path` is `parent` or a descendant of `parent`."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_safe_to_relabel(path: Path) -> bool:
    """Refuse to relabel anything dangerous.

    The Low-Integrity relabel is a real, persistent change to the object's
    mandatory label, so it must never touch a drive root, the user's home (or
    any of its ancestors), a system directory (or any path that contains one
    or lives under one), or a reparse point / junction / symlink itself. Only
    the designated temporary directory (and everything beneath it) is exempt
    from the system-directory rule, because it is the sandbox's own scratch
    space that `run_sandboxed` must be able to relabel.
    """
    try:
        p = path.resolve()
    except OSError:
        return False
    # Drive / filesystem roots.
    if p == p.parent or len(p.parts) <= 1:
        return False
    # The user's home and every ancestor of it.
    try:
        home = Path.home().resolve()
    except OSError:
        home = None
    if home is not None and (p == home or _under(home, p)):
        return False
    # System directories and their ancestors/descendants (unless in temp).
    try:
        temp = Path(tempfile.gettempdir()).resolve()
    except OSError:
        temp = None
    if temp is None or not _under(p, temp):
        for name in _SYSTEM_ENV_VARS:
            base = os.environ.get(name)
            if not base:
                continue
            try:
                bp = Path(base).resolve()
            except OSError:
                continue
            if p == bp or _under(bp, p) or _under(p, bp):
                return False
    # A reparse point / junction / symlink itself is never relabeled.
    try:
        if path.is_symlink():
            return False
        st = path.lstat()
    except OSError:
        return False
    if bool(getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT):
        return False
    return True


def _entry_is_reparse(entry) -> bool:
    """True when a scandir entry is a symlink / junction / reparse point."""
    try:
        st = entry.stat(follow_symlinks=False)
    except OSError:
        return True              # cannot inspect -> skip (fail closed)
    if bool(getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT):
        return True
    try:
        return bool(entry.is_symlink())
    except OSError:
        return True


_SE_FILE_OBJECT = 1
_LABEL_SECURITY_INFORMATION = 0x00000010
_SDDL_REVISION_1 = 1
_LABELED_CACHE: dict[str, str] = {}
_LABELED_LOCK = threading.Lock()
_SACL_CACHE: dict[str, tuple[ctypes.c_void_p, ctypes.c_void_p]] = {}


def _get_label_sacl(level: str) -> ctypes.c_void_p | None:
    """Build and cache an in-process mandatory integrity SACL (`L` or `M`)."""
    if os.name != "nt":
        return None
    with _LABELED_LOCK:
        cached = _SACL_CACHE.get(level)
        if cached is not None:
            return cached[1]
        sddl = "S:(ML;;NW;;;LW)" if level == "L" else "S:(ML;;NW;;;ME)"
        try:
            a = _advapi32()
            sd = ctypes.c_void_p()
            if not a.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    sddl, _SDDL_REVISION_1, ctypes.byref(sd), None):
                return None
            present = wintypes.BOOL()
            sacl = ctypes.c_void_p()
            defaulted = wintypes.BOOL()
            if not a.GetSecurityDescriptorSacl(
                    sd, ctypes.byref(present), ctypes.byref(sacl),
                    ctypes.byref(defaulted)) or not present.value:
                _kernel32().LocalFree(sd)
                return None
            # Keep `sd` alive for the lifetime of the process since `sacl`
            # points inside it.
            _SACL_CACHE[level] = (sd, sacl)
            return sacl
        except Exception:
            return None


def _win32_setintegrity(target: str, level: str) -> bool | None:
    """Fast in-process `SetNamedSecurityInfoW` integrity relabel (~0.05ms per
    entry vs ~25ms per `icacls.exe` spawn). Returns None when unavailable so
    callers may fall back to `_icacls_setintegrity`."""
    if os.name != "nt":
        return None
    sacl = _get_label_sacl(level)
    if sacl is None:
        return None
    try:
        rc = _advapi32().SetNamedSecurityInfoW(
            target, _SE_FILE_OBJECT, _LABEL_SECURITY_INFORMATION,
            None, None, None, sacl)
        return rc == 0
    except Exception:
        return None


def _icacls_setintegrity(target: str, level: str) -> bool:
    """Non-recursive integrity label (`L` or `M`) on one object -- never `/T`.

    Uses in-process `SetNamedSecurityInfoW` on Windows for sub-millisecond
    labeling (avoiding spawning `icacls.exe` per file), with an in-process
    cache so repeat commands on the same project skip already-labeled paths.
    Falls back to `icacls <target> /setintegritylevel <level> /L /Q` only when
    Win32 API is unavailable."""
    cache_key = os.path.normcase(target)
    with _LABELED_LOCK:
        if _LABELED_CACHE.get(cache_key) == level:
            return True
    win_res = _win32_setintegrity(target, level)
    if win_res is not None:
        if win_res:
            with _LABELED_LOCK:
                if len(_LABELED_CACHE) > 20000:
                    _LABELED_CACHE.clear()
                _LABELED_CACHE[cache_key] = level
        return win_res
    try:
        r = subprocess.run(
            ["icacls", target, "/setintegritylevel", level, "/L", "/Q"],
            capture_output=True, timeout=30)
        ok = r.returncode == 0
        if ok:
            with _LABELED_LOCK:
                _LABELED_CACHE[cache_key] = level
        return ok
    except Exception:
        return False


def _label_low_integrity(path: Path, *, guarded=None) -> bool:
    """Best-effort: relabel `path` (and its safe, non-reparse, non-guarded
    descendants) Low Integrity so the restricted-token child can write there
    -- and only there. The Low Integrity SID makes the child refuse writes to
    every Medium/High Integrity path outside the relabeled project and temp
    dirs (the Windows analogue of bwrap/seatbelt "writes limited to the
    project").

    Never uses `icacls /T`: a Python walker skips every symlink / NTFS
    junction / reparse point and every guarded name, relabels the root and its
    non-guarded children/files non-recursively, and pins guarded paths back to
    Medium Integrity if a previous run left them Low. A failure just means the
    child cannot write there (still a sandbox)."""
    if not _is_safe_to_relabel(path):
        return False
    guarded = set(guarded if guarded is not None else GUARDED)
    ok = _icacls_setintegrity(str(path), "L")
    budget = _MAX_RELABEL_ENTRIES
    stack = [path]
    while stack and budget > 0:
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for entry in entries:
            budget -= 1
            if budget <= 0:
                break
            if entry.name in guarded:
                # Keep `.git`, `.pedacito`, `.pedacito.toml`, ... Medium so a
                # Low-Integrity child cannot rewrite hooks/policy/config.
                if not _entry_is_reparse(entry):
                    ok = _icacls_setintegrity(entry.path, "M") and ok
                continue
            if _entry_is_reparse(entry):
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if _stat.S_ISDIR(st.st_mode):
                ok = _icacls_setintegrity(entry.path, "L") and ok
                stack.append(Path(entry.path))
            elif _stat.S_ISREG(st.st_mode):
                ok = _icacls_setintegrity(entry.path, "L") and ok
    return ok


# --------------------------------------------------------------------------- #
# probe                                                                       #
# --------------------------------------------------------------------------- #

def probe() -> WinSandboxCapabilities:
    """Whether the OS sandbox can really start a no-op child here."""
    if os.name != "nt":
        return WinSandboxCapabilities(False, False, False, False, False,
                                      "not Windows (os.name != 'nt')")
    return _probe_windows()


def _probe_windows() -> WinSandboxCapabilities:
    job_ok, job_err = _check_job_object()
    tok_ok, tok_err, low_ok = _check_restricted_token()
    ac_ok = _check_appcontainer()
    launch_ok, launch_err = _check_child_launch()
    # `low_ok` is required: a Medium-integrity child must NEVER be reported as
    # `available=True` (it would run with far more write access than the
    # Low-Integrity restricted token promises).
    available = job_ok and tok_ok and low_ok and launch_ok
    reason = ""
    if not available:
        parts = [p for p in (job_err, tok_err, launch_err) if p]
        if not parts and tok_ok and not low_ok:
            parts.append("Low Integrity restricted token unavailable")
        reason = parts[0] if parts else "sandbox unavailable"
    return WinSandboxCapabilities(
        available=available, job_object=job_ok,
        restricted_token=tok_ok, low_integrity=low_ok,
        appcontainer=ac_ok, reason=reason)


def _check_job_object() -> tuple[bool, str]:
    """Create a job object and set the hard limits."""
    try:
        k = _kernel32()
    except OSError as exc:
        return False, f"kernel32 unavailable ({exc})"
    job = k.CreateJobObjectW(None, None)
    if not job:
        err = ctypes.get_last_error()
        return False, f"CreateJobObjectW failed (winerror {err})"
    try:
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION)
        if not k.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            err = ctypes.get_last_error()
            return False, f"SetInformationJobObject failed (winerror {err})"
        return True, ""
    finally:
        k.CloseHandle(job)


def _check_restricted_token() -> tuple[bool, str, bool]:
    """Build a DISABLE_MAX_PRIVILEGE + Low Integrity restricted token."""
    try:
        a = _advapi32()
        k = _kernel32()
    except OSError as exc:
        return False, f"advapi32 unavailable ({exc})", False
    cur = wintypes.HANDLE()
    if not a.OpenProcessToken(k.GetCurrentProcess(), _OPEN_TOKEN_ACCESS,
                              ctypes.byref(cur)):
        err = ctypes.get_last_error()
        return False, f"OpenProcessToken failed (winerror {err})", False
    res = wintypes.HANDLE()
    try:
        if not a.CreateRestrictedToken(
                cur, DISABLE_MAX_PRIVILEGE, 0, None, 0, None, 0, None,
                ctypes.byref(res)):
            err = ctypes.get_last_error()
            return False, f"CreateRestrictedToken failed (winerror {err})", False
        low_ok = False
        sid = ctypes.c_void_p()
        if a.ConvertStringSidToSidW("S-1-16-4096", ctypes.byref(sid)) and sid:
            label = TOKEN_MANDATORY_LABEL()
            label.Label.Sid = sid
            label.Label.Attributes = SE_GROUP_INTEGRITY
            if a.SetTokenInformation(
                    res, TokenIntegrityLevel, ctypes.byref(label),
                    ctypes.sizeof(label)):
                low_ok = True
            k.LocalFree(sid)
        return True, "", low_ok
    finally:
        k.CloseHandle(cur)
        if res:
            k.CloseHandle(res)


def _check_appcontainer() -> bool:
    """Best-effort AppContainer detection (never leaves a profile behind)."""
    u = _userenv()
    if u is None:
        return False
    try:
        # Probe only for function availability, not by creating a profile.
        return hasattr(u, "CreateAppContainerProfile") and \
            hasattr(u, "DeleteAppContainerProfile")
    except Exception:
        return False


def _check_child_launch() -> tuple[bool, str]:
    """Really start a no-op child through the exact Job Object + restricted
    token + CreateProcessAsUserW path and verify exit code 0."""
    if os.name != "nt":
        return False, "not Windows"
    # Run the probe in a private temporary directory so `os.getcwd()` is NEVER
    # touched or relabeled (a probe must not relabel the agent's own cwd).
    with tempfile.TemporaryDirectory(prefix="pedacito-sandbox-probe-") as tmp:
        try:
            child = _spawn_child(
                [os.environ.get("COMSPEC") or "cmd.exe", "/c", "exit 0"],
                cwd=Path(tmp), env=dict(os.environ), stdin_text=None)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        try:
            rc = child.wait(10.0)
            if rc == 0:
                return True, ""
            err = child.last_error or f"child exited {rc}"
            return False, err
        finally:
            child.terminate()
            child.close()


# --------------------------------------------------------------------------- #
# run_sandboxed                                                               #
# --------------------------------------------------------------------------- #

def run_sandboxed(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float | None = None,
    idle_timeout: float = 0.0,
    max_output: int = 8000,
    stdin_text: str | None = None,
    allow_network: bool = False,
    writable_dirs: tuple[Path, ...] = (),
    cancel_event: threading.Event | None = None,
) -> tuple[int | None, str, str, float]:
    """Run `argv` under the Windows Job Object + restricted token.

    Returns `(exit_code, stdout_str, stderr_str, elapsed_s)`. `exit_code` is
    None when the child could not be started or was killed without an exit
    code. The whole job is reaped atomically on exit/cancel/timeout.
    """
    if os.name != "nt":
        return None, "", "not Windows", 0.0
    rc, out, err, elapsed, _stopped = _run_nt(
        argv, cwd=Path(cwd), env=env, timeout=timeout,
        idle_timeout=idle_timeout, max_output=max_output,
        stdin_text=stdin_text, cancel_event=cancel_event,
        writable_dirs=writable_dirs)
    return rc, out, err, elapsed


def _run_nt(argv, *, cwd, env, timeout, idle_timeout, max_output,
            stdin_text, cancel_event, writable_dirs=()):
    """The shared orchestration. Returns
    `(rc, stdout, stderr, elapsed, stopped)` where `stopped` is one of "",
    "idle", "timeout", "cancelled", or "error"."""
    started = time.monotonic()
    timeout = float(timeout or 0.0)
    idle_timeout = float(idle_timeout or 0.0)
    cap = max(200, int(max_output or 8000))
    head_cap = max(120, int(cap * 0.6))
    tail_cap = max(80, cap - head_cap)

    try:
        child = _spawn_child(argv, cwd=cwd, env=env, stdin_text=stdin_text,
                             writable_dirs=writable_dirs)
    except Exception as exc:            # could not start
        return None, "", f"{type(exc).__name__}: {exc}", \
            time.monotonic() - started, "error"

    state = {"last": started, "overflow": False}
    head = bytearray()
    tail = bytearray()
    lock = threading.Lock()

    def _append(dst, chunk):
        if not chunk:
            return
        with lock:
            state["last"] = time.monotonic()
            if not state["overflow"]:
                room = head_cap - len(dst)
                if len(chunk) <= room:
                    dst.extend(chunk)
                    return
                dst.extend(chunk[:room])
                state["overflow"] = True
                chunk = chunk[room:]
            tail.extend(chunk)
            if len(tail) > tail_cap:
                del tail[:len(tail) - tail_cap]

    def drain(reader, dst):
        while True:
            try:
                chunk = child.read_stdout() if reader == "out" \
                    else child.read_stderr()
            except Exception:
                break
            if chunk in (None, b""):
                break
            _append(dst, chunk)

    threads = [
        threading.Thread(target=drain, args=("out", head), daemon=True),
        threading.Thread(target=drain, args=("err", head), daemon=True),
    ]
    for t in threads:
        t.start()

    stopped = ""
    try:
        while True:
            rc = child.poll()
            if rc is not None:
                break
            if cancel_event is not None and cancel_event.is_set():
                child.terminate()
                stopped = "cancelled"
                break
            now = time.monotonic()
            if timeout and now - started >= timeout:
                child.terminate()
                stopped = "timeout"
                break
            with lock:
                last = state["last"]
            if idle_timeout and now - last >= idle_timeout:
                child.terminate()
                stopped = "idle"
                break
            time.sleep(0.02)
        if not stopped:
            rc = child.wait(5.0)
            if rc is None:
                child.terminate()
                stopped = "timeout"
    finally:
        for t in threads:
            t.join(timeout=1)
        child.close()

    with lock:
        raw_head = bytes(head)
        overflow = state["overflow"]
        raw_tail = bytes(tail) if overflow else b""
    out = raw_head.decode("utf-8", "replace")
    if overflow:
        out = (out + "\n[... output truncated: middle omitted; kept the "
               f"first {head_cap} and last {tail_cap} chars ...]\n"
               + raw_tail.decode("utf-8", "replace"))
    if stopped:
        rc = None
    return rc, out, "", time.monotonic() - started, stopped


# --------------------------------------------------------------------------- #
# process creation                                                            #
# --------------------------------------------------------------------------- #

class _WinChild:
    """A created child plus its job object, token, process and pipe handles."""

    def __init__(self, job, proc, thread, out_r, err_r,
                 token=None, last_error=""):
        self.job = job
        self.proc = proc
        self.thread = thread
        self.out_r = out_r
        self.err_r = err_r
        self.token = token
        self.last_error = last_error
        self._exit = None

    # -- lifecycle -----------------------------------------------------------
    def poll(self) -> int | None:
        if self._exit is not None:
            return self._exit
        k = _kernel32()
        code = wintypes.DWORD()
        if not k.GetExitCodeProcess(self.proc, ctypes.byref(code)):
            self._exit = None
            return None
        if code.value == STILL_ACTIVE:
            return None
        self._exit = int(code.value)
        return self._exit

    def wait(self, timeout: float) -> int | None:
        k = _kernel32()
        ms = INFINITE if timeout is None else int(max(0.0, timeout) * 1000)
        r = k.WaitForSingleObject(self.proc, ms)
        if r == WAIT_OBJECT_0:
            return self.poll()
        return None

    def read_stdout(self) -> bytes | None:
        return _read_pipe(self.out_r)

    def read_stderr(self) -> bytes | None:
        return _read_pipe(self.err_r)

    def terminate(self) -> None:
        k = _kernel32()
        if self.job:
            k.TerminateJobObject(self.job, 1)
        if self.proc:
            k.TerminateProcess(self.proc, 1)

    def close(self) -> None:
        k = _kernel32()
        for h in (self.out_r, self.err_r, self.thread, self.proc, self.job):
            if h:
                try:
                    k.CloseHandle(h)
                except Exception:
                    pass
        if self.token:
            try:
                k.CloseHandle(self.token)
            except Exception:
                pass


def _read_pipe(handle) -> bytes | None:
    k = _kernel32()
    chunk = 65536
    buf = ctypes.create_string_buffer(chunk)
    read = wintypes.DWORD()
    ok = k.ReadFile(handle, buf, chunk, ctypes.byref(read), None)
    if ok:
        return buf.raw[:read.value]
    err = ctypes.get_last_error()
    if err in (ERROR_BROKEN_PIPE,):
        return b""
    return None


def _make_pipe(inherit_read: bool = False, inherit_write: bool = False
               ) -> tuple[int, int]:
    """Create an anonymous pipe and return `(read_handle, write_handle)`.

    Each end's inheritability is set explicitly: the end the CHILD keeps must
    be inheritable, the end the PARENT keeps must not be, so that closing the
    parent's end delivers an immediate EOF (a leaked inheritable write end
    would keep a reader from ever seeing EOF, and a leaked inheritable read
    end would keep a writer from ever seeing a broken pipe)."""
    k = _kernel32()
    r = wintypes.HANDLE()
    w = wintypes.HANDLE()
    sa = SECURITY_ATTRIBUTES()
    sa.nLength = ctypes.sizeof(SECURITY_ATTRIBUTES)
    sa.bInheritHandle = True
    if not k.CreatePipe(ctypes.byref(r), ctypes.byref(w), ctypes.byref(sa), 0):
        err = ctypes.get_last_error()
        raise OSError(err, f"CreatePipe failed (winerror {err})")
    k.SetHandleInformation(r, HANDLE_FLAG_INHERIT,
                           HANDLE_FLAG_INHERIT if inherit_read else 0)
    k.SetHandleInformation(w, HANDLE_FLAG_INHERIT,
                           HANDLE_FLAG_INHERIT if inherit_write else 0)
    return int(r.value), int(w.value)


def _close_handle(h) -> None:
    """Close one Win32 handle (an int or a ``wintypes.HANDLE``), never raising."""
    if h is None:
        return
    try:
        if isinstance(h, int):
            h = wintypes.HANDLE(h)
        _kernel32().CloseHandle(h)
    except Exception:
        pass


def _spawn_child(argv, *, cwd: Path, env: dict[str, str],
                 stdin_text: str | None,
                 writable_dirs: tuple[Path, ...] = ()) -> _WinChild:
    if os.name != "nt":
        raise OSError("Windows-only sandbox backend")
    k = _kernel32()
    a = _advapi32()

    # The Low Integrity token blocks writes to every Medium/High path, so the
    # dirs the child MAY write (the project + any writable dirs) are relabeled
    # Low Integrity first. This is the Windows "writes limited to the project"
    # guarantee; a relabel failure just means the child cannot write.
    for d in dict.fromkeys((Path(cwd), *[Path(p) for p in writable_dirs])):
        _label_low_integrity(d)

    job = None
    token = None
    in_r = in_w = out_r = out_w = err_r = err_w = None
    proc = None
    thread = None
    try:
        # Job object.
        job = k.CreateJobObjectW(None, None)
        if not job:
            err = ctypes.get_last_error()
            raise OSError(err, f"CreateJobObjectW failed (winerror {err})")
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION)
        if not k.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            err = ctypes.get_last_error()
            raise OSError(err, f"SetInformationJobObject failed (winerror {err})")

        # Restricted token (DISABLE_MAX_PRIVILEGE + Low Integrity).
        cur = wintypes.HANDLE()
        if not a.OpenProcessToken(k.GetCurrentProcess(), _OPEN_TOKEN_ACCESS,
                                  ctypes.byref(cur)):
            err = ctypes.get_last_error()
            raise OSError(err, f"OpenProcessToken failed (winerror {err})")
        try:
            token = wintypes.HANDLE()
            if not a.CreateRestrictedToken(
                    cur, DISABLE_MAX_PRIVILEGE, 0, None, 0, None, 0, None,
                    ctypes.byref(token)):
                err = ctypes.get_last_error()
                raise OSError(err, f"CreateRestrictedToken failed (winerror {err})")
            sid = ctypes.c_void_p()
            if not (a.ConvertStringSidToSidW("S-1-16-4096",
                                             ctypes.byref(sid)) and sid):
                err = ctypes.get_last_error()
                raise OSError(
                    err, "ConvertStringSidToSidW failed (Low Integrity SID)")
            label = TOKEN_MANDATORY_LABEL()
            label.Label.Sid = sid
            label.Label.Attributes = SE_GROUP_INTEGRITY
            try:
                if not a.SetTokenInformation(
                        token, TokenIntegrityLevel, ctypes.byref(label),
                        ctypes.sizeof(label)):
                    err = ctypes.get_last_error()
                    raise OSError(
                        err, f"SetTokenInformation failed (winerror {err})")
            finally:
                k.LocalFree(sid)
        finally:
            k.CloseHandle(cur)

        # Pipes: the child keeps the stdin read end, the stdout/stderr write
        # ends; the parent keeps the opposite ends, non-inheritable.
        out_r, out_w = _make_pipe(inherit_read=False, inherit_write=True)
        err_r, err_w = _make_pipe(inherit_read=False, inherit_write=True)
        in_r, in_w = _make_pipe(inherit_read=True, inherit_write=False)

        si = STARTUPINFOW()
        si.cb = ctypes.sizeof(STARTUPINFOW)
        si.dwFlags = STARTF_USESTDHANDLES
        si.hStdInput = wintypes.HANDLE(in_r)
        si.hStdOutput = wintypes.HANDLE(out_w)
        si.hStdError = wintypes.HANDLE(err_w)
        pi = PROCESS_INFORMATION()

        cmdline = _command_line(argv)
        cmd_buf = ctypes.create_unicode_buffer(cmdline)
        env_block = _env_block(env)
        env_buf = ctypes.create_string_buffer(env_block)
        flags = CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT

        ok = k.CreateProcessAsUserW(
            token, None, cmd_buf, None, None, True, flags, env_buf,
            str(cwd), ctypes.byref(si), ctypes.byref(pi))
        if not ok:
            err = ctypes.get_last_error()
            raise OSError(err, f"CreateProcessAsUserW failed (winerror {err})")
        proc = pi.hProcess
        thread = pi.hThread

        # Parent closes the child-end of each pipe right after creation.
        _close_handle(out_w)
        out_w = None
        _close_handle(err_w)
        err_w = None
        _close_handle(in_r)
        in_r = None

        # Feed stdin, if any, then close the write end so the child sees EOF.
        if stdin_text is not None:
            data = stdin_text.encode("utf-8", "replace")
            written = wintypes.DWORD()
            k.WriteFile(wintypes.HANDLE(in_w), data, len(data),
                        ctypes.byref(written), None)
        _close_handle(in_w)
        in_w = None

        # Put the child in the job, then let it run. Every Win32 call is
        # checked: on failure the suspended child is terminated and never
        # resumed un-sandboxed.
        if not k.AssignProcessToJobObject(job, proc):
            err = ctypes.get_last_error()
            raise OSError(err, f"AssignProcessToJobObject failed (winerror {err})")
        if k.ResumeThread(thread) == 0xFFFFFFFF:
            err = ctypes.get_last_error()
            raise OSError(err, f"ResumeThread failed (winerror {err})")

        child = _WinChild(job, proc, thread, wintypes.HANDLE(out_r),
                          wintypes.HANDLE(err_r), token=token)
        job = token = proc = thread = None
        out_r = err_r = None
        return child
    except BaseException:
        # Fail closed: terminate the (still suspended) child, then close every
        # handle we opened -- never leak a job, token, pipe, process or thread.
        if proc is not None:
            try:
                k.TerminateProcess(proc, 1)
            except Exception:
                pass
        for h in (in_r, in_w, out_r, out_w, err_r, err_w, thread, proc,
                  job, token):
            _close_handle(h)
        raise


# Import guard for `ctypes.WinDLL` on non-Windows: the module-level import of
# `wintypes` is safe on every platform; only `_kernel32()`/`_advapi32()`
# actually load the DLLs, and both are guarded by `os.name == "nt"` callers.
