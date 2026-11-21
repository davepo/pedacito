"""Security hardening and the live human-in-the-loop Permission Broker.

Four guards plus one broker:

1. `resolve_safe_path(root, target)` -- strict workspace containment. Both the
   root and the candidate are fully resolved (symlinks included) and the
   candidate must remain under the root. Any `..` component and any symlink
   that points outside the project raises `SecurityError`.

2. `is_sensitive_path(path)` -- the secret/credential detector: `.env` files
   (except the documented examples), private keys, SSH/TLS material,
   `.npmrc`/`.pypirc`/`.netrc`, `credentials.json`... Sensitive files are
   never indexed (see `index.reindex_file`) and never read by the model
   unless a human approves the read through the Permission Broker.

3. `post_origin_allowed(headers, bound_host)` -- the GUI server's CSRF guard:
   a state-mutating POST that carries an `Origin` or `Referer` must come from
   a loopback page. A website open in the same browser cannot drive the local
   server, even when it can guess the port.

4. `ssrf_error(url, allowed_hosts)` -- the web tools' SSRF guard: only full
   http(s) URLs may be fetched, and loopback (`127.0.0.1`, `localhost`,
   `::1`), link-local/cloud-metadata (`169.254.169.254`) and private LAN
   ranges (RFC1918 / unique-local) are refused unless the host is named
   explicitly. A prompt-injected URL can never probe the user's network.
   `resolve_checked(host, allowed_hosts)` closes the DNS-rebinding hole the
   literal check leaves: it resolves the name once, judges EVERY returned
   IP, and hands back the validated addresses so the fetch can pin its
   connection to one of them (a second DNS answer cannot swap in a private
   address mid-request).

5. The Permission Broker -- `request_permission(...)` calls the hook installed
   by the GUI (`gui.core.install_permission_broker`) and blocks the calling
   worker thread until the user answers the modal: Allow Once / Allow for Run
   / Always Allow / Deny & Redirect. `setting_key` names the config knob an
   "Always Allow" should flip (e.g. `enable_web`). With no hook installed
   (plain CLI runs) the broker is inactive: tools keep their historical
   behaviour and return their own "disabled in config" error instead of
   waiting on a human who is not there. A denial always answers the model with
   `DENIED_MESSAGE`, a redirect that keeps the run alive instead of crashing
   it.

`is_install_command(argv)` recognises package-manager side quests (`pip
install`, `npm install`, ...) so an otherwise-allowed shell command still
asks for its own elevated trust.
"""

from __future__ import annotations

import fnmatch
import hashlib
import ipaddress
import json
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlparse

from . import fileio

__all__ = [
    "CATEGORY_LABELS",
    "CAT_MCP_TOOL",
    "CAT_PACKAGE_INSTALL",
    "CAT_SENSITIVE_FILE",
    "CAT_SHELL",
    "CAT_TEST_EXECUTION",
    "CAT_WEB",
    "CAT_WEB_ACCESS",
    "DENIED_MESSAGE",
    "MCP_DENIED_MESSAGE",
    "OUTBOUND_MODES",
    "REDACTED_SECRET",
    "SensitiveFileFinding",
    "SecretFinding",
    "SecurityError",
    "acknowledge_findings",
    "check_git_secret_hygiene",
    "clear_security_cache",
    "ensure_gitignore_entries",
    "format_finding",
    "install_permission_hook",
    "is_install_command",
    "is_private_host",
    "is_sensitive_path",
    "is_sensitive_rel",
    "load_acknowledged_fingerprints",
    "outbound_mode",
    "permission_broker_active",
    "permission_handler",
    "post_origin_allowed",
    "redact_messages",
    "redact_secrets_in_text",
    "request_permission",
    "resolve_checked",
    "resolve_safe_path",
    "scan_messages_for_secrets",
    "scan_text_for_secrets",
    "scan_workspace_secrets",
    "security_status",
    "set_permission_handler",
    "ssrf_error",
    "unacknowledged_findings",
]


class SecurityError(Exception):
    """A guard refused this path/origin: traversal, symlink escape, or a
    state-mutating request from a foreign website."""


# --- 1. workspace path containment ---------------------------------------------

def _has_parent_component(raw: str) -> bool:
    """True when the (mixed-separator) path text contains a `..` component."""
    norm = str(raw or "").replace("\\", "/")
    return any(part == ".." for part in norm.split("/"))


def resolve_safe_path(root, target) -> Path:
    """`target` resolved INSIDE `root`, or `SecurityError`.

    Both sides go through `Path.resolve()` so symlinks are followed before the
    containment test: a link under the project that aims at `/etc/passwd`
    fails the `is_relative_to` check exactly like a literal `../` walk does.
    A literal `..` component is refused outright -- even one that would
    collapse back inside the root -- because a model-supplied path should
    never need the parent directory to say where a project file is."""
    try:
        resolved_root = Path(str(root)).expanduser().resolve()
    except OSError as exc:
        raise SecurityError(f"project root could not be resolved: {root} ({exc})")
    if not resolved_root.is_dir():
        raise SecurityError(f"project root is not a directory: {root}")
    raw = str(target or "").strip().strip("`\"'")
    if not raw:
        raise SecurityError("empty path")
    if _has_parent_component(raw):
        raise SecurityError(f"path traversal refused ('..' is never allowed): {target}")
    cand = Path(raw).expanduser()
    if not cand.is_absolute():
        cand = resolved_root / cand
    try:
        resolved = cand.resolve()
    except OSError as exc:
        raise SecurityError(f"path could not be resolved: {target} ({exc})")
    if not resolved.is_relative_to(resolved_root):
        raise SecurityError(
            f"path escapes the project root (symlink or absolute target): {target}")
    return resolved


# --- 2. sensitive files ---------------------------------------------------------

# Exact file names (lower case) that always hold credentials or keys.
SENSITIVE_NAMES = {
    ".npmrc", ".pypirc", ".netrc", ".git-credentials",
    "credentials.json", "credentials.yml", "credentials.yaml",
    "secrets.json", "secrets.yaml", "secrets.yml",
}
# Globs (matched against the lower-cased file name).
SENSITIVE_PATTERNS = (
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.kdbx",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", "id_sk_ed25519*",
    "*.credentials", "service-account*.json", "*-service-account*.json",
)
# `.env` variants that are templates, not secrets.
ENV_TEMPLATE_NAMES = {".env.example", ".env.sample", ".env.template",
                      ".env.dist", ".env.defaults"}
# Directories whose very name means "credentials live here".
SENSITIVE_DIRS = {".ssh", ".gnupg", ".aws", ".docker", ".config/gcloud"}

# Is `.foo.env` sensitive? Only as an exact name; `app.env` style files are
# the project's own config samples and stay readable.

def is_sensitive_path(path) -> bool:
    """True when `path` names secret/credential material that must never be
    indexed or read into a model prompt without an explicit human approval."""
    p = PurePosixPath(str(path or "").replace("\\", "/"))
    parts = tuple(x.lower() for x in p.parts if x not in ("/", "", "."))
    if not parts:
        return False
    name = parts[-1]
    # `!*.pub`: public keys are safe and must stay readable/indexable -- the
    # check runs FIRST so `id_rsa.pub` never hits the `id_rsa*` pattern.
    if name.endswith(".pub"):
        return False
    # A file under .ssh/.gnupg/.aws/... is sensitive however it is named.
    for d in SENSITIVE_DIRS:
        sub = tuple(d.split("/"))
        for i in range(len(parts) - len(sub) + 1):
            if tuple(parts[i:i + len(sub)]) == sub:
                return True
    if name in ENV_TEMPLATE_NAMES:
        return False
    if name == ".env" or name.startswith(".env."):
        # `.env`, `.env.local`, `.env.production`... all hold live secrets.
        return True
    if name.endswith(".env"):
        return True                       # `prod.env` style dotenv files too
    if name in SENSITIVE_NAMES:
        return True
    for pat in SENSITIVE_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def is_sensitive_rel(root, rel: str) -> bool:
    """`is_sensitive_path` for a project-relative path."""
    return is_sensitive_path(Path(str(root)) / str(rel or ""))


# --- 3. GUI server origin guard ---------------------------------------------------

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "ip6-localhost",
                  "ip6-loopback"}


def origin_host_allowed(value: str, bound_host: str = "127.0.0.1") -> bool:
    """One `Origin`/`Referer` header value: its hostname must be loopback or
    the address the server is bound to. An unparseable value is NOT allowed --
    a browser sends a full URL; anything else is a forged or confused client."""
    if not value:
        return True
    try:
        host = urlparse(value.strip()).hostname
    except ValueError:
        return False
    if not host:
        return False
    host = host.strip("[]").lower().rstrip(".")
    bound = str(bound_host or "").strip("[]").lower()
    return host in LOOPBACK_HOSTS or host == bound


def post_origin_allowed(headers, bound_host: str = "127.0.0.1") -> bool:
    """CSRF guard for state-mutating requests: when a browser sends `Origin`
    or `Referer` on a POST, both (when present) must name a loopback host.
    A request with neither header (curl, the CLI, tests) is unconstrained by
    THIS check -- the server's Host / Content-Type guards still apply."""
    for name in ("Origin", "Referer"):
        value = headers.get(name) if hasattr(headers, "get") else None
        if value and not origin_host_allowed(value, bound_host):
            return False
    return True


# --- 4. SSRF guard for the web tools ------------------------------------------------

def _ip_is_private(ip) -> bool:
    if (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified
            or getattr(ip, "is_site_local", False)
            or not ip.is_global):
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None and _ip_is_private(mapped):
        return True
    sixtofour = getattr(ip, "sixtofour", None)
    if sixtofour is not None and _ip_is_private(sixtofour):
        return True
    teredo = getattr(ip, "teredo", None)
    if teredo is not None and any(_ip_is_private(addr) for addr in teredo):
        return True
    return False


def is_private_host(host: str) -> bool:
    """True when `host` names the local machine, a LAN address or a
    cloud-metadata endpoint: loopback (`127.0.0.1`, `::1`, `localhost`),
    link-local (the 169.254.169.254 metadata IP lives here), RFC1918 /
    unique-local / CGNAT (RFC 6598) / site-local ranges, reserved, multicast
    or unspecified addresses, IPv4-mapped/6to4/Teredo wrappers, and `*.local`
    / `*.localhost` names. Plain public hostnames are not private."""
    host = str(host or "").strip().rstrip(".").strip("[]").rstrip(".").lower()
    if not host:
        return True
    if host in ("localhost", "localhost.localdomain") or host.endswith(
            (".local", ".localhost")):
        return True
    host = host.partition("%")[0]      # an IPv6 zone id is not part of the IP
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Legacy numeric spellings (`2130706433`, `127.1`, `0x7f000001`,
        # `0177.0.0.1`) are not valid ipaddress input, but the OS resolver
        # still turns them into 127.0.0.1 -- normalise before judging.
        try:
            ip = ipaddress.ip_address(socket.inet_ntoa(socket.inet_aton(host)))
        except (OSError, ValueError):
            return False
    return _ip_is_private(ip)


def host_named_explicitly(host: str, allowed_hosts) -> bool:
    """True when `host` is named in `allowed_hosts` (exact or as a `.suffix`),
    with `"*"` matching anything. `ssrf_error`/`resolve_checked` strip the
    wildcard first: it means "any PUBLIC host", never an exemption for a
    private address."""
    host = str(host or "").strip().rstrip(".").lower()
    for a in (allowed_hosts or ()):
        a = str(a).lower().strip().rstrip(".")
        if a == "*" or host == a or host.endswith("." + a):
            return True
    return False


def ssrf_error(url: str, allowed_hosts=()) -> str:
    """'' when `url` may be fetched, else a one-line refusal.

    Only full http(s) URLs pass. Loopback/link-local/metadata/private hosts
    are refused unless named explicitly in `allowed_hosts`, so a
    prompt-injected URL cannot probe the user's network or the cloud metadata
    endpoint. A `"*"` entry means "any public host": it never exempts a
    private address (TECHNIQUES.md §32)."""
    raw = str(url or "").strip().strip("`\"'")
    try:
        u = urlparse(raw)
    except ValueError:
        return f"{url!r} is not a parseable URL"
    if u.scheme not in ("http", "https"):
        return (f"{raw!r} is not an http(s) URL; the web tools fetch "
                "http:// or https:// addresses only")
    host = (u.hostname or "").strip().rstrip(".").strip("[]").rstrip(".").lower()
    if not host:
        return f"{raw!r} has no hostname to fetch"
    named = tuple(a for a in (allowed_hosts or ()) if str(a).strip() != "*")
    if is_private_host(host) and not host_named_explicitly(host, named):
        return (f"{host} is a loopback/link-local/private-network address; "
                "web tools refuse it (SSRF guard) unless the host is named "
                "explicitly in fetch_allowed_hosts")
    return ""


def resolve_checked(host: str, allowed_hosts=()) -> tuple[str, list[str]]:
    """`(error, ips)`: resolve `host` ONCE and judge every address it names.

    `ssrf_error` judges the hostname as written, so a public name that
    *resolves* to a private IP (DNS rebinding) slips past it. This resolves
    the name with a single `getaddrinfo`, runs `is_private_host` on every
    returned IP, and refuses when any is private unless the host is named
    explicitly in `allowed_hosts`. The caller then pins its connection to
    one of the returned IPs, so a second DNS answer cannot swap in a
    private address mid-request.

    IP literals need no resolution. A name that fails to resolve fails
    closed (an error, no addresses). IPv4 sorts first: it is the address
    the pinned connection should prefer. A `"*"` allowlist entry means
    "any PUBLIC host": it never exempts a private address, only a host
    named explicitly (exact or suffix) does."""
    host = str(host or "").strip().rstrip(".").strip("[]").rstrip(".").lower()
    if not host:
        return "no hostname to resolve", []
    named = tuple(a for a in (allowed_hosts or ())
                  if str(a).strip() != "*")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass                                    # a name: resolve it below
    else:
        if is_private_host(host) and not host_named_explicitly(host, named):
            return (f"{host} is a loopback/link-local/private-network "
                    "address; web tools refuse it (SSRF guard) unless the "
                    "host is named explicitly in fetch_allowed_hosts"), []
        return "", [host]
    try:
        infos = socket.getaddrinfo(host, None)
    except (OSError, ValueError) as exc:
        return (f"{host} could not be resolved ({exc}); the web tools "
                "fail closed rather than guess an address"), []
    ips: list[str] = []
    for info in infos:
        ip = str(info[4][0]).partition("%")[0]   # drop any IPv6 zone id
        if ip and ip not in ips:
            ips.append(ip)
    if not ips:
        return f"{host} resolved to no addresses", []
    for ip in ips:
        if is_private_host(ip) and not host_named_explicitly(host, named):
            return (f"{host} resolves to {ip}, a loopback/link-local/"
                    "private-network address; web tools refuse it (SSRF "
                    "guard) unless the host is named explicitly in "
                    "fetch_allowed_hosts"), []
    ips.sort(key=lambda ip: ":" in ip)           # prefer IPv4 for the pin
    return "", ips


# --- 5. the live Permission Broker --------------------------------------------------

CAT_WEB = "web"
CAT_WEB_ACCESS = "web_access"
CAT_PACKAGE_INSTALL = "package_install"
CAT_SHELL = "shell"
CAT_SENSITIVE_FILE = "sensitive_file"
CAT_TEST_EXECUTION = "test_execution"
CAT_MCP_TOOL = "mcp_tool"

CATEGORY_LABELS = {
    CAT_WEB: "Web Access",
    CAT_WEB_ACCESS: "Web Fetch & Search",
    CAT_PACKAGE_INSTALL: "Package Install",
    CAT_SHELL: "Shell Command",
    CAT_SENSITIVE_FILE: "Sensitive File Read",
    CAT_TEST_EXECUTION: "Test Execution",
    CAT_MCP_TOOL: "MCP Tool",
}

# The exact text a gated tool returns when the human says no (or nobody is
# there to ask). Deliberately a redirect, not a crash: the model pivots to
# local files and the standard library instead of retrying forever.
DENIED_MESSAGE = ("[PERMISSION DENIED BY USER: Do not retry this action. "
                  "Proceed using local project files and standard library "
                  "only.]")

# The MCP variant: the decline steers back to the repository tools, not the
# standard library (the declined thing was an external tool call anyway).
MCP_DENIED_MESSAGE = ("[PERMISSION DENIED BY USER: MCP tool execution was "
                      "declined. Proceed using local repository tools only.]")

_hook = None
_hook_lock = threading.Lock()


def install_permission_hook(hook) -> None:
    """Activate the live broker: `hook(info) -> (allowed: bool, message: str)`
    is called for every gated action and BLOCKS until the user decides (the
    GUI installs its own hook here; only the GUI server should)."""
    global _hook
    with _hook_lock:
        _hook = hook


def set_permission_handler(hook) -> None:
    """Alias for `install_permission_hook`: install (or with None, remove) the
    live human-in-the-loop handler. The TUI registers its prompt-bar handler
    here and restores the previous hook on exit."""
    install_permission_hook(hook)


def permission_handler():
    """The currently installed broker hook, or None."""
    return _hook


def permission_broker_active() -> bool:
    """True while an interactive broker is installed (a GUI is listening)."""
    return _hook is not None


def request_permission(category: str, action_summary: str, detail: str = "",
                       risk_level: str = "medium", reason: str = "",
                       key: str = "", setting_key: str = "") -> tuple[bool, str]:
    """Ask the human. Returns `(allowed, message_for_model)`.

    `setting_key` optionally names the Config field an "Always Allow" should
    flip (e.g. `enable_web`), so one modal answer also makes the tool
    standalone-configurable from then on.

    With no broker installed this answers `(False, DENIED_MESSAGE)` -- callers
    check `permission_broker_active()` first and keep their own historical
    error text for the CLI, so this fallback only fires on an uninstall race,
    which must deny in the safe direction."""
    hook = _hook
    info = {"category": str(category or CAT_SHELL),
            "action_summary": str(action_summary or ""),
            "detail": str(detail or "")[:4000],
            "risk_level": str(risk_level or "medium"),
            "reason": str(reason or ""),
            "key": str(key or detail or "")[:500],
            "setting_key": str(setting_key or "")}
    if hook is None:
        return False, DENIED_MESSAGE
    try:
        allowed, message = hook(info)
    except Exception:
        return False, DENIED_MESSAGE
    allowed = bool(allowed)
    message = str(message or "")
    if not allowed and not message:
        message = DENIED_MESSAGE
    return allowed, message


# --- package-install detection --------------------------------------------------------

# The executable (lower-case, no path, no .exe) that installs code packages.
_INSTALL_EXES = {"pip", "pip3", "pipx", "uv", "poetry", "conda", "mamba",
                 "npm", "npx", "pnpm", "yarn", "bun", "deno", "cargo", "go"}
_INSTALL_SUBCOMMANDS = {"install", "add", "get"}


def _exe_name(name: str) -> str:
    """Bare lower-case executable name of argv[0], version suffix and `.exe`
    dropped (`pip3.12.exe` -> `pip`)."""
    base = (PureWindowsPath(name) if "\\" in name else PurePosixPath(name)).name.lower()
    base = base.removesuffix(".exe")
    stem = base.rstrip("0123456789.")
    return stem or base


def is_install_command(argv) -> bool:
    """True when argv installs a package from an index: `pip install X`,
    `python -m pip install X`, `uv pip install`, `npm i X`, `yarn add X`,
    `cargo add`, `go get`... These fetch and execute third-party code, so
    they need their own approval even with `allow_shell` on."""
    argv = [str(a) for a in (argv or [])]
    if not argv:
        return False
    exe = _exe_name(argv[0])
    rest = argv[1:]
    # Interpreters running an installer module: `python -m pip install`,
    # `node /usr/local/bin/npm install`...
    if exe in ("python", "python3", "py", "python2"):
        if len(rest) >= 3 and rest[0] == "-m" and _exe_name(rest[1]) in _INSTALL_EXES:
            return rest[2].lower() in _INSTALL_SUBCOMMANDS or (
                _exe_name(rest[1]) in ("npm", "npx") and rest[2].lower() in ("i", "ci", "install", "add"))
        # `python /path/to/npm install x` style: an installer path as script.
        if len(rest) >= 2 and _exe_name(rest[0]) in _INSTALL_EXES:
            return rest[1].lower() in _INSTALL_SUBCOMMANDS | {"i", "ci"}
        return False
    if exe not in _INSTALL_EXES:
        return False
    # Skip flags (`pip --quiet install x`).
    words = [w for w in rest if not w.startswith("-")]
    first = words[0].lower() if words else ""
    if exe in ("npm", "npx", "pnpm", "bun", "deno"):
        return first in ("i", "ci", "install", "add", "get")
    if exe in ("yarn",):
        return first in ("add", "install") or (
            first in ("global",) and len(words) > 1
            and words[1].lower() in ("add", "install"))
    if exe == "cargo":
        return first in _INSTALL_SUBCOMMANDS
    if exe == "go":
        return first in ("get", "install")
    if exe == "uv":
        # `uv pip install`, `uv add`, `uv tool install`, `uv venv` (not a pkg)
        if first == "pip":
            return len(words) > 1 and words[1].lower() in ("install",)
        return first in ("add", "install", "tool") or (
            first == "sync")          # uv sync pulls the declared environment
    if exe == "poetry":
        return first in ("add", "install", "sync")
    if exe in ("conda", "mamba"):
        return first in ("install", "create")     # envs also download packages
    if exe == "pipx":
        return first in ("install", "ensurepip", "inject")
    # pip / pip3
    return first in _INSTALL_SUBCOMMANDS or first == "install"


# =====================================================================
# Non-blocking secret & credential guard
# =====================================================================
#
# The scanner below is deliberately fast and local: plain regular
# expressions over text already on disk or already about to leave the
# machine. Nothing here ever blocks a run without the user's explicit
# choice, and a finding never carries the raw secret into a UI payload or
# an acknowledgment file. Severe findings are represented by a masked
# preview plus a SHA-256 fingerprint.

REDACTED_SECRET = "<REDACTED_SECRET>"
OUTBOUND_MODES = ("warn", "redact", "allow")

# Workspace scanning bounds: generous enough for a real project, small
# enough that the GUI's status poll never stalls on a giant repository.
_MAX_SCAN_FILES = 2500
_MAX_SCAN_BYTES = 1_000_000
_SKIP_SCAN_DIRS = {
    ".git", ".hg", ".svn", ".pedacito", "node_modules", "__pycache__",
    ".venv", "venv", "env", ".tox", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", "dist", "build", ".next", ".nuxt", "target",
}

# Rule order matters only for overlap suppression: specific token formats
# win over the generic assignment rule.
_TOKEN_RULES = (
    ("api_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("pypi_token", re.compile(r"pypi-[A-Za-z0-9_-]{20,}")),
    ("github_token", re.compile(
        r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}"
        r"|github_pat_[A-Za-z0-9_]{22,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("slack_token", re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}")),
    ("private_key", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
)

_GENERIC_ASSIGNMENT_RE = re.compile(
    r"(?P<key>\b(?:api[_-]?key|secret|token|password|passwd|authorization)\b)"
    r"(?P<sep>\s*[:=]\s*)"
    r"(?P<quote>[\"'])"
    r"(?P<value>[A-Za-z0-9_/+=-]{16,})"
    r"(?P=quote)",
    re.IGNORECASE,
)

_PEM_END_RE = re.compile(
    r"-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")

_PLACEHOLDER_WORDS = (
    "your_", "your-", "yourkey", "changeme", "change_me", "change-me",
    "example", "placeholder", "dummy", "sample", "redacted", "replace_me",
    "replace-me", "insert_", "insert-", "not_a_real", "not-a-real",
    "fake_", "fake-", "test_", "test-",
)

# File names whose presence in a repository is itself a hygiene problem
# when `.gitignore` does not exclude them (derived from the canonical lists).
_SENSITIVE_FILE_NAMES = SENSITIVE_NAMES | {
    ".env", ".env.local", ".env.production", "id_rsa", "id_ed25519",
}
_ENV_TEMPLATE_NAMES = ENV_TEMPLATE_NAMES
_ACK_SUBPATH = Path(".pedacito") / "security_ack.json"
_ACK_RE = re.compile(r"\A[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SecretFinding:
    """One detected secret. Only `file`, `line`, `kind`, `masked_preview`
    and `fingerprint` are safe to serialise: raw match fields are excluded
    from repr and never leave the scanner."""

    file: str
    line: int
    kind: str
    masked_preview: str
    fingerprint: str
    # Internal redaction data. repr/compare off so logging a finding can
    # never print the raw secret.
    raw_match: str = field(default="", repr=False, compare=False)
    secret: str = field(default="", repr=False, compare=False)
    start: int = field(default=0, repr=False, compare=False)
    end: int = field(default=0, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "kind": self.kind,
            "masked_preview": self.masked_preview,
            "fingerprint": self.fingerprint,
        }

    # A few call sites prefer the longer name.
    as_dict = to_dict


@dataclass(frozen=True)
class SensitiveFileFinding:
    """A sensitive file that exists but is not excluded by `.gitignore`."""

    file: str
    kind: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"file": self.file, "kind": self.kind, "detail": self.detail}

    as_dict = to_dict


def _masked_preview(value: str, kind: str) -> str:
    """A safe snippet: a short head, an ellipsis, and the last four chars.
    PEM blocks show only their header (the body is the private material)."""
    value = " ".join(str(value or "").split())
    if kind == "private_key":
        return "-----BEGIN ... PRIVATE KEY-----"
    if len(value) <= 8:
        return "***"
    head_len = min(7, max(3, len(value) // 6))
    return f"{value[:head_len]}...{value[-4:]}"


def _looks_like_placeholder(value: str) -> bool:
    """True for obvious stand-ins, env lookups and documentation samples."""
    raw = str(value or "").strip()
    low = raw.lower()
    if not raw:
        return True
    if raw.startswith("<") and raw.endswith(">"):
        return True
    if low.startswith(("os.environ", "getenv(", "process.env", "${", "{{",
                       "env:", "$")):
        return True
    if low in {"changeme", "change_me", "password", "secret", "token",
               "example", "sample", "dummy", "xxxx", "xxx", "none", "null"}:
        return True
    if any(word in low for word in _PLACEHOLDER_WORDS):
        return True
    if re.fullmatch(r"[xX]+", raw) or re.fullmatch(r"\*+", raw):
        return True
    if raw.startswith("***") or "..." in raw or "\u2026" in raw:
        return True
    if re.fullmatch(r"\.+", raw):
        return True
    return False


def _fingerprint(source: str, kind: str, raw_match: str) -> str:
    material = f"{source}:{kind}:{raw_match}".encode("utf-8", "replace")
    return hashlib.sha256(material).hexdigest()


def _overlaps(start: int, end: int, spans) -> bool:
    return any(start < e and end > s for s, e in spans)


def _private_key_end(text: str, start: int) -> int:
    """End of a PEM private-key block, or end of contiguous key lines when unclosed."""
    m = _PEM_END_RE.search(text, start)
    if m:
        return m.end()
    pos = start
    n = len(text)
    while pos < n and text[pos] in " \t":
        pos += 1
    if pos < n and text[pos] in "\r\n":
        pos += 2 if text[pos:pos + 2] == "\r\n" else 1
    last_key_end = start
    saw_armor_header = False
    while pos < n:
        nl = text.find("\n", pos)
        line_end = nl if nl != -1 else n
        line = text[pos:line_end].rstrip("\r").strip(" \t")
        if not line:
            if saw_armor_header:
                saw_armor_header = False
                pos = line_end + 1 if nl != -1 else n
                continue
            break
        if re.fullmatch(r"[A-Za-z0-9+/=_-]+", line):
            saw_armor_header = False
            last_key_end = line_end
            pos = line_end + 1 if nl != -1 else n
            continue
        if re.fullmatch(r"(?:Proc-Type|DEK-Info):[^\r\n]+", line, re.IGNORECASE):
            saw_armor_header = True
            last_key_end = line_end
            pos = line_end + 1 if nl != -1 else n
            continue
        break
    return last_key_end


def scan_text_for_secrets(text: str, source_label: str = "<prompt>") -> list[SecretFinding]:
    """Find known high-confidence credentials in `text`.

    `source_label` is the relative file path, or `"<prompt>"` for text that
    is not a file. Findings are ordered by position and include a masked
    preview only; the raw match is kept in memory for redaction and is
    excluded from `SecretFinding.to_dict()`.
    """
    source = str(source_label or "<prompt>")
    body = str(text or "")
    if not body:
        return []

    candidates: list[tuple[int, int, int, int, str, str, str]] = []
    # (match_start, match_end, secret_start, secret_end, kind, raw, secret)
    occupied: list[tuple[int, int]] = []

    for kind, rx in _TOKEN_RULES:
        for m in rx.finditer(body):
            raw = m.group(0)
            start, end = m.start(), m.end()
            secret_start, secret_end = start, end
            if kind == "private_key":
                secret_end = _private_key_end(body, end)
                secret_end = max(secret_end, end)
            raw = body[start:end]
            candidates.append(
                (start, end, secret_start, secret_end, kind, raw,
                 body[secret_start:secret_end]))
            occupied.append((start, end))

    for m in _GENERIC_ASSIGNMENT_RE.finditer(body):
        start, end = m.start("value"), m.end("value")
        if _overlaps(start, end, occupied):
            continue                     # already reported by a token rule
        value = m.group("value")
        if _looks_like_placeholder(value):
            continue
        candidates.append(
            (m.start(), m.end(), start, end, "credential_assignment",
             m.group(0), value))

    # Deduplicate identical spans and sort by position.
    seen: set[tuple[int, int, str]] = set()
    findings: list[SecretFinding] = []
    for match_start, match_end, sec_start, sec_end, kind, raw, secret in sorted(
            candidates, key=lambda c: (c[0], c[1], c[4])):
        key = (sec_start, sec_end, kind)
        if key in seen:
            continue
        seen.add(key)
        findings.append(SecretFinding(
            file=source,
            line=body.count("\n", 0, match_start) + 1,
            kind=kind,
            masked_preview=_masked_preview(secret, kind),
            fingerprint=_fingerprint(source, kind, raw),
            raw_match=raw,
            secret=secret,
            start=sec_start,
            end=sec_end,
        ))
    return findings


def _iter_workspace_files(root: Path):
    """Bounded text-ish file walk; ignores VCS/venv/build trees and the
    guard's own state directory."""
    stack = [Path(root)]
    seen = 0
    while stack and seen < _MAX_SCAN_FILES:
        directory = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for entry in entries:
            if seen >= _MAX_SCAN_FILES:
                return
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if entry.name.lower() not in _SKIP_SCAN_DIRS:
                        stack.append(entry)
                    continue
                if not entry.is_file():
                    continue
                if entry.stat().st_size > _MAX_SCAN_BYTES:
                    continue
                seen += 1
                yield entry
            except OSError:
                continue


def _read_scan_text(path: Path) -> str | None:
    try:
        with path.open("rb") as fh:
            data = fh.read(_MAX_SCAN_BYTES + 1)
    except OSError:
        return None
    if len(data) > _MAX_SCAN_BYTES or b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def scan_workspace_secrets(root, files=None) -> list[SecretFinding]:
    """Scan workspace files for credentials. `files` limits the scan to the
    named relative paths (used for a run's changed files); without it the
    whole tree is walked under the standard bounds."""
    if root is None:
        return []
    base = Path(str(root))
    if not base.is_dir():
        return []
    try:
        base_resolved = base.resolve()
    except OSError:
        return []
    if files is None:
        paths = list(_iter_workspace_files(base))
    else:
        paths = []
        for rel in files:
            rel = str(rel or "").strip()
            if not rel:
                continue
            candidate = Path(rel)
            if not candidate.is_absolute():
                candidate = base / candidate
            try:
                resolved = candidate.resolve()
                if not resolved.is_relative_to(base_resolved) or not resolved.is_file():
                    continue
                if resolved.stat().st_size > _MAX_SCAN_BYTES:
                    continue
            except OSError:
                continue
            paths.append(resolved)
    findings: list[SecretFinding] = []
    for path in paths:
        try:
            label = path.resolve().relative_to(base_resolved).as_posix()
        except (OSError, ValueError):
            label = path.name
        text = _read_scan_text(path)
        if text:
            findings.extend(scan_text_for_secrets(text, label))
    return findings


def _content_text(content) -> str:
    """Plain text of a message content field (string or multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def scan_messages_for_secrets(messages, root=None) -> list[SecretFinding]:
    """Scan every message's text content and tool/function arguments for unacknowledged secrets."""
    findings: list[SecretFinding] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        text = _content_text(message.get("content"))
        if text:
            findings.extend(scan_text_for_secrets(text, "<prompt>"))
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")
                if isinstance(fn, dict):
                    args = fn.get("arguments")
                    if isinstance(args, str) and args:
                        findings.extend(scan_text_for_secrets(args, "<prompt>"))
        fn_call = message.get("function_call")
        if isinstance(fn_call, dict):
            args = fn_call.get("arguments")
            if isinstance(args, str) and args:
                findings.extend(scan_text_for_secrets(args, "<prompt>"))
    return unacknowledged_findings(findings, root)


def _ack_key(root) -> str:
    if root is None:
        return ""
    try:
        return str(Path(str(root)).expanduser().resolve())
    except (OSError, ValueError):
        return str(root)


_SESSION_ACK: dict[str, set[str]] = {}
_ACK_LOCK = threading.RLock()


def _normalise_fingerprints(values) -> set[str]:
    out: set[str] = set()
    if isinstance(values, (str, SecretFinding)):
        values = [values]
    for item in values or ():
        raw = item.fingerprint if isinstance(item, SecretFinding) else str(item or "")
        raw = raw.strip().lower()
        if _ACK_RE.match(raw):
            out.add(raw)
    return out


def _read_ack_file(root) -> set[str]:
    path = Path(str(root)) / _ACK_SUBPATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if isinstance(data, list):
        values = data
    elif isinstance(data, dict):
        values = data.get("acknowledged")
        if values is None:
            values = data.get("fingerprints")
        if not isinstance(values, list):
            values = []
    else:
        values = []
    return _normalise_fingerprints(values)


def load_acknowledged_fingerprints(root) -> set[str]:
    """Disk acknowledgments for `root` plus this process's session set."""
    key = _ack_key(root)
    with _ACK_LOCK:
        session = set(_SESSION_ACK.get(key, set()))
    disk = _read_ack_file(root) if root is not None else set()
    return disk | session


def acknowledge_findings(root, fingerprints) -> set[str]:
    """Persist only SHA-256 fingerprints, never a raw secret.

    Returns the workspace's full acknowledged set. A missing/unwritable
    `.pedacito` is non-fatal: the in-memory session set still suppresses
    the warning for the rest of this process.
    """
    wanted = _normalise_fingerprints(fingerprints)
    key = _ack_key(root)
    # One lock covers the session set and the file so two threads answering
    # the same preflight cannot lose each other's fingerprints.
    with _ACK_LOCK:
        session = _SESSION_ACK.setdefault(key, set())
        session.update(wanted)
        clear_security_cache(root)
        if root is None or not wanted:
            return load_acknowledged_fingerprints(root)
        path = Path(str(root)) / _ACK_SUBPATH
        merged = _read_ack_file(root) | session
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {"version": 1, "acknowledged": sorted(merged)},
                indent=1) + "\n"
            fileio.write_bytes(path, payload.encode("utf-8"))
        except OSError:
            # Keep the session acknowledgment even when the file cannot land.
            pass
        return merged


def unacknowledged_findings(findings, root=None) -> list[SecretFinding]:
    """`findings` minus fingerprints the user already acknowledged."""
    acked = load_acknowledged_fingerprints(root) if root is not None else set()
    return [f for f in findings or () if f.fingerprint not in acked]


def _redact_findings(text: str, findings) -> tuple[str, int]:
    spans = sorted({(f.start, f.end) for f in findings
                    if 0 <= f.start < f.end <= len(text)})
    if not spans:
        return text, 0
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    out, pos = [], 0
    for start, end in merged:
        out.append(text[pos:start])
        out.append(REDACTED_SECRET)
        pos = end
    out.append(text[pos:])
    return "".join(out), len(merged)


def redact_secrets_in_text(text: str, root=None, *,
                           redact_acknowledged: bool = False) -> str:
    """Replace unacknowledged raw secrets with `<REDACTED_SECRET>`.

    With no `root` every detected secret is redacted (there is no
    acknowledgment store to consult). Pass `redact_acknowledged=True` to
    force redaction even for a finding the user marked as ignored.
    """
    body = str(text or "")
    findings = scan_text_for_secrets(body)
    if not findings:
        return body
    if root is not None and not redact_acknowledged:
        findings = unacknowledged_findings(findings, root)
    redacted, _count = _redact_findings(body, findings)
    return redacted


def redact_messages(messages, root=None):
    """Return `(messages, count, findings)` with unacknowledged secrets
    replaced in every text content and tool/function argument field."""
    out: list[dict] = []
    total = 0
    found: list[SecretFinding] = []
    acked = load_acknowledged_fingerprints(root) if root is not None else set()
    for message in messages or []:
        if not isinstance(message, dict):
            out.append(message)
            continue
        copy = dict(message)
        content = copy.get("content")
        if isinstance(content, str):
            findings = [f for f in scan_text_for_secrets(content, "<prompt>")
                        if f.fingerprint not in acked]
            if findings:
                copy["content"], count = _redact_findings(content, findings)
                total += count
                found.extend(findings)
        elif isinstance(content, list):
            parts = []
            changed = False
            for part in content:
                if isinstance(part, dict):
                    part = dict(part)
                    text = part.get("text")
                    if isinstance(text, str):
                        findings = [f for f in scan_text_for_secrets(
                            text, "<prompt>") if f.fingerprint not in acked]
                        if findings:
                            part["text"], count = _redact_findings(text, findings)
                            total += count
                            found.extend(findings)
                            changed = True
                parts.append(part)
            if changed:
                copy["content"] = parts
        tool_calls = copy.get("tool_calls")
        if isinstance(tool_calls, list):
            new_calls = []
            calls_changed = False
            for call in tool_calls:
                if isinstance(call, dict):
                    call_copy = dict(call)
                    fn = call_copy.get("function")
                    if isinstance(fn, dict):
                        args = fn.get("arguments")
                        if isinstance(args, str) and args:
                            findings = [f for f in scan_text_for_secrets(
                                args, "<prompt>") if f.fingerprint not in acked]
                            if findings:
                                fn_copy = dict(fn)
                                fn_copy["arguments"], count = _redact_findings(
                                    args, findings)
                                call_copy["function"] = fn_copy
                                total += count
                                found.extend(findings)
                                calls_changed = True
                    new_calls.append(call_copy)
                else:
                    new_calls.append(call)
            if calls_changed:
                copy["tool_calls"] = new_calls
        fn_call = copy.get("function_call")
        if isinstance(fn_call, dict):
            args = fn_call.get("arguments")
            if isinstance(args, str) and args:
                findings = [f for f in scan_text_for_secrets(
                    args, "<prompt>") if f.fingerprint not in acked]
                if findings:
                    fn_call_copy = dict(fn_call)
                    fn_call_copy["arguments"], count = _redact_findings(
                        args, findings)
                    copy["function_call"] = fn_call_copy
                    total += count
                    found.extend(findings)
        out.append(copy)
    return out, total, found


# --- .gitignore hygiene -------------------------------------------------

def _gitignore_patterns(root: Path) -> list[str]:
    path = Path(root) / ".gitignore"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def _pattern_matches(rel: str, pattern: str) -> bool:
    """Practical gitignore matching for the paths this guard checks."""
    pat = pattern.strip()
    if not pat:
        return False
    if pat.startswith("!"):
        return False
    pat = pat.strip()
    if pat.startswith("/"):
        pat = pat[1:]
    dir_only = pat.endswith("/")
    pat = pat.rstrip("/")
    if not pat:
        return False
    parts = PurePosixPath(rel).parts
    if not parts:
        return False
    if "/" in pat:
        # Anchored or path-shaped: match the whole relative path or a
        # directory prefix (a trailing "/" is the usual directory form).
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, f"**/{pat}"):
            return True
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i])
            if fnmatch.fnmatch(prefix, pat):
                return True
        return False
    # A bare name matches a component at any depth.
    matched = any(fnmatch.fnmatch(part, pat) for part in parts)
    if not matched and dir_only:
        return False
    return matched


def _path_ignored(root: Path, rel: str) -> bool:
    ignored = False
    for raw in _gitignore_patterns(root):
        negated = raw.startswith("!")
        pattern = raw[1:] if negated else raw
        if _pattern_matches(str(rel), pattern):
            ignored = not negated
    return ignored


def _sensitive_file_kind(name: str) -> str:
    low = name.lower()
    if low == ".env" or low.startswith(".env.") or low.endswith(".env"):
        return "unignored_env"
    if (low.startswith("id_")
            or low.endswith((".pem", ".key", ".p12", ".pfx", ".jks", ".kdbx"))):
        return "unignored_private_key"
    return "unignored_credentials"


def check_git_secret_hygiene(root) -> list[SensitiveFileFinding]:
    """Sensitive files present in the workspace but not ignored by
    `.gitignore`. Empty when there is nothing to fix (or no workspace)."""
    if root is None:
        return []
    base = Path(str(root))
    if not base.is_dir():
        return []
    try:
        base_resolved = base.resolve()
    except OSError:
        return []
    out: list[SensitiveFileFinding] = []
    seen: set[str] = set()
    for path in _iter_workspace_files(base):
        name = path.name
        try:
            rel = path.resolve().relative_to(base_resolved).as_posix()
        except (OSError, ValueError):
            rel = name
        if not is_sensitive_path(rel):
            continue
        if rel in seen:
            continue
        seen.add(rel)
        if _path_ignored(base, rel):
            continue
        out.append(SensitiveFileFinding(
            file=rel,
            kind=_sensitive_file_kind(name),
            detail=f"{rel} exists but .gitignore does not exclude it",
        ))
    return sorted(out, key=lambda f: f.file)


def ensure_gitignore_entries(root, patterns=None) -> list[str]:
    """Append missing sensitive-file patterns to the workspace's
    `.gitignore`; returns only the patterns actually written."""
    if root is None:
        return []
    patterns = list(patterns) if patterns is not None else [".env", ".env.*"]
    patterns = [str(p).strip() for p in patterns if str(p).strip()]
    if not patterns:
        return []
    base = Path(str(root))
    if not base.is_dir():
        return []
    path = base / ".gitignore"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    existing = {ln.strip() for ln in text.splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")}

    def covered(pattern: str) -> bool:
        if pattern in existing or pattern.rstrip("/") in existing:
            return True
        # A broader existing pattern (".env*", "*.env") already covers it.
        for line in existing:
            if line.startswith("!"):
                continue
            probe = line[1:] if line.startswith("/") else line
            if fnmatch.fnmatch(pattern.rstrip("/"), probe.rstrip("/")):
                return True
        return False

    to_add = [p for p in patterns if not covered(p)]
    if not to_add:
        return []
    try:
        with path.open("a", encoding="utf-8") as fh:
            if text and not text.endswith("\n"):
                fh.write("\n")
            if text:
                fh.write("\n")
            fh.write("# Pedacito: keep secret files out of git\n")
            for pattern in to_add:
                fh.write(f"{pattern}\n")
    except OSError:
        return []
    clear_security_cache(root)
    return to_add


# --- status, notifications and outbound policy --------------------------

_STATUS_CACHE: dict[str, tuple[float, dict]] = {}
_STATUS_LOCK = threading.Lock()
_STATUS_TTL = 20.0


def outbound_mode(cfg) -> str:
    """The workspace's outbound policy, normalised to a known value."""
    mode = str(getattr(cfg, "security_outbound_mode", "warn") or "warn").strip().lower()
    return mode if mode in OUTBOUND_MODES else "warn"


def clear_security_cache(root=None) -> None:
    """Drop cached status (after a run, an acknowledgment, or a fix)."""
    with _STATUS_LOCK:
        if root is None:
            _STATUS_CACHE.clear()
            return
        _STATUS_CACHE.pop(_ack_key(root), None)


def _status_payload(root, cfg, findings, hygiene) -> dict:
    mode = outbound_mode(cfg)
    return {
        "root": str(root) if root is not None else "",
        "mode": mode,
        "findings": [f.to_dict() for f in findings],
        "unignored_sensitive_files": [h.to_dict() for h in hygiene],
        "unignored_env": [h.file for h in hygiene
                          if h.kind == "unignored_env"],
        "secrets_found": bool(findings),
        "env_unignored": bool(hygiene),
        "count": len(findings),
    }


def security_status(root, cfg=None, files=None, use_cache: bool = False) -> dict:
    """Workspace security summary: unacknowledged secrets plus unignored
    sensitive files. `files` limits the secret scan to a run's changed
    files; `use_cache` is for the GUI's polling status payload."""
    key = _ack_key(root)
    if use_cache and files is None and root is not None:
        with _STATUS_LOCK:
            entry = _STATUS_CACHE.get(key)
            if entry and time.monotonic() - entry[0] <= _STATUS_TTL:
                return entry[1]
    findings = unacknowledged_findings(
        scan_workspace_secrets(root, files=files), root)
    hygiene = check_git_secret_hygiene(root)
    payload = _status_payload(root, cfg, findings, hygiene)
    if files is None and root is not None:
        with _STATUS_LOCK:
            _STATUS_CACHE[key] = (time.monotonic(), payload)
    return payload


def format_finding(finding) -> str:
    """One-line, masked finding suitable for a terminal or GUI banner."""
    if isinstance(finding, dict):
        return (f"{finding.get('file', '?')}:{finding.get('line', '?')} "
                f"[{finding.get('kind', '?')}] "
                f"{finding.get('masked_preview', '***')}")
    return (f"{finding.file}:{finding.line} [{finding.kind}] "
            f"{finding.masked_preview}")


def emit_post_run_notice(root, cfg=None, files=None) -> dict:
    """Scan after a run and emit a non-blocking `security_notice` event.

    Returns the same payload `security_status` reports. The event only
    carries masked findings; raw secrets never enter the event log.
    """
    status = security_status(root, cfg, files=files, use_cache=False)
    clear_security_cache(root)
    if not (status["secrets_found"] or status["env_unignored"]):
        return status
    message = (f"{status['count']} unacknowledged secret(s) and "
               f"{len(status['unignored_env'])} unignored .env file(s) "
               "detected in the workspace")
    try:
        from . import events as events_mod
        events_mod.get(root, cfg).emit(
            "security_notice",
            message=message,
            findings=status["findings"],
            unignored_env=status["unignored_env"],
        )
    except Exception:
        pass
    return status
