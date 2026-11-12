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
import ipaddress
import socket
import threading
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlparse

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
    "SecurityError",
    "install_permission_hook",
    "is_install_command",
    "is_private_host",
    "is_sensitive_path",
    "is_sensitive_rel",
    "permission_broker_active",
    "post_origin_allowed",
    "request_permission",
    "resolve_checked",
    "resolve_safe_path",
    "ssrf_error",
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

def is_private_host(host: str) -> bool:
    """True when `host` names the local machine, a LAN address or a
    cloud-metadata endpoint: loopback (`127.0.0.1`, `::1`, `localhost`),
    link-local (the 169.254.169.254 metadata IP lives here), RFC1918 /
    unique-local LAN ranges, reserved, multicast or unspecified addresses, and
    `*.local` names. Plain public hostnames are not private."""
    host = str(host or "").strip().lower().strip("[]")
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
    return (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def host_named_explicitly(host: str, allowed_hosts) -> bool:
    """True when `host` is named in `allowed_hosts` (exact or as a `.suffix`,
    with `"*"` matching anything)."""
    host = str(host or "").lower()
    for a in (allowed_hosts or ()):
        a = str(a).lower().strip()
        if a == "*" or host == a or host.endswith("." + a):
            return True
    return False


def ssrf_error(url: str, allowed_hosts=()) -> str:
    """'' when `url` may be fetched, else a one-line refusal.

    Only full http(s) URLs pass. Loopback/link-local/metadata/private hosts
    are refused unless named explicitly in `allowed_hosts`, so a
    prompt-injected URL cannot probe the user's network or the cloud metadata
    endpoint."""
    raw = str(url or "").strip().strip("`\"'")
    try:
        u = urlparse(raw)
    except ValueError:
        return f"{url!r} is not a parseable URL"
    if u.scheme not in ("http", "https"):
        return (f"{raw!r} is not an http(s) URL; the web tools fetch "
                "http:// or https:// addresses only")
    host = (u.hostname or "").lower().strip("[]")
    if not host:
        return f"{raw!r} has no hostname to fetch"
    if is_private_host(host) and not host_named_explicitly(
            host, allowed_hosts):
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
    host = str(host or "").strip().lower().strip("[]")
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
