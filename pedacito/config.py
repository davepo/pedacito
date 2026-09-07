"""Configuration: the Config dataclass, TOML loading, and profile resolution.

Usage
-----
    from .config import load

    cfg = load(root=project_root, explicit=None, profile="qwen")
    print(cfg.base_url, cfg.model)

Settings come from a TOML file. Precedence, lowest to highest:

    1. built-in defaults
    2. the user config file            your machine: LM Studio url, model
         Linux/macOS: ~/.config/pedacito/config.toml
         Windows:     %APPDATA%\\pedacito\\config.toml
    3. <project>/.pedacito.toml           per-project overrides
    4. [profiles.NAME] in any of those, selected with --profile NAME
    5. --config FILE                    an explicit file
    6. command-line flags

Put the connection details in the user config once and forget them. Per-project
files are for things that genuinely differ between projects, like
`extra_extensions` or a tighter `max_steps`.

Unknown keys are reported rather than ignored, because a silently-dropped typo
in a config file is a miserable thing to debug.
"""

from __future__ import annotations

import os
import tomllib
import difflib
from dataclasses import dataclass, fields
from pathlib import Path


# --------------------------------------------------------------------------
# User config file location
# --------------------------------------------------------------------------
def _user_config_dir() -> Path:
    """Return the per-user config directory: %APPDATA%\\pedacito on Windows,
    $XDG_CONFIG_HOME/pedacito (or ~/.config/pedacito) elsewhere."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / "pedacito"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "pedacito"


USER_CONFIG = _user_config_dir() / "config.toml"
PROJECT_CONFIG_NAMES = (".pedacito.toml", "pedacito.toml")


# --------------------------------------------------------------------------
# The settings themselves
# --------------------------------------------------------------------------
@dataclass
class Config:
    """All Pedacito settings, with their defaults. See `load()` for how a
    Config is actually assembled from files, profiles, and env vars."""

    # --- LM Studio connection -------------------------------------------------
    # Over Tailscale this is the tailnet IP or MagicDNS name of the box running
    # LM Studio, e.g. "http://desktop.tail1234.ts.net:1234/v1".
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = "google/gemma-3-27b"
    timeout: float = 600.0
    # How long to keep waiting while LM Studio loads a model. Switching models
    # means unloading the old one and reading 15GB or so off disk, so the useful
    # unit here is minutes, not retries.
    load_wait_seconds: int = 240
    # Stop a batch after this many failures in a row rather than failing 55
    # times identically and writing an index full of blank summaries.
    max_consecutive_failures: int = 3

    # --- Generation -----------------------------------------------------------
    # Low temp for tool selection; a 27B at 0.2 is fine for prose and code too.
    temperature: float = 0.2
    max_tokens_summary: int = 220
    # One tool call plus a short note. Models that narrate a plan in the
    # `reasoning` field need more; the schema puts the action first so a
    # truncated reply still loses only the explanation.
    max_tokens_step: int = 700
    max_tokens_answer: int = 3000

    # --- Map tiering ----------------------------------------------------------
    # If the full flat map fits under this, use it: when it fits it is strictly
    # more useful. Above it, fall back to per-file cards and let the model
    # descend with `outline`. 4k leaves room for gathered source inside a ~10k
    # working window, which is about a 27B's usable reasoning span.
    flat_map_token_budget: int = 4000
    # Attach L<start>-<end> to names on file cards. Costs ~40% more characters
    # in tier 0 and saves an `outline` round trip, the better trade at ~25 tok/s.
    file_card_line_ranges: bool = True

    # --- Indexing -------------------------------------------------------------
    # Classes shorter than this are indexed as one chunk instead of per-method.
    class_split_lines: int = 60
    # Chunks larger than this get their source truncated before summarising.
    max_chunk_chars: int = 6000
    # Skip summarising trivially short chunks; the signature says enough.
    min_lines_to_summarise: int = 4
    # Summarisation rationing. On 452KB of stdlib these two flags took the first
    # index from 442 LLM calls (~24 min) down to 76 (~4 min) with no meaningful
    # loss: a docstring is already a summary, and a method is covered by its
    # class summary plus its signature.
    summarise_methods: bool = False
    summarise_documented: bool = False

    # --- File selection -------------------------------------------------------
    # Non-Python files (.md .csv .json .toml .ini .yaml .txt .sql .sh ...) are
    # indexed structurally: the map shows a CSV's columns or a config's sections
    # so the model knows what is there, and can `read` or `grep` it on demand.
    # It never pulls their contents into the map.
    include_references: bool = True
    # Extra extensions to treat as reference files, e.g. [".jinja", ".proto"].
    extra_extensions: tuple[str, ...] = ()
    # Above this, a reference file is recorded by name and header only.
    max_reference_bytes: int = 2_000_000
    # Reference files get a shape line for free; an LLM summary on top is rarely
    # worth the minute it costs.
    summarise_references: bool = False

    # --- Agent loop -----------------------------------------------------------
    max_steps: int = 6
    # Hard cap on how much source the gather phase may accumulate. A 27B's
    # usable reasoning window is well under its advertised context.
    max_gathered_chars: int = 24000
    max_read_lines: int = 200

    # --- Reasoning models -----------------------------------------------------
    # Qwen3 and friends emit a <think> block by default. Pedacito strips it either
    # way, but generating it still costs tokens and time. These two switches
    # turn it off; which one works depends on the model and the runtime, so both
    # are available and setting both is harmless.
    #   disable_thinking -> sends chat_template_kwargs {"enable_thinking": false}
    #   thinking_suffix  -> appends a token like "/no_think" to your last message
    disable_thinking: bool = False
    thinking_suffix: str = ""

    # --- Editing --------------------------------------------------------------
    # Snapshot originals into .pedacito/backups/<timestamp>/ before writing.
    # `pedacito restore` puts them back.
    backup: bool = True
    # Write a readable record of each task to .pedacito/sessions/<timestamp>.md.
    log_sessions: bool = True
    # Open a side-by-side review page instead of printing a diff. The page is
    # served from 127.0.0.1 only -- it can write files, so it has no business
    # on a network interface.
    review: bool = False
    review_port: int = 0          # 0 lets the OS pick a free port
    open_browser: bool = True
    review_timeout: int = 1800    # seconds before an unanswered review expires

    # --- Workspace ------------------------------------------------------------
    # On first run, add .pedacito/ to the project's .gitignore so the index and
    # backups never end up in a commit.
    manage_gitignore: bool = True
    # On first run, create a commented .pedacitoignore for you to edit.
    create_ignore_file: bool = True

    # Profile applied when --profile is not given. Set it in your user config
    # for a lasting default, or export PEDACITO_PROFILE for one shell session.
    default_profile: str = ""

    # Bookkeeping for `pedacito config`. Not settings.
    sources: dict = None
    profiles: dict = None
    active_profile: str = ""

    def __post_init__(self):
        """Initialise the mutable bookkeeping dicts (dataclasses cannot use
        mutable defaults directly)."""
        if self.sources is None:
            self.sources = {}
        if self.profiles is None:
            self.profiles = {}


# TOML tables are allowed but optional: [lmstudio] base_url = ... works, and so
# does a flat base_url = ... at the top level.
# "server" is the documented section name for connection settings; "lmstudio"
# is kept as an accepted alias so existing config files keep working. Pedacito
# talks to any server exposing an OpenAI-compatible /v1 surface (LM Studio,
# Ollama, and others), so "server" describes it without implying one product.
_SECTIONS = ("server", "lmstudio", "generation", "map", "indexing", "files",
             "agent", "editing", "reasoning", "review")
PROFILE_SECTION = "profiles"

_FIELDS = {f.name: f for f in fields(Config)
           if f.name not in ("sources", "profiles", "active_profile")}


class ConfigError(RuntimeError):
    """Raised for any problem loading or validating configuration: invalid
    TOML, an unknown setting, a wrong type, or an unknown profile name."""


# --------------------------------------------------------------------------
# TOML parsing and validation (internal)
# --------------------------------------------------------------------------
def _flatten(data: dict, path: Path) -> tuple[dict, dict]:
    """Accept both flat keys and named sections. Reject anything unrecognised.

    Returns (settings, profiles). Profiles are named override sets:

        [profiles.qwen]
        model = "qwen3-32b"
        disable_thinking = true
    """
    flat: dict = {}
    profiles: dict = {}
    unknown: list[str] = []
    data = dict(data)
    for name, body in (data.pop(PROFILE_SECTION, None) or {}).items():
        if not isinstance(body, dict):
            raise ConfigError(f"{path}: [profiles.{name}] must be a table")
        bad = [k for k in body if k not in _FIELDS]
        if bad:
            raise ConfigError(f"{path}: [profiles.{name}] unknown setting(s): "
                              f"{', '.join(bad)}")
        profiles[name] = dict(body)
    for key, value in data.items():
        if isinstance(value, dict):
            if key not in _SECTIONS:
                unknown.append(f"[{key}]")
                continue
            for k, v in value.items():
                if k in _FIELDS:
                    flat[k] = v
                else:
                    unknown.append(f"{key}.{k}")
        elif key in _FIELDS:
            flat[key] = value
        else:
            unknown.append(key)
    if unknown:
        lines = [f"{path}: unknown setting(s):"]
        for u in unknown:
            bare = u.strip("[]").split(".")[-1]
            near = difflib.get_close_matches(bare, _FIELDS, n=1, cutoff=0.7)
            lines.append(f"  {u}" + (f"   did you mean '{near[0]}'?" if near else ""))
        lines.append("Run `pedacito config` to see every valid setting.")
        raise ConfigError("\n".join(lines))
    return flat, profiles


def _coerce(name: str, value, path: Path | None = None):
    """Convert a raw TOML value to the type its Config field expects,
    forgiving int-for-float and list-for-tuple, strict about everything else."""
    target = _FIELDS[name].type
    if isinstance(target, str):  # from __future__ annotations
        target = {"str": str, "float": float, "int": int, "bool": bool}.get(
            target, target)
    where = f"{path}: " if path else ""
    if name == "extra_extensions":
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}{name} must be a list, "
                              f'e.g. [".jinja", ".proto"]')
        return tuple(str(v) for v in value)
    if target is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if target in (str, int, float, bool) and not isinstance(value, target):
        raise ConfigError(
            f"{where}{name} should be {target.__name__}, "
            f"got {type(value).__name__} ({value!r})")
    return value


def _read(path: Path) -> tuple[dict, dict]:
    """Parse one TOML config file into (settings, profiles)."""
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e
    return _flatten(data, path)


def find_project_config(root: Path) -> Path | None:
    """Locate a project-level config file under `root`, if one exists."""
    for name in PROJECT_CONFIG_NAMES:
        p = root / name
        if p.exists():
            return p
    p = root / ".pedacito" / "config.toml"
    return p if p.exists() else None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def load(root: Path | None = None, explicit: Path | None = None,
         use_env: bool = True, profile: str | None = None) -> Config:
    """Build a Config by layering every source in precedence order: defaults,
    user config, project config, an explicit --config file, a selected
    profile, then PEDACITO_* environment variables. Each field records which
    layer set it, in `cfg.sources`, for `pedacito config` to display."""
    cfg = Config()
    for f in _FIELDS:
        cfg.sources[f] = "default"
    profiles: dict = {}

    layers: list[tuple[str, Path]] = []
    if USER_CONFIG.exists():
        layers.append(("user config", USER_CONFIG))
    if root is not None:
        proj = find_project_config(Path(root))
        if proj is not None:
            layers.append(("project config", proj))
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise ConfigError(f"--config {p} does not exist")
        layers.append(("--config", p))

    for label, path in layers:
        settings, found = _read(path)
        profiles.update(found)
        for key, value in settings.items():
            setattr(cfg, key, _coerce(key, value, path))
            cfg.sources[key] = f"{label} ({path})"

    cfg.profiles = profiles
    # Which profile: explicit flag, then the shell environment, then a lasting
    # default from the config file.
    origin = "--profile"
    if not profile:
        profile = os.environ.get("PEDACITO_PROFILE") or ""
        origin = "env PEDACITO_PROFILE"
    if not profile:
        profile = cfg.default_profile
        origin = "default_profile"
    if profile:
        if profile not in profiles:
            known = ", ".join(sorted(profiles)) or "(none defined)"
            raise ConfigError(
                f"No profile named '{profile}' (from {origin}). "
                f"Defined profiles: {known}\n"
                "Add one to your config:\n\n"
                f"  [profiles.{profile}]\n  model = \"...\"")
        for key, value in profiles[profile].items():
            setattr(cfg, key, _coerce(key, value))
            cfg.sources[key] = f"profile '{profile}' (via {origin})"
        cfg.active_profile = profile

    # Env vars still work, mostly for scripting and one-off overrides.
    if use_env:
        for key in _FIELDS:
            if key == "default_profile":
                continue   # handled above, before profiles were applied
            raw = os.environ.get("PEDACITO_" + key.upper())
            if raw is None:
                continue
            setattr(cfg, key, _parse_env(key, raw))
            cfg.sources[key] = f"env PEDACITO_{key.upper()}"
    return cfg


def _parse_env(name: str, raw: str):
    """Convert a raw PEDACITO_<NAME> environment variable string to the type
    its Config field expects."""
    target = _FIELDS[name].type
    if isinstance(target, str):
        target = {"str": str, "float": float, "int": int, "bool": bool}.get(target, str)
    if name == "extra_extensions":
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    if target is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    try:
        return target(raw) if target in (int, float) else raw
    except ValueError as e:
        raise ConfigError(f"PEDACITO_{name.upper()}={raw!r} is not a valid {target.__name__}") from e


# --------------------------------------------------------------------------
# `pedacito config --init` starter file
# --------------------------------------------------------------------------
TEMPLATE = '''\
# Pedacito configuration.
# Lives at ~/.config/pedacito/config.toml (Linux/macOS) or
# %APPDATA%\\pedacito\\config.toml (Windows). A .pedacito.toml in a project root
# overrides these for that project, and command-line flags override both.

# Uncomment to make one profile the default for every command.
# Override for a single shell with:  $env:PEDACITO_PROFILE = "qwen"
# default_profile = "devstral"

[server]
# Any server exposing an OpenAI-compatible /v1 API: LM Studio, Ollama, and
# others all work. The default port differs (LM Studio 1234, Ollama 11434).
# Over Tailscale, use the tailnet name or IP instead of localhost.
#
# LM Studio: enable "Serve on Local Network" in the Developer tab, or it only
# listens on 127.0.0.1.
# Ollama: listens on 0.0.0.0 by default; set OLLAMA_HOST to change that.
base_url = "http://localhost:1234/v1"

# Must match the id the server reports exactly. `pedacito config --check` shows
# what's available:
#   LM Studio: the id shown by /v1/models
#   Ollama:    the name shown by `ollama list`, e.g. "qwen2.5-coder:14b"
model = "google/gemma-3-27b"

# api_key = "lm-studio"      # ignored by both servers; any string works
# timeout = 600.0            # seconds; raise it if you index very large files

[review]
# review = true              # always open the side-by-side reviewer
# open_browser = false       # print the URL instead of launching a browser

[agent]
# max_steps = 6              # lookups before the model must answer
# max_gathered_chars = 24000 # ceiling on accumulated source per task

[map]
# flat_map_token_budget = 4000   # above this, the map switches to file cards

[indexing]
# summarise_methods = false      # true if your methods lack docstrings
# summarise_documented = false   # true if your docstrings are stale

[files]
# extra_extensions = [".jinja", ".proto"]
# include_references = true      # false indexes Python only

# Named presets. Select one with --profile NAME. Anything not listed in a
# profile falls back to the settings above.
#
#   pedacito ask myproj -t "..." --profile qwen
#
# [profiles.gemma]
# model = "google/gemma-3-27b"
#
# [profiles.devstral]
# model = "mistralai/devstral-small-2-24b-instruct-2512"
#
# [profiles.qwen]
# model = "qwen/qwen3-32b"
# disable_thinking = true          # skip the <think> block: faster, cheaper
# thinking_suffix = "/no_think"    # belt and braces; some builds need this one
#
# [profiles.qwen-thinking]
# model = "qwen/qwen3-32b"
# max_tokens_answer = 6000         # thinking needs headroom
'''


def write_template(path: Path | None = None, force: bool = False) -> Path:
    """Write the starter config template to `path` (default: USER_CONFIG).
    Refuses to overwrite an existing file unless `force` is set."""
    p = path or USER_CONFIG
    if p.exists() and not force:
        raise ConfigError(f"{p} already exists; pass --force to overwrite")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(TEMPLATE)
    return p
