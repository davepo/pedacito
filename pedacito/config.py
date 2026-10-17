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
import re
import tomllib
import difflib
from dataclasses import dataclass, fields, field
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
    # How the tool loop's JSON response is requested.
    #   "schema" -- send the full json_schema. Grammar-constrained on
    #     llama.cpp / LM Studio, so the model literally cannot emit
    #     unparseable JSON. This is the local-model default.
    #   "object" -- send {"type": "json_object"} instead. Required for
    #     DeepSeek and most other hosted OpenAI-compatible APIs, which
    #     reject json_schema outright. Slightly less reliable in principle,
    #     in practice fine on a frontier hosted model.
    json_mode: str = "schema"

    # --- Generation -----------------------------------------------------------
    # Low temp for tool selection; a 27B at 0.2 is fine for prose and code too.
    temperature: float = 0.2
    # Passthrough for sampler parameters that OpenAI's request schema does
    # not name: top_k, min_p, repeat_penalty, presence_penalty, seed, stop,
    # and anything else a given server accepts. Merged verbatim into every
    # /chat/completions request body, so what you can put here is whatever
    # your server actually reads. Pedacito's own structural keys -- model,
    # messages, stream -- are protected and cannot be overridden.
    #
    # The right place for this is a profile-scoped table, because the fields
    # that local runtimes accept (top_k, min_p, repeat_penalty) are exactly
    # the ones hosted APIs reject with a 400:
    #
    #   [profiles.lmstudio.sampler]
    #   top_p = 0.95
    #   top_k = 40
    #   min_p = 0.05
    #   repeat_penalty = 1.1
    sampler: dict = field(default_factory=dict)
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
    # Let the gather loop create/overwrite whole files via the `write` tool.
    # Off by default: a mid-gather write skips the edit path's backup, syntax
    # check and review page, so prefer search/replace blocks unless a model
    # genuinely cannot emit them.
    allow_write: bool = False
    # Ceiling on one `write` call's contents, in characters.
    max_write_chars: int = 200_000

    # --- Project scaffolding (`pedacito new`) ---------------------------------
    # Cap on how many files a plan may propose; longer plans are truncated and
    # the excess reported. Small models love to enumerate 40 tiny stubs.
    max_new_files: int = 24
    # Spec documents longer than this are truncated, with a loud warning.
    max_spec_chars: int = 60_000
    # Token budgets for the two scaffold calls that do not reuse an existing
    # answer budget.
    max_tokens_plan: int = 1200
    max_tokens_build: int = 4000

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
    active_tier: str = ""   # preset tier applied to the active profile, if any

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
             "agent", "scaffold", "editing", "reasoning", "review", "workspace")
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
        # `tier` is a preset selector consumed by _apply_preset, not a setting.
        bad = [k for k in body if k not in _FIELDS and k != "tier"]
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
    if name == "json_mode":
        if value not in ("schema", "object"):
            raise ConfigError(
                f"{where}{name} must be 'schema' or 'object', got {value!r}")
        return value
    if name == "extra_extensions":
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}{name} must be a list, "
                              f'e.g. [".jinja", ".proto"]')
        return tuple(str(v) for v in value)
    if name == "sampler":
        # A plain dict passthrough, however TOML spelled it: an inline table
        # (`sampler = { top_k = 40 }`) and a nested table
        # (`[profiles.x.sampler]` with keys beneath) both land here as dict.
        if not isinstance(value, dict):
            raise ConfigError(
                f"{where}{name} must be a table, e.g. "
                f"[profiles.NAME.sampler] with keys underneath")
        return dict(value)
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
        body, tier = _apply_preset(profile, profiles[profile], origin)
        for key, value in body.items():
            setattr(cfg, key, _coerce(key, value))
            cfg.sources[key] = f"profile '{profile}' (via {origin})"
        cfg.active_profile = profile
        cfg.active_tier = tier

    # Env vars still work, mostly for scripting and one-off overrides.
    # default_profile is deliberately excluded: profile selection has
    # already happened above, so setting it now would have no effect on
    # which profile is active and would only add a misleading source line.
    if use_env:
        for key in _FIELDS:
            if key == "default_profile":
                continue
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
    if name == "json_mode":
        if raw not in ("schema", "object"):
            raise ConfigError(
                f"PEDACITO_JSON_MODE={raw!r} must be 'schema' or 'object'")
        return raw
    if name == "extra_extensions":
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    if name == "sampler":
        # Not a natural environment-variable target; a JSON object is the
        # only sane spelling. Importing json here keeps the module's import
        # surface unchanged for the (much more common) case where nobody
        # sets this variable.
        import json as _json
        try:
            obj = _json.loads(raw)
        except ValueError as e:
            raise ConfigError(
                f"PEDACITO_SAMPLER={raw!r} is not valid JSON: {e}") from e
        if not isinstance(obj, dict):
            raise ConfigError(
                f"PEDACITO_SAMPLER={raw!r} must be a JSON object, "
                f'e.g. \'{{"top_k": 40}}\'')
        return obj
    if target is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    try:
        return target(raw) if target in (int, float) else raw
    except ValueError as e:
        raise ConfigError(f"PEDACITO_{name.upper()}={raw!r} is not a valid {target.__name__}") from e


# --------------------------------------------------------------------------
# Profile presets -- token limits per class of model/server
# --------------------------------------------------------------------------
# A 9B running locally on 8k context and deepseek-chat over an API are the
# same code path with wildly different budgets. Rather than make every user
# hand-tune six or seven token knobs, profiles that name a recognised tier
# start from the preset below; keys written in [profiles.NAME] override it.
# Presets never touch the top-level defaults: no-profile runs keep the
# conservative built-in values.
#
#   small   ~9B local models (qwen3-8b, gemma-3-12b, llama-3.1-8b)
#   medium  ~14B/32B local models (qwen3-14b, qwen3-32b, gemma-3-27b)
#   large   big local models (27B+ at full context, 70B rigs)
#   hosted  capable APIs: DeepSeek, OpenAI, Anthropic, xAI, ...
PRESET_TIERS = ("small", "medium", "large", "hosted")

PRESETS: dict[str, dict] = {
    "small": {
        # Tight enough that nothing overflows an 8k window at 25 tok/s.
        "max_tokens_summary": 160,
        "max_tokens_step": 500,
        "max_tokens_answer": 1500,
        "max_tokens_plan": 800,
        "max_tokens_build": 2000,
        "flat_map_token_budget": 1500,
        "max_gathered_chars": 12000,
        "max_read_lines": 120,
        "max_spec_chars": 20000,
        "max_new_files": 12,
    },
    "medium": {
        # The built-in defaults, spelled out so a profile can inherit them
        # explicitly and users see the ladder.
        "max_tokens_summary": 220,
        "max_tokens_step": 700,
        "max_tokens_answer": 3000,
        "max_tokens_plan": 1200,
        "max_tokens_build": 4000,
        "flat_map_token_budget": 4000,
        "max_gathered_chars": 24000,
        "max_read_lines": 200,
        "max_spec_chars": 60000,
        "max_new_files": 24,
    },
    "large": {
        "max_tokens_summary": 300,
        "max_tokens_step": 1000,
        "max_tokens_answer": 8000,
        "max_tokens_plan": 2500,
        "max_tokens_build": 8000,
        "flat_map_token_budget": 8000,
        "max_gathered_chars": 60000,
        "max_read_lines": 300,
        "max_spec_chars": 120000,
        "max_new_files": 40,
    },
    "hosted": {
        # API pricing tolerates generous windows; answers and build passes
        # get real headroom instead of truncation retries.
        "max_tokens_summary": 300,
        "max_tokens_step": 1200,
        "max_tokens_answer": 16000,
        "max_tokens_plan": 3000,
        "max_tokens_build": 12000,
        "flat_map_token_budget": 12000,
        "max_gathered_chars": 100000,
        "max_read_lines": 400,
        "max_spec_chars": 200000,
        "max_new_files": 60,
    },
}


def _guess_tier(name: str, body: dict) -> str | None:
    """Infer which PRESETS tier a profile is for, from its model/base_url."""
    hay = " ".join([name.lower(),
                    str(body.get("model", "")).lower(),
                    str(body.get("base_url", "")).lower()])
    if any(h in hay for h in ("deepseek", "openai", "anthropic", "api.",
                              "moonshot", "x.ai", "groq", "together",
                              "mistral.ai", "hosted")):
        return "hosted"
    m = re.search(r"(?<![\d.])(\d{1,3})\s*b(?![\d.])", hay)
    if m:
        size = int(m.group(1))
        if size <= 10:
            return "small"
        if size <= 20:
            return "medium"
        return "large"
    return None


def _apply_preset(name: str, body: dict, origin: str) -> tuple[dict, str]:
    """Return (settings, tier) for a profile, with its tier preset underneath.

    Tier source, first match wins:
      1. explicit `tier = "small|medium|large|hosted"` in the profile
         (the key is consumed here and never reaches Config);
      2. a heuristic on the profile/model/server name;
      3. none ("") -- the profile only carries what you wrote.
    Keys written in the profile always beat the preset.
    """
    body = dict(body)
    tier = body.pop("tier", None)
    if tier is not None:
        if tier not in PRESETS:
            raise ConfigError(
                f"profile '{name}' ({origin}): tier must be one of "
                f"{', '.join(PRESET_TIERS)}, got {tier!r}")
    else:
        tier = _guess_tier(name, body) or ""
    if not tier:
        return body, ""
    merged = dict(PRESETS[tier])
    merged.update(body)
    return merged, tier


# --------------------------------------------------------------------------
# `pedacito config --init` starter file
# --------------------------------------------------------------------------
TEMPLATE = '''\
# Pedacito configuration.
#
# Lives at ~/.config/pedacito/config.toml (Linux/macOS) or
# %APPDATA%\\pedacito\\config.toml (Windows). A .pedacito.toml in a project
# root overrides these for that project; command-line flags override both.
#
# Every setting below is optional -- the value shown in each commented line
# is the built-in default. Pedacito runs with no config file at all, on
# http://localhost:1234/v1 with google/gemma-3-27b.
#
# Precedence, lowest to highest:
#   defaults -> this file -> <project>/.pedacito.toml -> --config FILE ->
#   --profile NAME (tier preset underneath, your keys on top) -> CLI flags
#
# Run `pedacito config` to see every resolved value and which layer set it.
#
# TOKEN LIMITS ARE PROFILE-SCOPED ON PURPOSE. A 9B running locally on an 8k
# context window and deepseek-chat over an API are the same code path with
# wildly different budgets, so you do not hand-tune seven token knobs per
# model -- each profile below carries its own complete set, generated from a
# tier preset:
#
#   tier = "small"    ~9B local models (qwen3-8b, llama-3.1-8b)
#   tier = "medium"   ~14B/27B local models (gemma-3-27b, qwen3-14b)
#   tier = "large"    big local rigs (35B+, 70B)
#   tier = "hosted"   capable APIs (DeepSeek, OpenAI, Anthropic, ...)
#
# The tier line in each profile below is commented out because the limits
# spelled out under it ARE that tier's preset values, written out so you can
# edit any of them individually. Keys you write always win over the preset;
# uncommenting `tier =` just re-syncs the ones you left alone. Without a
# tier key, pedacito also infers one from the model name ("...9b" -> small,
# "...27b" -> large, deepseek/openai URLs -> hosted); `tier = ""` disables
# inference entirely. Running with no profile keeps the conservative built-in
# defaults shown in the reference sections below.

# ===========================================================================
# QUICK START -- one local server, one hosted server, switch with --profile
# ===========================================================================
#
# [server] holds the defaults for when no profile is selected. Every field
# in it, including base_url / api_key / model / json_mode, can be overridden
# by a profile -- so "switch servers" is just "switch profile":
#
#   pedacito ask proj -t "..." --profile deepseek
#   pedacito chat proj            # then /profile deepseek mid-session
#
# A fully worked two-server example is at the bottom of this file. The
# [server] section here is what that example falls back to.

[server]
base_url = "http://localhost:1234/v1"
model    = "google/gemma-3-27b"

# api_key = "lm-studio"        # ignored by local servers; any string works
# json_mode = "schema"         # "schema" for local llama.cpp/LM Studio;
#                              # "object" for hosted OpenAI-compatible APIs
#                              # (DeepSeek, OpenAI, ...), which reject schema
# timeout = 600.0              # seconds per request
# load_wait_seconds = 240      # how long to wait for a model to load
# max_consecutive_failures = 3 # abort an indexing batch after N fails in a row

# ===========================================================================
# EVERY SETTING, with its default
# ===========================================================================
#
# Nothing below the QUICK START block is required. It's a reference: the
# section names are organizational only (any valid field is accepted under
# any section), and every line is commented out. Uncomment what you want to
# change.

# --- Generation -------------------------------------------------------------
# [generation]
# temperature = 0.2            # raise toward 0.7 for prose; leave low for
#                              # tool calls -- higher makes them worse
# max_tokens_summary = 220     # budget for one summary sentence
# max_tokens_step = 700        # budget for one tool call in the gather loop
# max_tokens_answer = 3000     # budget for the final answer / edit reply

# Sampler parameters, merged verbatim into every /chat/completions request.
# This is where top_k, min_p, repeat_penalty, seed, stop, and anything else
# your local runtime accepts go -- the config has no first-class field for
# them because the accepted set differs per server. Hosted APIs (DeepSeek,
# OpenAI) 400 on unknown fields like top_k and min_p, so put this inside a
# profile targeted at a specific server, not here.
#
# [profiles.lmstudio.sampler]
# top_p          = 0.95
# top_k          = 40
# min_p          = 0.05
# repeat_penalty = 1.1
# seed           = -1

# --- Map tiering ------------------------------------------------------------
# [map]
# flat_map_token_budget = 4000     # above this, switch from flat map to cards
# file_card_line_ranges = true     # attach L11-21 to names on file cards;
#                                  # costs a bit more, saves an outline round trip

# --- Indexing ---------------------------------------------------------------
# [indexing]
# class_split_lines = 60           # classes shorter than this stay one chunk
# max_chunk_chars = 6000           # source truncated to this before summarising
# min_lines_to_summarise = 4       # skip summarising trivially short units
# summarise_methods = false        # true if your methods lack docstrings
# summarise_documented = false     # true if your existing docstrings are stale

# --- File selection ---------------------------------------------------------
# [files]
# include_references = true        # index .md/.json/.toml/... as reference files
# extra_extensions = [".jinja", ".proto"]   # treat these as reference too
# max_reference_bytes = 2000000    # above this, a reference file is header-only
# summarise_references = false     # LLM summary on reference files; rarely worth it

# --- Agent loop -------------------------------------------------------------
# [agent]
# max_steps = 6                    # lookups before the model must answer
# max_gathered_chars = 24000       # ceiling on accumulated source per task
# max_read_lines = 200             # cap on one `read` call's line count
# allow_write = false              # let the gather loop overwrite whole files
#                                  # (skips backup/review; prefer edit blocks)
# max_write_chars = 200000         # ceiling on one `write` call

# --- Project scaffolding (`pedacito new`) -----------------------------------
# [scaffold]
# max_new_files = 24               # cap on how many files a plan may propose
# max_spec_chars = 60000           # spec documents longer than this are cut
# max_tokens_plan = 1200           # budget for the planning call
# max_tokens_build = 4000          # budget for each per-file build pass

# --- Reasoning models -------------------------------------------------------
# Qwen3 and friends emit a <think> block by default. Pedacito strips it
# either way, but generating it still costs tokens and time. Both switches
# are provided because which one a given model honours varies; setting both
# is harmless.
#
# [reasoning]
# disable_thinking = false         # sends chat_template_kwargs {"enable_thinking": false}
# thinking_suffix = ""             # appends a token like "/no_think" to the prompt

# --- Editing ----------------------------------------------------------------
# [editing]
# backup = true                    # snapshot originals before writing;
#                                  # `pedacito restore` puts them back
# log_sessions = true              # write a session record per task

# --- Review page ------------------------------------------------------------
# [review]
# review = false                   # always open the side-by-side reviewer
# review_port = 0                  # 0 lets the OS pick a free port
# open_browser = true              # false prints the URL instead of launching
# review_timeout = 1800            # seconds before an unanswered review expires

# --- Workspace --------------------------------------------------------------
# [workspace]
# manage_gitignore = true          # add .pedacito/ to .gitignore on first run
# create_ignore_file = true        # create a starter .pedacitoignore

# ===========================================================================
# PROFILES -- one per server, or one per model, each with its own token limits
# ===========================================================================
#
# A profile overrides any setting in this file, not just model. Select with
# --profile NAME, /profile inside chat, PEDACITO_PROFILE in the shell, or
# default_profile below. The token budgets below are generated from the four
# tier presets (see the top of this file); edit any line to deviate from
# your tier, or uncomment `tier =` to inherit the preset instead of spelling
# it out.
#
# default_profile = "local27"

# Small local model (~9B): everything stays inside an 8k context window at
# ~25 tok/s. json_mode "schema" (the default) is grammar-enforced by
# llama.cpp / LM Studio, so a tool call always parses.
#
# [profiles.small9b]
# # tier = "small"
# base_url = "http://localhost:1234/v1"
# model    = "qwen3-8b"
# api_key  = "lm-studio"
# max_tokens_summary = 160
# max_tokens_step    = 500
# max_tokens_answer  = 1500
# max_tokens_plan    = 800
# max_tokens_build   = 2000
# flat_map_token_budget = 1500
# max_gathered_chars = 12000
# max_read_lines     = 120
# max_spec_chars     = 20000
# max_new_files      = 12
# disable_thinking   = true
# thinking_suffix    = "/no_think"
#
# [profiles.small9b.sampler]
# top_p          = 0.95
# top_k          = 40
# repeat_penalty = 1.1

# Mid local model (~14B/27B). The spelled-out values are the "medium" preset.
#
# [profiles.local27]
# # tier = "medium"
# base_url = "http://localhost:1234/v1"
# model    = "google/gemma-3-27b"
# max_tokens_summary = 220
# max_tokens_step    = 700
# max_tokens_answer  = 3000
# max_tokens_plan    = 1200
# max_tokens_build   = 4000
# flat_map_token_budget = 4000
# max_gathered_chars = 24000
# max_read_lines     = 200
# max_spec_chars     = 60000
# max_new_files      = 24

# Big local rig (35B+ with roomy context). The "large" preset.
#
# [profiles.local35]
# # tier = "large"
# base_url = "http://localhost:1234/v1"
# model    = "qwen3-32b"        # or a 70B on a multi-GPU rig
# max_tokens_summary = 300
# max_tokens_step    = 1000
# max_tokens_answer  = 8000
# max_tokens_plan    = 2500
# max_tokens_build   = 8000
# flat_map_token_budget = 8000
# max_gathered_chars = 60000
# max_read_lines     = 300
# max_spec_chars     = 120000
# max_new_files      = 40

# Hosted DeepSeek. json_mode must be "object": hosted APIs reject the
# json_schema Pedacito sends by default, and the gather loop fails on its
# first step without this. An API affords far higher limits than a small
# local model -- these are the "hosted" preset values.
#
# [profiles.deepseek]
# # tier = "hosted"
# base_url  = "https://api.deepseek.com/v1"
# model     = "deepseek-chat"
# api_key   = "sk-..."
# json_mode = "object"
# timeout   = 120.0
# max_tokens_summary = 300
# max_tokens_step    = 1200
# max_tokens_answer  = 16000
# max_tokens_plan    = 3000
# max_tokens_build   = 12000
# flat_map_token_budget = 12000
# max_gathered_chars = 100000
# max_read_lines     = 400
# max_spec_chars     = 200000
# max_new_files      = 60
#
# [profiles.deepseek.sampler]
# top_p            = 0.95
# presence_penalty = 0.0
# Note: no top_k or min_p here. DeepSeek 400s on those.

# Reasoning model. Thinking off is faster and cheaper at ~25 tok/s, and the
# two ways to ask cover models that honour only one of them. The "large"
# tier preset underneath gives thinking output room; max_tokens_answer is
# then raised further because reasoning burns budget on the <think> block.
#
# [profiles.qwen-thinking]
# tier = "large"
# model = "qwen/qwen3-32b"
# disable_thinking = true
# thinking_suffix = "/no_think"
# max_tokens_answer = 10000    # thinking needs headroom beyond the preset
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
