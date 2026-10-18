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
import sys
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
    # Nominal lookup budget for the gather phase. It is a guardrail against
    # looping, not a hard coverage limit: if the model says `done` while files
    # the task names are still unread, the loop extends step-by-step (up to an
    # internal ceiling of 80) rather than failing the answer phase outright.
    # Models routinely need 15-25 lookups to cover a real task (locate, read
    # each file fully, verify); at a stingy default the loop hit its nominal
    # limit mid-gather and ended with "Raise max_steps so gathering has room
    # to cover them". The guardrail only matters against looping models, and
    # the loop already extends step-by-step while coverage improves.
    max_steps: int = 30
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
    # Recognise rename/move intent in edit blocks: a creation of a new path
    # paired with a block that empties an existing same-extension file is
    # applied as a real move (contents relocated, old file deleted) instead of
    # leaving emptied stubs behind. Set false to keep strict create+clear
    # semantics and never delete a file automatically.
    allow_rename: bool = True
    # When a search/replace block fails to match, the model is fed the failure
    # and asked to re-anchor it. This caps those repair round-trips per apply.
    # Hosted models re-anchor reliably and merit more rounds; small local
    # models mostly repeat the same bad copy, so fewer rounds wastes less time.
    repair_rounds: int = 2

    # --- Context window -------------------------------------------------------
    # The server-side context length you configured for the model (LM Studio's
    # "Context length", Ollama's num_ctx, an API's advertised window). Used by
    # `pedacito config` to sanity-check the token budgets below against it --
    # profiles set this per tier (8k for a small local model, 128k for hosted).
    # The default is deliberately conservative (matches LM Studio's own stock
    # setting), but you rarely need to set it at all: when a local server
    # answers, pedacito probes its REAL configured context length via the
    # native API and uses that instead of this number, and `auto_fit` then
    # shrinks or grows every other budget to fit the window it found.
    context_tokens: int = 8192
    # Tokens held back from `context_tokens` for the reply itself. A gather
    # plan that fills the window to the last token leaves nothing to answer
    # with, which is how truncation errors like "ran past max_tokens_step"
    # usually start. auto_fit lowers this automatically when the window is
    # too small for it.
    reserve_tokens: int = 2048
    # Auto-fit every token budget to the server's real context window before
    # each run (see config.auto_fit): shrink oversized budgets so requests
    # stop hitting the limit mid-task, grow them when the window has room.
    # Set false only if you want to hand-tune every knob yourself.
    auto_fit: bool = True
    # Probe the server for its actual configured context length (LM Studio's
    # native /api/v0/models reports it; Ollama's /api/show does too). On
    # success this overrides context_tokens unless YOU set context_tokens
    # explicitly in your config/env. Set false to trust your file verbatim.
    auto_detect_context: bool = True
    # Ask the server for this many output tokens on generation calls. Some
    # llama.cpp-based servers reject max_tokens values above their configured
    # context window outright; auto_fit clamps this to fit. 0 means "omit
    # max_tokens from requests entirely" (the server then decides).
    request_max_tokens_cap: int = 32768

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
    # It may also be set at the TOP level of the file (`default_profile = ...`),
    # which `--setup` writes; a profile named here must exist in [profiles.*].
    default_profile: str = ""
    # Optional preset tier for the NO-PROFILE run: "small", "medium", "large"
    # or "hosted". Lets a simple two-line config (`tier = "hosted"` plus your
    # server details) get sane token budgets without defining any profiles.
    # Profiles carry their own `tier` key; this one only covers runs where no
    # profile is selected.
    tier: str = ""

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

    # Top-level `tier` (no-profile preset). Validated before profile
    # selection so a typo fails loudly here rather than silently doing
    # nothing later.
    top_tier = cfg.tier
    if top_tier:
        if top_tier not in PRESETS:
            raise ConfigError(
                f"tier must be one of {', '.join(PRESET_TIERS)}, "
                f"got {top_tier!r}"
                + (f" ({cfg.sources.get('tier')})" if cfg.sources.get("tier") else ""))
        for key, value in PRESETS[top_tier].items():
            if cfg.sources.get(key) == "default":
                setattr(cfg, key, _coerce(key, value))
                cfg.sources[key] = f"tier '{top_tier}' preset"
        cfg.active_tier = top_tier

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
            hint = ("Define it in your config, or clear `default_profile`. "
                    "Run `pedacito config --setup` to build both interactively."
                    if origin == "default_profile" and not profiles else "")
            raise ConfigError(
                f"No profile named '{profile}' (from {origin}). "
                f"Defined profiles: {known}\n"
                + (hint + "\n" if hint else "") +
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

    # Loud-but-not-fatal: budgets that cannot fit the declared context window
    # are how truncation errors reach the user, so surface them at load time.
    for warning in budget_warnings(cfg):
        print(f"[config warning] {warning}", file=sys.stderr)
    return cfg


def budget_warnings(cfg: Config) -> list[str]:
    """Check every token/character budget against `context_tokens`. Returns
    human-readable warnings; empty means the configuration fits. Never raises
    -- a conservative guess about someone's server should not block a run."""
    w: list[str] = []
    # Hosted APIs (DeepSeek, OpenAI, ...) have no local window to fit to --
    # their context_tokens is a budget, not hardware. Their server-side caps
    # are enormous compared with anything we send, so the local-style "this
    # cannot fit" warnings only create noise ("kept hitting limits" panic on
    # configs that work fine). Explicit per-profile values still apply; we
    # just stop second-guessing them here. auto_fit and llm.py likewise treat
    # hosted runs as grow-only / unclamped.
    if is_hosted(cfg):
        return w
    usable = cfg.context_tokens - cfg.reserve_tokens
    if usable <= 0:
        w.append(f"reserve_tokens ({cfg.reserve_tokens}) leaves nothing of "
                 f"context_tokens ({cfg.context_tokens}) for prompts; lower "
                 f"reserve_tokens or raise context_tokens")
        return w
    if cfg.max_tokens_answer > cfg.context_tokens:
        w.append(f"max_tokens_answer ({cfg.max_tokens_answer}) exceeds "
                 f"context_tokens ({cfg.context_tokens}); the answer will be "
                 f"cut off by the server no matter what pedacito asks for")
    if cfg.max_tokens_step > usable:
        w.append(f"max_tokens_step ({cfg.max_tokens_step}) exceeds the usable "
                 f"window (context_tokens - reserve_tokens = {usable}); a "
                 f"gather step can never finish before the window runs out")
    # ~4 chars/token is a generous estimate for code-heavy content.
    gathered_tok = cfg.max_gathered_chars // 4
    map_tok = cfg.flat_map_token_budget
    step_headroom = max(cfg.max_tokens_step, cfg.max_tokens_summary)
    if map_tok + gathered_tok + step_headroom > usable:
        w.append(f"flat_map_token_budget ({map_tok}) + gathered source "
                 f"(~{gathered_tok} tok from max_gathered_chars="
                 f"{cfg.max_gathered_chars}) + one step (~{step_headroom} tok) "
                 f"overflows the usable window ({usable} tok). Lower "
                 f"max_gathered_chars or flat_map_token_budget, or raise "
                 f"context_tokens if your server really has more room.")
    return w


def is_hosted(cfg: "Config") -> bool:
    """True when the active config targets a remote API rather than a local
    model server.

    Signals, in order of authority: an explicit `tier = "hosted"` (top level
    or profile); `json_mode = "object"`, which the setup wizard and every
    template set only for hosted APIs; a base_url that points anywhere other
    than localhost. Used by auto_fit to leave hosted budgets alone -- there
    is no hardware window to fit them to -- and by llm.py's truncation-retry
    ceiling so hosted runs are never clamped by a stale local default.
    """
    if cfg.active_tier == "hosted":
        return True
    if cfg.json_mode == "object":
        return True
    try:
        from urllib.parse import urlparse
        host = (urlparse(cfg.base_url).hostname or "").lower()
    except ValueError:
        return False
    return host not in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "")


def auto_fit(cfg: Config, verbose: bool = True) -> list[str]:
    """Fit every token budget to the server's real context window.

    The single biggest source of "nothing worked, it kept hitting limits":
    the built-in budgets assume a roomy window (they were tuned for 32k+),
    while LM Studio ships models at 8192 by default and Ollama caps at 4096
    unless told otherwise. A hosted profile (tier "hosted") also lands on
    local hardware with 128k/16k numbers that the server flatly rejects.
    Rather than fail mid-task, this rewrites the budgets to fit -- shrink
    when the window is smaller than assumed, grow when it has room:

      * reserve_tokens shrinks first so tiny windows keep usable space;
      * max_tokens_answer / build are clamped to what the window can hold;
      * max_gathered_chars and flat_map_token_budget are split between them
        so map + source + one step always fits in (context - reserve);
      * request_max_tokens_cap clamps every max_tokens we ever send, so a
        llama.cpp server never sees a value above its configured window.

    When cfg.auto_detect_context is on and the user did not set
    context_tokens themselves, the live server is probed (LM Studio native
    API / Ollama /api/show) and its REAL configured length is used. Probe
    failures are silent -- hosted APIs and offline runs keep the declared
    number. Returns human-readable notes about what changed (empty = no
    changes needed). Never raises.
    """
    import sys as _sys
    notes: list[str] = []
    if not cfg.auto_fit:
        return notes

    # Hosted APIs have no local window to fit budgets to. Their declared
    # context_tokens is a budget, not a hardware limit; clamping reply
    # budgets against it (or worse, against an 8192 default left over from
    # a local profile) is what made every deepseek step die at "tried up to
    # max_tokens=9600, ceiling 16384" -- the model was allowed to write more
    # than the clamp permitted, and the retry ceiling capped its recovery.
    # For hosted runs we only ever GROW budgets toward the declared window;
    # shrink/clamp is reserved for servers we can actually probe.
    hosted = is_hosted(cfg)

    ctx = cfg.context_tokens
    origin = cfg.sources.get("context_tokens", "default")
    if cfg.auto_detect_context and origin == "default" and not hosted:
        try:
            from .llm import LMStudio
            detected = LMStudio(cfg).detect_context()
        except Exception:
            detected = None
        if detected and detected != ctx:
            notes.append(f"context: server reports {detected:,} tokens "
                         f"(config default was {ctx:,}); using it")
            ctx = detected
            cfg.context_tokens = ctx
            cfg.sources["context_tokens"] = "auto-detected from server"

    def setv(key: str, value):
        cur = getattr(cfg, key)
        if cur == value:
            return
        src = cfg.sources.get(key, "default")
        # Respect explicit user settings within reason: only clamp values
        # that physically cannot work (bigger than the whole window).
        if src != "default" and value < cur:
            hard_max = max(1, ctx - 1)
            if cur <= hard_max:
                return          # user set it and it still fits -- leave it
        # Hosted runs are grow-only: never shrink a budget below what the
        # config/preset chose. There is no local window being enforced, and
        # shrinking reply budgets is exactly how hosted tasks hit their
        # server-side output caps mid-JSON instead of finishing.
        if hosted and value < cur:
            return
        setattr(cfg, key, value)
        cfg.sources[key] = f"auto-fit (window {ctx:,} tok)"
        notes.append(f"{key}: {cur:,} -> {value:,}")

    # Reserve: needs at least ~512 tokens of reply room; ideally >= answer.
    reserve = min(cfg.reserve_tokens, max(512, ctx // 4))
    setv("reserve_tokens", reserve)
    usable = max(ctx - cfg.reserve_tokens, ctx // 2)

    # Per-request output caps must fit inside the window itself.
    cap = max(256, min(cfg.request_max_tokens_cap, usable))
    setv("request_max_tokens_cap", cap)

    # Reply budgets must fit in the space LEFT OVER after the prompt, not
    # just inside the window. llama.cpp/LM Studio answer "prompt +
    # max_tokens > n_ctx" with a flat 400 -- which is exactly what DeepSeek-
    # style "kept hitting limits" feels like on a local server: every gather
    # step asked for 700-1200 reply tokens while the map+task prompt already
    # filled most of an 8k window. So step/summary budgets are clamped to
    # half the usable space (prompts during gathering are large), and the
    # one-shot answer budget may use nearly all of it (its prompt is charged
    # separately by max_gathered_chars).
    step_ceiling = max(128, usable // 2)
    if cfg.max_tokens_step > step_ceiling:
        setv("max_tokens_step", step_ceiling)
    summary_ceiling = max(64, min(cfg.max_tokens_summary, step_ceiling // 3))
    if cfg.max_tokens_summary != summary_ceiling:
        setv("max_tokens_summary", summary_ceiling)

    # Answer/build budgets: never ask for more than fits after the prompt.
    setv("max_tokens_answer", min(cfg.max_tokens_answer, cap, usable - 256))
    setv("max_tokens_build", min(cfg.max_tokens_build, cap, usable - 256))
    setv("max_tokens_plan", min(cfg.max_tokens_plan, cap, usable - 256))

    # Map + gathered source share the rest of the window. 4 chars/token.
    left = max(usable - max(cfg.max_tokens_step, cfg.max_tokens_summary), 512)
    map_tok = min(cfg.flat_map_token_budget, max(400, left // 3))
    setv("flat_map_token_budget", map_tok)
    gathered_tok = max(left - map_tok, 256)
    setv("max_gathered_chars", min(cfg.max_gathered_chars, gathered_tok * 4))

    if not hosted:
        # Final sanity pass: every reply budget must leave real room for the
        # prompt inside the window (llama.cpp 400s when prompt+max_tokens >
        # n_ctx; Ollama silently truncates instead). This catches combinations
        # the steps above could still miss -- e.g. a user who explicitly set a
        # huge max_tokens_answer AND context_tokens barely above it.
        # Skipped entirely for hosted runs: their budgets are deliberately
        # allowed to exceed any stale local default, and the API's own limits
        # (not ours) decide what actually fits.
        hard_cap = max(128, ctx - 512)
        for key in ("max_tokens_step", "max_tokens_summary", "max_tokens_answer",
                    "max_tokens_build", "max_tokens_plan"):
            v = getattr(cfg, key)
            if v > hard_cap:
                setv(key, hard_cap)

    if notes and verbose:
        print("[auto-fit] " + "; ".join(notes), file=_sys.stderr)
    return notes


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
        # max_steps is generous (40): a step is one lookup, cheap in tokens
        # and only bounded by the context guardrails elsewhere; hitting the
        # limit mid-gather is what produced "Raise max_steps so gathering
        # has room to cover them" failures on real tasks.
        "max_tokens_summary": 160,
        "max_tokens_step": 500,
        "max_steps": 40,
        "max_tokens_answer": 1500,
        "max_tokens_plan": 800,
        "max_tokens_build": 2000,
        "flat_map_token_budget": 1500,
        "max_gathered_chars": 12000,
        "max_read_lines": 120,
        "max_spec_chars": 20000,
        "max_new_files": 12,
        "repair_rounds": 1,
        "context_tokens": 8192,
        "reserve_tokens": 2048,
    },
    "medium": {
        # ~14B-32B local models. max_tokens_step is deliberately generous:
        # a THINKING model spends most of the step budget on hidden reasoning
        # before emitting the JSON action, and at 700 that leaves nothing --
        # finish_reason=length, no action, "didn't return an answer".
        "max_tokens_summary": 220,
        "max_tokens_step": 1500,
        "max_steps": 50,
        "max_tokens_answer": 3000,
        "max_tokens_plan": 1200,
        "max_tokens_build": 4000,
        "flat_map_token_budget": 4000,
        "max_gathered_chars": 24000,
        "max_read_lines": 200,
        "max_spec_chars": 60000,
        "max_new_files": 24,
        "repair_rounds": 2,
        "context_tokens": 32768,
        "reserve_tokens": 4096,
    },
    "large": {
        "max_tokens_summary": 300,
        "max_tokens_step": 1000,
        "max_steps": 60,
        "max_tokens_answer": 8000,
        "max_tokens_plan": 2500,
        "max_tokens_build": 8000,
        "flat_map_token_budget": 8000,
        "max_gathered_chars": 60000,
        "max_read_lines": 300,
        "max_spec_chars": 120000,
        "max_new_files": 40,
        "repair_rounds": 2,
        "context_tokens": 65536,
        "reserve_tokens": 8192,
    },
    "hosted": {
        # API pricing tolerates generous windows; answers and build passes
        # get real headroom instead of truncation retries.
        #
        # Hosted tiers are NOT auto-fit: the window here is a declared
        # budget, not a server limit (auto_fit skips local probing for
        # hosted profiles), so the budgets below only ever grow -- they
        # never clamp anything the user set. A model writing four 800-line
        # files in one answer needs ~40k output tokens; deepseek-chat's
        # real ceiling is 8k, which no config can raise, but the budget
        # must at least stop fighting the model within its true limits.
        # max_tokens_step at 8k also gives thinking models room to reason
        # and still emit a full write action in one step.
        "max_tokens_summary": 300,
        "max_tokens_step": 8000,
        "max_steps": 100,
        "max_tokens_answer": 40000,
        "max_tokens_plan": 8000,
        "max_tokens_build": 32000,
        "flat_map_token_budget": 12000,
        "max_gathered_chars": 100000,
        "max_read_lines": 400,
        "max_spec_chars": 200000,
        "max_new_files": 60,
        "repair_rounds": 3,
        "context_tokens": 128000,
        "reserve_tokens": 16384,
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
# Pedacito configuration.  Run `pedacito config --setup` to build this file
# interactively (it finds LM Studio / Ollama on this machine for you), or
# edit the two lines below.  Everything else is optional.
#
# MANDATORY (only if your server/model differ from these defaults):
#   base_url  where the model lives. Default: http://localhost:1234/v1
#   model     the exact name your server reports. Default: google/gemma-3-27b
#
base_url = "http://localhost:1234/v1"
model    = "google/gemma-3-27b"

# Optional: pick sane token budgets without defining any profiles.
# One of "small" (~9B local), "medium" (~14-27B local), "large" (35B+),
# or "hosted" (DeepSeek/OpenAI-style APIs).  Delete it to use built-ins.
tier = "medium"

# Profiles let one file hold several models/tiers; select with --profile NAME
# (or default_profile below).  A hosted API profile looks like this:
# [profiles.deepseek]
# base_url = "https://api.deepseek.com/v1"
# api_key  = "sk-..."
# model    = "deepseek-chat"
# tier     = "hosted"
# json_mode = "object"      # hosted APIs usually lack structured-output schemas

# default_profile = "deepseek"

# Every other knob (max_steps, context_tokens, temperature, thinking control,
# chunking, review...) has a working built-in default.  See them all with:
#   pedacito config --reference        (annotated full template)
#   pedacito config                    (current resolved values + who set them)
# Precedence: defaults < this file < ./.pedacito.toml < --config FILE <
# --profile NAME < CLI flags.
'''


# --------------------------------------------------------------------------
# Full annotated reference. `--init` writes the short TEMPLATE above;
# `pedacito config --reference` prints this one so every knob stays
# documented without overwhelming first-time users.
# --------------------------------------------------------------------------
REFERENCE_TEMPLATE = '''\
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
#
# Context window: what your server actually gives the model. Pedacito checks
# every budget below against these two numbers at load time and prints a
# warning if they cannot fit -- the usual root cause of "ran past the token
# limit" errors. Set context_tokens to the context length configured in LM
# Studio / Ollama / your API; profiles override both per model.
# context_tokens = 16384       # server-side context length for this model
# reserve_tokens = 2048        # tokens held back from the window for replies

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
# max_steps = 6                    # nominal lookup budget per task. Not a
#                                  # hard wall: if the model says `done` while
#                                  # files the task names are still unread,
#                                  # the loop grants extra steps one at a time
#                                  # (internal ceiling 80) until coverage is
#                                  # complete or the gathered-char budget runs
#                                  # out. Raise it for API models on big tasks.
# max_gathered_chars = 24000       # ceiling on accumulated source per task
# max_read_lines = 200             # cap on one `read` call's line count
# allow_write = false              # let the gather loop overwrite whole files
#                                  # (skips backup/review; prefer edit blocks)
# max_write_chars = 200000         # ceiling on one `write` call
# allow_rename = true              # treat "create new path + empty old path"
#                                  # block pairs as real renames (old file is
#                                  # deleted); set false to never auto-delete
# repair_rounds = 2                # re-anchor attempts per failed edit block

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
# max_steps          = 8
# repair_rounds      = 1
# max_tokens_answer  = 1500
# max_tokens_plan    = 800
# max_tokens_build   = 2000
# flat_map_token_budget = 1500
# max_gathered_chars = 12000
# max_read_lines     = 120
# max_spec_chars     = 20000
# max_new_files      = 12
# context_tokens     = 8192
# reserve_tokens     = 2048
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
# max_steps          = 12
# repair_rounds      = 2
# max_tokens_answer  = 3000
# max_tokens_plan    = 1200
# max_tokens_build   = 4000
# flat_map_token_budget = 4000
# max_gathered_chars = 24000
# max_read_lines     = 200
# max_spec_chars     = 60000
# max_new_files      = 24
# context_tokens     = 32768
# reserve_tokens     = 4096

# Big local rig (35B+ with roomy context). The "large" preset.
#
# [profiles.local35]
# # tier = "large"
# base_url = "http://localhost:1234/v1"
# model    = "qwen3-32b"        # or a 70B on a multi-GPU rig
# max_tokens_summary = 300
# max_tokens_step    = 1000
# max_steps          = 16
# repair_rounds      = 2
# max_tokens_answer  = 8000
# max_tokens_plan    = 2500
# max_tokens_build   = 8000
# flat_map_token_budget = 8000
# max_gathered_chars = 60000
# max_read_lines     = 300
# max_spec_chars     = 120000
# max_new_files      = 40
# context_tokens     = 65536
# reserve_tokens     = 8192

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
# max_steps          = 24
# repair_rounds      = 3
# max_tokens_answer  = 16000
# max_tokens_plan    = 3000
# max_tokens_build   = 12000
# flat_map_token_budget = 12000
# max_gathered_chars = 100000
# max_read_lines     = 400
# max_spec_chars     = 200000
# max_new_files      = 60
# context_tokens     = 128000
# reserve_tokens     = 16384
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




# --------------------------------------------------------------------------
# `pedacito config --setup`  interactive first-run configuration
# --------------------------------------------------------------------------
_KNOWN_SERVERS = (
    ("LM Studio", "http://localhost:1234/v1"),
    ("Ollama",    "http://localhost:11434/v1"),
)


def discover_servers(timeout: float = 2.0) -> list[tuple[str, str, list[str]]]:
    """Probe the known local OpenAI-compatible servers on this machine.

    Returns [(label, base_url, [model ids]), ...] for every server that
    answers GET <base_url>/models within `timeout`. A server that is up but
    lists no models still appears, with an empty list. Shared by
    interactive_setup() and the GUI's setup hint."""
    import requests
    found = []
    for label, url in _KNOWN_SERVERS:
        try:
            r = requests.get(url.rstrip("/") + "/models", timeout=timeout)
            r.raise_for_status()
            ids = sorted({m.get("id", "?") for m in r.json().get("data", [])})
        except Exception:
            continue
        found.append((label, url, ids))
    return found


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        raw = ""
    return raw or default


def interactive_setup(path: Path | None = None, force: bool = False,
                      stdin=input, stdout=None) -> Path:
    """Build a working config file step by step, then validate it.

    Asks three questions at most (where is your model, which model, hosted
    API?) and writes a minimal file: base_url, model, and a tier -- the only
    settings that are genuinely mandatory. Existing files are backed up to
    <path>.bak rather than silently overwritten. Raises ConfigError on bad
    answers; the written file is load()-checked before returning."""
    import sys
    out = stdout or sys.stdout
    def say(s=""):
        print(s, file=out)

    p = path or USER_CONFIG
    if p.exists() and not force:
        ans = _ask(f"{p} already exists. Overwrite it? [y/N]")
        if ans.lower() not in ("y", "yes"):
            raise ConfigError("setup cancelled; existing config left untouched")

    say("Pedacito setup -- only two settings are mandatory: where your model")
    say("lives (base_url) and its exact name (model). Everything else has a")
    say("working default. Press Enter to accept any suggestion in [brackets].")
    say()

    servers = discover_servers()
    base_url = model = ""
    hosted = False
    api_key = ""
    if servers:
        say("Found running model servers on this machine:")
        for i, (label, url, ids) in enumerate(servers, 1):
            say(f"  {i}) {label}  {url}  ({len(ids)} model(s) listed)")
        say("  0) none of these -- use a hosted API (DeepSeek, OpenAI, ...)")
        choice = _ask("Choose a server", str(len(servers)) if len(servers) == 1 else "1")
        if choice == "0":
            hosted = True
        else:
            try:
                label, base_url, ids = servers[int(choice) - 1]
            except (ValueError, IndexError):
                raise ConfigError(f"not a valid server choice: {choice!r}")
            if ids:
                say("Models that server reports:")
                for i, m in enumerate(ids, 1):
                    say(f"  {i}) {m}")
                pick = _ask("Which model should Pedacito use?", ids[0])
                try:
                    model = ids[int(pick) - 1]
                except (ValueError, IndexError):
                    model = pick          # allow typing an id not currently listed
            else:
                model = _ask("Model name (exactly as the server reports it)")
    else:
        say("No LM Studio or Ollama server answered on localhost.")
        hosted = _ask("Is this a hosted API (DeepSeek/OpenAI-style)? [Y/n]"
                      ).lower() != "n"

    if hosted:
        base_url = _ask("API base URL", "https://api.deepseek.com/v1")
        api_key = _ask("API key (sent over HTTPS; stored in this file)")
        model = _ask("Model name", "deepseek-chat")

    if not base_url or not model:
        raise ConfigError("base_url and model are mandatory -- setup needs both")

    tier = "hosted" if hosted else _ask(
        "Model size tier -- small (~9B), medium (~14-27B), large (35B+)",
        "medium")
    if tier not in PRESETS:
        raise ConfigError(f"tier must be one of {', '.join(PRESET_TIERS)}, got {tier!r}")

    lines = [
        "# Written by `pedacito config --setup`.",
        "# The only mandatory settings are base_url and model. See",
        "# `pedacito config --reference` for every other knob.",
        f'base_url = "{base_url}"',
        f'model    = "{model}"',
        f'tier     = "{tier}"',
    ]
    if api_key:
        lines.insert(4, f'api_key  = "{api_key}"')
        lines.insert(5, 'json_mode = "object"   # hosted APIs lack llama.cpp schemas')
    body = "\n".join(lines) + "\n"

    # Validate BEFORE touching the user's file: parse exactly as load() will.
    probe = p.with_suffix(".toml.setup-probe") if p.parent.exists() else \
        Path("pedacito-setup-probe.toml")
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(body)
        load(explicit=probe)
    finally:
        try:
            probe.unlink()
        except OSError:
            pass

    if p.exists():
        backup = p.with_name(p.name + ".bak")
        backup.write_bytes(p.read_bytes())
        say(f"(previous config saved to {backup})")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    say()
    say(f"Wrote {p}")
    say(f"  base_url = {base_url}")
    say(f"  model    = {model}")
    say(f"  tier     = {tier}" + ("  (api_key set, json_mode=object)" if api_key else ""))
    say("Test it any time with:  pedacito config --check")
    return p


def write_template(path: Path | None = None, force: bool = False) -> Path:
    """Write the starter config template to `path` (default: USER_CONFIG).
    Refuses to overwrite an existing file unless `force` is set."""
    p = path or USER_CONFIG
    if p.exists() and not force:
        raise ConfigError(f"{p} already exists; pass --force to overwrite")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(TEMPLATE)
    return p
