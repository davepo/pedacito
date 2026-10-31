"""Configuration: the Config dataclass, TOML loading, and profile resolution.

    from .config import load

    cfg = load(root=project_root, explicit=None, profile="qwen")

Settings come from TOML files. `[universal]`, `[indexing]` and `[agent]` hold
shared defaults; `[profiles.NAME]` holds server/model-specific keys, and
`[profiles.default]` is the fallback when no profile is named. Precedence,
lowest to highest:

    1. built-in defaults
    2. the user config file
         Linux/macOS: ~/.config/pedacito/config.toml
         Windows:     %APPDATA%\\pedacito\\config.toml
    3. <project>/.pedacito.toml           per-project overrides
    4. --config FILE                      an explicit file
    5. [profiles.NAME] from any of those, selected with --profile NAME
    6. PEDACITO_<KEY> environment variables
    7. command-line flags

Unknown keys are reported rather than ignored.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
import tomllib
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from urllib.parse import urlparse


def _user_config_dir() -> Path:
    """%APPDATA%\\pedacito on Windows, $XDG_CONFIG_HOME/pedacito (or
    ~/.config/pedacito) elsewhere."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / "pedacito"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "pedacito"


USER_CONFIG = _user_config_dir() / "config.toml"
PROJECT_CONFIG_NAMES = (".pedacito.toml", "pedacito.toml")


@dataclass
class Config:
    """All Pedacito settings with their defaults; see `load()` for assembly."""

    # --- Hardware (mandatory for local model servers) ------------------------
    # Total GPU VRAM in whole gigabytes. auto_fit combines it with the model's
    # file size (queried from the server) to estimate how much context fits,
    # keeping 0.5 GB free. 0 = unset; load() rejects that for local servers.
    vram_gb: int = 0
    # KV-cache quantization the server was loaded with, so the VRAM estimate
    # does not assume a 16-bit cache. Accepted: f16/fp16/bf16, f32/fp32, q8_0,
    # q6_k, q5_1, q5_0, q4_1, q4_0, iq4_nl, iq4_xs (case, dots and hyphens are
    # normalised). Empty = 16-bit.
    k_cache_quant: str = ""
    v_cache_quant: str = ""
    # Measured KV-cache bytes per token; 0 = estimate from the model size.
    # Bypasses both the size proxy and the quantization adjustment.
    kv_cache_bytes_per_token: int = 0

    # --- Server connection ----------------------------------------------------
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = "google/gemma-3-27b"
    timeout: float = 600.0
    # Seconds to wait while the server loads a model (unload + read from disk).
    load_wait_seconds: int = 240
    # Stop a batch after this many failures in a row.
    max_consecutive_failures: int = 3
    # How the tool loop's JSON response is requested:
    #   "schema" -- full json_schema; grammar-constrained on llama.cpp-based
    #     servers. The local default.
    #   "object" -- {"type": "json_object"}, for hosted APIs that reject
    #     json_schema (DeepSeek and most others).
    # Also accepted: "json_object", "json" (-> "object"), "json_schema" (-> "schema").
    json_mode: str = "schema"
    # Set by load() when the server is recognised as a hosted API (by URL or a
    # "deepseek" model name). Not a config key. Forces json_mode to "object"
    # unless the user set json_mode explicitly.
    _hosted_server: bool = False

    # --- Generation -----------------------------------------------------------
    temperature: float = 0.2
    # Extra sampler parameters (top_k, min_p, repeat_penalty, seed, stop, ...)
    # merged verbatim into every /chat/completions body; model, messages and
    # stream cannot be overridden. Best placed in a profile, because hosted
    # APIs reject the parameters local runtimes accept:
    #
    #   [profiles.lmstudio.sampler]
    #   top_p = 0.95
    #   top_k = 40
    sampler: dict = field(default_factory=dict)
    max_tokens_summary: int = 220
    # One tool call plus a short note; the schema puts the action first so a
    # truncated reply loses only the explanation. Hosted presets use ~2000+:
    # servers that ignore response_format let the model free-form around the
    # JSON. Keep in sync with PRESETS below (tests assert the hosted value).
    max_tokens_step: int = 700
    max_tokens_answer: int = 3000

    # --- Map tiering ----------------------------------------------------------
    # Use the full flat map when it fits this budget; otherwise fall back to
    # per-file cards and let the model descend with `outline`.
    flat_map_token_budget: int = 4000
    # Attach L<start>-<end> to names on file cards (saves an `outline` call).
    file_card_line_ranges: bool = True

    # --- Indexing -------------------------------------------------------------
    # Classes shorter than this are indexed as one chunk instead of per-method.
    class_split_lines: int = 60
    # A function/method (or module preamble/trailing block) larger than either
    # limit is split at top-level statement boundaries into name#2, name#3, ...
    max_unit_lines: int = 220
    max_unit_chars: int = 12000
    # Token guard (~3.6 chars/token) for small contexts; 0 disables.
    max_unit_tokens: int = 3000
    # Lines repeated between line windows when one giant statement must be
    # windowed. 0 = disjoint; statement-boundary splits never overlap.
    unit_overlap_lines: int = 0
    # Chunks larger than this get their source truncated before summarising.
    max_chunk_chars: int = 6000
    # Skip summarising trivially short chunks; the signature says enough.
    min_lines_to_summarise: int = 4
    # Summarisation rationing: a docstring is already a summary, and a method
    # is covered by its class summary plus its signature.
    summarise_methods: bool = False
    summarise_documented: bool = False

    # --- File selection -------------------------------------------------------
    # Index non-Python files (.md .csv .json .toml .yaml ...) structurally:
    # the map shows a CSV's columns or a config's sections; contents are never
    # pulled into the map.
    include_references: bool = True
    # Extra extensions to treat as reference files, e.g. [".jinja", ".proto"].
    extra_extensions: tuple[str, ...] = ()
    # Above this size a reference file is recorded by name and header only.
    max_reference_bytes: int = 2_000_000
    summarise_references: bool = False

    # --- Agent loop -----------------------------------------------------------
    # Include the repository's PEDACITO.md / AGENTS.md / CLAUDE.md in every prompt.
    use_project_instructions: bool = True
    max_instruction_chars: int = 2000
    # Cap on the model's own checklist when injected into a prompt.
    max_todo_chars: int = 1500
    # Hard lookup limit for the gather phase; 0 = none (token budget and the
    # model's `done` are the only bounds). `--steps N` sets it for one command.
    max_steps: int = 0
    # Limit on follow-up answer turns when the model defers remaining work;
    # 0 = no limit (stops on NOTHING MORE or when no blocks are produced).
    max_completion_rounds: int = 0
    # Limit on continuation turns when a reply is cut off mid-block; 0 = no
    # limit (the continuation char budget is the real bound).
    max_continuation_rounds: int = 0
    # Hard cap on source the gather phase may accumulate.
    max_gathered_chars: int = 24000
    max_read_lines: int = 200
    # Attach the enclosing signature and module imports when a `read` starts
    # mid-file.
    read_context_hint: bool = True
    # After each `auto` task (when no --verify command is set), run the
    # project's detected linter/type checker and feed failures into the repair
    # rounds. Off by default: pre-existing lint issues would be blamed on the
    # task. The `diagnose` tool is available whenever allow_shell is on.
    auto_diagnostics: bool = False
    diagnostics_timeout: int = 90
    max_diagnostics_chars: int = 8000
    # After every applied edit, run the cheap per-file check (one linter run
    # per changed file, else a syntax parse) and feed back only findings that
    # were NOT in the file before the task (baseline diff), in at most
    # repair_rounds fix rounds. On by default: unlike auto_diagnostics the
    # baseline keeps pre-existing issues out of the feedback.
    post_edit_check: bool = True
    # Restrict what one task may WRITE: existing files can only be modified,
    # moved or deleted when the task title names them or the lookup phase read
    # them; brand-new files are always allowed. Off by default.
    auto_allowlist: bool = False
    # Optional MCP client: exposes one `mcp` action plus a short tool catalog.
    allow_mcp: bool = False
    mcp_servers: tuple[str, ...] = ()
    mcp_timeout: int = 30
    # Let the gather loop create/overwrite files via `write`, grow them with
    # `append`, and rename/remove them with `move`/`delete`. Off by default: a
    # mid-gather write skips the edit path's syntax check and review page.
    allow_write: bool = False
    # Ceiling on one `write` call's contents, in characters.
    max_write_chars: int = 200_000
    # Maximum lines one reply may emit for one file (gather `write` or
    # answer-phase edit block). 120 keeps a step under deepseek-chat's ~8k
    # output-token ceiling; raise it only if the provider allows longer
    # messages, otherwise clean steps become mid-string truncations.
    max_write_lines: int = 120
    # Let the gather loop run ONE command per step and feed its output back.
    # Off by default: enabling it is a trust decision. Commands run WITHOUT a
    # shell (argv only), so pipes, redirection and `&&` do not work.
    allow_shell: bool = False
    # Seconds before a `run` command is killed.
    shell_timeout: int = 120
    # Kill a `run` command after this many seconds with NO output (keeps GUI
    # apps and servers from blocking the agent); 0 disables the idle rule.
    shell_idle_seconds: int = 10
    # Optional executable allowlist for `run`, e.g. ["pytest", "python"];
    # empty = any executable once allow_shell is on.
    shell_allowlist: tuple[str, ...] = ()
    # Cap on command output fed back to the model, in characters.
    max_shell_output_chars: int = 8000
    # Let the gather loop capture the screen as text (pixel statistics, OCR,
    # optional vision-model description). Off by default: it captures the whole
    # desktop; launching an app via `command` also needs allow_shell.
    allow_screenshots: bool = False
    # Seconds to wait after launching before capturing; the action's `start`
    # value overrides it.
    screenshot_wait_seconds: int = 5
    # Let the gather loop open a URL in a headless browser, interact, capture
    # the page and read its text. Can reach any http(s) URL: a trust decision
    # like allow_shell. Needs the optional Playwright install:
    #   pip install "pedacito-local[browser]"
    #   python -m playwright install chromium
    allow_browser: bool = False
    # Seconds to wait after navigation/interaction before capturing.
    browser_wait_seconds: int = 3
    # Seconds to wait for a launched server to print its localhost URL.
    browser_server_wait_seconds: int = 15
    browser_headless: bool = True
    # Per-navigation/action timeout in seconds.
    browser_timeout_seconds: int = 20
    # Multimodal model id (same server) used to describe a captured screenshot;
    # empty = pixel statistics and OCR text only.
    vision_model: str = ""
    vision_max_tokens: int = 400
    # When a `grep` finds nothing, fall back to the server's /v1/embeddings and
    # report the closest indexed units by meaning. Off by default: needs an
    # embedding model and one embedding pass over the index (then cached).
    semantic_search: bool = False
    # Embedding model id; empty = the chat model.
    embedding_model: str = ""
    semantic_top_k: int = 5
    # Cosine similarity below this is not shown; scales differ by model.
    semantic_min_score: float = 0.2
    # Read-only `diff` action: compare the working tree against Pedacito's
    # pre-write snapshots (.pedacito/backups/). Set false for models it confuses.
    allow_diff: bool = True
    # EXPERIMENTAL: require only `action` and `reasoning` in the gather step
    # schema so grammar-constrained decoding may omit unused fields (~20-40
    # output tokens per step). Off by default: all-required keeps the grammar
    # simplest for small models.
    compact_step_schema: bool = False
    # Treat a creation of a new path paired with a block that empties an
    # existing same-extension file as a real move (old file deleted). Set
    # false for strict create+clear semantics that never delete automatically.
    allow_rename: bool = True
    # Anchor a search block that doesn't match exactly to the most similar
    # window of the file. Off by default: edits then either match exactly (up
    # to whitespace) or fail. A window must score at least fuzzy_min_ratio and
    # clearly beat the runner-up; such edits are flagged as approximate.
    fuzzy_match: bool = False
    fuzzy_min_ratio: float = 0.85
    # Repair round-trips per apply when a search/replace block fails to match.
    repair_rounds: int = 2
    # Per-message OUTPUT ceiling the provider honours (hosted APIs cap one
    # reply far below their context window, e.g. ~8k tokens). When >0 and the
    # run targets a hosted API, auto_fit clamps reply budgets to it. Explicit
    # max_tokens_* values win over an auto-detected ceiling, not over one set
    # here. 0 = trust the declared budgets.
    max_output_tokens: int = 0

    # --- Context window -------------------------------------------------------
    # The server-side context length you configured for the model. Used to
    # sanity-check the budgets below. With auto_detect_context, the server's
    # real configured length replaces it, and auto_fit sizes every other
    # budget from the result.
    context_tokens: int = 8192
    # Tokens held back from context_tokens for the reply itself; auto_fit
    # lowers it when the window is too small.
    reserve_tokens: int = 2048
    # Fit every token budget to the real context window before each run
    # (see auto_fit): shrink oversized budgets, grow them when there is room.
    auto_fit: bool = True
    # Probe the server for its configured context length (LM Studio native
    # API, Lemonade /v1/models, Ollama /api/show). Overrides context_tokens
    # unless you set it explicitly. false = trust your file verbatim.
    auto_detect_context: bool = True
    # max_tokens sent on generation calls; auto_fit clamps it to the window.
    # 0 omits max_tokens from requests entirely.
    request_max_tokens_cap: int = 32768

    # --- Project scaffolding (`pedacito new`) ---------------------------------
    # Cap on files a plan may propose; the excess is reported.
    max_new_files: int = 24
    # Spec documents longer than this are truncated, with a warning.
    max_spec_chars: int = 60_000
    # Budgets for the two scaffold calls with no existing answer budget. Thinking
    # is disabled per call, so the whole budget goes to output.
    max_tokens_plan: int = 2000
    max_tokens_build: int = 8000

    # --- Autonomous loop (`pedacito auto`) ------------------------------------
    # A goal becomes a checklist; each task is implemented in a fresh context
    # with the plan file as the only surviving state.
    # Plan file (relative to the project root) used when --plan is not given.
    auto_plan_file: str = "PLAN.md"
    # Shell command run after every task; a failure feeds its output back for
    # repair. Empty = no verification.
    auto_verify: str = ""
    # Cap on tasks per `auto` run; 0 = no cap.
    auto_max_tasks: int = 0
    # Wall-clock cap per `auto` run in minutes; 0 = no cap. Checked between
    # tasks, so a running task finishes.
    auto_max_minutes: int = 0
    # Fix rounds per failing verification before a task is given up on.
    auto_repair_rounds: int = 2
    # `new --complex` / `auto`: a source file whose scaffold stub is longer than
    # this is split into several tasks (one per group of top-level
    # definitions). 0 disables splitting.
    auto_split_lines: int = 120
    # pip-install a missing third-party module into the running interpreter
    # when --verify fails on it, then retry. Off by default because the module
    # name comes from model-generated code; with it off the failure is reported
    # as a missing dependency and the repair loop is skipped.
    auto_install_deps: bool = False
    # At the end of a `--complex` build, run the program once: with no --verify
    # a project with main.py is smoke-run (still running at the timeout counts
    # as started); otherwise the entry module is imported. Failures are fed back
    # to the model.
    auto_smoke: bool = True
    auto_smoke_timeout: int = 20
    # Repair rounds for a failing runtime verification.
    auto_verify_rounds: int = 2

    # --- Reasoning models -----------------------------------------------------
    # <think> blocks are always stripped, but generating them costs tokens.
    # Which switch works depends on the model and runtime; setting both is harmless.
    #   disable_thinking -> sends chat_template_kwargs {"enable_thinking": false}
    #   thinking_suffix  -> appends a token like "/no_think" to your last message
    disable_thinking: bool = False
    thinking_suffix: str = ""

    # --- Editing --------------------------------------------------------------
    # Snapshot originals into .pedacito/backups/<timestamp>/ before writing;
    # `pedacito restore` puts them back.
    backup: bool = True
    # Write a readable record of each task to .pedacito/sessions/<timestamp>.md.
    log_sessions: bool = True
    # Open a side-by-side review page instead of printing a diff. Served from
    # 127.0.0.1 only, since it can write files.
    review: bool = False
    review_port: int = 0          # 0 lets the OS pick a free port
    open_browser: bool = True
    review_timeout: int = 1800    # seconds before an unanswered review expires

    # --- Workspace ------------------------------------------------------------
    # On first run, add .pedacito/ to the project's .gitignore.
    manage_gitignore: bool = True
    # On first run, create a commented .pedacitoignore.
    create_ignore_file: bool = True

    # Profile applied when --profile is not given (or PEDACITO_PROFILE). May be
    # set at the top level of the file, which `--setup` does; must exist in
    # [profiles.*].
    default_profile: str = ""
    # Preset tier ("tiny", "small", "medium", "large", "hosted") for a run
    # with no profile selected; profiles carry their own `tier` key.
    tier: str = ""

    # Bookkeeping for `pedacito config`; not settings.
    sources: dict = None
    profiles: dict = None
    active_profile: str = ""
    active_tier: str = ""   # preset tier applied to the active profile, if any
    # Model weights size in bytes, queried from the server by auto_fit.
    model_size_bytes: int = 0

    def __post_init__(self):
        # dataclasses cannot take mutable defaults directly
        if self.sources is None:
            self.sources = {}
        if self.profiles is None:
            self.profiles = {}


# "server" is the documented section name for connection settings; "lmstudio"
# is kept as an accepted alias so existing config files keep working. Sections
# are optional: a flat top-level `base_url = ...` works too.
_SECTIONS = ("universal", "server", "lmstudio", "hardware", "generation",
             "map", "indexing", "files", "agent", "scaffold", "editing",
             "reasoning", "review", "workspace")
PROFILE_SECTION = "profiles"

_FIELDS = {f.name: f for f in fields(Config)
           if f.name not in ("sources", "profiles", "active_profile",
                             "model_size_bytes", "_hosted_server")}

_ALIASES = {
    "vram": "vram_gb",
    # Common spellings from the runtime UIs / llama.cpp flags.
    "k_cache_type": "k_cache_quant",
    "v_cache_type": "v_cache_quant",
    "k_cache_quantization": "k_cache_quant",
    "v_cache_quantization": "v_cache_quant",
    "kv_cache_k_quant": "k_cache_quant",
    "kv_cache_v_quant": "v_cache_quant",
}

# Effective bits per cache element, including llama.cpp's per-block scale/min
# overhead. Only the size relative to f16 matters, so approximate is fine.
_KV_QUANT_BITS = {
    "f32": 32.0, "fp32": 32.0,
    "f16": 16.0, "fp16": 16.0, "bf16": 16.0,
    "q8_0": 8.5,
    "q6_k": 6.5625,
    "q5_1": 6.0, "q5_0": 5.5,
    "q4_1": 5.0, "q4_0": 4.5, "iq4_nl": 4.5, "iq4_xs": 4.25,
}
_KV_QUANT_SHORT = {
    "q8": "q8_0", "q6": "q6_k", "q6k": "q6_k",
    "q5": "q5_1", "q5.1": "q5_1", "q5.0": "q5_0",
    "q4": "q4_0", "q4.1": "q4_1", "q4.0": "q4_0",
    "float16": "f16", "half": "f16", "float32": "f32", "bfloat16": "bf16",
}


def _normalise_kv_quant(value) -> str:
    """Canonicalise a cache-quantization spelling, or raise ConfigError."""
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw or raw in ("auto", "default", "none", "false"):
        return ""
    raw = raw.replace(" ", "_")
    if raw in _KV_QUANT_BITS:
        return raw
    dotted = raw.replace("_", ".")
    if dotted in _KV_QUANT_SHORT:
        return _KV_QUANT_SHORT[dotted]
    if raw in _KV_QUANT_SHORT:
        return _KV_QUANT_SHORT[raw]
    raise ConfigError(
        f"k_cache_quant/v_cache_quant: unknown cache quantization {value!r}. "
        "Use f16, bf16, f32, q8_0, q6_k, q5_1, q5_0, q4_1, q4_0, "
        "iq4_nl or iq4_xs.")


def kv_quant_bits(value: str) -> float | None:
    """Effective bits per cache element for a quantization name, or None if unset."""
    name = _normalise_kv_quant(value)
    return _KV_QUANT_BITS.get(name) if name else None


def _canonical(name: str) -> str:
    """Resolve a config key spelling to its canonical Config field name."""
    return _ALIASES.get(name, name)


# Accepted spellings for json_mode; hosted APIs' docs use "json_object".
_JSON_MODE_ALIASES = {
    "schema": "schema", "json_schema": "schema",
    "object": "object", "json_object": "object", "json": "object",
}


class ConfigError(RuntimeError):
    """Invalid TOML, an unknown setting, a wrong type, or an unknown profile."""


def _flatten(data: dict, path: Path) -> tuple[dict, dict]:
    """Accept flat keys and named sections; reject anything unrecognised.

    Returns (settings, profiles), where profiles are named override sets
    (`[profiles.qwen]` tables).
    """
    flat: dict = {}
    profiles: dict = {}
    unknown: list[str] = []
    data = dict(data)
    for name, body in (data.pop(PROFILE_SECTION, None) or {}).items():
        if not isinstance(body, dict):
            raise ConfigError(f"{path}: [profiles.{name}] must be a table")
        # `tier` is a preset selector consumed by _apply_preset, not a setting.
        bad = [k for k in body
               if _canonical(k) not in _FIELDS and k != "tier"]
        if bad:
            raise ConfigError(f"{path}: [profiles.{name}] unknown setting(s): "
                              f"{', '.join(bad)}")
        profiles[name] = {_canonical(k): v for k, v in body.items()}
    for key, value in data.items():
        if isinstance(value, dict) and _canonical(key) != "sampler":
            if key not in _SECTIONS:
                unknown.append(f"[{key}]")
                continue
            for k, v in value.items():
                field = _canonical(k)
                if field in _FIELDS:
                    flat[field] = v
                else:
                    unknown.append(f"{key}.{k}")
        elif _canonical(key) in _FIELDS:
            flat[_canonical(key)] = value
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
    """Convert a raw TOML value to its Config field's type; forgives
    int-for-float and list-for-tuple, strict about everything else."""
    target = _FIELDS[name].type
    if isinstance(target, str):  # from __future__ annotations
        target = {"str": str, "float": float, "int": int, "bool": bool}.get(
            target, target)
    where = f"{path}: " if path else ""
    if name == "vram_gb":
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(
                f"{where}{name} must be a positive whole number of gigabytes, "
                f"got {value!r}")
        return value
    if name in ("k_cache_quant", "v_cache_quant"):
        return _normalise_kv_quant(value)
    if name == "kv_cache_bytes_per_token":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(
                f"{where}{name} must be a non-negative whole number of bytes, "
                f"got {value!r}")
        return value
    if name == "json_mode":
        norm = _JSON_MODE_ALIASES.get(str(value).strip().lower())
        if norm is None:
            raise ConfigError(
                f"{where}{name} must be 'schema' or 'object', got {value!r}"
                "  ('json_object' / 'json' mean 'object';"
                " 'json_schema' means 'schema')")
        return norm
    if name in ("extra_extensions", "shell_allowlist"):
        if not isinstance(value, (list, tuple)):
            example = ('["pytest", "python"]' if name == "shell_allowlist"
                       else '[".jinja", ".proto"]')
            raise ConfigError(f"{where}{name} must be a list, e.g. {example}")
        return tuple(str(v) for v in value)
    if name == "sampler":
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




_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "")


def _hostname(base_url: str) -> str:
    """Lower-cased host of `base_url`, or "" when it cannot be parsed."""
    try:
        return (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return ""


def load(root: Path | None = None, explicit: Path | None = None,
         use_env: bool = True, profile: str | None = None,
         require_hardware: bool = True, warn_budgets: bool = True) -> Config:
    """Build a Config by layering defaults, user config, project config, an
    explicit --config file, the selected profile, then PEDACITO_* environment
    variables. `cfg.sources` records which layer set each field.

    `vram_gb` is mandatory for local servers (auto-fit needs it); hosted APIs
    never need it. `pedacito config` passes require_hardware=False so it can
    inspect and repair an incomplete file."""
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

    # Top-level `tier` (no-profile preset); validated here so a typo fails loudly.
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

    # Profile choice: --profile, then PEDACITO_PROFILE, then default_profile,
    # then an implicit [profiles.default].
    origin = "--profile"
    if not profile:
        profile = os.environ.get("PEDACITO_PROFILE") or ""
        origin = "env PEDACITO_PROFILE"
    if not profile:
        profile = cfg.default_profile
        origin = "default_profile"
    if not profile and "default" in profiles:
        profile = "default"
        origin = "implicit [profiles.default]"
    if profile:
        if profile not in profiles:
            known = ", ".join(sorted(profiles)) or "(none defined)"
            # XDG_CONFIG_HOME can silently redirect the user config, so say
            # which files were actually searched.
            searched = "; ".join(f"{label}: {path}" for label, path in layers) \
                or f"no config file found (expected: {USER_CONFIG})"
            hint = ("Define it in your config, or clear `default_profile`. "
                    "Run `pedacito config --setup` to build both interactively."
                    if origin == "default_profile" and not profiles else "")
            raise ConfigError(
                f"No profile named '{profile}' (from {origin}). "
                f"Defined profiles: {known}\n"
                f"Looked in: {searched}\n"
                + (hint + "\n" if hint else "") +
                "Add one to your config:\n\n"
                f"  [profiles.{profile}]\n  model = \"...\"")
        explicit = set(profiles[profile])
        body, tier = _apply_preset(profile, profiles[profile], origin)
        for key, value in body.items():
            setattr(cfg, key, _coerce(key, value))
            # Preset-derived values may be replaced by auto_fit; keys the user
            # wrote in the profile are explicit limits.
            how = "explicit" if key in explicit else "tier preset"
            cfg.sources[key] = f"profile '{profile}' (via {origin}; {how})"
        cfg.active_profile = profile
        cfg.active_tier = tier

    # default_profile is excluded: profile selection has already happened.
    if use_env:
        for key in _FIELDS:
            if key == "default_profile":
                continue
            raw = os.environ.get("PEDACITO_" + key.upper())
            if raw is None:
                continue
            setattr(cfg, key, _parse_env(key, raw))
            cfg.sources[key] = f"env PEDACITO_{key.upper()}"

    # Hosted APIs (DeepSeek in particular) reject response_format json_schema,
    # so switch to "object" unless the user spelled json_mode somewhere.
    host = _hostname(cfg.base_url)
    remote = host not in _LOCAL_HOSTS
    deepseek = "deepseek" in cfg.model.lower() or "deepseek" in host
    if (remote or deepseek) and cfg.sources.get("json_mode") == "default":
        cfg.json_mode = "object"
        cfg._hosted_server = True
        cfg.sources["json_mode"] = (
            "auto-detected hosted API "
            f"({host or cfg.model}); set json_mode explicitly to override")
    elif deepseek and cfg.json_mode == "schema":
        print("[config warning] Model/server is DeepSeek but json_mode was "
              "'schema' -- switching to \"object\" automatically. DeepSeek "
              "rejects json_schema; without this, runs fail with 'would not "
              "produce valid JSON'. Set json_mode explicitly to override.",
              file=sys.stderr)
        cfg.json_mode = "object"
        cfg._hosted_server = True

    # Hosted thinking models (e.g. dashscope Qwen) cut a single reply off near
    # 8k tokens despite a huge context window; default max_output_tokens to
    # that so auto_fit clamps tier/default reply budgets. Not applied to plain
    # deepseek-chat or generic hosts, whose ceilings are larger; set
    # max_output_tokens explicitly for those.
    qwen_thinking = any(t in cfg.model.lower() for t in
                        ("qwq", "qwen3", "qwen-3", "thinking"))
    if qwen_thinking and remote and cfg.sources.get("max_output_tokens") == "default":
        cfg.max_output_tokens = 8000
        cfg.sources["max_output_tokens"] = (
            f"auto-detected output ceiling for hosted thinking model "
            f"({cfg.model}); set max_output_tokens explicitly to override")

    if require_hardware:
        missing = missing_hardware(cfg)
        if missing:
            raise ConfigError(
                "vram_gb is mandatory for local model servers: auto-fit uses "
                "it (with the model's file size, queried from the server) to "
                "size the context window.\n"
                "Add it to your config, in whole gigabytes of VRAM on the "
                "machine running the model:\n\n"
                "  vram_gb = 16\n\n"
                "Missing: " + ", ".join(missing) + "\n"
                "Run `pedacito config --setup` to have it written for you. "
                "Hosted APIs (DeepSeek/OpenAI) do not need it.")

    # Non-fatal: budgets that cannot fit the context window cause truncation
    # errors. `pedacito how` opts out: it uses its own excerpt budget.
    if warn_budgets:
        for warning in budget_warnings(cfg):
            print(f"[config warning] {warning}", file=sys.stderr)
    return cfg


def budget_warnings(cfg: Config) -> list[str]:
    """Check token/character budgets against `context_tokens`. Returns
    warnings (empty = fits); never raises."""
    w: list[str] = []
    # Hosted APIs have no local window: context_tokens is a budget there, so
    # the "cannot fit" warnings would only be noise.
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


def is_hosted(cfg: Config) -> bool:
    """True when the config targets a remote API rather than a local server.

    Signals: `tier = "hosted"`, a detected hosted server, `json_mode =
    "object"` (set only for hosted APIs), or a non-localhost base_url. auto_fit
    and llm.py leave hosted budgets unclamped because there is no hardware
    window to fit them to.
    """
    if cfg.active_tier == "hosted" or cfg._hosted_server:
        return True
    if cfg.json_mode == "object":
        return True
    return _hostname(cfg.base_url) not in _LOCAL_HOSTS


_GIB = 1024 ** 3
# KV-cache heuristic: at about this many tokens of context the cache is roughly
# as large as the model's weights on disk (fits common llama/gemma/qwen GGUFs:
# a 27B Q4 is ~16 GB at 32k).
_KV_REFERENCE_TOKENS = 32768


def missing_hardware(cfg: Config) -> list[str]:
    """Mandatory hardware settings still unset; always empty for hosted APIs."""
    if is_hosted(cfg):
        return []
    return ["vram_gb"] if int(cfg.vram_gb or 0) <= 0 else []


def hardware_context_tokens(vram_gb: int, model_size_bytes: int,
                            k_cache_quant: str = "",
                            v_cache_quant: str = "",
                            kv_cache_bytes_per_token: int = 0) -> int:
    """Estimate how many context tokens fit in VRAM beside the model weights.

    System RAM is not counted (spilling is slow); 0.5 GB is kept for the
    driver/desktop. Cache bytes per token come from the file size (see
    _KV_REFERENCE_TOKENS), scaled by the K/V quantization bits; an explicit
    `kv_cache_bytes_per_token` bypasses both. Returns 0 when the model does
    not fit; callers then keep the fallback budgets.
    """
    if model_size_bytes <= 0:
        return 0
    vram = max(0, int(vram_gb)) * _GIB
    free = vram - int(model_size_bytes) - (_GIB // 2)
    if free <= 0:
        return 0
    if kv_cache_bytes_per_token and kv_cache_bytes_per_token > 0:
        kv_per_token = int(kv_cache_bytes_per_token)
    else:
        kv_per_token = max(16 * 1024,
                           int(model_size_bytes) // _KV_REFERENCE_TOKENS)
        k_bits = kv_quant_bits(k_cache_quant)
        v_bits = kv_quant_bits(v_cache_quant)
        if k_bits is not None or v_bits is not None:
            # Baseline is f16 for both K and V; an unset side stays f16.
            k_bits = 16.0 if k_bits is None else k_bits
            v_bits = 16.0 if v_bits is None else v_bits
            kv_per_token = max(
                4 * 1024, int(kv_per_token * (k_bits + v_bits) / 32.0))
    return int(free // kv_per_token)


def _is_explicit_limit(source: str) -> bool:
    """True when a budget came from a user choice (file/profile/env/CLI).

    Presets, defaults and earlier auto_fit/auto-detected values are fallbacks
    that hardware-derived budgets may replace.
    """
    if not source or source == "default" or "preset" in source:
        return False
    return not source.startswith(("auto-fit", "auto-detected"))


def auto_fit(cfg: Config, verbose: bool = True) -> list[str]:
    """Fit every token budget to the server's real context window.

    Built-in budgets assume a roomy window, while servers often default to a
    small one (LM Studio 8192, Ollama 4096). This shrinks budgets that do not
    fit and grows them when there is room:

      * reserve_tokens shrinks first so tiny windows keep usable space;
      * max_tokens_answer / build are clamped to what the window can hold;
      * max_gathered_chars and flat_map_token_budget split the rest so map +
        source + one step fit in (context - reserve);
      * request_max_tokens_cap clamps every max_tokens we send.

    With cfg.auto_detect_context on and no explicit context_tokens, the live
    server is probed (LM Studio native API, Lemonade /v1/models, Ollama
    /api/show) and its configured length is used; probe failures are silent.

    For local servers, `vram_gb` and the model file size feed a KV-cache
    estimate. A server-reported or user-set context length is authoritative
    (the estimate only warns when it is lower); otherwise the estimate caps the
    fallback window. Tier/default budgets are kept when the whole set fits,
    else VRAM-derived ones are used; explicit user values override both.

    Returns notes about what changed (empty = nothing). Raises ConfigError if
    `vram_gb` is missing for a local server.
    """
    notes: list[str] = []

    # max_output_tokens is a hard provider cap, honoured even with auto_fit
    # off. An auto-detected cap (a guess) must not rewrite explicit user
    # budgets; one the user set themselves does.
    mot0 = int(cfg.max_output_tokens or 0)
    mot_explicit = _is_explicit_limit(
        cfg.sources.get("max_output_tokens", "default"))
    if mot0 > 0:
        for key in ("max_tokens_answer", "max_tokens_build",
                    "max_tokens_plan", "max_tokens_step"):
            v = getattr(cfg, key)
            if v <= mot0:
                continue
            if _is_explicit_limit(cfg.sources.get(key, "default")) \
                    and not mot_explicit:
                continue
            setattr(cfg, key, mot0)
            cfg.sources[key] = f"max_output_tokens ({mot0:,})"
            notes.append(f"{key}: {v:,} -> {mot0:,}")

    if not cfg.auto_fit:
        if mot0 > 0 and verbose and notes:
            print("[config] clamped to max_output_tokens: "
                  + "; ".join(notes), file=sys.stderr)
        return notes

    # Hosted APIs have no local window: context_tokens is a declared budget,
    # so hosted runs only ever grow budgets; shrinking is reserved for
    # servers we can probe.
    hosted = is_hosted(cfg)

    ctx = cfg.context_tokens
    origin = cfg.sources.get("context_tokens", "default")
    if not hosted:
        missing = missing_hardware(cfg)
        if missing:
            raise ConfigError(
                "auto-fit needs vram_gb for local model servers; missing: "
                + ", ".join(missing) + ".\n"
                "Set it in whole gigabytes of VRAM on the machine running "
                "the model, or run `pedacito config --setup`.")
        info: dict = {}
        # A pre-set size saves a round trip and works without a live server.
        preset_size = int(cfg.model_size_bytes or 0)
        need_probe = preset_size <= 0 or (
            cfg.auto_detect_context and origin == "default")
        if need_probe:
            try:
                from .llm import LLMClient
                info = LLMClient(cfg).model_info() or {}
            except Exception:
                info = {}
        try:
            model_bytes = int(info.get("size_bytes") or 0)
        except (TypeError, ValueError):
            model_bytes = 0
        if model_bytes <= 0:
            model_bytes = preset_size
        cfg.model_size_bytes = model_bytes
        if model_bytes > 0:
            cfg.sources.setdefault("model_size_bytes",
                                   "auto-detected from server")
        # Only the configured/loaded length is safe to fit to, not the
        # model's advertised maximum (max_context_length).
        detected = info.get("context_length")
        hw_ctx = (hardware_context_tokens(
                      cfg.vram_gb, model_bytes,
                      cfg.k_cache_quant, cfg.v_cache_quant,
                      int(cfg.kv_cache_bytes_per_token or 0))
                  if model_bytes > 0 else 0)
        user_ctx = _is_explicit_limit(origin)
        server_ctx = None
        if detected and cfg.auto_detect_context and not user_ctx:
            server_ctx = detected
            if detected != ctx:
                notes.append(f"context: server reports {detected:,} tokens "
                             f"(config fallback was {ctx:,}); using it")
                ctx = detected
                cfg.context_tokens = ctx
                cfg.sources["context_tokens"] = "auto-detected from server"

        # The VRAM estimate caps the window only when neither the server nor
        # the user supplied a length: it is a heuristic that can be off 2-4x
        # (GQA, cache quantization), so a configured length below it only
        # produces a speed warning.
        if model_bytes > 0:
            size_gb = model_bytes / _GIB
            mem = f"{cfg.vram_gb} GB VRAM"
            kv_hint = ""
            if cfg.kv_cache_bytes_per_token:
                kv_hint = (f"; KV cache assumed "
                           f"{int(cfg.kv_cache_bytes_per_token) / 1024:.0f} "
                           f"KiB/token")
            elif cfg.k_cache_quant or cfg.v_cache_quant:
                kv_hint = (f"; KV cache K={cfg.k_cache_quant or 'f16'}, "
                           f"V={cfg.v_cache_quant or 'f16'}")
            if hw_ctx <= 0:
                notes.append(
                    f"memory: model is {size_gb:.1f} GB and {mem} leaves no "
                    f"room after the 0.5 GB system reserve; keeping the "
                    f"configured/fallback window ({ctx:,} tokens){kv_hint}")
            else:
                notes.append(
                    f"memory: model {size_gb:.1f} GB on {mem} leaves about "
                    f"{hw_ctx:,} context tokens (0.5 GB kept for the "
                    f"system{kv_hint})")
                if user_ctx:
                    new_ctx = ctx
                    if hw_ctx < ctx:
                        notes.append(
                            f"warning: configured context is {ctx:,} tokens "
                            f"but the VRAM estimate only fits about "
                            f"{hw_ctx:,} -- the KV cache will spill to system "
                            f"RAM and tokens/sec will drop; lower the "
                            f"configured context or set k_cache_quant/"
                            f"v_cache_quant to match the server")
                    else:
                        notes.append(
                            f"context: keeping configured {ctx:,} (fits the "
                            f"VRAM estimate)")
                elif server_ctx:
                    new_ctx = server_ctx
                    if hw_ctx < server_ctx:
                        notes.append(
                            f"warning: server is configured for {server_ctx:,} "
                            f"tokens but the VRAM estimate only fits about "
                            f"{hw_ctx:,} -- the KV cache will spill to system "
                            f"RAM and tokens/sec will drop; lower the server's "
                            f"context length (LM Studio 'Context length' / "
                            f"Ollama num_ctx / Lemonade model settings), or "
                            f"set k_cache_quant/v_cache_quant to match the "
                            f"server")
                    else:
                        notes.append(
                            f"context: keeping {ctx:,} (fits the VRAM estimate)")
                else:
                    new_ctx = hw_ctx
                    if new_ctx < ctx:
                        notes.append(
                            f"context: VRAM estimate {new_ctx:,} < fallback "
                            f"{ctx:,}; using {new_ctx:,} so the KV cache stays in "
                            f"VRAM")
                    elif new_ctx > ctx:
                        notes.append(
                            f"context: VRAM estimate {new_ctx:,} > fallback "
                            f"{ctx:,}; using {new_ctx:,}")
                    else:
                        notes.append(
                            f"context: keeping {ctx:,} (fits the VRAM estimate)")
                if new_ctx != ctx:
                    ctx = new_ctx
                    cfg.context_tokens = ctx
                    cfg.sources["context_tokens"] = (
                        f"auto-fit (VRAM: {mem}, model {size_gb:.1f} GB)")
        elif model_bytes <= 0:
            notes.append(
                "memory: server did not report the model file size; using "
                "the server/fallback window without a VRAM estimate")

    def setv(key: str, value):
        cur = getattr(cfg, key)
        if cur == value:
            return
        src = cfg.sources.get(key, "default")
        # Explicit user values are only clamped when bigger than the window.
        if _is_explicit_limit(src) and value < cur:
            hard_max = max(1, ctx - 1)
            if cur <= hard_max:
                return          # user set it and it still fits -- leave it
        # Hosted runs are grow-only.
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

    mot = int(cfg.max_output_tokens or 0)

    # Hardware-derived budgets: tier/default values are floors while the whole
    # set fits the window, else the calculated allocation wins. Explicit user
    # values override both, subject to the window's physical cap.
    if not hosted and model_bytes > 0:
        # The VRAM estimate only caps the window when neither the server nor
        # the user supplied one.
        if server_ctx or user_ctx:
            calc_ctx = ctx
        else:
            calc_ctx = min(hw_ctx, ctx) if hw_ctx > 0 else 0
        calc_reserve = min(cfg.reserve_tokens, max(512, calc_ctx // 4))
        calc_usable = max(calc_ctx - calc_reserve, calc_ctx // 2)
        calc_step = max(512, min(4096, calc_usable // 8))
        calc_summary = max(160, min(400, calc_step // 4))
        calc_answer = max(1500, min(24000, calc_usable * 7 // 20))
        calc_build = max(4000, min(32000, calc_usable * 6 // 10))
        calc_plan = max(1200, min(8000, calc_usable // 6))
        if mot > 0:
            calc_step = min(calc_step, mot)
            calc_answer = min(calc_answer, mot)
            calc_build = min(calc_build, mot)
            calc_plan = min(calc_plan, mot)

        def want(key: str, calculated: int) -> int:
            """max(tier/default, calculated); explicit user values win."""
            cur = getattr(cfg, key)
            if _is_explicit_limit(cfg.sources.get(key, "default")):
                return cur
            return max(cur, calculated)

        # Calculated map/source budgets for the VRAM window.
        calc_prompt_room = max(calc_usable - calc_answer, 256)
        calc_map = max(1500, min(12000, calc_prompt_room // 3))
        calc_map = min(calc_map, max(256, calc_prompt_room - 256))
        calc_gathered = max(calc_prompt_room - calc_map, 256) * 4

        want_step = want("max_tokens_step", calc_step)
        want_summary = want("max_tokens_summary", calc_summary)
        want_answer = want("max_tokens_answer", calc_answer)
        want_build = want("max_tokens_build", calc_build)
        want_plan = want("max_tokens_plan", calc_plan)
        want_map = want("flat_map_token_budget", calc_map)
        want_gathered = want("max_gathered_chars", calc_gathered)

        # Tier/default floors are only safe if the answer prompt fits.
        floors_fit = (want_answer + want_map + want_gathered // 4 <= usable
                      and want_step <= max(128, usable // 2))
        if not floors_fit:
            def or_calculated(key: str, calculated: int) -> int:
                cur = getattr(cfg, key)
                if _is_explicit_limit(cfg.sources.get(key, "default")):
                    return cur
                return calculated
            want_answer = or_calculated("max_tokens_answer", calc_answer)
            want_build = or_calculated("max_tokens_build", calc_build)
            want_plan = or_calculated("max_tokens_plan", calc_plan)
            want_map = or_calculated("flat_map_token_budget", calc_map)
            want_gathered = or_calculated("max_gathered_chars", calc_gathered)
            notes.append(
                "budgets: tier/default floors do not all fit the VRAM "
                "window; using the calculated allocation")

        step_v = min(want_step, max(128, usable // 2))
        summary_v = min(want_summary, max(64, step_v // 3))
        answer_v = min(want_answer, cap, max(256, usable - 256))
        build_v = min(want_build, cap, max(256, usable - 256))
        plan_v = min(want_plan, cap, max(256, usable - 256))

        # Leave room for map + gathered source in the answer prompt.
        prompt_room = max(usable - answer_v, 256)
        map_v = min(want_map, max(256, prompt_room - 256))
        gathered_cap = max(1024, (prompt_room - map_v) * 4)
        gathered_v = min(want_gathered, gathered_cap, 200000)

        targets = (
            ("max_tokens_step", step_v),
            ("max_tokens_summary", summary_v),
            ("max_tokens_answer", answer_v),
            ("max_tokens_build", build_v),
            ("max_tokens_plan", plan_v),
            ("flat_map_token_budget", map_v),
            ("max_gathered_chars", gathered_v),
        )
        changed = False
        for key, value in targets:
            cur = getattr(cfg, key)
            if cur == value:
                continue
            setattr(cfg, key, value)
            cfg.sources[key] = (
                f"auto-fit (hardware budget, window {ctx:,} tok)")
            notes.append(f"{key}: {cur:,} -> {value:,} (hardware)")
            changed = True
        if changed and floors_fit:
            notes.append(
                "tier/default budgets were used as floors; hardware raised "
                "them where it had room, and explicit user values won")
    else:
        # No model size or a hosted API: keep tier/default values and only
        # clamp what cannot fit (setv never shrinks hosted budgets).
        step_ceiling = max(128, usable // 2)
        if cfg.max_tokens_step > step_ceiling:
            setv("max_tokens_step", step_ceiling)
        summary_ceiling = max(64, min(cfg.max_tokens_summary,
                                      step_ceiling // 3))
        if cfg.max_tokens_summary != summary_ceiling:
            setv("max_tokens_summary", summary_ceiling)

        # Answer/build budgets: never ask for more than fits after the prompt.
        setv("max_tokens_answer", min(cfg.max_tokens_answer, cap,
                                      usable - 256))
        setv("max_tokens_build", min(cfg.max_tokens_build, cap, usable - 256))
        setv("max_tokens_plan", min(cfg.max_tokens_plan, cap, usable - 256))

        # Provider per-message output ceiling: clamp reply budgets to it
        # (setv leaves explicit budgets alone unless the cap is user-set).
        if mot > 0:
            for key in ("max_tokens_answer", "max_tokens_build",
                        "max_tokens_plan", "max_tokens_step"):
                v = getattr(cfg, key)
                if v > mot:
                    before = v
                    setv(key, mot)
                    if getattr(cfg, key) != before:
                        cfg.sources[key] = (
                            f"auto-fit (max_output_tokens {mot:,})")

        # Map + gathered source share the rest of the window. 4 chars/token.
        left = max(usable - max(cfg.max_tokens_step, cfg.max_tokens_summary),
                   512)
        map_tok = min(cfg.flat_map_token_budget, max(400, left // 3))
        setv("flat_map_token_budget", map_tok)
        gathered_tok = max(left - map_tok, 256)
        setv("max_gathered_chars", min(cfg.max_gathered_chars,
                                       gathered_tok * 4))

    if not hosted:
        # Final pass: every reply budget must leave room for the prompt
        # (llama.cpp 400s when prompt + max_tokens > n_ctx; Ollama truncates).
        hard_cap = max(128, ctx - 512)
        for key in ("max_tokens_step", "max_tokens_summary", "max_tokens_answer",
                    "max_tokens_build", "max_tokens_plan"):
            v = getattr(cfg, key)
            if v > hard_cap:
                setv(key, hard_cap)

    if notes and verbose:
        print("[auto-fit] " + "; ".join(notes), file=sys.stderr)
    return notes


def _parse_env(name: str, raw: str):
    """Convert a raw PEDACITO_<NAME> environment string to the field's type."""
    target = _FIELDS[name].type
    if isinstance(target, str):
        target = {"str": str, "float": float, "int": int, "bool": bool}.get(target, str)
    if name == "json_mode":
        norm = _JSON_MODE_ALIASES.get(raw.strip().lower())
        if norm is None:
            raise ConfigError(
                f"PEDACITO_JSON_MODE={raw!r} must be 'schema' or 'object'"
                "  ('json_object' / 'json' mean 'object')")
        return norm
    if name in ("extra_extensions", "shell_allowlist"):
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    if name in ("k_cache_quant", "v_cache_quant"):
        return _normalise_kv_quant(raw)
    if name == "sampler":
        try:
            obj = json.loads(raw)
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
        value = target(raw) if target in (int, float) else raw
    except ValueError as e:
        raise ConfigError(f"PEDACITO_{name.upper()}={raw!r} is not a valid {target.__name__}") from e
    if name == "vram_gb" and value <= 0:
        raise ConfigError(
            f"PEDACITO_{name.upper()}={raw!r} must be a positive whole "
            f"number of gigabytes")
    return value


# Token-limit presets per class of model/server. A profile naming a tier
# starts from its preset; keys written in [profiles.NAME] override it. With no
# profile, built-in defaults apply (unless a top-level `tier` is set).
#
#   tiny    weakest local models (3-4B; or a 7-8B failing at small's budgets)
#   small   ~9B local models (qwen3-8b, gemma-3-12b, llama-3.1-8b)
#   medium  ~14B/32B local models (qwen3-14b, qwen3-32b, gemma-3-27b)
#   large   big local models (27B+ at full context, 70B rigs)
#   hosted  capable APIs: DeepSeek, OpenAI, Anthropic, xAI, ...
PRESET_TIERS = ("tiny", "small", "medium", "large", "hosted")

PRESETS: dict[str, dict] = {
    "tiny": {
        # 4k-window fallbacks for models that miss anchors or truncate at
        # small's budgets: shorter reads and replies, and the compact step
        # schema (unused fields may be omitted). Guessing picks this only for
        # <=4B model names; set tier = "tiny" explicitly for a larger one.
        "max_tokens_summary": 120,
        "max_tokens_step": 400,
        "max_steps": 0,
        "max_tokens_answer": 1200,
        "max_tokens_plan": 1000,
        "max_tokens_build": 3000,
        "flat_map_token_budget": 800,
        "max_gathered_chars": 7000,
        "max_read_lines": 80,
        "max_spec_chars": 12000,
        "max_new_files": 8,
        "repair_rounds": 1,
        "context_tokens": 4096,
        "reserve_tokens": 1024,
        "compact_step_schema": True,
    },
    "small": {
        # Fits an 8k window. These are fallbacks: when the server reports the
        # model size, auto_fit derives larger budgets from VRAM; they still
        # matter for max_steps, max_read_lines, max_spec_chars, max_new_files
        # and servers that expose no size.
        "max_tokens_summary": 160,
        "max_tokens_step": 500,
        "max_steps": 0,
        "max_tokens_answer": 1500,
        "max_tokens_plan": 1200,
        "max_tokens_build": 4000,
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
        # ~14B-32B local models, 32k-window fallbacks. max_tokens_step is
        # generous because thinking models spend it on hidden reasoning before
        # the JSON action.
        "max_tokens_summary": 240,
        "max_tokens_step": 1500,
        "max_steps": 0,
        "max_tokens_answer": 8000,
        "max_tokens_plan": 5000,
        "max_tokens_build": 16000,
        "flat_map_token_budget": 6000,
        "max_gathered_chars": 48000,
        "max_read_lines": 200,
        "max_spec_chars": 60000,
        "max_new_files": 24,
        "repair_rounds": 2,
        "context_tokens": 32768,
        "reserve_tokens": 4096,
    },
    "large": {
        # Big local rig: every budget is at least medium's.
        "max_tokens_summary": 320,
        "max_tokens_step": 2500,
        "max_steps": 0,
        "max_tokens_answer": 12000,
        "max_tokens_plan": 8000,
        "max_tokens_build": 24000,
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
        # The window is a declared budget, not a server limit: auto_fit only
        # grows these. max_tokens_step stays below ~8k because the provider's
        # per-message output ceiling cannot be raised by any client setting;
        # big outputs are chunked instead (see max_write_lines).
        "max_tokens_summary": 400,
        "max_tokens_step": 7000,
        "max_steps": 0,
        "max_tokens_answer": 40000,
        "max_tokens_plan": 8000,
        "max_tokens_build": 32000,
        "flat_map_token_budget": 12000,
        "max_gathered_chars": 100000,
        "max_read_lines": 400,
        "max_write_lines": 120,
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
        if size <= 4:
            return "tiny"
        if size <= 10:
            return "small"
        if size <= 20:
            return "medium"
        return "large"
    return None


def _apply_preset(name: str, body: dict, origin: str) -> tuple[dict, str]:
    """Return (settings, tier) for a profile, with its tier preset underneath.

    Tier source, first match wins: an explicit `tier` key (consumed here, never
    reaches Config); a heuristic on the profile/model/server name; none.
    Keys written in the profile always beat the preset.
    """
    body = dict(body)
    tier = body.pop("tier", None)
    if tier is not None:
        if tier not in PRESETS:
            raise ConfigError(
                f"profile '{name}' ({origin}): tier must be one of "
                f"{', '.join(PRESET_TIERS)}, got {tier!r}")
    elif name == "default":
        # [profiles.default] is an overlay on [universal]/[indexing]/[agent];
        # guessing a tier would silently replace those budgets.
        tier = ""
    else:
        tier = _guess_tier(name, body) or ""
    if not tier:
        return body, ""
    merged = dict(PRESETS[tier])
    merged.update(body)
    return merged, tier


def _toml_literal(value) -> str:
    """Render a Config default as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} = {_toml_literal(v)}"
                               for k, v in value.items()) + "}"
    return json.dumps(str(value))


_PROFILE_FIELD_EXCLUDE = {
    "base_url", "model", "vram_gb", "default_profile", "tier", "active_tier",
}


def _profile_override_comments() -> str:
    """Every profile-overridable setting, commented at its built-in default."""
    out: list[str] = []
    for name, f in _FIELDS.items():
        if name in _PROFILE_FIELD_EXCLUDE:
            continue
        if f.default is not MISSING:
            value = f.default
        elif f.default_factory is not MISSING:
            value = f.default_factory()
        else:
            continue
        out.append(f"# {name} = {_toml_literal(value)}")
    return "\n".join(out)


TEMPLATE = '''\
# Pedacito configuration.
#
# Run `pedacito config --setup` to generate this file interactively, or edit
# the values below. The file is profile-based:
#
#   [universal]  optional defaults that apply to every profile
#   [indexing]   chunking, retrieval, and semantic-search defaults
#   [agent]      tool opt-ins, budgets, and autonomous-loop settings
#   [profiles.X] server/model-specific settings
#
# `[profiles.default]` is the fallback when no profile is named. Every key
# there and in the sections above can be overridden by a profile. Select one
# with --profile NAME, PEDACITO_PROFILE, or default_profile below.
#
# MANDATORY for a local setup: base_url and model in [profiles.default].
# vram_gb is a shared hardware setting below; override it inside a profile only
# when that model runs on a different GPU. Hosted APIs do not need vram_gb.
#
# Precedence: built-ins < this file < project .pedacito.toml < --config FILE
#             < --profile NAME < CLI flags.

default_profile = "default"

[universal]
# Connection, generation, and shared hardware/workspace defaults. A profile
# overrides any of these. See `pedacito config --reference` for the full list.
vram_gb = 8
timeout = 600.0
load_wait_seconds = 240
max_consecutive_failures = 3
json_mode = "schema"
temperature = 0.2
# sampler = { top_p = 0.95, top_k = 40, min_p = 0.05, repeat_penalty = 1.1 }
disable_thinking = false
thinking_suffix = ""
k_cache_quant = ""
v_cache_quant = ""
kv_cache_bytes_per_token = 0
max_tokens_summary = 220
max_tokens_step = 700
max_tokens_answer = 3000
max_output_tokens = 0
context_tokens = 8192
reserve_tokens = 2048
auto_fit = true
auto_detect_context = true
request_max_tokens_cap = 32768
flat_map_token_budget = 4000
file_card_line_ranges = true
repair_rounds = 2
backup = true
log_sessions = true
review = false
review_port = 0
open_browser = true
review_timeout = 1800
manage_gitignore = true
create_ignore_file = true

[indexing]
# Chunking and retrieval. Oversized units split at statement boundaries.
class_split_lines = 60
max_unit_lines = 220
max_unit_chars = 12000
max_unit_tokens = 3000
unit_overlap_lines = 0
max_chunk_chars = 6000
min_lines_to_summarise = 4
summarise_methods = false
summarise_documented = false
include_references = true
extra_extensions = []
max_reference_bytes = 2000000
summarise_references = false
# Semantic search is opt-in. It needs an embeddings-capable server.
semantic_search = false
embedding_model = ""
semantic_top_k = 5
semantic_min_score = 0.2

[agent]
# Project context and the model's own checklist.
use_project_instructions = true
max_instruction_chars = 2000
max_todo_chars = 1500
max_steps = 0
max_completion_rounds = 0
max_continuation_rounds = 0
max_gathered_chars = 24000
max_read_lines = 200
read_context_hint = true

# Opt-in tool capabilities. Flip these only when the model should use them.
allow_write = false
max_write_chars = 200000
max_write_lines = 120
allow_shell = false
shell_timeout = 120
shell_idle_seconds = 10
shell_allowlist = []
max_shell_output_chars = 8000
allow_screenshots = false
screenshot_wait_seconds = 5
allow_browser = false
browser_wait_seconds = 3
browser_server_wait_seconds = 15
browser_headless = true
browser_timeout_seconds = 20
vision_model = ""
vision_max_tokens = 400
allow_diff = true
compact_step_schema = false
allow_rename = true
fuzzy_match = false
fuzzy_min_ratio = 0.85

# Diagnostics and MCP are opt-in. MCP also needs mcp_servers configured.
auto_diagnostics = false
post_edit_check = true
auto_allowlist = false
diagnostics_timeout = 90
max_diagnostics_chars = 8000
allow_mcp = false
mcp_servers = []
mcp_timeout = 30

# Autonomous loop (`pedacito auto`).
auto_plan_file = "PLAN.md"
auto_verify = ""
auto_max_tasks = 0
auto_max_minutes = 0
auto_repair_rounds = 2
auto_split_lines = 120
auto_install_deps = false
auto_smoke = true
auto_smoke_timeout = 20
auto_verify_rounds = 2

# Project scaffolding (`pedacito new`).
max_new_files = 24
max_spec_chars = 60000
max_tokens_plan = 2000
max_tokens_build = 8000

[profiles.default]
# The fallback profile. Replace these with your server and model.
base_url = "http://localhost:1234/v1"
model    = "google/gemma-3-27b"

# Every profile-overridable setting, commented at its built-in default.
# Uncomment only the keys this profile needs; a profile beats [universal],
# [indexing], and [agent].
# vram_gb = 8                   # optional per-model hardware override
# tier = "medium"               # tiny | small | medium | large | hosted
''' + _profile_override_comments() + '''

# --- Local model example (Qwen) --------------------------------------------
# [profiles.example_qwen]
# base_url = "http://127.0.0.1:11434/v1"
# model    = "Qwen3.8-27B-GSQ-RCO-GGUF"
# vram_gb  = 24
# tier     = "medium"
# json_mode = "schema"
# semantic_search = true
# k_cache_quant = "q8_0"
# v_cache_quant = "q5_1"
# allow_shell = true
# allow_screenshots = true
# allow_browser = true
# max_gathered_chars = 48000
# max_read_lines = 200

# --- Hosted API example (DeepSeek) -----------------------------------------
# [profiles.example_deepseek]
# base_url = "https://api.deepseek.com/v1"
# api_key  = "sk-..."
# model    = "deepseek-chat"
# tier     = "hosted"
# json_mode = "object"          # hosted APIs usually reject json_schema
# timeout  = 120.0
# max_tokens_answer = 40000
# context_tokens = 128000
# max_gathered_chars = 100000
# max_read_lines = 400
'''


# `--init` writes TEMPLATE; `--reference` appends these longer notes.
REFERENCE_TEMPLATE = TEMPLATE + '''

# --- Detailed notes ----------------------------------------------------------
#
# The value on each commented line is the built-in default. Uncomment what you
# need. Section names are organizational; any valid field is accepted under any
# section, and a profile may override any universal/indexing/agent field.
#
# --- Connection and generation ---------------------------------------------
# base_url / model are mandatory in the default local profile.
# vram_gb is a shared hardware setting in [universal]; override it in a
# profile when that model runs on a different GPU.
# api_key is ignored by local servers; any string works.
# json_mode: "schema" for llama.cpp/LM Studio, "object" for hosted APIs.
# timeout: seconds per request. load_wait_seconds: model load wait.
# max_consecutive_failures: abort an indexing batch after N failures.
# temperature: keep low for tool calls.
# sampler: per-server sampler keys, usually placed inside a profile.
#
# --- Context and token budgets ---------------------------------------------
# context_tokens is the server-side window. reserve_tokens is held back for
# replies. Pedacito auto-fits budgets from the server's reported context and
# VRAM; explicit profile values always win.
# k_cache_quant / v_cache_quant let the VRAM estimate match the runtime's
# KV-cache quantization (f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, ...).
# kv_cache_bytes_per_token is a measured escape hatch.
#
# --- Indexing ---------------------------------------------------------------
# max_unit_lines / max_unit_chars / max_unit_tokens split oversized functions,
# methods, classes, and preambles into name#2, name#3, ... parts.
# unit_overlap_lines adds context to giant-statement line windows.
# max_chunk_chars caps source sent to the summariser.
# min_lines_to_summarise avoids spending a call on tiny units.
# summarise_methods / summarise_documented re-enable summaries the defaults
# skip to save time.
# include_references indexes .md/.json/.toml/.yaml/.sql/.sh and similar files
# structurally; extra_extensions adds more.
# semantic_search is opt-in and needs an embeddings-capable server.
# embedding_model empty means "use the chat model".
#
# --- Agent loop and tool opt-ins --------------------------------------------
# max_steps 0 means no lookup limit; token budgets are the intended bound.
# max_gathered_chars caps accumulated source per task.
# max_read_lines caps one `read` call.
# read_context_hint adds the enclosing signature and module imports to a
# mid-file read.
# use_project_instructions includes PEDACITO.md / AGENTS.md / CLAUDE.md.
# allow_write enables write/append/move/delete during gather.
# allow_shell enables run/diagnose; commands execute without a shell.
# allow_screenshots enables desktop capture; allow_browser enables headless
# page capture. Both are privacy/trust decisions and are off by default.
# allow_diff keeps the read-only `diff` action.
# allow_rename treats create-new + clear-old blocks as a real move.
# fuzzy_match anchors near-miss search blocks to the most similar text.
# compact_step_schema is experimental: drops unused fields from step replies.
# auto_diagnostics runs the detected linter/typechecker after each auto task.
# post_edit_check runs a cheap per-file lint/syntax check after every applied
# edit and feeds back only NEW findings (diffed against a pre-task baseline);
# it is on by default and bounded by repair_rounds.
# auto_allowlist restricts writes per task: an EXISTING file may only be
# edited/moved/deleted if the task title names it or the lookup phase read it;
# new files are always allowed. Off by default.
# allow_mcp exposes one `mcp` action and a short tool catalog; mcp_servers is
# a list of command strings such as ["python server.py"].
# auto_plan_file / auto_verify / auto_* configure `pedacito auto`.
#
# --- Review, workspace, and backups ----------------------------------------
# backup snapshots originals before writes; `pedacito restore` restores them.
# log_sessions writes .pedacito/sessions/*.md.
# review/review_port/open_browser/review_timeout control the side-by-side page.
# manage_gitignore/create_ignore_file control first-run workspace scaffolding.
#
# --- Profiles ---------------------------------------------------------------
# A profile overrides any setting in [universal], [indexing], or [agent].
# `tier` is a shortcut that applies a preset (tiny, small, medium, large,
# hosted) underneath the keys you write explicitly. Without `tier`, Pedacito
# guesses from the model name; set tier = "" to disable guessing.
'''


_KNOWN_SERVERS = (
    ("LM Studio", "http://localhost:1234/v1"),
    ("Ollama",    "http://localhost:11434/v1"),
)


def discover_servers(timeout: float = 2.0) -> list[tuple[str, str, list[str]]]:
    """Probe the known local servers; returns [(label, base_url, [model ids])]
    for each one answering GET <base_url>/models within `timeout`."""
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
    """Ask for server, model, hosted-or-local and VRAM, then write the starter
    template with those values in [profiles.default].

    An existing file is backed up to <path>.bak. The result is load()-checked
    before being written; raises ConfigError on bad answers."""
    out = stdout or sys.stdout
    def say(s=""):
        print(s, file=out)

    p = path or USER_CONFIG
    if p.exists() and not force:
        ans = _ask(f"{p} already exists. Overwrite it? [y/N]")
        if ans.lower() not in ("y", "yes"):
            raise ConfigError("setup cancelled; existing config left untouched")

    say("Pedacito setup -- three settings are mandatory: where your model")
    say("lives (base_url), its exact name (model), and the VRAM on the machine")
    say("running the server (vram_gb). Everything else has a working default.")
    say("Press Enter to accept any suggestion in [brackets].")
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

    def _gb(prompt: str, default: str) -> int:
        raw = _ask(prompt, default)
        try:
            value = int(raw)
        except ValueError:
            raise ConfigError(f"{prompt}: expected a whole number of GB, "
                              f"got {raw!r}")
        if value <= 0:
            raise ConfigError(f"{prompt}: must be greater than zero")
        return value

    vram_gb = _gb("GPU VRAM on the model server, in GB", "8")

    tier = "hosted" if hosted else _ask(
        "Model size tier -- tiny (3-4B), small (~9B), medium (~14-27B), "
        "large (35B+)",
        "medium")
    if tier not in PRESETS:
        raise ConfigError(f"tier must be one of {', '.join(PRESET_TIERS)}, got {tier!r}")

    lines = [
        'default_profile = "default"',
        "",
        "[profiles.default]",
        f"base_url = {_toml_literal(base_url)}",
        f"model    = {_toml_literal(model)}",
        f"tier     = {_toml_literal(tier)}",
    ]
    if api_key:
        lines.append(f"api_key  = {_toml_literal(api_key)}")
    if hosted:
        lines.append('json_mode = "object"   # hosted APIs lack llama.cpp schemas')
    body = TEMPLATE.replace(
        '''[profiles.default]
# The fallback profile. Replace these with your server and model.
base_url = "http://localhost:1234/v1"
model    = "google/gemma-3-27b"
''',
        "[profiles.default]\n" + "\n".join(lines[3:]) + "\n")
    body = body.replace("vram_gb = 8\n", f"vram_gb = {vram_gb}\n", 1)
    # Fallback if the template shape changes: still write a valid file.
    if body == TEMPLATE:
        body = "\n".join(lines) + f"\n\n[universal]\nvram_gb = {vram_gb}\n"

    # Validate before touching the user's file.
    probe = p.with_suffix(".toml.setup-probe") if p.parent.exists() else \
        Path("pedacito-setup-probe.toml")
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(body, encoding="utf-8")
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
    p.write_text(body, encoding="utf-8")
    say()
    say(f"Wrote {p}")
    say(f"  base_url = {base_url}")
    say(f"  model    = {model}")
    say(f"  vram_gb  = {vram_gb}")
    say(f"  tier     = {tier}" + ("  (api_key set, json_mode=object)" if api_key else ""))
    say("Test it any time with:  pedacito config --check")
    return p


def write_template(path: Path | None = None, force: bool = False) -> Path:
    """Write the starter template to `path` (default: USER_CONFIG); refuses to
    overwrite an existing file unless `force`."""
    p = path or USER_CONFIG
    if p.exists() and not force:
        raise ConfigError(f"{p} already exists; pass --force to overwrite")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(TEMPLATE, encoding="utf-8")
    return p
