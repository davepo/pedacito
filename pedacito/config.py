"""Configuration: the Config dataclass, TOML loading, and profile resolution.

Usage
-----
    from .config import load

    cfg = load(root=project_root, explicit=None, profile="qwen")
    print(cfg.base_url, cfg.model)

Settings come from a TOML file. The file is profile-based: `[universal]`,
`[indexing]`, and `[agent]` hold shared defaults, while `[profiles.NAME]`
holds server/model-specific keys. `[profiles.default]` is the fallback when no
profile is named. Precedence, lowest to highest:

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
from dataclasses import MISSING, dataclass, fields, field
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

    # --- Hardware (mandatory for local model servers) ------------------------
    # Total GPU VRAM on the machine running the model, in whole gigabytes.
    # Pedacito asks the server for the model's file size at the start of every
    # run, then estimates how much context (KV cache) still fits in VRAM next
    # to the weights, keeping 0.5 GB free for the system. 0 means "not
    # configured"; load() rejects that for local servers. Hosted APIs never
    # use this value.
    vram_gb: int = 0
    # Optional KV-cache hints for the VRAM estimate. Servers rarely expose the
    # cache quantization they actually loaded, but LM Studio, llama.cpp and
    # Ollama all let you choose it (LM Studio "K/V Cache Quantization Type",
    # llama.cpp --cache-type-k/-v, Ollama OLLAMA_KV_CACHE_TYPE). Matching the
    # values here stops the estimate from assuming a 16-bit cache and
    # under-reporting the context that really fits. Accepted spellings are
    # f16/fp16/bf16, f32/fp32, q8_0, q6_k, q5_1, q5_0, q4_1, q4_0, iq4_nl
    # and iq4_xs (case, dots and hyphens are normalised). Empty = 16-bit.
    k_cache_quant: str = ""
    v_cache_quant: str = ""
    # Precise escape hatch: measured KV-cache bytes per token of context.
    # 0 (the default) uses the model-size estimate above. Set this when you
    # know the real per-token cache size for the loaded architecture; it
    # bypasses both the proxy and the quantization adjustment.
    kv_cache_bytes_per_token: int = 0

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
    #     in practice fine on a frontier hosted model.  Accepted spellings:
    #     "object", "json_object", "json" (and "json_schema" -> "schema"),
    #     so the string from your API's docs works directly here.
    json_mode: str = "schema"
    # True when the server was recognised as a hosted API at load time --
    # either from its URL (api.deepseek.com etc.) or because the model name
    # contains "deepseek". Set automatically; not a config key. When set,
    # json_mode is forced to "object" unless the user spelled json_mode
    # explicitly in their config/profile/env.
    _hosted_server: bool = False

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
    # NOTE: hosted presets (and the DeepSeek default profile) override this
    # to ~2000. A step reply is action+args+reasoning under grammar, but on a
    # server that IGNORES response_format (DeepSeek in json_object mode among
    # them) the model free-forms around the JSON and routinely blows past a
    # 700-token budget before finishing the object. Keep any change here in
    # sync with PRESETS below -- tests assert the hosted value.
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
    # Retrieval-sized units: a function/method (or module preamble/trailing
    # block) larger than either limit is split at top-level statement
    # boundaries into name#2, name#3, ... parts. Without this, one 2,000-line
    # function either forces the model to read it whole (burning most of
    # max_gathered_chars) or is never read at all.
    max_unit_lines: int = 220
    max_unit_chars: int = 12000
    # Extra token-budget guard (rough 3.6 chars/token estimate), useful at
    # small context sizes where 12k chars is already a large fraction of the
    # window. 0 disables the token check.
    max_unit_tokens: int = 3000
    # When a single giant statement (a huge dict/list literal) has to fall back
    # to line windows, the next window repeats this many lines for context.
    # 0 keeps windows disjoint; statement-boundary splits never overlap.
    unit_overlap_lines: int = 0
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
    # Include the repository's PEDACEITO.md / AGENTS.md / CLAUDE.md in every
    # prompt. One short conventions file prevents small models from inventing
    # build commands and edit rules. Off only if the project wants no
    # automatic context.
    use_project_instructions: bool = True
    max_instruction_chars: int = 2000
    # Cap on the model's own checklist when it is injected into a prompt.
    max_todo_chars: int = 1500
    # Hard lookup limit for the gather phase. 0 (the default) means NO step
    # limit: the gathered-source/token budget and the model's `done` are the
    # only bounds. Set a positive value only when you explicitly want a
    # per-task wall. `--steps N` sets it for one command.
    max_steps: int = 0
    # Limit on follow-up answer turns when a model defers remaining work
    # ("the next turn continues"). 0 = no limit; the loop stops when the model
    # says NOTHING MORE or stops producing blocks. Set a positive value only
    # if you want a hard wall on that deferral loop.
    max_completion_rounds: int = 0
    # Limit on truncation-continuation turns when a reply is cut off mid-block.
    # 0 = no limit; the continuation char budget (proportional to
    # max_tokens_answer) is the real bound. A positive value is a hard wall.
    max_continuation_rounds: int = 0
    # Hard cap on how much source the gather phase may accumulate. A 27B's
    # usable reasoning window is well under its advertised context. This is
    # the intended constraint: token/context budget, not step count.
    max_gathered_chars: int = 24000
    max_read_lines: int = 200
    # Attach the enclosing signature and module imports when a `read` starts
    # mid-file. Both are already in the index; this saves a round trip to the
    # file head for small models without changing what is on disk.
    read_context_hint: bool = True
    # Run the project's detected linter/type checker after each `auto` task
    # (when no --verify command is configured) and feed failures into the
    # existing repair rounds. Off by default: a linter may report pre-existing
    # issues unrelated to the task; enable it for a project whose checker is
    # clean. The `diagnose` tool is separately available whenever
    # allow_shell is on.
    auto_diagnostics: bool = False
    diagnostics_timeout: int = 90
    max_diagnostics_chars: int = 8000
    # Optional MCP client. Off by default: arbitrary external tools are a large,
    # unpredictable surface for a small model. When on, one `mcp` action is
    # exposed and a short tool catalog is injected into the prompt.
    allow_mcp: bool = False
    mcp_servers: tuple[str, ...] = ()
    mcp_timeout: int = 30
    # Let the gather loop create/overwrite whole files via `write`, grow them
    # with `append`, and rename/remove them with `move`/`delete`. Off by
    # default: a mid-gather write skips the edit path's syntax check and
    # review page, so prefer search/replace blocks unless a model genuinely
    # cannot emit them.
    allow_write: bool = False
    # Ceiling on one `write` call's contents, in characters.
    max_write_chars: int = 200_000
    # Budget for ONE write step (gather-loop `write` action or answer-phase
    # edit block), expressed as the maximum number of LINES a single reply is
    # allowed to emit for one file. The default (120) is sized so that a step
    # fits inside deepseek-chat's real per-message output ceiling (~8k tokens,
    # which no config can raise): 120 lines + reasoning stays under it with
    # room to spare. Raise it only if your provider allows longer messages --
    # setting it above the provider's true limit just converts clean steps
    # into mid-string truncations again.
    max_write_lines: int = 120
    # Let the gather loop run ONE command per step (tests, linters, build
    # checks) and feed its output back. Off by default: this executes code in
    # the project, so enabling it is an explicit trust decision. Commands run
    # WITHOUT a shell (argv only), so pipes, redirection, and `&&` do not
    # work; use a small script if you need those.
    allow_shell: bool = False
    # Seconds before a `run` command is killed and its timeout reported.
    shell_timeout: int = 120
    # Stop a `run` command after this many seconds with NO output. GUI apps and
    # servers print a startup banner and then go quiet, so this is what keeps
    # them from blocking the agent until someone closes the window. Raise it if
    # a legitimate test/build is silent while loading dependencies; 0 disables
    # the idle rule (only shell_timeout remains).
    shell_idle_seconds: int = 10
    # Optional executable allowlist for `run` (e.g. ["pytest", "python",
    # "ruff"]). Empty means "any executable" once allow_shell is on.
    shell_allowlist: tuple[str, ...] = ()
    # Cap on command output fed back to the model, in characters.
    max_shell_output_chars: int = 8000
    # Let the gather loop capture the screen and turn it into text the model
    # can read (pixel statistics, OCR, optional vision-model description).
    # Off by default: this captures the whole desktop, which is a privacy
    # decision, and launching an app via `command` also needs allow_shell.
    allow_screenshots: bool = False
    # Seconds to wait after launching before capturing (the app needs time to
    # open a window). The action's `start` value overrides this when set.
    screenshot_wait_seconds: int = 5
    # Let the gather loop open a URL in a headless browser, optionally press
    # keys/click, wait, capture the page, and read its text. This is how a
    # model can verify a web game/server; a desktop screenshot cannot see a
    # page that was never opened. It can reach any http(s) URL, so treat this
    # as a trust decision like allow_shell. Needs the optional Playwright
    # install:
    #   pip install "pedacito-local[browser]"
    #   python -m playwright install chromium
    allow_browser: bool = False
    # Seconds to wait after navigation/interaction before capturing.
    browser_wait_seconds: int = 3
    # Seconds to wait for a launched server to print its localhost URL.
    browser_server_wait_seconds: int = 15
    # Run the browser without a visible window (normal for screenshots).
    browser_headless: bool = True
    # Per-navigation/action timeout in seconds.
    browser_timeout_seconds: int = 20
    # Optional multimodal model id (same server) used to describe a captured
    # screenshot. Empty = no image model; the tool still reports pixel
    # statistics and OCR text. Point this at e.g. a Qwen-VL model.
    vision_model: str = ""
    vision_max_tokens: int = 400
    # Optional semantic search over indexed units. When on, a `grep` whose
    # regex/literal/case-insensitive search finds nothing falls back to your
    # server's /v1/embeddings endpoint and reports the closest units by
    # meaning. Off by default: it needs an embedding model and costs one
    # embedding pass over the index the first time (cached afterwards).
    semantic_search: bool = False
    # Model id used for embeddings; empty means the chat model id. Point this
    # at a dedicated embedding model (e.g. nomic-embed-text) when your server
    # serves one alongside the chat model.
    embedding_model: str = ""
    semantic_top_k: int = 5
    # Cosine similarity below this is not shown. Embedding scales differ by
    # model, so tune this if results are too noisy or too sparse.
    semantic_min_score: float = 0.2
    # Read-only `diff` action for the gather loop: compare the working tree
    # against Pedacito's own pre-write snapshots (.pedacito/backups/) to see
    # what earlier writes changed. On by default -- it cannot damage anything
    # -- but it is one more action on the tool surface, so set false for
    # models that get confused by it.
    allow_diff: bool = True
    # EXPERIMENTAL: mark only `action` and `reasoning` required in the gather
    # step schema, so grammar-constrained decoding may omit the empty fields
    # a step does not use (~20-40 output tokens saved per step). Off by
    # default: all-required keeps the grammar as simple as possible for small
    # models -- measure it against your server before enabling.
    compact_step_schema: bool = False
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
    # Per-message OUTPUT ceiling the provider actually honours. Hosted APIs
    # cap one reply far below their advertised context window (deepseek-chat
    # and dashscope Qwen: ~8k tokens; asking for more just truncates the
    # reply mid-block). When set (>0) and the run targets a hosted API,
    # auto_fit clamps tier/default reply budgets to this value so a huge
    # budget can never turn a clean first block into a doomed truncated one.
    # Explicit max_tokens_* values win over an AUTO-DETECTED ceiling; a value
    # you set here yourself remains a deliberate hard cap.
    # 0 keeps the historical behaviour (trust the declared budgets).
    max_output_tokens: int = 0

    # --- Context window -------------------------------------------------------
    # The server-side context length you configured for the model (LM Studio's
    # "Context length", Ollama's num_ctx, an API's advertised window). Used by
    # `pedacito config` to sanity-check the token budgets below against it --
    # profiles set this per tier (8k for a small local model, 128k for hosted).
    # The default is deliberately conservative (matches LM Studio's own stock
    # setting), but you rarely need to set it at all: when a local server
    # answers, pedacito probes its REAL configured context length via the
    # native API and uses that instead of this number. It also queries the
    # model's file size and estimates the window from `vram_gb`, then
    # `auto_fit` sizes every other budget from the resulting window.
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
    # native API, Lemonade's extended /v1/models, and Ollama's /api/show all
    # report it). On
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
    # answer budget. The plan call must clear the file-tree JSON AND, on a
    # reasoning model, the <think> block that precedes it -- a budget sized
    # for plain output spends itself on thinking and never emits the JSON.
    # (Thinking is disabled per-call for these two, so the whole budget goes
    # to the actual output; still sized generously for whole files.)
    max_tokens_plan: int = 2000
    max_tokens_build: int = 8000

    # --- Autonomous loop (`pedacito auto`) ------------------------------------
    # The Ralph-loop outer loop: a goal is broken into a checklist, then each
    # task is implemented one at a time in a fresh context window, with the
    # plan file as the only surviving state. These are the loop's knobs.
    # Plan file name (relative to the project root) used when --plan is not
    # given. The file is plain markdown so you can edit it mid-run.
    auto_plan_file: str = "PLAN.md"
    # Shell command run after every task; a failure feeds the command's output
    # back to the model for repair. Empty = no verification, so tasks are
    # accepted whenever they apply cleanly.
    auto_verify: str = ""
    # Cap on how many tasks one `auto` run attempts; 0 = no cap (run until the
    # checklist is empty). A guardrail against an unbounded night of calls.
    auto_max_tasks: int = 0
    # Wall-clock cap on one `auto` run, in minutes; 0 = no cap (the default).
    # Checked between tasks, so a task already underway is allowed to finish.
    # Per-command limits (shell_timeout, auto_smoke_timeout) still apply.
    auto_max_minutes: int = 0
    # Fix rounds per failing verification before a task is given up on.
    auto_repair_rounds: int = 2
    # Per-file line guard for `new --complex` / `auto`: a source file whose
    # scaffold stub is longer than this many lines is split into several
    # implementation tasks (one per group of top-level definitions) so no
    # single reply has to emit a whole oversized file and get cut off at the
    # output token limit. 0 disables splitting.
    auto_split_lines: int = 120
    # When a --verify command fails because a third-party module is missing,
    # pip-install it into the running interpreter's environment and retry.
    # Off by default: the module name comes from model-generated code, so
    # installing it automatically is opt-in. With it off, the failure is
    # reported as a missing dependency (not a code bug) and the repair loop
    # is skipped.
    auto_install_deps: bool = False
    # Runtime verification at the end of a `--complex` build: besides the
    # import and call/signature checks, actually run the program once. With no
    # --verify given, a project with main.py is smoke-run (a process still
    # running at the timeout counts as started); otherwise the entry module is
    # imported. Failures are fed back to the model before the run completes.
    auto_smoke: bool = True
    auto_smoke_timeout: int = 20
    # Repair rounds for a failing runtime verification (each round: feed the
    # failure's output to the model, apply its edits, re-run the check).
    auto_verify_rounds: int = 2

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
    # Size of the configured model's weights in bytes, queried from the
    # server by auto_fit. Not a user setting.
    model_size_bytes: int = 0

    def __post_init__(self):
        """Initialise the mutable bookkeeping dicts (dataclasses cannot use
        mutable defaults directly)."""
        if self.sources is None:
            self.sources = {}
        if self.profiles is None:
            self.profiles = {}

    # Friendly aliases. The canonical fields carry their unit in the name so
    # a bare 16 can never be mistaken for megabytes; these let callers use the
    # shorter spelling without needing to remember the suffix.
    @property
    def vram(self) -> int:
        return self.vram_gb

    @vram.setter
    def vram(self, value: int) -> None:
        self.vram_gb = int(value)


# TOML tables are allowed but optional: [lmstudio] base_url = ... works, and so
# does a flat base_url = ... at the top level.
# "server" is the documented section name for connection settings; "lmstudio"
# is kept as an accepted alias so existing config files keep working. Pedacito
# talks to any server exposing an OpenAI-compatible /v1 surface (LM Studio,
# Ollama, and others), so "server" describes it without implying one product.
_SECTIONS = ("universal", "server", "lmstudio", "hardware", "generation",
             "map", "indexing", "files", "agent", "scaffold", "editing",
             "reasoning", "review", "workspace")
PROFILE_SECTION = "profiles"

_FIELDS = {f.name: f for f in fields(Config)
           if f.name not in ("sources", "profiles", "active_profile",
                             "model_size_bytes", "_hosted_server")}

# `vram = 16` is a natural spelling, so accept it as an alias for the
# canonical (unit-bearing) field name. The value is always whole gigabytes;
# the suffix exists to make that impossible to misread.
_ALIASES = {
    "vram": "vram_gb",
    "vram_gb": "vram_gb",
    # Common spellings from the runtime UIs / llama.cpp flags.
    "k_cache_type": "k_cache_quant",
    "v_cache_type": "v_cache_quant",
    "k_cache_quantization": "k_cache_quant",
    "v_cache_quantization": "v_cache_quant",
    "kv_cache_k_quant": "k_cache_quant",
    "kv_cache_v_quant": "v_cache_quant",
}

# Effective bits per cache element, including the per-block scale/min overhead
# llama.cpp's quant blocks carry. Only the relative size against f16 matters,
# so approximate values are fine -- they still turn a 4x-wrong estimate into a
# useful one.
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
    "fp16": "f16", "float16": "f16", "half": "f16",
    "fp32": "f32", "float32": "f32",
    "bfloat16": "bf16",
}


def _normalise_kv_quant(value) -> str:
    """Canonicalise a cache-quantization spelling, or raise ConfigError."""
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw or raw in ("auto", "default", "none", "false"):
        return ""
    # LM Studio / llama.cpp also write Q8_0, q5.1, "Q4 0", etc.
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
    """Effective bits per cache element for a normalized quantization name."""
    name = _normalise_kv_quant(value)
    return _KV_QUANT_BITS.get(name) if name else None


def _canonical(name: str) -> str:
    """Resolve a config key spelling to its canonical Config field name."""
    return _ALIASES.get(name, name)

# Accepted spellings for json_mode. DeepSeek's own docs put the literal
# string "json_object" inside response_format, so users reasonably write it
# here; mapping every variant to the two internal modes removes a whole
# class of "config only accepts object or schema" dead ends.
_JSON_MODE_ALIASES = {
    "schema": "schema", "json_schema": "schema",
    "object": "object", "json_object": "object", "json": "object",
}


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
        bad = [k for k in body
               if _canonical(k) not in _FIELDS and k != "tier"]
        if bad:
            raise ConfigError(f"{path}: [profiles.{name}] unknown setting(s): "
                              f"{', '.join(bad)}")
        profiles[name] = {_canonical(k): v for k, v in body.items()}
    for key, value in data.items():
        if isinstance(value, dict):
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
    """Convert a raw TOML value to the type its Config field expects,
    forgiving int-for-float and list-for-tuple, strict about everything else."""
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
        # Accept every spelling of the two modes, including "json_object" --
        # the literal string DeepSeek's docs put in response_format. Users
        # reasonably try it here; rejecting it only produces confusion.
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
         use_env: bool = True, profile: str | None = None,
         require_hardware: bool = True, warn_budgets: bool = True) -> Config:
    """Build a Config by layering every source in precedence order: defaults,
    user config, project config, an explicit --config file, a selected
    profile, then PEDACITO_* environment variables. Each field records which
    layer set it, in `cfg.sources`, for `pedacito config` to display.

    `vram_gb` is mandatory for local servers: without it auto-fit cannot know
    how much context fits beside the model. The `config` subcommand passes
    require_hardware=False so it can still inspect (and --init/--setup can
    repair) an incomplete file. Hosted API configs never need it -- there is
    no local GPU to measure."""
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
    if not profile and "default" in profiles:
        # The profile-based config format makes [profiles.default] the
        # fallback when no profile is named.
        profile = "default"
        origin = "implicit [profiles.default]"
    if profile:
        if profile not in profiles:
            known = ", ".join(sorted(profiles)) or "(none defined)"
            # Say WHERE the profiles were looked for. XDG_CONFIG_HOME can
            # silently redirect the user config (flatpak shells, IDE
            # terminals), and "(none defined)" without the paths sends the
            # user hunting through a config.toml that was never opened.
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
            # Mark preset-derived values so auto_fit may replace them with a
            # hardware-sized budget. A key the user spelled in the profile is
            # an explicit limit and is never grown past.
            how = "explicit" if key in explicit else "tier preset"
            cfg.sources[key] = f"profile '{profile}' (via {origin}; {how})"
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

    # Hosted-API autodetect: DeepSeek (by URL or model name) cannot accept
    # response_format json_schema -- it 400s or silently ignores it, which
    # surfaces as "would not produce valid JSON". Users should never have to
    # know that. If we can tell the target is a hosted API and the user did
    # not spell json_mode anywhere, switch to "object" ourselves and say so.
    try:
        from urllib.parse import urlparse
        host = (urlparse(cfg.base_url).hostname or "").lower()
    except ValueError:
        host = ""
    remote = host not in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "")
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

    # Provider per-message OUTPUT ceiling autodetect. Some hosted APIs
    # advertise a 64k-128k CONTEXT window but cut a single *thinking* reply
    # off far earlier -- dashscope Qwen's thinking variants exhaust ~8k
    # before emitting JSON ("cut off" / "failure to return json"). For those
    # models, when max_output_tokens was not set explicitly, default it to
    # the conservative 8k they honour; auto_fit then clamps tier/default
    # reply budgets to it (the continuation loop in agent.py recovers
    # whatever blocks survive). Explicit max_tokens_* values are left alone:
    # the user's instruction outranks this heuristic.
    #
    # Deliberately NOT applied to plain deepseek-chat / generic remote hosts:
    # their documented message ceiling is far larger than 8k, the shipped
    # hosted preset budgets are already sized for them, and clamping here
    # would silently shrink answers for users who chose tier="hosted"
    # precisely to opt out of artificial limits. If a provider does truncate
    # at N tokens, set `max_output_tokens = N` in the config -- an explicit
    # value is honoured as a hard cap by auto_fit even when auto_fit is off.
    qwen_thinking = any(t in cfg.model.lower() for t in
                        ("qwq", "qwen3", "qwen-3", "thinking"))
    if (qwen_thinking or deepseek) and cfg.json_mode == "schema":
        # Same failure class from the other side: thinking models spend the
        # whole reply budget on reasoning text and never reach the JSON. A
        # schema response forces structured output immediately. Applies only
        # when the user did not spell json_mode (source stays "default"), so
        # an explicit choice is never overridden.
        if cfg.sources.get("json_mode") == "default":
            cfg.json_mode = "schema"
            cfg.sources["json_mode"] = (
                f"auto-selected for thinking model ({cfg.model}); set "
                "json_mode explicitly to override")
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

    # Loud-but-not-fatal: budgets that cannot fit the declared context window
    # are how truncation errors reach the user, so surface them at load time.
    # Help (`pedacito how`) opts out: it uses its own excerpt budget and can
    # run before VRAM/auto-fit has resized the coding budgets.
    if warn_budgets:
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
    if getattr(cfg, "_hosted_server", False):
        return True
    if cfg.json_mode == "object":
        return True
    try:
        from urllib.parse import urlparse
        host = (urlparse(cfg.base_url).hostname or "").lower()
    except ValueError:
        return False
    return host not in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "")


_GIB = 1024 ** 3
# KV-cache estimation constant: at roughly this many tokens of context, the
# cache is about the same size as the model's weights on disk. That is a good
# fit for the common llama/gemma/qwen GGUF families (a 27B Q4 is ~16 GB at
# 32k, an 8B Q4 ~4-5 GB), and it scales sensibly for larger models. It is an
# estimate, not a tokenizer: enough to keep auto-fit from planning a context
# the hardware cannot hold, conservative enough that being wrong is cheap.
_KV_REFERENCE_TOKENS = 32768


def missing_hardware(cfg: Config) -> list[str]:
    """Names of mandatory hardware settings still unset for a LOCAL server.

    Hosted APIs return an empty list: their context is a provider-side budget,
    not local GPU memory, so there is nothing to estimate. A remote URL is
    treated as hosted by `is_hosted` (that is the existing convention), so
    only localhost-style servers require the value.
    """
    if is_hosted(cfg):
        return []
    return [name for name in ("vram_gb",)
            if int(getattr(cfg, name, 0) or 0) <= 0]


def hardware_context_tokens(vram_gb: int, model_size_bytes: int,
                            k_cache_quant: str = "",
                            v_cache_quant: str = "",
                            kv_cache_bytes_per_token: int = 0) -> int:
    """Estimate how many context tokens fit in VRAM after loading a model of
    `model_size_bytes`.

    Only the model weights and the KV cache live in VRAM; system RAM is not
    counted, because spilling into it is slow and is exactly what this
    estimate must avoid. Half a gigabyte is kept for the driver/desktop, then
    the rest is available for KV cache. The cache size per token is derived
    from the file size (see _KV_REFERENCE_TOKENS), a proxy for parameter
    count that needs no architecture metadata and is stable across
    quantizations.

    `k_cache_quant` / `v_cache_quant` scale that f16-sized baseline by the
    effective bits of the quantization the runtime actually loaded (LM
    Studio's K/V cache types, llama.cpp's --cache-type-k/-v). A precise
    `kv_cache_bytes_per_token` bypasses both the proxy and that scaling.

    Returns the estimated token count. 0 or a negative free-memory result
    means the model does not fit at all; callers treat that as "use the
    fallback budgets", not as a hard failure.
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
            # The baseline is an f16 cache (16 bits for each of K and V).
            # Unset sides stay at f16 so setting only K still improves the
            # estimate without pretending V was quantized too.
            k_bits = 16.0 if k_bits is None else k_bits
            v_bits = 16.0 if v_bits is None else v_bits
            kv_per_token = max(
                4 * 1024, int(kv_per_token * (k_bits + v_bits) / 32.0))
    return int(free // kv_per_token)


def _is_explicit_limit(source: str) -> bool:
    """True when a budget came from an explicit user choice.

    Tier presets, built-in defaults, and values auto-fit itself set earlier
    are fallbacks: hardware-derived budgets may replace them. Anything the
    user wrote in a config file/profile/env/CLI is a deliberate limit and is
    only respected when it is SMALLER than the hardware target.
    """
    if not source or source == "default":
        return False
    if "preset" in source:
        return False
    if source.startswith("auto-fit") or source.startswith("auto-detected"):
        return False
    return True


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
    API, Lemonade /v1/models, Ollama /api/show) and its REAL configured
    length is used. Probe
    failures are silent -- hosted APIs and offline runs keep the declared
    number.

    For local servers the mandatory `vram_gb` setting and the model file size
    reported by the server feed a KV-cache estimate. When the server reports
    its CONFIGURED context length (LM Studio, Lemonade, Ollama), that is
    authoritative -- the estimate only warns when it is below it (the KV
    cache will spill to system RAM). When the server does not report one, the
    estimate caps the fallback window so weights plus KV cache keep fitting
    in VRAM with 0.5 GB kept for the system. Tier/default token
    budgets are used when the whole set fits that window; otherwise the
    VRAM-derived allocation is used. Explicit user values override both.
    Returns human-readable notes about what changed (empty = no changes
    needed). Raises ConfigError if `vram_gb` is missing for a local server.
    """
    import sys as _sys
    notes: list[str] = []

    # max_output_tokens is a HARD provider cap, honoured even when auto_fit
    # is off -- asking the server for more than it will ever emit only
    # guarantees a truncated reply. Clamp first; the window-fitting logic
    # below then sees budgets that already fit the provider's ceiling.
    #
    # Exception: an AUTO-DETECTED cap must not rewrite values the user wrote
    # explicitly. The detection heuristic (for example "hosted Qwen thinking
    # models emit ~8k tokens per message") is a guess; an explicit
    # max_tokens_answer/build/plan/step is a deliberate instruction. Explicit
    # values still remain bounded by the server's real context window and by
    # a max_output_tokens the user set themselves.
    mot0 = int(getattr(cfg, "max_output_tokens", 0) or 0)
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
                  + "; ".join(notes), file=_sys.stderr)
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
    if not hosted:
        missing = missing_hardware(cfg)
        if missing:
            raise ConfigError(
                "auto-fit needs vram_gb for local model servers; missing: "
                + ", ".join(missing) + ".\n"
                "Set it in whole gigabytes of VRAM on the machine running "
                "the model, or run `pedacito config --setup`.")
        info: dict = {}
        # A pre-set size (a test harness, or a caller that probed earlier)
        # saves a round trip and lets the estimate run without a live server.
        preset_size = int(getattr(cfg, "model_size_bytes", 0) or 0)
        need_probe = preset_size <= 0 or (
            cfg.auto_detect_context and origin == "default")
        if need_probe:
            try:
                from .llm import LMStudio
                info = LMStudio(cfg).model_info() or {}
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
        # Only the CONFIGURED/loaded length is safe to fit budgets to. The
        # model's advertised maximum (max_context_length) says nothing about
        # what the server was told to allocate; using it as the window is how
        # a stock 8k LM Studio setup gets 100k-token requests and 400s.
        detected = info.get("context_length")
        hw_ctx = (hardware_context_tokens(
                      cfg.vram_gb, model_bytes,
                      getattr(cfg, "k_cache_quant", ""),
                      getattr(cfg, "v_cache_quant", ""),
                      int(getattr(cfg, "kv_cache_bytes_per_token", 0) or 0))
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

        # VRAM-only estimate. System RAM is deliberately NOT counted: spilling
        # KV cache there is slow, and counting it produced context plans that
        # did not actually fit on the GPU. Keep 0.5 GB for the driver/desktop.
        # The estimate CAPS the context window only when neither the server
        # nor the user supplied a configured length. Both of those are facts;
        # the estimate is a heuristic with no knowledge of the model's KV
        # architecture (GQA) or cache quantization, and is routinely off by
        # 2-4x. A wrong clamp converted working 262k-token setups into 12k
        # windows whose budgets no thinking model could emit JSON inside.
        # When the configured length is below the estimate it stays a speed
        # advisory: KV cache spills to system RAM, which is slow but works.
        if model_bytes > 0:
            size_gb = model_bytes / _GIB
            mem = f"{cfg.vram_gb} GB VRAM"
            kv_hint = ""
            if getattr(cfg, "kv_cache_bytes_per_token", 0):
                kv_hint = (f"; KV cache assumed "
                           f"{int(cfg.kv_cache_bytes_per_token) / 1024:.0f} "
                           f"KiB/token")
            elif getattr(cfg, "k_cache_quant", "") or \
                    getattr(cfg, "v_cache_quant", ""):
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
        # Respect explicit user settings within reason: only clamp values
        # that physically cannot work (bigger than the whole window).
        # Tier/default values are fallbacks and may always be re-fitted.
        if _is_explicit_limit(src) and value < cur:
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

    mot = int(getattr(cfg, "max_output_tokens", 0) or 0)

    # Hardware-derived budgets. Tier/default values are used when the whole
    # set fits the VRAM-capped window; if it does not, the calculated
    # allocation wins so a default answer budget cannot squeeze gathered
    # source down to nothing. Explicit user values override both (they are
    # deliberate limits), though the window's physical cap still applies so
    # a request cannot exceed the server's context.
    if not hosted and model_bytes > 0:
        # Calculate from the authoritative window when the server reported
        # one or the user configured it: the VRAM estimate must not shrink a
        # window that actually works. Only when both are absent does the
        # estimate cap the fallback window.
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

        # Tier/default floors are only safe when the resulting answer prompt
        # actually fits the VRAM window. If the whole set does not fit, use
        # the calculated allocation instead of squeezing gathered source down
        # to nothing (which is what made reads useless on small VRAM windows).
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

        # Leave room for map + gathered source in the answer prompt. Using
        # usable - answer guarantees the answer() fit check passes with half
        # the reserve still free.
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
        # No model size (old/other server) or hosted API: keep the existing
        # tier/default values and only clamp what cannot physically fit.
        # Hosted runs are grow-only, so this branch never shrinks them.
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

        # Provider per-message OUTPUT ceiling. Hosted APIs advertise a huge
        # context window but cut a single reply off far earlier (~8k tokens on
        # deepseek-chat and dashscope Qwen). Asking for more than that never
        # yields a longer answer -- it truncates one mid-JSON/mid-block, which
        # is exactly the "son cut off / failure to return json" failure. When
        # max_output_tokens is set, clamp tier/default reply budgets to it.
        # An explicit max_tokens_* value still wins over an auto-detected
        # ceiling; setv() only enforces this shrink for non-explicit budgets
        # (or when max_output_tokens itself was set by the user).
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
        value = target(raw) if target in (int, float) else raw
    except ValueError as e:
        raise ConfigError(f"PEDACITO_{name.upper()}={raw!r} is not a valid {target.__name__}") from e
    if name == "vram_gb" and value <= 0:
        raise ConfigError(
            f"PEDACITO_{name.upper()}={raw!r} must be a positive whole "
            f"number of gigabytes")
    return value


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
        # max_steps is 0 (no lookup limit): the token budgets below are the
        # real constraint. Set a positive value only when you explicitly want
        # a per-task wall.
        #
        # These values are a FALLBACK. When the server reports the model's
        # file size, auto-fit derives larger budgets from VRAM (see
        # config.auto_fit). They matter for servers that do not expose a
        # size, and for the settings auto-fit does not generate (max_steps,
        # max_read_lines, max_spec_chars, max_new_files).
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
        # ~14B-32B local models. max_tokens_step is deliberately generous:
        # a THINKING model spends most of the step budget on hidden reasoning
        # before emitting the JSON action, and at 700 that leaves nothing --
        # finish_reason=length, no action, "didn't return an answer".
        #
        # Fallback budgets for a 32k window. auto-fit replaces these with
        # hardware-sized values on a server that reports the model size.
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
        # Big local rig: every reply/source budget is at least as large as
        # medium's. The old preset had a SMALLER step and build budget than
        # medium, which is exactly backwards for a model with a roomier
        # window and deeper reasoning.
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
        # API pricing tolerates generous windows; answers and build passes
        # get real headroom instead of truncation retries.
        #
        # Hosted tiers are NOT auto-fit: the window here is a declared
        # budget, not a server limit (auto_fit skips local probing for
        # hosted profiles), so the budgets below only ever grow -- they
        # never clamp anything the user set. There is deliberately no
        # hardware-derived cap: the provider's own limits decide.
        # A model writing four 800-line files in one answer needs ~40k output
        # tokens; deepseek-chat's real ceiling is 8k, which no config can
        # raise, but the budget must at least stop fighting the model within
        # its true limits. max_tokens_step stays below ~8k for the same
        # reason: the provider's per-message output ceiling cannot be
        # exceeded by any client setting, and asking for more only converts
        # clean steps into mid-string truncations. Big outputs are handled
        # by chunking: see max_write_lines and the gather/answer prompts.
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
    elif name == "default":
        # [profiles.default] is the fallback overlay, not a model-sized tier:
        # its keys sit on top of [universal]/[indexing]/[agent]. Guessing
        # "large" from "27b" would silently replace those universal budgets.
        tier = ""
    else:
        tier = _guess_tier(name, body) or ""
    if not tier:
        return body, ""
    merged = dict(PRESETS[tier])
    merged.update(body)
    return merged, tier


def _toml_literal(value) -> str:
    """Render a Config default as a TOML literal for the commented reference."""
    import json
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
        elif f.default_factory is not MISSING:      # type: ignore[comparison-overlap]
            value = f.default_factory()
        else:
            continue
        out.append(f"# {name} = {_toml_literal(value)}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# `pedacito config --init` starter file
# --------------------------------------------------------------------------
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

# Diagnostics and MCP are opt-in. MCP also needs mcp_servers configured.
auto_diagnostics = false
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
# tier = "medium"               # small | medium | large | hosted
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


# --------------------------------------------------------------------------
# Full annotated reference. `--init` writes the complete starter above;
# `--reference` appends the longer explanations so every knob stays documented.
# --------------------------------------------------------------------------
REFERENCE_TEMPLATE = TEMPLATE + '''

# ===========================================================================
# DETAILED NOTES
# ===========================================================================
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
# use_project_instructions includes PEDACEITO.md / AGENTS.md / CLAUDE.md.
# allow_write enables write/append/move/delete during gather.
# allow_shell enables run/diagnose; commands execute without a shell.
# allow_screenshots enables desktop capture; allow_browser enables headless
# page capture. Both are privacy/trust decisions and are off by default.
# allow_diff keeps the read-only `diff` action.
# allow_rename treats create-new + clear-old blocks as a real move.
# compact_step_schema is experimental: drops unused fields from step replies.
# auto_diagnostics runs the detected linter/typechecker after each auto task.
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
# `tier` is a shortcut that applies a preset (small, medium, large, hosted)
# underneath the keys you write explicitly. Without `tier`, Pedacito guesses
# from the model name; set tier = "" to disable guessing.
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

    Asks where the model lives, which model, whether it is a hosted API, and
    the server machine's VRAM, then writes the profile-based starter with your
    values in [profiles.default]: the universal/indexing/agent sections, every
    commented profile override, and the local/hosted examples stay as
    reference. Existing files are backed up to <path>.bak rather than silently
    overwritten. Raises ConfigError on bad answers; the written file is
    load()-checked before returning."""
    import sys
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

    # Mandatory for local servers; harmless for hosted profiles and useful if
    # the same file later gains a local profile.
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
        "Model size tier -- small (~9B), medium (~14-27B), large (35B+)",
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
    # VRAM is a shared hardware setting now: update [universal], not the
    # profile. The exact active line only exists once in the template.
    body = body.replace("vram_gb = 8\n", f"vram_gb = {vram_gb}\n", 1)
    # If the template shape ever changes, still write a valid profile-based
    # file rather than silently leaving the generic defaults in place.
    if body == TEMPLATE:
        body = "\n".join(lines) + f"\n\n[universal]\nvram_gb = {vram_gb}\n"

    # Validate BEFORE touching the user's file: parse exactly as load() will.
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
    """Write the starter config template to `path` (default: USER_CONFIG).
    Refuses to overwrite an existing file unless `force` is set."""
    p = path or USER_CONFIG
    if p.exists() and not force:
        raise ConfigError(f"{p} already exists; pass --force to overwrite")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(TEMPLATE, encoding="utf-8")
    return p
