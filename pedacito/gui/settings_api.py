"""The GUI's settings endpoints: read, edit, and write Pedacito's config files.

Thin glue over `settings.py` (pure text patching) -- this module owns the file
system: which file is meant, backups, atomic writes, secrets, and the live
server probe used by "test connection".
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import requests

from .. import config as config_mod
from .. import config_editor
from ..config import USER_CONFIG, ConfigError
from . import settings
from .core import STATE

SECRET_MASK_KEYS = settings.SECRET_KEYS


def config_path(which: str) -> Path:
    """The file an edit targets: the user config or the open project's."""
    if which == "user":
        return USER_CONFIG
    if which == "project":
        root = STATE.root
        if root is None:
            raise ConfigError("Open a project to edit its config.")
        return config_mod.find_project_config(root) or root / ".pedacito.toml"
    raise ConfigError(f"unknown config file {which!r}")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def write_config(path: Path, text: str) -> None:
    """Validate, back up the old file as `<name>.bak`, and write atomically."""
    settings.validate_text(text, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            Path(str(path) + ".bak").write_bytes(path.read_bytes())
        except OSError:
            pass
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


def _mask(body: dict) -> dict:
    out = dict(body)
    for key in SECRET_MASK_KEYS:
        if key in out:
            out[key] = ""
            out.setdefault("_secret_set", [])
            out["_secret_set"] = [*out["_secret_set"], key]
    return out


def file_info(which: str) -> dict:
    """One config file's state for the editor."""
    try:
        path = config_path(which)
    except ConfigError as e:
        return {"which": which, "available": False, "error": str(e)}
    text = _read_text(path)
    info = {"which": which, "available": True, "path": str(path),
            "exists": path.exists(), "text": text, "profiles": {},
            "settings": {}, "error": ""}
    if text.strip():
        try:
            flat, profiles = settings.validate_text(text, str(path))
            info["settings"] = {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in flat.items()
                                if k not in SECRET_MASK_KEYS}
            info["settings_secret_set"] = [k for k in flat
                                           if k in SECRET_MASK_KEYS]
            info["profiles"] = {n: _mask(b) for n, b in profiles.items()}
        except ConfigError as e:
            info["error"] = str(e)
    if which == "project" and path.exists():
        info["trusted"] = config_mod.is_trusted(path)
        info["exec_fields"] = list(config_mod.EXEC_FIELDS)
    return info


def payload() -> dict:
    cfg = STATE.cfg
    return {
        "schema": settings.schema(),
        # Universal settings must always carry a concrete effective value:
        # the open project's resolved one when there is a config, else the
        # built-in `Config()` defaults (a GUI session can start without a
        # project open, and the user file may not exist yet).
        "resolved": settings.resolved(cfg if cfg is not None
                                      else config_mod.Config()),
        "active_profile": cfg.active_profile if cfg is not None else "",
        "tiers": list(config_mod.PRESET_TIERS),
        "user": file_info("user"),
        "project": file_info("project"),
        "user_config_path": str(USER_CONFIG),
        "project_open": STATE.root is not None,
    }


def _changes_clean(changes: dict) -> dict:
    """Drop an empty secret (the form leaves it blank to mean 'unchanged')."""
    return {k: v for k, v in changes.items()
            if not (k in SECRET_MASK_KEYS and v in ("", None))}


def apply_changes(which: str, profile: str | None, changes: dict,
                  removes: list[str]) -> dict:
    path = config_path(which)
    text = _read_text(path)
    if not text and not path.exists() and which == "user":
        raise ConfigError("No user config file yet. Use \u201cCreate config\u201d "
                          "first.")
    new = settings.set_values(text, _changes_clean(changes), removes,
                              profile=profile or None)
    write_config(path, new)
    return payload()


def profile_action(which: str, action: str, name: str, values: dict,
                   source: str = "") -> dict:
    path = config_path(which)
    text = _read_text(path)
    if action == "create":
        new = settings.add_profile(text, name, _changes_clean(values))
    elif action == "delete":
        new = settings.remove_profile(text, name)
    elif action == "duplicate":
        body = settings.profile_body(text, source)
        if not body and source not in settings.profile_names(text):
            raise ConfigError(f"no profile named '{source}'")
        new = settings.add_profile(text, name, body)
    else:
        raise ConfigError(f"unknown profile action {action!r}")
    write_config(path, new)
    return payload()


def rename_profile(which: str, old_name: str, new_name: str) -> dict:
    """Rename `[profiles.OLD]` to `[profiles.NEW]` in place (comments and
    every other key survive), retarget `default_profile` and `roles`
    references, and reload STATE so the header dropdown and the settings
    payload show the new name at once."""
    old_name = str(old_name or "").strip()
    raw_new = str(new_name or "").strip()
    new_name = _sanitize_profile_name(raw_new)
    # version dots never fail a rename either (`qwen3.8` -> `qwen3_8`), but a
    # truly wrong human name (an org `path/`) is refused, like the import.
    if not new_name or not PROFILE_NAME_RE.fullmatch(new_name):
        raise ConfigError(
            f"profile name {raw_new!r} must use letters, digits, '-' and "
            "'_' (starting with a letter or digit, max "
            f"{_MAX_PROFILE_NAME} characters)")
    path = config_path(which)
    text = _read_text(path)
    known = config_editor.profile_names(text)
    if old_name not in known:
        raise ConfigError(f"no profile named '{old_name}' in this file")
    if new_name == old_name:
        raise ConfigError(f"the profile is already named '{old_name}'")
    if new_name in known:
        raise ConfigError(f"a profile named '{new_name}' already exists")
    write_config(path, config_editor.rename_profile(text, old_name, new_name))
    # Keep the selection following the renamed profile: an explicit
    # user-chosen profile (STATE.profile_request) or the active one.
    with STATE.lock:
        requested = STATE.profile_request
        active = STATE.cfg.active_profile if STATE.cfg is not None else ""
    out = reload_state_config(profile=new_name
                              if requested == old_name else None)
    if out["active_profile"] == "" and active == old_name:
        out = reload_state_config(profile=new_name)
    out["renamed"] = {"from": old_name, "to": new_name}
    out["profiles"] = sorted(set((out["user"].get("profiles") or {}))
                             | set((out["project"].get("profiles") or {})))
    return out


def save_raw(which: str, text: str) -> dict:
    write_config(config_path(which), text)
    return payload()


def create_user_config(base_url: str, model: str, vram_gb) -> dict:
    if USER_CONFIG.exists():
        raise ConfigError(f"{USER_CONFIG} already exists.")
    if not base_url or not model:
        raise ConfigError("Give the server URL and the model name.")
    try:
        vram = int(vram_gb) if vram_gb not in (None, "") else None
    except (TypeError, ValueError):
        raise ConfigError("vram_gb must be a whole number of gigabytes.")
    write_config(USER_CONFIG, settings.new_user_config(base_url, model, vram))
    return payload()


def trust_project_config() -> dict:
    path = config_path("project")
    if not path.exists():
        raise ConfigError("This project has no config file to trust.")
    config_mod.trust_project(path)
    return payload()


def _stored_api_key(which: str, profile: str) -> str:
    try:
        text = _read_text(config_path(which))
        body = settings.profile_body(text, profile) if profile else {}
        if body.get("api_key"):
            return str(body["api_key"])
        flat, _ = settings.validate_text(text)
        return str(flat.get("api_key", ""))
    except (ConfigError, OSError):
        return ""


def probe_server(base_url: str, api_key: str, which: str = "user",
                 profile: str = "") -> dict:
    """List the models a server reports (the "test connection" button)."""
    base_url = (base_url or "").strip()
    if not base_url:
        return {"ok": False, "error": "Enter the server URL first."}
    if not api_key:
        api_key = _stored_api_key(which, profile) or (
            STATE.cfg.api_key if STATE.cfg is not None else "")
    try:
        r = requests.get(base_url.rstrip("/") + "/models",
                         headers={"Authorization": f"Bearer {api_key}"},
                         timeout=8)
        r.raise_for_status()
        ids = sorted({m.get("id", "?") for m in r.json().get("data", [])})
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"ok": True, "models": ids}


def list_profiles() -> dict:
    """The profiles table (identity + measured scores + recommendations) for
    the settings page, from the same renderer the CLI uses."""
    from .. import profiles as profiles_mod
    cfg = STATE.cfg if STATE.cfg is not None else config_mod.Config()
    rows = profiles_mod.list_profiles(cfg)
    return {"profiles": rows, "recommendations": profiles_mod.recommend(rows)}


def scan_servers() -> dict:
    """Discover servers and list each one's models with metadata (loaded,
    size, context, embedding/too-big already dropped)."""
    from .. import discovery
    servers = []
    seen: set[str] = set()
    for label, base_url, _ids in discovery.discover_servers(timeout=1.5):
        key = _norm_url(base_url)
        if key in seen:
            continue          # the same server spelled two ways: probe once
        seen.add(key)
        servers.append({"label": label, "base_url": base_url,
                        "models": discovery.describe_models(base_url)})
    return {"servers": servers}


# --- reset to baseline defaults ------------------------------------------------

# The sections of the baseline file: (section, blurb, (key, extra-comment)).
# Values always come from the built-in Config() defaults -- the recommended
# pre-configuration for small local models (reasoning-first steps, the brief
# edit plan, the incremental edit loop, tight read/gather budgets).
_BASELINE_SECTIONS = (
    ("universal", "Connection, generation, and shared defaults.", (
        ("base_url", "the model server; import real servers from Settings \u2192 Model Discovery & Profiles"),
        ("model", "the model id exactly as the server reports it"),
        ("json_mode", '"schema" for llama.cpp-based servers, "object" for hosted APIs'),
        ("tool_mode", '"grammar" (schema-constrained steps) or "native" (the model\'s tool calls)'),
        ("temperature", None),
        ("temperature_answer", "greedy answers, so SEARCH text is copied exactly"),
        ("timeout", None),
        ("load_wait_seconds", None),
        ("context_tokens", "fallback window; the server\'s real length wins with auto_detect_context"),
        ("reserve_tokens", None),
        ("auto_fit", None),
        ("auto_detect_context", None),
    )),
    ("agent", "Agent-loop budgets, tuned for small local models.", (
        ("max_steps", "0 = no lookup cap; the budgets below and the model\'s `done` bound it"),
        ("max_total_model_calls", "hard ceiling on model calls per task"),
        ("max_no_progress_steps", "ends gather after this many fruitless steps"),
        ("max_read_tokens", "token budget for one read during gather"),
        ("max_read_lines", None),
        ("max_gathered_chars", None),
        ("read_skeleton", None),
        ("max_rehydration_rounds", None),
        ("step_reasoning_first", "state the reason before picking the action"),
        ("edit_plan_brief", "1-3 lines naming the file, the lines, and why"),
        ("incremental_edit_loop", "one symbol per verified micro-loop"),
        ("repair_rounds", None),
        ("post_edit_check", None),
        ("auto_repair_rounds", None),
        ("auto_smoke", None),
        ("use_model_card", "let `pedacito calibrate` pick the remaining defaults"),
    )),
    ("editing", "Editing and workspace behaviour.", (
        ("backup", None),
        ("log_sessions", None),
        ("allow_git", None),
        ("manage_gitignore", None),
        ("create_ignore_file", None),
    )),
)


def baseline_text() -> str:
    """The clean baseline config.toml text: recommended Config() defaults,
    no model profiles and no per-role assignments."""
    cfg = config_mod.Config()
    lines = [
        "# Pedacito baseline configuration.",
        "#",
        "# \"Reset to Baseline\" restored the recommended defaults for small",
        "# local models and removed every saved profile and per-role model",
        "# assignment. The previous file was kept next to this one as a",
        "# config.toml.reset.<timestamp>.bak copy.",
        "#",
        "# Next step: open Settings \u2192 Model Discovery & Profiles and scan your",
        "# local servers (or run `pedacito models scan`), then import the models",
        "# you use. Every setting is documented in `pedacito config --reference`.",
    ]
    for section, blurb, keys in _BASELINE_SECTIONS:
        lines.append("")
        lines.append(f"[{section}]")
        lines.append(f"# {blurb}")
        if section == "universal":
            lines.append("vram_gb = 8   # whole GB of VRAM on the model machine;"
                         " auto_fit sizes the window from it (hosted APIs ignore it)")
        for key, note in keys:
            value = getattr(cfg, key)
            if value is None:
                continue  # TOML has no null: unset lets the card/default decide
            tail = f"   # {note}" if note else ""
            lines.append(f"{key} = {settings.literal(value)}{tail}")
    lines.append("")
    return "\n".join(lines)


def reset_to_baseline() -> dict:
    """Back the user config up with a timestamp, wipe all profiles and role
    assignments, and restore the recommended baseline defaults."""
    path = config_path("user")
    backup = ""
    if path.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bp = Path(f"{path}.reset.{stamp}.bak")
        bp.write_bytes(path.read_bytes())
        backup = str(bp)
    text = baseline_text()
    settings.validate_text(text, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)
    out = reload_state_config()
    out["backup"] = backup
    return out


# --- GUI theme ------------------------------------------------------------------

def save_theme(name: str) -> dict:
    """Persist `gui_theme` in the user config and reflect it in the live
    config. Purely cosmetic, so nothing is reloaded or restarted; when no
    user config file exists yet the choice stays browser-local only and
    `persisted` reports false (the first-run wizard still owns that file)."""
    name = str(name or "").strip().lower()
    themes = list(config_mod._ENUM_FIELDS["gui_theme"])
    if name not in themes:
        raise ConfigError(f"unknown theme {name!r}. Use one of: "
                          + ", ".join(themes))
    persisted = False
    if USER_CONFIG.exists():
        text = _read_text(USER_CONFIG)
        write_config(USER_CONFIG, settings.set_values(text, {"gui_theme": name}))
        # Re-read the user file so /api/state reports the new theme even when
        # no project is open (STATE.cfg is None until one is). In-place and
        # job-free — the whole point of a dedicated theme route — and the
        # dropdown's explicit profile pick survives the reload.
        with STATE.lock:
            keep = STATE.profile_request or None
        reload_state_config(profile=keep)
        persisted = True
    else:
        with STATE.lock:
            if STATE.cfg is not None:
                STATE.cfg.gui_theme = name
    return {"gui_theme": name, "themes": themes, "persisted": persisted}


# --- Auto-Tune Universal Settings -----------------------------------------------

# The wizard answers are closed vocabularies; anything else is refused.
AUTOTUNE_QUESTIONS = {
    "hardware": ("lt8", "8to16", "16to24", "gt24", "hosted"),
    "model": ("small", "medium", "large", "hosted"),
    "codebase": ("small", "medium", "large"),
    "priority": ("token_saver", "balanced", "thorough"),
}

# Starting budgets per hardware tier: (max_read_tokens, max_gathered_chars,
# max_total_model_calls, max_no_progress_steps). A small VRAM box pays for
# every extra token with evictions and re-prefills, so gather stays lean.
_HW_TIERS = {
    "lt8": (800, 12000, 25, 3),
    "8to16": (1000, 16000, 30, 3),
    "16to24": (1400, 24000, 35, 4),
    "gt24": (2000, 32000, 45, 5),
    "hosted": (2000, 32000, 50, 5),
}


def _autotune_values(answers: dict) -> tuple[dict, dict]:
    """Deterministic (values, reasons) for the four wizard answers."""
    hw = answers["hardware"]
    model = answers["model"]
    code = answers["codebase"]
    prio = answers["priority"]
    read, gathered, calls, no_progress = _HW_TIERS[hw]
    weak_model = model in ("small", "medium")
    reasons: dict[str, str] = {}

    # Codebase size moves the budgets before the priority trade-off does: a
    # big repo needs bigger reads and more steps to map it, a small one is
    # mostly context already.
    if code == "large":
        calls = int(calls * 1.2 + 0.5)
        gathered = int(gathered * 1.25)
        if hw != "lt8":
            read = int(read * 1.15)
        reason_size = "a large codebase needs more lookup steps"
    elif code == "small":
        gathered = int(gathered * 0.75)
        reason_size = "a small codebase fits in fewer, smaller observations"
    else:
        reason_size = "a medium codebase sits at the tier defaults"

    # The priority answer decides where the remaining headroom goes.
    if prio == "token_saver":
        calls = int(calls * 0.8 + 0.001)
        no_progress = max(2, no_progress - 1)
        gathered = int(gathered * 0.8)
        verify = "off" if hw == "lt8" else "deterministic"
        best_of = 1
        reason_prio = "spending fewest tokens: tighter caps and no extra model rounds"
    elif prio == "thorough":
        calls = int(calls * 1.25 + 0.5)
        no_progress += 1
        verify = "full"
        best_of = 2 if weak_model else 1
        reason_prio = "quality first: more attempts, a verifier call, and two answer candidates on weaker models"
    else:
        verify = "deterministic"
        best_of = 1
        reason_prio = "balanced defaults for this tier"

    values: dict = {
        "max_read_tokens": read,
        "max_gathered_chars": gathered,
        "max_total_model_calls": calls,
        "max_no_progress_steps": no_progress,
        "verify_answer": verify,
        "best_of": best_of,
    }
    # Structured, incremental editing is the single biggest help for weaker
    # local models; big/hosted models hold a whole multi-file patch fine.
    if weak_model:
        values.update({
            "incremental_edit_loop": True,
            "step_reasoning_first": True,
            "edit_plan_brief": True,
            "dynamic_tool_pruning": True,
            "repair_rounds": 3,
            "post_edit_check": True,
        })
        reasons["incremental_edit_loop"] = (
            "one symbol per verified micro-loop keeps a "
            + ("small " if model == "small" else "mid-size ")
            + "model from emitting a broken multi-file patch")
        reasons["step_reasoning_first"] = (
            "state the reason before the action: fewer pointless steps on a "
            "model that does not plan ahead")
        reasons["edit_plan_brief"] = (
            "a 1-3 line plan before each edit block measurably improves block "
            "matching on weaker models")
        reasons["dynamic_tool_pruning"] = (
            "drop tools the task will not use: a small context spends its "
            "window on code, not on schema")
        reasons["repair_rounds"] = (
            "one more SEARCH/REPLACE retry round pays off when the model "
            "copies search text imperfectly")
        reasons["post_edit_check"] = (
            "a cheap per-file syntax/lint gate catches a broken hunk right "
            "after it lands, while the model still remembers the file")
    else:
        values["dynamic_tool_pruning"] = False
        values["step_reasoning_first"] = True
        reasons["dynamic_tool_pruning"] = (
            "a strong model with room to spare: keep every tool visible "
            "rather than pruning the list mid-task")
        reasons["step_reasoning_first"] = (
            "reasoning before the action still costs almost nothing and keeps "
            "the transcript auditable")

    reasons["max_read_tokens"] = (
        f"{hw.upper() if hw != 'hosted' else 'hosted'} hardware: "
        + (f"{read} tokens per gather read keeps one observation cheap"
           if hw in ("lt8", "8to16") else
           f"{read} tokens per gather read fits the window comfortably"))
    reasons["max_gathered_chars"] = (
        f"cap on what one task accumulates: {reason_size}")
    reasons["max_total_model_calls"] = (
        f"hard ceiling per task: {calls} calls, {reason_prio}")
    reasons["max_no_progress_steps"] = (
        f"end gather after {no_progress} fruitless steps, {reason_prio}")
    reasons["verify_answer"] = (
        f"check answers " + ("only with deterministic scans"
                             if verify == "deterministic" else
                             "with one verifier model call"
                             if verify == "full" else
                             "off: every extra round costs tokens"))
    reasons["best_of"] = (
        f"sample {best_of} candidate answer(s) and keep the best-scoring one")
    if prio == "thorough":
        reasons["max_read_tokens"] += "; thorough still caps one read"
    return values, reasons


def autotune(answers: dict, apply: bool = False) -> dict:
    """The \"Auto-Tune Universal Settings\" wizard: deterministic, offline
    recommendations for the universal budgets from four questions. Preview
    (`apply` false) writes nothing; applying stores the changes in the user
    config as `[universal]` values and reloads the live config."""
    clean: dict = {}
    for key, allowed in AUTOTUNE_QUESTIONS.items():
        val = str((answers or {}).get(key) or "").strip().lower()
        if val not in allowed:
            raise ConfigError(
                f"{key} must be one of: {', '.join(allowed)}")
        clean[key] = val
    values, reasons = _autotune_values(clean)
    cfg = STATE.cfg if STATE.cfg is not None else config_mod.Config()
    changes = [{"key": k, "value": v, "current": getattr(cfg, k),
                "reason": reasons.get(k, "")}
               for k, v in values.items()]
    out = {"answers": clean, "changes": changes, "applied": False}
    if not apply:
        return out
    path = config_path("user")
    text = _read_text(path)
    if not text and not path.exists():
        raise ConfigError("No user config file yet. Use \u201cCreate config\u201d "
                          "first.")
    write_config(path, settings.set_values(text, dict(values)))
    with STATE.lock:
        keep = STATE.profile_request or None
    out["payload"] = reload_state_config(profile=keep)
    out["applied"] = True
    return out


# --- immediate STATE config reload --------------------------------------------

def reload_state_config(profile: str | None = None) -> dict:
    """Re-read the config files into STATE so /api/state and /api/settings
    show the change at once (the profile dropdown included), without waiting
    for a project re-open job. Returns the fresh settings payload.

    `profile` forces the profile selection after the reload (the rename
    endpoint passes it when the renamed profile was the explicitly chosen
    one); None keeps the files' own selection and clears any stale request."""
    from ..agent import Agent
    from ..llm import LLMClient
    with STATE.lock:
        root = STATE.root
    try:
        cfg = config_mod.load(root=root, profile=profile,
                              require_hardware=False, warn_budgets=False)
    except ConfigError:
        cfg = None
    with STATE.lock:
        STATE.cfg = cfg
        STATE.client = LLMClient(cfg) if cfg is not None else None
        STATE.agent = (Agent(STATE.idx, STATE.client, cfg)
                       if cfg is not None and STATE.idx is not None else None)
        STATE.profile_request = profile or ""
        STATE.model_checked, STATE.model_status, STATE.model_ok = False, "", None
    return payload()


# --- selective model discovery and import ---------------------------------------

def _norm_url(url) -> str:
    """The canonical spelling of a server URL (see `discovery.normalize_url`):
    no trailing slash, `localhost` folded to `127.0.0.1`, `/v1` suffix gone."""
    from .. import discovery
    return discovery.normalize_url(url)


# What the profile editor (and the settings page) accept as a profile name.
PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")
_MAX_PROFILE_NAME = 32


def _sanitize_profile_name(raw_name: str) -> str:
    """Harden a profile name so version numbers survive the round trip.

    Model ids carry things a profile name may not: `qwen3.8-27b` smuggles a
    dot, `glm-4.7-flash:latest` a colon, `mistral@7b` an at-sign. Each run of
    characters outside `[a-z 0-9 _ -]` becomes one `_` (`3.8` -> `3_8`), the
    name is lower-cased, repeated and dangling separators are collapsed and
    trimmed, a redundant `-gguf`/`_gguf` tail is dropped, and the result is
    capped at `_MAX_PROFILE_NAME`. A `/` is deliberately left untouched: it
    marks a name a human typed (an org path or a slip), and the import check
    should refuse that loudly instead of guessing a rewrite.
    """
    s = str(raw_name or "").strip().lower()
    s = re.sub(r"[^a-z0-9_/-]+", "_", s)
    s = re.sub(r"[-_]{2,}", lambda m: m.group(0)[0], s)
    s = s.strip("-_")
    while len(s) > 4 and s.endswith(("-gguf", "_gguf")):
        s = s[: -len("-gguf")].strip("-_")
    if len(s) > _MAX_PROFILE_NAME:
        s = s[:_MAX_PROFILE_NAME].strip("-_")
    return s


def _suggest_profile_name(model_id: str, taken: set, label: str = "") -> str:
    """The scan's suggested profile name: `config_editor.short_name`'s slug,
    sanitized (version dots and all), and made unique with `_2`, `_3`, ... if
    two ids collapse to the same slug."""
    base = _sanitize_profile_name(config_editor.short_name(model_id, taken,
                                                           label)) or "model"
    name = base
    i = 2
    while name in taken:
        tail = "_%d" % i
        name = base[:_MAX_PROFILE_NAME - len(tail)].strip("-_") + tail
        i += 1
    taken.add(name)
    return name


def scan_models(urls=(), vram_gb=0) -> dict:
    """`pedacito models scan`'s discovery pass WITHOUT the writes: probe the
    well-known local ports, every configured profile's server, and `urls`;
    return one selectable row per model. Nothing is written to the config.

    Candidates arrive spelled many ways (`localhost` vs `127.0.0.1`, a
    trailing `/v1`, a trailing slash), so every server is keyed by its
    normalized URL: one server probes once, and the returned list holds one
    row per (normalized server URL, model id) -- never a duplicate."""
    from .. import cli as cli_mod
    from .. import discovery
    explicit = [str(u).strip() for u in (urls or ()) if str(u).strip()]
    try:
        profile_urls = cli_mod._profile_base_urls()
    except Exception:
        profile_urls = []
    try:
        vram = int(vram_gb or 0)
    except (TypeError, ValueError):
        vram = 0
    found = discovery.discover_servers(timeout=2.0, urls=explicit,
                                       profile_urls=profile_urls)
    try:
        flat, existing = settings.validate_text(_read_text(config_path("user")))
    except ConfigError:
        flat, existing = {}, {}
    imported = {(_norm_url(b.get("base_url", flat.get("base_url", ""))),
                 str(b.get("model") or "")) for b in existing.values()}
    taken = set(existing)
    servers_out, models_out = [], []
    seen_servers: set[str] = set()
    seen_models: set[tuple[str, str]] = set()
    for label, base_url, _ids in found:
        skey = _norm_url(base_url)
        if skey and skey in seen_servers:
            continue        # the same server spelled another way: probe once
        seen_servers.add(skey)
        try:
            recs = discovery.describe_models(base_url, vram_gb=vram)
        except Exception:
            recs = []
        rows = []
        for rec in recs:
            mid = str(rec.get("id") or "")
            if not mid or (skey, mid) in seen_models:
                continue    # one row per (normalized server URL, model id)
            seen_models.add((skey, mid))
            already = (skey, mid) in imported
            name = _suggest_profile_name(mid, taken, label)
            row = {"server_url": base_url, "provider": label, "model_id": mid,
                   "base_url": base_url, "loaded": rec.get("loaded"),
                   "context_length": int(rec.get("context_length") or 0),
                   "size_bytes": int(rec.get("size_bytes") or 0),
                   "quant": str(rec.get("quant") or ""),
                   "suggested_profile_name": name,
                   "already_imported": bool(already)}
            rows.append(row)
            models_out.append(row)
        servers_out.append({"provider": label, "base_url": base_url,
                            "models": rows})
    return {"servers": servers_out, "models": models_out}


def _read_config_text(path: Path) -> str:
    """The text to add profiles to: the existing file, else the starter
    template (so a first import produces a valid config)."""
    text = _read_text(path)
    return text if text.strip() else config_mod.TEMPLATE


def import_models(models) -> dict:
    """Write ONLY the user-selected profiles into the user config (with a
    .bak backup), then reload STATE so they appear in the header dropdown.

    `models` is a list of {profile_name, base_url, model, context_length,
    vram_gb, tier, set_default}. A name that already exists is skipped, not
    duplicated."""
    if not isinstance(models, list) or not models:
        raise ConfigError("Pick at least one model to import.")
    path = config_path("user")
    text = _read_config_text(path)
    imported: list[str] = []
    skipped: list[str] = []
    default = ""
    for sel in models:
        if not isinstance(sel, dict):
            continue
        name = str(sel.get("profile_name") or sel.get("name") or "").strip()
        mid = str(sel.get("model") or sel.get("model_id") or "").strip()
        base = str(sel.get("base_url") or sel.get("server_url") or "").strip()
        if not name or not mid or not base:
            raise ConfigError("Each imported model needs a profile name, a "
                              "model id, and a server URL.")
        # version numbers never fail an import: `qwen3.8` is auto-fixed to
        # `qwen3_8`. A name a human typed truly wrong (an org `path/`, say)
        # stays invalid and is refused rather than silently rewritten.
        fixed = _sanitize_profile_name(name)
        if not fixed or not PROFILE_NAME_RE.fullmatch(fixed):
            raise ConfigError(
                f"profile name {name!r} must use letters, digits, '-' and "
                "'_' (starting with a letter or digit, max "
                f"{_MAX_PROFILE_NAME} characters)")
        name = fixed
        if name in config_editor.profile_names(text):
            skipped.append(name)
            if sel.get("set_default"):
                default = default or name
            continue
        values: dict = {"base_url": base, "model": mid}
        tier = str(sel.get("tier") or "").strip()
        if not tier:
            tier = config_mod._guess_tier(name, values) or ""
        if tier:
            values["tier"] = tier
        for key, raw in (("vram_gb", sel.get("vram_gb")),
                         ("context_tokens", sel.get("context_length"))):
            try:
                n = int(raw or 0)
            except (TypeError, ValueError):
                n = 0
            if n > 0:
                values[key] = n
        text = config_editor.add_profile(text, name, values)
        imported.append(name)
        if sel.get("set_default"):
            default = name
    if not imported and not skipped:
        raise ConfigError("Nothing to import: the selection was empty.")
    if default:
        if default not in config_editor.profile_names(text):
            raise ConfigError(f"default profile '{default}' was not imported")
        text = settings.set_values(text, {"default_profile": default})
    write_config(path, text)
    out = reload_state_config()
    out["imported"] = imported
    out["skipped"] = skipped
    out["default_profile"] = default or out["active_profile"]
    return out


def known_profiles() -> list[str]:
    """Profile names the calibration picker may offer: the live config's, or
    the user file's when no project (and no loaded config) is open."""
    with STATE.lock:
        cfg = STATE.cfg
    if cfg is not None and cfg.profiles:
        return sorted(cfg.profiles)
    try:
        return sorted(config_editor.profile_names(
            _read_text(config_path("user"))))
    except (ConfigError, OSError):
        return []


# --- pre-configured hosted provider profiles -------------------------------------

# The major hosted providers, spelled the way Pedacito reaches them: every
# entry is an OpenAI-compatible chat-completions server (`llm.py` speaks that
# one protocol, and Anthropic publishes their API behind the same
# `/v1/chat/completions` surface). `provider` names the vendor for the UI.
# A preset carries only public facts -- base URL, the env var that holds the
# key, curated model ids, and a generous context budget. Nothing secret.
CLOUD_PROVIDERS: tuple[dict, ...] = (
    {
        "id": "anthropic", "label": "Claude (Anthropic)", "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY", "suggest": "claude",
        "context_tokens": 200000,
        "models": ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-5"],
    },
    {
        "id": "openai", "label": "ChatGPT (OpenAI)", "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY", "suggest": "openai",
        "context_tokens": 272000,
        "models": ["gpt-5.4", "gpt-5.4-mini", "o3-mini", "gpt-4.1",
                   "gpt-4.1-mini"],
    },
    {
        "id": "deepseek", "label": "DeepSeek", "provider": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY", "suggest": "deepseek",
        "context_tokens": 128000,
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    {
        "id": "qwen", "label": "Qwen (DashScope)", "provider": "openai",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY", "suggest": "qwen",
        "context_tokens": 256000,
        "models": ["qwen3-coder-plus", "qwen-max", "qwen-plus", "qwen-flash"],
    },
    {
        "id": "gemini", "label": "Google Gemini", "provider": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY", "suggest": "gemini",
        "context_tokens": 200000,
        "models": ["gemini-2.5-pro", "gemini-2.5-flash"],
    },
    {
        "id": "openrouter", "label": "OpenRouter", "provider": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY", "suggest": "openrouter",
        "context_tokens": 200000,
        "models": ["deepseek/deepseek-chat", "qwen/qwen3-coder",
                   "anthropic/claude-3.7-sonnet"],
    },
)

# Generous gather reads for cloud windows (the `hosted` tier's own budgets are
# the floor; a pasted key should never have to fight tiny local-model caps).
_CLOUD_READ_TOKENS = 3000


def _preset_by_id(preset_id: str) -> dict:
    pid = str(preset_id or "").strip().lower()
    for p in CLOUD_PROVIDERS:
        if p["id"] == pid:
            return p
    raise ConfigError(f"unknown provider preset {preset_id!r}. Use one of: "
                      + ", ".join(p["id"] for p in CLOUD_PROVIDERS))


def provider_presets() -> dict:
    """`GET /api/providers/presets`: the quick-add catalog for the UI."""
    return {"presets": [dict(p) for p in CLOUD_PROVIDERS]}


# Tokens that add nothing to an auto-suggested profile name: the vendor's own
# words (the preset's family prefix already says it) and pure version numbers.
# Keep style words (sonnet, mini, reasoner...) -- `claude-sonnet-4-5` should
# suggest `claude_sonnet`, not bare `claude`.
_FAMILY_WORDS = {"claude", "anthropic", "gpt", "chatgpt", "openai",
                 "deepseek", "qwen", "gemini", "google", "dashscope",
                 "openrouter", "o3", "o4"}


def _preset_profile_suggestion(preset: dict, model: str) -> str:
    """`claude_sonnet`, `deepseek_chat`, `gemini_flash`...: the preset's short
    family word plus up to two meaningful tokens of the model id (org paths
    like `qwen/qwen3-coder` lose the path, version dots lose the numbers)."""
    family = str(preset.get("suggest") or preset["id"])
    tail: list[str] = []
    for tok in re.split(r"[^A-Za-z0-9]+", model):
        low = tok.lower()
        if not tok or low in _FAMILY_WORDS or low == family or low.isdigit() \
                or (low != family and low.startswith(family)):
            continue
        tail.append(low)
        if len(tail) == 2:
            break
    if not tail:
        tail = [re.sub(r"[^A-Za-z0-9]+", "_", model).lower()]
    return _sanitize_profile_name("_".join([family] + tail)) or family


def add_provider_preset(body: dict) -> dict:
    """One-click hosted provider profile (Sprint J): the preset supplies the
    base URL, tier, context budgets and provider quirks; the user supplies an
    API key or the name of an environment variable that holds one. Writes the
    profile into the config file, reloads STATE, and returns the fresh
    settings payload so the topbar Profile dropdown can offer it immediately.

    `body`: {preset_id, model?, profile_name?, api_key?, api_key_env?,
             set_default?, file?}. A pasted key wins over the env var name
    (`api_key_env` beats `api_key` at request time, so only one is stored)."""
    body = body or {}
    preset = _preset_by_id(body.get("preset_id"))
    model = str(body.get("model") or "").strip() or preset["models"][0]
    raw_name = str(body.get("profile_name") or "").strip()
    name = _sanitize_profile_name(
        raw_name or _preset_profile_suggestion(preset, model))
    if not name or not PROFILE_NAME_RE.fullmatch(name):
        raise ConfigError(
            f"profile name {raw_name!r} must use letters, digits, '-' and "
            "'_' (starting with a letter or digit, max "
            f"{_MAX_PROFILE_NAME} characters)")
    api_key = str(body.get("api_key") or "").strip()
    api_key_env = str(body.get("api_key_env") or "").strip()
    if not api_key:
        api_key_env = api_key_env or preset["api_key_env"]
    values: dict = {"base_url": preset["base_url"], "model": model,
                    "tier": "hosted", "json_mode": "object",
                    "context_tokens": int(preset.get("context_tokens") or 0),
                    "max_read_tokens": _CLOUD_READ_TOKENS}
    # The provider table wins for the endpoints whose compatibility differs
    # from the generic hosted default (Anthropic's OpenAI-compatible surface
    # rejects json_object and cannot cache).
    from .. import providers
    values.update(providers.profile_overrides(preset["base_url"], model))
    if api_key:
        values["api_key"] = api_key
    elif api_key_env:
        values["api_key_env"] = api_key_env
    which = "project" if str(body.get("file") or "user") == "project" else "user"
    path = config_path(which)
    text = _read_config_text(path)
    if name in config_editor.profile_names(text):
        raise ConfigError(f"a profile named '{name}' already exists in {path}")
    text = config_editor.add_profile(text, name, values)
    set_default = bool(body.get("set_default"))
    if set_default:
        text = settings.set_values(text, {"default_profile": name})
    write_config(path, text)
    with STATE.lock:
        requested = STATE.profile_request
    # Selecting the new profile is the point of a one-click add only when it
    # is made the default; otherwise keep the profile the user was running.
    out = reload_state_config(profile=name if set_default
                              else (requested or None))
    out["added"] = name
    out["preset"] = preset["id"]
    out["model"] = model
    out["uses_env"] = bool(api_key_env and not api_key)
    return out
