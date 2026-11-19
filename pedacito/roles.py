"""Role routing: different call sites may use different models.

    roles = { planner = "big", verifier = "big", summarizer = "tiny" }

    client = roles.client_for(cfg, "verifier", default_client)

A local 9B can do the bulk lookups and edits while a hosted model plans the
checklist and reviews the diff; a tiny local model can write the index's
one-line summaries while a larger one works the tasks. Each role names a
profile from the config (`[profiles.NAME]`); the role's client is that profile
layered over the active settings. Unset roles use the worker's client. A role
whose server cannot be reached falls back to the worker with one warning, so a
sleeping hosted API never blocks a local run.

Roles: worker (default), planner, verifier, summarizer, repair, plus the
recovery rungs `escalate` (one more attempt on this model when a task ends with
no edit applied) and `fallback` (take over when the worker errors out).

`scout_client` lets the gather lookup loop run on the profile
named by `scout_profile` (a fast small model on a second GPU) while every
other call site -- planning, answer synthesis, edits, repair -- stays on the
worker (architect) client.

Specialist routing (`auto_route_specialties`, TECHNIQUES.md #50) picks a
whole-task worker by specialty tag instead: `select_specialist_profile`
scores the task against each profile's tags, and `switch_specialist_client`
loads the winner with a clean VRAM handoff when it shares the worker's local
server. Switches happen at task boundaries only, never mid-gather.
"""

from __future__ import annotations

import copy
import re
import sys
import time

ROLES = ("worker", "planner", "verifier", "summarizer", "repair",
         "escalate", "fallback")

def derive_config(cfg, profile: str):
    """A copy of `cfg` with profile `profile`'s settings (and tier preset)
    applied on top; None when the profile does not exist."""
    from . import config as config_mod
    body = (cfg.profiles or {}).get(profile)
    if body is None:
        return None
    new = copy.copy(cfg)
    new.sampler = dict(cfg.sampler or {})
    new.sources = dict(cfg.sources or {})
    merged, tier = config_mod._apply_preset(profile, body, "role")
    for key, value in merged.items():
        setattr(new, key, config_mod._coerce(key, value))
        new.sources[key] = f"role profile '{profile}'"
    new.active_profile = profile
    new.active_tier = tier or cfg.active_tier
    # A tiered MoE profile gets its output/timeout/stream floors BEFORE
    # auto_fit, so the window clamp sees the real budgets rather than raising
    # them back above a small context afterwards.
    config_mod.apply_local_moe_defaults(new)
    try:
        from . import profiler
        profiler.apply_auto_tune(new)
    except Exception:
        pass
    # a different server/model: it must re-derive hosted-ness and its budgets
    new._hosted_server = config_mod.is_hosted(new)
    if new._hosted_server and "json_mode" not in merged:
        new.json_mode = "object"
    try:
        config_mod.auto_fit(new, verbose=False)
    except Exception:
        pass
    return new


def client_for(cfg, role: str, default):
    """The client for `role`: a dedicated one when `cfg.roles[role]` names a
    profile, else `default`. Clients are cached on the config."""
    name = (getattr(cfg, "roles", None) or {}).get(role)
    if not name or default is None:
        return default
    cache = cfg.__dict__.setdefault("_role_clients", {})
    if role in cache:
        return cache[role] or default
    from .llm import LLMClient, LLMError
    new_cfg = derive_config(cfg, str(name))
    if new_cfg is None:
        print(f"[roles] role '{role}' names unknown profile '{name}'; using "
              "the worker model", file=sys.stderr)
        cache[role] = None
        return default
    try:
        client = LLMClient(new_cfg)
        client.check(warn=False)
        client.warmup()
    except LLMError as e:
        print(f"[roles] role '{role}' ({name}) is unavailable "
              f"({str(e).splitlines()[0][:100]}); using the worker model",
              file=sys.stderr)
        cache[role] = None
        return default
    client.active_model_role = {"profile": str(name), "model": new_cfg.model,
                                "role": str(role), "specialty": ""}
    # one usage ledger: the dedicated client reports into the worker's sink
    if hasattr(default, "on_call"):
        client.on_call = default.on_call
    cache[role] = client
    return client


def scout_client(cfg, default):
    """The gather-phase scout client named by `cfg.scout_profile`.

    Dual-model scout/architect routing: a fast profile answers the per-tool
    lookup steps (grep, outline, read, symbol, usages, run_test) while the
    worker (architect) keeps planning, synthesis, and edits. Returns `default`
    -- the worker client -- when no scout is configured, the name matches the
    active profile, the profile does not exist, or its server is unreachable
    (one warning, verdict cached on the config so it is asked once)."""
    name = str(getattr(cfg, "scout_profile", "") or "").strip()
    if not name or default is None or name == getattr(cfg, "active_profile", ""):
        return default
    cache = cfg.__dict__.setdefault("_scout_clients", {})
    if name in cache:
        return cache[name] or default
    from .llm import LLMClient, LLMError
    new_cfg = derive_config(cfg, name)
    if new_cfg is None:
        print(f"[scout] scout_profile names unknown profile '{name}'; gather "
              "runs on the active profile", file=sys.stderr)
        cache[name] = None
        return default
    try:
        client = LLMClient(new_cfg)
        client.check(warn=False)
        client.warmup()
    except LLMError as e:
        print(f"[scout] scout profile '{name}' is unavailable "
              f"({str(e).splitlines()[0][:100]}); gather runs on the active "
              "profile", file=sys.stderr)
        cache[name] = None
        return default
    client.active_model_role = {"profile": str(name), "model": new_cfg.model,
                                "role": "scout", "specialty": ""}
    # one usage ledger: the scout reports into the worker's sink
    if hasattr(default, "on_call"):
        client.on_call = default.on_call
    cache[name] = client
    return client


def resolve_model_config(cfg, provider: str, model: str):
    """A config copy pointed at an explicit (provider, model) pair.

    `provider` names a profile in [profiles.*] (by its name or its `provider`
    label), or a base URL ("http://127.0.0.1:1234/v1"); empty keeps the active
    endpoint. `model` overrides the model id. Shared by split-stage hybrid
    routing (the Stage-1 architect) and the sandbox arena's contestants.
    Returns None when `model` is empty -- there is nothing to resolve."""
    from . import config as config_mod
    provider = str(provider or "").strip()
    model = str(model or "").strip()
    if not model:
        return None
    new = None
    if provider:
        profiles = getattr(cfg, "profiles", None) or {}
        name = provider if provider in profiles else next(
            (n for n, body in profiles.items()
             if str((body or {}).get("provider", "") or "").strip()
             == provider), "")
        if name:
            new = derive_config(cfg, name)
        elif "://" in provider:
            new = copy.copy(cfg)
            new.sampler = dict(cfg.sampler or {})
            new.sources = dict(cfg.sources or {})
            new.base_url = provider
    if new is None:
        new = copy.copy(cfg)
        new.sampler = dict(cfg.sampler or {})
        new.sources = dict(cfg.sources or {})
    new.model = model
    new.sources["model"] = f"selected model '{model}'"
    # A different endpoint/model re-derives hosted-ness and its budgets.
    new._hosted_server = config_mod.is_hosted(new)
    if new._hosted_server:
        new.json_mode = "object"
    try:
        config_mod.auto_fit(new, verbose=False)
    except Exception:
        pass
    return new


# ------------------------------------------------- specialist routing (#50)
# auto_route_specialties sends each task to the profile whose specialty tags
# best match it. Routing happens ONLY at task boundaries (never mid-gather):
# the gather conversation's KV prefix stays cacheable on whichever server
# owns it. On a single local server the outgoing model is unloaded first so
# the specialist fits in VRAM (the clean handoff); profiles on different
# servers stay warm together. An unreachable specialist falls back to the
# worker with one warning, exactly like the other roles.

_SPECIALTY_TAG_RE = re.compile(r"\[specialty:\s*([A-Za-z0-9_-]+)\s*\]")


def _profile_card(cfg, name: str):
    """The saved model card for a profile (matched on base_url+model, the
    same key `calibrate` uses), or None."""
    from types import SimpleNamespace

    from . import modelcard
    body = (cfg.profiles or {}).get(name) or {}
    probe = SimpleNamespace(
        base_url=str(body.get("base_url") or cfg.base_url or ""),
        model=str(body.get("model") or cfg.model or ""))
    try:
        return modelcard.load(probe)
    except Exception:
        return None


def detected_specialties(cfg, name: str) -> tuple[str, ...]:
    """The tags a profile's model card reports; with no card (or cards off)
    the model-name heuristics stand in."""
    from . import modelcard
    if getattr(cfg, "use_model_card", True):
        card = _profile_card(cfg, name)
        if card:
            tags = (card.get("measured") or {}).get("specialties") \
                or (card.get("choices") or {}).get("specialties")
            if tags:
                return tuple(str(t) for t in tags)
    body = (cfg.profiles or {}).get(name) or {}
    model = str(body.get("model") or cfg.model or "")
    return tuple(modelcard.infer_name_specialties(model))


def specialties_for(cfg, name: str) -> tuple[str, ...]:
    """The tags that count for routing: an explicit `specialties` list in
    the profile body wins, then the active config's own effective list,
    then the detected tags."""
    from . import config as config_mod
    body = (cfg.profiles or {}).get(name) or {}
    explicit = body.get("specialties")
    if explicit:
        try:
            return config_mod._coerce("specialties", explicit)
        except config_mod.ConfigError:
            pass
    if name == getattr(cfg, "active_profile", "") \
            and getattr(cfg, "specialties", ()):
        return tuple(cfg.specialties)
    return detected_specialties(cfg, name)


def profile_is_hosted(cfg, name: str) -> bool:
    """True when profile `name` points at a remote API rather than a
    self-hosted server.

    Routing decides this from the ENDPOINT, not the profile's `tier`: a local
    GGUF whose name happens to contain "deepseek" may carry tier = "hosted"
    from the old name heuristic, but it still runs on the user's box and must
    stay in a local specialist pool. Only a non-local base_url (api.deepseek.com,
    OpenAI, a tunnel to a hosted endpoint) is excluded."""
    from types import SimpleNamespace

    from . import llm as llm_mod
    body = (cfg.profiles or {}).get(name) or {}
    base = str(body.get("base_url") or cfg.base_url or "")
    try:
        return not llm_mod._is_local_server(SimpleNamespace(base_url=base))
    except Exception:
        from .config import _LOCAL_HOSTS, _hostname
        return _hostname(base) not in _LOCAL_HOSTS


def _task_specialty_scores(task: str, idx=None) -> dict[str, float]:
    """Keyword + file-extension scores per canonical tag for one task."""
    from .config import SPECIALTIES
    text = (task or "").lower()
    scores = {tag: 0.0 for tag in SPECIALTIES}
    for tag, meta in SPECIALTIES.items():
        for kw in meta["keywords"]:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                scores[tag] += 1.0
    # Files the task names: resolved through the index when there is one,
    # else raw file-shaped tokens (a bare "styles.css" still counts).
    files: list[str] = []
    if idx is not None:
        try:
            from . import rank
            files = rank.task_files(idx, task)
        except Exception:
            files = []
    if not files:
        files = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_./\-]*\.[A-Za-z0-9]+",
                           task or "")
    for f in files:
        low = f.lower()
        ext = "." + low.rsplit(".", 1)[-1] if "." in low else ""
        for tag, meta in SPECIALTIES.items():
            if ext in meta["extensions"]:
                scores[tag] += 1.0
        base = low.rsplit("/", 1)[-1]
        if (base.startswith("test_") or base.endswith("_test.py")
                or ".test." in base or ".spec." in base
                or "/tests/" in low or base == "conftest.py"):
            scores["testing_qa"] += 1.5
    return scores


# Which measured micro-probe speaks for each specialty tag (item 16): the
# domain probe, not the name, is the evidence that a profile can do the work.
_DOMAIN_PROBES = {
    "frontend_ui": "svg_ui",
    "3d_graphics_svg": "svg_ui",
    "math_algorithms": "math_algo",
    "sql_data": "sql_data",
    "testing_qa": "testing_qa",
    "coding": "edit_format",
    "translation_i18n": "translation",
}

# A larger preset tier is a small tie-break bonus, never enough to outvote a
# measured skill difference (item 16).
_TIER_BONUS = {"tiny": 0.0, "small": 0.01, "medium": 0.02,
               "large": 0.03, "hosted": 0.03}

# The active profile keeps a task it already fits unless a candidate's
# composite skill beats it by at least this much (item 16).
_INERTIA_MARGIN = 0.08

# A fallback worker must at least hold the edit format: a specialist that
# streamed unbounded hidden reasoning and wrote nothing must hand the task to
# a profile measured above the same competence gate the specialty tags use
# (item 15). A top edit-format card (1.0) also qualifies as a math/algorithm
# worker when no math specialist can finish the task.
_CODING_FALLBACK_MIN_EDIT = 0.65


def _specialist_skill(cfg, profiles: dict, name: str, tag: str) -> float:
    """Composite skill of one profile for one tag (item 16):
    0.40 * the tag's domain probe + 0.35 * edit_format + 0.25 * overall,
    plus a small tier bonus. An uncalibrated profile scores only its tier."""
    card = _profile_card(cfg, name) or {}
    m = card.get("measured") or {}
    probe = _DOMAIN_PROBES.get(tag, "")
    domain = float(m.get(probe, 0.0) or 0.0) if probe else 0.0
    ed = float(m.get("edit_format", 0.0) or 0.0)
    overall = float(m.get("overall", 0.0) or 0.0)
    tier = str((profiles.get(name) or {}).get("tier") or "")
    return (0.40 * domain + 0.35 * ed + 0.25 * overall
            + _TIER_BONUS.get(tier, 0.0))


def select_specialist_profile(task: str, idx, cfg, explicit_tag: str = "",
                              resident=None) -> tuple[str, str]:
    """(profile_name, matched_tag) for the specialist that best fits `task`,
    or ("", "") when the worker should keep it: routing off, fewer than two
    profiles, no positive score, or the active profile already carries a
    winning tag AND no candidate measurably outskills it (item 16: a
    composite lead of >= 0.08, or an active model named like a router /
    orchestrator, overrides the inertia). A "[specialty: TAG]" prefix in the
    task (or `explicit_tag`) forces the tag. Ties prefer a resident model,
    then a separate server (no VRAM handoff needed), then the highest
    composite card skill (domain probe, edit format, overall, tier)."""
    from .config import SPECIALTIES, normalize_specialty
    if not getattr(cfg, "auto_route_specialties", False):
        return "", ""
    profiles = getattr(cfg, "profiles", None) or {}
    if len(profiles) < 2:
        return "", ""
    tag = ""
    if explicit_tag:
        try:
            tag = normalize_specialty(explicit_tag)
        except Exception:
            tag = ""
    if not tag:
        m = _SPECIALTY_TAG_RE.search(task or "")
        if m:
            tag = m.group(1).lower()
    if tag in profiles:
        # A planner asked for a PROFILE by name, not a specialty: the prefix
        # is malformed (a profile name is not a tag). Ignore it and score the
        # task's own words instead of forcing a tag no card carries, which
        # used to leave the orchestrator holding specialist work.
        tag = ""
    if not tag:
        scores = _task_specialty_scores(task, idx)
    else:
        scores = {t: (1.0 if t == tag else 0.0) for t in SPECIALTIES}
        scores.setdefault(tag, 1.0)      # custom slugs route too
    active = getattr(cfg, "active_profile", "") or ""
    active_tags = set(specialties_for(cfg, active))
    allow_hosted = bool(getattr(cfg, "auto_route_allow_hosted", False))

    def _candidates(score_set):
        best = max(score_set.values()) if score_set else 0.0
        if best <= 0:
            return [], []
        winners = [t for t, s in score_set.items() if s == best]
        rows = []
        for name in profiles:
            if not allow_hosted and profile_is_hosted(cfg, name):
                # Local-only pool (the default): never jump to a paid cloud
                # API just because it carries a matching tag.
                continue
            hit = [t for t in winners if t in set(specialties_for(cfg, name))]
            if hit:
                rows.append((name, hit[0]))
        return winners, rows

    winning, candidates = _candidates(scores)
    if tag and not candidates:
        # A forced tag no local card carries (a custom slug only the user
        # knows, or a mistyped one): fall back to the task's measured
        # keyword score so the task still leaves an orchestrator.
        winning, candidates = _candidates(_task_specialty_scores(task, idx))
    if not candidates:
        return "", ""
    if active_tags.intersection(winning):
        # Inertia: a profile that already fits keeps the task -- but not at
        # any cost (item 16). A candidate whose composite skill leads by
        # _INERTIA_MARGIN takes it, and a model named like a router /
        # orchestrator never hoards specialist work it cannot do itself.
        active_model = str((profiles.get(active) or {}).get("model")
                           or cfg.model or "").lower()
        hoarding = any(w in active_model for w in ("orchestrator", "router"))
        if not hoarding:
            active_tag = next(t for t in winning if t in active_tags)
            active_skill = _specialist_skill(cfg, profiles, active, active_tag)
            best_skill = max(_specialist_skill(cfg, profiles, n, t)
                             for n, t in candidates)
            if best_skill < active_skill + _INERTIA_MARGIN:
                return "", ""
    resident = set(resident or ())
    worker_base = str(cfg.base_url or "").rstrip("/")

    def sort_key(item):
        name, tag = item
        body = profiles.get(name) or {}
        model = str(body.get("model") or cfg.model or "")
        base = str(body.get("base_url") or cfg.base_url or "").rstrip("/")
        return (0 if model in resident else 1,
                0 if base != worker_base else 1,
                -_specialist_skill(cfg, profiles, name, tag))

    candidates.sort(key=sort_key)
    name, won = candidates[0]
    if name == active:
        return "", ""
    return name, won


def select_coding_fallback(cfg, exclude=()) -> str:
    """Top-scoring LOCAL coding specialist for a zero-edit retry, or "".

    BUG_HUNT #14 Finding #5: a card can carry a specialty (e.g.
    ``math_algorithms``) yet burn its whole answer on hidden reasoning and
    write no files. The specialist router then leaves the task marked failed
    even though a capable coder sits idle. This picks the replacement worker:

    - local profiles only unless ``auto_route_allow_hosted`` is on;
    - ``exclude`` skips the specialist that just failed;
    - a measured card must clear ``_CODING_FALLBACK_MIN_EDIT`` (the same
      ``edit_format >= 0.65`` competence gate the specialty tags use);
    - a profile carrying the ``coding`` tag wins over a card that merely has
      a top edit-format score. The latter stays eligible so a top coding card
      (``edit_format == 1.0``) can take over math/algorithm work when the
      math specialist stalls.

    Ties go to the higher composite coding skill (edit format, overall,
    tier). Returns "" when routing is off or no candidate qualifies."""
    if not getattr(cfg, "auto_route_specialties", False):
        return ""
    profiles = getattr(cfg, "profiles", None) or {}
    exclude = {str(n) for n in (exclude or ()) if n}
    allow_hosted = bool(getattr(cfg, "auto_route_allow_hosted", False))
    rows: list[tuple[int, float, str]] = []
    for name in profiles:
        if name in exclude:
            continue
        if not allow_hosted and profile_is_hosted(cfg, name):
            continue
        card = _profile_card(cfg, name) or {}
        measured = card.get("measured") or {}
        ed = float(measured.get("edit_format", 0.0) or 0.0)
        tags = set(specialties_for(cfg, name))
        if measured and ed < _CODING_FALLBACK_MIN_EDIT:
            continue
        if "coding" in tags:
            rank = 0
        elif measured and ed >= _CODING_FALLBACK_MIN_EDIT:
            # No coding tag, but a measured top edit format: eligible as the
            # math/algorithm worker the failing specialist could not be.
            rank = 1
        else:
            continue
        rows.append((rank,
                     -_specialist_skill(cfg, profiles, name, "coding"),
                     name))
    if not rows:
        return ""
    rows.sort()
    return rows[0][2]


def _share_ledger(default, client) -> None:
    """One usage ledger and one budget: the specialist reports into the
    worker's stats dict under the same lock, and into its on_call sink, so
    tokens/cost merge into the main totals (warmup included)."""
    if hasattr(default, "stats") and hasattr(client, "stats"):
        client.stats = default.stats
    if hasattr(default, "_stats_lock") and hasattr(client, "_stats_lock"):
        client._stats_lock = default._stats_lock
    if hasattr(default, "on_call"):
        client.on_call = default.on_call


def switch_specialist_client(cfg, default, name: str):
    """The specialist profile's client, with a clean VRAM handoff when it
    shares the worker's LOCAL server: evict whoever holds the card (the
    worker, or the previous specialist), then load the new one. Clients are
    cached per profile on the config; the handoff itself runs on every
    switch. A one-shot `_swap_info` dict on the client records the swap for
    telemetry. Returns `default` with one warning when the profile is
    unknown or its server cannot be reached."""
    if not name or default is None:
        return default
    from . import llm as llm_mod
    cache = cfg.__dict__.setdefault("_specialist_clients", {})
    if cache.get(name) is None and name in cache:
        return default                     # unreachable verdict, cached
    new_cfg = derive_config(cfg, name)
    if new_cfg is None:
        print(f"[specialist router] unknown profile '{name}'; using the "
              "worker model", file=sys.stderr)
        cache[name] = None
        return default
    same_server = new_cfg.base_url.rstrip("/") == str(cfg.base_url).rstrip("/")
    local = llm_mod._is_local_server(new_cfg)
    handoff = same_server and local
    holder = cfg.__dict__.get("_specialist_on_worker_server", "")
    client = cache.get(name)
    t0 = time.time()
    unloaded = ""
    if client is None:
        try:
            client = llm_mod.LLMClient(new_cfg)
            client.check(warn=False)
        except llm_mod.LLMError as e:
            print(f"[specialist router] profile '{name}' is unavailable "
                  f"({str(e).splitlines()[0][:100]}); using the worker model",
                  file=sys.stderr)
            cache[name] = None
            return default
        # the ledger is shared BEFORE warmup, so its tokens count too
        _share_ledger(default, client)
        if handoff and holder != name:
            victim = cache.get(holder) if holder else default
            unloaded = llm_mod.unload_model(victim or default)
        try:
            client.warmup()
        except llm_mod.LLMError as e:
            print(f"[specialist router] profile '{name}' failed to load "
                  f"({str(e).splitlines()[0][:100]}); using the worker model",
                  file=sys.stderr)
            cache[name] = None
            return default
        cache[name] = client
    elif handoff and holder != name:
        # cached client, but the card changed hands since: hand off again
        victim = cache.get(holder) if holder else default
        unloaded = llm_mod.unload_model(victim or default)
        try:
            client.warmup()
        except llm_mod.LLMError as e:
            print(f"[specialist router] profile '{name}' failed to reload "
                  f"({str(e).splitlines()[0][:100]}); using the worker model",
                  file=sys.stderr)
            return default
    client.active_model_role = {"profile": str(name), "model": new_cfg.model,
                                "role": "specialist", "specialty": ""}
    if handoff:
        cfg.__dict__["_specialist_on_worker_server"] = name
        client._swap_info = {"unloaded": unloaded, "loaded": new_cfg.model,
                             "elapsed_s": round(time.time() - t0, 2)}
    return client


def restore_worker_client(cfg, worker):
    """After a task that did NOT route: when a specialist still holds the
    worker's local server, unload it and re-warm the worker so the next
    task starts on the model it expects. Returns the swap dict for
    telemetry, or None when nothing changed."""
    name = cfg.__dict__.pop("_specialist_on_worker_server", "")
    if not name or worker is None:
        return None
    from . import llm as llm_mod
    t0 = time.time()
    specialist = cfg.__dict__.get("_specialist_clients", {}).get(name)
    unloaded = llm_mod.unload_model(specialist) if specialist is not None else ""
    try:
        worker.warmup()
    except Exception as e:
        print(f"[specialist router] worker failed to re-warm after "
              f"'{name}' ({str(e).splitlines()[0][:100]}); the next task "
              "will retry it", file=sys.stderr)
    return {"unloaded": unloaded,
            "loaded": str(getattr(worker.cfg, "model", "") or ""),
            "elapsed_s": round(time.time() - t0, 2)}


def specialty_planner_note(cfg) -> str:
    """A planner-prompt addendum listing the specialists in play, so plans
    can prefix tasks with [specialty: TAG] and group same-specialty work
    together (fewer VRAM swaps) without breaking dependencies. Empty when
    routing is off or there is nothing to route between."""
    if not getattr(cfg, "auto_route_specialties", False):
        return ""
    profiles = getattr(cfg, "profiles", None) or {}
    if len(profiles) < 2:
        return ""
    tagged = {}
    allow_hosted = bool(getattr(cfg, "auto_route_allow_hosted", False))
    for name in profiles:
        if not allow_hosted and profile_is_hosted(cfg, name):
            continue
        tags = specialties_for(cfg, name)
        if tags and tags != ("general",):
            tagged[name] = tags
    if not tagged:
        return ""
    lines = ["SPECIALIST ROUTING is on: each task runs on the profile whose "
             "specialty tags match it best. Specialists:"]
    for name, tags in tagged.items():
        lines.append(f"- {name}: {', '.join(tags)}")
    vocabulary = sorted({t for tags in tagged.values() for t in tags
                         if t != "general"})
    if vocabulary:
        lines.append("TAG vocabulary (the ONLY values allowed after "
                     "'[specialty: '): " + ", ".join(vocabulary) + ".")
    lines.append(
        "Prefix a task title with [specialty: TAG] when one specialty "
        "clearly fits; TAG must come from that vocabulary -- never a "
        "profile name. When dependencies allow it, group tasks that use "
        "the same specialist next to each other: every switch between "
        "specialists on one server costs a model unload/load. Never move a "
        "task before the tasks it depends on.")
    return "\n".join(lines)


def describe(cfg) -> list[str]:
    """One line per configured role for `pedacito config`."""
    return [f"{r}: profile '{p}'" for r, p in sorted((cfg.roles or {}).items())]
