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
"""

from __future__ import annotations

import copy
import sys

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
    # one usage ledger: the dedicated client reports into the worker's sink
    if hasattr(default, "on_call"):
        client.on_call = default.on_call
    cache[role] = client
    return client


def describe(cfg) -> list[str]:
    """One line per configured role for `pedacito config`."""
    return [f"{r}: profile '{p}'" for r, p in sorted((cfg.roles or {}).items())]
