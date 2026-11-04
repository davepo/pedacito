"""Profile listing, scoring, and role recommendations.

`pedacito profiles` renders one table that the `config` command and the chat
`/profiles` command also use, so the three listings agree. It reads the model
cards (`pedacito calibrate`) and reports each profile's harness-fit as
mean +- spread with a confidence marker, its generation speed, the context
that fits its VRAM, and its card status. Role recommendations follow the B4
rules and are suggestions only -- nothing is written until the user confirms.
"""

from __future__ import annotations

import time

from . import modelcard

# The roles the planner can route to another profile (plus the worker, which
# is the top-level `default_profile` key).
_SPECIALIST_ROLES = ("escalate", "verifier", "planner", "summarizer")
_WORKER_MIN_CONTEXT = 16_384      # fit context floor for a worker
_WORKER_MIN_TOK_S = 15.0          # generation-speed floor for a worker
_SUMMARIZER_MIN_FIT = 0.55
_ESCALATE_MIN_GAIN = 0.1          # how much a specialist must beat the worker
_LOW_CONFIDENCE = "low confidence: run `calibrate --recalibrate`"


def _cards() -> dict:
    return modelcard._load_all()


def _spread(measured: dict) -> float | None:
    sds = [v for k, v in measured.items()
           if k.endswith("_sd") and isinstance(v, (int, float))]
    return max(sds) if sds else None


def list_profiles(cfg) -> list[dict]:
    """One record per profile, in config order: identity plus the card-derived
    score. The card is matched on (base_url, model), exactly as `calibrate`
    keys it."""
    from .config import _guess_tier, hardware_context_tokens
    from .evalrun import eval_reports as _eval_reports

    cards = _cards()
    reports = _eval_reports()
    out: list[dict] = []
    for name, body in cfg.profiles.items():
        model = body.get("model") or cfg.model
        base_url = body.get("base_url") or cfg.base_url
        rec = {
            "name": name,
            "model": model,
            "server": base_url,
            "tier": body.get("tier") or _guess_tier(name, body) or "",
            "active": name == cfg.active_profile,
        }
        key = f"{(base_url or '').rstrip('/')}|{model}"
        rep = reports.get(key) if model else None
        rec["eval"] = (rep.get("summary") or {}).get("pass_rate") \
            if isinstance(rep, dict) else None
        card = cards.get(key) if model else None
        if isinstance(card, dict):
            measured = card.get("measured") or {}
            rec["fit"] = measured.get("overall")
            rec["spread"] = _spread(measured)
            rec["tok_per_s"] = measured.get("tok_per_s")
            rec["repeats"] = card.get("repeats", 1)
            rec["size_bytes"] = measured.get("size_bytes")
            age = (time.time() - float(card.get("ts", 0))) / 86400
            rec["age_days"] = age
            if card.get("version") != modelcard.CARD_VERSION:
                rec["status"] = "re-calibrate"
            elif age > modelcard.CARD_TTL_DAYS:
                rec["status"] = "stale"
            else:
                rec["status"] = "ok"
        else:
            rec["fit"] = None
            rec["spread"] = None
            rec["tok_per_s"] = None
            rec["repeats"] = None
            rec["size_bytes"] = None
            rec["age_days"] = None
            rec["status"] = "uncalibrated"
        vram = body.get("vram_gb", getattr(cfg, "vram_gb", 0))
        if rec["size_bytes"] and vram and rec["tier"] != "hosted":
            rec["context_fits"] = hardware_context_tokens(
                int(vram), int(rec["size_bytes"]))
        else:
            rec["context_fits"] = None
        out.append(rec)
    return out


def _is_local(rec: dict) -> bool:
    return rec.get("tier") != "hosted"


def _score(rec: dict):
    """The ranking score: the eval pass rate when a report exists, else the
    card's overall harness-fit. Eval evidence is the strongest signal."""
    if rec.get("eval") is not None:
        return rec["eval"]
    return rec.get("fit")


def _confidence(rec: dict) -> str:
    if (rec.get("repeats") or 0) <= 1 or (rec.get("spread") or 0.0) > 0.1:
        return _LOW_CONFIDENCE
    return ""


def recommend(profiles: list[dict]) -> dict:
    """Role -> {"name", "reason"} following the B4 rules, or None when no
    profile clears the rule. Suggestions only; `apply_roles` writes them."""
    recs: dict = {"default_profile": None, "escalate": None, "verifier": None,
                  "planner": None, "summarizer": None}

    # Worker: highest score among local profiles with context that fits
    # >= 16k and a generation speed above the floor.
    local = [p for p in profiles if _is_local(p) and _score(p) is not None]
    worker = None
    for p in local:
        if (p.get("context_fits") or 0) < _WORKER_MIN_CONTEXT:
            continue
        if (p.get("tok_per_s") or 0.0) < _WORKER_MIN_TOK_S:
            continue
        if worker is None or _score(p) > _score(worker):
            worker = p
    if worker is None:
        recs["default_profile"] = {
            "reason": "no local profile fits >= 16k context and generates "
                      "over 15 tok/s; run `calibrate` to fill in the scores"}
    else:
        recs["default_profile"] = {
            "name": worker["name"],
            "reason": f"highest score ({_score(worker):.2f}) among local "
                      f"profiles that fit the context and speed floors"}

    # Specialists: highest score overall (local or hosted), different from the
    # worker, only when it beats the worker by the gain floor; speed breaks
    # ties. The same profile is picked for escalate/verifier/planner.
    scored = [p for p in profiles if _score(p) is not None
              and (worker is None or p["name"] != worker["name"])]
    specialist = max(
        scored, key=lambda p: (_score(p), p.get("tok_per_s") or 0.0)) \
        if scored else None
    if worker is not None and specialist is not None \
            and _score(specialist) + 1e-9 >= _score(worker) + _ESCALATE_MIN_GAIN:
        for role in ("escalate", "verifier", "planner"):
            recs[role] = {
                "name": specialist["name"],
                "reason": f"highest score ({_score(specialist):.2f}) "
                          f"that beats the worker by >= 0.1"}
    else:
        for role in ("escalate", "verifier", "planner"):
            recs[role] = {
                "reason": "no profile beats the worker by >= 0.1 (evidence-"
                          "gated, like items 1.1 and P1b)"}

    # Summarizer: fastest profile with score >= 0.55.
    fast = [p for p in profiles
            if (_score(p) or 0.0) >= _SUMMARIZER_MIN_FIT
            and p.get("tok_per_s") is not None]
    if fast:
        pick = max(fast, key=lambda p: p["tok_per_s"])
        recs["summarizer"] = {
            "name": pick["name"],
            "reason": f"fastest ({pick['tok_per_s']:.0f} tok/s) with "
                      f"score >= 0.55"}
    else:
        recs["summarizer"] = {
            "reason": "no profile has score >= 0.55 with a measured "
                      "speed; run `calibrate`"}

    # Confidence markers ride on the reason.
    for role, rec in recs.items():
        if not rec or not rec.get("name"):
            continue
        conf = _confidence(next(p for p in profiles if p["name"] == rec["name"]))
        if conf:
            rec["reason"] += f" -- {conf}"
    return recs


def apply_roles(text: str, recs: dict) -> str:
    """Return `text` with `roles = {...}` written from the recommendations.
    Only the specialist/summarizer roles are written; the worker is the
    top-level `default_profile` and stays the user's call."""
    from .config_editor import set_values
    roles = {role: rec["name"] for role, rec in recs.items()
             if role in _SPECIALIST_ROLES and rec and rec.get("name")}
    if not roles:
        return text
    return set_values(text, {"roles": roles})


def render(profiles: list[dict], recommend_rows=None) -> str:
    """The shared table. `recommend_rows` is `recommend(profiles)` when the
    caller wants the per-role suggestions appended."""
    header = ("name", "model", "tier", "harness-fit", "eval", "tok/s",
              "context", "status")
    rows = []
    for p in profiles:
        fit = f"{p['fit']:.2f}" if p.get("fit") is not None else "-"
        if p.get("spread"):
            fit += f" +-{p['spread']:.2f}"
        ev = f"{p['eval']:.0%}" if p.get("eval") is not None else "-"
        tok = f"{p['tok_per_s']:.0f}" if p.get("tok_per_s") is not None else "-"
        ctx = (f"{p['context_fits'] / 1000:.0f}k"
               if p.get("context_fits") else "-")
        status = p["status"]
        if status == "ok":
            status = f"ok ({p['age_days']:.0f}d)"
        name = ("* " if p["active"] else "  ") + p["name"]
        rows.append((name, p["model"] or "-", p["tier"] or "-", fit, ev, tok,
                     ctx, status))
    width = [len(h) for h in header]
    for r in rows:
        for i, cell in enumerate(r):
            width[i] = max(width[i], len(cell))
    sep = "  ".join("-" * w for w in width)
    lines = ["  ".join(h.ljust(width[i]) for i, h in enumerate(header)), sep]
    lines += ["  ".join(c.ljust(width[i]) for i, c in enumerate(r))
              for r in rows]
    if recommend_rows:
        lines.append("")
        lines.append("Recommendations (suggestions -- confirm before applying):")
        for role in ("default_profile", "escalate", "verifier", "planner",
                     "summarizer"):
            rec = recommend_rows.get(role)
            if rec and rec.get("name"):
                lines.append(f"  {role:<16} {rec['name']:<16} {rec['reason']}")
            elif rec:
                lines.append(f"  {role:<16} -                 {rec['reason']}")
    return "\n".join(lines)
