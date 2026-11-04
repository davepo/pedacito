"""B4: profile listing, scoring, and role recommendations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import modelcard, profiles
from pedacito.config import Config


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(modelcard, "_path",
                        lambda: tmp_path / "home" / "model_cards.json")
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def _rec(name, fit, tok, ctx, tier="medium", repeats=3, spread=0.0):
    return {"name": name, "model": name, "server": "http://x/v1", "tier": tier,
            "active": False, "fit": fit, "spread": spread, "tok_per_s": tok,
            "repeats": repeats, "context_fits": ctx, "status": "ok"}


def _cfg(profiles):
    cfg = Config()
    cfg.base_url = "http://localhost:11434/v1"
    cfg.model = "m"
    cfg.vram_gb = 24
    cfg.profiles = profiles
    cfg.active_profile = next(iter(profiles))
    return cfg


def _save_card(model, url, overall, tok, size, repeats=3, sd=0.0, ts=None):
    import time
    card = {"version": modelcard.CARD_VERSION,
            "key": f"{url.rstrip('/')}|{model}", "ts": ts or time.time(),
            "model": model, "url": url,
            "measured": {"overall": overall, "overall_sd": sd,
                         "tok_per_s": tok, "size_bytes": size},
            "choices": {}, "notes": [], "repeats": repeats, "seconds": 1.0}
    modelcard.save_card(card)


def test_list_profiles_reads_card_scores(home):
    _save_card("qwen3:14b", "http://localhost:11434/v1", 0.82, 23.0, 8_900_000_000)
    cfg = _cfg({"qwen": {"model": "qwen3:14b",
                         "base_url": "http://localhost:11434/v1",
                         "tier": "medium", "vram_gb": 24},
                "bare": {"model": "other", "base_url": "http://localhost:1234/v1"}})
    rows = profiles.list_profiles(cfg)
    by_name = {r["name"]: r for r in rows}
    q = by_name["qwen"]
    assert q["fit"] == 0.82 and q["tok_per_s"] == 23.0
    assert q["tier"] == "medium" and q["status"] == "ok"
    assert q["context_fits"] > 0
    b = by_name["bare"]
    assert b["fit"] is None and b["status"] == "uncalibrated"


def test_recommend_is_deterministic():
    profs = [
        _rec("w", 0.70, 30, 32768),           # worker (meets floors)
        _rec("s", 0.55, 12, 32768),           # below the tok/s floor
        _rec("h", 0.85, 20, 32768, tier="hosted"),   # hosted: specialist
        _rec("fast", 0.60, 80, 8192),         # summarizer (fastest, fit >= .55)
    ]
    recs = profiles.recommend(profs)
    assert recs["default_profile"]["name"] == "w"
    for role in ("escalate", "verifier", "planner"):
        assert recs[role]["name"] == "h"
    assert recs["summarizer"]["name"] == "fast"


def test_recommend_declines_a_weak_specialist():
    profs = [_rec("w", 0.70, 30, 32768), _rec("h", 0.74, 20, 32768,
                                              tier="hosted")]
    recs = profiles.recommend(profs)
    # h beats w by only 0.04 (< 0.1), so no specialist is recommended
    assert recs["escalate"].get("name") is None
    assert "0.1" in recs["escalate"]["reason"]


def test_recommend_marks_single_run_low_confidence():
    profs = [_rec("w", 0.70, 30, 32768, repeats=1)]
    recs = profiles.recommend(profs)
    assert "low confidence" in recs["default_profile"]["reason"]


def test_recommend_says_so_when_nothing_clears_the_floors():
    recs = profiles.recommend([_rec("w", 0.70, 5, 32768)])   # tok/s too low
    assert recs["default_profile"].get("name") is None
    assert "15 tok/s" in recs["default_profile"]["reason"]


def test_render_table(home):
    cfg = _cfg({"qwen": {"model": "qwen3:14b",
                         "base_url": "http://localhost:11434/v1"}})
    out = profiles.render(profiles.list_profiles(cfg))
    assert "harness-fit" in out and "uncalibrated" in out
    assert "qwen" in out


def test_apply_roles_writes_only_the_roles_table():
    text = "[universal]\nvram_gb = 24\n"
    recs = {"default_profile": {"name": "w", "reason": "r"},
            "escalate": {"name": "big", "reason": "r"},
            "verifier": {"name": "big", "reason": "r"},
            "planner": None,
            "summarizer": {"name": "fast", "reason": "r"}}
    out = profiles.apply_roles(text, recs)
    assert 'escalate = "big"' in out
    assert 'summarizer = "fast"' in out
    # the worker is the top-level default_profile, never written into roles;
    # a None recommendation is not written either
    assert "default_profile" not in out
    assert "planner" not in out
    # the result still validates
    from pedacito import config_editor
    flat, _ = config_editor.validate_text(out)
    assert flat["roles"] == {"escalate": "big", "verifier": "big",
                             "summarizer": "fast"}


def test_profiles_cli_json(tmp_path, monkeypatch, capsys):
    from pedacito import cli
    from pedacito import config as config_mod
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "no-user.toml")
    monkeypatch.setattr(modelcard, "_path", lambda: tmp_path / "model_cards.json")
    _save_card("qwen3:14b", "http://localhost:11434/v1", 0.82, 23.0, 8_900_000_000)
    cfgfile = tmp_path / "config.toml"
    cfgfile.write_text(
        'default_profile = "qwen"\n'
        '[profiles.qwen]\n'
        'base_url = "http://localhost:11434/v1"\n'
        'model = "qwen3:14b"\n'
        'vram_gb = 24\n', encoding="utf-8")
    rc = cli.main(["profiles", "--json", "--config", str(cfgfile)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profiles"][0]["name"] == "qwen"
    assert payload["profiles"][0]["fit"] == 0.82
