"""B6: eval evidence links into `profiles` and is preferred over harness-fit."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import evalrun, profiles
from pedacito import config as config_mod


def test_save_and_load_eval_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        tmp_path / "home" / "config.toml")
    (tmp_path / "home").mkdir()
    cfg = config_mod.Config()
    cfg.base_url = "http://localhost:11434/v1"
    cfg.model = "qwen3:14b"
    p = evalrun.save_eval_summary(cfg, {"pass_rate": 0.9, "repeats": 3})
    assert p.exists()
    reports = evalrun.eval_reports()
    key = "http://localhost:11434/v1|qwen3:14b"
    assert reports[key]["summary"]["pass_rate"] == 0.9


def test_profiles_show_eval_pass_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        tmp_path / "home" / "config.toml")
    (tmp_path / "home").mkdir()
    cfg0 = config_mod.Config()
    cfg0.base_url = "http://localhost:11434/v1"
    cfg0.model = "qwen3:14b"
    evalrun.save_eval_summary(cfg0, {"pass_rate": 0.9})
    cfg = config_mod.Config()
    cfg.base_url = "http://localhost:11434/v1"
    cfg.model = "qwen3:14b"
    cfg.vram_gb = 24
    cfg.profiles = {"qwen": {"model": "qwen3:14b",
                             "base_url": "http://localhost:11434/v1"}}
    cfg.active_profile = "qwen"
    rows = profiles.list_profiles(cfg)
    assert rows[0]["eval"] == 0.9
    assert "eval" in profiles.render(rows)


def test_recommend_prefers_eval_over_fit():
    profs = [
        {"name": "a", "model": "a", "server": "http://x/v1", "tier": "medium",
         "fit": 0.9, "eval": None, "spread": 0.0, "tok_per_s": 30,
         "repeats": 3, "context_fits": 32768, "active": False},
        {"name": "b", "model": "b", "server": "http://x/v1", "tier": "medium",
         "fit": 0.5, "eval": 0.95, "spread": 0.0, "tok_per_s": 30,
         "repeats": 3, "context_fits": 32768, "active": False},
    ]
    recs = profiles.recommend(profs)
    # b's eval (0.95) beats a's harness-fit (0.9)
    assert recs["default_profile"]["name"] == "b"
