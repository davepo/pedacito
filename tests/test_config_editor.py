"""B2: profile naming (`short_name`) and the idempotent profile writer."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import config_editor as E


# --- short_name ---------------------------------------------------------------

def test_short_name_derivation():
    cases = {
        "qwen/qwen3-14b@q4_k_m": "qwen3-14b",
        "qwen3:14b": "qwen3-14b",
        "qwen3:latest": "qwen3",
        "org/model.gguf": "model",
        "meta-llama/Llama-3.2-8B-Instruct": "llama-3.2-8b-instruct",
        "Qwen3:14B": "qwen3-14b",
        "org/with space@q4_0": "with-space",
    }
    for model_id, want in cases.items():
        assert E.short_name(model_id) == want, model_id


def test_short_name_collisions():
    # quant disambiguates first
    assert E.short_name("qwen/qwen3-14b@q4_k_m", {"qwen3-14b"}) == "qwen3-14b-q4km"
    # then the server label
    assert E.short_name("qwen3:14b", {"qwen3-14b"}, "Ollama") == "qwen3-14b-ollama"
    # then a numeric suffix
    assert E.short_name("qwen3:14b", {"qwen3-14b", "qwen3-14b-ollama"}, "Ollama") \
        == "qwen3-14b-2"
    # no collision: the plain name, taken is untouched
    assert E.short_name("qwen3:14b", {"other"}) == "qwen3-14b"


# --- upsert_profiles ----------------------------------------------------------

def test_upsert_appends_identity_only_and_loads(tmp_path):
    text = config_mod.TEMPLATE
    out = E.upsert_profiles(text, [
        {"model": "qwen3:14b", "base_url": "http://localhost:11434/v1",
         "server_label": "Ollama", "tier": "medium", "vram_gb": 24,
         "quant": "Q4_K_M", "size_gb": 8.9}],
        now="2026-11-02")
    assert "# added by pedacito models scan, 2026-11-02, Ollama, Q4_K_M 8.9 GB" \
        in out
    assert '[profiles.qwen3-14b]' in out
    assert 'base_url = "http://localhost:11434/v1"' in out
    assert 'model = "qwen3:14b"' in out
    # identity only: no card choice is ever written into the new profile
    block = out.split("[profiles.qwen3-14b]", 1)[1].split("[", 1)[0]
    for key in ("prompt_pack", "best_of", "verify_answer", "answer_mode"):
        assert f"{key} =" not in block
    p = tmp_path / "config.toml"
    p.write_text(out, encoding="utf-8")
    cfg = config_mod.load(explicit=p, use_env=False, require_hardware=False,
                          warn_budgets=False)
    assert "qwen3-14b" in cfg.profiles


def test_upsert_is_idempotent():
    text = '[universal]\nvram_gb = 24\n'
    found = [{"model": "a", "base_url": "http://localhost:1234/v1",
              "server_label": "LM Studio"}]
    once = E.upsert_profiles(text, found, now="2026-11-02")
    twice = E.upsert_profiles(once, found, now="2026-11-02")
    assert twice == once


def test_upsert_preserves_user_comments_and_keys():
    text = (
        "# my custom comment\n"
        "[universal]\n"
        "vram_gb = 24\n"
        "\n"
        "[profiles.default]\n"
        "base_url = \"http://localhost:11434/v1\"\n"
        "model = \"qwen3:14b\"\n"
        "max_steps = 42  # user-added key\n"
    )
    out = E.upsert_profiles(text, [
        {"model": "new-model", "base_url": "http://localhost:1234/v1",
         "server_label": "LM Studio"}],
        now="2026-11-02")
    # the whole original survives verbatim as a prefix; only the new profile
    # is appended after it
    assert out.startswith(text.rstrip("\n"))
    assert "# my custom comment" in out
    assert "max_steps = 42  # user-added key" in out
    assert '[profiles.new-model]' in out


def test_upsert_matches_on_base_url_and_model():
    text = ('[profiles.default]\n'
            'base_url = "http://localhost:11434/v1"\n'
            'model = "qwen3:14b"\n')
    out = E.upsert_profiles(text, [
        {"model": "qwen3:14b", "base_url": "http://localhost:11434/v1"},
        {"model": "qwen3:14b", "base_url": "http://localhost:1234/v1"},
    ], now="2026-11-02")
    # the same (base_url, model) is skipped; the same model on another server
    # gets its own profile
    assert out.count("base_url = \"http://localhost:11434/v1\"") == 1
    assert '[profiles.qwen3-14b]' in out
    assert "base_url = \"http://localhost:1234/v1\"" in out


def test_upsert_takes_profile_names_as_collisions():
    text = ('[profiles.qwen3-14b]\n'
            'base_url = "http://localhost:11434/v1"\n'
            'model = "qwen3:14b"\n')
    # a *different* server's qwen3:14b collides on the short name, so it gets
    # the server-label suffix instead of clobbering the existing profile.
    out = E.upsert_profiles(text, [
        {"model": "qwen3:14b", "base_url": "http://localhost:1234/v1",
         "server_label": "LM Studio"}],
        now="2026-11-02")
    assert '[profiles.qwen3-14b-lm-studio]' in out


def test_editor_is_importable_without_the_gui_package():
    # The CLI needs the editor without pulling in pedacito.gui (which imports
    # the review server). Every test above exercised `config_editor` directly;
    # its source imports only `config`, so no gui module is involved.
    import inspect
    src = inspect.getsource(E.validate_text)
    assert "gui" not in src


def test_gui_settings_reexports_the_editor():
    from pedacito.gui import settings as S
    assert S.set_values is E.set_values
    assert S.add_profile is E.add_profile
    assert S.validate_text is E.validate_text
    assert S.upsert_profiles is E.upsert_profiles
