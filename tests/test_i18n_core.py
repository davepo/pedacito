"""WP-01: i18n core engine, config persistence, and the LLM reply directive.

Covers language normalisation and validity, detection precedence, catalog
fallback, safe formatting, `save_user_language` comment preservation, the
`PEDACITO_LANG` alias, the Spanish-only reply-language directive (and the
byte-identity of English prompts), and the guarantee that `gather_system`
never changes with the interface language.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import i18n
from pedacito import prompts
from pedacito.config import Config


@pytest.fixture(autouse=True)
def _reset_language():
    """`set_current_lang` mutates a process global; keep every test hermetic."""
    yield
    i18n.set_current_lang("en")


# --- normalization and validity -------------------------------------------------

def test_normalize_lang_maps_every_spelling():
    for raw in ("es", "es-419", "es-MX", "es_ES", "es-PE", "español", "spanish",
                "ES", "Es", " Español "):
        assert i18n.normalize_lang(raw) == "es", raw
    for raw in ("en", "en-US", "en_GB", "english", "EN", "English"):
        assert i18n.normalize_lang(raw) == "en", raw
    # Unrecognised or absent values fall back to English, never raise.
    assert i18n.normalize_lang("fr") == "en"
    assert i18n.normalize_lang("de") == "en"
    assert i18n.normalize_lang(None) == "en"
    assert i18n.normalize_lang("") == "en"


def test_is_valid_lang_is_strict():
    assert i18n.is_valid_lang("es") is True
    assert i18n.is_valid_lang("es-MX") is True
    assert i18n.is_valid_lang("spanish") is True
    assert i18n.is_valid_lang("en") is True
    assert i18n.is_valid_lang("english") is True
    assert i18n.is_valid_lang("fr") is False
    assert i18n.is_valid_lang("de") is False
    assert i18n.is_valid_lang(None) is False
    assert i18n.is_valid_lang("") is False


# --- constants and catalogs -----------------------------------------------------

def test_constants():
    assert i18n.SUPPORTED_LANGUAGES == ("en", "es")
    assert i18n.DEFAULT_LANGUAGE == "en"


def test_catalogs_have_identical_keys():
    en = i18n.load_catalog("en")
    es = i18n.load_catalog("es")
    assert en and es
    assert set(en) == set(es)
    for ns in ("common.ok", "errors.invalid_language", "lang.switched",
               "cli.usage", "tui.quit", "gui.settings", "settings.language"):
        assert ns in en and ns in es


def test_load_catalog_caches_and_normalises():
    assert i18n.load_catalog("en") is i18n.load_catalog("en")
    assert i18n.load_catalog("es-MX") is i18n.load_catalog("es")
    assert set(i18n.all_catalogs()) == {"en", "es"}


# --- detection precedence -------------------------------------------------------

class _Cfg:
    def __init__(self, language="es"):
        self.language = language


def test_detect_lang_precedence():
    cfg = _Cfg("es")
    # cli_lang wins over everything.
    assert i18n.detect_lang("en", cfg,
                            env={"PEDACITO_LANG": "es"}) == "en"
    # PEDACITO_LANG beats PEDACITO_LANGUAGE.
    assert i18n.detect_lang(None, cfg, env={
        "PEDACITO_LANG": "es", "PEDACITO_LANGUAGE": "en"}) == "es"
    # PEDACITO_LANGUAGE alone.
    assert i18n.detect_lang(None, cfg,
                            env={"PEDACITO_LANGUAGE": "es"}) == "es"
    # cfg.language.
    assert i18n.detect_lang(None, cfg, env={}) == "es"
    # default.
    assert i18n.detect_lang(None, None, env={}) == "en"


def test_detect_lang_uses_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("PEDACITO_LANG", "es")
    assert i18n.detect_lang(None, None) == "es"


# --- lookup fallback and safe formatting ----------------------------------------

def test_t_translates_and_falls_back():
    assert i18n.t("common.ok", lang="es") == "Aceptar"
    assert i18n.t("common.ok", lang="en") == "OK"
    # An unknown key degrades to the raw key, not an exception.
    assert i18n.t("totally.missing.key", lang="es") == "totally.missing.key"


def test_t_falls_back_to_english_when_spanish_is_missing(monkeypatch):
    real = i18n.load_catalog

    def fake(code):
        catalog = dict(real(code))
        if code == "es":
            catalog.pop("common.ok", None)
        return catalog

    monkeypatch.setattr(i18n, "load_catalog", fake)
    assert i18n.t("common.ok", lang="es") == "OK"


def test_t_formats_safely():
    assert i18n.t("cli.error", lang="en", message="boom") == "Error: boom"
    # A missing placeholder key renders literally instead of raising KeyError.
    assert i18n.t("cli.error", lang="en") == "Error: {message}"
    # Malformed format strings and stray placeholders never raise.
    assert i18n._safe_format("unclosed {brace", {}) == "unclosed {brace"
    assert i18n._safe_format("{missing}", {}) == "{missing}"
    assert i18n._safe_format("{name}", {"name": "x"}) == "x"


def test_current_lang_roundtrip():
    i18n.set_current_lang("es")
    assert i18n.get_current_lang() == "es"
    assert i18n.t("common.ok") == "Aceptar"
    i18n.set_current_lang("es-MX")
    assert i18n.get_current_lang() == "es"


# --- config persistence ---------------------------------------------------------

def test_save_user_language_preserves_comments(tmp_path, monkeypatch):
    from pedacito import config as config_mod

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "# my special comment -- do not touch\n"
        "[universal]\n"
        "vram_gb = 8\n",
        encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)

    result = i18n.save_user_language("es")
    assert result == {"language": "es", "persisted": True, "created": False}

    text = cfg_path.read_text(encoding="utf-8")
    assert "# my special comment -- do not touch" in text
    assert 'language = "es"' in text
    assert "vram_gb = 8" in text


def test_save_user_language_creates_from_template(tmp_path, monkeypatch):
    from pedacito import config as config_mod

    cfg_path = tmp_path / "brand-new.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)

    result = i18n.save_user_language("en")
    assert result["created"] is True
    assert 'language = "en"' in cfg_path.read_text(encoding="utf-8")


def test_save_user_language_normalises_and_rejects(tmp_path, monkeypatch):
    from pedacito import config as config_mod

    cfg_path = tmp_path / "c.toml"
    monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_path)
    # "spanish" normalises to "es" before it is written.
    assert i18n.save_user_language("spanish")["language"] == "es"
    assert 'language = "es"' in cfg_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        i18n.save_user_language("fr")


# --- config integration ---------------------------------------------------------

def test_language_fields_are_not_exec():
    from pedacito import config as config_mod
    assert "language" not in config_mod.EXEC_FIELDS
    assert "llm_reply_language" not in config_mod.EXEC_FIELDS


def test_pedacito_lang_env_alias(monkeypatch):
    from pedacito import config as config_mod

    monkeypatch.setenv("PEDACITO_LANGUAGE", "en")
    monkeypatch.setenv("PEDACITO_LANG", "es")
    cfg = config_mod.load(use_env=True, require_hardware=False,
                          warn_budgets=False)
    assert cfg.language == "es"
    assert cfg.sources["language"] == "env PEDACITO_LANG"


def test_pedacito_language_env_alone(monkeypatch):
    from pedacito import config as config_mod

    monkeypatch.setenv("PEDACITO_LANGUAGE", "es")
    cfg = config_mod.load(use_env=True, require_hardware=False,
                          warn_budgets=False)
    assert cfg.language == "es"
    assert cfg.sources["language"] == "env PEDACITO_LANGUAGE"


# --- the LLM reply-language directive -------------------------------------------

def test_effective_reply_language_matrix():
    en = Config()
    assert prompts.effective_reply_language(en) == "en"

    es_ui = Config()
    es_ui.language = "es"
    assert prompts.effective_reply_language(es_ui) == "es"

    force_es = Config()
    force_es.llm_reply_language = "es"
    assert prompts.effective_reply_language(force_es) == "es"

    force_en = Config()
    force_en.llm_reply_language = "en"
    force_en.language = "es"
    assert prompts.effective_reply_language(force_en) == "en"

    auto = Config()
    auto.llm_reply_language = "auto"
    assert prompts.effective_reply_language(auto) == "auto"


def test_reply_language_directive_en_vs_es():
    en = Config()
    assert prompts.reply_language_directive(en) == ""

    es = Config()
    es.language = "es"
    directive = prompts.reply_language_directive(es)
    assert directive == prompts._REPLY_LANGUAGE_DIRECTIVE
    assert "Latin American Spanish" in directive
    assert "SEARCH/REPLACE" in directive


def test_answer_system_is_byte_identical_for_english():
    en = Config()
    es = Config()
    es.language = "es"
    en_text = prompts.answer_system(en, "EDIT FORMAT", task="fix a bug")
    es_text = prompts.answer_system(es, "EDIT FORMAT", task="fix a bug")
    assert "Latin American Spanish" not in en_text
    # The Spanish brief is exactly the English brief plus the appended
    # directive, so English output is unchanged and Spanish adds only one block.
    assert es_text == en_text.rstrip() + "\n\n" + prompts._REPLY_LANGUAGE_DIRECTIVE


def test_gather_system_is_language_invariant():
    en = Config()
    es = Config()
    es.language = "es"
    es.llm_reply_language = "es"
    base = "GATHER BASE\n\nRules:\n- Do not guess at unread code."
    g_en = prompts.gather_system(en, base, "")
    g_es = prompts.gather_system(es, base, "")
    assert g_en == g_es
    assert "Latin American Spanish" not in g_en


def test_fingerprint_unchanged_for_english():
    en = Config()
    es = Config()
    es.language = "es"
    auto = Config()
    auto.llm_reply_language = "auto"
    # English and "auto" both omit the marker, so they share a hash; Spanish
    # adds it and therefore differs.
    assert prompts.fingerprint(en) == prompts.fingerprint(auto)
    assert prompts.fingerprint(en) != prompts.fingerprint(es)
