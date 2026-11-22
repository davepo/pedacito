"""KV-cache sizing model: the single source of truth for cache-quantization
spellings, effective bits per element, and the per-token byte estimate shared
by `config.auto_fit`/`hardware_context_tokens` and the profiler footprint.

The precedence the helper encodes is exactly the documented order:

1. `kv_cache_bytes_per_token` (a measured value) is authoritative;
2. a known `model_size_bytes` gives the llama.cpp-style file-size heuristic,
   scaled by the K/V cache quantization;
3. architecture metadata (`n_layers`, `n_kv_heads`, `head_dim`) is the
   fallback when no file size is known.

An unset K or V side defaults to f16 (16 bits); f16 is the scaling baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import kvcache
from pedacito.kvcache import kv_bytes_per_token, kv_quant_bits, normalise_kv_quant


# ------------------------------------------------------------------ normalisation
@pytest.mark.parametrize("spelling,canonical", [
    ("q8_0", "q8_0"),
    ("Q8_0", "q8_0"),            # case folds
    ("f16", "f16"),
    ("fp16", "fp16"),
    ("bf16", "bf16"),
    ("f32", "f32"),
    ("q6_k", "q6_k"),
    ("q5_1", "q5_1"),
    ("q5_0", "q5_0"),
    ("q4_1", "q4_1"),
    ("q4_0", "q4_0"),
    ("iq4_nl", "iq4_nl"),
    ("iq4_xs", "iq4_xs"),
])
def test_normalise_keeps_canonical_names(spelling, canonical):
    assert normalise_kv_quant(spelling) == canonical


@pytest.mark.parametrize("spelling,canonical", [
    ("q8", "q8_0"),
    ("q6", "q6_k"),
    ("q6k", "q6_k"),
    ("q5", "q5_1"),
    ("q4", "q4_0"),
    ("q4.0", "q4_0"),
    ("q4.1", "q4_1"),
    ("float16", "f16"),
    ("half", "f16"),
    ("float32", "f32"),
    ("bfloat16", "bf16"),
    ("q8-0", "q8_0"),            # dashes normalise to underscores
    ("q8 0", "q8_0"),            # spaces normalise to underscores
])
def test_normalise_resolves_short_aliases(spelling, canonical):
    assert normalise_kv_quant(spelling) == canonical


@pytest.mark.parametrize("value", ["", "auto", "default", "none", "false",
                                   None, 0])
def test_normalise_maps_empty_auto_and_none_to_blank(value):
    assert normalise_kv_quant(value) == ""


def test_normalise_rejects_an_unknown_quantization():
    with pytest.raises(ValueError):
        normalise_kv_quant("q2_banana")


# ------------------------------------------------------------------ bit widths
@pytest.mark.parametrize("spelling,bits", [
    ("f32", 32.0),
    ("f16", 16.0),
    ("bf16", 16.0),
    ("q8_0", 8.5),
    ("q6_k", 6.5625),
    ("q5_1", 6.0),
    ("q5_0", 5.5),
    ("q4_1", 5.0),
    ("q4_0", 4.5),
    ("iq4_nl", 4.5),
    ("iq4_xs", 4.25),
    ("q8", 8.5),                 # short aliases resolve first
    ("q4", 4.5),
    ("float16", 16.0),
])
def test_kv_quant_bits_for_known_spellings(spelling, bits):
    assert kv_quant_bits(spelling) == bits


def test_kv_quant_bits_is_none_when_unset():
    assert kv_quant_bits("") is None
    assert kv_quant_bits("auto") is None
    assert kv_quant_bits("none") is None


def test_kv_quant_bits_rejects_an_unknown_spelling():
    with pytest.raises(ValueError):
        kv_quant_bits("q7_k")


# ------------------------------------------------- tier 1: an explicit byte count
def test_explicit_kv_bytes_per_token_wins_over_size_and_arch():
    assert kv_bytes_per_token(
        kv_cache_bytes_per_token=96 * 1024,
        model_size_bytes=7_000_000_000,
        n_layers=40, n_kv_heads=8, head_dim=128,
        k_cache_quant="q8_0", v_cache_quant="q4_0") == 96 * 1024


# ------------------------------------------------- tier 2: the model file size
def test_model_size_without_cache_quant_keeps_the_historical_heuristic():
    # 7 GB // 32768 reference tokens, floored at 16 KiB.
    assert kv_bytes_per_token(model_size_bytes=7_000_000_000) == 213_623
    assert kv_bytes_per_token(model_size_bytes=1_000) == 16 * 1024


def test_model_size_scales_with_asymmetric_k_and_v_cache_quant():
    base = 7_000_000_000 // 32768
    # k=q8_0 (8.5b) + v=q4_0 (4.5b), relative to the f16+f16 baseline.
    expected = max(4 * 1024, int(base * (8.5 + 4.5) / 32.0))
    got = kv_bytes_per_token(model_size_bytes=7_000_000_000,
                             k_cache_quant="q8_0", v_cache_quant="q4_0")
    assert got == expected == 86_784


def test_model_size_defaults_the_unset_side_to_f16():
    base = 7_000_000_000 // 32768
    expected = max(4 * 1024, int(base * (8.5 + 16.0) / 32.0))
    got = kv_bytes_per_token(model_size_bytes=7_000_000_000,
                             k_cache_quant="q8_0")
    assert got == expected == 163_555


# --------------------------------------------- tier 3: architecture metadata
def test_architecture_formula_keeps_the_weight_quantization_convention():
    elements = 2 * 40 * 8 * 128
    got = kv_bytes_per_token(n_layers=40, n_kv_heads=8, head_dim=128,
                             weight_bytes_per_param=0.5)
    assert got == int(elements * 0.5) == 40_960


def test_architecture_formula_uses_cache_bits_when_quantized():
    elements = 2 * 40 * 8 * 128
    got = kv_bytes_per_token(n_layers=40, n_kv_heads=8, head_dim=128,
                             k_cache_quant="q8_0", v_cache_quant="q4_0")
    assert got == int(elements * (8.5 + 4.5) / 16.0) == 66_560


def test_architecture_formula_defaults_an_unset_side_to_f16():
    elements = 2 * 40 * 8 * 128
    got = kv_bytes_per_token(n_layers=40, n_kv_heads=8, head_dim=128,
                             k_cache_quant="q8_0")
    assert got == int(elements * (8.5 + 16.0) / 16.0) == 125_440


# ---------------------------------------------------------- zero / missing input
def test_zero_and_missing_input_returns_zero():
    assert kv_bytes_per_token() == 0
    assert kv_bytes_per_token(model_size_bytes=0) == 0
    assert kv_bytes_per_token(n_layers=40) == 0           # heads/dim missing
    assert kv_bytes_per_token(n_layers=40, n_kv_heads=8) == 0
    assert kv_bytes_per_token(n_layers=0, n_kv_heads=8, head_dim=128) == 0


def test_kv_cache_module_is_the_shared_import():
    """config and profiler both re-export this module's model, so drift in one
    subsystem cannot silently fork the cache estimate."""
    from pedacito import config as config_mod
    from pedacito import profiler
    assert config_mod._normalise_kv_quant("q4") == normalise_kv_quant("q4")
    assert profiler.kv_bytes_per_token is kvcache.kv_bytes_per_token
