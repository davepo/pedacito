"""One KV-cache size model shared by config and the profiler.

Two subsystems used to size the same cache with different physics:
`config.hardware_context_tokens` used the model file size (a 32k-token cache
is roughly the weight file), while `profiler.estimate_model_footprint` used
the architecture formula with the *weight* quantization. This module owns the
single calculation both call.

Evidence precedence:

1. `kv_cache_bytes_per_token`: a measured value; authoritative.
2. A known model file size: the llama.cpp-style heuristic above, scaled by
   the K/V cache quantization. This is what `auto_fit` has always used, so
   results with the knobs unset stay byte-identical.
3. Architecture metadata (layers, KV heads, head dim): the fallback when no
   file size is known. Without explicit cache quantization it keeps the
   profiler's historical weight-quantization convention so existing footprint
   reports do not drift; with explicit cache quantization it uses the cache
   bits directly.

Cache quantization is independent of weight quantization. An unset side is
f16 (16 bits); f16 is the baseline for relative scaling.
"""

from __future__ import annotations

KV_REFERENCE_TOKENS = 32768
MIN_HEURISTIC_BYTES = 16 * 1024
MIN_QUANTIZED_BYTES = 4 * 1024
F16_BITS = 16.0

# Effective bits per cache element, including llama.cpp's per-block scale/min
# overhead. Only the size relative to f16 matters, so approximate is fine.
KV_QUANT_BITS = {
    "f32": 32.0, "fp32": 32.0,
    "f16": 16.0, "fp16": 16.0, "bf16": 16.0,
    "q8_0": 8.5,
    "q6_k": 6.5625,
    "q5_1": 6.0, "q5_0": 5.5,
    "q4_1": 5.0, "q4_0": 4.5, "iq4_nl": 4.5, "iq4_xs": 4.25,
}
KV_QUANT_SHORT = {
    "q8": "q8_0", "q6": "q6_k", "q6k": "q6_k",
    "q5": "q5_1", "q5.1": "q5_1", "q5.0": "q5_0",
    "q4": "q4_0", "q4.1": "q4_1", "q4.0": "q4_0",
    "float16": "f16", "half": "f16", "float32": "f32", "bfloat16": "bf16",
}


def normalise_kv_quant(value) -> str:
    """Canonicalise a cache-quantization spelling; ValueError if unknown."""
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw or raw in ("auto", "default", "none", "false"):
        return ""
    raw = raw.replace(" ", "_")
    if raw in KV_QUANT_BITS:
        return raw
    dotted = raw.replace("_", ".")
    if dotted in KV_QUANT_SHORT:
        return KV_QUANT_SHORT[dotted]
    if raw in KV_QUANT_SHORT:
        return KV_QUANT_SHORT[raw]
    raise ValueError(
        f"k_cache_quant/v_cache_quant: unknown cache quantization {value!r}. "
        "Use f16, bf16, f32, q8_0, q6_k, q5_1, q5_0, q4_1, q4_0, "
        "iq4_nl or iq4_xs.")


def kv_quant_bits(value: str) -> float | None:
    """Effective bits per cache element for a quantization name, or None if
    unset. Raises ValueError for an unknown spelling."""
    name = normalise_kv_quant(value)
    return KV_QUANT_BITS.get(name) if name else None


def _cache_bits(k_cache_quant: str, v_cache_quant: str):
    """(k_bits, v_bits) with f16 defaults, or None when neither is set."""
    k_bits = kv_quant_bits(k_cache_quant)
    v_bits = kv_quant_bits(v_cache_quant)
    if k_bits is None and v_bits is None:
        return None
    return (F16_BITS if k_bits is None else k_bits,
            F16_BITS if v_bits is None else v_bits)


def kv_bytes_per_token(*, kv_cache_bytes_per_token: int = 0,
                       model_size_bytes: int = 0,
                       n_layers: int = 0, n_kv_heads: int = 0,
                       head_dim: int = 0,
                       weight_bytes_per_param: float = 0.0,
                       k_cache_quant: str = "",
                       v_cache_quant: str = "") -> int:
    """KV-cache bytes per token from the best available evidence.

    Returns 0 only when neither a size nor usable architecture fields are
    known. `weight_bytes_per_param` is the legacy fallback used by the
    profiler when no file size and no cache quantization are available; the
    config path always has a file size, so it never reaches that branch.
    """
    explicit = int(kv_cache_bytes_per_token or 0)
    if explicit > 0:
        return explicit

    bits = _cache_bits(k_cache_quant, v_cache_quant)
    size_bytes = int(model_size_bytes or 0)
    if size_bytes > 0:
        per_token = max(MIN_HEURISTIC_BYTES,
                        size_bytes // KV_REFERENCE_TOKENS)
        if bits is not None:
            k_bits, v_bits = bits
            # Baseline here already covers K+V at f16 (32 bits total).
            per_token = max(MIN_QUANTIZED_BYTES,
                            int(per_token * (k_bits + v_bits)
                                / (2 * F16_BITS)))
        return per_token

    # `elements` counts K and V elements per token: 2 * L * H * D.
    elements = (2 * max(0, int(n_layers)) * max(0, int(n_kv_heads))
                * max(0, int(head_dim)))
    if elements <= 0:
        return 0
    if bits is None:
        return int(elements * float(weight_bytes_per_param or 0.0))
    k_bits, v_bits = bits
    return int(elements * (k_bits + v_bits) / F16_BITS)
