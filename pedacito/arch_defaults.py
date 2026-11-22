"""Base-architecture recommended settings for local models.

`pedacito calibrate` measures what a model can do; this library records what
its architecture family is known to want -- the sampler temperature, the
context window it was trained for, and the specialty tags it advertises.
Calibration merges these into the model card's `choices` (only for LOCAL
models, and `apply()` only fills settings the user left at their defaults),
so a freshly imported Qwen-coder or Gemma starts from sensible numbers
instead of the generic tier preset.

Each entry carries two temperature pairs: the non-thinking pair
(`temperature` / `temperature_answer`, used for ordinary, JSON-tool-call
runs) and the thinking pair (`thinking_temperature` /
`thinking_temperature_answer`, used when the model keeps its reasoning
blocks on). `recommended_temperatures()` picks the right pair.

Matching is by model-id substring, case-insensitive, with `-`/`_` folded to
`.` on both sides so `Qwen2.5-Coder-32B`, `qwen2_5_coder` and
`qwen2.5-coder` all land on the same entry. The library is ordered
specific-first: the coder variants precede the base family they contain.
"""

from __future__ import annotations

# Each entry: the id fragments that select it, the recommended sampler and
# window, and the specialty tags the family is good at. `family` is the
# human-readable label recorded on the card as `arch_family`.
ARCH_LIBRARY: tuple[dict, ...] = (
    # Temperatures stay warm enough to escape degenerate verbatim loops:
    # logic families answer at 0.3-0.4, the creative/frontend families at
    # 0.6. Cold 0.0-0.1 answers froze weak local models into repetition;
    # a retry nudge (presence_penalty 0.15) handles a loop that still forms.
    # Reasoning (thinking-on) runs get warmer temperatures, since the hidden
    # thought block already spends budget and a cold sampler cuts it short.
    {
        "family": "qwen2.5-coder",
        "patterns": ("qwen2.5-coder", "qwen3-coder"),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.5, "thinking_temperature_answer": 0.5,
        "sampler": {"top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05},
        "context_tokens": 32768,
        "specialties": ("coding", "testing_qa"),
    },
    {
        "family": "qwen3.5",
        "patterns": ("qwen3.5.", "qwen3.8.", "qwen4"),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.6, "thinking_temperature_answer": 0.6,
        "sampler": {"top_p": 0.95, "top_k": 20},
        "context_tokens": 32768,
        "specialties": ("general", "coding"),
    },
    {
        "family": "qwen3",
        "patterns": ("qwen3",),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.6, "thinking_temperature_answer": 0.6,
        "sampler": {"top_p": 0.8, "top_k": 20},
        "context_tokens": 32768,
        "specialties": ("general", "coding"),
    },
    {
        "family": "deepseek-coder",
        "patterns": ("deepseek-coder",),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.5, "thinking_temperature_answer": 0.5,
        "sampler": {"top_p": 0.95},
        "context_tokens": 32768,
        "specialties": ("coding", "testing_qa"),
    },
    {
        "family": "deepseek",
        "patterns": ("deepseek-r1", "deepseek-v3", "deepseek-v4", "qwq"),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.6, "thinking_temperature_answer": 0.6,
        "sampler": {"top_p": 0.95},
        "context_tokens": 32768,
        "specialties": ("math_algorithms",),
    },
    {
        "family": "glm",
        "patterns": ("glm-4", "glm-5"),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.6, "thinking_temperature_answer": 0.6,
        "sampler": {"top_p": 0.95},
        "context_tokens": 32768,
        "specialties": ("general",),
    },
    {
        "family": "gemma",
        "patterns": ("gemma-3", "gemma-4"),
        "temperature": 0.6, "temperature_answer": 0.6,
        "thinking_temperature": 0.7, "thinking_temperature_answer": 0.7,
        "sampler": {"top_p": 0.95, "top_k": 64},
        "context_tokens": 32768,
        "specialties": ("general", "frontend_ui", "3d_graphics_svg",
                        "translation_i18n"),
    },
    {
        "family": "gpt-oss",
        "patterns": ("gpt-oss",),
        "temperature": 0.4, "temperature_answer": 0.4,
        "thinking_temperature": 0.6, "thinking_temperature_answer": 0.6,
        "sampler": {"top_p": 0.95},
        "context_tokens": 32768,
        "specialties": ("general",),
    },
    {
        "family": "mistral",
        "patterns": ("codestral", "devstral", "ministral", "mistral"),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.5, "thinking_temperature_answer": 0.5,
        "sampler": {"top_p": 0.9},
        "context_tokens": 32768,
        "specialties": ("coding", "general"),
    },
    {
        "family": "llama",
        "patterns": ("llama-3", "llama-4"),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.5, "thinking_temperature_answer": 0.5,
        "sampler": {"top_p": 0.9},
        "context_tokens": 32768,
        "specialties": ("general", "coding"),
    },
    {
        "family": "nemotron",
        "patterns": ("nemotron",),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.5, "thinking_temperature_answer": 0.5,
        "sampler": {"top_p": 0.9},
        "context_tokens": 32768,
        "specialties": ("general",),
    },
    {
        "family": "granite",
        "patterns": ("granite",),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.5, "thinking_temperature_answer": 0.5,
        "sampler": {"top_p": 0.9},
        "context_tokens": 16384,
        "specialties": ("coding", "sql_data"),
    },
    {
        "family": "phi-4",
        "patterns": ("phi-4",),
        "temperature": 0.3, "temperature_answer": 0.4,
        "thinking_temperature": 0.5, "thinking_temperature_answer": 0.5,
        "sampler": {"top_p": 0.9},
        "context_tokens": 16384,
        "specialties": ("math_algorithms", "coding"),
    },
    {
        "family": "lfm",
        "patterns": ("lfm",),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.5, "thinking_temperature_answer": 0.5,
        "sampler": {"top_p": 0.9},
        "context_tokens": 32768,
        "specialties": ("general",),
    },
)


def _norm(text: str) -> str:
    return str(text or "").lower().replace("_", ".").replace("-", ".")


def match_arch(model_id: str) -> dict | None:
    """The library entry whose id fragment appears in `model_id`, or None.
    Specific-first: `qwen3-coder` matches the coder entry, not base qwen3."""
    name = _norm(model_id)
    if not name:
        return None
    for entry in ARCH_LIBRARY:
        if any(_norm(p) in name for p in entry["patterns"]):
            return entry
    return None


def recommended_temperatures(entry: dict, *, thinking_enabled: bool = False,
                             tier: str = "") -> tuple[float, float, dict]:
    """(temperature, temperature_answer, sampler) for a family entry.

    Selects the thinking vs non-thinking pair from the entry, and for tiny /
    small tiers with thinking OFF caps the gather temperature at <= 0.3 so
    strict JSON/tool calls stay reliable while the answer keeps its warmth.
    """
    if thinking_enabled:
        temp = entry.get("thinking_temperature", entry.get("temperature", 0.3))
        answer = entry.get("thinking_temperature_answer",
                           entry.get("temperature_answer", 0.3))
    else:
        temp = entry.get("temperature", 0.3)
        answer = entry.get("temperature_answer", 0.3)
        if str(tier or "") in ("tiny", "small"):
            temp = min(float(temp), 0.3)
    sampler = dict(entry.get("sampler") or {})
    return float(temp), float(answer), sampler


def arch_context_tokens(entry: dict, info: dict, vram_gb: int = 0,
                        size_bytes: int = 0) -> int:
    """The recommended window capped by what the machine can actually hold:
    the server's reported context length and the VRAM KV-cache estimate
    (the same `auto_fit` window). A missing signal simply does not cap."""
    from .config import hardware_context_tokens
    ctx = int(entry.get("context_tokens") or 8192)
    server = int((info or {}).get("context_length")
                 or (info or {}).get("max_context_length") or 0)
    if server > 0:
        ctx = min(ctx, server)
    if vram_gb and size_bytes:
        fits = hardware_context_tokens(int(vram_gb), int(size_bytes))
        if fits > 0:
            ctx = min(ctx, fits)
    return max(1024, ctx)


def needle_step_down(window: int) -> int | None:
    """The next smaller window to retry the long-context needle at after a
    miss (>16k -> 16384, then -> 8192); None when there is nothing smaller."""
    if window > 16384:
        return 16384
    if window > 8192:
        return 8192
    return None
