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

import os
import struct
from pathlib import Path

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
        "family": "starcoder",
        "patterns": ("starcoder", "dolphincoder"),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.5, "thinking_temperature_answer": 0.5,
        "sampler": {"top_p": 0.95},
        "context_tokens": 16384,
        "specialties": ("coding",),
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
        "family": "lfm2.5-24b-a2b",
        "patterns": ("lfm2.5-24b-a2b", "lfm-2.5-24b-a2b", "lfm2.5.24b"),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.6, "thinking_temperature_answer": 0.6,
        "sampler": {"top_p": 0.9, "min_p": 0.15, "repeat_penalty": 1.05},
        "context_tokens": 32768,
        "specialties": ("general", "coding"),
    },
    {
        "family": "lfm3",
        "patterns": ("lfm3", "lfm-3"),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.6, "thinking_temperature_answer": 0.6,
        "sampler": {"top_p": 0.9, "repeat_penalty": 1.05},
        "context_tokens": 32768,
        "specialties": ("general", "coding"),
    },
    {
        "family": "lfm",
        "patterns": ("lfm2.5", "lfm2", "lfm"),
        "temperature": 0.3, "temperature_answer": 0.3,
        "thinking_temperature": 0.5, "thinking_temperature_answer": 0.5,
        "sampler": {"top_p": 0.9, "min_p": 0.15, "repeat_penalty": 1.05},
        "context_tokens": 32768,
        "specialties": ("general",),
    },
)


# GGUF `general.architecture` (and Ollama/LM Studio `family` / `arch`) slugs
# mapped to an ARCH_LIBRARY `family` name. Used when a custom fine-tune's model
# id (e.g. `DaisyRose-27B-GGUF`) does not carry its base family name, but the
# runtime or GGUF header identifies the underlying architecture.
_GGUF_ARCH_MAP: dict[str, str] = {
    "qwen35": "qwen3.5",
    "qwen35moe": "qwen3.5",
    "qwen3.5": "qwen3.5",
    "qwen3": "qwen3",
    "qwen3moe": "qwen3",
    "qwen3next": "qwen3.5",
    "qwen2": "qwen3",
    "qwen2moe": "qwen3",
    "qwen2vl": "qwen3",
    "qwen25vl": "qwen3",
    "gemma4": "gemma",
    "gemma3": "gemma",
    "gemma3n": "gemma",
    "gemma2": "gemma",
    "gemma": "gemma",
    "deepseek2": "deepseek",
    "deepseek": "deepseek",
    "glm4": "glm",
    "glm4moe": "glm",
    "chatglm": "glm",
    "gptoss": "gpt-oss",
    "gpt.oss": "gpt-oss",
    "mistral": "mistral",
    "mistral3": "mistral",
    "mixtral": "mistral",
    "llama": "llama",
    "llama4": "llama",
    "nemotron": "nemotron",
    "nemotron.h": "nemotron",
    "nemotronh": "nemotron",
    "starcoder2": "starcoder",
    "starcoder": "starcoder",
    "granite": "granite",
    "granitemoe": "granite",
    "granitehybrid": "granite",
    "phi3": "phi-4",
    "phi2": "phi-4",
    "phimoe": "phi-4",
    "lfm3": "lfm3",
    "lfm2": "lfm",
    "lfm2moe": "lfm",
}

# Generic runtime wrappers that carry no model-family signal.
_GENERIC_ARCH_TAGS = frozenset((
    "", "llamacpp", "llama.cpp", "gguf", "onnx", "mlx", "transformers",
    "unknown", "custom", "auto",
))

_GGUF_WANT_KEYS = frozenset((
    "general.architecture",
    "general.basename",
    "general.name",
    "general.base_model.0.name",
    "general.finetune",
))

# GGUF scalar value type -> byte width (v2/v3 spec). Type 8 = string, 9 = array.
_GGUF_SCALAR_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}
_GGUF_MAX_KV = 256
_GGUF_MAX_STR = 4096
_GGUF_MAX_ARRAY_SKIP = 1_000_000


def _norm(text: str) -> str:
    return str(text or "").lower().replace("_", ".").replace("-", ".")


def _entry_by_family(family: str) -> dict | None:
    for entry in ARCH_LIBRARY:
        if entry["family"] == family:
            return entry
    return None


def _match_text(text: str) -> dict | None:
    name = _norm(text)
    if not name or name in _GENERIC_ARCH_TAGS:
        return None
    # Exact GGUF/runtime architecture slugs win first (e.g. "qwen35" must map
    # to "qwen3.5", not fall into the broader "qwen3" substring pattern).
    compact = name.replace(".", "")
    mapped = _GGUF_ARCH_MAP.get(name) or _GGUF_ARCH_MAP.get(compact)
    if mapped:
        return _entry_by_family(mapped)
    for entry in ARCH_LIBRARY:
        if any(_norm(p) in name for p in entry["patterns"]):
            return entry
    return None


def read_gguf_metadata(path: str | Path) -> dict[str, str]:
    """Read base-model identity keys from a local GGUF v2/v3 file header.

    Pure stdlib (`struct`), reads only header metadata (never weights), and
    bounds string/KV/array sizes so a corrupt or truncated file is always a
    fast silent miss returning `{}`."""
    p = Path(path)
    try:
        if not p.is_file():
            return {}
        out: dict[str, str] = {}
        with p.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                return {}
            hdr = fh.read(20)
            if len(hdr) < 20:
                return {}
            ver, _n_tensors, n_kv = struct.unpack("<IQQ", hdr)
            if ver not in (2, 3):
                return {}

            def read_u64() -> int:
                raw = fh.read(8)
                if len(raw) < 8:
                    raise ValueError("truncated u64")
                return struct.unpack("<Q", raw)[0]

            def read_str(capture: bool = True) -> str:
                n = read_u64()
                if n > _GGUF_MAX_STR:
                    if capture:
                        raise ValueError("oversized gguf string")
                    fh.seek(n, 1)
                    return ""
                raw = fh.read(n)
                if len(raw) < n:
                    raise ValueError("truncated gguf string")
                return raw.decode("utf-8", "replace") if capture else ""

            def skip_val(vtype: int) -> None:
                width = _GGUF_SCALAR_SIZES.get(vtype)
                if width is not None:
                    fh.seek(width, 1)
                    return
                if vtype == 8:
                    n = read_u64()
                    fh.seek(n, 1)
                    return
                if vtype == 9:
                    raw = fh.read(12)
                    if len(raw) < 12:
                        raise ValueError("truncated gguf array")
                    elem_type, count = struct.unpack("<IQ", raw)
                    if count > _GGUF_MAX_ARRAY_SKIP:
                        raise ValueError("oversized gguf array")
                    elem_w = _GGUF_SCALAR_SIZES.get(elem_type)
                    if elem_w is not None:
                        fh.seek(elem_w * count, 1)
                        return
                    for _ in range(count):
                        skip_val(elem_type)
                    return
                raise ValueError(f"unknown gguf type {vtype}")

            for _ in range(min(int(n_kv), _GGUF_MAX_KV)):
                key = read_str(capture=True)
                raw_vt = fh.read(4)
                if len(raw_vt) < 4:
                    break
                vtype = struct.unpack("<I", raw_vt)[0]
                if key in _GGUF_WANT_KEYS and vtype == 8:
                    val = read_str(capture=True).strip()
                    if val:
                        out[key] = val
                    if len(out) >= len(_GGUF_WANT_KEYS):
                        break
                else:
                    skip_val(vtype)
        return out
    except (OSError, ValueError, struct.error):
        return {}


def locate_gguf_for_checkpoint(checkpoint: str, *,
                               extra_dirs=()) -> Path | None:
    """Locate a local `.gguf` file for a HuggingFace/Lemonade checkpoint string
    like `RadixArk/DaisyRose-27B-GGUF:Q4_K_XL` or a direct local `.gguf` path.

    Checks only bounded HuggingFace / LM Studio / Lemonade cache directories
    (never recursive full-disk scans). Returns None when not found."""
    raw = str(checkpoint or "").strip()
    if not raw:
        return None
    direct = Path(raw).expanduser()
    if direct.suffix.lower() == ".gguf" and direct.is_file():
        return direct
    repo, _, quant = raw.partition(":")
    repo = repo.strip().strip("/")
    quant = quant.strip().lower()
    if "/" not in repo:
        return None
    org, _, name = repo.partition("/")
    if not org or not name:
        return None
    hub_dir_name = f"models--{org}--{name}"

    roots: list[Path] = []
    for d in extra_dirs or ():
        if d:
            roots.append(Path(d).expanduser())
    for env_key, sub in (
        ("HF_HUB_CACHE", ""),
        ("HF_HOME", "hub"),
        ("LEMONADE_CACHE_DIR", ""),
        ("LEMONADE_CACHE_DIR", "hub"),
    ):
        val = os.environ.get(env_key, "").strip()
        if val:
            p = Path(val).expanduser()
            roots.append(p / sub if sub else p)
    home = Path.home()
    roots.append(home / ".cache" / "huggingface" / "hub")
    roots.append(home / ".cache" / "lm-studio" / "models")

    seen: set[str] = set()
    candidates: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        # HuggingFace hub layout: <root>/models--Org--Repo/snapshots/*/*.gguf
        snap_root = root / hub_dir_name / "snapshots"
        try:
            if snap_root.is_dir():
                for g in sorted(snap_root.glob("*/*.gguf")):
                    if "mmproj" not in g.name.lower() and g.is_file():
                        candidates.append(g)
        except OSError:
            pass
        # LM Studio / direct repo layout: <root>/Org/Repo/*.gguf
        direct_repo = root / org / name
        try:
            if direct_repo.is_dir():
                for g in sorted(direct_repo.glob("*.gguf")):
                    if "mmproj" not in g.name.lower() and g.is_file():
                        candidates.append(g)
        except OSError:
            pass
    if not candidates:
        return None
    if quant:
        for g in candidates:
            if quant in g.name.lower():
                return g
    return candidates[0]


def match_arch(model_id: str, info: dict | None = None) -> dict | None:
    """The library entry matching `model_id` (or GGUF/runtime metadata in
    `info`), or None.

    Resolution order (specific-first):
      1. `model_id` itself (e.g. `qwen3-coder` matches coder, not base qwen3).
      2. Server-reported metadata in `info` (`base_model`, `basename`,
         `architecture`, `family`, `families`) -- covers Ollama `/api/show`,
         LM Studio `arch`, and llama-server `meta`.
      3. Local GGUF header metadata from `info["model_path"]` or a local cache
         hit for `info["checkpoint"]` (Lemonade fine-tunes like DaisyRose-27B).
    """
    hit = _match_text(model_id)
    if hit is not None:
        return hit
    if not isinstance(info, dict) or not info:
        return None
    # Check specific base-model names before coarse GGUF architecture slugs.
    candidates: list[str] = []
    for key in ("base_model", "basename", "architecture", "family"):
        val = info.get(key)
        if val:
            candidates.append(str(val))
    for fam in info.get("families") or ():
        if fam:
            candidates.append(str(fam))
    for cand in candidates:
        hit = _match_text(cand)
        if hit is not None:
            return hit

    # Last resort: read GGUF header metadata from a local path or checkpoint.
    gguf_path: Path | None = None
    raw_path = str(info.get("model_path") or "").strip()
    if raw_path:
        p = Path(raw_path).expanduser()
        if p.is_file():
            gguf_path = p
    if gguf_path is None and info.get("checkpoint"):
        gguf_path = locate_gguf_for_checkpoint(str(info["checkpoint"]))
    if gguf_path is not None:
        meta = read_gguf_metadata(gguf_path)
        for key in ("general.base_model.0.name", "general.basename",
                    "general.name", "general.architecture"):
            val = meta.get(key)
            if val:
                hit = _match_text(val)
                if hit is not None:
                    return hit
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
                        size_bytes: int = 0, *,
                        is_local_moe: bool = False) -> int:
    """The recommended window capped by what the machine can actually hold:
    the server's reported context length and the VRAM KV-cache estimate
    (the same `auto_fit` window). A missing signal simply does not cap.

    For tiered `local-moe` servers (Strata/Avalon), weights spill across DDR5
    and the server explicitly advertises its configured `max_context` (often
    64k–131k): trust the server's reported window instead of shrinking it down
    to a dense-GPU 32k architecture preset or GPU-only VRAM cap."""
    from .config import hardware_context_tokens
    ctx = int(entry.get("context_tokens") or 8192)
    server = int((info or {}).get("context_length")
                 or (info or {}).get("max_context_length") or 0)
    if is_local_moe:
        if server > 0:
            return max(1024, server)
        return max(1024, ctx)
    if server > 0:
        ctx = min(ctx, server)
    if vram_gb and size_bytes:
        fits = hardware_context_tokens(int(vram_gb), int(size_bytes))
        if fits > 0:
            ctx = min(ctx, fits)
    return max(1024, ctx)


def needle_step_down(window: int) -> int | None:
    """The next smaller window to retry the long-context needle at after a
    miss (>64k -> 65536, >32k -> 32768, >16k -> 16384, then -> 8192);
    None when there is nothing smaller."""
    if window > 65536:
        return 65536
    if window > 32768:
        return 32768
    if window > 16384:
        return 16384
    if window > 8192:
        return 8192
    return None
