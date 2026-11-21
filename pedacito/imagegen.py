"""Local image generation: detection, calibration support, and the agent tool.

    from .imagegen import generate_image, sniff_image_format, is_image_model

    result = generate_image(cfg, "a lighthouse at dusk", size="1024x1024")

Image generation is a *discovery* modality, not a chat model: the same
OpenAI-compatible surface that serves `/chat/completions` can also serve
`/images/generations` (Lemonade's `sd-cpp` recipes, Stable Diffusion, Flux, and
the hosted DALL-E / GPT-Image endpoints). This module detects which configured
profile is an image generator (by `modality`, server `labels`/`recipe`
metadata, or the model id), posts a JSON generation request, decodes the
`b64_json` (or fetches the returned `url`), sniffs the pixel format and
dimensions without Pillow, and writes the bytes atomically into the workspace.

Zero new runtime dependencies: stdlib plus the `requests` already used by
`llm.py`. No Pillow.
"""

from __future__ import annotations

import base64
import copy
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import fileio, security


class ImageGenError(RuntimeError):
    """Image generation failed, or no image-generation profile is configured."""


# Model-id fragments that mark an image GENERATOR. Kept separate from the
# vision-language chat models below: those take images in, they do not emit
# them, so they must never be offered a generate_image tool.
_IMAGE_MODEL_MARKERS = (
    "sd-turbo", "sdxl", "sd-1.5", "sd-2.1", "sd3", "stable-diffusion",
    "flux", "z-image", "qwen-image", "lsdir", "dall-e", "gpt-image",
    "imagen", "kolors", "auraflow", "pixart", "kandinsky", "hidream",
    "cogview", "hunyuan-dit",
)

# Vision-LANGUAGE chat models: they understand images, they do not produce
# them. These win over any generator marker so a `gemma-3`/`qwen-vl` profile is
# never mislabelled as an image generator.
_VISION_LANG_MARKERS = (
    "-vl", "qwen-vl", "minicpm-o", "llava", "pixtral", "internvl",
    "gemma-3", "gemma-4",
)

# Recipe spellings Lemonade reports for its image backends.
_IMAGE_RECIPES = frozenset({
    "sd-cpp", "sdcpp", "stable-diffusion", "thenoise", "diffusers", "flux",
})

# Non-image modalities that close the question even without server metadata.
_NON_IMAGE_MODALITIES = frozenset({"text", "audio", "embedding"})


def is_image_model(model_id: str, *, labels=None, recipe=None,
                   modality=None) -> bool:
    """Whether `model_id` is an image GENERATOR (not a vision chat model).

    Decided in order: an explicit `modality` wins outright ("image" -> True,
    a known non-image modality -> False); then server metadata (a lowercase
    `labels` entry of "image", or a `recipe` in the known image set); then the
    model-id heuristics, which first refuse the vision-language chat families.
    """
    if modality is not None:
        mod = str(modality).strip().lower()
        if mod == "image":
            return True
        if mod in _NON_IMAGE_MODALITIES:
            return False

    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        for label in labels or ():
            if str(label).strip().lower() == "image":
                return True

    if recipe is not None:
        if str(recipe).strip().lower() in _IMAGE_RECIPES:
            return True

    name = str(model_id or "").lower()
    # Vision-language chat models understand images, they do not generate them.
    if any(vl in name for vl in _VISION_LANG_MARKERS):
        return False
    return any(m in name for m in _IMAGE_MODEL_MARKERS)


def profile_is_image_model(cfg, name: str | None = None) -> bool:
    """True when profile `name` (the active profile when None) is an image
    generator, from its `modality` and `model` (or the top-level config)."""
    if name is None:
        name = str(getattr(cfg, "active_profile", "") or "")
    profiles = getattr(cfg, "profiles", None) or {}
    body = profiles.get(name) if isinstance(profiles, dict) else None
    if isinstance(body, dict):
        modality = str(body.get("modality", "") or "").strip()
        model = str(body.get("model", "") or "")
        if modality or model:
            return is_image_model(model, modality=modality or None)
    modality = str(getattr(cfg, "modality", "") or "").strip()
    model = str(getattr(cfg, "model", "") or "")
    return is_image_model(model, modality=modality or None)


def _profile_cfg(cfg, name: str):
    """A Config copy with profile `name` applied (via the roles helper, so the
    tier preset, coercion and hosted-ness are re-derived)."""
    try:
        from . import roles
        derived = roles.derive_config(cfg, name)
        if derived is not None:
            return derived
    except Exception:
        pass
    new = copy.copy(cfg)
    body = (getattr(cfg, "profiles", None) or {}).get(name) or {}
    if isinstance(body, dict):
        for key, value in body.items():
            setattr(new, key, value)
    new.active_profile = name
    return new


def _find_image_profile_name(cfg, explicit: str | None = None) -> str | None:
    """The image profile's NAME (no config derivation, no network)."""
    profiles = getattr(cfg, "profiles", None) or {}
    if explicit:
        name = str(explicit).strip()
        return name if name and name in profiles else None
    image_profile = str(getattr(cfg, "image_profile", "") or "").strip()
    if image_profile and image_profile in profiles:
        return image_profile
    for name in profiles:
        if profile_is_image_model(cfg, name):
            return name
    return None


def has_image_profile(cfg) -> bool:
    """True when an image profile is configured (cheap, safe for `schema_for`):
    a named image profile exists, or the active config itself is one."""
    return _find_image_profile_name(cfg) is not None \
        or profile_is_image_model(cfg)


def find_image_profile(cfg, explicit: str | None = None):
    """(profile_name, Config) for the image profile to use, or (None, None).

    Resolution order: the explicit name -> `cfg.image_profile` -> the first
    profile in `cfg.profiles` that is an image model -> the active config
    itself when it already IS an image model.
    """
    name = _find_image_profile_name(cfg, explicit)
    if name is not None:
        return name, _profile_cfg(cfg, name)
    # No named profile: the active config may itself be an image model.
    if profile_is_image_model(cfg):
        return str(getattr(cfg, "active_profile", "") or "") or None, cfg
    return None, None


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a JPEG's SOF marker, or (0, 0)."""
    n = len(data)
    i = 2                                   # skip SOI
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:                  # fill byte between markers
            i += 1
            continue
        if marker in (0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7):
            i += 2                          # standalone marker, no length
            continue
        if marker in (0xD8, 0xD9):
            i += 2
            continue
        if i + 4 > n:
            break
        seglen = int.from_bytes(data[i + 2:i + 4], "big")
        if seglen < 2:
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 9 > n:
                break
            h = int.from_bytes(data[i + 5:i + 7], "big")
            w = int.from_bytes(data[i + 7:i + 9], "big")
            return w, h
        i += 2 + seglen
    return 0, 0


def _webp_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a WebP container (VP8X, VP8L or VP8), or (0, 0)."""
    n = len(data)
    if n < 16:
        return 0, 0
    fourcc = data[12:16]
    if fourcc == b"VP8X" and n >= 30:
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
        return w, h
    if fourcc == b"VP8L" and n >= 25:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if fourcc == b"VP8 " and n >= 26:
        bits = int.from_bytes(data[23:26], "little")
        return bits & 0x3FFF, (bits >> 14) & 0x3FFF
    return 0, 0


def sniff_image_format(data: bytes) -> tuple[str, int, int]:
    """(format, width, height) for PNG, JPEG or WebP bytes, else
    ("bin", 0, 0). Pure magic-byte + header parsing, no Pillow."""
    data = bytes(data or b"")
    if len(data) < 8:
        return "bin", 0, 0
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if len(data) >= 24:
            return "png", int.from_bytes(data[16:20], "big"), \
                int.from_bytes(data[20:24], "big")
        return "png", 0, 0
    if data[:2] == b"\xff\xd8":
        w, h = _jpeg_size(data)
        return "jpg", w, h
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        w, h = _webp_size(data)
        return "webp", w, h
    return "bin", 0, 0


def images_endpoint(base_url: str) -> str:
    """`<base_url>/images/generations`, normalised for `/v1` and `/api/v1`
    roots (and idempotent when the path is already the full endpoint)."""
    base = str(base_url or "").strip().rstrip("/")
    if base.endswith("/images/generations"):
        return base
    return base + "/images/generations"


def _fetch_image_bytes(url: str, base_url: str, timeout: float) -> bytes:
    """Fetch an image `url` returned by the endpoint, loopback-safe.

    The image server may hand back a URL on its own host (loopback or LAN), so
    that host is always allowed; a different loopback/private host is refused
    (an image response must not become an SSRF probe of the user's network).
    Public hosts are reached normally. Returns b"" on any refusal/failure."""
    raw = str(url or "").strip().strip("`\"'")
    if not raw:
        return b""
    try:
        host = (urlparse(raw).hostname or "").lower()
        base_host = (urlparse(str(base_url or "")).hostname or "").lower()
    except ValueError:
        return b""
    if not host or (urlparse(raw).scheme or "").lower() not in ("http", "https"):
        return b""
    if host != base_host and security.is_private_host(host):
        return b""                          # SSRF guard: never a private host
    try:
        resp = requests.get(raw, timeout=timeout,
                            headers={"User-Agent": "pedacito-imagegen/1.0"})
    except requests.exceptions.RequestException:
        return b""
    if getattr(resp, "status_code", 0) != 200:
        return b""
    content = resp.content
    resp.close()
    return bytes(content or b"")


_SETUP_MSG = (
    "No image-generation profile is configured. Add one to your config with "
    "`modality = \"image\"`, for example:\n\n"
    "  [profiles.flux]\n"
    "  base_url = \"http://localhost:8001/v1\"\n"
    "  model = \"flux-schnell\"\n"
    "  modality = \"image\"\n\n"
    "Then set `image_profile = \"flux\"` (or select it with --profile). Image "
    "generation is opt-in: set `allow_image_gen = true` to expose the "
    "generate_image tool to the agent."
)


def _slug(prompt: str) -> str:
    """A filename-safe slug of the prompt, for the default output name."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(prompt or "").lower()).strip("-")
    return s[:40] or "image"


def generate_image(cfg, prompt: str, *, output_path=None, size="", steps=0,
                   cfg_scale=None, seed=None, negative_prompt="", profile=None,
                   root=None, timeout=None) -> dict:
    """Generate one image and write it into the workspace (or `output_path`).

    Returns `{ok, path, rel_path, format, width, height, bytes, elapsed_s,
    model, profile, base_url, prompt}`. Raises `ImageGenError` when no image
    profile is configured or the endpoint returns no image; a path outside
    `root` raises `security.SecurityError`.
    """
    from .llm import _api_key, _is_local_server

    profile_name, pcfg = find_image_profile(cfg, explicit=profile)
    if pcfg is None:
        raise ImageGenError(_SETUP_MSG)

    if root is None:
        root = getattr(cfg, "_root", None) or Path.cwd()
    resolved_root = Path(root).expanduser().resolve()

    # A local image server shares the GPU with the chat server(s): evict any
    # other resident local model first (best-effort, never fatal).
    if getattr(pcfg, "auto_evict_servers", True):
        try:
            if _is_local_server(pcfg):
                from . import servers
                servers.evict_other_local_servers(
                    str(getattr(pcfg, "active_profile", "") or "") or pcfg.base_url,
                    pcfg)
        except Exception:
            pass

    model = str(getattr(pcfg, "model", "") or "").strip()
    base_url = str(getattr(pcfg, "base_url", "") or "").strip()
    if not model or not base_url:
        raise ImageGenError(
            "The image profile is missing a model id or base_url. Set both in "
            f"the [profiles.{profile_name}] table (model and base_url).")

    size = str(size or "").strip() \
        or str(getattr(pcfg, "image_default_size", "") or "").strip()
    steps = int(steps or 0) or int(getattr(pcfg, "image_default_steps", 0) or 0)
    timeout = float(timeout) if timeout else \
        float(getattr(pcfg, "timeout", 0) or 120.0)

    endpoint = images_endpoint(base_url)
    headers = {"Content-Type": "application/json"}
    key = _api_key(pcfg)
    if key:
        headers["Authorization"] = f"Bearer {key}"

    body: dict = {"model": model, "prompt": str(prompt or ""),
                  "n": 1, "response_format": "b64_json"}
    if size:
        body["size"] = size
    if steps:
        body["steps"] = steps
    if cfg_scale is not None:
        body["cfg_scale"] = cfg_scale
    if seed is not None:
        body["seed"] = seed
    if negative_prompt:
        body["negative_prompt"] = negative_prompt

    t0 = time.time()

    def _post(payload: dict):
        return requests.post(endpoint, headers=headers, json=payload,
                             timeout=timeout)

    resp = _post(body)
    if resp.status_code in (400, 422) and any(
            k in body for k in ("steps", "cfg_scale", "seed",
                                "negative_prompt", "response_format")):
        # A server that rejects the local-only keys gets one stripped retry.
        stripped = {k: v for k, v in body.items()
                    if k not in ("steps", "cfg_scale", "seed",
                                 "negative_prompt", "response_format")}
        resp = _post(stripped)

    if resp.status_code != 200:
        try:
            detail = str(resp.text or "")[:300]
        except Exception:
            detail = ""
        raise ImageGenError(
            f"image endpoint returned HTTP {resp.status_code} "
            f"for '{model}': {detail}")

    try:
        payload = resp.json()
    except Exception:
        raise ImageGenError("image endpoint returned a non-JSON response")

    items = (payload or {}).get("data") or []
    item = items[0] if isinstance(items, list) and items else {}
    image_bytes = b""
    b64 = item.get("b64_json") if isinstance(item, dict) else None
    if b64:
        try:
            image_bytes = base64.b64decode(str(b64))
        except (ValueError, TypeError):
            image_bytes = b""
    if not image_bytes and isinstance(item, dict) and item.get("url"):
        image_bytes = _fetch_image_bytes(item.get("url"), base_url, timeout)

    if not image_bytes:
        raise ImageGenError(
            "image endpoint returned no image data (no b64_json and no "
            "fetchable url)")

    fmt, width, height = sniff_image_format(image_bytes)

    if output_path:
        dest = security.resolve_safe_path(resolved_root, str(output_path))
    else:
        ext = {"png": "png", "jpg": "jpg", "webp": "webp"}.get(fmt, "bin")
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{_slug(str(prompt or ''))}.{ext}"
        dest = security.resolve_safe_path(
            resolved_root, str(Path(".pedacito") / "images" / name))
    rel = dest.relative_to(resolved_root).as_posix()

    fileio.write_bytes(dest, image_bytes)

    return {
        "ok": True,
        "path": str(dest),
        "rel_path": rel,
        "format": fmt,
        "width": width,
        "height": height,
        "bytes": len(image_bytes),
        "elapsed_s": round(time.time() - t0, 3),
        "model": model,
        "profile": profile_name or "",
        "base_url": base_url,
        "prompt": str(prompt or ""),
    }
