"""WP-07: local image generation core, model detection, calibration, and tool.

Hermetic: minimal hand-built PNG/JPEG/WebP headers for the byte sniffer, and
threaded fake HTTP servers for the generation request -- no Pillow, no real
model server.
"""
from __future__ import annotations

import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import imagegen, modelcard, security, tools
from pedacito.config import Config
from pedacito.local_server import LocalHTTPServer


# ----------------------------------------------------------- image headers ----

def _png(width=800, height=600) -> bytes:
    data = b"\x89PNG\r\n\x1a\n"
    data += (13).to_bytes(4, "big") + b"IHDR"
    data += width.to_bytes(4, "big") + height.to_bytes(4, "big")
    data += bytes([8, 2, 0, 0, 0])           # bit depth, colour type, ..., ...
    data += b"\x00\x00\x00\x00"             # a fake CRC the sniffer ignores
    return data


def _jpeg(width=400, height=300) -> bytes:
    data = b"\xff\xd8"                       # SOI
    data += (b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00"
             + bytes([1, 1, 0]) + bytes([0, 1]) + bytes([0, 1]) + bytes([0, 0]))
    data += b"\xff\xc0" + (17).to_bytes(2, "big") + bytes([8])  # SOF0
    data += height.to_bytes(2, "big") + width.to_bytes(2, "big")
    data += bytes([3]) + bytes([0] * 9)      # 3 components x 3 bytes
    return data


def _webp(width=640, height=480) -> bytes:
    data = b"RIFF" + (18).to_bytes(4, "little") + b"WEBP" + b"VP8X"
    data += (10).to_bytes(4, "little")       # chunk size
    data += bytes([0])                       # flags
    data += bytes([0, 0, 0])                 # reserved
    data += (width - 1).to_bytes(3, "little")
    data += (height - 1).to_bytes(3, "little")
    return data


# ----------------------------------------------------------- is_image_model ---

def test_is_image_model_explicit_modality_wins():
    assert imagegen.is_image_model("anything", modality="image") is True
    assert imagegen.is_image_model("flux-schnell", modality="text") is False
    assert imagegen.is_image_model("sd-turbo", modality="audio") is False
    assert imagegen.is_image_model("sd-turbo", modality="embedding") is False


def test_is_image_model_server_metadata():
    assert imagegen.is_image_model("model-a", labels=["image"]) is True
    assert imagegen.is_image_model("model-a", labels=["image", "text"]) is True
    assert imagegen.is_image_model("model-a", labels="image") is True
    assert imagegen.is_image_model("model-a", recipe="sd-cpp") is True
    assert imagegen.is_image_model("model-a", recipe="sdcpp") is True
    assert imagegen.is_image_model("model-a", recipe="stable-diffusion") is True
    assert imagegen.is_image_model("model-a", recipe="diffusers") is True
    assert imagegen.is_image_model("model-a", recipe="flux") is True
    assert imagegen.is_image_model("model-a", recipe="thenoise") is True
    assert imagegen.is_image_model("model-a", recipe="llama") is False
    assert imagegen.is_image_model("model-a", labels=["text"]) is False


@pytest.mark.parametrize("mid", [
    "sd-turbo", "stabilityai/sdxl-turbo", "sd-1.5", "sd-2.1", "sd3-medium",
    "stable-diffusion-xl", "flux-schnell", "z-image-turbo", "qwen-image",
    "lsdir-v1", "dall-e-3", "gpt-image-1", "imagen-3", "kolors",
    "auraflow-v0.3", "pixart-alpha", "kandinsky-3", "hidream-i1",
    "cogview-3", "hunyuan-dit",
])
def test_is_image_model_id_heuristics(mid):
    assert imagegen.is_image_model(mid) is True


@pytest.mark.parametrize("mid", [
    "qwen-vl-max", "Qwen2.5-VL-7B", "minicpm-o-2.6", "llava-1.6",
    "pixtral-12b", "internvl2-8b", "gemma-3-27b", "gemma-4-12b",
    "google/gemma-3-27b",
])
def test_is_image_model_vision_chat_models_are_not_generators(mid):
    assert imagegen.is_image_model(mid) is False


# --------------------------------------------------------- sniff_image_format -

def test_sniff_image_format_png_jpeg_webp_and_bin():
    assert imagegen.sniff_image_format(_png(800, 600)) == ("png", 800, 600)
    assert imagegen.sniff_image_format(_jpeg(400, 300)) == ("jpg", 400, 300)
    assert imagegen.sniff_image_format(_webp(640, 480)) == ("webp", 640, 480)
    assert imagegen.sniff_image_format(b"not an image at all") == ("bin", 0, 0)
    assert imagegen.sniff_image_format(b"") == ("bin", 0, 0)


def test_images_endpoint_normalises_v1_and_api_v1():
    assert imagegen.images_endpoint("http://localhost:8001/v1") == \
        "http://localhost:8001/v1/images/generations"
    assert imagegen.images_endpoint("http://localhost:8001/api/v1") == \
        "http://localhost:8001/api/v1/images/generations"
    assert imagegen.images_endpoint("http://localhost:8001/v1/") == \
        "http://localhost:8001/v1/images/generations"
    # idempotent when the full path is already given
    assert imagegen.images_endpoint(
        "http://localhost:8001/v1/images/generations") == \
        "http://localhost:8001/v1/images/generations"


# ------------------------------------------------------------ generate_image --

def _img_cfg(base_url, model="flux-schnell", **kw) -> Config:
    cfg = Config()
    cfg.base_url = base_url
    cfg.model = model
    cfg.modality = "image"
    cfg.image_profile = "flux"
    cfg.profiles = {"flux": {"base_url": base_url, "model": model,
                             "modality": "image"}}
    cfg.auto_evict_servers = False
    cfg.auto_fit = False
    cfg.auto_detect_context = False
    cfg.timeout = 10.0
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _make_handler(bodies, responses):
    state = {"i": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                bodies.append(json.loads(raw))
            except Exception:
                bodies.append(raw)
            i = state["i"]
            state["i"] += 1
            status, payload = responses[i] if i < len(responses) else (500, {})
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return H


def _start_server(handler):
    srv = LocalHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_generate_image_posts_to_v1_and_writes_bytes(tmp_path):
    png = _png(64, 64)
    b64 = base64.b64encode(png).decode()
    bodies, responses = [], [(200, {"data": [{"b64_json": b64}]})]
    srv = _start_server(_make_handler(bodies, responses))
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        result = imagegen.generate_image(_img_cfg(base), "a lighthouse",
                                         root=tmp_path)
        assert result["ok"] is True
        assert result["format"] == "png"
        assert result["width"] == 64 and result["height"] == 64
        assert result["bytes"] == len(png)
        assert result["model"] == "flux-schnell"
        assert result["profile"] == "flux"
        assert result["base_url"] == base
        assert result["prompt"] == "a lighthouse"
        assert result["rel_path"].startswith(".pedacito/images/")
        assert Path(result["path"]).read_bytes() == png
        assert bodies[0]["model"] == "flux-schnell"
        assert bodies[0]["prompt"] == "a lighthouse"
        assert bodies[0]["n"] == 1
        assert bodies[0]["response_format"] == "b64_json"
    finally:
        srv.shutdown()
        srv.server_close()


def test_generate_image_lemonade_api_v1(tmp_path):
    png = _png(48, 48)
    b64 = base64.b64encode(png).decode()
    bodies, responses = [], [(200, {"data": [{"b64_json": b64}]})]
    srv = _start_server(_make_handler(bodies, responses))
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/api/v1"
        result = imagegen.generate_image(_img_cfg(base), "a test image",
                                         root=tmp_path)
        assert result["ok"] is True
        assert result["format"] == "png"
    finally:
        srv.shutdown()
        srv.server_close()


def test_generate_image_retries_on_400_stripping_optional_params(tmp_path):
    png = _png(32, 32)
    b64 = base64.b64encode(png).decode()
    bodies, responses = [], [
        (400, {"error": {"message": "unknown field: steps"}}),
        (200, {"data": [{"b64_json": b64}]}),
    ]
    srv = _start_server(_make_handler(bodies, responses))
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        result = imagegen.generate_image(
            _img_cfg(base), "x", size="512x512", steps=4, cfg_scale=7.0,
            seed=42, negative_prompt="blurry", root=tmp_path)
        assert result["ok"] is True
        assert len(bodies) == 2
        # first request carried the optional local params
        assert bodies[0]["steps"] == 4
        assert bodies[0]["cfg_scale"] == 7.0
        assert bodies[0]["seed"] == 42
        assert bodies[0]["negative_prompt"] == "blurry"
        assert bodies[0]["response_format"] == "b64_json"
        # the stripped retry dropped them, but kept model/prompt/n/size
        assert "steps" not in bodies[1]
        assert "cfg_scale" not in bodies[1]
        assert "seed" not in bodies[1]
        assert "negative_prompt" not in bodies[1]
        assert "response_format" not in bodies[1]
        assert bodies[1]["model"] == "flux-schnell"
        assert bodies[1]["size"] == "512x512"
    finally:
        srv.shutdown()
        srv.server_close()


def test_generate_image_refuses_path_escape(tmp_path):
    png = _png(16, 16)
    b64 = base64.b64encode(png).decode()
    bodies, responses = [], [(200, {"data": [{"b64_json": b64}]})]
    srv = _start_server(_make_handler(bodies, responses))
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        with pytest.raises(security.SecurityError):
            imagegen.generate_image(_img_cfg(base), "x",
                                    output_path="../escape.png", root=tmp_path)
    finally:
        srv.shutdown()
        srv.server_close()


def test_generate_image_raises_without_image_profile():
    cfg = Config()
    cfg.model = "qwen2.5-coder"
    with pytest.raises(imagegen.ImageGenError):
        imagegen.generate_image(cfg, "x", root="/tmp")


# ----------------------------------------- calibration dispatch and metadata --

@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated model-card storage (never the real user's cards)."""
    monkeypatch.setattr(modelcard, "_path",
                        lambda: tmp_path / "home" / "model_cards.json")
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def _img_card_cfg() -> Config:
    cfg = Config()
    cfg.base_url = "http://127.0.0.1:9/v1"
    cfg.model = "flux-schnell"
    cfg.modality = "image"
    return cfg


def test_calibrate_dispatches_to_image_calibration(home, monkeypatch):
    cfg = _img_card_cfg()
    calls: list = []

    def fake_generate(cfg_, prompt, **kw):
        calls.append((prompt, kw))
        return {"ok": True, "path": "/tmp/x.png",
                "rel_path": ".pedacito/images/x.png", "format": "png",
                "width": 64, "height": 64, "bytes": 100, "elapsed_s": 0.5,
                "model": cfg_.model, "profile": "", "base_url": cfg_.base_url,
                "prompt": prompt}

    monkeypatch.setattr(imagegen, "generate_image", fake_generate)
    # A None client must never be touched: the image dispatch happens first.
    card = modelcard.calibrate(None, cfg, verbose=False)
    assert card["modality"] == "image"
    assert card["measured"]["image_ok"] is True
    assert card["measured"]["overall"] == 10.0
    assert card["measured"]["latency_s"] == 0.5
    assert card["measured"]["specialties"] == ["image_generation"]
    assert card["choices"] == {}
    assert calls and calls[0][0] == "solid blue square on white background"


def test_card_meta_exposes_image_model():
    card = {"version": modelcard.CARD_VERSION, "ts": time.time(),
            "model": "flux-schnell", "modality": "image",
            "measured": {"overall": 10.0, "image_ok": True}, "choices": {}}
    meta = modelcard.card_meta(card)
    assert meta["is_image_model"] is True
    assert meta["modality"] == "image"

    text = {"version": modelcard.CARD_VERSION, "ts": time.time(),
            "model": "qwen2.5-coder", "measured": {"overall": 0.8},
            "choices": {}}
    assert modelcard.card_meta(text)["is_image_model"] is False
    assert modelcard.card_meta(text)["modality"] == "text"
    assert modelcard.card_meta(None)["is_image_model"] is False
    assert modelcard.card_meta(None)["modality"] == "text"


def test_calibrate_image_model_saves_card(home, monkeypatch):
    cfg = _img_card_cfg()
    monkeypatch.setattr(imagegen, "generate_image",
                        lambda cfg_, prompt, **kw: {
                            "ok": True, "path": "/tmp/x.png",
                            "rel_path": ".pedacito/images/x.png",
                            "format": "webp", "width": 512, "height": 512,
                            "bytes": 99, "elapsed_s": 0.2, "model": cfg_.model,
                            "profile": "", "base_url": cfg_.base_url,
                            "prompt": prompt})
    card = modelcard.calibrate_image_model(cfg, verbose=False)
    assert card["modality"] == "image"
    assert card["measured"]["format"] == "webp"
    loaded = modelcard.load(cfg)
    assert loaded is not None
    assert modelcard.card_meta(loaded)["is_image_model"] is True


# ------------------------------------------------ schema byte-identity / tool -

def test_schema_for_byte_identity_without_image_profile():
    cfg = Config()
    cfg.profiles = {"text": {"model": "qwen2.5-coder-14b",
                             "base_url": "http://127.0.0.1:1/v1"}}
    schema = tools.schema_for(cfg)
    assert "generate_image" not in schema["properties"]["action"]["enum"]
    assert "prompt" not in schema["properties"]
    assert "size" not in schema["properties"]
    # allow_image_gen alone (no image profile) changes nothing
    cfg2 = Config()
    cfg2.allow_image_gen = True
    cfg2.profiles = {"text": {"model": "qwen2.5-coder-14b",
                              "base_url": "http://127.0.0.1:1/v1"}}
    assert json.dumps(tools.schema_for(cfg), sort_keys=True) == \
        json.dumps(tools.schema_for(cfg2), sort_keys=True)


def test_schema_for_exposes_generate_image_when_configured():
    cfg = Config()
    cfg.allow_image_gen = True
    cfg.profiles = {"flux": {"model": "flux-schnell", "modality": "image",
                             "base_url": "http://127.0.0.1:1/v1"}}
    schema = tools.schema_for(cfg)
    assert "generate_image" in schema["properties"]["action"]["enum"]
    assert "prompt" in schema["properties"]
    assert "size" in schema["properties"]


def test_image_gen_offered_gates(monkeypatch):
    # no image profile: never offered
    assert tools._image_gen_offered(Config()) is False
    # image profile, but not opted in and no broker
    cfg = Config()
    cfg.profiles = {"flux": {"model": "flux-schnell", "modality": "image",
                             "base_url": "http://127.0.0.1:1/v1"}}
    monkeypatch.setattr(security, "permission_broker_active", lambda: False)
    assert tools._image_gen_offered(cfg) is False
    # opted in -> offered
    cfg.allow_image_gen = True
    assert tools._image_gen_offered(cfg) is True
    # broker active -> offered even without allow_image_gen
    cfg2 = Config()
    cfg2.profiles = {"flux": {"model": "flux-schnell", "modality": "image",
                              "base_url": "http://127.0.0.1:1/v1"}}
    monkeypatch.setattr(security, "permission_broker_active", lambda: True)
    assert tools._image_gen_offered(cfg2) is True
