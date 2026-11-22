"""Multimodal image input across Core, CLI, TUI & GUI (WP-13).

Uses ``pedacito.local_server.LocalHTTPServer`` (never stdlib ``HTTPServer``)
for all loopback fake-server tests.
"""
from __future__ import annotations

import base64
import json
import struct
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli as cli_mod
from pedacito import config as config_mod
from pedacito import explain as explain_mod
from pedacito import imagegen
from pedacito import index as index_mod
from pedacito import llm as llm_mod
from pedacito import loop as loop_mod
from pedacito import replay as replay_mod
from pedacito import security as security_mod
from pedacito import trajectories
from pedacito import tui as tui_mod
from pedacito.agent import Agent
from pedacito.gui import jobs as gui_jobs
from pedacito.gui import server as gui_server
from pedacito.llm import LLMClient, LLMError
from pedacito.local_server import LocalHTTPServer


# ------------------------------------------------------------------ fixture bytes

def _make_png(w: int = 4, h: int = 3) -> bytes:
    """Minimal valid PNG with IHDR + empty IDAT + IEND."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * w for _ in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def _make_jpeg(w: int = 16, h: int = 12) -> bytes:
    """Minimal JPEG bytes with SOI + SOF0 + EOI."""
    sof0_body = struct.pack(">BHHB", 8, h, w, 1) + b"\x01\x11\x00"
    return (
        b"\xff\xd8"
        + b"\xff\xc0" + struct.pack(">H", len(sof0_body) + 2) + sof0_body
        + b"\xff\xd9"
    )


def _make_webp(w: int = 10, h: int = 8) -> bytes:
    """Minimal VP8X WebP header with dimensions."""
    vp8x = (
        b"VP8X"
        + struct.pack("<I", 10)
        + b"\x00\x00\x00\x00"
        + (w - 1).to_bytes(3, "little")
        + (h - 1).to_bytes(3, "little")
    )
    return b"RIFF" + struct.pack("<I", 4 + len(vp8x)) + b"WEBP" + vp8x


def _make_gif(w: int = 6, h: int = 5) -> bytes:
    """Minimal GIF89a header."""
    return b"GIF89a" + struct.pack("<HH", w, h) + b"\x00\x00\x00\x3b"


def _serve(handler_cls):
    srv = LocalHTTPServer(("127.0.0.1", 0), handler_cls)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port


# --------------------------------------------------------- 1. imagegen helpers

def test_sniff_and_load_all_four_formats(tmp_path):
    cases = [
        ("a.png", _make_png(4, 3), "png", "image/png", 4, 3),
        ("b.jpg", _make_jpeg(16, 12), "jpg", "image/jpeg", 16, 12),
        ("c.webp", _make_webp(10, 8), "webp", "image/webp", 10, 8),
        ("d.gif", _make_gif(6, 5), "gif", "image/gif", 6, 5),
    ]
    for fname, raw, fmt, mime, w, h in cases:
        assert imagegen.sniff_image_format(raw) == (fmt, w, h)
        p = tmp_path / fname
        p.write_bytes(raw)
        att = imagegen.load_image_attachment(p)
        assert att["name"] == fname
        assert att["kind"] == "image"
        assert att["mime"] == mime
        assert att["width"] == w
        assert att["height"] == h
        assert att["bytes"] == len(raw)
        assert base64.b64decode(att["data_b64"]) == raw


def test_load_and_normalize_reject_corrupt_oversize_and_excess(tmp_path):
    # Corrupt magic bytes
    bad = tmp_path / "fake.png"
    bad.write_bytes(b"not really a png at all")
    with pytest.raises(ValueError, match="not a supported"):
        imagegen.load_image_attachment(bad)

    with pytest.raises(ValueError, match="not a supported"):
        imagegen.normalize_image_attachment({
            "name": "fake.png",
            "data_b64": base64.b64encode(b"plain text").decode("ascii"),
        })

    # >10 MB rejected
    big_bytes = _make_png(2, 2) + b"\x00" * (imagegen.MAX_IMAGE_ATTACH_BYTES + 1)
    big = tmp_path / "huge.png"
    big.write_bytes(big_bytes)
    with pytest.raises(ValueError, match="too large"):
        imagegen.load_image_attachment(big)

    with pytest.raises(ValueError, match="exceeds|too large"):
        imagegen.normalize_image_attachment({
            "name": "huge.png",
            "data_b64": base64.b64encode(big_bytes).decode("ascii"),
        })

    # >4 attachments rejected
    valid_item = {
        "name": "ok.png",
        "data_url": "data:image/png;base64,"
                    + base64.b64encode(_make_png(2, 2)).decode("ascii"),
    }
    norm = imagegen.normalize_image_attachments([valid_item] * 4)
    assert len(norm) == 4
    with pytest.raises(ValueError, match="Too many"):
        imagegen.normalize_image_attachments([valid_item] * 5)


def test_make_multimodal_content_text_only_byte_identical():
    assert imagegen.make_multimodal_content("hello", []) == "hello"
    assert imagegen.make_multimodal_content("hello", None) == "hello"
    raw = _make_png(3, 3)
    img = imagegen.normalize_image_attachment({
        "name": "shot.png",
        "data_b64": base64.b64encode(raw).decode("ascii"),
    })
    parts = imagegen.make_multimodal_content("describe this", [img])
    assert isinstance(parts, list)
    assert parts[0] == {"type": "text", "text": "describe this"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


# --------------------------------------------------------- 2. llm + hardening

def test_llm_multimodal_helpers_and_sanitization():
    b64 = base64.b64encode(_make_png(4, 4)).decode("ascii")
    img_part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [{"type": "text", "text": "Look at this"}, img_part]},
        {"role": "user", "content": "And fix the alignment."},
    ]

    merged = llm_mod.merge_consecutive(msgs)
    assert len(merged) == 2
    assert isinstance(merged[1]["content"], list)
    assert merged[1]["content"][0]["text"] == "Look at this"
    assert merged[1]["content"][1] == img_part
    assert merged[1]["content"][2]["text"] == "And fix the alignment."

    folded = llm_mod.fold_system(msgs)
    assert len(folded) == 1
    assert folded[0]["role"] == "user"
    assert isinstance(folded[0]["content"], list)
    assert "You are helpful." in folded[0]["content"][0]["text"]
    assert "Look at this" in folded[0]["content"][0]["text"]
    assert any(p.get("type") == "image_url" for p in folded[0]["content"])

    # _append_message_text appends to the last text part without dropping image_url
    m_copy = {"role": "user", "content": [{"type": "text", "text": "hi"}, img_part]}
    llm_mod._append_message_text(m_copy, "\n\nNext action?")
    assert m_copy["content"][0]["text"] == "hi\n\nNext action?"
    assert m_copy["content"][1] == img_part

    # estimate_messages counts ~1,000 tokens per attached image
    text_only_est = llm_mod.estimate_messages([{"role": "user", "content": "Look at this"}])
    mm_est = llm_mod.estimate_messages([
        {"role": "user", "content": [{"type": "text", "text": "Look at this"}, img_part]},
    ])
    assert mm_est - text_only_est == llm_mod.IMAGE_TOKENS_ESTIMATE

    # Secret scanner & redactor keep image_url intact while redacting text
    secret_msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "key is sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"},
            img_part,
        ],
    }]
    findings = security_mod.scan_messages_for_secrets(secret_msg)
    assert findings
    redacted, count, _ = security_mod.redact_messages(secret_msg)
    assert count >= 1
    assert redacted[0]["content"][1] == img_part
    assert "sk-proj-" not in redacted[0]["content"][0]["text"]

    # Trajectories and replay preview never dump raw base64
    traj_msg = trajectories._message(msgs[1])
    traj_json = json.dumps(traj_msg)
    assert b64 not in traj_json
    assert "[image: image/png]" in traj_json

    preview = replay_mod._preview([msgs[1]])
    assert b64 not in preview
    assert "[image: image/png]" in preview


# ----------------------------------------- 3. e2e LocalHTTPServer (OpenAI + Anthropic)

def test_e2e_openai_compat_and_anthropic_receive_image():
    raw = _make_png(5, 4)
    img = imagegen.normalize_image_attachment({
        "name": "ui.png",
        "data_b64": base64.b64encode(raw).decode("ascii"),
    })
    expected_b64 = img["data_b64"]

    # --- OpenAI-compatible endpoint ---
    oa_captured: list[dict] = []

    class OpenAIHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length))
            oa_captured.append(body)
            resp = json.dumps({
                "choices": [{"message": {"content": "I see a 5x4 PNG."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1020, "completion_tokens": 10},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):
            pass

    srv, port = _serve(OpenAIHandler)
    try:
        cfg = config_mod.Config()
        cfg.base_url = f"http://127.0.0.1:{port}/v1"
        cfg.model = "gemma-4-vision"
        cfg.stream = False
        client = LLMClient(cfg)
        content = imagegen.make_multimodal_content("What is in this image?", [img])
        reply = client.chat([{"role": "user", "content": content}], max_tokens=64, stream=False)
        assert "5x4 PNG" in reply
        sent_msg = oa_captured[0]["messages"][-1]
        assert isinstance(sent_msg["content"], list)
        assert sent_msg["content"][1]["type"] == "image_url"
        assert sent_msg["content"][1]["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"
    finally:
        srv.shutdown()
        srv.server_close()

    # --- Native Anthropic /v1/messages endpoint ---
    ant_captured: list[dict] = []

    class AnthropicHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/messages"
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length))
            ant_captured.append(body)
            resp = json.dumps({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Claude sees the PNG."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1050, "output_tokens": 8},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):
            pass

    srv, port = _serve(AnthropicHandler)
    try:
        cfg = config_mod.Config()
        cfg.base_url = f"http://127.0.0.1:{port}/v1"
        cfg.provider = "anthropic"
        cfg.model = "claude-sonnet-4-6"
        cfg.api_key = "sk-ant-fake"
        cfg.stream = False
        client = LLMClient(cfg)
        content = imagegen.make_multimodal_content("Describe this screenshot", [img])
        reply = client.chat([{"role": "user", "content": content}], max_tokens=64, stream=False)
        assert "Claude sees" in reply
        blocks = ant_captured[0]["messages"][0]["content"]
        img_blocks = [b for b in blocks if b.get("type") == "image"]
        assert len(img_blocks) == 1
        assert img_blocks[0]["source"] == {
            "type": "base64",
            "media_type": "image/png",
            "data": expected_b64,
        }
    finally:
        srv.shutdown()
        srv.server_close()


def test_http_400_vision_unsupported_surfaces_clear_error():
    class RejectVisionHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            body = json.dumps({
                "error": {"message": "Invalid message: content must be a string; image_url is not supported"}
            }).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv, port = _serve(RejectVisionHandler)
    try:
        cfg = config_mod.Config()
        cfg.base_url = f"http://127.0.0.1:{port}/v1"
        cfg.model = "text-only-7b"
        cfg.stream = False
        client = LLMClient(cfg)
        img = imagegen.normalize_image_attachment({
            "name": "shot.png",
            "data_b64": base64.b64encode(_make_png(2, 2)).decode("ascii"),
        })
        content = imagegen.make_multimodal_content("see this", [img])
        with pytest.raises(LLMError, match="does not support image inputs"):
            client.chat([{"role": "user", "content": content}], max_tokens=32, stream=False)
    finally:
        srv.shutdown()
        srv.server_close()


def test_agent_and_explain_thread_images(tmp_path):
    (tmp_path / "app.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    cfg = config_mod.Config()
    cfg.model = "vision-fake"
    cfg.answer_mode = "rebuild"
    idx = index_mod.build([str(tmp_path)], cfg, client=None, root=tmp_path, summarise=False)

    seen_gather = []
    seen_answer = []

    class FakeClient:
        def __init__(self, c):
            self.cfg = c
            self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0, "reasoning_tokens": 0, "seconds": 0.0}

        def estimate_tokens(self, text):
            return llm_mod.estimate_tokens(text)

        def chat_json(self, messages, schema, max_tokens, **kw):
            seen_gather.append(messages)
            return {"action": "done", "file": "", "start": 0, "end": 0,
                    "symbol": "", "pattern": "", "command": "", "content": "",
                    "reasoning": "done"}

        def chat(self, messages, max_tokens, **kw):
            seen_answer.append(messages)
            return "Explained with image."

    img = imagegen.normalize_image_attachment({
        "name": "bug.png",
        "data_b64": base64.b64encode(_make_png(3, 3)).decode("ascii"),
    })
    agent = Agent(idx, FakeClient(cfg), cfg)
    out = explain_mod.explain_task(agent, "What is wrong?", verbose=False, images=[img])
    assert out == "Explained with image."
    assert isinstance(seen_gather[0][1]["content"], list)
    assert any(p.get("type") == "image_url" for p in seen_gather[0][1]["content"])
    assert isinstance(seen_answer[0][1]["content"], list)
    assert any(p.get("type") == "image_url" for p in seen_answer[0][1]["content"])


# --------------------------------------------------------- 4. CLI --image

def test_cli_image_flag_parsing_and_loader(tmp_path):
    img1 = tmp_path / "one.png"
    img2 = tmp_path / "two.gif"
    img1.write_bytes(_make_png(4, 4))
    img2.write_bytes(_make_gif(3, 2))

    parser = cli_mod.build_parser()
    for argv in (
        ["ask", ".", "-t", "fix", "--image", str(img1), "--image", str(img2)],
        ["explain", ".", "-q", "what?", "--image", str(img1)],
        ["auto", ".", "-t", "goal", "--image", str(img1)],
        ["chat", ".", "--image", str(img1)],
        ["tui", ".", "--image", str(img1)],
        ["new", str(tmp_path / "proj"), "-t", "demo", "--image", str(img1)],
    ):
        args = parser.parse_args(argv)
        assert str(img1) in args.image

    args = parser.parse_args(["ask", ".", "-t", "fix", "--image", str(img1), "--image", str(img2)])
    loaded = cli_mod._load_cli_images(args)
    assert len(loaded) == 2
    assert loaded[0]["mime"] == "image/png"
    assert loaded[1]["mime"] == "image/gif"


# --------------------------------------------------------- 5. TUI /attach & @img

def test_tui_attach_and_at_image_staging(tmp_path, monkeypatch):
    shot = tmp_path / "screen.png"
    shot.write_bytes(_make_png(8, 6))
    second = tmp_path / "ref.webp"
    second.write_bytes(_make_webp(5, 5))

    cfg = config_mod.Config()
    cfg.model = "vision-model"
    sess = tui_mod.TuiSession(tmp_path, cfg, headless=True)
    try:
        # /attach PATH stages the image and shows the header badge
        sess.handle_line("/attach screen.png")
        assert len(sess.pending_images) == 1
        assert sess.pending_images[0]["name"] == "screen.png"
        header = "\n".join(sess.header_lines(140))
        assert "1 img" in header

        # /attach list and /attach clear
        sess.handle_line("/attach list")
        assert any("screen.png" in text for _k, text in sess.transcript)
        sess.handle_line("/attach clear")
        assert sess.pending_images == []

        # Auto-detect @ref.webp in prompt + combine with staged /attach screen.png
        sess.handle_line("/attach screen.png")
        captured_images = []

        def fake_run_task(root, cfg_, client, agent, idx, task, **kw):
            captured_images.append(kw.get("images"))
            from types import SimpleNamespace
            return SimpleNamespace(reply="ok", applied=False, results=[], steps=[],
                                   stats={}, tests_ok=None, ok=True, idx=idx,
                                   changed=[], fabricated=False, unaccepted=[])

        monkeypatch.setattr(loop_mod, "run_task", fake_run_task)
        monkeypatch.setattr(sess, "_ensure_backend", lambda: (object(), object(), object()))
        monkeypatch.setattr(sess, "_refresh_warmup", lambda c: None)
        monkeypatch.setattr(sess, "_evict_servers", lambda: None)

        sess.run_task("Compare @ref.webp with the staged screenshot")
        assert len(captured_images) == 1
        names = [img["name"] for img in captured_images[0]]
        assert "screen.png" in names
        assert "ref.webp" in names
        # Staged pending_images are cleared once sent
        assert sess.pending_images == []
    finally:
        sess.close()


# --------------------------------------------------------- 6. GUI /api/ask validation

def test_gui_api_ask_image_validation_and_lightweight_history(monkeypatch):
    valid_png_b64 = base64.b64encode(_make_png(4, 4)).decode("ascii")
    spawned = []

    def fake_spawn(mode, title, fn, echo=None):
        if echo:
            echo()
        spawned.append((mode, title))
        from types import SimpleNamespace
        return SimpleNamespace(id="job-1", mode=mode)

    monkeypatch.setattr(gui_jobs, "spawn", fake_spawn)

    # Corrupt magic bytes -> HTTP 400
    code, resp = gui_server._run_ask({}, {
        "text": "what is this?",
        "attachments": [{"kind": "image", "name": "bad.png",
                         "data_b64": base64.b64encode(b"not-an-image").decode("ascii")}],
    })
    assert code == 400
    assert resp["ok"] is False

    # >10 MB -> HTTP 400
    huge_b64 = base64.b64encode(_make_png(2, 2) + b"\x00" * (imagegen.MAX_IMAGE_ATTACH_BYTES + 1)).decode("ascii")
    code, resp = gui_server._run_ask({}, {
        "text": "too big",
        "attachments": [{"kind": "image", "name": "huge.png", "data_b64": huge_b64}],
    })
    assert code == 400
    assert resp["ok"] is False

    # >4 images -> HTTP 400
    one_img = {"kind": "image", "name": "ok.png", "data_b64": valid_png_b64}
    code, resp = gui_server._run_ask({}, {
        "text": "too many",
        "attachments": [one_img] * 5,
    })
    assert code == 400
    assert resp["ok"] is False

    # Valid PNG -> HTTP 200 and lightweight history (no raw data_b64 stored!)
    recorded_msgs = []
    monkeypatch.setattr(gui_jobs.STATE, "add_message",
                        lambda role, text, **kw: recorded_msgs.append((role, text, kw)))

    code, resp = gui_server._run_ask({}, {
        "text": "describe this",
        "attachments": [{"kind": "image", "name": "ok.png", "data_b64": valid_png_b64}],
    })
    assert code == 202
    assert resp["ok"] is True
    assert spawned and spawned[-1][0] == "explain"
    assert recorded_msgs
    saved_att = recorded_msgs[-1][2]["attachments"][0]
    assert saved_att["kind"] == "image"
    assert saved_att["name"] == "ok.png"
    assert saved_att["width"] == 4 and saved_att["height"] == 4
    assert "data_b64" not in saved_att
