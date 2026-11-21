"""WP-08: Image Generation UX across CLI, TUI, Web GUI, and i18n.

Hermetic tests: local threaded fake `/v1/images/generations` HTTP server,
hand-crafted PNG header bytes, and isolated temp workspaces.
"""
from __future__ import annotations

import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli, config_editor, modelcard, tui
from pedacito.config import Config
from pedacito.gui import jobs as gui_jobs
from pedacito.gui import server as gui_server
from pedacito.gui.core import STATE, Job


def _png(width=256, height=192) -> bytes:
    data = b"\x89PNG\r\n\x1a\n"
    data += (13).to_bytes(4, "big") + b"IHDR"
    data += width.to_bytes(4, "big") + height.to_bytes(4, "big")
    data += bytes([8, 2, 0, 0, 0])
    data += b"\x00\x00\x00\x00"
    return data


class _FakeImageServer:
    def __init__(self, width=256, height=192):
        self.width = width
        self.height = height
        self.requests: list[dict] = []
        self.chat_hits = 0
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception:
                    body = {}
                if self.path.endswith("/images/generations"):
                    parent.requests.append(body)
                    payload = json.dumps({
                        "created": int(time.time()),
                        "data": [{"b64_json": base64.b64encode(
                            _png(parent.width, parent.height)).decode("ascii")}],
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    if "/chat/completions" in self.path:
                        parent.chat_hits += 1
                    self.send_response(404)
                    self.end_headers()

        self._srv = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._srv.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._srv.shutdown()
        self._srv.server_close()


def _write_image_config(path: Path, base_url: str, model: str = "flux-schnell") -> Path:
    path.write_text(
        'default_profile = "flux"\n'
        'image_profile = "flux"\n'
        'vram_gb = 16\n'
        'auto_evict_servers = false\n\n'
        '[profiles.flux]\n'
        f'base_url = "{base_url}"\n'
        f'model = "{model}"\n'
        'modality = "image"\n',
        encoding="utf-8",
    )
    return path


# ------------------------------------------------------------------ CLI tests --

def test_cli_image_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["image", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--out", "--size", "--steps", "--cfg-scale", "--seed",
                 "--negative", "--profile", "--root", "--config", "--lang", "--json"):
        assert flag in out


def test_cli_image_json_and_text_output(tmp_path, capsys):
    with _FakeImageServer(width=512, height=384) as srv:
        cfg_file = _write_image_config(tmp_path / "cfg.toml", srv.base_url)
        ws = tmp_path / "ws"
        ws.mkdir()

        rc = cli._main([
            "image", "a cube",
            "--out", "assets/cube.png",
            "--size", "512x384",
            "--steps", "8",
            "--cfg-scale", "3.5",
            "--seed", "42",
            "--negative", "blurry",
            "--root", str(ws),
            "--config", str(cfg_file),
            "--json",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["ok"] is True
        assert data["rel_path"] == "assets/cube.png"
        assert data["width"] == 512
        assert data["height"] == 384
        assert data["model"] == "flux-schnell"
        assert (ws / "assets" / "cube.png").is_file()

        assert len(srv.requests) == 1
        req = srv.requests[0]
        assert req["prompt"] == "a cube"
        assert req["size"] == "512x384"
        assert req["steps"] == 8
        assert req["cfg_scale"] == 3.5
        assert req["seed"] == 42
        assert req["negative_prompt"] == "blurry"

        # Human-readable output
        rc2 = cli._main([
            "image", "a sphere",
            "--root", str(ws),
            "--config", str(cfg_file),
        ])
        assert rc2 == 0
        text_out = capsys.readouterr().out
        assert "512x384" in text_out
        assert "flux-schnell" in text_out


def test_cli_models_scan_labels_image_models_and_writes_modality(capsys, monkeypatch):
    models = [
        {"id": "flux-schnell", "base_url": "http://127.0.0.1:8001/v1",
         "loaded": True, "labels": ["image"], "recipe": "sd-cpp"},
        {"id": "qwen2.5-coder-7b", "base_url": "http://127.0.0.1:8001/v1",
         "loaded": True, "labels": ["text"]},
    ]
    monkeypatch.setattr(cli, "_ask", lambda prompt, default="": "all")
    picked = cli._interactive_pick("Lemonade", "http://127.0.0.1:8001/v1", models)
    out = capsys.readouterr().out
    assert len(picked) == 2
    assert "[image]" in out

    entries = [cli._to_scan_entry(m) for m in picked]
    assert entries[0]["modality"] == "image"
    assert entries[1]["modality"] is None

    toml_text = config_editor.upsert_profiles('base_url = "http://127.0.0.1:8001/v1"\n', entries)
    assert 'modality = "image"' in toml_text


def test_cli_calibrate_all_probes_image_profile_without_chat_warmup(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(modelcard, "_path", lambda: tmp_path / "model_cards.json")
    with _FakeImageServer(width=512, height=512) as srv:
        cfg_file = _write_image_config(tmp_path / "cfg.toml", srv.base_url)
        rc = cli._main([
            "calibrate", "--all", "-y", "--recalibrate",
            "--config", str(cfg_file),
            "--root", str(tmp_path),
        ])
        assert rc == 0
        assert srv.chat_hits == 0
        assert len(srv.requests) == 1
        out = capsys.readouterr().out
        assert "Saved image card for 'flux'" in out


# ------------------------------------------------------------------ TUI tests --

def test_tui_image_slash_command_registration_completion_and_help():
    assert "/image" in tui._SLASH_COMMANDS
    assert tui._TAB_PREFIX["/im"] == "/image "
    assert "/image PROMPT" in tui.HELP_TEXT
    assert "/image PROMPT" in tui.HELP_TEXT_ES


def test_tui_image_execution_and_header_badge(tmp_path, monkeypatch):
    monkeypatch.setattr(modelcard, "_path", lambda: tmp_path / "model_cards.json")
    with _FakeImageServer(width=640, height=480) as srv:
        cfg = Config()
        cfg.base_url = srv.base_url
        cfg.model = "flux-schnell"
        cfg.modality = "image"
        cfg.active_profile = "flux"
        cfg.image_profile = "flux"
        cfg.profiles = {"flux": {"base_url": srv.base_url, "model": "flux-schnell",
                                 "modality": "image"}}
        cfg.auto_evict_servers = False
        cfg.auto_fit = False
        cfg.auto_detect_context = False

        session = tui.TuiSession(tmp_path, cfg, headless=True)
        try:
            # Before calibration: 🎨 Image
            assert session._calibration_badge() == "\U0001f3a8 Image"

            # Tab completion for /im
            session.input_buf = "/im"
            session.cursor_pos = len(session.input_buf)
            session._complete()
            assert session.input_buf == "/image "

            # Execute /image
            keep_open = session.handle_line('/image "a lighthouse" --out art/lh.png --size 640x480')
            assert keep_open is True
            assert (tmp_path / "art" / "lh.png").is_file()
            transcript = "\n".join(text for _, text in session.transcript)
            assert "art/lh.png" in transcript
            assert "640x480" in transcript

            # After image calibration probe: 🎨 Image ✓
            card = modelcard.calibrate_image_model(cfg, verbose=False)
            modelcard.save_card(card)
            assert session._calibration_badge() == "\U0001f3a8 Image \u2713"
        finally:
            session.close()


# -------------------------------------------------------------- Web GUI tests --

def test_gui_image_generate_and_file_serving_and_traversal_refusal(tmp_path, monkeypatch):
    with _FakeImageServer(width=320, height=240) as srv:
        cfg = Config()
        cfg.base_url = srv.base_url
        cfg.model = "flux-schnell"
        cfg.modality = "image"
        cfg.active_profile = "flux"
        cfg.image_profile = "flux"
        cfg.profiles = {"flux": {"base_url": srv.base_url, "model": "flux-schnell",
                                 "modality": "image"}}
        cfg.auto_evict_servers = False

        ws = tmp_path / "proj"
        ws.mkdir()
        monkeypatch.setattr(STATE, "root", ws)
        monkeypatch.setattr(STATE, "cfg", cfg)

        # POST /api/image/generate
        gen_fn = gui_server.POST_ROUTES["/api/image/generate"]
        code, body = gen_fn({}, {
            "prompt": "a red cube",
            "output_path": "assets/red.png",
            "size": "320x240",
            "steps": 4,
            "seed": 7,
            "profile": "flux",
        })
        assert code == 200
        assert body["ok"] is True
        assert body["rel_path"] == "assets/red.png"
        assert body["url"] == "/api/image/file?path=assets/red.png"
        assert body["width"] == 320
        assert body["height"] == 240
        assert body["model"] == "flux-schnell"
        assert body["profile"] == "flux"

        # GET /api/image/file serves the PNG bytes with image/png
        file_fn = gui_server.GET_ROUTES["/api/image/file"]
        f_code, f_bytes, f_ctype = file_fn({"path": ["assets/red.png"]})
        assert f_code == 200
        assert f_ctype == "image/png"
        assert f_bytes[:8] == b"\x89PNG\r\n\x1a\n"

        # Path traversal is strictly refused
        t_code, t_body = file_fn({"path": ["../outside.png"]})
        assert t_code == 403
        assert t_body["ok"] is False

        # Non-image file inside workspace is refused (415)
        (ws / "secret.txt").write_text("not an image", encoding="utf-8")
        n_code, n_body = file_fn({"path": ["secret.txt"]})
        assert n_code == 415
        assert n_body["ok"] is False

        # Scratchpad containment when STATE.root is None
        scratch = tmp_path / "scratch"
        (scratch / ".pedacito" / "images").mkdir(parents=True)
        (scratch / ".pedacito" / "images" / "ok.png").write_bytes(_png(64, 64))
        (scratch / "outside_img.png").write_bytes(_png(64, 64))
        monkeypatch.setattr(STATE, "root", None)
        monkeypatch.setattr(STATE, "scratchpad_path", lambda: scratch)

        s_code, s_bytes, s_ctype = file_fn({"path": [".pedacito/images/ok.png"]})
        assert s_code == 200
        assert s_ctype == "image/png"

        bad_code, bad_body = file_fn({"path": ["outside_img.png"]})
        assert bad_code == 403
        assert bad_body["ok"] is False


def test_gui_jobs_calibrate_all_handles_image_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(modelcard, "_path", lambda: tmp_path / "model_cards.json")
    with _FakeImageServer(width=512, height=512) as srv:
        cfg_file = _write_image_config(tmp_path / "config.toml", srv.base_url)
        from pedacito import config as config_mod
        monkeypatch.setattr(config_mod, "USER_CONFIG", cfg_file)
        monkeypatch.setattr(STATE, "root", None)

        job = Job(1, "calibrate", "calibrate all")
        gui_jobs.run_calibrate_all(job, {"mode": "quick", "recalibrate": True})
        assert srv.chat_hits == 0
        assert len(srv.requests) == 1


# ----------------------------------------------------------- Frontend checks --

def test_frontend_static_assets_handle_is_image_model():
    root = Path(__file__).resolve().parents[1] / "pedacito" / "gui" / "static"
    app_js = (root / "app.js").read_text(encoding="utf-8")
    settings_js = (root / "settings.js").read_text(encoding="utf-8")
    app_css = (root / "app.css").read_text(encoding="utf-8")

    # app.js: CalibrationBadge, ProfileSelect, TopBar, Composer, imageGenDialog
    assert "cal.is_image_model" in app_js
    assert "gui.cal.image_verified" in app_js
    assert "gui.cal.image_badge" in app_js
    assert "function imageGenDialog()" in app_js
    assert "/api/image/generate" in app_js
    assert "gui.menu.generate_image" in app_js
    assert "id=\"imageGenBtn\"" in app_js
    assert "\\uD83C\\uDFA8 image" in app_js

    # settings.js: image tag on profiles & scanner, Probe image button
    assert "cal.is_image_model" in settings_js
    assert "r.is_image_model" in settings_js
    assert "settings.profile.probe_image" in settings_js
    assert "tag image" in settings_js

    # app.css: badge and dialog styles
    assert ".cal-chip.image" in app_css
    assert ".img-gen-dialog" in app_css
    assert ".img-gen-preview" in app_css
