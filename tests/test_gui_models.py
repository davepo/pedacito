"""The GUI's model-workflow endpoints: reset to the recommended baseline,
discovery scan, selective profile import, and the one-click calibration job.

Everything runs against the fake OpenAI-style server from
``test_models_scan``: no model, no GPU, and no real port is required (the
scan probes the usual local ports too, but the assertions only look at the
rows that came from the fake server's URL)."""
# `gui` below is test_gui's server fixture: pytest hands it to each test as a
# parameter, which ruff otherwise reads as a redefinition.
# ruff: noqa: F811
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import config as config_mod
from pedacito import modelcard
from pedacito.gui import core, jobs
from test_gui import gui  # noqa: F401  (the shared server fixture)
from test_models_scan import _lmstudio, _serve

GIB = 1024 ** 3

# A config with profiles, a default, and per-role assignments: exactly the
# state "Reset to Baseline" is supposed to wipe.
DIRTY = '''\
default_profile = "qwen3-8b"

[universal]
vram_gb = 12
base_url = "http://localhost:1234/v1"
model = "qwen3-8b"

[agent]
roles = { planner = "big", summarizer = "tiny" }
max_total_model_calls = 99
max_read_tokens = 9999

[profiles.qwen3-8b]
base_url = "http://localhost:1234/v1"
model = "qwen3-8b"

[profiles.gemma-4b]
base_url = "http://localhost:1234/v1"
model = "gemma-4b-it"
'''


@pytest.fixture
def fake_local():
    """Start the fake server; yields its OpenAI-style base URL."""
    servers = []

    def start(models):
        srv = _serve(_lmstudio(models))
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}/v1"

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _write(gui_client, text):
    gui_client.cfg_path.parent.mkdir(parents=True, exist_ok=True)
    gui_client.cfg_path.write_text(text, encoding="utf-8")


def _rows(j, base_url):
    return [m for m in j["models"] if m["server_url"] == base_url]


# ----------------------------------------------------------- reset to baseline

def test_reset_backs_up_wipes_profiles_and_restores_defaults(gui):
    _write(gui, DIRTY)
    status, j = gui.post("/api/settings/reset", {})
    assert status == 200 and j["ok"] is True

    backups = list(gui.cfg_path.parent.glob("config.toml.reset.*.bak"))
    assert len(backups) == 1 and Path(j["backup"]) == backups[0]
    assert backups[0].read_text(encoding="utf-8") == DIRTY

    new = gui.cfg_path.read_text(encoding="utf-8")
    assert "[profiles." not in new
    assert "default_profile" not in new
    assert "roles = " not in new
    # the recommended small-model defaults are what a fresh setup should get
    for line in ('step_reasoning_first = true', 'edit_plan_brief = true',
                 'incremental_edit_loop = true', 'max_read_tokens = 1200',
                 'max_total_model_calls = 30', 'max_no_progress_steps = 4',
                 'temperature_answer = 0.0', 'vram_gb = 8'):
        assert line in new, line
    assert "max_total_model_calls = 99" not in new

    # ...and the live state agrees: no profiles left anywhere
    status, s = gui.get("/api/settings")
    assert status == 200 and s["user"]["profiles"] == {}
    status, st = gui.get("/api/state")
    assert st["profiles"] == [] and st["needs_setup"] is False


def test_reset_on_a_missing_config_still_writes_the_baseline(gui):
    assert not gui.cfg_path.exists()
    status, j = gui.post("/api/settings/reset", {})
    assert status == 200 and j["ok"] and j["backup"] == ""
    assert "step_reasoning_first = true" in gui.cfg_path.read_text(encoding="utf-8")
    assert list(gui.cfg_path.parent.glob("*.bak")) == []


def test_reset_response_carries_the_settings_payload(gui):
    _write(gui, DIRTY)
    _status, j = gui.post("/api/settings/reset", {})
    assert j["schema"] and j["user"]["available"] and "active_profile" in j


# ---------------------------------------------------------------- scan (read-only)

MODELS = [
    {"id": "lmstudio-ai/qwen3-8b", "size_bytes": 5 * GIB,
     "context_length": 40960, "quantization": "q4_k_m", "loaded_in_memory": True},
    {"id": "text-embedding-small", "type": "embeddings", "size_bytes": 1 * GIB},
]


def test_scan_lists_models_without_touching_the_config(gui, fake_local):
    base = fake_local(MODELS)
    status, j = gui.post("/api/models/scan", {"urls": [base], "vram_gb": 16})
    assert status == 200 and j["ok"]
    rows = _rows(j, base)
    assert [r["model_id"] for r in rows] == ["lmstudio-ai/qwen3-8b"]  # embeddings skipped
    r = rows[0]
    assert r["provider"] and r["base_url"] == base
    assert r["suggested_profile_name"] == "qwen3-8b"
    assert r["context_length"] == 40960 and r["loaded"] is True
    assert r["already_imported"] is False
    assert not gui.cfg_path.exists()          # a scan writes nothing
    assert j["servers"][0]["base_url"] == base
    assert j["servers"][0]["models"] == rows


def test_scan_marks_models_a_profile_already_imports(gui, fake_local):
    base = fake_local(MODELS)
    _write(gui, DIRTY.replace("http://localhost:1234/v1", base)
           .replace('model = "qwen3-8b"', 'model = "lmstudio-ai/qwen3-8b"'))
    _status, j = gui.post("/api/models/scan", {"urls": [base], "vram_gb": 16})
    rows = _rows(j, base)
    assert [r["already_imported"] for r in rows] == [True]
    # the suggested name must not collide with the profile that is already there
    assert rows[0]["suggested_profile_name"] != "qwen3-8b"


def test_scan_of_an_unreachable_url_reports_no_rows(gui):
    status, j = gui.post("/api/models/scan", {"urls": ["http://127.0.0.1:1/v1"]})
    assert status == 200 and _rows(j, "http://127.0.0.1:1/v1") == []


def test_scan_collapses_url_spellings_into_one_server_and_row(gui, fake_local):
    """`localhost` vs `127.0.0.1`, with or without `/v1` or a trailing slash:
    one server probed once, listed once, one row per model."""
    base = fake_local(MODELS)                       # http://127.0.0.1:PORT/v1
    port = base.rsplit(":", 1)[1].split("/")[0]
    spellings = [f"http://localhost:{port}/v1/", base,
                 f"http://127.0.0.1:{port}/v1/"]
    status, j = gui.post("/api/models/scan", {"urls": spellings, "vram_gb": 16})
    assert status == 200 and j["ok"]
    # (this machine may run a real LM Studio on a default port: look only at
    # the rows that came from the fake server's URL)
    assert len(_rows(j, base)) == 1
    assert [r["model_id"] for r in _rows(j, base)] == ["lmstudio-ai/qwen3-8b"]
    same = [s for s in j["servers"] if any(m["server_url"] == base for m in s["models"])]
    assert len(same) == 1 and same[0]["base_url"] == base


def test_scan_reads_profiles_from_the_gui_config_not_the_cli_global(
        gui, fake_local, monkeypatch, tmp_path):
    """The scan's profile URLs come from the GUI's own config path. A stray
    cli-level config global (the real user config on a dev machine, with its
    remote servers) must not leak extra candidates into the probe list."""
    from pedacito import cli as cli_mod
    from pedacito import discovery
    base = fake_local(MODELS)
    _write(gui, DIRTY.replace("http://localhost:1234/v1", base))
    stray = tmp_path / "stray-config.toml"
    stray.write_text('[profiles.stray]\nbase_url = "http://stray.invalid/v1"\n'
                     'model = "stray-model"\n', encoding="utf-8")
    monkeypatch.setattr(cli_mod, "USER_CONFIG", stray)
    seen = {}

    def spy(*a, **k):
        seen.update(k)
        return []

    monkeypatch.setattr(discovery, "discover_servers", spy)
    status, j = gui.post("/api/models/scan", {"urls": [base], "vram_gb": 16})
    assert status == 200 and j["ok"]
    assert seen["urls"] == [base]
    assert seen["profile_urls"] == [base]      # the stray server is not probed


def test_scan_dedups_even_when_discovery_repeats_a_server(gui, fake_local,
                                                          monkeypatch):
    """Belt and braces at the display layer: if discovery ever hands the same
    server twice (differently spelled), the scan still returns one row per
    (normalized server URL, model id)."""
    from pedacito import discovery
    base = fake_local(MODELS)
    dup = [("LM Studio", base, ["x"]),
           ("LM Studio", base + "/", ["x"])]
    monkeypatch.setattr(discovery, "discover_servers",
                        lambda *a, **k: list(dup))
    status, j = gui.post("/api/models/scan", {"urls": [base], "vram_gb": 16})
    assert status == 200 and j["ok"]
    assert len(j["models"]) == 1 and len(j["servers"]) == 1
    assert j["models"][0]["server_url"] == base


# ---------------------------------------------------------------- selective import

def _sel(base, name, model, default=False, ctx=40960, vram=16):
    return {"profile_name": name, "base_url": base, "model": model,
            "context_length": ctx, "vram_gb": vram, "set_default": default}


def test_import_writes_only_what_was_selected(gui, fake_local):
    base = fake_local(MODELS)
    status, j = gui.post("/api/models/import", {"models": [
        _sel(base, "qwen3-8b", "lmstudio-ai/qwen3-8b", default=True),
        _sel(base, "gemma-4b", "gemma-4b-it", ctx=8192)]})
    assert status == 200 and sorted(j["imported"]) == ["gemma-4b", "qwen3-8b"]
    assert j["skipped"] == [] and j["default_profile"] == "qwen3-8b"

    text = gui.cfg_path.read_text(encoding="utf-8")
    assert '[profiles.qwen3-8b]' in text and '[profiles.gemma-4b]' in text
    assert 'default_profile = "qwen3-8b"' in text
    assert "vram_gb = 16" in text and "context_tokens = 8192" in text
    assert not gui.cfg_path.with_name("model_cards.json").exists()

    status, st = gui.get("/api/state")
    # the starter template's own `default` profile survives; the two imported
    # ones are added next to it and the default pointer moves to qwen3-8b
    assert set(st["profiles"]) >= {"gemma-4b", "qwen3-8b"}
    assert st["default_profile"] == "qwen3-8b"
    # the file that did not exist is now valid TOML the loader accepts
    assert config_mod.load(root=None, require_hardware=False).model == "lmstudio-ai/qwen3-8b"


def test_import_skips_names_that_already_exist(gui, fake_local):
    base = fake_local(MODELS)
    _write(gui, DIRTY.replace("http://localhost:1234/v1", base))
    before = gui.cfg_path.read_text(encoding="utf-8")
    status, j = gui.post("/api/models/import", {"models": [
        _sel(base, "qwen3-8b", "lmstudio-ai/qwen3-8b")]})
    assert status == 200 and j["imported"] == []
    assert j["skipped"] == ["qwen3-8b"]
    assert gui.cfg_path.read_text(encoding="utf-8") == before


def test_import_refuses_empty_and_bad_names_without_writing(gui, fake_local):
    base = fake_local(MODELS)
    assert gui.post("/api/models/import", {"models": []})[0] == 400
    assert gui.post("/api/models/import", {})[0] == 400
    status, j = gui.post("/api/models/import", {"models": [
        _sel(base, "bad/name", "lmstudio-ai/qwen3-8b")]})
    assert status == 400 and "detail" in j
    assert not gui.cfg_path.exists()


# ---------------------------------------------------------------- calibrate (job)

def test_calibrate_needs_a_known_profile(gui):
    _write(gui, DIRTY)
    status, j = gui.post("/api/models/calibrate", {"profile": "nope", "mode": "quick"})
    assert status == 400 and "nope" in j["detail"] and "qwen3-8b" in j["detail"]
    assert gui.post("/api/models/calibrate", {})[0] == 400


def test_calibrate_runs_in_the_background_and_saves_a_card(gui, monkeypatch):
    _write(gui, DIRTY)
    cards = []

    class FakeClient:
        def __init__(self, cfg):
            self.cfg = cfg

        def check(self, warn=True):
            pass

        def warmup(self):
            pass

    def fake_calibrate(client, cfg, verbose=True, quick=False):
        print(f"probing {cfg.model} (quick={quick})")
        return {"version": modelcard.CARD_VERSION,
                "key": modelcard.card_key(cfg), "ts": 1, "model": cfg.model,
                "url": cfg.base_url, "seconds": 1.5, "repeats": 1,
                "measured": {"overall": 0.83, "cap_tools": True,
                             "tok_per_sec": 42.0},
                "choices": {}, "notes": []}

    monkeypatch.setattr(jobs, "LLMClient", FakeClient)
    monkeypatch.setattr(modelcard, "calibrate", fake_calibrate)
    real_save = modelcard.save_card
    monkeypatch.setattr(modelcard, "save_card",
                        lambda card: (cards.append(card), real_save(card))[1])

    # in production the server installs the stdout tee; this harness does not,
    # so mirror it here to prove the suite's progress reaches the pane
    orig = core.install_tee()
    try:
        status, j = gui.post("/api/models/calibrate", {"profile": "qwen3-8b",
                                                      "mode": "quick"})
        assert status == 202 and j["ok"]
        assert gui.wait_idle(30), "calibration job did not finish"
    finally:
        core.uninstall_tee(orig)
    assert not gui.get("/api/state")[1]["job"]

    _status, st = gui.get("/api/state")
    texts = " ".join(m["text"] for m in st["messages"])
    assert "Calibrating" in texts and "qwen3-8b" in texts
    assert "Calibration of 'qwen3-8b' finished" in texts and "0.83" in texts
    # the tee of the job's stdout reached the terminal pane
    assert "probing qwen3-8b" in gui.get("/api/snapshot")[1]["term"]
    saved = gui.cfg_path.parent / "model_cards.json"
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8"))  # keyed by server+model
    assert cards and cards[0]["measured"]["overall"] == 0.83
