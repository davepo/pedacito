"""GUI profile availability: local profile models checked against the server
states already fetched by /api/servers, plus the switch guard and the picker.

No test here contacts a model server: server states are stubbed, and the
hosted-profile test proves the availability code makes no request at all.
"""
# `gui` is test_gui's server fixture, imported for each test below.
# ruff: noqa: F811
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import config as config_mod  # noqa: E402
from pedacito import profiler, servers  # noqa: E402
from pedacito.gui import core  # noqa: E402
from pedacito.gui import server as gui_server  # noqa: E402
from test_gui import gui  # noqa: F401  (the shared server fixture)

ROOT = Path(__file__).resolve().parents[1]

CONFIG = '''\
default_profile = "present"

[profiles.present]
base_url = "http://127.0.0.1:8000/v1"
model = "present-model"
tier = "small"

[profiles.absent]
base_url = "http://127.0.0.1:8000/v1"
model = "deleted-model"
tier = "small"

[profiles.empty]
base_url = "http://127.0.0.1:8001/v1"
model = "empty-model"
tier = "small"

[profiles.offline]
base_url = "http://127.0.0.1:8002/v1"
model = "offline-model"
tier = "small"

[profiles.cloud]
base_url = "https://api.example.com/v1"
model = "hosted-model"
tier = "hosted"

[profiles.unmatched]
base_url = "http://127.0.0.1:9999/v1"
model = "unmatched-model"
tier = "small"
'''


def _write(client, text=CONFIG) -> None:
    client.cfg_path.parent.mkdir(parents=True, exist_ok=True)
    client.cfg_path.write_text(text, encoding="utf-8")


def _memory() -> dict:
    return {
        "ram": {"total_gb": 64.0, "used_gb": 32.0, "percent": 50.0},
        "vram": {"total_gb": 24.0, "used_gb": 20.0, "percent": 83.3,
                 "cards": []},
        "ram_total_gb": 64.0, "ram_used_gb": 32.0,
        "vram_total_gb": 24.0, "vram_used_gb": 20.0,
        "available": True,
    }


def _state(base_url, *, online, models, name="server") -> dict:
    return {"name": name, "base_url": base_url, "online": online,
            "models": list(models), "loaded_models": []}


STATES = [
    _state("http://127.0.0.1:8000/v1", online=True,
           models=["present-model", "other-model"], name="main"),
    # Reachable, but the endpoint answered with no model list.
    _state("http://127.0.0.1:8001/v1", online=True, models=[], name="empty"),
    # Not reachable: fail open, never a 409.
    _state("http://127.0.0.1:8002/v1", online=False,
           models=["offline-model"], name="sleepy"),
]


def _config() -> config_mod.Config:
    cfg = config_mod.Config()
    cfg.profiles = {
        "present": {"base_url": "http://127.0.0.1:8000/v1",
                    "model": "present-model"},
        "absent": {"base_url": "http://localhost:8000/v1",
                   "model": "deleted-model"},
        "empty": {"base_url": "http://127.0.0.1:8001/v1",
                  "model": "empty-model"},
        "offline": {"base_url": "http://127.0.0.1:8002/v1",
                    "model": "offline-model"},
        "cloud": {"base_url": "https://api.example.com/v1",
                  "model": "hosted-model"},
        "unmatched": {"base_url": "http://127.0.0.1:9999/v1",
                      "model": "unmatched-model"},
    }
    cfg.active_profile = "present"
    cfg.base_url = "http://127.0.0.1:8000/v1"
    cfg.model = "present-model"
    return cfg


# ------------------------------------------------------------- availability ---

def test_profile_availability_matrix():
    """Present/absent/empty/offline/hosted/unmatched, with localhost folded
    to 127.0.0.1 by servers.normalize_url."""
    out = gui_server._profile_availability(_config(), STATES)
    assert out == {
        "present": "available",
        "absent": "unavailable",
        "empty": "unknown",
        "offline": "offline",
        "cloud": "unknown",
        "unmatched": "unknown",
    }


def test_servers_endpoint_returns_and_caches_the_map(gui, monkeypatch):
    _write(gui)
    monkeypatch.setattr(profiler, "snapshot", lambda *a, **k: _memory())
    monkeypatch.setattr(servers, "list_server_states",
                        lambda *a, **k: STATES)

    status, j = gui.get("/api/servers")
    assert status == 200 and j["ok"] is True
    assert j["profile_availability"] == {
        "present": "available",
        "absent": "unavailable",
        "empty": "unknown",
        "offline": "offline",
        "cloud": "unknown",
        "unmatched": "unknown",
    }
    with core.STATE.lock:
        cached = dict(core.STATE.profile_availability)
    assert cached == j["profile_availability"]
    # The endpoint still carries the server states the map was built from.
    assert j["servers"][0]["models"] == ["present-model", "other-model"]


def test_hosted_profiles_are_never_contacted(monkeypatch):
    contacted = []

    def fail(*a, **k):
        contacted.append((a, k))
        raise AssertionError("availability must not contact a hosted endpoint")

    monkeypatch.setattr(servers.requests, "get", fail)
    monkeypatch.setattr(servers.requests, "post", fail)
    cfg = _config()
    cfg.profiles = {"cloud": cfg.profiles["cloud"]}
    cfg.active_profile = "cloud"
    cfg.base_url = "https://api.example.com/v1"
    cfg.model = "hosted-model"

    # Hosted endpoints are not configured local servers, and the availability
    # pass only inspects the states it was handed: zero requests.
    assert servers.configured_local_servers(cfg) == []
    assert gui_server._profile_availability(cfg, []) == {"cloud": "unknown"}
    assert contacted == []


# -------------------------------------------------------------- switch guard ---

def test_profile_switch_refuses_known_unavailable_and_fails_open(gui):
    _write(gui)
    with core.STATE.lock:
        core.STATE.profile_availability = {
            "present": "available", "absent": "unavailable",
            "empty": "unknown", "offline": "offline", "cloud": "unknown",
        }

    status, j = gui.post("/api/profile", {"profile": "absent"})
    assert status == 409 and j["ok"] is False
    assert "deleted-model" in j["detail"]
    assert "not available" in j["detail"]
    assert "may have been deleted" in j["detail"]
    assert "Settings" in j["detail"]
    assert gui.get("/api/state")[1]["profile"] == "present"

    # Unknown and offline both fail open; the switch succeeds.
    assert gui.post("/api/profile", {"profile": "empty"})[0] == 200
    assert gui.post("/api/profile", {"profile": "offline"})[0] == 200
    assert gui.post("/api/profile", {"profile": "cloud"})[0] == 200

    # No cache yet: fail open even for the profile that was unavailable.
    with core.STATE.lock:
        core.STATE.profile_availability = {}
    status, j = gui.post("/api/profile", {"profile": "absent"})
    assert status == 200 and j["ok"] is True
    assert j["profile"] == "absent" and j["model"] == "deleted-model"


# -------------------------------------------------------------------- picker ---

def test_profile_select_disables_unavailable_and_offline_options():
    src = (ROOT / "pedacito" / "gui" / "static" / "app.js").read_text(
        encoding="utf-8")

    badge = src[src.index("function HardwareBadge"):
                src.index("\nfunction serversDialog", src.index(
                    "function HardwareBadge"))]
    assert "S.profile_availability = j.profile_availability" in badge
    assert "setInterval(load, 15000)" in badge

    block = src[src.index("function ProfileSelect"):
                src.index("\n// ------------------------------------------------- calibration",
                          src.index("function ProfileSelect"))]
    assert "S.profile_availability" in block
    assert "state === 'unavailable'" in block
    assert "disabled: true" in block
    assert "unavailable" in block
    assert "server offline" in block
    assert "was not found on its server" in block
    assert "update the profile in Settings" in block
    assert "disabled=${o.disabled}" in block
    # The default option is rendered before and apart from the profile map, so
    # it can never be disabled.
    assert '<option value="">(default' in block
