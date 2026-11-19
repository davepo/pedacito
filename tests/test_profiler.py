"""Profiler: hardware readers, footprint math, tuning, probe and cache.

Every test is hermetic: readers receive injected fake output, HTTP is mocked
at the ``requests`` seam, and no test needs a GPU, a server or the network.
"""

from __future__ import annotations

import copy
import json
import time

import pytest

from pedacito import config as config_mod
from pedacito import profiler
from pedacito.config import Config
import pedacito.llm as llm


GIB = 1024 ** 3


# --- snapshot readers ---------------------------------------------------------

def test_snapshot_nvidia_smi_csv_with_injected_reader(monkeypatch, tmp_path):
    calls = []

    def run(argv, timeout=2.0):
        calls.append(list(argv))
        return "0, 24576, 4096\n1, 8192, 1024\n"

    monkeypatch.setattr(
        profiler.shutil, "which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(profiler, "_run_argv", run)
    snap = profiler.snapshot(meminfo_path=tmp_path / "missing",
                             drm_root=tmp_path / "drm", platform="linux")
    assert calls and calls[0][0].endswith("nvidia-smi")
    assert "--format=csv,noheader,nounits" in calls[0]
    assert snap["vram"]["total_gb"] == 32.0
    assert snap["vram"]["used_gb"] == 5.0
    assert snap["vram"]["free_gb"] == 27.0
    assert snap["sources"]["vram_total_gb"] == "nvidia-smi"
    assert [c["card"] for c in snap["vram"]["cards"]] == ["nvidia0", "nvidia1"]
    assert snap["vram"]["cards"][0]["free_gb"] == 20.0
    assert snap["available"] is True
    json.dumps(snap)                       # JSON-safe by construction


def test_snapshot_macos_sysctl_and_vm_stat(monkeypatch, tmp_path):
    def run(argv, timeout=2.0):
        if argv[0].endswith("sysctl"):
            return "17179869184\n"               # 16 GiB
        if argv[0].endswith("vm_stat"):
            return (
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                "Pages free:                               1000.\n"
                "Pages inactive:                           2000.\n"
                "Pages speculative:                         500.\n"
                "Pages wired down:                          100.\n")
        return ""

    monkeypatch.setattr(
        profiler.shutil, "which",
        lambda name: f"/usr/bin/{name}" if name in ("sysctl", "vm_stat") else None)
    monkeypatch.setattr(profiler, "_run_argv", run)
    snap = profiler.snapshot(meminfo_path=tmp_path / "missing",
                             drm_root=tmp_path / "drm", platform="darwin")
    assert snap["ram"]["total_gb"] == 16.0
    assert snap["ram"]["free_gb"] == round(3500 * 16384 / GIB, 2)
    assert snap["sources"]["ram_total_gb"] == "vm_stat"
    assert snap["unified_memory"] is True
    # Apple Silicon unified memory is RAM, never a separate VRAM claim.
    assert snap["vram"]["total_gb"] == 0.0
    assert snap["vram"]["cards"] == []


def test_snapshot_windows_global_memory_status(monkeypatch, tmp_path):
    class FakeKernel32:
        def GlobalMemoryStatusEx(self, ptr):
            status = ptr._obj
            status.ullTotalPhys = 32 * GIB
            status.ullAvailPhys = 18 * GIB
            return 1

    monkeypatch.setattr(profiler.shutil, "which", lambda name: None)
    snap = profiler.snapshot(meminfo_path=tmp_path / "missing",
                             drm_root=tmp_path / "drm", platform="win32",
                             kernel32=FakeKernel32(), include_nvidia=False)
    assert snap["ram"]["total_gb"] == 32.0
    assert snap["ram"]["free_gb"] == 18.0
    assert snap["sources"]["ram_total_gb"] == "winapi"
    assert snap["vram"]["total_gb"] == 0.0
    assert snap["available"] is True


def test_snapshot_linux_drm_sysfs_keeps_get_hardware_memory_shape(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16777216 kB\n"
                       "MemAvailable:    8388608 kB\n", encoding="ascii")
    device = tmp_path / "drm" / "card0" / "device"
    device.mkdir(parents=True)
    (device / "mem_info_vram_total").write_text(str(8 * GIB), encoding="ascii")
    (device / "mem_info_vram_used").write_text(str(2 * GIB), encoding="ascii")
    snap = profiler.snapshot(meminfo_path=meminfo, drm_root=tmp_path / "drm",
                             platform="linux", include_nvidia=False,
                             which=lambda name: None)
    assert snap["ram"]["total_gb"] == 16.0
    assert snap["ram"]["free_gb"] == 8.0
    assert snap["vram"]["total_gb"] == 8.0
    assert snap["vram"]["used_gb"] == 2.0
    assert snap["sources"]["ram_total_gb"] == "drm-sysfs"
    assert snap["sources"]["vram_total_gb"] == "drm-sysfs"
    assert snap["vram"]["cards"][0]["source"] == "drm-sysfs"


# --- footprint math -----------------------------------------------------------

def test_footprint_dense_14b_q4_golden():
    arch = {"total_params_b": 14, "n_layers": 48, "n_kv_heads": 8,
            "head_dim": 128, "source": "config"}
    fp = profiler.estimate_model_footprint(arch, "q4", 8192)
    assert fp["resident_gb"] == 7.0
    assert fp["vram_resident_gb"] == 7.0
    assert fp["kv_gb_per_1k_tokens"] == pytest.approx(0.045776, abs=1e-6)
    assert fp["kv_gb"] == pytest.approx(0.375, abs=1e-6)
    assert fp["arch"] == "dense"
    assert fp["confidence"] >= 0.9


def test_footprint_moe_30b_a3b_q4_golden():
    arch = {"total_params_b": 30, "active_params_b": 3, "is_moe": True,
            "n_layers": 48, "n_kv_heads": 4, "head_dim": 128,
            "source": "config"}
    fp = profiler.estimate_model_footprint(arch, "q4", 8192)
    # All experts live in RAM; only the active experts are VRAM-resident.
    assert fp["resident_gb"] == 15.0
    assert fp["active_gb"] == 1.5
    assert fp["vram_resident_gb"] == 1.5
    assert fp["kv_gb_per_1k_tokens"] == pytest.approx(0.022888, abs=1e-6)
    assert fp["arch"] == "moe"


def test_footprint_missing_metadata_is_low_confidence():
    fp = profiler.estimate_model_footprint("mystery-model", "q4", 8192)
    assert fp["confidence"] < 0.5
    assert fp["resident_gb"] == 0.0
    assert fp["estimated_fields"]

    named = profiler.estimate_model_footprint("some-14b-model", "", 8192)
    assert named["confidence"] < 0.5
    assert named["resident_gb"] == 8.4           # 14B * conservative 0.6
    assert "quant" in named["estimated_fields"]


# --- one shared KV model (config.auto_fit + profiler) -------------------------

_REFERENCE_SIZE = 7_000_000_000
_REFERENCE_ARCH = {"total_params_b": 14, "n_layers": 40, "n_kv_heads": 8,
                   "head_dim": 128, "source": "config"}


def test_shared_kv_model_agrees_on_dense_14b_q4_reference():
    """The report's dense-14B-Q4-on-24GB case: the footprint used to assume
    40,960 bytes/token; auto_fit used the file-size heuristic (213,623).
    Both now come from the same helper and agree."""
    cfg = Config()
    cfg.vram_gb = 24
    cfg.model_size_bytes = _REFERENCE_SIZE
    fp = profiler.estimate_model_footprint(
        _REFERENCE_ARCH, "q4", 8192, cfg=cfg)
    expected = max(16 * 1024, _REFERENCE_SIZE // 32768)
    assert expected == 213_623
    assert fp["kv_bytes_per_token"] == expected
    assert fp["kv_gb_per_1k_tokens"] == pytest.approx(
        expected * 1000.0 / GIB, abs=1e-6)
    # The old architecture-only number (2*L*H*D * q4 bytes) is the 5.2x
    # divergence the review measured.
    assert fp["kv_bytes_per_token"] != 40_960
    free = 24 * GIB - _REFERENCE_SIZE - GIB // 2
    assert config_mod.hardware_context_tokens(
        24, _REFERENCE_SIZE) == int(free // expected)


def test_q8_cache_enlarges_the_kv_estimate_in_both_paths():
    """q8_0 is a bigger cache than the q4_0 the old tuner assumed. Both the
    footprint and auto_fit must honor it and reach the same bytes/token."""
    q4_cfg = Config()
    q4_cfg.vram_gb = 24
    q4_cfg.model_size_bytes = _REFERENCE_SIZE
    q4_cfg.k_cache_quant = "q4_0"
    q8_cfg = copy.copy(q4_cfg)
    q8_cfg.k_cache_quant = "q8_0"
    q4_fp = profiler.estimate_model_footprint(
        _REFERENCE_ARCH, "q4", 8192, cfg=q4_cfg)
    q8_fp = profiler.estimate_model_footprint(
        _REFERENCE_ARCH, "q4", 8192, cfg=q8_cfg)
    assert q8_fp["kv_bytes_per_token"] > q4_fp["kv_bytes_per_token"]
    # It is also bigger than the old, quantization-blind footprint.
    legacy = profiler.estimate_model_footprint(_REFERENCE_ARCH, "q4", 8192)
    assert legacy["kv_bytes_per_token"] == 40_960
    assert q8_fp["kv_bytes_per_token"] > legacy["kv_bytes_per_token"]
    # config.auto_fit's path sees the same larger cache in tokens.
    assert (config_mod.hardware_context_tokens(24, _REFERENCE_SIZE, "q8_0")
            < config_mod.hardware_context_tokens(24, _REFERENCE_SIZE, "q4_0"))
    base = max(16 * 1024, _REFERENCE_SIZE // 32768)
    expected_q8 = max(4 * 1024, int(base * (8.5 + 16) / 32.0))
    assert q8_fp["kv_bytes_per_token"] == expected_q8
    free = 24 * GIB - _REFERENCE_SIZE - GIB // 2
    assert (config_mod.hardware_context_tokens(24, _REFERENCE_SIZE, "q8_0")
            == int(free // expected_q8))


def test_explicit_kv_bytes_per_token_bypasses_both_paths():
    cfg = Config()
    cfg.vram_gb = 24
    cfg.model_size_bytes = _REFERENCE_SIZE
    cfg.k_cache_quant = "q8_0"
    cfg.v_cache_quant = "q5_1"
    cfg.kv_cache_bytes_per_token = 96 * 1024
    fp = profiler.estimate_model_footprint(
        _REFERENCE_ARCH, "q4", 8192, cfg=cfg)
    assert fp["kv_bytes_per_token"] == 96 * 1024
    free = 24 * GIB - _REFERENCE_SIZE - GIB // 2
    assert config_mod.hardware_context_tokens(
        24, _REFERENCE_SIZE, "q8_0", "q5_1", 96 * 1024) == int(
            free // (96 * 1024))
    # The same bypass works through explicit estimate arguments (no cfg).
    direct = profiler.estimate_model_footprint(
        _REFERENCE_ARCH, "q4", 8192, model_size_bytes=_REFERENCE_SIZE,
        k_cache_quant="q8_0", v_cache_quant="q5_1",
        kv_cache_bytes_per_token=96 * 1024)
    assert direct["kv_bytes_per_token"] == 96 * 1024


def test_tune_growth_is_capped_by_hardware_context_tokens():
    """A MoE footprint counts only active weights in VRAM, so its growth
    proposal can exceed auto_fit's whole-file estimate. The cap keeps the
    tuner from suggesting a window the server's q8 cache cannot hold."""
    cfg = _cfg(vram_gb=24, model_size_bytes=15_000_000_000,
               k_cache_quant="q8_0")
    arch = {"total_params_b": 30, "active_params_b": 3, "is_moe": True,
            "n_layers": 48, "n_kv_heads": 4, "head_dim": 128,
            "source": "config"}
    fp = profiler.estimate_model_footprint(arch, "q4", 8192, cfg=cfg)
    out = profiler.tune_profile(
        cfg, _snap(vram_total=24.0, vram_free=20.0), fp, None,
        server_window=262144)
    cap = config_mod.hardware_context_tokens(
        24, cfg.model_size_bytes, cfg.k_cache_quant)
    assert cap > 0
    assert out["values"]["context_tokens"] == 8192   # growth is not applied
    assert out["proposed_context_tokens"] == cap
    assert any("capped" in note for note in out["notes"])

    q4_cfg = copy.copy(cfg)
    q4_cfg.k_cache_quant = "q4_0"
    q4_fp = profiler.estimate_model_footprint(arch, "q4", 8192, cfg=q4_cfg)
    q4_out = profiler.tune_profile(
        q4_cfg, _snap(vram_total=24.0, vram_free=20.0), q4_fp, None,
        server_window=262144)
    assert out["proposed_context_tokens"] < q4_out["proposed_context_tokens"]


# --- tuning decisions ---------------------------------------------------------

def _cfg(**overrides) -> Config:
    cfg = Config()
    cfg.auto_tune = True
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _dense_footprint():
    return profiler.estimate_model_footprint(
        {"total_params_b": 14, "n_layers": 48, "n_kv_heads": 8,
         "head_dim": 128, "source": "config"}, "q4", 8192)


def _snap(vram_total=24.0, vram_free=20.0, ram_total=64.0,
          ram_free=48.0, sources=None, unified=False):
    vram_used = max(0.0, vram_total - vram_free)
    ram_used = max(0.0, ram_total - ram_free)
    src = {"ram_total_gb": "drm-sysfs", "ram_free_gb": "drm-sysfs",
           "vram_total_gb": "drm-sysfs", "vram_free_gb": "drm-sysfs"}
    src.update(sources or {})
    return {
        "available": True,
        "unified_memory": unified,
        "ram": {"total_gb": ram_total, "used_gb": ram_used,
                "free_gb": ram_free, "available_gb": ram_free},
        "vram": {"total_gb": vram_total, "used_gb": vram_used,
                 "free_gb": vram_free, "cards": []},
        "sources": src,
    }


def test_tuning_tight_vram_shrinks_to_safety_floor():
    cfg = _cfg()
    out = profiler.tune_profile(cfg, _snap(vram_total=8.0, vram_free=2.0),
                                _dense_footprint(), None)
    assert out["applied"] is True
    assert out["values"]["context_tokens"] == 4096
    assert out["sources"]["context_tokens"].startswith("auto-tune")
    assert "4,096" in out["sources"]["context_tokens"]


def test_low_confidence_name_only_footprint_proposes_without_applying():
    """R3.1: a 0.10-confidence footprint is a guess. The tuner may show the
    would-be cuts, but must leave every config value alone."""
    cfg = _cfg(model="some-14b-model")
    before = copy.deepcopy(cfg.__dict__)
    fp = profiler.estimate_model_footprint("some-14b-model", "", 8192)
    assert fp["confidence"] < 0.5
    out = profiler.tune_profile(cfg, _snap(vram_total=8.0, vram_free=2.0),
                                fp, None)
    assert out["applied"] is False
    assert out["values"]["context_tokens"] == 8192
    assert out["values"]["max_gathered_chars"] == 24000
    assert out["values"]["flat_map_token_budget"] == 4000
    assert out["sources"] == {}
    text = " ".join(out["notes"])
    assert "would shrink" in text and "proposal only" in text
    assert "metadata guessed" in text
    assert "model_layers" in text and "pdc profile --run" in text

    out2 = profiler.apply_auto_tune(
        cfg, snap=_snap(vram_total=8.0, vram_free=2.0), footprint=fp,
        probe={})
    assert out2["applied"] is False
    assert cfg.__dict__ == before


def test_high_confidence_metadata_still_shrinks_to_floor():
    cfg = _cfg(model="some-14b-model")
    fp = profiler.estimate_model_footprint(
        {"total_params_b": 14, "n_layers": 48, "n_kv_heads": 8,
         "head_dim": 128, "source": "llama.cpp /props"}, "q4", 8192)
    assert fp["confidence"] >= 0.5
    out = profiler.tune_profile(cfg, _snap(vram_total=8.0, vram_free=2.0),
                                fp, None)
    assert out["applied"] is True
    assert out["values"]["context_tokens"] == 4096
    assert out["values"]["max_gathered_chars"] == 2728
    assert out["values"]["flat_map_token_budget"] == 800
    assert not any("metadata guessed" in n for n in out["notes"])


def test_auto_tune_source_precedence_is_one_way():
    """R3.3: an auto-tune write is explicit to config.auto_fit (so it is not
    replaced) but soft to the tuner itself (so a later load can refine it)."""
    source = ("auto-tune (VRAM 19.8 GB free via nvidia-smi; weights 7.0 GB; "
              "KV 0.1989 GB/1k tok; confidence 0.95; floor 4096)")
    assert config_mod._is_explicit_limit(source) is True
    cfg = _cfg()
    cfg.sources["context_tokens"] = source
    assert profiler._is_explicit(cfg, "context_tokens") is False
    # A user value is explicit to both subsystems.
    cfg.sources["context_tokens"] = "config file"
    assert config_mod._is_explicit_limit("config file") is True
    assert profiler._is_explicit(cfg, "context_tokens") is True


def test_tuning_roomy_vram_proposes_growth_only():
    cfg = _cfg()
    out = profiler.tune_profile(
        cfg, _snap(vram_total=48.0, vram_free=40.0), _dense_footprint(),
        None, server_window=32768)
    assert out["values"]["context_tokens"] == 8192      # never forced
    assert out["proposed_context_tokens"] == 32768
    assert "proposed" in " ".join(out["notes"])


def test_tuning_floor_respects_smaller_user_value():
    cfg = _cfg(context_tokens=2048)
    out = profiler.tune_profile(cfg, _snap(vram_total=4.0, vram_free=0.5),
                                _dense_footprint(), None)
    assert out["values"]["context_tokens"] == 2048


def test_tuning_hosted_cloud_has_no_hardware_tuning():
    cfg = _cfg(base_url="https://api.example.com/v1")
    out = profiler.tune_profile(cfg, _snap(), _dense_footprint(), None)
    assert out["values"]["context_tokens"] == 8192
    assert "context_tokens" not in (out["sources"] or {})
    assert "max_gathered_chars" not in (out["sources"] or {})
    # The one static hosted rule is the shorter default timeout.
    assert out["values"]["timeout"] == 30.0


def test_tuning_tiered_moe_never_raises_concurrency():
    cfg = _cfg(provider="strata", model="qwen3-30b-a3b", max_concurrency=4)
    out = profiler.tune_profile(cfg, _snap(), _dense_footprint(), None)
    assert out["values"]["max_concurrency"] <= 1
    assert out["values"]["timeout"] >= 600.0


def test_tuning_without_vram_abstains_on_non_unified_platform():
    cfg = _cfg(base_url="http://127.0.0.1:1234/v1")
    snap = _snap(vram_total=0.0, vram_free=0.0, ram_total=32.0,
                 ram_free=8.0, sources={"ram_total_gb": "winapi",
                                        "ram_free_gb": "winapi"})
    out = profiler.tune_profile(cfg, snap, _dense_footprint(), None)
    assert out["applied"] is False
    assert out["reason"] == "no memory telemetry"
    assert out["values"]["context_tokens"] == 8192
    # A missing VRAM reader is not zero free memory: no silent shrink.
    assert out["sources"] == {}


def test_tuning_with_unified_memory_uses_ram_without_shrinking():
    cfg = _cfg()
    snap = _snap(vram_total=0.0, vram_free=0.0, ram_total=64.0,
                 ram_free=40.0, unified=True,
                 sources={"ram_total_gb": "vm_stat",
                          "ram_free_gb": "vm_stat"})
    out = profiler.tune_profile(cfg, snap, _dense_footprint(), None,
                                server_window=16384)
    assert out["values"]["context_tokens"] >= 8192
    assert out.get("proposed_context_tokens", 0) >= 8192


def test_tuning_derives_free_vram_from_total_and_used():
    cfg = _cfg()
    snap = {"vram": {"total_gb": 8.0, "used_gb": 6.0},
            "ram": {}, "sources": {"vram_total_gb": "drm-sysfs"}}
    out = profiler.tune_profile(cfg, snap, _dense_footprint(), None)
    assert out["applied"] is True
    assert out["values"]["context_tokens"] == 4096


def test_tuning_macos_unknown_free_memory_abstains():
    cfg = _cfg()
    snap = {"unified_memory": True,
            "ram": {"total_gb": 32.0, "used_gb": 0.0, "free_gb": 0.0},
            "vram": {},
            "sources": {"ram_total_gb": "vm_stat"}}   # vm_stat gave no free
    out = profiler.tune_profile(cfg, snap, _dense_footprint(), None)
    assert out["applied"] is False
    assert out["values"]["context_tokens"] == 8192


def test_tuning_user_pinned_values_win():
    cfg = _cfg()
    cfg.sources["context_tokens"] = "profile 'x' (explicit)"
    cfg.sources["max_gathered_chars"] = "profile 'x' (explicit)"
    cfg.sources["flat_map_token_budget"] = "profile 'x' (explicit)"
    cfg.sources["timeout"] = "profile 'x' (explicit)"
    cfg.sources["max_concurrency"] = "profile 'x' (explicit)"
    out = profiler.tune_profile(cfg, _snap(vram_total=4.0, vram_free=0.2),
                                _dense_footprint(), None)
    assert out["applied"] is False
    assert out["reason"] == "user-pinned"
    assert out["values"]["context_tokens"] == 8192


def test_apply_auto_tune_opt_out_is_byte_identical():
    cfg = Config()
    before = copy.deepcopy(cfg.__dict__)
    out = profiler.apply_auto_tune(cfg, snap=_snap())
    assert out["applied"] is False
    assert cfg.__dict__ == before


def test_apply_auto_tune_no_telemetry_leaves_config_identical():
    cfg = _cfg()
    before = copy.deepcopy(cfg.__dict__)
    empty = {"available": False, "unified_memory": False, "ram": {},
             "vram": {}, "sources": {}}
    out = profiler.apply_auto_tune(cfg, snap=empty)
    assert out["applied"] is False
    assert cfg.__dict__ == before


# --- speed probe --------------------------------------------------------------

class _SSE:
    status_code = 200
    encoding = "utf-8"
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, frames, delay=0.005):
        self.frames = frames
        self.delay = delay

    def iter_lines(self, decode_unicode=True):
        for frame in self.frames:
            time.sleep(self.delay)
            yield "data: " + json.dumps(frame)

    def close(self):
        pass


def _sse_frames(timings=None, prompt_tokens=2000, completion_tokens=1):
    frames = [
        {"choices": [{"delta": {"content": "O"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "K"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": prompt_tokens,
                                  "completion_tokens": completion_tokens}},
        {"choices": []},
    ]
    if timings is not None:
        frames[3]["timings"] = timings
    return frames


def test_probe_uses_sse_server_timings(monkeypatch):
    def fake_post(url, **kw):
        return _SSE(_sse_frames({"prompt_per_second": 850.0,
                                 "predicted_per_second": 42.0}))

    monkeypatch.setattr(llm.requests, "post", fake_post)
    out = profiler.probe_model_speed(
        "http://127.0.0.1:1234/v1", "probe-model", Config(),
        force=True, context_window=8192)
    assert out["ok"] is True
    assert out["prefill_tok_per_s"] == pytest.approx(850.0)
    assert out["generation_tok_per_s"] == pytest.approx(42.0)
    assert out["probe_tokens"] == 2048
    assert out["cached"] is False


def test_probe_measures_stream_timings_without_server_timings(monkeypatch):
    def fake_post(url, **kw):
        return _SSE(_sse_frames(timings=None, prompt_tokens=1800,
                                completion_tokens=2), delay=0.01)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    out = profiler.probe_model_speed(
        "http://127.0.0.1:1234/v1", "probe-model-2", Config(),
        force=True, context_window=8192)
    assert out["ok"] is True
    assert out["prefill_tok_per_s"] > 0
    assert out["generation_tok_per_s"] > 0
    assert out["prefill_tok_per_s"] < 1e7
    assert out["generation_tok_per_s"] < 1e7


def test_probe_http_error_is_silent_and_not_cached(monkeypatch):
    class Bad:
        status_code = 400
        encoding = "utf-8"
        headers = {"Content-Type": "application/json"}
        text = "bad request"

        def close(self):
            pass

    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: Bad())
    before = profiler.cache_path()
    out = profiler.probe_model_speed(
        "http://127.0.0.1:1234/v1", "bad-model", Config(),
        force=True, context_window=8192)
    assert out["ok"] is False
    assert "detail" in out
    assert profiler.read_cached_probe(
        "http://127.0.0.1:1234/v1", "bad-model") is None
    assert not before.exists() or "bad-model" not in before.read_text(
        encoding="utf-8")


def test_cli_profile_run_refuses_hosted_endpoint(monkeypatch, tmp_path,
                                                 capsys):
    from pedacito import cli
    monkeypatch.chdir(tmp_path)
    cfgfile = tmp_path / "config.toml"
    cfgfile.write_text(
        'default_profile = "hosted"\n'
        '[profiles.hosted]\n'
        'base_url = "https://api.deepseek.com/v1"\n'
        'model = "deepseek-chat"\n', encoding="utf-8")
    built: list[str] = []

    class RecordingClient:
        def __init__(self, cfg):
            built.append(str(cfg.base_url))
            raise AssertionError("hosted endpoint must not build a client")

    monkeypatch.setattr(llm, "LLMClient", RecordingClient)
    rc = cli.main(["profile", "--run", "--config", str(cfgfile)])
    err = capsys.readouterr().err
    assert rc == 1
    assert built == []
    assert "hosted model" in err
    assert "refusing to probe" in err


def test_cli_profile_run_probes_local_endpoint(monkeypatch, tmp_path, capsys):
    from pedacito import cli
    monkeypatch.chdir(tmp_path)
    cfgfile = tmp_path / "config.toml"
    cfgfile.write_text(
        'default_profile = "local"\n'
        '[profiles.local]\n'
        'base_url = "http://127.0.0.1:1234/v1"\n'
        'model = "local-model"\n', encoding="utf-8")
    built: list[str] = []
    chats: list[dict] = []

    class ProbeClient:
        def __init__(self, cfg):
            built.append(str(cfg.base_url))
            self.last_timings = {"prompt_per_second": 800.0,
                                 "predicted_per_second": 40.0}
            self.last_elapsed_s = 0.25
            self.first_token_at = 0.10
            self.request_started_at = 0.0
            self.last_prompt_tokens = 2048
            self.last_completion_tokens = 2
            self.last_stream_tok_per_s = 0.0

        def _chat_once(self, messages, **kw):
            chats.append(kw)
            return "OK"

        def estimate_tokens(self, text):
            return max(1, len(str(text)) // 4)

    monkeypatch.setattr(llm, "LLMClient", ProbeClient)
    monkeypatch.setattr(profiler, "detect_context_window",
                        lambda *a, **k: 8192)
    monkeypatch.setattr(profiler, "resolve_architecture",
                        lambda *a, **k: {})
    monkeypatch.setattr(profiler, "snapshot",
                        lambda *a, **k: {"available": False, "ram": {},
                                         "vram": {}, "sources": {}})
    rc = cli.main(["profile", "--run", "--config", str(cfgfile)])
    out = capsys.readouterr().out
    assert rc == 0
    assert built == ["http://127.0.0.1:1234/v1"]
    assert len(chats) == 1
    assert "speed probe (this run)" in out


def test_probe_cache_ttl_and_model_invalidation(monkeypatch):
    now = 1_800_000_000.0
    result = {"ok": True, "base_url": "http://127.0.0.1:1234/v1",
              "model": "model-a", "prefill_tok_per_s": 100.0,
              "generation_tok_per_s": 10.0, "measured_at": now}
    profiler.store_probe(result)
    assert profiler.read_cached_probe("http://127.0.0.1:1234/v1",
                                      "model-a", now=now + 10) is not None
    assert profiler.read_cached_probe("http://127.0.0.1:1234/v1",
                                      "model-a",
                                      now=now + profiler.PROBE_TTL + 1) is None
    # Switching models never returns the old measurement.
    assert profiler.read_cached_probe("http://127.0.0.1:1234/v1",
                                      "model-b", now=now + 10) is None
    profiler.clear_cache()
    assert profiler.read_cached_probe("http://127.0.0.1:1234/v1",
                                      "model-a", now=now + 10) is None


def test_successful_server_load_schedules_the_probe(monkeypatch):
    from pedacito import servers
    calls = []
    monkeypatch.setattr(servers, "_server_action", lambda *a, **k: {
        "ok": True, "base_url": "http://127.0.0.1:1234/v1",
        "model": "loaded-model"})
    monkeypatch.setattr(profiler, "maybe_probe_async",
                        lambda cfg, **kw: calls.append(kw))
    cfg = Config()
    servers.load_server_model("x", "loaded-model", cfg)
    assert calls == [{"base_url": "http://127.0.0.1:1234/v1",
                      "model": "loaded-model"}]


def test_failed_server_load_does_not_schedule_a_probe(monkeypatch):
    from pedacito import servers
    calls = []
    monkeypatch.setattr(servers, "_server_action",
                        lambda *a, **k: {"ok": False})
    monkeypatch.setattr(profiler, "maybe_probe_async",
                        lambda cfg, **kw: calls.append(kw))
    servers.load_server_model("x", "loaded-model", Config())
    assert calls == []


# --- wiring -------------------------------------------------------------------

def test_config_load_opt_out_never_touches_the_hardware_reader(
        monkeypatch, tmp_path):
    def boom(**_kw):
        raise AssertionError("auto-tune read hardware while opted out")

    monkeypatch.setattr(profiler, "snapshot", boom)
    path = tmp_path / "config.toml"
    path.write_text(
        "[profiles.x]\n"
        'base_url = "http://127.0.0.1:1234/v1"\n'
        'model = "some-14b"\n'
        "vram_gb = 8\n", encoding="utf-8")
    cfg = config_mod.load(root=tmp_path, explicit=path, profile="x",
                          require_hardware=False, warn_budgets=False)
    assert cfg.auto_tune is False
    assert not cfg.sources["context_tokens"].startswith("auto-tune")
    assert cfg.context_tokens > 0


def test_config_load_auto_tune_uses_the_snapshot(monkeypatch, tmp_path):
    def fake_snapshot(**_kw):
        return _snap(vram_total=8.0, vram_free=2.0)

    monkeypatch.setattr(profiler, "snapshot", fake_snapshot)
    path = tmp_path / "config.toml"
    path.write_text(
        "[profiles.x]\n"
        'base_url = "http://127.0.0.1:1234/v1"\n'
        'model = "some-14b"\n'
        "model_params_b = 14\n"
        "model_layers = 48\n"
        "model_kv_heads = 8\n"
        "model_head_dim = 128\n"
        'model_quant = "q4"\n'
        "vram_gb = 8\n"
        "auto_tune = true\n", encoding="utf-8")
    cfg = config_mod.load(root=tmp_path, explicit=path, profile="x",
                          require_hardware=False, warn_budgets=False)
    assert cfg.auto_tune is True
    assert cfg.context_tokens == 4096
    assert cfg.sources["context_tokens"].startswith("auto-tune")


def test_models_and_profile_commands_parse():
    from pedacito.cli import build_parser
    parser = build_parser()
    assert parser.parse_args(["profile"]).cmd == "profile"
    assert parser.parse_args(["profile", "--run"]).run is True
    assert parser.parse_args(["profile", "--reset"]).reset is True
    assert parser.parse_args(["models"]).action == "show"
    assert parser.parse_args(["models", "scan"]).action == "scan"


def test_how_card_and_docs_exist():
    from pathlib import Path
    from pedacito import how_index
    names = {card["name"] for card in how_index.CLI_COMMANDS}
    assert "profile" in names
    topics = {topic["id"]: topic for topic in how_index.GUI_TOPICS}
    assert "profiling-auto-tune" in topics
    assert "auto_tune" in topics["profiling-auto-tune"]["settings"]
    root = Path(__file__).resolve().parents[1]
    assert "auto_tune" in (root / "docs" / "CONFIG.md").read_text(
        encoding="utf-8")
    assert "Advanced telemetry and auto-tuning" in (
        root / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
