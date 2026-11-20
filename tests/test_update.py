"""Context-aware auto-update (pedacito.update): fully offline tests.

No network and no real pip/pipx ever run here: urlopen is monkeypatched,
upgrade subprocesses are either injected fakes or tiny `python -c` snippets,
and install-method detection is driven by monkeypatched paths/prefixes.
The conftest guard (PEDACITO_CHECK_FOR_UPDATES=false) keeps every other test
in the suite offline; feature tests re-enable it explicitly.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import __version__
from pedacito import cli
from pedacito import config as config_mod
from pedacito import update as update_mod
from pedacito.config import Config
from pedacito.gui import core
from pedacito.gui import server as gui_server
from test_gui import fake_port, gui, project  # noqa: F401  (shared GUI fixtures)
from test_tui import FakeClient


# --- helpers -------------------------------------------------------------------

class FakeResponse:
    """The context-manager response object urlopen returns."""

    def __init__(self, payload, status=200, raw=None):
        self._data = raw if raw is not None else json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def pypi_payload(latest="9.9.9", yanked=False):
    return {"info": {"version": latest},
            "releases": {latest: [{"yanked": yanked}],
                         "0.0.1": [{"yanked": False}]}}


def enable(monkeypatch):
    monkeypatch.setenv("PEDACITO_CHECK_FOR_UPDATES", "true")


def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call while updates are disabled/offline")
    monkeypatch.setattr(update_mod, "urlopen", boom)


def neutral_detection(monkeypatch, tmp_path, pkg=None):
    """Strip the real environment's signals: this repo IS an editable install
    (a live __editable__ finder sits in sys.meta_path), so detection tests
    must start from a clean slate."""
    monkeypatch.setattr(sys, "meta_path", [])
    monkeypatch.setattr(update_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(update_mod, "_package_dirs",
                        lambda: [pkg if pkg is not None else tmp_path / "pkg"])
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "base"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))


def transcript_text(session):
    return "\n".join(line for _kind, line in session.transcript)


def wait_until(fn, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        got = fn()
        if got:
            return got
        time.sleep(0.05)
    return None


# --- version comparison ----------------------------------------------------------

def test_parse_version_shapes():
    assert update_mod.parse_version("1.2.3") == ((1, 2, 3), 4, 0)
    assert update_mod.parse_version("v1.2") == ((1, 2), 4, 0)
    assert update_mod.parse_version("1.0rc1") == ((1, 0), 3, 1)
    assert update_mod.parse_version("1.0.0.dev2") == ((1, 0, 0), 0, 2)
    assert update_mod.parse_version("1.0.post1") == ((1, 0), 5, 1)
    assert update_mod.parse_version("1.0+build.5") == ((1, 0), 4, 0)
    assert update_mod.parse_version("banana") is None
    assert update_mod.parse_version("") is None
    assert update_mod.parse_version(None) is None


def test_prerelease_ordering():
    order = ["1.0.0.dev1", "1.0.0a1", "1.0.0a2", "1.0.0b1", "1.0.0rc1",
             "1.0.0", "1.0.0.post1", "1.0.1.dev1", "1.0.1"]
    for older, newer in zip(order, order[1:]):
        assert update_mod.is_newer(newer, older), (older, newer)
        assert not update_mod.is_newer(older, newer), (older, newer)


def test_is_newer_never_nags_down_or_equal():
    assert update_mod.is_newer("0.3.1", "0.3.0")
    assert not update_mod.is_newer("0.3.0", "0.3.0")
    assert not update_mod.is_newer("0.2.9", "0.3.0")
    assert not update_mod.is_newer("1.0", "1.0.0")      # zero-padding
    assert not update_mod.is_newer("1.0.0rc1", "1.0.0")  # pre < final
    assert update_mod.is_newer("1.0.0", "1.0.0rc1")
    assert not update_mod.is_newer("0.3.0+local", "0.3.0")  # build ignored
    assert not update_mod.is_newer("banana", "0.3.0")
    assert not update_mod.is_newer("0.3.1", "banana")    # conservative


# --- cache -----------------------------------------------------------------------

def test_cache_lives_beside_user_config():
    assert update_mod.cache_path().parent == config_mod.USER_CONFIG.parent
    assert update_mod.cache_path().name == "update_check.json"


def test_cache_roundtrip_and_ttl():
    now = 1_000_000.0
    assert update_mod.read_cache(now=now) is None       # missing file
    update_mod.write_cache("0.3.1", now=now)
    fresh = update_mod.read_cache(now=now + update_mod.CACHE_TTL - 1)
    assert fresh == {"checked_at": now, "latest": "0.3.1"}
    assert update_mod.read_cache(now=now + update_mod.CACHE_TTL) is None
    assert update_mod.read_cache(now=now - 120) is None  # future stamp


def test_cache_corrupt_reads_as_missing():
    path = update_mod.cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert update_mod.read_cache() is None
    path.write_text(json.dumps({"checked_at": "x", "latest": 3}),
                    encoding="utf-8")
    assert update_mod.read_cache() is None
    path.write_text(json.dumps({"checked_at": time.time(), "latest": ""}),
                    encoding="utf-8")
    assert update_mod.read_cache() is None


def test_write_cache_concurrent_checks_do_not_share_a_tmp_file(monkeypatch):
    """Two checks finishing at once (GUI startup thread + `pdc update`) must
    not write through one fixed `.tmp` name: the second writer truncates the
    first's temp file, the first os.replace then fails and its result is
    silently lost. Each writer needs its own unique temp file."""
    real_replace = update_mod.os.replace
    fired = []

    def racing_replace(src, dst):
        if not fired:
            fired.append(True)
            # A second check completes while this one is mid-write.
            update_mod.write_cache("9.9.9", now=2000.0)
        return real_replace(src, dst)

    monkeypatch.setattr(update_mod.os, "replace", racing_replace)
    update_mod.write_cache("0.3.1", now=1000.0)
    data = json.loads(update_mod.cache_path().read_text(encoding="utf-8"))
    assert data == {"checked_at": 1000.0, "latest": "0.3.1"}
    assert not list(update_mod.cache_path().parent.glob("*.tmp"))


# --- payload parsing ----------------------------------------------------------------

def test_latest_skips_yanked_and_empty_releases():
    payload = {"info": {"version": "9.9.9"},
               "releases": {"9.9.9": [{"yanked": True}],
                            "0.4.0": [],
                            "0.3.2": [{"yanked": False}],
                            "0.3.10": [{"yanked": False}]}}
    assert update_mod._latest_from_payload(payload) == "0.3.10"


def test_latest_falls_back_to_info_version():
    payload = {"info": {"version": "0.3.1"},
               "releases": {"0.3.1": [{"yanked": True}]}}
    assert update_mod._latest_from_payload(payload) == "0.3.1"
    assert update_mod._latest_from_payload({"info": {"version": "0.3.1"}}) == "0.3.1"
    assert update_mod._latest_from_payload(None) is None
    assert update_mod._latest_from_payload({"info": {"version": "junk"}}) is None
    # The fallback obeys the channel too.
    rc = {"info": {"version": "1.0.0rc1"}}
    assert update_mod._latest_from_payload(rc) is None
    assert update_mod._latest_from_payload(rc, channel="any") == "1.0.0rc1"


def test_latest_defaults_to_stable_and_skips_prereleases():
    payload = {"info": {"version": "1.0.0rc1"},
               "releases": {
                   "1.0.0.dev2": [{"yanked": False}],
                   "1.0.0a1": [{"yanked": False}],
                   "1.0.0b2": [{"yanked": False}],
                   "1.0.0rc1": [{"yanked": False}],
                   "0.9.9": [{"yanked": False}],
               }}
    assert update_mod._latest_from_payload(payload) == "0.9.9"
    assert update_mod._latest_from_payload(payload, channel="stable") == "0.9.9"
    assert update_mod._latest_from_payload(payload, channel="any") == "1.0.0rc1"


def test_stable_channel_still_ignores_yanked_releases():
    payload = {"info": {"version": "1.0.0"},
               "releases": {
                   "1.0.0": [{"yanked": True}],
                   "1.0.0rc1": [{"yanked": False}],
                   "0.9.9": [{"yanked": False}],
               }}
    assert update_mod._latest_from_payload(payload) == "0.9.9"
    assert update_mod._latest_from_payload(payload, channel="any") == "1.0.0rc1"


# --- the check: offline, disabled, cached ----------------------------------------------

def test_offline_is_a_silent_noop(monkeypatch):
    enable(monkeypatch)
    for exc in (OSError("no route"), TimeoutError("timed out"),
                ValueError("bad json")):
        def raise_exc(*a, _exc=exc, **k):
            raise _exc
        monkeypatch.setattr(update_mod, "urlopen", raise_exc)
        info = update_mod.check_for_update()
        assert info["ok"] is False and info["detail"] == "offline"
        assert info["update_available"] is False
        assert not update_mod.cache_path().exists()


def test_non_200_and_garbage_are_offline(monkeypatch):
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse({}, status=404))
    assert update_mod.check_for_update()["detail"] == "offline"
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(None, raw=b"<html>nope"))
    assert update_mod.check_for_update()["detail"] == "offline"


def test_fetch_uses_the_anonymous_pypi_url_with_2s_timeout(monkeypatch):
    enable(monkeypatch)
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append((url, timeout))
        return FakeResponse(pypi_payload())

    monkeypatch.setattr(update_mod, "urlopen", fake_urlopen)
    info = update_mod.check_for_update()
    assert calls == [("https://pypi.org/pypi/pedacito/json", 2.0)]
    assert info["ok"] is True and info["latest"] == "9.9.9"
    assert info["update_available"] is True
    assert info["current"] == __version__
    assert info["method"] and info["instructions"]
    assert update_mod.cache_path().exists()


def test_cache_hits_and_manual_bypass(monkeypatch):
    enable(monkeypatch)
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        return FakeResponse(pypi_payload())

    monkeypatch.setattr(update_mod, "urlopen", fake_urlopen)
    first = update_mod.check_for_update()
    assert first["cached"] is False and len(calls) == 1
    second = update_mod.check_for_update()          # fresh cache: no fetch
    assert second["cached"] is True and len(calls) == 1
    assert second["latest"] == "9.9.9"
    manual = update_mod.check_for_update(manual=True)  # bypasses the cache
    assert manual["cached"] is False and len(calls) == 2


def test_stale_cache_refetches(monkeypatch):
    enable(monkeypatch)
    now = 2_000_000.0
    update_mod.write_cache("0.0.1", now=now - update_mod.CACHE_TTL - 1)
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(pypi_payload("0.3.9")))
    info = update_mod.check_for_update(now=now)
    assert info["cached"] is False and info["latest"] == "0.3.9"


def test_no_warning_when_installed_is_newer(monkeypatch):
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(pypi_payload("0.0.1")))
    info = update_mod.check_for_update()
    assert info["ok"] is True and info["update_available"] is False


def test_disabled_means_zero_network_and_zero_subprocess(monkeypatch):
    # conftest already sets PEDACITO_CHECK_FOR_UPDATES=false
    no_network(monkeypatch)

    def no_proc(*a, **k):
        raise AssertionError("subprocess while disabled")

    monkeypatch.setattr(update_mod.subprocess, "run", no_proc)
    monkeypatch.setattr(update_mod.subprocess, "Popen", no_proc)
    info = update_mod.check_for_update()
    assert info["detail"] == "disabled" and info["ok"] is False
    assert info["instructions"]                     # manual command still shown
    assert not update_mod.cache_path().exists()
    assert update_mod.check_for_update_async() is None


def test_updates_enabled_precedence(monkeypatch):
    cfg = Config()
    monkeypatch.setenv("PEDACITO_CHECK_FOR_UPDATES", "false")
    assert update_mod.updates_enabled(cfg) is False
    monkeypatch.setenv("PEDACITO_CHECK_FOR_UPDATES", "true")
    cfg.check_for_updates = False
    assert update_mod.updates_enabled(cfg) is True  # env wins
    monkeypatch.delenv("PEDACITO_CHECK_FOR_UPDATES")
    assert update_mod.updates_enabled(cfg) is False  # then the loaded config
    cfg.check_for_updates = True
    assert update_mod.updates_enabled(cfg) is True
    # No cfg: a light read of the user config's [universal] table.
    config_mod.USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    config_mod.USER_CONFIG.write_text(
        "[universal]\ncheck_for_updates = false\n", encoding="utf-8")
    assert update_mod.updates_enabled(None) is False
    config_mod.USER_CONFIG.write_text("[universal]\n", encoding="utf-8")
    assert update_mod.updates_enabled(None) is True  # default


def test_update_channel_defaults_to_stable_and_any_admits_an_rc(monkeypatch):
    enable(monkeypatch)
    payload = {"info": {"version": "9.9.9rc1"},
               "releases": {"9.9.9rc1": [{"yanked": False}],
                            "0.3.1": [{"yanked": False}]}}
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(payload))
    stable = update_mod.check_for_update(manual=True)
    assert stable["channel"] == "stable"
    assert stable["latest"] == "0.3.1"
    assert stable["update_available"] is False
    cfg = Config()
    cfg.update_channel = "any"
    any_cfg = update_mod.check_for_update(cfg, manual=True)
    assert any_cfg["channel"] == "any"
    assert any_cfg["latest"] == "9.9.9rc1"
    assert any_cfg["update_available"] is True


def test_update_channel_env_override_admits_an_rc(monkeypatch):
    enable(monkeypatch)
    monkeypatch.setenv("PEDACITO_UPDATE_CHANNEL", "any")
    payload = {"info": {"version": "9.9.9rc1"},
               "releases": {"9.9.9rc1": [{"yanked": False}]}}
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(payload))
    info = update_mod.check_for_update(manual=True)
    assert info["channel"] == "any"
    assert info["latest"] == "9.9.9rc1"
    assert info["update_available"] is True


def test_update_channel_flows_through_config_and_env(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[universal]\nupdate_channel = "any"\n',
                        encoding="utf-8")
    cfg = config_mod.load(root=tmp_path, explicit=cfg_path,
                          require_hardware=False, warn_budgets=False)
    assert cfg.update_channel == "any"
    monkeypatch.setenv("PEDACITO_UPDATE_CHANNEL", "stable")
    cfg = config_mod.load(root=tmp_path, explicit=cfg_path,
                          require_hardware=False, warn_budgets=False)
    assert cfg.update_channel == "stable"


def test_stable_channel_refetches_a_cached_prerelease(monkeypatch):
    enable(monkeypatch)
    update_mod.write_cache("9.9.9rc1")
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        return FakeResponse(pypi_payload("0.3.2"))

    monkeypatch.setattr(update_mod, "urlopen", fake_urlopen)
    info = update_mod.check_for_update()
    assert calls == [update_mod.PYPI_URL]
    assert info["cached"] is False
    assert info["latest"] == "0.3.2"


def test_async_check_is_daemon_deduped_and_reports(monkeypatch):
    enable(monkeypatch)
    started, release = threading.Event(), threading.Event()
    results = []

    def slow_fetch(url, timeout):
        started.set()
        release.wait(10)
        return pypi_payload()

    t1 = update_mod.check_for_update_async(fetch=slow_fetch,
                                           on_done=results.append)
    assert t1 is not None and t1.daemon and t1.name == "pdc-update-check"
    assert started.wait(5)
    assert update_mod.check_for_update_async(fetch=slow_fetch) is None  # busy
    release.set()
    t1.join(5)
    assert not t1.is_alive() and not update_mod._check_running
    assert len(results) == 1 and results[0]["latest"] == "9.9.9"


# --- install-method detection -------------------------------------------------------

def test_detect_editable_via_git_dir(monkeypatch, tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    neutral_detection(monkeypatch, tmp_path, pkg)
    (pkg / ".git").mkdir()
    assert update_mod.detect_install_method() == "editable"


def test_detect_editable_via_parent_git(monkeypatch, tmp_path):
    pkg = tmp_path / "src" / "pedacito"
    pkg.mkdir(parents=True)
    neutral_detection(monkeypatch, tmp_path, pkg)
    (tmp_path / "src" / ".git").mkdir()
    assert update_mod.detect_install_method() == "editable"


def test_detect_editable_via_meta_path_finder(monkeypatch, tmp_path):
    neutral_detection(monkeypatch, tmp_path)
    finder = type("__editable___pedacito_0_3_0_finder", (), {})()
    monkeypatch.setattr(sys, "meta_path", [finder])
    assert update_mod.detect_install_method() == "editable"


def test_detect_editable_via_egg_link(monkeypatch, tmp_path):
    neutral_detection(monkeypatch, tmp_path)
    site = tmp_path / "site"
    site.mkdir()
    (site / "pedacito.egg-link").write_text("/src/pedacito\n", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(site)])
    assert update_mod.detect_install_method() == "editable"


def test_detect_pipx_via_path(monkeypatch, tmp_path):
    pkg = tmp_path / "pipx" / "venvs" / "pedacito" / "lib" / "pedacito"
    pkg.mkdir(parents=True)
    neutral_detection(monkeypatch, tmp_path, pkg)
    assert update_mod.detect_install_method() == "pipx"


def test_detect_pipx_via_pyvenv_cfg(monkeypatch, tmp_path):
    neutral_detection(monkeypatch, tmp_path)
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n# pipx\n",
                                     encoding="utf-8")
    monkeypatch.setattr(sys, "prefix", str(venv))
    assert update_mod.detect_install_method() == "pipx"


def test_detect_pipx_via_guarded_list(monkeypatch, tmp_path):
    neutral_detection(monkeypatch, tmp_path)
    monkeypatch.setattr(update_mod.shutil, "which", lambda name: "/bin/pipx")
    ran = []

    def fake_run(argv, **kw):
        ran.append(argv)
        assert kw.get("shell") is False
        return subprocess.CompletedProcess(argv, 0,
                                           stdout=json.dumps(
                                               {"venvs": {"pedacito": {}}}))

    monkeypatch.setattr(update_mod.subprocess, "run", fake_run)
    assert update_mod.detect_install_method() == "pipx"
    assert ran == [["/bin/pipx", "list", "--json"]]
    # query_pipx=False (the opt-out path) never spawns anything.
    ran.clear()
    assert update_mod.detect_install_method(query_pipx=False) == "user-pip"
    assert ran == []


def test_detect_pipx_list_failure_is_not_pipx(monkeypatch, tmp_path):
    neutral_detection(monkeypatch, tmp_path)
    monkeypatch.setattr(update_mod.shutil, "which", lambda name: "/bin/pipx")

    def fake_run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 3)

    monkeypatch.setattr(update_mod.subprocess, "run", fake_run)
    assert update_mod.detect_install_method() == "user-pip"


def test_detect_venv_and_user_pip(monkeypatch, tmp_path):
    neutral_detection(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    assert update_mod.detect_install_method() == "venv"
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "base"))
    assert update_mod.detect_install_method() == "user-pip"


def test_detect_unknown_without_package_dir(monkeypatch, tmp_path):
    neutral_detection(monkeypatch, tmp_path)
    monkeypatch.setattr(update_mod, "_package_dirs", lambda: [])
    assert update_mod.detect_install_method() == "unknown"


# --- upgrade commands ------------------------------------------------------------------

def test_upgrade_command_is_always_an_argv_list(monkeypatch):
    monkeypatch.setattr(update_mod.shutil, "which", lambda name: None)
    for method in ("pipx", "venv", "user-pip", "editable", "unknown"):
        argv, instructions = update_mod.upgrade_command(method)
        assert instructions
        if argv is not None:
            assert isinstance(argv, list)
            assert all(isinstance(a, str) for a in argv)
            assert not any("&&" in a or "|" in a for a in argv)


def test_upgrade_command_per_method(monkeypatch):
    monkeypatch.setattr(update_mod.shutil, "which",
                        lambda name: "/usr/local/bin/pipx" if name == "pipx"
                        else None)
    argv, _ = update_mod.upgrade_command("pipx")
    assert argv == ["/usr/local/bin/pipx", "upgrade", "pedacito"]
    argv, _ = update_mod.upgrade_command("venv")
    assert argv == [sys.executable, "-m", "pip", "install", "--upgrade",
                    "pedacito"]
    argv, note = update_mod.upgrade_command("user-pip")
    assert argv == [sys.executable, "-m", "pip", "install", "--upgrade",
                    "--user", "pedacito"]
    argv, note = update_mod.upgrade_command("editable")
    assert argv is None and "git pull" in note
    argv, note = update_mod.upgrade_command("unknown")
    assert argv is None and "pip" in note


def test_pep668_note(monkeypatch, tmp_path):
    import sysconfig
    # The marker detection itself: stdlib dir with/without EXTERNALLY-MANAGED.
    monkeypatch.setattr(sysconfig, "get_path", lambda _kind: str(tmp_path))
    assert update_mod.externally_managed() is False
    (tmp_path / "EXTERNALLY-MANAGED").write_text("[externally-managed]\n",
                                                 encoding="utf-8")
    assert update_mod.externally_managed() is True
    # And the instruction note it produces for a user-pip install.
    _argv, note = update_mod.upgrade_command("user-pip")
    assert "PEP 668" in note


# --- running the upgrade ------------------------------------------------------------------

def test_run_upgrade_streams_and_returns_exit_code():
    lines = []
    code = update_mod.run_upgrade(
        [sys.executable, "-c", "print('hello'); print('world')"],
        on_line=lines.append)
    assert code == 0 and lines == ["hello", "world"]


def test_run_upgrade_merges_stderr_and_propagates_failure():
    lines = []
    code = update_mod.run_upgrade(
        [sys.executable, "-c",
         "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"],
        on_line=lines.append)
    assert code == 3 and lines == ["boom"]


def test_run_upgrade_spawn_failure_is_exit_1():
    lines = []
    code = update_mod.run_upgrade(["/nonexistent/pedacito-test-binary"],
                                  on_line=lines.append)
    assert code == 1 and lines and "could not start" in lines[0]


def test_run_upgrade_timeout_kills_and_reports():
    lines = []
    code = update_mod.run_upgrade(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        on_line=lines.append, timeout=0.3)
    assert code == 1 and any("timed out" in ln for ln in lines)


def test_run_upgrade_timeout_stops_a_child_that_keeps_printing():
    """The deadline is wall-clock: a child that never goes silent must be
    stopped too. The old loop only checked the deadline on queue.Empty, so
    continuous output streamed past the timeout until the child exited."""
    chatter = ("import time\n"
               "end = time.monotonic() + 3.0\n"
               "while time.monotonic() < end:\n"
               "    print('chatter', flush=True)\n"
               "    time.sleep(0.001)\n")
    lines = []

    def slow_sink(line):
        # Slower than the child prints, so the line queue never empties:
        # the queue.Empty branch (the old deadline check) is never reached.
        lines.append(line)
        time.sleep(0.005)

    started = time.monotonic()
    code = update_mod.run_upgrade([sys.executable, "-c", chatter],
                                  on_line=slow_sink, timeout=0.5)
    elapsed = time.monotonic() - started
    assert code == 1 and any("timed out" in ln for ln in lines)
    assert "chatter" in lines                  # it really was streaming
    assert elapsed < 2.5                       # stopped near the deadline


def test_run_upgrade_cancel_kills_child_and_reports():
    lines = []
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    code = update_mod.run_upgrade(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        on_line=lines.append, timeout=30, cancel=cancel)
    assert code == 1 and any("cancelled" in ln for ln in lines)
    assert time.monotonic() - started < 10


# --- GUI: data layer and routes ---------------------------------------------------------

def test_gui_state_carries_update_fields(gui):  # noqa: F811
    state = gui.get("/api/state")[1]
    assert state["update"] is None
    assert state["update_run"]["running"] is False
    assert state["check_for_updates"] is False      # fixture keeps it offline


def test_gui_update_dismissal_is_persisted_across_reloads():
    """R2.3: the dismissed version survives a reload; only a strictly newer
    latest release re-shows the banner (same pattern as the other app.js
    localStorage guards)."""
    js = (Path(__file__).resolve().parents[1] / "pedacito" / "gui"
          / "static" / "app.js").read_text(encoding="utf-8")
    assert "pedacito_update_dismissed" in js
    assert "localStorage.getItem(UPDATE_DISMISSED_KEY)" in js
    assert "localStorage.setItem(UPDATE_DISMISSED_KEY, version)" in js
    assert "updateDismissed: savedUpdateDismissed" in js
    assert "writeUpdateDismissed(info.latest)" in js
    assert "isNewerVersion(info.latest, dismissed)" in js


def test_gui_routes_stand_down_when_disabled(gui, monkeypatch):  # noqa: F811
    no_network(monkeypatch)

    def no_proc(*a, **k):
        raise AssertionError("subprocess while disabled")

    monkeypatch.setattr(update_mod, "run_upgrade", no_proc)
    status, j = gui.post("/api/update/check", {})
    assert status == 200 and j["update"]["detail"] == "disabled"
    assert j["update"]["update_available"] is False
    status, j = gui.post("/api/update/run", {})
    assert status == 200 and j["ok"] is False and j["detail"]
    state = gui.get("/api/state")[1]
    assert state["check_for_updates"] is False
    assert state["update"]["update_available"] is False


def test_gui_manual_check_populates_banner_data(gui, monkeypatch):  # noqa: F811
    enable(monkeypatch)
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append((url, timeout))
        return FakeResponse(pypi_payload())

    monkeypatch.setattr(update_mod, "urlopen", fake_urlopen)
    status, j = gui.post("/api/update/check", {})
    assert status == 200
    assert j["update"]["update_available"] is True
    assert j["update"]["latest"] == "9.9.9"
    assert calls == [(update_mod.PYPI_URL, 2.0)]
    state = gui.get("/api/state")[1]
    assert state["update"]["latest"] == "9.9.9"
    assert state["check_for_updates"] is True
    status, j = gui.get("/api/update")
    assert status == 200 and j["update"]["latest"] == "9.9.9"


def test_gui_check_offline_keeps_banner_empty(gui, monkeypatch):  # noqa: F811
    enable(monkeypatch)

    def raise_exc(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(update_mod, "urlopen", raise_exc)
    status, j = gui.post("/api/update/check", {})
    assert status == 200 and j["update"]["detail"] == "offline"
    assert gui.get("/api/state")[1]["update"]["update_available"] is False


def test_gui_run_editable_returns_instructions(gui, monkeypatch):  # noqa: F811
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "detect_install_method",
                        lambda **kw: "editable")

    def no_proc(*a, **k):
        raise AssertionError("upgrade subprocess for an editable install")

    monkeypatch.setattr(update_mod, "run_upgrade", no_proc)
    status, j = gui.post("/api/update/run", {})
    assert status == 200 and j["ok"] is False
    assert "git pull" in j["detail"]
    assert gui.get("/api/update")[1]["update_run"]["running"] is False


def test_gui_run_streams_output_and_finishes(gui, monkeypatch):  # noqa: F811
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "detect_install_method",
                        lambda **kw: "venv")

    def fake_run(argv, on_line=None, timeout=None, cancel=None):
        assert isinstance(argv, list) and argv[0] == sys.executable
        assert timeout == gui_server.UPDATE_RUN_TIMEOUT
        assert cancel is not None
        on_line("Collecting pedacito")
        on_line("Successfully installed pedacito-9.9.9")
        return 0

    monkeypatch.setattr(update_mod, "run_upgrade", fake_run)
    status, j = gui.post("/api/update/run", {})
    assert status == 202 and j["ok"] is True
    assert j["command"] == [sys.executable, "-m", "pip", "install",
                            "--upgrade", "pedacito"]
    run = wait_until(lambda: (lambda r: r if not r["running"] else None)(
        gui.get("/api/update")[1]["update_run"]))
    assert run is not None, "upgrade worker did not finish"
    assert run["ok"] is True and run["code"] == 0
    assert "Successfully installed pedacito-9.9.9" in run["output"]
    # (Restart itself re-execs the process, so tests only exercise its guard.)


def test_gui_run_busy_is_409_and_restart_needs_success(gui, monkeypatch):  # noqa: F811
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "detect_install_method",
                        lambda **kw: "venv")
    # No successful run yet: restart refuses.
    status, _j = gui.post("/api/update/restart", {})
    assert status == 409
    release = threading.Event()

    def blocking_run(argv, on_line=None, timeout=None, cancel=None):
        on_line("working")
        release.wait(10)
        return 1

    monkeypatch.setattr(update_mod, "run_upgrade", blocking_run)
    status, _j = gui.post("/api/update/run", {})
    assert status == 202
    status, j = gui.post("/api/update/run", {})
    assert status == 409 and j["ok"] is False
    release.set()
    run = wait_until(lambda: (lambda r: r if not r["running"] else None)(
        gui.get("/api/update")[1]["update_run"]))
    assert run is not None and run["ok"] is False and run["code"] == 1
    status, _j = gui.post("/api/update/restart", {})
    assert status == 409                            # failed run: no restart


def test_gui_run_timeout_reports_failure_and_manual_instructions(gui, monkeypatch):  # noqa: F811
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "detect_install_method",
                        lambda **kw: "venv")
    manual = "Manual: python -m pip install --upgrade pedacito"
    monkeypatch.setattr(update_mod, "upgrade_command",
                        lambda method: ([sys.executable, "-c",
                                         "import time; time.sleep(30)"], manual))
    monkeypatch.setattr(gui_server, "UPDATE_RUN_TIMEOUT", 0.3)
    status, j = gui.post("/api/update/run", {})
    assert status == 202 and j["ok"] is True
    run = wait_until(lambda: (lambda r: r if not r["running"] else None)(
        gui.get("/api/update")[1]["update_run"]))
    assert run is not None, "timed-out upgrade worker did not finish"
    assert run["ok"] is False and run["code"] == 1
    assert "timed out" in run["detail"].lower()
    assert manual in run["detail"]
    status, _j = gui.post("/api/update/restart", {})
    assert status == 409                    # a timeout is not exit 0


def test_gui_cancel_stops_run_and_reports_failure(gui, monkeypatch):  # noqa: F811
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "detect_install_method",
                        lambda **kw: "venv")
    manual = "Manual: pipx upgrade pedacito"
    monkeypatch.setattr(update_mod, "upgrade_command",
                        lambda method: ([sys.executable, "-c",
                                         "import time; time.sleep(30)"], manual))
    status, j = gui.post("/api/update/run", {})
    assert status == 202 and j["ok"] is True
    assert wait_until(
        lambda: gui.get("/api/update")[1]["update_run"]["running"])
    status, j = gui.post("/api/update/cancel", {})
    assert status == 200 and j["ok"] is True and j["cancelling"] is True
    run = wait_until(lambda: (lambda r: r if not r["running"] else None)(
        gui.get("/api/update")[1]["update_run"]))
    assert run is not None, "cancelled upgrade worker did not finish"
    assert run["ok"] is False and run["code"] == 1
    assert "cancelled" in run["detail"].lower()
    assert manual in run["detail"]
    assert run["cancelling"] is False
    status, _j = gui.post("/api/update/restart", {})
    assert status == 409                    # a cancel is not exit 0


def test_gui_cancel_when_idle_is_409(gui):  # noqa: F811
    status, j = gui.post("/api/update/cancel", {})
    assert status == 409 and j["ok"] is False


def test_restart_on_windows_spawns_replacement_and_exits(monkeypatch):
    """R4.4: os.execv does not replace the process on Windows, so the restart
    path must Popen the same argv and hard-exit instead. No real exec."""
    calls = {"popen": [], "exit": None, "execv": False}
    monkeypatch.setattr(gui_server.os, "name", "nt")
    monkeypatch.setattr(gui_server.time, "sleep", lambda _s: None)
    monkeypatch.setattr(gui_server, "shutdown_server", lambda: None)
    monkeypatch.setattr(gui_server, "_RUN_ARGS", SimpleNamespace(
        port=8765, open_browser=False, project="/tmp/proj", profile="small"))
    monkeypatch.setattr(gui_server.subprocess, "Popen",
                        lambda argv, **kw: calls["popen"].append(argv))

    def fake_exit(code):
        calls["exit"] = code

    def no_exec(*_a, **_k):
        calls["execv"] = True

    monkeypatch.setattr(gui_server.os, "_exit", fake_exit)
    monkeypatch.setattr(gui_server.os, "execv", no_exec)
    gui_server._restart_process(gui_server._restart_argv())
    assert calls["popen"] == [[sys.executable, "-m", "pedacito", "gui",
                               "--port", "8765", "--no-open", "/tmp/proj",
                               "--profile", "small"]]
    assert calls["exit"] == 0 and calls["execv"] is False


def test_gui_manual_check_opt_out_never_touches_network(gui, monkeypatch):  # noqa: F811
    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise AssertionError("network call with check_for_updates=false")

    monkeypatch.setattr(update_mod, "urlopen", boom)
    status, j = gui.post("/api/update/check", {})
    assert status == 200 and j["update"]["detail"] == "disabled"
    assert j["update"]["update_available"] is False
    assert calls == []


def test_gui_settings_wires_manual_update_check():
    """R5.2/R4.5: the Settings tab calls both ready-made update endpoints,
    reports cache vs fresh, and the drawer offers Cancel while running."""
    static = Path(__file__).resolve().parents[1] / "pedacito" / "gui" / "static"
    js = (static / "settings.js").read_text(encoding="utf-8")
    assert "/api/update/check" in js and "Check for updates" in js
    assert "24 h cache" in js and "fresh check" in js
    assert "check_for_updates = false" in js
    app = (static / "app.js").read_text(encoding="utf-8")
    assert "/api/update/cancel" in app and "Stop" in app


def test_update_output_buffer_is_capped():
    st = core.State()
    try:
        st.update_run_start([sys.executable, "-m", "pip"])
        st.note_update_output("a" * 60_000)
        st.note_update_output("b" * 60_000)
        out = st.update_run["output"]
        assert len(out) <= core.MAX_UPDATE_OUTPUT
        assert out.endswith("b" * 100 + "\n")       # newest tail kept
        st.update_run_done(0)
        assert st.update_run["ok"] is True and st.update_run["running"] is False
        st.note_update({"update_available": True, "latest": "9.9.9"})
        assert st.status()["update"]["latest"] == "9.9.9"
        st.note_update(None)
        assert st.status()["update"] is None
    finally:
        st.bus.close()


@pytest.mark.network
def test_open_project_with_opt_out_clears_the_banner(project, monkeypatch):  # noqa: F811
    # Drop the fixture's env guard so the PROJECT config is what decides
    # (env would win over it, matching config precedence).
    monkeypatch.delenv("PEDACITO_CHECK_FOR_UPDATES", raising=False)
    core.STATE.note_update({"update_available": True, "latest": "9.9.9",
                            "current": __version__, "ok": True})
    assert project.get("/api/state")[1]["update"] is not None
    (project.root / ".pedacito.toml").write_text(
        "[universal]\ncheck_for_updates = false\n", encoding="utf-8")
    status, _ = project.post("/api/project",
                             {"path": str(project.root), "profile": ""})
    assert status == 202 and project.wait_idle(30)
    state = project.get("/api/state")[1]
    assert state["update"] is None
    assert state["check_for_updates"] is False


# --- TUI ---------------------------------------------------------------------------------

def _tui(tmp_path, monkeypatch, startup_check=False):
    from pedacito.tui import TuiSession
    if not startup_check:
        # Keep the startup hook out of tests that are about /update itself:
        # a stray daemon thread must never outlive its monkeypatched fetch.
        monkeypatch.setattr(update_mod, "check_for_update_async",
                            lambda *a, **k: None)
    cfg = Config()
    cfg.model = "fake-model"
    cfg.active_profile = "default"
    return TuiSession(tmp_path, cfg, client=FakeClient(), headless=True,
                      stdout=io.StringIO())


def test_tui_update_disabled_is_offline(tmp_path, monkeypatch):
    no_network(monkeypatch)
    session = _tui(tmp_path, monkeypatch, startup_check=True)
    try:
        assert session.update_info is None          # startup hook stood down
        assert session._update_thread is None
        session.handle_line("/update")
        text = transcript_text(session)
        assert "disabled" in text and "check_for_updates" in text
    finally:
        session.close()


def test_tui_update_run_upgrades(tmp_path, monkeypatch):
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(pypi_payload()))
    monkeypatch.setattr(update_mod, "detect_install_method",
                        lambda **kw: "venv")
    ran = []

    def fake_run(argv, on_line=None, timeout=None, cancel=None):
        ran.append(argv)
        on_line("Successfully installed pedacito-9.9.9")
        return 0

    monkeypatch.setattr(update_mod, "run_upgrade", fake_run)
    session = _tui(tmp_path, monkeypatch)
    try:
        session.handle_line("/update run")
        text = transcript_text(session)
        assert f"Installed: {__version__}" in text
        assert "Latest: 9.9.9" in text
        assert "Install method: venv" in text
        assert "Upgrade finished (exit 0)" in text
        assert ran and isinstance(ran[0], list)
    finally:
        session.close()


def test_tui_update_editable_shows_git_instructions(tmp_path, monkeypatch):
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(pypi_payload()))
    monkeypatch.setattr(update_mod, "detect_install_method",
                        lambda **kw: "editable")

    def no_proc(*a, **k):
        raise AssertionError("upgrade subprocess for an editable install")

    monkeypatch.setattr(update_mod, "run_upgrade", no_proc)
    session = _tui(tmp_path, monkeypatch)
    try:
        session.handle_line("/update")              # no confirmation needed
        text = transcript_text(session)
        assert "git pull" in text and "editable" in text
    finally:
        session.close()


def test_tui_update_up_to_date_and_offline(tmp_path, monkeypatch):
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(pypi_payload("0.0.1")))
    session = _tui(tmp_path, monkeypatch)
    try:
        session.handle_line("/update")
        assert "up to date" in transcript_text(session)
    finally:
        session.close()

    def raise_exc(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(update_mod, "urlopen", raise_exc)
    session = _tui(tmp_path, monkeypatch)
    try:
        session.handle_line("/update")
        assert "Could not reach PyPI" in transcript_text(session)
    finally:
        session.close()


def test_tui_startup_check_notes_a_new_release(tmp_path, monkeypatch):
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(pypi_payload()))
    session = _tui(tmp_path, monkeypatch, startup_check=True)
    try:
        assert session._update_thread is not None
        info = wait_until(lambda: session.update_info)
        assert info is not None and info["update_available"] is True
        assert "9.9.9 is available" in transcript_text(session)
        assert "/update" in transcript_text(session)
    finally:
        if session._update_thread is not None:
            session._update_thread.join(5)
        session.close()


def test_tui_help_lists_update(tmp_path, monkeypatch):
    session = _tui(tmp_path, monkeypatch)
    try:
        session.handle_line("/help")
        assert "/update [run]" in transcript_text(session)
    finally:
        session.close()


# --- CLI ---------------------------------------------------------------------------------

def test_cli_parser_accepts_update():
    args = cli.build_parser().parse_args(["update", "--run"])
    assert args.cmd == "update" and args.run is True
    args = cli.build_parser().parse_args(["update"])
    assert args.run is False


def test_cli_update_disabled_exits_0_without_network(monkeypatch, capsys):
    no_network(monkeypatch)
    assert cli._main(["update"]) == 0
    out = capsys.readouterr().out
    assert "disabled" in out and "check_for_updates" in out


def test_cli_update_offline_exits_1(monkeypatch, capsys):
    enable(monkeypatch)

    def raise_exc(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(update_mod, "urlopen", raise_exc)
    assert cli._main(["update"]) == 1
    assert "Could not reach PyPI" in capsys.readouterr().err


def test_cli_update_up_to_date(monkeypatch, capsys):
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(pypi_payload("0.0.1")))
    assert cli._main(["update"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_cli_update_newer_without_run(monkeypatch, capsys):
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(pypi_payload()))
    assert cli._main(["update"]) == 0
    out = capsys.readouterr().out
    assert "Latest: 9.9.9" in out and "update --run" in out


def test_cli_update_run_returns_upgrade_exit_code(monkeypatch, capsys):
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(pypi_payload()))
    monkeypatch.setattr(update_mod, "detect_install_method",
                        lambda **kw: "venv")
    codes = iter([0, 3])
    ran = []

    def fake_run(argv, on_line=None, timeout=None, cancel=None):
        ran.append(argv)
        on_line("pip output line")
        return next(codes)

    monkeypatch.setattr(update_mod, "run_upgrade", fake_run)
    assert cli._main(["update", "--run"]) == 0
    assert "Upgrade finished" in capsys.readouterr().out
    assert cli._main(["update", "--run"]) == 3
    assert "Upgrade failed (exit 3)" in capsys.readouterr().err
    assert all(isinstance(argv, list) for argv in ran)


def test_cli_update_run_editable_never_spawns(monkeypatch, capsys):
    enable(monkeypatch)
    monkeypatch.setattr(update_mod, "urlopen",
                        lambda *a, **k: FakeResponse(pypi_payload()))
    monkeypatch.setattr(update_mod, "detect_install_method",
                        lambda **kw: "editable")

    def no_proc(*a, **k):
        raise AssertionError("upgrade subprocess for an editable install")

    monkeypatch.setattr(update_mod, "run_upgrade", no_proc)
    assert cli._main(["update", "--run"]) == 0
    assert "git pull" in capsys.readouterr().out
