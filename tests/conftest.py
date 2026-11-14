"""Shared test isolation.

Every test must be hermetic: it must never read the developer's real
`~/.config/pedacito/config.toml` (which can carry a `default_profile`, VRAM,
cache quantization and other settings that would change the result) and must
never honour ambient `PEDACITO_*` environment variables. This autouse fixture
points `USER_CONFIG` at a per-test path that does not exist, points the
per-user state folder (`USER_STATE_DIR`, where project-less chats and session
logs live) at a per-test path, drops any `PEDACITO_*` variables, and redirects
the environment `_user_config_dir()` prefers over `$HOME`
(`$XDG_CONFIG_HOME` on POSIX, `%APPDATA%` on Windows), so the whole suite is
deterministic regardless of the machine it runs on. All of those paths live
BESIDE `tmp_path`, never inside it: tests assert that a dry run or a fresh
scaffold left `tmp_path` empty, and isolation litter would read as the code
under test writing files.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod


@pytest.fixture(autouse=True)
def _isolate_environment(tmp_path, tmp_path_factory, monkeypatch):
    # Everything redirected here must stay OUT of tmp_path: that directory is
    # the test's own project/workspace, and assertions like "a dry run wrote
    # nothing" or a folder listing count every entry there.
    iso = tmp_path_factory.mktemp("isolation")
    monkeypatch.setattr(config_mod, "USER_CONFIG", iso / "config.toml")
    # Project-less chat threads and session logs go to the per-user state
    # folder: keep every test away from the developer's real conversations.
    monkeypatch.setattr(config_mod, "USER_STATE_DIR", iso / "state" / "pedacito")
    for key in [k for k in os.environ if k.startswith("PEDACITO_")]:
        monkeypatch.delenv(key, raising=False)
    # `_user_config_dir()` prefers $XDG_CONFIG_HOME (or %APPDATA% on Windows)
    # over $HOME. CI images set it to the runner's real profile, so a test
    # that points HOME at a tmp dir would still touch the real user config.
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    if os.name == "nt":
        appdata = iso / "appdata"
        (appdata / "pedacito").mkdir(parents=True)
        monkeypatch.setenv("APPDATA", str(appdata))
        localappdata = iso / "localappdata"
        (localappdata / "pedacito").mkdir(parents=True)
        monkeypatch.setenv("LOCALAPPDATA", str(localappdata))
