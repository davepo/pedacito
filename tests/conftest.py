"""Shared test isolation.

Every test must be hermetic: it must never read the developer's real
`~/.config/pedacito/config.toml` (which can carry a `default_profile`, VRAM,
cache quantization and other settings that would change the result) and must
never honour ambient `PEDACITO_*` environment variables. This autouse fixture
points `USER_CONFIG` at a per-test path that does not exist and drops any
`PEDACITO_*` variables, so the whole suite is deterministic regardless of the
machine it runs on.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod


@pytest.fixture(autouse=True)
def _isolate_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "USER_CONFIG",
                        tmp_path / "no-user-config.toml")
    for key in [k for k in os.environ if k.startswith("PEDACITO_")]:
        monkeypatch.delenv(key, raising=False)
