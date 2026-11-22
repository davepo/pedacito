"""Regression: --role worker=PROFILE must select the default worker client.

The worker role is special: it is the client every task call uses by default,
not a side client like planner or verifier. Parsing it into cfg.roles was not
enough; the LLMClient was built from the active profile before the role was
consulted, so the flag looked accepted and did nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import cli


def test_worker_role_replaces_the_default_worker_config(tmp_path):
    cfg_file = tmp_path / "user.toml"
    cfg_file.write_text(
        'vram_gb = 24\n'
        'model = "base-model"\n'
        'base_url = "http://127.0.0.1:9/v1"\n'
        '\n[profiles.small]\n'
        'base_url = "http://127.0.0.1:9/v1"\n'
        'model = "small-model"\n'
        'tier = "tiny"\n',
        encoding="utf-8")
    args = SimpleNamespace(
        cmd="ask", config=str(cfg_file), profile=None, root=str(tmp_path),
        paths=[str(tmp_path)], role=["worker=small"], no_refs=False,
        jsonl=False, rpc=False, auto_route=False, no_auto_route=False,
        creative=False)
    cfg = cli._config(args)
    assert cfg.model == "small-model"
    assert cfg.active_profile == "small"
    assert cfg.roles["worker"] == "small"
