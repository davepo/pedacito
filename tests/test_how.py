"""Tests for `pedacito how` (built-in help from README, QUICKSTART, code)."""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import config as config_mod
from pedacito import how
from pedacito.cli import main


FAKE = Path(__file__).resolve().parent / "fake_llm.py"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_corpus_contains_docs_templates_and_source():
    chunks = how.build_corpus()
    files = {c.file for c in chunks}
    assert "README.md" in files
    assert "QUICKSTART.md" in files
    assert "pedacito/cli.py" in files
    assert "pedacito/config.py" in files
    # The starter config template is a module-level string, not a function;
    # it is extracted specifically so config answers can quote it.
    assert any(c.file == "pedacito/config.py" and c.qualname == "TEMPLATE"
               for c in chunks)
    # Every excerpt has real text and a usable line range.
    for c in chunks:
        assert c.source.strip()
        assert c.start >= 1 and c.end >= c.start


def test_retrieval_grounds_config_answer_in_docs_and_code():
    chunks = how.build_corpus()
    text, picked = how.select_references(
        "how do I set up a config file with vram?", chunks, config_mod.Config())
    files = {c.file for c in picked}
    assert "README.md" in files
    assert "QUICKSTART.md" in files
    assert any(f.startswith("pedacito/") for f in files)
    assert "vram_gb" in text
    assert "pedacito config" in text


def test_retrieval_finds_command_answers():
    chunks = how.build_corpus()
    text, picked = how.select_references(
        "what command do I run to index a project?", chunks,
        config_mod.Config())
    assert any(c.file == "QUICKSTART.md" for c in picked)
    assert "pedacito index" in text
    assert "pedacito files" in text or "pedacito ask" in text


def test_config_snapshot_hides_api_key(tmp_path):
    cfg = config_mod.Config()
    cfg.api_key = "sk-do-not-leak"
    cfg.sources["api_key"] = "user config"
    text = how.config_snapshot(cfg, tmp_path)
    assert "sk-do-not-leak" not in text
    assert "api_key" in text and "(set)" in text


def test_direct_flags_override_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "missing.toml")
    for key in list(os.environ):
        if key.startswith("PEDACITO_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(how, "auto_fit", lambda cfg, verbose=True: [])

    args = argparse.Namespace(config=None, profile=None,
                              url="http://example.test/v1",
                              api_key="secret", model="my-model",
                              vram_gb=12, question=[], question_opt=None)
    cfg = how.config_from_args(args)
    assert cfg.base_url == "http://example.test/v1"
    assert cfg.api_key == "secret"
    assert cfg.model == "my-model"
    assert cfg.vram_gb == 12
    assert cfg.sources["vram_gb"] == "--vram"


def test_how_does_not_require_vram_for_local_help(monkeypatch, tmp_path,
                                                 capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "missing.toml")
    for key in list(os.environ):
        if key.startswith("PEDACITO_"):
            monkeypatch.delenv(key, raising=False)
    args = argparse.Namespace(config=None, profile=None,
                              url="http://localhost:1234/v1",
                              api_key=None, model="m", vram_gb=None,
                              question=[], question_opt=None)
    cfg = how.config_from_args(args)
    assert cfg.vram_gb == 0
    assert "vram_gb is not set" in capsys.readouterr().err


@pytest.mark.network
def test_how_cli_runs_with_minimal_config_and_direct_flags(
        tmp_path, monkeypatch, capsys):
    port = _free_port()
    reply = "Run `pedacito config --init` to write a starter config."
    env = dict(os.environ, FAKE_PORT=str(port), FAKE_REPLY=reply)
    env.pop("FAKE_ACTIONS", None)
    proc = subprocess.Popen([sys.executable, str(FAKE)], env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), 0.5):
                    break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail("fake_llm exited early")
                time.sleep(0.1)
        else:
            pytest.fail("fake_llm did not start")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "none.toml")
        for key in list(os.environ):
            if key.startswith("PEDACITO_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(how, "auto_fit", lambda cfg, verbose=True: [])

        # A bare-minimum config file: server, model, and vram only.
        cfgfile = tmp_path / "min.toml"
        cfgfile.write_text(
            f'base_url = "http://127.0.0.1:{port}/v1"\n'
            'model = "fake-model"\n'
            'vram_gb = 8\n', encoding="utf-8")
        code = main(["how", "how do I set up a config file?",
                     "--config", str(cfgfile)])
        assert code == 0
        out = capsys.readouterr().out
        assert "pedacito config --init" in out

        # And with no config at all, everything passed directly.
        monkeypatch.setattr(config_mod, "USER_CONFIG",
                            tmp_path / "still-missing.toml")
        code = main(["how", "--url", f"http://127.0.0.1:{port}/v1",
                     "--api-key", "test-key", "--model", "fake-model",
                     "--vram", "8", "what command indexes a project?"])
        assert code == 0
        out = capsys.readouterr().out
        assert "pedacito config --init" in out
        assert not (tmp_path / ".pedacito").exists(), \
            "help must not create a project index/workspace"
    finally:
        proc.terminate()
        proc.wait(timeout=5)
