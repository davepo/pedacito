"""End-to-end tests against the fake OpenAI-compatible server.

These exercise the full agent pipeline -- gather loop (JSON-schema actions),
answer phase (plain text), edit application, and the review staging path --
without any real model or GPU. They open sockets on 127.0.0.1, hence the
`network` marker; CI runs them too because the fake server is local.

Run directly:  python -m pytest tests/test_e2e_fake.py -q -p no:libtmux
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import index as index_mod          # noqa: E402
from pedacito.agent import Agent                 # noqa: E402
from pedacito.config import Config               # noqa: E402
from pedacito.llm import LMStudio                # noqa: E402
from pedacito.review import diffs_from_results   # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_llm.py"

RENAME_REPLY = (
    'Renaming the file.\n\n'
    '--- FILE: new_name.py\n'
    '<<<<<<< SEARCH\n'
    '=======\n'
    'print("hi")\n'
    '>>>>>>> REPLACE\n\n'
    '--- FILE: old_name.py\n'
    '<<<<<<< SEARCH\n'
    'print("hi")\n'
    '=======\n'
    '>>>>>>> REPLACE\n'
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# NOTE: the `network` marker lives only on the test functions below, never
# on this fixture -- pytest >=8.4 hard-errors on marks applied to fixtures.
@pytest.fixture(scope="module")
def fake_port():
    """Start tests/fake_llm.py once for the module.

    The FAKE_ACTIONS list is deliberately NOT sent here: with no actions
    configured, the fake server answers the first schema request with its
    built-in default -- read old_name.py lines 1-200 (the file every test's
    fixture project creates) -- then done, which lets each test drive the
    gather loop against whatever files it created."""
    port = _free_port()
    env = dict(os.environ, FAKE_PORT=str(port), FAKE_REPLY=RENAME_REPLY)
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
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def _cfg(port: int) -> Config:
    cfg = Config()
    cfg.base_url = f"http://127.0.0.1:{port}/v1"
    cfg.model = "fake-model"
    cfg.max_steps = 6
    return cfg


def _project(tmp_path: Path) -> Path:
    (tmp_path / "old_name.py").write_text('print("hi")', encoding="utf-8")
    return tmp_path


@pytest.mark.network
def test_agent_rename_end_to_end(tmp_path, fake_port):
    """The user-reported bug in one shot: a simple rename request must walk
    gather -> answer -> apply and leave a reviewable diff, not 'nothing to
    review'."""
    root = _project(tmp_path)
    cfg = _cfg(fake_port)
    idx = index_mod.build(["."], cfg, client=None, summarise=False, root=root)
    agent = Agent(idx, LMStudio(cfg), cfg)

    gathered = agent.gather("rename old_name.py to new_name.py", verbose=False)
    assert "print" in gathered, "gather loop never surfaced the file's source"

    reply = agent.answer("rename old_name.py to new_name.py", gathered,
                         stream=False)
    assert "new_name.py" in reply

    results, _ = agent.apply(reply, dry_run=True)
    ok = [r for r in results if r.ok]
    assert ok, [r.message for r in results]
    ren = next(r for r in ok if r.file == "new_name.py")
    assert ren.renames_from == ("old_name.py",), \
        "rename pairing lost end to end -- GUI would show nothing to review"

    diffs = diffs_from_results(results, root=root)
    assert diffs, "review page would say 'nothing to review'"

    # Now apply for real through the same path the GUI/CLI use.
    results, _ = agent.apply(reply, dry_run=False)
    assert all(r.ok for r in results), [r.message for r in results]
    assert (root / "new_name.py").is_file()
    assert not (root / "old_name.py").exists()
    assert 'print("hi")' in (root / "new_name.py").read_text()


@pytest.mark.network
def test_gather_coverage_ledger_counts_reads(tmp_path, fake_port):
    """A `read` of './old_name.py'-style paths must register as coverage so
    the done-nudge stops firing forever (the ledger key mismatch bug)."""
    root = _project(tmp_path)
    cfg = _cfg(fake_port)
    idx = index_mod.build(["."], cfg, client=None, summarise=False, root=root)
    agent = Agent(idx, LMStudio(cfg), cfg)
    agent.gather("rename ./old_name.py to new_name.py", verbose=False)
    # The task names both files; the read covered old_name.py only, but the
    # point is the ledger normalised its key rather than storing "./...".
    keys = set(agent._read_spans)
    assert "old_name.py" in keys, f"ledger keys not normalised: {keys}"
