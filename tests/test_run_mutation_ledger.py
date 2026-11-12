"""`run`-command mutations feed the task ledger (TECHNIQUES.md #47).

A worker that finishes the job with shell commands produces no answer-phase
edit blocks; the files its commands rewrote must still count as landed work
(verification, `applied`/`changed`, no wasted zero-edit recovery re-ask, no
false zero-edit escalation statistic). `tool_run` already computes
`index.stale()` to refresh the index -- the fix records that same result on
`cfg._task_mutations`, at zero added cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_context import Scripted, _agent  # noqa: E402

from pedacito import index as index_mod, loop, tools  # noqa: E402
from pedacito.config import Config  # noqa: E402


def _indexed(tmp_path, files):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    cfg = Config()
    cfg.allow_shell = True
    cfg.log_events = False
    idx = index_mod.build([str(tmp_path)], cfg, client=None, summarise=False,
                          root=tmp_path, verbose=False)
    return idx, cfg


def _py(code):
    return f'{sys.executable} -c "{code}"'


def test_run_records_rewritten_indexed_files(tmp_path):
    idx, cfg = _indexed(tmp_path, {"f.py": "x = 1\n"})
    out = tools.tool_run(idx, cfg, _py(
        "from pathlib import Path; Path('f.py').write_text('x = 2\\n')"))
    assert "EXIT: 0" in out
    assert cfg._task_mutations == ["f.py"]


def test_run_records_mutations_even_when_the_command_fails(tmp_path):
    idx, cfg = _indexed(tmp_path, {"f.py": "x = 1\n"})
    out = tools.tool_run(idx, cfg, _py(
        "from pathlib import Path; import sys; "
        "Path('f.py').write_text('x = 2\\n'); sys.exit(1)"))
    assert "FAILED" in out
    # the ledger tracks DISK state: a failed command that rewrote files
    # still needs post-edit verification
    assert cfg._task_mutations == ["f.py"]


def test_readonly_run_records_nothing(tmp_path):
    idx, cfg = _indexed(tmp_path, {"f.py": "x = 1\n"})
    out = tools.tool_run(idx, cfg, _py("print('hello')"))
    assert "EXIT: 0" in out
    assert not getattr(cfg, "_task_mutations", [])


def test_run_records_deleted_indexed_files(tmp_path):
    idx, cfg = _indexed(tmp_path, {"f.py": "x = 1\n", "g.py": "y = 1\n"})
    tools.tool_run(idx, cfg, _py(
        "from pathlib import Path; Path('f.py').unlink()"))
    assert cfg._task_mutations == ["f.py"]
    assert "f.py" not in idx.files          # the index forgot it too


def test_new_unindexed_files_are_not_recorded(tmp_path):
    # documented limitation: stale() sees indexed files only; a command that
    # creates a brand-new file is invisible to the ledger (model-driven
    # creation is covered by the write tools)
    idx, cfg = _indexed(tmp_path, {"f.py": "x = 1\n"})
    tools.tool_run(idx, cfg, _py(
        "from pathlib import Path; Path('brand_new.py').write_text('z = 3\\n')"))
    assert not getattr(cfg, "_task_mutations", [])


def test_ledger_is_duplicate_free_across_runs(tmp_path):
    idx, cfg = _indexed(tmp_path, {"f.py": "x = 1\n"})
    tools.tool_run(idx, cfg, _py(
        "from pathlib import Path; Path('f.py').write_text('x = 2\\n')"))
    tools.tool_run(idx, cfg, _py(
        "from pathlib import Path; Path('f.py').write_text('x = 3\\n')"))
    assert cfg._task_mutations == ["f.py"]


def test_shell_only_task_counts_as_landed_work(tmp_path):
    """End to end: gather finishes the job with one `run`, the answer has no
    edit blocks -- the task must still be applied/changed, and the zero-edit
    recovery re-ask must NOT fire."""
    client = Scripted(
        steps=[{"action": "read", "file": "f.py", "start": 1, "end": 1},
               {"action": "run",
                "command": _py("from pathlib import Path; "
                               "Path('f.py').write_text('x = 2\\n')")},
               {"action": "done"}],
        replies=["I fixed f.py by running the script."])
    agent = _agent(tmp_path, client, {"f.py": "x = 1\n"},
                   allow_shell=True, verify_tests=False,
                   post_edit_check=False, static_gate=False,
                   interface_lock=False, frontend_contracts=False,
                   verify_dom=False, use_memory=False, backup=False)
    tr = loop.run_task(tmp_path, agent.cfg, client, agent, agent.index,
                       "fix f.py by running the script")
    assert tr.applied is True
    assert tr.changed == ["f.py"]
    assert (tmp_path / "f.py").read_text() == "x = 2\n"
    # one answer call only: no zero-edit recovery re-ask was spent
    assert len(client.chat_calls) == 1