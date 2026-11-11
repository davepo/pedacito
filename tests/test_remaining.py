"""Second-round features: duplicate-result pointers, observation caps, worked
examples, import-aware callers, summary batching, plan mode, steering and the
question action, deterministic rename, the escalation ladder."""
from __future__ import annotations

import os
import queue
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_context import Scripted, _agent

from pedacito import index as index_mod, loop, plan, refactor, steering, tools
from pedacito.config import Config

SRC = {
    "pkg/__init__.py": "",
    "pkg/core.py": "def apply_discount(price):\n    return price * 0.9\n",
    "pkg/shop.py": ("from pkg.core import apply_discount\n\n\n"
                    "def checkout(p):\n    # apply_discount here\n"
                    "    note = 'apply_discount'\n    return apply_discount(p)\n"),
    "pkg/other.py": "def helper():\n    return 1\n",
}


def _step(**kw):
    return {"action": "grep", "pattern": "x", **kw}


def test_repeated_readonly_step_gets_a_pointer(tmp_path):
    client = Scripted(steps=[
        {"action": "grep", "pattern": "checkout"},
        {"action": "find", "pattern": "helper"},
        {"action": "grep", "pattern": "checkout"},
        {"action": "done"}])
    agent = _agent(tmp_path, client, SRC)
    agent.gather("look around", verbose=False)
    seen = "\n".join(m["content"] for m in client.json_calls[-1])
    assert "already ran at step 1" in seen


def test_observation_cap_trims_and_says_so(tmp_path):
    agent = _agent(tmp_path, Scripted(), SRC, max_observation_tokens=300)
    big = "\n".join(f"line {i} " + "x" * 40 for i in range(400))
    out = agent._cap_observation("grep", big)
    assert len(out) < len(big) // 4
    assert "output capped" in out and "more line(s)" in out
    assert agent._cap_observation("read", big) == big
    assert agent._cap_observation("grep", "ERROR: " + big) == "ERROR: " + big


def test_fewshot_examples_use_a_real_file(tmp_path):
    from pedacito import llm
    client = llm.LLMClient.__new__(llm.LLMClient)
    client.fewshot = []
    agent = _agent(tmp_path, client, SRC)
    agent._prime_fewshot()
    files = {s.get("file") for s in client.fewshot if s.get("file")}
    assert files and files <= set(agent.index.files)
    assert llm.schema_example_with(tools.schema_for(agent.cfg), **client.fewshot[0])


def test_callers_separate_confident_from_name_only(tmp_path):
    files = dict(SRC)
    files["pkg/other.py"] = ("def helper():\n    return apply_discount(2)\n\n"
                             "def apply_discount(x):\n    return x\n")
    agent = _agent(tmp_path, Scripted(), {k: v for k, v in files.items()
                                          if k != "pkg/other.py"})
    sure, approx = agent.index.callers_ranked("apply_discount", "pkg/core.py")
    assert [c.file for c in sure] == ["pkg/shop.py"]
    assert approx == []
    out = tools.tool_callers(agent.index, "apply_discount")
    assert "pkg/shop.py" in out


def test_parse_batch_tolerates_noise():
    got = index_mod._parse_batch("1: Adds one.\n2) Subtracts.\njunk\n4: Out of range", 3)
    assert got == ["Adds one.", "Subtracts.", ""]


def test_summary_batching_makes_fewer_requests(tmp_path):
    class Batch:
        def __init__(self):
            self.calls = 0

        def warmup(self):
            pass

        def chat(self, messages, max_tokens=0, **kw):
            self.calls += 1
            n = int(messages[-1]["content"].split("N = ")[1].split()[0]) \
                if "N = " in messages[-1]["content"] else 1
            return "\n".join(f"{k}: Does thing {k}." for k in range(1, n + 1)) \
                if n > 1 else "Does a thing."

    body = "".join(f"def f{i}(x):\n    a = x + {i}\n    b = a * 2\n    c = b - 1\n    return c\n\n\n" for i in range(8))
    (tmp_path / "m.py").write_text(body)
    cfg = Config()
    cfg.log_events = False
    cfg.summary_batch = 4
    cfg.summary_workers = 1
    client = Batch()
    idx = index_mod.build([str(tmp_path)], cfg, client=client, summarise=True,
                          root=tmp_path, verbose=False)
    summaries = [c.summary for c in idx.chunks if c.qualname.startswith("f")]
    assert len(summaries) == 8 and all(summaries)
    assert client.calls < 8


def test_requirement_split_and_dependency_order():
    titles = plan.requirement_titles(
        "- add retries to sync\n- log every retry\n- document it in the README")
    assert len(titles) == 3
    assert loop.order_by_dependencies(["a", "b", "c"], [[1], [], [0]]) == ["b", "a", "c"]
    assert loop.order_by_dependencies(["a", "b"], [[1], [0]]) == ["a", "b"]
    raw = {"tasks": [{"title": "a"}, {"title": "b", "depends_on": [1, 9, "x"]}]}
    assert loop.deps_from(raw, 2) == [[], [0]]
    assert plan.requirement_titles("fix the typo in utils.py") == []


def test_replace_unchecked_keeps_finished_steps(tmp_path):
    p = tmp_path / "plan.md"
    loop.write_plan(p, "goal", ["one", "two", "three"])
    loop.mark_done(p, "one")
    loop.mark_failed(p, "two")
    loop.replace_unchecked(p, ["three-a", "three-b"])
    text = p.read_text()
    assert "[x] one" in text and "[!] two" in text
    assert "[ ] three-a" in text and "[ ] three-b" in text and "three\n" not in text


def test_replan_asks_planner_once_and_drops_the_failed_step(tmp_path):
    p = tmp_path / "plan.md"
    loop.write_plan(p, "goal", ["one", "two"])
    loop.mark_failed(p, "one")
    client = Scripted(steps=[{"tasks": [{"title": "one"}, {"title": "investigate"},
                                        {"title": "two"}]}])
    got = plan.replan("goal", p, client, Config(), "one")
    assert got == ["investigate", "two"]
    assert len(client.json_calls) == 1
    assert "FAILED" in client.json_calls[0][-1]["content"]


def test_should_plan_respects_threshold(tmp_path):
    agent = _agent(tmp_path, Scripted(), SRC)
    task = "- change core\n- change shop\n- update docs"
    agent.cfg.plan_threshold = 0
    assert not plan.should_plan(task, agent.index, agent.cfg)
    agent.cfg.plan_threshold = 3
    assert plan.should_plan(task, agent.index, agent.cfg)
    assert not plan.should_plan("fix a typo", agent.index, agent.cfg)


def test_wide_task_plans_even_with_the_threshold_off(tmp_path):
    """P11: a task that names WIDE_TASK_FILES or more files gets planned even
    when plan_threshold is 0 -- the default, and what a strong model's card
    picks. One window cannot hold the lookups for that many files, and the
    deepclitest runs kept losing the tail of exactly those tasks. The card's
    0 is not a user opt-out (it measured skill on small probes, not window
    capacity); an explicit config plan_threshold still governs."""
    files = dict(SRC)
    files.update({f"pkg/extra{i}.py": f"def f{i}():\n    return {i}\n"
                  for i in range(4)})
    agent = _agent(tmp_path, Scripted(), files)
    wide = ("touch pkg/core.py, pkg/shop.py, pkg/other.py, pkg/extra0.py "
            "and pkg/extra1.py")
    agent.cfg.plan_threshold = 0
    agent.cfg.sources = {}                            # the default origin
    assert plan.should_plan(wide, agent.index, agent.cfg)
    assert not plan.should_plan("fix a typo in pkg/core.py",
                                agent.index, agent.cfg)
    # a model card's 0 does not silence the wide rule ...
    agent.cfg.sources = {"plan_threshold": "model card"}
    assert plan.should_plan(wide, agent.index, agent.cfg)
    # ... an explicit user setting does, whatever its value
    agent.cfg.sources = {"plan_threshold": "config file (/x/pedacito.toml)"}
    assert not plan.should_plan(wide, agent.index, agent.cfg)
    agent.cfg.plan_threshold = 2
    assert plan.should_plan("update pkg/core.py and pkg/shop.py",
                            agent.index, agent.cfg)


@pytest.mark.skipif(
    os.name == "nt", reason="Windows steering reads the console, not a pipe")
def test_steering_reads_lines_from_a_pipe():
    r, w = os.pipe()
    with steering.Steering(fd=r, poll=0.02) as s:
        os.write(w, b"look at sync.py\n\nsecond\n")
        got = []
        for _ in range(100):
            got += steering.drain(s.queue)
            if len(got) >= 2:
                break
            import time; time.sleep(0.02)
    os.close(r); os.close(w)
    assert got == ["look at sync.py", "second"]


def test_steering_note_reaches_the_next_step(tmp_path):
    q = queue.Queue()
    q.put("the bug is in shop.py")
    client = Scripted(steps=[{"action": "done"}])
    agent = _agent(tmp_path, client, SRC)
    agent.cfg._steer = q
    agent.gather("fix checkout", verbose=False)
    assert "the bug is in shop.py" in client.json_calls[0][-1]["content"]


def test_question_action_is_bounded_and_interactive_only(tmp_path):
    asked = []
    client = Scripted(steps=[{"action": "question", "content": "which file?"}] * 3
                      + [{"action": "done"}])
    agent = _agent(tmp_path, client, SRC)
    assert "question" not in tools.schema_for(agent.cfg)["properties"]["action"]["enum"]
    agent.cfg._interactive = True
    agent.cfg._ask_user = lambda t: asked.append(t) or "shop.py"
    assert "question" in tools.schema_for(agent.cfg)["properties"]["action"]["enum"]
    agent.gather("fix checkout", verbose=False)
    assert len(asked) == steering.MAX_QUESTIONS
    seen = "\n".join(m["content"] for m in client.json_calls[-1])
    assert "USER ANSWER: shop.py" in seen and "maximum number of" in seen


def _rename_cfg():
    cfg = Config()
    cfg.allow_write = True
    cfg.backup = False
    cfg.log_events = False
    return cfg


def test_rename_touches_definition_and_importers_only(tmp_path):
    agent = _agent(tmp_path, Scripted(), SRC, allow_write=True)
    out = refactor.tool_rename(agent.index, agent.cfg, "apply_discount", "discounted")
    assert out.startswith("RENAMED")
    core = (tmp_path / "pkg/core.py").read_text()
    shop = (tmp_path / "pkg/shop.py").read_text()
    assert "def discounted(" in core
    assert "import discounted" in shop and "return discounted(p)" in shop
    assert "# apply_discount here" in shop and "'apply_discount'" in shop


def test_rename_refuses_ambiguous_and_taken_names(tmp_path):
    files = dict(SRC)
    files["pkg/other.py"] = "def apply_discount(x):\n    return x\n"
    agent = _agent(tmp_path, Scripted(), files, allow_write=True)
    out = refactor.tool_rename(agent.index, agent.cfg, "apply_discount", "d")
    assert out.startswith("ERROR") and "defined in 2 places" in out
    assert (tmp_path / "pkg/core.py").read_text() == SRC["pkg/core.py"]
    agent2 = _agent(tmp_path / "b", Scripted(), SRC, allow_write=True)
    assert "already defined" in refactor.tool_rename(
        agent2.index, agent2.cfg, "apply_discount", "checkout")
    assert "valid identifier" in refactor.tool_rename(
        agent2.index, agent2.cfg, "apply_discount", "class")
    ro = _agent(tmp_path / "c", Scripted(), SRC, allow_write=False)
    assert "allow_write" in refactor.tool_rename(ro.index, ro.cfg, "apply_discount", "z")


def test_mutating_actions_fill_the_task_mutation_ledger(tmp_path):
    """O2: every gather-phase tool action that writes records the files it
    touched on cfg._task_mutations, so the task loop can verify a gather-only
    run's work (hooks, impacted tests, lint) even when the answer phase
    produces no edit block at all."""
    agent = _agent(tmp_path, Scripted(), SRC, allow_write=True)
    cfg = agent.cfg
    out = refactor.tool_rename(agent.index, cfg, "apply_discount", "discounted")
    assert out.startswith("RENAMED")
    assert set(cfg._task_mutations) == {"pkg/core.py", "pkg/shop.py"}
    assert tools.tool_write(agent.index, cfg, "pkg/notes.md",
                            "# notes\n").startswith("CREATED")
    assert tools.tool_move(agent.index, cfg, "pkg/notes.md",
                           "pkg/NOTES.md").startswith("MOVED")
    assert tools.tool_delete(agent.index, cfg,
                             "pkg/NOTES.md").startswith("DELETED")
    # A move records both endpoints; touching a recorded path again does not
    # duplicate it (the ledger stays ordered and duplicate-free).
    assert cfg._task_mutations[-2:] == ["pkg/notes.md", "pkg/NOTES.md"]
    assert set(cfg._task_mutations) == {"pkg/core.py", "pkg/shop.py",
                                        "pkg/notes.md", "pkg/NOTES.md"}
    assert len(cfg._task_mutations) == 4


def test_undo_reverts_every_mutation_of_the_task(tmp_path):
    """P8: a gather-phase worker whose mutations went wrong gets its files
    back from the task's snapshot dir instead of rewriting them from memory
    (which is how botched renames ended in half-remembered files): the
    overwritten file returns to its pre-task bytes, created files go away,
    and a move un-happens at both endpoints."""
    from pedacito import workspace
    agent = _agent(tmp_path, Scripted(), SRC, allow_write=True, backup=True)
    cfg = agent.cfg
    cfg._backup_dir = workspace.new_backup_dir(tmp_path)
    assert tools.tool_write(agent.index, cfg, "pkg/core.py",
                            "BROKEN = True\n").startswith("WROTE")
    assert tools.tool_write(agent.index, cfg, "pkg/notes.md",
                            "# notes\n").startswith("CREATED")
    assert tools.tool_move(agent.index, cfg, "pkg/shop.py",
                           "pkg/store.py").startswith("MOVED")
    out = tools.tool_undo(agent.index, cfg, "")
    assert "UNDONE (pre-task contents restored)" in out
    assert "REMOVED (created by this task" in out
    assert (tmp_path / "pkg/core.py").read_text() == SRC["pkg/core.py"]
    assert (tmp_path / "pkg/shop.py").read_text() == SRC["pkg/shop.py"]
    assert not (tmp_path / "pkg/notes.md").exists()
    assert not (tmp_path / "pkg/store.py").exists()
    # the index sees the restored world again
    assert agent.index.resolve_file("shop.py") == "pkg/shop.py"
    assert agent.index.resolve_file("store.py") is None


def test_undo_one_file_leaves_the_other_mutations_alone(tmp_path):
    from pedacito import workspace
    agent = _agent(tmp_path, Scripted(), SRC, allow_write=True, backup=True)
    cfg = agent.cfg
    cfg._backup_dir = workspace.new_backup_dir(tmp_path)
    # a fresh task has nothing to undo
    assert "has not mutated any file yet" in tools.tool_undo(
        agent.index, cfg, "")
    tools.tool_write(agent.index, cfg, "pkg/core.py", "BROKEN = True\n")
    tools.tool_write(agent.index, cfg, "pkg/other.py", "BROKEN = True\n")
    out = tools.tool_undo(agent.index, cfg, "pkg/core.py")
    assert (tmp_path / "pkg/core.py").read_text() == SRC["pkg/core.py"]
    assert (tmp_path / "pkg/other.py").read_text() == "BROKEN = True\n"
    assert "pkg/core.py" in out and "pkg/other.py" not in out
    # a name the ledger does not know lists what the task actually touched
    out = tools.tool_undo(agent.index, cfg, "pkg/nope.py")
    assert "Nothing to undo for 'pkg/nope.py'" in out
    assert "pkg/other.py" in out
    # gates: read-only or backups-off runs cannot undo
    ro = _agent(tmp_path / "ro", Scripted(), SRC, allow_write=False,
                backup=True)
    assert "allow_write = false" in tools.tool_undo(ro.index, ro.cfg, "")
    nb = _agent(tmp_path / "nb", Scripted(), SRC, allow_write=True)
    assert "backup = false" in tools.tool_undo(nb.index, nb.cfg, "")


def test_undo_is_offered_and_dispatched_only_when_it_can_work(tmp_path):
    # the schema only lists `undo` when snapshots actually exist to undo
    cfg = Config()
    cfg.allow_write, cfg.backup = True, True
    assert "undo" in tools.schema_for(cfg)["properties"]["action"]["enum"]
    cfg.backup = False
    assert "undo" not in tools.schema_for(cfg)["properties"]["action"]["enum"]
    ro = Config()
    ro.allow_write, ro.backup = False, True
    assert "undo" not in tools.schema_for(ro)["properties"]["action"]["enum"]

    from pedacito import workspace
    agent = _agent(tmp_path, Scripted(), SRC, allow_write=True, backup=True)
    cfg = agent.cfg
    cfg._backup_dir = workspace.new_backup_dir(tmp_path)
    tools.dispatch(agent.index, cfg, {"action": "write", "file": "pkg/core.py",
                                      "content": "BROKEN = True\n"})
    out = tools.dispatch(agent.index, cfg, {"action": "undo", "file": ""})
    assert "UNDONE" in out
    assert (tmp_path / "pkg/core.py").read_text() == SRC["pkg/core.py"]
    # without write permission the generic mutation gate refuses undo too
    ro_agent = _agent(tmp_path / "ro", Scripted(), SRC, allow_write=False,
                      backup=True)
    out = tools.dispatch(ro_agent.index, ro_agent.cfg,
                         {"action": "undo", "file": ""})
    assert "`undo` is disabled" in out


def test_rename_text_generic_skips_strings_and_comments():
    text = 'const a = foo(1); // foo\nconst s = "foo"; foo2(foo);\n'
    out, n = refactor.rename_text("x.js", text, "foo", "bar")
    assert out == 'const a = bar(1); // foo\nconst s = "foo"; foo2(bar);\n' and n == 2


def test_run_task_escalates_once_when_nothing_landed(tmp_path, monkeypatch):
    calls = []

    class TR:
        def __init__(self, ok):
            self.results = [type("R", (), {"ok": ok})()] if ok else []
            self.applied = ok
            self.reply = "x"
            self.idx = None

    def fake_once(root, cfg, client, agent, idx, task, **kw):
        calls.append(client)
        return TR(client == "strong")

    from pedacito import roles
    monkeypatch.setattr(loop, "_run_task_once", fake_once)
    monkeypatch.setattr(roles, "client_for",
                        lambda cfg, role, c: "strong" if role == "escalate" else c)
    monkeypatch.setattr("pedacito.agent.Agent", lambda idx, c, cfg: object())
    tr = loop.run_task(tmp_path, Config(), "weak", object(), None, "t")
    assert calls == ["weak", "strong"] and tr.applied


def test_run_lock_blocks_a_live_holder_and_takes_over_a_stale_one(tmp_path):
    import subprocess
    from pedacito import workspace
    (tmp_path / ".pedacito").mkdir()
    lock = tmp_path / ".pedacito" / "run.lock"
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock.write_text(f"{sleeper.pid}\n")
        try:
            workspace.acquire_lock(tmp_path)
            raise AssertionError("expected ProjectBusy")
        except workspace.ProjectBusy as e:
            assert str(sleeper.pid) in str(e)
            assert sleeper.poll() is None       # probing must not kill the holder
    finally:
        sleeper.kill()
        sleeper.wait()
    workspace.acquire_lock(tmp_path)            # holder is dead: taken over
    assert lock.read_text().strip() == str(os.getpid())
    workspace.release_lock(tmp_path)
    assert not lock.exists()


def test_failed_attempts_warn_a_similar_task_and_clear_on_success(tmp_path):
    from pedacito import memory
    memory.note_attempt(tmp_path, "make checkout apply the discount", "no valid edit",
                        ["pkg/shop.py"])
    note = memory.attempts_note(tmp_path, "checkout should apply discount twice")
    assert "EARLIER FAILED" in note and "pkg/shop.py" in note
    assert memory.attempts_note(tmp_path, "rewrite the readme header") == ""
    memory.clear_attempts(tmp_path, "make checkout apply the discount")
    assert memory.attempts_note(tmp_path, "make checkout apply the discount") == ""


def test_mypy_detected_only_when_configured(tmp_path, monkeypatch):
    from pedacito import diagnostics
    monkeypatch.setattr(diagnostics, "_which",
                        lambda n: f"/usr/bin/{n}" if n in ("ruff", "mypy") else "")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    labels = [c.label for c in diagnostics.detect(tmp_path, ["a.py"])]
    assert labels == ["ruff"]
    (tmp_path / "pyproject.toml").write_text("[tool.mypy]\nstrict = true\n")
    labels = [c.label for c in diagnostics.detect(tmp_path, ["a.py"])]
    assert labels == ["ruff", "mypy"]


_LANG_SAMPLES = {
    ".php": ("<?php\nnamespace App;\nuse Foo\\Bar;\nfunction top($a) { return helper($a); }\n"
             "class A extends B {\n  public function m($x) { return $this->go($x); }\n}\n"),
    ".kt": ("package a\nimport foo.Bar\nfun top(a: Int): Int { return helper(a) }\n"
            "class A : B() {\n  fun m(x: Int): Int { return this.go(x) }\n}\n"),
    ".swift": ("import Foundation\nfunc top(a: Int) -> Int { return helper(a) }\n"
               "class A: B {\n  func m(x: Int) -> Int { return self.go(x) }\n}\n"),
    ".scala": ("package a\nimport foo.Bar\ndef top(a: Int): Int = helper(a)\n"
               "class A extends B {\n  def m(x: Int): Int = go(x)\n}\n"),
    ".lua": ("local M = {}\nfunction top(a) return helper(a) end\n"
             "function M.method(x) return obj:go(x) end\n"),
}


@pytest.mark.parametrize("ext", sorted(_LANG_SAMPLES))
def test_extra_tree_sitter_languages_chunk_with_calls(ext):
    from pedacito import chunker_ts, select
    if not chunker_ts.TS_AVAILABLE:
        pytest.skip("tree-sitter language pack not installed")
    assert ext in select.CODE_EXTS
    try:
        chunks = chunker_ts.chunk_file_ts("f" + ext, _LANG_SAMPLES[ext], ext)
    except Exception as e:                      # grammar missing in this build
        pytest.skip(f"grammar unavailable: {e}")
    by = {c.qualname: c for c in chunks}
    assert "top" in by and by["top"].calls == ["helper"]
    callers = [c for c in chunks if "go" in c.calls]
    assert callers, f"no chunk records the call to go() in {ext}"
    bad, _ = chunker_ts.has_syntax_error("function top( {", chunker_ts.LANG_BY_EXT[ext])
    assert isinstance(bad, bool)


def test_tree_sitter_is_pinned_below_the_026_segfault_regression():
    root = Path(__file__).resolve().parents[1]
    for name in ("pyproject.toml", "requirements.txt"):
        text = (root / name).read_text(encoding="utf-8")
        assert "tree-sitter>=0.23,<0.26" in text, name


def test_mcp_dead_server_cools_down_instead_of_retrying_every_call():
    from pedacito import mcp
    mcp.close_all()
    cmd = "definitely-not-a-real-mcp-server --stdio"
    for _ in range(2):
        try:
            mcp._live_session(cmd, 1)
        except Exception:
            pass
    assert mcp._cooling(cmd) > 0
    try:
        mcp._live_session(cmd, 1)
        raise AssertionError("expected a cool-down error")
    except mcp.MCPError as e:
        assert "retrying in" in str(e)
    mcp.close_all()
    assert mcp._cooling(cmd) == 0


@pytest.mark.skipif(
    os.name == "nt", reason="Seatbelt/bwrap profiles are POSIX-only")
def test_sandbox_profile_and_args_cover_the_guards(tmp_path, monkeypatch):
    from pedacito import sandbox
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pedacito").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home" / ".ssh").mkdir(parents=True)
    root = (tmp_path / "proj")
    root.mkdir()
    (root / ".git").mkdir()
    prof = sandbox.seatbelt_profile(root, False, [])
    assert "(deny network*)" in prof and "(deny file-write*)" in prof
    assert f'(deny file-write* (subpath "{root}/.git"))' in prof
    assert f'(deny file-read* (subpath "{tmp_path}/home/.ssh"))' in prof
    assert "(deny network*)" not in sandbox.seatbelt_profile(root, True, [])
    args = sandbox._bwrap_args(root, False, [])
    assert "--unshare-net" in args and args[-1] == "--"
    assert "--unshare-net" not in sandbox._bwrap_args(root, True, [])
    i = args.index("--ro-bind", args.index(str(root)))
    assert args[i + 1] == str(root / ".git")
    assert str(tmp_path / "home" / ".ssh") in args       # hidden by a tmpfs


def test_sandbox_degrades_with_one_notice_and_strict_refuses(tmp_path, monkeypatch, capsys):
    from pedacito import sandbox
    monkeypatch.setattr(sandbox, "backend_name", lambda: "")
    sandbox._noticed.clear()
    cfg = Config()
    argv, note = sandbox.wrap(["echo", "x"], cfg, tmp_path)
    sandbox.wrap(["echo", "x"], cfg, tmp_path)
    err = capsys.readouterr().err
    assert argv == ["echo", "x"] and note == "" and err.count("[sandbox]") == 1
    cfg.sandbox_strict = True
    try:
        sandbox.wrap(["echo"], cfg, tmp_path)
        raise AssertionError("expected SandboxRefused")
    except sandbox.SandboxRefused:
        pass
    cfg.sandbox = False
    assert sandbox.wrap(["echo"], cfg, tmp_path) == (["echo"], "")


def test_sandbox_wraps_when_active(tmp_path, monkeypatch):
    from pedacito import sandbox
    monkeypatch.setattr(sandbox, "backend_name", lambda: "bwrap")
    monkeypatch.setattr(sandbox, "_probe", lambda b, n: (True, ""))
    argv, note = sandbox.wrap(["pytest", "-q"], Config(), tmp_path)
    assert argv[0] == "bwrap" and argv[-2:] == ["pytest", "-q"]
    assert "network off" in note


def test_sandbox_settings_need_trust_in_a_project_config():
    from pedacito import config
    assert {"sandbox", "sandbox_network", "sandbox_writable",
            "sandbox_strict"} <= set(config.EXEC_FIELDS)


def test_sandbox_really_blocks_writes_when_a_backend_works(tmp_path):
    import subprocess as sp
    from pedacito import sandbox
    cfg = Config()
    if sandbox.status(cfg)[0] != "active":
        pytest.skip("no working sandbox backend on this machine")
    # The parent of tmp_path is allowed on macOS (/private/var/folders), so
    # aim at the home directory: both backends deny writes outside the root.
    outside = Path.home() / f".pedacito-sandbox-escape-{os.getpid()}.txt"
    try:
        argv, _ = sandbox.wrap([sys.executable, "-c",
                                f"open({str(outside)!r}, 'w').write('x'); "
                                "open('inside.txt', 'w').write('y')"], cfg, tmp_path)
        r = sp.run(argv, cwd=tmp_path, capture_output=True)
        assert not outside.exists() and r.returncode != 0
    finally:
        outside.unlink(missing_ok=True)


def test_sandbox_rebinds_an_interpreter_living_under_tmp(tmp_path, monkeypatch):
    from pedacito import sandbox
    prefix = f"/tmp/pdc-sbx-prefix-{os.getpid()}"
    monkeypatch.setattr(sys, "prefix", prefix)
    monkeypatch.setattr(sys, "base_prefix", prefix)
    args = sandbox._bwrap_args(tmp_path, False, [])
    i = args.index("--tmpfs")
    assert args[i + 1] == "/tmp"
    # the re-bind must come after the tmpfs, or the tmpfs hides it again
    j = args.index("--ro-bind", i + 2)
    assert args[j + 1] == args[j + 2] == prefix
    # an interpreter outside /tmp needs no re-bind
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    assert prefix not in sandbox._bwrap_args(tmp_path, False, [])


def test_sandbox_runs_an_interpreter_installed_under_tmp(tmp_path, monkeypatch):
    import shutil
    import subprocess as sp
    import venv
    from pedacito import sandbox
    cfg = Config()
    if sandbox.status(cfg)[0] != "active":
        pytest.skip("no working sandbox backend on this machine")
    if sandbox.backend_name() != "bwrap":
        pytest.skip("the /tmp tmpfs is bwrap-specific")
    venv_dir = Path("/tmp") / f"pdc-sbx-venv-{os.getpid()}"
    venv.create(venv_dir, with_pip=False)
    try:
        python = venv_dir / "bin" / "python"
        # commands run with pedacito's own interpreter: simulate pedacito
        # itself installed in the /tmp venv
        monkeypatch.setattr(sys, "prefix", str(venv_dir))
        argv, _ = sandbox.wrap([str(python), "-c", "print('ok')"], cfg, tmp_path)
        r = sp.run(argv, cwd=tmp_path, capture_output=True, timeout=60)
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
        assert b"ok" in r.stdout
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)


def test_atomic_edits_write_nothing_when_one_file_fails(tmp_path):
    reply = ("--- FILE: a.py\n<<<<<<< SEARCH\nX = 1\n=======\nX = 2\n>>>>>>> REPLACE\n"
             "--- FILE: b.py\n<<<<<<< SEARCH\nnot there\n=======\nY = 2\n>>>>>>> REPLACE\n")
    files = {"a.py": "X = 1\n", "b.py": "Y = 1\n"}
    agent = _agent(tmp_path / "off", Scripted(), files, allow_write=True)
    results, _ = agent.apply(reply, dry_run=False, task="t", gathered="")
    assert (tmp_path / "off" / "a.py").read_text() == "X = 2\n"        # partial, as before
    agent = _agent(tmp_path / "on", Scripted(), files, allow_write=True,
                   atomic_edits=True)
    results, _ = agent.apply(reply, dry_run=False, task="t", gathered="")
    assert (tmp_path / "on" / "a.py").read_text() == "X = 1\n"
    held = [r for r in results if r.file == "a.py"]
    assert held and not held[0].ok and "atomic_edits" in held[0].message
    assert any(r.file == "b.py" and not r.ok for r in results)


def test_restore_delete_created_removes_only_files_the_run_made(tmp_path):
    from pedacito import workspace
    from pedacito.edits import apply_edits, parse_edits
    (tmp_path / "keep.py").write_text("A = 1\n")
    reply = ("--- FILE: keep.py\n<<<<<<< SEARCH\nA = 1\n=======\nA = 2\n>>>>>>> REPLACE\n"
             "--- FILE: pkg/new.py\n<<<<<<< SEARCH\n=======\nB = 1\n>>>>>>> REPLACE\n")
    bdir = workspace.new_backup_dir(tmp_path)
    res = apply_edits(tmp_path, parse_edits(reply), backup=True, backup_dir=bdir)
    assert all(r.ok for r in res) and (tmp_path / "pkg/new.py").exists()
    assert workspace.list_backups(tmp_path) == [bdir]          # manifest is not a snapshot
    assert workspace.restore(tmp_path, bdir) == ["keep.py"]
    assert (tmp_path / "keep.py").read_text() == "A = 1\n"
    assert workspace.created_files(tmp_path, bdir) == ["pkg/new.py"]
    assert workspace.delete_created(tmp_path, bdir, dry_run=True) == ["pkg/new.py"]
    assert (tmp_path / "pkg/new.py").exists()
    assert workspace.delete_created(tmp_path, bdir) == ["pkg/new.py"]
    assert not (tmp_path / "pkg").exists() and (tmp_path / "keep.py").exists()


def test_calibration_probe_retries_with_thinking_off_when_the_reply_is_empty():
    from pedacito import modelcard

    class Thinker:
        last_finish_reason = ""
        calls: list = []

        def chat(self, messages, max_tokens, disable_thinking=None, **kw):
            self.calls.append((max_tokens, disable_thinking))
            if disable_thinking:
                self.last_finish_reason = "stop"
                return "OK: 1"
            self.last_finish_reason = "length"
            return ""

    t = Thinker()
    reply, note = modelcard._probe(t, [{"role": "user", "content": "x"}], 30)
    assert reply == "OK: 1" and "thinking off" in note
    assert t.calls[0][0] >= 1500 and t.calls[1][1] is True
