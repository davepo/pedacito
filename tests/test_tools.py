"""Unit tests for the agent's navigation tools.

These cover the forgiving behaviours added for small local models: reading a
definition by symbol instead of a guessed line range, grep's literal and
case-insensitive fallbacks plus directory/glob scopes, call-site snippets in
`callers`, and the explicit `append` action that replaced the broken
"repeat the tail and write again" chunking protocol.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import tools, workspace
from pedacito.chunker import chunk_source
from pedacito.config import Config
from pedacito.index import Index


APP = '''\
"""demo module"""


def helper(value):
    return value + 1


class Server:
    def start(self):
        return helper(1)

    def stop(self):
        return helper(2)


def run():
    s = Server()
    return s.start()


def CLI_ONLY():
    return "cli"
'''


def _index(tmp_path: Path, files: dict[str, str] | None = None) -> Index:
    files = files or {"pkg/app.py": APP}
    chunks = []
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
        chunks.extend(chunk_source(rel, src, 60))
    return Index(tmp_path, chunks)


def test_schema_hides_disabled_actions_and_fields():
    plain = tools.schema_for(Config())
    assert plain["properties"]["action"]["enum"] == [
        "read", "grep", "callers", "outline", "sessions", "done"]
    assert "content" not in plain["properties"]
    assert "to" not in plain["properties"]
    assert "command" not in plain["properties"]

    write_only = Config()
    write_only.allow_write = True
    wschema = tools.schema_for(write_only)
    assert "write" in wschema["properties"]["action"]["enum"]
    assert "run" not in wschema["properties"]["action"]["enum"]
    assert {"content", "to"} <= set(wschema["required"])
    assert "command" not in wschema["properties"]

    shell_only = Config()
    shell_only.allow_shell = True
    sschema = tools.schema_for(shell_only)
    assert "run" in sschema["properties"]["action"]["enum"]
    assert "write" not in sschema["properties"]["action"]["enum"]
    assert "command" in sschema["required"]
    assert "content" not in sschema["properties"]

    cfg = Config()
    cfg.allow_write = True
    cfg.allow_shell = True
    full = tools.schema_for(cfg)
    assert {"write", "append", "move", "delete", "run"} <= set(
        full["properties"]["action"]["enum"])
    assert {"content", "to", "command"} <= set(full["required"])


def test_read_by_symbol_resolves_definition(tmp_path):
    idx = _index(tmp_path)
    obs = tools.tool_read(idx, Config(), "", 0, 0, symbol="helper")
    assert obs.startswith("pkg/app.py lines"), obs
    assert "def helper" in obs
    assert "resolved to helper" in obs


def test_read_ambiguous_symbol_lists_choices(tmp_path):
    idx = _index(tmp_path, {
        "a.py": "def same():\n    return 1\n",
        "b.py": "def same():\n    return 2\n",
    })
    obs = tools.tool_read(idx, Config(), "", 0, 0, symbol="same")
    assert "2 definitions" in obs
    assert "a.py:1-" in obs and "b.py:1-" in obs


def test_read_missing_file_suggests_a_close_match(tmp_path):
    idx = _index(tmp_path)
    obs = tools.tool_read(idx, Config(), "pkg/ap.py", 1, 10)
    assert "Did you mean" in obs
    assert "pkg/app.py" in obs


def test_read_recovers_a_symbol_put_in_the_file_field(tmp_path):
    idx = _index(tmp_path)
    obs = tools.tool_read(idx, Config(), "helper", 0, 0)
    assert obs.startswith("pkg/app.py lines"), obs
    assert "def helper" in obs


def test_outline_accepts_a_symbol_name_as_its_file(tmp_path):
    idx = _index(tmp_path)
    obs = tools.tool_outline(idx, Config(), "helper")
    assert "pkg/app.py" in obs and "helper" in obs


def test_read_missing_symbol_suggests_a_close_match(tmp_path):
    idx = _index(tmp_path)
    obs = tools.tool_read(idx, Config(), "", 0, 0, symbol="helpr")
    assert "Did you mean" in obs
    assert "helper" in obs


def test_read_file_plus_symbol_narrows_to_that_file(tmp_path):
    idx = _index(tmp_path, {
        "a.py": "def same():\n    return 1\n",
        "b.py": "def same():\n    return 2\n",
    })
    obs = tools.tool_read(idx, Config(), "b.py", 0, 0, symbol="same")
    assert obs.startswith("b.py lines")
    assert "return 2" in obs


def test_grep_invalid_regex_is_searched_literally(tmp_path):
    idx = _index(tmp_path)
    obs = tools.tool_grep(idx, "def helper(")
    assert "not valid regex" in obs
    assert "def helper" in obs


def test_grep_retries_case_insensitively(tmp_path):
    idx = _index(tmp_path)
    obs = tools.tool_grep(idx, "cli_only(")
    assert "case-insensitively" in obs
    assert "CLI_ONLY" in obs


def test_grep_scope_accepts_directory_and_glob(tmp_path):
    idx = _index(tmp_path, {
        "pkg/app.py": "TOKEN = 1\n",
        "other/app.py": "TOKEN = 2\n",
    })
    only_pkg = tools.tool_grep(idx, "TOKEN", "pkg")
    assert "pkg/app.py" in only_pkg and "other/app.py" not in only_pkg

    both = tools.tool_grep(idx, "TOKEN", "*.py")
    assert "pkg/app.py" in both and "other/app.py" in both


def test_grep_hit_names_the_enclosing_symbol(tmp_path):
    idx = _index(tmp_path)
    obs = tools.tool_grep(idx, r"return helper")
    assert "[Server]" in obs


def test_callers_shows_call_sites_and_callees(tmp_path):
    idx = _index(tmp_path)
    helper = tools.tool_callers(idx, "helper")
    assert "Called from:" in helper
    assert "helper(" in helper        # a real call-site line, not just a range

    run = tools.tool_callers(idx, "run")
    assert "Calls (indexed definitions):" in run
    assert "Server:" in run and "start:" in run


def test_append_creates_then_grows_a_file(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_write = True

    first = tools.dispatch(idx, cfg, {"action": "append", "file": "new.py",
                                      "content": "a = 1\nb = 2"})
    assert first.startswith("CREATED new.py") and "append" in first
    second = tools.dispatch(idx, cfg, {"action": "append", "file": "new.py",
                                       "content": "c = 3\nd = 4"})
    assert second.startswith("APPENDED 2 line(s)")
    assert (tmp_path / "new.py").read_text() == "a = 1\nb = 2\nc = 3\nd = 4\n"


def test_oversized_append_is_told_to_split_the_append(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_write = True
    cfg.max_write_lines = 2

    obs = tools.dispatch(idx, cfg, {"action": "append", "file": "x.py",
                                    "content": "a\nb\nc"})
    assert obs.startswith("ERROR") and "`append` at most" in obs
    assert "first `write`" not in obs
    assert not (tmp_path / "x.py").exists()


def test_append_is_gated_by_allow_write(tmp_path):
    idx = _index(tmp_path)
    obs = tools.dispatch(idx, Config(), {"action": "append", "file": "x.py",
                                         "content": "a = 1"})
    assert obs.startswith("ERROR") and "allow_write" in obs
    assert not (tmp_path / "x.py").exists()


# --------------------------------------------------------------- move/delete
def test_move_and_delete_are_gated_by_allow_write(tmp_path):
    idx = _index(tmp_path)
    moved = tools.dispatch(idx, Config(), {"action": "move",
                                           "file": "pkg/app.py",
                                           "to": "pkg/new.py"})
    deleted = tools.dispatch(idx, Config(), {"action": "delete",
                                             "file": "pkg/app.py"})
    assert moved.startswith("ERROR") and "allow_write" in moved
    assert deleted.startswith("ERROR") and "allow_write" in deleted
    assert (tmp_path / "pkg/app.py").is_file()


def test_move_then_delete_one_file(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_write = True
    cfg.backup = False

    moved = tools.dispatch(idx, cfg, {"action": "move", "file": "pkg/app.py",
                                      "to": "pkg/renamed.py"})
    assert moved.startswith("MOVED pkg/app.py -> pkg/renamed.py"), moved
    assert not (tmp_path / "pkg/app.py").exists()
    assert (tmp_path / "pkg/renamed.py").is_file()
    assert "pkg/renamed.py" in idx.files

    deleted = tools.dispatch(idx, cfg, {"action": "delete",
                                        "file": "pkg/renamed.py"})
    assert deleted.startswith("DELETED pkg/renamed.py"), deleted
    assert not (tmp_path / "pkg/renamed.py").exists()
    assert "pkg/renamed.py" not in idx.files


def test_move_refuses_to_overwrite(tmp_path):
    idx = _index(tmp_path)
    (tmp_path / "other.py").write_text("x = 1\n", encoding="utf-8")
    cfg = Config()
    cfg.allow_write = True
    obs = tools.dispatch(idx, cfg, {"action": "move", "file": "pkg/app.py",
                                    "to": "other.py"})
    assert obs.startswith("ERROR") and "already exists" in obs
    assert (tmp_path / "pkg/app.py").is_file()
    assert (tmp_path / "other.py").read_text() == "x = 1\n"


# -------------------------------------------------------------------- run
def test_run_is_disabled_by_default(tmp_path):
    idx = _index(tmp_path)
    obs = tools.dispatch(idx, Config(), {"action": "run",
                                         "command": "pytest -q"})
    assert obs.startswith("ERROR") and "allow_shell" in obs


def test_run_executes_without_a_shell(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_shell = True
    cfg.shell_timeout = 10
    obs = tools.dispatch(idx, cfg, {
        "action": "run",
        "command": f'{sys.executable} -c "print(123)"',
    })
    assert "EXIT: 0 (OK)" in obs
    assert "123" in obs


def test_run_stops_a_silent_gui_like_process(tmp_path):
    """The reported bug: `run python game.py` opens a window and blocks until
    someone closes it. A process that goes quiet must be stopped automatically,
    with an observation that continues the task instead of hanging."""
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_shell = True
    cfg.shell_timeout = 30
    cfg.shell_idle_seconds = 1
    started = time.time()
    obs = tools.dispatch(idx, cfg, {
        "action": "run",
        # A GUI typically prints a startup banner, then goes quiet forever.
        "command": (f'{sys.executable} -c "print(\'pygame 2.5.2\'); '
                    f'import time; time.sleep(30)"'),
    })
    assert "pygame 2.5.2" in obs
    assert "STOPPED: no output for 1s" in obs
    assert "GUI or a server" in obs
    assert time.time() - started < 10


def test_proc_stops_the_whole_process_tree(tmp_path):
    """A launcher that spawns the visible window as a child must not survive
    the idle stop -- that is the difference between a closed game and a window
    left on screen after pedacito continues."""
    if os.name == "nt":
        return                       # taskkill path is Windows-specific
    from pedacito import proc
    marker = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)'])\n"
        f"open({str(marker)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(30)\n", encoding="utf-8")
    _, _, stopped, _ = proc.run([sys.executable, str(script)], cwd=tmp_path,
                                timeout=20, idle_timeout=1)
    assert stopped == "idle"
    pid = int(marker.read_text())
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    raise AssertionError(f"grandchild {pid} survived the idle stop")


def test_run_timeout_is_reported(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_shell = True
    cfg.shell_timeout = 1
    started = time.time()
    obs = tools.dispatch(idx, cfg, {
        "action": "run",
        "command": f'{sys.executable} -c "import time; time.sleep(5)"',
    })
    assert "timed out after 1s" in obs
    assert time.time() - started < 4


def test_run_honours_allowlist(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_shell = True
    cfg.shell_allowlist = ("pytest",)
    obs = tools.dispatch(idx, cfg, {
        "action": "run",
        "command": f'{sys.executable} -c "print(1)"',
    })
    assert obs.startswith("ERROR") and "shell_allowlist" in obs


# ---------------------------------------------------------- semantic search
class _EmbedClient:
    """Deterministic stand-in for an OpenAI-compatible embeddings endpoint."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for text in texts:
            low = text.lower()
            if "alpha" in low or "increment" in low:
                out.append([1.0, 0.0, 0.0])
            elif "beta" in low or "decrement" in low:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return out


SEMANTIC_APP = """\
def alpha():
    return 1


def beta():
    return 2
"""


def test_grep_falls_back_to_semantic_search(tmp_path):
    idx = _index(tmp_path, {"pkg/app.py": SEMANTIC_APP})
    cfg = Config()
    cfg.semantic_search = True
    cfg.semantic_min_score = 0.1
    cfg.semantic_top_k = 2
    client = _EmbedClient()

    obs = tools.tool_grep(idx, "increment", cfg=cfg, client=client)
    assert "Semantic matches" in obs
    assert "alpha" in obs and "beta" not in obs
    assert "No matches" not in obs

    first_calls = client.calls
    tools.tool_grep(idx, "increment", cfg=cfg, client=client)
    assert client.calls == first_calls + 1, "unit embeddings were not cached"


# ----------------------------------------------------------------- sessions
def _session(tmp_path, when, task, reply, steps=("read pkg/app.py",)):
    workspace.write_session(tmp_path, task, list(steps), reply, [],
                            None, when=when)
    return f"{when}.md"


def test_sessions_lists_recent_runs(tmp_path):
    idx = _index(tmp_path)
    _session(tmp_path, "20260101-120000", "add retry logic", "done")
    _session(tmp_path, "20260102-120000", "rename the parser", "done")

    obs = tools.dispatch(idx, Config(), {"action": "sessions"})
    assert "20260102-120000.md" in obs
    assert "rename the parser" in obs
    assert obs.index("20260102-120000.md") < obs.index("20260101-120000.md")


def test_sessions_searches_and_retries_case_insensitively(tmp_path):
    idx = _index(tmp_path)
    _session(tmp_path, "20260101-120000", "add retry logic",
             "wrapped the sync call in a retry loop")

    obs = tools.dispatch(idx, Config(), {"action": "sessions",
                                         "pattern": "RETRY"})
    assert "case-insensitively" in obs
    assert "retry loop" in obs


def test_sessions_falls_back_to_literal_search(tmp_path):
    idx = _index(tmp_path)
    _session(tmp_path, "20260101-120000", "fix parser",
             "broke at def parse(text)")

    obs = tools.dispatch(idx, Config(), {"action": "sessions",
                                         "pattern": "def parse("})
    assert "not valid regex" in obs
    assert "def parse" in obs


def test_sessions_reads_one_log_safely(tmp_path):
    idx = _index(tmp_path)
    name = _session(tmp_path, "20260101-120000", "add retry logic", "done")

    obs = tools.dispatch(idx, Config(), {"action": "sessions", "file": name,
                                         "start": 1, "end": 5})
    assert f"{name} (session log, not source) lines 1-5" in obs
    assert "**Task:** add retry logic" in obs

    bare = tools.dispatch(idx, Config(), {"action": "sessions",
                                           "file": name[:-3]})
    assert "add retry logic" in bare

    escaped = tools.dispatch(idx, Config(), {"action": "sessions",
                                             "file": "../index.json"})
    assert escaped.startswith("ERROR")


def test_sessions_scope_restricts_search(tmp_path):
    idx = _index(tmp_path)
    one = _session(tmp_path, "20260101-120000", "alpha task",
                   "shared keyword here")
    _session(tmp_path, "20260102-120000", "beta task", "shared keyword here")

    obs = tools.dispatch(idx, Config(), {"action": "sessions",
                                         "pattern": "shared keyword",
                                         "file": one})
    assert one in obs
    assert "20260102-120000.md" not in obs


def test_lmstudio_embed_returns_vectors_in_input_order(monkeypatch):
    from pedacito.llm import LMStudio

    class Response:
        status_code = 200
        text = ""

        def json(self):
            # Deliberately out of order; embed() must sort by `index`.
            return {"data": [{"index": 1, "embedding": [2.0, 0.0]},
                             {"index": 0, "embedding": [1.0, 0.0]}]}

    monkeypatch.setattr("pedacito.llm.requests.post",
                        lambda *a, **k: Response())
    cfg = Config()
    cfg.base_url = "http://example.test/v1"
    cfg.model = "m"
    assert LMStudio(cfg).embed(["a", "b"]) == [[1.0, 0.0], [2.0, 0.0]]


def test_semantic_failure_does_not_lose_the_grep_step(tmp_path):
    idx = _index(tmp_path, {"pkg/app.py": SEMANTIC_APP})

    class BadEmbed:
        def embed(self, texts):
            raise RuntimeError("server has no embeddings")

    cfg = Config()
    cfg.semantic_search = True
    obs = tools.tool_grep(idx, "increment", cfg=cfg, client=BadEmbed())
    assert "semantic search unavailable" in obs
    assert "No matches" in obs
