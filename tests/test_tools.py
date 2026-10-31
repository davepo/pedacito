"""Unit tests for the agent's navigation tools.

These cover the forgiving behaviours added for small local models: reading a
definition by symbol instead of a guessed line range, grep's literal and
case-insensitive fallbacks plus directory/glob scopes, call-site snippets in
`callers`, and the explicit `append` action that replaced the broken
"repeat the tail and write again" chunking protocol.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import tools, workspace
from pedacito.agent import Agent
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
        "read", "grep", "find", "callers", "outline", "sessions", "todo",
        "diff", "done"]
    # `content` stays because the always-available todo action uses it.
    assert "content" in plain["properties"]
    assert "to" not in plain["properties"]
    assert "command" not in plain["properties"]

    nodiff = Config()
    nodiff.allow_diff = False
    nschema = tools.schema_for(nodiff)
    assert "diff" not in nschema["properties"]["action"]["enum"]

    compact = Config()
    compact.compact_step_schema = True
    cschema = tools.schema_for(compact)
    assert cschema["required"] == ["action", "reasoning"]
    # The load-bearing field ORDER must survive: args before reasoning, so a
    # truncated reply still salvages to a complete action.
    assert list(cschema["properties"]) == [
        "action", "file", "start", "end", "pattern", "symbol", "content",
        "reasoning"]

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
    assert "diagnose" in sschema["properties"]["action"]["enum"]
    assert "write" not in sschema["properties"]["action"]["enum"]
    assert "command" in sschema["required"]
    assert "content" in sschema["properties"]

    cfg = Config()
    cfg.allow_write = True
    cfg.allow_shell = True
    full = tools.schema_for(cfg)
    assert {"write", "append", "move", "delete", "run"} <= set(
        full["properties"]["action"]["enum"])
    assert {"content", "to", "command"} <= set(full["required"])

    shot = Config()
    shot.allow_screenshots = True
    shschema = tools.schema_for(shot)
    assert "screenshot" in shschema["properties"]["action"]["enum"]
    assert "run" not in shschema["properties"]["action"]["enum"]
    assert "command" in shschema["required"]
    assert "content" in shschema["properties"]


def test_gather_has_no_default_step_cap(tmp_path):
    """max_steps=0 (the default) must not stop at the old internal ceiling;
    only the token/gathered budget or a `done` ends the gather loop."""
    idx = _index(tmp_path)
    cfg = Config()
    assert cfg.max_steps == 0

    class ManySteps:
        step_truncated = False

        def __init__(self, target):
            self.target = target
            self.n = 0

        def chat_json(self, messages, schema, max_tokens, **kw):
            self.n += 1
            if self.n > self.target:
                return {"action": "done", "reasoning": "ok"}
            if self.n == 1:
                return {"action": "read", "file": "pkg/app.py",
                        "start": 1, "end": 1, "reasoning": "start"}
            # Vary the lookup so this checks the step ceiling, not the
            # duplicate-read guard (which correctly stops the same read).
            return {"action": "grep", "file": "pkg/app.py",
                    "pattern": f"needle_{self.n}", "reasoning": "again"}

    client = ManySteps(85)
    Agent(idx, client, cfg).gather("look at the app", verbose=False)
    assert client.n == 86, f"gather stopped after {client.n - 1} steps"


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


def test_outline_directory_lists_file_cards(tmp_path):
    """`outline` pointed at a directory renders the file cards under it --
    the way down from a directory-tier map to readable line ranges."""
    idx = _index(tmp_path, {
        "pkg/a.py": "def alpha():\n    return 1\n",
        "pkg/b.py": "def beta():\n    return 2\n",
        "other/c.py": "def gamma():\n    return 3\n",
    })
    obs = tools.tool_outline(idx, Config(), "pkg")
    assert "### pkg (2 files)" in obs
    assert "pkg/a.py" in obs and "pkg/b.py" in obs
    assert "other/c.py" not in obs
    assert "alpha:L1-2" in obs              # cards keep their line ranges
    assert "Directory listing" in obs

    # A glob works the same way.
    globbed = tools.tool_outline(idx, Config(), "*.py")
    assert "pkg/a.py" in globbed and "other/c.py" in globbed

    # A name that matches nothing still gets the file-not-found hint, not a
    # dump of every card in the project.
    miss = tools.tool_outline(idx, Config(), "nosuchdir")
    assert "ERROR" in miss and "### " not in miss


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


def test_grep_sparse_hits_include_context_lines(tmp_path):
    """A grep with a handful of hits shows one line of context around each
    (GNU-grep style `file-N-` separator): with so few matches the model is
    looking at THE place, and the neighbouring line saves a `read` round
    trip."""
    idx = _index(tmp_path, {
        "m.py": "alpha = 1\nNEEDLE = 2\nomega = 3\n",
    })
    obs = tools.tool_grep(idx, "NEEDLE")
    assert "m.py:2: NEEDLE = 2" in obs
    assert "m.py-1- alpha = 1" in obs       # context before...
    assert "m.py-3- omega = 3" in obs       # ...and after
    assert "few matches" in obs             # and the note explaining '-'


def test_grep_dense_hits_have_no_context(tmp_path):
    """Once a grep returns many hits, context is noise: the plain hit list
    must come back without it (and without doubling the tokens)."""
    body = "".join(f"target_{i} = {i}\nfiller_{i} = {i}\n" for i in range(8))
    idx = _index(tmp_path, {"m.py": body})
    obs = tools.tool_grep(idx, "target_")
    assert "m.py:1: target_0" in obs
    assert "filler_0" not in obs            # no context lines
    assert "few matches" not in obs


def test_grep_miss_suggests_similar_symbols(tmp_path):
    """A total miss should not be a dead end: the index knows every defined
    name, so a near-miss of one comes back as a suggestion instead."""
    idx = _index(tmp_path)
    obs = tools.tool_grep(idx, "helpr")
    assert "No matches" in obs
    assert "similar symbol names" in obs
    assert "helper" in obs


def test_grep_file_path_reports_import_and_text_references(tmp_path):
    """`grep collision.py` is how small models phrase "find references to
    collision.py". It used to search for that literal string, find nothing,
    and fall into semantic search (which re-embedded the project). Return a
    proper import+mention report instead."""
    idx = _index(tmp_path, {
        "collision.py": "def check():\n    return True\n",
        "fly_spawner.py": "from collision import check\nx = 1\n",
        "notes.py": "# See collision.py for the collision rules.\nx = 1\n",
    })
    obs = tools.tool_grep(idx, "collision.py")
    assert "References to collision.py" in obs
    assert "fly_spawner.py:1" in obs and "import" in obs
    assert "notes.py:1" in obs and "text" in obs
    assert "Found 2 code reference(s)" in obs


def test_grep_file_path_with_no_references_reports_dead_code(tmp_path):
    idx = _index(tmp_path, {
        "dead.py": "def unused():\n    return 1\n",
        "live.py": "x = 1\n",
    })
    obs = tools.tool_grep(idx, "dead.py")
    assert "No references to dead.py" in obs
    assert "dead-code candidate" in obs


def test_grep_file_reference_respects_an_explicit_scope(tmp_path):
    """An explicit scope means "search inside this file", not "find
    references to that file" -- the reference shortcut must not fire."""
    idx = _index(tmp_path, {
        "collision.py": "check = 1\n",
        "notes.py": "# collision.py is mentioned here\nx = 1\n",
    })
    obs = tools.tool_grep(idx, "collision.py", "notes.py")
    assert "References to" not in obs
    assert "notes.py:1" in obs


def test_read_mid_file_includes_the_enclosing_signature_and_imports(tmp_path):
    """A mid-function read otherwise shows bare statements with no def line
    and no import list. Both are already indexed, so include them for free."""
    src = ("import os\nimport sys\n\n"
           "def big():\n    a = os.getcwd()\n    b = sys.argv\n")
    idx = _index(tmp_path, {"pkg/app.py": src})
    obs = tools.tool_read(idx, Config(), "pkg/app.py", 5, 6)
    assert "[context: def big()" in obs
    assert "module imports: os, sys" in obs


def test_read_context_hint_can_be_disabled(tmp_path):
    src = "import os\n\ndef big():\n    a = os.getcwd()\n"
    idx = _index(tmp_path, {"pkg/app.py": src})
    cfg = Config()
    cfg.read_context_hint = False
    obs = tools.tool_read(idx, cfg, "pkg/app.py", 4, 4)
    assert "[context:" not in obs


def test_callers_shows_call_sites_and_callees(tmp_path):
    idx = _index(tmp_path)
    helper = tools.tool_callers(idx, "helper")
    assert "Called from:" in helper
    assert "helper(" in helper        # a real call-site line, not just a range

    run = tools.tool_callers(idx, "run")
    assert "Calls (indexed definitions):" in run
    assert "Server:" in run and "start:" in run


def test_callers_falls_back_to_cross_file_importers(tmp_path):
    idx = _index(tmp_path, {
        "pkg/app.py": "def helper():\n    return 1\n",
        "pkg/other.py": "from pkg.app import helper\nx = 1\n",
    })
    obs = tools.tool_callers(idx, "helper")
    assert "No callers" in obs
    assert "Imported by" in obs and "pkg/other.py" in obs


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


def test_write_warns_when_python_does_not_parse(tmp_path):
    """Advisory syntax check: the edit path refuses broken code, but the
    write path cannot (a chunked file is legitimately unparseable until its
    last append), so it must at least WARN in the same observation."""
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_write = True

    obs = tools.dispatch(idx, cfg, {"action": "write", "file": "bad.py",
                                    "content": "def f(:\n    pass\n"})
    assert obs.startswith("CREATED bad.py")
    assert "WARNING" in obs and "does not parse" in obs

    ok = tools.dispatch(idx, cfg, {"action": "write", "file": "good.py",
                                   "content": "def f():\n    return 1\n"})
    assert ok.startswith("CREATED good.py")
    assert "WARNING" not in ok


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
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        # A killed process can linger as a zombie until its new parent reaps
        # it. macOS/launchd can take a moment; "Z" means it is already gone.
        try:
            stat = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                capture_output=True, text=True, timeout=5).stdout.strip()
        except OSError:
            stat = ""
        if not stat or stat.startswith("Z"):
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


def test_run_output_keeps_head_and_tail(tmp_path):
    """Truncation must keep the TAIL: a test runner's failure summary and a
    traceback's final `Error:` line live at the END of the output, which
    head-only truncation threw away exactly when output was long enough to
    matter."""
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_shell = True
    cfg.max_shell_output_chars = 400
    obs = tools.dispatch(idx, cfg, {
        "action": "run",
        "command": (f'{sys.executable} -c "print(\'A\' * 2000); '
                    f'print(\'ZZTAILZZ\')"'),
    })
    assert "A" * 50 in obs           # head kept
    assert "ZZTAILZZ" in obs         # tail kept
    assert "middle omitted" in obs   # and the marker says what happened


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


# ------------------------------------------------------------- screenshot
def test_screenshot_is_disabled_by_default(tmp_path):
    idx = _index(tmp_path)
    obs = tools.dispatch(idx, Config(), {"action": "screenshot"})
    assert obs.startswith("ERROR") and "allow_screenshots" in obs


def test_screenshot_launch_also_needs_allow_shell(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_screenshots = True
    obs = tools.dispatch(idx, cfg, {"action": "screenshot",
                                    "command": "python main.py"})
    assert obs.startswith("ERROR") and "allow_shell" in obs


def test_screenshot_analyzes_black_and_colored_images(tmp_path):
    pytest.importorskip("PIL.Image")
    from PIL import Image

    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_screenshots = True

    black = tmp_path / "black.png"
    Image.new("RGB", (320, 200), (0, 0, 0)).save(black)
    obs = tools.dispatch(idx, cfg, {"action": "screenshot",
                                    "file": "black.png"})
    assert "essentially BLACK/BLANK" in obs
    assert "320x200" in obs

    colors = tmp_path / "colors.png"
    img = Image.new("RGB", (320, 200), (0, 0, 0))
    for x in range(320):
        for y in range(200, 100, -1):
            img.putpixel((x, y - 1), ((x * 7) % 256, (y * 5) % 256, 120))
    img.save(colors)
    obs2 = tools.dispatch(idx, cfg, {"action": "screenshot",
                                     "file": "colors.png"})
    assert "has visible variation" in obs2


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


def _backup(tmp_path: Path, stamp: str, rel: str, text: str) -> Path:
    """Simulate the pre-write copy every Pedacito write path makes into
    .pedacito/backups/<stamp>/, mirroring the tree."""
    snap = workspace.new_backup_dir(tmp_path, stamp)
    dest = snap / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return snap


def test_diff_summarizes_changes_since_snapshots(tmp_path):
    idx = _index(tmp_path)
    _backup(tmp_path, "20260101-120000", "pkg/app.py", APP)
    (tmp_path / "pkg/app.py").write_text(APP + "\nEXTRA = 1\n",
                                         encoding="utf-8")

    obs = tools.dispatch(idx, Config(), {"action": "diff"})
    assert "1 file(s) changed" in obs
    assert "DIFF pkg/app.py" in obs
    assert "+EXTRA = 1" in obs
    assert "not git" in obs                  # the provenance note


def test_diff_of_one_file_shows_the_unified_diff(tmp_path):
    idx = _index(tmp_path)
    _backup(tmp_path, "20260101-120000", "pkg/app.py", APP)
    (tmp_path / "pkg/app.py").write_text(
        APP.replace("def helper(value):", "def helper(value, extra=0):"),
        encoding="utf-8")

    obs = tools.dispatch(idx, Config(), {"action": "diff",
                                         "file": "pkg/app.py"})
    assert "snapshot 20260101-120000" in obs
    assert "-def helper(value):" in obs
    assert "+def helper(value, extra=0):" in obs


def test_diff_uses_the_newest_snapshot_of_a_file(tmp_path):
    idx = _index(tmp_path)
    _backup(tmp_path, "20260101-120000", "pkg/app.py", "OLD = 1\n")
    _backup(tmp_path, "20260102-120000", "pkg/app.py", "MIDDLE = 1\n")
    (tmp_path / "pkg/app.py").write_text("NEW = 1\n", encoding="utf-8")

    obs = tools.dispatch(idx, Config(), {"action": "diff",
                                         "file": "pkg/app.py"})
    assert "snapshot 20260102-120000" in obs
    assert "-MIDDLE = 1" in obs
    assert "OLD = 1" not in obs


def test_diff_marks_a_file_deleted_since_the_snapshot(tmp_path):
    idx = _index(tmp_path, {"pkg/app.py": APP, "gone.py": "x = 1\n"})
    _backup(tmp_path, "20260101-120000", "gone.py", "x = 1\n")
    (tmp_path / "gone.py").unlink()

    obs = tools.dispatch(idx, Config(), {"action": "diff", "file": "gone.py"})
    assert "DELETED" in obs


def test_diff_summary_skips_unchanged_files(tmp_path):
    idx = _index(tmp_path)
    _backup(tmp_path, "20260101-120000", "pkg/app.py", APP)   # == on disk

    obs = tools.dispatch(idx, Config(), {"action": "diff"})
    assert "No changes" in obs


def test_diff_without_snapshots_says_so(tmp_path):
    idx = _index(tmp_path)
    obs = tools.dispatch(idx, Config(), {"action": "diff"})
    assert "No Pedacito snapshots" in obs


def test_diff_unknown_file_lists_what_is_snapshotted(tmp_path):
    idx = _index(tmp_path)
    _backup(tmp_path, "20260101-120000", "pkg/app.py", APP)

    obs = tools.dispatch(idx, Config(), {"action": "diff", "file": "other.py"})
    assert "No snapshot of 'other.py'" in obs
    assert "pkg/app.py" in obs


def test_diff_is_gated_by_allow_diff(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_diff = False
    assert "diff" not in tools.schema_for(cfg)["properties"]["action"]["enum"]
    obs = tools.tool_diff(idx, cfg)
    assert obs.startswith("ERROR")


def test_lmstudio_embed_returns_vectors_in_input_order(monkeypatch):
    from pedacito.llm import LLMClient

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
    assert LLMClient(cfg).embed(["a", "b"]) == [[1.0, 0.0], [2.0, 0.0]]


def test_lmstudio_vision_sends_a_data_uri(tmp_path, monkeypatch):
    from pedacito.llm import LLMClient

    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-png")
    seen = {}

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "a black window"}}]}

    def post(*args, **kwargs):
        seen["payload"] = kwargs.get("json") or {}
        return Response()

    monkeypatch.setattr("pedacito.llm.requests.post", post)
    cfg = Config()
    cfg.base_url = "http://example.test/v1"
    cfg.model = "m"
    cfg.vision_model = "vl"
    assert LLMClient(cfg).vision("describe", image) == "a black window"
    assert seen["payload"]["model"] == "vl"
    content = seen["payload"]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


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


def test_unsupported_embeddings_endpoint_is_not_retried(tmp_path):
    """Lemonade's llama.cpp backend answers /v1/embeddings with 501 unless it
    was started with --embeddings. Every no-hit grep used to re-embed all
    units, fail again, and repeat the 189-unit banner. The verdict must be
    remembered for the run (keyed by endpoint+model)."""
    from pedacito import semantic
    from pedacito.llm import LLMError

    idx = _index(tmp_path, {"pkg/app.py": SEMANTIC_APP})

    class Unsupported:
        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            raise LLMError("Embeddings request failed (501): This server "
                           "does not support embeddings.")

    cfg = Config()
    cfg.semantic_search = True
    cfg.base_url = "http://test-501.invalid/v1"
    cfg.model = "no-embed-model"
    client = Unsupported()
    try:
        first = tools.tool_grep(idx, "increment", cfg=cfg, client=client)
        assert "semantic search unavailable" in first
        calls = client.calls
        second = tools.tool_grep(idx, "increment", cfg=cfg, client=client)
        assert client.calls == calls, \
            "the unsupported endpoint was retried instead of cached"
        assert "disabled for this run" in second
    finally:
        semantic._DISABLED.pop((cfg.base_url.rstrip("/"), cfg.model), None)


def test_ocr_ignores_missing_tesseract_language_noise(monkeypatch):
    """tesseract installed without eng.traineddata writes its init errors to
    stderr; they used to be returned as OCR text and fed to the model as page
    evidence. A failed run must report no OCR text."""
    from pedacito import vision

    monkeypatch.setattr(vision.shutil, "which",
                        lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(vision, "_run", lambda cmd, timeout=30: (False, (
        "Error opening data file /usr/share/tessdata/eng.traineddata\n"
        "Please make sure the TESSDATA_PREFIX environment variable is set "
        "to your \"tessdata\" directory.\n"
        "Failed loading language 'eng'\n"
        "Tesseract couldn't load any languages!\n"
        "Could not initialize tesseract.")))
    assert vision.ocr("screen.png") == ""


def test_ocr_keeps_real_text(monkeypatch):
    from pedacito import vision

    monkeypatch.setattr(vision.shutil, "which",
                        lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(vision, "_run",
                        lambda cmd, timeout=30: (True, "Time: 4\nScore: 403"))
    assert vision.ocr("screen.png") == "Time: 4\nScore: 403"


def test_session_log_is_utf8_on_any_platform(tmp_path):
    """Model replies contain em dashes, arrows, and non-ASCII paths. Session
    logs used Path.write_text() with the platform default encoding, which
    raises UnicodeEncodeError on a Windows cp1252 console host."""
    p = workspace.write_session(
        tmp_path, "move the dead — module", ["1. grep — collision"],
        "Done — moved the file.", [], None, when="20260101-000000")
    text = p.read_text(encoding="utf-8")
    assert "dead — module" in text
    assert "Done — moved the file." in text


def test_windows_command_split_keeps_backslashes(monkeypatch):
    """POSIX shlex eats backslashes in Windows paths; the non-POSIX splitter
    keeps them and leaves surrounding quotes for the stripper."""
    monkeypatch.setattr(tools.os, "name", "nt")
    argv = tools._split_command(
        '"C:\\Program Files\\Python\\python.exe" "a b.py"')
    assert argv == ["C:\\Program Files\\Python\\python.exe", "a b.py"]


def test_allowlist_matches_python_exe_on_windows(monkeypatch):
    monkeypatch.setattr(tools.os, "name", "nt")
    cfg = Config()
    cfg.shell_allowlist = ("python",)
    assert tools._allowlist_error(cfg, ["python.exe", "-V"]) == ""


# ------------------------------------------------------- index freshness/map
def test_stale_skips_hashing_when_stat_matches(tmp_path, monkeypatch):
    """The [mtime_ns, size] prefilter: a file whose stat has not moved must
    not be hashed at all; a touch must cost exactly one hash and then be
    remembered (stats_refreshed tells the caller to persist that); and a
    real edit must still be caught."""
    from pedacito import index as index_mod

    idx = _index(tmp_path)
    p = tmp_path / "pkg/app.py"
    idx.file_hashes["pkg/app.py"] = index_mod.file_hash(p)
    idx.file_stats["pkg/app.py"] = index_mod.file_stat(p)

    calls = []
    real_hash = index_mod.file_hash
    monkeypatch.setattr(index_mod, "file_hash",
                        lambda path: (calls.append(str(path)),
                                      real_hash(path))[1])

    assert idx.stale() == ([], [])
    assert calls == []                       # stat matched: no hashing
    assert not idx.stats_refreshed

    future = time.time() + 5
    os.utime(p, (future, future))            # touch: stat differs, bytes same
    assert idx.stale() == ([], [])
    assert len(calls) == 1                   # paid exactly one hash
    assert idx.stats_refreshed

    assert idx.stale() == ([], [])
    assert len(calls) == 1                   # refreshed stat: no rehash

    p.write_text(APP + "\nEXTRA = 1\n", encoding="utf-8")
    changed, missing = idx.stale()
    assert changed == ["pkg/app.py"] and missing == []
    assert len(calls) == 2


def test_map_for_prompt_falls_back_to_dir_tier(tmp_path):
    """A project too big even for file cards gets the directory-tier map:
    one card per top-level directory, which `outline <dir>` descends."""
    files = {f"pkg{d}/mod{m}.py": f"def f{m}():\n    return {m}\n"
             for d in range(3) for m in range(4)}
    idx = _index(tmp_path, files)

    cfg = Config()
    cfg.flat_map_token_budget = 5            # nothing fits but the dir cards
    text, tier = idx.map_for_prompt(cfg)
    assert tier == "dirs"
    assert "### pkg0/" in text
    assert "pkg0/mod0.py" in text

    # And the directory tier is what `outline` descends from.
    obs = tools.tool_outline(idx, cfg, "pkg1")
    assert "### pkg1 (4 files)" in obs
    assert "pkg1/mod2.py" in obs

    cfg.flat_map_token_budget = 10 ** 9
    text, tier = idx.map_for_prompt(cfg)
    assert tier == "flat"


def test_read_lines_cached_invalidates_on_change(tmp_path):
    from pedacito import fileio

    p = tmp_path / "f.txt"
    p.write_text("one\n", encoding="utf-8")
    first = fileio.read_lines_cached(p)
    assert first == ["one", ""]
    assert fileio.read_lines_cached(p) is first      # served from the cache

    p.write_text("two\nthree\n", encoding="utf-8")   # new stat -> new entry
    assert fileio.read_lines_cached(p) == ["two", "three", ""]


# ------------------------------------------------- runaway / malformed input
def test_gather_stops_when_the_model_repeats_a_failing_read(tmp_path):
    """The regression that forced a manual cancel: a model stuck on one bad
    `read` produced an ERROR observation every step. Errors skipped the
    gathered-source budget and `continue`d before the step guard, so with the
    default max_steps=0 the gather loop never ended."""
    idx = _index(tmp_path)
    cfg = Config()
    cfg.max_gathered_chars = 100_000

    class Stuck:
        step_truncated = False

        def __init__(self):
            self.n = 0

        def chat_json(self, messages, schema, max_tokens, **kw):
            self.n += 1
            return {"action": "read", "file": "does_not_exist.py",
                    "start": 1, "end": 20, "reasoning": "try again"}

    client = Stuck()
    Agent(idx, client, cfg).gather("look at the app", verbose=False)
    assert client.n <= 4, f"gather kept spinning for {client.n} steps"


def test_gather_stops_after_consecutive_different_errors(tmp_path):
    """Different bad paths are still a stall: cap the error streak instead of
    relying only on the exact-repeat signature."""
    idx = _index(tmp_path)
    cfg = Config()
    cfg.max_gathered_chars = 100_000

    class Failing:
        step_truncated = False

        def __init__(self):
            self.n = 0

        def chat_json(self, messages, schema, max_tokens, **kw):
            self.n += 1
            return {"action": "read", "file": f"missing_{self.n}.py",
                    "start": 1, "end": 20, "reasoning": "next"}

    client = Failing()
    Agent(idx, client, cfg).gather("look at the app", verbose=False)
    assert client.n <= 5, f"gather did not cap a failure streak ({client.n})"


def test_read_and_grep_cannot_escape_the_project_root(tmp_path):
    """`../secret.py` used to be adopted into the index on demand, which made
    read/outline/grep serve a file outside the project root."""
    root = tmp_path / "proj"
    idx = _index(root)
    (tmp_path / "secret.py").write_text("TOPSECRET = 1\n", encoding="utf-8")
    cfg = Config()

    read = tools.tool_read(idx, cfg, "../secret.py", 1, 10)
    assert read.startswith("ERROR") and "TOPSECRET" not in read

    grep = tools.tool_grep(idx, "TOPSECRET", "../secret.py")
    assert "No matches for /TOPSECRET/" in grep
    assert "= 1" not in grep

    outline = tools.tool_outline(idx, cfg, "../secret.py")
    assert "TOPSECRET" not in outline


def test_absolute_path_inside_root_is_stored_relative(tmp_path):
    """A model that sends an absolute path must not create an absolute
    index/backup key (which silently skipped the backup and leaked host
    paths into observations)."""
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_write = True
    cfg._backup_dir = tmp_path / ".pedacito" / "backups" / "s"
    abs_path = str(tmp_path / "pkg" / "app.py")

    obs = tools.dispatch(idx, cfg, {"action": "write", "file": abs_path,
                                    "content": "def helper():\n    return 9\n"})
    assert obs.startswith("WROTE pkg/app.py"), obs
    assert (cfg._backup_dir / "pkg" / "app.py").is_file()
    assert (tmp_path / "pkg" / "app.py").read_text().endswith("return 9\n")


def test_write_refuses_non_utf8_files_without_crashing(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_write = True
    binary = tmp_path / "bin.py"
    original = b"\xff\xfe\x00binary\n"
    binary.write_bytes(original)

    obs = tools.dispatch(idx, cfg, {"action": "write", "file": "bin.py",
                                    "content": "x = 1\n"})
    assert obs.startswith("ERROR") and "UTF-8" in obs
    assert binary.read_bytes() == original


def test_move_and_delete_accept_non_utf8_files(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_write = True
    cfg.backup = False
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\xff\x00\xfe")

    moved = tools.dispatch(idx, cfg, {"action": "move", "file": "blob.bin",
                                      "to": "moved.bin"})
    assert moved.startswith("MOVED blob.bin -> moved.bin"), moved
    deleted = tools.dispatch(idx, cfg, {"action": "delete",
                                        "file": "moved.bin"})
    assert deleted.startswith("DELETED moved.bin"), deleted
    assert not (tmp_path / "moved.bin").exists()


def test_null_and_non_numeric_fields_do_not_crash_a_step(tmp_path):
    idx = _index(tmp_path)
    cfg = Config()

    assert tools.dispatch(idx, cfg, {"action": "callers",
                                     "symbol": None}).startswith("ERROR")
    # A non-numeric range is coerced to the defaults, not raised.
    obs = tools.dispatch(idx, cfg, {"action": "read", "file": "pkg/app.py",
                                    "start": "oops", "end": None})
    assert obs.startswith("pkg/app.py lines"), obs
    # A malformed step object is reported, not an AttributeError.
    assert tools.dispatch(idx, cfg, None).startswith("ERROR")


def test_gather_write_snapshots_into_backups_not_a_bak_beside_source(tmp_path):
    """Nothing set `cfg._backup_dir`, so gather-phase writes fell through to
    `<file>.bak`, which the next index pass could pick up as project code.
    The snapshot belongs under .pedacito/backups/ like every other path."""
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_write = True
    cfg.backup = True

    class Writer:
        step_truncated = False

        def __init__(self):
            self.n = 0

        def chat_json(self, messages, schema, max_tokens, **kw):
            self.n += 1
            if self.n == 1:
                return {"action": "write", "file": "pkg/app.py",
                        "content": "x = 1\n", "reasoning": "replace"}
            return {"action": "done", "reasoning": "ok"}

    Agent(idx, Writer(), cfg).gather("change the app", verbose=False)
    assert (tmp_path / "pkg" / "app.py").read_text() == "x = 1\n"
    assert not (tmp_path / "pkg" / "app.py.bak").exists()
    backups = list((tmp_path / ".pedacito" / "backups").rglob("pkg/app.py"))
    assert backups, "gather write did not snapshot into .pedacito/backups"


def test_repeated_writes_keep_the_original_in_the_backup(tmp_path):
    idx = _index(tmp_path)
    original = (tmp_path / "pkg" / "app.py").read_bytes()
    cfg = Config()
    cfg.allow_write = True
    cfg.backup = True
    cfg._backup_dir = tmp_path / "snap"
    for body in ("x = 1\n", "x = 2\n"):
        obs = tools.dispatch(idx, cfg, {"action": "write", "file": "pkg/app.py",
                                        "content": body})
        assert obs.startswith("WROTE"), obs
    assert (tmp_path / "snap" / "pkg" / "app.py").read_bytes() == original


def test_read_empty_file_reports_zero_lines(tmp_path):
    idx = _index(tmp_path)
    (tmp_path / "empty.py").write_text("", encoding="utf-8")
    obs = tools.tool_read(idx, Config(), "empty.py", 1, 10)
    assert "empty (0 lines)" in obs


# ---------------------------------------------------------------- web / browser
def test_browser_schema_is_hidden_until_enabled():
    plain = tools.schema_for(Config())
    assert "browser" not in plain["properties"]["action"]["enum"]
    assert "command" not in plain["properties"]

    cfg = Config()
    cfg.allow_browser = True
    schema = tools.schema_for(cfg)
    assert "browser" in schema["properties"]["action"]["enum"]
    assert "command" in schema["required"]
    assert "content" in schema["properties"]


def test_browser_server_url_detection_and_action_parser():
    from pedacito import browser

    assert browser.find_server_url(
        " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)"
    ) == "http://127.0.0.1:5000/"
    assert browser.find_server_url(
        "Serving at http://0.0.0.0:8000/game") == \
        "http://127.0.0.1:8000/game"
    assert browser.find_server_url("no URL here") == ""
    assert browser.parse_actions(
        "press Enter; click #start; wait 2; fill #name Ada; type hi") == [
        ("press", "Enter"), ("click", "#start"), ("wait", 2.0),
        ("fill", ("#name", "Ada")), ("type", "hi")]
    assert browser.parse_actions("Enter") == [("press", "Enter")]
    # Run 9's frog-scale verify could only RECOMMEND "force state.data.scale
    # = 2 in the browser console" -- eval gives the pattern language that
    # reach. Run 10 then showed the `;` splitter shredding multi-statement
    # JS into a SyntaxError eval plus `press d.scale=2: Unknown key`
    # fragments, so an eval now consumes the REST of the pattern (the docs
    # say "put eval last"); `js` is accepted as an alias.
    assert browser.parse_actions(
        "press Enter; wait 2; eval var d = window.__game.data; "
        "d.scale = 3") == [
        ("press", "Enter"), ("wait", 2.0),
        ("eval", "var d = window.__game.data; d.scale = 3")]
    assert browser.parse_actions("eval") == []      # empty expr: dropped
    # Run 10's forced-state expressions were truncated mid-token at 200,
    # and schema-constrained sampling cannot emit what the schema forbids.
    # Run 13 then guillotined every ~750-char measurement eval at exactly
    # 600: the grammar force-closes the string AT maxLength -- valid JSON,
    # finish_reason "stop", no truncation flag -- and the browser ran the
    # fragment (identical SyntaxError in every retry; five checks retired).
    # The cap must hold a legitimate measurement payload, and agent.gather
    # refuses to dispatch one arriving at exactly it.
    assert tools.STEP_SCHEMA["properties"]["pattern"]["maxLength"] == 2000
    # Fix ZC (run 13e): the RESULT side needs the same budget, advertised
    # where the model reads it every step -- five checks retired on
    # measurements that HAD succeeded but reached the model decapitated
    # at the old 240-char value clip ("the live eval output was truncated
    # before returning whiteTop and blackBelow").
    assert ("2000 chars" in
            tools.STEP_SCHEMA["properties"]["pattern"]["description"])


def test_browser_eval_action_runs_and_reports_its_value(tmp_path, monkeypatch):
    """`eval` must execute one JS expression in the page and put the RESULT
    into the action log -- the feedback is what lets a model confirm the
    forced state took before the capture grades it (run 9's frog-scale
    check needed exactly this and could only recommend it in prose)."""
    import sys
    import types as pytypes
    from pedacito import browser

    class FakeKeyboard:
        def __init__(self):
            self.pressed = []

        def press(self, key):
            self.pressed.append(key)

        def type(self, text):
            pass

    class FakePage:
        url = "http://127.0.0.1:5000/"

        def __init__(self):
            self.evaled = []
            self.keyboard = FakeKeyboard()

        def set_default_timeout(self, ms):
            pass

        def on(self, event, handler):
            pass

        def goto(self, url, wait_until=None, timeout=None):
            pass

        def evaluate(self, expr):
            if expr.startswith("()"):
                return []          # the _canvas_info probe
            self.evaled.append(expr)
            return 3               # as if the assignment yielded the value

        def wait_for_timeout(self, ms):
            pass

        def screenshot(self, path=None, full_page=False):
            with open(path, "wb") as fh:
                fh.write(b"fake png")

        def inner_text(self, sel):
            return ""

        def title(self):
            return "Game"

    class FakeBrowser:
        def __init__(self, page):
            self._page = page

        def new_page(self, viewport=None):
            return self._page

        def close(self):
            pass

    class FakeChromium:
        def __init__(self, page):
            self._page = page

        def launch(self, headless=True):
            return FakeBrowser(self._page)

    class FakePW:
        def __init__(self, page):
            self.chromium = FakeChromium(page)

    class FakeCM:
        def __init__(self, pw):
            self._pw = pw

        def __enter__(self):
            return self._pw

        def __exit__(self, *exc):
            return False

    page = FakePage()
    fake_sync = pytypes.ModuleType("playwright.sync_api")
    fake_sync.sync_playwright = lambda: FakeCM(FakePW(page))
    fake_root = pytypes.ModuleType("playwright")
    fake_root.sync_api = fake_sync
    monkeypatch.setitem(sys.modules, "playwright", fake_root)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync)
    monkeypatch.setattr(browser, "available", lambda: True)

    ok, info = browser.run(
        "http://127.0.0.1:5000/",
        actions="press Enter; eval state.data.scale = 3",
        wait=0, out=tmp_path / "shot.png")
    assert ok, info
    assert page.keyboard.pressed == ["Enter"]
    # Bug Z (run 13c): the page-side source now rides inside a race
    # wrapper (json-quoted) next to a settle-timer, so a never-resolving
    # forced-state Promise surfaces as an action error instead of hanging
    # evaluate -- which no set_default_timeout covers -- forever.
    assert len(page.evaled) == 1
    raced = page.evaled[0]
    assert "Promise.race" in raced
    assert '"state.data.scale = 3"' in raced      # source embedded verbatim
    assert "setTimeout" in raced and "did not settle" in raced
    assert "pressed Enter" in info["actions"]
    assert "eval'd state.data.scale = 3 -> 3" in info["actions"]
    assert info["action_errors"] == []


def test_raced_eval_bounds_a_never_settling_promise():
    """Bug Z end-to-end (run 13c): a forced-state `new Promise(res =>
    ...rAF...)` whose callback throws stays pending forever, and
    page.evaluate -- which set_default_timeout does NOT cover -- waits
    for it forever: the run hung 68 minutes with no log line while the
    LLM server idle-unloaded the model. The raced wrapper must turn that
    into a bounded rejection carrying the try/catch lesson, while plain
    completion-value evals, function-shaped evals, and settling Promises
    keep their exact semantics."""
    from pedacito import browser

    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        try:
            b = pw.chromium.launch(headless=True)
        except Exception as e:                  # pragma: no cover
            pytest.skip(f"chromium unavailable: {e}")
        try:
            pg = b.new_page()
            pg.goto("data:text/html,<title>t</title><body>hi</body>")

            t0 = time.monotonic()
            with pytest.raises(Exception) as ei:
                pg.evaluate(browser._raced_eval("new Promise(()=>{})", 1))
            elapsed = time.monotonic() - t0
            assert elapsed < 10, f"race timer never fired ({elapsed:.1f}s)"
            assert "did not settle within 1s" in str(ei.value)

            # The run-13c shape proper: the rAF callback throws, so the
            # Promise stays pending forever -- same bounded outcome.
            with pytest.raises(Exception):
                pg.evaluate(browser._raced_eval(
                    "new Promise(res => requestAnimationFrame(() => {"
                    "undefinedFunction(); res(1); }))", 1))

            # Completion-value semantics survive the wrapper (direct eval,
            # last statement's value), and a bare function shape is called
            # the way Playwright itself would call it.
            assert pg.evaluate(
                browser._raced_eval("var x = 41; x + 1", 5)) == 42
            assert pg.evaluate(browser._raced_eval("() => 7", 5)) == 7
            # A settling Promise resolves through the race with its value.
            assert pg.evaluate(browser._raced_eval(
                "new Promise(res => setTimeout(() => res('ok'), 50))",
                5)) == "ok"
        finally:
            b.close()


def test_unclosed_brackets_names_the_openers_that_never_close():
    """Bug ZB (run 13d): V8 reports an unclosed construct as a bare
    "Unexpected end of input" with NO position, so the model had no way
    to find the one missing `}` in its 1,174-char measurement eval (the
    IIFE body never closed). The scanner names every opener still live
    at end of input, with its position and surrounding context."""
    from pedacito import browser

    # The field shape: every inner level closes, the IIFE itself never
    # does -- the tail is one `}` short of `});})()`.
    hint = browser._unclosed_brackets(
        "(function(){return new Promise(function(res){res(1)})()")
    assert "unclosed" in hint
    assert "`(` from char 0" in hint
    assert "`{` from char 11" in hint
    assert "near the tail" in hint

    # Balanced expressions produce no hint -- including brackets that
    # live inside strings, template literals, or comments.
    assert browser._unclosed_brackets("(function(){return 1;})()") == ""
    assert browser._unclosed_brackets("var s = '})'; s + \"(\"") == ""
    assert browser._unclosed_brackets("var t = `a{b(`; t") == ""
    assert browser._unclosed_brackets("// }\nvar x = 1; x") == ""
    assert browser._unclosed_brackets("/* ( */ 1 + 1") == ""

    # A crossed pair and an unterminated string get their own diagnoses.
    assert "cross" in browser._unclosed_brackets("(x]")
    assert "string" in browser._unclosed_brackets("var s = 'abc")


def test_browser_eval_syntax_error_teaches_the_missing_closers(
        tmp_path, monkeypatch):
    """Bug ZB end-to-end through the tool: run 13d's model saw only
    "SyntaxError: Unexpected end of input" at a stack column pointing
    into Playwright's machinery, next to a 60-char head echo of its own
    arg -- and concluded the TOOL had truncated the string (it re-sent
    the same broken tail five times and blamed the harness in its
    FAILED verdict). The action error must state the length that
    arrived intact, carry the bracket report, and quote the tail where
    the missing closer belongs."""
    import sys
    import types as pytypes
    from pedacito import browser

    EXPR = "(function(){return new Promise(function(res){res(1)})()"

    class FakeKeyboard:
        def press(self, key):
            pass

        def type(self, text):
            pass

    class FakePage:
        url = "http://127.0.0.1:5000/"

        def __init__(self):
            self.keyboard = FakeKeyboard()

        def set_default_timeout(self, ms):
            pass

        def on(self, event, handler):
            pass

        def goto(self, url, wait_until=None, timeout=None):
            pass

        def evaluate(self, expr):
            if expr.startswith("()"):
                return []          # the _canvas_info probe
            raise Exception(
                "Page.evaluate: SyntaxError: Unexpected end of input\n"
                "    at eval (eval at evaluate (:370:30), "
                "<anonymous>:1:44)")

        def wait_for_timeout(self, ms):
            pass

        def screenshot(self, path=None, full_page=False):
            with open(path, "wb") as fh:
                fh.write(b"fake png")

        def inner_text(self, sel):
            return ""

        def title(self):
            return "Game"

    class FakeBrowser:
        def __init__(self, page):
            self._page = page

        def new_page(self, viewport=None):
            return self._page

        def close(self):
            pass

    class FakeChromium:
        def __init__(self, page):
            self._page = page

        def launch(self, headless=True):
            return FakeBrowser(self._page)

    class FakePW:
        def __init__(self, page):
            self.chromium = FakeChromium(page)

    class FakeCM:
        def __init__(self, pw):
            self._pw = pw

        def __enter__(self):
            return self._pw

        def __exit__(self, *exc):
            return False

    page = FakePage()
    fake_sync = pytypes.ModuleType("playwright.sync_api")
    fake_sync.sync_playwright = lambda: FakeCM(FakePW(page))
    fake_root = pytypes.ModuleType("playwright")
    fake_root.sync_api = fake_sync
    monkeypatch.setitem(sys.modules, "playwright", fake_root)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync)
    monkeypatch.setattr(browser, "available", lambda: True)

    ok, info = browser.run(
        "http://127.0.0.1:5000/",
        actions="press Enter; eval " + EXPR,
        wait=0, out=tmp_path / "shot.png")
    assert ok, info
    assert len(info["action_errors"]) == 1
    err = info["action_errors"][0]
    assert "SYNTAX ERROR" in err
    assert "Unexpected end of input" in err
    assert f"received all {len(EXPR)} chars" in err   # counters "truncated"
    assert "unclosed" in err and "`{` from char 11" in err
    assert "Source tail:" in err and EXPR[-12:] in err
    # The press before the eval still ran and logged; the eval itself
    # logged no value (it never executed).
    assert "pressed Enter" in info["actions"]
    assert not any("eval'd" in a for a in info["actions"])


def test_browser_eval_result_reaches_the_model_up_to_the_cap(
        tmp_path, monkeypatch):
    """Fix ZC (run 13e): five checks retired as failed on measurements
    that HAD succeeded -- the evals ran and resolved per-row pixel
    profiles, but the action log showed the model only str(value)[:240]
    ("the live eval output was truncated before returning whiteTop and
    blackBelow"), and the spec's checks demand per-row numbers, making
    them structurally unpassable. The result clip now mirrors the
    pattern cap: an eval may return as much as it may receive. The
    gather budget still caps the session total with its own explicit
    marker."""
    import sys
    import types as pytypes
    from pedacito import browser

    class FakeKeyboard:
        def press(self, key):
            pass

        def type(self, text):
            pass

    class FakePage:
        url = "http://127.0.0.1:5000/"

        def __init__(self):
            self.keyboard = FakeKeyboard()

        def set_default_timeout(self, ms):
            pass

        def on(self, event, handler):
            pass

        def goto(self, url, wait_until=None, timeout=None):
            pass

        def evaluate(self, expr):
            if expr.startswith("()"):
                return []          # the _canvas_info probe
            return "x" * 5000      # a per-row measurement profile, fat

        def wait_for_timeout(self, ms):
            pass

        def screenshot(self, path=None, full_page=False):
            with open(path, "wb") as fh:
                fh.write(b"fake png")

        def inner_text(self, sel):
            return ""

        def title(self):
            return "Game"

    class FakeBrowser:
        def __init__(self, page):
            self._page = page

        def new_page(self, viewport=None):
            return self._page

        def close(self):
            pass

    class FakeChromium:
        def __init__(self, page):
            self._page = page

        def launch(self, headless=True):
            return FakeBrowser(self._page)

    class FakePW:
        def __init__(self, page):
            self.chromium = FakeChromium(page)

    class FakeCM:
        def __init__(self, pw):
            self._pw = pw

        def __enter__(self):
            return self._pw

        def __exit__(self, *exc):
            return False

    page = FakePage()
    fake_sync = pytypes.ModuleType("playwright.sync_api")
    fake_sync.sync_playwright = lambda: FakeCM(FakePW(page))
    fake_root = pytypes.ModuleType("playwright")
    fake_root.sync_api = fake_sync
    monkeypatch.setitem(sys.modules, "playwright", fake_root)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync)
    monkeypatch.setattr(browser, "available", lambda: True)

    ok, info = browser.run(
        "http://127.0.0.1:5000/",
        actions="eval JSON.stringify(rows)",
        wait=0, out=tmp_path / "shot.png")
    assert ok, info
    entry = [a for a in info["actions"] if a.startswith("eval'd")][0]
    assert entry.startswith("eval'd JSON.stringify(rows) -> ")
    assert "x" * 2000 in entry          # the full cap reaches the model
    assert "x" * 2001 not in entry      # and not one char beyond


def test_browser_is_gated_and_explains_a_missing_backend(tmp_path, monkeypatch):
    idx = _index(tmp_path)
    obs = tools.tool_browser(idx, Config(), None, "http://127.0.0.1:5000")
    assert obs.startswith("ERROR") and "allow_browser" in obs

    cfg = Config()
    cfg.allow_browser = True
    monkeypatch.setattr(tools.browser, "available", lambda: False)
    monkeypatch.setattr(tools.browser, "chrome_executable", lambda: "")
    obs = tools.tool_browser(idx, cfg, None, "http://127.0.0.1:5000")
    assert obs.startswith("ERROR")
    assert "playwright" in obs.lower()
    assert "desktop screenshot" in obs.lower()


class _FakeProc:
    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def test_browser_launches_server_interacts_and_reports_page(tmp_path,
                                                            monkeypatch):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_browser = True
    cfg.allow_shell = True
    seen = {}

    def fake_launch(argv, *, cwd, env, wait_seconds):
        seen["argv"] = argv
        return _FakeProc(), "http://127.0.0.1:8000/game", "Serving at ..."

    def fake_run(url, *, actions, wait, out, timeout, headless):
        seen.update(url=url, actions=actions, wait=wait, out=out)
        return True, {
            "url": url, "title": "Turtle Game", "text": "Score: 0",
            "canvas": "800x600", "console": ["error: boom"],
            "actions": ["pressed Enter"], "action_errors": [],
            "method": "fake browser",
        }

    monkeypatch.setattr(tools.browser, "launch_server", fake_launch)
    monkeypatch.setattr(tools.browser, "run", fake_run)
    monkeypatch.setattr(tools.proc, "terminate", lambda p: None)
    monkeypatch.setattr(tools.vision, "analyze", lambda path, previous=None: "ANALYSIS")
    monkeypatch.setattr(tools.vision, "describe", lambda *a, **k: "")

    obs = tools.tool_browser(idx, cfg, None, "python main.py",
                             "press Enter; wait 3", 4)
    assert obs.startswith("BROWSER: http://127.0.0.1:8000/game")
    assert "LAUNCHED: python main.py" in obs
    assert "ACTIONS: pressed Enter" in obs
    assert "CANVAS: 800x600" in obs
    assert "PAGE TEXT:\nScore: 0" in obs
    assert "CONSOLE:" in obs and "boom" in obs
    assert "ANALYSIS:\nANALYSIS" in obs
    assert seen["url"] == "http://127.0.0.1:8000/game"
    assert seen["wait"] == 4


def test_screenshot_warns_that_a_desktop_shot_is_not_a_web_page(tmp_path,
                                                                monkeypatch):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_screenshots = True
    cfg.allow_shell = True

    monkeypatch.setattr(
        tools.browser, "launch_server",
        lambda argv, *, cwd, env, wait_seconds: (
            _FakeProc(), "http://127.0.0.1:8000/game", "Serving..."))
    monkeypatch.setattr(tools.proc, "terminate", lambda p: None)
    monkeypatch.setattr(tools.vision, "capture",
                        lambda path: (True, "fake capture"))
    monkeypatch.setattr(tools.vision, "analyze",
                        lambda path, previous=None: "ANALYSIS")
    monkeypatch.setattr(tools.vision, "describe", lambda *a, **k: "")

    obs = tools.tool_screenshot(idx, cfg, None, "python main.py", "", 1)
    assert "SERVER: http://127.0.0.1:8000/game" in obs
    assert "WEB SERVER" in obs and "DESKTOP" in obs
    assert "allow_browser is off" in obs or "allow_browser` is off" in obs


def test_run_reports_a_web_server_url(tmp_path, monkeypatch):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_shell = True
    monkeypatch.setattr(
        tools.proc, "run",
        lambda *a, **k: (0, " * Running on http://127.0.0.1:5000/", "idle", 1.0))
    obs = tools.tool_run(idx, cfg, "python main.py")
    assert "SERVER: http://127.0.0.1:5000/" in obs
    assert "WEB SERVER" in obs
    assert "browser" in obs


def test_browser_url_does_not_require_shell(tmp_path, monkeypatch):
    idx = _index(tmp_path)
    cfg = Config()
    cfg.allow_browser = True
    cfg.allow_shell = False
    monkeypatch.setattr(tools.browser, "run", lambda *a, **k: (
        True, {"url": "http://127.0.0.1:8000/", "title": "t",
               "text": "hello", "canvas": "", "console": [],
               "actions": [], "action_errors": [], "method": "fake"}))
    monkeypatch.setattr(tools.vision, "analyze",
                        lambda path, previous=None: "ANALYSIS")
    monkeypatch.setattr(tools.vision, "describe", lambda *a, **k: "")
    obs = tools.tool_browser(idx, cfg, None, "http://127.0.0.1:8000/")
    assert obs.startswith("BROWSER: http://127.0.0.1:8000/")
    assert "hello" in obs


# --------------------------------------------------------------------- find
def test_find_ranks_name_hits_above_signature_hits(tmp_path):
    src = ('def retry(fn):\n    return fn\n\n\n'
           'def wrapper(retry_once):\n    """Call it with backoff."""\n'
           '    return retry_once\n')
    idx = _index(tmp_path, {"pkg/app.py": src})
    obs = tools.tool_find(idx, "retry")
    assert "2 definition(s) matching 'retry'" in obs
    assert obs.index("def retry(fn)") < obs.index("def wrapper(retry_once)")


def test_find_matches_exact_and_partial_names(tmp_path):
    idx = _index(tmp_path)
    exact = tools.tool_find(idx, "helper")
    assert "def helper(value)" in exact
    partial = tools.tool_find(idx, "elp")     # substring of the qualname
    assert "def helper(value)" in partial


def test_find_matches_members_of_a_class_indexed_whole(tmp_path):
    """APP's Server fits in one chunk, so `start` lives in its members list
    and the qualified name finds it; the card stands for the whole class."""
    idx = _index(tmp_path)
    obs = tools.tool_find(idx, "Server.start")
    assert "[class] class Server" in obs
    assert "pkg/app.py:8-13" in obs


def test_find_matches_signature_and_docstring_words(tmp_path):
    idx = _index(tmp_path)
    by_sig = tools.tool_find(idx, "value")    # `def helper(value)`
    assert "def helper(value)" in by_sig
    src = 'def fetch(url):\n    """Download with backoff."""\n    return url\n'
    idx2 = _index(tmp_path / "sub", {"pkg/app.py": src})
    by_doc = tools.tool_find(idx2, "backoff")
    assert "def fetch(url)" in by_doc


def test_find_miss_suggests_close_names(tmp_path):
    idx = _index(tmp_path)
    obs = tools.tool_find(idx, "Servr")
    assert obs.startswith("No definition matches 'Servr'.")
    assert "Did you mean: Server?" in obs


def test_find_needs_a_pattern(tmp_path):
    idx = _index(tmp_path)
    obs = tools.tool_find(idx, "   ")
    assert obs.startswith("ERROR") and "`find` needs a pattern" in obs


def test_find_caps_the_hit_list(tmp_path):
    src = "".join(f"def target_{i}():\n    return {i}\n\n\n"
                  for i in range(tools.FIND_MAX_HITS + 5))
    idx = _index(tmp_path, {"many.py": src})
    obs = tools.tool_find(idx, "target_")
    assert obs.startswith(f"{tools.FIND_MAX_HITS + 5} definition(s) matching")
    assert obs.count("many.py:") == tools.FIND_MAX_HITS
    assert "(+5 more; narrow the pattern)" in obs


def test_find_is_dispatchable_in_gather(tmp_path):
    idx = _index(tmp_path)
    obs = tools.dispatch(idx, Config(), {"action": "find", "pattern": "Server"})
    assert "class Server" in obs


# ------------------------------------------------- read context: class owner
def test_read_method_shows_its_class_context(tmp_path):
    """A mid-method read shows the method signature AND the class header it
    belongs to: the class is too big to be in the window, but the chunk index
    already knows the owner (qualname `Big.m1` -> class `Big`)."""
    big = "import os\n\n\nclass Big:\n" + "".join(
        f"    def m{i}(self):\n        x = os.getcwd()\n        return {i}\n"
        + "\n" * 8 for i in range(7))
    idx = _index(tmp_path, {"big.py": big})
    m1 = next(c for c in idx.in_file("big.py") if c.qualname == "Big.m1")
    obs = tools.tool_read(idx, Config(), "big.py", m1.start + 1, m1.start + 1)
    assert "[context: def m1(self); in class Big" in obs


# ------------------------------------------------- cards: test/module links
def test_cards_link_modules_and_their_tests(tmp_path):
    idx = _index(tmp_path, {
        "pkg/foo.py": "def f():\n    return 1\n",
        "pkg/test_foo.py": ("from pkg.foo import f\n\n\n"
                            "def test_f():\n    assert f() == 1\n"),
    })
    cards = idx.render_cards(["pkg/foo.py", "pkg/test_foo.py"])
    foo_card, test_card = cards.split("\n\n")
    assert "  tested by: pkg/test_foo.py" in foo_card
    assert "  tests: pkg/foo.py" in test_card


def test_cards_test_link_needs_an_unambiguous_module(tmp_path):
    idx = _index(tmp_path, {
        "pkg/foo.py": "x = 1\n",
        "other/foo.py": "y = 2\n",
        "pkg/test_foo.py": "z = 3\n",
    })
    # Two modules share the stem: the test file gets no `tests:` link...
    assert "tests:" not in idx.render_cards(["pkg/test_foo.py"])
    # ...but each module still lists the test that covers its stem.
    assert "tested by: pkg/test_foo.py" in idx.render_cards(["pkg/foo.py"])


def test_cards_have_no_test_links_for_non_python(tmp_path):
    from pedacito.chunker_ts import TS_AVAILABLE, chunk_file_ts
    if not TS_AVAILABLE:
        pytest.skip("tree-sitter unavailable")
    files = {"web/app.js": "function a() { return 1; }\n",
             "web/test_app.js": "function b() { return 2; }\n"}
    chunks = []
    for rel, src in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(src, encoding="utf-8")
        chunks.extend(chunk_file_ts(rel, src, ".js", 60))
    idx = Index(tmp_path, chunks)
    assert "tested by:" not in idx.render_cards(["web/app.js"])
    assert "tests:" not in idx.render_cards(["web/test_app.js"])


# --------------------------------------------------- task write allowlist
def _armed_cfg() -> Config:
    cfg = Config()
    cfg.allow_write = True
    cfg.auto_allowlist = True
    cfg.backup = False
    return cfg


def test_auto_allowlist_blocks_writes_to_unread_files(tmp_path):
    idx = _index(tmp_path, {"pkg/app.py": APP, "pkg/other.py": "x = 1\n"})
    cfg = _armed_cfg()
    cfg._task_reads = {"pkg/app.py": [(1, 5)]}   # the live gather ledger
    cfg._task_allowlist = ()

    blocked = tools.dispatch(idx, cfg, {"action": "write",
                                        "file": "pkg/other.py",
                                        "content": "x = 2\n"})
    assert blocked.startswith("ERROR") and "task write allowlist" in blocked
    assert (tmp_path / "pkg/other.py").read_text() == "x = 1\n"

    allowed = tools.dispatch(idx, cfg, {"action": "write",
                                        "file": "pkg/app.py",
                                        "content": "y = 2\n"})
    assert allowed.startswith("WROTE pkg/app.py")

    created = tools.dispatch(idx, cfg, {"action": "write", "file": "fresh.py",
                                        "content": "z = 3\n"})
    assert created.startswith("CREATED fresh.py")   # creation always allowed


def test_auto_allowlist_honours_task_title_targets(tmp_path):
    idx = _index(tmp_path, {"pkg/other.py": "x = 1\n"})
    cfg = _armed_cfg()
    cfg._task_reads = {}                          # nothing was read...
    cfg._task_allowlist = ("pkg/other.py",)       # ...but the title names it
    obs = tools.dispatch(idx, cfg, {"action": "write", "file": "pkg/other.py",
                                    "content": "x = 2\n"})
    assert obs.startswith("WROTE pkg/other.py")


def test_auto_allowlist_off_or_unarmed_allows_everything(tmp_path):
    idx = _index(tmp_path, {"pkg/other.py": "x = 1\n"})
    off = Config()
    off.allow_write = True
    off.backup = False
    assert tools.dispatch(idx, off, {"action": "write", "file": "pkg/other.py",
                                     "content": "x = 2\n"}).startswith("WROTE")

    unarmed = _armed_cfg()      # on, but no gather ever armed the ledger
    assert tools.dispatch(idx, unarmed, {
        "action": "write", "file": "pkg/other.py",
        "content": "x = 3\n"}).startswith("WROTE")


def test_auto_allowlist_blocks_move_and_delete_of_unread_files(tmp_path):
    idx = _index(tmp_path, {"pkg/app.py": APP, "pkg/other.py": "x = 1\n"})
    cfg = _armed_cfg()
    cfg._task_reads = {"pkg/app.py": [(1, 5)]}
    cfg._task_allowlist = ()

    moved = tools.dispatch(idx, cfg, {"action": "move", "file": "pkg/other.py",
                                      "to": "pkg/moved.py"})
    assert moved.startswith("ERROR") and "task write allowlist" in moved
    assert (tmp_path / "pkg/other.py").is_file()

    deleted = tools.dispatch(idx, cfg, {"action": "delete",
                                        "file": "pkg/other.py"})
    assert deleted.startswith("ERROR") and "task write allowlist" in deleted
    assert (tmp_path / "pkg/other.py").is_file()

    ok = tools.dispatch(idx, cfg, {"action": "delete", "file": "pkg/app.py"})
    assert ok.startswith("DELETED pkg/app.py")
