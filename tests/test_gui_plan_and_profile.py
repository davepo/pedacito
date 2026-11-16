"""Regression tests for three GUI/plan flows and multi-clause HTML edits.

The GUI regression:
  * a profile switch is synchronous and never claims the runner slot (the Run
    button must not flip to Stop);
  * the composer draft lives in localStorage;
  * PLAN.md created while Auto runs is visible through /api/plan immediately,
    not only after the run ends.

The editing regression: every quoted literal in a multi-clause layout brief
is its own requirement (R1..Rn), and the coverage follow-up judges the literal
against the final edited HTML rather than stopping after R1.
"""
# The `gui` fixture is test_gui's server, imported for each test below.
# ruff: noqa: F811
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pedacito import cli, config as config_mod  # noqa: E402
from pedacito import edits, loop, verify  # noqa: E402
from pedacito.gui import core, jobs  # noqa: E402
from test_gui import gui  # noqa: F401  (the shared server fixture)


# --------------------------------------------------------------- helpers -----

def _default_config(gui):
    status, body = gui.post("/api/settings/setup", {
        "base_url": "http://127.0.0.1:1/v1", "model": "tiny", "vram_gb": "8"})
    assert status == 200 and body["ok"], body


def _open_project(gui, name="proj"):
    root = gui.tmp / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text("<h1>Jane</h1>\n", encoding="utf-8")
    status, body = gui.post("/api/project", {"path": str(root),
                                             "profile": ""})
    assert status == 202 and body["ok"], body
    assert gui.wait_idle(30)
    return root


def _fake_client(cfg):
    """The small slice of LLMClient run_auto/session touches."""
    return SimpleNamespace(
        cfg=cfg, stream_callback=None, echo_stream=False, site="",
        stats={"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
               "cost": 0.0, "seconds": 0.0},
        warmup=lambda: None, cancel=lambda: None)


# ----------------------------------------------------- profile switching -----

def test_profile_switch_never_claims_the_job_slot(gui, monkeypatch):
    _default_config(gui)
    assert gui.post("/api/settings/profile", {
        "file": "user", "action": "create", "name": "fast",
        "values": {"base_url": "http://127.0.0.1:1/v1", "model": "fast",
                   "vram_gb": 8, "tier": "small"}})[0] == 200
    _open_project(gui)

    spawned = []
    monkeypatch.setattr(jobs, "spawn",
                        lambda *a, **k: spawned.append((a, k)) or None)

    status, j = gui.post("/api/profile", {"profile": "fast"})
    assert status == 200 and j["ok"] is True, j
    assert j["profile"] == "fast" and j["model"] == "fast"
    # Synchronous: nothing was queued, and /api/state has no active job.
    assert spawned == []
    state = gui.get("/api/state")[1]
    assert state["job"] is None
    assert state["profile"] == "fast"

    # It stays true across the next poll (the Run button cannot become Stop).
    state = gui.get("/api/state")[1]
    assert state["job"] is None and state["profile"] == "fast"


def test_composer_draft_is_persisted_to_localstorage():
    src = (Path(__file__).resolve().parents[1] / "pedacito" / "gui"
           / "static" / "app.js").read_text(encoding="utf-8")
    assert "pedacito_draft_prompt" in src
    assert "localStorage.setItem(DRAFT_KEY, text)" in src
    assert "localStorage.removeItem(DRAFT_KEY)" in src
    assert "writeDraft(e.target.value)" in src        # oninput
    assert "clearDraft();" in src                     # submitted/cleared
    # The profile handler has a guaranteed non-busy exit.
    start = src.index("async function switchProfile")
    block = src[start:start + 1600]
    assert "setBusy(false)" in block
    assert "finally" in block


# ------------------------------------------- Auto replaces a completed plan ---

def _auto_args(root: Path, goal: str, *, apply: bool = False):
    return argparse.Namespace(
        task=goal, spec=None, plan=str(root / "PLAN.md"), apply=apply,
        verify=None, repair=None, max_tasks=None, max_minutes=None,
        keep_going=False, auto_install=False, force=False, refine=False,
        diagnostics=False, verify_final=False, strict=False, worktree="")


def test_new_goal_archives_completed_plan_and_generates_immediately(
        tmp_path, monkeypatch):
    root = tmp_path / "site"
    root.mkdir()
    plan = root / "PLAN.md"
    plan.write_text("# Plan\n\nGoal: old bio\n\n- [x] old task\n",
                    encoding="utf-8")
    cfg = config_mod.Config()
    cfg.auto_plan_file = "PLAN.md"
    emitted = []
    monkeypatch.setattr(loop, "generate_tasks",
                        lambda spec, client, cfg, root=None:
                        ["Restyle the bio page", "Polish the link badges"])
    monkeypatch.setattr(cli.roles_mod, "client_for",
                        lambda cfg, role, client: client)
    monkeypatch.setattr(loop, "emit_plan_updated",
                        lambda path, root=None, cfg=None, reason="updated":
                        emitted.append(reason))

    code = cli._cmd_auto(_auto_args(root, "Redesign the bio page with modern "
                                         "typography"), cfg, object(), None,
                         root)
    assert code == 0
    text = plan.read_text(encoding="utf-8")
    assert "Goal: Redesign the bio page" in text
    assert "- [ ] Restyle the bio page" in text
    assert "old task" not in text
    archived = list((root / ".pedacito" / "sessions").glob("*plan-PLAN.md"))
    assert archived and "old task" in archived[0].read_text(encoding="utf-8")
    assert "created" in emitted                       # emitted before Task 1


def test_api_plan_shows_plan_while_auto_is_still_running(gui, monkeypatch):
    _default_config(gui)
    root = _open_project(gui)
    (root / "PLAN.md").write_text(
        "# Plan\n\nGoal: old\n\n- [x] finished task\n", encoding="utf-8")
    with core.STATE.lock:
        core.STATE.client = _fake_client(core.STATE.cfg)
        core.STATE.idx = SimpleNamespace(files={}, chunks={})
        core.STATE.agent = SimpleNamespace(index=core.STATE.idx)

    release = threading.Event()
    written = threading.Event()

    def fake_auto(ns, cfg, client, idx, project_root):
        old = Path(project_root) / "PLAN.md"
        loop.archive_plan(old, project_root, reason="test")
        loop.write_plan(old, "new visual goal",
                        ["Restyle the hero", "Polish the cards"])
        loop.emit_plan_updated(old, root=project_root, cfg=cfg,
                               reason="created")
        written.set()
        release.wait(15)
        return 0

    monkeypatch.setattr(jobs.cli_mod, "_cmd_auto", fake_auto)
    status, body = gui.post("/api/run", {
        "mode": "auto", "text": "new visual goal", "plan_only": True})
    assert status == 202 and body["ok"], body
    assert written.wait(10)

    # While the job is still blocked, /api/plan already carries the new list.
    status, plan = gui.get("/api/plan")
    assert status == 200 and plan["exists"] is True
    assert "new visual goal" in plan["text"]
    assert [t["title"] for t in plan["tasks"]] == [
        "Restyle the hero", "Polish the cards"]
    state = gui.get("/api/state")[1]
    assert state["job"] is not None                  # run not finished yet
    assert state["plan"]["text"] == plan["text"]     # /api/state carries it too

    release.set()
    assert gui.wait_idle(30)
    assert list((root / ".pedacito" / "sessions").glob("*plan-PLAN.md"))


# ------------------------------------- multi-clause quoted HTML requirements --

BRIEF = (
    'Remove the "Our Mission" sub heading, move the GitHub and LinkedIn '
    'links under the <h1>, add a "Here\'s a bit about me..." heading, and '
    'replace the bio text with "I build thoughtful software."')

BEFORE = """<h1>Jane Doe</h1>
<h2>Our Mission</h2>
<p>Old generic bio text.</p>
<a href="https://github.com/jane">GitHub</a>
<a href="https://linkedin.com/in/jane">LinkedIn</a>
"""

AFTER_R1 = """<h1>Jane Doe</h1>
<p>Old generic bio text.</p>
<a href="https://github.com/jane">GitHub</a>
<a href="https://linkedin.com/in/jane">LinkedIn</a>
"""

AFTER_ALL = """<h1>Jane Doe</h1>
<h2>Here's a bit about me...</h2>
<p>I build thoughtful software.</p>
<a href="https://github.com/jane">GitHub</a>
<a href="https://linkedin.com/in/jane">LinkedIn</a>
"""


def test_every_quoted_clause_is_extracted_with_its_literal():
    reqs = verify._requirements_from_task(BRIEF)
    assert [r.req_id for r in reqs] == ["R1", "R2", "R3", "R4"]
    assert reqs[0].literals == ["Our Mission"] and reqs[0].literal_remove
    assert reqs[2].literals == ["Here's a bit about me..."]
    assert reqs[3].literals == ["I build thoughtful software."]
    # Nothing was merged away or clipped after the first period.
    assert "replace the bio text" in reqs[3].text.lower()


def test_coverage_followup_checks_literals_against_final_html():
    only_r1 = edits.Result(True, "index.html", "replaced contents",
                           before=BEFORE, after=AFTER_R1)
    seeds = verify.coverage_seeds(BRIEF, "Removed the subheading.", [only_r1])
    ids = {r.req_id for r in seeds}
    assert {"R3", "R4"} <= ids
    by_id = {r.req_id: r for r in seeds}
    assert by_id["R3"].missing_literals == ["Here's a bit about me..."]
    assert by_id["R4"].missing_literals == ["I build thoughtful software."]
    # R1 is satisfied (the literal is gone from the final file).
    assert "R1" not in ids

    complete = edits.Result(True, "index.html", "replaced contents",
                            before=BEFORE, after=AFTER_ALL)
    assert verify.coverage_seeds(BRIEF, "Done.", [complete]) == []

    # A removal literal still present in the final page is NOT accepted.
    still_there = edits.Result(True, "index.html", "replaced contents",
                               before=BEFORE, after=AFTER_ALL.replace(
                                   "<h1>Jane Doe</h1>",
                                   "<h1>Jane Doe</h1>\n<h2>Our Mission</h2>"))
    seeds = verify.coverage_seeds(BRIEF, "Done.", [still_there])
    assert any(r.req_id == "R1" and r.missing_literals == ["Our Mission"]
               for r in seeds)


def test_small_html_restyle_allows_a_whole_section_rewrite():
    before = "\n".join(f'<section class="old-{i}">row {i}</section>'
                       for i in range(40)) + "\n"
    after = ('<main class="hero"><h1>Jane</h1>'
             '<p>Here\'s a bit about me...</p></main>\n')
    assert edits.fragment_refusal(
        "index.html", before, after,
        task="restyle the landing page and move the social links") is None
    # The wipe guard still refuses an empty replacement.
    assert edits.fragment_refusal("index.html", before, " \n",
                                  task="restyle the page")
