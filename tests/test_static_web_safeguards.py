"""Static-web safeguards: three preventative guards and their regression tests.

1. webcheck flags HTML that wears Tailwind utility classes while loading
   neither the Play CDN nor a stylesheet that compiles them (the page would
   render completely unstyled).
2. A workspace with no package.json may not gain unrequested build-tool
   files (package.json / tailwind.config.* / postcss.config.*): the prose
   rule rides with the answer brief, and the apply path strips the
   creations deterministically.
3. A clean slate (every source file deleted) archives old handoffs, failed
   attempts and session listings instead of seeding the next task with a
   vanished project; memory lines whose files are gone are annotated stale.
   The GUI reset clears sessions, the consolidated handoff, memory AND the
   root PLAN.md, preserving source.
"""
# ruff: noqa: F811  (the `gui` fixture is re-exported from test_gui)
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_context import Scripted, _agent  # noqa: E402
from test_gui import gui  # noqa: F401,E402  (the shared server fixture)
from test_webcheck import _hit, _w  # noqa: E402

from pedacito import memory, prompts, sessions, verify, webcheck, workspace  # noqa: E402
from pedacito.config import Config  # noqa: E402

UTILITIES_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>t</title>
<link rel="stylesheet" href="styles.css"></head>
<body class="min-h-screen bg-[#141518] text-[#e9e7e2]">
<div class="max-w-5xl mx-auto flex-col border-white/10">hello</div>
</body></html>
"""

SEMANTIC_CSS = ("body { margin: 0; background: #141518; color: #e9e7e2; }\n"
                ".page { max-width: 64rem; margin-inline: auto; }\n")

COMPILED_CSS = r""".min-h-screen{min-height:100vh}
.bg-\[\#141518\]{background-color:#141518}
.text-\[\#e9e7e2\]{color:#e9e7e2}
.max-w-5xl{max-width:64rem}
.mx-auto{margin-left:auto;margin-right:auto}
.flex-col{flex-direction:column}
.border-white\/10{border-color:rgb(255 255 255 / .1)}
"""

EXACT_MESSAGE = (
    "HTML uses Tailwind utility classes (e.g. min-h-screen, bg-[#141518], "
    "text-[#e9e7e2]) but neither loads the Tailwind CDN script in <head> nor "
    "links a compiled stylesheet. Either write standard semantic CSS rules "
    'in styles.css or include <script src="https://cdn.tailwindcss.com">'
    "</script>.")


# --- 1. uncompiled Tailwind utilities -----------------------------------------

def test_webcheck_flags_uncompiled_tailwind_utilities(tmp_path):
    _w(tmp_path, "index.html", UTILITIES_HTML)
    _w(tmp_path, "styles.css", SEMANTIC_CSS)
    found = webcheck.run(tmp_path)
    assert _hit(found, "index.html", "Tailwind utility classes")
    f = next(f for f in found if "Tailwind utility classes" in f.message)
    assert f.message == EXACT_MESSAGE
    assert f.peer == "styles.css"      # editing either side surfaces it
    # A cluster of plain layout utilities (no arbitrary values) is enough.
    _w(tmp_path, "index.html",
       '<link rel="stylesheet" href="styles.css">\n'
       '<div class="min-h-screen max-w-5xl mx-auto flex-col">x</div>\n')
    assert _hit(webcheck.run(tmp_path), "index.html",
                "Tailwind utility classes")
    # A compiled sheet missing ONE utility still leaves the page partly
    # unstyled: the notice stays.
    _w(tmp_path, "index.html", UTILITIES_HTML)
    _w(tmp_path, "styles.css",
       COMPILED_CSS.replace(".mx-auto{margin-left:auto;margin-right:auto}\n",
                            ""))
    assert _hit(webcheck.run(tmp_path), "index.html",
                "Tailwind utility classes")


def test_webcheck_passes_semantic_css_cdn_or_compiled_sheet(tmp_path):
    # A semantic page wearing no utilities: nothing to flag.
    _w(tmp_path, "index.html",
       '<link rel="stylesheet" href="styles.css">\n'
       '<body class="page"><div class="page">hello</div></body>\n')
    _w(tmp_path, "styles.css", SEMANTIC_CSS)
    assert webcheck.run(tmp_path) == []
    # The Play CDN (v3 and v4) compiles the utilities at runtime.
    for src in ("https://cdn.tailwindcss.com",
                "https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"):
        _w(tmp_path, "index.html", UTILITIES_HTML.replace(
            '<link rel="stylesheet" href="styles.css">',
            f'<script src="{src}"></script>'))
        assert not _hit(webcheck.run(tmp_path), needle="Tailwind utility")
    # A real compiled stylesheet defines every utility the page wears.
    _w(tmp_path, "index.html", UTILITIES_HTML)
    _w(tmp_path, "styles.css", COMPILED_CSS)
    assert not _hit(webcheck.run(tmp_path), needle="Tailwind utility")
    # Compiled-in-<style> passes too.
    _w(tmp_path, "index.html", UTILITIES_HTML.replace(
        '<link rel="stylesheet" href="styles.css">',
        "<style>" + COMPILED_CSS + "</style>"))
    assert not _hit(webcheck.run(tmp_path), needle="Tailwind utility")


# --- 2. unrequested build-tool files on a static site --------------------------

def _creation(rel, text):
    return (f"--- FILE: {rel}\n<<<<<<< SEARCH\n\n=======\n{text}\n"
            ">>>>>>> REPLACE\n")


def test_apply_strips_unrequested_build_tool_creations(tmp_path):
    files = {"index.html": "<html><body>hi</body></html>\n",
             "styles.css": "body { margin: 0; }\n"}
    agent = _agent(tmp_path, Scripted(), files)
    agent.cfg._task_text = "make a sleek landing page"
    reply = (_creation("package.json", '{"name": "site"}\n')
             + _creation("tailwind.config.js", "module.exports = {}\n")
             + _creation("postcss.config.mjs", "export default {}\n")
             + "--- FILE: styles.css\n<<<<<<< SEARCH\nbody { margin: 0; }\n"
               "=======\nbody { margin: 0; background: #141518; }\n"
               ">>>>>>> REPLACE\n")
    res = agent._apply_once(reply, dry_run=False)
    by_file = {r.file: r for r in res}
    for name in ("package.json", "tailwind.config.js", "postcss.config.mjs"):
        assert not by_file[name].ok
        assert "refused" in by_file[name].message
        assert not (tmp_path / name).exists()
    # The legitimate stylesheet edit landed.
    assert by_file["styles.css"].ok
    assert "#141518" in (tmp_path / "styles.css").read_text(encoding="utf-8")
    # Dry runs refuse identically and write nothing.
    agent2 = _agent(tmp_path / "dry", Scripted(), files)
    agent2.cfg._task_text = "make a sleek landing page"
    dry = agent2._apply_once(_creation("package.json", "{}\n"), dry_run=True)
    assert not dry[0].ok and "refused" in dry[0].message
    assert not (tmp_path / "dry" / "package.json").exists()


def test_build_tool_guard_exempts_requested_tooling_and_existing_projects(
        tmp_path):
    files = {"index.html": "<html></html>\n"}
    # The task asks for npm: creations are a requested outcome.
    agent = _agent(tmp_path / "asked", Scripted(), files)
    agent.cfg._task_text = "set up npm and vite for the site"
    res = agent._apply_once(_creation("package.json", "{}\n"), dry_run=False)
    assert all(r.ok for r in res)
    assert (tmp_path / "asked" / "package.json").exists()
    # A workspace that already had a package.json: edits are not hijacks.
    agent2 = _agent(tmp_path / "existing", Scripted(),
                    dict(files, **{"package.json": '{"name": "x"}\n'}))
    agent2.cfg._task_text = "restyle the page"
    reply = ("--- FILE: package.json\n<<<<<<< SEARCH\n"
             '{"name": "x"}\n=======\n{"name": "y"}\n>>>>>>> REPLACE\n')
    res2 = agent2._apply_once(reply, dry_run=False)
    assert all(r.ok for r in res2)
    # The decision function itself: pre-existing keys, any-depth package.json.
    pre = {"index.html"}
    assert verify.unrequested_build_creations(
        pre, "restyle", ["package.json"]) == ["package.json"]
    assert verify.unrequested_build_creations(
        pre | {"site/package.json"}, "restyle", ["package.json"]) == []
    assert verify.unrequested_build_creations(
        pre, "add a package.json", ["package.json"]) == []


def test_static_site_rule_rides_with_the_answer_brief(tmp_path):
    cfg = Config()
    cfg._root = str(tmp_path)
    brief = prompts.answer_system(cfg, "EDIT FORMAT", "make a landing page")
    assert "Never create package.json" in brief
    assert "cdn.tailwindcss.com" in brief
    # A workspace with a package.json, or a task asking for tooling, skips it.
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    assert "Never create package.json" not in prompts.answer_system(
        cfg, "EDIT FORMAT", "make a landing page")
    (tmp_path / "package.json").unlink()
    assert "Never create package.json" not in prompts.answer_system(
        cfg, "EDIT FORMAT", "compile css with the tailwind cli")
    # The design skill carries the same rule.
    from pedacito import skills
    body = skills.BUILTIN_SKILLS["modern-web-design-ui"][1]
    assert "Never create package.json" in body


# --- 3. clean slate: stale handoffs, attempts and memory -----------------------

def _seed_history(root):
    state = workspace.state_dir(root)
    (state / "sessions").mkdir(parents=True, exist_ok=True)
    (state / "sessions" / "20260101-000000.md").write_text(
        "**Task:** build the old site\n"
        "- applied `index.html`\n- applied `styles.css`\n",
        encoding="utf-8")
    (state / "consolidated_session.md").write_text(
        "# Consolidated session handoff\n## Goal & Context\n"
        "- Current: build the old site\n", encoding="utf-8")
    memory.remember(root, "the hero lives in index.html", category="notes")
    memory.note_attempt(root, "build the old site", "no edit landed",
                        ["index.html"])


def test_clean_slate_archives_stale_handoffs_and_attempts(tmp_path):
    _seed_history(tmp_path)
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "styles.css").write_text("body{}", encoding="utf-8")
    # While the source lives, history is current and injectable.
    assert not sessions.clean_slate(tmp_path)
    assert "20260101" in sessions.recent(tmp_path)
    assert memory.attempts_note(tmp_path, "build the old site again")
    assert "[stale" not in memory.render(tmp_path)
    assert "build the old site" in memory.handoff_prompt(
        "build the old site", tmp_path)
    # Delete every source file: the project is a clean slate.
    (tmp_path / "index.html").unlink()
    (tmp_path / "styles.css").unlink()
    assert sessions.clean_slate(tmp_path)
    listing = sessions.recent(tmp_path)
    assert "Clean slate" in listing and "20260101" not in listing
    assert memory.attempts_note(tmp_path, "build the old site again") == ""
    handoff = memory.handoff_prompt("build the old site", tmp_path)
    assert "ARCHIVED" in handoff and "build the old site" not in handoff
    assert "[stale: the file(s) it names were deleted]" in memory.render(
        tmp_path)
    # Hidden state and vendored files do not count as source.
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden").write_text("x", encoding="utf-8")
    assert sessions.clean_slate(tmp_path)
    # One visible file back: no longer a clean slate, but a session whose
    # every named file is gone is annotated stale in the listing.
    (tmp_path / "README.md").write_text("hi", encoding="utf-8")
    assert not sessions.clean_slate(tmp_path)
    (workspace.state_dir(tmp_path) / "sessions" / "20260102-000000.md"
     ).write_text("**Task:** other\n- applied `gone.js`\n", encoding="utf-8")
    assert "[stale: every file it names was deleted]" in sessions.recent(
        tmp_path)


def test_reset_clears_plan_consolidated_sessions_and_memory(gui, tmp_path):
    from pedacito.gui import core
    root = tmp_path / "resetproj"
    root.mkdir()
    _seed_history(root)
    (root / "PLAN.md").write_text("- [ ] old task\n", encoding="utf-8")
    (root / "index.html").write_text("<html></html>", encoding="utf-8")
    state = workspace.state_dir(root)
    core.STATE.root = root
    # Preview reports the plan removal but deletes nothing.
    status, preview = gui.post("/api/project/reset", {"dry_run": True})
    assert status == 200 and preview["ok"] and preview["dry_run"]
    assert preview["plan_removed"] is True
    assert (root / "PLAN.md").exists()
    assert (state / "sessions" / "20260101-000000.md").exists()
    # The real reset clears sessions, handoff, memory and PLAN.md.
    status, done = gui.post("/api/project/reset", {})
    assert status == 200 and done["ok"], done
    assert done["plan_removed"] is True
    assert done["summary"] == ("Cleared session history, memory, and "
                               "PLAN.md — ready for a fresh start.")
    assert not (root / "PLAN.md").exists()
    assert not (state / "sessions").exists()
    assert not (state / "consolidated_session.md").exists()
    assert not (state / "memory.md").exists()
    assert (root / "index.html").exists()      # source preserved
    # keep_memory preserves memory.md and says so in the summary.
    (root / "PLAN.md").write_text("- [ ] again\n", encoding="utf-8")
    (state / "memory.md").write_text("# memory\n- a fact\n", encoding="utf-8")
    status, kept = gui.post("/api/project/reset", {"keep_memory": True})
    assert status == 200 and kept["ok"], kept
    assert (state / "memory.md").exists()
    assert not (root / "PLAN.md").exists()
    assert kept["summary"] == ("Cleared session history and PLAN.md — "
                               "ready for a fresh start.")
