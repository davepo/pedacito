"""Documentation drift guards (Pass B).

The docs and the built-in `how` index must keep pace with the code:

* every CLI subcommand the parser accepts is mentioned in README and
  QUICKSTART (a new subcommand cannot ship undocumented);
* stale claims (the old "no CLI command for it yet" benchmark wording)
  stay gone;
* every bundled skill is valid and loadable, and the README names them;
* the flags and GUI features added with the security pass (--strings,
  config --upgrade, the masked Raw file tab, the MCP Servers tab, the
  Scout + Architect pairing form, the Project ▾ upkeep dialogs) are
  documented in the user-facing docs and in the how-cards;
* the specialist auto-routing pass (--auto-route, the ✦ Auto-route
  topbar toggle, the specialist-routing how-card) stays documented;
* the calibration/telemetry pass (calibrate --all, the hosted lockout,
  the architecture library, the telemetry ribbon and GUI footer) stays
  documented;
* the creative/design pass (--creative, /creative, the ✨ Creative
  composer toggle, the 🌐 Live Preview tab, hosted sampling omission)
  stays documented in both guides and in the how-cards.
* the docs ownership split (docs/ is canonical reference, the root guides
  are canonical walkthroughs, README indexes both) stays pinned in README,
  docs/ARCHITECTURE.md, each root guide and CHANGELOG.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import how_index, llm, profiler, skills
from pedacito.cli import build_parser
from pedacito.config import Config

ROOT = Path(__file__).resolve().parents[1]


def _readme_corpus() -> str:
    """The user-facing reference: README.md plus the modular guides in docs/.
    The README is a summary; the detailed reference lives in docs/."""
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    return "\n\n".join(p.read_text(encoding="utf-8") for p in paths)


def _doc_text(doc_name: str) -> str:
    if doc_name == "README.md":
        return _readme_corpus()
    return (ROOT / doc_name).read_text(encoding="utf-8")


README = _readme_corpus()
QUICKSTART = (ROOT / "QUICKSTART.md").read_text(encoding="utf-8")
MODEL_GUIDE = (ROOT / "MODEL_GUIDE.md").read_text(encoding="utf-8")

STALE_PHRASE = "There is no CLI command for it yet"

# The Project ▾ upkeep dialogs, exactly as the GUI labels them.
PROJECT_MENU_LABELS = (
    "Memory & repo lessons…",
    "Skills library & creator…",
    "Background servers…",
    "Run project checks…",
    "Prune old sessions/backups…",
    "Reset project state…",
)


def _subcommand_names() -> list[str]:
    parser = build_parser()
    sub = next(a for a in parser._actions
               if isinstance(a, argparse._SubParsersAction))
    return sorted(sub.choices)


def _mentioned(text: str, name: str) -> bool:
    """`pedacito NAME` anywhere, or a bare backticked `NAME` (the README
    groups some commands as `pedacito index`, `map`, `files`, `cost`)."""
    return (re.search(rf"\bpedacito\s+{re.escape(name)}\b", text) is not None
            or re.search(rf"`{re.escape(name)}[`|\s]", text) is not None)


def _norm(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("doc_name", ["README.md", "QUICKSTART.md"])
def test_every_cli_subcommand_is_documented(doc_name):
    text = _doc_text(doc_name)
    missing = [name for name in _subcommand_names()
               if not _mentioned(text, name)]
    assert not missing, f"{doc_name} does not mention: {missing}"


@pytest.mark.parametrize("rel", [
    "README.md", "QUICKSTART.md", "MODEL_GUIDE.md", "TECHNIQUES.md",
    "pedacito/how.py", "pedacito/how_index.py",
])
def test_stale_benchmark_phrase_is_gone(rel):
    assert STALE_PHRASE not in (ROOT / rel).read_text(encoding="utf-8")


def test_stale_benchmark_phrase_is_gone_from_gui_scripts():
    for path in sorted((ROOT / "pedacito" / "gui" / "static").glob("*.js")):
        assert STALE_PHRASE not in path.read_text(encoding="utf-8"), path


def test_hosted_probe_claim_matches_the_code(monkeypatch):
    """R2.1: "Probes never run for hosted endpoints" is a behavioral claim.

    The explicit paths force the probe, so pin the documented sentence to a
    stubbed LLMClient: a forced probe on a hosted URL must build no client at
    all, which means no request can be sent.
    """
    config_doc = _norm((ROOT / "docs" / "CONFIG.md").read_text(
        encoding="utf-8"))
    assert "Probes never run for hosted endpoints" in config_doc

    built: list[str] = []

    class RecordingClient:
        def __init__(self, cfg):
            built.append(str(cfg.base_url))
            raise AssertionError("hosted endpoint must not build a client")

    monkeypatch.setattr(llm, "LLMClient", RecordingClient)
    out = profiler.probe_model_speed(
        "https://api.deepseek.com/v1", "hosted-model", Config(),
        force=True, context_window=8192)
    assert built == []
    assert out["ok"] is False
    assert out.get("refused") is True
    assert "hosted" in str(out.get("detail") or "").lower()
    # The automatic post-load path makes the same promise.
    assert profiler.maybe_probe_async(
        Config(), base_url="https://api.deepseek.com/v1",
        model="hosted-model") is None
    assert built == []


def test_bundled_skills_are_valid_and_loadable(tmp_path):
    assert len(skills.BUNDLED) == 19
    for name, (desc, body, triggers) in skills.BUNDLED.items():
        assert skills._NAME_RE.match(name), name
        assert desc.strip(), name
        assert body.strip(), name
        assert len(body) <= skills.MAX_SKILL_CHARS, \
            f"{name} would be truncated by load()"
        assert triggers and all(isinstance(t, str) and t.strip()
                                for t in triggers), name
        text, info = skills.load(tmp_path, None, name)
        assert info is not None and info["source"] == "bundled"
        assert text.startswith(f"SKILL {name} (bundled):")
        assert body.strip() in text
    # An empty project still sees the whole bundled catalog.
    catalog = skills.catalog(tmp_path, None, limit=10000)
    for name in skills.BUNDLED:
        assert f"- {name}:" in catalog


def test_second_eight_bodies_stay_crisp():
    """The domain checklists added in the 8->16 pass, and later bundled
    additions, keep the <=550-char body budget, and every built-in declares
    usable triggers."""
    second = ("fastapi-rest-endpoint", "canvas-svg-2d-3d",
              "sqlite-sql-migrations", "async-concurrency-safety",
              "security-ssrf-xss-hardening", "i18n-localization-catalog",
              "parser-lexer-state-machine", "performance-memory-profiling")
    for name in second + ("safe-git-commit",):
        desc, body, triggers = skills.BUNDLED[name]
        assert len(body) <= skills.MAX_BUILTIN_BODY_CHARS, \
            f"{name}: {len(body)} chars"
        assert len(triggers) >= 4, name


def test_readme_names_every_bundled_skill():
    assert "Nineteen starters" in README
    for name in skills.BUNDLED:
        assert f"`{name}`" in README, f"README does not name skill {name}"


# (doc, needle) pairs for the features synced in the security pass.
# Needles are matched on whitespace-normalized text so line wraps in the
# markdown do not matter.
FEATURE_NEEDLES = [
    ("--strings", "rename --strings"),
    ("--include-labels", "rename --include-labels"),
    ("superseded stopped", "procs collapsed restart history"),
    ("--upgrade", "config --upgrade"),
    ("--dry-run", "config --upgrade --dry-run"),
    ("***REDACTED***", "masked Raw file tab"),
    ("MCP Servers", "Settings MCP Servers tab"),
    ("Dual-GPU / Scout + Architect Pairing", "Profiles pairing form"),
    ("Create starter Scout & pair…", "Auto-Tune dual-GPU pairing"),
    ("--auto-route", "specialist auto-routing flag"),
    ("✦ Auto-route", "Auto-route topbar toggle"),
    ("calibrate --all", "calibrate --all batch"),
    ("Calibrate All Local Models", "Calibrate All menu item"),
    ("provider-table defaults", "hosted calibration lockout"),
    ("six micro-probes", "the six specialty micro-probes"),
    ("show_telemetry", "telemetry ribbon setting"),
    ("skills add", "custom skill creation CLI"),
    ("skills draft", "model-drafted skill CLI"),
    ("✦ Auto-Draft from Description", "GUI skill auto-draft button"),
    ("--consolidate", "memory --consolidate flag"),
    ("/consolidate", "chat consolidation slash command"),
    ("Consolidate Session", "GUI consolidation button"),
    ("consolidated_session.md", "consolidation checkpoint file"),
    ("--creative", "creative mode CLI flag"),
    ("/creative", "TUI creative slash command"),
    ("creative_mode", "creative mode setting"),
    ("creative_auto", "creative auto-detect setting"),
    ("✨ Creative", "GUI creative toggle"),
    ("Senior Product Engineer & Visual Designer", "the Builder persona"),
    ("🌐 Preview", "GUI Live Preview tab"),
    ("Live Preview", "the Live Preview feature"),
    ("send_temperature_to_hosted", "hosted temperature opt-in"),
    ("hosted_temperature", "hosted temperature pin"),
    ("presence_penalty", "the local repetition nudge"),
    ("check_for_updates", "the update opt-out setting"),
    ("/update", "TUI update slash command"),
    ("update --run", "CLI update --run"),
    ("--lang", "CLI language flag"),
    ("/lang", "TUI language slash command"),
    ("EN | ES", "GUI language switcher"),
    ("llm_reply_language", "model reply language setting"),
] + [(label, "Project ▾ upkeep dialog") for label in PROJECT_MENU_LABELS]


@pytest.mark.parametrize("doc_name", ["README.md", "QUICKSTART.md"])
@pytest.mark.parametrize("needle,feature", FEATURE_NEEDLES)
def test_new_features_are_documented(doc_name, needle, feature):
    text = _norm(_doc_text(doc_name))
    assert needle in text, f"{doc_name} missing {feature}: {needle!r}"


@pytest.mark.parametrize("needle", [
    "Dual-GPU / Scout + Architect Pairing",
    "Create starter Scout & pair…",
])
def test_model_guide_documents_the_pairing_paths(needle):
    assert needle in _norm(MODEL_GUIDE)


@pytest.mark.parametrize("needle", [
    "edit_format >= 0.65",
    "4B limits",
    "does not reserve VRAM",
])
def test_model_guide_documents_the_hardening_notes(needle):
    assert needle in _norm(MODEL_GUIDE)


def test_readme_fetch_row_names_the_loopback_allowlist():
    assert 'fetch_allowed_hosts = ["127.0.0.1", "localhost"]' in _norm(README)


def test_how_cards_cover_the_new_features():
    cards = {c["name"]: c for c in how_index.CLI_COMMANDS}
    rename = cards["rename"]
    assert "--strings" in rename["usage"]
    assert any(f["flag"] == "--strings" for f in rename["flags"])
    assert "--include-labels" in rename["usage"]
    assert any(f["flag"] == "--include-labels" for f in rename["flags"])
    procs = cards["procs"]
    assert "--all" in procs["usage"]
    assert any(f["flag"] == "--all" for f in procs["flags"])

    topics = {t["id"]: t for t in how_index.GUI_TOPICS}
    mcp_steps = " ".join(topics["named-mcp-servers"]["steps"])
    assert "MCP Servers" in mcp_steps
    scout_steps = " ".join(topics["scout-model"]["steps"])
    assert "Pairing" in scout_steps
    assert "Auto-Tune" in scout_steps
    upkeep_steps = " ".join(topics["project-upkeep"]["steps"])
    for label in PROJECT_MENU_LABELS:
        assert label in upkeep_steps
    gather = topics["run-test-gather"]["summary"]
    assert "Vitest" in gather and "cargo" in gather
    key_steps = " ".join(topics["api-key-hosted"]["steps"])
    assert "***REDACTED***" in key_steps
    specialist = topics["specialist-routing"]
    spec_steps = " ".join(specialist["steps"])
    assert "✦ Auto-route" in spec_steps
    assert "--auto-route" in spec_steps
    assert "[specialty: TAG]" in spec_steps
    assert "auto_route_specialties" in specialist["settings"]
    assert "specialties" in specialist["settings"]
    cal = cards["calibrate"]
    assert "--all" in cal["usage"]
    assert any(f["flag"] == "--all" for f in cal["flags"])
    assert any(f["flag"] == "-y, --yes" for f in cal["flags"])
    cal_steps = " ".join(topics["calibrate-model"]["steps"])
    assert "Calibrate All Local Models" in cal_steps
    assert "provider-table defaults" in topics["calibrate-model"]["summary"]
    tele = topics["live-telemetry"]
    assert "show_telemetry" in tele["settings"]
    assert "█" in " ".join(tele["steps"])      # the ribbon example
    # The skills & consolidation pass: CLI cards and GUI topics stay synced.
    sk = cards["skills"]
    for act in ("list", "show", "add", "draft", "rm"):
        assert act in sk["usage"], act
    assert any(f["flag"] == "--global" for f in sk["flags"])
    assert any(f["flag"] == "--triggers K1,K2,..." for f in sk["flags"])
    mem = cards["memory"]
    assert "--consolidate" in mem["usage"]
    assert any(f["flag"] == "--consolidate" for f in mem["flags"])
    chat_notes = " ".join(cards["chat"]["notes"])
    assert "/consolidate" in chat_notes and "/compact" in chat_notes
    assert "consolidate" in cards["rpc"]["summary"]
    lib = topics["skills-library"]
    lib_steps = " ".join(lib["steps"])
    assert "Skills library & creator…" in lib_steps
    assert "✦ Auto-Draft from Description" in lib_steps
    assert "use_skills" in lib["settings"]
    con = topics["consolidate-session"]
    con_steps = " ".join(con["steps"])
    assert "Consolidate Session" in con_steps
    assert "Goal & Context" in con_steps
    assert "/compact" in con_steps
    # The creative/design + hosted-sampling pass: flags, cards and topics.
    assert any(f["flag"] == "--creative" for f in how_index.COMMON_FLAGS)
    assert any(f["flag"] == "--creative" for f in cards["new"]["flags"])
    assert any(f["flag"] == "--creative" for f in cards["tui"]["flags"])
    assert "/creative" in " ".join(cards["tui"]["notes"])
    assert "--creative" in " ".join(cards["chat"]["notes"])
    creative = topics["creative-mode"]
    cre_steps = " ".join(creative["steps"])
    assert "✨ Creative" in cre_steps
    assert "/creative" in cre_steps
    assert "--creative" in cre_steps
    assert "creative_mode" in creative["settings"]
    assert "creative_auto" in creative["settings"]
    preview = topics["live-preview"]
    pre_steps = " ".join(preview["steps"])
    assert "🌐 Preview" in pre_steps
    assert "Refresh" in pre_steps
    assert preview["cli"] == ["gui"]
    hosted = topics["hosted-sampling"]
    assert "send_temperature_to_hosted" in hosted["settings"]
    assert "hosted_temperature" in hosted["settings"]
    assert "presence_penalty" in hosted["summary"] + " ".join(hosted["steps"])
    # The context-aware auto-update pass: CLI card, GUI topic, TUI notes.
    upd = cards["update"]
    assert "--run" in upd["usage"]
    assert any(f["flag"] == "--run" for f in upd["flags"])
    assert "check_for_updates" in " ".join(upd["notes"])
    assert "/update" in " ".join(cards["tui"]["notes"])
    upd_topic = topics["update-pedacito"]
    assert "check_for_updates" in upd_topic["settings"]
    assert upd_topic["cli"] == ["update"]
    assert "Restart Pedacito" in " ".join(upd_topic["steps"])
    # The bilingual UI pass: flag, TUI notes and the language-switch topic.
    assert any(f["flag"].startswith("--lang") for f in how_index.COMMON_FLAGS)
    assert any(f["flag"].startswith("--lang") for f in cards["gui"]["flags"])
    tui_notes = " ".join(cards["tui"]["notes"])
    assert "/lang" in tui_notes and "F3" in tui_notes
    lang_topic = topics["language-switch"]
    lang_steps = " ".join(lang_topic["steps"])
    for needle in ("EN | ES", "F3", "/lang", "--lang", "PEDACITO_LANG"):
        assert needle in lang_steps, needle
    assert "language" in lang_topic["settings"]
    assert "llm_reply_language" in lang_topic["settings"]


def test_how_topic_references_resolve():
    """Every GUI topic's cli/settings cross-references must exist."""
    cli_names = {c["name"] for c in how_index.CLI_COMMANDS}
    setting_keys = {s["key"] for s in how_index.settings_dictionary()}
    for topic in how_index.GUI_TOPICS:
        for name in topic.get("cli", []):
            assert name in cli_names, (topic["id"], name)
        for key in topic.get("settings", []):
            assert key in setting_keys, (topic["id"], key)


@pytest.mark.parametrize("question,topic", [
    ("how do I add an mcp server in the gui?", "named-mcp-servers"),
    ("how do I pair a scout with the architect?", "scout-model"),
    ("how do I reset the project state?", "project-upkeep"),
    ("how do I route tasks to specialist profiles?", "specialist-routing"),
    ("how do I see tokens per second in the gui?", "live-telemetry"),
    ("how do I calibrate all my local models?", "calibrate-model"),
    ("how do I create a custom skill?", "skills-library"),
    ("how do I consolidate a long session?", "consolidate-session"),
    ("how do I turn on creative mode for design tasks?", "creative-mode"),
    ("how do I preview my html page in the gui?", "live-preview"),
    ("why does pedacito not send temperature to my hosted api?",
     "hosted-sampling"),
    ("how do I update pedacito to a newer release?", "update-pedacito"),
    ("how do I switch the interface language to spanish?", "language-switch"),
])
def test_new_topics_are_retrievable(question, topic):
    ids = [m["id"] for m in how_index.search(question)["gui"]]
    assert topic in ids, (question, ids)


# --- preview isolation (R4.3) ------------------------------------------------
# Project pages render inside the GUI origin but must not script it. Both
# preview iframes are sandboxed without `allow-same-origin`; the preview CSP is
# eval-free while the GUI's own page CSP stays as it was.

GUI_STATIC = ROOT / "pedacito" / "gui" / "static"
GUI_SERVER = ROOT / "pedacito" / "gui" / "server.py"


def test_both_preview_iframes_are_script_sandboxed():
    js = (GUI_STATIC / "app.js").read_text(encoding="utf-8")
    tags = re.findall(r"<iframe\b[^>]*>", js, flags=re.S)
    previews = [tag for tag in tags
                if "live-preview-frame" in tag or "runner-frame" in tag]
    assert len(previews) == 2, previews
    for tag in previews:
        assert 'sandbox="allow-scripts"' in tag, tag
        # allow-same-origin would cancel the isolation: the preview would be
        # back in the GUI origin and could drive the agent API.
        assert "allow-same-origin" not in tag, tag
    # New tab remains the deliberate full-origin escape hatch.
    assert "onClick=${openPreviewTab}" in js
    assert "onClick=${openRunnerPreview}" in js


def test_preview_csp_is_eval_free_and_the_gui_csp_is_unchanged():
    src = GUI_SERVER.read_text(encoding="utf-8")
    preview = re.search(r"^PREVIEW_CSP = \((.*?)\)\n", src,
                        flags=re.M | re.S)
    assert preview, "PREVIEW_CSP is missing from gui/server.py"
    block = preview.group(1)
    assert "unsafe-eval" not in block
    assert "script-src 'self' 'unsafe-inline'" in block
    # Project assets are served through the preview constant, not their own
    # inline policy that could drift back to eval.
    assert 'send_header("Content-Security-Policy", PREVIEW_CSP)' in src
    gui = re.search(r"^CSP = \((.*?)\)\n", src, flags=re.M | re.S)
    assert gui, "the GUI's own CSP is missing"
    assert "unsafe-eval" not in gui.group(1)
    assert "script-src 'self'" in gui.group(1)


def test_lemonade_pinning_and_effort_spelling_are_documented():
    """The pin surface and the `default`/`off` reconciliation stay in the
    user-facing docs, not only in the plan file."""
    config = _norm((ROOT / "docs" / "CONFIG.md").read_text(encoding="utf-8"))
    guide = _norm((ROOT / "MODEL_GUIDE.md").read_text(encoding="utf-8"))
    for text in (config, guide):
        assert "internal/pin" in text
        assert "/servers pin" in text
        assert "shipped spelling of the plan's `off`" in text


# --- project website and contact ---------------------------------------------
# The website and contact address are interface furniture and docs, never
# prompt or telemetry content. Pin the places that name them so a doc or GUI
# edit cannot silently drop the only private reporting address.

def test_security_reporting_names_the_contact_address():
    text = (ROOT / "docs" / "SECURITY.md").read_text(encoding="utf-8")
    section = text.split("## Reporting a vulnerability", 1)[1]
    section = section.split("\n## ", 1)[0]
    assert "contact@pedacito.dev" in section
    # The private-report path stays explicit: email or advisory, not a public
    # issue.
    assert "private security advisory" in section
    assert "public issue" in section


def test_trajectory_flag_is_documented_as_not_trust_gated():
    """record_trajectories is a local-only logging flag (like log_sessions),
    not an EXEC_FIELDS setting: SECURITY.md must not tell users to run
    `pedacito trust` before a project file can set it, and the trust list
    must not grow to include it."""
    from pedacito.config import EXEC_FIELDS
    assert "record_trajectories" not in EXEC_FIELDS
    text = (ROOT / "docs" / "SECURITY.md").read_text(encoding="utf-8")
    section = text.split("## Trajectory export", 1)[1].split("\n## ", 1)[0]
    assert "after `pedacito trust`" not in section
    assert "without `pedacito trust`" in section
    trust = text.split("### Project config is not trusted by default",
                       1)[1].split("\n### ", 1)[0]
    assert "record_trajectories" not in trust


def test_readme_names_the_project_website_and_contact():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://www.pedacito.dev" in readme
    assert "contact@pedacito.dev" in readme


def test_settings_dialog_links_to_the_project_website_and_contact():
    js = (GUI_STATIC / "settings.js").read_text(encoding="utf-8")
    assert "https://www.pedacito.dev" in js
    assert "mailto:contact@pedacito.dev" in js
    assert "rel: 'noopener'" in js
    assert "target: '_blank'" in js


def test_tui_help_names_the_project_website_and_contact():
    from pedacito import tui
    assert ("Web: https://www.pedacito.dev \u00b7 "
            "Contact: contact@pedacito.dev") in tui.HELP_TEXT


# --- docs ownership (R5.5) ----------------------------------------------------
# docs/ is the canonical reference, the three root guides are the canonical
# task-oriented guides, README is the index and CHANGELOG is the history. Pin
# the links both ways and the shared decision paragraph so neither set can
# silently drift into a copy of the other.

ROOT_GUIDES = ("QUICKSTART.md", "MODEL_GUIDE.md", "TECHNIQUES.md")
DOCS_SET = ("INSTALLATION", "ARCHITECTURE", "USAGE", "CONFIG",
            "SKILLS_AND_TOOLS", "SECURITY", "CONTRIBUTING")


def _ownership_paragraph(text: str) -> str:
    start = text.index("**Docs ownership.**")
    return _norm(text[start:].split("\n\n", 1)[0]).replace("](../", "](")


@pytest.mark.parametrize("guide", ROOT_GUIDES)
def test_root_guide_links_into_docs(guide):
    text = (ROOT / guide).read_text(encoding="utf-8")
    linked = re.findall(r"\]\((docs/[A-Za-z]+\.md)", text)
    assert linked, f"{guide} links into no docs/ page"
    for rel in linked:
        assert (ROOT / rel).is_file(), f"{guide} links to missing {rel}"


def test_readme_indexes_the_root_guides_and_docs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for guide in ROOT_GUIDES:
        assert f"]({guide})" in readme, f"README does not link {guide}"
    for name in DOCS_SET:
        assert f"](docs/{name}.md" in readme, f"README does not link {name}"


def test_ownership_decision_is_echoed_in_architecture():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "**Docs ownership.**" in readme
    assert _ownership_paragraph(readme) == _ownership_paragraph(arch)


def test_changelog_records_the_ownership_split_under_0_3_1():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    section = text.split("## v0.3.1", 1)[1].split("\n## ", 1)[0]
    assert "Docs Ownership" in section
    assert "canonical" in section


# --- CLI-at-a-glance + install/teering claims (WP-12) -------------------------
# docs/USAGE.md's table is the canonical index of subcommands; docs/INSTALLATION
# and docs/CONFIG.md must not drift from the behaviour the code actually ships.

def test_usage_at_a_glance_table_lists_every_cli_subcommand():
    """Every parser subcommand is named in the `### The CLI at a glance` table,
    so a new subcommand cannot ship without its one-line reference there."""
    text = (ROOT / "docs" / "USAGE.md").read_text(encoding="utf-8")
    section = text.split("### The CLI at a glance", 1)[1].split("\n## ", 1)[0]
    missing = [name for name in _subcommand_names()
               if not _mentioned(section, name)]
    assert not missing, f"USAGE.md CLI table does not mention: {missing}"


def test_installation_no_longer_claims_nvidia_vram_counter_reads():
    """The install guide once described reading NVIDIA VRAM counters; that
    claim was removed because telemetry stays local and quiet."""
    text = (ROOT / "docs" / "INSTALLATION.md").read_text(encoding="utf-8")
    assert "does not read NVIDIA VRAM counters" not in text


def test_config_names_the_local_moe_tier():
    text = (ROOT / "docs" / "CONFIG.md").read_text(encoding="utf-8")
    assert "local-moe" in text
