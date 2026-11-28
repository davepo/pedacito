"""WP-4102C: the codemod CLI and the gated agent action.

The tests pin the external surface rather than the recipe engines:

* `pdc codemod list`, `preview` (also as the omitted default) and `apply`;
* `--option KEY=VALUE` is parsed as data, not shell text or subprocess;
* a recipe id with path separators and a file that escapes the project are
  refused before any read/write;
* preview/list are side-effect free (no backup, checkpoint or index build);
* apply needs allow_write AND the write policy, refuses protected, sensitive
  and task-allowlisted paths, records the mutation ledger, and commits one
  atomic transaction;
* the agent schema, inspector, prompt help and `how` index agree about the
  `codemod` action and its fields;
* tasks that never ask for a mechanical transformation keep the old schema
  and prompt byte-for-byte (no unconditional prompt growth).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pedacito.codemod as codemod_module
from pedacito import cli, config as config_mod, how_index, prompts, tools
from pedacito import index as index_mod
from pedacito.codemod import (list_recipes, parse_option_pairs,
                              parse_options_json)
from pedacito.config import Config


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path


def _cfg(root: Path, task: str = "add an import of os") -> Config:
    cfg = Config()
    cfg._root = root
    cfg._task_text = task
    cfg.log_events = False
    cfg.log_sessions = False
    return cfg


def _index(root: Path) -> SimpleNamespace:
    """A read-only-root stand-in; recipes that need more fall back to their
    deterministic text scan because the attribute methods are absent."""
    return SimpleNamespace(root=root)


def _run(capsys, argv: list[str]) -> tuple[int, str, str]:
    code = cli.main(["--lang", "en", *argv])
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An isolated project: no user config, no PEDACITO_* environment."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "USER_CONFIG", tmp_path / "missing.toml")
    for key in list(os.environ):
        if key.startswith("PEDACITO_"):
            monkeypatch.delenv(key, raising=False)
    return tmp_path


class _WriteDenyPolicy:
    """Allows reads, denies writes: exercises the apply-time gate alone."""

    def evaluate(self, request):
        allowed = not str(request.capability).startswith("write")

        class Decision:
            decision = "allow" if allowed else "deny"
            matched_rule = "test-read-ok" if allowed else "test-write-deny"
            reason = ("reads allowed" if allowed
                      else "writes are denied here")

        return Decision()


# ---------------------------------------------------------------------------
# option parsing: data, never commands
# ---------------------------------------------------------------------------
def test_parse_option_pairs_keeps_values_as_data():
    parsed = parse_option_pairs([
        "module=os",
        "default={'a': 1}",
        "annotation=dict[str, int]",
        "expression=a=b",
    ])
    assert parsed == {
        "module": "os",
        "default": "{'a': 1}",
        "annotation": "dict[str, int]",
        "expression": "a=b",
    }
    # No shell expansion or command execution: text stays text.
    assert ";" in parse_option_pairs(["name=os; rm -rf /"])["name"]


@pytest.mark.parametrize("bad", [
    "moduleos", "=os", "mo dule=os", "",
])
def test_parse_option_pairs_rejects_malformed_items(bad):
    with pytest.raises(ValueError):
        parse_option_pairs([bad])


def test_parse_options_json_is_strict_and_typed():
    assert parse_options_json("") == {}
    assert parse_options_json("{\"module\": \"os\"}") == {"module": "os"}
    assert parse_options_json({"module": "os"}) == {"module": "os"}
    for bad in ('{"module": 1}', '["os"]', "not-json", '{"": "os"}'):
        with pytest.raises(ValueError):
            parse_options_json(bad)


# ---------------------------------------------------------------------------
# CLI: list, preview (default), apply
# ---------------------------------------------------------------------------
def test_cli_list_prints_the_catalog(project, capsys):
    code, out, err = _run(capsys, ["codemod", "list", str(project)])
    assert code == 0, err
    assert "Codemod recipes (13)" in out
    for recipe in list_recipes():
        assert recipe["id"] in out
    assert "add_import" in out and "options:" in out


def test_cli_preview_is_the_default_and_writes_nothing(project, capsys):
    path = _write(project, "app.py", "x = 1\n")
    code, out, err = _run(capsys, [
        "codemod", "add_import", "--file", "app.py",
        "--option", "module=os", str(project),
    ])
    assert code == 0, err
    assert "Codemod preview" in out
    assert "+import os" in out
    assert "nothing was written" in out
    assert path.read_text(encoding="utf-8") == "x = 1\n"
    # A preview must not create a backup dir, checkpoint, index, journal or
    # any other .pedacito state.
    assert not (project / ".pedacito").exists()


def test_cli_preview_uses_repeatable_options(project, capsys):
    path = _write(project, "app.py", "x = 1\n")
    code, out, err = _run(capsys, [
        "codemod", "preview", "add_import", "--file", "app.py",
        "--option", "module=typing", "--option", "name=Any",
        str(project),
    ])
    assert code == 0, err
    assert "+from typing import Any" in out
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_cli_apply_requires_allow_write(project, capsys):
    path = _write(project, "app.py", "x = 1\n")
    code, out, err = _run(capsys, [
        "codemod", "apply", "add_import", "--file", "app.py",
        "--option", "module=os", str(project),
    ])
    assert code != 0
    assert "allow_write" in err
    assert path.read_text(encoding="utf-8") == "x = 1\n"
    assert not (project / ".pedacito").exists()


def test_cli_apply_writes_one_snapshotted_change(project, capsys):
    path = _write(project, "app.py", "x = 1\n")
    cfg_path = _write(project, "allowed.toml", "allow_write = true\n")
    code, out, err = _run(capsys, [
        "codemod", "apply", "add_import", "--file", "app.py",
        "--option", "module=os", "--config", str(cfg_path),
        str(project),
    ])
    assert code == 0, err
    assert "Codemod applied" in out
    assert path.read_text(encoding="utf-8") == "import os\n\nx = 1\n"
    assert list((project / ".pedacito" / "backups").glob("*/app.py"))


def test_cli_move_file_reindex_drops_deleted_source(project, capsys):
    _write(project, "pkg/__init__.py", "")
    _write(project, "pkg/core.py", "def helper():\n    return 1\n")
    _write(project, "pkg/user.py", "from pkg.core import helper\n\nhelper()\n")
    index_mod.build([str(project)], Config(), client=None, summarise=False,
                    root=project, verbose=False)
    cfg_path = _write(project, "allowed.toml", "allow_write = true\n")
    code, out, err = _run(capsys, [
        "codemod", "apply", "move_file", "--file", "pkg/core.py",
        "--option", "to_path=pkg/moved.py", "--config", str(cfg_path),
        str(project),
    ])
    assert code == 0, err
    assert "Codemod applied" in out
    assert "could not be re-indexed" not in err
    assert "[warn]" not in err
    live = index_mod.Index.load(project)
    assert live is not None
    assert "pkg/core.py" not in live.files
    assert "pkg/moved.py" in live.files


@pytest.mark.parametrize("bad,needle", [
    (["--option", "moduleos"], "not KEY=VALUE"),
    (["--option", "=os"], "empty key"),
    (["--option", "mo dule=os"], "whitespace"),
])
def test_cli_reports_bad_option_data(project, capsys, bad, needle):
    _write(project, "app.py", "x = 1\n")
    code, out, err = _run(capsys, [
        "codemod", "preview", "add_import", "--file", "app.py",
        *bad, str(project),
    ])
    assert code == 2
    assert needle in err
    assert (project / "app.py").read_text(encoding="utf-8") == "x = 1\n"


def test_cli_rejects_path_separators_in_recipe_ids(project, capsys):
    _write(project, "app.py", "x = 1\n")
    code, out, err = _run(capsys, [
        "codemod", "preview", "../add_import", "--file", "app.py",
        "--option", "module=os", str(project),
    ])
    assert code != 0
    assert "path separator" in err
    assert (project / "app.py").read_text(encoding="utf-8") == "x = 1\n"


def test_cli_rejects_file_path_escapes(project, capsys):
    outside = _write(project.parent, "outside.py", "x = 1\n")
    _write(project, "app.py", "x = 1\n")
    code, out, err = _run(capsys, [
        "codemod", "preview", "add_import", "--file", "../outside.py",
        "--option", "module=os", str(project),
    ])
    assert code != 0
    assert "traversal" in err or "escapes" in err
    assert outside.read_text(encoding="utf-8") == "x = 1\n"


# ---------------------------------------------------------------------------
# Agent action: list / preview / apply, gates and ledger
# ---------------------------------------------------------------------------
def test_tool_list_is_read_only(project):
    out = tools.tool_codemod(None, _cfg(project), "list")
    assert "CODEMOD RECIPES" in out
    for recipe in list_recipes():
        assert recipe["id"] in out
    assert not (project / ".pedacito").exists()


def test_tool_preview_is_side_effect_free(project):
    path = _write(project, "app.py", "x = 1\n")
    cfg = _cfg(project)
    out = tools.tool_codemod(
        _index(project), cfg, "preview", "add_import", "app.py", "",
        '{"module": "os"}')
    assert "CODEMOD PREVIEW" in out
    assert "nothing was written" in out
    assert path.read_text(encoding="utf-8") == "x = 1\n"
    assert getattr(cfg, "_task_mutations", []) == []
    assert not (project / ".pedacito").exists()


def test_tool_apply_is_denied_without_allow_write(project):
    path = _write(project, "app.py", "x = 1\n")
    cfg = _cfg(project)
    out = tools.tool_codemod(
        _index(project), cfg, "apply", "add_import", "app.py", "",
        '{"module": "os"}')
    assert out.startswith("ERROR:")
    assert "allow_write" in out
    assert path.read_text(encoding="utf-8") == "x = 1\n"
    assert not (project / ".pedacito").exists()


def test_tool_operation_defaults_to_preview_and_dispatch_routes_it(project):
    path = _write(project, "app.py", "x = 1\n")
    cfg = _cfg(project)
    index = SimpleNamespace(root=project)
    out = tools.dispatch(index, cfg, {
        "action": "codemod", "operation": "",
        "recipe": "add_import", "file": "app.py", "symbol": "",
        "options": '{"module": "os"}',
    })
    assert "CODEMOD PREVIEW" in out
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_tool_apply_writes_and_records_the_mutation(project):
    path = _write(project, "app.py", "x = 1\n")
    cfg = _cfg(project)
    cfg.allow_write = True
    out = tools.tool_codemod(
        _index(project), cfg, "apply", "add_import", "app.py", "",
        '{"module": "os"}')
    assert out.startswith("CODEMOD APPLIED")
    assert path.read_text(encoding="utf-8") == "import os\n\nx = 1\n"
    assert cfg._task_mutations == ["app.py"]
    assert list((project / ".pedacito" / "backups").glob("*/app.py"))


def test_apply_engine_enforces_a_write_denying_policy(project):
    path = _write(project, "app.py", "x = 1\n")
    cfg = _cfg(project)
    cfg.allow_write = True
    cfg.policy = _WriteDenyPolicy()
    result = codemod_module.apply(
        project, codemod_module.CodemodRequest(
            "add_import", "app.py", "", {"module": "os"}), cfg=cfg)
    assert not result.ok
    assert "denied by policy" in result.error
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_tool_apply_honours_the_write_policy_gate(project):
    path = _write(project, "app.py", "x = 1\n")
    cfg = _cfg(project)
    cfg.allow_write = True
    cfg.policy = _WriteDenyPolicy()
    out = tools.tool_codemod(
        _index(project), cfg, "apply", "add_import", "app.py", "",
        '{"module": "os"}')
    assert out.startswith("ERROR:")
    assert "denied by policy" in out
    assert "ERROR: ERROR:" not in out
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_tool_apply_honours_the_task_write_allowlist(project):
    path = _write(project, "app.py", "x = 1\n")
    cfg = _cfg(project)
    cfg.allow_write = True
    cfg.auto_allowlist = True
    cfg._task_reads = []
    out = tools.tool_codemod(
        _index(project), cfg, "apply", "add_import", "app.py", "",
        '{"module": "os"}')
    assert out.startswith("ERROR:")
    assert "allowlist" in out
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_tool_apply_uses_one_atomic_transaction_for_a_batch(
        project, monkeypatch):
    _write(project, "pkg/__init__.py", "")
    _write(project, "pkg/core.py", "def helper():\n    return 1\n")
    _write(project, "pkg/user.py",
           "from pkg.core import helper\n\nhelper()\n")
    cfg = _cfg(project, "move a symbol helper to pkg/util.py")
    cfg.allow_write = True
    idx = index_mod.build([str(project)], Config(), client=None,
                          summarise=False, root=project, verbose=False)
    calls = {"n": 0}
    real = codemod_module.apply_edits_atomic

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(codemod_module, "apply_edits_atomic", counted)
    out = tools.tool_codemod(
        idx, cfg, "apply", "move_symbol", "pkg/core.py",
        "helper", '{"to_file": "pkg/util.py"}')
    assert out.startswith("CODEMOD APPLIED"), out
    assert calls["n"] == 1
    assert not (project / "pkg/core.py").exists()
    assert "def helper" in (project / "pkg/util.py").read_text(encoding="utf-8")
    assert "from pkg.util import helper" in (
        project / "pkg/user.py").read_text(encoding="utf-8")


def test_tool_rejects_recipe_separators_and_file_escapes(project):
    outside = _write(project.parent, "outside.py", "x = 1\n")
    _write(project, "app.py", "x = 1\n")
    cfg = _cfg(project)
    cfg.allow_write = True
    separator = tools.tool_codemod(
        _index(project), cfg, "preview", "../add_import", "app.py", "",
        '{"module": "os"}')
    assert "path separator" in separator
    escaped = tools.tool_codemod(
        _index(project), cfg, "preview", "add_import", "../outside.py", "",
        '{"module": "os"}')
    assert "traversal" in escaped or "escapes" in escaped
    assert outside.read_text(encoding="utf-8") == "x = 1\n"


def test_tool_reports_bad_options_as_data_errors(project):
    _write(project, "app.py", "x = 1\n")
    cfg = _cfg(project)
    for bad, needle in (
            ("not-json", "not valid JSON"),
            ('{"module": 1}', "must be a string"),
    ):
        out = tools.tool_codemod(
            _index(project), cfg, "preview", "add_import", "app.py", "",
            bad)
        assert out.startswith("ERROR: codemod options"), out
        assert needle in out


def test_tool_rejects_an_unknown_operation(project):
    out = tools.tool_codemod(None, _cfg(project), "explode")
    assert "operation must be list, preview or apply" in out


# ---------------------------------------------------------------------------
# Schema, inspector, prompt help and how-index parity
# ---------------------------------------------------------------------------
def test_codemod_action_is_offered_only_when_the_task_asks(project):
    idle = _cfg(project, "fix the failing login test")
    schema = tools.schema_for(idle)
    assert "codemod" not in schema["properties"]["action"]["enum"]
    assert "recipe" not in schema["properties"]
    assert "operation" not in schema["properties"]
    assert "options" not in schema["properties"]
    assert "codemod" not in prompts.tool_help_block(idle, tools.TOOL_HELP)

    active = _cfg(project, "add an import of typing to app.py")
    schema = tools.schema_for(active)
    assert "codemod" in schema["properties"]["action"]["enum"]
    for field in ("recipe", "operation", "options"):
        assert field in schema["properties"]
        assert field in schema["required"]


def test_inspect_tools_and_schema_for_agree_about_codemod(project):
    cfg = _cfg(project, "extract a constant from app.py")
    rows = {name: reason for name, reason in tools._offer_rows(cfg)}
    assert rows.get("codemod") is None
    schema = tools.schema_for(cfg)
    assert "codemod" in schema["properties"]["action"]["enum"]
    inspected = {item.name: item
                 for item in tools.inspect_tools(cfg, None, cfg._task_text)}
    item = inspected["codemod"]
    assert item.active is True
    assert item.reason == "active"
    assert item.permission == "allow_write"
    assert "codemod" in tools.ACTION_PERMISSION


def test_prompt_help_and_how_index_cover_the_same_action(project):
    cfg = _cfg(project, "turn these prints into a logger")
    help_text = prompts.tool_help_block(cfg, tools.TOOL_HELP)
    assert "codemod" in help_text
    assert "operation" in help_text and "allow_write" in help_text
    cards = {card["name"]: card for card in how_index.CLI_COMMANDS}
    card = cards["codemod"]
    for word in ("list", "preview", "apply", "--file", "--option",
                 "allow_write"):
        assert word in card["usage"] + " " + " ".join(
            flag["flag"] + " " + flag["help"] for flag in card["flags"])
    hits = how_index.search("how do I preview a codemod recipe")
    assert "codemod" in [hit["name"] for hit in hits["cli"]]


def test_cli_parser_exposes_the_documented_codemod_grammar():
    parser = cli.build_parser("en")
    sub = next(action for action in parser._actions
               if action.__class__.__name__ == "_SubParsersAction")
    codemod_parser = sub.choices["codemod"]
    actions = next(action for action in codemod_parser._actions
                   if action.__class__.__name__ == "_SubParsersAction")
    assert set(actions.choices) == {"list", "preview", "apply"}
    for name in ("preview", "apply"):
        dests = {action.dest for action in actions.choices[name]._actions}
        assert {"recipe", "project", "file", "symbol", "option"} <= dests


def test_default_action_insertion_only_touches_the_codemod_command():
    assert cli._default_codemod_action(
        ["codemod", "add_import", "--file", "app.py"]) == [
            "codemod", "preview", "add_import", "--file", "app.py"]
    assert cli._default_codemod_action(
        ["--lang", "es", "codemod", "apply", "add_import"]) == [
            "--lang", "es", "codemod", "apply", "add_import"]
    assert cli._default_codemod_action(
        ["codemod", "list", "."]) == ["codemod", "list", "."]
    # A project directory or option value named "codemod" is data elsewhere.
    assert cli._default_codemod_action(
        ["index", "codemod"]) == ["index", "codemod"]
    assert cli._default_codemod_action(
        ["--config", "codemod", "ask", "."]) == [
            "--config", "codemod", "ask", "."]
