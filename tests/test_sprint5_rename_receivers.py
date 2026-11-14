"""Rename upgrades: receiver-aware qualified renames, and the
transactional guarantees -- a write failure anywhere in a multi-file batch
restores every file this call touched (no half-applied tree)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_agent_context import Scripted, _agent

from pedacito import edits, refactor
from pedacito.edits import apply_edits, parse_edits
from pedacito.llm import LLMError  # noqa: F401  (parity with other suites)

CLS_SRC = {
    "pkg/__init__.py": "",
    "pkg/store.py": (
        "class Store:\n"
        "    def save(self, item):\n"
        "        return item\n"
        "\n"
        "    def flush(self):\n"
        "        return self.save('all')\n"
        "\n"
        "\n"
        "class Other:\n"
        "    def save(self, item):\n"
        "        return item\n"
    ),
    "pkg/app.py": (
        "from pkg.store import Other, Store\n"
        "\n"
        "\n"
        "def run():\n"
        "    x = Store()\n"
        "    y = Other()\n"
        "    a = x.save(1)\n"
        "    b = y.save(2)\n"
        "    s = Store.save\n"
        "    return a, b, s\n"
        "\n"
        "\n"
        "def use2(s: Store):\n"
        "    return s.save(8)\n"
        "\n"
        "\n"
        "def use(z):\n"
        "    return z.save(9)\n"
    ),
    # a same-named function the agent never imported from Store: the bare
    # name is ambiguous, so `save` alone must refuse...
    "pkg/helpers.py": "def save(blob):\n    return blob\n",
}


def _cfg():
    from pedacito.config import Config
    cfg = Config()
    cfg.allow_write = True
    cfg.backup = False
    cfg.log_events = False
    return cfg


# ------------------------------------------------------ receiver awareness

def test_qualified_rename_follows_bound_receivers(tmp_path):
    agent = _agent(tmp_path, Scripted(), CLS_SRC, allow_write=True)
    out = refactor.tool_rename(agent.index, agent.cfg, "Store.save", "persist")
    assert out.startswith("RENAMED")

    store = (tmp_path / "pkg/store.py").read_text(encoding="utf-8")
    assert "def persist(self" in store and "self.persist('all')" in store
    assert "class Other:\n    def save(self" in store        # foreign def kept

    app = (tmp_path / "pkg/app.py").read_text(encoding="utf-8")
    assert "x.persist(1)" in app                            # ctor-bound receiver
    assert "Store.persist" in app                            # static access
    assert "s.persist(8)" in app                             # annotated param
    assert "y.save(2)" in app                                # typed to Other
    assert "z.save(9)" in app                                # untyped receiver
    assert "NOT renamed (ambiguous or foreign receivers)" in out
    assert "untyped receiver" in out
    assert "another `save` definition (Other.save)" in out
    assert "def persist(blob)" not in (tmp_path / "pkg/helpers.py").read_text(encoding="utf-8")
    assert "def save(blob)" in (tmp_path / "pkg/helpers.py").read_text(encoding="utf-8")


def test_qualified_rename_is_the_only_way_past_the_ambiguity(tmp_path):
    agent = _agent(tmp_path, Scripted(), CLS_SRC, allow_write=True)
    bare = refactor.tool_rename(agent.index, agent.cfg, "save", "persist")
    assert bare.startswith("ERROR") and "defined in" in bare
    assert (tmp_path / "pkg/store.py").read_text(encoding="utf-8") == CLS_SRC["pkg/store.py"]


def test_qualified_rename_accepts_module_prefix(tmp_path):
    agent = _agent(tmp_path, Scripted(), CLS_SRC, allow_write=True)
    out = refactor.tool_rename(agent.index, agent.cfg, "pkg.Store.save",
                               "persist")
    assert out.startswith("RENAMED")
    assert "def persist(self" in (tmp_path / "pkg/store.py").read_text(encoding="utf-8")


def test_qualified_rename_of_the_other_class_leaves_store_alone(tmp_path):
    agent = _agent(tmp_path, Scripted(), CLS_SRC, allow_write=True)
    out = refactor.tool_rename(agent.index, agent.cfg, "Other.save", "keep")
    assert out.startswith("RENAMED")
    app = (tmp_path / "pkg/app.py").read_text(encoding="utf-8")
    assert "y.keep(2)" in app and "x.save(1)" in app
    store = (tmp_path / "pkg/store.py").read_text(encoding="utf-8")
    assert "class Other:\n    def keep(self" in store
    assert "def save(self" in store        # Store.save untouched


def test_qualified_rename_unknown_class_is_reported(tmp_path):
    agent = _agent(tmp_path, Scripted(), CLS_SRC, allow_write=True)
    out = refactor.tool_rename(agent.index, agent.cfg, "Nope.save", "persist")
    assert out.startswith("ERROR") and "the index has no `Nope.save`" in out


# --------------------------------------- transactional multi-file rollback

def test_engine_write_failure_restores_every_written_file(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def apply_discount(p):\n    return p\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("from a import apply_discount\n"
                                   "X = apply_discount(1)\n", encoding="utf-8")
    real = refactor.fileio.write

    def boom(path, text, style=None):
        if Path(path).name == "b.py":
            raise OSError(28, "No space left on device")
        return real(path, text, style)

    monkeypatch.setattr(refactor.fileio, "write", boom)
    res = refactor.rename_symbol_across_project(tmp_path, "apply_discount",
                                                "discounted")
    assert not res["ok"]
    assert "write failed mid-rename" in res["error"]
    assert "restored 1 of 1" in res["error"]
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == \
        "def apply_discount(p):\n    return p\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == \
        "from a import apply_discount\nX = apply_discount(1)\n"


def test_tool_rename_write_failure_rolls_back_written_files(tmp_path,
                                                            monkeypatch):
    files = {
        "pkg/__init__.py": "",
        "pkg/core.py": "def apply_discount(price):\n    return price * 0.9\n",
        "pkg/shop.py": ("from pkg.core import apply_discount\n\n\n"
                        "def checkout(p):\n    return apply_discount(p)\n"),
    }
    agent = _agent(tmp_path, Scripted(), files, allow_write=True)
    real = refactor.fileio.write

    def boom(path, text, style=None):
        if Path(path).name == "shop.py":
            raise OSError(13, "Permission denied")
        return real(path, text, style)

    monkeypatch.setattr(refactor.fileio, "write", boom)
    out = refactor.tool_rename(agent.index, agent.cfg, "apply_discount", "d")
    assert "rename aborted" in out and "rolled back 1" in out
    assert (tmp_path / "pkg/core.py").read_text(encoding="utf-8") == files["pkg/core.py"]
    assert (tmp_path / "pkg/shop.py").read_text(encoding="utf-8") == files["pkg/shop.py"]


def test_snapshot_failure_aborts_before_writing_anything(tmp_path, monkeypatch):
    files = {
        "pkg/__init__.py": "",
        "pkg/core.py": "def apply_discount(price):\n    return price\n",
        "pkg/shop.py": ("from pkg.core import apply_discount\n\n"
                        "Y = apply_discount(2)\n"),
    }
    agent = _agent(tmp_path, Scripted(), files, allow_write=True)
    from pedacito import tools
    monkeypatch.setattr(tools, "_snapshot",
                        lambda *a, **k: "cannot snapshot (disk locked)")
    out = refactor.tool_rename(agent.index, agent.cfg, "apply_discount", "d")
    assert "rename aborted" in out and "rolled back 0" in out
    assert (tmp_path / "pkg/core.py").read_text(encoding="utf-8") == files["pkg/core.py"]


def test_apply_edits_oserror_restores_writes_and_created_files(tmp_path,
                                                               monkeypatch):
    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("Y = 1\n", encoding="utf-8")
    reply = (
        "--- FILE: a.py\n<<<<<<< SEARCH\nX = 1\n=======\nX = 2\n>>>>>>> REPLACE\n"
        "--- FILE: new.py\n<<<<<<< SEARCH\n=======\nB = 1\n>>>>>>> REPLACE\n"
        "--- FILE: b.py\n<<<<<<< SEARCH\nY = 1\n=======\nY = 2\n>>>>>>> REPLACE\n")
    parsed = parse_edits(reply)
    assert len(parsed) == 3
    real = edits.fileio.write

    def boom(path, text, style=None):
        if Path(path).name == "b.py":
            raise OSError(28, "No space left on device")
        return real(path, text, style)

    monkeypatch.setattr(edits.fileio, "write", boom)
    res = apply_edits(tmp_path, parsed, backup=False)
    assert len(res) == 1 and not res[0].ok and res[0].file == "*"
    assert "write batch failed" in res[0].message
    assert "rolled back 2 of 2" in res[0].message
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "X = 1\n"
    assert not (tmp_path / "new.py").exists()
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "Y = 1\n"
