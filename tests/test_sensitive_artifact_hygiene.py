"""WP-4004 (D10): sensitive artifacts never leak from any engine or artifact store.

The shared gate (`security.check_sensitive_write`) must refuse secret paths in
the two-phase engine (`edits`, `tools`) and the interleaved engine
(`workspace_overlay`) identically; checkpoints, ratchet snapshots, backups,
review diffs, review pages and GUI JSON previews must never carry their bytes.
Positive contract preserved: `.env.example`, `.env.sample`, `.env.template`
and `id_rsa.pub` stay allowed.
"""
from __future__ import annotations

import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import (  # noqa: E402
    checkpoints,
    edits,
    fileio,
    ratchet,
    review,
    security,
    tools,
    workspace_overlay,
)
from pedacito.config import Config  # noqa: E402
from pedacito.gui import core as gui_core  # noqa: E402
from pedacito.index import Index  # noqa: E402
from pedacito.local_server import LocalHTTPServer  # noqa: E402
from pedacito.policy import PolicyDecision  # noqa: E402

SENSITIVE_FILES = {
    "sub/.env": "API_KEY=super-secret-value\n",
    "keys/id_rsa": "-----BEGIN PRIVATE KEY-----\nabc\n",
    "certs/server.pem": "-----BEGIN CERTIFICATE-----\nabc\n",
    "credentials.json": '{"token": "abc"}\n',
    "secrets.yaml": "password: hunter2\n",
}
ALLOWED_FILES = {
    ".env.example": "API_KEY=your-key-here\n",
    ".env.sample": "API_KEY=your-key-here\n",
    ".env.template": "API_KEY=your-key-here\n",
    "id_rsa.pub": "ssh-rsa AAAA public\n",
    "app.py": "x = 1\n",
}


def _populate(root: Path) -> None:
    for rel, text in {**SENSITIVE_FILES, **ALLOWED_FILES}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


# --------------------------------------------------------------- shared gate ---


def test_check_sensitive_write_posix_windows_and_allowed_positives(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    for rel in ("sub/.env", "keys/id_rsa", "certs/server.pem",
                "credentials.json", "secrets.yaml", r"sub\.env",
                r"C:\proj\credentials.json"):
        allowed, why = security.check_sensitive_write(root, rel)
        assert not allowed and "sensitive" in why.lower()
    for rel in (".env.example", ".env.sample", ".env.template", "id_rsa.pub",
                "app.py", "sub\\app.py"):
        allowed, why = security.check_sensitive_write(root, rel)
        assert allowed and why == ""

    # An absolute path UNDER the root is reported cleanly relative.
    abs_env = root / "sub" / ".env"
    allowed, why = security.check_sensitive_write(root, str(abs_env))
    assert not allowed and "sub/.env" in why


# ------------------------------------------------------------ artifact stores ---


def test_checkpoints_exclude_sensitive_files_but_keep_examples(tmp_path):
    root = tmp_path / "proj"
    _populate(root)

    listing = dict(checkpoints.iter_workspace_files(root))
    for rel in SENSITIVE_FILES:
        assert rel not in listing
    for rel in ALLOWED_FILES:
        assert rel in listing

    cp = checkpoints.create_checkpoint(root, "hygiene", mode="manual")
    snap = checkpoints.snapshot_dir(root, cp["id"])
    for rel in SENSITIVE_FILES:
        assert not (snap / rel).exists(), rel
    for rel in ALLOWED_FILES:
        assert (snap / rel).is_file(), rel


def test_ratchet_excludes_non_dotfile_secrets_and_pem(tmp_path):
    root = tmp_path / "proj"
    _populate(root)

    candidates = {p.relative_to(root).as_posix()
                  for p, _trunc in ratchet._iter_candidate_files(root, 500)}
    for rel in ("credentials.json", "secrets.yaml", "certs/server.pem"):
        assert rel not in candidates
    assert "app.py" in candidates

    info = ratchet.begin(root, 1, "hygiene")
    assert "app.py" in info["files"]
    for rel in ("credentials.json", "secrets.yaml", "certs/server.pem",
                ".env"):
        assert rel not in info["files"]
    step = ratchet.step_dir(root, 1)
    for rel in ("credentials.json", "secrets.yaml", "certs/server.pem",
                ".env"):
        assert not (step / rel).exists()


def test_backup_copy_refuses_sensitive_and_creates_nothing(tmp_path):
    root = tmp_path / "proj"
    _populate(root)
    backup = tmp_path / "backup"

    assert fileio.backup_copy(root / "sub" / ".env", backup / "sub" / ".env") is False
    assert not (backup / "sub" / ".env").exists()
    assert fileio.backup_copy(root / "certs" / "server.pem",
                              backup / "certs" / "server.pem") is False
    assert not (backup / "certs" / "server.pem").exists()

    assert fileio.backup_copy(root / ".env.example",
                              backup / ".env.example") is True
    assert (backup / ".env.example").is_file()


# ------------------------------------------------------- interleaved engine ---


def test_virtual_workspace_refuses_sensitive_and_mirror_excludes_it(tmp_path):
    root = tmp_path / "proj"
    _populate(root)

    ws = workspace_overlay.VirtualWorkspace(root, apply=False)
    try:
        for rel in ("sub/.env", "credentials.json", "certs/server.pem"):
            result = ws.write(rel, "changed\n")
            assert not result.ok and "sensitive" in result.message.lower()
            result = ws.delete(rel)
            assert not result.ok and "sensitive" in result.message.lower()
        assert ws.write(".env.example", "API_KEY=changed\n").ok
        assert ws.delete("id_rsa.pub").ok

        with ws.exec_root() as mirror:
            for rel in SENSITIVE_FILES:
                assert not (mirror / rel).exists(), rel
            assert (mirror / "app.py").is_file()
            assert (mirror / ".env.example").is_file()
    finally:
        ws.close()


# --------------------------------------------------------- two-phase engine ---


def test_two_phase_edits_and_tools_refuse_sensitive_identically(tmp_path):
    root = tmp_path / "proj"
    _populate(root)

    env = root / "sub" / ".env"
    results = edits.apply_edits(
        root, [edits.Edit("sub/.env", "API_KEY=super-secret-value",
                          "API_KEY=leaked")],
        backup=False, dry_run=True)
    assert len(results) == 1 and not results[0].ok
    assert "sensitive" in results[0].message.lower()
    assert env.read_text(encoding="utf-8") == SENSITIVE_FILES["sub/.env"]

    # A rename whose SOURCE is sensitive is refused at the destination.
    (root / "target.json").write_text("", encoding="utf-8")
    reply = ("--- FILE: target.json\n<<<<<<< SEARCH\n=======\nplaceholder\n"
             ">>>>>>> REPLACE\n"
             "--- FILE: credentials.json\n<<<<<<< SEARCH\n"
             '{"token": "abc"}\n=======\n>>>>>>> REPLACE\n')
    results = edits.apply_edits(root, edits.parse_edits(reply),
                                backup=False, dry_run=True)
    target_results = [r for r in results if r.file == "target.json"]
    assert target_results and all(not r.ok for r in target_results)
    assert "sensitive" in " ".join(r.message for r in results).lower()

    index = Index(root, [])
    cfg = Config()
    cfg.allow_write = True
    cfg.backup = False
    assert "sensitive" in tools.tool_write(
        index, cfg, "credentials.json", '{"token": "leaked"}\n').lower()
    assert root.joinpath("credentials.json").read_text(encoding="utf-8") \
        == '{"token": "abc"}\n'
    assert "sensitive" in tools.tool_delete(
        index, cfg, "credentials.json").lower()

    (root / "plain.txt").write_text("plain\n", encoding="utf-8")
    assert "sensitive" in tools.tool_move(
        index, cfg, "plain.txt", "sub/.env").lower()
    assert "sensitive" in tools.tool_move(
        index, cfg, "credentials.json", "moved.json").lower()
    assert (root / "plain.txt").is_file()
    assert (root / "credentials.json").is_file()
    assert not (root / "moved.json").exists()


# --------------------------------------------------------------- review path ---


def _fd(rel: str, before: str, after: str) -> review.FileDiff:
    rows, added, removed, hunks = review.build_diff(before, after)
    return review.FileDiff(rel, rows, added, removed, review._sha(before),
                           after, "edit", hunks=hunks, before_text=before)


def test_review_diffs_and_write_files_refuse_sensitive(tmp_path):
    root = tmp_path / "proj"
    _populate(root)

    results = [
        edits.Result(True, ".env", "edit", "", "A=1\n", "A=2\n", "sha"),
        edits.Result(True, "credentials.json", "edit", "",
                     "{}\n", '{"x": 1}\n', "sha"),
        edits.Result(True, ".env.example", "edit", "",
                     "A=old\n", "A=new\n", "sha"),
    ]
    diffs = review.diffs_from_results(results, root=root)
    assert [d.rel for d in diffs] == [".env.example"]

    backup = tmp_path / "backup"
    (backup / "sub").mkdir(parents=True)
    (backup / "sub" / ".env").write_text("old\n", encoding="utf-8")
    (backup / "app.py").write_text("old\n", encoding="utf-8")
    (root / "sub" / ".env").write_text("new\n", encoding="utf-8")
    (root / "app.py").write_text("new\n", encoding="utf-8")
    rebuilt = review.diffs_from_backup(root, backup)
    assert [d.rel for d in rebuilt] == ["app.py"]

    fd = review.FileDiff("sub/.env", [], 1, 0, "", "leaked\n", "edit")
    ok, detail = review.write_files(root, [fd], None)
    assert not ok and "sensitive" in detail.lower()
    assert (root / "sub" / ".env").read_text(encoding="utf-8") == "new\n"


def test_review_page_and_gui_rows_redact_but_keep_apply_text(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    token = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    before = "API = 'old-value-123456'\nx = 1\n"
    after = f"API = '{token}'\nx = 1\n"
    fd = _fd("app.py", before, after)

    page = review.build_page([fd], f"task {token}", "model", "profile",
                             "nonce", False, "stamp")
    assert token not in page
    assert "[REDACTED]" in page
    # The real after text (used by the writer) is untouched.
    assert token in fd.after_text

    payload = json.dumps(gui_core.pack_rows(fd.rows))
    assert token not in payload
    assert "[REDACTED]" in payload

    # Partial hunk apply still reconstructs real code from the intact text.
    pad = "".join(f"# pad {i}\n" for i in range(10))
    safe = _fd("safe.py", "a = 1\n" + pad + "b = 2\n",
               "a = 10\n" + pad + "b = 20\n")
    assert len(safe.hunks) == 2
    ok, detail = review.apply_hunk_selection([safe], {"safe.py": [0]})
    assert ok and detail == ""
    assert "a = 10" in safe.after_text
    assert "b = 2" in safe.after_text
    assert "b = 20" not in safe.after_text


# ------------------------------------------------------------- H6 NTFS tricks ---

def test_is_sensitive_path_windows_ntfs_tricks():
    for name in (".env ", "sub/.env ", ".env::$DATA", "server.pem ",
                 "credentials.json.", "sub\\server.pem ", "id_rsa.old",
                 "server.pem.bak", "credentials.json.orig", "terraform.tfstate",
                 "terraform.tfstate.backup", "service_account.json", ".pgpass",
                 ".my.cnf", ".htpasswd", "kubeconfig", ".env.production.bak"):
        assert security.is_sensitive_path(Path("proj") / name), name
    for name in (".env.example", ".env.sample", ".env.template",
                 "id_rsa.pub", "server.crt", "app.py"):
        assert not security.is_sensitive_path(Path("proj") / name), name


def test_resolved_path_parity_in_edits_and_review(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".env").write_text("SECRET=x\n", encoding="utf-8")
    try:
        (root / "notes.txt").symlink_to(root / ".env")
    except (OSError, NotImplementedError):
        import pytest
        pytest.skip("symlinks unavailable on this platform")

    # edits.apply_edits on the symlink (resolves to .env) is refused
    results = edits.apply_edits(
        root, [edits.Edit("notes.txt", "SECRET=x", "SECRET=y")],
        backup=False, dry_run=True, cfg=Config())
    assert results and not results[0].ok
    assert "sensitive" in results[0].message.lower()

    # review.write_files on the symlink (resolves to .env) is refused
    fd = review.FileDiff("notes.txt", [], 1, 0, "", "leaked\n", "edit")
    ok, detail = review.write_files(root, [fd], None)
    assert not ok and "sensitive" in detail.lower()
    assert (root / ".env").read_text(encoding="utf-8") == "SECRET=x\n"


# ------------------------------------------------------- H7 central policy gate ---

def test_plan_mode_denies_mutating_and_external_tools(tmp_path):
    from pedacito import refactor
    idx = Index(tmp_path, [])
    cfg = Config()
    cfg.permission_mode = "plan"
    cfg.allow_write = True
    cfg.allow_shell = True
    cfg.allow_image_gen = True
    cfg.allow_screenshots = True
    cfg.allow_browser = True

    assert "denied by policy" in tools.tool_generate_image(
        idx, cfg, "a cat", file="out.png")
    assert "denied by policy" in refactor.tool_rename(idx, cfg, "old", "new")
    assert "denied by policy" in refactor.tool_rename_symbol(
        idx, cfg, "old", "new")
    assert "denied by policy" in tools.tool_browser(
        idx, cfg, None, command="http://example.com")
    assert "denied by policy" in tools.tool_screenshot(idx, cfg, None)
    assert "denied by policy" in tools.dispatch(
        idx, cfg, {"action": "remember", "content": "a durable fact"})


def _png(width=8, height=8) -> bytes:
    data = b"\x89PNG\r\n\x1a\n"
    data += (13).to_bytes(4, "big") + b"IHDR"
    data += width.to_bytes(4, "big") + height.to_bytes(4, "big")
    data += bytes([8, 2, 0, 0, 0]) + b"\x00\x00\x00\x00"
    return data


def _img_server():
    b64 = base64.b64encode(_png()).decode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            self.rfile.read(n)
            body = json.dumps({"data": [{"b64_json": b64}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = LocalHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _img_cfg(base_url: str) -> Config:
    cfg = Config()
    cfg.base_url = base_url
    cfg.model = "flux-schnell"
    cfg.modality = "image"
    cfg.image_profile = "flux"
    cfg.profiles = {"flux": {"base_url": base_url, "model": "flux-schnell",
                             "modality": "image"}}
    cfg.auto_evict_servers = False
    cfg.auto_fit = False
    cfg.auto_detect_context = False
    cfg.timeout = 10.0
    return cfg


def test_generate_image_refuses_sensitive_and_protected_outputs(tmp_path):
    srv = _img_server()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        cfg = _img_cfg(base)
        cfg.allow_image_gen = True
        idx = Index(tmp_path, [])
        for bad in (".env", "id_rsa", ".git/hooks/x", ".pedacito/policy.toml"):
            out = tools.tool_generate_image(idx, cfg, "x", file=bad)
            assert out.startswith("ERROR"), (bad, out)
            assert "sensitive" in out or "refused" in out, (bad, out)
            assert not (tmp_path / bad).exists(), bad
    finally:
        srv.shutdown()
        srv.server_close()


# ------------------------------------------------------------- MEDIUM 5 broker ---

def test_check_sensitive_write_uses_per_file_key(tmp_path, monkeypatch):
    from pedacito import security as security_mod
    keys: list[str] = []

    def fake_request(category, summary, **kw):
        keys.append(kw.get("key"))
        return False, "denied"

    monkeypatch.setattr(security_mod, "request_permission", fake_request)

    class Allow:
        def evaluate(self, **kw):
            return PolicyDecision("allow", "policy.toml:paths.allow", "ok")

    root = tmp_path / "proj"
    (root / "sub").mkdir(parents=True)
    (root / "certs").mkdir()
    (root / "sub" / ".env").write_text("x", encoding="utf-8")
    (root / "certs" / "server.pem").write_text("x", encoding="utf-8")

    security_mod.check_sensitive_write(root, "sub/.env", action="write",
                                       policy=Allow())
    security_mod.check_sensitive_write(root, "certs/server.pem", action="write",
                                       policy=Allow())
    assert keys == ["sensitive:write:sub/.env",
                    "sensitive:write:certs/server.pem"], keys


def test_check_sensitive_write_key_uses_resolved_path(tmp_path, monkeypatch):
    from pedacito import security as security_mod
    keys: list[str] = []

    def fake_request(category, summary, **kw):
        keys.append(kw.get("key"))
        return False, "denied"

    monkeypatch.setattr(security_mod, "request_permission", fake_request)

    class Allow:
        def evaluate(self, **kw):
            return PolicyDecision("allow", "policy.toml:paths.allow", "ok")

    root = tmp_path / "proj"
    root.mkdir()
    (root / ".env").write_text("SECRET=x\n", encoding="utf-8")
    try:
        (root / "notes.txt").symlink_to(root / ".env")
    except (OSError, NotImplementedError):
        import pytest
        pytest.skip("symlinks unavailable on this platform")
    security_mod.check_sensitive_write(root, "notes.txt", action="write",
                                       policy=Allow())
    assert keys and keys[0] == "sensitive:write:.env", keys


def test_broker_hashes_long_keys_and_never_beats_grants():
    from pedacito.security import SessionPermissionBroker, _broker_key
    assert _broker_key("x" * 400) == "x" * 400
    hashed = _broker_key("y" * 600)
    assert hashed.startswith("sha256:") and len(hashed) == 7 + 64
    # no prefix collision: two long keys differing only at the tail hash apart
    assert _broker_key("a" * 500 + "x") != _broker_key("a" * 500 + "y")

    broker = SessionPermissionBroker("ses_x", approval_policy="never",
                                     interactive=True)
    broker._session_grants.add((security.CAT_SHELL, "cmd"))
    ok, _msg = broker.request({"category": security.CAT_SHELL, "detail": "cmd",
                               "key": "cmd"})
    assert not ok
