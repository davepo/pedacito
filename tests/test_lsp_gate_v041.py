"""WP-4103C: the opt-in LSP diagnostics gate.

Pure delta/URI tests plus end-to-end runs against a fake stdio language server
that publishes versioned, stale, unversioned or no diagnostics on demand. No
real language server is needed.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import env_resolve, lsp, loop
from pedacito.config import EXEC_FIELDS, Config, project_template
from pedacito.lsp import Diagnostic

FAKE_GATE_SERVER = r'''
import json, sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "versioned"
def read():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        k, v = line.decode("ascii", "replace").split(":", 1)
        headers[k.strip().lower()] = v.strip()
    n = int(headers.get("content-length", "0"))
    body = sys.stdin.buffer.read(n) if n > 0 else b""
    return json.loads(body.decode("utf-8")) if body else None
def send(obj):
    body = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    sys.stdout.buffer.flush()
def publish(uri, diags, version):
    params = {"uri": uri, "diagnostics": diags}
    if version is not None:
        params["version"] = version
    send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": params})
def stale(uri, version):
    return [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
             "severity": 1, "code": "E9", "message": "stale finding"}]
def diags_for(text):
    out = []
    for i, line in enumerate(text.splitlines()):
        if "BUG" in line:
            out.append({"range": {"start": {"line": i, "character": 0}, "end": {"line": i, "character": 3}},
                        "severity": 1, "code": "E1", "message": "bug: " + line.strip()})
        if "SPAN" in line:
            out.append({"range": {"start": {"line": i, "character": 0}, "end": {"line": i + 1, "character": 0}},
                        "severity": 2, "code": "E2", "message": "span"})
    return out
while True:
    msg = read()
    if msg is None:
        break
    method, rid = msg.get("method"), msg.get("id")
    if method == "exit":
        break
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"capabilities": {}}})
        continue
    if method in ("initialized", "shutdown"):
        if rid is not None:
            send({"jsonrpc": "2.0", "id": rid, "result": None})
        continue
    if method in ("textDocument/didOpen", "textDocument/didChange"):
        params = msg.get("params") or {}
        td = params.get("textDocument") or {}
        uri, ver = td.get("uri"), td.get("version")
        if MODE == "silent":
            continue
        if method == "textDocument/didOpen":
            text = td.get("text", "")
        else:
            text = params["contentChanges"][0]["text"]
        if MODE == "unversioned":
            publish(uri, diags_for(text), None)
        elif MODE == "stale":
            if ver and ver > 1:
                publish(uri, stale(uri, ver - 1), ver - 1)
            publish(uri, diags_for(text), ver)
        else:
            publish(uri, diags_for(text), ver)
        continue
    if rid is not None:
        send({"jsonrpc": "2.0", "id": rid, "result": None})
'''


def _server(tmp_path, mode="versioned") -> str:
    script = tmp_path / "fake_gate_lsp.py"
    script.write_text(FAKE_GATE_SERVER, encoding="utf-8")
    return f'"{sys.executable}" "{script}" {mode}'


def _cfg(tmp_path, mode="versioned", *, gate=True, allow=True):
    cfg = Config()
    cfg.allow_lsp = allow
    cfg.lsp_diagnostics_gate = gate
    cfg.lsp_timeout = 5
    cfg.lsp_servers = {".py": _server(tmp_path, mode)}
    return cfg


def _root(tmp_path, text="x = 1\nBUG old\ny = 2\n", name="a.py"):
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    (root / name).write_text(text, encoding="utf-8")
    return root


def _env(root):
    return env_resolve.resolve_environment(Path(root))


def _deadline(seconds=lsp.GATE_BUDGET_SECONDS):
    return time.monotonic() + seconds


# ------------------------------------------------------------------ URIs

def test_from_uri_strips_windows_drive_slash_and_decodes():
    uri = PureWindowsPath("C:/Users/dev/my proj/a.py").as_uri()
    assert uri == "file:///C:/Users/dev/my%20proj/a.py"
    assert lsp._from_uri(uri) == str(PureWindowsPath("C:/Users/dev/my proj/a.py"))


def test_from_uri_rebuilds_unc_share():
    uri = PureWindowsPath("//server/share/src/a.py").as_uri()
    assert uri == "file://server/share/src/a.py"
    assert lsp._from_uri(uri) == str(PureWindowsPath("//server/share/src/a.py"))


def test_from_uri_leaves_posix_and_non_file_uris_alone():
    assert lsp._from_uri("file:///home/dev/a.py") == "/home/dev/a.py"
    assert lsp._from_uri("untitled:Untitled-1") == "untitled:Untitled-1"


def test_uri_key_matches_file_uri_of_same_path(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x\n", encoding="utf-8")
    assert lsp._uri_key(lsp._to_uri(f)) == lsp._uri_key(str(f))
    if os.name == "nt":
        uri = lsp._to_uri(f)
        swapped = uri[:8] + uri[8].swapcase() + uri[9:]
        assert lsp._uri_key(swapped) == lsp._uri_key(str(f))


# ------------------------------------------------------------- line maps

def test_diff_line_maps_marks_inserted_lines_and_maps_unchanged():
    changed, mapping = lsp.diff_line_maps("a\nb\nc\n", "a\nNEW\nb\nc\n")
    assert changed == {2}
    assert mapping == {1: 1, 2: 3, 3: 4}


def test_diff_line_maps_deleted_line_has_no_mapping():
    changed, mapping = lsp.diff_line_maps("a\nb\nc\n", "a\nc\n")
    assert changed == set()
    assert 2 not in mapping and mapping[3] == 2


# ------------------------------------------------------------- delta rules

def _d(line, end=None, code="E1", msg="bug", ver=2, file="a.py", sev=1):
    return Diagnostic(file, line, end or line, code, msg, sev, ver)


def test_delta_keeps_only_new_findings_on_changed_lines():
    before = (_d(2, ver=1),)
    after = (_d(2), _d(5))
    out = lsp.diagnostics_delta(before, after, {"a.py": {5}},
                                line_maps={"a.py": {2: 2}})
    assert out == (_d(5),)


def test_delta_identical_finding_moved_by_mapping_is_not_new():
    before = (_d(1, ver=1),)
    after = (_d(3),)
    out = lsp.diagnostics_delta(before, after, {"a.py": {3}},
                                line_maps={"a.py": {1: 3}})
    assert out == ()


def test_delta_same_message_on_other_line_is_new():
    before = (_d(1, ver=1),)
    after = (_d(4),)
    out = lsp.diagnostics_delta(before, after, {"a.py": {4}},
                                line_maps={"a.py": {1: 1}})
    assert out == (_d(4),)


def test_delta_without_line_map_is_conservative_on_same_message():
    before = (_d(9, ver=1),)
    after = (_d(4),)
    assert lsp.diagnostics_delta(before, after, {"a.py": {4}}) == ()


def test_delta_multiline_range_intersecting_a_changed_line_is_actionable():
    after = (_d(3, end=6, code="E2", msg="span"),)
    out = lsp.diagnostics_delta((), after, {"a.py": {5}})
    assert out == after


def test_delta_unversioned_or_unchanged_region_is_not_actionable():
    assert lsp.diagnostics_delta((), (_d(4, ver=None),), {"a.py": {4}}) == ()
    assert lsp.diagnostics_delta((), (_d(4),), {"a.py": {9}}) == ()


def test_delta_deleted_baseline_line_counts_as_pre_existing():
    before = (_d(2, ver=1),)
    after = (_d(7),)
    out = lsp.diagnostics_delta(before, after, {"a.py": {7}},
                                line_maps={"a.py": {}})
    assert out == ()


# ------------------------------------------------------------ gate gating

def test_gate_denied_unless_both_flags_are_set(tmp_path):
    root = _root(tmp_path)
    for gate, allow in ((False, True), (True, False), (False, False)):
        cfg = _cfg(tmp_path, gate=gate, allow=allow)
        status, diags = lsp.collect_gate_diagnostics(
            root, ["a.py"], cfg=cfg, environment=_env(root), deadline=_deadline())
        assert status == "denied" and diags == ()


def test_gate_refuses_paths_outside_the_project(tmp_path):
    root = _root(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("BUG\n", encoding="utf-8")
    cfg = _cfg(tmp_path)
    for rel in ("../outside.py", str(outside)):
        status, diags = lsp.collect_gate_diagnostics(
            root, [rel], cfg=cfg, environment=_env(root), deadline=_deadline())
        assert status == "denied" and diags == ()


def test_config_flag_is_execution_gated_and_commented_in_template():
    assert Config().lsp_diagnostics_gate is False
    assert "lsp_diagnostics_gate" in EXEC_FIELDS
    tmpl = project_template()
    assert re.search(r"^#\s*lsp_diagnostics_gate\s*=", tmpl, re.M)
    assert not re.search(r"^lsp_diagnostics_gate\s*=", tmpl, re.M)


# ------------------------------------------------- end-to-end, fake server

def test_baseline_then_after_reports_only_the_new_finding(tmp_path):
    root = _root(tmp_path)
    cfg = _cfg(tmp_path)
    lsp.close_all()
    try:
        env = _env(root)
        st0, before = lsp.collect_gate_diagnostics(
            root, ["a.py"], cfg=cfg, environment=env, deadline=_deadline())
        assert st0 == "ok" and [d.line for d in before] == [2]
        old = (root / "a.py").read_text(encoding="utf-8")
        new = "x = 1\nBUG old\nBUG new\ny = 2\nSPAN here\nz = 3\n"
        (root / "a.py").write_text(new, encoding="utf-8")
        st1, after = lsp.collect_gate_diagnostics(
            root, ["a.py"], cfg=cfg, environment=env, deadline=_deadline())
        assert st1 == "ok" and all(d.version == 2 for d in after)
        changed, mapping = lsp.diff_line_maps(old, new)
        out = lsp.diagnostics_delta(before, after, {"a.py": changed},
                                    line_maps={"a.py": mapping})
        assert sorted((d.code, d.line, d.end_line) for d in out) == [
            ("E1", 3, 3), ("E2", 5, 6)]
    finally:
        lsp.close_all()


def test_stale_delayed_notifications_are_ignored(tmp_path):
    root = _root(tmp_path)
    cfg = _cfg(tmp_path, "stale")
    lsp.close_all()
    try:
        env = _env(root)
        lsp.collect_gate_diagnostics(root, ["a.py"], cfg=cfg, environment=env,
                                     deadline=_deadline())
        (root / "a.py").write_text("BUG fresh\n", encoding="utf-8")
        status, after = lsp.collect_gate_diagnostics(
            root, ["a.py"], cfg=cfg, environment=env, deadline=_deadline())
        assert status == "ok"
        assert [d.message for d in after] == ["bug: BUG fresh"]
        assert all(d.version == 2 for d in after)
    finally:
        lsp.close_all()


def test_unversioned_server_is_stale_not_green(tmp_path):
    root = _root(tmp_path)
    cfg = _cfg(tmp_path, "unversioned")
    lsp.close_all()
    try:
        status, diags = lsp.collect_gate_diagnostics(
            root, ["a.py"], cfg=cfg, environment=_env(root), deadline=_deadline())
        assert status == "stale" and diags == ()
    finally:
        lsp.close_all()


def test_silent_server_times_out_within_budget_and_is_reaped(tmp_path, monkeypatch):
    root = _root(tmp_path)
    cfg = _cfg(tmp_path, "silent")
    lsp.close_all()
    evicted: list = []
    real_evict = lsp._evict

    def spy(client):
        evicted.append(client)
        real_evict(client)

    monkeypatch.setattr(lsp, "_evict", spy)
    t0 = time.monotonic()
    status, diags = lsp.collect_gate_diagnostics(
        root, ["a.py"], cfg=cfg, environment=_env(root), deadline=_deadline())
    elapsed = time.monotonic() - t0
    assert status == "timeout" and diags == ()
    assert elapsed < lsp.GATE_BUDGET_SECONDS + 1.0
    assert evicted and evicted[0].proc.poll() is not None


def test_budget_is_shared_across_files_with_no_per_file_reset(tmp_path):
    root = _root(tmp_path)
    (root / "b.py").write_text("y = 1\n", encoding="utf-8")
    cfg = _cfg(tmp_path, "silent")
    lsp.close_all()
    t0 = time.monotonic()
    status, _ = lsp.collect_gate_diagnostics(
        root, ["a.py", "b.py"], cfg=cfg, environment=_env(root),
        deadline=_deadline(1.5))
    assert status == "timeout"
    assert time.monotonic() - t0 < 1.5 + 1.0


def test_missing_server_is_unavailable_not_green(tmp_path):
    root = _root(tmp_path)
    cfg = _cfg(tmp_path)
    cfg.lsp_servers = {".py": "pedacito-no-such-language-server-xyz --stdio"}
    lsp.close_all()
    status, diags = lsp.collect_gate_diagnostics(
        root, ["a.py"], cfg=cfg, environment=_env(root), deadline=_deadline())
    assert status == "unavailable" and diags == ()


def test_deleted_file_has_nothing_to_compare(tmp_path):
    root = _root(tmp_path)
    cfg = _cfg(tmp_path)
    lsp.close_all()
    try:
        (root / "a.py").unlink()
        status, diags = lsp.collect_gate_diagnostics(
            root, ["a.py"], cfg=cfg, environment=_env(root), deadline=_deadline())
        assert status == "ok" and diags == ()
    finally:
        lsp.close_all()


def test_legacy_diagnostics_op_flags_unversioned_output(tmp_path):
    root = _root(tmp_path)
    lsp.close_all()
    try:
        fresh = lsp.result(root, _cfg(tmp_path), "diagnostics", "a.py")
        assert "bug: BUG old" in fresh and "unverified" not in fresh
        lsp.close_all()
        loose = lsp.result(root, _cfg(tmp_path, "unversioned"), "diagnostics", "a.py")
        assert "bug: BUG old" in loose and "unverified" in loose
    finally:
        lsp.close_all()


# ------------------------------------------------------- loop integration

def test_loop_gate_yields_findings_only_for_new_problems(tmp_path):
    root = _root(tmp_path)
    cfg = _cfg(tmp_path)
    lsp.close_all()
    try:
        gate = loop._lsp_gate_begin(root, cfg, ["a.py"])
        assert gate is not None and gate.before_status == "ok"
        (root / "a.py").write_text("x = 1\nBUG old\nBUG new\ny = 2\n",
                                   encoding="utf-8")
        found = loop._lsp_gate_finish(gate, root, cfg)
        assert [(f.kind, f.severity, f.file, f.line) for f in found] == [
            ("lsp", "error", "a.py", 3)]
        assert "bug: BUG new" in found[0].message
    finally:
        lsp.close_all()


def test_loop_gate_is_off_by_default_and_unverified_stays_silent(tmp_path):
    root = _root(tmp_path)
    assert loop._lsp_gate_begin(root, Config(), ["a.py"]) is None
    cfg = _cfg(tmp_path, "silent")
    lsp.close_all()
    gate = loop._lsp_gate_begin(root, cfg, ["a.py"])
    assert gate is not None and gate.before_status == "timeout"
    assert loop._lsp_gate_finish(gate, root, cfg) == []
    lsp.close_all()
