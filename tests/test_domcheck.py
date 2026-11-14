"""U6: the headless DOM smoke check (pedacito/domcheck.py).

The deterministic helpers (entry detection, snapshot diffing, finding keys,
the missing-install path) are tested without a browser. The live checks load
real pages in headless Chromium and are skipped when Playwright is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import domcheck  # noqa: E402

_HAS_PW = domcheck.available()[0]
requires_pw = pytest.mark.skipif(
    not _HAS_PW, reason="Playwright + Chromium not installed")


# --- deterministic helpers ---------------------------------------------------

def test_find_entry_prefers_root_index(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "index.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    assert domcheck.find_entry(tmp_path) == "index.html"


def test_find_entry_prefers_module_page_over_plain(tmp_path):
    (tmp_path / "about.html").write_text("<html><body>hi</body></html>", encoding="utf-8")
    (tmp_path / "app.html").write_text(
        "<html><script type=\"module\" src=\"a.js\"></script></html>", encoding="utf-8")
    assert domcheck.find_entry(tmp_path) == "app.html"


def test_find_entry_override_and_missing(tmp_path):
    (tmp_path / "page.html").write_text("<html></html>", encoding="utf-8")
    assert domcheck.find_entry(tmp_path, "page.html") == "page.html"
    assert domcheck.find_entry(tmp_path, "nope.html") == ""
    assert domcheck.find_entry(tmp_path) == "page.html"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert domcheck.find_entry(empty) == ""


def test_finding_key_ignores_shifting_digits():
    a = domcheck.DomFinding("page_error", "error", "boom at app.js:12")
    b = domcheck.DomFinding("page_error", "error", "boom at app.js:99")
    assert a.key == b.key
    c = domcheck.DomFinding("load_error", "error", "boom at app.js:12")
    assert a.key != c.key


def _snap(elements, controls=(), text_len=100):
    return {"elements": elements, "controls": list(controls),
            "text_len": text_len}


def test_diff_mounted_flags_lost_named_elements_and_controls():
    before = _snap(
        [{"tag": "div", "id": "app", "classes": ""},
         {"tag": "button", "id": "go", "classes": ""},
         {"tag": "span", "id": "label", "classes": ""}],
        controls=[{"tag": "button", "id": "go", "classes": ""}])
    after = _snap([{"tag": "div", "id": "app", "classes": ""}])
    found = domcheck._diff_mounted(before, after)
    assert found and found[0].kind == "mount_regression"
    assert found[0].severity == "error"
    assert "go" in found[0].message


def test_diff_mounted_ignores_anonymous_churn_and_additions():
    before = _snap([{"tag": "div", "id": "app", "classes": ""},
                    {"tag": "p", "id": "", "classes": "x"}])
    # the named element stays; only an anonymous <p> is swapped, and a new
    # element is added -- not a regression
    after = _snap([{"tag": "div", "id": "app", "classes": ""},
                   {"tag": "p", "id": "", "classes": "y"},
                   {"tag": "section", "id": "", "classes": ""}])
    assert domcheck._diff_mounted(before, after) == []


def test_diff_mounted_flags_text_collapse():
    before = _snap([{"tag": "div", "id": "app", "classes": ""}], text_len=800)
    after = _snap([{"tag": "div", "id": "app", "classes": ""}], text_len=100)
    found = domcheck._diff_mounted(before, after)
    assert found and "text collapsed" in found[0].message


def test_check_reports_missing_install_without_a_browser(monkeypatch):
    monkeypatch.setattr(domcheck, "available", lambda: (False, "install it"))
    res = domcheck.check(Path("/nonexistent"))
    assert res.error == "install it" and not res.ok


def test_check_reports_no_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(domcheck, "available", lambda: (True, ""))
    res = domcheck.check(tmp_path)
    assert "no HTML entry" in res.error and not res.ok


def test_snapshot_roundtrip(tmp_path):
    snap = _snap([{"tag": "div", "id": "app", "classes": ""}])
    p = tmp_path / "snap.json"
    domcheck.save_snapshot(p, snap)
    assert domcheck.load_snapshot(p) == snap
    assert domcheck.load_snapshot(tmp_path / "missing.json") is None


# --- live headless checks ----------------------------------------------------

INDEX = ('<!doctype html><html><head><title>t</title></head>'
         '<body><div id="app"></div>'
         '<script type="module" src="app.js"></script></body></html>')


def _project(tmp_path, app_js):
    (tmp_path / "index.html").write_text(INDEX, encoding="utf-8")
    (tmp_path / "app.js").write_text(app_js, encoding="utf-8")
    return tmp_path


@requires_pw
def test_clean_mounted_page_passes(tmp_path):
    _project(tmp_path, (
        "const app = document.getElementById('app');\n"
        "const b = document.createElement('button');\n"
        "b.id = 'go'; b.textContent = 'Go';\n"
        "b.addEventListener('click', () => {});\n"
        "app.appendChild(b);\n"))
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    assert res.ok, [str(f) for f in res.findings]
    assert res.snapshot and res.snapshot["body_children"] >= 1


@requires_pw
def test_uncaught_page_error_is_reported(tmp_path):
    _project(tmp_path, "throw new Error('boom');\n")
    res = domcheck.check(tmp_path)
    assert not res.error
    assert any(f.kind == "page_error" and f.severity == "error"
               for f in res.findings), [str(f) for f in res.findings]
    assert not res.ok


@requires_pw
def test_missing_module_is_a_load_error(tmp_path):
    (tmp_path / "index.html").write_text(
        '<!doctype html><html><body><div id="app"></div>'
        '<script type="module" src="missing.js"></script></body></html>', encoding="utf-8")
    res = domcheck.check(tmp_path)
    assert not res.error
    assert any(f.kind == "load_error" for f in res.findings), \
        [str(f) for f in res.findings]
    assert not res.ok


@requires_pw
def test_unbound_control_is_a_warning_and_delegation_suppresses_it(tmp_path):
    # a lone button with no listener anywhere -> unbound (warn, not error)
    (tmp_path / "index.html").write_text(
        '<!doctype html><html><body><button id="lonely">x</button>'
        '</body></html>', encoding="utf-8")
    res = domcheck.check(tmp_path)
    assert not res.error
    assert any(f.kind == "unbound_control" and f.severity == "warn"
               for f in res.findings), [str(f) for f in res.findings]
    assert res.ok                        # warnings do not gate

    # a document-level (delegated) listener suppresses the finding
    (tmp_path / "index.html").write_text(
        '<!doctype html><html><body><button id="lonely">x</button>'
        '<script>document.addEventListener("click", () => {});</script>'
        '</body></html>', encoding="utf-8")
    res = domcheck.check(tmp_path)
    assert not any(f.kind == "unbound_control" for f in res.findings), \
        [str(f) for f in res.findings]


@requires_pw
def test_snapshot_diff_catches_a_replaced_entry_module(tmp_path):
    # BUG-216 shape: the entry module mounts UI, then is replaced by one that
    # mounts nothing; the mounted-surface diff must flag the regression.
    _project(tmp_path, (
        "const app = document.getElementById('app');\n"
        "app.innerHTML = '<button id=\"go\">Go</button>"
        "<span id=\"label\">hi</span>';\n"
        "document.getElementById('go').addEventListener('click', () => {});\n"))
    before = domcheck.check(tmp_path)
    assert before.ok and before.snapshot
    # replace the entry module with one that no longer mounts the UI
    (tmp_path / "app.js").write_text("// replaced module: mounts nothing\n", encoding="utf-8")
    after = domcheck.check(tmp_path, baseline=before.snapshot)
    assert any(f.kind == "mount_regression" and f.severity == "error"
               for f in after.findings), [str(f) for f in after.findings]
    assert not after.ok


# --- Web Audio telemetry: deterministic verdicts ------------------------------
# _audio_findings is a pure function of (report, probe), so every failure
# shape is unit-testable without a browser. The synthetic reports mirror
# what window.__pdcAudioReport() returns.

def _actx(cid=1, kind="realtime", state="running", resume=0,
          resume_gesture=0, created_gesture=False, dest=2):
    return {"id": cid, "kind": kind, "state": state, "resumeCalls": resume,
            "resumeInGesture": resume_gesture,
            "createdInGesture": created_gesture, "dest": dest}


def _anode(nid, ntype, ctx=1):
    return {"id": nid, "ctx": ctx, "type": ntype, "started": False}


def _areport(nodes=(), edges=(), starts=(), contexts=(), issues=(),
             gains=None, connects=0):
    return {"nodes": list(nodes), "edges": list(edges),
            "starts": list(starts), "contexts": list(contexts),
            "paramIssues": list(issues), "gains": gains or {},
            "connectCalls": connects}


def _aprobe(play=True, before=0, after=1, media=0, steps=0):
    return {"playClicked": play, "playLabel": "Play", "stepToggled": False,
            "stepsActiveBefore": 0, "stepCandidates": steps,
            "startedBefore": before, "startedAfter": after,
            "mediaPlaying": media}


def _healthy_graph():
    # osc(3) -> gain(4) -> destination(2), one started source, context
    # created inside the Play click: the shape a working sequencer records.
    return _areport(
        nodes=[_anode(2, "AudioDestinationNode"),
               _anode(3, "OscillatorNode"), _anode(4, "GainNode")],
        edges=[{"from": 3, "to": 4, "param": False},
               {"from": 4, "to": 2, "param": False}],
        starts=[{"node": 3, "ctx": 1, "time": 0}],
        contexts=[_actx(created_gesture=True)],
        gains={"4": {"value": 0.5, "auto": []}}, connects=2)


def test_audio_findings_dormant_without_a_report():
    assert domcheck._audio_findings(None, _aprobe()) == []
    assert domcheck._audio_findings({}, _aprobe()) == []


def test_audio_healthy_graph_has_no_findings():
    assert domcheck._audio_findings(_healthy_graph(), _aprobe()) == []


def test_audio_param_invalid_shapes():
    rep = _healthy_graph()
    rep["paramIssues"] = [
        {"nodeType": "GainNode", "method": "setValueAtTime",
         "kind": "nonfinite", "value": "NaN"},
        {"nodeType": "GainNode", "method": "exponentialRampToValueAtTime",
         "kind": "nonpositive", "value": 0},
        {"nodeType": "OscillatorNode", "method": "setValueAtTime",
         "kind": "nonfinite_time", "value": 0},
        {"nodeType": "GainNode", "method": "setTargetAtTime",
         "kind": "negative_constant", "value": 0.1},
    ]
    found = domcheck._audio_findings(rep, _aprobe())
    assert len(found) == 4
    assert all(f.kind == "audio_param_invalid" and f.severity == "error"
               for f in found)
    assert any("exponential" in f.message for f in found)


def test_audio_graph_disconnected_is_caught():
    rep = _healthy_graph()
    # the gain never reaches ctx.destination
    rep["edges"] = [{"from": 3, "to": 4, "param": False}]
    found = domcheck._audio_findings(rep, _aprobe())
    assert [f.kind for f in found] == ["audio_graph_disconnected"]


def test_audio_graph_disconnected_needs_a_playback_attempt():
    rep = _healthy_graph()
    rep["edges"] = []
    rep["starts"] = []
    # an idle synth that nobody played is not a finding
    assert domcheck._audio_findings(rep, _aprobe(play=False, after=0)) == []


def test_audio_graph_judged_per_context():
    # an offline render graph wired to its OWN destination is healthy
    rep = _areport(
        nodes=[_anode(2, "AudioDestinationNode"),
               _anode(3, "OscillatorNode")],
        edges=[{"from": 3, "to": 2, "param": False}],
        starts=[{"node": 3, "ctx": 1, "time": 0}],
        contexts=[_actx(kind="offline")], connects=1)
    assert domcheck._audio_findings(rep, _aprobe()) == []


def test_audio_context_suspended_without_resume():
    rep = _healthy_graph()
    rep["contexts"] = [_actx(state="suspended")]  # load-time ctx, no resume
    found = domcheck._audio_findings(rep, _aprobe())
    assert [f.kind for f in found] == ["audio_context_suspended"]
    assert "state observed: suspended" in found[0].message


def test_audio_context_suspended_gate_is_behavioral():
    # headless Chromium reports "running" where a real browser would suspend:
    # no resume + non-gesture origin + playback attempted is the finding
    rep = _healthy_graph()
    rep["contexts"] = [_actx(state="running")]
    found = domcheck._audio_findings(rep, _aprobe())
    assert [f.kind for f in found] == ["audio_context_suspended"]
    assert "state observed" not in found[0].message


def test_audio_context_gesture_and_resume_exemptions():
    for ctx in (_actx(created_gesture=True),        # created in the click
                _actx(resume=1, resume_gesture=1),  # resumed in a gesture
                _actx(kind="offline")):             # offline renders exempt
        rep = _healthy_graph()
        rep["contexts"] = [ctx]
        assert domcheck._audio_findings(rep, _aprobe()) == []


def test_audio_context_resume_only_outside_gesture():
    rep = _healthy_graph()
    rep["contexts"] = [_actx(state="suspended", resume=1, resume_gesture=0)]
    found = domcheck._audio_findings(rep, _aprobe())
    assert [f.kind for f in found] == ["audio_context_suspended"]
    assert "outside a user gesture" in found[0].message


def test_audio_context_unlock_on_first_interaction_is_exempt():
    # the GridBastion shape: a load-time jingle creates the context and
    # calls resume() outside a gesture, and the real unlock lives in the
    # first click handler. The probe finds no Play button and clicks
    # nothing: that design is correct, not a finding.
    rep = _healthy_graph()
    rep["contexts"] = [_actx(state="suspended", resume=1, resume_gesture=0)]
    assert domcheck._audio_findings(rep, _aprobe(play=False)) == []


def test_audio_context_unused_is_exempt():
    rep = _areport(contexts=[_actx()])  # no nodes, no edges, nothing started
    assert domcheck._audio_findings(rep, _aprobe(after=0)) == []


def test_audio_dead_transport_is_caught():
    rep = _healthy_graph()
    rep["starts"] = []
    found = domcheck._audio_findings(rep, _aprobe(before=0, after=0))
    assert [f.kind for f in found] == ["audio_dead_transport"]


def test_audio_dead_transport_exemptions():
    rep = _healthy_graph()
    rep["starts"] = []
    # media-element playback does not call start()
    assert domcheck._audio_findings(rep, _aprobe(after=0, media=1)) == []
    # a routed AudioWorkletNode synthesizes without start()
    worklet = _areport(
        nodes=[_anode(2, "AudioDestinationNode"),
               _anode(5, "AudioWorkletNode")],
        edges=[{"from": 5, "to": 2, "param": False}],
        contexts=[_actx(created_gesture=True)], connects=1)
    assert domcheck._audio_findings(worklet, _aprobe(after=0)) == []
    # no audio wiring at all: a Play button alone is not an audio finding
    bare = _areport(contexts=[_actx(created_gesture=True)])
    assert domcheck._audio_findings(bare, _aprobe(after=0)) == []


def test_audio_gain_zeroed_is_caught():
    rep = _healthy_graph()
    rep["gains"] = {"4": {"value": 0, "auto": []}}
    found = domcheck._audio_findings(rep, _aprobe())
    assert [f.kind for f in found] == ["audio_gain_zeroed"]


def test_audio_gain_envelope_or_bypass_is_not_zeroed():
    # automation raises the gain later: not pinned
    rep = _healthy_graph()
    rep["gains"] = {"4": {"value": 0,
                          "auto": [{"m": "linearRampToValueAtTime",
                                    "v": 0.8}]}}
    assert domcheck._audio_findings(rep, _aprobe()) == []
    # a parallel un-muted path reaches the destination
    rep = _healthy_graph()
    rep["gains"] = {"4": {"value": 0, "auto": []},
                    "6": {"value": 1, "auto": []}}
    rep["nodes"].append(_anode(6, "GainNode"))
    rep["edges"] += [{"from": 3, "to": 6, "param": False},
                     {"from": 6, "to": 2, "param": False}]
    assert domcheck._audio_findings(rep, _aprobe()) == []


def test_audio_finding_keys_are_digit_free_and_ignore_shifting_ids():
    rep = _healthy_graph()
    rep["contexts"] = [_actx()]   # suspended-gate finding
    rep["starts"] = []            # + dead transport
    a = domcheck._audio_findings(rep, _aprobe(after=0))
    assert a
    for f in a:
        assert not any(ch.isdigit() for ch in f.key)
    # the same shape with shifted node ids must produce identical keys
    shift = {2: 22, 3: 33, 4: 44}
    rep2 = _areport(
        nodes=[dict(n, id=shift[n["id"]]) for n in rep["nodes"]],
        edges=[{"from": shift[e["from"]], "to": shift[e["to"]],
                "param": e["param"]} for e in rep["edges"]],
        contexts=[dict(rep["contexts"][0], dest=shift[2])], connects=2)
    b = domcheck._audio_findings(rep2, _aprobe(after=0))
    assert [f.key for f in a] == [f.key for f in b]


def test_audio_static_presence(tmp_path):
    (tmp_path / "app.js").write_text("const c = new AudioContext();", encoding="utf-8")
    assert domcheck._audio_static_presence(tmp_path)
    (tmp_path / "app.js").write_text("document.title = 'hi';", encoding="utf-8")
    assert not domcheck._audio_static_presence(tmp_path)
    # a lazy context referenced only in inline HTML still counts
    (tmp_path / "lazy.html").write_text("<script>ctx.createOscillator()</script>", encoding="utf-8")
    assert domcheck._audio_static_presence(tmp_path)


def test_audio_static_presence_skips_vendor_and_backups(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("new AudioContext()", encoding="utf-8")
    (tmp_path / ".pedacito").mkdir()
    (tmp_path / ".pedacito" / "old.js").write_text("new AudioContext()", encoding="utf-8")
    assert not domcheck._audio_static_presence(tmp_path)


# --- Web Audio telemetry: live headless checks --------------------------------

@requires_pw
def test_working_audio_page_passes(tmp_path):
    _project(tmp_path, (
        "const app = document.getElementById('app');\n"
        "const b = document.createElement('button');\n"
        "b.id = 'play'; b.textContent = 'Play';\n"
        "b.addEventListener('click', () => {\n"
        "  const ctx = new AudioContext();\n"
        "  ctx.resume();\n"
        "  const osc = ctx.createOscillator();\n"
        "  const g = ctx.createGain();\n"
        "  g.gain.value = 0.2;\n"
        "  osc.connect(g); g.connect(ctx.destination);\n"
        "  osc.start(); osc.stop(ctx.currentTime + 0.3);\n"
        "});\n"
        "app.appendChild(b);\n"))
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    assert res.ok, [str(f) for f in res.findings]
    assert not any(f.kind.startswith("audio_") for f in res.findings)
    assert res.snapshot["audio"]["sources_started"] >= 1
    assert res.snapshot["audio"]["play_clicked"] is True


@requires_pw
def test_scene1_shape_unstarted_voices_and_dead_scheduler(tmp_path):
    # bug_hunting_2: context at load (never resumed), voices wired but never
    # start()ed, and the scheduler reads a pattern that was never assigned.
    _project(tmp_path, (
        "const ctx = new AudioContext();\n"
        "const app = document.getElementById('app');\n"
        "const b = document.createElement('button');\n"
        "b.id = 'play'; b.textContent = 'Play';\n"
        "b.addEventListener('click', () => {\n"
        "  const osc = ctx.createOscillator();\n"
        "  const g = ctx.createGain();\n"
        "  osc.connect(g); g.connect(ctx.destination);\n"
        "  const pattern = undefined;\n"
        "  void pattern.solo;\n"
        "});\n"
        "app.appendChild(b);\n"))
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    kinds = {f.kind for f in res.findings}
    assert kinds == {"page_error", "audio_context_suspended",
                     "audio_dead_transport"}, [str(f) for f in res.findings]
    assert not res.ok


@requires_pw
def test_scene2_shape_lazy_context_in_timer_never_resumed(tmp_path):
    # bug_hunting_3: the context is created lazily inside setInterval (outside
    # any gesture) and resume() is never called; sources DO start, so the
    # silence is the suspended context, not a dead transport.
    _project(tmp_path, (
        "const app = document.getElementById('app');\n"
        "let ctx = null, timer = null;\n"
        "const b = document.createElement('button');\n"
        "b.id = 'play'; b.textContent = 'Play';\n"
        "b.addEventListener('click', () => {\n"
        "  if (timer) return;\n"
        "  timer = setInterval(() => {\n"
        "    if (!ctx) ctx = new AudioContext();\n"
        "    const osc = ctx.createOscillator();\n"
        "    const g = ctx.createGain();\n"
        "    osc.connect(g); g.connect(ctx.destination);\n"
        "    osc.start(); osc.stop(ctx.currentTime + 0.05);\n"
        "  }, 100);\n"
        "});\n"
        "app.appendChild(b);\n"))
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    kinds = {f.kind for f in res.findings}
    assert kinds == {"audio_context_suspended"}, \
        [str(f) for f in res.findings]
    assert res.snapshot["audio"]["sources_started"] >= 1


@requires_pw
def test_disconnected_graph_is_caught(tmp_path):
    _project(tmp_path, (
        "const app = document.getElementById('app');\n"
        "const b = document.createElement('button');\n"
        "b.id = 'play'; b.textContent = 'Play';\n"
        "b.addEventListener('click', () => {\n"
        "  const ctx = new AudioContext();\n"
        "  const osc = ctx.createOscillator();\n"
        "  osc.start();\n"   # started but never connected
        "});\n"
        "app.appendChild(b);\n"))
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    kinds = {f.kind for f in res.findings}
    assert kinds == {"audio_graph_disconnected"}, \
        [str(f) for f in res.findings]


@requires_pw
def test_gain_pinned_at_zero_is_caught(tmp_path):
    _project(tmp_path, (
        "const app = document.getElementById('app');\n"
        "const b = document.createElement('button');\n"
        "b.id = 'play'; b.textContent = 'Play';\n"
        "b.addEventListener('click', () => {\n"
        "  const ctx = new AudioContext();\n"
        "  const osc = ctx.createOscillator();\n"
        "  const g = ctx.createGain();\n"
        "  g.gain.value = 0;\n"
        "  osc.connect(g); g.connect(ctx.destination);\n"
        "  osc.start(); osc.stop(ctx.currentTime + 0.3);\n"
        "});\n"
        "app.appendChild(b);\n"))
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    kinds = {f.kind for f in res.findings}
    assert kinds == {"audio_gain_zeroed"}, [str(f) for f in res.findings]


@requires_pw
def test_swallowed_param_errors_are_caught(tmp_path):
    # the app catches the TypeError itself: no page_error, but the telemetry
    # recorded the invalid automation BEFORE the native call threw
    _project(tmp_path, (
        "const app = document.getElementById('app');\n"
        "const b = document.createElement('button');\n"
        "b.id = 'play'; b.textContent = 'Play';\n"
        "b.addEventListener('click', () => {\n"
        "  const ctx = new AudioContext();\n"
        "  const osc = ctx.createOscillator();\n"
        "  const g = ctx.createGain();\n"
        "  osc.connect(g); g.connect(ctx.destination);\n"
        "  try { g.gain.setValueAtTime(NaN, ctx.currentTime); } catch (e) {}\n"
        "  try {\n"
        "    g.gain.exponentialRampToValueAtTime(0, ctx.currentTime + 0.1);\n"
        "  } catch (e) {}\n"
        "  osc.start();\n"
        "});\n"
        "app.appendChild(b);\n"))
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    audio = [f for f in res.findings if f.kind == "audio_param_invalid"]
    assert len(audio) == 2, [str(f) for f in res.findings]
    assert not any(f.kind == "page_error" for f in res.findings)
    assert not res.ok


@requires_pw
def test_slider_sweep_catches_a_method_name_typo(tmp_path):
    # bug_hunting_2's slider bug: the handler calls setCutoff, the synth only
    # defines setFilterCutoff -- a TypeError nobody sees until a slider moves.
    _project(tmp_path, (
        "class Synth { setFilterCutoff(v) {} }\n"
        "const synth = new Synth();\n"
        "const app = document.getElementById('app');\n"
        "const b = document.createElement('button');\n"
        "b.id = 'play'; b.textContent = 'Play';\n"
        "b.addEventListener('click', () => {\n"
        "  const ctx = new AudioContext();\n"
        "  const osc = ctx.createOscillator();\n"
        "  const g = ctx.createGain();\n"
        "  osc.connect(g); g.connect(ctx.destination);\n"
        "  osc.start(); osc.stop(ctx.currentTime + 0.3);\n"
        "});\n"
        "app.appendChild(b);\n"
        "const s = document.createElement('input');\n"
        "s.type = 'range'; s.id = 'cutoff'; s.min = '0'; s.max = '100';\n"
        "s.addEventListener('input', () => synth.setCutoff(Number(s.value)));\n"
        "app.appendChild(s);\n"))
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    assert any(f.kind == "page_error" and "setCutoff" in f.message
               for f in res.findings), [str(f) for f in res.findings]
    assert not any(f.kind.startswith("audio_") for f in res.findings)


def _sequencer_project(tmp_path, double_flip):
    """A 4-step grid + Play that only starts a source when a step is active.
    double_flip reproduces bug_hunting_3's toggle: the click handler flips
    the state twice (ui.js + main.js), so a click never activates a step."""
    flips = "on = !on; on = !on;" if double_flip else "on = !on;"
    _project(tmp_path, (
        "const app = document.getElementById('app');\n"
        "const active = [false, false, false, false];\n"
        "for (let i = 0; i < 4; i++) {\n"
        "  const s = document.createElement('button');\n"
        "  s.className = 'step'; s.dataset.step = String(i);\n"
        "  s.textContent = String(i + 1);\n"
        "  s.addEventListener('click', () => {\n"
        f"    let on = active[i]; {flips} active[i] = on;\n"
        "    s.classList.toggle('active', active[i]);\n"
        "  });\n"
        "  app.appendChild(s);\n"
        "}\n"
        "const b = document.createElement('button');\n"
        "b.id = 'play'; b.textContent = 'Play';\n"
        "b.addEventListener('click', () => {\n"
        "  if (!active.some(Boolean)) return;\n"
        "  const ctx = new AudioContext();\n"
        "  const osc = ctx.createOscillator();\n"
        "  const g = ctx.createGain();\n"
        "  osc.connect(g); g.connect(ctx.destination);\n"
        "  osc.start(); osc.stop(ctx.currentTime + 0.3);\n"
        "});\n"
        "app.appendChild(b);\n"))


@requires_pw
def test_probe_enables_an_empty_grid_then_play_starts_sources(tmp_path):
    _sequencer_project(tmp_path, double_flip=False)
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    assert res.ok, [str(f) for f in res.findings]
    assert res.snapshot["audio"]["sources_started"] >= 1


@requires_pw
def test_double_flip_toggle_leaves_a_dead_transport(tmp_path):
    _sequencer_project(tmp_path, double_flip=True)
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    kinds = {f.kind for f in res.findings}
    assert kinds == {"audio_dead_transport"}, [str(f) for f in res.findings]


@requires_pw
def test_non_audio_page_has_zero_audio_findings(tmp_path):
    # a Play button with no Web Audio anywhere: the telemetry stays dormant
    # (no static reference, no runtime activity) and must not gate the run
    _project(tmp_path, (
        "const app = document.getElementById('app');\n"
        "const b = document.createElement('button');\n"
        "b.id = 'play'; b.textContent = 'Play';\n"
        "b.addEventListener('click', () => {});\n"
        "app.appendChild(b);\n"))
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    assert res.ok, [str(f) for f in res.findings]
    assert not any(f.kind.startswith("audio_") for f in res.findings)
    assert "audio" not in res.snapshot


@requires_pw
def test_game_unlocking_audio_on_first_click_is_exempt(tmp_path):
    # the GridBastion shape (bug_hunting control): a load-time jingle
    # creates/resumes the context outside a gesture, the real unlock is
    # bound to ordinary clicks, and there is no Play button. The probe
    # clicks nothing; the correct design must not be flagged.
    _project(tmp_path, (
        "let ctx = null;\n"
        "function ensureCtx() {\n"
        "  if (!ctx) ctx = new AudioContext();\n"
        "  if (ctx.state === 'suspended') ctx.resume().catch(() => {});\n"
        "  return ctx;\n"
        "}\n"
        "function jingle() {\n"
        "  const c = ensureCtx();\n"
        "  const o = c.createOscillator();\n"
        "  const g = c.createGain();\n"
        "  o.connect(g); g.connect(c.destination);\n"
        "  o.start(); o.stop(c.currentTime + 0.08);\n"
        "}\n"
        "jingle();\n"   # load-time wave start, outside any gesture
        "const app = document.getElementById('app');\n"
        "const b = document.createElement('button');\n"
        "b.id = 'tower-pulse'; b.textContent = 'Pulse (50)';\n"
        "b.addEventListener('click', () => { ensureCtx(); jingle(); });\n"
        "app.appendChild(b);\n"))
    res = domcheck.check(tmp_path)
    assert not res.error, res.error
    assert res.ok, [str(f) for f in res.findings]
    assert not any(f.kind.startswith("audio_") for f in res.findings)