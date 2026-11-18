"""The 2-model sandbox arena: run one prompt on two contestants, each in its
own isolated copy of the workspace, then compare the results side by side.

Neither contestant touches the real project: both run in sandboxes under
`.pedacito/arena/<arena_id>/{a,b}` (source files only -- no .git, .venv,
node_modules, checkpoints, dist or build trees). The comparison is stored in
`.pedacito/arena_last.json`; `apply_arena_winner` copies the chosen sandbox's
changes into the real workspace between two checkpoints, so an arena result
is always undoable.

Contestants are named by profile ("coder"), by "provider:model"
("local:qwen3-coder"), or by a {"provider": ..., "model": ...} dict. When both
contestants live on different LOCAL servers, the VRAM handoff evicts the
outgoing server between runs (auto_evict_servers).
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path

from . import checkpoints as checkpoints_mod
from . import events as events_mod
from . import index as index_mod
from . import llm as llm_mod
from . import loop as loop_mod
from . import roles as roles_mod
from . import workspace
from .agent import Agent

ARENA_DIR = "arena"
LAST_FILE = "arena_last.json"

# Edit blocks are implementation detail; the scorecard shows the prose.
_EDIT_BLOCK_RE = re.compile(
    r"^---\s*FILE:.*?^>>>>>>>\s*REPLACE[ \t]*$", re.MULTILINE | re.DOTALL)


class ArenaError(ValueError):
    """An arena request is invalid (bad contestant, unknown id, no sandbox)."""


# The most recent comparison, in memory beside its on-disk record.
_LAST: dict | None = None


def _arena_root(root) -> Path:
    return index_mod.state_dir(Path(root), ARENA_DIR)


def _last_path(root) -> Path:
    return index_mod.state_dir(Path(root), LAST_FILE)


def _save_last(root, state: dict) -> None:
    global _LAST
    _LAST = state
    path = _last_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def last_arena(root) -> dict | None:
    """The stored comparison for `root` (None before the first arena)."""
    try:
        data = json.loads(_last_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _LAST if isinstance(_LAST, dict) else None
    if isinstance(data, dict) and data.get("arena_id"):
        return data
    return None


def _parse_contestant(cfg, spec) -> tuple[str, str]:
    """(provider, model) from a profile name, a 'provider:model' string, or
    a {"provider", "model"} dict. A bare model id runs on the active
    endpoint; a profile name carries its own server and model."""
    if isinstance(spec, dict):
        provider = str(spec.get("provider") or "").strip()
        model = str(spec.get("model") or "").strip()
    else:
        text = str(spec or "").strip()
        if not text:
            raise ArenaError("A contestant needs a profile or model name.")
        profiles = getattr(cfg, "profiles", None) or {}
        if text in profiles:
            body = profiles[text] or {}
            provider = text
            model = str(body.get("model") or "").strip()
        elif "://" in text:
            # "http://host:port/v1:model" -- the URL keeps its own colons.
            provider, _, model = text.rpartition(":")
            provider, model = provider.strip(), model.strip()
        elif ":" in text:
            provider, _, model = text.partition(":")
            provider, model = provider.strip(), model.strip()
        else:
            provider, model = "", text
    if not model:
        raise ArenaError(f"Contestant {spec!r} names no model.")
    return provider, model


def _copy_workspace(src: Path, dest: Path) -> None:
    """Copy the snapshot-tracked source files into a sandbox. The checkpoint
    walker already excludes .git, .pedacito (checkpoints included), .venv,
    node_modules, dist, build and the big binary artifacts."""
    dest.mkdir(parents=True, exist_ok=True)
    for rel, path in checkpoints_mod.iter_workspace_files(src):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _contestant_config(cfg, provider: str, model: str):
    """A config copy for one contestant: its endpoint/model, with the
    cross-model machinery OFF -- routing, escalation and hybrid would hand
    the run to a third model and corrupt the comparison."""
    ccfg = roles_mod.resolve_model_config(cfg, provider, model)
    if ccfg is None:
        raise ArenaError(f"Contestant model {model!r} is empty.")
    ccfg.hybrid_routing = False
    ccfg.architect_model = ""
    ccfg.auto_route_specialties = False
    ccfg.auto_escalate = False
    ccfg.roles = {}
    ccfg.scout_profile = ""
    ccfg.backup = False
    return ccfg


def _note(root, cfg, on_event, kind: str, **fields) -> None:
    try:
        events_mod.get(root, cfg).emit(kind, **fields)
    except Exception:
        pass
    if on_event is not None:
        try:
            on_event(kind, fields)
        except Exception:
            pass


def _prose(reply: str) -> str:
    text = _EDIT_BLOCK_RE.sub("", reply or "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _run_contestant(root, sandbox: Path, cfg, provider: str, model: str,
                    prompt: str, side: str, baseline: dict,
                    on_event=None) -> dict:
    """One isolated run: fresh index, fresh client, apply inside the sandbox.
    A failed run is a scorecard row (ok=False, the error as its summary),
    never an aborted arena. A user Stop still aborts the whole arena."""
    label = f"Arena {side.upper()}: {model}"
    _note(root, cfg, on_event, "arena_stage", side=side, phase="start",
          model=model, provider=provider, label=label)
    workspace.log_event(root, label)
    ccfg = _contestant_config(cfg, provider, model)
    rec = {"side": side, "provider": provider, "model": model, "ok": False,
           "files_changed": [], "diff": "", "elapsed_sec": 0.0,
           "tok_per_sec": 0.0, "total_tokens": 0, "summary": "", "stats": {}}
    started = time.time()
    client = None
    try:
        client = llm_mod.LLMClient(ccfg)
        client.check(warn=False)
        idx = index_mod.build([str(sandbox)], ccfg, client=None, root=sandbox,
                              summarise=False, verbose=False)
        agent = Agent(idx, client, ccfg)
        tr = loop_mod.run_task(sandbox, ccfg, client, agent, idx, prompt,
                               apply=True)
        after = dict(checkpoints_mod.iter_workspace_files(sandbox))
        changed = checkpoints_mod._changed_paths(baseline, after)
        diffs = [checkpoints_mod._diff_pair(rel, baseline.get(rel), after.get(rel))
                 for rel in changed]
        stats = dict(tr.stats or {})
        seconds = float(stats.get("seconds", 0.0) or 0.0)
        out_tok = int(stats.get("completion_tokens", 0) or 0)
        rec.update({
            "ok": bool(tr.applied and (any(r.ok for r in tr.results)
                                       or tr.changed)),
            "files_changed": changed,
            "diff": "\n".join(d["diff"] for d in diffs if d.get("diff")),
            "elapsed_sec": round(time.time() - started, 2),
            "tok_per_sec": round(out_tok / seconds, 1) if seconds > 0 else 0.0,
            "total_tokens": out_tok + int(stats.get("prompt_tokens", 0) or 0),
            "summary": _prose(tr.reply)[:600],
            "stats": stats,
        })
    except llm_mod.CancelledError:
        raise                               # a user Stop aborts the arena
    except Exception as e:                  # a contestant failure is a result
        rec["elapsed_sec"] = round(time.time() - started, 2)
        rec["summary"] = f"run failed: {str(e).splitlines()[0][:300]}"
        if client is not None:
            st = getattr(client, "stats", None) or {}
            rec["total_tokens"] = (int(st.get("prompt_tokens", 0) or 0)
                                   + int(st.get("completion_tokens", 0) or 0))
    _note(root, cfg, on_event, "arena_stage", side=side, phase="done",
          model=model, provider=provider, ok=rec["ok"],
          files_changed=len(rec["files_changed"]),
          elapsed_sec=rec["elapsed_sec"], label=label)
    return rec


def run_arena_comparison(root, prompt: str, model_a, model_b, cfg, *,
                         on_event=None) -> dict:
    """Run `prompt` on two contestants in isolated sandboxes and return the
    comparison (also stored as `.pedacito/arena_last.json`). The real
    workspace is never written; `apply_arena_winner` adopts a result later."""
    global _LAST
    root = Path(root).resolve()
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ArenaError("Describe the task both contestants should run.")
    if cfg is None:
        raise ArenaError("The arena needs the project config.")
    pa, ma = _parse_contestant(cfg, model_a)
    pb, mb = _parse_contestant(cfg, model_b)
    ca = roles_mod.resolve_model_config(cfg, pa, ma)
    cb = roles_mod.resolve_model_config(cfg, pb, mb)
    if ca is None or cb is None:
        raise ArenaError("Both contestants need a model.")
    arena_id = f"arena_{uuid.uuid4().hex[:8]}"
    base = _arena_root(root)
    # Only the newest arena stays applyable; older sandboxes would pile up.
    if base.is_dir():
        for old in base.iterdir():
            if old.is_dir() and old.name != arena_id:
                shutil.rmtree(old, ignore_errors=True)
    sandboxes = {"a": base / arena_id / "a", "b": base / arena_id / "b"}
    baseline = dict(checkpoints_mod.iter_workspace_files(root))
    for side in ("a", "b"):
        _copy_workspace(root, sandboxes[side])
    _note(root, cfg, on_event, "arena_stage", phase="arena_start",
          arena_id=arena_id, model_a=ma, model_b=mb,
          label=f"Arena {arena_id}: {ma} vs {mb}")
    url_a = str(getattr(ca, "base_url", "") or "")
    url_b = str(getattr(cb, "base_url", "") or "")
    rec_a = _run_contestant(root, sandboxes["a"], cfg, pa, ma, prompt, "a",
                            baseline, on_event)
    # Sequential runs: hand the VRAM from contestant A's server to B's when
    # both are local and different (the same handshake hybrid routing uses).
    loop_mod._hybrid_evict(cfg, url_a, url_b)
    rec_b = _run_contestant(root, sandboxes["b"], cfg, pb, mb, prompt, "b",
                            baseline, on_event)
    state = {
        "arena_id": arena_id,
        "prompt": prompt,
        "created": checkpoints_mod._now_iso(),
        "root": str(root),
        "contestants": {"a": rec_a, "b": rec_b},
        "sandboxes": {s: str(p) for s, p in sandboxes.items()},
        "baseline_files": sorted(baseline),
        "winner": None,
    }
    _save_last(root, state)
    _note(root, cfg, on_event, "arena_stage", phase="arena_done",
          arena_id=arena_id, ok_a=rec_a["ok"], ok_b=rec_b["ok"],
          label=f"Arena {arena_id} finished")
    return state


def _load_state(root, arena_id: str) -> dict:
    state = _LAST if isinstance(_LAST, dict) else None
    if state is None or str(state.get("arena_id")) != str(arena_id):
        state = last_arena(root)
    if state is None or str(state.get("arena_id")) != str(arena_id):
        raise ArenaError(f"No arena matches {arena_id!r} in this project.")
    return state


def apply_arena_winner(root, arena_id: str, winner: str = "a") -> dict:
    """Adopt one contestant's sandbox into the real workspace: a pre-apply
    checkpoint, the file copies (and the baseline deletions the winner made),
    then a post-apply checkpoint labeled 'Arena winner: <model>'. Returns the
    report; the stored comparison remembers the winner."""
    root = Path(root).resolve()
    winner = str(winner or "").strip().lower()
    if winner not in ("a", "b"):
        raise ArenaError("winner must be 'a' or 'b'.")
    state = _load_state(root, arena_id)
    rec = (state.get("contestants") or {}).get(winner) or {}
    sandbox = Path((state.get("sandboxes") or {}).get(winner) or "")
    if not sandbox.is_dir():
        raise ArenaError(f"The arena {arena_id} sandbox is gone; run it again.")
    model = str(rec.get("model") or winner)
    before = dict(checkpoints_mod.iter_workspace_files(root))
    # Freeze the pre-apply hashes NOW: `before` holds live workspace paths the
    # copy below overwrites, so hashing them afterwards would compare the edit
    # against itself and report nothing changed.
    before_h = checkpoints_mod._hash_map(before)
    pre = checkpoints_mod.create_checkpoint(
        root, f"Arena pre-apply ({arena_id})", mode="manual")
    after_files = dict(checkpoints_mod.iter_workspace_files(sandbox))
    copied: list[str] = []
    for rel, src in after_files.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(rel)
    # Files the winner deleted: only baseline files it no longer has, never
    # anything created in the real workspace after the arena ran.
    deleted: list[str] = []
    for rel in state.get("baseline_files") or []:
        if rel not in after_files and (root / rel).exists():
            try:
                (root / rel).unlink()
                deleted.append(rel)
            except OSError:
                pass
    after = dict(checkpoints_mod.iter_workspace_files(root))
    after_h = checkpoints_mod._hash_map(after)
    changed = sorted(rel for rel in set(before_h) | set(after_h)
                     if before_h.get(rel) != after_h.get(rel))
    post = checkpoints_mod.create_checkpoint(
        root, f"Arena winner: {model}", mode="manual")
    state["winner"] = winner
    state["applied"] = {"checkpoint": post["id"], "files_changed": changed,
                        "deleted": deleted}
    _save_last(root, state)
    _note(root, None, None, "arena_applied", arena_id=arena_id, winner=winner,
          model=model, files_changed=changed, checkpoint=post["id"],
          label=f"Arena winner: {model}")
    return {"ok": True, "arena_id": arena_id, "winner": winner, "model": model,
            "files_changed": changed, "copied": len(copied),
            "deleted": deleted, "pre_checkpoint": pre["id"],
            "post_checkpoint": post["id"]}


__all__ = ["ArenaError", "apply_arena_winner", "last_arena",
           "run_arena_comparison"]
