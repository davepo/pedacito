"""Background jobs for the GUI: open a project, index, check the model, and the
four ways of working (task, explain, auto, new).

Each job runs the same code the command line does (`loop.run_task`,
`plan.run_planned`, `cli._cmd_auto`, `scaffold.run_new`), so the GUI gets
planning, preflight checks, escalation, and post-edit tests for free. What the
GUI adds is plumbing: streamed tokens and lookup steps go to the chat, text
printed to the terminal goes to the terminal pane, the model's questions are
answered in the chat, and reviews are signed off in the page.
"""

from __future__ import annotations

import ctypes
import re
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from .. import cli as cli_mod
from .. import events as events_mod
from .. import explain as explain_mod
from .. import fileio, gitinfo
from .. import index as index_mod
from .. import loop, scaffold, steering, workspace
from .. import plan as plan_mod
from ..agent import Agent
from ..config import ConfigError
from ..llm import LLMClient, LLMError
from .core import STATE, Job, RUN_KINDS, args_for, remember_recent

MAX_TEXT_CHARS = 60000
_EDIT_BLOCK_RE = re.compile(
    r"^---\s*FILE:.*?^>>>>>>>\s*REPLACE[ \t]*$", re.MULTILINE | re.DOTALL)


# --- running jobs -------------------------------------------------------------

def spawn(kind: str, title: str, fn, *, echo=None) -> Job | None:
    """Run `fn(job)` on a background thread; None while another job runs.

    `echo` (when given) runs synchronously after the job slot is claimed and
    before the thread starts, so a submitted prompt reaches the chat
    transcript -- and the SSE stream -- without waiting for the worker."""
    job = STATE.begin_job(kind, title)
    if job is None:
        return None
    if echo is not None:
        try:
            echo()
        except Exception:
            pass                    # a chat echo must never block the job

    def worker():
        err = ""
        try:
            try:
                fn(job)
            except KeyboardInterrupt:
                STATE.add_message(
                    "system", "Stopped. Anything already written can be "
                    "undone with Restore (project menu).", kind="cancelled")
            except (ConfigError, LLMError) as e:
                err = str(e)
                if job.kind in RUN_KINDS:
                    STATE.add_message("assistant", err, kind="error",
                                      level="warn")
                STATE.add_message("error", err)
            except Exception as e:  # surface it in the UI, keep the server up
                traceback.print_exc()
                err = f"{type(e).__name__}: {e}"
                if job.kind in RUN_KINDS:
                    STATE.add_message("assistant", err, kind="error",
                                      level="warn")
                STATE.add_message("error", err)
        except KeyboardInterrupt:
            pass            # a Stop that arrived after the job body finished
        finally:
            while True:     # a late Stop must not leave the job slot taken
                try:
                    STATE.end_job(job, err)
                    break
                except KeyboardInterrupt:
                    continue
            # the turn is over: the thread on disk must match what the user
            # just saw, including anything the throttled saves held back
            STATE.save_active_thread(force=True)

    job.thread = threading.Thread(target=worker, daemon=True)
    job.thread.start()
    return job


def stop_job() -> bool:
    """Ask the running job to stop. The model client is told to drop an
    in-flight stream, and a KeyboardInterrupt (what Ctrl-C does on the command
    line) is raised in the worker thread."""
    with STATE.lock:
        job = STATE.job
        if job is None or not job.active or job.cancelled:
            return False
        job.cancelled = True
        client = getattr(job, "client", None) or STATE.client
        thread = job.thread
        if thread is not None and thread.is_alive() and thread.ident:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident), ctypes.py_object(KeyboardInterrupt))
    try:
        if client is not None:
            client.cancel()
    except Exception:
        pass
    STATE.bus.publish("job", job=job.id, state="stopping", kind=job.kind)
    return True


# --- wiring a model client to the page ------------------------------------------

class _Feed:
    """Routes one job's streamed tokens and lookup steps into the chat."""

    def __init__(self, client, job: Job):
        self.client = client
        self.job = job
        self.stream_mid = 0
        self.stream_site = None
        self.activity_mid = 0

    def token(self, delta: str) -> None:
        site = getattr(self.client, "site", "") or "answer"
        if not self.stream_mid or site != self.stream_site:
            msg = STATE.add_message("stream", "", site=site, job=self.job.id)
            self.stream_mid, self.stream_site = msg["id"], site
        STATE.append_message(self.stream_mid, delta)
        STATE.term_write(delta)

    def end_stream(self) -> int:
        mid, self.stream_mid, self.stream_site = self.stream_mid, 0, None
        return mid

    def finish(self, text: str = "", **fields):
        """Promote the current stream to the final assistant message.

        The streamed text is the fallback when the engine returns no final
        answer, so a table of tokens already on screen is never replaced by
        an empty bubble. With no stream, a normal assistant message is added.
        """
        mid = self.end_stream()
        text = (text or "").strip()
        if mid:
            if not text:
                text = STATE.message_text(mid).strip()
            if text:
                fields["text"] = text
            STATE.set_message(mid, role="assistant", **fields)
            return mid
        if text:
            return STATE.add_message("assistant", text, **fields)["id"]
        return 0

    def activity(self, line: str) -> None:
        if not self.activity_mid:
            msg = STATE.add_message("activity", "", job=self.job.id)
            self.activity_mid = msg["id"]
            STATE.append_message(self.activity_mid, line)
        else:
            STATE.append_message(self.activity_mid, "\n" + line)

    def event(self, kind: str, f: dict) -> None:
        STATE.note_event(kind, f)      # feeds the ACTIVITY & PLAN panel
        if kind == "step":
            self.activity(f"{f.get('n', '?')}. {f.get('action', '')} "
                          f"{f.get('detail', '')}".rstrip())
        elif kind == "recovery":
            self.activity(f"recovery: {f.get('signal', '')}")
        elif kind == "steer":
            self.activity("your note was delivered to the model")
        elif kind == "fold":
            self.activity("context folded to stay inside the window")
        elif kind == "degrade":
            self.activity("degraded: " + "; ".join(map(str, f.get("notices", []))))
        elif kind == "tests":
            self.activity(f"tests: {'passed' if f.get('ok') else 'failed'}")
        elif kind == "call":
            STATE.bus.publish("usage", usage=STATE.status()["usage"])


@contextmanager
def session(p: SimpleNamespace, job: Job, *, interactive: bool, lock: bool):
    """Attach the page to a client/config for the length of one job."""
    cfg, client, root = p.cfg, p.client, p.root
    job.client = client
    feed = _Feed(client, job)
    saved_cb, saved_echo = client.stream_callback, client.echo_stream
    client.stream_callback = feed.token
    elog = events_mod.get(root, cfg)
    elog.listeners.append(feed.event)
    held = False
    had = {k: cfg.__dict__.get(k, _MISSING)
           for k in ("_steer", "_interactive", "_ask_user")}
    if interactive:
        cfg._interactive = True
        cfg._ask_user = STATE.ask_user
        if cfg.allow_steering:
            steering.drain(STATE.steer_queue)
            cfg._steer = STATE.steer_queue
    try:
        if lock:
            try:
                workspace.acquire_lock(root)
                held = True
            except workspace.ProjectBusy as e:
                raise ConfigError(str(e)) from e
        yield feed
    finally:
        client.stream_callback, client.echo_stream = saved_cb, saved_echo
        try:
            elog.listeners.remove(feed.event)
        except ValueError:
            pass
        for k, v in had.items():
            if v is _MISSING:
                cfg.__dict__.pop(k, None)
            else:
                cfg.__dict__[k] = v
        if held:
            workspace.release_lock(root)


_MISSING = object()


def _project(need_index: bool = True) -> SimpleNamespace:
    """A snapshot of the open project's pieces for a worker to use unlocked."""
    with STATE.lock:
        if STATE.root is None:
            raise ConfigError("Open a project first.")
        if need_index and STATE.agent is None:
            raise ConfigError("This project has no index yet -- build it "
                              "first (the banner in the chat has a button).")
        return SimpleNamespace(root=STATE.root, args=STATE.args, cfg=STATE.cfg,
                               client=STATE.client, idx=STATE.idx,
                               agent=STATE.agent)


def _swap_index(idx) -> None:
    with STATE.lock:
        STATE.idx = idx
        STATE.index_gen += 1
        if STATE.agent is not None:
            STATE.agent.index = idx
    STATE.apply_index(idx)


def _refresh_index(p: SimpleNamespace) -> None:
    """Bring the index in line with disk after a job that wrote many files."""
    idx = cli_mod._refresh(STATE.idx or p.idx, p.args, p.cfg, p.root,
                           verbose=False)
    _swap_index(idx)


def _prose(reply: str) -> str:
    """The model's reply without its search/replace blocks."""
    text = _EDIT_BLOCK_RE.sub("", reply or "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def compose(text: str, attachments: list | None) -> str:
    """The typed text plus any attached files, capped to a sane size."""
    parts = [(text or "").strip()]
    for a in attachments or []:
        name = str(a.get("name") or "attachment")
        body = str(a.get("text") or "").strip()
        if body:
            parts.append(f"--- attached file: {name} ---\n{body}")
    out = "\n\n".join(p for p in parts if p)
    if len(out) > MAX_TEXT_CHARS:
        STATE.add_message("system", f"Input truncated from {len(out):,} to "
                          f"{MAX_TEXT_CHARS:,} characters.")
        out = out[:MAX_TEXT_CHARS]
    return out


def _echo_user(mode: str, text: str, attachments: list | None,
               opts: dict | None = None) -> None:
    if opts is not None and opts.get("_echoed"):
        return                      # already recorded synchronously by spawn
    shown = text if len(text) <= 4000 else text[:4000] + "\n[...]"
    STATE.add_message("user", shown, mode=mode, attachments=[
        {"name": str(a.get("name", "")), "chars": len(str(a.get("text", "")))}
        for a in attachments or []])


def echo_submission(mode: str, body: dict) -> None:
    """Record the submitted prompt before the worker thread starts.

    The job functions also call `_echo_user`, but they see the `_echoed`
    marker and skip it, so the prompt appears exactly once -- immediately,
    even while the model is still warming up."""
    text = str(body.get("text") or "")
    attachments = body.get("attachments") or []
    plan_file = str(body.get("plan_file") or "").strip()
    if not text.strip() and not attachments and not (
            mode == "auto" and plan_file):
        return
    shown = text or (f"plan: {plan_file}" if mode == "auto" and plan_file
                     else "")
    _echo_user(mode, shown, attachments)


# --- project ------------------------------------------------------------------

def open_project(path: str, profile: str, announce: bool = True) -> None:
    """Make a directory the active project: load its config, ensure workspace
    scaffolding, load (never build) its index, and test the server cheaply."""
    p = Path(path).expanduser()
    if not p.is_dir():
        raise ConfigError(f"not a directory: {p}")
    args = args_for(str(p), profile)
    cfg = cli_mod._config(args)
    root = cli_mod._root(args)
    STATE.set_root(root)            # adopt + auto-detect any existing index
    workspace.ensure(root, cfg.manage_gitignore, cfg.create_ignore_file)

    idx = index_mod.Index.load(root)
    if idx is not None:
        idx = cli_mod._refresh(idx, args, cfg, root, verbose=False)

    client = LLMClient(cfg)
    agent = Agent(idx, client, cfg) if idx is not None else None

    with STATE.lock:
        STATE.args, STATE.cfg = args, cfg
        STATE.client, STATE.idx, STATE.agent = client, idx, agent
        STATE.profile_request = profile or ""
        STATE.index_gen += 1
        STATE.model_checked, STATE.model_status, STATE.model_ok = False, "", None
        STATE.review = None
        STATE.plan_path = None
        STATE.last_auto = {}
    STATE.apply_index(idx)
    remember_recent(str(root))
    _probe_server(client, cfg)
    if announce:
        bits = [f"model {cfg.model}"]
        if cfg.active_profile:
            bits.insert(0, f"profile {cfg.active_profile}")
        state = (f"index: {len(idx.files)} files" if idx is not None
                 else "no index yet")
        STATE.add_message("system", f"Opened {root} \u2014 {', '.join(bits)}; "
                          f"{state}.", kind="opened",
                          needs_index=idx is None)


def _probe_server(client, cfg) -> None:
    """Cheap reachability check (no model load) for the header light."""
    try:
        loaded = client.check(warn=False).split(", ")
    except LLMError as e:
        with STATE.lock:
            STATE.model_ok = False
            STATE.model_status = str(e).splitlines()[0][:200]
        return
    listed = cfg.model in loaded
    with STATE.lock:
        STATE.model_ok = listed
        STATE.model_status = ("server reachable" if listed else
                              f"server reachable, but model '{cfg.model}' is "
                              "not in its list")


def check_model(job: Job) -> None:
    """Server reachable and the model actually generates (loads it), like
    `pedacito config --check`."""
    p = _project(need_index=False)
    loaded = p.client.check(warn=False)
    p.client.warmup()
    with STATE.lock:
        STATE.model_checked, STATE.model_ok = True, True
        STATE.model_status = (
            f"reachable and generating \u2014 window {p.cfg.context_tokens:,} "
            f"({p.cfg.sources.get('context_tokens', 'default')}) \u00b7 reply "
            f"cap {p.cfg.max_tokens_step:,} \u00b7 models: {loaded}")
    STATE.add_message("system", STATE.model_status, kind="model")


def run_calibrate(job: Job, opts: dict) -> None:
    """`pedacito calibrate` for one profile: run the scripted mini-suite
    against its model and save the measured card. The suite's progress prints
    to stdout, which the GUI tee streams into the terminal pane; the score
    lands in the chat when it finishes."""
    from .. import config as config_mod
    from .. import modelcard
    name = str(opts.get("profile") or "").strip()
    quick = str(opts.get("mode") or "quick") != "full"
    with STATE.lock:
        root = STATE.root
    cfg = config_mod.load(root=root, profile=name or None,
                          require_hardware=False, warn_budgets=False)
    label = name or cfg.active_profile or "the current config"
    STATE.add_message(
        "system", f"Calibrating '{label}' \u2014 {cfg.model} at {cfg.base_url} "
        f"({'quick' if quick else 'full'} suite). Progress streams to the "
        "TERMINAL tab.", kind="calibrating")
    client = LLMClient(cfg)
    client.check(warn=False)
    client.warmup()
    card = modelcard.calibrate(client, cfg, verbose=True, quick=quick)
    path = modelcard.save_card(card)
    m = card.get("measured") or {}
    caps = ", ".join(
        f"{k[4:]}={'yes' if v else 'no' if v is False else '?'}"
        for k, v in sorted(m.items()) if k.startswith("cap_"))
    STATE.add_message(
        "system", f"Calibration of '{label}' finished \u2014 "
        f"overall score {m.get('overall', '?')} "
        f"in {card.get('seconds', '?')} s. Card saved to {path}."
        + (f" Capabilities: {caps}." if caps else ""), kind="calibrated")


def build_index(job: Job, summarise: bool = True) -> None:
    """Build the index, like `pedacito index` (summaries cost one model call
    per unit, so it is an explicit action)."""
    p = _project(need_index=False)
    with session(p, job, interactive=False, lock=False):
        idx = index_mod.build(
            p.args.paths, p.cfg, client=p.client if summarise else None,
            root=p.root, summarise=summarise,
            matcher=cli_mod._matcher(p.args, p.root))
    with STATE.lock:
        STATE.idx = idx
        STATE.agent = Agent(idx, p.client, p.cfg)
        STATE.index_gen += 1
    STATE.apply_index(idx)
    STATE.add_message("system", f"Indexed {len(idx.chunks)} units across "
                      f"{len(idx.files)} files.", kind="indexed")


def restore_snapshot(name: str = "") -> tuple[bool, str]:
    """Copy a backup snapshot back over the working tree (`pedacito restore`).
    Files created since the snapshot stay on disk."""
    with STATE.lock:
        root, cfg, idx = STATE.root, STATE.cfg, STATE.idx
        if STATE.job is not None and STATE.job.active:
            return False, "A job is running; wait for it or stop it first."
    if root is None:
        return False, "No project is open."
    snaps = workspace.list_backups(root)
    if not snaps:
        return False, "No snapshots exist for this project yet."
    snap = snaps[0]
    if name:
        found = [s for s in snaps if s.name == name]
        if not found:
            return False, f"Snapshot {name!r} no longer exists."
        snap = found[0]
    restored = workspace.restore(root, snap)
    if idx is not None and cfg is not None:
        for rel in restored:
            idx = index_mod.reindex_file(idx, rel, cfg)
        _swap_index(idx)
    detail = f"Restored {len(restored)} file(s) from {snap.name}."
    STATE.add_message("system", detail, kind="restored")
    return True, detail


# --- the four ways of working ----------------------------------------------------

def run_task(job: Job, opts: dict) -> None:
    """`ask`: gather, edit, and review or write. `write` is "review" (sign off
    in the page) or "apply" (write now, backups first)."""
    p = _project()
    task = compose(opts.get("text", ""), opts.get("attachments"))
    if not task:
        raise ConfigError("Describe the change you want first.")
    _echo_user("task", opts.get("text", "") or task, opts.get("attachments"),
               opts)
    write = opts.get("write", "review")
    reviewing, apply = write == "review", write == "apply"
    plan_pref = opts.get("plan", "auto")
    steps = int(opts.get("steps") or 0)
    cfg, client, root = p.cfg, p.client, p.root

    saved_steps = cfg.max_steps
    if steps > 0:
        cfg.max_steps = steps
    try:
        with session(p, job, interactive=True, lock=apply) as feed:
            if opts.get("reindex"):
                _refresh_index(p)
                p.agent.index = STATE.idx
            client.warmup()
            idx = STATE.idx
            want_plan = plan_pref == "always" or (
                plan_pref == "auto" and plan_mod.should_plan(task, idx, cfg))
            if want_plan and not reviewing:
                started = _now()
                code = plan_mod.run_planned(root, cfg, client, idx, task,
                                            apply=apply)
                _note_plan_dir(root, started)
                if apply:
                    _refresh_index(p)
                feed.end_stream()
                STATE.add_message(
                    "assistant", "Plan finished." if code == 0 else
                    "Plan stopped before every step finished \u2014 see the "
                    "Plan tab and terminal.",
                    kind="result", level="ok" if code == 0 else "warn",
                    plan=True, stats=_usage_stats(client))
                return
            tr = loop.run_task(root, cfg, client, p.agent, idx, task,
                               apply=apply, review=reviewing)
            _swap_index(tr.idx)
            feed.end_stream()
            _post_task_result(tr, client, reviewing)
    finally:
        cfg.max_steps = saved_steps


def _now() -> float:
    return time.time()


def _note_plan_dir(root: Path, since: float) -> None:
    d = Path(root) / ".pedacito" / "plans"
    try:
        newest = max((f for f in d.glob("*.md") if f.stat().st_mtime >= since - 1),
                     key=lambda f: f.stat().st_mtime, default=None)
    except OSError:
        newest = None
    if newest is not None:
        with STATE.lock:
            STATE.plan_path = newest


def _usage_stats(client) -> dict:
    s = client.stats
    return {"calls": s["calls"], "tokens_out": s["completion_tokens"],
            "tokens_in": s.get("prompt_tokens", 0),
            "cost": s.get("cost", 0.0), "seconds": s.get("seconds", 0.0)}


def _post_task_result(tr, client, reviewing: bool) -> None:
    results = [{"file": r.file, "ok": r.ok, "message": r.message}
               for r in tr.results
               if r.ok or "renamed" not in (r.message or "")]
    ok_n = sum(1 for r in tr.results if r.ok)
    if reviewing:
        written = sum(1 for r in tr.results
                      if r.ok and (r.message or "").startswith("applied via review"))
    else:
        written = ok_n if tr.applied else 0
    prose = _prose(tr.reply)
    if written:
        level, head = "ok", f"Wrote {written} file(s)."
    elif reviewing and ok_n:
        level, head = "info", "Nothing was written."
    elif tr.results:
        level, head = "warn", "No edit was applied."
    else:
        level, head = "info", ""
    if not prose:
        # A reply made only of edit blocks still needs a visible result card.
        prose = head or "The task finished. See the diff and activity panes."
    STATE.add_message("assistant", prose, kind="result", level=level,
                      head=head, results=results, stats=tr.stats or
                      _usage_stats(client))


def run_explain(job: Job, opts: dict) -> None:
    """`explain`: a prose answer about the code, streamed into the chat."""
    p = _project()
    question = compose(opts.get("text", ""), opts.get("attachments"))
    if not question:
        raise ConfigError("Ask a question first.")
    _echo_user("explain", opts.get("text", "") or question,
               opts.get("attachments"), opts)
    with session(p, job, interactive=False, lock=False) as feed:
        p.client.warmup()
        answer = explain_mod.answer(p.agent, question, verbose=True,
                                    stream=True)
    text = (answer or "").strip()
    if not text and feed.stream_mid:
        text = STATE.message_text(feed.stream_mid).strip()
    if not text:
        text = "The model finished without returning an answer."
    feed.finish(text, kind="answer", stats=_usage_stats(p.client))


def auto_plan_only(opts: dict) -> bool:
    """True only when the user explicitly chose the Auto "Plan only" option.

    The default is to apply changes (the GUI mirrors `pedacito auto --apply`).
    The `run` alias keeps older clients that sent run="plan" working."""
    if "plan_only" in opts:
        value = opts.get("plan_only")
        if isinstance(value, str):
            return value.strip().lower() in (
                "1", "true", "yes", "on", "plan", "plan-only", "plan_only")
        return bool(value)
    run = str(opts.get("run") or "").strip().lower()
    return run in ("plan", "plan-only", "plan_only")


def _files_in_backups(dirs) -> list[str]:
    """Relative paths a set of backup snapshots recorded as written/created."""
    names: set[str] = set()
    for d in dirs:
        try:
            names.update(p.relative_to(d).as_posix()
                         for p in d.rglob("*") if p.is_file())
        except OSError:
            continue
        try:
            manifest = fileio.created_manifest(d)
            names.update(line.strip() for line in
                         manifest.read_text(encoding="utf-8",
                                            errors="replace").splitlines()
                         if line.strip())
        except OSError:
            pass
    return sorted(names)


def _auto_changed_files(root: Path, before_backups: set,
                        git_status: dict) -> list[str]:
    """The files this auto run wrote, from its task backups (falling back to
    git's dirty/untracked list when the project has no backup snapshots)."""
    new_dirs = [d for d in workspace.list_backups(root)
                if d not in before_backups]
    names = _files_in_backups(new_dirs)
    if not names and not new_dirs:
        # No task snapshot was taken (e.g. a custom runner): fall back to the
        # repository's post-run dirty/untracked files.
        names = sorted(set(git_status.get("dirty") or [])
                       | set(git_status.get("untracked") or []))
    return names


def run_auto(job: Job, opts: dict) -> None:
    """`auto`: turn a goal into a task list, then implement it task by task.

    Applying is the default -- each step writes its changes and runs the
    configured verification, like `pedacito auto --apply`. The user opts out
    explicitly with the "Plan only" option (or run="plan" from older clients).

    An applying Auto run is also an explicit acceptance of the tool trust
    boundary: the gather phase gets `allow_write` (move/write/delete/rename)
    and `allow_shell` (run/diagnose) for the run, even when the global config
    leaves those opt-ins off. The original values are restored afterwards;
    Plan-only keeps the user's real settings."""
    p = _project()
    goal = compose(opts.get("text", ""), opts.get("attachments"))
    plan_file = (opts.get("plan_file") or "").strip()
    if not goal and not plan_file:
        raise ConfigError("Give a goal, attach a spec, or choose a plan file.")
    _echo_user("auto", opts.get("text", "") or goal or f"plan: {plan_file}",
               opts.get("attachments"), opts)
    # Applying is the default: the GUI behaves like `pedacito auto --apply`
    # unless the user explicitly ticked "Plan only".
    apply = not auto_plan_only(opts)
    ns = SimpleNamespace(
        task=goal or None, spec=None, plan=plan_file or None, apply=apply,
        verify=(opts.get("verify") or None),
        repair=_int_or_none(opts.get("repair")),
        diagnostics=bool(opts.get("diagnostics")),
        worktree=(opts.get("worktree") or "").strip(),
        max_tasks=_int_or_none(opts.get("max_tasks")),
        max_minutes=_int_or_none(opts.get("max_minutes")),
        keep_going=bool(opts.get("keep_going")),
        auto_install=bool(opts.get("auto_install")),
        force=bool(opts.get("force")))
    with STATE.lock:
        STATE.last_auto = {"text": opts.get("text", ""), "plan_file": plan_file}
    cfg, client, root = p.cfg, p.client, p.root
    plan_path = Path(plan_file) if plan_file else root / cfg.auto_plan_file
    if not plan_path.is_absolute():
        plan_path = (root / plan_path).resolve()
    with STATE.lock:
        STATE.plan_path = plan_path
    # Snapshot the backup names before the run so the files this run writes
    # can be reported afterwards. Forcing cfg.backup on while applying makes
    # the loop take its usual pre-edit snapshot of every file it touches,
    # even if the user normally turns backups off -- otherwise Restore would
    # have nothing to roll back to.
    before_backups = set(workspace.list_backups(root)) if apply else set()
    saved_backup = getattr(cfg, "backup", True)
    # An applying Auto run accepts the tool trust boundary: the lookup phase
    # may move, write, delete and run commands even when the global opt-ins
    # are off. Plan-only keeps the user's real settings (it must not write).
    auto_optins = ("allow_write", "allow_shell")
    saved_optins = {name: cfg.__dict__.get(name, _MISSING)
                    for name in auto_optins}
    if apply:
        cfg.backup = True
        for name in auto_optins:
            setattr(cfg, name, True)
    try:
        with session(p, job, interactive=False, lock=apply) as feed:
            client.warmup()
            code = cli_mod._cmd_auto(ns, cfg, client, STATE.idx, root)
            feed.end_stream()
    finally:
        if apply:
            cfg.backup = saved_backup
            for name, value in saved_optins.items():
                if value is _MISSING:
                    cfg.__dict__.pop(name, None)
                else:
                    setattr(cfg, name, value)
    git_status: dict = {}
    changed: list[str] = []
    if apply:
        _refresh_index(p)
        git_status = gitinfo.status(root)
        changed = _auto_changed_files(root, before_backups, git_status)
    done, total = plan_progress(plan_path)
    if not apply and code == 0:
        STATE.add_message(
            "assistant", f"Plan ready: {total} task(s). Review or edit it in "
            "the Plan tab, then run it.", kind="plan_ready", level="info",
            auto=True, files=[])
    else:
        text = (f"Auto finished: {done}/{total} task(s) done."
                if code == 0 else
                f"Auto stopped: {done}/{total} task(s) done; see the Plan "
                "tab and terminal.")
        if changed:
            shown = changed[:40]
            text += (f"\n\nApplied changes to {len(changed)} file(s):\n"
                     + "\n".join(f"- {name}" for name in shown))
            if len(changed) > len(shown):
                text += f"\n- ... and {len(changed) - len(shown)} more"
        elif apply and code == 0:
            text += "\n\nNo files were changed."
        STATE.add_message(
            "assistant", text, kind="result",
            level="ok" if code == 0 else "warn", auto=True,
            files=changed, changed=changed, git=git_status,
            stats=_usage_stats(client))


def _int_or_none(value) -> int | None:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


def _int_or(value, default: int) -> int:
    got = _int_or_none(value)
    return default if got is None else got


def run_new(job: Job, opts: dict) -> None:
    """`new`: plan a project from a description and build it from scratch."""
    spec_text = compose(opts.get("text", ""), opts.get("attachments"))
    project = (opts.get("project") or "").strip()
    if not project:
        raise ConfigError("Choose a directory for the new project.")
    if not spec_text:
        raise ConfigError("Describe the project (or attach a spec file).")
    target = Path(project).expanduser().resolve()
    _echo_user("new", opts.get("text", "") or spec_text,
               opts.get("attachments"), opts)
    mode = opts.get("build", "preview")       # preview | plan | build | review
    parent = target
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    with STATE.lock:
        profile = STATE.profile_request
    args = args_for(str(parent), profile)
    cfg = cli_mod._config(args)
    client = LLMClient(cfg)
    client.check(warn=False)
    spec = scaffold.load_spec(prompt=spec_text, max_chars=cfg.max_spec_chars)
    p = SimpleNamespace(cfg=cfg, client=client, root=target)
    with session(p, job, interactive=False, lock=False) as feed:
        client.warmup()
        if mode != "preview":
            workspace.log_event(target, "command: gui new")
        code = scaffold.run_new(
            target, cfg, client, spec,
            apply=mode in ("build",), review=mode == "review",
            plan_only=mode == "plan", force=bool(opts.get("force")),
            verbose=True, complex=bool(opts.get("complex")),
            verify=opts.get("verify") or cfg.auto_verify,
            repair_rounds=_int_or(opts.get("repair"), cfg.auto_repair_rounds),
            max_tasks=_int_or(opts.get("max_tasks"), cfg.auto_max_tasks),
            max_minutes=_int_or(opts.get("max_minutes"), cfg.auto_max_minutes),
            keep_going=bool(opts.get("keep_going")),
            auto_install=bool(opts.get("auto_install") or cfg.auto_install_deps))
        feed.end_stream()
    wrote = mode != "preview" and target.is_dir() and any(
        c.name not in (".pedacito",) for c in target.iterdir())
    if mode == "preview":
        STATE.add_message(
            "assistant", "That was a preview: nothing was written. Pick "
            "\u201cBuild\u201d to create the project.", kind="result",
            level="info", stats=_usage_stats(client))
    elif code != 0:
        STATE.add_message(
            "assistant", "The build finished with failures \u2014 see the "
            "terminal.", kind="result", level="warn",
            stats=_usage_stats(client))
    else:
        STATE.add_message("assistant", f"Project created at {target}.",
                          kind="result", level="ok",
                          stats=_usage_stats(client))
    if wrote:
        open_project(str(target), profile, announce=True)


# --- plan files -----------------------------------------------------------------

_PLAN_LINE = re.compile(r"^\s*[-*]\s*\[(.)\]\s*(.*\S)?\s*$")
_STATUS = {" ": "todo", "x": "done", "X": "done", "!": "failed", "-": "skipped",
           "~": "skipped"}


def parse_plan(text: str) -> list[dict]:
    out = []
    for i, line in enumerate(text.splitlines()):
        m = _PLAN_LINE.match(line)
        if m:
            out.append({"line": i, "status": _STATUS.get(m.group(1), "todo"),
                        "title": (m.group(2) or "").strip()})
    return out


def plan_progress(path: Path) -> tuple[int, int]:
    try:
        tasks = parse_plan(Path(path).read_text(encoding="utf-8",
                                                errors="replace"))
    except OSError:
        return 0, 0
    return sum(1 for t in tasks if t["status"] == "done"), len(tasks)
