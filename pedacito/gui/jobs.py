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
from .. import security as security_mod
from .. import update as update_mod
from ..agent import Agent
from ..config import ConfigError
from ..llm import CancelledError, LLMClient, LLMError
from .core import STATE, Job, RUN_KINDS, args_for, remember_recent

MAX_TEXT_CHARS = 60000
_EDIT_BLOCK_RE = re.compile(
    r"^---\s*FILE:.*?^>>>>>>>\s*REPLACE[ \t]*$", re.MULTILINE | re.DOTALL)

# Status shown when Build Index runs on a workspace with 0 indexable files.
# The empty index is valid and the project is marked indexed; the user is told
# how to populate it instead of being left on a dead "no index" banner.
EMPTY_INDEX_MESSAGE = ("Index ready (0 files in workspace — run a Task or Auto "
                       "prompt to create files).")


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
        # One run-scoped telemetry collector per job: every LLM call the job
        # makes folds into it (events.note_call), and the result cards read
        # the aggregate while the run is still open.
        events_mod.begin_run()
        try:
            try:
                fn(job)
            except KeyboardInterrupt:
                STATE.add_message(
                    "system", "Stopped. Anything already written can be "
                    "undone with Restore (project menu).", kind="cancelled")
            except CancelledError:
                # A streamed call aborted through its cancel_event (the clean
                # path Stop uses); show the user-facing stop state instead of
                # an opaque transport error.
                job.stop_reason = job.stop_reason or "Stopped by user"
                job.stream_phase = "stopped"
                STATE.add_message(
                    "system", "Stopped by user. Anything already written can "
                    "be undone with Restore (project menu).",
                    kind="cancelled")
            except (ConfigError, LLMError) as e:
                err = str(e)
                if job.kind in RUN_KINDS:
                    STATE.add_message("assistant", err, kind="error",
                                      level="warn")
                STATE.add_message("error", err)
            except SystemExit as e:
                # index.build signals a fatal condition (e.g. the model cannot
                # respond) with SystemExit. It is a BaseException, so without
                # this handler it would kill the worker thread silently and
                # leave the page stuck with no error shown.
                err = str(e).strip() or "The job stopped before finishing."
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
            job.telemetry = events_mod.end_run()
            while True:     # a late Stop must not leave the job slot taken
                try:
                    STATE.end_job(job, err)
                    break
                except KeyboardInterrupt:
                    continue
            # the turn is over: the thread on disk must match what the user
            # just saw, including anything the throttled saves held back
            STATE.save_active_thread(force=True)

    job.thread = threading.Thread(target=worker, daemon=True,
                                  name=f"pedacito-gui-job-{job.id}")
    job.thread.start()
    return job


def stop_job() -> bool:
    """Ask the running job to stop. The model client is told to drop an
    in-flight stream, and a KeyboardInterrupt (what Ctrl-C does on the command
    line) is raised in the worker thread.

    The job's `cancel_event` is set first: the LLM transport checks it on every
    SSE line and closes the live socket, so a 3,000-token generation stops
    within milliseconds instead of finishing.
    """
    with STATE.lock:
        job = STATE.job
        if job is None or not job.active or job.cancelled:
            return False
        job.cancelled = True
        job.stop_reason = "Stopped by user"
        job.stream_phase = "stopped"
        event = getattr(job, "cancel_event", None)
        if event is not None:
            try:
                event.set()
            except Exception:
                pass
        client = getattr(job, "client", None) or STATE.client
        if client is not None and hasattr(client, "cancel_event"):
            try:
                client.cancel_event = event
            except Exception:
                pass
        thread = job.thread
        if thread is not None and thread.is_alive() and thread.ident:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident), ctypes.py_object(KeyboardInterrupt))
    try:
        if client is not None:
            client.cancel()
    except Exception:
        pass
    STATE.bus.publish("job", job=job.id, state="stopping", kind=job.kind,
                      reason=job.stop_reason)
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
        if kind in ("plan_updated", "plan_archived"):
            # A live checklist change: render it now (the page also polls
            # /api/state, but the event lands the instant the box changes).
            STATE.bus.publish("plan", path=str(f.get("path") or ""),
                              reason=str(f.get("reason") or kind))
            return
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
        elif kind == "specialist_routed":
            self.activity(f"specialist router: {f.get('profile', '?')} "
                          f"({f.get('model', '?')}) for "
                          f"'{f.get('specialty', '?')}'")
        elif kind == "hybrid_stage":
            self.activity(str(f.get("label") or
                              f"Hybrid Stage {f.get('stage', '?')}: "
                              f"{f.get('model', '?')}"))
        elif kind == "arena_stage":
            phase = str(f.get("phase") or "")
            if phase == "start":
                self.activity(str(f.get("label") or
                                  f"Arena {f.get('side', '?')}: "
                                  f"{f.get('model', '?')}"))
            elif phase == "done":
                self.activity(
                    f"arena {str(f.get('side', '?')).upper()} finished: "
                    f"{'edits landed' if f.get('ok') else 'no edits'} "
                    f"({f.get('files_changed', 0)} file(s), "
                    f"{f.get('elapsed_sec', 0)}s)")
            elif phase in ("arena_start", "arena_done"):
                self.activity(str(f.get("label") or "arena"))
        elif kind == "model_swap":
            self.activity(f"model swap: {f.get('unloaded') or '-'} -> "
                          f"{f.get('loaded', '?')} ({f.get('elapsed_s', 0)}s)")
        elif kind == "call":
            STATE.bus.publish("usage", usage=STATE.status()["usage"])
        elif kind == "complex_auto":
            self.activity(str(f.get("text") or
                              "Auto-enabled Complex scaffolding."))
        elif kind == "security_redacted":
            # Auto-redaction happened inside the LLM transport: surface the
            # notification in the page, never the raw secret.
            message = str(f.get("message") or
                          "Auto-redacted secrets before sending.")
            self.activity("[security] " + message)
            STATE.bus.publish("security", message=message)
        elif kind == "security_notice":
            # A post-run scan found something: the banner refreshes from
            # /api/state (the cache was invalidated by the scan).
            STATE.bus.publish("security",
                              message=str(f.get("message") or ""))


@contextmanager
def session(p: SimpleNamespace, job: Job, *, interactive: bool, lock: bool):
    """Attach the page to a client/config for the length of one job."""
    cfg, client, root = p.cfg, p.client, p.root
    job.client = client
    feed = _Feed(client, job)
    saved_cb, saved_echo = client.stream_callback, client.echo_stream
    saved_cancel = getattr(client, "cancel_event", _MISSING)
    try:
        client.cancel_event = job.cancel_event
    except Exception:
        pass
    client.stream_callback = feed.token
    elog = events_mod.get(root, cfg)
    elog.listeners.append(feed.event)
    # Route per-call usage AND the pre-request heartbeat into the event log
    # (the scratchpad Ask has no agent to attach its own client). A New
    # preview's target does not exist yet: emitting into its EventLog would
    # create the very folder the preview promised not to touch, so attach
    # only for a real workspace (root None is the diskless scratchpad log).
    if root is None or Path(root).is_dir():
        events_mod.attach(client, elog)
    held = False
    had = {k: cfg.__dict__.get(k, _MISSING)
           for k in ("_steer", "_interactive", "_ask_user", "_cancel_event")}
    cfg._cancel_event = job.cancel_event
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
            if saved_cancel is _MISSING:
                delattr(client, "cancel_event")
            else:
                client.cancel_event = saved_cancel
        except Exception:
            pass
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


def _global_project(profile: str = "") -> SimpleNamespace:
    """A model-only workspace for Ask with no project open.

    The question is answered directly by `LLMClient(cfg)` on the selected
    global profile; there is no index and no project root, and nothing is
    written. The header reports it as the scratchpad."""
    cfg = STATE.global_config(profile=profile or None)
    return SimpleNamespace(root=None, args=None, cfg=cfg,
                           client=LLMClient(cfg), idx=None, agent=None)


def _evict_other_servers(cfg) -> None:
    """Unload models resident on OTHER local servers before a run.

    The handshake is a no-op with one server, with `auto_evict_servers` off,
    or when every other server is asleep. It runs in the job's worker thread,
    so a slow unload never blocks the page."""
    try:
        from .. import servers
        result = servers.evict_other_local_servers(
            getattr(cfg, "active_profile", "") or cfg.base_url, cfg)
    except Exception:
        return
    evicted = [r for r in result.get("evicted", []) if r]
    if evicted:
        names = ", ".join(str(r.get("server") or r.get("base_url") or "?")
                          for r in evicted)
        STATE.add_message(
            "system", f"Unloaded {names} from another local server so this "
                      "run can use the VRAM/RAM.", kind="servers")


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


def _ensure_index(job: Job) -> None:
    """Zero-friction runs: a Task or Auto started in an unindexed workspace
    builds the index on demand instead of refusing with "build it first".
    Quick (no summaries -- that stays an explicit action), and an empty
    folder is fine: 0 files is a valid index the run can populate."""
    with STATE.lock:
        if STATE.root is None:
            raise ConfigError("Open a project first.")
        if STATE.agent is not None:
            return
    build_index(job, summarise=False)


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
        # A project that opts out of update checks must not keep showing the
        # startup check's banner (pedacito.update).
        if not update_mod.updates_enabled(cfg):
            STATE.update = None
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


def switch_profile(profile: str) -> dict:
    """Switch the active profile IN PLACE, synchronously.

    Re-opening the whole project as a background "project" job made the Run
    button turn into Stop while the header select stayed open -- and a slow
    model probe could leave that job state visible long after the switch.
    Switching a profile does not touch the index or the plan: swap the config
    and client, keep the loaded index, and never claim the task-runner slot.
    """
    with STATE.lock:
        root = STATE.root
        job = STATE.job
        idx = STATE.idx
        if root is None:
            raise ConfigError("Open a project first.")
        if job is not None and job.active:
            raise ConfigError("A job is running. Stop it before switching "
                              "profiles.")
    args = args_for(str(root), profile)
    cfg = cli_mod._config(args)
    name = str(profile or "").strip()
    if name and name not in (cfg.profiles or {}):
        raise ConfigError(f"no profile named {name!r}")
    client = LLMClient(cfg)
    agent = Agent(idx, client, cfg) if idx is not None else None
    with STATE.lock:
        STATE.args, STATE.cfg = args, cfg
        STATE.client, STATE.agent = client, agent
        STATE.profile_request = name
        STATE.model_checked, STATE.model_ok = False, None
        STATE.model_status = "profile switched; click the model chip to check"
        STATE.index_gen += 1
    STATE.bus.publish("status", profile=cfg.active_profile, model=cfg.model)
    return {"ok": True, "profile": cfg.active_profile, "model": cfg.model,
            "profiles": sorted(cfg.profiles or {}),
            "auto_route_specialties": bool(cfg.auto_route_specialties),
            "disable_thinking": bool(cfg.disable_thinking)}


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
    lands in the chat when it finishes. Hosted profiles are refused (the
    endpoint checks too; this keeps a direct job call honest)."""
    from .. import config as config_mod
    from .. import modelcard
    from . import settings_api
    name = str(opts.get("profile") or "").strip()
    quick = str(opts.get("mode") or "quick") != "full"
    hosted_err = settings_api.hosted_profile_error(name) if name else ""
    if hosted_err:
        raise ConfigError(hosted_err)
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


def run_calibrate_all(job: Job, opts: dict) -> None:
    """`pedacito calibrate --all` for the GUI: every LOCAL profile in turn,
    hosted profiles excluded, fresh cards skipped unless `recalibrate`.
    Each card is saved as soon as it finishes and the model is unloaded
    between profiles; `job.title` names the profile in flight so the modal
    (polling /api/state) can show live progress, and the `calibration` map
    marks each profile done."""
    from .. import config as config_mod
    from .. import llm as llm_mod
    from .. import modelcard, roles
    quick = str(opts.get("mode") or "quick") != "full"
    recalibrate = bool(opts.get("recalibrate"))
    with STATE.lock:
        root = STATE.root
    base = config_mod.load(root=root, require_hardware=False,
                           warn_budgets=False)
    names = [n for n in (base.profiles or {})
             if not roles.profile_is_hosted(base, n)]
    todo: list[tuple[str, object]] = []
    for n in names:
        cfg = config_mod.load(root=root, profile=n, require_hardware=False,
                              warn_budgets=False)
        if modelcard.load(cfg) and not recalibrate:
            continue
        todo.append((n, cfg))
    if not todo:
        STATE.add_message(
            "system", "No local profiles need calibration"
            + (" \u2014 every local profile already has a fresh card. Tick "
               "\u201cRe-calibrate\u201d to measure again." if names else
               " \u2014 no local profiles are configured (hosted provider "
               "models use provider-table defaults)."), kind="calibrated")
        return
    STATE.add_message(
        "system", f"Calibrating {len(todo)} local profile(s): "
        f"{', '.join(n for n, _ in todo)}. Each model loads in turn; "
        "progress streams to the TERMINAL tab.", kind="calibrating")
    done = 0
    try:
        for n, cfg in todo:
            if job.cancelled:
                break
            with STATE.lock:
                job.title = (f"calibrating {n} "
                             f"({done + 1}/{len(todo)})")
            client = LLMClient(cfg)
            try:
                client.check(warn=False)
                client.warmup()
                card = modelcard.calibrate(client, cfg, verbose=True,
                                           quick=quick)
                path = modelcard.save_card(card)
                done += 1
                m = card.get("measured") or {}
                STATE.add_message(
                    "system", f"Calibration of '{n}' finished \u2014 "
                    f"overall score {m.get('overall', '?')}. "
                    f"Card saved to {path}.", kind="calibrated")
            finally:
                llm_mod.unload_model(client)
    except KeyboardInterrupt:
        pass                    # spawn's handler reports the stop
    STATE.add_message(
        "system", f"Calibrate-all finished: {done} of {len(todo)} local "
        "profile(s) calibrated.", kind="calibrated")


def run_benchmark(job: Job, opts: dict) -> None:
    """Run one built-in model benchmark as a background job.

    Progress from the scaffold and the two verification passes is streamed
    into one ACTIVITY message (and the terminal, because the scaffold prints
    through the GUI's stderr tee). The finished result is stored on
    ``STATE.benchmark`` for the Benchmarks dialog and summarised in chat.
    """
    from .. import benchmarks

    profile = str(opts.get("profile") or "").strip()
    benchmark_id = str(opts.get("benchmark_id") or "").strip()
    base_dir = str(opts.get("base_dir") or "").strip() or None
    with STATE.lock:
        current_root = STATE.root
        STATE.benchmark = None
    activity = STATE.add_message(
        "activity", f"Benchmark `{benchmark_id}` on profile "
        f"'{profile or 'default'}' starting\u2026", job=job.id)
    mid = activity["id"]

    def log(line: str) -> None:
        STATE.append_message(mid, "\n" + line.rstrip())

    result = benchmarks.run_benchmark(
        profile=profile or "", benchmark_id=benchmark_id,
        base_dir=base_dir, config_root=current_root,
        current_root=current_root, log=log, job_id=job.id)
    data = result.to_dict()
    with STATE.lock:
        STATE.benchmark = data
    level = ("ok" if result.rubric.ok and result.model_tests.ok else
             "warn" if result.rubric.passed or result.model_tests.passed
             else "error")
    lines = [
        f"Benchmark **{result.title}** finished for profile "
        f"`{result.profile}`\u2009\u00b7\u2009`{result.model}`.",
        "",
        f"- **Holdout rubric:** {result.rubric.passed}/{result.rubric.total} "
        f"checks passed",
        f"- **Model's own tests:** {result.model_tests.passed}/"
        f"{result.model_tests.total}"
        + (" (none generated)" if not result.model_tests.ran else ""),
        f"- **Duration:** {result.duration:.1f}s \u00b7 "
        f"{int(result.stats.get('calls', 0))} calls \u00b7 "
        f"{int(result.stats.get('total_tokens', 0)):,} tokens",
        f"- **Project:** `{result.project_dir}`",
        f"- **Report:** `{result.report_path}`",
    ]
    short = result.shortcomings
    if short:
        lines += ["", "**Shortcomings detected:**"]
        lines += [f"- {item}" for item in short[:8]]
        if len(short) > 8:
            lines.append(f"- \u2026 and {len(short) - 8} more")
    lines += ["", "Use **Benchmarks \u2192 Open Generated Project** to inspect "
              "it, or open that folder from the project bar."]
    STATE.add_message("assistant", "\n".join(lines), kind="benchmark",
                      level=level, benchmark={
                          "benchmark_id": result.benchmark_id,
                          "profile": result.profile,
                          "project_dir": str(result.project_dir),
                          "score": result.score,
                          "report_path": str(result.report_path or ""),
                      })


def empty_workspace_index() -> "dict | None":
    """Synchronous Build Index for a workspace with 0 indexable files.

    Returns the 200 reply dict after writing a valid empty index and marking
    the project indexed, or None when the normal background job should run
    (files exist, no project is open, or a job already holds the slot). An
    empty workspace has nothing to chunk or summarise, so answering inline
    gives the page a definite result instead of a background job -- and never
    touches the model, even for "Build index (with summaries)".
    """
    with STATE.lock:
        root, args, cfg = STATE.root, STATE.args, STATE.cfg
        client = STATE.client
        busy = STATE.job is not None and STATE.job.active
    if root is None or busy or args is None or cfg is None:
        return None
    matcher = cli_mod._matcher(args, root)
    if index_mod.select(args.paths, root, matcher,
                        index_mod.indexable_exts(cfg)):
        return None
    idx = index_mod.build(args.paths, cfg, client=None, root=root,
                          summarise=False, matcher=matcher)
    with STATE.lock:
        STATE.idx = idx
        STATE.agent = Agent(idx, client, cfg)
        STATE.index_gen += 1
    STATE.apply_index(idx)
    STATE.add_message("system", EMPTY_INDEX_MESSAGE, kind="indexed")
    return {"ok": True, "detail": EMPTY_INDEX_MESSAGE, "indexed": True,
            "files": 0, "chunks": 0}


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
    if not idx.files:
        STATE.add_message("system", EMPTY_INDEX_MESSAGE, kind="indexed")
    else:
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
        # Restore belongs to the workspace-backed group: with no project open
        # it reads the scratchpad's backups instead of refusing.
        root = STATE.scratchpad_path()
    snaps = workspace.list_backups(root)
    if not snaps:
        return False, "No snapshots exist for this workspace yet."
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

def _apply_security_choices(cfg, opts: dict) -> None:
    """Carry the GUI preflight's three-way choice into this run.

    `acknowledge_secrets` sends the request as-is (and persists the
    fingerprints); `redact_secrets` replaces every detected secret before
    the request leaves. Reset on every run so one choice never leaks into
    the next prompt.
    """
    try:
        cfg._outbound_acknowledge_secrets = bool(
            opts.get("acknowledge_secrets"))
        cfg._outbound_redact_secrets = bool(opts.get("redact_secrets"))
    except Exception:
        pass


def _set_pinned_refs(cfg, opts: dict) -> None:
    """Attach the composer's explicit pinned context to this run.

    The prompt's own `@file`/`@symbol` mentions are parsed by the agent; this
    list carries the Pinned pills (and API clients' `pinned` payload) so a
    pin survives even when the visible prompt text no longer shows the token.
    """
    values = opts.get("pinned") or []
    if isinstance(values, dict):
        values = [values]
    refs: list[str] = []
    for value in values if isinstance(values, (list, tuple, set)) else [values]:
        if isinstance(value, dict):
            value = (value.get("insert") or value.get("label")
                     or value.get("path") or value.get("file") or "")
        text = str(value or "").strip().lstrip("@").strip()
        if text and text not in refs:
            refs.append(text)
    try:
        cfg._pinned_refs = refs
    except Exception:
        pass


def _apply_hybrid_opts(cfg, opts: dict) -> None:
    """The Options bar's split-stage hybrid choices for this run: the
    Architect toggle plus its provider/model selector. Selecting a model
    arms hybrid even without the explicit flag; a provider-only selection
    takes the model from that profile's body. Session-scoped: run_task
    restores the previous values when the job ends."""
    if not any(k in opts for k in ("hybrid", "architect_provider",
                                   "architect_model")):
        return
    cfg.hybrid_routing = bool(opts.get("hybrid"))
    provider = str(opts.get("architect_provider") or "").strip()
    model = str(opts.get("architect_model") or "").strip()
    if provider and not model:
        body = (cfg.profiles or {}).get(provider) or {}
        model = str(body.get("model") or "").strip()
    cfg.architect_provider = provider
    cfg.architect_model = model
    if model and model != cfg.model:
        cfg.hybrid_routing = True


def run_arena(job: Job, opts: dict) -> None:
    """The 2-model sandbox arena: one prompt, two contestants, each in an
    isolated copy of the workspace. The real project stays untouched until
    the user applies a winner from the scorecard."""
    from .. import arena as arena_mod
    p = _project(need_index=False)
    prompt = compose(opts.get("text", ""), opts.get("attachments"))
    if not prompt:
        raise ConfigError("Describe the task both contestants should run.")
    model_a = opts.get("model_a") or opts.get("a") or ""
    model_b = opts.get("model_b") or opts.get("b") or ""
    if not model_a or not model_b:
        raise ConfigError("Pick two contestants (profile or provider:model).")
    _echo_user("arena", opts.get("text", "") or prompt,
               opts.get("attachments"), opts)
    cfg, root = p.cfg, p.root
    _evict_other_servers(cfg)
    with session(p, job, interactive=False, lock=True) as feed:
        state = arena_mod.run_arena_comparison(root, prompt, model_a, model_b,
                                               cfg)
        feed.end_stream()
        ca = state["contestants"]["a"]
        cb = state["contestants"]["b"]
        STATE.add_message(
            "assistant",
            f"Arena finished: A ({ca['model']}) "
            f"{'landed edits' if ca['ok'] else 'no edits'} in "
            f"{ca['elapsed_sec']}s; B ({cb['model']}) "
            f"{'landed edits' if cb['ok'] else 'no edits'} in "
            f"{cb['elapsed_sec']}s. Compare the runs and apply a winner.",
            kind="arena", arena_id=state["arena_id"])


def _security_post_run(root, cfg, files=None) -> None:
    """Refresh and announce the non-blocking workspace security banner."""
    try:
        status = security_mod.security_status(
            root, cfg, files=files, use_cache=False)
        security_mod.clear_security_cache(root)
        if status.get("secrets_found") or status.get("env_unignored"):
            STATE.bus.publish("security", security=status)
    except Exception:
        pass


def run_task(job: Job, opts: dict) -> None:
    """`ask`: gather, edit, and review or write. `write` is "review" (sign off
    in the page) or "apply" (write now, backups first)."""
    _ensure_index(job)
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
    _apply_security_choices(cfg, opts)
    _set_pinned_refs(cfg, opts)

    _evict_other_servers(cfg)
    saved_steps = cfg.max_steps
    saved_creative = cfg.creative_mode
    saved_hybrid = (cfg.hybrid_routing, cfg.architect_provider,
                    cfg.architect_model)
    _apply_hybrid_opts(cfg, opts)
    if steps > 0:
        cfg.max_steps = steps
    if "creative" in opts:
        cfg.creative_mode = bool(opts.get("creative"))
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
                def _bind_plan(path):
                    with STATE.lock:
                        STATE.plan_path = Path(path)

                code = plan_mod.run_planned(root, cfg, client, idx, task,
                                            apply=apply, on_plan=_bind_plan)
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
            if apply:
                # Pick up anything the run created beyond its edit results
                # (tool writes, generated files) so the tree and symbol
                # count are current the moment the job ends.
                _refresh_index(p)
            feed.end_stream()
            _post_task_result(tr, client, reviewing)
    finally:
        cfg.max_steps = saved_steps
        cfg.creative_mode = saved_creative
        (cfg.hybrid_routing, cfg.architect_provider,
         cfg.architect_model) = saved_hybrid
        _security_post_run(root, cfg)


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
    out = {"calls": s["calls"], "tokens_out": s["completion_tokens"],
           "tokens_in": s.get("prompt_tokens", 0),
           "cost": s.get("cost", 0.0), "seconds": s.get("seconds", 0.0)}
    # While a run is open, its aggregate (this job's calls only, with
    # provider/model and generation speed) rides along on the result card.
    run = events_mod.current_run()
    if run is not None and run.calls:
        out.update(run.summary())
    return out


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


SCRATCH_SYSTEM = """\
You are a helpful coding assistant answering a standalone question. No project
folder is open, so there is no source tree to inspect: answer from the question
itself, and say plainly when you would need the user to open a project or paste
the relevant code. Keep the answer focused and do not emit edit blocks."""


def run_explain(job: Job, opts: dict) -> None:
    """`explain`: a prose answer about the code, streamed into the chat.

    With no project open this becomes the standalone Ask: answer the prompt
    directly with the selected global profile instead of refusing. The state
    payload marks the session as the scratchpad until a project is opened."""
    question = compose(opts.get("text", ""), opts.get("attachments"))
    if not question:
        raise ConfigError("Ask a question first.")
    with STATE.lock:
        root = STATE.root
    if root is None:
        p = _global_project(str(opts.get("profile") or ""))
    else:
        p = _project()
    _echo_user("explain", opts.get("text", "") or question,
               opts.get("attachments"), opts)
    _set_pinned_refs(p.cfg, opts)
    _evict_other_servers(p.cfg)
    with session(p, job, interactive=False, lock=False) as feed:
        p.client.warmup()
        if p.agent is not None:
            answer = explain_mod.answer(p.agent, question, verbose=True,
                                        stream=True)
        else:
            p.client.site = "answer"
            answer = p.client.chat(
                [{"role": "system", "content": SCRATCH_SYSTEM},
                 {"role": "user", "content": question}],
                max_tokens=p.cfg.max_tokens_answer, stream=True)
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
    _ensure_index(job)
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
        force=bool(opts.get("force")),
        # The GUI's pre-run choice: "Run Auto Anyway" confirms the multi-step
        # loop for a single-pass-looking prompt (the CLI's --force-auto).
        force_auto=bool(opts.get("force_auto")))
    with STATE.lock:
        STATE.last_auto = {"text": opts.get("text", ""), "plan_file": plan_file}
    cfg, client, root = p.cfg, p.client, p.root
    _apply_security_choices(cfg, opts)
    _set_pinned_refs(cfg, opts)
    _evict_other_servers(cfg)
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
    saved_creative = cfg.creative_mode
    if "creative" in opts:
        cfg.creative_mode = bool(opts.get("creative"))
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
        cfg.creative_mode = saved_creative
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
    _security_post_run(root, cfg, files=changed or None)


def _int_or_none(value) -> int | None:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


def _int_or(value, default: int) -> int:
    got = _int_or_none(value)
    return default if got is None else got


def run_new(job: Job, opts: dict) -> None:
    """`new`: plan a project from a description and build it from scratch.

    A real build switches the workspace FIRST: the target folder becomes the
    active project (fresh session, empty index) before scaffolding starts, so
    every progress event and streamed token lands in the new project's
    session while it is being built. A preview writes nothing, stays unbound,
    and keeps the old workspace."""
    spec_text = compose(opts.get("text", ""), opts.get("attachments"))
    project = (opts.get("project") or "").strip()
    if not project:
        raise ConfigError("Choose a directory for the new project.")
    if not spec_text:
        raise ConfigError("Describe the project (or attach a spec file).")
    target = Path(project).expanduser().resolve()
    _echo_user("new", opts.get("text", "") or spec_text,
               opts.get("attachments"), opts)
    mode = opts.get("build", "build")         # build | preview | plan | review
    # The composer's profile picker wins for a no-project New; fall back to
    # the topbar's remembered selection when the request omits it.
    with STATE.lock:
        profile = str(opts.get("profile") or "").strip() \
            or STATE.profile_request
    if profile:
        STATE.set_profile(profile)
    if mode == "preview":
        # Nothing is written: a throwaway client on the nearest existing
        # ancestor's config, and the current workspace stays active.
        parent = target
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        args = args_for(str(parent), profile)
        cfg = cli_mod._config(args)
        _apply_security_choices(cfg, opts)
        if "creative" in opts:
            cfg.creative_mode = bool(opts.get("creative"))
        client = LLMClient(cfg)
        client.check(warn=False)
    else:
        # A real build owns its target: create it up front (parents included),
        # then switch the GUI's active root AND session before scaffolding.
        # The creation turn travels with the switch, so the new project's
        # session log opens on the very conversation that builds it.
        target.mkdir(parents=True, exist_ok=True)
        transcript = STATE.messages_snapshot()
        open_project(str(target), profile, announce=False)
        STATE.carry_transcript(transcript, title=spec_text)
        job.project_root = str(target)
        with STATE.lock:
            cfg, client = STATE.cfg, STATE.client
        _apply_security_choices(cfg, opts)
        if "creative" in opts:
            cfg.creative_mode = bool(opts.get("creative"))
        client.check(warn=False)
        # An empty index right now (0 files is valid): the page lands on an
        # indexed workspace immediately, and the post-build refresh picks up
        # every file the scaffold writes. Built inline because the job slot
        # is taken (empty_workspace_index would defer to a background job).
        idx = index_mod.build(STATE.args.paths, cfg, client=None, root=target,
                              summarise=False,
                              matcher=cli_mod._matcher(STATE.args, target))
        with STATE.lock:
            STATE.idx = idx
            STATE.agent = Agent(idx, client, cfg)
            STATE.index_gen += 1
        STATE.apply_index(idx)
    spec = scaffold.load_spec(prompt=spec_text, max_chars=cfg.max_spec_chars)
    p = SimpleNamespace(cfg=cfg, client=client, root=target)
    with session(p, job, interactive=False, lock=False) as feed:
        client.warmup()
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
            auto_install=bool(opts.get("auto_install") or cfg.auto_install_deps),
            log_command="" if mode == "preview" else "command: gui new")
        feed.end_stream()
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
    # The workspace switched before the build; bring the index in line with
    # what was actually written (the build persists its own index too) and
    # report the landed state. The completed job reports the root so the page
    # can follow without a manual refresh (mode drops back to Task).
    if mode != "preview" and target.is_dir():
        idx = cli_mod._refresh(STATE.idx, STATE.args, STATE.cfg, target,
                               verbose=False)
        _swap_index(idx)
        state = (f"index: {len(idx.files)} files" if idx is not None
                 else "no index yet")
        STATE.add_message("system", f"Opened {target} \u2014 {state}.",
                          kind="opened", needs_index=idx is None)
    _security_post_run(target, cfg)


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
