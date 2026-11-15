"""`pedacito tui`: a zero-dependency ANSI split-screen terminal UI.

The TUI keeps the same warm index/session as `pedacito chat`, but paints a
four-zone screen instead of a scrolling REPL:

    +--------------------------------------------------------------+
    | 1. header: project, profile/model, calibration, routing, git  |
    +------------------------------------+-------------------------+
    | 2. transcript, diffs and answers    | live activity / plan    |
    +------------------------------------+-------------------------+
    | 3. telemetry: context meter, burn rate, speed, cache, model   |
    +--------------------------------------------------------------+
    | 4. prompt: pdc tui (profile) >                                |
    +--------------------------------------------------------------+

Only the standard library is used. POSIX terminals enter the alternate
screen; Windows console handles get ENABLE_VIRTUAL_TERMINAL_PROCESSING via
`ctypes`. The same engine also runs headless (a non-TTY stdin or `--headless`)
so tests can feed a scripted stream and assert on complete frames without a
PTY. Every task runs on a worker thread while a 150 ms watcher tails the
active `.pedacito/events/*.jsonl` log and the client's telemetry, repainting
under a lock.

Slash commands mirror `pedacito chat` and add the TUI-specific panes:
`/help`, `/status` (`/telemetry`), `/auto`, `/plan`, `/consolidate`
(`/compact`), `/profile`, `/auto-route`, `/think`, `/apply`, `/skills`,
`/memory`, `/procs`, `/diff`, `/undo`, `/reset`, `/sidebar`, `/clear`, and
`/quit` (`/exit`).
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import signal
import sys
import textwrap
import threading
import time
from pathlib import Path

from . import events as events_mod
from . import gitinfo
from . import index as index_mod
from . import loop
from . import memory as memory_mod
from . import modelcard
from . import processes
from . import profiles as profiles_mod
from . import roles as roles_mod
from . import security
from . import skills as skills_mod
from . import workspace
from .agent import Agent
from .config import Config
from .llm import LLMClient, LLMError

__all__ = ["TuiSession", "enable_vt_mode", "run_tui", "HELP_TEXT"]

# ANSI sequences. They are emitted only for a real interactive TTY; headless
# frames stay plain text so CI logs and StringIO assertions are deterministic.
_ALT_SCREEN_ON = "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H"
_ALT_SCREEN_OFF = "\x1b[?1049l\x1b[r\x1b[?25h"
_CLEAR = "\x1b[2J\x1b[H"
_HOME = "\x1b[H"

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_PLAN_RE = re.compile(r"^\s*[-*]\s*\[([ xX])\]\s*(.*)$")
_SWAP_ARROW = "\u2192"

# Commands shown by /help and in the prompt bar's hint line.
HELP_TEXT = """\
Pedacito TUI commands
  /help                     this list and the keyboard shortcuts
  /status  /telemetry       full telemetry table and per-call-site tokens
  /auto GOAL                plan GOAL, then run every task with live updates
  /plan [GOAL]              show PLAN.md, or generate it from GOAL
  /consolidate /compact     compact the session into a handoff checkpoint
  /profile [NAME]           list profiles, or switch to NAME
  /model ID                 switch to a model id directly
  /auto-route [on|off]      specialist auto-routing on/off
  /think [on|off]           thinking on/off (off sets disable_thinking)
  /apply [on|off]           write edits to disk, or preview them
  /skills [list|show NAME|draft TEXT]
  /memory [list|add FACT|forget TEXT]
  /procs [list|logs NAME|stop NAME]
  /diff                     git status and working-tree diff
  /undo                     restore the newest .pedacito/backups snapshot
  /reset                    wipe .pedacito state (keeps memory and config)
  /sidebar                  toggle the activity / plan pane
  /clear                    clear the transcript pane
  /quit  /exit              leave and restore the terminal

Keyboard: Enter sends, Ctrl-C cancels the running task, Ctrl-D leaves.
Plain text is a task: it runs through the normal gather/edit/verify loop."""


def enable_vt_mode(stream=None) -> bool:
    """Turn on VT100 processing where the platform needs permission.

    POSIX terminals already understand ANSI, so this is a no-op there. On
    Windows, `ENABLE_VIRTUAL_TERMINAL_PROCESSING` (0x0004) is set on
    STD_OUTPUT_HANDLE. Non-console handles (pipes, StringIO, redirected
    output) are expected to fail and must never crash the caller.
    """
    if os.name != "nt":
        return True
    try:                                  # ctypes.windll exists on Windows
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return False
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _isatty(stream) -> bool:
    try:
        return bool(stream is not None and stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _display_width(text: str) -> int:
    """Printable width of a string that may contain CSI escape sequences."""
    return len(_ANSI_RE.sub("", text))


def _truncate_visible(text: str, width: int) -> str:
    """Cut `text` to `width` printable characters without splitting ANSI."""
    width = max(0, int(width))
    out: list[str] = []
    used = 0
    i = 0
    while i < len(text) and used < width:
        match = _ANSI_RE.match(text, i)
        if match:
            out.append(match.group(0))
            i = match.end()
            continue
        out.append(text[i])
        used += 1
        i += 1
    result = "".join(out)
    # Do not let a truncated colored string bleed into the next pane.
    if "\x1b[" in result and not result.endswith("\x1b[0m"):
        result += "\x1b[0m"
    return result


def _fit(text: str, width: int) -> str:
    """Truncate (ANSI-aware) and pad one rendered line to `width`."""
    width = max(0, int(width))
    cut = _truncate_visible(str(text), width)
    pad = width - _display_width(cut)
    return cut + (" " * max(0, pad))


def _compact_tokens(n) -> str:
    """14,200 -> 14.2k; small values stay exact (the telemetry footer)."""
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        return "0"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}m"
    if abs(n) >= 1000:
        return f"{n / 1000:.1f}k"
    return f"{n:,}"


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _meter(pct: float, width: int = 16) -> tuple[str, str, str]:
    """(bar, ANSI color, state): green <70, amber 70-85, red >85."""
    pct = max(0.0, min(100.0, _as_float(pct)))
    filled = max(0, min(width, int(round(pct / 100 * width))))
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    if pct > 85:
        return bar, "31", "high"
    if pct >= 70:
        return bar, "33", "warn"
    return bar, "32", "ok"


def _paint(text: str, color: str, enabled: bool = True) -> str:
    if not enabled or not color:
        return text
    return f"\x1b[{color}m{text}\x1b[0m"


def _active_label(value) -> str:
    """Normalise `active_model_role` (dict or display string) to one line."""
    if value is None:
        return "worker"
    if isinstance(value, str):
        return value.strip() or "worker"
    if isinstance(value, dict):
        profile = str(value.get("profile") or "")
        model = str(value.get("model") or "")
        role = str(value.get("role") or "")
        specialty = str(value.get("specialty") or "")
        label = profile or model or "model"
        detail = "/".join(part for part in (role, specialty) if part)
        if detail:
            return f"{label} [{detail}]"
        return label
    return str(value)


def _screen_size(default=(100, 30)) -> tuple[int, int]:
    try:
        size = shutil.get_terminal_size(default)
    except (OSError, ValueError):
        size = os.terminal_size(default)
    return max(40, int(size.columns)), max(12, int(size.lines))


class TuiSession:
    """One split-screen session.

    The constructor is deliberately side-effect-light: it installs the live
    event listener and permission broker, but connects to no model and builds
    no index. Those happen lazily on the first plain-text task. Headless
    callers can pass width/height to make frames byte-for-byte deterministic.
    """

    def __init__(self, root, cfg: Config | None = None, client=None, *,
                 agent=None, idx=None,
                 apply: bool = True, headless: bool | None = None,
                 stdin=None, stdout=None, initial_task: str = "",
                 width: int | None = None, height: int | None = None,
                 color: bool | None = None, sidebar: bool = True):
        self.root = Path(str(root)).expanduser().resolve()
        self.cfg = cfg if cfg is not None else Config()
        self.client = client
        self.agent = agent
        self.idx = idx
        self.apply = bool(apply)
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout
        self.initial_task = str(initial_task or "")
        self.width = int(width) if width else None
        self.height = int(height) if height else None
        if headless is None:
            headless = not _isatty(self.stdin)
        self.headless = bool(headless) or not _isatty(self.stdin)
        if color is None:
            color = not self.headless
        self.color = bool(color)
        self.sidebar = bool(sidebar)
        self.sidebar_available = True

        self.transcript: list[tuple[str, str]] = []
        self.activity: list[str] = []
        self.modified: list[str] = []
        self.folds = 0
        self.resume_context = ""
        self.last_telemetry: dict = {}
        self._event_prompt = 0
        self._event_completion = 0
        self._event_speed = 0.0
        self._event_session_prompt = 0
        self._event_session_completion = 0
        self._event_cached = 0
        self._event_cost = 0.0
        self._event_active = ""
        self._context_override: int | None = None
        self._prompt_banner = ""
        self._dirty = True
        self._quit = False
        self._running = False
        self._write_lock = threading.RLock()
        self._input_lock = threading.Lock()
        self._terminal_ready = False
        self._old_signals: dict = {}
        self._event_path: Path | None = None
        self._event_offset = 0
        self._recent_events: dict[tuple, float] = {}
        self._last_telemetry_key: tuple = ()
        self._headless_sent = 0
        self._headless_activity_sent = 0
        self._watch_stop = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self._git_cache: tuple[float, dict] = (0.0, {})
        self._session_allows: set[str] = set()
        get_hook = getattr(security, "permission_handler", None)
        try:
            self._prev_permission_hook = get_hook() if callable(get_hook) else None
        except Exception:
            self._prev_permission_hook = None

        self.elog = events_mod.get(self.root, self.cfg)
        self._listener = self._on_event
        try:
            self.elog.listeners.append(self._listener)
        except Exception:
            pass
        self._init_event_tail()
        self.install_permission_handler()

    # ---------------------------------------------------------------- state
    @property
    def profile(self) -> str:
        return str(getattr(self.cfg, "active_profile", "") or "default")

    @property
    def interactive(self) -> bool:
        return not self.headless

    def _size(self) -> tuple[int, int]:
        if self.width and self.height:
            return self.width, self.height
        columns, lines = _screen_size()
        return self.width or columns, self.height or lines

    def set_size(self, width: int, height: int) -> None:
        """Pin the frame size (tests) or nudge it after a resize event."""
        self.width = max(40, int(width))
        self.height = max(12, int(height))
        self._dirty = True

    # ------------------------------------------------------------ telemetry
    def telemetry(self) -> dict:
        """A tolerant snapshot of the client's live telemetry.

        Real `LLMClient` instances expose `telemetry()`; lightweight test
        doubles may expose only the fields. Both are accepted, and a missing
        field never raises while the UI is painting.
        """
        raw: dict = {}
        client = self.client
        if client is not None:
            fn = getattr(client, "telemetry", None)
            if callable(fn):
                try:
                    got = fn()
                    if isinstance(got, dict):
                        raw.update(got)
                except Exception:
                    pass
            for key in ("last_prompt_tokens", "last_completion_tokens",
                        "last_tok_per_s", "avg_tok_per_s", "context_tokens",
                        "context_utilization_pct", "cache_hit_pct",
                        "active_model_role", "stats", "last_elapsed_s"):
                if key in raw:
                    continue
                try:
                    raw[key] = getattr(client, key)
                except Exception:
                    continue

        stats = raw.get("stats")
        if not isinstance(stats, dict):
            try:
                stats = getattr(client, "stats", {}) or {}
            except Exception:
                stats = {}
        if not isinstance(stats, dict):
            stats = {}

        prompt = _as_int(raw.get("last_prompt_tokens"))
        completion = _as_int(raw.get("last_completion_tokens"))
        context = _as_int(raw.get("context_tokens"))
        if not context:
            context = _as_int(getattr(self.cfg, "context_tokens", 0))
        # Events are the live source of truth while a task runs; a test double
        # that only emits `call` events still moves the meter.
        if not prompt:
            prompt = self._event_prompt
        if not completion:
            completion = self._event_completion
        if self._context_override is not None:
            prompt = self._context_override
        raw_pct = raw.get("context_utilization_pct")
        pct = _as_float(raw_pct)
        if self._context_override is not None:
            pct = (prompt / context * 100) if context else 0.0
        elif context and (prompt and pct <= 0 or not isinstance(
                raw_pct, (int, float))):
            pct = prompt / context * 100
        pct = round(max(0.0, min(100.0, pct)), 1)

        avg = _as_float(raw.get("avg_tok_per_s"))
        if "avg_tok_per_s" not in raw:
            seconds = _as_float(stats.get("seconds"))
            avg = (_as_int(stats.get("completion_tokens")) / seconds
                   if seconds > 0 else 0.0)

        session_prompt = _as_int(raw.get("session_prompt_tokens"),
                                 _as_int(stats.get("prompt_tokens")))
        session_completion = _as_int(
            raw.get("session_completion_tokens"),
            _as_int(stats.get("completion_tokens")))
        if not session_prompt:
            session_prompt = self._event_session_prompt
        if not session_completion:
            session_completion = self._event_session_completion
        if not session_prompt and prompt:
            session_prompt = prompt
        if not session_completion and completion:
            session_completion = completion

        cache = raw.get("cache_hit_pct")
        if cache is None:
            total = _as_int(stats.get("prompt_tokens"))
            cached = _as_int(stats.get("cached_tokens"))
            if not total and self._event_session_prompt:
                total = self._event_session_prompt
                cached = self._event_cached
            cache = (cached / total * 100) if total else 0.0
        last_speed = _as_float(raw.get("last_tok_per_s"))
        if not last_speed:
            last_speed = self._event_speed
        cost = _as_float(stats.get("cost")) or self._event_cost
        active = self._event_active or raw.get("active_model_role", "worker")

        return {
            "last_prompt_tokens": prompt,
            "last_completion_tokens": completion,
            "context_tokens": context,
            "context_utilization_pct": pct,
            "last_tok_per_s": last_speed,
            "avg_tok_per_s": round(avg, 1),
            "session_prompt_tokens": session_prompt,
            "session_completion_tokens": session_completion,
            "cache_hit_pct": round(_as_float(cache), 1),
            "cost": cost,
            "active_model_role": active,
            "folds": int(self.folds),
            "stats": stats,
        }

    def context_bar(self, pct: float | None = None, width: int = 16) -> str:
        """The plain `[████░░░░]` meter, useful to tests and callers."""
        if pct is None:
            pct = self.telemetry()["context_utilization_pct"]
        return _meter(pct, width)[0]

    def telemetry_lines(self, width: int | None = None) -> list[str]:
        """The two footer rows, before line fitting."""
        width = width or self._size()[0]
        t = self.telemetry()
        pct = t["context_utilization_pct"]
        bar, color, state = _meter(pct, 16)
        painted = _paint(bar, color, self.color)
        prompt = t["last_prompt_tokens"]
        completion = t["last_completion_tokens"]
        context = t["context_tokens"]
        speed = t["last_tok_per_s"]
        avg = t["avg_tok_per_s"]
        cache = t["cache_hit_pct"]
        folds = t["folds"]
        active = _active_label(t["active_model_role"])
        cost = t["cost"]
        if cost:
            active += f" (${cost:.4f})"
        state_note = " !HIGH" if state == "high" else (
            " !WARN" if state == "warn" else "")

        if width >= 120:
            line1 = (
                f"Ctx: {painted} {prompt:,} / {context:,} ({pct:.1f}%) | "
                f"Turn: +{prompt:,} in / +{completion:,} out | "
                f"Speed: {speed:.1f} tok/s (avg {avg:.1f} tok/s){state_note}")
            line2 = (
                f"Session: {_compact_tokens(t['session_prompt_tokens'])} in / "
                f"{_compact_tokens(t['session_completion_tokens'])} out | "
                f"KV Cache: {cache:.0f}% | "
                f"Folds: {folds} | Active: {active}")
        else:
            line1 = (f"Ctx: {prompt:,} / {context:,} ({pct:.1f}%) "
                     f"{painted} {speed:.1f} tok/s{state_note}")
            line2 = (f"Turn: +{prompt:,}/+{completion:,} | "
                     f"Sess: {_compact_tokens(t['session_prompt_tokens'])}/"
                     f"{_compact_tokens(t['session_completion_tokens'])} | "
                     f"KV: {cache:.0f}% | {active}")
        return [line1, line2]

    # --------------------------------------------------------------- header
    def _git_state(self) -> dict:
        now = time.time()
        fetched, cached = self._git_cache
        if now - fetched < 2.0:
            return cached
        try:
            state = gitinfo.status(self.root)
        except Exception:
            state = {"branch": "", "dirty": [], "untracked": [],
                     "conflicts": []}
        self._git_cache = (now, state)
        return state

    def _calibration_badge(self) -> str:
        try:
            hosted = bool(getattr(self.cfg, "_hosted_server", False))
            if not hosted and hasattr(roles_mod, "profile_is_hosted"):
                hosted = roles_mod.profile_is_hosted(
                    self.cfg, getattr(self.cfg, "active_profile", "") or "")
        except Exception:
            hosted = False
        if hosted:
            return "\u2601 Hosted"
        return "\u2713 Calibrated"

    def header_lines(self, width: int | None = None) -> list[str]:
        width = width or self._size()[0]
        profile = self.profile
        model = str(getattr(self.cfg, "model", "") or "?")
        badge = self._calibration_badge()
        state = self._git_state()
        branch = state.get("branch") or "no-git"
        dirty = bool(state.get("dirty") or state.get("untracked")
                     or state.get("conflicts"))
        scout = str(getattr(self.cfg, "scout_profile", "") or "none")
        routing = "ON" if getattr(self.cfg, "auto_route_specialties",
                                  False) else "OFF"
        sidebar = "on" if (self.sidebar and width >= 100) else "off"
        project = self.root.name or str(self.root)
        line1 = (f" pdc tui \u2502 {project} \u2502 profile: {profile} "
                 f"\u2502 model: {model} \u2502 {badge}")
        line2 = (f" Auto-Route: {routing} \u2502 Scout: {scout} "
                 f"\u2502 git: {branch}{'*' if dirty else ''} "
                 f"\u2502 sidebar: {sidebar}")
        return [line1, line2]

    # -------------------------------------------------------------- sidebar
    def _plan_path(self) -> Path:
        raw = str(getattr(self.cfg, "auto_plan_file", "") or "PLAN.md")
        path = Path(raw)
        return path if path.is_absolute() else self.root / path

    def plan_items(self) -> list[tuple[bool, str]]:
        """`[(checked, text)]` from the active PLAN.md, in file order."""
        path = self._plan_path()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out: list[tuple[bool, str]] = []
        for line in lines:
            match = _PLAN_RE.match(line)
            if match:
                out.append((match.group(1).lower() == "x",
                            match.group(2).strip()))
        return out

    def _sidebar_lines(self, width: int, height: int) -> list[str]:
        width = max(20, width)
        rows: list[str] = ["ACTIVITY & PLAN"]
        items = self.plan_items()
        rows.append(f"PLAN: {self._plan_path().name}")
        if items:
            room = max(2, height // 3)
            for checked, text in items[:room]:
                mark = "x" if checked else " "
                rows.append(f"[{mark}] {text}")
            if len(items) > room:
                rows.append(f"    (+{len(items) - room} more)")
        else:
            rows.append("[ ] (no PLAN.md tasks yet)")
        rows.append("")
        rows.append("LIVE ACTIVITY / ROUTING")
        if self.activity:
            room = max(3, height // 3)
            for row in self.activity[-room:]:
                rows.append(row if row.startswith("\u00b7 ") else f"\u00b7 {row}")
        else:
            rows.append("\u00b7 (waiting for a task)")
        rows.append("")
        rows.append("MODIFIED FILES")
        if self.modified:
            for rel in self.modified[-max(2, height // 5):]:
                rows.append(f"~ {rel}")
        else:
            rows.append("~ (none yet)")
        return rows[:height]

    # ------------------------------------------------------------- painting
    def _transcript_rows(self, width: int, height: int) -> list[str]:
        if not self.transcript:
            intro = [
                ("system", f"Pedacito TUI \u2014 {self.root}"),
                ("system", "Type a coding task, or /help for commands."),
                ("system", "Telemetry updates after every model call."),
            ]
            source = intro
        else:
            source = self.transcript
        rows: list[str] = []
        prefixes = {
            "user": "\u00bb ",
            "assistant": "\u25c6 ",
            "system": "  ",
            "activity": "  ",
            "receipt": "  ",
            "diff": "  ",
            "error": "! ",
        }
        for kind, text in source[-300:]:
            prefix = prefixes.get(kind, "  ")
            body = str(text)
            for raw in body.splitlines() or [""]:
                chunks = textwrap.wrap(
                    prefix + raw, width=max(1, width),
                    replace_whitespace=False, drop_whitespace=False,
                    break_long_words=True, break_on_hyphens=False) or [prefix]
                rows.extend(chunks)
                prefix = "  "
        return rows[-max(1, height):]

    def render_frame(self, width: int | None = None,
                     height: int | None = None,
                     sidebar: bool | None = None) -> str:
        """Render one complete frame. The result has exactly `height` lines."""
        width, height = (int(width) if width else self.width,
                         int(height) if height else self.height)
        if not width or not height:
            default_w, default_h = self._size()
            width, height = width or default_w, height or default_h
        width, height = max(40, width), max(12, height)
        show_sidebar = self.sidebar if sidebar is None else bool(sidebar)
        show_sidebar = show_sidebar and width >= 100

        header = self.header_lines(width)
        footer = self.telemetry_lines(width)
        hint = ("Enter: send | /help: commands | /consolidate: compact | "
                "/quit: exit")
        prompt_text = f"pdc tui ({self.profile}) > "
        if self._prompt_banner:
            prompt_text = f"{self._prompt_banner} "
        body_height = max(4, height - len(header) - len(footer) - 2)

        if show_sidebar:
            side_width = min(46, max(32, width // 3))
            left_width = width - side_width - 1
            left = ["TRANSCRIPT"] + self._transcript_rows(
                left_width - 1, body_height - 1)
            right = self._sidebar_lines(side_width - 1, body_height)
            body: list[str] = []
            for row in range(body_height):
                lt = left[row] if row < len(left) else ""
                rt = right[row] if row < len(right) else ""
                body.append(_fit(lt, left_width) + "\u2502"
                            + _fit(rt, side_width))
        else:
            body = ["TRANSCRIPT"] + self._transcript_rows(
                width - 1, body_height - 1)
            body = [_fit(row, width) for row in body[:body_height]]
            while len(body) < body_height:
                body.append(_fit("", width))

        lines = [_fit(row, width) for row in header]
        lines.extend(body)
        lines.extend(_fit(row, width) for row in footer)
        lines.append(_fit(hint, width))
        lines.append(_fit(prompt_text, width))

        if len(lines) < height:
            lines.extend(_fit("", width) for _ in range(height - len(lines)))
        return "\n".join(lines[:height])

    # Friendly aliases for programmatic callers (the tests drive the session
    # directly rather than through a PTY).
    def render(self, width: int | None = None,
               height: int | None = None) -> str:
        return self.render_frame(width, height)

    def layout(self, width: int | None = None,
               height: int | None = None) -> str:
        return self.render_frame(width, height)

    # --------------------------------------------------------------- output
    def _write(self, text: str) -> None:
        with self._write_lock:
            try:
                self.stdout.write(text)
                self.stdout.flush()
            except (OSError, ValueError, AttributeError):
                pass

    def _redraw(self, force: bool = False) -> None:
        if not force and not self._dirty:
            return
        width, height = self._size()
        self.width, self.height = width, height
        frame = self.render_frame(width, height)
        if self.headless:
            # A viewport only shows its tail; scripted consumers (tests, CI)
            # still need every command output and activity line, so emit the
            # newly appended rows as plain text before the frame.
            new_rows = self.transcript[self._headless_sent:]
            new_activity = self.activity[self._headless_activity_sent:]
            self._headless_sent = len(self.transcript)
            self._headless_activity_sent = len(self.activity)
            chunks: list[str] = []
            if new_rows:
                chunks.append("\n".join(f"{kind}: {text}"
                                        for kind, text in new_rows))
            if new_activity:
                chunks.append("\n".join(f"activity: {row}"
                                        for row in new_activity))
            if chunks:
                with self._write_lock:
                    self._write("\n" + "\n".join(chunks) + "\n")
        with self._write_lock:
            if self.interactive:
                self._write(_HOME + frame + "\x1b[J")
            else:
                self._write(frame + "\n")
        self._dirty = False

    def _append(self, text: str, kind: str = "system") -> None:
        body = str(text or "")
        if len(body) > 12000:
            body = body[:12000] + "\n[truncated in the transcript view]"
        for line in body.splitlines() or [""]:
            self.transcript.append((kind, line))
        self._dirty = True

    def _append_many(self, text: str, kind: str = "system") -> None:
        self._append(text, kind)

    def _set_banner(self, text: str) -> None:
        self._prompt_banner = str(text or "")

    # --------------------------------------------------------------- events
    def _init_event_tail(self) -> None:
        """Start at the end of the newest log so old sessions stay out."""
        directory = self.root / ".pedacito" / "events"
        try:
            files = sorted(directory.glob("*.jsonl"),
                           key=lambda p: p.stat().st_mtime_ns)
        except OSError:
            return
        if not files:
            return
        self._event_path = files[-1]
        try:
            self._event_offset = files[-1].stat().st_size
        except OSError:
            self._event_offset = 0

    def _event_key(self, kind: str, fields: dict) -> tuple:
        try:
            payload = json.dumps(fields, sort_keys=True, default=str)
        except (TypeError, ValueError):
            payload = repr(fields)
        return (str(kind), payload[:2000])

    def _remember_event(self, kind: str, fields: dict,
                        source: str = "live") -> bool:
        """False when a file-tail record repeats the live listener's event."""
        key = self._event_key(kind, fields)
        now = time.monotonic()
        seen = self._recent_events.get(key)
        if source == "file" and seen is not None and now - seen < 2.0:
            return False
        self._recent_events[key] = now
        if len(self._recent_events) > 400:
            cutoff = now - 5.0
            self._recent_events = {
                k: ts for k, ts in self._recent_events.items() if ts >= cutoff}
        return True

    def _on_event(self, kind: str, fields: dict) -> None:
        self._apply_event(kind, fields, source="live")

    def _apply_event(self, kind: str, fields: dict,
                     source: str = "live") -> None:
        if not isinstance(fields, dict):
            fields = {}
        if not self._remember_event(kind, fields, source):
            return
        kind = str(kind or "")
        if kind == "step":
            action = str(fields.get("action") or "step")
            detail = str(fields.get("detail") or fields.get("file") or "")
            if fields.get("file") and fields.get("file") not in detail:
                detail = f"{fields['file']} {detail}".strip()
            self.activity.append(f"{action} {detail}".rstrip())
        elif kind == "call":
            site = str(fields.get("site") or "call")
            pt = _as_int(fields.get("prompt_tokens"))
            ct = _as_int(fields.get("completion_tokens"))
            seconds = _as_float(fields.get("seconds"))
            cached = _as_int(fields.get("cached_tokens"))
            self._context_override = None
            self._event_prompt = pt
            self._event_completion = ct
            self._event_session_prompt += pt
            self._event_session_completion += ct
            self._event_cached += cached
            self._event_cost += _as_float(fields.get("cost"))
            if seconds > 0:
                self._event_speed = ct / seconds
            model = str(fields.get("model") or "")
            if model:
                self._event_active = model
            self.activity.append(f"call {site} +{pt:,} in / +{ct:,} out")
        elif kind == "specialist_routed":
            profile = str(fields.get("profile") or "?")
            specialty = str(fields.get("specialty") or "")
            model = str(fields.get("model") or "")
            tail = f" ({specialty})" if specialty else ""
            self.activity.append(
                f"specialist_routed {_SWAP_ARROW} {profile}{tail} "
                f"{model}".strip())
            detail = f" ({specialty})" if specialty else ""
            self._event_active = f"{profile}{detail}"
        elif kind == "model_swap":
            old = str(fields.get("unloaded") or fields.get("from") or "?")
            new = str(fields.get("loaded") or fields.get("to") or "?")
            self.activity.append(
                f"model_swap: {old} {_SWAP_ARROW} {new}")
            if new and new != "?":
                self._event_active = new
        elif kind == "verify":
            errors = _as_int(fields.get("errors"))
            warnings = _as_int(fields.get("warnings"))
            mark = "\u2713" if not errors else "\u2717"
            text = (f"{mark} verify: {errors} error(s), "
                    f"{warnings} warning(s)")
            self.activity.append(text)
            self._append(f"verification: {errors} error(s), "
                         f"{warnings} warning(s)", "receipt")
        elif kind == "fold":
            n = _as_int(fields.get("n"), self.folds + 1)
            self.folds = max(self.folds + 1, n)
            self.activity.append(f"fold {self.folds} (context compacted)")
        elif kind == "task":
            title = str(fields.get("task") or fields.get("title") or "task")
            self.activity.append(f"task: {title}")
        elif kind == "command":
            cmd = str(fields.get("cmd") or fields.get("command") or "")
            if cmd:
                self.activity.append(f"command: {cmd}")
        if len(self.activity) > 200:
            del self.activity[:-200]
        self._dirty = True

    def _poll_events(self, force: bool = False) -> None:
        """Read complete new lines from the active event log (150 ms ticks)."""
        directory = self.root / ".pedacito" / "events"
        try:
            files = sorted(directory.glob("*.jsonl"),
                           key=lambda p: p.stat().st_mtime_ns)
        except OSError:
            return
        if not files:
            return
        path = files[-1]
        if path != self._event_path:
            self._event_path = path
            self._event_offset = 0
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self._event_offset:
            self._event_offset = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._event_offset)
                while True:
                    line = fh.readline()
                    if not line:
                        break
                    self._event_offset = fh.tell()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(record, dict):
                        continue
                    kind = str(record.get("kind") or "")
                    fields = {k: v for k, v in record.items()
                              if k not in ("t", "kind")}
                    self._apply_event(kind, fields, source="file")
        except OSError:
            return

    # ------------------------------------------------------------- watcher
    def _start_watcher(self) -> None:
        if self._watch_thread is not None and self._watch_thread.is_alive():
            return
        self._watch_stop.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_loop, name="pdc-tui-watch", daemon=True)
        self._watch_thread.start()

    def _stop_watcher(self) -> None:
        self._watch_stop.set()
        thread = self._watch_thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._watch_thread = None

    def _watch_loop(self) -> None:
        while not self._watch_stop.wait(0.15):
            self._poll_events()
            before = self._last_telemetry_key
            snapshot = self.telemetry()
            key = (snapshot["last_prompt_tokens"],
                   snapshot["last_completion_tokens"],
                   snapshot["last_tok_per_s"],
                   snapshot["avg_tok_per_s"],
                   snapshot["context_utilization_pct"],
                   snapshot["cache_hit_pct"],
                   snapshot["session_prompt_tokens"],
                   snapshot["session_completion_tokens"],
                   snapshot["folds"])
            if key != before:
                self.last_telemetry = snapshot
                self._last_telemetry_key = key
                self._dirty = True
            if self.interactive and self._dirty:
                self._redraw()

    # -------------------------------------------------------- permissions
    def install_permission_handler(self) -> None:
        security.set_permission_handler(self._permission_hook)

    def _permission_hook(self, info: dict) -> tuple[bool, str]:
        category = str(info.get("category") or security.CAT_SHELL)
        label = security.CATEGORY_LABELS.get(category, category)
        detail = (str(info.get("detail") or "")
                  or str(info.get("action_summary") or "")).strip()
        key = str(info.get("key") or category)
        if key in self._session_allows or f"{category}:*" in self._session_allows:
            return True, f"Allowed for this session ({label})."
        banner = (f"[Permission Required: {category} ({label}) "
                  f"\u2014 {detail[:160]}] Allow? "
                  f"(y = once / a = session / n = deny):")
        self._set_banner(banner)
        self._append(banner, "activity")
        self._redraw(force=True)
        answer = self._readline()
        self._set_banner("")
        text = (answer or "").strip().lower()
        first = text.split()[0] if text else "n"
        if first in ("y", "yes", "once"):
            self._append("permission: allowed once", "receipt")
            self._dirty = True
            return True, f"Allowed once ({label})."
        if first in ("a", "all", "always", "session"):
            self._session_allows.add(key)
            self._session_allows.add(f"{category}:*")
            self._append("permission: allowed for this session", "receipt")
            self._dirty = True
            return True, f"Allowed for this session ({label})."
        self._append("permission: denied", "receipt")
        self._dirty = True
        return False, security.DENIED_MESSAGE

    # --------------------------------------------------------------- input
    def _readline(self) -> str | None:
        """One line, or None at EOF. Permission prompts use the same stream."""
        with self._input_lock:
            stream = self.stdin
            if stream is None:
                return None
            readline = getattr(stream, "readline", None)
            if callable(readline):
                try:
                    raw = readline()
                except (EOFError, KeyboardInterrupt, OSError):
                    return None
            else:
                try:
                    raw = next(iter(stream))
                except (StopIteration, TypeError):
                    return None
            if raw == "" or raw is None:
                return None
            return str(raw).rstrip("\r\n")

    # ---------------------------------------------------------- backend lazily
    def _ensure_client(self):
        if self.client is None:
            self.client = LLMClient(self.cfg)
        try:
            events_mod.attach(self.client, self.elog)
        except Exception:
            pass
        return self.client

    def _ensure_backend(self):
        client = self._ensure_client()
        try:
            from . import config as config_mod
            config_mod.auto_fit(self.cfg, verbose=False)
        except Exception:
            pass
        if self.idx is None:
            try:
                self.idx = index_mod.Index.load(self.root)
            except Exception:
                self.idx = None
            if self.idx is None:
                try:
                    workspace.ensure(self.root, self.cfg.manage_gitignore,
                                     self.cfg.create_ignore_file)
                    self.idx = index_mod.build(
                        [str(self.root)], self.cfg, client=None,
                        root=self.root, summarise=False, verbose=False)
                except (SystemExit, Exception):
                    # An empty or all-ignored project must still open a UI:
                    # the model can answer questions without local source.
                    self.idx = index_mod.Index(self.root, [])
        if self.agent is None:
            self.agent = Agent(self.idx, client, self.cfg)
        return client, self.agent, self.idx

    def _attach_runtime(self) -> None:
        """Apply live toggles to already-built objects after /profile etc."""
        if self.agent is not None:
            self.agent.cfg = self.cfg
        if self.idx is not None and self.agent is not None:
            self.agent.index = self.idx

    def _refresh_warmup(self, client) -> None:
        check = getattr(client, "check", None)
        warmup = getattr(client, "warmup", None)
        try:
            if callable(check):
                check(warn=False)
            if callable(warmup):
                warmup()
        except LLMError as exc:
            self._append(f"model warning: {exc}", "error")
        except Exception:
            pass

    # --------------------------------------------------------------- tasks
    def _run_in_worker(self, work) -> dict:
        holder: dict = {"result": None, "error": None}
        self._start_watcher()

        def runner():
            try:
                holder["result"] = work()
            except BaseException as exc:          # shown, never silently lost
                holder["error"] = exc

        thread = threading.Thread(target=runner, name="pdc-tui-task",
                                  daemon=True)
        thread.start()
        try:
            thread.join()
        except KeyboardInterrupt:
            cancel = getattr(self.client, "cancel", None)
            try:
                if callable(cancel):
                    cancel()
            except Exception:
                pass
            self._append("cancelled (the session stays open)", "error")
            holder["error"] = KeyboardInterrupt()
        finally:
            self._poll_events(force=True)
            self._stop_watcher()
            self._sync_telemetry()
        return holder

    def _sync_telemetry(self) -> None:
        snapshot = self.telemetry()
        self.last_telemetry = snapshot
        self._last_telemetry_key = (
            snapshot["last_prompt_tokens"],
            snapshot["last_completion_tokens"],
            snapshot["last_tok_per_s"],
            snapshot["avg_tok_per_s"],
            snapshot["context_utilization_pct"],
            snapshot["cache_hit_pct"],
            snapshot["session_prompt_tokens"],
            snapshot["session_completion_tokens"],
            snapshot["folds"])
        self._dirty = True

    def _record_result(self, task: str, tr) -> None:
        def field(row, name, default=None):
            if isinstance(row, dict):
                return row.get(name, default)
            return getattr(row, name, default)

        if tr is None:
            self._append(f"task finished: {task}", "system")
            return
        if isinstance(tr, int):
            self._append(f"task finished with exit code {tr}", "system")
            return
        reply = field(tr, "reply", "") or ""
        if reply:
            self._append(reply, "assistant")
        for row in field(tr, "results", []) or []:
            ok = bool(field(row, "ok", False))
            rel = str(field(row, "file", "") or "?")
            msg = str(field(row, "message", "") or "")
            mark = "\u2713 applied" if ok else "\u2717 failed"
            self._append(f"{mark} {rel}: {msg}".rstrip(), "receipt")
            if ok:
                if rel not in self.modified:
                    self.modified.append(rel)
                for line in (field(row, "diff", "") or "").splitlines()[:24]:
                    self._append(line, "diff")
        for rel in field(tr, "changed", []) or []:
            rel = str(rel)
            if rel and rel not in self.modified:
                self.modified.append(rel)
        for step in field(tr, "steps", []) or []:
            text = str(step).strip()
            if text:
                self.activity.append(text)
        tests_ok = field(tr, "tests_ok")
        if tests_ok is True:
            self._append("tests: GREEN", "receipt")
        elif tests_ok is False:
            self._append("tests: RED", "error")
        for finding in field(tr, "findings", []) or []:
            if field(finding, "severity", "error") == "error":
                render = field(finding, "render")
                detail = render() if callable(render) else str(finding)
                self._append(f"finding: {detail}", "error")
        if field(tr, "rolled_back", False):
            self._append("green ratchet: this step was rolled back", "error")
        self._dirty = True

    def run_task(self, task: str) -> None:
        """Run one plain-text coding task, keeping the UI live while it runs."""
        self._context_override = None
        self._event_active = ""
        self._append(f"\u00bb {task}", "user")
        client, agent, idx = self._ensure_backend()
        cfg = self.cfg
        if self.resume_context:
            task = self.resume_context + "\n\nNEW TASK\n" + task
            self.resume_context = ""
        self._append(
            f"running {'apply' if self.apply else 'dry run'} "
            f"on {cfg.model} ...", "system")
        self._refresh_warmup(client)

        def work():
            return loop.run_task(self.root, cfg, client, agent, idx, task,
                                 apply=self.apply)

        holder = self._run_in_worker(work)
        error = holder.get("error")
        if error is not None:
            if isinstance(error, KeyboardInterrupt):
                pass
            elif isinstance(error, LLMError):
                self._append(f"ERROR: {error}", "error")
            else:
                self._append(f"ERROR: {type(error).__name__}: {error}",
                             "error")
            self._redraw(force=True)
            return
        self._record_result(task, holder.get("result"))
        self._redraw(force=True)

    def run_auto(self, goal: str) -> None:
        """`/auto`: generate or resume PLAN.md, then run the outer loop."""
        goal = " ".join(str(goal or "").split())
        if not goal:
            self._append("Usage: /auto <goal>", "error")
            return
        client, agent, idx = self._ensure_backend()
        cfg = self.cfg
        plan_path = self._plan_path()
        self._append(f"\u00bb /auto {goal}", "user")
        self._refresh_warmup(client)

        def work():
            lines = loop.read_plan(plan_path)
            if not lines or loop.next_task(lines) is None:
                titles = loop.generate_tasks(
                    goal, roles_mod.client_for(cfg, "planner", client), cfg,
                    root=self.root)
                loop.write_plan(plan_path, goal, titles)
            self._append(f"plan: {plan_path}", "system")
            for checked, text in self.plan_items():
                self._append(f"[{'x' if checked else ' '}] {text}", "system")
            return loop.run_loop(
                self.root, cfg, client, idx, plan_path, apply=self.apply,
                verify=cfg.auto_verify, repair_rounds=cfg.auto_repair_rounds,
                max_tasks=cfg.auto_max_tasks, max_minutes=cfg.auto_max_minutes,
                stop_on_fail=True, auto_install=cfg.auto_install_deps,
                goal=goal)

        holder = self._run_in_worker(work)
        error = holder.get("error")
        if error is not None:
            if not isinstance(error, KeyboardInterrupt):
                self._append(f"ERROR: {error}", "error")
        else:
            result = holder.get("result")
            if isinstance(result, int):
                self._append(f"auto finished with exit code {result}",
                             "system")
            self._append("auto loop finished", "system")
        self._sync_telemetry()
        self._redraw(force=True)

    # -------------------------------------------------------------- planner
    def show_plan(self, goal: str = "") -> None:
        goal = " ".join(str(goal or "").split())
        if goal:
            client, _agent, _idx = self._ensure_backend()
            self._refresh_warmup(client)
            self._append(f"\u00bb /plan {goal}", "user")
            try:
                titles = loop.generate_tasks(
                    goal, roles_mod.client_for(self.cfg, "planner", client),
                    self.cfg, root=self.root)
            except Exception as exc:
                self._append(f"plan generation failed: {exc}", "error")
                return
            loop.write_plan(self._plan_path(), goal, titles)
            self._append(f"wrote {self._plan_path()}", "system")
        items = self.plan_items()
        if not items:
            self._append(f"No plan tasks at {self._plan_path()}. "
                         "Use /plan <goal> or /auto <goal>.", "system")
            return
        for checked, text in items:
            self._append(f"[{'x' if checked else ' '}] {text}", "system")

    # ------------------------------------------------------------- commands
    def handle_line(self, line: str) -> bool:
        """Handle one prompt line. Returns False when the session should end."""
        text = str(line or "").strip()
        if not text:
            return not self._quit
        if text.lower() in ("quit", "exit", "q"):
            self._quit = True
            self._append("Goodbye.", "system")
            return False
        if text.startswith("/"):
            self._slash(text)
        else:
            self.run_task(text)
        return not self._quit

    def _slash(self, line: str) -> None:
        parts = line[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        self._append(f"\u00bb {line}", "user")

        if cmd in ("quit", "exit", "q"):
            self._quit = True
            self._append("Goodbye.", "system")
            return
        if cmd in ("help", "h", "?"):
            self._append(HELP_TEXT, "system")
            return
        if cmd in ("status", "telemetry"):
            self._append(self.status_report(), "system")
            return
        if cmd == "cost":
            self._append(self.status_report(), "system")
            return
        if cmd == "steps":
            try:
                n = max(1, int(arg))
            except ValueError:
                self._append("Usage: /steps N", "error")
                return
            self.cfg.max_steps = n
            if self.agent is not None:
                self.agent.cfg = self.cfg
            self._append(f"max_steps = {n}", "system")
            return
        if cmd == "auto":
            self.run_auto(arg)
            return
        if cmd == "plan":
            self.show_plan(arg)
            return
        if cmd in ("consolidate", "compact"):
            self._consolidate()
            return
        if cmd in ("profile", "profiles"):
            self._profiles(arg)
            return
        if cmd == "model":
            self._model(arg)
            return
        if cmd in ("auto-route", "autoroute"):
            self._toggle("auto_route_specialties", arg, "Auto-Route")
            return
        if cmd == "think":
            self._think(arg)
            return
        if cmd == "apply":
            self._apply_toggle(arg)
            return
        if cmd == "skills":
            self._skills(arg)
            return
        if cmd == "memory":
            self._memory(arg)
            return
        if cmd == "procs":
            self._procs(arg)
            return
        if cmd == "diff":
            self._diff()
            return
        if cmd == "undo":
            self._undo()
            return
        if cmd == "reset":
            self._reset()
            return
        if cmd == "sidebar":
            self.sidebar = not self.sidebar
            shown = self.sidebar and self._size()[0] >= 100
            self._append(f"Sidebar: {'on' if shown else 'off'}", "system")
            if self.sidebar and not shown:
                self._append("(the activity pane needs a terminal at least "
                             "100 columns wide)", "system")
            self._dirty = True
            return
        if cmd == "clear":
            self.transcript = []
            self._headless_sent = 0
            self._append("Transcript cleared.", "system")
            self._dirty = True
            return
        self._append(f"Unknown command '/{cmd}'. Try /help.", "error")

    def status_report(self) -> str:
        t = self.telemetry()
        bar, _color, state = _meter(t["context_utilization_pct"], 16)
        lines = [
            "Pedacito TUI telemetry",
            f"  profile : {self.profile}",
            f"  model   : {getattr(self.cfg, 'model', '?')}",
            f"  writes  : {'ON' if self.apply else 'off (dry run)'}",
            f"  context : [{bar}] {t['context_utilization_pct']:.1f}% "
            f"({t['last_prompt_tokens']:,} / {t['context_tokens']:,})"
            + ("  HIGH" if state == "high" else ""),
            f"  turn    : +{t['last_prompt_tokens']:,} in / "
            f"+{t['last_completion_tokens']:,} out",
            f"  session : {t['session_prompt_tokens']:,} in / "
            f"{t['session_completion_tokens']:,} out",
            f"  speed   : {t['last_tok_per_s']:.1f} tok/s last turn, "
            f"{t['avg_tok_per_s']:.1f} tok/s session average",
            f"  cache   : {t['cache_hit_pct']:.1f}% KV cache hit",
            f"  folds   : {t['folds']}",
            f"  active  : {_active_label(t['active_model_role'])}",
            f"  routing : Auto-Route "
            f"{'ON' if getattr(self.cfg, 'auto_route_specialties', False) else 'OFF'}"
            f", thinking {'off' if getattr(self.cfg, 'disable_thinking', False) else 'on'}",
        ]
        card = getattr(self.cfg, "_card", None)
        if not isinstance(card, dict):
            try:
                card = modelcard.load(self.cfg)
            except Exception:
                card = None
        if isinstance(card, dict):
            score = (card.get("measured") or {}).get("overall")
            lines.append(f"  model card: {score if score is not None else '?'}"
                         " / 10 (calibrated)")
        else:
            lines.append("  model card: none (run `pedacito calibrate`)")
        try:
            specs = roles_mod.specialties_for(
                self.cfg, getattr(self.cfg, "active_profile", "") or "")
        except Exception:
            specs = ()
        lines.append("  specialties: " + (", ".join(specs) if specs else "(none)"))
        stats = t.get("stats") if isinstance(t.get("stats"), dict) else {}
        by_site = stats.get("by_site") if isinstance(stats, dict) else {}
        lines.append("  per-call-site tokens:")
        if isinstance(by_site, dict) and by_site:
            for site, row in sorted(by_site.items()):
                row = row if isinstance(row, dict) else {}
                lines.append(
                    f"    {site:<12} {_as_int(row.get('calls')):>3} calls  "
                    f"{_as_int(row.get('prompt_tokens')):>8,} in  "
                    f"{_as_int(row.get('completion_tokens')):>7,} out  "
                    f"{_as_int(row.get('cached_tokens')):>8,} cached")
        else:
            lines.append("    (no model calls yet)")
        return "\n".join(lines)

    def _consolidate(self) -> None:
        messages = getattr(self.agent, "_messages", None) if self.agent else None
        try:
            result = memory_mod.consolidate_session(
                self.root, self.cfg, messages)
        except OSError as exc:
            self._append(f"consolidate failed: {exc}", "error")
            return
        self.resume_context = memory_mod.handoff_prompt(result["summary"])
        try:
            if self.client is not None:
                self.client.last_prompt_tokens = int(result["after_tokens"])
        except Exception:
            pass
        # Even a telemetry-less session (headless, no client yet) must show the
        # meter drop immediately after compaction.
        self._event_prompt = int(result["after_tokens"])
        self._event_session_prompt = int(result["after_tokens"])
        self._context_override = int(result["after_tokens"])
        self._append(
            f"consolidated: ~{result['before_tokens']:,} -> "
            f"~{result['after_tokens']:,} tokens; the next task starts from "
            f"the handoff ({result['path']})", "system")
        self._append(result["summary"], "receipt")
        self._sync_telemetry()
        self._dirty = True

    def _profiles(self, arg: str) -> None:
        if not arg:
            try:
                rows = profiles_mod.list_profiles(self.cfg)
            except Exception as exc:
                self._append(f"could not list profiles: {exc}", "error")
                return
            if not rows:
                self._append("No profiles defined in the config. The default "
                             "profile is in use.", "system")
                return
            self._append(profiles_mod.render(rows), "system")
            return
        if arg not in (getattr(self.cfg, "profiles", None) or {}):
            self._append(f"No profile named '{arg}'.", "error")
            return
        try:
            new_cfg = roles_mod.derive_config(self.cfg, arg)
        except Exception as exc:
            self._append(f"could not switch profile: {exc}", "error")
            return
        if new_cfg is None:
            self._append(f"could not switch profile: '{arg}' is unknown.",
                         "error")
            return
        self.cfg = new_cfg
        self.client = LLMClient(new_cfg)
        self._event_active = ""
        try:
            events_mod.attach(self.client, events_mod.get(self.root, new_cfg))
        except Exception:
            pass
        if self.idx is not None:
            self.agent = Agent(self.idx, self.client, new_cfg)
        self._attach_runtime()
        self._sync_telemetry()
        self._append(f"now using profile {new_cfg.active_profile} "
                     f"(model {new_cfg.model})", "system")

    def _model(self, arg: str) -> None:
        if not arg:
            self._append(f"model: {getattr(self.cfg, 'model', '?')}",
                         "system")
            return
        new_cfg = copy.copy(self.cfg)
        new_cfg.model = arg
        new_cfg.active_profile = ""
        new_cfg.sources = dict(getattr(self.cfg, "sources", {}) or {})
        new_cfg.sources["model"] = "/model"
        try:
            self.client = LLMClient(new_cfg)
        except Exception as exc:
            self._append(f"could not create a client for {arg}: {exc}",
                         "error")
            return
        self.cfg = new_cfg
        self._event_active = ""
        try:
            events_mod.attach(self.client, events_mod.get(self.root, new_cfg))
        except Exception:
            pass
        if self.idx is not None:
            self.agent = Agent(self.idx, self.client, new_cfg)
        self._attach_runtime()
        self._sync_telemetry()
        self._append(f"now using model {arg}", "system")

    def _toggle(self, attr: str, arg: str, label: str) -> None:
        value = arg.strip().lower()
        if value in ("", "toggle"):
            setattr(self.cfg, attr, not bool(getattr(self.cfg, attr, False)))
        elif value in ("on", "true", "yes", "1"):
            setattr(self.cfg, attr, True)
        elif value in ("off", "false", "no", "0"):
            setattr(self.cfg, attr, False)
        else:
            self._append(f"Usage: /{label.lower().replace(' ', '-')} on|off",
                         "error")
            return
        try:
            self.cfg.sources[attr] = "tui toggle"
        except Exception:
            pass
        self._append(f"{label}: "
                     f"{'ON' if getattr(self.cfg, attr, False) else 'OFF'}",
                     "system")

    def _think(self, arg: str) -> None:
        value = arg.strip().lower()
        if value in ("", "toggle"):
            self.cfg.disable_thinking = not bool(self.cfg.disable_thinking)
        elif value in ("on", "true", "yes", "1"):
            self.cfg.disable_thinking = False
        elif value in ("off", "false", "no", "0"):
            self.cfg.disable_thinking = True
        else:
            self._append("Usage: /think on|off", "error")
            return
        try:
            self.cfg.sources["disable_thinking"] = "tui toggle"
        except Exception:
            pass
        self._append("Thinking: "
                     f"{'OFF' if self.cfg.disable_thinking else 'ON'}", "system")

    def _apply_toggle(self, arg: str) -> None:
        value = arg.strip().lower()
        if value in ("", "toggle"):
            self.apply = not self.apply
        elif value in ("on", "true", "yes", "1"):
            self.apply = True
        elif value in ("off", "false", "no", "0"):
            self.apply = False
        else:
            self._append("Usage: /apply on|off", "error")
            return
        self._append(f"Writes: {'ON' if self.apply else 'off (dry run)'}",
                     "system")

    def _skills(self, arg: str) -> None:
        words = arg.split(None, 1)
        action = words[0].lower() if words else "list"
        rest = words[1].strip() if len(words) > 1 else ""
        if action == "list":
            try:
                rows = skills_mod.list_skills(self.root, self.cfg)
            except Exception as exc:
                self._append(f"could not list skills: {exc}", "error")
                return
            for row in rows:
                self._append(f"{row['name']:<28} [{row['scope']:<7}] "
                             f"{row['description']}", "system")
            return
        if action == "show":
            if not rest:
                self._append("Usage: /skills show <name>", "error")
                return
            text, _info = skills_mod.load(self.root, self.cfg, rest)
            self._append(text, "system")
            return
        if action == "draft":
            if not rest:
                self._append("Usage: /skills draft <description>", "error")
                return
            client = self._ensure_client()
            try:
                draft = skills_mod.draft_skill_with_llm(client, rest)
                path = skills_mod.save_skill(self.root, scope="project",
                                             **draft)
            except (LLMError, ValueError, OSError) as exc:
                self._append(f"skills draft: {exc}", "error")
                return
            self._append(f"Drafted skill '{draft['name']}' -> {path}", "system")
            return
        self._append("Usage: /skills [list|show NAME|draft TEXT]", "error")

    def _memory(self, arg: str) -> None:
        words = arg.split(None, 1)
        action = words[0].lower() if words else "list"
        rest = words[1].strip() if len(words) > 1 else ""
        if action == "add":
            if not rest:
                self._append("Usage: /memory add <fact>", "error")
                return
            ok, msg = memory_mod.remember(self.root, rest)
            self._append(("remembered: " if ok else "memory: ") + msg,
                         "system")
            return
        if action == "forget":
            if not rest:
                self._append("Usage: /memory forget <subtext>", "error")
                return
            n = memory_mod.forget(self.root, rest)
            self._append(f"forgot {n} fact(s) matching {rest!r}", "system")
            return
        if action != "list":
            self._append("Usage: /memory [list|add FACT|forget TEXT]",
                         "error")
            return
        rows = memory_mod.entries(self.root)
        if not rows:
            self._append("(nothing remembered yet)", "system")
            return
        for scope, category, text in rows:
            self._append(f"[{scope}/{category}] {text}", "system")

    def _procs(self, arg: str) -> None:
        words = arg.split(None, 1)
        action = words[0].lower() if words else "list"
        rest = words[1].strip() if len(words) > 1 else ""
        if action == "list":
            try:
                self._append(processes.render_list(self.root), "system")
            except Exception as exc:
                self._append(f"could not list processes: {exc}", "error")
            return
        if action in ("logs", "stop"):
            if not rest:
                self._append(f"Usage: /procs {action} <name>", "error")
                return
            try:
                if action == "logs":
                    self._append(processes.logs(self.root, rest), "system")
                else:
                    self._append(processes.stop(self.root, rest), "system")
            except Exception as exc:
                self._append(f"procs {action}: {exc}", "error")
            return
        self._append("Usage: /procs [list|logs NAME|stop NAME]", "error")

    def _diff(self) -> None:
        if self.idx is None:
            try:
                self.idx = index_mod.Index.load(self.root)
            except Exception:
                self.idx = None
        if self.idx is not None:
            try:
                text = gitinfo.tool_git_diff(self.idx, self.cfg)
                self._append(text, "diff")
                return
            except Exception as exc:
                self._append(f"git diff unavailable: {exc}", "diff")
                return
        st = self._git_state()
        if st.get("branch") or st.get("dirty") or st.get("untracked"):
            self._append(f"branch {st.get('branch') or '?'}", "diff")
            for rel in (st.get("dirty") or []) + (st.get("untracked") or []):
                self._append(f"  {rel}", "diff")
            return
        backups = workspace.list_backups(self.root)
        if backups:
            self._append(f"no git diff; newest snapshot {backups[0].name}",
                         "system")
            for path in sorted(backups[0].rglob("*")):
                if path.is_file():
                    self._append("  " + path.relative_to(
                        backups[0]).as_posix(), "diff")
            return
        self._append("No git changes or backup snapshots to show.", "system")

    def _undo(self) -> None:
        try:
            snapshots = workspace.list_backups(self.root)
        except OSError as exc:
            self._append(f"could not list backups: {exc}", "error")
            return
        if not snapshots:
            self._append("No backups under .pedacito/backups/ to restore.",
                         "system")
            return
        snapshot = snapshots[0]
        try:
            files = workspace.restore(self.root, snapshot)
        except OSError as exc:
            self._append(f"undo failed: {exc}", "error")
            return
        self._append(f"undo: restored {len(files)} file(s) from "
                     f"{snapshot.name}", "system")
        for rel in files:
            self._append(f"  restored {rel}", "system")

    def _confirm(self, question: str) -> bool:
        self._set_banner(question)
        self._append(question, "activity")
        self._redraw(force=True)
        answer = self._readline()
        self._set_banner("")
        text = (answer or "").strip().lower()
        return text in ("y", "yes")

    def _reset(self) -> None:
        if not self._confirm("Reset .pedacito state (memory and config kept)? "
                             "(y/N):"):
            self._append("reset cancelled", "system")
            return
        try:
            result = workspace.reset(self.root, keep_memory=True,
                                     keep_config=True)
        except OSError as exc:
            self._append(f"reset failed: {exc}", "error")
            return
        self.transcript = []
        self.activity = []
        self.modified = []
        self.idx = None
        self.agent = None
        self._append(f"reset: removed {result['files']} file(s), kept "
                     f"{', '.join(result['kept']) or '(nothing)'}", "system")
        self._dirty = True

    # --------------------------------------------------------------- terminal
    def _signal_handler(self, _signum, _frame):
        self._restore_terminal()
        raise KeyboardInterrupt

    def _prepare_terminal(self) -> None:
        if not self.interactive:
            return
        enable_vt_mode(self.stdout)
        for sig in (getattr(signal, "SIGINT", None),
                    getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                self._old_signals[sig] = signal.getsignal(sig)
                signal.signal(sig, self._signal_handler)
            except (ValueError, OSError, RuntimeError):
                pass
        self._terminal_ready = True
        self._write(_ALT_SCREEN_ON)

    def _restore_terminal(self) -> None:
        if not self._terminal_ready:
            return
        self._terminal_ready = False
        self._write(_ALT_SCREEN_OFF)
        for sig, handler in list(self._old_signals.items()):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, RuntimeError):
                pass
        self._old_signals.clear()

    def close(self) -> None:
        """Idempotent teardown: watcher, listeners, broker, terminal."""
        self._stop_watcher()
        try:
            if self._listener in self.elog.listeners:
                self.elog.listeners.remove(self._listener)
        except Exception:
            pass
        current = getattr(security, "permission_handler", None)
        try:
            installed = current() if callable(current) else None
        except Exception:
            installed = None
        # Never clobber a broker another session installed after this one.
        if installed is None or installed == self._permission_hook:
            security.set_permission_handler(self._prev_permission_hook)
        self._set_banner("")
        self._restore_terminal()

    # ------------------------------------------------------------------ run
    def run(self, initial_task: str = "") -> int:
        """Paint the session and read commands until EOF or /quit."""
        self._running = True
        try:
            self._prepare_terminal()
            self._redraw(force=True)
            task = initial_task or self.initial_task
            if task:
                self.handle_line(task)
                if self.headless:
                    self._redraw(force=True)
            while not self._quit:
                line = self._readline()
                if line is None:
                    break
                if not line.strip():
                    continue
                self.handle_line(line)
                if self.headless:
                    self._redraw(force=True)
                elif self._dirty:
                    self._redraw()
        except KeyboardInterrupt:
            if self.headless:
                self._append("interrupted", "error")
            else:
                self._write("\n")
        finally:
            self.close()
        return 0

    def handle_command(self, line: str) -> bool:
        """Compatibility alias for callers that drive the session directly."""
        return self.handle_line(line)


def run_tui(root: Path, cfg: Config, *, initial_task: str = "",
            headless: bool = False, stdin=None, stdout=None,
            client=None, apply: bool = True) -> int:
    """Entry point for `pedacito tui`; returns the process exit code.

    `headless=True`, or any non-TTY stdin, reads commands from `stdin` and
    writes complete plain-text frames to `stdout`. This is the deterministic
    path exercised on Linux and Windows CI without a real PTY.
    """
    session = TuiSession(root, cfg, client=client, apply=apply,
                         headless=headless, stdin=stdin, stdout=stdout,
                         initial_task=initial_task)
    return session.run()
