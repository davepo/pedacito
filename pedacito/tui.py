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
(`/compact`), `/profile`, `/auto-route`, `/think`, `/apply`, `/verify`,
`/skills`, `/skill NAME`, `/memory`, `/procs`, `/diff`, `/checkpoints`, `/undo`, `/checkout`,
`/run`, `/fix`, `/update`, `/reset`, `/sidebar`, `/clear`, and `/quit`
(`/exit`).
"""

from __future__ import annotations

import codecs
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

from . import checkpoints as checkpoints_mod
from . import events as events_mod
from . import gitinfo
from . import index as index_mod
from . import llm as llm_mod
from . import loop
from . import memory as memory_mod
from . import modelcard
from . import processes
from . import profiles as profiles_mod
from . import prompts as prompts_mod
from . import roles as roles_mod
from . import runner as runner_mod
from . import security
from . import select as select_mod
from . import servers as servers_mod
from . import skills as skills_mod
from . import testsel
from . import themes as themes_mod
from . import update as update_mod
from . import workspace
from .agent import Agent
from .config import Config
from .llm import LLMClient, LLMError

__all__ = ["TuiSession", "enable_vt_mode", "parse_escape_sequence",
           "run_tui", "HELP_TEXT", "KEYS_TEXT"]

# ANSI sequences. They are emitted only for a real interactive TTY; headless
# frames stay plain text so CI logs and StringIO assertions are deterministic.
# ?1007h is xterm Alternate Scroll: the wheel becomes Up/Down key bursts while
# left-click-drag selection and the OS copy/paste stay native. ?1000h/?1006h
# (SGR click capture) stay OFF by default; /mouse on or F2 enables them.
_ALT_SCREEN_ON = ("\x1b[?1049h\x1b[?1007h\x1b[?1000l\x1b[?1006l"
                  "\x1b[?25l\x1b[2J\x1b[H")
_ALT_SCREEN_OFF = ("\x1b[?1007l\x1b[?1000l\x1b[?1006l"
                   "\x1b[?1049l\x1b[r\x1b[?25h")
_MOUSE_SGR_ON = "\x1b[?1007l\x1b[?1000h\x1b[?1006h"
_MOUSE_SGR_OFF = "\x1b[?1000l\x1b[?1006l\x1b[?1007h"
_SYNC_ON = "\x1b[?2026h"
_SYNC_OFF = "\x1b[?2026l"
_CLEAR = "\x1b[2J\x1b[H"
_HOME = "\x1b[H"
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Everything a repo file, log line or tool result could smuggle into the
# terminal: OSC (window/tab title, hyperlinks, clipboard), DCS/SOS/PM/APC
# strings, CSI, and bare two-byte escapes. Styling is applied at render time,
# never stored in the transcript, so stripping these is lossless for the UI.
_TERM_ESCAPE_RE = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"
    r"|\x1b[PX^_][^\x1b]*(?:\x1b\\)?"
    r"|\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b[@-Z\\-_]"
    r"|\x1b")
_TERM_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _untrusted_text(text) -> str:
    """`text` with terminal escape sequences and control characters removed
    (tabs and newlines kept), so repo or log content cannot retitle the window,
    write to the clipboard or move the cursor outside the pane it was given."""
    return _TERM_CONTROL_RE.sub("", _TERM_ESCAPE_RE.sub("", str(text or "")))
_SGR_MOUSE_RE = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)([Mm])")
_PLAN_RE = re.compile(r"^\s*[-*]\s*\[([ xX])\]\s*(.*)$")
_SWAP_ARROW = "\u2192"

# Prefix completions explicitly requested by the TUI keybinding contract.
_TAB_PREFIX = {
    "/con": "/consolidate ",
    "/pro": "/profile ",
    "/pi": "/pin ",
    "/ru": "/rules ",
    "/aut": "/auto-route ",
    "/cre": "/creative ",
    "/eff": "/effort ",
    "/ser": "/servers",
    "/unl": "/unload ",
    "/hy": "/hybrid ",
    "/ar": "/arena ",
}
_SLASH_COMMANDS = [
    "/help", "/keys", "/status", "/telemetry", "/new", "/auto", "/auto!", "/plan",
    "/consolidate", "/compact", "/profile", "/profiles", "/model",
    "/auto-route", "/think", "/effort", "/creative", "/servers", "/unload",
    "/apply", "/verify", "/skills", "/skill", "/memory", "/procs", "/security",
    "/theme",
    "/pin", "/rules", "/hybrid", "/arena",
    "/diff", "/checkpoints", "/undo", "/checkout", "/run", "/fix",
    "/update",
    "/reset", "/sidebar", "/scroll", "/mouse", "/clear",
    "/quit", "/exit",
]
_KNOWN_KEYS = {
    "ENTER", "TAB", "BACKSPACE", "DELETE", "INSERT", "ESC",
    "UP", "DOWN", "LEFT", "RIGHT", "HOME", "END",
    "PAGE_UP", "PAGE_DOWN", "SHIFT_UP", "SHIFT_DOWN",
    "SHIFT_LEFT", "SHIFT_RIGHT", "ALT_PAGE_UP", "ALT_PAGE_DOWN",
    "WHEEL_UP", "WHEEL_DOWN", "CTRL_C", "CTRL_D", "CTRL_K", "CTRL_L",
    "CTRL_U", "CTRL_W", "F2", "SHIFT_TAB",
}

# Commands shown by /help and in the prompt bar's hint line.
KEYS_TEXT = """\
Pedacito TUI keybindings
  Enter                 send the task or slash command
  PgUp / PgDn           transcript half a page up / down
  Shift+Up / Shift+Down transcript 3 lines up / down
  Ctrl+U / Ctrl+D       half-page scroll when the prompt is empty
  Alt+PgUp / Alt+PgDn   scroll the Activity / Plan sidebar
  Mouse wheel           transcript 3 lines, or the sidebar under pointer
  Up / Down             command history (or scroll when scrolled back)
  Left / Right          move the prompt cursor
  Home / End            start / end of the prompt (End returns to live)
  Esc                   jump to the live bottom
  Backspace / Delete    edit the prompt; Ctrl+W deletes a word
  Ctrl+K / Ctrl+L       kill to end / force a full repaint
  Tab                   complete slash commands and profile names
  F2                    toggle SGR mouse mode (native selection default)
  Ctrl+C                cancel the running task
  /quit or /exit        leave and restore the terminal"""

HELP_TEXT = """\
Pedacito TUI commands
  /help                     this list and the keyboard shortcuts
  /keys                     detailed keybindings and mouse modes
  /status  /telemetry       full telemetry table and per-call-site tokens
  /new DIR PROMPT           scaffold a brand-new project in DIR and switch
                            this session into it before the build starts
                            (Complex scaffolding auto-detected from PROMPT)
  /auto GOAL                plan GOAL, then run every task with live updates
                            (single-pass edits get a Task-mode suggestion
                            first; Enter/y accepts, n keeps Auto)
  /auto! GOAL               force the multi-step Auto loop (skip the
                            Task-mode suggestion)
  /plan [GOAL]              show PLAN.md, or generate it from GOAL
  /consolidate /compact     compact the session into a handoff checkpoint
  /profile [NAME]           list profiles, or switch to NAME
  /profile show|run|reset   hardware snapshot + last probe + active auto-tune;
                            run forces a probe, reset drops cached measurements
  /model ID                 switch to a model id directly
  /auto-route [on|off]      specialist auto-routing on/off
  /hybrid [off | PROVIDER MODEL]
                            split-stage routing: an architect model plans
                            (Stage 1), the active editor implements (Stage 2);
                            off disables it for this session
  /arena A:B PROMPT         run PROMPT on two contestants (profile names or
                            provider:model) in isolated sandboxes, compare
  /arena apply a|b          adopt a winner's changes (checkpointed, undoable)
  /think [on|off]           thinking on/off (off sets disable_thinking)
  /effort [LEVEL]           reasoning effort: default, low, medium or high
                            (forwarded to local-moe and reasoning APIs)
  /servers                  live VRAM/RAM and each local server's status
  /servers pin|unpin SERVER [MODEL]
                            pin/unpin a loaded Lemonade model; LRU eviction
                            skips pinned models (other servers: no-op)
  /unload [SERVER]          unload models on SERVER, or every idle local server
  /creative [on|off|auto]   design mode: on=every task, auto=detect design tasks,
                            off=surgical edits (warmer sampling + whole-file CSS/HTML)
  /theme [NAME]             list color themes, or switch to NAME and save it
                            (shared with the GUI Theme Manager)
  /apply [on|off]           write edits to disk, or preview them
  /verify [on|off|status]   post-edit verification: on detects the project's
                            test command, off clears it, status shows the
                            gates, repair rounds and ratchet
  /skills [list|show NAME|draft TEXT]
  /skill NAME               load a skill's full checklist into the next
                            task through the pinned-context block; /pin
                            remove skill:NAME or /pin clear unpins it
  /memory [list|add FACT|forget TEXT]
  /procs [list|logs NAME|stop NAME]
  /pin [list|add REF|remove REF|clear]
                            explicit context pins for this session; REF is
                            a file path or indexed symbol (Class.method)
  /rules [list|set TEXT|clear]
                            show or edit persistent .pedacito/rules.md
  /security                 scan and show unacknowledged secrets / .env files
  /security ack             acknowledge every current secret finding
  /security gitignore       add .env patterns to .gitignore
  /diff                     git status and working-tree diff
  /checkpoints [N]          show the time-travel checkpoint tree
  /undo                     restore the state before the last run (or the
                            newest .pedacito/backups snapshot on old projects)
  /checkout ID              restore the whole workspace from a checkpoint
                            (a unique id prefix works)
  /run [CMD]                run the detected entrypoint or CMD, show output
  /fix                      send the last /run crash traceback into a repair
                            task
  /update [run]             check PyPI for a newer Pedacito and offer the
                            upgrade (run skips the confirmation; editable
                            installs get git-pull instructions)
  /reset                    wipe .pedacito state (keeps memory and config)
  /sidebar                  toggle the activity / plan pane
  /scroll up|down|top|bottom|N
                            scroll the transcript (also works in --headless)
  /mouse [on|off]           SGR mouse wheel capture (F2); off keeps native
                            text selection and OS copy/paste
  /clear                    clear the transcript pane
  /quit  /exit              leave and restore the terminal

Enter: send | PgUp/PgDn/Wheel: scroll | Up/Down: history | Tab: complete
F2: mouse mode | Esc/End: live bottom | Ctrl+C: cancel | /help | /quit
Plain text is a task: it runs through the normal gather/edit/verify loop.
Web: https://www.pedacito.dev · Contact: contact@pedacito.dev"""


def _windows_key_sequence(code: str) -> str:
    """msvcrt's second byte -> the equivalent VT escape sequence."""
    return {
        "H": "\x1b[A", "P": "\x1b[B", "K": "\x1b[D", "M": "\x1b[C",
        "G": "\x1b[H", "O": "\x1b[F", "S": "\x1b[3~",
        "I": "\x1b[5~", "Q": "\x1b[6~",
        "<": "\x1bOQ",           # F2 on the Windows console
        "R": "\x1b[2~",          # Insert (recognised as UNKNOWN)
    }.get(code, "")


def parse_escape_sequence(buf: str) -> tuple[str, int]:
    """Parse the first key event in `buf`; return `(key_name, consumed)`.

    `key_name` is one of the named navigation/editing keys, a single printable
    character, or `""` when the buffer holds an incomplete sequence (the raw
    reader then waits for more bytes). Bursts of two or more Up/Down arrows in
    the same drain window are classified as WHEEL_UP/WHEEL_DOWN: that is the
    `?1007h` Alternate Scroll contract terminals use for the mouse wheel.
    """
    if not buf:
        return "", 0

    # SGR mouse wheel, before the generic CSI checks.
    match = _SGR_MOUSE_RE.match(buf)
    if match:
        button = int(match.group(1))
        key = {64: "WHEEL_UP", 65: "WHEEL_DOWN"}.get(button)
        return (key, match.end()) if key else ("UNKNOWN", match.end())

    # Alternate-scroll bursts. Count repeated sequences so one drain returns
    # WHEEL rather than leaking N history events into the prompt.
    for seq, key in (("\x1b[A", "UP"), ("\x1b[B", "DOWN"),
                     ("\xe0H", "UP"), ("\xe0P", "DOWN")):
        if buf.startswith(seq):
            count = 0
            pos = 0
            while buf.startswith(seq, pos):
                count += 1
                pos += len(seq)
            if count >= 2:
                return ("WHEEL_UP" if key == "UP" else "WHEEL_DOWN", pos)
            return key, len(seq)

    exact = {
        "\x1b[5~": "PAGE_UP",
        "\x1b[6~": "PAGE_DOWN",
        "\x1b[5;3~": "ALT_PAGE_UP",
        "\x1b[6;3~": "ALT_PAGE_DOWN",
        "\x1b[1;2A": "SHIFT_UP",
        "\x1b[1;2B": "SHIFT_DOWN",
        "\x1b[1;2C": "SHIFT_RIGHT",
        "\x1b[1;2D": "SHIFT_LEFT",
        "\x1b[3~": "DELETE",
        "\x1b[2~": "INSERT",
        "\x1b[H": "HOME",
        "\x1b[F": "END",
        "\x1b[1~": "HOME",
        "\x1b[4~": "END",
        "\x1b[7~": "HOME",
        "\x1b[8~": "END",
        "\x1bOQ": "F2",
        "\x1b[12~": "F2",
        "\x1b[Z": "SHIFT_TAB",
        "\x00I": "PAGE_UP", "\xe0I": "PAGE_UP",
        "\x00Q": "PAGE_DOWN", "\xe0Q": "PAGE_DOWN",
        "\x00G": "HOME", "\xe0G": "HOME",
        "\x00O": "END", "\xe0O": "END",
        "\x00S": "DELETE", "\xe0S": "DELETE",
        "\x00<": "F2", "\xe0<": "F2",
    }
    for seq, key in exact.items():
        if buf.startswith(seq):
            return key, len(seq)

    # Remaining SS3 sequences (F1-F4 and friends): consume one final byte.
    if buf.startswith("\x1bO"):
        if len(buf) < 3:
            return "", 0
        return "UNKNOWN", 3

    # Generic CSI sequence. Incomplete prefixes wait for more bytes.
    if buf.startswith("\x1b["):
        match = re.match(r"\x1b\[[0-9;?]*[A-Za-z~]", buf)
        if match:
            return "UNKNOWN", match.end()
        return "", 0

    # A lone ESC: the reader gives it a short drain grace before returning it,
    # so Alt+key sequences are not mistaken for Escape.
    if buf.startswith("\x1b"):
        if len(buf) == 1:
            return "ESC", 1
        return "ALT_" + buf[1].upper(), 2

    char = buf[0]
    controls = {
        "\r": "ENTER", "\n": "ENTER", "\t": "TAB",
        "\x7f": "BACKSPACE", "\x08": "BACKSPACE",
        "\x01": "HOME", "\x05": "END",
        "\x03": "CTRL_C", "\x04": "CTRL_D",
        "\x0b": "CTRL_K", "\x0c": "CTRL_L",
        "\x15": "CTRL_U", "\x17": "CTRL_W",
    }
    if char in controls:
        return controls[char], 1
    return char, 1


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
        # The active palette is shared with the GUI. Loading is tolerant:
        # a hand-edited themes.json can never stop the session from opening.
        try:
            self.theme_name = themes_mod.get_active_theme_name(self.root)
            self.theme = themes_mod.get_active_theme(self.root)
        except Exception:
            self.theme_name = themes_mod.DEFAULT_THEME
            self.theme = dict(themes_mod.BUILTIN_THEMES[themes_mod.DEFAULT_THEME])
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
        # Prompt editing and viewport state. scroll_offset == 0 is pinned to
        # the live bottom; positive values mean "N wrapped lines back".
        self.input_buf = ""
        self.cursor_pos = 0
        self.command_history: list[str] = []
        self._history_pos = 0
        self._history_draft = ""
        self.scroll_offset = 0
        self.sidebar_scroll_offset = 0
        self.mouse_sgr = False
        self._last_mouse_col = 0
        self._prompt_view_start = 0
        self._raw_buf = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._old_termios = None
        self._raw_mode = False
        self._last_render_scroll_total = 0
        self._scroll_seen_total = 0
        self._scroll_seen_width = 0
        self.folds = 0
        self.resume_context = ""
        # The last failed /run result, kept so /fix can turn the exact
        # command, exit code and traceback into a repair task. `last_traceback`
        # is the same text a scripted caller would display.
        self.last_run_failure: dict | None = None
        self.last_traceback = ""
        # Explicit session pins added with `/pin add`. They ride every Task
        # and Auto prompt through cfg._pinned_refs, independently of the
        # prompt's own @file/@symbol mentions.
        self.pins: list[str] = []
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
        # Compact live stream indicator, updated by `llm_stream` events and
        # shown before the prompt while a reply is being generated:
        # "[thinking... 412 tok | 38.4 tok/s]".
        self._stream_status = ""
        self._stream_phase = ""
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
        # Security notices already printed in this session (fingerprints and
        # unignored-file keys), so a run does not repeat the same warning.
        self._security_notified: set[str] = set()
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
        # Quiet release check on a daemon thread (pedacito.update): never
        # delays startup by a network round-trip; silent when offline or
        # opted out (check_for_updates = false).
        self.update_info: dict | None = None
        self._update_thread = update_mod.check_for_update_async(
            self.cfg, on_done=self._note_update)

    # ---------------------------------------------------------------- state
    @property
    def profile(self) -> str:
        return str(getattr(self.cfg, "active_profile", "") or "default")

    @property
    def pinned(self) -> list[str]:
        """Canonical session pin references (the `/pin list` contents)."""
        return list(self.pins)

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

    # ---------------------------------------------------------------- theme
    def _colors_on(self) -> bool:
        """ANSI styling needs an interactive color terminal and no NO_COLOR."""
        return bool(self.color) and not os.environ.get("NO_COLOR")

    def _style(self, text: str, token: str | None = None, *,
               fg: str | None = None, bold: bool = False) -> str:
        """Paint `text` with one token from the active shared palette."""
        if not self._colors_on():
            return text
        color = fg or self.theme.get(token or "")
        if not color:
            return text
        return themes_mod.style_text(text, fg=color, bold=bold)

    def _reload_theme(self) -> None:
        try:
            self.theme_name = themes_mod.get_active_theme_name(self.root)
            self.theme = themes_mod.get_active_theme(self.root)
        except Exception:
            self.theme_name = themes_mod.DEFAULT_THEME
            self.theme = dict(themes_mod.BUILTIN_THEMES[themes_mod.DEFAULT_THEME])
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
        bar, _color, state = _meter(pct, 16)
        meter_token = ("danger" if state == "high"
                       else "warning" if state == "warn" else "success")
        painted = self._style(bar, meter_token)
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
        state_note = (self._style(" !HIGH", "danger", bold=True)
                      if state == "high" else
                      self._style(" !WARN", "warning", bold=True)
                      if state == "warn" else "")
        active_text = self._style(active, "accent")
        label = lambda text: self._style(text, "muted")  # noqa: E731

        if width >= 120:
            line1 = (
                f"{label('Ctx:')} {painted} {prompt:,} / {context:,} "
                f"({pct:.1f}%) | {label('Turn:')} +{prompt:,} in / "
                f"+{completion:,} out | {label('Speed:')} {speed:.1f} tok/s "
                f"(avg {avg:.1f} tok/s){state_note}")
            line2 = (
                f"{label('Session:')} "
                f"{_compact_tokens(t['session_prompt_tokens'])} in / "
                f"{_compact_tokens(t['session_completion_tokens'])} out | "
                f"{label('KV Cache:')} {cache:.0f}% | "
                f"{label('Folds:')} {folds} | {label('Active:')} "
                f"{active_text}")
        else:
            line1 = (f"{label('Ctx:')} {prompt:,} / {context:,} ({pct:.1f}%) "
                     f"{painted} {speed:.1f} tok/s{state_note}")
            line2 = (f"{label('Turn:')} +{prompt:,}/+{completion:,} | "
                     f"{label('Sess:')} "
                     f"{_compact_tokens(t['session_prompt_tokens'])}/"
                     f"{_compact_tokens(t['session_completion_tokens'])} | "
                     f"{label('KV:')} {cache:.0f}% | {active_text}")
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
        sep = self._style("\u2502", "muted")
        brand = self._style(" pdc tui ", "accent", bold=True)
        profile_text = self._style(profile, "accent")
        model_text = self._style(model, "text")
        badge_text = self._style(
            badge, "warning" if badge.startswith("\u2601") else "success")
        line1 = (f"{brand}{sep} {self._style(project, 'text')} "
                 f"{sep} profile: {profile_text} {sep} model: {model_text} "
                 f"{sep} {badge_text}")
        line2 = (f" {self._style('Auto-Route:', 'muted')} "
                 f"{self._style(routing, 'success' if routing == 'ON' else 'muted')} "
                 f"{sep} {self._style('Scout:', 'muted')} "
                 f"{self._style(scout, 'text')} "
                 f"{sep} {self._style('git:', 'muted')} "
                 f"{self._style(branch + ('*' if dirty else ''), 'warning' if dirty else 'text')} "
                 f"{sep} {self._style('sidebar:', 'muted')} "
                 f"{self._style(sidebar, 'text')}")
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

    def _all_sidebar_rows(self, width: int) -> list[str]:
        width = max(20, width)
        rows: list[str] = ["ACTIVITY & PLAN"]
        items = self.plan_items()
        rows.append(f"PLAN: {self._plan_path().name}")
        if items:
            for checked, text in items:
                mark = "x" if checked else " "
                rows.append(f"[{mark}] {text}")
        else:
            rows.append("[ ] (no PLAN.md tasks yet)")
        rows.append("")
        rows.append("LIVE ACTIVITY / ROUTING")
        if self.activity:
            for row in self.activity:
                rows.append(row if row.startswith("\u00b7 ") else f"\u00b7 {row}")
        else:
            rows.append("\u00b7 (waiting for a task)")
        rows.append("")
        rows.append("MODIFIED FILES")
        if self.modified:
            for rel in self.modified:
                rows.append(f"~ {rel}")
        else:
            rows.append("~ (none yet)")
        return rows

    def _sidebar_lines(self, width: int, height: int) -> list[str]:
        rows = self._all_sidebar_rows(width)
        viewport = max(1, height)
        total = len(rows)
        limit = max(0, total - viewport)
        self.sidebar_scroll_offset = max(
            0, min(int(self.sidebar_scroll_offset), limit))
        offset = int(self.sidebar_scroll_offset)
        end = max(0, total - offset)
        start = max(0, end - viewport)
        visible = rows[start:end]
        if offset > 0:
            pill = f"[ \u2191 sidebar {offset} lines \u2014 Alt+PgDn ]"
            if visible:
                visible[-1] = pill
            else:
                visible = [pill]
        return visible

    # ------------------------------------------------------------- painting
    def _show_sidebar(self, width: int | None = None) -> bool:
        width = width if width is not None else self._size()[0]
        return bool(self.sidebar and width >= 100)

    def _transcript_width(self, width: int | None = None) -> int:
        width = width if width is not None else self._size()[0]
        if self._show_sidebar(width):
            side_width = min(46, max(32, width // 3))
            left_width = width - side_width - 1
            return max(1, left_width - 1)
        return max(1, width - 1)

    def _body_height(self, height: int | None = None) -> int:
        height = height if height is not None else self._size()[1]
        header = 2
        footer = 2
        return max(4, height - header - footer - 2)

    def _transcript_viewport_height(self, height: int | None = None) -> int:
        return max(1, self._body_height(height) - 1)

    def _wrap_transcript_line(self, kind: str, text: str,
                             width: int) -> list[str]:
        prefixes = {
            "banner": "\u25c6 ",
            "user": "\u00bb ",
            "assistant": "\u25c6 ",
            "system": "  ",
            "activity": "  ",
            "receipt": "  ",
            "success": "  ",
            "warn": "! ",
            "diff": "  ",
            "error": "! ",
        }
        tokens = {
            "banner": "accent", "user": "accent", "assistant": "accent",
            "receipt": "success", "success": "success",
            "warn": "warning", "error": "danger",
        }
        prefix = prefixes.get(kind, "  ")
        styled_prefix = self._style(prefix, tokens.get(kind), bold=True) \
            if kind in tokens else prefix
        whole_line = kind in ("receipt", "success", "warn", "error")
        body = str(text)
        chunks: list[str] = []
        for raw in body.splitlines() or [""]:
            start = len(chunks)
            chunks.extend(textwrap.wrap(
                prefix + raw, width=max(1, width),
                replace_whitespace=False, drop_whitespace=False,
                break_long_words=True, break_on_hyphens=False) or [prefix])
            if whole_line:
                chunks[start:] = [self._style(chunk, tokens.get(kind))
                                  for chunk in chunks[start:]]
            elif styled_prefix != prefix and len(chunks) > start:
                lead = chunks[start]
                if lead.startswith(prefix):
                    chunks[start] = styled_prefix + lead[len(prefix):]
            prefix = "  "
        return chunks

    def _all_transcript_rows(self, width: int | None = None) -> list[str]:
        width = width if width is not None else self._transcript_width()
        if not self.transcript:
            source = [
                ("banner", f"Pedacito TUI \u2014 {self.root}"),
                ("system", "Type a coding task, or /help for commands."),
                ("system", "PgUp/PgDn or the mouse wheel scrolls the log."),
            ]
        else:
            source = self.transcript
        rows: list[str] = []
        for kind, text in source:
            rows.extend(self._wrap_transcript_line(kind, text, width))
        return rows

    def max_scroll(self) -> int:
        """Largest useful transcript scroll_offset for the current size."""
        width, height = self._size()
        total = len(self._all_transcript_rows(self._transcript_width(width)))
        return max(0, total - self._transcript_viewport_height(height))

    def sidebar_max_scroll(self) -> int:
        width, height = self._size()
        if not self._show_sidebar(width):
            return 0
        side_width = min(46, max(32, width // 3))
        total = len(self._all_sidebar_rows(max(20, side_width - 1)))
        return max(0, total - self._body_height(height))

    def _sync_scroll_baseline(self) -> int:
        """Record the current wrapped-line total before changing the offset."""
        width = self._transcript_width()
        total = len(self._all_transcript_rows(width))
        self._scroll_seen_width = width
        self._scroll_seen_total = total
        return total

    def _clamp_scroll(self, width: int | None = None,
                      height: int | None = None) -> None:
        width = width if width is not None else self._size()[0]
        height = height if height is not None else self._size()[1]
        total = len(self._all_transcript_rows(self._transcript_width(width)))
        # Direct transcript appends (or event listeners from another thread)
        # are not required to call _append. If the wrapped total grew while
        # the user was scrolled back, preserve the text under their eyes.
        if self._scroll_seen_width == width:
            if self.scroll_offset > 0 and total > self._scroll_seen_total:
                self.scroll_offset += total - self._scroll_seen_total
        self._scroll_seen_width = width
        self._scroll_seen_total = total
        limit = max(0, total - self._transcript_viewport_height(height))
        self.scroll_offset = max(0, min(int(self.scroll_offset), limit))
        if self._show_sidebar(width):
            side_width = min(46, max(32, width // 3))
            side_total = len(self._all_sidebar_rows(max(20, side_width - 1)))
            side_limit = max(0, side_total - self._body_height(height))
            self.sidebar_scroll_offset = max(
                0, min(int(self.sidebar_scroll_offset), side_limit))
        else:
            self.sidebar_scroll_offset = 0

    def scroll_transcript(self, delta: int) -> int:
        """Positive = up/back, negative = down/toward live. Returns offset."""
        self._sync_scroll_baseline()
        self.scroll_offset = max(
            0, min(self.max_scroll(), int(self.scroll_offset) + int(delta)))
        self._dirty = True
        return self.scroll_offset

    def scroll_sidebar(self, delta: int) -> int:
        self.sidebar_scroll_offset = max(
            0, min(self.sidebar_max_scroll(),
                   int(self.sidebar_scroll_offset) + int(delta)))
        self._dirty = True
        return self.sidebar_scroll_offset

    def scroll_to_top(self) -> int:
        self._sync_scroll_baseline()
        self.scroll_offset = self.max_scroll()
        self._dirty = True
        return self.scroll_offset

    def scroll_to_bottom(self) -> int:
        self.scroll_offset = 0
        self._dirty = True
        return self.scroll_offset

    def _transcript_rows(self, width: int, height: int) -> list[str]:
        rows = self._all_transcript_rows(width)
        viewport = max(1, height)
        total = len(rows)
        limit = max(0, total - viewport)
        self.scroll_offset = max(0, min(int(self.scroll_offset), limit))
        offset = int(self.scroll_offset)
        end = max(0, total - offset)
        start = max(0, end - viewport)
        visible = rows[start:end]
        if offset > 0:
            pill = (f"[ \u2191 Scrolled back {offset} lines \u2014 press "
                    "PgDn / End / Esc to jump to live bottom ]")
            if visible:
                visible[-1] = pill
            else:
                visible = [pill]
        self._last_render_scroll_total = total
        return visible

    def _prompt_prefix(self) -> str:
        prefix = f"pdc tui ({self.profile}) > "
        if self._stream_status:
            return f"{self._stream_status} {prefix}"
        return prefix

    def _prompt_visible(self, width: int) -> tuple[str, int]:
        """Prompt line plus the cursor's 0-based column inside it.

        When the typed line is wider than the terminal, the visible window
        follows the cursor so editing never runs off-screen.
        """
        prefix = self._prompt_prefix()
        if self._prompt_banner:
            prefix = f"{self._prompt_banner} "
        avail = max(1, int(width) - len(prefix))
        buf = self.input_buf
        cursor = max(0, min(int(self.cursor_pos), len(buf)))
        if len(buf) <= avail:
            start = 0
            shown = buf
        else:
            start = max(0, cursor - avail + 1)
            start = min(start, max(0, len(buf) - avail))
            shown = buf[start:start + avail]
        self._prompt_view_start = start
        line = prefix + shown
        col = min(max(0, len(prefix) + (cursor - start)), max(0, width - 1))
        if self._colors_on():
            if self._prompt_banner:
                token = ("danger" if "denied" in prefix.lower()
                         or "security" in prefix.lower() else "warning")
                display_prefix = self._style(prefix, token, bold=True)
            elif self._stream_status:
                display_prefix = self._style(prefix, "muted")
            else:
                display_prefix = self._style(prefix, "accent", bold=True)
            line = display_prefix + shown
        return line, col

    def _cursor_screen_position(self, width: int,
                                height: int) -> tuple[int, int]:
        _line, col = self._prompt_visible(width)
        # The prompt is the final frame row; terminal coordinates are 1-based.
        return max(1, int(height)), max(1, min(int(width), col + 1))

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
        self._clamp_scroll(width, height)
        show_sidebar = self.sidebar if sidebar is None else bool(sidebar)
        show_sidebar = show_sidebar and width >= 100

        header = self.header_lines(width)
        footer = self.telemetry_lines(width)
        hint = ("Enter: send | PgUp/PgDn/Wheel: scroll | \u2191/\u2193: history | "
                "Tab: complete | F2: mouse mode | /help | /quit")
        prompt_text, _cursor_col = self._prompt_visible(width)
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
                row, col = self._cursor_screen_position(width, height)
                self._write(_SYNC_ON + _HOME + frame + "\x1b[J"
                            + _HIDE_CURSOR + f"\x1b[{row};{col}H"
                            + _SHOW_CURSOR + _SYNC_OFF)
            else:
                self._write(frame + "\n")
        self._dirty = False

    def _append(self, text: str, kind: str = "system") -> None:
        body = _untrusted_text(text)
        if len(body) > 12000:
            body = body[:12000] + "\n[truncated in the transcript view]"
        lines = body.splitlines() or [""]
        width = self._transcript_width()
        added = sum(len(self._wrap_transcript_line(kind, line, width))
                    for line in lines)
        if self.scroll_offset > 0:
            # Keep the line the user is reading stationary: the viewport grew
            # by exactly this many wrapped rows, so offset grows with it.
            self.scroll_offset += max(0, added)
        same_width = self._scroll_seen_width == width
        for line in lines:
            self.transcript.append((kind, line))
        if same_width:
            self._scroll_seen_total += max(0, added)
        else:
            self._scroll_seen_width = width
            self._scroll_seen_total = len(self._all_transcript_rows(width))
        self._dirty = True

    def append(self, text: str, kind: str = "system") -> None:
        """Public log append (keeps a scrolled-back viewport stationary)."""
        self._append(text, kind)

    def log(self, text: str, kind: str = "system") -> None:
        self._append(text, kind)

    def _append_many(self, text: str, kind: str = "system") -> None:
        self._append(text, kind)

    def _set_banner(self, text: str) -> None:
        self._prompt_banner = _untrusted_text(text)

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
        if kind == "llm_stream":
            # Throttled live deltas: keep one compact status line instead of
            # appending every token to the transcript (which would flood it).
            phase = str(fields.get("phase") or "")
            tokens = _as_int(fields.get("tokens_so_far"))
            speed = _as_float(fields.get("tok_per_sec"))
            label = "thinking..." if phase == "reasoning" else "generating..."
            self._stream_phase = phase
            self._stream_status = f"[{label} {tokens} tok | {speed:.1f} tok/s]"
            self._dirty = True
            return
        if kind == "llm_start":
            self._stream_status = ""
            self._stream_phase = ""
            provider = str(fields.get("provider") or "")
            model = str(fields.get("model") or "")
            label = (f"{provider}/{model}" if provider and model
                     else model or provider or "model")
            phase = str(fields.get("phase") or "")
            tail = f" ({phase})" if phase else ""
            self.activity.append(f"[llm] Waiting for {label}...{tail}")
        elif kind == "complex_auto":
            text = str(fields.get("text") or "")
            if text:
                self.activity.append(text)
                self._append(text, "system")
        elif kind == "step":
            action = str(fields.get("action") or "step")
            detail = str(fields.get("detail") or fields.get("file") or "")
            if fields.get("file") and fields.get("file") not in detail:
                detail = f"{fields['file']} {detail}".strip()
            self.activity.append(f"{action} {detail}".rstrip())
        elif kind == "call":
            self._stream_status = ""
            self._stream_phase = ""
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
        elif kind == "hybrid_stage":
            label = str(fields.get("label") or
                        f"Hybrid Stage {fields.get('stage', '?')}")
            self.activity.append(label)
            self._append(label, "system")
            model = str(fields.get("model") or "")
            if model:
                self._event_active = model
        elif kind == "arena_stage":
            label = str(fields.get("label") or "arena")
            self.activity.append(label)
            if str(fields.get("phase") or "") in ("arena_start", "start"):
                self._append(label, "system")
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
        answer = self._read_answer_line()
        self.input_buf = ""
        self.cursor_pos = 0
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
        """One line, or None at EOF. Headless scripts use this stream reader."""
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

    # ------------------------------------------------------- raw key input
    def _enter_raw_mode(self) -> None:
        """cbreak on POSIX (signals still work); msvcrt needs no mode."""
        if os.name == "nt":
            return
        try:
            import termios
            import tty
            fd = self.stdin.fileno()
            self._old_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self._raw_mode = True
        except (AttributeError, ImportError, OSError, ValueError):
            self._old_termios = None
            self._raw_mode = False

    def _restore_raw_mode(self) -> None:
        if not self._raw_mode:
            return
        self._raw_mode = False
        attrs = self._old_termios
        self._old_termios = None
        if attrs is None or os.name == "nt":
            return
        try:
            import termios
            fd = self.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        except (AttributeError, ImportError, OSError, ValueError):
            pass

    def _read_raw_chunk(self, wait: float) -> bool:
        """Read available bytes/keys into `self._raw_buf`; False on timeout."""
        wait = max(0.0, float(wait))
        fd = None
        try:
            fd = self.stdin.fileno()
        except (AttributeError, OSError, ValueError):
            fd = None
        # StringIO/fake TTY tests (and redirected stdin) use the deterministic
        # line-backed fallback on every platform, including Windows CI.
        if fd is None:
            return self._read_fake_chunk(wait)
        if os.name == "nt":
            return self._read_windows_chunk(wait)
        try:
            import select
            ready, _w, _x = select.select([fd], [], [], wait)
        except (ImportError, OSError, ValueError):
            return False
        if not ready:
            return False
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            return False
        if not chunk:
            return False
        self._raw_buf += self._decoder.decode(chunk)
        return True

    def _read_windows_chunk(self, wait: float) -> bool:
        try:
            import msvcrt
        except ImportError:
            return False
        deadline = time.monotonic() + wait
        while not msvcrt.kbhit():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        text = ""
        while msvcrt.kbhit():
            char = msvcrt.getwch()
            if char in ("\x00", "\xe0"):
                code = msvcrt.getwch() if msvcrt.kbhit() else ""
                text += _windows_key_sequence(code)
            else:
                text += char
        if not text:
            return False
        self._raw_buf += text
        return True

    def _read_fake_chunk(self, wait: float) -> bool:
        """Deterministic fallback for StringIO/fake TTY tests."""
        if self._raw_buf:
            return True
        line = self._readline()
        if line is None:
            return False
        self._raw_buf += line + "\n"
        return True

    def read_key_event(self, timeout: float = 0.05) -> str | None:
        """Read one parsed key event, or None when `timeout` expires.

        POSIX uses cbreak + `select`; Windows uses `msvcrt.kbhit`/`getwch`.
        After the first bytes arrive, a 12 ms drain window catches wheel
        bursts from `?1007h` terminls and complete escape sequences.
        """
        if not self.interactive:
            return None
        deadline = time.monotonic() + max(0.0, float(timeout))
        if not self._raw_buf:
            while not self._raw_buf:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._read_raw_chunk(min(0.02, remaining))
        # A wheel notch arrives as a rapid burst of arrows. Wait 12 ms once
        # data is present, without extending a tiny explicit timeout.
        drain_until = time.monotonic() + 0.012
        stop = min(deadline, drain_until) if timeout < 0.012 else drain_until
        while time.monotonic() < stop:
            remaining = stop - time.monotonic()
            if not self._read_raw_chunk(max(0.0, remaining)):
                break
        while True:
            key, used = parse_escape_sequence(self._raw_buf)
            if key:
                # A lone ESC might be the start of Alt+key or a CSI sequence.
                if key == "ESC" and len(self._raw_buf) == 1:
                    if self._read_raw_chunk(0.012):
                        continue
                if key in ("WHEEL_UP", "WHEEL_DOWN"):
                    match = _SGR_MOUSE_RE.match(self._raw_buf)
                    if match:
                        self._last_mouse_col = int(match.group(2))
                    else:
                        self._last_mouse_col = 0
                self._raw_buf = self._raw_buf[used:]
                return key
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if not self._read_raw_chunk(min(0.02, remaining)):
                return None

    # ----------------------------------------------------- prompt editing
    def _insert_text(self, text: str) -> None:
        pos = max(0, min(int(self.cursor_pos), len(self.input_buf)))
        self.input_buf = (self.input_buf[:pos] + text
                          + self.input_buf[pos:])
        self.cursor_pos = pos + len(text)
        self._dirty = True

    def _history_prev(self) -> None:
        if not self.command_history:
            return
        # A caller may seed command_history directly; treat a mismatched
        # position as "at the draft" so the typed text is saved and restored.
        if (self._history_pos >= len(self.command_history)
                or (self._history_pos == 0 and self.input_buf
                    and not self._history_draft
                    and self.input_buf != self.command_history[0])):
            self._history_draft = self.input_buf
            self._history_pos = len(self.command_history)
        self._history_pos = max(0, self._history_pos - 1)
        self.input_buf = self.command_history[self._history_pos]
        self.cursor_pos = len(self.input_buf)
        self._dirty = True

    def _history_next(self) -> None:
        if not self.command_history:
            return
        if self._history_pos < len(self.command_history) - 1:
            self._history_pos += 1
            self.input_buf = self.command_history[self._history_pos]
        else:
            self._history_pos = len(self.command_history)
            self.input_buf = self._history_draft
        self.cursor_pos = len(self.input_buf)
        self._dirty = True

    def _submit_input(self) -> bool:
        line = self.input_buf.strip()
        self.input_buf = ""
        self.cursor_pos = 0
        self.scroll_offset = 0
        self.sidebar_scroll_offset = 0
        self._history_pos = len(self.command_history)
        self._history_draft = ""
        if not line:
            self._dirty = True
            return not self._quit
        if not self.command_history or self.command_history[-1] != line:
            self.command_history.append(line)
        self._history_pos = len(self.command_history)
        result = self.handle_line(line)
        self._dirty = True
        return result

    def _complete(self) -> None:
        text = self.input_buf
        pos = max(0, min(int(self.cursor_pos), len(text)))
        pre, post = text[:pos], text[pos:]
        stripped = pre.lstrip()
        if not stripped.startswith("/"):
            return
        # A slash command name with no argument yet.
        if " " not in stripped:
            completion = _TAB_PREFIX.get(stripped)
            if completion is None:
                matches = [cmd for cmd in _SLASH_COMMANDS
                           if cmd.startswith(stripped)]
                if len(matches) == 1:
                    completion = matches[0] + " "
                elif len(matches) > 1:
                    common = os.path.commonprefix(matches)
                    if len(common) > len(stripped):
                        completion = common
            if completion and completion != stripped:
                self.input_buf = completion + post
                self.cursor_pos = len(completion)
                self._dirty = True
            return
        # Profile-name completion after /profile or /profiles.
        head, _sep, tail = stripped.partition(" ")
        if head not in ("/profile", "/profiles"):
            return
        token = tail.split()[-1] if tail.split() else ""
        names = sorted(str(n) for n in (getattr(self.cfg, "profiles", None)
                                        or {}))
        matches = [name for name in names if name.startswith(token)]
        if not matches:
            return
        completed = os.path.commonprefix(matches)
        if len(matches) == 1:
            completed += " "
        self.input_buf = f"{head} {completed}" + (
            post if post and not post.startswith(" ") else "")
        self.cursor_pos = len(self.input_buf)
        self._dirty = True

    def _sidebar_column(self, col: int) -> bool:
        width, _height = self._size()
        if not self._show_sidebar(width):
            return False
        side_width = min(46, max(32, width // 3))
        left_width = width - side_width - 1
        return int(col) >= left_width + 2

    def _toggle_mouse(self, on: bool | None = None) -> bool:
        self.mouse_sgr = (not self.mouse_sgr) if on is None else bool(on)
        if self.interactive and self._terminal_ready:
            self._write(_MOUSE_SGR_ON if self.mouse_sgr else _MOUSE_SGR_OFF)
        described = "SGR wheel capture ON (click capture on; use F2 to restore "\
                    "native selection)" if self.mouse_sgr else \
                    "native selection ON (?1007h wheel-to-arrow; SGR mouse off)"
        self._append(f"Mouse mode: {described}", "system")
        self._dirty = True
        return self.mouse_sgr

    def handle_key(self, key: str, *, col: int | None = None) -> bool:
        """Apply one parsed key event to the prompt or viewport."""
        key = str(key)
        width, height = self._size()
        half = max(1, self._transcript_viewport_height(height) // 2)
        side_half = max(1, self._body_height(height) // 2)
        handled = True

        if key == "F2":
            self._toggle_mouse()
        elif key == "PAGE_UP":
            self.scroll_transcript(half)
        elif key == "PAGE_DOWN":
            self.scroll_transcript(-half)
        elif key == "SHIFT_UP":
            self.scroll_transcript(3)
        elif key == "SHIFT_DOWN":
            self.scroll_transcript(-3)
        elif key == "ALT_PAGE_UP":
            self.scroll_sidebar(side_half)
        elif key == "ALT_PAGE_DOWN":
            self.scroll_sidebar(-side_half)
        elif key in ("WHEEL_UP", "WHEEL_DOWN"):
            delta = 3 if key == "WHEEL_UP" else -3
            use_col = self._last_mouse_col if col is None else int(col)
            if use_col and self._sidebar_column(use_col):
                self.scroll_sidebar(delta)
            else:
                self.scroll_transcript(delta)
        elif key == "CTRL_U":
            if self.input_buf:
                self.input_buf = self.input_buf[self.cursor_pos:]
                self.cursor_pos = 0
                self._dirty = True
            else:
                self.scroll_transcript(half)
        elif key == "CTRL_D":
            if self.input_buf:
                pos = self.cursor_pos
                self.input_buf = self.input_buf[:pos] + self.input_buf[pos + 1:]
                self._dirty = True
            else:
                self.scroll_transcript(-half)
        elif key == "CTRL_K":
            self.input_buf = self.input_buf[:self.cursor_pos]
            self._dirty = True
        elif key == "CTRL_L":
            self._dirty = True
            self._redraw(force=True)
        elif key == "CTRL_W":
            pos = self.cursor_pos
            start = pos
            while start > 0 and self.input_buf[start - 1].isspace():
                start -= 1
            while start > 0 and not self.input_buf[start - 1].isspace():
                start -= 1
            self.input_buf = self.input_buf[:start] + self.input_buf[pos:]
            self.cursor_pos = start
            self._dirty = True
        elif key == "ENTER":
            self._submit_input()
        elif key == "TAB":
            self._complete()
        elif key == "ESC":
            if self.input_buf:
                self._dirty = True
            else:
                self.scroll_to_bottom()
                self.sidebar_scroll_offset = 0
        elif key == "UP":
            if not self.input_buf and self.scroll_offset > 0:
                self.scroll_transcript(1)
            else:
                self._history_prev()
        elif key == "DOWN":
            if not self.input_buf and self.scroll_offset > 0:
                self.scroll_transcript(-1)
            else:
                self._history_next()
        elif key == "LEFT":
            self.cursor_pos = max(0, self.cursor_pos - 1)
            self._dirty = True
        elif key == "RIGHT":
            self.cursor_pos = min(len(self.input_buf), self.cursor_pos + 1)
            self._dirty = True
        elif key == "HOME":
            self.cursor_pos = 0
            self._dirty = True
        elif key == "END":
            self.cursor_pos = len(self.input_buf)
            self.scroll_offset = 0
            self.sidebar_scroll_offset = 0
            self._dirty = True
        elif key == "BACKSPACE":
            if self.cursor_pos > 0:
                pos = self.cursor_pos
                self.input_buf = self.input_buf[:pos - 1] + self.input_buf[pos:]
                self.cursor_pos = pos - 1
                self._dirty = True
        elif key == "DELETE":
            pos = self.cursor_pos
            if pos < len(self.input_buf):
                self.input_buf = self.input_buf[:pos] + self.input_buf[pos + 1:]
                self._dirty = True
        elif key == "CTRL_C":
            if self.input_buf:
                self.input_buf = ""
                self.cursor_pos = 0
                self._dirty = True
        elif len(key) == 1 and key >= " ":
            self._insert_text(key)
        else:
            handled = False
        if handled:
            self._dirty = True
        return handled

    def feed_key_event(self, key: str, *, col: int | None = None) -> bool:
        return self.handle_key(key, col=col)

    def feed_key(self, key: str, *, col: int | None = None) -> bool:
        return self.handle_key(key, col=col)

    def handle_key_event(self, key: str, *, col: int | None = None) -> bool:
        """Alias used by programmatic drivers and tests."""
        return self.handle_key(key, col=col)

    def feed_keys(self, keys, *, col: int | None = None) -> str:
        """Apply key names, raw escape sequences or literal text in order."""
        if isinstance(keys, str):
            if keys in _KNOWN_KEYS or keys.startswith("ALT_"):
                keys = [keys]
            else:
                keys = list(keys)
        for item in keys:
            item = str(item)
            if item in _KNOWN_KEYS or item.startswith("ALT_") or len(item) <= 1:
                self.handle_key(item, col=col)
            elif item.startswith("\x1b") or item.startswith("\x00") \
                    or item.startswith("\xe0"):
                raw = item
                while raw:
                    mouse = _SGR_MOUSE_RE.match(raw)
                    key, used = parse_escape_sequence(raw)
                    if not key or used <= 0:
                        break
                    self.handle_key(
                        key, col=(int(mouse.group(2)) if mouse else col))
                    raw = raw[used:]
            else:
                for char in item:
                    self.handle_key(char)
        return self.input_buf

    def type_text(self, text: str) -> str:
        """Insert literal text at the cursor (tests and scripted drivers)."""
        self._insert_text(str(text))
        return self.input_buf

    def feed_key_events(self, keys, *, col: int | None = None) -> str:
        """Alias for `feed_keys` used by deterministic key-event tests."""
        return self.feed_keys(keys, col=col)

    def _read_answer_line(self) -> str | None:
        """Permission/confirmation input: raw keystrokes when interactive."""
        if not self.interactive:
            return self._readline()
        buf = ""
        self.input_buf = ""
        self.cursor_pos = 0
        while True:
            key = self.read_key_event(0.05)
            if key is None:
                self._poll_events()
                if self._dirty:
                    self._redraw()
                continue
            if key == "ENTER":
                self.input_buf = ""
                self.cursor_pos = 0
                return buf
            if key in ("CTRL_C", "ESC"):
                self.input_buf = ""
                self.cursor_pos = 0
                return ""
            if key in ("BACKSPACE", "DELETE"):
                buf = buf[:-1] if buf else ""
            elif len(key) == 1 and key >= " ":
                buf += key
            self.input_buf = buf
            self.cursor_pos = len(buf)
            self._redraw(force=True)

    # ---------------------------------------------------------- backend lazily
    def _ensure_client(self):
        if self.client is None:
            self.client = LLMClient(self.cfg)
        try:
            if not getattr(self.cfg, "_root", ""):
                # The LLM transport uses this to route `llm_stream` events to
                # the project's EventLog (the TUI status listener reads them).
                self.cfg._root = str(self.root)
        except Exception:
            pass
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

    def _sync_pins(self) -> None:
        """Publish the session pin list to every phase of the next run."""
        try:
            self.cfg._pinned_refs = list(self.pins)
        except Exception:
            pass
        if self.agent is not None:
            self.agent.cfg = self.cfg

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
        holder: dict = {"result": None, "error": None, "telemetry": {}}
        self._start_watcher()

        def runner():
            # One run-scoped collector per task: every LLM call folds into it
            # (events.note_call) so the closing aggregate covers exactly this
            # run, never a previous one.
            events_mod.begin_run()
            try:
                holder["result"] = work()
            except BaseException as exc:          # shown, never silently lost
                holder["error"] = exc
            finally:
                holder["telemetry"] = events_mod.end_run()

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
            self._stream_status = ""
            self._stream_phase = ""
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

    def _report_run_telemetry(self, holder: dict) -> None:
        """The closing aggregate line after a run -- the same `[run: ...]`
        text the CLI prints, so both surfaces tell one tokens/speed story."""
        line = events_mod.run_line(holder.get("telemetry") or {})
        if line:
            self._append(line, "receipt")

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

    # ------------------------------------------------------------ security
    def _security_outbound_guard(self, text: str) -> bool:
        """Ask before a prompt carrying a secret leaves for a cloud provider.

        Returns False only when the user cancels. Non-interactive sessions
        choose auto-redaction, exactly like the CLI. Loopback/LAN providers
        are never inspected.
        """
        cfg = self.cfg
        try:
            cfg._outbound_acknowledge_secrets = False
            cfg._outbound_redact_secrets = False
        except Exception:
            return True
        if not llm_mod.is_external_provider(cfg):
            return True
        mode = security.outbound_mode(cfg)
        if mode == "allow":
            return True
        if mode == "redact":
            cfg._outbound_redact_secrets = True
            return True
        findings = security.unacknowledged_findings(
            security.scan_text_for_secrets(str(text or ""), "<prompt>"),
            self.root)
        if not findings and self.interactive:
            return True
        if not self.interactive:
            # A headless script must never park on a prompt. Set the run-wide
            # redact flag so secrets found in gathered context are covered
            # too, and report the decision without blocking.
            cfg._outbound_redact_secrets = True
            for finding in findings:
                self._append(
                    f"[security] {security.format_finding(finding)}", "receipt")
            if findings:
                self._append(
                    "[security] non-interactive session: auto-redacting "
                    "before sending to the external provider", "receipt")
            return True
        self._append(
            "[security] This prompt contains credentials that would be "
            "sent to an external provider:", "receipt")
        for finding in findings:
            self._append(
                f"[security] {security.format_finding(finding)}", "receipt")
        question = ("[security] [A]cknowledge & send / [R]edact & send / "
                    "[C]ancel (default: R):")
        self._set_banner(question)
        self._append(question, "activity")
        self._redraw(force=True)
        answer = self._read_answer_line()
        self.input_buf = ""
        self.cursor_pos = 0
        self._set_banner("")
        first = ((answer or "").strip().lower().split() or ["r"])[0]
        if first in ("c", "cancel", "n", "no"):
            self._append("[security] cancelled; nothing was sent", "receipt")
            self._dirty = True
            return False
        if first in ("a", "ack", "acknowledge", "y", "yes"):
            security.acknowledge_findings(
                self.root, [f.fingerprint for f in findings])
            cfg._outbound_acknowledge_secrets = True
            self._append(
                f"[security] acknowledged {len(findings)} finding(s); "
                "sending as-is", "receipt")
        else:
            cfg._outbound_redact_secrets = True
            self._append("[security] redacting secrets before sending",
                         "receipt")
        self._dirty = True
        return True

    def _security_notice(self, files=None) -> dict:
        """Print a non-blocking `[security]` notice after a run."""
        try:
            status = security.security_status(
                self.root, self.cfg, files=files, use_cache=False)
        except Exception:
            return {}
        new_keys: list[str] = []
        lines: list[str] = []
        for finding in status.get("findings", []):
            key = str(finding.get("fingerprint") or "")
            if not key or key in self._security_notified:
                continue
            self._security_notified.add(key)
            new_keys.append(key)
            lines.append(
                "[security] secret at "
                f"{finding.get('file', '?')}:{finding.get('line', '?')} "
                f"[{finding.get('kind', '?')}] "
                f"{finding.get('masked_preview', '***')}")
        for item in status.get("unignored_sensitive_files", []):
            rel = str(item.get("file") or "")
            key = "file:" + rel
            if not rel or key in self._security_notified:
                continue
            self._security_notified.add(key)
            new_keys.append(key)
            lines.append(
                f"[security] {rel} is not ignored by .gitignore")
        if not new_keys:
            return status
        self._append("[security] Security notice (unacknowledged findings):",
                     "receipt")
        for line in lines[:12]:
            self._append(line, "receipt")
        if len(lines) > 12:
            self._append(f"[security] ... and {len(lines) - 12} more",
                         "receipt")
        self._append(
            "[security] Non-blocking: nothing was changed. Run /security for "
            "details, /security ack to ignore secret findings, or "
            "/security gitignore to add .env patterns.", "receipt")
        return status

    def _security(self, arg: str) -> None:
        """`/security`, `/security ack`, `/security gitignore`."""
        sub = (arg or "").strip().lower().split(" ", 1)[0]
        status = security.security_status(
            self.root, self.cfg, use_cache=False)
        if sub in ("ack", "acknowledge", "ignore"):
            fingerprints = [f["fingerprint"] for f in
                            status.get("findings", []) if f.get("fingerprint")]
            if not fingerprints:
                self._append("[security] No unacknowledged secrets to "
                             "acknowledge.", "system")
                return
            security.acknowledge_findings(self.root, fingerprints)
            self._security_notified.clear()
            self._append(
                f"[security] Acknowledged {len(fingerprints)} finding(s); "
                "the same keys will not be reported again.", "system")
            return
        if sub in ("gitignore", "ignore-files"):
            added = security.ensure_gitignore_entries(self.root)
            if added:
                self._append(
                    "[security] Added to .gitignore: " + ", ".join(added),
                    "system")
            else:
                self._append(
                    "[security] .gitignore already covers .env patterns.",
                    "system")
            self._security_notified.clear()
            return
        if not status.get("secrets_found") and not status.get("env_unignored"):
            self._append(
                "[security] No unacknowledged secrets or unignored .env "
                "files detected.", "system")
            return
        self._append("[security] Workspace scan:", "system")
        for finding in status.get("findings", []):
            self._append(
                "[security] " + security.format_finding(finding), "receipt")
        for item in status.get("unignored_sensitive_files", []):
            self._append(
                f"[security] {item.get('file', '?')} is not ignored by "
                ".gitignore", "receipt")
        self._append(
            "[security] /security ack acknowledges secret findings; "
            "/security gitignore adds .env patterns.", "system")

    def _theme(self, arg: str) -> None:
        """`/theme` lists palettes; `/theme NAME` switches and saves."""
        name = (arg or "").strip()
        if not name:
            data = themes_mod.list_themes(self.root)
            builtin = list(data.get("builtin_names") or [])
            custom = [n for n in data.get("custom_names") or []
                      if n not in builtin]
            names = builtin + sorted(custom)
            lines = ["Color themes (shared with the GUI):"]
            for item in names:
                active = item == data.get("active_theme")
                marker = "*" if active else " "
                kind = "built-in" if item in builtin else "custom"
                line = f" {marker} {item:<20} {kind}"
                if active:
                    line += "  <- active"
                lines.append(line)
            override = data.get("project_theme")
            if override:
                lines.append(f"Project override: {override}")
            lines.append("Use /theme <name> to switch; the choice is saved.")
            self._append("\n".join(lines), "system")
            return
        try:
            tokens = themes_mod.set_active_theme(name, self.root)
        except ValueError as exc:
            self._append(str(exc), "error")
            return
        self.theme = tokens
        self._reload_theme()
        self._append(f"Theme: {self.theme_name} (saved)", "receipt")
        self._dirty = True

    def run_task(self, task: str) -> None:
        """Run one plain-text coding task, keeping the UI live while it runs."""
        self._context_override = None
        self._event_active = ""
        self._append(f"\u00bb {task}", "user")
        if not self._security_outbound_guard(task):
            self._redraw(force=True)
            return
        self._evict_servers()
        client, agent, idx = self._ensure_backend()
        cfg = self.cfg
        self._sync_pins()
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
            elif isinstance(error, LLMError) \
                    and getattr(error, "kind", "") == "cancelled":
                self._append("stopped by user (the session stays open)",
                             "system")
            elif isinstance(error, LLMError):
                self._append(f"ERROR: {error}", "error")
            else:
                self._append(f"ERROR: {type(error).__name__}: {error}",
                             "error")
            self._security_notice()
            self._report_run_telemetry(holder)
            self._redraw(force=True)
            return
        self._record_result(task, holder.get("result"))
        self._security_notice()
        self._report_run_telemetry(holder)
        self._redraw(force=True)

    def _auto_with_nudge(self, goal: str) -> None:
        """`/auto`: recommend single-pass Task mode for one-pass edits first.

        Interactive sessions get the recommendation and a [Y/n] question
        (default Yes, like the CLI nudge); headless/scripted sessions are
        never blocked on an answer -- they log a note and keep Auto.
        `/auto!` skips the check entirely."""
        goal = " ".join(str(goal or "").split())
        if not goal:
            self._append("Usage: /auto <goal>", "error")
            return
        verdict = loop.evaluate_mode_suitability(goal, self.root, "auto")
        if not verdict["mismatch"]:
            self.run_auto(goal)
            return
        self._append("\U0001f4a1 This request looks better suited for "
                     "single-pass Task mode than multi-step Auto mode:",
                     "system")
        self._append("   " + verdict["reason"], "system")
        if not self.interactive:
            self._append("note: continuing with Auto (non-interactive; "
                         "use /auto! to force, or type the task plainly "
                         "for one pass)", "system")
            self.run_auto(goal)
            return
        question = "Run as a single-pass Task instead? [Y/n]:"
        self._set_banner(question)
        self._append(question, "activity")
        self._redraw(force=True)
        answer = self._read_answer_line()
        self.input_buf = ""
        self.cursor_pos = 0
        self._set_banner("")
        if (answer or "").strip().lower() in ("", "y", "yes"):
            self._append("running as a single-pass Task", "system")
            self.run_task(goal)
        else:
            self.run_auto(goal)

    def run_auto(self, goal: str) -> None:
        """`/auto`: generate or resume PLAN.md, then run the outer loop."""
        goal = " ".join(str(goal or "").split())
        if not goal:
            self._append("Usage: /auto <goal>", "error")
            return
        self._evict_servers()
        client, agent, idx = self._ensure_backend()
        cfg = self.cfg
        self._sync_pins()
        plan_path = self._plan_path()
        self._append(f"\u00bb /auto {goal}", "user")
        if not self._security_outbound_guard(goal):
            self._redraw(force=True)
            return
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
            cancelled = (isinstance(error, LLMError)
                         and getattr(error, "kind", "") == "cancelled")
            if not isinstance(error, KeyboardInterrupt) and not cancelled:
                self._append(f"ERROR: {error}", "error")
            elif cancelled:
                self._append("stopped by user (the session stays open)",
                             "system")
        else:
            result = holder.get("result")
            if isinstance(result, int):
                self._append(f"auto finished with exit code {result}",
                             "system")
            self._append("auto loop finished", "system")
        self._security_notice()
        self._report_run_telemetry(holder)
        self._sync_telemetry()
        self._redraw(force=True)

    def run_new(self, arg: str) -> None:
        """`/new DIR PROMPT`: scaffold a brand-new project, switching this
        session's active root into DIR BEFORE the first LLM call -- the same
        lifecycle as the GUI's New and `pdc new`. Complex scaffolding is
        auto-detected from the prompt (never a sticky toggle)."""
        from . import scaffold

        parts = str(arg or "").strip().split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            self._append("Usage: /new <dir> <prompt>", "error")
            self._redraw(force=True)
            return
        target = Path(parts[0]).expanduser()
        if not target.is_absolute():
            target = self.root / target
        target = target.resolve()
        prompt = " ".join(parts[1].split())
        if not self._security_outbound_guard(prompt):
            self._redraw(force=True)
            return
        if target.is_dir():
            entries = sorted(p.name for p in target.iterdir()
                             if not scaffold.is_harness_owned(p.name))
            if entries:
                shown = ", ".join(entries[:8])
                more = f" (+{len(entries) - 8} more)" if len(entries) > 8 else ""
                self._append(f"ERROR: {target} is not empty ({shown}{more}). "
                             "New projects start in a fresh directory.",
                             "error")
                self._redraw(force=True)
                return
        target.mkdir(parents=True, exist_ok=True)
        # Switch the workspace FIRST: rebind the event log so every
        # llm_start/call of the build lands in the new project's session.
        try:
            if self._listener in self.elog.listeners:
                self.elog.listeners.remove(self._listener)
        except Exception:
            pass
        self.root = target
        self.elog = events_mod.get(target, self.cfg)
        try:
            self.elog.listeners.append(self._listener)
        except Exception:
            pass
        self._event_path = None
        self._event_offset = 0
        self._recent_events = {}
        self._init_event_tail()
        if self.client is not None:
            try:
                events_mod.attach(self.client, self.elog)
            except Exception:
                pass
        # Fresh session state: an empty index right now (0 files is valid),
        # rebuilt from disk once the scaffold has landed its files.
        self.agent = None
        self.modified = []
        self._security_notified.clear()
        self.resume_context = ""
        try:
            workspace.ensure(target, self.cfg.manage_gitignore,
                             self.cfg.create_ignore_file)
            self.idx = index_mod.build([str(target)], self.cfg, client=None,
                                       root=target, summarise=False,
                                       verbose=False)
        except (SystemExit, Exception):
            self.idx = index_mod.Index(target, [])
        self._append(f"switched workspace to {target}", "system")
        client = self._ensure_client()
        cfg = self.cfg
        self._append(f"running new on {cfg.model} ...", "system")
        self._refresh_warmup(client)
        spec = scaffold.load_spec(prompt=prompt,
                                  max_chars=cfg.max_spec_chars)
        verify = cfg.auto_verify
        if verify and loop._task_named_tests(verify):
            cfg._tests_user_consent = True

        def work():
            workspace.log_event(target, "command: tui new")
            return scaffold.run_new(
                target, cfg, client, spec, apply=True, verbose=True,
                complex=False, verify=verify,
                repair_rounds=cfg.auto_repair_rounds,
                max_tasks=cfg.auto_max_tasks,
                max_minutes=cfg.auto_max_minutes,
                auto_install=cfg.auto_install_deps)

        holder = self._run_in_worker(work)
        error = holder.get("error")
        if error is not None:
            cancelled = (isinstance(error, LLMError)
                         and getattr(error, "kind", "") == "cancelled")
            if not isinstance(error, KeyboardInterrupt) and not cancelled:
                self._append(f"ERROR: {error}", "error")
            elif cancelled:
                self._append("stopped by user (the session stays open)",
                             "system")
        else:
            code = holder.get("result")
            if code == 0:
                self._append(f"Project created at {target}.", "system")
            else:
                self._append(f"new finished with exit code {code}", "error")
        # Bring the session's index in line with what the scaffold wrote so
        # the next plain-text task sees every new file (the same refresh the
        # GUI runs after a New build).
        try:
            from types import SimpleNamespace
            from . import cli as cli_mod
            base = index_mod.Index.load(target)
            if base is None:
                base = self.idx if self.idx is not None \
                    else index_mod.Index(target, [])
            self.idx = cli_mod._refresh(
                base, SimpleNamespace(paths=[str(target)]), cfg, target,
                verbose=False)
        except (SystemExit, Exception):
            pass
        self.agent = None
        self._security_notice()
        self._report_run_telemetry(holder)
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
        if cmd == "keys":
            self._append(KEYS_TEXT, "system")
            return
        if cmd == "scroll":
            self._scroll_command(arg)
            return
        if cmd == "mouse":
            self._mouse_command(arg)
            return
        if cmd in ("status", "telemetry"):
            self._append(self.status_report(), "system")
            return
        if cmd == "security":
            self._security(arg)
            return
        if cmd == "theme":
            self._theme(arg)
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
        if cmd == "new":
            self.run_new(arg)
            return
        if cmd == "auto":
            self._auto_with_nudge(arg)
            return
        if cmd == "auto!":
            self.run_auto(arg)
            return
        if cmd == "plan":
            self.show_plan(arg)
            return
        if cmd == "pin":
            self._pin(arg)
            return
        if cmd == "rules":
            self._rules(arg)
            return
        if cmd in ("consolidate", "compact"):
            self._consolidate()
            return
        if cmd in ("profile", "profiles"):
            if cmd == "profile" and arg.strip().lower() in (
                    "show", "run", "reset"):
                self._profiler_command(arg)
                return
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
        if cmd == "effort":
            self._effort(arg)
            return
        if cmd == "hybrid":
            self._hybrid(arg)
            return
        if cmd == "arena":
            self._arena(arg)
            return
        if cmd == "servers":
            self._servers(arg)
            return
        if cmd == "unload":
            self._unload(arg)
            return
        if cmd == "creative":
            self._creative(arg)
            return
        if cmd == "apply":
            self._apply_toggle(arg)
            return
        if cmd == "verify":
            self._verify(arg)
            return
        if cmd == "skills":
            self._skills(arg)
            return
        if cmd == "skill":
            self._skill(arg)
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
        if cmd == "checkpoints":
            self._checkpoints(arg)
            return
        if cmd == "checkout":
            self._checkout(arg)
            return
        if cmd == "undo":
            self._undo()
            return
        if cmd == "run":
            self._run_command(arg)
            return
        if cmd == "fix":
            self._fix_last_crash()
            return
        if cmd == "update":
            self._update(arg)
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
            self.scroll_offset = 0
            self._scroll_seen_total = 0
            self._scroll_seen_width = 0
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
            f"  creative: {'ON' if getattr(self.cfg, 'creative_mode', False) else ('auto' if getattr(self.cfg, 'creative_auto', True) else 'off')}",
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
            f"  class   : {self._model_class()}"
            f", effort {getattr(self.cfg, 'reasoning_effort', 'default') or 'default'}"
            f", streams {getattr(self.cfg, 'max_concurrency', 0) or 'auto'}",
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

    def _model_class(self) -> str:
        try:
            from . import config as config_mod
            return config_mod.capability_profile(self.cfg)
        except Exception:
            return "?"

    def _consolidate(self) -> None:
        messages = getattr(self.agent, "_messages", None) if self.agent else None
        try:
            result = memory_mod.consolidate_session(
                self.root, self.cfg, messages)
        except OSError as exc:
            self._append(f"consolidate failed: {exc}", "error")
            return
        self.resume_context = memory_mod.handoff_prompt(result["summary"],
                                                        self.root)
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

    def _scroll_command(self, arg: str) -> None:
        parts = arg.split()
        action = parts[0].lower() if parts else ""
        try:
            amount = int(parts[1]) if len(parts) > 1 else None
        except ValueError:
            amount = None
        height = self._size()[1]
        half = max(1, self._transcript_viewport_height(height) // 2)
        side_half = max(1, self._body_height(height) // 2)
        if action in ("", "status"):
            self._append(
                f"scroll: {self.scroll_offset} line(s) back; "
                f"sidebar {self.sidebar_scroll_offset} line(s); "
                f"max {self.max_scroll()}", "system")
            return
        target = self.scroll_offset
        if action in ("up", "back"):
            target = self.scroll_transcript(
                amount if amount is not None else half)
        elif action in ("down", "forward"):
            target = self.scroll_transcript(
                -(amount if amount is not None else half))
        elif action == "top":
            target = self.scroll_to_top()
        elif action in ("bottom", "live"):
            target = self.scroll_to_bottom()
            self.sidebar_scroll_offset = 0
        elif action == "sidebar-up":
            self.scroll_sidebar(amount if amount is not None else side_half)
        elif action == "sidebar-down":
            self.scroll_sidebar(-(amount if amount is not None else side_half))
        else:
            try:
                target = max(0, min(self.max_scroll(), int(action)))
                self.scroll_offset = target
            except ValueError:
                self._append("Usage: /scroll up|down|top|bottom|N", "error")
                return
        self._append(f"scroll: {target} line(s) back "
                     f"(live bottom = 0)", "system")
        # The receipt itself must not perturb the requested scripted position.
        if action == "top":
            self.scroll_offset = self.max_scroll()
        elif action in ("bottom", "live"):
            self.scroll_offset = 0
        elif action not in ("sidebar-up", "sidebar-down"):
            self.scroll_offset = max(0, min(self.max_scroll(), int(target)))

    def _mouse_command(self, arg: str) -> None:
        value = arg.strip().lower()
        if value in ("", "toggle"):
            self._toggle_mouse()
        elif value in ("on", "true", "yes", "1", "sgr"):
            self._toggle_mouse(True)
        elif value in ("off", "false", "no", "0", "native"):
            self._toggle_mouse(False)
        else:
            self._append("Usage: /mouse [on|off]", "error")

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

    def _effort(self, arg: str) -> None:
        """`/effort [default|low|medium|high]`: reasoning effort forwarded to
        local-MoE runtimes and reasoning-capable hosted APIs."""
        value = arg.strip().lower()
        if not value:
            current = str(getattr(self.cfg, "reasoning_effort", "default")
                          or "default")
            self._append(
                f"Reasoning effort: {current}\n"
                "Usage: /effort default|low|medium|high", "system")
            return
        if value in ("auto", "none", "off"):
            value = "default"
        if value not in ("default", "low", "medium", "high"):
            self._append("Usage: /effort default|low|medium|high", "error")
            return
        self.cfg.reasoning_effort = value
        try:
            self.cfg.sources["reasoning_effort"] = "tui /effort"
        except Exception:
            pass
        klass = ""
        try:
            from . import config as config_mod
            klass = config_mod.capability_profile(self.cfg)
        except Exception:
            pass
        self._append(
            f"Reasoning effort: {value}"
            + (f" ({klass})" if klass else "")
            + (" \u2014 no field is sent at default"
               if value == "default" else " \u2014 sent as reasoning_effort"),
            "system")

    def _hybrid(self, arg: str) -> None:
        """`/hybrid [off | <architect_provider> <architect_model>]`: arm the
        split-stage architect/editor pipeline for this session. A single
        token names a profile (its model comes from the profile body) or a
        bare model id on the active endpoint."""
        value = arg.strip()
        if not value:
            on = bool(getattr(self.cfg, "hybrid_routing", False))
            provider = str(getattr(self.cfg, "architect_provider", "") or "")
            model = str(getattr(self.cfg, "architect_model", "") or "")
            detail = (f" — architect {provider + '/' if provider else ''}"
                      f"{model}" if model else "")
            self._append(
                f"Hybrid routing: {'ON' if on else 'OFF'}{detail}\n"
                "Usage: /hybrid off | /hybrid <architect_provider> "
                "<architect_model>", "system")
            return
        if value.lower() in ("off", "false", "no", "0"):
            self.cfg.hybrid_routing = False
            try:
                self.cfg.sources["hybrid_routing"] = "tui /hybrid"
            except Exception:
                pass
            self._append("Hybrid routing: OFF", "system")
            return
        parts = value.split()
        if len(parts) == 1:
            name = parts[0]
            body = (self.cfg.profiles or {}).get(name)
            if body is not None:
                provider = name
                model = str((body or {}).get("model") or "").strip()
                if not model:
                    self._append(f"Profile '{name}' names no model.", "error")
                    return
            else:
                provider, model = "", name
        else:
            provider, model = parts[0], parts[1]
        if model == str(getattr(self.cfg, "model", "") or ""):
            self._append("The architect model equals the editor model; "
                         "hybrid would pay twice for one brain. Pick a "
                         "different architect.", "error")
            return
        self.cfg.hybrid_routing = True
        self.cfg.architect_provider = provider
        self.cfg.architect_model = model
        for key in ("hybrid_routing", "architect_provider", "architect_model"):
            try:
                self.cfg.sources[key] = "tui /hybrid"
            except Exception:
                pass
        self._append(
            f"Hybrid routing: ON — Stage 1 architect "
            f"{provider + '/' if provider else ''}{model}, Stage 2 editor "
            f"{self.cfg.model}", "system")

    def _arena_scorecard(self, state: dict) -> str:
        state = state or {}
        lines = [f"Arena {state.get('arena_id', '?')} — "
                 f"{str(state.get('prompt') or '')[:100]}"]
        for side in ("a", "b"):
            rec = (state.get("contestants") or {}).get(side) or {}
            mark = "\u2713" if rec.get("ok") else "\u2717"
            provider = str(rec.get("provider") or "")
            lines.append(
                f"  {side.upper()} {mark} "
                f"{provider + '/' if provider else ''}{rec.get('model', '?')}"
                f" — {len(rec.get('files_changed') or [])} file(s), "
                f"{rec.get('elapsed_sec', 0)}s, "
                f"{rec.get('tok_per_sec', 0)} tok/s, "
                f"{int(rec.get('total_tokens', 0) or 0):,} tokens")
            for rel in (rec.get("files_changed") or [])[:8]:
                lines.append(f"      {rel}")
            summary = str(rec.get("summary") or "").strip()
            if summary:
                lines.append("      " + summary.splitlines()[0][:120])
        if state.get("winner"):
            lines.append(f"  winner applied: {str(state['winner']).upper()}")
        return "\n".join(lines)

    def _arena(self, arg: str) -> None:
        """`/arena <provider_a>:<model_a> <provider_b>:<model_b> <prompt>`
        runs both contestants in isolated sandboxes; `/arena apply a|b`
        adopts a winner; bare `/arena` shows the last scorecard."""
        from . import arena as arena_mod
        value = arg.strip()
        if not value:
            state = arena_mod.last_arena(self.root)
            if state is None:
                self._append(
                    "Usage: /arena <provider_a>:<model_a> <provider_b>:<model_b>"
                    " <prompt>\n       /arena apply a|b", "system")
                return
            self._append(self._arena_scorecard(state), "system")
            return
        parts = value.split(None, 1)
        if parts[0].lower() == "apply":
            self._arena_apply(parts[1].strip() if len(parts) > 1 else "")
            return
        tokens = value.split(None, 2)
        if len(tokens) < 3 or not tokens[2].strip():
            self._append(
                "Usage: /arena <provider_a>:<model_a> <provider_b>:<model_b>"
                " <prompt>", "error")
            return
        spec_a, spec_b, prompt = tokens[0], tokens[1], tokens[2].strip()
        self._evict_servers()
        self._append(f"arena: {spec_a} vs {spec_b} — both run in isolated "
                     "sandboxes; the workspace stays untouched ...", "system")

        def work():
            return arena_mod.run_arena_comparison(self.root, prompt,
                                                  spec_a, spec_b, self.cfg)

        holder = self._run_in_worker(work)
        error = holder.get("error")
        if error is not None:
            if isinstance(error, KeyboardInterrupt):
                pass
            elif isinstance(error, arena_mod.ArenaError):
                self._append(f"arena: {error}", "error")
            else:
                self._append(f"ERROR: {type(error).__name__}: {error}",
                             "error")
            self._report_run_telemetry(holder)
            self._redraw(force=True)
            return
        state = holder.get("result") or {}
        self._last_arena_id = str(state.get("arena_id") or "")
        self._append(self._arena_scorecard(state), "system")
        self._append("Apply a winner with /arena apply a  or  /arena apply b",
                     "system")
        self._report_run_telemetry(holder)
        self._redraw(force=True)

    def _arena_apply(self, arg: str) -> None:
        from . import arena as arena_mod
        winner = arg.strip().lower()
        if winner not in ("a", "b"):
            self._append("Usage: /arena apply a|b", "error")
            return
        arena_id = str(getattr(self, "_last_arena_id", "") or "")
        if not arena_id:
            state = arena_mod.last_arena(self.root)
            arena_id = str((state or {}).get("arena_id") or "")
        if not arena_id:
            self._append("No arena to apply — run /arena first.", "error")
            return
        try:
            out = arena_mod.apply_arena_winner(self.root, arena_id, winner)
        except (arena_mod.ArenaError, checkpoints_mod.CheckpointError,
                OSError) as exc:
            self._append(f"arena apply: {exc}", "error")
            return
        # The workspace changed under the index: rebuild it before the next
        # task gathers (same fresh static index a newly opened project gets).
        try:
            self.idx = index_mod.build([str(self.root)], self.cfg, client=None,
                                       root=self.root, summarise=False,
                                       verbose=False)
            self.agent = Agent(self.idx, self._ensure_client(), self.cfg)
        except Exception:
            pass
        self._append(
            f"Arena winner applied: {out['model']} — "
            f"{len(out['files_changed'])} file(s) changed.\n"
            f"Checkpoint {out['post_checkpoint']} (undo with /undo).",
            "receipt")
        self._redraw(force=True)

    def _profiler_command(self, arg: str) -> None:
        """`/profile show|run|reset`: the CLI's profiler actions in the TUI."""
        from . import profiler
        action = (arg or "show").strip().lower()
        probe = None
        if action == "reset":
            path = profiler.reset()
            changed = ""
            try:
                from . import config as config_mod
                from . import config_editor
                config_path = Path(
                    getattr(self.cfg, "_config_path", "")
                    or config_mod.USER_CONFIG)
                changed = config_editor.disable_auto_tune_opt_in(
                    config_path,
                    str(getattr(self.cfg, "active_profile", "") or ""))
            except Exception:
                changed = ""
            try:
                self.cfg.auto_tune = False
                self.cfg.sources["auto_tune"] = "user (TUI /profile reset)"
            except Exception:
                pass
            message = f"Dropped cached profiling measurements ({path}). "
            if changed:
                message += f"Set auto_tune = false in {changed}. "
            else:
                message += ("No explicit auto_tune = true was found in the "
                            "selected config file. ")
            message += "auto_tune is off for this session."
            self._append(message, "system")
            return
        if action == "run":
            if not servers_mod.is_local_url(self.cfg.base_url):
                self._append(
                    f"{profiler.HOSTED_PROBE_REFUSAL}: {self.cfg.model} on "
                    f"{self.cfg.base_url} (speed probes are local-only).",
                    "system")
                return
            self._append(
                f"Probing {self.cfg.model} on {self.cfg.base_url} "
                "(one prefill, one tiny completion)...", "system")
            probe = profiler.probe_model_speed(
                self.cfg.base_url, self.cfg.model, self.cfg, force=True)
            if not probe.get("ok"):
                self._append("Probe failed: "
                             + str(probe.get("detail") or
                                   "the server did not answer"), "error")
                return
        report = profiler.build_report(self.cfg, probe=probe,
                                       fetch_metadata=True)
        self._append(profiler.format_profile_report(report), "system")

    def _servers(self, arg: str = "") -> None:
        """`/servers`: live VRAM/RAM plus each configured local server.
        `/servers pin|unpin SERVER [MODEL]` toggles Lemonade's LRU exemption
        for a loaded model."""
        parts = arg.strip().split(None, 2)
        if parts and parts[0].lower() in ("pin", "unpin"):
            if len(parts) < 2:
                self._append(
                    "usage: /servers pin SERVER [MODEL]  |  "
                    "/servers unpin SERVER [MODEL]", "error")
                return
            action = parts[0].lower()
            target = parts[1]
            model = parts[2].strip() if len(parts) > 2 else ""
            try:
                result = servers_mod.pin_server_model(
                    target, model, action == "pin", self.cfg)
            except Exception as exc:
                self._append(f"{action} failed: {exc}", "error")
                return
            name = result.get("server") or target
            chosen = str(result.get("model") or "").strip()
            detail = str(result.get("detail") or "").strip()
            label = ("SKIPPED" if result.get("skipped")
                     else "OK" if result.get("ok") else "FAILED")
            self._append(
                f"{label} {action} {name}"
                + (f" \u00b7 {chosen}" if chosen else "")
                + (f" \u2014 {detail}" if detail else ""),
                "system" if result.get("ok") else "error")
            return
        try:
            payload = servers_mod.server_summary(self.cfg)
        except Exception as exc:
            self._append(f"could not read local servers: {exc}", "error")
            return
        self._append(
            servers_mod.format_server_states(
                payload.get("servers") or [], payload.get("memory")),
            "system")

    def _unload(self, arg: str) -> None:
        """`/unload [server]`: unload one server, or every idle local server."""
        target = arg.strip()
        try:
            if target:
                result = servers_mod.unload_server_model(
                    target, "", self.cfg)
                results = [result]
            else:
                results = servers_mod.unload_all_local_servers(self.cfg)
        except Exception as exc:
            self._append(f"unload failed: {exc}", "error")
            return
        if not results:
            self._append("No resident model on any configured local server.",
                         "system")
            return
        lines = ["Unload results"]
        for result in results:
            ok = bool(result.get("ok"))
            name = result.get("server") or result.get("base_url") or "?"
            detail = str(result.get("detail") or "").strip()
            model = str(result.get("model") or "").strip()
            lines.append(
                f"  {'OK' if ok else 'FAILED'} {name}"
                + (f" \u00b7 {model}" if model else "")
                + (f" \u2014 {detail}" if detail else ""))
        try:
            mem = servers_mod.get_hardware_memory()
            lines.append(f"  {servers_mod.memory_badge(mem)}")
        except Exception:
            pass
        self._append("\n".join(lines),
                     "system" if all(r.get("ok") for r in results)
                     else "error")

    def _evict_servers(self) -> None:
        """Auto-evict handshake before a run starts on this profile's server."""
        try:
            result = servers_mod.evict_other_local_servers(
                getattr(self.cfg, "active_profile", "") or self.cfg.base_url,
                self.cfg)
        except Exception:
            return
        evicted = [r for r in result.get("evicted", []) if r]
        if evicted:
            names = ", ".join(str(r.get("server") or r.get("base_url") or "?")
                              for r in evicted)
            self._append(
                f"Unloaded {names} from another local server so this run can "
                "use the VRAM/RAM.", "system")

    def _creative(self, arg: str) -> None:
        """Tri-state creative/design mode: on forces the Builder persona for
        every task, off keeps surgical edits everywhere, auto detects design
        tasks. No argument cycles auto -> on -> off."""
        value = arg.strip().lower()
        if value in ("on", "true", "yes", "1"):
            self.cfg.creative_mode = True
        elif value in ("off", "false", "no", "0"):
            self.cfg.creative_mode = False
            self.cfg.creative_auto = False
        elif value in ("auto", "detect"):
            self.cfg.creative_mode = False
            self.cfg.creative_auto = True
        elif value in ("", "toggle"):
            if self.cfg.creative_mode:
                self.cfg.creative_mode = False
                self.cfg.creative_auto = False
            elif getattr(self.cfg, "creative_auto", True):
                self.cfg.creative_mode = True
            else:
                self.cfg.creative_auto = True
        else:
            self._append("Usage: /creative on|off|auto", "error")
            return
        try:
            self.cfg.sources["creative_mode"] = "tui toggle"
            self.cfg.sources["creative_auto"] = "tui toggle"
        except Exception:
            pass
        if self.cfg.creative_mode:
            state = "ON (every task)"
        elif getattr(self.cfg, "creative_auto", True):
            state = "AUTO (detect design tasks)"
        else:
            state = "OFF (surgical edits)"
        self._append(f"Creative mode: {state}", "system")

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

    def _verify(self, arg: str) -> None:
        """`/verify [on|off|status]`: control the post-edit verification
        command without touching the always-on cheap gates. `on` fills an
        empty `auto_verify` from the project's detected test command."""
        value = arg.strip().lower()
        cfg = self.cfg
        if value in ("", "status", "show"):
            cmd = str(getattr(cfg, "auto_verify", "") or "").strip()
            static = bool(getattr(cfg, "static_gate", True))
            static_s = float(getattr(cfg, "static_gate_timeout", 3.0) or 0.0)
            repair = int(getattr(cfg, "auto_repair_rounds", 2) or 0)
            rounds = int(getattr(cfg, "auto_verify_rounds", 2) or 0)
            final = bool(getattr(cfg, "auto_verify_final", False))
            ratchet = bool(getattr(cfg, "auto_green_ratchet", True))
            lock = bool(getattr(cfg, "interface_lock", True))
            timeout = int(getattr(cfg, "test_timeout", 300) or 300)
            self._append("\n".join([
                "Post-edit verify (auto_verify): "
                + (f"ON \u2014 {cmd}" if cmd
                   else "OFF (auto_verify is empty)"),
                f"Static gate (static_gate): {'ON' if static else 'OFF'} "
                f"(ruff or AST fallback, {static_s:g} s, new errors only)",
                f"Repair rounds (auto_repair_rounds): {repair}; runtime "
                f"verification rounds (auto_verify_rounds): {rounds}"
                + ("; final-only (auto_verify_final)" if final else ""),
                "Scoped tests: import-graph impacted tests only "
                f"(testsel.run_impacted), bounded by test_timeout "
                f"({timeout} s); the full suite is the verify command",
                f"Green ratchet (auto_green_ratchet): "
                f"{'ON' if ratchet else 'OFF'} (rolls back a failed Auto "
                f"step); interface lock (interface_lock): "
                f"{'ON' if lock else 'OFF'}",
            ]), "system")
            return
        if value in ("on", "true", "yes", "1"):
            cmd = str(getattr(cfg, "auto_verify", "") or "").strip()
            if cmd:
                self._append(f"Post-edit verify: ON \u2014 {cmd}", "system")
                return
            detected = ""
            try:
                detected = str(testsel.detect_test_command(
                    self.root, cfg) or "").strip()
            except Exception:
                detected = ""
            if detected:
                cfg.auto_verify = detected
                try:
                    cfg.sources["auto_verify"] = "tui /verify"
                except Exception:
                    pass
                self._append(
                    f"Post-edit verify: ON \u2014 detected {detected}",
                    "system")
            else:
                cfg.auto_verify = ""
                try:
                    cfg.sources["auto_verify"] = "tui /verify"
                except Exception:
                    pass
                self._append(
                    "Post-edit verify: no test command detected; auto_verify "
                    "stays empty.\nThe syntax/AST, static and "
                    "interface-contract gates and the scoped impacted tests "
                    "still run on every edit. Set test_command or auto_verify "
                    "in config.toml to add a full-suite check.", "system")
            return
        if value in ("off", "false", "no", "0"):
            cfg.auto_verify = ""
            try:
                cfg.sources["auto_verify"] = "tui /verify"
            except Exception:
                pass
            self._append(
                "Post-edit verify: OFF \u2014 auto_verify cleared. The "
                "syntax/AST, static and interface-contract gates and the "
                "scoped impacted tests still run on every edit.", "system")
            return
        self._append("Usage: /verify on|off|status", "error")

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

    def _skill(self, arg: str) -> None:
        """`/skill NAME` -- load a skill body into the next task's pinned
        context. The body is read through `skills.load`, validated now, and
        stored as a `skill:NAME` pin so every later task (and `/pin list`)
        sees exactly the same pinned-context mechanism as @file/@symbol."""
        name = arg.strip().lower()
        if not name:
            self._append(
                "Usage: /skill <name> \u2014 loads the full skill checklist "
                "into the next task's pinned context. `/skills` lists names.",
                "error")
            return
        text, info = skills_mod.load(self.root, self.cfg, name)
        if info is None:
            self._append(text, "error")
            return
        ref = f"skill:{name}"
        if ref not in self.pins:
            self.pins.append(ref)
            self._sync_pins()
        source = info.get("source") or "unknown"
        body = (text.split("\n", 1)[1] if "\n" in text else text).strip()
        self._append(
            f"Skill '{name}' pinned for the next task ({source}, "
            f"{len(body)} chars). It rides in the pinned-context block above "
            f"the map; `/pin remove {ref}` or `/pin clear` unpins it.",
            "receipt")

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

    def _pin(self, arg: str) -> None:
        """`/pin [list|add REF|remove REF|clear]` -- explicit session pins."""
        words = arg.split(None, 1)
        action = words[0].lower() if words else "list"
        rest = words[1].strip() if len(words) > 1 else ""
        if action in ("list", "show", ""):
            if not self.pins:
                self._append(
                    "No session pins. Use `/pin add <file|symbol>` "
                    "(or type @file/@Class.method in a prompt).", "system")
                return
            self._append(f"Pinned context ({len(self.pins)}):", "system")
            for ref in self.pins:
                self._append(f"  {ref}", "system")
            return
        if action == "clear":
            self.pins = []
            self._sync_pins()
            self._append("Session pins cleared.", "system")
            return
        if action in ("add", "remove"):
            if not rest:
                self._append(f"Usage: /pin {action} <file|symbol>", "error")
                return
            if action == "remove":
                wanted = rest.lstrip("@").strip()
                hit = [p for p in self.pins
                       if p.lower() == wanted.lower()]
                if not hit:
                    hit = [p for p in self.pins
                           if wanted.lower() in p.lower()]
                if not hit:
                    self._append(f"Not pinned: {wanted}", "error")
                    return
                for ref in hit:
                    self.pins.remove(ref)
                self._sync_pins()
                self._append(f"Unpinned {', '.join(hit)}", "system")
                return
            target = self.idx
            if target is None:
                target = self.root
            try:
                pins = select_mod.resolve_pinned_references(rest, target)
            except Exception as exc:
                self._append(f"pin: {exc}", "error")
                return
            if not pins:
                self._append(
                    f"Could not resolve '{rest}' in this workspace. "
                    "Try an exact path or indexed symbol name.", "error")
                return
            added = []
            for pin in pins:
                ref = (pin.symbol_name if pin.kind == "symbol" else pin.path)
                if ref and ref not in self.pins:
                    self.pins.append(ref)
                    added.append(ref)
            self._sync_pins()
            if added:
                self._append("Pinned " + ", ".join(added) + " (session)",
                             "system")
            else:
                self._append("Already pinned.", "system")
            return
        self._append("Usage: /pin [list|add REF|remove REF|clear]", "error")

    def _rules(self, arg: str) -> None:
        """`/rules [list|set TEXT|clear]` -- `.pedacito/rules.md`."""
        words = arg.split(None, 1)
        action = words[0].lower() if words else "list"
        rest = words[1].strip() if len(words) > 1 else ""
        path = prompts_mod.project_rules_path(self.root)
        rel = prompts_mod.RULES_REL
        if action in ("list", "show"):
            text = prompts_mod.load_project_rules(self.root)
            if text:
                self._append(f"Project rules ({rel}):", "system")
                self._append(text, "system")
            else:
                self._append(
                    f"{rel} is empty or absent. Use `/rules set <text>` "
                    "to add persistent project instructions.", "system")
            return
        if action == "set":
            if not rest:
                self._append("Usage: /rules set <text>", "error")
                return
            text = rest
            if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
                text = text[1:-1]
            try:
                prompts_mod.save_project_rules(self.root, text)
            except (OSError, ValueError) as exc:
                self._append(f"rules: {exc}", "error")
                return
            self._append(f"Project rules saved to {rel}.", "system")
            return
        if action == "clear":
            try:
                prompts_mod.save_project_rules(self.root, "")
            except (OSError, ValueError) as exc:
                self._append(f"rules: {exc}", "error")
                return
            self._append(f"Project rules cleared ({rel}).", "system")
            return
        if action == "path" and path is not None:
            self._append(str(path), "system")
            return
        self._append("Usage: /rules [list|set TEXT|clear]", "error")

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

    def _checkpoints(self, arg: str = "") -> None:
        limit = 0
        text = str(arg or "").strip()
        if text:
            try:
                limit = max(1, int(text))
            except ValueError:
                limit = 0
        try:
            self._append(checkpoints_mod.render_list(self.root, limit=limit),
                         "system")
        except OSError as exc:
            self._append(f"checkpoints: {exc}", "error")

    def _checkout(self, arg: str) -> None:
        checkpoint_id = str(arg or "").strip()
        if not checkpoint_id:
            self._append("Usage: /checkout <id> (see /checkpoints)",
                         "error")
            return
        try:
            resolved = checkpoints_mod.resolve_id(self.root, checkpoint_id)
            result = checkpoints_mod.restore_checkpoint(self.root, resolved)
        except (checkpoints_mod.CheckpointError, OSError) as exc:
            self._append(f"checkout: {exc}", "error")
            return
        self._append(f"checkout: restored {result['files']} file(s) from "
                     f"{resolved}; head is now {resolved}", "system")
        for rel in result["restored"]:
            self._append(f"  restored {rel}", "system")
        for rel in result["deleted"]:
            self._append(f"  deleted  {rel}", "system")
        if result.get("skipped"):
            self._append(f"  skipped {len(result['skipped'])} unsafe path(s)",
                         "system")
        if result.get("safety_id"):
            self._append(f"  safety snapshot: {result['safety_id']}", "system")

    def _undo(self) -> None:
        result = checkpoints_mod.undo_last_run(self.root)
        if result.get("ok"):
            self._append(f"undo: back to {result['undone_to']} "
                         f"({result['files']} file(s) restored)", "system")
            for rel in result["restored"]:
                self._append(f"  restored {rel}", "system")
            for rel in result["deleted"]:
                self._append(f"  deleted  {rel}", "system")
            if result.get("safety_id"):
                self._append(f"  safety snapshot: {result['safety_id']}",
                             "system")
            return
        # Projects that predate the checkpoint tree still have backup
        # snapshots; keep the original one-step undo working there.
        try:
            snapshots = workspace.list_backups(self.root)
        except OSError as exc:
            self._append(f"could not list backups: {exc}", "error")
            return
        if not snapshots:
            self._append("No checkpoints or backups to restore.", "system")
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

    def _run_command(self, arg: str) -> None:
        cmd = str(arg or "").strip() or None
        if cmd is None:
            targets = runner_mod.detect_runnable_targets(self.root)
            if not targets["commands"]:
                self._append("No runnable entrypoint detected. Use "
                             "/run <command> to run one explicitly.", "system")
                return
            cmd = targets["commands"][0]
        shown = (cmd if isinstance(cmd, str)
                 else str((cmd or {}).get("command") or ""))
        self._append(f"\u00bb /run {shown}".rstrip(), "user")

        def work():
            return runner_mod.run_project_command(self.root, cmd)

        holder = self._run_in_worker(work)
        error = holder.get("error")
        if error is not None:
            text = f"{type(error).__name__}: {error}"
            self.last_traceback = text
            self.last_run_failure = {
                "ok": False, "command": shown, "exit_code": None,
                "stdout": "", "stderr": text, "traceback_summary": "",
            }
            self._append(f"run failed: {text}", "error")
            return
        result = holder.get("result") or {}
        for line in str(result.get("stdout") or "").splitlines():
            self._append(line, "system")
        for line in str(result.get("stderr") or "").splitlines():
            self._append(line, "error")
        tb = str(result.get("traceback_summary") or "").strip()
        if tb and tb not in str(result.get("stderr") or ""):
            for line in tb.splitlines():
                self._append(line, "error")
        if result.get("ok"):
            self.last_run_failure = None
            self.last_traceback = ""
            self._append(f"run: ok (exit 0, {result.get('duration_ms', 0)} ms)"
                         f" \u00b7 {result.get('command', '')}", "system")
            return
        self.last_run_failure = dict(result)
        self.last_traceback = (str(result.get("traceback_summary") or "").strip()
                               or str(result.get("stderr") or "").strip())
        self._append(
            f"run: FAILED exit {result.get('exit_code')} "
            f"\u00b7 {result.get('command', '')} \u2014 type /fix to send the "
            "failure into a repair task", "error")

    def _fix_last_crash(self) -> None:
        failure = self.last_run_failure
        if not failure:
            self._append("No failed /run to fix. Run something first.",
                         "system")
            return
        prompt = runner_mod.crash_prompt(failure)
        self._append("Sending the last crash into a repair task\u2026",
                     "system")
        self.run_task(prompt)

    def _update(self, arg: str) -> None:
        """`/update [run]`: installed vs latest PyPI release, the detected
        install method, and the matching upgrade. `run` skips the y/N
        confirmation; editable installs get git instructions, no command."""
        run_now = arg.strip().lower() in ("run", "--run", "-y")
        info = update_mod.check_for_update(self.cfg, manual=True)
        if info["detail"] == "disabled":
            self._append("Update checks are disabled (check_for_updates = "
                         "false). No network call was made.\n"
                         + info["instructions"], "system")
            return
        if not info["ok"]:
            self._append("Could not reach PyPI (offline?). Nothing was "
                         "changed.\nManual upgrade: " + info["instructions"],
                         "system")
            return
        lines = [f"Installed: {info['current']}    Latest: {info['latest']}",
                 f"Install method: {info['method']}"]
        if not info["update_available"]:
            self._append("\n".join(lines + ["You are up to date."]), "system")
            return
        lines.append(info["instructions"])
        self._append("\n".join(lines), "system")
        if info["command"] is None:
            return                      # editable/source: instructions only
        if not run_now and not self._confirm(
                f"Upgrade to {info['latest']} now? (y/N):"):
            self._append("upgrade cancelled", "system")
            return
        self._append("Running: " + " ".join(info["command"]), "system")
        out: list[str] = []
        code = update_mod.run_upgrade(info["command"], on_line=out.append)
        tail = "\n".join(out[-40:])
        prefix = tail + "\n" if tail else ""
        if code == 0:
            self._append(prefix + f"Upgrade finished (exit 0). Restart "
                         f"Pedacito to run {info['latest']}.", "system")
        else:
            self._append(prefix + f"Upgrade failed (exit {code}).\n"
                         f"Manual upgrade: {info['instructions']}", "error")

    def _note_update(self, info: dict) -> None:
        """The startup check's daemon-thread result: remember it, and drop
        one quiet line into the transcript when a newer release exists."""
        self.update_info = info
        if info.get("update_available"):
            self._append(f"Pedacito v{info['latest']} is available (you have "
                         f"{info['current']}). Run /update to upgrade.",
                         "system")

    def _confirm(self, question: str) -> bool:
        self._set_banner(question)
        self._append(question, "activity")
        self._redraw(force=True)
        answer = self._read_answer_line()
        self.input_buf = ""
        self.cursor_pos = 0
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
        self.scroll_offset = 0
        self.sidebar_scroll_offset = 0
        self._scroll_seen_total = 0
        self._scroll_seen_width = 0
        self.idx = None
        self.agent = None
        self._append(f"reset: removed {result['files']} file(s), kept "
                     f"{', '.join(result['kept']) or '(nothing)'}", "system")
        self._dirty = True

    # --------------------------------------------------------------- terminal
    def _signal_handler(self, _signum, _frame):
        # The run loop / task joiner catches KeyboardInterrupt and the
        # try/finally restores the terminal; restoring here would leave the
        # session alive with cbreak/alternate-screen already torn down.
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
        self._enter_raw_mode()
        self._terminal_ready = True
        self._write(_ALT_SCREEN_ON)
        if self.mouse_sgr:
            self._write(_MOUSE_SGR_ON)

    def _restore_terminal(self) -> None:
        if self._terminal_ready:
            self._terminal_ready = False
            self._write(_ALT_SCREEN_OFF)
            self._restore_raw_mode()
        # Signal handlers are restored even when setup failed part-way, so an
        # exception during _prepare_terminal never leaves SIGINT rerouted.
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
        self._restore_raw_mode()

    # ------------------------------------------------------------------ run
    def _idle_tick(self) -> None:
        """Poll events/telemetry and repaint while no key is available."""
        self._poll_events()
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
        if key != self._last_telemetry_key:
            self.last_telemetry = snapshot
            self._last_telemetry_key = key
            self._dirty = True
        if self._dirty:
            self._redraw()

    def _run_interactive(self) -> int:
        """Raw-key loop: every repaint keeps the prompt and cursor intact."""
        while not self._quit:
            key = self.read_key_event(0.05)
            if key is None:
                self._idle_tick()
                continue
            if key == "CTRL_C":
                if self.input_buf:
                    self.input_buf = ""
                    self.cursor_pos = 0
                    self._dirty = True
                else:
                    self._append("cancelled (the session stays open)", "error")
                if self._dirty:
                    self._redraw()
                continue
            # Ctrl+D scrolls down when there is scrollback to consume; at the
            # live bottom with an empty prompt it keeps its old "leave" role.
            if key == "CTRL_D" and not self.input_buf \
                    and self.scroll_offset == 0:
                break
            self.handle_key(key)
            if self._quit:
                break
            if self._dirty:
                self._redraw()
        return 0

    def _run_headless(self) -> int:
        while not self._quit:
            line = self._readline()
            if line is None:
                break
            if not line.strip():
                continue
            self.handle_line(line)
            self._redraw(force=True)
        return 0

    def run(self, initial_task: str = "") -> int:
        """Paint the session and read commands until EOF or /quit."""
        self._running = True
        try:
            self._prepare_terminal()
            self._redraw(force=True)
            task = initial_task or self.initial_task
            if task:
                self.handle_line(task)
                self._redraw(force=True)
            if self.interactive:
                return self._run_interactive()
            return self._run_headless()
        except KeyboardInterrupt:
            if self.headless:
                self._append("interrupted", "error")
                try:
                    self._redraw(force=True)
                except Exception:
                    pass
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
