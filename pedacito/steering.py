"""Interactive steering and clarifying questions.

    with steering.Steering() as steer:          # while a task runs in `chat`
        cfg._steer = steer.queue                # gather delivers between steps

The user can type a line while the model is looking things up ("stop reading
tests, the bug is in sync.py"). Lines are queued and delivered at the next step
boundary, appended to the turn that has not been sent yet -- so the server's
cached prefix is untouched and the redirection costs a few tokens. Reading uses
`select` on POSIX and `msvcrt` on Windows with a short poll, so it never
swallows the NEXT prompt: the reader is stopped before `input()` is called
again.

`ask_user` is the other direction: the model's `question` action. It exists
only in interactive sessions (never in `auto`, where the model must state the
ambiguity and take the safest reading) and is bounded per task.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time

MAX_QUESTIONS = 2


class Steering:
    """Background reader that turns typed lines into queued steering notes."""

    def __init__(self, fd: int | None = None, poll: float = 0.15):
        self.queue: "queue.Queue[str]" = queue.Queue()
        self._fd = sys.stdin.fileno() if fd is None else fd
        self._poll = poll
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._buf = ""

    @staticmethod
    def available() -> bool:
        """Steering needs an interactive terminal."""
        try:
            return sys.stdin.isatty() and sys.stdout.isatty()
        except (AttributeError, ValueError):
            return False

    def _ready(self) -> bool:
        if os.name == "nt":                      # pragma: no cover - Windows
            import msvcrt
            return msvcrt.kbhit()
        import select
        try:
            r, _w, _x = select.select([self._fd], [], [], self._poll)
        except (OSError, ValueError):
            return False
        return bool(r)

    def _read(self) -> str:
        if os.name == "nt":                      # pragma: no cover - Windows
            import msvcrt
            out = ""
            while msvcrt.kbhit():
                out += msvcrt.getwche()
                if out.endswith("\r"):
                    out = out[:-1] + "\n"
                    break
            return out
        try:
            return os.read(self._fd, 4096).decode("utf-8", "replace")
        except OSError:
            return ""

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._ready():
                if os.name == "nt":              # pragma: no cover
                    time.sleep(self._poll)
                continue
            chunk = self._read()
            if not chunk:
                time.sleep(self._poll)
                continue
            self._buf += chunk
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    self.queue.put(line.strip()[:500])

    def start(self) -> "Steering":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> list[str]:
        """Stop reading; returns any notes never delivered."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        left = []
        while True:
            try:
                left.append(self.queue.get_nowait())
            except queue.Empty:
                return left

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def drain(q) -> list[str]:
    """Pending steering notes (oldest first); [] when there is no queue."""
    out: list[str] = []
    if q is None:
        return out
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def terminal_question(text: str) -> str:
    """Ask the user on the terminal; '' when they give no answer."""
    print(f"\n  ? {text}", file=sys.stderr)
    try:
        return input("  your answer (Enter to let the model choose): ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
