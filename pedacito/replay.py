"""Record a run's model responses, and replay them later with no server.

    pedacito ask . -t "..." --record              # .pedacito/replay/<stamp>.jsonl
    pedacito ask . -t "..." --replay FILE         # same run, no model needed

A recording is the ordered list of what the model returned (text, parsed step,
or the error) plus the flags the harness reads after a call (a cut-off reply).
Replaying feeds them back in order, so the whole harness -- gather loop,
folding, preflight, edit parsing, repair -- runs deterministically and free.
That is how you test a change to the HARNESS rather than to the model: if a
prompt, fold or recovery change alters what the model would be asked, the
replay reports it ("call 7 diverged") instead of silently feeding it the old
answer. Only the main model is recorded; role models (`roles`) and embeddings
are not. Prompts are stored as a hash plus a short redacted preview, never in
full, so a recording carries no secrets from your source.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from . import safety
from .llm import LLMError

_FLAGS = ("step_truncated", "stream_truncated", "stream_repetition",
          "partial_text", "truncated_field")


def _digest(messages) -> str:
    raw = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:12]


def _preview(messages) -> str:
    last = messages[-1]["content"] if messages else ""
    if not isinstance(last, str):
        last = str(last)
    return safety.redact(last[-300:])


def record_path(root) -> Path:
    from . import workspace
    d = Path(root) / ".pedacito" / "replay"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{workspace.stamp()}.jsonl"


class Recorder:
    """Transparent proxy around a client that logs each chat/chat_json result."""

    def __init__(self, client, path: Path):
        object.__setattr__(self, "_c", client)
        object.__setattr__(self, "_path", Path(path))
        object.__setattr__(self, "_n", 0)
        self._write({"kind": "meta", "model": getattr(
            getattr(client, "cfg", None), "model", ""), "ts": time.time()})

    def _write(self, rec: dict) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def _call(self, kind: str, messages, fn, *args, **kw):
        rec = {"kind": kind, "n": self._n, "digest": _digest(messages),
               "preview": _preview(messages)}
        object.__setattr__(self, "_n", self._n + 1)
        try:
            out = fn(messages, *args, **kw)
        except LLMError as e:
            rec["error"] = str(e)
            self._write(rec)
            raise
        rec["out"] = out
        rec["flags"] = {k: getattr(self._c, k, None) for k in _FLAGS
                        if getattr(self._c, k, None)}
        self._write(rec)
        return out

    def chat(self, messages, *args, **kw):
        return self._call("chat", messages, self._c.chat, *args, **kw)

    def chat_json(self, messages, *args, **kw):
        return self._call("chat_json", messages, self._c.chat_json, *args, **kw)

    def __getattr__(self, name):
        return getattr(self._c, name)

    def __setattr__(self, name, value):
        setattr(self._c, name, value)


class Replayer:
    """A client that answers from a recording. Raises when it runs out."""

    step_truncated = False
    stream_truncated = False
    stream_repetition = False
    partial_text = ""
    truncated_field = ""

    def __init__(self, path):
        self.path = Path(path)
        self.records = []
        for ln in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if rec.get("kind") in ("chat", "chat_json"):
                self.records.append(rec)
        self.used: set[int] = set()
        self.diverged: list[int] = []
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "seconds": 0.0, "cache_hit": 0.0}
        self.site = ""
        self.fewshot: list = []
        self.on_call = None
        self.cfg = None
        self._schema_supported = True
        self.chars_per_token = 0.0

    def _pick(self, kind: str, messages) -> dict | None:
        """The recorded call this request corresponds to: the next unused one
        of the same kind with the same prompt hash, else the next unused of the
        same kind (flagged as diverged). Calls the recorded run made that this
        one skips (e.g. summaries now cached) are simply left unused."""
        digest = _digest(messages)
        same = [i for i, r in enumerate(self.records)
                if i not in self.used and r["kind"] == kind]
        for i in same:
            if self.records[i].get("digest") == digest:
                self.used.add(i)
                return self.records[i]
        if same:
            i = same[0]
            self.used.add(i)
            self.diverged.append(self.records[i].get("n", i))
            return self.records[i]
        return None

    def _next(self, kind: str, messages):
        rec = self._pick(kind, messages)
        if rec is None:
            raise LLMError(f"replay exhausted: the recording has no further "
                           f"`{kind}` call (the harness now makes more calls "
                           "than it did when this was recorded)")
        self.stats["calls"] += 1
        flags = rec.get("flags") or {}
        for k in _FLAGS:
            blank = "" if k in ("partial_text", "truncated_field") else False
            setattr(self, k, flags.get(k, blank))
        if "error" in rec:
            raise LLMError(rec["error"])
        return rec["out"]

    def chat(self, messages, *args, **kw):
        return self._next("chat", messages)

    def chat_json(self, messages, *args, **kw):
        return self._next("chat_json", messages)

    def check(self, warn: bool = True) -> str:
        return "(replay)"

    def warmup(self) -> None:
        return None

    def model_info(self):
        return None

    def loaded_models(self):
        return None

    def probe_json_modes(self):
        return []

    def cancel(self) -> None:
        return None

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def embed(self, texts):
        raise LLMError("embeddings are not recorded; replay has no embedding model")

    def summary(self) -> str:
        used = f"{len(self.used)}/{len(self.records)} recorded call(s) used"
        if self.diverged:
            return (f"replay: {used}; the harness's prompt differed from the "
                    f"recording at call(s) {self.diverged[:8]}")
        return f"replay: {used}; every prompt matched the recording"


def report(client, file=None) -> None:
    if isinstance(client, Replayer):
        print("\n[" + client.summary() + "]", file=file or sys.stderr)
