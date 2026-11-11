"""The two-phase agent: gather relevant source, then answer or edit.

Usage
-----
    from .agent import Agent

    agent = Agent(index, llm_client, cfg)
    gathered = agent.gather("add retry logic to the sync client")
    reply = agent.answer(task, gathered)
    results, reply = agent.apply(reply, dry_run=True, task=task, gathered=gathered)

Phase 1 (gather): the model picks tools under a JSON grammar (see tools.py), so
tool calls never need freeform-text parsing. max_steps = 0 (the default) runs
until the model says `done`, the gathered-source budget is exhausted, or one
of the action-independent global bounds (cfg.max_total_model_calls,
cfg.max_no_progress_steps) ends the loop cleanly with a harness note; a
positive max_steps is a hard limit. When the model says `done` but files the
task names were never read, one nudge asks it to read them (repeated only
after real progress).

Phase 2 (answer): free text, no grammar. Code is not forced through a JSON
string field because escaping degrades small-model output; edits are fenced
search/replace blocks instead (see edits.py).

The system message and repo map come first and stay constant within a gather
loop, so the server's prefix cache hits on every turn.

A content-addressed evidence store hashes every source span the harness
displays (`read`, prefetch, `context` -- post token-clip) and records it on
the Agent as an `EvidenceSpan`. Two rehydration paths hand a span back to the
model without ever rewriting history: the explicit
`NEED path:120-168` directive (answer, repair and verifier replies) and the
implicit repair auto-rehydration for unread-region / fuzzy-match findings
that touch recorded evidence. Both APPEND at the conversation tail, so the
prompt-prefix cache survives the round (TECHNIQUES.md, section 11).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field

from . import diagnostics as diagnostics_mod
from . import events as events_mod
from . import fileio
from . import fold as fold_mod
from . import gitinfo
from . import hooks as hooks_mod
from . import instructions as instructions_mod
from . import mcp as mcp_mod
from . import memory as memory_mod
from . import prompts as prompts_mod
from . import safety
from . import skills as skills_mod
from . import steering as steering_mod
from . import todo as todo_mod
from .config import is_hosted
from .edits import Result, apply_edits, locate, malformed_blocks, parse_edits
from .llm import LLMError, estimate_messages, estimate_tokens
from .tools import TOOL_HELP, dispatch, mcp_injection, schema_for

# Re-asks after a gather step reply dies at the output cap or has no salvageable
# JSON, before giving up and answering from what was gathered.
_MAX_GATHER_RECOVERIES = 2

# Consecutive re-reads that add no new source before gather stops (one duplicate
# only gets a compact "already read" observation).
_MAX_DUPLICATE_READS = 2

# Failed steps (bad path, bad range, disabled action) are charged to the budget;
# a streak of errors or the same exact step repeated ends the gather phase.
_MAX_CONSECUTIVE_ERRORS = 4
_MAX_REPEATED_STEPS = 3

# Read-only steps whose result cannot change when the previous step is the
# identical call. `read` has its own duplicate-range guard.
_DEDUP_STEP_ACTIONS = {"grep", "find", "outline", "usages", "callers",
                       "sessions", "diff", "git_diff", "diagnose"}

# Read-only steps whose repeat (even many steps later) adds nothing while the
# index, the conversation and the working tree are unchanged.
_CACHEABLE_ACTIONS = {"grep", "find", "outline", "usages", "callers",
                      "context", "git", "git_diff", "diagnose", "sessions",
                      "fetch", "fetch_url", "web_search"}
# Steps that can change what the cacheable ones return.
_MUTATING = {"write", "append", "move", "delete", "undo", "rename",
             "rename_symbol", "run",
             "run_test", "browser", "screenshot", "mcp", "mcp_call", "task"}

# Steps discarded for a cut-off payload before gather gives up.
_MAX_PAYLOAD_CUTS = 2

# Chars-per-token heuristic for the `read`/`context` token budget
# (cfg.max_read_tokens) when the server has no calibrated tokenizer;
# a calibrated `client.chars_per_token` measures instead.
_READ_CHARS_PER_TOKEN = 3.5

# --- content-addressed evidence + append-only exact-span rehydration
# `NEED path:120-168` re-asks allowed per generation (Guiding Constraint #3:
# a bounded loop). cfg.max_rehydration_rounds overrides.
_MAX_REHYDRATION_ROUNDS = 2
# Lines of slack when a repair finding is matched to a recorded evidence span
# (Guiding Constraint #1: hand back only what is relevant, not the file).
_EVIDENCE_WINDOW = 30
# `read_step` value recorded for spans the harness re-fetched (during NEED
# rehydration or repair auto-rehydration) rather than a gather step showed.
_READ_STEP_REFETCH = -1
# `NEED` lines honoured per round; keeps one reply's tail small (#1).
_MAX_NEEDS_PER_ROUND = 4
# Hard char ceiling on one appended rehydration body.
_MAX_REHYDRATED_CHARS = 8000
# The standing offer the answer, repair and verifier instructions make: ask
# for exact lines instead of guessing them. Lives at the TAIL of a turn the
# harness builds fresh, so it never rewrites cached history bytes.
_NEED_HINT = (
    "\n\nIf a specific line range you need is missing from the source above, "
    "do NOT guess its contents: put `NEED path/to/file.py:START-END` on its "
    "own line (the exact range only, at most two such lines). The harness "
    "will fetch those exact lines and ask you again. If you have everything "
    "you need, answer now and write no NEED line.")


# The block rides in user0, which every lookup call re-sends (the server's
# cache makes it cheap, but `prompt_tokens` still counts it on each call), so
# it is kept as short as it can be: a one-line label and three compact steps.
FEWSHOT_HEADER = "EXAMPLE STEPS (this project):"
# A copied `read` example stays in the transcript for every later call, so the
# user0 examples read only a few lines. (The no-grammar fallback keeps 40.)
FEWSHOT_USER0_READ_LINES = 12


def fewshot_enabled(cfg) -> bool:
    """True when `fewshot_steps` asks for worked steps in the first user turn.
    Only an explicit True (the user's, or a model card's) turns it on."""
    return getattr(cfg, "fewshot_steps", None) is True


def fewshot_actions(file: str, symbol: str, start: int, end: int) -> list[dict]:
    """The three worked steps for a definition `symbol` in `file`."""
    return [
        {"action": "grep", "pattern": symbol,
         "reasoning": "find where it is used"},
        {"action": "read", "file": file, "start": start, "end": end,
         "reasoning": "read the definition"},
        {"action": "done", "reasoning": "have what I need"},
    ]


def fewshot_block(shots: list[dict], schema: dict) -> str:
    """The worked-examples text for the first user turn: three lines, one
    step each, in the step schema's field order, unused fields left out (the
    transcript echoes steps the same way). "" when there are no shots."""
    from .llm import schema_example_with
    lines = []
    for shot in shots[:3]:
        full = json.loads(schema_example_with(schema, **shot))
        lines.append(json.dumps({k: v for k, v in full.items()
                                 if v not in ("", 0, None)}))
    if not lines:
        return ""
    return FEWSHOT_HEADER + "\n" + "\n".join(lines)


@dataclass(frozen=True)
class EvidenceSpan:
    """One piece of exact source the harness actually showed the model.

    `start`/`end` are 1-based inclusive lines as delivered (post token-clip),
    `content_sha` is the SHA-256 of `exact_text`, and `read_step` is the
    gather step that displayed it: 0 for a harness prefetch, a positive step
    number during the lookup loop, or `_READ_STEP_REFETCH` (-1) when the
    harness re-read the range for a NEED rehydration or a repair hint. The
    hash is what makes the span *addressable*: a `NEED` line or a repair
    re-fetch can prove the text now on disk is still the text the model read."""

    file: str
    start: int
    end: int
    content_sha: str
    exact_text: str
    read_step: int


def sha256_text(text: str) -> str:
    """Content hash of one evidence span's exact text."""
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()


@dataclass
class EvidenceStore:
    """Append-only, content-addressed record of the source spans displayed.

    Lookups are tolerant of the two ways a small model writes a path
    (`pkg/a.py` and `./pkg/a.py`), matching `_norm_key`. A span records what
    *was* shown; it is never a promise about what is on disk now, so every
    rehydration reads afresh and compares hashes instead of serving the
    stored text back. `record_displayed` is how the gather loop feeds this
    store from a post-clip observation."""

    spans: list[EvidenceSpan] = field(default_factory=list)

    def record(self, file: str, start: int, end: int, exact_text: str,
               read_step: int) -> EvidenceSpan:
        """Append one displayed span; returns what was recorded."""
        span = EvidenceSpan(fileio.normalise_rel(file), int(start), int(end),
                            sha256_text(exact_text), exact_text,
                            int(read_step))
        self.spans.append(span)
        return span

    def record_displayed(self, obs: str, read_step: int) -> EvidenceSpan | None:
        """Record the numbered source lines a delivered `read`/`context`
        observation actually shows, stripping the `NNN | ` prefixes.

        Called after the token clip, so the range and the hashed text are what
        the model saw -- not what was requested. Returns None when the text is
        not a numbered read (an outline, an ERROR line, a compacted
        duplicate-range note): those display no quotable source."""
        m = _READ_HEAD_RX.match(obs or "")
        if not m:
            return None
        head_len = obs.index("\n") + 1 if "\n" in obs else len(obs)
        lines = obs[head_len:].split("\n")
        # The char-budget tail can cut the final numbered line mid-text; a
        # half line is not addressable source, so it is not credited.
        if lines and lines[-1].startswith("[truncated: context budget"):
            while lines and lines[-1].startswith("["):
                lines.pop()
            while lines and not _NUMBERED_LINE_RX.match(lines[-1]):
                lines.pop()
            if lines:
                lines.pop()           # the possibly-cut last line
        lo = int(m.group(2))
        shown: list[str] = []
        n = lo
        for ln in lines:
            pm = _NUMBERED_LINE_RX.match(ln)
            if not pm:
                # A mid-file read shows a `[context: enclosing signature;
                # module imports]` preamble before its first numbered line:
                # navigation, not source -- step over it, stop at real prose.
                if not shown and ln.startswith("["):
                    continue
                break             # notes / trailing text: the display ends
            if int(pm.group(1)) != n:
                break
            rest = ln[pm.end():]      # everything after the `|`
            shown.append(rest[1:] if rest.startswith(" ") else rest)
            n += 1
        if not shown:
            return None
        start, end = lo, lo + len(shown) - 1
        return self.record(m.group(1).strip(), start, end,
                           "\n".join(shown), read_step)

    def _match(self, want: str) -> list[EvidenceSpan]:
        key = _norm_key(want)
        return [s for s in self.spans
                if _norm_key(s.file) == key
                or _norm_key(s.file).endswith("/" + key)
                or key.endswith("/" + _norm_key(s.file))]

    def for_file(self, file: str) -> list[EvidenceSpan]:
        return self._match(file)

    def covering(self, file: str, start: int, end: int) -> list[EvidenceSpan]:
        """Spans inside `file` that already show every line of `start`-`end`."""
        lo, hi = int(start), int(end)
        return [s for s in self._match(file) if s.start <= lo and hi <= s.end]

    def nearby(self, file: str, start: int, end: int,
               window: int = _EVIDENCE_WINDOW) -> list[EvidenceSpan]:
        """Spans in `file` overlapping `start`-`end` or within `window` lines
        of it: the proximity rule the repair auto-rehydration is gated on."""
        lo, hi = int(start), int(end)
        return [s for s in self._match(file)
                if not (s.end < lo - window or s.start > hi + window)]

    def exact(self, file: str, start: int, end: int) -> EvidenceSpan | None:
        """The newest span recorded for exactly this range, or None."""
        hits = [s for s in self._match(file)
                if (s.start, s.end) == (int(start), int(end))]
        return hits[-1] if hits else None

    def digest(self) -> str:
        """One line per recorded span: `rel a-b sha=…`. A cheap durable
        record of what was displayed, hash-pinned so a later `NEED` or a
        repair can address exactly the text the model saw -- even after the
        conversation itself has been folded or evicted."""
        return "\n".join(
            f"{s.file} {s.start}-{s.end} sha={s.content_sha[:12]}"
            for s in self.spans)[:2000]


def _step_signature(step: dict) -> tuple:
    """Identity of a gather step for repetition detection. `reasoning` is
    excluded: the same action with a reworded note is still the same loop."""
    return tuple(sorted(
        (k, str(v)) for k, v in (step or {}).items()
        if k != "reasoning"))


def _norm_key(path: str) -> str:
    """Normalise a read-ledger path. Models alternate between `pkg/a.py` and
    `./pkg/a.py`; the ledger, eviction pass and coverage must share one key."""
    return fileio.path_key(path)


# Spots deferral language ("the next turn continues") in a cleanly finished
# reply so it triggers another answer turn; a false positive costs one
# "NOTHING MORE" round-trip.
_MORE_TO_DO_RE = re.compile(
    r"next turn|will follow|following turn|still to do|left to do|"
    r"remaining after|remaining work|remains to|to be done|not yet done",
    re.IGNORECASE,
)
# First line of a successful tool_read observation: "prep/index.html lines 315-325:".
_READ_HEAD_RX = re.compile(r"^(.+?) lines (\d+)-(\d+):")
# File-mode `context` header (tools.tool_context): "FILE CONTEXT prep/app.js".
_FILE_CTX_RX = re.compile(r"^FILE CONTEXT (\S+)", re.M)
# A numbered source line in a read observation: "  42 | text".
_NUMBERED_LINE_RX = re.compile(r"\s*(\d+) \|")
_FILE_TOKEN_RX = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./\-]*")
_FILE_EXT_RX = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,7}$")


def _subtract_spans(uncovered: list[tuple[int, int]],
                    covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove every `covered` (start, end) span from the `uncovered` spans."""
    for a, b in covered:
        nxt = []
        for cs, ce in uncovered:
            if b < cs or a > ce:
                nxt.append((cs, ce))
                continue
            if cs < a:
                nxt.append((cs, a - 1))
            if ce > b:
                nxt.append((b + 1, ce))
        uncovered = nxt
    return uncovered


def _first_seen_spans(spans_by_file: dict[str, list[tuple[int, int]]]
                      ) -> dict[str, list[tuple[int, int]]]:
    """Replay each file's read spans in step order and return, per file, the
    (start, end) sub-spans that were NEW when they were first shown."""
    out: dict[str, list[tuple[int, int]]] = {}
    for key, seq in spans_by_file.items():
        seen: list[tuple[int, int]] = []
        deltas: list[tuple[int, int]] = []
        for s, e in seq:
            deltas.extend(_subtract_spans([(s, e)], seen))
            seen = _fold_spans(seen + [(s, e)])
        if deltas:
            out[key] = deltas
    return out


def _split_read_obs(obs: str) -> tuple[str, str]:
    """Split a read observation into (header line, body)."""
    head_end = obs.index("\n") if "\n" in obs else len(obs)
    return obs[:head_end], obs[head_end + 1:]


def _slice_numbered(obs: str, keep: list[tuple[int, int]]) -> str:
    """Narrow a tool_read observation ("rel lines a-b:\\nNNN | text...") to the
    numbered lines inside `keep` spans. Returns the original text if the format
    is unexpected."""
    m = _READ_HEAD_RX.match(obs)
    if not m:
        return obs
    head, body = _split_read_obs(obs)
    kept: list[str] = []
    for ln in body.split("\n"):
        pm = _NUMBERED_LINE_RX.match(ln)
        if not pm:
            kept.append(ln)     # trailing note lines ([truncated...] etc.)
            continue
        n = int(pm.group(1))
        if any(a <= n <= b for a, b in keep):
            kept.append(ln)
    if not kept:
        return f"{m.group(1).strip()} lines {m.group(2)}-{m.group(3)}: [all of these lines were duplicates of an earlier read]"
    return head + "\n" + "\n".join(kept)


def _delta_len(obs: str, prior_spans: list[tuple[int, int]]) -> int:
    """Length of the NEW part of a read observation: the lines not covered by
    `prior_spans`. Used to charge max_gathered_chars per new byte."""
    if not _READ_HEAD_RX.match(obs):
        return len(obs)
    _, body = _split_read_obs(obs)
    total = 0
    for ln in body.split("\n"):
        pm = _NUMBERED_LINE_RX.match(ln)
        if not pm or not any(a <= int(pm.group(1)) <= b for a, b in prior_spans):
            total += len(ln) + 1
    return total


def _read_has_new_lines(obs: str, prior_spans: list[tuple[int, int]]) -> bool:
    """True when a read observation contains source lines not in `prior_spans`.

    Separate from `_delta_len`: trailing notes ("[truncated ...]") are tokens
    but not new source, and counting them would let an identical truncated
    re-read evade the duplicate-read guard.
    """
    if not _READ_HEAD_RX.match(obs):
        return True
    _, body = _split_read_obs(obs)
    for ln in body.split("\n"):
        pm = _NUMBERED_LINE_RX.match(ln)
        if pm and not any(a <= int(pm.group(1)) <= b for a, b in prior_spans):
            return True
    return False


def _evict_duplicate_reads(messages: list[dict], gathered: list[str],
                           spans: dict[str, list[tuple[int, int]]]) -> list[str]:
    """Rewrite duplicate source out of the conversation and the SOURCE block.

    Overlapping re-reads (L1-685 then L401-685) leave several copies of the
    same lines in the live conversation, which can overflow the server's
    window. Each read's RESULT turn is cut down to the lines it was the first
    to show, and the gathered SOURCE items are rebuilt from the same deltas so
    the answer prompt sees every line once. Returns the new item list.
    """
    if not spans or not any(len(seq) > 1 for seq in spans.values()):
        return gathered
    deltas = _first_seen_spans(spans)

    new_items: list[str] = []
    for g in gathered:
        head = g.split("\n", 1)[0]
        if head.startswith("--- read ") and "\n" in g:
            rest = head[len("--- read "):].strip()
            key = _norm_key(rest.rsplit(" L", 1)[0])
            body = g.split("\n", 1)[1]
            keep = deltas.get(key)
            if keep is None:
                new_items.append(g)
            else:
                new_items.append(head + "\n" + _slice_numbered(body, keep))
        else:
            new_items.append(g)

    # Oldest -> newest, so each turn keeps only lines no earlier turn showed.
    for i, msg in enumerate(messages):
        if msg["role"] != "user" or not msg["content"].startswith("RESULT:\n"):
            continue
        body = msg["content"][len("RESULT:\n"):]
        m = _READ_HEAD_RX.match(body)
        if not m:
            continue
        key = _norm_key(m.group(1))
        s_, e_ = int(m.group(2)), int(m.group(3))
        prior = _fold_spans(_turn_spans(messages[:i], key))
        uncovered = _subtract_spans([(s_, e_)], prior)
        if uncovered == [(s_, e_)]:
            continue          # this turn showed nothing that came before it
        trimmed = _slice_numbered(body, uncovered) if uncovered else \
            f"{m.group(1).strip()} lines {s_}-{e_}: [duplicate of an earlier read; nothing new]"
        messages[i] = {"role": "user", "content": "RESULT:\n" + trimmed}
    return new_items


def _turn_spans(messages: list[dict], key: str) -> list[tuple[int, int]]:
    """All (start, end) spans for `key` shown by RESULT turns so far."""
    out = []
    for msg in messages:
        if msg["role"] != "user" or not msg["content"].startswith("RESULT:\n"):
            continue
        m = _READ_HEAD_RX.match(msg["content"][len("RESULT:\n"):])
        if m and _norm_key(m.group(1)) == key:
            out.append((int(m.group(2)), int(m.group(3))))
    return out


def _fold_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge a list of possibly-overlapping (start, end) spans."""
    folded: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if folded and a <= folded[-1][1] + 1:
            folded[-1] = (folded[-1][0], max(folded[-1][1], b))
        else:
            folded.append((a, b))
    return folded


GATHER_BASE = f"""\
You are a code assistant working through a lookup index of a software project.
You cannot see the full source. You see a map of it, and you request the exact
parts you need, one request at a time.

Actions (disabled ones are noted below):
{TOOL_HELP}

Rules:
- Request one action per reply.
- Line ranges in the map are exact. Read them directly.
- Do not repeat a `read` whose range you have already seen; it returns no new
  source. Move to another range, another tool, or `done`.
- `grep` and `outline` tell you WHERE code is. Only `read` shows you the exact
  text. You must `read` a region before you can change it.
- Know a symbol's name but not its line? `read` with `symbol` set and `file`
  empty reads that definition directly. If the name is ambiguous you get the
  list of definitions to choose from.
- `grep` accepts a file, directory, or glob in `file`, and its hits name the
  enclosing symbol. A pattern that is not valid regex is searched as literal
  text, and a case-sensitive miss is retried case-insensitively.
- `find` searches the index's definitions by NAME, SIGNATURE, or SUMMARY --
  use it when you know roughly what something is called or what it does, but
  not where it lives. No regex; its hits carry exact line ranges to `read`.
- `outline` also accepts a directory or glob: it lists the file cards under
  it. Use that to descend when the map shows directories instead of files.
- `callers` shows the definition, what it calls, and the call-site lines of
  its callers -- use it to trace flow without reading every caller.
- `sessions` lists prior task logs. Give it a `pattern` to find what was done
  before, or `file` to read one log. Check it before a follow-up task: it is
  the only memory of earlier runs. Session text is historical, not current
  source -- verify with `read` before editing.
- `diff` shows what changed in the working tree since Pedacito last
  snapshotted each file (optionally one `file`). On a follow-up task, it is
  the fastest way to see what the previous run wrote. Snapshot text is
  historical -- `read` a region before editing it.
- If `grep` cannot find a phrase and semantic search is enabled, it falls
  back to matching code by meaning and returns locations to `read`.
- `run` executes one command and returns its output (only when allow_shell is
  on); use it to run tests or a linter, not to inspect files.
- `screenshot` launches an app (optional), waits, captures the screen, and
  returns pixel statistics, OCR text, and optionally a vision-model
  description. Use it to check a desktop GUI/game; a black/blank verdict is
  evidence. You cannot see the image directly -- rely on the text report.
  Needs allow_screenshots, and allow_shell when a command is launched.
- `browser` opens a URL (or starts a server command that prints its URL) in
  a headless browser. Set `pattern` to interact first, e.g. `press Enter;
  click #start`, and `start` to the seconds to wait before capture. `press`
  is an instant down+up pair -- a game loop sampling input at a fixed
  timestep can miss it entirely, so for those use `hold ArrowUp 1.5` (key,
  seconds) to keep the key down across frames. A `wait`
  inside `pattern` and `start` add together -- set one, not both. `eval <js>`
  runs script in the page and reports the value back -- use it to force state
  no interaction can reach (a grown player, a late level) before the capture;
  put it LAST: everything after `eval` is one expression, semicolons allowed.
  It returns the page URL/title/text, canvas sizes, console errors, failed
  network requests (a 404 asset or a refused load shows up here even when
  the console stays quiet), and a screenshot.
  Use it for anything served over HTTP. A desktop `screenshot` cannot see a
  web page that no browser opened. Needs allow_browser; launching a server
  command also needs allow_shell.
- `eval` is a measuring instrument, not only a state forcer: the value it
  returns is evidence you can quote, while a screenshot's vision
  description is secondhand and often hedged. To verify a visual claim,
  compute a NUMBER in the page -- count a sprite's non-background pixels
  with `getImageData`, measure its drawn bounding box, or diff two forced
  animation frames -- and return compact numbers or short JSON (the
  logged value clips at 240 chars). "Frog pixels 84 -> 756 when scale
  went 1 -> 3" is a defensible verdict; "looks larger" is not. The step
  reply has its own token budget and every field a schema length cap: keep
  eval JS minified (short names, no comments or newlines), and if the
  harness reports your payload arrived cut off, send a shorter one or split
  the measurement across two calls.
- The page keeps running between your `eval` and the screenshot: a game
  loop animates frames, moves sprites, and resumes spawning in that gap,
  so the capture can show a later state than the one you forced. Build
  forced states to survive it -- zero velocities, pin animation timers,
  hold elapsed time under any spawn/phase threshold, or set the page's
  paused flag if it has one -- and re-assert the state right before each
  capture. When an eval'd number disagrees with the screenshot, the gap
  explains it: re-force and re-measure rather than trusting either alone.
- Canvas pixels are not game coordinates. Pixel-art sprites are drawn
  zoomed -- each grid cell a block of device pixels, a 4x or 6x zoom is
  common -- and may sit under a live `ctx.scale()`/`ctx.translate()`
  transform (a growth multiplier, a camera). Before sampling pixels or
  bounding boxes with `eval`, derive the mapping from the render source
  or from canvas.width versus the logical world size, and scan a
  generous region around a sprite's position instead of one exact point.
- Do not claim dynamic behavior (spawn rates, animation, timing) from one
  screenshot or from reading code. If the app needs interaction and the
  available tools cannot drive it, say so plainly instead of guessing.
- When the task asks you to LOCATE the code behind a behavior, the first
  matching file is a hypothesis, not an answer: before `done`, confirm the
  program's entry point actually loads it -- the import, the `<script>` tag,
  or the instantiation site. A matching file that nothing loads is dead
  code, and "fixing" it changes nothing.
- Before blaming or editing a CSS/layout rule, confirm the selector actually
  matches the live element: read the markup for its id/classes, and use
  `browser` + `eval` for computed styles or dimensions when available. A rule
  whose selector matches nothing has no effect, however plausible it looks --
  check for a script (canvas.width/height, inline styles) that sets the value
  instead.
- For a web UI, confirm WHICH document is actually served before editing a
  template or stylesheet: read the route that answers `/`, and check the
  browser report's TITLE and PAGE TEXT. A template is not live unless the
  route renders it; a missing static file can silently select an inline
  fallback, and that fallback may never link the stylesheet you are editing.
- `move` and `delete` are available only when allow_write is on. Prefer the
  search/replace answer format for edits; use these for whole-file moves or
  removals the task explicitly asks for.
- Do not guess at code you have not read. If you need it, read it.
- Choose `done` as soon as you have enough to do the task. Do not browse.
- Fill every field. Use "" for unused strings and 0 for unused numbers.
- Keep `reasoning` under 15 words. It is a note to yourself, not a plan."""

# Map tier hints: every unit listed.
FLAT_HINT = """
The map below lists every function and class in the project with its line range.
Go straight to `read` for the ones you need."""

# One card per file.
CARDS_HINT = """
The map below is one card per FILE, not per function. Each card names the file's
top-level definitions with their line ranges, but not their bodies or their
methods.

How to navigate it:
- If a card already shows the symbol and line range you want, `read` it directly.
- If you need to see inside a file (its methods, or definitions the card did not
  list), `outline` that file first. Outline is cheap; guessing is not.
- `grep` when you do not know which file something lives in. Set `file` to
  restrict the search to one file.
- Never invent a line range. If a range is not in the map or an outline you have
  already run, you do not know it yet."""

# The largest projects: likely-relevant file cards over a directory skeleton.
RANKED_HINT = """
The map below is too big to list every file, so it has two parts. First,
LIKELY RELEVANT FILES: the files most related to your task (named in it,
matching its words, or close to those in the import graph), with their
definitions and line ranges -- `read` straight from them. Second, a
DIRECTORY SKELETON of the whole project.

How to navigate it:
- Start with the likely-relevant files; the "why listed" line says why each is
  there. That ranking is a hint: it can be wrong.
- If what you need is not listed, `find` by name or concept, `grep` for text,
  or `outline` a directory from the skeleton to descend.
- Never invent a line range: use one from a card or an outline you have run."""

# One card per directory.
DIRS_HINT = """
The map below is one card per DIRECTORY, not per file: this project is too big
to list its files up front. Each card names some of the directory's files.

How to navigate it:
- `outline` a directory to see its files' cards, with line ranges.
- `outline` a file for its units; `read` for exact text.
- A file already named in a card can be `read` or `outline`d directly.
- `grep` when you do not know where something lives; its hits name the file,
  line, and enclosing symbol.
- Never invent a line range: use one from a card or an outline you have run."""

# The search/replace edit format shared by every answer brief; custom system
# prompts must carry it so the model does not invent its own format.
EDIT_FORMAT = """\
If the task requires changing code, output edits as search/replace blocks in
exactly this format, and nothing else around them:

--- FILE: relative/path.py
<<<<<<< SEARCH
(the exact original lines, copied character for character from what you read)
=======
(the replacement lines)
>>>>>>> REPLACE

Rules for edits:
- The SEARCH text must match the file exactly, including indentation. Do not
  include the line-number prefixes ("  42 | ") that you saw when reading.
- Only quote text you saw via `read`. Lines from `grep` are trimmed and may be
  truncated; a SEARCH block copied from grep output will never match.
- If you did not read the code you want to change, say so and propose nothing
  rather than guessing at an edit.
- You are in the FINAL answer phase: no tools are available, and asking to
  read another file cannot be fulfilled here. Emit search/replace blocks
  only, for source you have already seen. If you lack a file you need, work
  with what you have and note what you could not verify.
- Include enough surrounding lines that the SEARCH text appears only once.
- Instead of quoting old text, a block's FILE header may carry an anchor:
  `--- FILE: path.py LINES 40-60` (a 1-based inclusive line range, copied
  from what you read) or `--- FILE: path.py SYMBOL Class.method` (exactly one
  definition of that file). For an anchored block leave SEARCH EMPTY and put
  the FULL replacement text for that range in REPLACE; an empty REPLACE
  deletes the range. Prefer an anchor when quoting the old text exactly is
  risky, and widen a LINES range until it covers whole statements.
- To replace a WHOLE function, method, or class without echoing any old
  text, drop the SEARCH machinery entirely:

      <<<<<<< SYMBOL path/to/file.py :: ClassName.method_name
      def method_name(self, x):
          return x + 1
      >>>>>>> END_SYMBOL

  No FILE header, no SEARCH, no line numbers: the AST locates the current
  definition (decorators included), re-aligns your indentation to the file's,
  and keeps the decorators if your new body does not repeat them. Use
  `<<<<<<< AFTER_SYMBOL path :: symbol` (same closer) to insert your body
  right AFTER that definition instead of replacing it.
- Keep each block small and local. Use several blocks rather than one large one.
- A single reply must stay short enough to finish: emit at most a few blocks
  per reply, never a whole large file in one REPLACE/SEARCH section. If the
  task needs changes to several files, edit ONE file (or a few small blocks)
  this turn and say what remains; the next turn continues where you left off.
  Long replies get cut off mid-block at the token limit and are discarded.
- Do not output the whole file. Do not output a diff or patch format.
- Do not wrap the blocks in triple-backtick fences; emit them as plain text.
- If you are adding a new function, anchor the SEARCH on an existing nearby line.
- To CREATE a brand-new file: emit exactly one block for it with an EMPTY
  SEARCH section (nothing between <<<<<<< SEARCH and =======) and the full
  file contents in REPLACE.
- To RENAME or MOVE a file: emit exactly two blocks -- (1) an empty-SEARCH
  creation block for the NEW path whose REPLACE is just a short placeholder
  comment; (2) a block for the OLD path whose SEARCH is its FIRST few lines
  and whose REPLACE is EMPTY. Pedacito recognises the pair, moves the old
  file's REAL contents to the new path itself, and removes the old file.
  NEVER copy the old file's contents into either section -- not into REPLACE
  (that would double them) and not into SEARCH (a whole-file SEARCH runs
  past the output limit and gets cut off mid-block). The clearing block's
  SEARCH text is only used to identify which file to move; three to five
  exact opening lines are enough. Never create the placeholder without its
  clearing block in the same reply: Pedacito refuses lone placeholders so a
  half-finished rename cannot leave a stub file behind.

Before the blocks, write a short plain explanation of what you changed and why.
If the task is a question rather than a change, just answer it and output no blocks."""

ANSWER_SYSTEM = """\
You are a code assistant. You have a map of the project and the specific
source you asked to see. Complete the user's task.

""" + EDIT_FORMAT


# ------------------------------------------------------------------------
# Symbol-anchored incremental edit-verify microloops
# ------------------------------------------------------------------------
# A 4B-9B model can emit one small, exact hunk reliably; a multi-file patch is
# where truncation, stale line numbers and cross-block contamination appear.
# The harness therefore asks for a short JSON intent list, resolves one symbol
# at a time, appends that symbol's CURRENT exact source at the conversation
# tail, and accepts exactly one hunk only after a dry apply + deterministic
# checks + the AST/tree-sitter syntax gate. Temporary source is popped after
# each symbol (a one-line receipt remains), so context does not accumulate.
INCREMENTAL_INTENT_SYSTEM = """\
You plan a code change as a short sequence of symbol-sized edits. Reply with
ONLY a JSON array, no prose and no code fences:
[{"file": "relative/path.py", "symbol": "function_or_class",
  "intended_change": "one sentence describing the change"}]

Rules:
- One item per definition that must change. Use the exact symbol name shown in
  the project map or source, or "" for a whole new file.
- Keep the list in a safe edit order (definitions before their callers when
  the callers must be updated too).
- Do not include files or symbols you do not intend to edit.
- If the task needs no code change, reply exactly []."""

_MAX_EDIT_INTENTS = 8
_INTERNAL_SYMBOLS = {"", "file", "whole file", "*", "(file)", "(whole file)"}


def parse_edit_intents(text: str) -> list[dict]:
    """Structured `[{file, symbol, intended_change}]` intents from a reply.

    Tolerates prose and code fences around one JSON array (small models add
    both), and the common wrappers `{"edits": [...]}` / `{"intents": [...]}`.
    Returns [] rather than raising: an unparseable plan falls back to the
    ordinary single-reply edit path."""
    raw = (text or "").strip()
    if not raw:
        return []
    if raw.startswith("```"):
        raw = re.sub(r"^\s*```[A-Za-z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    candidates: list[str] = []
    starts = [i for i in (raw.find("["), raw.find("{")) if i >= 0]
    if starts:
        start = min(starts)
        end = max(raw.rfind("]"), raw.rfind("}"))
        if end > start:
            candidates.append(raw[start:end + 1])
    candidates.append(raw)

    data = None
    for cand in candidates:
        try:
            data = json.loads(cand)
            break
        except (ValueError, TypeError):
            continue
    if isinstance(data, dict):
        for key in ("edits", "intents", "plan", "targets"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data] if (data.get("file") or data.get("symbol")) else []
    if not isinstance(data, list):
        return []

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in data:
        if isinstance(item, str):
            m = re.match(r"\s*([^\s:]+)\s*:\s*([^\s]+)\s*(?:[-:]\s*(.*))?$",
                         item)
            if not m:
                continue
            item = {"file": m.group(1), "symbol": m.group(2),
                    "intended_change": (m.group(3) or "").strip()}
        if not isinstance(item, dict):
            continue
        file = str(item.get("file") or item.get("path") or "").strip()
        symbol = str(item.get("symbol") or item.get("name") or "").strip()
        change = str(item.get("intended_change") or item.get("change")
                     or item.get("why") or "").strip()
        if not file and not symbol:
            continue
        key = (fileio.path_key(file), fileio.path_key(symbol))
        if key in seen:
            continue
        seen.add(key)
        out.append({"file": file, "symbol": symbol,
                    "intended_change": change[:300]})
        if len(out) >= _MAX_EDIT_INTENTS:
            break
    return out


def _same_rel(a: str, b: str) -> bool:
    """Path equality tolerant of `pkg/a.py` vs `./pkg/a.py`."""
    ka, kb = fileio.path_key(a or ""), fileio.path_key(b or "")
    return bool(ka and kb and (ka == kb or ka.endswith("/" + kb)
                               or kb.endswith("/" + ka)))


_TOOLISH_RE = re.compile(
    r"^\s*\{[^}]*['\"]action['\"]\s*:|<tool_call|<function\s*=|<invoke\b",
    re.IGNORECASE)


def _unusable_continuation(reply: str) -> bool:
    """A continued answer that is empty or still shaped like a tool call (the
    model imitating its own earlier JSON actions) must be rebuilt."""
    text = (reply or "").strip()
    return not text or bool(_TOOLISH_RE.match(text))


_ITEM_SPLIT = re.compile(r"\n\n(?=--- )")


# ---------------------------------------------------- utility source packing
# At the gather -> answer/edit boundary the source is packed by
# evidence utility instead of FIFO. Gather itself stays append-only; only this
# rendered copy is reordered/packed, so the live prompt prefix is untouched.
_PACK_HEAD_RE = re.compile(r"^---\s+(\w+)\s*(.*)$")
_PACK_TARGET_RE = re.compile(r"^(.*?)(?:\s+L(\d+)-(\d+))?$")
_PACK_BODY_HEAD_RE = re.compile(r"^(.+?) lines (\d+)-(\d+):", re.MULTILINE)
_PACK_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_PACK_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s*\|")


@dataclass
class _PackItem:
    """One gathered SOURCE item plus the metadata the packer scores."""
    text: str
    order: int
    kind: str = ""
    file: str = ""
    start: int = 0
    end: int = 0
    tokens: set = field(default_factory=set)
    lower: str = ""


def _pack_item(text: str, order: int) -> _PackItem:
    head, _, body = text.partition("\n")
    m = _PACK_HEAD_RE.match(head)
    kind = m.group(1) if m else ""
    target = m.group(2).strip() if m else head.strip("- ")
    meta = _PackItem(text=text, order=order, kind=kind, lower=text.lower())
    tm = _PACK_TARGET_RE.match(target)
    if tm:
        first = (tm.group(1).strip().split() or [""])[0]
        if re.search(r"\.[A-Za-z0-9]{1,8}$", first):
            meta.file = fileio.normalise_rel(first)
        if tm.group(2):
            meta.start, meta.end = int(tm.group(2)), int(tm.group(3))
    bm = _PACK_BODY_HEAD_RE.search(body)
    if bm and kind in ("read", "context", ""):
        meta.file = fileio.normalise_rel(bm.group(1).strip())
        meta.start, meta.end = int(bm.group(2)), int(bm.group(3))
    clean = _PACK_NUMBER_PREFIX_RE.sub("", body).lower()
    meta.tokens = set(_PACK_TOKEN_RE.findall(clean))
    return meta


def _pack_same(a: str, b: str) -> bool:
    a, b = fileio.path_key(a or ""), fileio.path_key(b or "")
    return bool(a and b and (a == b or a.endswith("/" + b)
                             or b.endswith("/" + a)))


def _pack_overlap(a1: int, b1: int, a2: int, b2: int) -> bool:
    if not a1 or not a2:
        return True
    b1 = b1 or 10 ** 9
    b2 = b2 or 10 ** 9
    return not (b1 < a2 or b2 < a1)


def _pack_term_in(term: str, text: str) -> bool:
    term = (term or "").strip().lower()
    if not term or not text:
        return False
    if re.search(r"(?<![\w])" + re.escape(term) + r"(?![\w])", text):
        return True
    stem = term.rstrip("s")
    return stem != term and bool(
        re.search(r"(?<![\w])" + re.escape(stem) + r"(?![\w])", text))


def _pack_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _diagnostic_pins(diagnostics) -> list[tuple]:
    """(file, start, end) spans for active failing diagnostics. Accepts the
    dict quick_findings/new_findings return, or Finding-like objects."""
    pins: list[tuple] = []
    if not diagnostics:
        return pins
    if isinstance(diagnostics, dict):
        for rel, rows in diagnostics.items():
            if isinstance(rows, str):
                rows = [rows]
            for row in rows or ():
                text = row if isinstance(row, str) else str(row)
                m = (re.search(r":(\d+)(?::\d+)?[:\s]", text)
                     or re.search(r"\bline\s+(\d+)", text, re.IGNORECASE))
                line = int(m.group(1)) if m else 0
                pins.append((fileio.normalise_rel(rel), line or 1, line or 0))
        return pins
    if not isinstance(diagnostics, (list, tuple, set)):
        diagnostics = [diagnostics]
    for f in diagnostics:
        rel = fileio.normalise_rel(
            getattr(f, "file", "") or (f.get("file", "")
                                       if isinstance(f, dict) else ""))
        lines = getattr(f, "lines", ()) if not isinstance(f, dict) else \
            f.get("lines", ())
        line = getattr(f, "line", 0) if not isinstance(f, dict) else \
            f.get("line", 0)
        if lines:
            pins.append((rel, int(lines[0]), int(lines[-1])))
        else:
            pins.append((rel, int(line or 1), int(line or 0)))
    return pins


def _item_has_pin(meta: _PackItem, pins) -> bool:
    for rel, a, b in pins:
        if not rel:
            continue
        same = _pack_same(meta.file, rel) if meta.file else False
        if not same and rel.lower() not in meta.lower:
            continue
        if not meta.start or _pack_overlap(meta.start, meta.end, a, b):
            return True
    return False


def _clip_item_to_pins(meta: _PackItem, pins) -> str:
    """Keep only the numbered lines around this item's pinned spans -- used
    when even the pinned set alone exceeds the request budget."""
    if meta.kind not in ("read", "context", "") or not meta.start:
        return meta.text
    spans = []
    for rel, a, b in pins:
        if not _pack_same(meta.file, rel):
            continue
        lo = max(meta.start, (a or meta.start) - 3)
        hi = min(meta.end or (b or a), (b or a) + 3)
        if hi >= lo:
            spans.append((lo, hi))
    if not spans:
        return meta.text
    head, _, body = meta.text.partition("\n")
    return head + "\n" + _slice_numbered(body, _fold_spans(spans))


def _trim_gathered(src: str, too_big, requirements=None, diagnostics=None,
                   index=None) -> tuple[str, int]:
    """Deterministically pack gathered SOURCE items to fit `too_big`.

    Unlike the old FIFO drop, this is evidence-aware:

    * exact spans for requirements that are LOCATED/READ/EDITED and active
      failing diagnostics are PINNED and never evicted wholesale;
    * the rest is scored by requirement-keyword overlap, call-graph proximity,
      and new information per token (near-duplicates lose via token Jaccard);
    * definition + caller items earn a proximity bonus, so a coherent chain
      survives ahead of scattered unrelated reads.

    A single oversize item is still clipped from its front on a line boundary
    as a last resort. Returns (text, items_dropped).
    """
    items = _ITEM_SPLIT.split(src)
    dropped: list[str] = []
    keep_tail = [i for i in items if i.startswith("--- digest")]
    body = [i for i in items if not i.startswith("--- digest")]
    metas = [_pack_item(item, i) for i, item in enumerate(body)]

    def render(idxs: list[int]) -> str:
        parts = [metas[i].text for i in idxs] + keep_tail
        text = "\n\n".join(parts)
        if dropped:
            pointers = "; ".join(dropped[:8]) + ("; ..." if len(dropped) > 8
                                                 else "")
            text = ("[older gathered source trimmed to fit the request limit "
                    f"(re-`read` if you need it again: {pointers})]\n\n"
                    + text)
        return text

    pins = list(requirements.pinned_spans()) \
        if hasattr(requirements, "pinned_spans") else []
    pins += _diagnostic_pins(diagnostics)
    pinned = {i for i, m in enumerate(metas) if _item_has_pin(m, pins)}
    terms = requirements.packing_terms() \
        if hasattr(requirements, "packing_terms") else []
    anchor_files = {rel for rel, _a, _b in pins if rel}

    graph = getattr(index, "graph", None) if index is not None else None
    neighbor_cache: dict[str, set] = {}

    def neighbors(rel: str) -> set:
        if not rel or graph is None:
            return set()
        if rel not in neighbor_cache:
            try:
                neighbor_cache[rel] = set(graph.neighbors(rel))
            except Exception:
                neighbor_cache[rel] = set()
        return neighbor_cache[rel]

    def utility(i: int, keep: list[int]) -> float:
        m = metas[i]
        base = 0.0
        for req_terms, weight in terms:
            hits = sum(1 for t in req_terms if _pack_term_in(t, m.lower))
            base += weight * min(hits, 3)
        prox = 0.0
        if m.file:
            if any(_pack_same(m.file, f) for f in anchor_files):
                prox = 3.0
            elif any(_pack_same(m.file, n) for f in anchor_files
                     for n in neighbors(f)):
                prox = 2.0
        # A coherent definition + caller chain beats scattered reads.
        for j in keep:
            if j == i:
                continue
            other = metas[j]
            if not m.file or not other.file:
                continue
            if _pack_same(m.file, other.file):
                prox = max(prox, 1.5)
            elif other.file in neighbors(m.file):
                prox = max(prox, 1.0)
        base += 2.0 * prox
        base += {"read": 1.5, "context": 1.2, "grep": 0.4,
                 "find": 0.4}.get(m.kind, 0.1)
        sim = 0.0
        for j in keep:
            if j != i:
                sim = max(sim, _pack_jaccard(m.tokens, metas[j].tokens))
        base += 1.25 * (1.0 - sim)       # MMR: near-duplicates lose value
        return base / (1.0 + len(m.tokens) / 600.0)

    keep = list(range(len(metas)))
    while too_big(render(keep)) and len(keep) > 1:
        candidates = [i for i in keep if i not in pinned]
        if not candidates:
            break                       # pins are never evicted
        worst = min(candidates, key=lambda i: (utility(i, keep), i))
        keep.remove(worst)
        head = metas[worst].text.split("\n", 1)[0][4:].strip()
        dropped.append(head[:70])

    # If the pinned set alone is still too big, narrow each pinned read to the
    # exact anchor region instead of dropping it.
    if too_big(render(keep)) and pinned:
        for i in sorted(pinned):
            metas[i].text = _clip_item_to_pins(metas[i], pins)
    text = render(keep)

    if too_big(text) and keep:
        # one item is still too big: keep its tail, whole lines only. A pinned
        # item is never clipped away; if only pins remain, preserve them.
        candidates = [i for i in keep if i not in pinned]
        if not candidates:
            return text, len(dropped)
        idx = max(candidates, key=lambda i: len(metas[i].text))
        only = metas[idx].text
        n_dropped = 0
        while too_big(render(keep)) and len(only) > 4000:
            keep_chars = max(4000, len(only) * 3 // 4)
            cut = only[len(only) - keep_chars:]
            nl = cut.find("\n")
            only = cut[nl + 1:] if 0 <= nl < len(cut) - 1 else cut
            metas[idx].text = only
            n_dropped += 1
        if n_dropped and not dropped:
            dropped.append("the earliest part of one long read")
        text = render(keep)
    return text, len(dropped)


def _compact_result_marker(content: str) -> str:
    """A one-line extractive reminder of a RESULT turn about to be dropped:
    a read compacts to a `read rel:a-b` stub; anything else keeps its
    tool/header line without the source."""
    m = _READ_HEAD_RX.match((content or "")[len("RESULT:\n"):])
    if m:
        return ("RESULT:\n[compacted earlier lookup: read "
                f"{m.group(1).strip()}:{m.group(2)}-{m.group(3)}]")
    body = [ln.strip() for ln in (content or "").splitlines()[1:]
            if ln.strip()]
    if not body:
        return ""
    head = next((ln for ln in body if ln.startswith("--- ")), body[0])
    head = head[4:].strip() if head.startswith("--- ") else head
    if len(head) > 160:
        head = head[:157].rstrip() + "..."
    return f"RESULT:\n[compacted earlier lookup: {head}]"


def _evict_overflow_turns(messages: list[dict], client, cfg) -> bool:
    """Shrink the live conversation until it fits the server's window.

    Returns True if anything was dropped. Drops the OLDEST RESULT turns first
    (their source is already in `gathered`); a lone huge turn is clipped
    line-by-line, keeping its head so the file/line header survives. Never
    touches the map+task message or assistant turns.

    Trimming has hysteresis: it stops a headroom's worth of tokens BELOW the
    window, because each rewrite invalidates the server's prompt-prefix cache
    from that turn onward.
    """
    room = max(cfg.context_tokens - cfg.max_tokens_step
               - cfg.reserve_tokens // 2, 512)
    headroom = max(256, room // 10)
    # A client whose tokenizer was calibrated measures instead of guessing.
    est = getattr(client, "estimate_tokens", None) or estimate_tokens

    def over_by() -> int:
        return sum(est(m["content"]) for m in messages) - room

    changed_any = False
    while over_by() > -headroom:
        idxs = [i for i, m in enumerate(messages)
                if m["role"] == "user" and m["content"].startswith("RESULT:\n")]
        if not idxs:
            break
        if over_by() <= 0:
            # Under the window: trimming is only cache amortisation, so never
            # cost the newest observation.
            if len(idxs) < 2:
                break
            idxs = idxs[:-1]
        i = idxs[0]
        content = messages[i]["content"]
        lines = content.split("\n")
        if len(lines) > 6:
            keep = max(4, len(lines) // 2)
            clipped = "\n".join(
                lines[:keep] + [("[older read dropped to fit the model's "
                                 "context window; re-`read` it if you need "
                                 "the rest]"), "", ""] + lines[-2:])
            # For 8-10 lines the clip can be as large as the original; only
            # rewrite when it saves tokens, else drop the turn so the loop
            # always makes progress.
            if est(clipped) < est(content):
                messages[i] = {"role": "user", "content": clipped}
            else:
                lines = []      # force the removal path below
        if not lines or len(lines) <= 6:
            content_now = messages[i]["content"]
            if not content_now.startswith(
                    "RESULT:\n[compacted earlier lookup:"):
                marker = _compact_result_marker(content_now)
                if marker and est(marker) < est(content_now):
                    # If this still leaves the prompt over budget, the next
                    # pass drops the marker.
                    messages[i] = {"role": "user", "content": marker}
                    changed_any = True
                    continue
            # Drop the RESULT and its paired assistant turn so the sequence
            # stays alternating.
            drop = {i}
            if i > 0 and messages[i - 1]["role"] == "assistant":
                drop.add(i - 1)
            elif i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                drop.add(i + 1)
            messages[:] = [m for k, m in enumerate(messages) if k not in drop]
        changed_any = True
    if changed_any:
        print("[conversation trimmed to fit the model's context window]",
              file=sys.stderr)
    return changed_any


class Agent:
    """Runs one task against an Index and a model client: gather source,
    generate an answer or edits, and apply those edits."""

    def __init__(self, index, client, cfg):
        """Bind the agent to a project Index, an LLMClient, and a Config."""
        self.index = index
        self.client = client
        self.cfg = cfg
        # The tool dispatcher resolves a `task` subagent's parent through this
        # back-reference (kept out of dispatch()'s call signature so existing
        # monkeypatched dispatchers remain compatible).
        try:
            self.cfg._agent = self
        except Exception:
            pass
        # Set by gather(); _repair_source() falls back to it for files no
        # longer on disk.
        self.gathered_source = ""
        # When set, gather() mines "files the task will change" from this text
        # (the clean plan title) instead of the full prompt, so goal prose and
        # prior reports do not become forced must-reads.
        self.task_targets: str | None = None
        # True once a run/browser/screenshot step executed successfully. The
        # outer loop's evidence gate reads this: an attempted-but-failed
        # browser launch is not live evidence.
        self.live_evidence_ok = False
        # WHICH live actions succeeded ("run", "run_test", "browser",
        # "screenshot"). A title asking what the page/canvas SHOWED is not
        # satisfied by a `run` that only started the process, so the gate
        # reads the kinds, not just the flag.
        self.live_evidence_kinds: set[str] = set()
        # Telemetry sink (events.EventLog); bound on first gather.
        self.events = None
        # Gather conversation kept for answer continuation (see answer()).
        self._messages: list[dict] = []
        # Content-addressed evidence store and the read ledger:
        # every displayed span is hashed here; the ledger maps normalised
        # file -> (start, end) spans in display order. Both reset per run.
        self.evidence = EvidenceStore()
        self._read_spans: dict[str, list[tuple[int, int]]] = {}
        # Explicit NEED / implicit repair rehydration rounds used this run
        # (telemetry; reset in gather).
        self.rehydration_rounds = 0
        # Requirement evidence state machine and the diagnostics whose
        # exact spans the answer packer must pin. Both reset per task.
        self.requirements = None
        self.active_diagnostics: dict = {}
        self._lossy = False
        self.fold_count = 0
        self.answer_mode_used = "rebuild"
        self._gather_task = ""
        self._rules_shown: set[str] = set()
        # SEARCH blocks that matched nowhere in the file, over every apply()
        # round of this agent (the eval row reports it next to repair calls).
        self.search_mismatches = 0

    # ------------------------------------------------------------ helpers
    def _phase_temp(self, setting: str, client=None):
        """Context manager that samples at `cfg.<setting>`
        (temperature_answer / temperature_repair) for the calls inside it.
        A no-op when the setting is None (the global temperature applies),
        when the client cannot override (replay), or when an outer override is
        already active (best_of's higher temperature for its extra candidates
        must not be reset to the answer value)."""
        client = self.client if client is None else client
        cfg = getattr(client, "cfg", None) or self.cfg
        value = getattr(cfg, setting, None)
        if value is None or not hasattr(client, "temperature") \
                or getattr(client, "_temp_override", None) is not None:
            return contextlib.nullcontext()
        return client.temperature(float(value))

    def _chat_at(self, setting: str, messages, client=None, **kw):
        """`client.chat(messages, **kw)` at the phase temperature `setting`."""
        client = self.client if client is None else client
        with self._phase_temp(setting, client):
            return client.chat(messages, **kw)

    def _set_site(self, site: str, client=None) -> None:
        """Label the client's current call site for usage accounting."""
        try:
            (self.client if client is None else client).site = site
        except Exception:
            pass

    def _gather_client(self):
        """The client the gather loop runs its steps on (scout routing):
        `cfg.scout_profile`'s client when configured and reachable, else the
        active architect client. Resolution failures fall back softly."""
        if not str(getattr(self.cfg, "scout_profile", "") or "").strip():
            return self.client
        from . import roles as roles_mod
        try:
            return roles_mod.scout_client(self.cfg, self.client)
        except Exception:
            return self.client

    def _map(self, task: str) -> tuple[str, str]:
        """(map text, tier) for `task`; index doubles that predate the
        ranked map (`map_for_prompt(cfg)`) still work."""
        try:
            return self.index.map_for_prompt(self.cfg, task=task)
        except TypeError:
            return self.index.map_for_prompt(self.cfg)

    def _fewshot_shots(self, read_lines: int = 40) -> list[dict]:
        """Three worked steps from THIS project (a real file, a real symbol):
        grep it, read its definition (at most `read_lines` lines), done. []
        when the project has no code."""
        try:
            for f in self.index.code_files:
                tops = self.index.top_level(f)
                if tops:
                    unit = tops[0]
                    return fewshot_actions(
                        unit.file, unit.qualname.split(".")[0], unit.start,
                        min(unit.scope_end or unit.end,
                            unit.start + read_lines))
        except Exception:
            pass
        return []

    def _prime_fewshot(self, client=None) -> None:
        """Worked step examples from THIS project for the client to show when
        the server cannot enforce a JSON grammar: a small model copies a
        concrete example far better than a dummy `{"action": "read", "file":
        ""}`. With `fewshot_steps` on, gather puts the same examples in the
        first user turn instead, so the client falls back to its one dummy
        example rather than repeating them."""
        client = self.client if client is None else client
        if not hasattr(client, "fewshot"):
            return
        client.fewshot = [] if fewshot_enabled(self.cfg) \
            else self._fewshot_shots()

    def _lead_notes(self) -> str:
        """Per-task orientation placed above the map: repo brief and git state."""
        bits: list[str] = []
        if getattr(self.cfg, "repo_brief", True):
            try:
                from .brief import build_brief
                budget = int(getattr(self.cfg, "flat_map_token_budget", 4000))
                brief = build_brief(
                    self.index, self.cfg,
                    max_tokens=min(int(getattr(self.cfg, "max_brief_tokens", 500)),
                                   max(80, budget // 5)))
                if brief:
                    bits.append(brief)
            except Exception:
                pass
        try:
            from . import design as design_mod
            design_brief = design_mod.brief(self.index.root)
            if design_brief:
                bits.append(design_brief)
        except Exception:
            pass
        try:
            git = gitinfo.preamble(self.index.root, self.cfg)
        except Exception:
            git = ""
        if git:
            bits.append(git)
        return "\n".join(bits)

    def _emit(self, kind: str, **fields) -> None:
        if self.events is not None:
            self.events.emit(kind, **fields)

    def _hook(self, event: str, **fields):
        """Fire a user hook (hooks.py) when one is configured for `event`."""
        if not hooks_mod.has_hooks(self.cfg, event):
            return hooks_mod.HookResult()
        return hooks_mod.fire(self.cfg, self.index.root, event, **fields)

    def _cap_observation(self, action: str, obs: str) -> str:
        """Per-action ceiling for everything but source reads: one broad grep,
        a chatty command or a big MCP reply must not eat the gather budget or
        the tail of the window. Cut on a line boundary, say how much was left
        and how to narrow it."""
        if action in ("read", "context", "done", "todo") \
                or obs.startswith("ERROR"):
            return obs
        tokens = int(getattr(self.cfg, "max_observation_tokens", 0) or 0)
        if tokens <= 0:
            # auto: a quarter of the usable window, between 300 and 2,200 tokens
            tokens = max(300, min(2200, self._room() // 4))
        cap = int(tokens * 3.6)
        if len(obs) <= cap:
            return obs
        cut = obs[:cap]
        nl = cut.rfind("\n")
        if nl > cap // 2:
            cut = cut[:nl]
        left = obs.count("\n") - cut.count("\n")
        return (cut + f"\n[output capped at ~{tokens} tokens; {max(left, 1)} "
                "more line(s) not shown -- narrow the query (a file, a "
                "tighter pattern) to see them]")

    def _read_budget_chars(self) -> float:
        """Char room for cfg.max_read_tokens on one read/context observation:
        the server's calibrated tokenizer when one answered, ~3.5
        chars/token otherwise. 0.0 = budget off."""
        cap = int(getattr(self.cfg, "max_read_tokens", 0) or 0)
        if cap <= 0:
            return 0.0
        cpt = float(getattr(self.client, "chars_per_token", 0.0) or 0.0)
        if cpt <= 0:
            cpt = _READ_CHARS_PER_TOKEN
        return cap * cpt

    def _clip_read_to_token_budget(self, obs: str) -> str:
        """Enforce the read token budget on a `read`/`context` observation
        BEFORE it is displayed and before the span ledger records it.

        Truncates strictly at whole-line boundaries, rewrites the header to
        the range actually delivered, and appends a next_start hint -- so the
        ledger credits (start, actual_end): lines clipped away are never
        falsely recorded as read, and an edit quoting them stays flagged as
        quoting unread source."""
        budget = self._read_budget_chars()
        if not budget or len(obs) <= budget:
            return obs
        m = _READ_HEAD_RX.match(obs)
        head, body = _split_read_obs(obs)
        start = int(m.group(2)) if m else 0
        body_lines = body.split("\n")
        kept: list[str] = []
        used = len(head) + 1
        shown_end = start - 1        # last numbered line that survived
        for ln in body_lines:
            # Truncate strictly at whole-line boundaries: stop on the first
            # line that no longer fits -- but never leave a numbered read
            # with zero source lines (keep going until the first numbered
            # line is kept).
            if (used + len(ln) + 1 > budget and kept
                    and (not m or shown_end >= start)):
                break
            used += len(ln) + 1
            kept.append(ln)
            pm = _NUMBERED_LINE_RX.match(ln)
            if pm:
                shown_end = max(shown_end, int(pm.group(1)))
        if len(kept) == len(body_lines):
            return obs               # odd or short body: nothing was clipped
        if not m or shown_end < start:
            # Not a numbered read (e.g. `context` for a whole file): the
            # ledger never saw this shape, so a line-boundary cut suffices.
            return (head + "\n" + "\n".join(kept) +
                    "\n[CLIPPED due to token budget; use a narrower range "
                    "to see the rest.]")
        return (f"{m.group(1).strip()} lines {start}-{shown_end}:" +
                head[m.end():] + "\n" + "\n".join(kept) +
                f"\n[CLIPPED at line {shown_end} due to token budget. Use "
                f"next_start={shown_end + 1} to continue reading.]")

    # ------------------------------------------- evidence & exact-span rehydration
    def _record_evidence(self, obs: str, step: int) -> EvidenceSpan | None:
        """Hash and record the source lines a delivered read/context
        observation actually shows (post clip), into `self.evidence`.
        Observations with no numbered source -- outline cards, ERROR lines,
        duplicate-range notes -- record nothing."""
        span = self.evidence.record_displayed(obs, step)
        if span is not None:
            self._emit("evidence", file=span.file, start=span.start,
                       end=span.end, sha=span.content_sha[:12],
                       step=span.read_step)
        return span

    def _collapse_repair_turns(self, messages: list[dict], prefix_index: int,
                               *, receipt: str = "",
                               clean_reply: str = "",
                               reason: str = "") -> int:
        """Pop failed drafts/diagnostics back to snapshot `prefix_index`.

        The snapshot is taken immediately before asking for a hunk or entering
        a repair loop. A successful repair leaves a clean state in the same
        list -- a compact receipt (incremental mode) or the accepted reply --
        so earlier cached bytes are reused instead of re-prefilling a broken
        draft. An exhausted repair never calls this helper, keeping its final
        failure visible to telemetry and shallow failure memory.

        `reason` names what was wrong with the drafts being dropped. A run
        that collapsed ten times in 137 seconds told its caller nothing: the
        event carried a message count, and the failed block text lived only
        in the transcript the collapse had just deleted. One line makes the
        churn readable afterwards."""
        if not bool(getattr(self.cfg, "collapse_repair_turns", True)):
            return 0
        try:
            prefix_index = int(prefix_index)
        except (TypeError, ValueError):
            return 0
        if prefix_index < 0 or prefix_index > len(messages):
            return 0
        removed = len(messages) - prefix_index
        if removed <= 0:
            return 0
        del messages[prefix_index:]
        if receipt:
            messages.append({"role": "user", "content": receipt})
        elif clean_reply:
            messages.append({"role": "assistant", "content": clean_reply})
        self._emit("repair_collapse", messages=removed, prefix=prefix_index,
                   reason=(reason or "").replace("\n", " ")[:160])
        return removed

    @staticmethod
    def _span_matches_requirement(span: EvidenceSpan, req) -> bool:
        """Whether one evidence span names or overlaps a requirement."""
        rel = _norm_key(span.file)
        text = f"{span.file}\n{span.exact_text}".lower()
        for term in getattr(req, "terms", ()) or ():
            term = str(term or "").strip().lower()
            if term and term in text:
                return True
        for a_rel, a, b in (list(getattr(req, "anchors", ()) or ())
                            + list(getattr(req, "read_spans", ()) or ())):
            if _norm_key(a_rel) != rel:
                continue
            if b and not (int(b) < int(span.start)
                          or int(a) > int(span.end)):
                return True
            if not b and int(span.start) <= int(a) <= int(span.end):
                return True
        return False

    def _span_utility(self, span: EvidenceSpan, parent_reqs, child_reqs,
                      pinned: set[tuple]) -> float:
        """Deterministic utility rank for a subagent's read span."""
        score = 0.0
        for req in parent_reqs:
            if self._span_matches_requirement(span, req):
                score += 100.0
                score += 5.0 * min(len(getattr(req, "terms", ()) or ()), 8)
        for req in child_reqs:
            if self._span_matches_requirement(span, req):
                score += 10.0
        if (_norm_key(span.file), int(span.start), int(span.end)) in pinned:
            score += 25.0
        # A larger exact span carries more reusable source, but never let size
        # outrank a direct requirement match. Newer reads win ties.
        score += min(max(0, int(span.end) - int(span.start) + 1), 120) * 0.1
        score += max(0, int(getattr(span, "read_step", 0) or 0)) * 0.01
        return score

    def select_subagent_evidence(self, child, limit: int = 2
                                 ) -> list[EvidenceSpan]:
        """The 1-2 highest-utility spans the child actually read and pinned.

        Requirement-matching spans win, then pinned child evidence, then span
        size and recency. Identical file/range/hash records are de-duplicated
        so the parent tail never repeats the same source twice.
        """
        store = getattr(child, "evidence", None)
        spans = list(getattr(store, "spans", ()) or ())
        if not spans:
            return []
        parent_reqs = list(getattr(getattr(self, "requirements", None),
                                  "items", ()) or ())
        child_tracker = getattr(child, "requirements", None)
        child_reqs = list(getattr(child_tracker, "items", ()) or ())
        pinned = {(_norm_key(rel), int(a), int(b))
                  for rel, a, b in (getattr(child_tracker, "pinned_spans",
                                            lambda: [])() or [])}
        ranked: list[tuple[float, int, EvidenceSpan]] = []
        seen: set[tuple] = set()
        for order, span in enumerate(spans):
            if (span is None or not getattr(span, "exact_text", None)
                    or not hasattr(span, "file")):
                continue
            key = (_norm_key(span.file), int(span.start), int(span.end),
                   getattr(span, "content_sha", ""))
            if key in seen:
                continue
            seen.add(key)
            ranked.append((self._span_utility(span, parent_reqs, child_reqs,
                                              pinned), order, span))
        ranked.sort(key=lambda row: (-row[0], -row[1]))
        return [span for _score, _order, span in ranked[:max(1, int(limit))]]

    def absorb_subagent_evidence(self, child, limit: int = 2
                                 ) -> list[EvidenceSpan]:
        """Copy a child's best spans into this agent's evidence + read ledger.

        This is the read-only handoff that lets a parent edit lines the helper
        displayed without paying for a second `read`: the spans are hashed,
        credited to the same ledger `check_unread` consults, and used to move
        matching requirement clauses to LOCATED/READ.
        """
        selected = self.select_subagent_evidence(child, limit=limit)
        copied: list[EvidenceSpan] = []
        if not isinstance(getattr(self, "_read_spans", None), dict):
            self._read_spans = {}
        for span in selected:
            copy = self.evidence.record(span.file, span.start, span.end,
                                        span.exact_text, span.read_step)
            copied.append(copy)
            key = _norm_key(copy.file)
            ledger = self._read_spans.setdefault(key, [])
            if (copy.start, copy.end) not in ledger:
                ledger.append((copy.start, copy.end))
            tracker = getattr(self, "requirements", None)
            if tracker is not None and getattr(tracker, "task", ""):
                try:
                    tracker.observe_search(
                        "read", {"file": copy.file},
                        f"{copy.file}\n{copy.exact_text}", self.index)
                    tracker.observe_read(
                        copy.file, copy.start, copy.end,
                        f"{copy.file}\n{copy.exact_text}", self.index)
                except Exception:
                    pass
            self._emit("subagent_evidence", file=copy.file,
                       start=copy.start, end=copy.end,
                       sha=copy.content_sha[:12])
        return copied

    def _render_subagent_evidence(self,
                                  spans: list[EvidenceSpan]) -> str:
        """Numbered, token-bounded evidence block appended under the digest."""
        if not spans:
            return ""
        budget = self._read_budget_chars()
        used = 0
        chunks: list[str] = []
        for span in spans:
            lines = str(span.exact_text or "").split("\n")
            if not lines or not lines[0]:
                continue
            width = max(1, len(str(int(span.start) + len(lines) - 1)))
            header = (f"{span.file} lines {int(span.start)}-"
                      f"{int(span.start) + len(lines) - 1}:")
            kept: list[str] = []
            body_used = 0
            for offset, line in enumerate(lines):
                rendered = f"{int(span.start) + offset:>{width}} | {line}"
                extra = len(rendered) + 1
                if budget and kept and used + len(header) + 1 + body_used \
                        + extra > budget:
                    break
                kept.append(rendered)
                body_used += extra
            if not kept:
                break
            if budget and used + len(header) + 1 + body_used > budget \
                    and chunks:
                break
            end = int(span.start) + len(kept) - 1
            header = f"{span.file} lines {int(span.start)}-{end}:"
            chunk = header + "\n" + "\n".join(kept)
            chunks.append(chunk)
            used += len(chunk) + 1
            if budget and used >= budget:
                break
        if not chunks:
            return ""
        return ("\n\nEXACT EVIDENCE ALREADY READ BY THE HELPER (numbered like "
                "a `read`, hashed into this run's evidence store, and credited "
                "to the read ledger -- use these lines directly; do not pay "
                "for a second `read` of the same span):\n\n"
                + "\n\n".join(chunks))

    def append_subagent_evidence(self, digest: str,
                                 spans: list[EvidenceSpan]) -> str:
        """Append the promoted spans beneath a child digest for RESULT:."""
        block = self._render_subagent_evidence(spans)
        return ((digest or "").rstrip() + block) if block else (digest or "")

    def _rehydration_rounds_left(self) -> int:
        """Rounds the explicit NEED rehydration may still run (Guiding
        Constraint #3): cfg.max_rehydration_rounds, default 2; 0 disables."""
        return max(0, int(getattr(self.cfg, "max_rehydration_rounds",
                                  _MAX_REHYDRATION_ROUNDS) or 0))

    def _fetch_exact_span(self, name: str, start: int, end: int) -> str:
        """Rehydration's single fetch point: lines `start`-`end` read FRESH
        from disk, rendered exactly like a `read` observation, clipped to the
        same `cfg.max_read_tokens` budget, hashed into the evidence store and
        credited to the read ledger. Returns the rendered text (with a
        hash-compare note when the file moved since a span covered by the
        same range was recorded), or "" when the file/range is not on disk."""
        rel = self.index.resolve_file(name) or fileio.normalise_rel(name)
        path = self.index.root / rel
        if not rel or not path.is_file():
            return ""
        try:
            lines = fileio.read_lines(path)
        except OSError:
            return ""
        if lines and lines[-1] == "" and len(lines) > 1:
            lines = lines[:-1]             # same convention as tool_read
        s = max(1, int(start))
        e = min(max(int(end), s), len(lines))
        if not lines or s > len(lines):
            return ""
        width = max(1, len(str(e)))
        body = "\n".join(f"{i:>{width}} | {lines[i - 1]}"
                         for i in range(s, e + 1))
        obs = self._clip_read_to_token_budget(f"{rel} lines {s}-{e}:\n{body}")
        # The span that LAST showed these lines: an exact-range record
        # compares whole, a wider covering record contributes just the
        # s-e slice of its stored text (else every repair re-fetch would
        # "change" against a superset hash that never matched to begin with).
        covers = self.evidence.covering(rel, s, e)
        prior = covers[-1] if covers and covers[-1].exact_text else None
        prior_sha = ""
        if prior is not None:
            if (prior.start, prior.end) == (s, e):
                prior_sha = prior.content_sha
            else:
                seg = "\n".join(prior.exact_text.split("\n")
                                [s - prior.start:e - prior.start + 1])
                prior_sha = sha256_text(seg)
        span = self._record_evidence(obs, _READ_STEP_REFETCH)
        if span is None:
            span = self.evidence.record(rel, s, e, "", _READ_STEP_REFETCH)
        key = _norm_key(rel)
        if not isinstance(getattr(self, "_read_spans", None), dict):
            self._read_spans = {}
        self._read_spans.setdefault(key, []).append((span.start, span.end))
        if prior and prior_sha != span.content_sha:
            obs += (f"\n[content changed since it was last shown at step "
                    f"{prior.read_step}: was {prior_sha[:12]}..., "
                    f"now {span.content_sha[:12]}...]")
        elif prior:
            obs += f"\n[unchanged since step {prior.read_step}]"
        return obs

    def _chat_needs(self, chat, msgs: list[dict]) -> str:
        """Generate with the explicit `NEED path/to/file.py:120-168`
        directive honoured (explicit): one `chat(msgs)` pass, and
        while the reply's prose carries NEED lines and rehydration rounds
        remain, fetch ONLY those exact spans and APPEND them as a fresh
        assistant+user turn pair at the tail -- history is never rewritten,
        so the server's prefix cache still covers every earlier byte -- then
        re-ask. Bounded by cfg.max_rehydration_rounds (default 2, 0 = off).
        `chat` raises LLMError on failure; callers keep their own handling."""
        from . import verify as verify_mod
        reply = chat(msgs)
        rounds = self._rehydration_rounds_left()
        while rounds:
            wants = verify_mod.find_needs(reply or "")
            if not wants:
                break
            rounds -= 1
            fetched = []
            for name, a, b in wants[:_MAX_NEEDS_PER_ROUND]:
                obs = self._fetch_exact_span(name, a, b)
                if obs:
                    fetched.append(obs)
            self.rehydration_rounds += 1
            self._emit("rehydrate", mode="needs", asked=len(wants),
                       fetched=len(fetched), round=self.rehydration_rounds)
            msgs.append({"role": "assistant", "content": reply})
            if fetched:
                msgs.append({"role": "user", "content":
                    "EXACT SOURCE FETCHED FOR YOUR `NEED` LINES (re-read from "
                    "disk just now, numbered like a `read`; copy SEARCH text "
                    "from it byte for byte, ignoring the line-number "
                    "prefixes). Continue the task now. Do not output a NEED "
                    "line for a span shown here.\n\n"
                    + "\n\n".join(fetched)})
            else:
                msgs.append({"role": "user", "content":
                    "None of those `NEED` ranges could be fetched (the file "
                    "or the lines are not on disk). Answer from the source "
                    "you have already seen and say what you could not "
                    "verify."})
            reply = chat(msgs)
        return reply

    def _auto_rehydrate(self, findings) -> str:
        """Implicit repair auto-rehydration: for unread-region
        and fuzzy-match findings that touch recorded evidence, re-fetch the
        flagged spans fresh from disk and return a block to append at the
        TAIL of the repair prompt. The read ledger is credited, so a
        corrected reply quoting the rehydrated text no longer reads as a
        guess. "" when no finding touches evidence."""
        from . import verify as verify_mod
        targets = verify_mod.evidence_rehydrations(self, findings)
        chunks = []
        for rel, a, b in targets:
            obs = self._fetch_exact_span(rel, a, b)
            if obs:
                chunks.append(obs)
        if not chunks:
            return ""
        self.rehydration_rounds += 1
        self._emit("rehydrate", mode="auto", spans=len(chunks),
                   round=self.rehydration_rounds)
        return ("\n\nFRESH SOURCE FOR THE FLAGGED REGIONS (re-read from disk "
                "after the check flagged them; numbered like a `read` -- "
                "copy SEARCH text from here, byte for byte, ignoring the "
                "line-number prefixes):\n\n"
                + "\n\n".join(chunks))[:_MAX_REHYDRATED_CHARS]

    def _ask_user(self, text: str, asked: int) -> str:
        """The `question` action: relay one question to the user (bounded per
        task); an empty answer tells the model to choose the safest reading."""
        ask = getattr(self.cfg, "_ask_user", None)
        if ask is None or not text.strip():
            return ("ERROR: no user is available to answer. State the "
                    "ambiguity and choose the safest interpretation.")
        if asked >= steering_mod.MAX_QUESTIONS:
            return ("ERROR: you already asked the maximum number of "
                    "questions. Decide, and state your assumption.")
        answer = (ask(text.strip()[:600]) or "").strip()
        self._emit("question", asked=asked + 1, answered=bool(answer))
        if not answer:
            return ("USER ANSWER: (none) -- choose the safest interpretation "
                    "and say what you assumed.")
        return f"USER ANSWER: {answer[:500]}"

    def _scoped_rules(self, rel: str) -> str:
        """Path-scoped rules (`.pedacito/rules/*.md`, nested AGENTS.md) for a
        file the model just read, once per rule per task."""
        if not getattr(self.cfg, "use_memory", True):
            return ""
        try:
            found = memory_mod.rules_for(self.index.root, rel)
        except Exception:
            return ""
        out = []
        for src, text in found:
            if src in self._rules_shown or not text.strip():
                continue
            self._rules_shown.add(src)
            out.append(f"\n[project rule from {src} -- applies to this "
                       f"file; a hint, verify against source]\n{text}")
        return "".join(out)

    # ----------------------------------------------------- window pressure
    def _room(self) -> int:
        cfg = self.cfg
        return max(cfg.context_tokens - cfg.max_tokens_step
                   - cfg.reserve_tokens // 2, 512)

    def _fit(self, messages: list[dict], client=None) -> bool:
        """Keep the live conversation inside the server's window. Only acts on
        REAL overflow: it folds the older band into one STATE digest (one
        cache break; see fold.py), and falls back to the legacy per-turn clip
        only if the recent tail alone is still too big. Returns True when it
        changed anything."""
        cfg = self.cfg
        client = self.client if client is None else client
        est = getattr(client, "estimate_tokens", None) or estimate_tokens
        room = self._room()
        if estimate_messages(messages, est) <= room:
            return False
        changed = False
        if getattr(cfg, "compaction", "deterministic") != "off":
            keep = fold_mod.tail_pairs_for(cfg)
            max_chars = max(500, min(2500, int(room * 3.6 * 0.10)))
            while True:
                before = estimate_messages(messages, est)
                if fold_mod.fold_middle(messages, keep, max_chars):
                    self._after_fold(messages, est, before, client)
                    changed = True
                if estimate_messages(messages, est) <= room or keep <= 1:
                    break
                keep -= 1
        if estimate_messages(messages, est) > room:
            if _evict_overflow_turns(messages, client, cfg):
                self._lossy = True
                changed = True
        return changed

    def _after_fold(self, messages: list[dict], est, before: int,
                    client=None) -> None:
        """Bookkeeping (and the optional model-written STATE) after a fold."""
        self.fold_count += 1
        self._lossy = True
        if getattr(self.cfg, "compaction", "") == "model" and len(messages) > 3:
            state = messages[3]["content"] if messages[3]["role"] == "user" \
                else ""
            better = self._model_state(state, client)
            if better:
                messages[3] = {"role": "user", "content": better}
        self._emit("fold", n=self.fold_count, tokens_before=before,
                   tokens_after=estimate_messages(messages, est))
        self._hook("post_compact", folds=self.fold_count)
        print("[older lookups folded into a STATE digest to fit the model's "
              "context window]", file=sys.stderr)

    def _model_state(self, state_msg: str, client=None) -> str:
        """One small call that rewrites the extractive STATE into goal /
        findings / next-step lines (cfg.compaction = "model"); '' on any
        failure so the deterministic digest stays."""
        from . import fold as fold_mod_
        client = self.client if client is None else client
        body = state_msg[len(fold_mod_.STATE_HEAD):].strip()
        if not body:
            return ""
        try:
            self._set_site("compaction", client)
            text = client.chat(
                [{"role": "system", "content":
                  "You compress a coding assistant's lookup log. Reply with at "
                  "most 8 short lines: GOAL, FOUND (files and line ranges that "
                  "matter), RULED OUT, NEXT. Keep file paths and numbers exact. "
                  "No preamble."},
                 {"role": "user", "content":
                  f"TASK\n{self._gather_task[:600]}\n\nLOOKUP LOG\n{body}"}],
                max_tokens=int(getattr(self.cfg, "max_tokens_summary", 220)) + 120)
        except Exception:
            return ""
        finally:
            self._set_site("gather", client)
        text = (text or "").strip()
        return (fold_mod_.STATE_HEAD + "\n" + text[:1600]) if text else ""

    # ------------------------------------------------------------------ phase 1
    def gather(self, task: str, verbose: bool = True,
               on_step=None, base: str | None = None,
               backup_dir=None,
               prefix_messages: list[dict] | None = None) -> str:
        """Run the tool-calling loop until the model says `done`, the
        gathered-source budget is exhausted, or cfg.max_steps (0 = no limit)
        is reached.

        Returns the concatenated text of everything read, ready for `answer()`.
        Also sets `self.read_any` (whether any `read` succeeded) and
        `self.steps_log` (one line per step, used for the session log).

        `on_step`, when given, is called with the step number after every
        completed step (the GUI uses it for live progress).

        `base` replaces `GATHER_BASE` as the system prompt preamble (the map
        tier hints are still appended); `explain` uses it.

        `backup_dir` is where gather-phase write/append/move/delete snapshots
        go; when None the snapshot helper makes its own directory under
        `.pedacito/backups/`.

        `self.task_targets`, when set, replaces `task` as the text mined for
        the forced-read coverage check.
        """
        # A fresh run: drop any backup directory left by a previous task.
        self.cfg._backup_dir = backup_dir
        root = self.index.root
        self.cfg._root = str(root)
        self.cfg._is_git = gitinfo.is_repo(root)
        # Dynamic tool pruning reads the task text for trigger words and the
        # repository's per-tool task counters. A task start bumps the counter;
        # dispatch records actual uses while this gather runs.
        self.cfg._task_text = task
        try:
            memory_mod.begin_tool_task(root, self.cfg)
        except Exception:
            pass
        # Scout/architect split: gather's per-tool lookup steps run on
        # cfg.scout_profile when it names another reachable profile; dispatch,
        # subagents, and the answer phase stay on the architect client.
        gclient = self._gather_client()
        if verbose and gclient is not self.client:
            print(f"  [scout: gather runs on profile "
                  f"'{getattr(self.cfg, 'scout_profile', '')}']",
                  file=sys.stderr)
        if self.events is None:
            self.events = events_mod.get(root, self.cfg)
            events_mod.attach(self.client, self.events)
        if gclient is not self.client:
            events_mod.attach(gclient, self.events)
        self._set_site("gather", gclient)
        self._prime_fewshot(gclient)
        repo_map, tier = self._map(task)
        hint = {"flat": FLAT_HINT, "cards": CARDS_HINT,
                "ranked": RANKED_HINT}.get(tier, DIRS_HINT)
        system = prompts_mod.gather_system(
            self.cfg, base if base is not None else GATHER_BASE, TOOL_HELP) + hint
        project_notes = instructions_mod.render(root, self.cfg)
        todo_notes = todo_mod.render(root, self.cfg,
                                     limit=int(getattr(
                                         self.cfg, "max_todo_chars", 1500)))
        if project_notes:
            system += "\n\n" + project_notes
        memory_notes = memory_mod.render(root, self.cfg)
        if memory_notes:
            system += "\n\n" + memory_notes
        try:
            micro_notes = memory_mod.render_micro_facts(root, self.cfg)
        except Exception:
            micro_notes = ""
        if micro_notes:
            system += "\n\n" + micro_notes
        try:
            lesson_notes = memory_mod.render_anti_patterns(root, self.cfg)
        except Exception:
            lesson_notes = ""
        if lesson_notes:
            system += "\n\n" + lesson_notes
        if "skill" in schema_for(self.cfg)["properties"]["action"]["enum"]:
            skill_notes = skills_mod.catalog(root, self.cfg, task=task)
            if skill_notes:
                system += "\n\n" + skill_notes
        if todo_notes:
            system += "\n\n" + todo_notes
        mcp_notes = (mcp_mod.render(self.cfg)
                     if getattr(self.cfg, "allow_mcp", False) else "")
        # Named-server path: only the handful of the servers'
        # tools that match THIS task, each a compact one-line signature.
        named_notes = mcp_injection(self.cfg)
        mcp_notes = "\n\n".join(x for x in (mcp_notes, named_notes) if x)
        if mcp_notes:
            system += "\n\n" + mcp_notes
        if verbose:
            print(f"  [map: {tier}, ~{estimate_tokens(repo_map)} tokens, "
                  f"{len(self.index.files)} files]", file=sys.stderr)
        # Orientation that changes per task (git state) or per index (the
        # brief) rides in the first user turn, not the system prompt, so the
        # system prompt stays byte-identical across tasks for the server cache.
        lead = self._lead_notes()
        try:
            tried = memory_mod.attempts_note(self.index.root, task, self.cfg)
        except Exception:
            tried = ""
        if tried:
            lead = (lead + "\n" if lead else "") + tried
        examples = ""
        if fewshot_enabled(self.cfg):
            # Once, here, never appended to a later turn: user0 is the cached
            # prefix of every step.
            examples = fewshot_block(
                self._fewshot_shots(FEWSHOT_USER0_READ_LINES),
                schema_for(self.cfg))
        user0 = (lead + "\n\n" if lead else "") \
            + f"PROJECT MAP\n\n{repo_map}\n\n" \
            + (examples + "\n\n" if examples else "") + f"TASK\n{task}"
        if safety.has_secret(task) and is_hosted(self.cfg) \
                and getattr(self.cfg, "redact_secrets", True):
            print("  [warning: the task text looks like it contains a secret "
                  "and goes to a hosted API]", file=sys.stderr)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user0},
        ]
        if prefix_messages:
            # Single-slot local server (TECHNIQUES.md #28): fork the parent's
            # stable cached prefix byte-for-byte instead of sending a fresh
            # system prompt that would evict the architect's KV cache. The
            # helper brief rides at the tail as a normal assistant/user pair.
            messages = [dict(m) for m in prefix_messages]
            messages.append({"role": "assistant", "content":
                             "I will handle this bounded, read-only lookup."})
            messages.append({"role": "user", "content":
                             "HELPER TASK (read-only, bounded research for "
                             "the parent agent)\n" + task +
                             "\n\nRequest only the exact source you need, "
                             "then choose done."})
        self._messages = messages
        self._lossy = False          # True once history was folded/evicted
        self.fold_count = 0
        self._gather_task = task
        self._rules_shown: set[str] = set()
        # Fresh evidence store and rehydration counters for this run, like
        # the ledger and backup dir below.
        self.evidence = EvidenceStore()
        self.rehydration_rounds = 0
        # One requirement tracker for this task. It advances only
        # from tool results and patches, never from an extra model call.
        from . import verify as verify_mod
        self.requirements = verify_mod.RequirementTracker(task)
        self.active_diagnostics = {}
        gathered: list[str] = []
        self.steps_log: list[str] = []
        self.live_evidence_ok = False
        self.live_evidence_kinds = set()
        did_read = False
        read_nudged = False
        # Successful reads only, per normalised file as (start, end) spans in
        # step order. Coverage counts only real source the model saw, not grep
        # hits or ERROR lines; the eviction pass replays it to find which
        # lines each RESULT turn was first to show.
        spans: dict[str, list[tuple[int, int]]] = {}
        self._read_spans = spans
        step_cap = max(0, int(getattr(self.cfg, "max_steps", 0) or 0))
        unlimited = step_cap <= 0
        # Files the task text names but no `read` has covered yet, refreshed
        # after every successful read for the `done` nudge.
        self._targets_text = getattr(self, "task_targets", None) or task
        self._unread_files: list[str] = self._compute_unread(self._targets_text)
        # Inputs for the opt-in write allowlist (cfg.auto_allowlist): the live
        # read ledger (same dict object, so it grows as gather reads) and the
        # files the task title names. Reset per task, like _backup_dir above.
        self.cfg._task_reads = spans
        self.cfg._task_allowlist = self._task_file_targets(self._targets_text)
        # Charged per NEW byte, so re-reading lines already in context is free.
        budget = self.cfg.max_gathered_chars

        def nudge(step: int) -> str:
            """Suffix telling the model how many lookups remain."""
            if unlimited:
                return ("[Next action? No lookup limit -- choose 'done' once "
                        "the task is covered.]")
            remaining = step_cap - step
            return (f"[{remaining} lookups left after this one] Next action?"
                    if remaining else
                    "This is your LAST lookup. Choose 'done' unless you truly need one more.")

        # Nudges ride with the preceding user message: some chat templates
        # reject two user messages in a row.
        messages[-1]["content"] += "\n\n" + nudge(1)

        # Step-0 prefetch (opt-in): read the task-named files/symbols before
        # the first model step. Runs through the normal tool path so the read
        # ledger and SOURCE block are populated; the reads do not count toward
        # max_steps.
        if getattr(self.cfg, "prefetch_targets", 0):
            budget = self._prefetch_step0(task, messages, gathered, spans,
                                          budget, nudge, verbose, gclient)

        step = 0
        step_schema = schema_for(self.cfg)
        done_confirmed = False   # one-shot "are you really done?" after a truncated reply
        step_recoveries = 0      # re-asks after a step died at the output cap
        duplicate_reads = 0      # consecutive reads that added no new lines
        error_streak = 0         # consecutive failed (ERROR) observations
        last_sig = None          # previous step's action+args, for loops
        repeated = 0
        payload_cuts = 0         # steps discarded for a cut-off payload
        # One targeted recovery prompt per distress signal before gather gives
        # up (cfg.stuck_recovery): a small model often escapes a loop when
        # told, specifically, what it is doing.
        recoveries = {"repeat": 0, "errors": 0, "dup": 0}
        questions = 0            # clarifying questions asked so far
        ran: dict = {}           # step signature -> (step, index gen, folds, mutations)
        mutations = 0            # bumps when something that can change results ran
        recover_on = bool(getattr(self.cfg, "stuck_recovery", True))
        # Action-independent global leases: they bound gather even when
        # max_steps = 0 has removed the fixed lookup cap. `model_calls` is a
        # hard ceiling on LLM invocations for this task; `no_progress` counts
        # consecutive gather steps that added no new unread line span to the
        # read ledger and resolved no new symbol. 0 on either turns it off.
        call_cap = max(0, int(getattr(self.cfg, "max_total_model_calls", 0) or 0))
        no_progress_cap = max(0, int(getattr(self.cfg, "max_no_progress_steps", 0) or 0))
        model_calls = 0
        no_progress = 0
        loop_guard_fired = False    # TokenEconomizer r = 1 pre-lease redirect
        resolved_symbols: set[str] = set()

        def harness_note(text: str) -> None:
            """Attach a harness verdict to the pending user turn; two bare
            user messages in a row break some chat templates, so it rides
            with the last one (as the nudges do)."""
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n" + text
            else:
                messages.append({"role": "user", "content": text})

        def note_progress(made_progress: bool) -> bool:
            """Update the no-progress streak; True return means the lease is
            spent and gather must move on to the next phase.

            One step before the hard lease, inject a single high-salience
            redirect (TokenEconomizer's Loop Guard, r = 1): retrying the same
            strategy is the commonest small-model stall, so name a different
            tool before terminating. It fires at most once per turn."""
            nonlocal no_progress, loop_guard_fired
            no_progress = 0 if made_progress else no_progress + 1
            if no_progress_cap <= 0 or no_progress < no_progress_cap:
                if (no_progress_cap > 1 and no_progress == no_progress_cap - 1
                        and not loop_guard_fired):
                    loop_guard_fired = True
                    harness_note(
                        f"[LOOP GUARD: You have taken {no_progress} steps "
                        "without adding new source lines or resolving new "
                        "symbols. Do NOT repeat your previous search/lookup "
                        "strategy. Either try a different tool (e.g. "
                        "outline/usages) on an unread requirement target, or "
                        "emit 'done' now.]")
                    self._emit("recovery", signal="loop_guard", step=step)
                    if verbose:
                        print("  [loop guard: one redirect before the "
                              "no-progress lease; asking for a different "
                              "strategy]", file=sys.stderr)
                return False
            harness_note(f"[Harness: gather stopped after {no_progress} "
                         "consecutive steps that added no unread line span "
                         "to the read ledger and resolved no new symbol. "
                         "Answer from what was already gathered.]")
            self._emit("recovery", signal="no_progress", step=step)
            if verbose:
                print("  [gather stopped: the no-progress lease ran out; "
                      "moving to answer]", file=sys.stderr)
            return True

        # --- deterministic requirement completion gate ---------------------
        # One cheap harness action per unmet requirement (never a model call):
        # read a located anchor, or grep for a specific term. The model sees
        # the appended RESULT and gets the next turn; a stall cleanup variant
        # appends at the tail without re-entering the loop.
        req_assists: set[int] = set()
        self._requirement_assists = req_assists

        def _requirement_action(plan: dict, req, tail: bool) -> bool:
            nonlocal budget, did_read, read_nudged
            act = plan.get("action")
            if act not in ("read", "grep"):
                return False
            payload = {"action": act,
                       "reasoning": "harness requirement assist"}
            if act == "read":
                payload.update(file=plan.get("file", ""),
                               start=plan.get("start", 1),
                               end=plan.get("end", 0))
                detail = f"{payload['file']} L{payload['start']}-{payload['end']}"
            else:
                payload["pattern"] = plan.get("pattern", "")
                detail = str(payload["pattern"])
            try:
                obs = dispatch(self.index, self.cfg, payload,
                               client=self.client)
            except Exception as e:
                obs = (f"ERROR: {act} failed ({type(e).__name__}): "
                       f"{str(e)[:160]}")
            if not obs.startswith("ERROR"):
                obs = safety.frame(act, obs, self.cfg)
                obs = self._cap_observation(act, obs)
            changed = False
            if act == "read":
                if not obs.startswith("ERROR"):
                    obs = self._clip_read_to_token_budget(obs)
                m = _READ_HEAD_RX.match(obs)
                if m:
                    key = _norm_key(m.group(1))
                    s_, e_ = int(m.group(2)), int(m.group(3))
                    prior = list(spans.get(key, []))
                    new_lines = _read_has_new_lines(obs, prior)
                    delta = _delta_len(obs, prior)
                    spans.setdefault(key, []).append((s_, e_))
                    self.requirements.observe_read(
                        m.group(1).strip(), s_, e_, obs, self.index)
                    self._unread_files = self._compute_unread(
                        self._targets_text)
                    if new_lines:
                        did_read = True
                        read_nudged = False
                        self._record_evidence(obs, step)
                        gathered.append(
                            f"--- read {payload.get('file') or m.group(1).strip()}"
                            f"\n{obs}")
                        changed = True
                    if delta > budget:
                        obs = (obs[:budget] +
                               "\n[truncated: context budget reached]")
                        budget = 0
                    else:
                        budget -= delta
                else:
                    budget -= len(obs)
                    self.requirements.observe_search(
                        act, payload, obs, self.index)
            else:
                budget -= len(obs)
                self.requirements.observe_search(act, payload, obs, self.index)
                gathered.append(f"--- {act} {detail}\n{obs}")
                changed = True
            if obs.startswith("ERROR"):
                gathered.append(f"--- {act} {detail} failed: {obs[:160]}")
            self.steps_log.append(
                f"{step}. auto-{act} {detail}  (requirement assist)")
            if tail and messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += (
                    f"\n\n[HARNESS {act.upper()} for an unmet requirement]\n"
                    f"RESULT:\n{obs}")
            else:
                messages.append({"role": "assistant",
                                 "content": _compact(payload)})
                messages.append({"role": "user",
                                 "content": f"RESULT:\n{obs}\n\n{nudge(step + 1)}"})
            self._emit("step", n=step, action=act, detail=detail[:120],
                       obs_chars=len(obs), auto=True)
            return changed or True

        def _requirement_assist(tail: bool = False) -> bool:
            """Run up to two deterministic actions for the oldest unmet
            requirement (grep -> read when the grep located an anchor)."""
            acted = False
            last_idx = None
            last_sig = None
            for _ in range(2):
                if budget <= 0 or not self.requirements:
                    break
                plan, req = self.requirements.next_action(self.index)
                if plan is None or req is None:
                    break
                idx = self.requirements.items.index(req)
                if idx in req_assists:
                    break
                sig = (plan.get("action"), plan.get("file"), plan.get("start"),
                       plan.get("end"), plan.get("pattern"))
                if sig == last_sig:
                    break
                last_sig, last_idx = sig, idx
                _requirement_action(plan, req, tail)
                acted = True
                # A grep that located an anchor earns the immediate read of
                # that exact span; READ/final requirements need no more.
                if (plan.get("action") == "grep"
                        and req.state == verify_mod.REQ_LOCATED
                        and req.anchors):
                    continue
                break
            if acted and last_idx is not None:
                req_assists.add(last_idx)
            return acted

        while unlimited or step < step_cap:
            if call_cap > 0 and model_calls >= call_cap:
                harness_note(f"[Harness: gather stopped -- this task reached "
                             f"its ceiling of {call_cap} model calls. Answer "
                             f"from what was already gathered.]")
                self._emit("recovery", signal="model_calls", step=step)
                if verbose:
                    print(f"  [gather stopped: {model_calls} model calls "
                          f"(cfg.max_total_model_calls); moving to answer]",
                          file=sys.stderr)
                break
            step += 1
            # Steering: lines the user typed mid-run join the turn that has
            # not been sent yet, so the cached prefix is untouched.
            for note in steering_mod.drain(getattr(self.cfg, "_steer", None)):
                messages[-1]["content"] += (
                    f"\n\n[The USER says, mid-task: {note}  -- follow this "
                    "over your earlier plan.]")
                self._emit("steer", step=step, chars=len(note))
                if verbose:
                    print(f"  [steering delivered: {note[:70]}]", file=sys.stderr)
            model_calls += 1
            try:
                stepjson = gclient.chat_json(messages, step_schema,
                                                 self.cfg.max_tokens_step,
                                                 disable_thinking=True)
            except LLMError as e:
                # A step that cannot produce JSON ends gather, not the run:
                # the answer phase works from whatever was gathered. Re-ask
                # a couple of times first so earlier lookups are not wasted.
                msg = str(e).lower()
                if "token limit" in msg or "valid json" in msg:
                    step_recoveries += 1
                    if step_recoveries <= _MAX_GATHER_RECOVERIES:
                        messages.append({"role": "user", "content":
                            "[Your last reply hit the output token limit and "
                            "contained no valid action. Choose your next "
                            "action now: put ONE short phrase in 'reasoning' "
                            "(under 10 words) and fill 'action' plus its "
                            "fields. Reply with the JSON object only.]"})
                        step -= 1        # the failed turn costs no step
                        continue
                    print(f"  [gather step failed: {str(e)[:120]}...]\n"
                          "  [proceeding to the answer phase with what was "
                          "gathered so far]", file=sys.stderr)
                    break
                raise
            if not isinstance(stepjson, dict):
                print(f"  [gather step was {type(stepjson).__name__}, not an "
                      f"action object; moving to answer]", file=sys.stderr)
                break
            action = str(stepjson.get("action") or "").strip()
            why = str(stepjson.get("reasoning") or "").strip()
            sig = _step_signature(stepjson)
            is_repeat = sig == last_sig
            if is_repeat:
                repeated += 1
            else:
                last_sig = sig
                repeated = 1

            # Never dispatch a step whose payload arrived cut off: either the
            # reply hit the token limit and the salvage kept half a field
            # (client.truncated_field), or constrained sampling closed a string
            # at exactly its schema maxLength. Discard without a live-step log
            # line (it would falsely satisfy the live-evidence gate) and ask
            # for a shorter or split payload.
            cut = ""
            note = ""
            if getattr(gclient, "step_truncated", False):
                tf = getattr(gclient, "truncated_field", "")
                if tf and tf != "reasoning":
                    cut = tf
                    note = (f"your `{tf}` value was cut off by the reply "
                            f"token limit before it finished")
            if not cut:
                props = (step_schema or {}).get("properties", {})
                for tf, val in stepjson.items():
                    if tf == "reasoning" or not isinstance(val, str):
                        continue
                    cap = int((props.get(tf) or {}).get("maxLength") or 0)
                    if cap and len(val) >= cap:
                        cut = tf
                        note = (f"your `{tf}` value arrived at exactly {cap} "
                                f"chars -- the schema length cap closed it "
                                f"mid-payload")
                        break
            if cut:
                payload_cuts += 1
                self.steps_log.append(
                    f"{step}. (discarded: `{cut}` payload cut off; NOT "
                    f"executed)")
                obs = (f"[Step NOT executed: {note}, so running it would "
                       f"send half a payload. Send ONE shorter step now: "
                       f"minify it (short names, no comments or extra "
                       f"spaces), or split the work across two calls"
                       + ("; build large file content with `write` then "
                          "`append` chunks" if cut == "content" else "")
                       + ".]")
                messages.append({"role": "assistant",
                                 "content": _compact(stepjson)})
                messages.append({"role": "user",
                                 "content": f"{obs}\n\n{nudge(step + 1)}"})
                budget -= len(obs)
                self._fit(messages, gclient)
                if verbose:
                    print(f"  [step {step} discarded: `{cut}` payload cut "
                          f"off ({payload_cuts}/{_MAX_PAYLOAD_CUTS}); NOT "
                          f"executed]", file=sys.stderr)
                if payload_cuts >= _MAX_PAYLOAD_CUTS or budget <= 0:
                    if verbose:
                        print("  [gather stopped: payloads keep arriving "
                              "cut off; moving to answer]", file=sys.stderr)
                    break
                # A discarded payload made no retrieval progress.
                if note_progress(False):
                    break
                continue

            detail = (stepjson.get("file") or stepjson.get("pattern")
                      or stepjson.get("symbol") or stepjson.get("command") or "")
            if action == "read":
                detail = f"{stepjson.get('file')} L{stepjson.get('start')}-{stepjson.get('end')}"
            elif action == "move":
                detail = f"{stepjson.get('file')} -> {stepjson.get('to')}"
            elif action == "run":
                detail = stepjson.get("command", "")
            elif action == "run_test":
                detail = ((stepjson.get("target")
                           or stepjson.get("file") or "(whole suite)")
                          + (f" -k {stepjson.get('filter')}"
                             if stepjson.get("filter") else ""))
            elif action == "browser":
                detail = ((stepjson.get("command") or "")
                          + (f" [{stepjson.get('pattern')}]"
                             if stepjson.get("pattern") else ""))
            self.steps_log.append(f"{step}. {action} {detail}  ({why})")
            if on_step is not None:
                try:
                    on_step(step)
                except Exception:
                    pass   # progress reporting must never break the loop
            if verbose:
                print(f"  step {step}: {action} {detail}  ({why[:80]})", file=sys.stderr)

            if action == "done":
                missing = list(self._unread_files)
                at_limit = (not unlimited and step >= step_cap)
                # A `done` salvaged from a reply cut off at the token limit may
                # be a truncation artifact; ask once to confirm.
                if getattr(gclient, "step_truncated", False) \
                        and not done_confirmed and not at_limit:
                    done_confirmed = True
                    messages.append({"role": "assistant",
                                     "content": _compact(stepjson)})
                    messages.append({"role": "user", "content":
                        "[Your previous reply hit the output token limit "
                        "before it finished. If you truly have everything "
                        "you need, say `done` again with 'reasoning' under 5 "
                        "words. Otherwise take ONE more lookup.]"})
                    if verbose:
                        print("  (done after a truncated reply; asking once "
                              "to confirm)", file=sys.stderr)
                    continue
                if missing and not at_limit and not read_nudged:
                    # The task names files never `read`; the answer phase
                    # cannot quote unread code. Nudge once per round of
                    # progress, then accept `done`.
                    listing = "\n".join(f"- {f}" for f in missing[:6])
                    read_nudged = True
                    messages.append({"role": "assistant", "content": _compact(stepjson)})
                    messages.append({"role": "user", "content":
                        "Your task will change these files, but you have not "
                        "`read` them yet:\n" + listing + "\n"
                        "Edits must quote a file's text exactly, so you cannot "
                        "write them blind. `read` each one above (start at line "
                        "1; the reply tells you how many lines it has), then "
                        "say done."})
                    if verbose:
                        print("  (files the task needs were never read; asking "
                              "it to read them before answering)", file=sys.stderr)
                    continue
                if not did_read and not at_limit and not read_nudged:
                    # Grep output is trimmed, so edits cannot anchor on it.
                    read_nudged = True
                    messages.append({"role": "assistant", "content": _compact(stepjson)})
                    messages.append({"role": "user", "content":
                        "You have not read any source yet, only located it. Edits "
                        "must quote the file exactly, and grep output is trimmed. "
                        "`read` the lines you intend to change, then say done."})
                    if verbose:
                        print("  (no source read yet; asking it to read before "
                              "answering)", file=sys.stderr)
                    continue
                # Before accepting `done`, make sure every
                # requirement has at least LOCATED/READ evidence. The cheapest
                # action is deterministic: read a located anchor, else grep a
                # specific requirement term. One assist per requirement. The
                # gate is for multi-requirement tasks; a single clause already
                # has the read nudge / prefetch path and must not turn an
                # opt-out into a surprise lookup.
                if (not at_limit and budget > 0 and self.requirements
                        and len(self.requirements) >= 2):
                    req = self.requirements.next_unmet()
                    idx = (self.requirements.items.index(req)
                           if req is not None else -1)
                    if req is not None and idx not in req_assists:
                        messages.append({"role": "assistant",
                                         "content": _compact(stepjson)})
                        messages.append({"role": "user", "content":
                            self.requirements.prompt(req)
                            + "\n[Harness: taking the cheapest lookup for it "
                              "now.]"})
                        if _requirement_assist():
                            if verbose:
                                print("  (a requirement was still "
                                      f"{req.state}; the harness fetched the "
                                      "cheapest lookup before accepting done)",
                                      file=sys.stderr)
                            continue
                break

            if action == "question":
                obs = self._ask_user(str(stepjson.get("content") or ""),
                                     questions)
                questions += 1
                messages.append({"role": "assistant",
                                 "content": _compact(stepjson)})
                messages.append({"role": "user",
                                 "content": f"RESULT:\n{obs}\n\n{nudge(step + 1)}"})
                self.steps_log[-1] += "  [asked the user]"
                if note_progress(False):
                    break
                continue

            if repeated >= _MAX_REPEATED_STEPS:
                if recover_on and recoveries["repeat"] < 1:
                    recoveries["repeat"] += 1
                    note = (f"[STUCK: you have sent the identical `{action}` "
                            f"step {repeated} times; its result cannot change. "
                            "Do something DIFFERENT now: a different file or "
                            "range, `find`/`grep` for what you are missing, or "
                            "`done` if you already have what you need.]")
                    messages.append({"role": "assistant",
                                     "content": _compact(stepjson)})
                    messages.append({"role": "user",
                                     "content": f"{note}\n\n{nudge(step + 1)}"})
                    repeated, last_sig = 0, None
                    self._emit("recovery", signal="repeat", step=step)
                    if verbose:
                        print(f"  [recovery: the same `{action}` step was "
                              f"repeated; asking for a different move]",
                              file=sys.stderr)
                    self._fit(messages, gclient)
                    if note_progress(False):
                        break
                    continue
                if verbose:
                    print(f"  [gather stopped: the model repeated the same "
                          f"`{action}` step {repeated} times; moving to "
                          f"answer]", file=sys.stderr)
                break

            dedup_step = is_repeat and action in _DEDUP_STEP_ACTIONS
            earlier = ran.get(sig) if action in _CACHEABLE_ACTIONS else None
            if earlier is not None and (earlier[1] != self.index.generation
                                        or earlier[2] != self.fold_count
                                        or earlier[3] != mutations):
                earlier = None       # the world or the conversation changed
            if dedup_step:
                # Identical to the previous (read-only) step: skip the dispatch.
                obs = (f"[This exact `{action}` step already ran and its "
                       "result is above. Repeating it cannot add information "
                       "-- choose a different query/path or say done.]")
            elif earlier is not None:
                # The same read-only query ran earlier (not just one step ago)
                # and nothing changed since: point at it instead of paying for
                # the same observation twice.
                obs = (f"[This exact `{action}` already ran at step "
                       f"{earlier[0]} and nothing changed since; its result is "
                       "earlier in this conversation. Use it, or choose a "
                       "different query.]")
                self._emit("cached_step", action=action, first=earlier[0])
            else:
                try:
                    hk = self._hook("pre_tool", action=action,
                                    args={k: v for k, v in stepjson.items()
                                          if v not in ("", 0, None)
                                          and k != "reasoning"})
                    if not hk.denied and action == "run":
                        hk = self._hook("pre_run", command=str(
                            stepjson.get("command", "")))
                    if hk.denied:
                        obs = (f"ERROR: `{action}` was blocked by a pre_tool "
                               f"hook: {hk.message}")
                    else:
                        obs = dispatch(self.index, self.cfg, stepjson,
                                       client=self.client)
                        post = self._hook("post_run" if action == "run"
                                          else "post_tool", action=action,
                                          ok=not obs.startswith("ERROR"))
                        if post.context and not obs.startswith("ERROR"):
                            obs += f"\n[hook output: {post.context}]"
                    if (action in ("run", "run_test", "screenshot", "browser")
                            and not obs.startswith("ERROR")):
                        self.live_evidence_ok = True
                        self.live_evidence_kinds.add(action)
                except Exception as e:
                    # A tool failure becomes an ERROR observation, not a crash.
                    print(f"  [{action} tool failed: "
                          f"{type(e).__name__}: {e}]", file=sys.stderr)
                    obs = (f"ERROR: {action} failed "
                           f"({type(e).__name__}): {str(e)[:200]}")
                # Output from outside the project's own text is fenced as
                # data, control characters are stripped, and instruction-like
                # wording is flagged (small models are suggestible).
                obs = safety.frame(action, obs, self.cfg)
                obs = self._cap_observation(action, obs)
                if action in _CACHEABLE_ACTIONS and not obs.startswith("ERROR"):
                    ran[sig] = (step, self.index.generation, self.fold_count,
                                mutations)
                if action in _MUTATING:
                    mutations += 1
            # The cut almost always lands inside 'reasoning'; tell the model
            # so it stops writing essays.
            if getattr(gclient, "step_truncated", False):
                obs += ("\n[Note: your previous reply hit the output token "
                        "limit before finishing its reasoning. Keep 'reasoning' "
                        "under 10 words -- long ones get cut off and wasted.]")
                # If the cut landed inside `content` of a write step, give the
                # chunk size that fits one message (dispatch() refuses the
                # oversized write).
                if action in ("write", "append") and \
                        len((stepjson.get("content") or "").splitlines()) > \
                        max(1, int(getattr(self.cfg, "max_write_lines", 0) or 0)):
                    obs += (f"\n[Your `{action}` was CUT OFF mid-content by the "
                            "output token limit -- it was NOT applied. Split it: "
                            f"`write` at most ~80 lines now; then `append` the "
                            "next ~80. Repeat `append` until the file is "
                            "complete. Never put a whole large file in one "
                            "reply.]")
            duplicate_read = False
            read_delta = None
            step_progress = False
            if action in ("read", "context") and not obs.startswith("ERROR"):
                # Tokenizer-aware hard budget (cfg.max_read_tokens) applied
                # BEFORE the source is displayed and before the ledger sees
                # it: a clipped read has its header rewritten to the range
                # that was actually delivered, so the span recorded below
                # credits only the lines the model was shown.
                obs = self._clip_read_to_token_budget(obs)
                # A read is progress only when it shows lines not already in the
                # span ledger; an exact duplicate is replaced by a compact note
                # so the model does not lock onto one range.
                m = _READ_HEAD_RX.match(obs)
                if m:
                    key = _norm_key(m.group(1))
                    s_, e_ = int(m.group(2)), int(m.group(3))
                    prior = list(spans.get(key, []))
                    new_lines = _read_has_new_lines(obs, prior)
                    step_progress = new_lines
                    read_delta = _delta_len(obs, prior)
                    duplicate_read = (action == "read" and not new_lines)
                    spans.setdefault(key, []).append((s_, e_))
                    if self.requirements:
                        self.requirements.observe_read(
                            m.group(1).strip(), s_, e_, obs, self.index)
                    if duplicate_read:
                        obs = (f"{m.group(1).strip()} lines {s_}-{e_}: "
                               "[already read; this range contains no new "
                               "source. Do not repeat it -- choose `done` or "
                               "a different file/range.]")
                if not duplicate_read:
                    did_read = True
                    read_nudged = False     # progress re-arms the coverage nudge
                    # Path-scoped rules ride the OBSERVATION TAIL only, never
                    # the system prompt: mid-gather prompt edits would break
                    # the byte-stable prefix the server cache pins.
                    if m:
                        obs += self._scoped_rules(m.group(1).strip())
                    elif action == "context":
                        rel = str(stepjson.get("file") or "").strip()
                        if not rel:
                            fm = _FILE_CTX_RX.search(obs)
                            rel = fm.group(1) if fm else ""
                        if rel:
                            obs += self._scoped_rules(rel)
                self._unread_files = self._compute_unread(self._targets_text)
            # Progress also means a lookup step RESOLVED a symbol the ledger
            # has never seen: naming a definition is information the answer
            # phase could not derive from an earlier turn.
            sym = str(stepjson.get("symbol") or "").strip().lower()
            if (sym and action in ("read", "context")
                    and not obs.startswith("ERROR")
                    and sym not in resolved_symbols):
                resolved_symbols.add(sym)
                step_progress = True
            # An ERROR is navigation feedback, not source: record it compactly
            # so the SOURCE section does not look populated without code.
            if obs.startswith("ERROR"):
                error_streak += 1
                duplicate_reads = 0      # an error is not a read
                arg = (stepjson.get("file") or stepjson.get("pattern")
                       or stepjson.get("symbol") or stepjson.get("command") or "")
                arg = arg if isinstance(arg, str) else str(arg)
                item = f"--- {action} {arg} failed: {obs[:160]}"
                gathered.append(item)
                self._emit("step", n=step, action=action, detail=arg[:120],
                           obs_chars=len(obs), error=True)
                messages.append({"role": "assistant", "content": _compact(stepjson)})
                messages.append({"role": "user",
                                 "content": f"{obs}\n\n{nudge(step + 1)}"})
                # Charge errors too, so a model repeating one failing step
                # cannot loop forever at max_steps = 0.
                budget -= len(obs)
                self._fit(messages, gclient)
                if error_streak >= _MAX_CONSECUTIVE_ERRORS:
                    if recover_on and recoveries["errors"] < 1 and budget > 0:
                        recoveries["errors"] += 1
                        messages[-1]["content"] += (
                            f"\n\n[STUCK: your last {error_streak} actions "
                            "all failed. Re-read the PROJECT MAP for exact "
                            "paths and line ranges; use `find` or `outline` "
                            "to get a valid file name before `read`; never "
                            "invent a range.]")
                        error_streak = 0
                        self._emit("recovery", signal="errors", step=step)
                        if verbose:
                            print("  [recovery: several failed steps in a row; "
                                  "pointing the model back at the map]",
                                  file=sys.stderr)
                        continue
                    if verbose:
                        print(f"  [gather stopped after {error_streak} "
                              f"consecutive failed steps; moving to answer]",
                              file=sys.stderr)
                    break
                if budget <= 0:
                    if verbose:
                        print("  [context budget exhausted; moving to answer]",
                              file=sys.stderr)
                    break
                if note_progress(False):      # an ERROR showed no new source
                    break
                continue
            error_streak = 0
            if self.requirements:
                if action in ("write", "append", "move", "delete", "rename",
                              "rename_symbol"):
                    target = (stepjson.get("file") or stepjson.get("to") or "")
                    self.requirements.observe_patch(str(target), obs)
                elif action not in ("read", "context"):
                    self.requirements.observe_search(
                        action, stepjson, obs, self.index)
            # Reads are charged by new lines only (the eviction pass at the
            # end dedups the conversation).
            if read_delta is not None:
                if read_delta > budget:
                    obs = obs[:budget] + "\n[truncated: context budget reached]"
                    budget = 0
                else:
                    budget -= read_delta
            else:
                if len(obs) > budget:
                    obs = obs[:budget] + "\n[truncated: context budget reached]"
                budget -= len(obs)

            if duplicate_read:
                duplicate_reads += 1
                if duplicate_reads >= _MAX_DUPLICATE_READS:
                    if recover_on and recoveries["dup"] < 1:
                        recoveries["dup"] += 1
                        duplicate_reads = 0
                        obs += ("\n[STUCK: you keep re-reading lines you "
                                "already have. They are above in this "
                                "conversation. Read a DIFFERENT range or "
                                "file, or say `done`.]")
                        self._emit("recovery", signal="duplicate_reads",
                                   step=step)
                        if verbose:
                            print("  [recovery: repeated reads without new "
                                  "source; asking for a new range]",
                                  file=sys.stderr)
                    else:
                        if verbose:
                            print("  [gather stopped: the model repeated the "
                                  "same read without new source; moving to "
                                  "answer]", file=sys.stderr)
                        break
            else:
                duplicate_reads = 0

            arg = (stepjson.get("file") or stepjson.get("pattern")
                   or stepjson.get("symbol") or stepjson.get("command"))
            arg = arg if isinstance(arg, str) else ("" if arg is None else str(arg))
            if action == "outline":
                # Outlines are navigation only; keep them out of the answer prompt.
                item = (f"--- outline {arg} (used for navigation; "
                        f"{len(self.index.in_file(self.index.resolve_file(arg) or ''))} units)")
            else:
                item = f"--- {action} {arg}\n{obs}"
            if not duplicate_read:
                gathered.append(item)

            if action in ("read", "context") and not duplicate_read:
                # Content-address what the model was ACTUALLY shown: the obs
                # here has survived the token clip, the char-budget cut and
                # any scoped-rule note, so the span records the delivered
                # lines, not the requested ones.
                self._record_evidence(obs, step)
            self._emit("step", n=step, action=action, detail=str(detail)[:120],
                       obs_chars=len(obs), duplicate=duplicate_read)
            messages.append({"role": "assistant", "content": _compact(stepjson)})
            messages.append({"role": "user",
                             "content": f"RESULT:\n{obs}\n\n{nudge(step + 1)}"})

            # The live conversation can outgrow the server's window even when
            # the char budget has room (some servers reject the request,
            # others silently drop the middle of the prompt): fold the older
            # band ONCE (see fold.py) instead of trimming turn by turn.
            self._fit(messages, gclient)

            if budget <= 0:
                if verbose:
                    print("  [context budget exhausted; moving to answer]", file=sys.stderr)
                break

            if note_progress(step_progress):
                break

        # A stall exit on a multi-requirement task still gets one
        # deterministic requirement lookup while budget remains. It is appended
        # at the tail, never rewriting a cached prefix, and costs no model call.
        if (budget > 0 and self.requirements
                and len(self.requirements) >= 2):
            req = self.requirements.next_unmet()
            if req is not None:
                idx = self.requirements.items.index(req)
                if idx not in req_assists:
                    harness_note(self.requirements.prompt(req)
                                 + "\n[Harness: one final deterministic "
                                   "lookup for it.]")
                    if _requirement_assist(tail=True):
                        if verbose:
                            print("  [gather stalled; the harness fetched "
                                  "the oldest unmet requirement]",
                                  file=sys.stderr)

        # P6 (A4.2): one bounded post-gather prefetch when named targets are
        # still unread or no ranked top-3 file was read.
        if getattr(self.cfg, "prefetch_targets", 0):
            did_read = self._post_gather_prefetch(
                task, messages, gathered, spans, nudge, verbose) or did_read

        self.read_any = did_read
        # Rewrite the SOURCE items down to the lines each read was first to
        # show, so the answer phase sees every line once. The live
        # conversation is NOT rewritten: it is only needed for answer
        # continuation, which wants the cached prefix untouched.
        new_items = _evict_duplicate_reads(list(messages), gathered, spans)
        digest = self._read_digest(spans)
        if digest:
            new_items = new_items + [digest]
        self.gathered_source = "\n\n".join(new_items)
        self._emit("gather_done", steps=step, files_read=len(spans),
                   folds=self.fold_count, lossy=self._lossy,
                   gathered_chars=len(self.gathered_source))
        return "\n\n".join(new_items)

    def _read_digest(self, spans: dict) -> str:
        """One line per file read during gather: the merged line ranges and
        the definitions they cover. A cheap durable record of the lookups,
        appended to the gathered source, so the answer phase keeps the map
        even when eviction has dropped the original RESULT turns."""
        lines = []
        for key in sorted(spans):
            folded = _fold_spans(spans[key])
            ranges = ", ".join(f"{a}-{b}" for a, b in folded[:8])
            if len(folded) > 8:
                ranges += ", ..."
            rel = self.index.resolve_file(key) or key
            names = []
            for c in self.index.in_file(rel):
                if c.kind in ("module", "reference") \
                        or c.qualname in ("module", "__trailing__"):
                    continue
                if any(c.start <= b and a <= (c.scope_end or c.end)
                       for a, b in folded):
                    names.append(c.qualname)
            line = f"- {rel} lines {ranges}"
            if names:
                shown = ", ".join(names[:6])
                if len(names) > 6:
                    shown += f", +{len(names) - 6} more"
                line += f" ({shown})"
            lines.append(line)
        if not lines:
            return ""
        return ("--- digest (files read during gather)\n"
                + "\n".join(lines))[:2000]

    # ------------------------------------------------------ coverage bookkeeping
    @staticmethod
    def _file_token_match(r: str, w: str) -> bool:
        """True when the normalised task token `w` names the normalised index
        path `r`: exactly, or as the path's trailing segment (a bare `app.py`
        naming `web/app.py`). The REVERSE was the reported misread: a task
        naming `web/app.py` also matched a root `app.py`, so an unrelated
        same-named file was forced-read and the "not yet read" nudge never
        cleared after the real target was read in full.
        """
        return r == w or r.endswith("/" + w)

    def _task_file_targets(self, task: str) -> set[str]:
        """Normalised paths mentioned in the task text. Only tokens containing
        a slash or a dot-extension count, so prose words are ignored."""
        out = set()
        for t in _FILE_TOKEN_RX.findall(task or ""):
            if "/" in t or _FILE_EXT_RX.search(t):
                out.add(_norm_key(t))
        return out

    def _compute_unread(self, task: str) -> list[str]:
        """Files the task names but no successful `read` covered, as entries
        like "prep/sentences.js (not yet read -- it has 222 lines, use start=1
        end=222)". Any successful read of a file counts as coverage; grep hits
        and ERROR lines do not."""
        wanted = self._task_file_targets(task)
        if not wanted:
            return []
        matched: dict[str, str] = {}          # normalised rel -> canonical rel
        for rel in self.index.files:
            r = _norm_key(rel)
            for w in wanted:
                if self._file_token_match(r, w):
                    matched[r] = rel
        if not matched:
            return []
        missing = []
        for low, rel in sorted(matched.items()):
            if self._read_spans.get(low):
                continue
            try:
                lines = fileio.read_lines(self.index.root / rel)
                if lines and lines[-1] == "" and len(lines) > 1:
                    lines = lines[:-1]   # same convention as tool_read
                if not lines or lines == [""]:
                    continue             # an empty file needs no read
                n = max(1, len(lines))
            except OSError:
                n = 0
            hint = f"it has {n} lines, use start=1 end={n}" if n else \
                   "it does not exist on disk yet"
            missing.append(f"{rel} (not yet read -- {hint})")
        return missing

    def _post_gather_prefetch(self, task, messages, gathered, spans, nudge,
                              verbose) -> bool:
        """P6 (A4.2): after gather, if named targets are still unread or no
        ranked top-3 file was read, do ONE extra read through the normal tool
        path (2.3's mechanism) -- a bounded expansion, never a new loop.
        Returns True when it read something."""
        if int(getattr(self.cfg, "prefetch_targets", 0) or 0) <= 0:
            return False
        from . import rank as rank_mod
        max_lines = max(1, int(getattr(self.cfg, "max_read_lines", 200) or 200))

        def lines_of(rel):
            try:
                lines = fileio.read_lines(self.index.root / rel)
                if lines and lines[-1] == "" and len(lines) > 1:
                    lines = lines[:-1]
                return max(1, len(lines)) if lines and lines != [""] else 0
            except OSError:
                return 0

        target = None
        # Trigger 1: a task-named file no `read` covered yet.
        wanted = self._task_file_targets(task)
        for rel in sorted(self.index.files):
            r = _norm_key(rel)
            if any(self._file_token_match(r, w)
                   for w in wanted) and not self._read_spans.get(r):
                n = lines_of(rel)
                if n:
                    target = (rel, 1, min(n, max_lines))
                    break
        # Trigger 2: zero reads of any ranked top-3 file.
        if target is None:
            for rk in rank_mod.rank_files(self.index, task)[:3]:
                if not self._read_spans.get(_norm_key(rk.file)):
                    n = lines_of(rk.file)
                    if n:
                        target = (rk.file, 1, min(n, max_lines))
                    break
        if target is None:
            return False
        rel, start, end = target
        if end <= 0:
            return False
        stepjson = {"action": "read", "file": rel, "start": start,
                    "end": end, "reasoning": "prefetch"}
        try:
            obs = dispatch(self.index, self.cfg, stepjson, client=self.client)
        except Exception:
            return False
        obs = safety.frame("read", obs, self.cfg)
        obs = self._cap_observation("read", obs)
        if obs.startswith("ERROR"):
            return False
        # Same token budget as live reads, so the ledger credits exactly the
        # lines the prefetch actually displayed.
        obs = self._clip_read_to_token_budget(obs)
        m = _READ_HEAD_RX.match(obs)
        if m:
            spans.setdefault(_norm_key(m.group(1)), []).append(
                (int(m.group(2)), int(m.group(3))))
        self._record_evidence(obs, 0)      # prefetch displays = step 0
        gathered.append(f"--- read {rel}\n{obs}")
        self.steps_log.append(f"post read {rel} L{start}-{end}  (prefetch)")
        messages.append({"role": "assistant", "content": _compact(stepjson)})
        messages.append({"role": "user",
                         "content": f"RESULT:\n{obs}\n\n{nudge(1)}"})
        self._unread_files = self._compute_unread(self._targets_text)
        if verbose:
            print(f"  (post-gather prefetch: read {rel})", file=sys.stderr)
        return True

    def _prefetch_step0(self, task: str, messages: list, gathered: list,
                        spans: dict, budget: int, nudge, verbose: bool,
                        client=None) -> int:
        """Read up to `cfg.prefetch_targets` task-named files and symbols
        before the first model step, through the normal tool path so `spans`,
        `gathered` and `steps_log` update. Returns the remaining char budget.
        Prefetch reads do not count toward `max_steps`; the whole step-0 band
        is capped at a quarter of the observation room."""
        limit = max(0, int(getattr(self.cfg, "prefetch_targets", 0) or 0))
        if limit <= 0 or budget <= 0:
            return budget
        from . import rank as rank_mod
        max_lines = max(1, int(getattr(self.cfg, "max_read_lines", 200) or 200))
        est = getattr(self.client if client is None else client,
                      "estimate_tokens", None) or estimate_tokens
        room = max(1, self._room()) // 4        # <= 25% of the observation room

        wanted = self._task_file_targets(task)
        named: dict[str, str] = {}              # normalised key -> canonical rel
        for rel in self.index.files:
            r = _norm_key(rel)
            for w in wanted:
                if self._file_token_match(r, w):
                    named[r] = rel

        def lines_of(rel: str) -> int:
            try:
                lines = fileio.read_lines(self.index.root / rel)
                if lines and lines[-1] == "" and len(lines) > 1:
                    lines = lines[:-1]
                return max(1, len(lines)) if lines and lines != [""] else 0
            except OSError:
                return 0

        targets: list[tuple[str, int, int]] = []
        seen: set[str] = set()
        # 1) task-named files small enough to read whole.
        for rel in sorted(named.values()):
            n = lines_of(rel)
            if 0 < n <= max_lines:
                seen.add(rel)
                targets.append((rel, 1, n))
        # 2) the files defining task-named symbols, trimmed to max_read_lines.
        for c in rank_mod.task_symbols(self.index, task):
            if c.file in seen:
                continue
            end = c.scope_end or c.end
            if end - c.start + 1 > max_lines:
                end = c.start + max_lines - 1
            seen.add(c.file)
            targets.append((c.file, c.start, end))

        spent = 0                                # prefetched tokens so far
        prefetched = 0
        for rel, start, end in targets[:limit]:
            if prefetched >= limit or budget <= 0 or spent >= room:
                break
            stepjson = {"action": "read", "file": rel, "start": start,
                        "end": end, "reasoning": "prefetch"}
            try:
                obs = dispatch(self.index, self.cfg, stepjson,
                               client=self.client)
            except Exception:
                continue
            obs = safety.frame("read", obs, self.cfg)
            obs = self._cap_observation("read", obs)
            if obs.startswith("ERROR"):
                continue
            obs = self._clip_read_to_token_budget(obs)
            m = _READ_HEAD_RX.match(obs)
            if m:
                spans.setdefault(_norm_key(m.group(1)), []).append(
                    (int(m.group(2)), int(m.group(3))))
                delta = _delta_len(obs, [])
            else:
                delta = len(obs)
            self._record_evidence(obs, 0)  # prefetch displays = step 0
            budget -= delta
            gathered.append(f"--- read {rel}\n{obs}")
            self.steps_log.append(f"0. read {rel} L{start}-{end}  (prefetch)")
            messages.append({"role": "assistant", "content": _compact(stepjson)})
            messages.append({"role": "user",
                             "content": f"RESULT:\n{obs}\n\n{nudge(1)}"})
            spent += est(obs)
            prefetched += 1
        self._unread_files = self._compute_unread(self._targets_text)
        return budget

    # ------------------------------------------------------------------ phase 2
    def answer(self, task: str, gathered: str, stream: bool = True,
               system: str | None = None, max_tokens: int | None = None,
               include_edit_format: bool = True) -> str:
        """Ask the model to answer the task, or produce search/replace edits,
        given the project map and whatever source `gather()` collected.

        `system` and `max_tokens` override the defaults for flows that need a
        different brief (e.g. scaffold.py). `include_edit_format=False` drops
        the search/replace format spec even when `system` is given, for
        prose-only flows (`explain`).

        Two ways to build the prompt (`answer_mode`): REBUILD sends a fresh
        two-message conversation (map + source + task); CONTINUE appends one
        instruction to the gather conversation so the server's prompt cache
        covers the map and every lookup. `auto` continues and falls back to a
        rebuild when the continued reply is empty or tool-shaped.

        Samples at `temperature_answer` when set (the reply copies SEARCH text
        character for character from what was read)."""
        with self._phase_temp("temperature_answer"):
            return self._answer(task, gathered, stream, system, max_tokens,
                                include_edit_format)

    def _answer(self, task: str, gathered: str, stream: bool,
                system: str | None, max_tokens: int | None,
                include_edit_format: bool) -> str:
        self._set_site("answer")
        brief = self._answer_brief(system, include_edit_format, task)
        mode = self._answer_mode()
        if mode in ("continue", "auto") and self._can_continue(task, brief,
                                                               max_tokens):
            reply = self._answer_continue(task, brief, stream, max_tokens)
            if reply is not None and (mode == "continue"
                                      or not _unusable_continuation(reply)):
                self.answer_mode_used = "continue"
                self._emit("answer", mode="continue",
                           reply_chars=len(reply or ""))
                return reply
            if reply is not None:
                print("  [the continued answer was unusable; rebuilding the "
                      "answer prompt]", file=sys.stderr)
        self.answer_mode_used = "rebuild"
        reply = self._answer_rebuild(task, gathered, stream, brief, max_tokens)
        self._emit("answer", mode="rebuild", reply_chars=len(reply or ""))
        return reply

    # ------------------------------------------------- verify before applying
    # ----------------------------------------- incremental edit microloop
    def incremental_eligible(self, task: str) -> bool:
        """True when this task is multi-hunk/multi-file enough to use the
        symbol-anchored microloop. Single-target edits keep the ordinary
        one-reply path: the extra intent call would only add latency."""
        if not bool(getattr(self.cfg, "incremental_edit_loop", True)):
            return False
        tracker = getattr(self, "requirements", None)
        edits = [r for r in getattr(tracker, "items", ())
                 if getattr(r, "is_edit", False)]
        if len(edits) >= 2:
            return True
        text = getattr(self, "task_targets", None) or task or ""
        if len(self._task_file_targets(text)) >= 2:
            return True
        names: set[str] = set()
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", text)[:48]:
            if len(tok) < 3 or tok.lower() in {
                    "the", "and", "for", "with", "this", "that", "from",
                    "into", "file", "files", "code", "test", "tests"}:
                continue
            try:
                if self.index.find_symbol(tok):
                    names.add(tok)
            except Exception:
                pass
        return len(names) >= 2

    def _intent_source(self, gathered: str) -> str:
        """Gathered source trimmed for the intent-planner call (token-budgeted
        by the same utility packer the answer phase uses)."""
        est = getattr(self.client, "estimate_tokens", None) or estimate_tokens
        limit = max(1500, int((self.cfg.context_tokens - 1200) * 0.55))
        src, _dropped = _trim_gathered(
            gathered or "(nothing was looked up)",
            lambda t: est(t) > limit,
            requirements=getattr(self, "requirements", None),
            diagnostics=getattr(self, "active_diagnostics", None),
            index=self.index)
        return src

    def plan_edit_intents(self, task: str, gathered: str,
                          verbose: bool = False) -> list[dict]:
        """Ask for the structured `[{file, symbol, intended_change}]` plan.

        One small greedy call. An empty/unparseable reply returns [] so the
        caller can fall back to the ordinary answer path; no failure here can
        block a task."""
        try:
            repo_map, _tier = self._map(task)
        except Exception:
            repo_map = ""
        source = self._intent_source(gathered)
        messages = [
            {"role": "system", "content": INCREMENTAL_INTENT_SYSTEM},
            {"role": "user", "content":
             f"PROJECT MAP\n\n{repo_map}\n\n"
             f"SOURCE YOU REQUESTED\n\n{source}\n\n"
             f"TASK\n{task}\n\n"
             "List the symbol-sized edits this task needs. JSON array only."},
        ]
        try:
            self._set_site("answer")
            text = self._chat_at(
                "temperature_answer", messages,
                max_tokens=min(int(getattr(self.cfg, "max_tokens_answer",
                                            3000) or 3000), 1200),
                stream=False)
        except LLMError as e:
            if verbose:
                print(f"  [edit-intent plan failed ({type(e).__name__}); "
                      "using the ordinary answer path]", file=sys.stderr)
            return []
        intents = parse_edit_intents(text)
        self._emit("edit_intents", count=len(intents),
                   files=sorted({i["file"] for i in intents if i["file"]}))
        if verbose and intents:
            print(f"  [incremental edit plan: {len(intents)} symbol(s)]",
                  file=sys.stderr)
        return intents

    def _resolve_edit_intent(self, intent: dict):
        """Resolve one intent to (target, failure). `target` carries the file,
        symbol, exact 1-based line range and the planner's intended change."""
        raw_file = str(intent.get("file") or "").strip()
        symbol = str(intent.get("symbol") or "").strip()
        change = str(intent.get("intended_change") or "").strip()
        rel = self.index.resolve_file(raw_file) if raw_file else None
        if not rel and raw_file:
            cand = fileio.normalise_rel(raw_file)
            if cand and ((self.index.root / cand).exists()
                         or not symbol
                         or symbol.lower() in _INTERNAL_SYMBOLS):
                rel = cand
        if not rel and symbol:
            hits = self.index.find_symbol(symbol)
            files = {c.file for c in hits if c.file}
            if len(files) == 1:
                rel = files.pop()
        if not rel:
            return None, Result(
                False, raw_file or symbol,
                f"edit intent cannot be resolved to a project file "
                f"(file={raw_file!r}, symbol={symbol!r})")

        path = self.index.root / rel
        if not symbol or symbol.lower() in _INTERNAL_SYMBOLS:
            if path.is_file():
                try:
                    lines = fileio.read_lines(path)
                except OSError as e:
                    return None, Result(False, rel, f"cannot read file: {e}")
                end = len(lines)
                if lines and lines[-1] == "" and len(lines) > 1:
                    end -= 1
                start = 1
            else:
                start, end = 1, 0          # a new file: nothing to quote
            return {"file": rel, "symbol": "", "start": start, "end": end,
                    "intended_change": change}, None

        same = [c for c in self.index.find_symbol(symbol)
                if _same_rel(c.file, rel)]
        want = fileio.path_key(symbol)
        exact = [c for c in same if fileio.path_key(c.qualname) == want
                 or fileio.path_key(c.qualname).endswith("." + want)]
        if len(exact) == 1:
            c = exact[0]
        elif len(same) == 1:
            c = same[0]                    # a method inside a whole-class chunk
        else:
            if path.is_file():
                # No index chunk for the symbol (definitions nested inside a
                # function, an IIFE or a closure are deliberately not indexed)
                # or several match in this file. The symbol-anchored hunk
                # cannot be positioned -- but failing the whole intent here
                # would strand work the planner already scoped. Fall back to
                # the WHOLE-FILE target: the model gets the file verbatim and
                # emits a plain quoted SEARCH/REPLACE block for it.
                self._emit("incremental_edit_fallback", file=rel,
                           symbol=symbol,
                           reason=("no definition by that name in this file"
                                   if not same else
                                   f"{len(same)} definitions match"))
                try:
                    lines = fileio.read_lines(path)
                except OSError as e:
                    return None, Result(False, rel, f"cannot read file: {e}")
                end = len(lines)
                if lines and lines[-1] == "" and len(lines) > 1:
                    end -= 1
                return {"file": rel, "symbol": "", "start": 1, "end": end,
                        "intended_change": change}, None
            if not same:
                why = "no definition by that name in this file"
            else:
                listing = ", ".join(f"{c.qualname} L{c.start}-{c.end}"
                                    for c in same[:5])
                why = f"{len(same)} definitions match ({listing})"
            return None, Result(
                False, rel, f"edit intent symbol {symbol!r}: {why}")
        start = max(1, int(c.start or 1))
        end = int(getattr(c, "scope_end", 0) or c.end or start)
        return {"file": rel, "symbol": symbol, "start": start, "end": end,
                "intended_change": change}, None

    def _disk_line_count(self, rel: str) -> int:
        """Lines of `rel` as it sits on disk, 0 when it is not there (a file
        this hunk would create) or unreadable."""
        try:
            path = self.index.root / rel
            if not path.is_file():
                return 0
            return len(path.read_text(encoding="utf-8",
                                      errors="replace").splitlines())
        except (OSError, ValueError):
            return 0

    def _check_incremental_hunk(self, task: str, gathered: str, rel: str,
                                symbol: str, reply: str):
        """Dry-apply ONE target hunk, then run verify.py and the AST gate."""
        from . import verify as verify_mod
        parsed = parse_edits(reply)
        if len(parsed) != 1:
            return [Result(
                False, rel,
                f"incremental edit must contain exactly ONE block for "
                f"{rel}:{symbol or '(file)'}; got {len(parsed)}")], []
        block = parsed[0]
        if not _same_rel(block.file, rel):
            return [Result(
                False, rel,
                f"incremental edit targeted {block.file!r}, but the current "
                f"symbol is in {rel}")], []
        if symbol and block.symbol:
            bw, tw = fileio.path_key(block.symbol), fileio.path_key(symbol)
            if not (bw == tw or bw.endswith("." + tw) or tw.endswith("." + bw)):
                return [Result(
                    False, rel,
                    f"incremental edit anchored {block.symbol!r}, but the "
                    f"current target is {symbol!r}")], []
        if symbol and not block.symbol and not block.lines \
                and not block.search.strip():
            return [Result(
                False, rel,
                "an empty SEARCH without a SYMBOL/LINES anchor would replace "
                f"the WHOLE file; target {symbol!r} with a SYMBOL header")], []
        if symbol and not block.symbol and block.lines:
            # `LINES 1-171` of a 171-line file is a whole-file rewrite wearing
            # a range anchor: the symbol-sized target this loop asked for is
            # gone, and the answer token cap cuts the copied file mid-definition
            # -- which is how one run lost a class and a factory on a retry.
            first, last = block.lines
            total = self._disk_line_count(rel)
            if (total >= 40 and first <= 2
                    and (last - first + 1) * 4 >= total * 3):
                return [Result(
                    False, rel,
                    f"this block replaces lines {first}-{last} of a {total}-"
                    f"line file -- that is the WHOLE file, for one definition "
                    f"({symbol!r}). Re-emit it anchored on the symbol, with "
                    f"the full new definition as the body and no SEARCH "
                    f"section at all:\n"
                    f"<<<<<<< SYMBOL {rel} :: {symbol}\n"
                    f"...the complete new {symbol}...\n"
                    f">>>>>>> END_SYMBOL\n"
                    f"Never copy a whole file into a reply: it runs past the "
                    f"output limit and is cut off mid-line.")], []
        results = self._apply_once(reply, dry_run=True)
        # Run the syntax gate explicitly (verify.check also invokes it; the
        # duplicate is cheap and keeps this microloop's contract self-evident).
        syntax = verify_mod.check_syntax(results)
        findings = syntax + [
            f for f in verify_mod.check(self, task, reply, results, gathered)
            if getattr(f, "kind", "") != "syntax"]
        return results, findings

    @staticmethod
    def _incremental_receipt(rel: str, symbol: str, ok: bool,
                             reason: str = "", repairs: int = 0) -> str:
        name = f"{rel}:{symbol}" if symbol else rel
        if ok:
            tail = f" (repaired {int(repairs)}x)" if repairs else ""
            return f"[EDITED {name}{tail}]"
        tail = f" -- {reason}" if reason else ""
        return f"[FAILED {name}{tail}]"

    def apply_edit_intents(self, task: str, gathered: str, intents: list[dict],
                           backup_dir=None, verbose: bool = False):
        """Apply a planned intent list one symbol at a time.

        Each target is resolved against the CURRENT index/disk, its exact source
        is appended at the prompt tail, and the model emits exactly one hunk.
        The hunk is dry-applied and checked before anything is written; after a
        success the temporary source is gone from the next prompt (only a
        one-line receipt remains). Returns (results, reply, findings)."""
        from . import verify as verify_mod
        self._set_site("answer")
        system = self._answer_sys(task) + (
            "\n\nINCREMENTAL EDIT MODE: you are editing ONE symbol at a time. "
            "Emit exactly ONE search/replace block and nothing else. Prefer "
            "the SYMBOL anchor with EMPTY SEARCH and the FULL new definition "
            "in REPLACE; otherwise quote the exact current source. Do not "
            "repeat earlier hunks or edit any other symbol.")
        history = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"TASK\n{task}"},
        ]
        results_all: list = []
        findings_all: list = []
        receipts: list[str] = []
        attempts = 1 + max(0, min(int(getattr(self.cfg, "repair_rounds", 2)
                                           or 0), 3))
        total = len(intents)
        for n, intent in enumerate(intents, 1):
            target, failure = self._resolve_edit_intent(intent)
            if target is None:
                results_all.append(failure)
                reason = (failure.message.split(":", 1)[-1].strip()
                          if failure and failure.message else "unresolved")
                receipts.append(self._incremental_receipt(
                    getattr(failure, "file", "") or str(intent.get("file")),
                    str(intent.get("symbol") or ""), False, reason[:120]))
                self._emit("incremental_edit", target=n, of=total, ok=False,
                           reason=reason[:160])
                continue
            rel = target["file"]
            symbol = target["symbol"]
            start, end = target["start"], target["end"]
            source = self._fetch_exact_span(rel, start, end) if end >= start \
                else ""
            if not source:
                source = (f"{rel} lines {start}-{end}: (this file does not "
                          "exist yet; the REPLACE body is its full contents)")
            already = ("\n\nEDITS ALREADY VERIFIED (their source was removed "
                       "to save context):\n" + "\n".join(receipts)
                       if receipts else "")
            # Snapshot P immediately before requesting this symbol's hunk: a
            # successful repair collapses back here, so failed drafts and
            # diagnostic turns never poison the next symbol.
            prefix_index = len(history)
            target_msg = (
                already + f"\n\nINCREMENTAL TARGET {n}/{total}\n"
                f"file: {rel}\nsymbol: {symbol or '(whole file)'}\n"
                f"intended change: {target['intended_change'] or task}\n\n"
                "EXACT CURRENT SOURCE (fresh from disk, numbered like a "
                "`read`; ignore the line-number prefixes):\n"
                f"{source}\n\n"
                "Emit EXACTLY ONE search/replace block for THIS target now. "
                "Use `--- FILE: " + rel + (" SYMBOL " + symbol if symbol
                                           else "") +
                "` with an EMPTY SEARCH and the full new definition in "
                "REPLACE, or quote the exact lines in SEARCH. No other text "
                "and no other block.")
            msgs = history + [{"role": "user", "content": target_msg}]
            last_results: list = []
            last_findings: list = []
            last_reason = ""               # why the last draft was rejected
            accepted = None
            for attempt in range(attempts):
                try:
                    reply = self._chat_at(
                        "temperature_answer", msgs,
                        max_tokens=self.cfg.max_tokens_answer, stream=False)
                except LLMError as e:
                    last_results = [Result(
                        False, rel, f"edit generation failed: {type(e).__name__}")]
                    break
                hunk_results, hunk_findings = self._check_incremental_hunk(
                    task, gathered, rel, symbol, reply)
                errors = [f for f in hunk_findings if f.severity == "error"]
                clean = bool(hunk_results) and all(
                    getattr(r, "ok", False) for r in hunk_results) and not errors
                if clean and getattr(self.client, "stream_truncated", False):
                    clean = False
                    hunk_results = [Result(
                        False, rel, "the incremental hunk was truncated at the "
                                    "output limit")]
                if clean:
                    applied = self._apply_once(reply, False, backup_dir)
                    if applied and all(getattr(r, "ok", False) for r in applied):
                        accepted = (applied, hunk_findings, reply, attempt)
                        break
                    hunk_results = applied or hunk_results
                    errors = []
                last_results, last_findings = hunk_results, hunk_findings
                self._emit("incremental_edit", target=n, of=total, ok=False,
                           attempt=attempt + 1,
                           errors=len(errors),
                           failed=sum(1 for r in hunk_results
                                      if not getattr(r, "ok", False)))
                if attempt + 1 >= attempts:
                    break
                problems = verify_mod.render(errors) or "\n".join(
                    f"- {r.file}: {r.message}" for r in hunk_results
                    if not getattr(r, "ok", False))
                last_reason = (problems or "").strip().split("\n")[0][:180]
                msgs.append({"role": "assistant", "content": reply})
                msgs.append({"role": "user", "content":
                    f"THE HUNK FOR {rel}:{symbol or '(file)'} WAS REJECTED "
                    f"before writing:\n{problems}\n\n"
                    "Re-emit EXACTLY ONE corrected search/replace block for "
                    "this same target. Copy SEARCH text exactly from the "
                    "source shown above, or use the SYMBOL anchor with an "
                    "EMPTY SEARCH and the full new definition. No other text."})
            if accepted:
                applied, hunk_findings, _reply, attempt = accepted
                results_all.extend(applied)
                findings_all.extend(hunk_findings)
                try:
                    from . import index as index_mod
                    index_mod.reindex_file(self.index, rel, self.cfg,
                                           save=False)
                except Exception:
                    pass
                receipt = self._incremental_receipt(
                    rel, symbol, True, repairs=max(0, attempt))
                receipts.append(receipt)
                # Collapse the target request and any rejected drafts back to
                # P, leaving only the compact verified receipt in the tail.
                self._collapse_repair_turns(
                    msgs, prefix_index, receipt=receipt,
                    reason=(f"{rel}:{symbol or '(file)'} after "
                            f"{max(0, attempt)} rejected draft(s): "
                            + (last_reason or "re-anchored"))
                    if attempt else "")
                self._emit("incremental_edit", target=n, of=total, ok=True,
                           attempt=attempt + 1)
                if verbose:
                    print(f"  [incremental {n}/{total}: {rel}:{symbol or '(file)'}"
                          " applied and verified]", file=sys.stderr)
            else:
                if not last_results:
                    last_results = [Result(
                        False, rel, "no usable incremental hunk was produced")]
                results_all.extend(last_results)
                findings_all.extend(last_findings)
                reason = next((r.message for r in last_results
                               if not getattr(r, "ok", False)), "")
                receipts.append(self._incremental_receipt(
                    rel, symbol, False, (reason or "rejected")[:120]))
                if verbose:
                    print(f"  [incremental {n}/{total}: {rel}:{symbol or '(file)'}"
                          f" not applied -- {(reason or 'rejected')[:120]}]",
                          file=sys.stderr)
        tracker = getattr(self, "requirements", None)
        if tracker is not None and getattr(tracker, "task", "") == task:
            tracker.observe_edit(results_all)
            if results_all and all(getattr(r, "ok", False)
                                   for r in results_all):
                tracker.observe_verified(
                    {r.file for r in results_all if r.file})
        mixed = [r for r in results_all if not getattr(r, "ok", False)]
        if results_all and mixed and len(mixed) < len(results_all):
            # Half a feature is on disk and half is not. Nothing rolls back:
            # every hunk that landed was verified landing, and silently
            # reverting a user's working tree would be the worse surprise.
            # What this batch owed was the honesty of its own shape -- the
            # run that lost `MortarShell` to a rejected block reported the
            # other ten targets as if the feature were complete.
            names = ", ".join(sorted({r.file for r in mixed if r.file})[:6])
            extra = (f" (+{len(mixed)} more)" if len(mixed) > 6 else "")
            undo = (f" restore the pre-edit backup at {backup_dir}"
                    if backup_dir else " undo the applied targets by hand")
            self._emit("incremental_partial", applied=len(results_all)
                       - len(mixed), failed=len(mixed), files=names)
            if verbose:
                print(f"\n[INCREMENTAL BATCH PARTIALLY APPLIED: "
                      f"{len(results_all) - len(mixed)} of "
                      f"{len(results_all)} block(s) landed; NOT applied: "
                      f"{names}{extra}]\n"
                      f"  The files already written stay written -- the "
                      f"feature is half-implemented, and a later step reading "
                      f"them sees the mix. To start over:{undo}.",
                      file=sys.stderr)
        reply = ("Symbol-anchored incremental edits:\n" + "\n".join(receipts)
                 if receipts else "No symbol-anchored edits were applied.")
        return results_all, reply, findings_all

    def incremental_edit(self, task: str, gathered: str, backup_dir=None,
                         verbose: bool = False):
        """Plan then apply the incremental microloop, or None when the plan has
        fewer than two targets (the caller should use the ordinary answer)."""
        intents = self.plan_edit_intents(task, gathered, verbose=verbose)
        if len(intents) < 2:
            return None
        return self.apply_edit_intents(task, gathered, intents,
                                       backup_dir=backup_dir, verbose=verbose)

    def preflight(self, task: str, reply: str, gathered: str = "",
                  verbose: bool = True):
        """Check an edit reply BEFORE anything is written: dry-apply every
        block, run the deterministic checks (verify.py), and -- when they find
        errors -- ask the model for one corrected reply. With
        verify_answer = "full" and a clean deterministic pass, one adversarial
        verifier call is made too. Returns (reply, findings)."""
        from . import verify as verify_mod
        mode = (getattr(self.cfg, "verify_answer", "deterministic") or "off")
        if mode == "off" or not parse_edits(reply):
            return reply, []
        self._set_site("verify")
        findings: list = []
        for round_ in range(2):          # the check, and at most one repair
            results = self._apply_once(reply, dry_run=True)
            findings = verify_mod.check(self, task, reply, results, gathered)
            if mode == "full" and round_ == 0 and not any(
                    f.severity == "error" for f in findings):
                findings += self._model_verify(task, reply, gathered)
            errors = [f for f in findings if f.severity == "error"]
            self._emit("verify", round=round_, errors=len(errors),
                       warnings=len(findings) - len(errors),
                       kinds=sorted({f.kind for f in findings}))
            if not errors or round_ == 1:
                break
            if verbose:
                print(f"\n[preflight: {len(errors)} problem(s) in the "
                      "proposed edits; asking for a corrected reply]",
                      file=sys.stderr)
                for f in errors[:4]:
                    print("  " + f.render(), file=sys.stderr)
            fixed = self._repair_from_findings(task, reply, errors, results,
                                               findings=findings)
            if not fixed:
                break
            reply = fixed
        tracker = getattr(self, "requirements", None)
        if tracker is not None and getattr(tracker, "task", "") == task:
            tracker.observe_edit(results)
        if verbose:
            for f in findings:
                if f.severity == "warn":
                    print("  " + f.render(), file=sys.stderr)
        return reply, findings

    def _repair_from_findings(self, task: str, reply: str, errors,
                              results, findings=None) -> str:
        """One re-ask that shows the model its own edits plus the check's
        findings; '' when it produced no usable blocks -- said out loud, so
        a give-up mid-repair never reads as a clean run. `findings` (all
        severities) additionally enables the implicit rehydration for fuzzy
        WARNINGS; `errors` alone still drives the repair."""
        from . import verify as verify_mod

        class _F:
            def __init__(self, file):
                self.file = file
                self.message = "check failed"

        files = [fileio.normalise_rel(f.file) for f in errors if f.file]
        src = self._repair_source([_F(f) for f in dict.fromkeys(files)],
                                  "") if files else ""
        blocks = self._failed_block_texts(
            reply, [_F(e.file) for e in parse_edits(reply)])
        client = self._role_client("repair")
        system = self._answer_sys(task)
        if any(getattr(f, "kind", "") == "tests" for f in errors):
            # The repair IS about failing tests; the value-bound rules ride
            # along even when the task text never says "test" ("fix the cart
            # total") -- the classic move there is weakening the red
            # assertion instead of fixing the code.
            rules = prompts_mod.test_rules(self.cfg)
            if rules not in system:
                system += "\n\n" + rules
        # Implicit rehydration: flagged regions the model WAS
        # shown get their fresh exact bytes appended at the TAIL of this
        # repair prompt, and the read ledger is credited -- the corrected
        # quote can then land without re-flagging as an unread edit.
        rehyd = self._auto_rehydrate(findings if findings is not None
                                     else errors)
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content":
                    (f"SOURCE (current contents of the files involved)\n\n{src}\n\n"
                     if src else "")
                    + f"TASK\n{task}\n\n"
                    + f"YOUR PROPOSED EDIT BLOCKS:\n{blocks}\n\n"
                    + "An automatic check of these edits found problems:\n"
                    + verify_mod.render(errors) + "\n\n"
                    + "Output the COMPLETE corrected set of search/replace blocks "
                    "for the whole task (not only the changed ones). Fix each "
                    "problem, or -- if a problem means you are editing code you "
                    "never read -- drop that edit and say what you could not "
                    "verify. Copy SEARCH text exactly from the SOURCE above. "
                    "Blocks only after a one-line explanation."
                    + (_NEED_HINT if self._rehydration_rounds_left() else "")
                    + rehyd}]
        # Snapshot P before the repair generation. A usable corrected reply
        # collapses any NEED/rehydration turns _chat_needs appended, leaving
        # one clean accepted reply; an exhausted repair keeps its failure.
        repair_prefix = len(msgs)
        try:
            new = self._chat_needs(
                lambda m: self._chat_at("temperature_repair", m, client=client,
                                        max_tokens=self.cfg.max_tokens_answer,
                                        stream=False),
                msgs)
        except LLMError as e:
            print("  [repair: re-ask failed (%s); failing checks remain]"
                  % type(e).__name__, file=sys.stderr)
            return ""
        if not parse_edits(new):
            print("  [repair: re-ask produced no usable fix; failing checks "
                  "remain]", file=sys.stderr)
            return ""
        self._collapse_repair_turns(msgs, repair_prefix, clean_reply=new,
                                    reason="repair accepted after "
                                           f"{len(msgs) - repair_prefix} "
                                           "diagnostic turn(s)")
        return new

    def _model_verify(self, task: str, reply: str, gathered: str):
        """The optional adversarial verifier call (verify_answer = "full")."""
        from . import verify as verify_mod
        est = getattr(self.client, "estimate_tokens", None) or estimate_tokens
        limit = max(1500, int((self.cfg.context_tokens - 900) * 0.5))
        src, _ = _trim_gathered(
            gathered or "(nothing was looked up)",
            lambda t: est(t) > limit,
            requirements=getattr(self, "requirements", None),
            diagnostics=getattr(self, "active_diagnostics", None),
            index=self.index)
        blocks = self._failed_block_texts(
            reply, [type("F", (), {"file": e.file})() for e in parse_edits(reply)])
        client = self._role_client("verifier")
        try:
            # The verifier may also ask: a `NEED file:120-168` line gets that
            # exact span appended at the tail and one bounded re-ask, so its
            # verdict judges real source instead of a trimmed summary.
            text = self._chat_needs(
                lambda m: client.chat(m, max_tokens=350, stream=False),
                [{"role": "system", "content": verify_mod.VERIFIER_SYSTEM},
                 {"role": "user", "content":
                  f"TASK\n{task}\n\nSOURCE\n{src}\n\nPROPOSED EDITS\n{blocks}"
                  + (_NEED_HINT if self._rehydration_rounds_left() else "")}])
        except LLMError:
            return []
        return verify_mod.parse_verdict(text, self.index.resolve_file,
                                        self.index.root)

    def _role_client(self, role: str):
        """The client a role resolves to (falls back to the worker)."""
        from . import roles as roles_mod
        try:
            return roles_mod.client_for(self.cfg, role, self.client)
        except Exception:
            return self.client

    def best_of(self, task: str, first_reply: str, gathered: str = "",
                verbose: bool = True) -> str:
        """Best-of-N for weak models: sample N-1 more candidate replies at a
        higher temperature, score every candidate with the deterministic
        checks (blocks dry-apply, no placeholder/unread/stale errors,
        requirements addressed), and keep the best. N = cfg.best_of."""
        from . import verify as verify_mod
        n = int(getattr(self.cfg, "best_of", 1) or 1)
        if n <= 1 or not parse_edits(first_reply):
            return first_reply
        candidates = [first_reply]
        base = getattr(self.cfg, "temperature_answer", None)
        if base is None:
            base = getattr(self.cfg, "temperature", 0.2)
        temp = min(1.0, float(base) + 0.4)
        for i in range(n - 1):
            try:
                ctx = self.client.temperature(temp) \
                    if hasattr(self.client, "temperature") else None
                if ctx is not None:
                    with ctx:
                        cand = self.answer(task, gathered, stream=False)
                else:
                    cand = self.answer(task, gathered, stream=False)
            except LLMError:
                break
            if cand and parse_edits(cand):
                candidates.append(cand)
        if len(candidates) == 1:
            return first_reply
        best, best_score = first_reply, None
        scores = []
        for cand in candidates:
            results = self._apply_once(cand, dry_run=True)
            score, _f = verify_mod.score_candidate(self, task, cand, results,
                                                   gathered)
            scores.append(round(score, 1))
            if best_score is None or score > best_score:
                best, best_score = cand, score
        if verbose:
            print(f"  [best-of-{len(candidates)}: candidate scores {scores}; "
                  f"kept #{candidates.index(best) + 1}]", file=sys.stderr)
        self._emit("best_of", scores=scores, kept=candidates.index(best))
        return best

    # ---- answer helpers
    def _answer_sys(self, task: str = "") -> str:
        """The default answer-phase system prompt (pack-aware, overridable;
        a test-shaped `task` adds the value-bound test rules)."""
        return prompts_mod.answer_system(self.cfg, EDIT_FORMAT, task)

    def _answer_brief(self, system: str | None, include_edit_format: bool,
                      task: str = "") -> str:
        """The answer-phase instructions: the default edit brief, or a
        caller's system text (plus the edit format when asked)."""
        if system is None:
            return prompts_mod.answer_system(self.cfg, EDIT_FORMAT, task)
        brief = system.rstrip()
        if include_edit_format:
            brief += "\n\n" + prompts_mod.edit_format(self.cfg, EDIT_FORMAT)
        return brief

    def _answer_mode(self) -> str:
        """rebuild | continue | auto: the setting, else the model card, else
        auto for hosted APIs (cheap, and well-behaved) and rebuild for local
        models (which sometimes imitate their own tool JSON)."""
        mode = (getattr(self.cfg, "answer_mode", "") or "").strip().lower()
        if mode in ("rebuild", "continue", "auto"):
            return mode
        card = getattr(self.cfg, "_card", None) or {}
        chosen = (card.get("choices") or {}).get("answer_mode")
        if chosen in ("rebuild", "continue", "auto"):
            return chosen
        return "auto" if is_hosted(self.cfg) else "rebuild"

    def _can_continue(self, task: str, brief: str, max_tokens: int | None
                      ) -> bool:
        """The gather conversation can host the answer: it exists, nothing in
        it was folded away (the answer needs the source), and the final
        instruction still fits the window."""
        msgs = self._messages
        if len(msgs) < 2 or self._lossy:
            return False
        est = getattr(self.client, "estimate_tokens", None) or estimate_tokens
        reply_room = min(max_tokens or self.cfg.max_tokens_answer,
                         self.cfg.request_max_tokens_cap
                         or self.cfg.max_tokens_answer)
        need = (estimate_messages(msgs, est) + est(brief) + est(task)
                + 120 + reply_room)
        window = self.cfg.context_tokens - self.cfg.reserve_tokens // 2
        return need <= window if not is_hosted(self.cfg) else True

    def _answer_continue(self, task: str, brief: str, stream: bool,
                         max_tokens: int | None) -> str | None:
        """Append the final instruction to the gather conversation and answer
        with no grammar. The server reuses the whole warm prefix. Returns None
        when the call itself failed in a way a rebuild might survive."""
        msgs = [dict(m) for m in self._messages]
        digest = self._read_digest(self._read_spans) \
            if getattr(self, "_read_spans", None) else ""
        if msgs[-1]["role"] == "user":
            msgs.append({"role": "assistant", "content":
                         "{'action': 'done', 'reasoning': 'enough'}"})
        final = ("THE LOOKUP PHASE IS OVER. No tools are available now: do "
                 "NOT emit JSON actions, tool calls or `done`. Answer the task "
                 "below using only the source you read above.\n\n"
                 f"TASK\n{task}\n\n"
                 + (digest + "\n\n" if digest else "")
                 + "ANSWER INSTRUCTIONS\n" + brief
                 + (_NEED_HINT if self._rehydration_rounds_left() else ""))
        msgs.append({"role": "user", "content": final})
        est = getattr(self.client, "estimate_tokens", None) or estimate_tokens
        print(f"[answer continues the lookup conversation, "
              f"~{estimate_messages(msgs, est)} tokens, prefix cached]",
              file=sys.stderr)
        try:
            # `NEED path:120-168` rehydration (2B): fetched spans append at
            # this conversation's TAIL, so the cached gather prefix is never
            # rewritten and the extra round costs a cache hit, not a miss.
            return self._chat_needs(
                lambda m: self.client.chat(
                    m, max_tokens=max_tokens or self.cfg.max_tokens_answer,
                    stream=stream),
                msgs)
        except LLMError as e:
            print(f"  [continued answer failed ({str(e)[:100]}); rebuilding]",
                  file=sys.stderr)
            return None

    def _answer_rebuild(self, task: str, gathered: str, stream: bool,
                        brief: str, max_tokens: int | None) -> str:
        """A fresh two-message answer conversation. Over budget it DEGRADES
        (drop project notes, shrink the map, drop the oldest whole source
        items with re-read pointers) before it ever refuses."""
        repo_map, _tier = self._map(task)
        project_notes = instructions_mod.render(self.index.root, self.cfg)
        todo_notes = todo_mod.render(self.index.root, self.cfg,
                                     limit=int(getattr(
                                         self.cfg, "max_todo_chars", 1500)))
        memory_notes = memory_mod.render(self.index.root, self.cfg)
        try:
            micro_notes = memory_mod.render_micro_facts(self.index.root, self.cfg)
        except Exception:
            micro_notes = ""
        try:
            lesson_notes = memory_mod.render_anti_patterns(self.index.root,
                                                           self.cfg)
        except Exception:
            lesson_notes = ""
        notes_prefix = ""
        if project_notes:
            notes_prefix += project_notes + "\n\n"
        if memory_notes:
            notes_prefix += memory_notes + "\n\n"
        if micro_notes:
            notes_prefix += micro_notes + "\n\n"
        if lesson_notes:
            notes_prefix += lesson_notes + "\n\n"
        if todo_notes:
            notes_prefix += todo_notes + "\n\n"
        mcp_notes = (mcp_mod.render(self.cfg)
                     if getattr(self.cfg, "allow_mcp", False) else "")
        named_notes = mcp_injection(self.cfg)
        mcp_notes = "\n\n".join(x for x in (mcp_notes, named_notes) if x)
        if mcp_notes:
            notes_prefix += mcp_notes + "\n\n"
        src = gathered or "(nothing was looked up)"
        need_hint = _NEED_HINT if self._rehydration_rounds_left() else ""

        def assemble(notes: str, rmap: str, source: str) -> str:
            return (notes + f"PROJECT MAP\n\n{rmap}\n\n"
                    f"SOURCE YOU REQUESTED\n\n{source}\n\n"
                    f"TASK\n{task}" + need_hint)

        est = getattr(self.client, "estimate_tokens", None) or estimate_tokens
        reply_room = min(max_tokens or self.cfg.max_tokens_answer,
                         self.cfg.request_max_tokens_cap or
                         self.cfg.max_tokens_answer)
        hosted = is_hosted(self.cfg)
        if hosted:
            # Hosted APIs have a request-size limit and bill per token.
            limit = int(getattr(self.cfg, "max_prompt_tokens", 0) or 60000)
        else:
            limit = self.cfg.context_tokens - reply_room \
                - self.cfg.reserve_tokens // 2
        user = assemble(notes_prefix, repo_map, src)
        notices: list[str] = []
        if est(user) > limit > 0:
            # 1. project notes / memory / checklist are the first to go
            if notes_prefix and not hosted:
                notes_prefix = ""
                notices.append("project notes omitted")
                user = assemble(notes_prefix, repo_map, src)
        if est(user) > limit > 0 and not hosted:
            # 2. a smaller map (the model can `outline`/`find` for the rest)
            import copy as _copy
            small = _copy.copy(self.cfg)
            for _ in range(2):
                if est(user) <= limit:
                    break
                small.flat_map_token_budget = max(
                    200, int(small.flat_map_token_budget) // 2)
                try:
                    repo_map, _t = self.index.map_for_prompt(small, task=task)
                except TypeError:
                    break
                user = assemble(notes_prefix, repo_map, src)
                notices.append(f"map shrunk to ~{small.flat_map_token_budget}"
                               " tokens")
        if est(user) > limit > 0:
            # 3. pack source by evidence utility (pinned requirement anchors,
            # diagnostics, then keyword/graph/information-per-token score)
            head = assemble(notes_prefix, repo_map, "")
            src, dropped = _trim_gathered(
                src, lambda text: est(head) + est(text) > limit,
                requirements=getattr(self, "requirements", None),
                diagnostics=getattr(self, "active_diagnostics", None),
                index=self.index)
            if dropped:
                notices.append(f"{dropped} older source item(s) trimmed")
            user = assemble(notes_prefix, repo_map, src)
        if est(user) > limit > 0 and not hosted:
            over = est(user) - limit
            raise LLMError(
                f"The answer prompt would be ~{est(user)} tokens "
                f"but this model's window only allows ~{limit} here "
                f"(context {self.cfg.context_tokens:,} minus a "
                f"{reply_room:,}-token reply budget).\n"
                "Even after dropping notes, shrinking the map and trimming "
                "source, the TASK plus the minimum context does not fit.\n"
                "Fix by lowering ONE of these in your active profile:\n"
                "  * max_gathered_chars   (less source quoted into the answer)"
                "\n  * flat_map_token_budget (smaller project map)"
                f"\n  * context_tokens       (currently {self.cfg.context_tokens:,};"
                " LM Studio's stock\n    default is 8192 -- raising its runtime "
                "'Context length' is the\n    better fix if your GPU can take it)"
                f"\n(over-budget by ~{over} tokens)")
        if notices:
            print("[answer prompt over budget; degraded: "
                  + "; ".join(notices) + "]", file=sys.stderr)
            self._emit("degrade", notices=notices)
        print(f"[answer context ~{est(user)} tokens]", file=sys.stderr)
        return self._chat_needs(
            lambda m: self.client.chat(
                m, max_tokens=max_tokens or self.cfg.max_tokens_answer,
                stream=stream),
            [{"role": "system", "content": brief},
             {"role": "user", "content": user}])

    # ------------------------------------------------------------------- edits
    def apply(self, reply: str, dry_run: bool, repair_rounds: int | None = None,
              task: str = "", gathered: str = "",
              backup_dir=None) -> tuple[list, str]:
        """Parse and apply the edits in `reply`, feeding any failures back to
        the model (up to `repair_rounds` times) so it can re-anchor its
        search text. Returns (results, final_reply_text).

        A brand-new file is created from its block as soon as the block parses;
        only a block that must match existing text needs the repair loop."""
        if repair_rounds is None:
            repair_rounds = self.cfg.repair_rounds
        self._set_site("repair")
        # Files whose blocks in this reply were no-ops (SEARCH == REPLACE) and
        # dropped by parse_edits; loop.run_task reads this to treat fabricated
        # null edits as a refusal. Reset on every call.
        self.noop_discards: list[str] = []
        parse_edits(reply, discards=self.noop_discards)
        # Truncation continuation: when the streamed reply was cut off at the
        # output cap (`stream_truncated`; `last_finish_reason` is not set for
        # streams), keep the complete blocks, name the files still to do, and
        # ask for just those. Bounded by a char budget and a round cap.
        cont_budget = 0 if dry_run else \
            max(1, int(getattr(self.cfg, "max_tokens_answer", 40000))) * 4
        cont_rounds = 0
        empty_rounds = 0
        cont_cap = max(0, int(getattr(self.cfg, "max_continuation_rounds", 0)
                              or 0))
        # Each continuation is a fresh conversation, so the source already seen
        # rides along (most recent slice if over budget); otherwise the model
        # asks to `read` again instead of writing blocks.
        cont_source = (gathered or getattr(self, "gathered_source", "") or "").strip()
        if cont_source:
            limit = max(4000, int(getattr(self.cfg, "max_gathered_chars",
                                           24000) or 24000))
            if len(cont_source) > limit:
                cont_source = ("[earlier gathered source trimmed]\n"
                               + cont_source[-limit:])
            cont_source = f"SOURCE YOU REQUESTED\n\n{cont_source}\n\n"
        while getattr(self.client, "stream_truncated", False) \
                and cont_budget > 0:
            cont_rounds += 1
            if cont_cap and cont_rounds > cont_cap:
                print(f"\n[continuation loop reached its {cont_cap}"
                      "-round cap; stopping]", file=sys.stderr)
                break
            parsed = parse_edits(reply)
            done_files = {_norm_key(b.file) for b in parsed}
            # Drop the unfinished block at the cut-off point; the model is
            # about to re-emit it, and it would poison later repair rounds.
            if getattr(self.client, "stream_truncated", False):
                trimmed = self._trim_dangling_block(reply)
                if trimmed != reply:
                    print("[dropped the unfinished block at the cut-off "
                          "point; asking for it again]", file=sys.stderr)
                    reply = trimmed
            cont_budget -= len(reply)
            if done_files:
                empty_rounds = 0
                print(f"\n[reply hit the output limit mid-way; continuing with "
                      f"{len(done_files)} file(s) done so far]", file=sys.stderr)
                prompt = (
                    f"TASK\n{task}\n\n{cont_source}"
                    f"EDITS ALREADY DELIVERED (do NOT repeat these):\n"
                    + "\n".join(f"- {f}" for f in sorted(done_files)) +
                    "\n\nYour previous reply was CUT OFF by the output "
                    "token limit. Emit ONLY the remaining search/replace "
                    "blocks for files not listed above -- at most two or "
                    "three files per reply, never a whole large file in "
                    "one REPLACE. If everything is already covered, reply "
                    "with a single line: NOTHING MORE.")
            elif empty_rounds:
                # The generic re-ask was also cut off: escalate once to a
                # single-small-chunk demand.
                print("\n[reply was cut off before any complete block again; "
                      "asking for a single small chunk]", file=sys.stderr)
                prompt = (
                    f"TASK\n{task}\n\n{cont_source}"
                    "Your replies keep getting CUT OFF by the output token "
                    "limit because they are far too long. STOP emitting the "
                    "whole file. Emit exactly ONE search/replace block that "
                    "covers only the FIRST ~30 lines of the file -- nothing "
                    "else. The remaining content is requested in later turns.")
            else:
                # Cut off before one block finished: re-ask, teaching the chunk
                # protocol for files too large for one message.
                print("\n[reply was cut off before any complete block; "
                      "re-asking for the file]", file=sys.stderr)
                prompt = (
                    f"TASK\n{task}\n\n{cont_source}"
                    "Your previous reply was CUT OFF by the output token "
                    "limit before it finished even one block, so nothing was "
                    "saved. Emit the search/replace blocks again now. If a "
                    "file is large, split it into chunks of at most ~80 lines "
                    "across successive blocks. Emit blocks only, no "
                    "explanation.")
            more = self._chat_at(
                "temperature_answer",
                [{"role": "system", "content": self._answer_sys()},
                 {"role": "user", "content": prompt}],
                max_tokens=self.cfg.max_tokens_answer,
                stream=True,
            )
            more_has_edits = bool(parse_edits(more))
            if "NOTHING MORE" in more and not more_has_edits:
                break
            if not more_has_edits:
                if getattr(self.client, "stream_truncated", False):
                    # The re-ask was also cut off: one escalated attempt, then stop.
                    empty_rounds += 1
                    if empty_rounds >= 2:
                        print("\n[repeatedly cut off before any complete "
                              "block; stopping the continuation loop]",
                              file=sys.stderr)
                        break
                    print("[re-ask was truncated too; trying once more]",
                          file=sys.stderr)
                    continue
                # No blocks at all: keep the parseable original and stop.
                break
            empty_rounds = 0
            reply = (reply.rstrip() + "\n\n" + more) if reply.strip() else more
        # Trim a dangling tail the loop above did not (a truncated first
        # reply, or the last round when the budget ran out), so repair prompts
        # never echo half a block.
        if getattr(self.client, "stream_truncated", False):
            reply = self._trim_dangling_block(reply)
        # Post-edit check baseline: the findings these files ALREADY had
        # before this call touches them, so only new problems get fed back.
        post_check = not dry_run and bool(getattr(self.cfg, "post_edit_check",
                                                  True))
        baseline: dict[str, list[str]] = {}
        if post_check:
            self._baseline_findings(baseline, parse_edits(reply))
        results = self._apply_counted(reply, dry_run, backup_dir)
        # Captured before the repair loop reassigns `reply`; used by the
        # completion loop below.
        delivered = {_norm_key(b.file) for b in parse_edits(reply)}
        deferred = bool(_MORE_TO_DO_RE.search(reply))

        for _round in range(repair_rounds):
            failures = [r for r in results if not r.ok]
            if not failures:
                break
            print(f"\n[{len(failures)} block(s) failed; asking the model to re-anchor]", file=sys.stderr)
            fb = "\n".join(f"- {r.file}: {r.message}" for r in failures)
            # Ground the repair prompt in the files on disk (the region around
            # the fuzzy-matched anchor), not the gathered buffer.
            prev_blocks = self._failed_block_texts(reply, failures)
            src_slice = self._repair_source(failures, prev_blocks)
            # A placeholder-creation refusal is a rename-protocol problem, not
            # an anchor problem; the generic "copy the SEARCH text" wording
            # would invite doomed full-content re-typing.
            rename_refusals = [r for r in failures
                               if "placeholder comment" in r.message]
            extras: list[str] = []
            if rename_refusals:
                names = ", ".join(sorted({r.file for r in rename_refusals}))
                extras.append(
                    f"RENAME PROTOCOL FOR {names}: these creations were "
                    "refused because they are placeholders with no matching "
                    "clearing block. Do NOT copy any file contents into your "
                    "reply -- a whole-file SEARCH would be cut off by the "
                    "output limit and is wasted effort. Instead emit ONLY:\n"
                    "(1) one block for each NEW path: empty SEARCH section "
                    "(nothing between <<<<<<< SEARCH and =======), REPLACE "
                    "is a single short comment line;\n"
                    "(2) one block for each OLD path being renamed away: its "
                    "SEARCH may be just the FIRST 3-5 lines of the old file "
                    "and its REPLACE must be EMPTY -- Pedacito recognises the "
                    "pair and moves the entire real file itself.\n"
                    "Keep this reply tiny: four blocks total, nothing else.")
            if any("search block not found" in r.message for r in failures):
                # An anchor miss is often a botched whole-file rewrite; offer
                # the empty-SEARCH form, which needs no quote of the old text.
                extras.append(
                    "WHOLE-FILE REWRITE: if one of these edits was meant to "
                    "replace an entire file (typically implementing a stub), "
                    "do NOT re-quote its old contents -- that is what just "
                    "failed. Emit exactly one block for that file with an "
                    "EMPTY SEARCH section (nothing between <<<<<<< SEARCH "
                    "and =======) and the complete new content in REPLACE.")
                # ... and the form that cannot miss: a line range, quoted
                # from the numbers SOURCE already shows. Mojibake, a tab and
                # a CRLF all break a SEARCH match while the same two lines
                # addressed by number apply exactly.
                extras.append(
                    "SMALL FIXES (LINES ANCHOR): when the change touches a "
                    "few lines, stop quoting them -- put the range in the "
                    "header and it replaces exactly those lines, with the "
                    "SEARCH section ignored:\n"
                    "--- FILE: path/to/file.ext LINES 40-46\n"
                    "<<<<<<< SEARCH\n=======\n"
                    "the new lines here\n"
                    ">>>>>>> REPLACE\n"
                    "The SOURCE above is numbered (`  40| text` -- those are "
                    "the real line numbers, and each header says how many "
                    "lines the file has). Use it when the SEARCH text looks "
                    "identical to the source yet will not match: something "
                    "in those bytes (a mangled character, a tab, a CR) is "
                    "not what it shows. Never re-emit a SEARCH that already "
                    "failed unchanged.")
            extra = ("\n\n" + "\n\n".join(extras)) if extras else ""
            reply = self._chat_at(
                "temperature_repair",
                [
                    {"role": "system", "content": self._answer_sys()},
                    {"role": "user", "content":
                        f"SOURCE (current contents of the files that failed)\n\n{src_slice}\n\n"
                        f"TASK\n{task}\n\n"
                        f"YOUR PREVIOUS EDIT BLOCKS THAT FAILED:\n{prev_blocks}\n\n"
                        f"These edits could not be applied:\n{fb}{extra}\n\n"
                        "Output ONLY the corrected search/replace blocks for the "
                        "failed edits -- do not repeat edits that already worked. "
                        "Except where a note above says otherwise, copy the "
                        "SEARCH text exactly from the SOURCE above, byte for "
                        "byte, keeping its real indentation (ignore the "
                        "line-number prefixes), and keep every block small. Do "
                        "not ask for more context -- the source you need is "
                        "provided."},
                ],
                max_tokens=self.cfg.max_tokens_answer,
                stream=True,
            )
            if getattr(self.client, "stream_truncated", False):
                reply = self._trim_dangling_block(reply)
            if not parse_edits(reply) \
                    and getattr(self.client, "stream_truncated", False):
                # Cut off before the first block closed: one smaller re-ask
                # (empty-SEARCH whole-file form) can fit under the cap.
                print("[repair reply was cut off mid-block; re-asking for "
                      "one small block]", file=sys.stderr)
                reply = self._chat_at(
                    "temperature_repair",
                    [
                        {"role": "system", "content": self._answer_sys()},
                        {"role": "user", "content":
                            f"SOURCE (current contents of the files that failed)\n\n{src_slice}\n\n"
                            f"TASK\n{task}\n\n"
                            "Your previous reply was CUT OFF by the output "
                            "token limit before it finished one edit block, "
                            "so nothing was saved. Emit exactly ONE "
                            "corrected block now -- nothing else. For a "
                            "whole-file rewrite, use an EMPTY SEARCH section "
                            "(nothing between <<<<<<< SEARCH and =======) "
                            "and put the new content in REPLACE; never "
                            "re-type the old content. Blocks only, no "
                            "explanation."},
                    ],
                    max_tokens=self.cfg.max_tokens_answer,
                    stream=True,
                )
                if getattr(self.client, "stream_truncated", False):
                    reply = self._trim_dangling_block(reply)
            if not parse_edits(reply):
                # Prose instead of blocks (e.g. "I need a read first"): the
                # same prompt will not end differently, so stop repairing.
                print("[repair produced no edit blocks; stopping]",
                      file=sys.stderr)
                break
            good = [r for r in results if r.ok]
            if post_check:
                self._baseline_findings(baseline, parse_edits(reply))
            results = good + self._apply_counted(reply, dry_run, backup_dir)

        # Completion loop: the answer prompt lets the model split a multi-file
        # task across turns. When a cleanly finished reply defers the rest, run
        # the next turn until it reports NOTHING MORE or emits no blocks.
        completion_cap = max(0, int(getattr(self.cfg, "max_completion_rounds", 0)
                                    or 0))
        # Token-derived budget so an endlessly deferring model stops on tokens.
        completion_budget = 0 if dry_run else \
            max(1, int(getattr(self.cfg, "max_tokens_answer", 40000))) * 4
        if not dry_run and deferred:
            rounds_done = 0
            while (completion_cap <= 0 or rounds_done < completion_cap) \
                    and completion_budget > 0:
                rounds_done += 1
                src = (gathered or "").strip()
                more = self._chat_at(
                    "temperature_answer",
                    [
                        {"role": "system", "content": self._answer_sys()},
                        {"role": "user", "content":
                            f"TASK\n{task}\n\n"
                            + (f"SOURCE YOU REQUESTED\n\n{gathered}\n\n"
                               if src else "")
                            + "EDITS ALREADY DELIVERED (do NOT repeat these):\n"
                            + "\n".join(f"- {f}" for f in sorted(delivered))
                            + "\n\nYou indicated there is still work to do. "
                            "Emit ONLY the remaining search/replace blocks "
                            "now -- a few files at most, never a whole large "
                            "file in one block. If the task is actually "
                            "complete, reply with a single line: NOTHING "
                            "MORE."},
                    ],
                    max_tokens=self.cfg.max_tokens_answer,
                    stream=True,
                )
                more_edits = parse_edits(more)
                if not more_edits:
                    break                      # NOTHING MORE, or no blocks
                completion_budget -= len(more)
                reply = (reply.rstrip() + "\n\n" + more) if reply.strip() else more
                if post_check:
                    self._baseline_findings(baseline, more_edits)
                results += self._apply_counted(more, dry_run, backup_dir)
                delivered |= {_norm_key(b.file) for b in more_edits}
                if not _MORE_TO_DO_RE.search(more):
                    break
            if completion_budget <= 0:
                print("\n[completion loop token budget exhausted; stopping]",
                      file=sys.stderr)

        # Post-edit check: cheap lint of the files this call changed, diffed
        # against their pre-edit baseline, so only problems the edits
        # introduced are fed back. Runs last: the loops above are about
        # applying blocks at all, this one is about what they did.
        if post_check:
            for _round in range(max(0, int(repair_rounds or 0))):
                changed = sorted({fileio.normalise_rel(r.file)
                                  for r in results if r.ok and r.file})
                if not changed:
                    break
                try:
                    current = diagnostics_mod.quick_findings(self.index.root,
                                                             changed)
                except Exception:
                    break
                found = diagnostics_mod.new_findings(baseline, current)
                self.active_diagnostics = found or {}
                if not found:
                    break
                n = sum(len(v) for v in found.values())
                print(f"\n[post-edit check: {n} new finding(s) in the changed "
                      "files; asking for a fix]", file=sys.stderr)
                fix = self._chat_at(
                    "temperature_repair",
                    [
                        {"role": "system", "content": self._answer_sys()},
                        {"role": "user", "content":
                            f"TASK\n{task}\n\n"
                            "Your edits applied, but a quick lint of the "
                            "changed files reports NEW problems they "
                            "introduced:\n"
                            f"{diagnostics_mod.render_findings(found)}\n\n"
                            "SOURCE (current contents around each finding)\n\n"
                            f"{self._diagnostics_source(found)}\n\n"
                            "Fix ONLY these problems with minimal search/"
                            "replace blocks -- copy the SEARCH text exactly "
                            "from the SOURCE above, byte for byte, keeping "
                            "its real indentation (ignore the line-number "
                            "prefixes). Do not repeat edits that already "
                            "worked and do not refactor anything else. If a "
                            "finding is a false positive or needs no code "
                            "change, reply with a single line: NOTHING MORE."},
                    ],
                    max_tokens=self.cfg.max_tokens_answer,
                    stream=True,
                )
                if getattr(self.client, "stream_truncated", False):
                    fix = self._trim_dangling_block(fix)
                if not parse_edits(fix):
                    break                      # NOTHING MORE, or no blocks
                # A file the fix touches for the first time gets its current
                # findings as baseline; files already baselined keep the
                # findings captured before this apply() call touched them.
                self._baseline_findings(baseline, parse_edits(fix))
                reply = (reply.rstrip() + "\n\n" + fix) if reply.strip() else fix
                results += self._apply_counted(fix, dry_run, backup_dir)
        if not dry_run and getattr(self.cfg, "auto_heal_imports", True):
            # Deterministic import auto-healing (TECHNIQUES.md #19): once the
            # edits are on disk, an unambiguous missing import is a fact, not
            # a model decision. Fix the buffer and the file here, before any
            # test/lint subprocess or repair round can spend a call on it.
            from . import verify as verify_mod
            receipts = verify_mod.auto_heal_static(
                results, root=self.index.root, cfg=self.cfg, persist=True)
            for receipt in receipts:
                print(f"  {receipt}", file=sys.stderr)
            if receipts:
                # The inserted import shifts every line below it: refresh the
                # index now so later ranges, receipts and checks name the
                # bytes actually on disk, not the pre-heal offsets.
                from . import index as index_mod
                for rel in sorted({r.file for r in results
                                   if getattr(r, "ok", False) and r.file}):
                    try:
                        index_mod.reindex_file(self.index, rel, self.cfg,
                                               save=False)
                    except Exception:
                        pass
        tracker = getattr(self, "requirements", None)
        if tracker is not None and task and getattr(tracker, "task", "") == task:
            tracker.observe_edit(results)
            if not dry_run and not any(not r.ok for r in results):
                tracker.observe_verified(
                    {r.file for r in results if r.ok and r.file})
        return results, reply

    # ------------------------------------------------------------- repair help
    @staticmethod
    def _trim_dangling_block(reply: str) -> str:
        """Drop a truncated reply's unfinished tail: everything from the last
        `--- FILE:` header (or a bare `<<<<<<< SEARCH`) that has no closing
        `>>>>>>> REPLACE`. Complete blocks are untouched."""
        if not reply:
            return reply
        heads = [m.start() for m in re.finditer(r"(?m)^---\s*FILE:", reply)]
        if not heads:
            return reply
        last = heads[-1]
        tail = reply[last:]
        if re.search(r"(?m)^>{5,}\s*REPLACE", tail):
            return reply               # final block is complete -- keep all
        cut = last
        # A bare SEARCH marker without a header can also be cut mid-block.
        m = re.search(r"(?m)^<{5,}\s*SEARCH", tail)
        if m and not re.search(r"(?m)^>{5,}\s*REPLACE", tail[m.end():]):
            cut = min(cut, last + m.start())
        trimmed = reply[:cut].rstrip()
        return trimmed + "\n" if trimmed else ""

    @staticmethod
    def _failed_block_texts(reply: str, failures) -> str:
        """Echo back the blocks the model wrote for the failed files, so it can
        see which search text missed. Anchored headers keep their suffix."""
        wanted = {_norm_key(r.file) for r in failures}
        out = []
        for e in parse_edits(reply):
            low = _norm_key(e.file)
            if any(low == w or low.endswith("/" + w) or
                   (w and low.endswith(w)) for w in wanted if w):
                head = f"--- FILE: {e.file}"
                if e.lines:
                    head += f" LINES {e.lines[0]}-{e.lines[1]}"
                elif e.symbol:
                    head += f" SYMBOL {e.symbol}"
                out.append(f"{head}\n<<<<<<< SEARCH\n{e.search}\n"
                           f"=======\n{e.replace}\n>>>>>>> REPLACE")
        return "\n\n".join(out)[:6000] or "(previous blocks unavailable)"

    def _baseline_findings(self, baseline: dict, edits) -> None:
        """Capture the lint findings each not-yet-seen file in `edits` has on
        disk right now, BEFORE any block touches it. First capture wins: a
        later round never re-baselines a file, so problems this apply() call
        introduces cannot launder themselves into the baseline."""
        fresh: list[str] = []
        for e in edits or []:
            rel = fileio.normalise_rel(e.file or "")
            if rel and rel not in baseline:
                baseline[rel] = []
                fresh.append(rel)
        if not fresh:
            return
        try:
            found = diagnostics_mod.quick_findings(self.index.root, fresh)
        except Exception:
            found = {}
        for rel in fresh:
            baseline[rel] = found.get(rel, [])

    def _diagnostics_source(self, found: dict) -> str:
        """Numbered source windows around each finding, read fresh from disk,
        so the fix prompt quotes the files as they are right now."""
        chunks = []
        for rel in sorted(found)[:6]:
            path = self.index.root / rel
            try:
                text, _style = fileio.read(path, errors="strict")
            except (OSError, UnicodeDecodeError):
                continue
            lines = text.split("\n")
            spans: list[tuple[int, int]] = []
            for it in (found[rel] or [])[:12]:
                m = re.search(r":(\d+)(?::\d+)?[:\s]", it) \
                    or re.search(r"line (\d+)", it)
                if not m:
                    continue
                ln = int(m.group(1))
                if 1 <= ln <= len(lines):
                    spans.append((max(0, ln - 5),
                                  min(len(lines), ln + 4)))  # ±4, 0-based
            if not spans:
                spans = [(0, min(len(lines), 20))]
            shown: list[str] = []
            for s, e in _fold_spans(spans):
                shown.append(f"... lines {s + 1}-{e} ...")
                shown.extend(f"{i + 1:>5}| {lines[i]}" for i in range(s, e))
            chunks.append(f"FILE: {rel}\n" + "\n".join(shown))
        return "\n\n".join(chunks)[:8000] or "(source unavailable)"

    def _repair_source(self, failures, prev_blocks: str = "") -> str:
        """Current source of the failed files, read fresh from disk: the lines
        around the fuzzy-located previous SEARCH text, else the file head up
        to max_read_lines, else what gather captured."""
        chunks = []
        seen: set[str] = set()
        for r in failures:
            rel = fileio.normalise_rel(r.file or "")
            low = rel.lower()
            if not rel or low in seen:
                continue
            seen.add(low)
            path = self.index.root / rel
            text = None
            if path.is_file():
                try:
                    text, _style = fileio.read(path, errors="strict")
                except (OSError, UnicodeDecodeError):
                    text = None
            if text is None:
                # Not on disk: fall back to the gathered buffer.
                for item in (self.gathered_source or "").split("\n\n"):
                    m = re.match(r"--- \w+ (\S+)", item)
                    g = fileio.path_key(m.group(1)) if m else ""
                    if g and (g == low or g.endswith("/" + low) or
                              low.endswith("/" + g)):
                        chunks.append(item[:8000])
                        break
                continue
            lines = text.split("\n")
            start, end = 0, min(len(lines), int(getattr(self.cfg, "max_read_lines", 400)))
            span = None
            for blk in prev_blocks.split("--- FILE:")[1:]:
                m = re.search(r"<<<<<<< SEARCH\n(.*?)\n=======", blk, re.DOTALL)
                if not m:
                    continue
                needle = m.group(1)
                if rel.lower() not in blk.lower().split("<<<<<<<")[0]:
                    continue
                span = locate(text, needle, self.cfg)
                if span:
                    break
            if span:
                hit_line = text.count("\n", 0, span.start)
                start = max(0, hit_line - 30)
                end = min(len(lines), hit_line + 60)
                end = max(end, min(start + 60, len(lines)))
            numbered = "\n".join(f"{i:>5}| {lines[i]}"
                                 for i in range(start, end))
            header = f"FILE: {rel} ({len(lines)} lines total; showing {start + 1}-{end})"
            chunks.append(header + "\n" + numbered[:8000])
        joined = "\n\n".join(chunks)
        return joined or "(source unavailable -- the failed files are new; " \
                         "output the FULL intended file contents in one block)"

    def _apply_counted(self, reply: str, dry_run: bool, backup_dir=None) -> list:
        """`_apply_once` for apply()'s rounds, counting SEARCH blocks that
        matched nowhere (the eval's `search_mismatches`)."""
        results = self._apply_once(reply, dry_run, backup_dir)
        self.search_mismatches += sum(
            1 for r in results
            if not r.ok and "search block not found" in r.message)
        return results

    def _apply_once(self, reply: str, dry_run: bool, backup_dir=None) -> list:
        """One parse-and-apply pass, with no repair round-trip. Duplicate FILE
        headers are not merged: apply_edits groups blocks by path and needs
        them separate to detect renames. SYMBOL anchors are resolved to line
        ranges here from the index when possible; anything the index misses is
        resolved from the AST at apply time (`apply_edits` owns the fallback),
        and zero-anchor SYMBOL/AFTER_SYMBOL blocks skip SEARCH quoting
        entirely."""
        discards: list = []
        parsed = parse_edits(reply, discards=discards)
        failed, edits = self._resolve_symbol_anchors(parsed)
        # A block the model started but never finished is not silently
        # ignored: it becomes a failed result the repair loop reports back.
        if not getattr(self.client, "stream_truncated", False):
            for f, hint in malformed_blocks(reply, len(parsed) + len(discards)):
                failed.append(Result(False, f, f"malformed edit block: {hint}"))
        if edits and not dry_run:
            # never edit a file that is mid-merge: the conflict markers are
            # the user's to resolve, and an edit would bury them
            try:
                conflicted = set(gitinfo.status(self.index.root)["conflicts"]) \
                    if gitinfo.is_repo(self.index.root) else set()
            except Exception:
                conflicted = set()
            if conflicted:
                blocked = [e for e in edits
                           if fileio.normalise_rel(e.file) in conflicted]
                for f in dict.fromkeys(e.file for e in blocked):
                    failed.append(Result(
                        False, f, "refused: this file has unresolved git merge "
                                  "conflicts; resolve them first"))
                edits = [e for e in edits if e not in blocked]
        if edits and not dry_run and getattr(self.cfg, "atomic_edits", False):
            # All or nothing: rehearse first, and write nothing when any
            # file's edit (or any block already rejected above) would fail,
            # so a coupled multi-file change is never left half-applied.
            trial = apply_edits(self.index.root, edits, backup=False,
                                dry_run=True, cfg=self.cfg,
                                allowlist=self._write_allowlist())
            bad = [r for r in trial if not r.ok]
            if bad or failed:
                held = [Result(False, r.file,
                               "not written: atomic_edits is on and another "
                               "file's edit failed; fix that one and resend "
                               "ALL the blocks")
                        for r in trial if r.ok]
                return failed + bad + held
        if edits:
            failed += apply_edits(self.index.root, edits,
                                  backup=self.cfg.backup, dry_run=dry_run,
                                  backup_dir=backup_dir, cfg=self.cfg,
                                  allowlist=None if dry_run
                                  else self._write_allowlist())
        return failed

    def _write_allowlist(self):
        """Normalised paths this task may modify when cfg.auto_allowlist is
        on: the files its title names plus the files gather actually read.
        New creations stay allowed (apply_edits only gates existing files).
        None disables the gate."""
        if not getattr(self.cfg, "auto_allowlist", False):
            return None
        reads = getattr(self.cfg, "_task_reads", None)
        if reads is None:
            return None          # no gather ever armed the ledger
        allowed = set(getattr(self.cfg, "_task_allowlist", None) or ())
        allowed |= set(reads)
        return allowed

    def _resolve_symbol_anchors(self, edits: list) -> tuple[list, list]:
        """Turn `SYMBOL name` anchors into LINES ranges via the index,
        falling back to `chunker.locate_symbol_span` (AST / tree-sitter on the
        current file bytes) when the index misses. Returns (failure Results,
        edits left to apply). A file with a missing or ambiguous symbol fails
        whole -- mirroring apply_edits' per-file failure -- and the message
        feeds the repair loop."""
        results: list = []
        bad: set[str] = set()
        for e in edits:
            if not e.symbol or e.lines:
                continue
            want = _norm_key(e.symbol)
            same = [c for c in self.index.find_symbol(e.symbol)
                    if _norm_key(c.file) == _norm_key(e.file)]
            # find_symbol also matches via class members; only a name that
            # really ends with the symbol may be replaced.
            exact = [c for c in same
                     if _norm_key(c.qualname) == want
                     or _norm_key(c.qualname).endswith("." + want)]
            if len(exact) == 1:
                c = exact[0]
                # A classheader chunk stands for the whole class.
                e.lines = (c.start, c.scope_end or c.end)
                continue
            # Symbol-splice fallback: the index missed it (a method inside a
            # whole-class chunk, a file not indexed yet, a stale line map).
            # Ask the AST (Python) or tree-sitter directly where the symbol
            # sits in the CURRENT bytes before failing the whole file.
            from .chunker import locate_symbol_span
            disk = locate_symbol_span(self.index.root / e.file, e.symbol)
            if disk is not None:
                e.lines = (disk[0], disk[1])
                continue
            bad.add(_norm_key(e.file))
            if not same:
                why = ("no definition by that name in this file; check the "
                       "spelling against an `outline`")
            elif not exact:
                why = ("no definition NAMED that in this file (it only "
                       "appears inside another unit); anchor with LINES "
                       "<start>-<end> from a `read`")
            else:
                listing = ", ".join(f"{c.qualname} L{c.start}-{c.end}"
                                    for c in exact[:5])
                why = (f"{len(exact)} definitions match ({listing}); name "
                       "one exactly or anchor with LINES <start>-<end>")
            results.append(Result(
                False, e.file,
                f"SYMBOL anchor '{e.symbol}': {why}"))
        if bad:
            edits = [e for e in edits if _norm_key(e.file) not in bad]
        return results, edits


def _compact(step: dict) -> str:
    """Render a parsed step with empty fields dropped, to echo the model's
    action into the conversation history."""
    keep = {k: v for k, v in step.items() if v not in ("", 0, None)}
    return str(keep)
