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
until the model says `done` or the gathered-source budget is exhausted; a
positive max_steps is a hard limit. When the model says `done` but files the
task names were never read, one nudge asks it to read them (repeated only
after real progress).

Phase 2 (answer): free text, no grammar. Code is not forced through a JSON
string field because escaping degrades small-model output; edits are fenced
search/replace blocks instead (see edits.py).

The system message and repo map come first and stay constant within a gather
loop, so the server's prefix cache hits on every turn.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys

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
from .tools import TOOL_HELP, dispatch, schema_for

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
_DEDUP_STEP_ACTIONS = {"grep", "find", "outline", "callers", "sessions",
                       "diff", "diagnose"}

# Read-only steps whose repeat (even many steps later) adds nothing while the
# index, the conversation and the working tree are unchanged.
_CACHEABLE_ACTIONS = {"grep", "find", "outline", "callers", "context", "git",
                      "diagnose", "sessions", "fetch"}
# Steps that can change what the cacheable ones return.
_MUTATING = {"write", "append", "move", "delete", "undo", "rename", "run",
             "browser", "screenshot", "mcp", "task"}

# Steps discarded for a cut-off payload before gather gives up.
_MAX_PAYLOAD_CUTS = 2


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
  It returns the page URL/title/text, canvas sizes, console errors, and a
  screenshot.
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


_TOOLISH_RE = re.compile(
    r"^\s*\{[^}]*['\"]action['\"]\s*:|<tool_call|<function\s*=|<invoke\b",
    re.IGNORECASE)


def _unusable_continuation(reply: str) -> bool:
    """A continued answer that is empty or still shaped like a tool call (the
    model imitating its own earlier JSON actions) must be rebuilt."""
    text = (reply or "").strip()
    return not text or bool(_TOOLISH_RE.match(text))


_ITEM_SPLIT = re.compile(r"\n\n(?=--- )")


def _trim_gathered(src: str, too_big) -> tuple[str, int]:
    """Drop the OLDEST whole gathered-source items until `too_big(text)` is
    false, leaving a re-read pointer per dropped item and always keeping the
    final digest. A single oversize item is clipped from its front on a line
    boundary as a last resort. Returns (text, items_dropped)."""
    items = _ITEM_SPLIT.split(src)
    dropped: list[str] = []
    keep_tail = [i for i in items if i.startswith("--- digest")]
    body = [i for i in items if not i.startswith("--- digest")]

    def render(b: list[str]) -> str:
        parts = b + keep_tail
        text = "\n\n".join(parts)
        if dropped:
            pointers = "; ".join(dropped[:8]) + ("; ..." if len(dropped) > 8
                                                  else "")
            text = ("[older gathered source trimmed to fit the request limit "
                    f"(re-`read` if you need it again: {pointers})]\n\n"
                    + text)
        return text

    while too_big(render(body)) and len(body) > 1:
        gone = body.pop(0)
        head = gone.split("\n", 1)[0][4:].strip()
        dropped.append(head[:70])
    text = render(body)
    if too_big(text) and body:
        # one item is still too big: keep its tail, whole lines only
        only = body[0]
        n_dropped = 0
        while too_big(render([only])) and len(only) > 4000:
            keep = max(4000, len(only) * 3 // 4)
            cut = only[len(only) - keep:]
            nl = cut.find("\n")
            only = cut[nl + 1:] if 0 <= nl < len(cut) - 1 else cut
            n_dropped += 1
        if n_dropped and not dropped:
            dropped.append("the earliest part of one long read")
        body = [only]
        text = render(body)
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
        # Telemetry sink (events.EventLog); bound on first gather.
        self.events = None
        # Gather conversation kept for answer continuation (see answer()).
        self._messages: list[dict] = []
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

    def _set_site(self, site: str) -> None:
        """Label the client's current call site for usage accounting."""
        try:
            self.client.site = site
        except Exception:
            pass

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

    def _prime_fewshot(self) -> None:
        """Worked step examples from THIS project for the client to show when
        the server cannot enforce a JSON grammar: a small model copies a
        concrete example far better than a dummy `{"action": "read", "file":
        ""}`. With `fewshot_steps` on, gather puts the same examples in the
        first user turn instead, so the client falls back to its one dummy
        example rather than repeating them."""
        if not hasattr(self.client, "fewshot"):
            return
        self.client.fewshot = [] if fewshot_enabled(self.cfg) \
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

    def _fit(self, messages: list[dict]) -> bool:
        """Keep the live conversation inside the server's window. Only acts on
        REAL overflow: it folds the older band into one STATE digest (one
        cache break; see fold.py), and falls back to the legacy per-turn clip
        only if the recent tail alone is still too big. Returns True when it
        changed anything."""
        cfg = self.cfg
        est = getattr(self.client, "estimate_tokens", None) or estimate_tokens
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
                    self._after_fold(messages, est, before)
                    changed = True
                if estimate_messages(messages, est) <= room or keep <= 1:
                    break
                keep -= 1
        if estimate_messages(messages, est) > room:
            if _evict_overflow_turns(messages, self.client, cfg):
                self._lossy = True
                changed = True
        return changed

    def _after_fold(self, messages: list[dict], est, before: int) -> None:
        """Bookkeeping (and the optional model-written STATE) after a fold."""
        self.fold_count += 1
        self._lossy = True
        if getattr(self.cfg, "compaction", "") == "model" and len(messages) > 3:
            state = messages[3]["content"] if messages[3]["role"] == "user" \
                else ""
            better = self._model_state(state)
            if better:
                messages[3] = {"role": "user", "content": better}
        self._emit("fold", n=self.fold_count, tokens_before=before,
                   tokens_after=estimate_messages(messages, est))
        self._hook("post_compact", folds=self.fold_count)
        print("[older lookups folded into a STATE digest to fit the model's "
              "context window]", file=sys.stderr)

    def _model_state(self, state_msg: str) -> str:
        """One small call that rewrites the extractive STATE into goal /
        findings / next-step lines (cfg.compaction = "model"); '' on any
        failure so the deterministic digest stays."""
        from . import fold as fold_mod_
        body = state_msg[len(fold_mod_.STATE_HEAD):].strip()
        if not body:
            return ""
        try:
            self._set_site("compaction")
            text = self.client.chat(
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
            self._set_site("gather")
        text = (text or "").strip()
        return (fold_mod_.STATE_HEAD + "\n" + text[:1600]) if text else ""

    # ------------------------------------------------------------------ phase 1
    def gather(self, task: str, verbose: bool = True,
               on_step=None, base: str | None = None,
               backup_dir=None) -> str:
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
        if self.events is None:
            self.events = events_mod.get(root, self.cfg)
            events_mod.attach(self.client, self.events)
        self._set_site("gather")
        self._prime_fewshot()
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
        if "skill" in schema_for(self.cfg)["properties"]["action"]["enum"]:
            skill_notes = skills_mod.catalog(root, self.cfg, task=task)
            if skill_notes:
                system += "\n\n" + skill_notes
        if todo_notes:
            system += "\n\n" + todo_notes
        if getattr(self.cfg, "allow_mcp", False):
            mcp_notes = mcp_mod.render(self.cfg)
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
        self._messages = messages
        self._lossy = False          # True once history was folded/evicted
        self.fold_count = 0
        self._gather_task = task
        self._rules_shown: set[str] = set()
        gathered: list[str] = []
        self.steps_log: list[str] = []
        self.live_evidence_ok = False
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
                                          budget, nudge, verbose)

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
        while unlimited or step < step_cap:
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
            try:
                stepjson = self.client.chat_json(messages, step_schema,
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
            if getattr(self.client, "step_truncated", False):
                tf = getattr(self.client, "truncated_field", "")
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
                self._fit(messages)
                if verbose:
                    print(f"  [step {step} discarded: `{cut}` payload cut "
                          f"off ({payload_cuts}/{_MAX_PAYLOAD_CUTS}); NOT "
                          f"executed]", file=sys.stderr)
                if payload_cuts >= _MAX_PAYLOAD_CUTS or budget <= 0:
                    if verbose:
                        print("  [gather stopped: payloads keep arriving "
                              "cut off; moving to answer]", file=sys.stderr)
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
                if getattr(self.client, "step_truncated", False) \
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
                    self._fit(messages)
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
                    if (action in ("run", "screenshot", "browser")
                            and not obs.startswith("ERROR")):
                        self.live_evidence_ok = True
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
            if getattr(self.client, "step_truncated", False):
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
            if action in ("read", "context") and not obs.startswith("ERROR"):
                # A read is progress only when it shows lines not already in the
                # span ledger; an exact duplicate is replaced by a compact note
                # so the model does not lock onto one range.
                m = _READ_HEAD_RX.match(obs)
                if m:
                    key = _norm_key(m.group(1))
                    s_, e_ = int(m.group(2)), int(m.group(3))
                    prior = list(spans.get(key, []))
                    read_delta = _delta_len(obs, prior)
                    duplicate_read = (action == "read"
                                      and not _read_has_new_lines(obs, prior))
                    spans.setdefault(key, []).append((s_, e_))
                    if duplicate_read:
                        obs = (f"{m.group(1).strip()} lines {s_}-{e_}: "
                               "[already read; this range contains no new "
                               "source. Do not repeat it -- choose `done` or "
                               "a different file/range.]")
                if not duplicate_read:
                    did_read = True
                    read_nudged = False     # progress re-arms the coverage nudge
                    if m:
                        obs += self._scoped_rules(m.group(1).strip())
                self._unread_files = self._compute_unread(self._targets_text)
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
                self._fit(messages)
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
                continue
            error_streak = 0
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

            self._emit("step", n=step, action=action, detail=str(detail)[:120],
                       obs_chars=len(obs), duplicate=duplicate_read)
            messages.append({"role": "assistant", "content": _compact(stepjson)})
            messages.append({"role": "user",
                             "content": f"RESULT:\n{obs}\n\n{nudge(step + 1)}"})

            # The live conversation can outgrow the server's window even when
            # the char budget has room (some servers reject the request,
            # others silently drop the middle of the prompt): fold the older
            # band ONCE (see fold.py) instead of trimming turn by turn.
            self._fit(messages)

            if budget <= 0:
                if verbose:
                    print("  [context budget exhausted; moving to answer]", file=sys.stderr)
                break

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
                if r == w or r.endswith("/" + w) or w.endswith("/" + r):
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

    def _prefetch_step0(self, task: str, messages: list, gathered: list,
                        spans: dict, budget: int, nudge, verbose: bool) -> int:
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
        est = getattr(self.client, "estimate_tokens", None) or estimate_tokens
        room = max(1, self._room()) // 4        # <= 25% of the observation room

        wanted = self._task_file_targets(task)
        named: dict[str, str] = {}              # normalised key -> canonical rel
        for rel in self.index.files:
            r = _norm_key(rel)
            for w in wanted:
                if r == w or r.endswith("/" + w) or w.endswith("/" + r):
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
            m = _READ_HEAD_RX.match(obs)
            if m:
                spans.setdefault(_norm_key(m.group(1)), []).append(
                    (int(m.group(2)), int(m.group(3))))
                delta = _delta_len(obs, [])
            else:
                delta = len(obs)
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
            fixed = self._repair_from_findings(task, reply, errors, results)
            if not fixed:
                break
            reply = fixed
        if verbose:
            for f in findings:
                if f.severity == "warn":
                    print("  " + f.render(), file=sys.stderr)
        return reply, findings

    def _repair_from_findings(self, task: str, reply: str, errors,
                              results) -> str:
        """One re-ask that shows the model its own edits plus the check's
        findings; '' when it produced no usable blocks -- said out loud, so
        a give-up mid-repair never reads as a clean run."""
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
        try:
            new = self._chat_at(
                "temperature_repair",
                [{"role": "system", "content": system},
                 {"role": "user", "content":
                  (f"SOURCE (current contents of the files involved)\n\n{src}\n\n"
                   if src else "")
                  + f"TASK\n{task}\n\n"
                  f"YOUR PROPOSED EDIT BLOCKS:\n{blocks}\n\n"
                  "An automatic check of these edits found problems:\n"
                  + verify_mod.render(errors) + "\n\n"
                  "Output the COMPLETE corrected set of search/replace blocks "
                  "for the whole task (not only the changed ones). Fix each "
                  "problem, or -- if a problem means you are editing code you "
                  "never read -- drop that edit and say what you could not "
                  "verify. Copy SEARCH text exactly from the SOURCE above. "
                  "Blocks only after a one-line explanation."}],
                client=client, max_tokens=self.cfg.max_tokens_answer,
                stream=False)
        except LLMError as e:
            print("  [repair: re-ask failed (%s); failing checks remain]"
                  % type(e).__name__, file=sys.stderr)
            return ""
        if not parse_edits(new):
            print("  [repair: re-ask produced no usable fix; failing checks "
                  "remain]", file=sys.stderr)
            return ""
        return new

    def _model_verify(self, task: str, reply: str, gathered: str):
        """The optional adversarial verifier call (verify_answer = "full")."""
        from . import verify as verify_mod
        est = getattr(self.client, "estimate_tokens", None) or estimate_tokens
        limit = max(1500, int((self.cfg.context_tokens - 900) * 0.5))
        src, _ = _trim_gathered(gathered or "(nothing was looked up)",
                                lambda t: est(t) > limit)
        blocks = self._failed_block_texts(
            reply, [type("F", (), {"file": e.file})() for e in parse_edits(reply)])
        client = self._role_client("verifier")
        try:
            text = client.chat(
                [{"role": "system", "content": verify_mod.VERIFIER_SYSTEM},
                 {"role": "user", "content":
                  f"TASK\n{task}\n\nSOURCE\n{src}\n\nPROPOSED EDITS\n{blocks}"}],
                max_tokens=350, stream=False)
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
                 + "ANSWER INSTRUCTIONS\n" + brief)
        msgs.append({"role": "user", "content": final})
        est = getattr(self.client, "estimate_tokens", None) or estimate_tokens
        print(f"[answer continues the lookup conversation, "
              f"~{estimate_messages(msgs, est)} tokens, prefix cached]",
              file=sys.stderr)
        try:
            return self.client.chat(
                msgs, max_tokens=max_tokens or self.cfg.max_tokens_answer,
                stream=stream)
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
        notes_prefix = ""
        if project_notes:
            notes_prefix += project_notes + "\n\n"
        if memory_notes:
            notes_prefix += memory_notes + "\n\n"
        if todo_notes:
            notes_prefix += todo_notes + "\n\n"
        if getattr(self.cfg, "allow_mcp", False):
            mcp_notes = mcp_mod.render(self.cfg)
            if mcp_notes:
                notes_prefix += mcp_notes + "\n\n"
        src = gathered or "(nothing was looked up)"

        def assemble(notes: str, rmap: str, source: str) -> str:
            return (notes + f"PROJECT MAP\n\n{rmap}\n\n"
                    f"SOURCE YOU REQUESTED\n\n{source}\n\n"
                    f"TASK\n{task}")

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
            # 3. oldest whole source items out, with re-read pointers
            head = assemble(notes_prefix, repo_map, "")
            src, dropped = _trim_gathered(
                src, lambda text: est(head) + est(text) > limit)
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
        return self.client.chat(
            [{"role": "system", "content": brief},
             {"role": "user", "content": user}],
            max_tokens=max_tokens or self.cfg.max_tokens_answer,
            stream=stream,
        )

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
        ranges here: the agent owns the index, apply_edits does not."""
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
        """Turn `SYMBOL name` anchors into LINES ranges via the index.
        Returns (failure Results, edits left to apply). A file with a missing
        or ambiguous symbol fails whole -- mirroring apply_edits' per-file
        failure -- and the message feeds the repair loop."""
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
