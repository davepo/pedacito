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

import re
import sys

from . import diagnostics as diagnostics_mod
from . import fileio
from . import instructions as instructions_mod
from . import mcp as mcp_mod
from . import todo as todo_mod
from .config import is_hosted
from .edits import Result, apply_edits, locate, parse_edits
from .llm import LLMError, estimate_tokens
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

# Steps discarded for a cut-off payload before gather gives up.
_MAX_PAYLOAD_CUTS = 2


def _step_signature(step: dict) -> tuple:
    """Identity of a gather step for repetition detection. `reasoning` is
    excluded: the same action with a reworded note is still the same loop."""
    return tuple(sorted(
        (k, str(v)) for k, v in (step or {}).items()
        if k != "reasoning"))


def _norm_key(path: str) -> str:
    """Normalise a read-ledger path. Models alternate between `pkg/a.py` and
    `./pkg/a.py`; the ledger, eviction pass and coverage must share one key."""
    return (path or "").strip().lower().lstrip("./")


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
  click #start`, and `start` to the seconds to wait before capture. A `wait`
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
        repo_map, tier = self.index.map_for_prompt(self.cfg)
        system = (base if base is not None else GATHER_BASE) + \
            (FLAT_HINT if tier == "flat"
             else CARDS_HINT if tier == "cards" else DIRS_HINT)
        project_notes = instructions_mod.render(self.index.root, self.cfg)
        todo_notes = todo_mod.render(self.index.root, self.cfg,
                                     limit=int(getattr(
                                         self.cfg, "max_todo_chars", 1500)))
        if project_notes:
            system += "\n\n" + project_notes
        if todo_notes:
            system += "\n\n" + todo_notes
        if getattr(self.cfg, "allow_mcp", False):
            mcp_notes = mcp_mod.render(self.cfg)
            if mcp_notes:
                system += "\n\n" + mcp_notes
        if verbose:
            print(f"  [map: {tier}, ~{estimate_tokens(repo_map)} tokens, "
                  f"{len(self.index.files)} files]", file=sys.stderr)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"PROJECT MAP\n\n{repo_map}\n\nTASK\n{task}"},
        ]
        gathered: list[str] = []
        self.steps_log: list[str] = []
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

        step = 0
        step_schema = schema_for(self.cfg)
        done_confirmed = False   # one-shot "are you really done?" after a truncated reply
        step_recoveries = 0      # re-asks after a step died at the output cap
        duplicate_reads = 0      # consecutive reads that added no new lines
        error_streak = 0         # consecutive failed (ERROR) observations
        last_sig = None          # previous step's action+args, for loops
        repeated = 0
        payload_cuts = 0         # steps discarded for a cut-off payload
        while unlimited or step < step_cap:
            step += 1
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
                _evict_overflow_turns(messages, self.client, self.cfg)
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

            if repeated >= _MAX_REPEATED_STEPS:
                if verbose:
                    print(f"  [gather stopped: the model repeated the same "
                          f"`{action}` step {repeated} times; moving to "
                          f"answer]", file=sys.stderr)
                break

            dedup_step = is_repeat and action in _DEDUP_STEP_ACTIONS
            if dedup_step:
                # Identical to the previous (read-only) step: skip the dispatch.
                obs = (f"[This exact `{action}` step already ran and its "
                       "result is above. Repeating it cannot add information "
                       "-- choose a different query/path or say done.]")
            else:
                try:
                    obs = dispatch(self.index, self.cfg, stepjson,
                                   client=self.client)
                except Exception as e:
                    # A tool failure becomes an ERROR observation, not a crash.
                    print(f"  [{action} tool failed: "
                          f"{type(e).__name__}: {e}]", file=sys.stderr)
                    obs = (f"ERROR: {action} failed "
                           f"({type(e).__name__}): {str(e)[:200]}")
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
            if action == "read" and not obs.startswith("ERROR"):
                # A read is progress only when it shows lines not already in the
                # span ledger; an exact duplicate is replaced by a compact note
                # so the model does not lock onto one range.
                m = _READ_HEAD_RX.match(obs)
                if m:
                    key = _norm_key(m.group(1))
                    s_, e_ = int(m.group(2)), int(m.group(3))
                    prior = list(spans.get(key, []))
                    read_delta = _delta_len(obs, prior)
                    duplicate_read = not _read_has_new_lines(obs, prior)
                    spans.setdefault(key, []).append((s_, e_))
                    if duplicate_read:
                        obs = (f"{m.group(1).strip()} lines {s_}-{e_}: "
                               "[already read; this range contains no new "
                               "source. Do not repeat it -- choose `done` or "
                               "a different file/range.]")
                if not duplicate_read:
                    did_read = True
                    read_nudged = False     # progress re-arms the coverage nudge
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
                messages.append({"role": "assistant", "content": _compact(stepjson)})
                messages.append({"role": "user",
                                 "content": f"{obs}\n\n{nudge(step + 1)}"})
                # Charge errors too, so a model repeating one failing step
                # cannot loop forever at max_steps = 0.
                budget -= len(obs)
                _evict_overflow_turns(messages, self.client, self.cfg)
                if error_streak >= _MAX_CONSECUTIVE_ERRORS:
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
                    if verbose:
                        print("  [gather stopped: the model repeated the same "
                              "read without new source; moving to answer]",
                              file=sys.stderr)
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

            messages.append({"role": "assistant", "content": _compact(stepjson)})
            messages.append({"role": "user",
                             "content": f"RESULT:\n{obs}\n\n{nudge(step + 1)}"})

            # The live conversation can outgrow the server's window even when
            # the char budget has room (some servers reject the request,
            # others silently drop the middle of the prompt); trim first.
            _evict_overflow_turns(messages, self.client, self.cfg)

            if budget <= 0:
                if verbose:
                    print("  [context budget exhausted; moving to answer]", file=sys.stderr)
                break

        self.read_any = did_read
        # Rewrite RESULT turns and SOURCE items down to the lines each read
        # was first to show, so the answer phase sees every line once.
        new_items = _evict_duplicate_reads(messages, gathered, spans)
        digest = self._read_digest(spans)
        if digest:
            new_items = new_items + [digest]
        self.gathered_source = "\n\n".join(new_items)
        # Nudge turns can push the conversation over the window between reads.
        _evict_overflow_turns(messages, self.client, self.cfg)
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

    # ------------------------------------------------------------------ phase 2
    def answer(self, task: str, gathered: str, stream: bool = True,
               system: str | None = None, max_tokens: int | None = None,
               include_edit_format: bool = True) -> str:
        """Ask the model to answer the task, or produce search/replace edits,
        given the project map and whatever source `gather()` collected.

        `system` and `max_tokens` override the defaults for flows that need a
        different brief (e.g. scaffold.py). `include_edit_format=False` drops
        the search/replace format spec even when `system` is given, for
        prose-only flows (`explain`)."""
        repo_map, _ = self.index.map_for_prompt(self.cfg)
        project_notes = instructions_mod.render(self.index.root, self.cfg)
        todo_notes = todo_mod.render(self.index.root, self.cfg,
                                     limit=int(getattr(
                                         self.cfg, "max_todo_chars", 1500)))
        notes_prefix = ""
        if project_notes:
            notes_prefix += project_notes + "\n\n"
        if todo_notes:
            notes_prefix += todo_notes + "\n\n"
        if getattr(self.cfg, "allow_mcp", False):
            mcp_notes = mcp_mod.render(self.cfg)
            if mcp_notes:
                notes_prefix += mcp_notes + "\n\n"
        user = (
            notes_prefix +
            f"PROJECT MAP\n\n{repo_map}\n\n"
            f"SOURCE YOU REQUESTED\n\n{gathered or '(nothing was looked up)'}\n\n"
            f"TASK\n{task}"
        )
        # Hard fit check: a prompt over the window is rejected by some servers
        # and silently truncated from the middle by others, which would cut
        # off the TASK at the end. Refuse loudly instead.
        est = getattr(self.client, "estimate_tokens", None) or estimate_tokens
        reply_room = min(self.cfg.max_tokens_answer,
                         self.cfg.request_max_tokens_cap or
                         self.cfg.max_tokens_answer)
        hosted = is_hosted(self.cfg)
        if hosted:
            # Hosted APIs have a request-size limit and bill per token: drop
            # the OLDEST gathered source until the prompt fits a ceiling.
            head = notes_prefix + f"PROJECT MAP\n\n{repo_map}\n\n"
            tail = f"\n\nTASK\n{task}"
            ceil_ = int(getattr(self.cfg, "max_prompt_tokens", 0) or 60000)
            src = gathered or "(nothing was looked up)"
            dropped = 0
            while est(head + src + tail) > ceil_ and len(src) > 4000:
                keep = max(4000, len(src) * 3 // 4)
                dropped += len(src) - keep
                src = src[-keep:]
            if dropped:
                src = ("[older gathered source trimmed to fit the request "
                       f"limit ({dropped} chars dropped -- re-read a file if "
                       "you need it again)]\n\n" + src)
            user = head + f"SOURCE YOU REQUESTED\n\n{src}\n" + tail
        room = self.cfg.context_tokens - reply_room - self.cfg.reserve_tokens // 2
        if not hosted and est(user) > room > 0 and notes_prefix:
            # Project instructions must never turn a fitting prompt into a
            # hard failure: drop them first.
            user = user[len(notes_prefix):]
            print("[project context (instructions/checklist) omitted: answer "
                  "prompt is over the window budget]", file=sys.stderr)
        if not hosted and est(user) > room > 0:
            over = est(user) - room
            raise LLMError(
                f"The answer prompt would be ~{est(user)} tokens "
                f"but this model's window only allows ~{room} here "
                f"(context {self.cfg.context_tokens:,} minus a "
                f"{reply_room:,}-token reply budget).\n"
                "Sending it anyway means the server truncates the TASK itself "
                "-- the model answers without ever seeing your request.\n"
                "Fix by lowering ONE of these in your active profile:\n"
                "  * max_gathered_chars   (less source quoted into the answer)"
                "\n  * flat_map_token_budget (smaller project map)"
                f"\n  * context_tokens       (currently {self.cfg.context_tokens:,};"
                " LM Studio's stock\n    default is 8192 -- raising its runtime "
                "'Context length' is the\n    better fix if your GPU can take it)"
                f"\n(over-budget by ~{over} tokens)")
        print(f"[answer context ~{est(user)} tokens]", file=sys.stderr)
        sys_prompt = ANSWER_SYSTEM
        if system is not None:
            sys_prompt = system.rstrip()
            if include_edit_format:
                sys_prompt += "\n\n" + EDIT_FORMAT
        return self.client.chat(
            [{"role": "system", "content": sys_prompt},
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
            more = self.client.chat(
                [{"role": "system", "content": ANSWER_SYSTEM},
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
        results = self._apply_once(reply, dry_run, backup_dir)
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
            reply = self.client.chat(
                [
                    {"role": "system", "content": ANSWER_SYSTEM},
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
                reply = self.client.chat(
                    [
                        {"role": "system", "content": ANSWER_SYSTEM},
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
            results = good + self._apply_once(reply, dry_run, backup_dir)

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
                more = self.client.chat(
                    [
                        {"role": "system", "content": ANSWER_SYSTEM},
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
                results += self._apply_once(more, dry_run, backup_dir)
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
                fix = self.client.chat(
                    [
                        {"role": "system", "content": ANSWER_SYSTEM},
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
                results += self._apply_once(fix, dry_run, backup_dir)
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
                    g = m.group(1).lower().lstrip("./") if m else ""
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

    def _apply_once(self, reply: str, dry_run: bool, backup_dir=None) -> list:
        """One parse-and-apply pass, with no repair round-trip. Duplicate FILE
        headers are not merged: apply_edits groups blocks by path and needs
        them separate to detect renames. SYMBOL anchors are resolved to line
        ranges here: the agent owns the index, apply_edits does not."""
        failed, edits = self._resolve_symbol_anchors(parse_edits(reply))
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
