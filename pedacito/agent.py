"""The two-phase agent: gather relevant source, then answer or edit.

Usage
-----
    from .agent import Agent

    agent = Agent(index, llm_client, cfg)
    gathered = agent.gather("add retry logic to the sync client")
    reply = agent.answer(task, gathered)
    results, reply = agent.apply(reply, dry_run=True, task=task, gathered=gathered)

Phase 1 (gather): the model picks tools under a JSON grammar (see tools.py).
Constrained decoding here means tool calls never need freeform-text parsing.
There is no default step limit: max_steps = 0 (the default) lets the loop run
until the model says `done` or the gathered-source token budget is exhausted.
When the model says `done` but files the task names were never read, one
coverage nudge asks it to read them (repeated only after real progress), so a
small model does not enter an answer phase that can only fail. A positive
max_steps is an explicit hard user limit.

Phase 2 (answer): free text, no grammar. Code must NOT be forced through a
JSON string field, since escaping newlines and quotes measurably degrades
small-model output quality. So the edit format is fenced search/replace
blocks instead (see edits.py).

Prompt layout is deliberate: the system message and repo map come first and
never change within a gather loop, so llama.cpp's prefix cache hits on every
turn and only the new observation has to be processed, not the whole map
again.
"""

from __future__ import annotations

import re
import sys

from .edits import apply_edits, parse_edits
from .llm import LLMError, estimate_tokens
from .tools import TOOL_HELP, dispatch, schema_for

# How many times the gather loop re-asks after a step reply dies at the
# output cap / produces no salvageable JSON, before giving up and answering
# from what was gathered. Each recovery costs one API call but keeps paid
# lookups alive; two is enough for a model that rambles once in a while and
# small enough that a persistently broken server cannot bill forever.
_MAX_GATHER_RECOVERIES = 2

# A re-read of a range that added no new source is navigation churn, not
# progress. Small models can lock onto one range and emit it dozens of times
# (each echo also keeps the bad pattern in context). One duplicate gets a
# compact "already read" observation; after this many consecutive duplicates
# the gather phase stops and lets the answer phase use the source it already
# collected.
_MAX_DUPLICATE_READS = 2

# The answer prompt tells the model it may split a multi-file task across
# turns ("say what remains; the next turn continues"). A cleanly-finished
# reply that says more work is coming must actually trigger a next turn --
# otherwise a multi-file task silently stops at the first batch. This regex
# spots the deferral language; a false positive costs one "NOTHING MORE"
# round-trip, a false negative leaves the task half done.
_MORE_TO_DO_RE = re.compile(
    r"next turn|will follow|following turn|still to do|left to do|"
    r"remaining after|remaining work|remains to|to be done|not yet done",
    re.IGNORECASE,
)
# Matches the first line of a successful tool_read observation:
#   "prep/index.html lines 315-325:"
# Used for the read-span ledger and the eviction pass below. The path may
# contain spaces; the numbers never do, so the regex is unambiguous.
import re as _re_mod

_READ_HEAD_RX = _re_mod.compile(r"^(.+?) lines (\d+)-(\d+):")


def _first_seen_spans(spans_by_file: dict[str, list[tuple[int, int]]]
                      ) -> dict[str, list[tuple[int, int]]]:
    """Replay each file's read spans in step order and return, per file, the
    set of (start, end) sub-spans that were NEW when they were first shown."""
    out: dict[str, list[tuple[int, int]]] = {}
    for key, seq in spans_by_file.items():
        seen: list[tuple[int, int]] = []
        deltas: list[tuple[int, int]] = []
        for s, e in seq:
            # Lines already covered by earlier reads of this file.
            uncovered = [(s, e)]
            for a, b in seen:
                nxt = []
                for cs, ce in uncovered:
                    if b < cs or a > ce:          # no overlap
                        nxt.append((cs, ce))
                        continue
                    if cs < a:
                        nxt.append((cs, a - 1))
                    if ce > b:
                        nxt.append((b + 1, ce))
                uncovered = nxt
            for cs, ce in uncovered:
                deltas.append((cs, ce))
            merged = sorted(seen + [(s, e)])
            folded: list[tuple[int, int]] = []
            for a, b in merged:
                if folded and a <= folded[-1][1] + 1:
                    folded[-1] = (folded[-1][0], max(folded[-1][1], b))
                else:
                    folded.append((a, b))
            seen = folded
        if deltas:
            out[key] = deltas
    return out


def _slice_numbered(obs: str, keep: list[tuple[int, int]]) -> str:
    """Given a tool_read observation ("rel lines a-b:\\nNNN | text..."),
    return the same observation narrowed to the numbered lines whose numbers
    fall inside `keep` spans. Falls back to the original text if anything
    about the format is unexpected -- correctness of the header matters more
    than savings here."""
    m = _READ_HEAD_RX.match(obs)
    if not m:
        return obs
    head_end = obs.index("\n") if "\n" in obs else len(obs)
    head, body = obs[:head_end], obs[head_end + 1:]
    kept: list[str] = []
    for ln in body.split("\n"):
        pm = _re_mod.match(r"\s*(\d+) \|", ln)
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
    """Length of the NEW part of a read observation: the numbered lines not
    already covered by `prior_spans`. Used to charge max_gathered_chars per
    new byte, so re-reads of the same region cost nothing."""
    m = _READ_HEAD_RX.match(obs)
    if not m:
        return len(obs)
    head_end = obs.index("\n") if "\n" in obs else len(obs)
    body = obs[head_end + 1:]
    total = 0
    for ln in body.split("\n"):
        pm = _re_mod.match(r"\s*(\d+) \|", ln)
        if not pm or not any(a <= int(pm.group(1)) <= b for a, b in prior_spans):
            total += len(ln) + 1
    return total


def _read_has_new_lines(obs: str, prior_spans: list[tuple[int, int]]) -> bool:
    """True when a read observation contains source lines not in `prior_spans`.

    Separate from `_delta_len` because trailing notes ("[truncated ...]",
    symbol-resolution notes) are real tokens but are NOT new source. Counting
    them as progress would let an identical truncated re-read evade the
    duplicate-read guard forever.
    """
    m = _READ_HEAD_RX.match(obs)
    if not m:
        return True
    head_end = obs.index("\n") if "\n" in obs else len(obs)
    body = obs[head_end + 1:]
    for ln in body.split("\n"):
        pm = _re_mod.match(r"\s*(\d+) \|", ln)
        if pm and not any(a <= int(pm.group(1)) <= b for a, b in prior_spans):
            return True
    return False


def _evict_duplicate_reads(messages: list[dict], gathered: list[str],
                           spans: dict[str, list[tuple[int, int]]]) -> list[str]:
    """Rewrite duplicate source out of the conversation and the SOURCE block.

    A small model covering a big file re-reads it in overlapping chunks
    (L1-685 then L401-685), and every RESULT turn carried its full text. By
    the time gathering ends the live conversation holds several copies of
    the same lines -- which blows the *server's* context window even when
    our own budget math says there is room, and makes the answer phase
    process the same code two or three times.

    Each read's RESULT turn is rewritten down to the lines it was the FIRST
    to show (earlier turns keep theirs; later turns lose the overlap). The
    gathered SOURCE items are rebuilt from the same deltas so the answer
    prompt sees every line exactly once. Returns the new item list.
    """
    if not spans or not any(len(seq) > 1 for seq in spans.values()):
        return gathered
    deltas = _first_seen_spans(spans)

    # Rebuild the SOURCE items: reads keep only their first-seen lines.
    new_items: list[str] = []
    for g in gathered:
        head = g.split("\n", 1)[0]
        if head.startswith("--- read ") and "\n" in g:
            rest = head[len("--- read "):].strip()
            key = rest.rsplit(" L", 1)[0].strip().lower()
            body = g.split("\n", 1)[1]
            keep = deltas.get(key)
            if keep is None:
                new_items.append(g)
            else:
                new_items.append(head + "\n" + _slice_numbered(body, keep))
        else:
            new_items.append(g)

    # Rewrite RESULT turns in the live conversation, oldest -> newest, so
    # each turn keeps only the lines no earlier turn had already shown.
    for i, msg in enumerate(messages):
        if msg["role"] != "user" or not msg["content"].startswith("RESULT:\n"):
            continue
        content = msg["content"]
        body = content[len("RESULT:\n"):]
        m = _READ_HEAD_RX.match(body)
        if not m:
            continue
        key = m.group(1).strip().lower()
        s_, e_ = int(m.group(2)), int(m.group(3))
        # Spans of this same file shown by EARLIER turns only.
        prior = _fold_spans(_turn_spans(messages[:i], key))
        uncovered = [(s_, e_)]
        for a, b in prior:
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
        if m and m.group(1).strip().lower() == key:
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
You are a code assistant working through a lookup index of a Python project.
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
  description. Use it to check a GUI/game; a black/blank verdict is evidence.
  You cannot see the image directly -- rely on the text report. Needs
  allow_screenshots, and allow_shell when a command is launched.
- `move` and `delete` are available only when allow_write is on. Prefer the
  search/replace answer format for edits; use these for whole-file moves or
  removals the task explicitly asks for.
- Do not guess at code you have not read. If you need it, read it.
- Choose `done` as soon as you have enough to do the task. Do not browse.
- Fill every field. Use "" for unused strings and 0 for unused numbers.
- Keep `reasoning` under 15 words. It is a note to yourself, not a plan."""

# The project is small enough that every unit is already listed.
FLAT_HINT = """
The map below lists every function and class in the project with its line range.
Go straight to `read` for the ones you need."""

# The project is too big for a full map; the model must descend a level.
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

# The project is too big even for file cards; the map is one card per
# directory and the model descends two levels (dir -> file cards -> units).
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

# The search/replace edit format, shared by every answer brief. Custom system
# prompts (scaffold stubs, resume) override the ROLE of the answer but must
# always carry this exact format spec, or a small model invents its own --
# markdown fences, a bare filename, `search/replace` tags -- and the parser
# finds nothing ("no edit blocks in the reply").
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
- You are in the FINAL answer phase: no tools are available. Do not emit
  tool-call requests (no <tool_call> or <invoke> blocks) -- they cannot run
  here. Emit search/replace blocks only, for source you have already read.
  If you lack a file you need, work with what you have and note what you
  could not verify.
- Include enough surrounding lines that the SEARCH text appears only once.
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
You are a Python code assistant. You have a map of the project and the specific
source you asked to see. Complete the user's task.

""" + EDIT_FORMAT


def _evict_overflow_turns(messages: list[dict], client, cfg) -> bool:
    """Shrink the live conversation until it fits the server's window.

    Returns True if anything was dropped. Drops the OLDEST RESULT turns
    first (their source is already collected in `gathered` and reaches the
    answer phase anyway); a lone huge turn is clipped line-by-line, keeping
    its head -- which preserves the file/line header the coverage ledger
    parses. Never touches the map+task message or the assistant turns.

    Trimming runs with hysteresis: once over the edge, turns are dropped
    until the conversation sits a headroom's worth of tokens BELOW it, not
    exactly at it. Each rewrite invalidates the server's prompt-prefix cache
    from that turn onward, so without headroom a run hovering at the window
    edge re-trims (and re-pays the prefill) on every single observation.
    """
    import sys as _sys
    from .llm import estimate_tokens

    room = max(cfg.context_tokens - cfg.max_tokens_step
               - cfg.reserve_tokens // 2, 512)
    headroom = max(256, room // 10)

    def over_by() -> int:
        return sum(estimate_tokens(m["content"]) for m in messages) - room

    changed_any = False
    while over_by() > -headroom:
        idxs = [i for i, m in enumerate(messages)
                if m["role"] == "user" and m["content"].startswith("RESULT:\n")]
        if not idxs:
            break
        if over_by() <= 0:
            # Still UNDER the window -- trimming here is only cache
            # amortisation, so it must never cost the newest observation
            # (the model's next decision depends on seeing it). Once the
            # conversation is actually over the window (over_by() > 0),
            # every RESULT turn is fair game, newest included.
            if len(idxs) < 2:
                break
            idxs = idxs[:-1]
        i = idxs[0]
        content = messages[i]["content"]
        lines = content.split("\n")
        if len(lines) > 6:
            keep = max(4, len(lines) // 2)
            clipped = "\n".join(
                lines[:keep] + ["[older read dropped to fit the model's "
                                "context window; re-`read` it if you need "
                                "the rest]", "", ""] + lines[-2:])
            # A short turn can reach a fixed point: for 8-10 lines the
            # head+tail rewrite is the same size (or larger) than the
            # original, so re-clipping it forever never gets under the
            # window. Only rewrite when it actually saves tokens; otherwise
            # fall through and drop the turn entirely so this loop always
            # makes progress.
            if estimate_tokens(clipped) < estimate_tokens(content):
                messages[i] = {"role": "user", "content": clipped}
            else:
                lines = []      # force the removal path below
        if not lines or len(lines) <= 6:
            drop = {i}
            nxt = [j for j in idxs[1:] if messages[j]["content"].startswith("RESULT:\n")]
            if nxt:
                drop.add(nxt[0])   # keep roles alternating after removal
            messages[:] = [m for k, m in enumerate(messages) if k not in drop]
        changed_any = True
    if changed_any:
        print("[conversation trimmed to fit the model's context window]",
              file=_sys.stderr)
    return changed_any


class Agent:
    """Runs one task against an Index and an LM Studio client: gather source,
    generate an answer or edits, and apply those edits."""

    def __init__(self, index, client, cfg):
        """Bind the agent to a project Index, an LMStudio client, and a Config."""
        self.index = index
        self.client = client
        self.cfg = cfg
        # Set by gather(); read by _repair_source() when a failed file is no
        # longer on disk. Initialised here so apply() works even if it is
        # called without a preceding gather().
        self.gathered_source = ""

    # ------------------------------------------------------------------ phase 1
    def gather(self, task: str, verbose: bool = True,
               on_step=None, base: str | None = None) -> str:
        """Run the tool-calling loop: repeatedly ask the model to choose an
        action (read/grep/callers/outline/write/append/done) until it says
        `done`, the gathered-source token budget is exhausted, or an explicit
        cfg.max_steps limit is reached. `max_steps = 0` means no step limit
        (the default); only the token/context budgets bound the loop.

        Returns the concatenated text of everything read, ready to hand to
        `answer()`. Also sets `self.read_any` (whether any `read` succeeded)
        and `self.steps_log` (a human-readable record of each step, used for
        the session log).

        `on_step`, when given, is called with the step number after every
        completed step (including the final `done`). The GUI passes a
        callback here so the browser can show live progress; CLI callers
        leave it None and see no difference.

        `base`, when given, replaces `GATHER_BASE` as the system prompt's
        preamble (the FLAT/CARDS navigation hints are still appended). The
        `explain` flow uses this to frame gathering as answering a question
        rather than as preparing an edit.
        """
        repo_map, tier = self.index.map_for_prompt(self.cfg)
        system = (base if base is not None else GATHER_BASE) + \
            (FLAT_HINT if tier == "flat"
             else CARDS_HINT if tier == "cards" else DIRS_HINT)
        if verbose:
            print(f"  [map: {tier}, ~{estimate_tokens(repo_map)} tokens, "
                  f"{len(self.index.files)} files]", file=sys.stderr)
        base = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"PROJECT MAP\n\n{repo_map}\n\nTASK\n{task}"},
        ]
        messages = list(base)
        gathered: list[str] = []
        self.steps_log: list[str] = []
        did_read = False
        read_nudged = False
        # Successful reads only, as (rel_lower, start, end). Coverage checks
        # must count ONLY real source the model saw -- a grep hit or an ERROR
        # line does not let it quote the file, and counting those as coverage
        # is how a task ends with "answered without proposing any edits".
        self._read_spans: dict[str, list[tuple[int, int]]] = {}
        # 0 = no step limit: the gathered-source/token budgets are the only
        # bound. A positive max_steps is an explicit, hard user limit. The
        # done-coverage nudges below are progress checks, not a step budget.
        step_cap = max(0, int(getattr(self.cfg, "max_steps", 0) or 0))
        unlimited = step_cap <= 0
        # Files the task text names but no `read` has covered yet; refreshed
        # after every successful read so the `done` nudge can act on it.
        self._unread_files: list[str] = self._compute_unread(task)
        # The gathered-source ceiling is charged per NEW byte, not per echoed
        # byte: a re-read of lines already in context costs nothing, so a
        # small model bouncing over the same big file cannot exhaust the
        # budget before it reaches the code it must edit.
        budget = self.cfg.max_gathered_chars
        # Per-file line ledger: every read appends its (start, end) span here
        # in step order. The eviction pass at the end of the loop replays the
        # same sequence to work out which lines each RESULT turn was the first
        # to show -- so the bookkeeping during the loop stays dead simple and
        # a re-read costs nothing against max_gathered_chars.
        spans: dict[str, list[tuple[int, int]]] = {}
        self._read_spans = spans   # coverage truth for _compute_unread

        def nudge(step: int) -> str:
            """Text appended to a turn reminding the model how many lookups
            remain, or that there is no limit and only the token budget ends
            the phase."""
            if unlimited:
                return ("[Next action? No lookup limit -- choose 'done' once "
                        "the task is covered.]")
            remaining = step_cap - step
            return (f"[{remaining} lookups left after this one] Next action?"
                    if remaining else
                    "This is your LAST lookup. Choose 'done' unless you truly need one more.")

        # The first nudge rides along with the map+task message. Every later one
        # rides with its tool result. Mistral-family chat templates reject two
        # user messages in a row, so the sequence must stay strictly
        # system, user, assistant, user, assistant, ...
        messages[-1]["content"] += "\n\n" + nudge(1)

        step = 0
        step_schema = schema_for(self.cfg)
        done_confirmed = False   # one-shot guard for the post-truncation
                                 # "are you really done?" nudge below
        step_recoveries = 0      # budget for re-asking after a gather step
                                 # died at the output cap (see LLMError path)
        duplicate_reads = 0      # consecutive reads that added no new lines
        while unlimited or step < step_cap:
            step += 1
            try:
                stepjson = self.client.chat_json(messages, step_schema,
                                                 self.cfg.max_tokens_step,
                                                 disable_thinking=True)
            except LLMError as e:
                # A gather step that cannot produce JSON at all must not kill
                # the whole run -- the DeepSeek failure class: json_object
                # mode is only SOFT grammar there, so a rambling reply gets
                # cut mid-essay and every salvage attempt fails. The answer
                # phase rebuilds its own prompt from whatever was gathered
                # (or the map alone), which is strictly better than an error
                # exit after minutes of paid lookups.
                msg = str(e).lower()
                if "token limit" in msg or "valid json" in msg:
                    # Do not bail on the FIRST bad step. A hosted model that
                    # rambles past the output cap produces one doomed reply;
                    # asking again with a short-reasoning instruction usually
                    # gets a clean action back, and losing ten paid lookups of
                    # context to an answer phase that cannot edit blind is the
                    # worse outcome (this is exactly how the rename test died:
                    # gather broke at step 10, the answer phase had no source,
                    # and it emitted a lone placeholder stub). One retry per
                    # failure, two failures total, then proceed as before.
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
            action = (stepjson.get("action") or "").strip()
            why = (stepjson.get("reasoning") or "").strip()

            detail = (stepjson.get("file") or stepjson.get("pattern")
                      or stepjson.get("symbol") or stepjson.get("command") or "")
            if action == "read":
                detail = f"{stepjson.get('file')} L{stepjson.get('start')}-{stepjson.get('end')}"
            elif action == "move":
                detail = f"{stepjson.get('file')} -> {stepjson.get('to')}"
            elif action == "run":
                detail = stepjson.get("command", "")
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
                # If the model kept rambling in 'reasoning' until the reply
                # was cut off (step_truncated is set by chat_json's salvage
                # path), its LAST words were chopped -- a deepseek-chat
                # classic: it free-forms a long essay and hits the provider
                # cap mid-sentence. A `done` that arrives that way may be an
                # accidental truncation artifact, not a decision; give it one
                # explicit chance to confirm or continue before honouring it.
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
                    # The model gathered on some files but wants to change
                    # others it never `read`. Its answer rules forbid quoting
                    # unread code, so it would answer with zero blocks. Nudge
                    # once per round of progress; if it still says `done`
                    # without reading, accept and answer rather than looping
                    # `done` -> nudge forever.
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
                    # Answering from grep snippets produces edits that cannot
                    # anchor, because grep output is trimmed. Nudge once; if
                    # it still refuses to read, answer with what it has.
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

            obs = dispatch(self.index, self.cfg, stepjson, client=self.client)
            # If the model's last reply was cut off at the token limit and we
            # salvaged the action from it, say so in the observation: the cut
            # almost always happened inside 'reasoning', and an unmarked
            # recovery lets the model keep writing essays until every step
            # burns a truncation-retry round trip. One short sentence here
            # stops that spiral.
            if getattr(self.client, "step_truncated", False):
                obs += ("\n[Note: your previous reply hit the output token "
                        "limit before finishing its reasoning. Keep 'reasoning' "
                        "under 10 words -- long ones get cut off and wasted.]")
                # ...unless the cut landed INSIDE `content` of a write step:
                # then what survived is half a file and the note above is the
                # wrong advice. The partial content itself never reaches disk
                # (dispatch() refuses oversized writes), but without this the
                # model re-rolls the same giant write forever. Tell it the
                # exact chunk size that fits one message instead.
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
                # A successful read is progress only when it shows lines the
                # conversation has not already supplied. Compare against the
                # span ledger BEFORE recording this read, then keep only a
                # compact "already read" note for an exact duplicate: echoing
                # the same numbered lines back is what lets a small model lock
                # onto one range and repeat it forever.
                m = _READ_HEAD_RX.match(obs)
                if m:
                    # Normalise exactly like `_task_file_targets` does:
                    # lower-case AND strip leading "./". Without the lstrip,
                    # a model that asks for "./src/app.js" records coverage
                    # under "./src/app.js" while `_compute_unread` looks up
                    # "src/app.js" -- the file reads as permanently unread,
                    # the `done` nudge loops until the step ceiling, and the
                    # answer phase ends with zero edit blocks ("nothing to
                    # review" in the GUI despite successful reads).
                    key = m.group(1).strip().lower().lstrip("./")
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
                    # A real read is progress, so a later `done` with new
                    # missing files is allowed another coverage nudge.
                    read_nudged = False
                self._unread_files = self._compute_unread(task)
            # An ERROR observation is navigation feedback ("that range does
            # not exist"), not source. Carrying it into the answer prompt as
            # gathered text makes the SOURCE section look populated while it
            # contains no code at all -- the model then answers from the map
            # alone and proposes nothing, which surfaces in the GUI as a
            # blank review page. Record failures compactly instead.
            if obs.startswith("ERROR"):
                arg = (stepjson.get("file") or stepjson.get("pattern")
                       or stepjson.get("symbol") or stepjson.get("command") or "")
                item = f"--- {action} {arg} failed: {obs[:160]}"
                gathered.append(item)
                messages.append({"role": "assistant", "content": _compact(stepjson)})
                messages.append({"role": "user",
                                 "content": f"{obs}\n\n{nudge(step + 1)}"})
                continue
            # Charge the gather budget by NEW LINES ONLY. A re-read of lines
            # already in context costs nothing here -- otherwise a small model
            # bouncing over one big file exhausts max_gathered_chars before it
            # reaches the code it must edit, and the loop cuts to the answer
            # phase mid-gather. The RESULT turn still shows the full read (the
            # eviction pass at the end of the loop dedups the conversation).
            if action == "read" and _READ_HEAD_RX.match(obs):
                if read_delta is None:
                    k0 = _READ_HEAD_RX.match(obs).group(1).strip().lower()
                    delta = _delta_len(obs, list(spans.get(k0, [])))
                else:
                    delta = read_delta
                if delta > budget:
                    obs = obs[:budget] + "\n[truncated: context budget reached]"
                    budget = 0
                else:
                    budget -= delta
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
            if action == "outline":
                # Outlines are navigation, not evidence. The model needs them to
                # decide what to read; the answer phase needs the code it read.
                # Carrying a 2,400-token outline into the answer prompt is pure
                # waste, so record only that it happened.
                item = (f"--- outline {arg} (used for navigation; "
                        f"{len(self.index.in_file(self.index.resolve_file(arg) or ''))} units)")
            else:
                item = f"--- {action} {arg}\n{obs}"
            if not duplicate_read:
                gathered.append(item)

            messages.append({"role": "assistant", "content": _compact(stepjson)})
            messages.append({"role": "user",
                             "content": f"RESULT:\n{obs}\n\n{nudge(step + 1)}"})

            # Proactive window guard: our own char budget says there's room,
            # but the live conversation (map + every RESULT turn echoed back
            # on each request) can still outgrow the server's configured
            # window -- LM Studio then 400s mid-task ("kept hitting limits")
            # and Ollama silently drops the middle of the prompt (the model
            # answers nonsense). Trim before sending the next request.
            _evict_overflow_turns(messages, self.client, self.cfg)

            if budget <= 0:
                if verbose:
                    print("  [context budget exhausted; moving to answer]", file=sys.stderr)
                break

        self.read_any = did_read
        # Eviction pass. The dedup rule above charges the gather budget by
        # NEW bytes only -- but until this pass every RESULT turn still
        # carried its full text, so a model that re-reads a big file in
        # overlapping chunks leaves multiple copies sitting in the live
        # conversation. On a 9B/14B with an 8k-32k window that is how the
        # *server* runs out of context mid-task even though our own budget
        # says there is room. Rewrite each read's RESULT turn down to the
        # lines it was the first to show, and rebuild the SOURCE section
        # from those same deltas so the answer phase sees each line exactly
        # once.
        new_items = _evict_duplicate_reads(messages, gathered, spans)
        # Remember the deduped source so later phases (answer, repair) can
        # rebuild their prompts without re-quoting every read from scratch.
        self.gathered_source = "\n\n".join(new_items)
        # Final proactive trim: the loop above trims after each tool result,
        # but nudge turns (the "read these files first" prompts) can push the
        # conversation over the window between reads. Do one sweep before the
        # answer phase so the next request cannot 400 or be silently
        # truncated server-side.
        _evict_overflow_turns(messages, self.client, self.cfg)
        return "\n\n".join(new_items)

    # ------------------------------------------------------ coverage bookkeeping
    def _task_file_targets(self, task: str) -> set[str]:
        """Paths that look like files mentioned in the task text.

        A bare word like `index` inside a sentence is not a path; only tokens
        containing a slash or a dot-extension count, so this never nags about
        prose. Matched against the index case-insensitively by the caller.
        """
        import re as _re
        toks = _re.findall(r"[A-Za-z0-9_][A-Za-z0-9_./\-]*", task or "")
        out = set()
        for t in toks:
            if "/" in t or _re.search(r"\.[A-Za-z][A-Za-z0-9]{0,7}$", t):
                out.add(t.lower().lstrip("./"))
        return out

    def _compute_unread(self, task: str) -> list[str]:
        """Files the task names but no successful `read` covered.

        Coverage comes from `self._read_spans` -- recorded only when a read
        actually returned source text. A grep hit or an ERROR line is
        navigation, not evidence; counting those as coverage is how a task
        ends with "the model answered without proposing any edits" while the
        terminal looks healthy.

        Returns entries like "prep/sentences.js (not yet read -- it has 222
        lines, use start=1 end=222)" so the nudge can carry exact ranges.
        Any successful read of a file counts as coverage for it (a partial
        read means the model saw real text and knows what remains).
        """
        wanted = self._task_file_targets(task)
        if not wanted:
            return []
        matched: dict[str, str] = {}          # normalised rel -> canonical rel
        for rel in self.index.files:
            r = rel.lower().lstrip("./")      # same normalisation as the
            for w in wanted:                  # spans ledger and _task_file_targets
                if r == w or r.endswith("/" + w) or w.endswith("/" + r):
                    matched[r] = rel
        if not matched:
            return []
        missing = []
        for low, rel in sorted(matched.items()):
            if any(span for span in self._read_spans.get(low, ())):
                continue
            from . import fileio
            try:
                lines = fileio.read_lines(self.index.root / rel)
                if lines and lines[-1] == "" and len(lines) > 1:
                    lines = lines[:-1]   # same convention as tool_read
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

        `system` and `max_tokens` override the defaults for special flows that
        need a different brief than "edit this existing code" -- e.g. resuming
        an unfinished scaffold (`Agent.resume_task`) or writing one whole file
        of a new project (see scaffold.py). `include_edit_format=False` drops
        the search/replace format spec even when `system` is given, for flows
        that must answer in prose only (`explain`)."""
        repo_map, _ = self.index.map_for_prompt(self.cfg)
        user = (
            f"PROJECT MAP\n\n{repo_map}\n\n"
            f"SOURCE YOU REQUESTED\n\n{gathered or '(nothing was looked up)'}\n\n"
            f"TASK\n{task}"
        )
        # Hard fit check: llama.cpp/LM Studio 400 when prompt + max_tokens
        # exceeds the configured window, and Ollama does something worse --
        # it silently truncates the PROMPT from the middle. The TASK sits at
        # the very end of this message, so on a small window the model is
        # shown a map and some source but never the actual request; it then
        # "runs to completion" with a confident answer to some other task.
        # That is exactly the local-model "provided answer is nonsense"
        # symptom. Refuse loudly instead of letting the server mangle it.
        reply_room = min(self.cfg.max_tokens_answer,
                         self.cfg.request_max_tokens_cap or
                         self.cfg.max_tokens_answer)
        from .config import is_hosted as _is_hosted
        if _is_hosted(self.cfg):
            # Hosted APIs (DeepSeek, OpenAI, ...) have a real request-size
            # limit too -- deepseek-chat rejects prompts over ~128k tokens,
            # and a giant echoed history costs money and latency. Trim the
            # OLDEST gathered chunks until the prompt fits a generous ceiling
            # instead of refusing outright; the model still sees its most
            # recent reads, which are the ones it wants to edit.
            head = f"PROJECT MAP\n\n{repo_map}\n\n"
            tail = f"\n\nTASK\n{task}"
            ceil_ = int(getattr(self.cfg, "max_prompt_tokens", 0) or 60000)
            src = gathered or "(nothing was looked up)"
            while estimate_tokens(head + src + tail) > ceil_ and len(src) > 4000:
                cut = max(4000, len(src) * 3 // 4)
                dropped = src[cut:]
                src = src[:cut]
                src += ("\n\n[older gathered source trimmed to fit the "
                        "request limit (%d chars dropped -- re-read a file "
                        "if you need it again]" % len(dropped))
            user = head + f"SOURCE YOU REQUESTED\n\n{src}\n" + tail
        room = self.cfg.context_tokens - reply_room - self.cfg.reserve_tokens // 2
        if not _is_hosted(self.cfg) and estimate_tokens(user) > room > 0:
            over = estimate_tokens(user) - room
            raise LLMError(
                f"The answer prompt would be ~{estimate_tokens(user)} tokens "
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
        print(f"[answer context ~{estimate_tokens(user)} tokens]", file=sys.stderr)
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
        # Answer-phase truncation continuation. Hosted APIs cap one message's
        # OUTPUT (~8k tokens on deepseek-chat); a big multi-file task then gets
        # cut off mid-reply no matter how large max_tokens_answer is. The old
        # behaviour was fatal -- "kept running past the token limit" after 3
        # identical doomed retries. Instead: keep whatever complete edit blocks
        # survived, tell the model exactly which files still need work, and
        # ask for just those in the next message. Each round emits a few
        # blocks, fits under the provider cap, and the task finishes across
        # turns. Guarded by a char budget AND a round cap (a model that
        # re-emits one oversized block every round would otherwise spin for
        # the whole budget, generating the full max_tokens_answer each time).
        #
        # Detection lives in `stream_truncated`, NOT `last_finish_reason`:
        # the answer phase streams (the user watches the blocks appear), and
        # OpenAI-compatible SSE puts finish_reason only on the final chunk --
        # the buffered parser that sets last_finish_reason never runs for a
        # streamed reply, so it stayed "" after every stream and this loop
        # could never trigger. That is why truncated answers silently lost
        # their later files instead of continuing across turns.
        cont_budget = 0 if dry_run else \
            max(1, int(getattr(self.cfg, "max_tokens_answer", 40000))) * 4
        cont_rounds = 0
        empty_rounds = 0
        cont_cap = max(0, int(getattr(self.cfg, "max_continuation_rounds", 0)
                              or 0))
        while getattr(self.client, "stream_truncated", False) \
                and cont_budget > 0:
            cont_rounds += 1
            if cont_cap and cont_rounds > cont_cap:
                print(f"\n[continuation loop reached its {cont_cap}"
                      "-round cap; stopping]", file=sys.stderr)
                break
            parsed = parse_edits(reply)
            done_files = {b.file.strip().lower().lstrip("./") for b in parsed}
            # The last streamed reply was cut mid-block: its dangling tail
            # (header + partial SEARCH, maybe a ======= line but no closing
            # REPLACE) would otherwise ride along into the next continuation
            # prompt -- confusing the model and, worse, poisoning any later
            # repair round that re-parses this accumulated text. Drop it now;
            # we are about to ask the model to re-emit exactly that file.
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
                    f"TASK\n{task}\n\n"
                    f"EDITS ALREADY DELIVERED (do NOT repeat these):\n"
                    + "\n".join(f"- {f}" for f in sorted(done_files)) +
                    "\n\nYour previous reply was CUT OFF by the output "
                    "token limit. Emit ONLY the remaining search/replace "
                    "blocks for files not listed above -- at most two or "
                    "three files per reply, never a whole large file in "
                    "one REPLACE. If everything is already covered, reply "
                    "with a single line: NOTHING MORE.")
            elif empty_rounds:
                # Second consecutive empty round: the generic re-ask was
                # ALSO cut off before any block, so the model is re-emitting
                # one oversized block every time. Escalate once to an
                # explicit single-small-chunk demand instead of rolling the
                # same doomed request again; if that too is cut off, the
                # loop below gives up.
                print(f"\n[reply was cut off before any complete block again; "
                      f"asking for a single small chunk]", file=sys.stderr)
                prompt = (
                    f"TASK\n{task}\n\n"
                    "Your replies keep getting CUT OFF by the output token "
                    "limit because they are far too long. STOP emitting the "
                    "whole file. Emit exactly ONE search/replace block that "
                    "covers only the FIRST ~30 lines of the file -- nothing "
                    "else. The remaining content is requested in later turns.")
            else:
                # Cut off before finishing even one block: the whole reply
                # was lost. Re-ask for it, and teach the chunk protocol so a
                # file too large for one message gets split across replies.
                print(f"\n[reply was cut off before any complete block; "
                      f"re-asking for the file]", file=sys.stderr)
                prompt = (
                    f"TASK\n{task}\n\n"
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
            if "NOTHING MORE" in more and not parse_edits(more):
                break
            if not parse_edits(more):
                if getattr(self.client, "stream_truncated", False):
                    # The re-ask was ALSO cut off before any complete block.
                    # One escalated attempt (above), then stop: a model that
                    # keeps outgrowing the cap would otherwise spin this loop
                    # for the whole char budget, generating the full
                    # max_tokens_answer every round.
                    empty_rounds += 1
                    if empty_rounds >= 2:
                        print("\n[repeatedly cut off before any complete "
                              "block; stopping the continuation loop]",
                              file=sys.stderr)
                        break
                    print("[re-ask was truncated too; trying once more]",
                          file=sys.stderr)
                    continue
                # Continuation produced no edit blocks at all -- keep the
                # truncated-but-parseable original rather than replacing it
                # with prose, and stop so we cannot loop on empty replies.
                break
            empty_rounds = 0
            reply = (reply.rstrip() + "\n\n" + more) if reply.strip() else more
        # If the reply (original or its last continuation round) was cut off
        # mid-block, trim the dangling tail before anything parses it. The
        # loop above already trimmed between continuation rounds; this catches
        # a truncated FIRST reply that never entered the loop (zero complete
        # blocks -> `break` with nothing to continue from), and the final
        # round when the budget ran out. BLOCK_RE needs a closing REPLACE to
        # match, so an unterminated block would otherwise leave its header in
        # the text -- which `_failed_block_texts` echoes back to the model as
        # half a block, inviting it to "complete" the fragment instead of
        # re-anchoring it.
        if getattr(self.client, "stream_truncated", False):
            reply = self._trim_dangling_block(reply)
        results = self._apply_once(reply, dry_run, backup_dir)
        # Snapshot what this reply already delivered (including any truncation
        # continuations folded into `reply` above) and whether it defers the
        # rest to "the next turn" -- used by the completion loop below, and
        # captured before the repair loop reassigns `reply`.
        delivered = {b.file.strip().lower().lstrip("./") for b in parse_edits(reply)}
        deferred = bool(_MORE_TO_DO_RE.search(reply))

        for _round in range(repair_rounds):
            failures = [r for r in results if not r.ok]
            if not failures:
                break
            print(f"\n[{len(failures)} block(s) failed; asking the model to re-anchor]", file=sys.stderr)
            fb = "\n".join(f"- {r.file}: {r.message}" for r in failures)
            # Re-ground the repair prompt FROM DISK, not from the gathered
            # buffer. The old code sliced the gather output (regex-filtered,
            # 12k chars), so when a SEARCH block failed the exact lines it
            # needed were often missing or truncated -- and local models
            # correctly replied "I only have grep output, I can't produce a
            # corrected edit" and burned every repair round on prose. Read
            # the real file and quote the region around the fuzzy-matched
            # anchor (or the target line range), with line numbers, so the
            # model can copy exact text including real indentation.
            prev_blocks = self._failed_block_texts(reply, failures)
            src_slice = self._repair_source(failures, prev_blocks)
            # A placeholder-creation refusal is a RENAME instruction problem,
            # not an anchor problem: the model never needs to copy bytes (the
            # clearing block's SEARCH is discarded -- Pedacito moves the real
            # file). The generic "copy the SEARCH text byte for byte" wording
            # above invites exactly the doomed full-content re-typing that
            # gets cut off at the provider cap mid-block. Override it.
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
                # An anchor miss on an existing file is very often a botched
                # WHOLE-FILE rewrite (typically a stub implementation): the
                # model quoted the old contents from memory -- or translated
                # them into what the file should contain -- and no exact or
                # fuzzy anchor survives. The base instruction below ("copy
                # the SEARCH text exactly") then steers it straight back into
                # the same wall, and a small model answers with prose and
                # burns the round. Offer the supported way out instead: an
                # empty SEARCH replaces the whole file, needing no quote of
                # the old text at all.
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
                # Cut off before the first block closed: a token-limit
                # casualty, not a refusal. Unlike the prose case below, one
                # smaller re-ask plausibly lands -- the empty-SEARCH
                # whole-file form needs no re-quoting of the old text, so it
                # fits caps the full quote-and-replace reply outran.
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
                # The model answered with prose instead of blocks (the classic
                # small-model "I need a read first" reply). Re-sending the same
                # prompt cannot plausibly end differently -- stop repairing
                # rather than burning the remaining rounds/tokens on it.
                print("[repair produced no edit blocks; stopping]",
                      file=sys.stderr)
                break
            good = [r for r in results if r.ok]
            results = good + self._apply_once(reply, dry_run, backup_dir)

        # Completion loop. The answer prompt invites the model to split a
        # multi-file task across turns ("edit a few files this turn and say
        # what remains"). A cleanly-finished reply that says more work is
        # coming is NOT a truncation, so the continuation loop above never
        # fired -- and the task would stop half done (the exact "only half my
        # request was addressed" failure). When the reply defers, actually run
        # the next turn: re-prompt with the delivered files and gathered
        # source, apply any new blocks, and keep going until it reports
        # NOTHING MORE or stops producing blocks.
        completion_cap = max(0, int(getattr(self.cfg, "max_completion_rounds", 0)
                                    or 0))
        # Token-derived budget (same shape as the truncation-continuation
        # budget) so an endlessly deferring model stops on tokens, not an
        # arbitrary turn count.
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
                if "NOTHING MORE" in more and not parse_edits(more):
                    break
                if not parse_edits(more):
                    break                      # no blocks -> treat as finished
                completion_budget -= len(more)
                reply = (reply.rstrip() + "\n\n" + more) if reply.strip() else more
                results += self._apply_once(more, dry_run, backup_dir)
                delivered |= {b.file.strip().lower().lstrip("./")
                              for b in parse_edits(more)}
                if not _MORE_TO_DO_RE.search(more):
                    break
            if completion_budget <= 0:
                print("\n[completion loop token budget exhausted; stopping]",
                      file=sys.stderr)
        return results, reply

    # ------------------------------------------------------------- repair help
    @staticmethod
    def _trim_dangling_block(reply: str) -> str:
        """Drop a truncated reply's unfinished tail.

        A streamed answer cut off by the provider's output cap ends inside a
        block -- after `--- FILE:` (maybe mid-SEARCH, maybe past `=======`)
        but always without the closing `>>>>>>> REPLACE` the parser needs.
        That fragment never survives parse_edits(), yet its header still
        matches regex searches elsewhere (`_failed_block_texts`, repair
        prompts), which is how a half-written file gets treated as "already
        delivered" and silently lost. Trim from the last header that has no
        closing marker after it; complete blocks earlier in the reply are
        untouched."""
        if not reply:
            return reply
        heads = [m.start() for m in re.finditer(r"(?m)^---\s*FILE:", reply)]
        if not heads:
            return reply
        last = heads[-1]
        tail = reply[last:]
        if re.search(r"(?m)^>{5,}\s*REPLACE", tail):
            return reply               # final block is complete -- keep all
        # Cut at the earliest dangling marker inside that last block: a bare
        # `<<<<<<< SEARCH` with no header above it can happen when a model
        # omits `--- FILE:` entirely and got cut mid-block anyway.
        cut = last
        m = re.search(r"(?m)^<{5,}\s*SEARCH", tail)
        if m and not re.search(r"(?m)^>{5,}\s*REPLACE", tail[m.end():]):
            cut = min(cut, last + m.start())
        trimmed = reply[:cut].rstrip()
        return trimmed + "\n" if trimmed else ""

    @staticmethod
    def _failed_block_texts(reply: str, failures) -> str:
        """Echo back the blocks the model actually wrote for the failed files.
        Telling it WHICH search text missed is what lets it re-anchor; without
        this the feedback ('search block not found') gives nothing to correct."""
        wanted = {(r.file or "").strip().lower().lstrip("./") for r in failures}
        out = []
        for e in parse_edits(reply):
            low = e.file.strip().lower().lstrip("./")
            if any(low == w or low.endswith("/" + w) or
                   (w and low.endswith(w)) for w in wanted if w):
                out.append(f"--- FILE: {e.file}\n<<<<<<< SEARCH\n{e.search}\n"
                           f"=======\n{e.replace}\n>>>>>>> REPLACE")
        return "\n\n".join(out)[:6000] or "(previous blocks unavailable)"

    def _repair_source(self, failures, prev_blocks: str = "") -> str:
        """Ground-truth source for the failed files, read fresh from disk.

        For each failure we try to locate the model's previous SEARCH text in
        the file fuzzily (locate() tolerates whitespace drift); on a hit we
        quote +/-30 lines around it with line-number prefixes. On a miss we
        fall back to the full file up to max_read_lines, then to whatever the
        gather phase captured. Small windows keep the repair prompt cheap on
        hosted APIs while still containing the exact bytes to copy."""
        from . import fileio
        from .edits import locate as _locate
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
                except OSError:
                    text = None
            if text is None:
                # Not on disk (new-file create that mis-grouped, deleted
                # mid-task): fall back to the gathered buffer for this file.
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
                m = re.search(r"<<<<<<< SEARCH\n(.*?)\n=======", blk, re.S)
                if not m:
                    continue
                needle = m.group(1)
                if rel.lower() not in blk.lower().split("<<<<<<<")[0]:
                    continue
                span = _locate(text, needle, self.cfg)
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
        """One parse-and-apply pass, with no repair round-trip.

        Duplicate FILE headers are NOT merged here any more: apply_edits
        groups blocks by path itself, classifies each block individually for
        rename detection, and applies each path's blocks in order. Pre-merging
        here glued a rename's create-block to its own clear-block under one
        header, which made the rename pair invisible to detection -- the task
        ended with a blank review page and the old files untouched. It also
        corrupted new-file bodies with stray leading blank lines, and fused
        the SEARCH bodies of ordinary multi-block edits into a needle that
        could never match.
        """
        return apply_edits(self.index.root, parse_edits(reply),
                           backup=self.cfg.backup, dry_run=dry_run,
                           backup_dir=backup_dir, cfg=self.cfg)

    # ------------------------------------------------------------------ resume
    def resume_task(self, task: str, stream: bool = True, dry_run: bool = False,
                    backup_dir=None) -> tuple[list, str]:
        """Finish a scaffolded project whose files are still stubs.

        `new --plan-only` leaves a real tree of one-line headers and a saved
        index, so the ordinary gather loop can navigate it -- every file has an
        outline and readable line numbers. The only thing missing is a brief that
        says "implement these stubs" instead of "answer about this code", which
        is what the `system` override below supplies. Everything else is the
        normal path: gather, answer, edit blocks, repair rounds, backups, review.

        Returns (results, reply)."""
        from .scaffold import RESUME_SYSTEM

        gathered = self.gather(task, verbose=True)
        reply = self.answer(task, gathered, stream=stream, system=RESUME_SYSTEM)
        return self.apply(reply, dry_run=dry_run, task=task, gathered=gathered,
                          backup_dir=backup_dir)


def _compact(step: dict) -> str:
    """Render a parsed step back to text with empty fields dropped, used to
    echo the model's own prior action into the conversation history."""
    keep = {k: v for k, v in step.items() if v not in ("", 0, None)}
    return str(keep)
