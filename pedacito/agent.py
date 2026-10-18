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
cfg.max_steps is a wall-clock guardrail, not a coverage budget: when the
model says `done` but files the task names were never read, the loop extends
the budget step by step (bounded by _MAX_STEPS_CEIL) rather than entering an
answer phase that can only fail -- reading real source is always progress.

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

import sys

from .edits import apply_edits, parse_edits
from .llm import estimate_tokens
from .tools import STEP_SCHEMA, TOOL_HELP, dispatch

# Absolute ceiling on gather-loop steps regardless of cfg.max_steps. The
# budget-extension logic exists so a small model can finish covering the
# files a task names even when max_steps is tight; this keeps a confused
# model from ping-ponging `done` <-> nudge forever. Each extension grants
# exactly one more lookup and only while real coverage gaps remain.
_MAX_STEPS_CEIL = 40

# Matches the first line of a successful tool_read observation:
#   "prep/index.html lines 315-325:"
# Used for the read-span ledger and the eviction pass below. The path may
# contain spaces; the numbers never do, so the regex is unambiguous.
import re as _re_mod

_READ_HEAD_RX = _re_mod.compile(r"^(.+?) lines (\d+)-(\d+):")

# Matches a file path with an extension inside prose (task text, or a model
# reply). Used both for coverage bookkeeping and to detect when a failed edit
# is really the model *asking for a re-read* rather than proposing a corrected
# block ("I can't produce a corrected edit for `subj-imp/index.html` because I
# only have grep output...").
_FILE_TOKEN_RX = _re_mod.compile(
    r"[A-Za-z0-9_][A-Za-z0-9_.\-\u00c0-\uffff/\\]*\.[A-Za-z][A-Za-z0-9]{0,7}")


def _reply_requests_reread(reply: str) -> list[str]:
    """File paths the model names in its reply while asking for more context.

    The repair loop's first pass feeds back only the failure messages and the
    already-gathered source. A small model that genuinely lacks the exact text
    answers that with prose -- "I need a `read` of index.html around line 415"
    -- instead of a corrected block, and the old loop then simply repeated the
    same unanchorable edit once more and gave up. If the reply contains no
    parseable edit blocks but does name files and reads like a request for a
    re-read, return those file paths so the caller can actually perform the
    reads and try one more time."""
    if parse_edits(reply):
        return []          # it did emit edits; nothing to fetch for it
    low = reply.lower()
    if not any(k in low for k in ("read", "grep", "context", "line",
                                   "anchor", "match")):
        return []
    out: list[str] = []
    for tok in _FILE_TOKEN_RX.findall(reply):
        t = tok.strip("`*'\"()[]{}.,;:").lstrip("./").lower()
        if len(t) > 2 and t not in out:
            out.append(t)
    return out[:6]


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


# The block markers, loose enough to recognise a half-written one. Mirrors
# edits.BLOCK_RE's tolerance (5+ angle brackets) without importing it here.
_SEARCH_OPEN_RX = _re_mod.compile(r"<{5,}\s*SEARCH")
_REPLACE_CLOSE_RX = _re_mod.compile(r">{5,}\s*REPLACE")


def _ends_mid_block(reply: str) -> bool:
    """True when `reply` stops inside an unterminated search/replace block.

    A truncated answer almost always dies mid-REPLACE: its last SEARCH opener
    never gets a REPLACE closer. Detecting this by shape matters because some
    servers omit `finish_reason` from streamed chunks -- without it, a silent
    cut-off would sail through as "the model answered without proposing any
    edits"."""
    opens = len(_SEARCH_OPEN_RX.findall(reply or ""))
    closes = len(_REPLACE_CLOSE_RX.findall(reply or ""))
    return opens > closes


def _merge_reply_blocks(first: str, second: str) -> str:
    """Join a truncated answer with its continuation so `parse_edits` sees one
    coherent reply.

    Concatenating raw text can glue a partial block from `first` onto the
    continuation in `second` and corrupt both. Instead keep everything up to
    (and including) the last complete `>>>>>>> REPLACE` in `first`, then append
    `second`. If `first` has no complete block at all, the continuation is the
    whole answer."""
    ends = [m.end() for m in _REPLACE_CLOSE_RX.finditer(first)]
    if not ends:
        return second
    return first[:ends[-1]] + "\n\n" + second


GATHER_BASE = f"""\
You are a code assistant working through a lookup index of a Python project.
You cannot see the full source. You see a map of it, and you request the exact
parts you need, one request at a time.

Available actions:
{TOOL_HELP}

Rules:
- Request one action per reply.
- Line ranges in the map are exact. Read them directly.
- `grep` and `outline` tell you WHERE code is. Only `read` shows you the exact
  text. You must `read` a region before you can change it.
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

ANSWER_SYSTEM = """\
You are a Python code assistant. You have a map of the project and the specific
source you asked to see. Complete the user's task.

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
- Include enough surrounding lines that the SEARCH text appears only once.
- Keep each block small and local. Use several blocks rather than one large one.
- Do not output the whole file. Do not output a diff or patch format.
- If you are adding a new function, anchor the SEARCH on an existing nearby line.
- To CREATE a brand-new file: emit exactly one block for it with an EMPTY
  SEARCH section (nothing between <<<<<<< SEARCH and =======) and the full
  file contents in REPLACE.
- To RENAME or MOVE a file: emit three blocks -- (1) an empty-SEARCH creation
  block for the NEW path whose REPLACE is just a short placeholder comment;
  (2) for the OLD path, a block whose SEARCH is its entire current content
  and whose REPLACE is empty; (3) optionally a block that edits the new path's
  placeholder. When the creation and the clearing of the old path are both
  present, Pedacito moves the old file's real contents to the new path and
  removes the old file -- never type the old file's contents into REPLACE,
  as that would double them.

Before the blocks, write a short plain explanation of what you changed and why.
If the task is a question rather than a change, just answer it and output no blocks."""


class Agent:
    """Runs one task against an Index and an LM Studio client: gather source,
    generate an answer or edits, and apply those edits."""

    def __init__(self, index, client, cfg):
        """Bind the agent to a project Index, an LMStudio client, and a Config."""
        self.index = index
        self.client = client
        self.cfg = cfg

    # ------------------------------------------------------------------ phase 1
    def gather(self, task: str, verbose: bool = True,
               on_step=None) -> str:
        """Run the tool-calling loop: repeatedly ask the model to choose an
        action (read/grep/callers/outline/done) until it says `done` or
        cfg.max_steps is reached, accumulating the source it read.

        Returns the concatenated text of everything read, ready to hand to
        `answer()`. Also sets `self.read_any` (whether any `read` succeeded)
        and `self.steps_log` (a human-readable record of each step, used for
        the session log).

        `on_step`, when given, is called with the step number after every
        completed step (including the final `done`). The GUI passes a
        callback here so the browser can show live progress; CLI callers
        leave it None and see no difference.
        """
        repo_map, tier = self.index.map_for_prompt(self.cfg)
        system = GATHER_BASE + (FLAT_HINT if tier == "flat" else CARDS_HINT)
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
        nudged = False
        read_nudged = False
        # Successful reads only, as (rel_lower, start, end). Coverage checks
        # must count ONLY real source the model saw -- a grep hit or an ERROR
        # line does not let it quote the file, and counting those as coverage
        # is how a task ends with "answered without proposing any edits".
        self._read_spans: dict[str, list[tuple[int, int]]] = {}
        # max_steps is a wall-clock guardrail against loops, not a coverage
        # budget. When the model says `done` but files the task names were
        # never read, one more lookup is cheaper than a guaranteed-failed
        # answer phase -- so allow it to run past the nominal limit while it
        # is making real progress (bounded by _MAX_STEPS_CEIL).
        step_cap = self.cfg.max_steps
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
            remain, or that this is its last one."""
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
        while step < step_cap:
            step += 1
            stepjson = self.client.chat_json(messages, STEP_SCHEMA, self.cfg.max_tokens_step)
            action = (stepjson.get("action") or "").strip()
            why = (stepjson.get("reasoning") or "").strip()

            detail = stepjson.get("file") or stepjson.get("pattern") or stepjson.get("symbol") or ""
            if action == "read":
                detail = f"{stepjson.get('file')} L{stepjson.get('start')}-{stepjson.get('end')}"
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
                can_extend = (step_cap < _MAX_STEPS_CEIL and budget > 0)
                if missing and (step < self.cfg.max_steps or can_extend):
                    # The model gathered on some files but wants to change
                    # others it never `read`. Its answer rules forbid quoting
                    # unread code, so it answers with zero blocks -- which
                    # surfaces as "No review: the model answered without
                    # proposing any edits" while the terminal looks healthy.
                    # Hand it the exact ranges to fetch before it commits,
                    # extending the step budget past max_steps when needed:
                    # reading a file the task names is always progress, and
                    # a guaranteed-failed answer phase costs more than one
                    # extra lookup.
                    listing = "\n".join(f"- {f}" for f in missing[:6])
                    if step >= self.cfg.max_steps:
                        step_cap += 1
                        if verbose:
                            print(f"  (extending budget: {len(missing)} file(s) "
                                  f"the task names were never read; now up to "
                                  f"{step_cap} steps)", file=sys.stderr)
                    elif read_nudged:
                        # Already nudged once inside the nominal budget and it
                        # still says done with gaps -- extend instead of nagging.
                        step_cap += 1
                    else:
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
                if not did_read and step < step_cap:
                    # Answering from grep snippets produces edits that cannot
                    # anchor, because grep output is trimmed. Send it back once.
                    nudged = True
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

            obs = dispatch(self.index, self.cfg, stepjson)
            if action == "read" and not obs.startswith("ERROR"):
                did_read = True
                # Append-only span ledger. The eviction pass at the end of
                # the loop replays these spans in order to work out which
                # lines each RESULT turn was the first to show -- so a
                # re-read costs nothing against max_gathered_chars and the
                # conversation cannot fill itself with duplicate copies of
                # the same file.
                m = _READ_HEAD_RX.match(obs)
                if m:
                    key = m.group(1).strip().lower()
                    spans.setdefault(key, []).append(
                        (int(m.group(2)), int(m.group(3))))
                self._unread_files = self._compute_unread(task)
            # An ERROR observation is navigation feedback ("that range does
            # not exist"), not source. Carrying it into the answer prompt as
            # gathered text makes the SOURCE section look populated while it
            # contains no code at all -- the model then answers from the map
            # alone and proposes nothing, which surfaces in the GUI as a
            # blank review page. Record failures compactly instead.
            if obs.startswith("ERROR"):
                arg = stepjson.get("file") or stepjson.get("pattern") or stepjson.get("symbol") or ""
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
                k0 = _READ_HEAD_RX.match(obs).group(1).strip().lower()
                prior = [sp for sp in spans.get(k0, [])[:-1]]
                delta = _delta_len(obs, prior)
                if delta > budget:
                    obs = obs[:budget] + "\n[truncated: context budget reached]"
                    budget = 0
                else:
                    budget -= delta
            else:
                if len(obs) > budget:
                    obs = obs[:budget] + "\n[truncated: context budget reached]"
                budget -= len(obs)

            arg = stepjson.get("file") or stepjson.get("pattern") or stepjson.get("symbol")
            if action == "outline":
                # Outlines are navigation, not evidence. The model needs them to
                # decide what to read; the answer phase needs the code it read.
                # Carrying a 2,400-token outline into the answer prompt is pure
                # waste, so record only that it happened.
                item = (f"--- outline {arg} (used for navigation; "
                        f"{len(self.index.in_file(self.index.resolve_file(arg) or ''))} units)")
            else:
                item = f"--- {action} {arg}\n{obs}"
            gathered.append(item)

            messages.append({"role": "assistant", "content": _compact(stepjson)})
            messages.append({"role": "user",
                             "content": f"RESULT:\n{obs}\n\n{nudge(step + 1)}"})

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
        matched: dict[str, str] = {}          # lower rel -> canonical rel
        for rel in self.index.files:
            r = rel.lower()
            for w in wanted:
                if r == w or r.endswith("/" + w) or w.endswith("/" + r) or w == r:
                    matched[rel.lower()] = rel
        if not matched:
            return []
        missing = []
        for low, rel in sorted(matched.items()):
            if self._read_spans.get(low):
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
    def _answer_user_content(self, task: str, gathered: str) -> str:
        """Assemble the answer prompt's user turn: map + source + task. Kept in
        one place so the retry-on-truncation path rebuilds exactly the same
        request."""
        repo_map, _ = self.index.map_for_prompt(self.cfg)
        return (
            f"PROJECT MAP\n\n{repo_map}\n\n"
            f"SOURCE YOU REQUESTED\n\n{gathered or '(nothing was looked up)'}\n\n"
            f"TASK\n{task}"
        )

    def answer(self, task: str, gathered: str, stream: bool = True,
               system: str | None = None, max_tokens: int | None = None) -> str:
        """Ask the model to answer the task, or produce search/replace edits,
        given the project map and whatever source `gather()` collected.

        `system` and `max_tokens` override the defaults for special flows that
        need a different brief than "edit this existing code" -- e.g. resuming
        an unfinished scaffold (`Agent.resume_task`) or writing one whole file
        of a new project (see scaffold.py).

        The answer is free text with fenced search/replace blocks, so the one
        failure mode that used to end a run silently was a reply cut off at
        `max_tokens_answer`: the last block loses its `>>>>>>> REPLACE` closer,
        parses as nothing, and the user sees "the model answered without
        proposing any edits" while the terminal looks healthy. Detect the
        truncation (`finish_reason == "length"`), salvage every *complete*
        block from what did arrive, and -- when the reply stopped mid-block --
        spend one more call asking the model to continue exactly where it
        stopped, with a doubled token budget.

        A server whose streaming chunks omit `finish_reason` (some do) is
        handled by shape alone: if the reply ends inside an unterminated
        search/replace block, treat it as truncated rather than letting a
        half-written block fail against the file later.
        """
        budget = max_tokens or self.cfg.max_tokens_answer
        messages = [{"role": "system", "content": system or ANSWER_SYSTEM},
                    {"role": "user", "content": self._answer_user_content(task, gathered)}]
        print(f"[answer context ~{estimate_tokens(messages[-1]['content'])} tokens]",
              file=sys.stderr)
        reply = self.client.chat(messages, max_tokens=budget, stream=stream)
        truncated = (self.client.last_finish_reason == "length"
                     or _ends_mid_block(reply))
        if not truncated:
            return reply

        complete = parse_edits(reply)
        if not complete and not _ends_mid_block(reply):
            # Cut off, but not inside an edit block -- there is nothing a
            # continuation could add that would parse. Don't burn a second
            # generation on an essay; hand back what we have.
            return reply

        nxt = min(max(budget * 2, 2048), self._answer_budget_ceiling(budget))
        if nxt <= budget:
            print("[answer was cut off at the token limit; the ceiling is already "
                  "at max_tokens_answer -- raise it (or context_tokens) in your "
                  "profile]", file=sys.stderr)
            return reply
        print(f"[answer hit the {budget}-token limit after "
              f"{len(complete)} complete block(s); asking for a continuation with "
              f"max_tokens={nxt}]", file=sys.stderr)
        cont_messages = list(messages)
        if complete:
            # The earlier blocks are safe to keep -- they anchor on text the
            # model already quoted correctly -- so only ask for what remains.
            cont_messages.append({"role": "assistant", "content": reply})
            cont_messages.append({"role": "user", "content":
                "Your reply was cut off by the token limit. The blocks above were "
                "received. Output ONLY the remaining search/replace blocks, "
                "starting exactly where you stopped -- no explanation, no recap."})
        else:
            cont_messages.append({"role": "assistant", "content": reply})
            cont_messages.append({"role": "user", "content":
                "Your reply was cut off by the token limit before any complete "
                "block arrived. Re-output the full set of search/replace blocks "
                "now, keeping each block small and local so the reply fits."})
        cont = self.client.chat(cont_messages, max_tokens=nxt, stream=False)
        merged = _merge_reply_blocks(reply, cont)
        if self.client.last_finish_reason == "length":
            print("[continuation was also cut off; using everything that parsed]",
                  file=sys.stderr)
        return merged

    def _answer_budget_ceiling(self, current: int) -> int:
        """Largest answer budget worth retrying at: never past what the
        declared context window can hold, mirroring chat_json's escalation
        cap so raising max_tokens_answer/reserve_tokens raises this too."""
        usable = self.cfg.context_tokens - self.cfg.reserve_tokens
        return max(min(16000, max(usable, 512)), current)

    def reread_for(self, files: list[str], task_hint: str = "") -> str:
        """Perform `read` lookups on Pedacito's own initiative and return the
        source text, formatted like the gather loop's SOURCE items.

        Used by the repair loop: when a model replies "I can't produce a
        corrected edit because I only have grep output for that file", arguing
        is pointless -- fetch the exact lines it asked for and put them in
        front of it. Reads whole files in `max_read_lines` chunks, capped to a
        slice of the gather budget per repair round so a confused model cannot
        make us re-index the repo through this door."""
        from .tools import tool_read
        out: list[str] = []
        budget = max(2000, self.cfg.max_gathered_chars // 3)
        for f in files:
            rel = self.index.resolve_file(f)
            if rel is None:
                continue
            start = 1
            total = None
            while True:
                obs = tool_read(self.index, self.cfg, rel, start,
                                start + self.cfg.max_read_lines - 1)
                if obs.startswith("ERROR"):
                    break
                m = _READ_HEAD_RX.match(obs)
                if not m:
                    break
                s_, e_ = int(m.group(2)), int(m.group(3))
                item = f"--- read {rel} L{s_}-{e_} (fetched during repair)\n{obs}"
                if len(item) > budget:
                    break
                out.append(item)
                budget -= len(item)
                total = total or e_
                if e_ < s_ or (total is not None and e_ >= total):
                    break          # clamped read reached the file end
                if e_ - s_ + 1 < self.cfg.max_read_lines:
                    break          # short read: that was the whole file
                start = e_ + 1
                # Remember what we covered so coverage bookkeeping stays true.
                self._read_spans.setdefault(rel.lower(), []).append((s_, e_))
        return "\n\n".join(out)

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
        results = self._apply_once(reply, dry_run, backup_dir)

        # One extra round-trip is reserved for the "the model asked for a
        # re-read" case below; `repair_rounds` still caps the ordinary
        # re-anchor attempts.
        rounds_left = repair_rounds
        reread_done = False
        while True:
            failures = [r for r in results if not r.ok]
            if not failures:
                break
            print(f"\n[{len(failures)} block(s) failed; asking the model to re-anchor]", file=sys.stderr)
            fb = "\n".join(f"- {r.file}: {r.message}" for r in failures)
            asked = _reply_requests_reread(reply)
            extra_src = ""
            if asked and not reread_done:
                # The model isn't proposing a corrected copy -- it is telling
                # us it never saw the exact text (grep-only context). Arguing
                # back wastes the round; do the reads it asked for itself and
                # put real source in front of it instead.
                reread_done = True
                extra_src = self.reread_for(asked)
                if extra_src:
                    gathered = (gathered + "\n\n" + extra_src) if gathered else extra_src
                    print(f"  (fetched {len(asked)} file(s) it asked to re-read "
                          f"before retrying)", file=sys.stderr)
                    rounds_left += 1   # this round was spent fetching, not fixing
            if rounds_left <= 0:
                break
            rounds_left -= 1
            ask = ("Re-read the source above and output corrected search/replace "
                   "blocks for ONLY the failed edits. Copy the SEARCH text exactly "
                   "from `read` output (without the line-number prefixes)."
                   if not extra_src else
                   "You asked for a `read` before you could write the edit -- here "
                   "it is, in the SOURCE section above, with exact line numbers. "
                   "Now output corrected search/replace blocks for ONLY the failed "
                   "edits, copying the SEARCH text character for character.")
            reply = self.client.chat(
                [
                    {"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user", "content": f"SOURCE\n\n{gathered}\n\nTASK\n{task}"},
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content":
                        f"These edits could not be applied:\n{fb}\n\n" + ask},
                ],
                max_tokens=self.cfg.max_tokens_answer,
                stream=True,
            )
            good = [r for r in results if r.ok]
            results = good + self._apply_once(reply, dry_run, backup_dir)
        return results, reply

    def _apply_once(self, reply: str, dry_run: bool, backup_dir=None) -> list:
        """One parse-and-apply pass, with no repair round-trip.

        Duplicate FILE headers are NOT merged here any more: apply_edits
        groups blocks by path itself and classifies each block individually
        before folding. Pre-merging here glued a rename's create-block to its
        own clear-block under one header, which made the rename pair invisible
        to detection -- the task ended with a blank review page and the old
        files untouched. It also corrupted new-file bodies with stray leading
        blank lines.
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
