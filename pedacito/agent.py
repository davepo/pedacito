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
Hard-capped at cfg.max_steps -- when it runs out, the agent moves on to
answering with whatever it has rather than looping forever, which is the
usual small-model failure mode.

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
        budget = self.cfg.max_gathered_chars

        def nudge(step: int) -> str:
            """Text appended to a turn reminding the model how many lookups
            remain, or that this is its last one."""
            remaining = self.cfg.max_steps - step
            return (f"[{remaining} lookups left after this one] Next action?"
                    if remaining else
                    "This is your LAST lookup. Choose 'done' unless you truly need one more.")

        # The first nudge rides along with the map+task message. Every later one
        # rides with its tool result. Mistral-family chat templates reject two
        # user messages in a row, so the sequence must stay strictly
        # system, user, assistant, user, assistant, ...
        messages[-1]["content"] += "\n\n" + nudge(1)

        for step in range(1, self.cfg.max_steps + 1):
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
                if not did_read and step < self.cfg.max_steps:
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
                if not gathered and step < self.cfg.max_steps:
                    # A `done` whose every lookup failed to produce anything
                    # usable (bad line range, unresolvable file, ...) would end
                    # the loop with an empty SOURCE section, and the answer
                    # phase can only refuse -- which surfaces as a blank review
                    # page ("the model answered without proposing any edits").
                    # Tell it what went wrong so it can pick a real range.
                    # Only fires when nothing succeeded at all; a mix of one
                    # good read and some failures is normal navigation noise.
                    nudged = True
                    last_err = next((g for g in gathered if " failed: " in g), None)
                    detail = (f"Your last result said: {last_err}\n" if last_err else "")
                    messages.append({"role": "assistant", "content": _compact(stepjson)})
                    messages.append({"role": "user", "content":
                        "None of your lookups returned any source, so you still "
                        "cannot see the code you want to change. " + detail +
                        "`read` the exact lines you need (ranges come from the "
                        "map or an outline), then say done."})
                    if verbose:
                        print("  (all lookups failed; asking it to read real "
                              "lines before answering)", file=sys.stderr)
                    continue
                break

            obs = dispatch(self.index, self.cfg, stepjson)
            if action == "read" and not obs.startswith("ERROR"):
                did_read = True
            # An ERROR observation is navigation feedback ("that range does
            # not exist"), not source. Carrying it into the answer prompt as
            # gathered text makes the SOURCE section look populated while it
            # contains no code at all -- the model then answers from the map
            # alone and proposes nothing, which surfaces in the GUI as a
            # blank review page. Record failures compactly instead.
            if obs.startswith("ERROR"):
                arg = stepjson.get("file") or stepjson.get("pattern") or stepjson.get("symbol") or ""
                gathered.append(f"--- {action} {arg} failed: {obs[:160]}")
                messages.append({"role": "assistant", "content": _compact(stepjson)})
                messages.append({"role": "user",
                                 "content": f"{obs}\n\n{nudge(step + 1)}"})
                continue
            if len(obs) > budget:
                obs = obs[:budget] + "\n[truncated: context budget reached]"
            budget -= len(obs)

            arg = stepjson.get("file") or stepjson.get("pattern") or stepjson.get("symbol")
            if action == "outline":
                # Outlines are navigation, not evidence. The model needs them to
                # decide what to read; the answer phase needs the code it read.
                # Carrying a 2,400-token outline into the answer prompt is pure
                # waste, so record only that it happened.
                gathered.append(f"--- outline {arg} (used for navigation; "
                                f"{len(self.index.in_file(self.index.resolve_file(arg) or ''))} units)")
            else:
                gathered.append(f"--- {action} {arg}\n{obs}")

            messages.append({"role": "assistant", "content": _compact(stepjson)})
            messages.append({"role": "user",
                             "content": f"RESULT:\n{obs}\n\n{nudge(step + 1)}"})

            if budget <= 0:
                if verbose:
                    print("  [context budget exhausted; moving to answer]", file=sys.stderr)
                break

        self.read_any = did_read
        return "\n\n".join(gathered)

    # ------------------------------------------------------------------ phase 2
    def answer(self, task: str, gathered: str, stream: bool = True,
               system: str | None = None, max_tokens: int | None = None) -> str:
        """Ask the model to answer the task, or produce search/replace edits,
        given the project map and whatever source `gather()` collected.

        `system` and `max_tokens` override the defaults for special flows that
        need a different brief than "edit this existing code" -- e.g. resuming
        an unfinished scaffold (`Agent.resume_task`) or writing one whole file
        of a new project (see scaffold.py)."""
        repo_map, _ = self.index.map_for_prompt(self.cfg)
        user = (
            f"PROJECT MAP\n\n{repo_map}\n\n"
            f"SOURCE YOU REQUESTED\n\n{gathered or '(nothing was looked up)'}\n\n"
            f"TASK\n{task}"
        )
        print(f"[answer context ~{estimate_tokens(user)} tokens]", file=sys.stderr)
        return self.client.chat(
            [{"role": "system", "content": system or ANSWER_SYSTEM},
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
        results = self._apply_once(reply, dry_run, backup_dir)

        for _ in range(repair_rounds):
            failures = [r for r in results if not r.ok]
            if not failures:
                break
            print(f"\n[{len(failures)} block(s) failed; asking the model to re-anchor]", file=sys.stderr)
            fb = "\n".join(f"- {r.file}: {r.message}" for r in failures)
            reply = self.client.chat(
                [
                    {"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user", "content": f"SOURCE\n\n{gathered}\n\nTASK\n{task}"},
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content":
                        f"These edits could not be applied:\n{fb}\n\n"
                        "Re-read the source above and output corrected search/replace blocks "
                        "for ONLY the failed edits. Copy the SEARCH text exactly."},
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
