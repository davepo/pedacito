"""`pedacito explain`: answer questions about an indexed project's code.

Usage
-----
    pedacito explain src/ -q "how does the sync client retry?"
    pedacito explain . -q "what does src/app.py do?"
    pedacito explain . -q "trace the flow from request to response"
    pedacito explain src/                          # interactive prompt

Where `ask` does tasks and edits and `how` answers questions about Pedacito
itself, `explain` answers questions about the *project the user is working
on*: how a function works, what a file is responsible for, how data or
control flows between parts, and so on. It is always scoped to the indexed
Pedacito project -- it loads that project's existing index (never builds one)
and, through the same gather loop `ask` uses, reads the relevant source before
answering in prose. It produces no edits and no search/replace blocks.

The caller (cli.py) is responsible for the no-index case: if the project has
no index, it points the user at `pedacito index` / navigating to an indexed
project rather than invoking this module.
"""

from __future__ import annotations

from .tools import TOOL_HELP

# The gather phase, reframed from "gather enough to edit" to "gather enough to
# answer". Same tools and navigation hints (FLAT_HINT/CARDS_HINT are appended
# by Agent.gather), only the framing differs.
EXPLAIN_GATHER_BASE = f"""\
You are a code assistant working through a lookup index of a project. You
cannot see the full source; you see a map of it, and you request the exact
parts you need to answer the user's question, one request at a time.

Available actions:
{TOOL_HELP}

Rules:
- Request one action per reply.
- Line ranges in the map are exact. Read them directly.
- `grep`, `callers`, and `outline` tell you WHERE code is. Only `read` shows
  you the exact text. Read the parts your answer will rely on.
- Trace data and control flow across files: use `callers` to find who calls a
  function, `outline` to see a file's structure, and `read` to see bodies.
- Do not guess at code you have not read. If you need it, read it.
- Choose `done` as soon as you have enough to answer the question. Do not browse.
- Fill every field. Use "" for unused strings and 0 for unused numbers.
- Keep `reasoning` under 15 words. It is a note to yourself, not a plan."""

EXPLAIN_SYSTEM = """\
You are a code-explanation assistant for an indexed project. You have a map of
the project and the specific source you asked to see. Answer the user's
question about the code, in prose.

Ground every claim in the SOURCE YOU REQUESTED and the PROJECT MAP below.
Quote exact file paths, function/class names, and (when useful) line ranges.
Do not invent code, files, or behaviour you did not read.

How to answer:
- Start with a direct answer in one or two sentences.
- Explain WHAT a piece of code does, WHY it is written that way, and HOW data
  or control flows through it, as the question asks.
- For "how does X work?" trace the flow step by step, naming the functions and
  files involved in the order they run.
- For questions about a specific file, structure your answer around that
  file's definitions and their responsibilities.
- For data/process-flow questions, describe inputs, outputs, and the order of
  steps, pointing at the code that implements each.
- Use short bullet points or numbered steps when there is more than one part.
- Quote a few key lines (with their line numbers) where they make the answer
  concrete, but do not dump whole files.
- You are only explaining -- do NOT propose changes, and emit no edit blocks.
- If the source does not contain the answer, say so plainly and point to the
  closest code you did read. Never guess.
"""


def answer(agent, question: str, verbose: bool = True, on_step=None) -> str:
    """Answer one question about the indexed project, end to end.

    Runs the gather loop (with the explanation framing above) to collect the
    relevant source, then answers as prose -- no edit format, no apply.
    Returns the answer text. `agent` is an Agent bound to the project's
    index and client (see cli.py).

    `on_step`, when given, is called with the step number after each gather
    step, mirroring `Agent.gather`'s callback so a caller (e.g. the GUI) can
    show live progress.
    """
    gathered = agent.gather(question, verbose=verbose,
                            base=EXPLAIN_GATHER_BASE, on_step=on_step)
    return agent.answer(question, gathered, system=EXPLAIN_SYSTEM,
                        include_edit_format=False, stream=False)
