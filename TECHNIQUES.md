# Techniques: getting the most out of small models

Pedacito is built for local models in roughly the 4B–27B range, usually with a
tight context window and no spare VRAM for parallel reasoning. That changes the
whole calculus: techniques that help a frontier model for free are often too
expensive, too slow, or actively harmful here.

Everything below is judged on one question: **does it raise task success and
answer faithfulness per token and per second?** A technique ships only when it
wins that trade-off, and it is gated behind a config flag, a model card, or a
measured win, never enabled on trust. This page lists what Pedacito actually
does, and, at the end, the well-known techniques it deliberately does *not*.

The reasons and measurement plan behind each choice were drafted as internal
design notes. They are not part of this repository, so the rules below are
the public record.

---

## The guiding constraints

Four rules shape every decision:

1. **Token economy.** Every token the model re-reads costs time and window
   space. The harness tries to hand the model only the relevant lines, once.
2. **Prompt-cache reuse.** Local servers cache the prompt *prefix*, and the
   first user turn is the fold point. A technique that rewrites the head of the
   conversation throws the cache away for the whole task.
3. **Bounded loops.** Every recovery is a fixed number of rounds, never an
   open-ended "keep thinking".
4. **Deterministic before model-based.** Code correctness is checkable against
   source and tests, so Pedacito verifies for free first and only pays for a
   model's opinion where there is nothing to check mechanically (prose).

---

## What Pedacito does

### 1. A harness-managed retrieval loop, not free-form reasoning

The `gather` phase is a ReAct loop, but the harness owns it. Each step is a
grammar-constrained JSON `{action, file, pattern, symbol, start, end, ...}`
followed by a `RESULT:` observation, until the model says `done` or hits a
budget. Because the schema constrains the *choice of action* (not just the
format), a small model cannot wander into unsupported actions. Recovery nudges
fire once each for the classic stalls, re-reading the same range, repeating a
step, a run of errors, or a `[STUCK:]` on identical actions.

Two action-independent global leases back-stop the loop when `max_steps = 0`
removes the fixed lookup cap: `max_total_model_calls` (default 30) is a hard
ceiling on LLM invocations per task, and `max_no_progress_steps` (default 4)
bounds consecutive gather steps that neither add a new unread line span to
the read ledger nor resolve a new symbol, a run of greps, a run of errors,
and a run of identical steps all count alike. Whichever ceiling trips first
ends gather cleanly with a harness note appended to the conversation, and the
run moves on to the answer phase with what was gathered. One step before the
no-progress lease trips, a single high-salience loop-guard redirect names a
different strategy first (§16); the hard ceiling is unchanged.

Gather also carries a free **requirement evidence state machine**. Each task
clause is tracked from its first mention to its closing evidence: an edit
requirement walks `UNLOCATED -> LOCATED -> READ -> EDITED -> VERIFIED`, while
an investigation/explain clause walks `UNLOCATED -> (LOCATED) -> READ ->
REPORTED`. Search, find, outline and caller results advance a clause to
`LOCATED` only when their query/result names its terms; a displayed read span
advances it to `READ`; a clean patch touching its file/terms advances it to
`EDITED`; passing verification advances it to `VERIFIED` (prose that answers
a read investigation reports it). No transition costs a model call, every
one is derived from a tool result or patch the harness already has.

When the model says `done`, or gather stalls with budget left, the oldest
requirement still `UNLOCATED` or `LOCATED` gets the cheapest deterministic
action: read a located anchor directly, otherwise `grep` its most specific
term, then read the anchor that search located. The result is APPENDED at the
conversation tail (never a rewrite of earlier turns), bounded to one assist
per requirement, and only on multi-requirement tasks, a single clause still
uses the ordinary read nudge / prefetch path. Gather's SOURCE log itself stays
append-only; packing evidence is a phase-boundary operation (section 4), so
the prompt-prefix cache survives the whole lookup loop.

Why it helps: the loop itself is the biggest intelligence win, a ranked map,
symbol resolution, and bounded lookups get a 4B model to the right 300 lines of
a 12,000-line project without it having to hold the whole thing in context.

### 2. Reasoning *before* acting (zero-cost "chain of thought")

Small non-thinking models commit to an action before writing any rationale.
Four changes move the deliberation in front of the decision at roughly zero
extra cost:

- **Reason-first steps** (`step_reasoning_first`, on by default). The `reasoning`
  field is emitted *before* the action, so the model deliberates before it
  commits. The model card still picks the ordering per model, since thinking
  models already deliberate and gain nothing.
- **Edit-plan brief** (`edit_plan_brief`, on by default). Before the SEARCH
  blocks, the answer states 1–3 lines: the file, the lines or symbol it will
  change, and why. This is the only pre-edit reasoning the answer phase gets,
  and it replaces a self-contradictory "explain what you did" line.
- **Symbol-anchored incremental edit-verify microloops**
  (`incremental_edit_loop`, on by default; model cards enable it for weaker
  edit-format scores). On a multi-hunk or multi-file task the model first
  emits structured edit intents, `[{file, symbol, intended_change}]`. The
  harness resolves one symbol at a time, appends that symbol's exact current
  body at the conversation tail, and asks for **one** SEARCH/REPLACE hunk at
  `temperature_answer=0`. It dry-applies the hunk, runs `verify.py` and the
  AST/tree-sitter syntax gate, writes it only if clean, then replaces the
  temporary source with a one-line `[EDITED file:symbol]` receipt before
  moving to the next symbol. The full impacted-test and lint pass still runs
  once after the last hunk. Section 5 covers why the single-hunk boundary is
  the reliability win.
- **Worked examples** (`fewshot_steps`, off by default). Three short worked
  steps from the project itself (grep a symbol, read it, done) shown once in the
  first user turn, so grammar/native mode also teaches *good action choice*,
  not just well-formed output.

### 3. Deterministic grounding and verification (free checks)

Before any edit is written, and, in shadow mode, for prose answers too, a set
of zero-call checks catches the specific ways small models go wrong:

- **Unread edits.** A SEARCH block that quotes lines the model never `read` is
  flagged: it is guessing, not editing. The read ledger tracks exactly which
  line spans were shown. When the flagged region touches what the evidence
  store recorded (§11), the repair re-ask gets those exact lines re-read from
  disk appended at its tail.
- **Ledger-accurate token-budgeted reads.** Every `read`/`context` observation
  is clipped to `max_read_tokens` (default 1200) *before* the source is shown
  and before the span ledger sees it. The clip uses the server's calibrated
  tokenizer when `pedacito calibrate` measured one, ~3.5 chars per token
  otherwise, and truncates strictly at whole-line boundaries: the header is
  rewritten to the delivered range and a `Use next_start=… to continue
  reading.` hint tells the model where the cut landed. The ledger records
  `(start, actual_end)`, not the requested range, so lines that were
  clipped away are never falsely credited as read, and an edit quoting them
  stays flagged as an unread edit.
- **Citations.** A `file:line` claim about a file (or line) the model never saw
  is a warning; an invented path that doesn't resolve is reported too.
- **Placeholders, stale references, moved imports.** `TODO`/`pass` stubs the
  task didn't ask for, a renamed function still called elsewhere, a moved file
  still imported by its old path.
- **Coverage and requirement states.** Every extracted task clause carries an
  evidence state, not just a post-answer keyword check: an edit requirement
  must reach `EDITED`/`VERIFIED`, and an investigation clause must reach
  `REPORTED`, before it counts as addressed. The advance rules are strict
  enough to be useful: a search can only supply `LOCATED`, a displayed read
  `READ`, a dry-applied patch `EDITED`, and a patch accepted by the checks (or
  the post-edit tests/lint) `VERIFIED`. Coverage then warns about the clauses
  whose state never got there, with the state named in the message.
- **Verdict filter.** When a model *verifier* reports a `DEFECT`, it only counts
  as an error if the cited file resolves and the line is inside it, a
  hallucinating small-model verifier can't feed the repair loop.
- **Fuzzy-match warnings.** A block that anchored only approximately is flagged
  for review, never silently credited as clean.
- **Prose shadow verification.** For no-edit answers (`explain`), the citation
  and identifier checks run and are *logged only*, never acted on, until they
  prove precise enough to promote.

### 4. Harness initiative: save the model its most error-prone steps

The harness already knows what the task names, so it can fetch it instead of
making a small model hunt:

- **Step-0 prefetch** (`prefetch_targets`, off by default). Task-named files and
  symbols are read through the normal tool path *before* the first model step,
  so the read ledger is populated and edits can quote real source.
- **Post-gather prefetch** (a narrow P6 trigger). If the model finishes gather
  with named targets still unread, or never read any ranked top-3 file, the
  harness does *one* extra bounded read before answering, a single expansion,
  never a new loop.
- **Requirement completion assist.** On a multi-requirement task, a `done` (or
  a stall exit) with the oldest clause still `UNLOCATED`/`LOCATED` gets one
  deterministic lookup: read the located anchor, else grep the most specific
  requirement term and read what it locates. The observation is appended, one
  assist per clause, so the model never spends a call deciding what to do next
  and the cached prefix is untouched.
- **Deterministic utility source packing.** At the gather -> answer/edit
  boundary the SOURCE copy is packed by evidence utility instead of FIFO.
  Exact spans for located/read requirements and active failing diagnostics are
  **pinned**; every other item is scored by requirement-keyword overlap,
  call-graph proximity, and new information per token, with token-Jaccard/MMR
  discounting near-duplicates. A definition plus its caller chain therefore
  survives ahead of scattered unrelated reads. Dropped items leave whole-item
  re-read pointers and the digest always survives. This runs only on the
  phase-boundary copy, never inside gather, so the prompt-prefix cache stays
  intact.

### 5. Execution-grounded repair (objective error signals)

The cheapest reliable feedback is the code itself:

- **Verify→repair.** Deterministic findings feed one bounded re-ask; the model
  corrects against concrete, quoted problems rather than a vague "try again".
- **Post-edit tests and lint.** After an edit, impacted tests run and
  linter/type-checker output is diffed against a pre-edit baseline, and
  failures feed the repair rounds. An applied edit that breaks tests is an
  objective signal no model self-critique can fake.
- **Immediate AST syntax gate.** Before any linter or pytest subprocess starts,
  the dry-applied in-memory buffers are parsed in-process: `ast.parse` for
  Python, the existing tree-sitter grammar for every other supported language.
  A file that parsed before but does not parse after is rejected immediately
  (typically under 5 ms), and repair receives only the parser diagnostic plus a
  single 10-line window around the error. The subprocess check is deferred
  until the patch parses. Because the gate compares before and after, a file
  that was already broken can still be repaired, only a *new* syntax error
  trips it.
- **Symbol-anchored incremental edit-verify microloops.** A multi-hunk answer
  is not one atomic gamble. The intent list names each `{file, symbol}` target;
  the harness then runs a bounded loop per symbol: append that definition's
  exact current source, take one greedy hunk, dry-apply it, run `verify.py`
  plus the AST gate, write it only when clean, and replace the temporary source
  with `[EDITED file:symbol]` before the next target. A rejected hunk repairs
  only itself; it cannot poison the hunks that follow. Once every symbol has
  landed, the normal impacted-test and lint verification runs over the whole
  changed set, so the final gate still judges the combined patch.
- **Failure-identity diffing.** Pytest's stable node ids (plus a normalized
  assertion signature when no id survives truncation) identify each failure
  across runs. The post-edit run is classified against the pre-edit baseline as
  `new`, `fixed`, `pre-existing`, or `still_failing`. A `new` failure triggers
  repair even when the total failure count is unchanged or lower, the scalar
  count cannot see "one old failure fixed, one new failure introduced".
- **Traceback Repair Packets.** Instead of the last 1,800 characters of raw
  output, repair gets a structured packet: the failing test's node id and
  normalized signature, a ~15-line window around the failing assertion, and a
  ~15-line window around the innermost implicated project traceback frame.
  Frames from `.venv`, `site-packages`, `dist-packages`, `node_modules` and
  other environment directories are discarded, as are unrelated pre-existing
  failures, so the model sees the code that actually broke and not a
  transcript.

Why single-hunk generation prevents patch corruption on 4B–9B models: a large
reply makes three failures correlated. The model must copy several SEARCH
regions exactly while its attention is spread across unrelated files; one stale
line number or indentation drift makes a later block anchor in the wrong place
or not at all. The output cap truncates the tail mid-block, so the blocks after
the cut are silently lost, and a small model asked to repair the combined reply
often rewrites blocks that already worked. Keeping exactly one symbol and one
hunk in flight removes that coupling: the current body is the last thing in the
prompt (fresh from disk, not from memory), the hunk is short enough to finish
before the cap, and verification either accepts it or rejects only that symbol.
The one-line receipt preserves the *fact* of the edit without re-sending its
source, so context stays small and later hunks start from the real post-edit
file rather than a stale multi-file patch.

### 6. Sampling the right amount

- **Per-phase temperature** (`temperature_answer=0`, `temperature_repair`). The
  answer and repair phases copy SEARCH text character-for-character, so they
  decode greedily; the gather phase keeps a little diversity for when it gets
  stuck.
- **Best-of-N** (`best_of`, off by default). A weak model may sample a second
  edit reply at higher temperature, both scored deterministically, keeping the
  best. No majority voting, see below.

### 7. Adaptive defaults from a measured card

`pedacito calibrate` runs a short scripted suite, tool choice, JSON adherence,
edit format, planted-injection and honesty, long-context retrieval,
continuation, a thinking A/B, and capabilities, and stores a per-model card.
The card then picks `prompt_pack`, `answer_mode`, `best_of`, `verify_answer`,
`tool_mode`, `fuzzy_match`, and `edit_style` *only for settings the user left
at their default*, with an explicit config value always winning. The direction
is deliberate: the *weakest* models get the *most* safety net.

### 8. Prompt-cache layout

The system prompt and first user turn are byte-identical within a gather, and
history is folded past one point, so the server's prompt-prefix cache survives
every step. On hosted providers, explicit `cache_control` breakpoints are placed
at the stable prefix (and the provider table decides which API accepts them),
so the repeated re-sends of the gather prefix cost as little as possible.

### 9. Shallow failure memory

A failed task is remembered as harness facts, what was tried, why, which files
were touched, and injected (capped, cache-stable) into similar later tasks.
The memory is *facts*, not model-written reflection, which a small model would
get wrong.

### 10. Discovery and role routing

`pedacito models scan` + `pedacito profiles` find servers, calibrate models,
and recommend which profile to use for each role. This matters for small models
because the highest-leverage trick is often *routing* the hard job, the
`verifier` or `planner`, to a larger local or hosted model, while the cheap
work stays on the small worker.

### 11. Content-addressed evidence and append-only rehydration

Two moves that share one mechanism: the harness remembers the *exact bytes* it
showed the model, hashed, and the only way source reaches the model after
gather is by **appending** at the conversation tail.

- **Evidence store.** Every `read`, `context` and prefetch observation is
  parsed *after* the token clip: the displayed lines, stripped of their
  `NNN | ` prefixes, are SHA-256'd and recorded as
  `EvidenceSpan(file, start, end, content_sha, exact_text, read_step)`. The
  span is content-addressed, its hash pins exactly what the model saw, so
  the harness can later prove whether the file moved, instead of trusting the
  model's (or its own) memory of it. Clipped-away lines are never hashed or
  credited: the ledger and the store record the same delivered range.
- **Explicit rehydration (`NEED` directive).** The answer, repair and
  verifier phases have no tools, but the model may emit one line:
  `NEED shop/pricing.py:120-168`. The harness fetches *only* those lines from
  disk under the same `max_read_tokens` budget as a normal read, compares
  them against the recorded hash (`[unchanged since step N]`, or a
  `[content changed …]` note with both hashes when the file moved since),
  credits the read ledger, and APPENDS the exact span as a fresh turn pair at
  the tail before re-asking. Bounded by `max_rehydration_rounds` (default 2;
  0 disables the directive), a few spans per round.
- **Implicit rehydration.** When the deterministic checks flag an
  unread-region edit or a fuzzy-matched anchor, and the flagged span is
  covered by, or sits within ±30 lines of, text the store recorded as
  *displayed*, the repair prompt automatically gets that region's fresh exact
  bytes appended at its tail before the model is asked to fix the patch. The
  ledger credit means a corrected quote stops reading as a guess on the next
  check. Where no recorded evidence is anywhere near, nothing rehydrates: a
  model that never read a file should have `read` it during gather, not be
  handed unseen code to copy from.

Why it helps: this is Guiding Constraint #2 applied to the phases after
gather. A reply that needs one more span gets it, instead of hallucinating a
quote, and it arrives after the last cached byte, so every earlier byte of
the conversation still rides the server's prompt-prefix cache: the re-ask
costs a cache hit, not a fresh prefill. It also makes the free checks
(Constraint #4) self-repairing: an unread-edit finding near real evidence now
comes with its ground truth instead of only a complaint. Token economy (#1)
holds because the rehydrated payload is triple-bounded, spans not files,
the read token budget, and round and span caps.

### 12. Index-driven retrieval: sidecars, candidate-first RRF, skeleton reads

Three upgrades to the indexing layer (chunker / search / semantic / tools),
each extending Guiding Constraints #1 and #4, *deterministic first, bounded
always*:

- **Persisted lexical body sidecar.** BM25 (`search.py`) deliberately refused
  to score source bodies: a loaded index hydrates source lazily, and a
  ranking that changed depending on whether a chunk happened to have been
  read from disk yet was a determinism bug, not a feature. Instead of giving
  up on body words, chunking (`chunker.py`) now extracts a **bounded lexical
  sidecar** per unit, split identifier tokens, docstring/comment words,
  string-literal words (error messages, prompts, config values) and
  config/dict keys, categories capped separately so an identifier-heavy body
  cannot evict the strings a query would actually look for, and it ships
  inside `index.json`. `search.py` scores it as the body field at the
  existing low weight (0.3, behind name 3.0 / sig 1.5), and matched-term
  reasons now see it too. Consequence: `find`, the BM25 miss-suggestions, and
  file-level task ranking can hit an error string, a config key, or an
  internal call name even with `--no-summaries`, and the ranking is
  **100% hydration-stable**, the sidecar travels with the record, so a
  freshly built index and a reloaded one rank identically. Chunks indexed
  before the sidecar existed simply score no body terms; the next re-chunk
  fills them in.
- **Candidate-first hybrid retrieval with RRF.** Embeddings used to be a
  last resort that either never fired (any free suggestion suppressed it) or
  embedded *every* chunk on first use, one no-hit grep cost a pass over the
  whole project. Now a no-hit `grep` first builds a bounded top-K (K = 50)
  candidate short-list from the *free* matchers, BM25 over names,
  summaries and the sidecar, similar symbol names, call-graph hits, and
  only the query plus those K units are hydrated and embedded. The lexical
  and embedding rankings are merged with **Reciprocal Rank Fusion**
  (`score = Σ 1/(60 + rank)` per list): RRF needs ranks, not comparable
  scores, so BM25 points and cosine similarity fuse without any scale
  tuning, and a unit that ranks well in *both* lists beats one that only one
  signal likes. A no-hit grep now costs at most K + 1 embeddings instead of
  N + 1; chunk-sha embedding caching, the similarity floor, and the
  remember-the-permanent-failure behavior (a server without `--embeddings`
  is not retried within a run) are all unchanged. Candidates whose cosine
  falls below `semantic_min_score` are dropped rather than promoted by a
  lexical vote alone, a pure-name miss is still better served by the free
  suggestions.
- **Hierarchical chunks and skeleton views.** Every chunk now persists its
  place in the tree: `parent` (the owning definition of a method), and
  `part_of` / `part` (which split unit a `name#2` fragment continues).
  `read` uses that lineage twice. **Adjacent fragments auto-merge:** reading
  a symbol whose definition was split for the retrieval budget returns the
  whole merged span, so a model can never act on a function's head while
  its tail went unseen, and no extra read step is spent stitching parts.
  **Skeleton view:** when a class or module exceeds the read budget, `read`
  returns a deterministic AST skeleton, signatures, first docstring lines,
  and `[lines X-Y]` pointers with every body folded, instead of a truncated
  first page. The breadcrumb (file, unit, full span, module docstring) is
  printed exactly once and no parent/child source is duplicated. A
  500-line class arrives as ~35 lines of named ranges; the model then reads
  exactly the one method it intends to edit. Zero model calls (all facts
  come from the index), bounded (row and char caps), and gated behind
  `read_skeleton` (on by default; `false` restores plain truncation).

Why it helps: all three keep the retrieval contract small models depend on:
the same query must return the same ranking in the same run *and* across
runs, every paid step must be bounded, and the output a model re-reads must
be only what it needs. The sidecar buys body-word recall without importing
nondeterministic hydration; candidate-first RRF buys by-meaning search
without the N-embedding bill it used to cost; skeletons buy a map of a giant
unit at ~7% of its tokens, replacing "read page 1 of 3" hunts with one
targeted read.

### 13. A deterministic inspection toolkit: `outline`, `usages`, `git_diff`

Three additions to the gather action set target the most token-expensive
small-model habits: reading a whole file to find its shape, running several
blind greps to find callers, and asking for a free-form review of work it
cannot see.

- **`outline` is a compact table of contents.** It renders one line per
  class, function and method with the exact `[start-end]` range, no bodies and
  no docstrings: `class Server [4-9]`, `def start(self) [5-6]`. A whole-class
  chunk is expanded from the Python AST so its methods still get exact ranges;
  a directory or glob keeps the existing file-card view. The output comes
  entirely from the chunker/AST index, no model call, no source read, so a
  several-hundred-line file arrives as a short map that tells the model
  exactly which lines to `read` next.
- **`usages` is one-step cross-reference.** Given a symbol, it joins the call
  graph, the resolved import graph and the persisted lexical sidecar, then
  reports deduplicated exact `file:line` call sites and import lines. It
  handles methods (`Store.get`) and module-level names, caps the result, and
  marks the evidence as name-based so the model still reads before editing.
  One `usages` step replaces the several blind `grep` calls a small model
  otherwise spends on the same question.
- **`git_diff` is read-only working-tree inspection.** It runs
  `git status --short` plus a bounded `git diff` (staged and unstaged, against
  `HEAD` when one exists) and caps the diff by `max_read_tokens`; an optional
  `file` narrows both. It is distinct from `diff` (Pedacito's own pre-write
  snapshots) and from `git` (general read-only history), and it gives the
  model immediate evidence when the task says "review" or "fix recent work".

All three are navigation-only, deterministic and bounded. `outline` and
`usages` are in the always-on core; `git_diff` is optional and can be pruned
by the rule in §15. `git_diff` is offered only in a git repository and obeys
`allow_git`.

### 14. Repository micro-facts with contradiction retirement

Adapted from Microsoft Research's **TokenEconomizer** Micro-Skills: a session
should not pay a model call to rediscover facts the harness can observe for
free. Pedacito keeps the environment/command slice of that idea in
`.pedacito/micro_facts.json`:

```json
{"fact_id": "cmd:…", "text": "Use `python -m pytest -q` here.",
 "confirmations": 2, "contradictions": 0, "last_seen": 1712345678.0}
```

- **Promotion.** Every command whose outcome the harness sees (`run`,
  impacted tests, post-edit checks, `auto --verify`) is recorded as pending.
  It is promoted to a confirmed fact when it succeeds after an earlier
  failure, or when it succeeds across at least two runs. This is exactly the
  `pytest` vs `python -m pytest` case, a required flag, or a working linter
  invocation: the truth is measured, never inferred by a model.
- **Injection.** Confirmed facts are rendered into the stable session prefix
  at task start (the system prompt, and the rebuild answer prompt), ordered by
  confirmation count, capped by `max_micro_fact_tokens` (default ~150 tokens)
  and labelled as hints. Repeating facts cost no extra model calls and sit in
  the cached prefix.
- **Contradiction retirement.** If a later run executes a stored command and
  it fails, the fact's `contradictions` count is incremented and the fact is
  moved out of the active list into `retired` immediately. It is never
  injected again; a fresh promotion has to earn its place. Stale environment
  advice therefore cannot outlive the run that disproved it.

The file is deterministic JSON under `.pedacito/`, so it can be inspected,
edited or deleted like the rest of the harness state. `micro_facts = false`
turns recording and injection off.

### 15. Dynamic tool schema pruning

A 4B–9B model chooses worse when every optional tool definition is in front
of it on every turn. Pedacito now tracks per-repository usage in
`.pedacito/tool_stats.json` (`tasks`, `uses`, `last_task` per action) and
prunes the prompt schema deterministically:

- **Core actions never disappear:** `read`, `grep`, `find`, `callers`,
  `context`, `outline`, `usages`, `sessions`, `todo` and `done`.
- **Config-disabled actions are already omitted**, as before.
- **An enabled optional action is pruned** once it has gone unused for
  `tool_prune_after_tasks` (default 10) repository tasks *and* the current
  task text contains none of its trigger words: no URL/`http`/`web` for
  `fetch`, no `git`/`uncommitted`/`diff`/`review` for `git_diff`, no
  `lsp`/language-server wording for `lsp`, no `subagent`/`delegate`/`research`
  for `task`, and so on. A trigger word brings it back for that task; a recent
  use keeps it active regardless of the prompt.
- **The help text follows the schema.** `prompts.tool_help` lists only the
  actions `schema_for` actually offers, so pruning removes both the tool
  definition and its prose advice. That matters most in native tool mode,
  where the definition is sent verbatim, and for models that imitate nearby
  tool descriptions.

`dynamic_tool_pruning = false` restores the full enabled schema. The stats are
local, small and per repository; nothing is sent to a server.

### 16. Pre-lease loop-guard redirects

TokenEconomizer's Loop Guard stops a trajectory from repeating the same
failing strategy forever. Pedacito already has the hard lease
(`max_no_progress_steps`, §1); this adds its one-shot pre-lease redirect
(`r = 1`):

- When the no-progress streak reaches `max_no_progress_steps - 1`, one step
  before hard termination, the harness appends a single high-salience
  `[LOOP GUARD: …]` observation to the conversation. It names the number of
  unproductive steps, forbids repeating the previous search/lookup strategy,
  and suggests a different tool (`outline`/`usages`) on an unread requirement
  target or an immediate `done`.
- It fires **at most once per turn**. The message is appended at the tail,
  never written into cached history, and costs no extra model call; the model
  simply gets one last chance to change strategy before the existing lease
  ends gather cleanly.
- The redirect does not weaken the bound: if the next step still adds no new
  source line and resolves no new symbol, the normal no-progress note fires
  and the run moves to the answer phase with what it already gathered.

### 17. The Prompt/Spec Architect

Small models execute detailed specs well but invent structure when handed a
one-line idea -- and in `auto`, that invented structure is what Step 7 will
contradict. `pedacito/spec.py:refine_prompt` spends **one** schema-constrained
call up front to turn a brief into a fixed five-section spec: *Objective /
Files to Create or Modify / Interface Contracts / Requirements / Edge Cases /
Verification*, with the original request appended verbatim so nothing is lost.
(`--refine` on `ask`/`new`/`auto`; the GUI composer's **✦ Refine Prompt**
button, with **↶ Undo Refine** to restore the brief.)

- The user message carries a zero-LLM repo snapshot: up to 40 scored file
  paths plus AST outlines of the files the prompt *mentions*. The architect
  plans against the real tree ("modify `parser.py`", not `parser_v2.py`)
  without a search phase.
- The system prompt is written for 9B--35B instruct models: every contract is
  one exact signature line, one spelling, defined once -- the vocabulary the
  later steps will be locked to (see §18). Numbers and names the brief only
  implies must be made explicit; extra features are forbidden.
- **It cannot dead-end.** If the server is unreachable or answers nothing
  usable, a deterministic fallback shapes the same sections from the snapshot,
  so the button and the flag always return a spec of identical shape.
- Cost: one call of ~3k tokens per task the user *starts*, replacing several
  repair rounds caused by an under-specified first task.

### 18. Contract-first interface locking & cross-file verification

The documented killer of multi-step small-model plans is name drift: Step K
forgets Step 1's function name, re-reads files at full token cost to
rediscover it, or calls a signature that no longer exists. `interface_lock`
(default on) closes that hole with a *lock* and a *gate*, shared by `auto`
and `new --complex`:

- **The lock.** Before every task, `extract_project_contracts` renders the
  project's exact public surface -- `def parse_line(text, sep=',')`,
  `class TableParser: __init__(encoding='utf-8')`, JS/TS `function`
  signatures -- into a ~300-token block headed
  `[PROJECT INTERFACE CONTRACTS — DO NOT BREAK SIGNATURES]` and injects it
  into the task prompt. Files touched by the last step sort first so the
  freshest contracts survive the cap; `new`'s build passes read the same map
  ("PUBLIC API"). Deterministic AST/regex parsing: zero model tokens, no
  subprocess, re-run per task.
- **The gate.** After every applied task, beside the existing syntax gate,
  `verify_cross_file_contracts` walks every local import and imported-name
  call in the project: a `from parser import parse_row` that references a
  symbol the module does not define, or a call whose argument count cannot
  satisfy the real signature (defaults, `*args`/`**kwargs` respected), fails
  the task *immediately* -- before any verify subprocess -- and the exact
  diagnostic (`ContractError in cli.py:18: imports 'parse_row' from
  parser.py, but parser.py only defines ['parse_line', 'TableParser']`) is
  fed straight into the repair prompt instead of letting the error propagate
  through three more steps.
- **Deliberately conservative.** Unknown absolute imports are treated as
  third-party, star-imports and shadowed names are trusted, classes without
  an explicit `__init__` are never arity-checked: a false positive would
  waste a repair round on a working project, which the technique exists to
  prevent. The same drift listing also extends `new`'s `fix_api_mismatches`
  repair and its final consistency warning.

Together the two halves turn interfaces into a shared, checked fact of the
plan rather than something each fresh context must rediscover.

---

## Techniques that exist but are not a good fit

These are real methods with real literature behind them. Pedacito rejects them
not because they never work, but because they cost more than they return for a
4B–27B model on one GPU, or because a cheaper deterministic substitute already
covers the need.

| Technique | Why not, here |
|---|---|
| **Full Chain-of-Verification** (generate verification questions, answer them independently) | Adds calls without new information for *code*: correctness is checkable against source and tests, so the questions just re-derive what the deterministic checks already know. |
| **Tree / Graph of Thoughts** (branch, score, backtrack) | Too many calls and too much context; a local window can't keep multiple branches alive. The one useful fragment, a second plan on failure, is a small deterministic lint + rescore, not a search tree. |
| **Majority-vote self-consistency** (sample N, take the modal answer) | `best_of` already samples and re-ranks; voting over code *diffs* is brittle, and small-model errors are correlated, so "majority" doesn't buy independence. |
| **Always-on model self-critique / Reflexion** (the model reviews its own full draft) | Small-model self-critique is unreliable, and re-sending the full draft/source costs tokens and can break the warm prefix. Deterministic checks + one bounded repair cover the same ground more cheaply. |
| **Model-written reflection notes in memory** | Same unreliability; harness-fact records (what was tried, why) give the benefit without asking the model to write truthful prose about itself. |
| **Model-based compaction** (`compaction = "model"`) | One extra call per fold, and a small model's summary drops exactly the line-level facts the edit phase must later quote. The deterministic extractive fold is the right default. |
| **Parallel `n > 1` sampling** | Local servers split one GPU across slots, so latency barely improves; `best_of` is off by default anyway. |
| **Logprob / entropy confidence as an effort trigger** | No logprob plumbing exists, server support differs across runtimes, and the extra fields would perturb replay. The cheap per-task triggers (no edit, no locate, repeated steps) are enough. |
| **LSP default-on** | External language servers are slow to start and optional; the step-0 prefetch already recovers most of the "go to definition" value without one. |

The through-line: **if a deterministic, harness-side check can answer the
question, Pedacito uses that before paying for a model call, and it never
spends a call where a small model's own opinion is the weak link.**
