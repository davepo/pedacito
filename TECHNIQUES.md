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
  a "project interface contracts, do not break signatures" banner (`CONTRACT_HEADER`
  in `spec.py`) and injects it
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

### 19. The static analysis gate: a linter-shaped repair trigger for free

The inner loop already refuses to start a test subprocess when the dry-applied
buffer will not parse (§ around `check_syntax`). The next-costliest class of
small-model mistake survives parsing: a line that says `json.dumps(...)` in a
file that never imported json, a typo'd name, an `__all__` entry that does not
exist. Each one is a guaranteed crash -- and the harness only learns of it
after a full interpreter start, collection, and often seconds of unrelated
tests, or after a repair round trips through the model. `static_gate`
(default on) inserts a zero-token gate between the two:

- **Engine.** `spec.run_fast_static_gate` runs `ruff check --isolated
  --select F821,F822,F823,F402,F404` over only the changed `.py` files --
  five pyflakes rules, no project config, a `static_gate_timeout` (default
  3 s) ceiling. When ruff cannot run at all, exits oddly, or exceeds the
  timeout, a pure-stdlib fallback (`ast_static_errors`) resolves the same
  questions with an two-pass scope walker: pass one builds the module/class/
  function/lambda/comprehension scope tree and collects every binding and
  every load; pass two resolves loads through the scope chain. Class scopes
  are correctly invisible to methods (the classic `return attr` bug),
  walrus/comprehension/global/nonlocal bindings are respected, and -- house
  rule -- anything ambiguous is skipped: star-imports, `exec`/`eval`/`locals`
  anywhere in the file, names bound in *any* function scope (a lazy import
  may still land first), `global`/`nonlocal` declarations, pathological
  files. ruff's newer `invalid-syntax` label is normalized to E999 so both
  engines (and the baseline) share error keys; common forgotten imports get
  a one-line hint appended (`-- did you forget `import json`?`) in either
  engine's output.
- **Baseline diffing.** A gate that blamed the project's pre-existing
  breakage on every task would be noise, so the loop records the errors the
  project *already had* before each step (`_capture_static_baseline`, cached
  per file mtime/size) and only NEW keys -- identity without line numbers,
  so a shifted line is not a new error -- can act. `ask`/`chat`/`eval` record
  the same baseline once per task; `verify.check_static` compares the
  dry-applied `after` buffer against the `before` buffer for the same file.
- **Where it fires.** Inside `_post_edit_tests`, before the impacted-test
  subprocess: the exact error lines go straight to `_repair_from_findings`,
  the fix is applied, and only then does the (possibly saved) test run
  start. Inside `run_loop`'s `_gated_check`, beside the syntax and contract
  gates: a step that introduces an undefined name fails its check with the
  exact lines, without ever spawning `--verify`. And through `verify.check`,
  so `preflight`, `best_of` scoring, and the incremental per-hunk gate all
  reject a hunk that breaks name resolution -- before it is ever written.

Tokens spent: zero, unless the gate finds something, when it buys the
cheapest possible repair prompt ("these names do not exist here") instead of
a subprocess transcript the model must interpret. Ablation flag:
`static_gate = false`.

### 20. Green ratchet step checkpoints for the autonomous loop

In `auto` mode every task builds on the previous one -- which means a late
step that fails after exhausting its repair rounds used to leave the
workspace holding its wreckage: modified files in a state that never passed
any check, new files with no caller, and all later steps now writing code
against a broken baseline. Repair rounds spent on that state were spent
against the wrong code. `auto_green_ratchet` (default on) makes Auto mode
monotonic:

- **A snapshot per step.** Before each step's worker starts,
  `ratchet.begin` copies every source-ish file (extension allowlist, dot
  directories and vendored/state directories skipped, size and count caps via
  `auto_checkpoint_max_file_bytes` / `auto_checkpoint_max_files`) into
  `.pedacito/checkpoints/step_<k>/` with a `manifest.json` recording size
  and skip-state. The plan file itself is on the never-snapshot list: the
  loop ticks boxes mid-step and no rollback may un-tick them.
- **The frontier.** When a step verifies green, `ratchet.green` advances a
  pointer (`green_step.txt`) and its snapshot is kept as the restore point.
  When a step fails its check -- after all `auto_repair_rounds` and any
  worse-rollback guard -- `ratchet.rollback` undoes exactly that step:
  files whose mtime/size prove the step modified them are restored from the
  snapshot, current files absent from the snapshot are the step's creations
  and are deleted (now-empty directories it created are pruned), the snapshot
  is consumed. The workspace is provably back at the last verified-green
  state; the next step inherits real work, not a poisoned tree.
- **What the loop says.** A rollback is telemetry, not silence: the step
  gets a `green_ratchet` event, an Activity Log line and a chat-facing
  `### ROLLED BACK: <task>` context note (restored/deleted file lists) so a
  later step knows the code it builds on was NOT changed by the failed one;
  the run summary and the `notification` hook report the rollback count; the
  step's TaskResult carries `rolled_back`/`rollback_info`. A step that never
  touched a file (the worker died before editing) rolls back to nothing and
  is reported as such. `auto_keep_checkpoints` prunes old green snapshots
  when the run finishes; the two features compound: the static gate catches
  a broken step faster, and the ratchet guarantees the next step never sees
  its corpse.

Pure file I/O -- zero model tokens, no subprocess. Ablation flag:
`auto_green_ratchet = false`.

---

### 21. Trust boundaries: deterministic guards and a human permission broker

A small local model should not be trusted with the whole machine, and a web page
it reads may contain instructions meant for it. Four deterministic guards run
before any file, URL or command is touched, and one human gate covers the
decisions the guards cannot make:

- **Workspace containment** (`security.resolve_safe_path`). Every path a tool
  touches is resolved with its symlinks. A `..` component, or a link that leaves
  the project, is refused before any read or write.
- **Credential guard** (`security.is_sensitive_path`). Secret-bearing files
  (`.env` and its variants, keys, SSH and TLS material, package-registry tokens)
  are never indexed, and a read of one needs approval through the broker.
- **Localhost CSRF guard** (`security.post_origin_allowed`). A state-changing
  request to the GUI server must come from a loopback page when it carries an
  `Origin` or `Referer`, so another site open in the same browser cannot drive it.
- **Outbound guard** (`security.ssrf_error`). The web tools accept only http(s)
  URLs and refuse loopback, link-local and cloud-metadata addresses, and private
  networks, unless the host is named in `fetch_allowed_hosts`.
- **Permission broker** (`security.request_permission`). Web access, package
  installs, unapproved shell commands and sensitive reads park the worker thread
  until the GUI answers: **Allow Once**, **Allow for This Run**, **Always Allow**
  (which flips the matching setting, such as `enable_web`), or **Deny & Redirect**.
  A denial returns a message that keeps the run going. No answer (a timeout or
  Stop) counts as a denial. Plain CLI runs have no broker, so a gated tool reports
  that it is disabled in the config instead of waiting for a person who is not
  there.

The guards cost no model tokens. The broker costs only the human's time, and it is
reached only for an action the deterministic checks could not settle.

---

### 22. `run_test`: a micro-dose of live evidence inside the gather phase

The evidence rules (§3, §5) say a claim of "tested, all green" without a
real test run is a lie the loop refuses. But during *investigation* the
model has no execution tool at all: it guesses what a fix will do, writes
it, and only then -- seconds and tokens later -- discovers it guessed
wrong. `run_test` micro-doses execution into gather: one scoped pytest
run, with the harness owning every parameter.

```json
{"action": "run_test", "target": "tests/test_app.py::test_save",
 "filter": "save or retry"}
```

- **No shell, structurally.** The observation comes from an argv list
  (`python -m pytest -q --tb=short --maxfail=3 ...`) through the same
  sandbox wrapper as `run`, so pipes, semicolons and backticks are inert
  characters in an argument -- nothing interprets them. The file portion of
  `target` goes through `security.resolve_safe_path` (no traversal, no
  symlink escape, must exist), which makes a `"tests; rm -rf /"` string a
  nonexistent path, not a command; the node-id portion must match a
  conservative charset; a `filter` starting with `-` is refused (it would
  read as an option). There is no path from these fields to a shell.
- **Cheap and bounded.** The timeout is `min(test_timeout, 15)` seconds with
  the same idle kill; collection errors and the tail count as bounded as
  well. The output is *not* the raw transcript: the harness renders a tally
  line, then up to four condensed failure blocks (project frames only, the
  traceback condenser from the repair path), and caps the whole observation
  at `max_read_tokens`. A green run is reported as green with an explicit
  "do not re-run to watch it pass again" line.
- **Offered when it can help.** `run_test` joins the step schema when the
  repository visibly has tests (`tests/`, `test_*.py`, `*.test.js`) or the
  task text names a failure (test/fail/bug/fix/error/traceback/reproduce/
  broken); dynamic pruning (§15) never hides it while the repo has tests --
  a stale usage counter must not remove the one tool that turns a guess
  into evidence. It is a step-phase tool: the final answer phase has no
  tools at all, and the `done` action still ends gather with full honesty
  rules unchanged.
- **Still permissioned.** With `enable_run_test_in_gather = false` every
  invocation parks at the permission broker under the *Test Execution*
  category (Allow Once / Always Allow / Deny), the same as unapproved
  shell; CLI runs without a broker report it as disabled instead of
  pretending otherwise.
- **It feeds the ledger too.** Each run is recorded through the
  micro-facts observer (§14), so a suite command that works twice becomes a
  confirmed repository fact; a contradiction retires it.

The point is not to let the model run anything: it is that "read the live
traceback before writing the edit" is now possible for zero configuration,
and still cannot become a shell. Ablation flag: set
`enable_run_test_in_gather = false` to gate every run through the human.

---

### 23. The DOM contract gate: what a browser would check at page load

Import and signature contracts have a deterministic twin on the browser
side, and nothing in the Python test stack can see it: a `<script
src="app.js">` renamed on one side only, a `getElementById('save-btn')`
against a template that renders `id="save-button"`, an `app.js` whose last
edit left one brace open. Each of these is a *runtime* error in a browser
and a *silent* one in the loop -- the task verifies green and ships a page
that does not load. `verify_frontend_contracts` (in `spec.py`, beside the
static gate) closes that gap with three mechanical checks:

- **Asset link integrity.** Every `<script src>` and stylesheet `<link href>`
  in every HTML file resolves against the file's own directory and the
  project root; a missing target is an error. External URLs, `data:` URIs
  and template placeholders (`{{ ... }}`, `<%= ... %>`, `${...}`) are
  skipped -- static verification cannot see the real path of a routed
  asset, and guessing is not verification.
- **DOM id cross-reference.** Every *literal* query
  (`getElementById('x')`, `querySelector('#x')`) must match an `id="x"` in
  some HTML file -- or a literal id the JS itself creates (`.id = 'x'`,
  `setAttribute('id', 'x')`, ids inside template strings). When a file
  builds ids with interpolation (`id = \`row-${i}\``), its queries are
  exempt: a moving target cannot be verified, and claiming otherwise would
  block real code with fake errors. An id that exists in zero templates
  without dynamic generation is the classic typo, and is reported against
  the files that define the id space it missed.
- **JS parses.** `node --check` is the ground truth when node is installed;
  otherwise the tree-sitter grammar the indexer already uses reports the
  first ERROR node. With neither available the check stays silent -- no
  regex pretends to be a JS parser.

Errors share the shape the repair path already reads:

```
FrontendContractError in app.js:42: queries DOM element '#add-task-btn',
but index.html only defines ids ['board', 'new-task-btn', 'todo-col'] --
render it, or fix the selector
```

- **Baseline-diffed like the static gate (§19).** The whole-tree scan runs
  once per step/task (`_capture_frontend_baseline`); only errors that are
  NEW for the current task block it. Pre-existing rot elsewhere in the
  project cannot sink an unrelated change.
- **Scoped like the contract gate.** With a touched-file list, only errors
  in those files report -- plus one deliberate exception: when an HTML file
  changed, every JS file's queries are re-verified against the moved id
  registry, because that is exactly how an edit to a template silently
  breaks a script.
- **Where it fires.** In `_post_edit_tests` after the static gate and
  before the test subprocess, and in `run_loop`'s `_gated_check` beside the
  syntax and cross-file contract gates: findings go to `_repair_from_findings`,
  the fix is applied, and the subprocess (which could never have caught
  these anyway) is skipped until the page would load.

A repo without HTML/JS exits before opening a file, so the gate is on by
default and free there. Ablation flag: `frontend_contracts = false`.

---

### 24. The repair lesson ledger: distilled anti-patterns in micro_facts

A round of `auto` that went broken -> fixed is the cheapest lesson a
repository will ever produce -- the harness already knows what failed, what
the repair was, and that it worked -- and today that knowledge evaporates
with the session, so the next session pays the same tuition. The anti-pattern
ledger keeps it: at most eight one-line lessons in
`.pedacito/micro_facts.json` under `"anti_patterns"`, injected into the
stable prefix right after the micro-facts (§14):

```
[REPO GOTCHAS & LESSONS] (repair lessons distilled from earlier runs;
hints, not truth):
- dom_id_mismatch: '#add-task-btn' was queried by JS but no rendered
  element carried that id -- add the id to the template or query what exists
```

- **When it writes.** Only on a proven transition: a gate (syntax, static,
  frontend contract) or a test run that went red *during this task* and is
  green again after the task's own repair rounds, at round >= 1. A lesson
  requires the evidence that the fix worked -- an unverified lesson is a
  rumour.
- **What it writes.** `distill_repair_lesson` recognizes the failure
  *shapes* the gates and pytest actually print -- a DOM query with no
  element, a link to a missing asset, an undefined name (F821/NameError),
  an unsatisfiable signature, an import that cannot resolve, a module
  attribute that does not exist, the first failing test id -- and renders
  one line from a fixed template. No LLM prose: an invented lesson would
  teach the next session a fiction, and a distilled one can be checked by
  the same deterministic tools that produced it.
- **Bounded and self-cleaning.** Eight rows max, deduplicated by hash of
  kind + text; repeats bump `confirmations` and refresh `last_seen` instead
  of growing the file. Lessons carry the files they name, and a lesson whose
  files have all been deleted is retired -- dropped from the document, not
  just hidden -- because a warning about a component that no longer exists
  would teach the model to dodge a ghost. The render budget is the same
  `max_micro_fact_tokens` knob; overflow drops the oldest lines first.
- **Hint, never truth.** Like micro-facts and memory, the block is labelled
  and sits in the prompt only: the loop re-checks everything live, so a
  wrong lesson costs a glance, not a wrong decision.

Zero model calls, zero subprocesses; one `sha1` and one small JSON rewrite
per proven fix. Ablation flag: `micro_facts = false` silences recording and
injection together with §14.

### 25. Zero-anchor AST surgical splicing: replace a definition without echoing it

A search/replace block asks a 7B model to reproduce, byte for byte, the *whole*
body it wants to change: the docstring, the blank lines, the trailing comment.
Small models are unreliable copyists, so a 40-line function edit usually lands
with one character wrong, and the harness spends a repair round telling it so.
The worse cost is the one nobody measures: the model re-reads the definition and
then re-emits every line of it, so the tokens paid for a 3-line change are the
tokens of a 40-line function.

Pedacito now offers a splice. The model names the definition and writes only the
new body:

```
--- FILE: src/orders.py
<<<<<<< SYMBOL src/orders.py :: Order.total
    def total(self, tax=0.0):
        return sum(l.price for l in self.lines) * (1 + tax)
>>>>>>> END_SYMBOL
```

No SEARCH half at all. `chunker.locate_symbol_span()` answers where the
definition lives: `ast` for Python, and the same tree-sitter grammars the index
already uses (§12) for JavaScript, TypeScript, Go and Rust, so `Class.method`
qualifiers mean exactly what the index says they mean. The agent resolves the
anchor through the index when it has one and falls back to the file's own AST
when it does not, so the block works mid-run after earlier edits moved lines
around. Concretely:

- **Decorators belong to the definition.** The span starts at the first
  `@decorator` line above a `def`, and if the replacement body does not re-echo
  it, the harness carries the original decorators over. Forgetting
  `@staticmethod` is a silent behaviour change, and this makes it a non-event.
  An empty body deletes the definition, decorators included.
- **Indentation is the harness's job.** A method body written at column 0 (the
  habitual shape) is re-aligned to the definition's real column before it is
  spliced, so the model never has to think about where the block sits.
- **Insertion is the same trick.** `<<<<<<< AFTER_SYMBOL file :: name` (or
  `{"edit_action": "insert_after_symbol"}`) splices a new sibling *after* that
  definition, which is how a model adds a method to a class without re-emitting
  the class.
- **The grammar is additive, never breaking.** Classic
  `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` blocks parse and apply exactly as
  before, and one reply may mix both shapes; blocks apply in reply order. The
  structured payload form `{"edit_action": "replace_symbol", "file": ...,
  "symbol": ..., "replacement": ...}` exists for constrained sampling, and an
  unknown or half-written payload is skipped rather than guessed at.
- **Same gate, same honesty.** The spliced result goes through the identical
  syntax check as every other edit, so a body that does not parse refuses the
  write instead of landing. An unknown or ambiguous symbol name is refused with
  a message that names the anchor and tells the model what to do instead:
  `outline` the file, or re-emit the block with `LINES <start>-<end>`. An
  unterminated splice block is reported as malformed, so an ignored edit is
  never silent.

Cost: the new body only. Nothing is re-read, nothing is re-echoed, and a wrong
guess costs one observation line rather than a fuzzy-match repair round. The
technique needs no flag: it is a shape the model may use, and the old shape
still works.

### 26. Deterministic multi-file symbol renaming: a refactor that costs zero tokens

Asking a model to rename a symbol is asking it to be a compiler. It must find
every reference, write every file's block from a re-read, and somehow avoid
touching the log message and the SQL column that merely *contain* the same
word. In a 12-file project that is twelve reads, twelve blocks, twelve chances
to be wrong, and it is the single most expensive thing a small model can be
asked to do that no model needs to do at all.

`refactor.rename_symbol_across_project()` does it without a model call:

- **Identifiers, not text.** Python is renamed through `tokenize`: every NAME
  token equal to the old name is rewritten, every string and comment is not.
  JavaScript, TypeScript, Go and Rust go through their tree-sitter identifier
  nodes, for the same guarantee. A language with no grammar available falls
  back to a word matcher that also skips quotes and comments, and says so in
  `warnings` rather than quietly trusting it.
- **Scoped by the import graph.** The default is the whole indexed source tree,
  skipping venvs, `.pedacito/` state, generated egg-info, hidden directories,
  unreadable or binary files and anything over 4 MB. Passing one file renames
  that file plus the files that import it, using the index's `importers_of`
  (§12) when the index exists and a textual importer scan when it does not.
- **All-or-nothing, like the edit path.** Every candidate result is validated by
  the same AST/tree-sitter gate (§19) *before* a single byte is written; one
  file that the rename would break refuses the whole run. A `before_write(rel,
  path)` hook lets a caller permission-gate or snapshot each file just before
  the write pass, which is how the tool layer's task allowlist and the CLI's
  backups plug in.
- **Name-based on purpose.** This renames an *identifier*, so a parameter in an
  unrelated function that happens to share the name moves too; that is what a
  project-wide rename means, and the diff preview shows it before anything is
  written. It is deliberately distinct from the older index-aware `rename`
  action, which resolves a single definition, refuses ambiguity, and rewrites
  only that symbol's declaration and its known callers.

Three surfaces, one engine: the agent action `rename_symbol` (offered only when
`allow_write` is on, and registered as a mutating action so the task loop's
mutation ledger, protection rules and snapshots apply, §21), the CLI
`pedacito rename OLD NEW [--file F] [--apply]`, and `Project ▾ → Rename symbol…`
in the GUI, backed by `POST /api/refactor/rename`. All three default to a
preview with unified diffs; writing snapshots the originals under
`.pedacito/backups/`, so `pedacito restore` (or the GUI's Restore dialog) undoes
a rename that turned out to be the wrong call. Applied files are re-indexed in
place so the next search does not answer from a stale symbol table.

Cost: one tokenize or parse pass per file, no model, no repair rounds, and a
guarantee a model cannot give -- a string literal is never a reference.

### 27. Engine-enforced JSON grammar: the schema moves from the prompt into the sampler

Everything before §1 only *asked* the model to answer in the step grammar:
the schema travelled as text in the system prompt, and compliance was the
model's good behaviour. Syntax retries, salvage passes and the truncation
refusals exist because that ask sometimes fails. OpenAI-compatible engines
with a grammar sampler (`llama.cpp`, vLLM, Ollama, hosted APIs) can instead
be *told*: `response_format={"type": "json_schema", "strict": true, ...}`
constrains sampling itself, so every emitted token is legal under the schema
and an unparseable or out-of-set step becomes physically impossible.

`tools.build_gather_json_schema(allowed_actions, step_reasoning_first)` is
the engine-readable form of the gather grammar: flat, every property
required, `additionalProperties` false, the `action` enum listing *exactly*
the dynamically pruned tool set for this task (§15 pruning feeds it, so the
grammar and the prompt never disagree), and field order load-bearing -- the
sampler emits keys in schema order, so the reason-first ordering (§2)
moves `reasoning` to the front for the servers that support it. `schema_for`
now delegates to it and stays byte-identical, because the server's
prompt-prefix cache reads that text too.

`llm.py` requests the grammar for every schema'd call (gather steps and plan
calls alike) and walks a three-rung ladder per `(base_url, model)` pair:
strict `json_schema`, then `json_object` (any JSON, prompt still carries the
shape), then plain prompting with the few-shot worked examples (§2) as
the only discipline. A 400 that names `response_format` / `json_schema` /
`json_object` flips the refused tier off in an in-process capability cache
and re-sends *one rung lower immediately* -- the first call never fails over
support, later calls skip the doomed tier outright. A 400 at the plain tier,
or a 400 that is unrelated to structured output, raises normally, so a
server that errors for some other reason cannot loop the downgrade. This
cache learns from live traffic and is distinct from the probed, on-disk
`server_modes.json` verdict; `cfg.enforce_json_schema = false` opts the
ladder out down to prompt-only mode for servers that misbehave under
constrained sampling.

Cost: a strict schema adds a few hundred grammar bytes per request that
cached prefixes absorb, and one extra round-trip the first time a server
refuses a tier -- after which it never asks again. The payoff is the whole
class of "model emitted prose instead of JSON" failures removed at the
engine level, on exactly the calls (five to thirty per task) where it used
to cost re-asks.

### 28. Scout/architect routing: a fast model does the lookups, the big model does the thinking

Investigation is a different job than judgement. During gather the model
issues short, highly constrained steps -- grep this, read that range, outline
this file -- whose only hard requirement is *tool-set literacy*; a 3B model
answers those correctly all day, while the same tokens in the answer phase
produce a mediocre patch. The classic single-profile loop pays the big
model's latency thirty times for work a small model does in a tenth of the
time, or starves the edit of the big model because the budget went to
lookups.

`cfg.scout_profile` names a second profile whose client runs *only* the
gather loop. `roles.scout_client()` derives the scout's config the same way
the role clients are derived (§10), warms it, and soft-falls back to the
architect when the profile is unknown, unreachable, or merely a copy of the
active one -- a scout outage degrades speed, never correctness. `Agent.gather`
resolves it once per task: step calls, truncation flags, few-shot priming,
context fitting and compaction digests ride the scout, while `dispatch()`
(whose `task` subagent and edit tools plan real work), the answer phase and
the repair loop stay on the architect. One usage ledger: the scout reports
into the same event log, so the topbar's call/cost counters see both models.
The verdict and the built client are cached on the config, so a long session
asks about the scout server exactly once.

On one GPU the win is KV-cache economics: the architect's 20k-token gather
conversation is *not* resident while a tiny scout model answers the steps
beside it; on two GPUs it is the obvious split entirely -- the scout keeps
the chatty prefix-cache-hot loop on the cheap card and the architect loads
once, for the answer. For the answer phase alone the loop still shows the
scout's gathered material in one context, so routing cannot introduce a
"second model misunderstood the task" failure mode: the architect sees the
identical transcript.

Cost: a second server (or a second model slot), a warm-up ping before the
first step, and one more config surface -- the GUI shows an active scout as
a `Scout: <profile>` badge next to the model chip so it is never invisible
which answering brain a step came from.

### 29. Local inference payload acceleration: `cache_prompt` and prompt lookup

Local servers get the same prefix problem §8 solves for hosted providers, but
for a key the server itself holds: llama.cpp's OpenAI server re-prefills the
whole conversation on every
step unless the request says otherwise, and its n-gram lookup decoder only
drafts when the request names a span. Gather is the worst case for both: the
prompt grows one step at a time, every re-send repeats the system prompt, the
interface contract, the map, and every earlier step verbatim, and the output
(SEARCH blocks, file paths, signatures) is largely *copy* of text already in
the prompt.

Two request keys fix that, and `llm.py` attaches them to local endpoints only.
`enable_prompt_cache` (on by default) adds `cache_prompt: true`, which keeps
the server's KV prefix alive between calls so the stable opening of the gather
conversation is prefilled once, not thirty times. `enable_prompt_lookup` (off
by default) adds `prompt_lookup_num_tokens: N` (default 10), which turns on
n-gram prompt-lookup decoding: the drafter finds the current suffix in the
prompt and drafts forward from its match, at essentially zero model cost, and
the verifier accepts what matches. Because lookup drafts *from the prompt*, it
wins hardest exactly where small models spend their time here: quoting source
lines into SEARCH blocks, repeating signatures, echoing paths.

The guard is the interesting part, because a key a server does not know is a
hard 400, not a shrug. Three layers keep the failure surface at zero. First,
the keys are only ever attached when `_is_local_server` is true: a loopback or
`.local` host, an address inside RFC1918, CGNAT, or IPv6 unique-local ranges
(the IANA documentation carve-outs like 203.0.113/24 read as "private" to the
stdlib and are deliberately excluded), or a provider entry that names a local
runtime (`llamacpp`, `lemond`, `vllm`, `lmstudio`, `ollama`). A hosted API
never sees them. Second, a 400 or 422 that lands while either key is attached
is treated as a refusal of the keys, not of the request: the payload is
stripped of both, the same request is re-sent once immediately, and the answer
arrives normally. Third, the refusal is remembered per `(base_url, model)`
pair, in the same engine-side cache that §27 uses for JSON tiers, so every
later call starts stripped and the total cost of a server that predates the
keys is one extra request per session. A refusal on a payload that had nothing
left to strip re-raises; there is no retry loop. And a key spelled by the user
in the profile's `[sampler]` table is the user's, not the engine's: presence
of the key suppresses the accelerator entirely.

Cost: two payload fields on local calls, a small capability cache, and one
wasted request per (server, model) pair that refuses, once per session.

### 30. Pruned, permission-gated MCP bridges (named servers + `mcp_call`)

The legacy `mcp` action renders the full catalog of command-list servers
into the prompt: every tool, every parameter, every time. That is fine for one
server with five tools and ruinous for the common real shape: a docs server
with forty, a tracker with sixty, a 4B model whose gather window cannot afford
a hundred irrelevant signatures, and no story at all for "may this server
*run* things".

Named servers are declared as tables (`[mcp_servers.docs]` with `command`,
`args`, `url`, `env`, `enabled`, `trusted`) and reach the model through one
new action, `mcp_call {server, tool, arguments}`, under two disciplines.

**Pruning.** Once per session the bridge lists each enabled server's tools (a
5-second budget per server; an unreachable server contributes nothing and is
not retried until the cache expires), and every gather step scores the task
text against tool names and descriptions, injects at most four matches as
one-line `server.name(args) -- summary` signatures, and *only then* offers
`mcp_call` in the step schema. If nothing matches, the action is absent, the
fields (`server`, `tool`, `arguments`) are absent, and the schema is
byte-identical to a run with no MCP configured at all. The injected list IS
the boundary: the model is told it may call only what it sees, so a task about
documentation never gets an angle on the tracker's `delete_ticket`, even
though it is one string away. The same relevance list decides the rebuild-phase
injection, so the edit model sees exactly what the gather model saw.

**Permission.** `trusted = true` on a server means the user vouched for it and
its tools run without asking. Everything else routes through the human broker
of §21 with category **MCP Tool**, a summary line (`Run MCP tool docs.search`),
and a detail blob with the exact arguments, so the dialog shows what will
actually run. A decline returns fixed text ("MCP tool execution was declined.
Proceed using local repository tools only.") instead of an exception, because
the productive outcome is the model falling back to `grep` and `read`, not
another step burning tokens on a hostile server. Returned text is capped at
about `max_read_tokens` tokens with an explicit truncation note, and malformed
`arguments` JSON is refused locally, before any server is asked.

The trust layers of §21 apply to the config itself: `mcp_servers` is an
execution-bearing field, so an untrusted project's `.pedacito.toml` cannot
define servers at all; named bridges come from the user config or a project
the user explicitly trusted. Stdio servers are spawned from argv without a
shell and inherit a scrubbed environment. Subagent children get the bridge
disabled. `pedacito mcp list` and `GET /api/mcp/tools` show the same pruned
view the model sees, so "why can't it use that tool" has an inspectable
answer.

Cost: a per-session tool listing (one `tools/list` per server), a 5-second
listing budget per server, and one permission dialog per call to an
untrusted server. The legacy command-list path is untouched and both formats
coexist.

## Quick reference: techniques 22 to 30

| Technique | Section | How to turn it off | Where you see it |
|---|---|---|---|
| Live test introspection (`run_test`) | §22 | `enable_run_test_in_gather = false` (each run then asks first) | A gather action. Permission dialog category **Test Execution** |
| Frontend contract checks (`verify_frontend_contracts`) | §23 | `frontend_contracts = false` | Automatic after applied edits. `pedacito webcheck` for an on-demand check |
| Repair lesson ledger (anti-patterns) | §24 | `micro_facts = false` (also silences §14) | `.pedacito/micro_facts.json`, under `anti_patterns` |
| Zero-anchor AST symbol splicing | §25 | No flag. Steer with `edit_style`; plain SEARCH/REPLACE still works | `<<<<<<< SYMBOL`, `AFTER_SYMBOL` edit blocks, and the `replace_symbol` payload |
| Deterministic rename (`rename_symbol`) | §26 | None for the CLI and GUI. The agent action needs `allow_write = true` | `pedacito rename`, and **Project ▾ → Rename symbol…** |
| Engine-enforced JSON schema | §27 | `enforce_json_schema = false` | Automatic on every structured call. `pedacito config --doctor` shows the tier |
| Scout/architect routing | §28 | `scout_profile = ""` | `scout_profile` in Settings. The top bar shows a `Scout: <profile>` badge |
| Local payload acceleration | §29 | `enable_prompt_cache = false`, `enable_prompt_lookup = false` | Automatic on local endpoints only. `cache_prompt` / `prompt_lookup_num_tokens` appear in the request payload (`log_events` shows calls) |
| Pruned MCP bridge (`mcp_call`) | §30 | `enable_mcp = false` | A gather action, offered only when a tool matches. `pedacito mcp list`. Permission dialog category **MCP Tool** |

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
