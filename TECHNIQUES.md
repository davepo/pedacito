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
  moving to the next symbol. A symbol that cannot be resolved in a file that
  already exists does not lose the intent: the microloop falls back to the
  whole file as the anchor and records `incremental_edit_fallback`, while an
  unresolved symbol in an absent file still fails hard. The full impacted-test
  and lint pass still runs once after the last hunk. Section 5 covers why the
  single-hunk boundary is the reliability win.
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
  whose state never got there, with the state named in the message. In the
  incremental edit loop, unaddressed *edit* clauses also seed one bounded
  follow-up pass (§35), and each clause carries a stable id (`R1`, `R2`, ...)
  that the review page shows as a per-file badge (§39).
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
  failures feed the repair rounds. A failure the command ledger already
  recorded before this run is labelled pre-existing and is not fed to repair
  as edit-caused (§33). An applied edit that breaks tests is an
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
  Two things that loop used to get wrong are now refused or said out loud:
  - *A range anchor that is really the whole file.* The loop asked for one
    definition; `--- FILE: src/entities.js LINES 1-171` of a 171-line file is
    a whole-file rewrite wearing a range anchor, and the answer cap cuts the
    copied file mid-definition -- which is how one run lost `MortarShell` and
    its factory. A symbol-targeted hunk whose range covers three quarters of
    a file of 40+ lines from the top is rejected before it is written, with
    the form that fits printed for it (`<<<<<<< SYMBOL file :: name`, full new
    definition as the body, no SEARCH section), plus the standing rule: never
    copy a whole file into a reply.
  - *A batch that half-landed.* Nothing rolls back -- each written hunk was
    verified on its own, and reverting them would undo proven work -- but the
    run no longer describes a mixed tree as applied. When some targets land
    and others do not, the batch emits an `incremental_partial` event
    (applied/failed counts and the failing files) and a verbose
    `[INCREMENTAL BATCH PARTIALLY APPLIED]` warning that names what is missing
    and how to start over. A summary that reads "applied" is what let the next
    step build on a feature that was never finished.
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
- **Import auto-healing (`auto_heal_imports`).** NEW F821 findings from the
  static gate (§19) are first healed by the harness itself, before the model
  ever sees them: a stdlib top-level name gets the canonical
  `import <name>`, and a project name exported by *exactly one* local module
  gets `from <module> import <name>` inserted after the header block. Every
  candidate insertion is validated with `ast.parse` plus the same AST scope
  resolver the gate uses, and kept only when it resolves the undefined name
  and introduces no new error key. Anything ambiguous is never guessed: star
  imports, `exec`/`eval`/`locals` in the file, builtins, already-bound names,
  and names two modules export all go to the model untouched. The gate then
  re-runs on the healed files, so a repair round only pays for what
  computation could not fix; each insertion leaves a one-line receipt in the
  event log. Ablation flag: `auto_heal_imports = false`.
- **Repair-trajectory surgical collapse (`collapse_repair_turns`).** The
  conversation is snapshotted immediately before every hunk or repair ask.
  When the repair succeeds, the failed drafts and their diagnostic turns are
  surgically popped off the message list -- the list is truncated back to the
  snapshot and a compact receipt (incremental mode) or the accepted reply
  takes their place -- so every byte the server had already cached stays
  cached and the next step is a prefix-cache hit instead of a re-prefill of a
  broken trajectory. An exhausted repair never collapses: its final failure
  stays visible to telemetry and shallow-failure memory. Each collapse
  records *why* it happened (the gate or check that rejected the draft, in
  one line) on the `repair_collapse` event, and the answer path keeps the
  last reason on hand: a run that collapsed ten times in a row spent ~137 s
  of retry churn and told its caller nothing but a count, which is a
  diagnosis of none. Ablation flag: `collapse_repair_turns = false`.

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

The retry that re-anchors a failed block offers one form the model cannot
get wrong: when the quoted SEARCH text does not match -- a line-number prefix
copied in, a smart quote or mojibake the file never had, tabs, CRLF -- the
answer may list the changed lines instead of quoting them. The retry prompt
shows the numbered source window and says to use the range anchor
(`--- FILE: path LINES 40-46`), whose needle is the file's own text for those
lines: the quoted SEARCH section is ignored, so nothing left of the mismatch
can miss twice. Copy-the-text-exactly was the only advice
offered before, so a byte-level mismatch produced three identical doomed
retries.

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
and recommend which profile to use for each role. The GUI's scan runs under a
wall-clock budget and returns the partial results gathered so far when it is
spent (§34). This matters for small models
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
- **Subagent EvidenceSpan transfer.** When the gather loop delegates
  research to the `task` helper, the child's *conversation* never crosses
  the boundary: its ephemeral turns are dropped when it returns, and what
  crosses back is its digest plus the 1-2 highest-utility spans it actually
  read and pinned -- selected deterministically (requirement matches first,
  then pinned child evidence, then span size and recency), deduplicated by
  file/range/content hash so the parent tail never repeats the same source
  twice, copied into the parent's evidence store *and* read ledger, and
  appended to the parent's conversation tail under the same read-token
  budget as a normal `read`. The parent may then edit lines the child
  displayed: `check_unread` credits transferred spans, so re-`read`ing the
  same bytes to satisfy the harness would cost tokens for nothing.

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
  otherwise spends on the same question. The chunkers also
  record the *receiver-qualified* target next to the bare name (`self.stop()`
  inside `Server` adds `Server.stop` as well as `stop`; same for `this.x()` in
  JS/TS, Go receiver pairs and typed `var` bindings, Rust impls and
  `Foo::new`, constructor- and annotation-bound locals in every grammar), so
  `callers`/`usages` on a qualified query rank the chunks that recorded the
  exact `Class.method` entry first, drop sites whose line proves they call a
  DIFFERENT capitalized class (`K2.run()` cannot be `K1.run`), and label the
  genuinely ambiguous ones `(unqualified match)` instead of silently mixing
  them together.
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

The same anti-loop principle covers the output stream. A reply that emits
tool-call syntax verbatim, or cycles the same text, is stopped mid-generation:
the harness watches the stream tail for a repeating block and breaks at the
third identical copy, marks the answer `stream_repetition` (with
`stream_truncated`) and says so on stderr. Stopping early costs the model
nothing and leaves a usable truncated answer instead of a lost one.

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
- **Fixture signature locking.** `conftest.py` fixtures are part of the same
  contract surface: `extract_pytest_fixtures` walks every discovered
  conftest with `ast.parse` (zero subprocesses, zero model calls), captures
  each `@pytest.fixture` function's exact signature, literal scope and a
  one-line docstring/yield hint, and appends a bounded
  `# pytest fixtures (conftest.py)` subsection to the contract block --
  only when the repository has conftest files or the task mentions tests, so
  a non-Python project pays nothing. A test-writing step can no longer
  invent a fixture name or call one with parameters it does not accept:
  the signature the test must use is quoted to it before it writes.
- **The gate.** After every applied task, beside the existing syntax gate,
  `verify_cross_file_contracts` walks every local import and imported-name
  call in the project: a `from parser import parse_row` that references a
  symbol the module does not define, or a call whose argument count cannot
  satisfy the real signature (defaults, `*args`/`**kwargs` respected), fails
  the task *immediately* -- before any verify subprocess -- and the exact
  diagnostic (`ContractError in cli.py:18: imports 'parse_row' from
  parser.py, but parser.py only defines ['parse_line', 'TableParser']`) is
  fed straight into the repair prompt instead of letting the error propagate
  through three more steps. JS/TS ES modules run the same contract through
  the same gate (`_js_cross_file_errors`): a relative `import ... from
  './x.js'` must point at a file that exists and name exports the target
  really has, and imported calls are arity-checked against the exporter's
  scanned signature -- bare packages and dynamic imports are deliberately
  out of scope. (§23's frontend gate reports the same findings scoped to
  the touched files.)
- **Incoming test callers are part of the contract.** A changed
  signature in `app/logic.py` breaks `tests/test_logic.py` even when the test
  was not touched -- and the from-import pass above cannot see the two
  spellings tests actually use: `import app.logic` + `app.logic.add(1)`, and
  instance methods on a variable bound by constructor (`s = Server()`) or
  annotation (`s: Server`). `_incoming_test_caller_errors` scans every test
  file for exactly those receivers, arity-checks each call against the
  source's real signature and emits the same
  `ContractError in tests/test_logic.py:4: calls 'logic.add' with 1 arg(s),
  but updated signature in app/logic.py requires 3 positional argument(s)`
  line -- so the loop's contract gate vetoes the pytest spawn (both in
  `_gated_check`, before the verify subprocess, and in `_post_edit_tests`,
  before the test run) and the exact line goes to repair instead of minutes
  of interpreter startup. Shadowed names (a local `class Srv` hiding the
  import), decorated defs, inherited methods, star-args calls and non-test
  files stay untouched: the same false-positive discipline as the lock.
- **Deliberately conservative.** Unknown absolute imports are treated as
  third-party, star-imports and shadowed names are trusted, classes without
  an explicit `__init__` are never arity-checked: a false positive would
  waste a repair round on a working project, which the technique exists to
  prevent. The same drift listing also extends `new`'s `fix_api_mismatches`
  repair and its final consistency warning.
- **JS linking is part of the contract too.** The `import`/`export` surface
  of a project's ES modules is verified beside the Python import graph, here
  and in §23's frontend gate -- which adds the two shapes a parse cannot see:
  a name bound twice at module level, and a call to a name no module binds.
  A Python project and a browser project are held to the same rule: a name a
  file uses must exist where the file says it comes from.

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
- **Heal before asking (`auto_heal_imports`, §5).** In every one of those
  seats, NEW F821 findings are offered to deterministic import auto-healing
  first (`heal_static_gate_errors`): names with exactly one provable import
  (stdlib top-level modules, or a project symbol exported by one module)
  are fixed on disk, the gate re-runs, and only findings that survive reach
  the model. A forgotten `import json` costs a file write and zero tokens;
  the repair prompt sees exactly what it would have seen had the healing
  feature not existed.

Tokens spent: zero, unless the gate finds something, when it buys the
cheapest possible repair prompt ("these names do not exist here") instead of
a subprocess transcript the model must interpret. Ablation flag:
`static_gate = false`.

**The zero-subprocess gates are not gated on the shell.** Deciding whether to
*check* an edit and deciding whether a *test command may run* are two
different questions, and `_post_edit_tests` used to ask only the second one:
`verify_tests` or `allow_shell` off skipped the syntax, static, frontend and
contract gates along with pytest, so the default profile (`allow_shell =
false`) vetted nothing at all, and an `ask` build could report "applied and
verified" for a module graph that could not link. The chain is split in two
now: `gates_wanted` (in-process, always asked) and `tests_wanted` (the
subprocess, still honouring both flags), and every gate that ends a run red
records its verdict as an unresolved finding, so the caller reports it
instead of a clean summary.

### 20. Green ratchet step checkpoints for the autonomous loop

In `auto` mode every task builds on the previous one -- which means a late
step that fails after exhausting its repair rounds used to leave the
workspace holding its wreckage: modified files in a state that never passed
any check, new files with no caller, and all later steps now writing code
against a broken baseline. Repair rounds spent on that state were spent
against the wrong code. `auto_green_ratchet` (default on) makes Auto mode
monotonic:

- **A snapshot per step.** Before each step's worker starts,
  `ratchet.begin` checkpoints every source-ish file (extension allowlist, dot
  directories and vendored/state directories skipped, size and count caps via
  `auto_checkpoint_max_file_bytes` / `auto_checkpoint_max_files`) into
  `.pedacito/checkpoints/step_<k>/` with a `manifest.json` recording size
  and skip-state. The checkpoint is a **hard link** where the
  filesystem allows it (`os.link`, with a `shutil.copy2` fallback for
  EXDEV/EPERM/FAT) -- free, byte-identical by construction, no copy of a
  multi-megabyte tree per step. That buys the price of one invariant: every
  write that can touch a snapshotted path must REPLACE the file, never
  rewrite it in place, because `open(path, "wb")` truncates through the
  shared inode and would silently corrupt history. `fileio.write` is exactly
  that atomic replacement (mkstemp beside the target, mode preserved,
  `os.replace`), every loop write path goes through it, and `rollback`
  restores the same way (snapshot bytes into a temp file, `os.replace`), so
  the fast path -- skip the restore when size+mtime already match, which is
  the common case for a true hard link -- stays safe.
  The plan file itself is on the never-snapshot list: the
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
- **Outbound guard** (`security.ssrf_error` + `resolve_checked`). The web tools
  accept only http(s) URLs and refuse loopback, link-local and cloud-metadata
  addresses, and private networks, unless the host is named in
  `fetch_allowed_hosts`. The check runs on every IP the name resolves to, and
  the connection is pinned to the validated address, so a public name cannot
  rebind to a private one mid-request (§32).
- **Permission broker** (`security.request_permission`). Web access, package
  installs, unapproved shell commands and sensitive reads park the worker thread
  until the GUI answers: **Allow Once**, **Allow for This Run**, **Always Allow**
  (which flips the matching setting, such as `enable_web`), or **Deny & Redirect**.
  A denial returns a message that keeps the run going. No answer (a timeout or
  Stop) counts as a denial. Two hosts install this broker: the GUI (its dialog
  answers the parked worker) and `pedacito rpc` (§31), which turns the same four
  answers into JSONL frames -- a `permission_request` event carries the
  category, the action and a `request_id`, a `permission_reply` request decides
  it, and **Always Allow** flips the matching setting for the live session only,
  never persisted to the user's config. Plain one-shot CLI runs have no broker,
  so a gated tool reports that it is disabled in the config instead of waiting
  for a person who is not there.

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
- **An exit code is translated, not just reported.** `EXIT: 5` is not a
  failure of the code under test; it means *nothing was collected*, and a
  model that reads only the number fixes the number. So the verdict line
  spells it out (`EXIT: 5 (NOT COLLECTED)`), the reply carries an `EXIT CODE
  MEANS:` line for the codes that are easy to misread (2 = usage or a crash
  in collection, 4 = no tests matched, 5 = nothing collected), and a `target`
  naming a Python file with no `def test_...` / `class Test...` is refused
  *before* pytest starts: it is probably a standalone verification script, so
  the refusal names the working alternative (`run` it, or read it) and says
  plainly not to rename working functions to chase the code. One campaign
  called `run_test` on its own passing smoke script four times, read `no
  tests ran`, and concluded the bug was that `main()` should have been called
  `test_smoke()`.
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
edit left one brace open, and a module graph that parses but cannot link.
Each of these is a *runtime* error in a browser and a *silent* one in the
loop -- the task verifies green and ships a page that does not load.
`verify_frontend_contracts` (in `spec.py`, beside the static gate) closes
that gap with five mechanical checks:

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
- **ES module contracts.** A relative `import ... from './util.js'` must
  resolve to a file that exists and name exports that file really has --
  the exporter's surface is scanned with the indexer's tree-sitter grammar
  (`export function/const/class`, `export {a, b}`, `export default`, and
  `export ... from` re-exports) -- and a call to an imported function is
  arity-checked against the exporter's real signature, the same
  `ContractError` diagnostics and same conservative skips as §18's Python
  gate (`_js_cross_file_errors` is shared by both). Bare package imports and
  dynamic `import()` are deliberately absent: bundler territory, not
  static-check territory.
- **A module-level name bound twice.** A parse accepts `import { WAYPOINTS }
  from './a.js'` twice in one file -- tree-sitter builds a clean tree, and so
  does every earlier gate. The browser does not: `SyntaxError: Identifier
  'WAYPOINTS' has already been declared` kills that file and every module
  importing it, which is the shape an *incremental* loop produces, because
  each hunk adds its own import line without reading the ones already there.
  The gate enumerates the module's top-level bindings with the grammar and
  reports any name bound twice. The one repeat an ES module allows is two
  `var`s of one name; repeated `function`s and `var`/`function` mixes are the
  same SyntaxError there (a classic script does allow them, last one wins,
  and the gate keeps that distinction).
- **A call to a name the module cannot reach.** ES modules see only what they
  declare or import, so a free identifier is a `ReferenceError` on the first
  line that runs it -- invisible to a parser, and the exact residue of a task
  that edits the call site and forgets the import. Three buckets, because the
  fix differs: another module *exports* the name (say which, and print the
  import line), another module declares it *without exporting* it (say that
  every module has its own scope), and nothing defines it at all. Names a
  browser or a CDN `<script>` provides (`document`, `THREE`, `io`, `admin`)
  are matched against a named vocabulary plus the page's own script tags, and
  a classic script's top level is treated as the global scope it really is.

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

- **None of it is hostage to the shell.** Every check above reads files in
  this process, so the gate vetoes with `allow_shell = false` and
  `verify_tests = false` -- the default profile, and the one a browser build
  with a broken module graph slipped through while still reporting
  "applied and verified".
- **The name vocabulary is shared.** The host/CDN/test-global list and the
  "is this a member call or a definition" suppressions belong to
  `webcheck.py`, and `spec.py` imports them, so the vetting gate and the
  `pedacito webcheck` notices cannot disagree about what a name means; the
  gate is grammar-based and vetoes, the regex checks are heuristic and only
  advise. Two lists had already drifted apart once, and the gate's had grown
  HTML *tag* names and misspelled interfaces (`Menu`, `Table`, `Form`,
  `WriteableStream`) into what is supposed to be a list of things the browser
  provides. That is not caution, it is blindness: a call to a `Menu()` no
  file defines is the exact bug these checks exist to report, and under the
  old list it was unspeakable. The merge kept every real Web API either list
  knew (`addEventListener`, `alert`, `MessagePort`, `IndexedDB`, the audio
  nodes) and dropped the rest -- one curated list, honest about what a host
  actually provides.
- **The regex reader has three rules about what it is reading.** They are the
  difference between a notice and a nuisance, and each was learned from a file
  that parses and runs fine:
  - *Read uses out of blanked code, imports out of raw text.*
    `_js_code()` blanks comments, strings and template literals;
    `_js_no_comments()` blanks only comments, because an `import` statement
    carries its specifier inside quotes and blanking the quotes erases the
    import. Applying the aggressive blanker to the import sweep is how a
    checker reports `./game-core.js` as a call to `core()`.
  - *Blanking is a walk, not a regex.* Lit-html nests a template inside an
    interpolation (`html`...${ok ? html`x` : null}...`), so a string pattern
    that stops at the first inner backtick hands the closing markup back as
    code -- prose like "when a model server is reachable (falls back to the"
    becomes a call to `reachable()`. The scanner keeps a stack of quotes,
    templates and `${ }` regions that are code again, skips regex literals so
    `/['"]/` cannot open a string that never ends, recovers from an
    unterminated quote at the line break instead of blanking the file, and
    preserves every byte's position so a finding's line number is still the
    line it came from.
  - *A binding is a binding.* `const [open, setOpen] = useState(false)` binds
    both names and `const {x, y: z}` binds `x` and `z`; reading the pattern as
    one token reports `setOpen()` as unreachable. Names are pooled from the
    grammar (gate) or split out of the destructuring head (advisor) -- never
    from the raw text of the declaration line.

A repo without HTML/JS exits before opening a file, so the gate is on by
default and free there. Ablation flag: `frontend_contracts = false`.

**It also gates a scaffold.** `pedacito new --apply` writes each file in its
own call, so `index.html` invents one set of ids and `main.js` queries
another, and the per-file pass calls both complete. `run_new` now runs the
vetting half over the finished tree, feeds the violations back for repair
rounds (`fix_web_contracts`), and if any survive, prints `BUILD INCOMPLETE`
with the list and exits 1. A project that cannot load is not a finished
project.

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
  would teach the model to dodge a ghost. Ineffectiveness retires a lesson
  too: harness-fact `runs` / `recurred` / `avoided` counters, and a lesson
  that never avoids its failure kind across five observed runs moves to
  `retired_lessons` (§36). The render budget is the same
  `max_micro_fact_tokens` knob; overflow drops the oldest lines first.
- **Path-scoped rule promotion.** A lesson that has proven itself twice
  *and* whose named files all live under one subdirectory (`src/api/...`,
  `frontend/...`) graduates out of the global ledger: the harness writes it
  into a deterministic auto-managed rule file
  `.pedacito/rules/auto_<scope>.md` --

  ```markdown
  ---
  paths: src/api/*
  auto_generated: true
  ---
  ```

  holding at most five deduplicated one-line lessons for that directory --
  and the JSON row moves to the document's `promoted` list with a
  `promoted_to` pointer. A repo-root file in the set, or files with no
  shared parent, disqualify the lesson: a glob of `*` is not a scope, and
  such lessons stay global. Graduated lessons no longer spend the eight-slot
  budget and no longer ride the stable system prefix on unrelated tasks;
  they reach the model only appended to the tail of a `read` or `context`
  observation whose file matches the rule's glob -- the same mechanism as
  hand-authored scoped rules, so the system prompt is never mutated
  mid-gather and the byte-pinned prefix cache stays intact. Cleaning is
  symmetric with retirement: when every file a promoted lesson names has
  been deleted, its line is dropped from the rule file; when a rule file
  holds no lessons left, the file itself is deleted. Only files bearing
  `auto_generated: true` are ever written or pruned -- a user-authored
  `auto_*.md` with a colliding name wins, permanently, and the lesson stays
  in the global ledger. Re-confirming a promoted lesson refreshes the row
  in place; it never resurrects a second copy.
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
  backups plug in. The guarantee also holds *during* the write pass: a failed
  snapshot or an OSError (full disk, permission flip) part-way through restores
  every file already written in that batch from the pre-rename text still held
  in memory. The same undo ledger guards the multi-file edit path
  (`edits.apply_edits`): block three of five failing to land never leaves a
  half-applied tree, not even a file created and then abandoned.
- **Name-based, or receiver-qualified.** The engine above renames an
  *identifier*, so a parameter in an unrelated function that happens to share
  the name moves too; that is what a project-wide rename means, and the diff
  preview shows it before anything is written. The index-aware `rename` action
  is the careful sibling: a bare name resolves only when exactly one definition
  exists anywhere (methods included -- they are matched through their class
  chunk's member list), and it can also take a qualified query, `Store.save`.
  There the rename follows *types* through the AST: the definition, `self.save`
  inside that class, static `Store.save`/`pkg.Store.save` access, and `x.save`
  where `x` is deterministically bound to `Store` by constructor assignment or
  annotation -- the same `_local_types` evidence the call graph uses
  (§12). A receiver typed to another class, an untyped one, or a bare name
  outside the class is listed in the reply as deliberately *not renamed*, and
  files that construct the class are pulled into scope even without importing
  its module. Receiver tracking is Python-only: elsewhere the qualified query
  warns and falls back to the whole-word pass.

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

The same hardware-safety rule governs the `task` subagent on a single-slot
local server -- a server that keeps exactly one model's KV prefix resident.
Forking a helper with its own fresh system prompt would evict the architect's
cached conversation and make the parent's very next step pay a full
re-prefill, so the child never sends one: it forks the parent's stable
prefix **byte-for-byte** (`prefix_messages`), and the helper brief rides at
the tail as a normal assistant/user pair -- the identical append-only shape
every other mid-gather payload uses (§11). The child reads and digests
inside its own tail-appended turns; when it returns, only its digest and
its best pinned EvidenceSpans cross back (§11), and the parent's cached
prefix has not been disturbed at any point. Ablating the prefix fork would
turn each delegated lookup into a cache miss for the whole rest of the
gather.

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

---

### 31. Bidirectional stdio JSONL-RPC: a warm engine an editor can drive

`ask --jsonl` already proves the engine can narrate itself as events, but it is
one-shot: every request pays for process start, index load, profile negotiation,
and a cold session with no history. An editor plugin or a peer agent wants the
opposite -- one long-lived process, machine-readable in and out, interruptible
mid-task. `pedacito rpc [PROJECT]` (equivalently `pedacito chat --rpc`) is that
transport, in `pedacito/rpc.py`.

- **Two pure streams.** stdin takes one JSON request per line
  (`{"id": ..., "method": ..., "params": {...}}`); stdout emits *only* JSONL
  frames. `RpcOut` owns the real stdout, and the CLI immediately rebinds
  `sys.stdout` to stderr, so any forgotten human-facing print lands on a
  diagnostic channel instead of corrupting the protocol. A tee event line that
  does not round-trip as JSON is dropped to stderr, never passed through raw.
- **The tee is the transport.** `RpcOut` doubles as `EventLog.tee`: the engine
  already writes its events as JSON lines, so the RPC layer re-parses them,
  tags each with the active request's `req_id`, counts the engine's terminal
  `result` frame, and adds its own compact summary only when the engine emitted
  none. That summary answers the only two questions a machine client has left:
  did the edits land (`applied`, `changed`) and is the result known-good? The
  second one is not `applied`, so the frame carries `blockers` as its own
  field -- the error-severity gate findings the run ended on (frontend,
  contract, static, syntax), capped and deduplicated. `applied: true` with a
  non-empty `blockers` reads exactly as it means: the code is on disk and the
  harness could not verify it clean. `applied` was never stretched to cover a
  gate veto, so a caller built on the honest field still gets the honest field.
  Every method closes with an `rpc_response` carrying both `id` and the
  `ok`/`result`/`error` triple -- the stream never ends mid-task.
- **Warm across requests.** The index, the model profiles and the session
  ledger (and with them the prompt cache of §8) persist; `switch_profile`
  rebuilds the client in place, re-attaches the event log, and answers with
  the new model's label -- refused while a task runs, and rolled back on a bad
  reference. The workspace run-lock is taken lazily at the first request that
  would write (`ask` with `apply`), so read-only sessions never block a plain
  `pedacito ask`.
- **A busy session is still reachable.** Task methods run on a worker thread
  while the main thread keeps reading stdin, so `steer`, `permission_reply` and
  `cancel` land mid-task. Steering pushes into the same queue the loop drains at
  a step boundary -- appended state, no history rewrite, cache-safe. `cancel`
  stops the task and keeps the process (and everything warm in it) alive. One
  task runs at a time; a second `ask` is refused with a `busy:` error, not a
  deadlock.
- **The permission broker, framed** (§21). The same hook the GUI dialog rides
  is installed for stdio: a gated tool publishes a `permission_request` frame
  (`request_id`, category, action, risk, `turn_id`) and parks in 0.5 s beats
  until `permission_reply` answers `allow_once`, `allow_run`, `allow_always` or
  `deny`. A session-scoped cache keyed by `(category, action-key)` makes
  repeated identical tools silent; **Always Allow** flips the matching setting
  (web → `allow_fetch`, shell → `allow_shell`, ...) on the live config only --
  never persisted -- so a trusted run can go quiet without editing files. No
  answer by `--permission-timeout` (default 120 s), a cancel, or stdin EOF
  denies, exactly like the GUI's Stop.
- **EOF is a flush, not an abort.** If the client hangs up mid-task, `serve()`
  keeps the process alive until the worker's reply and its `rpc_response` have
  been written (`EOF_JOIN_GRACE`), denying any permission the task parks on for
  want of a responder and cancelling only a task that outstays the grace.

Cost: no model calls, no new prompt shapes -- the price is transport discipline
(one JSON object per line, stdout purity, injected closures for everything the
CLI knows). That discipline is what makes the server unit-testable: the tests
drive `RpcServer` with fakes over `io.StringIO` and never spawn a process.

---

## Audit hardening (R1–R8)

A follow-up audit checked the promises above against the code and found six
places where the behaviour fell short of the doc, plus two guardrails worth
adding. Each item below closes one gap. They extend the existing techniques
rather than replace them, and none is a rejected method in disguise: every one
is a deterministic, harness-side check or a bounded pass, in the spirit of the
through-line. The two test-only guardrails (§37, §38) cost nothing at run time.

### 32. Resolution-pinned, rebinding-proof fetch (extends §21)

The outbound guard used to judge the hostname *as written*, so a public name
that resolves to a private address (DNS rebinding) slipped past it. Now
`security.resolve_checked(host, allowed_hosts)` resolves the name once with
`getaddrinfo`, runs the private-address check on **every** returned IP, and
refuses if any is private -- a wildcard allow-list entry never exempts a
private range. The connection is then **pinned** to the validated IP: the
socket connects by address while `Host`, SNI and the certificate check keep the
original name, so a second DNS answer cannot swap in a private address
mid-request. Redirects are followed by hand and re-resolved at every hop
(§21's per-hop check), and the opener disables automatic redirects so no
unvalidated hop can execute. Cost: one DNS lookup per hop, no model tokens.
Tests monkeypatch `socket.getaddrinfo` to map a public name to `127.0.0.1` and
assert refusal with no request issued.

### 33. Baseline-aware test verdicts (extends §5, §22)

`run_impacted` used to report *any* failure as edit-caused, sending the repair
loop chasing errors a repo already had. The command ledger now records a
`last_ok` flag per observed command, and `memory.command_was_failing` answers
whether the same narrowed command was already failing before this run. When it
was, the verdict is labelled "pre-existing failure (recorded before this run)"
and is not fed to repair as edit-caused -- unless this run captured its own
baseline, which supersedes the history. Zero model calls; it reuses facts the
ledger already keeps.

### 34. Wall-clock budget for discovery (extends §10)

`scan_models` probes each candidate with a per-request timeout but had no
overall deadline, so a config full of remote profile URLs could block the GUI
well past a reasonable wait. A monotonic deadline is now threaded through
discovery: `scan_models(budget=20.0)` reserves half the budget for finding
servers and the rest for describing them, so one slow server cannot starve the
model rows, and it returns the partial results gathered so far when the budget
is spent. Deterministic to test with a local server that sleeps past the budget.

### 35. Coverage-seeded second-pass intents (extends §1, §3)

`check_coverage` warned about unaddressed task clauses but nothing acted on the
warning, so a half-answered multi-clause task could end with a note instead of a
second pass. The unaddressed-clause core is now shared (`verify._unaddressed`),
and `coverage_seeds` returns the unaddressed **edit** clauses (report clauses
are left to the answer prose). After a batch, the incremental planner seeds up
to three follow-up intents from those clauses' anchors, read spans and terms --
a located definition, else a symbol search, else the whole file -- and runs
**one** bounded follow-up pass under the same per-target retry ceiling, emitting
a `coverage_followup` event and a receipt. This is a deterministic trigger from
the requirement state machine, not self-critique or a search tree: the model is
never asked to grade its own draft.

### 36. Lesson effectiveness counters (extends §24)

Lessons were injected and retired by file existence alone, so an ineffective
lesson persisted forever. Each live lesson now carries harness-fact counters --
`runs`, `recurred`, `avoided` -- updated by `memory.note_lesson_run` once per
observed check run. A lesson that has been seen for `_LESSON_RETIRE_RUNS` (5)
runs without ever avoiding its failure kind is retired into `retired_lessons`;
auto-promoted path-scoped rules are exempt, and lessons born mid-run are not
judged against runs that preceded them. The loop emits a `lesson_retired` event.
These are records of what happened, explicitly **not** the rejected
"model-written reflection notes": the model never writes prose about itself.

### 37. CLI / how-index drift guard (test-only)

`test_how.py` asserted coverage against a hardcoded list of CLI names, so a new
subcommand could ship without a how-card and the test stayed green. The parser
is now built by `cli.build_parser()`, and the test derives the expected command
set from the actual argparse subparsers (mapping `models` to its `models scan`
card) and checks every action choice is documented. Adding a subcommand without
a card now fails the test. Static and deterministic; no run-time cost.

### 38. Config-editor round-trip property test (test-only)

The textual config patcher is guarded by a seeded generative test, not just one
regression. It generates comment/string/table shapes -- including the FIX-011
class of comments quoting triple delimiters, literal strings and multi-line
values -- and asserts `set_values` / `remove_profile` / `add_profile` always
yield text `validate_text` accepts, never duplicate a table, and never drop an
unrelated key or comment. Writing it surfaced two real scanner bugs (an escaped
quote inside a multi-line basic string ended the value early in `_value_end`,
and a naive closing-delimiter search in `_triple_toggles` hid later table
headers); both are fixed and covered. Deterministic seeds, no model.

### 39. Review hunk provenance (extends §3)

A reviewer could accept some files and reject others without seeing which task
clause each file served. Every requirement clause now carries a stable id
(`R1`, `R2`, ...), the tracker maps files to the clauses whose patches touched
them (`clause_map`), and each `FileDiff` carries those ids. The review page and
the GUI show a clause badge per file (titled with the clause text), and a
partial accept flags any clause none of whose files was accepted
(`unaccepted`), recording it in the sign-off detail and the chat. Provenance is
computed from the same deterministic edit evidence the tracker already gathers.

## Upgrade paths (validation report 2)

The second validation pass (`PEDACITO_VALIDATION_REPORT_2.md`) named six
outstanding upgrade paths. Each is implemented as a deterministic, harness-side
mechanism -- no extra model calls -- and each is covered by
`tests/test_upgrade_paths.py` (U1–U5) and `tests/test_domcheck.py` (U6).

### 40. Escalation policy: know when a profile will not unstick itself (U1)

A small model can stall in ways that retrying on the *same* profile will not
fix: it re-picks an identical read-only step until the lease dies, spends lease
after lease with no progress, produces edits that apply but fail at runtime, or
ends run after run with zero edits. The harness now recognises each shape and
suggests the `escalate` rung instead of burning a third local attempt:

- **Repeat suppression.** An identical read-only step is skipped outright after
  the first repeat (its result cannot change); mutating steps stay dispatchable.
- **Two-lease strikes.** Two no-progress leases on one task (across its
  attempts) are an escalation signal; a success clears the task's strikes.
- **Semantic-failure signal.** Test output that still fails after the repair
  rounds with a runtime shape -- `TypeError`, `ReferenceError`, `RangeError`,
  `RecursionError`, a failed assertion, an infinite loop / max call stack -- and
  that this task made worse, is an escalation signal. `SyntaxError` is
  deliberately excluded: a parse failure is the syntax gate's repair skill, not
  a stronger-model case.
- **Per-profile zero-edit statistic.** `.pedacito/profile_stats.json` counts
  zero-edit runs per profile; `escalation_zero_edit_runs` (default 3) in a row
  queues the suggestion for the next task. A landed edit clears the counter.

One `escalation_suggested` event is emitted per task (the first reason wins)
even when no `escalate` role is configured, so a human or an outer orchestrator
still sees the signal. When the role IS configured and `auto_escalate` is on,
the ladder re-runs the task ONCE on the stronger model -- even if edits landed,
because a patch that applies but does not work is still a failed task. It keeps
the first attempt only when that one was green and the escalated one came back
red. Turn it off with `auto_escalate = false` (and
`escalation_zero_edit_runs = 0` for the statistic alone).

### 41. Context-pressure edit discipline (U2)

On a small window, a model asked to change a large file re-emits the WHOLE file;
the draft is cut off mid-file and drops working wiring (the GLM-4.7-Flash
Phase-5a shape, the static half of BUG-216). Two deterministic guards close it:

- **The brief.** When the effective window is at or below `small_context_tokens`
  (default 16384), the answer brief demands the forms that fit -- small
  SEARCH/REPLACE blocks, a SYMBOL anchor with the full new definition, or
  successive append chunks -- and names the task's files already over the
  ceiling (`prompts.small_window_note`).
- **The gate.** `Result.whole_file` marks an empty-SEARCH overwrite or a LINES
  range covering every line. `verify.check_oversized_rewrite` refuses a
  whole-file draft larger than `max_whole_file_lines` (default 120) BEFORE it can
  land -- error by default, warn only when the task explicitly asks to rewrite
  that very file. It runs in preflight and in the incremental dry-apply veto, so
  the oversized draft never reaches disk.

Turn it off with `small_context_tokens = 0`.

### 42. Numeric-SEARCH auto-reanchor (U3)

A tiny-profile slip is to quote the `read`'s line-number prefix (`4`) as the
SEARCH body instead of the line's text. The block then matches nothing and
burns a whole retry round. When a SEARCH body is digits only AND does not occur
literally in the file, the harness reads it as `LINES n-n` and applies the
replacement to that exact line, noting the reanchor in the receipt. A number
that DOES occur literally keeps normal text-match semantics, and an
out-of-range number is refused with a teaching message that shows the real line
count. Deterministic; saves the retry. Well-formed blocks are unaffected.

### 43. Naming the prose-vs-edit gap (U4, extends §35)

A model can describe a feature in prose ("I will add `compute_tax`...") and emit
no edit for it; a generic "unaddressed" finding does not tell it that
description is not implementation. `Requirement.prose_only` now marks an edit
clause whose terms appear in the answer's prose but in no edit, and
`check_coverage` renders it explicitly: "you described 'X' in prose but no edit
implements it -- description is not implementation, emit the edit for it."
`coverage_seeds` carries the same wording into the follow-up intent's
`intended_change`, so the second pass is aimed at the gap rather than a vague
"finish the task". A clause absent from both prose and edits keeps the generic
wording.

### 44. Shrinking micro-loop retry budgets (U5)

In the incremental edit loop a rejected hunk used to be re-asked at the full
answer cap, so a model re-emitting the same failing draft burned the whole
budget on every attempt. Now the first attempt gets the full answer budget and
each re-ask halves it, with a floor of 2048 so one real block still fits:
attempt `k` gets `max(2048, full // (2*k))`. It is the same discipline as the
repair rounds, applied per hunk, and it bounds the cost of a stuck micro-loop
without reducing the number of attempts (`repair_rounds` still sets that).

### 45. The headless DOM smoke check (U6)

`webcheck` (§23) proves the STATIC contracts a browser enforces at load; it
cannot prove the page actually MOUNTS. A replaced entry module can leave the
page silently unmounted -- the runtime half of BUG-216 -- and no regex notices.
`pedacito domcheck` is the separate, heavier runtime tool:

- It serves the project over a loopback-only HTTP server (ES modules refuse
  `file://`, so the page needs a real origin) and loads the entry page in
  headless Chromium via Playwright.
- It reports uncaught page errors and failed same-origin loads (a module that
  404s takes the whole graph down), and flags interactive controls with no
  listener, inline handler, or delegation -- conservatively: any binding on the
  control, an ancestor, or `document`/`window` suppresses the finding.
- It snapshots the mounted surface (element identities, controls, text volume)
  and can diff it against a pre-edit snapshot (`--snapshot` then `--baseline`),
  catching the regression shape where named elements / controls disappear or
  the visible text collapses.

In-run, `verify_dom` AND `allow_shell` gate applied web edits on it: a pre-edit
baseline is captured, and only NEW errors / lost mounted UI feed a micro-repair
round (pre-existing page errors are not blamed on the task). It is opt-in and
permission-gated because it executes project code -- the same boundary as the
test subprocess; an explicit `pedacito domcheck` invocation is itself the
consent. A missing Playwright install reports an honest install hint, never a
false green. `webcheck` stays regex-only and notice-level; this tool's errors
can drive repair. Off by default (`verify_dom = false`); `ask --verify-dom`
flips it per run, `dom_entry` / `dom_timeout` tune it.

## Upgrade paths (harness leaderboard)

A gatekeeper audit of five candidate upgrades (C1–C5) against the existing
mechanisms shipped four of them; the fifth (environment/config hardening) was
already covered by `PEDACITO_<KEY>` overrides, `api_key_env`, the default-off
`allow_shell`/`allow_fetch`, and `scrub_env`. Each shipped item is a
deterministic, harness-side mechanism -- no extra model calls on the green
path, no new config keys -- and each has a dedicated test file. The full
evaluation lives in `upgrades/HARNESS_LEADERBOARD_UPGRADES.md`.

### 46. The repair-phase test-file lock (C1)

A repair loop pointed at a red test can "fix" it by weakening the test --
deleting the assertion, widening the tolerance, skipping the case. That games
the verdict instead of earning it. Post-edit gate repairs (the findings-driven
repair seat and the incremental-lint seat) now arm a per-task lock: an edit to
a PRE-EXISTING test file is refused with a teaching message ("a repair may
not change a pre-existing test file -- fix the code under test"). Pre-existing
means present in the index snapshot taken at task start, so a test file the
task itself creates stays editable. Exemptions are explicit: a task that asks
for test changes (`prompts.wants_test_changes`), an edit-kind requirement that
matches it, and files the gate itself flagged -- except `tests`-kind findings,
because a red assertion is not proof the test is wrong. Preflight repair is
unlocked: it repairs the task's own draft before any verdict exists. The lock
costs nothing on a green run and never blocks a legitimate test-writing task.
Covered by `tests/test_repair_test_lock.py`.

### 47. Shell mutations feed the task ledger (C2)

A worker that finishes the job with `run` commands produces no answer-phase
edit blocks, and its disk changes used to bypass post-edit verification
entirely. `tool_run` already computes `index.stale()` to refresh the index
after a command; the same result now feeds the gather-mutation ledger
(`_note_mutation`), so hooks, impacted tests, and lint see shell-rewritten
files exactly like write/move/delete actions. Recorded whatever the exit code:
the ledger tracks disk state, not command success -- a failed command that
rewrote files still needs verification. Zero added cost (one reused `stale()`
result). Known limit: `stale()` sees indexed files only, so a brand-new file a
command creates is invisible here; model-driven creation uses the write tools,
which already record. Covered by `tests/test_run_mutation_ledger.py`.

### 48. Cargo test narrowing, bounded to provably-safe cases (C3)

`narrow_command` now accepts the task's `changed` files, and the post-edit
hook narrows `cargo test` to `cargo test --test NAME...` when -- and only when
-- every changed file is a top-level integration test (`tests/*.rs`). Each
integration test is its own crate, so no other target can be affected and the
changed targets ARE the whole change. Any `src/` change (its unit tests would
be skipped), manifest or helper-module change, nested target, unknown changed
set, or a command that already selects targets keeps the full suite. The
`pedacito tests` CLI reports the same narrowing. The audit rejected the
candidate's Go/Rust spec-gate half: toolchain subprocesses with false-positive
risk against spec.py's conservative rule. Covered by
`tests/test_cargo_narrow.py`.

### 49. One strategy shift when escalation has nowhere to go (C4)

When an escalation reason fires (§40) but NO distinct `escalate` role is
configured, the ladder used to give up. Now the task gets ONE more attempt on
the SAME client and budgets with the edit style flipped -- `anchor` <-> `quote`,
unset -> `anchor` -- because a model stuck on exact-quote SEARCH blocks often
succeeds with SYMBOL splices, and vice versa. The gate is strict: a reason must
have fired, the run must be an applying run with `auto_escalate` on, and the
first attempt must not be green (a green result, a landing with no red verdict,
and a coherent "nothing to change" all stand). The retry must land to win;
when both attempts are proven red the first is kept. `edit_style` is restored
afterwards, a `strategy_shift` event is logged, and the shift never recurses.
Zero impact on green runs. Covered by `tests/test_strategy_shift.py`.

## Quick reference: techniques 5, 11, 17 to 49

| Technique | Section | How to turn it off | Where you see it |
|---|---|---|---|
| Prompt/Spec architect | §17 | No flag. Skip per run with no `--refine` | `--refine` on `ask`, `auto`, `new`; **✦ Refine Prompt** in the GUI, `↶ Undo Refine` restores the text |
| Interface lock + pre-pytest contract/arity gate (`interface_lock`), incl. conftest fixture signatures | §18 | `interface_lock = false` | The `do not break signatures` block in the prompt; `ContractError` messages before any test runs |
| Static analysis gate (`static_gate`, F821/F822/F823/F402/F404 + E999) | §19 | `static_gate = false` | `static gate` findings in the repair prompt; baseline-diffed, so only new errors count |
| Green ratchet (`auto_green_ratchet`), hard-link checkpoints | §20 | `auto_green_ratchet = false` | `.pedacito/checkpoints/step_<k>/`; the Activity log says `ROLLED BACK` |
| Import auto-healing (`auto_heal_imports`) | §5, §19 | `auto_heal_imports = false` | `[AUTO-HEALED: added 'import json' in x.py]` lines before repair rounds; "import auto-heal" in the event log |
| Repair-trajectory collapse (`collapse_repair_turns`) | §5 | `collapse_repair_turns = false` | `repair_collapse` telemetry events; successful repairs leave a receipt, not the broken drafts |
| Subagent EvidenceSpan transfer + single-slot prefix fork | §11, §28 | No flag (part of the `task` subagent) | `subagent_evidence` telemetry; promoted source appended under `HELPER RESULT` |
| Live test introspection (`run_test`), incl. translated pytest exit codes | §22 | `enable_run_test_in_gather = false` (each run then asks first) | A gather action: `EXIT: 5 (NOT COLLECTED)`, `EXIT CODE MEANS:` lines. Permission dialog category **Test Execution** |
| Frontend contract checks (`verify_frontend_contracts`), incl. ESM export/arity, duplicate module bindings and unreachable calls; also gates `new --apply` | §23 | `frontend_contracts = false` | Automatic after applied edits; `BUILD INCOMPLETE` from `new`; `pedacito webcheck` for an on-demand check |
| Repair lesson ledger (anti-patterns) | §24 | `micro_facts = false` (also silences §14) | `.pedacito/micro_facts.json`, under `anti_patterns` |
| Auto-promoted path-scoped rules | §24 | `micro_facts = false` (also silences §14/§24) | `.pedacito/rules/auto_*.md`, `auto_generated: true`; `promoted` rows in `micro_facts.json` |
| Zero-anchor AST symbol splicing | §25 | No flag. Steer with `edit_style`; plain SEARCH/REPLACE still works | `<<<<<<< SYMBOL`, `AFTER_SYMBOL` edit blocks, and the `replace_symbol` payload |
| Receiver-qualified call graph (`Class.method`) | §13 | None (always on; the chunkers record qualified targets) | `callers` / `usages` on a qualified query: exact receiver matches first, `(unqualified match)` label for ambiguous ones |
| Deterministic rename (`rename_symbol`) | §26 | None for the CLI and GUI. The agent action needs `allow_write = true` | `pedacito rename`, and **Project ▾ → Rename symbol…** |
| Engine-enforced JSON schema | §27 | `enforce_json_schema = false` | Automatic on every structured call. `pedacito config --doctor` shows the tier |
| Scout/architect routing | §28 | `scout_profile = ""` | `scout_profile` in Settings. The top bar shows a `Scout: <profile>` badge |
| Local payload acceleration | §29 | `enable_prompt_cache = false`, `enable_prompt_lookup = false` | Automatic on local endpoints only. `cache_prompt` / `prompt_lookup_num_tokens` appear in the request payload (`log_events` shows calls) |
| Pruned MCP bridge (`mcp_call`) | §30 | `enable_mcp = false` | A gather action, offered only when a tool matches. `pedacito mcp list`. Permission dialog category **MCP Tool** |
| Warm stdio JSONL-RPC (`pedacito rpc`) | §31 | None. Run `pedacito rpc PROJ` or `pedacito chat --rpc`; `--permission-timeout` paces the broker | One JSON request per line on stdin, pure JSONL on stdout, events tagged `req_id`. Protocol documented in `pedacito/rpc.py` |
| Resolution-pinned fetch (`resolve_checked`), DNS-rebinding-proof | §32 | None (always on; it is the outbound guard itself) | `web_fetch` / MCP fetch refuse a public name that resolves to any private IP; the socket is pinned to the validated address and every redirect hop is re-resolved |
| Baseline-aware test verdicts (`command_was_failing`) | §33 | `micro_facts = false` (no ledger: every failure stays edit-caused) | Impacted-test verdicts labelled `pre-existing failure (recorded before this run)`; repair is not triggered by them |
| Wall-clock budget for model discovery | §34 | None in the GUI scan (fixed 20 s cap, half reserved for discovery) | **Models ▾ → Scan** returns the rows gathered so far when the budget is spent instead of blocking |
| Coverage-seeded follow-up pass | §35 | No flag. Fires only when the requirement tracker holds unaddressed edit clauses | `coverage_followup` event; a "Coverage follow-up" receipt appended to the incremental-edit reply (one bounded pass, ≤3 seeds) |
| Lesson effectiveness counters + retirement | §36 | `micro_facts = false` (also silences §14/§24) | `runs` / `recurred` / `avoided` and `retired_lessons` in `.pedacito/micro_facts.json`; `lesson_retired` event |
| CLI / how-index drift guard | §37 | Test-only; no run-time cost | `tests/test_how.py` derives the command set from `cli.build_parser()`; a subcommand without a how-card fails the suite |
| Config-editor round-trip property test | §38 | Test-only; no run-time cost | Seeded generative shapes in `tests/test_config_editor.py` (comments, strings, tables); guards `set_values` / `remove_profile` / `add_profile` |
| Review hunk provenance (clause badges) | §39 | No flag | `R1`, `R2`, … badges per file on the review page and in the GUI; a partial accept flags clauses whose files were all rejected |
| Escalation policy (repeat suppression, lease strikes, semantic signal, zero-edit statistic) | §40 | `auto_escalate = false`; `escalation_zero_edit_runs = 0` for the statistic | `escalation_suggested` events; `.pedacito/profile_stats.json`; one retry on the `escalate` role's model |
| Context-pressure edit discipline (whole-file ceiling) | §41 | `small_context_tokens = 0` | The `CONTEXT PRESSURE` answer brief; `oversized_rewrite` findings before an oversized whole-file draft can land |
| Numeric-SEARCH auto-reanchor | §42 | No flag (a recovery; well-formed blocks are unaffected) | A `numeric SEARCH auto-reanchored to LINES n-n` receipt note |
| Prose-vs-edit gap naming | §43 | No flag (wording refinement of the coverage check) | "you described 'X' in prose but no edit implements it" coverage findings and follow-up intents |
| Shrinking micro-loop retry budgets | §44 | No flag (a budget bound; `repair_rounds` still sets attempts) | Each rejected incremental hunk is re-asked at half the previous token cap (floor 2048) |
| Headless DOM smoke check (`domcheck`) | §45 | `verify_dom = false` (default); the CLI is explicit consent | `pedacito domcheck`; `dom gate` findings after applied web edits; `--snapshot`/`--baseline` mounted-surface diff |
| Repair-phase test-file lock | §46 | No flag (a repair invariant); tasks that ask for test changes are exempt | `refused: a repair may not change a pre-existing test file` receipts when a gate repair aims at an existing test |
| Shell mutations feed the task ledger | §47 | No flag (part of the gather-mutation ledger) | Post-edit hooks/impacted tests/lint run after a shell-only task; `[run] index refreshed` lines |
| Cargo test narrowing (integration-only changes) | §48 | No flag; fires only when provably safe (`test_command` wins) | `cargo test --test NAME` in the post-edit hook and `pedacito tests` output |
| Strategy shift when no `escalate` role exists | §49 | `auto_escalate = false` | `strategy_shift` event; one same-model retry with the edit style flipped after an escalation reason |

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
