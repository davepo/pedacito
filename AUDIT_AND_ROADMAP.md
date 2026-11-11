# Pedacito — Deep-Dive Audit & Roadmap

Scope: a full read of `README.md`, `TECHNIQUES.md`, `MODEL_GUIDE.md`, `QUICKSTART.md`,
then a line-level audit of the agent loop, retrieval/ranking, edit/verify/safety,
LLM/provider/config plumbing, and the GUI server. Every fix below is a surgical
root-cause change with a targeted regression test, and each was verified against
the **full** pytest suite (no new dependencies; clean idiomatic English only).

Baseline before any change: **9 failed / 1666 passed / 2 skipped**. The 9 failures
root-caused to 3 real bugs (FIX-001, FIX-002, FIX-003). All remaining fixes were
found by manual audit and reproduced before being fixed.

Test environment: editable install in a private venv; `python -m pytest -q`.
Suite grew from 1666 to **1698 passed / 2 skipped**: 23 new regression tests plus
the 9 baseline failures now passing (the 2 skips are pre-existing environment guards).

---

## 1. Bugs fixed

Each entry: location, root cause, the fix, and the regression test that pins it.
Every test was confirmed to **fail on the pre-fix code** and pass after.

### [FIX-001] Sandbox hides an interpreter installed under `/tmp`
- **Location:** `pedacito/sandbox.py` `_bwrap_args`.
- **Root cause:** the bubblewrap profile mounts `--tmpfs /tmp`, which shadows any
  Python interpreter whose `sys.prefix` lives under `/tmp` (a venv, a standalone
  build). The sandboxed child then cannot start its own interpreter, so every
  sandboxed test/run fails for an environment reason, not a code reason.
- **Fix:** after the tmpfs, re-bind `sys.prefix` and `sys.base_prefix` read-only
  when they resolve under `/tmp`, so the interpreter survives the tmpfs.
- **Test:** `tests/test_remaining.py::test_sandbox_rebinds_an_interpreter_living_under_tmp`.

### [FIX-002] JS duplicate-binding gate wrong about ES modules
- **Location:** `pedacito/spec.py` `_js_top_level_bindings`, `_js_module_link_errors`;
  `pedacito/webcheck.py` `_js_module_links`, `_report_duplicate`.
- **Root cause:** the gate treated `function` and `var` as freely repeatable in
  **all** JS. That is true for a classic script, but an ES module only allows two
  `var`s of one name; repeated `function`s and `var`/`function` mixes are a
  `SyntaxError` that kills the whole module graph. The gate both missed real
  module errors and (in its old test) asserted the wrong rule.
- **Fix:** split the binding `kind` into `function` vs `hoisted` (var); compute
  `repeatable = ("hoisted",)` for an ES module and `("hoisted", "function")` for a
  classic script, and apply it in both the spec and webcheck gates.
- **Tests:** `tests/test_buglog_validation.py::test_module_refuses_two_functions_with_one_name`,
  `::test_module_refuses_a_function_next_to_a_var`, `::test_module_allows_two_vars_with_one_name`,
  and the corrected classic-script case. `TECHNIQUES.md` §(JS gate) updated to match.

### [FIX-003] GUI model scan reads the CLI's config global, not the GUI's
- **Location:** `pedacito/gui/settings_api.py` `scan_models`.
- **Root cause:** profile URLs came from `cli._profile_base_urls()`, which reads the
  CLI's own config global. On a dev machine that global can point at the real user
  config (with remote servers), leaking extra probe candidates into a scan that is
  supposed to reflect only the GUI's config. Non-hermetic and a privacy surprise.
- **Fix:** derive `profile_urls` from the same `settings.validate_text(config_path("user"))`
  the import already reads — one source of truth.
- **Test:** `tests/test_gui_models.py::test_scan_reads_profiles_from_the_gui_config_not_the_cli_global`.

### [FIX-004] `lstrip("./")` mangles dot-files and case in path comparison
- **Location:** `pedacito/loop.py` `_same_rel`, `distill_repair_lesson` (new `_lesson_file`).
- **Root cause:** `str.lstrip("./")` strips *characters*, not a prefix: `.eslintrc.js`
  became `eslintrc.js`, and case was not normalised consistently. A repair lesson was
  recorded under a filename that does not exist and retired on the next load; two
  distinct dot-files could compare equal.
- **Fix:** `_same_rel` now uses `fileio.path_key`; a new `_lesson_file` strips leading
  `./` segments while preserving the leading dot and case.
- **Test:** `tests/test_sprint_o_introspection.py::test_repair_lesson_keeps_dot_file_names`.

### [FIX-005] Requirement ledger clipped mid-word
- **Location:** `pedacito/verify.py` requirement ledger (`p[:140]`).
- **Root cause:** the word-boundary clip helper `_clip_clause` existed, but the ledger
  still cut with a raw `p[:140]`, so a plan step inherited a requirement title ending
  mid-word.
- **Fix:** use `_clip_clause(p)`.
- **Test:** `tests/test_buglog_validation.py::test_requirement_ledger_uses_the_word_boundary_clip`.

### [FIX-006] SSRF guard blind to legacy numeric loopback spellings
- **Location:** `pedacito/security.py` `is_private_host`; `pedacito/webfetch.py` `_is_private`.
- **Root cause:** `ipaddress.ip_address` rejects `2130706433`, `127.1`, `0x7f000001`,
  `0177.0.0.1`, but the OS resolver still maps them to `127.0.0.1`. The guard waved
  them through, and `webfetch` carried a second, weaker copy of the private-host test.
- **Fix:** normalise via `socket.inet_aton`/`inet_ntoa` before judging; strip an IPv6
  zone id; make `webfetch._is_private` delegate to the one guard.
- **Test:** `tests/test_security_and_permissions.py::test_ssrf_guard_sees_legacy_numeric_loopback_spellings`.

### [FIX-007] Child env leaks `PIP_INDEX_URL`-style secrets
- **Location:** `pedacito/safety.py` `_ENV_SECRETISH`.
- **Root cause:** the secret-ish pattern did not match `index_url`, so
  `PIP_INDEX_URL` / `PIP_EXTRA_INDEX_URL` (which can embed a token) passed into child
  processes despite the scrub.
- **Fix:** add `index_url` to the pattern.
- **Test:** `tests/test_safety_and_workspace.py::test_child_env_scrubs_ambient_secrets` (extended).

### [FIX-008] Rename pass-1 paired alphabetically, not by name affinity
- **Location:** `pedacito/edits.py` rename pairing, pass 1.
- **Root cause:** with several same-directory candidates, pass 1 took `min(cand)`
  (alphabetical) instead of scoring like pass 2. Renaming `x.py` with candidates
  `a.py` and `b_x.py` moved the real bytes into `a.py`.
- **Fix:** pass 1 uses the scored `_pair(old, cand)` and guards `if new:`.
- **Test:** `tests/test_smoke.py::test_rename_pairs_by_name_affinity_not_alphabet`.

### [FIX-009] Partial-batch overflow counted blocks, not files
- **Location:** `pedacito/agent.py` `apply_edit_intents` partial-batch banner.
- **Root cause:** the banner lists at most six failed *files*, but the `(+N more)`
  overflow used `len(mixed)` (failed *blocks*). Eight ghost targets in eight files
  printed `(+8 more)` after six names.
- **Fix:** compute `failed_files` once; overflow is `len(failed_files) - 6`.
- **Test:** `tests/test_buglog_validation.py::test_partial_batch_overflow_counts_files_not_blocks`.

### [FIX-010] Coverage dry-runs advanced the live requirement tracker
- **Location:** `pedacito/verify.py` `check_coverage`.
- **Root cause:** when a live `RequirementTracker` was supplied, `check_coverage`
  called `observe_edit`/`observe_report` on it directly. Dry-applied drafts and
  scored candidates therefore promoted requirements to EDITED/REPORTED; a rejected
  draft marked its clause addressed and the next coverage check skipped it.
- **Fix:** deep-copy the supplied tracker before observing; only the agent's
  post-apply `observe_edit` advances the authoritative state.
- **Test:** `tests/test_requirement_state.py::test_coverage_dry_run_does_not_advance_the_live_tracker`.

### [FIX-011] Config scanner counted triple-quotes inside comments/strings
- **Location:** `pedacito/config_editor.py` `_triple_toggles`.
- **Root cause:** the scanner counted `"""`/`'''` occurrences per line. A `#` comment
  mentioning `"""` flipped it into "inside a multi-line string", swallowing every
  later table header: `profile_names` lost the profiles and `set_values` appended a
  duplicate table the loader then rejected.
- **Fix:** rewrite as a character scanner that respects `#` comments, single-line
  strings, and escaped quotes, and only toggles on a real triple delimiter.
- **Tests:** `tests/test_config_editor.py::test_triple_toggles_scans_characters_not_counts`,
  `::test_comment_quoting_triple_delimiter_does_not_swallow_tables`.

### [FIX-012] Discovery crashed on a server with zero models
- **Location:** `pedacito/discovery.py` `_probe_root`.
- **Root cause:** a fresh Ollama answers `/api/tags` with `{"models": []}`; the listing
  helpers return `None` for an empty list, and the final comprehension iterated
  `entries` unguarded → `TypeError`. (`describe_models` already used `entries or []`.)
- **Fix:** iterate `(entries or [])`.
- **Test:** `tests/test_discovery.py::test_ollama_with_zero_models_is_still_a_server`.

### [FIX-013] RPC turn-id: `r` prefix shadowed `req`
- **Location:** `pedacito/rpc.py` `_turn_id`.
- **Root cause:** prefixes were tried in the order `("r", "req", "turn")`, so
  `"req11"` matched `"r"` first, leaving `"eq11"`, and `int()` failed → `None`.
  The docstring promises both `r11` and `req11` resolve; a permission event could
  not name its turn for `req`-prefixed ids.
- **Fix:** try the longer prefixes first: `("req", "turn", "r")`.
- **Test:** `tests/test_sprint5_rpc.py::test_turn_id_resolves_every_documented_prefix`.

### [FIX-014] Capability probe ignored `api_key_env`
- **Location:** `pedacito/modelcard.py` `_t_capabilities`.
- **Root cause:** the raw `/chat/completions` capability probes built
  `Authorization: Bearer {cfg.api_key}` directly, bypassing `llm._api_key`. With
  only `api_key_env` set (the recommended secret-free config), they sent an empty
  `Bearer `, so a hosted model recorded `native_tools`/`images` as unsupported.
- **Fix:** use `_api_key(cfg)` and add the header only when a key exists.
- **Test:** `tests/test_profiles.py::test_capability_probe_authenticates_with_api_key_env`.

### [FIX-015] MCP tool call reused the listing's 5s timeout
- **Location:** `pedacito/mcp.py` `_named_session` (reuse branch), `close_all`.
- **Root cause:** the cached session is created for the 5s `tools/list` budget;
  the reuse branch returned it unchanged, so `call_mcp_tool`'s larger timeout was
  ignored. A tool slower than 5s timed out, the session was dropped, and the call
  retried — executing a side-effecting tool twice. Separately, `close_all` cleared
  `_EXTRA_CACHE` but not the legacy `_CATALOG_CACHE`, so a listing survived the reset.
- **Fix:** set `s.timeout = max(1, timeout)` on reuse; clear `_CATALOG_CACHE`/`_CATALOG_TIME`
  in `close_all`; removed the never-called `_named_session_key` helper (dead code).
- **Tests:** `tests/test_sprint_r_acceleration_and_mcp.py::test_call_mcp_tool_gives_the_cached_session_its_own_timeout`,
  `::test_close_all_clears_the_legacy_catalog_cache`.

### [FIX-016] Lesson ledger crashed when no project root was set
- **Location:** `pedacito/memory.py` `load_anti_patterns`.
- **Root cause:** the function normalised a falsy root to `None` but then called
  `_facts_doc(root)` unconditionally; `micro_facts_file(None)` reaches `Path(None)`
  and raises `TypeError`. Any session without a project root (the GUI before a
  folder is opened) crashed while rendering the stable prefix, which calls
  `render_anti_patterns` → `load_anti_patterns`.
- **Fix:** return `[]` when the root is `None` (no project, no lessons). Also
  corrected the `tool_unused_tasks` docstring: a never-used action returns all
  tasks so far, not 0 (see drift list).
- **Test:** `tests/test_sprint3_auto_rules.py::test_lesson_ledger_is_empty_without_a_project_root`.

### [FIX-017] Empty partial-accept destroyed the staged review
- **Location:** `pedacito/gui/core.py` `decide_review`.
- **Root cause:** for a partial accept, `rv.diffs[:]` was truncated to the selected
  files *before* the "at least one file" check. An apply with `files=[]` (all
  checkboxes cleared) therefore emptied the staged diff list and only then returned
  the error — the review could never be applied afterwards, because every retry saw
  an empty selection. The user's only escape was Reject, discarding good changes.
- **Fix:** compute the kept list first, refuse an all-empty selection before any
  mutation, and only then replace `rv.diffs` and fold in hand edits.
- **Test:** `tests/test_review_server.py::test_empty_partial_selection_keeps_the_review_applicable`
  (empty selection leaves diffs and outcome untouched; a follow-up selection applies).

### [FIX-018] Redirect SSRF: private hosts were contacted before the check
- **Location:** `pedacito/webfetch.py` `tool_fetch`; `pedacito/tools.py` `_web_open`,
  `fetch_url`, `web_search`.
- **Root cause:** both fetch paths let the HTTP client follow redirects automatically
  and only validated the *final* URL afterwards. A public allowlisted page that
  answered `302 → http://169.254.169.254/…` (or any LAN host) caused the request to
  the private address to execute — and in `tools.fetch_url` the response body was
  even read — before the guard refused it. The check stopped the *output*, not the
  *request*, which is the classic SSRF-via-redirect bypass.
- **Fix:** follow redirects by hand (≤6 hops) and validate every next URL with the
  same guard *before* requesting it. `webfetch` uses `allow_redirects=False` plus a
  hop loop applying the existing allowlist/private-host expression; `tools` uses an
  opener whose `HTTPRedirectHandler.redirect_request` returns `None` (3xx surfaces
  as `HTTPError`), joins `Location` with `urljoin`, and runs `security.ssrf_error`
  per hop. `fetch_url` also moved its final-URL check ahead of the body read;
  `web_search` passes the same allowlist. The test seam `_serve` now patches
  `OpenerDirector.open` instead of `urlopen`.
- **Tests:** `tests/test_security_and_permissions.py::test_fetch_url_validates_every_redirect_hop_before_requesting_it`,
  `tests/test_features_misc.py::test_fetch_checks_each_redirect_hop_before_requesting_it`
  (a 302 to the metadata IP is refused with the private URL never requested; an
  allowlisted chain, including a relative `Location`, is followed to the end).

---

## 2. Doc-to-code drift (and updates applied)

- **`TECHNIQUES.md` JS gate** — described `function`/`var` as repeatable without the
  script/module distinction. **Updated** to state the ES-module rule (FIX-002).
- **`pedacito/processes.py` `start` docstring** — "A name already used by a live
  process is suffixed." `_unique_name` suffixes against *any* existing record file,
  live or exited. **Docstring corrected** to "an existing record".
- **`pedacito/brief.py` `build_brief` docstring** — "trimmed by dropping the layout
  lines last-first." The code pops whole lines from the end (layout line first, then
  commands, entry points, stack notes). **Docstring corrected** to match.
- **`pedacito/memory.py` `tool_unused_tasks` docstring** — "(0 when never used)".
  Never-used returns `doc["tasks"]` (all tasks so far), not 0. **Docstring corrected**.
- **`pedacito/mcp.py` `_named_session_key`** — a helper documented to restart a cached
  session when its definition changes, but it is never called (dead code). The reuse
  branch keys on name only. **Removed** as dead code; the real staleness gap it was
  meant to guard is tracked in the roadmap.
- **`pedacito/mcp.py` `close_all`** — cleared `_EXTRA_CACHE` but not the legacy
  `_CATALOG_CACHE`/`_CATALOG_TIME`, so a catalog listing survived the reset tests rely
  on. **Fixed** to clear both (part of FIX-015).

---

## 3. Complex bugs — two remediation options

For the fixes with a real design choice, the option taken and the alternative:

- **FIX-006 (numeric SSRF).** *Taken:* normalise legacy spellings with
  `inet_aton`/`inet_ntoa` inside the one guard. *Alternative:* resolve every hostname
  with `getaddrinfo` and judge the resulting IPs. Rejected here because it adds a DNS
  lookup to every check and still misses rebinding; resolution-time pinning is the
  roadmap item instead.
- **FIX-011 (config scanner).** *Taken:* a character scanner local to `_triple_toggles`.
  *Alternative:* parse the file with `tomllib` and map values back to byte ranges.
  Rejected: the editor is deliberately a textual patcher that preserves comments, and
  a full parse/rewrite would lose that guarantee.
- **FIX-018 (redirect SSRF).** *Taken:* follow redirects by hand and validate every hop
  before requesting it. *Alternative:* keep auto-redirect and only re-check the final
  URL. Rejected: intermediate hops to private hosts still execute server-side (the
  classic bypass), so a final-only check is insufficient.

---

## 4. Shortcomings (grounded in the docs)

These are gaps relative to what the docs promise, not style nits:

- **SSRF stops at the literal host.** `TECHNIQUES.md`/`security.py` promise the web
  tools "cannot probe the user's network or the cloud metadata endpoint." The guard
  judges the hostname as written; a public name that *resolves* to a private IP
  (DNS rebinding) is still not covered. Redirect hops are now validated per hop
  (FIX-018); resolution-time pinning is roadmap item R1.
- **No wall-clock budget on discovery.** `scan_models` probes every candidate with a
  per-request timeout but no overall deadline; many profile URLs (the real user config
  has remote ones) can block the GUI request well past a reasonable budget.
- **Test verdicts ignore the pre-edit baseline.** `run_impacted` reports any failure as
  edit-caused; a repo that already fails sends the repair loop chasing pre-existing
  errors, despite `memory.observe_command` already recording command history.
- **Coverage findings are advisory only.** `check_coverage` warns about unaddressed
  clauses but nothing seeds a follow-up edit from them; a half-answered multi-clause
  task can end with a warning instead of a second pass.
- **Memory lessons have no effectiveness signal.** Lessons are injected and retired by
  file existence, but nothing records whether a lesson actually prevented its failure,
  so ineffective lessons persist indefinitely.
- **`how_index` coverage is asserted against a hardcoded list.** `test_how.py` checks a
  fixed set of CLI names, not the actual argparse subcommands, so a new CLI command can
  ship without a how-card and the test stays green.

---

## 5. Roadmap (extends TECHNIQUES.md; avoids the 31 implemented + 9 rejected)

Prioritized by impact vs complexity. Each item names exact files, the algorithm, and a
deterministic test (using `tests/fake_llm.py` / `fake_mcp.py` / `fake_lsp.py` where a
model or server is needed). None duplicates an implemented technique or a rejected one.

### R1. Resolution-pinned, rebinding-proof fetch  (impact: high, complexity: medium)
- **Files:** `pedacito/security.py` (new `resolve_checked(host)`), `pedacito/webfetch.py`,
  `pedacito/tools.py` `_web_open`.
- **Algorithm:** resolve the hostname once with `getaddrinfo`, run `is_private_host`
  on **every** returned IP, refuse if any is private (unless explicitly allowed), then
  pin the connection to the validated IP (connect by IP, set `Host`/SNI to the name) so
  a second DNS answer cannot swap in a private address mid-request.
- **Test:** monkeypatch `socket.getaddrinfo` to map `evil.example` → `127.0.0.1` and
  assert refusal with no request issued; a public IP passes. Deterministic, no network.
- **Why not rejected:** this is the deterministic substitute for a model judgement, in
  the spirit of the through-line; it closes the last SSRF hole the docs claim is shut.

### R2. Baseline-aware test verdicts  (impact: high, complexity: low–medium)
- **Files:** `pedacito/testsel.py` `run_impacted`, `pedacito/memory.py`.
- **Algorithm:** before attributing a failure to the edit, consult the recorded command
  history (`observe_command` facts). If the same narrowed command was already failing
  before this run, label the result "pre-existing failure (recorded before this run)"
  and do not feed it to the repair loop as edit-caused.
- **Test:** seed a failing command fact, run `run_impacted` with a stub runner that
  fails, assert the description carries the pre-existing label and the repair trigger
  is suppressed. Uses the existing fake runner seam.
- **Why not rejected:** zero model calls; reuses harness facts the ledger already keeps.

### R3. Wall-clock budget for discovery/scan  (impact: medium, complexity: low)
- **Files:** `pedacito/discovery.py` `discover_servers`/`describe_models`,
  `pedacito/gui/settings_api.py` `scan_models`.
- **Algorithm:** thread a monotonic deadline through the probe loop; allocate each root
  a slice of the remaining budget and stop early (returning partial results) when the
  budget is spent.
- **Test:** a fake server that sleeps past the budget; assert the scan returns within
  ~budget with the servers probed so far. Deterministic with a local threaded server.

### R4. Coverage-seeded second-pass intents  (impact: medium, complexity: medium)
- **Files:** `pedacito/agent.py` `plan_edit_intents`/`apply_edit_intents`,
  `pedacito/verify.py` `check_coverage`.
- **Algorithm:** after a batch, if `check_coverage` flags unaddressed edit clauses,
  seed the incremental planner's intents from those requirements' terms/anchors and
  run one bounded follow-up pass (respecting the existing call ceilings).
- **Test:** `fake_llm.py` scripted two-clause task where the first batch addresses only
  clause one; assert a second pass emits an intent naming clause two. Deterministic.
- **Why not rejected:** not self-critique or a search tree — a deterministic trigger
  from the existing requirement state machine, one bounded pass.

### R5. Lesson effectiveness counters  (impact: medium, complexity: low–medium)
- **Files:** `pedacito/memory.py`, `pedacito/loop.py`.
- **Algorithm:** when an injected lesson's failure kind recurs, increment a per-lesson
  "recurred" counter; when a run avoids it, increment "avoided". Retire lessons with
  zero effect after N runs. Harness-fact records only — no model-written reflection.
- **Test:** inject a lesson, simulate the same failure kind recurring, assert the
  counter increments and an ineffective lesson is retired after the threshold.
- **Why not rejected:** extends §24 memory with deterministic counters, explicitly not
  the rejected "model-written reflection notes".

### R6. CLI / how-index drift guard  (impact: medium, complexity: low)
- **Files:** `tests/test_how.py`, `pedacito/how_index.py`, `pedacito/cli.py`.
- **Algorithm:** derive the expected CLI command set from the actual argparse
  subparsers and assert `how_index.CLI_COMMANDS` covers it (and gather actions ↔ tool
  cards), replacing the hardcoded list.
- **Test:** static, deterministic; adding a subcommand without a card fails the test.

### R7. Config-editor round-trip property test  (impact: medium, complexity: low)
- **Files:** `tests/test_config_editor.py`.
- **Algorithm:** generate comment/string/table shapes (including the FIX-011 class:
  comments quoting triple delimiters, literal strings, multi-line values) and assert
  `set_values`/`remove_profile`/`add_profile` always yield text `validate_text` accepts
  and never duplicate a table.
- **Test:** deterministic seeded generation; no model. Guards the whole textual-patch
  invariant, not just one regression.

### R8. Review hunk provenance  (impact: medium, complexity: medium–high)
- **Files:** `pedacito/gui/core.py`, `pedacito/review.py`, `pedacito/verify.py`.
- **Algorithm:** attach the requirement clause that motivated each diff hunk, so the
  partial-accept UI can show which clause a file serves and coverage can flag a clause
  left unaccepted.
- **Test:** `fake_llm.py` two-clause task; assert review rows carry clause ids and
  accepting one file leaves the other clause flagged. Deterministic.

---

## 6. Verification log

Full-suite results as each fix landed (2 skipped throughout; skips are pre-existing
environment guards, not regressions):

| Stage | Passed |
|---|---|
| baseline (before fixes) | 1666 (9 failed) |
| FIX-004 | 1682 |
| FIX-005 | 1683 |
| FIX-006 | 1684 |
| FIX-007 | 1684 |
| FIX-008 | 1685 |
| FIX-009 | 1686 |
| FIX-010 | 1687 |
| FIX-011 | 1689 |
| FIX-012 | 1690 |
| FIX-013 | 1691 |
| FIX-014 | 1692 |
| FIX-015 | 1694 |
| FIX-016 | 1695 |
| FIX-017 | 1696 |
| FIX-018 | 1698 |
| final (docstring drift pass) | 1698 |