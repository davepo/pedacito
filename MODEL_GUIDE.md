# Pedacito model guide

This guide helps you choose a model for your hardware, set it up in Pedacito,
and check that it is working. It applies to any OpenAI-compatible server (LM
Studio, Ollama, Lemonade Server, llama.cpp, vLLM, and others) and to hosted APIs.
Model names and quantization details change often. Treat the examples as
starting points, and measure on your own machine.

Pedacito is not a chat benchmark. It asks a model to do three jobs:

1. **Choose lookups** through a JSON tool schema (`read`, `grep`, `find`,
   `callers`, `context`, `done`).
2. **Write or edit code** as exact search/replace blocks, against source it has
   actually read.
3. **Verify or explain** what it found, in a short prose answer.

A model that wins a chat leaderboard is not always the model that wins here.
Reliable tool calls, following the edit format, and token discipline matter more
than raw parameter count.

---

## The short version

- **Pick the largest model that fits with 1.5 to 2.5 GB of VRAM to spare.** A
  model that spills out of VRAM is much slower than a smaller one that fits.
- **Start with thinking off** for your worker model. Turn it on only for a
  planner or verifier profile, and only after measuring it.
- **Use Q4_K_M as the default quantization.** Use Q5 or Q6 if the model still fits
  comfortably. Avoid Q2 and IQ2 for code editing.
- **Calibrate every model once** with `pedacito calibrate`. Compare models on
  your own hardware with the six built-in benchmarks, or with `pedacito eval`.
- **Do not pick a setup from the model name alone.**
- **Let the server hold the JSON grammar.** Pedacito asks the engine for a
  strict JSON schema on every gather step and, per model, falls back to
  json_object or plain prompting when a server refuses it
  (`enforce_json_schema`, TECHNIQUES.md §27). Weak models stop failing on
  syntax.
- **Pair a fast scout with the architect.** `scout_profile = "<small
  profile>"` runs the lookup steps of gather on a 4B to 9B model while the
  big model keeps planning, answers and edits (TECHNIQUES.md §28). On one
  GPU it keeps the architect's context free; on two it is the natural split.

---

## Hardware tiers

Find the row that matches the VRAM available to your model server. That is the
GPU memory the server can use, not the card's total if other programs use it.
On Apple Silicon or another unified-memory machine, count only the slice you
give the model.

| Hardware | Model class | Examples (Q4_K_M unless noted) | Context | Pedacito tier | Worker thinking |
|---|---|---|---|---|---|
| **Under 8 GB VRAM** | 4B to 7B | Qwen3-4B, Gemma 3 4B, Qwen2.5-Coder 7B (tight) | 4k to 8k | `tiny` or `small` | off |
| **8 to 16 GB VRAM** | 8B to 14B | Qwen3-8B, Gemma 3 12B, Qwen3-14B (comfortable at 16 GB, tight at 12 GB) | 8k to 16k | `small` or `medium` | off. Planner or verifier profile may use thinking |
| **24 GB VRAM** | 24B to 35B, dense or MoE | Mistral Small 24B, Gemma 3 27B, Qwen3-32B, Qwen3-30B-A3B (MoE) | 16k to 32k | `medium` or `large` | off by default. Measure before enabling |
| **Hybrid local + cloud** | Any of the above, plus a hosted model | A local worker, with a hosted planner or verifier | Per profile | Per profile | Per profile |

Sizes below are for the weights alone. Add the KV cache and about 1.5 to 2.5 GB of
overhead.

### Under 8 GB VRAM (4B to 7B)

- **Good fits:** 3B to 4B models at Q4_K_M or Q5_K_M, such as Qwen3-4B or
  Gemma 3 4B. A 7B coder model at Q4 works at 4k to 8k context with no other GPU
  programs running.
- **Expect:** a smaller map and less source per lookup, so tasks should be narrow.
  Name the file in the task.
- **Avoid:** models of 12B and up, and thinking on a 4B model for routine edits.
- **Settings:** `tier = "tiny"` (4B and under) or `"small"` (up to 10B),
  `disable_thinking = true`, and context 4k to 8k. Let `auto_fit` choose the
  budgets. Set `context_tokens` only if the server reports the wrong window.

```toml
[universal]
vram_gb = 8

[profiles.worker]
model = "qwen3-4b"
tier = "tiny"
disable_thinking = true
```

### 8 to 16 GB VRAM (8B to 14B)

- **Good fits:** 8B models at Q6 or Q8, a 12B model at Q4, and 14B models at Q4 to
  Q6. A 14B model at Q4 needs about 9 GB for its weights, so it is comfortable on
  16 GB and tight on 12 GB.
- **Expect:** a 16k context to be practical on 12 GB or 16 GB. Lower it if the
  server reports less free memory.
- **Avoid:** 24B and larger at Q4 on this hardware. The weights leave too little
  room for a useful KV cache.
- **Settings:** `tier = "small"` or `"medium"`, `disable_thinking = true` for the
  worker, and context 8k to 16k. This is the first tier where a thinking planner
  profile is worth trying. See [Hybrid local + cloud role routing](#hybrid-local--cloud-role-routing).

```toml
[universal]
vram_gb = 16
roles = { planner = "think", verifier = "think" }

[profiles.worker]
model = "qwen3-14b"
tier = "medium"
disable_thinking = true

[profiles.think]
model = "qwen3-14b"
tier = "medium"
disable_thinking = false
max_tokens_answer = 8000     # room for reasoning plus the edit
context_tokens = 16384
```

### 24 GB VRAM (24B to 35B, dense or MoE)

- **Good fits:** Mistral Small 24B at Q4 to Q6, Gemma 3 27B at Q4 or Q5,
  Qwen3-32B at Q4, and Qwen3-30B-A3B. That last one is a mixture-of-experts model
  with about 3B active parameters per token.
- **MoE models:** VRAM holds the total weights, not just the active ones. A 30B
  MoE at Q4 still needs roughly 18 GB for its weights. It generates faster than a
  dense model of the same size, which makes it attractive on 24 GB. Size these by
  the file size the server reports.
- **Expect:** a 32k context at Q4 on a 24 GB card with nothing else on the GPU.
- **Avoid:** 70B models at Q3 on this hardware. They may load, but editing quality
  and speed suffer.
- **Settings:** `tier = "medium"` or `"large"`, context 16k to 32k, and worker
  thinking off until you have measured it. Role routing for the planner and
  verifier is useful at this size.

```toml
[universal]
vram_gb = 24
roles = { planner = "think", verifier = "think", summarizer = "worker" }

[profiles.worker]
model = "qwen3-32b"
tier = "large"
disable_thinking = true

[profiles.think]
model = "qwen3-32b"
tier = "large"
disable_thinking = false
max_tokens_answer = 12000
context_tokens = 32768
```

Do not route `summarizer` to a separate model on one GPU unless the extra quality
is measurable. Summary calls are frequent, and reloading a model is slow.

### Hybrid local + cloud role routing

A local worker handles most of the work, and a hosted model takes the jobs where
a stronger model makes the biggest difference: planning a large goal, verifying a
change, or escalating after a failed attempt. Pedacito routes each job by role,
and each role is optional.

```toml
[universal]
roles = { planner = "big", verifier = "big", summarizer = "tiny", escalate = "big" }

[profiles.local]                  # the default worker: lookups and edits
model = "qwen3-9b"

[profiles.big]
base_url = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"  # read from the environment, not stored in the file
model = "deepseek-chat"

[profiles.tiny]
model = "qwen3-1.7b"
```

| Role | What it does | Suggested setting |
|---|---|---|
| `worker` | Lookups and edits, the active model | Local; thinking off for small models |
| `planner` | Turns a goal into a checklist, and plan steps | A stronger local or hosted model, thinking on if it fits |
| `verifier` | Adversarial check of an answer before it is written | Thinking on for 16 GB and up, or a hosted model |
| `repair` | Fixes failed edits and test failures | Usually the worker. Thinking can help with format errors |
| `summarizer` | One-line summaries and compaction | The worker, or a tiny local model if you measured the gain |
| `escalate` | One more attempt on a larger model when a task failed: no edit landed, a semantic test failure outlived repair, two no-progress leases, or repeated zero-edit runs | A larger local or hosted model |
| `fallback` | Takes over when the worker errors out | A hosted or larger model |

Each escalation rung is used at most once per task, and only when configured, so
a bad local run cannot quietly spend a hosted budget. Use `--role planner=big` to
route one job for a single run. `pedacito profiles --recommend` suggests a
profile for each role, using measured scores, and `--apply-roles` writes the
choice after you confirm it.

Escalation is reason-based (`auto_escalate`, on by default): a patch that
applies but does not work earns the retry too. The reasons are tests still
failing with runtime/semantic errors (a TypeError, a failed assertion, an
infinite loop) after the repair rounds, two no-progress leases on one task,
and `escalation_zero_edit_runs` consecutive zero-edit runs on the active
profile (default 3; counted per profile in `.pedacito/profile_stats.json`,
0 disables the statistic). Every reason emits an `escalation_suggested`
event even when no `escalate` role is configured, so watch for it before
spending on a hosted rung: it names WHICH failure shape keeps recurring. A
profile that keeps suggesting escalation on zero-edit runs usually wants a
better prompt pack or a larger worker, not more retries -- the rung still
fires at most once per task, and a success clears the profile's count. When
no distinct `escalate` role is configured, a fired reason buys ONE same-model
retry with the edit style flipped (anchor <-> quote, unset -> anchor) instead
of nothing -- a `strategy_shift` event marks it. Green runs are never retried,
and `auto_escalate = false` disables the shift along with the rest.

Hosted models receive the task text and the code excerpts the task needs. Pedacito
warns before sending a task that appears to contain a secret. Cap spending with
`max_total_tokens` or `max_cost`, and set `price_input_per_mtok` and the matching
output price so the cost is counted.

---

## Dual-GPU / Dual-Model Scout & Architect Routing (`scout_profile`)

Gather is the loop where the model looks things up before it edits. Its steps are
short and tightly constrained: grep for this, read that range, outline this file.
A small model does those well and quickly. Planning, the answer and the edits need
the larger model. `scout_profile` gives each job to the right model:

- **Scout (4B to 9B):** the lookup steps of gather. These are `read`, `grep`,
  `find`, `outline`, `callers`, `context`, `usages`, `run_test` and the other
  investigation actions. Put it on a second GPU, or on a lightweight endpoint.
- **Architect (27B to 35B):** the active profile. It does the planning, the
  answer, every edit, repair and verification.

```toml
[universal]
scout_profile = "scout"             # the lookup model; must name a profile below

[profiles.architect]                # the active profile: the big model
base_url = "http://localhost:1234/v1"
model    = "qwen3-32b"
tier     = "large"
vram_gb  = 24
disable_thinking = true
context_tokens = 32768

[profiles.scout]                    # the second GPU, or a lightweight endpoint
base_url = "http://192.168.1.50:11434/v1"
model    = "qwen3-4b"
tier     = "tiny"
vram_gb  = 8
disable_thinking = true
```

Set the profile's `vram_gb` to the memory of the machine that serves that model.
Each profile is sized on its own, so the scout's 8 GB does not limit the architect.

What to expect:

- **Architect and scout are used in the same task.** The answer sees the same
  gathered material whichever model ran the lookups, so routing does not add a
  second opinion to the answer. Usage from both models appears in the same totals.
- **A missing or unreachable scout is not an error.** Pedacito warns once and runs
  the lookups on the architect. A scout set to the same profile as the architect is
  ignored.
- **The GUI shows which model is the scout.** While it is active, the top bar shows
  a `Scout: <profile>` badge. In **⚙ Settings → All settings**, `scout_profile` is a
  dropdown of your profiles, with **None (use active profile)** to turn it off.
- **Hosted scouts send your code.** A hosted scout receives the lookup results,
  which are excerpts of your code. The same warnings that apply to hosted roles
  apply here.

Check the scout before you rely on it. Its job is to pick the right lookup, so run
`pedacito calibrate --profile scout` (tool choice is the probe that matters most)
and compare `pedacito eval` before and after you add it.

**On one GPU**, the benefit is smaller. Both models must fit in VRAM together, with
room for the architect's KV cache. If they do not, every switch reloads a model,
which is slower than running the lookups on the architect. The routing pays off most
when the scout runs on its own card, or when the scout stays loaded next to the
architect. The cost is a second server, a warm-up ping before the first lookup, and
one more setting to keep in sync with your profiles.

---

## Engine-enforced JSON by server (`enforce_json_schema`)

Each gather step is a JSON object with a fixed set of fields. With
`enforce_json_schema = true` (the default), Pedacito sends the step schema to the
server as `response_format`, so the server constrains sampling to valid JSON for
that schema. A step that does not match the schema cannot be produced, so most of
the retries that used to repair malformed replies are no longer needed.

Pedacito walks down three tiers, one request at a time:

| Tier | Request sent | What the server guarantees |
|---|---|---|
| `schema` | `json_schema`, `strict: true` | Valid JSON for the step schema, enforced while sampling |
| `object` | `json_object` | Valid JSON, but any shape. The prompt lists the fields |
| `plain` | nothing extra | Nothing from the server. The prompt and the few-shot examples keep the format |

If a server answers HTTP 400 and the message names `response_format`,
`json_schema` or `json_object`, that tier is switched off for this `base_url` and
model, and the same request is sent one tier lower straight away. The first call
pays for the refusal, and later calls skip the tier that was refused. The result is
remembered for the session. `pedacito config --doctor` probes the modes up front and
saves the answer in `server_modes.json`, so a server that refuses a tier is handled
before the first task.

Per server, this is what to expect. Server features change between releases, so
treat this as a starting point and confirm with `pedacito config --doctor`.

| Server | Structured output | What to expect |
|---|---|---|
| **llama.cpp** (`llama-server`) | `json_schema` is converted to a grammar | The `schema` tier on current builds. An older build that refuses `json_schema` falls to `object` |
| **Lemonade Server** (`lemond`) | Its llama.cpp backend accepts `response_format` | The `schema` tier when the loaded backend accepts it. `--doctor` shows which tier it uses |
| **LM Studio** | `json_schema` structured output | The `schema` tier on recent versions. Older versions fall to `object` |
| **Ollama** (`/v1`) | `response_format` with `json_schema` | The `schema` tier on recent versions. Older versions fall to `object` |
| **vLLM** | `response_format` with `json_schema` (guided decoding) | The `schema` tier when the structured-output backend is available in your build |
| **Hosted APIs** (OpenAI, DeepSeek and others) | Pedacito defaults remote hosts to `json_mode = "object"`, which suits DeepSeek and most others | The `object` tier by default. Set `json_mode = "schema"` in a profile for an API that accepts `json_schema`, such as OpenAI, to use the `schema` tier |

Two things to know:

- **Native tool calls bypass this.** With `tool_mode = "native"`, the step is sent
  through the model's tool-call channel, and `response_format` is not used. The
  setting still applies to every other JSON call, such as planning.
- **Turn it off only for a misbehaving server.** `enforce_json_schema = false` sends
  plain prompts only. Use it when a server produces broken output under constrained
  sampling. Keep it on otherwise, since it is the cheapest way to avoid malformed
  steps.

---

## Local inference acceleration (`enable_prompt_cache`)

Gather is a loop of short calls that re-send a growing, mostly unchanged
conversation. Self-hosted runtimes can make that cheaper with two request keys,
and Pedacito adds them for you:

- `cache_prompt: true` asks the server to keep the KV prefix of the identical
  prompt start between calls, so step 12 does not re-prefill steps 1 through 11.
  On by default (`enable_prompt_cache = true`).
- `prompt_lookup_num_tokens: N` turns on n-gram prompt-lookup decoding: the
  drafter copies continuations from the prompt itself. It is a strong fit for
  this workload because the edit phase quotes source lines that are already
  sitting in the context. Off by default (`enable_prompt_lookup = false`);
  `prompt_lookup_num_tokens` sets the draft span (default 10).

Per runtime, what to expect (confirm with `pedacito config --doctor`):

| Server | `cache_prompt` | Prompt lookup |
|---|---|---|
| **llama.cpp** (`llama-server`) | Accepted; pairs with `--cache-reuse` on the server for best results | Accepted; enable `--lookup-ngram-draft` style flags on the server, or leave it and the refusal is learned |
| **Lemonade Server** (`lemond`) | Its llama.cpp backend accepts the key | Same surface as llama.cpp |
| **LM Studio** | Accepted on recent versions | Support varies by version |
| **Ollama** (`/v1`) | Accepted | Not supported on older builds; the fallback below covers it |
| **vLLM** | Accepted (`--enable-prefix-caching` is on by default in recent versions) | Not the same mechanism; expect the fallback |
| **Hosted APIs** | Never sent | Never sent |

Two safety properties matter. The keys are attached only to endpoints that are
actually yours: loopback, a LAN address (RFC1918, CGNAT, IPv6 unique-local), a
`.local` name, or a provider entry that names a local runtime. A hosted API
never sees them. And a server that does not understand a key is not a broken
setup: when a request answered by HTTP 400 or 422 still had either key
attached, Pedacito strips the keys, re-sends that one request immediately, and
remembers the refusal for this server and model, so later calls start clean.
The cost of an unsupported key is one extra request per session, never a
failing task. A key you spell yourself in a profile's `[sampler]` table always
wins; the acceleration never overwrites the user's own setting.

Rule of thumb: keep `enable_prompt_cache = true` on any local server. Turn on
`enable_prompt_lookup = true` when edits dominate the run (SEARCH blocks,
refactors) and the GPU is compute-bound; measure with `pedacito calibrate` or
a side-by-side `chat` session if in doubt.

---

## Thinking per profile

Some models reason before they answer, in a separate thinking block. That helps
on hard problems. On a small model it can also use up the answer budget, so the
edit never arrives.

### Turn it off or on for one profile

In the GUI, open **Settings → Profiles**, select the profile, and set
`disable_thinking` under **Reasoning & Execution Overrides**. The chip in the top
bar shows **no-think** or **thinking** for the active profile.

In the config file:

```toml
[profiles.worker]
model = "qwen3-14b"
disable_thinking = true      # no reasoning for answers, edits, repair, or verifier calls

[profiles.think]
model = "qwen3-14b"
disable_thinking = false
max_tokens_answer = 8000     # raise the budget when thinking is on
```

Pedacito already turns thinking off for its lookup loop, planning, and scaffolding
calls, because those are JSON calls where reasoning only adds cost. The setting
controls the answer, repair, verifier, and summary calls, where reasoning
competes with the output budget.

If a model has a prompt switch instead of an API switch, set `thinking_suffix`,
such as `/no_think` for Qwen3. Leave it empty when the server already honors
`disable_thinking`.

### When to turn thinking on

- Hard debugging, cross-file tracing, architecture questions, or a task with
  several requirements that interact.
- A planner that must split a broad goal into tasks that do not overlap.
- A verifier that must catch a subtle logic error.

Raise `max_tokens_answer` by about 2,000 to 4,000 when thinking is on. Without
the extra room, most runs end in truncation and retries.

### When to turn it off

- Routine lookups and small, localized edits.
- Windows of 4k to 8k, where reasoning competes with the map and the source.
- Models that reason at length but write weak edits. The answer budget runs out
  before the search/replace block.
- Any setup where the logs show `finish_reason = "length"` or repeated "would not
  produce valid JSON" errors.

`pedacito calibrate` runs a thinking A/B test. If a model only works with
thinking off, the model card records `disable_thinking = true`, so later runs
start in that mode.

### Models that always reason

Some models have no off switch: R1-style distills, LFM2.5-Thinking,
LFM2.5-24B-A2B, GPT-OSS, and Kimi K2.7 and K3. Pedacito strips their reasoning
blocks from the output, but it cannot remove the token cost. For a small worker,
prefer an instruct or non-thinking variant of the same family when one exists.

### Per-family notes

These are examples. Check the model card and chat template for the exact switch.

| Family | What to expect |
|---|---|
| **Qwen3 / 3.5** | Hybrid thinking, switched with `enable_thinking` or `/no_think`. Strong tool use. A good worker with thinking off and a good planner with it on. |
| **Gemma 3 / 4** | Gemma 3 has no thinking. Gemma 4 has reasoning off by default and opt-in. Good instruction following. |
| **Nanbeige 4.x** | Thinking is on by default, and output includes `<think>` blocks. Strong for its size, but token-heavy. Turn it off for the worker. |
| **LiquidAI LFM2.5 / LFM3** | Very small and fast. LFM2.5-Thinking and LFM2.5-24B-A2B always reason. LFM3 supports `enable_thinking`. Prefer an Instruct variant for routine editing. |
| **DeepSeek R1 and V3.x** | R1-style models always think and can be slow. V3.1 and later hybrids expose a `thinking` switch. Budget for it, or use a non-thinking coding model as the worker. |
| **GLM 4.x / 5** | Hybrid `enable_thinking` and strong tool use. Large variants need substantial VRAM. |
| **GPT-OSS** | Always reasons. `reasoning_effort` lowers the cost but does not turn it off. Good tool use. The 20B size is the realistic local option. |
| **Mistral / Magistral** | Magistral-style models use their own reasoning format and often always reason. Mistral Small 4 has an effort toggle. |
| **Granite** | 3.2 and 4.2 have thinking toggles. Small and instruction-oriented. Check the version. |
| **Phi-4 reasoning** | Reasoning is written in `<think>`-style tags, imposed by the prompt. No standard off switch. Can be slow on long reasoning. |

---

## Calibration: `pedacito calibrate`

Calibration measures a model on the things Pedacito depends on, then saves the
result as a **model card**. The card tunes Pedacito's defaults for that model. It
does not replace your own measurements.

```bash
pedacito calibrate --profile worker          # full run, about a few minutes
pedacito calibrate --quick --profile worker  # skip the long-context and capability probes
pedacito calibrate --repeats 5               # run each probe five times
pedacito calibrate --recalibrate             # measure again even with a fresh card
```

Calibration runs a scripted set of probes:

- **Tool choice:** does the model pick the right lookup action?
- **JSON adherence:** does it reply in the required structure?
- **Edit format:** does it write valid search/replace blocks?
- **Injection and honesty:** does it ignore a planted instruction in the source,
  and does it admit what it has not read?
- **Long-context retrieval:** can it find a needle in a long prompt?
- **Continuation:** can it answer from a warm conversation?
- **Thinking A/B:** does hidden reasoning eat the answer budget? Each question is
  answered with thinking on and off.
- **Native tool calls:** the same scenarios through the model's tool-call channel,
  compared with the JSON grammar.

Each probe runs three times by default, or once with `--quick`. The card records
the mean and the spread, so the decisions come from an average, not one lucky
reply.

The card sets these values:

- `prompt_pack`, which controls how much instruction text the model gets
- `answer_mode`, whether the answer continues the lookup conversation
- `verify_answer` and `best_of`, which add checks and extra samples for weaker models
- `tool_mode`, `grammar` or `native`
- `fuzzy_match` and `edit_style`
- `disable_thinking`, when the thinking A/B shows the model needs it off

Only settings you left at their defaults follow the card. Anything you set
explicitly still wins. `pedacito config` shows the card and whether it is in use.

Cards expire after 30 days, and a card from an older version is shown with a note
and not applied. Calibrate again after you change the model, its quantization, or
its chat template. In the GUI, each profile in **Settings → Profiles** has a
**Calibrate** button that runs a quick measurement and streams the output to the
terminal panel. `pedacito models scan` also calibrates each new model with a quick
run by default, so you get a card as you add models.

---

## Compare models with the diagnostic benchmarks

Pedacito includes six small diagnostic projects. For each one, a model builds the
project through the normal Pedacito loop, and the result is graded in two ways:

1. **The model's own tests** (`pytest -q`, or the project's verification script).
2. **A hidden holdout rubric**, written by Pedacito, that uses only the public
   API the task describes. It checks the things small models often get wrong.

The six projects contain 71 holdout checks in total.

| Benchmark | What it asks for | What it catches |
|---|---|---|
| **Stateful CLI Expression & RPN Calculator** (`rpn_expression_calc`, 12 checks) | An infix calculator with variables and an RPN mode, as a module and a CLI | Operator precedence, right-associative `**`, unary minus, variables across arguments, exit codes |
| **Markdown Table & Box-Drawing Formatter** (`markdown_table_fmt`, 12 checks) | Parse Markdown and CSV tables, and write aligned Markdown or box-drawing output | Padding off by one, escaped pipes, wide CJK characters, alignment markers, truncation |
| **Thread-Safe TTL + LRU Cache** (`ttl_lru_cache`, 13 checks) | A two-file cache with TTL, LRU eviction, a memoize decorator, and stats | Eviction order, expired items evicted before live ones, `functools.wraps`, cross-file imports, thread safety |
| **Streaming Log Parser & JSON Report** (`log_anomaly_analyzer`, 11 checks) | Parse mixed logs, compute percentiles per service, flag 5xx spikes, and emit strict JSON | Regex accuracy, nearest-rank percentiles, rolling 60-second windows, JSON discipline |
| **Single-File Interactive Kanban Board** (`mini_kanban_html`, 13 checks) | A zero-dependency HTML, CSS, and JavaScript board with filters, persistence, and import/export | DOM ids that match between HTML and JS, event wiring, localStorage, CSS layout, offline delivery |
| **Pelican Riding a Bicycle (Parametric SVG & Geometry)** (`pelican_bicycle_svg`, 10 checks) | A Python script that writes a labelled SVG drawing of a pelican on a bicycle, a browser preview page, and geometry tests | 2D coordinate geometry, the seven required semantic groups, well-formed XML with `viewBox="0 0 800 600"`, two wheels on one ground line, a large beak, a body above the axle, and a minimum of visual detail |

The pelican benchmark is the spatial one. Its wheel and body checks read the
coordinates the model wrote, so a drawing made of two empty circles fails. Its
visual checks also need at least 15 graphic primitives and four colours.

### Run a benchmark

1. Click **▦ Benchmarks** in the GUI top bar.
2. Choose the profile to test. Each model you want to compare needs its own
   profile, so create one per model first.
3. Optionally choose the folder for the output. By default it is a `benchmarks/`
   folder beside the project you have open.
4. Click **Run Benchmark**. The dialog shows progress, then the result.

Each run creates a new folder named like `pedacito_bench_<benchmark>_<profile>_<time>`.
Nothing is deleted, so you can open the project, run its tests yourself, and
read the code.

### Read the report

Each run writes `BENCHMARK_REPORT.md` in its folder, along with `benchmark_run.json`.
The report has these sections:

- **Run:** the benchmark, profile, model, start time, duration, and whether
  generation finished cleanly.
- **Score:** an overall PASS or FAIL, with the model's test results and the
  holdout results. The overall result is PASS only when both pass.
- **Usage:** tokens and model calls for the run.
- **Shortcomings detected:** the specific failures the harness found.
- **SVG preview:** when the project contains an SVG (the pelican benchmark's
  `pelican_bicycle.svg`), the GUI scorecard shows the drawing inline. It is
  rendered as a sandboxed image, so nothing in the file can run.
- **Harness holdout rubric:** each check, its description, and pass or fail.
- **Requirement ledger:** each clause of the task and the state it reached.
- **Model test output** and **Holdout rubric output:** the full pytest output.
- **Run it yourself** and **Generated files:** the commands to rerun the project.

### Compare two models fairly

1. Use the same hardware and the same Pedacito settings. Only the model should
   change.
2. Run the same benchmark with each model's profile. For a stronger comparison,
   run each benchmark more than once, because a single run is noisy.
3. Compare the holdout checks passed out of the total, then the overall PASS or
   FAIL, then the model's own tests.
4. Look at the time and tokens in **Usage**. A model that passes at twice the cost
   may still be the wrong choice for your hardware.
5. Read the **Shortcomings** section. A model that passes the rubric but leaves a
   placeholder or an untested edge case tells you something the score does not.

Benchmarks need a model that can build a small project from a spec in one run,
so they say more about the whole loop than about a model's raw knowledge. Use
them to compare candidate models on the same job, not as a leaderboard.

---

## How to measure

### 1. Check the setup

```bash
pedacito config --check --profile worker
pedacito config --doctor --profile worker
```

`--check` verifies the server and the model, and reports the real context window.
`--doctor` probes structured-output modes, the tokenizer rate, and server details.
If the server reports a stale window, set `context_tokens` yourself.

### 2. Calibrate the model

```bash
pedacito calibrate --profile worker
```

See [Calibration](#calibration-pedacito-calibrate) above.

### 3. Run the eval, then A/B it

`pedacito eval` runs a fixed set of tasks over small fixture repositories (a
Python shop, a TypeScript app, a Go CLI, and a larger Python service). Some tasks
plant a prompt injection in the source, so the eval also checks whether the model
follows instructions it finds in the code it reads.

```bash
pedacito eval --profile worker --out worker-off.json
pedacito eval --profile worker-think --out worker-on.json \
  --baseline worker-off.json
```

If the second run regresses, the command exits nonzero and lists what got worse:

- pass rate, locate rate, and first-apply-clean rate
- prompt injections that were followed
- prompt and completion tokens, calls, and wall time
- new failure types

Tune the gate with `--max-token-growth` and `--max-time-growth` if your hardware
has a known cost profile. Use `--repeats 3 --seed 1` to see the spread across
runs.

Pass rate alone is not enough. Prefer a setup that also keeps or improves:

- **first-apply-clean** (fewer repair calls)
- mean prompt and completion tokens
- cache-hit ratio
- wall time
- the rate of truncation and format failures

### 4. Test it on real work

The eval uses small fixtures. Before you trust a new model or profile, try it on
your own code:

```bash
pedacito ask PROJ -t "a real, narrow task"        # dry run
pedacito ask PROJ -t "..." --review               # inspect the diff
pedacito ask PROJ -t "..." --apply                # once you are confident
```

`--record` on a run, and `--replay` later, test changes to Pedacito's own
harness without a model. They do not measure a model upgrade. In CI, add
`--strict-replay` to fail the build when the harness asks a different prompt
than the recording.

### 5. Watch VRAM

Use your GPU tool (`nvidia-smi`, `rocm-smi`, or the server's own panel) while a
task runs. If the model spills into system RAM, generation slows sharply, and the
run looks like a model-quality problem when it is really a sizing problem. Lower
`context_tokens`, use a smaller model or quantization, or quantize the KV cache.

---

## Sizing and quantization

Pedacito's `auto_fit` fits its token budgets to the model's real window using:

- `vram_gb`, the GPU memory available to the model server
- the model's weight file size, reported by the server
- `context_tokens`, `reserve_tokens`, and `k_cache_quant` / `v_cache_quant`

It keeps about 0.5 GB free. Values you set explicitly in a profile always win.

A rough budget:

```
weights (quantized) + KV cache (grows with context) + 1.5 to 2.5 GB overhead
```

The KV cache is the second knob after the weights. Doubling the context roughly
doubles the KV cost. If your runtime quantizes the KV cache, set
`k_cache_quant` and `v_cache_quant` (for example `"q8_0"`) so `auto_fit` does not
overestimate. If the server runs with `--parallel` or a large batch, leave more
headroom.

| Quant | Rough size | Practical use |
|---|---|---|
| Q4_K_M | 0.55 to 0.6 bytes per parameter | The default. Best size-to-quality trade-off |
| Q5_K_M | About 0.7 bytes per parameter | When the model still fits comfortably |
| Q6_K | About 0.8 bytes per parameter | Small models, or spare VRAM. Better than Q4 for edits |
| Q8_0 | About 1.0 byte per parameter | Only when weights and context clearly fit |
| Q3 / IQ3 | About 0.4 bytes per parameter | A last resort for very large models. Avoid for editing |
| Q2 / IQ2 | About 0.3 bytes per parameter | Avoid for Pedacito work |

Rough weight sizes at Q4: a 7B model is about 4 to 5 GB, a 14B model about 9 to
10 GB, a 24B model about 14 to 16 GB, and a 32B model about 19 to 21 GB. Exact
sizes vary by architecture and tokenizer. MLX, EXL2, AWQ, and GPTQ use different
names for quantization. Aim for the equivalent of about 4 bits per weight.

> **Rule of thumb:** if you are unsure, pick a model one size smaller and give it
> more context. Pedacito benefits more from a model that fits with a 16k to 32k
> window than from the largest weights that barely load at 4k.

---

## Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| "Kept running past the token limit", or no JSON | Thinking on with a small answer budget | Set `disable_thinking = true` for the worker, or raise `max_tokens_answer` |
| Empty answer with hidden reasoning | Thinking runaway | Pedacito retries with thinking off. Set it off by default for small models |
| "search block not found" | The model copied line numbers, or the file shifted | Narrow the task. Pedacito re-anchors. Check the model's edit-format score in calibration |
| Answers are vague, or name the wrong file | The map is too compressed, or the model is weak at tool choice | Raise `flat_map_token_budget` or the context, or use a model that is better at tool calls |
| The model ignores tools | Grammar and native mode do not match | Run `pedacito calibrate` and compare `tool_mode = "grammar"` with `"native"` |
| Slow after a few minutes | The model spilled out of VRAM, or the context is too large | Lower the context or quantization. Watch GPU memory during a run |
| Role switches take a minute | Different models on one GPU reload their weights | Use the same model for the roles, or accept the load cost |
| Good chat, bad edits | The model is not edit-oriented | Prefer a coding or instruct model. Check first-apply-clean in `eval` |

---

## Quick checklist

- [ ] Set `vram_gb` to the memory actually available to the server.
- [ ] Start with Q4_K_M, and leave 1.5 to 2.5 GB for the KV cache and overhead.
- [ ] Under 8 GB, use a 4B to 7B instruct or non-thinking worker, with thinking off.
- [ ] At 8 to 16 GB, keep the worker non-thinking, and try thinking only for the
      planner or verifier.
- [ ] At 24 GB, measure thinking on and off before making it the default.
- [ ] Run `pedacito calibrate` once per model and quantization.
- [ ] Compare candidate models with the six benchmarks, on the same hardware.
- [ ] Run `pedacito eval` before and after a change, and use `--baseline` as a gate.
- [ ] Judge by first-apply-clean and token and time numbers, not pass rate alone.
- [ ] Keep role routing simple on a single GPU. Every model switch can reload.
- [ ] Do your first dozen real tasks without `--apply`, then use `--review`.
- [ ] Run `pedacito config --doctor` once per server to confirm its JSON tier.
- [ ] With two GPUs, set `scout_profile` to a 4B to 9B profile, and keep the 27B to
      35B model as the active architect.
