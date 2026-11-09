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
  your own hardware with the five built-in benchmarks, or with `pedacito eval`.
- **Do not pick a setup from the model name alone.**

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
| `escalate` | One more attempt on a larger model when a task applied no edit | A larger local or hosted model |
| `fallback` | Takes over when the worker errors out | A hosted or larger model |

Each escalation rung is used at most once per task, and only when configured, so
a bad local run cannot quietly spend a hosted budget. Use `--role planner=big` to
route one job for a single run. `pedacito profiles --recommend` suggests a
profile for each role, using measured scores, and `--apply-roles` writes the
choice after you confirm it.

Hosted models receive the task text and the code excerpts the task needs. Pedacito
warns before sending a task that appears to contain a secret. Cap spending with
`max_total_tokens` or `max_cost`, and set `price_input_per_mtok` and the matching
output price so the cost is counted.

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

Pedacito includes five small diagnostic projects. For each one, a model builds the
project through the normal Pedacito loop, and the result is graded in two ways:

1. **The model's own tests** (`pytest -q`, or the project's verification script).
2. **A hidden holdout rubric**, written by Pedacito, that uses only the public
   API the task describes. It checks the things small models often get wrong.

The five projects contain 61 holdout checks in total.

| Benchmark | What it asks for | What it catches |
|---|---|---|
| **Stateful CLI Expression & RPN Calculator** (`rpn_expression_calc`, 12 checks) | An infix calculator with variables and an RPN mode, as a module and a CLI | Operator precedence, right-associative `**`, unary minus, variables across arguments, exit codes |
| **Markdown Table & Box-Drawing Formatter** (`markdown_table_fmt`, 12 checks) | Parse Markdown and CSV tables, and write aligned Markdown or box-drawing output | Padding off by one, escaped pipes, wide CJK characters, alignment markers, truncation |
| **Thread-Safe TTL + LRU Cache** (`ttl_lru_cache`, 13 checks) | A two-file cache with TTL, LRU eviction, a memoize decorator, and stats | Eviction order, expired items evicted before live ones, `functools.wraps`, cross-file imports, thread safety |
| **Streaming Log Parser & JSON Report** (`log_anomaly_analyzer`, 11 checks) | Parse mixed logs, compute percentiles per service, flag 5xx spikes, and emit strict JSON | Regex accuracy, nearest-rank percentiles, rolling 60-second windows, JSON discipline |
| **Single-File Interactive Kanban Board** (`mini_kanban_html`, 13 checks) | A zero-dependency HTML, CSS, and JavaScript board with filters, persistence, and import/export | DOM ids that match between HTML and JS, event wiring, localStorage, CSS layout, offline delivery |

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
- [ ] Compare candidate models with the five benchmarks, on the same hardware.
- [ ] Run `pedacito eval` before and after a change, and use `--baseline` as a gate.
- [ ] Judge by first-apply-clean and token and time numbers, not pass rate alone.
- [ ] Keep role routing simple on a single GPU. Every model switch can reload.
- [ ] Do your first dozen real tasks without `--apply`, then use `--review`.
