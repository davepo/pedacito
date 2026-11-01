# Pedacito model guide

Practical guidance for choosing and configuring a local model for Pedacito.
It is model-agnostic: the examples name current popular families, but the
reasoning applies to any OpenAI-compatible server (LM Studio, Ollama,
Lemonade, llama.cpp, vLLM, SGLang, ...).

Pedacito is not a chat benchmark. It asks a model to do three different jobs:

1. **Choose lookups** through a JSON tool schema (read, grep, find, callers,
   context, done).
2. **Write or edit code** as exact search/replace blocks, against source it
   has actually read.
3. **Verify or explain** what it found, with a bounded prose answer.

The model that wins a chat benchmark is not always the model that wins here.
Tool-call reliability, instruction following, edit-block accuracy, and token
discipline matter more than raw parameter count.

---

## The short version

- **Small local models: thinking off** for the worker profile. Pedacito already
  disables thinking for gather/plan/scaffold calls; `disable_thinking` mainly
  controls answer/edit, repair, verifier, and compaction calls, where reasoning
  competes with the output budget.
- **Medium/large local models: thinking off by default, measure before turning
  it on.** A separate thinking-on profile for `planner`/`verifier` is often the
  best compromise.
- **Use the largest model that leaves 1.5–2.5 GB of VRAM** for the KV cache,
  desktop compositor, and server overhead. A model that spills out of VRAM is
  much slower than a smaller model that fits.
- **Q4_K_M is the default quant.** Q5_K_M/Q6_K are worth it when the model
  still fits; avoid Q2/IQ2 for code editing.
- **Measure once per model/quant** with `pedacito calibrate`, and measure
  changes with `pedacito eval --baseline`. Do not choose a setup from model name
  alone.

---

## How Pedacito uses a model

Pedacito has several model call sites, and they do not all want the same
settings:

| Phase | What the model does | Thinking in Pedacito |
|---|---|---|
| Planning / task splitting | JSON task list | **Forced off** |
| Gather loop | JSON lookup actions | **Forced off** |
| Scaffolding / interface stubs | JSON/source stubs | **Forced off** |
| Answer / edit | Search/replace blocks or prose | Config (`disable_thinking=false` by default: thinking on if the model reasons) |
| Repair | Correct failed edits | Config |
| Verifier | Adversarial check | Config |
| Compaction | Rewrite lookup digest | Config |
| `explain` | Prose answer | Config |

So `disable_thinking = true` does **not** change the lookup loop (it is already
off). It changes the phases where a small model is most likely to burn its
`max_tokens_answer` budget on reasoning instead of producing a usable edit.

`thinking_suffix` is separate: it appends a prompt soft switch such as
`/no_think`. Empty means auto when thinking is disabled (`/no_think` for
Qwen3/SmolLM3 only). Use it only for a model whose template does not honour the
hard switch.

### Models with no off switch

Some models always reason: R1 distills, LFM2.5-Thinking, LFM2.5-24B-A2B,
GPT-OSS, Kimi K2.7/K3, and similar. Pedacito still strips their output blocks,
but it cannot stop the token cost. For a small worker, prefer an **instruct** or
**non-thinking** variant of the same family when one exists.

---

## Sizing rules

Pedacito's `auto_fit` uses:

- `vram_gb`: the GPU memory available to the model server, not necessarily the
  card's total. A desktop compositor, browser, or another model may already
  use some. On Apple Silicon or another unified-memory system, set a
  conservative slice of the shared memory, not the whole machine's RAM.
- The model's weight file size, reported by the server.
- `context_tokens`, `reserve_tokens`, and `k_cache_quant`/`v_cache_quant`.

It keeps roughly 0.5 GB free and fits budgets to the window. Explicit values in
a profile still win. If the server reports a stale context length, set
`context_tokens` yourself.

A rough budget:

```
weights (quantized) + KV cache (grows with context) + 1.5-2.5 GB overhead
```

**KV cache is the second knob after weights.** Doubling context roughly doubles
the KV cost. If your runtime uses a quantized KV cache, set
`k_cache_quant` and `v_cache_quant` (for example `"q8_0"`) so `auto_fit` does
not overestimate. If the server runs with `--parallel` or a large batch, leave
more headroom.

### Quantization

| Quant | Rough weights | Practical use |
|---|---|---|
| Q4_K_M | ~0.55–0.6 bytes/param | Default for most local setups; best size/quality trade-off |
| Q5_K_M | ~0.7 bytes/param | Use when the model still fits comfortably |
| Q6_K | ~0.8 bytes/param | Small models or extra VRAM; better than Q4 for edits |
| Q8_0 | ~1.0 byte/param | Only when weights + context clearly fit; often not worth the VRAM |
| Q3 / IQ3 | ~0.4 bytes/param | Last resort for very large models; avoid for code editing |
| Q2 / IQ2 | ~0.3 bytes/param | Avoid for Pedacito work |

A 7B model at Q4 is roughly 4–5 GB; 14B is roughly 9–10 GB; 24B is roughly
14–16 GB; 32B is roughly 19–21 GB. These are only weights — add the KV cache
and overhead. Exact sizes vary by architecture and tokenizer. MLX, EXL2, AWQ,
and GPTQ use different quant names; aim for the equivalent ~4-bit size.

---

## What fits at each VRAM size

These are starting points, not guarantees. Actual file sizes vary by model,
quantization, architecture, and server. Quantized weights below assume GGUF at
Q4_K_M unless noted.

| VRAM | Practical model class | Context target | Pedacito tier | Worker thinking | Roles |
|---|---|---|---|---|---|
| **6 GB** | 3–4B Q4/Q5; 7–8B Q4 only at 4k and tight | 4k–8k | `tiny` | off | single model |
| **8 GB** | 7–9B Q4; 12B Q4 only at 4k | 8k–16k | `small` | off | single model |
| **12 GB** | 12–14B Q4/Q5, or 8B Q6/Q8 | 16k | `small` or `medium` | off for worker; think planner only if 16k+ | optional think profile |
| **16 GB** | 14B Q6 (Q8 small context), or 24B Q4 | 16k–32k | `medium` | off worker; think planner/verifier | yes |
| **24 GB** | 27–32B Q4 | 32k | `medium` or `large` | off or measured; think planner/verifier | yes |
| **32 GB** | 32B Q5/Q6, or 27B Q8 | 32k–64k | `large` | off default; measure on/off | yes |

### 6 GB

- **Safe:** 3B–4B at Q4_K_M/Q5_K_M. Examples: Qwen3-4B (Instruct or Thinking),
  Nanbeige4.2-3B, Gemma 3 4B, LFM2.5-1.2B/2.6B.
- **Possible but tight:** 7–8B Q4 at 4k context, no desktop apps using the GPU.
  Expect slower generation and less map/source budget.
- **Avoid:** 12B+; thinking on a 4B model for routine edits.
- **Settings:** `tier = "tiny"`, `disable_thinking = true`, context 4k–8k.
  Let `auto_fit` choose; set `context_tokens = 4096` only if detection is wrong.

### 8 GB

- **Safe:** 7–9B at Q4_K_M. Examples: Qwen3-8B, Llama 3.1 8B, Mistral 7B,
  Phi-4-mini, LFM2.5-8B-A1B, Nanbeige4.2-3B at Q6/Q8.
- **Possible:** Gemma 3 12B Q4 at 4k, or a 12B Q4 with a small window; not
  comfortable.
- **Avoid:** 14B+; thinking on the 8B worker.
- **Settings:** `tier = "small"`, `disable_thinking = true`, context 8k–16k.
  Use the model card to pick `tool_mode` (native vs grammar) after calibration.

### 12 GB

- **Safe:** 12–14B Q4/Q5. Examples: Qwen3-14B Q4, Gemma 3 12B Q5, Phi-4 14B
  Q4, DeepSeek-R1-Distill-Qwen-14B Q4 (always thinks).
- **Also good:** 8B Q6/Q8 if you value quality per parameter over raw size.
- **Avoid:** 24B Q4; 14B thinking-on for every task.
- **Settings:** `tier = "small"` or `"medium"` depending on window; context
  16k; worker thinking off. A thinking-on planner profile is reasonable if it
  fits at the same context.

### 16 GB

- **Safe:** 14B Q6, or 24B Q4. Examples: Qwen3-14B Q6, Gemma 3 12B Q6/Q8,
  Mistral Small 24B Q4. A 14B Q8 fits only at a small context.
- **Avoid:** 27B/32B Q4 on 16 GB; the weights alone leave too little for a
  useful KV cache.
- **Settings:** `tier = "medium"`, context 16k–32k; worker thinking off;
  thinking-on planner/verifier if the same model fits. This is the first tier
  where role routing is usually worth it.

### 24 GB

- **Safe:** 27–32B Q4. Examples: Qwen3-32B Q4, Gemma 3 27B Q4/Q5,
  Nanbeige4.3-33B Q4, DeepSeek-R1-Distill-Qwen-32B Q4, Mistral Small 24B Q6.
- **Avoid:** 70B at Q3; it may load, but editing quality and speed suffer.
- **Settings:** `tier = "medium"` or `"large"`, context 32k; worker thinking off
  by default, then measure on/off. Role routing for planner/verifier is useful.

### 32 GB

- **Safe:** 32B Q5/Q6, or 27B Q8 at a moderate context. Examples: Qwen3-32B
  Q5/Q6, Gemma 3 27B Q6/Q8, Nanbeige4.3-33B Q5, Mistral Small 24B Q8.
- **Possible:** 70B Q3/IQ3 at a small context; not recommended for code edits.
  Use a hosted model for 70B-class work instead.
- **Settings:** `tier = "large"`, context 32k–64k; worker thinking off default;
  measure. Use `escalate`/`fallback` only if you also have a second model or
  hosted profile.

> **Rule of thumb:** if you are unsure, choose one size smaller and increase
> context. Pedacito benefits more from a model that fits with a 16k–32k window
> than from the largest weights that barely load at 4k.

---

## Profile recipes

`vram_gb` is a hardware setting and normally belongs in `[universal]`. A profile
can override it when a model runs on a different GPU.

### Minimal worker (6–8 GB)

```toml
[universal]
vram_gb = 8

[profiles.worker]
model = "qwen3-8b"          # exact id/name your server reports
tier = "small"
disable_thinking = true
# context_tokens and token budgets are auto-fitted; set them only to override.
```

### Worker + thinking planner/verifier (12–16 GB)

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
max_tokens_answer = 8000     # leave room for reasoning + the edit
context_tokens = 32768
```

If the two profiles point at the same server and model, switching roles should
not reload weights. If they point at different models on a single GPU, expect a
model-load wait on every switch; keep roles simple.

| Role | Suggested setting |
|---|---|
| `worker` | The active model; thinking off for small models |
| `planner` | Thinking on for 12 GB+ if it fits; same model or a stronger one |
| `verifier` | Thinking on for 16 GB+; same model or a stronger one |
| `repair` | Usually the worker; thinking on can help fix syntax/format failures |
| `summarizer` | Worker or a tiny fast model; do not switch models on one GPU unless measured |
| `escalate` | Optional larger local or hosted model, one retry after no edit landed |
| `fallback` | Optional hosted/larger model when the worker errors out |

Every role is optional. With one model, leave `roles` empty; the extra routing
only adds value when a second profile is genuinely better at that job.

### Larger worker, thinking only where it helps (24–32 GB)

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

Do not route `summarizer` to a separate model on one GPU unless the extra
quality is measurable; summary calls are frequent and model swaps are expensive.

### Hosted fallback (optional)

A hosted `fallback` or `escalate` role is useful when the local worker fails,
but it changes the privacy and cost model:

```toml
[universal]
roles = { escalate = "hosted" }

[profiles.hosted]
base_url = "https://api.example.com/v1"
api_key = "sk-..."
model = "your-hosted-model"
tier = "hosted"
```

Pedacito tries the local worker first and only escalates when no edit landed.

---

## Thinking / reasoning settings

### Default recommendation by size

| VRAM / class | Worker | Planner | Verifier | Notes |
|---|---|---|---|---|
| 6 GB (3–4B) | off | off | off | thinking is usually a net loss |
| 8 GB (7–9B) | off | off | off | keep the answer budget for edits |
| 12 GB (12–14B) | off | on if 16k+ | on if 16k+ | first useful thinking split |
| 16 GB (14–24B) | off | on | on | measure worker on/off |
| 24 GB (27–32B) | off default | on | on | measure; thinking can help hard tasks |
| 32 GB (32B+) | off default | on | on | measure; use `eval` before keeping it on |

Thinking-on profiles should raise `max_tokens_answer` by roughly 2k–4k and keep
enough context for the prompt. Turning thinking on without raising the budget
mostly produces truncation and retries.

### When thinking helps

- Hard debugging, cross-file tracing, architecture, or a task with several
  interacting requirements.
- A planner that must split a broad goal into non-overlapping tasks.
- A verifier that must catch a subtle logic error.
- Models that are trained to use reasoning for code (some Qwen/GLM/Nanbeige/
  Magistral variants), **if** they still follow the edit format.

### When it hurts

- Routine lookup selection and small localized edits.
- 4k–8k windows, where reasoning competes with map and source.
- Models that emit long reasoning but weak edits; the answer budget dies before
  the search/replace block.
- Any setup where `finish_reason = "length"` or “would not produce valid JSON”
  appears regularly.

Pedacito has recovery paths for these (larger budgets, no-reasoning nudges,
salvaging a truncated action), but prevention is cheaper.

A hybrid model in non-thinking mode is not always identical to the family's
dedicated instruct model. If both are available, calibrate/eval both; the
instruct variant is often better at strict tool calling and edit formats.

### Per-family notes

These are examples; check the model card for the exact chat template.

| Family | What to expect |
|---|---|
| **Qwen3 / 3.5 / 3.8** | Hybrid thinking; `enable_thinking` and `/no_think`. Strong tool use. Good worker with thinking off; good planner with it on. |
| **Gemma 3 / 4** | Gemma 3 has no thinking; Gemma 4 reasoning is off by default and opt-in. Good instruction following; check the exact template. |
| **Nanbeige 4.x** | `enable_thinking` on by default; `<think>` output. Strong small models, but thinking is token-heavy; turn off for the worker. |
| **LiquidAI LFM2.5 / LFM3** | Very small and fast. LFM2.5-Thinking and LFM2.5-24B-A2B always reason; LFM3 supports `enable_thinking`. Prefer an Instruct variant for routine editing. |
| **DeepSeek R1 / V3.x distills** | R1-style models always think and can be slow; V3.1+ hybrids expose `thinking`. Budget accordingly, or use a non-thinking coding model for the worker. |
| **GLM 4.x / 5** | Hybrid `enable_thinking`; strong tool use. Large variants need substantial VRAM. |
| **GPT-OSS** | Harmony format, always reasons; `reasoning_effort` can lower it but not turn it off. Good tool use; 20B is the realistic local size. |
| **Mistral / Magistral** | Magistral-style models use `[THINK]`/`reasoning_effort`; often always reason. Mistral Small 4 supports an effort toggle. |
| **Granite** | 3.2/4.2 have thinking toggles; small and instruction-oriented. Check the version. |
| **Phi-4 reasoning** | `<think>`-style output, prompt-imposed; no standard off switch. Small but can be slow on long reasoning. |

Model names, quant availability, and template switches change. Confirm with your
server's model list and a quick `pedacito calibrate`.

---

## How to measure

### 1. Check the setup

```bash
pedacito config --check --profile worker
pedacito config --doctor --profile worker
```

`--check` verifies the server/model and reports the real context window.
`--doctor` probes structured-output modes, tokenizer rate, and server details.
If the server reports a stale window, set `context_tokens` explicitly.

### 2. Calibrate the model

```bash
pedacito calibrate --profile worker
```

This measures tool choice, JSON adherence, edit format, instruction following,
honesty, long-context retrieval, and native-vs-grammar tool calling. The saved
model card can set:

- `prompt_pack`
- `answer_mode`
- `verify_answer`
- `best_of`
- `tool_mode`

It does **not** decide thinking on/off; measure that separately. Re-run
calibration after changing the model, quantization, or chat template.

### 3. Run the eval, then A/B it

```bash
pedacito eval --profile worker --out worker-off.json
pedacito eval --profile worker-think --out worker-on.json \
  --baseline worker-off.json
```

If the thinking-on profile regresses, the second command exits nonzero and
prints the regressions:

- pass rate, locate rate, first-apply-clean rate
- followed prompt injections
- prompt/completion tokens, calls, wall time
- new failure types

Tune the gate with `--max-token-growth` and `--max-time-growth` if your hardware
or model has a known cost profile.

Pass rate alone is not enough. Prefer a setup that also improves or preserves:

- **first-apply-clean** (fewer repair calls)
- **mean prompt/completion tokens**
- **cache-hit ratio**
- **wall time**
- a lower rate of truncation/format failures

### 4. Test the real work

The eval uses small fixture repos. Before trusting a new model or profile:

```bash
pedacito ask PROJ -t "a real narrow task"          # dry run
pedacito ask PROJ -t "..." --review                # inspect the diff
pedacito ask PROJ -t "..." --apply                 # once confident
```

Use `--record` on a run and `--replay` later; replay is for testing harness
changes without a model, not for judging a model upgrade. Add `--strict-replay`
in CI to fail when the harness asks a different prompt than the recording.

### 5. Watch VRAM

Use your GPU tool (`nvidia-smi`, `rocm-smi`, the server's own panel) while a
task runs. If the model spills to system RAM, generation slows sharply and the
run may look like a model-quality problem when it is a sizing problem. Lower
`context_tokens`, use a smaller quant/model, or quantize the KV cache.

---

## Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| “kept running past the token limit” / no JSON | thinking on with a small answer budget | `disable_thinking = true` for the worker, or raise `max_tokens_answer` |
| Empty streamed answer with hidden reasoning | thinking runaway | Pedacito retries with thinking off; set it off by default for small models |
| “search block not found” | model copied line numbers, or the file shifted | narrow the task; Pedacito re-anchors; check the model’s edit-format calibration |
| Answers vague / wrong file | map too compressed, or model weak at tool choice | raise `flat_map_token_budget`/context, or use a better tool-calling model |
| Model ignores tools | grammar/native mismatch | run `pedacito calibrate` and compare `tool_mode = "grammar"` vs `"native"` |
| Slow after a few minutes | model spilled out of VRAM or context is too large | lower context/quant; check GPU memory while running |
| Role switches take a minute | different models on one GPU reload weights | use the same model for roles, or accept the load cost |
| Good chat, bad edits | model is not edit-oriented | prefer a coding/instruct model; check first-apply-clean in eval |

---

## Quick checklist

- [ ] Set `vram_gb` to the memory actually available to the server.
- [ ] Start with Q4_K_M; leave 1.5–2.5 GB for KV cache and overhead.
- [ ] On 6–8 GB, use a 3–9B instruct/non-thinking worker and keep thinking off.
- [ ] On 12–16 GB, keep the worker off and try thinking only for
  `planner`/`verifier`.
- [ ] On 24–32 GB, measure thinking on/off before making it the default.
- [ ] Run `pedacito calibrate` once per model/quant.
- [ ] Run `pedacito eval` before and after a change; use `--baseline` as a gate.
- [ ] Prefer first-apply-clean and token/time numbers over pass rate alone.
- [ ] Keep role routing simple on a single GPU; every model switch can reload.
- [ ] Do the first dozen real tasks without `--apply`, then use `--review`.
