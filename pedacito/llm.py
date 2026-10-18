"""OpenAI-compatible client: chat completions, JSON-schema tool calls, and
model warmup.

Usage
-----
    from .llm import LMStudio

    client = LMStudio(cfg)
    client.check()                       # verify the server and model
    text = client.chat(messages, max_tokens=500)
    step = client.chat_json(messages, schema, max_tokens=500)

Works against any server exposing an OpenAI-compatible /v1 surface --
tested against LM Studio and Ollama. The class is named LMStudio for
historical reasons; it isn't LM-Studio-specific. This module uses two
features of that surface:

  * plain chat completions (for summaries and the final answer)
  * response_format={"type": "json_schema", ...} for the tool loop, which is
    grammar-constrained under the hood, so the model *cannot* emit unparseable
    JSON. This matters far more with a small local model than with a frontier
    hosted one.

On top of the raw HTTP calls, this module handles the failure modes specific
to running a model locally: templates that reject certain message shapes,
responses truncated at the token limit, reasoning models that wrap their
answer in a <think> block, and a model that is still loading.

Only dependency is `requests`.
"""

from __future__ import annotations

import json
import re
import sys
import time

import requests


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class LLMError(RuntimeError):
    """Base class for any failure talking to LM Studio."""


class ModelLoadError(LLMError):
    """LM Studio could not load the model. Retrying rarely helps."""


class TemplateError(LLMError):
    """The model's chat template rejected the message sequence."""


class ContextOverflowError(LLMError):
    """The server refused the request because the prompt is bigger than its
    configured context window. Recoverable: `chat()` trims the conversation
    and retries before giving up."""


class _Transient(LLMError):
    """Worth retrying: model still loading, or the server is busy."""


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------
# Chat templates differ in what they accept. Mistral-family templates raise a
# Jinja exception on two user messages in a row; Gemma has no system role at
# all. Neither is a server fault, so the fix is to reshape the request.
_TEMPLATE_HINTS = (
    "jinja", "roles must alternate", "conversation roles",
    "system role not supported", "only user and assistant roles",
)

# LM Studio returns 400 for "still loading" and for "failed to load" alike, so
# the status code alone can't distinguish transient from fatal.
_LOADING = ("is loading", "loading model", "model is not loaded", "please wait")
_LOAD_FAILED = ("failed to load model", "error loading model", "no model loaded")

# Shown on a plain connection failure. Covers both servers Pedacito is tested
# against, since the fix differs: LM Studio needs a setting flipped, Ollama
# needs an environment variable set before it starts.
CONNECT_HELP = """\
  Cannot reach the server at {url}.
  - Is a model server actually running there?
  - LM Studio: enable "Serve on Local Network" in the Developer tab, or it
    only listens on 127.0.0.1 and a remote machine gets refused.
  - Ollama: it binds to 127.0.0.1 by default. Set OLLAMA_HOST=0.0.0.0 (or the
    machine's address) in its environment before starting it, then restart.
  - If you're on a different machine from the server, check the URL and port
    in your config match where it's actually listening (LM Studio default
    port 1234, Ollama default port 11434)."""

LOAD_HELP = """\
  The server found the model but could not load it. Usual causes, in order:
  1. Not enough free VRAM. Another model is probably still resident -- open
     LM Studio and eject it, or lower this model's context length.
  2. The context length is set higher than the card can hold. Try 16384 with
     K/V cache at Q8_0 and flash attention on.
  3. An incomplete or corrupt download. Re-download it in LM Studio.
  4. A runtime mismatch. Try switching between ROCm and Vulkan for this model.

  Fastest way to tell: load the model by hand in the LM Studio UI. Whatever
  error it shows there is the real one -- the API only reports that it failed."""


# --------------------------------------------------------------------------
# Reasoning-model output handling
# --------------------------------------------------------------------------
# Reasoning models (Qwen3, DeepSeek-R1 distills, and Qwen3-Coder) emit a
# thinking block before their real answer. Some runtimes strip it into a
# separate field, some leave it inline in `content`. Handle both: an unstripped
# <think> block breaks JSON parsing in the gather loop, and would land verbatim
# in the answer text and the session log.
_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
# An unterminated block: the model ran out of tokens mid-thought.
_THINK_OPEN_RE = re.compile(r"^\s*<(think|thinking|reasoning)>.*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove any <think>/<thinking>/<reasoning> block from model output. If
    the text is nothing but an unterminated opening tag (truncated mid-thought),
    returns an empty string rather than the thinking fragment."""
    if not text:
        return ""
    out = _THINK_RE.sub("", text)
    if _THINK_OPEN_RE.match(out):
        # Nothing after the opening tag survived; there is no answer here.
        return ""
    return out.strip()


def extract_content(message: dict) -> str:
    """Pull the answer text out of a response message, whichever field the
    runtime put it in (content, or a separate reasoning field), with any
    thinking block stripped."""
    content = message.get("content") or ""
    if not content.strip():
        # Some runtimes route everything into a reasoning field and leave
        # content empty. Better a thinking-tinged answer than none.
        content = message.get("reasoning_content") or message.get("reasoning") or ""
    return strip_reasoning(content)


# --------------------------------------------------------------------------
# Truncated-JSON repair
# --------------------------------------------------------------------------
def repair_truncated_json(text: str) -> str | None:
    """Salvage the complete top-level keys of a JSON object that was cut off
    mid-generation.

    Everything from the first incomplete token onward -- an unterminated
    string, a bare key, a dangling colon or comma, even a number typed only
    halfway -- is dropped, and the surviving pairs are closed into an object.
    Nothing partial is ever completed or guessed at: `"search_lib"` could grow
    into any of several file names, and acting on the fragment would silently
    do the wrong thing, so the whole key goes instead.

    Returns the repaired JSON text, or None if `text` was not actually
    truncated or nothing salvageable survived.

    This is why the step schema (see tools.py) puts `action` and its
    arguments before `reasoning`: a reply truncated inside the trailing
    explanation still has everything the agent needs, so recovering the
    prefix yields a usable step instead of costing another round trip.
    """
    start = text.find("{")
    if start == -1:
        return None
    buf = text[start:]
    n = len(buf)

    def _complete_value_end(i: int) -> int | None:
        """If a value begins at index i and is fully present, return the
        index just after it; None if it is missing or cut off."""
        while i < n and buf[i] in " \t\r\n":
            i += 1
        if i >= n:
            return None
        ch = buf[i]
        if ch == '"':
            j = i + 1
            while j < n:
                if buf[j] == "\\":
                    j += 2
                    continue
                if buf[j] == '"':
                    return j + 1
                j += 1
            return None                      # string never closed
        if ch in "{[":
            depth, k, instr, esc = 0, i, False, False
            while k < n:
                c = buf[k]
                if esc:
                    esc = False
                elif instr:
                    if c == "\\":
                        esc = True
                    elif c == '"':
                        instr = False
                elif c == '"':
                    instr = True
                elif c in "{[":
                    depth += 1
                elif c in "}]":
                    depth -= 1
                    if depth == 0:
                        return k + 1         # nested structure survived whole
                    if depth < 0:
                        return None
                k += 1
            return None                      # container left open
        m = re.match(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", buf[i:])
        if m:
            tok = m.group(0)
            j = i + len(tok)
            # A digit followed straight by '.', 'e', or an exponent sign was
            # cut mid-token ("1.5" truncated to "1", "1e3" to "1"). A trailing
            # '.' or '-' can never legally follow a complete number.
            if j < n and (buf[j] == "." or (tok[-1].isdigit() and buf[j] in "eE+-")):
                return None
            # At the very end of the buffer a bare integer could still be
            # unfinished; anything terminated by a delimiter is fine.
            if j == n and tok[-1].isdigit():
                return None
            return j
        if buf.startswith(("true", "false", "null"), i):
            return i + 4
        return None

    # Walk the object's members, keeping only key:value pairs that arrived
    # complete. Stops at the first fragment, at the object's own close, or at
    # the end of the buffer.
    kept: list[str] = []
    i = 1                                      # past the opening brace
    while True:
        while i < n and buf[i] in " \t\r\n,":
            i += 1
        if i >= n or buf[i] == "}":
            break
        if buf[i] != '"':
            break                              # malformed; salvage what we have
        kend = _complete_value_end(i)
        if kend is None:
            break                              # dangling key
        j = kend
        while j < n and buf[j] in " \t\r\n":
            j += 1
        if j >= n or buf[j] != ":":
            break                              # key with no colon yet
        vend = _complete_value_end(j + 1)
        if vend is None:
            break                              # value missing or cut off
        kept.append(buf[i:vend])
        i = vend
    if not kept:
        return None
    return "{" + ",".join(kept) + "}"


# --------------------------------------------------------------------------
# Message-shape normalisation
# --------------------------------------------------------------------------
def merge_consecutive(messages: list[dict]) -> list[dict]:
    """Collapse adjacent same-role messages into one.

    Belt and braces: the agent is written to alternate, but a single stray
    append would otherwise surface as an opaque Jinja traceback from the server.
    """
    out: list[dict] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"],
                       "content": f"{out[-1]['content']}\n\n{m['content']}"}
        else:
            out.append(dict(m))
    return out


def fold_system(messages: list[dict]) -> list[dict]:
    """Move system content into the first user turn, for templates without a
    system role (Gemma being the common one)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if not system:
        return rest
    if rest and rest[0]["role"] == "user":
        rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
    else:
        rest.insert(0, {"role": "user", "content": system})
    return merge_consecutive(rest)


def estimate_tokens(text: str) -> int:
    """Rough token count estimate. Crude but adequate: ~3.6 chars/token for
    source code."""
    return int(len(text) / 3.6) + 1


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------
class LMStudio:
    """A thin client for one LM Studio server + model combination, handling
    retries, template quirks, and reasoning-model output transparently.

    Sampler parameters supplied via `cfg.sampler` are merged into every
    request body verbatim, so a profile targeting LM Studio can pass top_k,
    min_p, repeat_penalty, seed, and anything else llama.cpp understands --
    none of which have a first-class config field, because the accepted set
    differs per server.
    """

    def __init__(self, cfg):
        """Initialise the client from a Config: connection details and the
        per-instance state used to remember quirks discovered at runtime."""
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        }
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        # Set once we learn this model's template has no system role, so we
        # don't burn a failed request rediscovering it on every call.
        self._no_system_role = False
        # "length" here means the model was cut off mid-answer, which for a
        # grammar-constrained call is the difference between "bad JSON" and
        # "valid JSON we never let it finish".
        self.last_finish_reason = ""

    # ------------------------------------------------------------------ core
    def _context_too_long(self, body: str) -> bool:
        """True when a 400 body says the PROMPT overflowed the server's
        context window -- distinct from "max_tokens too large", which is a
        request-shape problem. llama.cpp/LM Studio phrase it as
        "prompt ... exceeds ... context" / "n_ctx"; Ollama says
        "context length exceeded" or "too many tokens"."""
        low = (body or "").lower()
        return any(h in low for h in (
            "exceeds", "context length", "n_ctx", "ctx_len",
            "too many tokens", "prompt is too long", "maximum context",
            "out of range", "overfull"))

    def _post(self, payload: dict, stream: bool = False):
        """Send one HTTP request to the chat completions endpoint, raising a
        specific LLMError subclass based on the failure (connection refused,
        model failed to load, template rejected the request, transient
        loading state, or an unclassified server error)."""
        try:
            r = requests.post(
                self.url, headers=self.headers, json=payload,
                timeout=self.cfg.timeout, stream=stream,
            )
        except requests.exceptions.ConnectionError as e:
            raise LLMError(CONNECT_HELP.format(url=self.cfg.base_url)) from e
        if r.status_code >= 400:
            body = r.text[:800]
            low = body.lower()
            if any(k in low for k in _LOAD_FAILED):
                raise ModelLoadError(
                    f"The server could not load '{self.cfg.model}'.\n{LOAD_HELP}")
            if any(k in low for k in _TEMPLATE_HINTS):
                raise TemplateError(body[:400])
            if r.status_code in (429, 503) or any(k in low for k in _LOADING):
                raise _Transient(f"{r.status_code}: {body[:200]}")
            if self._context_too_long(body):
                # The PROMPT overflowed the server's window. chat() catches
                # this, trims the conversation, and retries -- a mid-task
                # death here is what "kept hitting limits" actually feels
                # like to a user on stock LM Studio settings.
                raise ContextOverflowError(
                    f"Server says the request exceeds its context window: "
                    f"{body[:300]}")
            raise LLMError(f"Server returned {r.status_code}: {body[:500]}")
        return r

    def chat(self, messages: list[dict], max_tokens: int, schema: dict | None = None,
             stream: bool = False) -> str:
        """Send a chat completion request and return the response text.

        Handles retries for transient failures (model still loading) and
        automatic recovery from template incompatibilities (folding the
        system prompt into the first user turn if the template rejects a
        system role). A model that is still loading answers 400 or 503 for a
        while; without a retry loop, the first call of a batch would fail and
        every subsequent one would fail the same way.
        """
        messages = merge_consecutive(messages)
        if self._no_system_role:
            messages = fold_system(messages)
        delay = 2.0
        started = time.time()
        announced = False
        attempt = 0
        trimmed_once = False
        while True:
            attempt += 1
            try:
                return self._chat_once(messages, max_tokens, schema, stream)
            except ContextOverflowError as e:
                # The server's window is smaller than our budgets assumed
                # (stock LM Studio 8192 / Ollama 4096 vs a hosted-tier
                # profile, or simply a long gather). Salvage the call by
                # dropping the oldest RESULT turns -- the newest tool output
                # and the task itself are what the model needs to answer;
                # stale middle history is the cheapest thing to lose. Two
                # rounds of halving, then fail with a real diagnosis.
                if not trimmed_once:
                    body_idx = [i for i, m in enumerate(messages)
                                if m["role"] == "user"
                                and m["content"].startswith("RESULT:\n")]
                    if len(body_idx) > 2:
                        drop = set(body_idx[: len(body_idx) // 2])
                        messages = [m for i, m in enumerate(messages)
                                    if i not in drop]
                        trimmed_once = True
                        print("[server context overflow: dropped older tool "
                              "results and retrying]", file=sys.stderr)
                        continue
                if any(m["role"] == "system" for m in messages) and \
                        len([m for m in messages if m["role"] != "system"]) > 2:
                    # No RESULT turns to shed (answer phase): shrink by
                    # replacing middle user turns with a stub, keeping first
                    # (map+task) and last (latest instruction/result).
                    keep_first = next(i for i, m in enumerate(messages)
                                      if m["role"] == "user")
                    keep_last = max(i for i, m in enumerate(messages)
                                    if m["role"] == "user")
                    if keep_last - keep_first > 1:
                        mid = [i for i in range(keep_first + 1, keep_last)
                               if messages[i]["role"] == "assistant"]
                        if mid:
                            messages = [m for i, m in enumerate(messages)
                                        if i not in set(mid)]
                            print("[server context overflow: shortened the "
                                  "conversation and retrying]", file=sys.stderr)
                            continue
                raise LLMError(
                    f"{e}\n\n"
                    "Pedacito already trimmed the conversation once and the "
                    "server still refused.\nThis model's configured context "
                    "window is too small for even a minimal run.\n"
                    "  * LM Studio: raise 'Context length' in the model's "
                    "runtime settings (8192 is\n"
                    "    the stock default; 16384-32768 suits a 27B on most "
                    "GPUs) and re-run.\n"
                    "  * Ollama: set num_ctx (e.g. OLLAMA_NUM_CTX=16384, or "
                    "options.num_ctx in a\n"
                    "    Modelfile). Pedacito reads it automatically.\n"
                    "  * Or lower context_tokens/max_gathered_chars in your "
                    "profile so auto-fit\n"
                    "    plans smaller prompts from the start.") from e
            except TemplateError as e:
                # Most likely this model has no system role. Try again with the
                # system prompt folded into the first user turn, and remember it.
                if any(m["role"] == "system" for m in messages):
                    if not self._no_system_role:
                        print(f"['{self.cfg.model}' has no system role; folding "
                              "it into the first user message]", file=sys.stderr)
                    self._no_system_role = True
                    messages = fold_system(messages)
                    continue
                raise LLMError(
                    f"The chat template for '{self.cfg.model}' rejected the "
                    f"request:\n{e}\n\n"
                    "This is a model-template incompatibility, not a Pedacito bug "
                    "in itself.\nTry a different quant or a different model; "
                    "Gemma and Qwen templates are\nmore permissive than "
                    "Mistral's.") from e
            except _Transient as e:
                waited = time.time() - started
                if waited >= self.cfg.load_wait_seconds:
                    raise LLMError(
                        f"LM Studio was still not ready for '{self.cfg.model}' "
                        f"after {waited:.0f}s.\n"
                        "  If you are switching models, LM Studio may be waiting "
                        "for the previous one\n"
                        "  to unload. Check its Developer tab, or raise "
                        "load_wait_seconds in your config.\n"
                        f"  Last response: {e}") from e
                if not announced:
                    print(f"[waiting for '{self.cfg.model}' to load -- switching "
                          f"models can take a minute or two]", file=sys.stderr)
                    announced = True
                elif attempt % 5 == 0:
                    print(f"[still waiting, {waited:.0f}s elapsed]", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 1.6, 10.0)

    def _chat_once(self, messages: list[dict], max_tokens: int,
                   schema: dict | None = None, stream: bool = False) -> str:
        """Send exactly one HTTP request (no retry logic) and return the
        response text, either buffered or streamed to stdout."""
        msgs = [dict(m) for m in messages]
        if self.cfg.thinking_suffix and msgs:
            msgs[-1]["content"] += "\n\n" + self.cfg.thinking_suffix
        if schema is not None and self.cfg.json_mode == "object" and msgs:
            # Hosted OpenAI-compatible APIs require the word "json" to appear
            # in the prompt before they'll honour json_object mode, and naming
            # the expected fields in that same sentence measurably improves
            # compliance on the schema-required keys.
            fields = ", ".join(schema.get("properties", {}).keys())
            msgs[-1]["content"] += (
                f"\n\nReply with one JSON object only, no prose. "
                f"Fields: {fields}.")

        payload = {
            "model": self.cfg.model,
            "messages": msgs,
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        # Honour request_max_tokens_cap: llama.cpp-based servers (LM Studio)
        # 400 when max_tokens exceeds the configured context window, and
        # auto_fit lowers the cap to fit -- but a caller asking for more than
        # the cap (an answer budget on a big window, an escalated truncation
        # retry) must never smuggle a too-large value through. 0 disables
        # clamping entirely for servers that prefer to decide themselves.
        cap = getattr(self.cfg, "request_max_tokens_cap", 0) or 0
        if cap and max_tokens > cap:
            max_tokens = cap
            payload["max_tokens"] = cap
        # User-supplied sampler parameters, merged verbatim. Anything the
        # server accepts can go here (top_p, top_k, min_p, repeat_penalty,
        # seed, stop, ...); anything it doesn't will 400 the request, so
        # this belongs in the profile for the specific server it targets.
        # The structural keys below are skipped: overriding them would
        # produce a request Pedacito's own parsing cannot handle.
        for k, v in (self.cfg.sampler or {}).items():
            if k in ("model", "messages", "stream"):
                continue
            payload[k] = v
        # chat_template_kwargs is a llama.cpp/LM Studio convention. Hosted
        # APIs don't understand it, and a strict one could 400 on it, so it's
        # only sent when we're clearly talking to a local server (i.e. not
        # json_mode = "object").
        if self.cfg.disable_thinking and self.cfg.json_mode == "schema":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if schema is not None:
            if self.cfg.json_mode == "object":
                payload["response_format"] = {"type": "json_object"}
            else:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "step", "strict": True, "schema": schema},
                }

        t0 = time.time()
        self.stats["calls"] += 1

        if not stream:
            r = self._post(payload)
            data = r.json()
            try:
                self.last_finish_reason = data["choices"][0].get("finish_reason") or ""
            except (KeyError, IndexError):
                self.last_finish_reason = ""
            usage = data.get("usage") or {}
            self.stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.stats["completion_tokens"] += usage.get("completion_tokens", 0)
            self.stats["seconds"] += time.time() - t0
            try:
                return extract_content(data["choices"][0]["message"])
            except (KeyError, IndexError) as e:
                raise LLMError(f"Unexpected response shape: {json.dumps(data)[:500]}") from e

        # Streaming: worth it at ~25 tok/s so the user sees progress.
        out = []
        r = self._post(payload, stream=True)
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            # requests only decodes when the response declares a charset, and
            # LM Studio's text/event-stream often doesn't. Handle both.
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            if not raw.startswith("data: "):
                continue
            body = raw[6:]
            if body.strip() == "[DONE]":
                break
            try:
                d = json.loads(body)["choices"][0]["delta"]
            except Exception:
                continue
            delta = d.get("content")
            if not delta and not d.get("reasoning_content"):
                continue
            if delta:
                out.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()
        sys.stdout.write("\n")
        self.stats["seconds"] += time.time() - t0
        # Reasoning models vary in where the thinking block lands. Some put
        # it inline in `content` (so it's in `out` and needs stripping);
        # others route it to a separate `reasoning_content` field (which the
        # streaming loop above skips, since it only appends `content`).
        # Either way, the returned value never contains a <think> block, so
        # it's safe to hand straight to edit parsing.
        return strip_reasoning("".join(out))

    # --------------------------------------------------------------- helpers
    def chat_json(self, messages: list[dict], schema: dict, max_tokens: int,
                  retries: int = 2, cap: int | None = None) -> dict:
        """Send a grammar-constrained chat request and return the parsed
        JSON object.

        The grammar guarantees well-formed JSON *if the model is allowed to
        finish*. The common failure is not malformed output, it is a response
        cut off at max_tokens mid-string. So a truncated attempt retries with
        a larger budget rather than the same one, and says so.
        """
        last = ""
        budget = max_tokens
        # Ceiling for truncation-retry escalation. Default is NOT a hard
        # 2000 -- it is derived from the caller's own config (reserve /
        # answer budgets), so raising max_tokens_step or reserve_tokens in
        # the profile actually raises this limit too.
        if cap is None:
            # Never escalate past what the model's context window can hold:
            # a request bigger than (context - reserve) gets clamped by the
            # server anyway, and a truncated retry at that size just wastes
            # round trips. The ceiling is the smaller of the generous default
            # and the caller's own usable window; it always clears max_tokens.
            usable = self.cfg.context_tokens - self.cfg.reserve_tokens
            cap = min(max(2048, self.cfg.reserve_tokens,
                          self.cfg.max_tokens_answer, max_tokens * 4),
                      max(usable, max_tokens))
        cap = max(cap, max_tokens)
        for attempt in range(retries + 1):
            raw = self.chat(messages, max_tokens=budget, schema=schema)
            last = raw
            truncated = self.last_finish_reason == "length"
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.split("\n", 1)[-1] if "\n" in text else text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Salvage the outermost object if the model added prose.
                i, j = text.find("{"), text.rfind("}")
                if i != -1 and j > i:
                    try:
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        pass
                if truncated:
                    patched = repair_truncated_json(text)
                    if patched:
                        try:
                            obj = json.loads(patched)
                        except json.JSONDecodeError:
                            obj = None
                        # Only trust it if the part that matters survived.
                        if isinstance(obj, dict) and obj.get("action"):
                            print("[reply was cut off; recovered the action from "
                                  "the partial reply]", file=sys.stderr)
                            return obj
                if truncated:
                    nxt = min(budget * 2, cap)
                    if nxt > budget:
                        budget = nxt
                        print(f"[reply was cut off at the token limit; retrying "
                              f"with max_tokens={budget}] ", file=sys.stderr)
                    else:
                        print(f"[reply was cut off at the token limit and "
                              f"max_tokens is already at the ceiling "
                              f"({budget}); raising it further in your config]",
                              file=sys.stderr)
                    continue
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "That was not valid JSON. Reply with "
                                                "the JSON object only, and keep "
                                                "'reasoning' under 15 words."},
                ]
        # If the LAST attempt was cut off at the token limit, that -- not a
        # grammar failure -- is what went wrong (common with thinking models
        # that spend the whole budget on reasoning before emitting JSON).
        # Raise the informative error instead of the generic one below.
        if truncated:
            last = raw
        if self.last_finish_reason == "length":
            raise LLMError(
                f"The model kept running past the token limit before finishing its "
                f"JSON (tried up to max_tokens={budget}, ceiling {cap}).\n"
                "  Either it is writing an essay in the 'reasoning' field, or a\n"
                "  'content'/'replacement' block is longer than one step budget.\n"
                "  Raise max_tokens_step (and/or reserve_tokens) in your active\n"
                "  profile -- the ceiling follows those settings, it is not fixed.\n"
                f"  Last partial reply: {last[-160:]}")
        raise LLMError(
            f"Model would not produce valid JSON after {retries + 1} tries.\n"
            "  If this is a hosted OpenAI-compatible API (DeepSeek, OpenAI, "
            "...), set\n"
            "  json_mode = \"object\" under [server] -- hosted APIs reject "
            "json_schema.\n"
            "  If it's a local server, the runtime may be ignoring the JSON "
            "schema\n"
            "  (common with MLX builds; llama.cpp/GGUF supports it).\n"
            f"  Last reply: {last[:300]}")

    def loaded_models(self) -> dict[str, str] | None:
        """Query LM Studio's native REST API for {model_id: state}.

        The OpenAI-compatible /v1/models lists everything *downloaded* when
        JIT loading is on, so it cannot tell you what is resident.
        /api/v0/models reports a state field instead. Returns None on any
        older LM Studio build that lacks this endpoint.
        """
        base = self.cfg.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        try:
            r = requests.get(base + "/api/v0/models", headers=self.headers, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json().get("data", [])
        except Exception:
            return None
        out = {m.get("id", "?"): m.get("state", "unknown") for m in data if isinstance(m, dict)}
        return out or None

    def detect_context(self) -> int | None:
        """Probe the server for the REAL context length configured for
        `cfg.model`, so token budgets can be auto-fitted to what the model
        will actually accept instead of a guessed config number.

        Works against the two local servers Pedacito targets:
          * LM Studio -- native /api/v0/models carries `context_length`
            (the value typed into its UI; llama.cpp rejects requests bigger
            than this with a 400, which is half of "kept hitting limits").
          * Ollama -- POST /api/show returns parameters including num_ctx
            (defaults to 4096 when unset, which explains why Ollama users
            hit limits that a 32k-token model 'should' handle).

        Returns None for hosted APIs and any server that doesn't answer --
        callers then keep whatever context_tokens the config declares.
        Never raises: this is an optimisation probe, not a requirement.
        """
        base = self.cfg.base_url.rstrip("/")
        root = base[:-3] if base.endswith("/v1") else base
        # LM Studio native API.
        try:
            r = requests.get(root + "/api/v0/models", headers=self.headers,
                             timeout=5)
            if r.status_code == 200:
                for m in r.json().get("data", []):
                    if not isinstance(m, dict):
                        continue
                    if m.get("id") == self.cfg.model or m.get("key") == self.cfg.model:
                        ctx = (m.get("context_length")
                               or (m.get("metadata") or {}).get("context_length"))
                        if isinstance(ctx, int) and ctx > 0:
                            return ctx
        except Exception:
            pass
        # Ollama native API.
        try:
            r = requests.post(root + "/api/show",
                              json={"name": self.cfg.model}, timeout=5)
            if r.status_code == 200:
                d = r.json()
                ctx = ((d.get("model_info") or {}).get(
                    "llama.context_length")
                    or d.get("parameters", {}).get("num_ctx"))
                if isinstance(ctx, int) and ctx > 0:
                    return ctx
        except Exception:
            pass
        return None

    def warmup(self, verbose: bool = True) -> None:
        """Force the model to load with one tiny generation before real work
        begins.

        Two jobs: it surfaces a load failure once, up front, with a real
        diagnosis rather than as repeated identical warnings mid-index; and
        when a profile switch changed the requested model, it absorbs the
        load wait here with a clear message, rather than stalling silently
        inside the gather loop.
        """
        state = self.loaded_models()
        if verbose and state is not None:
            resident = [m for m, st in state.items() if st == "loaded"]
            if resident and self.cfg.model not in resident:
                print(f"[server currently has {', '.join(resident)} loaded; "
                      f"requesting '{self.cfg.model}']", file=sys.stderr)
        t0 = time.time()
        self.chat([{"role": "user", "content": "Reply with the single word: ok"}],
                  max_tokens=5)
        took = time.time() - t0
        if verbose and took > 5:
            print(f"[model ready after {took:.0f}s]", file=sys.stderr)

    def check(self, warn: bool = True) -> str:
        """Verify the server endpoint is reachable and return the
        comma-joined list of model ids/names it reports. If `warn` is set
        and the configured model isn't in that list, prints a warning."""
        url = self.cfg.base_url.rstrip("/") + "/models"
        try:
            r = requests.get(url, headers=self.headers, timeout=15)
            r.raise_for_status()
        except Exception as e:
            raise LLMError(
                CONNECT_HELP.format(url=self.cfg.base_url) + "\n"
                "  - Set base_url in your config: `pedacito config --init`, then\n"
                "    `pedacito config --check` to test it."
            ) from e
        ids = [m.get("id", "?") for m in r.json().get("data", [])]
        if warn and self.cfg.model not in ids:
            print(f"[warn] model '{self.cfg.model}' not in loaded list: {ids}",
                  file=sys.stderr)
        return ", ".join(ids)


def probe_server(base_url: str, timeout: float = 2.0) -> list[str] | None:
    """Ask an arbitrary OpenAI-compatible endpoint which models it has.

    Returns the sorted list of model ids, or None when nothing answers at
    `<base_url>/models` within `timeout`. Unlike LMStudio.check() this never
    raises -- callers (the interactive setup wizard, the GUI's server hint)
    treat "no answer" as a normal outcome, not an error."""
    try:
        r = requests.get(base_url.rstrip("/") + "/models", timeout=timeout)
        r.raise_for_status()
        return sorted({m.get("id", "?") for m in r.json().get("data", [])})
    except Exception:
        return None