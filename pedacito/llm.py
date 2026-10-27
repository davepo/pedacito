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

import base64
import json
import re
import sys
import time
from pathlib import Path

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


_GIB = 1024 ** 3


def _server_root(base_url: str) -> str:
    """Strip an OpenAI-style API suffix from base_url, leaving the server root.

    Both `/v1` and `/api/v1` are common (Lemonade accepts either), and the
    native endpoints live at the root (`/api/v0/models`, `/api/show`, ...).
    """
    base = base_url.rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base


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
            # Value is cut off mid-generation. If it is an *unterminated
            # string* (the common DeepSeek failure: reply hits max_tokens
            # inside a 'content' block), keep the raw fragment so callers can
            # salvage partial content instead of discarding the step. The
            # repair below closes the quote and the object.
            k = j + 1
            while k < n and buf[k] in " \t\r\n":
                k += 1
            if k < n and buf[k] == '"' and _complete_value_end(k) is None:
                # The fragment starts with the opening quote, which we keep;
                # escape only what came AFTER it (the original `buf[k]` itself
                # must not be escaped or the repaired JSON has a stray \").
                frag = buf[k + 1:]
                frag = (frag.replace("\\", "\\\\")
                            .replace('"', '\\"').replace("\n", "\\n")
                            .replace("\r", "\\r").replace("\t", "\\t"))
                kept.append(buf[i:k] + '"' + frag + '"')
                break                          # nothing after the cut survives
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
        # Tri-state for grammar-constrained JSON support: None = not probed
        # yet (assume supported, the historical default), True = server
        # advertises response_format/json_schema, False = it answered our
        # probe WITHOUT advertising it (Ollama, plain OpenAI-compatible
        # endpoints) -- those requests fall back to JSON prompting instead
        # of 400ing or silently ignoring the schema.
        self._schema_supported: bool | None = None
        # "length" here means the model was cut off mid-answer, which for a
        # grammar-constrained call is the difference between "bad JSON" and
        # "valid JSON we never let it finish".
        self.last_finish_reason = ""
        # Truncation signal for STREAMED (free-text, answer-phase) replies.
        # DeepSeek and other OpenAI-compatible servers put finish_reason on
        # the FINAL SSE chunk, which the delta loop never inspected -- so
        # last_finish_reason stayed "" after every stream and apply()'s
        # continuation loop ("reply hit the output limit; emit the rest")
        # could never trigger. That is why a big edit run silently lost its
        # later files instead of continuing across turns. The streaming path
        # sets this directly; chat() clears it at the start of each call.
        self.stream_truncated = False

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

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text via `<base_url>/embeddings`.

        Only used by optional semantic search (see semantic.py). Kept on the
        same client so it inherits the configured base_url, API key, and
        timeout; it does not participate in chat retry/template handling.
        Accepts a plain string for convenience and always returns vectors in
        input order (the API may return them out of order)."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        url = self.cfg.base_url.rstrip("/") + "/embeddings"
        model = getattr(self.cfg, "embedding_model", "") or self.cfg.model
        try:
            r = requests.post(url, headers=self.headers,
                              json={"model": model, "input": list(texts)},
                              timeout=self.cfg.timeout)
        except requests.exceptions.ConnectionError as e:
            raise LLMError(CONNECT_HELP.format(url=self.cfg.base_url)) from e
        if r.status_code >= 400:
            raise LLMError(
                f"Embeddings request failed ({r.status_code}): {r.text[:300]}. "
                f"Set embedding_model to a model your server can load, or turn "
                f"semantic_search off.")
        try:
            data = r.json()["data"]
            if isinstance(data, dict):        # single-input server shape
                data = [data]
            data = sorted(data, key=lambda d: d.get("index", 0))
            self.stats["calls"] += 1
            return [[float(x) for x in d["embedding"]] for d in data]
        except Exception as e:                  # pragma: no cover - server shape
            raise LLMError(
                f"Unexpected embeddings response: {r.text[:300]}") from e

    def vision(self, prompt: str, image_path, model: str | None = None,
               max_tokens: int = 400) -> str:
        """Ask a multimodal model to describe one local image.

        Sent as an OpenAI-compatible image_url data URI, which LM Studio and
        Ollama's OpenAI endpoints accept for vision models. Kept separate from
        chat() because it is only used by the opt-in screenshot action; a
        text-only model will reject it, which the caller reports as an
        observation rather than failing the task."""
        path = Path(image_path)
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp"}.get(path.suffix.lower(), "image/png")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "model": model or getattr(self.cfg, "vision_model", "")
            or self.cfg.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            "max_tokens": max_tokens,
            "temperature": getattr(self.cfg, "temperature", 0.2),
        }
        r = self._post(payload)
        try:
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            self.stats["calls"] += 1
            return strip_reasoning(content).strip()
        except Exception as e:                  # pragma: no cover - server shape
            raise LLMError(
                f"Unexpected vision response: {r.text[:300]}") from e

    def chat(self, messages: list[dict], max_tokens: int, schema: dict | None = None,
             stream: bool = False, disable_thinking: bool | None = None) -> str:
        """Send a chat completion request and return the response text.

        Handles retries for transient failures (model still loading),
        proactive conversation trimming so llama.cpp-style servers never see
        prompt+reply beyond their window, and automatic recovery from
        template incompatibilities (folding the system prompt into the first
        user turn if the template rejects a system role). A model that is
        still loading answers 400 or 503 for a while; without a retry loop,
        the first call of a batch would fail and every subsequent one would
        fail the same way.
        """
        messages = merge_consecutive(messages)
        if self._no_system_role:
            messages = fold_system(messages)
        delay = 2.0
        started = time.time()
        announced = False
        attempt = 0
        trimmed_once = False
        shrunk_once = False
        while True:
            attempt += 1
            # Proactive fit: llama.cpp/LM Studio reject any request whose
            # prompt + max_tokens exceeds the configured window ("kept
            # hitting limits" even on correct configs), and Ollama silently
            # truncates the middle instead -- which is how a local model
            # "runs to completion" with an answer to a task it never saw.
            # Shrink the oldest tool-result turns until the reply budget
            # genuinely fits; only the newest RESULT matters mid-gather,
            # since gathered source reaches the answer phase separately.
            room = self.cfg.context_tokens - max_tokens
            total = sum(estimate_tokens(m["content"]) for m in messages)
            if room > 0 and total > room * 0.9:
                new_messages = list(messages)
                # Drop the OLDEST RESULT turn together with the assistant
                # action that requested it -- removing only one side would
                # leave two same-role messages in a row, which strict chat
                # templates (Mistral especially) reject outright.
                while True:
                    cur = sum(estimate_tokens(m["content"]) for m in new_messages)
                    if cur <= room * 0.85:
                        break
                    idxs = [i for i, m in enumerate(new_messages)
                            if m["role"] == "user"
                            and m["content"].startswith("RESULT:\n")]
                    if len(idxs) < 2:
                        break     # keep the newest result; server decides
                    drop = idxs[0]
                    pair = drop - 1 if drop and new_messages[drop - 1]["role"] == "assistant" else None
                    kill = {drop} | ({pair} if pair is not None else set())
                    new_messages = [m for k, m in enumerate(new_messages) if k not in kill]
                if len(new_messages) != len(messages):
                    print(f"[prompt was ~{total:,} tokens for a "
                          f"{self.cfg.context_tokens:,}-token window; dropped "
                          f"older tool results before sending]", file=sys.stderr)
                    messages = new_messages
            try:
                return self._chat_once(messages, max_tokens, schema, stream,
                                       disable_thinking)
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
                if not shrunk_once and max_tokens > 256:
                    # Our char/token estimate was optimistic (code-heavy text
                    # is denser than prose). Halve the REPLY budget for a
                    # second attempt: the prompt is already trimmed, and a
                    # shorter step reply still carries the action (the schema
                    # puts it first, so truncation loses only `reasoning`).
                    shrunk_once = True
                    max_tokens = max(256, max_tokens // 2)
                    print("[server context overflow again; halving the reply "
                          "budget and retrying once]", file=sys.stderr)
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
                   schema: dict | None = None, stream: bool = False,
                   disable_thinking: bool | None = None) -> str:
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
        # APIs don't understand it, and a strict one could 400 on it, so gate
        # on the URL being remote rather than on json_mode: a LOCAL server
        # (LM Studio/Ollama) running a Qwen3 build with json_mode="object"
        # must still be able to switch thinking off -- that combination was
        # silently ignored before, which is exactly "I set disable_thinking
        # and it still thinks / never returns JSON".
        def _url_is_remote() -> bool:
            try:
                from urllib.parse import urlparse
                h = (urlparse(self.cfg.base_url).hostname or "").lower()
            except ValueError:
                return False
            return h not in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "")

        if (disable_thinking if disable_thinking is not None
                else self.cfg.disable_thinking) and not _url_is_remote():
            # Two spellings, one intent: Qwen/DeepSeek-R1 family templates
            # read chat_template_kwargs.enable_thinking, llama.cpp's own
            # sampler reads no_temp/top_k style keys, and some builds accept
            # the flat `think: false` field. Servers ignore unknown fields;
            # servers that DON'T ignore them are exactly the thinking models
            # this targets. Without any of these, a thinking model spends its
            # entire step budget on hidden reasoning and emits nothing --
            # "with thinking off it didn't return an answer" is the other
            # face of the same coin.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if schema is not None:
            if self.cfg.json_mode == "object":
                payload["response_format"] = {"type": "json_object"}
            else:
                # json_schema (grammar-constrained) is LM Studio's feature.
                # Ollama and many hosted APIs 400 on response_format entirely,
                # or silently ignore it and answer in prose -- which then
                # surfaces as "kept hitting limits" or "would not produce
                # valid JSON". So only send it when the server actually
                # advertised support: LM Studio reports `supported_parameters`
                # on its native /api/v0/models endpoint. Servers that never
                # answered the probe are assumed to support it (that was the
                # historical default); servers that answered WITHOUT it fall
                # back to plain JSON prompting for this request.
                if self._schema_supported is False:
                    fields = ", ".join(schema.get("properties", {}).keys())
                    msgs[-1]["content"] += (
                        "\n\nReply with one JSON object only, no prose. "
                        f"Fields: {fields}.")
                else:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "step", "strict": True, "schema": schema},
                    }

        t0 = time.time()
        self.stats["calls"] += 1
        # One truncation flag per LOGICAL call, not per HTTP attempt. chat()
        # retries _chat_once after context-overflow trimming and transient
        # failures; clearing here means a stale "length" from an abandoned
        # earlier attempt can never make apply()'s continuation loop ask the
        # model to repeat edits that already landed cleanly. The streaming
        # path ORs into this so an escalation retry inside one logical call
        # keeps the signal; chat_json sets last_finish_reason explicitly on
        # every buffered attempt it inspects.
        self.stream_truncated = False

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
                chunk = json.loads(body)["choices"][0]
                d = chunk.get("delta") or {}
                fr = chunk.get("finish_reason") or ""
            except Exception:
                continue
            if fr == "length":
                # OR, not =: a logical chat() call may contain several HTTP
                # attempts (context-overflow retries); once any streamed
                # attempt was cut, the reply we return IS cut.
                self.stream_truncated = True
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
    @staticmethod
    def _schema_has_big_fields(schema: dict | None) -> bool:
        """True when the schema contains a string field large enough to hold
        real file content (search/replace blocks, whole-file writes).

        `content` in STEP_SCHEMA is deliberately capped at 6000 chars so a
        step reply can never legitimately carry a whole large file -- if a
        step reply gets cut off with that schema, it died inside 'reasoning',
        and doubling max_tokens cannot fix that. Schemas WITHOUT such fields
        therefore skip the budget-escalation retry loop entirely.
        """
        BIG_FIELD_MIN = 8000   # maxLength above which a string field is "large"
        props = (schema or {}).get("properties", {})
        return any(
            p.get("type") == "string" and int(p.get("maxLength") or 0) > BIG_FIELD_MIN
            for p in props.values())

    def chat_json(self, messages: list[dict], schema: dict, max_tokens: int,
                  retries: int = 2, cap: int | None = None,
                  disable_thinking: bool | None = None) -> dict:
        """Send a grammar-constrained chat request and return the parsed
        JSON object.

        The grammar guarantees well-formed JSON *if the model is allowed to
        finish*. The common failure is not malformed output, it is a response
        cut off at max_tokens mid-string. So a truncated attempt retries with
        a larger budget rather than the same one, and says so.
        """
        last = ""
        budget = max_tokens
        # Truncation recovery state. `step_cut` counts how many times a
        # schema-constrained step reply was cut off mid-'reasoning' (the only
        # field left when the schema orders action/args first). Retrying such
        # a step verbatim just re-rolls the same rambling essay, so after the
        # second cut we stop retrying and treat the recovered action as final.
        self.step_truncated = False
        step_cut = 0
        # One re-ask per call for the "whole budget spent on a <think>/prose
        # preamble, no JSON at all" failure: retrying verbatim just re-rolls
        # the same essay, so the retry carries an explicit no-reasoning
        # instruction instead. Bounded so a model that cannot stop rambling
        # still falls through to the informative error.
        trunc_nudged = False
        # Ceiling for truncation-retry escalation. Default is NOT a hard
        # 2000 -- it is derived from the caller's own config (reserve /
        # answer budgets), so raising max_tokens_step or reserve_tokens in
        # the profile actually raises this limit too.
        if cap is None:
            from .config import is_hosted
            if is_hosted(self.cfg):
                # Hosted APIs have no local window to clamp against, but
                # they DO have a provider-side per-message OUTPUT ceiling
                # (~8k tokens on deepseek-chat). The old ceiling here was
                # max(65536, ...) unconditionally, which promised more than
                # any hosted API can deliver: an escalated retry re-rolled
                # the same doomed request twice more before erroring, with
                # a message ("ceiling 65536") that sent users chasing a
                # setting that cannot fix anything. Honour the configured
                # provider ceiling when known; otherwise stay generous for
                # providers that genuinely allow long messages.
                mot = int(getattr(self.cfg, "max_output_tokens", 0) or 0)
                if mot > 0:
                    cap = max(mot, max_tokens)
                else:
                    cap = max(65536, self.cfg.max_tokens_answer,
                              self.cfg.max_tokens_step * 8, max_tokens * 8)
            else:
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
            raw = self.chat(messages, max_tokens=budget, schema=schema,
                            disable_thinking=disable_thinking)
            last = raw
            truncated = self.last_finish_reason == "length"
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.split("\n", 1)[-1] if "\n" in text else text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Salvage a JSON object if the model added prose around it.
                # Scan EVERY '{' with raw_decode instead of slicing
                # find("{")..rfind("}"): thinking models often emit several
                # candidate objects (reasoning notes, examples) before the
                # real reply, and the outermost slice glues junk between the
                # first and last brace into one unparseable blob -- "failure
                # to return json" even though valid JSON sat in the middle.
                # Requiring an "action" key keeps a stray {"note": ...} from
                # being mistaken for the step reply.
                dec = json.JSONDecoder()
                start = 0
                while True:
                    i = text.find("{", start)
                    if i == -1:
                        break
                    try:
                        obj, _end = dec.raw_decode(text[i:])
                    except json.JSONDecodeError:
                        start = i + 1
                        continue
                    if isinstance(obj, dict) and ("action" in obj or "edits" in obj):
                        if i > 0 or _end + i < len(text.strip()):
                            print("[prose around JSON; salvaged the object]",
                                  file=sys.stderr)
                        return obj
                    start = i + 1
                i, j = text.find("{"), text.rfind("}")
                if i != -1 and j > i:
                    frag = text[i : j + 1]
                    try:
                        return json.loads(frag)
                    except json.JSONDecodeError:
                        pass
                    # Some non-grammar-constrained servers (DeepSeek in
                    # json_object mode among them) emit Python-style dicts
                    # with single quotes instead of JSON. Convert only when
                    # the fragment has no double quotes at all -- otherwise
                    # apostrophes inside real strings would get mangled.
                    if '"' not in frag and "'" in frag:
                        cand = frag.replace("'", '"')
                        try:
                            obj = json.loads(cand)
                        except json.JSONDecodeError:
                            obj = None
                        if isinstance(obj, dict) and obj:
                            print("[reply was a single-quoted dict; "
                                  "converted to JSON]", file=sys.stderr)
                            return obj
                if truncated:
                    patched = repair_truncated_json(text)
                    if patched:
                        try:
                            obj = json.loads(patched)
                        except json.JSONDecodeError:
                            obj = None
                        # Only trust it if the part that matters survived.
                        if isinstance(obj, dict) and obj.get("action"):
                            if not self._schema_has_big_fields(schema):
                                # A schema step's args all precede 'reasoning',
                                # so a recovered action is COMPLETE -- retrying
                                # only re-rolls the essay that got cut. Accept
                                # it, flag the truncation so gather() can tell
                                # the model to keep its next reasoning short,
                                # and cap pointless retries at two.
                                print("[reply was cut off; recovered the "
                                      "action from the partial reply]",
                                      file=sys.stderr)
                                self.step_truncated = True
                                return obj
                            # Big-field (answer-style) schema: the cut landed
                            # inside a content/replacement string, so what we
                            # salvaged is PARTIAL. Do not act on half a file --
                            # fall through to escalation/salvage-last logic.
                            step_cut += 1
                            if step_cut >= 2:
                                print("[reply was cut off mid-content; last "
                                      "attempt will be salvaged whole]",
                                      file=sys.stderr)
                                self.step_truncated = True
                                return obj
                if truncated:
                    # A step schema has NO large fields -- a truncated step
                    # reply can only have died inside 'reasoning', so there is
                    # nothing to salvage and doubling the budget cannot help.
                    # The old escalation path (7000 -> 14000 -> 28000 -> ...)
                    # just re-sent the same oversized request three times on
                    # DeepSeek, whose per-message OUTPUT cap (~8k tokens) is a
                    # provider-side fact no client setting can raise. Skip the
                    # escalation entirely for schema steps.
                    if not self._schema_has_big_fields(schema):
                        # ...but do re-ask ONCE with an explicit no-reasoning
                        # instruction before giving up. A thinking model that
                        # spent its whole budget on the preamble (the frequent
                        # local-server failure: "kept running past the token
                        # limit before finishing its JSON") often lands the
                        # JSON immediately when told to skip it -- this is the
                        # same nudge agent.gather applies, moved down into
                        # chat_json so plan/summary callers get it too.
                        if not trunc_nudged and attempt < retries:
                            trunc_nudged = True
                            print("[reply was cut off before any JSON was "
                                  "emitted; re-asking with a no-reasoning "
                                  "nudge]", file=sys.stderr)
                            messages = messages + [
                                {"role": "assistant", "content": raw},
                                {"role": "user", "content":
                                    "Your previous reply was cut off by the "
                                    "output token limit before you emitted "
                                    "the JSON object. Do not think or explain: "
                                    "output the JSON object itself now, with "
                                    "'reasoning' kept under 10 words."},
                            ]
                            continue
                        break   # fall through to the informative error below
                    nxt = min(budget * 2, cap)
                    if nxt > budget:
                        budget = nxt
                        print(f"[reply was cut off at the token limit; retrying "
                              f"with max_tokens={budget}] ", file=sys.stderr)
                    else:
                        print(f"[reply was cut off at the token limit and "
                              f"max_tokens is already at the ceiling "
                              f"({budget}); the answer will be salvaged from "
                              f"what survived]",
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
                "  A reasoning (Qwen3/DeepSeek-R1) model spends this budget on its\n"
                "  <think> block before emitting any JSON -- the usual fix is a\n"
                "  larger budget, or disabling thinking so it writes JSON right away.\n"
                "  Raise the budget for THIS call in your active profile:\n"
                "    max_tokens_plan  (pedacito new planning)\n"
                "    max_tokens_step  (gather-loop tool calls)\n"
                "    max_tokens_answer (the final answer / edit reply)\n"
                "  ...and/or set disable_thinking = true. The retry ceiling follows\n"
                "  reserve_tokens / max_tokens_answer, it is not a fixed limit.\n"
                "  NOTE for hosted APIs (DeepSeek/OpenAI): their per-message OUTPUT\n"
                "  cap (~8k tokens on deepseek-chat) is provider-side and cannot be\n"
                "  raised from here -- keep max_tokens_step at/below it and let the\n"
                "  chunked-write protocol split big files across turns instead.\n"
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
        base = _server_root(self.cfg.base_url)
        try:
            r = requests.get(base + "/api/v0/models", headers=self.headers, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json().get("data", [])
        except Exception:
            return None
        out = {m.get("id", "?"): m.get("state", "unknown") for m in data if isinstance(m, dict)}
        return out or None

    @staticmethod
    def _pick_model(entries, model: str) -> dict | None:
        """Find the entry for `model` in a server's model list.

        Servers report the same model under several spellings: LM Studio uses
        `id`/`key` (sometimes with an `@q4_k_m` variant suffix), Ollama uses
        `name`, and OpenAI endpoints use `id`. Exact matches win; a match on
        the part before `@` handles the variant suffix.
        """
        want = str(model or "").strip().lower()
        if not want:
            return None

        def names(entry: dict) -> list[str]:
            vals = (entry.get("id"), entry.get("key"), entry.get("name"),
                    entry.get("model"), entry.get("model_name"),
                    entry.get("display_name"))
            return [str(v).strip().lower() for v in vals if v]

        for entry in entries:
            if isinstance(entry, dict) and want in names(entry):
                return entry
        base = want.split("@", 1)[0]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if any(n.split("@", 1)[0] == base for n in names(entry)):
                return entry
        return None

    def _absorb_model_list(self, entries, source: str) -> dict:
        """Extract size/context from one model-list entry.

        Servers extend the OpenAI /v1/models shape differently:
          * LM Studio v1: `size_bytes`, `loaded_instances[].config.context_length`,
            `max_context_length`.
          * Lemonade: `size` in GB, `context_length` (configured/loaded), and
            `max_context_window`.
        Both spellings are absorbed here so model_info does not need to know
        which server answered.
        """
        m = self._pick_model(entries, self.cfg.model)
        if not m:
            return {}

        def ctx_ok(value) -> int | None:
            try:
                n = int(value)
            except (TypeError, ValueError):
                return None
            return n if n > 0 else None

        got: dict = {}
        raw_size = m.get("size_bytes")
        if raw_size is not None:
            try:
                got["size_bytes"] = int(raw_size)
            except (TypeError, ValueError):
                pass
        if not got.get("size_bytes") and m.get("size") is not None:
            # Lemonade reports model size as a GB number, not bytes.
            try:
                got["size_bytes"] = int(float(m["size"]) * _GIB)
            except (TypeError, ValueError):
                pass
        loaded = m.get("loaded_instances") or []
        loaded_ctx = None
        if loaded and isinstance(loaded[0], dict):
            loaded_ctx = ctx_ok(
                (loaded[0].get("config") or {}).get("context_length"))
        # Lemonade puts the model's effective (including saved overrides)
        # recipe options in the same list entry. Prefer that over its
        # top-level context_length, which some versions leave at the
        # registry/default value instead of the configured ctx_size.
        recipe_ctx = ctx_ok((m.get("recipe_options") or {}).get("ctx_size"))
        ctx = (loaded_ctx or recipe_ctx or ctx_ok(m.get("context_length"))
               or ctx_ok((m.get("recipe_options") or {}).get("num_ctx")))
        if ctx:
            got["context_length"] = ctx
        max_ctx = ctx_ok(m.get("max_context_length")
                         or m.get("max_context_window"))
        if max_ctx:
            got["max_context_length"] = max_ctx
        if got:
            if str(m.get("owned_by", "")).lower() == "lemonade" \
                    or m.get("recipe"):
                source = "lemonade"
            got["source"] = source
        return got

    def _lemonade_health_context(self, root: str) -> tuple[int | None, int | None]:
        """Read the context a currently loaded Lemonade model is using.

        Lemonade's /v1/models `context_length` can lag behind the options a
        model was loaded with (and older builds reported a registry default).
        The health endpoint lists resident models with the recipe options they
        were actually launched with, which is the fact the budget fit needs.
        Tries both prefixes because older builds expose /api/v0/health.
        """
        def ctx_ok(value):
            try:
                n = int(value)
            except (TypeError, ValueError):
                return None
            return n if n > 0 else None

        for path in ("/api/v1/health", "/api/v0/health"):
            try:
                r = requests.get(root + path, headers=self.headers, timeout=5)
                if r.status_code != 200:
                    continue
                data = r.json()
            except Exception:
                continue
            entries = data.get("all_models_loaded") or []
            entry = self._pick_model(entries, self.cfg.model)
            if not entry:
                continue
            opts = entry.get("recipe_options") or {}
            ctx = (ctx_ok(opts.get("ctx_size"))
                   or ctx_ok(entry.get("context_length")))
            max_ctx = ctx_ok(entry.get("max_context_window"))
            if ctx or max_ctx:
                return ctx, max_ctx
        return None, None

    def _lemonade_options_context(self, root: str) -> int | None:
        """Ask Lemonade which ctx_size a load would use right now.

        /v1/models/{id}/options separates saved overrides from defaults and
        reports `resolved_ctx_size` (including automatic sizing), so it is
        more trustworthy than the model-list `context_length` on builds that
        report a default there. Absent on older Lemonade versions, hence the
        silent failure.
        """
        from urllib.parse import quote

        def ctx_ok(value):
            try:
                n = int(value)
            except (TypeError, ValueError):
                return None
            return n if n > 0 else None

        url = (root + "/v1/models/" + quote(str(self.cfg.model), safe="")
               + "/options")
        try:
            r = requests.get(url, headers=self.headers, timeout=5)
            if r.status_code != 200:
                return None
            data = r.json()
        except Exception:
            return None
        for key in ("resolved_ctx_size",):
            ctx = ctx_ok(data.get(key))
            if ctx:
                return ctx
        for key in ("effective", "saved"):
            opts = data.get(key) or {}
            if isinstance(opts, dict):
                ctx = ctx_ok(opts.get("ctx_size"))
                if ctx:
                    return ctx
        return None

    def model_info(self) -> dict | None:
        """Ask the server for metadata about `cfg.model`: file size, configured
        context length, and the model's maximum context.

        Returns a dict with any of `size_bytes`, `context_length`,
        `max_context_length`, plus `source`; None when nothing answers. This
        is the start-of-run probe auto-fit uses to size the context window
        from the machine's actual VRAM instead of trusting a hand-written
        token budget.

        Tried in order, stopping once both size and context are known:
          * LM Studio v1 `/api/v1/models` -- `size_bytes`, loaded instance's
            configured context, and `max_context_length`.
          * Any OpenAI-compatible `/v1/models` -- Lemonade returns `size` (GB),
            `context_length` and `max_context_window` there; plain servers
            simply yield nothing useful.
          * LM Studio v0 `/api/v0/models` -- context length on older builds
            (and `size_bytes` when the build reports it).
          * Ollama `/api/show` -- the weights' `size` and `llama.context_length`
            / `num_ctx` (Lemonade also implements this endpoint).

        Every probe is best-effort: a server that does not implement an
        endpoint (or is offline) simply yields nothing, and the caller keeps
        the config's declared context.
        """
        root = _server_root(self.cfg.base_url)
        out: dict = {}

        def enough() -> bool:
            return bool(out.get("size_bytes") and (
                out.get("context_length") or out.get("max_context_length")))

        def merge(got: dict) -> None:
            for key, value in got.items():
                if key == "source":
                    continue
                if value and not out.get(key):
                    out[key] = value
            if got.get("source"):
                out.setdefault("source", got["source"])

        for url, source in (
                (root + "/api/v1/models", "lmstudio-v1"),
                (root + "/v1/models", "openai-models"),
                (root + "/api/v0/models", "lmstudio-v0")):
            if enough():
                break
            try:
                r = requests.get(url, headers=self.headers, timeout=5)
                if r.status_code != 200:
                    continue
                data = r.json()
                if isinstance(data, dict):
                    entries = data.get("models") or data.get("data") or []
                else:
                    entries = data if isinstance(data, list) else []
                merge(self._absorb_model_list(entries, source))
            except Exception:
                pass

        # Lemonade exposes the loaded recipe options separately from the
        # OpenAI model list, and its per-model options endpoint knows the
        # saved ctx_size override. Those are more accurate than a list entry
        # that may carry the registry/default window.
        if out.get("source") == "lemonade":
            health_ctx, health_max = self._lemonade_health_context(root)
            if health_max and not out.get("max_context_length"):
                out["max_context_length"] = health_max
            if health_ctx:
                out["context_length"] = health_ctx
                out["context_source"] = "lemonade loaded options"
            else:
                options_ctx = self._lemonade_options_context(root)
                if options_ctx:
                    out["context_length"] = options_ctx
                    out["context_source"] = "lemonade saved options"

        # Ollama /api/show (Lemonade implements it too) reports both the
        # weights size and the configured context length.
        if not enough():
            try:
                r = requests.post(root + "/api/show",
                                  json={"name": self.cfg.model}, timeout=5)
                if r.status_code == 200:
                    d = r.json()
                    got: dict = {}
                    size = d.get("size")
                    if size is not None:
                        try:
                            got["size_bytes"] = int(size)
                        except (TypeError, ValueError):
                            pass
                    info = d.get("model_info") or {}
                    params = d.get("parameters") or {}
                    num_ctx = (params.get("num_ctx")
                               if isinstance(params, dict) else None)
                    ctx = num_ctx or info.get("llama.context_length")
                    try:
                        ctx = int(ctx)
                    except (TypeError, ValueError):
                        ctx = None
                    if ctx and ctx > 0:
                        got["context_length"] = ctx
                    max_ctx = info.get("llama.context_length")
                    try:
                        max_ctx = int(max_ctx)
                    except (TypeError, ValueError):
                        max_ctx = None
                    if max_ctx and max_ctx > 0:
                        got["max_context_length"] = max_ctx
                    details = d.get("details") or {}
                    if not got.get("size_bytes") and details.get("parameter_size"):
                        got["parameter_size"] = details["parameter_size"]
                    if got:
                        got["source"] = "ollama"
                    merge(got)
            except Exception:
                pass

        if not (out.get("size_bytes") or out.get("context_length")
                or out.get("max_context_length")):
            return None
        return out

    def detect_context(self) -> int | None:
        """Probe the server for the REAL context length configured for
        `cfg.model`, so token budgets can be auto-fitted to what the model
        will actually accept instead of a guessed config number.

        Works against the two local servers Pedacito targets:
          * LM Studio -- native /api/v1/models and /api/v0/models carry the
            configured and maximum context lengths.
          * Ollama -- POST /api/show returns model_info including
            llama.context_length (and num_ctx when explicitly set; Ollama
            defaults to 4096 when unset).

        Returns None for hosted APIs and any server that doesn't answer --
        callers then keep whatever context_tokens the config declares.
        Never raises: this is an optimisation probe, not a requirement.
        """
        info = self.model_info() or {}
        return info.get("context_length")

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
        # Learn whether this server honours grammar-constrained JSON before
        # the first real step request. LM Studio's native API lists
        # `supported_parameters` per model; when it answers WITHOUT
        # response_format / json_schema in that list (or the model entry
        # lacks the field entirely on an older build), we downgrade to plain
        # JSON prompting rather than getting prose back from a call we asked
        # to be constrained -- which previously surfaced as endless retries
        # and "would not produce valid JSON".
        if self._schema_supported is None and self.cfg.json_mode != "object":
            try:
                root = _server_root(self.cfg.base_url)
                r = requests.get(root + "/api/v0/models", headers=self.headers,
                                 timeout=5)
                if r.status_code == 200:
                    for m in r.json().get("data", []):
                        if not isinstance(m, dict):
                            continue
                        if m.get("id") == self.cfg.model or m.get("key") == self.cfg.model:
                            sp = [str(x).lower() for x in
                                  (m.get("supported_parameters") or [])]
                            if any("response_format" in x or "json_schema" in x
                                   for x in sp):
                                self._schema_supported = True
                            elif sp:
                                self._schema_supported = False
                                if verbose:
                                    print("[server does not support constrained "
                                          "JSON; using plain JSON prompting]",
                                          file=sys.stderr)
                            break
            except Exception:
                pass   # probe is an optimisation; never fatal
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