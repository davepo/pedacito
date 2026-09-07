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
    """Close a JSON object that was cut off mid-flight, dropping any
    partially-written string value (which could be a truncated file path or
    similar, unsafe to guess at) and any dangling key with no value.

    Returns the repaired JSON text, or None if `text` was not actually
    truncated or nothing salvageable survived.

    This is why the step schema (see tools.py) puts `action` and its
    arguments before `reasoning`: a reply truncated inside the trailing
    explanation still has everything the agent needs, so closing the quote
    and the brace recovers a usable step instead of costing another round
    trip.
    """
    start = text.find("{")
    if start == -1:
        return None
    buf = text[start:]
    depth = 0
    in_str = False
    esc = False
    str_start = -1
    for i, ch in enumerate(buf):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            if in_str:
                in_str = False
            else:
                in_str = True
                str_start = i
            continue
        if in_str:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    if depth <= 0 and not in_str:
        return None            # not actually truncated

    if in_str:
        # The value was cut mid-string, so it is a fragment: "search_lib" could
        # be any of several files. Keeping it would silently act on the wrong
        # one, so drop the whole key rather than guess.
        buf = buf[:str_start]
    # Remove a dangling key, with or without its colon:
    #   {"action":"read","file":     ->  {"action":"read"
    #   {"action":"done","reasoning" ->  {"action":"done"
    buf = re.sub(r'[,{]\s*"[^"]*"\s*:?\s*$',
                 lambda m: "{" if m.group(0)[0] == "{" else "", buf)
    buf = re.sub(r',\s*$', "", buf)
    buf = re.sub(r'\{\s*$', "{", buf)
    if buf.rstrip().endswith("{"):
        return None            # nothing survived worth trusting
    return buf + "}" * max(depth, 0)


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
    retries, template quirks, and reasoning-model output transparently."""

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
        while True:
            attempt += 1
            try:
                return self._chat_once(messages, max_tokens, schema, stream)
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
        if self.cfg.thinking_suffix and messages:
            messages = [dict(m) for m in messages]
            messages[-1]["content"] += "\n\n" + self.cfg.thinking_suffix
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if self.cfg.disable_thinking:
            # llama.cpp/LM Studio forward this into the chat template. Harmless
            # on templates that ignore it.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if schema is not None:
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
        # A reasoning model streams its thinking inline; drop it from the value
        # we return so edit parsing never sees it, even though it was displayed.
        return strip_reasoning("".join(out))

    # --------------------------------------------------------------- helpers
    def chat_json(self, messages: list[dict], schema: dict, max_tokens: int,
                  retries: int = 2) -> dict:
        """Send a grammar-constrained chat request and return the parsed
        JSON object.

        The grammar guarantees well-formed JSON *if the model is allowed to
        finish*. The common failure is not malformed output, it is a response
        cut off at max_tokens mid-string. So a truncated attempt retries with
        a larger budget rather than the same one, and says so.
        """
        last = ""
        budget = max_tokens
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
                    budget = min(budget * 2, 2000)
                    print(f"[reply was cut off at the token limit; retrying with "
                          f"max_tokens={budget}]", file=sys.stderr)
                    continue
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "That was not valid JSON. Reply with "
                                                "the JSON object only, and keep "
                                                "'reasoning' under 15 words."},
                ]
        if self.last_finish_reason == "length":
            raise LLMError(
                f"The model kept running past the token limit before finishing its "
                f"JSON (tried up to max_tokens={budget}).\n"
                "  It is writing an essay in the 'reasoning' field. Raise "
                "max_tokens_step in your\n"
                "  config, or use a model that follows the 'under 15 words' "
                "instruction better.\n"
                f"  Last partial reply: {last[-160:]}")
        raise LLMError(
            f"Model would not produce valid JSON after {retries + 1} tries.\n"
            "  If this persists, the runtime may be ignoring the JSON schema "
            "(common with MLX\n  builds; llama.cpp/GGUF supports it).\n"
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
