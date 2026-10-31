"""OpenAI-compatible client: chat completions, JSON-schema tool calls, and
model warmup.

Usage
-----
    from .llm import LLMClient

    client = LLMClient(cfg)
    client.check()                       # verify the server and model
    text = client.chat(messages, max_tokens=500)
    step = client.chat_json(messages, schema, max_tokens=500)

Works against any server exposing an OpenAI-compatible /v1 surface (tested
against LM Studio, Lemonade, and Ollama). The tool loop uses
response_format={"type": "json_schema", ...}, which servers with grammar-
constrained decoding enforce, so a small model cannot emit unparseable JSON.

Also handles local-model failure modes: templates that reject certain message
shapes, replies truncated at the token limit, reasoning models that wrap their
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
from urllib.parse import quote, urlparse

import requests


class LLMError(RuntimeError):
    """Base class for any failure talking to the model server."""


class ModelLoadError(LLMError):
    """The server could not load the model. Retrying rarely helps."""


class TemplateError(LLMError):
    """The model's chat template rejected the message sequence."""


class ContextOverflowError(LLMError):
    """The prompt is bigger than the server's context window. `chat()` trims
    the conversation and retries before giving up."""


class _Transient(LLMError):
    """Worth retrying: model still loading, or the server is busy."""


# Mistral-family templates raise a Jinja exception on two user messages in a
# row; Gemma has no system role. The fix is to reshape the request.
_TEMPLATE_HINTS = (
    "jinja", "roles must alternate", "conversation roles",
    "system role not supported", "only user and assistant roles",
)

# LM Studio returns 400 for both "still loading" and "failed to load", so the
# status code alone can't distinguish transient from fatal.
_LOADING = ("is loading", "loading model", "model is not loaded", "please wait")
_LOAD_FAILED = ("failed to load model", "error loading model", "no model loaded")

# Shown on a plain connection failure; the fix differs per server.
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
  1. Not enough free VRAM. Another model is probably still resident --
     unload it in your model server, or lower this model's context length.
  2. The context length is set higher than the card can hold. Try 16384 with
     K/V cache at Q8_0 and flash attention on.
  3. An incomplete or corrupt download. Re-download the model.
  4. A runtime mismatch. Try switching between ROCm and Vulkan for this model.

  Fastest way to tell: load the model by hand in the server's own UI. Whatever
  error it shows there is the real one -- the API only reports that it failed."""


_GIB = 1024 ** 3


def _positive_int(value) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _is_remote(base_url: str) -> bool:
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return False
    return host not in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "")


def _server_root(base_url: str) -> str:
    """Strip a `/v1` or `/api/v1` suffix from base_url, leaving the server root
    where the native endpoints (`/api/v0/models`, `/api/show`, ...) live."""
    base = base_url.rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base


# A constrained-JSON verdict costs one round-trip to probe; remembering it per
# server+model keeps repeat runs consistent and skips the probe.
_MODE_CACHE_TTL = 7 * 24 * 3600


def _mode_cache_key(base_url: str, model: str) -> str:
    return f"{base_url.rstrip('/')}|{model}"


def _load_mode_cache(base_url: str, model: str) -> bool | None:
    """Cached `_schema_supported` verdict, or None when absent/stale/corrupt."""
    try:
        from . import config
        path = config.USER_CONFIG.parent / "server_modes.json"
        entry = json.loads(path.read_text(encoding="utf-8")).get(
            _mode_cache_key(base_url, model))
        if isinstance(entry, dict) and \
                isinstance(entry.get("schema_supported"), bool) and \
                time.time() - float(entry.get("ts", 0)) < _MODE_CACHE_TTL:
            return entry["schema_supported"]
    except Exception:
        pass
    return None


def _save_mode_cache(base_url: str, model: str, supported: bool) -> None:
    """Persist a probed verdict; an unwritable config dir never kills a run."""
    try:
        from . import config
        path = config.USER_CONFIG.parent / "server_modes.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        data[_mode_cache_key(base_url, model)] = {
            "schema_supported": supported, "ts": time.time()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


def _schema_example(schema: dict) -> str:
    """One well-formed dummy object for `schema`: the few-shot example shown
    when structured output degrades to plain JSON prompting."""
    ex = {}
    for k, p in (schema or {}).get("properties", {}).items():
        t = p.get("type")
        if p.get("enum"):
            ex[k] = p["enum"][0]
        elif t in ("integer", "number"):
            ex[k] = 0
        elif t == "boolean":
            ex[k] = False
        elif t == "array":
            ex[k] = []
        elif t == "object":
            ex[k] = {}
        else:
            ex[k] = ""
    return json.dumps(ex)


# Tiny step-shaped schema for `config --doctor`'s structured-output probes.
_PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["read", "done"]},
        "file": {"type": "string"},
        "reasoning": {"type": "string", "maxLength": 40},
    },
    "required": ["action", "file", "reasoning"],
    "additionalProperties": False,
}

# Representative prose+code sample for tokenizer calibration; token density
# differs per model family, so the server's own count beats a fixed 3.6.
_TOKENIZER_SAMPLE = (
    "def gather(self, task, verbose=True):\n"
    '    """Run the tool-calling loop until the model says done."""\n'
    "    budget = self.cfg.max_gathered_chars\n"
    "    while budget > 0:\n"
    "        step = self.client.chat_json(messages, schema, max_tokens=500)\n"
    "        if step.get('action') == 'read':\n"
    "            obs = tool_read(self.index, step['file'], step['start'])\n"
    "            budget -= len(obs)\n"
    "The quick brown fox jumps over the lazy dog. Structured output keeps "
    "tool calls parseable: a small model cannot emit unparseable JSON when "
    "the server constrains decoding with a grammar. Estimates matter too, "
    "because prompt plus reply must fit the context window or the server "
    "silently truncates the middle of the conversation.\n"
)


# Reasoning models emit a thinking block before the answer. Some runtimes move
# it to a separate field, some leave it inline in `content`; an unstripped
# block breaks JSON parsing and lands in the answer text.
_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
# An unterminated block: the model ran out of tokens mid-thought.
_THINK_OPEN_RE = re.compile(r"^\s*<(think|thinking|reasoning)>.*", re.DOTALL | re.IGNORECASE)
# Some templates emit only the closing marker when the runtime already routed
# the thinking text to a separate field.
_STRAY_THINK_CLOSE_RE = re.compile(r"</(think|thinking|reasoning)>",
                                   re.IGNORECASE)
# Native tool-call syntax in the answer phase means the model ignored "no
# tools here". A few are tolerated (recovery honours up to four reads), but a
# stream that keeps emitting them is a degenerate loop; stop it at six.
_NATIVE_TOOL_OPEN_RE = re.compile(
    r"<(?:tool_call|invoke|read)\b|<function\s*=", re.IGNORECASE)
_MAX_STREAM_TOOL_OPENS = 6


def strip_reasoning(text: str) -> str:
    """Remove <think>/<thinking>/<reasoning> blocks from model output. Text
    that is only an unterminated opening tag (cut mid-thought) becomes ""."""
    if not text:
        return ""
    out = _THINK_RE.sub("", text)
    if _THINK_OPEN_RE.match(out):
        return ""
    out = _STRAY_THINK_CLOSE_RE.sub("", out)
    return out.strip()


def extract_content(message: dict) -> str:
    """Answer text of a response message (from `content`, else a reasoning
    field), with any thinking block stripped."""
    content = message.get("content") or ""
    if not content.strip():
        # Some runtimes put everything in a reasoning field; better than none.
        content = message.get("reasoning_content") or message.get("reasoning") or ""
    return strip_reasoning(content)


def _cut_field(text: str, obj: dict, schema: dict | None) -> str:
    """Which schema field a truncated reply died inside, or "" when benign.

    An unescaped `"key":` in the raw reply is a real key opening, never payload
    text (quotes inside string values are escaped). The LAST opened key is where
    generation stopped; it counts as cut when repair_truncated_json dropped it
    whole (absent from obj) or kept an unterminated string fragment. Callers
    treat "reasoning" as benign and refuse to run steps whose payload field was
    cut.
    """
    props = sorted((schema or {}).get("properties") or {})
    if not props:
        return ""
    rx = re.compile(r'(?<!\\)"(' + "|".join(re.escape(p) for p in props)
                    + r')"\s*:')
    matches = list(rx.finditer(text))
    if not matches:
        return ""
    last = matches[-1]
    if last.group(1) not in obj:
        return last.group(1)         # dropped whole by the salvage
    # Fragment-kept case: no closing quote before EOF means a partial payload.
    k = last.end()
    while k < len(text) and text[k] in " \t\r\n":
        k += 1
    if k < len(text) and text[k] == '"':
        j = k + 1
        while j < len(text):
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == '"':
                return ""            # value completed; the cut landed later
            j += 1
        return last.group(1)         # unterminated: kept as a fragment
    return ""                        # number/bool present in obj: complete


def repair_truncated_json(text: str) -> str | None:
    """Salvage the complete top-level keys of a JSON object cut off mid-
    generation.

    Everything from the first incomplete token onward (unterminated string,
    bare key, dangling colon/comma, half-typed number) is dropped and the
    surviving pairs are closed into an object; nothing partial is guessed at,
    except that an unterminated string VALUE is kept as a fragment (see
    `_cut_field`). The step schema puts `action` and its arguments before
    `reasoning`, so a cut inside the trailing explanation still yields a usable
    step.

    Returns the repaired JSON text, or None if nothing salvageable survived.
    """
    start = text.find("{")
    if start == -1:
        return None
    buf = text[start:]
    n = len(buf)

    def _complete_value_end(i: int) -> int | None:
        """Index just after the value starting at i, or None if it is missing
        or cut off."""
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
            # A digit followed by '.', 'e', or an exponent sign was cut
            # mid-token; '.' or '-' can never legally follow a complete number.
            if j < n and (buf[j] == "." or (tok[-1].isdigit() and buf[j] in "eE+-")):
                return None
            # A bare integer at the very end of the buffer may be unfinished.
            if j == n and tok[-1].isdigit():
                return None
            return j
        for lit in ("true", "false", "null"):
            if buf.startswith(lit, i):
                return i + len(lit)
        return None

    # Keep only key:value pairs that arrived complete; stop at the first
    # fragment, the object's own close, or the end of the buffer.
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
            # An unterminated string value (reply hit max_tokens inside e.g. a
            # 'content' block) is kept as a fragment so callers can salvage
            # partial content; the quote and object are closed below.
            k = j + 1
            while k < n and buf[k] in " \t\r\n":
                k += 1
            if k < n and buf[k] == '"' and _complete_value_end(k) is None:
                # Keep the opening quote; escape only what follows it.
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


def merge_consecutive(messages: list[dict]) -> list[dict]:
    """Collapse adjacent same-role messages into one; a stray repeat would
    otherwise surface as an opaque Jinja error from the server."""
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
    """Rough token estimate (~3.6 chars/token for source code)."""
    return int(len(text) / 3.6) + 1


class LLMClient:
    """A thin client for one model server + model, handling retries, template
    quirks, and reasoning-model output transparently.

    Sampler parameters in `cfg.sampler` are merged into every request body
    verbatim (top_k, min_p, repeat_penalty, seed, ...); the accepted set
    differs per server, so they have no first-class config field.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        }
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        # Learned once the template rejects a system role.
        self._no_system_role = False
        # Grammar-constrained JSON support: None = not probed (assume yes),
        # True = server advertises response_format/json_schema, False = it
        # answered the probe without advertising it (such requests fall back to
        # JSON prompting).
        self._schema_supported: bool | None = None
        # "length" means cut off mid-answer: "valid JSON we never let it finish".
        self.last_finish_reason = ""
        # Calibrated chars-per-token from the server's tokenizer endpoint
        # (calibrate_tokenizer); 0.0 = uncalibrated, module heuristic applies.
        self.chars_per_token = 0.0
        # Truncation signal for streamed replies (finish_reason arrives on the
        # final SSE chunk); cleared per _chat_once call, sticky within it.
        self.stream_truncated = False
        # Chars of hidden reasoning in the last streamed attempt. chat()'s
        # empty-completion retry uses it to tell an instant-EOS flake (retry
        # identical) from a thinking runaway (retry with thinking disabled).
        self.last_reasoning_chars = 0

    def estimate_tokens(self, text: str) -> int:
        """Token estimate for fit math: the server's calibrated chars-per-token
        when warmup got one, else the module heuristic (~3.6)."""
        if self.chars_per_token:
            return int(len(text) / self.chars_per_token) + 1
        return estimate_tokens(text)

    def _context_too_long(self, body: str) -> bool:
        """True when a 400 body says the prompt overflowed the context window.
        llama.cpp/LM Studio say "exceeds ... context" / "n_ctx"; Ollama says
        "context length exceeded" or "too many tokens"."""
        low = (body or "").lower()
        return any(h in low for h in (
            "exceeds", "context length", "n_ctx", "ctx_len",
            "too many tokens", "prompt is too long", "maximum context",
            "out of range", "overfull"))

    def _post(self, payload: dict, stream: bool = False):
        """POST to the chat completions endpoint, raising a specific LLMError
        subclass on failure (connection, load failure, template, transient,
        context overflow, or other server error)."""
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
                # chat() catches this, trims the conversation, and retries.
                raise ContextOverflowError(
                    f"Server says the request exceeds its context window: "
                    f"{body[:300]}")
            raise LLMError(f"Server returned {r.status_code}: {body[:500]}")
        return r

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One embedding vector per text via `<base_url>/embeddings`, in input
        order. Used only by optional semantic search (semantic.py); no chat
        retry/template handling."""
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
        """Ask a multimodal model to describe one local image (sent as an
        image_url data URI). A text-only model will reject it; the caller
        reports that as an observation."""
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

        Retries while the model is still loading, trims old tool results so
        prompt + reply fit the context window, recovers from context overflow,
        and folds the system prompt into the first user turn if the template
        rejects a system role.
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
        empty_streak = 0        # consecutive empty streamed replies
        while True:
            attempt += 1
            # Proactive fit: llama.cpp/LM Studio reject requests whose prompt +
            # max_tokens exceeds the window, and Ollama silently truncates the
            # middle. Shrink the oldest tool-result turns until the reply
            # budget fits; only the newest RESULT matters mid-gather.
            room = self.cfg.context_tokens - max_tokens
            total = sum(self.estimate_tokens(m["content"]) for m in messages)
            if room > 0 and total > room * 0.9:
                new_messages = list(messages)
                # Drop the oldest RESULT turn with the assistant action that
                # requested it, so roles keep alternating.
                while True:
                    cur = sum(self.estimate_tokens(m["content"])
                              for m in new_messages)
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
                text = self._chat_once(messages, max_tokens, schema, stream,
                                       disable_thinking)
            except ContextOverflowError as e:
                # The server's window is smaller than our budgets assumed
                # (stock LM Studio 8192 / Ollama 4096). Shed stale middle
                # history first, then halve the reply budget, then fail.
                if not trimmed_once:
                    body_idx = [i for i, m in enumerate(messages)
                                if m["role"] == "user"
                                and m["content"].startswith("RESULT:\n")]
                    if len(body_idx) > 2:
                        drop = set(body_idx[: len(body_idx) // 2])
                        messages = merge_consecutive(
                            [m for i, m in enumerate(messages)
                             if i not in drop])
                        trimmed_once = True
                        print("[server context overflow: dropped older tool "
                              "results and retrying]", file=sys.stderr)
                        continue
                if any(m["role"] == "system" for m in messages) and \
                        len([m for m in messages if m["role"] != "system"]) > 2:
                    # No RESULT turns to shed (answer phase): drop the middle
                    # assistant turns, keeping the first user turn (map+task)
                    # and the last (latest instruction/result).
                    keep_first = next(i for i, m in enumerate(messages)
                                      if m["role"] == "user")
                    keep_last = max(i for i, m in enumerate(messages)
                                    if m["role"] == "user")
                    if keep_last - keep_first > 1:
                        mid = [i for i in range(keep_first + 1, keep_last)
                               if messages[i]["role"] == "assistant"]
                        if mid:
                            mid = set(mid)
                            messages = merge_consecutive(
                                [m for i, m in enumerate(messages)
                                 if i not in mid])
                            print("[server context overflow: shortened the "
                                  "conversation and retrying]", file=sys.stderr)
                            continue
                if not shrunk_once and max_tokens > 256:
                    # Our token estimate was optimistic (code is denser than
                    # prose). Halve the reply budget; the step schema puts the
                    # action first, so a shorter reply loses only `reasoning`.
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
                # Most likely no system role: fold it into the first user turn
                # and remember that.
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
                        f"The model server was still not ready for "
                        f"'{self.cfg.model}' after {waited:.0f}s.\n"
                        "  If you are switching models, the server may be waiting "
                        "for the previous one\n"
                        "  to unload. Check its model-management UI, or raise "
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
                continue

            if not stream or (text or "").strip():
                return text
            # An empty streamed answer is a provider flake or a thinking
            # runaway (budget spent on hidden reasoning), never a legitimate
            # answer. Retry up to twice here, then hand the "" back to the
            # callers' own fallbacks.
            empty_streak += 1
            if empty_streak > 2:
                return text
            effective_thinking_off = (disable_thinking
                                      if disable_thinking is not None
                                      else self.cfg.disable_thinking)
            if self.last_reasoning_chars and not effective_thinking_off:
                # Thinking runaway: re-rolling would think again; switch
                # thinking off instead.
                disable_thinking = True
                how = " with thinking disabled"
            else:
                how = ""
            print(f"[the model streamed an EMPTY answer (~"
                  f"{self.last_reasoning_chars:,} chars of hidden reasoning, "
                  f"finish '{self.last_finish_reason or 'stop'}'); "
                  f"retrying{how}]", file=sys.stderr)
            time.sleep(1.0)

    def _chat_once(self, messages: list[dict], max_tokens: int,
                   schema: dict | None = None, stream: bool = False,
                   disable_thinking: bool | None = None) -> str:
        """One HTTP request (no retry logic); returns the response text,
        buffered or streamed to stdout."""
        msgs = [dict(m) for m in messages]
        if self.cfg.thinking_suffix and msgs:
            msgs[-1]["content"] += "\n\n" + self.cfg.thinking_suffix
        if schema is not None and self.cfg.json_mode == "object" and msgs:
            # Hosted APIs require the word "json" in the prompt for
            # json_object mode; naming the fields improves compliance.
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
        # request_max_tokens_cap (0 = off): llama.cpp-based servers 400 when
        # max_tokens exceeds the context window, so never exceed the cap.
        cap = getattr(self.cfg, "request_max_tokens_cap", 0) or 0
        if cap and max_tokens > cap:
            payload["max_tokens"] = cap
        # Sampler parameters are merged verbatim (an unsupported one 400s the
        # request); structural keys are skipped.
        for k, v in (self.cfg.sampler or {}).items():
            if k in ("model", "messages", "stream"):
                continue
            payload[k] = v
        # Thinking-off fields are only sent to local servers: a strict hosted
        # API could 400 on them.
        if (disable_thinking if disable_thinking is not None
                else self.cfg.disable_thinking) and not _is_remote(self.cfg.base_url):
            # Two spellings: llama.cpp/LM Studio templates read
            # chat_template_kwargs.enable_thinking, Ollama's OpenAI endpoint
            # reads the flat `think`. Servers ignore the one they don't know.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["think"] = False
        if schema is not None:
            if self.cfg.json_mode == "object":
                payload["response_format"] = {"type": "json_object"}
            else:
                # Send json_schema unless the warmup probe found the server
                # answering without advertising support (LM Studio's
                # /api/v0/models `supported_parameters`); Ollama and many
                # hosted APIs 400 on it or answer in prose.
                if self._schema_supported is False:
                    fields = ", ".join(schema.get("properties", {}).keys())
                    msgs[-1]["content"] += (
                        "\n\nReply with one JSON object only, no prose. "
                        f"Fields: {fields}.\nExample of a well-formed reply:\n"
                        + _schema_example(schema))
                else:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "step", "strict": True, "schema": schema},
                    }

        t0 = time.time()
        self.stats["calls"] += 1
        # Cleared per attempt so a stale "length" from an abandoned attempt
        # cannot trigger apply()'s continuation loop.
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

        # Streaming, so the user sees progress at local-model speeds.
        out = []
        tool_opens = 0
        probe_tail = ""
        reasoning_chars = 0
        # Per-attempt diagnostics for chat()'s empty-completion retry.
        self.last_finish_reason = ""
        self.last_reasoning_chars = 0
        r = self._post(payload, stream=True)
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            # requests only decodes when the response declares a charset, and
            # text/event-stream often doesn't.
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
            if fr:
                self.last_finish_reason = fr
            if fr == "length":
                self.stream_truncated = True
            delta = d.get("content")
            rc = d.get("reasoning_content") or d.get("reasoning")
            if not delta and not rc:
                continue
            if rc:
                reasoning_chars += len(rc)       # counted, never echoed
            if delta:
                out.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()
                # Degenerate native-tool-call loop guard; the 4-char overlap
                # catches an opening tag split across chunks.
                probe = probe_tail + delta
                tool_opens += len(_NATIVE_TOOL_OPEN_RE.findall(probe))
                probe_tail = probe[-4:]
                if tool_opens >= _MAX_STREAM_TOOL_OPENS:
                    print("\n[stopped the reply: the model is repeating "
                          "tool-call syntax instead of answering; no tools "
                          "can run in this phase]", file=sys.stderr)
                    try:
                        r.close()
                    except Exception:
                        pass
                    break
        sys.stdout.write("\n")
        self.stats["seconds"] += time.time() - t0
        self.last_reasoning_chars = reasoning_chars
        return strip_reasoning("".join(out))

    @staticmethod
    def _schema_has_big_fields(schema: dict | None) -> bool:
        """True when the schema has a string field large enough to hold real
        file content. STEP_SCHEMA caps `content` at 6000 chars, so a cut step
        reply died inside 'reasoning' and a bigger budget cannot help; such
        schemas skip budget escalation."""
        BIG_FIELD_MIN = 8000   # maxLength above which a string field is "large"
        props = (schema or {}).get("properties", {})
        return any(
            p.get("type") == "string" and int(p.get("maxLength") or 0) > BIG_FIELD_MIN
            for p in props.values())

    def chat_json(self, messages: list[dict], schema: dict, max_tokens: int,
                  disable_thinking: bool | None = None) -> dict:
        """Send a grammar-constrained chat request and return the parsed
        JSON object.

        The common failure is not malformed output but a reply cut off at
        max_tokens, so a truncated attempt retries with a larger budget (see
        `_schema_has_big_fields` for when it does not), or salvages the
        complete prefix of the object.
        """
        retries = 2
        budget = max_tokens
        # Set when a truncated reply was salvaged; `truncated_field` names the
        # schema field the cut landed in ("" = nothing cut, or benign) so
        # agent.gather can refuse to dispatch a step whose payload was cut.
        self.step_truncated = False
        self.truncated_field = ""
        # Cuts of an answer-style schema; after the second, accept the salvage.
        step_cut = 0
        # One re-ask per call with an explicit no-reasoning instruction for
        # "whole budget spent on a preamble, no JSON at all".
        trunc_nudged = False
        # Ceiling for budget escalation, derived from the caller's config.
        from .config import is_hosted
        if is_hosted(self.cfg):
            # Hosted APIs have a provider-side output ceiling (~8k tokens on
            # deepseek-chat) that escalation cannot beat; honour it when known.
            mot = int(getattr(self.cfg, "max_output_tokens", 0) or 0)
            if mot > 0:
                cap = max(mot, max_tokens)
            else:
                cap = max(65536, self.cfg.max_tokens_answer,
                          self.cfg.max_tokens_step * 8, max_tokens * 8)
        else:
            # Never escalate past what the context window can hold.
            usable = self.cfg.context_tokens - self.cfg.reserve_tokens
            cap = min(max(2048, self.cfg.reserve_tokens,
                          self.cfg.max_tokens_answer, max_tokens * 4),
                      max(usable, max_tokens))
        cap = max(cap, max_tokens)
        for attempt in range(retries + 1):
            raw = self.chat(messages, max_tokens=budget, schema=schema,
                            disable_thinking=disable_thinking)
            truncated = self.last_finish_reason == "length"
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.split("\n", 1)[-1] if "\n" in text else text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Salvage a JSON object surrounded by prose. Scan every '{'
                # with raw_decode (thinking models emit several candidate
                # objects first); requiring an "action"/"edits" key keeps a
                # stray {"note": ...} from being taken for the reply.
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
                        if i > 0 or _end + i < len(text):
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
                    # Some servers (DeepSeek in json_object mode) emit
                    # single-quoted Python dicts. Convert only when the
                    # fragment has no double quotes, so apostrophes inside
                    # real strings are not mangled.
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
                                # Args precede 'reasoning', so a cut usually
                                # lands in the essay. A long `pattern` or
                                # `content` can also be cut; record which
                                # field so agent.gather can refuse it.
                                self.truncated_field = _cut_field(text, obj,
                                                                  schema)
                                print("[reply was cut off; recovered the "
                                      "action from the partial reply]",
                                      file=sys.stderr)
                                self.step_truncated = True
                                return obj
                            # Answer-style schema: the salvage is partial, so
                            # do not act on half a file; escalate instead.
                            step_cut += 1
                            if step_cut >= 2:
                                print("[reply was cut off mid-content; last "
                                      "attempt will be salvaged whole]",
                                      file=sys.stderr)
                                self.step_truncated = True
                                self.truncated_field = _cut_field(text, obj,
                                                                  schema)
                                return obj
                if truncated:
                    # Step schemas: the reply died before any usable action,
                    # so a bigger budget cannot help. Re-ask once with a
                    # no-reasoning nudge, then fail.
                    if not self._schema_has_big_fields(schema):
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
        # A cut-off last attempt, not a grammar failure, is what went wrong.
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
                f"  Last partial reply: {raw[-160:]}")
        raise LLMError(
            f"Model would not produce valid JSON after {retries + 1} tries.\n"
            "  If this is a hosted OpenAI-compatible API (DeepSeek, OpenAI, "
            "...), set\n"
            "  json_mode = \"object\" under [server] -- hosted APIs reject "
            "json_schema.\n"
            "  If it's a local server, the runtime may be ignoring the JSON "
            "schema\n"
            "  (common with MLX builds; llama.cpp/GGUF supports it).\n"
            f"  Last reply: {raw[:300]}")

    def loaded_models(self) -> dict[str, str] | None:
        """{model_id: state} from LM Studio's native /api/v0/models (the
        OpenAI /v1/models lists everything downloaded, not what is resident).
        None when the server lacks the endpoint."""
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

        Servers spell the name differently (`id`/`key` with an optional
        `@q4_k_m` variant suffix on LM Studio, `name` on Ollama). Exact matches
        win, then a match ignoring the `@` suffix.
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
        """Extract size/context from the entry for `cfg.model` in a model list.

        Absorbs both extensions of the OpenAI /v1/models shape:
          * LM Studio v1: `size_bytes`, `loaded_instances[].config.context_length`,
            `max_context_length`.
          * Lemonade: `size` in GB, `context_length`, `max_context_window`.
        """
        m = self._pick_model(entries, self.cfg.model)
        if not m:
            return {}

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
            loaded_ctx = _positive_int(
                (loaded[0].get("config") or {}).get("context_length"))
        # Lemonade's recipe_options carry the effective ctx_size (with saved
        # overrides); prefer it over the top-level context_length, which some
        # versions leave at the registry default.
        recipe = m.get("recipe_options") or {}
        ctx = (loaded_ctx or _positive_int(recipe.get("ctx_size"))
               or _positive_int(m.get("context_length"))
               or _positive_int(recipe.get("num_ctx")))
        if ctx:
            got["context_length"] = ctx
        max_ctx = _positive_int(m.get("max_context_length")
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
        """Context a currently loaded Lemonade model was launched with, from its
        health endpoint (the model list's `context_length` can lag behind).
        Tries both prefixes because older builds expose /api/v0/health."""
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
            ctx = (_positive_int(opts.get("ctx_size"))
                   or _positive_int(entry.get("context_length")))
            max_ctx = _positive_int(entry.get("max_context_window"))
            if ctx or max_ctx:
                return ctx, max_ctx
        return None, None

    def _lemonade_options_context(self, root: str) -> int | None:
        """The ctx_size Lemonade would use for a load right now, from
        /v1/models/{id}/options (`resolved_ctx_size` covers automatic sizing).
        Absent on older versions, hence the silent failure."""
        url = (root + "/v1/models/" + quote(str(self.cfg.model), safe="")
               + "/options")
        try:
            r = requests.get(url, headers=self.headers, timeout=5)
            if r.status_code != 200:
                return None
            data = r.json()
        except Exception:
            return None
        ctx = _positive_int(data.get("resolved_ctx_size"))
        if ctx:
            return ctx
        for key in ("effective", "saved"):
            opts = data.get(key) or {}
            if isinstance(opts, dict):
                ctx = _positive_int(opts.get("ctx_size"))
                if ctx:
                    return ctx
        return None

    def model_info(self) -> dict | None:
        """Metadata for `cfg.model`: any of `size_bytes`, `context_length`,
        `max_context_length`, plus `source`; None when nothing answers. Used by
        auto-fit to size the context window from the machine's VRAM.

        Tried in order, stopping once both size and context are known:
          * LM Studio v1 `/api/v1/models`
          * OpenAI-compatible `/v1/models` (Lemonade adds `size` in GB,
            `context_length`, `max_context_window`)
          * LM Studio v0 `/api/v0/models` (older builds)
          * Ollama `/api/show` (also implemented by Lemonade)

        Every probe is best-effort; failures just yield nothing.
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

        # Lemonade's loaded/saved recipe options beat a list entry that may
        # carry the registry default window.
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

        # Ollama /api/show reports the weights size and context length.
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
                    # Keys are architecture-prefixed (llama., qwen2., gemma.,
                    # ...): accept any *.context_length, taking the largest.
                    max_ctx = max(
                        (v for v in (_positive_int(x)
                                     for k, x in info.items()
                                     if k.endswith(".context_length")) if v),
                        default=0) or None
                    ctx = _positive_int(num_ctx) or max_ctx
                    if ctx:
                        got["context_length"] = ctx
                    if max_ctx:
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

    def warmup(self) -> None:
        """Load the model with one tiny generation before real work begins, so
        a load failure or model-switch wait surfaces once, up front."""
        state = self.loaded_models()
        if state is not None:
            resident = [m for m, st in state.items() if st == "loaded"]
            if resident and self.cfg.model not in resident:
                print(f"[server currently has {', '.join(resident)} loaded; "
                      f"requesting '{self.cfg.model}']", file=sys.stderr)
        # LM Studio's native API lists `supported_parameters` per model; if
        # that list omits response_format/json_schema, use plain JSON
        # prompting instead of getting prose back from a "constrained" call.
        # A remembered verdict (server_modes.json) skips the probe.
        if self._schema_supported is None and self.cfg.json_mode != "object":
            cached = _load_mode_cache(self.cfg.base_url, self.cfg.model)
            if cached is not None:
                self._schema_supported = cached
                if not cached:
                    print("[remembered server capability: no constrained "
                          "JSON; using plain JSON prompting]", file=sys.stderr)
            else:
                try:
                    root = _server_root(self.cfg.base_url)
                    r = requests.get(root + "/api/v0/models",
                                     headers=self.headers, timeout=5)
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
                                    print("[server does not support constrained "
                                          "JSON; using plain JSON prompting]",
                                          file=sys.stderr)
                                break
                except Exception:
                    pass   # best-effort probe
                if self._schema_supported is not None:
                    _save_mode_cache(self.cfg.base_url, self.cfg.model,
                                     self._schema_supported)
        t0 = time.time()
        self.chat([{"role": "user", "content": "Reply with the single word: ok"}],
                  max_tokens=5)
        took = time.time() - t0
        if took > 5:
            print(f"[model ready after {took:.0f}s]", file=sys.stderr)
        # The model is resident now: one cheap request can measure the real
        # chars-per-token so the fit math stops guessing.
        self.calibrate_tokenizer()

    def calibrate_tokenizer(self) -> None:
        """Derive chars-per-token from the server's tokenizer endpoint
        (LM Studio `/api/v0/tokenize`, llama.cpp `/tokenize`). Best-effort:
        with no endpoint the module heuristic (~3.6) stays in effect."""
        sample = _TOKENIZER_SAMPLE
        root = _server_root(self.cfg.base_url)
        for url, payload in (
                (root + "/api/v0/tokenize",
                 {"model": self.cfg.model, "prompt": sample,
                  "add_special_tokens": False}),
                (root + "/tokenize", {"content": sample})):
            try:
                r = requests.post(url, headers=self.headers, json=payload,
                                  timeout=10)
                if r.status_code != 200:
                    continue
                d = r.json()
                n = _positive_int(d.get("count")) or \
                    (len(d["tokens"]) if isinstance(d.get("tokens"), list) else 0)
                if n > 0:
                    cpt = len(sample) / n
                    if 1.0 <= cpt <= 8.0:
                        self.chars_per_token = cpt
                        return
            except Exception:
                continue

    def probe_json_modes(self) -> list[tuple[str, bool, str]]:
        """`pedacito config --doctor`: one tiny request per structured-output
        mode. Returns [(mode, ok, detail)] for json_schema (strict grammar),
        json_object, and plain prompting; `ok` means the reply parsed as the
        expected JSON object. A json_schema verdict is remembered for runs."""
        user = ("Reply with one JSON object only, no prose: "
                '{"action": "done", "file": "", "reasoning": "probe"}')
        results: list[tuple[str, bool, str]] = []
        for mode, fmt in (
                ("json_schema",
                 {"type": "json_schema",
                  "json_schema": {"name": "step", "strict": True,
                                  "schema": _PROBE_SCHEMA}}),
                ("json_object", {"type": "json_object"}),
                ("prompt", None)):
            payload = {
                "model": self.cfg.model,
                "messages": [{"role": "user", "content": user}],
                "temperature": self.cfg.temperature,
                "max_tokens": 120,
                "stream": False,
            }
            if fmt is not None:
                payload["response_format"] = fmt
            try:
                data = self._post(payload).json()
                text = (extract_content(data["choices"][0]["message"]) or "").strip()
                i, j = text.find("{"), text.rfind("}")
                obj = json.loads(text[i:j + 1]) if i != -1 and j > i else None
                if isinstance(obj, dict) and obj.get("action") == "done":
                    results.append((mode, True, "valid JSON object"))
                else:
                    results.append((mode, False,
                                    f"reply was not the expected object: "
                                    f"{text[:80]!r}"))
            except LLMError as e:
                results.append((mode, False, str(e).split("\n")[0][:160]))
            except Exception as e:
                results.append((mode, False, f"{type(e).__name__}: {e}"[:160]))
        if self.cfg.json_mode != "object":
            for mode, ok, _detail in results:
                if mode == "json_schema":
                    self._schema_supported = ok
                    _save_mode_cache(self.cfg.base_url, self.cfg.model, ok)
        return results

    def check(self, warn: bool = True) -> str:
        """Verify the server is reachable and return its comma-joined model ids;
        with `warn`, print a warning if the configured model is not listed."""
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
