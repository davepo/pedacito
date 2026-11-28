"""Native Anthropic ``/v1/messages`` transport.

Anthropic exposes an OpenAI-compatible ``/v1/chat/completions`` surface, but it
lacks the things Pedacito relies on most: prompt caching, streaming with usage
reporting, thinking, tools and images all work on the NATIVE ``/v1/messages``
API and are degraded or absent on the compatibility shim. This module builds
native requests and parses native responses, while the callers (``llm.py``)
keep the OpenAI-shaped message representation right up to the wire.

Everything here is a pure function of its inputs; it has no state and no
network I/O. ``LLMClient`` stays the only thing that talks HTTP, and the
outbound secret guard, replay digest, telemetry and cost accounting all still
run on the OpenAI-shaped messages *before* they are translated.

Only dependency is the standard library (plus ``requests`` is not even needed
here -- the caller passes an already-opened response).
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

# Reasoning-effort -> Anthropic thinking budget tokens. The budgets are the
# provider's documented tiers; "default" sends no thinking block at all.
_EFFORT_BUDGET = {"low": 2048, "medium": 6144, "high": 12288}

# Anthropic allows at most 4 cache_control breakpoints per request.
_MAX_BREAKPOINTS = 4

# Anthropic stop_reason -> OpenAI finish_reason (the vocabulary the harness,
# the replay flags and the continuation machinery already speak).
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
}

_ANTHROPIC_VERSION = "2023-06-01"
_EXTENDED_TTL_BETA = "extended-cache-ttl-2025-04-11"


class StreamCancelled(Exception):
    """``parse_sse_stream`` saw ``cancel_check()`` return True."""


class StreamError(Exception):
    """An Anthropic ``error`` event arrived mid-stream."""

    def __init__(self, err_type: str = "", message: str = ""):
        super().__init__(message or err_type or "Anthropic stream error")
        self.err_type = err_type or ""


def _pos(value) -> int:
    """First positive int for a usage field, else 0."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def is_anthropic_native(cfg) -> bool:
    """True when this config should use the native ``/v1/messages`` transport.

    The flag defaults on; it is the one escape hatch back to the
    OpenAI-compatible surface. The host must be Anthropic's (or the user must
    have labelled the provider ``anthropic``, which is how a loopback test
    double or an Anthropic-compatible gateway opts in).
    """
    if not getattr(cfg, "anthropic_native", True):
        return False
    try:
        host = (urlparse(getattr(cfg, "base_url", "") or "").hostname or "").lower()
    except ValueError:
        host = ""
    provider = str(getattr(cfg, "provider", "") or "").lower()
    return "api.anthropic.com" in host or provider == "anthropic"


def messages_endpoint(base_url: str) -> str:
    """Map a base URL to its native ``<origin>/v1/messages`` endpoint.

    ``https://api.anthropic.com``, ``https://api.anthropic.com/v1`` and
    ``https://api.anthropic.com/v1/messages`` all land on the same route, and a
    loopback test URL (``http://127.0.0.1:PORT/v1``) cleanly becomes
    ``http://127.0.0.1:PORT/v1/messages``.
    """
    base = (base_url or "").rstrip("/")
    if not base:
        return ""
    parsed = urlparse(base)
    if parsed.netloc:
        origin = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path.rstrip("/")
    else:
        origin, path = "", base
    if path in ("", "/v1"):
        return origin + "/v1/messages"
    if path.endswith("/messages"):
        return origin + path
    return origin + path + "/messages"


def build_headers(api_key: str, *, ttl: str = "5m",
                  idempotency_key: str | None = None) -> dict[str, str]:
    """Anthropic request headers: key, version, JSON content type, and the
    extended-TTL beta flag (only for the ``1h`` cache TTL).

    ``idempotency_key`` is attached ONLY when the caller already confirmed the
    provider documents idempotency support (``ProviderCapabilities.idempotency``
    is True); an empty/None key omits the header, never guessed."""
    headers = {
        "x-api-key": api_key or "",
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    if ttl == "1h":
        headers["anthropic-beta"] = _EXTENDED_TTL_BETA
    if idempotency_key:
        headers["idempotency-key"] = str(idempotency_key)
    return headers


def _cache_control(ttl: str) -> dict:
    """The Anthropic ``cache_control`` marker for a breakpoint.

    ``5m`` is the provider default (no ``ttl`` key needed); ``1h`` requires the
    extended-TTL beta header that ``build_headers`` adds for that TTL.
    """
    if ttl == "1h":
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _text_of(content) -> str:
    """Text of an OpenAI ``content`` field (string or content-part list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text") for p in content
                 if isinstance(p, dict) and p.get("type") == "text"
                 and isinstance(p.get("text"), str)]
        return "\n".join(parts)
    return ""


def _image_source(url) -> dict | None:
    """Anthropic base64 image source from an OpenAI ``data:`` image URL, or
    None when the URL is not a base64 data URI."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, _, data = url.partition(",")
    spec = header[len("data:"):]
    if "base64" not in spec.lower() or not data:
        return None
    parts = spec.split(";")
    mime = parts[0] if parts and parts[0] else "image/png"
    return {"type": "base64", "media_type": mime, "data": data}


def _content_part(part) -> dict | None:
    """One OpenAI content part -> one Anthropic content block (or None)."""
    if not isinstance(part, dict):
        return None
    ptype = part.get("type")
    if ptype == "text":
        return {"type": "text", "text": part.get("text") or ""}
    if ptype == "image_url":
        source = _image_source((part.get("image_url") or {}).get("url"))
        if source:
            return {"type": "image", "source": source}
    return None


def _tool_arguments(fn: dict) -> dict:
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (TypeError, ValueError):
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}
    return args if isinstance(args, dict) else {}


def _reasoning_blocks(message: dict) -> list[dict]:
    """Opaque thinking blocks a prior assistant turn sent back to us.

    Only complete blocks are replayed: Anthropic rejects unsigned thinking
    blocks, so a bare `reasoning_content` string from another transport is
    left out rather than fabricated into a thinking block.
    """
    out: list[dict] = []
    for item in message.get("reasoning_items") or ():
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "thinking":
            text = item.get("thinking")
            signature = item.get("signature")
            if isinstance(text, str) and text and signature:
                out.append({"type": "thinking", "thinking": text,
                            "signature": str(signature)})
        elif itype == "redacted_thinking" and item.get("data"):
            out.append({"type": "redacted_thinking",
                        "data": str(item["data"])})
    return out


def _message_blocks(message: dict) -> list[dict]:
    """Anthropic content blocks for one non-system OpenAI message."""
    role = message.get("role")
    content = message.get("content")
    blocks: list[dict] = []
    # Prior assistant turns carry opaque thinking blocks (text plus the
    # provider's signature).  They must go back verbatim on the next
    # tool-call turn, but never inside the visible text content.
    if role == "assistant":
        blocks.extend(_reasoning_blocks(message))
    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for part in content:
            block = _content_part(part)
            if block is not None:
                blocks.append(block)
    if role == "assistant":
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            blocks.append({
                "type": "tool_use",
                "id": call.get("id") or "",
                "name": fn.get("name") or call.get("name") or "",
                "input": _tool_arguments(fn),
            })
    elif role == "tool":
        blocks = [{
            "type": "tool_result",
            "tool_use_id": message.get("tool_call_id") or "",
            "content": _text_of(content),
        }]
    return blocks


def translate_messages(messages: list[dict], *, cache_enabled: bool = True,
                       cache_ttl: str = "5m") -> tuple[list[dict], list[dict]]:
    """Translate OpenAI-shaped messages to Anthropic ``(system, messages)``.

    Returns ``(system_blocks, translated_messages)`` where ``system_blocks`` is
    a list of ``{"type": "text", "text": ...}`` blocks and
    ``translated_messages`` is the alternating user/assistant conversation with
    native content blocks (``tool_use`` / ``tool_result`` / ``image``).

    With ``cache_enabled``, a ``cache_control`` marker is placed on the last
    system block, the first user turn's last block (the ``user0`` repo-map
    prefix), and -- once there are three user turns -- the penultimate user
    turn, never exceeding 4 breakpoints in total.
    """
    system_texts = []
    conversation: list[dict] = []  # {"role": ..., "blocks": [...]}
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            system_texts.append(_text_of(message.get("content")))
            continue
        blocks = _message_blocks(message)
        if not blocks:
            continue
        conv_role = "user" if role == "tool" else role
        if conversation and conversation[-1]["role"] == conv_role:
            conversation[-1]["blocks"].extend(blocks)
        else:
            conversation.append({"role": conv_role, "blocks": blocks})

    system_blocks = []
    for i, text in enumerate(system_texts):
        block = {"type": "text", "text": text}
        if cache_enabled and i == len(system_texts) - 1:
            block["cache_control"] = _cache_control(cache_ttl)
        system_blocks.append(block)

    if cache_enabled:
        used = 1 if system_blocks else 0
        user_positions = [i for i, c in enumerate(conversation)
                          if c["role"] == "user"]
        breakpoints: set[int] = set()
        if user_positions:
            breakpoints.add(user_positions[0])
            if len(user_positions) >= 3:
                breakpoints.add(user_positions[-2])
        for pos in sorted(breakpoints):
            if used >= _MAX_BREAKPOINTS:
                break
            blocks = conversation[pos]["blocks"]
            if blocks:
                blocks[-1]["cache_control"] = _cache_control(cache_ttl)
                used += 1

    translated = [{"role": c["role"], "content": c["blocks"]}
                  for c in conversation]
    return system_blocks, translated


def _forced_tool_name(tool_choice) -> str | None:
    """The function name a forced OpenAI ``tool_choice`` names, else None."""
    if not isinstance(tool_choice, dict):
        return None
    fn = tool_choice.get("function")
    if isinstance(fn, dict):
        return fn.get("name") or tool_choice.get("name")
    if tool_choice.get("type") == "function":
        return tool_choice.get("name")
    return tool_choice.get("name")


def translate_tools(openai_tools: list[dict] | None, tool_choice=None,
                    *, thinking_on: bool = False) -> tuple[list[dict] | None, dict | None]:
    """Convert OpenAI function tools to Anthropic tools and map a forced
    ``tool_choice``.

    A forced named choice becomes ``{"type": "tool", "name": ...}``; when
    ``thinking_on`` it is downgraded to ``{"type": "auto"}`` because Anthropic
    rejects forced tool selection while extended thinking is enabled.
    """
    tools: list[dict] | None = None
    if openai_tools:
        out = []
        for tool in openai_tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
            out.append({
                "name": fn.get("name") or tool.get("name") or "",
                "description": fn.get("description") or tool.get("description") or "",
                "input_schema": fn.get("parameters") or tool.get("parameters")
                or {"type": "object", "properties": {}},
            })
        if out:
            tools = out

    name = _forced_tool_name(tool_choice)
    choice: dict | None = None
    if name:
        choice = {"type": "auto"} if thinking_on else {"type": "tool", "name": name}
    return tools, choice


def build_payload(cfg, messages: list[dict], max_tokens: int, *,
                  stream: bool = False, disable_thinking: bool | None = None,
                  tools: list[dict] | None = None, tool_choice=None,
                  temperature=None, cache_enabled: bool = True,
                  cache_ttl: str = "5m") -> dict:
    """Build a valid native ``/v1/messages`` request body from OpenAI inputs."""
    want_off = (disable_thinking if disable_thinking is not None
                else bool(getattr(cfg, "disable_thinking", False)))
    system_blocks, translated = translate_messages(
        messages, cache_enabled=cache_enabled, cache_ttl=cache_ttl)

    effort = str(getattr(cfg, "reasoning_effort", "default")
                 or "default").strip().lower()
    thinking = None
    if not want_off and effort in _EFFORT_BUDGET:
        budget = _EFFORT_BUDGET[effort]
        thinking = {"type": "enabled", "budget_tokens": budget}
        # Anthropic requires max_tokens > budget_tokens.
        if max_tokens <= budget:
            max_tokens = budget + 1

    anthropic_tools, anthropic_choice = translate_tools(
        tools, tool_choice, thinking_on=thinking is not None)

    body: dict = {
        "model": getattr(cfg, "model", "") or "",
        "max_tokens": int(max_tokens),
        "messages": translated,
        "stream": bool(stream),
    }
    if system_blocks:
        body["system"] = system_blocks
    if anthropic_tools:
        body["tools"] = anthropic_tools
    if anthropic_choice:
        body["tool_choice"] = anthropic_choice
    if thinking is not None:
        body["thinking"] = thinking
    elif temperature is not None:
        body["temperature"] = temperature
    return body


def _map_stop_reason(stop_reason: str | None) -> str:
    return _STOP_REASON_MAP.get(stop_reason or "", stop_reason or "")


def parse_json_response(data: dict) -> dict:
    """Normalise one buffered ``/v1/messages`` response.

    Returns ``{text, reasoning, tool_calls, finish_reason, usage}`` where
    ``tool_calls`` is OpenAI-like ``{id, name, arguments_json}`` and ``usage``
    carries the Pedacito accounting keys (prompt/completion/cached/write).
    """
    data = data or {}
    content = data.get("content") or []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_items: list[dict] = []
    tool_calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text") or "")
        elif btype == "thinking":
            thinking = block.get("thinking") or ""
            reasoning_parts.append(thinking)
            item = {"type": "thinking", "thinking": thinking}
            if block.get("signature"):
                item["signature"] = str(block["signature"])
            reasoning_items.append(item)
        elif btype == "redacted_thinking" and block.get("data"):
            reasoning_items.append({"type": "redacted_thinking",
                                    "data": str(block["data"])})
        elif btype == "tool_use":
            raw = block.get("input") or {}
            args = raw if isinstance(raw, str) else json.dumps(raw)
            tool_calls.append({
                "id": block.get("id") or "",
                "name": block.get("name") or "",
                "arguments_json": args,
            })
    usage = data.get("usage") or {}
    input_tokens = _pos(usage.get("input_tokens"))
    output_tokens = _pos(usage.get("output_tokens"))
    cache_read = _pos(usage.get("cache_read_input_tokens"))
    cache_creation = _pos(usage.get("cache_creation_input_tokens"))
    return {
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "reasoning_items": reasoning_items,
        "tool_calls": tool_calls,
        "finish_reason": _map_stop_reason(data.get("stop_reason")),
        "usage": {
            "prompt_tokens": input_tokens + cache_read + cache_creation,
            "completion_tokens": output_tokens,
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_creation,
        },
    }


def parse_sse_stream(resp, *, on_text=None, on_reasoning=None,
                     cancel_check=None) -> dict:
    """Parse an Anthropic SSE stream into the same dict as
    :func:`parse_json_response`.

    Handles ``message_start`` (usage), ``content_block_start`` /
    ``content_block_delta`` (``text_delta``, ``thinking_delta``,
    ``input_json_delta``), ``message_delta`` (stop reason + output tokens),
    ``message_stop`` and ``error``. ``on_text`` / ``on_reasoning`` are called
    with each delta; ``cancel_check`` (returning truthy) aborts with
    :class:`StreamCancelled`.
    """
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_uses: dict[int, dict] = {}
    thinking_blocks: dict[int, dict] = {}
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_creation = 0
    stop_reason = ""

    for raw in resp.iter_lines(decode_unicode=True):
        if cancel_check is not None and cancel_check():
            raise StreamCancelled()
        if not raw:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        if raw.startswith(":"):
            continue                       # SSE comment / keep-alive
        if not raw.startswith("data:"):
            continue                       # "event:" line; type is in the data
        body = raw[5:].lstrip()
        if body.strip() == "[DONE]":
            break
        try:
            obj = json.loads(body)
        except (TypeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        etype = obj.get("type") or ""
        if etype == "message_start":
            usage = (obj.get("message") or {}).get("usage") or {}
            input_tokens = max(input_tokens, _pos(usage.get("input_tokens")))
            cache_read = max(cache_read, _pos(usage.get("cache_read_input_tokens")))
            cache_creation = max(cache_creation,
                                 _pos(usage.get("cache_creation_input_tokens")))
        elif etype == "content_block_start":
            index = obj.get("index", 0)
            block = obj.get("content_block") or {}
            if block.get("type") == "tool_use":
                tool_uses[index] = {"id": block.get("id") or "",
                                    "name": block.get("name") or "",
                                    "json": []}
            elif block.get("type") == "thinking":
                thinking_blocks[index] = {
                    "type": "thinking",
                    "thinking": block.get("thinking") or "",
                    "signature": block.get("signature") or "",
                }
            elif block.get("type") == "redacted_thinking" and block.get("data"):
                thinking_blocks[index] = {
                    "type": "redacted_thinking",
                    "data": str(block["data"]),
                }
        elif etype == "content_block_delta":
            delta = obj.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = delta.get("text") or ""
                if text:
                    text_parts.append(text)
                    if on_text is not None:
                        on_text(text)
            elif dtype == "thinking_delta":
                text = delta.get("thinking") or ""
                index = obj.get("index", 0)
                entry = thinking_blocks.setdefault(
                    index, {"type": "thinking", "thinking": "",
                            "signature": ""})
                entry["thinking"] = (entry.get("thinking") or "") + text
                if text:
                    reasoning_parts.append(text)
                    if on_reasoning is not None:
                        on_reasoning(text)
            elif dtype == "signature_delta":
                index = obj.get("index", 0)
                entry = thinking_blocks.setdefault(
                    index, {"type": "thinking", "thinking": "",
                            "signature": ""})
                entry["signature"] = (
                    (entry.get("signature") or "")
                    + str(delta.get("signature") or ""))
            elif dtype == "input_json_delta":
                partial = delta.get("partial_json") or ""
                index = obj.get("index", 0)
                tool_uses.setdefault(index, {"id": "", "name": "", "json": []})
                tool_uses[index]["json"].append(partial)
        elif etype == "message_delta":
            stop_reason = (obj.get("delta") or {}).get("stop_reason") or stop_reason
            usage = obj.get("usage") or {}
            output_tokens = max(output_tokens, _pos(usage.get("output_tokens")))
            cache_read = max(cache_read, _pos(usage.get("cache_read_input_tokens")))
            cache_creation = max(cache_creation,
                                 _pos(usage.get("cache_creation_input_tokens")))
        elif etype == "message_stop":
            break
        elif etype == "error":
            err = obj.get("error") or {}
            raise StreamError(err.get("type") or "",
                              err.get("message") or "")

    tool_calls = []
    for index in sorted(tool_uses):
        entry = tool_uses[index]
        if entry.get("id") or entry.get("name") or entry.get("json"):
            tool_calls.append({
                "id": entry.get("id") or "",
                "name": entry.get("name") or "",
                "arguments_json": "".join(entry.get("json") or []),
            })

    prompt_tokens = input_tokens + cache_read + cache_creation
    reasoning_items = [
        block for _index, block in sorted(thinking_blocks.items())
        if block.get("type") == "redacted_thinking"
        or (block.get("thinking") or block.get("signature"))
    ]
    return {
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "reasoning_items": reasoning_items,
        "tool_calls": tool_calls,
        "finish_reason": _map_stop_reason(stop_reason),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_creation,
        },
    }
