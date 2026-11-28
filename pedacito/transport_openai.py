"""OpenAI Responses API (``/v1/responses``) transport.

Opt-in transport for reasoning models (``o3`` / ``o4`` / ``gpt-5`` series)
selected via ``cfg.openai_responses = True`` or
``cfg.provider = "openai-responses"``. Translates OpenAI Chat Completions
messages and tools into ``instructions`` + ``input``, supports
``reasoning: {"effort": ...}``, ``prompt_cache_key``, ``previous_response_id``,
and preserves encrypted reasoning items across multi-turn tool loops while
extracting cached and reasoning token usage.

Everything here is a pure function of its inputs; it has no state and no
network I/O. ``LLMClient`` remains the sole owner of HTTP, retries, outbound
secret scanning, replay digests, streaming telemetry, and cost accounting.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse


class StreamCancelled(Exception):
    """``parse_sse_stream`` saw ``cancel_check()`` return True."""


class StreamError(Exception):
    """An OpenAI Responses ``error`` event arrived mid-stream."""

    def __init__(self, err_type: str = "", message: str = ""):
        super().__init__(message or err_type or "OpenAI Responses stream error")
        self.err_type = err_type or ""


def _pos(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def is_openai_responses(cfg) -> bool:
    """True when this config opts into the OpenAI ``/v1/responses`` API."""
    provider = str(getattr(cfg, "provider", "") or "").strip().lower()
    if provider == "openai-responses":
        return True
    return bool(getattr(cfg, "openai_responses", False))


def responses_endpoint(base_url: str) -> str:
    """Map a base URL to ``<origin>/v1/responses``."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return ""
    parsed = urlparse(base)
    if parsed.netloc:
        origin = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path.rstrip("/")
    else:
        origin, path = "", base
    if path in ("", "/v1"):
        return origin + "/v1/responses"
    if path.endswith("/responses"):
        return origin + path
    return origin + path + "/responses"


def models_endpoint(base_url: str) -> str:
    """Map a base URL to ``<origin>/v1/models``."""
    ep = responses_endpoint(base_url)
    if ep.endswith("/responses"):
        return ep[:-len("/responses")] + "/models"
    return (base_url or "").rstrip("/") + "/models"


def build_headers(api_key: str, idempotency_key: str | None = None) -> dict[str, str]:
    """OpenAI Responses request headers.

    ``idempotency_key`` is attached ONLY when the caller already confirmed the
    provider documents idempotency support (``ProviderCapabilities.idempotency``
    is True); an empty/None key means the header is omitted, never guessed.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if idempotency_key:
        headers["Idempotency-Key"] = str(idempotency_key)
    return headers


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text")
            for p in content
            if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text")
            and isinstance(p.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def _translate_user_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[dict] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                parts.append({"type": "input_text", "text": part.get("text") or ""})
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    parts.append({"type": "input_image", "image_url": url})
        return parts or ""
    return str(content or "")


def _tool_args_json(fn: dict) -> str:
    args = fn.get("arguments")
    if isinstance(args, str):
        return args
    if isinstance(args, dict):
        return json.dumps(args)
    return "{}"


def _reasoning_items(message: dict) -> list[dict]:
    """Opaque reasoning carried by one assistant turn.

    Responses items (`reasoning_items`, usually encrypted) are replayed as
    returned by the API.  A chat-completions `reasoning_content` string has no
    Responses item id or signature, so it is carried as a summary-only item
    for gateways that understand it; either way it stays out of visible text.
    """
    items = list(message.get("reasoning_items")
                 or message.get("encrypted_reasoning") or ())
    out = [dict(item) for item in items if isinstance(item, dict)]
    if not out:
        content = message.get("reasoning_content")
        if isinstance(content, str) and content:
            out.append({
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": content}],
            })
    return out


def translate_messages(
    messages: list[dict],
    *,
    encrypted_reasoning_items: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Translate OpenAI Chat Completions messages into ``(instructions, input_items)``."""
    system_texts: list[str] = []
    input_items: list[dict] = []

    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _text_of(content)
            if text:
                system_texts.append(text)
            continue
        if role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": str(message.get("tool_call_id") or ""),
                "output": _text_of(content),
            })
            continue
        if role == "assistant":
            input_items.extend(_reasoning_items(message))
            text = _text_of(content)
            if text:
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                })
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                input_items.append({
                    "type": "function_call",
                    "call_id": str(call.get("id") or ""),
                    "name": str(fn.get("name") or call.get("name") or ""),
                    "arguments": _tool_args_json(fn),
                })
            continue
        # User (or developer/other) message
        tcontent = _translate_user_content(content)
        if tcontent or tcontent == "":
            if isinstance(tcontent, str) and not tcontent:
                continue
            input_items.append({
                "type": "message",
                "role": role or "user",
                "content": tcontent,
            })

    if encrypted_reasoning_items:
        # Insert any caller-supplied encrypted reasoning items right before the
        # trailing assistant/tool items if not already attached to a message.
        existing_ids = {
            item.get("id") for item in input_items
            if isinstance(item, dict) and item.get("type") == "reasoning" and item.get("id")
        }
        to_prepend = [
            dict(item) for item in encrypted_reasoning_items
            if isinstance(item, dict) and (not item.get("id") or item.get("id") not in existing_ids)
        ]
        if to_prepend:
            # Place before the last assistant/function_call_output stretch when possible
            insert_at = len(input_items)
            for idx in range(len(input_items) - 1, -1, -1):
                itype = input_items[idx].get("type")
                if itype in ("function_call_output", "function_call") or (
                    itype == "message" and input_items[idx].get("role") == "assistant"
                ):
                    insert_at = idx
                else:
                    break
            input_items = input_items[:insert_at] + to_prepend + input_items[insert_at:]

    return "\n\n".join(system_texts), input_items


def _forced_tool_name(tool_choice) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    fn = tool_choice.get("function")
    if isinstance(fn, dict):
        return fn.get("name") or tool_choice.get("name")
    if tool_choice.get("type") == "function":
        return tool_choice.get("name")
    return tool_choice.get("name")


def translate_tools(openai_tools: list[dict] | None,
                    tool_choice=None) -> tuple[list[dict] | None, dict | str | None]:
    """Convert Chat Completions tools to Responses API flat function tools."""
    tools: list[dict] | None = None
    if openai_tools:
        out: list[dict] = []
        for tool in openai_tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
            out.append({
                "type": "function",
                "name": fn.get("name") or tool.get("name") or "",
                "description": fn.get("description") or tool.get("description") or "",
                "parameters": fn.get("parameters") or tool.get("parameters")
                or {"type": "object", "properties": {}},
            })
        if out:
            tools = out

    choice: dict | str | None = None
    forced = _forced_tool_name(tool_choice)
    if forced:
        choice = {"type": "function", "name": forced}
    elif isinstance(tool_choice, str) and tool_choice.strip().lower() in ("auto", "none", "required"):
        choice = tool_choice.strip().lower()
    return tools, choice


def build_request(
    cfg,
    messages: list[dict],
    max_tokens: int,
    *,
    stream: bool = False,
    disable_thinking: bool | None = None,
    tools: list[dict] | None = None,
    tool_choice=None,
    temperature=None,
    prompt_cache_key: str = "",
    previous_response_id: str | None = None,
    encrypted_reasoning_items: list[dict] | None = None,
    json_schema: dict | None = None,
) -> dict:
    """Build a valid ``/v1/responses`` request body."""
    want_off = (disable_thinking if disable_thinking is not None
                else bool(getattr(cfg, "disable_thinking", False)))
    instructions, input_items = translate_messages(
        messages, encrypted_reasoning_items=encrypted_reasoning_items)

    body: dict = {
        "model": getattr(cfg, "model", "") or "",
        "input": input_items,
        "stream": bool(stream),
    }
    if int(max_tokens or 0) > 0:
        body["max_output_tokens"] = int(max_tokens)
    if instructions:
        body["instructions"] = instructions
    if previous_response_id:
        body["previous_response_id"] = previous_response_id
    if prompt_cache_key:
        body["prompt_cache_key"] = prompt_cache_key

    effort = str(getattr(cfg, "reasoning_effort", "default")
                 or "default").strip().lower()
    reasoning_active = False
    if not want_off and effort in ("low", "medium", "high"):
        body["reasoning"] = {"effort": effort}
        body["include"] = ["reasoning.encrypted_content"]
        reasoning_active = True

    if temperature is not None and not reasoning_active:
        body["temperature"] = temperature

    resp_tools, resp_choice = translate_tools(tools, tool_choice)
    if resp_tools:
        body["tools"] = resp_tools
        if resp_choice is not None:
            body["tool_choice"] = resp_choice
    elif json_schema is not None:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": "pedacito_action",
                "strict": True,
                "schema": json_schema,
            }
        }

    return body


build_payload = build_request


def _map_status(status: str | None, *, incomplete_reason: str = "",
                has_tool_calls: bool = False) -> str:
    if has_tool_calls and status in ("", "completed", None):
        return "tool_calls"
    s = str(status or "").strip().lower()
    if s == "completed":
        return "stop"
    if s == "incomplete":
        if incomplete_reason == "content_filter":
            return "content_filter"
        return "length"
    if s in ("failed", "cancelled"):
        return "stop"
    return s or ("tool_calls" if has_tool_calls else "")


def _normalise_usage(usage: dict | None) -> dict:
    usage = usage or {}
    in_details = usage.get("input_tokens_details")
    if not isinstance(in_details, dict):
        in_details = {}
    out_details = usage.get("output_tokens_details")
    if not isinstance(out_details, dict):
        out_details = {}

    prompt_tokens = _pos(usage.get("input_tokens")) or _pos(usage.get("prompt_tokens"))
    completion_tokens = _pos(usage.get("output_tokens")) or _pos(usage.get("completion_tokens"))
    cached_tokens = _pos(in_details.get("cached_tokens")) or _pos(usage.get("cached_tokens"))
    reasoning_tokens = _pos(out_details.get("reasoning_tokens")) or _pos(usage.get("reasoning_tokens"))

    out: dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }
    if reasoning_tokens > 0:
        out["reasoning_tokens"] = reasoning_tokens
        out["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    total_tokens = _pos(usage.get("total_tokens"))
    if total_tokens > 0:
        out["total_tokens"] = total_tokens
    return out


def _extract_reasoning_text(item: dict) -> str:
    parts: list[str] = []
    for key in ("summary", "content"):
        seq = item.get(key)
        if isinstance(seq, str) and seq:
            parts.append(seq)
        elif isinstance(seq, list):
            for entry in seq:
                if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                    parts.append(entry["text"])
    return "".join(parts)


def parse_response(data: dict) -> dict:
    """Normalise one buffered ``/v1/responses`` JSON response."""
    data = data or {}
    response_id = str(data.get("id") or "")
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_items: list[dict] = []
    tool_calls: list[dict] = []

    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("output_text", "text") and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif ptype == "refusal" and isinstance(part.get("refusal"), str):
                    text_parts.append(part["refusal"])
        elif itype == "reasoning":
            reasoning_items.append(dict(item))
            rtext = _extract_reasoning_text(item)
            if rtext:
                reasoning_parts.append(rtext)
        elif itype == "function_call":
            args = item.get("arguments")
            args_json = args if isinstance(args, str) else json.dumps(args or {})
            tool_calls.append({
                "id": str(item.get("call_id") or item.get("id") or ""),
                "name": str(item.get("name") or ""),
                "arguments_json": args_json,
            })

    if not text_parts and isinstance(data.get("output_text"), str):
        text_parts.append(data["output_text"])

    inc = data.get("incomplete_details")
    inc_reason = str((inc or {}).get("reason") or "") if isinstance(inc, dict) else ""
    finish_reason = _map_status(
        data.get("status"),
        incomplete_reason=inc_reason,
        has_tool_calls=bool(tool_calls),
    )

    return {
        "id": response_id,
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "reasoning_items": reasoning_items,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": _normalise_usage(data.get("usage")),
    }


parse_json_response = parse_response


def parse_sse_stream(resp, *, on_text=None, on_reasoning=None,
                     cancel_check=None) -> dict:
    """Parse an OpenAI ``/v1/responses`` SSE stream into the same dict shape
    as :func:`parse_response`."""
    response_id = ""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_items: list[dict] = []
    tool_buffers: dict[int | str, dict] = {}
    tool_order: list[int | str] = []
    usage: dict = {}
    status = ""
    inc_reason = ""

    for raw in resp.iter_lines(decode_unicode=True):
        if cancel_check is not None and cancel_check():
            raise StreamCancelled()
        if not raw:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        if raw.startswith(":"):
            continue
        if not raw.startswith("data:"):
            continue
        body = raw[5:].lstrip()
        if body.strip() == "[DONE]":
            break
        try:
            obj = json.loads(body)
        except (TypeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue

        etype = str(obj.get("type") or "")
        if etype in ("error", "response.error"):
            err = obj.get("error") if isinstance(obj.get("error"), dict) else obj
            raise StreamError(
                str(err.get("code") or err.get("type") or ""),
                str(err.get("message") or ""),
            )

        if etype == "response.created":
            resp_obj = obj.get("response") or {}
            if isinstance(resp_obj, dict) and resp_obj.get("id"):
                response_id = str(resp_obj["id"])
        elif etype == "response.output_text.delta":
            delta = obj.get("delta") or ""
            if delta:
                text_parts.append(delta)
                if on_text is not None:
                    on_text(delta)
        elif etype in ("response.reasoning_summary_text.delta",
                       "response.reasoning.delta",
                       "response.reasoning_text.delta"):
            delta = obj.get("delta") or ""
            if delta:
                reasoning_parts.append(delta)
                if on_reasoning is not None:
                    on_reasoning(delta)
        elif etype == "response.output_item.added":
            item = obj.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "function_call":
                key = obj.get("output_index", item.get("id") or len(tool_order))
                if key not in tool_buffers:
                    tool_order.append(key)
                tool_buffers[key] = {
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "name": str(item.get("name") or ""),
                    "json": [],
                }
        elif etype == "response.function_call_arguments.delta":
            delta = obj.get("delta") or ""
            key = obj.get("output_index", obj.get("item_id") or 0)
            if key not in tool_buffers:
                tool_order.append(key)
                tool_buffers[key] = {"id": "", "name": "", "json": []}
            tool_buffers[key]["json"].append(delta)
        elif etype == "response.output_item.done":
            item = obj.get("item") or {}
            if isinstance(item, dict):
                if item.get("type") == "reasoning":
                    reasoning_items.append(dict(item))
                elif item.get("type") == "function_call":
                    key = obj.get("output_index", item.get("id") or len(tool_order))
                    if key not in tool_buffers:
                        tool_order.append(key)
                        tool_buffers[key] = {"id": "", "name": "", "json": []}
                    entry = tool_buffers[key]
                    if not entry["id"]:
                        entry["id"] = str(item.get("call_id") or item.get("id") or "")
                    if not entry["name"]:
                        entry["name"] = str(item.get("name") or "")
                    if not entry["json"] and item.get("arguments"):
                        args = item["arguments"]
                        entry["json"].append(
                            args if isinstance(args, str) else json.dumps(args))
        elif etype in ("response.completed", "response.incomplete", "response.failed"):
            resp_obj = obj.get("response") or {}
            if isinstance(resp_obj, dict):
                if resp_obj.get("id"):
                    response_id = str(resp_obj["id"])
                status = str(resp_obj.get("status") or etype.split(".")[-1])
                inc = resp_obj.get("incomplete_details")
                if isinstance(inc, dict):
                    inc_reason = str(inc.get("reason") or "")
                if isinstance(resp_obj.get("usage"), dict):
                    usage = resp_obj["usage"]
                # Fallback if output items arrived only on completion
                if not text_parts and not tool_buffers and resp_obj.get("output"):
                    done_parsed = parse_response(resp_obj)
                    if done_parsed["text"]:
                        text_parts.append(done_parsed["text"])
                    if not reasoning_parts and done_parsed["reasoning"]:
                        reasoning_parts.append(done_parsed["reasoning"])
                    if not reasoning_items and done_parsed["reasoning_items"]:
                        reasoning_items.extend(done_parsed["reasoning_items"])
                    for tc in done_parsed["tool_calls"]:
                        k = f"done_{len(tool_order)}"
                        tool_order.append(k)
                        tool_buffers[k] = {
                            "id": tc["id"],
                            "name": tc["name"],
                            "json": [tc["arguments_json"]],
                        }

    tool_calls: list[dict] = []
    for key in tool_order:
        entry = tool_buffers[key]
        if entry.get("id") or entry.get("name") or entry.get("json"):
            tool_calls.append({
                "id": entry.get("id") or "",
                "name": entry.get("name") or "",
                "arguments_json": "".join(entry.get("json") or []) or "{}",
            })

    return {
        "id": response_id,
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "reasoning_items": reasoning_items,
        "tool_calls": tool_calls,
        "finish_reason": _map_status(
            status, incomplete_reason=inc_reason, has_tool_calls=bool(tool_calls)),
        "usage": _normalise_usage(usage),
    }
