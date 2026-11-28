"""Native Google Gemini ``/v1beta`` transport.

Google Gemini exposes an OpenAI-compatible ``/v1beta/openai`` surface, but the
native ``generateContent`` / ``streamGenerateContent`` API provides features the
compatibility shim omits or flattens: ``x-goog-api-key`` auth, native thought
parts (``part.thought == True``) with ``thoughtsTokenCount``, inline base64
images (``inlineData``), ``thinkingConfig``, ``responseMimeType`` /
``responseSchema``, and explicit context caching (``cachedContent``) alongside
implicit ``cachedContentTokenCount`` accounting.

Everything here is a pure function of its inputs; it has no state and no
network I/O. ``LLMClient`` remains the sole owner of HTTP, retries, outbound
secret scanning, replay digests, streaming telemetry, and cost accounting.
"""

from __future__ import annotations

import json
from urllib.parse import quote, urlparse

# Reasoning effort -> Gemini thinkingConfig.thinkingBudget token budgets.
_EFFORT_BUDGET = {"low": 2048, "medium": 8192, "high": 24576}

# Gemini finishReason -> OpenAI finish_reason vocabulary.
_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "LANGUAGE": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
    "OTHER": "stop",
}


class StreamCancelled(Exception):
    """``parse_sse_stream`` saw ``cancel_check()`` return True."""


class StreamError(Exception):
    """A Gemini ``error`` payload arrived mid-stream."""

    def __init__(self, err_type: str = "", message: str = ""):
        super().__init__(message or err_type or "Gemini stream error")
        self.err_type = err_type or ""


def _pos(value) -> int:
    """First positive int for a usage field, else 0."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _clean_model_name(model: str) -> str:
    raw = str(model or "").strip()
    if raw.startswith("models/"):
        raw = raw[len("models/"):].strip()
    return raw


def is_gemini_native(cfg) -> bool:
    """True when this config should use the native Gemini ``/v1beta`` API.

    Enabled by default (``cfg.gemini_native = True``) whenever the endpoint host
    targets ``generativelanguage.googleapis.com`` without ``/openai`` in the
    path, or when ``provider == "gemini"``.
    """
    if not getattr(cfg, "gemini_native", True):
        return False
    raw_url = getattr(cfg, "base_url", "") or ""
    try:
        parsed = urlparse(raw_url)
        host = (parsed.hostname or "").lower()
        path = (parsed.path or "").lower()
    except ValueError:
        host, path = "", ""
    provider = str(getattr(cfg, "provider", "") or "").strip().lower()
    return ("generativelanguage.googleapis.com" in host and "/openai" not in path) \
        or provider == "gemini"


def _api_root(base_url: str) -> str:
    """Normalize ``base_url`` to the Gemini version prefix (e.g. ``.../v1beta``)."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return ""
    parsed = urlparse(base)
    if parsed.netloc:
        origin = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path.rstrip("/")
    else:
        origin, path = "", base
    if path.endswith("/openai"):
        path = path[:-len("/openai")].rstrip("/")
    if path.endswith("/models"):
        path = path[:-len("/models")].rstrip("/")
    if path in ("", "/v1"):
        return origin + "/v1beta"
    if "/v1beta" in path or "/v1alpha" in path:
        return origin + path
    return origin + path + "/v1beta"


def models_endpoint(base_url: str) -> str:
    """Map a base URL to ``.../v1beta/models``."""
    root = _api_root(base_url)
    return (root + "/models") if root else ""


def endpoint(base_url: str, model: str, *, stream: bool = False) -> str:
    """Map ``base_url`` and ``model`` to ``.../v1beta/models/{model}:generateContent``
    or ``:streamGenerateContent?alt=sse``."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if ":generateContent" in base or ":streamGenerateContent" in base:
        stem = base.split(":", 1)[0]
        # Preserve scheme colon (https://...) by splitting from the right on ':'
        idx = base.rfind(":")
        stem = base[:idx]
        return stem + (":streamGenerateContent?alt=sse" if stream else ":generateContent")
    root = _api_root(base)
    mid = quote(_clean_model_name(model), safe="-._~")
    suffix = ":streamGenerateContent?alt=sse" if stream else ":generateContent"
    return f"{root}/models/{mid}{suffix}"


def build_headers(api_key: str) -> dict[str, str]:
    """Native Gemini request headers.

    Gemini's documented ``generateContent`` API has no idempotency-key header
    and no opaque stream-resume cursor, so this transport never attaches either
    (the capability gate in ``LLMClient`` stays ``None``); a resumed stream is
    therefore never claimed, and an interrupted stream is exposed as partial
    text, not auto-replayed.
    """
    return {
        "x-goog-api-key": api_key or "",
        "Content-Type": "application/json",
    }


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
            and isinstance(p.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def _inline_image_part(url) -> dict | None:
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, _, data = url.partition(",")
    spec = header[len("data:"):]
    if "base64" not in spec.lower() or not data:
        return None
    parts = spec.split(";")
    mime = parts[0] if parts and parts[0] else "image/png"
    return {"inlineData": {"mimeType": mime, "data": data}}


def _content_part(part) -> dict | None:
    if not isinstance(part, dict):
        return None
    ptype = part.get("type")
    if ptype == "text":
        text = part.get("text") or ""
        return {"text": text} if text else None
    if ptype == "image_url":
        url = (part.get("image_url") or {}).get("url")
        return _inline_image_part(url)
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


def _clean_schema(schema):
    """Strip JSON Schema keywords Gemini's OpenAPI-subset Schema rejects."""
    if isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    for k, v in schema.items():
        if k in ("additionalProperties", "$schema"):
            continue
        out[k] = _clean_schema(v)
    return out


def _reasoning_parts(message: dict) -> list[dict]:
    """Opaque thought parts a prior assistant turn returned to us.

    Gemini pairs thought text with an optional ``thoughtSignature``; replay it
    in the model turn so a following function response keeps the signed
    context.  Nothing here is ever added to visible text.
    """
    out: list[dict] = []
    for item in message.get("reasoning_items") or ():
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str):
            text = item.get("thinking")
        if not isinstance(text, str) or not text:
            continue
        part: dict = {"text": text, "thought": True}
        signature = item.get("thoughtSignature") or item.get("signature")
        if signature:
            part["thoughtSignature"] = str(signature)
        out.append(part)
    return out


def translate_messages(messages: list[dict]) -> tuple[dict | None, list[dict]]:
    """Translate OpenAI-shaped messages into Gemini ``(systemInstruction, contents)``."""
    system_texts: list[str] = []
    contents: list[dict] = []
    call_names: dict[str, str] = {}

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

        parts: list[dict] = []
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            fn_name = str(message.get("name") or call_names.get(call_id) or "step")
            raw_text = _text_of(content)
            resp_obj: dict = {"result": raw_text}
            if raw_text.lstrip().startswith("{"):
                try:
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict):
                        resp_obj = parsed
                except (TypeError, ValueError):
                    pass
            parts.append({
                "functionResponse": {
                    "name": fn_name,
                    "response": resp_obj,
                }
            })
            gemini_role = "user"
        else:
            if role == "assistant":
                parts.extend(_reasoning_parts(message))
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    gpart = _content_part(part)
                    if gpart is not None:
                        parts.append(gpart)
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function") or {}
                    name = str(fn.get("name") or call.get("name") or "")
                    cid = str(call.get("id") or "")
                    if cid and name:
                        call_names[cid] = name
                    fc_part: dict = {
                        "functionCall": {
                            "name": name,
                            "args": _tool_arguments(fn),
                        }
                    }
                    if call.get("thoughtSignature"):
                        fc_part["thoughtSignature"] = call["thoughtSignature"]
                    parts.append(fc_part)
            gemini_role = "model" if role == "assistant" else "user"

        if not parts:
            continue
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": gemini_role, "parts": parts})

    system_instruction = (
        {"parts": [{"text": "\n\n".join(system_texts)}]}
        if system_texts else None
    )
    return system_instruction, contents


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
                    tool_choice=None) -> tuple[list[dict] | None, dict | None]:
    """Convert OpenAI function tools to Gemini ``functionDeclarations`` + ``toolConfig``."""
    tools: list[dict] | None = None
    if openai_tools:
        decls = []
        for tool in openai_tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
            params = fn.get("parameters") or tool.get("parameters") or {
                "type": "object", "properties": {},
            }
            decls.append({
                "name": fn.get("name") or tool.get("name") or "",
                "description": fn.get("description") or tool.get("description") or "",
                "parameters": _clean_schema(params),
            })
        if decls:
            tools = [{"functionDeclarations": decls}]

    tool_config: dict | None = None
    forced = _forced_tool_name(tool_choice)
    if forced:
        tool_config = {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [forced],
            }
        }
    elif isinstance(tool_choice, str):
        mode_map = {"none": "NONE", "auto": "AUTO", "required": "ANY"}
        mapped = mode_map.get(tool_choice.strip().lower())
        if mapped:
            tool_config = {"functionCallingConfig": {"mode": mapped}}
    return tools, tool_config


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
    response_mime_type: str | None = None,
    response_schema: dict | None = None,
    cached_content: str | None = None,
) -> dict:
    """Build a valid native Gemini ``generateContent`` request body."""
    want_off = (disable_thinking if disable_thinking is not None
                else bool(getattr(cfg, "disable_thinking", False)))
    system_instruction, contents = translate_messages(messages)

    gen_config: dict = {}
    if int(max_tokens or 0) > 0:
        gen_config["maxOutputTokens"] = int(max_tokens)
    if temperature is not None:
        gen_config["temperature"] = temperature

    effort = str(getattr(cfg, "reasoning_effort", "default")
                 or "default").strip().lower()
    if want_off:
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}
    elif effort in _EFFORT_BUDGET:
        gen_config["thinkingConfig"] = {
            "thinkingBudget": _EFFORT_BUDGET[effort],
            "includeThoughts": True,
        }

    gemini_tools, tool_config = translate_tools(tools, tool_choice)
    if not gemini_tools:
        if response_schema is not None:
            gen_config["responseMimeType"] = response_mime_type or "application/json"
            gen_config["responseSchema"] = _clean_schema(response_schema)
        elif response_mime_type:
            gen_config["responseMimeType"] = response_mime_type

    body: dict = {"contents": contents}
    if system_instruction is not None:
        body["systemInstruction"] = system_instruction
    if gen_config:
        body["generationConfig"] = gen_config
    if gemini_tools:
        body["tools"] = gemini_tools
    if tool_config:
        body["toolConfig"] = tool_config

    cached_ref = (cached_content
                  or getattr(cfg, "gemini_cached_content", "")
                  or getattr(cfg, "cached_content", "")
                  or "").strip()
    if cached_ref:
        body["cachedContent"] = cached_ref

    return body


build_payload = build_request


def _map_finish_reason(reason: str | None, *, has_tool_calls: bool = False) -> str:
    raw = str(reason or "").strip().upper()
    if has_tool_calls and raw in ("", "STOP"):
        return "tool_calls"
    if not raw:
        return ""
    return _FINISH_REASON_MAP.get(raw, raw.lower())


def _normalise_usage(um: dict | None) -> dict:
    um = um or {}
    prompt_tokens = _pos(um.get("promptTokenCount"))
    candidates_tokens = _pos(um.get("candidatesTokenCount"))
    thoughts_tokens = _pos(um.get("thoughtsTokenCount"))
    cached_tokens = _pos(um.get("cachedContentTokenCount"))
    completion_tokens = candidates_tokens + thoughts_tokens
    usage: dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }
    if thoughts_tokens > 0:
        usage["completion_tokens_details"] = {
            "reasoning_tokens": thoughts_tokens,
        }
    total_tokens = _pos(um.get("totalTokenCount"))
    if total_tokens > 0:
        usage["total_tokens"] = total_tokens
    return usage


def parse_response(data: dict) -> dict:
    """Normalise one buffered Gemini ``generateContent`` JSON response."""
    data = data or {}
    candidates = data.get("candidates") or []
    cand = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    parts = ((cand.get("content") or {}).get("parts") or [])

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_items: list[dict] = []
    tool_calls: list[dict] = []

    for part in parts:
        if not isinstance(part, dict):
            continue
        if "functionCall" in part and isinstance(part["functionCall"], dict):
            fc = part["functionCall"]
            args = fc.get("args") if fc.get("args") is not None else {}
            args_json = args if isinstance(args, str) else json.dumps(args)
            call = {
                "id": fc.get("id") or f"call_{len(tool_calls)}",
                "name": fc.get("name") or "",
                "arguments_json": args_json,
            }
            if part.get("thoughtSignature"):
                call["thoughtSignature"] = part["thoughtSignature"]
            tool_calls.append(call)
        elif "text" in part and isinstance(part.get("text"), str):
            if part.get("thought"):
                reasoning_parts.append(part["text"])
                item = {"text": part["text"], "thought": True}
                if part.get("thoughtSignature"):
                    item["thoughtSignature"] = part["thoughtSignature"]
                reasoning_items.append(item)
            else:
                text_parts.append(part["text"])

    finish_reason = _map_finish_reason(
        cand.get("finishReason"), has_tool_calls=bool(tool_calls))

    return {
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "reasoning_items": reasoning_items,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": _normalise_usage(data.get("usageMetadata")),
    }


parse_json_response = parse_response


def parse_sse_stream(resp, *, on_text=None, on_reasoning=None,
                     cancel_check=None) -> dict:
    """Parse a Gemini ``streamGenerateContent?alt=sse`` stream into the same
    dict shape as :func:`parse_response`."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_items: list[dict] = []
    tool_calls: list[dict] = []
    last_usage: dict = {}
    raw_finish = ""

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
        if isinstance(obj.get("error"), dict):
            err = obj["error"]
            raise StreamError(
                str(err.get("status") or err.get("code") or ""),
                str(err.get("message") or ""),
            )
        if isinstance(obj.get("usageMetadata"), dict):
            last_usage = obj["usageMetadata"]

        candidates = obj.get("candidates") or []
        cand = candidates[0] if candidates and isinstance(candidates[0], dict) else None
        if cand is None:
            continue
        if cand.get("finishReason"):
            raw_finish = str(cand["finishReason"])
        parts = ((cand.get("content") or {}).get("parts") or [])
        for part in parts:
            if not isinstance(part, dict):
                continue
            if "functionCall" in part and isinstance(part["functionCall"], dict):
                fc = part["functionCall"]
                args = fc.get("args") if fc.get("args") is not None else {}
                args_json = args if isinstance(args, str) else json.dumps(args)
                call = {
                    "id": fc.get("id") or f"call_{len(tool_calls)}",
                    "name": fc.get("name") or "",
                    "arguments_json": args_json,
                }
                if part.get("thoughtSignature"):
                    call["thoughtSignature"] = part["thoughtSignature"]
                tool_calls.append(call)
            elif "text" in part and isinstance(part.get("text"), str):
                delta = part["text"]
                if not delta:
                    continue
                if part.get("thought"):
                    reasoning_parts.append(delta)
                    item = {"text": delta, "thought": True}
                    if part.get("thoughtSignature"):
                        item["thoughtSignature"] = part["thoughtSignature"]
                    reasoning_items.append(item)
                    if on_reasoning is not None:
                        on_reasoning(delta)
                else:
                    text_parts.append(delta)
                    if on_text is not None:
                        on_text(delta)

    return {
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "reasoning_items": reasoning_items,
        "tool_calls": tool_calls,
        "finish_reason": _map_finish_reason(
            raw_finish, has_tool_calls=bool(tool_calls)),
        "usage": _normalise_usage(last_usage),
    }
