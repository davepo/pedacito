"""Cache-aware conversation folding: one fold point, never a rolling rewrite.

    if fold_middle(messages, keep_pairs=4): ...     # in place

A gather conversation is [system, first user turn (map + task), then
(assistant action, user RESULT) pairs]. A model server caches the prompt
PREFIX: editing an old turn invalidates the cache from that turn onward and
forces a re-prefill of everything after it. The old policy clipped the oldest
RESULT on nearly every step once the window filled, so the cache was thrown
away over and over.

Instead, at real window pressure the whole older band -- everything between
the first user turn and the last K pairs -- is replaced by ONE pair holding a
compact `STATE` digest (what was looked up, what each lookup returned in one
line). The head stays byte-identical, the recent tail stays verbatim, and the
cache breaks exactly once, at the fold point. The source those lookups showed
is not lost to the task: gather's ledger and the answer phase's digest still
know every range read, and the STATE says plainly that the text must be
re-`read` before it is quoted.
"""

from __future__ import annotations

import ast
import json
import re

STATE_HEAD = ("RESULT:\nSTATE (earlier lookups, folded to free context -- the "
              "source they showed is no longer in this conversation; re-`read` "
              "anything you need to quote exactly):")
TRIED_FAILED_HEAD = "TRIED & FAILED:"
_MAX_HYPOTHESES = 5
_FOLDED_ACTION = '{"action": "outline", "reasoning": "earlier lookups folded"}'
_LEGACY_FOLDED_ACTION = "{'action': 'outline', 'reasoning': 'earlier lookups folded'}"
_READ_HEAD_RX = re.compile(r"^(.+?) lines (\d+)-(\d+):")


def tail_pairs_for(cfg) -> int:
    """Recent (action, result) pairs kept verbatim, by tier."""
    n = int(getattr(cfg, "fold_tail_steps", 0) or 0)
    if n > 0:
        return n
    return {"tiny": 2, "small": 3, "medium": 4, "large": 5,
            "hosted": 6}.get(getattr(cfg, "active_tier", "") or "", 4)


def _parse_action(content: str) -> dict:
    try:
        obj = json.loads(content)
        return obj if isinstance(obj, dict) else {}
    except (ValueError, TypeError):
        pass
    try:
        obj = ast.literal_eval(content)
        return obj if isinstance(obj, dict) else {}
    except (ValueError, SyntaxError, MemoryError):
        return {}


def _gist(action: dict, result: str, raw_action: str) -> str:
    """One line describing a step and what it returned."""
    name = str(action.get("action") or "").strip() or raw_action[:40]
    body = result[len("RESULT:\n"):] if result.startswith("RESULT:\n") else result
    first = body.strip().splitlines()[0] if body.strip() else ""
    m = _READ_HEAD_RX.match(body)
    if name == "read" or m:
        if m:
            return f"read {m.group(1).strip()} L{m.group(2)}-{m.group(3)}"
        if first.startswith("[compacted earlier lookup:"):
            return first.strip("[]")[:100]
        return f"read {action.get('file') or action.get('symbol') or '?'}"
    arg = (action.get("file") or action.get("pattern") or action.get("symbol")
           or action.get("command") or "")
    arg = arg if isinstance(arg, str) else str(arg)
    arg = f" {arg[:60]!r}" if arg else ""
    if body.startswith("ERROR"):
        return f"{name}{arg} -> {first[:70]}"
    lines = [ln for ln in body.splitlines() if ln.strip()]
    return f"{name}{arg} -> {len(lines)} line(s)"


def format_hypotheses(hypotheses: list[tuple[str, str]] | None,
                      max_items: int = _MAX_HYPOTHESES) -> str:
    """Format up to `max_items` compact `(target, error)` pairs under a
    `TRIED & FAILED:` header, or '' when none exist."""
    if not hypotheses:
        return ""
    dedup: list[tuple[str, str]] = []
    for raw_t, raw_e in hypotheses:
        tgt = " ".join(str(raw_t or "").split())[:80] or "(general)"
        err_lines = [ln.strip() for ln in str(raw_e or "").splitlines()
                     if ln.strip()]
        err = " ".join((err_lines[0] if err_lines else "").split())[:140]
        if not err:
            continue
        pair = (tgt, err)
        if pair in dedup:
            dedup.remove(pair)
        dedup.append(pair)
    dedup = dedup[-max(1, max_items):]
    if not dedup:
        return ""
    body = "\n".join(f"- {t} -> {e}" for t, e in dedup)
    return f"{TRIED_FAILED_HEAD}\n{body}"


def digest(band: list[dict], max_chars: int = 1400,
           hypotheses: list[tuple[str, str]] | None = None) -> str:
    """STATE text for a band of folded messages. Older digests inside the band
    (a fold of a fold) are carried forward; when over `max_chars` the OLDEST
    lines go first. When `hypotheses` (or carried-forward `TRIED & FAILED:`
    entries) are present, up to 5 compact `(target -> error)` entries are
    appended so the model does not repeat failed approaches."""
    lines: list[str] = []
    carried_hyps: list[tuple[str, str]] = []
    i = 0
    while i < len(band):
        m = band[i]
        content = m.get("content") or ""
        if m["role"] == "user" and content.startswith(STATE_HEAD):
            tail_txt = content[len(STATE_HEAD):]
            if TRIED_FAILED_HEAD in tail_txt:
                head_part, hyp_part = tail_txt.split(TRIED_FAILED_HEAD, 1)
                for ln in hyp_part.splitlines():
                    if ln.startswith("- ") and " -> " in ln:
                        t, e = ln[2:].split(" -> ", 1)
                        carried_hyps.append((t.strip(), e.strip()))
            else:
                head_part = tail_txt
            lines += [ln for ln in head_part.splitlines()
                      if ln.startswith("- ")]
            i += 1
            continue
        if m["role"] == "assistant":
            if content in (_FOLDED_ACTION, _LEGACY_FOLDED_ACTION):      # the placeholder of an earlier fold
                i += 1
                continue
            action = _parse_action(content)
            result = ""
            if i + 1 < len(band) and band[i + 1]["role"] == "user":
                result = band[i + 1].get("content") or ""
                i += 1
            lines.append("- " + _gist(action, result, content))
        i += 1
    # collapse exact repeats, keep order
    dedup: list[str] = []
    for ln in lines:
        if not dedup or dedup[-1] != ln:
            dedup.append(ln)
    merged_hyps = carried_hyps + list(hypotheses or [])
    hyp_block = format_hypotheses(merged_hyps)
    avail = max(120, max_chars - (len(hyp_block) + 2 if hyp_block else 0))
    while dedup and sum(len(x) + 1 for x in dedup) > avail:
        dedup.pop(0)
    body = "\n".join(dedup)
    if hyp_block:
        return (body + "\n\n" + hyp_block) if body else hyp_block
    return body


def fold_middle(messages: list[dict], keep_pairs: int,
                max_chars: int = 1400,
                hypotheses: list[tuple[str, str]] | None = None) -> bool:
    """Replace the band between the first user turn and the last `keep_pairs`
    pairs with one STATE pair, in place. Returns False (and changes nothing)
    when there is nothing worth folding (fewer than two pairs in the band)."""
    if len(messages) < 4 or messages[0]["role"] != "system" \
            or messages[1]["role"] != "user":
        return False
    end = len(messages) - 2 * max(1, keep_pairs)
    # the tail must start with an assistant turn so roles keep alternating
    while end > 2 and messages[end]["role"] != "assistant":
        end -= 1
    if end - 2 < 4:
        return False
    band = messages[2:end]
    state = digest(band, max_chars, hypotheses=hypotheses)
    if not state:
        return False
    messages[2:end] = [
        {"role": "assistant", "content": _FOLDED_ACTION},
        {"role": "user", "content": STATE_HEAD + "\n" + state},
    ]
    return True
