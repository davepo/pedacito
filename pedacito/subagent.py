"""The `task` action: a read-only helper agent with a fresh context window.

    tool_task(index, cfg, client, "who ends up calling retry() from the CLI?")

A long investigation (trace a call chain across six files, triage a test log)
fills a small model's window with material the parent does not need. The
parent hands the question to a CHILD agent instead: a new conversation, a
narrow brief, a step cap, READ-ONLY tools, and only a short digest comes back.
The child shares the index (and the client, so usage is counted once) but not
the transcript, and runs on its own copy of the config, so the parent's
per-task state (write allowlist, read ledger) cannot be disturbed.

Depth is capped (1 for local models, 2 for hosted ones); the deepest child
cannot spawn another. Writes, shell, browser and MCP are always off in a child.
"""

from __future__ import annotations

import copy


DIGEST_MAX_CHARS = 1500

SUBAGENT_BASE_NOTE = (
    "You are a HELPER agent. Another agent asked you ONE bounded research "
    "question. You cannot edit anything. Find the answer with as few lookups "
    "as you can, then say `done`.")

DIGEST_SYSTEM = """\
You are a helper agent reporting back to another agent. Answer the question
from the SOURCE YOU REQUESTED only, in at most 12 short lines. Cite every claim
as file:line. If you could not verify something, say 'unverified:' and what.
No code blocks longer than 3 lines, no edit blocks, no preamble."""


def max_depth(cfg) -> int:
    from .config import is_hosted
    return 2 if is_hosted(cfg) else 1


def _tier(cfg) -> str:
    """Effective tier for the single-slot guard: an applied profile tier wins,
    otherwise the same model/base-url guess config.py uses for profiles."""
    tier = str(getattr(cfg, "active_tier", "")
               or getattr(cfg, "tier", "") or "").strip().lower()
    if tier:
        return tier
    try:
        from .config import _guess_tier
        return str(_guess_tier(
            str(getattr(cfg, "active_profile", "") or ""),
            {"model": getattr(cfg, "model", ""),
             "base_url": getattr(cfg, "base_url", "")}) or "").lower()
    except Exception:
        return ""


def _shared_prefix_eligible(cfg) -> bool:
    """True when a child may fork the parent's prefix on a single-slot local
    server: no separate scout, a local endpoint, and a tiny/small profile."""
    if str(getattr(cfg, "scout_profile", "") or "").strip():
        return False
    try:
        from .llm import _is_local_server
        if not _is_local_server(cfg):
            return False
    except Exception:
        return False
    tier = _tier(cfg)
    if tier:
        return tier in ("tiny", "small")
    try:
        from .prompts import pack_for
        return pack_for(cfg) in ("tiny", "small")
    except Exception:
        return False


def child_config(cfg):
    """A read-only copy of `cfg` for a child agent."""
    child = copy.copy(cfg)
    depth = int(getattr(cfg, "_depth", 0) or 0) + 1
    child._depth = depth
    for flag in ("allow_write", "allow_shell", "allow_browser",
                 "allow_screenshots", "allow_mcp", "enable_mcp",
                 "allow_fetch", "enable_web",
                 "semantic_search", "verify_tests", "auto_allowlist"):
        setattr(child, flag, False)
    child.allow_subagents = depth < max_depth(cfg)
    child.max_steps = max(1, int(getattr(cfg, "subagent_max_steps", 8) or 8))
    child.answer_mode = "rebuild"
    child.git_context = False
    child._interactive = False        # a helper never interrupts the user
    child.__dict__.pop("_steer", None)
    child.log_events = getattr(cfg, "log_events", True)
    for scratch in ("_backup_dir", "_task_reads", "_task_allowlist",
                    "_task_mutations", "_agent"):
        child.__dict__.pop(scratch, None)
    return child


def tool_task(index, cfg, client, brief: str, parent=None) -> str:
    """Run a child agent on `brief` and return its digest.

    `parent`, when given, receives the child's best 1-2 content-addressed
    `EvidenceSpan`s (hashed exact source the helper really read) in its own
    evidence store and read ledger, plus matching requirement-state advances.
    The digest observation is extended with the bounded numbered spans so the
    parent can edit them without paying for a second `read`.
    """
    if not getattr(cfg, "allow_subagents", False):
        return ("ERROR: `task` is disabled (config allow_subagents = false). "
                "Research the question yourself with `read`/`grep`/`callers`.")
    brief = (brief or "").strip()
    if not brief:
        return "ERROR: task needs a question in `content`."
    if client is None:
        return "ERROR: task needs a model client."
    if int(getattr(cfg, "_depth", 0) or 0) >= max_depth(cfg):
        return "ERROR: sub-agent depth limit reached; do this yourself."
    from .agent import Agent
    from .explain import EXPLAIN_GATHER_BASE

    # Scout routing: when a separate fast profile is configured and reachable,
    # the child's gather (and its small digest) runs there, so the architect's
    # KV cache is not evicted by helper lookups.
    gather_client = client
    scout_active = False
    if str(getattr(cfg, "scout_profile", "") or "").strip():
        try:
            from . import roles
            resolved = roles.scout_client(cfg, client)
            if resolved is not None and resolved is not client:
                gather_client = resolved
                scout_active = True
        except Exception:
            gather_client = client

    child_cfg = child_config(cfg)
    # Single-slot shared-prefix forking: with no scout on a local tiny/small
    # profile, reuse the parent's System Prompt + First User Turn byte-for-byte
    # and append the helper brief, instead of wiping the server's cached KV.
    seed: list[dict] = []
    if parent is not None and not scout_active and _shared_prefix_eligible(cfg):
        parent_msgs = getattr(parent, "_messages", None) or []
        if (len(parent_msgs) >= 2
                and parent_msgs[0].get("role") == "system"
                and parent_msgs[1].get("role") == "user"):
            seed = [dict(m) for m in parent_msgs[:2]]

    child = Agent(index, gather_client, child_cfg)
    prev_site = getattr(gather_client, "site", "")
    try:
        gather_client.site = "subagent"
        gathered = child.gather(brief, verbose=False,
                                base=EXPLAIN_GATHER_BASE
                                + "\n\n" + SUBAGENT_BASE_NOTE,
                                prefix_messages=seed or None)
        digest = child.answer(brief, gathered, system=DIGEST_SYSTEM,
                              include_edit_format=False, stream=False,
                              max_tokens=700)
    except Exception as e:                          # a child never kills the parent
        return f"ERROR: helper agent failed ({type(e).__name__}: {str(e)[:160]})."
    finally:
        gather_client.site = prev_site
        # Drop the child's ephemeral turns: only its digest + evidence cross
        # the boundary back to the parent.
        try:
            del child._messages[len(seed):]
        except (AttributeError, TypeError):
            pass
    digest = (digest or "").strip()
    spans = []
    if parent is not None and hasattr(parent, "absorb_subagent_evidence"):
        try:
            spans = parent.absorb_subagent_evidence(child)
        except Exception:
            spans = []
    if not digest:
        steps = "; ".join(getattr(child, "steps_log", [])[:6]) or "no steps"
        return f"HELPER RESULT: the helper produced no answer ({steps})."
    if len(digest) > DIGEST_MAX_CHARS:
        digest = digest[:DIGEST_MAX_CHARS].rsplit("\n", 1)[0] + "\n[digest clipped]"
    if spans and hasattr(parent, "append_subagent_evidence"):
        try:
            digest = parent.append_subagent_evidence(digest, spans)
        except Exception:
            pass
    return ("HELPER RESULT (a read-only helper's digest; verify what you will "
            "rely on):\n" + digest)
