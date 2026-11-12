"""Command line interface: argument parsing and the pedacito subcommands.

Usage
-----
    pedacito index  src/                     # build the index (slow, cached)
    pedacito ask    src/ -t "add retries"    # one task, end to end
    pedacito chat   src/                     # interactive, index stays warm
    pedacito map    src/                     # print the index, no LLM
    pedacito files  .                        # what would be indexed, and why
    pedacito cost   src/                     # what indexing would cost
    pedacito config --init                   # write a starter config file
    pedacito how "how do I set up a config?" # built-in help from docs + code
    pedacito explain src/ -q "how does X work?" # answer questions about the project
    pedacito restore src/                    # undo the last --apply
    pedacito reset  ~/code/app               # wipe .pedacito/ state (code stays)
    pedacito new  my-app -t "todo CLI"       # start a project from scratch
    pedacito auto . -t "add retries" --apply # autonomous outer loop over tasks
    pedacito rename old_name new_name        # rename a symbol across files, no model call

Run as `pedacito <command> ...` or `python -m pedacito <command> ...`; `pdc`
is a shorthand alias accepting the same commands. `main()` is the entry point.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path

from . import config
from . import config_editor
from . import discovery
from . import events as events_mod
from . import hooks as hooks_mod
from . import index as index_mod
from . import loop
from . import modelcard
from . import profiles as profiles_mod
from . import roles as roles_mod
from . import workspace
from .agent import Agent
from .config import (Config, ConfigError, REFERENCE_TEMPLATE, TEMPLATE,
                     USER_CONFIG, _ask, _guess_tier, auto_fit,
                     find_project_config, interactive_setup, load as load_config,
                     write_template)
from .llm import LLMClient, LLMError, estimate_tokens
from .select import CODE_EXTS, REFERENCE_EXTS, Matcher, is_venv_dir, select


def _add_common(p, paths_nargs="+"):
    """Register the flags shared by the project subcommands. `paths_nargs="?"`
    lets a command run from the current directory with no argument (rpc)."""
    p.add_argument("paths", nargs=paths_nargs,
                   default=["."] if paths_nargs == "?" else None,
                   help="files or directories to index")
    p.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                   help="gitignore-style pattern to skip; repeatable "
                        "(e.g. --exclude 'deprecated/' --exclude '*_old.py')")
    p.add_argument("--include", action="append", default=[], metavar="PATTERN",
                   help="re-include something an ignore rule excluded; repeatable")
    p.add_argument("--ext", action="append", default=[], metavar=".EXT",
                   help="extra file extension to index as a reference file")
    p.add_argument("--no-refs", action="store_true",
                   help="Python only; skip .md/.csv/.json/.toml and friends")
    p.add_argument("--no-gitignore", action="store_true",
                   help="do not read .gitignore (.pedacitoignore is still read)")
    p.add_argument("--url", help="server base url, e.g. http://box.tail1234.ts.net:1234/v1 "
                                 "(LM Studio) or http://box.tail1234.ts.net:11434/v1 (Ollama)")
    p.add_argument("--model", help="model id/name as the server reports it")
    p.add_argument("--root", default=None,
                   help="project root (default: inferred from the paths given)")
    p.add_argument("--config", metavar="FILE", help="use this config file as well")
    p.add_argument("--profile", metavar="NAME",
                   help="apply a [profiles.NAME] block from your config")
    p.add_argument("--steps", type=int,
                   help="hard lookup limit for the gather phase; 0 = no limit "
                        "(the default)")
    p.add_argument("--max-tokens-budget", type=int, default=None, metavar="N",
                   help="stop before the next model call once this many "
                        "prompt+completion tokens were used (0 = no limit)")
    p.add_argument("--role", action="append", default=[], metavar="ROLE=PROFILE",
                   help="route one job to another profile for this run, e.g. "
                        "--role planner=big --role summarizer=tiny; repeatable "
                        "(roles: worker planner verifier summarizer repair "
                        "escalate fallback)")
    p.add_argument("--record", action="store_true",
                   help="save this run's model responses to "
                        ".pedacito/replay/<stamp>.jsonl")
    p.add_argument("--replay", metavar="FILE",
                   help="run against a recording instead of a model server "
                        "(no model needed; reports where the harness's prompts "
                        "differ from the recording)")
    p.add_argument("--strict-replay", dest="strict_replay", action="store_true",
                   help="with --replay: exit 1 when any prompt differs from "
                        "the recording (CI A/B gate)")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 when a task applied edits but left a failed "
                        "edit block or impacted tests still failing after the "
                        "repair rounds (ask/auto; without it both are "
                        "reported but the run exits 0)")
    p.add_argument("--jsonl", action="store_true",
                   help="machine-readable run: one JSON event per line on "
                        "stdout (steps, calls, the answer and diffs), human "
                        "text moves to stderr")
    p.add_argument("--max-cost", type=float, default=None, metavar="USD",
                   help="stop before the next call once the estimated spend "
                        "reaches this (needs price_*_per_mtok in config)")


def _root(args) -> Path:
    """--root if given, otherwise inferred from the paths."""
    if getattr(args, "root", None):
        return Path(args.root).resolve()
    return index_mod.infer_root(getattr(args, "paths", []) or ["."])


def _matcher(args, root: Path) -> Matcher:
    """File-selection Matcher: ignore files plus --exclude/--include flags."""
    names = (".pedacitoignore",) if getattr(args, "no_gitignore", False) \
        else (".pedacitoignore", ".gitignore")
    m = Matcher.from_files(root, names=names)
    m.extend(getattr(args, "exclude", []))
    # Last match wins, so includes go last.
    m.extend("!" + p for p in getattr(args, "include", []))
    return m


def _config(args) -> Config:
    """Load the Config from files/profile, then apply command-line overrides."""
    # `config` must be able to inspect a file that lacks the mandatory vram_gb.
    cfg = load_config(root=_root(args), explicit=getattr(args, "config", None),
                      profile=getattr(args, "profile", None),
                      require_hardware=(getattr(args, "cmd", "") != "config"))
    if getattr(args, "no_refs", False):
        cfg.include_references = False
    if getattr(args, "jsonl", False) or getattr(args, "rpc", False) or \
            getattr(args, "cmd", "") == "rpc":
        # Event tee is the RPC transport itself; the log must be alive.
        cfg.log_events = True
    for spec in getattr(args, "role", None) or []:
        role, _eq, prof = spec.partition("=")
        if role.strip() not in roles_mod.ROLES or not prof.strip():
            raise ConfigError(f"--role {spec!r}: use ROLE=PROFILE with ROLE "
                              f"one of {', '.join(roles_mod.ROLES)}")
        cfg.roles = {**(cfg.roles or {}), role.strip(): prof.strip()}
    if getattr(args, "ext", None):
        cfg.extra_extensions = tuple(args.ext)
    if getattr(args, "url", None):
        cfg.base_url = args.url
        cfg.sources["base_url"] = "--url"
    if getattr(args, "model", None):
        cfg.model = args.model
        cfg.sources["model"] = "--model"
    if getattr(args, "steps", None) is not None:
        cfg.max_steps = args.steps
        cfg.sources["max_steps"] = "--steps"
    if getattr(args, "max_tokens_budget", None) is not None:
        cfg.max_total_tokens = args.max_tokens_budget
        cfg.sources["max_total_tokens"] = "--max-tokens-budget"
    if getattr(args, "max_cost", None) is not None:
        cfg.max_cost = args.max_cost
        cfg.sources["max_cost"] = "--max-cost"
    # Fit token budgets to the server's real context window before any request;
    # `config` skips this so it reports the true sources.
    if getattr(args, "cmd", "") != "config":
        auto_fit(cfg, verbose=True)
    # A measured model card (`pedacito calibrate`) fills settings the user
    # left at their defaults; explicit values always win.
    if getattr(args, "cmd", "") not in ("config", "calibrate"):
        modelcard.apply(cfg)
    cfg._root = str(_root(args)) if getattr(args, "paths", None) else ""
    return cfg


def _refresh(idx, args, cfg, root: Path, verbose: bool = True):
    """Bring a loaded index back in line with what is on disk.

    Files edited between sessions are re-chunked (fast, local); summaries
    survive by content hash, so only changed parts lose theirs.
    """
    changed, missing = idx.stale()
    # stale() may refresh recorded [mtime, size] entries; persist them so the
    # next run can skip the hashes.
    stats_dirty = idx.stats_refreshed
    for rel in missing:
        idx.forget(rel)
    for rel in changed:
        idx = index_mod.reindex_file(idx, rel, cfg, save=False)

    # Files added since the last index.
    exts = set(CODE_EXTS)
    if cfg.include_references:
        exts |= set(REFERENCE_EXTS)
    exts |= {e if e.startswith(".") else "." + e for e in cfg.extra_extensions}
    on_disk = set()
    for f in select(args.paths, root, _matcher(args, root), exts):
        try:
            on_disk.add(f.resolve().relative_to(root).as_posix())
        except ValueError:
            on_disk.add(f.name)
    added = sorted(on_disk - set(idx.files))

    if verbose and (changed or missing or added):
        bits = []
        if changed:
            bits.append(f"{len(changed)} changed")
        if missing:
            bits.append(f"{len(missing)} removed")
        if added:
            bits.append(f"{len(added)} new")
        print(f"  index refreshed ({', '.join(bits)})", file=sys.stderr)

    # Splice new files in rather than rebuilding, which would discard every
    # cached summary.
    for rel in added:
        idx = index_mod.reindex_file(idx, rel, cfg, save=False)
    if verbose and added:
        print("  new files need summaries: run `pedacito index` when convenient",
              file=sys.stderr)

    if changed or missing or added or stats_dirty:
        idx.save()
    return idx


def _get_index(args, cfg, client, rebuild: bool, summarise: bool = True):
    """Load and refresh the project's index, or build it if none exists or
    `rebuild` is set."""
    root = _root(args)
    if not rebuild:
        existing = index_mod.Index.load(root)
        if existing is not None:
            return _refresh(existing, args, cfg, root)
    workspace.ensure(root, cfg.manage_gitignore, cfg.create_ignore_file)
    print("Indexing...", file=sys.stderr)
    return index_mod.build(args.paths, cfg, client=client if summarise else None,
                           root=root, summarise=summarise,
                           matcher=_matcher(args, root))


SECRETISH = {"api_key"}


def _tree_sitter_report() -> str:
    """One `config --doctor` probe line: whether tree-sitter powers the exact
    syntax checks, and when it does not, why not. An unsafe tree-sitter
    (>=0.26, py-tree-sitter#472) silently disables the chunker and every
    edit then gets only regex-level syntax feedback -- the probe makes that
    degradation visible where the user can fix it."""
    from . import chunker_ts
    ver = getattr(chunker_ts, "_ts_version", "") or ""
    if chunker_ts.TS_AVAILABLE:
        tail = f" (tree-sitter {ver})" if ver else ""
        return (f"Tree-sitter: active{tail} -- exact syntax gate for edits "
                "and per-language chunking.")
    why = getattr(chunker_ts, "_TS_DISABLED_REASON", "") or "not importable"
    return (f"Tree-sitter: INACTIVE -- {why}. Edit syntax checks fall back "
            "to regex-level heuristics (weaker); fix the install to get "
            "them back.")


def _cmd_config(args, cfg) -> int:
    """`pedacito config`: write a starter (--init) or the reference template,
    else show each resolved setting and its origin; --check also tests the
    server connection and model, --doctor additionally probes which
    structured-output modes the server supports."""
    if getattr(args, "reference", False):
        target = Path(args.path) if args.path else None
        if target is not None:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(REFERENCE_TEMPLATE, encoding="utf-8")
            except OSError as e:
                print(f"Could not write {target}: {e}", file=sys.stderr)
                return 2
            print(f"Wrote reference config to {target}", file=sys.stderr)
        else:
            sys.stdout.write(REFERENCE_TEMPLATE)
        return 0

    if args.init:
        try:
            path = write_template(Path(args.path) if args.path else None, args.force)
        except ConfigError as e:
            print(e, file=sys.stderr)
            return 2
        print(f"Wrote {path}", file=sys.stderr)
        print("Mandatory settings: base_url, model and vram_gb in "
              "[profiles.default].\n"
              "Everything else has a working default -- see\n"
              "`pedacito config --reference` for the annotated full list, or run\n"
              "`pedacito config --setup` to have it built and tested interactively.\n"
              "Then: pedacito config --check", file=sys.stderr)
        return 0

    root = _root(args)
    print("Config files, in increasing precedence:")
    for label, path in (("user   ", USER_CONFIG), ("project", find_project_config(root))):
        if path is None:
            print(f"  {label}  (none in {root})")
        else:
            print(f"  {label}  {path}" + ("" if path.exists() else "   [missing]"))
    if args.config:
        print(f"  explicit {args.config}")

    if cfg.profiles:
        print(f"\nProfiles defined ({len(cfg.profiles)}):")
        print(profiles_mod.render(profiles_mod.list_profiles(cfg)))
    if cfg.active_profile:
        tier = f", tier {cfg.active_tier}" if cfg.active_tier else ""
        print(f"  active: {cfg.active_profile}{tier}")

    print("\nResolved settings (non-default marked):")
    width = max(len(k) for k in cfg.sources)
    for key in sorted(cfg.sources):
        value = getattr(cfg, key)
        if key in SECRETISH and value:
            value = "***"
        origin = cfg.sources[key]
        mark = "  " if origin == "default" else "* "
        note = "" if origin == "default" else f"   <- {origin}"
        print(f"  {mark}{key:<{width}}  {value!r}{note}")

    if cfg.roles:
        print("\nRoles: " + "; ".join(roles_mod.describe(cfg)))
    card = modelcard.load(cfg)
    if card:
        print("\n" + modelcard.render(card))
    else:
        print("\nNo model card for this model yet: `pedacito calibrate` "
              "measures it and tunes prompt pack, answer mode and verifier.")
    doctor = getattr(args, "doctor", False)
    if not args.check and not doctor:
        print("\nRun `pedacito config --check` to test the connection, or "
              "`--doctor` to also probe its structured-output modes.",
              file=sys.stderr)
        return 0

    print(f"\nConnecting to {cfg.base_url} ...")
    client = LLMClient(cfg)
    try:
        loaded = client.check(warn=False)
    except LLMError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"  reachable. Models available: {loaded}")
    if cfg.model not in loaded.split(", "):
        print(f"  configured model '{cfg.model}': NOT IN THE LIST above.",
              file=sys.stderr)
        print("  Copy the id exactly as shown.", file=sys.stderr)
        return 1
    # Listing a model does not prove it can load; generate a token to find out.
    state = client.loaded_models()
    if state is not None:
        resident = [m for m, st in state.items() if st == "loaded"]
        print(f"  currently loaded: {', '.join(resident) if resident else '(none)'}")
        if resident and cfg.model not in resident:
            print(f"  '{cfg.model}' is not resident; it will be loaded on first "
                  f"use (up to {cfg.load_wait_seconds}s).")
    print(f"  '{cfg.model}' is listed. Testing that it actually generates...")
    try:
        client.warmup()
    except LLMError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    if not doctor:
        print("  generation works. You're good.")
        return 0
    print("  generation works.")

    # --doctor: one-probe self-calibration. Reports what the server really
    # supports so the config stops guessing.
    print(f"\nProbing structured-output modes on '{cfg.model}' ...")
    results = client.probe_json_modes()
    width = max(len(m) for m, _ok, _d in results)
    ok_modes = []
    for mode, ok, detail in results:
        print(f"  {mode:<{width}}  {'ok    ' if ok else 'FAILED'}  {detail}")
        if ok:
            ok_modes.append(mode)
    if "json_schema" in ok_modes:
        rec = "schema"
        why = ("the server enforces a grammar, so step replies cannot come "
               "back unparseable")
    elif "json_object" in ok_modes:
        rec = "object"
        why = "no grammar support, but the server does enforce JSON mode"
    else:
        rec = "schema"
        why = ("no structured mode works; Pedacito will fall back to JSON "
               "prompting automatically, which needs an instruction-following "
               "model")
    print(f"\nRecommended: json_mode = \"{rec}\" -- {why}.")
    if cfg.json_mode != rec:
        print(f"  Your config currently says json_mode = \"{cfg.json_mode}\".")
    info = client.model_info()
    if info:
        bits = []
        if info.get("context_length"):
            bits.append(f"context {info['context_length']:,}")
        if info.get("max_context_length"):
            bits.append(f"max {info['max_context_length']:,}")
        if info.get("size_bytes"):
            bits.append(f"weights {info['size_bytes'] / 1024 ** 3:.1f} GiB")
        if bits:
            print(f"Server reports ({info.get('source', '?')}): "
                  + ", ".join(bits) + ".")
    cpt = getattr(client, "chars_per_token", 0.0)
    if cpt:
        print(f"Tokenizer calibrated: ~{cpt:.2f} chars/token "
              "(fit math will use this instead of the 3.6 default).")
    else:
        print("No tokenizer endpoint answered; fit math uses the "
              "~3.6 chars/token heuristic.")
    from . import testsel
    tc = testsel.detect_test_command(_root(args), cfg)
    print(f"Test command detected: {tc or '(none)'}")
    from . import sandbox
    state, detail = sandbox.status(cfg)
    print(f"Sandbox: {state} -- {detail}")
    print(_tree_sitter_report())
    # Prompt-cache doctor: resolve the mode from the provider table (the
    # setting wins), then send one 1.2k-token prompt twice and report whether
    # the second call hit a cache.
    from . import providers
    mode = providers.cache_mode_for(cfg)
    cap = client.doctor_prompt_cache()
    print(f"\nPrompt cache: {mode} -- observed {cap}.")
    print("\nNext: `pedacito calibrate` measures tool choice, edit format, "
          "honesty and long-context retrieval, and saves a model card.")
    return 0


CHAT_HELP = """\
  /profile NAME   switch model profile (loads it, waits if needed)
  /model ID       switch to a model id directly
  /apply on|off   enable or disable writing edits
  /steps N        how many lookups the model gets per task
  /status         current profile, model, write mode, prompt pack, budgets
  /cost           tokens, cache-hit ratio and spend so far, by call site
  /memory         what Pedacito remembers about this project
  /skills         the available skills
  /profiles       list the profiles in your config
  /quit           leave
  Ctrl-C          cancel the running task (the session stays open)"""


def _switch_profile(args, ref, arg: str, kind: str = "profile") -> str:
    """Switch the live session to another profile/model WITHOUT rebuilding the
    index - the whole point of `chat` and `rpc`. Mutates `ref` (cfg/client/
    agent re-resolve, `idx` stays). Returns a human label of what is now in
    use. Raises `ConfigError`/`LLMError` - having restored `args` and left
    `ref` untouched - when the new profile does not work."""
    cfg = ref["cfg"]
    prev_profile, prev_model = args.profile, args.model
    if kind == "profile":
        args.profile, args.model = arg, None
    else:
        args.model = arg
    try:
        new_cfg = _config(args)
    except ConfigError:
        args.profile, args.model = prev_profile, prev_model
        raise
    new_client = LLMClient(new_cfg)
    try:
        new_client.check(warn=False)
        new_client.warmup()
    except LLMError:
        args.profile, args.model = prev_profile, prev_model
        raise
    # Live session choices survive a model switch: terminal/steering hooks,
    # the scout selection (independent of the active profile), role routing,
    # and hosts a human allowed from the permission broker this session.
    for keep in ("_interactive", "_ask_user", "_steer", "scout_profile",
                 "roles", "fetch_allowed_hosts"):
        if keep in cfg.__dict__:
            new_cfg.__dict__[keep] = cfg.__dict__[keep]
    ref["cfg"], ref["client"] = new_cfg, new_client
    ref["agent"] = Agent(ref["idx"], new_client, new_cfg)
    # the new client's per-call usage must reach the same event log (and, in
    # rpc mode, the same stdout tee) as the old one's did
    events_mod.attach(new_client, events_mod.get(_root(args), new_cfg))
    label = ""
    if new_cfg.active_profile:
        label = f"profile {new_cfg.active_profile}"
        if new_cfg.active_tier:
            label += f" [{new_cfg.active_tier} limits]"
        label += ", "
    return f"{label}model {new_cfg.model}"


def _chat_command(line: str, args, cfg, ref) -> bool:
    """Handle a /command typed in `pedacito chat`; returns False to end the
    session. Profile switching keeps the index warm between models."""
    parts = line[1:].split(None, 1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("quit", "exit", "q"):
        return False

    if cmd in ("help", "h", "?"):
        print(CHAT_HELP, file=sys.stderr)
        return True

    if cmd == "profiles":
        if not cfg.profiles:
            print(_profiles_empty_note(), end="", file=sys.stderr)
        else:
            print(profiles_mod.render(profiles_mod.list_profiles(cfg)),
                  file=sys.stderr)
        return True

    if cmd == "status":
        print(f"  profile : {cfg.active_profile or '(none)'}", file=sys.stderr)
        print(f"  model   : {cfg.model}", file=sys.stderr)
        print(f"  steps   : {cfg.max_steps}", file=sys.stderr)
        from . import sandbox
        print(f"  sandbox : {sandbox.status(cfg)[0]}", file=sys.stderr)
        print(f"  writes  : {'ON' if args.apply else 'off (dry run)'}", file=sys.stderr)
        print(f"  prompts : pack {prompts_pack(cfg)}, answer "
              f"{ref['agent']._answer_mode()}, verify {cfg.verify_answer}"
              f", best_of {cfg.best_of}", file=sys.stderr)
        print(f"  window  : {cfg.context_tokens:,} tokens "
              f"(map budget {cfg.flat_map_token_budget:,}, gathered source "
              f"{cfg.max_gathered_chars:,} chars)", file=sys.stderr)
        card = getattr(cfg, "_card", None)
        print(f"  card    : {'calibrated ' + format(card['measured'].get('overall', '?')) if card else 'none (run `pedacito calibrate`)'}",
              file=sys.stderr)
        if cfg.max_total_tokens or cfg.max_cost:
            print(f"  budget  : {cfg.max_total_tokens or 'no'} tokens / "
                  f"${cfg.max_cost or 0:.2f}", file=sys.stderr)
        return True

    if cmd == "cost":
        c = ref["client"]
        print("  " + _usage_line(c), file=sys.stderr)
        for site, row in sorted(c.stats.get("by_site", {}).items()):
            print(f"    {site:<12} {row['calls']:>3} calls  "
                  f"{row['prompt_tokens']:>8,} in  "
                  f"{row['completion_tokens']:>7,} out  "
                  f"{row['cached_tokens']:>8,} cached"
                  + (f"  {row['cache_write_tokens']:>8,} written"
                     if row.get("cache_write_tokens") else ""),
                  file=sys.stderr)
        return True

    if cmd == "memory":
        from . import memory
        rows = memory.entries(_root(args))
        if not rows:
            print("  (nothing remembered yet)", file=sys.stderr)
        for sc, cat, text in rows:
            print(f"  [{sc}/{cat}] {text}", file=sys.stderr)
        return True

    if cmd == "skills":
        from . import skills
        for name, info in sorted(skills._scan(_root(args), cfg).items()):
            print(f"  {name:<22} {info['description']}", file=sys.stderr)
        return True

    if cmd == "apply":
        if arg.lower() in ("on", "true", "yes"):
            args.apply = True
        elif arg.lower() in ("off", "false", "no"):
            args.apply = False
        else:
            print("  Usage: /apply on   or   /apply off", file=sys.stderr)
            return True
        print(f"  writes {'ON' if args.apply else 'off (dry run)'}", file=sys.stderr)
        return True

    if cmd == "steps":
        try:
            n = max(1, int(arg))
        except ValueError:
            print("  Usage: /steps 8", file=sys.stderr)
            return True
        # Also update args: a later `/profile` re-resolves config via
        # `_config(args)`, which applies args.steps on top.
        args.steps = n
        ref["cfg"].max_steps = n
        ref["agent"].cfg = ref["cfg"]
        print(f"  max_steps = {n}", file=sys.stderr)
        return True

    if cmd in ("profile", "model"):
        if not arg:
            print(f"  Usage: /{cmd} NAME", file=sys.stderr)
            return True
        try:
            label = _switch_profile(args, ref, arg,
                                    "profile" if cmd == "profile" else "model")
        except ConfigError as e:
            print(f"  {e}", file=sys.stderr)
            return True
        except LLMError as e:
            print(f"  Could not switch: {e}", file=sys.stderr)
            print(f"  Staying on {cfg.model}.", file=sys.stderr)
            return True
        print(f"  now using {label}", file=sys.stderr)
        return True

    print(f"  Unknown command '/{cmd}'. Try /help.", file=sys.stderr)
    return True


def prompts_pack(cfg) -> str:
    from . import prompts
    return prompts.pack_for(cfg)


def prompts_fingerprint(cfg) -> str:
    from . import prompts
    return prompts.fingerprint(cfg)


def _usage_line(client) -> str:
    """The closing `[N calls ...]` line: tokens, cache hit, cost."""
    s = client.stats
    bits = [f"{s['calls']} calls", f"{s['completion_tokens']} tokens out"]
    if s.get("prompt_tokens"):
        bits.append(f"{s['prompt_tokens']:,} in")
        if s.get("cached_tokens"):
            bits.append(f"cache hit "
                        f"{events_mod.cache_hit_ratio(s):.0%}")
        if s.get("cache_write_tokens"):
            bits.append(f"{s['cache_write_tokens']:,} written to cache")
    if s.get("cost"):
        bits.append(f"${s['cost']:.4f}")
    bits.append(f"{s['seconds']:.0f}s")
    return "[" + ", ".join(bits) + "]"


def _confirm_plan(plan_path) -> bool:
    """`--review-plan`: let the user read/edit the plan file, then continue."""
    try:
        ans = input(f"\nEdit {plan_path} if you like, then press Enter to run "
                    "it (n to cancel): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans not in ("n", "no", "q")


def _wants_review(args, cfg) -> bool:
    """--review or config `review`, unless --no-review overrides it."""
    if getattr(args, "no_review", False):
        return False
    return bool(getattr(args, "review", False) or cfg.review)


def _strict_replay_error(args, client) -> str:
    """The strict-replay failure text, or '' when the gate passes/disabled."""
    if not getattr(args, "strict_replay", False):
        return ""
    from . import replay as replay_mod
    if isinstance(client, replay_mod.Replayer) and client.diverged:
        return ("strict replay: the harness's prompts differed from the "
                f"recording at call(s) {client.diverged[:8]}")
    return ""


def _strict_task_error(args, tr) -> str:
    """The --strict failure text for one task result, or '' when the gate
    passes or is disabled: a failed edit block (a partial application) or
    impacted tests still red after the repair rounds."""
    if not getattr(args, "strict", False) or tr is None:
        return ""
    bad = [r for r in getattr(tr, "results", []) if not r.ok]
    if bad:
        detail = "; ".join(f"{r.file}: {r.message}" for r in bad)
        return f"strict: {len(bad)} edit block(s) failed: {detail[:300]}"
    if getattr(tr, "tests_ok", None) is False:
        return "strict: impacted tests still fail after the repair rounds"
    return ""


def _refine_task_text(cfg, client, root: Path, text: str, mode: str) -> str:
    """`--refine`: one Prompt/Spec Architect pass over a brief idea before it
    drives any pipeline. Prints the expanded, sectioned spec (objective /
    files / interface contracts / requirements / verification) to stderr and
    returns it; the original text comes back untouched if even the architect's
    deterministic fallback cannot run, so --refine never dead-ends a command."""
    from . import spec as spec_mod
    try:
        got = spec_mod.refine_prompt(
            text, root, cfg, roles_mod.client_for(cfg, "planner", client),
            mode=mode)
    except Exception as e:                       # must not lose the task
        print(f"  [refine failed ({e}); using the original prompt]",
              file=sys.stderr)
        return text
    print("  [prompt refined by the spec architect -- the run below uses "
          "this spec]", file=sys.stderr)
    print(got["refined"], file=sys.stderr)
    return got["refined"]


def _cmd_auto(args, cfg, client, idx, root: Path) -> int:
    """`pedacito auto`: generate a task list from a goal (or reuse a plan
    file), then implement every unchecked task in a fresh context window,
    optionally verifying each with --verify. Without --apply it only
    generates and prints the plan."""
    from . import scaffold

    workspace.log_event(root, "command: " + " ".join(sys.argv))

    if getattr(args, "worktree", ""):
        from . import worktree
        try:
            root = worktree.create(root, args.worktree)
        except worktree.WorktreeError as e:
            print(f"worktree error: {e}", file=sys.stderr)
            return 2
        print(f"  [running in git worktree {root}]", file=sys.stderr)
        idx = index_mod.build([str(root)], cfg, client=client, root=root,
                              summarise=False, verbose=True)

    plan_path = Path(args.plan) if getattr(args, "plan", None) else \
        (root / cfg.auto_plan_file)
    if not plan_path.is_absolute():
        plan_path = (root / plan_path).resolve()

    if args.task or args.spec:
        spec = scaffold.load_spec(
            prompt=args.task, spec_file=args.spec,
            max_chars=cfg.max_spec_chars)
        if getattr(args, "refine", False):
            spec.text = _refine_task_text(cfg, client, root, spec.text, "auto")
        # The full spec travels with every task prompt so constraints dropped
        # from the title still bind the worker.
        spec_text = spec.text
        # Resume an existing plan only while it has open tasks; otherwise a new
        # goal starts a fresh plan.
        regenerate = args.force or \
            loop.next_task(loop.read_plan(plan_path)) is None
        if regenerate:
            if plan_path.exists():
                print("  [existing plan has no open tasks; generating a new "
                      "one from the goal]", file=sys.stderr)
            titles = loop.generate_tasks(
                spec.text, roles_mod.client_for(cfg, "planner", client), cfg,
                root=root)
            # Persist the FULL spec, not just the title: a later `auto` resume
            # without -t/--spec reads the goal back from the plan file
            # (read_plan_goal) so workers still see every constraint.
            loop.write_plan(plan_path, spec.text, titles)
        if not args.apply:
            print(plan_path.read_text(encoding="utf-8"), file=sys.stderr)
            print(f"Plan: {plan_path}", file=sys.stderr)
            print("Run with --apply to implement it. Edit the checklist to "
                  "reorder, add, or remove tasks first if you like.",
                  file=sys.stderr)
            return 0
    else:
        if not plan_path.exists():
            print(f"No plan file at {plan_path}.\n"
                  "  Give -t \"goal\" or --spec FILE to generate one, or\n"
                  "  --plan FILE to point at an existing task checklist.",
                  file=sys.stderr)
            return 2
        # Resume of an existing plan with no fresh -t/--spec: the workers must
        # still see the ORIGINAL goal, or every constraint that was never
        # repeated in a task title is silently lost on the resumed run.
        spec_text = loop.read_plan_goal(plan_path)

    verify = args.verify if getattr(args, "verify", None) is not None \
        else cfg.auto_verify
    verify_final = bool(getattr(args, "verify_final", False)
                        or getattr(cfg, "auto_verify_final", False))
    repair = args.repair if getattr(args, "repair", None) is not None \
        else cfg.auto_repair_rounds
    max_tasks = args.max_tasks if getattr(args, "max_tasks", None) is not None \
        else cfg.auto_max_tasks
    max_minutes = args.max_minutes \
        if getattr(args, "max_minutes", None) is not None \
        else cfg.auto_max_minutes
    auto_install = bool(getattr(args, "auto_install", False)
                        or cfg.auto_install_deps)
    if getattr(args, "diagnostics", False):
        cfg.auto_diagnostics = True

    code = loop.run_loop(root, cfg, client, idx, plan_path,
                         apply=args.apply, verify=verify, repair_rounds=repair,
                         max_tasks=max_tasks, max_minutes=max_minutes,
                         stop_on_fail=not args.keep_going,
                         auto_install=auto_install, goal=spec_text,
                         strict=bool(getattr(args, "strict", False)),
                         verify_final=verify_final)

    # Post-run verification: exercise the project once (--verify and/or a smoke
    # run of main.py) and feed a failure back for repair.
    smoke = bool(cfg.auto_smoke)
    if args.apply and (verify or smoke):
        skip = {".pedacito", "__pycache__", ".venv", "venv",
                "node_modules", ".git"}
        # Prune skipped dirs and virtualenvs (any name, via pyvenv.cfg) while
        # walking so their contents are never traversed.
        py_rels = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip
                           and not is_venv_dir(Path(dirpath, d))]
            py_rels.extend(Path(dirpath, f).relative_to(root).as_posix()
                           for f in filenames if f.endswith(".py"))
        py_rels.sort()
        v_ok, v_out = scaffold.verify_and_fix(
            client, cfg, idx, py_rels, root.name, None,
            explicit_cmd=verify,
            rounds=int(cfg.auto_verify_rounds or 0),
            smoke=smoke,
            smoke_timeout=int(cfg.auto_smoke_timeout or 20),
            verbose=True)
        if v_ok:
            note = "verify: passed" + (f" ({v_out})" if v_out else "")
            print(f"  {note}", file=sys.stderr)
            workspace.log_event(root, note)
        else:
            print("\n  [warning] runtime verification failed after repair:\n"
                  + "\n".join(f"    {ln}" for ln in v_out.splitlines()[:10]),
                  file=sys.stderr)
            workspace.log_event(root, f"verify failed: {v_out[:200]}", "ERROR")
            if code == 0:
                code = 1
    return code


def _cmd_explain(args, client, agent) -> int:
    """`pedacito explain`: answer questions about the indexed project in prose
    (no edits). `main()` has already checked that an index exists."""
    from . import explain

    question = (getattr(args, "question_opt", None) or "").strip()

    def ask(q: str) -> str:
        print("\n--- looking things up ---", file=sys.stderr)
        return explain.answer(agent, q)

    if question:
        print(ask(question))
    elif not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if not piped:
            print('Usage: pedacito explain PROJECT -q "your question"',
                  file=sys.stderr)
            return 2
        print(ask(piped))
    else:
        print("Ask about this project's code. Type /quit to leave.",
              file=sys.stderr)
        while True:
            try:
                line = input("\nexplain> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
            if not line:
                continue
            if line.lower() in {"/quit", "/exit", "quit", "exit"}:
                break
            if line.lower() in {"/help", "help", "?"}:
                print("  Ask anything about this project's code: how a "
                      "function works,\n  what a file does, or how data or "
                      "control flows between parts.\n  Examples: \"how does the "
                      "sync client retry?\", \"what does src/app.py do?\",\n"
                      "  \"trace the flow from request to response\".",
                      file=sys.stderr)
                continue
            print()
            print(ask(line))

    print("\n" + _usage_line(client), file=sys.stderr)
    return 0


def _cmd_worktree(args) -> int:
    """`pedacito worktree create|list|remove`."""
    from . import worktree
    root = Path(args.root).expanduser().resolve() if args.root else Path(".").resolve()
    try:
        if args.action == "create":
            if not args.name:
                print("worktree create needs a NAME", file=sys.stderr)
                return 2
            dest = worktree.create(root, args.name, base=args.base)
            print(f"Worktree: {dest}")
            print(f"Run there with:  cd {dest} && pedacito auto ...",
                  file=sys.stderr)
            return 0
        if args.action == "list":
            entries = worktree.list_paths(root)
            for path, branch in entries:
                print(f"{path}  [{branch}]")
            return 0
        if args.action == "remove":
            if not args.name:
                print("worktree remove needs a NAME", file=sys.stderr)
                return 2
            worktree.remove(root, args.name, force=args.force)
            print(f"Removed worktree {args.name}", file=sys.stderr)
            return 0
        print(f"unknown worktree action: {args.action}", file=sys.stderr)
        return 2
    except worktree.WorktreeError as e:
        print(f"worktree error: {e}", file=sys.stderr)
        return 1


def _cmd_diagnose(args) -> int:
    """`pedacito diagnose`."""
    from . import diagnostics
    root = Path(args.root).expanduser().resolve() if args.root else Path(".").resolve()
    files = [args.file] if getattr(args, "file", "") else None
    ok, report = diagnostics.run(root, Config(), files=files)
    print(report)
    return 0 if ok else 1


def _light_config(args) -> Config:
    """Config for commands that never talk to a model (no vram_gb needed)."""
    return load_config(root=_root(args), explicit=getattr(args, "config", None),
                       profile=getattr(args, "profile", None),
                       require_hardware=False, warn_budgets=False)


def _cmd_calibrate(args, cfg) -> int:
    """`pedacito calibrate`: run the scripted mini-suite against the configured
    model and save its model card."""
    existing = modelcard.load(cfg)
    if existing and not args.recalibrate:
        print(modelcard.render(existing))
        print("\nA fresh card exists; pass --recalibrate to measure again.")
        return 0
    client = LLMClient(cfg)
    client.check()
    client.warmup()
    card = modelcard.calibrate(client, cfg, verbose=True, quick=args.quick,
                               repeats=getattr(args, "repeats", None))
    path = modelcard.save_card(card)
    print()
    print(modelcard.render(card))
    print(f"\nSaved to {path}. Settings you left at their defaults now follow "
          "the card (prompt pack, answer mode, verifier, best-of-N); explicit "
          "config values still win.")
    return 0


def _profile_base_urls() -> list[str]:
    """base_urls already recorded in the user config, so a re-scan re-finds a
    server on a custom port without the user repeating `--url`."""
    try:
        text = USER_CONFIG.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        flat, profiles = config_editor.validate_text(text)
    except ConfigError:
        return []
    urls: list[str] = []
    if flat.get("base_url"):
        urls.append(flat["base_url"])
    for body in profiles.values():
        if body.get("base_url"):
            urls.append(body["base_url"])
    return list(dict.fromkeys(urls))


def _ask_vram(label: str, base_url: str, args) -> int:
    """The VRAM of the machine running this server, asked once per server."""
    if getattr(args, "vram_gb", None):
        return int(args.vram_gb)
    if getattr(args, "yes", False):
        return 0                       # non-interactive: leave VRAM unset
    raw = _ask(f"GPU VRAM on the '{label}' server machine ({base_url}), in GB",
               "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def _scan_filter(models: list[dict], args) -> list[dict]:
    """Apply `--match`, `--all`, `--loaded` to one server's model list,
    keeping loaded models first."""
    pat = (getattr(args, "match", "") or "").lower()
    out = []
    for m in models:
        if pat and not fnmatch.fnmatch(m["id"].lower(), pat):
            continue
        if getattr(args, "loaded", False) and m.get("loaded") is not True:
            continue
        if not getattr(args, "all", False) and m.get("loaded") is False:
            continue
        out.append(m)
    out.sort(key=lambda m: (0 if m.get("loaded") else 1))
    return out


def _interactive_pick(label: str, base_url: str, models: list[dict]) -> list[dict]:
    """Let the user pick models from one server's list."""
    print(f"\n{label}  {base_url}  ({len(models)} model(s))")
    for i, m in enumerate(models, 1):
        mark = "*" if m.get("loaded") else " "
        bits = [f"{i:>2}) {mark} {m['id']}"]
        if m.get("size_bytes"):
            bits.append(f"{m['size_bytes'] / (1024 ** 3):.1f} GB")
        print("  " + "  ".join(bits))
    raw = _ask("Choose models (numbers, comma-separated, or 'all')", "all")
    if raw.strip().lower() in ("", "all"):
        return models
    picks: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                picks.update(range(int(a), int(b) + 1))
            except ValueError:
                pass
        else:
            try:
                picks.add(int(part))
            except ValueError:
                pass
    return [m for i, m in enumerate(models, 1) if i in picks]


def _read_config_text(target: Path) -> str:
    """The text to add profiles to: the target file if it has content, else
    the starter template (so a first scan produces a valid config)."""
    try:
        text = target.read_text(encoding="utf-8")
        if text.strip():
            return text
    except OSError:
        pass
    return TEMPLATE


def _write_config_backup(target: Path, text: str) -> None:
    """Validate, back up the old file as `.bak`, and write atomically."""
    config_editor.validate_text(text, str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            Path(str(target) + ".bak").write_bytes(target.read_bytes())
        except OSError:
            pass
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, target)


def _scan_cfg(model: dict) -> Config:
    """A Config for one scanned model, with identity only (no card applied)."""
    cfg = Config()
    cfg.base_url = model["base_url"]
    cfg.model = model["id"]
    if model.get("vram_gb"):
        cfg.vram_gb = int(model["vram_gb"])
    if model.get("tier"):
        cfg.tier = model["tier"]
    cfg.sources = {}
    return cfg


def _scan_calibrate(models: list[dict], quick: bool) -> None:
    """Calibrate the selected models, loaded first, skipping fresh cards. Each
    card is saved as soon as it finishes, so Ctrl-C keeps the cards done so
    far."""
    order = sorted(models, key=lambda m: (0 if m.get("loaded") else 1))
    todo: list[Config] = []
    for m in order:
        cfg = _scan_cfg(m)
        if modelcard.load(cfg):
            continue
        todo.append(cfg)
    if not todo:
        print("\nAll selected models already have a fresh card.")
        return
    print(f"\nCalibrating {len(todo)} model(s) "
          f"({'quick' if quick else 'full'}); this can take a few minutes each.")
    done = 0
    try:
        for cfg in todo:
            client = LLMClient(cfg)
            client.check()
            client.warmup()
            card = modelcard.calibrate(client, cfg, verbose=True, quick=quick)
            modelcard.save_card(card)
            done += 1
    except KeyboardInterrupt:
        print(f"\nInterrupted: kept the {done} card(s) finished so far.")
    except LLMError as e:
        print(f"\nCalibration stopped: {e}", file=sys.stderr)


def _is_hosted_url(url: str) -> bool:
    from urllib.parse import urlparse
    host = (urlparse(url or "").hostname or "").lower()
    return bool(host) and host not in ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def _host_label(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url or "").hostname or "hosted"


def _scan_hosted(args, urls: list[str]) -> list[dict]:
    """Hosted APIs are never auto-probed: the user passes --url + --api-key-env.
    The key is read from the environment, never written to disk, and `--match`
    is required because hosted catalogues can hold hundreds of models."""
    from . import providers
    if not urls:
        return []
    api_key_env = getattr(args, "api_key_env", None)
    if not api_key_env:
        print("A hosted --url needs --api-key-env VAR (the secret is never "
              "written to the config).", file=sys.stderr)
        return []
    key = os.environ.get(api_key_env, "")
    if not key:
        print(f"--api-key-env {api_key_env}: not set in the environment.",
              file=sys.stderr)
        return []
    pat = (getattr(args, "match", "") or "").lower()
    if not pat:
        print("Hosted catalogues can hold hundreds of models; pass "
              "--match PATTERN to choose which to add.", file=sys.stderr)
        return []
    out: list[dict] = []
    for url in urls:
        headers = {"Authorization": f"Bearer {key}"}
        for mid in discovery.list_models(url, headers=headers):
            if pat and not fnmatch.fnmatch(mid.lower(), pat):
                continue
            rec = {"id": mid, "base_url": url, "server_label": _host_label(url),
                   "tier": "hosted", "api_key_env": api_key_env}
            rec.update(providers.profile_overrides(url, mid))
            out.append(rec)
    return out


def _hosted_calibrate(models: list[dict], quick: bool) -> None:
    """Calibration for hosted models is opt-in: print a token estimate and
    confirm before spending."""
    est = len(models) * (60_000 if quick else 120_000)
    print(f"\nCalibrating hosted model(s) costs roughly {est:,} tokens.")
    ans = _ask("Proceed with hosted calibration? [y/N]", "n")
    if ans.lower() not in ("y", "yes"):
        print("Skipped hosted calibration.")
        return
    _scan_calibrate(models, quick=quick)


def _to_scan_entry(m: dict) -> dict:
    """The keys `upsert_profiles` writes for one scanned model."""
    entry: dict = {"model": m["id"], "base_url": m["base_url"],
                   "server_label": m.get("server_label", ""),
                   "tier": m.get("tier") or "",
                   "vram_gb": m.get("vram_gb"),
                   "quant": m.get("quant"),
                   "size_gb": (f"{m['size_bytes'] / (1024 ** 3):.1f}"
                               if m.get("size_bytes") else None),
                   "api_key_env": m.get("api_key_env"),
                   "prompt_cache": m.get("prompt_cache"),
                   "json_mode": m.get("json_mode"),
                   "tool_mode": m.get("tool_mode")}
    return entry


def _cmd_models_scan(args) -> int:
    """`pedacito models scan`: discover local servers and/or probe a hosted API,
    pick models, write identity profiles (with a backup), then optionally
    calibrate the chosen models."""
    urls = list(getattr(args, "url", None) or [])
    local_urls = [u for u in urls if not _is_hosted_url(u)]
    hosted_urls = [u for u in urls if _is_hosted_url(u)]

    servers = discovery.discover_servers(timeout=2.0, urls=local_urls,
                                         profile_urls=_profile_base_urls())
    selected: list[dict] = []
    if servers:
        for label, base_url, _ids in servers:
            vram = _ask_vram(label, base_url, args)
            models = discovery.describe_models(base_url, vram_gb=vram)
            picked = _scan_filter(models, args)
            chosen = picked if getattr(args, "yes", False) \
                else _interactive_pick(label, base_url, picked)
            for m in chosen:
                m["server_label"] = label
                m["vram_gb"] = vram
                if not m.get("tier"):
                    m["tier"] = _guess_tier(
                        m["id"], {"model": m["id"], "base_url": base_url}) or ""
            selected.extend(chosen)
    elif not hosted_urls:
        print("No local model servers answered.", file=sys.stderr)
        print("  Start one (LM Studio, Ollama, Lemonade, llama.cpp) and re-run,"
              " or pass --url to probe a specific address.", file=sys.stderr)
        return 1

    hosted = _scan_hosted(args, hosted_urls)
    if hosted_urls and not hosted:
        return 1
    selected += hosted

    if not selected:
        print("No models matched the selection.", file=sys.stderr)
        return 1

    target = Path(args.config) if getattr(args, "config", None) else USER_CONFIG
    text = _read_config_text(target)
    entries = [_to_scan_entry(m) for m in selected]
    new = config_editor.upsert_profiles(text, entries)
    if new == text:
        print("All selected models already have profiles.")
    else:
        _write_config_backup(target, new)
        print(f"Wrote {len(entries)} profile(s) to {target}")

    # Calibration: local defaults to quick; hosted is opt-in (only when
    # --calibrate is given explicitly).
    calibrate = getattr(args, "calibrate", None)
    if calibrate != "none":
        local = [m for m in selected if m.get("tier") != "hosted"]
        hosted_models = [m for m in selected if m.get("tier") == "hosted"]
        if local:
            _scan_calibrate(local, quick=(calibrate or "quick") != "full")
        if hosted_models and calibrate:
            _hosted_calibrate(hosted_models, quick=calibrate != "full")
    return 0


def _profiles_empty_note() -> str:
    """Why the profile table is empty, when the reason is a config file
    pedacito never opened: `XDG_CONFIG_HOME` replaces ~/.config, so the
    profiles the user wrote can sit in a shadowed file. Nothing is shadowed
    (or the file simply has no profiles): the plain where-to-write line."""
    note = config.xdg_shadow_note()
    if note:
        return note + "\n"
    return (f"  No profiles defined. Add [profiles.NAME] blocks to "
            f"{USER_CONFIG} (or pass --config), then re-run.\n")


def _cmd_profiles(args, cfg) -> int:
    """`pedacito profiles`: the shared table, optional recommendations, --json,
    and --apply-roles."""
    rows = profiles_mod.list_profiles(cfg)
    recs = profiles_mod.recommend(rows)
    if getattr(args, "json", False):
        import json as _json
        payload: dict = {"profiles": rows}
        if getattr(args, "recommend", False):
            payload["recommendations"] = recs
        print(_json.dumps(payload, default=str))
        return 0
    print(profiles_mod.render(
        rows, recs if getattr(args, "recommend", False) else None))
    if not rows:
        print(_profiles_empty_note(), end="", file=sys.stderr)
    if getattr(args, "apply_roles", False):
        return _cmd_profiles_apply(args, recs)
    return 0


def _cmd_profiles_apply(args, recs) -> int:
    """Write the recommended `roles = {...}` after confirmation."""
    target = Path(args.config) if getattr(args, "config", None) else USER_CONFIG
    text = _read_config_text(target)
    new = profiles_mod.apply_roles(text, recs)
    if new == text:
        print("No role recommendations to apply.")
        return 0
    roles = {role: rec["name"] for role, rec in recs.items()
             if role in profiles_mod._SPECIALIST_ROLES and rec.get("name")}
    if not getattr(args, "yes", False):
        ans = _ask(f"Write roles = {roles} to {target}? [y/N]", "n")
        if ans.lower() not in ("y", "yes"):
            print("Not written.")
            return 0
    _write_config_backup(target, new)
    print(f"Wrote roles = {roles} to {target}")
    return 0


def _cmd_prune(args) -> int:
    """`pedacito prune`: retention for sessions, backups, events and log.txt."""
    root = _root(args)
    cfg = _light_config(args)
    keep_s = args.keep_sessions if args.keep_sessions is not None \
        else cfg.keep_sessions
    keep_b = args.keep_backups if args.keep_backups is not None \
        else cfg.keep_backups
    removed = workspace.prune(root, keep_s, keep_b, cfg.keep_events,
                              cfg.log_max_bytes, dry_run=args.dry_run)
    verb = "would remove" if args.dry_run else "removed"
    print(f"{verb}: {removed['sessions']} session(s), {removed['backups']} "
          f"backup snapshot(s), {removed['events']} event file(s), "
          f"{removed['log_bytes']:,} byte(s) of log.txt")
    if not (keep_s or keep_b):
        print("Nothing to do for sessions/backups: pass --keep-sessions N and "
              "--keep-backups N (or set keep_sessions/keep_backups).")
    return 0


def _reset_worktrees(root: Path) -> list[tuple[str, str]]:
    """Git worktrees registered directly under `.pedacito/worktrees/`."""
    from . import worktree
    base = workspace.state_dir(root, "worktrees")
    try:
        base = base.resolve()
        entries = worktree.list_paths(root)
    except (OSError, worktree.WorktreeError):
        return []
    out: list[tuple[str, str]] = []
    for path, branch in entries:
        try:
            p = Path(path).resolve()
        except OSError:
            continue
        if p.parent == base:
            out.append((p.name, branch))
    return sorted(out)


def _reset_reindex(args, root: Path) -> int:
    """Rebuild the index after `pedacito reset --reindex`."""
    try:
        cfg = _config(args)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    no_summaries = bool(getattr(args, "no_summaries", False))
    client = None
    try:
        if not no_summaries:
            client = LLMClient(cfg)
            client.check()
            client = roles_mod.client_for(cfg, "summarizer", client)
        idx = index_mod.build([str(root)], cfg, client=client, root=root,
                              summarise=not no_summaries,
                              matcher=_matcher(args, root))
    except (LLMError, OSError) as e:
        print(f"reset: index rebuild failed: {e}", file=sys.stderr)
        print("The state was reset; run `pedacito index PROJ` to retry.",
              file=sys.stderr)
        return 1
    print(f"Indexed {len(idx.chunks)} units across {len(idx.files)} files "
          f"({len(idx.code_files)} code, {len(idx.reference_files)} reference) "
          f"-> {index_mod.state_dir(root)}", file=sys.stderr)
    return 0


def _cmd_reset(args) -> int:
    """`pedacito reset`: wipe this project's `.pedacito/` runtime state.

    Code, `.pedacito.toml`, `.pedacitoignore` and `.gitignore` are untouched.
    Registered background servers are stopped first; live git worktrees under
    `.pedacito/worktrees/` refuse the reset unless --force is given.
    """
    from . import processes, worktree

    root = _root(args)
    if not root.is_dir():
        print(f"reset: no such project directory: {root}", file=sys.stderr)
        return 2
    d = workspace.state_dir(root)
    if d.is_symlink():
        print(f"reset: refusing to reset {d}: it is a symlink", file=sys.stderr)
        return 1
    dry_run = bool(getattr(args, "dry_run", False))
    keep_memory = bool(getattr(args, "keep_memory", False))
    keep_config = bool(getattr(args, "keep_config", False))
    force = bool(getattr(args, "force", False))

    lock_held = False
    if not dry_run:
        try:
            workspace.acquire_lock(root)
            lock_held = True
        except workspace.ProjectBusy as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
    try:
        running = [r for r in processes.list_procs(root)
                   if r.get("status") == "running"]
        wts = _reset_worktrees(root)
        if wts and not force:
            names = ", ".join(name for name, _ in wts)
            print(f"reset: {len(wts)} git worktree(s) live under "
                  f"{d}/worktrees ({names}).\n  Remove them with "
                  "`pedacito worktree remove NAME` first, or pass --force to "
                  "discard anything uncommitted there.", file=sys.stderr)
            return 2

        try:
            plan = workspace.reset(root, keep_memory=keep_memory,
                                   keep_config=keep_config, dry_run=True)
        except OSError as e:
            print(f"reset: {e}", file=sys.stderr)
            return 1

        size = plan["bytes"]
        size_text = (f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024
                     else f"{size / 1024:.1f} KB")
        print(f"{'Would reset' if dry_run else 'Resetting'} {d}: "
              f"{plan['files']} file(s), {size_text}", file=sys.stderr)
        if plan["removed"]:
            shown = ", ".join(plan["removed"][:12])
            extra = len(plan["removed"]) - 12
            print(f"  remove: {shown}" + (f" (+{extra} more)" if extra > 0 else ""),
                  file=sys.stderr)
        if plan["kept"]:
            print(f"  keep  : {', '.join(plan['kept'])}", file=sys.stderr)
        if running:
            print(f"  {'stop' if dry_run else 'stopping'}: "
                  + ", ".join(str(r.get("name")) for r in running),
                  file=sys.stderr)
        if wts:
            print(f"  {'remove' if dry_run else 'removing'} worktree(s): "
                  + ", ".join(name for name, _ in wts), file=sys.stderr)

        if not dry_run and not getattr(args, "yes", False) \
                and sys.stdin.isatty():
            try:
                ans = input("Delete this state? Code is untouched. [y/N] "
                            ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            if ans not in ("y", "yes"):
                print("Nothing reset.", file=sys.stderr)
                return 1

        if dry_run:
            print("Dry run: nothing was deleted.", file=sys.stderr)
            return 0

        for rec in running:
            msg = processes.stop(root, str(rec.get("name") or ""))
            if msg.startswith("ERROR"):
                print(f"  warning: {msg}", file=sys.stderr)
            else:
                print(f"  {msg}", file=sys.stderr)
        for name, _branch in wts:
            worktree.remove(root, name, force=True)
            print(f"  removed worktree {name}", file=sys.stderr)
        try:
            result = workspace.reset(root, keep_memory=keep_memory,
                                     keep_config=keep_config)
        except OSError as e:
            print(f"reset: {e}", file=sys.stderr)
            return 1
        print(f"  cleared {result['files']} file(s) under {d}", file=sys.stderr)
        if getattr(args, "reindex", False):
            return _reset_reindex(args, root)
    finally:
        if lock_held:
            workspace.release_lock(root)

    print("Code and project config were left untouched. "
          "Run `pedacito index PROJ` when you want a fresh map.",
          file=sys.stderr)
    return 0


def _cmd_serve(args) -> int:
    """`pedacito serve`: start a background dev server and register it."""
    from . import processes
    root = _root(args)
    cfg = _light_config(args)
    command = [c for c in (getattr(args, "command", None) or [])
               if c != "--"]
    if not command:
        print("Usage: pedacito serve --root PROJ [--name NAME] -- COMMAND...",
              file=sys.stderr)
        return 2
    try:
        rec = processes.start(
            root, command, name=getattr(args, "name", "") or "",
            cfg=cfg,
            wait=(args.wait if getattr(args, "wait", None) is not None
                  else cfg.serve_wait))
    except (OSError, ValueError) as e:
        print(f"serve error: {e}", file=sys.stderr)
        return 2
    print(f"{rec['name']} [{rec.get('status')}] pid {rec.get('pid')}")
    if rec.get("url"):
        print(f"url: {rec['url']}")
    print(f"log: {rec['log']}")
    return 0 if rec.get("status") == "running" else 1


def _cmd_procs(args) -> int:
    """`pedacito procs`: list, show logs from, or stop background servers."""
    from . import processes
    root = _root(args)
    action = getattr(args, "action", "list") or "list"
    name = getattr(args, "name", "") or ""
    if action == "list":
        print(processes.render_list(root))
        return 0
    if not name:
        print(f"procs {action} needs a process NAME", file=sys.stderr)
        return 2
    if action == "logs":
        print(processes.logs(root, name, tail=args.tail))
        return 0
    if action == "stop":
        print(processes.stop(root, name))
        return 0
    print(f"unknown procs action: {action}", file=sys.stderr)
    return 2


def _cmd_lsp(args) -> int:
    """`pedacito lsp`: definition/references/hover/diagnostics via a server."""
    from . import lsp
    root = _root(args)
    cfg = _light_config(args)
    rel = (args.file or "").replace("\\", "/")
    try:
        rel = Path(rel).resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        pass
    print(lsp.result(root, cfg, args.op, rel,
                     line=args.line, column=args.column))
    return 0


def _cmd_mcp(args) -> int:
    """`pedacito mcp list`: every tool the named MCP servers expose, with the
    trust flag each call would show or skip."""
    from . import mcp
    cfg = _light_config(args)
    if args.action != "list":
        print(f"unknown mcp action: {args.action}", file=sys.stderr)
        return 2
    servers = mcp.named_servers(cfg)
    if not servers:
        if not getattr(cfg, "enable_mcp", True):
            print("MCP is disabled (config enable_mcp = false).")
            return 0
        print("No named MCP servers. Add [mcp_servers.<name>] tables "
              "(command + args for a stdio server, or url for a Streamable "
              "HTTP one) to the config. A plain mcp_servers command list "
              "serves the legacy `mcp` action instead.")
        return 0
    tools = mcp.list_mcp_tools(cfg)
    if not tools:
        print(f"No tools listed by {len(servers)} server(s): they are down, "
              "slow, or expose none.")
        return 1
    for d in tools:
        line = mcp.compact_line(d)
        if not d["trusted"]:
            line += "   [asks permission]"
        print(line)
    print(f"\n{len(tools)} tool(s) on {len(servers)} server(s). Gather "
          "injects only the tools matching the current task; the model calls "
          "them with the `mcp_call` action.")
    return 0


def _cmd_memory(args) -> int:
    """`pedacito memory`: list, add or forget remembered facts."""
    from . import memory
    root = _root(args)
    scope = "global" if args.use_global else "project"
    if args.add:
        ok, msg = memory.remember(root, args.add, scope=scope)
        print(msg)
        return 0 if ok else 1
    if args.forget:
        n = memory.forget(root, args.forget, scope=scope)
        print(f"forgot {n} line(s)")
        return 0
    rows = memory.entries(root)
    if not rows:
        print("No memory yet. The model saves facts with its `remember` "
              "action; add one with: pedacito memory PROJECT --add \"...\"")
        return 0
    for sc, cat, text in rows:
        print(f"[{sc}/{cat}] {text}")
    return 0


def _cmd_trust(args) -> int:
    """`pedacito trust`: approve this project's `.pedacito.toml` to set
    execution-capable settings (hooks, mcp_servers, auto_verify, test_command)."""
    from .config import EXEC_FIELDS, trust_project
    root = _root(args)
    path = find_project_config(root)
    if path is None:
        print(f"No .pedacito.toml / pedacito.toml in {root}; nothing to trust.")
        return 0
    print(f"{path}:\n")
    print(path.read_text(encoding="utf-8", errors="replace"))
    print(f"\nSettings that can run commands: {', '.join(EXEC_FIELDS)}.")
    if not args.yes:
        try:
            ans = input("Trust this file as written? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans not in ("y", "yes"):
            print("Not trusted.")
            return 1
    trust_project(path)
    print("Trusted (until the file changes).")
    return 0


def _cmd_tests(args) -> int:
    """`pedacito tests`: the detected test command and, for given files, the
    tests a change to them can break."""
    from . import testsel
    root = _root(args)
    cfg = _light_config(args)
    cmd = testsel.detect_test_command(root, cfg)
    print(f"test command: {cmd or '(none detected; set test_command)'}")
    if args.files:
        idx = index_mod.Index.load(root)
        if idx is None:
            print("No index yet; run `pedacito index` first.")
            return 1
        rels = [idx.resolve_file(f) or f for f in args.files]
        picked = testsel.impacted_tests(idx, rels)
        if not picked:
            print("no linked tests found")
        for t, why in picked:
            print(f"  {t}  ({why})")
        narrowed, did = testsel.narrow_command(cmd, [t for t, _ in picked]) \
            if cmd else ("", False)
        if cmd and did:
            print(f"narrowed: {narrowed}")
    return 0


def _cmd_webcheck(args) -> int:
    """`pedacito webcheck`: static cross-file contract checks for web
    projects (DOM ids, ESM behind classic scripts, duplicate globals,
    undeclared imports, undefined members). Findings go to stdout in the
    usual `rel:line: message` shape; the exit code gates on them."""
    from . import webcheck
    root = _root(args)
    files = [Path(f).as_posix() for f in args.files] if args.files else None
    found = webcheck.run(root, files=files)
    if not found:
        print("no web contract problems found (looked at every .html/.js "
              "outside node_modules and friends)")
        return 0
    for f in found:
        print(f)
    print(f"{len(found)} finding(s); re-run after fixing to confirm")
    return 1


def _cmd_skills(args) -> int:
    from . import skills
    root = _root(args)
    cfg = _light_config(args)
    found = skills._scan(root, cfg)
    for name in sorted(found):
        info = found[name]
        print(f"{name:<22} {info['description']}  [{info['source']}]")
    return 0


def _cmd_rename(args) -> int:
    """`pedacito rename <old> <new>`: the deterministic symbol rename.
    Zero LLM tokens: Python identifiers move through the real tokenizer,
    tree-sitter languages through their identifier nodes (so a string literal
    or comment that merely contains the old name survives), and every changed
    file passes the AST syntax gate BEFORE anything is written. Default is a
    dry-run diff preview; --apply writes, snapshotting the originals under
    .pedacito/backups/ so `pedacito restore` can undo the run."""
    from . import fileio, refactor
    root = _root(args)
    state = {"dir": None}

    def _snapshot_before_write(rel: str, path: Path):
        if state["dir"] is None:
            state["dir"] = workspace.new_backup_dir(root, unique=True)
        fileio.backup_copy(path, state["dir"] / rel)
        return None

    res = refactor.rename_symbol_across_project(
        root, args.old, args.new, scope_file=args.file or None,
        dry_run=not args.apply,
        before_write=_snapshot_before_write if args.apply else None)
    if not res["ok"]:
        print(f"rename refused: {res.get('error', 'unknown failure')}",
              file=sys.stderr)
        return 2
    if not res["files_modified"]:
        print(f"no exact identifier `{args.old}` found in the "
              f"{'scoped' if args.file else 'project'} source; nothing "
              "renamed.")
        if res.get("skipped"):
            print("skipped: " + "; ".join(res["skipped"][:8]))
        return 0
    verb = "renamed" if args.apply else "would rename"
    print(f"{verb} `{args.old}` -> `{args.new}`: "
          f"{res['total_replacements']} occurrence(s) in "
          f"{len(res['files_modified'])} file(s)")
    for rel in res["files_modified"]:
        print(f"  {rel}: {res['per_file'].get(rel, 0)} replacement(s)")
    for w in res.get("warnings", []):
        print(f"  warning: {w}")
    for s in res.get("skipped", [])[:8]:
        print(f"  not checked: {s}")
    if not args.apply:
        for rel in res["files_modified"]:
            diff = res["diffs"].get(rel, "")
            if diff:
                print()
                print(diff.rstrip())
        print("\n(dry run: nothing written; add --apply to write)")
    elif state["dir"] is not None:
        print(f"\noriginals snapshotted under {state['dir']}; "
              "`pedacito restore` undoes the rename.")
    return 0


def _cmd_rpc(args, cfg, client, idx, root, locals_ref, run, save_checkpoint,
             elog) -> int:
    """`pedacito rpc` / `pedacito chat --rpc`: the bidirectional JSONL-RPC
    loop over stdio (protocol in `pedacito/rpc.py`). The project index, the
    model profiles and the session ledger stay warm between requests; every
    stdout byte is one JSON event and every human-facing line goes to stderr,
    so an editor or another agent can drive this process without parsing
    prose. Clients can `ask`, `steer` a running task at its next step
    boundary, answer `permission_request` events, `cancel` without killing
    the session, and switch profiles in place."""
    from . import explain as explain_mod
    from . import rpc as rpc_mod
    from . import security

    server_box: dict = {}

    def _tag():
        s = server_box.get("s")
        return s._tag_for(None) if s is not None else None

    real_stdout = sys.stdout
    out = rpc_mod.RpcOut(real_stdout, _tag)
    elog.tee = out
    sys.stdout = sys.stderr   # anything that forgets to say stderr still
                              # cannot corrupt the JSONL stream

    locked = {"done": False}

    def _ensure_lock():
        """Take the run lock the first time a request asks for writes, and
        keep it until the process exits (same lock `ask --apply` uses)."""
        if locked["done"]:
            return
        try:
            workspace.acquire_lock(root)
        except workspace.ProjectBusy as e:
            raise SystemExit(str(e)) from e
        locked["done"] = True
        _LOCKED.append(root)

    def _rpc_run(task, apply=False, refine=False, plan=None):
        s = server_box["s"]
        s.set_client(locals_ref["client"])
        cur = locals_ref["cfg"]
        # steer() pushes into the queue; the agent drains it at a step
        # boundary, which never rewrites sent history -> prompt cache safe
        cur._steer = s.steer_queue()
        try:
            return run(task, apply=apply, refine=refine, plan=plan)
        finally:
            cur.__dict__.pop("_steer", None)

    def _rpc_explain(question):
        server_box["s"].set_client(locals_ref["client"])
        return explain_mod.answer(locals_ref["agent"], question)

    def _rpc_status():
        cur = locals_ref["cfg"]
        c = locals_ref["client"]
        now = locals_ref["idx"]
        s = c.stats or {}
        roles = dict(cur.roles or {})
        return {
            "profile": cur.active_profile or "(none)",
            "model": cur.model,
            "tier": cur.active_tier,
            "scout": (str(getattr(cur, "scout_profile", "") or "").strip()
                      or roles.get("scout") or "(none)"),
            "steps": cur.max_steps,
            "writes": "per request (ask params.apply)",
            "index": {"units": len(now.chunks), "files": len(now.files),
                      "code_files": len(now.code_files),
                      "reference_files": len(now.reference_files)},
            "session": {k: s.get(k) for k in
                        ("calls", "prompt_tokens", "completion_tokens",
                         "cached_tokens", "cost", "seconds")},
            "cache_hit": events_mod.cache_hit_ratio(s),
        }

    def _rpc_switch(ref, kind):
        if kind == "scout":
            cur = locals_ref["cfg"]
            if ref not in (cur.profiles or {}):
                have = ", ".join(sorted(cur.profiles or {})) or "none"
                raise ConfigError(f"unknown profile {ref!r}; available: {have}")
            # Gather routing reads `scout_profile` (roles.scout_client); a
            # roles entry would be inert. Clear the cached reachability
            # verdict so the next gather actually re-resolves the new one.
            cur.scout_profile = ref
            cur.__dict__.pop("_scout_clients", None)
            return f"scout -> profile {ref}"
        label = _switch_profile(args, locals_ref, ref, "profile")
        server_box["s"].set_client(locals_ref["client"])
        return label

    def _always_apply(category, key, setting_key):
        """The GUI's `_always_allow` behaviour, in-memory only: flip the
        matching knob on the LIVE config so this session stops asking; never
        persisted to the user's config file."""
        cur = locals_ref["cfg"]
        setting = setting_key or {
            security.CAT_WEB: "allow_fetch",
            security.CAT_SHELL: "allow_shell",
            security.CAT_PACKAGE_INSTALL: "auto_install_deps",
            security.CAT_TEST_EXECUTION: "enable_run_test_in_gather",
        }.get(category)
        if setting:
            try:
                setattr(cur, setting, True)
            except (TypeError, AttributeError):
                pass
        if category == security.CAT_WEB and key:
            hosts = tuple(getattr(cur, "fetch_allowed_hosts", ()) or ())
            if "*" not in hosts and key not in hosts:
                cur.fetch_allowed_hosts = hosts + (key,)

    timeout = (getattr(args, "permission_timeout", None)
               or rpc_mod.PERMISSION_TIMEOUT)
    server = rpc_mod.RpcServer(
        run=_rpc_run, explain=_rpc_explain, status=_rpc_status,
        switch=_rpc_switch, ensure_lock=_ensure_lock,
        save_checkpoint=save_checkpoint, always_apply=_always_apply,
        permission_timeout=float(timeout), output=out, input_stream=sys.stdin)
    server_box["s"] = server
    server.set_client(client)
    print("pedacito rpc: JSONL-RPC over stdin/stdout - one request per line "
          "({\"id\":1,\"method\":\"status\"}), Ctrl-D to exit.",
          file=sys.stderr)
    try:
        return server.serve()
    finally:
        elog.tee = None
        sys.stdout = real_stdout


def _configure_stdio() -> None:
    """Make stdout/stderr UTF-8 with replacement so non-ASCII output cannot
    raise UnicodeEncodeError on Windows code pages. Streams without
    `reconfigure` (StringIO harnesses) are skipped."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


_LOCKED: list = []        # project roots this process holds the run lock for


def main(argv=None) -> int:
    """Parse arguments and dispatch to the subcommand; returns the exit code."""
    real_stdout = sys.stdout
    try:
        return _main(argv)
    finally:
        sys.stdout = real_stdout     # `--jsonl` points it at stderr meanwhile
        while _LOCKED:
            workspace.release_lock(_LOCKED.pop())


def build_parser() -> argparse.ArgumentParser:
    """The full CLI parser, split out from dispatch so tests can enumerate
    the real subcommand set (the `how` index drift guard)."""
    ap = argparse.ArgumentParser(prog="pedacito", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="build or refresh the index")
    _add_common(p_index)
    p_index.add_argument("--no-summaries", action="store_true",
                         help="static index only; instant, no LLM calls")
    p_index.add_argument("--resummarise", "--resummarize", action="store_true",
                         dest="resummarise",
                         help="discard cached summaries and regenerate with the "
                              "current model")

    p_map = sub.add_parser("map", help="print the current index")
    _add_common(p_map)
    p_map.add_argument("--tier", choices=["auto", "cards", "dirs", "flat", "ranked"],
                       default="auto")
    p_map.add_argument("-t", "--task", default="",
                       help="rank the map for this task (as `ask` would see it)")

    p_conf = sub.add_parser("config", help="show, create, or test the config file")
    p_conf.add_argument("--init", action="store_true",
                        help=f"write a starter config to {USER_CONFIG}")
    p_conf.add_argument("--path", metavar="FILE", help="write --init here instead")
    p_conf.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_conf.add_argument("--check", action="store_true",
                        help="connect to the model server and verify the model is loaded")
    p_conf.add_argument("--doctor", action="store_true",
                        help="like --check, plus probe which structured-output "
                             "mode (json_schema / json_object / plain prompting) "
                             "this server+model actually supports, and report "
                             "its context length, tokenizer ratio and the "
                             "tree-sitter syntax-check status")
    p_conf.add_argument("--setup", action="store_true",
                        help="interactive first-run setup: finds a local server, "
                             "writes and validates a minimal config")
    p_conf.add_argument("--reference", action="store_true",
                        help="print (or with --path, write) the full annotated "
                             "config template")
    p_conf.add_argument("--root", default=None)
    p_conf.add_argument("--profile", metavar="NAME")
    p_conf.add_argument("paths", nargs="*", default=["."], help=argparse.SUPPRESS)
    p_conf.add_argument("--config", metavar="FILE")

    p_how = sub.add_parser(
        "how", help="ask how to use pedacito (smart built-in help)")
    p_how.add_argument("question", nargs="*",
                       help="your question; omit for an interactive prompt")
    p_how.add_argument("-q", "--question", dest="question_opt", metavar="TEXT",
                       help="your question (alternative to the positional form)")
    p_how.add_argument("--offline", "--no-model", "--fast", dest="offline",
                       action="store_true",
                       help="print the deterministic built-in reference only; "
                            "no model server is needed or contacted")
    p_how.add_argument("--json", action="store_true",
                       help="print structured matches (GUI paths, CLI commands, "
                            "settings) plus the formatted answer as JSON")
    p_how.add_argument("--url", "--server", dest="url",
                       help="server base url override for this help session")
    p_how.add_argument("--api-key", dest="api_key",
                       help="API key override for this help session")
    p_how.add_argument("--model", help="model id/name override")
    p_how.add_argument("--vram", "--vram-gb", dest="vram_gb", type=int,
                       metavar="GB", help="GPU VRAM in whole gigabytes")
    p_how.add_argument("--config", metavar="FILE",
                       help="use this config file as well")
    p_how.add_argument("--profile", metavar="NAME",
                       help="apply a [profiles.NAME] block from your config")

    p_rest = sub.add_parser("restore", help="undo an --apply run from its backup snapshot")
    p_rest.add_argument("paths", nargs="*", default=["."],
                        help="the project (default: current directory)")
    p_rest.add_argument("--snapshot", metavar="NAME",
                        help="which snapshot; default is the newest")
    p_rest.add_argument("--root", default=None)
    p_rest.add_argument("--list", action="store_true", help="list snapshots and exit")
    p_rest.add_argument("--dry-run", action="store_true")
    p_rest.add_argument("--delete-created", action="store_true",
                        dest="delete_created",
                        help="also delete the files that run created (they "
                             "have no original to restore)")
    p_rest.add_argument("--config", metavar="FILE")

    p_wt = sub.add_parser(
        "worktree",
        help="create/list/remove an isolated git worktree for risky runs")
    p_wt.add_argument("action", choices=["create", "list", "remove"])
    p_wt.add_argument("name", nargs="?", default="")
    p_wt.add_argument("--root", default=None,
                      help="the repository (default: current directory)")
    p_wt.add_argument("--base", default="HEAD",
                      help="commit/branch for a new worktree (default: HEAD)")
    p_wt.add_argument("--force", action="store_true",
                      help="with remove: discard modified/untracked files")

    p_diag = sub.add_parser(
        "diagnose", help="run the project's detected linter/typechecker")
    p_diag.add_argument("--root", default=None,
                        help="the project (default: current directory)")
    p_diag.add_argument("--file", default="",
                        help="restrict to one project-relative file")

    p_files = sub.add_parser("files", help="show which files would be indexed, and why")
    _add_common(p_files)

    p_cost = sub.add_parser("cost", help="estimate indexing cost without spending it")
    _add_common(p_cost)
    p_cost.add_argument("--tps", type=float, default=25.0, help="your tokens/sec")

    p_ask = sub.add_parser("ask", help="run one task")
    _add_common(p_ask)
    p_ask.add_argument("-t", "--task", required=True)
    p_ask.add_argument("--apply", action="store_true", help="write edits to disk")
    p_ask.add_argument("--review", action="store_true",
                       help="open a side-by-side review page and apply from there")
    p_ask.add_argument("--no-review", dest="no_review", action="store_true",
                       help="never open the reviewer, even if config enables it")
    p_ask.add_argument("--reindex", action="store_true", help="force a fresh index first")
    p_ask.add_argument("--continue", dest="continue_", action="store_true",
                       help="resume the newest ask checkpoint as context")
    p_ask.add_argument("--plan", action="store_true",
                       help="split the task into single-focus steps, each run "
                            "in a fresh context (with one re-plan after a "
                            "failure); automatic for multi-part tasks when "
                            "plan_threshold is set, and for tasks that name "
                            "5+ files")
    p_ask.add_argument("--no-plan", dest="no_plan", action="store_true",
                       help="never auto-plan this task")
    p_ask.add_argument("--review-plan", dest="review_plan", action="store_true",
                       help="show the plan and ask before running it")
    p_ask.add_argument("--refine", action="store_true",
                       help="expand the task into a repo-aware specification "
                            "(objective, files, interface contracts, edge "
                            "cases, verification) with the spec architect "
                            "before running it")

    p_explain = sub.add_parser(
        "explain", help="answer questions about an indexed project's code")
    _add_common(p_explain)
    p_explain.add_argument("-q", "--question", dest="question_opt", metavar="TEXT",
                           help="your question; omit for an interactive prompt")

    p_new = sub.add_parser(
        "new", help="start a project from scratch: plan, scaffold, build")
    p_new.add_argument("project", help="directory for the new project "
                                       "(created if missing; must be empty)")
    p_new.add_argument("-t", "--task", default=None,
                       help="one-line project description")
    p_new.add_argument("--spec", metavar="FILE", default=None,
                       help="a larger spec document (.md/.txt); may be combined "
                            "with -t, or piped in via stdin")
    p_new.add_argument("--plan-only", dest="plan_only", action="store_true",
                       help="stop after the plan + header skeleton; implement "
                           "later with ordinary `ask --apply` runs")
    p_new.add_argument("--design", action="store_true",
                       help="design interfaces, data shapes and tests first; "
                            "save .pedacito/design.md and build tests before "
                            "sources")
    p_new.add_argument("--apply", action="store_true",
                       help="write the generated files to disk (default: dry run)")
    p_new.add_argument("--review", action="store_true",
                       help="build, then sign off on everything in the "
                            "side-by-side reviewer")
    p_new.add_argument("--force", action="store_true",
                       help="allow scaffolding into a non-empty directory")
    p_new.add_argument("--complex", action="store_true",
                       help="with --apply: after planning and scaffolding, hand "
                            "implementation to the autonomous loop (one task per "
                            "file, fresh context each, resumable PLAN.md)")
    p_new.add_argument("--verify", default=None,
                       help="(with --complex) shell command run after each task, "
                            "e.g. 'pytest -q'; failures feed back for repair")
    p_new.add_argument("--repair", type=int, default=None,
                       help="(with --complex) fix rounds per failing verification")
    p_new.add_argument("--max-tasks", type=int, default=None,
                       help="(with --complex) stop after attempting this many tasks")
    p_new.add_argument("--max-minutes", type=int, default=None,
                       help="(with --complex) stop before starting a task once "
                            "this many minutes have elapsed (0 = no limit)")
    p_new.add_argument("--keep-going", action="store_true",
                       help="(with --complex) continue past failed tasks "
                            "instead of stopping")
    p_new.add_argument("--auto-install", dest="auto_install",
                       action="store_true",
                       help="(with --complex --verify) pip-install a missing "
                            "Python package and retry instead of reporting it")
    p_new.add_argument("--url", help="server base url override")
    p_new.add_argument("--model", help="model id/name override")
    p_new.add_argument("--config", metavar="FILE", help="use this config file as well")
    p_new.add_argument("--profile", metavar="NAME",
                       help="apply a [profiles.NAME] block from your config")
    p_new.add_argument("--no-review", dest="no_review", action="store_true")
    p_new.add_argument("--refine", action="store_true",
                       help="expand the project description into a concrete, "
                            "sectioned spec (with interface contracts) before "
                            "planning")

    p_auto = sub.add_parser(
        "auto", help="autonomous loop: plan a goal, then implement its tasks "
                     "one at a time in fresh contexts")
    _add_common(p_auto)
    p_auto.add_argument("-t", "--task", default=None,
                        help="one-line goal to break into tasks")
    p_auto.add_argument("--spec", metavar="FILE", default=None,
                        help="a larger spec document (.md/.txt); may be combined "
                             "with -t, or piped in via stdin")
    p_auto.add_argument("--plan", metavar="FILE", default=None,
                        help="use this task-list file (default: PLAN.md in the "
                             "project root)")
    p_auto.add_argument("--apply", action="store_true",
                        help="implement the tasks and write edits (default: "
                             "generate/print the plan only)")
    p_auto.add_argument("--verify", default=None,
                        help="shell command to run after each task, e.g. "
                             "'pytest -q'; a failure feeds back for repair")
    p_auto.add_argument("--verify-final", action="store_true",
                        help="with --verify, run the command once after the "
                             "LAST unchecked task instead of after every "
                             "task: a whole-goal check cannot pass mid-plan, "
                             "so per-task runs roll back valid partial fixes "
                             "(cheap gates still run every step)")
    p_auto.add_argument("--repair", type=int, default=None,
                        help="how many fix rounds a failing verification gets")
    p_auto.add_argument("--diagnostics", action="store_true",
                        help="after each task, run the project's detected "
                             "linter/typechecker and feed findings into the "
                             "repair rounds (only when --verify is not set)")
    p_auto.add_argument("--refine", action="store_true",
                        help="expand the goal into a concrete, sectioned spec "
                             "(objective, files, interface contracts, edge "
                             "cases, verification) before planning tasks")
    p_auto.add_argument("--worktree", default="",
                        help="run the plan in an isolated git worktree "
                             ".pedacito/worktrees/NAME (created from --base, "
                             "default HEAD)")
    p_auto.add_argument("--max-tasks", type=int, default=None,
                        help="stop after attempting this many tasks")
    p_auto.add_argument("--max-minutes", type=int, default=None,
                        help="stop before starting a task once this many "
                             "minutes have elapsed (0 = no limit, the default)")
    p_auto.add_argument("--keep-going", action="store_true",
                        help="continue past failed tasks instead of stopping")
    p_auto.add_argument("--auto-install", dest="auto_install",
                        action="store_true",
                        help="when --verify fails on a missing Python package, "
                             "pip-install it and retry (off by default)")
    p_auto.add_argument("--force", action="store_true",
                        help="regenerate the plan file, discarding the existing one")

    p_chat = sub.add_parser("chat", help="interactive session")
    _add_common(p_chat)
    p_chat.add_argument("--apply", action="store_true")
    p_chat.add_argument("--review", action="store_true")
    p_chat.add_argument("--no-review", dest="no_review", action="store_true")
    p_chat.add_argument("--resume", action="store_true",
                        help="seed the first task with the newest chat "
                             "checkpoint")
    p_chat.add_argument("--rpc", action="store_true",
                        help="drive the session over stdin/stdout as "
                             "newline-delimited JSON (see `pedacito rpc`)")

    p_rpc = sub.add_parser(
        "rpc", help="interactive JSONL-RPC over stdio for editors and other "
                    "machine clients (index and profiles stay warm)")
    _add_common(p_rpc, paths_nargs="?")
    from .rpc import PERMISSION_TIMEOUT as _PT
    p_rpc.add_argument("--permission-timeout", type=float, default=None,
                       metavar="SECONDS",
                       help="how long a gated tool waits for a "
                            "permission_reply before being denied "
                            f"(default {_PT})")
    # The RPC protocol carries these per-request; the defaults below only
    # mirror `chat`: never write, never review, no terminal steering.
    p_rpc.set_defaults(apply=False, review=False, no_review=True,
                       refine=False, plan=False, no_plan=False,
                       resume=False, jsonl=False, rpc=True)

    p_cal = sub.add_parser(
        "calibrate", help="measure the configured model (tool choice, edit "
                          "format, honesty, long context, ...) and save a "
                          "model card that tunes the defaults")
    p_cal.add_argument("--recalibrate", action="store_true",
                       help="measure again even if a fresh card exists")
    p_cal.add_argument("--quick", action="store_true",
                       help="skip the long-context and capability probes")
    p_cal.add_argument("--repeats", type=int, default=None,
                       help="run each probe this many times (default: 3, or 1 "
                            "with --quick) and record the mean and spread")
    for flag in ("--url", "--model", "--config", "--profile"):
        p_cal.add_argument(flag)
    p_cal.add_argument("--root", default=None)
    p_cal.add_argument("paths", nargs="*", default=["."], help=argparse.SUPPRESS)

    p_models = sub.add_parser(
        "models", help="discover model servers, write profiles, calibrate")
    p_models.add_argument("action", choices=["scan"],
                          help="'scan' finds local servers and sets up profiles")
    p_models.add_argument("--url", action="append", default=[], metavar="URL",
                          help="server base url to probe; repeatable")
    p_models.add_argument("--api-key-env", metavar="VAR",
                          help="environment variable holding the API key "
                               "(hosted servers only)")
    p_models.add_argument("--match", metavar="PATTERN",
                          help="only consider models whose id matches (fnmatch)")
    grp = p_models.add_mutually_exclusive_group()
    grp.add_argument("--all", action="store_true",
                     help="include models the server has not loaded")
    grp.add_argument("--loaded", action="store_true",
                     help="only models the server reports as loaded")
    p_models.add_argument("--calibrate", choices=["none", "quick", "full"],
                          default=None,
                          help="calibrate selected models (local default: quick; "
                               "hosted: only when given)")
    p_models.add_argument("-y", "--yes", action="store_true",
                          help="non-interactive: select all matching models")
    p_models.add_argument("--vram-gb", type=int, default=None, metavar="GB",
                          help="GPU VRAM of the server machine (asked if omitted)")
    p_models.add_argument("--config", metavar="FILE",
                          help="write profiles to this config file "
                               "(default: the user config)")

    p_profiles = sub.add_parser(
        "profiles", help="list model profiles and their measured scores")
    p_profiles.add_argument("--recommend", action="store_true",
                            help="suggest a profile per role, with reasons")
    p_profiles.add_argument("--apply-roles", dest="apply_roles",
                            action="store_true",
                            help="write the recommended roles = {...} (asks first)")
    p_profiles.add_argument("--json", action="store_true",
                            help="print the profiles as JSON")
    p_profiles.add_argument("-y", "--yes", action="store_true",
                            help="with --apply-roles: do not ask for confirmation")
    p_profiles.add_argument("--config", metavar="FILE")
    p_profiles.add_argument("--profile", metavar="NAME")

    p_reset = sub.add_parser(
        "reset", help="wipe this project's Pedacito state (index, plans, "
                      "caches, sessions); code is untouched")
    p_reset.add_argument("paths", nargs="*", default=["."],
                         help=argparse.SUPPRESS)
    p_reset.add_argument("--root", default=None)
    p_reset.add_argument("--dry-run", action="store_true",
                         help="show what would be removed, then exit")
    p_reset.add_argument("-y", "--yes", action="store_true",
                         help="do not ask for confirmation")
    p_reset.add_argument("--keep-memory", action="store_true",
                         help="keep .pedacito/memory.md (remembered facts)")
    p_reset.add_argument("--keep-config", "--keep-authored",
                         dest="keep_config", action="store_true",
                         help="keep authored .pedacito/rules, skills, prompts "
                              "and instruction files")
    p_reset.add_argument("--force", action="store_true",
                         help="also remove git worktrees under "
                              ".pedacito/worktrees (discards uncommitted work)")
    p_reset.add_argument("--reindex", action="store_true",
                         help="rebuild the index after the wipe")
    p_reset.add_argument("--no-summaries", action="store_true",
                         help="with --reindex: build the static index only "
                              "(no model calls)")
    p_reset.add_argument("--url", help="server base url (with --reindex)")
    p_reset.add_argument("--model", help="model id/name (with --reindex)")
    p_reset.add_argument("--config", metavar="FILE")
    p_reset.add_argument("--profile", metavar="NAME")

    p_prune = sub.add_parser(
        "prune", help="delete old sessions/backups/events, trim log.txt")
    p_prune.add_argument("paths", nargs="*", default=["."])
    p_prune.add_argument("--keep-sessions", type=int, default=None, metavar="N")
    p_prune.add_argument("--keep-backups", type=int, default=None, metavar="N")
    p_prune.add_argument("--dry-run", action="store_true")
    p_prune.add_argument("--root", default=None)
    p_prune.add_argument("--config", metavar="FILE")
    p_prune.add_argument("--profile", metavar="NAME")

    p_serve = sub.add_parser(
        "serve", help="start a background dev server (registered under "
                      ".pedacito/procs/)")
    p_serve.add_argument("command", nargs=argparse.REMAINDER,
                         help="the command to run; use `-- COMMAND...`")
    p_serve.add_argument("--name", default="",
                         help="registry name (default: derived from the command)")
    p_serve.add_argument("--wait", type=float, default=None,
                         help="seconds to wait for a printed URL")
    p_serve.add_argument("--root", default=None)
    p_serve.add_argument("--config", metavar="FILE")
    p_serve.add_argument("--profile", metavar="NAME")

    p_procs = sub.add_parser(
        "procs", help="list, show logs from, or stop background servers")
    p_procs.add_argument("action", nargs="?", default="list",
                         choices=["list", "logs", "stop"])
    p_procs.add_argument("name", nargs="?", default="")
    p_procs.add_argument("--tail", type=int, default=4000,
                         help="with logs: last N characters (default 4000)")
    p_procs.add_argument("--root", default=None)
    p_procs.add_argument("--config", metavar="FILE")
    p_procs.add_argument("--profile", metavar="NAME")

    p_lsp = sub.add_parser(
        "lsp", help="exact symbols/diagnostics through an optional language "
                    "server (allow_lsp)")
    p_lsp.add_argument("--file", required=True, help="project-relative file")
    p_lsp.add_argument("--op", default="definition",
                       choices=["definition", "references", "hover",
                                "diagnostics"])
    p_lsp.add_argument("--line", type=int, default=1)
    p_lsp.add_argument("--column", type=int, default=1)
    p_lsp.add_argument("--root", default=None)
    p_lsp.add_argument("--config", metavar="FILE")
    p_lsp.add_argument("--profile", metavar="NAME")

    p_mcp = sub.add_parser(
        "mcp", help="inspect named MCP servers ([mcp_servers.<name>]): "
                    "list the tools they expose and their trust flag")
    p_mcp.add_argument("action", nargs="?", default="list", choices=["list"])
    p_mcp.add_argument("--root", default=None)
    p_mcp.add_argument("--config", metavar="FILE")
    p_mcp.add_argument("--profile", metavar="NAME")

    p_mem = sub.add_parser(
        "memory", help="list, add or forget the facts Pedacito remembers")
    p_mem.add_argument("paths", nargs="*", default=["."])
    p_mem.add_argument("--add", metavar="TEXT")
    p_mem.add_argument("--forget", metavar="TEXT")
    p_mem.add_argument("--global", dest="use_global", action="store_true",
                       help="use the user-level memory instead of the project's")
    p_mem.add_argument("--root", default=None)
    p_mem.add_argument("--config", metavar="FILE")
    p_mem.add_argument("--profile", metavar="NAME")

    p_trust = sub.add_parser(
        "trust", help="approve this project's .pedacito.toml to run hooks, "
                      "MCP servers and verify commands")
    p_trust.add_argument("paths", nargs="*", default=["."])
    p_trust.add_argument("--yes", action="store_true")
    p_trust.add_argument("--root", default=None)
    p_trust.add_argument("--config", metavar="FILE")
    p_trust.add_argument("--profile", metavar="NAME")

    p_tests = sub.add_parser(
        "tests", help="show the detected test command and the tests a change "
                      "to FILES can break")
    p_tests.add_argument("paths", nargs="*", default=["."])
    p_tests.add_argument("--files", nargs="*", default=[], metavar="FILE")
    p_tests.add_argument("--root", default=None)
    p_tests.add_argument("--config", metavar="FILE")
    p_tests.add_argument("--profile", metavar="NAME")

    p_web = sub.add_parser(
        "webcheck", help="static contract checks for a web project: DOM id "
                         "references, ESM syntax in classic scripts, duplicate "
                         "globals, undeclared imports, undefined members")
    p_web.add_argument("paths", nargs="*", default=["."])
    p_web.add_argument("--files", nargs="*", default=[], metavar="FILE")
    p_web.add_argument("--root", default=None)
    p_web.add_argument("--config", metavar="FILE")
    p_web.add_argument("--profile", metavar="NAME")

    p_skills = sub.add_parser("skills", help="list the available skills")
    p_skills.add_argument("paths", nargs="*", default=["."])
    p_skills.add_argument("--root", default=None)
    p_skills.add_argument("--config", metavar="FILE")
    p_skills.add_argument("--profile", metavar="NAME")

    p_rename = sub.add_parser(
        "rename", help="deterministic multi-file symbol rename via the AST "
                       "and the tokenizer (no LLM calls; strings and comments "
                       "are never touched)")
    p_rename.add_argument("old", help="the existing identifier")
    p_rename.add_argument("new", help="the new identifier")
    p_rename.add_argument("--file", default="", metavar="PATH",
                          help="scope: one file and the files that import it "
                               "(default: all indexed project source)")
    p_rename.add_argument("--apply", action="store_true",
                          help="write the rename (default: dry-run preview "
                               "with unified diffs)")
    p_rename.add_argument("--root", default=None)

    p_eval = sub.add_parser(
        "eval", help="score the harness on built-in fixture repos (locate "
                     "precision, edit success, tokens, cache hits)")
    p_eval.add_argument("--task", action="append", default=[], metavar="ID",
                        help="run only this task id; repeatable")
    p_eval.add_argument("--out", metavar="FILE", help="write the JSON report here")
    p_eval.add_argument("--baseline", metavar="FILE",
                        help="A/B gate: compare this run's metrics with an "
                             "earlier eval --out report; exit nonzero on a "
                             "regression")
    p_eval.add_argument("--max-token-growth", type=float, default=0.20,
                        metavar="FRAC",
                        help="allow this much token growth vs baseline "
                             "(default 0.20)")
    p_eval.add_argument("--max-time-growth", type=float, default=0.50,
                        metavar="FRAC",
                        help="allow this much wall-time growth vs baseline "
                             "(default 0.50)")
    p_eval.add_argument("--repeats", type=int, default=1, metavar="N",
                        help="run the whole task set N times, each repeat with "
                             "its own sampler.seed, and report per-task pass "
                             "counts and mean ± spread (default 1)")
    p_eval.add_argument("--seed", type=int, default=None, metavar="S",
                        help="base sampler.seed: repeat i uses S+i (default: "
                             "your sampler seed, else 0, when --repeats > 1; "
                             "unset for a single run)")
    p_eval.add_argument("--list", action="store_true", help="list tasks and exit")
    for flag in ("--url", "--model", "--config", "--profile"):
        p_eval.add_argument(flag)
    p_eval.add_argument("--root", default=None)
    p_eval.add_argument("paths", nargs="*", default=["."], help=argparse.SUPPRESS)

    p_gui = sub.add_parser("gui", help="open the browser control panel")
    p_gui.add_argument("project", nargs="?", default=None,
                       help="project folder to open at startup")
    p_gui.add_argument("--profile", metavar="NAME",
                       help="model profile for that project")
    p_gui.add_argument("--port", type=int, default=0)
    p_gui.add_argument("--no-open", dest="open_browser", action="store_false")

    return ap


def _main(argv=None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)

    # `config --setup` repairs a broken config, so it runs before _config().
    if args.cmd == "config" and getattr(args, "setup", False):
        try:
            interactive_setup(Path(args.path) if args.path else None,
                              force=args.force)
        except ConfigError as e:
            print(f"Setup failed: {e}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\nSetup cancelled.", file=sys.stderr)
            return 130
        return 0

    # The GUI loads config per project (and can create it), so it must start
    # even when no valid config exists yet.
    if args.cmd == "gui":
        from . import gui as gui_mod
        gui_mod.run(port=args.port, open_browser=args.open_browser,
                    project=args.project, profile=args.profile)
        return 0

    if args.cmd == "worktree":
        return _cmd_worktree(args)
    if args.cmd == "diagnose":
        return _cmd_diagnose(args)
    # `models scan` discovers servers and writes profiles, so it must run
    # before _config() (a first-run machine may have no valid config yet).
    if args.cmd == "models":
        return _cmd_models_scan(args)
    if args.cmd == "profiles":
        try:
            return _cmd_profiles(args, _light_config(args))
        except ConfigError as e:
            print(f"Config error: {e}", file=sys.stderr)
            return 2
    light = {"prune": _cmd_prune, "reset": _cmd_reset, "memory": _cmd_memory,
             "trust": _cmd_trust, "tests": _cmd_tests,
             "webcheck": _cmd_webcheck, "skills": _cmd_skills,
             "rename": _cmd_rename,
             "serve": _cmd_serve, "procs": _cmd_procs, "lsp": _cmd_lsp,
             "mcp": _cmd_mcp}
    if args.cmd in light:
        try:
            return light[args.cmd](args)
        except ConfigError as e:
            print(f"Config error: {e}", file=sys.stderr)
            return 2

    # `how` needs a model connection but no project, index or vram budget.
    if args.cmd == "how":
        from .how import HowError, run_cli as run_how
        try:
            return run_how(args)
        except (ConfigError, HowError) as e:
            print(f"How error: {e}", file=sys.stderr)
            return 2
        except LLMError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\nCancelled.", file=sys.stderr)
            return 130

    # `explain` never builds an index; check before touching config or server.
    if args.cmd == "explain":
        root = _root(args)
        if index_mod.Index.load(root) is None:
            print(
                f"\nNo index found for {root}.\n"
                "`pedacito explain` answers questions about an already-indexed\n"
                "project; it never builds an index itself.\n\n"
                "To use it:\n"
                "  1. navigate to a project you have indexed, or index this one\n"
                f"     with:  pedacito index {root}\n"
                f"  2. then run: pedacito explain {root} -q \"your question\"\n",
                file=sys.stderr)
            return 0

    try:
        cfg = _config(args)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    try:
        if args.cmd == "config":
            return _cmd_config(args, cfg)

        if args.cmd == "calibrate":
            return _cmd_calibrate(args, cfg)

        if args.cmd == "eval":
            from . import evalrun
            return evalrun.run_cli(args, cfg)

        if args.cmd == "map":
            idx = _get_index(args, cfg, None, rebuild=False, summarise=False)
            if args.tier == "flat":
                text = idx.render()
            elif args.tier == "cards":
                text = idx.render_files(line_ranges=cfg.file_card_line_ranges)
            elif args.tier == "dirs":
                text = idx.render_dirs()
            elif args.tier == "ranked":
                budget = max(400, int(cfg.flat_map_token_budget))
                text = idx.render_ranked(cfg, args.task, budget)
                print(f"[tier: ranked; budget {budget}]", file=sys.stderr)
            else:
                text, tier = idx.map_for_prompt(cfg, task=args.task)
                print(f"[tier: {tier}]", file=sys.stderr)
            print(text)
            print(f"\n[{estimate_tokens(text):,} tokens in every prompt]", file=sys.stderr)
            return 0

        if args.cmd == "restore":
            root = _root(args)
            snaps = workspace.list_backups(root)
            if not snaps:
                print(f"No backups under {root}/.pedacito/backups/", file=sys.stderr)
                return 1
            if args.list:
                for sdir in snaps:
                    files = [f for f in sdir.rglob("*") if f.is_file()]
                    print(f"  {sdir.name}   {len(files)} file(s): "
                          + ", ".join(f.relative_to(sdir).as_posix() for f in files[:4]))
                return 0
            chosen = next((s2 for s2 in snaps if s2.name == args.snapshot), None) \
                if args.snapshot else snaps[0]
            if chosen is None:
                print(f"No snapshot named {args.snapshot}. "
                      f"Try `pedacito restore --list`.", file=sys.stderr)
                return 1
            files = workspace.restore(root, chosen, dry_run=args.dry_run)
            verb = "would restore" if args.dry_run else "restored"
            for f in files:
                print(f"  {verb} {f}", file=sys.stderr)
            print(f"{verb} {len(files)} file(s) from {chosen.name}", file=sys.stderr)
            if getattr(args, "delete_created", False):
                gone = workspace.delete_created(root, chosen, dry_run=args.dry_run)
                dverb = "would delete" if args.dry_run else "deleted"
                for f in gone:
                    print(f"  {dverb} {f} (created by that run)", file=sys.stderr)
                print(f"{dverb} {len(gone)} created file(s)", file=sys.stderr)
            else:
                made = workspace.created_files(root, chosen)
                if made:
                    print(f"  that run also created {len(made)} file(s) "
                          f"({', '.join(made[:4])}); add --delete-created to "
                          "remove them", file=sys.stderr)
            return 0

        if args.cmd == "files":
            root = _root(args)
            # `files` is the documented first command; bootstrap the workspace
            # so there is a .pedacitoignore to edit.
            workspace.ensure(root, cfg.manage_gitignore, cfg.create_ignore_file)
            m = _matcher(args, root)
            exts = set(CODE_EXTS)
            if cfg.include_references:
                exts |= set(REFERENCE_EXTS)
            exts |= {e if e.startswith(".") else "." + e for e in cfg.extra_extensions}
            chosen = {f.resolve() for f in select(args.paths, root, m, exts)}
            code = ref = skip = 0
            pruned: dict[str, int] = {}
            for p in args.paths:
                pp = Path(p)
                for f in sorted(pp.rglob("*")) if pp.is_dir() else [pp]:
                    if not f.is_file():
                        continue
                    try:
                        rel = f.resolve().relative_to(root).as_posix()
                    except ValueError:
                        rel = f.name
                    if f.resolve() in chosen:
                        kind = "code " if f.suffix.lower() in CODE_EXTS else "ref  "
                        code, ref = code + (kind == "code "), ref + (kind == "ref  ")
                        print(f"  {kind} {rel}")
                        continue
                    skip += 1
                    # Collapse ignored directories to one line.
                    parts = rel.split("/")
                    top = next((("/".join(parts[: i + 1]))
                                for i in range(len(parts) - 1)
                                if m.excluded("/".join(parts[: i + 1]) + "/x")), None)
                    if top:
                        pruned[top] = pruned.get(top, 0) + 1
                        continue
                    why = ("ignored" if m.excluded(rel) else
                           f"extension {f.suffix or '(none)'} not indexed")
                    print(f"  skip  {rel}   [{why}]", file=sys.stderr)
            for d, n in sorted(pruned.items()):
                print(f"  skip  {d}/   [ignored directory, {n} file(s)]", file=sys.stderr)
            print(f"\n{code} code + {ref} reference indexed, {skip} skipped",
                  file=sys.stderr)
            return 0

        if args.cmd == "cost":
            idx = _get_index(args, cfg, None, rebuild=True, summarise=False)
            plan = index_mod.summary_plan(idx, cfg)
            calls = plan["unit_calls"] + plan["file_calls"]
            mins = calls * 80 / args.tps / 60
            flat_t = estimate_tokens(idx.render())
            cards_t = estimate_tokens(idx.render_files(line_ranges=cfg.file_card_line_ranges))
            dirs_t = estimate_tokens(idx.render_dirs())
            print(f"{plan['units']} units across {len(idx.files)} files "
                  f"({len(idx.code_files)} code, {len(idx.reference_files)} reference)")
            print(f"  eligible for a summary : {plan['eligible']}")
            print(f"  skipped, has docstring : {plan['skipped_documented']}")
            print(f"  skipped, is a method   : {plan['skipped_methods']}")
            print(f"  reference (never summ.): {plan['skipped_references']}")
            print(f"  LLM calls              : {plan['unit_calls']} units + "
                  f"{plan['file_calls']} files = {calls}")
            print(f"  first index @ {args.tps:.0f} tok/s : ~{mins:.0f} min (cached after)")
            print(f"\nmap size, flat  : {flat_t:,} tokens")
            print(f"map size, cards : {cards_t:,} tokens"
                  f"   <- used above {cfg.flat_map_token_budget:,}")
            print(f"map size, dirs  : {dirs_t:,} tokens"
                  f"   <- used when cards exceed "
                  f"{cfg.flat_map_token_budget * 2:,}")
            return 0

        if getattr(args, "replay", None):
            from . import replay as replay_mod
            try:
                client = replay_mod.Replayer(args.replay)
            except OSError as e:
                print(f"Cannot read the recording: {e}", file=sys.stderr)
                return 2
            client.cfg = cfg
            print(f"[replaying {args.replay}: no model server is used]",
                  file=sys.stderr)
        else:
            client = LLMClient(cfg)
            client.check()
            if getattr(args, "record", False):
                from . import replay as replay_mod
                rpath = replay_mod.record_path(_root(args))
                client = replay_mod.Recorder(client, rpath)
                print(f"[recording model responses to {rpath}]", file=sys.stderr)

        # `new` runs before indexing: the project directory usually does not
        # exist yet, and the skeleton gets its own in-memory index.
        if args.cmd == "new":
            from . import scaffold
            root = Path(args.project).expanduser().resolve()
            workspace.log_event(root, "command: " + " ".join(sys.argv))
            spec = scaffold.load_spec(
                prompt=args.task, spec_file=args.spec,
                max_chars=cfg.max_spec_chars)
            if getattr(args, "refine", False):
                spec.text = _refine_task_text(cfg, client, root, spec.text,
                                              "new")
            code = scaffold.run_new(
                root, cfg, client, spec,
                apply=args.apply, review=args.review,
                plan_only=args.plan_only, force=args.force,
                verbose=True, complex=args.complex,
                verify=args.verify if getattr(args, "verify", None) is not None
                else cfg.auto_verify,
                repair_rounds=args.repair if getattr(args, "repair", None) is not None
                else cfg.auto_repair_rounds,
                max_tasks=args.max_tasks if getattr(args, "max_tasks", None) is not None
                else cfg.auto_max_tasks,
                max_minutes=args.max_minutes
                if getattr(args, "max_minutes", None) is not None
                else cfg.auto_max_minutes,
                keep_going=args.keep_going,
                design=bool(getattr(args, "design", False)),
                auto_install=bool(getattr(args, "auto_install", False)
                                  or cfg.auto_install_deps))
            s = client.stats
            print(f"\n[{s['calls']} calls, {s['completion_tokens']} tokens out, "
                  f"{s['seconds']:.0f}s]", file=sys.stderr)
            return code

        if args.cmd == "index":
            root = _root(args)
            workspace.ensure(root, cfg.manage_gitignore, cfg.create_ignore_file)
            idx = index_mod.build(args.paths, cfg,
                                  client=None if args.no_summaries
                                  else roles_mod.client_for(cfg, "summarizer",
                                                            client),
                                  root=root, summarise=not args.no_summaries,
                                  matcher=_matcher(args, root),
                                  resummarise=args.resummarise)
            print(f"Indexed {len(idx.chunks)} units across {len(idx.files)} files "
                  f"({len(idx.code_files)} code, {len(idx.reference_files)} reference) "
                  f"-> {index_mod.state_dir(root)}", file=sys.stderr)
            return 0

        idx = _get_index(args, cfg, client, rebuild=getattr(args, "reindex", False))
        # Load the model here with a clear message rather than stalling
        # silently inside the first gather step.
        client.warmup()
        if args.cmd in ("ask", "chat", "auto", "rpc") and cfg.refresh_summaries:
            index_mod.refresh_summaries(
                idx, roles_mod.client_for(cfg, "summarizer", client), cfg,
                limit=int(cfg.refresh_summaries))
        agent = Agent(idx, client, cfg)
        from . import steering
        if args.cmd in ("ask", "chat") and steering.Steering.available():
            cfg._interactive = True
            cfg._ask_user = steering.terminal_question
        elog = events_mod.get(_root(args), cfg)
        if getattr(args, "jsonl", False):
            elog.tee = sys.stdout
            sys.stdout = sys.stderr
        events_mod.attach(client, elog)
        elog.emit("session_start", cmd=args.cmd, model=cfg.model,
                  tier=cfg.active_tier, pack=prompts_pack(cfg),
                  fingerprint=prompts_fingerprint(cfg))
        hooks_mod.fire(cfg, _root(args), "session_start", cmd=args.cmd) \
            if hooks_mod.has_hooks(cfg, "session_start") else None

        root = _root(args)
        from . import checkpoint
        resume_context = ""
        if args.cmd == "ask" and getattr(args, "continue_", False):
            cp = checkpoint.latest(root, "ask")
            if cp is None:
                print("No ask checkpoint to continue. Run `pedacito ask` "
                      "once first, or omit --continue.", file=sys.stderr)
                return 2
            resume_context = checkpoint.render(cp)
        elif args.cmd == "chat" and getattr(args, "resume", False):
            cp = checkpoint.latest(root, "chat")
            if cp is None:
                print("No chat checkpoint to resume.", file=sys.stderr)
                return 2
            resume_context = checkpoint.render(cp)
            print("\n" + resume_context, file=sys.stderr)
        if args.cmd in ("ask", "chat", "auto") and getattr(args, "apply", False):
            try:
                workspace.acquire_lock(root)
            except workspace.ProjectBusy as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 2
            _LOCKED.append(root)
        # Slash commands rebind these, so share them through a dict.
        locals_ref = {"cfg": cfg, "client": client, "agent": agent, "idx": idx}

        if args.cmd == "explain":
            return _cmd_explain(args, client, agent)

        def run(task: str, apply=None, refine=None, plan=None):
            """Execute one task end to end through the shared inner loop.
            `apply`/`refine`/`plan` override the command-line defaults (the
            `rpc` requests carry them per turn); None means "use the args"."""
            nonlocal idx, agent, cfg, client, resume_context
            cfg, client, agent = locals_ref["cfg"], locals_ref["client"], locals_ref["agent"]
            idx = locals_ref["idx"]
            if refine:
                task = _refine_task_text(cfg, client, root, task, "ask")
            if resume_context:
                task = (resume_context + "\n\nNEW TASK\n" + task)
                resume_context = ""
            locals_ref["last_result"] = None
            want_apply = bool(args.apply) if apply is None else bool(apply)
            from . import plan as plan_mod
            if plan is None:
                want_plan = getattr(args, "plan", False) or (
                    not getattr(args, "no_plan", False)
                    and plan_mod.should_plan(task, idx, cfg))
            else:
                want_plan = bool(plan)
            if want_plan and not _wants_review(args, cfg):
                code = plan_mod.run_planned(
                    root, cfg, client, idx, task, apply=want_apply,
                    confirm=_confirm_plan if getattr(args, "review_plan", False)
                    else None, strict=bool(getattr(args, "strict", False)))
                fresh = index_mod.Index.load(root)
                if fresh is not None:
                    idx = fresh
                    agent.index = idx
                    locals_ref["idx"] = idx
                if isinstance(code, int):
                    # the planned path reports an exit code; give `rpc` a
                    # result-shaped object so its summary frame is uniform
                    from types import SimpleNamespace
                    tr = SimpleNamespace(reply=None, applied=False, results=[],
                                         steps=[], stats={}, tests_ok=None,
                                         ok=(code == 0), idx=idx)
                    locals_ref["last_result"] = tr
                    return tr
                return code
            tr = loop.run_task(root, cfg, client, agent, idx, task,
                               apply=want_apply, review=_wants_review(args, cfg))
            locals_ref["last_result"] = tr
            idx = tr.idx
            agent.index = idx
            locals_ref["idx"] = idx
            return tr

        def save_checkpoint(task: str):
            """Persist what the last task actually said, for --continue/--resume."""
            tr = locals_ref.get("last_result")
            if tr is None:
                return
            msgs = getattr(agent, "_messages", None) or []
            checkpoint.save(
                root, args.cmd, task, msgs,
                getattr(tr, "reply", "") if tr is not None else "", cfg,
                extra={"steps": list(getattr(tr, "steps", []) or [])[:60],
                       "applied": bool(getattr(tr, "applied", False))})

        if args.cmd == "rpc" or getattr(args, "rpc", False):
            # machine-client transport instead of the terminal REPL
            return _cmd_rpc(args, cfg, client, idx, root, locals_ref,
                            run, save_checkpoint, elog)

        if args.cmd == "ask":
            ask_task = args.task
            if getattr(args, "refine", False):
                ask_task = _refine_task_text(cfg, client, root, ask_task,
                                             "ask")
            try:
                run(ask_task)
                try:
                    save_checkpoint(ask_task)
                except (OSError, ValueError):
                    pass                 # checkpointing must not lose the task
            except KeyboardInterrupt:
                print("\nCancelled. Nothing further was applied; "
                      "`pedacito restore` undoes anything already written.",
                      file=sys.stderr)
                return 130
            print("\n" + _usage_line(client), file=sys.stderr)
            from . import replay as replay_mod
            replay_mod.report(client)
            strict = (_strict_replay_error(args, client)
                      or _strict_task_error(args,
                                            locals_ref.get("last_result")))
            if strict:
                print(f"ERROR: {strict}", file=sys.stderr)
                elog.emit("exit", code=1)
                return 1
            elog.emit("exit", code=0)
            return 0

        if args.cmd == "auto":
            code = _cmd_auto(args, cfg, client, idx, root)
            strict = _strict_replay_error(args, client)
            if strict:
                print(f"ERROR: {strict}", file=sys.stderr)
                return 1
            return code

        # chat: interactive loop with the index kept warm between tasks
        def banner():
            """One-line status shown at the start of a chat session."""
            bits = [f"model {cfg.model}"]
            if cfg.active_profile:
                bits.insert(0, f"profile {cfg.active_profile}")
            bits.append("writes ON" if args.apply else "dry run")
            return " | ".join(bits)

        print(f"Index is warm. {banner()}", file=sys.stderr)
        print("Type a task, or /help for commands.", file=sys.stderr)

        while True:
            try:
                task = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
            if not task:
                continue
            if task.lower() in {"quit", "exit", "q"}:
                break
            if task.startswith("/"):
                if _chat_command(task, args, cfg, locals_ref) is False:
                    break
                cfg, client, agent = locals_ref["cfg"], locals_ref["client"], locals_ref["agent"]
                continue
            try:
                if steering.Steering.available() and cfg.allow_steering:
                    with steering.Steering() as steer:
                        locals_ref["cfg"]._steer = steer.queue
                        try:
                            run(task)
                        finally:
                            locals_ref["cfg"].__dict__.pop("_steer", None)
                else:
                    run(task)
                try:
                    save_checkpoint(task)
                except (OSError, ValueError):
                    pass                 # checkpointing must not end the session
            except KeyboardInterrupt:
                # Ctrl-C stops the current task, never the whole session.
                try:
                    locals_ref["client"].cancel()
                except Exception:
                    pass
                print("\n  [cancelled; the session is still open -- edits "
                      "already written can be undone with `pedacito restore`]",
                      file=sys.stderr)
            except LLMError as e:
                print(f"\nERROR: {e}", file=sys.stderr)
        strict = _strict_replay_error(args, client)
        if strict:
            print(f"ERROR: {strict}", file=sys.stderr)
            return 1
        return 0

    except LLMError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        try:
            workspace.log_event(_root(args), f"ERROR: {e}", "ERROR")
        except Exception:
            pass                       # no project root yet; nothing to log to
        return 1


if __name__ == "__main__":
    raise SystemExit(main())