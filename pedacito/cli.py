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
    pedacito new  my-app -t "todo CLI"       # start a project from scratch
    pedacito auto . -t "add retries" --apply # autonomous outer loop over tasks

Run as `pedacito <command> ...` (after `pip install -e .`) or
`python -m pedacito <command> ...`. `main()` is the single entry point invoked
by both `__main__.py` files. `pdc` is installed as a shorthand alias for
`pedacito` and accepts the exact same commands, e.g. `pdc ask src/ -t "..."`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import index as index_mod
from . import loop
from . import workspace
from .agent import Agent
from .config import (Config, ConfigError, REFERENCE_TEMPLATE,
                     USER_CONFIG, auto_fit, find_project_config,
                     interactive_setup, load as load_config, write_template)
from .llm import LMStudio, LLMError, estimate_tokens
from .select import CODE_EXTS, REFERENCE_EXTS, Matcher, is_venv_dir, select


# --------------------------------------------------------------------------
# Shared argument setup
# --------------------------------------------------------------------------
def _add_common(p):
    """Register the argument flags shared by every subcommand that operates
    on a project: paths, file selection overrides, connection overrides,
    and config selection."""
    p.add_argument("paths", nargs="+", help="files or directories to index")
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


# --------------------------------------------------------------------------
# Argument -> runtime object helpers
# --------------------------------------------------------------------------
def _root(args) -> Path:
    """Resolve the effective project root for a parsed args namespace:
    --root if given, otherwise inferred from the paths."""
    if getattr(args, "root", None):
        return Path(args.root).resolve()
    return index_mod.infer_root(getattr(args, "paths", []) or ["."])


def _matcher(args, root: Path) -> Matcher:
    """Build the file-selection Matcher for a parsed args namespace:
    .pedacitoignore/.gitignore defaults plus any --exclude/--include flags."""
    names = (".pedacitoignore",) if getattr(args, "no_gitignore", False) \
        else (".pedacitoignore", ".gitignore")
    m = Matcher.from_files(root, names=names)
    m.extend(getattr(args, "exclude", []))
    # Includes are appended last so they win: last match wins.
    m.extend("!" + p for p in getattr(args, "include", []))
    return m


def _config(args) -> Config:
    """Build the effective Config for a parsed args namespace: load from
    files/profile, then apply any command-line overrides on top."""
    # The config subcommand must be able to inspect (and repair) a file that
    # predates the mandatory vram_gb setting; every real run requires it for
    # local servers.
    cfg = load_config(root=_root(args), explicit=getattr(args, "config", None),
                      profile=getattr(args, "profile", None),
                      require_hardware=(getattr(args, "cmd", "") != "config"))
    if getattr(args, "no_refs", False):
        cfg.include_references = False
    if getattr(args, "ext", None):
        cfg.extra_extensions = tuple(args.ext)
    if getattr(args, "url", None):
        cfg.base_url = args.url
    if getattr(args, "model", None):
        cfg.model = args.model
    if getattr(args, "steps", None) is not None:
        cfg.max_steps = args.steps
        cfg.sources["max_steps"] = "--steps"
    for flag, key in (("url", "base_url"), ("model", "model")):
        if getattr(args, flag, None):
            cfg.sources[key] = f"--{flag}"
    # Fit every token budget to the server's real context window before any
    # request goes out. Without this, built-in/hosted-tier budgets (3k-16k
    # answers, 24k-100k gathered source) sail past LM Studio's stock 8192 or
    # Ollama's default 4096 and every call gets truncated or rejected --
    # "it just kept hitting limits" even with a correct base_url/model.
    # `pedacito config` itself skips fitting so it can report true sources.
    if not getattr(args, "cmd", "") == "config":
        auto_fit(cfg, verbose=True)
    return cfg


# --------------------------------------------------------------------------
# Index loading and freshness
# --------------------------------------------------------------------------
def _refresh(idx, args, cfg, root: Path, verbose: bool = True):
    """Bring a loaded index back in line with what is on disk.

    Line ranges are only meaningful against the bytes they were built from, so
    a file you edited yourself between sessions would send the model to the
    wrong lines. Re-chunking is fast and local; summaries survive by content
    hash, so only the parts you actually changed lose theirs.
    """
    changed, missing = idx.stale()
    # stale() may have refreshed recorded [mtime, size] entries for files it
    # hashed and found unchanged (a touch, a checkout); persisting those lets
    # the NEXT run skip the hashes too.
    stats_dirty = idx.stats_refreshed
    for rel in missing:
        idx.forget(rel)
    for rel in changed:
        idx = index_mod.reindex_file(idx, rel, cfg, save=False)

    # Files added since the last index will not be in it at all.
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

    # Splice new files into the existing index one at a time rather than
    # rebuilding from scratch: a rebuild discards every cached summary (see
    # the cache handling in index.build), and re-chunking the entire project
    # to pick up one added file is a needless amount of work besides.
    for rel in added:
        idx = index_mod.reindex_file(idx, rel, cfg, save=False)
    if verbose and added:
        print("  new files need summaries: run `pedacito index` when convenient",
              file=sys.stderr)

    if changed or missing or added or stats_dirty:
        idx.save()
    return idx


def _get_index(args, cfg, client, rebuild: bool, summarise: bool = True):
    """Load the project's index (refreshing it for staleness) or build it
    from scratch if none exists yet or `rebuild` is set."""
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


# --------------------------------------------------------------------------
# `pedacito config`
# --------------------------------------------------------------------------
SECRETISH = {"api_key"}


def _cmd_config(args, cfg) -> int:
    """Implement `pedacito config`: with --init, write a starter config file;
    otherwise show every resolved setting and which file/profile set it,
    and with --check, verify the LM Studio connection and model."""
    if getattr(args, "reference", False):
        # The full annotated template: every knob, documented. Kept separate
        # from the short starter file that --init writes so first-run users
        # are not handed a 300-line wall of comments.
        target = Path(args.path) if args.path else None
        if target is not None:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(REFERENCE_TEMPLATE)
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
            print(f"{e}", file=sys.stderr)
            return 2
        print(f"Wrote {path}", file=sys.stderr)
        print("Mandatory settings: base_url, model and vram_gb (top of the\n"
              "file). Everything else has a working default -- see\n"
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
        print(f"\nProfiles defined: {', '.join(sorted(cfg.profiles))}")
        print(f"  use with: pedacito ask PROJ -t \"...\" --profile "
              f"{sorted(cfg.profiles)[0]}")
        from .config import _apply_preset
        for name in sorted(cfg.profiles):
            _, tier = _apply_preset(name, cfg.profiles[name], "config file")
            if tier:
                print(f"  {name:<20} tier '{tier}' preset applied "
                      "(token limits)")
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

    if not args.check:
        print("\nRun `pedacito config --check` to test the connection.", file=sys.stderr)
        return 0

    print(f"\nConnecting to {cfg.base_url} ...")
    client = LMStudio(cfg)
    try:
        loaded = client.check(warn=False)
    except LLMError as e:
        print(f"{e}", file=sys.stderr)
        return 1
    print(f"  reachable. Models available: {loaded}")
    if cfg.model not in loaded.split(", "):
        print(f"  configured model '{cfg.model}': NOT IN THE LIST above.",
              file=sys.stderr)
        print("  Copy the id exactly as shown.", file=sys.stderr)
        return 1
    # Listing a model does not prove it can load. Generate one token and find
    # out now, rather than 35 chunks into an index.
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
    print("  generation works. You're good.")
    return 0


# --------------------------------------------------------------------------
# `pedacito chat` slash commands
# --------------------------------------------------------------------------
CHAT_HELP = """\
  /profile NAME   switch model profile (loads it, waits if needed)
  /model ID       switch to a model id directly
  /apply on|off   enable or disable writing edits
  /steps N        how many lookups the model gets per task
  /status         current profile, model, and write mode
  /profiles       list the profiles in your config
  /quit           leave"""


def _chat_command(line: str, args, cfg, ref) -> bool:
    """Handle a /command typed in `pedacito chat`. Returns False to end the
    session, True to continue.

    Switching profiles mid-session is the point of this: comparing two
    models on the same task is much easier when the index stays warm
    between them.
    """
    parts = line[1:].split(None, 1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("quit", "exit", "q"):
        return False

    if cmd in ("help", "h", "?"):
        print(CHAT_HELP, file=sys.stderr)
        return True

    if cmd == "profiles":
        from .config import _apply_preset
        if not cfg.profiles:
            print("  No profiles defined. Add [profiles.NAME] to your config.",
                  file=sys.stderr)
        for name, body in sorted(cfg.profiles.items()):
            mark = "*" if name == cfg.active_profile else " "
            _, tier = _apply_preset(name, body, "config")
            limits = f"  [tier {tier}]" if tier else ""
            print(f"  {mark} {name}: {body.get('model', '(inherits model)')}"
                  f"{limits}", file=sys.stderr)
        return True

    if cmd == "status":
        print(f"  profile : {cfg.active_profile or '(none)'}", file=sys.stderr)
        print(f"  model   : {cfg.model}", file=sys.stderr)
        print(f"  steps   : {cfg.max_steps}", file=sys.stderr)
        print(f"  writes  : {'ON' if args.apply else 'off (dry run)'}", file=sys.stderr)
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
        # Update the parsed-args namespace too, not just the live cfg: a
        # later `/profile` re-resolves config through `_config(args)`, which
        # applies args.steps on top. Without this, the original --steps value
        # (or None) would silently clobber the /steps override.
        args.steps = n
        ref["cfg"].max_steps = n
        ref["agent"].cfg = ref["cfg"]
        print(f"  max_steps = {n}", file=sys.stderr)
        return True

    if cmd in ("profile", "model"):
        if not arg:
            print(f"  Usage: /{cmd} NAME", file=sys.stderr)
            return True
        prev_profile, prev_model = args.profile, args.model
        if cmd == "profile":
            args.profile, args.model = arg, None
        else:
            args.model = arg
        try:
            new_cfg = _config(args)
        except ConfigError as e:
            args.profile, args.model = prev_profile, prev_model
            print(f"  {e}", file=sys.stderr)
            return True
        new_client = LMStudio(new_cfg)
        try:
            new_client.check(warn=False)
            new_client.warmup()
        except LLMError as e:
            args.profile, args.model = prev_profile, prev_model
            print(f"  Could not switch: {e}", file=sys.stderr)
            print(f"  Staying on {cfg.model}.", file=sys.stderr)
            return True
        ref["cfg"], ref["client"] = new_cfg, new_client
        ref["agent"] = Agent(ref["idx"], new_client, new_cfg)
        label = f"profile {new_cfg.active_profile}" + (f" [{new_cfg.active_tier} limits]" if new_cfg.active_tier else "") + ", " if new_cfg.active_profile else ""
        print(f"  now using {label}model {new_cfg.model}", file=sys.stderr)
        return True

    print(f"  Unknown command '/{cmd}'. Try /help.", file=sys.stderr)
    return True


# --------------------------------------------------------------------------
# Review-page integration
# --------------------------------------------------------------------------
def _wants_review(args, cfg) -> bool:
    """Decide whether this run should use the side-by-side review page:
    --review or config `review=true`, unless --no-review overrides it."""
    if getattr(args, "no_review", False):
        return False
    return bool(getattr(args, "review", False) or cfg.review)


# --------------------------------------------------------------------------
# `pedacito auto`: the autonomous outer loop
# --------------------------------------------------------------------------
def _cmd_auto(args, cfg, client, idx, root: Path) -> int:
    """Implement `pedacito auto`: generate a task list from a goal (or reuse
    an existing plan file), then run the Ralph loop until it is empty.

    Without --apply this is plan-only: generate/print the plan and stop, so a
    bare `auto` never writes code. With --apply it implements every unchecked
    task in a fresh context window, optionally verifying each with --verify."""
    from . import scaffold

    workspace.log_event(root, "command: " + " ".join(sys.argv))

    plan_path = Path(args.plan) if getattr(args, "plan", None) else \
        (root / cfg.auto_plan_file)
    if not plan_path.is_absolute():
        plan_path = (root / plan_path).resolve()

    if args.task or args.spec:
        spec = scaffold.load_spec(
            prompt=args.task, spec_file=args.spec,
            max_chars=getattr(cfg, "max_spec_chars", scaffold.MAX_SPEC_CHARS))
        # Reuse an existing plan to RESUME only while it still has open tasks;
        # once it is complete (or missing), a new -t/--spec goal starts a fresh
        # plan. Otherwise `auto -t "fix the bug"` on a finished project would
        # silently re-read the old all-checked checklist and report "all tasks
        # complete" without touching the new goal at all.
        regenerate = args.force or \
            loop.next_task(loop.read_plan(plan_path)) is None
        if regenerate:
            if plan_path.exists():
                print("  [existing plan has no open tasks; generating a new "
                      "one from the goal]", file=sys.stderr)
            titles = loop.generate_tasks(spec.text, client, cfg)
            loop.write_plan(plan_path, spec.title, titles)
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

    verify = args.verify if getattr(args, "verify", None) is not None \
        else cfg.auto_verify
    repair = args.repair if getattr(args, "repair", None) is not None \
        else cfg.auto_repair_rounds
    max_tasks = args.max_tasks if getattr(args, "max_tasks", None) is not None \
        else cfg.auto_max_tasks
    max_minutes = args.max_minutes \
        if getattr(args, "max_minutes", None) is not None \
        else getattr(cfg, "auto_max_minutes", 0)
    auto_install = bool(getattr(args, "auto_install", False)
                        or getattr(cfg, "auto_install_deps", False))

    code = loop.run_loop(root, cfg, client, idx, plan_path,
                         apply=args.apply, verify=verify, repair_rounds=repair,
                         max_tasks=max_tasks, max_minutes=max_minutes,
                         stop_on_fail=not args.keep_going,
                         auto_install=auto_install)

    # Automatic post-run verification: after the checklist is done, exercise
    # the project once (the explicit --verify and/or a smoke run of main.py)
    # and feed a failure back for repair before returning. This is what lets a
    # run correct itself instead of declaring success on broken code.
    smoke = bool(getattr(cfg, "auto_smoke", True))
    if args.apply and (verify or smoke):
        skip = {".pedacito", "__pycache__", ".venv", "venv",
                "node_modules", ".git"}

        def _under_venv(p: Path) -> bool:
            """True when `p` sits under a Python virtualenv (pyvenv.cfg marker),
            so a venv with a non-standard name inside the project is not fed to
            the smoke checker as if its site-packages were project code."""
            return any(is_venv_dir(parent) for parent in p.parents)

        py_rels = sorted(
            p.relative_to(root).as_posix() for p in root.rglob("*.py")
            if not (set(p.parts) & skip) and not _under_venv(p))
        v_ok, v_out = scaffold.verify_and_fix(
            client, cfg, idx, py_rels, root.name, None,
            explicit_cmd=verify,
            rounds=int(getattr(cfg, "auto_verify_rounds", 2) or 0),
            smoke=smoke,
            smoke_timeout=int(getattr(cfg, "auto_smoke_timeout", 20) or 20),
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


# --------------------------------------------------------------------------
# `pedacito explain`: questions about the indexed project
# --------------------------------------------------------------------------
def _cmd_explain(args, client, agent) -> int:
    """Implement `pedacito explain`: answer a question about the indexed
    project, in prose (no edits, no apply).

    The no-index case is gated in `main()` before a model connection is
    made, so by the time this runs the index exists and is warm."""
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

    s = client.stats
    print(f"\n[{s['calls']} calls, {s['completion_tokens']} tokens out, "
          f"{s['seconds']:.0f}s]", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    """Parse command-line arguments and dispatch to the requested
    subcommand. Returns the process exit code."""
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
    p_map.add_argument("--tier", choices=["auto", "cards", "dirs", "flat"], default="auto")

    p_conf = sub.add_parser("config", help="show, create, or test the config file")
    p_conf.add_argument("--init", action="store_true",
                        help=f"write a starter config to {USER_CONFIG}")
    p_conf.add_argument("--path", metavar="FILE", help="write --init here instead")
    p_conf.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_conf.add_argument("--check", action="store_true",
                        help="connect to LM Studio and verify the model is loaded")
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

    # Same positional shape as every other command: the project path first.
    p_rest = sub.add_parser("restore", help="undo an --apply run from its backup snapshot")
    p_rest.add_argument("paths", nargs="*", default=["."],
                        help="the project (default: current directory)")
    p_rest.add_argument("--snapshot", metavar="NAME",
                        help="which snapshot; default is the newest")
    p_rest.add_argument("--root", default=None)
    p_rest.add_argument("--list", action="store_true", help="list snapshots and exit")
    p_rest.add_argument("--dry-run", action="store_true")
    p_rest.add_argument("--config", metavar="FILE")

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
    p_auto.add_argument("--repair", type=int, default=None,
                        help="how many fix rounds a failing verification gets")
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

    p_gui = sub.add_parser("gui", help="open the browser control panel")
    p_gui.add_argument("--port", type=int, default=0)
    p_gui.add_argument("--no-open", dest="open_browser", action="store_false")

    args = ap.parse_args(argv)

    # `config --setup` is the recovery path for a broken or missing config,
    # so it must run even when loading the current one fails. Handled here,
    # before _config().
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

    # `how` is standalone help: it needs a model connection, but no project
    # path, index, or vram-derived token budget. Handle it before the shared
    # config/index path so it works with a bare-minimum config or direct flags.
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

    # `explain` is scoped to an already-indexed project. Check for an index
    # before touching the config or the model server, so a missing one gets a
    # helpful direction immediately -- no model connection or vram_gb needed.
    if args.cmd == "explain":
        root = _root(args)
        if index_mod.Index.load(root) is None:
            print(
                f"\nNo index found for {root}.\n"
                "`pedacito explain` answers questions about an already-indexed\n"
                "project; it never builds an index itself.\n\n"
                "To use it:\n"
                f"  1. navigate to a project you have indexed, or index this one\n"
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

        if args.cmd == "map":
            idx = _get_index(args, cfg, None, rebuild=False, summarise=False)
            if args.tier == "flat":
                text = idx.render()
            elif args.tier == "cards":
                text = idx.render_files(line_ranges=cfg.file_card_line_ranges)
            elif args.tier == "dirs":
                text = idx.render_dirs()
            else:
                text, tier = idx.map_for_prompt(cfg)
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
            return 0

        if args.cmd == "gui":
            from . import gui as gui_mod
            gui_mod.run(port=args.port, open_browser=args.open_browser)
            return 0

        if args.cmd == "files":
            root = _root(args)
            # `files` is the documented first command, so bootstrap here too:
            # the whole point of running it is to get a .pedacitoignore to edit.
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
                    # Collapse ignored directories to one line. Listing every
                    # file inside .git/ buries the answer you came for.
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

        client = LMStudio(cfg)
        client.check()

        # --- new: start a project from scratch (plan -> skeleton -> build) ---
        # Handled before indexing: the project directory usually does not exist
        # yet, and the skeleton gets its own in-memory index.
        if args.cmd == "new":
            from . import scaffold
            root = Path(args.project).expanduser().resolve()
            workspace.log_event(root, "command: " + " ".join(sys.argv))
            spec = scaffold.load_spec(
                prompt=args.task, spec_file=args.spec,
                max_chars=getattr(cfg, "max_spec_chars", scaffold.MAX_SPEC_CHARS))
            code = scaffold.run_new(
                root, cfg, client, spec,
                apply=args.apply, review=args.review,
                plan_only=args.plan_only, force=args.force,
                verbose=True, complex=args.complex,
                verify=args.verify if getattr(args, "verify", None) is not None
                else getattr(cfg, "auto_verify", ""),
                repair_rounds=args.repair if getattr(args, "repair", None) is not None
                else getattr(cfg, "auto_repair_rounds", 2),
                max_tasks=args.max_tasks if getattr(args, "max_tasks", None) is not None
                else getattr(cfg, "auto_max_tasks", 0),
                max_minutes=args.max_minutes
                if getattr(args, "max_minutes", None) is not None
                else getattr(cfg, "auto_max_minutes", 0),
                keep_going=args.keep_going,
                auto_install=bool(getattr(args, "auto_install", False)
                                  or getattr(cfg, "auto_install_deps", False)))
            s = client.stats
            print(f"\n[{s['calls']} calls, {s['completion_tokens']} tokens out, "
                  f"{s['seconds']:.0f}s]", file=sys.stderr)
            return code

        if args.cmd == "index":
            root = _root(args)
            workspace.ensure(root, cfg.manage_gitignore, cfg.create_ignore_file)
            idx = index_mod.build(args.paths, cfg,
                                  client=None if args.no_summaries else client,
                                  root=root, summarise=not args.no_summaries,
                                  matcher=_matcher(args, root),
                                  resummarise=args.resummarise)
            print(f"Indexed {len(idx.chunks)} units across {len(idx.files)} files "
                  f"({len(idx.code_files)} code, {len(idx.reference_files)} reference) "
                  f"-> {index_mod.state_dir(root)}", file=sys.stderr)
            return 0

        idx = _get_index(args, cfg, client, rebuild=getattr(args, "reindex", False))
        # Take the model load here, once, with a clear message. Otherwise a
        # profile switch stalls silently inside the first gather step.
        client.warmup()
        agent = Agent(idx, client, cfg)

        root = _root(args)
        # Slash commands rebind these, so they are shared through a dict rather
        # than captured by value.
        locals_ref = {"cfg": cfg, "client": client, "agent": agent, "idx": idx}

        if args.cmd == "explain":
            return _cmd_explain(args, client, agent)

        def run(task: str):
            """Execute one task end to end through the shared inner loop."""
            nonlocal idx, agent, cfg, client
            cfg, client, agent = locals_ref["cfg"], locals_ref["client"], locals_ref["agent"]
            idx = locals_ref["idx"]
            tr = loop.run_task(root, cfg, client, agent, idx, task,
                               apply=args.apply, review=_wants_review(args, cfg))
            idx = tr.idx
            agent.index = idx
            locals_ref["idx"] = idx

        if args.cmd == "ask":
            run(args.task)
            s = client.stats
            print(f"\n[{s['calls']} calls, {s['completion_tokens']} tokens out, "
                  f"{s['seconds']:.0f}s]", file=sys.stderr)
            return 0

        if args.cmd == "auto":
            return _cmd_auto(args, cfg, client, idx, root)

        # --- chat: interactive loop with the index kept warm between tasks ---
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
            run(task)
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