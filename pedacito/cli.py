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
    pedacito restore src/                    # undo the last --apply

Run as `pedacito <command> ...` (after `pip install -e .`) or
`python -m pedacito <command> ...`. `main()` is the single entry point invoked
by both `__main__.py` files. `pdc` is installed as a shorthand alias for
`pedacito` and accepts the exact same commands, e.g. `pdc ask src/ -t "..."`.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from . import index as index_mod
from .agent import Agent
from .edits import parse_edits
from .config import Config, ConfigError, USER_CONFIG, find_project_config, load as load_config, write_template
from .llm import LMStudio, LLMError, estimate_tokens
from .select import CODE_EXTS, REFERENCE_EXTS, Matcher, select
from . import review as review_mod
from . import workspace


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
    p.add_argument("--steps", type=int, help="max lookups in the gather phase")


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
    cfg = load_config(root=_root(args), explicit=getattr(args, "config", None),
                      profile=getattr(args, "profile", None))
    if getattr(args, "no_refs", False):
        cfg.include_references = False
    if getattr(args, "ext", None):
        cfg.extra_extensions = tuple(args.ext)
    if getattr(args, "url", None):
        cfg.base_url = args.url
    if getattr(args, "model", None):
        cfg.model = args.model
    if getattr(args, "steps", None):
        cfg.max_steps = args.steps
        cfg.sources["max_steps"] = "--steps"
    for flag, key in (("url", "base_url"), ("model", "model")):
        if getattr(args, flag, None):
            cfg.sources[key] = f"--{flag}"
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
    for rel in missing:
        idx.forget(rel)
    for rel in changed:
        idx = index_mod.reindex_file(idx, rel, cfg)

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
    if added:
        idx = index_mod.build(args.paths, cfg, client=None, root=root,
                              summarise=False, verbose=False,
                              matcher=_matcher(args, root))
        if verbose:
            print(f"  new files need summaries: run `pedacito index` when convenient",
                  file=sys.stderr)
    if changed or missing:
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
    if args.init:
        try:
            path = write_template(Path(args.path) if args.path else None, args.force)
        except ConfigError as e:
            print(f"{e}", file=sys.stderr)
            return 2
        print(f"Wrote {path}", file=sys.stderr)
        print("Edit base_url and model, then run `pedacito config --check`.", file=sys.stderr)
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
    if cfg.active_profile:
        print(f"  active: {cfg.active_profile}")

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
        print(f"  configured model '{cfg.model}': NOT IN THE LIST above.")
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
        if not cfg.profiles:
            print("  No profiles defined. Add [profiles.NAME] to your config.",
                  file=sys.stderr)
        for name, body in sorted(cfg.profiles.items()):
            mark = "*" if name == cfg.active_profile else " "
            print(f"  {mark} {name}: {body.get('model', '(inherits model)')}",
                  file=sys.stderr)
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
            ref["cfg"].max_steps = max(1, int(arg))
        except ValueError:
            print("  Usage: /steps 8", file=sys.stderr)
            return True
        ref["agent"].cfg = ref["cfg"]
        print(f"  max_steps = {ref['cfg'].max_steps}", file=sys.stderr)
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
        label = f"profile {new_cfg.active_profile}, " if new_cfg.active_profile else ""
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


def _review(root: Path, results, task: str, cfg, backup_dir, stamp: str) -> list:
    """Build and serve the review page for a set of edit results, block
    until the person applies or discards, and return the results as they
    finally stand (so the session log records what actually happened)."""
    diffs = review_mod.diffs_from_results(results)
    if not diffs:
        # Nothing anchored, so there is nothing to review. Say why rather than
        # silently doing nothing.
        print("\n  No review page: no edit block applied cleanly.", file=sys.stderr)
        _report(results, dry_run=True)
        return results
    nonce = review_mod.new_nonce()
    page = review_mod.build_page(diffs, task, cfg.model, cfg.active_profile,
                                 nonce, interactive=True, stamp=stamp)
    saved = review_mod.save(root, page, stamp)

    def writer():
        """Write the reviewed files to disk when the page's Apply fires."""
        return review_mod.write_files(root, diffs, backup_dir)

    print(f"  review page: {saved}", file=sys.stderr)
    outcome, detail = review_mod.run(
        page, nonce, writer, port=cfg.review_port,
        open_browser=cfg.open_browser, timeout=cfg.review_timeout,
        on_url=lambda u: print(f"  waiting for sign-off at {u}", file=sys.stderr))

    if outcome == "applied":
        print(f"  {detail}", file=sys.stderr)
        return [dataclasses.replace(r, message="applied via review")
                if r.ok else r for r in results]
    print(f"  {detail}", file=sys.stderr)
    return [dataclasses.replace(r, message=f"not written ({outcome})")
            if r.ok else r for r in results]


# --------------------------------------------------------------------------
# Terminal reporting
# --------------------------------------------------------------------------
def _report(results, dry_run: bool):
    """Print each edit Result to stderr, including its unified diff when in
    dry-run mode."""
    if not results:
        print("\n[no edit blocks in the reply]", file=sys.stderr)
        return
    print("", file=sys.stderr)
    for r in results:
        mark = "ok " if r.ok else "FAIL"
        print(f"[{mark}] {r.file}: {r.message}", file=sys.stderr)
        if r.ok and dry_run and r.diff:
            print(r.diff)


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
    p_map.add_argument("--tier", choices=["auto", "cards", "flat"], default="auto")

    p_conf = sub.add_parser("config", help="show, create, or test the config file")
    p_conf.add_argument("--init", action="store_true",
                        help=f"write a starter config to {USER_CONFIG}")
    p_conf.add_argument("--path", metavar="FILE", help="write --init here instead")
    p_conf.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_conf.add_argument("--check", action="store_true",
                        help="connect to LM Studio and verify the model is loaded")
    p_conf.add_argument("--root", default=None)
    p_conf.add_argument("--profile", metavar="NAME")
    p_conf.add_argument("paths", nargs="*", default=["."], help=argparse.SUPPRESS)
    p_conf.add_argument("--config", metavar="FILE")

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

    p_chat = sub.add_parser("chat", help="interactive session")
    _add_common(p_chat)
    p_chat.add_argument("--apply", action="store_true")
    p_chat.add_argument("--review", action="store_true")
    p_chat.add_argument("--no-review", dest="no_review", action="store_true")

    p_gui = sub.add_parser("gui", help="open the browser control panel")
    p_gui.add_argument("--port", type=int, default=0)
    p_gui.add_argument("--no-open", dest="open_browser", action="store_false")

    args = ap.parse_args(argv)
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
            return 0

        client = LMStudio(cfg)
        client.check()

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

        def run(task: str):
            """Execute one task end to end: gather source, get an answer or
            edits, apply or review them, and write the session log."""
            nonlocal idx, agent, cfg, client
            cfg, client, agent = locals_ref["cfg"], locals_ref["client"], locals_ref["agent"]
            idx = locals_ref["idx"]
            when = workspace.stamp()
            reviewing = _wants_review(args, cfg)
            wants_write = args.apply or reviewing
            backup_dir = (workspace.new_backup_dir(root, when)
                          if wants_write and cfg.backup else None)
            print("\n--- looking things up ---", file=sys.stderr)
            gathered = agent.gather(task)
            print("--- answering ---\n", file=sys.stderr)
            reply = agent.answer(task, gathered)
            # With the reviewer, the run stays a dry run: the page writes.
            results, reply = agent.apply(reply, dry_run=reviewing or not args.apply,
                                         task=task, gathered=gathered,
                                         backup_dir=None if reviewing else backup_dir)
            if reviewing:
                if not parse_edits(reply):
                    print("\n  The model proposed no edits.", file=sys.stderr)
                results = _review(root, results, task, cfg, backup_dir, when)
                applied = any(r.ok and "applied" in r.message for r in results)
            else:
                _report(results, dry_run=not args.apply)
                applied = args.apply
            if applied:
                for r in results:
                    if r.ok:
                        idx = index_mod.reindex_file(idx, r.file, cfg)
                agent.index = idx
                locals_ref["idx"] = idx
            if cfg.log_sessions:
                p = workspace.write_session(root, task, getattr(agent, "steps_log", []),
                                            reply, results, backup_dir, when)
                print(f"[session: {p.relative_to(root)}]", file=sys.stderr)

        if args.cmd == "ask":
            run(args.task)
            s = client.stats
            print(f"\n[{s['calls']} calls, {s['completion_tokens']} tokens out, "
                  f"{s['seconds']:.0f}s]", file=sys.stderr)
            return 0

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
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
