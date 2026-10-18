"""Starting a project from nothing: spec in, plan out, skeleton on disk.

Usage
-----
    from .scaffold import load_spec, build_plan, render_plan, apply_skeleton

    spec = load_spec(prompt="a CLI todo manager", spec_file=None)
    plan = build_plan(spec, client, cfg, root)
    print(render_plan(plan))            # what it intends to create
    written = apply_skeleton(root, plan, dry_run=True)

Why this exists as its own module: `ask` and `chat` are built around an index,
and an index is built around files that already exist. A brand-new project has
neither, so the gather loop would have nothing to look up. Here the model does
not need to find code -- it writes all of it -- so the flow is inverted into
three cheap, reviewable stages:

  1. PLAN     one JSON-constrained call turns the spec into a file list with
              one-line descriptions. Cheap, and it bounds what the next stage
              can do: a small model that rambles for 40 files gets stopped here
              by `max_new_files`.
  2. SKELETON local, zero LLM calls. Every planned file is created with only a
              header comment (plus a starter pyproject/README/gitignore when
              none exists). The project becomes real and indexable immediately,
              so the implementation phase has line numbers and structure to
              anchor its edits against instead of an empty directory.
  3. BUILD    per-file generation passes, run through the ordinary edit path
              (search/replace blocks anchored on the skeleton's header lines),
              so they get the same backup, syntax validation, reviewer, restore
              and session logging as any other Pedacito run.

Nothing here re-implements editing or prompting: it borrows `parse_edits`,
`apply_edits`, the answer system prompt, and the tool schema from the modules
that already own them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import fileio
from .chunker import Chunk, chunk_source
from .config import ConfigError
from .edits import apply_edits, parse_edits
from .index import Index, file_hash
from .llm import LLMError

MAX_SPEC_CHARS = 60_000          # a spec longer than this is truncated, loudly
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "project_name": {"type": "string", "maxLength": 80},
        "summary": {"type": "string", "maxLength": 600},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": 200},
                    "purpose": {"type": "string", "maxLength": 240},
                },
                "required": ["path", "purpose"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["project_name", "summary", "files"],
    "additionalProperties": False,
}

PLAN_SYSTEM = """\
You are designing a new software project from a specification. Output a file
plan only: every file that should exist, each with a one-sentence purpose.

Rules:
- Paths are relative to the project root, forward slashes, no leading "./".
- Include source files, tests, a README, and packaging/config if the spec
  implies them. Do not include lock files, binaries, or directories.
- If the spec describes a runnable program or command-line tool, include a
  `main.py` at the project root with a `main()` function and an
  `if __name__ == "__main__": main()` guard, so it runs as `python main.py`
  (packaging wires it to a console command too). Import the rest of the
  project from `main.py` rather than duplicating logic there.
- Prefer few, substantial files over many stubs: a small model loses track of
  a project that is spread too thin.
- Keep it inside the limit you are given. If the spec needs more than that,
  plan the core and say so in `summary`."""

BUILD_SYSTEM = """\
You are writing one file of a new project, from its specification and a list of
the other files that exist. Write the complete contents of YOUR file.

Output exactly one search/replace block, in this format and nothing else after it:

--- FILE: relative/path.py
<<<<<<< SEARCH
(the header comment line that file currently starts with, copied exactly)
=======
(the full contents of the file)
>>>>>>> REPLACE

Rules:
- Replace the whole file: keep the header line as the first line of your output.
- Real, working code. No placeholders like TODO or pass-only bodies unless the
  spec asks for a stub.
- Import only from files listed as existing. Never invent a sibling module.
- If YOUR file is `main.py`, give it a `main()` function and an
  `if __name__ == "__main__": main()` guard so `python main.py` runs the tool.
- Do not write any other file, and do not explain yourself."""

RESUME_SYSTEM = """\
You are implementing a project that has already been planned and scaffolded:
every file listed in the map exists on disk but holds only a one-line header
comment stating its purpose. Your task is to replace those headers with real,
working code -- one search/replace block per file you touch, anchored on that
file's exact current contents (use `read` before you edit; never invent a
SEARCH block).

Rules:
- Keep each file's header line as the first line of your replacement.
- Implement every function/class the header and the task promise; no TODOs or
  pass-only bodies unless the task explicitly asks for stubs.
- Import only from files that exist in the map. Check a sibling's outline if
  you need to know what it exposes.
- Prefer finishing a few files completely over touching many partially."""


# --------------------------------------------------------------------------
# The specification
# --------------------------------------------------------------------------
@dataclass
class Spec:
    """A project description, however it was supplied: typed, from a file, or
    read from stdin. `text` is always LF-normalised and length-capped."""

    text: str
    origin: str = "prompt"
    name_hint: str = ""

    @property
    def title(self) -> str:
        """Short label for logs and the review page header."""
        if self.name_hint:
            return self.name_hint
        for ln in self.text.splitlines():
            if ln.strip():
                return ln.strip()[:72]
        return "(empty spec)"


def load_spec(prompt: str | None = None, spec_file: str | None = None,
              max_chars: int = MAX_SPEC_CHARS) -> Spec:
    """Assemble the project specification from a -t string, a document file, or
    piped stdin (any combination; they are concatenated in that order).

    A document file is read through `fileio` so CRLF/BOM variants normalise the
    same way source files do. If nothing was supplied at all, raises ConfigError
    with the exact commands that would work."""
    parts: list[str] = []
    origin_bits: list[str] = []
    hint = ""

    if spec_file:
        p = Path(spec_file).expanduser()
        if not p.is_file():
            raise ConfigError(f"--spec file not found: {p}")
        try:
            text, _style = fileio.read(p, errors="strict")
        except (OSError, UnicodeDecodeError) as e:
            raise ConfigError(f"--spec {p} could not be read as UTF-8 text: {e}") from e
        parts.append(text.strip())
        origin_bits.append(f"file {p.name}")
        hint = p.stem.replace("_", "-").replace(" ", "-")

    if prompt:
        parts.append(prompt.strip())
        origin_bits.append("prompt")

    if not parts and not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            parts.append(piped)
            origin_bits.append("stdin")

    if not parts:
        raise ConfigError(
            "No project description given.\n"
            "  pedacito new PROJ -t \"a CLI todo manager with sqlite storage\"\n"
            "  pedacito new PROJ --spec docs/SPEC.md\n"
            "  cat SPEC.md | pedacito new PROJ")

    text = "\n\n".join(parts)
    note = ""
    if len(text) > max_chars:
        note = (f"[spec truncated from {len(text):,} to {max_chars:,} characters; "
                f"raise max_spec_chars in your config if this cut something "
                f"important]")
        text = text[:max_chars]
    if note:
        print(note, file=sys.stderr)
    return Spec(text=text, origin=", ".join(origin_bits), name_hint=hint)


# --------------------------------------------------------------------------
# Stage 1: the plan
# --------------------------------------------------------------------------
@dataclass
class PlannedFile:
    """One entry of a project plan."""

    path: str
    purpose: str = ""


@dataclass
class Plan:
    """A proposed new project: its name, summary, and the files it will contain."""

    project_name: str
    summary: str
    files: list[PlannedFile] = field(default_factory=list)
    spec: Spec | None = None

    @property
    def sources(self) -> list[PlannedFile]:
        """Non-test, non-config files -- the ones worth a build pass each."""
        return [f for f in self.files
                if "/test" not in f.path and not f.path.startswith("test")
                and not f.path.endswith((".md", ".toml", ".txt", ".cfg", ".ini"))]


def _safe_rel(path: str, root: Path) -> str | None:
    """Normalise a model-supplied path and refuse anything that escapes the
    project root (absolute paths, '..', backslash tricks). Returns the clean
    POSIX-relative path, or None if the entry must be dropped."""
    rel = fileio.normalise_rel(path).lstrip("/")
    while rel.startswith("./"):
        rel = rel[2:]
    if not rel or rel.endswith("/"):
        return None
    probe = (root / rel) if root is not None else Path(rel)
    try:
        probe.resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return None
    if ".." in Path(rel).parts:
        return None
    return rel


def build_plan(spec: Spec, client, cfg, root: Path, verbose: bool = True) -> Plan:
    """Ask the model for a file plan, then sanitise it: dedupe paths, drop
    entries that escape the root or have no extension, and cap the count at
    cfg.max_new_files. Raises LLMError if the model returns nothing usable."""
    limit = getattr(cfg, "max_new_files", 24)
    budget = getattr(cfg, "max_spec_chars", MAX_SPEC_CHARS)
    body = spec.text[:budget]
    user = (f"SPECIFICATION ({spec.origin})\n\n{body}\n\n"
            f"Plan at most {limit} files for this project.")
    messages = [{"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": user}]
    raw = client.chat_json(messages, PLAN_SCHEMA,
                           max_tokens=getattr(cfg, "max_tokens_plan", 2000),
                           disable_thinking=True)
    if not isinstance(raw, dict):
        raise LLMError("The planning call did not return an object.")

    seen: set[str] = set()
    files: list[PlannedFile] = []
    dropped = 0
    for entry in raw.get("files") or []:
        if not isinstance(entry, dict):
            dropped += 1
            continue
        rel = _safe_rel(str(entry.get("path", "")), root)
        if rel is None or rel in seen:
            dropped += 1
            continue
        if "." not in Path(rel).name:
            dropped += 1               # a directory, not a file
            continue
        seen.add(rel)
        files.append(PlannedFile(rel, str(entry.get("purpose", "")).strip()))

    if len(files) > limit:
        dropped += len(files) - limit
        files = files[:limit]
    if not files:
        raise LLMError(
            "The model proposed no usable files.\n"
            "  Try a more concrete spec, or a stronger profile: planning a file\n"
            "  tree is the easiest task in this tool and a model that fails it\n"
            "  will not implement one either.")
    files.sort(key=lambda f: f.path)

    plan = Plan(project_name=str(raw.get("project_name") or spec.title)[:80],
                summary=str(raw.get("summary") or "").strip(),
                files=files, spec=spec)
    if verbose:
        print(f"  plan: {len(files)} file(s)"
              + (f", {dropped} unusable entr{'y' if dropped == 1 else 'ies'} dropped"
                 if dropped else ""), file=sys.stderr)
    return plan


def render_plan(plan: Plan) -> str:
    """Human-readable form of a plan, used both for --dry-run output and as the
    shared context handed to every build pass."""
    lines = [f"# {plan.project_name}", ""]
    if plan.summary:
        lines += [plan.summary, ""]
    lines.append("Files:")
    for f in plan.files:
        lines.append(f"  {f.path}" + (f"  -- {f.purpose}" if f.purpose else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Stage 2: the skeleton (no LLM calls)
# --------------------------------------------------------------------------
STARTER_GITIGNORE = """\
__pycache__/
*.py[cod]
.venv/
venv/
build/
dist/
*.egg-info/
"""

STARTER_PYPROJECT = """\
[project]
name = "{name}"
version = "0.1.0"
description = "{summary}"
requires-python = ">=3.11"

[project.scripts]
{name} = "main:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
"""


def _header(rel: str, purpose: str) -> str:
    """Skeleton content for one planned file: a docstring/comment naming the
    file and its purpose. This is also the anchor line the build pass replaces,
    so it must be a single line that parses in any language."""
    safe = (purpose or rel).replace("\n", " ").strip()
    ext = Path(rel).suffix.lower()
    if ext == ".py":
        return f'"""{rel} -- {safe}"""\n'
    if ext in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs", ".css"):
        return f"// {rel} -- {safe}\n"
    if ext in (".rb", ".sh", ".yaml", ".yml", ".toml", ".ini", ".cfg"):
        return f"# {rel} -- {safe}\n"
    if ext in (".md", ".rst", ".txt"):
        return f"{rel}\n\n{safe}\n"
    if ext == ".json":
        return '{\n  "_comment": "' + safe.replace('"', "'") + '"\n}\n'
    return f"{rel}\n{safe}\n"


def apply_skeleton(root: Path, plan: Plan, dry_run: bool = False,
                   verbose: bool = True) -> list[str]:
    """Create every planned file (and a starter pyproject.toml, README, and
    .gitignore when missing) containing only its header line. Existing files
    are never touched, so a second `new` run on the same directory is safe.

    Returns the relative paths actually created (or that would be created)."""
    created: list[str] = []
    wanted: dict[str, str] = {}
    for f in plan.files:
        wanted[f.path] = _header(f.path, f.purpose)

    readme = next((f.path for f in plan.files
                   if f.path.lower() in ("readme.md", "docs/readme.md")), None)
    if readme is None:
        wanted.setdefault("README.md",
                          f"# {plan.project_name}\n\n{plan.summary or ''}\n\n"
                          "## Run\n\n```bash\npython main.py\n```\n")
    name = "".join(c for c in plan.project_name.lower().replace(" ", "-")
                   if c.isalnum() or c == "-").strip("-") or "project"
    wanted.setdefault("pyproject.toml",
                      STARTER_PYPROJECT.format(name=name,
                                               summary=(plan.summary or name)[:120]))
    wanted.setdefault(".gitignore", STARTER_GITIGNORE)

    for rel, text in sorted(wanted.items()):
        path = root / rel
        if path.exists():
            continue
        if dry_run:
            created.append(rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        fileio.write(path, text)
        created.append(rel)
    if verbose and created:
        print(f"  skeleton: {len(created)} file(s) "
              f"{'would be created' if dry_run else 'created'}", file=sys.stderr)
    return created


# --------------------------------------------------------------------------
# Indexing the skeleton
# --------------------------------------------------------------------------
def skeleton_index(root: Path, plan: Plan, cfg) -> Index:
    """Build an Index straight from the skeleton, without walking the tree.

    `index.build()` refuses to run on a project with nothing selectable in it,
    and it would also pick up unrelated neighbours; here the file list is known
    exactly, so chunk each planned file directly. Static only -- no summaries,
    which cost an LLM call each and are pointless before the files have bodies.
    """
    chunks: list[Chunk] = []
    hashes: dict[str, str] = {}
    for f in plan.files:
        path = root / f.path
        if not path.is_file():
            continue
        hashes[f.path] = file_hash(path)
        src = fileio.read_text(path, errors="replace")
        try:
            chunks.extend(chunk_source(f.path, src, cfg.class_split_lines))
        except SyntaxError:
            pass                       # a header line cannot break, but be safe
    idx = Index(root, chunks, {}, hashes)
    idx.save()
    return idx


# --------------------------------------------------------------------------
# Stage 3: building the files
# --------------------------------------------------------------------------
def _existing_map(idx, plan: Plan, cfg, max_files: int = 12,
                  max_chars: int = 4000) -> str:
    """Compact `path: signature` listing of everything already implemented,
    so a build pass can import siblings correctly. Falls back to the plan's own
    file list when the index is still empty."""
    lines: list[str] = []
    for rel in idx.files[:max_files]:
        names = [c.signature or c.qualname for c in idx.top_level(rel)][:12]
        lines.append(f"{rel}: " + ", ".join(n for n in names if n))
    if not lines:
        lines = [f"{f.path}: {f.purpose}" for f in plan.files[:max_files]]
    text = "\n".join(lines)
    return text[:max_chars]


def build_pass(client, cfg, idx, plan: Plan, target: PlannedFile,
               done: list[str], verbose: bool = True,
               backup_dir=None) -> tuple[list, str]:
    """Generate one file's contents and apply it through the normal edit path.

    The prompt contains the spec, the plan, what already exists, and the exact
    header line to anchor on -- no repo map, because there is nothing to
    navigate yet. Returns (results, reply)."""
    from .agent import ANSWER_SYSTEM
    from .edits import Result

    anchor = fileio.read_text(idx.root / target.path, errors="replace").split("\n")[0]
    spec = plan.spec.text if plan.spec else ""
    budget = getattr(cfg, "max_spec_chars", MAX_SPEC_CHARS) // 2
    ctx = [
        f"PROJECT: {plan.project_name}",
        "",
        f"SPECIFICATION\n{spec[:budget]}",
        "",
        "FILE TREE\n" + "\n".join(
            f"  {f.path}" + ("  <- YOU ARE WRITING THIS ONE" if f.path == target.path
                             else f"  -- {f.purpose}" if f.purpose else "")
            for f in plan.files),
        "",
        "ALREADY IMPLEMENTED\n" + (_existing_map(idx, plan, cfg) or "(nothing yet)"),
        "",
        f"YOUR FILE: {target.path}",
        f"It currently contains exactly one line:\n  {anchor}",
        "",
        "Write this file now, as one search/replace block that replaces that "
        "line with the complete file.",
    ]
    if done:
        ctx.insert(-2, "DONE SO FAR: " + ", ".join(done) + "\n")
    messages = [
        {"role": "system", "content": BUILD_SYSTEM + "\n\n" + ANSWER_SYSTEM},
        {"role": "user", "content": "\n".join(ctx)},
    ]
    max_tokens = getattr(cfg, "max_tokens_build", 8000)

    reply = client.chat(messages, max_tokens=max_tokens, stream=verbose,
                        disable_thinking=True)
    if not parse_edits(reply):
        # A reasoning model can spend its whole budget on the <think> block and
        # return nothing usable -- some Qwen quants ignore enable_thinking=false.
        # Nudge once with the no-think suffix plus a blunt instruction, then
        # report the failure rather than silently dropping the file.
        suffix = getattr(cfg, "thinking_suffix", "") or "/no_think"
        messages[-1]["content"] += (
            f"\n\n{suffix}\n"
            "Output the complete file NOW as one search/replace block, and "
            "nothing else. Do not explain.")
        reply = client.chat(messages, max_tokens=max_tokens, stream=verbose,
                            disable_thinking=True)

    if not parse_edits(reply):
        return [Result(False, target.path,
                       "model returned no edit block (a thinking model may have "
                       "spent the budget on its <think> block; try "
                       "thinking_suffix=/no_think or a non-reasoning model)")], reply
    results = apply_edits(idx.root, parse_edits(reply), backup=cfg.backup,
                          dry_run=False, cfg=cfg, backup_dir=backup_dir)
    return results, reply


def build_project(client, cfg, idx, plan: Plan, verbose: bool = True,
                  on_file=None, backup_dir=None) -> dict:
    """Run a build pass for each planned source file, in plan order, stopping
    early if the server goes away. Returns a summary dict:

        {"done": [paths], "failed": [(path, message)], "skipped": [paths]}

    Files are built one at a time rather than in one giant answer: a 9B model
    asked for a whole project emits a plausible-looking wall of code, whereas
    asked for one file it usually writes a working one. Order matters -- the
    plan is sorted, so modules come before the things that import them where
    the model planned sensibly."""
    targets = plan.sources or plan.files
    done: list[str] = []
    failed: list[tuple[str, str]] = []
    skipped: list[str] = []
    for t in targets:
        path = idx.root / t.path
        if not path.is_file():
            skipped.append(t.path)
            continue
        if verbose:
            print(f"\n--- building {t.path} ---", file=sys.stderr)
        try:
            results, _reply = build_pass(client, cfg, idx, plan, t, done,
                                         verbose, backup_dir=backup_dir)
        except LLMError as e:
            print(f"  stopped: {e}", file=sys.stderr)
            failed.append((t.path, "server error"))
            skipped.extend(x.path for x in targets
                           if x.path not in done and x.path != t.path)
            break
        ok = [r for r in results if r.ok]
        bad = [r for r in results if not r.ok]
        if ok:
            done.append(t.path)
            from . import index as index_mod
            for r in ok:
                idx = index_mod.reindex_file(idx, r.file, cfg)
            if on_file is not None:
                on_file(t.path, idx)
        if bad:
            failed.append((t.path, "; ".join(r.message for r in bad)[:200]))
            if verbose:
                for r in bad:
                    print(f"  FAILED {r.file}: {r.message}", file=sys.stderr)
    return {"done": done, "failed": failed, "skipped": skipped}


def summarise_result(summary: dict, verbose: bool = True) -> str:
    """One-paragraph report of a build run, printed at the end of `new`."""
    bits = [f"{len(summary['done'])} file(s) written"]
    if summary["failed"]:
        bits.append(f"{len(summary['failed'])} failed")
    if summary["skipped"]:
        bits.append(f"{len(summary['skipped'])} skipped")
    text = ", ".join(bits) + "."
    if verbose and summary["failed"]:
        text += "\n  " + "\n  ".join(f"{p}: {m}" for p, m in summary["failed"])
    return text


# --------------------------------------------------------------------------
# The whole flow, for the CLI's `new` command
# --------------------------------------------------------------------------
def run_new(root: Path, cfg, client, spec: Spec, *, apply: bool = False,
            review: bool = False, plan_only: bool = False, force: bool = False,
            verbose: bool = True) -> int:
    """Plan a new project from `spec`, lay down its skeleton, and (unless
    asked to stop earlier) build every source file through the normal edit
    path. Returns a process exit code; all progress goes to stderr so stdout
    stays clean.

    Stage gates, cheapest first:
      --plan-only  stop after stage 1+2: a tree of headers plus a saved index,
                   ready for `pedacito resume` or ordinary `ask --apply` runs.
      default      dry run: print the plan and a preview of the skeleton,
                   nothing is written to disk.
      --apply      write the generated files to disk (backups taken first).
      --review     build, then open the side-by-side reviewer with the result
                   (Discard restores the pre-build snapshot).

    A non-empty project directory is refused unless `force`: silently mixing a
    new project into an existing one is exactly the mistake this guards."""
    if not force and any(root.glob("*")):
        existing = sorted(p.name for p in root.iterdir()
                          if p.name not in (".git", ".pedacito"))[:8]
        if existing:
            print(f"ERROR: {root} is not empty ({', '.join(existing)}"
                  f"{'...' if len(list(root.iterdir())) > 8 else ''}).\n"
                  "  New projects start in a fresh directory. Pick another path,\n"
                  "  or pass --force to scaffold among those files anyway.",
                  file=sys.stderr)
            return 2

    plan = build_plan(spec, client, cfg, root, verbose=verbose)
    print(render_plan(plan))

    # Write modes. The default (no --apply/--review/--plan-only) is a true dry
    # run: show the plan and what the skeleton WOULD contain, write nothing.
    # --plan-only writes just the header skeleton; --apply and --review write
    # the skeleton and then build the files (review holds the result for
    # sign-off). The old code always wrote the skeleton AND the build passes,
    # so a bare `new` left a full project on disk and `--apply` simply re-ran
    # the same work.
    wants_skeleton = plan_only or apply or review
    created = apply_skeleton(root, plan, dry_run=not wants_skeleton,
                             verbose=verbose)
    if not wants_skeleton:
        if created:
            preview = ", ".join(created[:12])
            if len(created) > 12:
                preview += f", ... (+{len(created) - 12})"
            print(f"  [dry run] {len(created)} skeleton file(s) would be "
                  f"created: {preview}", file=sys.stderr)
        print("  Nothing was written. Re-run with --apply to build the "
              "project, or --plan-only to write just the header skeleton.",
              file=sys.stderr)
        return 0

    idx = skeleton_index(root, plan, cfg)
    print(f"  skeleton on disk ({len(created)} new file(s)), indexed -> "
          f"{idx.root / '.pedacito'}", file=sys.stderr)

    if plan_only:
        print("\nStopped after planning (--plan-only). The tree above exists "
              "with header stubs.\n"
              f"  Implement it later with:  pedacito ask {root} --apply "
              "-t \"implement the stub files\"", file=sys.stderr)
        return 0

    want_write = apply or review
    when = None
    backup_dir = None
    if want_write and cfg.backup:
        from . import workspace
        when = workspace.stamp()
        backup_dir = workspace.new_backup_dir(root, when)

    print(f"\n--- building {len(plan.sources)} file(s) "
          f"({'WRITING' if apply else 'for review'}) ---",
          file=sys.stderr)
    summary = build_project(client, cfg, idx, plan, verbose=verbose,
                            backup_dir=backup_dir)
    print("\n" + summarise_result(summary))

    if review and summary["done"]:
        results = _review_build(root, plan, summary, cfg, backup_dir, when)
        if results is None:
            return 1
    if cfg.log_sessions and when is not None:
        from . import workspace
        p = workspace.write_session(root, f"new: {plan.project_name}",
                                    [], render_plan(plan), [], backup_dir, when)
        print(f"[session: {p.relative_to(root)}]", file=sys.stderr)
    return 0 if not summary["failed"] else 1


def _review_build(root: Path, plan: Plan, summary: dict, cfg,
                  backup_dir, stamp) -> list | None:
    """Open the side-by-side reviewer over a finished build, using the
    pre-build backups as the 'before' side. The build path writes directly
    (it has no dry-run mode of its own), so `new --review` means: back up,
    build, then hold everything on the review page where Apply keeps the
    files and Discard restores them from the snapshot."""
    from . import review as review_mod
    diffs = review_mod.diffs_from_backup(root, backup_dir)
    if not diffs:
        print("  no review page: nothing differed from the backups", file=sys.stderr)
        return None
    nonce = review_mod.new_nonce()
    page = review_mod.build_page(diffs, f"new: {plan.project_name}", cfg.model,
                                 cfg.active_profile, nonce, interactive=True,
                                 stamp=stamp)
    saved = review_mod.save(root, page, stamp)
    print(f"  review page: {saved}", file=sys.stderr)

    def discard():
        """Undo the whole build: put every backed-up stub back."""
        from . import workspace
        workspace.restore(root, backup_dir)

    outcome, detail = review_mod.run(
        page, nonce, lambda: None,          # files are already on disk
        port=cfg.review_port, open_browser=cfg.open_browser,
        timeout=cfg.review_timeout, on_url=lambda u: print(
            f"  waiting for sign-off at {u}", file=sys.stderr),
        on_discard=discard)
    print(f"  {detail}", file=sys.stderr)
    return [] if outcome == "applied" else None
