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

from . import apicheck
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

INTERFACE_SYSTEM = """\
You are designing the public API of a new Python project. Given a specification
and a file plan, write, for EVERY planned source file, its complete public
interface: every function and class with its full signature and a one-line
docstring. Bodies must be `pass`, or `raise NotImplementedError` for functions
that must return a value. Include the imports each file needs. Do NOT implement
any body -- signatures only.

This single reply is the contract every later file is implemented against, so
the names must be consistent: a name imported in one file must be defined with
exactly that spelling in the file that exports it. Prefer simple, obvious names
(main, parse, render, run) and reuse the same name everywhere instead of
inventing aliases."""

INTERFACE_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": 200},
                    "code": {"type": "string", "maxLength": 12000},
                },
                "required": ["path", "code"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["files"],
    "additionalProperties": False,
}

BUILD_SYSTEM = """\
You are implementing one file of a new Python project. The file already exists
with its public interface stubbed out: every function and class has its full
signature and a one-line docstring, and the body is `pass` or
`raise NotImplementedError`.

Output exactly one search/replace block, in this format and nothing else after it:

--- FILE: relative/path.py
<<<<<<< SEARCH
(the file's current stubbed contents, copied exactly)
=======
(the same signatures with their bodies implemented)
>>>>>>> REPLACE

Rules:
- Keep every function/class signature and docstring EXACTLY as written; only
  replace the stub bodies with real code. Do not rename or reorder anything --
  other files import these by their exact names.
- Real, working code. No TODOs or pass-only bodies unless the spec asks for a
  stub.
- Import only from files listed as existing. Never invent a sibling module.
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

AUTO_IMPLEMENT_SYSTEM = """\
You are implementing part of a project that has already been planned and
scaffolded: the map's files exist on disk with their public interface stubbed
out -- full function/class signatures and one-line docstrings, with bodies of
`pass` or `raise NotImplementedError`.

The task names what to implement. Read it carefully:

- If the task names a whole file (just a path), output exactly one
  search/replace block for it with an EMPTY SEARCH section (nothing between
  <<<<<<< SEARCH and =======) and the COMPLETE implemented file in REPLACE.
  An empty SEARCH replaces the whole file; do NOT copy the stub into SEARCH.

- If the task names specific definitions ("Implement game/physics.py: foo, bar
  (lines N-M)"), output ONE search/replace block PER named definition: SEARCH
  is that definition's exact current stub (signature + docstring + `pass` or
  `raise NotImplementedError`), REPLACE is the same signature with a real body.
  Implement only the definitions the task names; leave the rest untouched.

Rules:
- `read` the file first so you can keep every signature and docstring EXACTLY
  as written -- other files import these names. Only fill in the stub bodies.
- No TODOs or pass-only bodies unless the task explicitly asks for stubs.
- Import only from files that exist in the map. Check a sibling's outline if
  you need to know what it exposes.
- Finish the named work completely; do not wander into other functions or files."""


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


def build_interface(plan: Plan, client, cfg, verbose: bool = True) -> dict[str, str]:
    """Fix the public API of every source file in ONE call, so the later
    per-file build only fills bodies against known signatures instead of
    inventing a different API per file. Returns {path: stub_code}.

    This is the step that makes multi-file scaffolding tractable for a small
    model: designing one consistent set of signatures is done in a single
    context (so names line up), and each build pass afterwards is the local
    task of "implement this one file's known signatures"."""
    sources = plan.sources or plan.files
    file_tree = "\n".join(f"  {f.path}  -- {f.purpose}" for f in sources)
    body = plan.spec.text[:getattr(cfg, "max_spec_chars", MAX_SPEC_CHARS) // 2]
    messages = [
        {"role": "system", "content": INTERFACE_SYSTEM},
        {"role": "user", "content":
            f"SPECIFICATION\n\n{body}\n\nFILE TREE\n\n{file_tree}\n\n"
            f"Write the signature stubs for each of these {len(sources)} "
            "files."},
    ]
    raw = client.chat_json(messages, INTERFACE_SCHEMA,
                           max_tokens=getattr(cfg, "max_tokens_build", 16000),
                           disable_thinking=True)
    out: dict[str, str] = {}
    for entry in raw.get("files") or []:
        if not isinstance(entry, dict):
            continue
        rel = fileio.normalise_rel(str(entry.get("path", "")))
        code = str(entry.get("code", ""))
        if rel and code.strip():
            out[rel] = code
    if verbose:
        print(f"  interface: {len(out)} file(s) with signature stubs",
              file=sys.stderr)
    return out


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


def _top_units(idx, rel: str) -> list[tuple[str, int, int]]:
    """A file's top-level implementation units as (qualname, start, end).

    A unit is one function, or one whole class (its methods included). Small
    classes arrive as a single "class" chunk; larger ones as a "classheader"
    chunk whose `scope_end` is the whole class, followed by "method" chunks
    that belong to it and must be skipped. This is the granularity a split
    task can be handed to the model as "implement these named definitions"."""
    units = []
    chunks = idx.in_file(rel)
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if c.kind == "classheader":
            units.append((c.qualname, c.start, c.scope_end or c.end))
            i += 1
            while i < len(chunks) and chunks[i].kind == "method":
                i += 1
        elif c.kind in ("function", "class"):
            units.append((c.qualname, c.start, c.end))
            i += 1
        else:
            i += 1
    return units


def plan_tasks(plan: Plan, idx=None, max_lines: int = 0) -> list[str]:
    """One implementation task per source file, in plan (dependency) order.

    This is the checklist `new --complex` hands to the autonomous loop: each
    task names exactly one source file to implement against its stub, so a
    fresh context window gets one bounded, verifiable unit of work.

    `__init__.py` files are skipped: a package marker is already complete the
    moment the skeleton is written, so tasking the model with "implement" one
    makes it (correctly) answer "nothing to do" -- which the loop would then
    read as a failed task and stop on.

    When `idx` and a positive `max_lines` are given, a source file whose stub
    is longer than `max_lines` is split into several tasks, one per group of
    top-level definitions (each group small enough to implement in one reply).
    This is the line guard that stops a whole-file write from overflowing the
    model's output budget."""
    tasks = []
    for f in (plan.sources or plan.files):
        if Path(f.path).name == "__init__.py":
            continue
        purpose = f" -- {f.purpose}" if f.purpose else ""
        units = _top_units(idx, f.path) if idx is not None else []
        total = max((end for _, _s, end in units), default=0)
        if max_lines > 0 and total > max_lines and len(units) > 1:
            for group in _group_units(units, max_lines):
                names = ", ".join(q for q, _s, _e in group)
                tasks.append(f"Implement {f.path}: {names} "
                             f"(lines {group[0][1]}-{group[-1][2]})")
        else:
            tasks.append(f"Implement {f.path}{purpose}")
    return tasks


def _group_units(units: list[tuple[str, int, int]],
                 max_lines: int) -> list[list[tuple[str, int, int]]]:
    """Split `units` (sorted by start) into consecutive groups whose combined
    line span stays within `max_lines`, keeping each unit whole."""
    groups: list[list[tuple[str, int, int]]] = []
    cur: list[tuple[str, int, int]] = []
    for u in units:
        if cur and (u[2] - cur[0][1] + 1) > max_lines:
            groups.append(cur)
            cur = []
        cur.append(u)
    if cur:
        groups.append(cur)
    return groups


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
                   verbose: bool = True,
                   stubs: dict[str, str] | None = None) -> list[str]:
    """Create every planned file (and a starter pyproject.toml, README, and
    .gitignore when missing). Source files with an entry in `stubs` get their
    signature-only stub; everything else gets a one-line header comment.
    Existing files are never touched, so a second `new` run on the same
    directory is safe.

    Returns the relative paths actually created (or that would be created)."""
    stubs = stubs or {}
    created: list[str] = []
    wanted: dict[str, str] = {}
    for f in plan.files:
        wanted[f.path] = stubs.get(f.path, _header(f.path, f.purpose))

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
    """Generate one file's implementation and apply it through the normal edit
    path, against the signature stub already written by the interface step.

    The prompt contains the spec, the plan, every file's known signature, and
    the full stub of the target file to replace -- no repo map, because there
    is nothing to navigate yet. Returns (results, reply)."""
    from .agent import ANSWER_SYSTEM
    from .edits import Result

    stub = fileio.read_text(idx.root / target.path, errors="replace")
    spec = plan.spec.text if plan.spec else ""
    budget = getattr(cfg, "max_spec_chars", MAX_SPEC_CHARS) // 2
    ctx = [
        f"PROJECT: {plan.project_name}",
        "",
        f"SPECIFICATION\n{spec[:budget]}",
        "",
        "FILE TREE\n" + "\n".join(
            f"  {f.path}" + ("  <- YOU ARE IMPLEMENTING THIS ONE" if f.path == target.path
                             else f"  -- {f.purpose}" if f.purpose else "")
            for f in plan.files),
        "",
        "PUBLIC API (signatures every file imports by)\n" +
        (_existing_map(idx, plan, cfg) or "(nothing yet)"),
        "",
        f"YOUR FILE: {target.path}",
        f"It currently contains this stubbed interface:\n```python\n{stub}\n```",
        "",
        "Implement it now, as one search/replace block that replaces the stub "
        "with the same signatures and real, working bodies.",
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
# Post-build import check / fix pass
# --------------------------------------------------------------------------
def _first_import_error(root: Path, plan: Plan) -> str | None:
    """Import the project's entry point in a subprocess and return the first
    error (ImportError / AttributeError / ...), or None if it imports cleanly.

    Files are generated one at a time, so a model can name an API in one file
    (`parse_markdown`) and define it under a different name in another
    (`parse`). This surfaces that mismatch instead of leaving the project to
    fail on its first run."""
    import subprocess
    py = [f.path for f in plan.files
          if f.path.endswith(".py") and (root / f.path).is_file()]
    if not py:
        return None
    entry = next((p for p in py if p == "main.py"), None)
    if entry is None:
        entry = next((p for p in py if p.endswith("__init__.py")), None)
    if entry is None:
        entry = py[0]
    mod = entry[:-3].replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {mod}"], cwd=str(root),
            capture_output=True, text=True, timeout=120)
    except Exception as e:                      # pragma: no cover
        return f"(could not run import check: {e})"
    if proc.returncode == 0:
        return None
    return (proc.stderr or proc.stdout).strip()


def fix_import_errors(client, cfg, idx: "Index", plan: Plan, backup_dir,
                      verbose: bool = True, rounds: int = 3) -> int:
    """Re-ask the model to repair import errors left by the per-file build.

    On each round: import the project, and if it fails, hand the model the
    error plus every built .py file and ask for search/replace fixes. Returns
    the number of successful fix blocks applied."""
    from .agent import ANSWER_SYSTEM

    fixed = 0
    for _ in range(rounds):
        err = _first_import_error(idx.root, plan)
        if err is None:
            break
        files = {}
        for f in plan.files:
            p = idx.root / f.path
            if f.path.endswith(".py") and p.is_file():
                files[f.path] = fileio.read_text(p)
        if not files:
            break
        listing = "\n\n".join(
            f"### {rel}\n```python\n{text}\n```"
            for rel, text in sorted(files.items()))
        user = (
            f"TASK\n{plan.project_name}: make the project import cleanly.\n\n"
            f"The project fails to import with this error:\n\n{err}\n\n"
            "CURRENT FILES\n" + listing + "\n\n"
            "Emit search/replace blocks that fix the import error: rename or "
            "add the missing names so every import above resolves. Keep the "
            "blocks small and do not rewrite whole files.")
        reply = client.chat(
            [{"role": "system", "content": ANSWER_SYSTEM},
             {"role": "user", "content": user}],
            max_tokens=getattr(cfg, "max_tokens_answer", 4000),
            stream=verbose, disable_thinking=True)
        if not parse_edits(reply):
            break
        results = apply_edits(idx.root, parse_edits(reply), backup=cfg.backup,
                              dry_run=False, cfg=cfg, backup_dir=backup_dir)
        ok = [r for r in results if r.ok]
        if not ok:
            break
        fixed += len(ok)
        from . import index as index_mod
        for r in ok:
            idx = index_mod.reindex_file(idx, r.file, cfg)
    return fixed


def _py_sources(root: Path, rels) -> dict[str, str]:
    """Read every Python file in `rels` that exists, as rel -> text."""
    out: dict[str, str] = {}
    for rel in rels:
        rel = str(rel)
        if not rel.endswith(".py"):
            continue
        p = root / rel
        if p.is_file():
            try:
                out[rel] = fileio.read_text(p)
            except OSError:
                pass
    return out


def fix_api_mismatches(client, cfg, idx: "Index", plan: Plan, backup_dir,
                       verbose: bool = True, rounds: int = 2) -> int:
    """Re-ask the model to repair cross-file call/signature mismatches.

    Each file is implemented independently, so `core.py` can call `Level()`
    while `level.py` defines `Level.__init__(self, level_data)`. An import
    check passes anyway (`import main` does not run the call), so this
    static pass finds the arity mismatches and asks for search/replace fixes.
    Returns the number of successful fix blocks applied."""
    from .agent import ANSWER_SYSTEM

    fixed = 0
    for _ in range(rounds):
        sources = _py_sources(idx.root, [f.path for f in plan.files])
        problems = apicheck.find_mismatches(sources)
        if not problems:
            break
        listing = "\n".join(f"- {p}" for p in problems[:40])
        files = "\n\n".join(
            f"### {rel}\n```python\n{text}\n```"
            for rel, text in sorted(sources.items()))
        user = (
            f"TASK\n{plan.project_name}: make every call match its definition.\n\n"
            "These calls pass the wrong number of arguments and would crash at "
            "runtime:\n" + listing + "\n\n"
            "CURRENT FILES\n" + files + "\n\n"
            "Emit search/replace blocks that fix each listed mismatch: either "
            "pass the missing argument(s) at the call site, or give the "
            "parameter a default in the definition. Keep the blocks small; do "
            "not rewrite whole files.")
        reply = client.chat(
            [{"role": "system", "content": ANSWER_SYSTEM},
             {"role": "user", "content": user}],
            max_tokens=getattr(cfg, "max_tokens_answer", 4000),
            stream=verbose, disable_thinking=True)
        if not parse_edits(reply):
            break
        results = apply_edits(idx.root, parse_edits(reply), backup=cfg.backup,
                              dry_run=False, cfg=cfg, backup_dir=backup_dir)
        ok = [r for r in results if r.ok]
        if not ok:
            break
        fixed += len(ok)
        from . import index as index_mod
        for r in ok:
            idx = index_mod.reindex_file(idx, r.file, cfg)
    return fixed


# --------------------------------------------------------------------------
# Runtime verification: actually run the thing, then repair
# --------------------------------------------------------------------------
# A headless box cannot open a window; those errors say nothing about the
# generated code, so they must not be "repaired" or counted as failures.
_ENV_ISSUE_HINTS = (
    "no available video device", "video system not initialized",
    "no display name", "cannot open display", "could not connect to display",
    "unable to open display", "tclerror: no display", "no x11 display",
    "xdg_runtime_dir is not set",
)


def _environment_issue(output: str) -> bool:
    """True when a run failure is the environment (headless display), not the
    generated code."""
    low = (output or "").lower()
    return any(h in low for h in _ENV_ISSUE_HINTS)


def smoke_run(root: Path, entry: str = "main.py",
              timeout: int = 20) -> tuple[bool, str]:
    """Run the project's entry point once. Returns (passed, output).

    A crash (non-zero exit before the timeout) is a failure with its traceback.
    A process still running at the timeout is a SUCCESS: games, servers, and
    UIs do not exit on their own, and having started is what the check is for.
    The process group is killed at the timeout, so the game/window actually
    closes instead of lingering after the check.
    """
    from . import proc as proc_mod
    rc, out, stopped, _ = proc_mod.run(
        [sys.executable, entry], cwd=root, timeout=timeout,
        max_output=200_000)
    if stopped == "timeout":
        return True, f"(still running after {timeout}s -- treated as started)"
    if stopped == "error":
        return False, f"(could not run {entry}: {out})"
    return rc == 0, out.strip()


def _fix_from_output(client, cfg, idx: "Index", py_rels, project_name: str,
                     backup_dir, label: str, output: str,
                     verbose: bool = True) -> int:
    """Ask the model to repair one failed check, given its output. Returns the
    number of applied fix blocks."""
    from .agent import ANSWER_SYSTEM

    sources = _py_sources(idx.root, py_rels)
    if not sources:
        return 0
    files = "\n\n".join(
        f"### {rel}\n```python\n{text}\n```"
        for rel, text in sorted(sources.items()))
    user = (
        f"TASK\n{project_name}: make this check pass.\n\n"
        f"The check failed ({label}) with this output:\n\n{output[:5000]}\n\n"
        "CURRENT FILES\n" + files + "\n\n"
        "Emit search/replace blocks that fix the failure. Keep the blocks "
        "small; do not rewrite whole files.")
    reply = client.chat(
        [{"role": "system", "content": ANSWER_SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=getattr(cfg, "max_tokens_answer", 4000),
        stream=verbose, disable_thinking=True)
    if not parse_edits(reply):
        return 0
    results = apply_edits(idx.root, parse_edits(reply), backup=cfg.backup,
                          dry_run=False, cfg=cfg, backup_dir=backup_dir)
    ok = [r for r in results if r.ok]
    if not ok:
        return 0
    from . import index as index_mod
    for r in ok:
        idx = index_mod.reindex_file(idx, r.file, cfg)
    return len(ok)


def verify_and_fix(client, cfg, idx: "Index", py_rels, project_name: str,
                   backup_dir, *, explicit_cmd: str = "", rounds: int = 2,
                   smoke: bool = True, smoke_timeout: int = 20,
                   verbose: bool = True) -> tuple[bool, str]:
    """Run the explicit --verify and/or an automatic smoke run, feeding any
    failure back to the model and re-running, up to `rounds` repairs.

    Returns (passed, final_output). Missing dependencies are reported rather
    than repaired by editing source; environment (headless display) failures
    are treated as "could not verify" rather than code failures."""
    from . import loop

    def checks():
        if explicit_cmd:
            ok, out = loop.run_verify(idx.root, explicit_cmd)
            yield f"$ {explicit_cmd}", ok, out
        if smoke and (idx.root / "main.py").is_file():
            ok, out = smoke_run(idx.root, "main.py", smoke_timeout)
            yield "smoke run: python main.py (timeout = started)", ok, out

    last = ""
    for attempt in range(rounds + 1):
        failures = [(label, out) for label, ok, out in checks() if not ok]
        if not failures:
            return True, ""
        # Split out failures that are not the code's fault: a headless display
        # or an absent package cannot be repaired by editing source.
        env = [(l, o) for l, o in failures if _environment_issue(o)]
        deps = [(l, o) for l, o in failures if loop.missing_modules(o)]
        code_fails = [(l, o) for l, o in failures
                      if not _environment_issue(o) and not loop.missing_modules(o)]
        if not code_fails:
            if deps:
                return False, deps[0][1]           # report as a dependency
            return True, (f"could not verify in this environment: "
                          f"{env[0][1][:200]}")
        label, last = code_fails[0]
        if attempt >= rounds:
            return False, last
        if verbose:
            print(f"\n  [verify failed ({label}); repairing, "
                  f"{attempt + 1}/{rounds}]", file=sys.stderr)
        if not _fix_from_output(client, cfg, idx, py_rels, project_name,
                                backup_dir, label, last, verbose):
            return False, last
    return False, last


# --------------------------------------------------------------------------
# The whole flow, for the CLI's `new` command
# --------------------------------------------------------------------------
def run_new(root: Path, cfg, client, spec: Spec, *, apply: bool = False,
            review: bool = False, plan_only: bool = False, force: bool = False,
            verbose: bool = True, complex: bool = False, verify: str = "",
            repair_rounds: int | None = None, max_tasks: int | None = None,
            max_minutes: int | None = None,
            keep_going: bool = False, auto_install: bool = False) -> int:
    """Plan a new project from `spec`, lay down its skeleton, and (unless
    asked to stop earlier) implement every source file. Returns a process exit
    code; all progress goes to stderr so stdout stays clean.

    Stage gates, cheapest first:
      --plan-only  stop after stage 1+2: a tree of headers plus a saved index,
                   ready for `pedacito resume` or ordinary `ask --apply` runs.
      default      dry run: print the plan and a preview of the skeleton,
                   nothing is written to disk.
      --apply      write the generated files, then build each one through the
                   normal edit path (backups taken first).
      --review     build, then open the side-by-side reviewer with the result
                   (Discard restores the pre-build snapshot).
      --complex    like --apply, but implementation is handed to the autonomous
                   outer loop (`pedacito auto`): one task per source file, each
                   in a fresh context window, with an optional --verify command
                   feeding failures back for repair. The plan becomes a
                   resumable checklist (auto_plan_file) instead of one
                   uninterruptible build pass.

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

    # Write modes. The default (no --apply/--review/--plan-only/--complex) is
    # a true dry run: show the plan and what the skeleton WOULD contain, write
    # nothing. --plan-only writes just the header skeleton; --apply/--review
    # write the skeleton and then build the files (review holds the result for
    # sign-off); --complex writes the skeleton and hands implementation to the
    # autonomous loop. The old code always wrote the skeleton AND the build
    # passes, so a bare `new` left a full project on disk and `--apply` simply
    # re-ran the same work.
    wants_skeleton = plan_only or apply or review
    stubs = {}
    if wants_skeleton:
        # Fix the API contract once, in a single context, before writing the
        # skeleton -- this is what keeps cross-file imports consistent for a
        # small model that could not invent a matching API per file.
        stubs = build_interface(plan, client, cfg, verbose=verbose)
    created = apply_skeleton(root, plan, dry_run=not wants_skeleton,
                             verbose=verbose, stubs=stubs)
    if not wants_skeleton:
        if created:
            preview = ", ".join(created[:12])
            if len(created) > 12:
                preview += f", ... (+{len(created) - 12})"
            print(f"  [dry run] {len(created)} skeleton file(s) would be "
                  f"created: {preview}", file=sys.stderr)
        if complex:
            print(f"  [dry run] --complex would hand these {len(plan_tasks(plan))} "
                  f"task(s) to the autonomous loop:", file=sys.stderr)
            for t in plan_tasks(plan):
                print(f"      - {t}", file=sys.stderr)
        print("  Nothing was written. Re-run with --apply to build the "
              "project (or --complex --apply for the autonomous loop), "
              "or --plan-only to write just the header skeleton.",
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

    if complex:
        return _run_complex(root, cfg, client, idx, plan, spec,
                            verify=verify, repair_rounds=repair_rounds,
                            max_tasks=max_tasks, max_minutes=max_minutes,
                            keep_going=keep_going,
                            verbose=verbose, auto_install=auto_install)

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

    # Files are generated independently, so cross-file API mismatches (import
    # `parse_markdown` but define `parse`) are common. Ask the model to repair
    # them until the entry point imports cleanly.
    n_fixed = fix_import_errors(client, cfg, idx, plan, backup_dir,
                                verbose=verbose)
    if n_fixed and verbose:
        print(f"  fixed {n_fixed} import issue(s)", file=sys.stderr)

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


def _run_complex(root: Path, cfg, client, idx, plan: Plan, spec: Spec, *,
                 verify: str = "", repair_rounds: int | None = None,
                 max_tasks: int | None = None,
                 max_minutes: int | None = None,
                 keep_going: bool = False,
                 verbose: bool = True, auto_install: bool = False) -> int:
    """Complex mode: the framework (skeleton + interface stubs + index) is in
    place, so hand implementation to the autonomous outer loop.

    The plan becomes a resumable checklist (cfg.auto_plan_file, default
    PLAN.md) with one task per source file; the loop implements each in a
    fresh context window, optionally re-running a `verify` command and feeding
    failures back for repair. Returns the loop's exit code."""
    from . import loop
    from . import workspace

    rr = cfg.auto_repair_rounds if repair_rounds is None else repair_rounds
    mt = cfg.auto_max_tasks if max_tasks is None else max_tasks
    mm = (getattr(cfg, "auto_max_minutes", 0) if max_minutes is None
          else max_minutes)
    vf = verify if verify else getattr(cfg, "auto_verify", "")

    plan_file = root / getattr(cfg, "auto_plan_file", "PLAN.md")
    split_lines = int(getattr(cfg, "auto_split_lines", 0) or 0)
    titles = plan_tasks(plan, idx=idx, max_lines=split_lines)
    loop.write_plan(plan_file, plan.project_name or spec.title, titles)

    print(f"\n--- auto: implementing {len(titles)} task(s) in fresh contexts "
          f"(plan: {plan_file}) ---", file=sys.stderr)
    workspace.log_event(root, f"complex: {len(titles)} task(s) -> {plan_file.name}")
    code = loop.run_loop(
        root, cfg, client, idx, plan_file,
        apply=True,
        verify=vf,
        repair_rounds=rr,
        max_tasks=mt,
        max_minutes=mm,
        stop_on_fail=not keep_going,
        answer_system=AUTO_IMPLEMENT_SYSTEM,
        auto_install=auto_install,
    )

    # Cross-file API repair. Every task is implemented in a FRESH context, so
    # the model can call `parse_markdown` in one file and define `parse` in
    # another. The simple `new` build closes that gap with fix_import_errors;
    # without the same pass here, complex mode can mark every task done and
    # still ship a tree that cannot import. This runs after the loop, when the
    # whole project exists. A missing third-party package is handled first and
    # separately: editing source cannot add a dependency, so it must never be
    # sent through the code-repair loop.
    has_py = any(f.path.endswith(".py") for f in plan.files)
    if has_py:
        when = workspace.stamp()
        fix_backup = workspace.new_backup_dir(root, when) if cfg.backup else None
        err = _first_import_error(root, plan)
        missing = loop.missing_modules(err or "")
        if missing:
            dists = ", ".join(loop.pip_name(m) for m in missing)
            if auto_install:
                print(f"  [import check: installing missing dependencies: "
                      f"{dists}]", file=sys.stderr)
                workspace.log_event(root, f"import check: installing {dists}")
                ok, _out = loop.install_modules(missing, root)
                workspace.log_event(root, f"pip install {'ok' if ok else 'FAILED'}: "
                                          f"{dists}", "INFO" if ok else "ERROR")
                err = _first_import_error(root, plan)
                missing = loop.missing_modules(err or "")
            else:
                print(f"  [import check blocked by missing dependencies: "
                      f"{dists}. Install with `pip install {dists}`, or rerun "
                      f"with --auto-install.]", file=sys.stderr)
                workspace.log_event(root, f"import check blocked by missing "
                                          f"deps: {dists}", "ERROR")
        if err and not missing:
            n_fixed = fix_import_errors(client, cfg, idx, plan, fix_backup,
                                        verbose=verbose)
            if n_fixed and verbose:
                print(f"  fixed {n_fixed} import issue(s)", file=sys.stderr)
            err = _first_import_error(root, plan)
            missing = loop.missing_modules(err or "")
        if err:
            if missing:
                dists = ", ".join(loop.pip_name(m) for m in missing)
                print(f"\n  [warning] the project needs dependencies that are "
                      f"not installed: {dists}", file=sys.stderr)
            else:
                print("\n  [warning] the project still does not import "
                      "cleanly:\n"
                      + "\n".join(f"    {ln}" for ln in err.splitlines()[:8]),
                      file=sys.stderr)
            workspace.log_event(root, f"import check failed: {err[:200]}",
                                "ERROR")
            if code == 0:
                code = 1
        else:
            note = ("imports cleanly (run behaviour was NOT exercised"
                    + ("" if vf else "; pass --verify CMD to gate on it") + ")")
            print(f"  import check: {note}", file=sys.stderr)
            workspace.log_event(root, f"import check: {note}")
            # An import does not execute a call, so `Level()` against
            # `Level.__init__(self, level_data)` passes the check above and
            # still crashes on first use. Check cross-file call arity
            # statically and repair it the same way.
            n_api = fix_api_mismatches(client, cfg, idx, plan, fix_backup,
                                       verbose=verbose)
            if n_api and verbose:
                print(f"  fixed {n_api} call/signature mismatch(es)",
                      file=sys.stderr)
            problems = apicheck.find_mismatches(
                _py_sources(root, [f.path for f in plan.files]))
            if problems:
                print("\n  [warning] call/signature mismatches remain:\n"
                      + "\n".join(f"    {p}" for p in problems[:10]),
                      file=sys.stderr)
                workspace.log_event(root,
                                    f"api mismatches remain: {problems[0]}",
                                    "ERROR")
                if code == 0:
                    code = 1
            else:
                print("  api check: call signatures consistent",
                      file=sys.stderr)
                workspace.log_event(root, "api check: consistent")

            # Runtime verification: static checks never EXECUTE the program, so
            # a crash on first use can still slip through. Run the explicit
            # --verify from the config, else smoke-run main.py (a process still
            # running at the timeout counts as started), then feed any failure
            # back to the model before the run is declared complete.
            rounds = int(getattr(cfg, "auto_verify_rounds", 2) or 0)
            do_smoke = bool(getattr(cfg, "auto_smoke", True))
            smoke_timeout = int(getattr(cfg, "auto_smoke_timeout", 20) or 20)
            if vf or do_smoke:
                v_ok, v_out = verify_and_fix(
                    client, cfg, idx, [f.path for f in plan.files],
                    plan.project_name, fix_backup, explicit_cmd=vf,
                    rounds=rounds, smoke=do_smoke,
                    smoke_timeout=smoke_timeout, verbose=verbose)
                if v_ok:
                    vnote = "verify: passed" + (f" ({v_out})" if v_out else "")
                    print(f"  {vnote}", file=sys.stderr)
                    workspace.log_event(root, vnote)
                else:
                    print("\n  [warning] runtime verification failed after "
                          "repair:\n"
                          + "\n".join(f"    {ln}" for ln in v_out.splitlines()[:10]),
                          file=sys.stderr)
                    workspace.log_event(root, f"verify failed: {v_out[:200]}",
                                        "ERROR")
                    if code == 0:
                        code = 1

    if cfg.log_sessions:
        p = workspace.write_session(root, f"new: {plan.project_name}",
                                    [], render_plan(plan), [], None,
                                    workspace.stamp())
        print(f"[session: {p.relative_to(root)}]", file=sys.stderr)
    return code


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
