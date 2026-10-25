"""Static cross-file API consistency for Python.

The `--complex` build implements each file in a FRESH context window, so the
model can call `Level()` in one file while defining
`Level.__init__(self, level_data)` in another. `import main` does not execute
those calls, so the import check never sees it -- the run reports success and
the program crashes on first use.

This AST pass compares bare-name calls against the project's own function and
class definitions and reports argument-count mismatches before the tree is
handed back. It is deliberately conservative, because a false positive would
send a working project into the repair loop:

- only calls to a bare ``Name`` (not ``obj.method(...)``) are checked;
- only names defined exactly once in the project are checked, so a shadowed
  or ambiguous name is never reported;
- calls using ``*args`` / ``**kwargs`` are skipped;
- classes with no ``__init__`` are treated as zero-argument (object's).

It is a heuristic, not a type checker: it catches missing/extra positional
arguments, which is the common failure when independent file implementations
invent slightly different APIs. It does not resolve imports, so a name that
collides with a builtin or a local variable could in principle be misread --
rare in generated projects, and the repair prompt shows the exact definition
it matched so a wrong report is easy to see.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class _Sig:
    """A callable's arity: required positional-or-keyword names, and the most
    positional arguments it accepts (``None`` when it takes ``*args``)."""
    required: tuple[str, ...]
    max_pos: int | None


@dataclass(frozen=True)
class Mismatch:
    """One call whose argument count cannot match its definition."""
    rel: str
    line: int
    name: str
    message: str
    defined_in: str

    def __str__(self) -> str:
        return (f"{self.rel}:{self.line}: {self.name}(...) {self.message} "
                f"(defined at {self.defined_in})")


def _sig(args: ast.arguments, drop_self: bool) -> _Sig:
    """Arity for a function/``__init__`` ``arguments`` node. ``drop_self``
    removes the leading self/cls (a class constructor is called without it)."""
    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    defaults = list(args.defaults)
    if drop_self and positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
    n_defaults = min(len(defaults), len(positional))
    required = positional[:len(positional) - n_defaults] if n_defaults else positional
    max_pos = None if args.vararg else len(positional)
    return _Sig(tuple(a.arg for a in required), max_pos)


def _collect(sources: dict[str, str]) -> dict[str, list[tuple[_Sig, str]]]:
    """Map every top-level function/class name to (signature, "file:line")."""
    defs: dict[str, list[tuple[_Sig, str]]] = {}
    for rel, src in sources.items():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:                     # top level only
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault(node.name, []).append(
                    (_sig(node.args, False), f"{rel}:{node.lineno}"))
            elif isinstance(node, ast.ClassDef):
                init = next((n for n in node.body
                             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                             and n.name == "__init__"), None)
                sig = _Sig((), 0) if init is None else _sig(init.args, True)
                defs.setdefault(node.name, []).append((sig, f"{rel}:{node.lineno}"))
    return defs


def _arity_error(sig: _Sig, call: ast.Call) -> str | None:
    """Return a message when `call` cannot satisfy `sig`, else None."""
    if any(isinstance(a, ast.Starred) for a in call.args):
        return None                                # *args: unknown
    if any(k.arg is None for k in call.keywords):
        return None                                # **kwargs: unknown
    pos = len(call.args)
    kws = {k.arg for k in call.keywords if k.arg}
    missing = [n for i, n in enumerate(sig.required) if i >= pos and n not in kws]
    if missing:
        return "missing required argument(s): " + ", ".join(missing)
    if sig.max_pos is not None and pos > sig.max_pos:
        return (f"takes at most {sig.max_pos} positional argument(s) but "
                f"{pos} were given")
    return None


def find_mismatches(sources: dict[str, str]) -> list[Mismatch]:
    """All bare-name calls in `sources` (rel path -> text) whose argument
    count cannot match a uniquely-defined project function/class."""
    defs = _collect(sources)
    unique: dict[str, tuple[_Sig, str]] = {}
    for name, entries in defs.items():
        sigs = {sig for sig, _where in entries}
        if len(sigs) == 1:                         # unambiguous only
            unique[name] = (next(iter(sigs)), entries[0][1])

    out: list[Mismatch] = []
    for rel, src in sources.items():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)):
                continue
            entry = unique.get(node.func.id)
            if entry is None:
                continue
            sig, defined_in = entry
            msg = _arity_error(sig, node)
            if msg:
                out.append(Mismatch(rel, node.lineno, node.func.id, msg,
                                    defined_in))
    out.sort(key=lambda m: (m.rel, m.line))
    return out
