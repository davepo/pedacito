"""Autofix ladder (§3.4): run the project's own formatters/linters on an
independent shadow copy and keep only a proven net-positive candidate.

    from . import autofix

    result = autofix.run_autofix(root, ["src/app.py"], cfg=cfg, env=resolved)
    if result.verified:
        ...    # result.kept are the files the fixer improved and we committed

Before the model repair rounds, `loop._post_edit_tests` calls this (opt-in via
``Config.autofix_enabled``) so a deterministic fix -- ``ruff check --fix``,
installed Biome or ESLint ``--fix``, ``gofmt``, ``cargo fmt`` then
``cargo clippy --fix`` -- can resolve the finding without spending model
tokens.

Safety contract (never "green" when unproven):

* every command runs through the bounded `proc` runner with `shell=False`
  (argument vectors only; Windows ``.cmd`` is never implicitly shell-executed)
  and through the OS sandbox, with the shadow copy as the writable root;
* a candidate is only ever run on an independent, no-link shadow copy
  (:func:`pedacito.workspace_overlay.copy_tree_no_link`) -- never on the
  real tree and never through the dependency-linking ``exec_root`` mirror;
* a candidate is retained ONLY when the baseline finding identity set strictly
  shrinks, no new findings appear, the impacted tests actually run and stay
  green relative to a green baseline, and the changed files are still confined
  to the requested set;
* missing tools, no tests, a red baseline, timeouts, out-of-scope
  modifications, red tests or a preimage hash mismatch all discard the
  candidate (``verified=False``);
* the commit re-verifies preimage hashes just before an atomic, rollback-led
  write, and accepted/rejected metadata is emitted without any raw tool
  output (which may carry secrets) on stdout.

The module is pure stdlib plus the existing core dependencies; no downloads,
no package scripts, no new dependency.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import env_resolve, fileio, proc, sandbox, security
from .workspace_overlay import copy_tree_no_link

__all__ = ["AutofixResult", "emit_autofix_result", "run_autofix"]


@dataclass(frozen=True)
class AutofixResult:
    """The outcome of one autofix ladder run (§3.4).

    All fields are project-relative ``Path.as_posix()`` strings (paths in JSON)
    or finding-identity strings, never raw source or tool output.
    """

    kept: tuple[str, ...]        # files committed from the retained candidate
    rejected: tuple[str, ...]    # files not committed (or all, when none kept)
    before: tuple[str, ...]      # baseline finding identities for the candidate
    after: tuple[str, ...]       # finding identities remaining after the fix
    verified: bool               # True only when a candidate was proven and kept
    reason: str                  # bounded, secret-free explanation


# A fixer/report command must not run unbounded; this is a ceiling on the wall
# time of the whole ladder (each subprocess still has its own timeout).
_DEFAULT_DEADLINE_S = 120.0
# Bounded subprocess output: a formatter that dumps a whole diff must not blow
# memory, and raw output is never surfaced (it may carry secrets).
_MAX_OUTPUT = 32_000


# --------------------------------------------------------------------------- #
# Adapters                                                                     #
# --------------------------------------------------------------------------- #

_JS_EXTS = frozenset({".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"})

# A finding line as `path:12:3: code message` (ruff/eslint/clippy concise) or a
# plain message. Positions are erased but the path is kept, so two findings
# with the same text in different files stay distinct while line shifts do not.
_LINTER_HEAD_RE = re.compile(r"^(\S+?):\d+(?::\d+)?:\s*(.*)$", re.DOTALL)
_LINE_WORD_RE = re.compile(r"\bline \d+\b")
# `cargo fmt -- --check` prints `Diff in <path> at line N:` per unformatted file.
_CARGO_FMT_DIFF_RE = re.compile(r"(?m)^\s*Diff in (.+?) at line \d+")


def _linter_identity(line: str) -> str:
    line = (line or "").strip()
    m = _LINTER_HEAD_RE.match(line)
    if m:
        path = m.group(1)
        msg = _LINE_WORD_RE.sub("line N", m.group(2)).strip()
        return f"{path}:{msg}" if msg else path
    return _LINE_WORD_RE.sub("line N", line)


def _parse_linter(out: str) -> frozenset[str]:
    return frozenset(
        _linter_identity(ln) for ln in (out or "").splitlines() if ln.strip())


def _parse_paths(out: str) -> frozenset[str]:
    return frozenset(ln.strip() for ln in (out or "").splitlines() if ln.strip())


def _parse_cargo_fmt(out: str) -> frozenset[str]:
    return frozenset(m.group(1).strip()
                     for m in _CARGO_FMT_DIFF_RE.finditer(out or ""))


@dataclass(frozen=True)
class _Adapter:
    label: str
    tool: str                 # executable resolved via env_resolve
    exts: frozenset[str]      # file suffixes this adapter can fix
    fix_argv: tuple[str, ...]     # argv prefix; changed files appended when
    report_argv: tuple[str, ...]  # files_as_args is True
    parse: str = "linter"     # "linter" | "paths" | "cargo-fmt"
    files_as_args: bool = True
    manifest: str = ""        # a root file that must exist for applicability


_ADAPTERS: dict[str, _Adapter] = {
    "ruff": _Adapter(
        "ruff", "ruff", frozenset({".py", ".pyi"}),
        ("ruff", "check", "--fix"),
        ("ruff", "check", "--quiet", "--output-format", "concise"),
        parse="linter"),
    "biome": _Adapter(
        "biome", "biome", _JS_EXTS,
        ("biome", "check", "--write"),
        ("biome", "check"),
        parse="linter"),
    "eslint": _Adapter(
        "eslint", "eslint", _JS_EXTS,
        ("eslint", "--fix"),
        ("eslint",),
        parse="linter"),
    "gofmt": _Adapter(
        "gofmt", "gofmt", frozenset({".go"}),
        ("gofmt", "-w"),
        ("gofmt", "-l"),
        parse="paths"),
    "cargo-fmt": _Adapter(
        "cargo fmt", "cargo", frozenset({".rs"}),
        ("cargo", "fmt"),
        ("cargo", "fmt", "--", "--check"),
        parse="cargo-fmt", files_as_args=False, manifest="Cargo.toml"),
    "cargo-clippy": _Adapter(
        "cargo clippy", "cargo", frozenset({".rs"}),
        ("cargo", "clippy", "--fix", "--allow-dirty", "--allow-staged",
         "--allow-no-vcs"),
        ("cargo", "clippy", "--message-format", "short"),
        parse="linter", files_as_args=False, manifest="Cargo.toml"),
}

_PARSERS = {
    "linter": _parse_linter,
    "paths": _parse_paths,
    "cargo-fmt": _parse_cargo_fmt,
}


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #

def _normalise_files(files) -> list[str]:
    out: list[str] = []
    for f in files:
        rel = fileio.normalise_rel(str(f or ""))
        if rel and rel not in out:
            out.append(rel)
    return out


def _resolve_deadline(deadline) -> float:
    return float(deadline) if deadline else time.monotonic() + _DEFAULT_DEADLINE_S


def _remaining(deadline: float) -> float:
    return max(0.0, float(deadline) - time.monotonic())


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_hashes(root: Path, files: list[str]) -> dict[str, str]:
    """{rel: sha256} of the current on-disk originals (the preimages)."""
    out: dict[str, str] = {}
    for rel in files:
        try:
            path = security.resolve_safe_path(root, rel)
        except (security.SecurityError, OSError, ValueError):
            continue
        try:
            if path.is_file():
                out[rel] = _sha(path)
        except OSError:
            continue
    return out


def _run(argv, *, cwd: Path, cfg, child_env: dict, bin_dirs, deadline: float):
    """Run a command sandboxed, as an argv with shell=False. Returns
    ``(rc, out, stopped)``; ``stopped`` is '' on a normal exit, else one of
    'error' (could not start, incl. sandbox refusal), 'timeout' or 'cancelled'."""
    timeout = _remaining(deadline)
    if timeout <= 0:
        return None, "deadline passed", "timeout"
    try:
        wrapped, _box = sandbox.wrap(list(argv), cfg, cwd)
    except sandbox.SandboxRefused as exc:
        return None, str(exc), "error"
    rc, out, stopped, _secs = proc.run(
        wrapped, cwd=str(cwd), timeout=timeout, idle_timeout=0,
        max_output=_MAX_OUTPUT, env=child_env,
        windows_sandbox=sandbox.windows_sandbox_active(cfg),
        writable_dirs=(cwd,),
        bin_dirs=bin_dirs)
    return rc, out, stopped


def _build_argv(prefix: tuple[str, ...], adapter: _Adapter,
                files: list[str]) -> list[str]:
    argv = list(prefix)
    if adapter.files_as_args:
        argv.extend(files)
    return argv


def _collect(adapter: _Adapter, root: Path, files: list[str], *, cfg, child_env,
             bin_dirs, deadline: float) -> frozenset[str] | None:
    """The finding identity set for `adapter` on `root`, or None when the
    report could not run (error/timeout) -- never a fabricated 'clean'."""
    argv = _build_argv(adapter.report_argv, adapter, files)
    _rc, out, stopped = _run(argv, cwd=root, cfg=cfg, child_env=child_env,
                             bin_dirs=bin_dirs, deadline=deadline)
    if stopped:
        return None
    return _PARSERS[adapter.parse](out)


def _candidate_adapters(root: Path, files: list[str], env, child_env: dict
                        ) -> list[_Adapter]:
    """The ladder in contract order, filtered to tools that are installed and
    files they apply to. Biome and ESLint share one slot: Biome wins."""
    exts = {Path(f).suffix.lower() for f in files}
    tools = env_resolve.find_executables(
        env, ("ruff", "biome", "eslint", "gofmt", "cargo"), env=child_env)
    out: list[_Adapter] = []
    if tools.get("ruff") and exts & _ADAPTERS["ruff"].exts:
        out.append(_ADAPTERS["ruff"])
    if exts & _JS_EXTS:
        if tools.get("biome"):
            out.append(_ADAPTERS["biome"])
        elif tools.get("eslint"):
            out.append(_ADAPTERS["eslint"])
    if tools.get("gofmt") and exts & _ADAPTERS["gofmt"].exts:
        out.append(_ADAPTERS["gofmt"])
    if tools.get("cargo") and exts & _ADAPTERS["cargo-fmt"].exts:
        for name in ("cargo-fmt", "cargo-clippy"):
            ad = _ADAPTERS[name]
            if not ad.manifest or (root / ad.manifest).is_file():
                out.append(ad)
    return out


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

def _test_command(root: Path, cfg) -> str:
    from . import testsel
    try:
        return (testsel.detect_test_command(root, cfg) or "").strip()
    except Exception:
        return ""


def _tests_green(root: Path, cfg, command: str, deadline: float) -> str | True:
    """True when the project's tests actually run and pass on `root`, else a
    bounded reason string. A test command that cannot be run is never green."""
    from . import testsel
    if not command:
        return "no test command detected (impacted tests cannot run)"
    if _remaining(deadline) <= 0:
        return "deadline passed before tests could run"
    try:
        passed, _out = testsel.run_command(root, cfg, command,
                                           timeout=int(_remaining(deadline)))
    except Exception as exc:
        return f"tests could not run: {type(exc).__name__}"
    if not passed:
        return "the impacted tests are red"
    return True


# --------------------------------------------------------------------------- #
# Shadow copy, scope and commit                                                #
# --------------------------------------------------------------------------- #

def _make_shadow(root: Path) -> Path:
    return Path(tempfile.mkdtemp(prefix="pedacito-autofix-"))


def _changed_files(root: Path, shadow: Path) -> list[str]:
    """Project-relative paths whose shadow bytes differ from the original."""
    from . import checkpoints
    changed: list[str] = []
    for rel, src in checkpoints.iter_workspace_files(shadow):
        if security.is_sensitive_path(rel):
            continue
        try:
            orig = security.resolve_safe_path(root, rel)
        except (security.SecurityError, OSError, ValueError):
            continue
        try:
            if orig.is_file() and _sha(src) != _sha(orig):
                changed.append(rel)
        except OSError:
            continue
    return changed


def _commit(root: Path, rels: list[str], shadow: Path, cfg) -> tuple[bool, str]:
    """Atomically write the shadow bytes over the originals, with a rollback
    ledger and a backup copy of each original. Re-validates nothing itself: the
    caller re-checks preimage hashes immediately before calling."""
    from . import workspace
    backup_dir = workspace.new_backup_dir(root, unique=True) \
        if getattr(cfg, "backup", True) else None
    written: list[tuple[str, bytes | None]] = []
    for rel in rels:
        try:
            target = security.resolve_safe_path(root, rel)
            shadow_path = security.resolve_safe_path(shadow, rel)
        except (security.SecurityError, OSError, ValueError) as exc:
            return False, f"autofix commit refused a path: {exc}"
        try:
            old = target.read_bytes() if target.is_file() else None
            new = shadow_path.read_bytes()
        except OSError as exc:
            return False, f"autofix commit could not read {rel}: {exc}"
        if old == new:
            continue
        if backup_dir is not None:
            try:
                if old is None:
                    fileio.note_created(backup_dir, rel)
                else:
                    fileio.backup_copy(target, backup_dir / rel)
            except OSError:
                pass
        try:
            fileio.write_bytes(target, new)
        except OSError as exc:
            for r, data in reversed(written):
                try:
                    if data is None:
                        (root / r).unlink(missing_ok=True)
                    else:
                        fileio.write_bytes(root / r, data)
                except OSError:
                    pass
            return False, f"autofix commit failed on {rel}: {exc}"
        written.append((rel, old))
    return True, ""


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def _try_adapter(root: Path, files: list[str], adapter: _Adapter, *, cfg, env,
                 child_env: dict, deadline: float, before_hashes: dict[str, str]
                 ) -> AutofixResult:
    shadow = _make_shadow(root)
    try:
        try:
            copy_tree_no_link(root, shadow)
        except Exception as exc:
            return AutofixResult(
                (), tuple(files), (), (), False,
                f"{adapter.label}: could not build a shadow copy: {exc}")

        before = _collect(adapter, shadow, files, cfg=cfg, child_env=child_env,
                          bin_dirs=env.bin_dirs, deadline=deadline)
        if before is None:
            return AutofixResult((), tuple(files), (), (), False,
                                 f"{adapter.label}: no baseline findings could "
                                 "be collected")

        _rc, _out, stopped = _run(
            _build_argv(adapter.fix_argv, adapter, files),
            cwd=shadow, cfg=cfg, child_env=child_env, bin_dirs=env.bin_dirs,
            deadline=deadline)
        if stopped:
            return AutofixResult((), tuple(files), (), (), False,
                                 f"{adapter.label}: fix was stopped ({stopped})")

        changed = _changed_files(root, shadow)
        out_of_scope = sorted(rel for rel in changed
                              if rel not in set(files))
        if out_of_scope:
            return AutofixResult(
                (), tuple(files), (), (), False,
                f"{adapter.label}: modified out-of-scope file(s): "
                f"{', '.join(out_of_scope[:5])}")

        after = _collect(adapter, shadow, files, cfg=cfg, child_env=child_env,
                         bin_dirs=env.bin_dirs, deadline=deadline)
        if after is None:
            return AutofixResult((), tuple(files), (), (), False,
                                 f"{adapter.label}: findings could not be "
                                 "re-collected after the fix")

        if not (after <= before and len(after) < len(before)):
            return AutofixResult((), tuple(files), tuple(sorted(before)),
                                 tuple(sorted(after)), False,
                                 f"{adapter.label}: no net improvement")

        in_scope = sorted(rel for rel in changed if rel in set(files))
        if not in_scope:
            return AutofixResult((), tuple(files), tuple(sorted(before)),
                                 tuple(sorted(after)), False,
                                 f"{adapter.label}: no in-scope files changed")

        # Preimage re-validation: the originals must be byte-identical to the
        # snapshot taken before any tool ran, or someone else wrote under us.
        for rel in in_scope:
            try:
                path = security.resolve_safe_path(root, rel)
            except (security.SecurityError, OSError, ValueError):
                return AutofixResult(
                    (), tuple(files), tuple(sorted(before)),
                    tuple(sorted(after)), False,
                    f"{adapter.label}: a path could not be re-validated: {rel}")
            try:
                if not path.is_file() or _sha(path) != before_hashes.get(rel):
                    return AutofixResult(
                        (), tuple(files), tuple(sorted(before)),
                        tuple(sorted(after)), False,
                        f"{adapter.label}: preimage hash mismatch on {rel}; "
                        "the file changed after it was copied")
            except OSError:
                return AutofixResult(
                    (), tuple(files), tuple(sorted(before)),
                    tuple(sorted(after)), False,
                    f"{adapter.label}: could not re-read {rel}")

        command = _test_command(root, cfg)
        if not command:
            return AutofixResult((), tuple(files), tuple(sorted(before)),
                                 tuple(sorted(after)), False,
                                 f"{adapter.label}: no test command detected")
        green = _tests_green(shadow, cfg, command, deadline)
        if green is not True:
            return AutofixResult((), tuple(files), tuple(sorted(before)),
                                 tuple(sorted(after)), False,
                                 f"{adapter.label}: {green}")

        ok, reason = _commit(root, in_scope, shadow, cfg)
        if not ok:
            return AutofixResult((), tuple(files), tuple(sorted(before)),
                                 tuple(sorted(after)), False, reason)

        kept = tuple(in_scope)
        rejected = tuple(f for f in files if f not in set(in_scope))
        return AutofixResult(
            kept, rejected, tuple(sorted(before)), tuple(sorted(after)), True,
            f"{adapter.label} fixed {len(kept)} file(s)")
    finally:
        shutil.rmtree(shadow, ignore_errors=True)


def run_autofix(root: Path, files, *, cfg, env=None, deadline=None
                ) -> AutofixResult:
    """Run the §3.4 autofix ladder and return its verdict.

    ``files`` are the project-relative paths the model's edit touched. ``env``
    is an optional :class:`~pedacito.env_resolve.ResolvedEnvironment` (resolved
    from ``root`` when omitted). ``deadline`` is a ``time.monotonic()`` stamp;
    when omitted a bounded default is used. A candidate is retained (and its
    files committed atomically) only when every guard below passes; otherwise
    the result is ``verified=False`` with a bounded, secret-free reason.
    """
    try:
        root = Path(root).expanduser().resolve()
    except (OSError, ValueError):
        root = Path(root)
    files = _normalise_files(files)
    if not files:
        return AutofixResult((), (), (), (), False, "no files to autofix")
    if not getattr(cfg, "autofix_enabled", False):
        return AutofixResult((), tuple(files), (), (), False,
                             "autofix is disabled (opt-in)")
    if env is None:
        env = env_resolve.resolve_environment(root)
    child_env = env_resolve.command_env(env, cfg)
    deadline = _resolve_deadline(deadline)

    # A green baseline is a precondition: without a test command, or with a red
    # one, no candidate can be "green relative to a green baseline".
    command = _test_command(root, cfg)
    if not command:
        emit_autofix_result(root, cfg, AutofixResult(
            (), tuple(files), (), (), False, "no test command detected"))
        return AutofixResult((), tuple(files), (), (), False,
                             "no test command detected")
    baseline = _tests_green(root, cfg, command, deadline)
    if baseline is not True:
        reason = f"baseline is not green: {baseline}"
        emit_autofix_result(root, cfg, AutofixResult(
            (), tuple(files), (), (), False, reason))
        return AutofixResult((), tuple(files), (), (), False, reason)

    adapters = _candidate_adapters(root, files, env, child_env)
    if not adapters:
        reason = "no autofix tool available for the changed files"
        emit_autofix_result(root, cfg, AutofixResult(
            (), tuple(files), (), (), False, reason))
        return AutofixResult((), tuple(files), (), (), False, reason)

    before_hashes = _snapshot_hashes(root, files)
    reasons: list[str] = []
    for adapter in adapters:
        if _remaining(deadline) <= 0:
            reasons.append("deadline passed")
            break
        result = _try_adapter(root, files, adapter, cfg=cfg, env=env,
                              child_env=child_env, deadline=deadline,
                              before_hashes=before_hashes)
        if result.verified:
            emit_autofix_result(root, cfg, result)
            return result
        reasons.append(result.reason or adapter.label)

    result = AutofixResult((), tuple(files), (), (), False,
                           "; ".join(reasons) or "no candidate retained")
    emit_autofix_result(root, cfg, result)
    return result


def emit_autofix_result(root: Path, cfg, result: AutofixResult) -> None:
    """Record accepted/rejected metadata, bounded and without raw tool output.

    Best effort: a failing event bus or log must never change the verdict.
    Only the ``reason`` (structured, secret-free) and the file/finding counts
    are stored -- never a command's stdout/stderr, which may carry secrets.
    """
    payload = {
        "kept": list(result.kept)[:50],
        "rejected": list(result.rejected)[:50],
        "verified": bool(result.verified),
        "reason": (result.reason or "")[:400],
        "before": len(result.before),
        "after": len(result.after),
    }
    try:
        from . import events
        events.get(root, cfg).emit("autofix", **payload)
    except Exception:
        pass
    try:
        from . import workspace
        workspace.log_event(
            root, f"autofix {'kept ' if result.verified else 'discarded: '}"
                  f"{payload['reason']}",
            "INFO" if result.verified else "WARN")
    except Exception:
        pass
