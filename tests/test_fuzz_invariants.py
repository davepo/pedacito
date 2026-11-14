"""Seeded generative property fuzzers for the deterministic edit/evidence core.

Zero third-party dependencies: every property runs a deterministic sequence of
``random.Random(seed)`` cases and asserts an invariant the harness relies on
rather than one hardcoded example.

Depth is configurable without touching the file:

    PEDACITO_FUZZ_ITERS=2500   # run 2,500 seeds per property (default 250)
    PEDACITO_FUZZ_SEED=41892   # run ONLY this seed, for instant debugging

A failing seed always reports the exact reproduction command. Seeds listed in
``PINNED_REGRESSION_SEEDS`` run on every default pass, whatever the depth.

  1. span arithmetic + ledger/evidence parity (agent.py, §3/§11): folded spans
     are sorted/disjoint/covering, replay deltas are disjoint and complete,
     and a delivered read observation credits EXACTLY the range the evidence
     store recorded -- the BUG-309 guard.
  2. edit parser, fuzzy locator and symbol splicing (edits.py/chunker.py,
     §25/§42): dry runs never touch disk, new Python syntax errors are refused,
     decorated methods keep their decorators and indent, numeric SEARCH bodies
     re-anchor (or refuse out of range).
  3. JS comment/regex/template blanker (webcheck.py, §23): byte length and
     every newline survive blanking, comment/string identifiers never become
     a false unreachable-call finding, and real calls inside ``${ ... }`` are
     still checked.
  4. (inode safety lives in tests/test_ratchet_inode_safety.py.)
"""
from __future__ import annotations

import ast
import hashlib
import os
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import fileio, verify, webcheck  # noqa: E402
from pedacito.agent import (  # noqa: E402
    Agent,
    EvidenceStore,
    _NUMBERED_LINE_RX,
    _delivered_end,
    _evict_duplicate_reads,
    _first_seen_spans,
    _fold_spans,
    _subtract_spans,
)
from pedacito.edits import Result, apply_edits, parse_edits  # noqa: E402
from pedacito.config import Config  # noqa: E402

DEFAULT_FUZZ_ITERS = 250

# Seeds that must always run, even at the default depth. Add the seed printed
# by a stress-run failure here after fixing the harness bug it exposed.
PINNED_REGRESSION_SEEDS: tuple[int, ...] = ()

# The suite-wide isolation fixture (`tests/conftest.py`) drops every ambient
# `PEDACITO_*` variable before each test. Capture the fuzz knobs at import
# time, which happens during collection and therefore before that fixture, so
# `PEDACITO_FUZZ_ITERS=2500 pytest ...` still reaches the loops.
_IMPORT_ENV = {
    "PEDACITO_FUZZ_ITERS": os.environ.get("PEDACITO_FUZZ_ITERS", ""),
    "PEDACITO_FUZZ_SEED": os.environ.get("PEDACITO_FUZZ_SEED", ""),
}


def _raw_env(name: str) -> str:
    raw = os.environ.get(name, "").strip()
    if raw:
        return raw
    return _IMPORT_ENV.get(name, "").strip()


def _env_seed(name: str) -> int | None:
    """An integer from `name` (current env or import-time capture), or None."""
    raw = _raw_env(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _fuzz_iters() -> int:
    """Iterations per property: PEDACITO_FUZZ_ITERS, else 250."""
    value = _env_seed("PEDACITO_FUZZ_ITERS")
    return value if value is not None and value > 0 else DEFAULT_FUZZ_ITERS


def _fuzz_seeds(offset: int = 0) -> list[int]:
    """Explicit seeds for one property loop.

    `PEDACITO_FUZZ_SEED` pins exactly one seed (offset is irrelevant: the RNG
    is constructed from the seed itself, so the failure is byte-reproducible).
    Otherwise the loop runs `offset .. offset + iters - 1`.
    """
    pinned = _env_seed("PEDACITO_FUZZ_SEED")
    if pinned is not None:
        return [pinned]
    return list(range(offset, offset + _fuzz_iters()))


def _seed_failure(seed: int) -> str:
    return (f"Fuzz failure at seed={seed} (reproduce with: "
            f"PEDACITO_FUZZ_SEED={seed} pytest tests/test_fuzz_invariants.py)")


def _fuzz_cases(case, offset: int = 0) -> None:
    """Run `case(seed)` over the configured seeds with a copyable failure."""
    seeds = _fuzz_seeds(offset)
    if _env_seed("PEDACITO_FUZZ_SEED") is None:
        seeds.extend(s for s in PINNED_REGRESSION_SEEDS if s not in seeds)
    for seed in seeds:
        try:
            case(seed)
        except AssertionError as exc:
            raise AssertionError(f"{_seed_failure(seed)}\n{exc}") from exc
        except Exception as exc:
            raise AssertionError(
                f"{_seed_failure(seed)}\n"
                f"{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _expand(spans) -> set[int]:
    out: set[int] = set()
    for a, b in spans:
        out.update(range(int(a), int(b) + 1))
    return out


def _numbered(src, start: int, end: int, preamble: str | None = None) -> str:
    body = [preamble] if preamble else []
    for n in range(start, end + 1):
        body.append(f"{n:>4} | {src[n - 1]}")
    return f"pkg/mod.py lines {start}-{end}:\n" + "\n".join(body)


def _clip_agent(max_tokens: int, chars_per_token: float):
    """The two Agent methods clipping needs, bound to a tiny stand-in."""
    obj = types.SimpleNamespace(
        cfg=types.SimpleNamespace(max_read_tokens=max_tokens),
        client=types.SimpleNamespace(chars_per_token=chars_per_token))
    obj._read_budget_chars = lambda: Agent._read_budget_chars(obj)
    obj._clip_read_to_token_budget = \
        lambda obs: Agent._clip_read_to_token_budget(obj, obs)
    return obj


def _dir_sha(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode("utf-8"))
            h.update(b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()


def _block(rel: str, search: str, replace: str) -> str:
    return (f"--- FILE: {rel}\n<<<<<<< SEARCH\n{search}\n"
            f"=======\n{replace}\n>>>>>>> REPLACE\n")


# ---------------------------------------------------------------------------
# 1. span arithmetic and ledger/evidence parity
# ---------------------------------------------------------------------------
def test_fold_and_subtract_spans_seeded():
    def case(seed):
        r = random.Random(seed)
        spans = []
        for _ in range(r.randrange(1, 16)):
            a = r.randrange(1, 60)
            b = a + r.randrange(0, 12)
            spans.append((a, b))
        # out-of-order, duplicate, overlapping and adjacent shapes
        r.shuffle(spans)
        spans.extend(r.choice(spans) for _ in range(r.randrange(0, 3)))

        folded = _fold_spans(spans)
        assert folded == sorted(folded), (seed, folded)
        for (a, b), (c, d) in zip(folded, folded[1:]):
            assert b + 1 < c, (seed, folded)       # strictly non-adjacent
            assert a <= b and c <= d, (seed, folded)
        assert _expand(folded) == _expand(spans), (seed, folded, spans)

        covered = []
        for _ in range(r.randrange(0, 5)):
            a = r.randrange(0, 60)
            covered.append((a, a + r.randrange(0, 15)))
        left = _subtract_spans(list(spans), covered)
        assert _expand(left) == _expand(spans) - _expand(covered), \
            (seed, spans, covered, left)
        for a, b in left:
            assert a <= b, (seed, left)
        assert not (_expand(left) & _expand(covered)), (seed, left, covered)

    _fuzz_cases(case)


def test_first_seen_spans_are_disjoint_and_complete_seeded():
    def case(seed):
        r = random.Random(seed)
        seq = []
        for _ in range(r.randrange(1, 14)):
            a = r.randrange(1, 50)
            seq.append((a, a + r.randrange(0, 12)))
        deltas = _first_seen_spans({"pkg/a.py": seq}).get("pkg/a.py", [])
        assert _expand(deltas) == _expand(seq), (seed, seq, deltas)

        # Replaying the reads by hand yields the same first-seen line set.
        first: set[int] = set()
        seen: set[int] = set()
        for a, b in seq:
            first |= set(range(a, b + 1)) - seen
            seen |= set(range(a, b + 1))
        assert _expand(deltas) == first, (seed, seq, deltas, first)

        # The emitted replay deltas themselves never repeat a line.
        emitted: set[int] = set()
        for a, b in deltas:
            got = set(range(a, b + 1))
            assert not (got & emitted), (seed, deltas)
            emitted |= got

    _fuzz_cases(case, 1000)


def test_eviction_never_drops_a_first_seen_line_seeded():
    def case(seed):
        r = random.Random(seed)
        nlines = r.randrange(10, 45)
        src = [f"line_{n}_{seed} = {n}" for n in range(1, nlines + 1)]
        spans = []
        for _ in range(r.randrange(2, 7)):
            a = r.randrange(1, nlines + 1)
            b = min(nlines, a + r.randrange(0, 14))
            spans.append((a, b))
        r.shuffle(spans)
        gathered, messages = [], []
        for a, b in spans:
            obs = _numbered(src, a, b)
            gathered.append(f"--- read pkg/mod.py\n{obs}")
            messages.append({"role": "assistant", "content": "{'read': 1}"})
            messages.append({"role": "user", "content": "RESULT:\n" + obs})

        new_items = _evict_duplicate_reads(messages, gathered,
                                           {"pkg/mod.py": spans})
        kept_lines: set[int] = set()
        for text in new_items:
            for ln in text.split("\n"):
                pm = _NUMBERED_LINE_RX.match(ln)
                if pm:
                    kept_lines.add(int(pm.group(1)))
        for msg in messages:
            if msg["role"] != "user":
                continue
            for ln in msg["content"].split("\n"):
                pm = _NUMBERED_LINE_RX.match(ln)
                if pm:
                    kept_lines.add(int(pm.group(1)))
        assert kept_lines == _expand(spans), (seed, spans, kept_lines)

    _fuzz_cases(case, 2000)


def test_clipped_read_ledger_matches_evidence_store_seeded():
    chars = "abcXYZ0123 你好 é—“”'\"{}()[]|"

    def case(seed):
        r = random.Random(seed)
        nlines = r.randrange(1, 45)
        src = ["".join(r.choice(chars) for _ in range(r.randrange(0, 28)))
               for _ in range(nlines)]
        # A source line can itself carry text that looks like the harness's
        # own truncation/context markers; only the trailing marker is a cut.
        if seed % 3 == 0:
            src[r.randrange(nlines)] = r.choice((
                "[truncated: context budget reached]",
                "[context: enclosing signature; module imports]",
                "[CLIPPED at line 3 due to token budget]",
            ))
        start = r.randrange(1, nlines + 1)
        end = r.randrange(start, nlines + 1)
        preamble = "[context: enclosing signature; module imports]" \
            if r.random() < 0.35 else None
        obs = _numbered(src, start, end, preamble)

        # token-budget clip (may be a no-op) ...
        agent = _clip_agent(r.randrange(1, 40), r.choice([2.0, 3.5, 4.0]))
        obs = agent._clip_read_to_token_budget(obs)
        # ... and the mid-line char-budget cut the gather loop applies next.
        if r.random() < 0.45:
            cut = r.randrange(1, len(obs)) if len(obs) > 1 else 0
            obs = obs[:cut] + "\n[truncated: context budget reached]"

        span = EvidenceStore().record_displayed(obs, seed)
        delivered = _delivered_end(obs, start)
        if span is None:
            assert delivered < start, (seed, start, delivered, obs)
            return
        assert (span.start, span.end) == (start, delivered), \
            (seed, (span.start, span.end), (start, delivered), obs)
        assert span.start <= span.end <= end, (seed, span, end)
        assert span.exact_text == "\n".join(src[span.start - 1:span.end]), \
            (seed, span.exact_text, obs)

    _fuzz_cases(case, 3000)


# ---------------------------------------------------------------------------
# 2. edits: dry-run purity, syntax gate, decorators, numeric re-anchor
# ---------------------------------------------------------------------------
_DECORATORS = ("@staticmethod", "@property", "@classmethod",
               "@decorator(1)", "@decorator('a', flag=True)")


def test_dry_run_never_mutates_disk_seeded(tmp_path):
    def case(seed):
        r = random.Random(seed)
        work = tmp_path / f"dry_{seed}"
        work.mkdir()
        if seed % 2:
            rel = f"case_{seed}.js"
            nl = "\r\n" if r.random() < 0.4 else "\n"
            value = r.randrange(100, 999)
            text = nl.join([
                "// notes: 你好, “smart quotes” and 'apostrophes'",
                f"const NAME = {value};",
                "function f() { return NAME; }",
            ]) + nl
            search = f"const NAME = {value};"
            replace = f"const NAME = {value + 1};"
        else:
            rel = f"case_{seed}.py"
            nl = "\r\n" if r.random() < 0.4 else "\n"
            value = r.randrange(100, 999)
            decs = r.sample(_DECORATORS, r.randrange(0, 3))
            text = nl.join([
                "# notes: 你好, “smart quotes”",
                *decs,
                "def f():",
                f"    return {value}",
            ]) + nl
            search = f"    return {value}"
            replace = f"    return {value + 1}"
        path = work / rel
        path.write_bytes(text.encode("utf-8"))
        before = _dir_sha(work)
        reply = _block(rel, search, replace)
        results = apply_edits(work, parse_edits(reply), dry_run=True)
        assert results and results[0].ok, (seed, results)
        assert _dir_sha(work) == before, seed   # dry run: no byte moved
        assert fileio.read_text(path) == text.replace("\r\n", "\n")
        assert results[0].before.replace("\r\n", "\n") == \
            text.replace("\r\n", "\n")

    _fuzz_cases(case, 4000)


def test_new_python_syntax_error_is_refused_seeded(tmp_path):
    bad_tails = ("return (1", "return 1 +", "def :", "return [1,",
                 "await", "class :", "return 1 if else 2")

    def case(seed):
        r = random.Random(seed)
        rel = f"broken_{seed}.py"
        value = r.randrange(1, 100)
        src = f"def f():\n    return {value}\n"
        path = tmp_path / rel
        path.write_text(src, encoding="utf-8")
        bad_tail = r.choice(bad_tails)
        reply = _block(rel, f"    return {value}", f"    {bad_tail}")
        results = apply_edits(tmp_path, parse_edits(reply), backup=False)
        assert results and not results[0].ok, (seed, bad_tail, results)
        assert ("syntax error" in results[0].message
                or "parse error" in results[0].message), \
            (seed, bad_tail, results[0].message)
        assert path.read_text(encoding="utf-8") == src, seed

        # check_syntax (the in-process gate) flags the same transition.
        after = src.replace(f"    return {value}", f"    {bad_tail}")
        synthetic = Result(True, rel, "", before=src, after=after)
        findings = verify.check_syntax([synthetic])
        assert findings and findings[0].kind == "syntax", (seed, bad_tail)

    _fuzz_cases(case, 5000)


def test_symbol_splice_keeps_decorators_and_indent_seeded(tmp_path):
    def case(seed):
        r = random.Random(seed)
        rel = f"deco_{seed}.py"
        decs = r.sample(_DECORATORS, r.randrange(1, 4))
        old = r.randrange(1, 100)
        new = r.randrange(100, 999)
        lines = ["class Service:"]
        lines.extend(f"    {d}" for d in decs)
        lines.extend(["    def run(self, x):", f"        return x + {old}"])
        nl = "\r\n" if r.random() < 0.35 else "\n"
        (tmp_path / rel).write_bytes((nl.join(lines) + nl).encode("utf-8"))

        body = f"def run(self, x):\n    return x + {new}\n"
        reply = (f"<<<<<<< SYMBOL {rel} :: Service.run\n{body}"
                 ">>>>>>> END_SYMBOL\n")
        results = apply_edits(tmp_path, parse_edits(reply), backup=False)
        assert results and results[0].ok, (seed, results)
        text = fileio.read_text(tmp_path / rel)
        ast.parse(text)
        for dec in decs:
            assert dec in text, (seed, dec, text)
        method_line = next(ln for ln in text.split("\n")
                           if ln.strip().startswith("def run"))
        assert method_line.startswith("    def run"), (seed, repr(method_line))
        assert f"return x + {new}" in text
        assert not any(ln.strip() == f"return x + {old}"
                       for ln in text.split("\n")), (seed, text)

    _fuzz_cases(case, 6000)


def test_numeric_search_reanchors_or_refuses_seeded(tmp_path):
    words = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
             "golf", "hotel", "india", "juliet", "kilo", "lima")

    def case(seed):
        r = random.Random(seed)
        rel = f"numbers_{seed}.txt"
        lines = [f"{w}" for w in r.sample(words, 8)]
        path = tmp_path / rel
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if seed % 2:
            line_no = r.randrange(1, len(lines) + 1)
            reply = _block(rel, str(line_no), f"REPLACED_{line_no}")
            results = apply_edits(tmp_path, parse_edits(reply), backup=False)
            assert results and results[0].ok, (seed, results)
            assert "auto-reanchored" in results[0].message, seed
            got = fileio.read_lines(path)
            assert got[line_no - 1] == f"REPLACED_{line_no}", (seed, got)
        else:
            line_no = len(lines) + r.randrange(1, 20)
            reply = _block(rel, str(line_no), "NOPE")
            before = path.read_text(encoding="utf-8")
            results = apply_edits(tmp_path, parse_edits(reply), backup=False)
            assert results and not results[0].ok, (seed, results)
            assert "line number" in results[0].message, (seed, results[0].message)
            assert path.read_text(encoding="utf-8") == before, seed

    _fuzz_cases(case, 7000)


def test_lines_anchor_replaces_or_refuses_out_of_range_seeded(tmp_path):
    def case(seed):
        r = random.Random(seed)
        js = bool(seed % 2)
        rel = f"lines_{seed}." + ("js" if js else "py")
        value = r.randrange(10, 90)
        if js:
            lines = ["function f() {", f"  return {value};", "}"]
            target = 2
            repl = f"  return {value + 1};"
        else:
            lines = ["def f():", f"    return {value}"]
            target = 2
            repl = f"    return {value + 1}"
        path = tmp_path / rel
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        reply = (f"--- FILE: {rel} LINES {target}-{target}\n"
                 "<<<<<<< SEARCH\n\n=======\n"
                 f"{repl}\n>>>>>>> REPLACE\n")
        results = apply_edits(tmp_path, parse_edits(reply), backup=False)
        assert results and results[0].ok, (seed, results)
        assert repl.strip() in fileio.read_text(path), seed

        bad = (f"--- FILE: {rel} LINES 99-120\n<<<<<<< SEARCH\n\n"
               "=======\n" + repl + "\n>>>>>>> REPLACE\n")
        before = path.read_text(encoding="utf-8")
        refused = apply_edits(tmp_path, parse_edits(bad), backup=False)
        assert refused and not refused[0].ok, (seed, refused)
        assert "out of bounds" in refused[0].message, \
            (seed, refused[0].message)
        assert path.read_text(encoding="utf-8") == before, seed

    _fuzz_cases(case, 7500)


def test_after_symbol_inserts_a_sibling_with_aligned_indent_seeded(tmp_path):
    def case(seed):
        r = random.Random(seed)
        rel = f"insert_{seed}.py"
        value = r.randrange(1, 100)
        (tmp_path / rel).write_text(
            "class Store:\n"
            "    def get(self):\n"
            f"        return {value}\n",
            encoding="utf-8")
        method = f"put_{seed}"
        body = (f"def {method}(self, value):\n"
                "    self.value = value\n")
        if seed % 2:                        # half at column 0, half aligned
            body = ("    def " + f"{method}(self, value):\n"
                    "        self.value = value\n")
        reply = (f"<<<<<<< AFTER_SYMBOL {rel} :: Store.get\n{body}"
                 ">>>>>>> END_SYMBOL\n")
        results = apply_edits(tmp_path, parse_edits(reply), backup=False)
        assert results and results[0].ok, (seed, results)
        text = fileio.read_text(tmp_path / rel)
        ast.parse(text)
        assert "def get(self):" in text and f"def {method}(self" in text, text
        put_line = next(ln for ln in text.split("\n")
                        if ln.strip().startswith(f"def {method}"))
        assert put_line.startswith("    def "), (seed, repr(put_line))
        assert text.index("def get") < text.index(f"def {method}")

    _fuzz_cases(case, 7700)


def test_fuzzy_locator_applies_a_near_miss_unique_anchor_seeded(tmp_path):
    # The fuzzy locator compares candidate windows line by line (§5/§25): a
    # multi-line block with one drifted line still anchors, while an ambiguous
    # exact match is refused earlier (tested in test_smoke.py).
    cfg = Config()
    cfg.fuzzy_match = True
    cfg.fuzzy_min_ratio = 0.6

    def case(seed):
        r = random.Random(seed)
        js = bool(seed % 2)
        rel = f"fuzzy_{seed}." + ("js" if js else "py")
        base = r.randrange(10, 90)
        new = base + r.randrange(2, 9)
        if js:
            search = ("function alpha_unique(value) {\n"
                      f"  const total = value * {base + 1};\n"
                      "  return total + 1;\n}")
            replace = ("function alpha_unique(value) {\n"
                       f"  const total = value * {new};\n"
                       "  return total + 1;\n}")
            text = ("function alpha_unique(value) {\n"
                    f"  const total = value * {base};\n"
                    "  return total + 1;\n}\n"
                    "function beta_unrelated(value) {\n"
                    "  return value - 7;\n}\n")
        else:
            search = ("def alpha_unique(value):\n"
                      f"    total = value * {base + 1}\n"
                      "    return total + 1\n")
            replace = ("def alpha_unique(value):\n"
                       f"    total = value * {new}\n"
                       "    return total + 1\n")
            text = ("def alpha_unique(value):\n"
                    f"    total = value * {base}\n"
                    "    return total + 1\n"
                    "\n"
                    "def beta_unrelated(value):\n"
                    "    return value - 7\n")
        path = tmp_path / rel
        path.write_text(text, encoding="utf-8")
        results = apply_edits(tmp_path, parse_edits(_block(rel, search, replace)),
                              backup=False, cfg=cfg)
        assert results and results[0].ok, (seed, results)
        assert results[0].fuzzy, (seed, results[0].message)
        assert "matched approximately" in results[0].message, seed
        assert replace.strip() in fileio.read_text(path), seed

    _fuzz_cases(case, 7900)


# ---------------------------------------------------------------------------
# 3. webcheck: comments, strings, regexes and nested template literals
# ---------------------------------------------------------------------------
_JS_FRAGS = (
    "// ghostComment_{n}()\n",
    "/* ghostBlock_{n}() */\n",
    "const s{n} = 'ghostSingle_{n}()';\n",
    'const d{n} = "ghostDouble_{n}()";\n',
    "const r{n} = /['\"]ghostRegex_{n}\\(\\)/g;\n",
    "const b{n} = /[{}]ghostClass_{n}()/g;\n",
    "const c{n} = 'unterminated_{n} at line break\n",
)


def _js_fragment(template: str, seed: int) -> str:
    return template.replace("{n}", str(seed))


def test_js_blanker_preserves_length_and_newlines_seeded():
    def case(seed):
        r = random.Random(seed)
        src = "".join(_js_fragment(r.choice(_JS_FRAGS), seed)
                      for _ in range(r.randrange(0, 10)))
        for blank in (webcheck._js_code, webcheck._js_no_comments):
            out = blank(src)
            assert len(out) == len(src), (seed, blank.__name__, src, out)
            assert [i for i, c in enumerate(out) if c == "\n"] == \
                [i for i, c in enumerate(src) if c == "\n"], \
                (seed, blank.__name__, src, out)

    _fuzz_cases(case, 8000)


def test_js_comment_string_false_positives_and_interpolation_calls_seeded(
        tmp_path):
    path = tmp_path / "app.js"

    def case(seed):
        r = random.Random(seed)
        frags = [_js_fragment(f, seed) for f in _JS_FRAGS]
        r.shuffle(frags)
        real = f"ghostCall_{seed}"
        nested = f"ghostNested_{seed}"
        template = (
            f"const t{seed} = html`<div>${{ok ? html`<span>${{{nested}()}}"
            f"</span>` : 'none'}}</div>`;\n"
            f"const u{seed} = html`<p>${{ /[}}]/.test(q) && {real}() }}</p>`;\n"
        )
        src = ("export const ready = true;\n" + template + "".join(frags))
        path.write_text(src, encoding="utf-8")
        code = webcheck._js_code(src)
        assert real in code and nested in code, (seed, code)

        findings = webcheck.run(tmp_path)
        messages = "\n".join(f.message for f in findings)
        assert real in messages, (seed, messages)
        assert nested in messages, (seed, messages)
        for kind in ("Comment", "Block", "Single", "Double", "Regex", "Class"):
            ghost = f"ghost{kind}_{seed}"
            assert ghost not in messages, (seed, ghost, messages)

    _fuzz_cases(case, 9000)


def test_env_knobs_select_depth_and_single_seed(monkeypatch):
    # Test the helper in isolation from any stress value captured at import.
    monkeypatch.setattr(sys.modules[__name__], "_IMPORT_ENV", {})
    monkeypatch.delenv("PEDACITO_FUZZ_ITERS", raising=False)
    monkeypatch.delenv("PEDACITO_FUZZ_SEED", raising=False)
    assert len(_fuzz_seeds()) == DEFAULT_FUZZ_ITERS == 250

    monkeypatch.setenv("PEDACITO_FUZZ_ITERS", "12")
    assert _fuzz_seeds() == list(range(12))
    assert _fuzz_seeds(100) == list(range(100, 112))

    monkeypatch.setenv("PEDACITO_FUZZ_ITERS", "10000")
    monkeypatch.setenv("PEDACITO_FUZZ_SEED", "999")
    assert _fuzz_seeds() == [999]
    assert _fuzz_seeds(5000) == [999]

    # Invalid/empty values fall back to the fast default, never a crash.
    monkeypatch.delenv("PEDACITO_FUZZ_SEED")
    for bad in ("", "nope", "0", "-3"):
        monkeypatch.setenv("PEDACITO_FUZZ_ITERS", bad)
        assert _fuzz_seeds() == list(range(DEFAULT_FUZZ_ITERS)), bad
    monkeypatch.setenv("PEDACITO_FUZZ_ITERS", "4")
    monkeypatch.setenv("PEDACITO_FUZZ_SEED", "not-a-number")
    assert _fuzz_seeds() == list(range(4))


def test_seed_failure_message_is_copy_pasteable():
    message = _seed_failure(41892)
    assert "seed=41892" in message
    assert "PEDACITO_FUZZ_SEED=41892" in message
    assert "pytest tests/test_fuzz_invariants.py" in message


def test_regex_brace_inside_interpolation_keeps_the_later_call():
    """A `}` inside a regex literal or comment used to close `${ ... }`
    early, blanking every real call after it (a missed ReferenceError)."""
    src = (
        "const a = html`x ${ /[}]/.test(q) && ghostOne() } y`;\n"
        "const b = html`x ${ // } comment\n ghostTwo() } y`;\n"
        "const c = html`x ${ /* } */ ghostThree() } y`;\n"
    )
    code = webcheck._js_code(src)
    for name in ("ghostOne", "ghostTwo", "ghostThree"):
        assert name in code, (name, code)
    assert len(code) == len(src)
