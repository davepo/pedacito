"""Regression tests for the tree-sitter version guard.

py-tree-sitter 0.26.0 regressed Node lifetime handling and segfaults CPython
while Pedacito walks a live tree (upstream #472). Pedacito's pyproject pins
`tree-sitter<0.26`, but an environment can still have 0.26 installed. The
guard must refuse it and let callers fall back to the structural chunker
instead of crashing the process.
"""

import gc
import threading
import pytest

from pedacito.chunker_ts import _tree_sitter_is_safe, _version_tuple


def test_version_tuple_parses_common_forms():
    assert _version_tuple("0.25.2") == (0, 25, 2)
    assert _version_tuple("0.26.0rc1") == (0, 26, 0)
    assert _version_tuple("1.2") == (1, 2)
    assert _version_tuple("") == (0,)


def test_tree_sitter_025_is_allowed():
    assert _tree_sitter_is_safe("0.23.0")
    assert _tree_sitter_is_safe("0.25.2")


def test_tree_sitter_026_and_later_are_rejected():
    assert not _tree_sitter_is_safe("0.26.0")
    assert not _tree_sitter_is_safe("0.26.0rc1")
    assert not _tree_sitter_is_safe("0.27.1")
    assert not _tree_sitter_is_safe("1.0.0")


def test_unavailable_tree_sitter_has_a_reason():
    from pedacito import chunker_ts
    if chunker_ts.TS_AVAILABLE:
        assert chunker_ts._TS_DISABLED_REASON == ""
    else:                                              # pragma: no cover
        assert chunker_ts._TS_DISABLED_REASON


def test_ts_nodes_gc_refcount_stays_disabled_across_overlapping_threads():
    """GC must stay disabled while ANY thread holds `ts_nodes()` and only
    re-enable when the outermost holder exits."""
    from pedacito import chunker_ts

    gc_was_on = gc.isenabled()
    gc.enable()
    a_entered = threading.Event()
    b_entered = threading.Event()
    a_exited = threading.Event()
    observed: dict[str, bool] = {}

    def thread_a():
        with chunker_ts.ts_nodes():
            observed["a_inside"] = gc.isenabled()
            a_entered.set()
            assert b_entered.wait(timeout=5)
        observed["a_after_exit"] = gc.isenabled()
        a_exited.set()

    def thread_b():
        assert a_entered.wait(timeout=5)
        with chunker_ts.ts_nodes():
            observed["b_inside_overlap"] = gc.isenabled()
            b_entered.set()
            assert a_exited.wait(timeout=5)
            # Thread A exited, but Thread B still holds ts_nodes(): GC must stay off.
            observed["b_after_a_exited"] = gc.isenabled()

    try:
        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join(timeout=10)
        tb.join(timeout=10)
        assert not ta.is_alive() and not tb.is_alive()
        assert observed["a_inside"] is False
        assert observed["b_inside_overlap"] is False
        assert observed["a_after_exit"] is False
        assert observed["b_after_a_exited"] is False
        assert gc.isenabled() is True
        assert chunker_ts._GC_DEPTH == 0

        # If GC was already disabled before the outermost ts_nodes(), keep it off.
        gc.disable()
        with chunker_ts.ts_nodes():
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        if gc_was_on:
            gc.enable()
        else:
            gc.disable()


def test_chunker_ts_multithreaded_parse_stress_and_thread_local_parsers():
    """Concurrent tree-sitter parses must use thread-local Parser instances,
    keep GC disabled while inside ts_nodes(), and restore GC cleanly."""
    from pedacito import chunker_ts

    if not chunker_ts.TS_AVAILABLE:
        pytest.skip("tree-sitter unavailable")

    ts_src = (
        "export class Service {\n"
        "  compute(x: number): number {\n"
        "    return x * 2;\n"
        "  }\n"
        "}\n"
        "export function helper(a: string): string {\n"
        "  return a.trim();\n"
        "}\n"
    )
    js_src = (
        "export function greet(name) {\n"
        "  return `hi ${name}`;\n"
        "}\n"
    )

    n_threads = 6
    iters = 20
    barrier = threading.Barrier(n_threads)
    parser_ids: list[int] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    gc_was_on = gc.isenabled()
    gc.enable()

    def worker(idx: int):
        try:
            barrier.wait(timeout=5)
            p1 = chunker_ts._get_parser("typescript")
            p2 = chunker_ts._get_parser("typescript")
            assert p1 is p2, "parser should be cached within the same thread"
            with lock:
                parser_ids.append(id(p1))
            for i in range(iters):
                chunks_ts = chunker_ts.chunk_file_ts(
                    f"svc_{idx}_{i}.ts", ts_src, ".ts", class_split_lines=1)
                quals_ts = {c.qualname for c in chunks_ts}
                assert "Service" in quals_ts
                assert "Service.compute" in quals_ts
                assert "helper" in quals_ts

                chunks_js = chunker_ts.chunk_file_ts(
                    f"mod_{idx}_{i}.js", js_src, ".js")
                assert any(c.qualname == "greet" for c in chunks_js)
            # Keep threads alive past barrier so thread-local parsers stay
            # distinct across all concurrent workers.
            barrier.wait(timeout=10)
        except BaseException as exc:  # pragma: no cover - surfaced below
            with lock:
                errors.append(exc)

    try:
        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive()
        assert errors == []
        assert len(parser_ids) == n_threads
        assert len(set(parser_ids)) == n_threads, (
            "each concurrent thread must have its own tree-sitter Parser"
        )
        assert chunker_ts._GC_DEPTH == 0
        assert gc.isenabled() is True
    finally:
        if gc_was_on:
            gc.enable()
        else:
            gc.disable()
