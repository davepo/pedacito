"""Regression tests for the tree-sitter version guard.

py-tree-sitter 0.26.0 regressed Node lifetime handling and segfaults CPython
while Pedacito walks a live tree (upstream #472). Pedacito's pyproject pins
`tree-sitter<0.26`, but an environment can still have 0.26 installed. The
guard must refuse it and let callers fall back to the structural chunker
instead of crashing the process.
"""

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
