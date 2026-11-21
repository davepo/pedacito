"""Regression: an edit header must not create a file named after a code line.

A model that lost the file header emitted the line it meant to replace as
`--- FILE: <line>`. The path stays inside the project, so it was accepted:
a real file appeared with backticks and `${...}` in its name while index.html
was left untouched. New-file paths now reject code/prose punctuation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import edits


def test_code_line_as_file_header_is_refused(tmp_path):
    reply = (
        '--- FILE: tile.className = `tile tile-${value}`;\n'
        '<<<<<<< SEARCH\n'
        '=======\n'
        ' tile.className = value === -1 ? "tile tile-block" : '
        '`tile tile-${value}`;\n'
        '>>>>>>> REPLACE\n')
    results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                backup=False)
    assert results and not results[0].ok
    assert "looks like a copied source line" in results[0].message
    assert [p.name for p in tmp_path.iterdir()] == []


def test_a_normal_new_file_is_still_created(tmp_path):
    reply = ('--- FILE: src/report.md\n'
             '<<<<<<< SEARCH\n'
             '=======\n'
             'Quarterly report body.\n'
             '>>>>>>> REPLACE\n')
    results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                backup=False)
    assert results and results[0].ok
    assert (tmp_path / "src" / "report.md").read_text(encoding="utf-8")
