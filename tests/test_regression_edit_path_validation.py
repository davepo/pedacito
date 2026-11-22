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


def test_single_quoted_code_line_as_file_header_is_refused(tmp_path):
    """The first refusal only caught backtick/${} punctuation; the common
    single-quoted/parenthesised code lines still created bogus files."""
    for line in ("console.log('hi')",
                 "import foo from 'bar'",
                 "def greet(name):",
                 "print('hello', name)",
                 "arr[0]",
                 "# a comment",
                 "@app.route",
                 "import os",
                 "return x"):
        reply = (f'--- FILE: {line}\n'
                 '<<<<<<< SEARCH\n'
                 '=======\n'
                 'body\n'
                 '>>>>>>> REPLACE\n')
        results = edits.apply_edits(tmp_path, edits.parse_edits(reply),
                                    backup=False)
        assert results and not results[0].ok, line
        assert "looks like a copied source line" in results[0].message, line
    assert list(tmp_path.iterdir()) == []


def test_ordinary_new_names_still_pass_the_plausibility_check():
    for rel in ("src/report.md", "notes/my file.txt", "nota-ñandú.md",
                "Makefile", "LICENSE", ".gitignore", "a+b.txt",
                "Q&A.md", "100%-done.md", "hello!.txt"):
        assert edits._plausible_new_path(rel), rel
