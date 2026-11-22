"""Regression: a --refine spec must not be auto-planned as its headings.

The sectioned spec is one task. `requirement_titles` correctly found its five
REQUIREMENTS lines, but should_plan then auto-split the whole spec and added
steps literally named OBJECTIVE, FILES TO CREATE OR MODIFY, INTERFACE
CONTRACTS and EDGE CASES. Applying those steps against the same file produced
duplicate edits, corrupt partial files and ten-minute runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import plan


REFINED = """OBJECTIVE
Add a footer note under the instructions.

FILES TO CREATE OR MODIFY
- [MODIFY] index.html -- add the footer

INTERFACE CONTRACTS (every later step must use these EXACT names and argument counts)
- none

REQUIREMENTS
1. The footer text is exact.
2. The footer is below the instructions.
3. The footer is centered.
4. The footer uses a small font.
5. The footer has vertical padding.

EDGE CASES
1. Missing instructions section.

VERIFICATION
Open index.html and confirm the footer.
"""


class _Cfg:
    plan_threshold = 0
    sources: dict = {}


def test_refined_spec_is_not_auto_planned():
    assert not plan.should_plan(REFINED, None, _Cfg())
