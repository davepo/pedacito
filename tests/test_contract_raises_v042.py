"""BUG-08: intentional invalid calls inside an exception expectation.

`with pytest.raises(TypeError): format_amount(...)` asserts that the call is
invalid, so the static contract gate must not report it as arity drift. A
genuine arity mismatch outside such a context is still a ContractError.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import spec  # noqa: E402

CURRENCY = (
    "def format_amount(amount, region):\n"
    "    return f'{amount} {region}'\n"
)

HEADER = (
    "import unittest\n"
    "from decimal import Decimal\n"
    "import pytest\n"
    "from currency import format_amount\n"
    "import currency\n\n"
)


def _project(tmp_path, test_body: str) -> Path:
    (tmp_path / "currency.py").write_text(CURRENCY, encoding="utf-8")
    (tmp_path / "test_currency.py").write_text(HEADER + test_body,
                                               encoding="utf-8")
    return tmp_path


def test_pytest_raises_block_call_is_not_flagged(tmp_path):
    root = _project(tmp_path,
                    "def test_rejects_extra_arg():\n"
                    "    with pytest.raises(TypeError):\n"
                    "        format_amount(Decimal('5'), 'US', True)\n")
    assert spec.verify_cross_file_contracts(root) == []


def test_pytest_raises_direct_call_arguments_are_not_flagged(tmp_path):
    root = _project(tmp_path,
                    "def test_direct_form():\n"
                    "    pytest.raises(TypeError, format_amount,\n"
                    "                  Decimal('5'), 'US', True)\n")
    assert spec.verify_cross_file_contracts(root) == []


def test_assert_raises_block_call_is_not_flagged(tmp_path):
    root = _project(tmp_path,
                    "class T(unittest.TestCase):\n"
                    "    def test_rejects_extra_arg(self):\n"
                    "        with self.assertRaises(TypeError):\n"
                    "            format_amount(Decimal('5'), 'US', True)\n")
    assert spec.verify_cross_file_contracts(root) == []


def test_dotted_call_inside_raises_block_is_not_flagged(tmp_path):
    root = _project(tmp_path,
                    "def test_dotted():\n"
                    "    with pytest.raises(TypeError):\n"
                    "        currency.format_amount(1, 2, 3)\n")
    assert spec.verify_cross_file_contracts(root) == []


def test_genuine_arity_mismatch_outside_raises_is_still_caught(tmp_path):
    root = _project(tmp_path,
                    "def test_real_drift():\n"
                    "    with pytest.raises(TypeError):\n"
                    "        format_amount(1, 'US')\n"
                    "    format_amount(Decimal('5'), 'US', True)\n")
    errs = spec.verify_cross_file_contracts(root)
    assert len(errs) == 1
    assert errs[0].startswith("ContractError in test_currency.py:")
    assert "format_amount" in errs[0] and "but 3 were given" in errs[0]


def test_genuine_dotted_mismatch_outside_raises_is_still_caught(tmp_path):
    root = _project(tmp_path,
                    "def test_dotted_drift():\n"
                    "    currency.format_amount(1, 2, 3)\n")
    errs = spec.verify_cross_file_contracts(root)
    assert any("format_amount" in e and "3 arg(s)" in e for e in errs)
