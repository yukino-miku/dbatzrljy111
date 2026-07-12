import json
from pathlib import Path

import pytest

from starlink_modeling.problem4_io import (
    EXPECTED_SIGNATURE,
    Problem3SelectionError,
    discover_problem3_standard,
    is_formal_result,
)


def test_repository_selects_standard_and_never_root_quick():
    root = Path(__file__).resolve().parents[1]
    selected = discover_problem3_standard(root)
    assert selected.mode == "standard"
    assert (selected.M, selected.N, selected.total_satellites) == (40, 46, 1840)
    assert selected.constellation_signature == EXPECTED_SIGNATURE
    assert "standard_run_20260711" in selected.source_directory
    assert all(item["mode"] != "quick" or not item["eligible"] for item in selected.candidate_audit)


def test_explicit_quick_directory_is_rejected():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(Problem3SelectionError, match="禁止用 Quick"):
        discover_problem3_standard(root, root / "outputs" / "problem3")


def test_quick_is_never_marked_as_formal_result():
    assert not is_formal_result("quick")
    assert not is_formal_result("standard", completed=False)
    assert is_formal_result("standard")
    assert is_formal_result("full")
