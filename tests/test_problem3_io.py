import json
from pathlib import Path

import pytest

from starlink_modeling.problem3_io import (
    ConstellationSelectionError,
    discover_problem2_constellation,
)


def _write_candidate(path: Path, *, M: int, inclination: float = 50.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "standard",
        "scenario": "single",
        "M": M,
        "N": 10,
        "total_satellites": M * 10,
        "inclination_deg": inclination,
        "phase_factor_F": 1,
        "Omega0_deg": 2.0,
        "u0_deg": 3.0,
        "altitude_km": 550.0,
        "theta_deg": 4.55,
        "validation_status": "standard_search_passed_requires_full",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discovers_repository_standard_single_constellation():
    root = Path(__file__).resolve().parents[1]
    selected = discover_problem2_constellation(root)
    assert selected.mode == "standard"
    assert selected.scenario == "single"
    assert (selected.M, selected.N, selected.total_satellites) == (34, 44, 1496)
    assert "standard_M25-50_N35-50" in selected.source_file
    assert len(selected.source_sha256) == 64


def test_equal_ranked_different_candidates_are_rejected(tmp_path):
    output = tmp_path / "outputs" / "problem2"
    _write_candidate(output / "standard_a" / "best_single_constellation.json", M=4)
    _write_candidate(output / "standard_b" / "best_single_constellation.json", M=5)
    with pytest.raises(ConstellationSelectionError, match="多套同优先级"):
        discover_problem2_constellation(tmp_path)


def test_explicit_constellation_file_resolves_ambiguity(tmp_path):
    path = tmp_path / "manual.json"
    _write_candidate(path, M=6)
    selected = discover_problem2_constellation(tmp_path, path)
    assert selected.M == 6
    assert selected.total_satellites == 60

