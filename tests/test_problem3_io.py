import json
from pathlib import Path

import pytest

from starlink_modeling.problem3_io import (
    CONSTELLATION_SIGNATURE,
    ConstellationSelectionError,
    discover_problem2_constellation,
)


def _write_candidate(
    path: Path,
    *,
    scenario: str = "double",
    M: int = 40,
    N: int = 46,
    inclination: float = 50.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "standard",
        "scenario": scenario,
        "M": M,
        "N": N,
        "total_satellites": M * N,
        "inclination_deg": inclination,
        "phase_factor_F": 28,
        "Omega0_deg": 2.0,
        "u0_deg": 3.0,
        "altitude_km": 550.0,
        "theta_deg": 4.55,
        "validation_status": "standard_search_passed_requires_full",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discovers_repository_standard_double_constellation():
    root = Path(__file__).resolve().parents[1]
    selected = discover_problem2_constellation(root)
    assert selected.mode == "standard"
    assert selected.scenario == "double"
    assert (selected.M, selected.N, selected.total_satellites) == (40, 46, 1840)
    assert selected.constellation_signature == CONSTELLATION_SIGNATURE
    assert "standard_M25-50_N35-50" in selected.source_file
    assert "best_double_constellation.json" in selected.source_file
    assert len(selected.source_sha256) == 64
    assert len(selected.constellation_cache_key) == 64


def test_equal_ranked_different_double_candidates_are_rejected(tmp_path):
    output = tmp_path / "outputs" / "problem2"
    _write_candidate(
        output / "standard_a" / "best_double_constellation.json",
        inclination=49.0,
    )
    _write_candidate(
        output / "standard_b" / "best_double_constellation.json",
        inclination=51.0,
    )
    with pytest.raises(ConstellationSelectionError, match="多套同优先级"):
        discover_problem2_constellation(tmp_path)


def test_single_constellation_is_never_used_as_fallback(tmp_path):
    output = tmp_path / "outputs" / "problem2"
    _write_candidate(
        output / "standard" / "best_double_constellation.json",
        scenario="single",
        M=34,
        N=44,
    )
    with pytest.raises(ConstellationSelectionError, match="不允许退回单重覆盖"):
        discover_problem2_constellation(tmp_path)


def test_explicit_wrong_constellation_is_rejected(tmp_path):
    path = tmp_path / "manual.json"
    _write_candidate(path, scenario="single", M=34, N=44)
    with pytest.raises(ConstellationSelectionError, match="不允许退回单重覆盖"):
        discover_problem2_constellation(tmp_path, path)


def test_cache_key_changes_with_orbit_parameter(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_candidate(first_path, inclination=50.0)
    _write_candidate(second_path, inclination=50.1)
    first = discover_problem2_constellation(tmp_path, first_path)
    second = discover_problem2_constellation(tmp_path, second_path)
    assert first.constellation_cache_key != second.constellation_cache_key
