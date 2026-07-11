import json
import logging
from pathlib import Path

import pandas as pd

from problem3_network_routing import (
    _write_empty_traffic_outputs,
    invalidate_stale_constellation_outputs,
    resume_outputs_match,
)
from starlink_modeling.problem3_io import discover_problem2_constellation


def test_missing_real_traffic_writes_explicit_skipped_status(tmp_path: Path):
    stale = tmp_path / "fig_problem3_traffic_profile.png"
    stale.write_bytes(b"stale-demo")
    summary = _write_empty_traffic_outputs(tmp_path, "missing test data")
    assert summary["status"] == "skipped_missing_real_data"
    assert not stale.exists()
    frame = pd.read_csv(tmp_path / "traffic_timeseries.csv")
    assert frame.empty
    assert "satellite_demand_gbps" in frame.columns


def test_old_single_constellation_outputs_are_invalidated(tmp_path: Path):
    selected = discover_problem2_constellation(Path(__file__).resolve().parents[1])
    (tmp_path / "selected_problem2_constellation.json").write_text(
        json.dumps(
            {
                "constellation_signature": "standard_single_M34_N44_S1496",
                "constellation_cache_key": "old-single-key",
            }
        ),
        encoding="utf-8",
    )
    stale = tmp_path / "topology_summary.json"
    stale.write_text("{}", encoding="utf-8")
    changed = invalidate_stale_constellation_outputs(
        tmp_path, selected, logging.getLogger("test-problem3-invalidation")
    )
    assert changed
    assert not stale.exists()


def test_quick_cache_cannot_be_resumed_as_standard(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    selected = discover_problem2_constellation(root)
    quick_config = json.loads((root / "configs" / "problem3_quick.json").read_text(encoding="utf-8"))
    metadata = {
        "completed": True,
        "constellation_cache_key": selected.constellation_cache_key,
        "mode": "quick",
        "pair_mode": "sampled",
        "config": quick_config,
    }
    (tmp_path / "problem3_run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    args = type("Args", (), {"mode": "standard", "pair_mode": "sampled"})()
    standard_config = json.loads(
        (root / "configs" / "problem3_standard.json").read_text(encoding="utf-8")
    )
    assert not resume_outputs_match(tmp_path, selected, standard_config, args)
