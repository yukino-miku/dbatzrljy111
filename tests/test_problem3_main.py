from pathlib import Path

import pandas as pd

from problem3_network_routing import _write_empty_traffic_outputs


def test_missing_real_traffic_writes_explicit_skipped_status(tmp_path: Path):
    stale = tmp_path / "fig_problem3_traffic_profile.png"
    stale.write_bytes(b"stale-demo")
    summary = _write_empty_traffic_outputs(tmp_path, "missing test data")
    assert summary["status"] == "skipped_missing_real_data"
    assert not stale.exists()
    frame = pd.read_csv(tmp_path / "traffic_timeseries.csv")
    assert frame.empty
    assert "satellite_demand_gbps" in frame.columns

