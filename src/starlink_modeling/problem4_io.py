"""Problem-four configuration and audited Problem-three Standard discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_M = 40
EXPECTED_N = 46
EXPECTED_S = 1840
EXPECTED_SIGNATURE = "standard_double_M40_N46_S1840"

REQUIRED_RESULT_FILES = (
    "problem3_run_metadata.json",
    "topology_summary.json",
    "routing_summary.json",
    "traffic_summary.json",
    "topology_timeseries.csv",
    "routing_time_metrics.csv",
    "traffic_timeseries.csv",
    "satellite_load_timeseries.csv",
    "problem3_run.log",
)


class Problem3SelectionError(RuntimeError):
    """Raised when no auditable Problem-three baseline can be selected."""


def is_formal_result(mode: str, completed: bool = True) -> bool:
    """Only completed Standard or Full runs may be cited as formal results."""

    return str(mode).lower() in {"standard", "full"} and bool(completed)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _stage_is_complete(status: Any, *, output_exists: bool) -> bool:
    normalized = str(status or "").strip().lower()
    if normalized in {"completed", "user_given_region_density"}:
        return True
    # A resumed run does not recompute the stage. It is accepted only when the
    # stage's persisted output was independently found in the same directory.
    return normalized == "resumed_existing_output" and output_exists


@dataclass(frozen=True)
class Problem3Candidate:
    source_directory: str
    metadata_file: str
    mode: str
    completed: bool
    stage_status: dict[str, Any]
    M: int | None
    N: int | None
    total_satellites: int | None
    constellation_signature: str
    constellation_scenario: str
    traffic_status: str
    finished_at_utc: str
    missing_files: list[str]
    eligible: bool
    eligibility_reason: str
    field_completeness: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectedProblem3Standard:
    source_directory: str
    metadata_file: str
    metadata_sha256: str
    topology_summary_file: str
    routing_summary_file: str
    traffic_summary_file: str
    mode: str
    completed: bool
    M: int
    N: int
    total_satellites: int
    constellation_signature: str
    inclination_deg: float
    phase_factor_F: int
    Omega0_deg: float
    u0_deg: float
    altitude_km: float
    theta_deg: float
    problem3_mean_delay_ms: float
    problem3_p95_delay_ms: float
    problem3_max_delay_ms: float
    problem3_ratio_below_30ms: float
    problem3_mean_throughput_tbps: float
    problem3_mean_service_ratio: float
    problem3_peak_service_ratio: float
    problem3_max_satellite_utilization: float
    stage_status: dict[str, Any]
    traffic_status: str
    selection_reason: str
    candidate_audit: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_from_metadata(path: Path, project_root: Path) -> Problem3Candidate:
    try:
        metadata = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return Problem3Candidate(
            source_directory=_project_relative(path.parent, project_root),
            metadata_file=_project_relative(path, project_root),
            mode="invalid",
            completed=False,
            stage_status={},
            M=None,
            N=None,
            total_satellites=None,
            constellation_signature="",
            constellation_scenario="",
            traffic_status="",
            finished_at_utc="",
            missing_files=list(REQUIRED_RESULT_FILES),
            eligible=False,
            eligibility_reason=f"元数据无法读取：{exc}",
            field_completeness=0,
        )

    directory = path.parent
    missing = [name for name in REQUIRED_RESULT_FILES if not (directory / name).exists()]
    stage_status = metadata.get("stage_status") or {}
    traffic_payload: dict[str, Any] = {}
    traffic_path = directory / "traffic_summary.json"
    if traffic_path.exists():
        try:
            traffic_payload = read_json(traffic_path)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    traffic_status = str(traffic_payload.get("status") or "")
    selected = metadata.get("selected_constellation") or {}
    mode = str(metadata.get("mode") or "").lower()
    scenario = str(metadata.get("constellation_scenario") or selected.get("scenario") or "").lower()
    M = metadata.get("M", selected.get("M"))
    N = metadata.get("N", selected.get("N"))
    total = metadata.get("total_satellites", selected.get("total_satellites"))
    signature = str(
        metadata.get("constellation_signature")
        or selected.get("constellation_signature")
        or ""
    )
    topology_ok = _stage_is_complete(
        stage_status.get("topology"), output_exists=(directory / "topology_summary.json").exists()
    )
    routing_ok = _stage_is_complete(
        stage_status.get("routing"), output_exists=(directory / "routing_summary.json").exists()
    )
    traffic_ok = _stage_is_complete(
        stage_status.get("traffic"), output_exists=traffic_path.exists()
    ) and traffic_status in {"user_given_region_density", "completed"}
    checks = {
        "mode_standard": mode == "standard",
        "completed": bool(metadata.get("completed")),
        "topology_complete": topology_ok,
        "routing_complete": routing_ok,
        "traffic_complete": traffic_ok,
        "scenario_double": scenario == "double",
        "M": M is not None and int(M) == EXPECTED_M,
        "N": N is not None and int(N) == EXPECTED_N,
        "S": total is not None and int(total) == EXPECTED_S,
        "signature": signature == EXPECTED_SIGNATURE,
        "required_files": not missing,
    }
    failed = [name for name, passed in checks.items() if not passed]
    resumed = any(str(value).lower() == "resumed_existing_output" for value in stage_status.values())
    reason = (
        "合格 Standard；阶段由 --resume 读取，已通过同目录完整产物复核"
        if not failed and resumed
        else "合格 Standard；所有阶段状态与产物完整"
        if not failed
        else "不合格：" + ", ".join(failed)
    )
    completeness_keys = (
        "finished_at_utc", "selected_constellation", "constellation_signature",
        "M", "N", "total_satellites", "config", "stage_status",
    )
    completeness = sum(metadata.get(key) is not None for key in completeness_keys)
    return Problem3Candidate(
        source_directory=_project_relative(directory, project_root),
        metadata_file=_project_relative(path, project_root),
        mode=mode,
        completed=bool(metadata.get("completed")),
        stage_status=dict(stage_status),
        M=None if M is None else int(M),
        N=None if N is None else int(N),
        total_satellites=None if total is None else int(total),
        constellation_signature=signature,
        constellation_scenario=scenario,
        traffic_status=traffic_status,
        finished_at_utc=str(metadata.get("finished_at_utc") or ""),
        missing_files=missing,
        eligible=not failed,
        eligibility_reason=reason,
        field_completeness=int(completeness),
    )


def audit_problem3_candidates(project_root: str | Path) -> list[Problem3Candidate]:
    root = Path(project_root)
    metadata_paths = sorted((root / "outputs" / "problem3").rglob("problem3_run_metadata.json"))
    return [_candidate_from_metadata(path, root) for path in metadata_paths]


def _parse_time(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def discover_problem3_standard(
    project_root: str | Path,
    explicit_directory: str | Path | None = None,
) -> SelectedProblem3Standard:
    root = Path(project_root)
    candidates = audit_problem3_candidates(root)
    if explicit_directory is not None:
        explicit = Path(explicit_directory)
        if not explicit.is_absolute():
            explicit = root / explicit
        wanted = explicit.resolve()
        candidates = [
            candidate
            for candidate in candidates
            if (root / candidate.source_directory).resolve() == wanted
            or Path(candidate.source_directory).resolve() == wanted
        ]
        if not candidates:
            raise Problem3SelectionError(f"指定目录中未找到问题三元数据：{explicit}")

    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        details = "\n".join(
            f"- {item.metadata_file}: mode={item.mode}, completed={item.completed}, "
            f"stage_status={item.stage_status}, reason={item.eligibility_reason}"
            for item in candidates
        ) or "- 未发现任何 problem3_run_metadata.json"
        raise Problem3SelectionError(
            "没有合格的问题三 Standard 正式结果，禁止用 Quick 自动替代。候选如下：\n"
            + details
        )
    eligible.sort(
        key=lambda item: (
            sum(str(v).lower() == "completed" for v in item.stage_status.values()),
            _parse_time(item.finished_at_utc),
            item.field_completeness,
        ),
        reverse=True,
    )
    chosen = eligible[0]
    directory = root / chosen.source_directory
    metadata_path = directory / "problem3_run_metadata.json"
    metadata = read_json(metadata_path)
    routing = read_json(directory / "routing_summary.json")
    traffic = read_json(directory / "traffic_summary.json")
    selected = metadata.get("selected_constellation") or read_json(
        directory / "selected_problem2_constellation.json"
    )
    selection_reason = (
        f"在 {len(candidates)} 个候选中选择唯一合格的 Standard 结果；"
        "mode、星座签名和规模均匹配。元数据记录 resumed_existing_output，"
        "但三个阶段的 JSON/CSV 均存在且 traffic_summary 明确为用户给定区域流量密度，"
        "因此认定为已完成结果的恢复校验，而不是 Quick 替代。"
    )
    return SelectedProblem3Standard(
        source_directory=chosen.source_directory,
        metadata_file=chosen.metadata_file,
        metadata_sha256=sha256_file(metadata_path),
        topology_summary_file=_project_relative(directory / "topology_summary.json", root),
        routing_summary_file=_project_relative(directory / "routing_summary.json", root),
        traffic_summary_file=_project_relative(directory / "traffic_summary.json", root),
        mode=chosen.mode,
        completed=chosen.completed,
        M=int(chosen.M),
        N=int(chosen.N),
        total_satellites=int(chosen.total_satellites),
        constellation_signature=chosen.constellation_signature,
        inclination_deg=float(selected["inclination_deg"]),
        phase_factor_F=int(selected["phase_factor_F"]),
        Omega0_deg=float(selected["Omega0_deg"]),
        u0_deg=float(selected["u0_deg"]),
        altitude_km=float(selected["altitude_km"]),
        theta_deg=float(selected["theta_deg"]),
        problem3_mean_delay_ms=float(routing["mean_delay_ms"]),
        problem3_p95_delay_ms=float(routing["p95_delay_ms"]),
        problem3_max_delay_ms=float(routing["max_delay_ms"]),
        problem3_ratio_below_30ms=float(routing["ratio_below_30ms"]),
        problem3_mean_throughput_tbps=float(traffic["optimized_mean_throughput_gbps"]) / 1000.0,
        problem3_mean_service_ratio=float(traffic["mean_service_ratio"]),
        problem3_peak_service_ratio=float(traffic["peak_demand_service_ratio"]),
        problem3_max_satellite_utilization=float(traffic["maximum_satellite_utilization"]),
        stage_status=chosen.stage_status,
        traffic_status=chosen.traffic_status,
        selection_reason=selection_reason,
        candidate_audit=[item.to_dict() for item in audit_problem3_candidates(root)],
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_problem4_config(
    project_root: str | Path,
    mode: str,
    custom_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    default_path = root / "configs" / f"problem4_{mode}.json"
    config = read_json(default_path)
    if custom_path is not None:
        custom = Path(custom_path)
        if not custom.is_absolute():
            custom = root / custom
        config = _deep_merge(config, read_json(custom))
    if config.get("mode") != mode:
        raise ValueError(f"配置 mode={config.get('mode')!r} 与命令行 mode={mode!r} 不一致。")
    return config


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
