"""Input discovery and reproducibility helpers for problem three."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


PREFERRED_VALIDATION_WORDS = (
    "passed",
    "validated",
    "success",
    "feasible",
    "search_passed",
)
EXPECTED_SCENARIO = "double"
EXPECTED_M = 40
EXPECTED_N = 46
EXPECTED_TOTAL_SATELLITES = 1840
CONSTELLATION_SIGNATURE = "standard_double_M40_N46_S1840"


class ConstellationSelectionError(RuntimeError):
    """Raised when a problem-two constellation cannot be selected safely."""


@dataclass(frozen=True)
class SelectedConstellation:
    source_file: str
    source_sha256: str
    source_modified_utc: str
    mode: str
    scenario: str
    M: int
    N: int
    total_satellites: int
    inclination_deg: float
    phase_factor_F: int
    Omega0_deg: float
    u0_deg: float
    altitude_km: float
    theta_deg: float
    raan_layout_deg: list[float]
    validation_status: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["constellation_signature"] = self.constellation_signature
        payload["constellation_cache_key"] = self.constellation_cache_key
        return payload

    @property
    def constellation_signature(self) -> str:
        return CONSTELLATION_SIGNATURE

    @property
    def constellation_cache_key(self) -> str:
        payload = {
            "M": self.M,
            "N": self.N,
            "scenario": self.scenario,
            "inclination_deg": self.inclination_deg,
            "phase_factor_F": self.phase_factor_F,
            "Omega0_deg": self.Omega0_deg,
            "u0_deg": self.u0_deg,
            "source_file_hash": self.source_sha256,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_problem3_config(project_root: Path, mode: str) -> dict[str, Any]:
    path = project_root / "configs" / f"problem3_{mode}.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到问题三配置文件：{path}")
    config = load_json(path)
    if config.get("mode") != mode:
        raise ValueError(f"配置文件 mode={config.get('mode')!r} 与命令行 {mode!r} 不一致。")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearby_metadata(path: Path) -> dict[str, Any]:
    for parent in (path.parent, *path.parents):
        metadata = parent / "run_metadata.json"
        if metadata.exists():
            try:
                return load_json(metadata)
            except (OSError, ValueError, json.JSONDecodeError):
                return {}
        if parent.name == "outputs":
            break
    return {}


def _first(payload: dict[str, Any], metadata: dict[str, Any], *keys: str) -> Any:
    for source in (payload, metadata, metadata.get("config", {})):
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def _candidate_from_json(path: Path, project_root: Path) -> SelectedConstellation:
    payload = load_json(path)
    metadata = _nearby_metadata(path)
    required = {
        "M": _first(payload, metadata, "M"),
        "N": _first(payload, metadata, "N"),
        "inclination_deg": _first(payload, metadata, "inclination_deg"),
        "phase_factor_F": _first(payload, metadata, "phase_factor_F", "phase_factor"),
        "Omega0_deg": _first(payload, metadata, "Omega0_deg"),
        "u0_deg": _first(payload, metadata, "u0_deg"),
        "altitude_km": _first(payload, metadata, "altitude_km"),
        "theta_deg": _first(payload, metadata, "theta_deg"),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ConstellationSelectionError(
            f"星座文件 {path} 缺少必要字段：{', '.join(missing)}"
        )

    M = int(required["M"])
    N = int(required["N"])
    Omega0_deg = float(required["Omega0_deg"])
    raan_layout = payload.get("raan_layout_deg")
    if not raan_layout:
        raan_layout = [float((Omega0_deg + 360.0 * p / M) % 360.0) for p in range(M)]

    mode = str(_first(payload, metadata, "mode") or "").lower()
    if not mode and "standard" in str(path).lower():
        mode = "standard"
    scenario = str(_first(payload, metadata, "scenario") or "single").lower()
    validation_status = str(
        _first(payload, metadata, "validation_status", "status") or "unknown"
    )
    try:
        source_file = str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        source_file = str(path.resolve())

    return SelectedConstellation(
        source_file=source_file.replace("\\", "/"),
        source_sha256=_sha256(path),
        source_modified_utc=datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        mode=mode,
        scenario=scenario,
        M=M,
        N=N,
        total_satellites=int(_first(payload, metadata, "total_satellites") or M * N),
        inclination_deg=float(required["inclination_deg"]),
        phase_factor_F=int(required["phase_factor_F"]),
        Omega0_deg=Omega0_deg,
        u0_deg=float(required["u0_deg"]),
        altitude_km=float(required["altitude_km"]),
        theta_deg=float(required["theta_deg"]),
        raan_layout_deg=[float(value) for value in raan_layout],
        validation_status=validation_status,
    )


def _validation_preferred(status: str) -> bool:
    normalized = status.lower().replace("-", "_")
    return any(word in normalized for word in PREFERRED_VALIDATION_WORDS) and not any(
        word in normalized for word in ("failed", "not_passed", "unvalidated")
    )


def _selection_score(candidate: SelectedConstellation, path: Path) -> tuple[int, ...]:
    return (
        int(candidate.mode == "standard"),
        int(candidate.scenario == "double"),
        int(_validation_preferred(candidate.validation_status)),
        int("standard" in str(path).lower()),
        int(path.name.lower().startswith("best_double_constellation")),
    )


def validate_expected_double_constellation(
    selected: SelectedConstellation,
) -> SelectedConstellation:
    expected = (
        selected.mode == "standard"
        and selected.scenario == EXPECTED_SCENARIO
        and selected.M == EXPECTED_M
        and selected.N == EXPECTED_N
        and selected.total_satellites == EXPECTED_TOTAL_SATELLITES
    )
    if not expected:
        raise ConstellationSelectionError(
            "问题三必须采用 Problem 2 Standard 二重覆盖星座 "
            f"{CONSTELLATION_SIGNATURE}；实际读取 mode={selected.mode!r}, "
            f"scenario={selected.scenario!r}, M={selected.M}, N={selected.N}, "
            f"S={selected.total_satellites}，不允许退回单重覆盖方案。"
        )
    return selected


def discover_problem2_constellation(
    project_root: str | Path,
    explicit_file: str | Path | None = None,
) -> SelectedConstellation:
    """Select the unique Problem 2 Standard 40x46 double-cover result.

    Equal-ranked candidates are rejected rather than selected by filesystem order.
    """

    root = Path(project_root).resolve()
    if explicit_file is not None:
        path = Path(explicit_file)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"指定的星座文件不存在：{path}")
        return validate_expected_double_constellation(
            _candidate_from_json(path.resolve(), root)
        )

    output_root = root / "outputs" / "problem2"
    patterns = (
        "**/best_double_constellation*.json",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(output_root.glob(pattern))
    paths = sorted({path.resolve() for path in paths if path.is_file()})
    if not paths:
        raise ConstellationSelectionError(
            "未在 outputs/problem2 下找到问题二星座 JSON；请先提供 standard 输出，"
            "或使用 --constellation-file 显式指定。"
        )

    candidates: list[tuple[tuple[int, ...], Path, SelectedConstellation]] = []
    errors: list[str] = []
    for path in paths:
        try:
            candidate = _candidate_from_json(path, root)
            if (
                candidate.mode == "standard"
                and candidate.scenario == EXPECTED_SCENARIO
                and candidate.M == EXPECTED_M
                and candidate.N == EXPECTED_N
                and candidate.total_satellites == EXPECTED_TOTAL_SATELLITES
            ):
                candidates.append((_selection_score(candidate, path), path, candidate))
        except (OSError, ValueError, json.JSONDecodeError, ConstellationSelectionError) as exc:
            errors.append(f"{path}: {exc}")
    if not candidates:
        detail = "\n".join(errors)
        raise ConstellationSelectionError(
            "未找到符合 standard/double/M=40/N=46/S=1840 的问题二星座，"
            f"且不允许退回单重覆盖方案。候选读取信息：\n{detail}"
        )

    best_score = max(item[0] for item in candidates)
    top = [item for item in candidates if item[0] == best_score]
    signatures = {
        (
            item[2].M,
            item[2].N,
            round(item[2].inclination_deg, 10),
            item[2].phase_factor_F,
            round(item[2].Omega0_deg, 10),
            round(item[2].u0_deg, 10),
        )
        for item in top
    }
    if len(top) > 1 and len(signatures) > 1:
        listed = "\n".join(f"- {path}" for _, path, _ in top)
        raise ConstellationSelectionError(
            "发现多套同优先级问题二星座，无法安全自动选择。请使用 "
            f"--constellation-file 指定其中一个：\n{listed}"
        )
    return validate_expected_double_constellation(
        sorted(top, key=lambda item: str(item[1]))[0][2]
    )


def write_selected_constellation(
    selected: SelectedConstellation, output_dir: str | Path
) -> Path:
    path = Path(output_dir) / "selected_problem2_constellation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(selected.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
