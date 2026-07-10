"""Chinese publication-style figures for problem-two computed outputs."""

from __future__ import annotations

import math
from pathlib import Path
import warnings

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from .problem2_grid import GroundGrid
from .problem2_orbit import (
    R_EARTH_KM,
    WalkerConstellation,
    eci_to_ecef,
    orbital_period_seconds,
    propagate_constellation,
    satellite_subpoints,
)


FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
]


def configure_chinese_font() -> str | None:
    """Select an installed Chinese font and keep minus signs visible."""

    installed = {font.name for font in fm.fontManager.ttflist}
    selected = next((font for font in FONT_CANDIDATES if font in installed), None)
    if selected is None:
        warnings.warn(
            "未找到常用中文字体，图中中文可能显示为方框；程序将继续运行。",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def _style() -> None:
    configure_chinese_font()
    plt.rcParams.update(
        {
            "figure.figsize": (8.4, 5.6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_region_grid(grid: GroundGrid, output_dir: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    scatter = ax.scatter(
        grid.longitude_deg,
        grid.latitude_deg,
        c=grid.weights,
        s=10,
        cmap="viridis",
        linewidths=0,
    )
    ax.plot([73, 135, 135, 73, 73], [4, 4, 53, 53, 4], color="#c0392b", lw=1.8)
    ax.set(xlabel="经度 / °E", ylabel="纬度 / °N", title="问题二目标区域与地面评价网格")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("归一化面积权重")
    _save(fig, output_dir / "fig_problem2_region_grid.png")


def plot_walker_layout_3d(
    constellation: WalkerConstellation, output_dir: Path
) -> None:
    _style()
    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    sphere_u = np.linspace(0, 2 * np.pi, 80)
    sphere_v = np.linspace(0, np.pi, 40)
    x = R_EARTH_KM * np.outer(np.cos(sphere_u), np.sin(sphere_v))
    y = R_EARTH_KM * np.outer(np.sin(sphere_u), np.sin(sphere_v))
    z = R_EARTH_KM * np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
    ax.plot_surface(x, y, z, color="#8ecae6", alpha=0.28, linewidth=0)

    u = np.linspace(0.0, 2.0 * np.pi, 240)
    a = R_EARTH_KM + constellation.altitude_km
    cos_i = math.cos(constellation.inclination_rad)
    sin_i = math.sin(constellation.inclination_rad)
    plane_raans = constellation.raan_rad.reshape(constellation.M, constellation.N)[:, 0]
    for raan in plane_raans:
        orbit_x = a * (math.cos(raan) * np.cos(u) - math.sin(raan) * np.sin(u) * cos_i)
        orbit_y = a * (math.sin(raan) * np.cos(u) + math.cos(raan) * np.sin(u) * cos_i)
        orbit_z = a * np.sin(u) * sin_i
        ax.plot(orbit_x, orbit_y, orbit_z, color="#5b6770", lw=0.55, alpha=0.55)

    positions = propagate_constellation(constellation, 0.0)[0]
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        s=7, c=constellation.plane_index, cmap="turbo", alpha=0.85,
    )
    limit = a * 1.08
    ax.set(
        xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
        xlabel="ECI x / km", ylabel="ECI y / km", zlabel="ECI z / km",
        title=f"Walker-Delta 星座三维布局（{constellation.M}×{constellation.N}）",
    )
    ax.set_box_aspect((1, 1, 1))
    _save(fig, output_dir / "fig_problem2_walker_layout_3d.png")


def plot_ground_tracks(
    constellation: WalkerConstellation, scenario: str, output_dir: Path
) -> None:
    _style()
    periods = 2.0
    times = np.linspace(0.0, periods * orbital_period_seconds(), 520)
    eci = propagate_constellation(constellation, times)
    ecef = eci_to_ecef(eci, times)
    latitude, longitude = satellite_subpoints(ecef)
    lat_deg = np.degrees(latitude)
    lon_deg = np.degrees(longitude)

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    stride = max(1, constellation.N // 4)
    selected = np.flatnonzero(constellation.satellite_index % stride == 0)
    for satellite_index in selected:
        lon = lon_deg[:, satellite_index]
        lat = lat_deg[:, satellite_index]
        jumps = np.flatnonzero(np.abs(np.diff(lon)) > 180.0) + 1
        for segment in np.split(np.arange(times.size), jumps):
            ax.plot(lon[segment], lat[segment], lw=0.45, alpha=0.38, color="#2878b5")
    ax.fill_between([73, 135], 4, 53, color="#f4a261", alpha=0.24, label="目标区域")
    ax.plot([73, 135, 135, 73, 73], [4, 4, 53, 53, 4], color="#c0392b", lw=1.5)
    scenario_cn = "单重方案" if scenario == "single" else "二重方案"
    ax.set(
        xlim=(-180, 180), ylim=(-90, 90),
        xlabel="经度 / °", ylabel="纬度 / °",
        title=f"{scenario_cn}多轨道面星下点轨迹",
    )
    ax.legend(loc="lower left")
    _save(fig, output_dir / f"fig_problem2_ground_tracks_{scenario}.png")


def plot_ga_convergence(history: pd.DataFrame, scenario: str, output_dir: Path) -> None:
    _style()
    if history.empty:
        return
    grouped = history.groupby("generation", as_index=False).agg(
        best_satellites=("global_best_total_satellites", "min"),
        best_violation=("global_best_constraint_violation", "min"),
        feasible_count=("feasible_count", "max"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), sharex=True)
    axes[0].step(grouped["generation"], grouped["best_satellites"], where="post", color="#2878b5")
    axes[0].set_ylabel("最优卫星数")
    axes[1].semilogy(
        grouped["generation"], np.maximum(grouped["best_violation"], 1e-12), color="#c0392b"
    )
    axes[1].set_ylabel("最优约束违反度")
    axes[2].plot(grouped["generation"], grouped["feasible_count"], color="#2a9d8f")
    axes[2].set(xlabel="遗传代数", ylabel="可行个体数")
    scenario_cn = "单重覆盖" if scenario == "single" else "二重覆盖"
    fig.suptitle(f"{scenario_cn}遗传算法收敛过程", y=0.995)
    _save(fig, output_dir / f"fig_problem2_ga_convergence_{scenario}.png")


def _plot_grid_field(
    grid_metrics: pd.DataFrame,
    value_column: str,
    title: str,
    colorbar_label: str,
    path: Path,
    *,
    cmap: str = "viridis",
) -> None:
    _style()
    x = grid_metrics["longitude_deg"].to_numpy()
    y = grid_metrics["latitude_deg"].to_numpy()
    z = grid_metrics[value_column].to_numpy()
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    if len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3 and np.ptp(z) > 1e-14:
        triangulation = mtri.Triangulation(x, y)
        artist = ax.tricontourf(triangulation, z, levels=18, cmap=cmap)
    else:
        artist = ax.scatter(x, y, c=z, cmap=cmap, s=14)
    ax.plot([73, 135, 135, 73, 73], [4, 4, 53, 53, 4], color="black", lw=0.9)
    ax.set(xlabel="经度 / °E", ylabel="纬度 / °N", title=title)
    colorbar = fig.colorbar(artist, ax=ax)
    colorbar.set_label(colorbar_label)
    _save(fig, path)


def plot_solution_grid_fields(
    grid_metrics: pd.DataFrame, scenario: str, output_dir: Path
) -> None:
    scenario_cn = "单重方案" if scenario == "single" else "二重方案"
    availability_column = "availability1" if scenario == "single" else "availability2"
    availability_cn = "单重可用率" if scenario == "single" else "二重可用率"
    _plot_grid_field(
        grid_metrics, availability_column,
        f"{scenario_cn}各网格点{availability_cn}", availability_cn,
        output_dir / f"fig_problem2_{scenario}_availability_heatmap.png",
    )
    _plot_grid_field(
        grid_metrics, "mean_multiplicity",
        f"{scenario_cn}平均覆盖重数空间分布", "平均覆盖重数",
        output_dir / f"fig_problem2_{scenario}_mean_multiplicity_heatmap.png",
        cmap="plasma",
    )
    gap_column = "max_gap_time_min" if scenario == "single" else "max_double_gap_time_min"
    gap_cn = "单重覆盖最大间隙 / min" if scenario == "single" else "二重覆盖最大间隙 / min"
    _plot_grid_field(
        grid_metrics, gap_column,
        f"{scenario_cn}最大覆盖间隙空间分布", gap_cn,
        output_dir / f"fig_problem2_{scenario}_max_gap_heatmap.png",
        cmap="magma",
    )


def plot_area_coverage_time(
    time_metrics: pd.DataFrame, scenario: str, output_dir: Path
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    ax.plot(time_metrics["time_hours"], time_metrics["A1_area_coverage"], label="瞬时单重面积覆盖率 A1(t)")
    ax.plot(time_metrics["time_hours"], time_metrics["A2_area_coverage"], label="瞬时二重面积覆盖率 A2(t)")
    ax.step(
        time_metrics["time_hours"], time_metrics["B1_full_region"],
        where="post", color="#c0392b", alpha=0.65, label="全区域单重覆盖标志 B1(t)",
    )
    ax.set(xlabel="仿真时间 / h", ylabel="覆盖率或覆盖标志", ylim=(-0.03, 1.04))
    scenario_cn = "单重方案" if scenario == "single" else "二重方案"
    ax.set_title(f"{scenario_cn}区域覆盖性能随时间变化")
    ax.legend(loc="lower right")
    _save(fig, output_dir / f"fig_problem2_{scenario}_area_coverage_time.png")


def plot_worst_snapshot(
    grid: GroundGrid,
    counts: np.ndarray,
    scenario: str,
    output_dir: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "latitude_deg": grid.latitude_deg,
            "longitude_deg": grid.longitude_deg,
            "coverage_multiplicity": counts,
        }
    )
    _plot_grid_field(
        frame,
        "coverage_multiplicity",
        ("单重方案" if scenario == "single" else "二重方案") + "最差时刻覆盖重数",
        "覆盖重数",
        output_dir / f"fig_problem2_worst_snapshot_{scenario}.png",
        cmap="turbo",
    )


def plot_mn_candidates(candidates: pd.DataFrame, output_dir: Path) -> None:
    _style()
    if candidates.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    values = np.log10(1.0 + candidates["constraint_violation"].clip(lower=0.0))
    sizes = 18.0 + 75.0 * candidates["total_satellites"] / candidates["total_satellites"].max()
    scatter = ax.scatter(
        candidates["M"], candidates["N"], c=values, s=sizes,
        cmap="viridis_r", alpha=0.68, edgecolors="white", linewidths=0.25,
    )
    ax.set(xlabel="轨道面数 M", ylabel="每轨卫星数 N", title="遗传算法 M-N 候选构型分布")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("log10(1+约束违反度)")
    _save(fig, output_dir / "fig_problem2_M_N_candidates.png")


def plot_cost_comparison(costs: pd.DataFrame, output_dir: Path) -> None:
    _style()
    if costs.empty:
        return
    labels = costs["scenario_cn"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.5))
    colors = ["#2878b5", "#e76f51"][: len(costs)]
    axes[0].bar(labels, costs["total_satellites"], color=colors)
    axes[0].set_ylabel("卫星数 / 颗")
    axes[1].bar(labels, costs["launch_count"], color=colors)
    axes[1].set_ylabel("发射次数 / 次")
    axes[2].bar(labels, costs["total_cost_100m_yuan"], color=colors)
    axes[2].set_ylabel("总部署成本 / 亿元")
    fig.suptitle("单重与二重覆盖方案规模及成本比较", y=1.02)
    _save(fig, output_dir / "fig_problem2_single_vs_double_cost.png")


def plot_resolution_sensitivity(sensitivity: pd.DataFrame, output_dir: Path) -> None:
    _style()
    completed = sensitivity[sensitivity["status"] == "completed"].copy()
    if completed.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for scenario, group in completed.groupby("scenario"):
        label = "单重方案" if scenario == "single" else "二重方案"
        x = np.arange(len(group))
        axes[0].plot(x, group["Q1_full_region"], marker="o", label=label)
        axes[1].plot(x, group["Q2_full_region"], marker="o", label=label)
    axes[0].set(title="全区域单重覆盖时间比例", xlabel="敏感性算例序号", ylabel="Q1")
    axes[1].set(title="全区域二重覆盖时间比例", xlabel="敏感性算例序号", ylabel="Q2")
    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
        ax.legend()
    fig.suptitle("网格、时间步长和仿真时长敏感性", y=1.02)
    _save(fig, output_dir / "fig_problem2_resolution_sensitivity.png")

