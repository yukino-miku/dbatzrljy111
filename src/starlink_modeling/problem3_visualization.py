"""Chinese publication figures for problem-three computed results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pandas as pd

from .problem2_grid import GroundGrid
from .problem2_orbit import (
    R_EARTH_KM,
    WalkerConstellation,
    eci_to_ecef,
    satellite_subpoints,
)
from .problem2_visualization import configure_chinese_font
from .problem3_topology import D_ISL_MAX_KM, TopologySnapshot


def _style() -> None:
    configure_chinese_font()
    plt.rcParams.update(
        {
            "figure.figsize": (8.8, 5.8),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.unicode_minus": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _earth_surface(ax) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    x = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH_KM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#8ecae6", alpha=0.25, linewidth=0)


def _edge_segments(positions: np.ndarray, edges: pd.DataFrame) -> np.ndarray:
    if edges.empty:
        return np.empty((0, 2, 3))
    source = edges["source_satellite"].to_numpy(int)
    target = edges["target_satellite"].to_numpy(int)
    return np.stack((positions[source], positions[target]), axis=1)


def plot_topology_3d_snapshot(
    constellation: WalkerConstellation,
    snapshot: TopologySnapshot,
    output_dir: Path,
) -> None:
    _style()
    fig = plt.figure(figsize=(9.0, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    _earth_surface(ax)
    positions = snapshot.positions_eci_km
    same = snapshot.edges[snapshot.edges["link_type"] == "same_plane"]
    cross = snapshot.edges[snapshot.edges["link_type"] == "cross_plane"]
    ax.add_collection3d(
        Line3DCollection(_edge_segments(positions, same), colors="#2878b5", linewidths=0.28, alpha=0.35)
    )
    ax.add_collection3d(
        Line3DCollection(_edge_segments(positions, cross), colors="#e76f51", linewidths=0.38, alpha=0.48)
    )
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=constellation.plane_index, cmap="turbo", s=4, alpha=0.85,
    )
    limit = (R_EARTH_KM + constellation.altitude_km) * 1.08
    ax.set(
        xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
        xlabel="ECI x / km", ylabel="ECI y / km", zlabel="ECI z / km",
        title=f"问题三星间链路三维拓扑快照（t={snapshot.time_seconds / 60:.1f} min）",
    )
    ax.set_box_aspect((1, 1, 1))
    ax.text2D(0.02, 0.96, "蓝色：同轨链路；红色：跨轨链路", transform=ax.transAxes)
    _save(fig, output_dir / "fig_problem3_topology_3d_snapshot.png")


def plot_topology_ground_projection(snapshot: TopologySnapshot, output_dir: Path) -> None:
    _style()
    ecef = eci_to_ecef(snapshot.positions_eci_km, snapshot.time_seconds)[0]
    latitude, longitude = satellite_subpoints(ecef)
    lon = np.degrees(longitude)
    lat = np.degrees(latitude)
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    for link_type, color, width, alpha in (
        ("same_plane", "#2878b5", 0.3, 0.25),
        ("cross_plane", "#e76f51", 0.4, 0.40),
    ):
        frame = snapshot.edges[snapshot.edges["link_type"] == link_type]
        segments = []
        for row in frame.itertuples(index=False):
            source, target = int(row.source_satellite), int(row.target_satellite)
            if abs(lon[source] - lon[target]) <= 180.0:
                segments.append([[lon[source], lat[source]], [lon[target], lat[target]]])
        if segments:
            ax.add_collection(LineCollection(segments, colors=color, linewidths=width, alpha=alpha))
    ax.scatter(lon, lat, s=3, c="#30343f", alpha=0.65)
    ax.plot([73, 135, 135, 73, 73], [4, 4, 53, 53, 4], color="#2a9d8f", lw=1.6)
    ax.set(
        xlim=(-180, 180), ylim=(-90, 90), xlabel="经度 / °", ylabel="纬度 / °",
        title="星下点与星间链路的地面投影",
    )
    _save(fig, output_dir / "fig_problem3_topology_ground_projection.png")


def plot_same_plane_distance(
    summary: dict[str, object], link_timeseries: pd.DataFrame, output_dir: Path
) -> None:
    _style()
    frame = link_timeseries[link_timeseries["link_type"] == "same_plane"]
    grouped = frame.groupby("time_seconds", as_index=False)["distance_km"].mean()
    analytic = float(summary["same_plane_distance_analytic_km"])
    fig, ax = plt.subplots()
    ax.plot(grouped["time_seconds"] / 60.0, grouped["distance_km"], color="#2878b5", label="三维坐标数值均值")
    ax.axhline(analytic, color="#c0392b", ls="--", label=f"解析值 {analytic:.3f} km")
    ax.set(xlabel="时间 / min", ylabel="同轨相邻卫星距离 / km", title="同轨相邻链路距离的解析与数值校核")
    ax.legend()
    _save(fig, output_dir / "fig_problem3_same_plane_distance.png")


def plot_cross_plane_distance_time(link_timeseries: pd.DataFrame, output_dir: Path) -> None:
    _style()
    frame = link_timeseries[link_timeseries["link_type"] == "cross_plane"].copy()
    counts = frame.groupby(["source_satellite", "target_satellite"]).size()
    source, target = counts.idxmax()
    selected = frame[
        (frame["source_satellite"] == source) & (frame["target_satellite"] == target)
    ].sort_values("time_seconds")
    fig, ax = plt.subplots()
    ax.plot(selected["time_seconds"] / 60.0, selected["distance_km"], marker="o", ms=2.5, lw=1.0, label=f"代表链路 {source}-{target}")
    ax.axhline(D_ISL_MAX_KM, color="#c0392b", ls="--", label="5000 km 通信阈值")
    ax.set(xlabel="时间 / min", ylabel="跨轨链路距离 / km", title="代表性跨轨链路距离随时间变化")
    ax.legend()
    _save(fig, output_dir / "fig_problem3_cross_plane_distance_time.png")


def plot_network_connectivity_time(topology: pd.DataFrame, output_dir: Path) -> None:
    _style()
    time = topology["time_seconds"] / 60.0
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 8.8), sharex=True)
    axes[0].plot(time, topology["edge_count"], color="#2878b5")
    axes[0].set_ylabel("边数")
    axes[1].plot(time, topology["average_degree"], color="#2a9d8f")
    axes[1].set_ylabel("平均度")
    axes[2].step(time, topology["connected_component_count"], where="post", color="#e76f51")
    axes[2].set_ylabel("连通分量数")
    axes[3].step(time, topology["is_connected"].astype(int), where="post", color="#6a4c93")
    axes[3].set(xlabel="时间 / min", ylabel="全网连通", yticks=[0, 1])
    fig.suptitle("星间网络连通性随时间变化", y=0.995)
    _save(fig, output_dir / "fig_problem3_network_connectivity_time.png")


def plot_link_availability_distribution(availability: pd.DataFrame, output_dir: Path) -> None:
    _style()
    fig, ax = plt.subplots()
    for link_type, label, color in (
        ("same_plane", "同轨链路", "#2878b5"),
        ("cross_plane", "跨轨链路", "#e76f51"),
    ):
        values = availability.loc[availability["link_type"] == link_type, "availability_ratio"]
        ax.hist(values, bins=np.linspace(0, 1, 21), alpha=0.55, label=label, color=color)
    ax.set(xlabel="链路可用率", ylabel="候选链路数量", title="星间链路可用率分布")
    ax.legend()
    _save(fig, output_dir / "fig_problem3_link_availability_distribution.png")


def plot_crosslink_switches(shifts: pd.DataFrame, output_dir: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0))
    selected_planes = sorted(shifts["left_plane"].unique())[:6]
    for plane in selected_planes:
        frame = shifts[shifts["left_plane"] == plane].sort_values("time_seconds")
        axes[0].step(frame["time_seconds"] / 60.0, frame["kappa"], where="post", label=f"轨道面对 {plane}-{(plane + 1)}")
    axes[0].set(xlabel="时间 / min", ylabel="循环偏移 κ", title="代表性相邻轨道面对的循环移位")
    axes[0].legend(ncol=2)
    switch_count = (
        shifts.sort_values(["left_plane", "time_seconds"])
        .groupby("left_plane")["kappa"]
        .apply(lambda values: np.count_nonzero(np.diff(values.to_numpy())))
    )
    axes[1].bar(switch_count.index, switch_count.values, color="#e76f51")
    axes[1].set(xlabel="左侧轨道面编号", ylabel="偏移切换次数", title="各相邻轨道面对的匹配切换次数")
    _save(fig, output_dir / "fig_problem3_crosslink_switches.png")


def plot_topology_figures(
    constellation: WalkerConstellation,
    snapshots: list[TopologySnapshot],
    summary: dict[str, object],
    topology_timeseries: pd.DataFrame,
    link_timeseries: pd.DataFrame,
    link_availability: pd.DataFrame,
    shift_timeseries: pd.DataFrame,
    output_dir: Path,
) -> None:
    representative = snapshots[int(np.argmin(topology_timeseries["largest_component_ratio"]))]
    plot_topology_3d_snapshot(constellation, representative, output_dir)
    plot_topology_ground_projection(representative, output_dir)
    plot_same_plane_distance(summary, link_timeseries, output_dir)
    plot_cross_plane_distance_time(link_timeseries, output_dir)
    plot_network_connectivity_time(topology_timeseries, output_dir)
    plot_link_availability_distribution(link_availability, output_dir)
    plot_crosslink_switches(shift_timeseries, output_dir)


def plot_example_route_3d(
    worst: dict[str, object], snapshot: TopologySnapshot, positions_ecef: np.ndarray, output_dir: Path
) -> None:
    if not worst.get("reachable"):
        return
    _style()
    path = json.loads(str(worst["satellite_path"]))
    source_lat, source_lon = math.radians(float(worst["source_lat"])), math.radians(float(worst["source_lon"]))
    dest_lat, dest_lon = math.radians(float(worst["destination_lat"])), math.radians(float(worst["destination_lon"]))
    ground = np.array(
        [
            [math.cos(source_lat) * math.cos(source_lon), math.cos(source_lat) * math.sin(source_lon), math.sin(source_lat)],
            [math.cos(dest_lat) * math.cos(dest_lon), math.cos(dest_lat) * math.sin(dest_lon), math.sin(dest_lat)],
        ]
    ) * R_EARTH_KM
    route_positions = np.vstack((ground[0], positions_ecef[path], ground[1]))
    fig = plt.figure(figsize=(8.8, 7.3))
    ax = fig.add_subplot(111, projection="3d")
    _earth_surface(ax)
    ax.plot(route_positions[:, 0], route_positions[:, 1], route_positions[:, 2], color="#c0392b", lw=2.0, marker="o", ms=4)
    ax.scatter(ground[:, 0], ground[:, 1], ground[:, 2], color="#2a9d8f", s=38)
    limit = np.linalg.norm(positions_ecef[0]) * 1.08
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit), xlabel="ECEF x / km", ylabel="ECEF y / km", zlabel="ECEF z / km", title="最大时延样本的端到端最小时延路径")
    ax.set_box_aspect((1, 1, 1))
    _save(fig, output_dir / "fig_problem3_example_route_3d.png")


def plot_example_route_map(worst: dict[str, object], positions_ecef: np.ndarray, output_dir: Path) -> None:
    if not worst.get("reachable"):
        return
    _style()
    path = json.loads(str(worst["satellite_path"]))
    lat, lon = satellite_subpoints(positions_ecef[path])
    route_lon = [float(worst["source_lon"]), *np.degrees(lon), float(worst["destination_lon"])]
    route_lat = [float(worst["source_lat"]), *np.degrees(lat), float(worst["destination_lat"])]
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.plot(route_lon, route_lat, color="#c0392b", marker="o", ms=4, lw=1.5)
    ax.plot([73, 135, 135, 73, 73], [4, 4, 53, 53, 4], color="#2a9d8f", lw=1.4)
    ax.set(xlabel="经度 / °E", ylabel="纬度 / °N", title="最大时延样本路径的地面投影")
    _save(fig, output_dir / "fig_problem3_example_route_map.png")


def plot_routing_figures(
    routes: pd.DataFrame,
    time_metrics: pd.DataFrame,
    worst: dict[str, object],
    worst_snapshot: TopologySnapshot,
    worst_positions_ecef: np.ndarray,
    heatmap: pd.DataFrame,
    output_dir: Path,
) -> None:
    reachable = routes[routes["reachable"]].copy()
    if reachable.empty:
        return
    plot_example_route_3d(worst, worst_snapshot, worst_positions_ecef, output_dir)
    plot_example_route_map(worst, worst_positions_ecef, output_dir)
    _style()
    delays = np.sort(reachable["total_delay_ms"].to_numpy(float))
    fig, ax = plt.subplots()
    ax.plot(delays, np.arange(1, len(delays) + 1) / len(delays), color="#2878b5")
    ax.axvline(30.0, color="#c0392b", ls="--", label="30 ms 阈值")
    ax.set(xlabel="端到端总时延 / ms", ylabel="累计概率", title="端到端最小时延累计分布")
    ax.legend()
    _save(fig, output_dir / "fig_problem3_delay_cdf.png")

    _style()
    fig, ax = plt.subplots()
    ax.plot(time_metrics["time_seconds"] / 3600.0, time_metrics["mean_delay_ms"], label="平均时延")
    ax.plot(time_metrics["time_seconds"] / 3600.0, time_metrics["p95_delay_ms"], label="95% 分位时延")
    ax.plot(time_metrics["time_seconds"] / 3600.0, time_metrics["max_delay_ms"], label="最大时延")
    ax.axhline(30.0, color="#c0392b", ls="--", label="30 ms 阈值")
    ax.set(xlabel="仿真时间 / h", ylabel="时延 / ms", title="区域点对时延随时间变化")
    ax.legend()
    _save(fig, output_dir / "fig_problem3_delay_time.png")

    _style()
    fig, ax = plt.subplots()
    components = [reachable["propagation_delay_ms"].mean(), reachable["processing_delay_ms"].mean()]
    ax.bar(["传播时延", "处理时延"], components, color=["#2878b5", "#e76f51"])
    ax.set(ylabel="平均时延 / ms", title="端到端时延构成")
    _save(fig, output_dir / "fig_problem3_delay_components.png")

    _style()
    fig, ax = plt.subplots()
    scatter = ax.scatter(reachable["ground_distance_km"], reachable["total_delay_ms"], c=reachable["ISL_hop_count"], s=18, cmap="viridis", alpha=0.7)
    ax.set(xlabel="地面点大圆距离 / km", ylabel="端到端总时延 / ms", title="地面距离与端到端时延关系")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("ISL 跳数")
    _save(fig, output_dir / "fig_problem3_delay_vs_ground_distance.png")

    if not heatmap.empty:
        _style()
        fig, ax = plt.subplots(figsize=(9.0, 5.8))
        scatter = ax.scatter(heatmap["destination_lon"], heatmap["destination_lat"], c=heatmap["total_delay_ms"], cmap="magma", s=26)
        ax.set(xlabel="目的地经度 / °E", ylabel="目的地纬度 / °N", title="选定源点到区域网格的最小时延")
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("端到端总时延 / ms")
        _save(fig, output_dir / "fig_problem3_delay_heatmap_from_source.png")


def plot_traffic_figures(
    traffic: pd.DataFrame,
    satellite_loads: pd.DataFrame,
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output_dir: Path,
) -> None:
    if traffic.empty:
        return
    hours = traffic["time_seconds"] / 3600.0
    _style()
    fig, ax = plt.subplots()
    ax.plot(hours, traffic["actual_region_demand_gbps"], label="区域实际总流量")
    ax.plot(hours, traffic["satellite_demand_gbps"], label="进入卫星网络的需求")
    ax.set(xlabel="时间 / h", ylabel="流量 / Gbps", title="区域流量与卫星承载流量曲线")
    ax.legend()
    _save(fig, output_dir / "fig_problem3_traffic_profile.png")

    _style()
    fig, ax = plt.subplots()
    ax.plot(hours, traffic["satellite_demand_gbps"], label="需求")
    ax.plot(hours, traffic["optimized_throughput_gbps"], label="成功吞吐量")
    ax.plot(hours, traffic["optimized_blocked_gbps"], label="阻塞流量")
    ax.set(xlabel="时间 / h", ylabel="流量 / Gbps", title="需求、吞吐量与阻塞流量")
    ax.legend()
    _save(fig, output_dir / "fig_problem3_throughput_and_demand.png")

    _style()
    fig, ax = plt.subplots()
    ax.plot(hours, traffic["optimized_service_ratio"], color="#2a9d8f")
    ax.set(xlabel="时间 / h", ylabel="公平服务比例 r(t)", ylim=(0, 1.03), title="公平服务比例随时间变化")
    _save(fig, output_dir / "fig_problem3_service_ratio_time.png")

    if not satellite_loads.empty:
        pivot = satellite_loads.pivot(index="time_seconds", columns="satellite_id", values="utilization").fillna(0.0)
        _style()
        fig, ax = plt.subplots(figsize=(10.0, 5.5))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=max(1.0, float(pivot.to_numpy().max())))
        ax.set(xlabel="卫星编号", ylabel="时间快照序号", title="时间—卫星接入利用率热力图")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("接入容量利用率")
        _save(fig, output_dir / "fig_problem3_satellite_utilization_heatmap.png")

    _style()
    fig, ax = plt.subplots()
    ax.plot(hours, traffic["optimized_maximum_utilization"], label="最大利用率")
    ax.plot(hours, traffic["optimized_mean_utilization"], label="平均利用率")
    ax.set(xlabel="时间 / h", ylabel="接入容量利用率", title="卫星最大与平均利用率")
    ax.legend()
    _save(fig, output_dir / "fig_problem3_max_utilization_time.png")

    if not comparison.empty:
        _style()
        metrics = ["mean_throughput_gbps", "maximum_utilization", "mean_delay_ms"]
        labels = ["平均吞吐量 / Gbps", "最大利用率", "平均接入代价 / ms"]
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2))
        for axis, metric, label in zip(axes, metrics, labels):
            axis.bar(comparison["strategy_cn"], comparison[metric], color=["#8d99ae", "#2a9d8f"])
            axis.set_title(label, fontsize=10)
        fig.suptitle("最近卫星基准与优化接入方案对比", y=1.02)
        _save(fig, output_dir / "fig_problem3_baseline_vs_optimized.png")

    if not sensitivity.empty:
        _style()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        x = sensitivity["eta_sat"] * 100.0
        line1 = ax1.plot(x, sensitivity["service_ratio"], marker="o", color="#2878b5", label="服务满足率")
        line2 = ax2.plot(x, sensitivity["maximum_utilization"], marker="s", color="#e76f51", label="最大利用率")
        ax1.set(xlabel="卫星业务承载比例 η_sat / %", ylabel="服务满足率", title="卫星承载比例敏感性")
        ax2.set_ylabel("最大卫星利用率")
        ax1.legend(line1 + line2, [line.get_label() for line in line1 + line2], loc="best")
        _save(fig, output_dir / "fig_problem3_traffic_share_sensitivity.png")

