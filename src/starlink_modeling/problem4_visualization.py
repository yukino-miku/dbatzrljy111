"""Chinese 300-dpi publication figures for Problem four."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .problem2_visualization import configure_chinese_font
from .problem4_collision import (
    collision_probability_exact,
    hard_body_radius_km,
    position_uncertainty_km,
)


def _style() -> None:
    configure_chinese_font()
    plt.rcParams.update(
        {
            "figure.figsize": (8.8, 5.8),
            "axes.grid": True,
            "grid.alpha": 0.24,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.unicode_minus": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_problem4_figures(output_dir: Path, config: dict[str, Any], total_satellites: int) -> None:
    _style()
    debris = pd.read_csv(output_dir / "debris_size_distribution.csv")
    fig, ax1 = plt.subplots()
    ax1.loglog(debris["diameter_m"], debris["diameter_pdf_per_m"], color="#2166ac", label="概率密度")
    ax1.set(xlabel="碎片直径 D / m", ylabel="概率密度 / (1/m)", title="碎片尺寸截断幂律分布")
    ax2 = ax1.twinx()
    ax2.semilogx(debris["diameter_m"], debris["survival_probability"], color="#b2182b", label="累计超越概率")
    ax2.set_ylabel("P(碎片直径 ≥ D)")
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    _save(fig, output_dir / "fig_problem4_debris_size_distribution.png")

    velocity = pd.read_csv(output_dir / "relative_velocity_distribution.csv")
    fig, ax = plt.subplots()
    ax.plot(velocity["velocity_km_s"], velocity["velocity_pdf_per_km_s"], color="#1b7837", lw=2)
    ax.fill_between(velocity["velocity_km_s"], velocity["velocity_pdf_per_km_s"], color="#a6dba0", alpha=0.55)
    ax.set(xlabel="相对速度 / (km/s)", ylabel="概率密度", title="候选交会相对速度截断正态分布")
    _save(fig, output_dir / "fig_problem4_relative_velocity_distribution.png")

    miss = np.linspace(0.0, 2.0, 600)
    hard = float(hard_body_radius_km(config["debris"]["D0_m"], config["satellite"]["effective_area_m2"]))
    fig, ax = plt.subplots()
    for warning in (6.0, 72.0, 168.0):
        sigma = float(position_uncertainty_km(warning, config["uncertainty"]["sigma0_km"], config["uncertainty"]["k_sigma_km_per_hour"]))
        ax.semilogy(miss, np.maximum(collision_probability_exact(miss, sigma, hard), 1e-14), label=f"预警 {warning:.0f} h")
    ax.axhline(config["avoidance"]["threshold"], color="#b2182b", ls="--", label="避撞阈值")
    ax.set(xlabel="预测最近距离 b / km", ylabel="单次预测碰撞概率", title="单次碰撞概率随预测最近距离变化", ylim=(1e-14, 1e-2))
    ax.legend()
    _save(fig, output_dir / "fig_problem4_collision_probability_vs_miss_distance.png")

    warning_values = np.linspace(6.0, 168.0, 500)
    sigma_values = position_uncertainty_km(warning_values, config["uncertainty"]["sigma0_km"], config["uncertainty"]["k_sigma_km_per_hour"])
    fig, ax = plt.subplots()
    for b in (0.05, 0.10, 0.25):
        ax.semilogy(warning_values, np.maximum(collision_probability_exact(b, sigma_values, hard), 1e-14), label=f"最近距离 {b:.2f} km")
    ax.axhline(config["avoidance"]["threshold"], color="#b2182b", ls="--", label="避撞阈值")
    ax.set(xlabel="预警时间 / h", ylabel="单次预测碰撞概率", title="预警时间、位置不确定性与碰撞概率")
    ax.legend()
    _save(fig, output_dir / "fig_problem4_collision_probability_vs_warning_time.png")

    consistency = _read_json(output_dir / "risk_model_consistency.json")
    fig, ax = plt.subplots()
    rates = np.array([consistency["flux_collision_rate_per_second"], consistency["conjunction_integrated_collision_rate_per_second"]])
    ax.bar(["通量×实体截面", "候选交会率×平均Pc"], rates, color=["#2166ac", "#d6604d"])
    ax.set(ylabel="碰撞危险率 / (1/s)", title="长期通量模型与候选交会模型一致性")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.text(0.5, max(rates) * 0.92, f"相对误差 {consistency['relative_consistency_error']:.2%}", ha="center")
    _save(fig, output_dir / "fig_problem4_flux_consistency.png")

    thresholds = pd.read_csv(output_dir / "single_satellite_threshold_comparison.csv").sort_values("threshold")
    xlabels = [f"{value:.0e}" for value in thresholds["threshold"]]
    fig, ax = plt.subplots()
    ax.plot(xlabels, thresholds["expected_annual_maneuvers"], marker="o", color="#2166ac")
    ax.set(xlabel="单次预测概率阈值 P_th", ylabel="单星年预期规避次数", title="避撞阈值对年度规避频次的影响")
    _save(fig, output_dir / "fig_problem4_threshold_vs_maneuvers.png")

    fig, ax = plt.subplots()
    ax.semilogy(xlabels, thresholds["annual_collision_probability_post"], marker="o", color="#b2182b")
    ax.set(xlabel="单次预测概率阈值 P_th", ylabel="避撞后年碰撞概率", title="避撞阈值对残余碰撞风险的影响")
    _save(fig, output_dir / "fig_problem4_threshold_vs_residual_risk.png")

    delta_v = pd.read_csv(output_dir / "maneuver_delta_v_distribution.csv")
    fig, ax = plt.subplots()
    if not delta_v.empty:
        ax.hist(delta_v["delta_v_actual_mps"], bins=45, color="#4393c3", alpha=0.82)
    ax.axvline(1.0, color="#b2182b", ls="--", label="1 m/s 上限")
    ax.set(xlabel="实际速度增量 Δv / (m/s)", ylabel="样本数", title="避撞机动速度增量分布")
    ax.legend()
    _save(fig, output_dir / "fig_problem4_delta_v_distribution.png")

    duration = pd.read_csv(output_dir / "maneuver_duration_distribution.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7))
    if not duration.empty:
        axes[0].hist(duration["burn_time_seconds"] / 60.0, bins=40, color="#4d9221", alpha=0.82)
        axes[1].hist(duration["degraded_duration_hours"], bins=40, color="#c51b7d", alpha=0.82)
    axes[0].set(xlabel="推进器点火时间 / min", ylabel="样本数", title="点火时间")
    axes[1].set(xlabel="完整业务降级时间 / h", ylabel="样本数", title="通信降级窗口")
    fig.suptitle("点火时间与通信业务降级时间的区分")
    _save(fig, output_dir / "fig_problem4_maneuver_duration_distribution.png")

    avoidances = pd.read_csv(output_dir / "annual_avoidance_distribution.csv")
    fig, ax = plt.subplots()
    ax.hist(avoidances["annual_avoidance_count"], bins=35, color="#8073ac", alpha=0.85)
    ax.axvline(avoidances["annual_avoidance_count"].mean(), color="#b2182b", ls="--", label="样本均值")
    ax.set(xlabel="星座年度总避撞次数", ylabel="模拟年份数", title=f"{total_satellites}星年度总避撞次数分布")
    ax.legend()
    _save(fig, output_dir / "fig_problem4_constellation_annual_avoidances.png")

    capacity = pd.read_csv(output_dir / "capacity_timeseries_sample.csv")
    fig, ax = plt.subplots()
    ax.plot(capacity["time_hours"] / 24.0, capacity["available_capacity_ratio"], color="#2166ac", lw=1.0)
    ax.set(xlabel="年内时间 / day", ylabel="接入容量可用比例", title="示例年份星座可用接入容量")
    _save(fig, output_dir / "fig_problem4_capacity_loss_time.png")

    fig, ax1 = plt.subplots()
    annual_cost = thresholds["expected_annual_maneuvers"] * total_satellites * config["costs"]["maneuver_cost_cny"] / 1e8
    ax1.plot(xlabels, annual_cost, marker="o", color="#2166ac", label="年度避撞成本")
    ax1.set(xlabel="单次预测概率阈值 P_th", ylabel="年度避撞成本 / 亿元")
    ax2 = ax1.twinx()
    ax2.semilogy(xlabels, thresholds["annual_collision_probability_post"], marker="s", color="#b2182b", label="残余年风险")
    ax2.set_ylabel("避撞后年碰撞概率")
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("避撞成本与残余风险权衡")
    _save(fig, output_dir / "fig_problem4_cost_risk_tradeoff.png")

    service = pd.read_csv(output_dir / "problem3_service_impact.csv")
    peak = service[service["demand_level"] == "peak"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].bar(peak["scenario"], peak["throughput_gbps"] / 1000.0, color="#4393c3")
    axes[1].bar(peak["scenario"], peak["service_ratio"], color="#d6604d")
    axes[0].set(ylabel="吞吐量 / Tbps", title="峰值时刻吞吐量")
    axes[1].set(ylabel="公平服务比例", title="峰值时刻服务比例")
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
    fig.suptitle("避撞、失效与备用接替对问题三业务的影响")
    _save(fig, output_dir / "fig_problem4_problem3_service_impact.png")

    summary = pd.read_csv(output_dir / "redundancy_monte_carlo_summary.csv")
    years = pd.read_csv(output_dir / "redundancy_year_samples.csv")
    representative = summary.sort_values("five_year_expected_cost_cny").groupby("scheme_type", as_index=False).first()
    ids = representative["candidate_id"].astype(int).tolist()
    labels = representative["scheme_type"].tolist()
    data = [years.loc[years["candidate_id"] == candidate, "effective_coverage_availability"].to_numpy() for candidate in ids]
    fig, ax = plt.subplots(figsize=(9.3, 5.8))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(0.99, color="#b2182b", ls="--", label="99%要求")
    ax.set(ylabel="全年基本覆盖可用率", title="不同冗余方案覆盖可用率分布")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    _save(fig, output_dir / "fig_problem4_coverage_availability_distribution.png")

    pareto = pd.read_csv(output_dir / "redundancy_pareto_front.csv")
    fig, ax = plt.subplots()
    for scheme, group in summary.groupby("scheme_type"):
        ax.scatter(group["five_year_expected_cost_cny"] / 1e8, group["q05_effective_coverage_availability"], s=25, alpha=0.65, label=scheme)
    ax.plot(pareto["five_year_expected_cost_cny"] / 1e8, pareto["q05_effective_coverage_availability"], color="black", lw=1.4, marker="o", label="Pareto前沿")
    ax.axhline(0.99, color="#b2182b", ls="--")
    ax.set(xlabel="五年预期增量成本 / 亿元", ylabel="覆盖可用率5%分位", title="冗余成本-覆盖可用率与Pareto前沿")
    ax.legend(ncol=2)
    _save(fig, output_dir / "fig_problem4_redundancy_cost_availability.png")

    fig, ax = plt.subplots()
    pure = summary[summary["scheme_type"].isin(["无冗余", "每轨在轨备用", "额外轨道面", "地面备用"])]
    pure = pure.sort_values("five_year_expected_cost_cny").groupby("scheme_type", as_index=False).first()
    ax.bar(pure["scheme_type"], pure["mean_effective_coverage_availability"], color=["#999999", "#2166ac", "#1b7837", "#d6604d"][:len(pure)])
    ax.axhline(0.99, color="#b2182b", ls="--", label="99%要求")
    ax.set(ylabel="平均全年基本覆盖可用率", title="三类纯冗余方案最低成本代表比较", ylim=(min(0.98, pure["mean_effective_coverage_availability"].min() - 0.002), 1.0002))
    ax.tick_params(axis="x", rotation=12)
    ax.legend()
    _save(fig, output_dir / "fig_problem4_redundancy_scheme_comparison.png")

    timeline = pd.read_csv(output_dir / "example_degradation_timeline.csv")
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.6), sharex=True)
    axes[0].plot(timeline["time_hours"] / 24.0, timeline["available_capacity_ratio"], color="#2166ac")
    axes[0].set(ylabel="容量可用比例", title="推荐方案典型年份容量状态")
    axes[1].step(timeline["time_hours"] / 24.0, timeline["basic_coverage_indicator"], where="post", color="#b2182b")
    axes[1].set(xlabel="年内时间 / day", ylabel="基本覆盖标志", title="故障、避撞与覆盖状态")
    _save(fig, output_dir / "fig_problem4_best_scheme_timeline.png")

    sensitivity = pd.read_csv(output_dir / "risk_sensitivity.csv")
    baseline_risk = _read_json(output_dir / "single_satellite_risk_summary.json")["annual_failure_probability"]
    tornado = sensitivity.groupby("parameter")["annual_failure_probability"].agg(["min", "max"])
    tornado["low"] = (tornado["min"] - baseline_risk) / max(baseline_risk, 1e-30)
    tornado["high"] = (tornado["max"] - baseline_risk) / max(baseline_risk, 1e-30)
    tornado = tornado.reindex(tornado[["low", "high"]].abs().max(axis=1).sort_values().index)
    fig, ax = plt.subplots()
    y = np.arange(len(tornado))
    ax.barh(y, tornado["low"], color="#4393c3", label="低端变化")
    ax.barh(y, tornado["high"], color="#d6604d", label="高端变化")
    ax.set(yticks=y, yticklabels=tornado.index, xlabel="年失效概率相对基准变化", title="关键风险参数敏感性龙卷风图")
    ax.axvline(0.0, color="black", lw=0.8)
    ax.legend()
    _save(fig, output_dir / "fig_problem4_sensitivity_tornado.png")
