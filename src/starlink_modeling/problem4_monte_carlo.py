"""Vectorized conjunction-event Monte Carlo for Problem four."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any

import numpy as np
import pandas as pd

from .problem4_avoidance import AvoidanceResult, apply_avoidance_policy
from .problem4_capacity import degraded_duration_hours
from .problem4_collision import (
    collision_cross_section_km2,
    collision_probability_exact,
    collision_probability_small_circle,
    conjunction_rate_per_second,
    expected_collision_cross_section_km2,
    flux_collision_rate_per_second,
    hard_body_radius_km,
    poisson_at_least_one_probability,
    position_uncertainty_km,
)
from .problem4_debris import (
    SECONDS_PER_YEAR,
    relative_velocity_mean_km_s,
    sample_miss_distance_km,
    sample_relative_velocity_km_s,
    sample_truncated_power_law,
    sample_warning_hours,
)


@dataclass
class RiskSimulation:
    summary: dict[str, Any]
    consistency: dict[str, Any]
    threshold_comparison: pd.DataFrame
    sensitivity: pd.DataFrame
    event_samples: pd.DataFrame
    maneuver_delta_v: pd.DataFrame
    maneuver_duration: pd.DataFrame
    arrays: dict[str, np.ndarray]


def _annual_metrics(
    pre: np.ndarray,
    avoidance: AvoidanceResult,
    mu_conjunction: float,
    q_fail: float,
    degraded_hours: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    lambda_pre = float(mu_conjunction * np.mean(pre))
    lambda_post = float(mu_conjunction * np.mean(avoidance.collision_probability_post))
    lambda_fail = float(mu_conjunction * q_fail * np.mean(avoidance.collision_probability_post))
    p_pre = float(poisson_at_least_one_probability(lambda_pre))
    p_post = float(poisson_at_least_one_probability(lambda_post))
    p_fail = float(poisson_at_least_one_probability(lambda_fail))
    executed = avoidance.maneuver_executed
    requested = avoidance.maneuver_requested
    delta_v = avoidance.delta_v_actual_mps[executed]
    durations = degraded_hours[executed]
    return {
        "threshold": float(threshold),
        "mean_annual_conjunctions": float(mu_conjunction),
        "expected_annual_maneuver_requests": float(mu_conjunction * np.mean(requested)),
        "expected_annual_maneuvers": float(mu_conjunction * np.mean(executed)),
        "expected_annual_late_warnings": float(mu_conjunction * np.mean(avoidance.late_warning)),
        "expected_annual_delta_v_limited": float(mu_conjunction * np.mean(avoidance.delta_v_limited)),
        "expected_annual_undetected_events": float(mu_conjunction * np.mean(~avoidance.detected)),
        "annual_collision_hazard_pre": lambda_pre,
        "annual_collision_hazard_post": lambda_post,
        "annual_failure_hazard": lambda_fail,
        "annual_collision_probability_pre": p_pre,
        "annual_collision_probability_post": p_post,
        "annual_failure_probability": p_fail,
        "risk_reduction_ratio": 0.0 if p_pre <= 0.0 else 1.0 - p_post / p_pre,
        "mean_delta_v_mps": float(np.mean(delta_v)) if delta_v.size else 0.0,
        "p95_delta_v_mps": float(np.quantile(delta_v, 0.95)) if delta_v.size else 0.0,
        "mean_degraded_duration_hours": float(np.mean(durations)) if durations.size else 0.0,
        "p95_degraded_duration_hours": float(np.quantile(durations, 0.95)) if durations.size else 0.0,
        "late_warning_ratio_among_requests": float(np.mean(avoidance.late_warning[requested]))
        if np.any(requested)
        else 0.0,
        "delta_v_limited_ratio_among_executed": float(np.mean(avoidance.delta_v_limited[executed]))
        if np.any(executed)
        else 0.0,
    }


def _run_policy(
    pre: np.ndarray,
    miss: np.ndarray,
    sigma: np.ndarray,
    hard_body: np.ndarray,
    warning: np.ndarray,
    detected: np.ndarray,
    config: dict[str, Any],
    threshold: float,
    rng: np.random.Generator,
) -> AvoidanceResult:
    avoidance = config["avoidance"]
    return apply_avoidance_policy(
        pre,
        miss,
        sigma,
        hard_body,
        warning,
        detected,
        threshold=float(threshold),
        target_factor=float(avoidance["target_probability_factor"]),
        minimum_warning_hours=float(config["warning"]["minimum_maneuver_hours"]),
        delta_v_max_mps=float(avoidance["delta_v_max_mps"]),
        displacement_factor=float(avoidance["displacement_factor"]),
        execution_error_std=float(avoidance["execution_error_std"]),
        rng=rng,
    )


def simulate_single_satellite_risk(
    config: dict[str, Any],
    *,
    seed: int,
    threshold_override: float | None = None,
) -> RiskSimulation:
    samples = int(config["precision"]["risk_event_samples"])
    output_rows = min(samples, int(config["precision"]["event_output_rows"]))
    rng = np.random.default_rng(int(seed))
    debris = config["debris"]
    velocity = config["velocity"]
    conjunction = config["conjunction"]
    warning_cfg = config["warning"]
    uncertainty = config["uncertainty"]
    avoidance_cfg = config["avoidance"]
    capacity = config["capacity"]

    # Stratification in b^2 stabilizes the rare small-miss-distance integral.
    u_miss = (np.arange(samples, dtype=float) + rng.random(samples)) / samples
    rng.shuffle(u_miss)
    diameter = sample_truncated_power_law(
        rng, samples, debris["D0_m"], debris["Dmax_m"], debris["beta"]
    )
    relative_velocity = sample_relative_velocity_km_s(
        rng,
        samples,
        velocity["mean_km_s"],
        velocity["std_km_s"],
        velocity["lower_km_s"],
        velocity["upper_km_s"],
    )
    miss = sample_miss_distance_km(
        rng, samples, conjunction["screen_radius_km"], uniform=u_miss
    )
    warning = sample_warning_hours(
        rng,
        samples,
        warning_cfg["minimum_hours"],
        warning_cfg["mode_hours"],
        warning_cfg["maximum_hours"],
    )
    sigma = position_uncertainty_km(
        warning, uncertainty["sigma0_km"], uncertainty["k_sigma_km_per_hour"]
    )
    hard_body = hard_body_radius_km(diameter, config["satellite"]["effective_area_m2"])
    pre = collision_probability_exact(miss, sigma, hard_body)
    approximate = collision_probability_small_circle(miss, sigma, hard_body)
    detection_uniform = rng.random(samples)
    detected = detection_uniform < float(avoidance_cfg["detection_probability"])
    threshold = float(
        avoidance_cfg["threshold"] if threshold_override is None else threshold_override
    )
    baseline_policy = _run_policy(
        pre, miss, sigma, hard_body, warning, detected, config, threshold, rng
    )
    burn = config["capacity"]
    degraded = degraded_duration_hours(
        capacity["satellite_mass_kg"],
        baseline_policy.delta_v_actual_mps,
        burn["thrust_n"],
        burn["recovery_overhead_hours"],
    )

    expected_velocity = relative_velocity_mean_km_s(
        velocity["mean_km_s"],
        velocity["std_km_s"],
        velocity["lower_km_s"],
        velocity["upper_km_s"],
    )
    expected_cross_section = expected_collision_cross_section_km2(
        debris["D0_m"],
        debris["Dmax_m"],
        debris["beta"],
        config["satellite"]["effective_area_m2"],
    )
    flux_rate = flux_collision_rate_per_second(
        debris["number_density_km3"], expected_cross_section, expected_velocity
    )
    conjunction_rate = conjunction_rate_per_second(
        debris["number_density_km3"], conjunction["screen_radius_km"], expected_velocity
    )
    mu_conjunction = conjunction_rate * SECONDS_PER_YEAR
    integrated_rate = conjunction_rate * float(np.mean(pre))
    consistency_error = abs(integrated_rate - flux_rate) / max(flux_rate, 1e-30)
    metrics = _annual_metrics(
        pre,
        baseline_policy,
        mu_conjunction,
        float(avoidance_cfg["conditional_failure_probability"]),
        degraded,
        threshold,
    )
    metrics.update(
        {
            "risk_event_samples": samples,
            "expected_relative_velocity_km_s": expected_velocity,
            "expected_collision_cross_section_km2": expected_cross_section,
            "mean_hard_body_radius_m": float(np.mean(hard_body) * 1000.0),
            "mean_position_uncertainty_km": float(np.mean(sigma)),
            "detection_probability": float(avoidance_cfg["detection_probability"]),
            "conditional_failure_probability": float(
                avoidance_cfg["conditional_failure_probability"]
            ),
            "post_collision_probability_is_not_forced_zero": bool(
                np.any(baseline_policy.collision_probability_post[baseline_policy.maneuver_executed] > 0.0)
            ),
        }
    )

    threshold_rows: list[dict[str, Any]] = []
    policies: dict[float, AvoidanceResult] = {threshold: baseline_policy}
    for candidate_threshold in avoidance_cfg["threshold_sensitivity"]:
        candidate_threshold = float(candidate_threshold)
        policy = policies.get(candidate_threshold)
        if policy is None:
            policy = _run_policy(
                pre, miss, sigma, hard_body, warning, detected, config, candidate_threshold, rng
            )
            policies[candidate_threshold] = policy
        candidate_degraded = degraded_duration_hours(
            capacity["satellite_mass_kg"],
            policy.delta_v_actual_mps,
            capacity["thrust_n"],
            capacity["recovery_overhead_hours"],
        )
        threshold_rows.append(
            _annual_metrics(
                pre,
                policy,
                mu_conjunction,
                float(avoidance_cfg["conditional_failure_probability"]),
                candidate_degraded,
                candidate_threshold,
            )
        )
    threshold_frame = pd.DataFrame(threshold_rows).sort_values("threshold", ascending=False)

    sensitivity_rows: list[dict[str, Any]] = []
    screen_stability: list[dict[str, Any]] = []
    for _, row in threshold_frame.iterrows():
        sensitivity_rows.append(
            {
                "parameter": "P_th",
                "value": row["threshold"],
                "scenario": f"阈值={row['threshold']:.0e}",
                "expected_annual_maneuvers": row["expected_annual_maneuvers"],
                "annual_collision_probability_post": row["annual_collision_probability_post"],
                "annual_failure_probability": row["annual_failure_probability"],
                "method": "同一事件样本重新执行决策",
            }
        )
    for q_fail in avoidance_cfg["conditional_failure_sensitivity"]:
        q_fail = float(q_fail)
        hazard = mu_conjunction * q_fail * float(
            np.mean(baseline_policy.collision_probability_post)
        )
        sensitivity_rows.append(
            {
                "parameter": "q_fail",
                "value": q_fail,
                "scenario": f"碰撞后条件失效概率={q_fail:.1f}",
                "expected_annual_maneuvers": metrics["expected_annual_maneuvers"],
                "annual_collision_probability_post": metrics["annual_collision_probability_post"],
                "annual_failure_probability": float(poisson_at_least_one_probability(hazard)),
                "method": "条件失效危险度缩放",
            }
        )
    for p_detect in avoidance_cfg["detection_probability_sensitivity"]:
        p_detect = float(p_detect)
        policy = _run_policy(
            pre,
            miss,
            sigma,
            hard_body,
            warning,
            detection_uniform < p_detect,
            config,
            threshold,
            rng,
        )
        candidate_degraded = degraded_duration_hours(
            capacity["satellite_mass_kg"], policy.delta_v_actual_mps,
            capacity["thrust_n"], capacity["recovery_overhead_hours"]
        )
        row = _annual_metrics(
            pre, policy, mu_conjunction,
            float(avoidance_cfg["conditional_failure_probability"]),
            candidate_degraded, threshold,
        )
        sensitivity_rows.append(
            {
                "parameter": "p_detect",
                "value": p_detect,
                "scenario": f"检测成功率={p_detect:.1f}",
                "expected_annual_maneuvers": row["expected_annual_maneuvers"],
                "annual_collision_probability_post": row["annual_collision_probability_post"],
                "annual_failure_probability": row["annual_failure_probability"],
                "method": "同一事件样本改变检测结果",
            }
        )
    for beta in debris["beta_sensitivity"]:
        cross_section = expected_collision_cross_section_km2(
            debris["D0_m"], debris["Dmax_m"], float(beta), config["satellite"]["effective_area_m2"]
        )
        scaled_hazard = flux_collision_rate_per_second(
            debris["number_density_km3"], cross_section, expected_velocity
        ) * SECONDS_PER_YEAR
        scale = scaled_hazard / max(metrics["annual_collision_hazard_pre"], 1e-30)
        sensitivity_rows.append(
            {
                "parameter": "beta",
                "value": float(beta),
                "scenario": f"尺寸幂律指数={float(beta):.1f}",
                "expected_annual_maneuvers": metrics["expected_annual_maneuvers"] * scale,
                "annual_collision_probability_post": float(
                    poisson_at_least_one_probability(metrics["annual_collision_hazard_post"] * scale)
                ),
                "annual_failure_probability": float(
                    poisson_at_least_one_probability(metrics["annual_failure_hazard"] * scale)
                ),
                "method": "通量危险度敏感性缩放",
            }
        )
    if bool(config["precision"].get("sensitivity_enabled", False)):
        sensitivity_count = min(samples, 100_000)
        sensitivity_slice = slice(0, sensitivity_count)
        diameter_s = diameter[sensitivity_slice]
        warning_s = warning[sensitivity_slice]
        miss_s = miss[sensitivity_slice]
        sigma_s = sigma[sensitivity_slice]
        detected_s = detected[sensitivity_slice]
        q_fail = float(avoidance_cfg["conditional_failure_probability"])

        for area in config["satellite"]["effective_area_sensitivity_m2"]:
            hard_s = hard_body_radius_km(diameter_s, float(area))
            pre_s = collision_probability_exact(miss_s, sigma_s, hard_s)
            policy_s = _run_policy(
                pre_s, miss_s, sigma_s, hard_s, warning_s, detected_s,
                config, threshold, rng,
            )
            duration_s = degraded_duration_hours(
                capacity["satellite_mass_kg"], policy_s.delta_v_actual_mps,
                capacity["thrust_n"], capacity["recovery_overhead_hours"],
            )
            row = _annual_metrics(pre_s, policy_s, mu_conjunction, q_fail, duration_s, threshold)
            sensitivity_rows.append(
                {
                    "parameter": "A_sat_eff",
                    "value": float(area),
                    "scenario": f"平均有效投影面积={float(area):.0f} m²",
                    "expected_annual_maneuvers": row["expected_annual_maneuvers"],
                    "annual_collision_probability_post": row["annual_collision_probability_post"],
                    "annual_failure_probability": row["annual_failure_probability"],
                    "mean_degraded_duration_hours": row["mean_degraded_duration_hours"],
                    "method": f"{sensitivity_count}个同源事件重新计算硬体半径和决策",
                }
            )

        baseline_expected_velocity = expected_velocity
        for velocity_mean in velocity["mean_sensitivity_km_s"]:
            expected_candidate_velocity = relative_velocity_mean_km_s(
                float(velocity_mean), velocity["std_km_s"],
                velocity["lower_km_s"], velocity["upper_km_s"],
            )
            rate_scale = expected_candidate_velocity / baseline_expected_velocity
            row = _annual_metrics(
                pre, baseline_policy, mu_conjunction * rate_scale,
                q_fail, degraded, threshold,
            )
            sensitivity_rows.append(
                {
                    "parameter": "E[V_rel]",
                    "value": float(velocity_mean),
                    "scenario": f"截断前速度均值={float(velocity_mean):.0f} km/s",
                    "expected_annual_maneuvers": row["expected_annual_maneuvers"],
                    "annual_collision_probability_post": row["annual_collision_probability_post"],
                    "annual_failure_probability": row["annual_failure_probability"],
                    "mean_degraded_duration_hours": row["mean_degraded_duration_hours"],
                    "method": "交会率按截断分布实际均值缩放",
                }
            )

        u_screen = (np.arange(sensitivity_count, dtype=float) + 0.5) / sensitivity_count
        hard_s = hard_body[sensitivity_slice]
        for screen_radius in conjunction["screen_radius_sensitivity_km"]:
            screen_radius = float(screen_radius)
            miss_screen = screen_radius * np.sqrt(u_screen)
            pre_screen = collision_probability_exact(miss_screen, sigma_s, hard_s)
            rate_screen = conjunction_rate_per_second(
                debris["number_density_km3"], screen_radius, expected_velocity
            )
            mu_screen = rate_screen * SECONDS_PER_YEAR
            policy_screen = _run_policy(
                pre_screen, miss_screen, sigma_s, hard_s, warning_s, detected_s,
                config, threshold, rng,
            )
            duration_screen = degraded_duration_hours(
                capacity["satellite_mass_kg"], policy_screen.delta_v_actual_mps,
                capacity["thrust_n"], capacity["recovery_overhead_hours"],
            )
            row = _annual_metrics(
                pre_screen, policy_screen, mu_screen, q_fail, duration_screen, threshold
            )
            integrated_screen = rate_screen * float(np.mean(pre_screen))
            relative_error_screen = abs(integrated_screen - flux_rate) / max(flux_rate, 1e-30)
            screen_stability.append(
                {
                    "screen_radius_km": screen_radius,
                    "flux_collision_rate_per_second": flux_rate,
                    "conjunction_integrated_collision_rate_per_second": integrated_screen,
                    "relative_consistency_error": relative_error_screen,
                    "expected_annual_maneuvers": row["expected_annual_maneuvers"],
                }
            )
            sensitivity_rows.append(
                {
                    "parameter": "B_screen",
                    "value": screen_radius,
                    "scenario": f"筛选半径={screen_radius:.0f} km",
                    "expected_annual_maneuvers": row["expected_annual_maneuvers"],
                    "annual_collision_probability_post": row["annual_collision_probability_post"],
                    "annual_failure_probability": row["annual_failure_probability"],
                    "mean_degraded_duration_hours": row["mean_degraded_duration_hours"],
                    "method": f"{sensitivity_count}个面积分层事件重新抽样最近距离",
                }
            )

        for scenario, values in config["warning"]["scenarios"].items():
            warning_candidate = sample_warning_hours(
                rng, sensitivity_count, float(values[0]), float(values[1]), float(values[2])
            )
            sigma_candidate = position_uncertainty_km(
                warning_candidate,
                uncertainty["sigma0_km"], uncertainty["k_sigma_km_per_hour"],
            )
            pre_candidate = collision_probability_exact(miss_s, sigma_candidate, hard_s)
            policy_candidate = _run_policy(
                pre_candidate, miss_s, sigma_candidate, hard_s, warning_candidate,
                detected_s, config, threshold, rng,
            )
            duration_candidate = degraded_duration_hours(
                capacity["satellite_mass_kg"], policy_candidate.delta_v_actual_mps,
                capacity["thrust_n"], capacity["recovery_overhead_hours"],
            )
            row = _annual_metrics(
                pre_candidate, policy_candidate, mu_conjunction, q_fail,
                duration_candidate, threshold,
            )
            sensitivity_rows.append(
                {
                    "parameter": "L_warning",
                    "value": float(values[1]),
                    "scenario": scenario,
                    "expected_annual_maneuvers": row["expected_annual_maneuvers"],
                    "annual_collision_probability_post": row["annual_collision_probability_post"],
                    "annual_failure_probability": row["annual_failure_probability"],
                    "mean_degraded_duration_hours": row["mean_degraded_duration_hours"],
                    "method": f"{sensitivity_count}个事件重新抽样预警时间",
                }
            )

        for scenario, values in config["uncertainty"]["scenarios"].items():
            sigma_candidate = position_uncertainty_km(warning_s, float(values[0]), float(values[1]))
            pre_candidate = collision_probability_exact(miss_s, sigma_candidate, hard_s)
            policy_candidate = _run_policy(
                pre_candidate, miss_s, sigma_candidate, hard_s, warning_s,
                detected_s, config, threshold, rng,
            )
            duration_candidate = degraded_duration_hours(
                capacity["satellite_mass_kg"], policy_candidate.delta_v_actual_mps,
                capacity["thrust_n"], capacity["recovery_overhead_hours"],
            )
            row = _annual_metrics(
                pre_candidate, policy_candidate, mu_conjunction, q_fail,
                duration_candidate, threshold,
            )
            sensitivity_rows.append(
                {
                    "parameter": "sigma_pos",
                    "value": float(values[0]),
                    "scenario": scenario,
                    "expected_annual_maneuvers": row["expected_annual_maneuvers"],
                    "annual_collision_probability_post": row["annual_collision_probability_post"],
                    "annual_failure_probability": row["annual_failure_probability"],
                    "mean_degraded_duration_hours": row["mean_degraded_duration_hours"],
                    "method": f"{sensitivity_count}个同源事件重新计算位置误差",
                }
            )

        executed = baseline_policy.maneuver_executed
        for thrust in capacity["thrust_sensitivity_n"]:
            duration_candidate = degraded_duration_hours(
                capacity["satellite_mass_kg"], baseline_policy.delta_v_actual_mps,
                float(thrust), capacity["recovery_overhead_hours"],
            )
            sensitivity_rows.append(
                {
                    "parameter": "F_thrust",
                    "value": float(thrust),
                    "scenario": f"推力={float(thrust):.2f} N",
                    "expected_annual_maneuvers": metrics["expected_annual_maneuvers"],
                    "annual_collision_probability_post": metrics["annual_collision_probability_post"],
                    "annual_failure_probability": metrics["annual_failure_probability"],
                    "mean_degraded_duration_hours": float(np.mean(duration_candidate[executed])),
                    "method": "保持决策不变，重新计算点火时间",
                }
            )
        for overhead in capacity["overhead_sensitivity_hours"]:
            duration_candidate = degraded_duration_hours(
                capacity["satellite_mass_kg"], baseline_policy.delta_v_actual_mps,
                capacity["thrust_n"], float(overhead),
            )
            sensitivity_rows.append(
                {
                    "parameter": "t_overhead",
                    "value": float(overhead),
                    "scenario": f"固定恢复开销={float(overhead):.1f} h",
                    "expected_annual_maneuvers": metrics["expected_annual_maneuvers"],
                    "annual_collision_probability_post": metrics["annual_collision_probability_post"],
                    "annual_failure_probability": metrics["annual_failure_probability"],
                    "mean_degraded_duration_hours": float(np.mean(duration_candidate[executed])),
                    "method": "保持决策不变，重新计算业务降级窗口",
                }
            )
        for execution_std in avoidance_cfg["execution_error_std_sensitivity"]:
            candidate_config = copy.deepcopy(config)
            candidate_config["avoidance"]["execution_error_std"] = float(execution_std)
            policy_candidate = _run_policy(
                pre[sensitivity_slice], miss_s, sigma_s, hard_s, warning_s,
                detected_s, candidate_config, threshold, rng,
            )
            duration_candidate = degraded_duration_hours(
                capacity["satellite_mass_kg"], policy_candidate.delta_v_actual_mps,
                capacity["thrust_n"], capacity["recovery_overhead_hours"],
            )
            row = _annual_metrics(
                pre[sensitivity_slice], policy_candidate, mu_conjunction, q_fail,
                duration_candidate, threshold,
            )
            sensitivity_rows.append(
                {
                    "parameter": "sigma_execution",
                    "value": float(execution_std),
                    "scenario": f"执行误差标准差={float(execution_std):.2f}",
                    "expected_annual_maneuvers": row["expected_annual_maneuvers"],
                    "annual_collision_probability_post": row["annual_collision_probability_post"],
                    "annual_failure_probability": row["annual_failure_probability"],
                    "mean_degraded_duration_hours": row["mean_degraded_duration_hours"],
                    "method": f"{sensitivity_count}个同源事件重新施加执行误差",
                }
            )
    sensitivity_frame = pd.DataFrame(sensitivity_rows)

    selection = np.linspace(0, samples - 1, output_rows, dtype=int)
    exact_selected = pre[selection]
    approximate_selected = approximate[selection]
    relative_error = np.abs(approximate_selected - exact_selected) / np.maximum(exact_selected, 1e-30)
    events = pd.DataFrame(
        {
            "event_sample_id": selection,
            "debris_diameter_m": diameter[selection],
            "relative_velocity_km_s": relative_velocity[selection],
            "warning_hours": warning[selection],
            "sigma_pos_km": sigma[selection],
            "predicted_miss_distance_km": miss[selection],
            "hard_body_radius_km": hard_body[selection],
            "collision_probability_pre": exact_selected,
            "collision_probability_small_circle": approximate_selected,
            "small_circle_relative_error": relative_error,
            "detected": baseline_policy.detected[selection],
            "maneuver_requested": baseline_policy.maneuver_requested[selection],
            "maneuver_executed": baseline_policy.maneuver_executed[selection],
            "late_warning": baseline_policy.late_warning[selection],
            "delta_v_limited": baseline_policy.delta_v_limited[selection],
            "delta_v_required_mps": baseline_policy.delta_v_required_mps[selection],
            "delta_v_actual_mps": baseline_policy.delta_v_actual_mps[selection],
            "post_miss_distance_km": baseline_policy.post_miss_distance_km[selection],
            "collision_probability_post": baseline_policy.collision_probability_post[selection],
        }
    )
    executed_index = np.flatnonzero(baseline_policy.maneuver_executed)
    maneuver_delta_v = pd.DataFrame(
        {
            "event_sample_id": executed_index,
            "delta_v_required_mps": baseline_policy.delta_v_required_mps[executed_index],
            "delta_v_commanded_mps": baseline_policy.delta_v_commanded_mps[executed_index],
            "delta_v_actual_mps": baseline_policy.delta_v_actual_mps[executed_index],
            "delta_v_limited": baseline_policy.delta_v_limited[executed_index],
        }
    )
    maneuver_duration = pd.DataFrame(
        {
            "event_sample_id": executed_index,
            "burn_time_seconds": (
                degraded[executed_index] - float(capacity["recovery_overhead_hours"])
            )
            * 3600.0,
            "recovery_overhead_hours": float(capacity["recovery_overhead_hours"]),
            "degraded_duration_hours": degraded[executed_index],
        }
    )
    consistency = {
        "flux_collision_rate_per_second": flux_rate,
        "conjunction_rate_per_second": conjunction_rate,
        "mean_annual_conjunctions": mu_conjunction,
        "mean_predicted_collision_probability_per_conjunction": float(np.mean(pre)),
        "conjunction_integrated_collision_rate_per_second": integrated_rate,
        "relative_consistency_error": consistency_error,
        "configured_tolerance": float(conjunction["consistency_tolerance"]),
        "passes_standard_tolerance": consistency_error <= float(conjunction["consistency_tolerance"]),
        "screen_radius_km": float(conjunction["screen_radius_km"]),
        "screen_radius_stability": screen_stability,
    }
    return RiskSimulation(
        summary=metrics,
        consistency=consistency,
        threshold_comparison=threshold_frame,
        sensitivity=sensitivity_frame,
        event_samples=events,
        maneuver_delta_v=maneuver_delta_v,
        maneuver_duration=maneuver_duration,
        arrays={
            "diameter_m": diameter,
            "relative_velocity_km_s": relative_velocity,
            "warning_hours": warning,
            "miss_distance_km": miss,
            "sigma_pos_km": sigma,
            "hard_body_radius_km": hard_body,
            "collision_probability_pre": pre,
            "collision_probability_post": baseline_policy.collision_probability_post,
            "delta_v_actual_mps": baseline_policy.delta_v_actual_mps,
            "degraded_duration_hours": degraded,
            "maneuver_executed": baseline_policy.maneuver_executed,
        },
    )
