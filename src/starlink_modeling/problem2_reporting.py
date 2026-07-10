"""Problem-two CSV/JSON export, cost model and paper-oriented Markdown."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .problem2_ga import Individual
from .problem2_orbit import WalkerConstellation, orbital_period_seconds


SATELLITE_MANUFACTURING_COST_YUAN = 5_000_000
LAUNCH_CAPACITY = 60
LAUNCH_COST_YUAN = 200_000_000


def deployment_cost(total_satellites: int) -> dict[str, float | int]:
    launches = int(math.ceil(total_satellites / LAUNCH_CAPACITY))
    manufacturing = int(total_satellites * SATELLITE_MANUFACTURING_COST_YUAN)
    launch = int(launches * LAUNCH_COST_YUAN)
    total = manufacturing + launch
    return {
        "manufacturing_cost_yuan": manufacturing,
        "manufacturing_cost_100m_yuan": manufacturing / 1e8,
        "launch_count": launches,
        "launch_cost_yuan": launch,
        "launch_cost_100m_yuan": launch / 1e8,
        "total_cost_yuan": total,
        "total_cost_100m_yuan": total / 1e8,
    }


def solution_payload(
    scenario: str,
    candidate: Individual,
    constellation: WalkerConstellation,
    config: dict[str, Any],
    theta_rad: float,
    random_seed: int,
    validation_status: str,
    validation_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    simulation = config["simulation"]
    payload: dict[str, Any] = {
        "scenario": scenario,
        **candidate.chromosome_dict(),
        "total_satellites": candidate.total_satellites,
        "raan_layout_deg": sorted(
            set(round(value, 10) for value in constellation.to_dataframe()["raan_deg"])
        ),
        "altitude_km": constellation.altitude_km,
        "theta_deg": math.degrees(theta_rad),
        "orbital_period_min": orbital_period_seconds() / 60.0,
        "simulation_days": simulation["simulation_days"],
        "time_step_seconds": simulation["time_step_seconds"],
        "grid_type": simulation["grid_type"],
        "grid_resolution": (
            simulation["equal_arc_spacing_km"]
            if simulation["grid_type"] == "equal_arc"
            else simulation["regular_grid_step_deg"]
        ),
        **candidate.metrics,
        **deployment_cost(candidate.total_satellites),
        "random_seed": int(random_seed),
        "validation_status": validation_status,
        "ga_feasible_at_search_resolution": bool(candidate.feasible),
        "mode": config["mode"],
        "validation_results": validation_results or {},
    }
    return payload


def write_solution_bundle(
    output_dir: Path,
    scenario: str,
    payload: dict[str, Any],
    constellation: WalkerConstellation,
    metrics: pd.DataFrame,
    time_metrics: pd.DataFrame,
    grid_metrics: pd.DataFrame,
) -> None:
    (output_dir / f"best_{scenario}_constellation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    constellation.to_dataframe().to_csv(
        output_dir / f"best_{scenario}_satellites.csv", index=False
    )
    metrics.to_csv(output_dir / f"{scenario}_solution_metrics.csv", index=False)
    time_metrics.to_csv(output_dir / f"{scenario}_time_metrics.csv", index=False)
    grid_metrics.to_csv(output_dir / f"{scenario}_grid_metrics.csv", index=False)


def build_cost_comparison(payloads: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for scenario in ("single", "double"):
        if scenario not in payloads:
            continue
        payload = payloads[scenario]
        rows.append(
            {
                "scenario": scenario,
                "scenario_cn": "单重覆盖" if scenario == "single" else "二重覆盖",
                "M": payload["M"],
                "N": payload["N"],
                "total_satellites": payload["total_satellites"],
                "launch_count": payload["launch_count"],
                "manufacturing_cost_yuan": payload["manufacturing_cost_yuan"],
                "manufacturing_cost_100m_yuan": payload["manufacturing_cost_100m_yuan"],
                "launch_cost_yuan": payload["launch_cost_yuan"],
                "launch_cost_100m_yuan": payload["launch_cost_100m_yuan"],
                "total_cost_yuan": payload["total_cost_yuan"],
                "total_cost_100m_yuan": payload["total_cost_100m_yuan"],
            }
        )
    frame = pd.DataFrame(rows)
    if {"single", "double"}.issubset(payloads):
        single = payloads["single"]
        double = payloads["double"]
        frame["double_minus_single_satellites"] = (
            double["total_satellites"] - single["total_satellites"]
        )
        frame["satellite_increase_ratio"] = (
            double["total_satellites"] / single["total_satellites"] - 1.0
        )
        frame["double_minus_single_cost_yuan"] = (
            double["total_cost_yuan"] - single["total_cost_yuan"]
        )
        frame["cost_increase_ratio"] = (
            double["total_cost_yuan"] / single["total_cost_yuan"] - 1.0
        )
    return frame


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _solution_result_paragraph(payload: dict[str, Any] | None, name: str) -> str:
    if payload is None:
        return f"本次运行未执行{name}场景，因此没有生成可供分析的数值结论。"
    status = payload["validation_status"]
    return (
        f"{name}场景在本次 `{payload['mode']}` 模式下得到的最佳已搜索构型为 "
        f"M={payload['M']}、N={payload['N']}、总卫星数 {payload['total_satellites']} 颗，"
        f"倾角 {payload['inclination_deg']:.4f}°、Walker 相位因子 F={payload['phase_factor_F']}、"
        f"Ω0={payload['Omega0_deg']:.4f}°、u0={payload['u0_deg']:.4f}°。"
        f"其离散仿真指标为 Q1={payload.get('Q1_full_region', float('nan')):.6f}、"
        f"Q2={payload.get('Q2_full_region', float('nan')):.6f}、"
        f"P1={payload.get('P1_space_time', float('nan')):.6f}、"
        f"P2={payload.get('P2_space_time', float('nan')):.6f}，"
        f"平均覆盖重数为 {payload.get('mean_multiplicity', float('nan')):.4f}，"
        f"单重最大间隙为 {payload.get('max_gap_time_min', float('nan')):.2f} min。"
        f"部署需发射 {payload['launch_count']} 次，总成本约 {payload['total_cost_100m_yuan']:.2f} 亿元。"
        f"验证状态为 `{status}`。"
    )


def _sensitivity_result_paragraph(
    sensitivity: pd.DataFrame, scenario: str, name: str
) -> str:
    if sensitivity.empty or "scenario" not in sensitivity.columns:
        return f"{name}场景未执行敏感性分析。"
    subset = sensitivity[
        (sensitivity["scenario"] == scenario)
        & (sensitivity["status"] == "completed")
    ]
    if subset.empty:
        return f"{name}场景未产生已完成的敏感性算例。"
    min_q1 = float(subset["Q1_full_region"].min())
    min_q2 = float(subset["Q2_full_region"].min())
    max_gap = float(subset["max_gap_time_min"].max())
    passed = min_q1 >= 1.0 - 1e-12 and max_gap <= 1e-12
    if scenario == "double":
        passed = passed and min_q2 >= 0.95 - 1e-12
    conclusion = "仍满足该场景约束" if passed else "未能在全部细化算例中保持约束"
    return (
        f"{name}场景完成 {len(subset)} 个实际敏感性算例，最小 Q1={min_q1:.6f}、"
        f"最小 Q2={min_q2:.6f}、最大单重间隙={max_gap:.2f} min，{conclusion}。"
    )


def generate_method_notes(
    output_dir: Path,
    config: dict[str, Any],
    active_scenarios: set[str] | None = None,
) -> str:
    """Generate paper-oriented notes by reading actual run artifacts."""

    active_scenarios = active_scenarios or {"single", "double"}
    single = (
        _load_json_if_exists(output_dir / "best_single_constellation.json")
        if "single" in active_scenarios
        else None
    )
    double = (
        _load_json_if_exists(output_dir / "best_double_constellation.json")
        if "double" in active_scenarios
        else None
    )
    sensitivity_path = output_dir / "resolution_sensitivity.csv"
    sensitivity = pd.read_csv(sensitivity_path) if sensitivity_path.exists() else pd.DataFrame()
    completed_sensitivity = int((sensitivity.get("status") == "completed").sum()) if not sensitivity.empty else 0
    single_raan = "、".join(f"{value:.2f}°" for value in single.get("raan_layout_deg", [])) if single else "未计算"
    double_raan = "、".join(f"{value:.2f}°" for value in double.get("raan_layout_deg", [])) if double else "未计算"

    notes = rf"""# 问题二：多轨道面星座组网优化设计方法说明

> 本文件由 `problem2_constellation_ga.py` 根据实际 JSON 与 CSV 结果生成。当前运行模式为 `{config['mode']}`。遗传算法只能说明在给定搜索范围、仿真时段、空间网格和多随机种子条件下找到的最优已搜索方案，不能证明数学意义上的全局最优。`quick` 结果只用于程序冒烟测试，不得直接作为论文最终结论。

## 1. 问题分析

问题一把同一轨道面卫星投影到纬度轴，研究的是一维纬度区间并集。问题二则要求同时处理经度、纬度和时间：需要由多个轨道面共同覆盖 4°N～53°N、73°E～135°E 的二维矩形区域，并在每个离散时刻检查区域内所有网格点。因此，问题二不是问题一最小卫星数曲线的简单叠加，而是星座规模、区域连续性、覆盖冗余与部署成本之间的约束优化。

本文分别建立两个优化场景。单重场景要求所有采样时刻全区域至少一重覆盖；二重场景在保持单重连续覆盖的基础上，要求至少 95% 的采样时刻全区域同时达到二重覆盖。目标函数均为最小化总卫星数 S=MN。

初步构型基准表明，原建议的 M≤30 无法进入当前 506 km 覆盖半径下的全区域连续覆盖可行域，因此三个配置文件把 M 上限扩展到 60，并把实际范围写入运行元数据。该调整不是改变覆盖判据，而是避免遗传算法在几何上不足的搜索域内反复计算。

## 2. 模型假设

地球近似为半径 6371 km 的球体；所有卫星均处于高度 550 km、偏心率为零的圆轨道；各轨道面高度和倾角相同，升交点赤经在 360° 范围内均匀分布；每个轨道面内卫星沿纬度辐角均匀分布，并使用 Walker-Delta 相位关系错开相邻轨道面。覆盖半径采用题目直接给出的 506 km，故覆盖地心角为 θ=506/6371。

主模型采用二体圆轨道加地球自转，忽略短期机动、姿态误差和摄动。J2 摄动会造成升交点长期漂移，但问题二重点是基础构型与几何覆盖，故未把 J2 纳入遗传算法主循环。后续可在更长时段任务中加入 J2 进行轨道保持与长期漂移校核。

所谓“连续覆盖”由空间网格和时间采样数值逼近。它不是对连续时空的严格解析证明，最终结论必须附带网格步长、时间步长和仿真天数。

## 3. Walker-Delta 星座构型

染色体写为 X=[M,N,i,F,Ω0,u0]。M 和 N 分别表示轨道面数与每轨卫星数，i 为公共倾角，F 为整数相位因子，Ω0 为第一个轨道面的升交点赤经，u0 为其第一颗卫星的初始纬度辐角，总星数为 S=MN。

第 p 个轨道面的升交点赤经为

$$Ω_p=Ω_0+\frac{{2πp}}{{M}},\quad p=0,1,\ldots,M-1.$$

第 p 个轨道面内第 q 颗卫星的初始纬度辐角为

$$u_{{pq}}(0)=u_0+\frac{{2πq}}{{N}}+\frac{{2πFp}}{{MN}},\quad q=0,1,\ldots,N-1.$$

因此同轨相邻卫星相隔 360°/N，相邻轨道面的整体相位偏移为 360°F/(MN)。利用构型对称性，Ω0 只需在 [0,360°/M) 内搜索，u0 只需在 [0,360°/N) 内搜索。每次交叉和变异后均执行 repair，修复整数基因、变量边界和周期角。

本次单重方案各轨道面升交点为：{single_raan}。

本次二重方案各轨道面升交点为：{double_raan}。

## 4. 卫星轨道与星下点模型

轨道半径 a=R+h，平均角速度与周期分别为

$$n=\sqrt{{\mu/a^3}},\qquad T=\frac{{2π}}{{n}}.$$

对任意卫星，令 Ω=Ωp、u=upq(0)+nt，则其 ECI 坐标为

$$x=a(\cos Ω\cos u-\sin Ω\sin u\cos i),$$
$$y=a(\sin Ω\cos u+\cos Ω\sin u\cos i),$$
$$z=a\sin u\sin i.$$

考虑地球自转后，使用 $\mathbf r_{{ECEF}}=R_3(-ω_Et)\mathbf r_{{ECI}}$。星下点纬度与经度由 `atan2(z,sqrt(x²+y²))` 和 `atan2(y,x)` 得到，经度归一化到 [-180°,180°)。负的地球自转项使星下点轨迹相对地表向西漂移，从而影响同一地区在相邻轨道圈次的访问时刻。

## 5. 目标区域网格化

只检查区域中心点会遗漏边界和角点，因此程序支持等经纬度网格与近似等弧长网格。等经纬度网格中每个点代表的面积随纬度变化，面积统计采用 $w_g\propto\cos φ_g$；等弧长网格令纬度步长近似为 Δl/R，并按 Δl/(R cosφ) 调整每一纬度行的经度步长，使点密度更接近等面积。所有权重最终归一化为 $\sum_gw_g=1$。

网格构造显式包含 4°N、53°N、73°E、135°E 和四个角点。面积平均指标使用权重，而连续覆盖约束仍逐点检查，任何一个网格点未覆盖都会使该时刻的全区域覆盖标志为零。这一思路与区域通信星座研究中“特征点和等弧长网格”的思想一致，但本程序按本题边界和覆盖半径独立实现。

## 6. 覆盖性能评价模型

将地面点与卫星星下点写成地心单位向量 g 和 s。若 $g\cdot s\ge\cos θ$，则该卫星覆盖该点。程序优先使用单位球 KDTree，查询半径为 $2\sin(θ/2)$；若 SciPy 不可用，则按地面点分块计算向量点积。计算过程逐时刻累计，不创建 time×satellite×grid 的超大三维数组。

覆盖重数 $c(g,t)$ 是时刻 t 覆盖点 g 的卫星数量。瞬时单重与二重面积覆盖率为

$$A_1(t)=\sum_gw_gI[c(g,t)\ge1],\qquad A_2(t)=\sum_gw_gI[c(g,t)\ge2].$$

时空覆盖率为 $P_1=\operatorname{{mean}}_tA_1(t)$、$P_2=\operatorname{{mean}}_tA_2(t)$。全区域覆盖标志为 $B_1(t)=I[\min_gc(g,t)\ge1]$ 和 $B_2(t)=I[\min_gc(g,t)\ge2]$，其时间平均分别记为 Q1 与 Q2。本文将“95% 以上时间实现二重覆盖”解释为 Q2≥0.95，即至少 95% 的采样时刻，区域内所有网格点均同时具有至少两颗可见卫星；同时补充报告 P2，以展示全部地面点—时间样本的二重覆盖比例。

程序还计算平均覆盖重数、最差时刻 A1/A2、单点可用率、最长连续未覆盖时间、可用率标准差、平均覆盖重数空间标准差和角度覆盖裕度。角度裕度定义为 $θ-\min_sψ_{{gs}}(t)$；若全局最小裕度接近零，方案可能对网格或时间离散误差敏感。

## 7. 优化模型

单重场景为

$$\min MN,\quad \text{{s.t. }}Q_1\ge1-\varepsilon,\ T_{{gap,max}}=0.$$

二重场景为

$$\min MN,\quad \text{{s.t. }}Q_1\ge1-\varepsilon,\ Q_2\ge0.95.$$

搜索范围为 M∈[{config['search_bounds']['M'][0]},{config['search_bounds']['M'][1]}]、N∈[{config['search_bounds']['N'][0]},{config['search_bounds']['N'][1]}]、i∈[{config['search_bounds']['inclination_deg'][0]}°,{config['search_bounds']['inclination_deg'][1]}°]、F∈{{0,…,M-1}}。几何必要条件 i+θ≥53° 在完整仿真前检查；不满足时最高可达纬度不足，直接判为不可行。

## 8. 遗传算法设计

覆盖评价由离散可见性判断、最小值和布尔约束构成，不连续且不可微，难以直接使用梯度方法。程序采用混合整数—实数遗传算法：M、N、F 是整数基因，i、Ω0、u0 是实数基因。初始种群混合启发式样本与全范围随机样本，既关注 49°～57° 的有效倾角区域，也保留探索能力。

选择采用锦标赛和精英保留；整数基因由父代继承并通过 ±1、±2 或随机重置变异，连续基因使用算术混合与高斯变异，变异尺度随代数缩小。停滞达到阈值时临时提高变异率。适应度评价按修复后的染色体缓存，固定随机种子保证重复运行一致；多个 worker 使用共享只读网格的线程并行，避免进程间复制大数组。每代写入当前/全局最优、可行个体数、平均违反度和卫星数分布，并保存可恢复检查点。

约束处理使用 Deb 规则：可行解始终优于不可行解；可行解先比较卫星数，再比较覆盖冗余和均匀性；不可行解比较透明的总约束违反度，而不是只用一个无法解释的标量罚函数。

## 9. 粗搜索—固定构型核验—局部加密—精细验证

阶段 A 在较粗网格和较大时间步长下同时搜索六个变量。阶段 B 枚举低于当前卫星数或位于给定裕量内的 (M,N)，固定两个整数后仅优化 i、F、Ω0、u0，用于检查遗传算法是否漏掉更小构型。阶段 C 对最佳候选执行坐标式局部加密。阶段 D 使用更细网格、更短时间步长和更长仿真时长复核。

程序会把配置范围内的固定 M,N 组合全部写入核验表。考虑到对数千个组合逐一运行多种子遗传算法的计算量，默认配置用 `max_pairs` 优先核验最接近当前下界的低卫星数组合，其余组合明确标记为未运行；若需要穷举，可在配置中把该值设为 `null`，但预计耗时会显著增加。

主循环不直接采用最高精度，是因为一次高精度评价已经包含大量时间—卫星—网格邻域查询，若再乘以种群规模、代数和随机种子数，计算量会过大。分层方法把算力集中到少量候选，同时保留反向核验。若低星数核验未找到可行方案，只能说“在给定范围和多随机种子核验下未发现更少的可行构型”，不能视作严格证明。

## 10. 结果分析

{_solution_result_paragraph(single, '单重覆盖')}

{_solution_result_paragraph(double, '二重覆盖')}

结果表中的最差区域由单点可用率最小的网格坐标给出；覆盖间隙热力图和最差时刻覆盖重数图用于定位薄弱边界。若 `validation_status` 为 `quick_smoke_only`、`search_infeasible` 或 `validation_failed`，说明该结果不能被当作最终论文方案，应继续运行 standard/full 或扩大搜索范围。

## 11. 敏感性和稳定性

本次输出中共有 {completed_sensitivity} 个实际完成的敏感性算例。`resolution_sensitivity.csv` 分别改变规则网格步长、时间步长和仿真天数，并记录 Q1、Q2、P1、P2、最大间隙与计算规模。若细化后 Q1/Q2 跌破约束，程序应把候选标记为 `validation_failed`，并选择更稳健方案，而不能隐藏失效结果。

{_sensitivity_result_paragraph(sensitivity, 'single', '单重覆盖')}

{_sensitivity_result_paragraph(sensitivity, 'double', '二重覆盖')}

随机种子、种群规模和变异率会影响遗传算法能否发现低卫星数可行解。standard/full 配置通过增加种群、代数和随机种子降低偶然性；最终论文应同时报告这些算法参数及固定 M,N 的反向核验结果。

## 12. 模型优点与局限

模型用六个 Walker 参数描述可实施的规则星座，参数少且结构清晰；三维传播、二维网格和逐点判据直接对应题意；覆盖评价、优化、成本和图表相互解耦，便于扩展到问题三。KDTree、分块回退、缓存和分层搜索使较大规模计算可执行。

局限在于：网格点全覆盖不能严格保证网格单元内部每一点覆盖；采样时刻全覆盖不能严格保证采样间隔内没有瞬时空隙；遗传算法不能证明全局最优；模型忽略 J2、姿态误差、链路余量和轨道维持；目标区域按矩形处理，覆盖半径固定为 506 km。本文通过细化网格、缩短时间步长、延长仿真天数和敏感性分析提高数值可信度，但结论仍只在指定分辨率下成立。

## 13. 可直接用于论文的结论模板

本文采用 Walker-Delta 规则构型，将轨道面数、每轨卫星数、倾角、相位因子、首轨道面升交点和初始相位作为混合优化变量，并以地面网格逐点可见性构造全区域连续覆盖约束。遗传算法在粗分辨率下搜索候选，再通过固定 M,N 反向核验、局部加密和高精度仿真复核，从而在计算成本与结论可信度之间取得平衡。

{_solution_result_paragraph(single, '单重覆盖')}

{_solution_result_paragraph(double, '二重覆盖')}

上述结果应表述为“在给定参数范围、仿真时段、空间网格精度和多随机种子条件下获得的最优已搜索方案”。只有通过 full 配置的 24 h 高空间/时间精度与 7 d 中等精度验证后，才适合作为论文最终数值结论。
"""
    return notes


def write_method_notes(
    output_dir: Path,
    config: dict[str, Any],
    project_root: Path,
    active_scenarios: set[str] | None = None,
) -> None:
    notes = generate_method_notes(output_dir, config, active_scenarios)
    (project_root / "problem2_method_notes.md").write_text(notes, encoding="utf-8")
    report_path = project_root / "docs" / "report" / "problem2_method_notes.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(notes, encoding="utf-8")
