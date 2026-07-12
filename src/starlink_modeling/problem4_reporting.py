"""Problem-four JSON helpers and auto-populated Chinese method notes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _money_yi(value: float) -> str:
    return f"{float(value) / 1e8:.4f} 亿元"


def _scheme_text(row: dict[str, Any] | None) -> str:
    if not row:
        return "当前搜索范围内无可行正冗余方案"
    return (
        f"{row['scheme_type']}：每轨备用 k={int(row['k_per_plane'])}，"
        f"额外轨道面 r={int(row['extra_plane_count'])}，地面备用 G={int(row['ground_spare_count'])}，"
        f"新增 {int(row['added_satellites'])} 颗，五年预期成本 {_money_yi(row['five_year_expected_cost_cny'])}，"
        f"平均可用率 {row['mean_effective_coverage_availability']:.6%}，"
        f"5%分位 {row['q05_effective_coverage_availability']:.6%}"
    )


def write_method_notes(
    project_root: Path,
    output_dir: Path,
    config: dict[str, Any],
    selected: Any,
) -> str:
    risk = _read_json(output_dir / "single_satellite_risk_summary.json")
    consistency = _read_json(output_dir / "risk_model_consistency.json")
    constellation = _read_json(output_dir / "constellation_annual_summary.json")
    capacity = _read_json(output_dir / "capacity_loss_summary.json")
    best = _read_json(output_dir / "best_redundancy_solution.json")
    redundancy = pd.read_csv(output_dir / "redundancy_monte_carlo_summary.csv")
    service = pd.read_csv(output_dir / "problem3_service_impact.csv")
    peak = service[service["demand_level"] == "peak"].set_index("scenario")
    normal_peak = peak.loc["正常基准"]
    combined_peak = peak.loc["避撞加失效"]
    recommended = best["recommended_candidate"]
    no_redundancy = best["no_redundancy"]
    mixed = redundancy[(redundancy["scheme_type"] == "混合方案") & redundancy["meets_mean_99"]]
    best_mixed = None if mixed.empty else mixed.sort_values("five_year_expected_cost_cny").iloc[0].to_dict()
    pure_rows = []
    for name, value in best["pure_scheme_minimums"].items():
        pure_rows.append(f"- {name}：{_scheme_text(value)}")
    formal_note = (
        "本次为 Standard 正式计算，可用于论文结果段，但结论仍受配置、样本数和模型假设限制。"
        if config["mode"] == "standard"
        else "本次为 Quick/Full 中间输出；Quick 仅用于程序验收，不能作为论文正式结论。"
    )
    text = rf"""# 问题四：碎片规避与星座鲁棒性设计方法说明

> 本文件由 `problem4_debris_robustness.py` 从本次实际输出自动生成。运行模式为 `{config['mode']}`。{formal_note}

## 1. 问题四研究内容

问题四在问题二覆盖星座和问题三通信网络的基础上，引入空间碎片候选交会、自主避撞、碰撞失效、通信降级、同轨重排和备用补充。三个小问依次回答单星风险与决策、1840 星年度经济和通信影响，以及满足 99% 基本覆盖时间比例所需的最低成本冗余。模型不重新优化 Walker 星座的 M、N、倾角或相位。

## 2. 与问题二、问题三的衔接

程序递归检查 `outputs/problem3/**/problem3_run_metadata.json`，实际选择 `{selected.source_directory}`。其元数据 SHA-256 为 `{selected.metadata_sha256}`，星座签名为 `{selected.constellation_signature}`。读取到 M={selected.M}、N={selected.N}、S={selected.total_satellites}，倾角 {selected.inclination_deg:.8f}°，F={selected.phase_factor_F}，Ω0={selected.Omega0_deg:.8f}°，u0={selected.u0_deg:.8f}°，高度 {selected.altitude_km:.1f} km，覆盖地心角 {selected.theta_deg:.8f}°。

所选问题三 Standard 的平均、p95 和最大路由时延分别为 {selected.problem3_mean_delay_ms:.6f} ms、{selected.problem3_p95_delay_ms:.6f} ms 和 {selected.problem3_max_delay_ms:.6f} ms，30 ms 以下比例为 {selected.problem3_ratio_below_30ms:.6%}；平均吞吐量为 {selected.problem3_mean_throughput_tbps:.6f} Tbps，平均和峰值服务比例分别为 {selected.problem3_mean_service_ratio:.6%} 与 {selected.problem3_peak_service_ratio:.6%}。根目录问题三结果是 Quick，不能进入正式结论。

## 3. 参数分类与审计

题目直接给出碎片数密度 $10^{{-8}}\,\mathrm{{km}}^{{-3}}$、单次 Δv 上限 1 m/s、单次避撞成本 2 万元、避撞期间 50% 通信能力、7 天同轨调整、227 kg 质量、20 Gbps 接入容量、500 万元制造成本、2 亿元发射成本、单箭 60 星和 5 年寿命。轨道和通信基准来自前问 Standard。碎片幂律指数、1 m 上界、15 m² 平均有效投影面积、速度与预警分布、位置误差、0.1 N 推力、1 h 恢复开销、6 h 在轨接替和 60 d 地面补充均是建模假设。

`A_sat_eff=15 m²` 是平均有效投影面积，不是卫星本体表面积，也不是太阳翼展开总面积。参数表没有填写未经实际访问的来源链接；Klinkrad、NASA、ESA、SpaceX/FCC 和 Akella-Alfriend 仅列为后续可核查的参考方向。

## 4. 三类概率必须分开

公式 $\sigma vnt$ 描述的是长期平均实体碰撞风险，不是每一次避撞预警使用的碰撞概率。长期风险由碎片通量与硬体截面积积分得到；单次交会预测概率由最近距离和位置协方差决定；碰撞后的失效风险还要乘条件概率 $q_{{fail}}$。主模型令 $q_{{fail}}=1$ 只是高速碎片碰撞的保守上界，并补充 0.5、0.8、1.0 敏感性。

## 5. 碎片尺寸幂律分布

在 $D_0=0.01$ m 到 $D_{{max}}=1$ m 之间采用截断幂律。归一化概率密度为

$$f_D(D)=\frac{{\beta D_0^\beta D^{{-\beta-1}}}}{{1-(D_0/D_{{max}})^\beta}}.$$

蒙特卡洛使用该分布的逆变换，不做均匀直径抽样。有限上界数值归一化误差记录在参数和一致性输出中。

## 6. 相对速度、预警时间与位置误差

相对速度采用均值 10 km/s、标准差 2 km/s、截断 5～15 km/s 的正态分布。预警时间采用 6 h、72 h、168 h 三角分布。交会平面位置标准差为

$$\sigma_{{pos}}(L)=\sqrt{{\sigma_0^2+(k_\sigma L)^2}},$$

其中 $\sigma_0=0.05$ km，$k_\sigma=0.005$ km/h。更长预警给机动留下更多位移时间，但轨道外推误差也可能更大，二者在单次概率与 Δv 中同时体现。

## 7. 候选交会泊松模型

筛选半径 $B=5$ km 只用于生成待评估事件，不是碰撞半径。候选率为 $\lambda_c=n\pi B^2E[V]$，年候选数服从泊松分布。最近距离按筛选圆面积均匀抽样，即 $b=B\sqrt U$。每颗卫星事件过程相互独立。

所有卫星可以拥有相同的年平均风险参数，但各颗卫星的交会事件由独立泊松过程产生，因此不会同时达到阈值。同分布不等于同时发生，就像同一城市中的司机具有相近年事故率，但不会在同一时刻发生事故。

## 8. 单次预测碰撞概率

把卫星等效半径与碎片半径相加得到硬体半径 $R_h$。在二维各向同性高斯误差下，$(r/\sigma)^2$ 服从自由度 2 的非中心卡方分布：

$$P_c=F_{{\chi'^2_2}}\left((R_h/\sigma)^2;(b/\sigma)^2\right).$$

程序使用 `scipy.stats.ncx2.cdf`，并用小圆近似 $R_h^2(2\sigma^2)^{{-1}}\exp[-b^2/(2\sigma^2)]$ 交叉检查。

## 9. 通量一致性验证

通量模型危险率为 {consistency['flux_collision_rate_per_second']:.8e} s⁻¹，候选交会积分危险率为 {consistency['conjunction_integrated_collision_rate_per_second']:.8e} s⁻¹，相对误差 {consistency['relative_consistency_error']:.4%}。Standard 容许上限为 {consistency['configured_tolerance']:.2%}，本次检查结果为 `{consistency['passes_standard_tolerance']}`。这一步用于发现筛选半径、单位或抽样错误。

## 10. 避撞阈值与决策

基准阈值为 $P_{{th}}={risk['threshold']:.0e}$，并比较 $10^{{-4}}$、$10^{{-5}}$、$10^{{-6}}$。只有事件已检测、$P_c\ge P_{{th}}$ 且预警不少于 6 h 时才执行机动；未检测事件保留原风险，晚预警事件不标记为成功避撞。

## 11. Δv 与残余风险

避撞不是把碰撞概率直接清零，而是改变预计最近距离，再重新计算残余碰撞概率。程序先求满足 $P_c\le0.1P_{{th}}$ 的最小 $b_{{target}}$，再由

$$\Delta v_{{req}}=\frac{{1000(b_{{target}}-b)}}{{3600L}}$$

计算速度增量。若所需 Δv 超过 1 m/s，则执行受限机动并按实际新距离重新计算，不能强行写成目标概率。本次平均 Δv 为 {risk['mean_delta_v_mps']:.8f} m/s，p95 为 {risk['p95_delta_v_mps']:.8f} m/s。

## 12. 点火与通信降级时间

规避持续时间包括推进器点火时间，以及姿态调整、天线恢复和重新定轨时间。点火时间为 $t_{{burn}}=m\Delta v/F$，完整降级窗口为 $\tau=t_{{burn}}+t_{{overhead}}$。本次平均和 p95 降级时间分别为 {risk['mean_degraded_duration_hours']:.6f} h 和 {risk['p95_degraded_duration_hours']:.6f} h；机动期间容量为 10 Gbps，正常为 20 Gbps，失效为 0。

## 13. 单星年度结果

单星年均候选交会数为 {risk['mean_annual_conjunctions']:.6f}，年预期规避次数为 {risk['expected_annual_maneuvers']:.6f}。避撞前年碰撞概率为 {risk['annual_collision_probability_pre']:.8e}，避撞后为 {risk['annual_collision_probability_post']:.8e}，保守年失效概率为 {risk['annual_failure_probability']:.8e}，风险降低率为 {risk['risk_reduction_ratio']:.4%}。$P_c>P_{{th}}$ 只触发决策，不表示必然碰撞或失效。

## 14. 1840 星年度统计与成本

正式仿真利用泊松叠加性质逐年抽样，等价于每星独立过程后合并。本次星座年度避撞次数均值为 {constellation['mean_annual_avoidance_count']:.4f}，标准差 {constellation['std_annual_avoidance_count']:.4f}；年度避撞成本均值为 {_money_yi(constellation['mean_annual_avoidance_cost_cny'])}。年预期失效卫星数为 {constellation['expected_annual_failure_count']:.8f}。

## 15. 容量下降解析与仿真

任意时刻单星正在机动的概率近似为 $1-\exp[-\nu E(\tau)/8760]$，因此机动导致的平均容量下降比例为其一半。解析值为 {capacity['maneuver_analytical_capacity_loss_ratio']:.8%}，示例年蒙特卡洛值为 {capacity['maneuver_mc_capacity_loss_ratio']:.8%}。程序通过事件区间叠加处理重叠，不把持续时间简单相加到 100% 以上。

## 16. 对问题三业务的影响

问题四把问题三三级 LP 的标量 20 Gbps 容量扩展为逐星容量向量。峰值时刻正常吞吐量为 {normal_peak['throughput_gbps']/1000:.6f} Tbps、服务比例 {normal_peak['service_ratio']:.6%}；“避撞加一颗条件失效压力情景”为 {combined_peak['throughput_gbps']/1000:.6f} Tbps 和 {combined_peak['service_ratio']:.6%}。这项压力情景不是无条件年度均值，具体状态基准已写入 CSV。

## 17. 基本覆盖的两个口径

等效服务覆盖令正常、机动、失效权重分别为 1、0.5、0，要求每个网格 $\sum_s a_sI_{{gs}}\ge1$；几何覆盖只检查 $a_s>0$，因此机动卫星仍有几何覆盖。主结论使用等效服务覆盖，同时报告几何口径。第三小问中的 99% 要求是基本覆盖时间比例，而不是问题三全部高流量需求的满足比例。

## 18. 7 天同轨调整

碰撞失效后前 7 天删除该星的覆盖贡献；7 天后，基本覆盖判定视为同轨重排恢复，但物理总容量仍少 20 Gbps，直到备用接替或补充入轨。若同轨有在轨备用，接替延迟使用 6 h；一颗备用只能替代一次失效。

## 19. 全年事件驱动蒙特卡洛

程序不建立 1840×1840 稠密矩阵，也不建立“MC×时间×网格×卫星”数组。它预计算 24 h 代表周期的稀疏覆盖矩阵，从实际低重数网格识别关键卫星比例，再对独立泊松事件生成时间区间并求并集。全年几何相位映射到代表周期，这是降低计算量的主假设。

## 20. 每轨在轨备用方案

每轨 k 颗备用共增加 40k 颗。备用正常不重复增加地面覆盖计数，失效后消耗一颗并经 6 h 接替；备用自身按风险因子 1 承担碎片风险。该方案响应快，但初始制造和发射成本较高。

## 21. 额外轨道面方案

每个额外轨道面增加 46 颗服务卫星。新轨道面依次放入当前最大 RAAN 间隙，初相位在一个面内星间距范围内网格搜索，以覆盖可用率、最小重数、关键卫星比例和平均重数排序。原 40 面不被重新均匀移动。

## 22. 地面备用方案

地面备用在发射前不提供覆盖、容量或在轨风险。首颗未补充失效触发流程，基准准备入轨时间 60 d、单箭最多 60 星。它不能消除最初 7 天覆盖缺口，主要改善长期物理容量和库存恢复。

## 23. 纯方案比较

无冗余平均基本覆盖可用率为 {no_redundancy['mean_effective_coverage_availability']:.6%}，5%分位为 {no_redundancy['q05_effective_coverage_availability']:.6%}。

{chr(10).join(pure_rows)}

## 24. 混合方案和 Pareto 前沿

Standard 枚举 `(k,r,G)`，先按平均 99% 和 5%分位 99% 判据分类，再比较五年成本、鲁棒可用率、新增在轨卫星数和发射次数。最低成本可行混合方案为：{_scheme_text(best_mixed)}。Pareto 表保留成本更低且可用率不被其他方案同时支配的候选。

## 25. 成本模型

新增在轨星和额外轨道面按 500 万元/星制造，并按 $\lceil R/60\rceil$ 次、2 亿元/次发射。地面库存先计制造费，实际补充再计期望发射和补充制造。五年成本不包含题目未给出的存储、保险、人员、折旧和资金时间价值。

## 26. 最终推荐

选择准则为 `{best['selection_criterion']}`。推荐方案：{_scheme_text(recommended)}。其初始成本为 {_money_yi(recommended['initial_cost_cny'])}，初始发射 {int(recommended['initial_launch_count'])} 次。若无冗余已经满足鲁棒判据，则零新增方案自然是最低成本解，不应为了得到正备用数而人为提高风险。

## 27. 敏感性分析

`risk_sensitivity.csv` 比较阈值、检测率、条件失效概率和碎片尺寸指数；配置还记录投影面积、速度、筛选半径、预警、位置误差、推力和恢复开销范围。阈值降低时规避次数不应减少；检测率降低不会降低风险；更大截面积和速度提高长期通量风险。

## 28. 模型优点

模型严格区分长期危险度、单次预测概率和条件失效概率；避撞后重新计算而非清零；输入选择可审计且拒绝 Quick；覆盖使用稀疏矩阵；冗余同时比较可靠性、容量和五年成本。随机种子、配置、候选选择和输出均可复核。

## 29. 模型局限

碎片环境取均匀数密度与简化尺寸谱，没有高度、倾角和地方时结构；交会协方差采用二维各向同性；未建模相关碎片云；同轨调整使用 7 天阶跃恢复；全年地面几何映射到 24 h 代表周期；问题三 ISL 仍无容量约束；外部假设尚需真实运行数据或权威资料校准。

## 30. 可直接用于论文的问题分析模板

在既定二重覆盖 Walker 星座下，问题四的关键不是重新优化轨道，而是把低概率碎片事件转化为可执行的避撞决策和系统级可靠性指标。为此，先由碎片通量得到长期背景危险度，再对候选交会建立最近距离—位置误差概率模型，并以单次预测概率阈值触发机动。随后将单星事件过程扩展为独立泊松星座过程，把机动、失效和备用接替映射为逐星服务权重，最终在基本覆盖约束下比较冗余成本。

## 31. 可直接用于论文的模型建立模板

本文采用截断幂律描述碎片尺寸，以截断正态描述相对速度，以三角分布描述预警时间，并在交会平面内采用二维各向同性高斯误差。单次碰撞概率由非中心卡方分布计算。避撞目标不是令概率为零，而是增加预计最近距离；速度增量由提前时间和所需位移反推，并受 1 m/s 约束。

## 32. 可直接用于论文的模型求解模板

首先对候选交会样本向量化抽样，核验“候选率×平均单次概率”与通量模型一致；其次用相同事件样本比较不同阈值，得到年度规避、残余风险和降级时间；再使用泊松叠加生成 1840 星年度统计。可靠性阶段预计算代表周期稀疏覆盖矩阵，通过事件区间并集计算全年可用率，并枚举在轨备用、额外轨道面和地面备用组合。

## 33. 可直接用于论文的结果分析模板

基准参数下，单星避撞把年碰撞概率从 {risk['annual_collision_probability_pre']:.8e} 降至 {risk['annual_collision_probability_post']:.8e}，但残余风险不为零。星座年均避撞 {constellation['mean_annual_avoidance_count']:.4f} 次，对总容量的平均影响为 {capacity['maneuver_analytical_capacity_loss_ratio']:.8%}。推荐配置的平均与 5%分位基本覆盖可用率分别为 {recommended['mean_effective_coverage_availability']:.6%} 和 {recommended['q05_effective_coverage_availability']:.6%}，需结合假设敏感性解释，不能外推为真实在轨统计。

## 34. 结论使用边界

Quick 输出只用于代码验收；Standard 才用于当前论文数值。Full 可进一步加密风险事件、可靠性年份、覆盖时间和网格。若未来获得真实碎片通量、协方差、跟踪率、推进器推力或在轨避撞日志，应替换相应假设并重新运行，而不是保留当前数值不变。

## 35. 复现信息

本次使用配置 `configs/problem4_{config['mode']}.json`，所有实际参数副本保存在 `outputs/problem4/problem4_config_used.json`，问题三候选审计保存在 `selected_problem3_standard.json`，运行阶段和版本信息保存在 `problem4_run_metadata.json`。
"""
    for path in (
        project_root / "problem4_method_notes.md",
        project_root / "docs" / "report" / "problem4_method_notes.md",
        output_dir / "problem4_method_notes.md",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text
