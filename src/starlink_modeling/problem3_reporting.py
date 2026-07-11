"""Result serialization and Chinese method notes for problem three."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any

from .problem3_io import SelectedConstellation


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any, digits: int = 4, suffix: str = "") -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "未计算"
    if not math.isfinite(value):
        return "未计算"
    return f"{value:.{digits}f}{suffix}"


def _status_text(summary: dict[str, Any] | None, key: str, digits: int = 4, suffix: str = "") -> str:
    return _number(summary.get(key), digits, suffix) if summary else "本次尚未运行"


def write_method_notes(
    project_root: Path,
    output_dir: Path,
    selected: SelectedConstellation,
    config: dict[str, Any],
    run_mode: str,
) -> Path:
    topology = _read_json(output_dir / "topology_summary.json")
    routing = _read_json(output_dir / "routing_summary.json")
    traffic = _read_json(output_dir / "traffic_summary.json")
    traffic_real = bool(traffic and traffic.get("status") == "real_data_loaded")
    traffic_demo = bool(traffic and traffic.get("is_synthetic_demo"))
    mode_warning = (
        "> **结果使用限制：** 本文档当前插入的是 Quick 冒烟结果，只用于验证代码，不能作为论文最终数值结论。\n"
        if run_mode == "quick"
        else ""
    )
    validation_warning = (
        "> **星座输入限制：** 所选 Problem 2 Standard 候选在主网格上满足单重连续覆盖，"
        f"但其记录的验证状态为 `{selected.validation_status}`。Problem 3 固定继承该输入，"
        "不把它重新解释为已通过 Full 验证。\n"
    )

    topology_result = (
        f"同轨相邻距离解析值为 {_status_text(topology, 'same_plane_distance_analytic_km', 3, ' km')}，"
        f"跨轨活动链路距离范围为 {_status_text(topology, 'cross_plane_distance_min_km', 3, ' km')} 至 "
        f"{_status_text(topology, 'cross_plane_distance_max_km', 3, ' km')}。"
        f"网络全连通时间比例为 {_status_text(topology, 'network_connected_time_ratio', 4)}，"
        f"观测到的最大节点度为 {_status_text(topology, 'maximum_degree', 0)}，"
        f"拓扑切换累计 {_status_text(topology, 'topology_switch_count', 0)} 次。"
        if topology
        else "拓扑阶段本次尚未运行，因此不填写数值。"
    )
    routing_result = (
        f"可达请求平均时延为 {_status_text(routing, 'mean_delay_ms', 4, ' ms')}，"
        f"95% 分位时延为 {_status_text(routing, 'p95_delay_ms', 4, ' ms')}，"
        f"最大时延为 {_status_text(routing, 'max_delay_ms', 4, ' ms')}，"
        f"不超过 30 ms 的比例为 {_status_text(routing, 'ratio_below_30ms', 4)}。"
        f"平均 ISL 跳数为 {_status_text(routing, 'mean_hops', 3)}，"
        f"最大跳数为 {_status_text(routing, 'max_hops', 0)}。"
        if routing
        else "路由阶段本次尚未运行，因此不填写数值。"
    )
    if traffic_real:
        traffic_result = (
            f"实际数据换算后的区域平均需求为 {_status_text(traffic, 'actual_average_demand_gbps', 4, ' Gbps')}；"
            f"优化方案平均吞吐量为 {_status_text(traffic, 'optimized_mean_throughput_gbps', 4, ' Gbps')}，"
            f"阻塞比例为 {_status_text(traffic, 'optimized_blocked_ratio', 4)}，"
            f"最大卫星利用率为 {_status_text(traffic, 'maximum_satellite_utilization', 4)}。"
        )
    elif traffic_demo:
        traffic_result = (
            "流量阶段使用 `synthetic_demo` 完成代码验收。仅作为程序复核，本次优化方案平均吞吐量为 "
            f"{_status_text(traffic, 'optimized_mean_throughput_gbps', 4, ' Gbps')}，"
            f"阻塞比例为 {_status_text(traffic, 'optimized_blocked_ratio', 4)}，"
            f"最低公平服务比例为 {_status_text(traffic, 'minimum_service_ratio', 4)}，"
            f"最大卫星利用率为 {_status_text(traffic, 'maximum_satellite_utilization', 4)}；"
            "这些数值不是实际区域通信数据，不得写入论文最终结论。"
        )
    elif traffic:
        traffic_result = "未提供真实流量文件，流量阶段状态为 `skipped_missing_real_data`，未虚构结果。"
    else:
        traffic_result = "流量阶段本次尚未运行。"

    source_names = "、".join(traffic.get("source_names", [])) if traffic else ""
    source_urls = "、".join(traffic.get("source_urls", [])) if traffic else ""
    source_names = source_names or "未提供"
    source_urls = source_urls or "未提供"
    content = rf"""# 问题三：星间链路与通信路由优化方法说明

{mode_warning}{validation_warning}

## 1. 问题三到底在研究什么

问题一研究单颗卫星和单轨道面能覆盖什么纬度范围，问题二进一步研究多轨道面星座怎样覆盖目标区域。问题三不再改变星座规模，而是把问题二得到的每颗卫星看成动态通信网络中的节点。第一小问回答“卫星之间的路是否存在”，第二小问回答“一个地面通信请求走哪条星间路径最快”，第三小问回答“大量地面业务同时接入时怎样分流，才能兼顾公平、吞吐量和负载均衡”。

## 2. Problem 2 Standard 星座输入

程序实际读取 `{selected.source_file}`，SHA-256 为 `{selected.source_sha256}`。星座参数为：轨道面数 $M={selected.M}$，每面卫星数 $N={selected.N}$，总卫星数 {selected.total_satellites}；倾角 $i={selected.inclination_deg:.8f}^\circ$，相位因子 $F={selected.phase_factor_F}$，初始升交点 $\Omega_0={selected.Omega0_deg:.8f}^\circ$，初始纬度辐角 $u_0={selected.u0_deg:.8f}^\circ$，高度 $h={selected.altitude_km:.1f}$ km，覆盖地心角 $\theta={selected.theta_deg:.8f}^\circ$。轨道面升交点按 $\Omega_p=\Omega_0+360^\circ p/M$ 均匀布置，面内初相位与相邻面相位差继续使用 Problem 2 的 Walker 规则。

问题三固定继承这些参数，因为题意研究的是给定星座的链路和路由。若再次调整 $M,N,i,F,\Omega_0,u_0$，就会把问题三错误地变成新的星座设计问题。

## 3. 时变图与星间拓扑

在时刻 $t$ 建立无向图 $G(t)=(V,E(t))$。节点 $V$ 是全部 {selected.total_satellites} 颗卫星，节点编号统一为 $s=pN+q$。边 $E(t)$ 是当时同时满足通信距离和地球视线条件的链路。卫星位置直接复用 Problem 2 的圆轨道二体传播：$u_{{pq}}(t)=u_0+2\pi q/N+2\pi Fp/(MN)+nt$，$n=\sqrt{{\mu/(R+h)^3}}$，再由倾角和升交点旋转得到 ECI 三维坐标。

同轨前后相邻卫星的相位差恒为 $2\pi/N$，故距离恒为

$$d_{{same}}=2(R+h)\sin(\pi/N).$$

跨轨链路随两轨道面空间夹角和卫星纬度辐角周期变化。对相邻面枚举循环偏移 $\kappa$，按“可用边数最大、最大距离最小、总距离最小”的字典序选取一一匹配 $(p,q)\leftrightarrow(p+1,(q+\kappa)\bmod N)$。这一规则避免多颗卫星独立争抢同一邻星；每颗卫星固定最多两条同轨边、左侧一条跨轨边和右侧一条跨轨边，因此总度不超过 4。

链路距离条件为 $d_{{ab}}\le 5000$ km。对线段 $\mathbf r(\lambda)=\mathbf r_a+\lambda(\mathbf r_b-\mathbf r_a)$，将 $\lambda$ 截断到 $[0,1]$ 后计算距地心最短距离；只有该距离大于地球半径时才具有视线。即使本星座中距离阈值通常已经排除了遮挡，代码仍独立执行视线判定。

## 4. 虚拟源节点和虚拟汇节点

虚拟源 A 和虚拟汇 B 不是实际设备，而是最短路建模工具。地面 A 往往同时看见多颗卫星，B 也可能看见多颗卫星。把 A 与全部可见入口星相连、把全部可见出口星与 B 相连后，Dijkstra 在一次计算中同时决定入口星、星间中继路径和出口星。

例如 A 可见卫星 1 和 2。卫星 1 的上行斜距更短，但其后续星间路径需要 8 跳；卫星 2 的上行斜距略长，却只需 3 跳。虚拟源方法会比较完整端到端时延，而预先强制选择最近的卫星 1 可能得到更慢的路径。

## 5. 端到端时延

地面点到卫星使用真实斜距 $l_{{gs}}=\|\mathbf r_s-\mathbf r_g\|$，不能用 550 km 高度代替。若路径经过 $K$ 颗卫星，星间总距离为 $L_{{ISL}}$，则主口径为

$$\tau_{{prop}}=\frac{{l_{{up}}+L_{{ISL}}+l_{{down}}}}{{c}},\qquad
\tau_{{proc}}=K\times0.5\text{{ ms}},\qquad
\tau_{{total}}=\tau_{{prop}}+\tau_{{proc}}.$$

同时计算仅对 ISL 跳数计处理时延的敏感性口径 $(K-1)\times0.5$ ms。拓扑采用准静态快照，即假设一个数据包传播的几十毫秒内卫星位置不显著改变。30 ms 判据分别检查平均值、95% 分位数、最大值和满足比例，不能只凭平均值合格就声称所有请求合格。

## 6. 区域平均与最大时延

单独选择一对地面点不能代表 4°N—53°N、73°E—135°E 的区域性能。程序先生成与 Problem 2 口径一致的地面网格，再以面积权重抽取点对，并强制包含角点、边界点、中心点和大跨度点对。平均值反映典型请求，95% 分位数反映大多数高时延请求的上界，最大值用于发现最弱位置和时刻。最大值仍依赖离散时间、网格和抽样密度，因此不是连续时空上的严格数学上界。

## 7. 第三小问到底优化什么

第三小问固定卫星数量、轨道参数、当前卫星位置、地面可见关系、星间拓扑和单星 20 Gbps 接入容量。真正的决策变量是 $x_{{gs}}(t)$，即地区 $g$ 在时刻 $t$ 分配给可见卫星 $s$ 的 Gbps 流量。比如某地区有 8 Gbps 需求，可以分成 5 Gbps 给卫星 1、3 Gbps 给卫星 2。因此优化本质是“地面业务接入负载均衡”，不是重新设计星座。

## 8. 为什么保留 20 Gbps、暂不限制 ISL 容量

20 Gbps 是题目明确给出的单星接入容量。若连这一约束也忽略，吞吐量恒等于需求，第三小问将失去拥塞控制含义。题目没有给出每条激光 ISL 的容量，因此主模型不编造链路带宽，只让 ISL 影响路径是否存在和端到端时延。该简化不能判断某条 ISL 是否发生带宽拥塞；获得链路容量后可扩展为带容量的多商品流模型。

## 9. 实际区域流量数据处理

数据接口支持平均 Gbps/Tbps、统计周期累计 GB/TB/PB/EB 和带时间戳的速率曲线。累计量按“字节数乘 8，再除以周期秒数和 $10^9$”换算为平均 Gbps。当前数据来源名称：{source_names}；来源链接：{source_urls}。配置中的卫星承载比例为 $\eta_{{sat}}={config['traffic']['eta_sat']}$；若取 1，只表示全部区域流量进入卫星系统的压力上界，不代表现实市场占有率。平均流量采用均值为 1、峰均比 1.5 的平滑周期曲线，空间权重按球面面积归一化，而不是给每个经纬度采样点相同流量。

## 10. 公平流量分配模型

为防止优化器只服务容易覆盖的中心地区，引入所有地区共同的服务比例 $r(t)\in[0,1]$：

$$\sum_{{s\in V_g(t)}}x_{{gs}}(t)=r(t)d_g(t),\quad
\sum_gx_{{gs}}(t)\le20,\quad x_{{gs}}(t)\ge0.$$

第一阶段最大化 $r(t)$。吞吐量为 $r(t)D(t)$，阻塞量为 $[1-r(t)]D(t)$。不可见的地区—卫星组合不创建变量，从结构上保证不会把业务分配给不可见卫星。

## 11. 负载均衡和时延优化

在最大服务比例 $r^*$ 固定后，第二阶段最小化最大卫星利用率 $u$，约束 $L_s/20\le u$。第三阶段在不降低 $r^*$、不恶化最优 $u$ 的前提下最小化分配时延代价。三阶段按字典序求解，避免把公平、负载和时延随意加权后因量纲或权重选择改变主目标。

## 12. 基准方案与优化方案

基准策略 `nearest_visible_satellite` 先把地区流量交给斜距最近的可见卫星，容量满后再溢出到次近星。它容易使地理上集中的业务压到少数卫星。优化方案允许多星分流，以可能的少量接入时延增加换取更高公共服务比例、更低最大利用率和更均匀负载。比较指标包括吞吐量、阻塞量、最低服务比例、最大利用率、利用率标准差、接入代价和切换次数。

## 13. 结果分析

拓扑结果：{topology_result}

路由结果：{routing_result}

流量结果：{traffic_result}

## 14. 流量工程建议

若最大利用率接近 1 或服务比例低于 1，应优先采用多星分流，并在高峰期使用预约或公平限流；接入选择应同时考虑斜距和卫星负载。若跨时刻分配变化频繁，可加入切换惩罚和滞回阈值。边缘地区应保留共同最低服务比例。未来若得到 ISL 容量，应把当前接入 LP 扩展为多路径、多商品流模型，不能直接沿用“ISL 无容量约束”的吞吐量结论。

## 15. 模型优点与局限

模型直接继承 Problem 2 的实际输出，采用可复现的时变图、可解释的最短路和公平接入 LP，并保留真实流量数据接口。局限包括圆轨道二体传播、离散时间快照、区域网格化、采样最大时延、暂不限制 ISL 容量，以及未考虑误码、天气和激光捕获时间。实际统计流量的业务口径也可能不等于卫星业务量，$\eta_{{sat}}$ 只能作为情景参数。

## 16. 可直接用于论文的文字模板

**问题分析段。** 在问题二确定区域覆盖星座后，问题三需要进一步判断该星座能否形成稳定的星间通信网络，并研究区域通信请求的最小时延路由以及高并发业务下的接入分流。为避免改变前问结论，本文固定采用 Problem 2 Standard 输出的 {selected.M}×{selected.N} Walker 星座。

**模型建立段。** 本文以卫星为节点、满足 5000 km 与地球视线条件的激光链路为边建立时变图。同轨节点连接前后邻星，跨轨节点采用相邻轨道面循环移位一一匹配。地面端到端路由通过虚拟源汇与多入口 Dijkstra 同时选择入口星、中继路径和出口星。业务分配采用“最大公平服务比例—最小最大利用率—最小时延代价”的三级线性规划。

**拓扑结果段。** {topology_result}

**路由结果段。** {routing_result}

**流量优化结果段。** {traffic_result}

**结论段。** 上述结论仅在记录的星座输入、离散快照、网格分辨率、点对样本和容量假设下成立。Quick 与演示流量只用于程序验收；论文最终结论必须使用 Standard 或 Full 路由结果及可核验的真实流量数据。
"""
    root_path = project_root / "problem3_method_notes.md"
    report_path = project_root / "docs" / "report" / "problem3_method_notes.md"
    output_path = output_dir / "problem3_method_notes.md"
    for path in (root_path, report_path, output_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    if root_path.resolve() != report_path.resolve():
        shutil.copystat(root_path, report_path)
    return root_path
