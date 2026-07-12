# 星链系统低轨卫星星座建模
本仓库用于数学建模比赛模拟练习，主题为低轨卫星星座建模与分析。

## 目录结构

- `data/raw/`: 原始数据，不直接修改。
- `data/processed/`: 清洗、转换后的中间数据。
- `data/external/`: 外部公开数据或第三方数据说明。
- `docs/problem/`: 题面、附件和赛题原始资料。
- `docs/references/`: 文献、标准、背景资料和链接摘录。
- `docs/notes/`: 建模过程记录、假设讨论、会议记录。
- `docs/report/`: 论文正文、方法说明和最终报告草稿。
- `src/starlink_modeling/`: 可复用建模、仿真、优化和可视化代码。
- `notebooks/`: 探索性分析和临时实验。
- `scripts/`: 可重复执行的数据处理、实验和绘图脚本。
- `models/`: 模型参数、拟合结果和可复现实验配置。
- `outputs/`: 图片、表格、日志和计算结果。
- `submissions/`: 最终提交材料。
- `tests/`: 关键函数和模型计算的测试。

## 问题一：单轨道面覆盖特性分析

当前实现位于根目录脚本：

```powershell
python problem1_latitude_coverage.py
```

如果 Windows 上的 `python` 指向 Microsoft Store 占位程序，可改用：

```powershell
py problem1_latitude_coverage.py
```

安装依赖：

```powershell
py -m pip install -r requirements.txt
```

运行后会自动生成：

- `outputs/problem1/single_satellite_coverage_summary.csv`
- `outputs/problem1/inclination_min_satellites.csv`
- `outputs/problem1/coverage_metrics_grid.csv`
- `outputs/problem1/fig_single_satellite_geometry.png`
- `outputs/problem1/fig_ground_track_example.png`
- `outputs/problem1/fig_latitude_coverage_time_example.png`
- `outputs/problem1/fig_min_satellites_vs_inclination.png`
- `outputs/problem1/fig_fixed_N_min_coverage_vs_inclination.png`
- `outputs/problem1/fig_fixed_N_mean_coverage_vs_inclination.png`
- `outputs/problem1/fig_coverage_heatmap_i_N.png`
- `outputs/problem1/fig_upper_reach_vs_inclination.png`
- `outputs/problem1/fig_spacing_full_coverage_ratio.png`
- `outputs/problem1/fig_spacing_redundancy_index.png`
- `outputs/problem1/fig_coverage_time_representative_cases.png`
- `outputs/problem1/fig_interval_snapshot_feasible_vs_infeasible.png`
- `problem1_method_notes.md`
- `docs/report/problem1_method_notes.md`

`problem1_method_notes.md` 是给论文撰写者使用的方法说明，覆盖“问题分析、模型建立、模型求解、结果分析”的主要内容，并会根据程序实际输出自动插入关键数值。

注意：本模型解决的是问题一中的单轨道面纬度投影覆盖分析，不等价于目标区域的完整二维连续覆盖。完整二维覆盖需要在问题二中引入经纬度网格、多个轨道面和升交点布局。

## 问题二：多轨道面星座组网优化设计

问题二使用 Walker-Delta 规则星座、三维 ECI/ECEF 轨道传播、二维地面网格覆盖评价和混合整数遗传算法，分别搜索全时单重覆盖方案与 95% 时间全区域二重覆盖方案。

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

运行方式：

```powershell
python problem2_constellation_ga.py --mode quick
python problem2_constellation_ga.py --mode standard
python problem2_constellation_ga.py --mode full
```

默认运行 `standard`，并同时计算 `single` 与 `double`。也可使用 `--scenario`、`--seed`、`--workers` 和 `--resume` 控制场景、随机种子、并行线程数和检查点恢复。例如：

```powershell
python problem2_constellation_ga.py --mode quick --scenario single --seed 20260710 --workers 4
python problem2_constellation_ga.py --mode standard --scenario both --resume
```

- `quick`：代码冒烟测试，结果不能作为论文最终结论。
- `standard`：主要遗传算法优化，计算时间较长。
- `full`：执行 24 h 高精度与 7 d 中等精度验证，并在候选失效时继续检查更稳健构型。

主要输出位于 `outputs/problem2/`：

- `best_single_constellation.json`、`best_double_constellation.json`：最佳已搜索星座及验证状态。
- `best_single_satellites.csv`、`best_double_satellites.csv`：逐卫星轨道面、升交点和初始相位。
- `single_solution_metrics.csv`、`double_solution_metrics.csv`：覆盖与间隙汇总指标。
- `single_grid_metrics.csv`、`double_grid_metrics.csv`：逐网格点可用率、重数和间隙。
- `single_time_metrics.csv`、`double_time_metrics.csv`：逐时刻 A1、A2、B1、B2。
- `ga_*_generation_history.csv`、`ga_*_all_candidates.csv`：遗传算法历史和实际评价候选。
- `fixed_MN_certification_*.csv`：低卫星数固定构型反向核验。
- `resolution_sensitivity.csv`、`cost_comparison.csv`、`run_metadata.json`：敏感性、成本和复现元数据。
- `checkpoints/`、`problem2_run.log`：恢复检查点和运行日志。
- `fig_problem2_*.png`：区域网格、三维构型、星下点轨迹、收敛、覆盖热力图、最差时刻、成本和敏感性图表，均为实际计算结果。
- `problem2_method_notes.md`、`docs/report/problem2_method_notes.md`：从实际 JSON/CSV 自动生成的论文方法和结果说明。

配置集中存放在 `configs/problem2_quick.json`、`configs/problem2_standard.json` 和 `configs/problem2_full.json`。基准检查表明原建议的 `M≤30` 无法进入当前全区域覆盖可行域，因此配置把 `M` 上限扩展到 60；这一调整保留在配置和运行元数据中，便于复核。

固定 `M,N` 核验会把配置范围内的组合全部列入 CSV，并优先实际搜索当前卫星数以下且最接近下界的组合。默认 `max_pairs` 用于控制总运行时间，未运行组合会明确标记；将其改为 `null` 可执行穷举式核验，但可能需要很长时间。

## 问题三：星间链路与通信路由优化

问题三固定读取 `outputs/problem2/` 中的 Problem 2 Standard 二重覆盖星座 `M=40, N=46, S=1840`，签名为 `standard_double_M40_N46_S1840`，不重新搜索轨道面数、卫星数、倾角或相位参数。默认或手工指定的文件若不是这套二重覆盖星座，程序会直接报错，不会退回单重覆盖方案。星座缓存键包含规模、场景、轨道参数和源文件哈希，输入变化时旧问题三输出自动失效。

安装依赖后运行：

```powershell
python problem3_network_routing.py --mode quick --stage all --allow-demo-traffic
python problem3_network_routing.py --mode standard --stage topology
python problem3_network_routing.py --mode standard --stage routing --pair-mode sampled
python problem3_network_routing.py --mode standard --stage traffic --traffic-file <实际流量CSV>
```

也可以只运行一个阶段：

```powershell
python problem3_network_routing.py --stage topology
python problem3_network_routing.py --stage routing
python problem3_network_routing.py --stage traffic
```

常用参数：

- `--constellation-file <JSON>`：手工指定 Problem 2 星座输出。
- `--traffic-file <CSV>`：提供可核验的实际流量数据。
- `--pair-mode sampled|exact`：面积加权点对抽样或离散网格全部点对分块计算。
- `--seed`：固定抽样和基准分配顺序。
- `--workers`：并行生成不同时间的拓扑快照。
- `--resume`：已有对应阶段汇总文件时跳过该阶段。
- `--allow-demo-traffic`：保留的测试开关，仅允许在 Quick 模式显式启用 `synthetic_demo`；正常运行无需该参数。

三级精度配置位于 `configs/problem3_quick.json`、`configs/problem3_standard.json` 和 `configs/problem3_full.json`：

- `quick`：一个轨道周期、粗网格和少量点对，只用于代码冒烟测试，不能作为论文最终结论。
- `standard`：两个轨道周期的拓扑、24 h 路由和中等网格，是主要计算配置。
- `full`：更密时间步长、0.5° 网格和更多点对，计算耗时很长，用于高精度验证。

真实流量模板和字段说明位于：

- `data/external/problem3_traffic_data_template.csv`
- `data/external/problem3_traffic_data_README.md`

未指定 `--traffic-file` 时，程序默认读取用户给定的 [problem3_region_traffic_density.csv](data/external/problem3_region_traffic_density.csv)：在 `4°N～53°N、73°E～135°E` 完整球面矩形内使用 `7.02139 Mbps/km²`。面积按 $R^2\Delta\lambda(\sin\varphi_{max}-\sin\varphi_{min})$ 计算，约为 `32,013,984.23 km²`；平均需求约为 `224.782669 Tbps`，峰值约为 `337.174003 Tbps`。每个网格按球面面积分配流量，主结果固定 `eta_sat=1`，不自动缩放流量密度。`--traffic-file` 仍可覆盖默认数据源。

主要输出位于 `outputs/problem3/`：

- `selected_problem2_constellation.json`、`satellite_id_mapping.csv`：实际采用的 Problem 2 文件、SHA-256、星座参数和卫星编号。
- `topology_summary.json`、`topology_timeseries.csv`、`link_timeseries.csv`、`link_availability.csv`：时变拓扑、距离、连通性和链路可用率。
- `crosslink_shift_timeseries.csv`、`satellite_degree_timeseries.csv`、`topology_snapshot_edges.csv`：跨轨循环偏移、节点度和代表快照。
- `routing_summary.json`、`routing_time_metrics.csv`、`route_pair_samples.csv`、`worst_delay_route.json`：平均、分位、最大时延及最差路径。
- `traffic_summary.json`、`traffic_timeseries.csv`、`satellite_load_timeseries.csv`、`flow_engineering_comparison.csv`：真实流量审计、公平准入、负载和基准对比。
- `fig_problem3_*.png`：拓扑、链路、路由、时延和流量工程中文图表，均为实际计算输出，300 dpi。
- `problem3_method_notes.md`、`docs/report/problem3_method_notes.md`：自动插入实际结果的论文方法说明。

模型必须区分两类容量：星间链路是否存在由 `5000 km` 距离和地球视线条件决定；单颗卫星可接入多少地面流量由 `20 Gbps` 限制。题目没有提供每条激光 ISL 的容量，因此当前主模型不分析 ISL 内部带宽拥塞，只让 ISL 影响路径与时延；后续获得链路容量后可扩展为多商品流模型。

## 问题四：碎片规避与星座鲁棒性设计

问题四固定继承问题二 Standard 二重覆盖星座以及问题三 Standard 通信结果。程序会递归审计 `outputs/problem3/**/problem3_run_metadata.json`，核对 `mode=standard`、完成状态、`M=40, N=46, S=1840`、星座签名和全部阶段产物。根目录 Quick 结果不会被正式运行自动选用。实际候选、元数据 SHA-256 和选择理由写入 `outputs/problem4/selected_problem3_standard.json`。

安装依赖后运行：

```powershell
python problem4_debris_robustness.py --mode quick
python problem4_debris_robustness.py --mode standard
python problem4_debris_robustness.py --mode full
```

默认执行 `standard --stage all`。也可以分阶段运行：

```powershell
python problem4_debris_robustness.py --mode quick --stage risk --force
python problem4_debris_robustness.py --mode quick --stage constellation --resume
python problem4_debris_robustness.py --mode quick --stage redundancy --resume
```

- `risk`：碎片尺寸和速度分布、候选交会、非中心卡方单次碰撞概率、避撞阈值、Delta-v 和残余风险。
- `constellation`：1840 星年度避撞次数、成本、通信降级、容量损失以及问题三逐星容量 LP 复算。
- `redundancy`：稀疏覆盖矩阵、全年事件驱动蒙特卡洛、三类冗余方案、五年成本和 Pareto 前沿。

常用参数：

- `--problem3-dir <目录>`：明确指定待审计的问题三 Standard 目录。
- `--config <JSON>`：在所选精度默认配置上覆盖参数。
- `--seed`、`--mc-runs`、`--threshold`：控制随机种子、可靠性年份和基准避撞阈值。
- `--resume`：跳过同一模式下已完成的阶段；不同模式不会复用前一模式的风险结果。
- `--force`：仅清理并重建 `outputs/problem4/`，不会改动问题一至三输出。
- `--no-traffic-impact`：跳过问题三 LP 复算，适合仅调试风险模型；完整图表和论文说明需要保留业务影响计算。

三级配置位于 `configs/problem4_quick.json`、`configs/problem4_standard.json` 和 `configs/problem4_full.json`：

- `quick`：2 万交会样本、50 个可靠性年份、120 min/3° 覆盖采样，仅用于冒烟测试。
- `standard`：100 万交会样本、1000 个可靠性年份、30 min/1° 覆盖采样，是论文主要配置。
- `full`：500 万交会样本、5000 个可靠性年份、10 min/0.5° 覆盖采样，用于高精度复核，耗时和内存需求很高。

模型严格区分长期通量碰撞危险度、单次近距离交会预测概率和碰撞后的条件失效概率。候选事件率为 $n\pi B^2E[V]$；单次预测概率使用二维各向同性高斯对应的非中心卡方分布；机动通过改变预测最近距离重新计算残余概率，不把风险直接设为零。降级时间由 $m\Delta v/F$ 点火时间与姿态、天线和重新定轨恢复开销组成。

第三小问比较三类方案：每轨在轨备用、在原 RAAN 最大间隙加入额外轨道面、地面库存补充；还会枚举受控范围内的混合方案。主判据为平均全年基本覆盖可用率不低于 99%，同时报告 5% 分位不低于 99% 的鲁棒判据。这里的“基本覆盖”不是问题三高流量需求的全部满足比例。

参数来源和假设记录在：

- `data/external/problem4_parameter_sources.csv`
- `data/external/problem4_parameter_assumptions.md`

主要输出位于 `outputs/problem4/`：

- `single_satellite_risk_summary.json`、`single_satellite_threshold_comparison.csv`、`risk_model_consistency.json`、`risk_sensitivity.csv`：单星年度风险、阈值和通量一致性。
- `collision_event_samples.csv`、`maneuver_delta_v_distribution.csv`、`maneuver_duration_distribution.csv`：可审计事件样本和机动分布。
- `constellation_annual_summary.json`、`annual_avoidance_distribution.csv`、`capacity_loss_summary.json`：星座年度次数、成本与容量损失。
- `problem3_service_impact.csv`：正常、避撞、条件失效和备用接替状态下的问题三 LP 结果。
- `redundancy_monte_carlo_summary.csv`、`redundancy_year_samples.csv`、`redundancy_pareto_front.csv`、`best_redundancy_solution.json`：可靠性、成本和最优方案。
- `fig_problem4_*.png`：碎片、碰撞、避撞、容量、通信影响、可靠性、成本与敏感性中文图表，均为 300 dpi。
- `problem4_method_notes.md`、`docs/report/problem4_method_notes.md`：自动读取本次实际结果生成的论文方法与结果说明。

Standard 和 Full 运行过慢时，可按 `risk`、`constellation`、`redundancy` 顺序运行并使用 `--resume`。程序捕获中断并保留已完成阶段元数据，不会静默降低为 Quick 参数。

## 工作约定

1. 先记录假设和符号，再写模型与代码。
2. 原始数据只放在 `data/raw/` 或 `docs/problem/`，处理后的结果放入 `data/processed/` 或 `outputs/`。
3. 每次完成一组有效更改后执行：

```powershell
git status
git add <changed-files>
git commit -m "说明本次更改"
git push
```
