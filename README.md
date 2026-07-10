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
