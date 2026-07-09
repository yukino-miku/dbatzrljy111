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
