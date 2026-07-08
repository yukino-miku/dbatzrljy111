# 星链系统低轨卫星星座建模

本项目用于数学建模比赛模拟练习，主题为低轨卫星星座建模与分析。

## 目录结构

- `data/raw/`: 原始数据，不直接修改。
- `data/processed/`: 清洗、转换后的中间数据。
- `data/external/`: 外部公开数据或第三方数据说明。
- `docs/problem/`: 题面、附件和赛题原始资料。
- `docs/references/`: 文献、标准、背景资料和链接摘录。
- `docs/notes/`: 建模过程记录、假设讨论、会议记录。
- `docs/report/`: 论文正文、公式推导和最终报告草稿。
- `src/starlink_modeling/`: 可复用建模、仿真、优化和可视化代码。
- `notebooks/`: 探索性分析和临时实验。
- `scripts/`: 可重复执行的数据处理、实验和绘图脚本。
- `models/`: 模型参数、拟合结果和可复现实验配置。
- `outputs/`: 图片、表格、日志和计算结果。
- `submissions/`: 最终提交材料。
- `tests/`: 关键函数和模型计算的测试。

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

4. 若 GitHub 远端尚未配置，先创建远端仓库并设置 `origin`。

