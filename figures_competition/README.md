# figures_competition

**策划书 / 竞赛用图**（图 4、7、8）输出目录，与 `figures_publication/` 中的**论文识别性能图**（认证矩阵等）分开。

## 生成正式三联 / 二联图（推荐）

```bash
python scripts/generate_competition_figures.py
```

产物：

- `fig4_length_optimization.{png,svg,pdf}` — 三联图：(a) 损耗 (b) 类内/类间距离与 **inter/intra distance ratio**（双 y 轴）(c) 熵；**数据优先** `results/length_optimization_green/tables/per_length_summary.csv`
- `fig7_dual_channel_characterization.{png,svg,pdf}` — 双通道表征
- `fig8_common_mode_suppression.{png,svg,pdf}` — 共模抑制（summary 柱状）

数据溯源见 **`manifest.csv`** 与 **`docs/competition_required_figures_status.md`**。

## 汇总到单一目录（PNG / SVG / PDF / CSV）

把 `figures/`、`figures_publication/`、`figures_competition/`（及本地存在的 `results/length_optimization_green/tables/*.csv`）复制到 **`paper_assets/`**，按格式分子文件夹，便于写论文一次找齐：

```bash
python scripts/collect_paper_assets.py --clean
```

说明：**`docs/paper_assets.md`**（`paper_assets/` 默认在仓库根 `.gitignore` 中，仅在本地生成）。

## 不要混用的图

- **`figures_publication/publication_fig03_length_optimization`**、**`publication_fig04_length_optimization`** 为期刊脚本生成的四联 composite，**不是**当前策划书 Word 中的「图 4」插稿版本。

## 图 4 文字口径（总长 / 比值）

见 **`docs/competition_figure_text_check.md`**（Total fiber length、inter/intra distance ratio 等）。
