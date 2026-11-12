# 论文绘图

从仓库根目录运行。Python 依赖见 [requirements.txt](requirements.txt)，渲染需要 Times New Roman。输出目录放在本地 `outputs/`。论文图的模板、正式结果数组和原始卫星数据单独归档。

~~~bash
python -m pip install -r figures/requirements.txt
~~~

## 图 1 · DRCT 架构

输入是调整标签前的架构图 `figure1_before_edits.pdf`。在非 Windows 系统通过 `--font` 和 `--bold-font` 指定对应字体文件。输出目录需要是新目录。

~~~bash
python figures/draw_figure1.py --source-pdf external_data/paper_figures/figure1_before_edits.pdf --output-dir outputs/paper/figure1
~~~

## 图 2 · GHI 密度散点

使用正式输出目录中的 `ghi_scatter_data.npz`、运行记录、完整测试指标与校验报告。脚本使用原始散点，重新计算 Viridis 密度着色。测试 CSV 默认使用仓库的正式索引。

~~~bash
python figures/draw_figure2_viridis.py --run-dir /path/to/formal_run --samples-root /path/to/samples --output-dir outputs/paper/figure2
~~~

## 图 3 · CSI 对比

`--selected-dir` 指向正式十样本的导出目录，里面有 `selected_csi_data.npz`、选样记录和校验过的原图。`--template` 指定原版式 PDF。`--current-pdf` 指定改用分位数色标之前的 CSI 图。

~~~bash
python figures/draw_figure3_quantiles.py --run-dir /path/to/formal_run --samples-root /path/to/samples --selected-dir /path/to/selected_examples --template external_data/paper_figures/figure45_4rows.pdf --current-pdf external_data/paper_figures/figure3_before_quantiles.pdf --output-dir outputs/paper/figure3
~~~

四行依次显示 FY-4A、Bicubic、DRCT 和 FY-3C。所有图块使用同一组分位数色标，图像数据按正式选样记录读取。
