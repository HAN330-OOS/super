# 正式 GHI 评价

formal_metrics.csv 的 Bicubic 和 DRCT-L 指标来自全量正式预测导出的 metrics.json。EDSR 与 SwinIR 来自各自的正式指标汇总。仓库提供 DRCT-L 和 Bicubic 的评价代码。

export_test_predictions.py 保留正式评价计算逻辑，仓库副本调整了默认路径与输出盘符限制。原始脚本、正式权重和固定划分的 SHA‑256 见 formal_run_reference.json。metric_reference.json 记录 PSNR 和 SSIM 使用的训练集 CSI 参考范围。

复算需准备正式样本、权重、清洗版晴空 GHI 数组、用于对齐的原始测试索引与行号映射、晴空 GHI 元数据和原独立测试结果目录。正式晴空 GHI 使用已对齐的派生数组。

    python evaluation/export_test_predictions.py --mode full --code-root DRCT --splits-root splits --samples-root /path/to/samples --checkpoint /path/to/best_epoch_065.pth --clear-ghi /path/to/test_clear_ghi_clean.npy --original-test-csv /path/to/original_test_samples.csv --clear-indices /path/to/test_clear_ghi_indices.npy --clear-metadata /path/to/test_clear_ghi.json --reference-evaluation /path/to/archived-independent-test --output-root outputs/evaluation

verify_test_export.py --run-dir <运行目录> 用于核对已导出的结果。GHI 计算使用 FY-3C 参考和共同有效像元，CSI 预测直接乘以晴空 GHI，保留原始预测尺度。

## EDSR 和 SwinIR 的 GHI 复算

两个脚本读取对应云端备份的 `Extracted` 目录和已保存的正式 CSI 预测。运行前准备与原实验对齐的晴空 GHI、原始卫星样本和正式 DRCT 导出归档。原归档的运行配置记录了清洗行号映射和原评价代码的位置。运行时这些路径须能访问。

使用单独的已存在输出目录，以保留已归档的正式指标。`--check-only` 用于核对 EDSR 与 SwinIR 的评价逻辑。

~~~bash
python evaluation/compute_edsr_ghi.py --check-only
python evaluation/compute_edsr_ghi.py --run --backup-root /path/to/EDSR --dataset-root /path/to/satellite_dataset --formal-drct-root /path/to/formal_run --output-dir /path/to/new_edsr_results
python evaluation/compute_swinir_ghi.py --backup-root /path/to/SwinIR --dataset-root /path/to/satellite_dataset --formal-drct-root /path/to/formal_run --output-dir /path/to/new_swinir_results
~~~
