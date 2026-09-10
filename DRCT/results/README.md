# 实验结果目录

本目录仅保留当前正式实验：`fy4a_fy3c_drctl_w8_scratch_50k_b16_e50`。该实验使用 FY-4A / FY-3C CSI 配对样本，从头训练单通道 DRCT-L ×4 模型，并在第 271 个 epoch 停止。

```text
fy4a_fy3c_drctl_w8_scratch_50k_b16_e50/
├── checkpoints/            # checkpoint；默认不提交大权重文件
├── training_cache_float32/ # 精确续训所需缓存；默认不提交
├── visuals/                # 每个 epoch 的固定五样本可视化
├── final/                  # 最终定量指标和论文图
├── training_log.csv        # 每个已完成 epoch 一行记录
├── run_config.json         # 实际训练配置
└── *_subset.json           # 可复现训练、验证样本选择
```

最终使用的模型权重为 `checkpoints/epoch_271_converged_final.pth`。由于体积较大，权重文件不随仓库上传；请从本地归档或发布附件中单独获取。

`final/paper_figures/` 中保存论文使用的十样本对比图、GHI 散点图及其 SVG 版本。`final/` 的 JSON/CSV 文件记录独立 10,000 对测试集上的汇总结果。
