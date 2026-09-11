# 绘图脚本

本目录只存放读已有数据、生成论文图或检查图的脚本；不会修改 FY 原始数据、匹配结果或训练样本。主处理流程仍在上一级的 `step_*.py` 与 `fy_io.py`。

在项目根目录执行，例如：

```powershell
py -3 FY_SSI_processing\plots\plot_fig5_1_fy4a_fy3c_products.py
py -3 FY_SSI_processing\plots\plot_fig5_2_time_matching.py
py -3 FY_SSI_processing\plots\plot_two_swaths.py --date 20180501
```

图 5-1 至图 5-6 分别由 `plot_fig5_1_*.py` 至 `plot_fig5_6_*.py` 生成。`matching_figure_data.py` 和 `prepare_fig5_6_native_sample.py` 是这些绘图脚本的内部辅助模块，一般不需要单独运行。
