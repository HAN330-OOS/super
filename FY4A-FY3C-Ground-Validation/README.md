# FY-4A / FY-3C 地基验证

本模块读取正式测试集的卫星配对数据，将 FY-4A、FY-3C、Bicubic 和 DRCT 的 GHI 与同一批 CMA 站点观测比较。

## 匹配流程

1. 按站点经纬度查找覆盖该站点的已配对测试图块，定位高、低分辨率像元。
2. 读取图块记录中的 FY-4A 扫描起止时刻。按站点筛选落在该扫描窗口内的 CMA 分钟记录，每个窗口至少需要 8 个不同的合格分钟。
3. 对窗口内合格的 CMA GHI 取均值。四种卫星产品在同一个站点与扫描窗口共用这个参考值。
4. 对四种产品和地基观测均有效的配对计算 R²、RMSE、MAE 和 MBE。MBE 定义为卫星值减地基参考值。

FY-3C 原中心估计时间和异常标记随结果保留，用于追溯原配对。地基参考的时间范围取自已有 FY-4A 扫描窗口。15 分钟指扫描窗口，CMA 均值由窗口内可用的合格分钟计算。

## 文件

| 文件 | 用途 |
| --- | --- |
| [step_1_prepare.py](step_1_prepare.py) | `build` 建索引，`locate` 查覆盖，`extract` 查卫星值，`match` 运行地基配对与评价 |
| [step_2_satellite.py](step_2_satellite.py) | 读取卫星数据、正式预测和晴空 GHI，按索引提取像元 |
| [step_3_ground.py](step_3_ground.py) | CMA 质控、分钟聚合、共同配对和误差统计 |
| [config.example.json](config.example.json) | 本地路径和实验参数示例 |
| [templates/stations_template.csv](templates/stations_template.csv) | 站点编号、经纬度 |
| [templates/minute_ghi_template.csv](templates/minute_ghi_template.csv) | 分钟 GHI、时区和质控字段 |
| [requirements.txt](requirements.txt) | Python 依赖 |

## 准备输入

将示例配置复制为 `config.json`，并将占位路径换成自己的归档路径。需要正式样本库及其逐日元数据、清洗后划分、原始测试 CSV 与清洗行号映射、对齐的晴空 GHI、正式 DRCT 预测导出目录和权重，以及全量 Bicubic CSI 数组和配套的来源记录。正式 Bicubic 文件来自已归档的 OpenCV 5.0.0 插值结果，地基验证直接读取该数组。

在本目录运行以下命令。已具备校验通过的本地索引时，可以直接运行 `locate` 和 `extract`。

~~~bash
python -m pip install -r requirements.txt
python step_1_prepare.py build
python step_1_prepare.py locate --lon 105.64 --lat 18.8 --date 20191115
python step_1_prepare.py extract --sample-id 230019 --row 64 --col 64
~~~

`build` 对照正式数据顺序和来源校验值，生成 `output/test_patch_index.sqlite`、可查阅的 CSV、数据校验记录与异常标记。

## CMA 分钟数据与输出

- GHI 单位为 W/m²，时间字段须带时区。按实际 CMA 数据说明确认缺测值和合格质控标志。
- `instantaneous` 记录按观测时刻判断是否位于左闭右开的窗口。`one_minute_mean` 记录需要填写实际的一分钟起止区间，只有完整落在窗口内的区间参与聚合。例如标记为 04:15 的均值如果覆盖 [04:14, 04:15)，该分钟属于 [04:00, 04:15)。
- `--qc-ok` 填实际的合格标志，可重复使用以接受多个标志。输出路径应为一个尚未存在的目录。

~~~bash
python step_1_prepare.py match --stations input/stations.csv --ground input/minute_ghi.csv --qc-ok PASS --output-dir output/cma_evaluation
~~~

命令中的 `PASS` 是占位示例。`pairs.csv` 记录每次配对的 CMA 参考值和四种卫星 GHI。`metrics.csv` 记录四种产品在共同配对上的指标、站点数和配对数。

目前已用正式索引验证卫星端定位和取值。实际 CMA 文件接入后需要核对时间字段、单位和质控标志，再运行正式地基评价。

## 导出完整 Bicubic 基线

已有完整 Bicubic 数组时可直接使用归档文件。需要重新生成时，先完成 `build`，再在安装了 `opencv-python==5.0.0.93` 的环境运行下列命令。输出目录需要是新目录。

~~~bash
python export_bicubic_full.py --config config.json --output-dir /path/to/new_bicubic_output
~~~

程序以正式测试索引顺序保存 `bicubic_prediction_csi.npy` 和 `bicubic_prediction_manifest.json`。新建配置时可预先把 Bicubic 的两个路径填成计划导出的路径，建好索引后再生成数组。
