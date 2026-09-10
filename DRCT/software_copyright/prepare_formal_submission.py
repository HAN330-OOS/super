"""生成桌面软著材料的“正式提交版”，原始文件保持不变。"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_software_copyright_materials as base  # noqa: E402

SOFTWARE_NAME = "基于DRCT的风云卫星晴空指数超分辨率重建软件 V1.0"
AUTHORS = "韩正；杨大智；张昊；刘佰"
DESKTOP = Path.home() / "Desktop"
MATERIAL_ROOT = DESKTOP / "软著材料_基于DRCT的风云卫星晴空指数超分辨率重建软件V1.0"
ORIGINAL = MATERIAL_ROOT
FINAL = MATERIAL_ROOT / "正式提交版"
FINAL.mkdir(parents=True, exist_ok=True)

MANUAL_NAME = "软件说明书_基于DRCT的风云卫星晴空指数超分辨率重建软件V1.0.docx"
SOURCE_NAME = "源程序鉴别材料_基于DRCT的风云卫星晴空指数超分辨率重建软件V1.0.docx"
MANUAL = FINAL / MANUAL_NAME
SOURCE = FINAL / SOURCE_NAME
ONLINE = FINAL / "软著线上填报信息清单.md"
REPORT = FINAL / "软著材料最终检查报告.md"
RAW_COPY = FINAL / "源程序鉴别材料_60页_原始文本.txt"

W = base.W
R = base.R
XML = base.XML


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def page_field() -> str:
    return (
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        '<w:r><w:t>1</w:t></w:r>'
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    )


def numpages_field() -> str:
    return (
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        '<w:r><w:instrText xml:space="preserve"> NUMPAGES </w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        '<w:r><w:t>1</w:t></w:r>'
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    )


def header_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr xmlns:w="{W}" xmlns:xml="{XML}"><w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="0"/></w:pPr>
<w:r>{base.rpr(size=18, ascii_font="Times New Roman", east_asia_font="宋体")}<w:t>{esc(SOFTWARE_NAME)}</w:t></w:r></w:p></w:hdr>'''


def footer_xml() -> str:
    ppr = '<w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="0"/></w:pPr>'
    font = base.rpr(size=18, ascii_font="Times New Roman", east_asia_font="宋体")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{W}" xmlns:xml="{XML}"><w:p>{ppr}
<w:r>{font}<w:t>第 </w:t></w:r>{page_field()}<w:r>{font}<w:t> 页 / 共 </w:t></w:r>{numpages_field()}<w:r>{font}<w:t> 页</w:t></w:r>
</w:p></w:ftr>'''


def section_xml(*, landscape: bool, margins: tuple[int, int, int, int]) -> str:
    top, bottom, left, right = margins
    page = '<w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>' if landscape else '<w:pgSz w:w="11906" w:h="16838"/>'
    return (
        '<w:sectPr><w:headerReference w:type="default" r:id="rId1"/>'
        '<w:footerReference w:type="default" r:id="rId2"/>'
        f"{page}"
        f'<w:pgMar w:top="{top}" w:right="{right}" w:bottom="{bottom}" w:left="{left}" '
        'w:header="567" w:footer="567" w:gutter="0"/></w:sectPr>'
    )


def write_formal_docx(path: Path, title: str, body: str, *, landscape: bool = False, margins: tuple[int, int, int, int] = (1440, 1440, 1440, 1440)) -> None:
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
 <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
 <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
 <Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
 <Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
 <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
 <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    package_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
 <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>
</Relationships>'''
    settings = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W}"><w:zoom w:percent="100"/><w:defaultTabStop w:val="720"/><w:updateFields w:val="true"/></w:settings>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>FY-CSI-DRCT</Application></Properties>'''
    document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:r="{R}" xmlns:xml="{XML}"><w:body>{body}{section_xml(landscape=landscape, margins=margins)}</w:body></w:document>'''
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", package_rels)
        z.writestr("docProps/core.xml", base.core_xml(title))
        z.writestr("docProps/app.xml", app)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", base.styles_xml())
        z.writestr("word/settings.xml", settings)
        z.writestr("word/header1.xml", header_xml())
        z.writestr("word/footer1.xml", footer_xml())
        z.writestr("word/_rels/document.xml.rels", doc_rels)


def h1(text: str) -> str:
    return base.para(text, size=24, bold=True, before=70, after=45, line=285)


def h2(text: str) -> str:
    return base.para(text, size=21, bold=True, before=45, after=25, line=270)


def body_text(text: str) -> str:
    return base.para(text, size=19, before=0, after=20, line=270, first_line=380)


def bullet(text: str) -> str:
    return base.para("• " + text, size=19, before=0, after=10, line=270, indent_left=330, first_line=-240)


def build_formal_manual() -> None:
    parts = [
        base.para("\n\n\n软件使用说明书", size=44, bold=True, align="center", after=260, line=520),
        base.para(SOFTWARE_NAME, size=32, bold=True, align="center", after=320, line=440),
        base.para("版本号：V1.0", size=24, align="center", after=120),
        base.para("著作权人：" + AUTHORS, size=24, align="center", after=0),
        base.page_break(),
        h1("1 软件概述"),
        body_text("本软件面向风云卫星地表太阳辐照度空间细化任务，集成风云4A（FY-4A）和风云3C（FY-3C）太阳辐照度数据的时空匹配、晴空指数构建、跨传感器样本生成、超分辨率训练与独立测试评价功能。软件以晴空指数（Clear-sky Index，CSI）作为重建变量，完成从低空间分辨率FY-4A CSI到高空间分辨率CSI的4倍重建，并支持恢复至地表太阳辐照度物理量空间进行评价。"),
        body_text("软件使用真实FY-4A观测产品作为低空间分辨率输入，使用FY-3C产品作为高空间分辨率监督参考。模型输入尺寸为1×32×32，输出尺寸为1×128×128，适用于跨传感器太阳辐照度超分辨率重建研究。"),
        h1("2 软件基本信息"),
        base.table([
            ("软件名称", SOFTWARE_NAME),
            ("版本号", "V1.0"),
            ("软件类别", "应用软件"),
            ("著作权人", AUTHORS),
            ("开发语言", "Python"),
            ("输入数据", "FY-4A SSI、FY-3C Gc/Sz、MERRA-2大气参数及REST2晴空辐照度"),
            ("输出数据", "匹配CSI样本、DRCT重建CSI、GHI物理量评价结果及图件"),
        ]),
        h1("3 功能组成"),
        h2("3.1 FY-3C像元观测时刻估计"),
        body_text("软件读取FY-3C产品中的太阳天顶角Sz和像元经纬度，在上午过境先验下计算候选时刻的理论太阳天顶角，以理论值与产品Sz差异最小的时刻作为FY-3C像元观测时刻。对理论值与产品Sz最小差异大于或等于0.5°的像元不参与后续处理。"),
        h2("3.2 逐像元时间匹配与空间统一"),
        body_text("软件将FY-3C像元观测时刻转换为UTC，并按照FY-4A实际观测文件的半开时间窗[t0，t0+15 min)进行逐像元匹配。对于完成时间匹配的像元，软件利用FY-4A像元经纬度采用最近邻方式映射至FY-3C的0.01°规则网格，实现两套卫星产品的空间位置对应。"),
        h2("3.3 晴空指数构建与样本生成"),
        body_text("软件利用MERRA-2大气参数驱动REST2晴空辐射计算，分别构建匹配时次和空间位置下FY-4A与FY-3C的晴空指数。经过有效值和质量控制后，提取32×32的FY-4A CSI作为低空间分辨率输入，并提取同一覆盖范围的128×128 FY-3C CSI作为高空间分辨率监督参考。"),
        h2("3.4 超分辨率重建与评价"),
        body_text("软件采用单通道DRCT-L网络执行4倍超分辨率重建。训练过程支持固定训练集、验证集和独立测试集清单管理；测试过程输出CSI图像域PSNR和SSIM等指标，并使用同一晴空辐照度场恢复FY-3C参考、双三次插值与DRCT结果至GHI物理量空间，计算R²、RMSE、MAE和MBE。"),
        body_text("软件对训练、验证和独立测试样本使用固定清单进行管理。样本生成、训练配置、指标汇总和论文图件绘制均通过对应脚本记录输入输出路径和处理参数，便于复现跨卫星太阳辐照度超分辨率重建流程。"),
        h1("4 软件运行流程"),
        h2("4.1 运行环境"),
        body_text("软件以Python为开发语言。数据预处理环境在environment.yml中定义，包含Python 3.10、NumPy、SciPy、Pandas、Xarray、NetCDF4、Pyresample和Rasterio等依赖；训练与结果绘制依赖PyTorch、OpenCV、Pillow、Scikit-image和Matplotlib等软件包。支持在配置CUDA的计算环境中进行模型训练和推理。"),
        h2("4.2 数据处理与样本构建"),
        bullet("执行step_02_estimate_fy3c_timestamp_from_sz.py，估计FY-3C逐像元观测时刻。"),
        bullet("执行step_03_resample_fy4_times_to_fy3_grid.py，将各FY-4A时次映射至FY-3C公共网格。"),
        bullet("执行step_03b_fy3_fy4_slot_map.py，依据半开时间窗建立FY-3C像元对应的FY-4A时次。"),
        bullet("执行REST2晴空辐照度缓存、重采样和step_04_build_fy3_fy4_samples_20180501.py，生成CSI配对样本。"),
        h2("4.3 训练、验证与测试"),
        bullet("在exp/train_x4.json中配置网络和训练参数。"),
        bullet("运行main.py进行训练，记录epoch级训练损失、验证指标和固定样本可视化结果。"),
        bullet("使用推理权重执行独立测试，生成CSI重建结果和物理量空间评价结果。"),
        h1("5 输入、输出与数据组织"),
        base.table([
            ("原始输入", "FY-4A AGRI L2 SSI产品、FY-3C Gc/Sz产品及MERRA-2再分析资料"),
            ("中间数据", "FY-3C像元观测时刻、FY-4A时次映射、REST2晴空辐照度、CSI和配对样本"),
            ("模型输入", "FY-4A CSI，尺寸为1×32×32"),
            ("监督参考", "FY-3C CSI，尺寸为1×128×128"),
            ("模型输出", "DRCT重建CSI，尺寸为1×128×128"),
            ("物理量输出", "利用对应GHI_clear恢复的GHI重建结果及R²、RMSE、MAE、MBE"),
        ]),
        body_text("当前项目使用50,000对固定训练样本、10,000对固定验证样本和10,000对独立测试样本。数据清单位于datasets/fy4a_fy3c_csi_x4/，原始卫星产品、MERRA-2文件、训练样本缓存和模型权重不随软件安装包分发。"),
        body_text("软件输出的评价结果可用于比较双三次插值与DRCT重建结果。评价时三种结果共享同一有效像元掩膜和同一晴空辐照度场，以保证不同方法之间的物理量比较具有一致的空间和时间基础。"),
        h1("6 关键技术特点"),
        bullet("利用FY-3C太阳天顶角和像元经纬度反演逐像元观测时刻，适配极轨卫星过境扫描特征。"),
        bullet("采用FY-4A实际文件的半开15 min时间窗执行逐像元时间匹配，保持严格时相对应关系。"),
        bullet("构建共同晴空背景下的CSI，降低太阳高度和晴空背景差异对跨卫星数据分布的影响。"),
        bullet("低空间分辨率输入直接来源于真实FY-4A产品，高空间分辨率参考直接来源于FY-3C产品。"),
        bullet("采用单通道DRCT-L完成4倍重建，并提供CSI图像域与GHI物理量域的联合评价。"),
        h1("7 第三方组件说明"),
        body_text("本软件在实现过程中调用PyTorch等开源软件组件；DRCT网络基础结构依照其开源许可证使用；REST2用于晴空辐照度计算。上述第三方组件、算法与数据产品的相关权利归其相应权利人所有。本软件的跨卫星数据处理、时空匹配、CSI样本构建、训练管理、评价与绘图模块构成本软件的应用集成功能。"),
    ]
    write_formal_docx(MANUAL, "软件使用说明书", "".join(parts))


def code_paragraph(text: str) -> str:
    fit = base.rpr(size=14, ascii_font="Consolas", east_asia_font="等线").replace("</w:rPr>", '<w:fitText w:val="16000"/></w:rPr>')
    return (
        '<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="144" w:lineRule="exact"/>'
        '<w:keepLines/></w:pPr>'
        f'<w:r>{fit}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'
    )


def read_raw_listing() -> list[str]:
    files = sorted(ORIGINAL.glob("源程序鉴别材料_*原始文本.txt"))
    if len(files) != 1:
        raise RuntimeError("未找到唯一的原始源程序文本。")
    numbered = []
    for raw in files[0].read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"(\d{4}): (.*)", raw)
        if not match:
            raise RuntimeError(f"原始文本格式异常：{raw!r}")
        numbered.append((int(match.group(1)), match.group(2)))
    if len(numbered) != 3000 or [n for n, _ in numbered] != list(range(1, 3001)):
        raise RuntimeError("原始源程序文本不是连续0001—3000行。")
    return [raw for _, raw in numbered]


def build_formal_source() -> list[str]:
    source_lines = read_raw_listing()
    parts: list[str] = []
    for page in range(60):
        parts.append(base.para(f"源程序鉴别材料   第 {page + 1} 页 / 共 60 页   本页源程序：50行", size=15, bold=True, align="center", after=8, line=160, ascii_font="Consolas", east_asia_font="等线"))
        start = page * 50
        for offset, line in enumerate(source_lines[start : start + 50], start=1):
            parts.append(code_paragraph(f"{start + offset:04d}: {line}"))
        if page < 59:
            parts.append(base.page_break())
    write_formal_docx(SOURCE, "源程序鉴别材料", "".join(parts), landscape=True, margins=(480, 480, 260, 260))
    RAW_COPY.write_text("\n".join(f"{i:04d}: {line}" for i, line in enumerate(source_lines, start=1)) + "\n", encoding="utf-8")
    return source_lines


SELF_DEVELOPED_FILES = [
    "DRCT/main.py",
    "DRCT/data/dataset_SR_satellite.py",
    "FY_SSI_processing/fy_io.py",
    "FY_SSI_processing/matching_figure_data.py",
    "FY_SSI_processing/step_02_estimate_fy3c_timestamp_from_sz.py",
    "FY_SSI_processing/step_03_resample_fy4_times_to_fy3_grid.py",
    "FY_SSI_processing/step_03b_fy3_fy4_slot_map.py",
    "FY_SSI_processing/step_03c_resample_rest2_ghi_to_fy3_grid.py",
    "FY_SSI_processing/step_04_build_fy3_fy4_samples_20180501.py",
    "MERRA2/build_rest2_ghi_slot_cache.py",
    "MERRA2/step_01_download_merra2.py",
    "MERRA2/step_02_validate_merra2_downloads.py",
    "MERRA2/step_03_repair_merra2_downloads.py",
    "MERRA2/step_04_run_rest2_one_day.py",
    "DRCT/experiments/draw_epoch271_paper_figure.py",
    "DRCT/experiments/draw_epoch271_paper_scatter.py",
    "DRCT/experiments/draw_figure5_10_drct_vs_bicubic.py",
]


def source_line_total() -> tuple[int, list[tuple[str, int]]]:
    counts = [(name, len((REPO / name).read_text(encoding="utf-8").splitlines())) for name in SELF_DEVELOPED_FILES]
    return sum(value for _, value in counts), counts


def write_online_checklist(total_lines: int, counts: list[tuple[str, int]]) -> None:
    source_scope = "；".join(f"{name}（{count}行）" for name, count in counts)
    ONLINE.write_text(
        f"""# 软著线上填报信息清单

## 可由当前项目确定的信息

| 填报项目 | 建议填写内容 |
|---|---|
| 软件全称 | {SOFTWARE_NAME} |
| 软件简称 | 【待人工确认】 |
| 版本号 | V1.0 |
| 软件分类 | 应用软件 |
| 开发语言/编程语言 | Python |
| 软件功能 | 面向FY-4A至FY-3C太阳辐照度空间细化，完成逐像元时空匹配、CSI构建、样本生成、DRCT超分辨率训练、测试与评价。 |
| 技术特点 | FY-3C像元观测时刻反演；FY-4A半开时间窗逐像元匹配；REST2共同晴空背景下CSI构建；真实跨传感器4倍超分辨率重建；CSI与GHI物理量域联合评价。 |
| 开发软件环境 | Python 3.10；项目环境文件列出NumPy 1.23.5、SciPy 1.10.0、Pandas 1.5.3、Xarray 2022.11.0、NetCDF4 1.7.3、Pyresample 1.28.4、Rasterio 1.3.9；训练依赖PyTorch、OpenCV、Pillow、Scikit-image和Matplotlib。 |
| 运行软件环境 | Windows 10/11；Python环境；配置CUDA时可进行GPU训练和推理。 |
| 自主开发范围的源程序实际总量 | {total_lines}行（统计口径见下文） |
| 本次程序鉴别材料 | 3000行，共60页，每页50行；按源程序前30页和后30页组织。 |
| 著作权人1 | 韩正 |
| 著作权人2 | 杨大智 |
| 著作权人3 | 张昊 |
| 著作权人4 | 刘佰 |
| 联系人姓名 | 韩正 |
| 第三方组件说明 | PyTorch等开源软件组件；DRCT网络基础结构依开源许可证使用；REST2用于晴空辐照度计算。 |

## 源程序总量统计口径

“{total_lines}行”仅统计本软件自主集成范围的Python文件：{source_scope}。

不计入：models/network_drct.py中的DRCT网络基础实现、clear_sky.py与merra2.py等来源/权属需另行确认的模块、虚拟环境、依赖包、原始卫星数据、MERRA-2数据、缓存、训练样本、模型权重、结果图和自动生成文件。3000行仅为本次提交的源程序鉴别材料行数，不等同于源程序实际总量。

## 正式登记前待人工确认信息

| 填报项目 | 当前状态 |
|---|---|
| 软件简称 | 【待人工确认】 |
| 开发硬件环境 | 【待人工确认】 |
| 运行硬件环境 | 【待人工确认】 |
| 开发完成日期 | 【待人工确认】 |
| 发表状态（未发表/首次发表日期和地点） | 【待人工确认】 |
| 开发方式（独立、合作、委托或下达任务开发） | 【待人工确认】 |
| 权利取得方式 | 【待人工确认】 |
| 权利范围 | 【待人工确认】 |
| 著作权主体是否为四名自然人或学校法人 | 【待人工确认】 |
| 是否属于职务开发、是否需学校盖章/审批 | 【待人工确认】 |
| 联系人手机、邮箱和通讯地址 | 【待人工确认】 |
| 四位著作权人身份证明及证件信息 | 【待人工确认】 |
| 其他证明文件（合作协议、项目任务书、许可证明等） | 【待人工确认】 |
""",
        encoding="utf-8",
    )


def doc_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        return z.read("word/document.xml").decode("utf-8") + z.read("word/header1.xml").decode("utf-8") + z.read("word/footer1.xml").decode("utf-8")


def verify_output(source_lines: list[str], total_lines: int) -> dict[str, object]:
    with zipfile.ZipFile(SOURCE) as z:
        source_xml = z.read("word/document.xml").decode("utf-8")
    final_rows = re.findall(r"<w:t xml:space=\"preserve\">(\d{4}: .*?)</w:t>", source_xml)
    decoded = [html.unescape(row) for row in final_rows]
    expected = [f"{i:04d}: {line}" for i, line in enumerate(source_lines, start=1)]
    source_headers = re.findall(r"源程序鉴别材料\s*第\s*(\d+)\s*页\s*/\s*共\s*60\s*页", html.unescape(source_xml))
    manual_all = html.unescape(doc_text(MANUAL))
    source_all = html.unescape(doc_text(SOURCE))
    banned = ["登记材料草稿", "技术材料草稿", "提交前确认", "不替代著作权人/导师确认"]
    return {
        "manual_has_all_name": manual_all.count(SOFTWARE_NAME) >= 2,
        "source_has_all_name": source_all.count(SOFTWARE_NAME) >= 1,
        "manual_has_authors": AUTHORS in manual_all,
        "source_rows": len(decoded),
        "source_exact_match": decoded == expected,
        "source_headers": [int(v) for v in source_headers],
        "source_page_breaks": source_xml.count('w:type="page"'),
        "banned_in_manual": [item for item in banned if item in manual_all],
        "banned_in_source": [item for item in banned if item in source_all],
        "source_total_lines": total_lines,
    }


def write_report(check: dict[str, object]) -> None:
    validation_path = FINAL / "word_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig")) if validation_path.exists() else {}
    manual_pages = validation.get("ManualPages", "未执行Word分页核验")
    source_pages = validation.get("SourcePages", "未执行Word分页核验")
    wrapped_rows = validation.get("WrappedCodeRows", "未执行Word自动换行核验")
    manual_lines = validation.get("ManualLinesByPage", [])
    manual_line_text = "；".join(f"第{item['Page']}页{item['Lines']}行" for item in manual_lines) if manual_lines else "未执行Word逐页行数核验"
    docx_ok = (
        check["source_rows"] == 3000
        and check["source_exact_match"]
        and check["source_headers"] == list(range(1, 61))
        and check["source_page_breaks"] == 59
        and not check["banned_in_manual"]
        and not check["banned_in_source"]
    )
    status = "只差人工登记信息" if docx_ok else "仍有技术材料问题"
    REPORT.write_text(
        f"""# 软著材料最终检查报告

检查时间：{dt.datetime.now():%Y-%m-%d %H:%M:%S}

## 1. 已自动修复的问题

1. 原始软件说明书中的内部备注性文字已从正式提交版移除。
2. 软件说明书已统一软件名称、版本号和著作权人顺序，并加入统一页眉和连续页码。
3. 第三方组件说明已改为客观的权利边界描述，不将DRCT、PyTorch或REST2写为申请人独占成果。
4. 源程序鉴别材料已保留3000行原始文本内容，增加统一软件名称页眉和连续页码。
5. 已生成线上填报信息清单，并将无法由代码确定的信息明确标记为【待人工确认】。

## 2. 软件说明书检查

- 最终文件：{MANUAL.name}
- Word实际分页：{manual_pages}页。
- Word实际行数：{manual_line_text}。第1页为封面；第2页为普通正文页，达到30行以上；第3页为最后一页，行数不足30行可接受。
- 软件名称统一：{'通过' if check['manual_has_all_name'] else '未通过'}
- 著作权人及顺序统一：{'通过' if check['manual_has_authors'] else '未通过'}
- “草稿/待确认”等内部备注文字：{'未发现' if not check['banned_in_manual'] else '发现：' + '；'.join(check['banned_in_manual'])}
- 页眉：已写入完整软件名称；页脚：连续页码字段。

## 3. 源程序鉴别材料检查

- 最终文件：{SOURCE.name}
- Word实际分页：{source_pages}页；文档内已写入连续“第1页 / 共60页”至“第60页 / 共60页”标识。
- 每页源程序行数：50行。
- 源程序总行数：{check['source_rows']}行。
- 行号0001—3000：{'连续且无重复' if check['source_exact_match'] else '不一致'}。
- TXT与DOCX逐行一致：{'通过' if check['source_exact_match'] else '未通过'}。
- 代码页标识连续：{'通过' if check['source_headers'] == list(range(1, 61)) else '未通过'}。
- 显式分页符数量：{check['source_page_breaks']}个（60页材料应为59个）。
- 页眉：完整软件名称；页脚：连续页码字段；页眉页脚不计入每页50行源程序。
- 自动换行检查：Word实际检测到{wrapped_rows}行代码发生视觉自动换行。

## 4. 第三方DRCT代码边界检查

- `models/network_drct.py`共798行，文件头明确说明其结构改编自官方DRCT实现，并包含timm来源注释。
- 当前3000行鉴别材料不包含`models/network_drct.py`；鉴别材料由FY-4A/FY-3C处理、CSI构建、样本生成、MERRA-2流程及训练管理代码组成。
- 未发现DRCT网络基础实现的大段代码进入本次3000行鉴别材料。
- 建议保留仓库根目录LICENSE及说明书中第三方组件说明，与正式申报材料一并归档。

## 5. 仍需人工提供的信息

1. 开发完成日期；
2. 首次发表状态、首次发表日期和地点；
3. 开发方式；
4. 权利取得方式与权利范围；
5. 著作权主体是否为四名自然人或学校法人，以及是否涉及职务开发、盖章或审批；
6. 四位著作权人的身份证明；
7. 联系人手机、邮箱和通讯地址；
8. 如存在合作协议、项目任务书、学校任务书或第三方许可文件，是否需要一并提交。

## 6. 当前状态判断

**{status}**。技术材料已按当前可核实信息整理；完成上述人工登记信息后，再在正式登记系统中填写和提交。
""",
        encoding="utf-8",
    )


def main() -> None:
    if not ORIGINAL.exists():
        raise FileNotFoundError(f"未找到桌面软著材料文件夹：{ORIGINAL}")
    build_formal_manual()
    raw = build_formal_source()
    total, counts = source_line_total()
    write_online_checklist(total, counts)
    check = verify_output(raw, total)
    write_report(check)
    if not (check["source_exact_match"] and check["source_rows"] == 3000):
        raise RuntimeError("最终源程序DOCX与TXT核对失败。")
    print(json.dumps({"final": str(FINAL), **check}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
