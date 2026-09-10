"""根据当前 FY-CSI-DRCT 项目生成软件著作权技术材料草稿。

输出：软件说明书 DOCX、60 页源程序鉴别材料 DOCX、登记信息核对单。
不从代码推断开发完成日期、首次发表日期、单位或权属，这些字段统一保留给著作权人确认。
"""

from __future__ import annotations

import datetime as dt
import html
import zipfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

SOFTWARE_NAME = "基于DRCT的风云卫星晴空指数超分辨率重建软件 V1.0"
AUTHORS = "韩正；杨大智；张昊；刘佰"
MANUAL = OUT / "软件说明书_基于DRCT的风云卫星晴空指数超分辨率重建软件V1.0.docx"
SOURCE = OUT / "源程序鉴别材料_基于DRCT的风云卫星晴空指数超分辨率重建软件V1.0.docx"
CHECKLIST = OUT / "软著登记信息核对单.md"
README = HERE / "README.md"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML = "http://www.w3.org/XML/1998/namespace"


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def rpr(
    *,
    size: int = 21,
    bold: bool = False,
    ascii_font: str = "Times New Roman",
    east_asia_font: str = "宋体",
    color: str | None = None,
) -> str:
    bold_tag = "<w:b/>" if bold else ""
    color_tag = f'<w:color w:val="{color}"/>' if color else ""
    return (
        "<w:rPr>"
        f'<w:rFonts w:ascii="{ascii_font}" w:hAnsi="{ascii_font}" '
        f'w:eastAsia="{east_asia_font}" w:cs="{ascii_font}"/>'
        f"<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>"
        f"{bold_tag}{color_tag}</w:rPr>"
    )


def para(
    text: str,
    *,
    size: int = 21,
    bold: bool = False,
    align: str = "left",
    before: int = 0,
    after: int = 80,
    line: int = 360,
    line_rule: str = "auto",
    page_break_before: bool = False,
    ascii_font: str = "Times New Roman",
    east_asia_font: str = "宋体",
    color: str | None = None,
    indent_left: int = 0,
    first_line: int = 0,
) -> str:
    pbreak = "<w:pageBreakBefore/>" if page_break_before else ""
    indent = (
        f'<w:ind w:left="{indent_left}" w:firstLine="{first_line}"/>'
        if indent_left or first_line
        else ""
    )
    ppr = (
        "<w:pPr>"
        f'<w:jc w:val="{align}"/>'
        f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="{line_rule}"/>'
        f"{indent}{pbreak}</w:pPr>"
    )
    pieces = text.split("\n")
    content = ""
    for i, piece in enumerate(pieces):
        if i:
            content += "<w:br/>"
        preserve = ' xml:space="preserve"' if piece.startswith(" ") or piece.endswith(" ") else ""
        content += f"<w:t{preserve}>{esc(piece)}</w:t>"
    return f"<w:p>{ppr}<w:r>{rpr(size=size, bold=bold, ascii_font=ascii_font, east_asia_font=east_asia_font, color=color)}{content}</w:r></w:p>"


def page_break() -> str:
    return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"


def sect_pr(*, landscape: bool = False, top: int = 1440, bottom: int = 1440, left: int = 1440, right: int = 1440) -> str:
    if landscape:
        page = '<w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>'
    else:
        page = '<w:pgSz w:w="11906" w:h="16838"/>'
    return (
        "<w:sectPr>"
        f"{page}"
        f'<w:pgMar w:top="{top}" w:right="{right}" w:bottom="{bottom}" w:left="{left}" '
        'w:header="567" w:footer="567" w:gutter="0"/>'
        '</w:sectPr>'
    )


def cell(text: str, *, bold: bool = False, width: int = 0) -> str:
    width_tag = f'<w:tcW w:w="{width}" w:type="dxa"/>' if width else ""
    return (
        "<w:tc><w:tcPr>"
        f"{width_tag}"
        '<w:tcMar><w:top w:w="80" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>'
        "</w:tcPr>"
        + para(text, size=20, bold=bold, after=0, line=300)
        + "</w:tc>"
    )


def table(rows: Iterable[tuple[str, str]]) -> str:
    rows_xml = "".join(
        "<w:tr>" + cell(label, bold=True, width=2400) + cell(value, width=6960) + "</w:tr>"
        for label, value in rows
    )
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="808080"/>'
        '<w:left w:val="single" w:sz="4" w:color="808080"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="808080"/>'
        '<w:right w:val="single" w:sz="4" w:color="808080"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="BFBFBF"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="BFBFBF"/></w:tblBorders>'
        '<w:tblLayout w:type="fixed"/></w:tblPr>'
        '<w:tblGrid><w:gridCol w:w="2400"/><w:gridCol w:w="6960"/></w:tblGrid>'
        f"{rows_xml}</w:tbl>"
    )


def styles_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
  <w:docDefaults><w:rPrDefault><w:rPr>{rpr()}</w:rPr></w:rPrDefault>
  <w:pPrDefault><w:pPr><w:spacing w:after="80" w:line="360" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
</w:styles>'''


def document_xml(body: str, *, landscape: bool = False, margins: tuple[int, int, int, int] = (1440, 1440, 1440, 1440)) -> str:
    top, bottom, left, right = margins
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:r="{R}" xmlns:xml="{XML}">
  <w:body>{body}{sect_pr(landscape=landscape, top=top, bottom=bottom, left=left, right=right)}</w:body>
</w:document>'''


def core_xml(title: str) -> str:
    timestamp = dt.datetime.now().replace(microsecond=0).isoformat() + "Z"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <dc:title>{esc(title)}</dc:title><dc:creator>{esc(AUTHORS)}</dc:creator><cp:lastModifiedBy>{esc(AUTHORS)}</cp:lastModifiedBy>
 <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
 <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>'''


def write_docx(path: Path, title: str, body: str, *, landscape: bool = False, margins: tuple[int, int, int, int] = (1440, 1440, 1440, 1440)) -> None:
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
 <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
 <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
 <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
 <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    package_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
 <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    doc_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"></Relationships>'''
    settings = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W}"><w:zoom w:percent="100"/><w:defaultTabStop w:val="720"/></w:settings>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>FY-CSI-DRCT</Application></Properties>'''
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", package_rels)
        docx.writestr("docProps/core.xml", core_xml(title))
        docx.writestr("docProps/app.xml", app)
        docx.writestr("word/document.xml", document_xml(body, landscape=landscape, margins=margins))
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("word/settings.xml", settings)
        docx.writestr("word/_rels/document.xml.rels", doc_rels)


def build_manual() -> None:
    body: list[str] = []
    body += [
        para("\n\n\n软件使用说明书", size=44, bold=True, align="center", after=260, line=520),
        para(SOFTWARE_NAME, size=32, bold=True, align="center", after=360, line=440),
        para("版本：V1.0", size=24, align="center", after=120),
        para(f"著作权人：{AUTHORS}", size=24, align="center", after=300),
        para("材料状态：登记材料草稿。开发完成日期、首次发表日期、单位及权属信息须在提交前确认。", size=21, align="center", after=0),
        page_break(),
        para("1 软件概述", size=28, bold=True, before=120, after=120),
        para("本软件面向风云卫星太阳辐照度空间细化任务，完成风云4A（FY-4A）低空间分辨率地表太阳辐照度与风云3C（FY-3C）高空间分辨率太阳辐照度数据的时空匹配、晴空指数构建、跨传感器样本生成、DRCT超分辨率训练及独立测试评价。软件以晴空指数（Clear-sky Index，CSI）作为模型学习变量，输出4倍空间分辨率重建结果，并可恢复至地表太阳辐照度物理量空间进行评价。", after=120, first_line=420),
        para("软件的核心目标是将真实FY-4A低分辨率CSI（1×32×32）重建为对应地理覆盖范围内的高分辨率CSI（1×128×128），并以FY-3C CSI作为监督参考。", after=120, first_line=420),
        para("2 软件基本信息", size=28, bold=True, before=160, after=120),
        table([
            ("软件名称", SOFTWARE_NAME),
            ("版本号", "V1.0"),
            ("著作权人", AUTHORS),
            ("开发语言", "Python"),
            ("运行平台", "Windows 10/11；支持CUDA的GPU计算环境"),
            ("主要依赖", "PyTorch、NumPy、SciPy、Rasterio、Matplotlib、REST2相关计算模块"),
            ("输入数据", "FY-4A SSI、FY-3C Gc/Sz、MERRA-2大气参数及REST2晴空辐照度"),
            ("输出数据", "匹配CSI样本、DRCT重建CSI、GHI物理量评价结果和论文图件"),
        ]),
        para("3 功能组成", size=28, bold=True, before=180, after=120),
        para("3.1 FY-3C像元观测时刻估计", size=24, bold=True, before=100, after=70),
        para("读取FY-3C产品中的太阳天顶角Sz和像元经纬度，在上午过境先验下计算候选时刻的理论太阳天顶角，选取与产品Sz差异最小的时刻作为每个FY-3C像元的观测时刻。软件仅保留理论值与产品Sz最小差异小于0.5°的像元，以控制时刻估计质量。", first_line=420),
        para("3.2 跨卫星逐像元时间匹配与空间统一", size=24, bold=True, before=100, after=70),
        para("将FY-3C估计时刻转换为UTC，并采用FY-4A实际观测文件的半开时间窗[t0, t0+15 min)执行逐像元匹配。随后依据FY-4A像元经纬度，采用最近邻方式映射至FY-3C的0.01°规则网格。该过程仅统一空间位置，不人为增加FY-4A原始空间细节。", first_line=420),
        para("3.3 晴空指数构建与样本生成", size=24, bold=True, before=100, after=70),
        para("软件调用MERRA-2大气参数驱动REST2晴空辐射计算，使用CSI=G/Gclear分别构建匹配后的FY-4A与FY-3C晴空指数。通过质量控制后，以32×32的真实FY-4A CSI作为低分辨率输入，以128×128的FY-3C CSI作为高分辨率监督参考，构建4倍超分辨率样本。", first_line=420),
        para("3.4 DRCT重建、验证与独立测试", size=24, bold=True, before=100, after=70),
        para("软件使用单通道DRCT-L网络完成CSI重建。训练入口统一由main.py管理，支持固定训练集、验证集和独立测试集清单。测试阶段同时输出CSI图像域PSNR、SSIM等指标，并使用同一晴空辐照度场将FY-3C、双三次插值和DRCT结果恢复到GHI物理量空间，计算R²、RMSE、MAE和MBE。", first_line=420),
        para("4 软件运行流程", size=28, bold=True, before=180, after=120),
        para("4.1 环境准备", size=24, bold=True, before=100, after=70),
        para("在Python环境中安装项目依赖。数据预处理建议按environment.yml创建环境；训练与结果绘制需要安装requirements.txt中列出的依赖，并在可用时配置CUDA。", first_line=420),
        para("示例命令：conda env create -f environment.yml；conda activate super-resolution；pip install -r requirements.txt。", size=18, ascii_font="Consolas", east_asia_font="等线", before=40, after=70),
        para("4.2 数据预处理和样本构建", size=24, bold=True, before=100, after=70),
        para("• 执行FY_SSI_processing/step_02_estimate_fy3c_timestamp_from_sz.py，估计FY-3C逐像元观测时刻。", indent_left=360, first_line=-260, after=40),
        para("• 执行FY_SSI_processing/step_03_resample_fy4_times_to_fy3_grid.py，将各FY-4A时次映射到FY-3C公共网格。", indent_left=360, first_line=-260, after=40),
        para("• 执行FY_SSI_processing/step_03b_fy3_fy4_slot_map.py，依据半开时间窗建立逐像元FY-4A时次匹配结果。", indent_left=360, first_line=-260, after=40),
        para("• 执行REST2晴空辐照度缓存和重采样脚本，再执行step_04_build_fy3_fy4_samples_20180501.py生成CSI样本。", indent_left=360, first_line=-260, after=80),
        para("4.3 训练、验证和测试", size=24, bold=True, before=100, after=70),
        para("• 在exp/train_x4.json中配置网络和训练参数。", indent_left=360, first_line=-260, after=40),
        para("• 运行python main.py进行训练，程序记录epoch级训练损失、验证指标、固定样本可视化及检查点。", indent_left=360, first_line=-260, after=40),
        para("• 使用训练完成的推理权重执行独立测试，生成物理量域评价结果和论文散点图、典型样本图。", indent_left=360, first_line=-260, after=100),
        para("5 输入、输出与结果说明", size=28, bold=True, before=180, after=120),
        table([
            ("原始输入", "FY-4A AGRI L2 SSI产品、FY-3C Gc/Sz产品、MERRA-2再分析资料"),
            ("中间数据", "FY-3C像元观测时刻、FY-4A时次映射结果、REST2晴空辐照度、CSI及配对样本"),
            ("模型输入", "FY-4A CSI，尺寸为1×32×32"),
            ("监督参考", "FY-3C CSI，尺寸为1×128×128"),
            ("模型输出", "DRCT重建CSI，尺寸为1×128×128"),
            ("物理量输出", "利用对应GHI_clear恢复的GHI重建结果及R²、RMSE、MAE、MBE"),
        ]),
        para("软件当前实验使用50,000对固定训练样本、10,000对固定验证样本与10,000对独立测试样本。数据清单位于datasets/fy4a_fy3c_csi_x4/，原始卫星产品、MERRA-2文件、训练patch和模型权重因体积或数据授权限制不随代码库发布。", before=80, first_line=420),
        para("6 关键技术特点", size=28, bold=True, before=180, after=120),
        para("• 针对极轨FY-3C产品无逐像元观测时刻的问题，利用Sz和经纬度实施像元级观测时间反演。", indent_left=360, first_line=-260, after=40),
        para("• 采用实际FY-4A文件的半开15 min时间窗匹配规则，避免以最近时次替代严格时相对应。", indent_left=360, first_line=-260, after=40),
        para("• 以REST2晴空辐照度构造两颗卫星共同晴空背景下的CSI，降低太阳高度和晴空背景差异影响。", indent_left=360, first_line=-260, after=40),
        para("• 低分辨率输入来自真实FY-4A产品，高分辨率参考来自FY-3C产品，不以FY-3C人工降采样替代真实跨传感器输入。", indent_left=360, first_line=-260, after=40),
        para("• 采用DRCT-L完成单通道4倍超分辨率重建，并同时给出CSI图像域与GHI物理量域的评价结果。", indent_left=360, first_line=-260, after=100),
        para("7 第三方组件与权利边界说明", size=28, bold=True, before=180, after=120),
        para("本软件整合使用开源深度学习框架及公开算法实现。DRCT网络基础实现保留原项目MIT License；REST2晴空辐射模型及相关科学计算模块用于完成晴空背景计算。软著申请时应将自主完成的跨卫星数据处理、时空匹配、CSI样本构建、训练管理与评价绘图程序作为登记范围，不得将第三方原始代码、模型名称或数据产品的权利主张为申请人独占权利。", first_line=420),
        para("本说明书为基于当前项目代码形成的技术材料草稿，不替代著作权人、导师、学校科技管理部门或代理机构对著作权归属、开发完成日期、首次发表状态和许可合规性的确认。", bold=True, before=100, first_line=420),
    ]
    write_docx(MANUAL, "软件使用说明书", "".join(body))


def build_source_material() -> None:
    source_files = [
        "FY_SSI_processing/fy_io.py",
        "FY_SSI_processing/matching_figure_data.py",
        "FY_SSI_processing/step_02_estimate_fy3c_timestamp_from_sz.py",
        "FY_SSI_processing/step_03_resample_fy4_times_to_fy3_grid.py",
        "FY_SSI_processing/step_03b_fy3_fy4_slot_map.py",
        "FY_SSI_processing/step_03c_resample_rest2_ghi_to_fy3_grid.py",
        "FY_SSI_processing/step_04_build_fy3_fy4_samples_20180501.py",
        "MERRA2/build_rest2_ghi_slot_cache.py",
        "DRCT/main.py",
    ]
    all_lines: list[str] = []
    for relative in source_files:
        all_lines.append(f"# ===== FILE: {relative} =====")
        all_lines.extend((ROOT / relative).read_text(encoding="utf-8").splitlines())
    if len(all_lines) < 3000:
        raise RuntimeError("核心源程序行数不足，无法生成60页鉴别材料。")
    selected = all_lines[:1500] + all_lines[-1500:]

    pages: list[str] = []
    for page_index in range(60):
        start = page_index * 50
        page_lines = selected[start : start + 50]
        pages.append(para(f"源程序鉴别材料   第 {page_index + 1} 页 / 共 60 页", size=15, bold=True, align="center", after=10, line=160, ascii_font="Consolas", east_asia_font="等线"))
        for local_index, code_line in enumerate(page_lines, start=1):
            displayed = f"{start + local_index:04d}: {code_line}"
            pages.append(para(displayed, size=12, after=0, line=108, line_rule="exact", ascii_font="Consolas", east_asia_font="等线"))
        if page_index != 59:
            pages.append(page_break())

    raw_listing = "\n".join(f"{idx + 1:04d}: {line}" for idx, line in enumerate(selected)) + "\n"
    (OUT / "源程序鉴别材料_60页_原始文本.txt").write_text(raw_listing, encoding="utf-8")
    write_docx(SOURCE, "源程序鉴别材料", "".join(pages), landscape=True, margins=(520, 520, 300, 300))


def write_auxiliary_files() -> None:
    CHECKLIST.write_text(
        f"""# 软件著作权登记信息核对单

## 已依据本次确认填写

| 项目 | 内容 |
|---|---|
| 软件名称 | {SOFTWARE_NAME} |
| 版本号 | V1.0 |
| 著作权人排序 | {AUTHORS} |
| 软件类别 | 应用软件 |
| 开发语言 | Python |
| 申请材料范围 | 软件说明书、源程序鉴别材料、登记信息核对单 |

## 提交前必须由著作权人/导师确认（不可由代码推断）

1. 著作权人是否为个人共同著作权，或应由哈尔滨工业大学等法人主体享有；如涉及职务开发，应以学校科技处或导师意见为准。
2. 每位著作权人的身份证件信息、联系方式、地址及签字/盖章方式。
3. 开发完成日期。
4. 首次发表状态：未发表，或首次发表的日期与地点。
5. 权利取得方式：通常为原始取得，但仍需确认。
6. 软件简称（如需）：FY-CSI-DRCT。
7. 申请人/联系人、代理机构（如有）以及提交渠道。

## 代码与许可核对

- 申请范围应聚焦自主完成的FY-4A/FY-3C时空匹配、CSI构建、样本生成、训练管理、评价与绘图软件集成。
- DRCT网络基础实现保留原项目MIT License；REST2与相关依赖属于第三方组件，不应作为申请人独占权利主张。
- FY-4A、FY-3C和MERRA-2原始数据不随软件著作权材料主张权利，使用前应遵守各自的数据使用条款。

生成时间：{dt.datetime.now():%Y-%m-%d %H:%M:%S}
""",
        encoding="utf-8",
    )
    README.write_text(
        f"""# 软件著作权材料草稿

本目录根据FY-CSI-DRCT当前代码生成软著登记材料初稿，软件名称为“{SOFTWARE_NAME}”。

## 输出文件

- `output/软件说明书_*.docx`：软件功能、运行流程、输入输出和许可边界说明。
- `output/源程序鉴别材料_*.docx`：基于核心自主集成代码生成的60页源程序鉴别材料，每页50行。
- `output/源程序鉴别材料_60页_原始文本.txt`：与DOCX对应的原始文本核对副本。
- `output/软著登记信息核对单.md`：提交前必须人工确认的权属、日期、发表状态等信息。

## 生成方式

```powershell
& D:\\ana\\envs\\super-resolution\\python.exe .\\build_software_copyright_materials.py
```

## 重要说明

本目录仅生成技术材料草稿，不替代著作权人、导师、学校科技管理部门或代理机构对权属及提交信息的确认。
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_auxiliary_files()
    build_manual()
    build_source_material()
    print(f"已生成：\n{MANUAL}\n{SOURCE}\n{CHECKLIST}")
