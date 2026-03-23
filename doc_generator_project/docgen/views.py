import io
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches, Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy


def set_cell_border(cell, **kwargs):
    """
    设置单元格边框
    kwargs: top, bottom, left, right - 每个值为包含 sz, val, color 的字典
    """
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()

    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)

    for edge in ('left', 'top', 'right', 'bottom', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = 'w:{}'.format(edge)
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key in ["sz", "val", "color", "space"]:
                if key in edge_data:
                    element.set(qn('w:{}'.format(key)), str(edge_data[key]))


def set_cell_background(cell, color_hex):
    """设置单元格背景色"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), color_hex)
    tcPr.append(shd)


def set_table_border(table, color="000000", sz=6):
    """设置整个表格边框"""
    tbl = table._tbl
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)

    tblBorders = OxmlElement('w:tblBorders')
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), str(sz))
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), color)
        tblBorders.append(border)
    tblPr.append(tblBorders)


def add_paragraph_with_style(cell, text, bold=False, font_size=10, color=None, alignment=WD_ALIGN_PARAGRAPH.LEFT):
    """向单元格添加带格式的段落"""
    para = cell.paragraphs[0]
    para.alignment = alignment
    run = para.add_run(text)
    run.bold = bold
    run.font.size = Pt(font_size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    return para


def set_column_width(table, col_index, width_cm):
    """设置列宽"""
    for row in table.rows:
        row.cells[col_index].width = Cm(width_cm)


def add_page_border(doc):
    """向文档添加页面边框"""
    section = doc.sections[0]
    sectPr = section._sectPr
    pgBorders = OxmlElement('w:pgBorders')
    pgBorders.set(qn('w:offsetFrom'), 'page')
    for border_name in ['top', 'left', 'bottom', 'right']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '6')
        border.set(qn('w:space'), '24')
        border.set(qn('w:color'), '000000')
        pgBorders.append(border)
    sectPr.append(pgBorders)


def set_cell_margins(cell, top=0, start=100, bottom=0, end=100):
    """设置单元格内边距 (单位: twips)"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for side, val in [('top', top), ('start', start), ('bottom', bottom), ('end', end)]:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def build_first_page(doc, data):
    """
    构建文档第一页，布局与图示保持一致：
    - 页面顶部：文档标题区域
    - INFO 区块 1：White Paper Type / Classification
    - INFO 区块 2：Owner / Title of Change / Change Description / Reason for Change
    - INFO 区块 3：Process Factors 三列表格
    - 页脚：Rev 2.0
    """
    BLUE = (0x1F, 0x56, 0x9A)       # 标题蓝色
    BORDER_COLOR = "1F569A"
    LIGHT_BLUE = "D6E4F0"           # 表头浅蓝背景
    WHITE = "FFFFFF"

    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.5)

    usable_width_cm = 17.0

    # ── 页眉区域：文档编号、标题、副标题 ──────────────────────────
    header_table = doc.add_table(rows=3, cols=2)
    header_table.style = 'Table Grid'
    header_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_border(header_table, color=BORDER_COLOR, sz=6)

    # 第1行：左侧文档编号，右侧页码占位
    row0 = header_table.rows[0]
    row0.cells[0].merge(row0.cells[1])
    cell_docnum = row0.cells[0]
    p = cell_docnum.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run(data.get('doc_number', 'DOC-2024-001'))
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(100, 100, 100)

    # 第2行：文档大标题
    row1 = header_table.rows[1]
    row1.cells[0].merge(row1.cells[1])
    cell_title = row1.cells[0]
    p = cell_title.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(data.get('document_title', 'PROCESS CHANGE REQUEST'))
    run.bold = True
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(*BLUE)

    # 第3行：文档副标题
    row2 = header_table.rows[2]
    row2.cells[0].merge(row2.cells[1])
    cell_subtitle = row2.cells[0]
    p = cell_subtitle.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(data.get('document_subtitle', 'Change Management Form'))
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(80, 80, 80)

    # 设置表头表格列宽
    for row in header_table.rows:
        for cell in row.cells:
            set_cell_margins(cell, top=60, start=80, bottom=60, end=80)

    doc.add_paragraph()

    # ── INFO 区块 1 ──────────────────────────────────────────────
    info1_label = doc.add_paragraph()
    info1_label.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = info1_label.add_run('INFO')
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(*BLUE)

    table1 = doc.add_table(rows=2, cols=2)
    table1.style = 'Table Grid'
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_border(table1, color=BORDER_COLOR, sz=6)

    fields_block1 = [
        ('1. White Paper Type', data.get('white_paper_type', '')),
        ('2. Classification', data.get('classification', '')),
    ]

    for i, (label, value) in enumerate(fields_block1):
        row = table1.rows[i]
        # 标签列
        label_cell = row.cells[0]
        set_cell_background(label_cell, LIGHT_BLUE)
        p = label_cell.paragraphs[0]
        run = p.add_run(label)
        run.bold = True
        run.font.size = Pt(10)
        label_cell.width = Cm(4.5)

        # 值列
        value_cell = row.cells[1]
        p = value_cell.paragraphs[0]
        run = p.add_run(value)
        run.font.size = Pt(10)

        for cell in [label_cell, value_cell]:
            set_cell_margins(cell, top=60, start=100, bottom=60, end=100)

    doc.add_paragraph()

    # ── INFO 区块 2 ──────────────────────────────────────────────
    info2_label = doc.add_paragraph()
    run = info2_label.add_run('INFO')
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(*BLUE)

    table2 = doc.add_table(rows=4, cols=2)
    table2.style = 'Table Grid'
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_border(table2, color=BORDER_COLOR, sz=6)

    fields_block2 = [
        ('3. Name of Owner', data.get('name_of_owner', '')),
        ('4. Title of Change', data.get('title_of_change', '')),
        ('5. Change Description', data.get('change_description', '')),
        ('6. Reason for Change', data.get('reason_for_change', '')),
    ]

    row_heights = [Cm(1.0), Cm(1.0), Cm(2.0), Cm(2.0)]

    for i, (label, value) in enumerate(fields_block2):
        row = table2.rows[i]
        row.height = row_heights[i]

        label_cell = row.cells[0]
        set_cell_background(label_cell, LIGHT_BLUE)
        label_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
        p = label_cell.paragraphs[0]
        run = p.add_run(label)
        run.bold = True
        run.font.size = Pt(10)
        label_cell.width = Cm(4.5)

        value_cell = row.cells[1]
        value_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
        p = value_cell.paragraphs[0]
        run = p.add_run(value)
        run.font.size = Pt(10)

        for cell in [label_cell, value_cell]:
            set_cell_margins(cell, top=60, start=100, bottom=60, end=100)

    doc.add_paragraph()

    # ── INFO 区块 3：Process Factors ──────────────────────────────
    info3_label = doc.add_paragraph()
    run = info3_label.add_run('INFO')
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(*BLUE)

    pf_title = doc.add_paragraph()
    run = pf_title.add_run('7. Process Factors')
    run.bold = True
    run.font.size = Pt(10)

    process_factors = data.get('process_factors', [
        {'factor': '', 'present_value': '', 'proposed_value': ''},
    ])

    pf_rows = 1 + len(process_factors)
    table3 = doc.add_table(rows=pf_rows, cols=3)
    table3.style = 'Table Grid'
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_border(table3, color=BORDER_COLOR, sz=6)

    # 表头
    headers = ['Process Factor', 'Present Value', 'Proposed Value']
    header_row = table3.rows[0]
    col_widths = [Cm(6.5), Cm(5.0), Cm(5.5)]
    for j, (hdr, width) in enumerate(zip(headers, col_widths)):
        cell = header_row.cells[j]
        set_cell_background(cell, LIGHT_BLUE)
        p = cell.paragraphs[0]
        run = p.add_run(hdr)
        run.bold = True
        run.font.size = Pt(10)
        cell.width = width
        set_cell_margins(cell, top=60, start=100, bottom=60, end=100)

    # 数据行
    for i, pf in enumerate(process_factors):
        row = table3.rows[i + 1]
        row.height = Cm(1.0)
        values = [
            pf.get('factor', ''),
            pf.get('present_value', ''),
            pf.get('proposed_value', ''),
        ]
        for j, (val, width) in enumerate(zip(values, col_widths)):
            cell = row.cells[j]
            cell.width = width
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.size = Pt(10)
            set_cell_margins(cell, top=60, start=100, bottom=60, end=100)

    # ── 页脚：Rev 版本号 ─────────────────────────────────────────
    section = doc.sections[0]
    footer = section.footer
    footer_para = footer.paragraphs[0]
    footer_para.clear()

    footer_table = footer.add_table(rows=1, cols=3, width=Cm(usable_width_cm))
    footer_table.alignment = WD_TABLE_ALIGNMENT.CENTER

    footer_table.rows[0].cells[0].paragraphs[0].add_run(
        data.get('rev_number', 'Rev 2.0')
    ).font.size = Pt(9)

    center_p = footer_table.rows[0].cells[1].paragraphs[0]
    center_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    center_p.add_run('').font.size = Pt(9)

    right_p = footer_table.rows[0].cells[2].paragraphs[0]
    right_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    right_run = right_p.add_run()
    right_run.font.size = Pt(9)
    # 插入页码字段
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'PAGE'
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')
    right_run._r.append(fldChar1)
    right_run._r.append(instrText)
    right_run._r.append(fldChar2)


def generate_doc(request):
    """生成并返回 .docx 文件下载"""
    if request.method == 'POST':
        import json
        try:
            body = json.loads(request.body)
        except Exception:
            body = {}
        data = body
    else:
        # GET 请求时使用示例数据演示效果
        data = {
            'doc_number': 'PCR-2024-0042',
            'document_title': 'PROCESS CHANGE REQUEST',
            'document_subtitle': 'Engineering Change Management Form',
            'white_paper_type': 'Engineering Change Notice',
            'classification': 'Confidential',
            'name_of_owner': 'John Smith  |  Engineering Dept., Product Line A, john.smith@example.com',
            'title_of_change': 'Update soldering temperature profile for PCB assembly',
            'change_description': (
                'Change the reflow oven peak temperature from 245°C to 255°C '
                'to improve solder joint quality on BGA components.'
            ),
            'reason_for_change': (
                'Field failure analysis revealed cold solder joints on BGA packages. '
                'Updated profile resolves defect per IPC-7711/7721 guideline.'
            ),
            'process_factors': [
                {
                    'factor': 'Reflow Peak Temperature',
                    'present_value': '245°C ± 3°C',
                    'proposed_value': '255°C ± 3°C',
                },
                {
                    'factor': 'Time Above Liquidus',
                    'present_value': '30–40 s',
                    'proposed_value': '45–60 s',
                },
            ],
            'rev_number': 'Rev 2.0',
        }

    doc = Document()
    build_first_page(doc, data)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    response = HttpResponse(
        buffer.getvalue(),
        content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    )
    response['Content-Disposition'] = 'attachment; filename="process_change_request.docx"'
    return response


def index(request):
    """表单页：填写数据后提交生成文档"""
    return render(request, 'docgen/index.html')
