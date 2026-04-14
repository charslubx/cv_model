"""
WLA CCB Monitor Change White Paper 文档生成器

所有 build_* / _build_* 函数均返回 bytes 或操作 doc 对象，
不写磁盘文件。调用方通过 build_wla_ccb_document() 获取 bytes 流。
"""

import io
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# ---------------------------------------------------------------------------
# 颜色常量
# ---------------------------------------------------------------------------
BLUE   = RGBColor(0x00, 0x00, 0xFF)
GREEN  = RGBColor(0x00, 0x80, 0x00)
BLACK  = RGBColor(0x00, 0x00, 0x00)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

HEADER_BG = 'D3D3D3'   # 表头灰色背景


# ---------------------------------------------------------------------------
# 底层工具函数（与 document_builder.py 风格一致）
# ---------------------------------------------------------------------------

def _set_run_font(run, font_name='Arial', size_pt=11,
                  bold=False, color=None, italic=False, underline=False):
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size_pt)
    run.font.underline = underline
    if color:
        run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    for attr in ('w:ascii', 'w:hAnsi', 'w:cs'):
        rFonts.set(qn(attr), font_name)


def _set_cell_shading(cell, fill_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex.lstrip('#'))
    tc_pr.append(shd)


def _set_cell_valign(cell, align='top'):
    tc_pr = cell._tc.get_or_add_tcPr()
    v_align = tc_pr.find(qn('w:vAlign'))
    if v_align is None:
        v_align = OxmlElement('w:vAlign')
        tc_pr.append(v_align)
    v_align.set(qn('w:val'), align)


def _set_row_height(row, height_cm):
    tr_pr = row._tr.get_or_add_trPr()
    trH = OxmlElement('w:trHeight')
    trH.set(qn('w:val'), str(int(height_cm * 567)))
    trH.set(qn('w:hRule'), 'atLeast')
    tr_pr.append(trH)


def _set_table_total_width(tbl, width_inch):
    """强制设置表格总宽（inch），解除 autofit"""
    tbl.autofit = False
    tbl_elem = tbl._tbl
    tbl_pr = tbl_elem.find(qn('w:tblPr'))
    if tbl_pr is None:
        tbl_pr = OxmlElement('w:tblPr')
        tbl_elem.insert(0, tbl_pr)
    old = tbl_pr.find(qn('w:tblW'))
    if old is not None:
        tbl_pr.remove(old)
    tbl_w = OxmlElement('w:tblW')
    tbl_w.set(qn('w:w'), str(int(width_inch * 1440)))
    tbl_w.set(qn('w:type'), 'dxa')
    tbl_pr.append(tbl_w)


def _set_table_indent(tbl, indent_cm):
    tbl_elem = tbl._tbl
    tbl_pr = tbl_elem.find(qn('w:tblPr'))
    if tbl_pr is None:
        tbl_pr = OxmlElement('w:tblPr')
        tbl_elem.insert(0, tbl_pr)
    existing = tbl_pr.find(qn('w:tblInd'))
    if existing is not None:
        tbl_pr.remove(existing)
    ind = OxmlElement('w:tblInd')
    ind.set(qn('w:w'), str(int(indent_cm * 567)))
    ind.set(qn('w:type'), 'dxa')
    tbl_pr.append(ind)


def _cell_write(cell, text, align=WD_ALIGN_PARAGRAPH.LEFT,
                font_name='Arial', size_pt=11,
                bold=False, color=None, italic=False, valign='top'):
    """写入单元格文字"""
    _set_cell_valign(cell, valign)
    cell.text = ''
    para = cell.paragraphs[0]
    para.alignment = align
    run = para.add_run('' if text is None else str(text))
    _set_run_font(run, font_name=font_name, size_pt=size_pt,
                  bold=bold, color=color, italic=italic)
    return para


def _cell_add_line(cell, text, font_name='Arial', size_pt=11,
                   bold=False, color=None):
    """在单元格追加一个段落"""
    para = cell.add_paragraph()
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(0)
    run = para.add_run('' if text is None else str(text))
    _set_run_font(run, font_name=font_name, size_pt=size_pt,
                  bold=bold, color=color)
    return para


def _para_add_run(para, text, font_name='Arial', size_pt=11,
                  bold=False, color=None, italic=False, underline=False):
    run = para.add_run(text)
    _set_run_font(run, font_name=font_name, size_pt=size_pt,
                  bold=bold, color=color, italic=italic, underline=underline)
    return run


def _add_heading(doc, text, size_pt=11, space_before=6, space_after=4):
    """1) 2) 这类标题1：加粗"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    _set_run_font(r, size_pt=size_pt, bold=True)
    return p


def _add_sub_heading(doc, text, size_pt=11, space_before=4, space_after=2):
    """a) b) 这类标题2：正常粗细"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    _set_run_font(r, size_pt=size_pt)
    return p


# ---------------------------------------------------------------------------
# 章节构建函数
# ---------------------------------------------------------------------------

def _build_title(doc):
    """居中大标题，加粗+下划线"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run('WLA CCB Monitor Change White Paper')
    _set_run_font(r, size_pt=16, bold=True, underline=True)


def _build_section1(doc, data, page_w_cm):
    """1) Phase, Classification, Related WPs"""
    _add_heading(doc, '1) Phase, Classification, Related WPs:')

    total_twip = int(page_w_cm / 2.54 * 1440)
    col0_twip = int(total_twip * 0.38)
    col1_twip = total_twip - col0_twip
    col0_w = Cm(col0_twip / 567)
    col1_w = Cm(col1_twip / 567)

    # 主表：Phase / FWP Horizon / Classification / Class IV / 参考WP 大标题
    tbl = doc.add_table(rows=5, cols=2)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    # Row 0: Phase
    _set_row_height(tbl.rows[0], 0.6)
    c0, c1 = tbl.rows[0].cells
    c0.width = col0_w
    c1.width = col1_w
    _cell_write(c0, 'Phase:', bold=True, valign='center')
    _set_cell_shading(c0, 'F2F2F2')
    p1 = _cell_write(c1, '', valign='center')
    _para_add_run(p1, '☐ PWP')
    _para_add_run(p1, '    ☒ FWP')
    _set_cell_shading(c1, 'FFFFFF')

    # Row 1: FWP Horizon（合并列）
    _set_row_height(tbl.rows[1], 0.6)
    merged1 = tbl.rows[1].cells[0].merge(tbl.rows[1].cells[1])
    merged1.width = Cm(page_w_cm / 2.54)
    _set_cell_valign(merged1, 'center')
    merged1.text = ''
    p = merged1.paragraphs[0]
    _para_add_run(p, 'For a FWP, document the PWP Horizon number (if applicable): ')
    _para_add_run(p, data.get('fwp_horizon', 'N/a'), bold=True, color=BLUE)

    # Row 2: Classification
    _set_row_height(tbl.rows[2], 0.6)
    c0, c1 = tbl.rows[2].cells
    c0.width = col0_w
    c1.width = col1_w
    _cell_write(c0, 'Classification:', bold=True, valign='center')
    _set_cell_shading(c0, 'F2F2F2')
    p2 = _cell_write(c1, '', valign='center')
    _para_add_run(p2, '☐ 1   ☐ 2   ☐ 3   ☐ 3N   ☒ 4')
    _set_cell_shading(c1, 'FFFFFF')

    # Row 3: Class IV PCCB（合并列）
    _set_row_height(tbl.rows[3], 0.6)
    merged3 = tbl.rows[3].cells[0].merge(tbl.rows[3].cells[1])
    merged3.width = Cm(page_w_cm / 2.54)
    _set_cell_valign(merged3, 'center')
    merged3.text = ''
    p = merged3.paragraphs[0]
    _para_add_run(p, 'For Class IV WPs, add name of PCCB member confirming classification: ')
    _para_add_run(p, data.get('pccb_member', 'N/A'), bold=True, color=BLUE)

    # Row 4: 参考WP 大标题（合并列，加粗）
    _set_row_height(tbl.rows[4], 0.7)
    merged4 = tbl.rows[4].cells[0].merge(tbl.rows[4].cells[1])
    merged4.width = Cm(page_w_cm / 2.54)
    _set_cell_valign(merged4, 'center')
    _cell_write(
        merged4,
        'Include any relevant reference white paper(s), "Me-Too" WPs, DRB, MRB, etc. in table below',
        bold=True, valign='center',
    )
    _set_cell_shading(merged4, 'F2F2F2')

    # 参考WP 子表：Horizon | Title 表头 + N/a 行
    tbl_ref = doc.add_table(rows=2, cols=2)
    tbl_ref.style = 'Table Grid'
    tbl_ref.autofit = False
    _set_table_total_width(tbl_ref, page_w_cm / 2.54)

    _set_row_height(tbl_ref.rows[0], 0.55)
    tbl_ref.rows[0].cells[0].width = col0_w
    tbl_ref.rows[0].cells[1].width = col1_w
    _cell_write(tbl_ref.rows[0].cells[0], 'Horizon or reference number',
                italic=True, valign='center')
    _cell_write(tbl_ref.rows[0].cells[1], 'Title',
                italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, valign='center')

    _set_row_height(tbl_ref.rows[1], 0.55)
    ref_rows = data.get('reference_wps', [{'horizon': 'N/a', 'title': 'N/a'}])
    for col_i, key in enumerate(['horizon', 'title']):
        val = ref_rows[0].get(key, 'N/a') if ref_rows else 'N/a'
        _cell_write(tbl_ref.rows[1].cells[col_i], val, valign='center')
        _set_cell_shading(tbl_ref.rows[1].cells[col_i], 'FFFFFF')


def _build_section2(doc, data):
    """2) Date —— 标题1 加粗"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    _para_add_run(p, '2) Date: ', bold=True)
    _para_add_run(p, data.get('date', '04/02/2026'), bold=True, color=BLUE)


def _build_section3(doc, data):
    """3) Authorship"""
    _add_heading(doc, '3) Authorship', space_before=10)

    items = [
        ('a.', 'Primary author: ',            data.get('primary_author', 'Yuan, Ji'), True),
        ('b.', 'Site (primary author only): ', data.get('site', 'CDDP'),              True),
        ('c.', 'Co-author(s):',               data.get('co_authors', ''),             False),
    ]
    for letter, label, value, blue in items:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1.27)
        p.paragraph_format.first_line_indent = Cm(-0.63)
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        _para_add_run(p, f'{letter}  ')   # 标题2：普通粗细
        _para_add_run(p, label)
        if value:
            _para_add_run(p, value, bold=True, color=(BLUE if blue else None))


def _build_section4(doc, data):
    """4) Title of Change —— 标题1 加粗"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    _para_add_run(p, '4) Title of Change: ', bold=True)
    _para_add_run(p, data.get('title_of_change', 'CD DGB chart limit change for CLSR flag'),
                  bold=True, color=BLUE)


def _build_section5_header(doc, data):
    """5) Change Description 标题1（加粗） + a/b/c 标题2（普通）"""
    _add_heading(doc, '5) Change Description:', space_before=10)

    sub_items = [
        ('a.', 'Equipment tool set affected (entity code or CEID): ',
         data.get('equipment_tool_set', '[Tool Set / CEID]')),
        ('b.', 'Products affected (if change is product specific, otherwise "All"): ',
         data.get('products_affected', '[Products]')),
        ('c.', 'Specific change items.', None),
    ]
    for letter, label, value in sub_items:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1.27)
        p.paragraph_format.first_line_indent = Cm(-0.63)
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        _para_add_run(p, f'{letter}  ')   # 标题2：普通粗细
        _para_add_run(p, label)
        if value:
            _para_add_run(p, value, bold=True, color=BLUE)


def _build_change_table(doc, data, page_w_cm):
    """
    5c) 变更项目表格

    列宽（twip，合计精确等于 page_w_cm 对应 twip）：
      #(0.4in) | Change items(2.5in) | Present value(3.2in) | Proposed value(3.2in)
    """
    total_twip = int(page_w_cm / 2.54 * 1440)
    col_twips = [int(0.4 * 1440), int(2.5 * 1440), int(3.2 * 1440)]
    col_twips.append(total_twip - sum(col_twips))   # 剩余全给最后一列

    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    # 表头行
    _set_row_height(tbl.rows[0], 0.6)
    headers = ['#', 'Change items', 'Present value', 'Proposed value']
    for j, (txt, tw) in enumerate(zip(headers, col_twips)):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, txt, align=WD_ALIGN_PARAGRAPH.CENTER,
                    bold=True, valign='center')
        _set_cell_shading(c, HEADER_BG)

    # 数据行
    change_rows = data.get('change_rows', [])
    for rec in change_rows:
        row = tbl.add_row()
        _set_row_height(row, 1.8)
        row.cells[0].width = Cm(col_twips[0] / 567)
        row.cells[1].width = Cm(col_twips[1] / 567)
        row.cells[2].width = Cm(col_twips[2] / 567)
        row.cells[3].width = Cm(col_twips[3] / 567)

        # Col 0: 序号
        _cell_write(row.cells[0], rec.get('number', '1'),
                    align=WD_ALIGN_PARAGRAPH.CENTER, valign='top')

        # Col 1: Change items（三行）
        c1 = row.cells[1]
        _cell_write(c1, '', valign='top')
        p = c1.paragraphs[0]
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(2)
        _para_add_run(p, 'Monitor set name: ')
        _para_add_run(p, rec.get('monitor_set', ''), bold=True, color=BLUE)

        p2 = c1.add_paragraph()
        p2.paragraph_format.space_before = Pt(2)
        p2.paragraph_format.space_after = Pt(2)
        _para_add_run(p2, 'Measurement set name: ')
        _para_add_run(p2, rec.get('measurement_set', ''), bold=True, color=BLUE)

        p3 = c1.add_paragraph()
        p3.paragraph_format.space_before = Pt(2)
        p3.paragraph_format.space_after = Pt(1)
        _para_add_run(p3, 'Chart type: ')
        _para_add_run(p3, rec.get('chart_type', 'CLSR'), bold=True, color=BLUE)

        # Col 2 & Col 3: Present / Proposed value（UCL / Centerline / LCL / CLSR flag）
        for col_idx, prefix in [(2, 'present'), (3, 'proposed')]:
            cv = row.cells[col_idx]
            _cell_write(cv, '', valign='top')

            limit_keys = [
                ('UCL',        f'{prefix}_ucl'),
                ('Centerline', f'{prefix}_cl'),
                ('LCL',        f'{prefix}_lcl'),
            ]
            first = True
            for label, key in limit_keys:
                p = cv.paragraphs[0] if first else cv.add_paragraph()
                first = False
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(2)
                _para_add_run(p, f'{label}  :  ')
                _para_add_run(p, rec.get(key, ''), bold=True, color=BLUE)

            # CLSR flag 行
            p_clsr = cv.add_paragraph()
            p_clsr.paragraph_format.space_before = Pt(2)
            p_clsr.paragraph_format.space_after = Pt(1)
            _para_add_run(p_clsr, 'CLSR  :  ')
            flag_val = rec.get(f'{prefix}_clsr_flag', '')
            if flag_val:
                _para_add_run(p_clsr, flag_val, bold=True, color=GREEN)

    return tbl


# ---------------------------------------------------------------------------
# 文档组装
# ---------------------------------------------------------------------------

def _build_footer(doc, data):
    """在第一个 section 的页脚写入三栏内容：左-中-右"""
    sec = doc.sections[0]
    footer = sec.footer
    footer.is_linked_to_previous = False

    # 清空已有段落
    for p in footer.paragraphs:
        p.clear()

    fp = footer.paragraphs[0]
    fp.paragraph_format.space_before = Pt(0)
    fp.paragraph_format.space_after = Pt(0)

    left_text   = data.get('footer_left',   'Intel Confidential')
    center_text = data.get('footer_center', 'WLA CCB Monitor Change White Paper')
    right_text  = data.get('footer_right',  'Rev 1.0')

    # 利用制表符实现左-中-右三栏布局
    # 段落格式：居中制表位 + 右对齐制表位
    from docx.oxml import OxmlElement as _el
    pPr = fp._p.get_or_add_pPr()
    tabs = _el('w:tabs')

    tab_center = _el('w:tab')
    tab_center.set(qn('w:val'), 'center')
    tab_center.set(qn('w:pos'), '4680')   # 约页面中央（9360 twip / 2）

    tab_right = _el('w:tab')
    tab_right.set(qn('w:val'), 'right')
    tab_right.set(qn('w:pos'), '9360')    # 右边界

    tabs.append(tab_center)
    tabs.append(tab_right)
    pPr.append(tabs)

    _para_add_run(fp, left_text,   size_pt=9)
    _para_add_run(fp, '\t',        size_pt=9)
    _para_add_run(fp, center_text, size_pt=9)
    _para_add_run(fp, '\t',        size_pt=9)
    _para_add_run(fp, right_text,  size_pt=9)


def _apply_doc_settings(doc, page_w_cm=21.59, page_h_cm=27.94):
    """页面设置（A4 竖向，2.54cm 边距）"""
    sec = doc.sections[0]
    sec.page_width  = Cm(page_w_cm)
    sec.page_height = Cm(page_h_cm)
    sec.left_margin   = Cm(2.54)
    sec.right_margin  = Cm(2.54)
    sec.top_margin    = Cm(2.54)
    sec.bottom_margin = Cm(2.54)


def build_wla_ccb_document(data: dict) -> bytes:
    """
    生成 WLA CCB Monitor Change White Paper 文档。

    参数
    ----
    data : dict
        文档内容字典，支持以下键（均有默认值，可按需覆盖）：

        fwp_horizon      str   FWP Horizon 编号，默认 'N/a'
        pccb_member      str   PCCB 成员名，默认 'N/A'
        reference_wps    list  [{'horizon': ..., 'title': ...}]，默认 [{'horizon':'N/a','title':'N/a'}]
        date             str   日期，默认 '04/02/2026'
        primary_author   str   主要作者，默认 'Yuan, Ji'
        site             str   站点，默认 'CDDP'
        co_authors       str   合著者，默认 ''
        title_of_change  str   变更标题
        equipment_tool_set  str
        products_affected   str
        change_rows      list  每条变更记录 dict，包含：
            number, monitor_set, measurement_set, chart_type
            present_ucl, present_cl, present_lcl, present_clsr_flag
            proposed_ucl, proposed_cl, proposed_lcl, proposed_clsr_flag

    返回
    ----
    bytes : 可直接作为 HTTP 响应体或写入 .docx 文件。
    """
    doc = Document()
    _apply_doc_settings(doc)

    # A4 内容区宽度 = 21.59 - 2.54*2
    page_w_cm = 21.59 - 2.54 * 2   # ≈ 16.51 cm

    _build_title(doc)
    _build_section1(doc, data, page_w_cm)
    _build_section2(doc, data)
    _build_section3(doc, data)
    _build_section4(doc, data)
    _build_section5_header(doc, data)
    _build_change_table(doc, data, page_w_cm)
    _build_footer(doc, data)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

def main():
    sample_data = {
        'fwp_horizon': 'N/a',
        'pccb_member': 'N/A',
        'reference_wps': [{'horizon': 'N/a', 'title': 'N/a'}],
        'date': '04/02/2026',
        'primary_author': 'Yuan, Ji',
        'site': 'CDDP',
        'co_authors': '',
        'title_of_change': 'CD DGB chart limit change for CLSR flag',
        'equipment_tool_set': 'EUV_Tool_A / CEID-12345',
        'products_affected': 'All',
        'footer_left':   'Intel Confidential',
        'footer_center': 'WLA CCB Monitor Change White Paper',
        'footer_right':  'Rev 1.0',
        'change_rows': [
            {
                'number': '1',
                'monitor_set': 'MON_SET_001',
                'measurement_set': 'MEAS_SET_001',
                'chart_type': 'CLSR',
                'present_ucl': '3.50', 'present_cl': '2.10', 'present_lcl': '0.70',
                'present_clsr_flag': 'Flag',
                'proposed_ucl': '3.80', 'proposed_cl': '2.20', 'proposed_lcl': '0.60',
                'proposed_clsr_flag': '',
            },
            {
                'number': '1',
                'monitor_set': 'MON_SET_002',
                'measurement_set': 'MEAS_SET_002',
                'chart_type': 'CLSR',
                'present_ucl': '4.00', 'present_cl': '2.50', 'present_lcl': '1.00',
                'present_clsr_flag': 'Flag',
                'proposed_ucl': '4.20', 'proposed_cl': '2.60', 'proposed_lcl': '1.00',
                'proposed_clsr_flag': '',
            },
            {
                'number': '1',
                'monitor_set': 'MON_SET_003',
                'measurement_set': 'MEAS_SET_003',
                'chart_type': 'CLSR',
                'present_ucl': '2.90', 'present_cl': '1.80', 'present_lcl': '0.70',
                'present_clsr_flag': '',
                'proposed_ucl': '3.10', 'proposed_cl': '1.90', 'proposed_lcl': '0.70',
                'proposed_clsr_flag': '',
            },
        ],
    }

    docx_bytes = build_wla_ccb_document(sample_data)

    output_path = 'WLA_CCB_Monitor_Change_White_Paper.docx'
    with open(output_path, 'wb') as f:
        f.write(docx_bytes)

    print(f'字节长度: {len(docx_bytes):,}')
    print(f'已写入: {output_path}')


if __name__ == '__main__':
    main()
