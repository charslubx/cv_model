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


def _set_font_name(run, name='Arial', size_pt=None, color=None):
    """设置字体名（及可选的字号、颜色），不触碰 bold 让样式自己定义"""
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    if color is not None:
        run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    for attr in ('w:ascii', 'w:hAnsi', 'w:cs'):
        rFonts.set(qn(attr), name)


def _add_heading(doc, text, space_before=None, space_after=None):
    """
    1) 2) 这类 —— Heading 1，Arial 12号黑色，大纲级别1。
    """
    p = doc.add_paragraph(style='Heading 1')
    if space_before is not None:
        p.paragraph_format.space_before = Pt(space_before)
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    p.clear()
    r = p.add_run(text)
    _set_font_name(r, size_pt=12, color=BLACK)
    return p


def _add_sub_item(doc, letter, label, value=None, blue=False):
    """
    a. b. c. 这类 —— Heading 2，Arial 10号黑色，大纲级别2。
    value 若为蓝色则单独设 color=BLUE。
    """
    p = doc.add_paragraph(style='Heading 2')
    p.paragraph_format.left_indent = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(-0.63)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.clear()
    _set_font_name(p.add_run(f'{letter}  '), size_pt=10, color=BLACK)
    _set_font_name(p.add_run(label),          size_pt=10, color=BLACK)
    if value:
        r_val = p.add_run(value)
        _set_font_name(r_val, size_pt=10, color=BLUE if blue else BLACK)
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
    """2) Date —— Heading 1"""
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.clear()
    _set_font_name(p.add_run('2) Date: '), size_pt=12, color=BLACK)
    _set_font_name(p.add_run(data.get('date', '04/02/2026')), size_pt=12, color=BLUE)


def _build_section3(doc, data):
    """3) Authorship"""
    _add_heading(doc, '3) Authorship', space_before=10)

    items = [
        ('a.', 'Primary author: ',            data.get('primary_author', 'Yuan, Ji'), True),
        ('b.', 'Site (primary author only): ', data.get('site', 'CDDP'),              True),
        ('c.', 'Co-author(s):',               data.get('co_authors', ''),             False),
    ]
    for letter, label, value, blue in items:
        _add_sub_item(doc, letter, label, value=value, blue=blue)


def _build_section4(doc, data):
    """4) Title of Change —— Heading 1"""
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.clear()
    _set_font_name(p.add_run('4) Title of Change: '), size_pt=12, color=BLACK)
    _set_font_name(p.add_run(data.get('title_of_change', 'CD DGB chart limit change for CLSR flag')),
                   size_pt=12, color=BLUE)


def _build_section5_header(doc, data):
    """5) Change Description —— Heading 1 + a/b/c Heading 2"""
    _add_heading(doc, '5) Change Description:', space_before=10)

    sub_items = [
        ('a.', 'Equipment tool set affected (entity code or CEID): ',
         data.get('equipment_tool_set', '[Tool Set / CEID]'), True),
        ('b.', 'Products affected (if change is product specific, otherwise "All"): ',
         data.get('products_affected', '[Products]'), True),
        ('c.', 'Specific change items.', None, False),
    ]
    for letter, label, value, blue in sub_items:
        _add_sub_item(doc, letter, label, value=value, blue=blue)


def _set_table_no_borders(tbl):
    """去掉表格所有单元格边框（嵌套表格用）"""
    tbl_elem = tbl._tbl
    tbl_pr = tbl_elem.find(qn('w:tblPr'))
    if tbl_pr is None:
        tbl_pr = OxmlElement('w:tblPr')
        tbl_elem.insert(0, tbl_pr)
    tbl_borders = OxmlElement('w:tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{side}')
        el.set(qn('w:val'), 'none')
        el.set(qn('w:sz'), '0')
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), 'auto')
        tbl_borders.append(el)
    old = tbl_pr.find(qn('w:tblBorders'))
    if old is not None:
        tbl_pr.remove(old)
    tbl_pr.append(tbl_borders)


def _set_cell_no_padding(cell):
    """去掉单元格内边距，让嵌套表格紧贴边框"""
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement('w:tcMar')
    for side in ('top', 'left', 'bottom', 'right'):
        el = OxmlElement(f'w:{side}')
        el.set(qn('w:w'), '0')
        el.set(qn('w:type'), 'dxa')
        mar.append(el)
    old = tc_pr.find(qn('w:tcMar'))
    if old is not None:
        tc_pr.remove(old)
    tc_pr.append(mar)


def _build_inner_value_table(doc, cell, limit_rows, cell_twip):
    """
    在外层单元格 cell 内嵌入一个 3 列无边框表格。

    limit_rows : list of dict，每行包含：
        label      str   第1列标签（如 'UCL'）
        value      str   第3列数值
        color      RGBColor | None   第3列文字颜色，None 表示默认黑色
    cell_twip  : 外层单元格宽度（twip），用于计算内嵌表格总宽

    内嵌表格列宽分配：label 45% | colon 10% | value 45%
    """
    n = len(limit_rows)
    if n == 0:
        return

    label_twip  = int(cell_twip * 0.45)
    colon_twip  = int(cell_twip * 0.10)
    value_twip  = cell_twip - label_twip - colon_twip

    # 在 doc body 临时创建表格，再移入 cell
    nested = doc.add_table(rows=n, cols=3)
    nested.style = 'Table Grid'
    nested.autofit = False
    _set_table_total_width(nested, cell_twip / 1440)
    _set_table_no_borders(nested)

    for i, row_data in enumerate(limit_rows):
        nr = nested.rows[i]
        _set_row_height(nr, 0.5)

        c0 = nr.cells[0]
        c0.width = Cm(label_twip / 567)
        _set_cell_no_padding(c0)
        _cell_write(c0, row_data.get('label', ''), size_pt=10, valign='center')

        c1 = nr.cells[1]
        c1.width = Cm(colon_twip / 567)
        _set_cell_no_padding(c1)
        _cell_write(c1, ':', size_pt=10,
                    align=WD_ALIGN_PARAGRAPH.CENTER, valign='center')

        c2 = nr.cells[2]
        c2.width = Cm(value_twip / 567)
        _set_cell_no_padding(c2)
        val = row_data.get('value', '')
        clr = row_data.get('color', None)
        _cell_write(c2, val, size_pt=10, bold=bool(val),
                    color=clr, valign='center')

    # 从 body 摘出，插入 cell 的 tc 元素（放在末尾空段落之前）
    nested_tbl_el = nested._tbl
    nested_tbl_el.getparent().remove(nested_tbl_el)

    tc = cell._tc
    # tc 末尾须保留一个 w:p（OOXML 规范），把表格插到最后一个 p 之前
    last_p = tc.findall(qn('w:p'))[-1]
    tc.insert(list(tc).index(last_p), nested_tbl_el)


def _limits_from_rec(rec, prefix):
    """
    从 change_row 记录中提取某个前缀（present / proposed）的 limit_rows。

    优先读取 rec['limits'] 列表（每项含 label / present / proposed）；
    若无则退化到旧式扁平键（{prefix}_ucl / {prefix}_cl 等）。

    返回 list of {'label', 'value', 'color'}
    """
    if 'limits' in rec:
        rows = []
        for item in rec['limits']:
            val = item.get(prefix, '')
            is_flag = item.get(f'{prefix}_is_flag', False)
            rows.append({
                'label': item.get('label', ''),
                'value': val,
                'color': GREEN if (is_flag and val) else (BLUE if val else None),
            })
        return rows

    # 旧式键退化兼容
    default_limits = [
        ('UCL',        f'{prefix}_ucl',       False),
        ('Centerline', f'{prefix}_cl',        False),
        ('LCL',        f'{prefix}_lcl',       False),
        ('CLSR',       f'{prefix}_clsr_flag', True),
    ]
    rows = []
    for label, key, is_flag in default_limits:
        val = rec.get(key, '')
        rows.append({
            'label': label,
            'value': val,
            'color': GREEN if (is_flag and val) else (BLUE if val else None),
        })
    return rows


def _build_change_table(doc, data, page_w_cm):
    """
    5c) 变更项目表格

    列宽（twip）：# | Change items | Present value | Proposed value
    Present / Proposed 列内部各放一个 3×n 无边框嵌套表格：
        label | : | value
    行数由每条记录的 limits 列表决定。
    """
    total_twip = int(page_w_cm / 2.54 * 1440)
    num_twip      = int(0.4 * 1440)   # # 列
    present_twip  = int(2.2 * 1440)   # Present value
    proposed_twip = int(2.2 * 1440)   # Proposed value
    items_twip    = total_twip - num_twip - present_twip - proposed_twip  # Change items（剩余）
    col_twips = [num_twip, items_twip, present_twip, proposed_twip]

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
        for j, tw in enumerate(col_twips):
            row.cells[j].width = Cm(tw / 567)

        # Col 0: 序号
        _cell_write(row.cells[0], rec.get('number', '1'),
                    align=WD_ALIGN_PARAGRAPH.CENTER, valign='top')

        # Col 1: Change items（Monitor set / Measurement set / Chart type）
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

        # Col 2 & Col 3: 嵌套 3×n 表格
        for col_idx, prefix in [(2, 'present'), (3, 'proposed')]:
            cv = row.cells[col_idx]
            _set_cell_valign(cv, 'top')
            cv.text = ''                          # 清空，保留末尾空 p
            limit_rows = _limits_from_rec(rec, prefix)
            _build_inner_value_table(doc, cv, limit_rows, col_twips[col_idx])

    return tbl


# ---------------------------------------------------------------------------
# 文档组装
# ---------------------------------------------------------------------------

def _build_page2(doc, data, page_w_cm):
    """第二页：第6-9节（自然排版，无强制分页符）"""
    _build_section6(doc, data)
    _build_section7(doc, data, page_w_cm)
    _build_section8(doc, data, page_w_cm)
    _build_section9(doc, data, page_w_cm)


def _build_section5_fwp_table(doc, data, page_w_cm):
    """
    5) a. (FWP only)* 子项 + Previous/Current value 对比表。
    紧跟在 c. Specific change items 的嵌套表之后。
    """
    # a. (FWP only)* 子项 —— Heading 2
    p = doc.add_paragraph(style='Heading 2')
    p.paragraph_format.left_indent = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(-0.63)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.clear()
    _set_font_name(p.add_run('a.  '), size_pt=10, color=BLACK)
    r_fwp = p.add_run('(FWP only)*')
    _set_font_name(r_fwp, size_pt=10, color=BLACK)
    r_fwp.font.bold = True
    desc = data.get(
        'fwp_only_desc',
        ' Incorporate the C-Spec (or equivalent) into the PWP Control Plan for '
        'future PWP versions. Otherwise, leave this blank.'
    )
    _set_font_name(p.add_run(desc), size_pt=10, color=BLACK)

    # 三列表格：# | Change items | Previous value | Current value
    total_twip = int(page_w_cm / 2.54 * 1440)
    num_twip   = int(0.4 * 1440)
    val_twip   = int(2.2 * 1440)
    items_twip = total_twip - num_twip - val_twip * 2

    col_twips = [num_twip, items_twip, val_twip, val_twip]
    headers   = ['#', 'Change items', 'Previous value', 'Current value']

    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    _set_row_height(tbl.rows[0], 0.6)
    for j, (txt, tw) in enumerate(zip(headers, col_twips)):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, txt, bold=True, size_pt=10,
                    align=WD_ALIGN_PARAGRAPH.CENTER, valign='center')
        _set_cell_shading(c, HEADER_BG)

    fwp_rows = data.get('fwp_change_rows', [])
    for rec in fwp_rows:
        row = tbl.add_row()
        _set_row_height(row, 0.8)
        for j, (key, tw) in enumerate(
                zip(['number', 'change_item', 'previous_value', 'current_value'],
                    col_twips)):
            c = row.cells[j]
            c.width = Cm(tw / 567)
            _cell_write(c, rec.get(key, ''), size_pt=10, valign='top')
            _set_cell_shading(c, 'FFFFFF')

    # 若无数据补两个空行
    if not fwp_rows:
        for _ in range(2):
            row = tbl.add_row()
            _set_row_height(row, 0.6)
            for j, tw in enumerate(col_twips):
                row.cells[j].width = Cm(tw / 567)


def _build_section6(doc, data):
    """6) Reason for Change"""
    _add_heading(doc, '6) Reason for Change:')
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    _para_add_run(p, data.get('reason_for_change', ''), color=BLUE, size_pt=11)


def _build_section7(doc, data, page_w_cm):
    """
    7) CEI/Site Implementation Owners
    表格：CEI Owners | SITE | Date reviewed and approved
    下方有一个要点说明（bullet）
    """
    # 标题：CEI 加下划线
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.clear()
    _set_font_name(p.add_run('7) '), size_pt=12, color=BLACK)
    r_cei = p.add_run('CEI')
    _set_font_name(r_cei, size_pt=12, color=BLACK)
    r_cei.font.underline = True
    _set_font_name(p.add_run('/Site Implementation Owners:'), size_pt=12, color=BLACK)

    # 表格：3列
    total_twip = int(page_w_cm / 2.54 * 1440)
    col_twips  = [int(total_twip * 0.38), int(total_twip * 0.12),
                  total_twip - int(total_twip * 0.38) - int(total_twip * 0.12)]

    tbl = doc.add_table(rows=2, cols=3)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    # 表头行
    _set_row_height(tbl.rows[0], 0.6)
    headers = ['CEI Owners:', 'SITE', 'Date reviewed and approved:']
    for j, (txt, tw) in enumerate(zip(headers, col_twips)):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, txt, bold=True, size_pt=10, valign='center')
        _set_cell_shading(c, 'F2F2F2')

    # 数据行
    _set_row_height(tbl.rows[1], 0.7)
    owners = data.get('cei_owners', [{'name': '', 'site': 'CDDP', 'date': ''}])
    owner  = owners[0] if owners else {}
    for j, (key, tw) in enumerate(zip(['name', 'site', 'date'], col_twips)):
        c = tbl.rows[1].cells[j]
        c.width = Cm(tw / 567)
        val = owner.get(key, '')
        color = BLUE if key == 'date' and val else None
        _cell_write(c, val, size_pt=10, color=color, valign='center')
        _set_cell_shading(c, 'FFFFFF')

    # 额外 owners 行
    for owner in owners[1:]:
        row = tbl.add_row()
        _set_row_height(row, 0.7)
        for j, (key, tw) in enumerate(zip(['name', 'site', 'date'], col_twips)):
            c = row.cells[j]
            c.width = Cm(tw / 567)
            val = owner.get(key, '')
            color = BLUE if key == 'date' and val else None
            _cell_write(c, val, size_pt=10, color=color, valign='center')
            _set_cell_shading(c, 'FFFFFF')

    # 要点说明（bullet）
    note = data.get('cei_note',
        'Any owners responsible for both FWP and PWP stages.')
    bp = doc.add_paragraph(style='List Bullet')
    bp.paragraph_format.space_before = Pt(4)
    bp.paragraph_format.space_after  = Pt(2)
    _para_add_run(bp, note, size_pt=10)


def _build_section8(doc, data, page_w_cm):
    """
    8) Specifications, Controlled Documents Affected
    标题带括号说明（小字斜体），下方两列表格
    """
    # 标题1
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    p.clear()
    _set_font_name(p.add_run('8) Specifications, Controlled Documents Affected:'),
                   size_pt=12, color=BLACK)
    r_note = p.add_run('  (list all affected by changes above)')
    _set_font_name(r_note, size_pt=10, color=BLACK)
    r_note.font.italic = True

    # 两列表格
    total_twip = int(page_w_cm / 2.54 * 1440)
    col0_twip  = int(total_twip * 0.45)
    col1_twip  = total_twip - col0_twip

    tbl = doc.add_table(rows=1, cols=2)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    # 表头
    _set_row_height(tbl.rows[0], 0.6)
    for j, (txt, tw) in enumerate(
            zip(['Spec and/or Controlled Document #', 'Document Title'],
                [col0_twip, col1_twip])):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, txt, bold=True, size_pt=10,
                    align=WD_ALIGN_PARAGRAPH.CENTER, valign='center')
        _set_cell_shading(c, HEADER_BG)

    # 数据行
    spec_rows = data.get('spec_rows', [{'spec_num': 'N/a', 'title': 'N/a'}])
    for rec in spec_rows:
        row = tbl.add_row()
        _set_row_height(row, 0.6)
        row.cells[0].width = Cm(col0_twip / 567)
        row.cells[1].width = Cm(col1_twip / 567)
        _cell_write(row.cells[0], rec.get('spec_num', ''), size_pt=10, valign='center')
        _cell_write(row.cells[1], rec.get('title', ''),    size_pt=10, valign='center')
        _set_cell_shading(row.cells[0], 'FFFFFF')
        _set_cell_shading(row.cells[1], 'FFFFFF')


def _build_section9(doc, data, page_w_cm):
    """
    9) Concerns and Considerations
    5列表格：# | Forum identifying concern | Issue | Resolution | Status
    """
    _add_heading(doc, '9) Concerns and Considerations:')

    total_twip = int(page_w_cm / 2.54 * 1440)
    # 列宽比例：# 小, Forum 适中, Issue 适中, Resolution 最宽, Status 小
    col_ratios = [0.05, 0.15, 0.20, 0.45, 0.15]
    col_twips  = [int(total_twip * r) for r in col_ratios]
    col_twips[-1] = total_twip - sum(col_twips[:-1])   # 最后列补齐误差

    headers = ['#', 'Forum\nidentifying\nconcern', 'Issue', 'Resolution',
               'Status:\n(Open or\nClosed)']

    tbl = doc.add_table(rows=1, cols=5)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    # 表头行
    _set_row_height(tbl.rows[0], 1.2)
    for j, (txt, tw) in enumerate(zip(headers, col_twips)):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, txt, bold=True, size_pt=10,
                    align=WD_ALIGN_PARAGRAPH.CENTER, valign='center')
        _set_cell_shading(c, HEADER_BG)

    # 数据行
    concern_rows = data.get('concern_rows', [])
    for rec in concern_rows:
        row = tbl.add_row()
        _set_row_height(row, 1.5)
        for j, tw in enumerate(col_twips):
            row.cells[j].width = Cm(tw / 567)

        keys = ['number', 'forum', 'issue', 'resolution', 'status']
        blue_cols = {0, 1, 4}   # #/Forum/Status 用蓝色
        for j, key in enumerate(keys):
            val = rec.get(key, '')
            color = BLUE if (j in blue_cols and val) else None
            _cell_write(row.cells[j], val, size_pt=10,
                        color=color, valign='top')

    # 若无数据行，补两个空行
    if not concern_rows:
        for _ in range(2):
            row = tbl.add_row()
            _set_row_height(row, 0.8)
            for j, tw in enumerate(col_twips):
                row.cells[j].width = Cm(tw / 567)

def _build_section10(doc, data, page_w_cm):
    """10) Control Chart Setup"""

    # ---------- 10) 标题 ----------
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(4)
    p.clear()
    _set_font_name(p.add_run(
        '10) Control Chart Setup: Only fill in section for SPC++ changes or SPC# chart creation'
    ), size_pt=12, color=BLACK)

    # ---------- a. Chart Modification ----------
    _add_sub_item(doc, 'a.', 'Chart Modification: (check one)')

    mod_sel = data.get('chart_modification', 'revision')
    for key, label in [
        ('revision',   'Chart Revision'),
        ('new',        'New Chart Creation. Please\ncomplete section 10b.'),
        ('deletion',   'Chart Deletion'),
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent    = Cm(2.54)
        p.paragraph_format.space_before   = Pt(1)
        p.paragraph_format.space_after    = Pt(1)
        mark = '☒' if mod_sel == key else '☐'
        _para_add_run(p, f'{mark}    {label}', size_pt=10)

    # ---------- b. Chart By ----------
    _add_sub_item(doc, 'b.', 'Chart By: (check one and fill out table below, only needed for New Chart Creation)')

    chart_by = data.get('chart_by', '')
    cb_options = [
        [('equipment',  'Equipment'),      ('all_categories', 'All Categories')],
        [('monitor',    'Monitor'),         ('process',        'Process')],
        [('operation',  'Operation'),       ('custom_context', 'Custom Context Categories')],
        [('product',    'Product'),         (None,             '')],
    ]
    for row_opts in cb_options:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent  = Cm(2.54)
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after  = Pt(1)
        parts = []
        for key, label in row_opts:
            if not label:
                parts.append('')
                continue
            mark = '☒' if chart_by == key else '☐'
            parts.append(f'{mark}  {label}')
        _para_add_run(p, f'{parts[0]:<35}{parts[1]}', size_pt=10)

    # 15列 SPC 表格（b 和 c 共用同一结构）
    _build_spc15_table(doc, data.get('spc_setup_rows', []), page_w_cm, prefix='setup')

    # Bullet 说明
    for note in data.get('spc_setup_notes', [
        'Oper: refer to SPC++ documentation.',
        'Class: refer to SPC++ documentation.',
        'Calc Method: (raw, percentage). For SPC#, Eng review required.',
    ]):
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_before = Pt(2)
        bp.paragraph_format.space_after  = Pt(2)
        _para_add_run(bp, note, size_pt=10)

    # ---------- c. SPC Rules ----------
    _add_sub_item(doc, 'c.', 'SPC Rules: (check one and fill out table below)')

    rules_sel = data.get('spc_rules', 'no_changes')
    for key, label in [
        ('no_rules',    'No Rules Set'),
        ('no_changes',  'No Changes Proposed'),
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent  = Cm(2.54)
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after  = Pt(1)
        mark = '☒' if rules_sel == key else '☐'
        _para_add_run(p, f'{mark}    {label}', size_pt=10)

    # 说明文字
    p_note = doc.add_paragraph()
    p_note.paragraph_format.space_before = Pt(4)
    p_note.paragraph_format.space_after  = Pt(4)
    _para_add_run(p_note, data.get('spc_rules_note',
        'Complete the following if there are changes in rules, note present rules.'),
        size_pt=10)

    # c 的 SPC 表格
    _build_spc15_table(doc, data.get('spc_rules_rows', []), page_w_cm, prefix='rules')

    # Rules 图例表
    _build_rules_legend(doc, page_w_cm)

    # 自定义规则说明
    p_custom = doc.add_paragraph()
    p_custom.paragraph_format.space_before = Pt(4)
    p_custom.paragraph_format.space_after  = Pt(4)
    _para_add_run(p_custom,
        'If you have more custom rules that do not fit the standard rule codes above, '
        'provide details here.', size_pt=10)

    # ---------- d. Control Chart Data Summary ----------
    _add_sub_item(doc, 'd.', 'Control Chart Data Summary:')


def _build_spc15_table(doc, data_rows, page_w_cm, prefix='setup'):
    """
    SPC 15列宽表格（Section 10b / 10c 共用）。
    列：Oper | SPC_FUNCTIONAL_AREA | MONITOR_SET_NAME | MEASUREMENT_SET_NAME |
        CHART_SUBSET/NUMBER | TYPE | LCL | CL | TARGET | LDL | UDL | LUL | UBL | Class | Calc Method
    """
    total_twip = int(page_w_cm / 2.54 * 1440)

    # 各列 twip（合计 = total_twip）
    col_defs = [
        ('Oper',                  0.045),
        ('SPC_\nFUNCTIONAL\n_AREA', 0.085),
        ('MONITOR_\nSET_NAME',    0.090),
        ('MEASUREMENT\n_SET_NAME',0.095),
        ('CHART_\nSUBSET/\nNUMBER', 0.060),
        ('TYPE',                  0.050),
        ('LCL',                   0.055),
        ('CL',                    0.055),
        ('TARGET',                0.060),
        ('LDL',                   0.050),
        ('UDL',                   0.050),
        ('LUL',                   0.050),
        ('UBL',                   0.050),
        ('Class',                 0.060),
        ('Calc\nMethod',          0.095),
    ]
    # 归一化确保列宽合计精确等于 total_twip
    ratio_sum = sum(r for _, r in col_defs)
    col_twips = [int(total_twip * r / ratio_sum) for _, r in col_defs]
    col_twips[-1] = total_twip - sum(col_twips[:-1])

    col_headers = [h for h, _ in col_defs]

    tbl = doc.add_table(rows=1, cols=len(col_defs))
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    _set_row_height(tbl.rows[0], 1.3)
    for j, (hdr, tw) in enumerate(zip(col_headers, col_twips)):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, hdr, bold=True, size_pt=8,
                    align=WD_ALIGN_PARAGRAPH.CENTER, valign='center')
        _set_cell_shading(c, HEADER_BG)

    # 数据行 / 空行
    rows_to_fill = data_rows if data_rows else [{} for _ in range(3)]
    keys = ['oper', 'spc_area', 'monitor_set', 'measurement_set',
            'subset_num', 'type', 'lcl', 'cl', 'target',
            'ldl', 'udl', 'lul', 'ubl', 'class_', 'calc_method']
    for rec in rows_to_fill:
        row = tbl.add_row()
        _set_row_height(row, 0.65)
        for j, (key, tw) in enumerate(zip(keys, col_twips)):
            c = row.cells[j]
            c.width = Cm(tw / 567)
            _cell_write(c, rec.get(key, ''), size_pt=8, valign='center')
            _set_cell_shading(c, 'FFFFFF')

    return tbl


def _build_rules_legend(doc, page_w_cm):
    """Rules 图例说明段落 + 两列对照小表"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(2)
    r = p.add_run('*Rules are denoted as follows:')
    _set_run_font(r, size_pt=9, bold=False)

    legend = [
        ('A',    '> UCL'),
        ('B',    '2/1 > 2 SIGMA'),
        ('C',    '4/5 > 1 SIGMA'),
        ('D',    'Last 8 > CL'),
        ('E',    '< LCL'),
        ('F',    '2/3 < -2 SIGMA'),
        ('G',    '4/5 < -1 SIGMA'),
        ('H',    'Last 8 < CL'),
        ('I',    '15 Inside SIGMA'),
        ('J',    '8 Outside SIGMA'),
        ('K',    '> UDL'),
        ('L',    '< LDL'),
        ('M',    'Missing Data'),
        ('N',    'Fail Disposition'),
        ('',     'No Limits'),
        ('NONE', 'No OOC Rules'),
    ]

    # 每行4条，共4列（code + desc 交替）
    cols_per_row = 4
    total_twip   = int(page_w_cm / 2.54 * 1440)
    code_twip    = int(total_twip / cols_per_row * 0.25)
    desc_twip    = int(total_twip / cols_per_row * 0.75)
    n_rows = (len(legend) + cols_per_row - 1) // cols_per_row

    tbl = doc.add_table(rows=n_rows, cols=cols_per_row * 2)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)
    _set_table_no_borders(tbl)

    for i, (code, desc) in enumerate(legend):
        ri = i // cols_per_row
        ci = (i % cols_per_row) * 2
        _set_row_height(tbl.rows[ri], 0.4)
        c_code = tbl.rows[ri].cells[ci]
        c_code.width = Cm(code_twip / 567)
        _cell_write(c_code, code, size_pt=8, bold=True, valign='center')
        c_desc = tbl.rows[ri].cells[ci + 1]
        c_desc.width = Cm(desc_twip / 567)
        _cell_write(c_desc, desc, size_pt=8, valign='center')


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
    _build_section5_fwp_table(doc, data, page_w_cm)
    _build_page2(doc, data, page_w_cm)
    _build_section10(doc, data, page_w_cm)
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
        'reason_for_change': 'To tighten the limit for CLSR flagging',
        'cei_owners': [
            {'name': 'Owner A', 'site': 'CDDP', 'date': '04/02/2026'},
        ],
        'cei_note': 'Any owners responsible for both FWP and PWP stages.',
        'spec_rows': [
            {'spec_num': 'N/a', 'title': 'N/a'},
        ],
        'concern_rows': [
            {'number': '1', 'forum': 'Originator', 'issue': 'Limit too wide?',
             'resolution': 'Reviewed data, new limit approved by team.',
             'status': 'Closed'},
            {'number': '2', 'forum': 'Module WG', 'issue': 'Will this impact yield?',
             'resolution': 'No impact confirmed. Change targeted at flag reduction.',
             'status': 'Closed'},
            {'number': '3', 'forum': 'Originator', 'issue': 'Need CDDP sign-off?',
             'resolution': 'CDDP reviewed and approved.',
             'status': 'Closed'},
        ],
        'change_rows': [
            {
                'number': '1',
                'monitor_set': 'MON_SET_001',
                'measurement_set': 'MEAS_SET_001',
                'chart_type': 'CLSR',
                'limits': [
                    {'label': 'UCL',        'present': '3.50', 'proposed': '3.80'},
                    {'label': 'Centerline', 'present': '2.10', 'proposed': '2.20'},
                    {'label': 'LCL',        'present': '0.70', 'proposed': '0.60'},
                    {'label': 'CLSR',       'present': 'Flag', 'proposed': '',
                     'present_is_flag': True},
                ],
            },
            {
                'number': '1',
                'monitor_set': 'MON_SET_002',
                'measurement_set': 'MEAS_SET_002',
                'chart_type': 'CLSR',
                'limits': [
                    {'label': 'UCL',        'present': '4.00', 'proposed': '4.20'},
                    {'label': 'Centerline', 'present': '2.50', 'proposed': '2.60'},
                    {'label': 'LCL',        'present': '1.00', 'proposed': '1.00'},
                    {'label': 'CLSR',       'present': 'Flag', 'proposed': '',
                     'present_is_flag': True},
                ],
            },
            {
                'number': '1',
                'monitor_set': 'MON_SET_003',
                'measurement_set': 'MEAS_SET_003',
                'chart_type': 'CLSR',
                'limits': [
                    {'label': 'UCL',        'present': '2.90', 'proposed': '3.10'},
                    {'label': 'Centerline', 'present': '1.80', 'proposed': '1.90'},
                    {'label': 'LCL',        'present': '0.70', 'proposed': '0.70'},
                    {'label': 'CLSR',       'present': '',     'proposed': ''},
                ],
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
