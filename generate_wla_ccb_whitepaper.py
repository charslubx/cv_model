import io
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from lxml import etree as _etree

_W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'

# ---------------------------------------------------------------------------
# 颜色常量
# ---------------------------------------------------------------------------
BLUE = RGBColor(0x00, 0x00, 0xFF)
GREEN = RGBColor(0x00, 0x80, 0x00)
BLACK = RGBColor(0x00, 0x00, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

HEADER_BG = 'D3D3D3'  # 表头灰色背景


# ---------------------------------------------------------------------------
# 底层工具函数（与 document_builder.py 风格一致）
# ---------------------------------------------------------------------------

def _make_checkbox_sdt(checked: bool = False) -> '_etree._Element':
    CHECKED_CHAR = '&#x2612;'  # ☒ U+2612
    UNCHECKED_CHAR = '&#x2610;'  # ☐ U+2610
    checked_val = '1' if checked else '0'
    char = CHECKED_CHAR if checked else UNCHECKED_CHAR

    xml = f'''<w:sdt xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                     xmlns:w14="{_W14}">
  <w:sdtPr>
    <w14:checkbox>
      <w14:checked w14:val="{checked_val}"/>
      <w14:checkedState w14:val="2612" w14:font="MS Gothic"/>
      <w14:uncheckedState w14:val="2610" w14:font="MS Gothic"/>
    </w14:checkbox>
  </w:sdtPr>
  <w:sdtContent>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="MS Gothic" w:hAnsi="MS Gothic" w:cs="MS Gothic"/>
        <w:sz w:val="20"/>
        <w:szCs w:val="20"/>
      </w:rPr>
      <w:t>{char}</w:t>
    </w:r>
  </w:sdtContent>
</w:sdt>'''
    return _etree.fromstring(xml)


def _append_checkbox(para, checked: bool = False, label: str = '',
                     size_pt: int = 10, space_after: bool = True):
    para._p.append(_make_checkbox_sdt(checked))
    if label:
        r = para.add_run(('  ' if space_after else '') + label)
        _set_run_font(r, size_pt=size_pt)


def _cell_write_checkbox(cell, checked: bool, label: str, size_pt: int = 10,
                         valign: str = 'center'):
    _set_cell_valign(cell, valign)
    cell.text = ''
    para = cell.paragraphs[0]
    para._p.append(_make_checkbox_sdt(checked))
    if label:
        r = para.add_run('  ' + label)
        _set_run_font(r, size_pt=size_pt)
    return para


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


def _set_cell_text_direction(cell, direction='btLr'):
    """设置单元格文字方向。btLr=从下到上竖排，lrTb=正常横排"""
    tc_pr = cell._tc.get_or_add_tcPr()
    td = tc_pr.find(qn('w:textDirection'))
    if td is None:
        td = OxmlElement('w:textDirection')
        tc_pr.append(td)
    td.set(qn('w:val'), direction)


def _set_row_height(row, height_cm):
    tr_pr = row._tr.get_or_add_trPr()
    trH = OxmlElement('w:trHeight')
    trH.set(qn('w:val'), str(int(height_cm * 567)))
    trH.set(qn('w:hRule'), 'atLeast')
    tr_pr.append(trH)


def _set_table_total_width(tbl, width_inch):
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


def _cell_write(cell, text, align=0,
                font_name='Arial', size_pt=10,
                bold=False, color=None, italic=False, valign='top'):
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
    para = cell.add_paragraph()
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(0)
    run = para.add_run('' if text is None else str(text))
    _set_run_font(run, font_name=font_name, size_pt=size_pt,
                  bold=bold, color=color)
    return para


def _para_add_run(para, text, font_name='Arial', size_pt=10,
                  bold=False, color=None, italic=False, underline=False):
    run = para.add_run(text)
    _set_run_font(run, font_name=font_name, size_pt=size_pt,
                  bold=bold, color=color, italic=italic, underline=underline)
    return run


def _set_font_name(run, name='Arial', size_pt=None, color=None):
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
    p = doc.add_paragraph(style='Heading 2')
    p.paragraph_format.left_indent = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(-0.63)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.clear()
    _set_font_name(p.add_run(f'{letter}  '), size_pt=10, color=BLACK)
    _set_font_name(p.add_run(label), size_pt=10, color=BLACK)
    if value:
        r_val = p.add_run(value)
        _set_font_name(r_val, size_pt=10, color=BLUE if blue else BLACK)
    return p


# ---------------------------------------------------------------------------
# 章节构建函数
# ---------------------------------------------------------------------------

def _build_title(doc):
    p = doc.add_paragraph()
    p.alignment = 1
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run('WLA CCB Monitor Change White Paper')
    _set_run_font(r, size_pt=16, bold=True, underline=True)


def _build_section1(doc, data, page_w_cm):
    _add_heading(doc, '1) Phase, Classification, Related WPs:')

    total_twip = int(page_w_cm / 2.54 * 1440)
    col0_twip = int(total_twip * 0.38)
    col1_twip = total_twip - col0_twip
    col0_w = Cm(col0_twip / 567)
    col1_w = Cm(col1_twip / 567)

    tbl = doc.add_table(rows=5, cols=2)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    _set_row_height(tbl.rows[0], 0.6)
    merged0 = tbl.rows[0].cells[0].merge(tbl.rows[0].cells[1])
    merged0.width = Cm(page_w_cm / 2.54)
    _set_cell_valign(merged0, 'center')
    merged0.text = ''
    p1 = merged0.paragraphs[0]
    _para_add_run(p1, 'Phase:', bold=True)
    _para_add_run(p1, '    ')
    _append_checkbox(p1, checked=False, label='PWP')
    _para_add_run(p1, '    ')
    _append_checkbox(p1, checked=True, label='FWP')
    _set_cell_shading(merged0, 'FFFFFF')

    _set_row_height(tbl.rows[1], 0.6)
    merged1 = tbl.rows[1].cells[0].merge(tbl.rows[1].cells[1])
    merged1.width = Cm(page_w_cm / 2.54)
    _set_cell_valign(merged1, 'center')
    merged1.text = ''
    p = merged1.paragraphs[0]
    _para_add_run(p, 'For a FWP, document the PWP Horizon number (if applicable): ')
    _para_add_run(p, data.get('fwp_horizon', 'N/a'), bold=True, color=BLUE)
    _set_cell_shading(merged1, 'FFFFFF')

    _set_row_height(tbl.rows[2], 0.6)
    merged2 = tbl.rows[2].cells[0].merge(tbl.rows[2].cells[1])
    merged2.width = Cm(page_w_cm / 2.54)
    _set_cell_valign(merged2, 'center')
    merged2.text = ''
    p2 = merged2.paragraphs[0]
    _para_add_run(p2, 'Classification:', bold=True)
    _para_add_run(p2, '    ')
    for label, is_checked in [('1', False), ('2', False), ('3', False),
                               ('3N', False), ('4', True)]:
        _append_checkbox(p2, checked=is_checked, label=label)
        _para_add_run(p2, '   ')
    _set_cell_shading(merged2, 'FFFFFF')

    _set_row_height(tbl.rows[3], 0.6)
    merged3 = tbl.rows[3].cells[0].merge(tbl.rows[3].cells[1])
    merged3.width = Cm(page_w_cm / 2.54)
    _set_cell_valign(merged3, 'center')
    merged3.text = ''
    p = merged3.paragraphs[0]
    _para_add_run(p, 'For Class IV WPs, add name of PCCB member confirming classification: ')
    _para_add_run(p, data.get('pccb_member', 'N/A'), bold=True, color=BLUE)
    _set_cell_shading(merged3, 'FFFFFF')

    _set_row_height(tbl.rows[4], 0.7)
    merged4 = tbl.rows[4].cells[0].merge(tbl.rows[4].cells[1])
    merged4.width = Cm(page_w_cm / 2.54)
    _set_cell_valign(merged4, 'center')
    _cell_write(
        merged4,
        'Include any relevant reference white paper(s), "Me-Too" WPs, DRB, MRB, etc. in table below',
        bold=True, valign='center',
    )
    _set_cell_shading(merged4, 'FFFFFF')

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
                italic=True, align=1, valign='center')

    _set_row_height(tbl_ref.rows[1], 0.55)
    ref_rows = data.get('reference_wps', [{'horizon': 'N/a', 'title': 'N/a'}])
    for col_i, key in enumerate(['horizon', 'title']):
        val = ref_rows[0].get(key, 'N/a') if ref_rows else 'N/a'
        _cell_write(tbl_ref.rows[1].cells[col_i], val, valign='center')
        _set_cell_shading(tbl_ref.rows[1].cells[col_i], 'FFFFFF')


def _build_section2(doc, data):
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.clear()
    _set_font_name(p.add_run('2) Date: '), size_pt=12, color=BLACK)
    _set_font_name(p.add_run(data.get('date', '04/02/2026')), size_pt=12, color=BLUE)


def _build_section3(doc, data):
    _add_heading(doc, '3) Authorship', space_before=10)

    items = [
        ('a.', 'Primary author: ', data.get('primary_author', 'Yuan, Ji'), True),
        ('b.', 'Site (primary author only): ', data.get('site', 'CDDP'), True),
        ('c.', 'Co-author(s):', data.get('co_authors', ''), False),
    ]
    for letter, label, value, blue in items:
        _add_sub_item(doc, letter, label, value=value, blue=blue)


def _build_section4(doc, data):
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.clear()
    _set_font_name(p.add_run('4) Title of Change: '), size_pt=12, color=BLACK)
    _set_font_name(p.add_run(data.get('title_of_change', 'CD DGB chart limit change for CLSR flag')),
                   size_pt=12, color=BLUE)


def _build_section5_header(doc, data):
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
    RED = RGBColor(0xFF, 0x00, 0x00)

    n = len(limit_rows)
    if n == 0:
        return

    label_twip = int(cell_twip * 0.45)
    colon_twip = int(cell_twip * 0.10)
    value_twip = cell_twip - label_twip - colon_twip

    nested = doc.add_table(rows=n, cols=3)
    nested.style = 'Table Grid'
    nested.autofit = False
    _set_table_total_width(nested, cell_twip / 1440)
    _set_table_no_borders(nested)

    for i, row_data in enumerate(limit_rows):
        nr = nested.rows[i]

        c0 = nr.cells[0]
        c0.width = Cm(label_twip / 567)
        _set_cell_no_padding(c0)
        _cell_write(c0, row_data.get('label', ''), size_pt=10, valign='center')

        c1 = nr.cells[1]
        c1.width = Cm(colon_twip / 567)
        _set_cell_no_padding(c1)
        _cell_write(c1, ':', size_pt=10, align=1, valign='center')

        c2 = nr.cells[2]
        c2.width = Cm(value_twip / 567)
        _set_cell_no_padding(c2)
        _set_cell_valign(c2, 'center')
        c2.text = ''
        p = c2.paragraphs[0]

        val = row_data.get('value', '')
        vclr = row_data.get('value_color', BLUE if val else None)
        flag = row_data.get('flag', '')
        fclr = row_data.get('flag_color', None)

        if val:
            r_val = p.add_run(val)
            _set_run_font(r_val, size_pt=10, bold=True, color=vclr)
        if flag:
            if val:
                r_sp = p.add_run(' ')
                _set_run_font(r_sp, size_pt=10)
            r_flag = p.add_run(flag)
            fc = fclr if fclr is not None else (vclr if val else GREEN)
            _set_run_font(r_flag, size_pt=10, bold=True, color=fc)

    nested_tbl_el = nested._tbl
    nested_tbl_el.getparent().remove(nested_tbl_el)

    tc = cell._tc
    last_p = tc.findall(qn('w:p'))[-1]
    tc.insert(list(tc).index(last_p), nested_tbl_el)


def _limits_from_rec(rec, prefix):
    RED = RGBColor(0xFF, 0x00, 0x00)

    if 'limits' in rec:
        rows = []
        for item in rec['limits']:
            val = item.get(prefix, '')
            flag = item.get(f'{prefix}_flag', '')
            fc_str = item.get(f'{prefix}_flag_color', 'green')
            flag_clr = RED if fc_str == 'red' else GREEN
            rows.append({
                'label': item.get('label', ''),
                'value': val,
                'value_color': BLUE if val else None,
                'flag': flag,
                'flag_color': flag_clr if flag else None,
            })
        return rows

    default_limits = [
        ('UCL', f'{prefix}_ucl', ''),
        ('Centerline', f'{prefix}_cl', ''),
        ('LCL', f'{prefix}_lcl', ''),
        ('CLSR', f'{prefix}_clsr', f'{prefix}_clsr_flag'),
    ]
    rows = []
    for label, val_key, flag_key in default_limits:
        val = rec.get(val_key, '')
        flag = rec.get(flag_key, '') if flag_key else ''
        rows.append({
            'label': label,
            'value': val,
            'value_color': BLUE if val else None,
            'flag': flag,
            'flag_color': GREEN if flag else None,
        })
    return rows


def _build_change_table(doc, data, page_w_cm):
    total_twip = int(page_w_cm / 2.54 * 1440)
    num_twip = int(0.4 * 1440)
    present_twip = int(2.2 * 1440)
    proposed_twip = int(2.2 * 1440)
    items_twip = total_twip - num_twip - present_twip - proposed_twip
    col_twips = [num_twip, items_twip, present_twip, proposed_twip]

    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    _set_row_height(tbl.rows[0], 0.6)
    headers = ['#', 'Change items', 'Present value', 'Proposed value']
    for j, (txt, tw) in enumerate(zip(headers, col_twips)):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, txt, align=1, bold=True, valign='center')
        _set_cell_shading(c, HEADER_BG)

    change_rows = data.get('change_rows', [])
    for rec in change_rows:
        row = tbl.add_row()
        for j, tw in enumerate(col_twips):
            row.cells[j].width = Cm(tw / 567)

        _cell_write(row.cells[0], rec.get('number', '1'), align=1, valign='top')

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

        for col_idx, prefix in [(2, 'present'), (3, 'proposed')]:
            cv = row.cells[col_idx]
            _set_cell_valign(cv, 'top')
            cv.text = ''
            limit_rows = _limits_from_rec(rec, prefix)
            _build_inner_value_table(doc, cv, limit_rows, col_twips[col_idx])

    return tbl


# ---------------------------------------------------------------------------
# 文档组装
# ---------------------------------------------------------------------------

def _build_page2(doc, data, page_w_cm):
    _build_section6(doc, data)
    _build_section7(doc, data, page_w_cm)
    _build_section8(doc, data, page_w_cm)
    _build_section9(doc, data, page_w_cm)


def _build_section5_fwp_table(doc, data, page_w_cm):
    p = doc.add_paragraph(style='Heading 2')
    p.paragraph_format.left_indent = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(-0.63)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.clear()
    _set_font_name(p.add_run('a.  '), size_pt=10, color=BLACK)
    r_fwp = p.add_run('(FWP only)*')
    _set_font_name(r_fwp, size_pt=10, color=BLACK)
    desc = data.get(
        'fwp_only_desc',
        ' Incorporate the C-Spec (or equivalent) into the PWP Control Plan for '
        'future PWP versions. Otherwise, leave this blank.'
    )
    _set_font_name(p.add_run(desc), size_pt=10, color=BLACK)

    total_twip = int(page_w_cm / 2.54 * 1440)
    num_twip = int(0.4 * 1440)
    val_twip = int(2.2 * 1440)
    items_twip = total_twip - num_twip - val_twip * 2

    col_twips = [num_twip, items_twip, val_twip, val_twip]
    headers = ['#', 'Change items', 'Previous value', 'Current value']

    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    _set_row_height(tbl.rows[0], 0.6)
    for j, (txt, tw) in enumerate(zip(headers, col_twips)):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, txt, bold=True, size_pt=10, align=1, valign='center')
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

    if not fwp_rows:
        for _ in range(2):
            row = tbl.add_row()
            _set_row_height(row, 0.6)
            for j, tw in enumerate(col_twips):
                row.cells[j].width = Cm(tw / 567)


def _build_section6(doc, data):
    _add_heading(doc, '6) Reason for Change:')
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    _para_add_run(p, data.get('reason_for_change', ''), color=BLUE, size_pt=11)


def _build_section7(doc, data, page_w_cm):
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.clear()
    _set_font_name(p.add_run('7) '), size_pt=12, color=BLACK)
    r_cei = p.add_run('CE!')
    _set_font_name(r_cei, size_pt=12, color=BLACK)
    _set_font_name(p.add_run('/Site Implementation Owners:'), size_pt=12, color=BLACK)

    total_twip = int(page_w_cm / 2.54 * 1440)
    col_twips = [int(total_twip * 0.38), int(total_twip * 0.12),
                 total_twip - int(total_twip * 0.38) - int(total_twip * 0.12)]

    tbl = doc.add_table(rows=2, cols=3)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    _set_row_height(tbl.rows[0], 0.6)
    headers = ['CE!Owners:', 'SITE', 'Date reviewed and approved:']
    for j, (txt, tw) in enumerate(zip(headers, col_twips)):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, txt, bold=True, size_pt=10, valign='center')
        _set_cell_shading(c, 'F2F2F2')

    _set_row_height(tbl.rows[1], 0.7)
    owners = data.get('cei_owners', [{'name': '', 'site': 'CDDP', 'date': ''}])
    owner = owners[0] if owners else {}
    for j, (key, tw) in enumerate(zip(['name', 'site', 'date'], col_twips)):
        c = tbl.rows[1].cells[j]
        c.width = Cm(tw / 567)
        val = owner.get(key, '')
        color = BLUE if key in ('name', 'date') and val else None
        _cell_write(c, val, size_pt=10, color=color, valign='center')
        _set_cell_shading(c, 'FFFFFF')

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

    note = data.get('cei_note', 'Any owners responsible for both FWP and PWP stages.')
    bp = doc.add_paragraph(style='List Bullet')
    bp.paragraph_format.space_before = Pt(4)
    bp.paragraph_format.space_after = Pt(2)
    _para_add_run(bp, note, size_pt=10)


def _build_section8(doc, data, page_w_cm):
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    p.clear()
    _set_font_name(p.add_run('8) Specifications, Controlled Documents Affected:'), size_pt=12, color=BLACK)
    r_note = p.add_run(' (list all affected by changes above)')
    _set_font_name(r_note, size_pt=10, color=BLACK)
    r_note.font.italic = True
    r_note.font.bold = False

    total_twip = int(page_w_cm / 2.54 * 1440)
    col0_twip = int(total_twip * 0.45)
    col1_twip = total_twip - col0_twip

    tbl = doc.add_table(rows=1, cols=2)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    _set_row_height(tbl.rows[0], 0.6)
    for j, (txt, tw) in enumerate(
            zip(['Spec and/or Controlled Document #', 'Document Title'],
                [col0_twip, col1_twip])):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, txt, bold=True, size_pt=10, align=1, valign='center')
        _set_cell_shading(c, HEADER_BG)

    spec_rows = data.get('spec_rows', [{'spec_num': 'N/a', 'title': 'N/a'}])
    for rec in spec_rows:
        row = tbl.add_row()
        _set_row_height(row, 0.6)
        row.cells[0].width = Cm(col0_twip / 567)
        row.cells[1].width = Cm(col1_twip / 567)
        _cell_write(row.cells[0], rec.get('spec_num', ''), size_pt=10, valign='center')
        _cell_write(row.cells[1], rec.get('title', ''), size_pt=10, valign='center')
        _set_cell_shading(row.cells[0], 'FFFFFF')
        _set_cell_shading(row.cells[1], 'FFFFFF')


def _build_section9(doc, data, page_w_cm):
    _add_heading(doc, '9) Concerns and Considerations:')

    total_twip = int(page_w_cm / 2.54 * 1440)
    col_ratios = [0.05, 0.15, 0.20, 0.45, 0.15]
    col_twips = [int(total_twip * r) for r in col_ratios]
    col_twips[-1] = total_twip - sum(col_twips[:-1])

    headers = ['#', 'Forum\nidentifying\nconcern', 'Issue', 'Resolution', 'Status:\n(Open or\nClosed)']

    tbl = doc.add_table(rows=1, cols=5)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    _set_row_height(tbl.rows[0], 1.2)
    for j, (txt, tw) in enumerate(zip(headers, col_twips)):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, txt, bold=True, size_pt=10, align=1, valign='center')
        _set_cell_shading(c, HEADER_BG)

    concern_rows = data.get('concern_rows', [])
    for rec in concern_rows:
        row = tbl.add_row()
        _set_row_height(row, 1.5)
        for j, tw in enumerate(col_twips):
            row.cells[j].width = Cm(tw / 567)

        keys = ['number', 'forum', 'issue', 'resolution', 'status']
        blue_cols = {0, 1, 4}
        for j, key in enumerate(keys):
            val = rec.get(key, '')
            _cell_write(row.cells[j], val, size_pt=10, color=BLUE, valign='top')

    if not concern_rows:
        for _ in range(2):
            row = tbl.add_row()
            _set_row_height(row, 0.8)
            for j, tw in enumerate(col_twips):
                row.cells[j].width = Cm(tw / 567)


def _build_section10(doc, data, page_w_cm):
    note = data.get('concerns_note', 'Any owners responsible for both FWP and PWP stages.')
    bp = doc.add_paragraph(style='List Bullet')
    bp.paragraph_format.space_before = Pt(0)
    bp.paragraph_format.space_after = Pt(2)
    _para_add_run(bp, note, size_pt=10)

    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.clear()
    _set_font_name(p.add_run(
        '10) Control Chart Setup: Only fill in section for SPC++ changes or SPC# chart creation'
    ), size_pt=12, color=BLACK)

    _add_sub_item(doc, 'a.', 'Chart Modification: (check one)')

    mod_sel = data.get('chart_modification', 'revision')
    mod_options = [
        ('revision', 'Chart Revision'),
        ('new', 'New Chart Creation. Please\ncomplete section 10b.'),
        ('deletion', 'Chart Deletion'),
    ]
    tbl_a = doc.add_table(rows=len(mod_options), cols=2)
    tbl_a.autofit = False
    _set_table_no_borders(tbl_a)
    _set_table_indent(tbl_a, 2.54)
    checkbox_tw = int(0.3 * 1440)
    label_tw = int(2.5 * 1440)
    _set_table_total_width(tbl_a, (checkbox_tw + label_tw) / 1440)
    for i, (key, label) in enumerate(mod_options):
        row = tbl_a.rows[i]
        c0 = row.cells[0]
        c0.width = Cm(checkbox_tw / 567)
        _set_cell_no_padding(c0)
        _cell_write_checkbox(c0, checked=(mod_sel == key), label='', valign='center')
        c1 = row.cells[1]
        c1.width = Cm(label_tw / 567)
        _set_cell_no_padding(c1)
        _cell_write(c1, label, size_pt=10, valign='center')

    _add_sub_item(doc, 'b.', 'Chart By: (check one and fill out table below, only needed for New Chart Creation)')

    chart_by = data.get('chart_by', 'all_categories')
    cb_options = [
        [('equipment', 'Equipment'), ('all_categories', 'All Categories')],
        [('monitor', 'Monitor'), ('process', 'Process')],
        [('operation', 'Operation'), ('custom_context', 'Custom Context Categories')],
        [('product', 'Product'), (None, '')],
    ]
    cb_tw = int(0.3 * 1440)
    cb_label_tw = int(2.2 * 1440)
    tbl_b = doc.add_table(rows=len(cb_options), cols=4)
    tbl_b.autofit = False
    _set_table_no_borders(tbl_b)
    _set_table_indent(tbl_b, 2.54)
    _set_table_total_width(tbl_b, (cb_tw + cb_label_tw) * 2 / 1440)
    for i, row_opts in enumerate(cb_options):
        row = tbl_b.rows[i]
        for col_pair, (key, label) in enumerate(row_opts):
            c_chk = row.cells[col_pair * 2]
            c_chk.width = Cm(cb_tw / 567)
            _set_cell_no_padding(c_chk)
            if label:
                _cell_write_checkbox(c_chk, checked=(chart_by == key),
                                     label='', valign='center')
            c_lbl = row.cells[col_pair * 2 + 1]
            c_lbl.width = Cm(cb_label_tw / 567)
            _set_cell_no_padding(c_lbl)
            _cell_write(c_lbl, label or '', size_pt=10, valign='center')

    # 15列 SPC 表格，第5-14列（索引4-13）表头竖向排列
    _build_spc15_table(doc, data.get('spc_setup_rows', []), page_w_cm, prefix='setup')

    for note in data.get('spc_setup_notes', [
        'Oper: refer to SPC++ documentation.',
        'Class: refer to SPC++ documentation.',
        'Calc Method: (raw, percentage). For SPC#, Eng review required.',
    ]):
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_before = Pt(2)
        bp.paragraph_format.space_after = Pt(2)
        _para_add_run(bp, note, size_pt=10)

    _add_sub_item(doc, 'c.', 'SPC Rules: (check one and fill out table below)')

    rules_sel = data.get('spc_rules', 'no_changes')
    rules_options = [
        ('no_rules', 'No Rules Set'),
        ('no_changes', 'No Changes Proposed'),
    ]
    tbl_c = doc.add_table(rows=len(rules_options), cols=2)
    tbl_c.autofit = False
    _set_table_no_borders(tbl_c)
    _set_table_indent(tbl_c, 2.54)
    _set_table_total_width(tbl_c, (checkbox_tw + label_tw) / 1440)
    for i, (key, label) in enumerate(rules_options):
        row = tbl_c.rows[i]
        c0 = row.cells[0]
        c0.width = Cm(checkbox_tw / 567)
        _set_cell_no_padding(c0)
        _cell_write_checkbox(c0, checked=(rules_sel == key), label='', valign='center')
        c1 = row.cells[1]
        c1.width = Cm(label_tw / 567)
        _set_cell_no_padding(c1)
        _cell_write(c1, label, size_pt=10, valign='center')

    p_note = doc.add_paragraph()
    p_note.paragraph_format.space_before = Pt(4)
    p_note.paragraph_format.space_after = Pt(4)
    _para_add_run(p_note, data.get('spc_rules_note',
                                   'Complete the following if there are changes in rules, note present rules.'),
                  size_pt=10)

    # c 的 SPC 表格，同样第5-14列表头竖向
    _build_spc15_table(doc, data.get('spc_rules_rows', []), page_w_cm, prefix='rules')

    _build_rules_legend(doc, page_w_cm)

    p_custom = doc.add_paragraph()
    p_custom.paragraph_format.space_before = Pt(4)
    p_custom.paragraph_format.space_after = Pt(4)
    _para_add_run(p_custom,
                  'If you have more custom rules that do not fit the standard rule codes above, '
                  'provide details here.', size_pt=10)

    _add_sub_item(doc, 'd.', 'Control Chart Data Summary:')

    p_d = doc.add_paragraph()
    p_d.paragraph_format.space_before = Pt(2)
    p_d.paragraph_format.space_after = Pt(4)
    _para_add_run(p_d, data.get('data_summary_note',
                                'Select type of limits, and explain assumptions. Expect ☒ for CEI monitor sets, '
                                'with VF-Common limits, except where required for local calibration wafer sets, '
                                'or where both matching data and Fab-specific deviation.'),
                  size_pt=10)

    ds_sel = data.get('data_summary_type', 'vf_common')
    ds_options = [
        ('tool_specific', 'Tool-Specific'),
        ('fab_specific',
         'Fab-Specific  (First time CEI deviation need to show justification of why fabs are different)'),
        ('vf_common', 'VF-Common'),
    ]
    tbl_d = doc.add_table(rows=len(ds_options), cols=2)
    tbl_d.autofit = False
    _set_table_no_borders(tbl_d)
    _set_table_indent(tbl_d, 2.54)
    ds_cb_tw = int(0.3 * 1440)
    ds_label_tw = int(4.5 * 1440)
    _set_table_total_width(tbl_d, (ds_cb_tw + ds_label_tw) / 1440)
    for i, (key, label) in enumerate(ds_options):
        row = tbl_d.rows[i]
        c0 = row.cells[0]
        c0.width = Cm(ds_cb_tw / 567)
        _set_cell_no_padding(c0)
        _cell_write_checkbox(c0, checked=(ds_sel == key), label='', valign='center')
        c1 = row.cells[1]
        c1.width = Cm(ds_label_tw / 567)
        _set_cell_no_padding(c1)
        _cell_write(c1, label, size_pt=10, valign='center')


def _build_section11(doc, data, page_w_cm):
    _add_heading(doc, '11) Specific Checklists:')

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    _para_add_run(p, data.get('checklist_note',
                              'Attach all appropriate checklists. Applicable checklist and worksheet to be obtained from '
                              '"Handbook and Templates".'),
                  size_pt=10)

    total_twip = int(page_w_cm / 2.54 * 1440)
    col0_tw = int(total_twip * 0.38)
    col1_tw = int(total_twip * 0.32)
    col2_tw = total_twip - col0_tw - col1_tw

    checklist_rows = data.get('checklist_rows', [
        {'item': 'APC Add/Change Checklist', 'doc': 'N/A', 'comments': ''},
        {'item': 'Software/Firmware Change Checklist', 'doc': 'N/A', 'comments': ''},
        {'item': 'Monitor Sampling Reduction Worksheet', 'doc': 'N/A', 'comments': ''},
        {'item': 'Other:', 'doc': 'N/A', 'comments': ''},
    ])

    tbl = doc.add_table(rows=1 + len(checklist_rows), cols=3)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    for j, (txt, tw) in enumerate(zip(
            ['Item', 'Embedded Document(s)', 'Comments'],
            [col0_tw, col1_tw, col2_tw])):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _cell_write(c, txt, bold=True, size_pt=10, align=1, valign='center')
        _set_cell_shading(c, HEADER_BG)

    for i, rec in enumerate(checklist_rows):
        row = tbl.rows[i + 1]
        for j, (key, tw) in enumerate(zip(
                ['item', 'doc', 'comments'],
                [col0_tw, col1_tw, col2_tw])):
            c = row.cells[j]
            c.width = Cm(tw / 567)
            val = rec.get(key, '')
            color = BLUE if key == 'doc' and val and val != '' else None
            _cell_write(c, val, size_pt=10, color=color,
                        align=1 if key == 'doc' else 0,
                        valign='center')
            _set_cell_shading(c, 'FFFFFF')


def _build_section12(doc, data):
    _add_heading(doc, '12) Data Details')

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    _para_add_run(p, data.get('data_details_note',
                              '(Include and clearly label all tables, key graphs and data summaries that support the intended '
                              'change, additionally summarize and document, including data as an attachment is strongly preferred.)'),
                  size_pt=10, italic=True)

    for item in data.get('data_details_items', []):
        p_item = doc.add_paragraph()
        p_item.paragraph_format.space_before = Pt(4)
        p_item.paragraph_format.space_after = Pt(2)
        _para_add_run(p_item, item.get('title', ''), bold=True, color=BLUE, size_pt=10)
        if item.get('description'):
            p_desc = doc.add_paragraph()
            p_desc.paragraph_format.space_before = Pt(0)
            p_desc.paragraph_format.space_after = Pt(4)
            _para_add_run(p_desc, item['description'], size_pt=10)


def _build_spc15_table(doc, data_rows, page_w_cm, prefix='setup'):
    """
    SPC 15列表格。前4列（Oper/SPC_AREA/MONITOR/MEASUREMENT）横排，
    第5-14列（CHART_SUBSET~UBL，索引4-13）表头竖向排列，第15列（Calc Method）横排。
    """
    total_twip = int(page_w_cm / 2.54 * 1440)

    col_defs = [
        ('Oper',                    0.045),
        ('SPC_\nFUNCTIONAL\n_AREA', 0.085),
        ('MONITOR_\nSET_NAME',      0.090),
        ('MEASUREMENT\n_SET_NAME',  0.095),
        ('CHART\nSUBSET/\nNUMBER',  0.060),  # 索引4，开始竖排
        ('TYPE',                    0.050),
        ('LCL',                     0.055),
        ('CL',                      0.055),
        ('TARGET',                  0.060),
        ('LDL',                     0.050),
        ('UDL',                     0.050),
        ('LUL',                     0.050),
        ('UBL',                     0.050),  # 索引12，竖排结束
        ('Class',                   0.060),
        ('Calc\nMethod',            0.095),
    ]
    # 竖排列索引范围：4~12（含）
    VERTICAL_HDR_START = 4
    VERTICAL_HDR_END   = 12   # inclusive

    ratio_sum = sum(r for _, r in col_defs)
    col_twips = [int(total_twip * r / ratio_sum) for _, r in col_defs]
    col_twips[-1] = total_twip - sum(col_twips[:-1])

    col_headers = [h for h, _ in col_defs]

    tbl = doc.add_table(rows=1, cols=len(col_defs))
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)

    # 竖排表头行需要固定一个较高的高度，让竖向文字能显示
    _set_row_height(tbl.rows[0], 1.5)

    for j, (hdr, tw) in enumerate(zip(col_headers, col_twips)):
        c = tbl.rows[0].cells[j]
        c.width = Cm(tw / 567)
        _set_cell_shading(c, HEADER_BG)

        if VERTICAL_HDR_START <= j <= VERTICAL_HDR_END:
            # 竖向：去掉 \n，改用无换行的单一字符串，设置文字方向 btLr
            hdr_clean = hdr.replace('\n', '')
            _cell_write(c, hdr_clean, bold=True, size_pt=7,
                        align=1, valign='center')
            _set_cell_text_direction(c, 'tbRl')
        else:
            _cell_write(c, hdr, bold=True, size_pt=7,
                        align=1, valign='center')

    rows_to_fill = data_rows if data_rows else [{} for _ in range(3)]
    keys = ['oper', 'spc_area', 'monitor_set', 'measurement_set',
            'subset_num', 'type', 'lcl', 'cl', 'target',
            'ldl', 'udl', 'lul', 'ubl', 'class_', 'calc_method']
    for rec in rows_to_fill:
        row = tbl.add_row()
        for j, (key, tw) in enumerate(zip(keys, col_twips)):
            c = row.cells[j]
            c.width = Cm(tw / 567)
            _cell_write(c, rec.get(key, ''), size_pt=7, valign='center')
            _set_cell_shading(c, 'FFFFFF')

    return tbl


def _build_rules_legend(doc, page_w_cm):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run('*Rules are denoted as follows:')
    _set_run_font(r, size_pt=9, bold=False)

    legend = [
        ('A', '> UCL'),
        ('B', '2/1 > 2 SIGMA'),
        ('C', '4/5 > 1 SIGMA'),
        ('D', 'Last 8 > CL'),
        ('E', '< LCL'),
        ('F', '2/3 < -2 SIGMA'),
        ('G', '4/5 < -1 SIGMA'),
        ('H', 'Last 8 < CL'),
        ('I', '15 Inside SIGMA'),
        ('J', '8 Outside SIGMA'),
        ('K', '> UDL'),
        ('L', '< LDL'),
        ('M', 'Missing Data'),
        ('N', 'Fail Disposition'),
        ('', 'No Limits'),
        ('NONE', 'No OOC Rules'),
    ]

    cols_per_row = 4
    total_twip = int(page_w_cm / 2.54 * 1440)
    code_twip = int(total_twip / cols_per_row * 0.25)
    desc_twip = int(total_twip / cols_per_row * 0.75)
    n_rows = (len(legend) + cols_per_row - 1) // cols_per_row

    tbl = doc.add_table(rows=n_rows, cols=cols_per_row * 2)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, page_w_cm / 2.54)
    _set_table_no_borders(tbl)

    for i, (code, desc) in enumerate(legend):
        ri = i // cols_per_row
        ci = (i % cols_per_row) * 2
        c_code = tbl.rows[ri].cells[ci]
        c_code.width = Cm(code_twip / 567)
        _cell_write(c_code, code, size_pt=8, bold=True, valign='center')
        c_desc = tbl.rows[ri].cells[ci + 1]
        c_desc.width = Cm(desc_twip / 567)
        _cell_write(c_desc, desc, size_pt=8, valign='center')


def _add_page_num_field(para):
    run = para.add_run()
    run.font.size = Pt(9)
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    for attr in ('w:ascii', 'w:hAnsi', 'w:cs'):
        rFonts.set(qn(attr), 'Arial')

    fc_begin = OxmlElement('w:fldChar')
    fc_begin.set(qn('w:fldCharType'), 'begin')
    run._r.append(fc_begin)

    run2 = para.add_run()
    run2.font.size = Pt(9)
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = ' PAGE '
    run2._r.append(instr)

    run3 = para.add_run()
    run3.font.size = Pt(9)
    fc_sep = OxmlElement('w:fldChar')
    fc_sep.set(qn('w:fldCharType'), 'separate')
    run3._r.append(fc_sep)

    run4 = para.add_run()
    run4.font.size = Pt(9)
    fc_end = OxmlElement('w:fldChar')
    fc_end.set(qn('w:fldCharType'), 'end')
    run4._r.append(fc_end)


def _build_footer(doc, data):
    sec = doc.sections[0]
    footer = sec.footer
    footer.is_linked_to_previous = False

    for p in footer.paragraphs:
        p.clear()

    fp = footer.paragraphs[0]
    fp.paragraph_format.space_before = Pt(0)
    fp.paragraph_format.space_after = Pt(0)

    left_text = data.get('footer_left', 'Intel Confidential')
    center_text = data.get('footer_center', 'WLA CCB Monitor Change White Paper')

    from docx.oxml import OxmlElement as _el
    pPr = fp._p.get_or_add_pPr()
    tabs = _el('w:tabs')

    tab_center = _el('w:tab')
    tab_center.set(qn('w:val'), 'center')
    tab_center.set(qn('w:pos'), '4680')

    tab_right = _el('w:tab')
    tab_right.set(qn('w:val'), 'right')
    tab_right.set(qn('w:pos'), '9360')

    tabs.append(tab_center)
    tabs.append(tab_right)
    pPr.append(tabs)

    _para_add_run(fp, left_text, size_pt=9)
    _para_add_run(fp, '\t', size_pt=9)
    _para_add_run(fp, center_text, size_pt=9)
    _para_add_run(fp, '\t', size_pt=9)
    _para_add_run(fp, 'Page ', size_pt=9)
    _add_page_num_field(fp)


def _apply_doc_settings(doc, page_w_cm=21.59, page_h_cm=27.94):
    sec = doc.sections[0]
    sec.page_width = Cm(page_w_cm)
    sec.page_height = Cm(page_h_cm)
    sec.left_margin = Cm(2.54)
    sec.right_margin = Cm(2.54)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)


def build_wla_ccb_document(data: dict) -> bytes:
    """
    生成 WLA CCB Monitor Change White Paper 文档。
    返回 bytes，可直接作为 HTTP 响应体或写入 .docx 文件。
    """
    doc = Document()
    _apply_doc_settings(doc)

    page_w_cm = 21.59 - 2.54 * 2  # ≈ 16.51 cm

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
    _build_section11(doc, data, page_w_cm)
    _build_section12(doc, data)
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
        'footer_left': 'Intel Confidential',
        'footer_center': 'WLA CCB Monitor Change White Paper',
        'footer_right': 'Rev 1.0',
        'reason_for_change': 'To tighten the limit for CLSR flagging',
        'cei_owners': [
            {'name': 'Owner A', 'site': 'CDDP', 'date': '04/02/2026'},
        ],
        'cei_note': 'Any owners responsible for both FWP and PWP stages.',
        'concerns_note': 'Any owners responsible for both FWP and PWP stages.',
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
                    {'label': 'UCL', 'present': '493', 'proposed': '488.4'},
                    {'label': 'Centerline', 'present': '487', 'proposed': '487'},
                    {'label': 'LCL', 'present': '481', 'proposed': '485.6'},
                    {'label': 'CLSR',
                     'present': '17.4', 'present_flag': 'Flag', 'present_flag_color': 'red',
                     'proposed': '4', 'proposed_flag': ''},
                ],
            },
            {
                'number': '1',
                'monitor_set': 'MON_SET_002',
                'measurement_set': 'MEAS_SET_002',
                'chart_type': 'CLSR',
                'limits': [
                    {'label': 'UCL', 'present': '350.25', 'proposed': '352.0'},
                    {'label': 'Centerline', 'present': '348.10', 'proposed': '348.1'},
                    {'label': 'LCL', 'present': '345.95', 'proposed': '344.2'},
                    {'label': 'CLSR',
                     'present': '8.2', 'present_flag': 'Flag', 'present_flag_color': 'green',
                     'proposed': '2', 'proposed_flag': ''},
                ],
            },
            {
                'number': '1',
                'monitor_set': 'MON_SET_003',
                'measurement_set': 'MEAS_SET_003',
                'chart_type': 'CLSR',
                'limits': [
                    {'label': 'UCL', 'present': '2.90', 'proposed': '3.10'},
                    {'label': 'Centerline', 'present': '1.80', 'proposed': '1.90'},
                    {'label': 'LCL', 'present': '0.70', 'proposed': '0.70'},
                    {'label': 'CLSR', 'present': '1.2', 'proposed': '0.8'},
                ],
            },
        ],
        'data_summary_type': 'vf_common',
        'checklist_rows': [
            {'item': 'APC Add/Change Checklist', 'doc': 'N/A', 'comments': ''},
            {'item': 'Software/Firmware Change Checklist', 'doc': 'N/A', 'comments': ''},
            {'item': 'Monitor Sampling Reduction Worksheet', 'doc': 'N/A', 'comments': ''},
            {'item': 'Other:', 'doc': 'N/A', 'comments': ''},
        ],
        'data_details_items': [
            {'title': '1. X-bar Control Limit Summary for "Value" (Statistical)', 'description': ''},
            {'title': '2. X-bar_CG', 'description': ''},
            {'title': '3. X-bar_LG', 'description': ''},
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
