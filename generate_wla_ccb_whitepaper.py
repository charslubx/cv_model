"""
生成 WLA CCB Monitor Change White Paper docx 文档
按照图片中的格式生成对应的 Word 文档
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy


def set_cell_border(cell, top=None, bottom=None, left=None, right=None):
    """设置单元格边框"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for border_name, border_val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        if border_val is not None:
            border_el = OxmlElement(f'w:{border_name}')
            border_el.set(qn('w:val'), border_val.get('val', 'single'))
            border_el.set(qn('w:sz'), str(border_val.get('sz', 4)))
            border_el.set(qn('w:space'), '0')
            border_el.set(qn('w:color'), border_val.get('color', '000000'))
            tcBorders.append(border_el)
    tcPr.append(tcBorders)


def set_cell_background(cell, color):
    """设置单元格背景颜色"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), color)
    tcPr.append(shd)


def set_run_font(run, name='Times New Roman', size=None, bold=False, color=None, italic=False):
    """设置文字字体样式"""
    run.font.name = name
    if size:
        run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def set_paragraph_spacing(para, before=0, after=0, line_rule=None, line=None):
    """设置段落间距"""
    pPr = para._p.get_or_add_pPr()
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:before'), str(before))
    spacing.set(qn('w:after'), str(after))
    if line_rule and line:
        spacing.set(qn('w:lineRule'), line_rule)
        spacing.set(qn('w:line'), str(line))
    pPr.append(spacing)


def add_colored_run(para, text, color_rgb=None, bold=False, size=11, underline=False, italic=False):
    """在段落中添加带颜色的文字"""
    run = para.add_run(text)
    set_run_font(run, size=size, bold=bold, color=color_rgb, italic=italic)
    if underline:
        run.font.underline = True
    return run


def set_table_column_widths(table, widths):
    """设置表格列宽"""
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            if idx < len(widths):
                cell.width = widths[idx]


def main():
    doc = Document()

    # 页面设置：A4
    section = doc.sections[0]
    section.page_width = Cm(21.59)
    section.page_height = Cm(27.94)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)

    # ===== 标题 =====
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_spacing(title_para, before=120, after=120)
    run = title_para.add_run('WLA CCB Monitor Change White Paper')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True

    doc.add_paragraph()

    # ===== 1) Phase, Classification, Related WPs =====
    h1 = doc.add_paragraph()
    set_paragraph_spacing(h1, before=80, after=80)
    r = h1.add_run('1) Phase, Classification, Related WPs:')
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11)
    r.font.bold = True

    # Phase / Classification 表格
    tbl1 = doc.add_table(rows=6, cols=2)
    tbl1.style = 'Table Grid'
    tbl1.alignment = WD_TABLE_ALIGNMENT.LEFT

    col_widths_tbl1 = [Cm(7), Cm(11.5)]
    set_table_column_widths(tbl1, col_widths_tbl1)

    border_style = {'val': 'single', 'sz': 4, 'color': '000000'}

    # Row 0: Phase
    row0 = tbl1.rows[0]
    row0.cells[0].text = ''
    p_phase_label = row0.cells[0].paragraphs[0]
    p_phase_label.clear()
    r_phase = p_phase_label.add_run('Phase:')
    r_phase.font.name = 'Times New Roman'
    r_phase.font.size = Pt(11)
    r_phase.font.bold = True

    p_phase_val = row0.cells[1].paragraphs[0]
    p_phase_val.clear()
    add_colored_run(p_phase_val, '☐ PWP', size=11)
    add_colored_run(p_phase_val, '  ☒ FWP', size=11)

    # Row 1: FWP Horizon
    row1 = tbl1.rows[1]
    row1.cells[0].merge(row1.cells[1])
    p_r1 = row1.cells[0].paragraphs[0]
    p_r1.clear()
    add_colored_run(p_r1, 'For a FWP, document the PWP Horizon number (if applicable): ', size=11)
    add_colored_run(p_r1, 'N/a', size=11, bold=True, color_rgb=(0, 0, 255))

    # Row 2: Classification
    row2 = tbl1.rows[2]
    p_cls_label = row2.cells[0].paragraphs[0]
    p_cls_label.clear()
    r_cls = p_cls_label.add_run('Classification:')
    r_cls.font.name = 'Times New Roman'
    r_cls.font.size = Pt(11)
    r_cls.font.bold = True

    p_cls_val = row2.cells[1].paragraphs[0]
    p_cls_val.clear()
    add_colored_run(p_cls_val, '☐ 1   ☐ 2   ☐ 3   ☐ 3N   ☒ 4', size=11)

    # Row 3: Class IV
    row3 = tbl1.rows[3]
    row3.cells[0].merge(row3.cells[1])
    p_r3 = row3.cells[0].paragraphs[0]
    p_r3.clear()
    add_colored_run(p_r3, 'For Class IV WPs, add name of PCCB member confirming classification: ', size=11)
    add_colored_run(p_r3, 'N/A', size=11, bold=True, color_rgb=(0, 0, 255))

    # Row 4: 参考白皮书标题行（加粗）
    row4 = tbl1.rows[4]
    row4.cells[0].merge(row4.cells[1])
    p_r4 = row4.cells[0].paragraphs[0]
    p_r4.clear()
    r_r4 = p_r4.add_run(
        'Include any relevant reference white paper(s), "Me-Too" WPs, DRB, MRB, etc. in table below'
    )
    r_r4.font.name = 'Times New Roman'
    r_r4.font.size = Pt(11)
    r_r4.font.bold = True

    # Row 5: Horizon / Title 表头
    row5 = tbl1.rows[5]
    p_r5_0 = row5.cells[0].paragraphs[0]
    p_r5_0.clear()
    p_r5_0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_r5_0 = p_r5_0.add_run('Horizon or reference number')
    r_r5_0.font.name = 'Times New Roman'
    r_r5_0.font.size = Pt(11)
    r_r5_0.font.italic = True

    p_r5_1 = row5.cells[1].paragraphs[0]
    p_r5_1.clear()
    p_r5_1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_r5_1 = p_r5_1.add_run('Title')
    r_r5_1.font.name = 'Times New Roman'
    r_r5_1.font.size = Pt(11)
    r_r5_1.font.italic = True

    # 添加 N/a 行（在表格外再加一个两列行）
    tbl1_extra = doc.add_table(rows=2, cols=2)
    tbl1_extra.style = 'Table Grid'
    tbl1_extra.alignment = WD_TABLE_ALIGNMENT.LEFT
    set_table_column_widths(tbl1_extra, col_widths_tbl1)

    for i, txt in enumerate(['N/a', 'N/a']):
        p = tbl1_extra.rows[0].cells[i].paragraphs[0]
        p.clear()
        r = p.add_run(txt)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)

    # 空行
    for row in tbl1_extra.rows[1].cells:
        row.paragraphs[0].clear()

    doc.add_paragraph()

    # ===== 2) Date =====
    h2 = doc.add_paragraph()
    set_paragraph_spacing(h2, before=80, after=80)
    r_h2 = h2.add_run('2) Date: ')
    r_h2.font.name = 'Times New Roman'
    r_h2.font.size = Pt(11)
    r_h2.font.bold = True
    add_colored_run(h2, '04/02/2026', size=11, bold=True, color_rgb=(0, 0, 255))

    doc.add_paragraph()

    # ===== 3) Authorship =====
    h3 = doc.add_paragraph()
    set_paragraph_spacing(h3, before=80, after=80)
    r_h3 = h3.add_run('3) Authorship')
    r_h3.font.name = 'Times New Roman'
    r_h3.font.size = Pt(11)
    r_h3.font.bold = True

    authorship_items = [
        ('a.', 'Primary author: ', 'Yuan, Ji', True),
        ('b.', 'Site (primary author only): ', 'CDDP', True),
        ('c.', 'Co-author(s):', '', False),
    ]

    for letter, label, value, blue in authorship_items:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.left_indent = Cm(1.27)
        p.paragraph_format.first_line_indent = Cm(-0.63)
        p.clear()
        # 手动排版
        p_new = doc.add_paragraph()
        p_new.paragraph_format.left_indent = Cm(3.0)
        p_new.paragraph_format.first_line_indent = Cm(-1.5)
        set_paragraph_spacing(p_new, before=40, after=40)
        r_let = p_new.add_run(f'{letter}\t')
        r_let.font.name = 'Times New Roman'
        r_let.font.size = Pt(11)
        r_lbl = p_new.add_run(label)
        r_lbl.font.name = 'Times New Roman'
        r_lbl.font.size = Pt(11)
        if value:
            color = (0, 0, 255) if blue else None
            add_colored_run(p_new, value, size=11, bold=True, color_rgb=color)
        # 删除之前创建的 List Bullet 段落
        p._element.getparent().remove(p._element)

    doc.add_paragraph()

    # ===== 4) Title of Change =====
    h4 = doc.add_paragraph()
    set_paragraph_spacing(h4, before=80, after=80)
    r_h4 = h4.add_run('4) Title of Change: ')
    r_h4.font.name = 'Times New Roman'
    r_h4.font.size = Pt(11)
    r_h4.font.bold = True
    add_colored_run(h4, 'CD DGB chart limit change for CLSR flag', size=11, bold=True, color_rgb=(0, 0, 255))

    doc.add_paragraph()

    # ===== 5) Change Description =====
    h5 = doc.add_paragraph()
    set_paragraph_spacing(h5, before=80, after=80)
    r_h5 = h5.add_run('5) Change Description:')
    r_h5.font.name = 'Times New Roman'
    r_h5.font.size = Pt(11)
    r_h5.font.bold = True

    # a. Equipment tool set
    pa = doc.add_paragraph()
    pa.paragraph_format.left_indent = Cm(3.0)
    pa.paragraph_format.first_line_indent = Cm(-1.5)
    set_paragraph_spacing(pa, before=40, after=40)
    r_pa_let = pa.add_run('a.\t')
    r_pa_let.font.name = 'Times New Roman'
    r_pa_let.font.size = Pt(11)
    r_pa_lbl = pa.add_run('Equipment tool set affected (entity code or CEID): ')
    r_pa_lbl.font.name = 'Times New Roman'
    r_pa_lbl.font.size = Pt(11)
    add_colored_run(pa, '[Tool Set / CEID]', size=11, bold=True, color_rgb=(0, 0, 255))

    # b. Products affected
    pb = doc.add_paragraph()
    pb.paragraph_format.left_indent = Cm(3.0)
    pb.paragraph_format.first_line_indent = Cm(-1.5)
    set_paragraph_spacing(pb, before=40, after=40)
    r_pb_let = pb.add_run('b.\t')
    r_pb_let.font.name = 'Times New Roman'
    r_pb_let.font.size = Pt(11)
    r_pb_lbl = pb.add_run('Products affected (if change is product specific, otherwise "All"): ')
    r_pb_lbl.font.name = 'Times New Roman'
    r_pb_lbl.font.size = Pt(11)
    add_colored_run(pb, '[Products]', size=11, bold=True, color_rgb=(0, 0, 255))

    # c. Specific change items
    pc = doc.add_paragraph()
    pc.paragraph_format.left_indent = Cm(3.0)
    pc.paragraph_format.first_line_indent = Cm(-1.5)
    set_paragraph_spacing(pc, before=40, after=80)
    r_pc_let = pc.add_run('c.\t')
    r_pc_let.font.name = 'Times New Roman'
    r_pc_let.font.size = Pt(11)
    r_pc_lbl = pc.add_run('Specific change items.')
    r_pc_lbl.font.name = 'Times New Roman'
    r_pc_lbl.font.size = Pt(11)

    # ===== 变更项目表格 =====
    # 表头
    change_table = doc.add_table(rows=1, cols=4)
    change_table.style = 'Table Grid'
    change_table.alignment = WD_TABLE_ALIGNMENT.LEFT

    col_widths_ct = [Cm(0.8), Cm(5.5), Cm(5.5), Cm(6.5)]
    set_table_column_widths(change_table, col_widths_ct)

    header_row = change_table.rows[0]
    headers = ['#', 'Change items', 'Present value', 'Proposed value']
    header_bg = 'D3D3D3'

    for idx, hdr_text in enumerate(headers):
        cell = header_row.cells[idx]
        set_cell_background(cell, header_bg)
        p = cell.paragraphs[0]
        p.clear()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(hdr_text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
        r.font.bold = True

    # 三条变更记录数据
    change_records = [
        {
            'monitor_set': '[Monitor Set Name 1]',
            'measurement_set': '[Measurement Set Name 1]',
            'present_ucl': '[UCL value]',
            'present_cl': '[CL value]',
            'present_lcl': '[LCL value]',
            'present_clsr_flag': 'Flag',
            'proposed_ucl': '[New UCL]',
            'proposed_cl': '[New CL]',
            'proposed_lcl': '[New LCL]',
            'proposed_clsr_flag': '',
        },
        {
            'monitor_set': '[Monitor Set Name 2]',
            'measurement_set': '[Measurement Set Name 2]',
            'present_ucl': '[UCL value]',
            'present_cl': '[CL value]',
            'present_lcl': '[LCL value]',
            'present_clsr_flag': 'Flag',
            'proposed_ucl': '[New UCL]',
            'proposed_cl': '[New CL]',
            'proposed_lcl': '[New LCL]',
            'proposed_clsr_flag': '',
        },
        {
            'monitor_set': '[Monitor Set Name 3]',
            'measurement_set': '[Measurement Set Name 3]',
            'present_ucl': '[UCL value]',
            'present_cl': '[CL value]',
            'present_lcl': '[LCL value]',
            'present_clsr_flag': '',
            'proposed_ucl': '[New UCL]',
            'proposed_cl': '[New CL]',
            'proposed_lcl': '[New LCL]',
            'proposed_clsr_flag': '',
        },
    ]

    for rec in change_records:
        data_row = change_table.add_row()
        set_table_column_widths(change_table, col_widths_ct)

        # Col 0: #
        cell_num = data_row.cells[0]
        p_num = cell_num.paragraphs[0]
        p_num.clear()
        p_num.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_num = p_num.add_run('1')
        r_num.font.name = 'Times New Roman'
        r_num.font.size = Pt(11)

        # Col 1: Change items
        cell_items = data_row.cells[1]
        p_items = cell_items.paragraphs[0]
        p_items.clear()

        def add_change_item_line(para, label, value, is_first=False):
            if not is_first:
                para = cell_items.add_paragraph()
            set_paragraph_spacing(para, before=20, after=20)
            r_lbl = para.add_run(label)
            r_lbl.font.name = 'Times New Roman'
            r_lbl.font.size = Pt(10)
            r_val = para.add_run(value)
            r_val.font.name = 'Times New Roman'
            r_val.font.size = Pt(10)
            r_val.font.bold = True
            r_val.font.color.rgb = RGBColor(0, 0, 255)
            return para

        add_change_item_line(p_items, 'Monitor set name: ', rec['monitor_set'], is_first=True)
        add_change_item_line(cell_items.add_paragraph(), 'Measurement set name: ', rec['measurement_set'])
        add_change_item_line(cell_items.add_paragraph(), 'Chart type: ', 'CLSR')

        # Col 2: Present value
        cell_present = data_row.cells[2]
        p_present = cell_present.paragraphs[0]
        p_present.clear()

        def add_value_lines(cell, ucl, cl, lcl, clsr_flag, col_type='present'):
            p = cell.paragraphs[0]
            p.clear()
            set_paragraph_spacing(p, before=20, after=20)

            def add_kv(para, key, val, flag_text=''):
                r_k = para.add_run(f'{key}  :  ')
                r_k.font.name = 'Times New Roman'
                r_k.font.size = Pt(10)
                r_v = para.add_run(val)
                r_v.font.name = 'Times New Roman'
                r_v.font.size = Pt(10)
                r_v.font.bold = True
                r_v.font.color.rgb = RGBColor(0, 0, 255)
                if flag_text:
                    r_f = para.add_run(f'  {flag_text}')
                    r_f.font.name = 'Times New Roman'
                    r_f.font.size = Pt(10)
                    r_f.font.bold = True
                    r_f.font.color.rgb = RGBColor(0, 128, 0)

            add_kv(p, 'UCL', ucl)
            p2 = cell.add_paragraph()
            set_paragraph_spacing(p2, before=20, after=20)
            add_kv(p2, 'Centerline', cl)
            p3 = cell.add_paragraph()
            set_paragraph_spacing(p3, before=20, after=20)
            add_kv(p3, 'LCL', lcl)
            p4 = cell.add_paragraph()
            set_paragraph_spacing(p4, before=20, after=20)
            r_clsr = p4.add_run('CLSR')
            r_clsr.font.name = 'Times New Roman'
            r_clsr.font.size = Pt(10)
            r_colon = p4.add_run('  :  ')
            r_colon.font.name = 'Times New Roman'
            r_colon.font.size = Pt(10)
            if clsr_flag:
                r_flag = p4.add_run(clsr_flag)
                r_flag.font.name = 'Times New Roman'
                r_flag.font.size = Pt(10)
                r_flag.font.bold = True
                r_flag.font.color.rgb = RGBColor(0, 128, 0)

        add_value_lines(
            cell_present,
            rec['present_ucl'], rec['present_cl'], rec['present_lcl'], rec['present_clsr_flag']
        )

        # Col 3: Proposed value
        cell_proposed = data_row.cells[3]
        add_value_lines(
            cell_proposed,
            rec['proposed_ucl'], rec['proposed_cl'], rec['proposed_lcl'], rec['proposed_clsr_flag'],
            col_type='proposed'
        )

    # 保存
    output_path = '/workspace/WLA_CCB_Monitor_Change_White_Paper.docx'
    doc.save(output_path)
    print(f'文档已生成: {output_path}')


if __name__ == '__main__':
    main()
