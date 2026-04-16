"""
WLA CCB Monitor Change White Paper — PDF 生成器

与 generate_wla_ccb_whitepaper.py（docx 版）对应，
所有 _build_* 函数操作 flowables 列表，不写磁盘文件。
入口函数 build_wla_ccb_pdf(data) -> bytes 返回字节流。

依赖：
    pip install reportlab pdfrw
"""

import io
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle,
    Spacer, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

# ---------------------------------------------------------------------------
# 页面尺寸 / 颜色常量
# ---------------------------------------------------------------------------
PAGE_W = 21.59 * cm   # A4 宽
PAGE_H = 27.94 * cm   # A4 高
MARGIN = 2.54 * cm
CW = PAGE_W - 2 * MARGIN   # 可用内容宽度

C_BLUE      = colors.Color(0x00/255, 0x00/255, 0xFF/255)
C_GREEN     = colors.Color(0x00/255, 0x80/255, 0x00/255)
C_RED       = colors.Color(0xFF/255, 0x00/255, 0x00/255)
C_BLACK     = colors.black
C_WHITE     = colors.white
C_GRAY_HDR  = colors.Color(0xD3/255, 0xD3/255, 0xD3/255)   # 表头灰
C_GRAY_CELL = colors.Color(0xF2/255, 0xF2/255, 0xF2/255)   # 轻灰底


# ---------------------------------------------------------------------------
# 样式工厂
# ---------------------------------------------------------------------------

def _s(size=10, bold=False, color=C_BLACK, align=TA_LEFT, leading_mul=1.25):
    return ParagraphStyle(
        'auto',
        fontName='Helvetica-Bold' if bold else 'Helvetica',
        fontSize=size,
        textColor=color,
        alignment=align,
        leading=max(size * leading_mul, size + 2),
        spaceBefore=0,
        spaceAfter=0,
    )


def _p(text, size=10, bold=False, color=C_BLACK, align=TA_LEFT):
    """构造 Paragraph，支持 HTML 标签（<b><font color=...>）"""
    return Paragraph(str(text) if text is not None else '', _s(size, bold, color, align))


def _sp(h=4):
    """垂直间距 Spacer"""
    return Spacer(1, h)


# ---------------------------------------------------------------------------
# 表格工具
# ---------------------------------------------------------------------------

def _tbl_base():
    return [
        ('GRID',          (0, 0), (-1, -1), 0.5, C_BLACK),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
        ('TOPPADDING',    (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]


def _tbl(data, col_widths, style_extra=None):
    s = _tbl_base() + (style_extra or [])
    return Table(data, colWidths=col_widths, style=TableStyle(s), hAlign='LEFT')


# ---------------------------------------------------------------------------
# 页面回调（页脚）
# ---------------------------------------------------------------------------

def _page_callback(canv, doc):
    """每页绘制页脚：左文字 | 中文字 | Page N"""
    canv.saveState()
    canv.setFont('Helvetica-Oblique', 8)
    canv.setFillColor(C_BLACK)
    footer_y = MARGIN * 0.45

    left  = getattr(doc, '_footer_left',   'WLA CCB Monitor Change White Paper')
    center = getattr(doc, '_footer_center', 'Intel Confidential')

    canv.drawString(MARGIN, footer_y, left)
    canv.drawCentredString(PAGE_W / 2, footer_y, center)
    page_text = f'Page {doc.page}'
    canv.drawRightString(PAGE_W - MARGIN, footer_y, page_text)
    canv.restoreState()


# ---------------------------------------------------------------------------
# 章节构建函数
# ---------------------------------------------------------------------------

def _build_title(data):
    title_text = (
        '<u><b><font size="16">WLA CCB Monitor Change White Paper</font></b></u>'
    )
    return [
        Paragraph(title_text, _s(size=16, bold=True, align=TA_CENTER)),
        _sp(12),
    ]


def _build_section1(data):
    items = []
    items.append(_p('<b>1) Phase, Classification, Related WPs:</b>', size=12))
    items.append(_sp(4))

    # Phase 行
    phase_val = (
        '☐ PWP    '
        + ('<b>☒</b>' if data.get('phase', 'fwp') == 'fwp' else '☐')
        + ' FWP'
    )
    # FWP Horizon 行
    fwp_h = data.get('fwp_horizon', 'N/a')
    # Classification 行
    cls_boxes = ''
    selected_cls = data.get('classification', '4')
    for lbl in ['1', '2', '3', '3N', '4']:
        mark = '<b>☒</b>' if lbl == selected_cls else '☐'
        cls_boxes += f'{mark} {lbl}   '
    pccb = data.get('pccb_member', 'N/A')

    ref_rows = data.get('reference_wps', [{'horizon': 'N/a', 'title': 'N/a'}])

    section1_data = [
        # row 0: Phase（合并行）
        [Paragraph(f'Phase:&nbsp;&nbsp;&nbsp;{phase_val}', _s(10))],
        # row 1: FWP Horizon（合并行）
        [Paragraph(
            f'For a FWP, document the PWP Horizon number (if applicable): '
            f'<font color="#0000FF"><b>{fwp_h}</b></font>', _s(10)
        )],
        # row 2: Classification（合并行）
        [Paragraph(f'Classification:&nbsp;&nbsp;&nbsp;{cls_boxes}', _s(10))],
        # row 3: PCCB（合并行）
        [Paragraph(
            f'For Class IV WPs, add name of PCCB member confirming classification: '
            f'<font color="#0000FF"><b>{pccb}</b></font>', _s(10)
        )],
        # row 4: 参考WP 大标题
        [Paragraph(
            '<b>Include any relevant reference white paper(s), "Me-Too" WPs, DRB, MRB, etc. in table below</b>',
            _s(10)
        )],
    ]
    s1 = _tbl_base() + [('SPAN', (0, 0), (-1, 0))]
    tbl1 = Table(section1_data, colWidths=[CW], style=TableStyle(_tbl_base()))
    items.append(tbl1)
    items.append(_sp(2))

    # Horizon/Title 子表
    ref_data = [
        [_p('<i>Horizon or reference number</i>', 10, align=TA_CENTER),
         _p('<i>Title</i>', 10, align=TA_CENTER)],
    ]
    for r in ref_rows[:1]:
        ref_data.append([_p(r.get('horizon', 'N/a'), 10),
                         _p(r.get('title', 'N/a'), 10)])
    # 补空行
    if len(ref_data) < 3:
        ref_data.append([_p(''), _p('')])

    tbl_ref = _tbl(ref_data, [CW * 0.38, CW * 0.62])
    items.append(tbl_ref)
    items.append(_sp(8))
    return items


def _build_section2(data):
    date_val = data.get('date', '04/02/2026')
    return [
        Paragraph(
            f'<b><font size="12">2) Date: </font></b>'
            f'<font color="#0000FF"><b><font size="12">{date_val}</font></b></font>',
            _s(12)
        ),
        _sp(8),
    ]


def _build_section3(data):
    items = [_p('<b><font size="12">3) Authorship</font></b>', 12), _sp(4)]
    author  = data.get('primary_author', 'Yuan, Ji')
    site    = data.get('site', 'CDDP')
    coauth  = data.get('co_authors', '')
    sub = [
        ('a.', 'Primary author: ', author, True),
        ('b.', 'Site (primary author only): ', site, True),
        ('c.', 'Co-author(s):', coauth, False),
    ]
    for letter, label, val, blue in sub:
        color_tag = f'<font color="#0000FF"><b>{val}</b></font>' if (blue and val) else val
        items.append(Paragraph(
            f'&nbsp;&nbsp;&nbsp;&nbsp;{letter}&nbsp;&nbsp;{label}{color_tag}',
            _s(10)
        ))
        items.append(_sp(2))
    items.append(_sp(4))
    return items


def _build_section4(data):
    title_val = data.get('title_of_change', 'CD DGB chart limit change for CLSR flag')
    return [
        Paragraph(
            f'<b><font size="12">4) Title of Change: </font></b>'
            f'<font color="#0000FF"><b><font size="12">{title_val}</font></b></font>',
            _s(12)
        ),
        _sp(8),
    ]


def _build_section5_header(data):
    items = [_p('<b><font size="12">5) Change Description:</font></b>', 12), _sp(4)]
    eq_tool = data.get('equipment_tool_set', '[Tool Set / CEID]')
    products = data.get('products_affected', '[Products]')
    sub = [
        ('a.', 'Equipment tool set affected (entity code or CEID): ', eq_tool, True),
        ('b.', 'Products affected (if change is product specific, otherwise "All"): ', products, True),
        ('c.', 'Specific change items.', None, False),
    ]
    for letter, label, val, blue in sub:
        if val:
            color_tag = f'<font color="#0000FF"><b>{val}</b></font>' if blue else val
            txt = f'&nbsp;&nbsp;&nbsp;&nbsp;{letter}&nbsp;&nbsp;{label}{color_tag}'
        else:
            txt = f'&nbsp;&nbsp;&nbsp;&nbsp;{letter}&nbsp;&nbsp;{label}'
        items.append(Paragraph(txt, _s(10)))
        items.append(_sp(2))
    items.append(_sp(2))
    return items


def _build_change_table(data):
    """5c) 变更项目表格，Present/Proposed 列内嵌小表格（label : value）"""
    items = []

    # 列宽
    num_w      = 0.4 * inch
    present_w  = 2.2 * inch
    proposed_w = 2.2 * inch
    items_w    = CW - num_w - present_w - proposed_w

    hdr_style = _s(10, bold=True, align=TA_CENTER)
    hdr_row = [
        Paragraph('#', hdr_style),
        Paragraph('Change items', hdr_style),
        Paragraph('Present value', hdr_style),
        Paragraph('Proposed value', hdr_style),
    ]

    rows = [hdr_row]
    tbl_style = _tbl_base() + [
        ('BACKGROUND', (0, 0), (-1, 0), C_GRAY_HDR),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN',      (0, 0), (-1, 0), 'CENTER'),
    ]

    for rec in data.get('change_rows', []):
        num = _p(rec.get('number', '1'), 10, align=TA_CENTER)

        # Change items 列
        ms  = rec.get('monitor_set', '')
        meas = rec.get('measurement_set', '')
        ct  = rec.get('chart_type', 'CLSR')
        change_cell = Paragraph(
            f'Monitor set name: <font color="#0000FF"><b>{ms}</b></font><br/>'
            f'Measurement set name: <font color="#0000FF"><b>{meas}</b></font><br/>'
            f'Chart type: <font color="#0000FF"><b>{ct}</b></font>',
            _s(9)
        )

        # Present / Proposed 列 — 内嵌小表格
        def _limit_inner(rec, prefix):
            limit_rows = _limits_from_rec_pdf(rec, prefix)
            lw = present_w * 0.45
            cw = present_w * 0.10
            vw = present_w - lw - cw
            inner_data = []
            for lr in limit_rows:
                label_p = _p(lr['label'], 8)
                colon_p = _p(':', 8, align=TA_CENTER)
                val     = lr.get('value', '')
                flag    = lr.get('flag', '')
                vclr    = lr.get('value_color', C_BLUE if val else C_BLACK)
                fclr    = lr.get('flag_color', C_GREEN)
                val_html = ''
                if val:
                    r, g, b = int(vclr.red*255), int(vclr.green*255), int(vclr.blue*255)
                    val_html += f'<font color="#{r:02X}{g:02X}{b:02X}"><b>{val}</b></font>'
                if flag:
                    r, g, b = int(fclr.red*255), int(fclr.green*255), int(fclr.blue*255)
                    val_html += f' <font color="#{r:02X}{g:02X}{b:02X}"><b>{flag}</b></font>'
                val_p = Paragraph(val_html, _s(8))
                inner_data.append([label_p, colon_p, val_p])
            inner_style = TableStyle([
                ('INNERGRID',     (0, 0), (-1, -1), 0, C_WHITE),
                ('BOX',           (0, 0), (-1, -1), 0, C_WHITE),
                ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING',   (0, 0), (-1, -1), 1),
                ('RIGHTPADDING',  (0, 0), (-1, -1), 1),
                ('TOPPADDING',    (0, 0), (-1, -1), 1),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
            ])
            return Table(inner_data, colWidths=[lw, cw, vw], style=inner_style, hAlign='LEFT')

        rows.append([num, change_cell,
                     _limit_inner(rec, 'present'),
                     _limit_inner(rec, 'proposed')])

    tbl = Table(rows,
                colWidths=[num_w, items_w, present_w, proposed_w],
                style=TableStyle(tbl_style),
                hAlign='LEFT')
    items.append(tbl)
    items.append(_sp(6))
    return items


def _limits_from_rec_pdf(rec, prefix):
    """从 change_row 提取 limit_rows（PDF 版，颜色改为 reportlab Color）"""
    if 'limits' in rec:
        rows = []
        for item in rec['limits']:
            val    = item.get(prefix, '')
            flag   = item.get(f'{prefix}_flag', '')
            fc_str = item.get(f'{prefix}_flag_color', 'green')
            flag_clr = C_RED if fc_str == 'red' else C_GREEN
            rows.append({
                'label':       item.get('label', ''),
                'value':       val,
                'value_color': C_BLUE if val else C_BLACK,
                'flag':        flag,
                'flag_color':  flag_clr if flag else C_GREEN,
            })
        return rows

    default_limits = [
        ('UCL',        f'{prefix}_ucl',  ''),
        ('Centerline', f'{prefix}_cl',   ''),
        ('LCL',        f'{prefix}_lcl',  ''),
        ('CLSR',       f'{prefix}_clsr', f'{prefix}_clsr_flag'),
    ]
    rows = []
    for label, val_key, flag_key in default_limits:
        val  = rec.get(val_key, '')
        flag = rec.get(flag_key, '') if flag_key else ''
        rows.append({
            'label':       label,
            'value':       val,
            'value_color': C_BLUE if val else C_BLACK,
            'flag':        flag,
            'flag_color':  C_GREEN,
        })
    return rows


def _build_section5_fwp_table(data):
    items = []
    desc = data.get(
        'fwp_only_desc',
        'Incorporate the C-Spec (or equivalent) into the PWP Control Plan for '
        'future PWP versions. Otherwise, leave this blank.'
    )
    items.append(Paragraph(
        f'&nbsp;&nbsp;&nbsp;&nbsp;a.&nbsp;&nbsp;<b>(FWP only)</b> {desc}', _s(10)
    ))
    items.append(_sp(4))

    num_w   = 0.4 * inch
    val_w   = 2.2 * inch
    items_w = CW - num_w - val_w * 2
    headers = ['#', 'Change items', 'Previous value', 'Current value']
    hdr_row = [Paragraph(h, _s(10, bold=True, align=TA_CENTER)) for h in headers]
    rows = [hdr_row]
    for rec in data.get('fwp_change_rows', []):
        rows.append([
            _p(rec.get('number', ''), 9, align=TA_CENTER),
            _p(rec.get('change_item', ''), 9),
            _p(rec.get('previous_value', ''), 9),
            _p(rec.get('current_value', ''), 9),
        ])
    if len(rows) < 3:
        for _ in range(3 - len(rows)):
            rows.append([_p(''), _p(''), _p(''), _p('')])

    tbl = _tbl(rows, [num_w, items_w, val_w, val_w], [
        ('BACKGROUND', (0, 0), (-1, 0), C_GRAY_HDR),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN',      (0, 0), (-1, 0), 'CENTER'),
    ])
    items.append(tbl)
    items.append(_sp(8))
    return items


def _build_section6(data):
    reason = data.get('reason_for_change', '')
    return [
        _p('<b><font size="12">6) Reason for Change:</font></b>', 12),
        _sp(4),
        Paragraph(f'<font color="#0000FF">{reason}</font>', _s(11)),
        _sp(8),
    ]


def _build_section7(data):
    items = [
        Paragraph(
            '<b><font size="12">7) <u>CE!</u>/Site Implementation Owners:</font></b>',
            _s(12)
        ),
        _sp(4),
    ]
    owners = data.get('cei_owners', [{'name': '', 'site': 'CDDP', 'date': ''}])
    col_w = [CW * 0.38, CW * 0.12, CW * 0.50]
    hdr_row = [
        _p('<b>CE! Owners:</b>', 10),
        _p('<b>SITE</b>', 10, align=TA_CENTER),
        _p('<b>Date reviewed and approved:</b>', 10),
    ]
    rows = [hdr_row]
    for o in owners:
        name_html = f'<font color="#0000FF">{o.get("name","")}</font>' if o.get('name') else ''
        date_html = f'<font color="#0000FF">{o.get("date","")}</font>' if o.get('date') else ''
        rows.append([Paragraph(name_html, _s(10)),
                     _p(o.get('site', ''), 10, align=TA_CENTER),
                     Paragraph(date_html, _s(10))])
    tbl = _tbl(rows, col_w, [
        ('BACKGROUND', (0, 0), (-1, 0), C_GRAY_CELL),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
    ])
    items.append(tbl)
    note = data.get('cei_note',
                    'Changes impacting VF sites need to be reviewed/approved by owners responsible for '
                    'implementation at their respective sites, at both PWP and FWP stages.')
    items.append(_sp(3))
    items.append(Paragraph(f'• {note}', _s(9)))
    items.append(_sp(8))
    return items


def _build_section8(data):
    items = [
        Paragraph(
            '<b><font size="12">8) Specifications, Controlled Documents Affected:</font></b>'
            ' <i><font size="9">(list all affected by changes above)</font></i>',
            _s(12)
        ),
        _sp(4),
    ]
    col_w = [CW * 0.45, CW * 0.55]
    hdr_row = [
        _p('<b>Spec and/or Controlled Document #</b>', 10, align=TA_CENTER),
        _p('<b>Document Title</b>', 10, align=TA_CENTER),
    ]
    rows = [hdr_row]
    for rec in data.get('spec_rows', [{'spec_num': 'N/a', 'title': 'N/a'}]):
        rows.append([_p(rec.get('spec_num', ''), 9), _p(rec.get('title', ''), 9)])
    tbl = _tbl(rows, col_w, [
        ('BACKGROUND', (0, 0), (-1, 0), C_GRAY_HDR),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN',      (0, 0), (-1, 0), 'CENTER'),
    ])
    items.append(tbl)
    items.append(_sp(8))
    return items


def _build_section9(data):
    items = [
        _p('<b><font size="12">9) Concerns and Considerations:</font></b>', 12),
        _sp(4),
    ]
    col_ratios = [0.05, 0.15, 0.20, 0.45, 0.15]
    col_w = [CW * r for r in col_ratios]
    hdr_row = [
        _p('<b>#</b>', 9, align=TA_CENTER),
        _p('<b>Forum\nidentifying\nconcern</b>', 9, align=TA_CENTER),
        _p('<b>Issue</b>', 9, align=TA_CENTER),
        _p('<b>Resolution</b>', 9, align=TA_CENTER),
        _p('<b>Status:\n(Open or\nClosed)</b>', 9, align=TA_CENTER),
    ]
    rows = [hdr_row]
    for rec in data.get('concern_rows', []):
        rows.append([
            Paragraph(f'<font color="#0000FF">{rec.get("number","")}</font>', _s(9)),
            Paragraph(f'<font color="#0000FF">{rec.get("forum","")}</font>', _s(9)),
            _p(rec.get('issue', ''), 9),
            _p(rec.get('resolution', ''), 9),
            Paragraph(f'<font color="#0000FF">{rec.get("status","")}</font>', _s(9)),
        ])
    if len(rows) < 3:
        for _ in range(3 - len(rows)):
            rows.append([_p(''), _p(''), _p(''), _p(''), _p('')])
    tbl = _tbl(rows, col_w, [
        ('BACKGROUND', (0, 0), (-1, 0), C_GRAY_HDR),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN',      (0, 0), (-1, 0), 'CENTER'),
    ])
    items.append(tbl)
    items.append(_sp(8))
    return items


def _checkbox(checked):
    return '☒' if checked else '☐'


def _build_section10(data):
    items = []

    # concerns_note bullet
    note = data.get('concerns_note', '')
    if note:
        items.append(Paragraph(f'• {note}', _s(9)))
        items.append(_sp(4))

    items.append(_p('<b><font size="12">10) Control Chart Setup: Only fill in section for SPC++ changes or SPC# chart creation</font></b>', 12))
    items.append(_sp(6))

    # a. Chart Modification
    items.append(Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;a.&nbsp;&nbsp;Chart Modification: (check one)', _s(10)))
    items.append(_sp(3))
    mod_sel = data.get('chart_modification', 'revision')
    for key, label in [('revision', 'Chart Revision'),
                       ('new',      'New Chart Creation. Please complete section 10b.'),
                       ('deletion', 'Chart Deletion')]:
        items.append(Paragraph(
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{_checkbox(mod_sel==key)}&nbsp;&nbsp;{label}',
            _s(10)
        ))
    items.append(_sp(4))

    # b. Chart By
    items.append(Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;b.&nbsp;&nbsp;Chart By: (check one and fill out table below, only needed for New Chart Creation)', _s(10)))
    items.append(_sp(3))
    chart_by = data.get('chart_by', ['all_categories'])
    if isinstance(chart_by, str):
        chart_by = [chart_by]
    cb_pairs = [
        [('equipment', 'Equipment'),           ('all_categories', 'All Categories')],
        [('monitor',   'Monitor'),              ('process',        'Process')],
        [('operation', 'Operation'),            ('custom_context', 'Custom Context Categories')],
        [('product',   'Product'),              (None,             '')],
    ]
    for row_opts in cb_pairs:
        left  = row_opts[0]
        right = row_opts[1]
        left_txt  = f'{_checkbox(left[0] in chart_by)}&nbsp;&nbsp;{left[1]}' if left[1] else ''
        right_txt = f'{_checkbox(right[0] in chart_by)}&nbsp;&nbsp;{right[1]}' if right[1] else ''
        items.append(Paragraph(
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{left_txt}'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{right_txt}',
            _s(10)
        ))
    items.append(_sp(4))

    # SPC 15列表格 (b)
    items += _build_spc15_table_pdf(data.get('spc_setup_rows', []))

    # Bullet 说明
    for note_txt in data.get('spc_setup_notes', [
        'Oper: Operation in which the chart will be posted.',
        'Class: Classification of the measurement, KPP or CPP for SPC++ / For SPC#, Key, Control, or Engineering.',
        'Calc Method: Control limit calculation method, Eng. Limits or Calculated (e.g. Std 3 sigma, 4 sigma, percentile).',
    ]):
        if ':' in note_txt:
            key_part, rest = note_txt.split(':', 1)
            items.append(Paragraph(
                f'• <u>{key_part}</u>:{rest}', _s(9)
            ))
        else:
            items.append(Paragraph(f'• {note_txt}', _s(9)))
        items.append(_sp(2))
    items.append(_sp(4))

    # c. SPC Rules
    items.append(Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;c.&nbsp;&nbsp;SPC Rules: (check one and fill out table below)', _s(10)))
    items.append(_sp(3))
    rules_sel = data.get('spc_rules', 'no_changes')
    for key, label in [('no_rules', 'No Rules Set'), ('no_changes', 'No Changes Proposed')]:
        items.append(Paragraph(
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{_checkbox(rules_sel==key)}&nbsp;&nbsp;{label}',
            _s(10)
        ))
    items.append(_sp(4))

    spc_rules_note = data.get('spc_rules_note',
        'Complete the following table by documenting all current and proposed rules.')
    items.append(_p(spc_rules_note, 9))
    items.append(_sp(3))
    items += _build_spc15_table_pdf(data.get('spc_rules_rows', []))
    items += _build_rules_legend_pdf()

    items.append(_p('If you have more custom rules that do not fit the standard rule codes above, provide details here.', 9))
    items.append(_sp(6))

    # d. Control Chart Data Summary
    items.append(Paragraph('&nbsp;&nbsp;&nbsp;&nbsp;d.&nbsp;&nbsp;Control Chart Data Summary:', _s(10)))
    items.append(_sp(3))
    summary_note = data.get('data_summary_note',
        'Select type of limits, and explain assumptions.')
    items.append(_p(summary_note, 9))
    items.append(_sp(3))

    ds_sel = data.get('data_summary_type', ['vf_common'])
    if isinstance(ds_sel, str):
        ds_sel = [ds_sel]
    ds_options = [
        ('tool_specific', 'Tool-Specific', None),
        ('fab_specific',  'Fab-Specific',  '(First time CEI deviation need to show justification of why fabs are different)'),
        ('vf_common',     'VF-Common',     None),
    ]
    for key, label, sub in ds_options:
        sub_txt = f' <font size="8">{sub}</font>' if sub else ''
        items.append(Paragraph(
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{_checkbox(key in ds_sel)}&nbsp;&nbsp;{label}{sub_txt}',
            _s(10)
        ))
    items.append(_sp(8))
    return items


def _build_spc15_table_pdf(data_rows):
    """15列 SPC 表格，第5-13列表头竖向（用换行+小字模拟）"""
    col_defs = [
        ('Oper',             0.045),
        ('SPC\nFUNCTIONAL\nAREA', 0.095),
        ('MONITOR\nSET\nNAME',   0.085),
        ('MEASUREMENT\nSET\nNAME', 0.110),
        ('CHART\nSUBSET/\nTestName', 0.065),
        ('TYPE',   0.030),
        ('LCL',    0.030),
        ('CL',     0.030),
        ('TARGET', 0.030),
        ('LDL',    0.030),
        ('UDL',    0.030),
        ('LUL',    0.030),
        ('UBL',    0.030),
        ('Class',  0.060),
        ('Calc\nMethod', 0.065),
    ]
    ratio_sum = sum(r for _, r in col_defs)
    col_w = [CW * r / ratio_sum for _, r in col_defs]

    hdr_row = [
        Paragraph(h.replace('\n', '<br/>'), _s(6, bold=True, align=TA_CENTER))
        for h, _ in col_defs
    ]
    rows = [hdr_row]
    keys = ['oper', 'spc_area', 'monitor_set', 'measurement_set',
            'subset_num', 'type', 'lcl', 'cl', 'target',
            'ldl', 'udl', 'lul', 'ubl', 'class_', 'calc_method']

    fill_rows = data_rows if data_rows else [{} for _ in range(3)]
    for rec in fill_rows:
        rows.append([_p(rec.get(k, ''), 7) for k in keys])

    tbl = _tbl(rows, col_w, [
        ('BACKGROUND', (0, 0), (-1, 0), C_GRAY_HDR),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN',      (0, 0), (-1, 0), 'CENTER'),
        ('ROWHEIGHT',  (0, 0), (-1, 0), 40),
    ])
    return [tbl, _sp(4)]


def _build_rules_legend_pdf():
    legend = [
        ('A', '> UCL'),         ('I', '15 Inside SIGMA'),
        ('B', '2/1 > 2 SIGMA'), ('J', '8 Outside SIGMA'),
        ('C', '4/5 > 1 SIGMA'), ('K', '> UDL'),
        ('D', 'Last 8 > CL'),   ('L', '< LDL'),
        ('E', '< LCL'),         ('M', 'Missing Data'),
        ('F', '2/3 < -2 SIGMA'),('N', 'Fail Disposition'),
        ('G', '4/5 < -1 SIGMA'),('O', 'No Limits'),
        ('H', 'Last 8 < CL'),   ('NONE', 'No OOC Rules'),
    ]
    items = [Paragraph('<super>1</super>Rules are denoted as follows:', _s(8))]
    cw = CW / 8
    rows = []
    for i in range(0, len(legend), 2):
        left  = legend[i]
        right = legend[i+1] if i+1 < len(legend) else ('', '')
        rows.append([
            _p(f'<b>{left[0]}</b>', 7),  _p(left[1], 7),
            _p(f'<b>{right[0]}</b>', 7), _p(right[1], 7),
        ])
    tbl = Table(rows, colWidths=[cw * 0.5, cw * 1.5, cw * 0.7, cw * 1.3] * 1,
                style=TableStyle([
                    ('INNERGRID', (0, 0), (-1, -1), 0, C_WHITE),
                    ('BOX',       (0, 0), (-1, -1), 0, C_WHITE),
                    ('VALIGN',    (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING',   (0, 0), (-1, -1), 2),
                    ('RIGHTPADDING',  (0, 0), (-1, -1), 2),
                    ('TOPPADDING',    (0, 0), (-1, -1), 1),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
                ]),
                hAlign='LEFT')
    items.append(tbl)
    items.append(_sp(4))
    return items


def _build_section11(data):
    items = [
        _p('<b><font size="12">11) Specific Checklists:</font></b>', 12),
        _sp(4),
    ]
    note = data.get('checklist_note',
                    'Attach all appropriate checklists in this section. '
                    'Applicable checklist and worksheet templates available from "Handbook and Templates".')
    items.append(_p(note, 9))
    items.append(_sp(4))

    col_w = [CW * 0.38, CW * 0.32, CW * 0.30]
    hdr_row = [
        _p('<b>Item</b>', 9, align=TA_CENTER),
        _p('<b>Embedded Document(s)</b>', 9, align=TA_CENTER),
        _p('<b>Comments</b>', 9, align=TA_CENTER),
    ]
    rows = [hdr_row]
    for rec in data.get('checklist_rows', [
        {'item': 'APC Add/Change Checklist',           'doc': 'N/A', 'comments': ''},
        {'item': 'Software/Firmware Change Checklist', 'doc': 'N/A', 'comments': ''},
        {'item': 'Monitor Sampling Reduction Worksheet','doc': 'N/A', 'comments': ''},
        {'item': 'Other:',                             'doc': 'N/A', 'comments': ''},
    ]):
        doc_html = (
            f'<font color="#0000FF">{rec.get("doc","")}</font>'
            if rec.get('doc') else ''
        )
        rows.append([
            _p(rec.get('item', ''), 9),
            Paragraph(doc_html, _s(9, align=TA_CENTER)),
            _p(rec.get('comments', ''), 9),
        ])
    tbl = _tbl(rows, col_w, [
        ('BACKGROUND', (0, 0), (-1, 0), C_GRAY_HDR),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN',      (0, 0), (-1, 0), 'CENTER'),
    ])
    items.append(tbl)
    items.append(_sp(8))
    return items


def _build_section12(data):
    items = [
        _p('<b><font size="12">12) Data Details</font></b>', 12),
        _sp(4),
    ]
    detail_note = data.get('data_details_note',
        '(Include and clearly label data tables, key graphs and statistical analysis summaries '
        'that support the intended change; including data as an embedded object is strongly preferred.)')
    items.append(Paragraph(f'<i>{detail_note}</i>', _s(9)))
    items.append(_sp(6))

    for item in data.get('data_details_items', []):
        title = item.get('title', '')
        items.append(Paragraph(
            f'<font color="#0000FF"><b>{title}</b></font>', _s(10)
        ))
        if item.get('description'):
            items.append(_p(item['description'], 9))
        # 图表占位（PDF 版暂用灰框）
        items.append(_sp(4))
        items.append(_insert_chart_placeholder_pdf(item))
        items.append(_sp(8))
    return items


def _insert_chart_placeholder_pdf(item):
    """
    PDF 版图表区域：
      - 若 item 中含 png_buf（bytes/BytesIO），尝试用 ReportLab 插入 PNG
      - 否则显示灰色占位框
    """
    from reportlab.platypus import Image as RLImage
    width_cm = item.get('svg_width_cm', 14.0)
    w_pt = width_cm * cm

    png_buf = item.get('png_buf')
    if png_buf is not None:
        try:
            if isinstance(png_buf, bytes):
                buf = io.BytesIO(png_buf)
            else:
                png_buf.seek(0)
                buf = png_buf
            img = RLImage(buf, width=w_pt)
            img.hAlign = 'LEFT'
            return img
        except Exception:
            pass

    # 灰色占位框
    class _GrayBox(Flowable):
        def __init__(self, w, h=4 * cm):
            super().__init__()
            self.w, self.h = w, h

        def draw(self):
            self.canv.setFillColor(colors.Color(0.92, 0.92, 0.92))
            self.canv.setStrokeColor(colors.Color(0.6, 0.6, 0.6))
            self.canv.rect(0, 0, self.w, self.h, fill=1, stroke=1)
            self.canv.setFont('Helvetica-Oblique', 9)
            self.canv.setFillColor(colors.Color(0.5, 0.5, 0.5))
            self.canv.drawCentredString(self.w / 2, self.h / 2 - 4,
                                        f'[Chart: {item.get("title", "")}]')

        def wrap(self, *args):
            return self.w, self.h

    return _GrayBox(w_pt)


# ---------------------------------------------------------------------------
# 文档组装
# ---------------------------------------------------------------------------

def build_wla_ccb_pdf(data: dict) -> bytes:
    """
    生成 WLA CCB Monitor Change White Paper PDF。

    参数
    ----
    data : dict  与 build_wla_ccb_document（docx版）完全相同的数据字典。

    返回
    ----
    bytes : 可直接作为 HTTP 响应体或写入 .pdf 文件。
    """
    flowables = []
    flowables += _build_title(data)
    flowables += _build_section1(data)
    flowables += _build_section2(data)
    flowables += _build_section3(data)
    flowables += _build_section4(data)
    flowables += _build_section5_header(data)
    flowables += _build_change_table(data)
    flowables += _build_section5_fwp_table(data)
    flowables += _build_section6(data)
    flowables += _build_section7(data)
    flowables += _build_section8(data)
    flowables += _build_section9(data)
    flowables += _build_section10(data)
    flowables += _build_section11(data)
    flowables += _build_section12(data)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=(PAGE_W, PAGE_H),
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN * 1.5,
    )
    # 注入页脚文字
    doc._footer_left   = data.get('footer_left',   'WLA CCB Monitor Change White Paper')
    doc._footer_center = data.get('footer_center', 'Intel Confidential')

    doc.build(flowables,
              onFirstPage=_page_callback,
              onLaterPages=_page_callback)
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
        'footer_left':   'WLA CCB Monitor Change White Paper',
        'footer_center': 'Intel Confidential',
        'reason_for_change': 'To tighten the limit for CLSR flagging',
        'cei_owners': [{'name': 'Owner A', 'site': 'CDDP', 'date': '04/02/2026'}],
        'cei_note': 'Any owners responsible for both FWP and PWP stages.',
        'concerns_note': 'Any owners responsible for both FWP and PWP stages.',
        'spec_rows': [{'spec_num': 'N/a', 'title': 'N/a'}],
        'concern_rows': [
            {'number': '1', 'forum': 'Originator', 'issue': 'Why do this change?',
             'resolution': 'Based on 25 weeks data.', 'status': 'Closed'},
            {'number': '2', 'forum': 'Module WG', 'issue': 'Why no QC required?',
             'resolution': 'Catalyst limit is sufficient.', 'status': 'Closed'},
            {'number': '3', 'forum': 'Originator', 'issue': 'Why no other sites?',
             'resolution': 'CDDP only.', 'status': 'Closed'},
        ],
        'change_rows': [
            {
                'number': '1',
                'monitor_set': 'MON_SET_001',
                'measurement_set': 'MEAS_SET_001',
                'chart_type': 'CLSR',
                'limits': [
                    {'label': 'UCL',        'present': '493',  'proposed': '488.4'},
                    {'label': 'Centerline', 'present': '487',  'proposed': '487'},
                    {'label': 'LCL',        'present': '481',  'proposed': '485.6'},
                    {'label': 'CLSR',
                     'present': '17.4', 'present_flag': 'Flag', 'present_flag_color': 'red',
                     'proposed': '4',   'proposed_flag': ''},
                ],
            },
            {
                'number': '1',
                'monitor_set': 'MON_SET_002',
                'measurement_set': 'MEAS_SET_002',
                'chart_type': 'CLSR',
                'limits': [
                    {'label': 'UCL',        'present': '350.25', 'proposed': '352.0'},
                    {'label': 'Centerline', 'present': '348.10', 'proposed': '348.1'},
                    {'label': 'LCL',        'present': '345.95', 'proposed': '344.2'},
                    {'label': 'CLSR',
                     'present': '8.2', 'present_flag': 'Flag', 'present_flag_color': 'green',
                     'proposed': '2',  'proposed_flag': ''},
                ],
            },
        ],
        'checklist_rows': [
            {'item': 'APC Add/Change Checklist',            'doc': 'N/A', 'comments': ''},
            {'item': 'Software/Firmware Change Checklist',  'doc': 'N/A', 'comments': ''},
            {'item': 'Monitor Sampling Reduction Worksheet','doc': 'N/A', 'comments': ''},
            {'item': 'Other:',                              'doc': 'N/A', 'comments': ''},
        ],
        'data_details_items': [
            {'title': '1. X-bar Control Limit Summary for "Value" (Statistical)',
             'description': '', 'svg_width_cm': 14.0},
        ],
    }

    pdf_bytes = build_wla_ccb_pdf(sample_data)

    output_path = 'WLA_CCB_Monitor_Change_White_Paper.pdf'
    with open(output_path, 'wb') as f:
        f.write(pdf_bytes)

    print(f'字节长度: {len(pdf_bytes):,}')
    print(f'已写入: {output_path}')


if __name__ == '__main__':
    main()
