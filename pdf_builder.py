"""
PDF 文档构建模块

依赖：
  - reportlab  （前两页：表格/文字内容）
  - pdfrw      （合并 ReportLab PDF 与 matplotlib PDF，纯 Python，无系统库，无 typing-extensions 依赖）
  - matplotlib （由 draw_spc_chart 生成真矢量 PDF 图表页）
"""
import io
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from pdfrw import PdfReader, PdfWriter
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from reportlab.platypus import Flowable

BG_COLOR = colors.Color(235 / 255, 235 / 255, 249 / 255)
DARK_BG  = colors.Color(0x2B / 255, 0x3D / 255, 0x28 / 255)
PINK_BG  = colors.Color(0x77 / 255, 0x20 / 255, 0x6D / 255)
GREEN_BG = colors.Color(0x3A / 255, 0x7C / 255, 0x22 / 255)
DARK_ROW = colors.Color(0xD1 / 255, 0xD1 / 255, 0xD1 / 255)
PINK_ROW = colors.Color(0xF2 / 255, 0xCE / 255, 0xED / 255)
GREEN_ROW = colors.Color(0xD9 / 255, 0xF2 / 255, 0xD0 / 255)
NOTE_BG  = colors.Color(0xC9 / 255, 0xCC / 255, 0xE8 / 255)
HDR_BG   = colors.Color(0xC5 / 255, 0xD3 / 255, 0xE8 / 255)
BORDER   = colors.black
WHITE    = colors.white
GRAY     = colors.Color(0.4, 0.4, 0.4)
BLUE     = colors.Color(0x28 / 255, 0x6E / 255, 0xC8 / 255)

PAGE_W = 35.56 * cm
PAGE_H = 21.59 * cm
MARGIN = 1.27 * cm
CW = PAGE_W - 2 * MARGIN


def _s(size=11, bold=False, color=colors.black, align=TA_LEFT):
    return ParagraphStyle(
        'auto',
        fontName='Helvetica-Bold' if bold else 'Helvetica',
        fontSize=size,
        textColor=color,
        alignment=align,
        leading=size * 1.3,
        spaceBefore=0,
        spaceAfter=0,
    )


def _p(text, size=11, bold=False, color=colors.black, align=TA_LEFT):
    return Paragraph(str(text) if text else '', _s(size, bold, color, align))


def _page_callback(canv, doc):
    canv.saveState()
    canv.setFillColor(BG_COLOR)
    canv.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canv.setFont('Helvetica', 9)
    canv.setFillColor(GRAY)
    canv.drawString(MARGIN, MARGIN * 0.4, 'Rev 2.0')
    canv.restoreState()


def _info_link(url):
    markup = '<a href="{0}"><font color="#286EC8"><u><b>INFO</b></u></font></a>'.format(url)
    return Paragraph(markup, _s(size=16, bold=True, color=BLUE))


def _tbl_defaults():
    return [
        ('GRID', (0, 0), (-1, -1), 1, BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]


def _first_page(data):
    el = [
        _p('AT INTEGRATED CCB WP TEMPLATE', size=16, bold=True, align=TA_CENTER),
        Spacer(1, 3),
        _p(
            'AS CORPORATE DIRECTIVE PROTECT YOUR CHANGE WITH ERM CREDENTIALS'
            ' FOR ATTD OR STTD CCB RELATED WPS ONLY',
            size=9, bold=True, align=TA_CENTER
        ),
        Spacer(1, 6),
        _info_link(data.get('info_url', 'http://mfgreports.ch.intel.com/WP_Documentation/Word_Template/Section_1-6.htm')),
        Spacer(1, 2)
    ]

    lw = 5.0 * cm
    vw = CW - lw
    note = ('DOE sections (9, 10, 11, 13, 16) are optional for class IV'
            ' if the changes do not require experiment data collection')

    t1_data = [
        [_p('1. White Paper Type', bold=True), _p(data.get('white_paper_type', ''))],
        [_p('2. Classification', bold=True), _p(data.get('classification', ''))],
        [_p(note, bold=True), ''],
        [_p('3. Name of Owner', bold=True), _p(data.get('owner_name', ''))],
        [_p('4. Title of Change', bold=True), _p(data.get('title_of_change', ''))],
        [_p('5. Change Description', bold=True), _p(data.get('change_description', ''))],
        [_p('6. Reason for Change', bold=True), _p(data.get('reason_for_change', ''))],
    ]

    t1 = Table(
        t1_data,
        colWidths=[lw, vw],
        rowHeights=[0.8 * cm, 0.8 * cm, 1.0 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm]
    )
    t1.setStyle(TableStyle(_tbl_defaults() + [
        ('SPAN', (0, 2), (1, 2)),
        ('BACKGROUND', (1, 0), (1, 1), colors.white),
        ('BACKGROUND', (1, 3), (1, 6), colors.white),
        ('LINEBEFORE', (0, 2), (0, 2), 2, BG_COLOR),
        ('LINEAFTER', (1, 2), (1, 2), 2, BG_COLOR),
    ]))
    el.append(t1)
    el.append(Spacer(1, 4))

    el.append(_info_link(data.get('info_url2', 'http://mfgreports.ch.intel.com/WP_Documentation/Word_Template/Section_7.htm')))
    el.append(Spacer(1, 4))
    el.append(_p('7. Process Factors', bold=True))
    el.append(Spacer(1, 2))

    factors = data.get('process_factors', [{'factor': '', 'present': '', 'proposed': ''}])
    pcw = CW / 3
    pf_data = [[_p('Process Factor', bold=True), _p('Present Value', bold=True), _p('Proposed Value', bold=True)]]
    for f in factors:
        pf_data.append([
            _p(f.get('factor', '') or 'Please refer to Table below'),
            _p(f.get('present', '') or 'Please refer to Table below'),
            _p(f.get('proposed', '') or 'Please refer to Table below'),
        ])

    pf = Table(pf_data,
               colWidths=[pcw, pcw, pcw],
               rowHeights=[0.7 * cm] + [0.8 * cm] * len(factors))
    pf.setStyle(TableStyle(_tbl_defaults() + [
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ]))
    el.append(pf)

    return el


def _second_page(data):
    el = [PageBreak()]

    fixed_headers = ['SPC area', 'Monitor set', 'Measurement set', 'Chart type', 'Control Limit Type']
    value_sub = ['LCL', 'CL', 'UCL', 'OCI', '%OOC']
    keys_fixed = ['spc_area', 'monitor_set', 'measurement_set', 'chart_type', 'control_limit_type']
    keys_val = ['lcl', 'cl', 'ucl', 'oci', 'ooc']

    fw = [2 * cm, 2.5 * cm, 3.5 * cm, 1.6 * cm, 2.5 * cm]
    vw = (13.06 * inch - sum(fw)) / 10
    spc_rows = data.get('spc_rows', [{}])
    n_data = max(len(spc_rows), 1)

    def _hw(t): return _p(t, size=10, bold=True, color=WHITE, align=TA_CENTER)
    def _hg(t): return _p(t, size=10, color=WHITE, align=TA_CENTER)
    def _dc(t): return _p(str(t) if t else '', size=10)

    row0 = [_hw(h) for h in fixed_headers] + [_hw('Present value')] + [''] * 4 + [_hw('Proposed value')] + [''] * 4
    row1 = [''] * 5 + [_hg(t) for t in value_sub] + [_hg(t) for t in value_sub]

    data_rows = []
    for rd in (spc_rows if spc_rows else [{}]):
        row = [_dc(rd.get(k, '')) for k in keys_fixed]
        row += [_dc(rd.get('present_' + k, '')) for k in keys_val]
        row += [_dc(rd.get('proposed_' + k, '')) for k in keys_val]
        data_rows.append(row)

    hdr_h = 0.22 * 2.54 * cm
    style_cmds = [
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('SPAN', (0, 0), (0, 1)),
        ('SPAN', (1, 0), (1, 1)),
        ('SPAN', (2, 0), (2, 1)),
        ('SPAN', (3, 0), (3, 1)),
        ('SPAN', (4, 0), (4, 1)),
        ('SPAN', (5, 0), (9, 0)),
        ('SPAN', (10, 0), (14, 0)),
        ('BACKGROUND', (0, 0), (4, 1), DARK_BG),
        ('BACKGROUND', (5, 0), (9, 0), PINK_BG),
        ('BACKGROUND', (10, 0), (14, 0), GREEN_BG),
        ('BACKGROUND', (5, 1), (9, 1), PINK_BG),
        ('BACKGROUND', (10, 1), (14, 1), GREEN_BG),
    ]
    for i in range(n_data):
        r = 2 + i
        style_cmds += [
            ('BACKGROUND', (0, r), (4, r), DARK_ROW),
            ('BACKGROUND', (5, r), (9, r), PINK_ROW),
            ('BACKGROUND', (10, r), (14, r), GREEN_ROW),
        ]

    spc_tbl = Table(
        [row0, row1] + data_rows,
        colWidths=fw + [vw] * 10,
        rowHeights=[hdr_h, hdr_h] + [0.24 * 2.54 * cm] * n_data
    )
    spc_tbl.setStyle(TableStyle(style_cmds))
    el.append(spc_tbl)
    el.append(Spacer(1, 5))

    slw = 2.06 * 2.54 * cm
    svw = CW - slw
    src_tbl = Table(
        [[_p('Source of Reference Data', bold=True), _p(data.get('source_ref', ''))]],
        colWidths=[slw, svw],
        rowHeights=[0.7 * cm]
    )
    src_tbl.setStyle(TableStyle([
        ('GRID', (1, 0), (1, 0), 0.5, BORDER),
        ('LINEAFTER', (0, 0), (0, 0), 0.5, BORDER),
        ('LINEBEFORE', (0, 0), (0, 0), 0, BG_COLOR),
        ('LINEABOVE', (0, 0), (0, 0), 0, BG_COLOR),
        ('LINEBELOW', (0, 0), (0, 0), 0, BG_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('BACKGROUND', (1, 0), (1, 0), WHITE),
    ]))
    el.append(src_tbl)

    return el


class _PdfImageFlowable(Flowable):
    """
    将 matplotlib 输出的单页 PDF 作为矢量 Form XObject 嵌入 ReportLab 页面。

    pdfrw.pagexobj  把 PDF 页转成 XObject（保留所有矢量路径/字体）。
    pdfrw.makerl    将 XObject 注册到 ReportLab canvas 的资源表。
    draw()          以指定宽高把 XObject 绘制到当前位置。
    """

    def __init__(self, pdf_bytes, width, height=None):
        Flowable.__init__(self)
        page = PdfReader(fdata=pdf_bytes).pages[0]
        self._xobj = pagexobj(page)

        # PDF MediaBox 的原始宽高（pt）
        mb = self._xobj.BBox
        src_w = float(mb[2]) - float(mb[0])
        src_h = float(mb[3]) - float(mb[1])

        self.width = width
        # 未指定高度时按原始宽高比等比缩放
        self.height = height if height is not None else width * (src_h / src_w)
        self._src_w = src_w
        self._src_h = src_h

    def draw(self):
        canv = self.canv
        rl_obj = makerl(canv, self._xobj)
        sx = self.width / self._src_w
        sy = self.height / self._src_h
        canv.saveState()
        canv.transform(sx, 0, 0, sy, 0, 0)
        canv.doForm(rl_obj)
        canv.restoreState()


def _normalize_chart_bufs(chart_pdf_bufs):
    """统一为 list[bytes]（兼容单 buf、三元组、列表三种形式）。"""
    if chart_pdf_bufs is None:
        return []
    if isinstance(chart_pdf_bufs, io.BytesIO):
        chart_pdf_bufs = [chart_pdf_bufs]
    elif isinstance(chart_pdf_bufs, tuple):
        chart_pdf_bufs = [chart_pdf_bufs[2]]
    else:
        normalized = []
        for item in chart_pdf_bufs:
            normalized.append(item[2] if isinstance(item, tuple) else item)
        chart_pdf_bufs = normalized

    result = []
    for b in chart_pdf_bufs:
        if isinstance(b, io.BytesIO):
            b.seek(0)
            result.append(b.read())
        else:
            result.append(bytes(b))
    return result


YELLOW = colors.Color(1, 1, 0)   # #FFFF00


def _p_highlight(text, size=14, bold=True):
    """黄色背景、指定字号的段落（对应 word rPr shd #FFFF00）。"""
    style = ParagraphStyle(
        'highlight',
        fontName='Helvetica-Bold' if bold else 'Helvetica',
        fontSize=size,
        backColor=YELLOW,
        leading=size * 1.3,
        spaceBefore=0,
        spaceAfter=2,
    )
    return Paragraph(str(text) if text else '', style)


def _image_page(data, chart_pdf_bytes_list):
    """
    图表页元素：INFO16 超链接、16. Summary of Results 标题、
    monitor_set 黄色描述、各图表（每张图前可选 caption）。

    对应 word build_document 中第三页的逻辑。
    """
    el = [PageBreak()]

    # INFO16 超链接
    info16_url = data.get('info_url16',
        'http://mfgreports.ch.intel.com/WP_Documentation/Word_Template/Section_16.htm')
    el.append(_info_link(info16_url))
    el.append(Spacer(1, 2))

    # 16. Summary of Results
    el.append(_p('16. Summary of Results', bold=True))
    el.append(Spacer(1, 4))

    # monitor_set 黄色描述（与 word 版对应）
    monitor_set = data.get('monitor_set', '')
    if monitor_set:
        el.append(_p_highlight(monitor_set, size=14, bold=True))
        el.append(Spacer(1, 4))

    captions = data.get('chart_captions') or []

    for idx, pdf_bytes in enumerate(chart_pdf_bytes_list):
        # 每张图前的 caption（黄色背景 14pt）
        caption = captions[idx] if idx < len(captions) else ''
        if caption:
            el.append(_p_highlight(caption, size=14, bold=True))
            el.append(Spacer(1, 2))

        el.append(_PdfImageFlowable(pdf_bytes, width=CW))
        el.append(PageBreak())

    return el


def _last_page(data):
    """最后一页：Notes/Explanations、17. Conclusions、18. Recommendations。"""
    el = [PageBreak()]

    # ── tbl1：Notes/Explanations ──
    col_ws = [CW * 0.2, CW * 0.8]

    notes = data.get('notes', [{'index': '1', 'content': 'N/A'}])
    t1_data = [[_p('Notes/Explanations', bold=True), '']]
    for note in notes:
        t1_data.append([
            _p(str(note.get('index', '')), bold=True),
            _p(str(note.get('content', ''))),
        ])

    t1 = Table(t1_data, colWidths=col_ws,
               rowHeights=[0.7 * cm] + [0.7 * cm] * len(notes))
    t1.hAlign = 'LEFT'
    t1.setStyle(TableStyle(_tbl_defaults() + [
        ('SPAN', (0, 0), (1, 0)),
        ('BACKGROUND', (0, 1), (-1, -1), WHITE),
    ]))
    el.append(t1)
    el.append(Spacer(1, 4))

    # ── 17. Conclusions ──
    el.append(_p('17. Conclusions', bold=True))
    el.append(Spacer(1, 2))

    conclusions = data.get('conclusions',
        'Proposed set of control limits are meeting the accept criteria')
    t2 = Table([[_p(conclusions)]], colWidths=[CW], rowHeights=[0.7 * cm])
    t2.hAlign = 'LEFT'
    t2.setStyle(TableStyle(_tbl_defaults() + [
        ('BACKGROUND', (0, 0), (0, 0), WHITE),
    ]))
    el.append(t2)
    el.append(Spacer(1, 4))

    # ── 18. Recommendations ──
    el.append(_p('18. Recommendations', bold=True))
    el.append(Spacer(1, 2))

    recommendations = data.get('recommendations',
        'To implement revised control limits for all affected sites'
        ' (A01, A04, A06, A15, A48, A90) and products to improve PCS indicators.')
    t3 = Table([[_p(recommendations)]], colWidths=[CW], rowHeights=[0.7 * cm])
    t3.hAlign = 'LEFT'
    t3.setStyle(TableStyle(_tbl_defaults() + [
        ('BACKGROUND', (0, 0), (0, 0), WHITE),
    ]))
    el.append(t3)

    return el


def build_pdf(data, chart_pdf_bufs=None) -> bytes:
    """
    生成完整 PDF。

    参数
    ----
    data : dict
        文档内容数据字典。
    chart_pdf_bufs : (svg_buf, png_buf, pdf_buf) 或 list[...] 或 None
        draw_spc_chart 返回的三元组（取 pdf_buf）。
        传入时在第二页后插入图表页，不传则跳过图表页。
    """
    chart_bytes_list = _normalize_chart_bufs(chart_pdf_bufs)

    flowables = _first_page(data) + _second_page(data)
    if chart_bytes_list:
        flowables += _image_page(data, chart_bytes_list)
    flowables += _last_page(data)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=(PAGE_W, PAGE_H),
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN * 1.5,
    )
    doc.build(
        flowables,
        onFirstPage=_page_callback,
        onLaterPages=_page_callback,
    )
    return buf.getvalue()


# 向后兼容别名
build_pdf_with_charts = build_pdf
