import io
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches, Emu
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.part import Part
from docx.opc.packuri import PackURI
from docx.oxml.ns import qn, nsmap
from docx.oxml import OxmlElement

# SVG 关系类型（与 PNG/JPEG 相同，区别在于 content-type）
_RT_IMAGE = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image'
# SVG blip 扩展 URI（Office 2016+）
_SVG_BLIP_URI = '{96DAC541-7B7A-43D3-8B79-37D633B846F1}'
# 必须同时存在的第一个 ext，标记"使用本地 DPI"
_USE_LOCAL_DPI_URI = '{28A0092B-C50C-407E-A947-70E740481C1C}'


def _set_run_font(run, font_name='Aptos', size_pt=10, bold=False, color=None):
    """统一设置字体（Aptos 需要直接写 XML，python-docx 的 font.name 只写 ascii）"""
    run.bold = bold
    run.font.size = Pt(size_pt)
    if color:
        run.font.color.rgb = color

    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    for attr in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        rFonts.set(qn(attr), font_name)


def _set_page_background(doc, r, g, b):
    """设置页面背景色，并在 settings 中启用显示"""
    hex_color = '{:02X}{:02X}{:02X}'.format(r, g, b)

    doc_elem = doc.element
    body = doc_elem.find(qn('w:body'))

    old_bg = doc_elem.find(qn('w:background'))
    if old_bg is not None:
        doc_elem.remove(old_bg)

    bg = OxmlElement('w:background')
    bg.set(qn('w:color'), hex_color)
    doc_elem.insert(list(doc_elem).index(body), bg)

    settings_elem = doc.settings.element
    if settings_elem.find(qn('w:displayBackgroundShape')) is None:
        dbs = OxmlElement('w:displayBackgroundShape')
        settings_elem.append(dbs)


def _set_cell_shading(cell, fill_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex.lstrip('#'))
    tc_pr.append(shd)


def _set_row_height(row, height_cm):
    tr_pr = row._tr.get_or_add_trPr()
    trH = OxmlElement('w:trHeight')
    trH.set(qn('w:val'), str(int(height_cm * 567)))
    trH.set(qn('w:hRule'), 'atLeast')
    tr_pr.append(trH)


def _cell_write(cell, text, valign='bottom', font_name='Aptos', size_pt=9, bold=False, color=None):
    """写入单元格文字：左对齐 + 垂直靠下"""
    _set_cell_valign(cell, valign)

    cell.text = ''
    para = cell.paragraphs[0]
    para.alignment = 0  # LEFT

    run = para.add_run(text)
    _set_run_font(run, font_name=font_name, size_pt=size_pt, bold=bold, color=color)
    return para


def _set_cell_valign(cell, align='bottom'):
    tc_pr = cell._tc.get_or_add_tcPr()
    v_align = tc_pr.find(qn('w:vAlign'))
    if v_align is None:
        v_align = OxmlElement('w:vAlign')
        tc_pr.append(v_align)
    v_align.set(qn('w:val'), align)


def _add_info_hyperlink(doc, url='https://example.com/info'):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = 0

    r_id = p.part.relate_to(url, RT.HYPERLINK, is_external=True)

    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)

    wr = OxmlElement('w:r')

    rPr = OxmlElement('w:rPr')

    rFonts = OxmlElement('w:rFonts')
    for attr in ('w:ascii', 'w:hAnsi', 'w:cs'):
        rFonts.set(qn(attr), 'Aptos')
    rPr.append(rFonts)

    b = OxmlElement('w:b')
    rPr.append(b)

    sz = OxmlElement('w:sz')
    sz.set(qn('w:val'), '32')  # 11pt × 2 = 22 half-points, but original uses 32
    rPr.append(sz)

    clr = OxmlElement('w:color')
    clr.set(qn('w:val'), '286EC8')
    rPr.append(clr)

    u = OxmlElement('w:u')
    u.set(qn('w:val'), 'single')
    rPr.append(u)

    wr.append(rPr)

    t = OxmlElement('w:t')
    t.text = 'INFO'
    wr.append(t)
    hyperlink.append(wr)
    p._p.append(hyperlink)

    return p


def _add_section_note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    _set_run_font(run, size_pt=11, bold=True)
    return p


def _set_cell_borders(cell, left='none', right='none', top='single', bottom='single', sz='4', color='auto'):
    """设置单元格边框"""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = OxmlElement('w:tcBorders')
    for edge, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        b = OxmlElement('w:{}'.format(edge))
        b.set(qn('w:val'), val)
        b.set(qn('w:sz'), sz)
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), color)
        tc_borders.append(b)
    tc_pr.append(tc_borders)


def _add_plain_text_cc(cell, placeholder='Click or tap here to enter text.', tag='', font_name='Aptos', size_pt=9):
    tc = cell._tc

    for p_elem in list(tc.findall(qn('w:p'))):
        tc.remove(p_elem)

    sdt = OxmlElement('w:sdt')

    sdtPr = OxmlElement('w:sdtPr')

    if tag:
        tag_el = OxmlElement('w:tag')
        tag_el.set(qn('w:val'), tag)
        sdtPr.append(tag_el)

    sdtPr.append(OxmlElement('w:showingPlcHdr'))
    sdtPr.append(OxmlElement('w:text'))

    sdt.append(sdtPr)

    sdtContent = OxmlElement('w:sdtContent')

    p = OxmlElement('w:p')

    pPr = OxmlElement('w:pPr')
    jc = OxmlElement('w:jc')
    jc.set(qn('w:val'), 'left')
    pPr.append(jc)
    p.append(pPr)

    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')

    rStyle = OxmlElement('w:rStyle')
    rStyle.set(qn('w:val'), 'PlaceholderText')
    rPr.append(rStyle)

    rFonts = OxmlElement('w:rFonts')
    for attr in ('w:ascii', 'w:hAnsi', 'w:cs'):
        rFonts.set(qn(attr), font_name)
    rPr.append(rFonts)

    sz = OxmlElement('w:sz')
    sz.set(qn('w:val'), str(size_pt * 2))
    rPr.append(sz)

    r.append(rPr)

    t = OxmlElement('w:t')
    t.text = placeholder
    r.append(t)
    p.append(r)
    sdtContent.append(p)

    sdt.append(sdtContent)
    tc.append(sdt)

    _set_cell_valign(cell, 'bottom')


def _set_table_indent(tbl, indent_cm):
    tbl_elem = tbl._tbl

    tbl_pr = tbl_elem.find(qn('w:tblPr'))
    if tbl_pr is None:
        tbl_pr = OxmlElement('w:tblPr')
        tbl_elem.insert(0, tbl_pr)

    existing = tbl_pr.find(qn('w:tblInd'))
    if existing is not None:
        tbl_pr.remove(existing)

    tbl_ind = OxmlElement('w:tblInd')
    tbl_ind.set(qn('w:w'), str(int(indent_cm * 567)))
    tbl_ind.set(qn('w:type'), 'dxa')
    tbl_pr.append(tbl_ind)


def _set_table_total_width(tbl, width, _type="inch"):
    tbl_elem = tbl._tbl
    tbl_pr = tbl_elem.find(qn('w:tblPr'))
    if tbl_pr is None:
        tbl_pr = OxmlElement('w:tblPr')
        tbl_elem.insert(0, tbl_pr)

    old = tbl_pr.find(qn('w:tblW'))
    if old is not None:
        tbl_pr.remove(old)

    tbl_w = OxmlElement('w:tblW')
    tbl_w.set(qn('w:w'), str(int(width * 1440)) if _type == "inch" else str(int(width * 567)))
    tbl_w.set(qn('w:type'), 'dxa')
    tbl_pr.append(tbl_w)


def _build_spc_table(doc, data, page_w_cm):
    """两行表头"""
    DARK_BG = '2B3D28'
    PINK_BG = '77206d'
    GREEN_BG = '3a7c22'
    DARK_ROW = 'd1d1d1'
    PINK_ROW = 'f2ceed'
    GREEN_ROW = 'd9f2d0'
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)

    fixed_headers = ['SPC area', 'Monitor set', 'Measurement set', 'Chart type', 'Control Limit Type']
    value_sub = ['LCL', 'CL', 'UCL', 'OCI', '%OOC']

    fw_vals = [2, 2.5, 3.5, 1.6, 2.5]
    fw = [Cm(v) for v in fw_vals]
    vw = Cm((page_w_cm - sum(fw_vals)) / 10)

    spc_rows = data.get('spc_rows', [{}])
    tbl = doc.add_table(rows=2 + len(spc_rows), cols=15)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_table_total_width(tbl, 13.06)

    _set_row_height(tbl.rows[0], 0.22 * 2.54)
    for j, txt in enumerate(fixed_headers):
        merged = tbl.rows[0].cells[j].merge(tbl.rows[1].cells[j])
        merged.width = fw[j]
        _set_cell_shading(merged, DARK_BG)
        _set_cell_valign(merged, 'center')
        p = merged.paragraphs[0]
        p.alignment = 1
        run = p.add_run(txt)
        _set_run_font(run, size_pt=10, bold=True, color=WHITE)

    pv = tbl.rows[0].cells[5].merge(tbl.rows[0].cells[9])
    _set_cell_shading(pv, PINK_BG)
    _set_cell_valign(pv, 'center')
    p = pv.paragraphs[0]
    p.alignment = 1
    run = p.add_run('Present value')
    _set_run_font(run, size_pt=10, bold=True, color=WHITE)

    prv = tbl.rows[0].cells[10].merge(tbl.rows[0].cells[14])
    _set_cell_shading(prv, GREEN_BG)
    _set_cell_valign(prv, 'center')
    p = prv.paragraphs[0]
    p.alignment = 1
    run = p.add_run('Proposed value')
    _set_run_font(run, size_pt=10, bold=True, color=WHITE)

    _set_row_height(tbl.rows[1], 0.22 * 2.54)
    for k, txt in enumerate(value_sub):
        for bg, offset in [(PINK_BG, 5), (GREEN_BG, 10)]:
            c = tbl.rows[1].cells[offset + k]
            c.width = vw
            _set_cell_shading(c, bg)
            _set_cell_valign(c, 'center')
            p = c.paragraphs[0]
            p.alignment = 1
            run = p.add_run(txt)
            _set_run_font(run, size_pt=10, color=WHITE)

    keys_fixed = ['spc_area', 'monitor_set', 'measurement_set', 'chart_type', 'control_limit_type']
    keys_val = ['lcl', 'cl', 'ucl', 'oci', 'ooc']

    for i, rd in enumerate(spc_rows):
        row = tbl.rows[2 + i]
        _set_row_height(row, 0.24 * 2.54)

        for j, key in enumerate(keys_fixed):
            c = row.cells[j]
            c.width = fw[j]
            _cell_write(c, rd.get(key, ''), size_pt=10)
            _set_cell_shading(c, DARK_ROW)

        for k, key in enumerate(keys_val):
            cp = row.cells[5 + k]
            cp.width = vw
            _cell_write(cp, rd.get('present_' + key, ''), size_pt=10)
            _set_cell_shading(cp, PINK_ROW)

            cq = row.cells[10 + k]
            cq.width = vw
            _cell_write(cq, rd.get('proposed_' + key, ''), size_pt=10)
            _set_cell_shading(cq, GREEN_ROW)

    return tbl


def _add_source_ref_table(doc, value, page_w_cm=25.7):
    """左侧单元格无上下左边框，仅保留右边框"""
    label_w = Cm(2.06 * 2.54)
    value_w = Cm(page_w_cm - label_w)

    tbl = doc.add_table(rows=1, cols=2)
    tbl.style = 'Table Grid'
    tbl.autofit = False
    _set_row_height(tbl.rows[0], 0.7)

    c0 = tbl.rows[0].cells[0]
    c0.width = label_w
    _cell_write(c0, 'Source of Reference Data', size_pt=11, bold=True)
    _set_cell_borders(c0, left='none', top='none', bottom='none', right='single')

    c1 = tbl.rows[0].cells[1]
    c1.width = value_w
    _cell_write(c1, value, size_pt=11)
    _set_cell_shading(c1, '#FFFFFF')

    return tbl


def _build_second_page(doc, data, page_w):
    _build_spc_table(doc, data, page_w)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(5)
    sp.paragraph_format.space_after = Pt(0)

    _add_source_ref_table(
        doc,
        value=data.get('source_ref', 'Catalyst PCS SPC++, filter key ({Module} Module, {function_area}, WW46\u201925 \u2013 WW04\u201926).'),
        page_w_cm=page_w
    )


def _build_first_page(doc, data, page_w):
    label_w = Cm(5.0)
    value_w = Cm(page_w - 5.0)

    tp = doc.add_paragraph()
    tp.alignment = 1
    tp.paragraph_format.space_after = Pt(2)
    r = tp.add_run('AT INTEGRATED CCB WP TEMPLATE')
    _set_run_font(r, size_pt=16, bold=True)

    sp = doc.add_paragraph()
    sp.alignment = 1
    sp.paragraph_format.space_after = Pt(8)
    r2 = sp.add_run(
        "AS CORPORATE DIRECTIVE PROTECT YOUR CHANGE WITH ERM CREDENTIALS FOR ATTD OR STTD CCB RELATED WPS ONLY"
    )
    _set_run_font(r2, size_pt=9, bold=True)

    _add_info_hyperlink(doc, url=data.get('info_url', 'http://mfgreports.ch.intel.com/WP_Documentation/Word_Template/Section_1-6.htm'))

    tbl1 = doc.add_table(rows=7, cols=2)
    _set_table_indent(tbl1, 0.1 * 2.54)
    tbl1.style = 'Table Grid'
    tbl1.autofit = False
    for index, value in enumerate([
        ('1. White Paper Type', data.get('white_paper_type', '')),
        ('2. Classification', data.get('classification', '')),
        ('DOE sections (9, 10, 11, 13, 16) are optional for class IV if the changes do not require experiment data collection', ''),
        ('3. Name of Owner', data.get('owner_name', '')),
        ('4. Title of Change', data.get('title_of_change', '')),
        ('5. Change Description', data.get('change_description', '')),
        ('6. Reason for Change', data.get('reason_for_change', '')),
    ]):
        lbl, val = value
        if index == 2:
            note_row = tbl1.rows[2]
            _set_row_height(note_row, 1)
            note_cell = note_row.cells[0].merge(note_row.cells[1])
            note_cell.width = Cm(page_w)
            _cell_write(note_cell, lbl, size_pt=11, bold=True, valign='center')
            _set_cell_borders(note_cell, left='none', right='none', top='single', bottom='single')
        else:
            row = tbl1.rows[index]
            _set_row_height(row, 1.3 if index > 2 else 0.8)
            c0, c1 = row.cells
            c0.width = label_w
            c1.width = value_w
            _cell_write(c0, lbl, size_pt=11, bold=True)
            _cell_write(c1, val, size_pt=11)
            _set_cell_shading(c1, '#FFFFFF')

    blank_pr = doc.add_paragraph()
    blank_pr.paragraph_format.space_before = Pt(0)
    blank_pr.paragraph_format.space_after = Pt(0)
    _add_info_hyperlink(doc, url=data.get('info_url', 'http://mfgreports.ch.intel.com/WP_Documentation/Word_Template/Section_7.htm'))
    blank_pr = doc.add_paragraph()
    blank_pr.paragraph_format.space_before = Pt(0)
    blank_pr.paragraph_format.space_after = Pt(0)

    pf_p = doc.add_paragraph()
    pf_p.paragraph_format.space_before = Pt(0)
    pf_p.paragraph_format.space_after = Pt(2)
    r = pf_p.add_run('7. Process Factors')
    _set_run_font(r, size_pt=11, bold=True)

    factors = data.get('process_factors', [{'factor': '', 'present': '', 'proposed': ''}])
    tbl3 = doc.add_table(rows=1 + len(factors), cols=3)
    _set_table_indent(tbl3, 0.1 * 2.54)
    tbl3.style = 'Table Grid'
    tbl3.autofit = False

    col_ws = [Cm(7.7724), Cm(7.7724), Cm(7.7724)]
    headers = ['Process Factor', 'Present Value', 'Proposed Value']

    hdr = tbl3.rows[0]
    _set_row_height(hdr, 0.7)
    for j, (txt, w) in enumerate(zip(headers, col_ws)):
        hdr.cells[j].width = w
        _cell_write(hdr.cells[j], txt, size_pt=11, bold=True)

    for i, factor in enumerate(factors):
        dr = tbl3.rows[i + 1]
        _set_row_height(dr, 0.8)
        for j, key in enumerate(['factor', 'present', 'proposed']):
            cell = dr.cells[j]
            cell.width = col_ws[j]
            val = factor.get(key, '').strip()

            if val:
                _cell_write(cell, val, size_pt=11)
            else:
                _add_plain_text_cc(
                    cell,
                    placeholder='Please refer to Table below',
                    tag='{}_row{}'.format(key, i),
                    size_pt=11
                )
            _set_cell_shading(cell, '#FFFFFF')


def _png_dimensions(png_bytes):
    """从 PNG 二进制数据中解析宽度和高度，无需第三方库。

    PNG 规范：文件头 8 字节签名 + IHDR chunk（4字节长度 + 4字节类型 + 13字节数据），
    IHDR 数据前 4 字节为宽度，后 4 字节为高度，均为大端序 uint32。
    """
    import struct
    # PNG 签名 8 字节，chunk 长度 4 字节，chunk 类型 4 字节，然后是 IHDR 数据
    w, h = struct.unpack('>II', png_bytes[16:24])
    return w, h


def _add_svg_inline(doc_part, svg_bytes, png_bytes, width_emu, height_emu, shape_id, img_name):
    """
    构造一个 w:drawing/wp:inline 元素，以 SVG 作为矢量主体，PNG 作为降级备用。

    Office 2016+ 通过 a:blip 上的扩展节点 (a:ext[@uri=SVG_BLIP_URI]/asvg:svgBlip)
    识别矢量图；旧版 Word 回退到 a:blip[@r:embed=png_rId] 中的 PNG。

    返回构造好的 lxml Element（w:drawing）。
    """
    # ── 1. 向 document part 注册 PNG part（降级用）──
    png_partname = PackURI('/word/media/{}.png'.format(img_name))
    png_part = Part(png_partname, 'image/png', png_bytes)
    png_rId = doc_part.relate_to(png_part, _RT_IMAGE)

    # ── 2. 向 document part 注册 SVG part ──
    svg_partname = PackURI('/word/media/{}.svg'.format(img_name))
    svg_part = Part(svg_partname, 'image/svg+xml', svg_bytes)
    svg_rId = doc_part.relate_to(svg_part, _RT_IMAGE)

    # ── 3. 构造 XML ──
    # 命名空间前缀
    NS_A    = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    NS_PIC  = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
    NS_WP   = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
    NS_R    = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    NS_ASVG = 'http://schemas.microsoft.com/office/drawing/2016/SVG/main'

    def _el(tag, ns=None):
        return OxmlElement(tag) if ns is None else OxmlElement('{%s}%s' % (ns, tag.split(':')[-1]) if ':' in tag else tag)

    def qn2(prefix, local):
        _ns = {
            'a': NS_A, 'pic': NS_PIC, 'wp': NS_WP,
            'r': NS_R, 'asvg': NS_ASVG,
        }
        return '{%s}%s' % (_ns[prefix], local)

    from lxml import etree as _etree

    cx_str = str(width_emu)
    cy_str = str(height_emu)

    # w:drawing
    drawing = OxmlElement('w:drawing')

    # wp:inline
    inline = OxmlElement('wp:inline')
    inline.set('distT', '0'); inline.set('distB', '0')
    inline.set('distL', '0'); inline.set('distR', '0')
    drawing.append(inline)

    # wp:extent
    extent = OxmlElement('wp:extent')
    extent.set('cx', cx_str); extent.set('cy', cy_str)
    inline.append(extent)

    # wp:effectExtent
    ee = OxmlElement('wp:effectExtent')
    for attr in ('l', 't', 'r', 'b'):
        ee.set(attr, '0')
    inline.append(ee)

    # wp:docPr
    docPr = OxmlElement('wp:docPr')
    docPr.set('id', str(shape_id)); docPr.set('name', img_name)
    inline.append(docPr)

    # wp:cNvGraphicFramePr / a:graphicFrameLocks
    cNvGfxPr = OxmlElement('wp:cNvGraphicFramePr')
    gfl = OxmlElement('a:graphicFrameLocks')
    gfl.set('noChangeAspect', '1')
    cNvGfxPr.append(gfl)
    inline.append(cNvGfxPr)

    # a:graphic / a:graphicData
    graphic = OxmlElement('a:graphic')
    inline.append(graphic)

    gdata = OxmlElement('a:graphicData')
    gdata.set('uri', NS_PIC)
    graphic.append(gdata)

    # pic:pic
    pic = OxmlElement('pic:pic')
    gdata.append(pic)

    # pic:nvPicPr
    nvPicPr = OxmlElement('pic:nvPicPr')
    pic.append(nvPicPr)

    nvDp = OxmlElement('pic:cNvPr')
    nvDp.set('id', '0'); nvDp.set('name', img_name)
    nvPicPr.append(nvDp)

    nvPicDp = OxmlElement('pic:cNvPicPr')
    nvPicPr.append(nvPicDp)

    # pic:blipFill
    blipFill = OxmlElement('pic:blipFill')
    pic.append(blipFill)

    # a:blip  — r:embed 指向 PNG（降级），扩展节点指向 SVG（矢量）
    blip = OxmlElement('a:blip')
    blip.set(qn2('r', 'embed'), png_rId)
    blipFill.append(blip)

    # a:blip/a:extLst
    extLst = OxmlElement('a:extLst')
    blip.append(extLst)

    # 第一个 ext：useLocalDpi（必须存在，否则 Word 不渲染 SVG）
    ext1 = OxmlElement('a:ext')
    ext1.set('uri', _USE_LOCAL_DPI_URI)
    NS_A14 = 'http://schemas.microsoft.com/office/drawing/2010/main'
    a14_useLocalDpi = _etree.SubElement(ext1, '{%s}useLocalDpi' % NS_A14)
    a14_useLocalDpi.set('val', '0')
    extLst.append(ext1)

    # 第二个 ext：svgBlip（指向 SVG part）
    ext2 = OxmlElement('a:ext')
    ext2.set('uri', _SVG_BLIP_URI)
    svgBlip = _etree.SubElement(ext2, '{%s}svgBlip' % NS_ASVG)
    svgBlip.set(qn2('r', 'embed'), svg_rId)
    extLst.append(ext2)

    # a:stretch / a:fillRect
    stretch = OxmlElement('a:stretch')
    stretch.append(OxmlElement('a:fillRect'))
    blipFill.append(stretch)

    # pic:spPr
    spPr = OxmlElement('pic:spPr')
    pic.append(spPr)

    xfrm = OxmlElement('a:xfrm')
    off = OxmlElement('a:off'); off.set('x', '0'); off.set('y', '0')
    ext = OxmlElement('a:ext'); ext.set('cx', cx_str); ext.set('cy', cy_str)
    xfrm.append(off); xfrm.append(ext)
    spPr.append(xfrm)

    prstGeom = OxmlElement('a:prstGeom')
    prstGeom.set('prst', 'rect')
    prstGeom.append(OxmlElement('a:avLst'))
    spPr.append(prstGeom)

    return drawing


def _build_third_page(doc, chart_bufs, page_w_cm):
    """
    将一个或多个图表以 SVG 矢量图插入第三页，同时内嵌 PNG 供旧版 Word 降级。

    Word 2016+ 直接渲染 SVG 矢量；旧版 Word 自动回退到 PNG 位图。
    所有内容均嵌入 docx 包内，无需任何系统级图形库（不依赖 cairosvg）。

    参数
    ----
    chart_bufs : (svg_buf, png_buf) 或 list[(svg_buf, png_buf)]
        draw_spc_chart 返回的 (svg_buf, png_buf) 元组，可以是单个也可以是列表。
    page_w_cm : float
        可用页宽（厘米），图片宽度撑满此值，高度按 PNG 宽高比等比缩放。
    """
    if isinstance(chart_bufs, tuple):
        chart_bufs = [chart_bufs]

    page_w_emu = int(page_w_cm / 2.54 * 914400)  # cm → EMU

    doc_part = doc.part

    used_ids = [int(s) for s in doc.element.xpath('//@id') if s.isdigit()]
    next_shape_id = (max(used_ids) + 1) if used_ids else 1

    # 兼容 (svg, png) 和 (svg, png, pdf) 两种元组长度
    for idx, bufs in enumerate(chart_bufs):
        svg_buf, png_buf = bufs[0], bufs[1]
        svg_buf.seek(0)
        png_buf.seek(0)
        svg_bytes = svg_buf.read()
        png_bytes = png_buf.read()

        # 从 PNG 读取宽高比，计算 EMU 高度（无需 cairosvg，用标准库解析 PNG 头）
        px_w, px_h = _png_dimensions(png_bytes)
        aspect = px_h / px_w if px_w else 1.0
        page_h_emu = int(page_w_emu * aspect)

        img_name = 'spc_chart_{}'.format(idx)

        drawing_el = _add_svg_inline(
            doc_part, svg_bytes, png_bytes,
            page_w_emu, page_h_emu,
            next_shape_id + idx, img_name
        )

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        p.alignment = 1

        run = p.add_run()
        run._r.append(drawing_el)


def _apply_doc_settings(doc):
    """页面通用设置（页边距、背景色、字体等）"""
    sec = doc.sections[0]
    sec.page_width = Cm(35.56)
    sec.page_height = Cm(21.59)
    sec.left_margin = Cm(1.27)
    sec.right_margin = Cm(1.27)
    sec.top_margin = Cm(1.27)
    sec.bottom_margin = Cm(1.27)

    _set_page_background(doc, 235, 235, 249)

    fp = doc.sections[0].footer.paragraphs[0]
    fp.clear()
    r = fp.add_run('Rev 2.0')
    fp.alignment = 0
    fp.paragraph_format.space_before = Pt(0)
    fp.paragraph_format.space_after = Pt(0)
    _set_run_font(r, size_pt=11)


def build_document(data, spc_svg_bufs=None) -> bytes:
    """
    生成完整文档（第一页 + 第二页 + 可选第三页）

    参数
    ----
    data : dict
        文档内容数据字典。
    spc_svg_bufs : (svg_buf, png_buf, pdf_buf) 或 list[...] 或 None
        draw_spc_chart 返回的三元组，docx 只使用前两个（svg/png）。
        传入时会在第三页插入对应图表，不传则不生成第三页。
    """
    doc = Document()
    page_w = 35.56 - 1.27 - 1.27
    _apply_doc_settings(doc)

    _build_first_page(doc, data, page_w)

    doc.add_page_break()

    _build_second_page(doc, data, page_w)

    if spc_svg_bufs is not None:
        doc.add_page_break()
        _build_third_page(doc, spc_svg_bufs, page_w)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
