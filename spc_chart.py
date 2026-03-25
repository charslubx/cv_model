import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _draw_summary_table(ax, table_rows):
    COLS = ['Reference', 'Total Lots', 'Lots Excluded', 'LCL', 'CL', 'UCL', 'OOC', 'CLSR', 'OCI', 'Process CPK']
    COL_W = [0.17, 0.09, 0.11, 0.09, 0.09, 0.09, 0.08, 0.10, 0.09, 0.09]
    OOC_IDX = 6
    CLSR_IDX = 7

    LEFT_PAD = -0.02
    col_x = [LEFT_PAD + sum(COL_W[:i]) for i in range(len(COL_W))]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    n = len(table_rows)
    row_h = 1.0 / (n + 1)
    hdr_y = 1.0 - row_h * 0.5

    def _hline(y, color='#CCCCCC', lw=0.8):
        ax.plot([LEFT_PAD, 1.0], [y, y], color=color, linewidth=lw, clip_on=False)

    _hline(1.0)

    for j, col in enumerate(COLS):
        ax.text(
            col_x[j], hdr_y, col,
            ha='left', va='center',
            fontsize=8, fontweight='bold', color='#333333', clip_on=False
        )

    # 表头下方分隔线
    _hline(1.0 - row_h)

    for i, row in enumerate(table_rows):
        row_y = 1.0 - row_h * (i + 1.5)
        cells = [
            row.get('ref', ''),
            str(row['total_lots']) if row.get('total_lots') is not None else '-',
            str(row['lots_excl']) if row.get('lots_excl') is not None else '-',
            str(row['lcl']) if row.get('lcl') is not None else '-',
            str(row['cl']) if row.get('cl') is not None else '-',
            str(row['ucl']) if row.get('ucl') is not None else '-',
            str(row['ooc']) if row.get('ooc') is not None else '-',
            str(row['clsr']) if row.get('clsr') is not None else '-',
            str(row['oci']) if row.get('oci') is not None else '-',
            str(row['cpk']) if row.get('cpk') is not None else '-',
        ]
        for j, val in enumerate(cells):
            red = row.get('highlight_ooc', False) and j in (OOC_IDX, CLSR_IDX)
            ax.text(col_x[j], row_y, val,
                    ha='left', va='center',
                    fontsize=8, color='#E53935' if red else '#555555', clip_on=False)

        # 只画最后一行
        if i == n - 1:
            _hline(1.0 - row_h * (i + 2))


def draw_spc_chart(
    x_labels, y_values, cl, user_cl, ucl=None, lcl=None, user_ucl=None, user_lcl=None,
    table_rows=None, chart_title='',
    fig_width_inch=14, fig_height_inch=5.5, dpi=600
):
    TABLE_H = 1.4

    if table_rows:
        chart_h = fig_height_inch - TABLE_H
        fig = plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
        gs = gridspec.GridSpec(
            2, 1, figure=fig,
            height_ratios=[TABLE_H, chart_h],
            hspace=0.18,
            left=0.05, right=0.97, top=0.97
        )
        ax_tbl = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])
        _draw_summary_table(ax_tbl, table_rows)
    else:
        fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)

    if chart_title:
        ax.text(
            -0.02, 1.0, chart_title,
            ha='left', va='bottom',
            fontsize=9, fontweight='bold', color='#333333',
            transform=ax.transAxes, clip_on=False
        )

    # ── 一对多 x-y：同一 x label 的所有 y 值落在同一 x 位置 ──
    unique_labels = list(dict.fromkeys(x_labels))
    label_to_pos = {lbl: i for i, lbl in enumerate(unique_labels)}
    x_pos = np.array([label_to_pos[lbl] for lbl in x_labels])
    y_arr = np.asarray(y_values, dtype=float)

    out = np.zeros(len(y_arr), dtype=bool)
    if ucl is not None:
        out |= (y_arr > ucl)
    if lcl is not None:
        out |= (y_arr < lcl)

    ax.scatter(x_pos[~out], y_arr[~out], color='#3399FF', s=18, zorder=3, linewidths=0)
    if out.any():
        ax.scatter(x_pos[out], y_arr[out], color='#E53935', s=22, zorder=4, linewidths=0)

    y_span = float(y_arr.max() - y_arr.min()) if y_arr.max() > y_arr.min() else 1.0
    offset = y_span * 0.01
    x_right = len(unique_labels)

    def _line(y_val, color, ls, lw, label):
        ax.axhline(y_val, color=color, linewidth=lw, linestyle=ls, zorder=2)
        ax.text(
            x_right, y_val + offset, label,
            ha='right', va='bottom', fontsize=7,
            color='#000000', clip_on=False
        )

    _line(cl, '#FF8C00', '--', 1.5, 'Online CL')
    if ucl is not None:
        _line(ucl, '#FF8C00', '--', 1.2, 'Online UCL')
    if lcl is not None:
        _line(lcl, '#FF8C00', '--', 1.2, 'Online LCL')

    _line(user_cl, '#2E7D32', '-', 1.5, 'User Defined CL')
    if user_ucl is not None:
        _line(user_ucl, '#2E7D32', '--', 1.2, 'User Defined UCL')
    if user_lcl is not None:
        _line(user_lcl, '#2E7D32', '--', 1.2, 'User Defined LCL')

    step = max(1, len(unique_labels) // 80)
    ax.set_xticks(range(0, len(unique_labels), step))
    ax.set_xticklabels(
        [unique_labels[i] for i in range(0, len(unique_labels), step)],
        rotation=90, fontsize=5, color='#999b9e'
    )
    ax.tick_params(axis='x', length=2, pad=1)
    ax.set_xlabel('LOT', fontsize=8, labelpad=4)

    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', labelsize=5, colors='#999b9e')
    ax.set_ylabel('VAL', fontsize=8, rotation=90, labelpad=6)

    ax.set_facecolor('#FFFFFF')
    fig.patch.set_facecolor('#FFFFFF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, color='#CCCCCC', alpha=0.7)
    ax.set_xlim(-1, len(unique_labels))

    # 将所有控制线的值都纳入范围计算，避免控制线超出 y 轴边界
    all_bounds = list(y_arr)
    for v in [ucl, lcl, user_ucl, user_lcl, cl, user_cl]:
        if v is not None:
            all_bounds.append(v)
    y_min, y_max = min(all_bounds), max(all_bounds)
    pad = (y_max - y_min) * 0.04
    ax.set_ylim(y_min - pad, y_max + pad)

    buf = io.BytesIO()
    fig.savefig(buf, format='svg')
    plt.close(fig)
    buf.seek(0)
    return buf


if __name__ == '__main__':
    np.random.seed(42)

    # 一对多示例：每个 LOT 有 3 个测量值
    lots = ['LOT{:04d}'.format(i) for i in range(100)]
    x_labels = [lot for lot in lots for _ in range(3)]   # 每个 LOT 重复 3 次
    y_values = np.random.normal(loc=-12.0, scale=0.5, size=len(x_labels))
    y_values[50]  = -10.2
    y_values[120] = -14.8
    y_values[200] = -9.8

    table_rows = [
        {'ref': 'Online', 'total_lots': 300, 'lots_excl': 0,
         'lcl': -14.5, 'cl': -12.0, 'ucl': -10.5,
         'ooc': 3, 'clsr': '1.00%', 'oci': 0.52, 'highlight_ooc': True},
        {'ref': 'Suggested', 'total_lots': 300, 'lots_excl': 2,
         'lcl': -14.8, 'cl': -12.1, 'ucl': -10.2,
         'ooc': 1, 'clsr': '0.33%', 'oci': 0.48, 'highlight_ooc': False},
        {'ref': 'user defined', 'total_lots': None, 'lots_excl': None,
         'lcl': None, 'cl': None, 'ucl': None,
         'ooc': None, 'clsr': None, 'oci': None, 'highlight_ooc': False},
    ]

    buf = draw_spc_chart(
        x_labels=x_labels,
        y_values=y_values,
        cl=-12.0, ucl=-10.5, lcl=-14.5,
        user_cl=-10.0, user_ucl=-9.5, user_lcl=-13.5,
        table_rows=table_rows,
        chart_title='Control Chart',
    )

    with open('spc_chart_demo.svg', 'wb') as f:
        f.write(buf.read())
    print('已保存 spc_chart_demo.svg')
