import time, os, sys, numpy as np
from chart_renderer import render_charts_parallel, render_charts_sequential

def make_params(seed):
    np.random.seed(seed)
    lots = ['LOT{:04d}'.format(i) for i in range(80)]
    x_labels = [lot for lot in lots for _ in range(3)]
    y_values = np.random.normal(loc=-12.0, scale=0.5, size=len(x_labels))
    table_rows = [
        {'ref': 'Online', 'total_lots': 240, 'lots_excl': 0, 'lcl': -14.5, 'cl': -12.0,
         'ucl': -10.5, 'ooc': 2, 'clsr': '0.8%', 'oci': 0.52, 'highlight_ooc': True, 'cpk': None},
        {'ref': 'user defined', 'total_lots': None, 'lots_excl': None, 'lcl': None, 'cl': None,
         'ucl': None, 'ooc': None, 'clsr': None, 'oci': None, 'highlight_ooc': False, 'cpk': None},
    ]
    return dict(x_labels=x_labels, y_values=y_values, cl=-12.0, ucl=-10.5, lcl=-14.5,
                user_cl=-10.0, user_ucl=-9.5, user_lcl=-13.5,
                table_rows=table_rows, chart_title=f'Chart #{seed}')

if __name__ == '__main__':
    N = 4
    params = [make_params(i) for i in range(N)]
    ctx = 'fork' if sys.platform != 'win32' else 'spawn'
    print(f'CPU 核数: {os.cpu_count()}，渲染 {N} 张图，multiprocessing context: {ctx}')

    t0 = time.perf_counter()
    seq_results = render_charts_sequential(params)
    t_seq = time.perf_counter() - t0
    print(f'串行: {t_seq:.2f}s')

    t0 = time.perf_counter()
    par_results = render_charts_parallel(params)
    t_par = time.perf_counter() - t0
    print(f'并行: {t_par:.2f}s  加速比: {t_seq/t_par:.1f}x')

    for i, (seq, par) in enumerate(zip(seq_results, par_results)):
        sv = seq[0].read() if hasattr(seq[0], 'read') else seq[0]
        pv = par[0].read() if hasattr(par[0], 'read') else par[0]
        print(f'  图{i} SVG size: seq={len(sv):,}  par={len(pv):,}  match={len(sv)==len(pv)}')
