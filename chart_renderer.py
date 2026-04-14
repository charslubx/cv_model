"""
并行图表渲染模块

核心思路
--------
matplotlib 渲染是纯 CPU 密集型操作，Python GIL 使多线程无法并行执行
CPU 密集任务，因此使用 multiprocessing.Pool（进程池）绕过 GIL。

每个子进程独立运行 draw_spc_chart，各自占用一个 CPU 核心，互不阻塞。
渲染完成后把 (svg_bytes, png_bytes, pdf_bytes) 以普通 bytes 返回主进程，
不需要任何共享内存或锁。

典型加速比：N 张图 / min(N, CPU核数) ≈ 线性加速，直到核数饱和。

使用方法
--------
    from chart_renderer import render_charts_parallel

    chart_params = [
        dict(x_labels=..., y_values=..., cl=..., user_cl=..., ...),
        dict(x_labels=..., y_values=..., cl=..., user_cl=..., ...),
    ]
    results = render_charts_parallel(chart_params)
    # results[i] = (svg_bytes, png_bytes, pdf_bytes)

注意事项
--------
- Linux/macOS 自动使用 'fork'：子进程复制父进程内存，matplotlib/numpy
  已加载，无需重新 import，启动开销极低（< 5ms）。
- Windows 自动使用 'spawn'：os.fork 不存在，必须重新启动解释器，
  每个子进程需重新 import，启动开销约 1~2 秒。
- 通过 mp.get_context() 指定，不调用 set_start_method()，不影响全局状态。
- 子进程数量默认 = os.cpu_count()，可通过 max_workers 参数限制，
  避免在多租户服务器上占满所有核心。
"""

import io
import os
import multiprocessing as mp
from typing import List, Dict, Any, Tuple


def _worker_init():
    """
    spawn 子进程初始化：Windows 上每个 worker 从零启动，需要先初始化
    Django app registry，否则任何间接 import Django model 的模块都会抛
    AppRegistryNotReady。
    fork 子进程直接复制父进程内存，registry 已就绪，此函数是空操作。
    """
    try:
        import django
        django.setup()
    except RuntimeError:
        # setup() 在同一进程内被重复调用时抛 RuntimeError，忽略即可
        pass


def _render_one(params: Dict[str, Any]) -> Tuple[bytes, bytes, bytes]:
    """
    子进程入口：渲染单张图表，返回 (svg_bytes, png_bytes, pdf_bytes)。

    必须是模块级函数且所在模块不能属于任何 Django app 包，否则 spawn
    反序列化函数引用时会触发 Django model import → AppRegistryNotReady。

    draw_spc_chart 的来源通过 params 里的两个保留键指定：
      _draw_module : str  模块路径，默认 'spc_chart'
      _draw_func   : str  函数名，  默认 'draw_spc_chart'
    这两个键在传给 draw_spc_chart 之前会被弹出。
    在此处 import（lazy），保证在 _worker_init 调用 django.setup() 之后执行。
    """
    import importlib
    params = dict(params)
    module_path = params.pop('_draw_module', 'spc_chart')
    func_name   = params.pop('_draw_func',   'draw_spc_chart')
    draw_spc_chart = getattr(importlib.import_module(module_path), func_name)

    svg_buf, png_buf, pdf_buf = draw_spc_chart(**params)
    return svg_buf.read(), png_buf.read(), pdf_buf.read()


def render_charts_parallel(
    chart_params: List[Dict[str, Any]],
    max_workers: int = None,
) -> List[Tuple[bytes, bytes, bytes]]:
    """
    并行渲染多张 SPC 图表。

    参数
    ----
    chart_params : list[dict]
        每个 dict 是传给 draw_spc_chart 的关键字参数。
        如果 draw_spc_chart 定义在 Django app 包内，需额外传入：
          _draw_module: 'api.custom.assembly_pcs.service.white_paper_images'
          _draw_func:   'draw_spc_chart'
        这两个键在进入子进程后会被弹出，不会传给实际函数。
    max_workers : int | None
        最大并发进程数。None 表示使用所有 CPU 核心。

    返回
    ----
    list[(svg_buf, png_buf, pdf_buf)]  各元素为 io.BytesIO，顺序与 chart_params 一一对应。
    """
    if not chart_params:
        return []

    # 单张图无需起子进程，直接在当前进程渲染（节省 fork 开销 ~50ms）
    # 注意：_render_one 返回 (bytes, bytes, bytes)，需与多张路径统一包成 BytesIO
    if len(chart_params) == 1:
        svg, png, pdf = _render_one(chart_params[0])
        return [(io.BytesIO(svg), io.BytesIO(png), io.BytesIO(pdf))]

    n_workers = min(
        max_workers or os.cpu_count() or 1,
        len(chart_params),
    )

    # Linux/macOS 用 fork：子进程直接复制父进程内存，无需重新 import，启动快
    # Windows 只支持 spawn（os.fork 不存在），强制使用 spawn
    import sys
    ctx_name = 'fork' if sys.platform != 'win32' else 'spawn'
    ctx = mp.get_context(ctx_name)
    # spawn 子进程需要 initializer 来初始化 Django app registry
    initializer = _worker_init if ctx_name == 'spawn' else None
    with ctx.Pool(processes=n_workers, initializer=initializer) as pool:
        results = pool.map(_render_one, chart_params)

    # 把 bytes 包回 BytesIO，与 draw_spc_chart 原始返回值接口一致
    return [
        (io.BytesIO(svg), io.BytesIO(png), io.BytesIO(pdf))
        for svg, png, pdf in results
    ]


def render_charts_sequential(
    chart_params: List[Dict[str, Any]],
) -> List[Tuple[io.BytesIO, io.BytesIO, io.BytesIO]]:
    """
    串行渲染（用于调试或单核环境），接口与 render_charts_parallel 相同。
    """
    from spc_chart import draw_spc_chart
    return [draw_spc_chart(**p) for p in chart_params]
