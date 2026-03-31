"""
SSE 流式文本 Django View 示例

AI 接口返回完整文本后，逐字推送给前端实现"一个字一个字蹦"的效果。

前端接收到的格式：
  data: {"text": "你"}\n\n
  data: {"text": "好"}\n\n
  ...
  data: [DONE]\n\n
"""
import json
from django.http import StreamingHttpResponse


def _char_stream_generator(full_text: str, interval: float = 0.04):
    """
    将完整文本逐字转成 SSE 事件流。

    参数
    ----
    full_text : str   AI 接口返回的完整回答文本
    interval  : float 每个字之间的间隔秒数，默认 40ms
    """
    import time
    for char in full_text:
        yield f'data: {json.dumps({"text": char}, ensure_ascii=False)}\n\n'
        time.sleep(interval)
    yield 'data: [DONE]\n\n'
