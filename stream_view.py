"""
SSE 流式文本 Django View 示例

前端接收到的格式：
  data: 你\n\n
  data: 好\n\n
  data: 世\n\n
  ...
  data: [DONE]\n\n
"""
import json
from django.http import StreamingHttpResponse


def _ai_stream_generator(prompt: str):
    """
    调用 AI 接口，逐字 yield SSE 格式数据。
    把这里替换成你实际使用的 AI SDK（OpenAI / 自研接口等）。
    """
    # ── 示例：OpenAI ChatCompletion stream ──────────────────────────────
    # import openai
    # response = openai.ChatCompletion.create(
    #     model='gpt-4',
    #     messages=[{'role': 'user', 'content': prompt}],
    #     stream=True,
    # )
    # for chunk in response:
    #     delta = chunk['choices'][0]['delta']
    #     text = delta.get('content', '')
    #     if text:
    #         yield f'data: {json.dumps({"text": text}, ensure_ascii=False)}\n\n'
    # yield 'data: [DONE]\n\n'

    # ── 示例：requests 调用自研流式接口 ────────────────────────────────
    # import requests
    # with requests.post(AI_URL, json={'prompt': prompt}, stream=True) as r:
    #     for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
    #         if chunk:
    #             yield f'data: {json.dumps({"text": chunk}, ensure_ascii=False)}\n\n'
    # yield 'data: [DONE]\n\n'

    # ── 本地模拟：逐字发送（测试用）────────────────────────────────────
    import time
    mock_text = f'你发送的是：{prompt}。这是一段模拟 AI 返回的流式文本，每个字逐个蹦出来。'
    for char in mock_text:
        yield f'data: {json.dumps({"text": char}, ensure_ascii=False)}\n\n'
        time.sleep(0.05)
    yield 'data: [DONE]\n\n'


def ai_stream_view(request):
    """
    GET /api/ai/stream/?prompt=你好

    返回 SSE（text/event-stream）流，前端用 EventSource 或 fetch ReadableStream 接收。
    """
    prompt = request.GET.get('prompt', '')

    response = StreamingHttpResponse(
        _ai_stream_generator(prompt),
        content_type='text/event-stream; charset=utf-8',
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'   # 关闭 Nginx 缓冲，保证实时推送
    return response
