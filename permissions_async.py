"""
协程优化版本：将 get_x_permissions 中对 /api/v1/permissions/filter 的多次请求改为并发执行。

主要改动：
- 使用 aiohttp.ClientSession + asyncio.gather 并发请求多个分批次的 permissions/filter
- 对 permission_ids 按 BATCH_SIZE 分批，所有批次同时发出，合并结果
- 对外提供同步包装函数 get_x_permissions，内部调用 asyncio.run / loop.run_until_complete
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import aiohttp
import requests

logger = logging.getLogger(__name__)

# 每批请求携带的 permission_id 数量，根据服务端限制调整
BATCH_SIZE = 100


# ---------- 数据模型（示意，与原项目保持一致）----------

@dataclass
class PermissionList:
    """与原代码中 PermissionList 保持一致，按实际字段调整。"""
    permission_id: str
    name: str = ""
    or_tag: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "PermissionList":
        return cls(
            permission_id=d.get("permission_id", ""),
            name=d.get("name", ""),
            or_tag=d.get("or_tag", ""),
        )


# ---------- 辅助函数 ----------

def _chunk(lst: list, size: int):
    """将列表切分成若干大小为 size 的子列表。"""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------- 核心异步实现 ----------

async def _fetch_permissions_batch(
    session: aiohttp.ClientSession,
    url: str,
    batch_ids: list[str],
    or_tag: str,
) -> list[dict]:
    """向 /api/v1/permissions/filter 发出单批次请求，返回原始 data 列表。"""
    payload = {"permission_id": batch_ids, "or_tag": or_tag}
    async with session.post(url, json=payload, ssl=False) as resp:
        resp.raise_for_status()
        body = await resp.json()
        return body.get("data") or []


async def _get_x_permissions_async(
    auth_url: str,
    cache: Any,
    cache_key: str,
) -> tuple[list[PermissionList], list[str]]:
    """
    异步核心逻辑：
    1. 检查 Redis 缓存
    2. 同步请求 /api/v1/resource/filter（一次性，串行即可）
    3. 对所有 permission_ids 按 BATCH_SIZE 分批，asyncio.gather 并发请求
    4. 合并结果写入缓存并返回
    """
    # ---- 1. 读取缓存 ----
    cache_data = cache.get(cache_key)
    if cache_data:
        permission_cache = json.loads(cache_data)
        if permission_cache.get("permissions"):
            return permission_cache["permissions"], permission_cache["permission_ids"]

    # ---- 2. 获取 resource → permission_ids（单次，用同步 requests 或 aiohttp 均可）----
    resource_url = auth_url + "/api/v1/resource/filter"
    resource_resp = requests.post(
        resource_url,
        json={"resource_type": "collateralx"},
        verify=False,
    )
    resource_resp.raise_for_status()
    permission_ids: list[str] = list(
        {item["permission_id"] for item in resource_resp.json()["data"]}
    )

    # ---- 3. 分批并发请求 /api/v1/permissions/filter ----
    permissions_url = auth_url + "/api/v1/permissions/filter"
    batches = list(_chunk(permission_ids, BATCH_SIZE))

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _fetch_permissions_batch(session, permissions_url, batch, "collateralx")
            for batch in batches
        ]
        # 所有批次同时发出，任一批次抛异常则整体失败
        results: list[list[dict]] = await asyncio.gather(*tasks)

    # ---- 4. 合并、去重、缓存 ----
    all_data: list[dict] = [item for batch_result in results for item in batch_result]
    # 按 permission_id 去重（多批次可能返回重复项）
    seen: set[str] = set()
    unique_data: list[dict] = []
    for item in all_data:
        pid = item.get("permission_id")
        if pid and pid not in seen:
            seen.add(pid)
            unique_data.append(item)

    permission_cache = {"permissions": unique_data, "permission_ids": permission_ids}
    cache.set(cache_key, json.dumps(permission_cache))

    permissions = [PermissionList.from_dict(d) for d in unique_data]
    return permissions, permission_ids


# ---------- 对外同步接口（与原函数签名兼容）----------

# 这两个变量与原代码保持一致，由调用方通过模块级变量或依赖注入提供
AUTH_AUTHENTICATE_URL: str = os.environ.get("AUTH_AUTHENTICATE_URL", "")


def get_x_permissions(cache: Any) -> tuple[list[PermissionList], list[str]]:
    """
    兼容原同步调用方式，内部使用协程并发请求。

    参数
    ----
    cache : 支持 .get(key) / .set(key, value) 的缓存对象（如 Redis 客户端）

    返回
    ----
    (permissions, permission_ids)
    """
    cache_key = f"{os.environ.get('O_ENV_MODE', 'PROD')}_x_permission_cache"
    try:
        return asyncio.run(
            _get_x_permissions_async(AUTH_AUTHENTICATE_URL, cache, cache_key)
        )
    except Exception as e:
        msg = f"Error happened while get x permissions. {e}"
        logger.error(msg)
        raise


# ---------- 若调用方本身已在异步上下文中，直接 await 此函数 ----------

async def get_x_permissions_async(cache: Any) -> tuple[list[PermissionList], list[str]]:
    """
    在已有事件循环中直接 await 使用的版本。

    示例：
        permissions, ids = await get_x_permissions_async(cache)
    """
    cache_key = f"{os.environ.get('O_ENV_MODE', 'PROD')}_x_permission_cache"
    try:
        return await _get_x_permissions_async(AUTH_AUTHENTICATE_URL, cache, cache_key)
    except Exception as e:
        msg = f"Error happened while get x permissions. {e}"
        logger.error(msg)
        raise
