import asyncio
import json
import os

import aiohttp
import requests
from asgiref.sync import async_to_sync

BATCH_SIZE = 100


async def _fetch_permissions_batch(session, url, batch_ids):
    async with session.post(url, json={"permission_id": batch_ids, "or_tag": "collateralx"}, ssl=False) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data.get("data") or []


async def _fetch_all_permissions(permission_ids):
    url = AUTH_AUTHENTICATE_URL + '/api/v1/permissions/filter'
    batches = [permission_ids[i:i + BATCH_SIZE] for i in range(0, len(permission_ids), BATCH_SIZE)]
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[
            _fetch_permissions_batch(session, url, batch) for batch in batches
        ])
    return [item for batch in results for item in batch]


def get_x_permissions():
    try:
        cache_data = cache.get(f"{os.environ.get('O_ENV_MODE', 'PROD')}_x_permission_cache")
        if cache_data:
            permission_cache = json.loads(cache_data)

        if not cache_data or not permission_cache["permissions"]:
            url = AUTH_AUTHENTICATE_URL + '/api/v1/resource/filter'
            response = requests.post(url, json={"resource_type": "collateralx"}, verify=False)
            response.raise_for_status()
            permission_ids = list({_["permission_id"] for _ in response.json()["data"]})

            all_data = async_to_sync(_fetch_all_permissions)(permission_ids)

            permission_cache = {"permissions": all_data, "permission_ids": permission_ids}
            permissions = [PermissionList(**_) for _ in all_data]
            cache.set(f"{os.environ.get('O_ENV_MODE', 'PROD')}_x_permission_cache", json.dumps(permission_cache))
            return permissions, permission_ids

        return permission_cache["permissions"], permission_cache["permission_ids"]
    except Exception as e:
        msg = f'Error happened while get x permissions. {e}'
        logger.error(msg)
        raise e
