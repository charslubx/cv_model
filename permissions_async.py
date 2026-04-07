import asyncio
import json
import os

import aiohttp
import requests
from asgiref.sync import async_to_sync
from sqlalchemy import and_, select

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


@staticmethod
async def get_user_list(filter_by, params=None):
    complex_conditions = []
    search_key = params.get('search_key') if params else None

    if params:
        for key, value in params.items():
            if key == 'search_key':
                continue
            column = getattr(PermissionUser, key)
            if not isinstance(value, (list, tuple, set)):
                filter_by[key] = value
            elif isinstance(value, (list, tuple, set)):
                complex_conditions.append(column.in_(value))

    async with g.db_async_session() as session:
        # 子查询：找出满足过滤条件的 user_id（逻辑不变）
        subquery = select(PermissionUser.user_id)
        if filter_by:
            subquery = subquery.filter_by(**filter_by)
        if complex_conditions:
            subquery = subquery.filter(and_(*complex_conditions))

        outer_conditions = [UserProfile.idsid.in_(subquery)]
        if search_key:
            outer_conditions.append(
                or_(
                    UserProfile.first_name.ilike(f'%{search_key}%'),
                    UserProfile.last_name.ilike(f'%{search_key}%'),
                )
            )

        user_fields = [c.name for c in UserProfile.__table__.columns if c.name != 'password']
        permission_fields = [c.name for c in PermissionList.__table__.columns]

        # 主查询：LEFT JOIN 拿每个用户的全量权限，不受过滤条件约束
        query = (
            select(
                *[getattr(UserProfile, f).label(f) for f in user_fields],
                *[getattr(PermissionList, f).label(f'perm_{f}') for f in permission_fields],
            )
            .select_from(UserProfile)
            .outerjoin(PermissionUser, UserProfile.idsid == PermissionUser.user_id)
            .outerjoin(PermissionList, PermissionUser.permission_id == PermissionList.permission_id)
            .filter(and_(*outer_conditions))
        )

        result = await session.execute(query)
        rows = result.mappings().fetchall()

        user_map = {}
        for row in rows:
            uid = row['idsid']
            if uid not in user_map:
                user_map[uid] = {
                    **{f: row[f] for f in user_fields},
                    'permissions': [],
                }
            if row.get('perm_permission_id') is not None:
                user_map[uid]['permissions'].append(
                    {f: row[f'perm_{f}'] for f in permission_fields}
                )

        return list(user_map.values())


@staticmethod
async def get_permission_user_list(filter_by=None, params=None):
    search_key = params.get('search_key') if params else None
    filter_by = filter_by or {}

    # 针对 PermissionUser 的等值条件（放入 JOIN ON 子句）
    join_conditions = [PermissionList.permission_id == PermissionUser.permission_id]
    # 针对 PermissionUser 的 IN 条件（同样放入 JOIN ON 子句）
    join_in_conditions = []

    if params:
        for key, value in params.items():
            if key == 'search_key':
                continue
            column = getattr(PermissionUser, key)
            if isinstance(value, (list, tuple, set)):
                join_in_conditions.append(column.in_(value))
            else:
                join_conditions.append(column == value)

    async with g.db_async_session() as session:
        permission_fields = [c.name for c in PermissionList.__table__.columns]
        user_fields = [c.name for c in UserProfile.__table__.columns if c.name != 'password']

        query = (
            select(
                *[getattr(PermissionList, f).label(f) for f in permission_fields],
                *[getattr(UserProfile, f).label(f'user_col_{f}') for f in user_fields],
            )
            .select_from(PermissionList)
            # 把所有 PermissionUser 过滤条件放进 ON，保证 LEFT JOIN 语义正确
            .outerjoin(PermissionUser, and_(*join_conditions, *join_in_conditions))
            .outerjoin(UserProfile, PermissionUser.user_id == UserProfile.idsid)
        )

        # search_key 针对主表，放 WHERE 没问题
        if search_key:
            query = query.filter(PermissionList.permission_name.ilike(f'%{search_key}%'))

        # filter_by 若仍有针对 PermissionList 自身的条件可在此追加
        if filter_by:
            query = query.filter(
                and_(*[getattr(PermissionList, k) == v for k, v in filter_by.items()])
            )

        result = await session.execute(query)
        rows = result.mappings().fetchall()

        permission_map: dict = {}
        for row in rows:
            perm_id = row['permission_id']
            if perm_id not in permission_map:
                permission_map[perm_id] = {
                    **{f: row[f] for f in permission_fields},
                    'users': [],
                }
            # user_id 为 None 说明该权限下无用户，不追加空行
            if row.get('user_col_idsid') is not None:
                permission_map[perm_id]['users'].append(
                    {f: row[f'user_col_{f}'] for f in user_fields}
                )

        return list(permission_map.values())
