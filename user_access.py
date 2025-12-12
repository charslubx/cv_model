"""
FastAPI 版本 - 用户权限访问接口
将 Django 风格的权限查询函数转换为 FastAPI 风格
"""

import json
import logging
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from redis import Redis

from models import PermissionUser

# 配置日志
log = logging.getLogger(__name__)

# 创建路由
router = APIRouter(prefix="/api/users", tags=["用户权限"])


# ============================================================
# 依赖项
# ============================================================

def get_db():
    """获取数据库会话"""
    from database import SessionLocal  # 假设你有这个配置
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis():
    """获取 Redis 连接"""
    from config import REDIS_HOST, REDIS_PORT, REDIS_DB  # 假设你有这个配置
    redis_client = Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )
    try:
        yield redis_client
    finally:
        redis_client.close()


# ============================================================
# Redis 缓存 Key 生成函数
# ============================================================

def get_redis_user_access_key(user_id: str) -> str:
    """生成用户权限 ID 列表的 Redis Key"""
    return f"user:access:{user_id}"


def get_redis_user_access_info_key(user_id: str) -> str:
    """生成用户权限详细信息的 Redis Key"""
    return f"user:access:info:{user_id}"


# ============================================================
# 辅助函数
# ============================================================

def get_redis_cache(redis: Redis, key: str) -> Optional[str]:
    """从 Redis 获取缓存"""
    try:
        return redis.get(key)
    except Exception as e:
        log.error(f"Redis 获取缓存失败: {key}, 错误: {e}")
        return None


def set_redis_cache(redis: Redis, key: str, value: str, expire: int = 3600) -> bool:
    """设置 Redis 缓存"""
    try:
        redis.setex(key, expire, value)
        return True
    except Exception as e:
        log.error(f"Redis 设置缓存失败: {key}, 错误: {e}")
        return False


def pull_user_group(db: Session, user_id: str) -> List[str]:
    """
    获取用户所属的所有组 ID
    需要根据实际的 UserGroup 模型实现
    """
    # 假设你有 UserGroup 模型
    # from models import UserGroup
    # groups = db.query(UserGroup.group_id).filter(
    #     UserGroup.user_id == user_id
    # ).all()
    # return [str(group.group_id) for group in groups]
    
    # 示例实现
    return []  # 返回组 ID 列表


def update_and_get_user_all_access(
    db: Session, 
    redis: Redis, 
    user_id: str, 
    sync_PDL: bool = True
) -> List[str]:
    """
    更新并获取用户的所有权限 ID
    """
    # 从数据库查询用户的直接权限
    permissions = db.query(PermissionUser.permission_id).filter(
        PermissionUser.user_id == user_id
    ).all()
    
    permission_id_list = [perm.permission_id for perm in permissions]
    
    # 缓存到 Redis
    set_redis_cache(
        redis, 
        get_redis_user_access_key(user_id), 
        json.dumps(permission_id_list)
    )
    
    return permission_id_list


def pull_user_permission_info(db: Session, user_ids: List[str]) -> List[Dict[str, Any]]:
    """
    根据用户/组 ID 列表获取权限详细信息
    """
    # 查询权限信息
    permissions = db.query(PermissionUser).filter(
        PermissionUser.user_id.in_(user_ids)
    ).all()
    
    # 转换为字典列表
    permission_infos = []
    for perm in permissions:
        permission_infos.append({
            'permission_id': perm.permission_id,
            'user_id': perm.user_id,
            'create_time': perm.create_time,
            'update_time': perm.update_time
        })
    
    return permission_infos


def get_multi_db_data(
    db: Session,
    model: Any,
    fields: List[str],
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    通用的多条件数据库查询
    """
    query = db.query(model)
    
    # 应用过滤条件
    for key, value in filters.items():
        if isinstance(value, list):
            # 如果是列表，使用 IN 查询
            query = query.filter(getattr(model, key).in_(value))
        else:
            query = query.filter(getattr(model, key) == value)
    
    results = query.all()
    
    # 转换为字典列表
    result_list = []
    for row in results:
        row_dict = {}
        for field in fields:
            row_dict[field] = getattr(row, field)
        result_list.append(row_dict)
    
    return result_list


# ============================================================
# 主要业务逻辑
# ============================================================

def pull_user_access_logic(
    user_id: str,
    res_type: str,
    db: Session,
    redis: Redis
) -> List[Any]:
    """
    获取用户权限访问
    
    Args:
        user_id: 用户 ID
        res_type: 返回类型 ('info' 返回详细信息, 其他返回 ID 列表)
        db: 数据库会话
        redis: Redis 连接
        
    Returns:
        权限列表（详细信息或 ID 列表）
    """
    log.info(f'start pull_user_access, user: {user_id}, type: {res_type}')
    
    # 获取用户组
    group_ids = pull_user_group(db, user_id)
    
    if res_type == 'info':
        # 返回权限详细信息
        permission_infos = get_redis_cache(redis, get_redis_user_access_info_key(user_id))
        
        if permission_infos:
            permission_infos = json.loads(permission_infos)
            log.info(f'pull_user_access done (cached), user: {user_id}, res_len: {len(permission_infos)}')
            return permission_infos
        
        # 构建查询的用户/组 ID 列表
        new_group_ids = [user_id]
        for group_id in group_ids:
            new_group_ids.append(str(group_id))
        
        group_ids.append(user_id)
        
        # 更新用户权限
        update_and_get_user_all_access(db, redis, user_id, sync_PDL=False)
        
        # 获取权限详细信息
        permission_infos = pull_user_permission_info(db, new_group_ids)
        
        # 转换时间字段为字符串
        for permission_info in permission_infos:
            if permission_info.get('create_time'):
                permission_info['create_time'] = str(permission_info['create_time'])
            if permission_info.get('update_time'):
                permission_info['update_time'] = str(permission_info['update_time'])
        
        # 缓存结果
        set_redis_cache(
            redis,
            get_redis_user_access_info_key(user_id),
            json.dumps(permission_infos, ensure_ascii=False)
        )
        
        log.info(f'pull_user_access done, user: {user_id}, res_len: {len(permission_infos)}')
        return permission_infos
    
    else:
        # 返回权限 ID 列表
        permission_id_list = get_redis_cache(redis, get_redis_user_access_key(user_id))
        
        if permission_id_list:
            permission_id_list = json.loads(permission_id_list)
            log.info(f'pull_user_access done (cached), user: {user_id}, res_len: {len(permission_id_list)}')
            return permission_id_list
        else:
            permission_id_list = update_and_get_user_all_access(db, redis, user_id)
        
        # 查询组权限
        res = get_multi_db_data(
            db,
            PermissionUser,
            ["permission_id"],
            filters={'user_id': group_ids}
        )
        
        for row in res:
            if row['permission_id'] not in permission_id_list:
                permission_id_list.append(row['permission_id'])
        
        # 缓存结果
        set_redis_cache(
            redis,
            get_redis_user_access_key(user_id),
            json.dumps(permission_id_list)
        )
        
        log.info(f'pull_user_access done, user: {user_id}, res_len: {len(permission_id_list)}')
        return permission_id_list


# ============================================================
# FastAPI 路由
# ============================================================

@router.get("/{user_id}/access")
async def get_user_access(
    user_id: str,
    res_type: Literal['info', 'ids'] = Query(
        default='info',
        description="返回类型: 'info' 返回详细信息, 'ids' 返回 ID 列表"
    ),
    db: Session = Depends(get_db),
    redis: Redis = Depends(get_redis)
):
    """
    获取用户权限访问
    
    - **user_id**: 用户 ID
    - **res_type**: 返回类型
        - `info`: 返回权限详细信息（包含 create_time, update_time 等）
        - `ids`: 仅返回权限 ID 列表
        
    返回示例 (res_type='info'):
    ```json
    [
        {
            "permission_id": "PERM001",
            "user_id": "USER123",
            "create_time": "2025-12-12 10:30:00",
            "update_time": "2025-12-12 10:30:00"
        }
    ]
    ```
    
    返回示例 (res_type='ids'):
    ```json
    ["PERM001", "PERM002", "PERM003"]
    ```
    """
    try:
        result = pull_user_access_logic(
            user_id=user_id,
            res_type=res_type,
            db=db,
            redis=redis
        )
        
        return {
            "code": 0,
            "message": "success",
            "data": result
        }
        
    except Exception as e:
        log.error(f"获取用户权限失败: user_id={user_id}, error={e}", exc_info=True)
        return {
            "code": -1,
            "message": f"获取用户权限失败: {str(e)}",
            "data": []
        }


@router.delete("/{user_id}/access/cache")
async def clear_user_access_cache(
    user_id: str,
    redis: Redis = Depends(get_redis)
):
    """
    清除用户权限缓存
    
    - **user_id**: 用户 ID
    """
    try:
        # 删除两种类型的缓存
        redis.delete(get_redis_user_access_key(user_id))
        redis.delete(get_redis_user_access_info_key(user_id))
        
        log.info(f"已清除用户权限缓存: user_id={user_id}")
        
        return {
            "code": 0,
            "message": "缓存清除成功",
            "data": {"user_id": user_id}
        }
        
    except Exception as e:
        log.error(f"清除缓存失败: user_id={user_id}, error={e}")
        return {
            "code": -1,
            "message": f"清除缓存失败: {str(e)}",
            "data": None
        }


# ============================================================
# 使用示例
# ============================================================

"""
# main.py
from fastapi import FastAPI
from user_access import router as user_access_router

app = FastAPI(title="用户权限管理系统")

# 注册路由
app.include_router(user_access_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# 使用示例：

# 1. 获取用户权限详细信息
# GET http://localhost:8000/api/users/USER123/access?res_type=info

# 2. 获取用户权限 ID 列表
# GET http://localhost:8000/api/users/USER123/access?res_type=ids

# 3. 清除用户权限缓存
# DELETE http://localhost:8000/api/users/USER123/access/cache
"""
