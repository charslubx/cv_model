"""
Redis异步工具类 - 基础操作版本
提供常用的Redis异步基础操作方法
"""

import json
from typing import Any, Dict, List, Optional, Union

from redis.asyncio import Redis


class AsyncRedisUtils:
    """Redis异步工具类，封装常用基础操作"""
    
    def __init__(self, redis_client: Redis):
        """
        初始化异步Redis工具类
        
        Args:
            redis_client: 已初始化的Redis异步客户端
        """
        self.client = redis_client
    
    # ==================== 基础操作 ====================
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return await self.client.exists(key) > 0
    
    async def delete(self, *keys: str) -> int:
        """删除一个或多个键"""
        return await self.client.delete(*keys)
    
    async def expire(self, key: str, seconds: int) -> bool:
        """设置键的过期时间（秒）"""
        return await self.client.expire(key, seconds)
    
    async def ttl(self, key: str) -> int:
        """获取键的剩余生存时间（秒），-1表示永不过期，-2表示键不存在"""
        return await self.client.ttl(key)
    
    # ==================== 字符串操作 ====================
    
    async def get(self, key: str) -> Optional[str]:
        """获取字符串值"""
        return await self.client.get(key)
    
    async def set(
        self,
        key: str,
        value: Union[str, int, float],
        ex: Optional[int] = None,
    ) -> bool:
        """
        设置字符串值
        
        Args:
            key: 键
            value: 值
            ex: 过期时间（秒）
        """
        return await self.client.set(key, value, ex=ex)
    
    async def mget(self, *keys: str) -> List[Optional[str]]:
        """批量获取多个键的值"""
        return await self.client.mget(*keys)
    
    async def mset(self, mapping: Dict[str, Union[str, int, float]]) -> bool:
        """批量设置多个键值对"""
        return await self.client.mset(mapping)
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """递增计数器"""
        return await self.client.incr(key, amount)
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """递减计数器"""
        return await self.client.decr(key, amount)
    
    # ==================== JSON操作 ====================
    
    async def get_json(self, key: str) -> Optional[Any]:
        """获取JSON值并反序列化"""
        value = await self.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    
    async def set_json(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
    ) -> bool:
        """序列化并设置JSON值"""
        json_str = json.dumps(value, ensure_ascii=False)
        return await self.set(key, json_str, ex=ex)
    
    # ==================== 哈希操作 ====================
    
    async def hget(self, key: str, field: str) -> Optional[str]:
        """获取哈希字段值"""
        return await self.client.hget(key, field)
    
    async def hset(self, key: str, field: str, value: Union[str, int, float]) -> int:
        """设置哈希字段值"""
        return await self.client.hset(key, field, value)
    
    async def hmget(self, key: str, *fields: str) -> List[Optional[str]]:
        """批量获取哈希字段值"""
        return await self.client.hmget(key, *fields)
    
    async def hmset(self, key: str, mapping: Dict[str, Union[str, int, float]]) -> int:
        """批量设置哈希字段值"""
        return await self.client.hset(key, mapping=mapping)
    
    async def hgetall(self, key: str) -> Dict[str, str]:
        """获取哈希的所有字段和值"""
        return await self.client.hgetall(key)
    
    async def hdel(self, key: str, *fields: str) -> int:
        """删除哈希字段"""
        return await self.client.hdel(key, *fields)
    
    async def hexists(self, key: str, field: str) -> bool:
        """检查哈希字段是否存在"""
        return await self.client.hexists(key, field)
    
    # ==================== 列表操作 ====================
    
    async def lpush(self, key: str, *values: Union[str, int, float]) -> int:
        """从左侧推入列表"""
        return await self.client.lpush(key, *values)
    
    async def rpush(self, key: str, *values: Union[str, int, float]) -> int:
        """从右侧推入列表"""
        return await self.client.rpush(key, *values)
    
    async def lpop(self, key: str) -> Optional[str]:
        """从左侧弹出列表元素"""
        return await self.client.lpop(key)
    
    async def rpop(self, key: str) -> Optional[str]:
        """从右侧弹出列表元素"""
        return await self.client.rpop(key)
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """获取列表指定范围的元素"""
        return await self.client.lrange(key, start, end)
    
    async def llen(self, key: str) -> int:
        """获取列表长度"""
        return await self.client.llen(key)
    
    # ==================== 集合操作 ====================
    
    async def sadd(self, key: str, *members: Union[str, int, float]) -> int:
        """添加成员到集合"""
        return await self.client.sadd(key, *members)
    
    async def srem(self, key: str, *members: Union[str, int, float]) -> int:
        """从集合移除成员"""
        return await self.client.srem(key, *members)
    
    async def smembers(self, key: str) -> set:
        """获取集合所有成员"""
        return await self.client.smembers(key)
    
    async def sismember(self, key: str, member: Union[str, int, float]) -> bool:
        """检查成员是否在集合中"""
        return await self.client.sismember(key, member)


# ==================== 使用示例 ====================

"""
使用方式：

from redis.asyncio import Redis
from redis_async_utils import AsyncRedisUtils

# 假设你已经有了初始化的redis_client（如FastAPI中的依赖注入）
redis_client: Redis = ...  # 你已有的redis客户端

# 创建工具实例
redis_utils = AsyncRedisUtils(redis_client)

# 使用示例
await redis_utils.set("key", "value", ex=60)
value = await redis_utils.get("key")

# JSON操作
await redis_utils.set_json("user:1", {"name": "张三", "age": 25}, ex=3600)
user = await redis_utils.get_json("user:1")

# 哈希操作
await redis_utils.hmset("user:2", {"name": "李四", "email": "lisi@example.com"})
user_data = await redis_utils.hgetall("user:2")

# 列表操作
await redis_utils.rpush("queue", "task1", "task2")
tasks = await redis_utils.lrange("queue", 0, -1)

# 集合操作
await redis_utils.sadd("tags", "python", "redis", "async")
tags = await redis_utils.smembers("tags")
"""
