"""
Redis Async Utils - Basic Operations
Provides common Redis async basic operations
"""

import json
from typing import Any, Dict, List, Optional, Union

from redis.asyncio import Redis


class AsyncRedisUtils:
    """Redis async utils class, encapsulates common basic operations"""
    
    def __init__(self, redis_client: Redis):
        """
        Initialize async Redis utils
        
        Args:
            redis_client: Initialized Redis async client
        """
        self.client = redis_client
    
    # ==================== Basic Operations ====================
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        return await self.client.exists(key) > 0
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys"""
        return await self.client.delete(*keys)
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set key expiration time in seconds"""
        return await self.client.expire(key, seconds)
    
    async def ttl(self, key: str) -> int:
        """Get key TTL in seconds, -1 means no expiration, -2 means key not exists"""
        return await self.client.ttl(key)
    
    # ==================== String Operations ====================
    
    async def get(self, key: str) -> Optional[str]:
        """Get string value"""
        return await self.client.get(key)
    
    async def set(
        self,
        key: str,
        value: Union[str, int, float],
        ex: Optional[int] = None,
    ) -> bool:
        """
        Set string value
        
        Args:
            key: Key
            value: Value
            ex: Expiration time in seconds
        """
        return await self.client.set(key, value, ex=ex)
    
    async def mget(self, *keys: str) -> List[Optional[str]]:
        """Get multiple keys in batch"""
        return await self.client.mget(*keys)
    
    async def mset(self, mapping: Dict[str, Union[str, int, float]]) -> bool:
        """Set multiple key-value pairs in batch"""
        return await self.client.mset(mapping)
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        return await self.client.incr(key, amount)
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """Decrement counter"""
        return await self.client.decr(key, amount)
    
    # ==================== JSON Operations ====================
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value and deserialize"""
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
        """Serialize and set JSON value"""
        json_str = json.dumps(value, ensure_ascii=False)
        return await self.set(key, json_str, ex=ex)
    
    # ==================== Hash Operations ====================
    
    async def hget(self, key: str, field: str) -> Optional[str]:
        """Get hash field value"""
        return await self.client.hget(key, field)
    
    async def hset(self, key: str, field: str, value: Union[str, int, float]) -> int:
        """Set hash field value"""
        return await self.client.hset(key, field, value)
    
    async def hmget(self, key: str, *fields: str) -> List[Optional[str]]:
        """Get multiple hash fields in batch"""
        return await self.client.hmget(key, *fields)
    
    async def hmset(self, key: str, mapping: Dict[str, Union[str, int, float]]) -> int:
        """Set multiple hash fields in batch"""
        return await self.client.hset(key, mapping=mapping)
    
    async def hgetall(self, key: str) -> Dict[str, str]:
        """Get all hash fields and values"""
        return await self.client.hgetall(key)
    
    async def hdel(self, key: str, *fields: str) -> int:
        """Delete hash fields"""
        return await self.client.hdel(key, *fields)
    
    async def hexists(self, key: str, field: str) -> bool:
        """Check if hash field exists"""
        return await self.client.hexists(key, field)
    
    # ==================== List Operations ====================
    
    async def lpush(self, key: str, *values: Union[str, int, float]) -> int:
        """Push values to list from left"""
        return await self.client.lpush(key, *values)
    
    async def rpush(self, key: str, *values: Union[str, int, float]) -> int:
        """Push values to list from right"""
        return await self.client.rpush(key, *values)
    
    async def lpop(self, key: str) -> Optional[str]:
        """Pop value from list left"""
        return await self.client.lpop(key)
    
    async def rpop(self, key: str) -> Optional[str]:
        """Pop value from list right"""
        return await self.client.rpop(key)
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Get list elements in specified range"""
        return await self.client.lrange(key, start, end)
    
    async def llen(self, key: str) -> int:
        """Get list length"""
        return await self.client.llen(key)
    
    # ==================== Set Operations ====================
    
    async def sadd(self, key: str, *members: Union[str, int, float]) -> int:
        """Add members to set"""
        return await self.client.sadd(key, *members)
    
    async def srem(self, key: str, *members: Union[str, int, float]) -> int:
        """Remove members from set"""
        return await self.client.srem(key, *members)
    
    async def smembers(self, key: str) -> set:
        """Get all set members"""
        return await self.client.smembers(key)
    
    async def sismember(self, key: str, member: Union[str, int, float]) -> bool:
        """Check if member exists in set"""
        return await self.client.sismember(key, member)


# ==================== Usage Example ====================

"""
Usage:

from redis.asyncio import Redis
from redis_async_utils import AsyncRedisUtils

# Assume you already have initialized redis_client (e.g., from FastAPI dependency injection)
redis_client: Redis = ...  # Your existing redis client

# Create utils instance
redis_utils = AsyncRedisUtils(redis_client)

# Usage examples
await redis_utils.set("key", "value", ex=60)
value = await redis_utils.get("key")

# JSON operations
await redis_utils.set_json("user:1", {"name": "John", "age": 25}, ex=3600)
user = await redis_utils.get_json("user:1")

# Hash operations
await redis_utils.hmset("user:2", {"name": "Jane", "email": "jane@example.com"})
user_data = await redis_utils.hgetall("user:2")

# List operations
await redis_utils.rpush("queue", "task1", "task2")
tasks = await redis_utils.lrange("queue", 0, -1)

# Set operations
await redis_utils.sadd("tags", "python", "redis", "async")
tags = await redis_utils.smembers("tags")
"""
