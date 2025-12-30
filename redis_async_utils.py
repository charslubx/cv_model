"""
Redis Utils - Basic Operations (Sync Version)
Provides common Redis basic operations
"""

import json
from typing import Any, Dict, List, Optional, Union


class RedisUtils:
    """Redis utils class, encapsulates common basic operations"""
    
    def __init__(self, redis_client):
        """
        Initialize Redis utils
        
        Args:
            redis_client: Initialized Redis client (RedisClient instance)
        """
        self.client = redis_client
    
    # ==================== Basic Operations ====================
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return self.client.exists(key) > 0
    
    def delete(self, *keys: str) -> int:
        """Delete one or more keys"""
        return self.client.delete(*keys)
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set key expiration time in seconds"""
        return self.client.expire(key, seconds)
    
    def ttl(self, key: str) -> int:
        """Get key TTL in seconds, -1 means no expiration, -2 means key not exists"""
        return self.client.ttl(key)
    
    # ==================== String Operations ====================
    
    def get(self, key: str) -> Optional[str]:
        """Get string value"""
        return self.client.get(key)
    
    def set(
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
        return self.client.set(key, value, ex=ex)
    
    def mget(self, *keys: str) -> List[Optional[str]]:
        """Get multiple keys in batch"""
        return self.client.mget(*keys)
    
    def mset(self, mapping: Dict[str, Union[str, int, float]]) -> bool:
        """Set multiple key-value pairs in batch"""
        return self.client.mset(mapping)
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        return self.client.incr(key, amount)
    
    def decr(self, key: str, amount: int = 1) -> int:
        """Decrement counter"""
        return self.client.decr(key, amount)
    
    # ==================== JSON Operations ====================
    
    def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value and deserialize"""
        value = self.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    
    def set_json(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
    ) -> bool:
        """Serialize and set JSON value"""
        json_str = json.dumps(value, ensure_ascii=False)
        return self.set(key, json_str, ex=ex)
    
    # ==================== Hash Operations ====================
    
    def hget(self, key: str, field: str) -> Optional[str]:
        """Get hash field value"""
        return self.client.hget(key, field)
    
    def hset(self, key: str, field: str, value: Union[str, int, float]) -> int:
        """Set hash field value"""
        return self.client.hset(key, field, value)
    
    def hmget(self, key: str, *fields: str) -> List[Optional[str]]:
        """Get multiple hash fields in batch"""
        return self.client.hmget(key, *fields)
    
    def hmset(self, key: str, mapping: Dict[str, Union[str, int, float]]) -> int:
        """Set multiple hash fields in batch"""
        return self.client.hset(key, mapping=mapping)
    
    def hgetall(self, key: str) -> Dict[str, str]:
        """Get all hash fields and values"""
        return self.client.hgetall(key)
    
    def hdel(self, key: str, *fields: str) -> int:
        """Delete hash fields"""
        return self.client.hdel(key, *fields)
    
    def hexists(self, key: str, field: str) -> bool:
        """Check if hash field exists"""
        return self.client.hexists(key, field)
    
    # ==================== List Operations ====================
    
    def lpush(self, key: str, *values: Union[str, int, float]) -> int:
        """Push values to list from left"""
        return self.client.lpush(key, *values)
    
    def rpush(self, key: str, *values: Union[str, int, float]) -> int:
        """Push values to list from right"""
        return self.client.rpush(key, *values)
    
    def lpop(self, key: str) -> Optional[str]:
        """Pop value from list left"""
        return self.client.lpop(key)
    
    def rpop(self, key: str) -> Optional[str]:
        """Pop value from list right"""
        return self.client.rpop(key)
    
    def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Get list elements in specified range"""
        return self.client.lrange(key, start, end)
    
    def llen(self, key: str) -> int:
        """Get list length"""
        return self.client.llen(key)
    
    # ==================== Set Operations ====================
    
    def sadd(self, key: str, *members: Union[str, int, float]) -> int:
        """Add members to set"""
        return self.client.sadd(key, *members)
    
    def srem(self, key: str, *members: Union[str, int, float]) -> int:
        """Remove members from set"""
        return self.client.srem(key, *members)
    
    def smembers(self, key: str) -> set:
        """Get all set members"""
        return self.client.smembers(key)
    
    def sismember(self, key: str, member: Union[str, int, float]) -> bool:
        """Check if member exists in set"""
        return self.client.sismember(key, member)


# ==================== Usage Example ====================

"""
Usage:

from redis_async_utils import RedisUtils

# Assume you already have initialized redis_client (e.g., from Flask g object)
redis_client = g.redis_client  # Your existing redis client

# Create utils instance
redis_utils = RedisUtils(redis_client)

# Usage examples
redis_utils.set("key", "value", ex=60)  # ex=60 means 60 seconds (1 minute)
value = redis_utils.get("key")

# JSON operations
redis_utils.set_json("user:1", {"name": "John", "age": 25}, ex=3600)  # 3600s = 1 hour
user = redis_utils.get_json("user:1")

# Hash operations
redis_utils.hmset("user:2", {"name": "Jane", "email": "jane@example.com"})
user_data = redis_utils.hgetall("user:2")

# List operations
redis_utils.rpush("queue", "task1", "task2")
tasks = redis_utils.lrange("queue", 0, -1)

# Set operations
redis_utils.sadd("tags", "python", "redis", "sync")
tags = redis_utils.smembers("tags")
"""
