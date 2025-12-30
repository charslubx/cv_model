"""
Redis异步工具类
提供常用的Redis异步操作方法，支持字符串、哈希、列表、集合、有序集合等操作
"""

import json
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool


class AsyncRedisClient:
    """异步Redis客户端封装类"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        decode_responses: bool = True,
        socket_connect_timeout: int = 5,
        socket_timeout: int = 5,
        retry_on_timeout: bool = True,
    ):
        """
        初始化异步Redis客户端
        
        Args:
            host: Redis服务器地址
            port: Redis服务器端口
            db: 数据库编号
            password: Redis密码
            max_connections: 最大连接数
            decode_responses: 是否自动解码响应
            socket_connect_timeout: 连接超时时间（秒）
            socket_timeout: 套接字超时时间（秒）
            retry_on_timeout: 超时是否重试
        """
        self.pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=decode_responses,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
            retry_on_timeout=retry_on_timeout,
        )
        self._client: Optional[Redis] = None
    
    async def connect(self) -> Redis:
        """建立连接"""
        if self._client is None:
            self._client = Redis(connection_pool=self.pool)
        return self._client
    
    async def close(self):
        """关闭连接"""
        if self._client:
            await self._client.aclose()
            await self.pool.aclose()
            self._client = None
    
    @asynccontextmanager
    async def get_client(self):
        """上下文管理器，用于获取客户端"""
        client = await self.connect()
        try:
            yield client
        finally:
            pass  # 不在这里关闭，由close方法统一管理
    
    # ==================== 基础操作 ====================
    
    async def ping(self) -> bool:
        """测试连接"""
        async with self.get_client() as client:
            return await client.ping()
    
    async def exists(self, *keys: str) -> int:
        """检查键是否存在"""
        async with self.get_client() as client:
            return await client.exists(*keys)
    
    async def delete(self, *keys: str) -> int:
        """删除键"""
        async with self.get_client() as client:
            return await client.delete(*keys)
    
    async def expire(self, key: str, seconds: int) -> bool:
        """设置键的过期时间（秒）"""
        async with self.get_client() as client:
            return await client.expire(key, seconds)
    
    async def ttl(self, key: str) -> int:
        """获取键的剩余生存时间（秒）"""
        async with self.get_client() as client:
            return await client.ttl(key)
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的所有键"""
        async with self.get_client() as client:
            return await client.keys(pattern)
    
    # ==================== 字符串操作 ====================
    
    async def get(self, key: str) -> Optional[str]:
        """获取字符串值"""
        async with self.get_client() as client:
            return await client.get(key)
    
    async def set(
        self,
        key: str,
        value: Union[str, int, float],
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """
        设置字符串值
        
        Args:
            key: 键
            value: 值
            ex: 过期时间（秒）
            px: 过期时间（毫秒）
            nx: 只在键不存在时设置
            xx: 只在键存在时设置
        """
        async with self.get_client() as client:
            return await client.set(key, value, ex=ex, px=px, nx=nx, xx=xx)
    
    async def setex(self, key: str, seconds: int, value: Union[str, int, float]) -> bool:
        """设置字符串值并设置过期时间（秒）"""
        async with self.get_client() as client:
            return await client.setex(key, seconds, value)
    
    async def setnx(self, key: str, value: Union[str, int, float]) -> bool:
        """只在键不存在时设置值"""
        async with self.get_client() as client:
            return await client.setnx(key, value)
    
    async def mget(self, *keys: str) -> List[Optional[str]]:
        """批量获取多个键的值"""
        async with self.get_client() as client:
            return await client.mget(*keys)
    
    async def mset(self, mapping: Dict[str, Union[str, int, float]]) -> bool:
        """批量设置多个键值对"""
        async with self.get_client() as client:
            return await client.mset(mapping)
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """递增"""
        async with self.get_client() as client:
            return await client.incr(key, amount)
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """递减"""
        async with self.get_client() as client:
            return await client.decr(key, amount)
    
    # ==================== JSON操作（使用序列化） ====================
    
    async def get_json(self, key: str) -> Optional[Any]:
        """获取JSON值"""
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
        px: Optional[int] = None,
    ) -> bool:
        """设置JSON值"""
        json_str = json.dumps(value, ensure_ascii=False)
        return await self.set(key, json_str, ex=ex, px=px)
    
    # ==================== 哈希操作 ====================
    
    async def hget(self, key: str, field: str) -> Optional[str]:
        """获取哈希字段值"""
        async with self.get_client() as client:
            return await client.hget(key, field)
    
    async def hset(self, key: str, field: str, value: Union[str, int, float]) -> int:
        """设置哈希字段值"""
        async with self.get_client() as client:
            return await client.hset(key, field, value)
    
    async def hmget(self, key: str, *fields: str) -> List[Optional[str]]:
        """批量获取哈希字段值"""
        async with self.get_client() as client:
            return await client.hmget(key, *fields)
    
    async def hmset(self, key: str, mapping: Dict[str, Union[str, int, float]]) -> bool:
        """批量设置哈希字段值"""
        async with self.get_client() as client:
            return await client.hset(key, mapping=mapping)
    
    async def hgetall(self, key: str) -> Dict[str, str]:
        """获取哈希的所有字段和值"""
        async with self.get_client() as client:
            return await client.hgetall(key)
    
    async def hdel(self, key: str, *fields: str) -> int:
        """删除哈希字段"""
        async with self.get_client() as client:
            return await client.hdel(key, *fields)
    
    async def hexists(self, key: str, field: str) -> bool:
        """检查哈希字段是否存在"""
        async with self.get_client() as client:
            return await client.hexists(key, field)
    
    async def hkeys(self, key: str) -> List[str]:
        """获取哈希的所有字段名"""
        async with self.get_client() as client:
            return await client.hkeys(key)
    
    async def hvals(self, key: str) -> List[str]:
        """获取哈希的所有值"""
        async with self.get_client() as client:
            return await client.hvals(key)
    
    async def hlen(self, key: str) -> int:
        """获取哈希字段数量"""
        async with self.get_client() as client:
            return await client.hlen(key)
    
    # ==================== 列表操作 ====================
    
    async def lpush(self, key: str, *values: Union[str, int, float]) -> int:
        """从左侧推入列表"""
        async with self.get_client() as client:
            return await client.lpush(key, *values)
    
    async def rpush(self, key: str, *values: Union[str, int, float]) -> int:
        """从右侧推入列表"""
        async with self.get_client() as client:
            return await client.rpush(key, *values)
    
    async def lpop(self, key: str) -> Optional[str]:
        """从左侧弹出列表元素"""
        async with self.get_client() as client:
            return await client.lpop(key)
    
    async def rpop(self, key: str) -> Optional[str]:
        """从右侧弹出列表元素"""
        async with self.get_client() as client:
            return await client.rpop(key)
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """获取列表指定范围的元素"""
        async with self.get_client() as client:
            return await client.lrange(key, start, end)
    
    async def llen(self, key: str) -> int:
        """获取列表长度"""
        async with self.get_client() as client:
            return await client.llen(key)
    
    async def lindex(self, key: str, index: int) -> Optional[str]:
        """获取列表指定索引的元素"""
        async with self.get_client() as client:
            return await client.lindex(key, index)
    
    async def lset(self, key: str, index: int, value: Union[str, int, float]) -> bool:
        """设置列表指定索引的元素"""
        async with self.get_client() as client:
            return await client.lset(key, index, value)
    
    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """修剪列表"""
        async with self.get_client() as client:
            return await client.ltrim(key, start, end)
    
    # ==================== 集合操作 ====================
    
    async def sadd(self, key: str, *members: Union[str, int, float]) -> int:
        """添加成员到集合"""
        async with self.get_client() as client:
            return await client.sadd(key, *members)
    
    async def srem(self, key: str, *members: Union[str, int, float]) -> int:
        """从集合移除成员"""
        async with self.get_client() as client:
            return await client.srem(key, *members)
    
    async def smembers(self, key: str) -> set:
        """获取集合所有成员"""
        async with self.get_client() as client:
            return await client.smembers(key)
    
    async def sismember(self, key: str, member: Union[str, int, float]) -> bool:
        """检查成员是否在集合中"""
        async with self.get_client() as client:
            return await client.sismember(key, member)
    
    async def scard(self, key: str) -> int:
        """获取集合成员数量"""
        async with self.get_client() as client:
            return await client.scard(key)
    
    async def sunion(self, *keys: str) -> set:
        """获取多个集合的并集"""
        async with self.get_client() as client:
            return await client.sunion(*keys)
    
    async def sinter(self, *keys: str) -> set:
        """获取多个集合的交集"""
        async with self.get_client() as client:
            return await client.sinter(*keys)
    
    async def sdiff(self, *keys: str) -> set:
        """获取多个集合的差集"""
        async with self.get_client() as client:
            return await client.sdiff(*keys)
    
    # ==================== 有序集合操作 ====================
    
    async def zadd(
        self,
        key: str,
        mapping: Dict[Union[str, int, float], float],
        nx: bool = False,
        xx: bool = False,
    ) -> int:
        """
        添加成员到有序集合
        
        Args:
            key: 键
            mapping: 成员和分数的字典 {member: score}
            nx: 只添加新成员
            xx: 只更新已存在的成员
        """
        async with self.get_client() as client:
            return await client.zadd(key, mapping, nx=nx, xx=xx)
    
    async def zrem(self, key: str, *members: Union[str, int, float]) -> int:
        """从有序集合移除成员"""
        async with self.get_client() as client:
            return await client.zrem(key, *members)
    
    async def zscore(self, key: str, member: Union[str, int, float]) -> Optional[float]:
        """获取有序集合成员的分数"""
        async with self.get_client() as client:
            return await client.zscore(key, member)
    
    async def zrange(
        self,
        key: str,
        start: int = 0,
        end: int = -1,
        withscores: bool = False,
    ) -> List:
        """获取有序集合指定范围的成员（按分数升序）"""
        async with self.get_client() as client:
            return await client.zrange(key, start, end, withscores=withscores)
    
    async def zrevrange(
        self,
        key: str,
        start: int = 0,
        end: int = -1,
        withscores: bool = False,
    ) -> List:
        """获取有序集合指定范围的成员（按分数降序）"""
        async with self.get_client() as client:
            return await client.zrevrange(key, start, end, withscores=withscores)
    
    async def zcard(self, key: str) -> int:
        """获取有序集合成员数量"""
        async with self.get_client() as client:
            return await client.zcard(key)
    
    async def zcount(self, key: str, min_score: float, max_score: float) -> int:
        """获取有序集合指定分数范围的成员数量"""
        async with self.get_client() as client:
            return await client.zcount(key, min_score, max_score)
    
    async def zincrby(self, key: str, amount: float, member: Union[str, int, float]) -> float:
        """增加有序集合成员的分数"""
        async with self.get_client() as client:
            return await client.zincrby(key, amount, member)
    
    async def zrank(self, key: str, member: Union[str, int, float]) -> Optional[int]:
        """获取有序集合成员的排名（升序，从0开始）"""
        async with self.get_client() as client:
            return await client.zrank(key, member)
    
    async def zrevrank(self, key: str, member: Union[str, int, float]) -> Optional[int]:
        """获取有序集合成员的排名（降序，从0开始）"""
        async with self.get_client() as client:
            return await client.zrevrank(key, member)
    
    # ==================== 高级操作 ====================
    
    async def pipeline(self):
        """获取管道对象，用于批量操作"""
        async with self.get_client() as client:
            return client.pipeline()
    
    async def scan(
        self,
        cursor: int = 0,
        match: Optional[str] = None,
        count: Optional[int] = None,
    ):
        """扫描键（用于大量键的遍历）"""
        async with self.get_client() as client:
            return await client.scan(cursor=cursor, match=match, count=count)
    
    async def scan_iter(self, match: Optional[str] = None, count: Optional[int] = None):
        """扫描键的迭代器"""
        async with self.get_client() as client:
            async for key in client.scan_iter(match=match, count=count):
                yield key
    
    # ==================== 发布订阅 ====================
    
    async def publish(self, channel: str, message: str) -> int:
        """发布消息到频道"""
        async with self.get_client() as client:
            return await client.publish(channel, message)
    
    async def subscribe(self, *channels: str):
        """订阅频道"""
        async with self.get_client() as client:
            pubsub = client.pubsub()
            await pubsub.subscribe(*channels)
            return pubsub
    
    # ==================== 分布式锁 ====================
    
    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: Optional[int] = None,
    ) -> bool:
        """
        获取分布式锁
        
        Args:
            lock_name: 锁名称
            timeout: 锁超时时间（秒）
            blocking: 是否阻塞等待
            blocking_timeout: 阻塞超时时间（秒）
        """
        async with self.get_client() as client:
            lock = client.lock(lock_name, timeout=timeout)
            return await lock.acquire(blocking=blocking, blocking_timeout=blocking_timeout)
    
    async def release_lock(self, lock_name: str) -> bool:
        """释放分布式锁"""
        async with self.get_client() as client:
            lock = client.lock(lock_name)
            try:
                await lock.release()
                return True
            except Exception:
                return False
    
    @asynccontextmanager
    async def lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: Optional[int] = None,
    ):
        """
        分布式锁上下文管理器
        
        使用示例:
        async with redis_client.lock("my_lock"):
            # 执行需要加锁的操作
            pass
        """
        async with self.get_client() as client:
            lock = client.lock(
                lock_name,
                timeout=timeout,
                blocking=blocking,
                blocking_timeout=blocking_timeout,
            )
            await lock.acquire()
            try:
                yield lock
            finally:
                await lock.release()
    
    # ==================== 缓存装饰器 ====================
    
    def cache(
        self,
        prefix: str = "",
        ex: Optional[int] = None,
        use_json: bool = False,
    ):
        """
        缓存装饰器
        
        Args:
            prefix: 缓存键前缀
            ex: 过期时间（秒）
            use_json: 是否使用JSON序列化
        
        使用示例:
        @redis_client.cache(prefix="user", ex=3600)
        async def get_user(user_id: int):
            return {"id": user_id, "name": "张三"}
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = f"{prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # 尝试从缓存获取
                if use_json:
                    cached_value = await self.get_json(cache_key)
                else:
                    cached_value = await self.get(cache_key)
                
                if cached_value is not None:
                    return cached_value
                
                # 执行函数并缓存结果
                result = await func(*args, **kwargs)
                if use_json:
                    await self.set_json(cache_key, result, ex=ex)
                else:
                    await self.set(cache_key, result, ex=ex)
                
                return result
            
            return wrapper
        return decorator


# ==================== 工厂函数 ====================

def init_async_redis_client(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    max_connections: int = 50,
    **kwargs,
) -> AsyncRedisClient:
    """
    初始化异步Redis客户端工厂函数
    
    Args:
        host: Redis服务器地址
        port: Redis服务器端口
        db: 数据库编号
        password: Redis密码
        max_connections: 最大连接数
        **kwargs: 其他参数
    
    Returns:
        AsyncRedisClient实例
    """
    return AsyncRedisClient(
        host=host,
        port=port,
        db=db,
        password=password,
        max_connections=max_connections,
        **kwargs,
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # 初始化客户端
        redis_client = init_async_redis_client(
            host="localhost",
            port=6379,
            db=0,
            password=None,
        )
        
        try:
            # 测试连接
            print("测试连接:", await redis_client.ping())
            
            # 字符串操作
            await redis_client.set("test_key", "test_value", ex=60)
            value = await redis_client.get("test_key")
            print(f"获取值: {value}")
            
            # JSON操作
            await redis_client.set_json("user:1", {"id": 1, "name": "张三"}, ex=60)
            user = await redis_client.get_json("user:1")
            print(f"获取JSON: {user}")
            
            # 哈希操作
            await redis_client.hmset("user:2", {"name": "李四", "age": "25"})
            user_data = await redis_client.hgetall("user:2")
            print(f"获取哈希: {user_data}")
            
            # 列表操作
            await redis_client.rpush("queue", "task1", "task2", "task3")
            tasks = await redis_client.lrange("queue", 0, -1)
            print(f"获取列表: {tasks}")
            
            # 集合操作
            await redis_client.sadd("tags", "python", "redis", "async")
            tags = await redis_client.smembers("tags")
            print(f"获取集合: {tags}")
            
            # 有序集合操作
            await redis_client.zadd("leaderboard", {"player1": 100, "player2": 200})
            top_players = await redis_client.zrevrange("leaderboard", 0, 1, withscores=True)
            print(f"获取排行榜: {top_players}")
            
            # 分布式锁
            async with redis_client.lock("my_lock", timeout=5):
                print("获取锁成功，执行业务逻辑")
                await asyncio.sleep(1)
            
            print("所有测试完成!")
            
        finally:
            # 关闭连接
            await redis_client.close()
    
    asyncio.run(main())
