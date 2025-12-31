"""
数据库通用工具函数 - 支持 SQLAlchemy 2.0 异步

包含单条创建、批量创建等通用函数
"""

from typing import List, Dict, Any, Optional, Union
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession


# ============================================================================
# 辅助函数：查询单条记录
# ============================================================================

async def fetch_one(
    session: AsyncSession,
    model,
    filter_by: dict
):
    """
    查询单条记录
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        filter_by: 过滤条件字典，例如 {"name": "产品1"}
        
    返回：
        查询到的记录对象，未找到返回 None
    """
    stmt = select(model)
    for key, value in filter_by.items():
        stmt = stmt.where(getattr(model, key) == value)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


# ============================================================================
# 原始单条创建函数
# ============================================================================

async def create(
    session: AsyncSession,
    model,
    data: dict,
    filter_by: dict = None,
) -> int:
    """
    创建单条记录
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        data: 要插入的数据字典
        filter_by: 可选，去重检查条件，例如 {"name": "产品1"}
        
    返回：
        插入记录的 ID，如果记录已存在返回 0
        
    示例：
        product_id = await create(
            session, 
            Product, 
            {"name": "iPhone", "price": 999.99},
            filter_by={"name": "iPhone"}
        )
    """
    try:
        if filter_by:
            result = await fetch_one(session, model, filter_by=filter_by)
            if result:
                return 0
        stmt = model(**data)
        session.add(stmt)
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    return stmt.id


# ============================================================================
# 批量创建函数 - 简化版（推荐）⭐
# ============================================================================

async def bulk_create(
    session: AsyncSession,
    model,
    data_list: List[dict],
    filter_by: Optional[str] = None,
) -> int:
    """
    批量创建记录（推荐使用）
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        data_list: 数据字典列表
        filter_by: 可选，去重字段名，例如 "name"
        
    返回：
        创建的记录数量
        
    示例：
        # 简单批量创建
        count = await bulk_create(session, Product, [
            {"name": "产品1", "price": 99.99},
            {"name": "产品2", "price": 199.99}
        ])
        
        # 带去重的批量创建
        count = await bulk_create(
            session, Product, data_list, filter_by="name"
        )
    """
    if not data_list:
        return 0
    
    try:
        # 如果需要去重
        if filter_by:
            # 获取需要检查的值列表
            check_values = [d.get(filter_by) for d in data_list if d.get(filter_by)]
            
            if check_values:
                # 查询已存在的记录
                stmt = select(getattr(model, filter_by)).where(
                    getattr(model, filter_by).in_(check_values)
                )
                result = await session.execute(stmt)
                existing_values = set(result.scalars().all())
                
                # 过滤掉已存在的数据
                data_list = [
                    d for d in data_list
                    if d.get(filter_by) not in existing_values
                ]
                
                if not data_list:
                    return 0
        
        # 批量插入
        stmt = insert(model).values(data_list)
        result = await session.execute(stmt)
        await session.commit()
        
        return result.rowcount
    
    except Exception:
        await session.rollback()
        raise


# ============================================================================
# 批量创建函数 - 返回 ID 列表
# ============================================================================

async def bulk_create_return_ids(
    session: AsyncSession,
    model,
    data_list: List[dict],
    filter_by: Optional[str] = None,
) -> List[int]:
    """
    批量创建记录并返回 ID 列表
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        data_list: 数据字典列表
        filter_by: 可选，去重字段名
        
    返回：
        创建的记录 ID 列表
        
    示例：
        ids = await bulk_create_return_ids(session, Product, [
            {"name": "产品1", "price": 99.99},
            {"name": "产品2", "price": 199.99}
        ])
        # 返回: [1, 2]
    """
    if not data_list:
        return []
    
    try:
        # 如果需要去重
        if filter_by:
            check_values = [d.get(filter_by) for d in data_list if d.get(filter_by)]
            
            if check_values:
                stmt = select(getattr(model, filter_by)).where(
                    getattr(model, filter_by).in_(check_values)
                )
                result = await session.execute(stmt)
                existing_values = set(result.scalars().all())
                
                data_list = [
                    d for d in data_list
                    if d.get(filter_by) not in existing_values
                ]
                
                if not data_list:
                    return []
        
        # 创建对象列表
        instances = [model(**data) for data in data_list]
        
        # 批量添加
        session.add_all(instances)
        await session.commit()
        
        # 返回 ID 列表
        return [inst.id for inst in instances]
    
    except Exception:
        await session.rollback()
        raise


# ============================================================================
# 批量创建函数 - 详细结果
# ============================================================================

async def bulk_create_detailed(
    session: AsyncSession,
    model,
    data_list: List[dict],
    filter_by: Optional[str] = None,
) -> dict:
    """
    批量创建记录并返回详细结果
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        data_list: 数据字典列表
        filter_by: 可选，去重字段名
        
    返回：
        {
            "created": 创建数量,
            "skipped": 跳过数量,
            "total": 总数量,
            "ids": ID列表
        }
        
    示例：
        result = await bulk_create_detailed(
            session, Product, data_list, filter_by="name"
        )
        print(f"创建: {result['created']}, 跳过: {result['skipped']}")
    """
    if not data_list:
        return {"created": 0, "skipped": 0, "total": 0, "ids": []}
    
    total = len(data_list)
    skipped = 0
    
    try:
        # 如果需要去重
        if filter_by:
            check_values = [d.get(filter_by) for d in data_list if d.get(filter_by)]
            
            if check_values:
                stmt = select(getattr(model, filter_by)).where(
                    getattr(model, filter_by).in_(check_values)
                )
                result = await session.execute(stmt)
                existing_values = set(result.scalars().all())
                
                # 统计跳过数量
                skipped = sum(1 for d in data_list if d.get(filter_by) in existing_values)
                
                # 过滤数据
                data_list = [
                    d for d in data_list
                    if d.get(filter_by) not in existing_values
                ]
                
                if not data_list:
                    return {"created": 0, "skipped": skipped, "total": total, "ids": []}
        
        # 创建对象列表
        instances = [model(**data) for data in data_list]
        
        # 批量添加
        session.add_all(instances)
        await session.commit()
        
        # 获取 ID 列表
        ids = [inst.id for inst in instances]
        
        return {
            "created": len(ids),
            "skipped": skipped,
            "total": total,
            "ids": ids
        }
    
    except Exception:
        await session.rollback()
        raise


# ============================================================================
# 批量更新函数
# ============================================================================

async def bulk_update(
    session: AsyncSession,
    model,
    updates: List[dict],
    key_field: str = "id"
) -> int:
    """
    批量更新记录
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        updates: 更新数据列表，每项必须包含 key_field
        key_field: 用于匹配记录的字段名，默认 "id"
        
    返回：
        更新的记录数量
        
    示例：
        count = await bulk_update(session, Product, [
            {"id": 1, "price": 99.99, "stock": 100},
            {"id": 2, "price": 199.99, "stock": 200}
        ])
    """
    if not updates:
        return 0
    
    try:
        count = 0
        for update_data in updates:
            key_value = update_data.get(key_field)
            if not key_value:
                continue
            
            # 查询记录
            stmt = select(model).where(getattr(model, key_field) == key_value)
            result = await session.execute(stmt)
            instance = result.scalar_one_or_none()
            
            if instance:
                # 更新字段
                for key, value in update_data.items():
                    if key != key_field and hasattr(instance, key):
                        setattr(instance, key, value)
                count += 1
        
        await session.commit()
        return count
    
    except Exception:
        await session.rollback()
        raise


# ============================================================================
# 删除函数 - 支持复杂条件
# ============================================================================

async def delete(
    session: AsyncSession,
    model,
    filter_by: Optional[dict] = None,
    complex_filter: Any = None,
) -> List[int]:
    """
    删除记录并返回被删除的 ID 列表
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        filter_by: 简单过滤条件字典，例如 {"category": "old"}
        complex_filter: 复杂过滤条件，例如 model.id.in_([1, 2, 3])
        
    返回：
        被删除的记录 ID 列表
        
    示例：
        # 简单条件删除
        deleted_ids = await delete(session, Product, filter_by={"category": "old"})
        
        # IN 条件删除
        deleted_ids = await delete(
            session, 
            Product, 
            complex_filter=Product.id.in_([1, 2, 3])
        )
        
        # 复杂条件删除
        deleted_ids = await delete(
            session,
            Product,
            complex_filter=(Product.price < 10) & (Product.stock == 0)
        )
    """
    from sqlalchemy import delete as delete_
    
    try:
        # 构建删除语句
        stmt = delete_(model)
        
        # 应用简单过滤条件
        if filter_by:
            stmt = stmt.filter_by(**filter_by)
        
        # 应用复杂过滤条件
        if complex_filter is not None:
            stmt = stmt.where(complex_filter)
        
        # 如果没有任何条件，抛出错误（防止误删全表）
        if not filter_by and complex_filter is None:
            raise ValueError("必须提供 filter_by 或 complex_filter，不允许删除全表")
        
        # PostgreSQL 支持 RETURNING，可以直接返回删除的 ID
        if session.bind.dialect.name == "postgresql":
            stmt = stmt.returning(model.id)
            result = await session.execute(stmt)
            deleted_ids = [row[0] for row in result.fetchall()]
            await session.commit()
        else:
            # 其他数据库：先查询要删除的 ID，再执行删除
            query_stmt = select(model.id)
            
            if filter_by:
                query_stmt = query_stmt.filter_by(**filter_by)
            
            if complex_filter is not None:
                query_stmt = query_stmt.where(complex_filter)
            
            # 查询要删除的 ID
            result = await session.execute(query_stmt)
            deleted_ids = list(result.scalars().all())
            
            # 如果有记录需要删除，执行删除
            if deleted_ids:
                await session.execute(stmt)
            
            await session.commit()
        
        return deleted_ids
    
    except Exception:
        await session.rollback()
        raise


# ============================================================================
# 批量删除函数（简化版本）
# ============================================================================

async def bulk_delete(
    session: AsyncSession,
    model,
    filter_by: Optional[dict] = None,
    ids: Optional[List[int]] = None
) -> int:
    """
    批量删除记录（简化版本，只返回删除数量）
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        filter_by: 过滤条件字典
        ids: ID 列表
        
    返回：
        删除的记录数量
        
    示例：
        # 按 ID 删除
        count = await bulk_delete(session, Product, ids=[1, 2, 3])
        
        # 按条件删除
        count = await bulk_delete(session, Product, filter_by={"category": "old"})
    """
    try:
        if ids:
            deleted_ids = await delete(session, model, complex_filter=model.id.in_(ids))
        elif filter_by:
            deleted_ids = await delete(session, model, filter_by=filter_by)
        else:
            raise ValueError("必须提供 ids 或 filter_by")
        
        return len(deleted_ids)
    
    except Exception:
        await session.rollback()
        raise


# ============================================================================
# 查询函数
# ============================================================================

async def fetch_all(
    session: AsyncSession,
    model,
    filter_by: Optional[dict] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> List:
    """
    查询多条记录
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        filter_by: 过滤条件字典
        order_by: 排序字段名
        limit: 限制数量
        offset: 偏移量
        
    返回：
        记录列表
        
    示例：
        products = await fetch_all(
            session, Product,
            filter_by={"category": "电子"},
            order_by="price",
            limit=10
        )
    """
    stmt = select(model)
    
    # 添加过滤条件
    if filter_by:
        for key, value in filter_by.items():
            stmt = stmt.where(getattr(model, key) == value)
    
    # 添加排序
    if order_by:
        if order_by.startswith("-"):
            # 降序
            field = order_by[1:]
            stmt = stmt.order_by(getattr(model, field).desc())
        else:
            # 升序
            stmt = stmt.order_by(getattr(model, order_by))
    
    # 添加分页
    if offset:
        stmt = stmt.offset(offset)
    if limit:
        stmt = stmt.limit(limit)
    
    result = await session.execute(stmt)
    return result.scalars().all()


async def count_records(
    session: AsyncSession,
    model,
    filter_by: Optional[dict] = None
) -> int:
    """
    统计记录数量
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        filter_by: 过滤条件字典
        
    返回：
        记录数量
        
    示例：
        total = await count_records(session, Product, filter_by={"category": "电子"})
    """
    from sqlalchemy import func
    
    stmt = select(func.count()).select_from(model)
    
    if filter_by:
        for key, value in filter_by.items():
            stmt = stmt.where(getattr(model, key) == value)
    
    result = await session.execute(stmt)
    return result.scalar()


# ============================================================================
# 使用示例
# ============================================================================

"""
# ============================================================================
# 单条创建
# ============================================================================
product_id = await create(
    session, 
    Product, 
    {"name": "iPhone", "price": 999.99},
    filter_by={"name": "iPhone"}
)

# ============================================================================
# 批量创建（推荐）
# ============================================================================
count = await bulk_create(session, Product, [
    {"name": "产品1", "price": 99.99},
    {"name": "产品2", "price": 199.99}
])

# ============================================================================
# 批量创建（带去重）
# ============================================================================
count = await bulk_create(
    session, 
    Product, 
    data_list,
    filter_by="name"  # 根据 name 去重
)

# ============================================================================
# 批量创建（返回ID）
# ============================================================================
ids = await bulk_create_return_ids(session, Product, data_list)

# ============================================================================
# 批量创建（详细结果）
# ============================================================================
result = await bulk_create_detailed(session, Product, data_list, filter_by="name")
print(f"创建: {result['created']}, 跳过: {result['skipped']}")

# ============================================================================
# 批量更新
# ============================================================================
count = await bulk_update(session, Product, [
    {"id": 1, "price": 89.99},
    {"id": 2, "price": 189.99}
])

# ============================================================================
# 批量删除
# ============================================================================
count = await bulk_delete(session, Product, ids=[1, 2, 3])

# ============================================================================
# 查询
# ============================================================================
products = await fetch_all(
    session, Product,
    filter_by={"category": "电子"},
    order_by="-price",  # 价格降序
    limit=10
)

# ============================================================================
# 统计
# ============================================================================
total = await count_records(session, Product, filter_by={"category": "电子"})
"""
