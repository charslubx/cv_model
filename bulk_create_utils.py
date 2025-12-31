"""
批量创建工具函数 - 基于你的单条创建函数改写

支持 SQLAlchemy 2.0 异步
"""

from typing import List, Dict, Any, Optional, Union
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError


# ============================================================================
# 原始的单条创建函数（你提供的）
# ============================================================================

async def fetch_one(session, model, filter_by: dict):
    """辅助函数：查询单条记录"""
    stmt = select(model)
    for key, value in filter_by.items():
        stmt = stmt.where(getattr(model, key) == value)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def create(
    session,
    model,
    data: dict,
    filter_by: dict = None,
) -> int:
    """原始的单条创建函数"""
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
# 方案1：批量创建 - 使用 insert()（推荐，性能最佳）⭐
# ============================================================================

async def bulk_create(
    session: AsyncSession,
    model,
    data_list: List[dict],
    filter_by_key: Optional[str] = None,
) -> int:
    """
    批量创建 - 使用 insert() 语句（推荐）
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        data_list: 数据字典列表
        filter_by_key: 用于去重的字段名（可选）
        
    返回：
        创建的记录数量
        
    示例：
        count = await bulk_create(session, Product, [
            {"name": "产品1", "price": 99.99},
            {"name": "产品2", "price": 199.99}
        ])
    """
    if not data_list:
        return 0
    
    try:
        # 如果需要去重
        if filter_by_key:
            # 获取需要检查的值列表
            check_values = [d.get(filter_by_key) for d in data_list if d.get(filter_by_key)]
            
            # 查询已存在的记录
            stmt = select(getattr(model, filter_by_key)).where(
                getattr(model, filter_by_key).in_(check_values)
            )
            result = await session.execute(stmt)
            existing_values = set(result.scalars().all())
            
            # 过滤掉已存在的数据
            data_list = [
                d for d in data_list
                if d.get(filter_by_key) not in existing_values
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
# 方案2：批量创建 - 使用 add_all()（可以返回 ID 列表）
# ============================================================================

async def bulk_create_with_ids(
    session: AsyncSession,
    model,
    data_list: List[dict],
    filter_by_key: Optional[str] = None,
) -> List[int]:
    """
    批量创建 - 使用 add_all()，返回 ID 列表
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        data_list: 数据字典列表
        filter_by_key: 用于去重的字段名（可选）
        
    返回：
        创建的记录 ID 列表
        
    示例：
        ids = await bulk_create_with_ids(session, Product, [
            {"name": "产品1", "price": 99.99},
            {"name": "产品2", "price": 199.99}
        ])
        # 返回: [1, 2]
    """
    if not data_list:
        return []
    
    try:
        # 如果需要去重
        if filter_by_key:
            check_values = [d.get(filter_by_key) for d in data_list if d.get(filter_by_key)]
            
            stmt = select(getattr(model, filter_by_key)).where(
                getattr(model, filter_by_key).in_(check_values)
            )
            result = await session.execute(stmt)
            existing_values = set(result.scalars().all())
            
            data_list = [
                d for d in data_list
                if d.get(filter_by_key) not in existing_values
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
# 方案3：批量创建 - 详细结果（包含成功和失败信息）
# ============================================================================

async def bulk_create_detailed(
    session: AsyncSession,
    model,
    data_list: List[dict],
    filter_by_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    批量创建 - 返回详细结果
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        data_list: 数据字典列表
        filter_by_key: 用于去重的字段名（可选）
        
    返回：
        {
            "success": True/False,
            "created_count": 创建的数量,
            "skipped_count": 跳过的数量（已存在）,
            "total_count": 总数量,
            "created_ids": [ID列表]  # 可选
        }
        
    示例：
        result = await bulk_create_detailed(session, Product, [
            {"name": "产品1", "price": 99.99},
            {"name": "产品2", "price": 199.99}
        ], filter_by_key="name")
    """
    if not data_list:
        return {
            "success": True,
            "created_count": 0,
            "skipped_count": 0,
            "total_count": 0,
            "created_ids": []
        }
    
    total_count = len(data_list)
    skipped_count = 0
    
    try:
        # 如果需要去重
        if filter_by_key:
            check_values = [d.get(filter_by_key) for d in data_list if d.get(filter_by_key)]
            
            stmt = select(getattr(model, filter_by_key)).where(
                getattr(model, filter_by_key).in_(check_values)
            )
            result = await session.execute(stmt)
            existing_values = set(result.scalars().all())
            
            # 统计跳过的数量
            original_count = len(data_list)
            data_list = [
                d for d in data_list
                if d.get(filter_by_key) not in existing_values
            ]
            skipped_count = original_count - len(data_list)
            
            if not data_list:
                return {
                    "success": True,
                    "created_count": 0,
                    "skipped_count": skipped_count,
                    "total_count": total_count,
                    "created_ids": []
                }
        
        # 创建对象列表
        instances = [model(**data) for data in data_list]
        
        # 批量添加
        session.add_all(instances)
        await session.commit()
        
        # 获取 ID 列表
        created_ids = [inst.id for inst in instances]
        
        return {
            "success": True,
            "created_count": len(created_ids),
            "skipped_count": skipped_count,
            "total_count": total_count,
            "created_ids": created_ids
        }
    
    except Exception as e:
        await session.rollback()
        return {
            "success": False,
            "created_count": 0,
            "skipped_count": skipped_count,
            "total_count": total_count,
            "error": str(e)
        }


# ============================================================================
# 方案4：批量创建 - 分批插入（大数据量）
# ============================================================================

async def bulk_create_batched(
    session: AsyncSession,
    model,
    data_list: List[dict],
    filter_by_key: Optional[str] = None,
    batch_size: int = 1000,
) -> int:
    """
    批量创建 - 分批插入，适合大数据量
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        data_list: 数据字典列表
        filter_by_key: 用于去重的字段名（可选）
        batch_size: 每批的数量
        
    返回：
        创建的记录总数
        
    示例：
        count = await bulk_create_batched(
            session, Product, large_data_list, batch_size=1000
        )
    """
    if not data_list:
        return 0
    
    try:
        # 如果需要去重
        if filter_by_key:
            check_values = [d.get(filter_by_key) for d in data_list if d.get(filter_by_key)]
            
            stmt = select(getattr(model, filter_by_key)).where(
                getattr(model, filter_by_key).in_(check_values)
            )
            result = await session.execute(stmt)
            existing_values = set(result.scalars().all())
            
            data_list = [
                d for d in data_list
                if d.get(filter_by_key) not in existing_values
            ]
            
            if not data_list:
                return 0
        
        # 分批插入
        total_created = 0
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            
            stmt = insert(model).values(batch)
            result = await session.execute(stmt)
            await session.commit()
            
            total_created += result.rowcount
        
        return total_created
    
    except Exception:
        await session.rollback()
        raise


# ============================================================================
# 方案5：批量创建 - 容错版本（部分失败不影响其他）
# ============================================================================

async def bulk_create_tolerant(
    session: AsyncSession,
    model,
    data_list: List[dict],
    filter_by_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    批量创建 - 容错版本，部分失败不影响其他记录
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        data_list: 数据字典列表
        filter_by_key: 用于去重的字段名（可选）
        
    返回：
        {
            "success": True,
            "created_count": 成功创建的数量,
            "failed_count": 失败的数量,
            "total_count": 总数量,
            "created_ids": [成功创建的ID列表],
            "failed_items": [失败的数据项]
        }
    """
    if not data_list:
        return {
            "success": True,
            "created_count": 0,
            "failed_count": 0,
            "total_count": 0,
            "created_ids": [],
            "failed_items": []
        }
    
    created_ids = []
    failed_items = []
    
    # 如果需要去重，先批量检查
    existing_values = set()
    if filter_by_key:
        check_values = [d.get(filter_by_key) for d in data_list if d.get(filter_by_key)]
        
        stmt = select(getattr(model, filter_by_key)).where(
            getattr(model, filter_by_key).in_(check_values)
        )
        result = await session.execute(stmt)
        existing_values = set(result.scalars().all())
    
    # 逐条处理（使用 savepoint）
    for data in data_list:
        # 检查是否已存在
        if filter_by_key and data.get(filter_by_key) in existing_values:
            failed_items.append({
                "data": data,
                "reason": "已存在"
            })
            continue
        
        # 创建 savepoint
        async with session.begin_nested():
            try:
                instance = model(**data)
                session.add(instance)
                await session.flush()
                
                created_ids.append(instance.id)
                
            except Exception as e:
                failed_items.append({
                    "data": data,
                    "reason": str(e)
                })
    
    # 统一提交
    try:
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise
    
    return {
        "success": True,
        "created_count": len(created_ids),
        "failed_count": len(failed_items),
        "total_count": len(data_list),
        "created_ids": created_ids,
        "failed_items": failed_items
    }


# ============================================================================
# 方案6：批量创建 - 使用 returning()（仅 PostgreSQL）
# ============================================================================

async def bulk_create_returning(
    session: AsyncSession,
    model,
    data_list: List[dict],
) -> List[Dict[str, Any]]:
    """
    批量创建 - 使用 returning() 返回插入的数据
    
    注意：仅在 PostgreSQL 中可用
    
    参数：
        session: 数据库会话
        model: 数据库模型类
        data_list: 数据字典列表
        
    返回：
        插入的记录列表（包含 ID 和其他字段）
        
    示例：
        records = await bulk_create_returning(session, Product, [
            {"name": "产品1", "price": 99.99},
            {"name": "产品2", "price": 199.99}
        ])
        # 返回: [{"id": 1, "name": "产品1"}, {"id": 2, "name": "产品2"}]
    """
    if not data_list:
        return []
    
    try:
        # 使用 returning() 获取插入的数据
        stmt = insert(model).values(data_list).returning(model)
        
        result = await session.execute(stmt)
        inserted_records = result.all()
        await session.commit()
        
        # 转换为字典列表
        return [
            {
                "id": record.id,
                **{k: getattr(record, k) for k in data_list[0].keys()}
            }
            for record in inserted_records
        ]
    
    except Exception:
        await session.rollback()
        raise


# ============================================================================
# 使用示例
# ============================================================================

"""
# 示例1：简单批量创建
count = await bulk_create(session, Product, [
    {"name": "产品1", "price": 99.99},
    {"name": "产品2", "price": 199.99}
])
print(f"创建了 {count} 个产品")

# 示例2：批量创建并返回 ID
ids = await bulk_create_with_ids(session, Product, [
    {"name": "产品1", "price": 99.99},
    {"name": "产品2", "price": 199.99}
])
print(f"创建的 ID: {ids}")

# 示例3：带去重的批量创建
count = await bulk_create(
    session, 
    Product, 
    [
        {"name": "产品1", "price": 99.99},
        {"name": "产品2", "price": 199.99}
    ],
    filter_by_key="name"  # 根据 name 字段去重
)

# 示例4：详细结果
result = await bulk_create_detailed(
    session, 
    Product, 
    data_list,
    filter_by_key="name"
)
print(f"创建: {result['created_count']}, 跳过: {result['skipped_count']}")

# 示例5：大数据量分批插入
count = await bulk_create_batched(
    session, 
    Product, 
    large_data_list,
    batch_size=1000
)

# 示例6：容错插入
result = await bulk_create_tolerant(session, Product, data_list)
print(f"成功: {result['created_count']}, 失败: {result['failed_count']}")
if result['failed_items']:
    for item in result['failed_items']:
        print(f"失败: {item['data']}, 原因: {item['reason']}")
"""
