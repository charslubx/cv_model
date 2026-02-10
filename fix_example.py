"""
SQLAlchemy JOIN 字段覆盖问题示例和解决方案
"""

from sqlalchemy import select, and_

# ❌ 错误的方式 - 会导致 operation_tag 被覆盖
async def _fetch_permission_resource_wrong(filter_by, complex_conditions):
    """
    问题：使用 row.__dict__ 会导致 JOIN 表的同名字段覆盖原表字段
    """
    async with g.db_async_session() as session:
        query = select(PermissionResource).join(
            PermissionList,
            PermissionResource.permission_id == PermissionList.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        print(result.scalars().all())  # 第一次调用 - 这里看到的数据是正确的
        
        # ❌ 问题出在这里：
        # 1. 再次调用 .all() 
        # 2. 使用 __dict__ 导致字段被覆盖
        return [
            {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
            for row in result.scalars().all()  # 第二次调用
        ]


# ✅ 解决方案 1：直接访问对象属性（推荐）
async def _fetch_permission_resource_solution1(filter_by, complex_conditions):
    """
    直接访问对象属性，不使用 __dict__
    """
    async with g.db_async_session() as session:
        query = select(PermissionResource).join(
            PermissionList,
            PermissionResource.permission_id == PermissionList.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        rows = result.scalars().all()  # ✅ 只调用一次
        
        # ✅ 直接访问属性
        return [
            {
                'resource_id': row.resource_id,
                'resource_type': row.resource_type,
                'operation_tag': row.operation_tag,  # ✅ SQLAlchemy 会返回正确的值
                'permission_id': row.permission_id,
                'update_time': row.update_time,
                'create_time': row.create_time
            }
            for row in rows
        ]


# ✅ 解决方案 2：显式指定查询列
async def _fetch_permission_resource_solution2(filter_by, complex_conditions):
    """
    显式指定要查询的列，避免字段冲突
    """
    async with g.db_async_session() as session:
        # ✅ 只查询 PermissionResource 的列
        query = select(
            PermissionResource.resource_id,
            PermissionResource.resource_type,
            PermissionResource.operation_tag,  # 明确是 PermissionResource 的字段
            PermissionResource.permission_id,
            PermissionResource.update_time,
            PermissionResource.create_time
        ).join(
            PermissionList,
            PermissionResource.permission_id == PermissionList.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        rows = result.all()  # 返回的是 Row 对象，不是 ORM 实例
        
        return [
            {
                'resource_id': row.resource_id,
                'resource_type': row.resource_type,
                'operation_tag': row.operation_tag,
                'permission_id': row.permission_id,
                'update_time': row.update_time,
                'create_time': row.create_time
            }
            for row in rows
        ]


# ✅ 解决方案 3：使用 load_only
async def _fetch_permission_resource_solution3(filter_by, complex_conditions):
    """
    使用 load_only 只加载需要的字段
    """
    from sqlalchemy.orm import load_only
    
    async with g.db_async_session() as session:
        query = select(PermissionResource).options(
            load_only(
                PermissionResource.resource_id,
                PermissionResource.resource_type,
                PermissionResource.operation_tag,
                PermissionResource.permission_id,
                PermissionResource.update_time,
                PermissionResource.create_time
            )
        ).join(
            PermissionList,
            PermissionResource.permission_id == PermissionList.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        rows = result.scalars().all()
        
        return [
            {
                'resource_id': row.resource_id,
                'resource_type': row.resource_type,
                'operation_tag': row.operation_tag,
                'permission_id': row.permission_id,
                'update_time': row.update_time,
                'create_time': row.create_time
            }
            for row in rows
        ]


# ✅ 解决方案 4：使用 SQLAlchemy inspect
async def _fetch_permission_resource_solution4(filter_by, complex_conditions):
    """
    使用 sqlalchemy.inspect 来安全地获取对象属性
    """
    from sqlalchemy import inspect as sa_inspect
    
    async with g.db_async_session() as session:
        query = select(PermissionResource).join(
            PermissionList,
            PermissionResource.permission_id == PermissionList.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        rows = result.scalars().all()
        
        return [
            {
                c.key: getattr(row, c.key)
                for c in sa_inspect(row).mapper.column_attrs
            }
            for row in rows
        ]


# 🔍 调试函数 - 用于验证问题
async def debug_fetch_permission_resource(filter_by, complex_conditions):
    """
    调试函数：对比不同方法的结果
    """
    async with g.db_async_session() as session:
        query = select(PermissionResource).join(
            PermissionList,
            PermissionResource.permission_id == PermissionList.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        rows = result.scalars().all()
        
        if rows:
            first_row = rows[0]
            
            print("=" * 80)
            print("调试信息：")
            print(f"直接访问属性: row.operation_tag = {first_row.operation_tag}")
            print(f"__dict__ 中的值: row.__dict__.get('operation_tag') = {first_row.__dict__.get('operation_tag')}")
            print(f"__dict__ 所有键: {[k for k in first_row.__dict__.keys() if not k.startswith('_')]}")
            print("=" * 80)
            
            # 检查所有 operation_tag 的唯一值
            operation_tags_from_attr = set(row.operation_tag for row in rows)
            operation_tags_from_dict = set(row.__dict__.get('operation_tag') for row in rows)
            
            print(f"通过属性访问的唯一值数量: {len(operation_tags_from_attr)}")
            print(f"通过 __dict__ 访问的唯一值数量: {len(operation_tags_from_dict)}")
            print(f"属性访问的值: {operation_tags_from_attr}")
            print(f"__dict__ 的值: {operation_tags_from_dict}")
            print("=" * 80)
        
        return []


# 📝 使用建议
"""
使用建议：

1. ✅ 优先使用方案 1（直接访问属性）- 最简单、最直观
2. ✅ 如果担心性能，使用方案 2（显式指定列）- 减少数据传输
3. ✅ 如果需要动态字段，使用方案 4（inspect）
4. ❌ 永远不要在 JOIN 查询后使用 row.__dict__.items()

问题根源：
- PermissionResource 和 PermissionList 两个表都有 operation_tag 字段
- JOIN 后使用 __dict__ 时，后加载的字段会覆盖先加载的字段
- 导致所有记录的 operation_tag 都变成同一个值（通常是 JOIN 表的值）

关键点：
- SQLAlchemy 的属性访问器能正确区分表字段
- __dict__ 只是一个普通字典，无法区分同名字段的来源
"""
