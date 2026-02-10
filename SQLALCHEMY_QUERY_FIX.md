# SQLAlchemy 查询结果缺失问题分析与解决方案

## 问题描述

在使用 SQLAlchemy 查询 `PermissionResource` 数据时，发现通过 ORM 获取的结果与直接执行打印出的 SQL 语句得到的结果不一致，特别是缺少了 `operation_tag` 以 `button:` 开头的数据。

## 问题代码

```python
async def _fetch_permission_resource(filter_by, complex_conditions):
    async with g.db_async_session() as session:
        query = select(PermissionResource).join(
            PermissionList,
            PermissionResource.permission_id == PermissionList.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        print(result.scalars().all())  # ⚠️ 第一次调用
        return [
            {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
            for row in result.scalars().all()  # ⚠️ 第二次调用 - 问题所在！
        ]
```

## 根本原因

**`result.scalars().all()` 被调用了两次！**

1. 第一次在 `print()` 语句中
2. 第二次在 `return` 的列表推导式中

在 SQLAlchemy 中，当你调用 `.all()` 方法时，它会**消耗掉结果集**。一旦结果集被消耗，再次调用 `.all()` 会返回空列表或使用缓存数据（取决于具体实现）。

这就是为什么：
- 打印语句能看到完整数据（包括 `operation_tag` 以 `button:` 开头的记录）
- 但函数返回的数据不完整或为空

## 解决方案

### 方案 1：先将结果保存到变量（推荐）

```python
async def _fetch_permission_resource(filter_by, complex_conditions):
    async with g.db_async_session() as session:
        query = select(PermissionResource).join(
            PermissionList,
            PermissionResource.permission_id == PermissionList.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        rows = result.scalars().all()  # ✅ 只调用一次，保存结果
        
        print(rows)  # 调试输出
        
        return [
            {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
            for row in rows  # ✅ 使用已保存的结果
        ]
```

### 方案 2：使用 `fetchall()` 后再处理

```python
async def _fetch_permission_resource(filter_by, complex_conditions):
    async with g.db_async_session() as session:
        query = select(PermissionResource).join(
            PermissionList,
            PermissionResource.permission_id == PermissionList.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        rows = result.scalars().all()
        
        # 转换为字典
        data = [
            {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
            for row in rows
        ]
        
        print(f"Found {len(data)} records")
        return data
```

### 方案 3：分离调试和生产代码

```python
async def _fetch_permission_resource(filter_by, complex_conditions):
    async with g.db_async_session() as session:
        query = select(PermissionResource).join(
            PermissionList,
            PermissionResource.permission_id == PermissionList.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        rows = result.scalars().all()
        
        # 可选：添加日志记录而不是 print
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Query returned {len(rows)} PermissionResource records")
        
        return [
            {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
            for row in rows
        ]
```

## 其他注意事项

### 1. JOIN 可能导致重复数据

从你的打印结果中可以看到，有很多重复的记录：

```
<PermissionResource(permission_id=per-pO52y6mk, resource_id=doc-cFmST8XZ, resource_type=online_document, operation_tag=read)>
<PermissionResource(permission_id=per-pO52y6mk, resource_id=doc-cFmST8XZ, resource_type=online_document, operation_tag=read)>
<PermissionResource(permission_id=per-pO52y6mk, resource_id=doc-cFmST8XZ, resource_type=online_document, operation_tag=read)>
```

这可能是因为 JOIN 操作导致的笛卡尔积或多对多关系。如果需要去重，可以：

```python
# 添加 distinct
query = select(PermissionResource).join(
    PermissionList,
    PermissionResource.permission_id == PermissionList.permission_id
).filter_by(**filter_by).distinct()
```

### 2. 检查 JOIN 条件

如果直接执行 SQL 能获取到 `button:` 开头的数据，但 ORM 查询获取不到，可能还需要检查：

- JOIN 的类型（INNER JOIN vs LEFT JOIN）
- PermissionList 表中的数据是否完整
- 是否需要添加额外的过滤条件

```python
# 如果需要 LEFT JOIN
from sqlalchemy import select
query = select(PermissionResource).outerjoin(
    PermissionList,
    PermissionResource.permission_id == PermissionList.permission_id
).filter_by(**filter_by)
```

### 3. 调试建议

在修复后，添加详细的日志来验证：

```python
async def _fetch_permission_resource(filter_by, complex_conditions):
    async with g.db_async_session() as session:
        query = select(PermissionResource).join(
            PermissionList,
            PermissionResource.permission_id == PermissionList.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        rows = result.scalars().all()
        
        # 详细统计
        total_count = len(rows)
        button_count = sum(1 for row in rows if row.operation_tag.startswith('button:'))
        
        print(f"Total records: {total_count}")
        print(f"Records with 'button:' prefix: {button_count}")
        
        # 打印所有 operation_tag 的值（去重）
        operation_tags = set(row.operation_tag for row in rows)
        print(f"Unique operation_tags: {operation_tags}")
        
        return [
            {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
            for row in rows
        ]
```

## 总结

主要问题是 **`result.scalars().all()` 被调用了两次**，导致第二次调用时结果集已被消耗。修复方法是将查询结果保存到变量中，然后重复使用这个变量，而不是多次调用 `.all()` 方法。

如果修复后仍然缺少 `button:` 开头的数据，需要进一步检查 JOIN 条件、表数据完整性以及过滤逻辑。
