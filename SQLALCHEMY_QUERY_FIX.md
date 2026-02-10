# SQLAlchemy 查询结果字段覆盖问题分析与解决方案

## 问题描述

在使用 SQLAlchemy 查询 `PermissionResource` 数据时，发现返回的数据数量是正确的，但是所有记录的 `operation_tag` 字段都变成了同一个值 "Margin Owner Check"，而不是预期的多样化值（如 read, write, upload_file, approve, reject 等）。

## 更新：真正的问题原因

**问题不是数据缺失，而是 `operation_tag` 字段被 JOIN 的表覆盖了！**

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

### 问题 1：`result.scalars().all()` 被调用了两次

这会导致第二次调用时结果集已被消耗（但这不是主要问题）。

### **问题 2：字段冲突（主要问题）**

**`PermissionResource` 和 `PermissionList` 表都有 `operation_tag` 字段！**

当使用 `.join()` 连接两个表时，如果两个表都有同名字段，SQLAlchemy 可能会出现字段覆盖的情况。特别是在使用 `row.__dict__` 转换为字典时，后加载的字段会覆盖先加载的字段。

这就是为什么：
- 打印语句能看到正确的 ORM 对象（SQLAlchemy 内部能正确区分）
- 但转换为字典后，`operation_tag` 被 `PermissionList.operation_tag` 覆盖了
- 所有记录的 `operation_tag` 都变成了 "Margin Owner Check"（这可能是某个 `PermissionList` 记录的值）

## 解决方案

### 方案 1：显式指定要查询的列（强烈推荐）

只查询 `PermissionResource` 的列，避免字段冲突：

```python
async def _fetch_permission_resource(filter_by, complex_conditions):
    async with g.db_async_session() as session:
        # ✅ 显式指定只选择 PermissionResource 的列
        query = select(
            PermissionResource.resource_id,
            PermissionResource.resource_type,
            PermissionResource.operation_tag,  # 明确指定是 PermissionResource 的 operation_tag
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
        rows = result.all()  # 注意：不用 scalars()，因为返回的是元组
        
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
```

### 方案 2：使用 load_only 加载特定字段

```python
from sqlalchemy.orm import load_only

async def _fetch_permission_resource(filter_by, complex_conditions):
    async with g.db_async_session() as session:
        # ✅ 只加载 PermissionResource 的特定字段
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
            {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
            for row in rows
        ]
```

### 方案 3：直接访问对象属性而不是 __dict__

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
        
        # ✅ 直接访问属性，不使用 __dict__
        return [
            {
                'resource_id': row.resource_id,
                'resource_type': row.resource_type,
                'operation_tag': row.operation_tag,  # SQLAlchemy 会正确返回 PermissionResource 的值
                'permission_id': row.permission_id,
                'update_time': row.update_time,
                'create_time': row.create_time
            }
            for row in rows
        ]
```

### 方案 4：使用表别名明确区分字段

```python
from sqlalchemy import alias

async def _fetch_permission_resource(filter_by, complex_conditions):
    async with g.db_async_session() as session:
        # 为表创建别名
        pr = alias(PermissionResource, name='pr')
        pl = alias(PermissionList, name='pl')
        
        query = select(pr).join(
            pl,
            pr.c.permission_id == pl.c.permission_id
        ).filter_by(**filter_by)

        if complex_conditions:
            query = query.filter(and_(*complex_conditions))

        result = await session.execute(query)
        rows = result.all()
        
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
```

## 为什么会发生字段覆盖？

### 原因分析

当你使用 `row.__dict__` 获取对象属性时，SQLAlchemy 会将所有加载的列放入实例的 `__dict__` 中。如果 JOIN 的两个表有同名字段，后面的字段可能会覆盖前面的字段。

从你的日志可以看到，`PermissionList` 表中有一些记录的 `operation_tag` 值为 "Margin Owner Check"：

```
<PermissionResource(permission_id=per-3HIU8CoJ, resource_id=excursion, resource_type=general_source, operation_tag=Margin Owner Check)>
```

如果 `PermissionList` 表也有 `operation_tag` 字段，并且在 JOIN 后被加载到 `PermissionResource` 对象的 `__dict__` 中，就会导致原始的 `operation_tag` 值被覆盖。

### 验证方法

你可以通过以下方式验证：

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
        
        # 调试：检查对象属性
        if rows:
            first_row = rows[0]
            print(f"直接访问属性: {first_row.operation_tag}")
            print(f"__dict__ 内容: {first_row.__dict__}")
            print(f"所有键: {list(first_row.__dict__.keys())}")
        
        return []
```

### 最佳实践

**永远不要依赖 `__dict__` 来获取 ORM 对象的属性值！**

应该使用：
1. 直接访问属性：`row.operation_tag`
2. 使用 `inspect`：`from sqlalchemy import inspect; inspect(row).dict`
3. 显式指定查询列

## 其他注意事项

### 1. JOIN 可能导致重复数据

从你的打印结果中可以看到，有很多重复的记录。这可能是因为 JOIN 操作导致的笛卡尔积。如果需要去重：

```python
query = select(PermissionResource).join(...).filter_by(**filter_by).distinct()
```

### 2. 调试建议

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
        
        # ✅ 使用属性访问而不是 __dict__
        operation_tags = set(row.operation_tag for row in rows)
        print(f"Total records: {total_count}")
        print(f"Unique operation_tags: {operation_tags}")
        
        # ✅ 正确的转换方式
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
```

## 总结

主要问题是 **使用 `row.__dict__` 时，JOIN 表的同名字段覆盖了原表的字段值**。

### 关键点

1. ✅ **不要使用 `row.__dict__.items()` 来获取 ORM 对象的属性**
2. ✅ **直接访问对象属性**：`row.operation_tag` 而不是 `row.__dict__['operation_tag']`
3. ✅ **或者显式指定查询的列**，避免 JOIN 带来的字段冲突
4. ✅ **避免多次调用 `.all()`**，将结果保存到变量中

### 推荐的最终解决方案

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
        rows = result.scalars().all()  # 只调用一次
        
        # 直接访问属性，不使用 __dict__
        return [
            {
                'resource_id': row.resource_id,
                'resource_type': row.resource_type,
                'operation_tag': row.operation_tag,  # 正确获取值
                'permission_id': row.permission_id,
                'update_time': row.update_time,
                'create_time': row.create_time
            }
            for row in rows
        ]
```

这样就能确保 `operation_tag` 获取的是 `PermissionResource` 表的真实值，而不是被 JOIN 表覆盖的值。
