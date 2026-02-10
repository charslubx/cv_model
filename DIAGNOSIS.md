# 诊断：operation_tag 返回错误值的真正原因

## 问题现状

1. 去除 print 后可以正常返回数据
2. 但所有 operation_tag 都是 "Margin Owner Check"
3. 直接执行 SQL 能看到正确的多样化的 operation_tag（read, write, button:xxx 等）
4. 通过 ORM 返回的 operation_tag 全部错误

## 可能的原因

### 1. PermissionList 表也有 operation_tag 字段

**请检查：PermissionList 表是否也有 operation_tag 字段？**

即使你认为没有冲突，但从症状来看，很可能：
- PermissionResource 表有 operation_tag（存储 read, write, button:xxx 等）
- PermissionList 表也有 operation_tag（可能存储 "Margin Owner Check" 等）

当使用 `row.__dict__.items()` 时，两个表的字段都被加载，后加载的覆盖了先加载的。

### 2. 验证方法

请在你的代码中添加以下诊断代码：

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
        
        if rows:
            first_row = rows[0]
            
            # 诊断 1：直接访问属性
            print(f"直接访问 operation_tag: {first_row.operation_tag}")
            
            # 诊断 2：查看 __dict__ 的所有键
            print(f"__dict__ 的所有键: {list(first_row.__dict__.keys())}")
            
            # 诊断 3：查看 __dict__ 中的 operation_tag
            print(f"__dict__['operation_tag']: {first_row.__dict__.get('operation_tag')}")
            
            # 诊断 4：使用 inspect
            from sqlalchemy import inspect
            mapper = inspect(first_row)
            print(f"所有列: {[col.key for col in mapper.mapper.columns]}")
            
            # 诊断 5：比较两种访问方式
            print(f"属性访问 == __dict__ 访问: {first_row.operation_tag == first_row.__dict__.get('operation_tag')}")
        
        return [
            {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
            for row in rows
        ]
```

### 3. 解决方案 A：直接访问属性（推荐）

如果 `first_row.operation_tag` 显示正确值，但 `first_row.__dict__['operation_tag']` 显示错误值，说明确实是字段冲突。

**解决方法：不要使用 `__dict__`，直接访问属性：**

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
        
        # ✅ 直接访问对象属性，不使用 __dict__
        return [
            {
                'resource_id': row.resource_id,
                'resource_type': row.resource_type,
                'operation_tag': row.operation_tag,  # 直接访问属性
                'permission_id': row.permission_id,
                'update_time': row.update_time,
                'create_time': row.create_time
            }
            for row in rows
        ]
```

### 4. 解决方案 B：只查询 PermissionResource 的列

```python
async def _fetch_permission_resource(filter_by, complex_conditions):
    async with g.db_async_session() as session:
        # ✅ 显式指定只选择 PermissionResource 的列
        query = select(
            PermissionResource.resource_id,
            PermissionResource.resource_type,
            PermissionResource.operation_tag,
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
        rows = result.all()  # 注意：不用 scalars()
        
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

### 5. 解决方案 C：检查 PermissionList 模型定义

请检查你的 PermissionList 模型定义，看是否有 operation_tag 字段：

```python
# 检查这个
class PermissionList(Base):
    __tablename__ = 'opeda_permission'
    
    permission_id = Column(...)
    status = Column(...)
    operation_tag = Column(...)  # ← 如果有这个字段，就是冲突的根源！
```

如果 PermissionList 确实有 operation_tag 字段，那就是字段冲突导致的问题。

## 为什么直接执行 SQL 是正确的？

当你直接执行 SQL 时：
```sql
SELECT opeda_permission_resource.operation_tag FROM ...
```

SQL 明确指定了是 `opeda_permission_resource` 表的 `operation_tag`，所以返回正确的值。

但使用 ORM 的 `select(PermissionResource).join(PermissionList)` 时，SQLAlchemy 会加载所有相关的列，如果两个表都有同名列，使用 `__dict__` 时可能会出现覆盖。

## 立即测试的最简单方案

**立即把这段代码改成：**

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
        
        # ✅ 测试：直接访问属性
        test_data = [
            {
                'resource_id': row.resource_id,
                'resource_type': row.resource_type,
                'operation_tag': row.operation_tag,  # 直接访问
                'permission_id': row.permission_id,
            }
            for row in rows
        ]
        
        print(f"前5条记录的 operation_tag: {[d['operation_tag'] for d in test_data[:5]]}")
        
        return test_data
```

**如果这样修改后 operation_tag 变正确了，说明问题就是 `__dict__` 的字段冲突！**
