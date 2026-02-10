# 问题根本原因与解决方案

## 问题描述

返回的数据数量正确，但所有记录的 `operation_tag` 都变成了 "Margin Owner Check"。
去除 print 语句后就正常了。

## 根本原因

**`result.scalars().all()` 被调用了两次！**

```python
# 问题代码
result = await session.execute(query)
print(result.scalars().all())  # 第一次调用 - 消耗结果集
return [
    {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
    for row in result.scalars().all()  # 第二次调用 - 结果集已被消耗！
]
```

在 SQLAlchemy 中，`result.scalars().all()` 会**消耗结果集**。第二次调用时：
- 结果集已经被第一次调用消耗
- 返回的是空列表、缓存数据或错误数据
- 导致 operation_tag 显示错误的值

## 解决方案

**只调用一次 `.all()`，将结果保存到变量中：**

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
        rows = result.scalars().all()  # ✅ 只调用一次，保存到变量
        
        # 如果需要调试，使用已保存的变量
        print(f"查询到 {len(rows)} 条记录")
        # print(rows[:5])  # 打印前5条用于调试
        
        return [
            {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
            for row in rows  # ✅ 使用保存的变量，不再调用 all()
        ]
```

## 为什么去除 print 就正常了？

因为去除 print 后，代码变成：

```python
result = await session.execute(query)
# print 被删除了，所以 result.scalars().all() 没有被调用
return [
    {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
    for row in result.scalars().all()  # 这是第一次也是唯一一次调用
]
```

此时 `result.scalars().all()` **只被调用了一次**，所以能正确获取数据。

## 验证方法

修改后，你可以这样验证：

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
        
        # 验证 operation_tag 的值
        operation_tags = [row.operation_tag for row in rows]
        unique_tags = set(operation_tags)
        
        print(f"总记录数: {len(rows)}")
        print(f"唯一的 operation_tag 值: {unique_tags}")
        print(f"是否都是 'Margin Owner Check': {unique_tags == {'Margin Owner Check'}}")
        
        return [
            {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
            for row in rows
        ]
```

## 关键点总结

1. ❌ **错误**：多次调用 `result.scalars().all()`
2. ✅ **正确**：只调用一次 `result.scalars().all()`，保存到变量
3. ✅ **正确**：需要多次使用时，使用保存的变量而不是重新调用

这就是为什么你的 operation_tag 值出现问题的根本原因！
