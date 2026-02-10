"""
测试 operation_tag 字段冲突问题的诊断和修复代码

请将这段代码中的修复方案应用到你的实际代码中
"""

# ========== 方案 1：直接访问属性（最简单，立即可测试）==========
async def _fetch_permission_resource_fix1(filter_by, complex_conditions):
    """
    修复方案 1：不使用 __dict__，直接访问对象属性
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
        
        # ✅ 关键修改：直接访问属性，不使用 __dict__
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


# ========== 方案 2：显式指定查询列 ==========
async def _fetch_permission_resource_fix2(filter_by, complex_conditions):
    """
    修复方案 2：显式指定只查询 PermissionResource 的列
    """
    async with g.db_async_session() as session:
        # ✅ 只查询需要的列
        query = select(
            PermissionResource.resource_id,
            PermissionResource.resource_type,
            PermissionResource.operation_tag,  # 明确指定是 PermissionResource 的列
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
        rows = result.all()  # 注意：这里用 all() 不用 scalars()
        
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


# ========== 诊断代码：找出问题根源 ==========
async def diagnose_operation_tag_issue(filter_by, complex_conditions):
    """
    诊断代码：比较不同访问方式的结果
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
        
        if not rows:
            print("没有查询到数据")
            return []
        
        print("\n========== 诊断信息 ==========")
        print(f"查询到记录数: {len(rows)}")
        
        # 取第一条记录进行诊断
        first_row = rows[0]
        
        print(f"\n第一条记录诊断:")
        print(f"1. 直接访问属性: row.operation_tag = '{first_row.operation_tag}'")
        print(f"2. 通过 __dict__ 访问: row.__dict__['operation_tag'] = '{first_row.__dict__.get('operation_tag')}'")
        print(f"3. 两者是否相同: {first_row.operation_tag == first_row.__dict__.get('operation_tag')}")
        
        print(f"\n__dict__ 中的所有键:")
        for key in first_row.__dict__.keys():
            if not key.startswith('_'):
                print(f"  - {key}: {first_row.__dict__[key]}")
        
        # 统计 operation_tag 的分布
        print(f"\n通过属性访问的 operation_tag 分布:")
        tag_by_attr = {}
        for row in rows:
            tag = row.operation_tag
            tag_by_attr[tag] = tag_by_attr.get(tag, 0) + 1
        for tag, count in sorted(tag_by_attr.items()):
            print(f"  - '{tag}': {count} 条")
        
        print(f"\n通过 __dict__ 访问的 operation_tag 分布:")
        tag_by_dict = {}
        for row in rows:
            tag = row.__dict__.get('operation_tag')
            tag_by_dict[tag] = tag_by_dict.get(tag, 0) + 1
        for tag, count in sorted(tag_by_dict.items()):
            print(f"  - '{tag}': {count} 条")
        
        print("\n========== 诊断结束 ==========\n")
        
        # 如果两者不同，说明是字段冲突问题
        if first_row.operation_tag != first_row.__dict__.get('operation_tag'):
            print("⚠️  检测到字段冲突！")
            print("   - 属性访问返回正确值")
            print("   - __dict__ 访问返回错误值")
            print("   - 解决方案：使用方案1或方案2的代码")
        else:
            print("✅ 未检测到字段冲突，问题可能在其他地方")
        
        return []


# ========== 对比原始方法（有问题的）==========
async def _fetch_permission_resource_original(filter_by, complex_conditions):
    """
    原始方法（有问题）
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
        
        # ❌ 问题：使用 __dict__ 时字段可能被覆盖
        return [
            {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
            for row in rows
        ]


"""
使用说明：

1. 先运行诊断代码 diagnose_operation_tag_issue()，查看诊断输出
2. 如果诊断显示"检测到字段冲突"，使用方案1或方案2
3. 推荐使用方案1（最简单，改动最小）

预期结果：
- 如果是字段冲突问题，方案1和方案2都能解决
- operation_tag 会显示正确的值（read, write, button:xxx 等）
"""
