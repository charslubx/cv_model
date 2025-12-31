# delete 函数使用指南

## 📋 你的原始代码问题分析

### 问题1：逻辑混乱
```python
# 原代码问题
if filter_by:
    query_stmt = select(model.id).filter_by(**filter_by)
if complex_filter is not None:  # 这里会覆盖上面的 query_stmt
    query_stmt = select(model.id).filter(complex_filter)
```

### 问题2：stmt 没有应用 complex_filter
```python
stmt = delete_(model)
if filter_by:
    stmt = stmt.filter_by(**filter_by)
# 缺少：应用 complex_filter 到 stmt
```

---

## ✅ 改进后的 delete 函数

### 完整代码

```python
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
    """
    from sqlalchemy import delete as delete_
    
    try:
        # 构建删除语句
        stmt = delete_(model)
        
        # 应用简单过滤条件
        if filter_by:
            stmt = stmt.filter_by(**filter_by)
        
        # 应用复杂过滤条件 ⭐ 关键改进
        if complex_filter is not None:
            stmt = stmt.where(complex_filter)
        
        # 防止误删全表
        if not filter_by and complex_filter is None:
            raise ValueError("必须提供 filter_by 或 complex_filter")
        
        # PostgreSQL 支持 RETURNING
        if session.bind.dialect.name == "postgresql":
            stmt = stmt.returning(model.id)
            result = await session.execute(stmt)
            deleted_ids = [row[0] for row in result.fetchall()]
            await session.commit()
        else:
            # 其他数据库：先查询 ID，再删除
            query_stmt = select(model.id)
            
            # ⭐ 关键改进：同时应用两种条件
            if filter_by:
                query_stmt = query_stmt.filter_by(**filter_by)
            
            if complex_filter is not None:
                query_stmt = query_stmt.where(complex_filter)
            
            # 查询要删除的 ID
            result = await session.execute(query_stmt)
            deleted_ids = list(result.scalars().all())
            
            # 如果有记录，执行删除
            if deleted_ids:
                await session.execute(stmt)
            
            await session.commit()
        
        return deleted_ids
    
    except Exception:
        await session.rollback()
        raise
```

---

## 🎯 主要改进点

### 1. stmt 应用 complex_filter
```python
# ✅ 正确：stmt 也需要应用 complex_filter
stmt = delete_(model)
if filter_by:
    stmt = stmt.filter_by(**filter_by)
if complex_filter is not None:
    stmt = stmt.where(complex_filter)  # ⭐ 关键
```

### 2. query_stmt 正确组合条件
```python
# ✅ 正确：同时应用两种条件，而不是覆盖
query_stmt = select(model.id)

if filter_by:
    query_stmt = query_stmt.filter_by(**filter_by)

if complex_filter is not None:
    query_stmt = query_stmt.where(complex_filter)
```

### 3. 防止误删全表
```python
# ✅ 安全检查
if not filter_by and complex_filter is None:
    raise ValueError("必须提供条件")
```

---

## 📖 使用示例

### 示例1：使用 IN 条件删除

```python
# 删除指定 ID 的产品
deleted_ids = await delete(
    session,
    Product,
    complex_filter=Product.id.in_([1, 2, 3])
)
print(f"删除了 {len(deleted_ids)} 个产品: {deleted_ids}")
```

### 示例2：使用 NOT IN 条件

```python
# 删除除指定 ID 外的所有产品
deleted_ids = await delete(
    session,
    Product,
    complex_filter=~Product.id.in_([1, 2])  # ~ 表示 NOT
)
```

### 示例3：简单条件

```python
# 按分类删除
deleted_ids = await delete(
    session,
    Product,
    filter_by={"category": "清仓"}
)
```

### 示例4：复杂条件（AND）

```python
# 删除价格低于100且库存为0的产品
deleted_ids = await delete(
    session,
    Product,
    complex_filter=(Product.price < 100) & (Product.stock == 0)
)
```

### 示例5：复杂条件（OR）

```python
from sqlalchemy import or_

# 删除库存为0或价格低于10的产品
deleted_ids = await delete(
    session,
    Product,
    complex_filter=or_(Product.stock == 0, Product.price < 10)
)
```

### 示例6：组合使用简单条件和复杂条件

```python
# 删除电子分类中价格高于1000的产品
deleted_ids = await delete(
    session,
    Product,
    filter_by={"category": "电子"},
    complex_filter=Product.price > 1000
)
```

### 示例7：使用 LIKE 条件

```python
# 删除名称包含"老产品"的记录
deleted_ids = await delete(
    session,
    Product,
    complex_filter=Product.name.like("%老产品%")
)
```

### 示例8：使用 BETWEEN

```python
from sqlalchemy import between

# 删除价格在100到500之间的产品
deleted_ids = await delete(
    session,
    Product,
    complex_filter=between(Product.price, 100, 500)
)
```

---

## 🚀 在 FastAPI 中使用

### 示例1：批量删除接口（IN 条件）

```python
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

app = FastAPI()

@app.delete("/products/batch")
async def batch_delete_products(
    ids: List[int],
    db: AsyncSession = Depends(get_db)
):
    """批量删除产品（按ID列表）"""
    deleted_ids = await delete(
        db,
        Product,
        complex_filter=Product.id.in_(ids)
    )
    
    return {
        "success": True,
        "deleted_count": len(deleted_ids),
        "deleted_ids": deleted_ids
    }
```

### 示例2：按条件删除接口

```python
@app.delete("/products/by-category/{category}")
async def delete_by_category(
    category: str,
    db: AsyncSession = Depends(get_db)
):
    """按分类删除产品"""
    deleted_ids = await delete(
        db,
        Product,
        filter_by={"category": category}
    )
    
    return {
        "success": True,
        "deleted_count": len(deleted_ids),
        "deleted_ids": deleted_ids
    }
```

### 示例3：清理库存接口（复杂条件）

```python
@app.delete("/products/clear-stock")
async def clear_old_stock(
    max_price: float = 50,
    db: AsyncSession = Depends(get_db)
):
    """清理低价且无库存的产品"""
    deleted_ids = await delete(
        db,
        Product,
        complex_filter=(Product.stock == 0) & (Product.price < max_price)
    )
    
    return {
        "success": True,
        "deleted_count": len(deleted_ids),
        "deleted_ids": deleted_ids
    }
```

### 示例4：动态条件删除

```python
from sqlalchemy import and_, or_

@app.delete("/products/advanced")
async def advanced_delete(
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    zero_stock: bool = False,
    categories: Optional[List[str]] = None,
    db: AsyncSession = Depends(get_db)
):
    """高级条件删除"""
    conditions = []
    
    # 价格范围
    if min_price is not None:
        conditions.append(Product.price >= min_price)
    if max_price is not None:
        conditions.append(Product.price <= max_price)
    
    # 零库存
    if zero_stock:
        conditions.append(Product.stock == 0)
    
    # 分类列表
    if categories:
        conditions.append(Product.category.in_(categories))
    
    if not conditions:
        raise HTTPException(status_code=400, detail="至少提供一个条件")
    
    # 组合所有条件（AND）
    complex_filter = and_(*conditions)
    
    deleted_ids = await delete(
        db,
        Product,
        complex_filter=complex_filter
    )
    
    return {
        "success": True,
        "deleted_count": len(deleted_ids),
        "deleted_ids": deleted_ids
    }
```

---

## 🔍 常见用法对比

| 场景 | 代码示例 |
|------|---------|
| **按 ID 列表删除** | `delete(session, Product, complex_filter=Product.id.in_([1,2,3]))` |
| **按单个条件** | `delete(session, Product, filter_by={"category": "old"})` |
| **按多个 AND 条件** | `delete(session, Product, complex_filter=(Product.price < 10) & (Product.stock == 0))` |
| **按 OR 条件** | `delete(session, Product, complex_filter=or_(cond1, cond2))` |
| **NOT IN** | `delete(session, Product, complex_filter=~Product.id.in_([1,2]))` |
| **LIKE** | `delete(session, Product, complex_filter=Product.name.like("%old%"))` |
| **组合简单和复杂条件** | `delete(session, Product, filter_by={"category": "电子"}, complex_filter=Product.price > 1000)` |

---

## ⚠️ 注意事项

### 1. 防止误删全表
```python
# ✗ 会抛出错误
deleted_ids = await delete(session, Product)  # 没有任何条件

# ✓ 必须提供条件
deleted_ids = await delete(session, Product, filter_by={"id": 1})
```

### 2. PostgreSQL vs 其他数据库

```python
# PostgreSQL：使用 RETURNING，一次查询
# DELETE FROM products WHERE id IN (1,2,3) RETURNING id;

# SQLite/MySQL：先查询后删除，两次查询
# SELECT id FROM products WHERE id IN (1,2,3);
# DELETE FROM products WHERE id IN (1,2,3);
```

### 3. 返回值

```python
deleted_ids = await delete(session, Product, ...)
# 返回: [1, 2, 3]  # 被删除的 ID 列表
```

---

## 🧪 运行测试

```bash
python test_delete_examples.py
```

测试内容：
- ✅ 简单条件删除
- ✅ IN 条件删除
- ✅ 复杂条件删除（AND/OR）
- ✅ NOT IN 条件
- ✅ LIKE 条件
- ✅ 组合条件
- ✅ 错误处理

---

## ✅ 总结

### 与原代码相比的改进

| 项目 | 原代码 | 改进后 |
|------|--------|--------|
| **stmt 应用 complex_filter** | ❌ 缺失 | ✅ 正确应用 |
| **query_stmt 条件组合** | ❌ 会覆盖 | ✅ 同时应用 |
| **防止误删** | ❌ 无检查 | ✅ 有安全检查 |
| **代码清晰度** | ⚠️ 逻辑混乱 | ✅ 清晰易懂 |

### 关键代码

```python
# ⭐ 核心：同时应用 filter_by 和 complex_filter
stmt = delete_(model)

if filter_by:
    stmt = stmt.filter_by(**filter_by)

if complex_filter is not None:
    stmt = stmt.where(complex_filter)  # 关键！
```

### 最常用场景

```python
# 1. IN 条件删除（最常用）
deleted_ids = await delete(
    session, Product, complex_filter=Product.id.in_([1, 2, 3])
)

# 2. 按条件删除
deleted_ids = await delete(
    session, Product, filter_by={"category": "old"}
)

# 3. 复杂条件删除
deleted_ids = await delete(
    session, Product, 
    complex_filter=(Product.price < 100) & (Product.stock == 0)
)
```

完整代码见：
- `db_utils.py` - delete 函数实现
- `test_delete_examples.py` - 完整测试示例
