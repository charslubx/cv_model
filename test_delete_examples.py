"""
测试 delete 函数的各种用法示例
"""

import asyncio
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, DateTime
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, AsyncAttrs

from db_utils import create, bulk_create, delete, fetch_all


# ============================================================================
# 测试数据库配置
# ============================================================================

DATABASE_URL = "sqlite+aiosqlite:///./test_delete.db"

async_engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)


# ============================================================================
# 测试模型
# ============================================================================

class Base(AsyncAttrs, DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    stock: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ============================================================================
# 初始化数据库
# ============================================================================

async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("✓ 数据库已初始化\n")


async def seed_data():
    """插入测试数据"""
    async with AsyncSessionLocal() as session:
        products = [
            {"name": "iPhone 15", "category": "电子", "price": 999.99, "stock": 100},
            {"name": "MacBook Pro", "category": "电子", "price": 2499.99, "stock": 50},
            {"name": "iPad Air", "category": "电子", "price": 599.99, "stock": 200},
            {"name": "AirPods Pro", "category": "配件", "price": 249.99, "stock": 500},
            {"name": "Magic Mouse", "category": "配件", "price": 99.99, "stock": 300},
            {"name": "老产品1", "category": "清仓", "price": 9.99, "stock": 0},
            {"name": "老产品2", "category": "清仓", "price": 5.99, "stock": 0},
            {"name": "老产品3", "category": "清仓", "price": 3.99, "stock": 0},
        ]
        
        await bulk_create(session, Product, products)
        print("✓ 测试数据已插入\n")


# ============================================================================
# 测试用例
# ============================================================================

async def test_delete_by_simple_filter():
    """测试1：简单条件删除"""
    print("=" * 70)
    print("测试1：简单条件删除（filter_by）")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 删除清仓分类的产品
        deleted_ids = await delete(
            session,
            Product,
            filter_by={"category": "清仓"}
        )
        
        print(f"✓ 删除了 {len(deleted_ids)} 个产品")
        print(f"✓ 被删除的 ID: {deleted_ids}\n")


async def test_delete_by_in_condition():
    """测试2：使用 IN 条件删除"""
    print("=" * 70)
    print("测试2：使用 IN 条件删除")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 查询现有产品
        products = await fetch_all(session, Product, limit=3)
        ids_to_delete = [p.id for p in products]
        
        print(f"准备删除 ID: {ids_to_delete}")
        
        # 使用 IN 条件删除
        deleted_ids = await delete(
            session,
            Product,
            complex_filter=Product.id.in_(ids_to_delete)
        )
        
        print(f"✓ 删除了 {len(deleted_ids)} 个产品")
        print(f"✓ 被删除的 ID: {deleted_ids}\n")


async def test_delete_by_complex_filter():
    """测试3：复杂条件删除"""
    print("=" * 70)
    print("测试3：复杂条件删除（多个条件组合）")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 删除价格低于100且库存为0的产品
        deleted_ids = await delete(
            session,
            Product,
            complex_filter=(Product.price < 100) & (Product.stock == 0)
        )
        
        print(f"✓ 删除了价格<100且库存=0的产品")
        print(f"✓ 删除了 {len(deleted_ids)} 个产品")
        print(f"✓ 被删除的 ID: {deleted_ids}\n")


async def test_delete_by_combined_conditions():
    """测试4：简单条件 + 复杂条件组合"""
    print("=" * 70)
    print("测试4：简单条件 + 复杂条件组合")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 删除配件分类中价格小于200的产品
        deleted_ids = await delete(
            session,
            Product,
            filter_by={"category": "配件"},
            complex_filter=Product.price < 200
        )
        
        print(f"✓ 删除了配件分类中价格<200的产品")
        print(f"✓ 删除了 {len(deleted_ids)} 个产品")
        print(f"✓ 被删除的 ID: {deleted_ids}\n")


async def test_delete_with_or_condition():
    """测试5：使用 OR 条件删除"""
    print("=" * 70)
    print("测试5：使用 OR 条件删除")
    print("=" * 70)
    
    from sqlalchemy import or_
    
    async with AsyncSessionLocal() as session:
        # 删除库存为0或者价格低于10的产品
        deleted_ids = await delete(
            session,
            Product,
            complex_filter=or_(Product.stock == 0, Product.price < 10)
        )
        
        print(f"✓ 删除了库存=0或价格<10的产品")
        print(f"✓ 删除了 {len(deleted_ids)} 个产品")
        print(f"✓ 被删除的 ID: {deleted_ids}\n")


async def test_delete_with_not_in():
    """测试6：使用 NOT IN 条件删除"""
    print("=" * 70)
    print("测试6：使用 NOT IN 条件删除")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 保留 ID 1, 2 的产品，删除其他所有
        keep_ids = [1, 2]
        
        deleted_ids = await delete(
            session,
            Product,
            complex_filter=~Product.id.in_(keep_ids)  # ~ 表示 NOT
        )
        
        print(f"✓ 删除了除 {keep_ids} 外的所有产品")
        print(f"✓ 删除了 {len(deleted_ids)} 个产品")
        print(f"✓ 被删除的 ID: {deleted_ids}\n")


async def test_delete_with_like():
    """测试7：使用 LIKE 条件删除"""
    print("=" * 70)
    print("测试7：使用 LIKE 条件删除")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 删除名称包含"老产品"的记录
        deleted_ids = await delete(
            session,
            Product,
            complex_filter=Product.name.like("%老产品%")
        )
        
        print(f"✓ 删除了名称包含'老产品'的记录")
        print(f"✓ 删除了 {len(deleted_ids)} 个产品")
        print(f"✓ 被删除的 ID: {deleted_ids}\n")


async def test_error_handling():
    """测试8：错误处理（防止误删全表）"""
    print("=" * 70)
    print("测试8：错误处理（防止误删全表）")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        try:
            # 尝试不带任何条件删除（应该抛出错误）
            deleted_ids = await delete(session, Product)
            print("✗ 错误：应该抛出异常")
        except ValueError as e:
            print(f"✓ 正确拦截了无条件删除: {e}\n")


async def show_remaining_products():
    """显示剩余的产品"""
    print("=" * 70)
    print("剩余产品列表")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        products = await fetch_all(session, Product)
        
        if products:
            print(f"剩余 {len(products)} 个产品:")
            for p in products:
                print(f"  ID: {p.id:2d} | {p.name:20s} | {p.category:6s} | ¥{p.price:8.2f} | 库存: {p.stock:3d}")
        else:
            print("没有剩余产品")
        print()


# ============================================================================
# 使用示例（FastAPI）
# ============================================================================

def fastapi_examples():
    """FastAPI 中的使用示例"""
    print("""
    ========================================================================
    FastAPI 中的使用示例
    ========================================================================
    
    # 1. 按 ID 列表删除
    @app.delete("/products/bulk-delete")
    async def delete_products_by_ids(
        ids: List[int],
        db: AsyncSession = Depends(get_db)
    ):
        deleted_ids = await delete(
            db, 
            Product, 
            complex_filter=Product.id.in_(ids)
        )
        return {"deleted": len(deleted_ids), "ids": deleted_ids}
    
    
    # 2. 按分类删除
    @app.delete("/products/by-category/{category}")
    async def delete_by_category(
        category: str,
        db: AsyncSession = Depends(get_db)
    ):
        deleted_ids = await delete(
            db,
            Product,
            filter_by={"category": category}
        )
        return {"deleted": len(deleted_ids), "ids": deleted_ids}
    
    
    # 3. 删除清仓产品（复杂条件）
    @app.delete("/products/clear-stock")
    async def clear_old_stock(db: AsyncSession = Depends(get_db)):
        # 删除库存为0且价格低于100的产品
        deleted_ids = await delete(
            db,
            Product,
            complex_filter=(Product.stock == 0) & (Product.price < 100)
        )
        return {"deleted": len(deleted_ids), "ids": deleted_ids}
    
    
    # 4. 批量删除（按条件）
    @app.delete("/products/batch")
    async def batch_delete(
        min_price: float = None,
        max_price: float = None,
        category: str = None,
        db: AsyncSession = Depends(get_db)
    ):
        from sqlalchemy import and_
        
        conditions = []
        if min_price:
            conditions.append(Product.price >= min_price)
        if max_price:
            conditions.append(Product.price <= max_price)
        if category:
            conditions.append(Product.category == category)
        
        if conditions:
            deleted_ids = await delete(
                db,
                Product,
                complex_filter=and_(*conditions)
            )
            return {"deleted": len(deleted_ids), "ids": deleted_ids}
        else:
            raise HTTPException(status_code=400, detail="至少提供一个条件")
    
    ========================================================================
    """)


# ============================================================================
# 主函数
# ============================================================================

async def main():
    print("""
    ========================================================================
    delete 函数测试 - 支持 IN 和复杂条件
    ========================================================================
    """)
    
    # 初始化
    await init_db()
    await seed_data()
    
    # 显示初始数据
    await show_remaining_products()
    
    # 测试1：简单条件删除
    await test_delete_by_simple_filter()
    await show_remaining_products()
    
    # 重新插入数据
    await seed_data()
    
    # 测试2：IN 条件删除
    await test_delete_by_in_condition()
    await show_remaining_products()
    
    # 重新插入数据
    await seed_data()
    
    # 测试3：复杂条件删除
    await test_delete_by_complex_filter()
    await show_remaining_products()
    
    # 重新插入数据
    await seed_data()
    
    # 测试4：组合条件删除
    await test_delete_by_combined_conditions()
    await show_remaining_products()
    
    # 重新插入数据
    await seed_data()
    
    # 测试5：OR 条件
    await test_delete_with_or_condition()
    await show_remaining_products()
    
    # 重新插入数据
    await seed_data()
    
    # 测试7：LIKE 条件
    await test_delete_with_like()
    await show_remaining_products()
    
    # 测试8：错误处理
    await test_error_handling()
    
    # FastAPI 示例
    fastapi_examples()
    
    print("=" * 70)
    print("✓ 所有测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
