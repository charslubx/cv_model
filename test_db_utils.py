"""
测试通用数据库工具函数
"""

import asyncio
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, DateTime
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, AsyncAttrs

# 导入工具函数
from db_utils import (
    create,
    bulk_create,
    bulk_create_return_ids,
    bulk_create_detailed,
    bulk_update,
    bulk_delete,
    fetch_one,
    fetch_all,
    count_records
)


# ============================================================================
# 测试数据库配置
# ============================================================================

DATABASE_URL = "sqlite+aiosqlite:///./test_utils.db"

async_engine = create_async_engine(DATABASE_URL, echo=True)
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
        await conn.run_sync(Base.metadata.drop_all)  # 清空
        await conn.run_sync(Base.metadata.create_all)  # 重建
    print("✓ 数据库已初始化\n")


# ============================================================================
# 测试函数
# ============================================================================

async def test_single_create():
    """测试单条创建"""
    print("=" * 70)
    print("测试1：单条创建")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 创建第一条
        product_id = await create(
            session,
            Product,
            {"name": "iPhone 15", "category": "电子", "price": 999.99, "stock": 100}
        )
        print(f"✓ 创建成功，ID: {product_id}\n")
        
        # 尝试创建重复（应该返回0）
        product_id = await create(
            session,
            Product,
            {"name": "iPhone 15", "category": "电子", "price": 899.99, "stock": 50},
            filter_by={"name": "iPhone 15"}
        )
        print(f"✓ 重复检测，返回: {product_id} (0表示已存在)\n")


async def test_bulk_create_simple():
    """测试简单批量创建"""
    print("=" * 70)
    print("测试2：简单批量创建")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        products = [
            {"name": "MacBook Pro", "category": "电子", "price": 2499.99, "stock": 50},
            {"name": "iPad Air", "category": "电子", "price": 599.99, "stock": 200},
            {"name": "AirPods Pro", "category": "电子", "price": 249.99, "stock": 500}
        ]
        
        count = await bulk_create(session, Product, products)
        print(f"✓ 批量创建成功，数量: {count}\n")


async def test_bulk_create_with_dedup():
    """测试带去重的批量创建"""
    print("=" * 70)
    print("测试3：带去重的批量创建")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        products = [
            {"name": "Magic Mouse", "category": "配件", "price": 99.99, "stock": 300},
            {"name": "Magic Keyboard", "category": "配件", "price": 149.99, "stock": 200},
            {"name": "iPhone 15", "category": "电子", "price": 999.99, "stock": 100},  # 重复
        ]
        
        count = await bulk_create(session, Product, products, filter_by="name")
        print(f"✓ 批量创建成功（去重后），数量: {count}\n")


async def test_bulk_create_return_ids():
    """测试返回ID的批量创建"""
    print("=" * 70)
    print("测试4：批量创建并返回ID")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        products = [
            {"name": "Apple Watch", "category": "配件", "price": 399.99, "stock": 100},
            {"name": "HomePod", "category": "音响", "price": 299.99, "stock": 150}
        ]
        
        ids = await bulk_create_return_ids(session, Product, products)
        print(f"✓ 批量创建成功，ID列表: {ids}\n")


async def test_bulk_create_detailed():
    """测试详细结果的批量创建"""
    print("=" * 70)
    print("测试5：批量创建（详细结果）")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        products = [
            {"name": "Mac Mini", "category": "电脑", "price": 699.99, "stock": 80},
            {"name": "Mac Studio", "category": "电脑", "price": 1999.99, "stock": 30},
            {"name": "iPhone 15", "category": "电子", "price": 999.99, "stock": 100},  # 重复
            {"name": "iPad Air", "category": "电子", "price": 599.99, "stock": 200},  # 重复
        ]
        
        result = await bulk_create_detailed(session, Product, products, filter_by="name")
        print(f"✓ 创建: {result['created']}")
        print(f"✓ 跳过: {result['skipped']}")
        print(f"✓ 总数: {result['total']}")
        print(f"✓ ID列表: {result['ids']}\n")


async def test_bulk_update():
    """测试批量更新"""
    print("=" * 70)
    print("测试6：批量更新")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 先查询一些产品ID
        products = await fetch_all(session, Product, limit=3)
        
        if products:
            updates = [
                {"id": products[0].id, "price": 888.88, "stock": 888},
                {"id": products[1].id, "price": 666.66, "stock": 666},
            ]
            
            count = await bulk_update(session, Product, updates)
            print(f"✓ 批量更新成功，数量: {count}\n")


async def test_bulk_delete():
    """测试批量删除"""
    print("=" * 70)
    print("测试7：批量删除")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 按条件删除
        count = await bulk_delete(session, Product, filter_by={"category": "配件"})
        print(f"✓ 批量删除成功（按条件），数量: {count}\n")


async def test_fetch_all():
    """测试查询所有"""
    print("=" * 70)
    print("测试8：查询记录")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 查询所有电子产品，按价格降序
        products = await fetch_all(
            session,
            Product,
            filter_by={"category": "电子"},
            order_by="-price",
            limit=5
        )
        
        print(f"✓ 查询到 {len(products)} 个电子产品:")
        for p in products:
            print(f"  - {p.name}: ¥{p.price} (库存: {p.stock})")
        print()


async def test_count():
    """测试统计"""
    print("=" * 70)
    print("测试9：统计记录")
    print("=" * 70)
    
    async with AsyncSessionLocal() as session:
        # 统计所有产品
        total = await count_records(session, Product)
        print(f"✓ 总产品数: {total}")
        
        # 统计电子产品
        electronics = await count_records(session, Product, filter_by={"category": "电子"})
        print(f"✓ 电子产品数: {electronics}\n")


async def test_performance():
    """测试性能对比"""
    print("=" * 70)
    print("测试10：性能对比（1000条数据）")
    print("=" * 70)
    
    import time
    
    # 生成测试数据
    test_data = [
        {
            "name": f"测试产品{i}",
            "category": "测试",
            "price": i * 10.0,
            "stock": i
        }
        for i in range(1, 1001)
    ]
    
    async with AsyncSessionLocal() as session:
        # 批量创建
        start = time.time()
        count = await bulk_create(session, Product, test_data)
        elapsed = time.time() - start
        
        print(f"✓ 批量创建 {count} 条数据")
        print(f"✓ 耗时: {elapsed:.4f} 秒")
        print(f"✓ 吞吐量: {count/elapsed:.0f} 条/秒\n")


# ============================================================================
# 运行所有测试
# ============================================================================

async def main():
    print("""
    ============================================================
    通用数据库工具函数测试
    ============================================================
    """)
    
    # 初始化数据库
    await init_db()
    
    # 运行测试
    await test_single_create()
    await test_bulk_create_simple()
    await test_bulk_create_with_dedup()
    await test_bulk_create_return_ids()
    await test_bulk_create_detailed()
    await test_bulk_update()
    await test_fetch_all()
    await test_count()
    await test_bulk_delete()
    await test_performance()
    
    print("=" * 70)
    print("✓ 所有测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
