"""
FastAPI 批量创建 - SQLAlchemy 2.0 异步版本

使用 SQLAlchemy 2.0 的异步特性，获得更好的性能
"""

from typing import List, Optional
from datetime import datetime
import asyncio

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

# SQLAlchemy 2.0 异步导入
from sqlalchemy import String, Integer, Float, DateTime, select, insert
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncAttrs
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ============================================================================
# 异步数据库配置
# ============================================================================

# SQLite 异步
DATABASE_URL = "sqlite+aiosqlite:///./test_async.db"

# MySQL 异步
# DATABASE_URL = "mysql+aiomysql://user:password@localhost:3306/database"

# PostgreSQL 异步
# DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/database"

async_engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    pool_size=20,
    max_overflow=40,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ============================================================================
# SQLAlchemy 2.0 异步声明式基类
# ============================================================================

class Base(AsyncAttrs, DeclarativeBase):
    """SQLAlchemy 2.0 异步声明式基类"""
    pass


# ============================================================================
# 模型定义
# ============================================================================

class Product(Base):
    """产品模型 - 异步版本"""
    __tablename__ = "products"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    stock: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# 初始化数据库
async def init_db():
    """异步初始化数据库"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ============================================================================
# Pydantic 模型
# ============================================================================

class ProductCreate(BaseModel):
    """创建产品的请求模型"""
    name: str = Field(..., min_length=1, max_length=200)
    category: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    stock: int = Field(default=0, ge=0)
    description: Optional[str] = Field(None, max_length=500)


class ProductResponse(BaseModel):
    """产品响应模型"""
    id: int
    name: str
    category: str
    price: float
    stock: int
    description: Optional[str]
    created_at: datetime
    
    model_config = {"from_attributes": True}


class BulkCreateResponse(BaseModel):
    """批量创建响应模型"""
    success: bool
    created_count: int
    message: str
    execution_time: Optional[float] = None


# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(
    title="FastAPI 批量创建 - SQLAlchemy 2.0 异步版本",
    version="2.0.0-async"
)


# ============================================================================
# 依赖项
# ============================================================================

async def get_db():
    """获取异步数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ============================================================================
# 启动事件
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库"""
    await init_db()
    print("✓ 异步数据库已初始化")


# ============================================================================
# 异步批量创建 - 方案1：使用 insert()（推荐）⭐
# ============================================================================

@app.post("/products/bulk-create/async-insert", response_model=BulkCreateResponse)
async def bulk_create_async_insert(
    products: List[ProductCreate],
    db: AsyncSession = Depends(get_db)
):
    """
    异步批量创建 - 使用 insert()
    
    这是 SQLAlchemy 2.0 异步的推荐方式
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    import time
    start_time = time.time()
    
    try:
        # 异步执行批量插入
        stmt = insert(Product).values(
            [product.model_dump() for product in products]
        )
        
        await db.execute(stmt)
        await db.commit()
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=len(products),
            message=f"成功异步批量创建 {len(products)} 个产品",
            execution_time=round(execution_time, 4)
        )
    
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 异步批量创建 - 方案2：分批插入
# ============================================================================

@app.post("/products/bulk-create/async-batched", response_model=BulkCreateResponse)
async def bulk_create_async_batched(
    products: List[ProductCreate],
    batch_size: int = 1000,
    db: AsyncSession = Depends(get_db)
):
    """
    异步分批插入
    
    适合大数据量场景
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    import time
    start_time = time.time()
    
    try:
        total_created = 0
        
        # 分批处理
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            
            stmt = insert(Product).values(
                [p.model_dump() for p in batch]
            )
            
            await db.execute(stmt)
            await db.commit()
            total_created += len(batch)
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=total_created,
            message=f"成功异步分批创建 {total_created} 个产品",
            execution_time=round(execution_time, 4)
        )
    
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 异步批量创建 - 方案3：使用 add_all
# ============================================================================

@app.post("/products/bulk-create/async-add-all", response_model=BulkCreateResponse)
async def bulk_create_async_add_all(
    products: List[ProductCreate],
    db: AsyncSession = Depends(get_db)
):
    """
    异步批量创建 - 使用 add_all
    
    可以获取插入后的对象
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    import time
    start_time = time.time()
    
    try:
        # 创建对象列表
        db_products = [
            Product(**product.model_dump())
            for product in products
        ]
        
        # 异步添加所有对象
        db.add_all(db_products)
        await db.commit()
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=len(products),
            message=f"成功异步批量创建 {len(products)} 个产品（add_all）",
            execution_time=round(execution_time, 4)
        )
    
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 异步查询接口
# ============================================================================

@app.get("/products", response_model=List[ProductResponse])
async def get_products(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """异步查询产品列表"""
    stmt = select(Product).offset(skip).limit(limit)
    result = await db.execute(stmt)
    products = result.scalars().all()
    
    return products


@app.get("/products/count")
async def get_product_count(db: AsyncSession = Depends(get_db)):
    """异步获取产品总数"""
    from sqlalchemy import func
    
    stmt = select(func.count()).select_from(Product)
    result = await db.execute(stmt)
    count = result.scalar()
    
    return {"total": count}


@app.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(product_id: int, db: AsyncSession = Depends(get_db)):
    """异步获取单个产品"""
    stmt = select(Product).where(Product.id == product_id)
    result = await db.execute(stmt)
    product = result.scalar_one_or_none()
    
    if not product:
        raise HTTPException(status_code=404, detail="产品不存在")
    
    return product


@app.delete("/products/clear")
async def clear_products(db: AsyncSession = Depends(get_db)):
    """异步清空产品表"""
    from sqlalchemy import delete
    
    try:
        stmt = delete(Product)
        result = await db.execute(stmt)
        await db.commit()
        
        return {"message": f"已删除 {result.rowcount} 个产品"}
    
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"清空失败: {str(e)}")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "FastAPI 批量创建 - SQLAlchemy 2.0 异步版本",
        "sqlalchemy_version": "2.0+ (Async)",
        "docs": "/docs",
        "endpoints": {
            "async_insert": "POST /products/bulk-create/async-insert (推荐)",
            "async_batched": "POST /products/bulk-create/async-batched (大数据量)",
            "async_add_all": "POST /products/bulk-create/async-add-all (需要对象)",
        }
    }


# ============================================================================
# 运行说明
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    =============================================================
    FastAPI 批量创建 - SQLAlchemy 2.0 异步版本
    =============================================================
    
    异步特性：
    ✓ 使用 AsyncSession
    ✓ 非阻塞 I/O
    ✓ 更好的并发性能
    ✓ 支持异步数据库驱动
    
    支持的数据库：
    - SQLite (aiosqlite)
    - MySQL (aiomysql)
    - PostgreSQL (asyncpg) - 推荐
    
    访问以下地址：
    - API 文档: http://localhost:8000/docs
    
    =============================================================
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
