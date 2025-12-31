"""
FastAPI 批量创建完整示例
使用上面的 bulk_create 工具函数
"""

from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from sqlalchemy import String, Integer, Float, DateTime
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, AsyncAttrs

# 导入批量创建工具函数
from bulk_create_utils import (
    bulk_create,
    bulk_create_with_ids,
    bulk_create_detailed,
    bulk_create_batched,
    bulk_create_tolerant,
)

# ============================================================================
# 数据库配置
# ============================================================================

DATABASE_URL = "sqlite+aiosqlite:///./test_bulk.db"

async_engine = create_async_engine(
    DATABASE_URL,
    echo=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# ============================================================================
# 模型定义
# ============================================================================

class Base(AsyncAttrs, DeclarativeBase):
    pass


class Product(Base):
    """产品模型"""
    __tablename__ = "products"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    stock: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# 初始化数据库
async def init_db():
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


class ProductResponse(BaseModel):
    """产品响应模型"""
    id: int
    name: str
    category: str
    price: float
    stock: int
    created_at: datetime
    
    model_config = {"from_attributes": True}


class BulkCreateResponse(BaseModel):
    """批量创建响应模型"""
    success: bool
    created_count: int
    message: str
    skipped_count: Optional[int] = None
    failed_count: Optional[int] = None
    created_ids: Optional[List[int]] = None


# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(title="批量创建工具函数示例")


@app.on_event("startup")
async def startup():
    await init_db()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ============================================================================
# 批量创建接口
# ============================================================================

@app.post("/products/bulk-create/simple", response_model=BulkCreateResponse)
async def bulk_create_simple(
    products: List[ProductCreate],
    db: AsyncSession = Depends(get_db)
):
    """
    方案1：简单批量创建（推荐）
    
    - 性能最佳
    - 返回创建数量
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    try:
        # 将 Pydantic 模型转换为字典列表
        data_list = [p.model_dump() for p in products]
        
        # 调用批量创建工具函数
        count = await bulk_create(db, Product, data_list)
        
        return BulkCreateResponse(
            success=True,
            created_count=count,
            message=f"成功创建 {count} 个产品"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


@app.post("/products/bulk-create/with-ids", response_model=BulkCreateResponse)
async def bulk_create_ids(
    products: List[ProductCreate],
    db: AsyncSession = Depends(get_db)
):
    """
    方案2：批量创建并返回 ID 列表
    
    - 可以获取插入的 ID
    - 适合需要后续操作的场景
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    try:
        data_list = [p.model_dump() for p in products]
        
        # 调用返回 ID 的批量创建函数
        ids = await bulk_create_with_ids(db, Product, data_list)
        
        return BulkCreateResponse(
            success=True,
            created_count=len(ids),
            message=f"成功创建 {len(ids)} 个产品",
            created_ids=ids
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


@app.post("/products/bulk-create/with-dedup", response_model=BulkCreateResponse)
async def bulk_create_dedup(
    products: List[ProductCreate],
    db: AsyncSession = Depends(get_db)
):
    """
    方案3：批量创建（带去重）
    
    - 根据 name 字段去重
    - 跳过已存在的记录
    - 返回详细结果
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    try:
        data_list = [p.model_dump() for p in products]
        
        # 调用带去重的详细批量创建函数
        result = await bulk_create_detailed(
            db, 
            Product, 
            data_list,
            filter_by_key="name"  # 根据 name 字段去重
        )
        
        return BulkCreateResponse(
            success=result["success"],
            created_count=result["created_count"],
            skipped_count=result["skipped_count"],
            message=f"创建 {result['created_count']} 个，跳过 {result['skipped_count']} 个",
            created_ids=result["created_ids"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


@app.post("/products/bulk-create/batched", response_model=BulkCreateResponse)
async def bulk_create_batch(
    products: List[ProductCreate],
    batch_size: int = 1000,
    db: AsyncSession = Depends(get_db)
):
    """
    方案4：分批批量创建
    
    - 适合大数据量
    - 避免单次插入过多
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    try:
        data_list = [p.model_dump() for p in products]
        
        # 调用分批插入函数
        count = await bulk_create_batched(
            db, 
            Product, 
            data_list,
            batch_size=batch_size
        )
        
        return BulkCreateResponse(
            success=True,
            created_count=count,
            message=f"成功分批创建 {count} 个产品（批次大小: {batch_size}）"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


@app.post("/products/bulk-create/tolerant", response_model=BulkCreateResponse)
async def bulk_create_fault_tolerant(
    products: List[ProductCreate],
    db: AsyncSession = Depends(get_db)
):
    """
    方案5：容错批量创建
    
    - 部分失败不影响其他记录
    - 返回成功和失败的详细信息
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    try:
        data_list = [p.model_dump() for p in products]
        
        # 调用容错批量创建函数
        result = await bulk_create_tolerant(
            db, 
            Product, 
            data_list,
            filter_by_key="name"
        )
        
        return BulkCreateResponse(
            success=result["success"],
            created_count=result["created_count"],
            failed_count=result["failed_count"],
            message=f"成功 {result['created_count']} 个，失败 {result['failed_count']} 个",
            created_ids=result["created_ids"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 查询接口
# ============================================================================

@app.get("/products", response_model=List[ProductResponse])
async def get_products(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """查询产品列表"""
    from sqlalchemy import select
    
    stmt = select(Product).offset(skip).limit(limit)
    result = await db.execute(stmt)
    products = result.scalars().all()
    
    return products


@app.get("/products/count")
async def get_product_count(db: AsyncSession = Depends(get_db)):
    """获取产品总数"""
    from sqlalchemy import func, select
    
    stmt = select(func.count()).select_from(Product)
    result = await db.execute(stmt)
    count = result.scalar()
    
    return {"total": count}


@app.delete("/products/clear")
async def clear_products(db: AsyncSession = Depends(get_db)):
    """清空产品表"""
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
        "message": "批量创建工具函数示例 API",
        "docs": "/docs",
        "endpoints": {
            "simple": "POST /products/bulk-create/simple - 简单批量创建（推荐）",
            "with_ids": "POST /products/bulk-create/with-ids - 返回ID列表",
            "with_dedup": "POST /products/bulk-create/with-dedup - 带去重",
            "batched": "POST /products/bulk-create/batched - 分批插入",
            "tolerant": "POST /products/bulk-create/tolerant - 容错插入",
        }
    }


# ============================================================================
# 运行说明
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    =============================================================
    批量创建工具函数示例
    =============================================================
    
    已提供5种批量创建方案：
    
    1. simple      - 简单批量创建（推荐）⭐
    2. with-ids    - 返回ID列表
    3. with-dedup  - 带去重检查
    4. batched     - 分批插入（大数据量）
    5. tolerant    - 容错插入（部分失败不影响其他）
    
    访问: http://localhost:8000/docs
    
    =============================================================
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
