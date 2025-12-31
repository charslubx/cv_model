"""
FastAPI 批量创建 - SQLAlchemy 2.0 版本

支持 SQLAlchemy 2.0 的新特性：
- 新的声明式基类
- 类型提示
- 新的查询 API
- 同步和异步支持
"""

from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

# SQLAlchemy 2.0 导入
from sqlalchemy import create_engine, String, Integer, Float, DateTime, select, insert
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker

# ============================================================================
# SQLAlchemy 2.0 数据库配置
# ============================================================================

DATABASE_URL = "sqlite:///./test_sqlalchemy2.db"
# MySQL: "mysql+pymysql://user:password@localhost:3306/database"
# PostgreSQL: "postgresql://user:password@localhost:5432/database"

engine = create_engine(
    DATABASE_URL,
    echo=True,  # 打印 SQL 日志
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False  # SQLAlchemy 2.0 推荐设置
)


# ============================================================================
# SQLAlchemy 2.0 声明式基类（新语法）
# ============================================================================

class Base(DeclarativeBase):
    """SQLAlchemy 2.0 声明式基类"""
    pass


# ============================================================================
# SQLAlchemy 2.0 模型定义（使用 Mapped 和 mapped_column）
# ============================================================================

class Product(Base):
    """产品模型 - SQLAlchemy 2.0 风格"""
    __tablename__ = "products"
    
    # 使用 Mapped 和 mapped_column 进行类型提示
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    stock: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"Product(id={self.id}, name={self.name}, price={self.price})"


class User(Base):
    """用户模型 - SQLAlchemy 2.0 风格"""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    is_active: Mapped[bool] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# 创建所有表
Base.metadata.create_all(bind=engine)


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
    
    model_config = {"from_attributes": True}  # Pydantic v2 语法


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
    title="FastAPI 批量创建 - SQLAlchemy 2.0",
    version="2.0.0"
)


# ============================================================================
# 依赖项
# ============================================================================

def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# 方案1：SQLAlchemy 2.0 推荐 - 使用 insert() 语句（最佳性能）⭐
# ============================================================================

@app.post("/products/bulk-create/v2-insert", response_model=BulkCreateResponse)
async def bulk_create_v2_insert(
    products: List[ProductCreate],
    db: Session = Depends(get_db)
):
    """
    SQLAlchemy 2.0 推荐方案 - 使用 insert() 语句
    
    这是 SQLAlchemy 2.0 的推荐方式，性能最佳
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    import time
    start_time = time.time()
    
    try:
        # SQLAlchemy 2.0 风格：使用 insert() 语句
        stmt = insert(Product).values(
            [product.model_dump() for product in products]
        )
        
        db.execute(stmt)
        db.commit()
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=len(products),
            message=f"成功批量创建 {len(products)} 个产品（SQLAlchemy 2.0 insert）",
            execution_time=round(execution_time, 4)
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 方案2：使用 bulk_insert_mappings（兼容方式，性能好）
# ============================================================================

@app.post("/products/bulk-create/mappings", response_model=BulkCreateResponse)
async def bulk_create_mappings(
    products: List[ProductCreate],
    db: Session = Depends(get_db)
):
    """
    使用 bulk_insert_mappings（SQLAlchemy 2.0 仍然支持）
    
    这是兼容方式，性能也很好
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    import time
    start_time = time.time()
    
    try:
        # 转换为字典列表
        product_dicts = [product.model_dump() for product in products]
        
        # bulk_insert_mappings 在 SQLAlchemy 2.0 中仍然可用
        db.bulk_insert_mappings(Product, product_dicts)
        db.commit()
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=len(products),
            message=f"成功批量创建 {len(products)} 个产品（bulk_insert_mappings）",
            execution_time=round(execution_time, 4)
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 方案3：分批插入（处理大数据量）
# ============================================================================

@app.post("/products/bulk-create/batched", response_model=BulkCreateResponse)
async def bulk_create_batched(
    products: List[ProductCreate],
    batch_size: int = 1000,
    db: Session = Depends(get_db)
):
    """
    分批插入 - SQLAlchemy 2.0 风格
    
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
            
            # 使用 SQLAlchemy 2.0 的 insert() 语句
            stmt = insert(Product).values(
                [p.model_dump() for p in batch]
            )
            
            db.execute(stmt)
            db.commit()
            total_created += len(batch)
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=total_created,
            message=f"成功分批创建 {total_created} 个产品",
            execution_time=round(execution_time, 4)
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 方案4：使用 add_all（需要返回对象时）
# ============================================================================

@app.post("/products/bulk-create/add-all", response_model=BulkCreateResponse)
async def bulk_create_add_all(
    products: List[ProductCreate],
    db: Session = Depends(get_db)
):
    """
    使用 add_all - 可以获取插入后的对象
    
    适合需要返回 ID 或处理关系的场景
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
        
        # SQLAlchemy 2.0 推荐使用 add_all
        db.add_all(db_products)
        db.commit()
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=len(products),
            message=f"成功批量创建 {len(products)} 个产品（add_all）",
            execution_time=round(execution_time, 4)
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 查询接口 - SQLAlchemy 2.0 新语法
# ============================================================================

@app.get("/products", response_model=List[ProductResponse])
async def get_products(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    查询产品列表 - SQLAlchemy 2.0 风格
    
    使用 select() 而不是 query()
    """
    # SQLAlchemy 2.0 推荐使用 select()
    stmt = select(Product).offset(skip).limit(limit)
    products = db.execute(stmt).scalars().all()
    
    return products


@app.get("/products/count")
async def get_product_count(db: Session = Depends(get_db)):
    """获取产品总数 - SQLAlchemy 2.0 风格"""
    from sqlalchemy import func
    
    stmt = select(func.count()).select_from(Product)
    count = db.execute(stmt).scalar()
    
    return {"total": count}


@app.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(product_id: int, db: Session = Depends(get_db)):
    """获取单个产品 - SQLAlchemy 2.0 风格"""
    stmt = select(Product).where(Product.id == product_id)
    product = db.execute(stmt).scalar_one_or_none()
    
    if not product:
        raise HTTPException(status_code=404, detail="产品不存在")
    
    return product


@app.delete("/products/clear")
async def clear_products(db: Session = Depends(get_db)):
    """清空产品表 - SQLAlchemy 2.0 风格"""
    from sqlalchemy import delete
    
    try:
        stmt = delete(Product)
        result = db.execute(stmt)
        db.commit()
        
        return {"message": f"已删除 {result.rowcount} 个产品"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"清空失败: {str(e)}")


# ============================================================================
# 高级用法：使用 returning() 获取插入的数据（PostgreSQL）
# ============================================================================

@app.post("/products/bulk-create/returning")
async def bulk_create_with_returning(
    products: List[ProductCreate],
    db: Session = Depends(get_db)
):
    """
    使用 returning() 获取插入的数据
    
    注意：仅在 PostgreSQL 中可用
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    try:
        # SQLAlchemy 2.0 的 returning() 用法
        stmt = (
            insert(Product)
            .values([p.model_dump() for p in products])
            .returning(Product.id, Product.name)  # 返回插入的 id 和 name
        )
        
        result = db.execute(stmt)
        inserted_data = result.all()
        db.commit()
        
        return {
            "success": True,
            "created_count": len(products),
            "inserted_data": [
                {"id": row.id, "name": row.name}
                for row in inserted_data
            ]
        }
    
    except Exception as e:
        db.rollback()
        # 如果不支持 returning，降级到普通插入
        if "returning" in str(e).lower():
            raise HTTPException(
                status_code=400,
                detail="returning() 仅在 PostgreSQL 中支持"
            )
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "FastAPI 批量创建 - SQLAlchemy 2.0",
        "sqlalchemy_version": "2.0+",
        "docs": "/docs",
        "endpoints": {
            "v2_insert": "POST /products/bulk-create/v2-insert (推荐)",
            "mappings": "POST /products/bulk-create/mappings (兼容)",
            "batched": "POST /products/bulk-create/batched (大数据量)",
            "add_all": "POST /products/bulk-create/add-all (需要对象)",
            "returning": "POST /products/bulk-create/returning (PostgreSQL)",
        }
    }


# ============================================================================
# 运行说明
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    =============================================================
    FastAPI 批量创建 - SQLAlchemy 2.0 版本
    =============================================================
    
    SQLAlchemy 2.0 新特性：
    ✓ 使用 DeclarativeBase 替代 declarative_base()
    ✓ 使用 Mapped 和 mapped_column 进行类型提示
    ✓ 使用 select() 替代 query()
    ✓ 使用 insert() 进行批量插入
    ✓ 支持 returning() (PostgreSQL)
    
    访问以下地址：
    - API 文档: http://localhost:8000/docs
    
    推荐方案：
    1. 最佳性能: /products/bulk-create/v2-insert ⭐
    2. 兼容性好: /products/bulk-create/mappings
    3. 大数据量: /products/bulk-create/batched
    4. 需要对象: /products/bulk-create/add-all
    
    =============================================================
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
