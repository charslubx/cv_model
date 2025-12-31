"""
FastAPI 批量创建（bulk_create）实现示例

这个文件展示了如何在 FastAPI 中实现高效的批量数据创建功能
支持多种方案：SQLAlchemy ORM 和原生 SQL
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional
from datetime import datetime
import time

# ============================================================================
# 数据库配置
# ============================================================================

# 数据库连接配置
DATABASE_URL = "mysql+pymysql://user:password@localhost:3306/database_name"
# 如果使用 SQLite 测试：DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False  # 设置为 True 可以看到 SQL 日志
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ============================================================================
# 数据库模型
# ============================================================================

class Product(Base):
    """产品数据库模型"""
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(200), nullable=False, index=True)
    category = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)
    description = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    """用户数据库模型"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(200), unique=True, nullable=False, index=True)
    full_name = Column(String(200))
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


# 创建所有表
Base.metadata.create_all(bind=engine)


# ============================================================================
# Pydantic 模型（用于请求和响应）
# ============================================================================

class ProductCreate(BaseModel):
    """创建产品的请求模型"""
    name: str = Field(..., min_length=1, max_length=200)
    category: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    stock: int = Field(default=0, ge=0)
    description: Optional[str] = Field(None, max_length=500)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "iPhone 15 Pro",
                "category": "Electronics",
                "price": 999.99,
                "stock": 100,
                "description": "Latest iPhone model"
            }
        }


class ProductResponse(BaseModel):
    """产品响应模型"""
    id: int
    name: str
    category: str
    price: float
    stock: int
    description: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """创建用户的请求模型"""
    username: str = Field(..., min_length=3, max_length=100)
    email: str = Field(..., min_length=5, max_length=200)
    full_name: Optional[str] = Field(None, max_length=200)
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "john@example.com",
                "full_name": "John Doe"
            }
        }


class UserResponse(BaseModel):
    """用户响应模型"""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class BulkCreateResponse(BaseModel):
    """批量创建响应模型"""
    success: bool
    created_count: int
    failed_count: int
    execution_time: float
    message: str
    created_ids: Optional[List[int]] = None


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
# FastAPI 应用
# ============================================================================

app = FastAPI(
    title="FastAPI 批量创建示例",
    description="演示如何在 FastAPI 中实现高效的批量数据创建",
    version="1.0.0"
)


# ============================================================================
# 方案1：使用 SQLAlchemy bulk_save_objects（推荐）
# ============================================================================

@app.post("/products/bulk-create/method1", response_model=BulkCreateResponse)
async def bulk_create_products_method1(
    products: List[ProductCreate],
    db: Session = Depends(get_db)
):
    """
    方案1：使用 SQLAlchemy 的 bulk_save_objects 批量创建产品
    
    优点：
    - 代码简洁，易于维护
    - 自动处理对象映射
    - 性能好，减少数据库往返次数
    
    缺点：
    - 默认不返回插入的 ID（需要设置 return_defaults=True）
    - return_defaults=True 会降低性能
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    start_time = time.time()
    
    try:
        # 将 Pydantic 模型转换为 SQLAlchemy 模型
        db_products = [
            Product(**product.model_dump())
            for product in products
        ]
        
        # 批量保存对象
        db.bulk_save_objects(db_products)
        db.commit()
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=len(products),
            failed_count=0,
            execution_time=round(execution_time, 4),
            message=f"成功批量创建 {len(products)} 个产品"
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 方案2：使用 SQLAlchemy bulk_insert_mappings（最快）
# ============================================================================

@app.post("/products/bulk-create/method2", response_model=BulkCreateResponse)
async def bulk_create_products_method2(
    products: List[ProductCreate],
    db: Session = Depends(get_db)
):
    """
    方案2：使用 SQLAlchemy 的 bulk_insert_mappings 批量创建产品
    
    优点：
    - 性能最佳，直接使用字典映射
    - 减少对象实例化开销
    - 适合大批量数据插入
    
    缺点：
    - 不返回插入的 ID
    - 跳过 ORM 层的一些验证
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    start_time = time.time()
    
    try:
        # 将 Pydantic 模型转换为字典列表
        product_dicts = [product.model_dump() for product in products]
        
        # 批量插入映射
        db.bulk_insert_mappings(Product, product_dicts)
        db.commit()
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=len(products),
            failed_count=0,
            execution_time=round(execution_time, 4),
            message=f"成功批量创建 {len(products)} 个产品（方案2）"
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 方案3：逐个添加后批量提交（返回 ID）
# ============================================================================

@app.post("/products/bulk-create/method3", response_model=BulkCreateResponse)
async def bulk_create_products_method3(
    products: List[ProductCreate],
    db: Session = Depends(get_db)
):
    """
    方案3：逐个添加到会话，最后批量提交，可以获取插入的 ID
    
    优点：
    - 可以获取插入后的 ID
    - 保留 ORM 层的所有功能（验证、关系等）
    - 可以返回完整的对象
    
    缺点：
    - 性能相对较低（但比逐个提交快很多）
    - 内存占用较高
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    start_time = time.time()
    created_ids = []
    
    try:
        # 逐个创建对象并添加到会话
        for product in products:
            db_product = Product(**product.model_dump())
            db.add(db_product)
        
        # 统一提交
        db.commit()
        
        # 刷新以获取 ID（如果需要）
        # 注意：这会增加额外的数据库查询
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=len(products),
            failed_count=0,
            execution_time=round(execution_time, 4),
            message=f"成功批量创建 {len(products)} 个产品（方案3）",
            created_ids=created_ids if created_ids else None
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 方案4：使用原生 SQL（最灵活）
# ============================================================================

@app.post("/products/bulk-create/method4", response_model=BulkCreateResponse)
async def bulk_create_products_method4(
    products: List[ProductCreate],
    db: Session = Depends(get_db)
):
    """
    方案4：使用原生 SQL 批量插入
    
    优点：
    - 完全控制 SQL 语句
    - 可以使用数据库特定的优化（如 MySQL 的 INSERT IGNORE）
    - 性能可以达到最优
    
    缺点：
    - 代码可读性降低
    - 失去 ORM 的便利性
    - 需要手动处理 SQL 注入风险
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    start_time = time.time()
    
    try:
        # 构建批量插入的 SQL
        from sqlalchemy import text
        
        sql = text("""
            INSERT INTO products (name, category, price, stock, description, created_at)
            VALUES (:name, :category, :price, :stock, :description, :created_at)
        """)
        
        # 准备数据
        values = [
            {
                **product.model_dump(),
                'created_at': datetime.utcnow()
            }
            for product in products
        ]
        
        # 批量执行
        db.execute(sql, values)
        db.commit()
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=len(products),
            failed_count=0,
            execution_time=round(execution_time, 4),
            message=f"成功批量创建 {len(products)} 个产品（方案4 - 原生SQL）"
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 方案5：分批插入（处理大量数据）
# ============================================================================

@app.post("/products/bulk-create/method5", response_model=BulkCreateResponse)
async def bulk_create_products_method5(
    products: List[ProductCreate],
    batch_size: int = 1000,
    db: Session = Depends(get_db)
):
    """
    方案5：分批插入，适合处理大量数据
    
    优点：
    - 避免单次插入数据量过大
    - 减少内存占用
    - 可以显示进度
    - 更好的错误处理
    
    缺点：
    - 代码相对复杂
    - 需要多次提交
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    start_time = time.time()
    total_created = 0
    total_failed = 0
    
    try:
        # 分批处理
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            
            try:
                # 批量插入当前批次
                product_dicts = [p.model_dump() for p in batch]
                db.bulk_insert_mappings(Product, product_dicts)
                db.commit()
                total_created += len(batch)
                
            except Exception as e:
                db.rollback()
                total_failed += len(batch)
                print(f"批次 {i//batch_size + 1} 失败: {str(e)}")
                # 可以选择继续处理下一批，或者抛出异常
                # 这里选择继续处理
                continue
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=total_failed == 0,
            created_count=total_created,
            failed_count=total_failed,
            execution_time=round(execution_time, 4),
            message=f"批量创建完成。成功: {total_created}, 失败: {total_failed}"
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 方案6：使用事务处理和错误恢复
# ============================================================================

@app.post("/products/bulk-create/method6", response_model=BulkCreateResponse)
async def bulk_create_products_method6(
    products: List[ProductCreate],
    db: Session = Depends(get_db)
):
    """
    方案6：带事务处理和错误恢复的批量创建
    
    优点：
    - 完整的事务支持
    - 详细的错误处理
    - 可以实现部分成功
    
    缺点：
    - 性能可能较低
    - 代码较复杂
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    start_time = time.time()
    created_count = 0
    failed_count = 0
    
    # 使用 savepoint 实现部分成功
    for product in products:
        # 为每个产品创建一个 savepoint
        savepoint = db.begin_nested()
        
        try:
            db_product = Product(**product.model_dump())
            db.add(db_product)
            db.flush()  # 刷新以捕获错误
            savepoint.commit()
            created_count += 1
            
        except Exception as e:
            savepoint.rollback()
            failed_count += 1
            print(f"创建产品失败: {product.name}, 错误: {str(e)}")
    
    # 提交所有成功的插入
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"提交失败: {str(e)}")
    
    execution_time = time.time() - start_time
    
    return BulkCreateResponse(
        success=failed_count == 0,
        created_count=created_count,
        failed_count=failed_count,
        execution_time=round(execution_time, 4),
        message=f"批量创建完成。成功: {created_count}, 失败: {failed_count}"
    )


# ============================================================================
# 用户批量创建示例
# ============================================================================

@app.post("/users/bulk-create", response_model=BulkCreateResponse)
async def bulk_create_users(
    users: List[UserCreate],
    db: Session = Depends(get_db)
):
    """
    批量创建用户示例
    使用最佳实践：bulk_insert_mappings
    """
    if not users:
        raise HTTPException(status_code=400, detail="用户列表不能为空")
    
    start_time = time.time()
    
    try:
        user_dicts = [user.model_dump() for user in users]
        db.bulk_insert_mappings(User, user_dicts)
        db.commit()
        
        execution_time = time.time() - start_time
        
        return BulkCreateResponse(
            success=True,
            created_count=len(users),
            failed_count=0,
            execution_time=round(execution_time, 4),
            message=f"成功批量创建 {len(users)} 个用户"
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# ============================================================================
# 查询接口（用于测试）
# ============================================================================

@app.get("/products", response_model=List[ProductResponse])
async def get_products(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取产品列表"""
    products = db.query(Product).offset(skip).limit(limit).all()
    return products


@app.get("/users", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取用户列表"""
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@app.delete("/products/clear")
async def clear_products(db: Session = Depends(get_db)):
    """清空产品表（测试用）"""
    try:
        db.query(Product).delete()
        db.commit()
        return {"message": "产品表已清空"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"清空失败: {str(e)}")


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "message": "FastAPI 批量创建示例 API",
        "endpoints": {
            "method1": "/products/bulk-create/method1 - bulk_save_objects",
            "method2": "/products/bulk-create/method2 - bulk_insert_mappings (推荐)",
            "method3": "/products/bulk-create/method3 - 逐个添加批量提交",
            "method4": "/products/bulk-create/method4 - 原生SQL",
            "method5": "/products/bulk-create/method5 - 分批插入",
            "method6": "/products/bulk-create/method6 - 事务处理",
            "users": "/users/bulk-create - 用户批量创建",
        },
        "docs": "/docs - Swagger UI 文档"
    }


# ============================================================================
# 运行说明
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    =============================================================
    FastAPI 批量创建示例启动
    =============================================================
    
    访问以下地址：
    - API 文档: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    
    推荐方案：
    1. 小数据量 (<1000): 使用 method1 (bulk_save_objects)
    2. 中等数据量 (1000-10000): 使用 method2 (bulk_insert_mappings)
    3. 大数据量 (>10000): 使用 method5 (分批插入)
    4. 需要返回ID: 使用 method3
    5. 需要最大控制: 使用 method4 (原生SQL)
    6. 需要容错: 使用 method6 (事务处理)
    
    =============================================================
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
