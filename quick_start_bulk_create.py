"""
FastAPI 批量创建快速入门脚本

这个脚本提供了最简单的批量创建示例，帮助你快速上手
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import List
import uvicorn

# =============================================================================
# 1. 数据库配置（使用 SQLite 进行快速测试）
# =============================================================================

DATABASE_URL = "sqlite:///./test_bulk_create.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite 需要这个参数
    echo=True  # 打印 SQL 日志，方便调试
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# =============================================================================
# 2. 定义数据模型
# =============================================================================

class Product(Base):
    """产品数据库模型"""
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    category = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)


# 创建表
Base.metadata.create_all(bind=engine)


# =============================================================================
# 3. 定义请求和响应模型
# =============================================================================

class ProductCreate(BaseModel):
    """创建产品的请求模型"""
    name: str
    category: str
    price: float


class ProductResponse(BaseModel):
    """产品响应模型"""
    id: int
    name: str
    category: str
    price: float
    
    class Config:
        from_attributes = True


# =============================================================================
# 4. 创建 FastAPI 应用
# =============================================================================

app = FastAPI(title="批量创建快速入门")


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# 5. 批量创建 API（推荐方案）
# =============================================================================

@app.post("/products/bulk-create")
async def bulk_create_products(
    products: List[ProductCreate],
    db: Session = Depends(get_db)
):
    """
    批量创建产品（推荐方案）
    
    使用 bulk_insert_mappings 获得最佳性能
    """
    if not products:
        raise HTTPException(status_code=400, detail="产品列表不能为空")
    
    try:
        # 将 Pydantic 模型转换为字典列表
        product_dicts = [product.model_dump() for product in products]
        
        # 使用 bulk_insert_mappings 批量插入
        db.bulk_insert_mappings(Product, product_dicts)
        db.commit()
        
        return {
            "success": True,
            "message": f"成功创建 {len(products)} 个产品",
            "count": len(products)
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量创建失败: {str(e)}")


# =============================================================================
# 6. 查询 API（用于验证）
# =============================================================================

@app.get("/products", response_model=List[ProductResponse])
async def get_products(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """查询产品列表"""
    products = db.query(Product).offset(skip).limit(limit).all()
    return products


@app.get("/products/count")
async def get_product_count(db: Session = Depends(get_db)):
    """获取产品总数"""
    count = db.query(Product).count()
    return {"total": count}


@app.delete("/products/clear")
async def clear_products(db: Session = Depends(get_db)):
    """清空所有产品（测试用）"""
    try:
        count = db.query(Product).delete()
        db.commit()
        return {"message": f"已删除 {count} 个产品"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"清空失败: {str(e)}")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "FastAPI 批量创建快速入门",
        "docs": "/docs",
        "endpoints": {
            "bulk_create": "POST /products/bulk-create",
            "list": "GET /products",
            "count": "GET /products/count",
            "clear": "DELETE /products/clear"
        }
    }


# =============================================================================
# 7. 主函数
# =============================================================================

if __name__ == "__main__":
    print("""
    =========================================================
    FastAPI 批量创建快速入门
    =========================================================
    
    服务启动后，请访问：
    
    📄 API 文档: http://localhost:8000/docs
    🔍 测试界面: http://localhost:8000/docs
    
    快速测试：
    
    1. 访问 API 文档页面
    2. 找到 POST /products/bulk-create 接口
    3. 点击 "Try it out"
    4. 输入测试数据：
       [
         {"name": "产品1", "category": "分类A", "price": 99.99},
         {"name": "产品2", "category": "分类B", "price": 199.99}
       ]
    5. 点击 "Execute" 执行
    6. 使用 GET /products 查看创建的数据
    
    =========================================================
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
