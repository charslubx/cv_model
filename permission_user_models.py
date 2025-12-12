"""
FastAPI 模型转换 - PermissionUser
将 Django ORM 模型转换为 FastAPI 使用的 Pydantic 和 SQLAlchemy 模型
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

# SQLAlchemy Base
Base = declarative_base()


# ============================================================
# SQLAlchemy ORM 模型 (用于数据库操作)
# ============================================================
class PermissionUser(Base):
    """
    SQLAlchemy ORM 模型 - 对应 Django 的 models.Model
    用于实际的数据库操作
    """
    __tablename__ = 'opeda_permission_user'
    
    # 由于没有明确的主键，需要定义一个
    # 如果 permission_id 和 user_id 组合是主键，可以这样设置：
    permission_id = Column(String(25), primary_key=True, nullable=False, comment="permission id")
    user_id = Column(String(50), primary_key=True, nullable=False, comment="user_id")
    
    # 自动时间戳字段
    # auto_now_add=True -> default=func.now()
    # auto_now=True -> default=func.now(), onupdate=func.now()
    create_time = Column(
        DateTime, 
        nullable=True,
        default=func.now(),  # 创建时自动设置当前时间
        comment="创建时间"
    )
    update_time = Column(
        DateTime, 
        nullable=True,
        default=func.now(),  # 创建时设置当前时间
        onupdate=func.now(),  # 更新时自动更新为当前时间
        comment="更新时间"
    )


# 如果使用自增 ID 作为主键，可以使用这个版本：
class PermissionUserWithId(Base):
    """
    带有自增 ID 的版本（如果需要独立主键）
    """
    __tablename__ = 'opeda_permission_user_v2'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    permission_id = Column(String(25), nullable=False, comment="permission id")
    user_id = Column(String(50), nullable=False, comment="user_id")
    create_time = Column(DateTime, nullable=True, default=func.now(), comment="创建时间")
    update_time = Column(DateTime, nullable=True, default=func.now(), onupdate=func.now(), comment="更新时间")


# ============================================================
# Pydantic 模型 (用于 API 数据验证和序列化)
# ============================================================

class PermissionUserBase(BaseModel):
    """
    基础 Pydantic 模型 - 包含共享字段
    """
    permission_id: str = Field(..., max_length=25, description="权限ID")
    user_id: str = Field(..., max_length=50, description="用户ID")
    
    class Config:
        from_attributes = True  # Pydantic v2
        # orm_mode = True  # Pydantic v1


class PermissionUserCreate(PermissionUserBase):
    """
    创建权限用户时使用的模型 (POST 请求)
    注意：不包含时间戳字段，因为它们是自动生成的
    """
    pass


class PermissionUserUpdate(BaseModel):
    """
    更新权限用户时使用的模型 (PUT/PATCH 请求)
    所有字段都是可选的
    注意：通常不允许修改主键字段
    """
    permission_id: Optional[str] = Field(None, max_length=25, description="权限ID")
    user_id: Optional[str] = Field(None, max_length=50, description="用户ID")
    # 时间戳字段通常不应该手动更新
    
    class Config:
        from_attributes = True


class PermissionUserResponse(PermissionUserBase):
    """
    API 响应模型 (GET 请求返回)
    包含时间戳字段
    """
    create_time: Optional[datetime] = Field(None, description="创建时间")
    update_time: Optional[datetime] = Field(None, description="更新时间")
    
    class Config:
        from_attributes = True
        # 示例数据
        json_schema_extra = {
            "example": {
                "permission_id": "PERM001",
                "user_id": "USER123",
                "create_time": "2025-12-12T10:30:00",
                "update_time": "2025-12-12T10:30:00"
            }
        }


# ============================================================
# 使用示例
# ============================================================

"""
# 1. 数据库配置示例
from sqlalchemy import create_engine, Integer
from sqlalchemy.orm import sessionmaker

# 创建数据库引擎
DATABASE_URL = "postgresql://user:password@localhost/dbname"
# 或者 MySQL: "mysql+pymysql://user:password@localhost/dbname"
# 或者 SQLite: "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建所有表
Base.metadata.create_all(bind=engine)


# 2. FastAPI 路由示例
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session

app = FastAPI()

# 依赖项：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 创建权限用户关系
@app.post("/permission-users/", response_model=PermissionUserResponse, status_code=status.HTTP_201_CREATED)
def create_permission_user(
    permission_user: PermissionUserCreate, 
    db: Session = Depends(get_db)
):
    # 检查是否已存在
    existing = db.query(PermissionUser).filter(
        PermissionUser.permission_id == permission_user.permission_id,
        PermissionUser.user_id == permission_user.user_id
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="该权限用户关系已存在"
        )
    
    # 创建新记录（时间戳会自动设置）
    db_permission_user = PermissionUser(**permission_user.model_dump())
    db.add(db_permission_user)
    db.commit()
    db.refresh(db_permission_user)
    return db_permission_user


# 获取单个权限用户关系
@app.get(
    "/permission-users/{permission_id}/{user_id}", 
    response_model=PermissionUserResponse
)
def read_permission_user(
    permission_id: str, 
    user_id: str, 
    db: Session = Depends(get_db)
):
    db_permission_user = db.query(PermissionUser).filter(
        PermissionUser.permission_id == permission_id,
        PermissionUser.user_id == user_id
    ).first()
    
    if db_permission_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="权限用户关系不存在"
        )
    
    return db_permission_user


# 获取用户的所有权限
@app.get("/users/{user_id}/permissions", response_model=list[PermissionUserResponse])
def read_user_permissions(
    user_id: str, 
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    permissions = db.query(PermissionUser).filter(
        PermissionUser.user_id == user_id
    ).offset(skip).limit(limit).all()
    
    return permissions


# 获取权限的所有用户
@app.get("/permissions/{permission_id}/users", response_model=list[PermissionUserResponse])
def read_permission_users(
    permission_id: str, 
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    users = db.query(PermissionUser).filter(
        PermissionUser.permission_id == permission_id
    ).offset(skip).limit(limit).all()
    
    return users


# 获取所有权限用户关系
@app.get("/permission-users/", response_model=list[PermissionUserResponse])
def read_permission_users_all(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    permission_users = db.query(PermissionUser).offset(skip).limit(limit).all()
    return permission_users


# 删除权限用户关系
@app.delete("/permission-users/{permission_id}/{user_id}")
def delete_permission_user(
    permission_id: str, 
    user_id: str, 
    db: Session = Depends(get_db)
):
    db_permission_user = db.query(PermissionUser).filter(
        PermissionUser.permission_id == permission_id,
        PermissionUser.user_id == user_id
    ).first()
    
    if db_permission_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="权限用户关系不存在"
        )
    
    db.delete(db_permission_user)
    db.commit()
    return {"message": "权限用户关系已删除"}


# 批量创建权限用户关系
@app.post("/permission-users/batch", response_model=list[PermissionUserResponse])
def create_permission_users_batch(
    permission_users: list[PermissionUserCreate], 
    db: Session = Depends(get_db)
):
    created_items = []
    
    for permission_user in permission_users:
        # 检查是否已存在
        existing = db.query(PermissionUser).filter(
            PermissionUser.permission_id == permission_user.permission_id,
            PermissionUser.user_id == permission_user.user_id
        ).first()
        
        if not existing:
            db_permission_user = PermissionUser(**permission_user.model_dump())
            db.add(db_permission_user)
            created_items.append(db_permission_user)
    
    db.commit()
    
    # 刷新所有创建的对象
    for item in created_items:
        db.refresh(item)
    
    return created_items


# 删除用户的所有权限
@app.delete("/users/{user_id}/permissions")
def delete_user_all_permissions(user_id: str, db: Session = Depends(get_db)):
    result = db.query(PermissionUser).filter(
        PermissionUser.user_id == user_id
    ).delete()
    
    db.commit()
    return {"message": f"已删除 {result} 个权限用户关系"}
"""
