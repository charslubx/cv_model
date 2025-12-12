"""
FastAPI 模型转换
将 Django ORM 模型转换为 FastAPI 使用的 Pydantic 和 SQLAlchemy 模型
"""

from typing import Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import DeclarativeBase


# SQLAlchemy Base
class DeclBase(DeclarativeBase):
    """SQLAlchemy 声明式基类"""
    pass


# ============================================================
# SQLAlchemy ORM 模型 (用于数据库操作)
# ============================================================
class UserGroup(DeclBase):
    """
    SQLAlchemy ORM 模型 - 对应 Django 的 models.Model
    用于实际的数据库操作
    """
    __tablename__ = 'user_group_detail'
    
    id = Column(Integer, primary_key=True, nullable=False)
    userprofile_id = Column(Integer, nullable=False)
    group_id = Column(Integer, nullable=False)
    idsid = Column(String(20), nullable=True)
    name = Column(String(1000), nullable=True)


# ============================================================
# Pydantic 模型 (用于 API 数据验证和序列化)
# ============================================================

class UserGroupBase(BaseModel):
    """
    基础 Pydantic 模型 - 包含共享字段
    """
    userprofile_id: int = Field(..., description="用户配置ID")
    group_id: int = Field(..., description="组ID")
    idsid: Optional[str] = Field(None, max_length=20, description="IDSID")
    name: Optional[str] = Field(None, max_length=1000, description="名称")
    
    class Config:
        # 允许从 ORM 对象创建 Pydantic 模型
        from_attributes = True  # Pydantic v2
        # orm_mode = True  # Pydantic v1 (如果使用旧版本)


class UserGroupCreate(UserGroupBase):
    """
    创建用户组时使用的模型 (POST 请求)
    """
    pass


class UserGroupUpdate(BaseModel):
    """
    更新用户组时使用的模型 (PUT/PATCH 请求)
    所有字段都是可选的
    """
    userprofile_id: Optional[int] = Field(None, description="用户配置ID")
    group_id: Optional[int] = Field(None, description="组ID")
    idsid: Optional[str] = Field(None, max_length=20, description="IDSID")
    name: Optional[str] = Field(None, max_length=1000, description="名称")
    
    class Config:
        from_attributes = True


class UserGroupResponse(UserGroupBase):
    """
    API 响应模型 (GET 请求返回)
    包含 ID 字段
    """
    id: int = Field(..., description="主键ID")
    
    class Config:
        from_attributes = True
        # 示例数据
        json_schema_extra = {
            "example": {
                "id": 1,
                "userprofile_id": 100,
                "group_id": 200,
                "idsid": "12345",
                "name": "示例组名"
            }
        }


# ============================================================
# 使用示例
# ============================================================

"""
# 1. 数据库配置示例
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 创建数据库引擎
DATABASE_URL = "postgresql://user:password@localhost/dbname"
# 或者 MySQL: "mysql+pymysql://user:password@localhost/dbname"
# 或者 SQLite: "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建所有表
DeclBase.metadata.create_all(bind=engine)


# 2. FastAPI 路由示例
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

app = FastAPI()

# 依赖项：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 创建用户组
@app.post("/user-groups/", response_model=UserGroupResponse)
def create_user_group(user_group: UserGroupCreate, db: Session = Depends(get_db)):
    db_user_group = UserGroup(**user_group.model_dump())
    db.add(db_user_group)
    db.commit()
    db.refresh(db_user_group)
    return db_user_group


# 获取单个用户组
@app.get("/user-groups/{user_group_id}", response_model=UserGroupResponse)
def read_user_group(user_group_id: int, db: Session = Depends(get_db)):
    db_user_group = db.query(UserGroup).filter(UserGroup.id == user_group_id).first()
    if db_user_group is None:
        raise HTTPException(status_code=404, detail="用户组不存在")
    return db_user_group


# 获取用户组列表
@app.get("/user-groups/", response_model=list[UserGroupResponse])
def read_user_groups(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    user_groups = db.query(UserGroup).offset(skip).limit(limit).all()
    return user_groups


# 更新用户组
@app.put("/user-groups/{user_group_id}", response_model=UserGroupResponse)
def update_user_group(
    user_group_id: int, 
    user_group: UserGroupUpdate, 
    db: Session = Depends(get_db)
):
    db_user_group = db.query(UserGroup).filter(UserGroup.id == user_group_id).first()
    if db_user_group is None:
        raise HTTPException(status_code=404, detail="用户组不存在")
    
    # 只更新提供的字段
    update_data = user_group.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_user_group, key, value)
    
    db.commit()
    db.refresh(db_user_group)
    return db_user_group


# 删除用户组
@app.delete("/user-groups/{user_group_id}")
def delete_user_group(user_group_id: int, db: Session = Depends(get_db)):
    db_user_group = db.query(UserGroup).filter(UserGroup.id == user_group_id).first()
    if db_user_group is None:
        raise HTTPException(status_code=404, detail="用户组不存在")
    
    db.delete(db_user_group)
    db.commit()
    return {"message": "用户组已删除"}
"""
