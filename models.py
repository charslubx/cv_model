"""
FastAPI 模型转换 - 统一模型文件
将 Django ORM 模型转换为 FastAPI 使用的 Pydantic 和 SQLAlchemy 模型
"""

from typing import Optional, Any, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, inspect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


# ============================================================
# DictMixin - 为模型添加 to_dict 方法
# ============================================================
class DictMixin:
    """
    混入类，为 SQLAlchemy 模型添加便捷的字典转换方法
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将模型实例转换为字典（简洁版本）
        
        Returns:
            字典
            
        Example:
            user = db.query(UserProfile).first()
            user_dict = user.to_dict()
            # {'id': 82, 'idsid': 'zhuqinyx', 'username': 'zhuqinyx'}
        """
        # noinspection PyUnresolvedReferences
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ============================================================
# SQLAlchemy Base
# ============================================================
class DeclBase(DeclarativeBase, DictMixin):
    """SQLAlchemy 声明式基类 + 字典转换功能"""
    pass


# ============================================================
# SQLAlchemy ORM 模型 (用于数据库操作)
# ============================================================

class UserGroup(DeclBase):
    """用户组模型"""
    __tablename__ = 'user_group_detail'
    
    id = Column(Integer, primary_key=True, nullable=False)
    userprofile_id = Column(Integer, nullable=False)
    group_id = Column(Integer, nullable=False)
    idsid = Column(String(20), nullable=True)
    name = Column(String(1000), nullable=True)


class PermissionUser(DeclBase):
    """权限用户关系模型"""
    __tablename__ = 'opeda_permission_user'
    
    permission_id = Column(String(25), primary_key=True, nullable=False, comment="permission id")
    user_id = Column(String(50), primary_key=True, nullable=False, comment="user_id")
    create_time = Column(DateTime, nullable=True, default=func.now(), comment="创建时间")
    update_time = Column(DateTime, nullable=True, default=func.now(), onupdate=func.now(), comment="更新时间")


# ============================================================
# Pydantic 模型 - UserGroup
# ============================================================

class UserGroupBase(BaseModel):
    """用户组基础模型"""
    userprofile_id: int = Field(..., description="用户配置ID")
    group_id: int = Field(..., description="组ID")
    idsid: Optional[str] = Field(None, max_length=20, description="IDSID")
    name: Optional[str] = Field(None, max_length=1000, description="名称")
    
    class Config:
        from_attributes = True


class UserGroupCreate(UserGroupBase):
    """创建用户组模型"""
    pass


class UserGroupUpdate(BaseModel):
    """更新用户组模型"""
    userprofile_id: Optional[int] = Field(None, description="用户配置ID")
    group_id: Optional[int] = Field(None, description="组ID")
    idsid: Optional[str] = Field(None, max_length=20, description="IDSID")
    name: Optional[str] = Field(None, max_length=1000, description="名称")
    
    class Config:
        from_attributes = True


class UserGroupResponse(UserGroupBase):
    """用户组响应模型"""
    id: int = Field(..., description="主键ID")
    
    class Config:
        from_attributes = True
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
# Pydantic 模型 - PermissionUser
# ============================================================

class PermissionUserBase(BaseModel):
    """权限用户基础模型"""
    permission_id: str = Field(..., max_length=25, description="权限ID")
    user_id: str = Field(..., max_length=50, description="用户ID")
    
    class Config:
        from_attributes = True


class PermissionUserCreate(PermissionUserBase):
    """创建权限用户模型"""
    pass


class PermissionUserUpdate(BaseModel):
    """更新权限用户模型"""
    permission_id: Optional[str] = Field(None, max_length=25, description="权限ID")
    user_id: Optional[str] = Field(None, max_length=50, description="用户ID")
    
    class Config:
        from_attributes = True


class PermissionUserResponse(PermissionUserBase):
    """权限用户响应模型"""
    create_time: Optional[datetime] = Field(None, description="创建时间")
    update_time: Optional[datetime] = Field(None, description="更新时间")
    
    class Config:
        from_attributes = True
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
# 1. 数据库配置
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://user:password@localhost/dbname"
# 或者 MySQL: "mysql+pymysql://user:password@localhost/dbname"
# 或者 SQLite: "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建所有表
DeclBase.metadata.create_all(bind=engine)


# 2. FastAPI 应用
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session

app = FastAPI()

# 数据库依赖
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==================== UserGroup 路由 ====================

@app.post("/user-groups/", response_model=UserGroupResponse)
def create_user_group(user_group: UserGroupCreate, db: Session = Depends(get_db)):
    db_user_group = UserGroup(**user_group.model_dump())
    db.add(db_user_group)
    db.commit()
    db.refresh(db_user_group)
    return db_user_group


@app.get("/user-groups/{user_group_id}", response_model=UserGroupResponse)
def read_user_group(user_group_id: int, db: Session = Depends(get_db)):
    db_user_group = db.query(UserGroup).filter(UserGroup.id == user_group_id).first()
    if db_user_group is None:
        raise HTTPException(status_code=404, detail="用户组不存在")
    return db_user_group


@app.get("/user-groups/", response_model=list[UserGroupResponse])
def read_user_groups(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    user_groups = db.query(UserGroup).offset(skip).limit(limit).all()
    return user_groups


@app.put("/user-groups/{user_group_id}", response_model=UserGroupResponse)
def update_user_group(
    user_group_id: int, 
    user_group: UserGroupUpdate, 
    db: Session = Depends(get_db)
):
    db_user_group = db.query(UserGroup).filter(UserGroup.id == user_group_id).first()
    if db_user_group is None:
        raise HTTPException(status_code=404, detail="用户组不存在")
    
    update_data = user_group.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_user_group, key, value)
    
    db.commit()
    db.refresh(db_user_group)
    return db_user_group


@app.delete("/user-groups/{user_group_id}")
def delete_user_group(user_group_id: int, db: Session = Depends(get_db)):
    db_user_group = db.query(UserGroup).filter(UserGroup.id == user_group_id).first()
    if db_user_group is None:
        raise HTTPException(status_code=404, detail="用户组不存在")
    
    db.delete(db_user_group)
    db.commit()
    return {"message": "用户组已删除"}


# ==================== PermissionUser 路由 ====================

@app.post("/permission-users/", response_model=PermissionUserResponse, status_code=status.HTTP_201_CREATED)
def create_permission_user(permission_user: PermissionUserCreate, db: Session = Depends(get_db)):
    # 检查是否已存在
    existing = db.query(PermissionUser).filter(
        PermissionUser.permission_id == permission_user.permission_id,
        PermissionUser.user_id == permission_user.user_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="该权限用户关系已存在")
    
    db_permission_user = PermissionUser(**permission_user.model_dump())
    db.add(db_permission_user)
    db.commit()
    db.refresh(db_permission_user)
    return db_permission_user


@app.get("/permission-users/{permission_id}/{user_id}", response_model=PermissionUserResponse)
def read_permission_user(permission_id: str, user_id: str, db: Session = Depends(get_db)):
    db_permission_user = db.query(PermissionUser).filter(
        PermissionUser.permission_id == permission_id,
        PermissionUser.user_id == user_id
    ).first()
    
    if db_permission_user is None:
        raise HTTPException(status_code=404, detail="权限用户关系不存在")
    
    return db_permission_user


@app.get("/users/{user_id}/permissions", response_model=list[PermissionUserResponse])
def read_user_permissions(user_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    permissions = db.query(PermissionUser).filter(
        PermissionUser.user_id == user_id
    ).offset(skip).limit(limit).all()
    return permissions


@app.get("/permissions/{permission_id}/users", response_model=list[PermissionUserResponse])
def read_permission_users(permission_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(PermissionUser).filter(
        PermissionUser.permission_id == permission_id
    ).offset(skip).limit(limit).all()
    return users


@app.delete("/permission-users/{permission_id}/{user_id}")
def delete_permission_user(permission_id: str, user_id: str, db: Session = Depends(get_db)):
    db_permission_user = db.query(PermissionUser).filter(
        PermissionUser.permission_id == permission_id,
        PermissionUser.user_id == user_id
    ).first()
    
    if db_permission_user is None:
        raise HTTPException(status_code=404, detail="权限用户关系不存在")
    
    db.delete(db_permission_user)
    db.commit()
    return {"message": "权限用户关系已删除"}


@app.post("/permission-users/batch", response_model=list[PermissionUserResponse])
def create_permission_users_batch(permission_users: list[PermissionUserCreate], db: Session = Depends(get_db)):
    created_items = []
    
    for permission_user in permission_users:
        existing = db.query(PermissionUser).filter(
            PermissionUser.permission_id == permission_user.permission_id,
            PermissionUser.user_id == permission_user.user_id
        ).first()
        
        if not existing:
            db_permission_user = PermissionUser(**permission_user.model_dump())
            db.add(db_permission_user)
            created_items.append(db_permission_user)
    
    db.commit()
    
    for item in created_items:
        db.refresh(item)
    
    return created_items
"""
