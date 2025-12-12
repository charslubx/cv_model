"""
FastAPI 中将 SQLAlchemy 模型转换为字典的多种方法
"""

from typing import Any, Dict
from sqlalchemy import inspect
from pydantic import BaseModel


# ============================================================
# 方法 1: 使用 Pydantic 模型（推荐 ⭐⭐⭐⭐⭐）
# ============================================================

class UserProfileResponse(BaseModel):
    """Pydantic 响应模型"""
    id: int
    idsid: str
    username: str
    
    class Config:
        from_attributes = True  # Pydantic v2
        # orm_mode = True  # Pydantic v1


def method_1_pydantic(user_profile):
    """
    使用 Pydantic 模型转换（最推荐）
    
    优点：
    - 类型安全
    - 自动验证
    - 与 FastAPI 完美集成
    - 可以定义响应格式
    """
    # 方式 A: 使用 model_validate
    result = UserProfileResponse.model_validate(user_profile)
    return result.model_dump()
    
    # 方式 B: 直接返回 Pydantic 模型（FastAPI 会自动转换）
    # return UserProfileResponse.model_validate(user_profile)


# ============================================================
# 方法 2: 使用 __dict__（简单快速）
# ============================================================

def method_2_dict(user_profile):
    """
    使用 __dict__ 转换
    
    优点：简单快速
    缺点：会包含 SQLAlchemy 内部属性（如 _sa_instance_state）
    """
    result = user_profile.__dict__.copy()
    # 移除 SQLAlchemy 内部属性
    result.pop('_sa_instance_state', None)
    return result


# ============================================================
# 方法 3: 使用 SQLAlchemy inspect（精确控制）
# ============================================================

def method_3_inspect(user_profile):
    """
    使用 SQLAlchemy inspect 转换
    
    优点：只包含实际的数据库字段
    """
    return {c.key: getattr(user_profile, c.key)
            for c in inspect(user_profile).mapper.column_attrs}


# ============================================================
# 方法 4: 手动转换（完全控制）
# ============================================================

def method_4_manual(user_profile):
    """
    手动转换
    
    优点：完全控制输出格式
    """
    return {
        'id': user_profile.id,
        'idsid': user_profile.idsid,
        'username': user_profile.username
    }


# ============================================================
# 方法 5: 通用转换函数
# ============================================================

def to_dict(model_instance, exclude_fields=None):
    """
    通用的模型转字典函数
    
    Args:
        model_instance: SQLAlchemy 模型实例
        exclude_fields: 要排除的字段列表
        
    Returns:
        字典
    """
    if exclude_fields is None:
        exclude_fields = []
    
    result = {}
    for c in inspect(model_instance).mapper.column_attrs:
        if c.key not in exclude_fields:
            value = getattr(model_instance, c.key)
            # 处理特殊类型（如 datetime）
            if hasattr(value, 'isoformat'):
                result[c.key] = value.isoformat()
            else:
                result[c.key] = value
    
    return result


# ============================================================
# 方法 6: 为模型添加 to_dict 方法（最优雅 ⭐⭐⭐⭐⭐）
# ============================================================

class DictMixin:
    """
    混入类，为 SQLAlchemy 模型添加 to_dict 方法
    """
    def to_dict(self, exclude=None, include_relationships=False):
        """
        将模型实例转换为字典
        
        Args:
            exclude: 要排除的字段列表
            include_relationships: 是否包含关系字段
            
        Returns:
            字典
        """
        if exclude is None:
            exclude = []
        
        result = {}
        
        # 获取所有列
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                # 处理 datetime 类型
                if hasattr(value, 'isoformat'):
                    result[column.name] = value.isoformat()
                else:
                    result[column.name] = value
        
        # 如果需要包含关系
        if include_relationships:
            for relationship in inspect(self.__class__).relationships:
                if relationship.key not in exclude:
                    related = getattr(self, relationship.key)
                    if related is not None:
                        if hasattr(related, 'to_dict'):
                            result[relationship.key] = related.to_dict()
                        elif isinstance(related, list):
                            result[relationship.key] = [
                                item.to_dict() if hasattr(item, 'to_dict') else str(item)
                                for item in related
                            ]
        
        return result


# 使用示例：在模型定义时混入
"""
from sqlalchemy.orm import DeclarativeBase

class DeclBase(DeclarativeBase, DictMixin):
    pass

class UserProfile(DeclBase):
    __tablename__ = 'user_profile'
    
    id = Column(Integer, primary_key=True)
    idsid = Column(String(50))
    username = Column(String(100))

# 使用
user = db.query(UserProfile).first()
user_dict = user.to_dict()  # 直接调用 to_dict()
"""


# ============================================================
# FastAPI 路由使用示例
# ============================================================

"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

router = APIRouter()

# ============================================================
# 示例 1: 使用 Pydantic 模型（最推荐）
# ============================================================

@router.get("/users/{user_id}", response_model=UserProfileResponse)
def get_user_v1(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    # FastAPI 会自动使用 Pydantic 模型转换
    return user


# ============================================================
# 示例 2: 手动转换为字典
# ============================================================

@router.get("/users/{user_id}/dict")
def get_user_v2(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    
    # 方法 1: 使用 Pydantic
    user_dict = UserProfileResponse.model_validate(user).model_dump()
    
    # 方法 2: 使用 __dict__
    # user_dict = user.__dict__.copy()
    # user_dict.pop('_sa_instance_state', None)
    
    # 方法 3: 使用 inspect
    # user_dict = {c.key: getattr(user, c.key)
    #              for c in inspect(user).mapper.column_attrs}
    
    # 方法 4: 使用 to_dict 方法（如果模型继承了 DictMixin）
    # user_dict = user.to_dict()
    
    return {
        "code": 0,
        "message": "success",
        "data": user_dict
    }


# ============================================================
# 示例 3: 批量转换
# ============================================================

@router.get("/users/")
def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(UserProfile).offset(skip).limit(limit).all()
    
    # 方法 1: 使用 Pydantic（推荐）
    users_list = [UserProfileResponse.model_validate(user).model_dump() for user in users]
    
    # 方法 2: 使用 to_dict
    # users_list = [user.to_dict() for user in users]
    
    # 方法 3: 使用通用函数
    # users_list = [to_dict(user) for user in users]
    
    return {
        "code": 0,
        "message": "success",
        "data": users_list,
        "total": len(users_list)
    }


# ============================================================
# 示例 4: 排除某些字段
# ============================================================

@router.get("/users/{user_id}/public")
def get_user_public(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    
    # 使用 Pydantic 排除字段
    user_dict = UserProfileResponse.model_validate(user).model_dump(
        exclude={'id'}  # 排除 id 字段
    )
    
    # 或使用 to_dict 函数
    # user_dict = to_dict(user, exclude_fields=['id'])
    
    return user_dict
"""


# ============================================================
# 性能对比和建议
# ============================================================

"""
性能对比（从快到慢）：
1. __dict__ (最快，但包含内部属性)
2. inspect (快，只包含列字段)
3. Pydantic (稍慢，但有验证和类型转换)
4. 手动转换 (取决于实现)

推荐使用场景：

⭐⭐⭐⭐⭐ Pydantic 模型 + response_model
- 适用于：API 响应
- 优点：类型安全、自动文档、验证
- 推荐度：最高

⭐⭐⭐⭐ 模型 to_dict() 方法（混入 DictMixin）
- 适用于：内部逻辑、灵活转换
- 优点：简洁、可控、复用
- 推荐度：高

⭐⭐⭐ SQLAlchemy inspect
- 适用于：通用转换、工具函数
- 优点：精确、不包含内部属性
- 推荐度：中

⭐⭐ __dict__
- 适用于：快速调试、性能关键场景
- 优点：最快
- 缺点：包含内部属性
- 推荐度：低

⭐ 手动转换
- 适用于：字段很少、需要特殊处理
- 优点：完全控制
- 缺点：维护困难
- 推荐度：最低
"""


# ============================================================
# 最佳实践示例
# ============================================================

"""
# 1. 定义 DeclBase 时混入 DictMixin
from sqlalchemy.orm import DeclarativeBase

class DeclBase(DeclarativeBase, DictMixin):
    pass


# 2. 定义模型
class UserProfile(DeclBase):
    __tablename__ = 'user_profile'
    
    id = Column(Integer, primary_key=True)
    idsid = Column(String(50))
    username = Column(String(100))


# 3. 定义 Pydantic 模型
class UserProfileResponse(BaseModel):
    id: int
    idsid: str
    username: str
    
    class Config:
        from_attributes = True


# 4. 在路由中使用
@router.get("/users/{user_id}", response_model=UserProfileResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    # FastAPI 自动转换，无需手动调用 to_dict()
    return user


# 5. 如果需要字典（用于其他逻辑）
def some_logic(user_id: int, db: Session):
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    
    # 方式 1: 使用模型的 to_dict() 方法
    user_dict = user.to_dict()
    
    # 方式 2: 使用 Pydantic
    user_dict = UserProfileResponse.model_validate(user).model_dump()
    
    return user_dict
"""
