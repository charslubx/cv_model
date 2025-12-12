"""
测试模型的 to_dict 方法
演示如何将 SQLAlchemy 模型转换为字典
"""

from models import UserGroup, PermissionUser, UserGroupResponse, PermissionUserResponse


# ============================================================
# 使用示例
# ============================================================

def example_basic_usage():
    """基础使用示例"""
    
    # 假设从数据库查询到了用户
    # user = db.query(UserProfile).filter(UserProfile.id == 82).first()
    # <UserProfile(id=82, idsid=zhuqinyx, username=zhuqinyx)>
    
    # 方法 1: 使用 to_dict() - 推荐 ⭐⭐⭐⭐⭐
    user_dict = user.to_dict()
    print(user_dict)
    # 输出: {'id': 82, 'idsid': 'zhuqinyx', 'username': 'zhuqinyx'}
    
    # 方法 2: 排除某些字段
    user_dict_no_id = user.to_dict(exclude=['id'])
    print(user_dict_no_id)
    # 输出: {'idsid': 'zhuqinyx', 'username': 'zhuqinyx'}
    
    # 方法 3: 使用 Pydantic 模型
    from pydantic import BaseModel
    
    class UserProfileResponse(BaseModel):
        id: int
        idsid: str
        username: str
        
        class Config:
            from_attributes = True
    
    user_pydantic = UserProfileResponse.model_validate(user).model_dump()
    print(user_pydantic)
    # 输出: {'id': 82, 'idsid': 'zhuqinyx', 'username': 'zhuqinyx'}


def example_with_datetime():
    """处理时间字段的示例"""
    
    # 假设查询到的 PermissionUser 对象
    # permission = db.query(PermissionUser).first()
    # <PermissionUser(permission_id='PERM001', user_id='USER123', 
    #                 create_time=datetime(...), update_time=datetime(...))>
    
    # 使用 to_dict() - 自动将 datetime 转为 ISO 格式字符串
    permission_dict = permission.to_dict()
    print(permission_dict)
    # 输出: {
    #     'permission_id': 'PERM001',
    #     'user_id': 'USER123',
    #     'create_time': '2025-12-12T10:30:00',
    #     'update_time': '2025-12-12T10:30:00'
    # }


def example_batch_conversion():
    """批量转换示例"""
    
    # 假设查询到多个用户
    # users = db.query(UserProfile).all()
    
    # 批量转换
    users_list = [user.to_dict() for user in users]
    print(users_list)
    # 输出: [
    #     {'id': 82, 'idsid': 'zhuqinyx', 'username': 'zhuqinyx'},
    #     {'id': 83, 'idsid': 'test', 'username': 'testuser'},
    #     ...
    # ]


def example_in_fastapi_route():
    """在 FastAPI 路由中使用"""
    
    from fastapi import APIRouter, Depends
    from sqlalchemy.orm import Session
    
    router = APIRouter()
    
    # 方式 1: 直接返回字典
    @router.get("/users/{user_id}")
    def get_user_dict(user_id: int, db: Session = Depends(get_db)):
        user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        
        if not user:
            return {"code": -1, "message": "用户不存在"}
        
        return {
            "code": 0,
            "message": "success",
            "data": user.to_dict()  # 使用 to_dict()
        }
    
    # 方式 2: 使用 Pydantic response_model（推荐）
    @router.get("/users/{user_id}", response_model=UserProfileResponse)
    def get_user_pydantic(user_id: int, db: Session = Depends(get_db)):
        user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        # FastAPI 自动转换，无需调用 to_dict()
        return user
    
    # 方式 3: 批量查询
    @router.get("/users/")
    def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
        users = db.query(UserProfile).offset(skip).limit(limit).all()
        
        return {
            "code": 0,
            "message": "success",
            "data": [user.to_dict() for user in users],  # 批量转换
            "total": len(users)
        }
    
    # 方式 4: 排除敏感字段
    @router.get("/users/{user_id}/public")
    def get_user_public(user_id: int, db: Session = Depends(get_db)):
        user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
        
        # 排除敏感字段
        return {
            "code": 0,
            "data": user.to_dict(exclude=['id', 'password'])
        }


def example_comparison():
    """各种方法的对比"""
    
    # 假设有这个对象
    # user = <UserProfile(id=82, idsid=zhuqinyx, username=zhuqinyx)>
    
    # ==================== 方法 1: to_dict() - 推荐 ⭐⭐⭐⭐⭐ ====================
    user_dict = user.to_dict()
    # 优点: 简单、自动处理 datetime、可排除字段
    # 缺点: 无
    
    # ==================== 方法 2: __dict__ ====================
    user_dict = user.__dict__.copy()
    user_dict.pop('_sa_instance_state', None)
    # 优点: 快速
    # 缺点: 包含内部属性、不处理 datetime
    
    # ==================== 方法 3: SQLAlchemy inspect ====================
    from sqlalchemy import inspect
    user_dict = {c.key: getattr(user, c.key)
                 for c in inspect(user).mapper.column_attrs}
    # 优点: 精确
    # 缺点: 代码冗长、不处理 datetime
    
    # ==================== 方法 4: Pydantic ====================
    from pydantic import BaseModel
    
    class UserProfileResponse(BaseModel):
        id: int
        idsid: str
        username: str
        class Config:
            from_attributes = True
    
    user_dict = UserProfileResponse.model_validate(user).model_dump()
    # 优点: 类型安全、验证
    # 缺点: 需要定义额外的 Pydantic 模型
    
    # ==================== 方法 5: 手动转换 ====================
    user_dict = {
        'id': user.id,
        'idsid': user.idsid,
        'username': user.username
    }
    # 优点: 完全控制
    # 缺点: 维护困难、代码冗长


# ============================================================
# 推荐使用方式总结
# ============================================================

"""
✅ 推荐方式 (按优先级):

1. 【最推荐】FastAPI 路由 + Pydantic response_model
   @router.get("/users/{id}", response_model=UserProfileResponse)
   def get_user(...):
       return user  # FastAPI 自动转换

2. 【次推荐】模型的 to_dict() 方法
   user_dict = user.to_dict()
   user_dict = user.to_dict(exclude=['password'])

3. 【备选】Pydantic 手动转换
   user_dict = UserProfileResponse.model_validate(user).model_dump()

❌ 不推荐:
- __dict__ (包含内部属性)
- 手动逐个字段转换 (维护困难)


📝 最佳实践:

# models.py
class DeclBase(DeclarativeBase, DictMixin):
    pass

class UserProfile(DeclBase):
    __tablename__ = 'user_profile'
    id = Column(Integer, primary_key=True)
    idsid = Column(String(50))
    username = Column(String(100))

# routes.py
@router.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    
    return {
        "code": 0,
        "data": user.to_dict()  # 简单优雅 ✨
    }
"""


if __name__ == "__main__":
    print("查看代码中的示例！")
