"""
认证服务使用示例 - 纯Service实现，不使用中间件

展示如何在接口中直接使用 AuthService
"""
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from typing import Optional

from auth_service import AuthService


# ============================================
# 方式 1: 使用依赖注入（推荐）
# ============================================

app = FastAPI(title="认证服务示例")

# 全局配置
class Config:
    ldap_server_url = "ldap://your-ldap-server.com"
    ldap_base_dn = "DC=example,DC=com"
    ldap_timeout = 30

# 创建全局认证服务实例
_auth_service = None

def get_auth_service() -> AuthService:
    """获取认证服务单例"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService(
            ldap_server_url=Config.ldap_server_url,
            ldap_base_dn=Config.ldap_base_dn,
            ldap_timeout=Config.ldap_timeout,
            db_session_factory=get_db_session  # 传入数据库会话工厂
        )
    return _auth_service


async def get_db_session():
    """
    获取数据库会话（依赖注入）
    
    TODO: 根据实际项目实现
    """
    # 示例1: 使用全局对象
    # from your_project.globals import g
    # async with g.user_db_async_session() as session:
    #     yield session
    
    # 示例2: 使用 SQLAlchemy
    # from your_project.database import async_session_maker
    # async with async_session_maker() as session:
    #     yield session
    
    # 占位实现
    class MockSession:
        pass
    yield MockSession()


async def authenticate_request(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db_session = Depends(get_db_session)
):
    """
    认证依赖 - 在需要认证的接口中使用
    
    自动执行认证并设置 request.state.user
    """
    await auth_service.authenticate(request, db_session)


# ============================================
# 接口示例 1: 获取用户信息
# ============================================

@app.get("/api/user/info", dependencies=[Depends(authenticate_request)])
async def get_user_info(request: Request):
    """
    获取用户信息
    
    ✅ 使用依赖注入自动认证
    ✅ 直接从 request.state.user 获取用户
    """
    user = request.state.user
    
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": "未认证"}
        )
    
    return {
        "idsid": getattr(user, 'idsid', None),
        "username": getattr(user, 'username', None),
        "email": getattr(user, 'email', None),
    }


# ============================================
# 接口示例 2: 手动调用认证
# ============================================

@app.get("/api/user/profile")
async def get_user_profile(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db_session = Depends(get_db_session)
):
    """
    获取用户资料
    
    ✅ 手动调用认证服务
    ✅ request.state.user 被自动设置
    """
    # 手动调用认证
    await auth_service.authenticate(request, db_session)
    
    user = request.state.user
    
    if not user:
        return JSONResponse(status_code=401, content={"error": "未认证"})
    
    return {
        "profile": {
            "idsid": getattr(user, 'idsid', None),
            "name": getattr(user, 'name', None),
            "email": getattr(user, 'email', None),
        }
    }


# ============================================
# 接口示例 3: 创建数据
# ============================================

@app.post("/api/data/create")
async def create_data(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db_session = Depends(get_db_session)
):
    """创建数据（需要用户认证）"""
    # 调用认证
    await auth_service.authenticate(request, db_session)
    
    user = request.state.user
    if not user:
        return JSONResponse(status_code=401, content={"error": "未认证"})
    
    # 获取请求数据
    body = await request.json()
    
    return {
        "message": "数据创建成功",
        "created_by": getattr(user, 'idsid', None),
        "data": body
    }


# ============================================
# 接口示例 4: 公开接口（可选认证）
# ============================================

@app.get("/api/public/info")
async def public_info(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db_session = Depends(get_db_session)
):
    """
    公开接口（不强制认证）
    
    ✅ 如果有 REMOTE_USER，会尝试认证
    ✅ 如果没有，user 为 None，接口仍然可用
    """
    # 尝试认证（不强制）
    await auth_service.authenticate(request, db_session)
    
    user = request.state.user
    
    return {
        "message": "这是公开信息",
        "is_authenticated": user is not None,
        "user": getattr(user, 'idsid', None) if user else "游客"
    }


# ============================================
# 辅助函数：简化接口代码
# ============================================

async def require_user(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db_session = Depends(get_db_session)
):
    """
    依赖：要求用户必须认证
    
    如果未认证，自动返回 401
    """
    from fastapi import HTTPException
    
    await auth_service.authenticate(request, db_session)
    user = request.state.user
    
    if not user:
        raise HTTPException(status_code=401, detail="未认证")
    
    return user


async def require_admin(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db_session = Depends(get_db_session)
):
    """
    依赖：要求管理员权限
    
    如果不是管理员，自动返回 403
    """
    from fastapi import HTTPException
    
    user = await require_user(request, auth_service, db_session)
    
    if not getattr(user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    return user


# ============================================
# 使用辅助函数的示例
# ============================================

@app.get("/api/v2/user/info")
async def get_user_info_v2(user = Depends(require_user)):
    """
    使用辅助函数的版本
    
    ✅ 代码更简洁
    ✅ 自动认证和错误处理
    """
    return {
        "idsid": getattr(user, 'idsid', None),
        "username": getattr(user, 'username', None),
    }


@app.get("/api/v2/admin/dashboard")
async def admin_dashboard(user = Depends(require_admin)):
    """管理员控制台"""
    return {
        "message": "欢迎来到管理员控制台",
        "admin": getattr(user, 'idsid', None)
    }


# ============================================
# 方式 2: 不使用依赖注入，直接在接口中使用
# ============================================

# 创建全局服务实例
auth_service_instance = AuthService(
    ldap_server_url=Config.ldap_server_url,
    ldap_base_dn=Config.ldap_base_dn,
    ldap_timeout=Config.ldap_timeout,
)

@app.get("/api/v3/user/info")
async def get_user_info_v3(request: Request):
    """
    直接使用全局服务实例
    
    ✅ 不使用依赖注入
    ✅ 手动传入 db_session
    """
    # 获取数据库会话
    async with get_db_session() as db:
        # 调用认证
        await auth_service_instance.authenticate(request, db)
    
    user = request.state.user
    if not user:
        return JSONResponse(status_code=401, content={"error": "未认证"})
    
    return {"user": getattr(user, 'idsid', None)}


# ============================================
# 健康检查
# ============================================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "认证服务 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check(request: Request):
    """健康检查"""
    return {
        "status": "healthy",
        "server": socket.gethostname(),
    }


# ============================================
# 运行应用
# ============================================

if __name__ == "__main__":
    import uvicorn
    import socket
    
    print("=" * 60)
    print("🚀 认证服务 API 已启动（纯Service实现）")
    print("=" * 60)
    print()
    print("📍 访问地址:")
    print("   - API 文档: http://localhost:8000/docs")
    print("   - 根路径:   http://localhost:8000/")
    print()
    print("📝 接口示例:")
    print("   方式1: 使用依赖注入（推荐）")
    print("   - GET  /api/user/info         - 使用依赖注入自动认证")
    print("   - GET  /api/user/profile      - 手动调用认证")
    print("   - POST /api/data/create       - 创建数据")
    print()
    print("   方式2: 使用辅助函数")
    print("   - GET  /api/v2/user/info      - 简洁版")
    print("   - GET  /api/v2/admin/dashboard - 管理员")
    print()
    print("   方式3: 直接使用全局实例")
    print("   - GET  /api/v3/user/info      - 不使用依赖注入")
    print()
    print("✅ 所有方式最终都通过 request.state.user 获取用户")
    print("✅ 认证逻辑完全由 AuthService 处理")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============================================
# 总结：三种使用方式
# ============================================
"""
方式 1: 使用依赖注入（推荐）
-----------------------------
@app.get("/api/user", dependencies=[Depends(authenticate_request)])
async def get_user(request: Request):
    user = request.state.user
    return {"user": user.idsid}

优点：
- ✅ 代码简洁
- ✅ 自动认证
- ✅ 易于测试


方式 2: 使用辅助函数
-----------------------------
@app.get("/api/user")
async def get_user(user = Depends(require_user)):
    return {"user": user.idsid}

优点：
- ✅ 更简洁
- ✅ 自动错误处理
- ✅ 直接获取 user 对象


方式 3: 直接使用服务实例
-----------------------------
auth_service = AuthService(...)

@app.get("/api/user")
async def get_user(request: Request):
    async with get_db_session() as db:
        await auth_service.authenticate(request, db)
    user = request.state.user
    return {"user": user.idsid}

优点：
- ✅ 完全控制
- ✅ 灵活
- ✅ 不依赖FastAPI特性


推荐使用方式 1 或 2！
"""
