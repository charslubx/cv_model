"""
简洁使用示例 - 在接口中通过 request.state.user 获取用户

所有认证逻辑由 service 自动处理，接口代码非常简洁
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from http_middleware import HttpMiddleware


# ============================================
# 步骤 1: 配置认证参数
# ============================================

class Config:
    """认证配置"""
    ldap_server_url = "ldap://your-ldap-server.com"
    ldap_base_dn = "DC=example,DC=com"
    ldap_timeout = 30


# ============================================
# 步骤 2: 创建应用并添加中间件
# ============================================

app = FastAPI(title="用户认证 API")

# 添加认证中间件 - 自动处理所有认证逻辑
app.add_middleware(HttpMiddleware, config=Config())


# ============================================
# 步骤 3: 在接口中直接使用 request.state.user
# ============================================

@app.get("/api/user/info")
async def get_user_info(request: Request):
    """
    获取当前用户信息
    
    ✅ 中间件自动认证
    ✅ 直接从 request.state.user 获取用户
    ✅ 无需任何认证代码
    """
    user = request.state.user
    
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": "未认证", "message": "请先登录"}
        )
    
    return {
        "idsid": user.idsid if hasattr(user, 'idsid') else None,
        "username": user.username if hasattr(user, 'username') else None,
        "email": user.email if hasattr(user, 'email') else None,
    }


@app.get("/api/user/profile")
async def get_user_profile(request: Request):
    """获取用户详细资料"""
    user = request.state.user
    
    if not user:
        return JSONResponse(status_code=401, content={"error": "未认证"})
    
    # 直接使用 user 对象，所有信息已经由 service 处理好
    return {
        "profile": {
            "idsid": getattr(user, 'idsid', None),
            "name": getattr(user, 'name', None),
            "email": getattr(user, 'email', None),
            "department": getattr(user, 'department', None),
            "last_login": str(getattr(user, 'last_login', None)),
        }
    }


@app.post("/api/data/create")
async def create_data(request: Request):
    """创建数据（需要用户认证）"""
    user = request.state.user
    
    if not user:
        return JSONResponse(status_code=401, content={"error": "未认证"})
    
    # 获取请求数据
    body = await request.json()
    
    # 使用用户信息
    return {
        "message": "数据创建成功",
        "created_by": user.idsid if hasattr(user, 'idsid') else None,
        "data": body
    }


@app.get("/api/public/info")
async def public_info(request: Request):
    """
    公开接口（不需要认证）
    
    ✅ 即使不需要认证，中间件也会尝试认证
    ✅ 如果有 REMOTE_USER，user 会被设置
    ✅ 如果没有，user 为 None，接口仍然可以访问
    """
    user = request.state.user
    
    return {
        "message": "这是公开信息",
        "is_authenticated": user is not None,
        "user": user.idsid if user and hasattr(user, 'idsid') else "游客"
    }


@app.get("/api/admin/users")
async def list_users(request: Request):
    """管理员接口 - 列出所有用户"""
    user = request.state.user
    
    if not user:
        return JSONResponse(status_code=401, content={"error": "未认证"})
    
    # 检查管理员权限
    if not getattr(user, 'is_admin', False):
        return JSONResponse(status_code=403, content={"error": "需要管理员权限"})
    
    return {
        "users": [
            {"idsid": user.idsid, "name": "用户1"},
            # ... 更多用户
        ]
    }


# ============================================
# 辅助函数：获取当前用户（可选）
# ============================================

def get_current_user(request: Request):
    """
    辅助函数：获取当前用户
    
    如果未认证，返回 None
    """
    return request.state.user


def require_user(request: Request):
    """
    辅助函数：要求用户必须认证
    
    如果未认证，抛出异常
    """
    from fastapi import HTTPException
    
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="未认证")
    return user


def require_admin(request: Request):
    """
    辅助函数：要求管理员权限
    
    如果不是管理员，抛出异常
    """
    from fastapi import HTTPException
    
    user = require_user(request)
    if not getattr(user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


# ============================================
# 使用辅助函数的示例
# ============================================

@app.get("/api/v2/user/info")
async def get_user_info_v2(request: Request):
    """使用辅助函数的版本"""
    try:
        user = require_user(request)
        return {"idsid": user.idsid}
    except Exception as e:
        return JSONResponse(status_code=401, content={"error": str(e)})


@app.get("/api/v2/admin/dashboard")
async def admin_dashboard(request: Request):
    """管理员控制台（使用辅助函数）"""
    try:
        user = require_admin(request)
        return {
            "message": "欢迎来到管理员控制台",
            "admin": user.idsid
        }
    except Exception as e:
        return JSONResponse(
            status_code=getattr(e, 'status_code', 500),
            content={"error": str(e)}
        )


# ============================================
# 健康检查和元信息
# ============================================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "用户认证 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check(request: Request):
    """健康检查"""
    return {
        "status": "healthy",
        "server": getattr(request.state, 'server_name', 'unknown'),
        "request_id": getattr(request.state, 'request_id', 'unknown')
    }


# ============================================
# 运行应用
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 用户认证 API 已启动")
    print("=" * 60)
    print()
    print("📍 访问地址:")
    print("   - API 文档: http://localhost:8000/docs")
    print("   - 根路径:   http://localhost:8000/")
    print()
    print("📝 接口示例:")
    print("   - GET  /api/user/info         - 获取用户信息")
    print("   - GET  /api/user/profile      - 获取用户资料")
    print("   - POST /api/data/create       - 创建数据")
    print("   - GET  /api/public/info       - 公开信息")
    print("   - GET  /api/admin/users       - 管理员：用户列表")
    print()
    print("✅ 所有接口通过 request.state.user 自动获取用户")
    print("✅ 认证逻辑完全由 service 处理")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============================================
# 总结：接口代码非常简洁
# ============================================
"""
使用方式总结：

1️⃣ 添加中间件（只需一次）：
   app.add_middleware(HttpMiddleware, config=Config())

2️⃣ 在接口中获取用户（2行代码）：
   user = request.state.user
   if not user:
       return JSONResponse(status_code=401, ...)

3️⃣ 使用用户信息：
   user.idsid
   user.username
   user.email
   ...

✅ 就是这么简单！所有认证逻辑都在 service 中自动处理
✅ 接口代码只关注业务逻辑，不需要处理认证细节
✅ 代码清晰、易读、易维护
"""
