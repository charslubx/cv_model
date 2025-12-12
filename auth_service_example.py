"""
认证服务使用示例

展示如何使用迁移后的服务类架构
"""
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from typing import Optional

from http_middleware import HttpMiddleware, create_http_middleware
from auth_service import AuthService, RequestContextService


# ============================================
# 示例1: 在中间件中使用（推荐方式）
# ============================================

def example1_middleware_usage():
    """使用中间件自动处理认证"""
    app = FastAPI()
    
    # 配置对象（根据实际项目调整）
    class Config:
        ldap_server_url = "ldap://your-ldap-server.com"
        ldap_base_dn = "DC=example,DC=com"
        ldap_timeout = 30
    
    config = Config()
    
    # 添加中间件
    app.add_middleware(HttpMiddleware, config=config)
    
    @app.get("/api/user")
    async def get_current_user(request: Request):
        """获取当前用户信息"""
        user = getattr(request.state, 'user', None)
        if not user:
            return JSONResponse(
                status_code=401,
                content={"error": "Not authenticated"}
            )
        return {"user": user.idsid if hasattr(user, 'idsid') else None}
    
    return app


# ============================================
# 示例2: 在路由中直接使用服务类
# ============================================

def example2_direct_service_usage():
    """在路由中直接使用认证服务"""
    app = FastAPI()
    
    # 初始化认证服务（单例模式）
    auth_service = AuthService(
        ldap_server_url="ldap://your-ldap-server.com",
        ldap_base_dn="DC=example,DC=com",
        ldap_timeout=30
    )
    
    async def get_db_session():
        """依赖注入：获取数据库会话"""
        # 这里需要根据实际项目实现
        # 示例：
        # from database import SessionLocal
        # db = SessionLocal()
        # try:
        #     yield db
        # finally:
        #     await db.close()
        pass
    
    @app.post("/api/authenticate")
    async def authenticate(
        request: Request,
        db_session = Depends(get_db_session)
    ):
        """手动触发认证"""
        user = await auth_service.authenticate_user(request, db_session)
        if not user:
            return JSONResponse(
                status_code=401,
                content={"error": "Authentication failed"}
            )
        return {"user": user}
    
    @app.post("/api/refresh-user")
    async def refresh_user(
        request: Request,
        db_session = Depends(get_db_session)
    ):
        """刷新用户信息"""
        user = getattr(request.state, 'user', None)
        if not user:
            return JSONResponse(
                status_code=401,
                content={"error": "Not authenticated"}
            )
        
        refreshed_user = await auth_service.refresh_user_if_needed(user, db_session)
        return {"user": refreshed_user}
    
    return app


# ============================================
# 示例3: 使用上下文服务管理请求信息
# ============================================

def example3_context_service_usage():
    """使用上下文服务"""
    app = FastAPI()
    context_service = RequestContextService()
    
    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        """自定义中间件：添加请求上下文"""
        # 生成 request ID
        request_id = context_service.get_or_create_request_id(request)
        context_service.set_request_context(request, request_id)
        
        # 处理请求
        response = await call_next(request)
        
        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        
        # 记录请求时长
        duration = context_service.get_request_duration(request)
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response
    
    @app.get("/api/info")
    async def get_request_info(request: Request):
        """获取请求信息"""
        return {
            "request_id": getattr(request.state, 'request_id', None),
            "start_time": str(getattr(request.state, 'start_time', None)),
            "duration": context_service.get_request_duration(request)
        }
    
    return app


# ============================================
# 示例4: 组合使用多个服务
# ============================================

def example4_combined_services():
    """组合使用认证服务和上下文服务"""
    app = FastAPI()
    
    class Config:
        ldap_server_url = "ldap://your-ldap-server.com"
        ldap_base_dn = "DC=example,DC=com"
        ldap_timeout = 30
    
    # 初始化服务
    auth_service = AuthService(
        ldap_server_url=Config.ldap_server_url,
        ldap_base_dn=Config.ldap_base_dn,
        ldap_timeout=Config.ldap_timeout
    )
    context_service = RequestContextService()
    
    @app.middleware("http")
    async def custom_middleware(request: Request, call_next):
        """自定义中间件：组合多个服务"""
        # 1. 设置上下文
        request_id = context_service.get_or_create_request_id(request)
        context_service.set_request_context(request, request_id)
        
        # 2. 认证（根据需要）
        if request.url.path.startswith("/api/protected/"):
            async with get_db_session() as session:  # 需要实现
                user = await auth_service.authenticate_user(request, session)
                if not user:
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Authentication required"}
                    )
        
        # 3. 继续处理请求
        response = await call_next(request)
        
        # 4. 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Server"] = auth_service.get_server_name()
        
        return response
    
    @app.get("/api/protected/data")
    async def protected_route(request: Request):
        """受保护的路由"""
        user = getattr(request.state, 'user', None)
        return {
            "message": "Protected data",
            "user": user.idsid if user and hasattr(user, 'idsid') else None
        }
    
    return app


# ============================================
# 主程序入口
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    # 选择一个示例运行
    # app = example1_middleware_usage()
    # app = example2_direct_service_usage()
    # app = example3_context_service_usage()
    app = example4_combined_services()
    
    print("服务已启动，访问 http://localhost:8000")
    print("API 文档: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============================================
# 测试辅助函数
# ============================================

async def get_db_session():
    """
    数据库会话获取函数（需要根据实际项目实现）
    
    示例实现：
    """
    # 方案1: SQLAlchemy async session
    # from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    # from sqlalchemy.orm import sessionmaker
    # 
    # engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
    # async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    # 
    # async with async_session() as session:
    #     yield session
    
    # 方案2: 使用全局对象
    # from globals import g
    # async with g.user_db_async_session() as session:
    #     yield session
    
    # 占位实现
    class MockSession:
        pass
    
    yield MockSession()
