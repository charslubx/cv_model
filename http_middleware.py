"""
HTTP 中间件 - 使用服务类重构
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from contextvars import ContextVar
import logging

from auth_service import AuthService, RequestContextService, ExceptionHandlerService

logger = logging.getLogger(__name__)

# 上下文变量用于跟踪 request ID
request_id_var: ContextVar[str] = ContextVar('request_id', default='')


class HttpMiddleware(BaseHTTPMiddleware):
    """
    HTTP 中间件
    
    职责：
    1. 管理请求/响应生命周期
    2. 设置请求上下文
    3. 调用认证服务
    4. 处理异常
    5. 添加响应头
    
    业务逻辑已迁移到服务类中
    """
    
    _HEADERS = {
        # 可添加相关头
    }

    def __init__(self, app, config=None):
        """
        初始化中间件
        
        Args:
            app: FastAPI 应用实例
            config: 配置对象，包含 ldap_server_url, ldap_base_dn, ldap_timeout
        """
        super().__init__(app)
        self.config = config
        
        # 初始化服务类
        self._auth_service = None
        self._context_service = RequestContextService()
        self._exception_handler = ExceptionHandlerService()
    
    @property
    def auth_service(self) -> AuthService:
        """延迟初始化认证服务"""
        if self._auth_service is None:
            if self.config:
                self._auth_service = AuthService(
                    ldap_server_url=self.config.ldap_server_url,
                    ldap_base_dn=self.config.ldap_base_dn,
                    ldap_timeout=getattr(self.config, 'ldap_timeout', 30),
                )
            else:
                # 如果没有配置，使用默认值或从环境变量读取
                import os
                self._auth_service = AuthService(
                    ldap_server_url=os.getenv('LDAP_SERVER_URL', ''),
                    ldap_base_dn=os.getenv('LDAP_BASE_DN', ''),
                    ldap_timeout=int(os.getenv('LDAP_TIMEOUT', '30')),
                )
        return self._auth_service

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """
        处理请求的主入口
        
        Args:
            request: FastAPI Request 对象
            call_next: 下一个请求处理器
        
        Returns:
            Response 对象
        """
        # 1. 设置请求上下文
        request_id = self._context_service.get_or_create_request_id(request)
        self._context_service.set_request_context(request, request_id)
        token = request_id_var.set(request_id)

        try:
            # 2. 执行认证流程（通过服务类）
            await self._authenticate_request(request)

            # 3. 继续处理请求
            response = await call_next(request)

            # 4. 添加响应头
            self._add_response_headers(response, request_id)
            
            return response

        except Exception as exc:
            # 5. 异常处理（通过服务类）
            return await self._exception_handler.handle_exception(request, exc)
        finally:
            # 6. 清理上下文变量
            request_id_var.reset(token)
            
            # 7. 记录请求处理时长（可选）
            duration = self._context_service.get_request_duration(request)
            logger.debug(
                f"[{request_id}] {request.method} {request.url.path} "
                f"completed in {duration:.3f}s"
            )

    async def _authenticate_request(self, request: Request):
        """
        执行认证流程（委托给认证服务）
        
        Args:
            request: FastAPI Request 对象
        """
        try:
            # 使用全局数据库会话（假设通过 g.user_db_async_session 获取）
            # 这里需要根据实际项目调整
            async with self._get_db_session() as session:
                user = await self.auth_service.authenticate_user(request, session)
                
                # 设置用户信息到请求状态
                request.state.user = user
                if user:
                    request.state.idsid = user.idsid if hasattr(user, 'idsid') else None
                else:
                    request.state.idsid = None
                
                # 设置服务器名称
                request.state.server_name = AuthService.get_server_name()
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            logger.exception(e)
            request.state.user = None
            request.state.idsid = None
    
    def _add_response_headers(self, response: Response, request_id: str):
        """
        添加响应头
        
        Args:
            response: Response 对象
            request_id: Request ID
        """
        response.headers["X-Request-ID"] = request_id
        for key, value in self._HEADERS.items():
            if key not in response.headers:
                response.headers[key] = value
    
    async def _get_db_session(self):
        """
        获取数据库会话
        
        这是一个占位方法，需要根据实际项目调整
        例如：从 g.user_db_async_session() 或依赖注入获取
        """
        # 示例实现1：使用全局对象
        try:
            from globals import g  # 假设的导入
            return g.user_db_async_session()
        except ImportError:
            pass
        
        # 示例实现2：使用数据库连接池
        try:
            from database import get_async_session  # 假设的导入
            async with get_async_session() as session:
                yield session
        except ImportError:
            pass
        
        # 如果都失败，抛出异常
        raise NotImplementedError(
            "Database session not configured. "
            "Please implement _get_db_session method."
        )


# 便捷函数：创建中间件实例
def create_http_middleware(config=None):
    """
    创建 HTTP 中间件的工厂函数
    
    Args:
        config: 配置对象
    
    Returns:
        中间件类（可用于 app.add_middleware）
    """
    def middleware_factory(app):
        return HttpMiddleware(app, config=config)
    
    return middleware_factory
