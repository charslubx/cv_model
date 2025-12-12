"""
认证服务类 - 从中间件迁移的认证逻辑
"""
import time
import socket
from datetime import datetime
from typing import Optional
from fastapi import Request
import logging

logger = logging.getLogger(__name__)


class AuthService:
    """
    认证服务类
    
    负责处理用户认证、创建、刷新等业务逻辑
    从 HttpMiddleware 迁移而来
    """
    
    def __init__(self, ldap_server_url: str, ldap_base_dn: str, ldap_timeout: int = 30):
        """
        初始化认证服务
        
        Args:
            ldap_server_url: LDAP 服务器地址
            ldap_base_dn: LDAP Base DN
            ldap_timeout: LDAP 超时时间（秒）
        """
        self.ldap_server_url = ldap_server_url
        self.ldap_base_dn = ldap_base_dn
        self.ldap_timeout = ldap_timeout
        
        # 延迟初始化，避免在导入时就创建连接
        self._auth_backend = None
        self._user_service = None
    
    @property
    def auth_backend(self):
        """延迟初始化认证后端"""
        if self._auth_backend is None:
            from user_backend import UserBackend  # 假设的导入路径
            self._auth_backend = UserBackend(
                ldap_server_url=self.ldap_server_url,
                ldap_base_dn=self.ldap_base_dn,
                ldap_timeout=self.ldap_timeout,
            )
        return self._auth_backend
    
    @property
    def user_service(self):
        """延迟初始化用户服务"""
        if self._user_service is None:
            from user_service import UserService  # 假设的导入路径
            self._user_service = UserService(
                ldap_server_url=self.ldap_server_url,
                ldap_base_dn=self.ldap_base_dn,
                ldap_timeout=self.ldap_timeout,
            )
        return self._user_service
    
    async def authenticate_user(self, request: Request, db_session) -> Optional[dict]:
        """
        认证用户主流程
        
        1. 从IIS获取REMOTE_USER
        2. 检查用户是否已认证
        3. 如果未认证:
           - 调用authenticate认证用户
           - 如果认证失败，创建新用户
           - 如果用户存在但超过30天未更新，刷新用户信息
           - 登录用户
        
        Args:
            request: FastAPI Request 对象
            db_session: 数据库会话
        
        Returns:
            认证后的用户对象，如果认证失败则返回 None
        """
        try:
            # 获取REMOTE_USER
            remote_user = self._get_remote_user(request)
            
            if not remote_user:
                logger.warning("REMOTE_USER not found, skip authentication")
                return None
            
            # 提取idsid和domain
            idsid, domain = self.auth_backend._extract_idsid(remote_user)
            
            if not idsid:
                logger.warning(f"Failed to extract idsid from REMOTE_USER: {remote_user}")
                return None
            
            # 检查是否已认证且需要刷新
            if hasattr(request.state, 'user') and request.state.user:
                if await self.user_service.should_refresh_user(request.state.user):
                    return await self.user_service.refresh_user(db_session, request.state.user)
                return request.state.user
            
            # 未认证，开始认证流程
            user = await self._authenticate_and_login(request, db_session, idsid, domain)
            return user
            
        except Exception as e:
            logger.error(f"Authentication service failed: {e}")
            logger.exception(e)
            return None
    
    async def _authenticate_and_login(
        self, 
        request: Request, 
        db_session, 
        idsid: str, 
        domain: str
    ) -> Optional[dict]:
        """
        认证并登录用户的内部方法
        
        Args:
            request: FastAPI Request 对象
            db_session: 数据库会话
            idsid: 用户ID
            domain: 域名
        
        Returns:
            用户对象或 None
        """
        try:
            # 1. 尝试认证
            user = await self.auth_backend.authenticate(request, db_session)
            
            if not user:
                # 2. 认证失败，创建新用户
                logger.info(f"Creating new user: {idsid}")
                user = await self.user_service.create_user(db_session, idsid, domain)
            else:
                # 3. 用户存在，检查是否需要刷新（超过30天）
                if await self.user_service.should_refresh_user(user):
                    logger.info(f"Refreshing user info: {idsid}")
                    user = await self.user_service.refresh_user(db_session, user)
            
            # 4. 登录用户（更新last_login和update_time）
            if user:
                user = await self.user_service.login_user(db_session, user)
                logger.debug(f"User logged in: {user.idsid}")
                return user
            else:
                logger.error(f"Failed to create user: {idsid}")
                return None
                
        except Exception as e:
            logger.error(f"Authentication flow failed for {idsid}: {e}")
            logger.exception(e)
            return None
    
    async def refresh_user_if_needed(self, user, db_session) -> Optional[dict]:
        """
        如果需要，刷新用户信息
        
        Args:
            user: 用户对象
            db_session: 数据库会话
        
        Returns:
            刷新后的用户对象
        """
        try:
            if await self.user_service.should_refresh_user(user):
                return await self.user_service.refresh_user(db_session, user)
            return user
        except Exception as e:
            logger.error(f"Failed to refresh user: {e}")
            return user
    
    @staticmethod
    def _get_remote_user(request: Request) -> Optional[str]:
        """
        从请求中获取 REMOTE_USER
        
        Args:
            request: FastAPI Request 对象
        
        Returns:
            REMOTE_USER 值或 None
        """
        # 尝试从多个可能的位置获取 REMOTE_USER
        remote_user = (
            request.headers.get("REMOTE_USER") or
            request.headers.get("X-Remote-User") or
            getattr(request.scope.get("environ", {}), "REMOTE_USER", None)
        )
        return remote_user
    
    @staticmethod
    def get_server_name() -> str:
        """获取服务器主机名"""
        return socket.gethostname()


class RequestContextService:
    """
    请求上下文服务类
    
    负责管理请求ID、时间戳等上下文信息
    """
    
    @staticmethod
    def get_or_create_request_id(request: Request, prefix: str = "req-") -> str:
        """
        获取或创建 Request ID
        
        Args:
            request: FastAPI Request 对象
            prefix: Request ID 前缀
        
        Returns:
            Request ID
        """
        import uuid
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = f"{prefix}{uuid.uuid4().hex}"
        return request_id
    
    @staticmethod
    def set_request_context(request: Request, request_id: str):
        """
        设置请求上下文信息
        
        Args:
            request: FastAPI Request 对象
            request_id: Request ID
        """
        request.state.request_id = request_id
        request.state.start_time = datetime.utcnow()
    
    @staticmethod
    def get_request_duration(request: Request) -> float:
        """
        获取请求处理时长（秒）
        
        Args:
            request: FastAPI Request 对象
        
        Returns:
            处理时长（秒）
        """
        if hasattr(request.state, 'start_time'):
            start_time = request.state.start_time
            duration = (datetime.utcnow() - start_time).total_seconds()
            return duration
        return 0.0


class ExceptionHandlerService:
    """
    异常处理服务类
    """
    
    @staticmethod
    async def handle_exception(
        request: Request,
        exc: Exception,
        is_traceback: bool = True,
    ):
        """
        处理异常并返回错误响应
        
        Args:
            request: FastAPI Request 对象
            exc: 异常对象
            is_traceback: 是否打印堆栈跟踪
        
        Returns:
            JSON 错误响应
        """
        from fastapi.responses import JSONResponse
        
        logger_func = logger.exception if is_traceback else logger.error
        
        # 构建错误日志消息
        request_id = getattr(request.state, 'request_id', 'unknown')
        log_msg = (
            f'[{request_id}] - "{request.method} {request.url.path}" '
            f'500 {type(exc).__name__}: {exc}'
        )
        logger_func(log_msg)
        
        # 返回错误响应
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "request_id": request_id,
            }
        )
