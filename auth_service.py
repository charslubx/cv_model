"""
认证服务类 - 纯Service实现，不依赖中间件

从中间件迁移而来，所有认证逻辑封装在Service中
在接口中通过依赖注入或直接调用使用
"""
import time
import socket
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Request
import logging

logger = logging.getLogger(__name__)


class AuthService:
    """
    认证服务类
    
    功能：
    1. 从请求中获取REMOTE_USER
    2. 认证用户或创建新用户
    3. 刷新过期用户信息
    4. 设置 request.state.user
    
    使用方式：
        auth_service = AuthService(ldap_url, base_dn, timeout)
        await auth_service.authenticate(request, db_session)
        # request.state.user 已被自动设置
    """
    
    def __init__(
        self, 
        ldap_server_url: str, 
        ldap_base_dn: str, 
        ldap_timeout: int = 30,
        db_session_factory=None
    ):
        """
        初始化认证服务
        
        Args:
            ldap_server_url: LDAP 服务器地址
            ldap_base_dn: LDAP Base DN
            ldap_timeout: LDAP 超时时间（秒）
            db_session_factory: 数据库会话工厂（可选）
        """
        self.ldap_server_url = ldap_server_url
        self.ldap_base_dn = ldap_base_dn
        self.ldap_timeout = ldap_timeout
        self.db_session_factory = db_session_factory
        
        # 延迟初始化
        self._auth_backend = None
        self._user_service = None
    
    @property
    def auth_backend(self):
        """延迟初始化认证后端"""
        if self._auth_backend is None:
            # TODO: 根据实际项目修改导入路径
            from user_backend import UserBackend
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
            # TODO: 根据实际项目修改导入路径
            from user_service import UserService
            self._user_service = UserService(
                ldap_server_url=self.ldap_server_url,
                ldap_base_dn=self.ldap_base_dn,
                ldap_timeout=self.ldap_timeout,
            )
        return self._user_service
    
    async def authenticate(
        self, 
        request: Request, 
        db_session=None
    ) -> Optional[dict]:
        """
        完整认证流程 - 主方法
        
        自动执行：
        1. 从IIS获取REMOTE_USER
        2. 检查用户是否已认证
        3. 认证或创建用户
        4. 刷新过期信息
        5. 设置 request.state.user
        6. 更新登录时间
        
        Args:
            request: FastAPI Request 对象
            db_session: 数据库会话（可选，如果未提供则使用工厂创建）
        
        Returns:
            认证后的用户对象，失败返回 None
            
        副作用：
            - 设置 request.state.user
            - 设置 request.state.idsid
            - 设置 request.state.server_name
        """
        try:
            # 使用传入的会话或创建新会话
            if db_session is not None:
                user = await self._do_authenticate(request, db_session)
            elif self.db_session_factory is not None:
                async with self.db_session_factory() as session:
                    user = await self._do_authenticate(request, session)
            else:
                raise RuntimeError(
                    "必须提供 db_session 参数或在初始化时提供 db_session_factory"
                )
            
            # 设置 request.state
            request.state.user = user
            request.state.idsid = user.idsid if user and hasattr(user, 'idsid') else None
            request.state.server_name = socket.gethostname()
            
            return user
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            logger.exception(e)
            request.state.user = None
            request.state.idsid = None
            return None
    
    async def _do_authenticate(
        self, 
        request: Request, 
        db_session
    ) -> Optional[dict]:
        """
        执行认证的内部方法
        
        Args:
            request: FastAPI Request 对象
            db_session: 数据库会话
        
        Returns:
            用户对象或 None
        """
        # 1. 获取REMOTE_USER
        remote_user = self.get_remote_user(request)
        if not remote_user:
            logger.warning("REMOTE_USER not found, skip authentication")
            return None
        
        # 2. 提取idsid和domain
        idsid, domain = self.auth_backend._extract_idsid(remote_user)
        if not idsid:
            logger.warning(f"Failed to extract idsid from REMOTE_USER: {remote_user}")
            return None
        
        # 3. 检查是否已认证且需要刷新
        if hasattr(request.state, 'user') and request.state.user:
            if await self.user_service.should_refresh_user(request.state.user):
                return await self.user_service.refresh_user(db_session, request.state.user)
            return request.state.user
        
        # 4. 未认证，开始认证流程
        user = await self._authenticate_and_login(request, db_session, idsid, domain)
        return user
    
    async def _authenticate_and_login(
        self, 
        request: Request, 
        db_session, 
        idsid: str, 
        domain: str
    ) -> Optional[dict]:
        """
        认证并登录用户
        
        流程：
        1. 尝试认证
        2. 认证失败则创建新用户
        3. 用户存在但过期则刷新
        4. 登录用户（更新last_login）
        
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
    
    async def refresh_user_if_needed(
        self, 
        user, 
        db_session
    ) -> Optional[dict]:
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
    def get_remote_user(request: Request) -> Optional[str]:
        """
        从请求中获取 REMOTE_USER
        
        尝试从多个位置获取：
        1. request.headers["REMOTE_USER"]
        2. request.headers["X-Remote-User"]
        3. request.scope["environ"]["REMOTE_USER"]
        
        Args:
            request: FastAPI Request 对象
        
        Returns:
            REMOTE_USER 值或 None
        """
        # 从请求头获取
        remote_user = (
            request.headers.get("REMOTE_USER") or
            request.headers.get("X-Remote-User")
        )
        
        if remote_user:
            return remote_user
        
        # 从 environ 获取
        try:
            environ = request.scope.get("environ", {})
            return environ.get("REMOTE_USER")
        except Exception:
            return None
    
    @staticmethod
    def set_request_context(request: Request, request_id: Optional[str] = None):
        """
        设置请求上下文信息
        
        Args:
            request: FastAPI Request 对象
            request_id: Request ID（可选，不提供则自动生成）
        """
        if request_id is None:
            request_id = f"req-{uuid.uuid4().hex}"
        
        request.state.request_id = request_id
        request.state.start_time = datetime.now(timezone.utc)
        request.state.server_name = socket.gethostname()
    
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
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return duration
        return 0.0
