"""
认证服务单元测试

测试迁移后的服务类功能
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from auth_service import AuthService, RequestContextService, ExceptionHandlerService


# ============================================
# AuthService 测试
# ============================================

class TestAuthService:
    """认证服务测试类"""
    
    @pytest.fixture
    def auth_service(self):
        """创建认证服务实例"""
        return AuthService(
            ldap_server_url="ldap://test-server.com",
            ldap_base_dn="DC=test,DC=com",
            ldap_timeout=30
        )
    
    @pytest.fixture
    def mock_request(self):
        """创建模拟请求对象"""
        request = Mock()
        request.headers = {
            "REMOTE_USER": "test-domain\\test-user"
        }
        request.state = Mock()
        request.scope = {"environ": {}}
        return request
    
    @pytest.fixture
    def mock_db_session(self):
        """创建模拟数据库会话"""
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, auth_service, mock_request, mock_db_session):
        """测试：用户认证成功"""
        # Mock 用户对象
        mock_user = Mock()
        mock_user.idsid = "test-user"
        mock_user.last_updated = datetime.utcnow()
        
        # Mock 认证后端
        with patch.object(auth_service, 'auth_backend') as mock_backend:
            mock_backend._extract_idsid.return_value = ("test-user", "test-domain")
            mock_backend.authenticate = AsyncMock(return_value=mock_user)
            
            # Mock 用户服务
            with patch.object(auth_service, 'user_service') as mock_service:
                mock_service.should_refresh_user = AsyncMock(return_value=False)
                mock_service.login_user = AsyncMock(return_value=mock_user)
                
                # 执行认证
                user = await auth_service.authenticate_user(mock_request, mock_db_session)
                
                # 验证结果
                assert user is not None
                assert user.idsid == "test-user"
                mock_backend.authenticate.assert_called_once()
                mock_service.login_user.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_authenticate_user_create_new(self, auth_service, mock_request, mock_db_session):
        """测试：认证失败时创建新用户"""
        mock_user = Mock()
        mock_user.idsid = "new-user"
        
        with patch.object(auth_service, 'auth_backend') as mock_backend:
            mock_backend._extract_idsid.return_value = ("new-user", "test-domain")
            mock_backend.authenticate = AsyncMock(return_value=None)  # 认证失败
            
            with patch.object(auth_service, 'user_service') as mock_service:
                mock_service.create_user = AsyncMock(return_value=mock_user)
                mock_service.login_user = AsyncMock(return_value=mock_user)
                
                # 执行认证
                user = await auth_service.authenticate_user(mock_request, mock_db_session)
                
                # 验证创建了新用户
                assert user is not None
                assert user.idsid == "new-user"
                mock_service.create_user.assert_called_once_with(
                    mock_db_session, "new-user", "test-domain"
                )
    
    @pytest.mark.asyncio
    async def test_authenticate_user_refresh_old_user(self, auth_service, mock_request, mock_db_session):
        """测试：用户信息过期时刷新"""
        old_user = Mock()
        old_user.idsid = "old-user"
        old_user.last_updated = datetime.utcnow() - timedelta(days=31)  # 31天前
        
        refreshed_user = Mock()
        refreshed_user.idsid = "old-user"
        refreshed_user.last_updated = datetime.utcnow()
        
        with patch.object(auth_service, 'auth_backend') as mock_backend:
            mock_backend._extract_idsid.return_value = ("old-user", "test-domain")
            mock_backend.authenticate = AsyncMock(return_value=old_user)
            
            with patch.object(auth_service, 'user_service') as mock_service:
                mock_service.should_refresh_user = AsyncMock(return_value=True)
                mock_service.refresh_user = AsyncMock(return_value=refreshed_user)
                mock_service.login_user = AsyncMock(return_value=refreshed_user)
                
                # 执行认证
                user = await auth_service.authenticate_user(mock_request, mock_db_session)
                
                # 验证刷新了用户
                assert user is not None
                mock_service.refresh_user.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_authenticate_user_no_remote_user(self, auth_service, mock_db_session):
        """测试：没有 REMOTE_USER 时返回 None"""
        request = Mock()
        request.headers = {}  # 没有 REMOTE_USER
        request.scope = {"environ": {}}
        
        user = await auth_service.authenticate_user(request, mock_db_session)
        assert user is None
    
    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_idsid(self, auth_service, mock_request, mock_db_session):
        """测试：无效的 idsid 格式"""
        with patch.object(auth_service, 'auth_backend') as mock_backend:
            mock_backend._extract_idsid.return_value = (None, None)  # 提取失败
            
            user = await auth_service.authenticate_user(mock_request, mock_db_session)
            assert user is None
    
    @pytest.mark.asyncio
    async def test_refresh_user_if_needed_refresh(self, auth_service):
        """测试：需要刷新用户"""
        old_user = Mock()
        old_user.idsid = "test-user"
        
        refreshed_user = Mock()
        refreshed_user.idsid = "test-user"
        refreshed_user.refreshed = True
        
        mock_session = AsyncMock()
        
        with patch.object(auth_service, 'user_service') as mock_service:
            mock_service.should_refresh_user = AsyncMock(return_value=True)
            mock_service.refresh_user = AsyncMock(return_value=refreshed_user)
            
            result = await auth_service.refresh_user_if_needed(old_user, mock_session)
            
            assert result.refreshed is True
            mock_service.refresh_user.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_refresh_user_if_needed_no_refresh(self, auth_service):
        """测试：不需要刷新用户"""
        user = Mock()
        user.idsid = "test-user"
        
        mock_session = AsyncMock()
        
        with patch.object(auth_service, 'user_service') as mock_service:
            mock_service.should_refresh_user = AsyncMock(return_value=False)
            
            result = await auth_service.refresh_user_if_needed(user, mock_session)
            
            assert result == user
            mock_service.refresh_user.assert_not_called()
    
    def test_get_remote_user_from_headers(self, auth_service):
        """测试：从请求头获取 REMOTE_USER"""
        request = Mock()
        request.headers = {"REMOTE_USER": "domain\\user"}
        request.scope = {"environ": {}}
        
        remote_user = auth_service._get_remote_user(request)
        assert remote_user == "domain\\user"
    
    def test_get_remote_user_from_x_remote_user(self, auth_service):
        """测试：从 X-Remote-User 头获取"""
        request = Mock()
        request.headers = {"X-Remote-User": "domain\\user"}
        request.scope = {"environ": {}}
        
        remote_user = auth_service._get_remote_user(request)
        assert remote_user == "domain\\user"
    
    def test_get_remote_user_none(self, auth_service):
        """测试：没有 REMOTE_USER"""
        request = Mock()
        request.headers = {}
        request.scope = {"environ": {}}
        
        remote_user = auth_service._get_remote_user(request)
        assert remote_user is None
    
    def test_get_server_name(self):
        """测试：获取服务器名称"""
        server_name = AuthService.get_server_name()
        assert isinstance(server_name, str)
        assert len(server_name) > 0


# ============================================
# RequestContextService 测试
# ============================================

class TestRequestContextService:
    """请求上下文服务测试类"""
    
    @pytest.fixture
    def context_service(self):
        """创建上下文服务实例"""
        return RequestContextService()
    
    @pytest.fixture
    def mock_request(self):
        """创建模拟请求对象"""
        request = Mock()
        request.headers = {}
        request.state = Mock()
        return request
    
    def test_get_or_create_request_id_from_header(self, context_service, mock_request):
        """测试：从请求头获取 Request ID"""
        mock_request.headers = {"X-Request-ID": "existing-id-123"}
        
        request_id = context_service.get_or_create_request_id(mock_request)
        assert request_id == "existing-id-123"
    
    def test_get_or_create_request_id_generate_new(self, context_service, mock_request):
        """测试：生成新的 Request ID"""
        request_id = context_service.get_or_create_request_id(mock_request)
        
        assert request_id.startswith("req-")
        assert len(request_id) == 36  # req- + 32位hex
    
    def test_get_or_create_request_id_custom_prefix(self, context_service, mock_request):
        """测试：自定义 Request ID 前缀"""
        request_id = context_service.get_or_create_request_id(mock_request, prefix="api-")
        
        assert request_id.startswith("api-")
    
    def test_set_request_context(self, context_service, mock_request):
        """测试：设置请求上下文"""
        request_id = "test-req-id-123"
        
        context_service.set_request_context(mock_request, request_id)
        
        assert mock_request.state.request_id == request_id
        assert hasattr(mock_request.state, 'start_time')
        assert isinstance(mock_request.state.start_time, datetime)
    
    def test_get_request_duration(self, context_service, mock_request):
        """测试：获取请求处理时长"""
        # 设置开始时间为1秒前
        mock_request.state.start_time = datetime.utcnow() - timedelta(seconds=1)
        
        duration = context_service.get_request_duration(mock_request)
        
        # 验证时长约为1秒（允许误差）
        assert 0.9 <= duration <= 1.1
    
    def test_get_request_duration_no_start_time(self, context_service, mock_request):
        """测试：没有开始时间时返回0"""
        delattr(mock_request.state, 'start_time')  # 删除 start_time 属性
        
        duration = context_service.get_request_duration(mock_request)
        assert duration == 0.0


# ============================================
# ExceptionHandlerService 测试
# ============================================

class TestExceptionHandlerService:
    """异常处理服务测试类"""
    
    @pytest.fixture
    def exception_handler(self):
        """创建异常处理服务实例"""
        return ExceptionHandlerService()
    
    @pytest.fixture
    def mock_request(self):
        """创建模拟请求对象"""
        request = Mock()
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/test"
        request.state = Mock()
        request.state.request_id = "test-req-123"
        return request
    
    @pytest.mark.asyncio
    async def test_handle_exception_with_traceback(self, exception_handler, mock_request):
        """测试：处理异常并记录堆栈"""
        exc = ValueError("Test error")
        
        with patch('auth_service.logger') as mock_logger:
            response = await exception_handler.handle_exception(
                mock_request, exc, is_traceback=True
            )
            
            # 验证日志
            mock_logger.exception.assert_called_once()
            
            # 验证响应
            assert response.status_code == 500
            content = eval(response.body.decode())  # 解析JSON
            assert content['success'] is False
            assert content['error']['type'] == 'ValueError'
            assert content['error']['message'] == 'Test error'
            assert content['request_id'] == 'test-req-123'
    
    @pytest.mark.asyncio
    async def test_handle_exception_without_traceback(self, exception_handler, mock_request):
        """测试：处理异常不记录堆栈"""
        exc = ValueError("Test error")
        
        with patch('auth_service.logger') as mock_logger:
            response = await exception_handler.handle_exception(
                mock_request, exc, is_traceback=False
            )
            
            # 验证使用 logger.error 而不是 exception
            mock_logger.error.assert_called_once()
            mock_logger.exception.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_handle_exception_no_request_id(self, exception_handler, mock_request):
        """测试：没有 request_id 时使用默认值"""
        delattr(mock_request.state, 'request_id')
        exc = ValueError("Test error")
        
        response = await exception_handler.handle_exception(mock_request, exc)
        
        content = eval(response.body.decode())
        assert content['request_id'] == 'unknown'


# ============================================
# 集成测试
# ============================================

class TestServiceIntegration:
    """服务集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_authentication_flow(self):
        """测试：完整认证流程"""
        # 创建服务
        auth_service = AuthService(
            ldap_server_url="ldap://test.com",
            ldap_base_dn="DC=test,DC=com",
            ldap_timeout=30
        )
        context_service = RequestContextService()
        
        # 创建模拟请求
        request = Mock()
        request.headers = {"REMOTE_USER": "domain\\testuser"}
        request.scope = {"environ": {}}
        request.state = Mock()
        
        # 模拟数据库会话
        db_session = AsyncMock()
        
        # Mock 认证流程
        mock_user = Mock()
        mock_user.idsid = "testuser"
        mock_user.last_updated = datetime.utcnow()
        
        with patch.object(auth_service, 'auth_backend') as mock_backend, \
             patch.object(auth_service, 'user_service') as mock_service:
            
            mock_backend._extract_idsid.return_value = ("testuser", "domain")
            mock_backend.authenticate = AsyncMock(return_value=mock_user)
            mock_service.should_refresh_user = AsyncMock(return_value=False)
            mock_service.login_user = AsyncMock(return_value=mock_user)
            
            # 1. 设置上下文
            request_id = context_service.get_or_create_request_id(request)
            context_service.set_request_context(request, request_id)
            
            # 2. 认证
            user = await auth_service.authenticate_user(request, db_session)
            
            # 3. 验证
            assert user is not None
            assert user.idsid == "testuser"
            assert hasattr(request.state, 'request_id')
            assert hasattr(request.state, 'start_time')
            
            # 4. 获取请求时长
            duration = context_service.get_request_duration(request)
            assert duration >= 0


# ============================================
# 运行测试
# ============================================

if __name__ == "__main__":
    # 使用 pytest 运行测试
    pytest.main([__file__, "-v", "-s"])
