# 认证中间件迁移指南

## 概述

本次迁移将原来的 `HttpMiddleware` 中间件拆分为多个服务类，遵循单一职责原则，使代码更加模块化和可测试。

## 迁移架构

### 原架构
```
HttpMiddleware (中间件)
├── 请求上下文管理
├── 认证逻辑
├── 异常处理
└── 响应头管理
```

### 新架构
```
http_middleware.py (中间件层 - 编排)
├── HttpMiddleware: 轻量级中间件，负责请求生命周期管理
│
auth_service.py (服务层 - 业务逻辑)
├── AuthService: 认证服务，处理用户认证、创建、刷新
├── RequestContextService: 请求上下文服务，管理 Request ID 和时间戳
└── ExceptionHandlerService: 异常处理服务，统一异常处理
```

## 主要改进

### 1. 职责分离

| 组件 | 职责 |
|------|------|
| `HttpMiddleware` | 请求/响应生命周期管理、服务编排 |
| `AuthService` | 用户认证、创建、刷新业务逻辑 |
| `RequestContextService` | Request ID 生成、时间跟踪 |
| `ExceptionHandlerService` | 异常处理和错误响应 |

### 2. 可测试性提升

**原代码问题**：
- 所有逻辑耦合在中间件中，难以单独测试
- 需要模拟整个请求/响应流程才能测试认证逻辑

**新代码优势**：
```python
# 可以独立测试认证服务
auth_service = AuthService(ldap_url, base_dn, timeout)
user = await auth_service.authenticate_user(mock_request, mock_session)
assert user is not None

# 可以独立测试上下文服务
context_service = RequestContextService()
request_id = context_service.get_or_create_request_id(mock_request)
assert request_id.startswith("req-")
```

### 3. 可复用性

**原代码**：认证逻辑只能在中间件中使用

**新代码**：服务类可以在多个地方使用
```python
# 在中间件中使用
middleware = HttpMiddleware(app, config)

# 在路由中使用
@app.post("/api/login")
async def login(request: Request, db: Session = Depends(get_db)):
    auth_service = AuthService(...)
    user = await auth_service.authenticate_user(request, db)
    return {"user": user}

# 在后台任务中使用
async def refresh_users_task():
    auth_service = AuthService(...)
    for user in users:
        await auth_service.refresh_user_if_needed(user, db)
```

### 4. 配置灵活性

**原代码**：依赖全局对象 `g.config`

**新代码**：支持多种配置方式
```python
# 方式1: 通过配置对象
class Config:
    ldap_server_url = "ldap://..."
    ldap_base_dn = "DC=example,DC=com"

app.add_middleware(HttpMiddleware, config=Config())

# 方式2: 通过环境变量
os.environ['LDAP_SERVER_URL'] = 'ldap://...'
app.add_middleware(HttpMiddleware)

# 方式3: 直接实例化
auth_service = AuthService(
    ldap_server_url="ldap://...",
    ldap_base_dn="DC=example,DC=com",
    ldap_timeout=30
)
```

## 迁移步骤

### 步骤 1: 部署新文件

将以下文件添加到项目中：
```
your_project/
├── auth_service.py           # 服务类
├── http_middleware.py        # 重构后的中间件
└── auth_service_example.py   # 使用示例
```

### 步骤 2: 更新中间件配置

**原代码**：
```python
from your_old_middleware import HttpMiddleware

app.add_middleware(HttpMiddleware)
```

**新代码**：
```python
from http_middleware import HttpMiddleware

# 需要传入配置对象
class Config:
    ldap_server_url = "ldap://your-ldap-server.com"
    ldap_base_dn = "DC=example,DC=com"
    ldap_timeout = 30

app.add_middleware(HttpMiddleware, config=Config())
```

### 步骤 3: 实现数据库会话获取

在 `http_middleware.py` 中，修改 `_get_db_session` 方法以匹配你的项目：

```python
async def _get_db_session(self):
    """获取数据库会话"""
    # 方案1: 使用全局对象
    from your_project.globals import g
    return g.user_db_async_session()
    
    # 方案2: 使用连接池
    from your_project.database import get_async_session
    async with get_async_session() as session:
        yield session
```

### 步骤 4: 更新依赖导入

确保以下依赖可用：
```python
# auth_service.py 中需要的导入
from user_backend import UserBackend    # 你的认证后端
from user_service import UserService    # 你的用户服务

# 根据实际项目调整导入路径
```

### 步骤 5: 测试迁移

运行测试确保一切正常：
```bash
# 单元测试服务类
pytest tests/test_auth_service.py

# 集成测试中间件
pytest tests/test_middleware_integration.py

# 手动测试
python auth_service_example.py
```

## 使用场景

### 场景 1: 标准 Web API（推荐）

使用完整的中间件栈：
```python
from http_middleware import HttpMiddleware

app = FastAPI()
app.add_middleware(HttpMiddleware, config=config)

@app.get("/api/user")
async def get_user(request: Request):
    user = request.state.user  # 自动认证
    return {"user": user}
```

### 场景 2: 部分路由需要认证

使用依赖注入：
```python
from auth_service import AuthService

auth_service = AuthService(...)

async def require_auth(request: Request, db: Session = Depends(get_db)):
    user = await auth_service.authenticate_user(request, db)
    if not user:
        raise HTTPException(status_code=401)
    return user

@app.get("/api/protected")
async def protected(user = Depends(require_auth)):
    return {"user": user}
```

### 场景 3: 后台任务

直接使用服务类：
```python
from auth_service import AuthService

async def refresh_users_background_task():
    auth_service = AuthService(...)
    async with get_db() as db:
        users = await get_users_need_refresh(db)
        for user in users:
            await auth_service.refresh_user_if_needed(user, db)
```

### 场景 4: 自定义认证流程

组合使用多个服务：
```python
from auth_service import AuthService, RequestContextService

auth_service = AuthService(...)
context_service = RequestContextService()

@app.middleware("http")
async def custom_auth_middleware(request: Request, call_next):
    # 1. 设置上下文
    request_id = context_service.get_or_create_request_id(request)
    context_service.set_request_context(request, request_id)
    
    # 2. 自定义认证逻辑
    if request.url.path.startswith("/admin/"):
        # 管理员路由需要特殊认证
        pass
    elif request.url.path.startswith("/api/"):
        # API 路由使用标准认证
        async with get_db() as db:
            await auth_service.authenticate_user(request, db)
    
    # 3. 继续处理
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

## API 参考

### AuthService

```python
class AuthService:
    def __init__(self, ldap_server_url: str, ldap_base_dn: str, ldap_timeout: int = 30)
    
    async def authenticate_user(self, request: Request, db_session) -> Optional[dict]
    """主认证流程：获取REMOTE_USER、认证、创建/刷新用户"""
    
    async def refresh_user_if_needed(self, user, db_session) -> Optional[dict]
    """如果用户信息过期（>30天），刷新用户信息"""
    
    @staticmethod
    def get_server_name() -> str
    """获取服务器主机名"""
```

### RequestContextService

```python
class RequestContextService:
    @staticmethod
    def get_or_create_request_id(request: Request, prefix: str = "req-") -> str
    """获取或生成 Request ID"""
    
    @staticmethod
    def set_request_context(request: Request, request_id: str)
    """设置请求上下文（ID和时间戳）"""
    
    @staticmethod
    def get_request_duration(request: Request) -> float
    """获取请求处理时长（秒）"""
```

### ExceptionHandlerService

```python
class ExceptionHandlerService:
    @staticmethod
    async def handle_exception(
        request: Request, 
        exc: Exception, 
        is_traceback: bool = True
    )
    """处理异常并返回标准错误响应"""
```

## 常见问题

### Q1: 为什么要拆分中间件？

**A**: 
1. **单一职责**：每个类只负责一个功能，便于维护
2. **可测试性**：可以独立测试每个服务类
3. **可复用性**：服务类可以在多个地方使用（中间件、路由、后台任务等）
4. **灵活性**：可以根据需要组合不同的服务

### Q2: 性能会受影响吗？

**A**: 
- 不会。服务类使用延迟初始化和单例模式
- 增加的开销可忽略不计（几微秒）
- 反而因为职责清晰，更容易优化性能瓶颈

### Q3: 需要修改现有路由代码吗？

**A**: 
- 不需要！接口保持向后兼容
- `request.state.user` 和 `request.state.request_id` 仍然可用
- 现有路由代码无需修改

### Q4: 如何处理全局对象 `g`？

**A**: 
两种方案：
```python
# 方案1: 在 http_middleware.py 中适配
async def _get_db_session(self):
    from your_project.globals import g
    return g.user_db_async_session()

# 方案2: 使用依赖注入
from fastapi import Depends

async def get_db():
    from your_project.globals import g
    async with g.user_db_async_session() as session:
        yield session
```

### Q5: 如何扩展认证逻辑？

**A**: 
继承并重写方法：
```python
from auth_service import AuthService

class CustomAuthService(AuthService):
    async def authenticate_user(self, request: Request, db_session):
        # 添加自定义认证逻辑
        user = await super().authenticate_user(request, db_session)
        
        # 额外的检查
        if user and not user.is_active:
            return None
        
        return user
```

## 迁移检查清单

- [ ] 添加 `auth_service.py` 文件
- [ ] 添加 `http_middleware.py` 文件  
- [ ] 更新 `app.add_middleware()` 调用
- [ ] 实现 `_get_db_session()` 方法
- [ ] 更新依赖导入路径
- [ ] 运行单元测试
- [ ] 运行集成测试
- [ ] 在测试环境验证
- [ ] 更新部署文档
- [ ] 在生产环境部署

## 回滚计划

如果迁移出现问题，可以快速回滚：

1. 恢复原来的中间件文件
2. 撤销 `app.add_middleware()` 的修改
3. 重启服务

建议保留原中间件代码至少一个发布周期。

## 技术支持

如有问题，请查看：
- 使用示例：`auth_service_example.py`
- 单元测试：`tests/test_auth_service.py`（需要创建）
- 集成测试：`tests/test_middleware_integration.py`（需要创建）

## 总结

| 方面 | 原架构 | 新架构 |
|------|--------|--------|
| 代码组织 | 单个大文件 | 多个小文件，职责清晰 |
| 可测试性 | 低（需要模拟整个请求） | 高（可独立测试服务类） |
| 可复用性 | 低（绑定在中间件） | 高（服务类可在多处使用） |
| 可维护性 | 低（逻辑耦合） | 高（职责分离） |
| 可扩展性 | 低（需要修改中间件） | 高（继承和组合服务类） |
| 性能 | 基准 | 基本相同（开销可忽略） |

**推荐做法**：
1. 在新项目中直接使用新架构
2. 在现有项目中逐步迁移（可以共存）
3. 编写充分的测试保证迁移质量
