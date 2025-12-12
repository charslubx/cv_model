# 认证服务 - 纯Service实现

> **Python 3.11+** - 使用现代 Python 特性（`type | None` 语法，timezone-aware datetime）

## 🎯 核心理念

**不使用中间件，所有认证逻辑封装在Service中，在接口中通过依赖注入或直接调用使用。**

最终效果：在接口中通过 `request.state.user` 获取认证用户。

---

## 📦 文件说明

| 文件 | 说明 |
|------|------|
| `auth_service.py` | **认证服务类** - 封装所有认证逻辑 |
| `使用示例.py` | **完整示例** - 3种使用方式 |
| `PYTHON_311_特性.md` | Python 3.11 特性说明 |
| `README.md` | 本文件 - 快速开始 |

## ⚙️ 环境要求

- **Python 3.11+** （推荐）或 **Python 3.10+** （最低）
- FastAPI
- Pydantic v2

```bash
# 检查 Python 版本
python --version  # 需要 >= 3.10

# 安装依赖
pip install fastapi uvicorn pydantic
```

---

## 🚀 快速开始

### 第 1 步：创建认证服务

```python
from auth_service import AuthService

# 创建服务实例
auth_service = AuthService(
    ldap_server_url="ldap://your-server.com",
    ldap_base_dn="DC=example,DC=com",
    ldap_timeout=30,
    db_session_factory=get_db_session  # 可选
)
```

### 第 2 步：在接口中使用

#### 方式 1: 使用依赖注入（推荐）

```python
from fastapi import FastAPI, Request, Depends

app = FastAPI()

# 定义认证依赖
async def authenticate_request(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db_session = Depends(get_db_session)
):
    await auth_service.authenticate(request, db_session)

# 在接口中使用
@app.get("/api/user", dependencies=[Depends(authenticate_request)])
async def get_user(request: Request):
    user = request.state.user  # ✅ 自动设置
    
    if not user:
        return {"error": "未认证"}
    
    return {"user": user.idsid}
```

#### 方式 2: 使用辅助函数（更简洁）

```python
# 定义辅助函数
async def require_user(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db_session = Depends(get_db_session)
):
    await auth_service.authenticate(request, db_session)
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="未认证")
    return user

# 在接口中使用
@app.get("/api/user")
async def get_user(user = Depends(require_user)):
    return {"user": user.idsid}  # ✅ 直接获取 user
```

#### 方式 3: 直接调用（不使用依赖注入）

```python
auth_service = AuthService(...)

@app.get("/api/user")
async def get_user(request: Request):
    # 手动调用认证
    async with get_db_session() as db:
        await auth_service.authenticate(request, db)
    
    user = request.state.user  # ✅ 自动设置
    
    if not user:
        return {"error": "未认证"}
    
    return {"user": user.idsid}
```

---

## 📋 核心方法

### `AuthService.authenticate(request, db_session)`

**主方法** - 执行完整认证流程

```python
auth_service = AuthService(...)
await auth_service.authenticate(request, db_session)

# 认证完成后：
# ✅ request.state.user 被设置
# ✅ request.state.idsid 被设置
# ✅ request.state.server_name 被设置
```

**自动执行：**
1. 从IIS获取REMOTE_USER
2. 认证用户或创建新用户
3. 刷新过期用户信息（>30天）
4. 更新登录时间
5. 设置 `request.state.user`

---

## 🎯 对比原中间件

### 原中间件

```python
class HttpMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.auth_backend = None
        self.user_service = None
    
    async def dispatch(self, request, call_next):
        # ... 大量认证逻辑 ...
        await self.process_request(request)
        response = await call_next(request)
        return response
    
    async def process_request(self, request):
        # ... 80+ 行认证逻辑 ...
        request.state.user = user

# 使用
app.add_middleware(HttpMiddleware)
```

### 新Service

```python
class AuthService:
    async def authenticate(self, request, db_session):
        # 所有认证逻辑封装在这里
        user = await self._do_authenticate(request, db_session)
        request.state.user = user
        return user

# 使用（依赖注入）
@app.get("/api/user", dependencies=[Depends(authenticate_request)])
async def get_user(request: Request):
    user = request.state.user
    return {"user": user.idsid}

# 或者（直接调用）
@app.get("/api/user")
async def get_user(request: Request):
    await auth_service.authenticate(request, db)
    user = request.state.user
    return {"user": user.idsid}
```

---

## ✅ 优势

### 1. **职责清晰**
- ❌ 中间件：自动拦截所有请求，无法选择
- ✅ Service：只在需要时调用，灵活控制

### 2. **易于测试**
```python
# 可以独立测试
auth_service = AuthService(...)
user = await auth_service.authenticate(mock_request, mock_db)
assert user.idsid == "expected"
```

### 3. **易于复用**
```python
# 在路由中使用
await auth_service.authenticate(request, db)

# 在后台任务中使用
user = await auth_service.refresh_user_if_needed(user, db)

# 在任何地方使用
remote_user = AuthService.get_remote_user(request)
```

### 4. **灵活性高**
```python
# 可以选择性认证
if request.url.path.startswith("/api/protected/"):
    await auth_service.authenticate(request, db)

# 可以自定义逻辑
user = await auth_service.authenticate(request, db)
if user and not user.is_active:
    raise HTTPException(403, "账号已禁用")
```

---

## 🔧 必需配置

### 1. 实现数据库会话

```python
async def get_db_session():
    """获取数据库会话"""
    # 方案1: 使用全局对象
    from your_project.globals import g
    async with g.user_db_async_session() as session:
        yield session
    
    # 方案2: 使用 SQLAlchemy
    # from your_project.database import async_session_maker
    # async with async_session_maker() as session:
    #     yield session
```

### 2. 更新导入路径

在 `auth_service.py` 中：

```python
# 修改为你的实际导入路径
from your_project.auth import UserBackend
from your_project.services import UserService
```

---

## 📖 完整示例

查看 `使用示例.py` 文件，包含：

- ✅ 方式1: 使用依赖注入（推荐）
- ✅ 方式2: 使用辅助函数（更简洁）
- ✅ 方式3: 直接使用全局实例
- ✅ 公开接口（可选认证）
- ✅ 管理员权限检查

**运行示例：**

```bash
python 使用示例.py
# 访问 http://localhost:8000/docs
```

---

## 🎓 推荐使用方式

### 方式 1: 依赖注入 + 辅助函数（最推荐）

```python
# 1. 定义辅助函数
async def require_user(request: Request, ...):
    await auth_service.authenticate(request, db)
    user = request.state.user
    if not user:
        raise HTTPException(401)
    return user

# 2. 在接口中使用
@app.get("/api/user")
async def get_user(user = Depends(require_user)):
    return {"user": user.idsid}  # ✅ 超简洁！
```

**优点：**
- ✅ 代码最简洁
- ✅ 自动错误处理
- ✅ 直接获取 user 对象
- ✅ 易于测试

---

## 📝 request.state 可用属性

认证成功后，以下属性会被自动设置：

| 属性 | 类型 | 说明 |
|------|------|------|
| `request.state.user` | User 或 None | **当前认证用户** |
| `request.state.idsid` | str 或 None | 用户ID |
| `request.state.server_name` | str | 服务器主机名 |

**可选设置（使用 `set_request_context`）：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `request.state.request_id` | str | 请求唯一标识 |
| `request.state.start_time` | datetime | 请求开始时间 |

```python
# 在接口中使用
AuthService.set_request_context(request)
print(request.state.request_id)
```

---

## 🚨 常见问题

### Q1: 为什么不用中间件？

**A:** 中间件会拦截所有请求，缺乏灵活性。Service方式可以：
- 选择性认证（只在需要的接口认证）
- 更容易测试（不需要模拟整个请求流程）
- 更容易复用（可以在多处使用）

### Q2: request.state.user 何时被设置？

**A:** 调用 `auth_service.authenticate(request, db_session)` 后自动设置。

### Q3: 如何处理公开接口？

**A:** 不调用认证，或调用后不检查 user 是否为 None：

```python
@app.get("/api/public")
async def public_info(request: Request):
    # 尝试认证（不强制）
    await auth_service.authenticate(request, db)
    
    user = request.state.user
    if user:
        return {"message": "已认证", "user": user.idsid}
    else:
        return {"message": "游客访问"}
```

### Q4: 如何扩展认证逻辑？

**A:** 继承 AuthService：

```python
class CustomAuthService(AuthService):
    async def authenticate(self, request, db_session):
        user = await super().authenticate(request, db_session)
        
        # 添加自定义逻辑
        if user and not user.is_active:
            request.state.user = None
            return None
        
        return user
```

---

## ✅ 检查清单

部署前检查：

- [ ] 已复制 `auth_service.py` 到项目
- [ ] 已实现 `get_db_session()` 数据库会话获取
- [ ] 已更新 `auth_service.py` 中的导入路径（UserBackend, UserService）
- [ ] 已创建认证服务实例
- [ ] 已定义认证依赖或辅助函数
- [ ] 已在接口中使用认证
- [ ] 已测试认证流程

---

## 🎉 总结

### 核心改变

| 方面 | 原中间件 | 新Service |
|------|----------|-----------|
| 实现方式 | 中间件拦截 | Service + 依赖注入 |
| 使用位置 | 自动拦截所有请求 | 只在需要的接口调用 |
| 灵活性 | 低（无法选择） | 高（完全控制） |
| 可测试性 | 低（需要模拟请求） | 高（独立测试） |
| 可复用性 | 低（绑定中间件） | 高（任意位置使用） |

### 接口代码

**完全相同！**

```python
# 接口代码不变
user = request.state.user

if not user:
    return {"error": "未认证"}

return {"user": user.idsid}
```

### 最佳实践

```python
# 1. 创建全局服务实例
auth_service = AuthService(...)

# 2. 定义辅助函数
async def require_user(request: Request, ...):
    await auth_service.authenticate(request, db)
    user = request.state.user
    if not user:
        raise HTTPException(401)
    return user

# 3. 在接口中使用
@app.get("/api/user")
async def get_user(user = Depends(require_user)):
    return {"user": user.idsid}  # ✅ 完美！
```

---

**开始使用：**

1. 📖 查看 `使用示例.py`
2. ▶️ 运行 `python 使用示例.py`
3. 🚀 在你的项目中使用

祝你使用愉快！🎉
