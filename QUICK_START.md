# 快速开始指南

## 🎯 核心理念

**在接口中，你只需要做一件事：**

```python
user = request.state.user  # ✅ 就这么简单！
```

所有认证逻辑（LDAP查询、用户创建、信息刷新等）都在 service 中自动完成。

---

## 📦 三步开始使用

### 第 1 步：配置中间件（只需配置一次）

```python
from fastapi import FastAPI
from http_middleware import HttpMiddleware

app = FastAPI()

# 配置认证参数
class Config:
    ldap_server_url = "ldap://your-ldap-server.com"
    ldap_base_dn = "DC=example,DC=com"
    ldap_timeout = 30

# 添加中间件
app.add_middleware(HttpMiddleware, config=Config())
```

### 第 2 步：实现数据库会话获取

在 `http_middleware.py` 中找到 `_get_db_session()` 方法，按照你的项目实现：

```python
async def _get_db_session(self):
    """获取数据库会话"""
    # 方案1: 使用全局对象 g
    from your_project.globals import g
    return g.user_db_async_session()
    
    # 方案2: 使用 SQLAlchemy
    # from your_project.database import async_session_maker
    # async with async_session_maker() as session:
    #     yield session
```

### 第 3 步：在接口中使用

```python
@app.get("/api/user/info")
async def get_user_info(request: Request):
    """获取用户信息"""
    # 就这么简单！
    user = request.state.user
    
    if not user:
        return {"error": "未认证"}
    
    return {
        "idsid": user.idsid,
        "username": user.username,
        "email": user.email,
    }
```

**完成！** 🎉

---

## 💡 实际使用示例

### 示例 1: 获取用户信息

```python
@app.get("/api/user/profile")
async def get_profile(request: Request):
    user = request.state.user
    
    if not user:
        return JSONResponse(status_code=401, content={"error": "未认证"})
    
    return {
        "idsid": user.idsid,
        "name": user.name,
        "email": user.email,
    }
```

### 示例 2: 创建数据

```python
@app.post("/api/article/create")
async def create_article(request: Request):
    user = request.state.user
    
    if not user:
        return JSONResponse(status_code=401, content={"error": "未认证"})
    
    body = await request.json()
    
    # 创建文章，关联到当前用户
    article = {
        "title": body["title"],
        "content": body["content"],
        "author": user.idsid,  # 使用认证用户
        "created_at": datetime.now()
    }
    
    return {"message": "创建成功", "article": article}
```

### 示例 3: 权限检查

```python
@app.get("/api/admin/settings")
async def admin_settings(request: Request):
    user = request.state.user
    
    # 检查认证
    if not user:
        return JSONResponse(status_code=401, content={"error": "未认证"})
    
    # 检查权限
    if not user.is_admin:
        return JSONResponse(status_code=403, content={"error": "需要管理员权限"})
    
    return {"settings": {...}}
```

### 示例 4: 可选认证（公开接口）

```python
@app.get("/api/articles")
async def list_articles(request: Request):
    user = request.state.user
    
    # 公开接口，但可以根据用户状态返回不同内容
    if user:
        # 已认证用户可以看到更多内容
        return {"articles": [...], "recommended": [...]}
    else:
        # 游客只能看到基础内容
        return {"articles": [...]}
```

---

## 🛠️ 辅助工具函数（可选）

如果你想让代码更简洁，可以使用这些辅助函数：

```python
from fastapi import HTTPException

def require_user(request: Request):
    """要求用户必须认证"""
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="未认证")
    return user

def require_admin(request: Request):
    """要求管理员权限"""
    user = require_user(request)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user
```

**使用示例：**

```python
@app.get("/api/user/info")
async def get_user_info(request: Request):
    user = require_user(request)  # 自动检查认证
    return {"idsid": user.idsid}

@app.get("/api/admin/dashboard")
async def admin_dashboard(request: Request):
    user = require_admin(request)  # 自动检查管理员
    return {"message": "管理员控制台"}
```

---

## 📋 完整的 request.state 属性

中间件会自动设置以下属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `request.state.user` | User 或 None | 当前认证用户 |
| `request.state.idsid` | str 或 None | 用户 ID |
| `request.state.request_id` | str | 请求唯一标识 |
| `request.state.start_time` | datetime | 请求开始时间 |
| `request.state.server_name` | str | 服务器主机名 |

**使用示例：**

```python
@app.get("/api/debug/info")
async def debug_info(request: Request):
    return {
        "user": request.state.user.idsid if request.state.user else None,
        "request_id": request.state.request_id,
        "server": request.state.server_name,
        "start_time": str(request.state.start_time),
    }
```

---

## 🔍 认证流程说明

**你不需要关心这些细节，但如果你好奇：**

1. **中间件自动执行** → 每个请求都会经过认证中间件
2. **获取 REMOTE_USER** → 从 IIS 或请求头获取
3. **查找或创建用户** → 在数据库中查找，不存在则创建
4. **刷新过期信息** → 如果用户信息超过30天，从 LDAP 刷新
5. **设置到 request.state** → 将用户对象设置到 `request.state.user`
6. **接口直接使用** → 你的接口代码只需读取 `request.state.user`

✅ **所有这些都是自动的！**

---

## 🚨 常见错误处理

### 错误 1: 未实现数据库会话

```
NotImplementedError: 请在 http_middleware.py 的 _get_db_session() 方法中实现数据库会话获取逻辑
```

**解决方法：** 在 `http_middleware.py` 中实现 `_get_db_session()` 方法。

### 错误 2: 导入错误

```
ImportError: cannot import name 'UserBackend' from 'user_backend'
```

**解决方法：** 更新 `auth_service.py` 中的导入路径：

```python
# 改成你项目中实际的导入路径
from your_project.auth import UserBackend
from your_project.services import UserService
```

### 错误 3: user 为 None

**原因：**
- 没有 REMOTE_USER（非 IIS 环境）
- 认证失败
- 用户创建失败

**处理方法：**

```python
@app.get("/api/data")
async def get_data(request: Request):
    user = request.state.user
    
    if not user:
        # 返回友好的错误信息
        return JSONResponse(
            status_code=401,
            content={
                "error": "未认证",
                "message": "请通过 IIS 认证后访问此接口"
            }
        )
    
    return {"data": "..."}
```

---

## 📝 代码对比

### ❌ 旧方式（复杂）

```python
@app.get("/api/user")
async def get_user(request: Request, db: Session = Depends(get_db)):
    # 手动获取 REMOTE_USER
    remote_user = request.headers.get("REMOTE_USER")
    if not remote_user:
        return {"error": "未认证"}
    
    # 手动解析 idsid
    idsid = parse_idsid(remote_user)
    
    # 手动查询数据库
    user = db.query(User).filter(User.idsid == idsid).first()
    if not user:
        # 手动创建用户
        user = create_user_from_ldap(idsid)
        db.add(user)
        db.commit()
    
    # 手动检查是否需要刷新
    if user.needs_refresh():
        user = refresh_from_ldap(user)
        db.commit()
    
    return {"user": user.idsid}
```

### ✅ 新方式（简洁）

```python
@app.get("/api/user")
async def get_user(request: Request):
    user = request.state.user  # 就这么简单！
    
    if not user:
        return {"error": "未认证"}
    
    return {"user": user.idsid}
```

**代码减少了 80%！** 🎉

---

## 🎯 最佳实践

### 1. 统一错误处理

```python
def get_authenticated_user(request: Request):
    """统一的用户获取函数"""
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="未认证")
    return user

# 在所有接口中使用
@app.get("/api/profile")
async def get_profile(request: Request):
    user = get_authenticated_user(request)
    return {"profile": user.to_dict()}
```

### 2. 使用依赖注入

```python
from fastapi import Depends

async def get_current_user(request: Request):
    """依赖：获取当前用户"""
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401)
    return user

# 使用依赖
@app.get("/api/profile")
async def get_profile(user = Depends(get_current_user)):
    return {"profile": user.to_dict()}
```

### 3. 权限装饰器

```python
from functools import wraps

def require_permission(permission: str):
    """权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user = request.state.user
            if not user:
                raise HTTPException(status_code=401)
            if not user.has_permission(permission):
                raise HTTPException(status_code=403)
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# 使用装饰器
@app.post("/api/admin/delete")
@require_permission("admin.delete")
async def delete_data(request: Request):
    return {"message": "删除成功"}
```

---

## 📚 相关文档

- **完整示例**：`simple_usage_example.py` - 包含多个实际使用场景
- **迁移指南**：`AUTH_MIGRATION_GUIDE.md` - 从旧代码迁移的详细步骤
- **API 参考**：`AUTH_SERVICE_README.md` - 服务类的完整 API 文档
- **单元测试**：`test_auth_service.py` - 如何测试认证逻辑

---

## ✅ 检查清单

在开始使用之前，确保：

- [ ] 已添加 `auth_service.py` 到项目
- [ ] 已添加 `http_middleware.py` 到项目
- [ ] 已在 `app.add_middleware()` 中配置中间件
- [ ] 已实现 `_get_db_session()` 方法
- [ ] 已更新 `auth_service.py` 中的导入路径（UserBackend, UserService）
- [ ] 已在测试环境验证

---

## 🎉 开始使用

```bash
# 1. 复制文件到你的项目
cp auth_service.py your_project/
cp http_middleware.py your_project/

# 2. 查看示例
python simple_usage_example.py

# 3. 在浏览器中访问
http://localhost:8000/docs
```

---

## 💡 核心要点

记住这三点：

1. **添加中间件一次** → `app.add_middleware(HttpMiddleware, config=Config())`
2. **接口中获取用户** → `user = request.state.user`
3. **检查认证状态** → `if not user: return 401`

就是这么简单！🚀

---

有任何问题，查看 `simple_usage_example.py` 中的实际代码示例。
