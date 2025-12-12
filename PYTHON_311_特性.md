# Python 3.11 特性说明

本项目已针对 Python 3.11+ 优化，使用了以下新特性：

---

## ✅ 已应用的 Python 3.11 特性

### 1. **联合类型简化语法**

#### 之前（Python 3.9-）
```python
from typing import Optional, Union

def get_user() -> Optional[dict]:  # ❌ 旧语法
    pass

def process(value: Union[str, int]):  # ❌ 旧语法
    pass
```

#### 之后（Python 3.10+，3.11 更好）
```python
# 无需导入 Optional, Union

def get_user() -> dict | None:  # ✅ 新语法
    pass

def process(value: str | int):  # ✅ 新语法
    pass
```

**本项目中的应用：**
```python
# auth_service.py
async def authenticate(self, request: Request, db_session=None) -> dict | None:
    """返回用户对象或 None"""
    pass

def get_remote_user(request: Request) -> str | None:
    """返回 REMOTE_USER 或 None"""
    pass
```

---

### 2. **timezone-aware datetime（推荐）**

#### 之前（❌ naive datetime）
```python
from datetime import datetime

now = datetime.utcnow()  # ⚠️ 已弃用，Python 3.12 会警告
```

#### 之后（✅ timezone-aware）
```python
from datetime import datetime, timezone

now = datetime.now(timezone.utc)  # ✅ 推荐
```

**本项目中的应用：**
```python
# auth_service.py
def set_request_context(request: Request, request_id: str | None = None):
    request.state.start_time = datetime.now(timezone.utc)  # ✅

def get_request_duration(request: Request) -> float:
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    return duration
```

---

### 3. **更好的 `datetime.fromisoformat()` 支持**

Python 3.11 增强了 `fromisoformat()` 的解析能力：

```python
from datetime import datetime

# Python 3.11 支持更多格式
dt1 = datetime.fromisoformat("2024-12-12T10:30:00")        # ✅
dt2 = datetime.fromisoformat("2024-12-12T10:30:00Z")       # ✅ 3.11+
dt3 = datetime.fromisoformat("2024-12-12T10:30:00+08:00")  # ✅
dt4 = datetime.fromisoformat("2024-12-12 10:30:00")        # ✅
```

**替代 `dateutil.parser.parse()`：**
```python
# ❌ 需要额外安装 python-dateutil
from dateutil.parser import parse
dt = parse("2024-12-12T10:30:00")

# ✅ Python 3.11 标准库就够了
from datetime import datetime
dt = datetime.fromisoformat("2024-12-12T10:30:00")
```

---

### 4. **性能提升**

Python 3.11 相比 3.10：
- ⚡ 整体性能提升 10-60%
- ⚡ CPython 优化
- ⚡ 更快的函数调用

**对本项目的影响：**
- 认证服务调用更快
- datetime 操作更快
- 整体响应时间减少

---

## 📋 兼容性说明

### 最低要求

- **Python 3.10+** - 支持 `type | None` 语法
- **Python 3.11+** - 推荐，性能更好

### 检查 Python 版本

```bash
python --version
# Python 3.11.x
```

### 如果使用 Python 3.9 或更早版本

需要改回旧语法：

```python
# 替换所有的 type | None
from typing import Optional

# dict | None  →  Optional[dict]
# str | None   →  Optional[str]
```

```bash
# 查找需要修改的地方
grep -r "| None" *.py
```

---

## 🎯 推荐配置

### `pyproject.toml`

```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.0"
pydantic = "^2.5.0"
```

### `requirements.txt`

```txt
# Python 3.11+
fastapi>=0.104.0
pydantic>=2.5.0
uvicorn>=0.24.0
python-dateutil>=2.8.2  # 可选，只在需要解析非标准格式时
```

---

## 🔄 从旧版本迁移

### 如果当前使用 Python 3.9

```bash
# 1. 升级 Python
pyenv install 3.11.7
pyenv local 3.11.7

# 2. 重新安装依赖
pip install -r requirements.txt

# 3. 代码已经兼容 3.11，无需修改
```

### 如果必须使用 Python 3.9

```bash
# 全局替换
sed -i 's/ | None/ = None/g' auth_service.py
sed -i 's/-> dict | None/-> Optional[dict]/g' auth_service.py
sed -i 's/-> str | None/-> Optional[str]/g' auth_service.py

# 添加导入
# 在文件开头添加：from typing import Optional
```

---

## 📚 相关资源

- [Python 3.11 新特性](https://docs.python.org/3.11/whatsnew/3.11.html)
- [PEP 604 - Union Type 语法](https://peps.python.org/pep-0604/)
- [datetime 最佳实践](https://docs.python.org/3/library/datetime.html)

---

## ✅ 总结

本项目充分利用了 Python 3.11 的现代特性：

| 特性 | 优势 |
|------|------|
| `type \| None` 语法 | 更简洁的类型注解 |
| `timezone.utc` | 避免 naive datetime 问题 |
| `fromisoformat()` | 无需 dateutil 包 |
| 性能提升 | 更快的响应速度 |

**推荐使用 Python 3.11+ 获得最佳体验！** ✅
