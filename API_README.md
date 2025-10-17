# FastAPI 模型服务

基于FastAPI构建的模型推理服务框架，支持单个和批量推理。

## 项目结构

```
.
├── app/                        # 应用主目录
│   ├── __init__.py
│   ├── main.py                 # FastAPI应用入口
│   ├── api/                    # API路由
│   │   ├── __init__.py
│   │   └── v1/                 # API版本1
│   │       ├── __init__.py
│   │       └── endpoints/      # API端点
│   │           ├── __init__.py
│   │           ├── health.py   # 健康检查
│   │           └── model.py    # 模型推理
│   ├── core/                   # 核心配置
│   │   ├── __init__.py
│   │   ├── config.py           # 配置管理
│   │   └── logging.py          # 日志配置
│   ├── models/                 # 模型定义
│   │   └── __init__.py
│   ├── schemas/                # Pydantic数据模型
│   │   ├── __init__.py
│   │   ├── request.py          # 请求模型
│   │   └── response.py         # 响应模型
│   └── services/               # 业务逻辑
│       ├── __init__.py
│       └── model_service.py    # 模型服务
├── .env.example                # 环境变量示例
├── requirements.txt            # Python依赖
└── run.py                      # 启动脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

主要配置项：
- `MODEL_PATH`: 模型文件路径
- `MODEL_DEVICE`: 使用设备 (cuda/cpu)
- `PORT`: 服务端口
- `BATCH_SIZE`: 批处理大小

### 3. 启动服务

```bash
# 方式1: 使用启动脚本
python run.py

# 方式2: 直接使用uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后，访问以下地址：
- API文档 (Swagger): http://localhost:8000/api/v1/docs
- API文档 (ReDoc): http://localhost:8000/api/v1/redoc

## API端点

### 健康检查

**GET** `/api/v1/health`

检查服务状态和模型是否已加载。

响应示例：
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "version": "0.1.0",
  "model_loaded": true
}
```

### 加载模型

**POST** `/api/v1/model/load`

加载模型到内存。

参数：
- `model_path` (可选): 模型路径

响应示例：
```json
{
  "success": true,
  "message": "模型加载成功",
  "model_loaded": true
}
```

### 单个样本推理

**POST** `/api/v1/model/predict`

对单个样本进行推理。

请求体：
```json
{
  "data": "your_input_data",
  "batch_size": 1
}
```

响应示例：
```json
{
  "success": true,
  "result": {
    "prediction": "result",
    "confidence": 0.95
  },
  "message": "推理成功",
  "processing_time": 0.123
}
```

### 批量推理

**POST** `/api/v1/model/batch_predict`

对多个样本进行批量推理。

请求体：
```json
{
  "data_list": ["data1", "data2", "data3"],
  "batch_size": 32
}
```

响应示例：
```json
{
  "success": true,
  "results": [
    {"prediction": "result1"},
    {"prediction": "result2"},
    {"prediction": "result3"}
  ],
  "message": "批量推理成功",
  "processing_time": 0.456,
  "total_count": 3
}
```

### 卸载模型

**POST** `/api/v1/model/unload`

卸载模型并释放资源。

响应示例：
```json
{
  "success": true,
  "message": "模型已卸载",
  "model_loaded": false
}
```

## 自定义模型服务

要集成你自己的模型，需要修改 `app/services/model_service.py`：

1. 在 `load_model()` 方法中添加模型加载逻辑
2. 在 `predict()` 方法中实现单样本推理
3. 在 `batch_predict()` 方法中实现批量推理

示例：

```python
def load_model(self, model_path: str = None):
    # 加载你的模型
    self.model = YourModel()
    self.model.load_state_dict(torch.load(model_path))
    self.model.to(self.device)
    self.model.eval()
    self.is_loaded = True

def predict(self, data: Any) -> Any:
    # 实现推理逻辑
    with torch.no_grad():
        result = self.model(data)
    return result
```

## 开发指南

### 添加新的API端点

1. 在 `app/api/v1/endpoints/` 创建新的端点文件
2. 在 `app/api/v1/__init__.py` 中注册路由
3. 在 `app/schemas/` 中定义请求和响应模型

### 修改配置

所有配置都在 `app/core/config.py` 中的 `Settings` 类中定义。

### 日志

使用 `app/core/logging.py` 中的 logger：

```python
from app.core.logging import logger

logger.info("信息日志")
logger.error("错误日志")
```

## 部署

### Docker部署

创建 `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py"]
```

构建和运行：

```bash
docker build -t model-service .
docker run -p 8000:8000 model-service
```

### 生产环境

使用Gunicorn + Uvicorn workers：

```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## 许可

MIT License
