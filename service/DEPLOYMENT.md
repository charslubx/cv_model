# 服装属性识别服务部署指南

## 目录
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [生产部署](#生产部署)
- [故障排查](#故障排查)

## 环境准备

### 系统要求
- Ubuntu 18.04+ / CentOS 7+ / macOS 10.14+
- Python 3.8+
- CUDA 11.8+ (可选，用于GPU加速)
- 4GB+ RAM (CPU模式) / 8GB+ RAM (GPU模式)

### 依赖安装

```bash
# 1. 安装主项目依赖
cd /path/to/project
pip install -r requirements.txt

# 2. 安装服务依赖
pip install -r service/requirements_service.txt
```

## 快速开始

### 1. 准备模型权重

如果有训练好的模型：
```bash
mkdir -p checkpoints
cp /path/to/your/model.pth checkpoints/best_model.pth
```

如果没有模型权重，服务会使用预训练权重初始化。

### 2. 启动服务

**方式1: 直接运行**
```bash
cd service
python3 app.py
```

**方式2: 使用启动脚本**
```bash
bash service/start_service.sh
```

**方式3: 使用uvicorn**
```bash
cd service
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 测试预测
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

## 详细配置

### 环境变量配置

创建 `.env` 文件：
```bash
cd service
cp .env.example .env
```

编辑 `.env` 文件：
```bash
# 服务配置
HOST=0.0.0.0
PORT=8000
WORKERS=1

# 模型配置
MODEL_PATH=../checkpoints/best_model.pth
DEVICE=cuda
NUM_CLASSES=26

# 推理配置
MAX_BATCH_SIZE=32
IMG_SIZE=224
```

### 配置说明

| 参数 | 说明 | 默认值 | 可选值 |
|-----|------|-------|--------|
| HOST | 监听地址 | 0.0.0.0 | localhost, 0.0.0.0 |
| PORT | 监听端口 | 8000 | 1024-65535 |
| WORKERS | Worker数量 | 1 | 1-8 |
| DEVICE | 推理设备 | cuda | cuda, cpu |
| MODEL_PATH | 模型路径 | ../checkpoints/best_model.pth | 任意路径 |
| NUM_CLASSES | 类别数量 | 26 | 正整数 |
| MAX_BATCH_SIZE | 最大批量 | 32 | 1-128 |

## 生产部署

### Docker部署

**1. 构建镜像**
```bash
docker build -t fashion-attr-service:latest -f service/Dockerfile .
```

**2. 运行容器**
```bash
# CPU模式
docker run -d \
  --name fashion-service \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fashion-attr-service:latest

# GPU模式
docker run -d \
  --name fashion-service \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fashion-attr-service:latest
```

**3. 查看日志**
```bash
docker logs -f fashion-service
```

### Nginx反向代理

创建nginx配置 `/etc/nginx/sites-available/fashion-service`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 增加超时时间（用于大文件上传）
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
```

启用配置：
```bash
sudo ln -s /etc/nginx/sites-available/fashion-service /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Systemd服务

创建服务文件 `/etc/systemd/system/fashion-service.service`:
```ini
[Unit]
Description=Fashion Attribute Recognition Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/project/service
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable fashion-service
sudo systemctl start fashion-service
sudo systemctl status fashion-service
```

### 性能优化

**1. 多worker模式**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

**2. 使用Gunicorn**
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app:app
```

**3. 负载均衡**
```nginx
upstream fashion_backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
    server 127.0.0.1:8004;
}

server {
    listen 80;
    location / {
        proxy_pass http://fashion_backend;
    }
}
```

## 故障排查

### 常见问题

**1. 模块导入错误**
```
ModuleNotFoundError: No module named 'torch'
```
解决方案：
```bash
pip install -r requirements.txt
pip install -r service/requirements_service.txt
```

**2. CUDA错误**
```
RuntimeError: CUDA out of memory
```
解决方案：
- 设置 `DEVICE=cpu` 使用CPU模式
- 减少 `MAX_BATCH_SIZE`
- 使用更少的worker

**3. 模型加载失败**
```
FileNotFoundError: Model file not found
```
解决方案：
- 检查 `MODEL_PATH` 配置是否正确
- 确认模型文件存在
- 如果没有模型文件，服务会使用预训练权重

**4. 端口被占用**
```
OSError: [Errno 98] Address already in use
```
解决方案：
```bash
# 查找占用端口的进程
lsof -i :8000
# 杀死进程
kill -9 <PID>
# 或更换端口
export PORT=8001
```

### 日志查看

**应用日志**
```bash
# 如果使用systemd
sudo journalctl -u fashion-service -f

# 如果使用Docker
docker logs -f fashion-service

# 直接运行
tail -f logs/service.log
```

### 性能监控

**CPU和内存使用**
```bash
# 查看进程资源使用
top -p $(pgrep -f "uvicorn app:app")

# 或使用htop
htop
```

**GPU使用**
```bash
# 实时监控GPU
nvidia-smi -l 1

# 查看GPU占用
watch -n 1 nvidia-smi
```

### 压力测试

使用Apache Bench:
```bash
# 安装ab
sudo apt-get install apache2-utils

# 测试健康检查接口
ab -n 1000 -c 10 http://localhost:8000/health

# 测试预测接口（需要准备测试图片）
ab -n 100 -c 5 -p test_image.jpg -T image/jpeg http://localhost:8000/predict
```

使用wrk:
```bash
# 安装wrk
sudo apt-get install wrk

# 压力测试
wrk -t4 -c100 -d30s http://localhost:8000/health
```

## 监控和告警

### 添加监控指标

在 `app.py` 中添加Prometheus指标：
```python
from prometheus_client import Counter, Histogram, generate_latest

# 定义指标
REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest())
```

### 配置Prometheus

创建 `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'fashion-service'
    static_configs:
      - targets: ['localhost:8000']
```

### 配置Grafana

导入Dashboard模板，监控：
- 请求QPS
- 响应延迟
- 错误率
- CPU/内存使用
- GPU使用率

## 安全建议

1. **API认证**: 添加JWT或API Key认证
2. **HTTPS**: 使用SSL/TLS加密通信
3. **限流**: 添加请求频率限制
4. **输入验证**: 验证上传文件大小和类型
5. **防火墙**: 只开放必要端口

## 更多资源

- [API文档](README.md)
- [测试指南](test_service.py)
- [客户端示例](client_example.py)
- [Docker配置](Dockerfile)
