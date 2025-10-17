# 服装属性多标签分类服务

基于GAT+GCN的服装属性多标签分类模型推理服务，使用FastAPI构建。

## 目录

- [功能特性](#功能特性)
- [关于多标签分类](#关于多标签分类)
- [快速开始](#快速开始)
- [API接口](#api接口)
- [配置说明](#配置说明)
- [阈值调节指南](#阈值调节指南)
- [部署方案](#部署方案)
- [性能优化](#性能优化)
- [故障排查](#故障排查)

---

## 功能特性

- ✅ 单张图片多标签分类
- ✅ 批量图片分类
- ✅ 支持26个服装属性的多标签分类
- ✅ 可调节分类阈值（默认0.5）
- ✅ 返回分类结果和置信度
- ✅ RESTful API接口
- ✅ 异步处理
- ✅ CORS支持
- ✅ Docker部署支持

---

## 关于多标签分类

### 什么是多标签分类？

**多标签分类**（Multi-Label Classification）是指一个样本可以同时属于多个类别的分类任务。

**与多分类的区别**：
- **多分类（Multi-Class）**：每个样本只能属于一个类别
  - 例如：图片分类为"猫"或"狗"或"鸟"（只能选一个）
  - 输出：使用softmax，所有类别概率和为1
  
- **多标签分类（Multi-Label）**：每个样本可以同时属于多个类别
  - 例如：服装同时具有"黑色"、"纯色"、"棉质"等多个属性
  - 输出：使用sigmoid，每个类别独立预测

### 本服务的任务

- **输入**：服装图片
- **输出**：26个属性的分类结果（每个属性0或1）和置信度（0-1之间的概率值）
- **阈值**：默认使用0.5作为分类阈值，置信度≥0.5的属性被分类为1（正类），否则为0（负类）

### 26个属性列表

```
颜色：black, blue, brown, cyan, gray, green, pink, purple, red, white, yellow
图案：pattern_solid, pattern_stripe, pattern_floral, pattern_graphics, pattern_plaid, pattern_spot
材质：cotton, denim, knitted, leather
其他：collar, necktie, scarf, skin_exposure, many_colors
```

### 分类流程

```
输入图片 → CNN特征提取(ResNet50) → GAT图注意力 → GCN图卷积 
→ 输出logits → Sigmoid激活 → 概率值 → 阈值分类 → 0/1结果
```

---

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.4.1+
- CUDA 11.8+ (可选，用于GPU加速)

### 安装依赖

```bash
# 1. 安装主项目依赖
pip install -r ../requirements.txt

# 2. 安装服务依赖
pip install -r requirements_service.txt
```

### 启动服务

**方式1: 直接运行**
```bash
cd service
python app.py
```

**方式2: 使用启动脚本**
```bash
bash service/start_service.sh
```

**方式3: 使用uvicorn**
```bash
cd service
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 测试分类
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

---

## API接口

### 1. 健康检查

```bash
GET /health
```

**响应**：
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. 单张图片分类

```bash
POST /predict?threshold=0.5
```

**参数**：
- `file`: 图片文件（必填）
- `threshold`: 分类阈值，0-1之间（可选，默认0.5）

**示例**：
```bash
# 使用默认阈值
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"

# 自定义阈值
curl -X POST "http://localhost:8000/predict?threshold=0.6" \
  -F "file=@image.jpg"
```

**响应**：
```json
{
  "success": true,
  "filename": "image.jpg",
  "predictions": {
    "attributes": {
      "black": 0.95,
      "pattern_solid": 0.89,
      "cotton": 0.78,
      "blue": 0.12,
      ...
    },
    "classifications": {
      "black": 1,
      "pattern_solid": 1,
      "cotton": 1,
      "blue": 0,
      ...
    },
    "positive_attributes": [
      {"attribute": "black", "confidence": 0.95},
      {"attribute": "pattern_solid", "confidence": 0.89},
      {"attribute": "cotton", "confidence": 0.78}
    ],
    "top_k_attributes": [
      {"attribute": "black", "confidence": 0.95, "classification": 1},
      ...
    ],
    "statistics": {
      "total_attributes": 26,
      "positive_count": 3,
      "threshold": 0.5
    }
  }
}
```

### 3. 批量图片分类

```bash
POST /predict_batch?threshold=0.5
```

**示例**：
```bash
curl -X POST "http://localhost:8000/predict_batch?threshold=0.5" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### 4. 获取属性列表

```bash
GET /attributes
```

**响应**：
```json
{
  "attributes": ["black", "blue", "brown", ...],
  "num_classes": 26
}
```

### Python客户端示例

```python
import requests

# 单张图片分类
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        params={'threshold': 0.5}
    )

result = response.json()

# 查看预测为正的属性
for attr in result['predictions']['positive_attributes']:
    print(f"{attr['attribute']}: {attr['confidence']:.2%}")
```

---

## 配置说明

### 环境变量

创建`.env`文件配置服务：

```bash
# 服务配置
HOST=0.0.0.0              # 监听地址
PORT=8000                 # 监听端口
WORKERS=1                 # Worker数量

# 模型配置
MODEL_PATH=../checkpoints/best_model.pth  # 模型权重路径
DEVICE=cuda               # 推理设备 (cuda/cpu)
NUM_CLASSES=26            # 属性类别数量

# 推理配置
MAX_BATCH_SIZE=32         # 最大批量大小
IMG_SIZE=224              # 输入图片大小
```

### 配置参数说明

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| HOST | 0.0.0.0 | 服务监听地址 |
| PORT | 8000 | 服务端口 |
| DEVICE | cuda | 推理设备（cuda/cpu） |
| MODEL_PATH | ../checkpoints/best_model.pth | 模型权重路径 |
| NUM_CLASSES | 26 | 属性类别数量 |
| MAX_BATCH_SIZE | 32 | 最大批量大小 |

---

## 阈值调节指南

### 阈值的作用

阈值（threshold）用于将概率值转换为二分类结果：
- `概率 >= 阈值` → 分类为1（正类，该属性存在）
- `概率 < 阈值` → 分类为0（负类，该属性不存在）

### 选择合适的阈值

| 阈值 | 效果 | 召回率 | 精确率 | 适用场景 |
|-----|------|-------|-------|---------|
| 0.3 | 更宽松，识别更多属性 | 高 | 低 | 电商推荐、标签生成 |
| 0.5 | 平衡（默认） | 中 | 中 | 一般应用场景 |
| 0.7 | 更严格，只保留高置信度 | 低 | 高 | 精准搜索、质量控制 |

### 示例对比

假设模型输出某属性的置信度为0.55：

```python
# 阈值=0.3: 0.55 >= 0.3 → 分类为1（有该属性）
# 阈值=0.5: 0.55 >= 0.5 → 分类为1（有该属性）  
# 阈值=0.7: 0.55 < 0.7  → 分类为0（无该属性）
```

### 根据场景选择

```python
# 电商推荐场景：希望多召回一些属性
result = predict(image, threshold=0.4)

# 精准搜索场景：希望结果准确
result = predict(image, threshold=0.6)

# 一般场景：使用默认值
result = predict(image, threshold=0.5)
```

---

## 部署方案

### Docker部署

**1. 构建镜像**
```bash
docker build -t fashion-attr-service:latest -f service/Dockerfile .
```

**2. 运行容器（CPU模式）**
```bash
docker run -d \
  --name fashion-service \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fashion-attr-service:latest
```

**3. 运行容器（GPU模式）**
```bash
docker run -d \
  --name fashion-service \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fashion-attr-service:latest
```

**4. 查看日志**
```bash
docker logs -f fashion-service
```

### Systemd服务

创建服务文件`/etc/systemd/system/fashion-service.service`：

```ini
[Unit]
Description=Fashion Attribute Classification Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/project/service
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable fashion-service
sudo systemctl start fashion-service
```

### Nginx反向代理

创建nginx配置：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 增加超时时间
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
```

---

## 性能优化

### 1. GPU加速

```bash
# 设置使用GPU
export DEVICE=cuda
python app.py
```

### 2. 批量处理

使用`/predict_batch`接口进行批量预测，性能更优：

```python
files = [('files', open(img, 'rb')) for img in image_list]
response = requests.post(
    'http://localhost:8000/predict_batch',
    files=files
)
```

### 3. 多Worker模式

```bash
# 使用多个worker
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. 使用Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app:app
```

### 5. 负载均衡

使用Nginx配置多个后端实例：

```nginx
upstream fashion_backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://fashion_backend;
    }
}
```

---

## 故障排查

### 1. 模型加载失败

**问题**：`FileNotFoundError: Model file not found`

**解决方案**：
```bash
# 检查模型路径
ls -la checkpoints/best_model.pth

# 如果没有模型文件，服务会使用预训练权重初始化
# 建议先训练模型并保存权重
```

### 2. CUDA错误

**问题**：`RuntimeError: CUDA out of memory`

**解决方案**：
```bash
# 方案1: 使用CPU模式
export DEVICE=cpu

# 方案2: 减少批量大小
export MAX_BATCH_SIZE=16

# 方案3: 使用更少的worker
uvicorn app:app --workers 1
```

### 3. 端口被占用

**问题**：`OSError: [Errno 98] Address already in use`

**解决方案**：
```bash
# 查找占用端口的进程
lsof -i :8000

# 杀死进程
kill -9 <PID>

# 或更换端口
export PORT=8001
```

### 4. 依赖安装失败

**问题**：`ModuleNotFoundError: No module named 'torch'`

**解决方案**：
```bash
# 重新安装依赖
pip install -r requirements.txt
pip install -r service/requirements_service.txt
```

### 5. 所有属性都被分类为0

**可能原因**：
- 阈值设置过高
- 图片质量问题
- 模型未正确加载

**解决方案**：
```bash
# 降低阈值
curl -X POST "http://localhost:8000/predict?threshold=0.3" \
  -F "file=@image.jpg"

# 查看置信度分布
# 检查attributes字段中的概率值
```

### 6. 内存不足

**解决方案**：
```bash
# 减少批量大小
export MAX_BATCH_SIZE=8

# 使用CPU模式
export DEVICE=cpu

# 减少worker数量
uvicorn app:app --workers 1
```

---

## 测试

### 运行测试脚本

```bash
python test_service.py
```

### 手动测试

```bash
# 1. 健康检查
curl http://localhost:8000/health

# 2. 测试分类
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg" | jq

# 3. 测试不同阈值
for t in 0.3 0.5 0.7; do
  echo "Threshold: $t"
  curl -X POST "http://localhost:8000/predict?threshold=$t" \
    -F "file=@test_image.jpg" | jq '.predictions.statistics'
done
```

---

## 开发指南

### 修改属性列表

编辑`model_loader.py`中的`_get_attribute_names`方法：

```python
def _get_attribute_names(self) -> List[str]:
    return [
        "black", "blue", "brown", ...
        # 在这里添加或修改属性名称
    ]
```

### 自定义模型

修改`model_loader.py`中的`ModelInference.__init__`方法：

```python
self.model = FullModel(
    num_classes=26,        # 修改类别数量
    cnn_type='resnet50',   # 修改backbone
    gat_dims=[1024, 512],  # 修改GAT维度
    ...
)
```

### 添加新的API接口

在`app.py`中添加新的路由：

```python
@app.post("/your_new_endpoint")
async def your_new_function(file: UploadFile = File(...)):
    # 实现你的逻辑
    pass
```

---

## 评估指标

对于多标签分类任务，常用评估指标：

### 1. 精确率（Precision）
```
Precision = TP / (TP + FP)
```
预测为正的样本中，真正为正的比例

### 2. 召回率（Recall）
```
Recall = TP / (TP + FN)
```
所有真正为正的样本中，被正确预测为正的比例

### 3. F1分数（F1-Score）
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
精确率和召回率的调和平均

### 4. Hamming Loss
```
Hamming Loss = (FP + FN) / (Total Labels)
```
预测错误的标签比例

---

## 许可证

MIT License

---

## 相关链接

- 模型定义: `../base_model.py`
- 训练脚本: `../training.py`
- 客户端示例: `client_example.py`
- 测试脚本: `test_service.py`
